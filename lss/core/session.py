"""入力受け口・チャンネル検証・完了マーカー(中断・再開)(T310 plan Step4項目3-5)。"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import soundfile as sf
from mutagen import MutagenError
from mutagen.flac import FLAC

from lss.core import audio_identity
from lss.harness.tsv_io import load_tsv, save_tsv
from lss.paths import guard_state_root, repo_root

MAX_CHANNELS = 2  # AC13: 3ch以上(5.1/7.1等)は処理前に弾く
AUDIO_EXTENSIONS = frozenset({
    ".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma",
})

_COMPLETION_HEADER = [
    "input_sha256", "recipe_id", "completed_at", "source_path", "outcome", "zero_reason",
    "validation_result", "outputs", "input_audio",
]
# 入力の復号sampleの同一性で比べる項目。容器（tag・chunk）の違いは入力の違いにしない。
_INPUT_AUDIO_FIELDS = ("schema_version", "method", "pcm_sha256", "frames", "samplerate", "channels", "subtype")
_PUBLICATION_VERSION = 1
ZERO_RESULT_REASONS = frozenset({"no_wanted_target", "no_wanted_score", "music_not_wanted", "no_novel_asset"})
_CHUNK = 1 << 20  # 1MiB
STALE_WORK_SECONDS = 24 * 60 * 60


class CompletionLedgerChangedError(RuntimeError):
    """記録中に完了台帳が変わったため、既存の内容を上書きしない。"""


class UnsupportedChannelLayoutError(ValueError):
    """メッセージに実測チャンネル数を含める(C7・AC13)。

    表示側は「Nch は分離できません」を実測値で書くため、数値を属性でも持つ。
    文字列から数を取り出させない。
    """

    def __init__(self, message: str, *, channels: int = 0):
        super().__init__(message)
        self.channels = channels


class InputChangedError(RuntimeError):
    """処理開始後に入力実体が変化し、来歴との一致を保証できない。"""


def state_dir() -> Path:
    override = os.environ.get("LSS_STATE_DIR")
    if override:
        return guard_state_root(Path(override))
    return guard_state_root(repo_root() / ".state")


def work_dir() -> Path:
    path = state_dir() / "work"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_stale_work(*, older_than_seconds: float = STALE_WORK_SECONDS, now: float | None = None) -> list[Path]:
    """Productの所有receiptを持ち、使用中でないscratchだけを回収する。"""
    from lss.core import scratch

    result = scratch.reconcile(work_dir(), older_than_seconds=older_than_seconds, now=now)
    return [Path(path) for path in result["removed"]]


def completion_ledger_path(state_root: Path | None = None) -> Path:
    return (Path(state_root) if state_root is not None else state_dir()) / "completed_recipes.tsv"


def publication_dir(state_root: Path | None = None) -> Path:
    return (Path(state_root) if state_root is not None else state_dir()) / "publications"


def input_sha256_of(
    path: Path, progress: Callable[[int, int], None] | None = None,
) -> str:
    total_bytes = path.stat().st_size
    completed_bytes = 0

    def report() -> None:
        if progress is not None:
            try:
                progress(completed_bytes, total_bytes)
            except Exception:
                # 表示側の失敗で指紋計算を中断しない。
                pass

    digest = hashlib.sha256()
    report()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
            completed_bytes += len(chunk)
            report()
    return digest.hexdigest()


def ensure_input_unchanged(
    path: Path, expected_sha256: str,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    try:
        actual_sha256 = input_sha256_of(path, progress=progress)
    except OSError as exc:
        raise InputChangedError(f"処理中に入力ファイルを確認できなくなりました: {path}") from exc
    if actual_sha256 != expected_sha256:
        raise InputChangedError(f"処理中に入力ファイルが変更されました: {path}")


def describe_completed_outputs(
    input_sha256: str, recipe_id: str, source_path: Path, *,
    progress: Callable[[Path, int, int], None] | None = None,
    state_root: Path | None = None,
) -> dict[Path, str]:
    """記録された出力ごとの状態。容器metadataだけの変更を、失われた出力と混同しない。"""
    path = completion_ledger_path(state_root)
    if not path.exists():
        return {}
    _, rows = load_tsv(path)
    for row in reversed(rows):
        if (row.get("input_sha256") != input_sha256 or row.get("recipe_id") != recipe_id
                or row.get("source_path") != str(source_path.resolve())):
            continue
        try:
            outputs = json.loads(row.get("outputs") or "[]")
            if not isinstance(outputs, list):
                return {}
            checked = {}
            for item in outputs:
                output = Path(item["path"])
                recorded = item["sha256"]
                try:
                    observed = input_sha256_of(
                        output,
                        progress=(
                            (lambda completed, total, output=output: progress(output, completed, total))
                            if progress is not None else None
                        ),
                    ) if output.is_file() else None
                except OSError:
                    checked[output] = "UNREADABLE"
                    continue
                checked[output] = audio_identity.diagnose(
                    output, recorded_sha256=recorded, recorded_audio=item.get("audio"),
                    observed_sha256=observed)
            return checked
        except (ValueError, KeyError, TypeError):
            return {}
    return {}


def check_completed_outputs(
    input_sha256: str, recipe_id: str, source_path: Path, *,
    progress: Callable[[Path, int, int], None] | None = None,
    state_root: Path | None = None,
) -> dict[Path, bool]:
    """記録された出力ごとの一致状態。旧形式の完了行だけでは完了扱いにしない。"""
    return {output: state in audio_identity.REUSABLE_STATES for output, state in
            describe_completed_outputs(input_sha256, recipe_id, source_path,
                                       progress=progress, state_root=state_root).items()}


def record_missing_audio_identity(
    *, input_sha256: str | None = None, recipe_id: str | None = None,
    source_path: Path | None = None, state_root: Path | None = None, dry_run: bool = False,
) -> dict:
    """検証済みのまま残っている出力にだけ、復号sampleの同一性を後から記録する。

    byte hashが一致する出力は、記録した時のものだと示せる。一致しない出力は差の
    原因を証明できないので記録しない。音声ファイルは読むだけで、記録済みの
    sha256・行の並び・他の列は変えない。dry_runは何件が対象かだけを返す。
    """
    path = completion_ledger_path(state_root)
    summary = {"rows": 0, "outputs": 0, "recorded": 0, "already_recorded": 0,
               "not_recordable": [], "written": False}
    if not path.exists():
        return summary
    before = path.read_bytes()
    header, rows = load_tsv(path)
    source_key = str(source_path.resolve()) if source_path is not None else None
    changed = False
    for row in rows:
        if ((input_sha256 is not None and row.get("input_sha256") != input_sha256)
                or (recipe_id is not None and row.get("recipe_id") != recipe_id)
                or (source_key is not None and row.get("source_path") != source_key)):
            continue
        try:
            items = json.loads(row.get("outputs") or "[]")
        except ValueError:
            continue
        if not isinstance(items, list):
            continue
        summary["rows"] += 1
        updated = False
        for item in items:
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                continue
            summary["outputs"] += 1
            if isinstance(item.get("audio"), dict):
                summary["already_recorded"] += 1
                continue
            output = Path(item["path"])
            try:
                if not output.is_file() or input_sha256_of(output) != item["sha256"]:
                    summary["not_recordable"].append(str(output))
                    continue
                item["audio"] = audio_identity.audio_identity(output)
            except (audio_identity.AudioIdentityError, OSError):
                summary["not_recordable"].append(str(output))
                continue
            summary["recorded"] += 1
            updated = True
        if updated:
            row["outputs"] = json.dumps(items, ensure_ascii=False)
            changed = True
    if changed and not dry_run:
        if path.read_bytes() != before:
            # 記録中に別の完了が書き込まれた。既存の台帳を上書きしない。
            raise CompletionLedgerChangedError("記録中に完了台帳が変更されています")
        save_tsv(path, list(dict.fromkeys(header + _COMPLETION_HEADER)), rows)
    summary["written"] = bool(changed) and not dry_run
    summary["not_recordable"] = sorted(set(summary["not_recordable"]))
    return summary


def same_input_audio(recorded: object, observed: object) -> bool:
    """記録した入力の復号sampleと今の入力の復号sampleが同じか。記録が無い・方式が違えば同じと言わない。"""
    return (isinstance(recorded, dict) and isinstance(observed, dict) and bool(recorded.get("pcm_sha256"))
            and all(recorded.get(key) == observed.get(key) for key in _INPUT_AUDIO_FIELDS))


def record_input_audio(
    input_sha256: str, recipe_id: str, source_path: Path, identity: dict, *, state_root: Path | None = None,
) -> bool:
    """完了の行へ、入力の復号sampleの同一性を後から残す（未記録の行だけ。他の列と行の並びは変えない）。

    再生ソフトが原本へtagを書き戻しても、同じ音の入力として完了を見つけるための記録。
    """
    path = completion_ledger_path(state_root)
    if not path.exists() or not isinstance(identity, dict):
        return False
    before = path.read_bytes()
    header, rows = load_tsv(path)
    source_key = str(Path(source_path).resolve())
    changed = False
    for row in rows:
        if (row.get("input_sha256") == input_sha256 and row.get("recipe_id") == recipe_id
                and row.get("source_path") == source_key and not row.get("input_audio")):
            row["input_audio"] = json.dumps(identity, ensure_ascii=False, sort_keys=True)
            changed = True
    if not changed:
        return False
    if path.read_bytes() != before:
        raise CompletionLedgerChangedError("記録中に完了台帳が変更されています")
    save_tsv(path, list(dict.fromkeys(header + _COMPLETION_HEADER)), rows)
    return True


def completions_with_input_audio(source_path: Path, *, state_root: Path | None = None) -> list[dict]:
    """同じ所在の完了の行のうち、入力の復号sampleの同一性を記録したもの（新しい順）。"""
    path = completion_ledger_path(state_root)
    if not path.exists():
        return []
    _, rows = load_tsv(path)
    source_key = str(Path(source_path).resolve())
    found = []
    for row in reversed(rows):
        if row.get("source_path") != source_key or not row.get("input_audio"):
            continue
        try:
            identity = json.loads(row["input_audio"])
        except ValueError:
            continue
        if isinstance(identity, dict):
            found.append({**row, "input_audio": identity})
    return found


def is_completed(
    input_sha256: str, recipe_id: str, source_path: Path, *, state_root: Path | None = None,
) -> bool:
    if completed_zero_result(
        input_sha256, recipe_id, source_path, state_root=state_root,
    ) is not None:
        return True
    checked = check_completed_outputs(input_sha256, recipe_id, source_path, state_root=state_root)
    return bool(checked) and all(checked.values())


def validate_zero_result(value: object) -> dict:
    """検証証拠のない空出力を成功扱いにしないための最小契約。"""
    if not isinstance(value, dict) or set(value) != {"reason", "validation_result"}:
        raise ValueError("zero_resultはreasonとvalidation_resultを持つ必要があります")
    reason = value.get("reason")
    if reason not in ZERO_RESULT_REASONS:
        raise ValueError(f"zero_result reasonが不正です: {reason!r}")
    validation = value.get("validation_result")
    if not isinstance(validation, dict):
        raise ValueError("zero_resultのvalidation_resultが不正です")
    for field in ("validator_id", "validator_revision"):
        if not isinstance(validation.get(field), str) or not validation[field].strip():
            raise ValueError(f"zero_resultの{field}が未指定です")
    if validation.get("hard_gates_passed") is not True:
        raise ValueError("zero_resultはhard gate通過証拠が必要です")
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("zero_resultをJSONへ保存できません") from exc
    return json.loads(json.dumps(value, ensure_ascii=False))


def completed_zero_result(
    input_sha256: str,
    recipe_id: str,
    source_path: Path,
    *,
    state_root: Path | None = None,
) -> dict | None:
    path = completion_ledger_path(state_root)
    if not path.exists():
        return None
    _, rows = load_tsv(path)
    source_key = str(source_path.resolve())
    for row in reversed(rows):
        if (
            row.get("input_sha256") != input_sha256
            or row.get("recipe_id") != recipe_id
            or row.get("source_path") != source_key
            or row.get("outcome") != "zero"
        ):
            continue
        try:
            outputs = json.loads(row.get("outputs") or "[]")
            validation = json.loads(row.get("validation_result") or "null")
            return validate_zero_result({
                "reason": row.get("zero_reason"),
                "validation_result": validation,
            }) if outputs == [] else None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def mark_completed(
    input_sha256: str, recipe_id: str, when: str | None = None, *, source_path: Path, outputs: list[Path],
    progress: Callable[[Path, int, int], None] | None = None,
    state_root: Path | None = None,
) -> None:
    if not outputs:
        raise ValueError("完了として記録する出力がありません")
    output_records = []
    for output in outputs:
        if output.stat().st_size == 0:
            raise ValueError(f"出力が空です: {output}")
        record = {
            "path": str(output.resolve()),
            "sha256": input_sha256_of(
                output,
                progress=(
                    (lambda completed, total, output=output: progress(output, completed, total))
                    if progress is not None else None
                ),
            ),
        }
        try:
            # 外部の再生ソフトがtagを書き戻しても、同じ音であることを後から示せるようにする。
            record["audio"] = audio_identity.audio_identity(output)
        except (audio_identity.AudioIdentityError, OSError):
            pass
        output_records.append(record)
    path = completion_ledger_path(state_root)
    if path.exists():
        header, rows = load_tsv(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        header, rows = _COMPLETION_HEADER, []

    header = list(dict.fromkeys(header + _COMPLETION_HEADER))
    source_key = str(source_path.resolve())
    rows = [row for row in rows if not (
        row.get("input_sha256") == input_sha256 and row.get("recipe_id") == recipe_id
        and row.get("source_path") == source_key
    )]

    rows.append({
        "input_sha256": input_sha256,
        "recipe_id": recipe_id,
        "completed_at": when or datetime.now(timezone.utc).isoformat(),
        "source_path": source_key,
        "outcome": "artifacts",
        "zero_reason": "",
        "validation_result": "",
        "outputs": json.dumps(output_records, ensure_ascii=False),
    })
    save_tsv(path, header, rows)


def mark_zero_result(
    input_sha256: str,
    recipe_id: str,
    *,
    source_path: Path,
    zero_result: dict,
    when: str | None = None,
    state_root: Path | None = None,
) -> None:
    validated = validate_zero_result(zero_result)
    path = completion_ledger_path(state_root)
    if path.exists():
        header, rows = load_tsv(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        header, rows = _COMPLETION_HEADER, []
    header = list(dict.fromkeys(header + _COMPLETION_HEADER))
    source_key = str(source_path.resolve())
    rows = [row for row in rows if not (
        row.get("input_sha256") == input_sha256 and row.get("recipe_id") == recipe_id
        and row.get("source_path") == source_key
    )]
    rows.append({
        "input_sha256": input_sha256,
        "recipe_id": recipe_id,
        "completed_at": when or datetime.now(timezone.utc).isoformat(),
        "source_path": source_key,
        "outcome": "zero",
        "zero_reason": validated["reason"],
        "validation_result": json.dumps(
            validated["validation_result"], ensure_ascii=False, separators=(",", ":"),
        ),
        "outputs": "[]",
    })
    save_tsv(path, header, rows)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}-", suffix=".tmp",
            encoding="utf-8", newline="\n", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _publication_artifact(
    final_path: Path, staged_path: Path | None,
) -> dict[str, str | None]:
    final = final_path.resolve()
    if final.suffix.lower() != ".flac":
        raise ValueError(f"公開対象はFLACである必要があります: {final}")
    if staged_path is None:
        source = final
        staged = None
    else:
        staged = staged_path.resolve()
        if (
            staged.parent != final.parent
            or not staged.name.startswith(".lss-")
            or staged.suffix.lower() != ".tmp"
        ):
            raise ValueError(f"公開一時ファイルが最終出力と同じ安全な領域にありません: {staged}")
        if final.exists():
            raise FileExistsError(f"公開先が既に存在します: {final}")
        source = staged
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError(f"公開候補が空か存在しません: {source}")
    return {
        "final_path": str(final),
        "staged_path": str(staged) if staged is not None else None,
        "sha256": input_sha256_of(source),
    }


def prepare_publication(
    input_sha256: str,
    recipe_id: str,
    *,
    source_path: Path,
    outputs: list[Path],
    staged: list[tuple[Path, Path]],
    state_root: Path | None = None,
) -> Path:
    """全候補を検証してから、rename前の復旧情報を永続化する。"""
    if not outputs:
        raise ValueError("公開する出力がありません")
    staged_by_final = {final.resolve(): temporary for temporary, final in staged}
    if len(staged_by_final) != len(staged):
        raise ValueError("同じ公開先が重複しています")
    artifacts = [
        _publication_artifact(output, staged_by_final.get(output.resolve()))
        for output in outputs
    ]
    transaction_id = uuid4().hex
    path = publication_dir(state_root) / f"{transaction_id}.json"
    _write_json_atomic(path, {
        "version": _PUBLICATION_VERSION,
        "transaction_id": transaction_id,
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_sha256,
        "recipe_id": recipe_id,
        "source_path": str(source_path.resolve()),
        "artifacts": artifacts,
    })
    return path


def _load_publication(path: Path, state_root: Path | None = None) -> dict:
    root = publication_dir(state_root).resolve()
    resolved = path.resolve()
    if resolved.parent != root or resolved.suffix.lower() != ".json":
        raise ValueError(f"公開ジャーナルが管理範囲外です: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != _PUBLICATION_VERSION:
        raise ValueError(f"公開ジャーナルのversionが不正です: {path}")
    for field in ("transaction_id", "input_sha256", "recipe_id", "source_path"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"公開ジャーナルの{field}が不正です: {path}")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError(f"公開ジャーナルのartifactsが不正です: {path}")
    artifacts = []
    finals: set[Path] = set()
    for raw in raw_artifacts:
        if not isinstance(raw, dict) or set(raw) != {"final_path", "staged_path", "sha256"}:
            raise ValueError(f"公開ジャーナルのartifactが不正です: {path}")
        final = Path(raw["final_path"]).resolve() if isinstance(raw.get("final_path"), str) else None
        staged_value = raw.get("staged_path")
        staged = Path(staged_value).resolve() if isinstance(staged_value, str) else None
        digest = raw.get("sha256")
        if (
            final is None or final.suffix.lower() != ".flac" or final in finals
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"公開ジャーナルのartifact識別が不正です: {path}")
        if staged is not None and (
            staged.parent != final.parent
            or not staged.name.startswith(".lss-")
            or staged.suffix.lower() != ".tmp"
        ):
            raise ValueError(f"公開ジャーナルの一時ファイルが管理境界外です: {staged}")
        finals.add(final)
        artifacts.append({"final": final, "staged": staged, "sha256": digest})
    payload["artifacts"] = artifacts
    payload["journal_path"] = resolved
    return payload


def _matches_sha256(path: Path, expected: str) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0 and input_sha256_of(path) == expected
    except FileNotFoundError:
        return False


def complete_publication(
    journal_path: Path,
    *,
    state_root: Path | None = None,
    progress: Callable[[Path, int, int], None] | None = None,
) -> list[Path]:
    payload = _load_publication(journal_path, state_root)
    outputs = [artifact["final"] for artifact in payload["artifacts"]]
    for artifact in payload["artifacts"]:
        if not _matches_sha256(artifact["final"], artifact["sha256"]):
            raise ValueError(f"公開後の成果物が候補と一致しません: {artifact['final']}")
    mark_completed(
        payload["input_sha256"], payload["recipe_id"], source_path=Path(payload["source_path"]),
        outputs=outputs, progress=progress, state_root=state_root,
    )
    payload["journal_path"].unlink()
    return outputs


def abort_publication(journal_path: Path, *, state_root: Path | None = None) -> None:
    payload = _load_publication(journal_path, state_root)
    payload["journal_path"].unlink(missing_ok=True)


def _quarantine_publication(path: Path, state_root: Path | None = None) -> Path:
    failure_root = (Path(state_root) if state_root is not None else state_dir()) / "publication_failures"
    failure_root.mkdir(parents=True, exist_ok=True)
    target = failure_root / path.name
    if target.exists():
        target = failure_root / f"{path.stem}-{uuid4().hex}.json"
    path.replace(target)
    return target


def reconcile_publications(state_root: Path | None = None) -> list[dict[str, str]]:
    """中断された複数成果物の公開を完成させるか、安全に部分公開を戻す。"""
    root = publication_dir(state_root)
    if not root.exists():
        return []
    results: list[dict[str, str]] = []
    for journal_path in sorted(root.glob("*.json")):
        try:
            payload = _load_publication(journal_path, state_root)
        except OSError as exc:
            results.append({"status": "pending", "journal": str(journal_path), "message": str(exc)})
            continue
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            try:
                quarantined = _quarantine_publication(journal_path, state_root)
            except OSError as io_exc:
                results.append({"status": "pending", "journal": str(journal_path), "message": str(io_exc)})
                continue
            results.append({"status": "failed", "journal": str(quarantined), "message": str(exc)})
            continue

        failure = None
        try:
            for artifact in payload["artifacts"]:
                final = artifact["final"]
                staged = artifact["staged"]
                digest = artifact["sha256"]
                if _matches_sha256(final, digest):
                    continue
                if final.exists():
                    raise ValueError(f"公開先に別の実体があります: {final}")
                if staged is None or not _matches_sha256(staged, digest):
                    raise ValueError(f"公開候補を復旧できません: {final}")
                staged.rename(final)
            complete_publication(journal_path, state_root=state_root)
            results.append({"status": "completed", "journal": str(journal_path)})
            continue
        except OSError as exc:
            # 一時的な読み取り・rename・台帳保存・journal削除の失敗は破損ではない。
            # 検証済み成果物と復旧情報を保持し、次回に同じ公開を再試行する。
            results.append({"status": "pending", "journal": str(journal_path), "message": str(exc)})
            continue
        except (TypeError, ValueError) as exc:
            failure = exc

        # 新規に公開したtransaction所有物だけを正確なhashで戻す。再利用済み成果物は触らない。
        try:
            for artifact in payload["artifacts"]:
                if artifact["staged"] is None:
                    continue
                if _matches_sha256(artifact["final"], artifact["sha256"]):
                    artifact["final"].unlink(missing_ok=True)
                if _matches_sha256(artifact["staged"], artifact["sha256"]):
                    artifact["staged"].unlink(missing_ok=True)
            quarantined = _quarantine_publication(journal_path, state_root)
        except OSError as exc:
            # 後処理も再試行可能。journalを残し、他のtransactionの復旧を続ける。
            results.append({"status": "pending", "journal": str(journal_path), "message": str(exc)})
            continue
        results.append({"status": "rolled_back", "journal": str(quarantined), "message": str(failure)})
    return results


def validate_input(path: Path) -> None:
    info = sf.info(path)
    if info.frames <= 0:
        raise ValueError(f"入力に音声サンプルがありません: {path}")
    channels = info.channels
    if channels > MAX_CHANNELS:
        raise UnsupportedChannelLayoutError(
            f"チャンネル数{channels}は上限{MAX_CHANNELS}を超える(path={path})",
            channels=channels,
        )


def iter_input_files(paths: list[Path], extensions: frozenset[str] = AUDIO_EXTENSIONS) -> list[Path]:
    """フォルダ登録ではLSSの生成物を除外し、繰り返し登録による再分離を防ぐ。

    生成物もファイルを直接指定した場合は入力として扱う。
    展開・重複除去した新規バッチを、アルバム番号または自然なパス順で揃える。
    """
    collected: dict[Path, None] = {}
    for path in paths:
        if path.is_dir():
            candidates = (
                p for p in sorted(path.rglob("*"))
                if p.is_file() and (p.suffix.lower() in extensions or _is_acquired_media(p))
                and not _is_separated_output(p)
            )
        else:
            candidates = [path]
        for candidate in candidates:
            collected[candidate.resolve()] = None
    from lss.core.input_order import order_input_batch

    return order_input_batch(list(collected))


# URLから取得した素材は、変換せずに届いた入れ物のまま保存される（WebM・MPEG-TS・MP4）。
ACQUIRED_MEDIA_EXTENSIONS = frozenset({".webm", ".ts", ".mp4", ".mka"})


def _is_acquired_media(path: Path) -> bool:
    """取得ライブラリ内の取得物だけを、フォルダ登録の対象に加える。一般のフォルダの動画は拾わない。"""
    if path.suffix.lower() not in ACQUIRED_MEDIA_EXTENSIONS:
        return False
    from lss.acquisition.store import library_root

    try:
        return path.resolve().is_relative_to(library_root().resolve())
    except OSError:
        return False


def _is_separated_output(path: Path) -> bool:
    if path.suffix.lower() != ".flac":
        return False
    try:
        return FLAC(path).get("LSS_SOURCE_TYPE") in (["separated"], ["bgm_candidate"])
    except (MutagenError, OSError):
        return False
