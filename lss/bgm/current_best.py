"""同じ原本のBGM公開結果のCurrentBest（今採用している結果）と、Recipeの版の置き換え（supersession）。

`docs/ASSETS_AND_PROVENANCE.md` の CurrentBest・Recipe supersession・Promotion to canonical result に対応する。

- 原本（内容SHA-256）ごとに、今採用している公開cueの集まりを ``<状態領域>/bgm-current-best/`` の記録で指す。
  CurrentBestは「最後のrunの結果」ではなく、cueごとに選ぶ（v13、bgm-product-repair-r3）。旧版のcueと新しい版の
  cueが同じ原本のCurrentBestに並んでよい（領域が重ならない限り）。
- **妥当性と最良の選択は別の判断。** 既に公開しているcueは、まず今も妥当か（``VALID``/``INVALID``/``UNKNOWN``）を
  そのcue自身の区間の観測で決める（`assess` は呼び出し側が渡す）。新しいrunが保留・未確定・失敗であることは、
  旧cueが不正になった根拠にしない。妥当な表現の中でだけ、どれが今の最良かを選ぶ（`select`）：
  - 旧cueが妥当で、新しい版が同じ領域を確かに良くしない → 旧cueを残す（同じ音・別の区切りでも置き換えない）
  - 旧cueが妥当で、新しい版がその領域を検証済みの改善（`IMPROVEMENTS`）で作り直した → 新しいcueへ置き換える
  - 旧cueが不正（観測の根拠つき）→ 公開から外して履歴へ移す（新しい版が保留でも外す）
  - 旧cueの無い領域に新しい版の公開できるcueがある → 加える
  ``UNKNOWN`` は不正ではない（残す）。不正・保留・未確定・無しの状態は記録で区別する。
- 新しいrunの公開cueは、まず公開フォルダと同じボリュームの作業領域（``.lss-bgm-staging``）へ書いて検証する。
  切り替えは記録（journal）を先に書いてから行う：外す旧ファイルを履歴（``.lss-bgm-history``）へhard link（できなければ
  複製して検証）→ 新ファイルを公開名へ原子的に置き換え → 外す旧ファイルの名前を外す → CurrentBestの記録を原子的に
  更新 → journalを消す。どの瞬間でも公開名は旧版か新版の完全なファイルを指し、外した旧版は履歴に残る。
- 中断した切り替えは次の起動・次の処理で、journalから続きを行う（同じ手順は何度行っても同じ結果）。新版の
  ファイルが失われていれば保存済みの保留候補から推論なしで作り直し、それもできなければ旧版へ戻す。
- 切り替えごとに、切り替える前のCurrentBest全体を履歴へ残す。`restore_previous` はその全体へ戻す（rollback）。
- 置き換えの対象は、同じ原本の、このRecipeが宣言した前の版（`SUPERSEDED_WORKFLOWS`）の完了runのcueだけ。
  記録の無い利用者のファイルには触れない。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

DIRECTORY = "bgm-current-best"
STAGING_DIR = ".lss-bgm-staging"
HISTORY_DIR = ".lss-bgm-history"
RECORD_TYPE = "bgm_current_best"
JOURNAL_TYPE = "bgm_current_best_switch"
# v2: CurrentBestをcueごとのファイルの集まりにする（各ファイルが自分のrun・Recipeを持つ）。v1（runごと）は読める。
SCHEMA_VERSION = 2
CAPABILITY = "bgm_episode_cues"
# このRecipe（v13）が、同じ原本について置き換えてよい前の版。公開cueを作っていた単話BGMの版（v6〜v12）。
# v11・v12は配備していない候補だが、隔離環境の結果も同じ扱いで置き換えられる。
SUPERSEDED_WORKFLOWS = tuple(f"episode-sfc-routed-cues-v{n}" for n in range(6, 13))
PUBLISHED_STATUS = "completed_published_cues"
RUN_TYPE = "bgm_episode_candidate_product_run"
VALIDITY_STATES = ("VALID", "INVALID", "UNKNOWN")
# 旧cueを確かに良くする、検証済みの改善。新しいcueがその処理を受け、旧cueのRecipeがその改善より前の版なら、
# その領域では新しいcueの方が良いと言える（改善の検証の範囲＝その処理が掛かったcue）。
IMPROVEMENTS = {
    # v8：SFCの音楽の出力に台詞が残るcueを、伴奏の既定modelとの共通成分にする（bgm-dialogue-residual-gate-r1）
    "dialogue_residual_whole_cue": {"introduced_in": 8,
                                    "evidence": ".state/validation/bgm-dialogue-residual-product-r1/report.json"},
    # v10：台詞の残る1秒窓の時間だけ同じ共通成分にする（bgm-model-dependence-r1、配備済み）
    "dialogue_residual_local_windows": {"introduced_in": 10,
                                        "evidence": ".state/validation/bgm-dialogue-local-product-r1/report.json"},
}
# 一曲の範囲が決まらなかった断片（cue_extent: unresolved）を、それを隙間なく覆い、範囲の決まった（unresolvedでない）
# 新しいcueが置き換える。断片か全体かは資産の完全性（docs/product/IDENTITY_AND_CURRENTBEST.md）で、新しいcueは同じ
# 原本の同じ音楽をより完全に持つ。音が同じなら良くしていない（`better`）。根拠：
# release-evidence/memory-extent-20261017/RESULTS.md（実137：20秒の断片 -> 309-380.37秒の一曲）。
EXTENT_IMPROVEMENT = "resolved_extent_covers_unresolved_fragment"
# 新旧のcueが「同じ領域」とみなす重なり（秒）。
OVERLAP_SECONDS = 0.5
# 旧cueの区間を新しいcueで覆ったとみなす端の許容（秒）。
COVER_TOLERANCE_SECONDS = 1.0


class CurrentBestError(RuntimeError):
    """CurrentBestの記録・切り替えを安全に進められない。"""


def supersession_declaration() -> dict:
    """Recipeへ入れる置き換えの宣言（Capability・適用範囲・置き換える前の版・選び方）。"""
    return {"capability": CAPABILITY,
            "applicability": "same source content (sha256); finished stereo program; single-episode BGM cues",
            "supersedes_workflows": list(SUPERSEDED_WORKFLOWS),
            "selection": "per-cue: validity of each public cue first (VALID/INVALID/UNKNOWN from observations of "
                         "the cue's own span), then the best valid representation per region; a new run being "
                         "held/uncertain never invalidates an old cue; replace a valid cue only by a declared "
                         "validated improvement",
            "improvements": {name: {"introduced_in": item["introduced_in"]} for name, item in IMPROVEMENTS.items()},
            "replacement": "per-source CurrentBest set; staged publication; journaled hard-link swap; "
                           "prior public files archived with lineage; whole previous set kept for rollback",
            "reprocessing": "user-started processing of a source whose CurrentBest includes a superseded version"}


def workflow_version(workflow: str | None) -> int | None:
    found = re.fullmatch(r"episode-sfc-routed-cues-v(\d+)", str(workflow or ""))
    return int(found.group(1)) if found else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def pcm_sha256(path: Path) -> str | None:
    """公開ファイルの音（sample）の同一性。tagだけの書き戻しでは変わらない。読めなければNone。"""
    from lss.core import audio_identity

    try:
        return audio_identity.audio_identity(Path(path)).get("pcm_sha256")
    except Exception:
        return None


def _link_or_copy(source: Path, target: Path) -> None:
    """``source`` の内容を ``target`` へ完全なファイルとして置く（hard link、できなければ複製して検証）。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    _hide(target.parent)
    if target.exists():
        return
    try:
        os.link(source, target)
        return
    except OSError:
        pass
    with NamedTemporaryFile(dir=target.parent, prefix=".copy-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        if _sha256(temporary) != _sha256(source):
            raise CurrentBestError(f"履歴への複製が元と一致しません: {source}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _place_copy(source: Path, target: Path) -> None:
    """``source`` を ``target`` の名前へ原子的に置く（元は残す）。同じボリュームならhard link。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, prefix=".lss-bgm-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    temporary.unlink()
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hide(folder: Path) -> None:
    """作業領域・履歴のフォルダを、Windowsの隠し属性にする（できなければそのまま）。"""
    for path in (folder, folder.parent):
        if path.name in (STAGING_DIR, HISTORY_DIR) and os.name == "nt":
            try:
                import ctypes
                ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x2)
            except Exception:
                pass


def _matches(path: Path, pcm: str | None) -> bool:
    return pcm is not None and Path(path).is_file() and pcm_sha256(path) == pcm


def published_files(state_root: Path, source_sha256: str) -> dict[str, dict]:
    """The source's current published files by cue key (``{}`` when there is no readable CurrentBest record).

    This record's ``public_path`` is the only place LSS may write a presentation refresh for a published cue;
    paths remembered elsewhere (cue memory, recurrence assets) are observations, never write targets.
    """
    try:
        record = read(state_root, source_sha256)
    except (OSError, ValueError, CurrentBestError):
        return {}
    return {str(item["cue_key"]): item for item in ((record or {}).get("current") or {}).get("files") or []
            if item.get("cue_key") is not None and item.get("public_path")}


def record_path(state_root: Path, source_sha256: str) -> Path:
    return Path(state_root) / DIRECTORY / f"{source_sha256}.json"


def journal_path(state_root: Path, source_sha256: str) -> Path:
    return Path(state_root) / DIRECTORY / f"{source_sha256}.switch.json"


_RUN_FIELDS = ("run_key", "recipe_id", "workflow", "run_path")
_FILE_FIELDS = ("public_path", "pcm_sha256", "cue_key", "start_seconds", "end_seconds")


def _upgrade(record: dict) -> dict:
    """v1（runごとのCurrentBest）の記録をv2（cueごとのファイルの集まり）として読む。"""
    if record.get("schema_version") == SCHEMA_VERSION:
        return record

    def files_of(entry: dict) -> list[dict]:
        return [{**{name: item.get(name) for name in (*_FILE_FIELDS, "archive_path") if name in item},
                 **{name: entry.get(name) for name in _RUN_FIELDS},
                 "adopted_at": entry.get("adopted_at"), "adoption": entry.get("adoption")}
                for item in entry.get("files") or []]

    current = record["current"]
    upgraded = {"type": RECORD_TYPE, "schema_version": SCHEMA_VERSION, "source_sha256": record["source_sha256"],
                "current": {"files": files_of(current), "pinned": bool(current.get("pinned")),
                            "evaluated_runs": [current["run_key"]], "adopted_at": current.get("adopted_at"),
                            "transition": current.get("adoption")},
                "history": [{"files": files_of(item), "pinned": bool(item.get("pinned")),
                             "evaluated_runs": [item.get("run_key")], "retired_at": item.get("retired_at"),
                             "replaced_by_transition": item.get("retirement"),
                             "replaced_by_run_key": item.get("retired_by_run_key")}
                            for item in record.get("history") or []],
                "updated_at": record.get("updated_at"), "upgraded_from_schema": record.get("schema_version")}
    return upgraded


def read(state_root: Path, source_sha256: str) -> dict | None:
    path = record_path(state_root, source_sha256)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if (record.get("type") != RECORD_TYPE or record.get("schema_version") not in (1, SCHEMA_VERSION)
            or record.get("source_sha256") != source_sha256 or not isinstance(record.get("current"), dict)):
        raise CurrentBestError(f"CurrentBestの記録が不正です: {path}")
    return _upgrade(record)


def adopted_runs(state_root: Path) -> dict[str, dict[str, set]]:
    """原本ごとに、公開中のcueを出したrunと、そのrunの公開中のcueの識別子（記録のある原本だけ）。"""
    found: dict[str, dict[str, set]] = {}
    folder = Path(state_root) / DIRECTORY
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        if path.name.endswith(".switch.json"):
            continue
        try:
            record = _upgrade(json.loads(path.read_text(encoding="utf-8")))
            runs: dict[str, set] = {}
            for item in record["current"]["files"]:
                runs.setdefault(item["run_key"], set()).add(item.get("cue_key"))
            found[record["source_sha256"]] = runs
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found


def file_entry(run: dict, run_key: str, run_path: Path, item: dict, public_path: Path, *, adoption: str) -> dict:
    """公開cue1件のCurrentBestの項目（そのcueを作ったrun・Recipeを持つ）。"""
    return {"public_path": str(public_path), "pcm_sha256": (item.get("audio") or {}).get("pcm_sha256"),
            "cue_key": item.get("key"), "start_seconds": item.get("start_seconds"),
            "end_seconds": item.get("end_seconds"), "run_key": run_key, "recipe_id": run["recipe_id"],
            "workflow": run.get("workflow"), "run_path": str(run_path), "adopted_at": _now(), "adoption": adoption,
            "improvements": cue_improvements(item), "extent": (item.get("cue_extent") or {}).get("state")}


def cue_improvements(item: dict) -> list[str]:
    """公開cueが受けた、検証済みの改善（`IMPROVEMENTS`）。"""
    residual = item.get("dialogue_residual") or {}
    if not residual.get("applied"):
        return []
    return ["dialogue_residual_local_windows" if residual.get("scope") == "local" else "dialogue_residual_whole_cue"]


def legacy_current(state_root: Path, source_sha256: str, exclude_key: str) -> list[tuple[str, Path, dict]]:
    """CurrentBestの記録が無い原本について、前の版の完了run（公開cueあり）を集める（記録を作る前の結果）。"""
    found = []
    root = Path(state_root) / "bgm-candidates"
    for run_path in sorted(root.glob("*/run/run.json")) if root.is_dir() else []:
        key = run_path.parent.parent.name
        if key == exclude_key:
            continue
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (run.get("type") == RUN_TYPE and (run.get("source") or {}).get("sha256") == source_sha256
                and run.get("status") == PUBLISHED_STATUS and run.get("workflow") in SUPERSEDED_WORKFLOWS):
            found.append((key, run_path, run))
    return found


def _overlap(a: dict, b: dict) -> float:
    try:
        return min(float(a["end_seconds"]), float(b["end_seconds"])) - max(float(a["start_seconds"]),
                                                                            float(b["start_seconds"]))
    except (KeyError, TypeError, ValueError):
        return 0.0


def _covers(old: dict, news: list[dict]) -> bool:
    """新しいcueの集まりが旧cueの区間を（端の許容のうちで）隙間なく覆うか。"""
    spans = sorted((float(item["start_seconds"]), float(item["end_seconds"])) for item in news)
    cursor = float(old["start_seconds"]) + COVER_TOLERANCE_SECONDS
    for a, b in spans:
        if a > cursor + 1e-9:
            return False
        cursor = max(cursor, b)
    return cursor >= float(old["end_seconds"]) - COVER_TOLERANCE_SECONDS - 1e-9


def better(new: dict, old: dict) -> list[str]:
    """``new`` が ``old`` を確かに良くする検証済みの改善の名前（無ければ空）。同じ音なら良くしていない。"""
    if new.get("pcm_sha256") is not None and new.get("pcm_sha256") == old.get("pcm_sha256"):
        return []
    if old.get("extent") == "unresolved" and new.get("extent") not in (None, "unresolved"):
        try:
            longer = (float(new["end_seconds"]) - float(new["start_seconds"])
                      > float(old["end_seconds"]) - float(old["start_seconds"]) + COVER_TOLERANCE_SECONDS)
        except (KeyError, TypeError, ValueError):
            longer = False
        if longer:
            return [EXTENT_IMPROVEMENT]
    version = workflow_version(old.get("workflow"))
    if version is None:
        return []
    return [name for name in new.get("improvements") or []
            if name in IMPROVEMENTS and version < IMPROVEMENTS[name]["introduced_in"]]


def select(old_files: list[dict], assessments: list[dict], new_files: list[dict]) -> dict:
    """妥当性を決めた旧cueと新しいrunの公開cueから、今のCurrentBestを選ぶ（ファイルは動かさない）。

    ``assessments[i]`` は ``old_files[i]`` の妥当性 ``{"state": VALID|INVALID|UNKNOWN, "basis": [...]}``。
    戻り値は keep（残す旧cue）・retire（外す旧cue、理由つき）・place（加える新cue）・not_placed（加えない新cue、
    理由つき）と、旧cue・新cueごとの判断（decisions）。
    """
    if len(old_files) != len(assessments):
        raise CurrentBestError("旧cueと妥当性の数が一致しません")
    keep, retire, decisions = [], [], []
    replaced_by: dict[int, list[int]] = {}
    for index, (old, assessment) in enumerate(zip(old_files, assessments)):
        state = assessment.get("state")
        if state not in VALIDITY_STATES:
            raise CurrentBestError(f"旧cueの妥当性が不正です: {state!r}")
        row = {"public_path": old["public_path"], "run_key": old.get("run_key"), "validity": state,
               "basis": list(assessment.get("basis") or []), "context": assessment.get("context")}
        if state == "INVALID":
            retire.append({**old, "retirement": "invalid", "basis": row["basis"]})
            decisions.append({**row, "decision": "retire_invalid"})
            continue
        overlapping = [j for j, new in enumerate(new_files) if _overlap(old, new) > OVERLAP_SECONDS]
        improved = [j for j in overlapping if better(new_files[j], old)]
        if improved and _covers(old, [new_files[j] for j in improved]):
            names = sorted({name for j in improved for name in better(new_files[j], old)})
            retire.append({**old, "retirement": "superseded_by_validated_improvement", "basis": names})
            replaced_by[index] = improved
            decisions.append({**row, "decision": "replace_with_better", "improvements": names,
                              "replaced_by": [new_files[j]["public_path"] for j in improved]})
            continue
        keep.append(old)
        decisions.append({**row, "decision": "keep",
                          "why": ("valid_and_new_not_demonstrably_better" if overlapping
                                  else "valid_and_no_new_cue_in_region") if state == "VALID"
                          else "validity_unknown_is_not_invalid"})
    place, not_placed = [], []
    for new in new_files:
        blocking = [old["public_path"] for old in keep if _overlap(old, new) > OVERLAP_SECONDS]
        if blocking:
            not_placed.append({**new, "reason": "region_has_valid_current_best", "kept": blocking})
        else:
            place.append(new)
    kept_names = {Path(item["public_path"]).name.casefold() for item in keep}
    clash = [item["public_path"] for item in place if Path(item["public_path"]).name.casefold() in kept_names]
    if clash:
        raise CurrentBestError(f"残す旧cueと新しいcueの公開名が衝突します: {clash}")
    return {"keep": keep, "retire": retire, "place": place, "not_placed": not_placed, "decisions": decisions,
            "new_decisions": [{"public_path": item["public_path"], "decision": "place"} for item in place]
            + [{"public_path": item["public_path"], "decision": "not_placed", "reason": item["reason"],
                "kept": item["kept"]} for item in not_placed]}


def _history_path(history_root: Path, run_key: str, public: Path) -> Path:
    return Path(history_root) / str(run_key)[:16] / Path(public).name


def plan_switch(state_root: Path, source_sha256: str, next_current: dict, staged: dict[str, Path],
                history_root: Path, *, retire: list[dict], record: dict | None, previous_current: dict | None,
                mode: str, history: list[dict] | None = None) -> dict:
    """切り替えのjournalを作る（まだ何も動かさない）。

    ``next_current`` は切り替えた後のCurrentBest全体（``files`` に残す旧cueと加える新cue）。``staged`` は加える
    新cueの公開名→作業領域（または履歴）のファイル。``retire`` は公開から外す旧cue。``previous_current`` は
    切り替える前のCurrentBest全体で、外すファイルの履歴の場所を付けて履歴へ残す（rollbackの戻り先）。
    """
    retired_at = _now()
    archive = {item["public_path"]: str(_history_path(history_root, item.get("run_key") or "unknown",
                                                      Path(item["public_path"])))
               for item in retire}
    retire_files = [{"public_path": item["public_path"], "pcm_sha256": item.get("pcm_sha256"),
                     "archive_path": archive[item["public_path"]], "run_key": item.get("run_key")}
                    for item in retire]
    kept = {item["public_path"] for item in next_current["files"]}
    place = [{"public_path": item["public_path"], "pcm_sha256": item["pcm_sha256"],
              "from_path": str(staged[item["public_path"]]) if item["public_path"] in staged else None}
             for item in next_current["files"] if item.get("placed_now")]
    for item in next_current["files"]:
        item.pop("placed_now", None)
    history = list((record or {}).get("history") or []) if history is None else history
    if previous_current is not None and previous_current.get("files") and (retire or place):
        snapshot = {**previous_current,
                    "files": [{**item, **({"archive_path": archive[item["public_path"]]}
                                          if item["public_path"] in archive else {})}
                              for item in previous_current["files"]],
                    "retired_at": retired_at, "replaced_by_transition": mode,
                    "replaced_by_run_key": next_current.get("evaluated_runs", [None])[-1]}
        history.append(snapshot)
    next_record = {"type": RECORD_TYPE, "schema_version": SCHEMA_VERSION, "source_sha256": source_sha256,
                   "current": next_current, "history": history, "updated_at": retired_at}
    if not kept >= {item["public_path"] for item in place}:
        raise CurrentBestError("置くファイルがCurrentBestに入っていません")
    return {"type": JOURNAL_TYPE, "schema_version": SCHEMA_VERSION, "source_sha256": source_sha256,
            "mode": mode, "created_at": retired_at, "history_root": str(history_root), "retire": retire_files,
            "place": place, "next_record": next_record, "previous_record": record}


def public_audio_key(item: dict) -> tuple[Path, str | None]:
    """Same physical location and PCM, preserving the recorded path as provenance.

    Windows junction aliases can spell one public asset through C: and E:.
    Treating those spellings as different assets can retire a file we kept.
    """
    return Path(item["public_path"]).resolve(), item.get("pcm_sha256")


def execute(state_root: Path, journal: dict,
            regenerate: Callable[[dict, Path], None] | None = None) -> dict:
    """journalどおりに切り替える。何度行っても同じ結果になる。新版を置けなければ旧版へ戻して例外。"""
    source_sha256 = journal["source_sha256"]
    path = journal_path(state_root, source_sha256)
    if not path.is_file():
        _write_json_atomic(path, journal)
    kept_audio = {public_audio_key(item) for item in journal["next_record"]["current"]["files"]}
    try:
        for item in journal["retire"]:
            public, archive = Path(item["public_path"]), Path(item["archive_path"])
            if archive.is_file() and _matches(archive, item["pcm_sha256"]):
                continue
            if _matches(public, item["pcm_sha256"]):
                _link_or_copy(public, archive)
            # 公開名にも履歴にも旧版が無い（利用者が消した等）。runの記録から作り直せるので止めない。
        retired = {item["public_path"]: item for item in journal["retire"]}
        for item in journal["place"]:
            public = Path(item["public_path"])
            if _matches(public, item["pcm_sha256"]):
                continue
            if public.exists():
                old = retired.get(item["public_path"])
                if (old is None or not _matches(public, old["pcm_sha256"])
                        or not _matches(Path(old["archive_path"]), old["pcm_sha256"])):
                    # 記録に無い（または履歴へ退避できていない）ファイルを無言で上書きしない。履歴の別の場所へ移す。
                    _displace(public, journal, path)
            source = Path(item["from_path"]) if item.get("from_path") else None
            if source is not None and _matches(source, item["pcm_sha256"]):
                public.parent.mkdir(parents=True, exist_ok=True)
                if HISTORY_DIR in source.parts:
                    _place_copy(source, public)   # 履歴の版は残したまま公開名へ置く
                else:
                    os.replace(source, public)    # 作業領域の新版を公開名へ移す
            elif regenerate is not None:
                regenerate(item, public)
            if not _matches(public, item["pcm_sha256"]):
                raise CurrentBestError(f"新しい公開cueを置けません: {public}")
        for item in journal["retire"]:
            public = Path(item["public_path"])
            if public_audio_key(item) not in kept_audio and _matches(public, item["pcm_sha256"]):
                if Path(item["archive_path"]).is_file() or not public.is_file():
                    public.unlink()
        _settle_kept_missing(path, journal, regenerate)
        for item in journal["next_record"]["current"]["files"]:
            if not _matches(Path(item["public_path"]), item.get("pcm_sha256")):
                raise CurrentBestError(f"切り替え後の公開cueが記録と一致しません: {item['public_path']}")
        _write_json_atomic(record_path(state_root, source_sha256), journal["next_record"])
    except Exception:
        _roll_back(state_root, journal)
        raise
    path.unlink(missing_ok=True)
    _remove_empty_staging(journal)
    return journal["next_record"]


def _displace(public: Path, journal: dict, journal_file: Path) -> None:
    """Move a file that is not LSS's recorded one out of a public name into ``history/displaced`` (never deleted)."""
    displaced = Path(journal["history_root"]) / "displaced" / journal["created_at"].replace(":", "")
    displaced.mkdir(parents=True, exist_ok=True)
    _hide(displaced.parent.parent)
    target = displaced / public.name
    suffix = 1
    while target.exists():
        suffix += 1
        target = displaced / f"{public.stem} (r{suffix}){public.suffix}"
    os.replace(public, target)
    journal["next_record"].setdefault("displaced", []).append({"public_path": str(public), "moved_to": str(target)})
    _write_json_atomic(journal_file, journal)


def _settle_kept_missing(journal_file: Path, journal: dict,
                         regenerate: Callable[[dict, Path], None] | None) -> None:
    """A cue kept from the previous record whose file is gone (deleted by the user) is rebuilt from its run record.

    The rebuilt PCM must equal the recorded ``pcm_sha256``. When it cannot be rebuilt (run record or audio gone) the
    switch leaves the cue out of ``current.files`` and records it in ``current.unavailable`` as UNKNOWN (not ABSENT:
    nothing says the cue is invalid), with the full former row so it can be restored. A file that is present with a
    different PCM is not LSS's recorded cue: it is moved to ``history/displaced`` (never deleted) and the cue is rebuilt.
    """
    placed = {item["public_path"] for item in journal["place"]}
    current = journal["next_record"]["current"]
    for item in current["files"]:
        public = Path(item["public_path"])
        if (item["public_path"] not in placed and item.get("pcm_sha256") and public.is_file()
                and not _matches(public, item["pcm_sha256"])):
            _displace(public, journal, journal_file)   # not the recorded sound: keep the bytes, free the name
    missing = [item for item in current["files"]
               if item["public_path"] not in placed and not Path(item["public_path"]).exists()]
    if not missing:
        return
    unavailable = []
    for item in missing:
        public = Path(item["public_path"])
        reason = "regenerate_unavailable"
        if regenerate is not None and item.get("pcm_sha256"):
            try:
                regenerate(item, public)
                if _matches(public, item["pcm_sha256"]):
                    continue
                reason = "regenerated_pcm_differs"
                if public.is_file():
                    public.unlink()   # LSS made this file just now; it is not the recorded sound
            except Exception as exc:  # noqa: BLE001 - an unrebuildable kept cue never fails the whole switch
                reason = f"regenerate_failed:{type(exc).__name__}"
        unavailable.append({**item, "state": "UNKNOWN", "reason": reason, "recorded_at": _now()})
    gone = {row["public_path"] for row in unavailable}
    current["files"] = [item for item in current["files"] if item["public_path"] not in gone]
    current["unavailable"] = [*(current.get("unavailable") or []), *unavailable]
    _write_json_atomic(journal_file, journal)


def _roll_back(state_root: Path, journal: dict) -> None:
    """新版を置けなかった切り替えを戻す。旧版のファイルを公開名へ戻し、記録は前のまま、journalを消す。"""
    previous = journal.get("previous_record")
    for item in journal["place"]:
        public = Path(item["public_path"])
        if not _matches(public, item["pcm_sha256"]):
            continue
        staged = Path(item["from_path"]) if item.get("from_path") else None
        if staged is not None and HISTORY_DIR not in staged.parts and not staged.exists():
            staged.parent.mkdir(parents=True, exist_ok=True)
            os.replace(public, staged)    # 新版は作業領域へ戻す（次の切り替えで使う）
        else:
            public.unlink()               # 履歴・作り直しから置いた版（元は残っている）
    for item in journal["retire"]:
        public, archive = Path(item["public_path"]), Path(item["archive_path"])
        if not _matches(public, item["pcm_sha256"]) and _matches(archive, item["pcm_sha256"]):
            if public.exists():
                _displace(public, journal, journal_path(state_root, journal["source_sha256"]))
            _place_copy(archive, public)
    for moved in journal["next_record"].get("displaced") or []:
        public, target = Path(moved["public_path"]), Path(moved["moved_to"])
        if target.is_file() and not public.exists():
            os.replace(target, public)
    if previous is not None:
        _write_json_atomic(record_path(state_root, journal["source_sha256"]), previous)
    journal_path(state_root, journal["source_sha256"]).unlink(missing_ok=True)


def _remove_empty_staging(journal: dict) -> None:
    for item in journal["place"]:
        source = item.get("from_path")
        if source and STAGING_DIR in Path(source).parts:
            folder = Path(source).parent
            try:
                folder.rmdir()
                folder.parent.rmdir()
            except OSError:
                pass


def reconcile(state_root: Path, source_sha256: str,
              regenerate: Callable[[dict, Path], None] | None = None) -> dict | None:
    """中断した切り替えがあれば続きを行う。無ければ何もしない。"""
    path = journal_path(state_root, source_sha256)
    try:
        journal = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError:
        # journal自体が書きかけ（原子的に書くので通常は起きない）。何も動いていないので捨てる。
        path.unlink(missing_ok=True)
        return None
    try:
        return execute(state_root, journal, regenerate)
    except Exception:
        return None


def reconcile_all(state_root: Path,
                  regenerator: Callable[[str], Callable[[dict, Path], None]] | None = None) -> int:
    """起動時に、中断した全ての切替の続きを行う。``regenerator(source_sha256)`` は作り直す関数を返す。"""
    folder = Path(state_root) / DIRECTORY
    count = 0
    for path in sorted(folder.glob("*.switch.json")) if folder.is_dir() else []:
        source_sha256 = path.name[:-len(".switch.json")]
        regenerate = regenerator(source_sha256) if regenerator is not None else None
        if reconcile(state_root, source_sha256, regenerate) is not None:
            count += 1
    return count


def restore_previous(state_root: Path, source_sha256: str, history_root: Path,
                     regenerate: Callable[[dict, Path], None] | None = None) -> dict:
    """直前の切り替えの前のCurrentBest全体へ戻す（rollback）。今のCurrentBest全体は履歴へ移す。

    戻した結果は利用者が選んだものとして固定（pinned）し、同じ原本を処理し直しても勝手に新しい版へ戻さない。
    もう一度呼ぶと、戻す前のCurrentBestへ戻る。
    """
    record = read(state_root, source_sha256)
    if record is None or not record.get("history"):
        raise CurrentBestError("戻せる前の結果がありません")
    target = record["history"][-1]
    current = record["current"]
    target_paths = {public_audio_key(item) for item in target["files"]}
    current_paths = {public_audio_key(item) for item in current["files"]}
    retire = [item for item in current["files"] if public_audio_key(item) not in target_paths]
    files = []
    staged = {}
    for item in target["files"]:
        entry = {name: value for name, value in item.items() if name != "archive_path"}
        if public_audio_key(item) not in current_paths:
            entry["placed_now"] = True
            if item.get("archive_path"):
                staged[item["public_path"]] = Path(item["archive_path"])
        files.append(entry)
    restored = {"files": files, "pinned": True, "adopted_at": _now(), "transition": "restored_previous",
                "evaluated_runs": list(target.get("evaluated_runs") or [])}
    journal = plan_switch(state_root, source_sha256, restored, staged, history_root, retire=retire,
                          record=record, previous_current=current, mode="restored_previous",
                          history=list(record["history"][:-1]))
    return execute(state_root, journal, regenerate)
