"""検証済みの処理更新（既定モデル・設定の変更）に対して、再処理の候補を抽出する。読み取りだけを行う。

- 各入力（内容SHA-256）の最新の完了記録について、使われたモデル・設定を成果物の来歴タグから読む。
- 現在の既定と違うモデル・設定の結果だけを候補にする。実行環境だけの違いは品質更新と見なさない。
- 既に現在の既定で処理済みの入力は候補にしない。
- 原本が同じ場所に同じ内容で残っているか、取得の記録から取り直せるかを示す。
- 再処理を始めるかどうかは利用者が決める。ここでは開始しない。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mutagen import MutagenError
from mutagen.flac import FLAC

from lss.core import session
from lss.harness.tsv_io import load_tsv

LEGACY_RECIPE_PREFIX = "separation-v"


@dataclass(frozen=True)
class ProcessingIdentity:
    model_id: str | None
    model_sha256: str | None
    config_sha256: str | None

    def compare(self, current: "ProcessingIdentity") -> str:
        if self.model_id is None:
            return "processing_unknown"
        if self.model_id != current.model_id or (
                self.model_sha256 and current.model_sha256 and self.model_sha256 != current.model_sha256):
            return "model_updated"
        if self.config_sha256 and current.config_sha256 and self.config_sha256 != current.config_sha256:
            return "config_updated"
        if any(not value for value in (self.model_sha256, self.config_sha256,
                                      current.model_sha256, current.config_sha256)):
            return "processing_unverifiable"
        return "current"


@dataclass(frozen=True)
class ReprocessCandidate:
    input_sha256: str
    source_path: str
    completed_at: str
    outcome: str
    reason: str
    previous: dict
    source_status: str
    reacquire_url: str | None
    outputs: list[str]
    source_locations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def current_identity() -> ProcessingIdentity:
    from lss.core import catalog

    row = catalog.default_model()
    return ProcessingIdentity(row["model_id"], row.get("sha256") or None, row.get("config_sha256") or None)


def _identity_from_outputs(outputs: list[dict], *, state_root: Path | None = None) -> ProcessingIdentity | None:
    from lss.assets.relocate import current_locations

    for record in outputs:
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str) or not record["sha256"]:
            continue
        try:
            original = Path(record["path"])
            paths = ([original] if original.is_file() and session.input_sha256_of(original) == record["sha256"]
                     else current_locations(record["sha256"], state_root=state_root))
        except (OSError, KeyError, TypeError):
            paths = current_locations(record["sha256"], state_root=state_root)
        for path in paths:
            try:
                tags = FLAC(path)
                found = None
                revision = tags.get("LSS_RECIPE_REVISION")
                if revision:
                    try:
                        stage = json.loads(revision[0])["stage_graph"][0]
                        if not isinstance(stage, dict):
                            raise ValueError("処理stageの来歴が不正です")
                        found = ProcessingIdentity(stage.get("model_id"), stage.get("model_sha256"), stage.get("config_sha256"))
                    except (ValueError, KeyError, IndexError, TypeError):
                        pass
                model = tags.get("LSS_MODEL")
                if found is None and model:
                    found = ProcessingIdentity(model[0], (tags.get("LSS_MODEL_SHA256") or [None])[0],
                                               (tags.get("LSS_CONFIG_SHA256") or [None])[0])
                # 所在だけでなく、読んだ来歴が完了時と同じ内容に属することを確認する。
                if found is not None and session.input_sha256_of(path) == record["sha256"]:
                    return found
            except (MutagenError, OSError):
                continue
    return None


def _verified_recipe_body(outputs: list, recipe_id: str) -> dict | None:
    """成果物のタグにあるRecipe本文のうち、台帳のrecipe_idと一致するものを返す。

    recipe_idは本文のhashなので、一致した本文はその完了で使った処理そのものである。
    再生ソフトがtagを書き戻して成果物のbytesが変わっていても、本文の正しさは変わらない。
    書き換えられた本文や別の処理の本文はhashが合わないので受け取らない。
    """
    from lss.core.recipe import validate_recipe_revision

    if not recipe_id.startswith(LEGACY_RECIPE_PREFIX):
        return None
    for item in outputs if isinstance(outputs, list) else []:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        path = Path(item["path"])
        try:
            if not path.is_file():
                continue
            revision = (FLAC(path).get("LSS_RECIPE_REVISION") or [None])[0]
        except (MutagenError, OSError):
            continue
        if not revision:
            continue
        try:
            body = validate_recipe_revision(json.loads(revision))
        except (ValueError, TypeError):
            continue
        graph = body.get("stage_graph")
        if (body.get("recipe_id") == recipe_id and isinstance(graph, list) and graph
                and isinstance(graph[0], dict)):
            return body
    return None


def _identity_of(row: dict, outputs: list[dict], *, state_root: Path | None = None) -> ProcessingIdentity:
    from lss.core import recipe_registry

    # 本文とIDの一致で検証できる自前の記録を先に使う。成果物のタグは外部から
    # 書き換えられ得るので、こちらの記録がある限りタグを読み直さない。
    recorded = recipe_registry.processing_identity(row.get("recipe_id") or "", state_root=state_root)
    if recorded is not None and recorded.get("model_id"):
        return ProcessingIdentity(recorded["model_id"], recorded.get("model_sha256"),
                                  recorded.get("config_sha256"))
    body = _verified_recipe_body(outputs, row.get("recipe_id") or "")
    if body is not None:
        stage = body["stage_graph"][0]
        return ProcessingIdentity(stage.get("model_id"), stage.get("model_sha256"),
                                  stage.get("config_sha256"))
    found = _identity_from_outputs(outputs, state_root=state_root)
    if found is not None:
        return found
    recipe_id = row.get("recipe_id") or ""
    if recipe_id and not recipe_id.startswith(LEGACY_RECIPE_PREFIX):
        # 初期の台帳はモデル識別子をrecipe_idとして記録していた。
        return ProcessingIdentity(recipe_id, None, None)
    return ProcessingIdentity(None, None, None)


def _source_status(path: str, input_sha256: str, *, verify: bool, state_root: Path | None) -> tuple[str, str | None, list[str]]:
    source = Path(path) if path else None
    if source is not None and source.is_file():
        if not verify:
            return "present_unverified", None, [str(source.resolve())]
        try:
            if session.input_sha256_of(source) == input_sha256:
                return "present", None, [str(source.resolve())]
            status = "changed"
        except OSError:
            status = "unreadable"
    else:
        status = "missing"
    from lss.assets.relocate import current_locations

    moved = sorted({str(location.resolve()) for location in current_locations(input_sha256, state_root=state_root)})
    if moved:
        return ("moved" if status == "missing" else f"{status}_with_verified_copy"), None, moved
    from lss.acquisition.store import records_for_sha256

    records = records_for_sha256(input_sha256, state_root)
    urls = [(r.get("reacquire") or {}).get("url") for r in records]
    urls = [url for url in urls if url]
    if urls:
        return f"{status}_reacquirable", urls[-1], []
    return status, None, []


def reprocess_candidates(*, state_root: Path | None = None, current: ProcessingIdentity | None = None,
                         verify_sources: bool = False) -> dict:
    """再処理の候補と、候補にしなかった理由の内訳を返す。"""
    current = current or current_identity()
    path = session.completion_ledger_path(state_root)
    try:
        _, rows = load_tsv(path)
    except FileNotFoundError:
        # 読めない既存台帳は「再処理候補なし」の根拠にならない。
        rows = []
    latest: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("input_sha256"):
            latest.setdefault(row["input_sha256"], []).append(row)

    candidates: list[ReprocessCandidate] = []
    counts: dict[str, int] = {}
    for input_sha256, entries in latest.items():
        entries.sort(key=lambda r: r.get("completed_at") or "")
        judged = []
        for row in entries:
            try:
                outputs = json.loads(row.get("outputs") or "[]")
            except ValueError:
                outputs = []
            outputs = outputs if isinstance(outputs, list) else []
            judged.append((row, outputs, _identity_of(row, outputs, state_root=state_root).compare(current)))
        if any(state == "current" for _, _, state in judged):
            counts["already_current"] = counts.get("already_current", 0) + 1
            continue
        row, outputs, state = judged[-1]
        counts[state] = counts.get(state, 0) + 1
        if state not in {"model_updated", "config_updated"}:
            continue
        status, url, locations = _source_status(row.get("source_path") or "", input_sha256,
                                                verify=verify_sources, state_root=state_root)
        identity = _identity_of(row, outputs, state_root=state_root)
        candidates.append(ReprocessCandidate(
            input_sha256, row.get("source_path") or "", row.get("completed_at") or "",
            row.get("outcome") or "artifacts", state, asdict(identity), status, url,
            [str(item.get("path")) for item in outputs if isinstance(item, dict)],
            locations,
        ))
    candidates.sort(key=lambda c: c.completed_at)
    return {"current": asdict(current), "counts": counts, "candidates": [c.to_dict() for c in candidates]}


def backfill_recipe_revisions(*, state_root: Path | None = None, dry_run: bool = False) -> dict:
    """成果物のタグからRecipe本文を拾い、自前の記録へ残す。

    本文は台帳のrecipe_idと一致することまで確かめてから保存する。IDは本文のhashなので、
    別の処理の本文を取り違えて記録することはない。再生ソフトがtagを書き戻して成果物の
    bytesが変わっていても、本文がIDと一致すれば拾える。
    """
    from lss.core import recipe_registry

    summary = {"rows": 0, "already_recorded": 0, "recorded": 0, "not_recoverable": [],
               "written": False}
    try:
        _, rows = load_tsv(session.completion_ledger_path(state_root))
    except (FileNotFoundError, OSError, ValueError):
        return summary
    seen = set()
    for row in rows:
        recipe_id = row.get("recipe_id") or ""
        if not recipe_id.startswith(LEGACY_RECIPE_PREFIX):
            # 初期の台帳はmodel_idをrecipe_idにしていた。本文のhashではないので拾えない。
            summary["not_recoverable"].append(recipe_id or "(空)")
            continue
        summary["rows"] += 1
        if recipe_id in seen:
            continue
        seen.add(recipe_id)
        if recipe_registry.revision_for(recipe_id, state_root=state_root) is not None:
            summary["already_recorded"] += 1
            continue
        try:
            outputs = json.loads(row.get("outputs") or "[]")
        except ValueError:
            outputs = []
        found = _verified_recipe_body(outputs, recipe_id)
        if found is None:
            summary["not_recoverable"].append(recipe_id)
            continue
        if not dry_run:
            recipe_registry.record(found, state_root=state_root)
            summary["written"] = True
        summary["recorded"] += 1
    summary["not_recoverable"] = sorted(set(summary["not_recoverable"]))
    return summary
