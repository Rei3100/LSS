"""入力ごとの処理結果の改訂履歴と、現在の最良結果を、完了台帳と成果物から読み取る。

- 改訂は完了記録1件（Recipe・処理時刻・成果物またはゼロ件の理由）。古い改訂も消さずに並べる。
- 各改訂の成果物が記録どおりの内容で残っているかを確かめる。
- 現在の最良結果は、成果物が記録どおりに残っている改訂のうち、台帳の現在の既定のモデル・設定で
  作られたものを優先し、無ければ残っている最新のもの。どれも残っていなければ無い（捏造しない）。
- 何も書き換えない。前の良い結果を守る公開の手順は、保存処理（publication journal）が担う。
"""
from __future__ import annotations

import json
from pathlib import Path

from lss.assets.reprocess import ProcessingIdentity, _identity_of, current_identity
from lss.core import session
from lss.harness.tsv_io import load_tsv


def _valid_zero(row: dict, decoded_outputs) -> bool:
    if decoded_outputs != []:
        return False
    try:
        session.validate_zero_result({"reason": row.get("zero_reason"),
                                      "validation_result": json.loads(row.get("validation_result") or "null")})
    except (TypeError, ValueError):
        return False
    return True


def _verified(outputs: list[dict], state_root: Path | None = None) -> bool:
    """成果物が記録どおりの内容で残っているか。移動していれば、記録した移動先も確かめる。"""
    from lss.assets.relocate import current_locations

    if not outputs:
        return False
    for item in outputs:
        try:
            path = Path(item["path"])
            if path.is_file() and session.input_sha256_of(path) == item["sha256"]:
                continue
            if not current_locations(item["sha256"], state_root=state_root):
                return False
        except (KeyError, TypeError, OSError):
            return False
    return True


def history(input_sha256: str, *, state_root: Path | None = None,
            current: ProcessingIdentity | None = None) -> dict:
    """1つの入力の改訂（古い順）と、現在の最良の改訂の位置を返す。"""
    current = current or current_identity()
    try:
        _, rows = load_tsv(session.completion_ledger_path(state_root))
    except FileNotFoundError:
        # 初回の台帳不在だけを空履歴にする。読取失敗で現在の最良結果を消さない。
        rows = []
    revisions = []
    for row in sorted((r for r in rows if r.get("input_sha256") == input_sha256),
                      key=lambda r: r.get("completed_at") or ""):
        try:
            decoded_outputs = json.loads(row.get("outputs") or "[]")
        except ValueError:
            decoded_outputs = None
        outputs = ([item for item in decoded_outputs if isinstance(item, dict)]
                   if isinstance(decoded_outputs, list) else [])
        identity = _identity_of(row, outputs, state_root=state_root)
        outcome = row.get("outcome") or ("artifacts" if outputs else "")
        revisions.append({
            "recipe_id": row.get("recipe_id"),
            "completed_at": row.get("completed_at"),
            "source_path": row.get("source_path"),
            "outcome": outcome,
            "zero_reason": row.get("zero_reason") or None,
            "processing": {"model_id": identity.model_id, "model_sha256": identity.model_sha256,
                           "config_sha256": identity.config_sha256},
            "processing_state": identity.compare(current),
            "outputs": [item.get("path") for item in outputs],
            "outputs_intact": _verified(outputs, state_root) if outcome == "artifacts" else None,
            "zero_result_valid": _valid_zero(row, decoded_outputs) if outcome == "zero" else None,
        })

    def usable(revision: dict) -> bool:
        return bool(revision["zero_result_valid"]) if revision["outcome"] == "zero" else bool(revision["outputs_intact"])

    best = None
    for position in range(len(revisions) - 1, -1, -1):
        if usable(revisions[position]) and revisions[position]["processing_state"] == "current":
            best = position
            break
    if best is None:
        best = next((position for position in range(len(revisions) - 1, -1, -1) if usable(revisions[position])), None)
    return {
        "input_sha256": input_sha256,
        "current": {"model_id": current.model_id, "model_sha256": current.model_sha256,
                    "config_sha256": current.config_sha256},
        "revisions": revisions,
        "current_best": best,
        "current_best_is_current_processing": best is not None and revisions[best]["processing_state"] == "current",
    }
