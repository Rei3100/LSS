"""記録済みの固定ケースに対して、伴奏抽出の対応範囲と限界を書き出す。

しきい値はこの道具の中に持たない。宣言したJSONを渡す。読み取りだけを行い、
音声・記録・既定を変更しない。範囲の内側だと言えることは「宣言した条件を満たした」
ことであって、独立Holdoutでの一般化や音楽的な良さの証明ではない。

    python scripts/describe_instrumental_scope.py --scope scope.json --out statement.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.harness.fetch import file_sha256
from lss.validation.instrumental_scope import InstrumentalScope, InstrumentalScopeError, describe
from scripts.validate_instrumental_controls import VARIANTS

# 作り方から決まる、入力に歌唱があるか。variant定義の歌gainがそのまま答えになる。
VOCALS_PRESENT = {name: vocal_gain > 0 for name, _, vocal_gain in VARIANTS}
DEFAULT_RESULTS = dev_state_root(ROOT) / "validation/instrumental-controls-development-r1"


def load_cases(results_dir: Path) -> list[dict]:
    cases = []
    for path in sorted(results_dir.glob("*-result.json")):
        record = json.loads(path.read_bytes())
        variant = record.get("variant")
        if variant not in VOCALS_PRESENT:
            raise InstrumentalScopeError(f"未知のvariantです: {variant!r} ({path.name})")
        cases.append({"case_id": record["case_id"], "variant": variant,
                      "vocals_present": VOCALS_PRESENT[variant], "quality": record["quality"],
                      "source": {"path": str(path), "sha256": file_sha256(path)}})
    if not cases:
        raise InstrumentalScopeError(f"記録済みのケース結果がありません: {results_dir}")
    return cases


def load_scope(path: Path) -> InstrumentalScope:
    payload = json.loads(Path(path).read_bytes())
    if not isinstance(payload, dict):
        raise InstrumentalScopeError("対応範囲の宣言はオブジェクトである必要があります")
    payload.pop("scope_id", None)
    if isinstance(payload.get("input_domains"), list):
        payload["input_domains"] = tuple(payload["input_domains"])
    try:
        scope = InstrumentalScope(**payload)
    except TypeError as exc:
        raise InstrumentalScopeError(f"対応範囲の宣言が不完全です: {exc}") from exc
    scope.to_dict()
    return scope


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", type=Path, required=True, help="対応範囲を宣言したJSON")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, help="書き出し先。省略時は標準出力のみ。")
    args = parser.parse_args()
    scope = load_scope(args.scope)
    cases = load_cases(args.results_dir)
    statement = {
        **describe([{key: value for key, value in case.items() if key != "source"}
                    for case in cases], scope),
        "role": "development_scope_statement_not_promotion",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "results_dir": str(args.results_dir.resolve()),
        "case_sources": [case["source"] for case in cases],
        "scope_declaration": {"path": str(args.scope.resolve()), "sha256": file_sha256(args.scope)},
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(statement, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: statement[key] for key in
                      ("case_count", "within_scope", "outside_scope", "reasons", "by_variant",
                       "scope_id")}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
