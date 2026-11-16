"""伴奏抽出の対応範囲と限界を、固定ケースの測定値に対して実行できる形で表す。

`ROADMAP.md` M2の「対応範囲と限界を説明できる」に対応する。診断値そのものは
`instrumental_quality.py` が測る。ここは、先に宣言した条件へ測定値を当てて、
どのケースが対応範囲の内側で、どのケースがどの理由で外側かを返すだけである。

既定のしきい値を持たない。宣言が一つでも欠ければ何も判断しない。範囲の内側だと
言えることは「宣言した条件を満たした」ことであって、音楽的な良さの証明ではない。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real

SCHEMA_VERSION = 1
METHOD = "instrumental-supported-scope-v1"
NOT_CHECKED = ("意味的な良し悪し", "独立Holdoutでの一般化", "長尺・商用マスターでの挙動",
               "全ジャンルの品質")


class InstrumentalScopeError(ValueError):
    """対応範囲の宣言、または渡された測定値が契約に合わない。"""


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise InstrumentalScopeError(f"{name}は有限の数値である必要があります")
    return float(value)


@dataclass(frozen=True)
class InstrumentalScope:
    """最終評価より前に固定する、伴奏抽出の対応条件。"""

    input_domains: tuple[str, ...]
    minimum_snr_db: float
    minimum_si_sdr_db: float
    maximum_level_loss_db: float
    maximum_vocal_transfer_db: float
    maximum_channel_gain_error_db: float
    maximum_side_balance_error_db: float
    maximum_target_absent_output_db: float
    allow_spatial_component_loss: bool
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def to_dict(self) -> dict:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise InstrumentalScopeError("未対応の対応範囲schemaです")
        if not isinstance(self.allow_spatial_component_loss, bool):
            raise InstrumentalScopeError("allow_spatial_component_lossは真偽値である必要があります")
        domains = self.input_domains
        if (not isinstance(domains, (list, tuple)) or not domains
                or any(not isinstance(item, str) or not item.strip() for item in domains)
                or len(set(domains)) != len(domains)):
            raise InstrumentalScopeError("input_domainsは重複のない非空の文字列の並びが必要です")
        for name in ("minimum_snr_db", "minimum_si_sdr_db", "maximum_level_loss_db",
                     "maximum_vocal_transfer_db", "maximum_channel_gain_error_db",
                     "maximum_side_balance_error_db", "maximum_target_absent_output_db"):
            _number(getattr(self, name), name)
        for name in ("maximum_level_loss_db", "maximum_channel_gain_error_db",
                     "maximum_side_balance_error_db"):
            if getattr(self, name) < 0:
                raise InstrumentalScopeError(f"{name}は0以上である必要があります")
        return {**asdict(self), "input_domains": list(domains)}

    @property
    def scope_id(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"instrumental-scope-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _fidelity(quality: dict) -> dict:
    fidelity = quality.get("fidelity")
    if not isinstance(fidelity, dict):
        raise InstrumentalScopeError("測定値にfidelityがありません")
    return fidelity


def evaluate(quality: dict, scope: InstrumentalScope, *, vocals_present: bool | None = None) -> dict:
    """1ケースの測定値を宣言した条件へ当てる。外れた理由は必ず名前で返す。

    `vocals_present` は入力に歌唱があったか。無い入力では歌唱残留を測れないので
    条件から外し、`not_applicable` として残す。分からない場合は測れないことを失敗とする。
    """
    if vocals_present is not None and not isinstance(vocals_present, bool):
        raise InstrumentalScopeError("vocals_presentは真偽値またはNoneである必要があります")
    scope.to_dict()
    if not isinstance(quality, dict) or "target_present" not in quality:
        raise InstrumentalScopeError("instrumental_qualityが作った測定値が必要です")
    failures = []

    def fail(check: str, observed, limit) -> None:
        failures.append({"check": check, "observed": observed, "limit": limit})

    fidelity = _fidelity(quality)
    if not quality["target_present"]:
        # 伴奏が無い入力では、出力が歌唱に対してどれだけ残るかだけを見る。
        observed = quality.get("target_absent_output_to_vocals_db")
        if observed is None or not math.isfinite(observed) or observed > scope.maximum_target_absent_output_db:
            fail("target_absent_output_to_vocals_db", observed, scope.maximum_target_absent_output_db)
        return _result(scope, failures, ["target_absent_output_to_vocals_db"], [])

    checked = []

    def check_minimum(name: str, observed, limit) -> None:
        checked.append(name)
        if observed is None or not isinstance(observed, Real) or isinstance(observed, bool) \
                or not math.isfinite(observed) or observed < limit:
            fail(name, observed, limit)

    def check_maximum(name: str, observed, limit, *, absolute: bool = False) -> None:
        checked.append(name)
        if observed is None or not isinstance(observed, Real) or isinstance(observed, bool) \
                or not math.isfinite(observed) or (abs(observed) if absolute else observed) > limit:
            fail(name, observed, limit)

    check_minimum("scale_dependent_snr_db", fidelity.get("scale_dependent_snr_db"),
                  scope.minimum_snr_db)
    check_minimum("si_sdr_db.joint", (fidelity.get("si_sdr_db") or {}).get("joint"),
                  scope.minimum_si_sdr_db)
    check_maximum("projection_gain_db.joint", (fidelity.get("projection_gain_db") or {}).get("joint"),
                  scope.maximum_level_loss_db, absolute=True)
    check_maximum("relative_channel_gain_error_db", fidelity.get("relative_channel_gain_error_db"),
                  scope.maximum_channel_gain_error_db, absolute=True)
    check_maximum("side_to_mid_balance_error_db", fidelity.get("side_to_mid_balance_error_db"),
                  scope.maximum_side_balance_error_db, absolute=True)
    fit = quality.get("component_fit")
    not_applicable = []
    if vocals_present is False:
        # 歌唱の無い入力に歌唱残留の条件を当てない。測っていないことを明示する。
        not_applicable.append("component_fit.vocals_transfer_db")
    elif not isinstance(fit, dict):
        # 歌唱の残留を測れない結果を、残留なしとして扱わない。
        checked.append("component_fit.vocals_transfer_db")
        fail("component_fit.vocals_transfer_db", None, scope.maximum_vocal_transfer_db)
    else:
        check_maximum("component_fit.vocals_transfer_db", fit.get("vocals_transfer_db"),
                      scope.maximum_vocal_transfer_db)
    lost = fidelity.get("spatial_component_lost")
    checked.append("spatial_component_lost")
    if not scope.allow_spatial_component_loss:
        if not isinstance(lost, dict) or any(bool(value) for value in lost.values()):
            fail("spatial_component_lost", lost, False)
    else:
        checked.pop()
        not_applicable.append("spatial_component_lost")
    return _result(scope, failures, checked, not_applicable)


def _result(scope: InstrumentalScope, failures: list[dict], checked: list[str],
            not_applicable: list[str]) -> dict:
    return {"schema_version": SCHEMA_VERSION, "method": METHOD, "scope_id": scope.scope_id,
            "within_scope": not failures, "failures": failures, "checked": checked,
            "not_applicable": not_applicable, "not_checked": list(NOT_CHECKED)}


def describe(cases: list[dict], scope: InstrumentalScope) -> dict:
    """固定ケース全体に対する対応範囲と限界。外側のケースは理由ごとに数える。"""
    scope.to_dict()
    rows, reasons = [], {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise InstrumentalScopeError("case_idを持つケースが必要です")
        decided = evaluate(case["quality"], scope, vocals_present=case.get("vocals_present"))
        rows.append({"case_id": case["case_id"], "variant": case.get("variant"),
                     "within_scope": decided["within_scope"],
                     "failures": [item["check"] for item in decided["failures"]],
                     "not_applicable": decided["not_applicable"],
                     "observed": {item["check"]: item["observed"] for item in decided["failures"]}})
        for item in decided["failures"]:
            reasons[item["check"]] = reasons.get(item["check"], 0) + 1
    identifiers = [row["case_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise InstrumentalScopeError("ケースIDが重複しています")
    variants = {}
    for row in rows:
        entry = variants.setdefault(row["variant"], {"cases": 0, "within_scope": 0})
        entry["cases"] += 1
        entry["within_scope"] += int(row["within_scope"])
    return {"schema_version": SCHEMA_VERSION, "method": METHOD, "scope": scope.to_dict(),
            "scope_id": scope.scope_id, "case_count": len(rows),
            "within_scope": sum(1 for row in rows if row["within_scope"]),
            "outside_scope": sum(1 for row in rows if not row["within_scope"]),
            "reasons": dict(sorted(reasons.items())), "by_variant": variants, "cases": rows,
            "not_checked": list(NOT_CHECKED)}
