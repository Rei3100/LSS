"""最初の対応領域を先に固定し、cueごとの公開・保留・未対応・正当なゼロを決める。

`docs/BGM.md` の「最初のsupported scope」「正常な対象なし結果」に対応する判断だけを持つ。
音を作らず、ファイルを書かず、根拠も測らない。既に記録された根拠から結論だけを出す。

固定していないscopeでは何も公開できない。判断に必要な宣言が一つでも欠けていれば拒否し、
公開しない理由は必ず名前で返す。除外したroleは黙って捨てず、`UNSUPPORTED`として残す。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real

from lss.validation.bgm_corpus import IDENTITY_STATES, SEMANTIC_ROLES
from lss.validation.occurrence_metrics import WANTED_BGM_ROLES

SCHEMA_VERSION = 1
# v2: 登録済みの参照録音と一致した候補（新しいassetにならない）と、方針で対象外にした音楽（主題歌の形）を
# 名前で分ける。対応領域の宣言の項目は同じ。
# v3: 実行時に観測できる安全の保留（`SAFETY_OBSERVATIONS`）を、対応領域が名前で宣言できる。
# v4: 公開を止めうる安全の観測を音の中身の観測だけにする（番組の役割のラベルは宣言できない）。編集境界に
# 挟まれた挿入の区画は、役割未確定の保留（``program_insert_policy``）として別に扱い、対象外にはしない。
METHOD = "first-supported-bgm-scope-v4"
UNRESOLVED_ROLES = frozenset({"unresolved", "unresolved_music_like"})
DISPOSITIONS = frozenset({"PUBLISHED", "HELD", "UNSUPPORTED"})
# 観測された歌唱の扱い。観測は役割ではないので、roleを与えず対応範囲だけを決める。
SINGING_POLICIES = ("ignore", "exclude_from_publication")
SINGING_OBSERVATIONS = ("absent", "present", "unresolved")
# 観測から導いた役割の扱い。policyが区別できないと宣言した役割を、公開の場で許すかどうか。
DERIVED_ROLE_POLICIES = ("accept_declared_ambiguity", "require_unambiguous_role")
# 候補が一曲かどうか。複数曲と分かった候補は方針によらず公開しない。
CUE_EXTENT_POLICIES = ("allow_unresolved_cue_extent", "require_single_cue_evidence")
CUE_EXTENT_STATES = ("single_cue", "multiple_cues", "unresolved")
# 登録済みの参照録音との一致。全体なら既発売の録音そのもの、一部なら一曲かどうかも既発売かどうかも決まらない。
KNOWN_RELEASE_STATES = ("exact", "partial")
# 欲しい劇伴にならない保留の理由。全てこれなら「新しいassetが無い」正当なゼロ。
NO_NOVEL_ASSET_REASONS = frozenset({"duplicate_of_published_cue", "known_released_recording"})
# 欲しい役割ではない音楽の理由。全てこれなら「音楽はあるが欲しい役割ではない」正当なゼロ。
NOT_WANTED_REASONS = frozenset({"semantic_role_not_wanted_bgm", "excluded_non_target_music"})
# 公開を止める安全の観測（観測は役割ではない。真ならHELDにして理由を名前で残す）。測定は `lss.bgm.cue_scope`。
# 音の中身の観測だけを置く。番組の端・放送挿入のタグ・番組の中ほど・境界の付近などの番組の役割は文脈であり、
# それだけで聞こえる劇伴を止めない（v12の保留がこれで本編の劇伴を止めていた。bgm-product-repair-r3）。
# - heavy_nonmusic_event_scene: 原本に非音楽のイベント音（効果音・動物・機械等）が密な場面。重い効果音の残りと
#   劇伴の削れが集中し、今の音声の手掛かりでは安全な部分だけを取り出せない
SAFETY_OBSERVATIONS = ("heavy_nonmusic_event_scene",)
# 編集境界に挟まれた挿入の区画（CM・告知・アイキャッチ・予告・番組内コーナー等。実行時に区別できない）の扱い。
# - publish_by_content: 区画であることを判断に使わない（音の中身だけで決める）
# - hold_role_unresolved: 役割未確定としてHELD（対象外にはしない。音は残る）
PROGRAM_INSERT_POLICIES = ("publish_by_content", "hold_role_unresolved")


class ScopeError(ValueError):
    """対応領域の宣言、または判断に渡された根拠が契約に合わない。"""


def _texts(value, name: str, *, allowed=None) -> tuple[str, ...]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or len(set(value)) != len(value)):
        raise ScopeError(f"{name}は重複のない非空の文字列の並びである必要があります")
    if allowed is not None and not set(value) <= set(allowed):
        raise ScopeError(f"{name}に未知の値があります: {sorted(set(value) - set(allowed))}")
    return tuple(value)


def _seconds(value, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 < float(value) < 86400:
        raise ScopeError(f"{name}は0より大きい有限の秒数である必要があります")
    return float(value)


@dataclass(frozen=True)
class SupportedScope:
    """最終評価より前に固定する対応領域。判断を持つ項目に既定値を置かない。"""

    source_domains: tuple[str, ...]
    published_semantic_roles: tuple[str, ...]
    covered_semantic_roles: tuple[str, ...]
    minimum_cue_seconds: float
    maximum_cue_seconds: float | None
    required_evidence: tuple[str, ...]
    quality_gates: dict
    singing_observation_policy: str
    derived_role_policy: str
    cue_extent_policy: str
    publication_allowed: bool
    safety_holds: tuple[str, ...] = ()
    program_insert_policy: str = "publish_by_content"
    short_cue_minimum_seconds: float | None = None
    short_cue_required_evidence: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def to_dict(self) -> dict:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise ScopeError("未対応の対応領域schemaです")
        if not isinstance(self.publication_allowed, bool):
            raise ScopeError("publication_allowedは真偽値である必要があります")
        if self.singing_observation_policy not in SINGING_POLICIES:
            raise ScopeError(f"singing_observation_policyは{list(SINGING_POLICIES)}のいずれかです")
        if self.derived_role_policy not in DERIVED_ROLE_POLICIES:
            raise ScopeError(f"derived_role_policyは{list(DERIVED_ROLE_POLICIES)}のいずれかです")
        if self.cue_extent_policy not in CUE_EXTENT_POLICIES:
            raise ScopeError(f"cue_extent_policyは{list(CUE_EXTENT_POLICIES)}のいずれかです")
        if self.program_insert_policy not in PROGRAM_INSERT_POLICIES:
            raise ScopeError(f"program_insert_policyは{list(PROGRAM_INSERT_POLICIES)}のいずれかです")
        published = _texts(self.published_semantic_roles, "published_semantic_roles", allowed=SEMANTIC_ROLES)
        covered = _texts(self.covered_semantic_roles, "covered_semantic_roles", allowed=SEMANTIC_ROLES)
        if set(published) & UNRESOLVED_ROLES:
            raise ScopeError("未解決のroleを公開対象にはできません")
        if not set(published) <= set(covered):
            raise ScopeError("公開対象のroleは対応領域に含まれている必要があります")
        minimum = _seconds(self.minimum_cue_seconds, "minimum_cue_seconds")
        short_minimum = _seconds(self.short_cue_minimum_seconds, "short_cue_minimum_seconds", allow_none=True)
        if short_minimum is not None:
            if not 0 < short_minimum < minimum or not self.short_cue_required_evidence:
                raise ScopeError("短いcueの下限は通常の下限未満で、専用の根拠が必要です")
            _texts(self.short_cue_required_evidence, "short_cue_required_evidence")
        elif self.short_cue_required_evidence:
            raise ScopeError("短いcueの根拠には短いcueの下限が必要です")
        maximum = _seconds(self.maximum_cue_seconds, "maximum_cue_seconds", allow_none=True)
        if maximum is not None and maximum < minimum:
            raise ScopeError("cue長の上限が下限より小さくなっています")
        if not isinstance(self.quality_gates, dict) or not self.quality_gates or any(
                not isinstance(key, str) or not key.strip() or isinstance(value, bool)
                or not isinstance(value, Real) or not float("-inf") < float(value) < float("inf")
                for key, value in self.quality_gates.items()):
            raise ScopeError("quality_gatesは指標名と有限の下限値の対応が必要です")
        holds = tuple(self.safety_holds)
        if (len(set(holds)) != len(holds) or not set(holds) <= set(SAFETY_OBSERVATIONS)
                or any(not isinstance(item, str) for item in holds)):
            raise ScopeError(f"safety_holdsは{list(SAFETY_OBSERVATIONS)}の重複のない並びです")
        payload = asdict(self)
        payload.update(source_domains=list(_texts(self.source_domains, "source_domains")),
                       safety_holds=sorted(holds),
                       published_semantic_roles=list(published),
                       covered_semantic_roles=list(covered),
                       required_evidence=list(_texts(self.required_evidence, "required_evidence")),
                       minimum_cue_seconds=minimum, maximum_cue_seconds=maximum,
                       quality_gates={key: float(value) for key, value in sorted(self.quality_gates.items())})
        if short_minimum is None:
            # Preserve historical scope identities when the optional route is off.
            payload.pop("short_cue_minimum_seconds")
            payload.pop("short_cue_required_evidence")
        else:
            payload["short_cue_minimum_seconds"] = short_minimum
            payload["short_cue_required_evidence"] = list(self.short_cue_required_evidence)
        return payload

    @property
    def scope_id(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-supported-scope-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _cue_seconds(candidate: dict) -> float:
    rate = candidate.get("sample_rate")
    start, end = candidate.get("start_frame"), candidate.get("end_frame")
    if (type(rate) is not int or rate <= 0 or type(start) is not int or type(end) is not int
            or start < 0 or end <= start):
        raise ScopeError("cueの時間軸が不正です")
    return (end - start) / rate


def classify_candidate(candidate: dict, scope: SupportedScope) -> dict:
    """cue一つの公開可否と、公開しない場合の理由を返す。音も記録も変えない。"""
    scope.to_dict()
    if not isinstance(candidate, dict) or not isinstance(candidate.get("candidate_id"), str):
        raise ScopeError("cue候補にはcandidate_idが必要です")
    seconds = _cue_seconds(candidate)
    identity_state = candidate.get("identity_state")
    if identity_state not in IDENTITY_STATES:
        raise ScopeError(f"cueのidentity_stateが不正です: {identity_state!r}")
    role = candidate.get("semantic_role")
    if not isinstance(role, str) or (role not in SEMANTIC_ROLES and role not in UNRESOLVED_ROLES):
        raise ScopeError(f"cueのsemantic_roleが不正です: {role!r}")

    def result(disposition: str, reason: str, **extra) -> dict:
        return {"candidate_id": candidate["candidate_id"], "disposition": disposition,
                "reason": reason, "scope_id": scope.scope_id, "cue_seconds": seconds,
                "semantic_role": role, "identity_state": identity_state, **extra}

    if candidate.get("source_domain") not in scope.source_domains:
        return result("UNSUPPORTED", "source_domain_out_of_scope")
    excluded = candidate.get("excluded_non_target")
    if excluded is not None:
        if not isinstance(excluded, str) or not excluded.strip():
            raise ScopeError(f"対象外にした理由が不正です: {excluded!r}")
        # 方針で欲しい劇伴にしない音楽（主題歌の形など）。捨てずに理由を残す。
        return result("UNSUPPORTED", "excluded_non_target_music", excluded_non_target=excluded)
    known = candidate.get("known_release")
    if known is not None and known not in KNOWN_RELEASE_STATES:
        raise ScopeError(f"参照録音との一致の状態が不正です: {known!r}")
    if known == "exact":
        # 既発売の録音そのもの。新しいassetとしては公開しない。
        return result("HELD", "known_released_recording")
    observations = candidate.get("observations") or {}
    if not isinstance(observations, dict):
        raise ScopeError("cueの観測は辞書である必要があります")
    singing = observations.get("singing")
    if singing is not None and singing not in SINGING_OBSERVATIONS:
        raise ScopeError(f"歌唱の観測が不正です: {singing!r}")
    if singing == "present" and scope.singing_observation_policy == "exclude_from_publication":
        # 観測はroleではない。役割を与えず、対応範囲の外として理由を残す。
        return result("UNSUPPORTED", "observed_singing_out_of_scope", observed_singing=singing)
    if role in UNRESOLVED_ROLES:
        return result("HELD", "semantic_role_unresolved")
    if role not in WANTED_BGM_ROLES:
        # 音楽ではあるが欲しい劇伴ではない。捨てずに理由を残す。
        return result("UNSUPPORTED", "semantic_role_not_wanted_bgm")
    if role not in scope.covered_semantic_roles:
        return result("UNSUPPORTED", "semantic_role_out_of_scope")
    if role not in scope.published_semantic_roles:
        return result("UNSUPPORTED", "semantic_role_covered_but_not_published")
    derivation = candidate.get("semantic_role_derivation")
    if derivation is not None and not isinstance(derivation, dict):
        raise ScopeError(f"役割の導出は辞書である必要があります: {derivation!r}")
    if derivation is not None and scope.derived_role_policy == "require_unambiguous_role":
        ambiguous = derivation.get("indistinguishable_roles") or []
        if not isinstance(ambiguous, (list, tuple)) or any(
                not isinstance(item, str) for item in ambiguous):
            raise ScopeError("区別できない役割の宣言が不正です")
        risky = sorted(set(ambiguous) - set(scope.published_semantic_roles))
        if risky:
            # 導いた役割が、公開しない役割と区別できないと宣言されている。捨てずに保留する。
            return result("HELD", "derived_role_cannot_exclude_other_roles",
                          ambiguous_roles=risky,
                          role_policy_id=derivation.get("policy_id"))
    if scope.safety_holds:
        safety = candidate.get("safety")
        if not isinstance(safety, dict):
            # 宣言した安全の観測が無い候補は、測れていないので公開しない。
            return result("HELD", "safety_not_observed", safety_holds=list(scope.safety_holds))
        unknown = [name for name in scope.safety_holds if not isinstance(safety.get(name), bool)]
        if unknown:
            return result("HELD", "safety_not_observed", safety_holds=unknown)
        holds = [name for name in scope.safety_holds if safety[name]]
        if holds:
            # 安全に公開できると言えない観測がある。候補は捨てず、理由を付けて保留する。
            return result("HELD", "safety_hold", safety_holds=holds)
    insert = candidate.get("program_insert")
    if insert is not None and not isinstance(insert, bool):
        raise ScopeError(f"挿入の区画の観測が不正です: {insert!r}")
    if insert and scope.program_insert_policy == "hold_role_unresolved":
        # 編集境界に挟まれた挿入の区画。CMか番組の挿入（サブタイトル・予告等）か決まらない。対象外とは言わない。
        return result("HELD", "program_insert_role_unresolved")
    if known == "partial":
        # 一部だけ既発売の録音と一致する。一曲かどうかも、新しい部分があるかも決まらない。
        return result("HELD", "partly_known_released_recording")
    extent = candidate.get("cue_extent")
    if extent is not None and (not isinstance(extent, dict)
                               or extent.get("state") not in CUE_EXTENT_STATES):
        raise ScopeError(f"cueの範囲の状態が不正です: {extent!r}")
    extent_state = (extent or {}).get("state", "unresolved")
    if extent_state == "multiple_cues":
        # 複数曲を含むと分かっている以上、一曲のcueとしては公開できない。捨てずに理由を残す。
        return result("HELD", "contains_more_than_one_cue", cue_extent_state=extent_state)
    if scope.cue_extent_policy == "require_single_cue_evidence" and extent_state != "single_cue":
        return result("HELD", "cue_extent_unresolved", cue_extent_state=extent_state)
    short = seconds < scope.minimum_cue_seconds
    if short and (scope.short_cue_minimum_seconds is None or seconds < scope.short_cue_minimum_seconds):
        return result("HELD", "below_minimum_cue_usefulness")
    if short and (not isinstance(candidate.get("evidence"), dict) or any(
            candidate["evidence"].get(key) in (None, "", [], {}) for key in scope.short_cue_required_evidence)):
        return result("HELD", "below_minimum_cue_usefulness",
                      missing_evidence=list(scope.short_cue_required_evidence))
    if scope.maximum_cue_seconds is not None and seconds > scope.maximum_cue_seconds:
        return result("HELD", "above_maximum_cue_length")
    evidence = candidate.get("evidence")
    if not isinstance(evidence, dict):
        raise ScopeError("cueの根拠は辞書である必要があります")
    required = tuple(scope.required_evidence) + (tuple(scope.short_cue_required_evidence) if short else ())
    missing = [key for key in required if evidence.get(key) in (None, "", [], {})]
    if missing:
        return result("HELD", "required_evidence_missing", missing_evidence=missing)
    measured = candidate.get("quality")
    if not isinstance(measured, dict):
        return result("HELD", "quality_not_measured")
    below = sorted(name for name, limit in scope.quality_gates.items()
                   if isinstance(measured.get(name), bool) or not isinstance(measured.get(name), Real)
                   or float(measured[name]) < limit)
    if below:
        return result("HELD", "below_quality_gate", failed_gates=below)
    if candidate.get("duplicate_of_published_cue_id") is not None:
        return result("HELD", "duplicate_of_published_cue")
    group = candidate.get("occurrence_group")
    if group is not None:
        if not isinstance(group, dict) or not isinstance(group.get("members"), (list, tuple)):
            raise ScopeError(f"使用箇所のまとまりが不正です: {group!r}")
        others = sorted(set(group["members"]) - {candidate["candidate_id"]})
        chosen = group.get("chosen_candidate_id")
        if others and chosen is None:
            # 同じ録音の使用箇所が複数ある。どれを公開するか決まるまで重複公開しない。
            return result("HELD", "occurrence_choice_not_made",
                          occurrence_group_id=group.get("group_id"),
                          other_occurrences=others)
        if others and chosen != candidate["candidate_id"]:
            return result("HELD", "another_occurrence_is_chosen",
                          occurrence_group_id=group.get("group_id"),
                          chosen_candidate_id=chosen)
    if not scope.publication_allowed:
        # Development用に固定したscopeからは、条件を満たしても公開しない。
        return result("HELD", "publication_not_allowed_by_scope")
    return result("PUBLISHED", "within_supported_scope")


def classify_result(candidates: list[dict], scope: SupportedScope, *, music_activity_found: bool) -> dict:
    """1本の処理結果をまとめる。候補ゼロ・全保留・全未対応を区別して返す。"""
    if not isinstance(music_activity_found, bool):
        raise ScopeError("music_activity_foundは真偽値である必要があります")
    rows = [classify_candidate(candidate, scope) for candidate in candidates]
    identifiers = [row["candidate_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ScopeError("cue候補のcandidate_idが重複しています")
    counts = {name: sum(row["disposition"] == name for row in rows) for name in sorted(DISPOSITIONS)}
    summary = {"scope_id": scope.scope_id, "schema_version": SCHEMA_VERSION,
               "publication_allowed": scope.publication_allowed, "candidates": rows,
               "counts": counts, "music_activity_found": music_activity_found}
    if counts["PUBLISHED"]:
        return {**summary, "disposition": "PUBLISHED", "reason": "published_cues_within_scope"}
    held = [row for row in rows if row["disposition"] == "HELD"]
    if held:
        if all(row["reason"] in NO_NOVEL_ASSET_REASONS for row in held):
            # 欲しい劇伴はあるが、公開できる新しいassetがない。
            return {**summary, "disposition": "ZERO", "reason": "no_novel_asset"}
        return {**summary, "disposition": "HELD",
                "reason": "candidates_without_sufficient_evidence_or_publication"}
    if counts["UNSUPPORTED"]:
        if all(row["reason"] in NOT_WANTED_REASONS for row in rows):
            return {**summary, "disposition": "ZERO", "reason": "music_not_wanted"}
        return {**summary, "disposition": "UNSUPPORTED", "reason": "all_candidates_out_of_scope"}
    # 候補がない正当なゼロ。未検出と「音楽はあるが欲しいroleではない」を混同しない。
    reason = "music_not_wanted" if music_activity_found else "no_wanted_score"
    return {**summary, "disposition": "ZERO", "reason": reason}
