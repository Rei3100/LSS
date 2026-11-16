"""観測された属性から、LSSの採用policyとして意味的roleを導く。

`docs/BGM.md` の「Acoustic sourceとsemantic targetを分ける」と
`docs/QUALITY_AND_VALIDATION.md` のL4に対応する。annotationの事実と、LSSがどの役割として
扱うかの採用policyを分ける。ここが返すのは後者であり、観測そのものではない。

役割を人手のラベルから受け取らない。LSSは観測から自分で決め、決められないときは
`unresolved_music_like` のまま保持する。曖昧さを消すために確からしさを作らない。

policyは版を持ち、結果を見る前に宣言する。区別できない役割は隠さず宣言へ含める。
宣言に無い観測は使わない。観測が無いことを「役割が無い」ことにしない。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from lss.bgm.supported_scope import SINGING_OBSERVATIONS, UNRESOLVED_ROLES
from lss.validation.bgm_corpus import SEMANTIC_ROLES

SCHEMA_VERSION = 1
METHOD = "observed-attributes-role-policy-v1"
REASONS = (
    "observed_singing",
    "no_observed_singing",
    "singing_observation_unresolved",
    "singing_not_observed",
)


class RolePolicyError(ValueError):
    """採用policyの宣言、または渡された観測が契約に合わない。"""


@dataclass(frozen=True)
class RolePolicy:
    """観測された歌唱の有無から役割を当てる採用policy。

    `indistinguishable_roles` は、この観測では出力する役割と区別できない役割である。
    区別できないことを宣言に含め、後段が誤って強い主張をしないようにする。
    """

    singing_present_role: str
    singing_absent_role: str
    unresolved_role: str
    indistinguishable_roles: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise RolePolicyError("未対応の採用policy schemaです")
        for field in ("singing_present_role", "singing_absent_role"):
            value = getattr(self, field)
            if value not in SEMANTIC_ROLES or value in UNRESOLVED_ROLES:
                raise RolePolicyError(f"{field}は解決済みの役割である必要があります")
        if self.unresolved_role not in UNRESOLVED_ROLES:
            raise RolePolicyError("unresolved_roleは未解決の役割である必要があります")
        roles = self.indistinguishable_roles
        if (not isinstance(roles, (list, tuple)) or not roles
                or len(set(roles)) != len(roles)
                or not set(roles) <= set(SEMANTIC_ROLES)
                or set(roles) & UNRESOLVED_ROLES):
            raise RolePolicyError(
                "indistinguishable_rolesは重複のない解決済みの役割の並びである必要があります")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["indistinguishable_roles"] = list(self.indistinguishable_roles)
        return payload

    @property
    def policy_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-role-policy-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def decide_role(observations: Any, policy: RolePolicy) -> dict:
    """記録済みの観測だけから役割を導く。観測が無ければ未解決のまま返す。"""
    policy.validate()
    if observations is None:
        observations = {}
    if not isinstance(observations, dict):
        raise RolePolicyError("観測は辞書である必要があります")
    singing = observations.get("singing")
    if singing is not None and singing not in SINGING_OBSERVATIONS:
        raise RolePolicyError(f"歌唱の観測が不正です: {singing!r}")

    if singing is None:
        role, reason = policy.unresolved_role, "singing_not_observed"
    elif singing == "unresolved":
        role, reason = policy.unresolved_role, "singing_observation_unresolved"
    elif singing == "present":
        role, reason = policy.singing_present_role, "observed_singing"
    else:
        role, reason = policy.singing_absent_role, "no_observed_singing"
    return {
        "semantic_role": role, "reason": reason, "policy_id": policy.policy_id,
        # 観測ではなく採用policyの結論である。annotationの事実と混同しない。
        "derived_by_policy": True,
        "observations_used": {"singing": singing},
        "indistinguishable_roles": list(policy.indistinguishable_roles),
    }


def decide_roles(observations: Any, policy: RolePolicy) -> dict[str, dict]:
    """候補IDごとの観測から、候補IDごとの役割を導く。"""
    policy.validate()
    if not isinstance(observations, dict):
        raise RolePolicyError("候補IDごとの観測が必要です")
    decided = {}
    for candidate_id, measured in observations.items():
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise RolePolicyError("候補IDが必要です")
        decided[candidate_id] = decide_role(measured, policy)
    return decided
