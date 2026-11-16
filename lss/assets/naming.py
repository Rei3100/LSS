"""BGMの表示名だけを改訂する。音声・cue同一性・CurrentBest・元来歴は変更しない。"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from lss.core.session import _write_json_atomic

SCHEMA_VERSION = 1
POLICY_ID = "local-bgm-provisional-title-v1"
KINDS = {"official", "provisional_related", "provisional_descriptive", "provisional_fallback"}


class NamingError(ValueError):
    """不正な証拠や壊れた履歴を名称未詳に読み替えない。"""


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise NamingError("名称と根拠は空でない200文字以内の文字列が必要です")
    if any(ord(char) < 32 for char in value):
        raise NamingError("名称に制御文字は使えません")
    return value.strip()


def _verified(item: dict, asset_id: str) -> bool:
    # 単なるreference labelやepisode titleは公式曲名の裏付けではない。
    return (item.get("verified") is True and item.get("asset_id") == asset_id
            and isinstance(item.get("evidence_ref"), str) and bool(item["evidence_ref"].strip()))


@contextmanager
def _locked(path: Path):
    """OSがprocess終了時にも解放するlock。lock fileの削除によるwriter間raceを作らない。"""
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise NamingError(f"命名記録は別のwriterが書込中です: {path}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _strength(proposal: dict) -> int:
    return {"insufficient_descriptive_evidence": 0, "local_acoustic_measurement": 1,
            "local_cue_context": 2, "verified_related_cue": 3,
            "verified_official_exact_version": 4}[proposal["evidence"][0]["signal"]]


def propose(asset_id: str, evidence: dict, *, policy_id: str = POLICY_ID) -> dict:
    """入力順・場所・処理日時に依存しない提案。公式の不確かな情報は採用しない。"""
    asset_id = _text(asset_id)
    policy_id = _text(policy_id)
    if not isinstance(evidence, dict):
        raise NamingError("命名証拠はobjectが必要です")
    for field in ("official", "related", "context", "acoustic"):
        if field in evidence and not isinstance(evidence[field], dict):
            raise NamingError(f"{field}はobjectが必要です")
    token = hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:10]
    basis: list[dict] = []
    official = evidence.get("official") or {}
    related = evidence.get("related") or {}
    context = evidence.get("context") or {}
    acoustic = evidence.get("acoustic") or {}
    if _verified(official, asset_id) and official.get("relation") == "exact_version":
        title, kind = _text(official.get("title")), "official"
        basis = [{"signal": "verified_official_exact_version", **official}]
    elif (_verified(related, asset_id)
          and related.get("relation") in ("same_cue_different_version", "same_theme")):
        title, kind = f"{_text(related.get('title'))} — 関連版", "provisional_related"
        basis = [{"signal": "verified_related_cue", **related}]
    elif (context.get("asset_id") == asset_id and context.get("scope") == "cue"
          and context.get("evidence_ref") and context.get("description")):
        title, kind = _text(context["description"]), "provisional_descriptive"
        basis = [{"signal": "local_cue_context", **context}]
    elif acoustic.get("status") == "measured" and acoustic.get("asset_id") == asset_id:
        colour = {"low": "低域が厚い響き", "high": "高域が際立つ響き",
                  "middle": "中域が際立つ響き", "distributed": "帯域に広がる響き"}
        motion = {"steady": "なだらかな強弱", "varying": "起伏のある強弱"}
        if acoustic.get("colour") not in colour or acoustic.get("motion") not in motion:
            raise NamingError("未対応の音響記述です")
        title = f"{colour[acoustic['colour']]} — {motion[acoustic['motion']]}"
        kind = "provisional_descriptive"
        basis = [{"signal": "local_acoustic_measurement", **acoustic}]
    else:
        title, kind = "名称未詳の劇伴", "provisional_fallback"
        basis = [{"signal": "insufficient_descriptive_evidence"}]
    provisional = kind != "official"
    display = f"{title}（仮題・{token}）" if provisional else title
    body = {"policy_id": policy_id, "title": title, "display_title": display,
            "kind": kind, "is_official": not provisional, "evidence": basis,
            "input_evidence": evidence, "evidence_sha256": digest(evidence)}
    return {**body, "proposal_id": digest(body)}


class NamingStore:
    """1 assetにつき原子的JSON。履歴を残し、同時writerは上書きせず拒否する。"""

    def __init__(self, root: Path):
        self.root = Path(root)

    def path_for(self, asset_id: str) -> Path:
        return self.root / f"{digest(_text(asset_id))}.json"

    def read(self, asset_id: str) -> dict | None:
        path = self.path_for(asset_id)
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if (type(state["schema_version"]) is not int or state["schema_version"] != SCHEMA_VERSION
                    or state["type"] != "bgm_presentation_naming" or state["asset_id"] != asset_id
                    or not isinstance(state["revisions"], list) or not state["revisions"]
                    or digest(state["identity"]) != state["identity_sha256"]):
                raise ValueError("schema/asset")
            previous = None
            for revision in state["revisions"]:
                body = {k: v for k, v in revision.items() if k != "revision_id"}
                if (revision["previous_revision_id"] != previous or digest(body) != revision["revision_id"]
                        or revision["identity_sha256"] != state["identity_sha256"]
                        or digest(revision["input_evidence"]) != revision["evidence_sha256"]
                        or revision["kind"] not in KINDS
                        or revision["is_official"] != (revision["kind"] == "official")):
                    raise ValueError("revision")
                previous = revision["revision_id"]
            if state["current_revision_id"] != previous:
                raise ValueError("current")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise NamingError(f"命名履歴を読み込めません: {path}") from exc
        return state

    def update(self, asset_id: str, evidence: dict, *, identity: dict,
               artifact: dict, policy_id: str = POLICY_ID) -> dict:
        """title変更でidentityを変更しない。Artifact hashは名称の入力証拠であってIDではない。"""
        if not isinstance(identity, dict) or not isinstance(artifact, dict):
            raise NamingError("同一性とartifactの記録はobjectが必要です")
        proposed = propose(asset_id, evidence, policy_id=policy_id)
        path = self.path_for(asset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(".lock")
        with _locked(lock):
            state = self.read(asset_id)
            if state is None:
                state = {"type": "bgm_presentation_naming", "schema_version": SCHEMA_VERSION,
                         "asset_id": asset_id, "identity": identity, "identity_sha256": digest(identity),
                         "revisions": [],
                         "current_revision_id": None}
            elif state["identity"] != identity:
                raise NamingError("表示名の更新でasset/CueIdentity/CurrentBestを変更できません")
            current = state["revisions"][-1] if state["revisions"] else None
            # 再実行による公式情報の欠落や弱い証拠は既知の強い表示を消さない。
            if current and (current["proposal_id"] == proposed["proposal_id"]
                            or _strength(current) > _strength(proposed)):
                return state
            revision = {**proposed, "artifact": artifact,
                        "identity_sha256": state["identity_sha256"],
                        "previous_revision_id": state["current_revision_id"]}
            revision["revision_id"] = digest(revision)
            state["revisions"].append(revision)
            state["current_revision_id"] = revision["revision_id"]
            _write_json_atomic(path, state)
            return state
