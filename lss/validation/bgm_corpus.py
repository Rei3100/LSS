"""版付きBGM区間コーパスの厳格な読み込みとSourceAsset検証。"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf


SCHEMA_VERSION = 1
SEMANTIC_ROLES = frozenset({
    "dramatic_score",
    "source_music",
    "singing_score",
    "musical_sfx",
    "vocalization",
    "unresolved_music_like",
})
IDENTITY_STATES = frozenset({"known_exact", "known_related", "unknown", "unresolved"})
EXPECTED_OUTCOMES = frozenset({
    "wanted_present",
    "no_wanted_score",
    "music_not_wanted",
    "no_novel_asset",
})
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_META_KEYS = frozenset({"type", "schema_version", "corpus_id", "revision", "description"})
_CASE_KEYS = frozenset({
    "type", "case_id", "description", "source", "expected_outcome", "attributes", "occurrences",
})
_SOURCE_KEYS = frozenset({"path", "sha256", "sample_rate", "frames"})
_OCCURRENCE_KEYS = frozenset({
    "occurrence_id", "start_frame", "end_frame", "semantic_role", "wanted", "novel",
    "identity_state", "cue_identity_id", "audio_version_id",
})


class InvalidBgmCorpusError(ValueError):
    """BGM検証コーパスがschemaまたは実体同一性の契約に合わない。"""


@dataclass(frozen=True)
class ReferenceOccurrence:
    occurrence_id: str
    start_frame: int
    end_frame: int
    semantic_role: str
    wanted: bool
    novel: bool | None
    identity_state: str
    cue_identity_id: str | None = None
    audio_version_id: str | None = None


@dataclass(frozen=True)
class BgmCase:
    case_id: str
    description: str
    source_path: Path
    source_sha256: str
    sample_rate: int
    frames: int
    expected_outcome: str
    attributes: tuple[str, ...]
    occurrences: tuple[ReferenceOccurrence, ...]


@dataclass(frozen=True)
class BgmCorpus:
    corpus_id: str
    revision: str
    description: str
    cases: tuple[BgmCase, ...]
    schema_version: int = SCHEMA_VERSION


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise InvalidBgmCorpusError(f"JSONキーが重複しています: {key}")
        value[key] = item
    return value


def _read_json(line: str, *, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, InvalidBgmCorpusError) as exc:
        raise InvalidBgmCorpusError(f"{line_number}行目のJSONが不正です: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidBgmCorpusError(f"{line_number}行目はオブジェクトである必要があります")
    return value


def _reject_unknown(value: dict[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise InvalidBgmCorpusError(f"{context}に未対応のキーがあります: {', '.join(unknown)}")


def _require_id(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise InvalidBgmCorpusError(f"{context}が不正です: {value!r}")
    return value


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidBgmCorpusError(f"{context}が未指定です")
    return value.strip()


def _require_positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InvalidBgmCorpusError(f"{context}は正の整数である必要があります")
    return value


def _optional_id(value: Any, context: str) -> str | None:
    return None if value is None else _require_id(value, context)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_occurrence(raw: Any, *, case_id: str, frames: int) -> ReferenceOccurrence:
    if not isinstance(raw, dict):
        raise InvalidBgmCorpusError(f"case {case_id}のoccurrenceはオブジェクトである必要があります")
    context = f"case {case_id}のoccurrence"
    _reject_unknown(raw, _OCCURRENCE_KEYS, context)
    occurrence_id = _require_id(raw.get("occurrence_id"), f"{context}.occurrence_id")
    start = raw.get("start_frame")
    end = raw.get("end_frame")
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise InvalidBgmCorpusError(f"{occurrence_id}.start_frameは0以上の整数である必要があります")
    if not isinstance(end, int) or isinstance(end, bool) or end <= start or end > frames:
        raise InvalidBgmCorpusError(
            f"{occurrence_id}の区間がsource範囲外です: start={start}, end={end}, frames={frames}"
        )
    role = raw.get("semantic_role")
    if role not in SEMANTIC_ROLES:
        raise InvalidBgmCorpusError(f"{occurrence_id}.semantic_roleが不正です: {role!r}")
    wanted = raw.get("wanted")
    if not isinstance(wanted, bool):
        raise InvalidBgmCorpusError(f"{occurrence_id}.wantedはboolである必要があります")
    novel = raw.get("novel")
    if novel is not None and not isinstance(novel, bool):
        raise InvalidBgmCorpusError(f"{occurrence_id}.novelはboolまたはnullである必要があります")
    identity_state = raw.get("identity_state")
    if identity_state not in IDENTITY_STATES:
        raise InvalidBgmCorpusError(f"{occurrence_id}.identity_stateが不正です: {identity_state!r}")
    cue_id = _optional_id(raw.get("cue_identity_id"), f"{occurrence_id}.cue_identity_id")
    version_id = _optional_id(raw.get("audio_version_id"), f"{occurrence_id}.audio_version_id")
    if identity_state == "known_exact" and (cue_id is None or version_id is None):
        raise InvalidBgmCorpusError(f"{occurrence_id}のknown_exactにはcue/version識別子が必要です")
    if identity_state == "known_related" and cue_id is None:
        raise InvalidBgmCorpusError(f"{occurrence_id}のknown_relatedにはcue識別子が必要です")
    if identity_state in {"unknown", "unresolved"} and (cue_id is not None or version_id is not None):
        raise InvalidBgmCorpusError(f"{occurrence_id}の未確定identityへ識別子を設定できません")
    return ReferenceOccurrence(
        occurrence_id=occurrence_id,
        start_frame=start,
        end_frame=end,
        semantic_role=role,
        wanted=wanted,
        novel=novel,
        identity_state=identity_state,
        cue_identity_id=cue_id,
        audio_version_id=version_id,
    )


def _validate_outcome(case_id: str, outcome: str, occurrences: tuple[ReferenceOccurrence, ...]) -> None:
    wanted = [item for item in occurrences if item.wanted]
    if outcome == "wanted_present" and not wanted:
        raise InvalidBgmCorpusError(f"case {case_id}はwanted_presentですがwanted区間がありません")
    if outcome == "no_wanted_score" and wanted:
        raise InvalidBgmCorpusError(f"case {case_id}のno_wanted_scoreにwanted区間があります")
    if outcome == "music_not_wanted" and (wanted or not occurrences):
        raise InvalidBgmCorpusError(
            f"case {case_id}のmusic_not_wantedにはunwanted区間だけが必要です"
        )
    if outcome == "no_novel_asset":
        if not wanted or any(item.novel is not False for item in wanted):
            raise InvalidBgmCorpusError(
                f"case {case_id}のno_novel_assetには既知のwanted区間が必要です"
            )


def _load_case(raw: dict[str, Any], *, manifest_dir: Path, verify_sources: bool) -> BgmCase:
    _reject_unknown(raw, _CASE_KEYS, "case")
    case_id = _require_id(raw.get("case_id"), "case_id")
    description = _require_text(raw.get("description"), f"case {case_id}.description")
    source = raw.get("source")
    if not isinstance(source, dict):
        raise InvalidBgmCorpusError(f"case {case_id}.sourceはオブジェクトである必要があります")
    _reject_unknown(source, _SOURCE_KEYS, f"case {case_id}.source")
    raw_path = _require_text(source.get("path"), f"case {case_id}.source.path")
    source_path = Path(raw_path)
    if not source_path.is_absolute():
        source_path = manifest_dir / source_path
    source_path = source_path.resolve()
    source_sha256 = source.get("sha256")
    if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
        raise InvalidBgmCorpusError(f"case {case_id}.source.sha256が不正です")
    sample_rate = _require_positive_int(source.get("sample_rate"), f"case {case_id}.source.sample_rate")
    frames = _require_positive_int(source.get("frames"), f"case {case_id}.source.frames")
    outcome = raw.get("expected_outcome")
    if outcome not in EXPECTED_OUTCOMES:
        raise InvalidBgmCorpusError(f"case {case_id}.expected_outcomeが不正です: {outcome!r}")
    attributes = raw.get("attributes", [])
    if (
        not isinstance(attributes, list)
        or not all(isinstance(item, str) and item.strip() for item in attributes)
        or len(set(attributes)) != len(attributes)
    ):
        raise InvalidBgmCorpusError(f"case {case_id}.attributesが不正です")
    raw_occurrences = raw.get("occurrences")
    if not isinstance(raw_occurrences, list):
        raise InvalidBgmCorpusError(f"case {case_id}.occurrencesは配列である必要があります")
    occurrences = tuple(
        _load_occurrence(item, case_id=case_id, frames=frames) for item in raw_occurrences
    )
    ids = [item.occurrence_id for item in occurrences]
    if len(ids) != len(set(ids)):
        raise InvalidBgmCorpusError(f"case {case_id}のoccurrence_idが重複しています")
    _validate_outcome(case_id, outcome, occurrences)

    if verify_sources:
        if not source_path.is_file():
            raise InvalidBgmCorpusError(f"case {case_id}のsourceが存在しません: {source_path}")
        try:
            info = sf.info(source_path)
        except (OSError, RuntimeError) as exc:
            raise InvalidBgmCorpusError(f"case {case_id}のsource音声を読めません: {source_path}") from exc
        if info.samplerate != sample_rate or info.frames != frames:
            raise InvalidBgmCorpusError(
                f"case {case_id}のsource layoutがmanifestと一致しません: "
                f"sample_rate={info.samplerate}/{sample_rate}, frames={info.frames}/{frames}"
            )
        if _file_sha256(source_path) != source_sha256:
            raise InvalidBgmCorpusError(f"case {case_id}のsource SHA-256が一致しません")

    return BgmCase(
        case_id=case_id,
        description=description,
        source_path=source_path,
        source_sha256=source_sha256,
        sample_rate=sample_rate,
        frames=frames,
        expected_outcome=outcome,
        attributes=tuple(attributes),
        occurrences=occurrences,
    )


def load_bgm_corpus(path: Path, *, verify_sources: bool = True) -> BgmCorpus:
    """JSONL manifestを読み、schemaと必要なら音声実体を厳格に検証する。"""
    path = Path(path)
    try:
        lines = [
            (number, line)
            for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
            if (line := raw.strip())
        ]
    except (OSError, UnicodeError) as exc:
        raise InvalidBgmCorpusError(f"BGM corpus manifestを読めません: {path}") from exc
    if not lines:
        raise InvalidBgmCorpusError("BGM corpus manifestが空です")

    meta = _read_json(lines[0][1], line_number=lines[0][0])
    _reject_unknown(meta, _META_KEYS, "manifest metadata")
    if meta.get("type") != "manifest":
        raise InvalidBgmCorpusError("先頭行はtype=manifestである必要があります")
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise InvalidBgmCorpusError(
            f"未対応のBGM corpus schemaです: {meta.get('schema_version')}"
        )
    corpus_id = _require_id(meta.get("corpus_id"), "corpus_id")
    revision = _require_text(meta.get("revision"), "revision")
    description = _require_text(meta.get("description"), "description")

    cases: list[BgmCase] = []
    for number, line in lines[1:]:
        raw = _read_json(line, line_number=number)
        if raw.get("type") != "case":
            raise InvalidBgmCorpusError(f"{number}行目はtype=caseである必要があります")
        cases.append(_load_case(raw, manifest_dir=path.parent, verify_sources=verify_sources))
    if not cases:
        raise InvalidBgmCorpusError("BGM corpusにcaseがありません")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise InvalidBgmCorpusError("case_idが重複しています")
    return BgmCorpus(
        corpus_id=corpus_id,
        revision=revision,
        description=description,
        cases=tuple(cases),
    )
