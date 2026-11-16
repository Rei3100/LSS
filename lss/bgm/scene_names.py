"""Scene-based provisional song names, score order-sheet lines and LSS M numbers for published BGM cues.

Contract (AGENTS.md "Non-negotiable meaning", docs/engine/LISTENING_AND_PRESENTATION.md "Identity versus title"):
a provisional title is either a scene-grounded Japanese name (with a quote from the cue) or a context_music
interpretation inspired by available work/transcript facts and measured music. The latter never asserts a scene,
actor action, or quote. The first name LSS gives a recording is its fixed name when the recording recurs in
other episodes; a verified title or the user's own edit still wins. Naming never blocks extraction: this module
only records and projects; the models run later (``scene_models``).

Stored under ``<state>/bgm-scene-names/scenes.json`` (one file, locked writes):
- ``cues[asset_id]``: what was heard (lines with times), the order-sheet line (use + direction), its
  category, the proposed name, guards that fired, model revision, audio identity.
- ``recordings[key]``: the fixed name of one recording (``label:<N>`` of the published-cue memory, or
  ``asset:<asset_id>`` for a cue heard once), who gave it first and when.
- ``works[work_key]``: the work's M-number prefix (two Latin capitals), fixed once.

M numbers follow the score-sheet layout of real TV score orders (sections in a fixed order, contiguous numbers,
first appearance inside a section; recordings first heard more than ``SECOND_SESSION_AFTER`` episodes after the
earliest known episode of the work take 101+, like a second recording session). They are ordering labels:
adding an earlier episode may renumber; the name never changes with them.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

DIRECTORY = "bgm-scene-names"
SCHEMA_VERSION = 1
POLICY_ID = "bgm-scene-names-v4-scene-or-context-music"
OBSERVATION_POLICY_ID = "timed-occurrence-quote-or-context-music-v3"

# Order-sheet sections in the order real TV score sheets list them (番組の決まり曲 -> メインテーマ系 -> 戦闘描写 ->
# 情景描写 -> 心理描写 -> 行動描写), and the category words the naming model chooses from.
SECTIONS = ("program", "theme", "battle", "scenery", "emotion", "action")
CATEGORIES = {"日常・のどか": "scenery", "楽しい・明るい": "scenery", "神秘・不思議": "scenery",
              "コミカル": "action", "緊迫・危機": "battle", "バトル": "battle", "不穏・怪しい": "battle",
              "悲しみ・切なさ": "emotion", "感動・希望": "emotion", "決意・出発": "emotion", "本編外": "program"}
SECOND_SESSION_AFTER = 13          # one cour
THEME_MIN_EPISODES = 4             # a recording reused in this many episodes is listed with the theme group
OUTSIDE = re.compile(r"放送中|毎週|公式YouTube|チャンネル登録|配信中|次回予告|プレゼント|キャンペーン")
ORDER_RETRY_BASE_SECONDS = 300
ORDER_RETRY_MAX_SECONDS = 3600
EPISODE = re.compile(r"第\s*(\d{1,4})\s*話|#\s*(\d{1,4})\b|[-_ ](\d{1,4})$")
KATAKANA = re.compile(r"[ァ-ヴー]{2,}")
GENERIC_KATAKANA = {"トレーナー", "バトル", "ボール", "メンバー", "チーム", "パワー",
                    "シーン", "ロープ", "テーマ", "ピンチ", "チャンス", "ライバル", "ヒーロー", "パーティ", "パーティー",
                    "ダンス", "レース", "コンテスト", "ステージ", "カレー", "フィールドワーク", "ドキドキ", "ワクワク",
                    "ハラハラ", "ニコニコ", "スタート", "ゴール", "ミッション", "トラブル", "パニック", "ショー",
                    "ルール", "タイプ", "レッスン", "ストーリー", "ヒント", "スキル", "ミステリー", "ライブ", "ニュース"}


class SceneNamesError(ValueError):
    """A record or input does not match the contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(state_root: Path) -> Path:
    return Path(state_root) / DIRECTORY / "scenes.json"


def _empty() -> dict:
    return {"type": "bgm_scene_names", "schema_version": SCHEMA_VERSION, "policy_id": POLICY_ID,
            "cues": {}, "recordings": {}, "works": {}}


def read(state_root: Path) -> dict:
    path = _path(state_root)
    if not path.exists():
        return _empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data.get("type") != "bgm_scene_names" or data.get("schema_version") != SCHEMA_VERSION
                or not all(isinstance(data.get(k), dict) for k in ("cues", "recordings", "works"))):
            raise ValueError("schema")
    except (OSError, ValueError) as exc:
        raise SceneNamesError(f"場面名の記録を読み込めません: {path}") from exc
    return data


def _write(state_root: Path, data: dict) -> None:
    from lss.core.session import _write_json_atomic

    path = _path(state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, data)


def update(state_root: Path, change) -> dict:
    """Read, apply ``change(data)`` and write under the naming lock (concurrent writers serialise)."""
    from lss.assets.naming import _locked

    path = _path(state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path.with_suffix(".lock")):
        data = read(state_root)
        data["policy_id"] = POLICY_ID
        change(data)
        _write(state_root, data)
    return data


# --- observed facts of the source file ----------------------------------------------------------------

def source_facts(source: Path, *, tags: dict | None = None) -> dict:
    """Title, work, episode number and cast list the source file itself carries (no lookup)."""
    from lss.bgm.reference_acquisition import work_key, work_title

    if tags is None:
        try:
            from lss.core.media_metadata import format_tags

            tags = format_tags(Path(source), timeout=60)
        except (OSError, ValueError, subprocess.SubprocessError):
            tags = {}
    tags = {str(k).lower(): v for k, v in (tags or {}).items()}
    title = tags.get("title") or Path(source).stem
    cast, inside = [], False
    for line in str(tags.get("description") or "").splitlines():
        if line.startswith("■"):
            inside = "キャスト" in line or "声の出演" in line
            continue
        found = re.match(r"\s*([^\s：:■▼※]{1,20})\s*[：:]\s*\S", line) if inside else None
        if found and found.group(1) not in cast:
            cast.append(found.group(1))
    episode = None
    for text in (unicodedata.normalize("NFKC", title), Path(source).stem):
        match = EPISODE.search(text)
        if match:
            episode = int(next(g for g in match.groups() if g))
            break
    work = work_title(title) or work_title(Path(source).stem) or Path(source).stem
    # A franchise title alone spans several series; the official site the file names
    # tells this series apart, so it is part of the work's key.
    from lss.bgm import work_info

    site = work_info.official_site(str(tags.get("description") or ""))
    key = work_key(f"{work}|{site}" if site else work)
    return {"title": title, "work_title": work, "work_key": key, "episode": episode, "cast": cast, "site": site,
            "description": str(tags.get("description") or "")[:20000]}


# --- guards -----------------------------------------------------------------------------------------

def outside_episode(start: float, end: float, heard: list[str]) -> str | None:
    """Only explicit program-announcement observations identify an outside-episode cue."""
    text = "".join(heard)
    hits = len(OUTSIDE.findall(text))
    if hits and (start < 60 or hits >= 2):
        return "announcement"
    return None


def unheard_names(fields: list[str | None], heard: str, cast: list[str]) -> list[str]:
    """Katakana names the model wrote that were never heard and are not in the file's cast list (invented)."""
    found = []
    for value in fields:
        for word in KATAKANA.findall(value or ""):
            if word in GENERIC_KATAKANA or word in heard or any(word in c or c in word for c in cast):
                continue
            found.append(word)
    return sorted(set(found))


# --- records ----------------------------------------------------------------------------------------

def occurrence_lines(heard: list[dict], start: float, end: float) -> list[dict]:
    """Nonempty timed observations overlapping this occurrence, not its synopsis or lead-in."""
    return [line for line in heard if isinstance(line.get("text"), str) and line["text"].strip()
            and all(isinstance(line.get(k), (int, float)) and math.isfinite(line[k]) for k in ("start", "end"))
            and line["start"] < line["end"] and line["start"] < end and line["end"] > start]


def scene_context_lines(heard: list[dict], start: float, end: float) -> list[dict]:
    """Timed observations in the cue and its existing 20-second lead-in naming context."""
    context_start = start - 20.0
    return [line for line in heard if isinstance(line.get("text"), str) and line["text"].strip()
            and all(isinstance(line.get(k), (int, float)) and math.isfinite(line[k]) for k in ("start", "end"))
            and line["start"] < line["end"] and line["start"] < end and line["end"] > context_start]


def _evidence_hash(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def context_music_evidence(facts: dict, heard: list[dict], music: dict | None,
                           measured_music_description: str | None = None) -> dict:
    """Bind a creative no-scene title to actual source context, transcript observations, and measured music input."""
    source_context = {key: facts.get(key) for key in ("title", "work_title", "episode", "cast")}
    description = facts.get("description")
    if isinstance(description, str) and description:
        source_context["description"] = description[:1200]
    observed = [{key: line.get(key) for key in ("start", "end", "text")} for line in heard]
    measured_music = music or {}
    return {"policy": "context-music-title-inputs-v1", "basis": "context_music", "scene_asserted": False,
            "source_context": source_context, "source_context_sha256": _evidence_hash(source_context),
            "observed_transcript_sha256": _evidence_hash(observed),
            "measured_music_sha256": _evidence_hash(measured_music),
            "measured_music_description": measured_music_description,
            "measured_music_description_sha256": _evidence_hash(measured_music_description)}


def _context_music_vocabulary(facts: dict, heard: list[dict], measured_music_description: str | None) -> set[str]:
    import unicodedata
    from lss.bgm import scene_name_spellings

    context_values = [facts.get(key) for key in ("title", "work_title", "description")]
    context_values.extend(facts.get("cast") or [])
    context_values.extend(line.get("text") for line in heard)
    context_values.append(measured_music_description)
    # Only text from validated public sources already supplied to the prompt can witness a spelling.
    for source in facts.get("published_name_sources") or []:
        if scene_name_spellings.source_headers([source]):
            context_values.append(source.get("text"))
    # General title vocabulary is not a cast list. Work-specific nouns must come from actual inputs.
    vocabulary = {"テーマ", "メロディ", "メロディー", "リズム", "ハーモニー", "ワルツ", "マーチ",
                  "セレナーデ", "ノクターン", "ラプソディ", "ラプソディー", "バラード", "ファンファーレ",
                  "シンフォニー", "アリア", "スケルツォ", "カノン", "エチュード", "プレリュード", "ロンド",
                  "ビート", "ステップ", "ダンス", "ドキドキ", "ワクワク", "スタート", "ゴール"}
    for value in context_values:
        if isinstance(value, str):
            vocabulary.update(KATAKANA.findall(unicodedata.normalize("NFKC", value)))
    return vocabulary


def record_cue(state_root: Path, asset_id: str, *, pcm_sha256: str, source_sha256: str, facts: dict,
               start_seconds: float, end_seconds: float, heard: list[dict], order: dict, model: dict,
               music: dict | None = None, name_basis_evidence: dict | None = None,
               measured_music_description: str | None = None,
               attempt_identity: dict | None = None,
               attempted_at: float | None = None, completed_at: float | None = None) -> dict:
    """Store what was heard and the proposed order line/name for one published cue (idempotent per PCM)."""
    for field in ("use", "direction", "category", "name"):
        value = order.get(field)
        if value is not None and (not isinstance(value, str) or len(value) > 200):
            raise SceneNamesError(f"{field}が不正です")
    if order.get("category") is not None and order["category"] not in CATEGORIES:
        order = {**order, "category": None}
    observed = occurrence_lines(heard, start_seconds, end_seconds)
    heard_text = "".join(line["text"] for line in observed)
    reason = outside_episode(start_seconds, end_seconds, [line["text"] for line in heard])
    from lss.bgm import scene_name_spellings

    guards = {}
    name, use = order.get("name"), order.get("use")
    basis = order.get("name_basis")
    published_sources = facts.get("published_name_sources") or []
    spelling_support = []
    # Legacy orders without the typed field stay on the existing strict scene-evidence route.
    if basis is None and (name or use or order.get("evidence")):
        basis = "scene"
    if reason:
        guards["outside_episode"] = reason
        name, use = None, None
    elif basis == "scene":
        if not observed:
            guards["scene_observation"] = "no_timed_occurrence_text"
            name, use = None, None
        elif name or use:
            quote = order.get("evidence")
            if not isinstance(quote, str) or not quote.strip() or not any(quote.strip() in line["text"] for line in observed):
                guards["scene_observation"] = "quote_not_in_occurrence_text"
                name, use = None, None
        # Inferred scene-use is evidence too: an unheard proper name must not be hidden by a generic title.
        spelling_support = scene_name_spellings.witnessed([name, use], observed, published_sources)
        supported = {item["canonical"] for item in spelling_support}
        invented = [word for word in unheard_names([name, use], heard_text, facts.get("cast") or [])
                    if word not in supported]
        if invented:
            guards["unheard_names"] = invented
            name = None
            if use and any(word in use for word in invented):
                use = None
    elif basis == "context_music":
        expected_context = context_music_evidence(facts, heard, music, measured_music_description)
        if name_basis_evidence != expected_context:
            guards["name_basis"] = "context_music_inputs_missing_or_mismatched"
            name = None
        has_source_context = any(isinstance(facts.get(key), str) and facts[key].strip()
                                 for key in ("title", "work_title", "description"))
        has_source_context = has_source_context or isinstance(facts.get("episode"), int)
        has_source_context = has_source_context or any(isinstance(name, str) and name.strip()
                                                       for name in facts.get("cast") or [])
        has_transcript_context = any(isinstance(line.get("text"), str) and line["text"].strip()
                                     for line in heard)
        has_measured_music = bool(music) and measured_music_description != "測定情報なし"
        if not (has_source_context or has_transcript_context or has_measured_music):
            guards["name_basis"] = "context_music_has_no_source_or_measured_inputs"
            name = None
        if use or order.get("evidence") or order.get("category") == "本編外":
            guards["scene_observation"] = "context_music_cannot_assert_a_scene"
            name, use = None, None
        if name and (not 3 <= len(name) <= 16 or not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", name)):
            guards["name_shape"] = "context_title_must_be_3_to_16_japanese_characters"
            name = None
        if name:
            unsupported = sorted({token for token in KATAKANA.findall(unicodedata.normalize("NFKC", name))
                                  if token not in _context_music_vocabulary(facts, heard,
                                                                            measured_music_description)})
            if unsupported:
                guards["unwitnessed_context_names"] = unsupported
                name = None
        # This title basis is expressly interpretive. It never creates a use/scene field or quote claim.
        use = None
    else:
        if name or use or order.get("evidence"):
            guards["name_basis"] = "missing_or_unknown_title_basis"
        name, use = None, None
    # Keep the established combined admission above as the first gate. A cast list can explain where
    # a spelling came from, but does not show that this actor was part of the observed scene. Once an
    # order is otherwise admitted, retain each field only when its non-generic Katakana tokens were
    # actually heard in the scene/lead-in or have an existing public spelling witness in that context.
    if name and basis == "scene" and model.get("id") != "fixed_name_of_recording":
        context = scene_context_lines(heard, start_seconds, end_seconds)
        context_tokens = {word for line in context
                          for word in KATAKANA.findall(unicodedata.normalize("NFKC", line["text"]))}
        context_support = scene_name_spellings.witnessed([name, use], context, published_sources)
        context_canonical = {item["canonical"] for item in context_support}

        def unsupported_tokens(value: str | None) -> list[str]:
            return sorted({word for word in KATAKANA.findall(unicodedata.normalize("NFKC", value or ""))
                           if word not in GENERIC_KATAKANA and word not in context_tokens
                           and word not in context_canonical})

        unsupported_name = unsupported_tokens(name)
        unsupported_use = unsupported_tokens(use)
        field_guard = {}
        if unsupported_name:
            field_guard["name"] = {"reason": "katakana_token_not_witnessed_in_scene_context",
                                   "tokens": unsupported_name}
            name = None
        if unsupported_use:
            field_guard["use"] = {"reason": "katakana_token_not_witnessed_in_scene_context",
                                  "tokens": unsupported_use}
            use = None
        if field_guard:
            guards["field_guard"] = field_guard
    status = ("fixed_name" if model.get("id") == "fixed_name_of_recording" else
              "transient_error" if order.get("error") else
              "accepted" if name else "guard_rejected" if guards else "unresolved")
    attempted_at = _now() if attempted_at is None else float(attempted_at)
    completed_at = attempted_at if completed_at is None else float(completed_at)
    entry = {"asset_id": asset_id, "pcm_sha256": pcm_sha256, "source_sha256": source_sha256,
             "work_key": facts["work_key"], "work_title": facts["work_title"], "episode": facts.get("episode"),
             "start_seconds": round(float(start_seconds), 3), "end_seconds": round(float(end_seconds), 3),
             "heard": heard, "published_name_sources": scene_name_spellings.source_headers(published_sources),
             "published_spelling_support": spelling_support, "use": use, "direction": order.get("direction"),
             "category": ("本編外" if reason else None if "scene_observation" in guards else order.get("category")),
             "name": name, "name_basis": basis if name else None,
             "name_basis_evidence": name_basis_evidence if basis == "context_music" else None,
             "music": music or {},
             "proposed": {**{k: order.get(k) for k in ("use", "direction", "category", "name", "evidence")},
                          **({"name_basis": order.get("name_basis")} if "name_basis" in order else {})},
             "guards": guards, "observation_policy": OBSERVATION_POLICY_ID, "model": model,
             "naming_policy": POLICY_ID,
             "attempt_identity": attempt_identity, "attempt_status": status, "retry_after": None,
             "attempts": [], "recorded_at": _now()}

    def change(data):
        previous = data["cues"].get(asset_id)
        attempts = list(previous.get("attempts") or []) if previous else []
        if previous and not attempts:
            # Preserve pre-attempt-identity proposals when this cue first becomes retryable.
            attempts.append({"status": "legacy_record", "pcm_sha256": previous.get("pcm_sha256"),
                             "source_sha256": previous.get("source_sha256"), "heard": previous.get("heard") or [],
                             "proposed": previous.get("proposed") or {}, "guards": previous.get("guards") or {},
                             "model": previous.get("model") or {}, "recorded_at": previous.get("recorded_at")})
        retry_after = None
        if status == "transient_error":
            retry_key = (attempt_identity or {}).get("retry_key")
            same_signature_errors = sum(1 for attempt in attempts
                                        if attempt.get("status") == "transient_error"
                                        and (attempt.get("identity") or {}).get("retry_key") == retry_key)
            delay = min(ORDER_RETRY_BASE_SECONDS * (2 ** min(same_signature_errors, 20)),
                        ORDER_RETRY_MAX_SECONDS)
            retry_after = completed_at + delay
        entry["retry_after"] = retry_after
        entry["attempts"] = attempts + [{"identity": attempt_identity, "status": status,
                                        "source_sha256": source_sha256, "pcm_sha256": pcm_sha256,
                                        "heard": heard, "published_name_sources": entry["published_name_sources"],
                                        "published_spelling_support": spelling_support,
                                        "proposed": entry["proposed"], "name": name, "use": use,
                                        "name_basis_evidence": entry["name_basis_evidence"],
                                        "guards": guards, "model": model, "error": order.get("error"),
                                        "attempted_at": attempted_at, "completed_at": completed_at,
                                        "retry_after": retry_after}]
        data["cues"][asset_id] = entry
        work = data["works"].setdefault(facts["work_key"], {"title": facts["work_title"], "prefix": None,
                                                              "created_at": _now()})
        work.setdefault("title", facts["work_title"])
    update(state_root, change)
    return entry


def cue_record(state_root: Path, asset_id: str, pcm_sha256: str | None = None) -> dict | None:
    entry = read(state_root)["cues"].get(asset_id)
    if entry is None or (pcm_sha256 is not None and entry["pcm_sha256"] != pcm_sha256):
        return None
    return entry


# How an automatic prefix is chosen; a prefix chosen under an older policy is chosen again (a user's stays).
PREFIX_POLICY = "series-subtitle-v3"


def prefix_needed(state_root: Path, work_key: str) -> bool:
    work = read(state_root)["works"].get(work_key) or {}
    if not work.get("prefix"):
        return True
    return work.get("prefix_basis") != "user" and work.get("prefix_policy") != PREFIX_POLICY


def set_work_prefix(state_root: Path, work_key: str, prefix: str, *, basis: str) -> str:
    """Fix the work's M-number prefix (two or three Latin capitals) once per policy. A user's prefix is kept."""
    prefix = unicodedata.normalize("NFKC", prefix or "").upper()
    prefix = re.sub(r"[^A-Z]", "", prefix)[:3]
    if len(prefix) < 2:
        raise SceneNamesError("作品の略号は英字2〜3文字が必要です")
    kept = {}

    def change(data):
        work = data["works"].setdefault(work_key, {"created_at": _now()})
        stale = work.get("prefix") and work.get("prefix_basis") != "user" and work.get("prefix_policy") != PREFIX_POLICY
        if not work.get("prefix") or stale:
            if stale:
                work.setdefault("previous_prefixes", []).append(
                    {"prefix": work["prefix"], "policy": work.get("prefix_policy"), "replaced_at": _now()})
            work.update(prefix=prefix, prefix_basis=basis, prefix_policy=PREFIX_POLICY, prefix_set_at=_now())
        kept["prefix"] = work["prefix"]
    update(state_root, change)
    return kept["prefix"]


def rekey_source(state_root: Path, source_sha256: str, facts: dict) -> int:
    """Move a source's cue records to the work key the source facts give now (a finer series key)."""
    moved = []

    def change(data):
        for cue in data["cues"].values():
            if cue["source_sha256"] == source_sha256 and cue["work_key"] != facts["work_key"]:
                cue["work_key"] = facts["work_key"]
                moved.append(cue["asset_id"])
        if moved:
            data["works"].setdefault(facts["work_key"], {"title": facts["work_title"], "created_at": _now()})
    update(state_root, change)
    return len(moved)


def fallback_prefix(work_title: str) -> str:
    """Deterministic prefix when no model proposed one: the first Latin letters of the title, else a hash."""
    letters = re.sub(r"[^A-Z]", "", unicodedata.normalize("NFKC", work_title or "").upper())
    if len(letters) >= 2:
        return letters[:2]
    digest = hashlib.sha256((work_title or "").encode("utf-8")).digest()
    return chr(65 + digest[0] % 26) + chr(65 + digest[1] % 26)


# --- recordings: fixed names and M numbers ------------------------------------------------------------

def _labels(memory: dict | None) -> dict[str, str]:
    """asset_id -> recording key from the published-cue memory (only labels that were not merged away)."""
    keys = {}
    for number, entry in ((memory or {}).get("labels") or {}).items():
        if entry.get("merged_into"):
            continue
        for member in entry.get("members") or []:
            keys[member] = f"label:{number}"
    return keys


def recording_key(asset_id: str, memory: dict | None) -> str:
    return _labels(memory).get(asset_id, f"asset:{asset_id}")


def fixed_recording(data: dict, asset_id: str, memory: dict | None) -> tuple[str, dict] | None:
    """The fixed name of this cue's recording: the group's, or the oldest name a member got while alone."""
    key = recording_key(asset_id, memory)
    candidates = []
    if key in data["recordings"]:
        candidates.append(data["recordings"][key])
    if key.startswith("label:"):
        members = (((memory or {}).get("labels") or {}).get(key[6:]) or {}).get("members") or []
        candidates += [data["recordings"][f"asset:{m}"] for m in members if f"asset:{m}" in data["recordings"]]
    if not candidates:
        return None
    return key, min(candidates, key=lambda r: r["named_at"])


def _name_key(name: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKC", str(name)).casefold().replace(" ", "")


LEGACY_POLICY = "legacy"
OWNED_BASES = ("scene", "context_music")   # LSS-owned names; a verified official or user title is never replaced


def policy_rank(policy: str | None) -> int:
    """Order of naming policies: the ``vN`` in ``bgm-scene-names-vN-...`` (a record without one ranks 0)."""
    found = re.search(r"-v(\d+)(?:-|$)", str(policy or ""))
    return int(found.group(1)) if found else 0


def needs_renaming(cue: dict | None) -> bool:
    """A cue named by an older naming policy than this LSS's: a newer LSS may give its recording a better name."""
    return bool(cue and cue.get("name") and policy_rank(cue.get("naming_policy")) < policy_rank(POLICY_ID))


def _revise_names(recordings: dict, by_recording: dict, cues: dict, labels: dict) -> None:
    """Replace an LSS-owned recording name with the one a newer naming policy gave a member cue.

    The whole identity changes at once (the name lives on the recording, every occurrence projects it); the old
    name, its policy and basis stay in ``revisions``. Same policy and same input renames nothing."""
    for key, members in by_recording.items():
        recording = recordings.get(key)
        if not recording or recording.get("basis") not in OWNED_BASES:
            continue
        held = policy_rank(recording.get("naming_policy"))
        newer = [m for m in members if m.get("name") and policy_rank(m.get("naming_policy")) > held
                 and m.get("name_basis") in OWNED_BASES]
        if not newer:
            continue
        best = max(newer, key=lambda m: (policy_rank(m.get("naming_policy")), m["recorded_at"], m["asset_id"]))
        work = best.get("work_key")
        taken = {_name_key(r["name"]) for k, r in recordings.items() if k != key
                 and not (k.startswith("asset:") and labels.get(k[6:]))
                 and (cues.get(r.get("first_asset")) or {}).get("work_key") == work}
        name, n = best["name"], 1
        while _name_key(name) in taken:
            n += 1
            name = f"{best['name']} その{n}"
        history = list(recording.get("revisions") or [])
        history.append({k: recording.get(k) for k in ("name", "basis", "naming_policy", "named_at", "first_asset")}
                       | {"replaced_at": _now()})
        revised = {**recording, "naming_policy": best["naming_policy"], "revisions": history}
        if _name_key(name) != _name_key(recording["name"]):
            revised.update(name=name, basis=best.get("name_basis") or "scene", first_asset=best["asset_id"],
                           named_at=best["recorded_at"],
                           order=" ".join(x for x in (best.get("use"), best.get("direction")) if x))
        else:
            revised["revisions"] = history[:-1]   # same name under a newer policy: only the policy advances
        recordings[key] = revised


def settle(state_root: Path, memory: dict | None = None) -> dict:
    """Give every recording with a named cue its fixed name (first given wins) and refresh M numbers."""
    labels = _labels(memory)

    def change(data):
        cues = data["cues"]
        by_recording: dict[str, list[dict]] = {}
        for asset_id, cue in cues.items():
            by_recording.setdefault(labels.get(asset_id, f"asset:{asset_id}"), []).append(cue)
        recordings = data["recordings"]
        fresh = []
        # A cue first named alone that later joined a recurring group: the oldest name of the group's members wins.
        for key, members in by_recording.items():
            candidates = [recordings[f"asset:{m['asset_id']}"] for m in members
                          if key != f"asset:{m['asset_id']}" and f"asset:{m['asset_id']}" in recordings]
            if key in recordings:
                candidates.append(recordings[key])
            if candidates:
                recordings[key] = min(candidates, key=lambda r: r["named_at"])
                continue
            named = sorted((m for m in members if m.get("name")), key=lambda m: m["recorded_at"])
            if named:
                fresh.append((named[0]["recorded_at"], key, named[0]))
        # Two different recordings of one work never share a fixed name (they share one series album): a name
        # given earlier keeps it, a later recording proposed the same name gets the next free ``その2``, ``その3``.
        # Only names given now are adjusted; a fixed name already given is never renamed.
        for _, key, first in sorted(fresh, key=lambda item: (item[0], item[1])):
            work = first.get("work_key")
            taken = {_name_key(r["name"]) for k, r in recordings.items() if k != key
                     and not (k.startswith("asset:") and labels.get(k[6:]))
                     and (cues.get(r.get("first_asset")) or {}).get("work_key") == work}
            name, n = first["name"], 1
            while _name_key(name) in taken:
                n += 1
                name = f"{first['name']} その{n}"
            recordings[key] = {"name": name, "category": first.get("category"),
                               "basis": first.get("name_basis") or "scene",
                               "order": " ".join(x for x in (first.get("use"), first.get("direction")) if x),
                               "first_asset": first["asset_id"], "named_at": first["recorded_at"],
                               "naming_policy": first.get("naming_policy") or LEGACY_POLICY}
        _revise_names(recordings, by_recording, cues, labels)
        for key in [k for k in recordings if k.startswith("asset:") and labels.get(k[6:])]:
            recordings.pop(key)
        data["m_numbers"] = _m_numbers(data, by_recording)
    return update(state_root, change)


def _section(recording: dict | None, members: list[dict]) -> str:
    episodes = {m.get("episode") for m in members if m.get("episode") is not None}
    category = (recording or {}).get("category") or next((m.get("category") for m in members if m.get("category")), None)
    if category == "本編外":
        return "program"
    if len(episodes) >= THEME_MIN_EPISODES:
        return "theme"
    return CATEGORIES.get(category or "", "scenery")


def _m_numbers(data: dict, by_recording: dict[str, list[dict]]) -> dict:
    works: dict[str, list] = {}
    for key, members in by_recording.items():
        work = members[0]["work_key"]
        first = min(members, key=lambda m: ((m.get("episode") if m.get("episode") is not None else 10 ** 6),
                                            m["start_seconds"]))
        works.setdefault(work, []).append((key, members, first))
    numbers = {}
    for work, entries in works.items():
        prefix = (data["works"].get(work) or {}).get("prefix") or fallback_prefix((data["works"].get(work) or {}).get("title", ""))
        known = [first.get("episode") for _, _, first in entries if first.get("episode") is not None]
        earliest = min(known) if known else None

        def session(first):
            return 2 if (earliest is not None and first.get("episode") is not None
                         and first["episode"] > earliest + SECOND_SESSION_AFTER - 1) else 1
        ordered = sorted(entries, key=lambda e: (session(e[2]), SECTIONS.index(_section(data["recordings"].get(e[0]), e[1])),
                                                 e[2].get("episode") if e[2].get("episode") is not None else 10 ** 6,
                                                 e[2]["start_seconds"], e[0]))
        counters = {1: 0, 2: 100}
        for key, members, first in ordered:
            s = session(first)
            counters[s] += 1
            code = f"{prefix}{counters[s]:02d}"
            for member in members:
                numbers[member["asset_id"]] = {"code": code, "recording": key,
                                               "section": _section(data["recordings"].get(key), members)}
    return numbers


# --- projection into the listening file --------------------------------------------------------------

def presentation(state_root: Path, asset_id: str, pcm_sha256: str | None = None) -> dict | None:
    """Naming evidence and LSS-owned tags for one cue, or None when there is no scene record for its audio."""
    try:
        data = read(state_root)
    except SceneNamesError:
        return None
    cue = data["cues"].get(asset_id)
    if cue is None or (pcm_sha256 is not None and cue["pcm_sha256"] != pcm_sha256):
        return None
    number = (data.get("m_numbers") or {}).get(asset_id) or {}
    recording = data["recordings"].get(number.get("recording") or f"asset:{asset_id}") or {}
    tags = {}
    if number.get("code"):
        tags["M_NUMBER"] = number["code"]
    order = " ".join(x for x in (cue.get("use"), cue.get("direction")) if x)
    if order:
        tags["ORDER_MENU"] = order
    from lss.bgm import music_facts

    tags.update(music_facts.tags(cue.get("music") or {}))
    evidence = None
    name = recording.get("name")
    if name:
        evidence = {"context": {"asset_id": asset_id, "scope": "cue", "description": name,
                                "basis": recording.get("basis") or "scene",
                                "recording": number.get("recording") or f"asset:{asset_id}",
                                "first_asset": recording.get("first_asset"),
                                "evidence_ref": f"bgm-scene:{number.get('recording') or asset_id}"}}
    return {"evidence": evidence, "tags": tags, "name": name, "category": cue.get("category")}
