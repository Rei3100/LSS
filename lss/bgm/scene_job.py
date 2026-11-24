"""Name the published BGM cues of one source by their scene, after publication (never blocking extraction).

Flow for one source (its CurrentBest files):
1. Cues whose recording already has a fixed name only get their order line copied (no inference).
2. The other cues: the dialogue heard from ``BEFORE`` seconds before the cue to its end is transcribed from the
   source mix (the separated speech stem was measured worse: release-evidence/scene-naming-20261016/), then one
   language-model call writes either a quote-grounded scene order/name or a context_music provisional title.
3. Guards (announcements, eyecatch stingers, never-heard names) and records (`scene_names.record_cue`), the
   work's M-number prefix once, fixed names and M numbers (`scene_names.settle`).
4. Every published file of the work whose presentation changed is refreshed through the listening finisher
   (PCM-checked; LSS rewrites its complete tag set).

``models`` provides ``transcribe(source, windows) -> [{start, end, text}]``, ``order(prompt) -> dict`` and
``prefix(work_title) -> str``; the default is ``scene_runtime.create_models()``: native whisper.cpp/llama.cpp
(``scene_native``, also in the packaged app) or the source-only Transformers ``scene_models.SceneModels``.
"""
from __future__ import annotations

import hashlib
import json
import os
import math
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

from lss.bgm import music_facts, scene_names, scene_stop

BEFORE = 20.0
MAX_BACKGROUND_ATTEMPTS = 3
SCENE_ATTEMPT_POLICY = "scene-attempt-signature-r1"
ORDER_PROMPT_POLICY = "scene-order-prompt-r2-scene-or-context-music"
SOURCE_FACTS_POLICY = "scene-source-facts-r1"
SPELLING_CONTEXT_POLICY = "published-spelling-input-r2"
MUSIC_INPUT_POLICY = "scene-music-input-r1"
MENU = ("日常・のどか / 楽しい・明るい / コミカル / 緊迫・危機 / バトル / 不穏・怪しい / 悲しみ・切なさ / 感動・希望 / "
        "決意・出発 / 神秘・不思議")
# How real score orders become released titles (other works' TV score sheets: order line -> soundtrack title).
# Never examples from a work LSS is naming, and never a reference title used to evaluate LSS's names.
ORDER_EXAMPLES = (("くつろいだ食事の時間 しっとりと", "いただきますのひととき"), ("淋しさ", "涙のあとで"),
                  ("感動 しずかに", "未来を信じて"), ("すれ違う気持ち 葛藤", "想いのほころび"),
                  ("立ちはだかる最大の敵", "立ちはだかる野望"), ("特訓 スポ根的", "特訓"),
                  ("星空 美しく", "美しい青空"), ("新たなる脅威 出撃", "強襲"),
                  ("先走って失敗 ユーモラスに", "あわてんぼうの一日"), ("追走 追いつ追われつ", "待てー！"))
STYLE = "、".join(f"発注「{order}」→「{title}」" for order, title in ORDER_EXAMPLES)


def _stable_hash(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _retry_clock() -> float:
    return time.time()


def _current_model_revision(models=None) -> dict:
    if models is not None:
        return models.revision()
    from lss.bgm import scene_runtime

    return scene_runtime.revision()


def _effective_facts_hash(facts: dict | None) -> str | None:
    if facts is None:
        return None
    return _stable_hash({k: facts.get(k) for k in
                         ("title", "work_title", "work_key", "episode", "cast", "site", "description")})


def _work_info_cache_identity(state_root: Path, facts: dict | None) -> str:
    """Fingerprint only local pages this source could consult; never fetch during eligibility checks."""
    import urllib.parse

    root = Path(state_root) / "bgm-work-info" / "pages"
    rows = []
    official_found = False
    description = (facts or {}).get("description") or ""
    episode, work_title = (facts or {}).get("episode"), (facts or {}).get("work_title") or ""
    site = None
    try:
        from lss.bgm.work_info import LISTING_HINT, episode_blocks, official_site

        site = official_site(description)
        if site and episode is not None:
            def read_page(url, *, links_only=False):
                path = root / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
                try:
                    page = json.loads(path.read_text(encoding="utf-8"))
                    row = {"url": url, "links": page.get("links") or []} if links_only else {
                        "url": url, "text_sha256": page.get("text_sha256") or _stable_hash(page.get("text") or "")}
                    return page, row
                except (OSError, ValueError, AttributeError):
                    return None, {"url": url, "missing": True}

            top, top_row = read_page(site, links_only=True)
            rows.append(top_row)
            top = top or {}
            base = urllib.parse.urlsplit(site)
            candidates = []
            for link in top.get("links") or []:
                parts = urllib.parse.urlsplit(link)
                if parts.netloc == base.netloc and parts.path.startswith(base.path) and LISTING_HINT.search(parts.path):
                    candidates.append(urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")))
            candidates.sort(key=lambda url: (url.count("/"), len(url)))
            if candidates:
                listing = candidates[0]
                page_base = listing if listing.endswith("/") else listing.rsplit("/", 1)[0] + "/"
                for url in [listing] + [f"{page_base}index_{n}.html" for n in range(2, 13)]:
                    page, row = read_page(url)
                    rows.append(row)
                    if page is None:
                        break
                    if int(episode) in episode_blocks(page.get("text") or ""):
                        official_found = True
                        break
    except (OSError, ValueError, AttributeError):
        pass
    if not official_found and work_title.strip():
        url = "https://ja.wikipedia.org/wiki/" + urllib.parse.quote(work_title, safe="")
        path = root / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
        try:
            page = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"url": url, "text_sha256": page.get("text_sha256") or _stable_hash(page.get("text") or ""),
                         "first_h1": page.get("first_h1")})
        except (OSError, ValueError, AttributeError):
            rows.append({"url": url, "missing": True})
    return _stable_hash(rows)


def _retry_key(state_root: Path, source_sha256: str, cue: dict, model_revision: dict,
               facts: dict | None = None, *, effective_facts_sha256: str | None = None) -> str:
    """Cheap compatibility key for deciding whether an unresolved attempt deserves new local inference.

    The source digest binds file-derived facts; policy identifiers cover changes to their extraction, guards,
    and prompt. The cache digest detects newly available/changed public spelling inputs without a network read.
    """
    return _stable_hash({"source_sha256": source_sha256, "pcm_sha256": cue["pcm_sha256"],
                         "model_revision": model_revision, "attempt_policy": SCENE_ATTEMPT_POLICY,
                         "order_prompt_policy": ORDER_PROMPT_POLICY, "source_facts_policy": SOURCE_FACTS_POLICY,
                         "spelling_context_policy": SPELLING_CONTEXT_POLICY,
                         "music_input_policy": MUSIC_INPUT_POLICY,
                         "scene_policy": scene_names.POLICY_ID,
                         "observation_policy": scene_names.OBSERVATION_POLICY_ID,
                         "work_info_enabled": _work_info_enabled(),
                         "effective_facts_sha256": effective_facts_sha256 or _effective_facts_hash(facts),
                         "work_info_cache": _work_info_cache_identity(state_root, facts) if _work_info_enabled() else None})


def _attempt_identity(state_root: Path, source_sha256: str, cue: dict, model_revision: dict,
                      facts: dict, synopsis: dict | None, heard: list[dict], prompt: str | None,
                      sound: str, named: list[str], transcription_receipt: dict | None = None,
                      transcription_window_ids: list[int] | None = None) -> dict:
    facts_input = {k: facts.get(k) for k in ("title", "work_title", "work_key", "episode", "cast", "site", "description")}
    spelling_input = {k: (synopsis or {}).get(k) for k in ("source_kind", "url", "text", "text_sha256", "page_text_sha256")}
    return {"retry_key": _retry_key(state_root, source_sha256, cue, model_revision, facts),
            "source_sha256": source_sha256, "pcm_sha256": cue["pcm_sha256"],
            "model_revision": model_revision, "attempt_policy": SCENE_ATTEMPT_POLICY,
            "order_prompt_policy": ORDER_PROMPT_POLICY, "source_facts_policy": SOURCE_FACTS_POLICY,
            "effective_facts": facts_input, "effective_facts_sha256": _stable_hash(facts_input),
            "spelling_input_sha256": _stable_hash(spelling_input),
            "observation_policy": scene_names.OBSERVATION_POLICY_ID,
            "observation_sha256": _stable_hash(heard), "prompt_sha256": _stable_hash(prompt) if prompt else None,
            "music_input_sha256": _stable_hash({"music": cue.get("music") or {}, "sound": sound}),
            "named_context_sha256": _stable_hash(named),
            "transcription_window_ids": transcription_window_ids or [],
            "transcription_receipt": transcription_receipt}


def _attempt_is_due(entry: dict, retry_key: str, now: float) -> bool:
    attempts = entry.get("attempts") or []
    latest = attempts[-1] if attempts else {}
    identity = latest.get("identity") or entry.get("attempt_identity") or {}
    if not identity.get("retry_key") or identity.get("retry_key") != retry_key:
        return True
    if latest.get("status", entry.get("attempt_status")) == "transient_error":
        retry_after = latest.get("retry_after", entry.get("retry_after"))
        return retry_after is None or float(retry_after) <= now
    return False


def _transient_attempt_count(entry: dict, retry_key: str) -> int:
    return sum(1 for attempt in entry.get("attempts") or []
               if attempt.get("status") == "transient_error"
               and (attempt.get("identity") or {}).get("retry_key") == retry_key)


def _has_retry_budget(entry: dict, retry_key: str) -> bool:
    return _transient_attempt_count(entry, retry_key) < MAX_BACKGROUND_ATTEMPTS


@contextmanager
def _background_worker_lease(state_root: Path, source_sha256: str | None):
    """Nonblocking process lock for one source worker or the single startup recovery coordinator."""
    lock_name = f"scene-{source_sha256.lower()}.lock" if source_sha256 else "scene-recovery.lock"
    path = Path(state_root) / scene_names.DIRECTORY / "jobs" / lock_name
    path.parent.mkdir(parents=True, exist_ok=True)
    acquired = False
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if __import__("os").name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - Product runtime is Windows
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (OSError, BlockingIOError):
            acquired = False
        try:
            yield acquired
        finally:
            if acquired:
                stream.seek(0)
                if __import__("os").name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _next_transient_retry_after(state_root: Path, source_sha256: str) -> float | None:
    """Earliest retry for a still-published cue; replaced/removed cues cannot keep a job pending."""
    try:
        due = []
        for cue in _published(Path(state_root), source_sha256):
            entry = scene_names.cue_record(state_root, cue["asset_id"], cue["pcm_sha256"])
            if entry is None:
                continue
            if entry.get("source_sha256") != source_sha256:
                continue
            attempts = entry.get("attempts") or []
            latest = attempts[-1] if attempts else {}
            if latest.get("status", entry.get("attempt_status")) != "transient_error":
                continue
            value = latest.get("retry_after", entry.get("retry_after"))
            if isinstance(value, (int, float)) and math.isfinite(value):
                due.append(float(value))
        return min(due) if due else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _pending_retry_with_budget(state_root: Path, source_sha256: str, *, source: Path | None = None,
                               models=None, strict: bool = False) -> float | None:
    """Earliest persisted transient retry whose current identity still has a bounded attempt available."""
    try:
        revision = _current_model_revision(models)
        facts = None
        facts_loaded = source is None
        due = []
        for cue in _published(Path(state_root), source_sha256):
            entry = scene_names.cue_record(state_root, cue["asset_id"], cue["pcm_sha256"])
            if not entry or entry.get("name"):
                continue
            attempts = entry.get("attempts") or []
            latest = attempts[-1] if attempts else {}
            if latest.get("status", entry.get("attempt_status")) != "transient_error":
                continue
            if not facts_loaded:
                facts = scene_names.source_facts(Path(source))
                facts_loaded = True
            identity = latest.get("identity") or entry.get("attempt_identity") or {}
            current_facts = facts if facts is not None else identity.get("effective_facts")
            retry_key = _retry_key(state_root, source_sha256, cue, revision, current_facts,
                                   effective_facts_sha256=None if current_facts is not None else
                                   identity.get("effective_facts_sha256"))
            if not _has_retry_budget(entry, retry_key):
                continue
            retry_after = latest.get("retry_after", entry.get("retry_after"))
            if isinstance(retry_after, (int, float)) and math.isfinite(retry_after):
                due.append(float(retry_after))
        return min(due) if due else None
    except Exception:  # noqa: BLE001 - an eligibility failure never blocks the published result
        if strict:
            raise
        return None


def _requests_dir(state_root: Path) -> Path:
    return Path(state_root) / scene_names.DIRECTORY / "jobs" / "requests"


def _requests(state_root: Path, source_sha256: str) -> list[tuple[Path, dict]]:
    """Naming requests of user-started jobs for this source, one immutable file per request."""
    rows = []
    folder = _requests_dir(state_root)
    for path in sorted(folder.glob(f"{str(source_sha256).lower()}-*.json")) if folder.is_dir() else []:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict):
            rows.append((path, record))
    requested_at = lambda row: row[1].get("requested_at") if isinstance(row[1].get("requested_at"), (int, float)) else 0.0
    return sorted(rows, key=requested_at)   # oldest first


def _record_request(state_root: Path, source: Path, source_sha256: str) -> Path:
    """Remember that a job the user started asked for this source's naming, until a worker finishes it.

    Launch recovery resumes only sources with a request (AGENTS.md section 2: no reprocessing of past media on
    LSS's own initiative; J6: interrupted jobs are recovered). Each request is its own file, so finishing one never
    removes a request that arrived later.
    """
    import uuid

    folder = _requests_dir(state_root)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{str(source_sha256).lower()}-{uuid.uuid4().hex}"
    temporary = folder / f"{name}.tmp"
    temporary.write_text(json.dumps({"source_sha256": str(source_sha256).lower(), "source": str(source),
                                     "requested_at": _retry_clock()}, ensure_ascii=False), encoding="utf-8")
    path = folder / f"{name}.json"
    os.replace(temporary, path)
    return path


def _bgm_job_active(state_root: Path, source_sha256: str) -> bool:
    """Whether a BGM job holds this source's CurrentBest lock (it will publish, then request naming itself)."""
    from lss.bgm import current_best

    path = Path(state_root) / current_best.DIRECTORY / str(source_sha256).lower() / ".owner.lock"
    if not path.is_file():
        return False
    try:
        with path.open("r+b") as stream:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - Product runtime is Windows
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(stream, fcntl.LOCK_UN)
    except OSError:
        return True
    return False


def _nothing_left(state_root: Path, source_sha256: str, source: Path | None) -> bool:
    """Strictly: no naming due now and no budgeted retry later. An unreadable state is not "nothing left",
    and neither is a CurrentBest switch still in progress (its new cues are not listed yet)."""
    try:
        from lss.bgm import current_best

        if current_best.journal_path(Path(state_root), source_sha256).is_file():
            return False
        return (not needs_naming(state_root, source_sha256, source=source, strict=True)
                and _pending_retry_with_budget(state_root, source_sha256, source=source, strict=True) is None)
    except Exception:  # noqa: BLE001 - keep the request when the state cannot be read
        return False


def _end_requests(state_root: Path, source_sha256: str, source: Path | None, *,
                  inside_bgm_job: bool = False) -> None:
    """End the user's naming requests that the current state shows finished.

    Order matters: the request list is taken first, then the source's BGM job lock, then the strict
    ``_nothing_left`` check. A job that starts after the lock check records its request after the list was
    taken, so its request survives; a job that ran before the lock check has published what the check reads.
    Nothing is ended while a BGM job of the source holds its lock, except by that job itself, and nothing while a
    naming worker of the source holds its lease: a launch worker's request check under that lease stays true for
    the whole of its naming.
    """
    with _background_worker_lease(state_root, source_sha256) as acquired:
        if acquired:
            _end_listed_requests(state_root, source_sha256, source, inside_bgm_job=inside_bgm_job)


def _end_listed_requests(state_root: Path, source_sha256: str, source: Path | None, *,
                         inside_bgm_job: bool) -> None:
    snapshot = _requests(state_root, source_sha256)
    if not snapshot:
        return
    if not inside_bgm_job and _bgm_job_active(state_root, source_sha256):
        return   # the running job publishes, then ends or keeps its own request in start_in_background
    if not _nothing_left(state_root, source_sha256, source):
        return
    cutoff = _retry_clock()
    for path, record in snapshot:
        requested_at = record.get("requested_at")
        if isinstance(requested_at, (int, float)) and requested_at > cutoff:
            continue
        try:
            path.unlink()
        except OSError:
            pass


def run_background_worker(state_root: Path, source: Path, source_sha256: str, *, wait=None, clock=None,
                          wait_for_retries: bool = True, require_request: bool = False) -> dict:
    """Run bounded naming attempts, sleeping without model/GPU resources only for persisted transient errors.

    A deploy/rollback stop request (``scene_stop``) ends the worker between cues, download blocks and waits;
    cues recorded so far are kept, nothing is recorded for the interrupted one, and the source stays pending.
    """
    scene_stop.enable()
    try:
        result = _run_background_worker(state_root, source, source_sha256, wait=wait or scene_stop.wait,
                                        clock=clock, wait_for_retries=wait_for_retries,
                                        require_request=require_request)
    except scene_stop.StopRequested:
        return {"state": "stopped", "reason": "scene_worker_stop_requested"}
    # The requested naming is over when nothing is left, or when only retries without a remaining budget are
    # left: a later model or policy change then makes a reprocess candidate, not the user's unfinished job.
    if result.get("state") in ("complete", "current", "retry_pending"):
        # Requests that arrived during this worker (a second job of the same source was told already_running)
        # are covered too when no BGM job of the source runs; otherwise only those made before it started.
        _end_requests(state_root, source_sha256, Path(source))
    return result


def _run_background_worker(state_root: Path, source: Path, source_sha256: str, *, wait=None, clock=None,
                           wait_for_retries: bool = True, require_request: bool = False) -> dict:
    source_sha256 = str(source_sha256).lower()
    if len(source_sha256) != 64 or any(char not in "0123456789abcdef" for char in source_sha256):
        raise ValueError("scene_naming_source_hash_invalid")
    state_root, source = Path(state_root), Path(source).resolve()
    wait = wait or time.sleep
    clock = clock or _retry_clock
    with _background_worker_lease(state_root, source_sha256) as acquired:
        if not acquired:
            return {"state": "already_running", "attempts": 0}
        if require_request and not _requests(state_root, source_sha256):
            # Launch recovery: another worker finished the request while this one waited in line.
            return {"state": "not_requested", "attempts": 0}
        pending = _pending_retry_with_budget(state_root, source_sha256, source=source)
        if not needs_naming(state_root, source_sha256, source=source, now=clock()):
            if pending is None:
                terminal = _next_transient_retry_after(state_root, source_sha256)
                return ({"state": "retry_pending", "attempts": 0, "retry_after": terminal}
                        if terminal is not None else {"state": "current", "attempts": 0})
            if not wait_for_retries:
                return {"state": "retry_pending", "attempts": 0, "retry_after": pending}
            delay = max(0.0, pending - clock())
            if delay:
                wait(delay)
            if require_request and not _requests(state_root, source_sha256):
                return {"state": "not_requested", "attempts": 0}
            if not needs_naming(state_root, source_sha256, source=source, now=clock()):
                return {"state": "retry_pending", "attempts": 0, "retry_after": pending}

        # A fresh frozen runtime prepares native models only after a pending cue and an exact source check.
        from lss.bgm import scene_runtime

        if not scene_runtime.available() and not scene_runtime.bootstrap_supported():
            from lss.paths import is_frozen
            import os

            if is_frozen() or (os.environ.get("LSS_SCENE_BACKEND") or "").strip():
                return {"state": "not_available", "attempts": 0}
        if not scene_runtime.available() and scene_runtime.bootstrap_supported():
            try:
                _validate_source_identity(source, source_sha256)
                scene_runtime.ensure_bootstrapped()
            except scene_stop.StopRequested:
                raise
            except Exception as exc:  # noqa: BLE001 - leave published and accepted state untouched
                return {"state": "bootstrap_failed", "attempts": 0,
                        "error_type": type(exc).__name__, "message": str(exc)}
        result = None
        attempts = 0
        while attempts < MAX_BACKGROUND_ATTEMPTS:
            if require_request and not _requests(state_root, source_sha256):
                # A user job ended the request while this launch worker prepared models or waited.
                return {"state": "not_requested", "attempts": attempts}
            # The checked entry rehashes the source before every attempt; model construction and release live
            # entirely inside name_source, so no model/GPU resource survives into the sleep below.
            result = name_source_checked(state_root, source, source_sha256)
            attempts += 1
            pending = _pending_retry_with_budget(state_root, source_sha256, source=source)
            terminal = _next_transient_retry_after(state_root, source_sha256)
            if terminal is None:
                return {"state": "complete", "attempts": attempts, "result": result}
            if pending is None:
                return {"state": "retry_pending", "attempts": attempts, "retry_after": terminal}
            if not wait_for_retries:
                return {"state": "retry_pending", "attempts": attempts, "retry_after": pending}
            if attempts >= MAX_BACKGROUND_ATTEMPTS:
                return {"state": "retry_pending", "attempts": attempts, "retry_after": pending}
            if needs_naming(state_root, source_sha256, source=source, now=clock()):
                continue  # changed source/model/policy or a due retry; no backoff sleep is needed
            delay = max(0.0, pending - clock())
            if delay:
                wait(delay)
            if needs_naming(state_root, source_sha256, source=source, now=clock()):
                continue
            return {"state": "retry_pending", "attempts": attempts, "retry_after": pending}
        return {"state": "retry_pending", "attempts": attempts,
                "retry_after": _pending_retry_with_budget(state_root, source_sha256, source=source)}


def sound_words(path: Path) -> str:
    """The music's direction in words from measured, verified features (mode, motion, dynamics)."""
    from lss.bgm import music_description

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(1)
    words = []
    try:
        key = music_description.key(mono, rate)
        words.append(("長調（明るめ）" if key["mode"] == "major" else "短調（暗め・切なめ）")
                     if key["status"] == "measured" else "調性ははっきりしない")
    except (ImportError, AttributeError, ValueError):
        pass
    hop = max(1, rate // 10)
    frames = mono[: len(mono) // hop * hop].reshape(-1, hop)
    if frames.size:
        db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
        active = db > db.max() - 40
        spectrum = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1))
        flux = np.maximum(np.diff(np.log1p(spectrum), axis=0), 0).sum(1)
        if flux.size > 2:
            peaks = int(((flux[1:-1] > flux[:-2]) & (flux[1:-1] > flux[2:]) & (flux[1:-1] > np.percentile(flux, 75))).sum())
            per_second = peaks / (len(mono) / rate)
            words.append("音の動きが細かく速い" if per_second > 3.2 else "音の動きはゆったり" if per_second < 1.8
                         else "音の動きは中くらい")
        swing = float(np.percentile(db[active], 90) - np.percentile(db[active], 10)) if active.any() else 0.0
        words.append("強弱の起伏が大きい" if swing > 12 else "強弱はなだらか")
    return "、".join(words)


def order_prompt(facts: dict, before: list[str], during: list[str], sound: str, named: list[str],
                 synopsis: str | None = None, *, context_only: bool = False,
                 transcription_unavailable: bool = False) -> str:
    # Only the official text's spellings of names are given (measured: giving the synopsis itself spread one
    # synopsis scene over several cues of the episode, invented 16 % -> 19 %, no gain in scene match).
    import re as _re

    names = sorted(set(_re.findall(r"[ァ-ヴー]{3,}", synopsis or ""))) if synopsis else []
    published = (f"公開ページの表記候補（人物・生き物などの固有名。曲の間の台詞に対応する語がある場合の表記補正だけに使う。登場や出来事の根拠にはしない）: "
                 f"{'、'.join(names[:40])}\n\n") if names else ""
    source_context = [f"作品名: {facts.get('work_title') or '不明'}",
                      f"話数: {facts.get('episode') or '不明'}",
                      f"入力ファイルの作品・話タイトル: {facts.get('title') or '不明'}"]
    if facts.get("description"):
        source_context.append(f"入力ファイルの説明欄（場面を証明しない参考情報）: {facts['description'][:1200]}")
    source_context_block = ("作品・入力情報（仮題の発想材料。個別の曲が流れた場面の証拠ではない）:\n"
                            + "\n".join(source_context) + "\n\n")
    transcription_notice = ("文字起こしは取得できなかった。台詞がないと確認できたわけではない。"
                            "取得済みの作品・話数情報と音楽情報だけを使う。\n" if transcription_unavailable else "")
    return (source_context_block + transcription_notice + f"作品と話数: {facts['title']}\n登場人物一覧（正しい表記。聞き違いはこの表記に直してよい）: {'、'.join(facts['cast'])}\n"
            f"この話で既に付けた曲名（重ならないように）: {'、'.join(n for n in named if n) or 'なし'}\n\n" + published +
            f"BGMが流れた区間の台詞（自動文字起こし・聞き違いあり）\n曲の直前: {' / '.join(before)[:300] or 'なし'}\n"
            f"曲の間: {' / '.join(during)[:900] or 'なし'}\n曲の音の特徴: {sound}\n\n"
            "あなたは劇伴の発注メニューを書き、曲名を付ける音響監督です。\n"
            "まず name_basis を選ぶ。scene は曲中の観測台詞で具体的な場面を確かめられる場合だけ。context_music は場面を確かめられない場合、"
            "台詞がない場合、または文字起こしが曖昧な場合に使う。作品名・話数・入力タイトル・説明欄・直前/曲中の文字起こし・実測された音の特徴から、"
            "場面を断定しない歌らしい仮題を作る。人物名を含む題名はその人物のテーマという創作上の解釈であり、場面への登場を示す事実ではない。\n"
            "use: 発注メニューの場面の欄。台詞で確かめられる具体的な場面（誰が・何をする／何が起きる）を、文ではなく20字以内の短い句で"
            "（例「くつろいだ食事の時間」「強敵の出現」「はじめての挑戦」）。どの曲にも当てはまる言葉（冒険・決意・希望・謎）で済ませない。"
            "話し手は内容（一人称・口調・呼びかけ）から判断してよい。人物名は一覧の表記で、台詞で確かな時だけ。\n"
            "direction: 発注メニューの曲調の欄。音の特徴に合う12字以内の指定（例「しっとりと」「軽快に」「重々しく」「コミカルに」「勇壮に」「緊迫感」）。\n"
            f"name: use と direction から付ける、劇伴の曲名らしい日本語3〜16文字。発注から曲名への手本（別作品）: {STYLE}。手本そのものは使わない。\n"
            f"category は次から一つ: {MENU}\n"
            "scene の場合: evidence は曲の間の台詞をそのまま引用。引用が曲中の観測にない場合は scene を選ばない。name は3〜16文字。\n"
            "context_music の場合: use と evidence は必ず null。引用、歌詞、人物の登場、具体的な出来事を主張しない。"
            "人物名を題名に使う場合も、その人物のテーマという創作上の解釈。direction と category は音の印象として測定情報に合う場合だけ書き、"
            "入力にないテンポ・楽器・リズム等は補わない。name は3〜16文字の歌らしい仮題。\n"
            + ("今回の入力では曲中の台詞から場面を確認できないため、name_basis=context_music とし、具体的な場面や行動は書かない。\n"
               if context_only else "") +
            '出力はJSONのみ: {"name_basis": "scene または context_music", "use": "...", "direction": "...", '
            '"category": "...", "name": "...", "evidence": "..."}')


def _measured_music_description(music: dict) -> str:
    """Describe only measurements actually present; the normal music_facts formatter has a dynamics fallback."""
    if not music:
        return "測定情報なし"
    measured = (music.get("key", {}).get("status") == "measured"
                or music.get("tempo", {}).get("status") == "measured"
                or isinstance(music.get("tempo", {}).get("onsets_per_second"), (int, float))
                or bool(music.get("instrumentation", {}).get("names"))
                or isinstance(music.get("dynamics_db"), (int, float)))
    if not measured:
        return "測定情報なし"
    words = music_facts.words(music).split("、")
    if not isinstance(music.get("dynamics_db"), (int, float)):
        words = [word for word in words if word not in ("強弱はなだらか", "強弱の起伏が大きい")]
    return "、".join(words) or "測定情報なし"


def series_evidence(state_root: Path, facts: dict) -> str:
    """What LSS itself holds about the series: the user's soundtrack album titles that carry the work's title
    (their "～subtitle～"), the official site's opening text LSS fetched, and the file's description."""
    import unicodedata

    parts, title = [], unicodedata.normalize("NFKC", facts.get("work_title") or "")
    try:
        from lss.bgm import reference_store

        albums = sorted({str(r.get("album")) for r in reference_store.records(state_root)
                         if r.get("album") and title and title in unicodedata.normalize("NFKC", str(r.get("album")))})
        if albums:
            parts.append("手元のサウンドトラックのアルバム名: " + " / ".join(albums[:5]))
    except Exception:  # noqa: BLE001 - no reference library is a normal state
        pass
    if facts.get("site") and _work_info_enabled():
        try:
            from lss.bgm import work_info

            parts.append("公式サイトの冒頭: " + work_info.fetch(facts["site"], state_root)["text"][:500])
        except Exception:  # noqa: BLE001
            pass
    if facts.get("description"):
        parts.append("録画ファイルの説明欄の冒頭: " + facts["description"][:500])
    return "\n".join(parts)


def _ensure_prefix(state_root: Path, facts: dict, models) -> None:
    if not scene_names.prefix_needed(state_root, facts["work_key"]):
        return
    try:
        scene_names.set_work_prefix(state_root, facts["work_key"],
                                    models.prefix(facts["work_title"], series_evidence(state_root, facts)), basis="model")
    except (ValueError, RuntimeError, TypeError):
        scene_names.set_work_prefix(state_root, facts["work_key"], scene_names.fallback_prefix(facts["work_title"]),
                                    basis="fallback")


def _music(cue: dict) -> dict:
    try:
        start, end, rate = cue["frames"]
        return music_facts.measure(cue["path"], music_facts.run_rows(cue.get("run_path"), start, end, rate))
    except Exception:  # noqa: BLE001 - a reading that fails is simply absent
        return {}


def _published(state_root: Path, source_sha256: str) -> list[dict]:
    from mutagen.flac import FLAC

    from lss.bgm import current_best

    record = current_best.read(state_root, source_sha256)
    rows = []
    for item in ((record or {}).get("current") or {}).get("files") or []:
        path = Path(item["public_path"])
        try:
            provenance = json.loads(FLAC(path)["LSS_BGM_PROVENANCE"][0])
        except (OSError, KeyError, IndexError, ValueError):
            continue
        start = float(provenance.get("start_seconds", item.get("start_seconds") or 0.0))
        end = start + (provenance["end_frame"] - provenance["start_frame"]) / provenance["sample_rate"]
        rows.append({"path": path, "asset_id": f"bgm-cue:{item['cue_key']}", "pcm_sha256": item["pcm_sha256"],
                     "start": start, "end": end, "run_path": item.get("run_path"),
                     "frames": (provenance["start_frame"], provenance["end_frame"], provenance["sample_rate"])})
    return rows


def _naming_transcription(models, source: Path, windows: list[tuple[float, float]],
                          model_revision: dict) -> tuple[list[dict], dict | None]:
    """Heard dialogue for naming, or ``[]`` with an explicit receipt saying why none is used (never silence)."""
    # Native whisper.cpp heard "ごめん" on a speech-free counterexample and the title became "ごめんのあとで":
    # its words are not grounded, so they are not requested and never reach a prompt or naming evidence.
    if isinstance(model_revision, dict) and model_revision.get("backend") == "native-whisper.cpp-llama.cpp":
        return [], {
            "state": "ASR_UNVERIFIED",
            "reason": "native_asr_grounding_unverified",
            "windows": [],
            "scope": "transcription_unverified_not_silence",
        }

    try:
        heard_all = models.transcribe(Path(source), windows)
    except SourceIdentityMismatch:
        raise
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        # Missing transcription does not remove the independent episode/music inputs.
        # Preserve the failure explicitly; never reuse a previous successful ASR receipt.
        failure = {
            "state": "ASR_UNAVAILABLE",
            "error_type": type(exc).__name__,
            "windows": [],
            "scope": "transcription_failure_not_silence",
        }
        reason_code = getattr(exc, "reason_code", None)
        if reason_code:
            # A named environment cause (e.g. a non-ASCII native path) rather than an anonymous failure.
            failure["reason"] = reason_code
        return [], failure

    return heard_all, None


_DECORATION = re.compile(r"(\s*(〜[^〜]*〜|～[^～]*～|[（(][^）)]*[）)]|[-－‐]\s*\S+\s*[-－‐]|ver\.?\s*\S*|その\d+))+\s*$",
                         re.IGNORECASE)


def clash_key(name: str) -> str:
    """A name without trailing version decorations: ``X 〜改訂版〜``, ``X (Ver.2)`` or ``X その2`` clash with ``X``.

    Such a decoration claims a version relation between two recordings that LSS never observed.
    """
    return scene_names._name_key(_DECORATION.sub("", str(name)) or str(name))


def _work_names(data: dict, work_key: str) -> dict[str, str]:
    """Clash keys of the work's fixed recording names -> the recording key that holds each."""
    cues = data.get("cues") or {}
    return {clash_key(r["name"]): key for key, r in (data.get("recordings") or {}).items()
            if r.get("name") and (cues.get(r.get("first_asset")) or {}).get("work_key") == work_key}


DISTINCT_NAME_ASKS = 3


def _distinct_name(models, prompt: str, order: dict, sound: str, taken: dict[str, str],
                   recording: str) -> tuple[str, dict]:
    """Ask again, up to ``DISTINCT_NAME_ASKS`` times, while the proposed name is another recording's in this work.

    Listing the names already given in the prompt does not stop the local model from repeating one (installed
    ep1, 2026-11-02: one name proposed for 9 of 14 different recordings, settled as その2..その9). Each re-ask
    lists every refused name. A version decoration is no escape (the model answered 〜改訂版〜/〜再編〜 for 3 of 14).
    When every answer clashes or one is unusable, the first order stays and ``settle`` numbers it.
    """
    name = order.get("name")
    holder = taken.get(clash_key(name)) if isinstance(name, str) and name else None
    if holder is None or holder == recording:
        return prompt, order
    refused = [name]
    for _ in range(DISTINCT_NAME_ASKS):
        retry_prompt = (prompt + f"\n\n{'、'.join(f'「{n}」' for n in refused)}はこの作品の別の曲の曲名として既に使われている。"
                        f"この曲の音の特徴（{sound}）と曲調に合わせ、既に付けた曲名と主な語が重ならない別の曲名を付け直す。"
                        "版・編・ver・その・番号などの付け足しで区別しない（別の曲との関係を主張することになる）。"
                        "他の欄の条件は同じ。出力は同じ形式のJSONのみ。")
        try:
            retry = models.order(retry_prompt)
        except (ValueError, RuntimeError):
            return prompt, order
        if not isinstance(retry, dict) or retry.get("name_basis") != order.get("name_basis") or any(
                retry.get(field) is not None and (not isinstance(retry.get(field), str) or len(retry[field]) > 200)
                for field in ("use", "direction", "category", "name")) or not retry.get("name"):
            return prompt, order
        if taken.get(clash_key(retry["name"]), recording) == recording:
            return retry_prompt, retry
        refused.append(retry["name"])
    return prompt, order


def name_source(state_root: Path, source: Path, source_sha256: str, *, models=None, validate_source=None) -> dict:
    """Name every published cue of ``source`` that has no scene record for its audio yet; refresh the work."""
    from lss.bgm import cue_memory

    state_root = Path(state_root)
    if validate_source is not None:
        validate_source()
    cues = _published(state_root, source_sha256)
    facts = scene_names.source_facts(Path(source))
    if validate_source is not None:
        validate_source()
    scene_names.rekey_source(state_root, source_sha256, facts)
    try:
        memory = cue_memory.read(state_root)
    except Exception:  # noqa: BLE001 - names work without the memory (a single occurrence is first-class)
        memory = None
    data = scene_names.read(state_root)
    candidates, copied = [], 0
    for cue in cues:
        existing = scene_names.cue_record(state_root, cue["asset_id"], cue["pcm_sha256"])
        found = scene_names.fixed_recording(data, cue["asset_id"], memory)
        stale = scene_names.needs_renaming(existing)
        if not stale and existing is not None and (existing.get("name") or existing.get("attempt_status") == "fixed_name"
                                                   or (existing.get("model") or {}).get("id") == "fixed_name_of_recording"):
            continue
        if found is not None and not stale:
            key, fixed = found
            # A fixed recording name always wins over retries of a different unresolved occurrence.
            cue["music"] = _music(cue)
            use, _, direction = (fixed.get("order") or "").rpartition(" ")
            scene_names.record_cue(state_root, cue["asset_id"], pcm_sha256=cue["pcm_sha256"], source_sha256=source_sha256,
                                   facts=facts, start_seconds=cue["start"], end_seconds=cue["end"], heard=[],
                                   order={"use": use or None, "direction": direction or None,
                                          "category": fixed.get("category"), "name": None},
                                   model={"id": "fixed_name_of_recording", "recording": key}, music=cue["music"])
            copied += 1
        else:
            candidates.append(cue)
    synopsis = None
    if candidates and _work_info_enabled():
        # Published facts LSS fetches itself (official site the file names), optional: None offline or refused.
        try:
            from lss.bgm import work_info

            synopsis = work_info.naming_context(facts.get("description") or "", facts.get("episode"),
                                                facts.get("work_title") or "", state_root)
        except Exception:  # noqa: BLE001 - published facts never block naming
            synopsis = None
        if synopsis is not None:
            # Public vocabulary is not a file cast list or an observation of this cue.
            facts = {**facts, "published_name_sources": [synopsis]}
    model_revision = _current_model_revision(models)
    now = _retry_clock()
    todo = []
    for cue in candidates:
        cue["music"] = _music(cue)
        existing = scene_names.cue_record(state_root, cue["asset_id"], cue["pcm_sha256"])
        retry_key = _retry_key(state_root, source_sha256, cue, model_revision, facts)
        if (existing is None or scene_names.needs_renaming(existing) or
                (_has_retry_budget(existing, retry_key) and _attempt_is_due(existing, retry_key, now))):
            todo.append(cue)
    named = []
    if todo:
        if models is None:
            from lss.bgm import scene_runtime

            models = scene_runtime.create_models()
        try:
            model_revision = models.revision()
            windows = [(max(0.0, c["start"] - BEFORE), c["end"]) for c in todo]
            heard_all, transcription_failure = _naming_transcription(models, Path(source), windows, model_revision)
            if validate_source is not None:
                validate_source()
            taken = _work_names(data, facts["work_key"])
            for cue in todo:
                scene_stop.check()  # between cues: every cue already recorded stays recorded
                heard = [h for h in heard_all if h["end"] > cue["start"] - BEFORE and h["start"] < cue["end"]]
                before = [h["text"] for h in heard if h["end"] <= cue["start"]]
                during = [h["text"] for h in heard if h["end"] > cue["start"]]
                sound = _measured_music_description(cue["music"])
                outside = scene_names.outside_episode(cue["start"], cue["end"], during)
                prompt = None
                occurrence = scene_names.occurrence_lines(heard, cue["start"], cue["end"])
                if outside:
                    order = {"use": None, "direction": None, "category": "本編外" if outside else None, "name": None}
                else:
                    try:
                        prompt = order_prompt(facts, before, during, sound, named,
                                              (synopsis or {}).get("text"), context_only=not occurrence,
                                              transcription_unavailable=transcription_failure is not None)
                        order = models.order(prompt)
                        if any(order.get(field) is not None and (not isinstance(order.get(field), str)
                                                                 or len(order[field]) > 200)
                               for field in ("use", "direction", "category", "name")):
                            # The record refuses such a field; one cue's unusable answer must not end the job.
                            raise scene_names.SceneNamesError("order_field_invalid")
                        # Older adapters/fakes that return a scene order remain on the strict scene guard path.
                        if not order.get("name_basis") and (order.get("use") or order.get("evidence")):
                            order = {**order, "name_basis": "scene"}
                        prompt, order = _distinct_name(models, prompt, order, sound, taken,
                                                       scene_names.recording_key(cue["asset_id"], memory))
                    except (ValueError, RuntimeError) as exc:
                        order = {"name_basis": "context_music" if not occurrence else "scene",
                                 "use": None, "direction": None, "category": None, "name": None,
                                 "error": type(exc).__name__}
                transcription_receipt = (transcription_failure if transcription_failure is not None else
                                         getattr(models, "last_transcription_receipt", None))
                cue_receipt = None
                transcription_window_ids = []
                if isinstance(transcription_receipt, dict):
                    cue_windows = [row for row in transcription_receipt.get("windows", [])
                                   if row.get("source_end_seconds", 0) > cue["start"] - BEFORE
                                   and row.get("source_start_seconds", float("inf")) < cue["end"]]
                    transcription_window_ids = [row["window_id"] for row in cue_windows]
                    cue_receipt = {**transcription_receipt, "windows": cue_windows}
                    if hasattr(models, "cadence_receipt"):
                        cue_receipt["execution_cadence"] = models.cadence_receipt()
                attempt_identity = _attempt_identity(state_root, source_sha256, cue, model_revision,
                                                     facts, synopsis, heard, prompt, sound, named,
                                                     transcription_receipt=cue_receipt,
                                                     transcription_window_ids=transcription_window_ids)
                entry = scene_names.record_cue(state_root, cue["asset_id"], pcm_sha256=cue["pcm_sha256"],
                                               source_sha256=source_sha256, facts=facts, start_seconds=cue["start"],
                                               end_seconds=cue["end"], heard=heard, order=order, model=model_revision,
                                               music=cue["music"],
                                               name_basis_evidence=(scene_names.context_music_evidence(
                                                   facts, heard, cue["music"], sound)
                                                                    if order.get("name_basis") == "context_music" else None),
                                               measured_music_description=sound,
                                               attempt_identity=attempt_identity,
                                               attempted_at=now, completed_at=_retry_clock())
                named.append(entry["name"])
                if entry["name"]:
                    taken.setdefault(clash_key(entry["name"]),
                                     scene_names.recording_key(cue["asset_id"], memory))
            _ensure_prefix(state_root, facts, models)
        finally:
            models.release()
    elif scene_names.prefix_needed(state_root, facts["work_key"]):
        if models is None:
            from lss.bgm import scene_runtime

            models = scene_runtime.create_models()
        try:
            _ensure_prefix(state_root, facts, models)
        finally:
            models.release()
    scene_names.settle(state_root, memory)
    return {"named": sum(1 for n in named if n), "unnamed": sum(1 for n in named if not n), "copied": copied,
            "refresh": refresh_work(state_root, facts["work_key"], memory),
            "execution_cadence": (models.cadence_receipt() if models is not None
                                  and hasattr(models, "cadence_receipt") else
                                  {"profile": "legacy", "rest_seconds": 0, "events": []})}


class SourceIdentityMismatch(RuntimeError):
    """The media at the naming path no longer matches its published source identity."""


def _validate_source_identity(source: Path, expected: str) -> None:
    source = Path(source).resolve()
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SourceIdentityMismatch("source_unavailable") from exc
    if digest.hexdigest() != str(expected).lower():
        raise SourceIdentityMismatch("source_identity_mismatch")


def name_source_checked(state_root: Path, source: Path, source_sha256: str, *, models=None) -> dict:
    """Public worker entry: bind all scene observations to the published source bytes."""
    source = Path(source).resolve()
    return name_source(state_root, source, source_sha256, models=models,
                       validate_source=lambda: _validate_source_identity(source, source_sha256))


def _refresh_failure(asset_id: str, exc: BaseException, member_path, path) -> dict:
    """A pending refresh entry that keeps the failure's own message (a type name alone was undiagnosable)."""
    result = {"asset_id": asset_id, "error": type(exc).__name__}
    if member_path and Path(member_path) != path:
        result["memory_path_ignored"] = str(member_path)
    msg = " ".join(str(exc).split())
    if msg:
        result["message"] = msg[:300]
    return result


def refresh_work(state_root: Path, work_key: str, memory: dict | None) -> dict:
    """Refresh the listening presentation of every published file of the work whose scene tags/name changed."""
    from lss.bgm import cue_memory, current_best, listening, write_targets

    data = scene_names.read(state_root)
    views = {}
    if memory is not None:
        try:
            views = {view["asset_id"]: view for view in cue_memory._views(memory).values()}
        except Exception:  # noqa: BLE001
            views = {}
    done, pending, current = 0, [], 0
    published: dict[str, dict] = {}
    for asset_id, cue in data["cues"].items():
        if cue["work_key"] != work_key:
            continue
        # The write target is the CurrentBest record's public_path for this cue and nothing else. Paths the cue
        # memory remembers (a library the state was copied from, a moved file) are never written.
        sha = cue["source_sha256"]
        if sha not in published:
            published[sha] = current_best.published_files(state_root, sha)
        row = published[sha].get(asset_id.removeprefix("bgm-cue:"))
        if row is None:
            pending.append({"asset_id": asset_id, "error": "not_a_current_published_path",
                            "reason": "no_current_best_file_for_cue"})
            continue
        path = Path(row["public_path"])
        member_path = (((memory or {}).get("members") or {}).get(asset_id) or {}).get("path")
        if not path.is_file():
            pending.append({"asset_id": asset_id, "error": "path_unknown"})
            continue
        try:
            # Containment: the row must name this state's own, still-identical public file (`write_targets`).
            path = write_targets.checked(state_root, row, asset_id.removeprefix("bgm-cue:"))
        except write_targets.WriteTargetRefused as refused:
            pending.append({"asset_id": asset_id, "error": "not_a_current_published_path",
                            "reason": refused.reason})
            continue
        if row.get("pcm_sha256") not in (None, cue["pcm_sha256"]):
            pending.append({"asset_id": asset_id, "error": "not_a_current_published_path",
                            "reason": "current_best_pcm_differs_from_scene_record"})
            continue
        try:
            if listening.current([path], state_root):
                current += 1
                continue
            listening.finish(path, state_root, expected_pcm=cue["pcm_sha256"], recurrence=views.get(asset_id))
            done += 1
        except Exception as exc:  # noqa: BLE001 - another file's presentation never fails this run
            pending.append(_refresh_failure(asset_id, exc, member_path, path))
    return {"refreshed": done, "already_current": current, "pending": pending}


def _work_info_enabled() -> bool:
    import os

    value = os.environ.get("LSS_WORK_INFO")
    return value is None or value.strip().lower() not in ("0", "off", "false", "no")


def needs_naming(state_root: Path, source_sha256: str, *, source: Path | None = None,
                 models=None, now: float | None = None, strict: bool = False) -> bool:
    """Whether a cue is new or an unresolved attempt has changed inputs / reached retry backoff."""
    try:
        revision = _current_model_revision(models)
        now = _retry_clock() if now is None else float(now)
        facts = None
        facts_loaded = source is None
        for cue in _published(Path(state_root), source_sha256):
            entry = scene_names.cue_record(state_root, cue["asset_id"], cue["pcm_sha256"])
            if entry is None:
                return True
            if (entry.get("name") or entry.get("attempt_status") == "fixed_name"
                    or (entry.get("model") or {}).get("id") == "fixed_name_of_recording"):
                continue
            if not facts_loaded:
                facts = scene_names.source_facts(Path(source))
                facts_loaded = True
            identity = (entry.get("attempts") or [{}])[-1].get("identity") or entry.get("attempt_identity") or {}
            current_facts = facts if facts is not None else identity.get("effective_facts")
            key = _retry_key(state_root, source_sha256, cue, revision, current_facts,
                             effective_facts_sha256=None if current_facts is not None else identity.get("effective_facts_sha256"))
            if _has_retry_budget(entry, key) and _attempt_is_due(entry, key, now):
                return True
        return False
    except Exception:  # noqa: BLE001 - a failed check never blocks the BGM result
        if strict:
            raise
        return False


def available() -> bool:
    """A local scene-model backend can run here: the native engines + weights (also in the packaged app) or, from
    source only, the Transformers stack (``scene_runtime``)."""
    from lss.bgm import scene_runtime

    return scene_runtime.available()


def start_in_background(state_root: Path, source: Path, source_sha256: str) -> dict:
    """Name the source's cues in a separate process that outlives the window; never blocks the BGM result."""
    import os
    import subprocess
    import sys

    from lss.paths import is_frozen

    from lss.bgm import scene_runtime

    if not available() and not scene_runtime.bootstrap_supported():
        return {"state": "not_available"}
    needs_now = needs_naming(state_root, source_sha256, source=source)
    pending = _pending_retry_with_budget(state_root, source_sha256, source=source)
    terminal = _next_transient_retry_after(state_root, source_sha256)
    if not needs_now and pending is None:
        # Nothing of this naming is left to do: the requests made so far are covered. The caller is the BGM job
        # itself, after its publication, so its own lock does not mean more cues are coming.
        _end_requests(state_root, source_sha256, Path(source), inside_bgm_job=True)
        return ({"state": "retry_pending", "retry_after": terminal}
                if terminal is not None else {"state": "current"})
    _record_request(state_root, Path(source).resolve(), source_sha256)
    # A child may already be sleeping through the retry delay. The OS lock check is
    # authoritative across the development and frozen command routes.
    with _background_worker_lease(state_root, source_sha256) as acquired:
        if not acquired:
            return {"state": "already_running"}
    log = Path(state_root) / scene_names.DIRECTORY / "jobs" / f"{source_sha256[:16]}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # DETACHED_PROCESS overrides CREATE_NO_WINDOW on Windows. Redirected stdio and the ordinary
    # subprocess lifetime already let this worker survive normal parent exit without a console.
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    state_root = Path(state_root).resolve()
    source = Path(source).resolve()
    if is_frozen():
        # A frozen sys.executable is LSS.exe and cannot run ``-m``; enter through lss_gui's
        # dedicated one-shot command instead. It exits after naming, so a failed job also releases
        # its process-owned GPU lease back to the regular inference worker.
        command = [sys.executable, "--scene-naming", str(state_root), str(source), source_sha256]
    else:
        command = [sys.executable, "-m", "lss.bgm.scene_job", str(state_root), str(source), source_sha256]
    with open(log, "ab") as stream:
        process = subprocess.Popen(command,
                                   stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   creationflags=flags, cwd=str(Path(__file__).resolve().parents[2]))
    return {"state": "started", "pid": process.pid, "log": str(log)}


def _recovery_log_path(state_root: Path) -> Path:
    return Path(state_root) / scene_names.DIRECTORY / "jobs" / "startup-recovery.log"


def _recovery_log(state_root: Path, event: dict) -> None:
    path = _recovery_log_path(state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": _retry_clock(), **event}, ensure_ascii=False, sort_keys=True) + "\n")


def _current_best_run_sources(state_root: Path) -> tuple[dict[str, list[Path]], list[dict]]:
    """Resolve only current published cue run records to their recorded source paths."""
    from lss.bgm import current_best

    root = Path(state_root)
    folder = root / current_best.DIRECTORY
    sources: dict[str, list[Path]] = {}
    diagnostics = []
    for record_file in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        if record_file.name.endswith(".switch.json"):
            continue
        digest = record_file.stem.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            diagnostics.append({"state": "skipped", "reason": "invalid_current_best_filename",
                                "record": str(record_file)})
            continue
        try:
            record = current_best.read(root, digest)
        except Exception as exc:  # noqa: BLE001 - one corrupt record cannot hide other published sources
            diagnostics.append({"state": "skipped", "reason": "current_best_unreadable",
                                "record": str(record_file), "error_type": type(exc).__name__})
            continue
        current_files = ((record or {}).get("current") or {}).get("files") or []
        for item in current_files:
            run_path_value = item.get("run_path")
            if not isinstance(run_path_value, str) or not run_path_value:
                diagnostics.append({"state": "skipped", "reason": "current_run_path_missing",
                                    "source_sha256": digest})
                continue
            run_path = Path(run_path_value)
            try:
                run = json.loads(run_path.read_text(encoding="utf-8"))
                source = run.get("source") or {}
                if (run.get("type") != current_best.RUN_TYPE
                        or run.get("status") != current_best.PUBLISHED_STATUS
                        or source.get("sha256") != digest
                        or not isinstance(source.get("path"), str) or not source.get("path")):
                    raise ValueError("current_run_identity_invalid")
                source_path = Path(source["path"])
                if not source_path.is_absolute():
                    raise ValueError("current_run_source_path_not_absolute")
            except Exception as exc:  # noqa: BLE001 - never guess a replacement path or alias
                diagnostics.append({"state": "skipped", "reason": "current_run_unreadable",
                                    "source_sha256": digest, "run_path": str(run_path),
                                    "error_type": type(exc).__name__, "message": str(exc)})
                continue
            rows = sources.setdefault(digest, [])
            if source_path not in rows:
                rows.append(source_path)
    return sources, diagnostics


def _scene_models_cached() -> bool:
    """Whether the currently selected backend is already ready without acquisition."""
    from lss.bgm import scene_runtime

    return scene_runtime.models_cached()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recovery_attempt_state(state_root: Path, source_sha256: str) -> str:
    """Fingerprint current cue attempt records to detect progress without retrying a no-op worker."""
    rows = []
    for cue in _published(state_root, source_sha256):
        entry = scene_names.cue_record(state_root, cue["asset_id"], cue["pcm_sha256"])
        if entry is None:
            rows.append((cue["asset_id"], cue["pcm_sha256"], None))
        else:
            rows.append((cue["asset_id"], cue["pcm_sha256"], entry.get("attempts") or [],
                         entry.get("attempt_status"), entry.get("retry_after")))
    return json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _eligible_recovery_source(state_root: Path, source_sha256: str, paths: list[Path], *, now: float,
                              verified_paths: dict | None = None) -> dict | None:
    """Check current cue eligibility before invoking the worker's final source hash guard.

    Only a source whose naming a user-started job requested and did not finish is eligible: a newer model, policy
    or public information alone makes a reprocess candidate, never launch-time work on a source.
    """
    requests = _requests(state_root, source_sha256)
    if not requests:
        return None
    # The user may have named the source at a new place; try the requested places before older runs' paths.
    requested = [str(record["source"]) for _, record in reversed(requests)   # newest request first
                 if isinstance(record.get("source"), str) and record["source"]]
    seen = set()
    ordered = []
    for path in [Path(text) for text in requested] + list(paths):
        if str(path).casefold() not in seen:
            seen.add(str(path).casefold())
            ordered.append(path)
    paths = ordered
    published = _published(state_root, source_sha256)
    if not published:
        return None
    rejected_paths = []
    for source in paths:
        if not source.is_file():
            rejected_paths.append({"path": str(source), "reason": "recorded_source_missing"})
            continue
        from lss.bgm import scene_runtime

        if not _scene_models_cached() and not scene_runtime.bootstrap_supported():
            # No hashing or model work without a usable backend; the request stays for a later launch.
            return {"skip_reason": "scene_naming_models_not_cached", "source": source,
                    "immediate": False, "retry_after": None}
        try:
            stat = source.stat()
            cache_key = (str(source.resolve()).casefold(), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            actual_sha256 = (verified_paths or {}).get(cache_key)
            if actual_sha256 is None:
                actual_sha256 = _sha256_file(source)
                if verified_paths is not None:
                    verified_paths[cache_key] = actual_sha256
        except OSError as exc:
            rejected_paths.append({"path": str(source), "reason": "recorded_source_unreadable",
                                   "error_type": type(exc).__name__})
            continue
        if actual_sha256 != source_sha256:
            rejected_paths.append({"path": str(source), "reason": "recorded_source_identity_mismatch",
                                   "actual_sha256": actual_sha256})
            continue
        # Only the verified original decides whether anything of the request is left.
        try:
            immediate = needs_naming(state_root, source_sha256, source=source, now=now, strict=True)
            retry_after = None if immediate else _pending_retry_with_budget(state_root, source_sha256,
                                                                            source=source, strict=True)
        except Exception:  # noqa: BLE001 - unreadable naming state: keep the request for a later launch
            return None
        if not immediate and retry_after is None:
            # The user's requested naming has nothing left: end the requests so a later model or policy change
            # does not turn them into launch-time work.
            _end_requests(state_root, source_sha256, source)
            return None
        return {"source": source, "immediate": immediate or retry_after is None or retry_after <= now,
                "retry_after": retry_after, "path_rejections": rejected_paths}
    if rejected_paths:
        raise ValueError(json.dumps({"source_sha256": source_sha256, "paths": rejected_paths}, ensure_ascii=False))
    return None


def run_current_best_recovery(state_root: Path, *, wait=None, clock=None) -> dict:
    """Resume current published cue naming serially; startup discovery itself never loads a model."""
    state_root = Path(state_root).resolve()
    clock = clock or _retry_clock
    scene_stop.enable()
    sources, diagnostics = _current_best_run_sources(state_root)
    if not sources:
        for event in diagnostics:
            _recovery_log(state_root, event)
        return {"state": "current", "sources": 0, "started": 0, "skipped": len(diagnostics)}
    with _background_worker_lease(state_root, None) as acquired:
        if not acquired:
            _recovery_log(state_root, {"state": "already_running"})
            return {"state": "already_running", "sources": len(sources), "started": 0,
                    "skipped": len(diagnostics)}
        _recovery_log(state_root, {"state": "started", "sources": len(sources)})
        outcomes = []
        local_attempts = {}
        last_attempt_state = {}
        diagnosed = set()
        diagnosed_paths = set()
        verified_paths = {}
        # Drain one pass of all currently eligible sources before honoring any future backoff wait.
        # Re-scan after each pass so retries that become due during a long immediate batch are picked up.
        while True:
            immediate = []
            for digest, paths in sources.items():
                try:
                    eligible = _eligible_recovery_source(state_root, digest, paths, now=clock(),
                                                         verified_paths=verified_paths)
                    if eligible is None:
                        if digest not in diagnosed:
                            diagnostics.append({"state": "skipped", "reason": "no_pending_current_cue",
                                                "source_sha256": digest})
                            diagnosed.add(digest)
                    elif eligible.get("skip_reason"):
                        if digest not in diagnosed:
                            diagnostics.append({"state": "skipped", "reason": eligible["skip_reason"],
                                                "source_sha256": digest})
                            diagnosed.add(digest)
                    if eligible is not None and eligible.get("path_rejections") and digest not in diagnosed_paths:
                        diagnostics.append({"state": "recorded_path_rejected", "source_sha256": digest,
                                            "paths": eligible["path_rejections"]})
                        diagnosed_paths.add(digest)
                    if (eligible is not None and not eligible.get("skip_reason") and eligible["immediate"]
                          and local_attempts.get(digest, 0) < MAX_BACKGROUND_ATTEMPTS
                          and _recovery_attempt_state(state_root, digest) != last_attempt_state.get(digest)):
                        immediate.append((digest, eligible))
                except Exception as exc:  # noqa: BLE001 - record source-boundary failures, then continue
                    if digest not in diagnosed:
                        event = {"state": "skipped", "reason": "source_validation_failed",
                                 "source_sha256": digest, "error_type": type(exc).__name__,
                                 "message": str(exc)}
                        try:
                            detail = json.loads(str(exc))
                            if detail.get("source_sha256") == digest and isinstance(detail.get("paths"), list):
                                event["paths"] = detail["paths"]
                        except (json.JSONDecodeError, TypeError):
                            pass
                        diagnostics.append(event)
                        diagnosed.add(digest)
            if not immediate:
                break
            for digest, item in immediate:
                last_attempt_state[digest] = _recovery_attempt_state(state_root, digest)
                local_attempts[digest] = local_attempts.get(digest, 0) + 1
                _recovery_log(state_root, {"state": "source_started", "source_sha256": digest})
                try:
                    result = run_background_worker(state_root, item["source"], digest, wait=wait, clock=clock,
                                                   wait_for_retries=False, require_request=True)
                    outcomes.append({"source_sha256": digest, "result": result})
                except Exception as exc:  # noqa: BLE001 - a source failure is durable and isolated
                    outcomes.append({"source_sha256": digest, "state": "failed",
                                     "error_type": type(exc).__name__, "message": str(exc)})
                if scene_stop.stop_requested():
                    return _recovery_stopped(state_root, sources, diagnostics, outcomes)
            for digest, item in immediate:
                if item.get("path_rejections") and digest not in diagnosed_paths:
                    diagnostics.append({"state": "recorded_path_rejected", "source_sha256": digest,
                                        "paths": item["path_rejections"]})
                    diagnosed_paths.add(digest)
        # All current ready cues have had one attempt; future retry waits now run earliest-first.
        future = []
        for digest, paths in sources.items():
            try:
                eligible = _eligible_recovery_source(state_root, digest, paths, now=clock(),
                                                     verified_paths=verified_paths)
                if (eligible is not None and not eligible.get("skip_reason") and not eligible["immediate"]
                        and eligible["retry_after"] is not None):
                    future.append((eligible["retry_after"], digest, eligible))
            except Exception as exc:  # noqa: BLE001
                event = {"state": "skipped", "reason": "source_revalidation_failed",
                         "source_sha256": digest, "error_type": type(exc).__name__, "message": str(exc)}
                try:
                    detail = json.loads(str(exc))
                    if detail.get("source_sha256") == digest and isinstance(detail.get("paths"), list):
                        event["paths"] = detail["paths"]
                except (json.JSONDecodeError, TypeError):
                    pass
                diagnostics.append(event)
        for _, digest, item in sorted(future, key=lambda row: (row[0], row[1])):
            _recovery_log(state_root, {"state": "source_started", "source_sha256": digest,
                                       "retry_after": item["retry_after"]})
            try:
                result = run_background_worker(state_root, item["source"], digest, wait=wait, clock=clock, require_request=True)
                outcomes.append({"source_sha256": digest, "result": result})
            except Exception as exc:  # noqa: BLE001
                outcomes.append({"source_sha256": digest, "state": "failed",
                                 "error_type": type(exc).__name__, "message": str(exc)})
            if scene_stop.stop_requested():
                return _recovery_stopped(state_root, sources, diagnostics, outcomes)
        for event in diagnostics + outcomes:
            _recovery_log(state_root, event)
        return {"state": "recovery_complete", "sources": len(sources), "started": len(outcomes),
                "skipped": len(diagnostics), "outcomes": outcomes}


def _recovery_stopped(state_root: Path, sources: dict, diagnostics: list, outcomes: list) -> dict:
    for event in diagnostics + outcomes + [{"state": "stopped", "reason": "scene_worker_stop_requested"}]:
        _recovery_log(state_root, event)
    return {"state": "stopped", "sources": len(sources), "started": len(outcomes),
            "skipped": len(diagnostics), "outcomes": outcomes}


def start_recovery_in_background(state_root: Path) -> dict:
    """Launch one detached coordinator when CurrentBest records exist; no source audio is hashed here."""
    import os
    import subprocess
    import sys

    from lss.bgm import current_best
    from lss.paths import is_frozen

    state_root = Path(state_root).resolve()
    folder = state_root / current_best.DIRECTORY
    if not folder.is_dir() or not any(path.is_file() and not path.name.endswith(".switch.json")
                                      for path in folder.glob("*.json")):
        return {"state": "current"}
    from lss.bgm import scene_runtime

    if not available() and not scene_runtime.bootstrap_supported():
        return {"state": "not_available"}
    with _background_worker_lease(state_root, None) as acquired:
        if not acquired:
            return {"state": "already_running"}
    log = _recovery_log_path(state_root)
    log.parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    if is_frozen():
        command = [sys.executable, "--scene-naming-recovery", str(state_root)]
    else:
        command = [sys.executable, "-m", "lss.bgm.scene_job", "--recover", str(state_root)]
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, creationflags=flags,
                                   cwd=str(Path(__file__).resolve().parents[2]))
    return {"state": "started", "pid": process.pid, "log": str(log)}


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--recover":
        result = run_current_best_recovery(Path(sys.argv[2]))
    else:
        result = run_background_worker(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
    print(json.dumps(result, ensure_ascii=False))
