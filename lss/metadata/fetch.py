"""LSS's own polite, cached reads of public metadata sources (raw responses, not page text).

Every response is stored once under ``<state>/metadata-sources/responses`` with its URL, time and hash, so
decisions can be traced and repeated offline. robots.txt is honoured for LSS's own user agent and requests are
paced through the shared acquisition courtesy. A documented public API (MusicBrainz WS/2, iTunes Search) is
used under its own published terms instead of robots.txt, which addresses crawlers of web pages (MusicBrainz's
robots.txt disallows ``/ws`` for crawlers while its API terms invite applications at one request per second). A miss (404, refused, offline) is remembered for ``MISS_TTL`` so a
work with no soundtrack yet is not re-queried on every run; a later run after that period asks again.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "LSS-metadata/1 (local personal music library tool; reads public metadata once and caches it)"
DIRECTORY = "metadata-sources"
TIMEOUT = 30
MAX_BYTES = 8 * 1024 * 1024
MISS_TTL = 30 * 86400


class Unavailable(RuntimeError):
    """The response cannot be read now (offline, refused by robots.txt, not found, error)."""


def permanent(exc: Unavailable) -> bool:
    """The source answered (not found, refused, robots.txt), as opposed to could-not-ask (offline, 5xx, timeout).

    Only an answer may read as "there is nothing"; could-not-ask leaves the fact unknown."""
    return bool(getattr(exc, "refused", False)) or getattr(exc, "status", None) in (401, 403, 404, 410)


def _root(state_root: Path) -> Path:
    return Path(state_root) / DIRECTORY


def _key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def http(url: str, accept: str = "*/*") -> str:
    """Plain GET with LSS's own identity; raises Unavailable (with ``status`` when the server answered)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept,
                                                   "Accept-Language": "ja,en;q=0.5"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 - public https only
            raw = response.read(MAX_BYTES + 1)
            charset = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        error = Unavailable(f"{url}: HTTP {exc.code}")
        error.status = exc.code
        raise error from exc
    except (OSError, ValueError) as exc:
        raise Unavailable(f"{url}: {type(exc).__name__}") from exc
    if len(raw) > MAX_BYTES:
        raise Unavailable(f"{url}: too large")
    return raw.decode(charset, errors="replace")


def _allowed(url: str, state_root: Path, opener) -> bool:
    import urllib.robotparser

    parts = urllib.parse.urlsplit(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    cached = _root(state_root) / "robots" / (_key(robots_url) + ".txt")
    try:
        if cached.is_file() and time.time() - cached.stat().st_mtime < 7 * 86400:
            body = cached.read_text(encoding="utf-8")
        else:
            body = opener(robots_url)
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(body, encoding="utf-8")
    except Unavailable as exc:
        if getattr(exc, "status", None) in (401, 403):
            return False   # the site refuses this client outright
        body = ""          # no robots.txt: the site states no restriction
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(body.splitlines())
    return parser.can_fetch(USER_AGENT, url)


def get(url: str, state_root: Path, *, accept: str = "*/*", opener=None, courtesy=None,
        purpose: str = "metadata", api: bool = False) -> dict:
    """``{"url", "fetched_at", "body", "sha256"}`` for ``url``, read once and cached. Raises Unavailable."""
    if not url.startswith("https://"):
        raise Unavailable("https only")
    root = _root(state_root)
    store = root / "responses" / (_key(url) + ".json")
    if store.is_file():
        return json.loads(store.read_text(encoding="utf-8"))
    miss = root / "misses" / (_key(url) + ".json")
    if miss.is_file():
        try:
            recorded = json.loads(miss.read_text(encoding="utf-8"))
            if time.time() - recorded["at"] < MISS_TTL:
                remembered = Unavailable(f"{url}: {recorded['reason']} (remembered)")
                status = str(recorded["reason"]).removeprefix("HTTP ")
                remembered.status = int(status) if status.isdigit() else None
                remembered.refused = recorded["reason"] == "robots"
                raise remembered
        except (OSError, ValueError, KeyError, TypeError):
            pass
    opener = opener or (lambda target: http(target, accept))

    def remember(reason: str) -> None:
        miss.parent.mkdir(parents=True, exist_ok=True)
        miss.write_text(json.dumps({"url": url, "at": time.time(), "reason": reason}), encoding="utf-8")

    if not api and not _allowed(url, state_root, opener):
        remember("robots")
        refused = Unavailable(f"{url}: robots.txt does not allow {USER_AGENT.split('/')[0]}")
        refused.refused = True
        raise refused
    if courtesy is None:
        from lss.acquisition import courtesy as courtesy_module

        courtesy = courtesy_module.shared(Path(state_root))
    with courtesy.slot(url, purpose=purpose):
        try:
            body = opener(url)
            courtesy.report(url)
        except Unavailable as exc:
            status = getattr(exc, "status", None)
            courtesy.report(url, None if status in (403, 404, 410) else "transient")
            if status in (403, 404, 410):
                remember(f"HTTP {status}")
            raise
    entry = {"url": url, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "user_agent": USER_AGENT,
             "body": body, "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}
    store.parent.mkdir(parents=True, exist_ok=True)
    tmp = store.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    tmp.replace(store)
    return entry


def get_json(url: str, state_root: Path, **kwargs) -> tuple[dict, dict]:
    """(parsed JSON, cache entry). Raises Unavailable, including for a body that is not JSON."""
    entry = get(url, state_root, accept="application/json", **kwargs)
    try:
        return json.loads(entry["body"]), entry
    except ValueError as exc:
        raise Unavailable(f"{url}: not JSON") from exc
