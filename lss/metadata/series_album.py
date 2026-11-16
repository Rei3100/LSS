"""The series album of a source video, acquired and reconciled by LSS itself.

From the video's own container title (else its file name): work, episode and subtitle. The series is the Wikipedia
article that lists this episode (TV) or the movie article; its staff music credit and start date anchor the search
for official soundtracks on MusicBrainz (composer must match, released on or after the series start). Each found
volume is completed from VGMdb (through MusicBrainz's link, read from archived copies) and iTunes (same release by
catalog/barcode/track lengths), merged by source rank, and the volumes form one album. Every step that cannot be
read (offline, nothing published yet) leaves that part unknown; nothing is guessed.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

from lss.metadata import fetch, resolve
from lss.metadata.model import Credit

DIRECTORY = "metadata-library"
_MOVIE = re.compile(r"劇場版|映画")


def _episode(source: Path, tags: dict) -> dict | None:
    from lss.bgm import collection_metadata

    title = next((v for k, v in tags.items() if str(k).casefold() == "title" and isinstance(v, str)), None)
    for text in (title, source.stem):
        if text:
            parsed = collection_metadata.parse_episode_title(unicodedata.normalize("NFC", text))
            if parsed:
                return {**parsed, "order": collection_metadata.episode_number(parsed["episode"]),
                        "kind": "movie" if _MOVIE.search(text) else "tv"}
    return None


def _itunes_id(release) -> str | None:
    found = re.search(r"/id(\d+)", release.url or "") or re.search(r"[?&]i?d=(\d+)", release.url or "")
    return found.group(1) if found else None


def volumes(work: str, series, state_root: Path, *, opener=None, courtesy=None, strict: bool = False) -> list[dict]:
    """Official soundtrack volumes of ``series`` (oldest first), each merged across sources.

    ``strict``: a transient MusicBrainz failure raises ``fetch.Unavailable`` instead of reading as "none found"."""
    from lss.metadata.sources import itunes, musicbrainz, vgmdb

    if series is None or not series.music:
        return []
    kw = {"opener": opener, "courtesy": courtesy}
    found = musicbrainz.search_soundtracks(work, list(series.music), series.first_date, state_root, strict=strict,
                                          **kw)
    out = []
    for release in found:
        views = [release]
        for url in release.related_urls:
            album_id = vgmdb.album_id_from_url(url)
            if album_id is not None:
                try:
                    archived = vgmdb.album(album_id, state_root, **kw)
                except fetch.Unavailable:
                    archived = None
                if archived is not None and resolve.same_release(release, archived):
                    views.append(archived)
                break
        try:
            for candidate in itunes.search_albums(release.title, state_root, **kw)[:5]:
                collection = _itunes_id(candidate)
                if collection is None:
                    continue
                store = itunes.album(collection, state_root, **kw)
                if resolve.same_release(release, store):
                    views.append(store)
                    break
        except fetch.Unavailable:
            pass
        out.append(resolve.merge_volume(views))
    return out


def _key(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


ALIASES = "aliases.json"


def _aliases(state_root: Path) -> dict:
    try:
        data = json.loads((Path(state_root) / DIRECTORY / ALIASES).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _remember_alias(state_root: Path, provisional: str, key: str, series: dict) -> None:
    """The work-name key -> the identified series, so a later run that cannot ask keeps this series' key."""
    data = _aliases(state_root)
    entry = {"key": key, "series": series}
    if data.get(provisional) == entry:
        return
    data[provisional] = entry
    folder = Path(state_root) / DIRECTORY
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / (ALIASES + ".tmp")
    tmp.write_bytes(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8"))
    tmp.replace(folder / ALIASES)


def resolve_source(source: Path, tags: dict, state_root: Path, *, opener=None, courtesy=None) -> dict | None:
    """``{"key", "work", "episode", "series", "album", "complete"}`` for one source video, or None when the title
    names no work and episode.

    Could-not-ask (offline, server error) is never read as "no series" or "no soundtrack": the result is then
    incomplete and is never recorded. It keeps the series key, album and series facts of the last complete
    decision for the same work when there is one, so numbering stays in that series' ledger; otherwise the
    work-name key is used provisionally (``provisional_key``) and the series ledger adopts those members once
    the series is identified. Facts nobody could read (date, official tracks) stay unknown.
    """
    from lss.metadata.sources import wikipedia

    episode = _episode(Path(source), tags or {})
    if episode is None:
        return None
    kw = {"opener": opener, "courtesy": courtesy}
    provisional = _key(f"{episode['kind']}:{resolve.compare_key(episode['work'])}")
    known = True
    try:
        series = wikipedia.series(episode["work"], episode["subtitle"], episode["kind"], state_root,
                                  episode_number=episode["order"], strict=True, **kw)
    except fetch.Unavailable:
        series, known = None, False
    alias = None if known else _aliases(state_root).get(provisional)
    key = _key(series.url) if series is not None else (alias or {}).get("key") or provisional
    series_facts = (None if series is None else {"title": series.title, "url": series.url, "kind": series.kind,
                                                  "first_date": series.first_date, "music": list(series.music)})
    if series_facts is None and alias is not None:
        series_facts = alias.get("series")
    complete = known
    try:
        if not known:
            raise fetch.Unavailable("series article could not be read")
        vols = volumes(episode["work"], series, state_root, strict=True, **kw)
        album = resolve.series_album(episode["work"], vols)
    except fetch.Unavailable:   # could not ask: reuse the last complete decision for this series, else partial
        complete = False
        album = _load(key, state_root) or resolve.series_album(episode["work"], [])
    aired = dict(series.episode_dates).get(episode["order"]) if series is not None else None
    summary = {
        "key": key, "work": episode["work"],
        "episode": {"label": episode["episode"], "order": episode["order"], "subtitle": episode["subtitle"],
                    "date": aired},
        "series": series_facts, "album": album, "complete": complete,
        "provisional_key": provisional if key != provisional else None,
    }
    if complete:
        _record(summary, state_root)
        if series is not None and key != provisional:
            _remember_alias(state_root, provisional, key, series_facts)
    return summary


def _choice(c) -> dict:
    return {"value": c.value, "source": c.source, "raw": c.raw}


def _unchoice(d: dict) -> resolve.Choice:
    return resolve.Choice(d.get("value"), d.get("source"), dict(d.get("raw") or {}))


def _record(summary: dict, state_root: Path) -> None:
    """Keep what was decided and from which source values (internal authority; tags are projections). The record
    holds everything needed to rebuild the album exactly (`_load`)."""
    album = summary["album"]
    data = {"resolved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "complete": True, "work": summary["work"],
            "series": summary["series"], "title": album["title"], "title_choice": _choice(album["title_choice"]),
            "album_title": album["title"], "album_title_sources": album["title_choice"].raw,
            "official_discs": album["official_discs"], "unreleased_disc": album["unreleased_disc"],
            "volume_titles": {k: list(v) for k, v in (album.get("volume_titles") or {}).items()},
            "tracks": [{"disc": t.disc, "number": t.number, "title": t.title.value, "title_source": t.title.source,
                        "title_raw": t.title.raw, "title_choice": _choice(t.title), "length_seconds": t.length_seconds,
                        "m_number": t.m_number.value, "m_number_choice": _choice(t.m_number),
                        "release": t.release_title,
                        "credits": [[c.role, c.role_key, c.name] for c in t.credits],
                        "credit_items": [{"role": c.role, "role_key": c.role_key, "name": c.name,
                                          "names": list(c.names)} for c in t.credits]} for t in album["tracks"]]}
    folder = Path(state_root) / DIRECTORY / summary["key"]
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "album.json.tmp"
    tmp.write_bytes(json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8"))
    tmp.replace(folder / "album.json")


def _load(key: str, state_root: Path) -> dict | None:
    """The album of the last complete decision recorded for this series key, rebuilt exactly; None when there is
    none (or it is unreadable / from before the record held everything)."""
    try:
        data = json.loads((Path(state_root) / DIRECTORY / key / "album.json").read_text(encoding="utf-8"))
        if not data.get("complete") or "title_choice" not in data:
            return None
        tracks = [resolve.ResolvedTrack(
            disc=t["disc"], number=t["number"], title=_unchoice(t["title_choice"]),
            length_seconds=t.get("length_seconds"), m_number=_unchoice(t["m_number_choice"]),
            credits=tuple(Credit(role=c["role"], role_key=c["role_key"], name=c["name"], names=tuple(c["names"]))
                          for c in t["credit_items"]),
            release_title=t["release"]) for t in data["tracks"]]
        return {"title": data["title"], "title_choice": _unchoice(data["title_choice"]), "tracks": tracks,
                "official_discs": data["official_discs"], "unreleased_disc": data["unreleased_disc"],
                "volume_titles": {k: tuple(v) for k, v in data["volume_titles"].items()}}
    except (OSError, ValueError, KeyError, TypeError):
        return None


_CACHE: dict[tuple, dict | None] = {}
_CACHE_LIMIT = 32


def _cached(path: str, size: int, mtime_ns: int, state_root: str) -> dict | None:
    """Memoised per file version, but never an incomplete result (a later call asks again)."""
    from lss.core import media_metadata

    key = (path, size, mtime_ns, state_root)
    if key in _CACHE:
        return _CACHE[key]
    try:
        tags = media_metadata.format_tags(Path(path))
    except Exception:  # noqa: BLE001 - an unreadable container leaves the file name
        tags = {}
    try:
        result = resolve_source(Path(path), tags, Path(state_root))
    except Exception:  # noqa: BLE001 - metadata never blocks a result
        return None
    if result is None or result.get("complete", True):
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = result
    return result


def for_source(source: Path | None, state_root: Path | None) -> dict | None:
    if source is None or state_root is None:
        return None
    try:
        stat = Path(source).stat()
    except OSError:
        return None
    return _cached(str(source), stat.st_size, stat.st_mtime_ns, str(state_root))
