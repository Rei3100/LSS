"""iTunes / Apple Music catalogue (lowest-ranked source): albums and tracks as the store wrote them.

Store titles carry label-supplied mistakes and added suffixes (`` (Instrumental)``); they are kept raw here and the
resolver decides. Reads go through `lss.metadata.fetch` (robots.txt, courtesy pacing, cached raw responses).
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path

from lss.metadata import fetch
from lss.metadata.model import Credit, Release, Track

SEARCH = "https://itunes.apple.com/search"
LOOKUP = "https://itunes.apple.com/lookup"
_COMMON = {"country": "jp", "lang": "ja_jp"}


def _url(base: str, **params) -> str:
    return base + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote, safe="")


def _date(value) -> str | None:
    return str(value)[:10] if value and str(value)[:4].isdigit() else None


def _album_credits(item: dict) -> tuple[Credit, ...]:
    name = (item.get("artistName") or "").strip()
    return (Credit(role="artistName", role_key="artist", name=name, names=(name,)),) if name else ()


def _release(item: dict, tracks: tuple[Track, ...], entry: dict, discs: int | None) -> Release:
    title = item.get("collectionName") or ""
    art = tuple(u for u in (item.get("artworkUrl100"),) if u)
    return Release(source="itunes", url=item.get("collectionViewUrl") or "", title=title,
                   titles={"ja": title} if title else {}, release_date=_date(item.get("releaseDate")),
                   label=None, discs=discs, tracks=tracks, credits=_album_credits(item),
                   notes=item.get("copyright") or None, related_urls=art,
                   fetched_at=entry.get("fetched_at"), raw_sha256=entry.get("sha256"))


def _track(item: dict, album_artist: str) -> Track:
    title = item.get("trackName") or ""
    millis = item.get("trackTimeMillis")
    credits = []
    composer = (item.get("composerName") or "").strip()
    if composer:
        credits.append(Credit(role="composerName", role_key="composer", name=composer, names=(composer,)))
    artist = (item.get("artistName") or "").strip()
    if artist and artist != album_artist:
        credits.append(Credit(role="artistName", role_key="artist", name=artist, names=(artist,)))
    return Track(disc=int(item.get("discNumber") or 1), number=int(item.get("trackNumber") or 0), title=title,
                 titles={"ja": title} if title else {},
                 length_seconds=round(millis / 1000.0, 3) if isinstance(millis, (int, float)) else None,
                 credits=tuple(credits))


def search_albums(term: str, state_root: Path, *, opener=None, courtesy=None) -> list[Release]:
    """Albums matching ``term`` (track lists are empty; use `album`). Offline or refused: empty list."""
    if not term or not term.strip():
        return []
    url = _url(SEARCH, term=term.strip(), **_COMMON, media="music", entity="album")
    try:
        data, entry = fetch.get_json(url, state_root, opener=opener, courtesy=courtesy, purpose="itunes", api=True)
    except fetch.Unavailable:
        return []
    found = []
    for item in data.get("results") or []:
        if item.get("wrapperType") == "collection" and item.get("collectionName"):
            found.append(_release(item, (), entry, None))
    return found


def album(collection_id: int | str, state_root: Path, *, opener=None, courtesy=None) -> Release:
    """One album with its songs. Raises `fetch.Unavailable` when it cannot be read or the id is unknown."""
    url = _url(LOOKUP, id=str(collection_id), entity="song", **_COMMON)
    data, entry = fetch.get_json(url, state_root, opener=opener, courtesy=courtesy, purpose="itunes", api=True)
    results = data.get("results") or []
    head = next((r for r in results if r.get("wrapperType") == "collection"), None)
    if head is None:
        raise fetch.Unavailable(f"{url}: no such album")
    artist = (head.get("artistName") or "").strip()
    songs = [r for r in results if r.get("wrapperType") == "track" and r.get("kind") == "song"]
    tracks = tuple(sorted((_track(r, artist) for r in songs), key=lambda t: (t.disc, t.number)))
    counts = [int(r["discCount"]) for r in songs if isinstance(r.get("discCount"), int)]
    discs = max(counts) if counts else (max((t.disc for t in tracks), default=None))
    return _release(head, tracks, entry, discs)
