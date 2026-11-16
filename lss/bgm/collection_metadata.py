"""Collection presentation of an episode's BGM cues from what the source media itself says.

A finished episode usually reaches LSS as a video whose container carries only the episode title, the
uploading channel, an upload date and a video id. The channel is not the music's artist, so it never
becomes ARTIST. One work is one album, written as a soundtrack album is (user, 2026-10-21): ``<work> オリジナル・サウンドトラック``
for every episode, and the container date as DATE. Episode and playback position describe where a piece
played, not which piece it is, so they never become DISCNUMBER/TRACKNUMBER (see the contract).
ALBUMARTIST is never written. ARTIST is the work's 劇伴 composer from its published
staff list (`lss.bgm.work_credits`), only when a state root allows the cached/paced web read. Nothing unobserved
is filled in; an unparsed title keeps the file-name album and the collection's own disc.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

PROVENANCE_TAG = "LSS_COLLECTION_METADATA"
ALBUM_SUFFIX = " オリジナル・サウンドトラック"
# Tags this module owns; LSS writes them as part of the complete tag set.
OWNED_KEYS = ("ARTIST", "DATE", PROVENANCE_TAG)

_OPEN, _CLOSE = "「『【〔［[〈《(（", "」』】〕］]〉》)）"
_EPISODE = re.compile(r"第\s*([0-9０-９]+|[〇零一二三四五六七八九十百千壱弐弍参肆伍陸漆捌玖拾佰]+)\s*[話回]")
# Bracketed channel/distribution decorations, never a work name.
_DECORATION = re.compile(r"公式|配信|期間限定|一挙|official|full|ep(isode)?\.?\s*\d", re.IGNORECASE)
_DATE = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})")


_DIGITS = dict(zip("〇零一壱二弐弍三参四肆五伍六陸七漆八捌九玖", (0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9)))
_UNITS = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000}


def episode_number(marker: str) -> int | None:
    """``第137話`` / ``第拾弍話`` / ``第一〇話`` -> integer; None if it is not a plain positive number."""
    match = _EPISODE.search(unicodedata.normalize("NFKC", marker or ""))
    if match is None:
        return None
    text = match.group(1)
    if text.isdigit():
        value = int(text)
    elif not any(char in _UNITS for char in text):
        value = int("".join(str(_DIGITS[char]) for char in text))
    else:
        value, digit = 0, None
        for char in text:
            if char in _UNITS:
                value += (1 if digit is None else digit) * _UNITS[char]
                digit = None
            else:
                digit = _DIGITS[char]
        value += digit or 0
    return value if value > 0 else None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("　", " ")).strip(" 　-－:：")


def _brackets(text: str):
    """Outermost bracketed spans as ``(start_of_inner, end_of_inner)``; inner brackets of any kind nest."""
    spans, stack = [], []
    for index, char in enumerate(text):
        if char in _OPEN:
            stack.append((char, index))
        elif char in _CLOSE and stack:
            opener, start = stack.pop()
            if _OPEN.index(opener) != _CLOSE.index(char):
                stack.clear()  # unbalanced: forget what was open
                continue
            if not stack:
                spans.append((start + 1, index))
    return spans


def parse_episode_title(text: str) -> dict | None:
    """``{"work", "episode", "subtitle"}`` from an episode title, or None when it does not name both.

    The work is the first bracketed name before the episode marker that is not a distribution decoration;
    the subtitle is the rest of the bracket holding the marker, or else the first bracketed text after it.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    episode = _EPISODE.search(text)
    if episode is None:
        return None
    spans = _brackets(text)
    work = next((_clean(text[a:b]) for a, b in spans if b < episode.start()
                 and _clean(text[a:b]) and not _DECORATION.search(text[a:b])), None)
    if work is None:
        return None
    subtitle = None
    holder = next(((a, b) for a, b in spans if a <= episode.start() < b), None)
    if holder is not None and _clean(text[episode.end():holder[1]]):
        subtitle = _clean(text[episode.end():holder[1]])
    else:
        subtitle = next((_clean(text[a:b]) for a, b in spans if a >= episode.end()
                         and _clean(text[a:b]) and not _DECORATION.search(text[a:b])), None)
    return {"work": work, "episode": _clean(episode.group(0)).replace(" ", ""), "subtitle": subtitle}


def _date(value) -> str | None:
    match = _DATE.match(str(value or "").strip())
    if match is None:
        return None
    year, month, day = (int(group) for group in match.groups())
    if not (1900 <= year <= 2200 and 1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def presentation(source: Path, tags: dict | None, *, credit=None) -> dict:
    """Album and Product-owned collection tags for ``source`` given its container tags (``{}`` if unread).

    ``credit`` is ``credit(work, subtitle) -> work_credits.credits(...) | None``; its composers become ARTIST.

    ``album`` is always present (the file-name form when the title names no work and episode); ``tags``
    holds only what was observed, plus ``LSS_COLLECTION_METADATA`` saying where each value came from.
    """
    source = Path(source)
    tags = {str(key).casefold(): value for key, value in (tags or {}).items()}
    origin, parsed = None, None
    title = tags.get("title")
    if isinstance(title, str):
        parsed, origin = parse_episode_title(unicodedata.normalize("NFC", title)), "container_title"
    if parsed is None:
        parsed, origin = parse_episode_title(unicodedata.normalize("NFC", source.stem)), "file_name"
    sources = {}
    owned = {}
    if parsed is None:
        album = f"{source.stem}{ALBUM_SUFFIX}"
        sources["album"] = "file_name"
    else:
        album = f"{parsed['work']}{ALBUM_SUFFIX}"
        sources["album"] = origin
        found = credit(parsed["work"], parsed["subtitle"]) if credit is not None else None
        if found and found.get("composers"):
            owned["ARTIST"] = list(found["composers"])
            sources["artist"] = found["url"]
    for key in ("date", "date_released", "creation_time"):
        date = _date(tags.get(key))
        if date is not None:
            owned["DATE"] = date
            sources["date"] = f"container_{key}"
            break
    owned[PROVENANCE_TAG] = json.dumps(sources, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"album": album, "tags": owned}


@lru_cache(maxsize=64)
def _read(path: str, size: int, mtime_ns: int, state_root: str | None) -> dict:
    from lss.core import media_metadata

    try:
        tags = media_metadata.format_tags(Path(path))
    except Exception:  # noqa: BLE001 - an unreadable container leaves the file-name presentation
        tags = {}
    credit = None
    if state_root is not None:
        from lss.bgm import work_credits

        def credit(work, subtitle):
            try:
                return work_credits.credits(work, subtitle, Path(state_root))
            except Exception:  # noqa: BLE001 - a credit lookup never blocks presentation
                return None
    return presentation(Path(path), tags, credit=credit)


def for_source(source: Path | None, state_root: Path | None = None) -> dict | None:
    """``presentation`` of the source media as it is now; None without a source. With ``state_root`` the
    work's composer is looked up (cached pages; offline or refused leaves ARTIST unknown)."""
    if source is None:
        return None
    source = Path(source)
    try:
        stat = source.stat()
    except OSError:
        return presentation(source, {})
    return _read(str(source), stat.st_size, stat.st_mtime_ns, None if state_root is None else str(state_root))


def album(source: Path) -> str:
    return for_source(source)["album"]
