"""Deterministic ordering for a newly imported batch; never reorder a live queue."""
from __future__ import annotations

import re
from pathlib import Path

from mutagen import File as MutagenFile


def _natural(path: Path) -> tuple:
    # Type markers keep numbered and unnumbered names comparable. The original
    # path breaks case/leading-zero ties independently of OS selection order.
    tokens = tuple((1, int(part)) if part.isdecimal() else (0, part.casefold())
                   for part in re.split(r"([0-9]+)", str(path)))
    return tokens, str(path).casefold(), str(path)


def _text(tags, key: str) -> str:
    values = tags.get(key, [])
    return str(values[0]).strip().casefold() if len(values) == 1 else ""


def _number(tags, key: str, *, default: int | None = None) -> int | None:
    values = tags.get(key, [])
    if not values:
        return default
    if len(values) != 1:
        return None
    match = re.fullmatch(r"\s*([0-9]+)(?:\s*/\s*([0-9]+))?\s*", str(values[0]))
    if match is None:
        return None
    number = int(match[1])
    total = int(match[2]) if match[2] is not None else number
    return number if 0 < number <= total else None


def order_input_batch(paths: list[Path]) -> list[Path]:
    """Use complete unambiguous album numbering, otherwise natural path order.

    Albums can span disc folders. A group without an album tag is local to its
    source folder. Missing/ambiguous numbering invalidates metadata ordering for
    the whole group, so unknown tracks are not pushed behind known tracks.
    Reading tags is best effort and never changes the source media.
    """
    groups: dict[tuple, list[tuple[Path, tuple[int | None, int | None]]]] = {}
    for path in sorted(paths, key=_natural):
        try:
            audio = MutagenFile(path, easy=True)
            tags = audio.tags if audio is not None and audio.tags is not None else {}
            album = _text(tags, "album")
            artist = _text(tags, "albumartist")
            group = ("album", album, artist) if album else ("folder", str(path.parent).casefold())
            position = (_number(tags, "discnumber", default=1), _number(tags, "tracknumber"))
        except Exception:
            # Unknown/corrupt formats must still enter the queue and receive the
            # normal processing failure. Metadata is not input validation.
            group = ("folder", str(path.parent).casefold())
            position = (None, None)
        groups.setdefault(group, []).append((path, position))
    result = []
    for members in groups.values():
        positions = [position for _, position in members]
        if all(disc is not None and track is not None for disc, track in positions) \
                and len(set(positions)) == len(positions):
            members.sort(key=lambda item: item[1])
        result.extend(path for path, _ in members)
    return result
