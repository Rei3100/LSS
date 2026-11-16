"""Reconcile what the sources say into one series album (contract: LISTENING_AND_PRESENTATION "Music-library metadata").

Pure functions over `lss.metadata.model` data. Every choice keeps the losing values: ``Choice.raw`` lists each
source's own text, so a disagreement (a store-wide label typo, a different edition title) stays inspectable.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from lss.metadata.model import SOURCE_RANK, Credit, Release

OST = "オリジナル・サウンドトラック"

# Volume marks of a multi-volume soundtrack; the album is the series, so they are removed from the album title.
_VOLUME = re.compile(
    r"\s*(?:[\(（\[［]\s*)?(?:vol(?:ume)?\.?\s*\d+|第\s*[0-9０-９一二三四五六七八九十]+\s*[弾集巻]|"
    r"(?:Part|PART|part)\s*\d+|[Ⅰ-Ⅻ])(?:\s*[\)）\]］])?\s*", re.IGNORECASE)
# Spellings of "original soundtrack" normalized to the middle-dot form (user: always オリジナル・サウンドトラック).
_OST_FORMS = re.compile(r"オリジナル[\s・･]*サウンド[\s・･]*トラック")
# Suffixes distribution stores append to every track of an instrumental album.
_STORE_SUFFIX = re.compile(r"\s*[\(（](?:Instrumental|インストゥルメンタル|Inst\.?)[\)）]\s*$", re.IGNORECASE)


def compare_key(text: str) -> str:
    """Spelling-insensitive key: NFKC, case-folded, without spaces, middle dots, brackets and wave dashes."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"[\s・･「」『』〜~～\-‐－:：!！?？、,.。()（）\[\]［］'’]", "", text)


def album_title(title: str, series_title: str | None = None) -> str:
    """The series album title from an official title: volume marks removed, OST spelled with the middle dot.

    Falls back to ``<series> オリジナル・サウンドトラック`` when removing the volume mark leaves no title text
    besides the OST words or leaves dangling punctuation.
    """
    text = _OST_FORMS.sub(OST, unicodedata.normalize("NFC", title or "").strip())
    stripped = _VOLUME.sub(" ", text)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip(" 　-－–~〜～:：/／")
    natural = bool(stripped) and compare_key(stripped) != compare_key(OST) and not re.search(r"[「『（(]\s*$", stripped)
    if natural and stripped.count("「") == stripped.count("」") and stripped.count("『") == stripped.count("』"):
        return stripped
    return f"{series_title} {OST}" if series_title else (stripped or text)


def store_suffix_pattern(titles: list[str]) -> bool:
    """True when a store appended an instrumental suffix to (nearly) every title of the album."""
    if not titles:
        return False
    marked = sum(1 for title in titles if _STORE_SUFFIX.search(title or ""))
    return marked >= max(2, int(0.8 * len(titles)))


def strip_store_suffix(title: str) -> str:
    return _STORE_SUFFIX.sub("", title or "").strip()


@dataclass(frozen=True)
class Choice:
    value: object
    source: str | None
    raw: dict = field(default_factory=dict)        # source -> its own value (losing values included)


def choose(values: dict) -> Choice:
    """The value of the best-ranked source that has one; ``values`` maps source name -> value (None = absent)."""
    present = {source: value for source, value in values.items() if value not in (None, "", (), [])}
    for source in SOURCE_RANK:
        if source in present:
            repaired = _unmangled(present[source], present)
            if repaired is not None:
                return Choice(present[repaired], repaired, dict(present))
            return Choice(present[source], source, dict(present))
    for source, value in present.items():   # an unranked source only when no ranked one speaks
        return Choice(value, source, dict(present))
    return Choice(None, None, {})


def _unmangled(value, present: dict) -> str | None:
    """A source whose text equals ``value`` except where ``value`` has an encoding placeholder (``?``/``？``/``�``)
    and that source has a real character — the same title before a lossy conversion. None when there is none."""
    if not isinstance(value, str) or not re.search(r"[?？�]", value):
        return None
    for source in SOURCE_RANK:
        other = present.get(source)
        if not isinstance(other, str) or other == value or len(other) != len(value):
            continue
        if all(a == b or (a in "?？�" and b not in "?？�") for a, b in zip(value, other)):
            return source
    return None


def same_release(a: Release, b: Release) -> bool:
    """Two sources describe the same release: equal catalog/barcode, or same track count and lengths."""
    if a.catalog and b.catalog and compare_key(a.catalog) == compare_key(b.catalog):
        return True
    if a.barcode and b.barcode and a.barcode.lstrip("0") == b.barcode.lstrip("0"):
        return True
    if not a.tracks or len(a.tracks) != len(b.tracks):
        return False
    lengths = [(x.length_seconds, y.length_seconds) for x, y in zip(a.tracks, b.tracks)]
    known = [(x, y) for x, y in lengths if x and y]
    return len(known) >= max(min(3, len(lengths)), len(lengths) // 2) and all(abs(x - y) <= 2.0 for x, y in known)


@dataclass(frozen=True)
class ResolvedTrack:
    disc: int                                      # disc number within the series album
    number: int
    title: Choice
    length_seconds: float | None
    m_number: Choice
    credits: tuple[Credit, ...]                    # album-level + track-level, deduplicated
    release_title: str                             # the official volume this track is on


# Sources that describe the physical release (media/disc structure). Stores often flatten a multi-disc set into one
# disc with running numbers, so disc/track structure never comes from them.
PHYSICAL_SOURCES = ("musicbrainz", "vgmdb")


def _ordered(view: Release) -> list:
    return sorted(view.tracks, key=lambda t: (t.disc, t.number))


def align(lead: Release, view: Release) -> dict:
    """``(disc, number)`` of each lead track -> the same track in ``view``.

    Same disc structure: by disc and number. Otherwise (a store's flattened numbering) by running order over the
    whole release, kept only where both lengths are known and agree within 2 s, or both are unknown.
    """
    mine, theirs = _ordered(lead), _ordered(view)
    if [(t.disc, t.number) for t in mine] == [(t.disc, t.number) for t in theirs]:
        return {(a.disc, a.number): b for a, b in zip(mine, theirs)}
    if len(mine) != len(theirs):
        return {}
    out = {}
    for a, b in zip(mine, theirs):
        if (a.length_seconds is None) == (b.length_seconds is None) and (
                a.length_seconds is None or abs(a.length_seconds - b.length_seconds) <= 2.0):
            out[(a.disc, a.number)] = b
    return out


def merge_volume(views: list[Release]) -> dict:
    """One official volume as every source sees it -> per-track choices (``views`` describe one release).

    Disc/track structure comes from the best-ranked physical source; other views are aligned to it (`align`).
    """
    by_source = {view.source: view for view in views}
    lead = next((by_source[s] for s in SOURCE_RANK if s in by_source and s in PHYSICAL_SOURCES and by_source[s].tracks),
                None) or next((by_source[s] for s in SOURCE_RANK if s in by_source), views[0])
    suffixed = {view.source for view in views if store_suffix_pattern([t.title for t in view.tracks])}
    aligned = {view.source: align(lead, view) for view in views}
    tracks = []
    for track in _ordered(lead):
        titles, numbers, credits, album_level = {}, {}, [], []
        for view in views:
            other = aligned[view.source].get((track.disc, track.number))
            if other is None:
                continue
            titles[view.source] = strip_store_suffix(other.title) if view.source in suffixed else other.title
            numbers[view.source] = other.m_number
            credits.extend(other.credits)
            album_level.extend(view.credits)
        # A track-level composer statement (e.g. notes "except: M-10 - Original Composer: …", or a work's
        # composer) is the exception an album-level composer credit does not cover.
        own = {c.role_key for c in credits}
        credits.extend(c for c in album_level
                       if not (c.role_key == "composer" and own & {"composer", "original composer"}))
        tracks.append({"disc": track.disc, "number": track.number, "title": choose(titles),
                       "length_seconds": track.length_seconds, "m_number": choose(numbers),
                       "credits": credits})
    unified = iter(unify_names([credit for row in tracks for credit in row["credits"]]))
    for row in tracks:
        row["credits"] = _dedupe([next(unified) for _ in row["credits"]])
    title = choose({view.source: view.title for view in views})
    date = choose({view.source: view.release_date for view in views})
    discs = max((t["disc"] for t in tracks), default=lead.discs or 1)
    return {"title": title, "release_date": date, "discs": discs, "tracks": tracks}


def _name_forms(name: str) -> set[str]:
    """Comparable forms of one written name: as written, and 'Last, First' turned to 'First Last'."""
    forms = {compare_key(name)}
    if "," in name:
        last, _, first = name.partition(",")
        forms.add(compare_key(f"{first.strip()} {last.strip()}"))
    return {form for form in forms if form}


def _japanese(text: str) -> bool:
    return bool(re.search(r"[぀-ヿ一-鿿]", text or ""))


def unify_names(credits) -> tuple[Credit, ...]:
    """Write each person the same way across sources: when any source pairs a romanized form with a Japanese
    one (credit ``names``), every credit of that person uses the Japanese form. Unpaired names stay as written."""
    credits = list(credits)
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    written: dict[str, set[str]] = {}
    for credit in credits:
        forms = (credit.name, *credit.names)
        keys = set().union(*(_name_forms(form) for form in forms if form))
        for form in forms:
            for key in _name_forms(form or ""):
                written.setdefault(key, set()).add(form)
        keys = sorted(keys)
        for key in keys[1:]:
            parent[find(key)] = find(keys[0])
    preferred: dict[str, str] = {}
    for key, forms in written.items():
        root = find(key)
        for form in sorted(forms):
            if _japanese(form) and (root not in preferred or not _japanese(preferred[root])):
                preferred[root] = form
    out = []
    for credit in credits:
        roots = {find(key) for key in _name_forms(credit.name)}
        best = next((preferred[root] for root in roots if root in preferred), credit.name)
        out.append(credit if best == credit.name else Credit(credit.role, credit.role_key, best, credit.names))
    return tuple(out)


def _dedupe(credits: list[Credit]) -> tuple[Credit, ...]:
    seen, out = set(), []
    for credit in credits:
        key = (credit.role_key, compare_key(credit.name))
        if key not in seen:
            seen.add(key)
            out.append(credit)
    return tuple(out)


def series_album(series_title: str, volumes: list[dict]) -> dict:
    """Volumes (``merge_volume`` results, oldest first) -> the series album with continuous disc numbers.

    Returns ``{"title", "title_choice", "tracks": [ResolvedTrack], "official_discs", "unreleased_disc"}``.
    """
    tracks, offset = [], 0
    for volume in volumes:
        for row in volume["tracks"]:
            tracks.append(ResolvedTrack(disc=offset + row["disc"], number=row["number"], title=row["title"],
                                        length_seconds=row["length_seconds"], m_number=row["m_number"],
                                        credits=row["credits"], release_title=str(volume["title"].value)))
        offset += volume["discs"]
    first = volumes[0]["title"] if volumes else Choice(None, None, {})
    title = album_title(str(first.value), series_title) if first.value else f"{series_title} {OST}"
    volume_titles = {str(v["title"].value): tuple(str(x) for x in v["title"].raw.values()) for v in volumes}
    return {"title": title, "title_choice": first, "tracks": tracks, "official_discs": offset,
            "unreleased_disc": offset + 1, "volume_titles": volume_titles}


# --------------------------------------------------------------------------------------------- credits -> tags

# Tag names are ASCII (Vorbis comment field names); a role keeps its own tag, never folded into another.
_ROLE_TAGS = {
    "composer": "COMPOSER", "original composer": "ORIGINALCOMPOSER", "arranger": "ARRANGER",
    "orchestrator": "ORCHESTRATOR", "lyricist": "LYRICIST", "writer": "WRITER", "conductor": "CONDUCTOR",
    "producer": "PRODUCER", "director": "DIRECTOR", "music director": "MUSICDIRECTOR", "engineer": "ENGINEER",
    "recording engineer": "RECORDINGENGINEER", "mix": "MIXER", "mixer": "MIXER", "mixing engineer": "MIXER",
    "mastering": "MASTERING", "mastering engineer": "MASTERING", "remixer": "REMIXER", "programming": "PROGRAMMING",
    "vocals": "VOCALS", "chorus": "CHORUS", "performer": "PERFORMER", "orchestra": "ORCHESTRA", "ensemble": "ENSEMBLE",
}
# Japanese-only role words (e.g. Wikipedia song credits) -> role keys.
_JAPANESE_ROLES = {
    "作曲": "composer", "原曲": "original composer", "原作曲": "original composer", "編曲": "arranger",
    "作詞": "lyricist", "作詩": "lyricist", "歌": "vocals", "歌唱": "vocals", "ボーカル": "vocals",
    "コーラス": "chorus", "指揮": "conductor", "演奏": "performer", "プロデューサー": "producer",
    "プロデュース": "producer", "ミックス": "mix", "ミキシング": "mix", "マスタリング": "mastering",
    "レコーディング": "recording engineer", "録音": "recording engineer", "プログラミング": "programming",
    "オーケストレーション": "orchestrator", "管弦楽": "orchestra",
}
# Release-level identifiers the user does not want as tags (2026-10-21).
_NOT_TAGGED = {"label", "publisher", "catalog", "catalog number", "distributor", "manufacturer", "artist", "music"}


def _english_role(role: str) -> str | None:
    """'Flute / フルート' -> 'Flute'; None when the source wrote no ASCII role name."""
    parts = [part.strip() for part in re.split(r"\s*/\s*", role or "") if part.strip()]
    ascii_parts = [part for part in parts if part.isascii() and re.search(r"[A-Za-z]", part)]
    return ascii_parts[0] if ascii_parts else None


def role_tag(credit: Credit) -> str | None:
    """The tag for this credit's role, or None when it is not tagged (release identifiers, unmapped words)."""
    key = (credit.role_key or "").lower().strip()
    if key in _NOT_TAGGED:
        return None
    if key.startswith("instrument:"):
        key = key.split(":", 1)[1]
    if key in _ROLE_TAGS:
        return _ROLE_TAGS[key]
    japanese = _JAPANESE_ROLES.get((credit.role or "").strip())
    if japanese:
        return _ROLE_TAGS.get(japanese)
    english = _english_role(credit.role) or (key if key.isascii() and re.search(r"[a-z]", key) else None)
    if not english:
        return None
    tag = re.sub(r"[^A-Z0-9 ]", "", unicodedata.normalize("NFKC", english).upper()).strip()
    return re.sub(r"\s+", " ", tag) or None


def credit_tags(credits) -> tuple[dict[str, list[str]], list[Credit]]:
    """(tags, untagged): every credited role under its own tag; credits with no tag stay internal."""
    tags: dict[str, list[str]] = {}
    untagged: list[Credit] = []
    for credit in credits:
        tag = role_tag(credit)
        if tag is None:
            if (credit.role_key or "").lower() not in _NOT_TAGGED:
                untagged.append(credit)
            continue
        values = tags.setdefault(tag, [])
        if credit.name not in values:
            values.append(credit.name)
    return tags, untagged
