"""What a public metadata source says about a soundtrack release or a series, as plain data.

Source adapters (`lss.metadata.sources.*`) return these; `lss.metadata.resolve` ranks and merges them. Values are
kept exactly as the source wrote them (``raw``); normalization happens only in the resolver, so a decision can
always be traced back to the source text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Ranking for every field (user, 2026-10-21): booklet-faithful databases first, distribution catalogues last.
SOURCE_RANK = ("musicbrainz", "vgmdb", "wikipedia", "itunes")


@dataclass(frozen=True)
class Credit:
    """One credited person or group in one role, as written by the source.

    ``role`` is the source's own role text (e.g. ``Composer / 作曲``, ``composer``, ``Flute / フルート``);
    ``role_key`` is a lower-case English key the resolver maps to tags (``composer``, ``arranger``,
    ``original composer``, ``lyricist``, ``vocals``, ``conductor``, ``performer``, ``instrument:flute`` …).
    ``name`` prefers the Japanese form when the source gives several.
    """
    role: str
    role_key: str
    name: str
    names: tuple[str, ...] = ()   # every written form (Japanese, romanized, English)


@dataclass(frozen=True)
class Track:
    disc: int
    number: int
    title: str
    titles: dict = field(default_factory=dict)     # language -> title as written ("ja", "en", "ja-Latn")
    length_seconds: float | None = None
    m_number: str | None = None                    # only when the source states it
    credits: tuple[Credit, ...] = ()               # track-level credits only


@dataclass(frozen=True)
class Release:
    source: str                                    # one of SOURCE_RANK
    url: str
    title: str
    titles: dict = field(default_factory=dict)
    release_date: str | None = None                # ISO date as precise as the source gives it
    catalog: str | None = None
    barcode: str | None = None
    label: str | None = None
    discs: int | None = None
    tracks: tuple[Track, ...] = ()
    credits: tuple[Credit, ...] = ()               # album-level credits (apply to every track they cover)
    notes: str | None = None                       # free text as written (may carry per-M credits)
    related_urls: tuple[str, ...] = ()             # e.g. MusicBrainz url-rels to VGMdb/Discogs
    fetched_at: str | None = None
    raw_sha256: str | None = None                  # hash of the cached response this was parsed from


@dataclass(frozen=True)
class Series:
    """A TV series or movie the program belongs to, from its own published article."""
    source: str
    url: str
    title: str
    kind: str                                      # "tv" | "movie"
    first_date: str | None                         # first broadcast (tv) / release (movie), ISO
    music: tuple[str, ...] = ()                    # staff-list 音楽 names, verbatim
    episode_listed: bool = False                   # the article lists the program's episode subtitle
    songs: tuple[dict, ...] = ()                   # [{"title", "credits": (Credit, ...)}] theme/insert songs
    episode_dates: tuple[tuple[int, str], ...] = ()  # (episode number, first broadcast ISO date); () for movies
    fetched_at: str | None = None
    raw_sha256: str | None = None
    match: str | None = None                       # "subtitle_listed" | "ongoing_continuation": which rule took this article
