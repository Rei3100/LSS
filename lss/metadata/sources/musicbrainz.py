"""MusicBrainz (WS/2) adapter: finds the official soundtrack release(s) of a series and returns full ``Release``s.

MusicBrainz is the first-ranked source. Requests go through ``lss.metadata.fetch`` (cached, robots-aware, paced by
the shared acquisition courtesy, whose per-host interval is far above MusicBrainz's 1 request/second limit).
Credits are kept exactly as MusicBrainz states them; nothing is inferred.
"""
from __future__ import annotations

import re
import unicodedata
import urllib.parse
from pathlib import Path

from lss.metadata import fetch
from lss.metadata.model import Credit, Release, Track

BASE = "https://musicbrainz.org/ws/2"
LOOKUP_INC = ("recordings+artist-credits+labels+url-rels+release-groups+recording-level-rels+work-level-rels"
              "+artist-rels+recording-rels+work-rels")
SOUNDTRACK_WORDS = ("サウンドトラック", "soundtrack")
TITLE_SOUNDTRACK_MARKS = ("サウンドトラック", "soundtrack", "ost", "サントラ")
SEARCH_LIMIT = 100

# MusicBrainz relationship type -> model role_key (others keep their own lower-case type text).
_ROLE_KEYS = {"vocal": "vocals", "performing orchestra": "orchestra", "mixer": "mix", "performance": "performer"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "")).casefold()


def _compact(text: str) -> str:
    """Title form for matching: NFKC, case-folded, only letters/digits/kana/kanji kept."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", text or "").casefold()
                   if unicodedata.category(ch)[0] in "LN")


def _has_cjk(text: str) -> bool:
    return any("぀" <= ch <= "ヿ" or "㐀" <= ch <= "鿿" or "ｦ" <= ch <= "ﾟ" for ch in text)


def _title_tokens(title: str) -> list[str]:
    parts = re.split(r"[\s〜～~「」『』()\[\]・:!?,、。]+", unicodedata.normalize("NFKC", title or ""))
    return [_compact(part) for part in parts if _compact(part)]


def _date_bounds(date: str) -> tuple[str, str]:
    """(earliest, latest) full ISO date a possibly partial date could mean."""
    parts = (date or "").split("-")
    month = parts[1] if len(parts) > 1 else None
    day = parts[2] if len(parts) > 2 else None
    return f"{parts[0]}-{month or '01'}-{day or '01'}", f"{parts[0]}-{month or '12'}-{day or '31'}"


def _date_ok(date: str | None, not_before: str | None) -> bool:
    if not date or not not_before:
        return True
    return _date_bounds(date)[1] >= _date_bounds(not_before)[0]


def _date_key(date: str | None) -> tuple[bool, str]:
    return (not date, _date_bounds(date)[0] if date else "")


def _artist_forms(artist: dict) -> list[str]:
    forms = [artist.get("name"), artist.get("sort-name")]
    for alias in artist.get("aliases") or []:
        forms += [alias.get("name"), alias.get("sort-name")]
    return list(dict.fromkeys(f for f in forms if f))


def _artist_name(artist: dict, credited: str | None = None) -> tuple[str, tuple[str, ...]]:
    """(preferred name, every form). Japanese script wins: a ja alias, else the name itself when it is Japanese."""
    forms = _artist_forms(artist)
    if credited and credited not in forms:
        forms.append(credited)
    candidates = []
    for alias in artist.get("aliases") or []:
        if str(alias.get("locale") or "").startswith("ja"):
            candidates.append(alias.get("name"))
    candidates += [artist.get("name"), credited] + forms
    name = next((c for c in candidates if c and _has_cjk(c)), None) or artist.get("name") or (forms[0] if forms else "")
    return name, tuple(forms)


def _credit_matches(artist_credit: list, composers: list[str]) -> bool:
    wanted = {_norm(c) for c in composers if c}
    forms: list[str] = []
    for item in artist_credit or []:
        forms.append(item.get("name"))
        forms += _artist_forms(item.get("artist") or {})
    return any(_norm(form) in wanted for form in forms if form)


def _credits_for(relation_type: str, attributes: list, artist: dict, credited: str | None) -> list[Credit]:
    name, names = _artist_name(artist, credited or None)
    attrs = [a for a in attributes or [] if a]
    if relation_type == "instrument" and attrs:
        return [Credit(role=f"instrument / {a}", role_key=f"instrument:{a.lower()}", name=name, names=names)
                for a in attrs]
    if relation_type == "vocal" and attrs:
        return [Credit(role=f"vocal / {a}", role_key="vocals", name=name, names=names) for a in attrs]
    key = _ROLE_KEYS.get(relation_type, relation_type.lower())
    return [Credit(role=relation_type, role_key=key, name=name, names=names)]


def _relation_credits(relations: list, *, in_work: bool = False) -> list[Credit]:
    out: list[Credit] = []
    for rel in relations or []:
        if rel.get("target-type") == "artist" and rel.get("artist"):
            out += _credits_for(rel.get("type") or "", rel.get("attributes"), rel["artist"], rel.get("target-credit"))
        elif (not in_work and rel.get("target-type") == "work" and rel.get("type") == "performance"
              and rel.get("work")):
            out += _relation_credits(rel["work"].get("relations"), in_work=True)   # composer/lyricist of the work
    return out


def _dedupe(credits: list[Credit]) -> tuple[Credit, ...]:
    seen, out = set(), []
    for credit in credits:
        key = (credit.role_key, credit.role, _norm(credit.name))
        if key not in seen:
            seen.add(key)
            out.append(credit)
    return tuple(out)


def _titles(title: str) -> dict:
    return {("ja" if _has_cjk(title) else "en"): title} if title else {}


def _artist_credits(items) -> list[Credit]:
    """Release or recording artist credit as role 'artist' (not a tag; it pairs written name forms)."""
    out = []
    for item in items or []:
        name, names = _artist_name(item.get("artist") or {}, item.get("name"))
        out.append(Credit(role="artist", role_key="artist", name=name, names=names))
    return out


def release(mbid: str, state_root, *, opener=None, courtesy=None) -> Release:
    """Full release lookup. Raises ``fetch.Unavailable`` when MusicBrainz cannot be read."""
    url = f"{BASE}/release/{mbid}?inc={LOOKUP_INC}&fmt=json"
    data, entry = fetch.get_json(url, Path(state_root), opener=opener, courtesy=courtesy, api=True,
                                 purpose="musicbrainz release")
    title = data.get("title") or ""
    media = data.get("media") or []
    tracks: list[Track] = []
    for medium in media:
        disc = int(medium.get("position") or 1)
        for item in medium.get("tracks") or []:
            recording = item.get("recording") or {}
            ttitle = item.get("title") or recording.get("title") or ""
            length = item.get("length") or recording.get("length")
            tracks.append(Track(disc=disc, number=int(item.get("position") or 0), title=ttitle,
                                titles=_titles(ttitle), length_seconds=(length / 1000.0 if length else None),
                                credits=_dedupe(_relation_credits(recording.get("relations"))
                                                + _artist_credits(recording.get("artist-credit")))))
    album: list[Credit] = _artist_credits(data.get("artist-credit"))
    album += _relation_credits(data.get("relations"))
    labels = data.get("label-info") or []
    catalog = next((li.get("catalog-number") for li in labels if li.get("catalog-number")), None)
    label = next(((li.get("label") or {}).get("name") for li in labels if (li.get("label") or {}).get("name")), None)
    related = tuple(rel["url"]["resource"] for rel in data.get("relations") or []
                    if rel.get("target-type") == "url" and (rel.get("url") or {}).get("resource"))
    return Release(source="musicbrainz", url=f"https://musicbrainz.org/release/{mbid}", title=title,
                   titles=_titles(title), release_date=data.get("date") or None, catalog=catalog,
                   barcode=data.get("barcode") or None, label=label, discs=len(media) or None,
                   tracks=tuple(tracks), credits=_dedupe(album), related_urls=related,
                   fetched_at=entry.get("fetched_at"), raw_sha256=entry.get("sha256"))


def is_transient(exc: fetch.Unavailable) -> bool:
    """True when the source could not be asked now (503 / timeout / offline); False for an answered "no"
    (404-class status, robots refusal, a remembered miss)."""
    status = getattr(exc, "status", None)
    text = str(exc)
    if "remembered" in text or "robots" in text or "https only" in text:
        return False
    if status is None:
        return True
    return status >= 500 or status in (408, 425, 429)


def _search(query: str, state_root, opener, courtesy, strict: bool = False) -> list[dict]:
    url = f"{BASE}/release/?" + urllib.parse.urlencode({"query": query, "limit": SEARCH_LIMIT, "fmt": "json"})
    try:
        data, _ = fetch.get_json(url, Path(state_root), opener=opener, courtesy=courtesy, api=True,
                                 purpose="musicbrainz search")
    except fetch.Unavailable as exc:
        if strict and is_transient(exc):
            raise
        return []
    return [r for r in data.get("releases") or [] if isinstance(r, dict) and r.get("id")]


def _lucene(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def search_soundtracks(series_title: str, composers: list[str], not_before: str | None, state_root, *,
                       opener=None, courtesy=None, strict: bool = False) -> list[Release]:
    """Official soundtrack releases of ``series_title`` credited to one of ``composers``.

    Returns ``[]`` when nothing qualifies or MusicBrainz cannot be read (UNKNOWN, not ABSENT; a cached miss
    expires and a later run asks again). One edition per release group is kept (the earliest); other editions of
    the same release group may exist.

    ``strict=True`` lets a caller tell "no soundtrack found" (``[]``) from "could not ask": a transient failure
    (503, timeout, offline; not 404 / robots / a remembered miss) raises ``fetch.Unavailable``.
    """
    composers = [c for c in composers if c]
    tokens = _title_tokens(series_title)
    if not tokens or not composers:
        return []
    ost = "(" + " OR ".join(f"release:{w}" for w in SOUNDTRACK_WORDS) + ")"
    queries = [" AND ".join(f"release:{_lucene(t)}" for t in tokens) + f" AND {ost}"]
    queries += [f"artist:{_lucene(c)} AND {ost}" for c in composers]
    found: dict[str, dict] = {}
    for query in queries:
        for item in _search(query, state_root, opener, courtesy, strict):
            found.setdefault(item["id"], item)
    wanted = {_norm(c) for c in composers}
    groups: dict[str, list[dict]] = {}
    for item in found.values():
        title = item.get("title") or ""
        compact = _compact(title)
        if not all(token in compact for token in tokens):
            continue
        if not any(mark in title.casefold() for mark in TITLE_SOUNDTRACK_MARKS):
            continue
        if not _date_ok(item.get("date"), not_before):
            continue
        groups.setdefault((item.get("release-group") or {}).get("id") or item["id"], []).append(item)
    result: list[Release] = []
    for items in groups.values():
        items.sort(key=lambda r: (_date_key(r.get("date")), r["id"]))   # earliest edition first
        for item in items:
            credited = _credit_matches(item.get("artist-credit"), composers)
            if not credited and any(_has_cjk(c.get("name") or "") for c in item.get("artist-credit") or []):
                continue   # a Japanese credit that is not the composer: not this soundtrack
            try:
                full = release(item["id"], state_root, opener=opener, courtesy=courtesy)
            except fetch.Unavailable as exc:
                if strict and is_transient(exc):
                    raise
                continue
            if not credited and not any(_norm(n) in wanted for c in full.credits for n in (c.name, *c.names)):
                continue   # romanized/alias credit that also fails the lookup's artist forms
            if not _date_ok(full.release_date, not_before):
                continue
            result.append(full)
            break
    result.sort(key=lambda r: (_date_key(r.release_date), r.title))
    return result
