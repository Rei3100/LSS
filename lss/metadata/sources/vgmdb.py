"""VGMdb album pages, read from the Internet Archive.

vgmdb.net itself answers non-browser clients with a Cloudflare challenge. LSS never impersonates a browser and never
solves a challenge: it reads the *archived* copy of the album page from the Wayback Machine with its own honest user
agent (``fetch.USER_AGENT``). The time-gate form ``https://web.archive.org/web/https://vgmdb.net/album/<id>`` redirects
to the newest snapshot (the ``id_`` raw form is served gzip-compressed and is not used). A page that was never archived
(404) or an archive that cannot be reached means "no VGMdb data now" (``None``), never an error: UNKNOWN is not ABSENT.

The snapshot can be months old; its timestamp is recorded in ``Release.related_urls`` (the snapshot URL), and
``Release.notes`` is only the page's own notes text.
"""
from __future__ import annotations

import html
import re
import time
from pathlib import Path

from lss.metadata import fetch
from lss.metadata.model import Credit, Release, Track

SOURCE = "vgmdb"
ALBUM_URL = "https://vgmdb.net/album/{id}"
ARCHIVE_URL = "https://web.archive.org/web/https://vgmdb.net/album/{id}"
SNAPSHOT_URL = "https://web.archive.org/web/{ts}/https://vgmdb.net/album/{id}"

_ALBUM_ID = re.compile(r"vgmdb\.net/album/(\d+)", re.I)
_INSTRUMENTS = {
    "piano", "celesta", "synthesizer", "keyboard", "organ", "harpsichord", "glockenspiel", "xylophone", "marimba",
    "vibraphone", "tubular bells", "flute", "alto flute", "piccolo", "oboe", "english horn", "bassoon", "clarinet",
    "b clarinet", "e clarinet", "bass clarinet", "saxophone", "horn", "trumpet", "trombone", "tenor trombone",
    "bass trombone", "tuba", "percussion", "timpani", "drums", "harp", "violin", "viola", "cello", "contrabass",
    "double bass", "guitar", "acoustic guitar", "electric guitar", "bass guitar", "ukulele", "banjo", "accordion",
    "harmonica", "recorder", "shamisen", "koto", "shakuhachi", "erhu", "mandolin", "sitar",
}
_CHORUS_PARTS = {"soprano", "alto", "tenor", "bass", "chorus", "choir"}
_ROLE_ALIASES = {"voice": "vocals", "vocal": "vocals", "vocals": "vocals", "vocalist": "vocals"}
_SPEC = (r"M-?\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*(?:M-?)?\d+(?:\s*[-–]\s*\d+)?)*")


def album_id_from_url(url: str) -> int | None:
    match = _ALBUM_ID.search(url or "")
    return int(match.group(1)) if match else None


def _text(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return html.unescape(fragment).replace("\xa0", " ")


def _flat(fragment: str) -> str:
    return re.sub(r"\s+", " ", _text(fragment)).strip()


def _lang_spans(fragment: str, css: str) -> dict:
    """``{lang: text}`` of the language spans (``en``, ``ja``, ``ja-Latn``) of one multilingual field."""
    found = {}
    for lang, inner in re.findall(rf'<span[^>]*class="{css}"[^>]*lang="([\w-]+)"[^>]*>(.*?)</span>', fragment, re.S):
        value = _flat(re.sub(r"<em>.*?</em>", "", inner, flags=re.S))
        if value and lang not in found:
            found[lang] = value
    return found


def _iso_date(text: str) -> str | None:
    text = text.strip()
    for pattern, out in (("%b %d, %Y", "%Y-%m-%d"), ("%B %d, %Y", "%Y-%m-%d"), ("%b %Y", "%Y-%m"), ("%Y", "%Y")):
        try:
            return time.strftime(out, time.strptime(text, pattern))
        except ValueError:
            continue
    return None


def _seconds(text: str) -> float | None:
    match = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{2})", text.strip())
    if not match:
        return None
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return float(hours * 3600 + minutes * 60 + seconds)


def _role_key(english: str) -> str:
    key = re.sub(r"\s+", " ", english.strip().lower())
    if key in _ROLE_ALIASES:
        return _ROLE_ALIASES[key]
    if key in _CHORUS_PARTS:
        return "chorus"
    if key in _INSTRUMENTS:
        return f"instrument:{key}"
    return key


def _unique(items) -> tuple:
    seen = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return tuple(seen)


def _info_rows(region: str) -> dict:
    rows = {}
    for label, value in re.findall(
            r'<tr[^>]*>\s*<td[^>]*>\s*<span class="label"><b>([^<]+)</b></span></td>\s*<td[^>]*>(.*?)</td>\s*</tr>',
            region, re.S):
        rows.setdefault(_flat(label), value)
    return rows


def _credits(region: str) -> tuple[tuple[Credit, ...], dict]:
    """Album credits plus a lookup ``{name-key: (japanese, all forms)}`` for naming people from the notes."""
    credits, lookup = [], {}
    for row in re.findall(r'<tr class="maincred">(.*?)</tr>', region, re.S):
        label_cell, _, value_cell = row.partition("</td>")
        label = _lang_spans(label_cell, "artistname")
        if "en" not in label:
            continue
        role = f"{label['en']} / {label['ja']}" if label.get("ja") and label["ja"] != label["en"] else label["en"]
        key = _role_key(label["en"])
        for link in re.findall(r"<a [^>]*>(.*?)</a>", value_cell, re.S):
            forms = _lang_spans(link, "artistname") or {"en": _flat(link)}
            ordered = _unique([forms.get("ja"), forms.get("en"), forms.get("ja-Latn")])
            if not ordered:
                continue
            credits.append(Credit(role=role, role_key=key, name=ordered[0], names=ordered))
            for form in ordered:
                lookup.setdefault(_name_key(form), ordered)
        rest = re.sub(r"<a [^>]*>.*?</a>", "|", value_cell, flags=re.S)
        for plain in _flat(rest).split("|"):
            plain = re.sub(r"\([^)]*\)", "", plain).strip(" ,/")
            if plain:
                credits.append(Credit(role=role, role_key=key, name=plain, names=(plain,)))
    return tuple(credits), lookup


def _name_key(name: str) -> str:
    return " ".join(sorted(re.sub(r"[^\w]+", " ", name.lower()).split()))


def _tracklists(body: str) -> dict:
    """``{language: {(disc, number): (title, seconds)}}`` for every tracklist tab, plus the disc count."""
    tabs = {rel: label for rel, label in re.findall(r'<a class="smallfont" rel="(tl\d+)" href="#">([^<]+)</a>', body)}
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'<span class="tl" id="(tl\d+)"', body)]
    lists = {}
    end_of_lists = body.find('id="notes"')
    for index, (start, ident) in enumerate(starts):
        stop = starts[index + 1][0] if index + 1 < len(starts) else (end_of_lists if end_of_lists > start else len(body))
        segment = body[start:stop]
        label = tabs.get(ident, "English")
        lang = {"english": "en", "japanese": "ja", "romaji": "ja-Latn"}.get(label.strip().lower(), label.strip().lower())
        entries, disc = {}, 1
        parts = re.split(r"<b>Disc\s*(\d+)[^<]*</b>", segment)
        chunks = [(1, parts[0])] + [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
        for disc, chunk in chunks:
            for row in re.findall(r'<tr class="rolebit">(.*?)</tr>', chunk, re.S):
                cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
                if len(cells) < 2:
                    continue
                number = re.sub(r"\D", "", _flat(cells[0]))
                title = _flat(cells[1])
                if not number or not title:
                    continue
                length = _seconds(_flat(cells[-1])) if len(cells) > 2 else None
                entries[(disc, int(number))] = (title, length)
        if entries:
            lists[lang] = entries
    return lists


def _expand(spec: str) -> list[int]:
    numbers = []
    for token in re.split(r"\s*,\s*", spec.strip()):
        token = re.sub(r"^M-?", "", token.strip(), flags=re.I)
        match = re.fullmatch(r"(\d+)(?:\s*[-–]\s*(\d+))?", token)
        if not match:
            continue
        low = int(match.group(1))
        high = int(match.group(2)) if match.group(2) else low
        if low <= high <= low + 200:
            numbers.extend(range(low, high + 1))
    return numbers


def _note_credits(notes: str, lookup: dict) -> dict:
    """``{track number: [Credit, ...]}`` for notes lines that name tracks with ``M-xx`` and a role."""
    per_track: dict = {}

    def person(written: str) -> tuple[str, tuple]:
        mapped = lookup.get(_name_key(written))
        if mapped:
            return mapped[0], _unique(list(mapped) + [written])
        return written, (written,)

    def add(spec_numbers, role_text, names_text):
        role_text = re.sub(r"\s+", " ", role_text.strip(" -–"))
        if not role_text or re.search(r"\bexcept\b", role_text, re.I):
            return
        key = _role_key(role_text)
        for written in (n.strip() for n in names_text.split(",")):
            written = re.sub(r"\s+", " ", written.strip(" .;"))
            if not written:
                continue
            name, forms = person(written)
            credit = Credit(role=role_text, role_key=key, name=name, names=forms)
            for number in spec_numbers:
                bucket = per_track.setdefault(number, [])
                if credit not in bucket:
                    bucket.append(credit)

    for line in notes.splitlines():
        line = line.strip()
        lead = re.match(rf"^({_SPEC})\s*[-–:]\s*(.+)$", line)
        if lead and ":" in lead.group(2):
            role, _, names = lead.group(2).partition(":")
            add(_expand(re.sub(r"^M-?", "", lead.group(1), count=1, flags=re.I)), role, names)
            continue
        if ":" not in line:
            continue
        role, _, rest = line.partition(":")
        if re.search(_SPEC, role):
            continue
        specs = list(re.finditer(rf"\(\s*({_SPEC})\s*\)", rest))
        if not specs:
            continue
        cursor = 0
        for found in specs:
            add(_expand(re.sub(r"^M-?", "", found.group(1), count=1, flags=re.I)), role, rest[cursor:found.start()])
            cursor = found.end()
            cursor += len(rest[cursor:]) - len(rest[cursor:].lstrip(" ,"))
    return per_track


def parse(body: str, album_id: int | str, *, fetched_at: str | None = None, raw_sha256: str | None = None) -> Release | None:
    """The Release a VGMdb album page (live or archived HTML) describes, or None when it is not an album page."""
    title_block = re.search(r'<h1>(.*?)</h1>', body, re.S)
    titles = _lang_spans(title_block.group(1), "albumtitle") if title_block else {}
    if not titles:
        return None
    title = titles.get("ja") or titles.get("en") or next(iter(titles.values()))

    credits_at = body.find('id="collapse_credits"')
    list_at = body.find('id="tlnav"')
    head = body[: credits_at if credits_at > 0 else (list_at if list_at > 0 else len(body))]
    info = _info_rows(head)
    credits_region = body[credits_at: list_at] if 0 < credits_at < list_at else ""
    credits, lookup = _credits(credits_region)

    catalog = _flat(info["Catalog Number"]) if "Catalog Number" in info else None
    barcode = _flat(info["Barcode"]) if "Barcode" in info else None
    release_date = _iso_date(_flat(info["Release Date"])) if "Release Date" in info else None
    label = None
    if "Label" in info:
        names = _lang_spans(info["Label"], "productname")
        label = names.get("ja") or names.get("en") or _flat(info["Label"]) or None

    lists = _tracklists(body)
    keys = sorted({key for entries in lists.values() for key in entries})
    notes_match = re.search(r'<div[^>]*id="notes"[^>]*>(.*?)</div>', body, re.S)
    notes = _text(notes_match.group(1)).strip() if notes_match else None
    notes = re.sub(r"[ \t]+\n", "\n", notes).strip() if notes else None
    per_track = _note_credits(notes or "", lookup)
    single_disc = len({disc for disc, _ in keys}) == 1   # note M-xx are track numbers only on one disc
    tracks = []
    for disc, number in keys:
        by_lang = {lang: entries[(disc, number)] for lang, entries in lists.items() if (disc, number) in entries}
        names = {lang: value[0] for lang, value in by_lang.items()}
        length = next((v[1] for v in by_lang.values() if v[1] is not None), None)
        shown = names.get("ja") or names.get("en") or next(iter(names.values()))
        tracks.append(Track(disc=disc, number=number, title=shown, titles=names, length_seconds=length,
                            credits=tuple(per_track.get(number, ())) if single_disc else ()))
    discs = len({disc for disc, _ in keys}) or None

    stamp = re.search(r"/web/(\d{14})", body)
    related = (SNAPSHOT_URL.format(ts=stamp.group(1), id=album_id),) if stamp else ()
    return Release(source=SOURCE, url=ALBUM_URL.format(id=album_id), title=title, titles=titles,
                   release_date=release_date, catalog=catalog or None, barcode=barcode or None, label=label,
                   discs=discs, tracks=tuple(tracks), credits=credits, notes=notes or None, related_urls=related,
                   fetched_at=fetched_at, raw_sha256=raw_sha256)


def album(album_id: int | str, state_root, *, opener=None, courtesy=None) -> Release | None:
    """The newest archived VGMdb snapshot of an album as a Release; None when none exists or it cannot be read."""
    try:
        number = int(str(album_id).strip())
    except ValueError:
        return None
    try:
        entry = fetch.get(ARCHIVE_URL.format(id=number), Path(state_root), accept="text/html", opener=opener,
                          courtesy=courtesy, purpose="vgmdb-archive")
    except fetch.Unavailable:
        return None
    return parse(entry["body"], number, fetched_at=entry.get("fetched_at"), raw_sha256=entry.get("sha256"))
