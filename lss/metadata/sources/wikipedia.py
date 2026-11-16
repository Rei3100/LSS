"""Series facts from the work's own Japanese Wikipedia article: first date, 音楽 staff names, theme songs.

The article is the exact work title page; when that is a franchise page, a linked ``<title> (…)`` page whose text
lists the program's episode subtitle verbatim (so a franchise name never borrows another series' facts). Raw HTML
is read through `lss.metadata.fetch` (robots.txt, courtesy pacing, cache); offline or no suitable article: None.
Everything is kept as the page wrote it (footnote marks removed); nothing is inferred.
"""
from __future__ import annotations

import html as html_module
import os
import re
import unicodedata
import urllib.parse
from pathlib import Path

from lss.bgm.work_credits import staff_music
from lss.bgm.work_info import _parse_details
from lss.metadata import fetch
from lss.metadata.model import Credit, Series

WIKI = "https://ja.wikipedia.org/wiki/"
_QUOTED = r"(?:\"[^\"]*\"|'[^']*'|[^'\">])*"
_TABLE = re.compile(r"<table\b(" + _QUOTED + r")>", re.I)
_ROW = re.compile(r"<th\b" + _QUOTED + r">(.*?)</th>\s*<td\b" + _QUOTED + r">(.*?)</td>", re.S | re.I)
_DATE = re.compile(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日")
_MONTH = re.compile(r"(\d{4})年\s*(\d{1,2})月")
_YEAR_ONLY = re.compile(r"^(\d{4})年$")
_MD = re.compile(r"^(\d{1,2})月(\d{1,2})日$")
_REF = re.compile(r"\[[^\]]*\]|［[^］]*］")
_HIRAGANA = re.compile(r"[ぁ-ゟ]")
_SECTION_END = {"放送日程", "各話リスト", "各話一覧", "放送・配信", "放送リスト", "サブタイトル", "配信"}


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text or "").split())


def _iso(text: str) -> str | None:
    found = _DATE.search(text or "")
    if found:
        return f"{int(found.group(1)):04d}-{int(found.group(2)):02d}-{int(found.group(3)):02d}"
    found = _MONTH.search(text or "")
    if found:
        return f"{int(found.group(1)):04d}-{int(found.group(2)):02d}"
    return None


def _cell_text(fragment: str) -> str:
    fragment = re.sub(r"<(sup|style|script)\b.*?</\1>", "", fragment, flags=re.S | re.I)
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.I)
    text = html_module.unescape(re.sub(r"<[^>]+>", "", fragment))
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _infobox(html: str) -> dict:
    """Label -> value text of the first infobox that states a broadcast period or a release."""
    for opened in _TABLE.finditer(html):
        if "infobox" not in opened.group(1):
            continue
        end = html.find("</table>", opened.end())
        rows = {}
        for label, value in _ROW.findall(html[opened.end():end if end >= 0 else len(html)]):
            label = _cell_text(label)
            if label and label not in rows:
                rows[label] = _cell_text(value)
        if any(key in rows for key in ("放送期間", "公開", "放送開始")):
            return rows
    return {}


def _title_of(url: str) -> str | None:
    if not url.startswith(WIKI):
        return None
    return urllib.parse.unquote(url[len(WIKI):].split("#", 1)[0]).replace("_", " ")


def _page(title: str, state_root: Path, opener, courtesy, strict: bool = False) -> dict | None:
    """The parsed article, or None when there is none. ``strict``: could-not-ask raises `fetch.Unavailable`."""
    url = WIKI + urllib.parse.quote(title, safe="")
    try:
        entry = fetch.get(url, state_root, opener=opener, courtesy=courtesy, purpose="wikipedia")
    except fetch.Unavailable as exc:
        if strict and not fetch.permanent(exc):
            raise
        return None
    text, links, h1 = _parse_details(entry["body"])
    return {"title": title, "h1": h1, "text": text, "lines": text.splitlines(), "links": links, "entry": entry,
            "disambiguation": bool(re.search(r'PageProp/Category"[^>]*曖昧さ回避', entry["body"])),
            "infobox": _infobox(entry["body"])}


# --- date ---------------------------------------------------------------------------------------------------

def _schedule_first_date(lines: list[str]) -> str | None:
    """First dated row of the 放送日程 table (a year cell followed by 月日, or a full date)."""
    for head in [i for i, line in enumerate(lines) if line == "放送日程"]:   # the first is the contents list
        year = None
        for line in lines[head + 1:head + 80]:
            if _DATE.fullmatch(line.strip()):
                return _iso(line)
            ym = _YEAR_ONLY.match(line)
            if ym:
                year = ym.group(1)
                continue
            md = _MD.match(line)
            if md and year:
                return f"{int(year):04d}-{int(md.group(1)):02d}-{int(md.group(2)):02d}"
    return None


def _first_date(page: dict, kind: str) -> str | None:
    box = page["infobox"]
    if kind == "movie":
        return _iso(box.get("公開", ""))
    return _iso(box.get("放送期間", "")) or _iso(box.get("放送開始", "")) or _schedule_first_date(page["lines"])


# --- episode broadcast dates ----------------------------------------------------------------------------------

_ROW_HTML = re.compile(r"<tr\b" + _QUOTED + r">(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<(t[hd])\b(" + _QUOTED + r")>(.*?)</\1>", re.S | re.I)
_SPAN = re.compile(r"(rowspan|colspan)\s*=\s*[\"']?(\d+)", re.I)
_KANJI = {c: i for i, c in enumerate("〇一二三四五六七八九")}
_NUMBER = re.compile(r"^(?:第|no\.?\s*|#)?\s*(\d+|[〇一二三四五六七八九十百]+)\s*話?$", re.I)
_DATE_PRIORITY = ("初放送日", "放送日")


def _number(text: str) -> int | None:
    found = _NUMBER.match(_norm(text))
    if not found:
        return None
    digits = found.group(1)
    if digits.isdigit():
        return int(digits)
    total, current = 0, 0
    for ch in digits:
        if ch == "百":
            total, current = total + (current or 1) * 100, 0
        elif ch == "十":
            total, current = total + (current or 1) * 10, 0
        else:
            current = current * 10 + _KANJI[ch]
    return total + current or None


def _grid(table: str) -> list[list[str]]:
    """Rows of cell texts with rowspan / colspan expanded (a merged cell repeats its text)."""
    grid, carry = [], {}
    for body in _ROW_HTML.findall(table):
        row, col = [], 0
        for _, attrs, content in _CELL.findall(body):
            while col in carry:
                text, left = carry[col]
                row.append(text)
                carry[col] = (text, left - 1)
                if left - 1 <= 0:
                    del carry[col]
                col += 1
            spans = {k.lower(): int(v) for k, v in _SPAN.findall(attrs)}
            text = _cell_text(content).replace(chr(10), " ")
            for _ in range(min(spans.get("colspan", 1), 20)):
                row.append(text)
                if spans.get("rowspan", 1) > 1:
                    carry[col] = (text, spans["rowspan"] - 1)
                col += 1
        while col in carry:
            text, left = carry[col]
            row.append(text)
            if left - 1 <= 0:
                del carry[col]
            else:
                carry[col] = (text, left - 1)
            col += 1
        grid.append(row)
    return grid


def _schedule_tables(html: str) -> list[tuple[str, list[list[str]]]]:
    """(date column label, rows) of every table whose header has 話数 and a 放送日 / 配信日 column."""
    found = []
    for opened in _TABLE.finditer(html):
        end = html.find("</table>", opened.end())
        grid = _grid(html[opened.end():end if end >= 0 else len(html)])
        for at, row in enumerate(grid[:3]):
            cells = [_norm(_REF.sub("", c)) for c in row]
            if "話数" not in cells:
                continue
            date = next((i for i, c in enumerate(cells) if c.endswith("放送日") or c.endswith("配信日")), None)
            if date is not None:
                found.append((cells[date], cells.index("話数"), date, grid[at + 1:]))
            break
    return [(label, [(r[n] if n < len(r) else "", r[d] if d < len(r) else "") for r in rows])
            for label, n, d, rows in found]


def _episode_dates(html: str) -> tuple[tuple[int, str], ...]:
    """(episode number, ISO date) from the 放送日程 tables: year-less dates continue the previous row's year
    (a smaller month starts the next year); un-numbered rows (総集編, 特別編) are skipped; the first row wins."""
    tables = _schedule_tables(html)
    label = next((t[0] for p in _DATE_PRIORITY for t in tables if t[0].endswith(p)), None)
    if label is not None:   # every broadcast table of the series (one per cour)
        chosen = [t for t in tables if t[0].endswith(label)]
    else:                   # only streaming dates exist: the first table is the main run
        chosen = tables[:1]
    year = month = None
    dates: dict[int, str] = {}
    for _, rows in chosen:
        for number_text, date_text in rows:
            text = _norm(_REF.sub("", date_text))
            full = _DATE.search(text)
            short = re.match(r"^(\d{1,2})月\s*(\d{1,2})日", text)
            if full:
                year, month, day = int(full.group(1)), int(full.group(2)), int(full.group(3))
            elif short and year is not None:
                new_month, day = int(short.group(1)), int(short.group(2))
                if month is not None and new_month < month:
                    year += 1
                month = new_month
            else:
                continue
            number = _number(number_text)
            if number is not None and number not in dates:
                dates[number] = f"{year:04d}-{month:02d}-{day:02d}"
    return tuple(sorted(dates.items()))


# --- songs --------------------------------------------------------------------------------------------------

_ROLE = {"作詞": "lyricist", "作詩": "lyricist", "作曲": "composer", "編曲": "arranger", "歌": "vocals",
         "歌唱": "vocals", "ボーカル": "vocals", "ヴォーカル": "vocals", "コーラス": "chorus", "演奏": "performer"}
_ROLE_WORDS = "|".join(sorted(_ROLE, key=len, reverse=True))
_ROLES = rf"(?:{_ROLE_WORDS})(?:[・、](?:{_ROLE_WORDS}))*"
_STRUCT = re.compile(rf"^(?P<roles>{_ROLES})\s*[-－–―]\s*(?P<names>.+)$")
_PROSE = re.compile(rf"(?P<roles>{_ROLES})は(?P<names>.+?)(?=、(?:{_ROLE_WORDS})[・は]|。|$)")
_TITLE = re.compile(r"^(?P<kind>[^「」【】]{0,12})「(?P<title>[^」]+)」(?P<rest>.*)$")
_KINDS = ("オープニングテーマ", "エンディングテーマ", "挿入歌", "主題歌", "劇中歌", "イメージソング")


def _split_names(text: str, *, conjunction: bool) -> list[str]:
    """Names in a credit value: split on 、, on ・ between names, and (prose only) on と between name characters."""
    text = _REF.sub("", text).strip(" 　。")
    parts, depth, buf = [], 0, ""
    for i, ch in enumerate(text):
        if ch in "（(":
            depth += 1
        elif ch in "）)":
            depth = max(0, depth - 1)
        cut = False
        if depth == 0:
            if ch == "、":
                cut = True
            elif ch == "・":
                cut = len(buf.strip()) >= 2 and len(text[i + 1:].split("、")[0].strip()) >= 2
            elif conjunction and ch == "と" and buf and i + 1 < len(text):
                before, after = buf[-1], text[i + 1]
                cut = before not in "（(" and not (_HIRAGANA.match(before) and _HIRAGANA.match(after))
        if cut:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    names = []
    for part in parts:
        name = part.strip(" 　")
        if name and name not in names:
            names.append(name)
    return names


def _credits(roles: str, names: list[str]) -> list[Credit]:
    out = []
    for role in re.split(r"[・、]", roles):
        out.extend(Credit(role=role, role_key=_ROLE[role], name=name, names=(name,)) for name in names)
    return out


def _line_credits(line: str, *, first_prose: bool) -> list[Credit]:
    """Credits on one line below a song title: ``作詞 - A / 作曲 - B`` or prose ``…による…。作詞はA、作曲はB。``."""
    line = _REF.sub("", line).strip()
    structured = [m for m in (_STRUCT.match(s.strip()) for s in re.split(r"\s/\s", line)) if m]
    found: list[Credit] = []
    if structured:
        for m in structured:
            found.extend(_credits(m.group("roles"), _split_names(m.group("names"), conjunction=False)))
        return found
    for sentence in line.split("。"):
        if first_prose and "による" in sentence and not found:
            name = sentence.split("による", 1)[0].strip()
            if name and len(name) <= 60 and "「" not in name:
                found.append(Credit(role="歌", role_key="vocals", name=name, names=(name,)))
        for m in _PROSE.finditer(sentence):
            found.extend(_credits(m.group("roles"), _split_names(m.group("names"), conjunction=True)))
    return found


def _songs(lines: list[str]) -> tuple[dict, ...]:
    """Theme / insert songs from the 主題歌 section that follows the main staff list."""
    staff = next((i for i, line in enumerate(lines) if re.match(r"^音楽\s*[-－–―]", line)), None)
    if staff is None:
        return ()
    start = next((i for i in range(staff + 1, len(lines))
                  if lines[i] == "主題歌" or lines[i].startswith("主題歌「")), None)
    if start is None:
        return ()
    songs: list[dict] = []
    category, current = "", None
    for i in range(start, len(lines)):
        line = lines[i]
        if i > start and line in _SECTION_END:
            break
        if line in _KINDS:
            category, current = line, None
            continue
        if line.startswith("【") or line.startswith("映像外部リンク"):
            continue
        m = _TITLE.match(line)
        if m and not _REF.sub("", m.group("rest")).strip():
            kind = next((k for k in _KINDS if k in m.group("kind")), category)
            current = {"title": m.group("title"), "kind": kind, "credits": [], "prose": False}
            songs.append(current)
            continue
        if current is None:
            continue
        current["credits"].extend(_line_credits(line, first_prose=not current["prose"]))
        if re.search(rf"{_ROLES}は|による", line):
            current["prose"] = True
    return tuple({"title": s["title"], "kind": s["kind"], "credits": tuple(s["credits"])} for s in songs)


# --- series -------------------------------------------------------------------------------------------------

def _build(page: dict, kind: str, listed: bool, match: str | None = None) -> Series | None:
    box = page["infobox"]
    own = "tv" if ("放送期間" in box or "放送開始" in box) else ("movie" if "公開" in box else None)
    wanted = "movie" if kind == "movie" else "tv"
    if own != wanted:
        return None
    entry = page["entry"]
    return Series(source="wikipedia", url=WIKI + urllib.parse.quote(page["title"], safe=""),
                  title=page["h1"] or page["title"], kind=wanted, first_date=_first_date(page, wanted),
                  music=tuple(staff_music(page["text"])), episode_listed=listed, songs=_songs(page["lines"]),
                  episode_dates=_episode_dates(entry["body"]) if wanted == "tv" else (),
                  fetched_at=entry.get("fetched_at"), raw_sha256=entry.get("sha256"),
                  match=match or ("subtitle_listed" if listed else None))


CONTINUATION_WINDOW = 8   # an ongoing article may trail the broadcast by up to this many episodes


def _open_ended(page: dict) -> bool:
    """The infobox 放送期間 states a start date and no end (still airing)."""
    period = _norm(page["infobox"].get("放送期間", ""))
    found = _DATE.search(period)
    return bool(found) and not re.search(r"\d", period[found.end():])


def _continues(page: dict, episode_number: int) -> bool:
    """An ongoing article that has not caught up: its last listed episode N < episode_number <= N + window."""
    if not _open_ended(page):
        return False
    numbers = [n for n, _ in _episode_dates(page["entry"]["body"])]
    return bool(numbers) and max(numbers) < episode_number <= max(numbers) + CONTINUATION_WINDOW


def series(work_title: str, episode_subtitle: str | None, kind: str, state_root: Path, *, opener=None,
           courtesy=None, episode_number: int | None = None, strict: bool = False) -> Series | None:
    """The work's own article facts, or None (offline, refused, no article that is this kind of program).

    A TV article is taken when it lists the episode subtitle (``match="subtitle_listed"``); failing that, only
    when it is still airing (open-ended 放送期間) and its last listed episode is less than ``episode_number`` by at
    most `CONTINUATION_WINDOW` (``match="ongoing_continuation"``). Nothing else is inferred.
    ``strict``: an article that could not be read (offline, server error) raises `fetch.Unavailable` instead of
    reading as "no such series".
    """
    if not work_title or not work_title.strip() or os.environ.get("LSS_WORK_INFO") == "0":
        return None
    top = _page(work_title.strip(), state_root, opener, courtesy, strict)
    if top is None or _norm(top["h1"]) != _norm(work_title):
        return None
    subtitle = _norm(episode_subtitle) if episode_subtitle else ""
    listed = bool(subtitle) and subtitle in _norm(top["text"])
    # a disambiguation page is only a source of links, never a series itself
    own = None if top["disambiguation"] else _build(top, kind, listed)
    if own is not None and listed:
        return own
    pages = []
    if not top["disambiguation"] and own is not None:
        pages.append(top)
    if subtitle or episode_number:
        prefix = _norm(work_title) + " ("
        candidates = []
        for link in top["links"]:
            title = _title_of(link)
            if title and _norm(title).startswith(prefix) and title not in candidates:
                candidates.append(title)
        for title in candidates[:12]:
            child = _page(title, state_root, opener, courtesy, strict)
            if child is None or _norm(child["h1"]) != _norm(title) or child["disambiguation"]:
                continue
            if subtitle and subtitle in _norm(child["text"]):
                built = _build(child, kind, True)
                if built is not None:
                    return built
            pages.append(child)
    if episode_number and kind != "movie":
        for page in pages:
            if _continues(page, episode_number):
                built = _build(page, kind, False, "ongoing_continuation")
                if built is not None:
                    return built
    return own
