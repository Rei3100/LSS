"""Public facts about a work and its episodes that LSS fetches itself for naming (optional, never required).

LSS's own HTTP client reads public pages (the work's official episode list, Wikipedia), honours robots.txt for its
own user agent, paces requests through the shared acquisition courtesy, and keeps every fetched page in the state
area with its URL and time, so a page is read once. The naming provider returns published spelling vocabulary from
the episode or verified work page. This can correct spelling against heard speech; it does not establish which scene
the music accompanied. Offline, refused by robots.txt, or nothing found: naming continues without public spelling
context.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser
from pathlib import Path

USER_AGENT = "LSS-work-info/1 (local personal media tool; fetches public episode pages once and caches them)"
DIRECTORY = "bgm-work-info"
TIMEOUT = 30
MAX_BYTES = 4 * 1024 * 1024


class WorkInfoUnavailable(RuntimeError):
    """The page cannot be read (offline, refused by robots.txt, error); naming continues without it."""


def _root(state_root: Path) -> Path:
    return Path(state_root) / DIRECTORY


class _Text(HTMLParser):
    BLOCK = {"p", "div", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "br", "section", "article", "dt", "dd"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.links = [], 0, []
        self.in_h1, self.h1_count, self.h1 = 0, 0, []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag in ("script", "style", "noscript", "svg"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        if tag == "h1":
            if not self.in_h1:
                self.h1_count += 1
            self.in_h1 += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        if tag == "h1" and self.in_h1:
            self.in_h1 -= 1

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)
            if self.in_h1 and self.h1_count == 1:
                self.h1.append(data)


def page_text(html: str) -> str:
    return _parse(html)[0]


def _parse(html: str) -> tuple[str, list[str]]:
    text, links, _ = _parse_details(html)
    return text, links


def _parse_details(html: str) -> tuple[str, list[str], str | None]:
    parser = _Text()
    parser.feed(html)
    lines = [" ".join(line.split()) for line in "".join(parser.out).splitlines()]
    text = "\n".join(line for line in lines if line)
    h1 = " ".join(" ".join(parser.h1).split()) or None
    return text, parser.links, h1


def _allowed(url: str, state_root: Path, *, opener) -> bool:
    parts = urllib.parse.urlsplit(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    cached = _root(state_root) / "robots" / (hashlib.sha256(robots_url.encode()).hexdigest()[:24] + ".txt")
    try:
        if cached.is_file() and time.time() - cached.stat().st_mtime < 7 * 86400:
            body = cached.read_text(encoding="utf-8")
        else:
            body = opener(robots_url)
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(body, encoding="utf-8")
    except WorkInfoUnavailable:
        body = ""      # no robots.txt: the site states no restriction
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(body.splitlines())
    return parser.can_fetch(USER_AGENT, url)


def _http(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.5"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 - public https pages only
            raw = response.read(MAX_BYTES + 1)
            charset = response.headers.get_content_charset() or "utf-8"
    except (OSError, ValueError) as exc:
        raise WorkInfoUnavailable(f"{url}: {type(exc).__name__}") from exc
    if len(raw) > MAX_BYTES:
        raise WorkInfoUnavailable(f"{url}: too large")
    try:
        return raw.decode(charset, errors="replace")
    except LookupError as exc:  # a page declaring a charset Python does not know
        raise WorkInfoUnavailable(f"{url}: unknown charset") from exc


def fetch(url: str, state_root: Path, *, opener=None, courtesy=None) -> dict:
    """The page as stored text (fetched once; robots.txt and pacing honoured). Raises WorkInfoUnavailable."""
    if not url.startswith("https://"):
        raise WorkInfoUnavailable("https only")
    store = _root(state_root) / "pages" / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
    if store.is_file():
        try:
            return json.loads(store.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass  # a torn or unreadable copy is no page: fetch it again rather than fail for ever
    opener = opener or _http
    if not _allowed(url, state_root, opener=opener):
        raise WorkInfoUnavailable(f"{url}: robots.txt does not allow {USER_AGENT.split('/')[0]}")
    if courtesy is None:
        from lss.acquisition import courtesy as courtesy_module

        courtesy = courtesy_module.shared(Path(state_root))
    with courtesy.slot(url, purpose="work_info"):
        try:
            html = opener(url)
            courtesy.report(url)
        except WorkInfoUnavailable:
            courtesy.report(url, "transient")
            raise
    text, links, first_h1 = _parse_details(html)
    text = text[:200000]
    entry = {"url": url, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "user_agent": USER_AGENT,
             "text": text, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
             "first_h1": first_h1, "links": [urllib.parse.urljoin(url, link) for link in links][:2000]}
    from lss.core.session import _write_json_atomic

    _write_json_atomic(store, entry)
    return entry


EPISODE_HEAD = re.compile(r"第\s*(\d{1,4})\s*話")


def episode_blocks(text: str) -> dict[int, str]:
    """Split a page's text into blocks that start with 第N話 (official episode lists use this form)."""
    blocks, current, lines = {}, None, []
    for line in text.splitlines():
        found = EPISODE_HEAD.search(line)
        if found and line.find(found.group(0)) < 8:
            if current is not None:
                blocks.setdefault(current, "\n".join(lines))
            current, lines = int(found.group(1)), [line]
        elif current is not None:
            lines.append(line)
    if current is not None:
        blocks.setdefault(current, "\n".join(lines))
    return {n: block[:3000] for n, block in blocks.items()}


LISTING_HINT = re.compile(r"episode|story|stories|あらすじ|ストーリー", re.I)


def official_site(description: str) -> str | None:
    """The official site the source file's own description points to (observed, not searched)."""
    lines = (description or "").splitlines()
    for i, line in enumerate(lines):
        if "公式サイト" in line or "公式HP" in line:
            for candidate in (line, lines[i + 1] if i + 1 < len(lines) else ""):
                found = re.search(r"https://[^\s　）)」]+", candidate)
                if found:
                    return found.group(0)
    return None


def episode_listing(site: str, state_root: Path, *, opener=None, courtesy=None) -> str | None:
    """The site's episode/story listing linked from its top page (same site only)."""
    top = fetch(site, state_root, opener=opener, courtesy=courtesy)
    base = urllib.parse.urlsplit(site)
    candidates = []
    for link in top.get("links") or []:
        parts = urllib.parse.urlsplit(link)
        if parts.netloc == base.netloc and parts.path.startswith(base.path) and LISTING_HINT.search(parts.path):
            candidates.append(urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")))
    candidates.sort(key=lambda u: (u.count("/"), len(u)))
    return candidates[0] if candidates else None


def episode_synopsis(description: str, episode: int | None, state_root: Path, *, opener=None, courtesy=None) -> dict | None:
    """This episode's block from the official listing, or None (no site, refused, offline, not listed)."""
    if episode is None:
        return None
    site = official_site(description)
    if site is None:
        return None
    try:
        listing = episode_listing(site, state_root, opener=opener, courtesy=courtesy)
        if listing is None:
            return None
        blocks = official_episode_pages(listing, state_root, opener=opener, courtesy=courtesy, until=episode)
    except WorkInfoUnavailable:
        return None
    found = blocks.get(episode)
    return {"episode": episode, **found, "site": site} if found else None


def _normalized_title(value: str) -> str:
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", value or "").split())


def naming_context(description: str, episode: int | None, work_title: str, state_root: Path, *,
                   opener=None, courtesy=None) -> dict | None:
    """Published naming context, preferring an official episode block then a title-verified Wikipedia page."""
    official = None
    site = official_site(description)
    if site and episode is not None:
        try:
            listing = episode_listing(site, state_root, opener=opener, courtesy=courtesy)
            blocks = (official_episode_pages(listing, state_root, opener=opener, courtesy=courtesy, until=episode)
                      if listing else {})
        except WorkInfoUnavailable:
            blocks = {}
        found = blocks.get(episode)
        if found:
            official = {"episode": episode, **found, "site": site}
    if official:
        text = official["text"]
        return {"source_kind": "official_episode", "url": official["url"], "text": text,
                "site": official["site"], "fetched_at": official.get("fetched_at"),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "page_text_sha256": _cached_text_sha256(official["url"], state_root)}

    if not work_title or not work_title.strip():
        return None
    url = "https://ja.wikipedia.org/wiki/" + urllib.parse.quote(work_title, safe="")
    try:
        page = fetch(url, state_root, opener=opener, courtesy=courtesy)
    except WorkInfoUnavailable:
        return None
    if _normalized_title(page.get("first_h1") or "") != _normalized_title(work_title):
        return None
    text = (page.get("text") or "")[:20000]
    if not text or re.search(r"曖昧さ回避|disambiguation", text[:2000], re.I):
        return None
    return {"source_kind": "wikipedia_work", "url": url, "text": text,
            "site": "https://ja.wikipedia.org", "fetched_at": page.get("fetched_at"),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "page_text_sha256": page.get("text_sha256")}


def _cached_text_sha256(url: str, state_root: Path) -> str | None:
    store = _root(state_root) / "pages" / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
    try:
        page = json.loads(store.read_text(encoding="utf-8"))
        return page.get("text_sha256") or (hashlib.sha256(page["text"].encode("utf-8")).hexdigest()
                                            if isinstance(page.get("text"), str) else None)
    except (OSError, ValueError, KeyError, AttributeError):
        return None


def official_episode_pages(listing_url: str, state_root: Path, *, pages: int = 12, opener=None, courtesy=None,
                           until: int | None = None) -> dict:
    """Episode blocks from an official listing and its numbered pagination (index_N.html), stopping at a gap or
    once episode ``until`` is found (so only the pages needed are read)."""
    found, urls = {}, [listing_url]
    base = listing_url if listing_url.endswith("/") else listing_url.rsplit("/", 1)[0] + "/"
    urls += [f"{base}index_{n}.html" for n in range(2, pages + 1)]
    for url in urls:
        try:
            entry = fetch(url, state_root, opener=opener, courtesy=courtesy)
        except WorkInfoUnavailable:
            break
        blocks = episode_blocks(entry["text"])
        if not blocks:
            break
        for number, block in blocks.items():
            found.setdefault(number, {"text": block, "url": url, "fetched_at": entry.get("fetched_at"),
                                      "page_text_sha256": entry.get("text_sha256")})
        if until is not None and until in found:
            break
    return found
