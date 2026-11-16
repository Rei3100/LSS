"""The work's music credit (劇伴 composer) from its published staff list, for BGM ``ARTIST``.

Pages come through `lss.bgm.work_info.fetch` (cache, robots.txt, courtesy pacing, offline = unavailable).
The page is the work's own Japanese Wikipedia article: the exact work title (title-verified, not a
disambiguation page), or a page that article links to as ``<work> (…)`` whose text lists this episode's
subtitle verbatim, so a franchise name never borrows another series' composer. The credit is the first
staff line ``音楽 - <names>`` (a ``→`` change keeps the names after it). Every name must appear verbatim in the
page; nothing is inferred, and no page or no staff line leaves ARTIST unknown.
"""
from __future__ import annotations

import os
import re
import unicodedata
import urllib.parse
from pathlib import Path

from lss.bgm import work_info

WIKI = "https://ja.wikipedia.org/wiki/"
_STAFF = re.compile(r"^\s*音楽\s*[-－–―:：]\s*(.+?)\s*$", re.MULTILINE)
_NOTE = re.compile(r"\[[^\]]*\]|［[^］]*］|（[^）]*）|\([^)]*\)")
_SPLIT = re.compile(r"[、，,／/]|・(?=[^\s・]{2,})")
_ROLE = re.compile(r"[-－–―:：]|音楽|作曲|編曲|協力|プロデューサー")


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text or "").split())


def staff_music(text: str) -> list[str]:
    """Names on the first ``音楽 - …`` staff line of a page's text (empty when there is none)."""
    found = _STAFF.search(text or "")
    if found is None:
        return []
    line = found.group(1).split("→")[-1]
    line = _NOTE.sub("", line)
    names = []
    for part in _SPLIT.split(line):
        name = part.strip(" 　")
        if not name or _ROLE.search(name) or len(name) > 40 or name in names:
            continue
        if name not in text:
            continue
        names.append(name)
    return names


def _title_of(url: str) -> str | None:
    if not url.startswith(WIKI):
        return None
    return urllib.parse.unquote(url[len(WIKI):].split("#", 1)[0]).replace("_", " ")


def credits(work: str, subtitle: str | None, state_root: Path, *, opener=None, courtesy=None) -> dict | None:
    """``{"composers", "url", "text_sha256", "matched"}`` or None (no work, offline, refused, no credit)."""
    if not work or not work.strip() or os.environ.get("LSS_WORK_INFO") == "0":
        return None
    def page(title):
        try:
            return work_info.fetch(WIKI + urllib.parse.quote(title, safe=""), state_root, opener=opener, courtesy=courtesy)
        except work_info.WorkInfoUnavailable:
            return None
    top = page(work)
    if top is None or _norm(top.get("first_h1")) != _norm(work):
        return None
    text = top.get("text") or ""
    if re.search(r"曖昧さ回避|disambiguation", text[:2000], re.I):
        return None
    candidates = []
    if subtitle:
        prefix = _norm(work) + " ("
        for link in top.get("links") or []:
            title = _title_of(link)
            if title and _norm(title).startswith(prefix) and title not in candidates:
                candidates.append(title)
    for title in candidates[:8]:
        child = page(title)
        if child is None or _norm(child.get("first_h1")) != _norm(title):
            continue
        body = child.get("text") or ""
        if _norm(subtitle) in _norm(body):
            names = staff_music(body)
            if names:
                return {"composers": names, "url": WIKI + urllib.parse.quote(title, safe=""),
                        "text_sha256": child.get("text_sha256"), "matched": "linked_page_lists_episode_subtitle"}
    names = staff_music(text)
    if names:
        return {"composers": names, "url": WIKI + urllib.parse.quote(work, safe=""),
                "text_sha256": top.get("text_sha256"), "matched": "work_title_page"}
    return None
