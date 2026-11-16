"""Published spellings witnessed in timed ASR, never published scene authority.

The comparison is deliberately lexical: whole katakana tokens, no edit distance,
no character/episode aliases. Voicing and long-vowel differences remain inferred
corrections, with the raw observation and public source retained separately.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

TOKEN = re.compile(r"[ァ-ヴー]{2,}")
POLICY = "published-katakana-whole-token-v1"
VOWELS = {}
for _vowel, _letters in (("ア", "アカサタナハマヤラワァャ"), ("イ", "イキシチニヒミリヰィ"),
                          ("ウ", "ウクスツヌフムユルゥュ"), ("エ", "エケセテネヘメレヱェ"),
                          ("オ", "オコソトノホモヨロヲォョ")):
    VOWELS.update(dict.fromkeys(_letters, _vowel))


def _key(word: str) -> str:
    plain = "".join(c for c in unicodedata.normalize("NFKD", word) if not unicodedata.combining(c))
    result = []
    for char in plain:
        if char == "ー":
            if not result or result[-1] not in VOWELS:
                return ""  # no guess about an unpronounceable long-vowel mark
            char = VOWELS[result[-1]]
        result.append(char)
    return "".join(result)


def source_headers(sources: list[dict]) -> list[dict]:
    """Bind stored support to the exact bounded public text, including legacy caches."""
    result = []
    for source in sources:
        text = source.get("text")
        if not isinstance(text, str) or source.get("source_kind") not in ("official_episode", "wikipedia_work"):
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if source.get("text_sha256") not in (None, digest):
            continue
        if not isinstance(source.get("url"), str) or not source["url"].startswith("https://"):
            continue
        result.append({**{k: source[k] for k in ("source_kind", "url", "fetched_at", "page_text_sha256") if k in source},
                       "text_sha256": digest})
    return result


def witnessed(fields: list[str | None], observed: list[dict], sources: list[dict]) -> list[dict]:
    """Return correction witnesses only; callers must still validate scene quotes."""
    valid = []
    vocabulary = {}
    for source in sources:
        headers = source_headers([source])
        if not headers:
            continue
        valid.append((source, headers[0]))
        for word in set(TOKEN.findall(unicodedata.normalize("NFKC", source["text"]))):
            key = _key(word)
            if key:
                vocabulary.setdefault(key, set()).add(word)
    proposed = sorted({word for field in fields for word in TOKEN.findall(field or "")})
    result = []
    for canonical in proposed:
        key = _key(canonical)
        for source, header in valid:
            words = set(TOKEN.findall(unicodedata.normalize("NFKC", source["text"])))
            if canonical not in words:
                continue
            for line in observed:
                for raw in sorted(set(TOKEN.findall(unicodedata.normalize("NFKC", line["text"])))):
                    exact = raw == canonical
                    if not exact and not (len(canonical) >= 4 and len(raw) >= 4 and key and _key(raw) == key
                                          and vocabulary.get(key) == {canonical}):
                        continue
                    result.append({"canonical": canonical, "observed_token": raw,
                                   "observed_text": line["text"], "start": line["start"], "end": line["end"],
                                   "match": "exact_token" if exact else "inferred_voicing_long_vowel",
                                   "policy": POLICY, "source": header})
    return result
