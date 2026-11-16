"""Deterministic Windows-safe presentation names; never asset identities."""
import hashlib
import re


def component(value: str, *, limit: int = 160) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).rstrip(" .") or "LSS"
    if re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", cleaned, re.I):
        cleaned = "_" + cleaned
    if len(cleaned.encode("utf-16-le")) // 2 > limit:
        token = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
        prefix, units = [], 0
        for character in cleaned:
            size = len(character.encode("utf-16-le")) // 2
            if units + size > limit - 13:
                break
            prefix.append(character)
            units += size
        cleaned = "".join(prefix).rstrip(" .") + " - " + token
    return cleaned
