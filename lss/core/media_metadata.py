"""Bounded reads of container-level media metadata through the bundled FFmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path


def _separator(line: str) -> int:
    escaped = False
    for index, char in enumerate(line):
        if char == "=" and not escaped:
            return index
        if char == "\\":
            escaped = not escaped
        else:
            escaped = False
    return -1


def _continued(value: str, has_newline: bool) -> tuple[str, bool]:
    slashes = len(value) - len(value.rstrip("\\"))
    return (value[:-1], True) if has_newline and slashes % 2 else (value, False)


def _unescape(value: str) -> str:
    out = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value) and value[index + 1] in "=;#\\":
            out.append(value[index + 1])
            index += 2
        else:
            out.append(value[index])
            index += 1
    return "".join(out)


def parse_ffmetadata(text: str) -> dict[str, str]:
    """Parse FFmpeg's UTF-8 ffmetadata output, returning global tags only.

    FFmetadata escapes punctuation with backslashes and encodes an actual value newline as
    a backslash at the end of one physical line. Metadata sections start with ``[`` and are
    deliberately excluded; chapter and stream titles are not source-wide titles.
    """
    lines = text.split("\n")
    if not lines or lines[0].removesuffix("\r") != ";FFMETADATA1":
        return {}
    tags: dict[str, str] = {}
    pending_key: str | None = None
    pending_value = ""

    def store() -> None:
        if pending_key is not None:
            tags[_unescape(pending_key)] = _unescape(pending_value)

    for index, physical in enumerate(lines[1:], start=1):
        has_newline = index < len(lines) - 1
        line = physical.removesuffix("\r")
        if pending_key is not None:
            fragment, continued = _continued(line, has_newline)
            pending_value += "\n" + fragment
            if continued:
                continue
            store()
            pending_key = None
            pending_value = ""
            continue
        if not line or line.startswith((";", "#")):
            continue
        if line.startswith("[") and line.endswith("]"):
            break
        split = _separator(line)
        if split < 0:
            continue
        pending_key, pending_value = line[:split], line[split + 1:]
        pending_value, continued = _continued(pending_value, has_newline)
        if not continued:
            store()
            pending_key = None
            pending_value = ""
    store()
    return tags


def format_tags(source: Path, *, timeout: float = 60) -> dict[str, str]:
    """Read container-wide tags without decoding audio or writing to the source media."""
    from lss.paths import is_frozen, resource_root

    ffmpeg = resource_root() / "bin" / "ffmpeg.exe"
    if ffmpeg.is_file():
        executable = str(ffmpeg)
    elif is_frozen():
        raise FileNotFoundError(ffmpeg)
    else:
        executable = "ffmpeg"  # developer convenience only; packaged runs fail closed
    result = subprocess.run([executable, "-nostdin", "-v", "error", "-i", str(source),
                              "-map_metadata", "0", "-f", "ffmetadata", "-"],
                            capture_output=True, check=True, timeout=timeout,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return parse_ffmetadata(result.stdout.decode("utf-8", errors="replace"))
