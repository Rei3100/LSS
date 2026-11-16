"""Optional, uncertain OCR evidence from one still frame.

The returned text is observation only. It must not be treated as a fact, scene,
credit, or identity without independent evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import subprocess
from typing import Callable

_POWERSHELL = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
_SCRIPT = Path(__file__).with_name("windows_frame_text.ps1")
_TIMEOUT_SECONDS = 45
_IS_WINDOWS = os.name == "nt"
_LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")


class FrameTextIntegrityError(RuntimeError):
    """The supplied image or the source during observation failed integrity checks."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unchanged(path: Path, expected: str, label: str) -> None:
    try:
        actual = _sha256(path)
    except OSError as exc:
        raise FrameTextIntegrityError(f"{label} became unavailable during OCR") from exc
    if actual != expected:
        raise FrameTextIntegrityError(f"{label} changed during OCR")


def _invalid_output(language: str, source_hash: str, script_hash: str,
                    stderr: str = "") -> dict:
    return {"status": "UNAVAILABLE", "reason": "invalid_runtime_output",
            "language": language, "source_sha256": source_hash,
            "script_sha256": script_hash, "stderr": stderr[:2000]}


def _valid_raw_line(line: object) -> bool:
    if not isinstance(line, dict) or not isinstance(line.get("text"), str):
        return False
    words = line.get("words")
    if not isinstance(words, list):
        return False
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("text"), str):
            return False
        box = word.get("box")
        if not isinstance(box, dict):
            return False
        coords = [box.get(key) for key in ("x", "y", "width", "height")]
        if any(not isinstance(value, Real) or isinstance(value, bool) or
               not math.isfinite(value) for value in coords):
            return False
        if box["width"] < 0 or box["height"] < 0:
            return False
    return True


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # A Python child of PowerShell7 can inherit a PSModulePath which hides the
    # Windows PowerShell5.1 system modules (including Get-FileHash). Use only
    # the selected host's built-in modules; never depend on profile modules.
    env = {key: value for key, value in os.environ.items() if key.upper() != "PSMODULEPATH"}
    env["PSModulePath"] = str(_POWERSHELL.parent / "Modules")
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, creationflags=flags,
                          check=False, env=env)


def observe(image_path: str | os.PathLike[str], expected_sha256: str,
            language: str = "ja-JP", *,
            _runner: Callable = _run) -> dict:
    """Read one frame using the pinned Windows PowerShell 5.1 WinRT OCR host.

    ``_runner`` is an internal seam for deterministic tests. Unsupported hosts,
    unavailable languages and runtime failures return ``status=UNAVAILABLE``.
    Source hash mismatches always raise :class:`FrameTextIntegrityError`.
    """
    path = Path(image_path).resolve()
    expected = str(expected_sha256).lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("expected_sha256 must be a 64-character SHA-256 hex digest")
    if not isinstance(language, str) or not _LANGUAGE_TAG.fullmatch(language):
        raise ValueError("language must be a BCP 47 language tag")
    try:
        before = _sha256(path)
    except OSError as exc:
        raise FrameTextIntegrityError("source image is unavailable") from exc
    if before != expected:
        raise FrameTextIntegrityError("source image does not match expected_sha256")
    if not _IS_WINDOWS or not _POWERSHELL.is_file() or not _SCRIPT.is_file():
        return {"status": "UNAVAILABLE", "reason": "unsupported_runtime",
                "language": language, "source_sha256": before}
    script_hash = _sha256(_SCRIPT)
    args = [str(_POWERSHELL), "-NoLogo", "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass", "-File", str(_SCRIPT),
            "-ImagePath", str(path), "-ExpectedSha256", expected, "-Language", language]
    try:
        completed = _runner(args, _TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _unchanged(path, before, "source image")
        _unchanged(_SCRIPT, script_hash, "provider script")
        return {"status": "UNAVAILABLE", "reason": "timeout", "language": language,
                "source_sha256": before, "script_sha256": script_hash}
    except OSError:
        _unchanged(path, before, "source image")
        _unchanged(_SCRIPT, script_hash, "provider script")
        return {"status": "UNAVAILABLE", "reason": "runtime_access_failure",
                "language": language, "source_sha256": before,
                "script_sha256": script_hash}
    _unchanged(path, before, "source image")
    _unchanged(_SCRIPT, script_hash, "provider script")
    try:
        if not isinstance(completed.stdout, str):
            raise TypeError("stdout was not text")
        payload = json.loads(completed.stdout.lstrip("\ufeff"))
    except (TypeError, json.JSONDecodeError):
        return _invalid_output(language, before, script_hash, completed.stderr or "")
    if not isinstance(payload, dict):
        return _invalid_output(language, before, script_hash, completed.stderr or "")
    reported_hash = payload.get("source_sha256")
    if reported_hash is not None and reported_hash != before:
        raise FrameTextIntegrityError("PowerShell source hash did not match expected image")
    if payload.get("status") == "OK" and reported_hash is None:
        raise FrameTextIntegrityError("PowerShell omitted source hash for successful OCR")
    common = {"language": language, "source_sha256": before,
              "script_sha256": script_hash,
              "powershell_executable": str(_POWERSHELL),
              "powershell_version": payload.get("powershell_version"),
              "width": payload.get("width"), "height": payload.get("height")}
    if completed.returncode != 0 or payload.get("status") != "OK":
        return {"status": "UNAVAILABLE", "reason": payload.get("reason", "runtime_failure"),
                "phase": payload.get("phase"), "error_type": payload.get("error_type"),
                "error_message": payload.get("error_message"),
                **common}
    if not isinstance(payload.get("powershell_version"), str) or not payload["powershell_version"].startswith("5.1."):
        return _invalid_output(language, before, script_hash, completed.stderr or "")
    width, height = payload.get("width"), payload.get("height")
    lines, raw_lines = payload.get("lines"), payload.get("raw_lines")
    if (not isinstance(width, int) or not isinstance(height, int) or
            not (0 < width <= 100_000 and 0 < height <= 100_000) or
            not isinstance(payload.get("recognized_text"), str) or
            not isinstance(lines, list) or not all(isinstance(line, str) for line in lines) or
            not isinstance(raw_lines, list) or not all(_valid_raw_line(line) for line in raw_lines)):
        return _invalid_output(language, before, script_hash, completed.stderr or "")
    return {"status": "OK", "recognized_text": payload.get("recognized_text", ""),
            "lines": lines, "raw_lines": raw_lines,
            "evidence_kind": "uncertain_ocr_observation", **common}
