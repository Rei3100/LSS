"""Fail-closed local publication checks. Python 3.11+, standard library only."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile

DEFAULT_DENYLIST = (str(Path(os.environ["LSS_PRIVATE_ROOT"]) / "publish-denylist.txt")
                    if os.environ.get("LSS_PRIVATE_ROOT") else "")
POLICY_NAME = ".publish-allow.json"
WORD_PREFIX = "word:"
OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
MAX_POLICY_BYTES = 1024 * 1024
HARD_DIRS = {".git", ".state", "ep001-truth", "release-candidate",
             "manual-parity", "media", "models", "venv", ".venv", "__pycache__"}
TYPE_EXTENSIONS = frozenset("""
.wav .wave .flac .mp3 .aac .m4a .ogg .opus .aiff .aif .wma .mid .midi
.mp4 .m4v .mkv .mov .avi .webm .mpeg .mpg .wmv .ts
.gguf .ggml .safetensors .pth .pt .ckpt .onnx .bin .npz .npy .h5 .hdf5 .pb
.zip .gz .tgz .bz2 .xz .7z .rar .tar .whl .jar
.exe .dll .com .msi .so .dylib .pyc .pyo .db .sqlite .sqlite3
.png .jpg .jpeg .gif .ico .webp .bmp .tif .tiff .ttf .otf .woff .woff2 .pdf
""".split())
KEYWORD_SECRET = (
    r"(?i:" r"\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret"
    r"|password|passwd|authorization)\s*[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_./+=:@%-]{16,})")
# In source code an upper-case constant holding a long string is ordinary: the
# generic assignment branch applies to every other file, the keyword branch to all.
CODE_SUFFIXES = frozenset((".py", ".pyw", ".ps1", ".js", ".html"))
CODE_ENV_SECRET = re.compile(KEYWORD_SECRET)
# In source code a quoted string holds escape sequences: "\\dist\\lss" is a relative path and "h:\n" is a
# colon followed by a newline. There a drive path needs an upper-case letter, or a lower-case one followed by
# a doubled backslash or a slash; a UNC path is written raw (two backslashes, then one) or escaped (four, then two).
CODE_PRIVATE_PATH = re.compile(
    r"(?i:\b[A-Z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[^\\/\s\"<>]+)"
    r"|\b[D-Z]:[\\/]+[^\s\"<>]*"
    r"|\b[d-z]:(?:\\\\|/)[^\s\"<>]*"
    r"|(?i:\bC:[\\/]+(?:FreeSoft|LSS-Private)(?:[\\/]|\b))"
    r"|(?<!\\)\\\\(?!\\)[A-Za-z0-9][A-Za-z0-9._-]*\\(?!\\)[A-Za-z0-9$][^\s\\\"<>]*"
    r"|(?<!\\)\\{4}(?!\\)[A-Za-z0-9][A-Za-z0-9._-]*\\\\(?!\\)[A-Za-z0-9$][^\s\\\"<>]*")
PATTERNS = {
    "PRIVATE_KEY": re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
    "TOKEN": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{22,}"
        r"|sk-(?:ant-)?[A-Za-z0-9_-]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16}"
        r"|AIza[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{16,}"
        r"|glpat-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,})\b"),
    "COOKIE": re.compile(
        r"(?im)^\s*#\s*(?:Netscape HTTP Cookie File|HTTP Cookie File)"
        r"|^\s*(?:#HttpOnly_)?\S+\t(?:TRUE|FALSE)\t[^\r\n]*"
        r"|\b(?:Cookie|Set-Cookie)\s*:\s*[^\r\n]{8,}"
        r'|"(?:cookies|cookie_jar)"\s*:\s*[\[{]'),
    "ENV_SECRET": re.compile(
        r"(?m)^\s*(?:export\s+)?[A-Z_][A-Z0-9_]*\s*=\s*"
        r"(?:\"[^\"\r\n]{24,}\"|'[^'\r\n]{24,}'|[A-Za-z0-9_./+=:@%-]{24,})\s*(?:#.*)?$"
        r"|" + KEYWORD_SECRET),
    "PRIVATE_PATH": re.compile(
        r"(?i)\b[A-Z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[^\\/\s\"<>]+"
        r"|\b[D-Z]:[\\/]+[^\s\"<>]*"
        r"|\bC:[\\/]+(?:FreeSoft|LSS-Private)(?:[\\/]|\b)"
        r"|\\{2,}[A-Za-z0-9][A-Za-z0-9._-]*\\+[A-Za-z0-9$][^\s\\\"<>]*"),
    "CANARY": re.compile(r"LSS-CANARY-[A-Za-z0-9]{8,}", re.IGNORECASE),
}


# In the printable runs of binary data a UNC path needs a server and a share of two characters or more:
# compressed font tables hold runs such as a backslash before every character.
BINARY_PRIVATE_PATH = re.compile(
    r"(?i)\b[A-Z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[^\\/\s\"<>]+"
    r"|\b[D-Z]:[\\/]+[^\s\"<>]*"
    r"|\bC:[\\/]+(?:FreeSoft|LSS-Private)(?:[\\/]|\b)"
    r"|\\{2,}[A-Za-z0-9][A-Za-z0-9._-]+\\+[A-Za-z0-9$][^\s\\\"<>]+")
BINARY_RUN_LENGTH = 10
BINARY_RUN = re.compile(rb"[\x20-\x7e]{%d,}" % BINARY_RUN_LENGTH)
BINARY_RUN_16LE = re.compile(rb"(?:[\x20-\x7e]\x00){%d,}" % BINARY_RUN_LENGTH)
BINARY_RUN_16BE = re.compile(rb"(?:\x00[\x20-\x7e]){%d,}" % BINARY_RUN_LENGTH)


def _encodings(term: str) -> set[bytes]:
    found = set()
    for encoding in ("utf-8", "utf-16-le", "utf-16-be", "cp932", "euc_jp"):
        try:
            found.add(term.encode(encoding))
        except UnicodeError:
            pass
    return found


class GuardError(Exception):
    """An incomplete check; messages must never contain input values."""


def url_rewrites(repo: Path) -> bool:
    """Whether git may rewrite URLs (url.*.insteadOf / pushInsteadOf); then a query can reach another place.

    Unknown counts as yes.
    """
    env = os.environ.copy()
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    # GIT_CONFIG narrows `git config` to one file but does not change what ls-remote reads.
    env.pop("GIT_CONFIG", None)
    proc = subprocess.run(["git", "-C", str(repo), "config", "--get-regexp", r"^url\..*\.(insteadof|pushinsteadof)$"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    return proc.returncode != 1


def published_main(repo: Path, remote: str | None, url: str | None) -> str | None:
    """The main that the push destination itself reports now, as a local commit id, or None.

    Its history is already published at that destination. Local remote-tracking refs are not used: they can
    still point at another remote's history after the URL changes. When the destination cannot be asked, or
    reports a main this repository does not have, the caller scans the whole history.
    """
    if not remote or not url or url_rewrites(repo):
        return None
    try:
        lines = git(repo, "ls-remote", "--", url, "refs/heads/main", timeout=LS_REMOTE_TIMEOUT).decode().splitlines()
        fields = lines[0].split() if len(lines) == 1 else []
        if len(fields) != 2 or fields[1] != "refs/heads/main" or not OID_RE.fullmatch(fields[0]):
            return None
        oid, kind = resolve_object(repo, fields[0])
    except (GuardError, IndexError):
        return None
    return oid if kind == "commit" and oid == fields[0] else None


# Asking the destination for its main must not hold up a push: a stalled connection falls back to a full scan.
LS_REMOTE_TIMEOUT = 30


def git(repo: Path, *args: str, input_data: bytes | None = None, timeout: float | None = None) -> bytes:
    env = os.environ.copy()
    # Replacements can hide the actual objects sent by push.
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    if timeout is None:
        proc = subprocess.run(["git", "-C", str(repo), *args], input=input_data,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        if proc.returncode:
            raise GuardError("Git operation failed")
        return proc.stdout
    # A bounded call never prompts, and writes to a file rather than a pipe: a helper git starts (remote-https,
    # ssh) can outlive git and would keep a pipe open past the timeout. Its whole process tree is ended.
    env["GIT_TERMINAL_PROMPT"] = "0"
    with tempfile.TemporaryFile() as out:
        proc = subprocess.Popen(["git", "-C", str(repo), *args], stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.DEVNULL, env=env)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                try:
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                except (OSError, subprocess.SubprocessError):
                    pass   # git itself is still ended below; helpers the tree kill missed may remain
            try:
                proc.kill()
                proc.wait(timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
            raise GuardError("Git operation timed out")
        if code:
            raise GuardError("Git operation failed")
        out.seek(0)
        return out.read()


def valid_path(path: str) -> bool:
    parts = path.split("/")
    return bool(path) and all(p not in ("", ".", "..") for p in parts) and not (
        any(c in path for c in '\\:<>"|?*') or any(ord(c) < 32 or ord(c) == 127 for c in path)
        or any(p.endswith((" ", ".")) for p in parts)
        or any(re.fullmatch(r"(?i)(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", p)
               for p in parts))


def link_like(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & 0x400)


def require_plain_path(path: Path) -> None:
    # Check ancestors before resolving: resolve() alone conceals junctions.
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            if link_like(part):
                raise GuardError("Link or reparse point in filesystem path")


def load_denylist(path: Path) -> list[str]:
    require_plain_path(path)
    data = path.read_bytes()
    if not data or len(data) > MAX_POLICY_BYTES:
        raise GuardError("Missing or invalid private denylist")
    lines = data.decode("utf-8-sig").splitlines()
    literals = [line.strip().casefold() for line in lines
                if line.strip() and not line.lstrip().startswith("#")]
    if (not literals or any("\x00" in item for item in literals)
            or any(item.startswith(WORD_PREFIX) and not item[len(WORD_PREFIX):].strip()
                   for item in literals)):
        raise GuardError("Empty or invalid private denylist")
    return literals


def unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GuardError("Duplicate policy key")
        result[key] = value
    return result


@dataclass
class Policy:
    include: list[str]
    deny: list[str]
    reviewed_code_paths: set[str]
    approved_paths: dict[str, str]
    max_blob_bytes: int
    digest: str

    @classmethod
    def parse(cls, raw: bytes) -> Policy:
        if len(raw) > MAX_POLICY_BYTES:
            raise GuardError("Policy too large")
        obj = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique_json)
        allowed = {"schema_version", "include", "deny", "reviewed_code_paths",
                   "approved_paths", "max_blob_bytes", "note"}
        if (not isinstance(obj, dict) or set(obj) - allowed
                or obj.get("schema_version") != 1):
            raise GuardError("Invalid policy schema")
        for key in ("include", "deny", "reviewed_code_paths"):
            if not isinstance(obj.get(key), list) or not all(
                    isinstance(s, str) and s for s in obj[key]):
                raise GuardError("Invalid policy path list")
        reviewed = obj["reviewed_code_paths"]
        if len(set(reviewed)) != len(reviewed) or any(not valid_path(s) for s in reviewed):
            raise GuardError("Invalid reviewed paths")
        approvals = obj.get("approved_paths")
        if not isinstance(approvals, dict):
            raise GuardError("Invalid individual approvals")
        pinned = {}
        for path, approval in approvals.items():
            if (not valid_path(path) or not isinstance(approval, dict)
                    or set(approval) != {"blob_oid", "reason"}
                    or not isinstance(approval["blob_oid"], str)
                    or not OID_RE.fullmatch(approval["blob_oid"])
                    or not isinstance(approval["reason"], str) or not approval["reason"].strip()
                    or path in reviewed or path == POLICY_NAME):
                raise GuardError("Invalid pinned approval")
            pinned[path] = approval["blob_oid"]
        limit = obj.get("max_blob_bytes")
        if type(limit) is not int or not 1 <= limit <= 100 * 1024 * 1024:
            raise GuardError("Invalid size ceiling")
        return cls(obj["include"], obj["deny"], set(reviewed), pinned, limit,
                   hashlib.sha256(raw).hexdigest())

    def denied(self, path: str) -> bool:
        parts = path.casefold().split("/")
        # `models` names weight folders, but upstream source packages use it too: only an individually
        # blob-pinned Python file may pass that one name; every other hard directory stays absolute.
        pinned_source = path in self.approved_paths and parts[-1].endswith(".py")
        return (any((p in HARD_DIRS and not (p == "models" and pinned_source))
                    or p.endswith("-evidence") or p.startswith("frozen-")
                    for p in parts)
                or any(fnmatch.fnmatchcase(path.casefold(), pat.casefold()) for pat in self.deny))

    def admission(self, path: str, oid: str) -> tuple[bool, str]:
        if not valid_path(path):
            return False, "INVALID_PATH"
        if self.denied(path):
            return False, "DENIED_PATH"
        if path == POLICY_NAME:
            return True, "SELF_POLICY"
        if path in self.approved_paths:
            return (True, "PINNED") if self.approved_paths[path] == oid else (False, "OID_MISMATCH")
        if path in self.reviewed_code_paths and any(
                fnmatch.fnmatchcase(path, pat) for pat in self.include):
            return True, "REVIEWED_CODE"
        return False, "UNAPPROVED_PATH"


@dataclass
class Entry:
    path: str
    mode: str
    oid: str | None
    size: int
    disk: Path | None = None


def tree_entries(repo: Path, oid: str) -> list[Entry]:
    result = []
    for record in git(repo, "ls-tree", "-rlz", "--full-tree", oid).split(b"\0"):
        if not record:
            continue
        header, name = record.split(b"\t", 1)
        mode, kind, blob, size = header.decode("ascii").split()
        result.append(Entry(name.decode("utf-8"), mode if kind == "blob" else "160000",
                            blob, int(size) if size != "-" else 0))
    return result


def working_entries(repo: Path) -> list[Entry]:
    result = []
    def walk(directory: Path, prefix: str = ""):
        with os.scandir(directory) as entries:
            for item in sorted(entries, key=lambda e: e.name):
                if not prefix and item.name == ".git":
                    continue
                name = prefix + item.name
                path = Path(item.path)
                info = item.stat(follow_symlinks=False)
                if link_like(path):
                    result.append(Entry(name, "120000", None, 0, path))
                elif stat.S_ISDIR(info.st_mode):
                    # Nested repositories must not disappear into the working-tree walk.
                    if (path / ".git").exists():
                        result.append(Entry(name, "160000", None, 0, path))
                    else:
                        walk(path, name + "/")
                elif stat.S_ISREG(info.st_mode):
                    result.append(Entry(name, "100644", None, info.st_size, path))
                else:
                    result.append(Entry(name, "SPECIAL", None, 0, path))
    walk(repo)
    return result


class BlobReader:
    """One `git cat-file --batch` process; every reply is checked against the request."""

    def __init__(self, repo: Path):
        env = os.environ.copy()
        env["GIT_NO_REPLACE_OBJECTS"] = "1"
        self.proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env)

    def read(self, oid: str, size: int) -> bytes:
        if not OID_RE.fullmatch(oid):
            raise GuardError("Invalid object ID")
        try:
            self.proc.stdin.write(oid.encode("ascii") + b"\n")
            self.proc.stdin.flush()
            header = self.proc.stdout.readline().split()
            if header != [oid.encode("ascii"), b"blob", str(size).encode("ascii")]:
                raise GuardError("Unexpected batch reply")
            data = self.proc.stdout.read(size)
            if len(data) != size or self.proc.stdout.read(1) != b"\n":
                raise GuardError("Short batch read")
        except OSError:
            raise GuardError("Batch read failed") from None
        return data

    def close(self) -> None:
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def blob_oid(data: bytes, object_format: str) -> str:
    digest = hashlib.sha256() if object_format == "sha256" else hashlib.sha1()
    digest.update(b"blob %d\0" % len(data))
    digest.update(data)
    return digest.hexdigest()


def read_entry(repo: Path, entry: Entry, reader: BlobReader | None = None) -> bytes:
    if entry.disk is None:
        if reader is not None:
            return reader.read(entry.oid, entry.size)
        return git(repo, "cat-file", "blob", entry.oid)
    require_plain_path(entry.disk)
    with entry.disk.open("rb") as handle:
        data = handle.read(entry.size + 1)
    if len(data) != entry.size:
        raise GuardError("File changed during scan")
    return data


def magic_type(data: bytes) -> bool:
    signatures = (b"fLaC", b"OggS", b"ID3", b"MThd", b"\x1aE\xdf\xa3", b"FLV",
                  b"GGUF", b"ggml", b"lmgg", b"ggmf", b"ggjt", b"\x93NUMPY",
                  b"\x89HDF\r\n\x1a\n", b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08",
                  b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"7z\xbc\xaf\x27\x1c", b"Rar!",
                  b"MZ", b"\x7fELF", b"\xfe\xed\xfa", b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe",
                  b"\xd0\xcf\x11\xe0", b"SQLite format 3\x00", b"\x80\x02", b"\x80\x04",
                  b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"\x00\x00\x01\x00", b"%PDF-",
                  b"OTTO", b"wOFF", b"wOF2", b"\x00\x01\x00\x00")
    if data.startswith(signatures):
        return True
    if (data[:4] in (b"RIFF", b"RIFX", b"RF64", b"FORM")
            and data[8:12] in (b"WAVE", b"AVI ", b"WEBP", b"AIFF", b"AIFC")):
        return True
    if data[4:8] == b"ftyp" or data[:4] in (b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"):
        return True
    if len(data) >= 2 and data[0] == 255 and data[1] & 0xE0 == 0xE0:
        return True
    if data[257:262] == b"ustar":
        return True
    # Safetensors starts with an unsigned little-endian JSON-header length.
    if len(data) >= 10:
        length = int.from_bytes(data[:8], "little")
        if 2 <= length <= len(data) - 8 and data[8:9] == b"{":
            return True
    return False


def forbidden_type(path: str, data: bytes) -> bool:
    if PurePosixPath(path).suffix.casefold() in TYPE_EXTENSIONS or magic_type(data):
        return True
    try:
        value = data.decode("utf-8-sig")
    except UnicodeError:
        return True
    return any(ord(c) < 32 and c not in "\t\r\n\f" for c in value)


ESCAPED_CHAR = re.compile(r"\\u([0-9a-fA-F]{4})|\\x([0-9a-fA-F]{2})")


class Reporter:
    def __init__(self, literals: list[str]):
        self.literals = literals
        self.findings: list[dict] = []
        self._seen = set()

    def hit(self, value: str) -> bool:
        # "word:<term>" matches only between non-alphanumeric neighbours; others are substrings.
        text = value.casefold()
        for item in self.literals:
            if item.startswith(WORD_PREFIX):
                if re.search(r"(?<![0-9a-z])" + re.escape(item[len(WORD_PREFIX):].strip())
                             + r"(?![0-9a-z])", text):
                    return True
            elif item in text:
                return True
        return False

    def safe_path(self, path: str) -> str:
        if (self.hit(path)
                or any(rx.search(path) for rx in PATTERNS.values())
                or any(ord(c) < 32 or ord(c) == 127 for c in path)):
            return "<masked-path:" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:16] + ">"
        return path

    def add(self, path: str, rule: str, line: int = 0):
        key = (path, rule, line)
        if key not in self._seen:
            self._seen.add(key)
            self.findings.append({"path": self.safe_path(path), "rule": rule,
                                  "line": line, "match": "<masked>"})

    def text(self, path: str, value: str, path_value: bool = False, binary: bool = False):
        for number, line in enumerate(value.splitlines() or [value], 1):
            if self.hit(line):
                self.add(path, "DENYLIST", 0 if path_value else number)
        code = PurePosixPath(path).suffix.casefold() in CODE_SUFFIXES
        for rule, rx in PATTERNS.items():
            if rule == "ENV_SECRET" and code and not path_value:
                rx = CODE_ENV_SECRET
            if rule == "PRIVATE_PATH" and code and not path_value:
                rx = CODE_PRIVATE_PATH
            if rule == "PRIVATE_PATH" and binary:
                rx = BINARY_PRIVATE_PATH
            for match in rx.finditer(value):
                self.add(path, rule, 0 if path_value else value.count("\n", 0, match.start()) + 1)

    def content(self, path: str, data: bytes):
        # Even individually approved binary blobs retain content/denylist checks.
        binary = b"\x00" in data
        if not binary:
            values = {data.decode("utf-8-sig", errors="replace"), data.decode("latin-1")}
            # Escaped text is still text: a backslash-u or backslash-x escape in JSON or source is the
            # character once it is loaded, so the same checks also read the content with them resolved.
            plain = data.decode("utf-8-sig", errors="replace")
            if "\\u" in plain or "\\x" in plain:
                values.add(ESCAPED_CHAR.sub(lambda m: chr(int(m.group(1) or m.group(2), 16)), plain))
        else:
            # Binary data (fonts, images): compressed bytes read as text match short patterns by chance, so the
            # checks read what a person could read in it: runs of printable characters (single-byte and both
            # UTF-16 orders), every whole-file wide decoding that is valid, and each non-ASCII private literal
            # as bytes in the common encodings.
            values = {"\n".join(m.group().decode("ascii") for m in BINARY_RUN.finditer(data))}
            for rx, encoding in ((BINARY_RUN_16LE, "utf-16-le"), (BINARY_RUN_16BE, "utf-16-be")):
                values.add("\n".join(m.group().decode(encoding) for m in rx.finditer(data)))
            for encoding in ("utf-16", "utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"):
                try:
                    values.add(data.decode(encoding))
                except UnicodeError:
                    pass
            for item in self.literals:
                term = item[len(WORD_PREFIX):].strip() if item.startswith(WORD_PREFIX) else item
                if not term.isascii() and any(encoded in data for encoded in _encodings(term)):
                    self.add(path, "DENYLIST", 0)
        for value in sorted(values):
            self.text(path, value, binary=binary)


def resolve_object(repo: Path, name: str) -> tuple[str, str]:
    oid = git(repo, "rev-parse", "--verify", "--end-of-options", name + "^{}").decode().strip()
    if not OID_RE.fullmatch(oid):
        raise GuardError("Invalid object ID")
    kind = git(repo, "cat-file", "-t", oid).decode().strip()
    if kind not in ("commit", "tree"):
        raise GuardError("Expected a commit or tree")
    return oid, kind


def scan(repo: Path, tree_ish: str | None = None, staged: bool = False,
         allowlist: Path | None = None, denylist: Path | None = None) -> dict:
    reporter = Reporter([])
    reader = None
    object_format = ""
    result = {"result": "BLOCK", "mode": "staged" if staged else "tree" if tree_ish else "working",
              "object_oid": None, "commit_oid": None, "allowlist_sha256": None,
              "files_checked": 0, "counts": {}, "findings": []}
    try:
        reporter.literals = load_denylist(denylist or Path(os.environ.get(
            "LSS_PUBLISH_DENYLIST", DEFAULT_DENYLIST)))
        require_plain_path(repo)
        repo = repo.absolute()
        if staged:
            tree_ish = git(repo, "write-tree").decode().strip()
        kind = None
        if tree_ish:
            oid, kind = resolve_object(repo, tree_ish)
            result["object_oid"] = oid
            if kind == "commit":
                result["commit_oid"] = oid
            entries = tree_entries(repo, oid)
            reader = BlobReader(repo)
        else:
            entries = working_entries(repo)
            object_format = git(repo, "rev-parse", "--show-object-format").decode().strip()
            if object_format not in ("sha1", "sha256"):
                raise GuardError("Unknown object format")
        policy_entry = next((e for e in entries if e.path == POLICY_NAME), None)
        if allowlist is not None:
            require_plain_path(allowlist)
            raw = allowlist.read_bytes()
        elif (policy_entry is not None and policy_entry.mode in ("100644", "100755")
              and policy_entry.size <= MAX_POLICY_BYTES):
            raw = read_entry(repo, policy_entry, reader)
        else:
            raise GuardError("Missing or invalid tree policy")
        policy = Policy.parse(raw)
        result["allowlist_sha256"] = policy.digest
        # An external policy is also scanned; it cannot silently disable content checks.
        reporter.content(POLICY_NAME, raw)
        if kind == "commit":
            reporter.content("<commit-metadata>", git(repo, "cat-file", "commit", result["object_oid"]))
        case_paths = set()
        for entry in entries:
            path = entry.path
            result["files_checked"] += 1
            reporter.text(path, path, path_value=True)
            if path.casefold() in case_paths:
                reporter.add(path, "PATH_COLLISION")
            case_paths.add(path.casefold())
            if entry.mode not in ("100644", "100755"):
                reporter.add(path, "LINK_OR_SUBMODULE")
                continue
            if entry.size > policy.max_blob_bytes:
                reporter.add(path, "OVERSIZE")
                # Do not read an arbitrarily large blob just to report more findings.
                continue
            data = read_entry(repo, entry, reader)
            blob = entry.oid
            if blob is None:
                blob = blob_oid(data, object_format)
            allowed, layer = policy.admission(path, blob)
            if not allowed:
                reporter.add(path, layer)
            if forbidden_type(path, data) and layer != "PINNED":
                reporter.add(path, "BINARY_OR_MEDIA")
            reporter.content(path, data)
        result["result"] = "BLOCK" if reporter.findings else "PASS"
    except (GuardError, OSError, UnicodeError, ValueError, KeyError, TypeError, RecursionError):
        reporter.add("<scan>", "SCAN_ERROR")
        result["result"] = "ERROR"
    finally:
        if reader is not None:
            reader.close()
    result["findings"] = reporter.findings
    result["counts"] = dict(sorted(Counter(f["rule"] for f in reporter.findings).items()))
    return result


def emit(result: dict) -> int:
    print(f"public_guard: {result['result']}; files={result['files_checked']}; "
          f"findings={len(result['findings'])}; oid={result['object_oid'] or 'working-tree'}; "
          f"allowlist_sha256={result['allowlist_sha256'] or 'unavailable'}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["result"] == "PASS" else 2 if result["result"] == "ERROR" else 1


def pre_push(repo: Path, allowlist: Path | None, denylist: Path | None,
             remote: str | None = None, url: str | None = None) -> int:
    try:
        commits = set()
        updates = 0
        for line in sys.stdin:
            fields = line.split()
            if len(fields) != 4 or not OID_RE.fullmatch(fields[1]) or not OID_RE.fullmatch(fields[3]):
                raise GuardError("Invalid pre-push input")
            local_ref, local_oid, remote_ref, remote_oid = fields
            if set(local_oid) == {"0"}:
                raise GuardError("Ref deletion is not supported")
            if remote_ref != "refs/heads/main" and not re.fullmatch(
                    r"refs/heads/(?:codex|claude)/[A-Za-z0-9][A-Za-z0-9._/-]*", remote_ref):
                raise GuardError("Only main and agent branches are supported")
            oid, kind = resolve_object(repo, local_oid)
            if kind != "commit" or oid != local_oid:
                raise GuardError("Only commit tips are supported")
            args = ["rev-list", oid]
            if set(remote_oid) != {"0"}:
                # Unknown remote objects cannot be assumed safe or skipped.
                remote, remote_kind = resolve_object(repo, remote_oid)
                if remote_kind != "commit":
                    raise GuardError("Invalid remote tip")
                args.append("^" + remote)
            else:
                published = published_main(repo, remote, url)
                if published:
                    # A new branch: commits on the destination's current main are already published there;
                    # scanning them again protects nothing and grows with main's history. lss-gh still scans
                    # every ancestor before it pushes.
                    args.append("^" + published)
            # Always inspect the exact tip, even when rev-list is empty.
            commits.add(oid)
            commits.update(git(repo, *args).decode().splitlines())
            updates += 1
        if updates > 1:
            raise GuardError("Only one public ref update is supported")
        status = 0
        for oid in sorted(commits):
            status = max(status, emit(scan(repo, oid, allowlist=allowlist, denylist=denylist)))
        return status
    except (GuardError, OSError, UnicodeError, ValueError):
        return emit({"result": "ERROR", "files_checked": 0, "object_oid": None,
                     "commit_oid": None, "allowlist_sha256": None,
                     "counts": {"PUSH_INPUT_ERROR": 1},
                     "findings": [{"path": "<push>", "rule": "PUSH_INPUT_ERROR",
                                   "line": 0, "match": "<masked>"}]})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tree-ish")
    modes.add_argument("--staged", action="store_true")
    modes.add_argument("--pre-push", action="store_true")
    parser.add_argument("--allowlist", type=Path, help="Explicit trusted policy; default: policy in checked tree")
    parser.add_argument("--denylist", type=Path, help="Private local denylist; environment override is also supported")
    parser.add_argument("push_target", nargs="*", help="With --pre-push: the remote name and URL git passes to the hook")
    args = parser.parse_args(argv)
    if args.push_target and (not args.pre_push or len(args.push_target) != 2):
        parser.error("push_target is the hook's remote name and URL, with --pre-push only")
    if args.pre_push:
        return pre_push(args.repo, args.allowlist, args.denylist, *(args.push_target or (None, None)))
    return emit(scan(args.repo, args.tree_ish, args.staged, args.allowlist, args.denylist))


if __name__ == "__main__":
    sys.exit(main())
