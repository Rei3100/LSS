"""Fail-closed project privacy hook; standard library only.

The parent supervises an isolated worker, including dependency imports. The registered
shell command converts failure to start this file into a deny as well. This heuristic
prevents accidental access; arbitrary code and host-level hook bypass require the
permission profile described by the environment design (S2).
"""
from __future__ import annotations

import fnmatch
import json
import ntpath
import os
from pathlib import Path
import re
import subprocess
import sys

REASON = "Access denied: protected cookie, secret or credential location."
FAILURE = "Access denied: privacy guard failed or timed out."
WORKER_TIMEOUT = 5
MAX_INPUT_BYTES = 1024 * 1024
S = None


def load_scanner():
    global S
    if S is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import shell_scan
        S = shell_scan
    return S


def rules():
    data = json.loads(Path(__file__).with_name("RULES.json").read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Invalid rules version")
    for key in ("protected_roots", "protected_names", "protected_fragments"):
        if not isinstance(data.get(key), list) or not data[key] or not all(
                isinstance(item, str) and item for item in data[key]):
            raise ValueError("Invalid privacy rules")
    return data


def expand(raw):
    # Recognize Windows, Bash and PowerShell environment references without executing them.
    raw = re.sub(r"\$env:([A-Za-z_]\w*)|%([A-Za-z_]\w*)%|\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)",
                 lambda m: os.environ.get(next(g for g in m.groups() if g), m.group(0)), raw,
                 flags=re.I)
    return raw.replace("~", os.environ.get("USERPROFILE", "~"), 1) if raw.startswith("~") else raw


def normalize(raw):
    scanner = load_scanner()
    return scanner.norm_path(expand(raw), os.getcwd()).rstrip("\\")


def protected_roots(data):
    # Defaults cover a stock installation without storing machine-specific paths in the tree.
    drive = os.environ.get("SystemDrive", "C:")
    roots = [drive + "/LSS-Private/secrets", drive + "/FreeSoft/yt-dlpGUI/Cookie"]
    roots.extend(data["protected_roots"])
    return [normalize(p) for p in roots]


def is_cookie(raw):
    data = rules()
    path = normalize(raw)
    # Resolve links without opening a target file. A linked repo path must not evade the guard.
    resolved = normalize(str(Path(expand(raw)).resolve()))
    for p in (path, resolved):
        if any(p == r or p.startswith(r + "\\") for r in protected_roots(data)):
            return True
        if any(fnmatch.fnmatchcase(ntpath.basename(p), pattern.casefold())
               for pattern in data["protected_names"]):
            return True
        if any(fragment.casefold().replace("/", "\\") in p
               for fragment in data["protected_fragments"]):
            return True
    return False


def mentions_protected(text):
    # Tokens handle unquoted paths; the whole text also covers paths embedded in code strings.
    if is_cookie(text):
        return True
    return any(is_cookie(token) for token in re.split(r"[\s\"'();=,]+", text)
               if token and any(c in token for c in "/\\.%$"))


CONTENT_VERBS = {"cat", "type", "get-content", "gc", "head", "tail", "more", "less", "select-string", "sls", "grep",
                 "egrep", "fgrep", "rg", "findstr", "sed", "awk", "gawk", "xxd", "od", "hexdump", "strings",
                 "format-hex", "base64", "certutil", "cut", "sort", "tac", "nl", "bat", "wc", "diff", "cmp",
                 "import-csv", "convertfrom-csv", "get-filehash", "ConvertFrom-String"}
CONTENT_VERBS = {v.lower() for v in CONTENT_VERBS} - {"wc", "get-filehash"}  # counts and hashes expose no values
PIPE_READERS = CONTENT_VERBS | {"xargs", "invoke-expression", "iex"}
READ_IN_CODE = re.compile(r"open\(|read_text|read_bytes|\.read\(|MozillaCookieJar|cookiejar|ReadAll(?:Text|Lines|Bytes)"
                          r"|OpenRead|StreamReader|Get-Content|\bcat\b|\btype\b", re.I)
VAR_ASSIGN = re.compile(r"(?<![\w$.\-])\$?(?:env:)?([A-Za-z_]\w*)\s*=\s*([^\n;]+)")

def cookie_vars(text: str) -> set[str]:
    return {m.group(1) for m in VAR_ASSIGN.finditer(text) if mentions_protected(m.group(2))}


def refs_cookie(token: str, names: set[str]) -> bool:
    if is_cookie(token):
        return True
    return any(re.search(r"\$\{?(?:env:)?%s\b|%%%s%%" % (re.escape(n), re.escape(n)), token, re.I) for n in names)


def command_reads_cookie(command: str, depth: int = 0) -> bool:
    if depth > 8:
        raise ValueError("Shell nesting limit exceeded")
    names = cookie_vars(command)
    if not mentions_protected(command) and not names:
        return False
    scan = S.tokenize(command)
    chain_has_cookie = False
    python_seen = False
    for seg in scan.segs:
        if seg.sep not in ("|", "{", "}"):
            chain_has_cookie = False
        parsed = S.segment_command(seg)
        if parsed is None:
            continue
        _, base, args = parsed
        cookie_arg = any(refs_cookie(a, names) for a in args) or any(refs_cookie(a, names) for a in seg.redir_in)
        if base in CONTENT_VERBS and (cookie_arg or chain_has_cookie):
            return True
        if base in PIPE_READERS and chain_has_cookie:
            return True
        if cookie_arg:
            chain_has_cookie = True
        payload = S.shell_payload(base, args)
        if payload and command_reads_cookie(payload, depth + 1):
            return True
        if re.match(r"^(?:python[\d.]*|pythonw|py)$", base):
            python_seen = True
            for k, a in enumerate(args):
                if a == "-c" and k + 1 < len(args) and mentions_protected(args[k + 1]) \
                        and READ_IN_CODE.search(args[k + 1]):
                    return True
    if python_seen and any(mentions_protected(b) and READ_IN_CODE.search(b) for b in scan.blobs):
        return True
    if re.search(r"ReadAll(?:Text|Lines|Bytes)|OpenRead|StreamReader", command) and mentions_protected(command):
        return True
    return any(command_reads_cookie(inner, depth + 1) for inner in scan.inners)


def grep_reaches_cookie(path):
    if is_cookie(path):
        return True
    p = normalize(path)
    return any(root.startswith(p + "\\") for root in protected_roots(rules()))


def adapt_event(event):
    if not isinstance(event, dict):
        raise ValueError("Expected hook object")
    name = event.get("tool_name")
    args = event.get("tool_input")
    # Codex shell tools use cmd (string or argv); Claude uses command.
    if name in ("exec_command", "shell_command", "shell", "functions.exec_command", "functions.shell_command"):
        if not isinstance(args, dict):
            raise ValueError("Expected tool input object")
        command = args.get("command", args.get("cmd"))
        if isinstance(command, list) and all(isinstance(s, str) for s in command):
            command = " ".join(command)
        event = {**event, "tool_name": "Bash", "tool_input": {**args, "command": command}}
    if not isinstance(event.get("tool_name"), str) or not isinstance(event.get("tool_input"), dict):
        raise ValueError("Missing tool name or input")
    return event


def blocked(event):
    load_scanner()
    rules()  # Missing or invalid policy is a failure, even for an otherwise harmless operation.
    event = adapt_event(event)
    name, args = event["tool_name"], event["tool_input"]
    if name in ("Read", "Edit", "MultiEdit", "Write", "Grep", "Glob"):
        key = "path" if name in ("Grep", "Glob") else "file_path"
        path = args.get(key, os.getcwd() if name in ("Grep", "Glob") else None)
        if not isinstance(path, str) or not path:
            raise ValueError("Missing target path")
        if name == "Glob" and is_cookie(args.get("pattern", "")):
            return True
        return grep_reaches_cookie(path) if name in ("Grep", "Glob") else is_cookie(path)
    if name in ("Bash", "PowerShell"):
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Missing shell command")
        # Existing cookie acquisition/listing workflows pass paths without reading values.
        # For other protected locations any direct reference is denied conservatively.
        if any(is_cookie(token) and "yt-dlpgui" not in token.casefold()
               for token in re.split(r"[\s\"'();=,]+", command)
               if token and any(c in token for c in "/\\.%$")
               and not token.startswith("$env:LSS_COOKIES_FILE_")):
            return True
        return command_reads_cookie(command)
    return False


def deny(reason=FAILURE):
    # Never include event text, paths or exception details (they can contain secrets).
    print(json.dumps({"decision": "block", "reason": reason, "hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason}}))
    print(reason, file=sys.stderr)
    return 2


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            return deny()
        if sys.argv[1:] == ["--worker"]:
            event = json.loads(raw.decode("utf-8"))
            return deny(REASON) if blocked(event) else 0
        if sys.argv[1:]:
            return deny()
        result = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker"],
                                input=raw, capture_output=True, timeout=WORKER_TIMEOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode == 2 and result.stderr.strip() == REASON.encode():
            return deny(REASON)
        if result.returncode != 0 or result.stdout or result.stderr:
            return deny()
        return 0
    except BaseException:
        return deny()


if __name__ == "__main__":
    sys.exit(main())
