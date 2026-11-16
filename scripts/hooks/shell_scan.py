"""Shared helpers for the PreToolUse guards (library, not a hook): a quote-aware shell splitter, path normalisation and
the transcript's genuine user prompts.

The splitter understands what the Bash and PowerShell tools receive: `;` `&&` `||` `|` `&` newline and braces as
separators outside quotes, redirections, bash heredocs, PowerShell here-strings, `$(...)` sub-commands, the `&` call
operator and leading `VAR=value` assignments. It is a heuristic for guarding agent mistakes, not a security sandbox:
a command assembled from variables the guard cannot see (a path built at run time, a script file's contents) is
outside its reach. Standard library only.
"""
from __future__ import annotations

import base64
import json
import ntpath
import re
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Seg:
    sep: str = ""
    tokens: list[str] = field(default_factory=list)
    redir_out: list[str] = field(default_factory=list)
    redir_in: list[str] = field(default_factory=list)


@dataclass
class Scan:
    segs: list[Seg] = field(default_factory=list)
    blobs: list[str] = field(default_factory=list)   # heredoc bodies and here-strings (code carried as data)
    inners: list[str] = field(default_factory=list)  # $(...) sub-commands


def _balanced(text: str, start: int) -> tuple[str, int]:
    """Return the text inside the parenthesis opened just before `start` and the index after its close."""
    depth, j = 1, start
    while j < len(text) and depth:
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
        j += 1
    return (text[start:j - 1] if depth == 0 else text[start:]), j


def _read_heredocs(text: str, i: int, pending: list[str], blobs: list[str]) -> int:
    for delim in pending:
        lines: list[str] = []
        while i < len(text):
            j = text.find("\n", i)
            line = text[i:] if j < 0 else text[i:j]
            i = len(text) if j < 0 else j + 1
            if line.strip() == delim:
                break
            lines.append(line)
        blobs.append("\n".join(lines))
    return i


def tokenize(text: str) -> Scan:
    scan = Scan()
    seg = Seg()
    tok: list[str] = []
    has_tok = False
    redir: str | None = None
    pending: list[str] = []
    braces: list[bool] = []
    i, n = 0, len(text)

    def flush() -> None:
        nonlocal tok, has_tok, redir
        if has_tok:
            word = "".join(tok)
            if redir == "out":
                seg.redir_out.append(word)
            elif redir == "in":
                seg.redir_in.append(word)
            else:
                seg.tokens.append(word)
            redir = None
        tok, has_tok = [], False

    def end_seg(sep: str) -> None:
        nonlocal seg
        flush()
        if seg.tokens or seg.redir_out or seg.redir_in:
            scan.segs.append(seg)
        seg = Seg(sep)

    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c in " \t\r":
            flush()
            i += 1
        elif c == "\n":
            end_seg("\n")
            i += 1
            if pending:
                i = _read_heredocs(text, i, pending, scan.blobs)
                pending = []
        elif c == "#" and not has_tok:
            while i < n and text[i] != "\n":
                i += 1
        elif c == "`":
            if nxt == "\n":
                i += 2
            elif nxt in ('"', "'", "$", "`"):
                tok.append(nxt)
                has_tok = True
                i += 2
            else:
                tok.append(c)
                has_tok = True
                i += 1
        elif c == "\\":
            if nxt == "\n":
                i += 2
            elif nxt in ('"', "'") and not (has_tok and tok and tok[-1].endswith(":")):
                tok.append(nxt)
                has_tok = True
                i += 2
            else:
                tok.append(c)
                has_tok = True
                i += 1
        elif c in "\"'":
            has_tok = True
            i += 1
            if c == "'":
                while i < n:
                    if text[i] == "'":
                        if i + 1 < n and text[i + 1] == "'":
                            tok.append("'")
                            i += 2
                            continue
                        break
                    tok.append(text[i])
                    i += 1
            else:
                while i < n and text[i] != '"':
                    if text[i] == "\\" and i + 1 < n and text[i + 1] == '"':
                        tok.append('"')
                        i += 2
                        continue
                    if text[i] == "`" and i + 1 < n:
                        tok.append(text[i + 1])
                        i += 2
                        continue
                    if text[i] == "$" and i + 1 < n and text[i + 1] == "(":
                        inner, _ = _balanced(text, i + 2)
                        scan.inners.append(inner)
                    tok.append(text[i])
                    i += 1
            i += 1
        elif c == ";":
            end_seg(";")
            i += 1
        elif c == "&":
            if nxt == "&":
                end_seg("&&")
                i += 2
            elif nxt == ">":
                flush()
                redir = "out"
                i += 3 if text[i + 2:i + 3] == ">" else 2
            elif not has_tok and not seg.tokens and nxt in " \t\"'$.\\":
                tok.append("&")
                has_tok = True
                i += 1
            else:
                end_seg("&")
                i += 1
        elif c == "|":
            if nxt == "|":
                end_seg("||")
                i += 2
            else:
                end_seg("|")
                i += 1
        elif c in "{}":
            prev = text[i - 1] if i else ""
            if c == "{" and prev in "$@" and has_tok:
                braces.append(True)
                tok.append(c)
                i += 1
            elif c == "}" and braces:
                braces.pop()
                tok.append(c)
                i += 1
            else:
                end_seg(c)
                i += 1
        elif c in "<>":
            word = "".join(tok)
            if has_tok and (word.isdigit() or word in ("*", "&")):
                tok, has_tok = [], False
            else:
                flush()
            if c == ">":
                j = i + 1
                if text[j:j + 1] == ">":
                    j += 1
                if text[j:j + 1] == "&":
                    j += 1
                redir = "out"
                i = j
            elif text.startswith("<<<", i):
                i += 3
            elif text.startswith("<<", i):
                j = i + 2
                if text[j:j + 1] == "-":
                    j += 1
                while j < n and text[j] in " \t":
                    j += 1
                if j < n and text[j] in "\"'":
                    q = text[j]
                    k = text.find(q, j + 1)
                    k = n if k < 0 else k
                    delim, j = text[j + 1:k], k + 1
                else:
                    k = j
                    while k < n and text[k] not in " \t\r\n;|&<>)":
                        k += 1
                    delim, j = text[j:k], k
                if delim:
                    pending.append(delim)
                i = j
            else:
                redir = "in"
                i += 1
        elif c == "@" and not has_tok and nxt in ("'", '"') and text[i + 2:].split("\n", 1)[0].strip() == "":
            body_start = text.find("\n", i + 2)
            end = text.find("\n" + nxt + "@", body_start) if body_start >= 0 else -1
            if body_start < 0 or end < 0:
                tok.append(c)
                has_tok = True
                i += 1
            else:
                scan.blobs.append(text[body_start + 1:end])
                tok.append("<here-string>")
                has_tok = True
                i = end + 3
        else:
            if c == "$" and nxt == "(":
                inner, _ = _balanced(text, i + 2)
                scan.inners.append(inner)
            tok.append(c)
            has_tok = True
            i += 1
    end_seg("")
    return scan


WRAPPERS = {"&", ".", "call", "sudo", "env", "nohup", "time", "command", "exec", "builtin", "then", "else", "do",
            "elif", "if", "while", "until", "!", "not"}
ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def segment_command(seg: Seg) -> tuple[str, str, list[str]] | None:
    """Return (verb as typed, lower-case base name without extension, args) or None for non-commands."""
    tokens = list(seg.tokens)
    while tokens:
        head = tokens[0]
        stripped = head.lstrip("(@$!") if head.startswith(("(", "$(", "@(")) else head
        if stripped != head:
            if not stripped:
                tokens.pop(0)
                continue
            tokens[0] = head = stripped
        if head.lower() in WRAPPERS or ENV_ASSIGN.match(head):
            tokens.pop(0)
            continue
        break
    if not tokens:
        return None
    head = tokens[0]
    if head.startswith("$") and (len(tokens) < 2 or tokens[1] == "=" or "=" in head):
        return None  # variable assignment
    if len(tokens) > 1 and tokens[1] in ("=", "+=", ":=", "-=") and re.match(r"^\w+$", head):
        return None  # `name = value` (python/PowerShell assignment), not a command
    base = ntpath.basename(head.replace("/", "\\")).lower()
    for ext in (".exe", ".cmd", ".bat", ".com"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return head, base, tokens[1:]


def shell_payload(base: str, args: list[str]) -> str | None:
    """The command string a nested shell invocation carries (`powershell -Command ...`, `bash -c ...`, `cmd /c ...`)."""
    low = [a.lower() for a in args]
    if base in ("powershell", "pwsh"):
        for k, a in enumerate(low):
            if a in ("-encodedcommand", "-enc", "-ec", "-e") and k + 1 < len(args):
                try:
                    return base64.b64decode(args[k + 1]).decode("utf-16le", "replace")
                except Exception:  # noqa: BLE001
                    return None
            if a.startswith("-c") and "-command".startswith(a.split(":")[0]) and k + 1 <= len(args):
                return " ".join(args[k + 1:]) if ":" not in a else a.split(":", 1)[1] + " " + " ".join(args[k + 1:])
        return None
    if base in ("bash", "sh", "zsh", "dash", "ksh"):
        for k, a in enumerate(args):
            if re.fullmatch(r"-[a-z]*c[a-z]*", a) and k + 1 < len(args):
                return args[k + 1]
        return None
    if base == "cmd":
        for k, a in enumerate(low):
            if a in ("/c", "/k", "/r"):
                return " ".join(args[k + 1:])
        return None
    if base in ("invoke-expression", "iex"):
        return " ".join(args)
    return None


def norm_path(raw: str, cwd: str | None = None) -> str:
    """Lower-case Windows form of a path token: forward slashes, /e/x and /mnt/e/x -> drive-letter form, relative paths joined to
    cwd, `..` collapsed (unless the token still holds a variable)."""
    p = raw.strip().strip("\"'").replace("/", "\\")
    m = re.match(r"^\\(?:mnt\\)?([a-zA-Z])(?=\\|$)", p)
    if m:
        p = m.group(1) + ":" + p[m.end():]
    dynamic = bool(re.search(r"[$%]", p))
    absolute = bool(re.match(r"^(?:[a-zA-Z]:|\\|~)", p))
    if cwd and not absolute and not dynamic and p not in ("", "-"):
        p = cwd.replace("/", "\\") + "\\" + p
    if not dynamic and p:
        p = ntpath.normpath(p)
    return p.lower()


def genuine_user_texts(transcript_path: str) -> Iterator[str]:
    """Text of every genuine user prompt in the main thread: not tool results, not meta, not sidechain, with the
    harness-injected blocks (agent messages, system reminders, task notifications) removed."""
    strip = re.compile(r"<(agent-message|system-reminder|task-notification|local-command-stdout|local-command-stderr|"
                       r"local-command-caveat|command-name|command-message|command-args)\b[^>]*>.*?</\1>", re.S)
    with open(transcript_path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if '"user"' not in line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not isinstance(entry, dict) or entry.get("type") != "user" or entry.get("isMeta") \
                    or entry.get("isSidechain"):
                continue
            message = entry.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                texts = [content]
            elif isinstance(content, list):
                if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                    continue
                texts = [b.get("text") or "" for b in content if isinstance(b, dict) and b.get("type") == "text"]
            else:
                continue
            for text in texts:
                cleaned = strip.sub(" ", text)
                if cleaned.strip():
                    yield cleaned


def deny_output(reason: str) -> str:
    return json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                              "permissionDecisionReason": reason}})
