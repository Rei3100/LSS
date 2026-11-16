"""Agent usage report: where Claude Code and Codex effort and tokens went over a time window.

    python scripts/agent_usage.py [--days 14] [--since YYYY-MM-DD] [--json OUT] [--top 30] [--cutoff ISO]

The window ends now and starts `--days` earlier (or at `--since`, a JST date). Eight sections are computed:
tokens by agent / role / model with an input-equivalent cost line, tool mix, repeated file reads, test runs and
their failure rate, hook blocks, branch overlap between sessions, session start latency and SessionStart hook
output, and idle gaps longer than one hour. The default output is a compact text report; `--json OUT` also writes
the full result (every table, untruncated to `--top`) to OUT.

Idle gaps list, per gap, the start (JST), the session id prefix, the gap length and the cache-write tokens of
the first assistant message after it. The text report splits gap count and post-gap cache-write tokens before vs
after the keep-alive cutoff (`--cutoff`, default env AGENT_USAGE_CUTOFF, else the commit that added
scripts/keepalive.py), so one can see whether >1 h gaps still happen once the keep-alive mechanism
(scripts/keepalive.py) exists. A gap counts as "after" when it starts at or after the cutoff.

Cost-equivalent units are input-equivalent tokens: uncached input 1.0, cache read 0.1, 5-minute cache write 1.25
(Claude subagents and every Codex role), 1-hour cache write 2.0 (Claude main conversations), output 5.0. They
compare effort between components; they are not prices.

Sources (read-only; transcripts are never modified and chat text is never printed):
  Claude  %USERPROFILE%\\.claude\\projects\\<dir whose name contains "LSS">\\**\\*.jsonl  (subagents/ = subagent)
  Codex   %USERPROFILE%\\.codex\\sessions\\**\\*.jsonl  rollouts (token_usage_record events, deduplicated by
          response id; session_meta gives root / subagent / guardian)

Caveats:
  - Test failure detection is text-based (tool-error flag plus failure phrases in the output); a passing run
    that prints "FAILED" in its output counts as failed, a failure that prints nothing recognisable does not.
  - Wall times are tool-call start to tool-result, so they include approval waits and queueing, not only runtime.
  - Repeated-read sizes use the file's current size and fold worktree paths onto repo-relative paths; a partial
    Read(limit) is approximated as limit * 70 bytes; shell reads are found by regex.
  - Claude assistant usage is the maximum per message id (streamed duplicates); Codex counts uncached input as
    input_tokens - cached_input_tokens.
  - Commands shown in the report are normalised (paths, long strings, numbers and token-like strings masked).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JST = timezone(timedelta(hours=9))
GAP_S = 3600
COST = dict(input=1.0, cache_read=0.1, write_5m=1.25, write_1h=2.0, output=5.0)


def home_dir() -> str:
    return os.environ.get("USERPROFILE") or os.environ.get("HOME") or os.path.expanduser("~")


def parse_ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def jst_day(dt):
    return dt.astimezone(JST).strftime("%Y-%m-%d")


def jst_stamp(dt):
    return dt.astimezone(JST).strftime("%Y-%m-%d %H:%M")


def new_tok():
    return dict(input=0, cache_write=0, cache_read=0, output=0, reasoning=0, msgs=0)


def add_tok(tgt, i, cw, cr, o, rs=0):
    tgt["input"] += i
    tgt["cache_write"] += cw
    tgt["cache_read"] += cr
    tgt["output"] += o
    tgt["reasoning"] += rs
    tgt["msgs"] += 1


def pct(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q * len(vals)))]


# ---------- command / path normalisation ----------
def norm_path(p):
    """Repo-relative form of a path, folding worktrees and the main checkout; other paths pass through."""
    if not p:
        return None
    p = p.strip().strip("\"'`").replace("\\", "/")
    p = re.sub(r"^/?([A-Za-z]):/", lambda m: m.group(1).upper() + ":/", p)
    p = re.sub(r"^/([a-z])/", lambda m: m.group(1).upper() + ":/", p)
    m = re.match(r"(?i)^.*?/(?:\.worktrees|\.claude/worktrees)/[^/]+/(.*)$", p)
    if m:
        return m.group(1)
    root = str(REPO).replace("\\", "/").rstrip("/") + "/"
    if p.lower().startswith(root.lower()):
        return p[len(root):]
    return p


PARTIAL_RE = re.compile(r"-First|-Skip|-Tail|-TotalCount|\bhead\b|\btail\b|sed\s+-n|Select-String|\bgrep\b|\brg\b", re.I)
POLL_RE = re.compile(r"Start-Sleep|\bsleep\b|AddMinutes|Get-Content|\buntil\b|\bgrep\b|\bsed\b|\brg\b|Select-String", re.I)
READ_CMD = re.compile(r"(?:^|[;&|\s(])(?:cat|head|tail|bat|type|Get-Content|gc|sed\s+-n\s+\S+|sed|nl|less)\s+((?:-[A-Za-z]+\s+(?:\d+\s+)?)*)([^\s|;&<>]+)", re.I)
PATH_OK = re.compile(r"\.(py|md|json|jsonl|ps1|txt|toml|yml|yaml|rs|ts|js|cfg|ini|csv|log|html|sh|lock)$", re.I)
TEST_RE = re.compile(r"scripts[\\/]+dev\.py[\'\"]?\s+test|\bpytest\b|dev\.py test|-m unittest", re.I)
FAIL_RE = re.compile(r"(\d+ failed|\bFAILED\b|Process exited with code [1-9]|exit code[:= ]+[1-9]|Exit code [1-9]|\bERROR\b.*collect)", re.I)
HOOK_RE = re.compile(r"PreToolUse|PostToolUse|hook error|hook block|blocked by (?:a )?hook|Stop hook|stop_gate|budget \d|keepalive|hook denied|hooks?/RULES|guard\.py", re.I)
SECRET_KV = re.compile(r"(?i)\b([\w-]*(?:token|secret|passw(?:or)?d|api[_-]?key|auth|credential)[\w-]*)(\s*[=:]\s*|\s+)(\S+)")
LONG_TOKEN = re.compile(r"(?<![\w/\\.])[A-Za-z0-9_\-]{24,}(?![\w/\\.])")


def reads_in_cmd(cmd):
    out = []
    for m in READ_CMD.finditer(cmd):
        p = m.group(2).strip("\"'`,")
        if PATH_OK.search(p):
            out.append(p)
    for m in re.finditer(r"(?:Get-Content|gc)\s+([^|;\n]+)", cmd, re.I):
        for part in m.group(1).split(","):
            p = re.sub(r"\s+-\w+.*$", "", part.strip().strip("\"'`"))
            if PATH_OK.search(p):
                out.append(p)
    return out


def norm_cmd(cmd):
    c = re.sub(r"\s+", " ", cmd.strip())
    c = re.sub(r"^cd\s+(\"[^\"]*\"|\S+)\s*(&&|;)\s*", "", c)
    c = SECRET_KV.sub(lambda m: m.group(1) + m.group(2) + "<X>", c)
    c = re.sub(r"(?i)[A-Z]:[\\/][^\s\"']*", "<P>", c)
    c = re.sub(r"(?<![\w.])/[a-z]/[^\s\"']*", "<P>", c)
    c = re.sub(r"\"[^\"]{12,}\"", '"<S>"', c)
    c = re.sub(r"'[^']{12,}'", "'<S>'", c)
    c = re.sub(r"\b[0-9a-f]{7,40}\b", "<H>", c)
    c = LONG_TOKEN.sub("<X>", c)
    c = re.sub(r"\b\d{2,}\b", "<N>", c)
    return c[:110]


def first_prog(cmd):
    c = cmd.strip()
    c = re.sub(r"^cd\s+(\"[^\"]*\"|\S+)\s*(&&|;)\s*", "", c)
    c = re.sub(r"^(\$env:\w+\s*=\s*\S+\s*;\s*)+", "", c)
    c = re.sub(r"^(\w+=\S+\s+)+", "", c)
    c = re.sub(r"^powershell(\.exe)?\s+(-\w+\s+(\w+\s+)?)*(-Command\s+)?", "powershell ", c, flags=re.I)
    c = c.lstrip("&").strip()
    m = re.match(r"(\"[^\"]+\"|'[^']+'|\S+)(?:\s+(\S+))?(?:\s+(\S+))?", c)
    if not m:
        return "(empty)"
    base = m.group(1).strip("\"'").replace("\\", "/").split("/")[-1].lower()
    rest = " ".join(x for x in (m.group(2), m.group(3)) if x)
    if base in ("python.exe", "python", "python3", "py", "python3.exe"):
        args = re.sub(r"^-I\s+", "", rest.replace("\\", "/"))
        mm = re.match(r"(\S*?scripts/(?:[\w/]+)\.py)\s*(\w+)?", args)
        sc = mm.group(1).split("/")[-1] if mm else args.split(" ")[0].split("/")[-1]
        if sc == "dev.py" and mm:
            return "python scripts/dev.py " + (mm.group(2) or "")
        if "pytest" in c or args.startswith("-m pytest"):
            return "python -m pytest"
        return "python " + sc[:40]
    if base in ("pytest", "pytest.exe"):
        return "pytest"
    if base.startswith("powershell") or base == "pwsh":
        mm = re.search(r"-File\s+(\"[^\"]+\"|\S+)", c)
        if mm:
            return "powershell -File " + mm.group(1).strip('"').replace("\\", "/").split("/")[-1]
        return "powershell (inline)"
    if base == "git":
        return "git " + (m.group(2) or "")
    if base in ("ls", "dir"):
        return "ls"
    return base[:30] if base else "(empty)"


def hook_sig(text):
    t = text[:1500]
    names = re.findall(r"([\w\-]+(?:_guard|_gate|gate|guard|check|hook)[\w\-]*)\.(?:py|ps1)", t)
    pre = re.findall(r"(PreToolUse:?[\w]*|PostToolUse:?[\w]*|Stop hook|stop_gate|budget \d+|keepalive[\w\-]*)", t, re.I)
    return (names[0] if names else (pre[0] if pre else "other")).lower()[:50]


# ---------- collection ----------
class Usage:
    def __init__(self, start, end):
        self.start, self.end = start, end
        self.notes = []
        self.agents = {}
        for nm in ("claude", "codex"):
            self.agents[nm] = dict(
                tok_total=new_tok(), tok_by_model=collections.defaultdict(new_tok),
                tok_by_day=collections.defaultdict(new_tok), tok_by_role=collections.defaultdict(new_tok),
                sessions=set(), sub_files=set(), sess_by_day=collections.defaultdict(set),
                tools=collections.Counter(), progs=collections.Counter(), cmds=collections.Counter())
        self.reads = collections.defaultdict(lambda: dict(count=0, sessions=set(), partial=0, partial_bytes=0))
        self.tests = dict(claude=[], codex=[])
        self.hooks = collections.Counter()
        self.hooks_agent = collections.defaultdict(collections.Counter)
        self.startup = []
        self.sessions = []
        self.gaps = []
        self.longcmd = dict(claude=[0, 0.0], codex=[0, 0.0])
        self.sleep = dict(codex_sleep_calls=0, codex_wait_calls=0)
        self._size = {}

    def inwin(self, t):
        return self.start <= t <= self.end

    def file_size(self, rel):
        if rel not in self._size:
            sz = None
            for base in (str(REPO) + os.sep, ""):
                try:
                    if os.path.isfile(base + rel):
                        sz = os.path.getsize(base + rel)
                        break
                except OSError:
                    pass
            self._size[rel] = sz
        return self._size[rel]

    def add_read(self, path, sid, partial_bytes=None):
        rel = norm_path(path)
        if not rel:
            return
        r = self.reads[rel]
        r["count"] += 1
        r["sessions"].add(sid)
        if partial_bytes is not None:
            r["partial"] += 1
            r["partial_bytes"] += partial_bytes

    def add_gap(self, agent, sid, prev, t, gap_s, cw, cr, inp):
        self.gaps.append(dict(agent=agent, sid=sid[:8], start=prev, resume=t, gap_s=gap_s,
                              cache_write=cw, cache_read=cr, input=inp))

    # ---- Claude ----
    def process_claude(self):
        base = os.path.join(home_dir(), ".claude", "projects")
        dirs = [d for d in glob.glob(os.path.join(base, "*")) if "LSS" in os.path.basename(d)]
        files = [f for d in dirs for f in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)
                 if os.path.getmtime(f) >= self.start.timestamp()]
        self.notes.append(f"claude project dirs: {len(dirs)}; jsonl files modified in window: {len(files)}")
        C = self.agents["claude"]
        for f in files:
            parts = os.path.relpath(f, base).replace("\\", "/").split("/")
            is_sub = "subagents" in parts
            sid = parts[1] if is_sub else os.path.basename(f)[:-6]
            events = []
            with open(f, encoding="utf8", errors="replace") as fh:
                for line in fh:
                    if line.startswith('{"type":"api-request') or line.startswith('{"type":"deferred'):
                        continue
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(d, dict) and d.get("timestamp"):
                        t = parse_ts(d["timestamp"])
                        if t:
                            events.append((t, d))
            if not events:
                continue
            msgs, tool_uses, times = {}, {}, []
            first_ts = first_asst_ts = first_u = None
            branches = collections.Counter()
            for t, d in events:
                ty = d.get("type")
                if first_ts is None:
                    first_ts = t
                if d.get("gitBranch"):
                    branches[d["gitBranch"]] += 1
                inwin = self.inwin(t)
                att = d.get("attachment")
                if not isinstance(att, dict):
                    att = None
                if att and att.get("hookEvent") == "SessionStart" and not is_sub:
                    self.startup.append(dict(agent="claude", sid=sid, kind=att.get("type"),
                                             stdout_len=len(att.get("stdout") or ""), content_len=len(att.get("content") or ""),
                                             dur_ms=att.get("durationMs"), cmd=os.path.basename((att.get("command") or "").replace("\\", "/"))[-60:]))
                if att and att.get("type") == "hook_blocking_error" and inwin:
                    be = att.get("blockingError") or {}
                    cmd = be.get("command", "") if isinstance(be, dict) else ""
                    key = os.path.basename(cmd.strip('"').replace("\\", "/")).strip('"') or att.get("hookName", "?")
                    self.hooks[f"{att.get('hookName')}:{key}"] += 1
                    self.hooks_agent["claude"][key] += 1
                if ty == "system" and d.get("subtype") == "stop_hook_summary" and inwin:
                    self.hooks["stop_hook_summary:total"] += 1
                    if d.get("preventedContinuation"):
                        self.hooks["stop_hook_summary:preventedContinuation"] += 1
                    if d.get("hookErrors"):
                        self.hooks["stop_hook_summary:hookErrors"] += 1
                if ty == "assistant":
                    m = d.get("message") or {}
                    mid = m.get("id") or d.get("uuid")
                    u = m.get("usage") or {}
                    ent = msgs.setdefault(mid, dict(model=m.get("model"), ts=t, u=dict(input=0, cw=0, cr=0, o=0)))
                    for k, src in (("input", "input_tokens"), ("cw", "cache_creation_input_tokens"),
                                   ("cr", "cache_read_input_tokens"), ("o", "output_tokens")):
                        ent["u"][k] = max(ent["u"][k], u.get(src) or 0)
                    if first_asst_ts is None and not is_sub:
                        first_asst_ts, first_u = t, ent
                    for c in m.get("content") or []:
                        if isinstance(c, dict) and c.get("type") == "tool_use":
                            tool_uses[c.get("id")] = (t, c.get("name"), c.get("input") or {})
                    times.append((t, "a", mid))
                elif ty == "user":
                    cont = (d.get("message") or {}).get("content")
                    if isinstance(cont, list):
                        for c in cont:
                            if not (isinstance(c, dict) and c.get("type") == "tool_result"):
                                continue
                            tu = tool_uses.get(c.get("tool_use_id"))
                            txt = c.get("content")
                            if isinstance(txt, list):
                                txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
                            txt = txt if isinstance(txt, str) else ""
                            if tu and self.inwin(tu[0]):
                                dur = (t - tu[0]).total_seconds()
                                cmd = tu[2].get("command") if tu[1] in ("Bash", "PowerShell") else None
                                if cmd and dur > 60:
                                    self.longcmd["claude"][0] += 1
                                    self.longcmd["claude"][1] += dur
                                if cmd and TEST_RE.search(cmd) and not POLL_RE.search(cmd):
                                    flag = bool(c.get("is_error"))
                                    failed = flag or bool(FAIL_RE.search(txt[:3000] + txt[-1500:]))
                                    self.tests["claude"].append((tu[0].isoformat(), failed, dur, norm_cmd(cmd), flag))
                            if inwin and HOOK_RE.search(txt[:2000]) and (c.get("is_error") or "hook" in txt[:600].lower()):
                                sg = hook_sig(txt)
                                self.hooks["toolresult:" + sg] += 1
                                self.hooks_agent["claude"][sg] += 1
                    times.append((t, "u", None))
            role = "subagent" if is_sub else "main"
            for ent in msgs.values():
                t, u, model = ent["ts"], ent["u"], ent["model"] or "?"
                if not self.inwin(t) or model.startswith("<"):
                    continue
                for tgt in (C["tok_total"], C["tok_by_model"][model], C["tok_by_day"][jst_day(t)], C["tok_by_role"][role]):
                    add_tok(tgt, u["input"], u["cw"], u["cr"], u["o"])
            for t, name, inp in tool_uses.values():
                if not self.inwin(t):
                    continue
                C["tools"][name] += 1
                if name in ("Bash", "PowerShell"):
                    cmd = inp.get("command") or ""
                    C["progs"][first_prog(cmd)] += 1
                    C["cmds"][norm_cmd(cmd)] += 1
                    for p in reads_in_cmd(cmd):
                        self.add_read(p, sid, 4000 if PARTIAL_RE.search(cmd) else None)
                elif name == "Read":
                    lim = inp.get("limit")
                    self.add_read(inp.get("file_path"), sid, int(lim) * 70 if lim else None)
            if is_sub:
                C["sub_files"].add(f)
                continue
            tw = [t for t, _, _ in times if self.inwin(t)]
            if tw:
                C["sessions"].add(sid)
                for dd in {jst_day(t) for t in tw}:
                    C["sess_by_day"][dd].add(sid)
                self.sessions.append(dict(agent="claude", sid=sid, branch=branches.most_common(1)[0][0] if branches else None,
                                          start=min(tw), end=max(tw)))
                if first_asst_ts and first_ts and self.inwin(first_ts):
                    fu = first_u["u"]
                    self.startup.append(dict(agent="claude", sid=sid, kind="latency",
                                             secs=(first_asst_ts - first_ts).total_seconds(),
                                             first_ctx_tokens=fu["input"] + fu["cw"] + fu["cr"]))
            times.sort(key=lambda x: x[0])
            prev, pending = None, None
            for t, k, mid in times:
                if prev is not None and (t - prev).total_seconds() > GAP_S and self.inwin(t):
                    pending = (prev, (t - prev).total_seconds())
                if k == "a" and pending is not None and mid in msgs:
                    u = msgs[mid]["u"]
                    self.add_gap("claude", sid, pending[0], t, pending[1], u["cw"], u["cr"], u["input"])
                    pending = None
                prev = t

    # ---- Codex ----
    def process_codex(self):
        base = os.path.join(home_dir(), ".codex", "sessions")
        files = [f for f in glob.glob(os.path.join(base, "**", "*.jsonl"), recursive=True)
                 if os.path.getmtime(f) >= self.start.timestamp()]
        self.notes.append(f"codex rollout files modified in window: {len(files)}")
        X = self.agents["codex"]
        meta = {}
        for f in files:
            with open(f, encoding="utf8", errors="replace") as fh:
                for line in fh:
                    if '"session_meta"' in line[:140]:
                        try:
                            d = json.loads(line)
                            p = d["payload"]
                        except (ValueError, KeyError):
                            break
                        src = p.get("source")
                        if isinstance(src, dict) and "subagent" in src:
                            sa = src["subagent"]
                            role = "guardian" if isinstance(sa, dict) and sa.get("other") == "guardian" else "subagent"
                        else:
                            role = "root"
                        meta[f] = dict(id=p.get("id"), parent=p.get("parent_thread_id") or p.get("session_id"), role=role,
                                       branch=(p.get("git") or {}).get("branch"))
                        break
        seen = set()
        for f in files:
            mt = meta.get(f)
            if not mt:
                continue
            role = mt["role"]
            sid = (mt["id"] if role == "root" else (mt["parent"] or mt["id"])) or os.path.basename(f)
            turn_model, calls, times = {}, {}, []
            first_ts = first_call_ts = prev = pending = None
            with open(f, encoding="utf8", errors="replace") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    t = parse_ts(d.get("timestamp", "")) if isinstance(d, dict) else None
                    if not t:
                        continue
                    first_ts = first_ts or t
                    ty = d.get("type")
                    p = d.get("payload") or {}
                    pty = p.get("type")
                    inwin = self.inwin(t)
                    if ty == "turn_context":
                        turn_model[p.get("turn_id")] = (p.get("model"), p.get("effort"))
                    if ty in ("response_item", "event_msg", "token_usage_record"):
                        if prev is not None and (t - prev).total_seconds() > GAP_S and inwin and role == "root":
                            pending = (prev, (t - prev).total_seconds())
                        prev = t
                        if role == "root":
                            times.append(t)
                    if ty == "token_usage_record" and inwin:
                        rid = p.get("response_id")
                        if rid in seen:
                            continue
                        seen.add(rid)
                        u = p.get("usage") or {}
                        inp, cr = u.get("input_tokens") or 0, u.get("cached_input_tokens") or 0
                        cw, o, rs = u.get("cache_write_input_tokens") or 0, u.get("output_tokens") or 0, u.get("reasoning_output_tokens") or 0
                        model, eff = turn_model.get(p.get("turn_id"), (None, None))
                        mk = (model or "?") + (":" + eff if eff else "")
                        for tgt in (X["tok_total"], X["tok_by_model"][mk], X["tok_by_day"][jst_day(t)], X["tok_by_role"][role]):
                            add_tok(tgt, inp - cr, cw, cr, o, rs)
                        if pending is not None:
                            self.add_gap("codex", sid, pending[0], t, pending[1], cw, cr, inp - cr)
                            pending = None
                    if ty == "response_item" and pty in ("custom_tool_call", "function_call") and inwin:
                        name = p.get("name")
                        X["tools"][name] += 1
                        if first_call_ts is None and role == "root":
                            first_call_ts = t
                        if name == "sleep":
                            self.sleep["codex_sleep_calls"] += 1
                        if name == "wait":
                            self.sleep["codex_wait_calls"] += 1
                        cmds = []
                        if pty == "custom_tool_call" and name == "exec":
                            for m in re.finditer(r"cmd\s*:\s*(\"(?:[^\"\\]|\\.)*\"|`[^`]*`|'[^']*')", p.get("input") or ""):
                                s = m.group(1)
                                try:
                                    s = json.loads(s) if s[0] == '"' else s[1:-1]
                                except ValueError:
                                    s = s[1:-1]
                                cmds.append(s)
                        calls[p.get("call_id")] = (t, name, cmds)
                        for cmd in cmds:
                            X["tools"]["exec_command"] += 1
                            X["progs"][first_prog(cmd)] += 1
                            X["cmds"][norm_cmd(cmd)] += 1
                            for rp in reads_in_cmd(cmd):
                                self.add_read(rp, sid, 4000 if PARTIAL_RE.search(cmd) else None)
                    if ty == "response_item" and pty == "custom_tool_call_output" and inwin:
                        c = calls.get(p.get("call_id"))
                        if not c:
                            continue
                        out = p.get("output")
                        if isinstance(out, list):
                            out = " ".join(x.get("text", "") for x in out if isinstance(x, dict))
                        out = out if isinstance(out, str) else ""
                        m = re.search(r"Wall time ([\d.]+) seconds", out)
                        wall = float(m.group(1)) if m else (t - c[0]).total_seconds()
                        if c[2] and wall > 60:
                            self.longcmd["codex"][0] += 1
                            self.longcmd["codex"][1] += wall
                        for cmd in c[2]:
                            if TEST_RE.search(cmd) and not POLL_RE.search(cmd):
                                failed = bool(FAIL_RE.search(out[:4000] + out[-2000:]))
                                self.tests["codex"].append((c[0].isoformat(), failed, wall, norm_cmd(cmd), False))
                        if HOOK_RE.search(out[:2000]) and re.search(r"hook", out[:1500], re.I):
                            sg = hook_sig(out)
                            self.hooks["codex:" + sg] += 1
                            self.hooks_agent["codex"][sg] += 1
            if role == "root":
                tw = [t for t in times if self.inwin(t)]
                if tw:
                    X["sessions"].add(sid)
                    for dd in {jst_day(t) for t in tw}:
                        X["sess_by_day"][dd].add(sid)
                    self.sessions.append(dict(agent="codex", sid=sid, branch=mt["branch"], start=min(tw), end=max(tw)))
                    if first_call_ts and first_ts:
                        self.startup.append(dict(agent="codex", sid=sid, kind="latency",
                                                 secs=(first_call_ts - first_ts).total_seconds()))


# ---------- summaries ----------
def cost_of(tok, write_mult):
    parts = dict(input=tok["input"] * COST["input"], cache_read=tok["cache_read"] * COST["cache_read"],
                 cache_write=tok["cache_write"] * write_mult, output=tok["output"] * COST["output"])
    total = sum(parts.values())
    return parts, total


def cost_summary(u):
    out = {}
    for nm, D in u.agents.items():
        parts = dict(input=0.0, cache_read=0.0, cache_write=0.0, output=0.0)
        by_role = {}
        for role, tok in D["tok_by_role"].items():
            mult = COST["write_1h"] if (nm == "claude" and role == "main") else COST["write_5m"]
            p, tot = cost_of(tok, mult)
            by_role[role] = round(tot)
            for k, v in p.items():
                parts[k] += v
        total = sum(parts.values())
        out[nm] = dict(units=round(total), by_role_units=by_role,
                       share={k: (round(v / total, 4) if total else 0.0) for k, v in parts.items()})
    return out


def gap_summary(u, cutoff):
    res = {}
    for ag in ("claude", "codex"):
        L = [g for g in u.gaps if g["agent"] == ag]
        r = dict(gaps_gt_1h=len(L), cache_write_after_gap=sum(g["cache_write"] for g in L),
                 cache_read_after_gap=sum(g["cache_read"] for g in L),
                 median_gap_h=round((pct([g["gap_s"] for g in L], .5) or 0) / 3600, 2),
                 total_gap_h=round(sum(g["gap_s"] for g in L) / 3600, 1))
        if cutoff is not None:
            for label, sel in (("before", [g for g in L if g["start"] < cutoff]), ("after", [g for g in L if g["start"] >= cutoff])):
                r[label] = dict(gaps=len(sel), cache_write_after_gap=sum(g["cache_write"] for g in sel),
                                gap_h=round(sum(g["gap_s"] for g in sel) / 3600, 1))
        res[ag] = r
    res["cutoff_jst"] = jst_stamp(cutoff) if cutoff else None
    res["list"] = [dict(agent=g["agent"], start_jst=jst_stamp(g["start"]), sid=g["sid"], gap_h=round(g["gap_s"] / 3600, 2),
                        cache_write_first_msg=g["cache_write"])
                   for g in sorted(u.gaps, key=lambda g: g["start"])]
    return res


def default_cutoff():
    env = os.environ.get("AGENT_USAGE_CUTOFF")
    if env:
        return parse_ts(env)
    try:
        out = subprocess.run(["git", "-C", str(REPO), "log", "--diff-filter=A", "--format=%cI", "--", "scripts/keepalive.py"],
                             capture_output=True, text=True, timeout=20, encoding="utf-8", errors="replace")
        lines = out.stdout.split()
        return parse_ts(lines[-1]) if out.returncode == 0 and lines else None
    except (OSError, subprocess.SubprocessError):
        return None


def build(u, cutoff, top):
    res = dict(window=[u.start.isoformat(), u.end.isoformat()], day_buckets="JST", notes=u.notes)
    for nm, D in u.agents.items():
        r = dict(sessions_main=len(D["sessions"]), sessions_per_day={k: len(v) for k, v in sorted(D["sess_by_day"].items())},
                 tokens_total=D["tok_total"],
                 tokens_by_model=dict(sorted(D["tok_by_model"].items(), key=lambda kv: -(kv[1]["input"] + kv[1]["cache_write"] + kv[1]["output"]))),
                 tokens_by_day=dict(sorted(D["tok_by_day"].items())), tokens_by_role=dict(D["tok_by_role"]),
                 tool_mix=dict(D["tools"].most_common()), shell_first_program=dict(D["progs"].most_common(top)),
                 top_cmds=D["cmds"].most_common(top))
        if nm == "claude":
            r["subagent_transcripts"] = len(D["sub_files"])
        res[nm] = r
    res["cost_equivalent"] = dict(weights=COST, **cost_summary(u))
    rows = []
    for rel, r in u.reads.items():
        sz, cnt = u.file_size(rel), r["count"]
        est = None if sz is None else (cnt - r["partial"]) * sz + min(r["partial_bytes"], r["partial"] * sz)
        rows.append(dict(path=rel, count=cnt, sessions=len(r["sessions"]), size=sz, total_bytes=est,
                         repeat_bytes=None if est is None else est - est / cnt))
    rows.sort(key=lambda x: -x["count"])
    known = [r for r in rows if r["total_bytes"] is not None]
    res["reads"] = dict(
        distinct_paths=len(rows), total_reads=sum(r["count"] for r in rows),
        reads_of_paths_in_ge3_sessions=sum(r["count"] for r in rows if r["sessions"] >= 3),
        est_tokens_total_known=sum(r["total_bytes"] for r in known) // 4,
        est_tokens_repeat_known=int(sum(r["repeat_bytes"] for r in known)) // 4,
        top_by_count=rows[:top],
        top_by_tokens=[dict(path=r["path"], count=r["count"], sessions=r["sessions"], est_tokens=r["total_bytes"] // 4)
                       for r in sorted(known, key=lambda x: -x["total_bytes"])[:top]])
    tests = {}
    for ag, L in u.tests.items():
        durs = [x[2] for x in L]
        by_day = collections.defaultdict(lambda: [0, 0, 0.0])
        for x in L:
            b = by_day[jst_day(parse_ts(x[0]))]
            b[0] += 1
            b[1] += int(x[1])
            b[2] += x[2]
        tests[ag] = dict(runs=len(L), failed=sum(1 for x in L if x[1]), is_error_flag=sum(1 for x in L if x[4]),
                         total_wall_s=round(sum(durs)), median_s=pct(durs, .5), p90_s=pct(durs, .9),
                         over_5min=sum(1 for d in durs if d > 300), over_5min_s=round(sum(d for d in durs if d > 300)),
                         by_day={k: dict(runs=v[0], failed=v[1], wall_s=round(v[2])) for k, v in sorted(by_day.items())},
                         top_cmds=collections.Counter(x[3] for x in L).most_common(top))
    res["tests"] = tests
    res["long_commands_gt60s"] = {k: dict(count=v[0], total_s=round(v[1])) for k, v in u.longcmd.items()}
    res["codex_sleep_wait_calls"] = u.sleep
    res["hooks"] = dict(counts=dict(u.hooks.most_common(top)), by_agent={k: dict(v.most_common(top)) for k, v in u.hooks_agent.items()})
    byb = collections.defaultdict(list)
    for s in u.sessions:
        if s["branch"]:
            byb[s["branch"]].append(s)
    dup = []
    for b, L in byb.items():
        L.sort(key=lambda s: s["start"])
        multi = [s for s in L if any(o is not s and o["start"] <= s["end"] + timedelta(hours=24)
                                     and o["end"] >= s["start"] - timedelta(hours=24) for o in L)]
        if multi:
            dup.append(dict(branch=b, sessions=len(L), sessions_in_24h_overlap=len(multi),
                            agents=dict(collections.Counter(s["agent"] for s in multi)),
                            span_h=round((L[-1]["end"] - L[0]["start"]).total_seconds() / 3600, 1)))
    dup.sort(key=lambda x: -x["sessions"])
    res["branch_overlap"] = dict(branches_with_sessions=len(byb), branches_with_24h_overlapping_sessions=len(dup),
                                 branches_with_claude_and_codex=sum(1 for d in dup if len(d["agents"]) > 1),
                                 sessions_without_branch=sum(1 for s in u.sessions if not s["branch"]), top=dup[:top])
    lat = [s for s in u.startup if s.get("kind") == "latency"]
    st = {ag: dict(n=len(v), median_s=pct(v, .5), p90_s=pct(v, .9))
          for ag in ("claude", "codex") for v in [[s["secs"] for s in lat if s["agent"] == ag]]}
    fc = [s["first_ctx_tokens"] for s in lat if "first_ctx_tokens" in s]
    st["claude_first_turn_ctx_tokens"] = dict(n=len(fc), median=pct(fc, .5), p90=pct(fc, .9))
    hs = [s for s in u.startup if s.get("kind") in ("hook_success", "hook_additional_context", "hook_non_blocking_error")]
    sizes = [s["stdout_len"] + s["content_len"] for s in hs]
    durs = [s["dur_ms"] for s in hs if s["dur_ms"] is not None]
    st["claude_sessionstart_hooks"] = dict(n=len(hs), output_chars_median=pct(sizes, .5), output_chars_max=max(sizes or [0]),
                                           dur_ms_median=pct(durs, .5), dur_ms_p90=pct(durs, .9),
                                           cmds=dict(collections.Counter(s["cmd"] for s in hs).most_common(4)))
    res["startup"] = st
    res["idle_gaps"] = gap_summary(u, cutoff)
    return res


# ---------- text report ----------
def fmt_n(n):
    n = n or 0
    return f"{n / 1e6:.1f}M" if n >= 1e6 else (f"{n / 1e3:.0f}k" if n >= 1e4 else str(int(n)))


def fmt_s(x):
    return "-" if x is None else f"{x:.0f}s"


def text_report(res, top):
    k = min(top, 6)
    L = [f"Agent usage {res['window'][0][:16]} .. {res['window'][1][:16]} UTC; " + "; ".join(res["notes"])]
    L.append("== 1 Tokens (input / cache-write / cache-read / output)")
    for nm in ("claude", "codex"):
        r = res[nm]
        t = r["tokens_total"]
        L.append(f"{nm}: {r['sessions_main']} sessions, {t['msgs']} msgs; in {fmt_n(t['input'])} cw {fmt_n(t['cache_write'])} "
                 f"cr {fmt_n(t['cache_read'])} out {fmt_n(t['output'])}")
        L.append("  role: " + ", ".join(f"{ro} {fmt_n(v['cache_write'])}cw/{fmt_n(v['output'])}out/{v['msgs']}m" for ro, v in r["tokens_by_role"].items()))
        L.append("  model: " + ", ".join(f"{m} {fmt_n(v['cache_write'] + v['input'])}in+cw/{fmt_n(v['output'])}out"
                                          for m, v in list(r["tokens_by_model"].items())[:3]))
    ce = res["cost_equivalent"]
    for nm in ("claude", "codex"):
        c = ce[nm]
        L.append(f"cost-equivalent {nm}: {fmt_n(c['units'])} input-eq units; share " + " ".join(f"{a} {b * 100:.0f}%" for a, b in c["share"].items()))
    L.append("  weights: input 1, read 0.1, write 1.25 (subagents)/2.0 (main), output 5")
    L.append("== 2 Tool mix")
    for nm in ("claude", "codex"):
        L.append(f"{nm}: " + ", ".join(f"{a} {b}" for a, b in list(res[nm]["tool_mix"].items())[:8]))
    L.append("  shell programs: " + ", ".join(f"{a} {b}" for a, b in list(res["claude"]["shell_first_program"].items())[:5]))
    rd = res["reads"]
    L.append("== 3 Repeated reads")
    L.append(f"{rd['total_reads']} reads of {rd['distinct_paths']} paths; {rd['reads_of_paths_in_ge3_sessions']} in paths read by >=3 sessions; "
             f"est repeat {fmt_n(rd['est_tokens_repeat_known'])} of {fmt_n(rd['est_tokens_total_known'])} tokens")
    for r in rd["top_by_count"][:k]:
        L.append(f"  {r['count']}x/{r['sessions']}s {r['path'][-70:]}")
    L.append("== 4 Tests")
    for nm, t in res["tests"].items():
        rate = f"{100 * t['failed'] / t['runs']:.0f}%" if t["runs"] else "-"
        L.append(f"{nm}: {t['runs']} runs, {t['failed']} failed ({rate}, text-based), wall {t['total_wall_s']}s, median {fmt_s(t['median_s'])}, "
                 f"p90 {fmt_s(t['p90_s'])}, >5min {t['over_5min']} ({t['over_5min_s']}s)")
    lc = res["long_commands_gt60s"]
    L.append("  commands >60s: " + ", ".join(f"{a} {v['count']} ({v['total_s']}s)" for a, v in lc.items()))
    L.append("== 5 Hook blocks")
    L.append("  " + ("; ".join(f"{a} {b}" for a, b in list(res["hooks"]["counts"].items())[:k]) or "none"))
    bo = res["branch_overlap"]
    L.append("== 6 Branch overlap")
    L.append(f"{bo['branches_with_sessions']} branches; {bo['branches_with_24h_overlapping_sessions']} with sessions overlapping within 24h; "
             f"{bo['branches_with_claude_and_codex']} with both agents; {bo['sessions_without_branch']} sessions without branch")
    for b in bo["top"][:3]:
        L.append(f"  {b['branch'][:50]}: {b['sessions']} sessions, {b['sessions_in_24h_overlap']} overlapping, {b['span_h']}h")
    st = res["startup"]
    L.append("== 7 Session start")
    L.append(f"latency to first reply/call: claude median {fmt_s(st['claude']['median_s'])} p90 {fmt_s(st['claude']['p90_s'])} (n={st['claude']['n']}); "
             f"codex median {fmt_s(st['codex']['median_s'])} p90 {fmt_s(st['codex']['p90_s'])} (n={st['codex']['n']})")
    h = st["claude_sessionstart_hooks"]
    L.append(f"  claude first-turn context median {fmt_n(st['claude_first_turn_ctx_tokens']['median'])} tokens; SessionStart hooks n={h['n']}, "
             f"output chars median {h['output_chars_median']} max {h['output_chars_max']}, median {h['dur_ms_median']} ms")
    g = res["idle_gaps"]
    L.append("== 8 Idle gaps > 1 h (first message after the gap)")
    for ag in ("claude", "codex"):
        r = g[ag]
        L.append(f"{ag}: {r['gaps_gt_1h']} gaps, median {r['median_gap_h']} h, total {r['total_gap_h']} h, post-gap cache-write {fmt_n(r['cache_write_after_gap'])}")
    if g["cutoff_jst"]:
        for ag in ("claude", "codex"):
            b, a = g[ag]["before"], g[ag]["after"]
            L.append(f"  {ag} keepalive cutoff {g['cutoff_jst']} JST: before {b['gaps']} gaps / {fmt_n(b['cache_write_after_gap'])} post-gap cw; "
                     f"after {a['gaps']} gaps / {fmt_n(a['cache_write_after_gap'])} post-gap cw")
    else:
        L.append("  no keepalive cutoff known (set --cutoff or AGENT_USAGE_CUTOFF)")
    cut = g["cutoff_jst"]
    recent = [x for x in g["list"] if x["agent"] == "claude" and (cut is None or x["start_jst"] >= cut)][-5:]
    for x in recent:
        L.append(f"  gap {x['start_jst']} JST {x['sid']} {x['gap_h']} h cw {fmt_n(x['cache_write_first_msg'])}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Report where Claude Code and Codex effort and tokens went.")
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--since", help="window start as a JST date YYYY-MM-DD (overrides --days)")
    ap.add_argument("--json", metavar="OUT", help="write the full result as JSON to OUT")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--cutoff", help="ISO instant splitting idle gaps before/after (default: keepalive.py's commit)")
    a = ap.parse_args(argv)
    end = datetime.now(timezone.utc)
    start = (datetime.strptime(a.since, "%Y-%m-%d").replace(tzinfo=JST).astimezone(timezone.utc)
             if a.since else end - timedelta(days=a.days))
    cutoff = parse_ts(a.cutoff) if a.cutoff else default_cutoff()
    if cutoff is not None and cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=JST)
    u = Usage(start, end)
    u.process_claude()
    u.process_codex()
    res = build(u, cutoff, a.top)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str, ensure_ascii=False), encoding="utf8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    print(text_report(res, a.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
