"""Development memory search: "have we tried / decided / seen this before, and where is the Evidence" in one call.

    python scripts/devmem.py query "<text>" [-k 8] [--json] [--no-refresh] [--mode auto|bm25|dense|hybrid]
    python scripts/devmem.py build [--no-embed]      full rebuild of the index from the sources
    python scripts/devmem.py refresh [--no-embed]    incremental (new commits, changed files, missing embeddings)
    python scripts/devmem.py stats

The index points at sources, it does not copy them out as Evidence: every hit prints path:line-range (or a commit
SHA) and the reader opens the source. The index lives in %LOCALAPPDATA%\\LSS-dev\\devmem\\devmem.sqlite, is
regenerable from the sources at any time and is never cited as Evidence.

Sources (text only, files under 1 MB, media / venvs / node_modules / .git skipped):
  git       message + body + changed file names of every commit of every local branch (`git log --all`)
  repo      checkout docs/**/*.md and root *.md; shared Evidence home release-evidence/**/*.md|json
  py        module docstrings of lss/**/*.py and scripts/*.py
  dev       optional explicitly configured SSD note roots (no HDD archive scan by default)
  worktree  the same file rule under <repo>\\.worktrees\\*\\
Identical chunks (same text in several places, e.g. docs copied into worktrees) are stored once; the other
locations are aliases and are reported as "+N copies".

Retrieval: SQLite FTS5 BM25 (unicode61; path / heading / text columns) fused by reciprocal rank with trigram BM25
(only for queries that contain Japanese) and dense retrieval (Qwen3-Embedding-0.6B vectors stored as blobs in the
same SQLite). Dense needs the query embedded by a local llama-server; when no embedding server is reachable
within about 2 s the query falls back to BM25 and the output says so. Refresh runs automatically at query time
(bounded by --budget seconds, new commits and changed files only); a full rebuild only through `build`.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root


import argparse
import ast
import contextlib
import ctypes
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:  # numpy is optional: without it only BM25 runs
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

REPO = Path(__file__).resolve().parents[1]
DB_DIR = Path(os.path.expandvars(r"%LOCALAPPDATA%\LSS-dev\devmem"))
SCHEMA = 4
MAX_FILE = 1_000_000
CHUNK_MAX = 1600
CHUNK_MIN = 350
EMBED_DIM = 1024
EMBED_MODEL = "qwen3-embedding-0.6b"
EMBED_URL_ENV = "DEVMEM_EMBED_URL"
QUERY_INSTRUCT = "Given a question about a software project's history, retrieve the notes, commit messages and reports that answer it"
PRUNE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "site-packages", "dist", "build", ".models",
              ".providers", "TestCorpus", "vendor", "bin", "assets", "_lss_doc_backups", ".lss_doc_backups", "_internal",
              "dist-packages", "pkgs", ".state-cache", "stems", "wav", "audio", "media"}
PRUNE_SUFFIX = (".dist-info", ".egg-info", ".dSYM")
MAX_DEPTH = 9
TEXT_SUFFIX = {".md", ".txt", ".json", ".tsv", ".csv", ".markdown", ""}
NAMED = ("verdict", "results", "report", "handoff")
KIND_PRIO = {"git": 0, "repo": 1, "py": 2, "dev": 3, "worktree": 4}
MAX_CHUNKS = {".json": 40, ".tsv": 40, ".csv": 40}
STOP = set("""a an the of to in on for with and or but is are was were be been being it its this that these those as at by
from into than then so such not no do does did have has had can could should would will what which who whom whose why how
when where we our us i you they them their there here about if else also only just very more most some any each other
between over under again further once""".split())
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]+")


@dataclass
class Config:
    repo: Path = REPO
    db_dir: Path = DB_DIR
    dev_roots: list = field(default_factory=list)
    worktrees: Path | None = None  # default <repo>\.worktrees

    def __post_init__(self):
        self.repo = Path(self.repo)
        self.db_dir = Path(self.db_dir)
        if self.worktrees is None:
            main = self.repo.parent.parent if self.repo.parent.name == ".worktrees" else self.repo
            self.worktrees = main / ".worktrees"

    @property
    def db(self) -> Path:
        return self.db_dir / "devmem.sqlite"


# --------------------------------------------------------------------------- chunking

def _split_long(line: str, limit: int):
    while len(line) > limit:
        cut = line.rfind(" ", 0, limit)
        cut = cut if cut > limit // 2 else limit
        yield line[:cut]
        line = line[cut:]
    yield line


def chunk_text(text: str, *, markdown: bool = True, limit: int = CHUNK_MAX, floor: int = CHUNK_MIN):
    """Yield (line_start, line_end, heading, text); 1-based inclusive lines. Splits at headings and paragraphs."""
    lines = text.splitlines()
    heads: list[tuple[int, str]] = []
    cur: list[tuple[int, str]] = []
    size = 0
    start_head = ""

    def flush():
        nonlocal cur, size
        body = "\n".join(t for _, t in cur).strip()
        if body:
            ln = [n for n, _ in cur]
            out.append((ln[0], ln[-1], start_head, body))
        cur, size = [], 0

    out: list = []
    for i, raw in enumerate(lines, 1):
        m = re.match(r"(#{1,6})\s+(.*)", raw) if markdown else None
        if m:
            level = len(m.group(1))
            heads = [h for h in heads if h[0] < level] + [(level, m.group(2).strip())]
            if size >= floor:
                flush()
            if not cur:
                start_head = " > ".join(h[1] for h in heads)
        elif not cur:
            start_head = " > ".join(h[1] for h in heads)
        blank = not raw.strip()
        for piece in _split_long(raw, limit):
            if size + len(piece) > limit and size >= floor:
                flush()
                start_head = " > ".join(h[1] for h in heads)
            cur.append((i, piece))
            size += len(piece) + 1
        if blank and size >= limit * 0.6:
            flush()
    flush()
    return out


def docstring_chunks(src: str):
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return []
    doc = ast.get_docstring(tree, clean=False)
    if not doc:
        return []
    node = tree.body[0]
    l0 = node.lineno
    l1 = getattr(node, "end_lineno", l0)
    parts = chunk_text(doc, markdown=False)
    if len(parts) == 1:
        return [(l0, l1, "", parts[0][3])]
    return [(l0 + a - 1, l0 + b - 1, "", t) for a, b, _, t in parts]


# --------------------------------------------------------------------------- sources

def _text_file_ok(entry: os.DirEntry) -> bool:
    name = entry.name.lower()
    ext = os.path.splitext(name)[1]
    if ext not in TEXT_SUFFIX:
        return False
    if name.endswith((".md", ".markdown")) or name.startswith(NAMED):
        return True
    return False


def _walk(root: Path, accept, *, deadline: float | None = None, follow_junctions: bool = False):
    """Files under `root` accepted by `accept(DirEntry)`. Junctions (tiered storage leaves them behind for demoted
    directories) are skipped unless `follow_junctions`, then each real target is walked once."""
    seen: set[str] = set()
    stack = [(str(root), 0)]
    while stack:
        if deadline and time.time() > deadline:
            raise TimeoutError
        cur, depth = stack.pop()
        try:
            entries = list(os.scandir(cur))
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    n = e.name
                    if n in PRUNE_DIRS or n.endswith(PRUNE_SUFFIX) or n.startswith((".pytest", "pytest-")):
                        continue
                    if getattr(e.stat(follow_symlinks=False), "st_file_attributes", 0) & 0x400:
                        real = os.path.normcase(os.path.realpath(e.path))
                        if not follow_junctions or real in seen:
                            continue
                        seen.add(real)
                    if depth < MAX_DEPTH:
                        stack.append((e.path, depth + 1))
                elif accept(e):
                    st = e.stat()
                    if 0 < st.st_size <= MAX_FILE:
                        yield e.path, st.st_mtime, st.st_size
            except OSError:
                continue


def repo_files(cfg: Config):
    repo = cfg.repo
    for e in os.scandir(repo):
        if e.is_file() and e.name.lower().endswith(".md"):
            st = e.stat()
            yield "repo", e.path, st.st_mtime, st.st_size
    for sub, exts in (("docs", (".md",)), ("release-evidence", (".md", ".json"))):
        root = (dev_evidence_root(repo) if sub == "release-evidence" else repo) / sub
        if root.is_dir():
            for p, m, s in _walk(root, lambda e, exts=exts: e.name.lower().endswith(exts)):
                yield "repo", p, m, s
    for sub, pattern in (("lss", True), ("scripts", False)):
        root = repo / sub
        if not root.is_dir():
            continue
        if pattern:
            it = _walk(root, lambda e: e.name.endswith(".py"))
        else:
            it = ((e.path, e.stat().st_mtime, e.stat().st_size) for e in os.scandir(root)
                  if e.is_file() and e.name.endswith(".py"))
        for p, m, s in it:
            yield "py", p, m, s


def dev_files(cfg: Config, *, deadline: float | None = None):
    from dev_hot_paths import require_hot, HotPathError
    for root in cfg.dev_roots:
        try:
            require_hot(root, "developer notes")
        except (HotPathError, OSError):
            continue
        if root.is_dir():
            for p, m, s in _walk(root, _text_file_ok, deadline=deadline, follow_junctions=True):
                yield "dev", p, m, s
    wt = cfg.worktrees
    if wt:
        try:
            require_hot(wt, "developer worktrees")
        except (HotPathError, OSError):
            return
    if wt and wt.is_dir():
        for p, m, s in _walk(wt, _text_file_ok, deadline=deadline):
            if "release-evidence" in p.replace("\\", "/") or True:
                yield "worktree", p, m, s


def file_chunks(kind: str, path: str, repo: Path):
    try:
        data = Path(path).read_bytes()
    except OSError:
        return []
    if b"\x00" in data[:4096]:
        return []
    text = data.decode("utf-8", errors="replace")
    ext = os.path.splitext(path)[1].lower()
    if kind == "py":
        return docstring_chunks(text)
    parts = chunk_text(text, markdown=ext in (".md", ".markdown", ".txt", ""))
    cap = MAX_CHUNKS.get(ext)
    return parts[:cap] if cap else parts


def display_path(path: str, cfg: Config) -> str:
    """Repo-relative for the repo, drive-qualified otherwise."""
    p = Path(path)
    try:
        return p.relative_to(cfg.repo).as_posix()
    except ValueError:
        return str(p).replace("\\", "/")


def _git(cfg: Config, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(cfg.repo), *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.stdout if r.returncode == 0 else ""


def git_shas(cfg: Config) -> list[str]:
    return _git(cfg, "rev-list", "--all").split()


def git_commits(cfg: Config, shas: list[str]):
    """Yield dicts for the given commits (message, body, refs, changed files)."""
    if not shas:
        return
    fmt = "%x1e%H%x1f%cI%x1f%an%x1f%D%x1f%s%x1f%b%x1f"
    for i in range(0, len(shas), 400):
        batch = shas[i:i + 400]
        r = subprocess.run(["git", "-C", str(cfg.repo), "show", "--no-patch", f"--format={fmt}", *batch],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        names = subprocess.run(["git", "-C", str(cfg.repo), "diff-tree", "--name-only", "-r", "--root", "--stdin"], input="\n".join(batch) + "\n",
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        # diff-tree --stdin prints the sha line before each file list
        files: dict[str, list[str]] = {}
        sha = None
        for line in names.splitlines():
            if re.fullmatch(r"[0-9a-f]{40}", line):
                sha = line
                files[sha] = []
            elif line and sha:
                files[sha].append(line)
        for rec in r.stdout.split("\x1e"):
            f = rec.split("\x1f")
            if len(f) < 6 or not re.fullmatch(r"[0-9a-f]{40}", f[0].strip()):
                continue
            yield {"sha": f[0].strip(), "date": f[1][:10], "author": f[2], "refs": f[3], "subject": f[4],
                   "body": f[5].strip(), "files": files.get(f[0].strip(), [])}


def commit_chunks(c: dict):
    text = c["body"]
    extra = []
    if c["files"]:
        extra.append("Files: " + ", ".join(c["files"][:40]))
    if c["refs"]:
        extra.append("Refs: " + c["refs"])
    full = (text + "\n" if text else "") + "\n".join(extra)
    parts = chunk_text(full, markdown=False, limit=CHUNK_MAX * 2)
    if not parts:
        return [(1, 1, c["subject"], "")]
    return [(a, b, c["subject"], t) for a, b, _, t in parts]


# --------------------------------------------------------------------------- store

def connect(cfg: Config, *, create: bool = True) -> sqlite3.Connection:
    cfg.db_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(cfg.db), timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    ver = con.execute("PRAGMA user_version").fetchone()[0]
    if ver != SCHEMA:
        if ver and not create:
            return con
        _reset(con)
    return con


def _reset(con: sqlite3.Connection):
    for t in ("fts", "tri"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    for t in ("chunk", "alias", "src", "emb", "meta"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.executescript("""
    CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE src(id INTEGER PRIMARY KEY, key TEXT UNIQUE, kind TEXT, mtime REAL, size INTEGER);
    CREATE TABLE chunk(id INTEGER PRIMARY KEY, src_id INTEGER, hash TEXT UNIQUE, kind TEXT, path TEXT,
                       l0 INTEGER, l1 INTEGER, date TEXT, sha TEXT, heading TEXT, text TEXT, prio INTEGER);
    CREATE INDEX chunk_src ON chunk(src_id);
    CREATE TABLE alias(id INTEGER PRIMARY KEY, src_id INTEGER, hash TEXT, path TEXT, l0 INTEGER, l1 INTEGER,
                       date TEXT, kind TEXT, prio INTEGER, sha TEXT);
    CREATE INDEX alias_src ON alias(src_id);
    CREATE INDEX alias_hash ON alias(hash);
    CREATE TABLE emb(hash TEXT PRIMARY KEY, vec BLOB);
    CREATE VIRTUAL TABLE fts USING fts5(path, heading, text, content='chunk', content_rowid='id',
                                        tokenize="unicode61 remove_diacritics 2");
    CREATE VIRTUAL TABLE tri USING fts5(path, heading, text, content='chunk', content_rowid='id',
                                        tokenize="trigram");
    """)
    con.execute(f"PRAGMA user_version={SCHEMA}")
    con.commit()


def _hash(path: str, heading: str, text: str) -> str:
    return hashlib.sha1(f"{heading}\x00{text}".encode("utf-8", "replace")).hexdigest()[:20]


def _drop_source(con: sqlite3.Connection, src_id: int, tri: bool = True):
    # Copies in the removed source cannot become the surviving primary.
    con.execute("DELETE FROM alias WHERE src_id=?", (src_id,))
    rows = con.execute("SELECT id, hash, path, heading, text FROM chunk WHERE src_id=?", (src_id,)).fetchall()
    for cid, h, path, heading, text in rows:
        con.execute("INSERT INTO fts(fts,rowid,path,heading,text) VALUES('delete',?,?,?,?)", (cid, path, heading, text))
        con.execute("INSERT INTO tri(tri,rowid,path,heading,text) VALUES('delete',?,?,?,?)", (cid, path, heading, text))
        con.execute("DELETE FROM chunk WHERE id=?", (cid,))
        alt = con.execute("SELECT id, src_id, path, l0, l1, date, kind, prio, sha FROM alias WHERE hash=? ORDER BY prio, id "
                          "LIMIT 1", (h,)).fetchone()
        if alt:  # promote a copy to primary
            aid, asrc, apath, l0, l1, adate, akind, aprio, sha = alt
            con.execute("DELETE FROM alias WHERE id=?", (aid,))
            cur = con.execute("INSERT INTO chunk(src_id,hash,kind,path,l0,l1,date,sha,heading,text,prio) "
                              "VALUES(?,?,?,?,?,?,?,?,?,?,?)", (asrc, h, akind, apath, l0, l1, adate, sha, heading, text, aprio))
            _index(con, cur.lastrowid, apath, heading, text)
        else:
            con.execute("DELETE FROM emb WHERE hash=?", (h,))
    con.execute("DELETE FROM alias WHERE src_id=?", (src_id,))
    con.execute("DELETE FROM src WHERE id=?", (src_id,))


def _index(con, cid, path, heading, text):
    con.execute("INSERT INTO fts(rowid,path,heading,text) VALUES(?,?,?,?)", (cid, path, heading, text))
    con.execute("INSERT INTO tri(rowid,path,heading,text) VALUES(?,?,?,?)", (cid, path, heading, text))


def _add_source(con, key, kind, mtime, size, chunks, path, date, sha=""):
    """chunks: iterable of (l0, l1, heading, text)."""
    row = con.execute("SELECT id FROM src WHERE key=?", (key,)).fetchone()
    if row:
        _drop_source(con, row[0])
    sid = con.execute("INSERT INTO src(key,kind,mtime,size) VALUES(?,?,?,?)", (key, kind, mtime, size)).lastrowid
    prio = KIND_PRIO[kind]
    n = 0
    for l0, l1, heading, text in chunks:
        h = _hash(path, heading, text)
        old = con.execute("SELECT id, prio FROM chunk WHERE hash=?", (h,)).fetchone()
        if old:
            con.execute("INSERT INTO alias(src_id,hash,path,l0,l1,date,kind,prio,sha) VALUES(?,?,?,?,?,?,?,?,?)",
                        (sid, h, path, l0, l1, date, kind, prio, sha))
            continue
        cid = con.execute("INSERT INTO chunk(src_id,hash,kind,path,l0,l1,date,sha,heading,text,prio) "
                          "VALUES(?,?,?,?,?,?,?,?,?,?,?)", (sid, h, kind, path, l0, l1, date, sha, heading, text, prio)).lastrowid
        _index(con, cid, path, heading, text)
        n += 1
    return n


# --------------------------------------------------------------------------- build / refresh

def _date(mtime: float) -> str:
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")


def refresh_git(con, cfg: Config) -> int:
    have = {r[0][4:] for r in con.execute("SELECT key FROM src WHERE kind='git'")}
    now = set(git_shas(cfg))
    new = [s for s in now if s not in have]
    for sha in have - now:
        row = con.execute("SELECT id FROM src WHERE key=?", ("git:" + sha,)).fetchone()
        if row:
            _drop_source(con, row[0])
    for c in git_commits(cfg, new):
        _add_source(con, "git:" + c["sha"], "git", 0, 0, commit_chunks(c), "git:" + c["sha"][:10], c["date"], c["sha"])
    con.commit()
    return len(new)


def refresh_files(con, cfg: Config, kinds: set[str], deadline: float | None = None) -> dict:
    """Index changed / new files of the given kinds; drop vanished ones only when the walk completed."""
    stats = {"changed": 0, "removed": 0, "complete": True}
    seen: dict[str, set] = {k: set() for k in kinds}
    known = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT key, mtime, size, kind FROM src WHERE kind!='git'")}
    gens = []
    if kinds & {"repo", "py"}:
        gens.append(repo_files(cfg))
    if kinds & {"dev", "worktree"}:
        gens.append(dev_files(cfg, deadline=deadline))
    try:
        for gen in gens:
            for kind, path, mtime, size in gen:
                key = os.path.normcase(path)
                seen.setdefault(kind, set()).add(key)
                if kind not in kinds:
                    continue
                prev = known.get(key)
                if prev and abs(prev[0] - mtime) < 1e-3 and prev[1] == size:
                    continue
                chunks = file_chunks(kind, path, cfg.repo)
                _add_source(con, key, kind, mtime, size, chunks, display_path(path, cfg), _date(mtime))
                stats["changed"] += 1
                if stats["changed"] % 150 == 0:
                    con.commit()
                if deadline and time.time() > deadline:
                    raise TimeoutError
    except TimeoutError:
        stats["complete"] = False
    if stats["complete"]:
        for key, (m, s, kind) in known.items():
            if kind in kinds and key not in seen.get(kind, set()):
                row = con.execute("SELECT id FROM src WHERE key=?", (key,)).fetchone()
                if row:
                    _drop_source(con, row[0])
                    stats["removed"] += 1
    con.commit()
    return stats


def refresh(cfg: Config, *, budget: float | None = None, embed: bool = True, kinds=None, log=None,
            embedder=None, lease_wait: float = 30.0) -> dict:
    """Incremental refresh. `kinds` limits the file kinds (query-time refresh skips the slow dev roots when fresh)."""
    t0 = time.time()
    deadline = t0 + budget if budget else None
    con = connect(cfg)
    out = {"git_new": refresh_git(con, cfg)}
    kinds = set(kinds or {"repo", "py", "dev", "worktree"})
    out["files"] = refresh_files(con, cfg, kinds, deadline)
    if kinds & {"dev", "worktree"} and out["files"]["complete"]:
        con.execute("INSERT OR REPLACE INTO meta VALUES('dev_scan', ?)", (str(time.time()),))
    if embed:
        if embedder is not None:
            out["embedded"] = embed_missing(con, embedder=embedder, deadline=deadline, log=log)
        elif con.execute("SELECT 1 FROM chunk c LEFT JOIN emb e ON e.hash=c.hash WHERE e.hash IS NULL LIMIT 1").fetchone():
            with query_embedder_session(lease_wait, startup=120.0) as emb:
                out["embedded"] = embed_missing(con, embedder=emb, deadline=deadline, log=log)
        else:
            out["embedded"] = 0
    con.execute("INSERT OR REPLACE INTO meta VALUES('refreshed', ?)", (str(time.time()),))
    con.commit()
    con.execute("INSERT INTO fts(fts) VALUES('optimize')") if out["git_new"] > 200 else None
    con.commit()
    con.close()
    out["seconds"] = round(time.time() - t0, 2)
    return out


def build(cfg: Config, *, embed: bool = True, log=None) -> dict:
    t0 = time.time()
    if cfg.db.exists():
        for ext in ("", "-wal", "-shm"):
            try:
                os.remove(str(cfg.db) + ext)
            except OSError:
                pass
    out = refresh(cfg, embed=embed, log=log)
    out["build_seconds"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- dense

def embed_text(path: str, heading: str, text: str) -> str:
    return f"{path}\n{heading}\n{text}"[:2200]


class Embedder:
    """OpenAI-style /v1/embeddings client for a llama-server started with --embeddings."""

    def __init__(self, url: str, timeout: float = 120):
        url = url.rstrip("/")
        self.url = url[:-3] if url.endswith("/v1") else url
        self.timeout = timeout

    def alive(self, wait: float = 0.0) -> bool:
        end = time.time() + wait
        while True:
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=1.5) as r:
                    return r.status == 200
            except (OSError, urllib.error.URLError):
                if time.time() >= end:
                    return False
                time.sleep(0.25)

    def embed(self, texts: list[str]):
        body = json.dumps({"input": texts, "model": EMBED_MODEL}).encode()
        req = urllib.request.Request(self.url + "/v1/embeddings", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())["data"]
        data.sort(key=lambda d: d["index"])
        m = np.asarray([d["embedding"] for d in data], dtype=np.float32)
        m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        return m

    def embed_query(self, q: str):
        return self.embed([f"Instruct: {QUERY_INSTRUCT}\nQuery:{q}"])[0]


LEASE_NAME = r"Local\LSS-CUDA-Device-0-v1"  # the named mutex of lss.core.gpu_lease


def lease_free(wait: float) -> bool:
    """True when the GPU lease can be taken within `wait` seconds (taken and released at once: a probe)."""
    if os.name != "nt":
        return True
    k = ctypes.windll.kernel32
    k.CreateMutexW.restype = ctypes.c_void_p
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.ReleaseMutex.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    h = k.CreateMutexW(None, False, LEASE_NAME)
    if not h:
        return False
    try:
        state = k.WaitForSingleObject(h, int(wait * 1000))
        if state in (0, 0x80):
            k.ReleaseMutex(h)
            return True
        return False
    finally:
        k.CloseHandle(h)


class _ServerThread(threading.Thread):
    """Runs scripts/local_llm.serving(...) in its own thread (the lease is released by the thread that took it)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.ready, self.stop = threading.Event(), threading.Event()
        self.base = self.error = None

    def run(self):
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import local_llm
            with local_llm.serving(EMBED_MODEL, purpose="devmem", normal_priority=True, log=lambda _m: None) as base:
                self.base = base
                self.ready.set()
                self.stop.wait()
        except BaseException as exc:  # refused, lease lost, model missing: the caller falls back to BM25
            self.error = exc
        finally:
            self.ready.set()


@contextlib.contextmanager
def query_embedder_session(lease_wait: float = 2.0, startup: float = 15.0):
    """Yield an Embedder, or None when dense retrieval is unavailable (output says so).

    DEVMEM_EMBED_URL names an already running embedding server. Otherwise scripts/local_llm.py serves
    qwen3-embedding-0.6b behind the GPU lease, but only when the lease is free within `lease_wait` seconds."""
    url = os.environ.get(EMBED_URL_ENV)
    if url:
        e = Embedder(url)
        yield e if e.alive(lease_wait) else None
        return
    if np is None or not lease_free(lease_wait):
        yield None
        return
    th = _ServerThread()
    th.start()
    th.ready.wait(startup)
    try:
        yield Embedder(th.base) if th.base else None
    finally:
        th.stop.set()
        th.join(10)


def embed_missing(con, *, embedder=None, deadline=None, log=None, batch: int = 16) -> int:
    if np is None:
        return 0
    rows = con.execute("SELECT c.hash, c.path, c.heading, c.text FROM chunk c LEFT JOIN emb e ON e.hash=c.hash "
                       "WHERE e.hash IS NULL").fetchall()
    if not rows:
        return 0
    emb = embedder
    if emb is None:
        if log:
            log(f"{len(rows)} chunks lack embeddings (no embedding server)")
        return 0
    rows.sort(key=lambda r: len(r[3]))
    done = 0
    for i in range(0, len(rows), batch):
        if deadline and time.time() > deadline:
            break
        part = rows[i:i + batch]
        try:
            m = emb.embed([embed_text(p, h, t) for _, p, h, t in part])
        except (OSError, urllib.error.URLError, KeyError, ValueError):
            break
        con.executemany("INSERT OR REPLACE INTO emb VALUES(?,?)",
                        [(r[0], m[j].astype(np.float16).tobytes()) for j, r in enumerate(part)])
        done += len(part)
        if i % (batch * 50) == 0:
            con.commit()
            if log:
                log(f"embedded {done}/{len(rows)}")
    con.commit()
    return done


_MATRIX: dict = {}


def load_matrix(con):
    """(hashes, float32 matrix) of all stored embeddings, cached per process by row count."""
    n = con.execute("SELECT COUNT(*) FROM emb").fetchone()[0]
    key = (str(con.execute("PRAGMA database_list").fetchone()[2]), n)
    if _MATRIX.get("key") == key:
        return _MATRIX["h"], _MATRIX["m"]
    hashes, blobs = [], []
    for h, v in con.execute("SELECT hash, vec FROM emb"):
        hashes.append(h)
        blobs.append(v)
    m = np.frombuffer(b"".join(blobs), dtype=np.float16).reshape(len(hashes), -1).astype(np.float32) if hashes else None
    _MATRIX.update(key=key, h=hashes, m=m)
    return hashes, m


# --------------------------------------------------------------------------- search

def _fts_terms(q: str):
    """(unicode61 OR-query, list of phrase fragments) for an English / identifier query."""
    toks = re.findall(r"[0-9A-Za-z_][0-9A-Za-z_.\-/\\]*[0-9A-Za-z_]|[0-9A-Za-z]", q)
    terms, phrases = [], []
    for t in toks:
        parts = [p for p in re.split(r"[_.\-/\\]+", t) if p]
        if len(parts) > 1:
            phrases.append('"' + " ".join(parts) + '"')
            terms.extend(p for p in parts if len(p) > 1 and p.lower() not in STOP)
            continue
        low = t.lower()
        if low in STOP or len(low) < 2:
            continue
        if re.fullmatch(r"[0-9a-f]{6,40}", low):
            terms.append(f'"{low}"*')
        else:
            terms.append(f'"{low}"')
    seen, out = set(), []
    for t in phrases + terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def bm25_ids(con, q: str, n: int = 50):
    terms = _fts_terms(q)
    if not terms:
        return []
    try:
        rows = con.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts, 2.0, 2.0, 1.0) LIMIT ?",
                           (" OR ".join(terms[:40]), n)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


def trigram_ids(con, q: str, n: int = 50):
    grams = []
    for run in CJK.findall(q):
        if len(run) < 3:
            continue
        grams.extend(run[i:i + 3] for i in range(len(run) - 2))
    for t in re.findall(r"[0-9A-Za-z_.\-]{3,}", q):
        if t.lower() not in STOP:
            grams.append(t.lower())
    grams = list(dict.fromkeys(grams))[:60]
    if not grams:
        return []
    expr = " OR ".join('"' + g.replace('"', "") + '"' for g in grams)
    try:
        rows = con.execute("SELECT rowid FROM tri WHERE tri MATCH ? ORDER BY bm25(tri, 2.0, 2.0, 1.0) LIMIT ?",
                           (expr, n)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


def dense_ids(con, qvec, n: int = 50):
    hashes, m = load_matrix(con)
    if m is None or qvec is None:
        return []
    sims = m @ qvec
    top = np.argpartition(-sims, min(n, len(sims) - 1))[:n]
    top = top[np.argsort(-sims[top])]
    marks = ",".join("?" * len(top))
    by = dict(con.execute(f"SELECT hash, id FROM chunk WHERE hash IN ({marks})", [hashes[i] for i in top]).fetchall())
    return [by[hashes[i]] for i in top if hashes[i] in by]


def rrf(lists, k: int = 60, weights=None):
    score: dict = {}
    for li, ids in enumerate(lists):
        w = weights[li] if weights else 1.0
        for r, cid in enumerate(ids):
            score[cid] = score.get(cid, 0.0) + w / (k + r + 1)
    return sorted(score, key=score.get, reverse=True)


def search(con, q: str, *, k: int = 8, mode: str = "auto", embedder=None, rerank=None) -> tuple[list[int], dict]:
    """Ranked chunk ids and a note dict (which retrievers contributed)."""
    note = {"mode": mode}
    has_cjk = bool(CJK.search(q))
    lists = []
    if mode in ("auto", "hybrid", "bm25"):
        lists.append(bm25_ids(con, q))
    if mode in ("auto", "hybrid", "tri") and (has_cjk or mode == "tri"):
        lists.append(trigram_ids(con, q))
    if mode == "tri" and not lists[-1:]:
        lists.append([])
    if mode in ("auto", "hybrid", "dense") and np is not None:
        qv = None
        if embedder is not None:
            try:
                qv = embedder.embed_query(q)
            except (OSError, urllib.error.URLError, KeyError, ValueError):
                qv = None
        if qv is not None:
            lists.append(dense_ids(con, qv))
            note["dense"] = True
        else:
            note["dense"] = False
    ids = rrf(lists) if len(lists) > 1 else (lists[0] if lists else [])
    if rerank is not None and ids:
        ids = rerank(q, ids[:30]) + ids[30:]
    return ids[:max(k, 10)], note


# --------------------------------------------------------------------------- output

def _snippet(text: str, q: str, width: int = 130) -> str:
    words = {w.lower() for w in re.findall(r"[0-9A-Za-z_]{3,}", q) if w.lower() not in STOP}
    grams = {g for run in CJK.findall(q) for g in (run[i:i + 2] for i in range(max(len(run) - 1, 0)))}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""

    def sc(l):
        low = l.lower()
        return sum(w in low for w in words) + 0.3 * sum(g in l for g in grams)
    best = max(range(len(lines)), key=lambda i: (sc(lines[i]), -i))
    s = lines[best]
    if len(s) > width:
        low = s.lower()
        pos = min((low.find(w) for w in words if w in low), default=0)
        a = max(0, pos - 30)
        s = ("..." if a else "") + s[a:a + width] + "..."
    return s


def hits(con, ids, q, k, cfg: Config):
    out = []
    for cid in ids[:k]:
        r = con.execute("SELECT kind, path, l0, l1, date, sha, heading, text, hash FROM chunk WHERE id=?", (cid,)).fetchone()
        if not r:
            continue
        kind, path, l0, l1, date, sha, heading, text, h = r
        copies = con.execute("SELECT COUNT(*) FROM alias WHERE hash=?", (h,)).fetchone()[0]
        out.append({"kind": kind, "path": path, "lines": [l0, l1], "date": date, "commit": sha, "heading": heading,
                    "snippet": _snippet(text, q), "copies": copies})
    return out


def format_hits(rows, note: dict, budget: int = 1500) -> str:
    """One block per hit: locator line (git: the subject too), then a one-line snippet. Stops at `budget` bytes."""
    blocks = []
    for i, h in enumerate(rows, 1):
        if h["kind"] == "git":
            head = f"{i}. commit {h['commit'][:10]} {h['date']} {h['heading'][:90]}"
        else:
            head = f"{i}. {h['path']}:{h['lines'][0]}-{h['lines'][1]} {h['date']}"
        if h["copies"]:
            head += f" (+{h['copies']} copies)"
        block = head + ("\n   " + h["snippet"] if h["snippet"] else "")
        if blocks and sum(len(x.encode()) + 1 for x in blocks) + len(block.encode()) > budget:
            break
        blocks.append(block)
    out = "\n".join(blocks) if blocks else "no hits"
    if note.get("dense") is False:
        out += "\n(dense retrieval unavailable: BM25 only)"
    return out


# --------------------------------------------------------------------------- CLI

def stale_kinds(con, cfg: Config, max_age: float = 1800.0) -> set[str]:
    """Repo files are checked on every query (fast SSD); dev roots and worktrees at most every `max_age` seconds."""
    kinds = {"repo", "py"}
    row = con.execute("SELECT v FROM meta WHERE k='dev_scan'").fetchone()
    if not row or time.time() - float(row[0]) > max_age:
        kinds |= {"dev", "worktree"}
    return kinds


def connect_readonly(cfg: Config) -> sqlite3.Connection:
    """Open a live index without schema setup or data writes; WAL sidecars may be managed."""
    uri = cfg.db.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30)


def cmd_query(args, cfg: Config) -> int:
    if not cfg.db.exists():
        print("devmem: no index yet; run `python scripts/devmem.py build`", file=sys.stderr)
        return 2
    con = connect_readonly(cfg) if args.no_refresh else connect(cfg, create=False)
    try:
        if not args.no_refresh:
            kinds = stale_kinds(con, cfg)
            con.close()
            refresh(cfg, budget=args.budget, embed=False, kinds=kinds)
            con = connect(cfg, create=False)
        session = query_embedder_session(2.0) if args.mode != "bm25" else contextlib.nullcontext(None)
        with session as emb:
            ids, note = search(con, args.text, k=args.k, mode=args.mode, embedder=emb)
        rows = hits(con, ids, args.text, args.k, cfg)
        if args.json:
            print(json.dumps({"hits": rows, "dense": note.get("dense")}, ensure_ascii=False))
        else:
            print(format_hits(rows, note))
        return 0
    finally:
        con.close()


def cmd_stats(args, cfg: Config) -> int:
    if not cfg.db.exists():
        print("no index")
        return 1
    con = connect(cfg, create=False)
    q = lambda s: con.execute(s).fetchone()[0]
    kinds = dict(con.execute("SELECT kind, COUNT(*) FROM chunk GROUP BY kind").fetchall())
    ref = con.execute("SELECT v FROM meta WHERE k='refreshed'").fetchone()
    print(json.dumps({"db": str(cfg.db), "bytes": cfg.db.stat().st_size, "sources": q("SELECT COUNT(*) FROM src"),
                      "chunks": q("SELECT COUNT(*) FROM chunk"), "aliases": q("SELECT COUNT(*) FROM alias"),
                      "embedded": q("SELECT COUNT(*) FROM emb"), "chunks_by_kind": kinds,
                      "refreshed": datetime.fromtimestamp(float(ref[0])).isoformat(timespec="seconds") if ref else None},
                     indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=8)
    q.add_argument("--json", action="store_true")
    q.add_argument("--no-refresh", action="store_true")
    q.add_argument("--budget", type=float, default=8.0, help="seconds the query-time refresh may take")
    q.add_argument("--mode", default="auto", choices=["auto", "bm25", "dense", "hybrid"])
    for name in ("build", "refresh"):
        p = sub.add_parser(name)
        p.add_argument("--no-embed", action="store_true")
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    cfg = Config()
    if args.cmd == "query":
        return cmd_query(args, cfg)
    if args.cmd == "stats":
        return cmd_stats(args, cfg)
    log = lambda s: print(s, file=sys.stderr, flush=True)
    fn = build if args.cmd == "build" else refresh
    print(json.dumps(fn(cfg, embed=not args.no_embed, log=log)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
