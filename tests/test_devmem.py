"""scripts/devmem.py on a tiny temporary corpus: chunking, git + file + docstring sources, BM25 / trigram / dense
fusion, incremental refresh (new commit, changed file, deleted file), duplicate aliases, output budget and the
BM25 fallback note when no embedding server answers."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "devmem.py"


@pytest.fixture()
def dm():
    spec = importlib.util.spec_from_file_location("devmem_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_default_memory_has_no_implicit_hdd_archive_roots(dm):
    assert dm.Config().dev_roots == []


def test_non_hot_notes_are_rejected_before_tree_walk(dm, tmp_path, monkeypatch):
    import dev_hot_paths
    def refuse(path, label, *args, **kwargs):
        raise dev_hot_paths.HotPathError({"label": label, "volume": "HDD", "resolved": str(path), "hot_volume": "SSD"})
    monkeypatch.setattr(dev_hot_paths, "require_hot", refuse)
    monkeypatch.setattr(dm, "_walk", lambda *a, **kw: pytest.fail("must not walk an HDD tree"))
    assert list(dm.dev_files(dm.Config(repo=tmp_path, dev_roots=[tmp_path / "archive"]))) == []


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          capture_output=True, text=True, check=True).stdout


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class FakeEmbedder:
    """Bag-of-words hashing into 64 dimensions: deterministic, shares vocabulary like a real embedding would."""
    dim = 2048

    def _vec(self, text: str):
        import numpy as np
        v = np.zeros(self.dim, dtype=np.float32)
        for w in text.lower().replace("\n", " ").split():
            v[zlib.crc32(w.strip(".,:;`()").encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts):
        import numpy as np
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, q):
        return self._vec(q)


@pytest.fixture()
def corpus(tmp_path, dm, monkeypatch):
    monkeypatch.setattr(dm, "dev_evidence_root", lambda checkout_root: checkout_root)
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    write(repo / "docs" / "dev" / "NOTES.md", "# Notes\n\n## Storage\nWork on the SSD, keep results on the HDD tier.\n\n"
                                               "## Separator\nThe frobnicator separator was rejected: clean music lost 6 dB.\n")
    write(repo / "docs" / "jp.md", "# 日本語メモ\n\nオルゴールの曲は安全保留になった。チャイムが自分の音楽を事象と数えたため。\n")
    write(repo / "release-evidence" / "run1" / "RESULTS.md", "# Run 1\nThe quuxify probe kept 3 of 7 cues.\n")
    write(repo / "release-evidence" / "run1" / "data.json", json.dumps({"key": "zorblax", "n": 1}))
    write(repo / "scripts" / "tool.py", '"""Tool that widgetizes spectra.\n\nUses the gizmo cache.\n"""\nX = 1\n')
    write(repo / "tests" / "test_x.py", '"""not indexed docstring wibblewobble"""\n')
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "First: add notes", "-m", "Body explains why the hyperwidget gate was added.")
    sha1 = git(repo, "rev-parse", "HEAD").strip()
    dev = tmp_path / "dev" / "lane1"
    write(dev / "REPORT.md", "# Lane report\nThe snorkeling experiment failed because of tide.\n")
    write(dev / "notes.txt", "ignored: not md and not named\n")
    write(dev / "VERDICT-X.json", json.dumps({"verdict": "PASS_WITH_LIMITS"}))
    write(dev / "huge.md", "x" * 1_200_000)
    wt = repo / ".worktrees" / "wt1"
    write(wt / "docs" / "dev" / "NOTES.md", (repo / "docs" / "dev" / "NOTES.md").read_text(encoding="utf-8"))
    write(wt / "node_modules" / "pkg" / "README.md", "# skip me\n")
    cfg = dm.Config(repo=repo, db_dir=tmp_path / "db", dev_roots=[tmp_path / "dev"])
    return cfg, repo, sha1


def ranked(dm, cfg, q, mode="bm25", k=8, embedder=None):
    con = dm.connect(cfg, create=False)
    ids, note = dm.search(con, q, k=k, mode=mode, embedder=embedder)
    return dm.hits(con, ids, q, k, cfg), note, con


def test_chunk_text_splits_on_headings_and_keeps_line_ranges(dm):
    text = "# A\n" + "para one\n" * 5 + "\n## B\n" + "word " * 400 + "\n\nend\n"
    parts = dm.chunk_text(text, limit=500, floor=50)
    assert len(parts) >= 3
    assert parts[0][0] == 1 and parts[0][2] == "A"
    assert any(h == "A > B" for _, _, h, _ in parts)
    assert all(len(t) <= 520 for *_, t in parts)
    lines = text.splitlines()
    for a, b, _, t in parts:
        assert 1 <= a <= b <= len(lines)


def test_docstring_chunks(dm):
    assert dm.docstring_chunks('"""Title\n\nbody"""\nX = 1\n')[0][3].startswith("Title")
    assert dm.docstring_chunks("X = 1\n") == []
    assert dm.docstring_chunks("def broken(:\n") == []


def test_build_indexes_each_source_kind_and_skips_the_rest(dm, corpus):
    cfg, repo, sha1 = corpus
    out = dm.build(cfg, embed=False)
    assert out["git_new"] == 1 and out["files"]["complete"]
    con = dm.connect(cfg, create=False)
    paths = {r[0] for r in con.execute("SELECT path FROM chunk")}
    paths |= {r[0] for r in con.execute("SELECT path FROM alias")}
    joined = "\n".join(paths)
    assert "docs/dev/NOTES.md" in paths and "scripts/tool.py" in paths and "release-evidence/run1/data.json" in paths
    assert "REPORT.md" in joined and "VERDICT-X.json" in joined
    assert "huge.md" not in joined and "notes.txt" not in joined and "node_modules" not in joined
    assert "test_x.py" not in joined
    kinds = {r[0] for r in con.execute("SELECT kind FROM chunk")}
    assert kinds >= {"git", "repo", "py", "dev"}


def test_exact_identifier_natural_text_and_japanese_queries(dm, corpus):
    cfg, repo, sha1 = corpus
    dm.build(cfg, embed=False)
    rows, _, _ = ranked(dm, cfg, "zorblax")
    assert rows[0]["path"] == "release-evidence/run1/data.json"
    rows, _, _ = ranked(dm, cfg, "why was the hyperwidget gate added")
    assert rows[0]["kind"] == "git" and rows[0]["commit"] == sha1
    rows, _, _ = ranked(dm, cfg, sha1[:8])
    assert rows and rows[0]["commit"] == sha1
    rows, _, _ = ranked(dm, cfg, "frobnicator separator rejected", k=3)
    assert rows[0]["path"] == "docs/dev/NOTES.md" and rows[0]["lines"][0] <= 6
    rows, _, _ = ranked(dm, cfg, "オルゴールの曲はなぜ保留になった", mode="auto")
    assert rows[0]["path"] == "docs/jp.md"
    rows, _, _ = ranked(dm, cfg, "no such thing xyzzyplugh")
    assert rows == []


def test_promoted_commit_alias_keeps_its_sha(dm, corpus):
    cfg, _, _ = corpus
    con = dm.connect(cfg)
    chunks = [(1, 1, "heading", "identical commit content")]
    dm._add_source(con, "git:first", "git", 0, 0, chunks, "git:first", "today", "first")
    dm._add_source(con, "git:second", "git", 0, 0, chunks, "git:second", "today", "second")
    sid = con.execute("SELECT id FROM src WHERE key='git:first'").fetchone()[0]
    dm._drop_source(con, sid)
    assert con.execute("SELECT sha FROM chunk").fetchone()[0] == "second"
    con.close()


def test_deleted_source_cannot_promote_its_own_duplicate(dm, corpus):
    cfg, _, _ = corpus
    con = dm.connect(cfg)
    chunks = [(1, 1, "heading", "repeated document content")] * 2
    dm._add_source(con, "repeat", "repo", 0, 0, chunks, "docs/repeat.md", "today")
    sid = con.execute("SELECT id FROM src").fetchone()[0]
    dm._drop_source(con, sid)
    assert con.execute("SELECT count(*) FROM chunk").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM alias").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM fts WHERE fts MATCH 'repeated'").fetchone()[0] == 0
    con.close()


def test_duplicate_chunks_are_stored_once_with_aliases(dm, corpus):
    cfg, repo, _ = corpus
    dm.build(cfg, embed=False)
    con = dm.connect(cfg, create=False)
    n = con.execute("SELECT COUNT(*) FROM chunk WHERE text LIKE '%frobnicator%'").fetchone()[0]
    assert n == 1
    assert con.execute("SELECT COUNT(*) FROM alias").fetchone()[0] >= 1
    rows, _, _ = ranked(dm, cfg, "frobnicator")
    assert rows[0]["path"] == "docs/dev/NOTES.md" and rows[0]["copies"] >= 1  # repo copy wins over the worktree copy


def test_refresh_picks_up_new_commit_changed_and_deleted_files(dm, corpus):
    cfg, repo, _ = corpus
    dm.build(cfg, embed=False)
    again = dm.refresh(cfg, embed=False)
    assert again["git_new"] == 0 and again["files"]["changed"] == 0
    write(repo / "docs" / "dev" / "NOTES.md", "# Notes\n\nA brand new paragraph about the plughole filter.\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "Second: change notes", "-m", "Mentions the narwhal calibration.")
    out = dm.refresh(cfg, embed=False)
    assert out["git_new"] == 1 and out["files"]["changed"] >= 1
    rows, _, _ = ranked(dm, cfg, "plughole")
    assert rows[0]["path"] == "docs/dev/NOTES.md"
    rows, _, _ = ranked(dm, cfg, "narwhal calibration")
    assert rows[0]["kind"] == "git"
    rows, _, _ = ranked(dm, cfg, "frobnicator")  # the old text is gone from the repo copy but lives on in the worktree copy
    assert rows and "wt1" in rows[0]["path"]
    (repo / "docs" / "jp.md").unlink()
    out = dm.refresh(cfg, embed=False)
    assert out["files"]["removed"] == 1
    rows, _, _ = ranked(dm, cfg, "オルゴールの曲はなぜ保留になった", mode="auto")
    assert not rows or rows[0]["path"] != "docs/jp.md"


def test_dense_and_hybrid_with_a_fake_embedder(dm, corpus):
    pytest.importorskip("numpy")
    cfg, repo, _ = corpus
    dm.build(cfg, embed=False)
    emb = FakeEmbedder()
    con = dm.connect(cfg)
    n = dm.embed_missing(con, embedder=emb)
    assert n == con.execute("SELECT COUNT(*) FROM chunk").fetchone()[0] > 0
    assert dm.embed_missing(con, embedder=emb) == 0  # nothing left to embed
    con.close()
    rows, note, _ = ranked(dm, cfg, "storage SSD HDD tier results", mode="auto", embedder=emb)
    assert note["dense"] is True and rows[0]["path"] == "docs/dev/NOTES.md"
    rows, note, _ = ranked(dm, cfg, "snorkeling tide", mode="dense", embedder=emb)
    assert "REPORT.md" in rows[0]["path"]


def test_without_an_embedding_server_it_falls_back_to_bm25_and_says_so(dm, corpus, monkeypatch, capsys):
    cfg, repo, _ = corpus
    dm.build(cfg, embed=False)
    monkeypatch.delenv(dm.EMBED_URL_ENV, raising=False)
    monkeypatch.setattr(dm, "query_embedder_session", lambda wait: dm.contextlib.nullcontext(None))
    monkeypatch.setattr(dm, "Config", lambda: cfg)
    assert dm.main(["query", "zorblax", "--no-refresh"]) == 0
    out = capsys.readouterr().out
    assert "release-evidence/run1/data.json" in out
    assert "BM25 only" in out


def test_output_is_compact_and_json_has_the_fields_an_agent_needs(dm, corpus):
    cfg, repo, _ = corpus
    for i in range(30):  # many near-identical long chunks
        write(repo / "docs" / f"big{i}.md", f"# Big {i}\n" + ("separator " * 30 + "\n") * 6)
    dm.build(cfg, embed=False)
    rows, note, _ = ranked(dm, cfg, "separator", k=20)
    text = dm.format_hits(rows, note)
    assert len(text.encode("utf-8")) <= 1700
    assert text.count("\n1. ") + text.startswith("1. ") == 1
    con = dm.connect(cfg, create=False)
    ids, note = dm.search(con, "separator", k=3, mode="bm25")
    r = dm.hits(con, ids, "separator", 3, cfg)[0]
    assert set(r) >= {"kind", "path", "lines", "date", "commit", "snippet", "copies"}
    json.dumps(r)


def test_query_time_refresh_is_bounded_and_skips_a_recent_dev_scan(dm, corpus):
    cfg, repo, _ = corpus
    dm.build(cfg, embed=False)
    con = dm.connect(cfg, create=False)
    assert dm.stale_kinds(con, cfg) == {"repo", "py"}  # dev roots scanned during build, so only the fast roots
    con.execute("UPDATE meta SET v='0' WHERE k='dev_scan'")
    con.commit()
    assert "dev" in dm.stale_kinds(con, cfg)
    con.close()
    out = dm.refresh(cfg, budget=0.0001, embed=False, kinds={"repo", "py", "dev", "worktree"})
    assert "seconds" in out  # an exhausted budget returns instead of hanging
