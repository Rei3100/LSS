"""The read-only SSD stage cache (scripts/stage.py): verified copy, staleness, LRU eviction, locks, storage_tiers hooks.

All on temp directories: the "HDD" is any path outside the policy's hot volume (a drive letter that does not exist),
C: free space is faked, and nothing is written outside tmp_path."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.machine_config_fixtures import machine_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage  # noqa: E402
import storage_tiers as tiers  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    free = {"gb": 500.0}
    monkeypatch.setattr(tiers, "free_gb", lambda volume: free["gb"])
    policy = {**machine_policy(tmp_path), "schema_version": 1, "hot_volume": ('Z' + chr(58) + '\\'), "local_state_root": str(tmp_path / "local"),
              "archive_root": str(tmp_path / "archive"), "receipts_root": str(tmp_path / "receipts"),
              "stage": {"root": str(tmp_path / "stage"), "max_gb": 1.0, "evict_first_below_free_gb": 100,
                        "min_free_after_gb": 30, "grace_minutes": 0}}
    source = tmp_path / "hdd"
    source.mkdir()
    return policy, source, free


def _write(path: Path, data: bytes, age_hours: float = 5.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    stamp = time.time() - age_hours * 3600
    os.utime(path, (stamp, stamp))
    return path


def test_a_file_is_copied_once_verified_and_read_only(env, monkeypatch):
    policy, source, _ = env
    original = _write(source / "episode.mkv", os.urandom(300_000))
    copies = []
    real_copy = tiers._copy_file
    monkeypatch.setattr(tiers, "_copy_file", lambda s, t: copies.append(s) or real_copy(s, t))
    first = stage.get(policy, original, purpose="probe")
    again = stage.get(policy, original)
    staged = Path(first["path"])
    assert first["staged"] and not first["hit"] and again["hit"] and again["path"] == first["path"]
    assert len(copies) == 1                                         # the second call is a hit: no copy, no hashing
    assert staged.is_relative_to(Path(policy["stage"]["root"])) and staged.name == "episode.mkv"
    assert staged.read_bytes() == original.read_bytes() and original.read_bytes()   # the HDD file is the original
    assert not os.access(staged, os.W_OK) and not (staged.stat().st_mode & stat.S_IWRITE)
    meta = json.loads((staged.parent / stage.META).read_text(encoding="utf-8"))
    assert meta["files"]["episode.mkv"]["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert meta["bytes"] == 300_000 and meta["purpose"] == "probe"
    assert stage.stage_path(original, policy=policy) == staged


def test_a_changed_source_is_a_new_entry_and_the_stale_one_goes(env):
    policy, source, _ = env
    original = _write(source / "a.wav", b"one" * 1000)
    old = Path(stage.get(policy, original)["path"])
    original.write_bytes(b"two!" * 1000)                           # new size and mtime
    new = Path(stage.get(policy, original)["path"])
    assert new != old and new.read_bytes() == b"two!" * 1000
    assert not old.exists() and [e["source"] for e in stage.entries(policy)] == [str(original)]
    os.utime(original, (time.time() + 50, time.time() + 50))      # same size, new mtime: stale as well
    assert Path(stage.get(policy, original)["path"]) != new


def test_a_directory_is_staged_file_by_file_and_verified(env, monkeypatch):
    policy, source, _ = env
    tree = source / "corpus"
    _write(tree / "a.wav", os.urandom(5000))
    _write(tree / "deep" / "b.json", b"{}")
    (tree / "empty").mkdir()
    result = stage.get(policy, tree)
    staged = Path(result["path"])
    assert staged.name == "corpus" and (staged / "deep" / "b.json").read_bytes() == b"{}" and (staged / "empty").is_dir()
    assert not os.access(staged / "a.wav", os.W_OK)
    meta = json.loads((staged.parent / stage.META).read_text(encoding="utf-8"))
    assert set(meta["files"]) == {"a.wav", os.path.join("deep", "b.json")}
    assert stage.get(policy, tree)["hit"]
    _write(tree / "deep" / "c.bin", b"new")                       # a new file changes the identity
    assert not stage.get(policy, tree)["hit"]
    junction_target = source / "elsewhere"
    junction_target.mkdir()
    tiers.make_junction(tree / "linked", junction_target)
    with pytest.raises(RuntimeError, match="link inside"):
        stage.get(policy, tree)


def test_a_failed_verification_stages_nothing(env, monkeypatch):
    policy, source, _ = env
    original = _write(source / "x.wav", os.urandom(4000))
    monkeypatch.setattr(tiers, "_hash", lambda path: "0" * 64)
    with pytest.raises(RuntimeError, match="verification failed"):
        stage.get(policy, original)
    root = Path(policy["stage"]["root"])
    assert not [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    assert stage.stage_path(original, policy=policy) == original  # falls back to the HDD file, with a warning


def test_a_hash_in_the_source_receipt_must_agree(env):
    policy, source, _ = env
    original = _write(source / "set" / "m.flac", os.urandom(2000))
    sha = hashlib.sha256(original.read_bytes()).hexdigest()
    manifest = source / "set" / tiers.MANIFEST
    manifest.write_text(json.dumps({"files": {"m.flac": {"bytes": 2000, "sha256": sha}}}), encoding="utf-8")
    assert stage.get(policy, original)["hash_source"] == "receipt+read"
    other = _write(source / "set" / "n.flac", os.urandom(2000))
    manifest.write_text(json.dumps({"files": {"n.flac": {"bytes": 2000, "sha256": "ab" * 32}}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt"):
        stage.get(policy, other)


def test_a_source_already_on_the_ssd_is_returned_unchanged(env):
    policy, source, _ = env
    original = _write(source / "local.wav", b"x" * 100)
    hot = {**policy, "hot_volume": os.path.splitdrive(str(original))[0] + "\\"}
    result = stage.get(hot, original)
    assert result["path"] == str(original) and not result["staged"]
    assert stage.get(hot, original, force=True)["staged"]


def test_prune_evicts_least_recently_used_first_and_keeps_the_recent(env):
    policy, source, _ = env
    paths = [Path(stage.get(policy, _write(source / f"f{i}.bin", bytes([i]) * 1000))["path"]) for i in range(3)]
    root = Path(policy["stage"]["root"])
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    for age, entry in zip((3000, 100, 2000), paths):               # f0 oldest, f2 older than f1
        index["entries"][entry.parent.name]["last_access"] = time.time() - age
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    stage.prune(policy, to_gb=2500 / 1e9, log=lambda *_: None)   # room for two of three 1000-byte entries
    assert [p.exists() for p in paths] == [False, True, True]
    stage.prune(policy, to_gb=1500 / 1e9, log=lambda *_: None)
    assert [p.exists() for p in paths] == [False, True, False]
    graced = {**policy, "stage": {**policy["stage"], "grace_minutes": 10}}
    stage.prune(graced, to_gb=0, log=lambda *_: None)             # f1 was used a moment ago (100 s): kept
    assert paths[1].exists()
    assert stage.prune(policy, to_gb=0, log=lambda *_: None)["remaining_bytes"] == 0


def test_an_entry_another_process_has_open_is_never_removed(env):
    policy, source, _ = env
    staged = Path(stage.get(policy, _write(source / "busy.wav", b"b" * 2000))["path"])
    with open(staged, "rb"):
        result = stage.prune(policy, to_gb=0, log=lambda *_: None)
        assert staged.exists() and result["evicted"] == 0 and result["in_use"]
    assert stage.prune(policy, to_gb=0, log=lambda *_: None)["evicted"] == 1 and not staged.exists()


def test_the_cap_is_enforced_on_get_and_an_oversized_file_is_refused(env):
    policy, source, _ = env
    policy["stage"]["max_gb"] = 2500 / 1e9
    first = Path(stage.get(policy, _write(source / "1.bin", b"1" * 1000))["path"])
    time.sleep(0.02)
    second = Path(stage.get(policy, _write(source / "2.bin", b"2" * 1000))["path"])
    third = Path(stage.get(policy, _write(source / "3.bin", b"3" * 1000))["path"])
    assert not first.exists() and second.exists() and third.exists()      # the least recently used made room
    big = _write(source / "big.bin", b"B" * 3000)
    with pytest.raises(stage.StageRefused, match="larger than the stage cap"):
        stage.get(policy, big)
    assert stage.stage_path(big, policy=policy) == big
    with pytest.raises(stage.StageRefused):
        stage.stage_path(big, policy=policy, strict=True)


def test_stage_volume_too_full_refuses_after_evicting_what_it_can(env):
    policy, source, free = env
    kept = Path(stage.get(policy, _write(source / "old.bin", b"o" * 1000))["path"])
    free["gb"] = 20.0                                                # below min_free_after_gb: nothing can help
    import re
    volume = stage.cache_volume(policy).rstrip('\\/')
    with pytest.raises(stage.StageRefused, match=re.escape(f"{volume} free")):
        stage.get(policy, _write(source / "new.bin", b"n" * 1000))
    assert not kept.exists()                                         # the cache was given back first


def test_two_processes_staging_the_same_file_copy_it_once(env, monkeypatch):
    policy, source, _ = env
    original = _write(source / "shared.mkv", os.urandom(200_000))
    copies, real_copy = [], tiers._copy_file

    def slow_copy(src, dst):
        copies.append(src)
        time.sleep(0.3)
        return real_copy(src, dst)

    monkeypatch.setattr(tiers, "_copy_file", slow_copy)
    results = []
    threads = [threading.Thread(target=lambda: results.append(stage.get(policy, original))) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(copies) == 1 and len(results) == 3
    assert sorted(r["hit"] for r in results) == [False, True, True]
    assert len({r["path"] for r in results}) == 1


def test_different_sources_cannot_spend_the_same_cache_capacity(env, monkeypatch):
    policy, source, _ = env
    policy["stage"].update(max_gb=1500 / 1e9, grace_minutes=10)
    first = _write(source / "first.bin", b"a" * 1000)
    second = _write(source / "second.bin", b"b" * 1000)
    copying, release, second_copy = threading.Event(), threading.Event(), threading.Event()
    real_copy = stage._copy_in
    outcomes = []

    def blocked_copy(src, *args):
        if src == first:
            copying.set()
            assert release.wait(10)
        else:
            second_copy.set()
        return real_copy(src, *args)

    def run(path):
        try:
            outcomes.append(stage.get(policy, path))
        except Exception as exc:
            outcomes.append(exc)

    monkeypatch.setattr(stage, "_copy_in", blocked_copy)
    workers = [threading.Thread(target=run, args=(p,)) for p in (first, second)]
    workers[0].start()
    try:
        assert copying.wait(5)
        workers[1].start()
        assert not second_copy.wait(0.5)
    finally:
        release.set()
        for worker in workers:
            if worker.ident is not None:
                worker.join(10)
    assert all(not worker.is_alive() for worker in workers)
    assert sum(isinstance(result, dict) for result in outcomes) == 1
    assert sum(isinstance(result, stage.StageRefused) for result in outcomes) == 1
    assert stage.total_bytes(policy) == 1000
    assert first.read_bytes() == b"a" * 1000 and second.read_bytes() == b"b" * 1000


def test_a_held_entry_lock_makes_a_second_stager_wait(env):
    policy, source, _ = env
    original = _write(source / "w.bin", b"w" * 500)
    st = original.stat()
    name = stage.entry_name(Path(os.path.realpath(original)), st.st_size, st.st_mtime_ns)
    root = Path(policy["stage"]["root"])
    done = threading.Event()
    with tiers.Lock(root / ".locks" / f"{name}.lock"):
        worker = threading.Thread(target=lambda: (stage.get(policy, original), done.set()))
        worker.start()
        assert not done.wait(0.6)                                    # blocked while the other holder copies
    worker.join(10)
    assert done.is_set()


def test_an_unfinished_copy_is_swept_and_never_served(env):
    policy, source, _ = env
    original = _write(source / "s.bin", b"s" * 500)
    root = Path(policy["stage"]["root"])
    st = original.stat()
    name = stage.entry_name(Path(os.path.realpath(original)), st.st_size, st.st_mtime_ns)
    _write(root / (name + stage.PARTIAL_SUFFIX) / "s.bin", b"half")
    _write(root / ("deadbeef-1-1" + stage.EVICTING_SUFFIX) / "x", b"x")
    assert Path(stage.get(policy, original)["path"]).read_bytes() == b"s" * 500
    stage.prune(policy, log=lambda *_: None)
    assert not (root / (name + stage.PARTIAL_SUFFIX)).exists() and not (root / ("deadbeef-1-1" + stage.EVICTING_SUFFIX)).exists()


def test_storage_tiers_gives_the_stage_back_first_and_never_demotes_it(env, monkeypatch, tmp_path):
    policy, source, free = env
    staged = Path(stage.get(policy, _write(source / "cold.bin", b"c" * 1000))["path"])
    free["gb"] = 50.0                                               # < evict_first_below_free_gb (100)
    assert stage.pressure(policy)
    assert tiers.stage_pressure(policy)
    order = []
    monkeypatch.setattr(tiers, "stage_relief", lambda *a, **k: order.append("stage"))
    monkeypatch.setattr(tiers, "clean_hot", lambda *a, **k: order.append("clean_hot"))
    monkeypatch.setattr(tiers, "demote", lambda *a, **k: order.append("demote"))
    monkeypatch.setattr(tiers, "idle_worktrees", lambda *a, **k: [])
    policy.update(start_free_gb=100, target_free_gb=150, prune_roots=None)
    tiers.maintain(policy, tmp_path)
    assert order == ["stage", "clean_hot", "demote"]
    monkeypatch.undo()
    monkeypatch.setattr(tiers, "free_gb", lambda volume: free["gb"])
    assert tiers.stage_relief(policy, log=lambda *_: None)["evicted"] == 1 and not staged.exists()
    other = Path(stage.get(policy, _write(source / "again.bin", b"a" * 1000))["path"]).parent
    with pytest.raises(RuntimeError, match="never demoted"):
        tiers.demote_one(policy, other)
    shipped = tiers.load_policy()
    assert shipped["_machine_overlay"] is None and shipped["stage"]["max_gb"] > 0
    with pytest.raises(RuntimeError, match="no machine overlay"):
        tiers.demote_one(shipped, other)


def test_a_prune_between_the_hit_check_and_the_touch_never_returns_a_vanished_path(env, monkeypatch):
    policy, source, _ = env
    original = _write(source / "race.bin", b"r" * 1000)
    first = Path(stage.get(policy, original)["path"])
    real_touch, raced = stage._touch, []

    def evicting_touch(root, name, **fields):
        if not raced:
            raced.append(stage._evict(policy, root / name))
        return real_touch(root, name, **fields)

    monkeypatch.setattr(stage, "_touch", evicting_touch)
    result = stage.get(policy, original)
    assert raced == [True] and Path(result["path"]).read_bytes() == b"r" * 1000 and not result["hit"]
    assert first.exists()                                           # restaged under the same name


def test_a_locked_entry_during_relief_does_not_abort_the_scheduled_pass(env, monkeypatch, tmp_path):
    policy, source, free = env
    staged = Path(stage.get(policy, _write(source / "held.bin", b"h" * 1000))["path"])
    free["gb"] = 50.0
    root = Path(policy["stage"]["root"])
    order = []
    monkeypatch.setattr(stage, "_locked", lambda *a, **k: (_ for _ in ()).throw(stage.StageRefused("lock held")))
    result = stage.prune_for_free(policy, log=lambda *_: None)      # the held entry is skipped, not raised
    assert staged.exists() and result["in_use"]
    monkeypatch.setattr(tiers, "stage_relief", lambda *a, **k: (_ for _ in ()).throw(stage.StageRefused("boom")))
    monkeypatch.setattr(tiers, "clean_hot", lambda *a, **k: order.append("clean_hot"))
    monkeypatch.setattr(tiers, "demote", lambda *a, **k: order.append("demote"))
    monkeypatch.setattr(tiers, "idle_worktrees", lambda *a, **k: [])
    policy.update(start_free_gb=100, target_free_gb=150, prune_roots=None)
    messages = []
    tiers.maintain(policy, tmp_path, log=messages.append)
    assert order == ["clean_hot", "demote"] and any("stage cache relief skipped" in m for m in messages)
    assert root.is_dir()


def test_removing_a_file_from_a_directory_is_a_change_even_when_size_and_mtime_agree(env):
    policy, source, _ = env
    tree = source / "set"
    _write(tree / "a.wav", b"a" * 100, age_hours=3)
    _write(tree / "b.wav", b"b" * 100, age_hours=3)
    os.utime(tree, (time.time() - 3 * 3600,) * 2)
    first = stage.get(policy, tree)
    os.unlink(tree / "b.wav")
    _write(tree / "a.wav", b"A" * 200, age_hours=3)                 # same total bytes, same newest mtime, one file fewer
    os.utime(tree, (time.time() - 3 * 3600,) * 2)
    second = stage.get(policy, tree)
    assert second["entry"] != first["entry"] and not second["hit"]
    assert (Path(second["path"]) / "a.wav").read_bytes() == b"A" * 200 and not (Path(first["path"])).exists()
