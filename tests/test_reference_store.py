"""参照の知識（軽い・残す）と重い参照音源の寿命（取得・保持・退役・取り直し）、作品の参照の発見・確認。

外部サービスへは接続しない（取得は偽の取得手段、または `tests/test_acquisition` のローカルHTTP＋実yt-dlp）。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.bgm import identity, reference_acquisition as ra, reference_relations as rr, reference_store as rs
from tests.test_acquisition import NO_WAIT, FakeBackend, serve  # noqa: F401 - fixture
from tests.test_bgm_identity import _music

LOW = identity.SAMPLE_RATE


@pytest.fixture
def state(tmp_path, monkeypatch):
    root = tmp_path / "state"
    monkeypatch.setenv("LSS_STATE_DIR", str(root))
    monkeypatch.setenv("LSS_ACQUISITION_DIR", str(tmp_path / "Downloads"))
    return root


def _track(path: Path, seed: int, seconds: float = 40.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, _music(seed, seconds), LOW, subtype="PCM_16")
    return path


def _cache(state: Path) -> Path:
    return state / "bgm-reference-fingerprints"


def _acquire(state, tmp_path, ids=("t1",), seeds=(1,), work="work-a"):
    media = {source_id: _track(tmp_path / "remote" / f"{source_id}.flac", seed) for source_id, seed in zip(ids, seeds)}
    backend = FakeBackend(media)
    results = [ra.acquire_into_store(f"https://example.test/{source_id}", cache_dir=_cache(state), state_root=state,
                                     work_keys=(work,), backend=backend, pacing=NO_WAIT) for source_id in ids]
    return backend, [sha for r in results for sha in r["ingested"]]


def _age(state, sha, days):
    record = rs.load(sha, state)
    record["media"]["last_used_at"] = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rs._write(rs._record_path(sha, state), record)


def test_an_acquired_reference_keeps_compact_knowledge_and_is_regenerable(state, tmp_path):
    _, [sha] = _acquire(state, tmp_path)
    record = rs.load(sha, state)
    assert record["origin"] == rs.ORIGIN_ACQUIRED and record["work_keys"] == ["work-a"]
    assert record["media"]["class"] == rs.CLASS_REGENERABLE and record["media"]["state"] == rs.STATE_PRESENT
    assert Path(record["media"]["path"]).is_relative_to(rs.media_root(state).resolve())
    assert record["reacquire"]["url"] == "https://example.test/t1"
    assert record["fingerprint"]["settings_id"] == identity.fingerprint_settings_id()
    assert (_cache(state) / record["fingerprint"]["file"]).is_file()
    assert (_cache(state) / record["chroma"]["file"]).is_file()
    # 知識は重い音源よりずっと小さい（cookie・署名つきURLは持たない）。
    text = json.dumps(record, ensure_ascii=False)
    assert "SECRET" not in text and "sig=" not in text
    assert rs.records(state, work_key="work-a")[0]["sha256"] == sha
    assert rs.records(state, work_key="other") == []


def test_idle_regenerable_media_is_retired_and_reacquired_identically_when_needed(state, tmp_path):
    backend, [sha] = _acquire(state, tmp_path)
    path = Path(rs.load(sha, state)["media"]["path"])
    recent = rs.retire_idle(cache_dir=_cache(state), state_root=state)
    assert [item["decision"] for item in recent["items"]] == ["keep_recently_used"] and path.is_file()

    _age(state, sha, 8)
    planned = rs.retire_idle(cache_dir=_cache(state), state_root=state, dry_run=True)
    assert planned["items"][0]["decision"] == "would_retire" and path.is_file()
    report = rs.retire_idle(cache_dir=_cache(state), state_root=state)
    assert report["retired"] == [sha] and report["reclaimed_bytes"] == path.stat().st_size if path.exists() else True
    assert not path.exists()
    record = rs.load(sha, state)
    assert record["media"]["state"] == rs.STATE_RETIRED and record["media"]["path"] is None
    assert (_cache(state) / record["fingerprint"]["file"]).is_file()  # 知識は残る

    # 照合の索引は知識だけで作れる（重い音源を読まない）。
    calls = []
    resolver = rs.MediaResolver(state_root=state, acquire_options={"backend": backend, "pacing": NO_WAIT},
                                fetched=calls)
    library, skipped = rs.build_library([record], cache_dir=_cache(state), state_root=state, resolver=resolver)
    assert skipped == [] and calls == [] and library.index.references[0].sha256 == sha
    # 波形が要る時だけ取り直し、同じ内容か確かめて使う。
    samples = library.samples(sha)
    assert samples.size > 0 and calls == [sha] and backend.downloads.count("t1") == 2
    record = rs.load(sha, state)
    assert record["media"]["state"] == rs.STATE_PRESENT and Path(record["media"]["path"]).is_file()
    assert [e["event"] for e in record["history"]][-1] == "reacquired"


def test_media_in_use_user_media_and_media_outside_the_product_folder_are_never_retired(state, tmp_path):
    _, [sha] = _acquire(state, tmp_path)
    _age(state, sha, 30)
    with rs.lease([sha], state):
        report = rs.retire_idle(cache_dir=_cache(state), state_root=state)
    assert report["items"][0]["decision"] == "keep_in_use_or_io_blocked"
    assert Path(rs.load(sha, state)["media"]["path"]).is_file()

    user = rs.ingest(_track(tmp_path / "References" / "mine.flac", 7), cache_dir=_cache(state), state_root=state,
                     origin=rs.ORIGIN_USER)
    outside = rs.ingest(_track(tmp_path / "Downloads" / "x.flac", 8), cache_dir=_cache(state), state_root=state,
                        acquisition={"logical_source": "example:x", "reacquire": {"url": "https://example.test/x"}})
    for record in (user, outside):
        _age(state, record["sha256"], 30)
    decisions = {item["sha256"]: item["decision"]
                 for item in rs.retire_idle(cache_dir=_cache(state), state_root=state)["items"]}
    assert decisions[user["sha256"]] == "keep_user_media"
    assert decisions[outside["sha256"]] == "keep_no_reacquisition_path"
    assert outside["media"]["class"] == rs.CLASS_RETAINED
    assert (tmp_path / "References" / "mine.flac").is_file() and (tmp_path / "Downloads" / "x.flac").is_file()


def test_an_interrupted_retirement_is_finished_next_time(state, tmp_path):
    _, [sha] = _acquire(state, tmp_path)
    record = rs.load(sha, state)
    path = Path(record["media"]["path"])
    record["media"]["state"] = rs.STATE_RETIRING   # 消す前に止まった
    rs._write(rs._record_path(sha, state), record)
    report = rs.retire_idle(cache_dir=_cache(state), state_root=state)
    assert report["retired"] == [sha] and not path.exists()

    _, [other] = _acquire(state, tmp_path, ids=("t2",), seeds=(2,))
    record = rs.load(other, state)
    Path(record["media"]["path"]).unlink()           # 消した後、退役済みを書く前に止まった
    record["media"]["state"] = rs.STATE_RETIRING
    rs._write(rs._record_path(other, state), record)
    report = rs.retire_idle(cache_dir=_cache(state), state_root=state)
    assert other in report["retired"] and rs.load(other, state)["media"]["state"] == rs.STATE_RETIRED


def test_changed_content_is_not_retired_and_a_different_reacquisition_is_not_used(state, tmp_path):
    backend, [sha] = _acquire(state, tmp_path)
    record = rs.load(sha, state)
    path = Path(record["media"]["path"])
    _age(state, sha, 30)
    original = path.read_bytes()
    path.write_bytes(original[:-10] + b"0123456789")  # 記録と違う内容
    assert rs.retire_idle(cache_dir=_cache(state), state_root=state)["items"][0]["decision"] == "keep_unsafe"
    assert path.is_file()
    path.write_bytes(original)
    rs.retire_idle(cache_dir=_cache(state), state_root=state)
    assert not path.exists()
    _track(tmp_path / "remote" / "t1.flac", 99)       # 配信側の内容が変わった
    with pytest.raises(rs.ReferenceMediaUnavailable):
        rs.ensure_media(sha, state_root=state, backend=backend, pacing=NO_WAIT)
    assert rs.load(sha, state)["history"][-1]["event"] == "reacquired_different_content"


def test_a_retired_reference_that_cannot_be_reacquired_never_makes_an_exact_decision(state, tmp_path):
    _, [sha] = _acquire(state, tmp_path)
    _age(state, sha, 30)
    rs.retire_idle(cache_dir=_cache(state), state_root=state)
    record = rs.load(sha, state)
    resolver = rs.MediaResolver(state_root=state, reacquire=False, strict=False)
    library, _ = rs.build_library([record], cache_dir=_cache(state), state_root=state, resolver=resolver)
    music = _music(1, 40.0)
    episode = np.zeros(int(60 * LOW), np.float32)
    episode[int(10 * LOW):int(40 * LOW)] = music[int(5 * LOW):int(35 * LOW)]
    matches = identity.match(episode, library.index, collapse=False)
    assert matches and all(m.identity_state != "known_exact" for m in matches)  # 指紋の候補まで
    assert sha in resolver.unavailable
    segment = {"start_seconds": 10.0, "end_seconds": 40.0, "core_seconds": [10.0, 40.0],
               "start_edge": {"kind": "silence"}, "end_edge": {"kind": "silence"}, "candidates": [[10, 40]]}
    [piece] = rr.relate_segments([segment], query=episode, library=library, matches=matches, music=episode)
    assert piece["reference_relation"]["state"] == rr.UNKNOWN
    assert piece["reference_relation"]["basis"] == "reference_waveform_unavailable"


# --- 作品の参照の発見・確認 ---------------------------------------------------------

def test_work_titles_come_from_episode_names():
    assert ra.work_title("『光の戦士テノ』第10話「合宿へ出発！」-公式配信-") == "光の戦士テノ"
    assert ra.work_title("見逃し配信【角醒ハンター オメガベル】 『第七話　打ち破れ！』[公式]") == "角醒ハンター オメガベル"
    assert ra.work_title("【公式】アニメ「星空冒険団」第1話「はじまりの灯り 前編」") == "星空冒険団"
    assert ra.work_key("光の戦士 テノ") == ra.work_key("光の戦士テノ")


# 発見（曲を単位にした発見・確認・収録盤・記憶）の試験は tests/test_reference_discovery.py。


def test_preparation_is_off_in_the_development_tree_unless_asked(monkeypatch, tmp_path):
    monkeypatch.delenv(ra.DISCOVERY_ENV, raising=False)
    prep = ra.Preparation(tmp_path / "x.wav", "a" * 64, tmp_path / "fp", tmp_path / "state").start()
    assert prep.wait_ready(0.1)["state"] == "disabled"
    monkeypatch.setenv(ra.DISCOVERY_ENV, "0")
    assert ra.enabled() is False
    monkeypatch.setenv(ra.DISCOVERY_ENV, "1")
    assert ra.enabled() is True


def test_real_ytdlp_acquires_a_reference_into_the_product_store_and_reacquires_it(state, tmp_path, serve):
    """実物のyt-dlp（ローカルHTTPの直リンク）で、参照音源の取得・退役・取り直しを通す。"""
    from tests.test_acquisition import YtDlpBackend

    remote = _track(tmp_path / "served" / "track.flac", 11)
    url, _handler = serve(remote.read_bytes(), "track.flac", "audio/flac")
    result = ra.acquire_into_store(url, cache_dir=_cache(state), state_root=state, work_keys=("work-x",),
                                   backend=YtDlpBackend(), pacing=NO_WAIT)
    [sha] = result["ingested"]
    record = rs.load(sha, state)
    assert record["media"]["class"] == rs.CLASS_REGENERABLE
    assert record["provenance"]["tool"]["name"] == "yt-dlp"
    _age(state, sha, 30)
    assert rs.retire_idle(cache_dir=_cache(state), state_root=state)["retired"] == [sha]
    path = rs.ensure_media(sha, state_root=state, backend=YtDlpBackend(), pacing=NO_WAIT)
    assert rs._file_sha256(path) == sha
