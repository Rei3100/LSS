"""番組の候補runへ、run本体を変えずに既知録音の一致の証拠を別記録で付ける。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.bgm import identity, identity_run
from lss.core import session
from lss.validation import episode_discovery as episode
from tests.test_bgm_identity import RATE, _dialogue_and_noise, _music


def _mono(path):
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == RATE
    return data.mean(axis=1)


@pytest.fixture
def completed_run(tmp_path):
    known = _music(21, 40.0)
    other = _music(22, 40.0)
    references = []
    for name, signal in (("known", known), ("other", other)):
        path = tmp_path / f"{name}.flac"
        sf.write(path, np.stack([signal, signal], axis=1), RATE, subtype="PCM_24")
        references.append(path)

    # 80〜140秒に音量の大きい1つの候補ができ、その中の90〜110秒に既知の録音が鳴る。
    mix = _dialogue_and_noise(200.0) * 0.05
    rng = np.random.default_rng(3)
    mix[int(80 * RATE):int(140 * RATE)] += _music(99, 60.0) * 0.6 + rng.standard_normal(60 * RATE).astype(np.float32) * 0.01
    mix[int(90 * RATE):int(110 * RATE)] += known[int(5 * RATE):int(25 * RATE)] * 1.5
    source = tmp_path / "episode.wav"
    sf.write(source, np.stack([mix, mix], axis=1), RATE, subtype="FLOAT")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "schema_version": 1, "method": "windowed_rms_hysteresis", "window_frames": RATE, "hop_frames": RATE // 2,
        "onset_rms": 0.1, "offset_rms": 0.05, "bridge_gap_frames": 2 * RATE, "minimum_active_frames": 5 * RATE,
        "padding_frames": 0,
    }), encoding="utf-8")

    def separate(job, *, destination_dir):
        signal = destination_dir / "signal.flac"
        sf.write(signal, np.stack([mix, mix], axis=1), RATE, subtype="PCM_24")
        tags = FLAC(signal)
        tags["LSS_STEM"] = "Instrumental"
        tags.save()
        return {"status": "ok", "outputs": [str(signal)], "input_sha256": session.input_sha256_of(source),
                "recipe_id": "separation-r1", "runtime_id": "runtime-r1"}

    run_path = episode.run_episode_discovery(source, [recipe], tmp_path / "run", source_id="ep", model_id="m",
                                             separate=separate)
    return run_path, references


def test_identity_is_attached_as_a_separate_record_without_changing_the_run(completed_run, tmp_path):
    run_path, references = completed_run
    before = run_path.read_bytes()
    record_path = identity_run.annotate_run(run_path, references, loader=_mono, cache_dir=tmp_path / "fp")

    assert run_path.read_bytes() == before
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["query_signal"]["kind"] == "separation_artifact"
    assert {item["path"] for item in record["external_evidence_used"]} == {str(p.resolve()) for p in references}
    [activity] = record["activity_runs"]
    known = [c for c in activity["candidates"] if c["identity_state"] == "known_exact"]
    assert len(known) == 1 and known[0]["identity_evidence"][0]["reference_id"] == "known"
    assert 90.0 - 0.1 <= known[0]["start_seconds"] and known[0]["end_seconds"] <= 110.1
    assert any(d["decision"] == "pending" for d in activity["novelty_decisions"])
    assert activity["zero_reason"] is None

    # 同じrun・参照・手法なら再計算しない。参照が変われば作り直す。
    stamp = record_path.stat().st_mtime_ns
    assert identity_run.annotate_run(run_path, references, loader=_mono) == record_path
    assert record_path.stat().st_mtime_ns == stamp
    identity_run.annotate_run(run_path, references[:1], loader=_mono)
    assert json.loads(record_path.read_text(encoding="utf-8"))["intent"]["references"][0]["sha256"] \
        == identity._sha256(references[0])


def test_a_changed_signal_falls_back_to_the_verified_source_or_fails(completed_run):
    run_path, references = completed_run
    run = json.loads(run_path.read_text(encoding="utf-8"))
    artifact = Path(run["separation"]["artifacts"][0]["path"])
    artifact.write_bytes(artifact.read_bytes() + b"\0")
    record = json.loads(identity_run.annotate_run(run_path, references, loader=_mono).read_text(encoding="utf-8"))
    assert record["query_signal"]["kind"] == "source"

    source = Path(run["source"]["path"])
    source.write_bytes(source.read_bytes()[:-1])
    with pytest.raises(identity_run.IdentityRunError):
        identity_run.annotate_run(run_path, references[:1], loader=_mono)


def test_routed_episode_candidates_get_identity_evidence_on_the_rendered_signal(tmp_path):
    """方向付き分岐の未公開候補（描画済み信号の標本位置で記録）にも、runを変えずに一致の証拠を付ける。"""
    known = _music(31, 40.0)
    reference = tmp_path / "known.flac"
    sf.write(reference, np.stack([known, known], axis=1), RATE, subtype="PCM_24")
    mix = _dialogue_and_noise(120.0) * 0.05
    mix[int(40 * RATE):int(60 * RATE)] += known[int(3 * RATE):int(23 * RATE)] * 1.5
    signal = tmp_path / "routed.wav"
    sf.write(signal, np.stack([mix, mix], axis=1), RATE, subtype="FLOAT")
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    offset = (7, 1000)

    def candidate(start, end, name):
        return {"origin": {"source_sha256": identity._sha256(source), "signal_sha256": identity._sha256(signal),
                           "separation_recipe_id": "renderer-r1", "activity_recipe_id": "share-r1",
                           "start_frame": int(start * RATE), "end_frame": int(end * RATE), "sample_rate": RATE,
                           "time_offset_numerator": offset[0], "time_offset_denominator": offset[1]},
                "asset": {"candidate_asset_id": name}}

    run_path = tmp_path / "routed" / "run.json"
    run_path.parent.mkdir()
    run_path.write_text(json.dumps({
        "schema_version": 1, "status": identity_run.ROUTED_STATUS,
        "source": {"path": str(source), "sha256": identity._sha256(source), "source_id": "series/ep1"},
        "rendering": {"signal": {"path": str(signal), "sha256": identity._sha256(signal)}},
        "candidates": [candidate(30.0, 70.0, "cue-a"), candidate(90.0, 110.0, "cue-b")],
    }), encoding="utf-8")
    before = run_path.read_bytes()

    record = json.loads(identity_run.annotate_run(run_path, [reference], loader=_mono).read_text(encoding="utf-8"))

    assert run_path.read_bytes() == before
    assert record["query_signal"]["kind"] == "routed_signal"
    [activity] = record["activity_runs"]
    assert activity["recipe_id"] == "renderer-r1+share-r1"
    known_pieces = [c for c in activity["candidates"] if c["identity_state"] == "known_exact"]
    assert len(known_pieces) == 1 and known_pieces[0]["derived_from"] == "cue-a"
    assert 39.9 <= known_pieces[0]["start_seconds"] and known_pieces[0]["end_seconds"] <= 60.2  # 元メディアの時刻
    assert {d["candidate_id"] for d in activity["novelty_decisions"] if d["decision"] == "pending"} >= {"cue-b"}


@pytest.fixture
def repeated_run(tmp_path):
    """参照録音が無く、同じ録音が二度鳴る番組。"""
    cue = _music(31, 25.0)
    mix = _dialogue_and_noise(300.0, seed=5) * 0.3
    for start in (30.0, 200.0):
        begin = int(start * RATE)
        mix[begin:begin + cue.size] += cue
    source = tmp_path / "episode.wav"
    sf.write(source, np.stack([mix, mix], axis=1), RATE, subtype="FLOAT")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "schema_version": 1, "method": "windowed_rms_hysteresis", "window_frames": RATE,
        "hop_frames": RATE // 2, "onset_rms": 0.1, "offset_rms": 0.05,
        "bridge_gap_frames": 2 * RATE, "minimum_active_frames": 5 * RATE, "padding_frames": 0,
    }), encoding="utf-8")

    def separate(job, *, destination_dir):
        signal = destination_dir / "signal.flac"
        sf.write(signal, np.stack([mix, mix], axis=1), RATE, subtype="PCM_24")
        tags = FLAC(signal)
        tags["LSS_STEM"] = "Instrumental"
        tags.save()
        return {"status": "ok", "outputs": [str(signal)],
                "input_sha256": session.input_sha256_of(source),
                "recipe_id": "separation-r1", "runtime_id": "runtime-r1"}

    return episode.run_episode_discovery(source, [recipe], tmp_path / "run", source_id="ep-repeat",
                                         model_id="m", separate=separate)


def test_a_repeat_inside_one_episode_is_recorded_without_any_reference(repeated_run, tmp_path):
    record_path = identity_run.annotate_run(repeated_run, [], loader=_mono)

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["matches"] == []
    assert record["external_evidence_used"] == []
    [repeat] = [item for item in record["repeated_recordings"]
                if abs(item["first_start_seconds"] - 30.0) <= 2.0]
    assert abs(repeat["second_start_seconds"] - 200.0) <= 2.0
    [activity] = record["activity_runs"]
    evidenced = [row for row in activity["candidates"] if row.get("recurring_evidence")]
    assert len(evidenced) == 2
    assert all(row["identity_state"] == "unresolved" for row in evidenced)
    assert all(row["cue_extent"]["state"] == "single_cue" for row in evidenced)


def test_the_same_episode_is_not_matched_again_when_nothing_changed(repeated_run):
    first = identity_run.annotate_run(repeated_run, [], loader=_mono)
    stamp = first.stat().st_mtime_ns

    assert identity_run.annotate_run(repeated_run, [], loader=_mono) == first
    assert first.stat().st_mtime_ns == stamp


def test_repeats_are_reused_from_the_cache_when_only_the_references_change(repeated_run, tmp_path,
                                                                          monkeypatch):
    from lss.bgm import self_recurrence

    reference = tmp_path / "unrelated.flac"
    signal = _music(41, 30.0)
    sf.write(reference, np.stack([signal, signal], axis=1), RATE, subtype="PCM_24")
    cache = tmp_path / "cache"
    first = identity_run.annotate_run(repeated_run, [], loader=_mono, cache_dir=cache)
    before = json.loads(first.read_text(encoding="utf-8"))["repeated_recordings"]
    assert before

    monkeypatch.setattr(self_recurrence, "repeated_recordings",
                        lambda *a, **k: pytest.fail("同じ信号と条件で再演を探し直さない"))
    identity_run.annotate_run(repeated_run, [reference], loader=_mono, cache_dir=cache)

    after = json.loads(first.read_text(encoding="utf-8"))["repeated_recordings"]
    assert after == before


def test_a_changed_repeat_condition_is_not_served_from_the_cache(repeated_run, tmp_path,
                                                                 monkeypatch):
    from lss.bgm import self_recurrence

    cache = tmp_path / "cache"
    identity_run.annotate_run(repeated_run, [], loader=_mono, cache_dir=cache)
    monkeypatch.setattr(self_recurrence, "MINIMUM_SEPARATION_SECONDS",
                        self_recurrence.MINIMUM_SEPARATION_SECONDS + 1.0)
    calls = []
    original = self_recurrence.repeated_recordings
    monkeypatch.setattr(self_recurrence, "repeated_recordings",
                        lambda *a, **k: (calls.append(1), original(*a, **k))[1])

    identity_run.annotate_run(repeated_run, [], loader=_mono, cache_dir=cache)

    assert calls


def test_two_appearances_of_one_recording_are_linked_without_merging_them(repeated_run):
    record_path = identity_run.annotate_run(repeated_run, [], loader=_mono)

    record = json.loads(record_path.read_text(encoding="utf-8"))
    [activity] = record["activity_runs"]
    [group] = activity["occurrence_groups"]
    assert len(group["members"]) == 2
    assert all(item["kind"] == "shared_recording" for item in group["evidence"])
    linked = {member[1] for member in group["members"]}
    # つないだだけで、候補は残っている。
    assert linked <= {row["candidate_id"] for row in activity["candidates"]}
    assert len(activity["candidates"]) > 2


@pytest.fixture
def interrupted_cue_run(tmp_path):
    """参照録音が無く、一つの候補の中で音楽が一度止まる番組。"""
    first, second = _music(61, 25.0), _music(62, 25.0)
    mix = _dialogue_and_noise(200.0, seed=9) * 0.02
    mix[int(30 * RATE):int(30 * RATE) + first.size] += first
    # 3秒の完全な無音を挟んでから別の曲を鳴らす。候補は音量で一つにまとまる。
    mix[int(58 * RATE):int(61 * RATE)] = 0.0
    mix[int(61 * RATE):int(61 * RATE) + second.size] += second
    source = tmp_path / "episode.wav"
    sf.write(source, np.stack([mix, mix], axis=1), RATE, subtype="FLOAT")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "schema_version": 1, "method": "windowed_rms_hysteresis", "window_frames": RATE,
        "hop_frames": RATE // 2, "onset_rms": 0.05, "offset_rms": 0.02,
        "bridge_gap_frames": 10 * RATE, "minimum_active_frames": 5 * RATE, "padding_frames": 0,
    }), encoding="utf-8")

    def separate(job, *, destination_dir):
        signal = destination_dir / "signal.flac"
        sf.write(signal, np.stack([mix, mix], axis=1), RATE, subtype="PCM_24")
        tags = FLAC(signal)
        tags["LSS_STEM"] = "Instrumental"
        tags.save()
        return {"status": "ok", "outputs": [str(signal)],
                "input_sha256": session.input_sha256_of(source),
                "recipe_id": "separation-r1", "runtime_id": "runtime-r1"}

    return episode.run_episode_discovery(source, [recipe], tmp_path / "run",
                                         source_id="ep-gap", model_id="m", separate=separate)


def test_music_stopping_inside_a_candidate_says_it_holds_more_than_one_cue(interrupted_cue_run):
    record_path = identity_run.annotate_run(interrupted_cue_run, [], loader=_mono)

    record = json.loads(record_path.read_text(encoding="utf-8"))
    [activity] = record["activity_runs"]
    held = [row for row in activity["candidates"]
            if (row.get("cue_extent") or {}).get("state") == "multiple_cues"]
    assert held, "止まりを含む候補が複数曲として読めていない"
    [reading] = [row["cue_extent"] for row in held]
    reasons = [item["evidence"].get("reason") for item in reading["evidence"]["readings"]]
    assert "music_stops_inside" in reasons
    assert record["intent"]["music_gap_recipe_id"].startswith("bgm-music-gap-v1-")
