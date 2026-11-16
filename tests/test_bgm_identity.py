"""既知録音の一致を、正の証拠としてだけ返すことを合成信号で確かめる。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lss.bgm import identity

RATE = identity.SAMPLE_RATE


def _music(seed: int, seconds: float, *, transpose: float = 0.0, tempo: float = 1.0,
           harmonics=(1.0, 0.5, 0.25), performance: int | None = None) -> np.ndarray:
    """音符の並びによる合成の「曲」。seedが同じなら同じ旋律。

    `performance` を与えると、同じ譜面を別に演奏した録音（位相と発音時刻の揺れが違う）になる。
    """
    rng = np.random.default_rng(seed)
    player = np.random.default_rng(performance) if performance is not None else None
    note = int(0.25 * RATE / tempo)
    count = int(seconds * RATE) // note + 1
    pitches = rng.integers(48, 84, size=(count, 3))
    out = np.zeros(count * note, np.float32)
    t = np.arange(note) / RATE
    envelope = np.minimum(1.0, t / 0.01) * np.exp(-3 * t)
    for index in range(count):
        chunk = np.zeros(note)
        delay = 0.0 if player is None else player.uniform(0, 0.01)
        for midi in pitches[index]:
            frequency = 440.0 * 2 ** ((midi + transpose - 69) / 12)
            for order, gain in enumerate(harmonics, start=1):
                phase = 0.0 if player is None else player.uniform(0, 2 * np.pi)
                if frequency * order < RATE / 2:
                    chunk += gain * np.sin(2 * np.pi * frequency * order * np.maximum(t - delay, 0) + phase)
        out[index * note:(index + 1) * note] = chunk * envelope
    out = out[:int(seconds * RATE)]
    return (out / (np.max(np.abs(out)) + 1e-9) * 0.5).astype(np.float32)


def _dialogue_and_noise(seconds: float, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = int(seconds * RATE)
    noise = np.cumsum(rng.standard_normal(size)).astype(np.float32)
    noise -= np.convolve(noise, np.ones(64) / 64, mode="same")
    noise *= 0.05 / (np.std(noise) + 1e-9)
    voice = np.zeros(size, np.float32)
    position = 0
    while position < size:
        length = int(rng.uniform(0.2, 0.6) * RATE)
        pitch = rng.uniform(100, 250)
        t = np.arange(min(length, size - position)) / RATE
        burst = sum(np.sin(2 * np.pi * pitch * k * t) / k for k in range(1, 12))
        voice[position:position + t.size] = 0.3 * burst * np.hanning(t.size)
        position += length + int(rng.uniform(0.05, 0.4) * RATE)
    return noise + voice


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


@pytest.fixture
def library(tmp_path):
    signals = {name: _music(seed, 40.0) for name, seed in (("A", 1), ("B", 2), ("C", 3))}
    paths = []
    for name in signals:
        path = tmp_path / f"{name}.flac"
        path.write_bytes(name.encode())
        paths.append(path)
    index = identity.ReferenceIndex.build(paths, loader=lambda p: signals[Path(p).stem])
    return index, signals


# 合成の台詞は倍音が密で実音声より厳しい。実番組での感度は固定ケースで別に測る。
@pytest.mark.parametrize("music_db", [0.0, -6.0])
def test_inserted_recording_is_found_at_its_time_and_offset(library, music_db):
    index, signals = library
    background = _dialogue_and_noise(60.0)
    clip = signals["A"][int(5 * RATE):int(17 * RATE)]
    gain = _rms(background) / _rms(clip) * 10 ** (music_db / 20)
    query = background.copy()
    query[int(20 * RATE):int(20 * RATE) + clip.size] += clip * gain

    matches = identity.match(query, index)
    exact = [m for m in matches if m.identity_state == "known_exact"]
    assert exact, matches
    assert {m.reference_id for m in exact} == {"A"}
    best = max(exact, key=lambda m: m.z)
    assert best.offset_seconds == pytest.approx(-15.0, abs=0.05)
    # 波形で裏付けられた範囲だけを返す。実際の区間をはみ出さず、半分以上を覆う。
    assert 20.0 - 0.05 <= best.query_start_seconds and best.query_end_seconds <= 32.0 + 0.05
    assert best.query_end_seconds - best.query_start_seconds >= 6.0
    assert best.reference_start_seconds == pytest.approx(best.query_start_seconds - 15.0, abs=0.05)


def test_no_match_is_not_absence_and_unrelated_audio_gives_no_exact_identity(library):
    index, _ = library
    matches = identity.match(_dialogue_and_noise(60.0, seed=7), index)
    assert not [m for m in matches if m.identity_state == "known_exact"]
    # 未知の曲（索引にない録音）が鳴っていても、既知曲だと主張しない。
    unknown = _dialogue_and_noise(40.0) + _music(42, 40.0)
    assert not [m for m in identity.match(unknown, index)
                if m.identity_state == "known_exact"]


@pytest.mark.parametrize("variant", [{"transpose": 2.0}, {"tempo": 1.06}, {"performance": 5}])
def test_a_different_arrangement_of_the_same_melody_is_not_the_same_recording(library, variant):
    index, _ = library
    arranged = _music(1, 40.0, **variant)
    matches = identity.match(_dialogue_and_noise(40.0) * 0.2 + arranged, index)
    # 別の演奏はピーク位置がほぼ同じで指紋は一致し得るが、波形は説明できないので同一録音にしない。
    assert not [m for m in matches if m.identity_state == "known_exact"]


def _exact(reference: str, start: float, end: float, *, sha: str | None = None) -> identity.RecordingMatch:
    return identity.RecordingMatch(
        reference, f"{reference}.flac", sha or reference * 64, start, end, 10.0, 10.0 - start, 100, 0.9, 0.5, 0.01,
        40.0, 3, 0.0, 0.8, 0.02, 5, 0, 0, "known_exact")


def test_candidates_gain_identity_only_when_an_exact_match_covers_them():
    candidates = [
        {"candidate_id": "c1", "start_seconds": 100.0, "end_seconds": 120.0, "identity_state": "unresolved",
         "audio_version_id": None},
        {"candidate_id": "c2", "start_seconds": 200.0, "end_seconds": 240.0, "identity_state": "unresolved",
         "audio_version_id": None},
        {"candidate_id": "c3", "start_seconds": 300.0, "end_seconds": 320.0, "identity_state": "unresolved",
         "audio_version_id": None},
    ]
    unresolved = replace_state(_exact("C", 300.0, 320.0), "unresolved")
    matches = [_exact("A", 98.0, 115.0), _exact("B", 230.0, 250.0), unresolved,
               _exact("D", 300.0, 318.0), _exact("E", 302.0, 320.0)]
    first, second, third = identity.annotate_candidates(candidates, matches)

    assert first["identity_state"] == "known_exact" and first["audio_version_id"] == "sha256:" + "A" * 64
    assert first["identity_evidence"][0]["covered_fraction"] == 0.75
    assert first["identity_evidence"][0]["reference_start_seconds"] == 12.0
    # 覆う割合が足りない一致で既知曲にしない。元の候補も変更しない。
    assert second == candidates[1] and candidates[0]["identity_state"] == "unresolved"
    # 別の録音ファイルが同じ区間を覆うときは、版を決めない。
    assert third["identity_state"] == "known_exact" and third["audio_version_id"] is None
    assert {item["reference_id"] for item in third["identity_evidence"]} == {"D", "E"}


def replace_state(match, state):
    from dataclasses import replace

    return replace(match, identity_state=state)


def test_the_same_unreleased_recording_is_found_across_two_episodes_without_references():
    cue = _music(77, 30.0)
    first = _dialogue_and_noise(300.0, seed=1)
    second = _dialogue_and_noise(260.0, seed=2)
    # 1本目では区間の境目（120秒）をまたいで、2本目では別の時刻に、同じ録音の同じ部分が鳴る。
    first[int(110 * RATE):int(110 * RATE) + cue.size] += cue * 2
    second[int(40 * RATE):int(40 * RATE) + cue.size] += cue * 2
    other = _music(78, 30.0)
    second[int(150 * RATE):int(150 * RATE) + other.size] += other * 2

    [shared] = identity.shared_recordings(first, second)
    assert 110.0 - 0.05 <= shared.first_start_seconds <= 116.0
    assert shared.first_end_seconds <= 140.05 and shared.first_end_seconds - shared.first_start_seconds >= 20
    assert shared.first_start_seconds - shared.second_start_seconds == pytest.approx(70.0, abs=0.05)


def test_candidates_are_linked_without_merging_by_shared_recordings_and_references():
    candidates = {
        "ep1": [{"candidate_id": "a", "start_seconds": 100.0, "end_seconds": 140.0},
                {"candidate_id": "b", "start_seconds": 300.0, "end_seconds": 330.0,
                 "identity_evidence": [{"reference_sha256": "r" * 64}]},
                {"candidate_id": "c", "start_seconds": 500.0, "end_seconds": 520.0}],
        "ep8": [{"candidate_id": "x", "start_seconds": 50.0, "end_seconds": 70.0},
                {"candidate_id": "y", "start_seconds": 900.0, "end_seconds": 930.0,
                 "identity_evidence": [{"reference_sha256": "r" * 64}]}],
    }
    shared = {("ep1", "ep8"): [
        identity.SharedOccurrence(110.0, 125.0, 52.0, 67.0, 0.9, 7),
        # 候補の外にはみ出す共通区間ではつながない。
        identity.SharedOccurrence(515.0, 540.0, 55.0, 80.0, 0.9, 12),
    ]}
    groups = identity.link_occurrences(candidates, shared)
    members = sorted(sorted(tuple(m) for m in g["members"]) for g in groups)
    assert members == [[("ep1", "a"), ("ep8", "x")], [("ep1", "b"), ("ep8", "y")]]
    kinds = {g["evidence"][0]["kind"] for g in groups}
    assert kinds == {"shared_recording", "same_reference_recording"}
    assert all(len(g["occurrence_group_id"]) == 16 for g in groups)


def test_a_long_activity_candidate_is_split_at_known_recording_boundaries():
    candidate = {"candidate_id": "long", "start_seconds": 100.0, "end_seconds": 200.0,
                 "identity_state": "unresolved", "audio_version_id": None}
    matches = [_exact("A", 110.0, 140.0), _exact("B", 141.5, 170.0), replace_state(_exact("C", 175.0, 199.0), "unresolved")]
    pieces = identity.split_candidates_at_matches([candidate], matches)

    assert [(p["start_seconds"], p["end_seconds"], p["identity_state"]) for p in pieces] == [
        (100.0, 110.0, "unresolved"), (110.0, 140.0, "known_exact"), (140.0, 141.5, "unresolved"),
        (141.5, 170.0, "known_exact"),
        (170.0, 200.0, "unresolved")]
    assert pieces[1]["audio_version_id"] == "sha256:" + "A" * 64
    assert pieces[2]["short_unresolved_remainder"] is True
    assert all("dropped_remainders" not in p for p in pieces)
    assert all(p["derived_from"] == "long" for p in pieces)
    assert candidate["end_seconds"] == 200.0 and "derived_from" not in candidate
    untouched = {"candidate_id": "other", "start_seconds": 300.0, "end_seconds": 320.0}
    assert identity.split_candidates_at_matches([untouched], matches) == [untouched]


def test_recurring_unreleased_spans_split_candidates_without_claiming_identity():
    candidate = {"candidate_id": "long", "start_seconds": 0.0, "end_seconds": 100.0}
    matches = [_exact("A", 10.0, 30.0)]
    recurring = [(25.0, 50.0, {"other_episode": "ep8"}), (60.0, 68.0, {"other_episode": "ep8"}), (80.0, 84.0, {"other_episode": "ep8"})]
    pieces = identity.split_candidates_at_matches([candidate], matches, recurring=recurring)

    assert [(p["start_seconds"], p["end_seconds"], p["identity_state"], "recurring_evidence" in p) for p in pieces] == [
        (0.0, 10.0, "unresolved", False), (10.0, 30.0, "known_exact", False), (30.0, 50.0, "unresolved", True),
        (50.0, 60.0, "unresolved", False), (60.0, 68.0, "unresolved", True), (68.0, 100.0, "unresolved", False)]


def test_known_recordings_are_not_new_assets_but_unknown_music_stays_pending():
    known = {"candidate_id": "k", "start_seconds": 0.0, "end_seconds": 20.0, "identity_state": "known_exact",
             "identity_evidence": [{"reference_id": "A", "reference_sha256": "a" * 64, "covered_fraction": 0.95,
                                    "reference_start_seconds": 3.0}]}
    partial = {"candidate_id": "p", "start_seconds": 30.0, "end_seconds": 60.0, "identity_state": "known_exact",
               "identity_evidence": [{"reference_id": "B", "reference_sha256": "b" * 64, "covered_fraction": 0.6}]}
    unknown = {"candidate_id": "u", "start_seconds": 70.0, "end_seconds": 90.0, "identity_state": "unresolved",
               "recurring_evidence": [{"other_episode": "ep8"}]}
    decisions = identity.novelty_decisions([known, partial, unknown])
    assert [d["decision"] for d in decisions] == ["known_recording_exists", "partly_known", "pending"]
    assert decisions[0]["reference"]["reference_start_seconds"] == 3.0
    assert decisions[2]["recurs_in"] == ["ep8"]
    assert identity.episode_zero_reason(decisions) is None
    assert identity.episode_zero_reason(decisions[:1]) == "no_novel_asset"
    assert identity.episode_zero_reason([]) is None


def test_reference_fingerprints_are_cached_by_content_and_settings(tmp_path):
    signal = _music(11, 30.0)
    path = tmp_path / "ref.flac"
    path.write_bytes(b"ref-content")
    calls = []

    def loader(p):
        calls.append(p)
        return signal

    cache = tmp_path / "cache"
    first = identity.ReferenceIndex.build([path], loader=loader, cache_dir=cache)
    second = identity.ReferenceIndex.build([path], loader=loader, cache_dir=cache)
    assert len(calls) == 1
    assert np.array_equal(first.hashes, second.hashes) and first.references[0].frames == second.references[0].frames
    path.write_bytes(b"changed")  # 内容が変われば保存物は使わない
    identity.ReferenceIndex.build([path], loader=loader, cache_dir=cache)
    assert len(calls) == 2


def test_a_clipped_recurring_span_moves_the_other_episode_span_by_the_same_amount():
    candidate = {"candidate_id": "c", "start_seconds": 20.0, "end_seconds": 60.0}
    recurring = [(0.0, 50.0, {"other_episode": "ep8", "other_start_seconds": 500.0,
                              "other_end_seconds": 550.0, "waveform_explained": 0.7})]
    pieces = identity.split_candidates_at_matches([candidate], [], recurring=recurring)

    [child] = [p for p in pieces if p.get("recurring_evidence")]
    assert (child["start_seconds"], child["end_seconds"]) == (20.0, 50.0)
    [evidence] = child["recurring_evidence"]
    # 候補の端で20秒切ったので、相手の番組側も20秒進んだ位置から同じ長さになる。
    assert evidence["other_start_seconds"] == 520.0
    assert evidence["other_end_seconds"] == 550.0


def test_a_recurring_span_without_other_episode_times_is_kept_as_recorded():
    candidate = {"candidate_id": "c", "start_seconds": 20.0, "end_seconds": 60.0}
    recurring = [(0.0, 50.0, {"other_episode": "ep8"})]
    pieces = identity.split_candidates_at_matches([candidate], [], recurring=recurring)

    [child] = [p for p in pieces if p.get("recurring_evidence")]
    assert child["recurring_evidence"] == [{"other_episode": "ep8"}]


def test_decode_mono_is_decoded_once_per_unchanged_file_and_is_read_only(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from lss.bgm import identity as module

    payload = np.linspace(-1, 1, 2000, dtype=np.float32)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=payload.tobytes(), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    module._DECODED.clear()
    path = tmp_path / "analysis.wav"
    path.write_bytes(b"one")
    first = module.decode_mono(path)
    second = module.decode_mono(path)
    assert len(calls) == 1 and first is second and first.tobytes() == payload.tobytes()
    with pytest.raises(ValueError):
        first[0] = 5.0
    module.decode_mono(path, start=1.0, duration=2.0)
    assert len(calls) == 2
    path.write_bytes(b"changed!")
    module.decode_mono(path)
    assert len(calls) == 3
