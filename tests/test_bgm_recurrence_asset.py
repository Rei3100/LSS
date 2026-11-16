"""Recurrence consensus asset: where two or more published occurrences of one recording play, the asset is cleaner
than the longest occurrence; it is published once per group, kept when the
members do not change, never written over a file LSS did not write, and episode cues are never touched."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile
from mutagen.flac import FLAC

from lss.bgm import cue_memory, cue_publication, current_best, recurrence_asset
from lss.core import audio_identity
from lss.output import result_report, results

RATE = 22050


def _music(seed: int, seconds: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    note = int(0.25 * RATE)
    count = int(seconds * RATE) // note + 1
    t = np.arange(note) / RATE
    envelope = np.minimum(1.0, t / 0.01) * np.exp(-3 * t)
    out = np.zeros(count * note)
    for index in range(count):
        chunk = np.zeros(note)
        for midi in rng.integers(48, 84, size=3):
            frequency = 440.0 * 2 ** ((midi - 69) / 12)
            chunk += np.sin(2 * np.pi * frequency * t) + 0.5 * np.sin(4 * np.pi * frequency * t)
        out[index * note:(index + 1) * note] = chunk * envelope
    out = out[:int(seconds * RATE)]
    return out / np.max(np.abs(out)) * 0.4


def _residue(seconds: float, seed: int, level: float = 0.05) -> np.ndarray:
    """Dialogue-like residue a separator leaves, different in every occurrence."""
    rng = np.random.default_rng(seed)
    size = int(seconds * RATE)
    voice = np.zeros(size)
    position = int(rng.uniform(0, 1.0) * RATE)
    while position < size:
        length = int(rng.uniform(0.4, 1.2) * RATE)
        pitch = rng.uniform(100, 250)
        t = np.arange(min(length, size - position)) / RATE
        burst = sum(np.sin(2 * np.pi * pitch * k * t) / k for k in range(1, 12))
        voice[position:position + t.size] = level * burst * np.hanning(t.size)
        position += length + int(rng.uniform(0.8, 2.5) * RATE)
    return voice


RECORDING = _music(1, 45.0)


def _cue(folder: Path, source: str, start: float, mono: np.ndarray, *, seed: int, level: float = 0.05) -> Path:
    audio = mono + _residue(mono.size / RATE, seed, level)
    stereo = np.stack([audio, audio * 0.9], axis=1)
    sha = (source.encode().hex() * 64)[:64]
    start_frame = int(start * RATE)
    end_frame = start_frame + stereo.shape[0]
    key = cue_publication.cue_key(sha, "recipe", start_frame, end_frame, RATE)
    provenance = {"schema_version": 1, "type": "bgm_published_cue",
                  "source": {"path": str(folder / f"{source}.mkv"), "sha256": sha},
                  "start_frame": start_frame, "end_frame": end_frame, "sample_rate": RATE,
                  "start_seconds": start, "end_seconds": start + stereo.shape[0] / RATE,
                  "cue_extent": {"state": "single_cue"}}
    name = cue_publication.file_name(Path(f"{source}.mkv"), start)
    record = cue_publication.publish(stereo, RATE, folder / f"{source} - BGM" / name, key=key,
                                     title=cue_publication.title(Path(f"{source}.mkv"), start), album=f"{source} BGM",
                                     provenance=provenance, input_sha256=sha, recipe_id="recipe", version="test")
    return Path(record["path"])


def _observe(state: Path, path: Path) -> dict:
    item = cue_memory.published_item(path)
    # The cue is also its source's CurrentBest file: only such a member names a library the asset may be written to.
    record = current_best.read(state, item["source_sha256"]) or {
        "type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
        "source_sha256": item["source_sha256"], "current": {"files": []}, "history": []}
    record["current"]["files"].append({"public_path": item["path"], "pcm_sha256": item["pcm_sha256"],
                                       "cue_key": item["cue_key"]})
    current_best._write_json_atomic(current_best.record_path(state, item["source_sha256"]), record)
    return cue_memory.observe(state, item["source_sha256"], [item])


def _sdr(estimate: np.ndarray, truth: np.ndarray) -> float:
    size = min(estimate.size, truth.size)
    error = estimate[:size] - truth[:size]
    return float(10 * np.log10(np.sum(truth[:size] ** 2) / np.sum(error ** 2)))


@pytest.fixture()
def library(tmp_path):
    state, out = tmp_path / "state", tmp_path / "LSS"
    # the longest occurrence plays 0-40 s; three others cover its first 30 s (different episodes, own clocks)
    paths = [_cue(out, "ep-a", 100.0, RECORDING[:40 * RATE], seed=11),
             _cue(out, "ep-b", 200.0, RECORDING[:30 * RATE], seed=12),
             _cue(out, "ep-c", 300.0, RECORDING[:30 * RATE], seed=13)]
    for path in paths:
        _observe(state, path)
    return state, out, paths


def test_three_occurrences_make_a_cleaner_asset_than_the_longest(library):
    state, _, paths = library
    data = cue_memory.read(state)
    assert len([entry for entry in data["labels"].values() if not entry.get("merged_into")]) == 1
    members = data["members"]
    signal, rate, explanation = recurrence_asset.build(members, data["links"])
    assert rate == RATE and explanation["base"] == "bgm-cue:" + cue_memory.published_item(paths[0])["cue_key"]
    assert explanation["consensus_seconds"] > 20
    base, _ = soundfile.read(str(paths[0]), always_2d=True)
    assert signal.shape == base.shape
    truth = RECORDING[:40 * RATE]
    shared = slice(1 * RATE, 29 * RATE)
    assert _sdr(signal[shared, 0], truth[shared]) > _sdr(base[shared, 0], truth[shared])
    # where only the longest occurrence plays, the asset is that occurrence
    alone = slice(31 * RATE, 39 * RATE)
    assert np.max(np.abs(signal[alone, 0] - base[alone, 0])) < 1e-3


def _pair(tmp_path, second: np.ndarray, *, seeds=(11, 12), level: float = 0.05):
    state, out = tmp_path / "state", tmp_path / "LSS"
    paths = [_cue(out, "ep-a", 100.0, RECORDING[:30 * RATE], seed=seeds[0], level=level),
             _cue(out, "ep-b", 200.0, second, seed=seeds[1])]
    for path in paths:
        _observe(state, path)
    data = cue_memory.read(state)
    return state, out, paths, data


def test_two_occurrences_with_different_residue_give_a_cleaner_asset(tmp_path):
    state, out, paths, data = _pair(tmp_path, RECORDING[:30 * RATE])
    signal, rate, explanation = recurrence_asset.build(data["members"], data["links"])
    base, _ = soundfile.read(str(paths[0]), always_2d=True)
    other, _ = soundfile.read(str(paths[1]), always_2d=True)
    assert signal.shape == base.shape and explanation["consensus_seconds"] == 0
    assert explanation["pair_seconds"] > 20 and explanation["evidence_seconds"] > 20
    truth = RECORDING[:30 * RATE]
    shared = slice(2 * RATE, 28 * RATE)
    assert _sdr(signal[shared, 0], truth[shared]) > _sdr(base[shared, 0], truth[shared]) + 1.0
    assert _sdr(signal[shared, 0], truth[shared]) > _sdr(other[shared, 0], truth[shared]) + 1.0
    found = recurrence_asset.update(state)
    assert [row["state"] for row in found["assets"]] == ["published"]
    assert (out / recurrence_asset.FOLDER_NAME).exists()


def test_two_occurrences_one_with_a_dropout_never_lose_more_than_the_floor(tmp_path):
    hole = RECORDING[:30 * RATE].copy()
    hole[12 * RATE:13 * RATE] = 0.0  # the other copy's separator dropped the music here
    state, out, paths, data = _pair(tmp_path, hole, level=0.15)
    signal, _, explanation = recurrence_asset.build(data["members"], data["links"])
    base, _ = soundfile.read(str(paths[0]), always_2d=True)
    other, _ = soundfile.read(str(paths[1]), always_2d=True)
    truth = RECORDING[:30 * RATE]
    shared = slice(2 * RATE, 28 * RATE)
    # residue in the base and a drop-out in the other copy: better than either over the shared passage
    assert _sdr(signal[shared, 0], truth[shared]) > _sdr(base[shared, 0], truth[shared])
    assert _sdr(signal[shared, 0], truth[shared]) > _sdr(other[shared, 0], truth[shared])
    # inside the drop-out the base is attenuated by at most the floor (-12 dB), never silenced
    gap = slice(12 * RATE + RATE // 5, 13 * RATE - RATE // 5)
    ratio = float(np.sqrt(np.mean(signal[gap, 0] ** 2) / np.mean(base[gap, 0] ** 2)))
    assert ratio > recurrence_asset.PAIR_FLOOR * 0.8
    assert explanation["pair_seconds"] > 20


def test_two_identical_occurrences_are_left_unchanged(tmp_path):
    state, out, paths, data = _pair(tmp_path, RECORDING[:30 * RATE], seeds=(11, 11))
    signal, _, _ = recurrence_asset.build(data["members"], data["links"])
    base, _ = soundfile.read(str(paths[0]), always_2d=True)
    assert signal.shape == base.shape
    inner = slice(1 * RATE, 29 * RATE)
    assert np.max(np.abs(signal[inner] - base[inner])) < 2e-3


def test_pair_policy_off_keeps_the_longest_occurrence_and_makes_no_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(recurrence_asset, "PAIR_GAMMA", None)
    state, out, paths, data = _pair(tmp_path, RECORDING[:30 * RATE])
    signal, _, explanation = recurrence_asset.build(data["members"], data["links"])
    base, _ = soundfile.read(str(paths[0]), always_2d=True)
    assert explanation["pair_seconds"] == 0
    inner = slice(1 * RATE, 29 * RATE)
    assert np.max(np.abs(signal[inner] - base[inner])) < 1e-3
    assert [row["state"] for row in recurrence_asset.update(state)["assets"]] == ["no_consensus"]


def test_published_once_then_reused_and_episode_cues_untouched(library):
    state, out, paths = library
    before = [audio_identity.file_digest(path) for path in paths]
    found = recurrence_asset.update(state)
    row, = found["assets"]
    assert row["state"] == "published" and found["published_count"] == 1
    target = Path(row["path"])
    assert target.parent == out / recurrence_asset.FOLDER_NAME and target.name == "劇伴#1 統合版.flac"
    tags = FLAC(target)
    assert "劇伴#" not in tags["TITLE"][0] and "統合版" not in tags["TITLE"][0] and tags["ALBUM"] == [recurrence_asset.ALBUM]
    provenance = json.loads(tags[recurrence_asset.ASSET_TAG][0])
    assert provenance["label_number"] == 1 and len(provenance["members"]) == 3
    assert "REPLAYGAIN_TRACK_GAIN" in tags

    def never(*_):
        raise AssertionError("an unchanged group is not rebuilt")

    again = recurrence_asset.update(state, builder=never)
    assert [item["state"] for item in again["assets"]] == ["reused"]
    assert [audio_identity.file_digest(path) for path in paths] == before


def test_new_occurrence_rebuilds_with_lss_title_and_keeps_previous_revision(library):
    state, out, _ = library
    first, = recurrence_asset.update(state)["assets"]
    target = Path(first["path"])
    tags = FLAC(target)
    tags["TITLE"] = "他のソフトが付けた曲名"
    tags.save()
    old_pcm = audio_identity.audio_identity(target)["pcm_sha256"]
    _observe(state, _cue(out, "ep-d", 400.0, RECORDING[:30 * RATE], seed=14))
    second, = recurrence_asset.update(state)["assets"]
    assert second["state"] == "published" and second["replaced_previous"] and second["members"] == 4
    assert "劇伴#" not in FLAC(target)["TITLE"][0]
    assert audio_identity.audio_identity(target)["pcm_sha256"] != old_pcm
    record = recurrence_asset.read_record(state, "1")
    kept = Path(record["history"][-1]["kept_copy"])
    assert kept.exists() and audio_identity.audio_identity(kept)["pcm_sha256"] == old_pcm


def test_file_lss_did_not_write_is_never_overwritten(library):
    state, out, _ = library
    target = out / recurrence_asset.FOLDER_NAME / "劇伴#1 統合版.flac"
    target.parent.mkdir(parents=True)
    soundfile.write(str(target), np.zeros((RATE, 2)), RATE, format="FLAC")
    before = target.read_bytes()
    row, = recurrence_asset.update(state)["assets"]
    assert row["state"] == "path_occupied" and target.read_bytes() == before


def test_a_failed_build_is_contained(library):
    state, out, _ = library

    def broken(*_):
        raise RuntimeError("decoder")

    row, = recurrence_asset.update(state, builder=broken)["assets"]
    assert row["state"] == "failed" and not (out / recurrence_asset.FOLDER_NAME).exists()


def test_fewer_consensus_seconds_keep_the_previous_revision(library):
    state, _, _ = library
    first, = recurrence_asset.update(state)["assets"]
    record = recurrence_asset.read_record(state, "1")
    # the published revision rests on more multi-occurrence evidence than a new member set gives
    record["current"].update(signature="another member set", evidence_seconds=999.0)
    recurrence_asset._write_record(state, "1", record)
    published = Path(first["path"]).read_bytes()
    row, = recurrence_asset.update(state)["assets"]
    assert row["state"] == "kept_previous" and row["path"] == first["path"]
    assert Path(first["path"]).read_bytes() == published

def test_projection_and_report_name_the_asset(library):
    state, out, paths = library
    assets = recurrence_asset.update(state)
    memory = _observe(state, paths[0])
    memory["assets"] = assets
    projection = results.project(disposition="succeeded", outputs=(paths[0],), recurrence=memory)
    assert projection["recurrence_asset_count"] == 1 and projection["recurrence_asset_updated_count"] == 1
    page = result_report.render(Path("ep-a.mkv"), "bgm", {**projection, "published_count": 1})
    assert "統合版" in page and "劇伴#1 統合版" in page
    unchecked = results.project(disposition="succeeded", outputs=(paths[0],),
                                recurrence={**memory, "assets": {"state": "not_checked", "assets": []}})
    assert unchecked["recurrence_asset_count"] is None


def test_legacy_link_ranges_sorted_apart_are_paired_by_length():
    # an edit: this cue's 0-6 s is the other's 40-46 s and its 10-22 s is the other's 2-14 s
    link = {"a": "x", "b": "y", "a_ranges": [[0.0, 6.0], [10.0, 22.0]], "b_ranges": [[2.0, 14.0], [40.0, 46.0]]}
    assert recurrence_asset._pieces(link, "x") == [(0.0, 6.0, 40.0, 1.0), (10.0, 22.0, 2.0, 1.0)]
    assert recurrence_asset._pieces(link, "y") == [(2.0, 14.0, 10.0, 1.0), (40.0, 46.0, 0.0, 1.0)]
    ambiguous = {**link, "b_ranges": [[2.0, 8.0], [40.0, 46.0]], "a_ranges": [[0.0, 6.0], [10.0, 16.0]]}
    assert recurrence_asset._pieces(ambiguous, "x") == []
    paired = {**link, "paired": True, "b_ranges": [[40.0, 46.0], [2.0, 14.0]]}
    assert recurrence_asset._pieces(paired, "x") == [(0.0, 6.0, 40.0, 1.0), (10.0, 22.0, 2.0, 1.0)]


def test_a_layer_of_other_music_is_left_out(library):
    state, out, paths = library
    data = cue_memory.read(state)
    wrong = _cue(out, "ep-x", 500.0, _music(7, 30.0), seed=21)
    item = cue_memory.published_item(wrong)
    key = f"bgm-cue:{item['cue_key']}"
    members = {**data["members"], key: {**item, "asset_id": key}}
    base = next(k for k, v in members.items() if v["duration_seconds"] > 35)
    # a link the matcher would never make: claims the unrelated cue is the base's first 30 s
    links = data["links"] + [{"a": base, "b": key, "paired": True, "a_ranges": [[0.0, 30.0]], "b_ranges": [[0.0, 30.0]]}]
    signal, _, explanation = recurrence_asset.build(members, links)
    assert [layer["source_name"] for layer in explanation["layers_left_out"]] == ["ep-x"]
    truth = RECORDING[:40 * RATE]
    good, _, _ = recurrence_asset.build(data["members"], data["links"])
    shared = slice(1 * RATE, 29 * RATE)
    assert abs(_sdr(signal[shared, 0], truth[shared]) - _sdr(good[shared, 0], truth[shared])) < 0.5

def test_a_longest_cue_of_other_music_publishes_nothing(library):
    state, out, _ = library
    data = cue_memory.read(state)
    wrong = _cue(out, "ep-x", 500.0, _music(7, 44.0), seed=21)
    item = cue_memory.published_item(wrong)
    key = f"bgm-cue:{item['cue_key']}"
    members = {**data["members"], key: {**item, "asset_id": key}}
    links = data["links"] + [{"a": key, "b": other, "paired": True, "a_ranges": [[0.0, 29.0]], "b_ranges": [[0.0, 29.0]]}
                             for other in data["members"]]
    with pytest.raises(recurrence_asset.RecurrenceAssetError):
        recurrence_asset.build(members, links)

def _stretch(signal: np.ndarray, rate: float, segment_seconds: float = 0.25, fade_seconds: float = 0.01) -> np.ndarray:
    """Pitch-preserving tempo change as editors make it: output time u plays input time u * rate (copied segments)."""
    segment, fade = int(segment_seconds * RATE), int(fade_seconds * RATE)
    length = int(signal.size / rate)
    out = np.zeros(length + segment + fade)
    norm = np.zeros_like(out)
    window = np.ones(segment + fade)
    window[:fade], window[-fade:] = np.linspace(0, 1, fade), np.linspace(1, 0, fade)
    for start in range(0, length, segment):
        piece = signal[int(start * rate):int(start * rate) + segment + fade]
        if piece.size < segment + fade:
            break
        out[start:start + piece.size] += piece * window
        norm[start:start + piece.size] += window
    return (out / np.maximum(norm, 1e-3))[:length]


@pytest.fixture()
def tempo_library(tmp_path):
    """The longest occurrence and one more at the recording's tempo, a third played 4 % slower (pitch kept)."""
    state, out = tmp_path / "state", tmp_path / "LSS"
    paths = [_cue(out, "ep-a", 100.0, RECORDING[:40 * RATE], seed=11),
             _cue(out, "ep-b", 200.0, RECORDING[:30 * RATE], seed=12),
             _cue(out, "ep-c", 300.0, _stretch(RECORDING[:30 * RATE], 1 / 1.04), seed=13)]
    for path in paths:
        _observe(state, path)
    return state, out, paths


def test_an_occurrence_at_another_tempo_joins_the_consensus(tempo_library):
    state, _, paths = tempo_library
    data = cue_memory.read(state)
    assert any(link.get("basis") == "stretched_lag_path" for link in data["links"])
    slower = f"bgm-cue:{cue_memory.published_item(paths[2])['cue_key']}"
    middle = f"bgm-cue:{cue_memory.published_item(paths[1])['cue_key']}"
    base, _ = soundfile.read(str(paths[0]), always_2d=True)
    truth = RECORDING[:40 * RATE]
    shared = slice(2 * RATE, 28 * RATE)
    # the longest occurrence and one heard 4 % slower only: before, nothing could be aligned and no asset was made
    members = {key: value for key, value in data["members"].items() if key != middle}
    links = [link for link in data["links"] if middle not in (link["a"], link["b"])]
    alone, _, explanation = recurrence_asset.build(members, links)
    (layer,) = [row for row in explanation["layers"] if row["member"] == slower]
    assert abs(layer["tempo_scale"] - 1.04) < 0.01 and layer["median_consistency"] >= 0.3
    assert slower in explanation["members_used"] and not explanation["layers_left_out"]
    assert explanation["tempo_seconds"] > 20 and explanation["evidence_seconds"] > 20
    assert _sdr(alone[shared, 0], truth[shared]) > _sdr(base[shared, 0], truth[shared]) + 1.5
    # with an occurrence at the recording's tempo there, the slower copy is not used: the pair rule decides
    signal, _, both = recurrence_asset.build(data["members"], data["links"])
    constant = [link for link in data["links"] if link.get("basis") != "stretched_lag_path"]
    two, _, _ = recurrence_asset.build({k: v for k, v in data["members"].items() if k != slower}, constant)
    assert both["tempo_seconds"] < 1 and both["pair_seconds"] > 20
    assert np.max(np.abs(signal[shared] - two[shared])) < 1e-4
    # the asset keeps the base's clock: where only the base plays it is that occurrence
    only = slice(32 * RATE, 39 * RATE)
    assert np.max(np.abs(signal[only, 0] - base[only, 0])) < 1e-3

def test_a_tempo_link_to_other_music_is_left_out(tempo_library):
    state, out, paths = tempo_library
    data = cue_memory.read(state)
    slower = f"bgm-cue:{cue_memory.published_item(paths[2])['cue_key']}"
    # the same claimed tempo passage, but the file holds other music played at that tempo
    wrong = _cue(out, "ep-x", 500.0, _stretch(_music(7, 30.0), 1 / 1.04), seed=21)
    members = {**data["members"], slower: {**data["members"][slower], "path": str(wrong)}}
    signal, _, explanation = recurrence_asset.build(members, data["links"])
    assert [row["member"] for row in explanation["layers_left_out"]] == [slower]
    assert explanation["layers_left_out"][0]["median_consistency"] < recurrence_asset.MIN_LAYER_CONSISTENCY


def test_tempo_layers_compose_through_a_constant_link():
    """A member linked only to a tempo-changed occurrence inherits that occurrence's tempo on the base clock."""
    members = {"base": {"duration_seconds": 40.0}, "slow": {"duration_seconds": 31.0},
               "slow-copy": {"duration_seconds": 31.0}}
    links = [{"a": "base", "b": "slow", "paired": True, "rate": 1 / 1.04, "a_ranges": [[0.0, 31.2]],
              "b_ranges": [[0.0, 30.0]]},
             {"a": "slow", "b": "slow-copy", "paired": True, "a_ranges": [[0.0, 30.0]], "b_ranges": [[1.0, 31.0]]}]
    found = recurrence_asset.plan(members, links, 1000)
    by = {layer.member: layer for layer in found.layers}
    assert abs(by["slow"].scale - 30.0 / 31.2) < 1e-9 and by["slow"].lag == 0
    assert abs(by["slow-copy"].scale - by["slow"].scale) < 1e-9 and by["slow-copy"].lag == 1000
    assert by["slow-copy"].via == "via:slow" and by["slow-copy"].end == by["slow"].end

def test_a_tempo_copy_does_not_move_the_consensus_of_constant_occurrences(library):
    state, out, paths = library
    constant, _, _ = recurrence_asset.build(cue_memory.read(state)["members"], cue_memory.read(state)["links"])
    slower = _cue(out, "ep-d", 400.0, _stretch(RECORDING[:30 * RATE], 1 / 1.04), seed=14)
    _observe(state, slower)
    data = cue_memory.read(state)
    assert any(link.get("basis") == "stretched_lag_path" for link in data["links"])
    signal, _, explanation = recurrence_asset.build(data["members"], data["links"])
    assert explanation["consensus_seconds"] > 20 and explanation["tempo_seconds"] < 1
    # identical wherever the constant occurrences play together; the copy may act only past their end (30 s),
    # where the base plays alone
    together = slice(0, int(29.5 * RATE))
    assert np.max(np.abs(signal[together] - constant[together])) < 1e-4