"""番組runをまたいで、リリースされていない録音の再登場を、runを変えずに別記録として付ける。"""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.bgm import identity_run, series_identity
from lss.core import session
from lss.validation import episode_discovery as episode
from tests.test_bgm_identity import RATE, _dialogue_and_noise, _music
from tests.test_bgm_identity_run import _mono


def _run(tmp_path, name, *, cue_at, cue, known=None, known_at=None, seed=0):
    """200秒の番組。`cue_at`秒から未発売の録音（cue）が20秒、任意で既知の録音が20秒鳴る。"""
    mix = _dialogue_and_noise(200.0, seed=seed) * 0.05
    rng = np.random.default_rng(seed + 10)
    for start in (cue_at, known_at):
        if start is not None:
            mix[int((start - 5) * RATE):int((start + 25) * RATE)] += (
                rng.standard_normal(30 * RATE).astype(np.float32) * 0.2)  # 候補ができる音量
    mix[int(cue_at * RATE):int((cue_at + 20) * RATE)] += cue[: 20 * RATE] * 1.5
    if known is not None:
        mix[int(known_at * RATE):int((known_at + 20) * RATE)] += known[: 20 * RATE] * 1.5
    source = tmp_path / f"{name}.wav"
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

    return episode.run_episode_discovery(source, [recipe], tmp_path / name, source_id=name, model_id="m",
                                         separate=separate)


@pytest.fixture
def two_runs(tmp_path):
    cue = _music(41, 30.0)
    known = _music(42, 30.0)
    reference = tmp_path / "known.flac"
    sf.write(reference, np.stack([known, known], axis=1), RATE, subtype="PCM_24")
    first = _run(tmp_path, "ep1", cue_at=30.0, cue=cue, known=known, known_at=120.0, seed=1)
    second = _run(tmp_path, "ep2", cue_at=140.0, cue=cue, seed=2)
    identity_run.annotate_run(first, [reference], loader=_mono)
    return first, second


def test_an_unreleased_recording_recurring_in_two_episodes_links_candidates_without_changing_runs(two_runs, tmp_path):
    first, second = two_runs
    before = {path: path.read_bytes() for path in (first, second, first.parent / "identity.json")}

    record_path = series_identity.annotate_series([first, second], tmp_path / "series", loader=_mono,
                                                  cache_dir=tmp_path / "cache")

    assert {path: path.read_bytes() for path in before} == before
    record = json.loads(record_path.read_text(encoding="utf-8"))
    [pair] = record["shared_recordings"]
    [occurrence] = pair["occurrences"]
    assert abs(occurrence["first_start_seconds"] - 30.0) <= 1.0 and abs(occurrence["second_start_seconds"] - 140.0) <= 1.0
    [recipe] = record["activity_recipes"]
    ep1, ep2 = recipe["candidates"]["ep1"], recipe["candidates"]["ep2"]
    assert any(c["identity_state"] == "known_exact" for c in ep1)  # 1本目の参照録音との一致も使う
    recurring1 = [c for c in ep1 if c.get("recurring_evidence")]
    recurring2 = [c for c in ep2 if c.get("recurring_evidence")]
    assert recurring1 and recurring2
    assert all(c["identity_state"] == "unresolved" for c in recurring1 + recurring2)  # 再登場は同一の既知曲ではない
    decisions = {d["candidate_id"]: d for d in recipe["novelty_decisions"]["ep2"]}
    assert decisions[recurring2[0]["candidate_id"]]["recurs_in"] == ["ep1"]
    [group] = [g for g in recipe["occurrence_groups"]
               if any(e["kind"] == "shared_recording" for e in g["evidence"])]
    assert {member[0] for member in group["members"]} == {"ep1", "ep2"}

    # 同じ入力なら作り直さない。
    stamp = record_path.stat().st_mtime_ns
    assert series_identity.annotate_series([second, first], tmp_path / "series", loader=_mono) == record_path
    assert record_path.stat().st_mtime_ns == stamp


def test_pair_results_are_reused_from_the_cache_by_signal_content(two_runs, tmp_path, monkeypatch):
    first, second = two_runs
    series_identity.annotate_series([first, second], tmp_path / "a", loader=_mono, cache_dir=tmp_path / "cache")

    def fail(*_args, **_kwargs):
        raise AssertionError("保存した組の照合を再計算した")

    monkeypatch.setattr(series_identity.identity, "match", fail)
    series_identity.annotate_series([first, second], tmp_path / "b", loader=_mono, cache_dir=tmp_path / "cache")


def test_a_single_run_or_a_changed_activity_record_is_refused(two_runs, tmp_path):
    first, second = two_runs
    with pytest.raises(series_identity.SeriesIdentityError):
        series_identity.annotate_series([first], tmp_path / "one", loader=_mono)
    run = json.loads(second.read_text(encoding="utf-8"))
    activity = run["activity_runs"][0]["record_path"]
    with open(activity, "a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(series_identity.SeriesIdentityError):
        series_identity.annotate_series([first, second], tmp_path / "changed", loader=_mono)


def test_the_asset_catalog_lists_a_recurring_recording_without_a_reference(two_runs, tmp_path):
    from lss.assets import catalog

    first, second = two_runs
    series = series_identity.annotate_series([first, second], tmp_path / "series", loader=_mono)
    database = catalog.build(state_root=tmp_path / "state", identity_records=[first.parent / "identity.json", series],
                             database=tmp_path / "catalog.sqlite")
    counts = catalog.summary(database=database)
    assert counts["recordings"] == 2 and counts["recordings_in_multiple_sources"] == 1
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(database)) as connection:
        [(recording,)] = connection.execute(
            "SELECT sha256 FROM recordings WHERE sha256 LIKE 'recurring:%'").fetchall()
    places = catalog.occurrences_of(recording, database=database)
    assert len(places) == 2 and sorted(round(p["start_seconds"]) for p in places) == [30, 140]


def test_each_episode_is_matched_once_against_all_other_episodes(tmp_path, monkeypatch):
    """番組の組ごとに照合せず、番組ごとに1回だけ照合する（長い連続番組でも番組数に比例）。"""
    cue = _music(51, 30.0)
    runs = [_run(tmp_path, f"ep{number}", cue_at=20.0 + 30 * number, cue=cue, seed=number) for number in range(4)]
    calls = []
    original = series_identity.identity.match

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(series_identity.identity, "match", counted)
    record = json.loads(series_identity.annotate_series(runs, tmp_path / "series", loader=_mono)
                        .read_text(encoding="utf-8"))
    assert len(calls) == 4
    assert len(record["shared_recordings"]) == 6
    assert all(len(pair["occurrences"]) >= 1 for pair in record["shared_recordings"])


def test_a_recurring_recording_gives_an_unreleased_candidate_a_readable_extent(two_runs, tmp_path):
    first, second = two_runs

    record_path = series_identity.annotate_series([first, second], tmp_path / "series", loader=_mono,
                                                  cache_dir=tmp_path / "cache")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    [recipe] = record["activity_recipes"]
    ep2 = recipe["candidates"]["ep2"]
    [recurring] = [c for c in ep2 if c.get("recurring_evidence")]
    # 2本目に参照録音は無い。再登場だけで範囲が読める。
    assert recurring["identity_state"] == "unresolved"
    assert recurring["cue_extent"]["state"] == "single_cue"
    states = [row["evidence"]["evidence_kind"]
              for row in recurring["cue_extent"]["evidence"]["readings"]
              if row["state"] == "single_cue"]
    assert states == ["recurring_recording"]
    assert record["intent"]["cue_extent_recipe_id"].startswith("bgm-cue-extent-v1-")


def test_candidates_without_identity_or_recurring_evidence_get_no_extent_reading(two_runs, tmp_path):
    first, second = two_runs

    record_path = series_identity.annotate_series([first, second], tmp_path / "series", loader=_mono,
                                                  cache_dir=tmp_path / "cache")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    [recipe] = record["activity_recipes"]
    bare = [c for c in recipe["candidates"]["ep2"]
            if not c.get("recurring_evidence") and not c.get("identity_evidence")]
    assert bare and all("cue_extent" not in c for c in bare)
