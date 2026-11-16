"""先に公開した別の原本のcueと同じ組み立て済み音声だけを重複にし、それ以外は公開を変えないことを確かめる。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss import application
from lss.bgm import cue_semantics, identity, published_duplicates as dup

RATE = identity.SAMPLE_RATE


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
    return (out / np.max(np.abs(out)) * 0.4).astype(np.float32)


def _voice(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = int(seconds * RATE)
    voice = np.zeros(size, np.float32)
    position = 0
    while position < size:
        length = int(rng.uniform(0.2, 0.6) * RATE)
        pitch = rng.uniform(100, 250)
        t = np.arange(min(length, size - position)) / RATE
        burst = sum(np.sin(2 * np.pi * pitch * k * t) / k for k in range(1, 12))
        voice[position:position + t.size] = 0.25 * burst * np.hanning(t.size)
        position += length + int(rng.uniform(0.05, 0.4) * RATE)
    return voice


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prior_run(root: Path, key: str, audio: np.ndarray, *, source_sha: str = "a" * 64,
               source_start: float = 100.0, published: tuple[float, float] | None = None,
               status: str = "completed_published_cues") -> Path:
    run_dir = root / key / "run"
    (run_dir / "candidates").mkdir(parents=True)
    path = run_dir / "candidates" / "cue-001-raw.flac"
    sf.write(path, audio, RATE, subtype="PCM_24")
    start = int(source_start * RATE)
    span = published or (source_start, source_start + audio.size / RATE)
    run = {"status": status, "source": {"sha256": source_sha},
           "held": [{"cue_id": "cue-001", "raw": {"path": str(path), "sha256": _sha(path),
                                                  "source_start_frame": start, "sample_rate": RATE}}],
           "published": [{"index": 1, "start_frame": int(span[0] * RATE), "end_frame": int(span[1] * RATE),
                          "sample_rate": RATE, "candidates": ["cue-001"]}]}
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return path


@pytest.fixture
def assembled():
    """話をまたいで同じ組み立て済み音声（音楽＋ナレーション）と、同じ音楽に別の台詞が重なった版。"""
    music = _music(7, 30.0)
    return music + _voice(30.0, 1), music + _voice(30.0, 2)


def _episode(middle: np.ndarray) -> np.ndarray:
    return np.concatenate([_music(11, 10.0), middle, _music(12, 20.0)]).astype(np.float32)


def test_identical_assembled_audio_of_a_published_cue_is_a_duplicate(tmp_path, assembled):
    same, _ = assembled
    _prior_run(tmp_path, "k1", same)
    parts, runs = dup.prior_publications(tmp_path, "b" * 64)
    assert runs == ["k1"] and len(parts) == 1
    spans = dup.identical_spans(_episode(same), parts, dup.default_recipe(), cache_dir=tmp_path / "fp")
    assert spans and all(s["waveform_explained"] >= 0.99 for s in spans)
    assert spans[0]["prior_run"] == "k1" and spans[0]["prior_cue_index"] == 1
    assert dup.duplicate_of(10.0, 40.0, spans, dup.default_recipe())
    # 前後の別の曲は重複にしない
    assert dup.duplicate_of(0.0, 10.0, spans, dup.default_recipe()) is None
    assert dup.duplicate_of(40.0, 60.0, spans, dup.default_recipe()) is None


def test_a_published_cue_made_of_two_candidates_is_matched_as_one_signal(tmp_path, assembled):
    """実番組では公開cue（0〜14秒）が保留候補の境目（13秒）をまたぐ。候補ごとに照合すると被覆が足りなくなる。"""
    same, _ = assembled
    run_dir = tmp_path / "k2" / "run"
    (run_dir / "candidates").mkdir(parents=True)
    cut = int(26 * RATE)
    held, members = [], []
    for number, (low, high) in enumerate(((0, cut), (cut, same.size)), 1):
        path = run_dir / "candidates" / f"cue-{number:03d}-raw.flac"
        sf.write(path, same[low:high], RATE, subtype="PCM_24")
        held.append({"cue_id": f"cue-{number:03d}", "raw": {"path": str(path), "sha256": _sha(path),
                                                           "source_start_frame": low, "sample_rate": RATE}})
    run = {"status": "completed_published_cues", "source": {"sha256": "a" * 64}, "held": held,
           "published": [{"index": 1, "start_frame": 0, "end_frame": same.size, "sample_rate": RATE,
                          "candidates": ["cue-001", "cue-002"]}]}
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    cues, _ = dup.prior_publications(tmp_path, "b" * 64)
    assert len(cues) == 1 and len(cues[0].members) == 2
    spans = dup.identical_spans(_episode(same), cues, dup.default_recipe())
    assert dup.duplicate_of(10.0, 40.0, spans, dup.default_recipe())


def test_same_music_under_different_dialogue_is_not_a_duplicate(tmp_path, assembled):
    same, other_dialogue = assembled
    _prior_run(tmp_path, "k1", same)
    parts, _ = dup.prior_publications(tmp_path, "b" * 64)
    spans = dup.identical_spans(_episode(other_dialogue), parts, dup.default_recipe())
    assert dup.duplicate_of(10.0, 40.0, spans, dup.default_recipe()) is None


def test_audio_outside_the_prior_published_span_is_not_evidence(tmp_path, assembled):
    same, _ = assembled
    # 相手の話ではこの音声の最初の10秒だけが公開cue（残りは保留のまま）。
    _prior_run(tmp_path, "k1", same, published=(100.0, 110.0))
    parts, _ = dup.prior_publications(tmp_path, "b" * 64)
    spans = dup.identical_spans(_episode(same), parts, dup.default_recipe())
    assert dup.duplicate_of(10.0, 40.0, spans, dup.default_recipe()) is None


def test_prior_material_is_limited_to_other_sources_with_published_cues(tmp_path, assembled):
    same, _ = assembled
    _prior_run(tmp_path, "self", same, source_sha="b" * 64)
    _prior_run(tmp_path, "held", same, status="completed_held_candidates")
    missing = _prior_run(tmp_path, "gone", same)
    missing.unlink()
    parts, runs = dup.prior_publications(tmp_path, "b" * 64)
    assert parts == [] and runs == []


def test_changed_prior_material_gives_no_evidence(tmp_path, assembled):
    same, _ = assembled
    path = _prior_run(tmp_path, "k1", same)
    parts, _ = dup.prior_publications(tmp_path, "b" * 64)
    sf.write(path, _music(99, 30.0), RATE, subtype="PCM_24")
    assert dup.identical_spans(_episode(same), parts, dup.default_recipe()) == []


def test_recipe_identity_and_bounds():
    recipe = dup.default_recipe()
    assert recipe.recipe_id.startswith("bgm-published-duplicates-v1-")
    assert dup.DuplicateRecipe(minimum_explained=0.98).recipe_id != recipe.recipe_id
    with pytest.raises(dup.PublishedDuplicatesError):
        dup.DuplicateRecipe(minimum_cover=0.1).validate()


def _decided(*spans):
    return [{"start_seconds": a, "end_seconds": b, "semantic_class": cue_semantics.WANTED, "reasons": []}
            for a, b in spans]


def test_product_step_marks_only_duplicates_and_records_the_evidence(tmp_path, assembled):
    same, _ = assembled
    candidates = tmp_path / "bgm-candidates"
    _prior_run(candidates, "k1", same)
    run_dir = candidates / "mine" / "run"
    run_dir.mkdir(parents=True)
    analysis = tmp_path / "analysis.wav"
    sf.write(analysis, _episode(same), RATE, subtype="FLOAT")
    decided = _decided((0.0, 10.0), (10.0, 40.0), (40.0, 60.0))
    record = application._bgm_published_duplicates(
        application._bgm_candidate_recipe(), run_dir, analysis, decided, "b" * 64, tmp_path / "fp")
    assert record["state"] == "checked" and record["prior_runs"] == ["k1"]
    assert [item["semantic_class"] for item in decided] == [
        cue_semantics.WANTED, cue_semantics.DUPLICATE_PUBLISHED, cue_semantics.WANTED]
    assert decided[1]["reasons"] == [dup.REASON] and decided[1]["duplicate_of"][0]["prior_run"] == "k1"


def test_product_step_without_prior_publication_changes_nothing(tmp_path):
    run_dir = tmp_path / "bgm-candidates" / "mine" / "run"
    run_dir.mkdir(parents=True)
    decided = _decided((0.0, 30.0))
    record = application._bgm_published_duplicates(
        application._bgm_candidate_recipe(), run_dir, tmp_path / "unused.wav", decided, "b" * 64, None)
    assert record["state"] == "no_prior_publication"
    assert decided[0]["semantic_class"] == cue_semantics.WANTED


def test_recipe_declares_the_duplicate_rule():
    recipe = application._bgm_candidate_recipe()
    assert recipe["recipe_id"].startswith("bgm-episode-candidates-v13-")
    # v11では、一曲かどうかだけが未確定の区間も重複の照合の対象（対応領域S1）。
    assert recipe["supported_scope_id"] == recipe["supported_scope"]["scope_id"]
    assert recipe["published_duplicates_id"] == dup.default_recipe().recipe_id
    assert "bgm/published_duplicates.py" in recipe["implementation"]
