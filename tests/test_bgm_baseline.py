import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from mutagen.flac import FLAC

from lss.output.tags import provenance_tags
from lss.validation.bgm_baseline import run_bgm_baseline


def _write_case(path, *, active=True):
    samples = np.zeros((1000, 2), dtype=np.float32)
    if active:
        samples[200:500] = 0.1
    sf.write(path, samples, 1000, subtype="PCM_16")


def _hash(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus(tmp_path):
    active = tmp_path / "active.flac"
    absent = tmp_path / "absent.flac"
    _write_case(active)
    _write_case(absent, active=False)
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {"type": "manifest", "schema_version": 1, "corpus_id": "baseline-test", "revision": "r1", "description": "test"},
        {
            "type": "case", "case_id": "active", "description": "active",
            "source": {"path": active.name, "sha256": _hash(active), "sample_rate": 1000, "frames": 1000},
            "expected_outcome": "wanted_present", "attributes": [],
            "occurrences": [{
                "occurrence_id": "wanted-1", "start_frame": 200, "end_frame": 500,
                "semantic_role": "dramatic_score", "wanted": True, "novel": True,
                "identity_state": "unknown", "cue_identity_id": None, "audio_version_id": None,
            }],
        },
        {
            "type": "case", "case_id": "absent", "description": "absent",
            "source": {"path": absent.name, "sha256": _hash(absent), "sample_rate": 1000, "frames": 1000},
            "expected_outcome": "no_wanted_score", "attributes": [], "occurrences": [],
        },
    ]
    manifest.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    return manifest


def _recipe(path, onset):
    path.write_text(json.dumps({
        "schema_version": 1, "method": "windowed_rms_hysteresis",
        "window_frames": 100, "hop_frames": 50,
        "onset_rms": onset, "offset_rms": onset / 2,
        "bridge_gap_frames": 0, "minimum_active_frames": 1, "padding_frames": 0,
    }), encoding="utf-8")


def _separate(job):
    source = Path(job["mix_path"])
    output = source.with_name(source.stem + "-Instrumental.flac")
    data, sample_rate = sf.read(source, always_2d=True)
    sf.write(output, data, sample_rate, subtype="PCM_16")
    audio = FLAC(output)
    tags = provenance_tags(
        recipe_id="model", model="model", model_sha256="m" * 64,
        processed_at="p", input_sha256="i" * 64, score="s", version="v",
        config_sha256="c" * 64,
        runtime_revision={"runtime_id": "r" * 64, "device": "cuda:0"},
    )
    tags["LSS_STEM"] = "Instrumental"
    for key, value in tags.items():
        audio[key] = value
    audio.save()
    return {
        "status": "ok", "outputs": [str(output)],
        "input_sha256": _hash(source),
    }


def test_runs_separation_once_per_case_and_ranks_activity_recipes(tmp_path):
    manifest = _corpus(tmp_path)
    recipe_a = tmp_path / "a.json"
    recipe_b = tmp_path / "b.json"
    _recipe(recipe_a, 0.08)
    _recipe(recipe_b, 0.2)
    calls = []

    def separate(job):
        calls.append(job)
        return _separate(job)

    scoreboard_path = run_bgm_baseline(
        manifest, [recipe_b, recipe_a], tmp_path / "run",
        model_id="model", stem_target="Instrumental", separate=separate,
    )

    scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    run = json.loads((scoreboard_path.parent / "run.json").read_text(encoding="utf-8"))
    assert len(calls) == 2
    assert scoreboard["selection_status"] == "best_observed_not_validated"
    assert scoreboard["recipes"][0]["summary"]["hard_failure_count"] == 0
    assert scoreboard["evaluation_layer"] == "activity_discovery"
    assert scoreboard["recipes"][0]["summary"]["macro_activity_coverage"] > 0
    assert (tmp_path / "run" / "run.json").is_file()
    assert len(run["evaluation_runtime"]["implementation_artifacts"]) == 3
    assert all(len(item["sha256"]) == 64 for item in run["evaluation_runtime"]["implementation_artifacts"])
    assert not list(tmp_path.glob(".run-*.tmp"))


def test_source_passthrough_runs_same_evaluator_without_separation(tmp_path):
    manifest = _corpus(tmp_path)
    recipe = tmp_path / "recipe.json"
    _recipe(recipe, 0.08)

    scoreboard_path = run_bgm_baseline(
        manifest, [recipe], tmp_path / "raw-run",
        model_id=None, stem_target="Instrumental",
        separate=lambda job: (_ for _ in ()).throw(AssertionError("分離しない")),
    )

    run = json.loads((scoreboard_path.parent / "run.json").read_text(encoding="utf-8"))
    assert run["stage_graph"] == "source_activity"
    assert all(item["stage_status"] == "source_passthrough" for item in run["separations"])


def test_rejects_worker_result_for_different_source_identity(tmp_path):
    manifest = _corpus(tmp_path)
    recipe = tmp_path / "recipe.json"
    _recipe(recipe, 0.08)

    def wrong_source(job):
        result = _separate(job)
        result["input_sha256"] = "0" * 64
        return result

    import pytest
    from lss.validation.bgm_baseline import BgmBaselineError
    with pytest.raises(BgmBaselineError, match="worker input identity"):
        run_bgm_baseline(
            manifest, [recipe], tmp_path / "run",
            model_id="model", stem_target="Instrumental", separate=wrong_source,
        )
