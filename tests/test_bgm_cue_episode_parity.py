"""既存Development番組の分岐音声を、全sampleで非破壊に照合する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
import soundfile as sf

from lss.bgm.activity import discover_stem_energy_share_files, load_stem_energy_share_recipe
from lss.bgm.cue_routing import CueRoutingRecipe, route_cues_stereo
from lss.bgm.routing import DisagreementRoutingRecipe
from lss.core.session import _write_json_atomic
from lss.harness.fetch import file_sha256
from lss.validation.routed_episode_discovery import read_audio
from scripts.audit_bgm_routing_stereo import verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get("LSS_RUN_BGM_CUE_EPISODE_PARITY") != "1",
                    reason="長いDevelopment番組の全sample比較は明示実行だけ")
def test_existing_episode_audio_and_cues_are_exactly_preserved(tmp_path):
    original_path = Path(os.environ["LSS_BGM_CUE_PARITY_PLAN"])
    rendering_path = original_path.parent / "rendering.json"
    plan = json.loads(original_path.read_text(encoding="utf-8"))
    saved = json.loads(rendering_path.read_text(encoding="utf-8"))
    assert plan["promotion_allowed"] is False and plan["holdout_consumed"] is False
    assert saved["plan_sha256"] == file_sha256(original_path)
    artifacts = plan["records"] + [saved["signal"]] + [
        {"path": str(p), "sha256": file_sha256(p)} for p in (original_path, rendering_path)]
    verify(artifacts)
    implementation = {str(p.relative_to(ROOT)): file_sha256(p) for p in (
        Path(__file__), ROOT / "lss/bgm/routing.py", ROOT / "lss/bgm/cue_routing.py",
        ROOT / "lss/bgm/activity.py", ROOT / "lss/validation/routed_episode_discovery.py")}
    directory = tmp_path / "cue-episode-parity"
    directory.mkdir(exist_ok=False)
    frozen_path = directory / "plan.json"
    _write_json_atomic(frozen_path, {
        "status": "FROZEN_BEFORE_EPISODE_PARITY", "promotion_allowed": False,
        "holdout_consumed": False, "artifacts": artifacts, "implementation": implementation,
        "scope": "existing_development_episode_exact_sample_and_cue_parity_without_inference",
    })
    started = time.monotonic()
    paths = {role.title(): Path(item["path"]) for role, item in plan["stems"].items()}
    activity = load_stem_energy_share_recipe(Path(plan["activity_recipe"]["path"]))
    discovery = discover_stem_energy_share_files(paths, activity)
    intervals = [(candidate.start_frame, candidate.end_frame) for candidate in discovery.candidates]
    route_data = dict(plan["cue_recipe"])
    route_data["window_recipe"] = DisagreementRoutingRecipe(**route_data["window_recipe"])
    recipe = CueRoutingRecipe(**route_data)
    assert recipe.recipe_id == plan["cue_recipe_id"]
    parent = read_audio(Path(plan["instrumental"]["path"]), restore_headroom=True)
    stems = {role: read_audio(path) for role, path in paths.items()}
    result = route_cues_stereo(parent, stems, intervals, recipe)
    assert json.loads(json.dumps(result.cues)) == saved["cues"]
    assert result.audio.shape == (2, plan["frames"])
    del parent, stems
    compared = 0
    with sf.SoundFile(saved["signal"]["path"]) as reference:
        assert reference.frames == plan["frames"] and reference.channels == 2 and reference.samplerate == 44100
        while compared < reference.frames:
            values = reference.read(min(1 << 18, reference.frames - compared), dtype="float64", always_2d=True).T
            np.testing.assert_array_equal(result.audio[:, compared:compared + values.shape[1]], values)
            compared += values.shape[1]
    verify(artifacts)
    assert all(file_sha256(ROOT / p) == sha for p, sha in implementation.items())
    _write_json_atomic(directory / "report.json", {
        "status": "complete_development_episode_exact_parity", "promotion_allowed": False,
        "default_changed": False, "holdout_consumed": False,
        "plan_sha256": file_sha256(frozen_path), "source_hashes_after_verified": True,
        "frames_compared": compared, "cue_count": len(result.cues), "exact_audio_equal": True,
        "exact_cues_equal": True, "elapsed_seconds": time.monotonic() - started,
        "limits": ["parity_not_semantic_or_quality_validation", "original_alignment_uncertainty_retained"],
    })
