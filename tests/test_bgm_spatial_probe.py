import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from lss.bgm.refinement import SpatialRefinementRecipe
from scripts.run_bgm_spatial_probe import evaluate, evaluation_interval, graph_parts, sha256
from lss.bgm.blend import StereoBlendRecipe


def frozen_probe(tmp_path):
    rng = np.random.default_rng(3)
    raw = rng.normal(0, 0.05, (1025, 2))
    paths = {}
    for name, values in (("raw", raw), ("reference", np.zeros_like(raw)),
                         ("music", np.zeros_like(raw)), ("speech", raw),
                         ("effects", np.zeros_like(raw))):
        path = tmp_path / f"{name}.wav"
        sf.write(path, values, 44100, subtype="FLOAT")
        paths[name] = {"path": str(path), "sha256": sha256(path), "frames": 1025}
    plan = {
        "status": "FROZEN_BEFORE_RESULTS", "promotion_allowed": False,
        "code": {}, "sample_rate": 44100,
        "candidates": [SpatialRefinementRecipe(n_fft=128, hop_frames=32, covariance_frames=9).to_dict()],
        "cases": [{
            "case_id": "no-music", "target_absent": True, "singing_guard": False,
            "evidence_role": "controlled-clean-component", "source": paths["raw"],
            "reference": paths["reference"],
            "stems": {role: paths[role] for role in ("music", "speech", "effects")},
            "signal_frames": [0, 1025], "evaluation_frames": [0, 1025],
        }],
        "decision_rule": {"maximum_absent_output_to_input_db": -30.0,
                          "minimum_material_controlled_improvement_db": 1.0},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return SimpleNamespace(plan=path, output=tmp_path / "result", receipt=None), paths


def test_probe_preserves_sources_and_cannot_promote_no_improvement(tmp_path):
    args, paths = frozen_probe(tmp_path)
    evaluate(args)
    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    assert report["source_identity_preserved"]
    assert not report["promotion_allowed"]
    assert not report["holdout_consumed"]
    assert report["l4_oracle"] is None
    assert report["summaries"][0]["status"] == "REJECTED_EXACT_REFINEMENT"
    assert report["summaries"][0]["failures"] == []
    assert all(sha256(__import__("pathlib").Path(item["path"])) == item["sha256"] for item in paths.values())


def test_modified_comparator_is_rejected_before_scoring(tmp_path):
    args, paths = frozen_probe(tmp_path)
    sf.write(paths["music"]["path"], np.ones((1025, 2)) * 0.2, 44100, subtype="FLOAT")
    with pytest.raises(ValueError, match="identity"):
        evaluate(args)
    assert not args.output.exists()


def test_unmatched_cuda_receipt_is_rejected_before_scoring(tmp_path):
    args, _ = frozen_probe(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"status": "COMPLETED_CUDA_WITHOUT_FALLBACK", "plan_sha256": "bad"}), encoding="utf-8")
    args.receipt = receipt
    with pytest.raises(ValueError, match="receipt"):
        evaluate(args)


def test_blended_graph_has_explicit_parent_weight_and_order_identity(tmp_path):
    spatial = SpatialRefinementRecipe(n_fft=128, hop_frames=32, covariance_frames=9).to_dict()
    first = {"spatial": spatial, "blend": StereoBlendRecipe(0.25).to_dict()}
    second = {"spatial": spatial, "blend": StereoBlendRecipe(0.5).to_dict()}
    assert graph_parts(first)[2] != graph_parts(second)[2]
    args, _ = frozen_probe(tmp_path)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan["candidates"] = [first, second]
    args.plan.write_text(json.dumps(plan), encoding="utf-8")
    evaluate(args)
    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    assert len(report["summaries"]) == 2
    assert all(not value["failures"] for value in report["summaries"])


def test_absence_interval_includes_padding_and_entire_timeline():
    assert evaluation_interval([], 2646000) == [0, 2646000]
    assert evaluation_interval([{"start_frame": 100, "end_frame": 300},
                                {"start_frame": 700, "end_frame": 800}], 1000) == [100, 800]


def test_absence_activity_rejects_short_phantom_music_at_source_start(tmp_path):
    args, paths = frozen_probe(tmp_path)
    phantom = np.zeros((1025, 2))
    phantom[:128] = 0.2
    sf.write(paths["music"]["path"], phantom, 44100, subtype="FLOAT")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan["cases"][0]["stems"]["music"]["sha256"] = sha256(__import__("pathlib").Path(paths["music"]["path"]))
    plan["candidates"] = [{"spatial": plan["candidates"][0], "blend": StereoBlendRecipe(0.25).to_dict()}]
    recipe_path = tmp_path / "activity.json"
    recipe_path.write_text(json.dumps({
        "schema_version": 1, "method": "stem_energy_share_hysteresis",
        "onset_share": 0.2, "offset_share": 0.1,
        "onset_music_rms": 0.001, "offset_music_rms": 0.0005,
        "window_frames": 128, "hop_frames": 64, "bridge_gap_frames": 0,
        "minimum_active_frames": 1, "padding_frames": 0,
    }), encoding="utf-8")
    plan["activity_recipe"] = {"path": str(recipe_path), "sha256": sha256(recipe_path)}
    plan["decision_rule"]["maximum_absent_activity_candidate_count"] = 0
    args.plan.write_text(json.dumps(plan), encoding="utf-8")
    evaluate(args)
    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    candidate = report["cases"][0]["candidates"][0]
    assert candidate["quality"]["activity_candidate_count"] > 0
    assert "target_absent_activity_false_positive" in candidate["failures"]
    assert not list(args.output.glob("absence-*"))
