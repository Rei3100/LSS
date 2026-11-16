import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.bgm.inference_variants import InferenceVariantRecipe
from scripts.run_bgm_inference_probe import checked_plan, evaluate
from scripts.run_bgm_spatial_probe import artifact, sha256


def fixture(tmp_path, *, absent=False):
    rng = np.random.default_rng(37)
    music = rng.normal(0, 0.03, (1025, 2)) if not absent else np.zeros((1025, 2))
    raw = music + rng.normal(0, 0.02, (1025, 2))

    def audio(name, values):
        path = tmp_path / (name + '.wav')
        sf.write(path, values, 44100, subtype='FLOAT')
        return artifact(path)

    source, reference = audio('source', raw), audio('reference', music)
    recipes = [InferenceVariantRecipe(), InferenceVariantRecipe(6.0)]
    records = []
    for index, recipe in enumerate(recipes):
        estimate = music if index or absent else raw
        stems = {'music': audio(f'music-{index}', estimate),
                 'speech': audio(f'speech-{index}', raw - estimate),
                 'effects': audio(f'effects-{index}', np.zeros_like(raw))}
        records.append({'case_id': 'case', 'variant_id': recipe.recipe_id,
                        'source_sha256': source['sha256'], 'stems': stems})
    activity = tmp_path / 'activity.json'
    activity.write_text(json.dumps({'schema_version': 1, 'method': 'stem_energy_share_hysteresis',
                                   'onset_share': 0.2, 'offset_share': 0.1, 'onset_music_rms': 0.001,
                                   'offset_music_rms': 0.0005, 'window_frames': 128, 'hop_frames': 64,
                                   'bridge_gap_frames': 0, 'minimum_active_frames': 1, 'padding_frames': 0}), encoding='utf-8')
    plan = {'status': 'FROZEN_BEFORE_RESULTS', 'promotion_allowed': False, 'holdout_consumed': False,
            'code': {}, 'provider': {}, 'variants': [recipe.to_dict() for recipe in recipes],
            'candidates': [{'label': 'gain', 'variants': [recipes[1].to_dict()]}],
            'cases': [{'case_id': 'case', 'evidence_role': 'controlled-clean-component', 'target_absent': absent,
                       'singing_guard': False, 'source': source, 'reference': reference, 'stems': {},
                       'evaluation_frames': [0, 1025]}],
            'activity_recipe': {'path': str(activity), 'sha256': sha256(activity)},
            'decision_rule': {'maximum_any_controlled_regression_db': 1, 'minimum_material_controlled_improvement_db': 1,
                              'maximum_absent_output_to_input_db': -30, 'maximum_absent_activity_candidate_count': 0},
            'claim_boundary': 'Development only'}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({'status': 'COMPLETED_CUDA_WITHOUT_FALLBACK', 'plan_sha256': sha256(path),
                                  'provider': {}, 'adapter_sha256': sha256(Path(__import__('scripts.run_bgm_inference_probe', fromlist=['x']).__file__)),
                                  'cases': records}), encoding='utf-8')
    return SimpleNamespace(plan=path, receipt=receipt, output=tmp_path / 'result'), records


def test_paired_improvement_remains_development_without_promotion(tmp_path):
    args, _ = fixture(tmp_path)
    evaluate(args)
    report = json.loads((args.output / 'report.json').read_text(encoding='utf-8'))
    assert report['summaries'][0]['status'] == 'DEVELOPMENT_SURVIVOR'
    assert report['summaries'][0]['failures'] == []
    assert report['source_identity_preserved'] and not report['promotion_allowed']
    assert not report['holdout_consumed'] and report['l4_oracle'] is None


def test_whole_source_absence_stays_zero_activity(tmp_path):
    args, _ = fixture(tmp_path, absent=True)
    evaluate(args)
    report = json.loads((args.output / 'report.json').read_text(encoding='utf-8'))
    assert report['cases'][0]['candidates'][0]['quality']['activity_candidate_count'] == 0
    assert not report['summaries'][0]['failures']


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'wrong-source', 'wrong-plan'])
def test_corrupted_receipt_is_rejected(tmp_path, mutation):
    args, _ = fixture(tmp_path)
    receipt = json.loads(args.receipt.read_text(encoding='utf-8'))
    if mutation == 'duplicate':
        receipt['cases'].append(receipt['cases'][0])
    elif mutation == 'missing':
        receipt['cases'].pop()
    elif mutation == 'wrong-source':
        receipt['cases'][0]['source_sha256'] = '0' * 64
    else:
        receipt['plan_sha256'] = '0' * 64
    args.receipt.write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(ValueError):
        evaluate(args)


def test_changed_inference_stem_is_rejected_before_output(tmp_path):
    args, records = fixture(tmp_path)
    sf.write(records[1]['stems']['music']['path'], np.ones((1025, 2)), 44100, subtype='FLOAT')
    with pytest.raises(ValueError, match='identity'):
        evaluate(args)
    assert not args.output.exists()


def test_all_si_scores_improving_cannot_hide_reference_gain_damage(tmp_path):
    args, records = fixture(tmp_path)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    reference = sf.read(plan["cases"][0]["reference"]["path"], dtype="float32", always_2d=True)[0]
    estimate = records[1]["stems"]["music"]
    sf.write(estimate["path"], reference * 0.5, 44100, subtype="FLOAT")
    estimate["sha256"] = sha256(Path(estimate["path"]))
    plan["decision_rule"]["spatial_gain_max_extra_error_db"] = 0.1
    plan["decision_rule"]["spatial_component_si_max_regression_db"] = 0.1
    args.plan.write_text(json.dumps(plan), encoding="utf-8")
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    receipt["plan_sha256"] = sha256(args.plan)
    receipt["cases"] = records
    args.receipt.write_text(json.dumps(receipt), encoding="utf-8")
    evaluate(args)
    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    candidate = report["cases"][0]["candidates"][0]
    assert all(delta >= 0 for delta in candidate["paired_delta_db"].values())
    assert candidate["failures"] == ["individual_stereo_component_regression"]
    assert any(row["metric"] == "projection_gain_db" for row in candidate["quality"]["spatial_regressions"])
    assert report["summaries"][0]["status"] == "REJECTED_EXACT_CONFIGURATION"


def test_holdout_path_cannot_enter_development(tmp_path):
    args, _ = fixture(tmp_path)
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    plan['cases'][0]['source']['path'] = str(tmp_path / 'Sources' / 'Holdout' / 'source.wav')
    args.plan.write_text(json.dumps(plan), encoding='utf-8')
    with pytest.raises(ValueError, match='Holdout'):
        checked_plan(args.plan)


@pytest.mark.parametrize("method", ["inference", "dominance"])
def test_whole_song_improvement_cannot_hide_wanted_accompaniment_loss(tmp_path, method):
    from tests.test_song_component_comparison import signals

    args, records = fixture(tmp_path)
    accompaniment, singing, noise = signals()
    accompaniment, singing = ((value * 0.5).astype(np.float32).astype(np.float64) for value in (accompaniment, singing))
    noise *= 0.5
    reference = (accompaniment + singing).astype(np.float32).astype(np.float64)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    case = plan["cases"][0]
    for item in (case["source"], case["reference"]):
        sf.write(item["path"], reference.T, 44100, subtype="FLOAT")
        item.update(artifact(Path(item["path"])))
    # FLOAT WAVのPEAK時刻で別ファイルのencoded hashが異なり得る。
    # 制御構成のwhole referenceは固定sourceそのものを指す。
    case["reference"] = dict(case["source"])
    case["evaluation_frames"] = [0, reference.shape[1]]
    case["wanted_components"] = {}
    for role, values in (("accompaniment", accompaniment), ("singing", singing)):
        path = tmp_path / (role + ".wav")
        sf.write(path, values.T, 44100, subtype="FLOAT")
        case["wanted_components"][role] = artifact(path)
    for record, values in zip(records, (reference + noise, accompaniment * 0.01 + singing)):
        record["source_sha256"] = case["source"]["sha256"]
        for role, item in record["stems"].items():
            sf.write(item["path"], (values if role == "music" else np.zeros_like(values)).T,
                     44100, subtype="FLOAT")
            item.update(artifact(Path(item["path"])))
    plan["decision_rule"]["maximum_extra_wanted_component_gain_error_db"] = 0.1
    args.plan.write_text(json.dumps(plan), encoding="utf-8")
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    receipt.update(plan_sha256=sha256(args.plan), cases=records)
    args.receipt.write_text(json.dumps(receipt), encoding="utf-8")
    run = evaluate
    if method == "dominance":
        from scripts.run_bgm_dominance_probe import definition, evaluate as routed_evaluate
        route_path = tmp_path / "routing.json"
        route_path.write_text(json.dumps({
            "status": "FROZEN_BEFORE_RESULTS", "promotion_allowed": False, "holdout_consumed": False,
            "parent_plan": {"path": str(args.plan), "sha256": sha256(args.plan)}, "code": {},
            "candidates": [definition("gain", [InferenceVariantRecipe(6.0).to_dict()], None)],
            "decision_rule": plan["decision_rule"], "evidence_scope": "Development only"}), encoding="utf-8")
        args.plan = route_path
        run = routed_evaluate
    run(args)
    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    candidate = report["cases"][0]["candidates"][0]
    assert all(delta >= 0 for delta in candidate["paired_delta_db"].values())
    assert candidate["failures"] == ["individual_wanted_component_regression"]
    assert candidate["quality"]["wanted_component_regressions"][0]["component"] == "accompaniment"


@pytest.mark.parametrize("mutation", ["hash", "sum", "partial_reference", "missing_role", "holdout"])
def test_wanted_component_diagnostic_requires_unchanged_complete_additive_truth(tmp_path, mutation):
    args, _ = fixture(tmp_path)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    case = plan["cases"][0]
    case["reference"] = dict(case["source"])
    source = sf.read(case["source"]["path"], dtype="float64", always_2d=True)[0]
    # 正確な加算構成。単独成分の例でも境界の検証は必須。
    path = tmp_path / "zero.wav"
    sf.write(path, np.zeros_like(source), 44100, subtype="FLOAT")
    case["wanted_components"] = {"accompaniment": dict(case["source"]), "singing": artifact(path)}
    if mutation == "hash":
        with path.open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "sum":
        sf.write(path, np.ones_like(source) * 0.01, 44100, subtype="FLOAT")
        case["wanted_components"]["singing"] = artifact(path)
    elif mutation == "partial_reference":
        case["reference"]["start_seconds"] = 0
    elif mutation == "missing_role":
        case["wanted_components"].pop("singing")
    else:
        case["wanted_components"]["singing"]["path"] = str(tmp_path / "Holdout" / "zero.wav")
    args.plan.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError):
        checked_plan(args.plan)
    assert not args.output.exists()
