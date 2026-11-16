import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lss.bgm.inference_variants import InferenceVariantRecipe
from scripts.run_bgm_dominance_probe import definition, evaluate
from scripts.run_bgm_spatial_probe import sha256
from tests.test_bgm_inference_probe import fixture


def routing_fixture(tmp_path, *, absent=False):
    args, _ = fixture(tmp_path, absent=absent)
    parent = json.loads(args.plan.read_text(encoding='utf-8'))
    adapter = Path(__import__('scripts.run_bgm_inference_probe', fromlist=['x']).__file__)
    parent['code']['scripts/run_bgm_inference_probe.py'] = sha256(adapter)
    args.plan.write_text(json.dumps(parent), encoding='utf-8')
    receipt = json.loads(args.receipt.read_text(encoding='utf-8'))
    receipt['plan_sha256'] = sha256(args.plan)
    args.receipt.write_text(json.dumps(receipt), encoding='utf-8')
    plan = {'status': 'FROZEN_BEFORE_RESULTS', 'promotion_allowed': False, 'holdout_consumed': False,
            'parent_plan': {'path': str(args.plan), 'sha256': sha256(args.plan)}, 'code': {},
            'candidates': [definition('dominant-gain', [InferenceVariantRecipe(6.0).to_dict()], None)],
            'decision_rule': parent['decision_rule'], 'evidence_scope': 'Development only'}
    path = tmp_path / 'routing.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    return SimpleNamespace(plan=path, receipt=args.receipt, output=args.output)


def test_routed_improvement_remains_development_with_paired_metrics(tmp_path):
    args = routing_fixture(tmp_path)
    evaluate(args)
    report = json.loads((args.output / 'report.json').read_text(encoding='utf-8'))
    assert report['summaries'][0]['status'] == 'DEVELOPMENT_SURVIVOR'
    assert not report['summaries'][0]['failures']
    assert not report['promotion_allowed'] and not report['holdout_consumed']
    assert report['source_identity_preserved'] and report['l4_oracle'] is None
    assert report['cases'][0]['candidates'][0]['mean_weights']['dominant'] == 1


def test_routing_preserves_whole_source_absence(tmp_path):
    args = routing_fixture(tmp_path, absent=True)
    evaluate(args)
    report = json.loads((args.output / 'report.json').read_text(encoding='utf-8'))
    assert report['cases'][0]['candidates'][0]['quality']['activity_candidate_count'] == 0
    assert report['cases'][0]['candidates'][0]['mean_weights']['parent'] == 1


def test_routing_rejects_wrong_parent_identity_before_output(tmp_path):
    args = routing_fixture(tmp_path)
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    plan['parent_plan']['sha256'] = '0' * 64
    args.plan.write_text(json.dumps(plan), encoding='utf-8')
    with pytest.raises(ValueError, match='parent'):
        evaluate(args)
    assert not args.output.exists()


def test_graph_identity_includes_child_variant_and_routing_role():
    child = [InferenceVariantRecipe(-6.0).to_dict()]
    assert definition('route', child, None)['recipe_id'] != definition('route', None, child)['recipe_id']
    assert definition('route', child, None)['recipe_id'] != definition('route', [InferenceVariantRecipe(6.0).to_dict()], None)['recipe_id']
    assert definition('route', child, None)['recipe_id'] != definition('route', child, None, True)['recipe_id']


def test_routing_plan_change_during_evaluation_cannot_publish_report(tmp_path, monkeypatch):
    from scripts import run_bgm_dominance_probe as probe

    args = routing_fixture(tmp_path)
    original = probe.route_stereo

    def change_plan(*values, **kwargs):
        result = original(*values, **kwargs)
        with args.plan.open("a") as stream:
            stream.write(" ")
        return result

    monkeypatch.setattr(probe, "route_stereo", change_plan)
    with pytest.raises(ValueError, match="比較中に固定routing plan"):
        evaluate(args)
    assert not (args.output / "report.json").exists()
