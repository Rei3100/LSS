"""Counterexamples that the earlier independent inventory oracle cannot reject."""
try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root

import importlib.util
from pathlib import Path


ARTIFACT = dev_evidence_root(Path(__file__).resolve().parents[1]) / 'release-evidence/strata-matrix-20261029/vision_binding.py'
spec = importlib.util.spec_from_file_location('strata_binding_controls', ARTIFACT)
binding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(binding)


def test_same_inventories_with_wrong_object_pairs_fail(tmp_path):
    rows = binding.fixtures(tmp_path)
    expected, swapped = rows[0]['objects'], rows[1]['objects']
    assert sorted(o['color'] for o in expected) == sorted(o['color'] for o in swapped)
    assert sorted(o['shape'] for o in expected) == sorted(o['shape'] for o in swapped)
    value = {'objects': swapped, 'unknown': False, 'scene_event': None}
    assert not binding.judge(value, expected, True)
    assert binding.judge(dict(value, objects=list(reversed(expected))), expected, True)
    assert len({row['sha256'] for row in rows}) == 5


def test_position_swap_and_duplicate_cannot_pass(tmp_path):
    expected = binding.fixtures(tmp_path)[0]['objects']
    swapped = [dict(o, position='right' if o['position'] == 'left' else 'left') for o in expected]
    assert not binding.judge({'objects': swapped, 'unknown': False, 'scene_event': None}, expected, True)
    assert not binding.judge({'objects': [expected[0], expected[0]], 'unknown': False, 'scene_event': None}, expected, True)


def test_blank_omitted_and_invented_event_stay_distinct():
    blank = {'objects': [], 'unknown': False, 'scene_event': None}
    assert binding.judge(blank, [], True)
    assert not binding.judge(blank, [], False)
    assert binding.judge(dict(blank, unknown=True), [], False)
    assert not binding.judge(dict(blank, scene_event='A battle begins'), [], True)
    assert not binding.judge(dict(blank, objects=['red square']), [], True)
