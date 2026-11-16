"""A previous answer or the OLD_ field cannot satisfy fresh retrieval controls."""
try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root

import ast
import importlib.util
from pathlib import Path


ARTIFACT = dev_evidence_root(Path(__file__).resolve().parents[1]) / 'release-evidence/strata-matrix-20261029/context_nonce.py'
spec = importlib.util.spec_from_file_location('strata_nonce_controls', ARTIFACT)
nonce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nonce)


def test_counterfactual_pairs_change_only_the_target_value():
    cases = {case['id']: case for case in nonce.cases()}
    assert len(cases) == 21
    values = []
    for key, case in cases.items():
        tree = ast.parse(case['document'])
        assignments = {n.targets[0].id: n.value.value for n in tree.body if isinstance(n, ast.Assign)}
        if key.endswith('/missing'):
            assert nonce.FIELD not in assignments
            assert case['expected'] == 'UNKNOWN'
        else:
            assert assignments[nonce.FIELD] == case['expected']
            assert assignments['OLD_' + nonce.FIELD] != case['expected']
            values.append(case['expected'])
        if key.endswith('/a'):
            other = cases[key[:-1] + 'b']
            assert case['document'].replace(case['expected'], '<value>') == other['document'].replace(other['expected'], '<value>')
    assert len(set(values)) == 18
    assert all(len(value) == 64 for value in values)
    assert 'accepted_cue_001' not in str(cases)
