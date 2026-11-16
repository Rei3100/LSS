try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = dev_evidence_root(ROOT) / 'release-evidence/strata-matrix-20261029/contract_trial.py'
sys.path.insert(0, str(PATH.parent))
spec = importlib.util.spec_from_file_location('contract_trial', PATH)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class ContractStudyTest(unittest.TestCase):
    def test_six_deterministic_parent_blob_contexts_without_fixed_or_test_sources(self):
        first = subject.build_population()
        self.assertEqual(first, subject.build_population())
        self.assertEqual(len(first), 6)
        for entry in first:
            task, receipt = entry['task'], entry['provenance']
            commit, path, target = subject.TARGETS[task['id']]
            self.assertEqual(receipt['parent_commit'], subject.git('rev-parse', commit + '^').strip())
            source = subject.git('show', receipt['parent_commit'] + ':' + path)
            self.assertEqual(receipt['blob_oid'], subject.git('rev-parse', receipt['parent_commit'] + ':' + path).strip())
            self.assertEqual(receipt['blob_sha256'], hashlib.sha256(source.encode()).hexdigest())
            self.assertLessEqual(len(receipt['context']), 6000)
            self.assertEqual(receipt['context_sha256'], hashlib.sha256(receipt['context'].encode()).hexdigest())
            tree = ast.parse(receipt['context'])
            self.assertNotIn(target, [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            self.assertNotIn('expected', receipt['context'].lower())
            self.assertNotIn('hidden', receipt['context'].lower())
            self.assertNotIn('/test', path.lower())
            self.assertTrue(receipt['contract'])
        current = next(x for x in first if x['task']['id'] == 'currentbest-ddfd56d1')
        self.assertIn('start_seconds', current['provenance']['context'])
        audio = next(x for x in first if x['task']['id'].endswith('model_input'))
        self.assertIn('keyword-only', audio['provenance']['contract']['rules'])

    def test_system_contract_and_effective_capabilities(self):
        audio_id = 'fix-f567482a-model_input'
        system = subject.system_instruction(audio_id)
        self.assertNotIn('Preserve the signature', system)
        self.assertNotIn('Do not introduce new imports', system)
        self.assertIn('keyword-only', system)
        self.assertEqual(set(subject.capabilities(audio_id)['nodes']),
                         {n.__name__ for n in subject.audio_evaluator.NODES})
        self.assertEqual(set(subject.capabilities(audio_id)['imports']), subject.audio_evaluator.IMPORTS)
        common = subject.capabilities('fix-c3049e9e-remaining_without_blocker')
        self.assertEqual(set(common['direct_calls']), set(subject.bench.SAFE_BUILTINS))
        self.assertEqual(set(common['attributes']), set(subject.bench.SAFE_METHODS) | {'S', 'I', 'M'})
        self.assertIn('any', common['direct_calls'])
        product = subject.capabilities('currentbest-ddfd56d1')
        self.assertIn('workflow_version', product['direct_calls'])
        self.assertNotIn('workflow_version', subject.capabilities('fix-c3049e9e-remaining_without_blocker')['direct_calls'])

    def test_output_budget_and_original_prompt_are_unchanged(self):
        frozen = {x['id']: x for x in map(__import__('json').loads, subject.trial.HISTORY.read_text(encoding='utf-8').splitlines())}
        for item in subject.build_population():
            self.assertEqual(item['task'], frozen[item['task']['id']])


if __name__ == '__main__':
    unittest.main()
