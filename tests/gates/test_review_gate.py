"""Review modes and freshness using only synthetic data and the standard library."""
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("review_gate_under_test", ROOT / "gates/review_gate.py")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.policy = {"mode": "local-records", "bot": {"login": None, "id": None, "app_id": None},
                       "completion_pattern": None}
        self.pr = {"base_sha": "b" * 40, "head_sha": "a" * 40, "diff_digest": "d" * 64,
                   "draft": False, "complete": True, "implementation_engines": ["claude"],
                   "reviews": [], "review_threads": [], "independent_records": [self.record("codex")]}

    def record(self, engine):
        return {"base_sha": "b" * 40, "review_head_sha": "a" * 40, "diff_digest": "d" * 64,
                "engine": engine, "complete": True, "summary": "Checked in a separate context",
                "findings_p0p1": 0, "unresolved_p0p1": 0}

    def test_checked_in_policy_selects_local_records(self):
        policy = json.loads((ROOT / ".github/review-policy.json").read_text(encoding="utf-8"))
        self.assertEqual(policy["mode"], "local-records")
        self.assertEqual(GATE.verify(self.pr, policy), [])

    def test_missing_or_unknown_mode_blocks(self):
        for value in (None, "", "local", "BOT", [], {}):
            with self.subTest(mode=value):
                policy = {**self.policy, "mode": value}
                self.assertTrue(GATE.verify(self.pr, policy))
        del self.policy["mode"]
        self.assertTrue(GATE.verify(self.pr, self.policy))

    def test_each_implementation_requires_another_engine(self):
        cases = [(["claude"], [], False), (["claude"], ["claude"], False),
                 (["claude"], ["codex"], True), (["codex"], [], False),
                 (["codex"], ["codex"], False), (["codex"], ["claude"], True),
                 (["claude", "codex"], ["claude"], False),
                 (["claude", "codex"], ["codex"], False),
                 (["claude", "codex"], ["claude", "codex"], True),
                 (["human"], [], False), (["human"], ["human"], False),
                 (["human"], ["claude"], True), (["human"], ["codex"], True),
                 (["human", "claude"], ["claude"], False),
                 (["human", "claude"], ["codex"], True)]
        for engines, reviewers, passed in cases:
            with self.subTest(engines=engines, reviewers=reviewers):
                self.pr["implementation_engines"] = engines
                self.pr["independent_records"] = [self.record(e) for e in reviewers]
                self.assertEqual(GATE.verify(self.pr, self.policy) == [], passed)

    def test_stale_revision_identity_blocks(self):
        for field, value in (("base_sha", "c" * 40), ("diff_digest", "e" * 64),
                             ("head_sha", "c" * 40)):
            with self.subTest(field=field):
                pr = {**self.pr, field: value}
                self.assertTrue(GATE.verify(pr, self.policy))

    def test_descendant_requires_ancestor_and_a_only_changes(self):
        old = self.pr["head_sha"]
        self.pr["head_sha"] = "c" * 40
        for ancestor, change, passed in ((True, "A", True), (False, "A", False),
                                        (True, "B", False), (True, "C", False),
                                        (True, None, False), (1, "A", False)):
            with self.subTest(ancestor=ancestor, change=change):
                self.pr.update(ancestors={old: ancestor}, changes_since_review={old: change})
                self.assertEqual(GATE.verify(self.pr, self.policy) == [], passed)
        for field, value in (("base_sha", "f" * 40), ("diff_digest", "f" * 64)):
            pr = {**self.pr, field: value, "ancestors": {old: True}, "changes_since_review": {old: "A"}}
            self.assertTrue(GATE.verify(pr, self.policy))

    def test_only_completed_valid_records_count(self):
        cases = [("complete", False), ("complete", 1), ("summary", " "), ("summary", None),
                 ("engine", "human"), ("review_head_sha", "bad"),
                 ("findings_p0p1", -1), ("findings_p0p1", True), ("findings_p0p1", 0.0),
                 ("findings_p0p1", "0"), ("findings_p0p1", None),
                 ("unresolved_p0p1", 1), ("unresolved_p0p1", -1),
                 ("unresolved_p0p1", False), ("unresolved_p0p1", 0.0),
                 ("unresolved_p0p1", "0"), ("unresolved_p0p1", None)]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self.pr["independent_records"] = [{**self.record("codex"), field: value}]
                self.assertTrue(GATE.verify(self.pr, self.policy))
        for field in self.record("codex"):
            with self.subTest(missing=field):
                record = self.record("codex")
                del record[field]
                self.pr["independent_records"] = [record]
                self.assertTrue(GATE.verify(self.pr, self.policy))

    def test_zero_findings_and_resolved_findings_count(self):
        for findings in (0, 3):
            self.pr["independent_records"] = [{**self.record("codex"), "findings_p0p1": findings}]
            self.assertEqual(GATE.verify(self.pr, self.policy), [])

    def test_one_current_record_is_enough_among_old_records(self):
        stale = {**self.record("codex"), "base_sha": "f" * 40}
        self.pr["independent_records"] = [stale, self.record("codex")]
        self.assertEqual(GATE.verify(self.pr, self.policy), [])

    def test_other_gate_checks_still_block(self):
        for field, value in (("draft", True), ("draft", None), ("complete", False),
                             ("head_sha", "bad"), ("base_sha", "bad"), ("diff_digest", "bad"),
                             ("implementation_engines", []), ("implementation_engines", ["other"]),
                             ("review_threads", None),
                             ("review_threads", [{"isResolved": False, "priority": "P0"}]),
                             ("review_threads", [{"isResolved": False, "priority": "P1"}]),
                             ("review_threads", [{"isResolved": False}]),
                             ("review_threads", [{"isResolved": True, "disposition": "rejected"}])):
            with self.subTest(field=field, value=value):
                self.assertTrue(GATE.verify({**self.pr, field: value}, self.policy))

    def test_malformed_records_cannot_satisfy_review(self):
        for records in (None, {}, "codex", [None], [True], ["codex"], [[]]):
            with self.subTest(records=records):
                self.pr["independent_records"] = records
                self.assertTrue(GATE.verify(self.pr, self.policy))

    def test_bot_mode_retains_identity_completion_and_independent_claude_checks(self):
        policy = {"mode": "bot", "bot": {"login": "synthetic-bot", "id": "node", "app_id": 123},
                  "completion_pattern": r"^Review complete$"}
        self.assertTrue(GATE.verify(self.pr, policy))  # Local records do not replace the bot here.
        review = {"author": {**policy["bot"], "type": "Bot"}, "app_id": 123,
                  "state": "COMMENTED", "commit_id": self.pr["head_sha"],
                  "body": "Review complete", "receipt": self.record("codex")}
        self.pr["reviews"] = [review]
        self.pr["independent_records"] = []
        self.assertEqual(GATE.verify(self.pr, policy), [])
        for field, value in (("state", "DISMISSED"), ("body", "Pending"), ("app_id", None)):
            with self.subTest(field=field):
                pr = {**self.pr, "reviews": [{**review, field: value}]}
                self.assertTrue(GATE.verify(pr, policy))
        self.pr["implementation_engines"] = ["codex"]
        self.assertTrue(GATE.verify(self.pr, policy))
        record = self.record("claude")
        del record["findings_p0p1"], record["unresolved_p0p1"]
        self.pr["independent_records"] = [record]
        self.assertEqual(GATE.verify(self.pr, policy), [])  # Preserve the existing bot-mode contract.
        self.assertTrue(GATE.verify(self.pr, {**self.policy, "mode": "bot"}))


if __name__ == "__main__":
    unittest.main()
