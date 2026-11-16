import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dev_full_source_followup.py"
SPEC = importlib.util.spec_from_file_location("dev_full_source_followup", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompletionGateTests(unittest.TestCase):
    def test_running_process_waits(self):
        self.assertEqual("wait", MODULE.decide(
            {"status": "RUNNING"}, {"status": "RUNNING", "exit_code": None}, True))

    def test_native_success_waits_for_launcher_zero(self):
        self.assertEqual("wait", MODULE.decide(
            {"status": MODULE.SUCCESS, "gpu_executed": True},
            {"status": "RUNNING", "exit_code": None}, True))

    def test_exited_successful_child_waits_for_launcher_finalization(self):
        run = {"status": MODULE.SUCCESS, "gpu_executed": True}
        launcher = {"status": "RUNNING", "exit_code": None}
        action, _, finalization = MODULE.bounded_decision(run, launcher, False, 0, 0)
        self.assertEqual("wait_for_launcher_finalization", action)
        self.assertEqual(1, finalization)
        action, _, finalization = MODULE.bounded_decision(run, launcher, False, 0, finalization)
        self.assertEqual("wait_for_launcher_finalization", action)
        self.assertEqual(2, finalization)
        action, _, _ = MODULE.bounded_decision(run, launcher, False, 0, finalization)
        self.assertEqual("refuse", action)

    def test_launcher_completion_after_finalization_wait_invokes(self):
        action, transient, finalization = MODULE.bounded_decision(
            {"status": MODULE.SUCCESS, "gpu_executed": True},
            {"status": "RUNNING", "exit_code": None}, False, 0, 1)
        self.assertEqual("wait_for_launcher_finalization", action)
        action, transient, finalization = MODULE.bounded_decision(
            {"status": MODULE.SUCCESS, "gpu_executed": True},
            {"status": "COMPLETED", "exit_code": 0}, False, transient, finalization)
        self.assertEqual("invoke", action)

    def test_exact_dual_success_invokes(self):
        self.assertEqual("invoke", MODULE.decide(
            {"status": MODULE.SUCCESS, "gpu_executed": True},
            {"status": "COMPLETED", "exit_code": 0}, False))

    def test_failed_or_missing_receipt_refuses(self):
        cases = [
            ({"status": MODULE.SUCCESS, "gpu_executed": True}, {"status": "FAILED", "exit_code": 1}, False),
            ({"status": "DEV_ONLY_FULL_SOURCE_SFC_REGEN_FAILED"}, {"status": "COMPLETED", "exit_code": 1}, False),
            ({"status": MODULE.SUCCESS, "gpu_executed": True}, {"status": "COMPLETED", "exit_code": 2}, False),
        ]
        for run, launcher, alive in cases:
            with self.subTest(run=run, launcher=launcher):
                self.assertEqual("refuse", MODULE.decide(run, launcher, alive))

    def test_missing_receipt_is_marked_for_bounded_retry(self):
        launcher = {"status": "RUNNING", "exit_code": None}
        action, count, _ = MODULE.bounded_decision(None, launcher, True, 0, 0)
        self.assertEqual(("retry_receipts", 1), (action, count))
        action, count, _ = MODULE.bounded_decision(None, launcher, True, count, 0)
        self.assertEqual(("retry_receipts", 2), (action, count))
        action, _, _ = MODULE.bounded_decision(None, launcher, True, count, 0)
        self.assertEqual("refuse", action)

    def test_missing_process_refuses_even_while_launcher_says_running(self):
        self.assertEqual("refuse", MODULE.decide(
            {"status": "RUNNING"}, {"status": "RUNNING", "exit_code": None}, False))

    def test_cim_creation_timestamp_parses_to_utc(self):
        stamp = MODULE.parse_expected_creation("20261021054601.923286+000")
        self.assertEqual("2026-10-21T05:46:01.923286+00:00", stamp.isoformat())


if __name__ == "__main__":
    unittest.main()
