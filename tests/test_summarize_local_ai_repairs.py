import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "scripts" / "summarize_local_ai_repairs.py"
spec = importlib.util.spec_from_file_location("summarize_local_ai_repairs", MODULE)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def report(tmp_path, rows, *, attempted=None, status=None, fence=True):
    path = tmp_path / "report.json"
    ids = [r["id"] for r in rows]
    path.write_text(json.dumps({"repairs": rows, "attempted_repair_ids": ids if attempted is None else attempted,
                                "status": status, "fence_adaptation_enabled": fence}), encoding="utf-8")
    return path


def row(id, passed, *, candidate=False):
    return {"id": id, ("candidate" if candidate else "validation"): {"pass": passed}}


def test_common_optional_are_separate_partial_population_and_fence(tmp_path):
    p = report(tmp_path, [row(summary.COMMON[0], True), row(summary.OPTIONAL, False, candidate=True)], fence=False)
    got = summary.summarize(p)
    assert got["common5"]["attempts"] == 1 and got["common5"]["accepted"] == 1
    assert got["optional6"]["attempts"] == 1 and got["optional6"]["accepted"] == 0
    assert got["attempts"] == 2 and got["fence_adaptation_enabled"] is False
    assert got["source"] == str(p.resolve()) and len(got["report_sha256"]) == 64


def test_admission_refusal_is_zero_attempts_not_failed_population(tmp_path):
    p = report(tmp_path, [], attempted=[], status="unmeasured_admission_refused")
    got = summary.summarize(p)
    assert got["attempts"] == 0 and got["common5"]["attempts"] == 0
    assert got["status"] == "unmeasured_admission_refused"


@pytest.mark.parametrize("rows,attempted", [
    ([row(summary.COMMON[0], True), row(summary.COMMON[0], True)], None),
    ([row(summary.COMMON[0], True)], [summary.COMMON[1]]),
    ([{"id": "mystery", "validation": {"pass": True}}], None),
    ([{"id": summary.COMMON[0], "validation": {"pass": 1}}], None),
    ([{"id": summary.COMMON[0], "validation": {"pass": True}, "candidate": {"pass": False}}], None),
])
def test_rejects_duplicate_unknown_or_malformed_flags(tmp_path, rows, attempted):
    p = report(tmp_path, rows, attempted=attempted)
    with pytest.raises(ValueError):
        summary.summarize(p)


def test_missing_outcome_is_rejected(tmp_path):
    p = report(tmp_path, [{"id": summary.COMMON[0], "status": "accepted"}])
    with pytest.raises(ValueError, match="missing evaluator outcome"):
        summary.summarize(p)


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    report_path = report(tmp_path, [row(summary.COMMON[0], True)])
    out = Path(__file__).parents[1] / ".cache" / "local-ai" / "test-summary-existing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["summarize_local_ai_repairs", str(report_path), "--out", str(out)])
    try:
        with pytest.raises(FileExistsError):
            summary.main()
        assert out.read_text(encoding="utf-8") == "preserve me"
    finally:
        out.unlink(missing_ok=True)
