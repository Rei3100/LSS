"""削除する道具の安全性を、隔離した状態領域で確かめる。"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "disk_cleanup.py"


@pytest.fixture
def cleanup(tmp_path, monkeypatch):
    """本物のリポジトリを触らないよう、ROOTを隔離した写しへ差し替える。"""
    spec = importlib.util.spec_from_file_location("disk_cleanup_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import storage_tiers
    from tests.machine_config_fixtures import machine_policy
    monkeypatch.setattr(storage_tiers, "load_policy", lambda: machine_policy(tmp_path))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "tracked_files", lambda prefix=".state": list(tracked))
    tracked: list[Path] = []
    module.tracked_list = tracked
    (tmp_path / ".state" / "validation").mkdir(parents=True)
    (tmp_path / ".providers").mkdir()
    return module


def comparison(cleanup, name: str, *, media: int = 1, records: dict | None = None) -> Path:
    directory = Path(cleanup.ROOT) / ".state/validation" / name
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(media):
        (directory / f"case-{index}.wav").write_bytes(b"audio" * 100)
    (directory / "plan.json").write_text(json.dumps(
        records if records is not None else {"status": "FROZEN_BEFORE_RESULTS"},
        ensure_ascii=False), encoding="utf-8")
    return directory


def test_unreferenced_work_is_listed_and_only_removed_with_apply(cleanup):
    directory = comparison(cleanup, "old-branch-r1")
    rows, unreadable = cleanup.survey()
    assert unreadable == []
    assert [row["path"] for row in cleanup.prunable(rows)] == [".state/validation/old-branch-r1"]
    cleanup.prune(False, 0)
    assert directory.is_dir()
    cleanup.prune(True, 0)
    assert not directory.exists()
    assert list((Path(cleanup.ROOT) / ".state/validation").glob("disk-cleanup-*.json"))


def test_work_referenced_from_outside_is_never_removed(cleanup):
    directory = comparison(cleanup, "used-branch-r1")
    pointer = Path(cleanup.ROOT) / ".state/validation" / "current-state.json"
    pointer.write_text(json.dumps({"inputs": [str(directory / "case-0.wav")]}), encoding="utf-8")
    rows, _ = cleanup.survey()
    assert cleanup.prunable(rows) == []
    cleanup.prune(True, 0)
    assert (directory / "case-0.wav").is_file()


def test_a_directory_that_only_points_at_itself_is_still_removable(cleanup):
    directory = comparison(cleanup, "self-referencing-r1")
    (directory / "inputs.json").write_text(json.dumps(
        {"inputs": [str(directory / "case-0.wav")]}), encoding="utf-8")
    rows, _ = cleanup.survey()
    assert [row["path"] for row in cleanup.prunable(rows)] == [".state/validation/self-referencing-r1"]


def test_a_comparison_with_a_conclusion_record_is_kept_even_without_references(cleanup):
    concluded = comparison(cleanup, "concluded-r1")
    (concluded / "report.json").write_text(json.dumps({"disposition": "development_survivor"}),
                                           encoding="utf-8")
    unfinished = comparison(cleanup, "unfinished-r1")
    rows, _ = cleanup.survey()
    assert [row["path"] for row in cleanup.prunable(rows)] == [".state/validation/unfinished-r1"]
    cleanup.prune(True, 0)
    assert (concluded / "report.json").is_file() and (concluded / "case-0.wav").is_file()
    assert not unfinished.exists()


def test_tracked_and_protected_work_is_never_removed(cleanup):
    tracked_dir = comparison(cleanup, "committed-r1")
    cleanup.tracked_list.append(tracked_dir / "plan.json")
    archive = comparison(cleanup, "downloaded-corpus-r1", media=0)
    (archive / "dataset.zip").write_bytes(b"archive")
    inputs = comparison(cleanup, "case-inputs")
    frozen = comparison(cleanup, "frozen-runtime-r1")
    rows, _ = cleanup.survey()
    cleanup.prune(True, 0)
    for directory in (tracked_dir, archive, inputs, frozen):
        assert directory.is_dir(), directory
    assert {row["path"] for row in rows} >= {".state/validation/committed-r1"}


def test_retire_removes_media_of_rejected_comparisons_and_keeps_the_record(cleanup):
    rejected = comparison(cleanup, "rejected-r1", media=3,
                          records={"status": "complete_development_comparison",
                                   "disposition": "reject_general_routing"})
    survivor = comparison(cleanup, "survivor-r1", media=2,
                          records={"status": "complete_development_comparison",
                                   "disposition": "development_survivor"})
    cleanup.retire(False)
    assert len(list(rejected.glob("*.wav"))) == 3
    cleanup.retire(True)
    assert list(rejected.glob("*.wav")) == [] and (rejected / "plan.json").is_file()
    assert len(list(survivor.glob("*.wav"))) == 2
    record = list((Path(cleanup.ROOT) / ".state/validation").glob("media-retirement-*.json"))
    assert len(record) == 1
    written = json.loads(record[0].read_text(encoding="utf-8"))
    assert written["applied"] is True and written["classes"][0]["files"] == 3


def test_retire_keeps_media_that_a_live_comparison_still_uses(cleanup):
    rejected = comparison(cleanup, "rejected-r2", media=2,
                          records={"decision": "REJECTED_EXACT_CONFIGURATION"})
    live = Path(cleanup.ROOT) / ".state/validation" / "live-r1"
    live.mkdir()
    (live / "plan.json").write_text(json.dumps(
        {"inputs": [str(rejected / "case-0.wav")]}), encoding="utf-8")
    cleanup.retire(True)
    assert (rejected / "case-0.wav").is_file() and not (rejected / "case-1.wav").exists()


def test_nothing_outside_the_allowed_roots_is_ever_considered(cleanup):
    outside = Path(cleanup.ROOT) / "lss" / "bgm"
    outside.mkdir(parents=True)
    (outside / "keep.wav").write_bytes(b"audio")
    rows, _ = cleanup.survey()
    assert all(row["path"].startswith((".state/validation/", ".providers", ".state/work"))
               for row in rows)
    cleanup.prune(True, 0)
    cleanup.retire(True)
    assert (outside / "keep.wav").is_file()


def dev_report(cleanup, name: str, *, when: float) -> Path:
    directory = Path(cleanup.ROOT) / ".state/validation/dev" / name
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / "report.json"
    report.write_text(json.dumps({"path": str(report), "status": "PASSED"}), encoding="utf-8")
    os.utime(directory, (when, when))
    return directory


def test_old_test_run_records_are_trimmed_but_the_newest_and_referenced_are_kept(cleanup):
    reports = [dev_report(cleanup, f"run-{index}", when=1000 + index) for index in range(6)]
    pointer = Path(cleanup.ROOT) / ".state/validation" / "current-state.json"
    pointer.write_text(json.dumps({"report": str(reports[0] / "report.json")}), encoding="utf-8")
    stale = cleanup.stale_dev_reports(3)
    assert {path.name for path in stale} == {"run-1", "run-2"}
    cleanup.prune(True, 0, 3)
    assert reports[0].is_dir() and reports[5].is_dir()
    assert not reports[1].exists() and not reports[2].exists()


def test_a_tracked_test_run_record_is_never_removed_from_the_working_tree(cleanup):
    kept = dev_report(cleanup, "committed-run", when=1000)
    [dev_report(cleanup, f"loose-{index}", when=1001 + index) for index in range(3)]
    cleanup.tracked_list.append(kept / "report.json")
    assert kept not in cleanup.stale_dev_reports(1)
    cleanup.prune(True, 0, 1)
    assert (kept / "report.json").is_file()
