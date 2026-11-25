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
    monkeypatch.delenv("LSS_DEV_STATE", raising=False)
    root = (tmp_path / "checkout").resolve()
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "tracked_files", lambda prefix=".state": list(tracked))
    tracked: list[Path] = []
    module.tracked_list = tracked
    (root / ".state" / "validation").mkdir(parents=True)
    (root / ".providers").mkdir()
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


def test_validation_media_gc_refuses_state_outside_the_checkout(cleanup, tmp_path, monkeypatch):
    state = tmp_path / "other-checkout" / ".state"
    stem = state / "validation" / "case" / "music.wav"
    stem.parent.mkdir(parents=True)
    stem.write_bytes(b"outside media")
    monkeypatch.setenv("LSS_DEV_STATE", str(state))

    rows, unreadable = cleanup.validation_media_candidates()
    assert unreadable == []
    assert [(row["path"], row["class"]) for row in rows] == [(stem.resolve().as_posix(), "RETIRE")]
    assert cleanup.validation_media_gc(False, quiet=True) == 0
    with pytest.raises(SystemExit, match="owns them"):
        cleanup.validation_media_gc(True, quiet=True)
    assert stem.read_bytes() == b"outside media"
    assert not list(stem.parent.parent.glob("media-gc-*.json"))


def _directory_link(link, target):
    from storage_tiers import make_junction

    try:
        make_junction(link, target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory links are unavailable: {exc}")


@pytest.mark.parametrize("base_name", [".providers", ".state/work", ".state/validation/dev"])
def test_prune_skips_a_linked_cleanup_base(cleanup, tmp_path, base_name):
    outside = tmp_path / "outside"
    candidate = outside / "old-run"
    candidate.mkdir(parents=True)
    payload = candidate / "keep.wav"
    payload.write_bytes(b"outside media")
    base = cleanup.ROOT / base_name
    if base.is_dir():
        base.rmdir()
    _directory_link(base, outside)

    assert cleanup.prune(True, limit=0, keep_reports=0) == 0
    assert payload.read_bytes() == b"outside media"


def test_prune_refuses_a_linked_validation_receipt_directory(cleanup, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    validation = cleanup.ROOT / ".state/validation"
    validation.rmdir()
    _directory_link(validation, outside)

    with pytest.raises(cleanup.StorageMutationRefused, match="receipt path"):
        cleanup.prune(True, limit=0, keep_reports=0)
    assert list(outside.iterdir()) == []


def test_provider_media_gc_refuses_media_behind_a_linked_root(cleanup, tmp_path):
    outside = tmp_path / "outside"
    run = outside / "run"
    run.mkdir(parents=True)
    (run / "plan.json").write_text("{}", encoding="utf-8")
    stem = run / "keep.wav"
    stem.write_bytes(b"outside media")
    providers = cleanup.ROOT / ".providers"
    providers.rmdir()
    _directory_link(providers, outside)

    with pytest.raises(SystemExit, match="keep.wav"):
        cleanup.provider_media_gc(True, quiet=True)
    assert stem.read_bytes() == b"outside media"


def test_retire_keeps_the_target_of_linked_media(cleanup, tmp_path):
    rejected = comparison(cleanup, "rejected", records={"decision": "REJECTED"})
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside media")
    link = rejected / "linked.wav"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")

    assert cleanup.retire(True) == 0
    assert outside.read_bytes() == b"outside media"
    assert link.is_symlink()
    assert not (rejected / "case-0.wav").exists()
