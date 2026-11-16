"""保留分岐更新前の一覧を失わず、同じ再開・破損・保存失敗を安全に扱う。"""
import hashlib
import json

import pytest

from lss import application
from tests.test_application_bgm_reference_residual import bundle, case, execute


def test_adding_alternatives_keeps_the_previous_held_list_readable_without_generation(bundle, monkeypatch):
    primary = execute(bundle, attach=False)
    before = primary.record_path.read_bytes()
    latest = execute(bundle)
    assert latest.disposition == application.ResultDisposition.HELD, latest.message
    history = primary.record_path.parent / "outcome-history" / hashlib.sha256(before).hexdigest() / "application-outcome.json"
    assert history.read_bytes() == before
    monkeypatch.setattr(application, "run_bgm_development", lambda *a, **k: pytest.fail("履歴は読戻しだけ"))
    old = application.read_bgm_development_outcome(history)
    assert old.disposition == application.ResultDisposition.HELD, old.message
    assert old.held_artifacts == primary.held_artifacts
    assert old.held_alternative_artifacts == old.held_reference_residual_artifacts == ()
    assert old.outputs == () and old.zero_result is None


def test_legacy_record_without_plan_remains_readable_and_catalogued_after_archiving(bundle):
    from lss.assets import held_catalog
    previous = execute(bundle, attach=False)
    legacy = json.loads(previous.record_path.read_bytes())
    legacy.pop("plan")
    before = (json.dumps(legacy, ensure_ascii=False, indent=2) + "\n").encode()
    previous.record_path.write_bytes(before)
    assert application.read_bgm_development_outcome(previous.record_path).disposition == application.ResultDisposition.HELD
    latest = execute(bundle)
    assert latest.disposition == application.ResultDisposition.HELD, latest.message
    history = previous.record_path.parent / "outcome-history" / hashlib.sha256(before).hexdigest() / "application-outcome.json"
    outcome = application.read_bgm_development_outcome(history)
    assert outcome.disposition == application.ResultDisposition.HELD, outcome.message
    assert outcome.held_artifacts == previous.held_artifacts
    assert outcome.outputs == () and outcome.zero_result is None
    [snapshot] = held_catalog.snapshots([history])
    assert snapshot.path == history and snapshot.record == legacy
    assert len(snapshot.candidates) == len(previous.held_artifacts)
    assert history.read_bytes() == before


def test_identical_resume_does_not_create_another_history_snapshot(bundle):
    previous = execute(bundle, attach=False)
    assert previous.disposition == application.ResultDisposition.HELD
    assert execute(bundle, attach=False).disposition == application.ResultDisposition.HELD
    folder = previous.record_path.parent / "outcome-history"
    assert not folder.exists()
    assert execute(bundle).disposition == application.ResultDisposition.HELD
    snapshots = list(folder.rglob("application-outcome.json"))
    assert len(snapshots) == 1
    assert execute(bundle, attach=False).disposition == application.ResultDisposition.HELD
    assert list(folder.rglob("application-outcome.json")) == snapshots


def test_history_creation_failure_preserves_the_current_outcome(bundle, monkeypatch):
    previous = execute(bundle, attach=False)
    before = previous.record_path.read_bytes()
    def failed_link(*a):
        raise OSError("履歴の保存容量がありません")
    monkeypatch.setattr(application.os, "link", failed_link)
    result = execute(bundle)
    assert result.disposition == application.ResultDisposition.FAILED
    assert previous.record_path.read_bytes() == before
    assert not list(previous.record_path.parent.rglob("*.tmp"))


def test_concurrent_update_during_history_creation_is_not_replaced(bundle, monkeypatch):
    previous = execute(bundle, attach=False)
    writer = application._archive_outcome
    updated = []
    def concurrent(path, snapshot, payload):
        writer(path, snapshot, payload)
        newer = {**payload, "concurrent_marker": "newer"}
        data = (json.dumps(newer, ensure_ascii=False, indent=2) + "\n").encode()
        path.write_bytes(data)
        updated.append(data)
    monkeypatch.setattr(application, "_archive_outcome", concurrent)
    result = execute(bundle)
    assert result.disposition == application.ResultDisposition.FAILED
    assert previous.record_path.read_bytes() == updated[0]


def test_damaged_history_cannot_be_read_as_the_digest_in_its_location(bundle):
    previous = execute(bundle, attach=False)
    assert execute(bundle).disposition == application.ResultDisposition.HELD
    [history] = (previous.record_path.parent / "outcome-history").rglob("application-outcome.json")
    history.write_bytes(history.read_bytes() + b"\n")
    assert application.read_bgm_development_outcome(history).disposition == application.ResultDisposition.FAILED


def test_input_change_during_history_write_preserves_the_current_record(bundle, monkeypatch):
    previous = execute(bundle, attach=False)
    before = previous.record_path.read_bytes()
    archive = application._archive_outcome
    def changed(path, snapshot, payload):
        result = archive(path, snapshot, payload)
        source = bundle[0].sources[0]
        source.write_bytes(source.read_bytes() + b"late change")
        return result
    monkeypatch.setattr(application, "_archive_outcome", changed)
    result = execute(bundle)
    assert result.disposition == application.ResultDisposition.FAILED
    assert previous.record_path.read_bytes() == before
