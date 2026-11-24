"""Cue memory re-evaluates only the source the user named; it never schedules another source."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lss import application as app
from lss.bgm import memory_refresh as refresh
from tests.test_application_bgm_candidates import _source, gpu  # noqa: F401


def _request(source):
    return app.ProcessingRequest.from_paths([source], app.UserIntent.EXTRACT_BGM_CUES)


def _outcome(source, sha="a" * 64):
    return app.ProcessingOutcome(disposition=app.ResultDisposition.PUBLISHED,
        outputs=(source.with_suffix(".flac"),), product_result={
            "state": "PUBLISHED", "published_count": 1, "source_sha256": sha, "paths": [], "assets": []})


def test_run_identity_preserves_existing_reuse_and_separates_memory_inputs():
    old = app._bgm_candidate_run_key("source", "recipe")
    assert old == app._bgm_candidate_run_key("source", "recipe", memory_refresh=None)
    assert old != app._bgm_candidate_run_key("source", "recipe", memory_refresh="first")
    assert app._bgm_candidate_run_key("source", "recipe", memory_refresh="first") != app._bgm_candidate_run_key(
        "source", "recipe", memory_refresh="second")


def _stub_entry(monkeypatch, plans, results):
    calls, finished, claimed = [], [], []
    def run(request, **kwargs):
        calls.append((request, kwargs))
        return results[min(len(calls), len(results)) - 1]
    monkeypatch.setattr(app, "_run_bgm_candidate_request", run)
    monkeypatch.setattr(refresh, "pending", lambda *a, **k: plans)
    monkeypatch.setattr(refresh, "claim", lambda root, p: claimed.append(p) or True)
    monkeypatch.setattr(refresh, "finish", lambda root, p, result: finished.append(result))
    return calls, finished, claimed


def test_ordinary_entry_never_processes_a_source_the_user_did_not_name(tmp_path, monkeypatch):
    foreground = tmp_path / "new.mkv"
    foreground.write_bytes(b"new")
    other = {"source_path": str(tmp_path / "old.mkv"), "source_sha256": "b" * 64, "token": "c" * 64}
    initial = _outcome(foreground, app.file_sha256(foreground))
    calls, finished, claimed = _stub_entry(monkeypatch, [other], [initial])
    result = app.run_bgm_candidate_request(_request(foreground), state_root=tmp_path)
    assert [c[0].sources for c in calls] == [(foreground.resolve(),)]
    assert "memory_refresh" not in calls[0][1]
    assert claimed == [] and finished == []
    assert result is initial and "related_bgm_refresh" not in result.product_result


@pytest.mark.parametrize("disposition", [app.ResultDisposition.PUBLISHED, app.ResultDisposition.HELD])
def test_named_source_runs_its_pending_memory_reevaluation(tmp_path, monkeypatch, disposition):
    named = tmp_path / "old.mkv"
    named.write_bytes(b"old")
    sha = app.file_sha256(named)
    plan = {"source_path": str(tmp_path / "moved.mkv"), "source_sha256": sha, "token": "c" * 64}
    unrelated = {"source_path": str(tmp_path / "x.mkv"), "source_sha256": "b" * 64, "token": "d" * 64}
    refreshed = replace(_outcome(named, sha), disposition=disposition)
    calls, finished, claimed = _stub_entry(monkeypatch, [unrelated, plan], [refreshed])
    result = app.run_bgm_candidate_request(_request(named), state_root=tmp_path)
    assert len(calls) == 1 and calls[0][1]["memory_refresh"] == plan
    assert calls[0][0].sources == (named.resolve(),)   # the named path, not the plan's recorded path
    assert claimed == [plan] and finished[0]["ok"] is True
    assert result.outputs == refreshed.outputs and result.disposition is disposition


def test_failed_reevaluation_is_terminal_and_the_ordinary_run_answers(tmp_path, monkeypatch):
    named = tmp_path / "old.mkv"
    named.write_bytes(b"old")
    sha = app.file_sha256(named)
    plan = {"source_path": str(named), "source_sha256": sha, "token": "c" * 64}
    monkeypatch.setattr(refresh, "pending", lambda *a, **k: [plan])
    failed = app.ProcessingOutcome(disposition=app.ResultDisposition.FAILED, error_code="processing_failed")
    ordinary = _outcome(named, sha)
    calls = []
    def run(request, **kwargs):
        calls.append(kwargs)
        return failed if "memory_refresh" in kwargs else ordinary
    monkeypatch.setattr(app, "_run_bgm_candidate_request", run)
    result = app.run_bgm_candidate_request(_request(named), state_root=tmp_path)
    assert [("memory_refresh" in c) for c in calls] == [True, False]
    assert result.outputs == ordinary.outputs
    receipt = json.loads((tmp_path / refresh.DIRECTORY / f"{plan['token']}.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert not refresh.claim(tmp_path, plan)


def test_abandoned_reevaluation_returns_its_claim(tmp_path, monkeypatch):
    named = tmp_path / "old.mkv"
    named.write_bytes(b"old")
    plan = {"source_path": str(named), "source_sha256": app.file_sha256(named), "token": "c" * 64}
    monkeypatch.setattr(refresh, "pending", lambda *a, **k: [plan])
    calls = []
    def run(request, **kwargs):
        calls.append(kwargs)
        assert (tmp_path / refresh.DIRECTORY / f"{plan['token']}.json").is_file()   # claimed while running
        raise app.quitting.JobAbandonedError()
    monkeypatch.setattr(app, "_run_bgm_candidate_request", run)
    assert app.run_bgm_candidate_request(_request(named), state_root=tmp_path).error_code == "job_abandoned"
    assert len(calls) == 1
    # The process that quit is still alive (a GUI keeps running): the claim must not stay "running" behind it.
    assert not (tmp_path / refresh.DIRECTORY / f"{plan['token']}.json").exists()
    assert refresh.claim(tmp_path, plan)


def test_unexpected_reevaluation_error_is_a_failed_attempt_and_the_ordinary_run_answers(tmp_path, monkeypatch):
    named = tmp_path / "old.mkv"
    named.write_bytes(b"old")
    sha = app.file_sha256(named)
    plan = {"source_path": str(named), "source_sha256": sha, "token": "c" * 64}
    monkeypatch.setattr(refresh, "pending", lambda *a, **k: [plan])
    ordinary = _outcome(named, sha)
    calls = []
    def run(request, **kwargs):
        calls.append(kwargs)
        if "memory_refresh" in kwargs:
            raise RuntimeError("reference set changed")
        return ordinary
    monkeypatch.setattr(app, "_run_bgm_candidate_request", run)
    assert app.run_bgm_candidate_request(_request(named), state_root=tmp_path).outputs == ordinary.outputs
    receipt = json.loads((tmp_path / refresh.DIRECTORY / f"{plan['token']}.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed" and len(calls) == 2


def test_a_live_attempt_blocks_another_token_for_the_same_source(tmp_path):
    first = {"source_sha256": "a" * 64, "token": "1" * 64}
    assert refresh.claim(tmp_path, first)
    assert not refresh.claim(tmp_path, {"source_sha256": "a" * 64, "token": "2" * 64})
    assert refresh.claim(tmp_path, {"source_sha256": "b" * 64, "token": "3" * 64})
    refresh.finish(tmp_path, first, {"ok": True})
    assert refresh.claim(tmp_path, {"source_sha256": "a" * 64, "token": "2" * 64})


def test_a_failed_reference_recheck_keeps_the_published_result(tmp_path, monkeypatch):
    from lss.bgm import reference_acquisition
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"x")
    published = _outcome(source)
    monkeypatch.setattr(app, "_run_bgm_candidate_request_once", lambda *a, **k: published)
    monkeypatch.setattr(reference_acquisition, "wait_active", lambda *a, **k: True)
    def broken(*a, **k):
        raise ValueError("reference set unreadable")
    monkeypatch.setattr(app, "_registered_references", broken)
    assert app._run_bgm_candidate_request(_request(source), state_root=tmp_path) is published


def test_named_reevaluation_survives_a_malformed_link_and_finishes_once(tmp_path, monkeypatch):
    from lss.bgm import cue_memory
    from tests.test_bgm_memory_refresh import _setup

    state = _setup(tmp_path, monkeypatch)
    path = cue_memory.root(state) / cue_memory.RECORD_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["links"].insert(0, {"paired": True, "waveform_explained": "broken"})
    path.write_text(json.dumps(record), encoding="utf-8")
    named = tmp_path / "audio" / "episode.mkv"
    monkeypatch.setattr(app, "file_sha256", lambda path: "a" * 64)
    calls = []

    def run(request, **kwargs):
        calls.append((request, kwargs))
        return _outcome(request.sources[0])

    monkeypatch.setattr(app, "_run_bgm_candidate_request", run)
    result = app.run_bgm_candidate_request(_request(named), state_root=state)
    assert len(calls) == 1
    assert calls[0][1]["memory_refresh"]["target_member"] == "bgm-cue:target"
    assert result.disposition is app.ResultDisposition.PUBLISHED
    assert refresh.pending(state, recipe_id=app._bgm_candidate_recipe()["recipe_id"]) == []
    receipts = [json.loads(p.read_text(encoding="utf-8")) for p in (state / refresh.DIRECTORY).glob("*.json")]
    assert [r["status"] for r in receipts] == ["complete"]


def test_real_source_entry_reuses_token_run_and_restores_lss_tags(tmp_path, gpu, monkeypatch):
    from mutagen.flac import FLAC
    monkeypatch.setattr(refresh, "pending", lambda *a, **k: [])
    source = _source(tmp_path / "episode.wav")
    request = _request(source)
    state = tmp_path / "state"
    first = app.run_bgm_candidate_request(request, state_root=state)
    assert first.disposition is app.ResultDisposition.PUBLISHED, first.message
    own_title = FLAC(first.outputs[0])["TITLE"]
    tags = FLAC(first.outputs[0])
    tags["TITLE"] = ["Foreign title"]
    tags.save()
    prior_pcm = app.audio_identity.audio_identity(first.outputs[0])["pcm_sha256"]
    plan = {"source_sha256": app.file_sha256(source), "token": "e" * 64}
    new = app._run_bgm_candidate_request(request, state_root=state, memory_refresh=plan)
    assert new.disposition is app.ResultDisposition.PUBLISHED, new.message
    assert new.record_path != first.record_path
    assert json.loads(new.record_path.read_text(encoding="utf-8"))["memory_refresh"] == plan
    count = len(gpu.calls)
    again = app._run_bgm_candidate_request(request, state_root=state, memory_refresh=plan)
    assert again.reused and len(gpu.calls) == count
    assert app.audio_identity.audio_identity(again.outputs[0])["pcm_sha256"] == prior_pcm
    assert FLAC(again.outputs[0])["TITLE"] == own_title
    changed = {**plan, "source_sha256": "f" * 64, "token": "f" * 64}
    failed = app._run_bgm_candidate_request(request, state_root=state, memory_refresh=changed)
    assert failed.disposition is app.ResultDisposition.FAILED
    assert len(gpu.calls) == count


def test_ui_reuse_does_not_hide_a_pending_reevaluation_of_the_same_source(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "pending", lambda *a, **k: [{"source_sha256": "a" * 64}])
    # This return happens before reference discovery or any processing: the ordinary entry, started by the
    # user naming this source, services the pending re-evaluation.
    monkeypatch.setattr(app, "_registered_references", lambda *a, **k: pytest.fail("reuse shortcut continued"))
    assert app.existing_bgm_candidates(tmp_path / "foreground.mkv", "a" * 64, state_root=tmp_path) is None


def test_ui_reuse_ignores_other_sources_plans(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "pending", lambda *a, **k: [{"source_sha256": "b" * 64}])
    reached = []
    def references(*a, **k):
        reached.append(True)
        raise RuntimeError("stop after the memory check")
    monkeypatch.setattr(app, "_registered_references", references)
    with pytest.raises(RuntimeError):
        app.existing_bgm_candidates(tmp_path / "foreground.mkv", "a" * 64, state_root=tmp_path)
    assert reached == [True]


def test_a_stale_plan_cannot_requery_a_witness_another_attempt_finished(tmp_path):
    common = {"source_sha256": "a" * 64, "processing_recipe_id": "r", "memory_recipe_id": "m",
              "witness_pcm_sha256": "9" * 64}
    first = {**common, "token": "1" * 64, "external_pcm_sha256": ["9" * 64]}
    later = {**common, "token": "2" * 64, "external_pcm_sha256": ["8" * 64, "9" * 64]}   # unrelated growth
    assert refresh.claim(tmp_path, first)
    refresh.finish(tmp_path, first, {"ok": True})
    assert not refresh.claim(tmp_path, later)


@pytest.mark.parametrize("saved_file_present", [True, False])
def test_a_failed_later_reference_decision_keeps_a_first_result_whose_files_remain(tmp_path, monkeypatch,
                                                                                     saved_file_present):
    from lss.bgm import reference_acquisition
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"x")
    saved = _outcome(source)
    if saved_file_present:
        saved.outputs[0].write_bytes(b"flac")
    failed = app.ProcessingOutcome(disposition=app.ResultDisposition.FAILED, error_code="worker_stopped")
    calls = []
    def once(*a, **k):
        calls.append(k)
        return saved if len(calls) == 1 else failed
    monkeypatch.setattr(app, "_run_bgm_candidate_request_once", once)
    monkeypatch.setattr(reference_acquisition, "wait_active", lambda *a, **k: True)
    monkeypatch.setattr(app, "_registered_references", lambda *a, **k: ([], "album", [], {}))
    result = app._run_bgm_candidate_request(_request(source), state_root=tmp_path)
    assert len(calls) == 2   # the later decision ran
    assert result is (saved if saved_file_present else failed)


def test_s3_t9_deliberate_failure():
    assert False, "S3 T9: a deliberately failing test; the next push removes it"
