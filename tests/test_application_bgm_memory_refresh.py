"""The normal BGM entry revisits one linked older source without granting new adoption authority."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lss import application as app
from lss.bgm import memory_refresh as refresh
from lss.output import result_report
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


@pytest.mark.parametrize("disposition", [app.ResultDisposition.PUBLISHED, app.ResultDisposition.HELD,
                                        app.ResultDisposition.FAILED])
def test_normal_entry_revisits_one_older_source_and_preserves_foreground(tmp_path, monkeypatch, disposition):
    foreground, older = tmp_path / "new.mkv", tmp_path / "old.mkv"
    initial = _outcome(foreground)
    plan = {"source_path": str(older), "source_sha256": "b" * 64, "token": "c" * 64}
    calls, finished = [], []
    def run(request, **kwargs):
        calls.append((request, kwargs))
        if len(calls) == 1:
            return initial
        return replace(_outcome(older, "b" * 64), disposition=disposition,
                       error_code="processing_failed" if disposition is app.ResultDisposition.FAILED else None)
    monkeypatch.setattr(app, "_run_bgm_candidate_request", run)
    monkeypatch.setattr(refresh, "pending", lambda *a, **k: [plan, {**plan, "token": "d" * 64}])
    monkeypatch.setattr(refresh, "claim", lambda *a: True)
    monkeypatch.setattr(refresh, "finish", lambda root, p, result: finished.append(result))
    result = app.run_bgm_candidate_request(_request(foreground), state_root=tmp_path)
    assert len(calls) == 2
    assert calls[1][0].sources == (older,)
    assert calls[1][1]["memory_refresh"] == plan
    assert result.outputs == initial.outputs and result.disposition == initial.disposition
    assert finished[0]["ok"] is (disposition is not app.ResultDisposition.FAILED)


def test_identical_foreground_at_old_path_does_not_hide_older_source(tmp_path, monkeypatch):
    initial = _outcome(tmp_path / "new.mkv")
    same = {"source_path": str(tmp_path / "old-location.mkv"), "source_sha256": "a" * 64, "token": "1" * 64}
    other = {"source_path": str(tmp_path / "other.mkv"), "source_sha256": "b" * 64, "token": "2" * 64}
    claimed = []
    monkeypatch.setattr(refresh, "pending", lambda *a, **k: [same, other])
    monkeypatch.setattr(refresh, "claim", lambda root, p: claimed.append(p) or False)
    assert app._refresh_bgm_from_memory(initial, _request(tmp_path / "new.mkv"), state_root=tmp_path) is initial
    assert claimed == [other]


def test_optional_revisit_survives_a_malformed_link_and_finishes_once(tmp_path, monkeypatch):
    from lss.bgm import cue_memory
    from tests.test_bgm_memory_refresh import _setup

    state = _setup(tmp_path, monkeypatch)
    path = cue_memory.root(state) / cue_memory.RECORD_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["links"].insert(0, {"paired": True, "waveform_explained": "broken"})
    path.write_text(json.dumps(record), encoding="utf-8")
    foreground = tmp_path / "foreground.mkv"
    foreground.write_bytes(b"foreground")
    initial = _outcome(foreground, "f" * 64)
    calls = []

    def revisit(request, **kwargs):
        calls.append((request, kwargs))
        return _outcome(request.sources[0])

    monkeypatch.setattr(app, "_run_bgm_candidate_request", revisit)
    result = app._refresh_bgm_from_memory(initial, _request(foreground), state_root=state)
    assert len(calls) == 1
    assert calls[0][1]["memory_refresh"]["target_member"] == "bgm-cue:target"
    assert result.outputs == initial.outputs and result.disposition == initial.disposition
    assert result.product_result["related_bgm_refresh"]["ok"] is True
    assert refresh.pending(state, recipe_id=app._bgm_candidate_recipe()["recipe_id"]) == []


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


def test_report_does_not_call_failed_refresh_completed():
    projection = {"state": "PUBLISHED", "published_count": 1, "paths": [], "assets": [],
                  "related_bgm_refresh": {"source": "old.mkv", "ok": False, "disposition": "FAILED"}}
    rendered = result_report.render(Path("new.mkv"), "bgm", projection)
    assert "再評価は完了できませんでした" in rendered
    assert "未確定BGMも再評価しました" not in rendered


def test_ui_reuse_does_not_hide_pending_older_source_after_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "pending", lambda *a, **k: [{"source_sha256": "b" * 64}])
    # This return happens before reference discovery or any processing. The regular entry reuses the
    # foreground audio, then services the stale-owner claim through its normal guarded path.
    monkeypatch.setattr(app, "_registered_references", lambda *a, **k: pytest.fail("reuse shortcut continued"))
    assert app.existing_bgm_candidates(tmp_path / "foreground.mkv", "a" * 64, state_root=tmp_path) is None
