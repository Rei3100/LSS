"""References that finish arriving after the BGM decision get one more decision with them (ep1: the album took 11 min,
the decision waited 4 min, and 11 of 14 released usages stayed published as redundant extractions)."""
from lss import application as app
from lss.bgm import reference_acquisition
from tests.test_application_bgm_memory_refresh import _outcome, _request


def _setup(monkeypatch, tmp_path, digests):
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"x")
    calls = []

    def once(request, *, state_root=None, analyze=None, memory_refresh=None):
        digest = digests[min(len(calls), len(digests) - 1)]
        calls.append(digest)
        key = app._bgm_candidate_run_key(app.file_sha256(source), app._bgm_candidate_recipe()["recipe_id"], digest)
        run = tmp_path / "state" / "bgm-candidates" / key / "run" / "run.json"
        run.parent.mkdir(parents=True, exist_ok=True)
        run.write_text("{}", encoding="utf-8")
        return _outcome(source)

    later = {"digest": digests[-1]}
    monkeypatch.setattr(app, "_run_bgm_candidate_request_once", once)
    monkeypatch.setattr(app, "_registered_references",
                        lambda root, **kw: ([], later["digest"], [], {}))
    monkeypatch.setattr(reference_acquisition, "wait_active", lambda *a, **k: True)
    return source, calls


def test_references_that_arrive_after_the_decision_trigger_one_more_decision(monkeypatch, tmp_path):
    source, calls = _setup(monkeypatch, tmp_path, [None, "album"])
    app._run_bgm_candidate_request(_request(source), state_root=tmp_path / "state")
    assert calls == [None, "album"]


def test_unchanged_references_decide_once(monkeypatch, tmp_path):
    source, calls = _setup(monkeypatch, tmp_path, ["album"])
    app._run_bgm_candidate_request(_request(source), state_root=tmp_path / "state")
    assert calls == ["album"]


def test_a_failed_first_run_is_returned_without_a_second(monkeypatch, tmp_path):
    source, calls = _setup(monkeypatch, tmp_path, [None, "album"])
    failed = app.ProcessingOutcome(disposition=app.ResultDisposition.FAILED, error_code="x", message="x")
    monkeypatch.setattr(app, "_run_bgm_candidate_request_once", lambda *a, **k: calls.append("f") or failed)
    assert app._run_bgm_candidate_request(_request(source), state_root=tmp_path / "state") is failed
    assert calls == ["f"]
