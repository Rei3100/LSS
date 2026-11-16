"""原候補・Music代替・参照残差を併存させ、任意の分岐の破損と移動から原候補を守る。"""
import json
from pathlib import Path
import shutil

import pytest

from lss import application
from lss.assets import relocate
from lss.validation import held_bgm_alternatives, reference_residual_episode as stage
from tests.test_reference_residual_episode import case


@pytest.fixture
def bundle(case, tmp_path):
    run, preparation, recipe = case
    residual_plan = stage.freeze(run, preparation, tmp_path / "residual", recipe=recipe)
    residual = stage.run(residual_plan)
    alternatives = held_bgm_alternatives.run(held_bgm_alternatives.freeze(run, tmp_path / "alternatives"))
    request = application.ProcessingRequest.from_paths([tmp_path / "source.wav"], application.UserIntent.EXTRACT_BGM_CUES)
    return request, run.parent / "plan.json", run, residual, alternatives


def execute(bundle, *, attach=True):
    request, plan, run, residual, alternatives = bundle
    return application.run_bgm_development(request, plan, runner=lambda *a, **k: run,
        alternative_run_path=alternatives if attach else None,
        reference_residual_run_path=residual if attach else None)


def test_three_held_views_coexist_through_readback_and_resume(bundle):
    outcome = execute(bundle)
    assert outcome.disposition == application.ResultDisposition.HELD, outcome.message
    assert len(outcome.held_artifacts) == len(outcome.held_alternative_artifacts) == len(outcome.held_reference_residual_artifacts) == 1
    assert outcome.outputs == () and outcome.zero_result is None
    record = json.loads(outcome.record_path.read_bytes())
    view = record["held_reference_residuals"]
    assert view["semantic_role"] == view["selection_state"] == "unresolved" and view["publication_allowed"] is False
    before = outcome.record_path.read_bytes()
    readback = application.read_bgm_development_outcome(outcome.record_path)
    assert readback.held_reference_residual_artifacts == outcome.held_reference_residual_artifacts
    assert readback.held_alternative_artifacts == outcome.held_alternative_artifacts
    assert readback.reference_residual_evidence_path == bundle[3]
    assert outcome.record_path.read_bytes() == before
    resumed = execute(bundle, attach=False)
    assert resumed.disposition == application.ResultDisposition.HELD
    assert resumed.held_reference_residual_artifacts == outcome.held_reference_residual_artifacts
    assert resumed.held_alternative_artifacts == outcome.held_alternative_artifacts


@pytest.mark.parametrize("damage", ["report", "audio", "signal", "plan", "parent"])
def test_invalid_explicit_reference_preserves_previous_outcome(bundle, damage):
    previous = execute(bundle)
    assert previous.disposition == application.ResultDisposition.HELD, previous.message
    before = previous.record_path.read_bytes()
    report = json.loads(bundle[3].read_bytes())
    target = {"report": bundle[3], "audio": previous.held_reference_residual_artifacts[0],
              "signal": Path(report["signal"]["path"]), "plan": bundle[3].parent / "plan.json",
              "parent": bundle[2]}[damage]
    target.write_bytes(target.read_bytes() + b"changed")
    result = execute(bundle)
    assert result.disposition == application.ResultDisposition.FAILED
    assert previous.record_path.read_bytes() == before


@pytest.mark.parametrize("damage", ["report", "audio", "signal", "plan", "malformed"])
def test_invalid_optional_reference_keeps_originals_and_music_alternatives(bundle, damage):
    previous = execute(bundle)
    assert previous.disposition == application.ResultDisposition.HELD, previous.message
    before = previous.record_path.read_bytes()
    report = json.loads(bundle[3].read_bytes())
    target = {"report": bundle[3], "audio": previous.held_reference_residual_artifacts[0],
              "signal": Path(report["signal"]["path"]), "plan": bundle[3].parent / "plan.json",
              "malformed": bundle[3]}[damage]
    target.write_bytes(b"[]" if damage == "malformed" else b"broken")
    readback = application.read_bgm_development_outcome(previous.record_path)
    assert readback.disposition == application.ResultDisposition.HELD, readback.message
    assert readback.held_artifacts == previous.held_artifacts
    assert readback.held_alternative_artifacts == previous.held_alternative_artifacts
    assert readback.held_reference_residual_artifacts == () and readback.reference_residual_evidence_path is None
    assert previous.record_path.read_bytes() == before
    resumed = execute(bundle, attach=False)
    assert resumed.disposition == application.ResultDisposition.HELD and resumed.held_reference_residual_artifacts == ()
    assert resumed.held_artifacts == previous.held_artifacts and resumed.held_alternative_artifacts == previous.held_alternative_artifacts


def test_late_reference_audio_change_cannot_replace_the_saved_result(bundle, monkeypatch):
    previous = execute(bundle)
    assert previous.disposition == application.ResultDisposition.HELD, previous.message
    before = previous.record_path.read_bytes()
    writer = application._write_outcome_atomic
    def late(path, payload, validate):
        def check():
            audio = previous.held_reference_residual_artifacts[0]
            audio.write_bytes(audio.read_bytes() + b"late")
            validate()
        writer(path, payload, check)
    monkeypatch.setattr(application, "_write_outcome_atomic", late)
    result = execute(bundle)
    assert result.disposition == application.ResultDisposition.FAILED
    assert previous.record_path.read_bytes() == before


def test_moved_reference_candidate_is_discovered_and_read_back_without_changing_origin(bundle, tmp_path):
    previous = execute(bundle)
    assert previous.disposition == application.ResultDisposition.HELD, previous.message
    before = previous.record_path.read_bytes()
    original = previous.held_reference_residual_artifacts[0]
    archive = tmp_path / "archive"; archive.mkdir()
    moved = archive / "renamed-reference-candidate.flac"
    shutil.move(original, moved)
    state = tmp_path / "state"
    moves = relocate.find_moved([archive], state_root=state, outcome_records=[previous.record_path])
    assert len(moves) == 1 and moves[0]["kind"] == "output"
    relocate.record_relocations(moves, state_root=state)
    readback = application.read_bgm_development_outcome(previous.record_path, state_root=state)
    assert readback.disposition == application.ResultDisposition.HELD, readback.message
    assert readback.held_reference_residual_artifacts == (moved,)
    assert readback.held_artifact_locations[original] == (moved,)
    assert previous.record_path.read_bytes() == before
