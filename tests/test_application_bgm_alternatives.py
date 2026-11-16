"""元候補と代替候補を同じ保留結果へ結び、誤った対応・再開・移動から来歴を守る。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from mutagen.flac import FLAC
import numpy as np
import pytest
import soundfile as sf

from lss import application
from lss.assets import relocate
from lss.bgm import held_results
from lss.harness.fetch import file_sha256
from lss.validation import held_bgm_alternatives as held
from tests.test_held_bgm_alternatives import upstream


@pytest.fixture
def case(tmp_path):
    run = upstream(tmp_path)
    plan = run.parent / "plan.json"
    alternative_plan = held.freeze(run, tmp_path / "alternatives")
    alternative_run = held.run(alternative_plan)
    request = application.ProcessingRequest.from_paths([tmp_path / "source.wav"], application.UserIntent.EXTRACT_BGM_CUES)
    return request, plan, run, alternative_plan, alternative_run


def _execute(case, *, attach=True):
    request, plan, run, _, alternative = case
    return application.run_bgm_development(request, plan, runner=lambda *a, **k: run,
                                           alternative_run_path=alternative if attach else None)


def test_original_and_alternative_are_retained_without_selection_or_publication(case):
    original = json.loads(case[2].read_bytes())
    parent_audio = Path(original["candidates"][0]["asset"]["path"])
    before = case[2].read_bytes(), parent_audio.read_bytes()
    outcome = _execute(case)
    assert outcome.disposition == application.ResultDisposition.HELD, outcome.message
    assert len(outcome.held_artifacts) == len(outcome.held_alternative_artifacts) == 1
    assert outcome.outputs == () and outcome.zero_result is None
    assert outcome.alternative_evidence_path == case[4]
    view = json.loads(outcome.record_path.read_bytes())["held_alternatives"]
    assert view["selection_state"] == view["semantic_role"] == "unresolved"
    assert view["artifacts"][0]["primary_candidate_asset_id"] == original["candidates"][0]["asset"]["candidate_asset_id"]
    assert (case[2].read_bytes(), parent_audio.read_bytes()) == before
    record_before = outcome.record_path.read_bytes()
    readback = application.read_bgm_development_outcome(outcome.record_path)
    assert readback.disposition == application.ResultDisposition.HELD and readback.reused
    assert readback.held_alternative_artifacts == outcome.held_alternative_artifacts
    assert outcome.record_path.read_bytes() == record_before
    resumed = _execute(case, attach=False)
    assert resumed.held_alternative_artifacts == outcome.held_alternative_artifacts


@pytest.mark.parametrize("damage", ["report", "candidate", "plan", "upstream_hash", "parent_id", "origin", "published"])
def test_invalid_explicit_alternatives_preserve_the_previous_product_outcome(case, damage):
    previous = _execute(case)
    assert previous.disposition == application.ResultDisposition.HELD
    before = previous.record_path.read_bytes()
    if damage == "candidate":
        path = previous.held_alternative_artifacts[0]
        path.write_bytes(path.read_bytes() + b"changed")
    elif damage == "report":
        case[4].write_bytes(b"[]")
    elif damage == "plan":
        case[3].write_bytes(b"null")
    elif damage == "upstream_hash":
        case[2].write_bytes(case[2].read_bytes() + b"\n")
    else:
        record = json.loads(case[4].read_bytes())
        if damage == "parent_id":
            record["candidates"][0]["primary_candidate_asset_id"] = "other"
        elif damage == "origin":
            record["candidates"][0]["origin"]["end_frame"] -= 1
        else:
            record["publication_state"] = "published"
        case[4].write_text(json.dumps(record), encoding="utf-8")
    result = _execute(case)
    assert result.disposition == application.ResultDisposition.FAILED
    assert previous.record_path.read_bytes() == before


@pytest.mark.parametrize("damage", ["report", "audio", "missing_audio", "plan"])
def test_stale_optional_alternatives_do_not_hide_original_held_cues(case, damage):
    previous = _execute(case)
    before = previous.record_path.read_bytes()
    target = {"report": case[4], "plan": case[3], "audio": previous.held_alternative_artifacts[0],
              "missing_audio": previous.held_alternative_artifacts[0]}[damage]
    if damage == "missing_audio":
        target.unlink()
    else:
        target.write_bytes(b"stale")
    readback = application.read_bgm_development_outcome(previous.record_path)
    assert readback.disposition == application.ResultDisposition.HELD
    assert readback.held_artifacts == previous.held_artifacts
    assert readback.held_alternative_artifacts == () and readback.alternative_evidence_path is None
    assert previous.record_path.read_bytes() == before
    resumed = _execute(case, attach=False)
    assert resumed.disposition == application.ResultDisposition.HELD
    assert resumed.held_artifacts == previous.held_artifacts and resumed.held_alternative_artifacts == ()


def test_changed_crop_cannot_be_accepted_by_updating_only_the_file_hash(case):
    outcome = _execute(case)
    path = outcome.held_alternative_artifacts[0]
    tags = dict(FLAC(path))
    sf.write(path, np.zeros((1024, 2)), 44100, subtype="PCM_24")
    metadata = FLAC(path)
    metadata.update(tags)
    metadata.save()
    record = json.loads(case[4].read_bytes())
    record["candidates"][0]["asset"]["sha256"] = file_sha256(path)
    case[4].write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(held_results.HeldResultsError, match="音声"):
        held_results.alternative_view(case[2], case[4])


def test_late_alternative_change_cannot_replace_the_previous_outcome(case, monkeypatch):
    previous = _execute(case)
    before = previous.record_path.read_bytes()
    write = application._write_outcome_atomic

    def changed(path, payload, validate):
        def check():
            candidate = previous.held_alternative_artifacts[0]
            candidate.write_bytes(candidate.read_bytes() + b"changed late")
            validate()
        write(path, payload, check)

    monkeypatch.setattr(application, "_write_outcome_atomic", changed)
    assert _execute(case).disposition == application.ResultDisposition.FAILED
    assert previous.record_path.read_bytes() == before


def test_moved_alternative_audio_is_discovered_and_reused_by_content(case, tmp_path):
    original = _execute(case)
    before = original.record_path.read_bytes()
    audio = original.held_alternative_artifacts[0]
    archive = tmp_path / "archive"
    archive.mkdir()
    moved = archive / "another-name.flac"
    shutil.move(audio, moved)
    state = tmp_path / "state"
    moves = relocate.find_moved([archive], state_root=state, outcome_records=[original.record_path])
    assert len(moves) == 1 and moves[0]["kind"] == "output"
    relocate.record_relocations(moves, state_root=state)
    outcome = application.read_bgm_development_outcome(original.record_path, state_root=state)
    assert outcome.disposition == application.ResultDisposition.HELD
    assert outcome.held_alternative_artifacts == (moved,)
    assert outcome.held_artifact_locations[audio] == (moved,)
    assert original.record_path.read_bytes() == before


def test_no_eligible_alternative_is_not_target_absence(tmp_path):
    run = upstream(tmp_path, ambiguous=False)
    plan = held.freeze(run, tmp_path / "alternatives")
    alternative_run = held.run(plan)
    request = application.ProcessingRequest.from_paths([tmp_path / "source.wav"], application.UserIntent.EXTRACT_BGM_CUES)
    result = application.run_bgm_development(request, run.parent / "plan.json", runner=lambda *a, **k: run,
                                             alternative_run_path=alternative_run)
    assert result.disposition == application.ResultDisposition.HELD
    assert len(result.held_artifacts) == 1 and result.held_alternative_artifacts == ()
    assert result.zero_result is None and result.outputs == ()
