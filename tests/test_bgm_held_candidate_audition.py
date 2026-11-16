"""試聴で候補のtagが書き戻されても、保存済みのBGM保留結果を読み戻せる。"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss import application
from lss.assets import relocate
from lss.bgm import identity, identity_run
from lss.core import audio_identity
from lss.harness.fetch import file_sha256
from tests.test_application import _bgm_development_plan


@pytest.fixture
def auditionable_result(tmp_path, monkeypatch):
    """復号できる候補音声を持つ保留結果。materialize_candidateと同じ記録の形。"""
    request, plan, source = _bgm_development_plan(tmp_path)
    signal = tmp_path / "routed.wav"
    sf.write(signal, np.zeros((40 * identity.SAMPLE_RATE, 2)), identity.SAMPLE_RATE)
    reference = tmp_path / "known.flac"
    reference.write_bytes(b"pinned reference")
    candidates = []
    for number, (name, start, end) in enumerate((("cue-a", 0, 20), ("cue-b", 25, 40)), 1):
        path = tmp_path / "candidates" / f"{name}.flac"
        path.parent.mkdir(exist_ok=True)
        rng = np.random.default_rng(number)
        sf.write(path, rng.standard_normal((8192, 2)) * 0.2, 44100, subtype="PCM_24")
        candidates.append({
            "asset": {"candidate_asset_id": name, "path": str(path), "sha256": file_sha256(path),
                      "audio": audio_identity.audio_identity(path),
                      "representation": "noncanonical_staged_candidate", "publication_state": "not_published"},
            "origin": {"separation_recipe_id": "renderer", "activity_recipe_id": "activity",
                       "sample_rate": identity.SAMPLE_RATE, "start_frame": start * identity.SAMPLE_RATE,
                       "end_frame": end * identity.SAMPLE_RATE},
        })
    report = tmp_path / "run.json"
    report.write_text(json.dumps({
        "status": identity_run.ROUTED_STATUS, "promotion_allowed": False, "holdout_consumed": False,
        "publication_state": "not_published", "source_hashes_after_verified": True,
        "plan_sha256": file_sha256(plan), "outcome": "unresolved_candidates",
        "source": {"path": str(source), "sha256": file_sha256(source)},
        "renderer_id": "bgm-development-cue-renderer-v1-test",
        "rendering": {"signal": {"path": str(signal), "sha256": file_sha256(signal)}},
        "candidates": candidates,
    }), encoding="utf-8")
    monkeypatch.setattr(identity.ReferenceIndex, "build", lambda *a, **k: None)
    monkeypatch.setattr(identity, "match", lambda *a, **k: [])
    return application.run_bgm_development(request, plan, runner=lambda *a, **k: report)


def audition(path):
    """再生ソフトが再生回数を書き戻したときと同じ変化を起こす。"""
    before, identity_before = file_sha256(path), audio_identity.audio_identity(path)
    tags = FLAC(path)
    tags["LSS_TEST_PLAY_COUNT"] = ["4"]
    tags.save()
    assert file_sha256(path) != before
    assert audio_identity.same_audio(identity_before, audio_identity.audio_identity(path))
    return path


def test_an_auditioned_candidate_is_still_read_back_as_the_same_held_result(auditionable_result, tmp_path):
    original = auditionable_result
    assert len(original.held_artifacts) == 2
    audition(original.held_artifacts[0])
    outcome = application.read_bgm_development_outcome(original.record_path, state_root=tmp_path / "state")
    assert outcome.disposition == application.ResultDisposition.HELD
    assert outcome.held_artifacts == original.held_artifacts
    assert outcome.outputs == () and outcome.reused is True


def test_a_candidate_whose_audio_changed_is_not_read_back(auditionable_result, tmp_path):
    original = auditionable_result
    shutil.copyfile(original.held_artifacts[1], original.held_artifacts[0])
    outcome = application.read_bgm_development_outcome(original.record_path, state_root=tmp_path / "state")
    assert outcome.disposition == application.ResultDisposition.FAILED
    assert outcome.held_artifacts == () and "BGM" in (outcome.message or "")


def test_an_auditioned_candidate_at_its_place_is_not_searched_for_as_moved(auditionable_result, tmp_path):
    original = auditionable_result
    audition(original.held_artifacts[0])
    assert relocate.find_moved([original.held_artifacts[0].parent], state_root=tmp_path / "state",
                               outcome_records=[original.record_path]) == []


def test_a_moved_and_auditioned_candidate_is_found_by_its_samples(auditionable_result, tmp_path):
    original = auditionable_result
    archive = tmp_path / "archive"
    archive.mkdir()
    moved = archive / "試聴済み.flac"
    shutil.move(original.held_artifacts[0], moved)
    audition(moved)
    state = tmp_path / "state"
    moves = relocate.find_moved([archive], state_root=state, outcome_records=[original.record_path])
    assert [move["found_at"] for move in moves] == [[str(moved.resolve())]]
    relocate.record_relocations(moves, state_root=state)
    outcome = application.read_bgm_development_outcome(original.record_path, state_root=state)
    assert outcome.held_artifacts[0] == moved.resolve()
