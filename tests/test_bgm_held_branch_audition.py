"""代替候補・参照残差分岐も、試聴でtagが書き戻された後に読み戻せる。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from mutagen.flac import FLAC

from lss.bgm import held_results
from lss.core import audio_identity
from lss.harness.fetch import file_sha256
from lss.validation import held_bgm_alternatives as held
from lss.validation import reference_residual_episode as stage
from tests.test_held_bgm_alternatives import upstream
from tests.test_reference_residual_episode import case, _freeze  # noqa: F401


def audition(path: Path) -> Path:
    """再生ソフトが再生回数を書き戻したときと同じ変化を起こす。"""
    before, identity = file_sha256(path), audio_identity.audio_identity(path)
    tags = FLAC(path)
    tags["LSS_TEST_PLAY_COUNT"] = ["2"]
    tags.save()
    assert file_sha256(path) != before
    assert audio_identity.same_audio(identity, audio_identity.audio_identity(path))
    return path


def test_an_auditioned_alternative_is_still_read_back_with_its_parent(tmp_path):
    run_path = upstream(tmp_path)
    report = held.run(held.freeze(run_path, tmp_path / "held"))
    before = report.read_bytes()
    candidate = Path(json.loads(before)["candidates"][0]["asset"]["path"])
    audition(candidate)
    view = held_results.alternative_view(run_path, report)
    assert [Path(item["path"]) for item in view["artifacts"]][0] == candidate.resolve()
    assert view["selection_state"] == "unresolved"
    assert report.read_bytes() == before


def test_an_alternative_whose_audio_changed_is_not_read_back(tmp_path):
    run_path = upstream(tmp_path)
    report = held.run(held.freeze(run_path, tmp_path / "held"))
    candidate = Path(json.loads(report.read_bytes())["candidates"][0]["asset"]["path"])
    values = bytearray(candidate.read_bytes())
    values[-1] ^= 0xFF
    candidate.write_bytes(bytes(values))
    with pytest.raises(held_results.HeldResultsError):
        held_results.alternative_view(run_path, report)


def test_an_auditioned_reference_residual_branch_is_still_read_back(case, tmp_path):
    report = stage.run(_freeze(case, tmp_path))
    before = report.read_bytes()
    candidate = Path(json.loads(before)["candidates"][0]["asset"]["path"])
    audition(candidate)
    view = stage.read_result(case[0], report)
    assert [Path(item["path"]) for item in view["artifacts"]][0] == candidate.resolve()
    assert report.read_bytes() == before


def test_a_reference_residual_candidate_whose_audio_changed_is_not_read_back(case, tmp_path):
    report = stage.run(_freeze(case, tmp_path))
    candidate = Path(json.loads(report.read_bytes())["candidates"][0]["asset"]["path"])
    values = bytearray(candidate.read_bytes())
    values[-1] ^= 0xFF
    candidate.write_bytes(bytes(values))
    with pytest.raises(stage.ReferenceResidualResultError):
        stage.read_result(case[0], report)
