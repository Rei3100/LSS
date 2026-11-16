"""来歴は自前の記録から引く。成果物のタグが書き換えられても更新判断を失わない。"""
from __future__ import annotations

import json

import pytest
from mutagen.flac import FLAC

from lss.assets import reprocess
from lss.core import recipe, recipe_registry, session
from tests.test_reprocess_candidates import CURRENT, _output, _source


def revision(model_id: str, model_sha: str, config_sha: str) -> dict:
    return recipe.separation_recipe_revision(
        provider="msst", arch="mel_band_roformer", model_id=model_id, model_sha256=model_sha,
        config_sha256=config_sha, runtime_id="runtime-1",
        expected_stems=["Instrumental", "Vocals"], output_samplerate=44100,
        input_preparation={"resample": "none"},
        serialization={"subtype": "PCM_24", "clipping": "none"})


def completed(tmp_path, state, *, model, model_sha, config_sha="c" * 64, register=True,
              tag_body=True):
    source, sha = _source(tmp_path / "source.flac", b"source")
    output = _output(tmp_path / "song - Instrumental.flac", model=model, model_sha=model_sha,
                     config_sha=config_sha)
    body = revision(model, model_sha, config_sha)
    if tag_body:
        # 実際の成果物と同じく、Recipe本文そのものをタグへ書く。
        tags = FLAC(output)
        tags["LSS_RECIPE_REVISION"] = json.dumps(body)
        tags.save()
    if register:
        recipe_registry.record(body, state_root=state)
    session.mark_completed(sha, body["recipe_id"], "2026-09-01", source_path=source,
                           outputs=[output], state_root=state)
    return source, sha, output, body


def played(path):
    tags = FLAC(path)
    tags["LSS_TEST_PLAY_COUNT"] = ["3"]
    tags.save()
    return path


def test_a_recorded_recipe_answers_without_reading_the_output(tmp_path):
    state = tmp_path / "state"
    _, _, output, _ = completed(tmp_path, state, model="model_old", model_sha="o" * 64)
    output.unlink()
    result = reprocess.reprocess_candidates(state_root=state, current=CURRENT)
    assert [item["reason"] for item in result["candidates"]] == ["model_updated"]
    assert result["candidates"][0]["previous"]["model_id"] == "model_old"


def test_a_played_output_keeps_its_update_judgement(tmp_path):
    state = tmp_path / "state"
    _, _, output, _ = completed(tmp_path, state, model="model_old", model_sha="o" * 64)
    played(output)
    result = reprocess.reprocess_candidates(state_root=state, current=CURRENT)
    assert [item["reason"] for item in result["candidates"]] == ["model_updated"]


def test_rewritten_tags_do_not_override_the_recorded_recipe(tmp_path):
    state = tmp_path / "state"
    _, _, output, _ = completed(tmp_path, state, model="model_old", model_sha="o" * 64)
    # 誰かが別のmodelの来歴タグを書き込んでも、こちらの記録が優先される。
    _output(output, model=CURRENT.model_id, model_sha=CURRENT.model_sha256,
            config_sha=CURRENT.config_sha256)
    result = reprocess.reprocess_candidates(state_root=state, current=CURRENT)
    assert [item["reason"] for item in result["candidates"]] == ["model_updated"]
    assert result["candidates"][0]["previous"]["model_id"] == "model_old"


def test_a_played_output_answers_with_a_body_that_matches_the_ledger(tmp_path):
    state = tmp_path / "state"
    _, _, output, _ = completed(tmp_path, state, model="model_old", model_sha="o" * 64,
                                register=False)
    played(output)
    result = reprocess.reprocess_candidates(state_root=state, current=CURRENT)
    # bytesは変わったが、タグの本文は台帳のrecipe_id（本文のhash）と一致する。
    assert [item["reason"] for item in result["candidates"]] == ["model_updated"]
    assert result["candidates"][0]["previous"]["model_id"] == "model_old"


def test_without_a_matching_body_a_changed_output_stays_unknown(tmp_path):
    state = tmp_path / "state"
    _, _, output, _ = completed(tmp_path, state, model="model_old", model_sha="o" * 64,
                                register=False, tag_body=False)
    played(output)
    result = reprocess.reprocess_candidates(state_root=state, current=CURRENT)
    # 本文の無い成果物の、書き換えられ得るmodelタグは読まない。判断できないまま残る。
    assert result["counts"].get("processing_unknown") == 1
    assert result["candidates"] == []


def test_a_forged_body_on_a_changed_output_is_not_trusted(tmp_path):
    state = tmp_path / "state"
    _, _, output, body = completed(tmp_path, state, model="model_old", model_sha="o" * 64,
                                   register=False)
    forged = dict(revision(CURRENT.model_id, CURRENT.model_sha256, CURRENT.config_sha256))
    forged["recipe_id"] = body["recipe_id"]
    tags = FLAC(output)
    tags["LSS_RECIPE_REVISION"] = json.dumps(forged)
    tags.save()
    result = reprocess.reprocess_candidates(state_root=state, current=CURRENT)
    assert result["counts"].get("already_current", 0) == 0
    assert result["counts"].get("processing_unknown") == 1


def test_a_recorded_current_recipe_is_not_offered_again(tmp_path):
    state = tmp_path / "state"
    completed(tmp_path, state, model=CURRENT.model_id, model_sha=CURRENT.model_sha256,
              config_sha=CURRENT.config_sha256)
    result = reprocess.reprocess_candidates(state_root=state, current=CURRENT)
    assert result["counts"]["already_current"] == 1 and result["candidates"] == []


def test_existing_results_can_hand_their_recipe_body_to_the_registry(tmp_path):
    state = tmp_path / "state"
    _, _, output, body = completed(tmp_path, state, model="model_old", model_sha="o" * 64,
                                   register=False)
    preview = reprocess.backfill_recipe_revisions(state_root=state, dry_run=True)
    assert preview["recorded"] == 1 and preview["written"] is False
    assert recipe_registry.revision_for(body["recipe_id"], state_root=state) is None
    written = reprocess.backfill_recipe_revisions(state_root=state)
    assert written["recorded"] == 1 and written["written"] is True
    assert recipe_registry.revision_for(body["recipe_id"], state_root=state) == body
    again = reprocess.backfill_recipe_revisions(state_root=state)
    assert again["already_recorded"] == 1 and again["recorded"] == 0
    # 記録が入った後は、tagを書き戻されても更新判断が残る。
    played(output)
    result = reprocess.reprocess_candidates(state_root=state, current=CURRENT)
    assert [item["reason"] for item in result["candidates"]] == ["model_updated"]


def test_a_played_output_can_hand_over_its_recipe_body(tmp_path):
    state = tmp_path / "state"
    _, _, output, body = completed(tmp_path, state, model="model_old", model_sha="o" * 64,
                                   register=False)
    played(output)
    summary = reprocess.backfill_recipe_revisions(state_root=state)
    assert summary["recorded"] == 1
    assert recipe_registry.revision_for(body["recipe_id"], state_root=state) == body


def test_a_played_output_without_a_body_cannot_hand_over_anything(tmp_path):
    state = tmp_path / "state"
    _, _, output, body = completed(tmp_path, state, model="model_old", model_sha="o" * 64,
                                   register=False, tag_body=False)
    played(output)
    summary = reprocess.backfill_recipe_revisions(state_root=state)
    assert summary["recorded"] == 0
    assert summary["not_recoverable"] == [body["recipe_id"]]
    assert recipe_registry.revision_for(body["recipe_id"], state_root=state) is None


def test_a_body_that_belongs_to_another_recipe_is_not_recorded(tmp_path):
    state = tmp_path / "state"
    _, _, output, body = completed(tmp_path, state, model="model_old", model_sha="o" * 64,
                                   register=False)
    other = revision("model_other", "z" * 64, "c" * 64)
    tags = FLAC(output)
    tags["LSS_RECIPE_REVISION"] = json.dumps(other)
    tags.save()
    summary = reprocess.backfill_recipe_revisions(state_root=state)
    assert summary["recorded"] == 0 and summary["not_recoverable"] == [body["recipe_id"]]
