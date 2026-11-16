"""同じ原本のBGM公開結果のCurrentBest（cueごと）とRecipeの版の置き換え（`lss.bgm.current_best`、v13）。

妥当性と最良の選択は別の判断：前から公開しているcueは、そのcue自身の区間の観測で今も妥当かを先に決め、
新しい版が保留・未確定でも妥当なcueは残す。妥当なcueを置き換えるのは検証済みの改善だけで、不正と観測された
cueは新しい版が何も公開しなくても公開から外して履歴へ移す。切替は中断しても一貫し、巻き戻せる。
GPU推論だけを偽物に置き換え、復号・候補保存・意味判定・公開・切替は製品の実装をそのまま通す。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss import application
from lss.bgm import cue_publication, current_best, event_tagging
from tests.test_application_bgm_candidates import _changing_embedding, _run, _source, gpu  # noqa: F401

V10_WORKFLOW = "episode-sfc-routed-cues-v10"
V7_WORKFLOW = "episode-sfc-routed-cues-v7"


def _legacy(tmp_path, monkeypatch, source, *, workflow=V10_WORKFLOW, music_from=None):
    """前の版（v10等）と同じ形の完了run・公開ファイル（番号付きの名前、CurrentBestの記録なし）を作る。

    ``music_from`` を渡すと、その秒より前の窓をタグ付けが音楽と聞かない（前の版の区切りが短い）。
    """
    real = application._bgm_candidate_recipe
    version = workflow.rsplit("-v", 1)[1]

    def old_recipe():
        recipe = dict(real())
        recipe.update(workflow=workflow, recipe_id=f"bgm-episode-candidates-v{version}-" + "0" * 16)
        return recipe

    real_tag = event_tagging.tag_signal

    def shortened(signal_path, *a, **k):
        probabilities, embeddings = real_tag(signal_path, *a, **k)
        if music_from is not None and Path(signal_path).name == "music.wav":
            probabilities = probabilities.copy()
            probabilities[:music_from, :] = 0
        return probabilities, embeddings

    with monkeypatch.context() as patch:
        patch.setattr(application, "_bgm_candidate_recipe", old_recipe)
        patch.setattr(application, "_BGM_CANDIDATE_WORKFLOW", workflow)
        patch.setattr(event_tagging, "tag_signal", shortened)
        patch.setattr(cue_publication, "file_name",
                      lambda source_path, start, unresolved_extent=False:
                      f"{Path(source_path).stem} - BGM 01 ({int(start) // 60:02d}m{int(start) % 60:02d}s).flac")
        outcome = _run(tmp_path, source)
    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert run["workflow"] == workflow
    # 前の版にはCurrentBestの記録が無い。
    current_best.record_path(tmp_path / "state", run["source"]["sha256"]).unlink()
    [old] = outcome.outputs
    return outcome, old, run


def _legacy_v10(tmp_path, monkeypatch, source):
    outcome, old, run = _legacy(tmp_path, monkeypatch, source)
    assert old.name == "episode - BGM 01 (00m05s).flac"
    return outcome, old, run


def _older_and_improved(tmp_path, monkeypatch, gpu):
    """v7の結果（区切りが短い）と、それを検証済みの改善（台詞の残りの除去）で確かに良くする新しい版。"""
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    _, old, old_run = _legacy(tmp_path, monkeypatch, source, workflow=V7_WORKFLOW, music_from=7)
    assert old.name == "episode - BGM 01 (00m07s).flac"
    monkeypatch.setattr(current_best, "cue_improvements", lambda item: ["dialogue_residual_whole_cue"])
    return source, old, old_run


def _heavy_events(monkeypatch):
    """原本（解析用複製）のタグに、非音楽のイベント音（Explosion）を全窓で足す。"""
    real_tag = event_tagging.tag_signal

    def with_events(signal_path, *a, **k):
        probabilities, embeddings = real_tag(signal_path, *a, **k)
        if Path(signal_path).name == "analysis.wav":
            probabilities = probabilities.copy()
            probabilities[:, 426] = 0.6
        return probabilities, embeddings

    monkeypatch.setattr(event_tagging, "tag_signal", with_events)


def _public(tmp_path, stem="episode"):
    folder = tmp_path / "LSS" / f"{stem} - BGM"
    return sorted(path.name for path in folder.glob("*.flac"))


def _history_files(tmp_path):
    return sorted(path for path in (tmp_path / "LSS" / current_best.HISTORY_DIR).rglob("*.flac"))


def test_first_publication_creates_a_current_best_with_the_public_files(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    outcome = _run(tmp_path, source)
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    record = current_best.read(tmp_path / "state", run["source"]["sha256"])
    key = outcome.record_path.parent.parent.name
    assert record["schema_version"] == 2 and record["current"]["evaluated_runs"] == [key]
    assert record["current"]["transition"] == "first_publication" and record["history"] == []
    assert [item["run_key"] for item in record["current"]["files"]] == [key]
    assert [Path(item["public_path"]) for item in record["current"]["files"]] == list(outcome.outputs)
    assert _public(tmp_path) == ["episode - BGM 00m05s.flac"]
    # 作業領域は片付き、公開フォルダには採用中の結果だけがある。
    assert not any((tmp_path / "LSS" / current_best.STAGING_DIR).rglob("*.flac"))


def test_a_valid_v10_cue_stays_current_when_the_new_version_is_not_demonstrably_better(tmp_path, gpu, monkeypatch):
    """同じ音を作るだけの新しい版は「良くした」ではない。妥当なv10のcueを最新runだからと置き換えない。"""
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    _, old, old_run = _legacy_v10(tmp_path, monkeypatch, source)
    old_bytes = old.read_bytes()

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED
    assert outcome.reused is False and len(gpu.calls) == 2  # 意図した処理し直し（重複扱いで飛ばさない）
    assert list(outcome.outputs) == [old] and old.read_bytes() == old_bytes
    assert _public(tmp_path) == ["episode - BGM 01 (00m05s).flac"]
    record = current_best.read(tmp_path / "state", old_run["source"]["sha256"])
    [kept] = record["current"]["files"]
    assert kept["workflow"] == V10_WORKFLOW and kept["adoption"] == "legacy_before_current_best"
    [decision] = record["current"]["decisions"]["old"]
    assert (decision["validity"], decision["decision"]) == ("VALID", "keep")
    assert decision["why"] == "valid_and_new_not_demonstrably_better"
    new_run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    [item] = new_run["published"]
    assert item["current_best"]["state"] == "not_placed"
    assert item["current_best"]["reason"] == "region_has_valid_current_best"
    assert record["history"] == []   # 何も動かしていない
    assert not any((tmp_path / "LSS" / current_best.STAGING_DIR).rglob("*.flac"))
    # 同じ原本をもう一度：推論せず、同じCurrentBestを返す。
    again = _run(tmp_path, source)
    assert again.reused is True and len(gpu.calls) == 2 and list(again.outputs) == [old]
    found = application.existing_bgm_candidates(source, old_run["source"]["sha256"], state_root=tmp_path / "state")
    assert found["k"] == "done" and found["files"] == ["episode - BGM 01 (00m05s).flac"]


def test_a_kept_older_cue_follows_the_same_content_processed_at_another_location(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    _, old, _ = _legacy_v10(tmp_path, monkeypatch, source)
    old_pcm = current_best.pcm_sha256(old)
    _run(tmp_path, source)
    other = tmp_path / "moved" / "copy.wav"
    other.parent.mkdir()
    other.write_bytes(source.read_bytes())
    moved = _run(tmp_path, other)
    assert moved.reused is True and len(gpu.calls) == 2   # 推論しない
    [here] = moved.outputs
    assert here == tmp_path / "moved" / "LSS" / "copy - BGM" / "copy - BGM 01 (00m05s).flac"
    assert current_best.pcm_sha256(here) == old_pcm and old.is_file()   # 前の場所のファイルには触れない


def test_the_existing_lookup_does_not_count_an_unevaluated_version_as_done(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    _, _, old_run = _legacy_v10(tmp_path, monkeypatch, source)
    assert application.existing_bgm_candidates(source, old_run["source"]["sha256"],
                                               state_root=tmp_path / "state") is None


def test_a_held_new_run_does_not_erase_a_valid_older_cue(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    _, old, old_run = _legacy_v10(tmp_path, monkeypatch, source)
    gpu.tagger.noncontinuous_music = True   # 新しい版ではMusic分類が途切れ、公開cueが無い（未確定）
    outcome = _run(tmp_path, source)
    assert outcome.disposition is application.ResultDisposition.PUBLISHED
    assert list(outcome.outputs) == [old] and old.is_file()
    assert _public(tmp_path) == ["episode - BGM 01 (00m05s).flac"]
    record = current_best.read(tmp_path / "state", old_run["source"]["sha256"])
    assert [item["workflow"] for item in record["current"]["files"]] == [V10_WORKFLOW]
    assert record["current"]["decisions"]["old"][0]["why"] == "valid_and_no_new_cue_in_region"


def test_an_older_cue_whose_only_negative_evidence_is_a_heavy_event_scene_is_kept_as_unknown(tmp_path, gpu,
                                                                                               monkeypatch):
    """A source-scene risk never removes earlier accepted sound. Even a newly
    rendered, content-checked candidate needs a demonstrated adoption improvement.
    """
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    _, old, old_run = _legacy_v10(tmp_path, monkeypatch, source)
    old_bytes = old.read_bytes()
    _heavy_events(monkeypatch)
    outcome = _run(tmp_path, source)
    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    assert list(outcome.outputs) == [old] and old.read_bytes() == old_bytes and _history_files(tmp_path) == []
    new_run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    [new_candidate] = new_run["published"]
    check = new_candidate["provenance"]["rendered_content_check"]
    assert check["state"] == "passed"
    assert check["checked_audio"]["pcm_sha256"] == new_candidate["audio"]["pcm_sha256"]
    # Candidate validity does not establish superiority over the accepted cue.
    record = current_best.read(tmp_path / "state", old_run["source"]["sha256"])
    assert record["current"]["adoption_policy"] == application._BGM_ADOPTION_POLICY
    [decision] = record["current"]["decisions"]["old"]
    assert (decision["validity"], decision["decision"]) == ("UNKNOWN", "keep")
    assert decision["risks"] == ["quality_risk:heavy_nonmusic_event_scene"]
    assert decision["why"] == "validity_unknown_is_not_invalid"


def test_a_demonstrably_better_new_cue_replaces_the_older_one_and_keeps_it_in_history(tmp_path, gpu, monkeypatch):
    source, old, old_run = _older_and_improved(tmp_path, monkeypatch, gpu)
    old_pcm = current_best.pcm_sha256(old)
    outcome = _run(tmp_path, source)
    assert outcome.disposition is application.ResultDisposition.PUBLISHED
    assert _public(tmp_path) == ["episode - BGM 00m05s.flac"]
    record = current_best.read(tmp_path / "state", old_run["source"]["sha256"])
    [current] = record["current"]["files"]
    assert current["recipe_id"].startswith("bgm-episode-candidates-v13-")
    [decision] = record["current"]["decisions"]["old"]
    assert decision["decision"] == "replace_with_better" and decision["improvements"] == ["dialogue_residual_whole_cue"]
    [previous] = record["history"]
    assert previous["files"][0]["workflow"] == V7_WORKFLOW
    [archived] = _history_files(tmp_path)
    assert archived.name == "episode - BGM 01 (00m07s).flac" and current_best.pcm_sha256(archived) == old_pcm
    again = _run(tmp_path, source)
    assert again.reused is True and len(gpu.calls) == 2 and list(again.outputs) == list(outcome.outputs)


@pytest.mark.parametrize("failing_step", ["archive", "place", "record"])
def test_an_interrupted_replacement_never_leaves_no_valid_result(tmp_path, gpu, monkeypatch, failing_step):
    source, old, old_run = _older_and_improved(tmp_path, monkeypatch, gpu)
    old_pcm = current_best.pcm_sha256(old)
    sha = old_run["source"]["sha256"]
    real_link, real_replace, real_write = current_best._link_or_copy, os.replace, current_best._write_json_atomic

    class Crash(BaseException):
        """電源断に相当（例外処理の戻しも走らない）。"""

    def crash_link(src, dst):
        if failing_step == "archive":
            raise Crash()
        return real_link(src, dst)

    def crash_replace(src, dst, *a, **k):
        if (failing_step == "place" and Path(dst).suffix == ".flac"
                and current_best.STAGING_DIR in Path(src).parts and current_best.STAGING_DIR not in Path(dst).parts):
            raise Crash()
        return real_replace(src, dst, *a, **k)

    def crash_write(path, payload):
        if failing_step == "record" and Path(path) == current_best.record_path(tmp_path / "state", sha):
            raise Crash()
        return real_write(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(current_best, "_link_or_copy", crash_link)
        patch.setattr(current_best.os, "replace", crash_replace)
        patch.setattr(current_best, "_write_json_atomic", crash_write)
        with pytest.raises(Crash):
            application.run_bgm_candidate_request(
                application.ProcessingRequest((source,), application.UserIntent.EXTRACT_BGM_CUES),
                state_root=tmp_path / "state")
    # 中断の直後でも、旧版は公開名か履歴に完全な形で残っている。
    survivors = [path for path in [old, *_history_files(tmp_path)] if current_best.pcm_sha256(path) == old_pcm]
    assert survivors, "旧版が失われた"
    assert current_best.journal_path(tmp_path / "state", sha).is_file()
    # 次の起動で切替を完了する（推論しない）。
    assert application.reconcile_bgm_current_best(tmp_path / "state") == 1
    record = current_best.read(tmp_path / "state", sha)
    assert record["current"]["files"][0]["recipe_id"].startswith("bgm-episode-candidates-v13-")
    assert _public(tmp_path) == ["episode - BGM 00m05s.flac"]
    assert [current_best.pcm_sha256(path) for path in _history_files(tmp_path)] == [old_pcm]
    assert not current_best.journal_path(tmp_path / "state", sha).exists()
    again = _run(tmp_path, source)
    assert again.reused is True and len(gpu.calls) == 2


def test_a_lost_staged_file_is_rebuilt_from_the_run_when_the_switch_resumes(tmp_path, gpu, monkeypatch):
    source, old, old_run = _older_and_improved(tmp_path, monkeypatch, gpu)
    sha = old_run["source"]["sha256"]

    class Crash(BaseException):
        pass

    real_execute = current_best.execute

    def journal_only(state_root, journal, regenerate=None):
        current_best._write_json_atomic(current_best.journal_path(state_root, journal["source_sha256"]), journal)
        raise Crash()

    with monkeypatch.context() as patch:
        patch.setattr(current_best, "execute", journal_only)
        with pytest.raises(Crash):
            _run(tmp_path, source)
    # 作業領域の新版が消えても（掃除・ディスク障害）、保留候補から推論なしで作り直して切替を終える。
    for path in (tmp_path / "LSS" / current_best.STAGING_DIR).rglob("*.flac"):
        path.unlink()
    # 公開フォルダの隣に、作り直しの置き場に似た名前の利用者の物がある（このprocessと同じPIDの名前も）
    lookalike = tmp_path / "LSS" / "episode - BGM" / f".lss-regenerate-{os.getpid()}"
    lookalike.mkdir(parents=True, exist_ok=True)
    (lookalike / "notes.txt").write_text("残して", encoding="utf-8")
    places = []
    regenerate = application._regenerate_published
    monkeypatch.setattr(application, "_regenerate_published",
                        lambda run, item, destination, *rest, **kwargs: places.append(Path(destination).parent)
                        or regenerate(run, item, destination, *rest, **kwargs))
    assert current_best.execute is real_execute
    assert application.reconcile_bgm_current_best(tmp_path / "state") == 1
    assert places and all(p.name.startswith(".lss-regenerate-") and p != lookalike for p in places)
    assert _public(tmp_path) == ["episode - BGM 00m05s.flac"] and len(gpu.calls) == 2
    assert current_best.read(tmp_path / "state", sha)["history"][0]["files"][0]["workflow"] == V7_WORKFLOW
    assert (lookalike / "notes.txt").read_text(encoding="utf-8") == "残して"
    assert [p.name for p in lookalike.parent.glob(".lss-regenerate-*")] == [lookalike.name]


def test_a_new_candidate_failing_integrity_does_not_replace_the_known_result(tmp_path, gpu, monkeypatch):
    from lss.bgm import cue_scope

    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    _, old, _ = _legacy_v10(tmp_path, monkeypatch, source)
    real = cue_scope.integrity
    monkeypatch.setattr(cue_scope, "integrity", lambda audio, rate: {**real(audio, rate), "integrity_passed": 0.0})
    outcome = _run(tmp_path, source)
    assert list(outcome.outputs) == [old]
    assert _public(tmp_path) == ["episode - BGM 01 (00m05s).flac"]


def test_restore_previous_rolls_back_and_pins_then_can_return(tmp_path, gpu, monkeypatch):
    source, old, old_run = _older_and_improved(tmp_path, monkeypatch, gpu)
    old_pcm = current_best.pcm_sha256(old)
    new = _run(tmp_path, source)
    [new_file] = new.outputs
    new_pcm = current_best.pcm_sha256(new_file)
    sha = old_run["source"]["sha256"]

    back = application.restore_previous_bgm_result(source, state_root=tmp_path / "state")
    assert back.disposition is application.ResultDisposition.PUBLISHED, back.message
    assert _public(tmp_path) == ["episode - BGM 01 (00m07s).flac"]
    assert current_best.pcm_sha256(tmp_path / "LSS" / "episode - BGM" / "episode - BGM 01 (00m07s).flac") == old_pcm
    record = current_best.read(tmp_path / "state", sha)
    assert record["current"]["files"][0]["workflow"] == V7_WORKFLOW and record["current"]["pinned"] is True
    assert record["history"][-1]["files"][0]["recipe_id"].startswith("bgm-episode-candidates-v13-")
    # 戻した原本は、処理し直しても勝手に新しい版へ戻らない（推論もしない）。
    kept = _run(tmp_path, source)
    assert kept.reused is True and len(gpu.calls) == 2
    assert _public(tmp_path) == ["episode - BGM 01 (00m07s).flac"]
    # もう一度戻すと、新しい版へ戻る。
    forward = application.restore_previous_bgm_result(source, state_root=tmp_path / "state")
    assert forward.disposition is application.ResultDisposition.PUBLISHED
    assert _public(tmp_path) == ["episode - BGM 00m05s.flac"]
    assert current_best.pcm_sha256(new_file) == new_pcm


def test_restore_succeeds_when_the_restored_cue_run_no_longer_lists_its_audio(tmp_path, gpu, monkeypatch):
    """Deployed states can hold a history row whose run record was rewritten after adoption: the run no longer
    lists the row's exact audio. Restoring that row must still finish (the archived file keeps its own identity),
    keep the published audio and the pinned CurrentBest, and never fail with an empty message after the switch."""
    source, old, old_run = _older_and_improved(tmp_path, monkeypatch, gpu)
    old_pcm = current_best.pcm_sha256(old)
    new = _run(tmp_path, source)
    [new_file] = new.outputs
    new_pcm = current_best.pcm_sha256(new_file)
    sha = old_run["source"]["sha256"]
    row = current_best.read(tmp_path / "state", sha)["history"][-1]["files"][0]
    run_path = Path(row["run_path"])
    rewritten = json.loads(run_path.read_text(encoding="utf-8"))
    assert any(item["audio"]["pcm_sha256"] == old_pcm for item in rewritten["published"])
    for item in rewritten["published"]:
        item["audio"]["pcm_sha256"] = "f" * 64
    run_path.write_text(json.dumps(rewritten), encoding="utf-8")

    back = application.restore_previous_bgm_result(source, state_root=tmp_path / "state")
    assert back.disposition is application.ResultDisposition.PUBLISHED, repr(back.message)
    assert _public(tmp_path) == ["episode - BGM 01 (00m07s).flac"]
    assert current_best.pcm_sha256(tmp_path / "LSS" / "episode - BGM" / "episode - BGM 01 (00m07s).flac") == old_pcm
    record = current_best.read(tmp_path / "state", sha)
    assert record["current"]["pinned"] is True and record["current"]["files"][0]["pcm_sha256"] == old_pcm
    assert record["history"][-1]["files"][0]["pcm_sha256"] == new_pcm
    forward = application.restore_previous_bgm_result(source, state_root=tmp_path / "state")
    assert forward.disposition is application.ResultDisposition.PUBLISHED, repr(forward.message)
    assert current_best.pcm_sha256(new_file) == new_pcm


def test_a_restore_failure_never_has_an_empty_message(tmp_path, monkeypatch):
    source = tmp_path / "episode.wav"
    sf.write(str(source), np.zeros((441, 2)), 44100)
    monkeypatch.setattr(current_best, "reconcile", lambda *a, **k: next(iter(())))   # StopIteration: str() is ""
    outcome = application.restore_previous_bgm_result(source, state_root=tmp_path / "state")
    assert outcome.disposition is application.ResultDisposition.FAILED
    assert outcome.error_code == "bgm_restore_failed" and outcome.message == "StopIteration"


def test_restore_rebuilds_a_missing_history_file_without_inference(tmp_path, gpu, monkeypatch):
    source, old, _ = _older_and_improved(tmp_path, monkeypatch, gpu)
    old_pcm = current_best.pcm_sha256(old)
    _run(tmp_path, source)
    for path in _history_files(tmp_path):
        path.unlink()
    back = application.restore_previous_bgm_result(source, state_root=tmp_path / "state")
    assert back.disposition is application.ResultDisposition.PUBLISHED, back.message
    [restored] = back.outputs
    assert current_best.pcm_sha256(restored) == old_pcm and len(gpu.calls) == 2


def test_an_unrecorded_file_under_a_public_name_is_never_overwritten(tmp_path, gpu, monkeypatch):
    source, _, _ = _older_and_improved(tmp_path, monkeypatch, gpu)
    mine = tmp_path / "LSS" / "episode - BGM" / "episode - BGM 00m05s.flac"
    sf.write(str(mine), np.full((4410, 2), 0.1), 44100, subtype="PCM_24")
    mine_bytes = mine.read_bytes()
    _run(tmp_path, source)
    displaced = list((tmp_path / "LSS" / current_best.HISTORY_DIR / "displaced").rglob("*.flac"))
    assert [path.read_bytes() for path in displaced] == [mine_bytes]


def test_a_result_of_an_undeclared_version_is_left_untouched(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    first = _run(tmp_path, source)
    sha = json.loads(first.record_path.read_text(encoding="utf-8"))["source"]["sha256"]
    path = current_best.record_path(tmp_path / "state", sha)
    record = json.loads(path.read_text(encoding="utf-8"))
    for item in record["current"]["files"]:
        item.update(workflow="episode-sfc-routed-cues-v99", run_key="f" * 64)
    record["current"]["evaluated_runs"] = ["f" * 64]
    path.write_text(json.dumps(record), encoding="utf-8")
    second = _run(tmp_path, source)
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["current"]["files"][0]["workflow"] == "episode-sfc-routed-cues-v99"
    assert second.reused is True


def test_public_names_do_not_shift_when_an_earlier_cue_is_added(tmp_path, gpu):
    """名前は開始時刻だけで決まり、前に別のcueが加わっても同じcueの名前は変わらない。"""
    late = _source(tmp_path / "late.wav", music=(15, 28), seconds=30)
    gpu.tagger.singing = 0.0
    [only] = _run(tmp_path, late).outputs
    assert only.name == "late - BGM 00m15s.flac"
    assert cue_publication.file_name(late, 15.0) == only.name
    assert cue_publication.file_name(late, 2.0) == "late - BGM 00m02s.flac"


def test_duplicate_checks_count_only_cues_that_are_currently_public(tmp_path, gpu, monkeypatch):
    from lss.bgm import published_duplicates

    source, _, _ = _older_and_improved(tmp_path, monkeypatch, gpu)
    new = _run(tmp_path, source)
    adopted = current_best.adopted_runs(tmp_path / "state")
    _, runs = published_duplicates.prior_publications(tmp_path / "state" / "bgm-candidates", "0" * 64,
                                                      adopted=adopted)
    assert runs == [new.record_path.parent.parent.name]


def test_a_schema_v1_record_reads_as_a_set_of_cues(tmp_path):
    sha = "b" * 64
    path = current_best.record_path(tmp_path, sha)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "type": current_best.RECORD_TYPE, "schema_version": 1, "source_sha256": sha,
        "current": {"run_key": "k1", "recipe_id": "r1", "workflow": "episode-sfc-routed-cues-v12", "run_path": "p1",
                    "adoption": "recipe_supersession", "pinned": False,
                    "files": [{"public_path": "a.flac", "pcm_sha256": "x", "cue_key": "c",
                               "start_seconds": 1.0, "end_seconds": 20.0}]},
        "history": [{"run_key": "k0", "recipe_id": "r0", "workflow": "episode-sfc-routed-cues-v10", "run_path": "p0",
                     "retired_at": "t", "retirement": "recipe_supersession",
                     "files": [{"public_path": "b.flac", "pcm_sha256": "y", "archive_path": "h/b.flac"}]}]}),
        encoding="utf-8")
    record = current_best.read(tmp_path, sha)
    assert record["schema_version"] == 2 and record["current"]["evaluated_runs"] == ["k1"]
    assert record["current"]["files"][0]["run_key"] == "k1"
    assert record["history"][0]["files"][0]["archive_path"] == "h/b.flac"


# ---- 旧cueの妥当性（そのcue自身の区間の観測だけで決める） ----

def _assessed(tmp_path, cues, *, boundaries, events=(), seconds=1400):
    folder = tmp_path / "run" / "semantics"
    folder.mkdir(parents=True)
    mixture = np.zeros((seconds, 527), dtype=np.float16)
    mixture[:, 137] = 0.9
    mixture[:, 524] = 0.5   # Television（文脈。根拠にしない）
    for first, last in events:
        mixture[first:last, 426] = 0.6
    np.save(folder / "mixture-probabilities.npy", mixture)
    (folder / "cue-semantics.json").write_text(json.dumps({"edit_boundaries": [list(b) for b in boundaries]}),
                                              encoding="utf-8")
    run = {"semantics": {"record": {"name": "cue-semantics.json"}}}
    return application._assess_old_cues(run, tmp_path / "run" / "run.json",
                                        [{"start_seconds": a, "end_seconds": b} for a, b in cues])


def test_older_cue_validity_uses_only_observations_of_its_own_span(tmp_path):
    boundaries = [(0.0, 0.5), (600.0, 600.5), (630.0, 630.5), (900.0, 900.4)]
    rows = _assessed(tmp_path, [(10.0, 60.0), (580.0, 620.0), (605.0, 625.0), (700.0, 760.0), (1300.0, 1399.0)],
                     boundaries=boundaries, events=[(700, 730)])
    states = [(row["state"], row["basis"]) for row in rows]
    states = [(row["state"], row["basis"]) for row in rows]
    assert states == [("VALID", ["no_invalidating_observation"]),
                      ("INVALID", ["crosses_edit_boundary"]),
                      ("UNKNOWN", ["no_asset_local_invalidation"]),
                      ("UNKNOWN", ["no_asset_local_invalidation"]),
                      ("VALID", ["no_invalidating_observation"])]
    assert [row["risks"] for row in rows] == [[], [], ["role_risk:within_edit_insert_section"],
                                              ["quality_risk:heavy_nonmusic_event_scene"], []]
    # 危険は妥当性とも公開からの撤回とも別（区画・場面の種類からの見込みだけで旧cueを外す経路は無い）。
    assert not any("withdraw" in row for row in rows)
    assert rows[0]["context"]["television_tag"] is True


def test_older_cue_validity_is_unknown_without_observations_and_unknown_is_kept(tmp_path):
    rows = application._assess_old_cues({"semantics": None}, tmp_path / "run.json",
                                        [{"start_seconds": 1.0, "end_seconds": 20.0}])
    assert rows == [{"state": "UNKNOWN", "basis": ["new_run_has_no_assessable_observations"], "risks": []}]
    chosen = current_best.select([_cue("a", 1, 20)], rows, [])
    assert [item["public_path"] for item in chosen["keep"]] == ["a"]


def test_short_material_is_not_given_edit_structure_when_assessing_older_cues(tmp_path):
    [row] = _assessed(tmp_path, [(5.0, 25.0)], boundaries=[(14.0, 14.6)], seconds=30)
    assert row["state"] == "VALID"


# ---- 選択の規則（ファイルを動かさない） ----

def _cue(path, start, end, *, workflow=V10_WORKFLOW, pcm=None, improvements=()):
    return {"public_path": path, "start_seconds": start, "end_seconds": end, "workflow": workflow,
            "pcm_sha256": pcm or f"{path}-pcm", "run_key": path, "improvements": list(improvements)}


def test_select_keeps_unknown_validity_and_retires_only_invalid():
    old = [_cue("a", 0, 30), _cue("b", 40, 70), _cue("c", 80, 100)]
    assessments = [{"state": "VALID", "basis": []}, {"state": "UNKNOWN", "basis": ["not_assessable"]},
                   {"state": "INVALID", "basis": ["crosses_edit_boundary"]}]
    new = [_cue("n1", 0, 30, workflow="v13"), _cue("n2", 80, 95, workflow="v13"),
           _cue("n3", 110, 130, workflow="v13")]
    chosen = current_best.select(old, assessments, new)
    assert [item["public_path"] for item in chosen["keep"]] == ["a", "b"]
    assert [item["public_path"] for item in chosen["retire"]] == ["c"]
    assert [item["public_path"] for item in chosen["place"]] == ["n2", "n3"]
    assert [item["public_path"] for item in chosen["not_placed"]] == ["n1"]
    decisions = {row["public_path"]: row for row in chosen["decisions"]}
    assert decisions["b"]["why"] == "validity_unknown_is_not_invalid"
    assert decisions["c"]["decision"] == "retire_invalid"
    with pytest.raises(current_best.CurrentBestError):
        current_best.select(old[:1], [{"state": "HELD"}], [])   # 保留は妥当性の状態ではない


def test_select_replaces_a_valid_cue_only_by_a_validated_improvement_that_covers_it():
    old = [_cue("a", 10, 40, workflow=V7_WORKFLOW)]
    valid = [{"state": "VALID", "basis": []}]
    partial = [_cue("n", 10, 25, workflow="v13", improvements=["dialogue_residual_whole_cue"])]
    assert current_best.select(old, valid, partial)["keep"] == old   # 一部しか覆わないなら旧cueを残す
    full = [_cue("n", 9, 41, workflow="v13", improvements=["dialogue_residual_whole_cue"])]
    chosen = current_best.select(old, valid, full)
    assert chosen["retire"][0]["retirement"] == "superseded_by_validated_improvement"
    assert [item["public_path"] for item in chosen["place"]] == ["n"]
    # v10は台詞の残りの除去（局所）を既に持つ。同じ改善では良くならない。
    v10 = [_cue("a", 10, 40)]
    local = [_cue("n", 9, 41, workflow="v13", improvements=["dialogue_residual_local_windows"])]
    assert current_best.select(v10, valid, local)["keep"] == v10
    # 同じ音は良くしていない。
    same = [_cue("n", 10, 40, workflow="v13", pcm="a-pcm", improvements=["dialogue_residual_whole_cue"])]
    assert current_best.select(old, valid, same)["keep"] == old


# ---- 番組の構造のある素材（10分以上）：境界をまたぐcue・挿入の区画・重い場面・採用の方針の更新 ----

PROGRAM_RATE = 8000


def _program(path: Path) -> Path:
    """620秒の番組。音楽A（100-140秒、120.0-120.6秒にデジタル無音＝編集境界）、音楽B（300-330秒、295-295.5秒と
    335-335.5秒のデジタル無音に挟まれた39.5秒の挿入の区画の中）、音楽C（400-430秒、本編）。"""
    rate, seconds = PROGRAM_RATE, 620
    rng = np.random.default_rng(0)
    values = rng.normal(0, 0.001, (seconds * rate, 2)).astype(np.float32)
    for first, last in ((100, 140), (300, 330), (400, 430)):
        t = np.arange((last - first) * rate) / rate
        tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        values[first * rate:last * rate, 0] += tone
        values[first * rate:last * rate, 1] += 0.6 * tone
    for a, b in ((120.0, 120.6), (295.0, 295.5), (335.0, 335.5)):
        values[int(a * rate):int(b * rate)] = 0.0
    sf.write(path, values, rate, subtype="PCM_24")
    return path


def _v10_like_scope(patch):
    """前の版（v10）の判断：編集境界で分けず、挿入の区画・重い場面で止めない。"""
    from lss.bgm import cue_scope

    patch.setattr(cue_scope, "split_at_edit_boundaries", lambda segments, runs, seconds: segments)
    real_context, real_safety = cue_scope.context, cue_scope.safety
    patch.setattr(cue_scope, "context", lambda *a, **k: {**real_context(*a, **k), "insert_section": False})
    patch.setattr(cue_scope, "safety", lambda *a, **k: {**real_safety(*a, **k), "heavy_nonmusic_event_scene": False})


def _heavy_events_in(monkeypatch, first, last):
    real_tag = event_tagging.tag_signal

    def with_events(signal_path, *a, **k):
        probabilities, embeddings = real_tag(signal_path, *a, **k)
        if Path(signal_path).name == "analysis.wav":
            probabilities = probabilities.copy()
            probabilities[first:last, 426] = 0.6
        return probabilities, embeddings

    monkeypatch.setattr(event_tagging, "tag_signal", with_events)


def _legacy_program(tmp_path, monkeypatch, gpu):
    source = _program(tmp_path / "program.wav")
    gpu.tagger.singing = 0.0
    with monkeypatch.context() as patch:
        _v10_like_scope(patch)
        outcome, _, run = None, None, None
        real = application._bgm_candidate_recipe

        def old_recipe():
            recipe = dict(real())
            recipe.update(workflow=V10_WORKFLOW, recipe_id="bgm-episode-candidates-v10-" + "0" * 16)
            return recipe

        patch.setattr(application, "_bgm_candidate_recipe", old_recipe)
        patch.setattr(application, "_BGM_CANDIDATE_WORKFLOW", V10_WORKFLOW)
        patch.setattr(cue_publication, "file_name",
                      lambda source_path, start, unresolved_extent=False:
                      f"{Path(source_path).stem} - BGM 01 ({int(start) // 60:02d}m{int(start) % 60:02d}s).flac")
        outcome = _run(tmp_path, source)
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    current_best.record_path(tmp_path / "state", run["source"]["sha256"]).unlink()
    spans = sorted((round(item["start_seconds"]), round(item["end_seconds"])) for item in run["published"])
    assert len(spans) == 3 and spans[0][0] <= 101 and spans[0][1] >= 139, spans   # Aは境界をまたいで一つ
    return source, run, {span: Path(item["path"]) for span, item in
                         zip(spans, sorted(run["published"], key=lambda item: item["start_seconds"]))}


def _by_span(record):
    return {(round(d_start), round(d_end)): row for row in record["current"]["decisions"]["old"]
            for d_start, d_end in [_span_of(record, row["public_path"])]}


def _span_of(record, public_path):
    for snapshot in [record["current"], *record["history"]]:
        for item in snapshot["files"]:
            if item["public_path"] == public_path:
                return item["start_seconds"], item["end_seconds"]
    raise KeyError(public_path)


def test_validity_and_risk_are_separate_for_older_cues_in_a_structured_program(tmp_path, gpu, monkeypatch):
    source, old_run, old = _legacy_program(tmp_path, monkeypatch, gpu)
    (a_span, b_span, c_span) = sorted(old)
    _heavy_events_in(monkeypatch, 398, 432)
    calls = len(gpu.calls)
    outcome = _run(tmp_path, source)
    assert len(gpu.calls) == calls + 1
    record = current_best.read(tmp_path / "state", old_run["source"]["sha256"])
    rows = _by_span(record)
    # A：cue自身の音に編集境界（デジタル無音）がある＝正の根拠で不正。公開から外し、本編の断片へ置き換わる。
    assert (rows[a_span]["validity"], rows[a_span]["decision"]) == ("INVALID", "retire_invalid")
    assert rows[a_span]["basis"] == ["crosses_edit_boundary"]
    # B：挿入の区画にある＝役割の危険（CMか番組の挿入か決められない）。不正の根拠ではないので、前から公開している
    # cueは UNKNOWN（危険あり）として同じファイルのまま残す。同じ区画の新しい候補は保留のまま公開しない。
    assert (rows[b_span]["validity"], rows[b_span]["decision"]) == ("UNKNOWN", "keep")
    assert rows[b_span]["risks"] == ["role_risk:within_edit_insert_section"]
    assert rows[b_span]["why"] == "validity_unknown_is_not_invalid"
    # C: a source-scene risk keeps the accepted cue UNKNOWN; a checked new
    # candidate still cannot replace it without a validated adoption improvement.
    assert (rows[c_span]["validity"], rows[c_span]["decision"]) == ("UNKNOWN", "keep")
    assert rows[c_span]["risks"] == ["quality_risk:heavy_nonmusic_event_scene"]
    public = {Path(item["public_path"]).name: item for item in record["current"]["files"]}
    assert old[c_span].name in public and old[b_span].name in public and old[a_span].name not in public
    assert current_best.pcm_sha256(old[b_span]) == public[old[b_span].name]["pcm_sha256"]
    assert current_best.pcm_sha256(old[c_span]) == public[old[c_span].name]["pcm_sha256"]
    new_spans = sorted((round(item["start_seconds"]), round(item["end_seconds"]))
                       for item in record["current"]["files"] if item["workflow"] != V10_WORKFLOW)
    assert new_spans and all(b <= 121 or a >= 120 for a, b in new_spans) and all(b <= 141 for a, b in new_spans)
    archived = {path.name for path in _history_files(tmp_path)}
    assert old[a_span].name in archived and old[b_span].name not in archived
    new_run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert not any(item["start_seconds"] >= 299 and item["end_seconds"] <= 331 for item in new_run["published"])
    recovered = [item for item in new_run["published"]
                 if item["start_seconds"] >= 397 and item["end_seconds"] <= 433]
    assert recovered and all(item["provenance"]["rendered_content_check"]["state"] == "passed"
                             for item in recovered)
    # 公開中のcueどうしは時間で重ならず、名前も重複しない。
    spans = sorted((item["start_seconds"], item["end_seconds"]) for item in record["current"]["files"])
    assert all(b0 <= a1 + 0.5 for (_, b0), (a1, _) in zip(spans, spans[1:]))
    # 巻き戻しは前の全体（v10の3件）へ戻る。
    back = application.restore_previous_bgm_result(source, state_root=tmp_path / "state")
    assert sorted(path.name for path in back.outputs) == sorted(path.name for path in old.values())


def test_a_record_reflected_under_the_previous_adoption_policy_is_reevaluated_without_inference(tmp_path, gpu,
                                                                                               monkeypatch):
    """採用の方針v1（e633e5c）は重い場面・挿入の区画の旧cueを不正として外していた。同じrunを今の方針で選び直し、
    外していた旧cueを履歴から公開へ戻す（推論しない、履歴は残る）。"""
    source, old_run, old = _legacy_program(tmp_path, monkeypatch, gpu)
    (a_span, b_span, c_span) = sorted(old)
    _heavy_events_in(monkeypatch, 398, 432)
    real_assess = application._assess_old_cues

    def v1_assess(run, run_path, old_files):
        return [{**row, "state": "INVALID", "basis": ["v1:" + risk for risk in row["risks"]]}
                if row.get("risks") and row["state"] != "INVALID" else row
                for row in real_assess(run, run_path, old_files)]

    with monkeypatch.context() as patch:
        patch.setattr(application, "_assess_old_cues", v1_assess)
        patch.setattr(application, "_BGM_ADOPTION_POLICY", "bgm-currentbest-adoption-v1")
        _run(tmp_path, source)
    sha = old_run["source"]["sha256"]
    before = current_best.read(tmp_path / "state", sha)
    assert not {old[b_span].name, old[c_span].name} & {Path(item["public_path"]).name
                                                       for item in before["current"]["files"]}
    assert application.existing_bgm_candidates(source, sha, state_root=tmp_path / "state") is None
    calls = len(gpu.calls)
    again = _run(tmp_path, source)
    assert len(gpu.calls) == calls and again.reused is True      # 推論しない
    after = current_best.read(tmp_path / "state", sha)
    assert after["current"]["adoption_policy"] == application._BGM_ADOPTION_POLICY
    assert after["current"]["transition"] == "adoption_policy_reevaluation"
    names = {Path(item["public_path"]).name for item in after["current"]["files"]}
    assert {old[b_span].name, old[c_span].name} <= names and old[a_span].name not in names
    for span in (b_span, c_span):
        assert current_best.pcm_sha256(tmp_path / "LSS" / "program - BGM" / old[span].name) == \
            current_best.pcm_sha256(next(path for path in _history_files(tmp_path) if path.name == old[span].name))
    assert len(after["history"]) == len(before["history"]) + 1   # 方針v1の集まりも履歴に残る
    assert application.existing_bgm_candidates(source, sha, state_root=tmp_path / "state")["k"] == "done"


def test_a_chain_of_adoption_policy_changes_reevaluates_from_the_set_before_the_run(tmp_path, gpu, monkeypatch):
    """方針の選び直しを重ねても（v1→v2→今）、基準はそのrunを反映する前の公開の集まり。前の選び直しの結果（前の方針で
    外した後の集まり）を基準にすると、外したままのcueが戻らない。"""
    source, old_run, old = _legacy_program(tmp_path, monkeypatch, gpu)
    (a_span, b_span, c_span) = sorted(old)
    _heavy_events_in(monkeypatch, 398, 432)
    real_assess = application._assess_old_cues

    def assess_retiring(kinds):
        def assess(run, run_path, old_files):
            return [{**row, "state": "INVALID", "basis": [f"old_policy:{risk}" for risk in row["risks"]]}
                    if set(row.get("risks") or []) & kinds and row["state"] != "INVALID" else row
                    for row in real_assess(run, run_path, old_files)]
        return assess

    sha = old_run["source"]["sha256"]
    with monkeypatch.context() as patch:   # v1：重い場面・挿入の区画とも外す
        patch.setattr(application, "_assess_old_cues", assess_retiring({"quality_risk:heavy_nonmusic_event_scene",
                                                                         "role_risk:within_edit_insert_section"}))
        patch.setattr(application, "_BGM_ADOPTION_POLICY", "bgm-currentbest-adoption-v1")
        _run(tmp_path, source)
    with monkeypatch.context() as patch:   # v2：挿入の区画だけ外す（選び直し）
        patch.setattr(application, "_assess_old_cues", assess_retiring({"role_risk:within_edit_insert_section"}))
        patch.setattr(application, "_BGM_ADOPTION_POLICY", "bgm-currentbest-adoption-v2")
        _run(tmp_path, source)
    middle = current_best.read(tmp_path / "state", sha)
    assert middle["current"]["transition"] == "adoption_policy_reevaluation"
    middle_names = {Path(item["public_path"]).name for item in middle["current"]["files"]}
    assert old[c_span].name in middle_names and old[b_span].name not in middle_names
    calls = len(gpu.calls)
    _run(tmp_path, source)                 # 今の方針
    assert len(gpu.calls) == calls         # 推論しない
    after = current_best.read(tmp_path / "state", sha)
    names = {Path(item["public_path"]).name for item in after["current"]["files"]}
    assert {old[b_span].name, old[c_span].name} <= names and old[a_span].name not in names
    assert after["current"]["evaluated_runs"] == middle["current"]["evaluated_runs"]
    # 履歴は全て残る（v10の集まり・v1の結果・v2の結果）。
    assert len(after["history"]) == len(middle["history"]) + 1
    assert after["history"][:len(middle["history"])] == middle["history"]
    spans = sorted((item["start_seconds"], item["end_seconds"]) for item in after["current"]["files"])
    assert all(b0 <= a1 + 0.5 for (_, b0), (a1, _) in zip(spans, spans[1:]))
    assert len(names) == len(after["current"]["files"])
