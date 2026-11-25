from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mutagen.flac import FLAC

from lss.bgm import cue_publication, listening, scene_names
from lss.core import audio_identity

TAGS = {"title": "【公式】アニメ「星空冒険団」第137話「出発！星空の港へ！！」",
        "DESCRIPTION": "あらすじ\n\n■キャスト\nハル：声優甲\nカイ：声優乙\nソラ：声優丙\n\n■スタッフ\n監督：監督甲"}


def facts(episode=137):
    found = scene_names.source_facts(f"series-alpha-{episode}.mkv", tags={**TAGS, "title": TAGS["title"].replace("137", str(episode))})
    return found


def heard(*texts, start=100.0):
    return [{"start": start + i * 3, "end": start + i * 3 + 2, "text": t} for i, t in enumerate(texts)]


def order(name, use="ソラがミズウオを探す", direction="ゆったりと", category="日常・のどか"):
    return {"name": name, "use": use, "direction": direction, "category": category, "evidence": "ミズウオだ"}


def record(root, asset, name, *, episode=137, start=100.0, end=None, lines=("ミズウオだ", "探そう"), **kw):
    end = start + 60.0 if end is None else end
    return scene_names.record_cue(root, asset, pcm_sha256="p-" + asset, source_sha256="s" * 64, facts=facts(episode),
                                  start_seconds=start, end_seconds=end, heard=heard(*lines, start=start),
                                  order=order(name, **kw), model={"id": "test"})


def test_source_facts_read_the_files_own_title_episode_and_cast():
    found = facts()
    assert found["episode"] == 137 and found["work_title"] == "星空冒険団"
    assert found["cast"] == ["ハル", "カイ", "ソラ"] and found["work_key"].startswith("work-")


def test_explicit_program_announcements_remain_outside_but_short_cues_are_not_classified_by_time(tmp_path):
    entry = record(tmp_path, "a", "告知の曲", start=10, end=40, lines=("毎週金曜よる6時55分から放送中",))
    assert entry["name"] is None and entry["category"] == "本編外" and entry["guards"]["outside_episode"] == "announcement"
    assert scene_names.outside_episode(640, 650, []) is None


def test_a_name_never_heard_and_not_in_the_cast_is_withheld_but_a_heard_one_is_kept(tmp_path):
    invented = record(tmp_path, "a", "メガソックウサの襲来", use="メガソックウサが襲う")
    assert invented["name"] is None and invented["guards"]["unheard_names"] == ["メガソックウサ"]
    kept = record(tmp_path, "b", "ミズウオを探せ", lines=("ソラがミズウオを探す", "ミズウオだ"),
                  use="ソラがミズウオを探す")
    assert kept["name"] == "ミズウオを探せ" and not kept["guards"]


def test_synthetic_order_use_only_unheard_name_is_rejected(tmp_path):
    import json

    cue, source_facts = _saved_replay_cue("unrelated")
    assert cue["old_shown_name"] is None
    assert cue["old_eval_guards"]["unheard_names"] == ["ウェーノベル"]

    entry = scene_names.record_cue(
        tmp_path, cue["asset_id"], pcm_sha256=cue["pcm_sha256"], source_sha256=cue["source_sha256"],
        facts=source_facts, start_seconds=cue["bounds_seconds"][0], end_seconds=cue["bounds_seconds"][1],
        heard=cue["heard"], order=cue["old_order"], model={"id": "synthetic-order-replay"},
    )

    assert entry["name"] is None and entry["use"] is None
    assert entry["guards"]["unheard_names"] == ["ウェーノベル"]
    assert entry["proposed"]["name"] == "眠りの罠"
    assert entry["proposed"]["use"] == "ウェーノベルの眠りの罠"


def test_cast_alone_does_not_witness_an_actor_in_scene_use(tmp_path):
    source_facts = {**facts(), "cast": ["ウェーノベル"]}
    entry = scene_names.record_cue(
        tmp_path, "cast-backed", pcm_sha256="pcm", source_sha256="source", facts=source_facts,
        start_seconds=100., end_seconds=140., heard=heard("ラッチョンレラーラッチョン"),
        order={"use": "ウェーノベルが眠る", "direction": "不穏に", "category": "不穏・怪しい",
               "name": "眠りの罠", "evidence": "ラッチョンレラーラッチョン"},
        model={"id": "test"},
    )
    assert entry["name"] == "眠りの罠" and entry["use"] is None
    assert entry["proposed"]["use"] == "ウェーノベルが眠る"
    assert entry["guards"]["field_guard"]["use"] == {
        "reason": "katakana_token_not_witnessed_in_scene_context", "tokens": ["ウェーノベル"]}
    assert "unheard_names" not in entry["guards"]
    assert scene_names.read(tmp_path)["policy_id"] == scene_names.POLICY_ID


def _saved_replay_cue(cue_id):
    cases = {
        "lead-in": (1, "出発の調べ", "冒険の出発 ハルの決意", "ハル", "ハルが出発する", "出発しよう", 90.),
        "unheard-use": (2, "広がる世界へ", "ハルの決意", "ハル", None, "出発しよう", 100.),
        "unrelated": (3, "眠りの罠", "ウェーノベルの眠りの罠", None, None, "ラッチョンレラーラッチョン", 100.),
        "voiced": (4, "導かれる扉の先", "メガソックウサの導き", None, None, "メガソッグーザが来た", 100.),
        "long-vowel": (4, "導かれる扉の先", "メガソックウサの導き", None, None, "メガソックーサが来た", 100.),
        "cast-one": (5, "謎の面の下", "ミナが面を取る", "ミナ", None, "面を取ろう", 100.),
        "cast-two": (5, "謎の面の下", "ミナが面を取る", "ミナ", None, "面を見よう", 100.),
    }
    for index, title in enumerate(["朝の風", "昼の波", "夕べの雲", "夜の星", "谷の道", "森の音", "海の灯", "丘の家"]):
        cases[f"accepted-{index}"] = (10 + index, title, "風が吹く", None, None, "風が吹く", 100.)
    syllables = ["ラ", "リ", "ル", "レ", "ロ", "ナ", "ニ", "ヌ"]
    for index in range(58):
        actor = "カナ" + syllables[index // 8] + syllables[index % 8]
        cases[f"rejected-{index}"] = (30 + index, actor + "の足音", actor + "が来た", None, None, "風が吹く", 100.)
    episode, name, use, actor, lead, text, line_start = cases[cue_id]
    lines = [{"start": line_start, "end": line_start + 3, "text": lead or text}]
    if lead:
        lines.append({"start": 100., "end": 103., "text": text})
    order_data = {"name": name, "use": use, "direction": "静かに", "category": "日常・のどか", "evidence": text}
    row = {"id": cue_id, "episode": str(episode), "asset_id": "synthetic-" + cue_id,
           "pcm_sha256": "pcm-" + cue_id, "source_sha256": "source-" + str(episode),
           "bounds_seconds": [100., 140.], "heard": lines, "old_order": order_data,
           "old_shown_name": None if cue_id == "unrelated" or cue_id.startswith("rejected-") else name,
           "old_eval_guards": {"unheard_names": ["ウェーノベル"]} if cue_id == "unrelated" else {}}
    source_facts = {"title": f"星空冒険団 第{episode}話", "work_title": "星空冒険団", "work_key": "synthetic",
                    "episode": episode, "cast": [actor] if actor else []}
    return row, source_facts


def test_saved_replay_lead_in_supports_haru_without_broadening_old_admission(tmp_path):
    row, source_facts = _saved_replay_cue("lead-in")
    entry = scene_names.record_cue(
        tmp_path, row["asset_id"], pcm_sha256=row["pcm_sha256"], source_sha256=row["source_sha256"],
        facts=source_facts, start_seconds=row["bounds_seconds"][0], end_seconds=row["bounds_seconds"][1],
        heard=row["heard"], order=row["old_order"], model={"id": "synthetic-order-replay"})
    assert entry["name"] == "出発の調べ" and entry["use"] == "冒険の出発 ハルの決意"
    assert "field_guard" not in entry["guards"]


def test_saved_replay_unwitnessed_haru_use_is_nulled_but_title_and_proposal_stay(tmp_path):
    row, source_facts = _saved_replay_cue("unheard-use")
    entry = scene_names.record_cue(
        tmp_path, row["asset_id"], pcm_sha256=row["pcm_sha256"], source_sha256=row["source_sha256"],
        facts=source_facts, start_seconds=row["bounds_seconds"][0], end_seconds=row["bounds_seconds"][1],
        heard=row["heard"], order=row["old_order"], model={"id": "synthetic-order-replay"})
    assert entry["name"] == row["old_shown_name"] == "広がる世界へ"
    assert entry["use"] is None and entry["attempt_status"] == "accepted"
    assert entry["proposed"] == {**row["old_order"]}
    assert entry["guards"]["field_guard"]["use"] == {
        "reason": "katakana_token_not_witnessed_in_scene_context", "tokens": ["ハル"]}


def test_saved_replay_existing_combined_admission_still_holds_unrelated(tmp_path):
    row, source_facts = _saved_replay_cue("unrelated")
    entry = scene_names.record_cue(
        tmp_path, row["asset_id"], pcm_sha256=row["pcm_sha256"], source_sha256=row["source_sha256"],
        facts=source_facts, start_seconds=row["bounds_seconds"][0], end_seconds=row["bounds_seconds"][1],
        heard=row["heard"], order=row["old_order"], model={"id": "synthetic-order-replay"})
    assert entry["name"] is None and entry["use"] is None
    assert entry["guards"]["unheard_names"] == ["ウェーノベル"]
    assert "field_guard" not in entry["guards"]


def test_saved_replay_canonical_public_spelling_witness_keeps_use(tmp_path):
    import json

    row, source_facts = _saved_replay_cue("voiced")
    import hashlib
    text = "メガソックウサが導く、星の扉。"
    source_facts = {**source_facts, "published_name_sources": [{
        "source_kind": "official_episode", "url": "https://example.org/episodes/4",
        "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}]}
    entry = scene_names.record_cue(
        tmp_path, row["asset_id"], pcm_sha256=row["pcm_sha256"], source_sha256=row["source_sha256"],
        facts=source_facts, start_seconds=row["bounds_seconds"][0], end_seconds=row["bounds_seconds"][1],
        heard=row["heard"], order=row["old_order"], model={"id": "synthetic-order-replay"})
    assert entry["name"] == "導かれる扉の先" and entry["use"] == "メガソックウサの導き"
    assert "field_guard" not in entry["guards"]


def test_synthetic_replay_keeps_title_admission_and_only_withholds_unwitnessed_uses(tmp_path):
    import hashlib

    text = "メガソックウサが導く、星の扉。"
    source = {"source_kind": "official_episode", "url": "https://example.org/episodes/4", "text": text,
              "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
    case_ids = ["lead-in", "unheard-use", "unrelated", "voiced", "long-vowel", "cast-one", "cast-two"]
    case_ids += [f"accepted-{i}" for i in range(8)] + [f"rejected-{i}" for i in range(58)]
    changed_uses, admitted = [], 0
    for cue_id in case_ids:
        row, source_facts = _saved_replay_cue(cue_id)
        if cue_id in ("voiced", "long-vowel"):
            source_facts = {**source_facts, "published_name_sources": [source]}
        entry = scene_names.record_cue(
            tmp_path, row["asset_id"], pcm_sha256=row["pcm_sha256"], source_sha256=row["source_sha256"],
            facts=source_facts, start_seconds=row["bounds_seconds"][0], end_seconds=row["bounds_seconds"][1],
            heard=row["heard"], order=row["old_order"], model={"id": "synthetic-replay"},
            attempted_at=1_800_000_000.0, completed_at=1_800_000_000.0)
        assert entry["name"] == row["old_shown_name"], cue_id
        accepted = cue_id != "unrelated" and not cue_id.startswith("rejected-")
        assert (entry["name"] is not None) == accepted, cue_id
        admitted += accepted
        if accepted and entry["use"] != row["old_order"]["use"]:
            changed_uses.append(cue_id)
    assert len(case_ids) == 73
    assert admitted == 14
    assert changed_uses == ["unheard-use", "cast-one", "cast-two"]


@pytest.mark.parametrize("cue_id", ["cast-one", "cast-two"])
def test_saved_replay_cast_only_mina_use_is_nulled_without_changing_generic_title(tmp_path, cue_id):
    row, source_facts = _saved_replay_cue(cue_id)
    entry = scene_names.record_cue(
        tmp_path, row["asset_id"], pcm_sha256=row["pcm_sha256"], source_sha256=row["source_sha256"],
        facts=source_facts, start_seconds=row["bounds_seconds"][0], end_seconds=row["bounds_seconds"][1],
        heard=row["heard"], order=row["old_order"], model={"id": "synthetic-order-replay"})
    assert entry["name"] == "謎の面の下" and entry["use"] is None
    assert entry["proposed"]["use"] == row["old_order"]["use"]
    assert entry["guards"]["field_guard"]["use"]["tokens"] == ["ミナ"]


def test_null_scene_use_refreshes_flac_without_actor_in_order_menu(tmp_path):
    path, asset = _cue(tmp_path)
    state = tmp_path / "state"
    pcm = audio_identity.audio_identity(path)["pcm_sha256"]
    entry = scene_names.record_cue(
        state, asset, pcm_sha256=pcm, source_sha256="a" * 64,
        facts={**facts(), "cast": ["ウェーノベル"]},
        start_seconds=100, end_seconds=105, heard=heard("ラッチョンレラーラッチョン"),
        order={"name": "眠りの罠", "use": "ウェーノベルが眠る", "direction": "不穏に",
               "category": "不穏・怪しい", "evidence": "ラッチョンレラーラッチョン"},
        model={"id": "test"})
    assert entry["name"] == "眠りの罠" and entry["use"] is None
    scene_names.settle(state)
    listening.finish(path, state)
    tags = FLAC(path)
    assert tags["TITLE"] == ["眠りの罠（仮題）"]
    assert tags["ORDER_MENU"] == ["不穏に"]
    assert "ウェーノベル" not in str(tags.get("ORDER_MENU"))
    assert audio_identity.audio_identity(path)["pcm_sha256"] == pcm


def test_copied_fixed_recording_name_survives_use_guard_change(tmp_path):
    state = tmp_path / "state"
    memory_data = {"labels": {"1": {"members": ["first", "copy"]}}}
    first = record(state, "first", "ミズウオを探せ", use="ソラがミズウオを探す")
    scene_names.settle(state, memory_data)
    recording = scene_names.fixed_recording(scene_names.read(state), "copy", memory_data)
    assert recording is not None
    key, fixed = recording
    use, _, direction = (fixed.get("order") or "").rpartition(" ")

    scene_names.record_cue(
        state, "copy", pcm_sha256="copy-pcm", source_sha256="s" * 64, facts=facts(),
        start_seconds=200., end_seconds=260., heard=[],
        order={"use": use or None, "direction": direction or None,
               "category": fixed.get("category"), "name": None},
        model={"id": "fixed_name_of_recording", "recording": key},
    )
    scene_names.settle(state, memory_data)

    assert first["name"] == "ミズウオを探せ"
    assert scene_names.presentation(state, "copy")["name"] == "ミズウオを探せ"
    assert scene_names.read(state)["recordings"][key]["name"] == "ミズウオを探せ"
    assert "unheard_names" not in scene_names.cue_record(state, "copy")["guards"]


def test_fixed_recording_model_id_cannot_bypass_unheard_inferred_use(tmp_path):
    evidence = "ミズウオを探せ"
    entry = scene_names.record_cue(
        tmp_path, "forged-fixed-copy", pcm_sha256="pcm", source_sha256="source", facts=facts(),
        start_seconds=100., end_seconds=140., heard=heard(evidence),
        order={"use": "ウェーノベルが眠る", "direction": "不穏に", "category": "不穏・怪しい",
               "name": "眠りの罠", "evidence": evidence},
        model={"id": "fixed_name_of_recording", "recording": "unrelated-recording"},
    )
    assert entry["name"] is None and entry["use"] is None
    assert entry["guards"]["unheard_names"] == ["ウェーノベル"]


@pytest.mark.parametrize("lines,evidence,reason", [
    ([], "湖の救出", "no_timed_occurrence_text"),
    ([{"start": 80., "end": 90., "text": "湖の救出"}], "湖の救出", "no_timed_occurrence_text"),
    (heard("ミズウオだ"), "湖の救出", "quote_not_in_occurrence_text"),
    (heard("ミズウオだ"), None, "quote_not_in_occurrence_text"),
])
def test_new_scene_name_requires_timed_observation_and_an_actual_quote(tmp_path, lines, evidence, reason):
    entry = scene_names.record_cue(tmp_path, "new", pcm_sha256="pcm", source_sha256="source", facts=facts(),
        start_seconds=100., end_seconds=140., heard=lines,
        order={**order("湖の救出"), "evidence": evidence}, model={"id": "invented-scene"})
    scene_names.settle(tmp_path)
    assert entry["name"] is None and entry["use"] is None
    assert entry["guards"]["scene_observation"] == reason
    assert entry["proposed"]["name"] == "湖の救出"  # inference retained separately from observed facts
    assert scene_names.presentation(tmp_path, "new")["name"] is None


def test_context_music_title_is_bound_to_inputs_and_does_not_assert_a_cast_scene(tmp_path):
    source_facts = facts()
    measured_music = {"tempo": {"status": "measured", "bpm": 120}}
    observations = []
    binding = scene_names.context_music_evidence(source_facts, observations, measured_music)
    entry = scene_names.record_cue(
        tmp_path, "context-title", pcm_sha256="pcm", source_sha256="source", facts=source_facts,
        start_seconds=100., end_seconds=140., heard=observations,
        order={"name_basis": "context_music", "name": "ハルのテーマ", "use": None, "evidence": None,
               "direction": "しっとりと", "category": "悲しみ・切なさ"},
        model={"id": "test"}, music=measured_music, name_basis_evidence=binding)
    scene_names.settle(tmp_path)
    assert entry["name"] == "ハルのテーマ" and entry["name_basis"] == "context_music"
    assert entry["use"] is None and entry["category"] == "悲しみ・切なさ"
    assert entry["direction"] == "しっとりと"
    assert entry["name_basis_evidence"]["scene_asserted"] is False
    assert entry["name_basis_evidence"]["source_context"]["cast"] == ["ハル", "カイ", "ソラ"]
    assert scene_names.presentation(tmp_path, "context-title")["evidence"]["context"]["basis"] == "context_music"


def test_context_music_cannot_be_used_to_smuggle_a_scene_claim_or_unbound_name(tmp_path):
    source_facts = facts()
    music = {"tempo": {"status": "measured", "bpm": 120}}
    forged = scene_names.record_cue(
        tmp_path, "unbound-context", pcm_sha256="pcm1", source_sha256="source", facts=source_facts,
        start_seconds=100., end_seconds=140., heard=[],
        order={"name_basis": "context_music", "name": "ハルのテーマ", "use": None, "evidence": None,
               "direction": None, "category": None}, model={"id": "test"}, music=music)
    assert forged["name"] is None
    assert forged["guards"]["name_basis"] == "context_music_inputs_missing_or_mismatched"

    binding = scene_names.context_music_evidence(source_facts, [], music)
    asserted = scene_names.record_cue(
        tmp_path, "asserted-context", pcm_sha256="pcm2", source_sha256="source", facts=source_facts,
        start_seconds=100., end_seconds=140., heard=[],
        order={"name_basis": "context_music", "name": "ハルの勝利", "use": "ハルが勝つ",
               "evidence": "なし", "direction": None, "category": None},
        model={"id": "test"}, music=music, name_basis_evidence=binding)
    assert asserted["name"] is None and asserted["use"] is None
    assert asserted["guards"]["scene_observation"] == "context_music_cannot_assert_a_scene"

    unsupported = scene_names.record_cue(
        tmp_path, "unsupported-context", pcm_sha256="pcm3", source_sha256="source", facts=source_facts,
        start_seconds=100., end_seconds=140., heard=[],
        order={"name_basis": "context_music", "name": "ウェーノベルのテーマ", "use": None,
               "evidence": None, "direction": None, "category": None},
        model={"id": "test"}, music=music, name_basis_evidence=binding)
    assert unsupported["name"] is None
    assert unsupported["guards"]["unwitnessed_context_names"] == ["ウェーノベル"]


def test_context_music_basis_does_not_accept_a_name_without_any_actual_inputs(tmp_path):
    facts_without_context = {"title": "", "work_title": "", "work_key": "empty", "episode": None, "cast": []}
    music = {}
    binding = scene_names.context_music_evidence(facts_without_context, [], music, "測定情報なし")
    entry = scene_names.record_cue(
        tmp_path, "empty-context", pcm_sha256="pcm", source_sha256="source", facts=facts_without_context,
        start_seconds=100., end_seconds=140., heard=[],
        order={"name_basis": "context_music", "name": "静かな調べ", "use": None,
               "evidence": None, "direction": None, "category": None}, model={"id": "test"},
        music=music, name_basis_evidence=binding, measured_music_description="測定情報なし")
    assert entry["name"] is None
    assert entry["guards"]["name_basis"] == "context_music_has_no_source_or_measured_inputs"


@pytest.mark.parametrize("title,accepted", [("夜のメロディー", True), ("ホシナカマの旅路", False)])
def test_context_title_vocabulary_is_general_or_supplied_by_the_actual_work(tmp_path, title, accepted):
    source_facts = {"title": "別作品の第1話", "work_title": "別作品", "work_key": "other", "episode": 1, "cast": []}
    music = {"tempo": {"status": "measured", "bpm": 90}}
    binding = scene_names.context_music_evidence(source_facts, [], music)
    entry = scene_names.record_cue(
        tmp_path, "other-work", pcm_sha256="pcm", source_sha256="source", facts=source_facts,
        start_seconds=100., end_seconds=140., heard=[],
        order={"name_basis": "context_music", "name": title, "use": None,
               "evidence": None, "direction": None, "category": None}, model={"id": "test"},
        music=music, name_basis_evidence=binding)
    assert bool(entry["name"]) is accepted
    if not accepted:
        assert entry["guards"]["unwitnessed_context_names"] == ["ホシナカマ"]


def test_scene_names_do_not_exempt_another_works_proper_nouns_from_observation(tmp_path):
    source_facts = {"title": "別作品の第1話", "work_title": "別作品", "work_key": "other", "episode": 1, "cast": []}
    entry = scene_names.record_cue(
        tmp_path, "other-scene", pcm_sha256="pcm", source_sha256="source", facts=source_facts,
        start_seconds=100., end_seconds=140., heard=heard("行こう"),
        order={"name": "ホシナカマのテーマ", "use": None, "evidence": "行こう",
               "direction": None, "category": None}, model={"id": "test"})
    assert entry["name"] is None and entry["guards"]["unheard_names"] == ["ホシナカマ"]


def test_scene_model_order_parser_keeps_only_known_name_basis_values():
    from lss.bgm.scene_models import SceneModels

    models = SceneModels()
    models._ask = lambda *_args, **_kwargs: (
        '{"name_basis":"context_music","use":null,"direction":null,"category":null,'
        '"name":"ハルのテーマ","evidence":null}')
    assert models.order("prompt")["name_basis"] == "context_music"
    models._ask = lambda *_args, **_kwargs: (
        '{"name_basis":"invented_scene","use":null,"direction":null,"category":null,'
        '"name":"ハルのテーマ","evidence":null}')
    assert models.order("prompt")["name_basis"] is None


def test_transcription_ffmpeg_is_hidden_without_loading_models(tmp_path, monkeypatch):
    import subprocess
    import sys
    from types import SimpleNamespace
    from lss import paths
    from lss.bgm.scene_models import SceneModels
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    executable.write_bytes(b"test ffmpeg")
    executable.chmod(0o755)
    expected_executable = str(executable.resolve())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("PATHEXT", ".exe")
    monkeypatch.setattr(paths, "resource_root", lambda: tmp_path / "resources_without_ffmpeg")
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=None))
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=np.zeros(3200, dtype=np.float32).tobytes())
    monkeypatch.setattr(subprocess, "run", run)
    models = SceneModels()
    monkeypatch.setattr(models, "_gpu", lambda: None)
    monkeypatch.setattr(models, "_drop", lambda *args: None)
    models._asr = lambda *args, **kwargs: {"chunks": [{"timestamp": [0., 1.], "text": "観測した台詞"}]}
    assert models.transcribe(tmp_path / "episode.mkv", [(10., 11.)])[0]["text"] == "観測した台詞"
    assert [command[1] for command, _ in calls] == ["-version", "-v"]
    for command, kwargs in calls:
        assert command[0] == expected_executable
        assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)


def memory(*groups):
    return {"labels": {str(n): {"members": list(members)} for n, members in enumerate(groups, 1)}}


def test_the_first_name_of_a_recording_stays_when_it_recurs_in_another_episode(tmp_path):
    record(tmp_path, "a", "ミズウオを探せ", episode=137)
    scene_names.settle(tmp_path, memory(["a"]))
    record(tmp_path, "b", "別の名前", episode=150)
    scene_names.settle(tmp_path, memory(["a", "b"]))
    for asset in ("a", "b"):
        shown = scene_names.presentation(tmp_path, asset)
        assert shown["name"] == "ミズウオを探せ"
        assert shown["evidence"]["context"]["description"] == "ミズウオを探せ"


def test_a_cue_named_alone_keeps_its_name_when_it_later_joins_a_group(tmp_path):
    record(tmp_path, "a", "ミズウオを探せ", episode=137)
    scene_names.settle(tmp_path, None)
    record(tmp_path, "b", "別の名前", episode=150)
    scene_names.settle(tmp_path, memory(["b", "a"]))
    data = scene_names.read(tmp_path)
    assert data["recordings"]["label:1"]["name"] == "ミズウオを探せ" and "asset:a" not in data["recordings"]


def test_m_numbers_follow_the_score_sheet_sections_and_a_second_session(tmp_path):
    record(tmp_path, "battle", "ボスに挑め", episode=137, start=500, category="バトル", lines=("いくぞ", "ミズウオだ"))
    record(tmp_path, "daily", "ゆるい散歩", episode=137, start=300, category="日常・のどか")
    record(tmp_path, "sad", "別れの涙", episode=138, start=100, category="悲しみ・切なさ")
    record(tmp_path, "late", "新たな脅威", episode=152, start=100, category="バトル")
    scene_names.set_work_prefix(tmp_path, facts()["work_key"], "lr", basis="test")
    scene_names.settle(tmp_path, None)
    codes = {a: scene_names.presentation(tmp_path, a)["tags"]["M_NUMBER"] for a in ("battle", "daily", "sad", "late")}
    # battle (戦闘描写) -> daily (情景描写) -> sad (心理描写); 15 episodes later -> second session
    assert codes == {"battle": "LR01", "daily": "LR02", "sad": "LR03", "late": "LR101"}


def test_adding_an_earlier_episode_may_renumber_but_never_renames(tmp_path):
    record(tmp_path, "b", "ゆるい散歩", episode=140, start=300)
    scene_names.settle(tmp_path, None)
    before = scene_names.presentation(tmp_path, "b")
    record(tmp_path, "a", "朝の情景", episode=137, start=200)
    scene_names.settle(tmp_path, None)
    after = scene_names.presentation(tmp_path, "b")
    assert after["name"] == before["name"] == "ゆるい散歩"
    assert scene_names.presentation(tmp_path, "a")["tags"]["M_NUMBER"].endswith("01")
    assert after["tags"]["M_NUMBER"].endswith("02") and before["tags"]["M_NUMBER"].endswith("01")


def test_a_prefix_chosen_under_an_older_policy_is_chosen_again_but_a_users_is_kept(tmp_path):
    key = facts()["work_key"]
    scene_names.update(tmp_path, lambda data: data["works"].update(
        {key: {"prefix": "PM", "prefix_basis": "model", "prefix_policy": None}, "w2": {"prefix": "MY", "prefix_basis": "user"}}))
    assert scene_names.prefix_needed(tmp_path, key) and not scene_names.prefix_needed(tmp_path, "w2")
    assert scene_names.set_work_prefix(tmp_path, key, "LR", basis="model") == "LR"
    assert scene_names.read(tmp_path)["works"][key]["previous_prefixes"][0]["prefix"] == "PM"
    assert scene_names.set_work_prefix(tmp_path, "w2", "ZZ", basis="model") == "MY"


def test_a_series_is_told_apart_by_the_official_site_its_file_names():
    one = scene_names.source_facts("a.mkv", tags={"title": "アニメ「星空冒険団」第1話",
                                                 "description": "公式サイトはこちら https://example.jp/series-alpha2023/"})
    other = scene_names.source_facts("b.mkv", tags={"title": "アニメ「星空冒険団」第1話",
                                                   "description": "公式サイトはこちら https://example.jp/series-alpha2019/"})
    assert one["work_title"] == other["work_title"] and one["work_key"] != other["work_key"]


def test_the_work_prefix_is_fixed_once(tmp_path):
    key = facts()["work_key"]
    assert scene_names.set_work_prefix(tmp_path, key, "LR", basis="model") == "LR"
    assert scene_names.set_work_prefix(tmp_path, key, "PM", basis="model") == "LR"
    with pytest.raises(scene_names.SceneNamesError):
        scene_names.set_work_prefix(tmp_path, "w", "ポ", basis="model")


RATE = 44100


def _cue(tmp_path):
    frames = RATE * 5
    t = np.arange(frames) / RATE
    signal = np.column_stack((0.2 * np.sin(2 * np.pi * 220 * t), 0.1 * np.sin(2 * np.pi * 330 * t)))
    key = cue_publication.cue_key("a" * 64, "accepted-recipe", 0, frames, RATE)
    path = tmp_path / "source - BGM 00m00s.flac"
    cue_publication.publish(signal, RATE, path, key=key, title="source BGM 00m00s", album="source BGM",
                            input_sha256="a" * 64, recipe_id="accepted-recipe", version="0.1.0",
                            provenance={"type": "bgm_published_cue", "source": {"sha256": "a" * 64},
                                        "start_frame": 0, "end_frame": frames, "sample_rate": RATE,
                                        "start_seconds": 100.0})
    return path, f"bgm-cue:{key}"


def _named(state, path, asset):
    pcm = audio_identity.audio_identity(path)["pcm_sha256"]
    scene_names.record_cue(state, asset, pcm_sha256=pcm, source_sha256="a" * 64, facts=facts(),
                           start_seconds=100, end_seconds=105,
                           heard=heard("ソラがミズウオを探す", "ミズウオだ"),
                           order=order("ミズウオを探せ"), model={"id": "test"})
    scene_names.set_work_prefix(state, facts()["work_key"], "LR", basis="test")
    scene_names.settle(state, None)


def test_the_listening_file_shows_the_scene_name_and_separate_number_and_order_tags(tmp_path):
    path, asset = _cue(tmp_path)
    state = tmp_path / "state"
    _named(state, path, asset)
    listening.finish(path, state)
    tags = FLAC(path)
    assert tags["TITLE"] == ["ミズウオを探せ（仮題）"]
    assert tags["M_NUMBER"] == ["LR01"] and tags["ORDER_MENU"] == ["ソラがミズウオを探す ゆったりと"]
    assert "TRACKNUMBER" not in tags and "LR01" not in tags["TITLE"][0]
    assert listening.current([path], state)


def test_a_renumbered_recording_refreshes_only_its_tags_and_a_foreign_edit_is_restored(tmp_path):
    path, asset = _cue(tmp_path)
    state = tmp_path / "state"
    _named(state, path, asset)
    listening.finish(path, state)
    scene_names.record_cue(state, "bgm-cue:" + "e" * 64, pcm_sha256="x", source_sha256="b" * 64, facts=facts(120),
                           start_seconds=50, end_seconds=90, heard=heard("ミズウオだ"),
                           order=order("朝の情景"), model={"id": "test"})
    scene_names.settle(state, None)
    assert not listening.current([path], state)
    listening.finish(path, state)
    # Episode 120 is now the earliest known: 137 is 17 episodes later, so it moves to the second session.
    assert FLAC(path)["M_NUMBER"] == ["LR101"] and FLAC(path)["TITLE"] == ["ミズウオを探せ（仮題）"]
    audio = FLAC(path)
    audio["M_NUMBER"] = "MY-07"
    audio.save()
    assert not listening.current([path], state)
    listening.finish(path, state)
    assert FLAC(path)["M_NUMBER"] == ["LR101"]


@pytest.mark.parametrize("owner", ["lss", "official", "related"])
def test_later_fixed_name_refreshes_without_renumbering_or_reseparating(tmp_path, monkeypatch, owner):
    from lss.bgm import current_best, scene_job

    path, asset = _cue(tmp_path)
    state = tmp_path / "state"
    pcm = audio_identity.audio_identity(path)["pcm_sha256"]
    kwargs = dict(pcm_sha256=pcm, source_sha256="a" * 64, facts=facts(),
                  start_seconds=100, end_seconds=105,
                  heard=heard("ソラがミズウオを探す", "ミズウオだ"), model={"id": "test"})
    scene_names.record_cue(state, asset, order=order(None), **kwargs)
    scene_names.set_work_prefix(state, facts()["work_key"], "LR", basis="test")
    scene_names.settle(state)
    listening.finish(path, state)
    if owner in ("official", "related"):
        relation = "exact_version" if owner == "official" else "same_theme"
        listening.finish(path, state, extra_evidence={owner: {
            "asset_id": asset, "title": "確認済みの曲名", "verified": True,
            "relation": relation, "evidence_ref": "test:verified"}})
    assert listening.current([path], state)
    before = FLAC(path)
    before_title = list(before["TITLE"])
    stable = {key: list(before[key]) for key in ("M_NUMBER", "ORDER_MENU")}

    scene_names.record_cue(state, asset, order=order("ミズウオを探せ"), **kwargs)
    scene_names.settle(state)
    stronger = owner in ("official", "related")
    assert listening.current([path], state) == stronger
    monkeypatch.setattr(current_best, "published_files", lambda *args: {
        asset.removeprefix("bgm-cue:"): {"public_path": str(path), "pcm_sha256": pcm}})
    result = scene_job.refresh_work(state, facts()["work_key"], None)
    assert result == {"refreshed": 0 if stronger else 1,
                      "already_current": 1 if stronger else 0, "pending": []}
    after = FLAC(path)
    assert after["TITLE"] == (["ミズウオを探せ（仮題）"] if owner == "lss" else before_title)
    assert {key: list(after[key]) for key in stable} == stable
    assert audio_identity.audio_identity(path)["pcm_sha256"] == pcm
    assert listening.current([path], state)
    assert scene_job.refresh_work(state, facts()["work_key"], None)["already_current"] == 1


def test_source_video_metadata_export_is_hidden_and_keeps_observed_metadata(monkeypatch, tmp_path):
    import subprocess
    from types import SimpleNamespace
    from lss import paths

    bundled = tmp_path / "fake-resources" / "bin" / "ffmpeg.exe"
    bundled.parent.mkdir(parents=True, exist_ok=True)
    bundled.write_bytes(b"fixture executable marker")
    monkeypatch.setattr(paths, "resource_root", lambda: bundled.parents[1])

    calls = []
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        description = TAGS["DESCRIPTION"].replace("\n", "\\\n")
        output = f";FFMETADATA1\ntitle={TAGS['title']}\nDESCRIPTION={description}\n"
        return SimpleNamespace(stdout=output.encode("utf-8"))

    monkeypatch.setattr(subprocess, "run", run)
    found = scene_names.source_facts("episode.mkv")
    assert found["episode"] == 137 and found["cast"] == ["ハル", "カイ", "ソラ"]
    assert calls[0][0][0] == str(bundled)
    assert calls[0][0][-3:] == ["-f", "ffmetadata", "-"]
    assert calls[0][1]["creationflags"] == 0x08000000
    assert calls[0][1]["capture_output"] and calls[0][1]["timeout"] == 60


# --- naming revision: a newer naming policy replaces an LSS-owned name for the whole identity -----------

def _under(monkeypatch, policy):
    monkeypatch.setattr(scene_names, "POLICY_ID", policy)


def test_newer_policy_replaces_owned_name_for_whole_identity_and_keeps_history(tmp_path, monkeypatch):
    memory_data = {"labels": {"1": {"members": ["a", "b"]}}}
    _under(monkeypatch, "bgm-scene-names-v4-old")
    record(tmp_path, "a", "古い名前")
    scene_names.settle(tmp_path, memory_data)
    _under(monkeypatch, "bgm-scene-names-v5-new")
    assert scene_names.needs_renaming(scene_names.cue_record(tmp_path, "a"))
    record(tmp_path, "a", "新しい名前")
    scene_names.settle(tmp_path, memory_data)
    data = scene_names.read(tmp_path)
    recording = data["recordings"]["label:1"]
    assert recording["name"] == "新しい名前" and recording["naming_policy"] == "bgm-scene-names-v5-new"
    assert [r["name"] for r in recording["revisions"]] == ["古い名前"]
    assert scene_names.presentation(tmp_path, "a")["name"] == "新しい名前"
    assert not scene_names.needs_renaming(scene_names.cue_record(tmp_path, "a"))


def test_same_policy_settles_again_without_renaming(tmp_path, monkeypatch):
    _under(monkeypatch, "bgm-scene-names-v5-new")
    record(tmp_path, "a", "最初の名前")
    scene_names.settle(tmp_path)
    record(tmp_path, "a", "別の案")
    scene_names.settle(tmp_path)
    scene_names.settle(tmp_path)
    recording = scene_names.read(tmp_path)["recordings"]["asset:a"]
    assert recording["name"] == "最初の名前" and not recording.get("revisions")


def test_newer_policy_never_replaces_a_title_that_is_not_lss_owned(tmp_path, monkeypatch):
    _under(monkeypatch, "bgm-scene-names-v4-old")
    record(tmp_path, "a", "公式の題")
    scene_names.settle(tmp_path)
    scene_names.update(tmp_path, lambda d: d["recordings"]["asset:a"].update(basis="official"))
    _under(monkeypatch, "bgm-scene-names-v5-new")
    record(tmp_path, "a", "新しい案")
    scene_names.settle(tmp_path)
    assert scene_names.read(tmp_path)["recordings"]["asset:a"]["name"] == "公式の題"


def test_renamed_recording_does_not_share_a_name_with_another_recording_of_the_work(tmp_path, monkeypatch):
    _under(monkeypatch, "bgm-scene-names-v4-old")
    record(tmp_path, "a", "旅立ちの朝")
    record(tmp_path, "b", "夕暮れの道")
    scene_names.settle(tmp_path)
    _under(monkeypatch, "bgm-scene-names-v5-new")
    record(tmp_path, "b", "旅立ちの朝")
    scene_names.settle(tmp_path)
    names = {k: r["name"] for k, r in scene_names.read(tmp_path)["recordings"].items()}
    assert names["asset:a"] == "旅立ちの朝" and names["asset:b"] == "旅立ちの朝 その2"


def test_revised_name_reaches_the_published_file_tags(tmp_path, monkeypatch):
    path, asset = _cue(tmp_path)
    state = tmp_path / "state"
    pcm = audio_identity.audio_identity(path)["pcm_sha256"]

    def named(name):
        scene_names.record_cue(
            state, asset, pcm_sha256=pcm, source_sha256="a" * 64, facts=facts(), start_seconds=100, end_seconds=105,
            heard=heard("ミズウオだ"), order={"name": name, "use": None, "direction": "穏やかに",
                                              "category": "日常・のどか", "evidence": "ミズウオだ"},
            model={"id": "test"})
        scene_names.settle(state)

    _under(monkeypatch, "bgm-scene-names-v4-old")
    named("古い名前")
    listening.finish(path, state)
    assert FLAC(path)["TITLE"] == ["古い名前（仮題）"]
    _under(monkeypatch, "bgm-scene-names-v5-new")
    named("新しい名前")
    assert not listening.current([path], state)           # the stale title is detected, the file is refreshed
    listening.finish(path, state)
    assert FLAC(path)["TITLE"] == ["新しい名前（仮題）"]
    assert audio_identity.audio_identity(path)["pcm_sha256"] == pcm


def test_policy_rank_reads_the_version_number():
    assert scene_names.policy_rank("bgm-scene-names-v4-scene-or-context-music") == 4
    assert scene_names.policy_rank("bgm-scene-names-v12-x") == 12
    assert scene_names.policy_rank("bgm-scene-names-v5") == 5
    assert scene_names.policy_rank("legacy") == 0
    assert scene_names.policy_rank(None) == 0
    assert scene_names.policy_rank("bgm-scene-names-v5") > scene_names.policy_rank("bgm-scene-names-v4-scene-or-context-music")


def test_needs_renaming_only_for_named_cues_of_an_older_policy(monkeypatch):
    _under(monkeypatch, 'bgm-scene-names-v5-new')
    assert scene_names.needs_renaming(None) is False
    assert scene_names.needs_renaming({'name': None, 'naming_policy': 'bgm-scene-names-v4-old'}) is False
    assert scene_names.needs_renaming({'name': 'x', 'naming_policy': 'bgm-scene-names-v4-old'}) is True
    assert scene_names.needs_renaming({'name': 'x'}) is True
    assert scene_names.needs_renaming({'name': 'x', 'naming_policy': 'bgm-scene-names-v5-new'}) is False


def test_newer_policy_with_the_same_name_only_advances_the_recorded_policy(tmp_path, monkeypatch):
    _under(monkeypatch, 'bgm-scene-names-v4-old')
    record(tmp_path, 'a', '同じ名前')
    scene_names.settle(tmp_path)
    _under(monkeypatch, 'bgm-scene-names-v5-new')
    record(tmp_path, 'a', '同じ名前')
    scene_names.settle(tmp_path)
    rec = scene_names.read(tmp_path)['recordings']['asset:a']
    assert rec['name'] == '同じ名前'
    assert rec['naming_policy'] == 'bgm-scene-names-v5-new'
    assert not rec.get('revisions')


def test_an_older_policy_proposal_never_replaces_a_newer_recording_name(tmp_path, monkeypatch):
    _under(monkeypatch, 'bgm-scene-names-v5-new')
    record(tmp_path, 'a', '新しい名前')
    scene_names.settle(tmp_path)

    _under(monkeypatch, 'bgm-scene-names-v4-old')
    record(tmp_path, 'a', '古い案')
    scene_names.settle(tmp_path)

    recording = scene_names.read(tmp_path)['recordings']['asset:a']
    assert recording['name'] == '新しい名前'
    assert recording.get('revisions') in (None, [])


def test_common_loanwords_are_not_invented_names():
    # ep1 738-764: "バトルのルールを知る" fit the scene but ルール was taken for an invented name.
    assert scene_names.unheard_names(["バトルのルールを知る", "タイプ相性のレッスン"], "", []) == []
    assert scene_names.unheard_names(["ダヌーラの冒険"], "", []) == ["ダヌーラ"]
