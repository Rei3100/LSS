"""1話だけから劇伴の候補を作る経路。解析stemは合成して渡す。"""
from __future__ import annotations

import json
from pathlib import Path

from mutagen.flac import FLAC
import numpy as np
import pytest
import soundfile as sf

from lss.bgm import episode_cues as ec

RATE = 8000


def _write(path: Path, signal: np.ndarray) -> Path:
    sf.write(path, np.stack([signal, signal], axis=1), RATE, subtype="FLOAT")
    return path


def _episode(tmp_path, seconds=120, music=((20, 60),), leaks=(), speech_level=0.3):
    rng = np.random.default_rng(0)
    frames = seconds * RATE
    speech = rng.normal(0, speech_level, frames)
    music_signal = np.zeros(frames)
    for start, end in music:
        t = np.arange((end - start) * RATE) / RATE
        # 台詞の下で鳴る劇伴：台詞より小さい（取り分は実番組の既知劇伴の下位10%程度）。
        music_signal[start * RATE:end * RATE] = 0.2 * np.sin(2 * np.pi * 220 * t)
    for at in leaks:  # 分離器が散発的に音楽へ漏らす1秒
        music_signal[at * RATE:(at + 1) * RATE] = 0.3 * rng.normal(0, 1, RATE)
    effects = rng.normal(0, 0.05, frames)
    stems = {"music": _write(tmp_path / "music.wav", music_signal),
             "speech": _write(tmp_path / "speech.wav", speech),
             "effects": _write(tmp_path / "effects.wav", effects)}
    source = _write(tmp_path / "source.wav", music_signal + speech + effects)
    return source, stems


def _origin(source: Path) -> ec.CueOrigin:
    return ec.CueOrigin(source_path=source, source_sha256=ec._sha256(source), source_id="source-test",
                        analysis_id="bgm-sfc-analysis-v1-test", runtime_id="runtime-test")


def _recipe(**overrides):
    values = dict(window_seconds=1.0, smoothing_windows=5, onset_share=0.15, offset_share=0.08,
                  floor_music_db=-60.0, bridge_seconds=3.0, minimum_cue_seconds=8.0,
                  minimum_active_fraction=0.5, exclusion_pad_seconds=1.0)
    values.update(overrides)
    return ec.EpisodeCueRecipe(**values)


def test_music_under_dialogue_becomes_a_held_cue(tmp_path):
    _, stems = _episode(tmp_path)
    plan = ec.plan_episode_cues(stems, _recipe())
    held = [c for c in plan["cues"] if c["disposition"] == ec.DISPOSITION_HELD]
    assert len(held) == 1
    assert abs(held[0]["start_seconds"] - 20) <= 3 and abs(held[0]["end_seconds"] - 60) <= 3
    # 台詞のほうが大きくても落とさない。
    assert held[0]["music_over_speech_db"] < 0


def test_sporadic_leaks_do_not_become_cues(tmp_path):
    _, stems = _episode(tmp_path, music=(), leaks=(10, 30, 31, 70, 95))
    plan = ec.plan_episode_cues(stems, _recipe())
    assert [c for c in plan["cues"] if c["disposition"] == ec.DISPOSITION_HELD] == []


def test_known_recordings_are_excluded_before_candidates(tmp_path):
    _, stems = _episode(tmp_path, music=((20, 60), (80, 110)))
    plan = ec.plan_episode_cues(stems, _recipe(), known_spans=[(80.0, 110.0)])
    held = [(c["start_seconds"], c["end_seconds"]) for c in plan["cues"]
            if c["disposition"] == ec.DISPOSITION_HELD]
    assert len(held) == 1 and held[0][1] <= 79.0
    assert plan["excluded_known_seconds"] > 25


def test_short_fragments_are_not_cues(tmp_path):
    _, stems = _episode(tmp_path, music=((20, 25),))
    assert ec.plan_episode_cues(stems, _recipe())["cues"] == []


def test_a_run_writes_held_audio_with_provenance(tmp_path):
    source, stems = _episode(tmp_path)
    record = ec.run_episode_cues(source, stems, tmp_path / "out", _recipe(), origin=_origin(source))
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETED_HELD_CANDIDATES" and payload["held_count"] == 1
    assert payload["publication_state"] == "not_published" and payload["promotion_allowed"] is False
    assert payload["references"]["known_recording_exclusion"] == "not_available"
    cue = payload["plan"]["cues"][0]
    for name in ec.RENDERERS:
        artifact = cue["artifacts"][name]
        info = sf.info(artifact["path"])
        assert info.channels == 2 and info.samplerate == RATE
        assert abs(info.frames / RATE - cue["seconds"]) < 0.01
        # 単独のファイルにも出典・時刻・未公開状態が残る。
        provenance = json.loads(FLAC(artifact["path"])["LSS_CANDIDATE_PROVENANCE"][0])
        assert provenance["source_path"] == str(source.resolve())
        assert provenance["publication_state"] == "not_published"
        assert abs(provenance["start_seconds"] - cue["start_seconds"]) < 1 / RATE
    assert payload["source"]["sha256"] == ec._sha256(source)
    assert payload["original"]["sha256"] == ec._sha256(source)

    # 同じ候補の再実行は、保存済みの候補を検証して再利用する。
    again = json.loads(ec.run_episode_cues(source, stems, tmp_path / "out", _recipe(),
                                           origin=_origin(source)).read_text(encoding="utf-8"))
    assert again["plan"]["cues"][0]["artifacts"] == cue["artifacts"]


def test_stems_on_another_timeline_are_refused(tmp_path):
    source, stems = _episode(tmp_path)
    _write(stems["music"], np.zeros(50 * RATE))
    with pytest.raises(ec.EpisodeCueError):
        ec.run_episode_cues(source, stems, tmp_path / "out", _recipe(), origin=_origin(source))


def test_the_recipe_identifier_follows_every_setting():
    assert _recipe().recipe_id != _recipe(onset_share=0.2).recipe_id
    assert ec.default_recipe().recipe_id.startswith("bgm-episode-cues-v1-")
    with pytest.raises(ec.EpisodeCueError):
        _recipe(offset_share=0.3).validate()


def test_the_raw_candidate_is_cut_from_the_native_rate_decode(tmp_path):
    source, stems = _episode(tmp_path)
    native_rate = RATE * 2
    data, _ = sf.read(source, always_2d=True)
    native = tmp_path / "native.wav"
    sf.write(native, np.repeat(data, 2, axis=0), native_rate, subtype="FLOAT")
    record = ec.run_episode_cues(source, stems, tmp_path / "out", _recipe(), raw_source=native,
                                 origin=_origin(source), provenance={"resample": "test"})
    payload = json.loads(record.read_text(encoding="utf-8"))
    cue = payload["plan"]["cues"][0]
    raw = sf.info(cue["artifacts"]["raw"]["path"])
    assert raw.samplerate == native_rate
    assert abs(raw.frames / native_rate - cue["seconds"]) < 0.01
    assert payload["input_preparation"] == {"resample": "test"}


def test_two_cues_that_follow_each_other_are_split(tmp_path):
    from lss.bgm import cue_boundaries

    rate = 16000
    t = np.arange(40 * rate) / rate
    first = 0.3 * np.sin(2 * np.pi * 220 * t)
    second = 0.1 * np.sign(np.sin(2 * np.pi * 1300 * t))
    left = np.concatenate([first, second])
    right = np.concatenate([first, -second])  # 二曲目は左右の混ぜ方も違う
    music = tmp_path / "music.wav"
    sf.write(music, np.stack([left, right], axis=1), rate, subtype="FLOAT")
    found = cue_boundaries.split_positions(
        music, 0.0, 80.0, cue_boundaries.CueBoundaryRecipe(40.0, 20.0, 1.5))
    assert [round(item["seconds"]) for item in found] in ([39], [40], [41])


def test_short_intervals_are_not_split(tmp_path):
    from lss.bgm import cue_boundaries

    music = _write(tmp_path / "music.wav", np.zeros(30 * RATE))
    assert cue_boundaries.split_positions(
        music, 0.0, 30.0, cue_boundaries.CueBoundaryRecipe(40.0, 20.0, 1.5)) == []


def _sustained(**overrides):
    return _recipe(sustained_seconds=8.0, **overrides)


def test_quiet_music_under_louder_dialogue_is_held_by_sustained_presence(tmp_path):
    """台詞より20dB小さい劇伴は取り分では優勢にならないが、途切れず続くので候補になる。"""
    rng = np.random.default_rng(1)
    frames = 60 * RATE
    t = np.arange(30 * RATE) / RATE
    music = np.zeros(frames)
    music[10 * RATE:40 * RATE] = 0.02 * np.sin(2 * np.pi * 330 * t)
    speech = rng.normal(0, 0.3, frames)
    stems = {"music": _write(tmp_path / "music.wav", music),
             "speech": _write(tmp_path / "speech.wav", speech),
             "effects": _write(tmp_path / "effects.wav", np.zeros(frames))}

    before = [c for c in ec.plan_episode_cues(stems, _recipe())["cues"]
              if c["disposition"] == ec.DISPOSITION_HELD]
    after = [c for c in ec.plan_episode_cues(stems, _sustained())["cues"]
             if c["disposition"] == ec.DISPOSITION_HELD]

    assert before == []
    [cue] = after
    assert cue["start_seconds"] == 10.0 and cue["end_seconds"] == 40.0
    assert cue["sustained_fraction"] == 1.0 and cue["share_fraction"] < 0.5
    assert cue["music_over_speech_db"] < -15


def test_effects_leaking_into_music_between_lines_do_not_become_sustained(tmp_path):
    """台詞のある場面で分離器が効果音を音楽へ漏らしても、鳴る窓だけなので途切れない存在にならない。"""
    rng = np.random.default_rng(2)
    frames = 60 * RATE
    music = np.zeros(frames)
    for second in range(10, 50, 2):  # 1秒おきの効果音の漏れ。間の窓は無音
        music[second * RATE:(second + 1) * RATE] = rng.normal(0, 0.05, RATE)
    stems = {"music": _write(tmp_path / "music.wav", music),
             "speech": _write(tmp_path / "speech.wav", rng.normal(0, 0.3, frames)),
             "effects": _write(tmp_path / "effects.wav", np.zeros(frames))}

    before = ec.plan_episode_cues(stems, _recipe())["cues"]
    after = ec.plan_episode_cues(stems, _sustained())["cues"]

    assert not ec.sustained_windows(ec.window_shares(stems, 1.0)[1], _sustained()).any()
    assert [c for c in after if c["disposition"] == ec.DISPOSITION_HELD] == \
        [c for c in before if c["disposition"] == ec.DISPOSITION_HELD] == []


def test_a_quiet_dip_inside_a_cue_does_not_cut_it(tmp_path):
    """下限を1秒だけ割る静かな箇所があっても、前後の途切れない存在はつながって1つの区間になる。"""
    frames = 60 * RATE
    t = np.arange(30 * RATE) / RATE
    music = np.zeros(frames)
    music[10 * RATE:40 * RATE] = 0.02 * np.sin(2 * np.pi * 330 * t)
    music[26 * RATE:27 * RATE] = 0.0
    stems = {"music": _write(tmp_path / "music.wav", music),
             "speech": _write(tmp_path / "speech.wav", np.random.default_rng(3).normal(0, 0.3, frames)),
             "effects": _write(tmp_path / "effects.wav", np.zeros(frames))}

    plan = ec.plan_episode_cues(stems, _sustained())

    assert plan["activity_interval_count"] == 1
    held = [(c["start_seconds"], c["end_seconds"]) for c in plan["cues"] if c["disposition"] == ec.DISPOSITION_HELD]
    assert held and held[0][0] == 10.0 and held[-1][1] == 40.0
    assert sum(end - start for start, end in held) == 30.0


def test_the_sustained_declaration_changes_the_identifier_only_when_used():
    assert _recipe().recipe_id == _recipe(sustained_seconds=None).recipe_id
    assert _recipe().recipe_id != _sustained().recipe_id
    assert "sustained_seconds" not in _recipe().to_dict()
    with pytest.raises(ec.EpisodeCueError):
        _recipe(sustained_seconds=0.5).validate()


def test_two_loud_cues_joined_by_quiet_music_stay_two_candidates(tmp_path):
    """途切れない存在が取り分の区間どうしをつないでも、隙間の最も小さい窓で切って別の候補に残す。"""
    rng = np.random.default_rng(4)
    frames = 90 * RATE
    t = np.arange(frames) / RATE
    music = 0.3 * np.sin(2 * np.pi * 220 * t)
    music[40 * RATE:50 * RATE] *= 0.05  # 台詞の下で小さく続く
    music[44 * RATE:45 * RATE] *= 0.3   # 隙間の中で最も小さい窓
    speech = np.zeros(frames)
    speech[40 * RATE:50 * RATE] = rng.normal(0, 0.3, 10 * RATE)
    stems = {"music": _write(tmp_path / "music.wav", music),
             "speech": _write(tmp_path / "speech.wav", speech),
             "effects": _write(tmp_path / "effects.wav", np.zeros(frames))}
    recipe = _sustained()

    shares, levels, _ = ec.window_shares(stems, 1.0)
    assert len(ec.share_intervals(shares, levels, recipe)) == 2
    assert ec.active_intervals(shares, levels, recipe) == [(0, 90)]
    assert ec.joined_gap_splits(shares, levels, recipe) == [44]
    held = [(c["start_seconds"], c["end_seconds"]) for c in ec.plan_episode_cues(stems, recipe)["cues"]
            if c["disposition"] == ec.DISPOSITION_HELD]
    assert held == [(0.0, 44.0), (44.0, 90.0)]


def test_the_default_recipe_keeps_quiet_music_under_dialogue():
    assert ec.default_recipe().sustained_seconds == 8.0
    assert ec.default_recipe().minimum_cue_seconds == 8.0


def _short(**overrides):
    return _sustained(**{"short_seconds": 3.0, **overrides})


def _stems_from(tmp_path, music, speech=None, effects=None):
    frames = music.size
    return {"music": _write(tmp_path / "music.wav", music),
            "speech": _write(tmp_path / "speech.wav", np.zeros(frames) if speech is None else speech),
            "effects": _write(tmp_path / "effects.wav", np.zeros(frames) if effects is None else effects)}


def _tone(seconds, level=0.2, hz=330):
    return level * np.sin(2 * np.pi * hz * np.arange(int(seconds * RATE)) / RATE)


def _held(plan):
    return [(c["start_seconds"], c["end_seconds"], c.get("short_cue")) for c in plan["cues"]
            if c["disposition"] == ec.DISPOSITION_HELD]


def test_a_complete_short_cue_in_silence_is_held(tmp_path):
    music = np.zeros(40 * RATE)
    music[10 * RATE:14 * RATE] = _tone(4)
    stems = _stems_from(tmp_path, music)

    assert _held(ec.plan_episode_cues(stems, _sustained())) == []
    assert _held(ec.plan_episode_cues(stems, _short())) == [(10.0, 14.0, True)]


def test_a_quiet_short_cue_under_dialogue_is_held(tmp_path):
    """台詞より20dB小さい5秒の曲。取り分は低いが、区間の全窓で途切れず存在する。"""
    rng = np.random.default_rng(5)
    music = np.zeros(40 * RATE)
    music[12 * RATE:17 * RATE] = _tone(5, level=0.02)
    speech = rng.normal(0, 0.3, 40 * RATE)
    stems = _stems_from(tmp_path, music, speech)

    assert _held(ec.plan_episode_cues(stems, _sustained())) == []
    assert _held(ec.plan_episode_cues(stems, _short())) == [(12.0, 17.0, True)]


def test_a_two_second_cue_needs_music_to_be_nearly_alone(tmp_path):
    music = np.zeros(40 * RATE)
    music[10 * RATE:12 * RATE] = _tone(2)
    stems = _stems_from(tmp_path, music)

    assert _held(ec.plan_episode_cues(stems, _short())) == []
    assert _held(ec.plan_episode_cues(stems, _short(short_dominant_share=0.9))) == [(10.0, 12.0, True)]


def test_effects_leaking_between_lines_do_not_become_short_cues(tmp_path):
    """台詞のある場面で効果音が音楽へ1窓ずつ漏れても、途切れるので短い音楽にならない。"""
    rng = np.random.default_rng(6)
    music = np.zeros(60 * RATE)
    for second in range(10, 50, 2):
        music[second * RATE:(second + 1) * RATE] = rng.normal(0, 0.3, RATE)
    stems = _stems_from(tmp_path, music, rng.normal(0, 0.1, 60 * RATE))

    shares, levels, _ = ec.window_shares(stems, 1.0)
    runs, marked = ec.short_intervals(shares, levels, [], _short(short_dominant_share=0.9))
    assert runs == [] and not marked.any()
    before = [(a, b) for a, b, _ in _held(ec.plan_episode_cues(stems, _sustained()))]
    after = _held(ec.plan_episode_cues(stems, _short(short_dominant_share=0.9)))
    assert [(a, b) for a, b, _ in after] == before and not any(short for _, _, short in after)


def test_a_short_remainder_of_a_known_recording_is_not_a_short_cue(tmp_path):
    """既知の録音を外した残りの短い断片は、長い曲の一部であって完結した短い音楽ではない。"""
    music = np.zeros(60 * RATE)
    music[10 * RATE:40 * RATE] = _tone(30)
    stems = _stems_from(tmp_path, music)

    plan = ec.plan_episode_cues(stems, _short(short_dominant_share=0.9), known_spans=[(15.0, 40.0)])

    assert _held(plan) == []


def test_short_presence_next_to_a_long_cue_does_not_move_its_edges(tmp_path):
    music = np.zeros(60 * RATE)
    music[10 * RATE:30 * RATE] = _tone(20)
    music[32 * RATE:36 * RATE] = _tone(4, level=0.02)  # 長い曲の直後（ブリッジ内）の小さな音
    rng = np.random.default_rng(7)
    speech = np.zeros(60 * RATE)
    speech[30 * RATE:40 * RATE] = rng.normal(0, 0.3, 10 * RATE)
    stems = _stems_from(tmp_path, music, speech)

    before = _held(ec.plan_episode_cues(stems, _sustained()))
    after = _held(ec.plan_episode_cues(stems, _short(short_dominant_share=0.9)))

    assert [(a, b) for a, b, _ in after] == [(a, b) for a, b, _ in before]


def test_the_short_declaration_changes_the_identifier_only_when_used():
    assert "short_seconds" not in _sustained().to_dict()
    assert _sustained().recipe_id != _short().recipe_id != _short(short_dominant_share=0.9).recipe_id
    with pytest.raises(ec.EpisodeCueError):
        _short(short_seconds=9.0).validate()
    with pytest.raises(ec.EpisodeCueError):
        _sustained(short_dominant_share=0.9).validate()
