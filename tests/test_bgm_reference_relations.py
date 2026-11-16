"""区間と参照録音の関係を、合成の録音で反証できる形に確かめる（v3：音楽の説明の会計・成分の欠け・領域の地図）。

- 一つの録音の連続した抜粋（台詞が重なっても）だけをEXACTにし、範囲の外の別の音楽は分けて残す。
- 参照が鳴っていても、登録していない音楽が重なる区間（重ね）・短い差し替え・切り貼りはEXACTにしない（消さない）。
- 曲の中の繰り返し（ループ）を編集と取り違えない。数秒だけの一致で区間全体の関係を言わない。
- Music stemが無い時はEXACTを出さない。同じ材料からは同じ結果。名前（タグ）は判定に使わない。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lss.bgm import cue_semantics, identity, reference_relations as rr
from tests.test_bgm_identity import _dialogue_and_noise, _music
from tests.release_receipts import distribution_receipt

RATE = identity.SAMPLE_RATE


def _library(tmp_path, signals: dict[str, np.ndarray]) -> rr.ReferenceLibrary:
    paths = []
    for name in signals:
        path = tmp_path / f"{name}.flac"
        path.write_bytes(name.encode())
        paths.append(path)
    library=rr.ReferenceLibrary.build(paths, loader=lambda p: signals[Path(p).stem])
    for reference in library.index.references:
        library.metadata[reference.sha256]['independent_distribution']=distribution_receipt(reference.sha256)
    return library


class Episode:
    """合成の番組：混合と、分離器の出力に見立てたMusic stem・台詞stem（台詞の一部がMusicへ漏れる）。"""

    def __init__(self, seconds: float, pieces: list[tuple[float, np.ndarray]], *, dialogue_gain: float = 0.0,
                 leak: float = 0.05, seed: int = 99):
        floor = np.random.default_rng(seed).normal(0, 1e-4, int(seconds * RATE)).astype(np.float32)
        self.music = floor.copy()
        for at, clip in pieces:
            self.music[int(at * RATE):int(at * RATE) + clip.size] += clip
        dialogue = (dialogue_gain * _dialogue_and_noise(seconds, seed=seed)).astype(np.float32)
        self.mixture = self.music + dialogue
        self.music = self.music + leak * dialogue
        self.dialogue = (1.0 - leak) * dialogue


def _segment(start: float, end: float) -> dict:
    return {"start_seconds": start, "end_seconds": end, "core_seconds": [start, end],
            "start_edge": {"kind": cue_semantics.EDGE_SILENCE}, "end_edge": {"kind": cue_semantics.EDGE_SILENCE},
            "stretch": [start, end], "candidates": [[start, end]]}


def _relate(tmp_path, signals, episode: Episode, segments, *, stems: bool = True, **kwargs):
    library = _library(tmp_path, signals)
    matches = identity.match(episode.mixture, library.index, collapse=False)
    return rr.relate_segments(segments, query=episode.mixture, library=library, matches=matches,
                              music=episode.music if stems else None,
                              interference=[episode.dialogue] if stems else None, **kwargs)


@pytest.fixture
def signals():
    return {"A": _music(1, 80.0), "B": _music(2, 60.0)}


def _states(pieces):
    return [(p["start_seconds"], p["end_seconds"], p["reference_relation"]["state"]) for p in pieces]


def _at(clip: np.ndarray, level: float) -> np.ndarray:
    return (clip * (level / (np.sqrt(np.mean(clip.astype(np.float64) ** 2)) + 1e-12))).astype(np.float32)


def test_a_contiguous_excerpt_under_dialogue_is_the_exact_released_recording(tmp_path, signals):
    episode = Episode(60.0, [(10.0, signals["A"][int(12 * RATE):int(42 * RATE)])], dialogue_gain=0.3)
    [piece] = _relate(tmp_path, signals, episode, [_segment(10, 40)])
    relation = piece["reference_relation"]
    assert relation["state"] == rr.EXACT and relation["basis"] == "one_alignment_accounts_for_all_music"
    assert relation["reference"]["offset_seconds"] == pytest.approx(2.0, abs=0.05)
    assert relation["reference_range_seconds"] == pytest.approx([12.0, 42.0], abs=0.05)
    assert relation["coverage"]["median_novelty"] <= rr.default_recipe().exact_max_novelty
    assert [r["kind"] for r in relation["regions"]] == [rr.CELL_SUPPORTED]
    assert (piece["start_seconds"], piece["end_seconds"]) == (10.0, 40.0)


def test_fractional_boundary_preserves_complete_source_span(tmp_path, signals):
    episode = Episode(60.0, [(10.0, signals["A"][int(12*RATE):int(43*RATE)])])
    parts = _relate(tmp_path, signals, episode, [_segment(10.27, 40.63)])
    assert parts[0]["start_seconds"] == 10.27
    assert parts[-1]["end_seconds"] == 40.63
    assert all(a["end_seconds"] == b["start_seconds"] for a,b in zip(parts,parts[1:]))
    assert sum(p["end_seconds"]-p["start_seconds"] for p in parts) == pytest.approx(30.36)
    assert all(p["reference_relation"]["measured_scope_seconds"] == [10,41] for p in parts)


def test_subsecond_unknown_span_cannot_disappear_by_rounding(tmp_path, signals):
    episode = Episode(15., [(10., _music(77, 1.))])
    parts = _relate(tmp_path, signals, episode, [_segment(10.51, 10.63)])
    assert _states(parts) == [(10.51,10.63,rr.UNKNOWN)]


def test_unknown_music_after_a_released_recording_is_split_off_and_kept(tmp_path, signals):
    episode = Episode(60.0, [(10.0, signals["A"][:int(20 * RATE)]), (30.0, _music(77, 20.0))])
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 50)])
    assert _states(pieces) == [(10.0, 30.0, rr.EXACT), (30.0, 50.0, rr.UNKNOWN)]
    # 分けた境目は参照の波形で裏付けた境目。外側の端は元のまま。
    assert pieces[0]["end_edge"]["kind"] == rr.EDGE_REFERENCE == pieces[1]["start_edge"]["kind"]
    assert pieces[0]["start_edge"]["kind"] == cue_semantics.EDGE_SILENCE
    assert pieces[1]["end_edge"]["kind"] == cue_semantics.EDGE_SILENCE
    assert pieces[0]["reference_relation"]["starts_at_reference_start"] is True
    assert all(p["split_from"] == [10.0, 50.0] for p in pieces)


@pytest.mark.parametrize("level", [1.0, 0.5])
@pytest.mark.parametrize("dialogue_gain", [0.0, 0.3])
def test_unregistered_music_over_a_released_recording_is_never_exact(tmp_path, signals, level, dialogue_gain):
    """独立反証 RA-P1-002：参照の存在は、区間の音楽が参照だけでできている証拠ではない。"""
    registered = _at(signals["A"][int(20 * RATE):int(52 * RATE)], 0.1)
    other = _at(_music(55, 32.0), 0.1 * level)
    episode = Episode(60.0, [(10.0, registered + other)], dialogue_gain=dialogue_gain)
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 42)])
    assert _states(pieces) == [(10.0, 42.0, rr.ADDITIONAL)]
    assert pieces[0]["reference_relation"]["basis"] == "reference_with_material_it_does_not_explain"
    applied = rr.apply([{**pieces[0], "semantic_class": cue_semantics.WANTED, "reasons": []}],
                       cue_semantics.KNOWN_RELEASED)
    assert applied[0]["semantic_class"] == cue_semantics.WANTED


def test_a_short_replacement_inside_one_recording_is_an_edit_with_its_region_found(tmp_path, signals):
    """独立反証 RA-P1-001：内側の2秒を同じ録音の別の位置へ差し替えた番組の版。"""
    clip = signals["A"][int(20 * RATE):int(52 * RATE)].copy()
    clip[int(14 * RATE):int(16 * RATE)] = signals["A"][int(60 * RATE):int(62 * RATE)]
    episode = Episode(60.0, [(10.0, clip)], dialogue_gain=0.2)
    [piece] = _relate(tmp_path, signals, episode, [_segment(10, 42)])
    relation = piece["reference_relation"]
    assert relation["state"] == rr.EDIT and relation["basis"] == "multiple_alignments_of_one_recording"
    replaced = [r for r in relation["regions"] if r.get("offset_seconds") is not None
                and abs(r["offset_seconds"] - (60.0 - 24.0)) < 0.1]
    assert replaced and replaced[0]["origin"] in ("local_search", "fingerprint")
    assert all(23 <= r["start_seconds"] and r["end_seconds"] <= 27 for r in replaced)


def test_an_episode_specific_edit_of_a_released_recording_is_not_exact(tmp_path, signals):
    edit = np.concatenate([signals["A"][:int(14 * RATE)], signals["A"][int(36 * RATE):int(54 * RATE)]])
    episode = Episode(60.0, [(10.0, edit)])
    [piece] = _relate(tmp_path, signals, episode, [_segment(10, 42)])
    relation = piece["reference_relation"]
    assert relation["state"] == rr.EDIT and relation["basis"] == "multiple_alignments_of_one_recording"
    supported = {round(r["offset_seconds"]) for r in relation["regions"] if r["kind"] == rr.CELL_SUPPORTED}
    assert supported == {-10, 12}


def test_a_loop_inside_one_recording_is_not_mistaken_for_an_edit(tmp_path):
    bar = _music(5, 4.0)
    looped = np.tile(bar, 15)  # 4秒の繰り返しだけでできた曲
    signals = {"L": looped, "B": _music(2, 60.0)}
    episode = Episode(60.0, [(10.0, looped[int(6 * RATE):int(36 * RATE)])])
    [piece] = _relate(tmp_path, signals, episode, [_segment(10, 40)])
    assert piece["reference_relation"]["state"] == rr.EXACT


@pytest.mark.parametrize("variant", [{"transpose": 2.0}, {"tempo": 1.06}, {"performance": 5}])
def test_another_performance_or_arrangement_is_never_exact(tmp_path, signals, variant):
    episode = Episode(60.0, [(10.0, _music(1, 30.0, **variant))], dialogue_gain=0.05)
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 40)])
    assert rr.EXACT not in {p["reference_relation"]["state"] for p in pieces}
    assert [(p["start_seconds"], p["end_seconds"]) for p in pieces] == [(10.0, 40.0)]


def test_an_unrelated_cue_stays_unknown_and_unsplit(tmp_path, signals):
    episode = Episode(60.0, [(10.0, _music(42, 30.0))], dialogue_gain=0.2)
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 40)])
    assert _states(pieces) == [(10.0, 40.0, rr.UNKNOWN)]


def test_a_few_seconds_of_a_recording_do_not_make_the_whole_cue_its_version(tmp_path, signals):
    """区間40秒のうち4秒だけ参照の録音が入っても、区間を録音の版（EDIT・EXACT）とは言わない。"""
    clip = _music(42, 40.0)
    clip[int(18 * RATE):int(22 * RATE)] = signals["A"][int(30 * RATE):int(34 * RATE)]
    episode = Episode(60.0, [(10.0, clip)])
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 50)])
    assert {p["reference_relation"]["state"] for p in pieces} <= {rr.UNKNOWN, rr.RELATED}


def test_without_a_music_stem_nothing_is_called_exact(tmp_path, signals):
    episode = Episode(60.0, [(10.0, signals["A"][int(12 * RATE):int(42 * RATE)])])
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 40)], stems=False)
    assert _states(pieces) == [(10.0, 40.0, rr.UNKNOWN)]


def test_shared_harmonic_material_marks_a_related_version_candidate(tmp_path):
    # 和声の外れ具合は集まりの中で測るので、集まりを判定に必要な数にする。
    signals = {f"R{n:02d}": _music(100 + n, 45.0) for n in range(22)}
    variant = _music(105, 45.0, transpose=2.0, harmonics=(1.0, 0.8, 0.6, 0.3))
    episode = Episode(60.0, [(5.0, variant[:int(40 * RATE)])])
    library = _library(tmp_path, signals)
    matches = identity.match(episode.mixture, library.index, collapse=False)
    [piece] = rr.relate_segments([_segment(5, 45)], query=episode.mixture, library=library, matches=matches,
                                 music=episode.music, interference=[episode.dialogue])
    relation = piece["reference_relation"]
    assert relation["state"] == rr.RELATED
    assert relation["candidates"][0]["reference_sha256"] == library.index.references[5].sha256


def test_the_same_evidence_gives_the_same_decision(tmp_path, signals):
    episode = Episode(60.0, [(10.0, signals["A"][:int(20 * RATE)]), (30.0, _music(77, 20.0))], dialogue_gain=0.2)
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(), second.mkdir()
    once = _relate(first, signals, episode, [_segment(10, 50)])
    again = _relate(second, signals, episode, [_segment(10, 50)])
    assert json.dumps(once, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_reference_names_are_recorded_but_never_decide(tmp_path, signals, monkeypatch):
    episode = Episode(60.0, [(10.0, signals["A"][int(12 * RATE):int(42 * RATE)])])
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(), second.mkdir()
    plain = _relate(first, signals, episode, [_segment(10, 40)])
    monkeypatch.setattr(rr, "reference_metadata",
                        lambda path: {"tags": {"title": "別の名前"}, "display_title": "別の名前",
                                      "title_source": "reference_file_tags", "file_name": Path(path).name})
    named = _relate(second, signals, episode, [_segment(10, 40)])
    assert _states(plain) == _states(named)
    assert plain[0]["reference_relation"]["reference"]["display_title"] is None
    assert named[0]["reference_relation"]["reference"]["display_title"] == "別の名前"


def test_apply_changes_only_exact_pieces_and_keeps_the_acoustic_judgement():
    decided = [
        {"start_seconds": 0.0, "end_seconds": 10.0, "semantic_class": cue_semantics.WANTED, "reasons": [],
         "reference_relation": {"state": rr.EXACT,"reference":{"reference_sha256":"a"*64,
                                "independent_distribution":distribution_receipt("a"*64)}}},
        {"start_seconds": 10.0, "end_seconds": 20.0, "semantic_class": cue_semantics.WANTED, "reasons": [],
         "reference_relation": {"state": rr.EDIT}},
        {"start_seconds": 20.0, "end_seconds": 30.0, "semantic_class": cue_semantics.WANTED, "reasons": [],
         "reference_relation": {"state": rr.ADDITIONAL}},
        {"start_seconds": 30.0, "end_seconds": 40.0, "semantic_class": cue_semantics.UNCERTAIN,
         "reasons": ["possible_change_inside"], "reference_relation": {"state": rr.UNKNOWN}},
    ]
    applied = rr.apply(decided, cue_semantics.KNOWN_RELEASED)
    assert [item["semantic_class"] for item in applied] == [cue_semantics.KNOWN_RELEASED, cue_semantics.WANTED,
                                                           cue_semantics.WANTED, cue_semantics.UNCERTAIN]
    assert applied[0]["acoustic_class"] == cue_semantics.WANTED and applied[0]["reasons"] == [rr.REASON_EXACT]
    assert decided[0]["semantic_class"] == cue_semantics.WANTED  # 入力は変えない


def test_occurrences_merge_pieces_of_one_continuous_use():
    def exact(start, end, offset):
        return {"start_seconds": start, "end_seconds": end, "reference_relation": {
            "state": rr.EXACT, "reference": {"reference_sha256": "a" * 64, "offset_seconds": offset,
                                             "display_title": "曲", "title_source": "reference_file_tags",
                                             "independent_distribution":distribution_receipt("a"*64)},
            "reference_range_seconds": [start + offset, end + offset], "starts_at_reference_start": False,
            "ends_at_reference_end": False}}
    found = rr.occurrences([exact(10, 20, 5.0), exact(20, 30, 5.0), exact(40, 50, -30.0)], time_offset=0.5)
    assert [(o["source_start_seconds"], o["source_end_seconds"]) for o in found] == [(10.5, 30.5), (40.5, 50.5)]
    assert found[0]["reference_range_seconds"] == [15.0, 35.0]
    assert found[0]["extraction"] == "not_created_independent_recording_available"


def test_related_occurrences_keep_the_region_map_in_source_time():
    decided = [{"start_seconds": 10.0, "end_seconds": 40.0, "reference_relation": {
        "state": rr.EDIT, "basis": "multiple_alignments_of_one_recording",
        "reference": {"reference_sha256": "b" * 64, "display_title": "曲"}, "donor_reconstructable": True,
        "measured_scope_seconds": [8.0, 42.0],
        "regions": [{"start_seconds": 10.0, "end_seconds": 24.0, "kind": rr.CELL_SUPPORTED,
                     "reference_sha256": "b" * 64, "offset_seconds": -10.0}]}}]
    [found] = rr.related_occurrences(decided, time_offset=2.0)
    assert found["relation"] == rr.EDIT and found["donor_reconstructable"] is True
    assert found["regions"][0]["source_start_seconds"] == 12.0
    assert found["measured_scope_seconds"] == [8.0, 42.0]
    assert rr.compact(decided[0]["reference_relation"])["regions"][0]["offset_seconds"] == -10.0
    assert rr.compact(decided[0]["reference_relation"])["measured_scope_seconds"] == [8.0, 42.0]


def test_restricting_exact_geometry_keeps_exact_state_and_applies_existing_edge_rules():
    relation = {"state": rr.EXACT, "reference": {"offset_seconds": -5., "duration_seconds": 30.},
                "reference_range_seconds": [5., 25.], "starts_at_reference_start": False,
                "ends_at_reference_end": False,
                "regions": [{"start_seconds": 10., "end_seconds": 30., "offset_seconds": -5.,
                             "reference_range_seconds": [5., 25.]}]}
    restricted = rr.restrict_relation_span(relation, 12., 28., parent_span=[10., 30.])
    assert restricted["state"] == rr.EXACT
    assert restricted["reference_range_seconds"] == [7., 23.]
    assert restricted["starts_at_reference_start"] is False
    assert restricted["ends_at_reference_end"] is False
    assert restricted["regions"] == [{"start_seconds": 12., "end_seconds": 28., "offset_seconds": -5.,
                                       "reference_range_seconds": [7., 23.]}]
    assert relation["reference_range_seconds"] == [5., 25.]


def test_recipe_identity_changes_with_any_threshold():
    base = rr.default_recipe()
    from dataclasses import replace
    assert replace(base, exact_max_novelty=0.006).recipe_id != base.recipe_id
    with pytest.raises(rr.ReferenceRelationsError):
        replace(base, exact_max_novelty=0.5).validate()
    with pytest.raises(rr.ReferenceRelationsError):
        replace(base, present_joint=0.6).validate()


# --- v3：成分の違う版（一つの層を抜いた・加えた版）とマスタリングの違い ----------------------

def _two_layers(seconds: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    """旋律の層と、低い音域で別の音高・拍で動く層。既発売の録音は二つの層を混ぜたもの。"""
    return _music(1, seconds), _music(3, seconds, harmonics=(1.0,), transpose=-12.0)


def test_a_version_with_one_layer_removed_is_a_component_variant_never_exact(tmp_path):
    """既発売の録音から一つの層を抜いた番組だけの版（ドラムレス・ギターレス等の代理）。録音の存在だけで既発売にしない。"""
    lead, layer = _two_layers()
    signals = {"FULL": (lead + 0.7 * layer).astype(np.float32), "B": _music(2, 60.0)}
    episode = Episode(60.0, [(10.0, lead[int(12 * RATE):int(42 * RATE)])], dialogue_gain=0.2)
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 40)])
    assert rr.EXACT not in {p["reference_relation"]["state"] for p in pieces}
    [piece] = pieces
    relation = piece["reference_relation"]
    assert relation["state"] == rr.COMPONENT
    assert relation["basis"] in ("reference_components_absent_in_episode",
                                 "reference_components_absent_and_material_added")
    assert relation["components"]["missing"]["energy_share"] > 0.02
    assert relation["coverage"]["median_absence_music"] > rr.default_recipe().exact_max_absence
    assert rr.identity_state(relation, references_registered=True) == "released_recording_component_variant"
    applied = rr.apply([{**piece, "semantic_class": cue_semantics.WANTED, "reasons": []}], cue_semantics.KNOWN_RELEASED)
    assert applied[0]["semantic_class"] == cue_semantics.WANTED
    [related] = rr.related_occurrences(pieces)
    assert related["relation"] == rr.COMPONENT and related["components"]["missing"]


def test_the_released_full_mix_over_an_episode_with_an_added_layer_stays_additional(tmp_path):
    """層を加えた版（番組の音＝既発売＋別の層）は、従来どおり追加の素材がある版（欠けではない）。"""
    lead, layer = _two_layers()
    signals = {"LEAD": lead, "B": _music(2, 60.0)}
    episode = Episode(60.0, [(10.0, (lead + 0.7 * layer)[int(12 * RATE):int(42 * RATE)])], dialogue_gain=0.2)
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 40)])
    assert rr.EXACT not in {p["reference_relation"]["state"] for p in pieces}
    assert pieces[0]["reference_relation"]["state"] in (rr.ADDITIONAL, rr.COMPONENT)
    assert pieces[0]["reference_relation"]["components"]["added"]["energy_share"] > 0.05


def test_mastering_and_eq_differences_of_the_same_recording_stay_exact(tmp_path, signals):
    """放送・配信のEQ（低域を下げる・高域を少し下げる）は成分の欠けではない。"""
    from scipy.signal import butter, sosfilt

    clip = signals["A"][int(12 * RATE):int(42 * RATE)].astype(np.float64)
    low = sosfilt(butter(2, 250, "low", fs=RATE, output="sos"), clip)
    high = sosfilt(butter(2, 3500, "high", fs=RATE, output="sos"), clip)
    mastered = (clip - 0.6 * low - 0.3 * high).astype(np.float32)
    episode = Episode(60.0, [(10.0, mastered)], dialogue_gain=0.2)
    [piece] = _relate(tmp_path, signals, episode, [_segment(10, 40)])
    relation = piece["reference_relation"]
    assert relation["state"] == rr.EXACT
    assert relation["coverage"]["median_absence"] <= rr.default_recipe().exact_max_absence


def test_absence_thresholds_are_declared_and_validated():
    from dataclasses import replace

    base = rr.default_recipe()
    assert replace(base, exact_max_absence=0.02).recipe_id != base.recipe_id
    with pytest.raises(rr.ReferenceRelationsError):
        replace(base, exact_max_absence=0.05, contradict_absence=0.04).validate()


# --- 独立検証（reference-intelligence-independent-20261007）の反例：消さない側へ倒す -----------------------

@pytest.mark.parametrize("db", [-12, -18])
@pytest.mark.parametrize("dialogue_gain", [0.0, 0.3])
def test_a_quiet_unregistered_layer_under_a_released_recording_is_never_exact(tmp_path, signals, db, dialogue_gain):
    """Gate 1：-12dB・-18dBの重ねは旧しきい値（novelty 0.06）の分解能の外で、既発売として消されていた。"""
    registered = _at(signals["A"][int(20 * RATE):int(52 * RATE)], 0.1)
    other = _at(_music(55, 32.0), 0.1 * 10 ** (db / 20))
    episode = Episode(60.0, [(10.0, registered + other)], dialogue_gain=dialogue_gain)
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 42)])
    assert rr.EXACT not in {p["reference_relation"]["state"] for p in pieces}
    applied = rr.apply([{**p, "semantic_class": cue_semantics.WANTED, "reasons": []} for p in pieces],
                       cue_semantics.KNOWN_RELEASED)
    assert all(item["semantic_class"] == cue_semantics.WANTED for item in applied)
    assert rr.occurrences(applied) == []


def test_the_exact_novelty_bound_is_below_what_a_minus_18_db_layer_adds():
    recipe = rr.default_recipe()
    assert recipe.exact_max_novelty <= 0.003 < 0.004     # -18dBの重ねの novelty の最小（Evidence）
    assert recipe.exact_max_novelty <= recipe.exact_max_local_novelty <= recipe.exact_max_cell_novelty


@pytest.mark.parametrize("seconds", [0.25, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("edge", ["start", "end"])
def test_a_short_unregistered_edge_never_vanishes_into_an_exact_span(tmp_path, signals, edge, seconds):
    """Gate 2：端の差し替え（地図が unexplained_music と認めたセルを含む）を、端の吸収でEXACTへ入れない。"""
    clip = signals["A"][int(20 * RATE):int(52 * RATE)].copy()
    k = int(seconds * RATE)
    foreign = _at(_music(77, 32.0), float(np.sqrt(np.mean(clip.astype(np.float64) ** 2))))
    if edge == "start":
        clip[:k] = foreign[:k]
    else:
        clip[-k:] = foreign[-k:]
    episode = Episode(60.0, [(10.0, clip)])
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 42)])
    span = (10.0, 10.0 + seconds) if edge == "start" else (42.0 - seconds, 42.0)
    for piece in pieces:
        if piece["reference_relation"]["state"] == rr.EXACT:
            assert min(span[1], piece["end_seconds"]) <= max(span[0], piece["start_seconds"])
    assert pieces and any(p["start_seconds"] <= span[0] and p["end_seconds"] >= span[1] and
                          p["reference_relation"]["state"] != rr.EXACT for p in pieces)


def test_an_edge_cell_the_map_calls_unexplained_is_not_absorbed(tmp_path, signals):
    clip = signals["A"][int(20 * RATE):int(52 * RATE)].copy()
    clip[-2 * RATE:] = _at(_music(78, 32.0), 0.1)[:2 * RATE]
    episode = Episode(60.0, [(10.0, clip)])
    [piece] = _relate(tmp_path, signals, episode, [_segment(10, 42)])
    relation = piece["reference_relation"]
    assert relation["state"] != rr.EXACT
    kinds = {r["kind"] for r in relation["regions"] if r["end_seconds"] > 40.0}
    assert kinds & {rr.CELL_UNEXPLAINED, rr.CELL_ADDITIONAL}      # 地図は認めている
    assert relation.get("coverage", {}).get("absorbed_edge_cells") is None


def test_a_long_foreign_edge_is_still_split_off_and_the_rest_stays_exact(tmp_path, signals):
    episode = Episode(60.0, [(10.0, signals["A"][:int(20 * RATE)]), (30.0, _music(77, 12.0))])
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 42)])
    assert [(p["start_seconds"], p["end_seconds"], p["reference_relation"]["state"]) for p in pieces][0] == \
        (10.0, 30.0, rr.EXACT)


def test_unregistered_music_that_is_also_in_the_nonmusic_stem_is_not_explained_away(tmp_path, signals):
    """joint-accounting 反例：重ねた曲を非音楽のstemも持つと、台詞・効果音の回帰がそれを『説明』してしまう。"""
    registered = signals["A"][int(20 * RATE):int(52 * RATE)]
    other = _at(_music(55, 32.0), float(np.sqrt(np.mean(registered.astype(np.float64) ** 2))))
    episode = Episode(60.0, [(10.0, registered + other)])
    library = _library(tmp_path, signals)
    matches = identity.match(episode.mixture, library.index, collapse=False)
    twin = np.zeros_like(episode.music)
    twin[int(10 * RATE):int(10 * RATE) + other.size] = other
    pieces = rr.relate_segments([_segment(10, 42)], query=episode.mixture, library=library, matches=matches,
                                music=episode.music, interference=[twin])
    assert rr.EXACT not in {p["reference_relation"]["state"] for p in pieces}


def test_exact_still_requires_a_clean_excerpt_and_records_its_margins(tmp_path, signals):
    episode = Episode(60.0, [(10.0, signals["A"][int(12 * RATE):int(42 * RATE)])], dialogue_gain=0.0)
    [piece] = _relate(tmp_path, signals, episode, [_segment(10, 40)])
    coverage = piece["reference_relation"]["coverage"]
    assert piece["reference_relation"]["state"] == rr.EXACT
    recipe = rr.default_recipe()
    assert coverage["median_novelty"] <= recipe.exact_max_novelty
    assert coverage["max_local_novelty"] <= recipe.exact_max_local_novelty
    assert coverage["interference_share"] <= recipe.exact_max_interference_share


def test_a_cue_cut_exactly_at_the_excerpt_is_not_blamed_for_the_music_next_to_it(tmp_path, signals):
    """端のセルの窓は区間の外へはみ出す。区間の外の音楽（次のcue）で、ぴったり切った既発売の抜粋を非EXACTにしない。"""
    episode = Episode(70.0, [(10.0, signals["A"][int(12 * RATE):int(42 * RATE)]), (40.0, _music(61, 25.0))])
    [piece] = _relate(tmp_path, signals, episode, [_segment(10, 40)])
    assert piece["reference_relation"]["state"] == rr.EXACT
    assert piece["reference_relation"]["coverage"]["absorbed_edge_cells"] == []

# --- v5：台詞の無いセルの「説明できない残り」（integrated-r2-claude の実の分離の反例） -------------------------

def _hidden_under(clip: np.ndarray, db: float, seed: int = 5) -> np.ndarray:
    """clip と同じ時間周波数の包絡で位相だけ無作為な音（参照の4倍を超える成分が出ない＝novelty に現れない重ね）。"""
    from scipy.signal import istft, stft

    _, _, spec = stft(clip.astype(np.float64), fs=RATE, nperseg=512)
    phase = np.exp(2j * np.pi * np.random.default_rng(seed).random(spec.shape))
    _, noise = istft(np.abs(spec) * phase, fs=RATE, nperseg=512)
    noise = noise[:clip.size]
    return _at(noise, float(np.sqrt(np.mean(clip.astype(np.float64) ** 2))) * 10 ** (db / 20))


def test_a_layer_hidden_under_the_reference_spectrum_is_never_exact(tmp_path, signals):
    """実の分離器は -18dB の重ねを novelty の閾値の下へ弱める。台詞の無いセルでは参照が同じ録音をほぼ完全に
    説明するので、説明できない残り（1 − joint）がその重ねを見つける。"""
    registered = signals["A"][int(20 * RATE):int(52 * RATE)]
    episode = Episode(60.0, [(10.0, registered + _hidden_under(registered, -20.0))])
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 42)])
    assert rr.EXACT not in {p["reference_relation"]["state"] for p in pieces}
    # この判断を決めたのは残りの上限（上限を外すと、novelty などの他の条件は満たしてEXACTになる）
    from dataclasses import replace

    loose = replace(rr.default_recipe(), exact_max_clean_residual=0.5, exact_max_local_clean_residual=0.5)
    (tmp_path / "loose").mkdir()
    loose_pieces = _relate(tmp_path / "loose", signals, episode, [_segment(10, 42)], recipe=loose)
    assert rr.EXACT in {p["reference_relation"]["state"] for p in loose_pieces}


def test_a_clean_released_excerpt_keeps_no_unexplained_residual_and_stays_exact(tmp_path, signals):
    episode = Episode(60.0, [(10.0, signals["A"][int(12 * RATE):int(42 * RATE)])])
    [piece] = _relate(tmp_path, signals, episode, [_segment(10, 40)])
    coverage = piece["reference_relation"]["coverage"]
    assert piece["reference_relation"]["state"] == rr.EXACT
    assert coverage["clean_residual"] <= rr.default_recipe().exact_max_clean_residual
    assert coverage["max_local_clean_residual"] <= rr.default_recipe().exact_max_local_clean_residual


def test_the_clean_residual_bounds_are_declared_and_validated():
    from dataclasses import replace

    base = rr.default_recipe()
    assert base.exact_max_clean_residual < 0.013     # -18dBの重ねの実の分離での残り（Evidence）
    assert replace(base, exact_max_clean_residual=0.004).recipe_id != base.recipe_id
    with pytest.raises(rr.ReferenceRelationsError):
        replace(base, exact_max_clean_residual=0.01, exact_max_local_clean_residual=0.005).validate()
    with pytest.raises(rr.ReferenceRelationsError):
        replace(base, exact_clean_interference=0.6).validate()


def _released(piece):
    return rr.apply([{**piece, "semantic_class": cue_semantics.WANTED, "reasons": []}],
                    cue_semantics.KNOWN_RELEASED)[0]


@pytest.mark.parametrize("dialogue_gain", [0.0, 0.3])
def test_a_shortened_cut_of_a_released_recording_is_that_recording_and_not_extracted(tmp_path, signals,
                                                                                     dialogue_gain):
    """番組が短くした（途中を切って繋いだ）既発売の録音は、その録音そのもの（ユーザー 2026-10-19）。"""
    edit = np.concatenate([signals["A"][:int(14 * RATE)], signals["A"][int(36 * RATE):int(54 * RATE)]])
    episode = Episode(60.0, [(10.0, edit)], dialogue_gain=dialogue_gain)
    [piece] = _relate(tmp_path, signals, episode, [_segment(10, 42)])
    relation = piece["reference_relation"]
    assert relation["state"] == rr.EDIT and relation["released_cover"]["complete"], relation["released_cover"]
    known = _released(piece)
    assert known["semantic_class"] == cue_semantics.KNOWN_RELEASED and known["reasons"] == [rr.REASON_EDIT]
    [use] = rr.occurrences([known])
    assert use["relation"] == rr.EDIT and use["extraction"] == "not_created_independent_recording_available"
    assert {round(p["reference_range_seconds"][0]) for p in use["reference_pieces"]} >= {0, 36}
    assert rr.related_occurrences([known]) == []


@pytest.mark.parametrize("overlay", [False, True])
def test_partial_alternate_alignment_does_not_hide_a_third_edit_piece(tmp_path, signals, overlay):
    # Fingerprints nominate the first two pieces. The third is still unknown
    # until the local waveform search accounts for it over its own full span.
    edit = np.concatenate([signals["A"][:20*RATE], signals["A"][40*RATE:48*RATE],
                           signals["A"][60*RATE:72*RATE]])
    if overlay:
        edit[28*RATE:] += _at(_music(77, 12.), float(np.sqrt(np.mean(edit[28*RATE:]**2))) * .125)
    episode = Episode(60.0, [(10.0, edit)])
    library = _library(tmp_path, signals)
    matches = identity.match(episode.mixture, library.index, collapse=False)
    matches = [m for m in matches if abs(m.offset_seconds+10) < .1 or abs(m.offset_seconds-10) < .1]
    assert matches and any(abs(m.offset_seconds-10) < .1 for m in matches)
    [piece] = rr.relate_segments([_segment(10, 50)], library=library, matches=matches,
                                music=episode.music, interference=[episode.dialogue])
    relation = piece["reference_relation"]
    assert relation["state"] == rr.EDIT
    if overlay:
        assert not relation["released_cover"]["complete"]
        assert _released(piece)["semantic_class"] == cue_semantics.WANTED
        return
    assert relation["released_cover"]["complete"], relation["released_cover"]
    assert _released(piece)["semantic_class"] == cue_semantics.KNOWN_RELEASED
    assert any(r.get("origin") == "local_search" and r["start_seconds"] >= 37
               for r in relation["regions"])


def test_a_cut_edit_without_a_distribution_receipt_is_still_extracted(tmp_path, signals):
    edit = np.concatenate([signals["A"][:int(14 * RATE)], signals["A"][int(36 * RATE):int(54 * RATE)]])
    episode = Episode(60.0, [(10.0, edit)])
    [piece] = _relate(tmp_path, signals, episode, [_segment(10, 42)])
    piece["reference_relation"]["reference"]["independent_distribution"] = {"state": "independent_distribution_unknown"}
    kept = _released(piece)
    assert kept["semantic_class"] == cue_semantics.WANTED
    assert kept["independent_availability"] == "unknown_preserve_extraction"


def test_a_cut_edit_with_another_recording_inserted_is_kept(tmp_path, signals):
    """切り貼りの間に別の曲（未知の素材）が入れば、録音の部分だけではできていない：消さない。"""
    edit = np.concatenate([signals["A"][:int(14 * RATE)], _music(77, 6.0), signals["A"][int(36 * RATE):int(48 * RATE)]])
    episode = Episode(60.0, [(10.0, edit)])
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 42)])
    for piece in pieces:
        known = _released(piece)
        if piece["start_seconds"] < 30 and piece["end_seconds"] > 26:
            assert known["semantic_class"] == cue_semantics.WANTED


@pytest.mark.parametrize("level", [1.0, 0.25, 0.125])
def test_unregistered_music_over_one_piece_of_a_cut_edit_keeps_the_cue(tmp_path, signals, level):
    """切り貼りの一部へ別の音楽を重ねた番組の版（-18dBまで）は、既発売の録音の使用にしない。"""
    first = signals["A"][:int(14 * RATE)].copy()
    second = signals["A"][int(36 * RATE):int(54 * RATE)].copy()
    rms = float(np.sqrt(np.mean(second.astype(np.float64) ** 2)))
    second = second + _at(_music(55, 18.0), rms * level)
    episode = Episode(60.0, [(10.0, np.concatenate([first, second]))])
    pieces = _relate(tmp_path, signals, episode, [_segment(10, 42)])
    covering = [p for p in pieces if p["start_seconds"] < 36 and p["end_seconds"] > 26]
    assert covering and all(_released(p)["semantic_class"] == cue_semantics.WANTED for p in covering)


def _theme_between_two_cues(seconds_each: float) -> tuple[Episode, dict]:
    """番組：知らない曲 → 登録した録音（主題歌に見立てる）→ 知らない曲。知らない曲の中に、登録した録音の
    別の位置が数秒だけ偶然に似る所（一致の点在）がある。"""
    theme = _music(1, 80.0)
    first, last = _music(77, seconds_each), _music(78, seconds_each)
    first[int(20 * RATE):int(23 * RATE)] = theme[int(70 * RATE):int(73 * RATE)]
    last[int(30 * RATE):int(33 * RATE)] = theme[int(5 * RATE):int(8 * RATE)]
    episode = Episode(2 * seconds_each + 70.0, [(0.0, first), (seconds_each, theme[:int(60 * RATE)]),
                                                (seconds_each + 60.0, last)])
    return episode, {"A": theme, "B": _music(2, 60.0)}


def test_a_held_long_span_is_split_at_the_registered_recording_and_the_neighbours_are_kept(tmp_path):
    """150秒を超えて保留される区間（OP/EDと隣の曲）：点在する偶然の一致があっても、録音の範囲で分ける。"""
    episode, signals = _theme_between_two_cues(55.0)
    pieces = _relate(tmp_path, signals, episode, [_segment(0, 170)])
    states = _states(pieces)
    assert (55.0, 115.0, rr.EXACT) in states
    # 隣の曲は消えず、録音と同じ片にもならない。
    assert min(p["start_seconds"] for p in pieces) == 0.0 and max(p["end_seconds"] for p in pieces) == 170.0
    assert all(state != rr.EXACT for start, end, state in states if end <= 55.0 or start >= 115.0)


def test_a_span_a_cue_can_hold_keeps_the_ordinary_edge_rule(tmp_path):
    """一曲の長さに収まる区間は従来どおり（点在する一致がある端は、主の録音の版として残す）。"""
    episode, signals = _theme_between_two_cues(40.0)
    pieces = _relate(tmp_path, signals, episode, [_segment(0, 140)])
    recipe = rr.default_recipe()
    assert 140 <= recipe.long_split_seconds
    assert rr.EXACT not in [state for _, _, state in _states(pieces)]
