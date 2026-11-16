"""既発売録音を部品にした再構成の判断（置き換える領域・新しい資産として出す理由）。"""
from __future__ import annotations

import numpy as np
import pytest
import hashlib
from tests.release_receipts import distribution_receipt

from lss.bgm import reference_reconstruction as rc


def _provenance(cue: float, rows: list[tuple[float, float, str, float, bool]]) -> dict:
    start=min((r[0] for r in rows),default=0.)
    return {"cue_seconds": cue,"cue_start_seconds":start,"cue_end_seconds":start+cue, "donor_regions": [
        {"start_seconds": a, "end_seconds": b, "reference_sha256": hashlib.sha256(sha.encode()).hexdigest(), "offset_seconds": offset, "accepted": ok}
        for a, b, sha, offset, ok in rows]}


def _availability(p):
    return rc.availability(p,distribution={r['reference_sha256']:distribution_receipt(r['reference_sha256'])
                                          for r in p['donor_regions']})


def test_a_cue_made_entirely_of_one_released_span_is_already_available():
    """番組の版が、既発売の録音の一続きの範囲でできている（残りは関係判定の1セル未満）：写しを新しく作らない。"""
    found = _availability(_provenance(46.0, [(156.0, 202.0, "r", -155.946, True)]))
    assert found["independently_available"] is True
    assert found["basis"] == "one_contiguous_released_span_makes_the_cue"
    assert found["reference_range_seconds"] == pytest.approx([0.054, 46.054], abs=0.01)
    assert found["unique_seconds"] == 0.


def test_a_short_episode_tail_is_kept_as_a_new_asset():
    """v4（integrated-r2-claude）：既発売の範囲＋番組だけの短い尻尾（同じ録音の別の位置へ移る2秒・最後の2秒の別の
    曲）は、既発売では手に入らない版。v3の「4秒かcueの10%」は、これを既発売として保留して欲しい音楽を消していた。"""
    jump = _availability(_provenance(46.0, [(156.0, 200.16, "r", -155.946, True),
                                               (200.16, 202.0, "r", -153.607, True)]))
    assert jump["independently_available"] is False
    assert jump["unique_seconds"] == pytest.approx(1.84, abs=0.01)
    replaced_end = _availability(_provenance(32.0, [(0.0, 30.0, "r", 20.0, True)]))
    assert replaced_end["independently_available"] is False
    assert replaced_end["basis"] == "episode_unique_structure_or_material"


@pytest.mark.parametrize("rows, cue", [
    # 番組だけの編集（同じ録音の二つの位置をほぼ同じ長さで繋ぐ）
    ([(5.0, 13.0, "r", -5.0, True), (13.0, 25.0, "r", 11.0, True)], 20.0),
    # 半分は置き換えていない（番組固有の素材・確かめていない領域は番組だけの部分として数える）
    ([(10.0, 30.0, "r", 2.0, True)], 40.0),
    # 二つの別の録音を繋いだ版
    ([(0.0, 20.0, "a", 1.0, True), (20.0, 40.0, "b", 3.0, True)], 40.0),
    # 置き換えなかった（描いた後の確認で外れた）
    ([(0.0, 40.0, "a", 1.0, False)], 40.0),
])
def test_unique_structure_or_material_is_worth_a_new_asset(rows, cue):
    assert _availability(_provenance(cue, rows))["independently_available"] is False


def test_even_a_subsecond_unknown_tail_is_preserved_at_any_cue_length():
    assert _availability(_provenance(120.0, [(0.0,119.5,"r",0.,True)]))["independently_available"] is False
    assert _availability(_provenance(120.0, [(0.0,118.,"r",0.,True)]))["independently_available"] is False


def test_overlap_duplicates_cannot_cover_an_internal_unknown_gap():
    p=_provenance(20.,[(0.,10.,'r',0.,True),(0.,10.,'r',0.,True),(10.5,20.,'r',0.,True)])
    found=_availability(p)
    assert not found['independently_available']
    assert found['unique_seconds']==.5 and found['uncovered_spans']==[[10.,10.5]]


def test_complete_reconstruction_still_needs_distribution_and_protected_material_survives():
    p=_provenance(20.,[(0.,20.,'r',0.,True)])
    assert not rc.availability(p)['independently_available']
    assert _availability(p)['independently_available']
    p['protected_spans']=[[19.75,20.]]
    assert not _availability(p)['independently_available']


def test_component_variants_donate_only_regions_without_missing_components():
    relation = {"state": "released_recording_component_variant", "regions": [
        {"start_seconds": 0.0, "end_seconds": 10.0, "kind": "supported", "reference_sha256": "r",
         "offset_seconds": 1.0, "median_joint": 0.9, "median_novelty": 0.003, "max_novelty": 0.02,
         "median_absence": 0.003, "max_absence": 0.01},
        {"start_seconds": 10.0, "end_seconds": 20.0, "kind": "missing_component"},
        {"start_seconds": 20.0, "end_seconds": 30.0, "kind": "supported", "reference_sha256": "r",
         "offset_seconds": 1.0, "median_joint": 0.9, "median_novelty": 0.003, "max_novelty": 0.02,
         "median_absence": 0.02, "max_absence": 0.03}]}
    planned = rc.plan(relation, 0.0, 30.0)
    assert planned["eligible"] is True
    assert [(d["start_seconds"], d["end_seconds"]) for d in planned["donors"]] == [(0.0, 10.0)]
    assert planned["rejected_regions"][0]["median_absence"] == 0.02
    assert rc.plan({"state": "released_recording_with_additional_material", "regions": []}, 0, 30)["eligible"] is False


# --- 参照では説明できない領域は、橋渡しも描画も置き換えない（独立検証 Gate 6） ---------------------------

def _gap_relation(kind: str, *, sha: str = "r") -> dict:
    clean = {"reference_sha256": sha, "offset_seconds": 0.0, "median_joint": 1.0, "median_novelty": 0.0,
             "max_novelty": 0.0, "median_absence": 0.0, "max_absence": 0.0}
    return {"state": "released_recording_edit", "regions": [
        {"start_seconds": 0.0, "end_seconds": 10.0, "kind": "supported", **clean},
        {"start_seconds": 10.0, "end_seconds": 11.0, "kind": kind, **({"reference_sha256": sha}
                                                                   if kind == "supported" else {})},
        {"start_seconds": 11.0, "end_seconds": 21.0, "kind": "supported", **clean}]}


@pytest.mark.parametrize("kind", ["unexplained_music", "additional_material", "missing_component"])
def test_a_gap_that_the_map_calls_unexplained_is_never_bridged_or_covered(kind):
    planned = rc.plan(_gap_relation(kind), 0.0, 21.0)
    assert [(d["start_seconds"], d["end_seconds"]) for d in planned["donors"]] == [(0.0, 10.0), (11.0, 21.0)]
    assert planned["protected_spans"] == [[10.0, 11.0]]


@pytest.mark.parametrize("kind", ["uncertain", "quiet", "supported"])
def test_a_gap_of_the_same_recording_window_straddle_is_still_bridged(kind):
    planned = rc.plan(_gap_relation(kind), 0.0, 21.0)
    assert [(d["start_seconds"], d["end_seconds"]) for d in planned["donors"]] == [(0.0, 10.5), (10.5, 21.0)]
    assert planned["protected_spans"] == []


def test_a_gap_supported_by_another_recording_or_missing_from_the_map_is_not_bridged():
    other = _gap_relation("supported")
    other["regions"][1].update(reference_sha256="other", offset_seconds=0.0)
    assert len(rc.plan(other, 0.0, 21.0)["donors"]) == 2
    assert rc.plan(other, 0.0, 21.0)["donors"][0]["end_seconds"] == 10.0
    hole = _gap_relation("quiet")
    del hole["regions"][1]
    assert rc.plan(hole, 0.0, 21.0)["donors"][0]["end_seconds"] == 10.0


def _db(x) -> float:
    return 10 * np.log10(max(float(np.mean(np.square(x))), 1e-30))


def test_render_leaves_the_local_material_between_two_donors_untouched():
    """独立反証 Gate 6：[16,17) の番組だけの素材を、両側の置き換えが覆って上書きしない。"""
    rate = 11025
    t = np.arange(33 * rate) / rate
    reference = np.vstack([0.12 * np.sin(2 * np.pi * 440 * t), 0.12 * np.sin(2 * np.pi * 550 * t)])
    base = reference.copy()
    a, b = 16 * rate, 17 * rate
    local = 0.12 * np.sin(2 * np.pi * 1320 * t[a:b])
    base[:, a:b] = local
    clean = {"reference_sha256": "r", "offset_seconds": 0.0, "median_joint": 1.0, "median_novelty": 0.0,
             "max_novelty": 0.0}
    relation = {"state": "released_recording_edit", "regions": [
        {"start_seconds": 0.0, "end_seconds": 16.0, "kind": "supported", **clean},
        {"start_seconds": 16.0, "end_seconds": 17.0, "kind": "unexplained_music"},
        {"start_seconds": 17.0, "end_seconds": 33.0, "kind": "supported", **clean}]}
    planned = rc.plan(relation, 0.0, 33.0)
    out, provenance = rc.render(base, rate, 0.0, planned["donors"], lambda _: reference)
    assert provenance["replaced_seconds"] > 20.0                # 両側は置き換える
    assert np.array_equal(out[:, a:b], base[:, a:b])           # 番組だけの素材は1標本も変えない
    # 局所の素材の1320Hz成分（出力の中の射影）は入力と1dB以内
    for channel in range(2):
        project = lambda x: float(np.abs(np.vdot(np.exp(-2j * np.pi * 1320 * t[a:b]), x[channel, a:b])))
        assert abs(20 * np.log10(project(out) / project(base))) < 1.0
    # 近傍（クロスフェードの幅）でも保護した領域には予測が入らない
    assert np.array_equal(out[:, a:b], base[:, a:b])


def test_render_never_gives_a_protected_span_to_a_prediction_even_when_the_plan_donor_overlaps_it():
    """計画が protected_spans を持つ限り、重なる置き換え（古い計画・手で作った計画）でも元の音を守る。"""
    rate = 11025
    t = np.arange(12 * rate) / rate
    reference = np.vstack([0.1 * np.sin(2 * np.pi * 330 * t)] * 2)
    base = reference.copy()
    base[:, 5 * rate:6 * rate] = 0.1 * np.sin(2 * np.pi * 990 * t[5 * rate:6 * rate])
    donor = {"start_seconds": 0.0, "end_seconds": 12.0, "reference_sha256": "r", "offset_seconds": 0.0,
             "protected_spans": [[5.0, 6.0]]}
    out, _ = rc.render(base, rate, 0.0, [donor], lambda _: reference)
    assert np.array_equal(out[:, 5 * rate:6 * rate], base[:, 5 * rate:6 * rate])
