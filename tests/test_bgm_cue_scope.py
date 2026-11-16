"""意味判定の区間を対応領域S3へ写し、音の中身の安全の観測と番組構造の文脈を加えて公開・保留・対象外を
名前付きで決める（`lss.bgm.cue_scope`）。番組の役割は文脈で、聞こえる劇伴を対象外にしない。"""
from __future__ import annotations

import numpy as np
import pytest

from lss.bgm import cue_scope, cue_semantics

SOURCE = {"path": "episode.mkv", "sha256": "a" * 64}


def segment(label=cue_semantics.WANTED, reasons=(), start=10.0, end=40.0, *, start_edge="music_gap",
            end_edge="music_gap", **extra) -> dict:
    return {"start_seconds": start, "end_seconds": end, "core_seconds": [start, end],
            "start_edge": {"kind": start_edge}, "end_edge": {"kind": end_edge},
            "semantic_class": label, "reasons": list(reasons),
            "features": {"inner_split_distance": 0.15 if "possible_change_inside" in reasons else 0.05},
            **extra}


def quiet_mixture(windows=200):
    """非音楽のイベント音も放送挿入も無い原本のタグ（音楽だけ）。"""
    mixture = np.zeros((windows, 527), dtype=np.float32)
    mixture[:, 137] = 0.9
    return mixture


def decided(*segments, mixture=None, boundaries=()):
    items = list(segments)
    result = cue_scope.decide(items, SOURCE, mixture=quiet_mixture() if mixture is None else mixture,
                              boundaries=boundaries)
    return items, result


def test_the_scope_is_frozen_and_identified():
    scope = cue_scope.product_scope()
    assert scope.cue_extent_policy == "allow_unresolved_cue_extent"
    assert scope.published_semantic_roles == ("dramatic_score", "singing_score")
    assert scope.minimum_cue_seconds == 10.0 and scope.maximum_cue_seconds == 150.0
    assert scope.quality_gates == {"integrity_passed": 1.0}
    assert cue_scope.recipe()["scope_id"] == scope.scope_id


def test_a_wanted_segment_is_published_as_one_cue():
    [item], result = decided(segment())
    assert item["scope"]["disposition"] == "PUBLISHED" and item["scope"]["reason"] == "within_supported_scope"
    assert item["scope"]["cue_extent"]["state"] == "single_cue"
    assert result["disposition"] == "PUBLISHED"


def test_observed_singing_is_preserved_without_asserting_a_singing_identity():
    [item], _ = decided(segment(features={"singing_present": True, "theme_shape": True}))
    assert item["scope"]["disposition"] == "PUBLISHED"
    candidate = cue_scope.candidate(segment(features={"singing_present": True, "theme_shape": True}), 0, SOURCE,
                                    {"integrity_passed": 1.0})
    assert candidate["semantic_role"] == "dramatic_score"
    assert candidate["observations"] == {"singing": "present", "theme_shape": True}


def test_a_segment_held_only_for_a_possible_change_inside_is_published_with_its_extent_unresolved():
    [item], _ = decided(segment(cue_semantics.UNCERTAIN, ["possible_change_inside"]))
    assert cue_scope.publishable(item) and cue_scope.unresolved_extent(item)
    assert item["scope"]["cue_extent"] == {"state": "unresolved", "reasons": ["possible_change_inside"],
                                           "inner_split_distance": 0.15}


@pytest.mark.parametrize("reasons,disposition,reason", [
    (["singing_present"], "PUBLISHED", "within_supported_scope"),
    (["possible_change_inside", "singing_present"], "PUBLISHED", "within_supported_scope"),
    (["music_not_continuous"], "HELD", "semantic_role_unresolved"),
    (["possible_change_inside", "end_edge_music_continues_outside_candidate"], "HELD", "required_evidence_missing"),
    (["start_edge_music_continues_outside_candidate"], "HELD", "required_evidence_missing"),
    (["partly_registered_reference"], "HELD", "partly_known_released_recording"),
])
def test_every_other_held_segment_keeps_a_named_reason(reasons, disposition, reason):
    [item], _ = decided(segment(cue_semantics.UNCERTAIN, reasons))
    assert (item["scope"]["disposition"], item["scope"]["reason"]) == (disposition, reason)
    assert cue_scope.publishable(item) == ("singing_present" in reasons)


def test_length_outside_one_cue_is_held_even_when_only_the_extent_is_in_doubt():
    [short, long_], _ = decided(segment(cue_semantics.UNCERTAIN, ["shorter_than_wanted_minimum"], 0.0, 9.0),
                                segment(cue_semantics.UNCERTAIN, ["longer_than_one_cue", "possible_change_inside"],
                                        100.0, 260.0))
    assert short["scope"]["reason"] == "below_minimum_cue_usefulness"
    assert long_["scope"]["reason"] == "above_maximum_cue_length"


def test_missing_edge_evidence_is_named():
    [item], _ = decided(segment(cue_semantics.UNCERTAIN, ["start_edge_music_continues_outside_candidate"],
                                start_edge="candidate_edge"))
    assert item["scope"]["missing_evidence"] == ["cue_edges"]


def test_known_release_theme_shape_and_duplicates_are_not_new_assets():
    known, theme, duplicate, quiet = segment(cue_semantics.KNOWN_RELEASED, ["registered_reference_exact_match"]), \
        segment(cue_semantics.OTHER_NON_TARGET, ["theme_song_shape_with_singing"], 0.0, 90.0), \
        segment(cue_semantics.DUPLICATE_PUBLISHED, ["identical_to_published_cue"],
                duplicate_of=[{"path": "earlier.flac"}]), \
        segment(cue_semantics.NON_BGM, ["music_not_heard"])
    items, result = decided(known, theme, duplicate, quiet)
    assert known["scope"]["reason"] == "known_released_recording"
    assert theme["scope"]["disposition"] == "UNSUPPORTED" and theme["scope"]["reason"] == "excluded_non_target_music"
    assert duplicate["scope"]["reason"] == "duplicate_of_published_cue"
    assert quiet["scope"] is None  # 音楽に聞こえない区間は候補ではない
    assert result["disposition"] == "ZERO" and result["reason"] == "no_novel_asset"


def test_only_theme_shaped_music_is_a_valid_zero_not_a_hold():
    _, result = decided(segment(cue_semantics.OTHER_NON_TARGET, ["theme_song_shape_with_singing"], 0.0, 90.0))
    assert (result["disposition"], result["reason"]) == ("ZERO", "music_not_wanted")


def test_no_candidate_is_a_zero_without_music():
    _, result = decided(segment(cue_semantics.NON_BGM, ["music_not_heard"]))
    assert (result["disposition"], result["reason"], result["music_activity_found"]) == (
        "ZERO", "no_wanted_score", False)


def test_a_cue_whose_written_audio_lacks_integrity_is_held():
    [item], _ = decided(segment(cue_semantics.UNCERTAIN, ["possible_change_inside"]))
    rate = 8000
    audio = np.zeros((rate * 30, 2))
    audio[:, 0] = 0.1 * np.sin(np.arange(rate * 30) / 7.0)
    measured = cue_scope.integrity(audio, rate)
    assert measured["integrity_passed"] == 0.0 and measured["silent_channel_indexes"] == [1]
    row = cue_scope.final(item, 0, SOURCE, measured)
    assert (row["disposition"], row["reason"], row["failed_gates"]) == ("HELD", "below_quality_gate",
                                                                        ["integrity_passed"])
    assert row["integrity"] is measured and row["cue_extent"]["state"] == "unresolved"


def test_quiet_and_band_limited_audio_keeps_its_integrity():
    """小さい音や帯域の狭い劇伴を、大きさ・帯域のしきい値で落とさない。"""
    rate = 8000
    t = np.arange(rate * 20) / rate
    quiet = np.stack([1e-4 * np.sin(2 * np.pi * 110 * t), 0.8e-4 * np.sin(2 * np.pi * 110 * t)], axis=1)
    measured = cue_scope.integrity(quiet, rate)
    assert measured["integrity_passed"] == 1.0
    broken = quiet.copy()
    broken[5, 0] = np.nan
    assert cue_scope.integrity(broken, rate)["integrity_passed"] == 0.0


def test_the_publication_candidates_for_duplicate_checks_are_wanted_and_extent_only():
    assert cue_scope.publication_candidate(segment())
    assert cue_scope.publication_candidate(segment(cue_semantics.UNCERTAIN, ["possible_change_inside"]))
    assert not cue_scope.publication_candidate(segment(cue_semantics.UNCERTAIN,
                                                      ["possible_change_inside", "singing_present"]))
    assert not cue_scope.publication_candidate(segment(cue_semantics.KNOWN_RELEASED))


def test_unmeasurable_values_of_silent_audio_are_recorded_as_none():
    import json
    measured = cue_scope.integrity(np.zeros((8000 * 12, 2)), 8000)
    assert measured["integrity_passed"] == 0.0 and measured["silent"] is True
    json.dumps(measured, allow_nan=False)  # 記録へそのまま書ける


def test_a_scene_dense_with_nonmusic_events_is_held_with_a_named_safety_reason():
    mixture = quiet_mixture()
    mixture[10:16, 426] = 0.6   # Explosion：30窓のうち6窓（20%）
    [item], result = decided(segment(), mixture=mixture)
    assert item["scope"]["disposition"] == "HELD" and item["scope"]["reason"] == "safety_hold"
    assert item["scope"]["safety_holds"] == ["heavy_nonmusic_event_scene"]
    assert item["safety"]["event_window_fraction"] == 0.2
    assert result["disposition"] == "HELD"


def test_sparse_events_below_the_scene_fraction_do_not_hold():
    mixture = quiet_mixture()
    mixture[10:15, 72] = 0.6    # Animal：30窓のうち5窓
    mixture[20:30, 504] = 0.19  # 窓の確率の下限に届かない効果音
    [item], _ = decided(segment(), mixture=mixture)
    assert item["scope"]["disposition"] == "PUBLISHED"
    assert item["safety"]["heavy_nonmusic_event_scene"] is False


def test_silence_and_acoustic_environment_classes_are_not_events():
    mixture = quiet_mixture()
    mixture[10:40, 500] = 0.9   # Silence
    mixture[10:40, 506] = 0.9   # Inside, small room
    [item], _ = decided(segment(), mixture=mixture)
    assert item["scope"]["disposition"] == "PUBLISHED"


def test_a_cue_touching_the_program_edge_is_published_with_the_edge_as_context():
    """番組の端は文脈。端に接するだけで劇伴を保留しない（v12は保留していた）。"""
    [item], _ = decided(segment(start=0.0, end=30.0, start_edge="episode_edge"))
    assert item["scope"]["disposition"] == "PUBLISHED"
    assert item["context"]["program_edge"] is True


def test_a_broadcast_tag_is_context_not_a_reason_to_hold_wanted_music():
    mixture = quiet_mixture()
    mixture[25, 524] = 0.24     # Television
    [item], _ = decided(segment(), mixture=mixture)
    assert item["scope"]["disposition"] == "PUBLISHED"
    assert item["context"]["television_tag"] is True and item["context"]["television_max"] == 0.24


def test_an_extent_only_segment_is_held_by_the_same_safety_observations():
    mixture = quiet_mixture()
    mixture[12:20, 300] = 0.5   # Vehicle
    [item], _ = decided(segment(cue_semantics.UNCERTAIN, ("possible_change_inside",)), mixture=mixture)
    assert item["scope"]["reason"] == "safety_hold"


def test_without_safety_observations_nothing_is_published():
    items = [segment()]
    cue_scope.decide(items, SOURCE)
    assert items[0]["scope"]["disposition"] == "HELD" and items[0]["scope"]["reason"] == "safety_not_observed"


def test_safety_is_kept_when_the_decision_is_remade_without_the_arrays():
    items, _ = decided(segment())
    cue_scope.decide(items, SOURCE, quality={0: {"integrity_passed": 1.0}})
    assert items[0]["scope"]["disposition"] == "PUBLISHED"


def test_the_scope_declares_only_content_observations_as_safety_holds():
    scope = cue_scope.product_scope()
    assert scope.safety_holds == ("heavy_nonmusic_event_scene",)
    assert scope.program_insert_policy == "hold_role_unresolved"
    declared = cue_scope.recipe()
    assert declared["safety"]["event_window_probability"] == 0.2 and declared["safety"]["event_scene_fraction"] == 0.2
    assert declared["context"]["publication_effect"] == "none except insert_section"


@pytest.mark.parametrize("label", ["program_edge", "broadcast_insert", "mid_program_short_insert", "insert_block"])
def test_program_role_labels_cannot_be_declared_as_publication_holds(label):
    """番組の役割のラベルは、対応領域の安全の保留として宣言すらできない（契約で閉じる）。"""
    from dataclasses import replace

    from lss.bgm.supported_scope import ScopeError

    with pytest.raises(ScopeError):
        replace(cue_scope.product_scope(), safety_holds=("heavy_nonmusic_event_scene", label)).to_dict()


def test_a_short_isolated_music_block_in_the_middle_of_the_program_is_published():
    """番組の中ほどの短い劇伴は、アイキャッチかもしれないという位置だけで止めない。"""
    [item], _ = decided(segment(start=595.0, end=610.0), mixture=quiet_mixture(1400))
    assert item["scope"]["disposition"] == "PUBLISHED"
    assert item["context"]["mid_program_short_music"] is True


def test_short_material_is_not_assumed_to_have_a_program_structure():
    [item], _ = decided(segment(start=95.0, end=110.0), mixture=quiet_mixture(200))
    assert item["scope"]["disposition"] == "PUBLISHED"


def test_a_candidate_is_split_at_interior_edit_boundaries_without_a_bumper():
    items = cue_scope.split_at_edit_boundaries([segment(start=100.0, end=200.0)], [(140.3, 141.0), (170.2, 170.6)], 1400.0)
    assert [(i["start_seconds"], i["end_seconds"]) for i in items] == [(100.0, 140.3), (141.0, 170.2), (170.6, 200.0)]
    assert items[0]["start_edge"]["kind"] == "music_gap" and items[0]["end_edge"]["kind"] == "edit_boundary"
    assert items[1]["start_edge"]["kind"] == items[1]["end_edge"]["kind"] == "edit_boundary"
    assert all(i["split_from"] == [100.0, 200.0] for i in items)


def test_edit_split_clips_relation_maps_and_preserves_parent_measurement():
    from copy import deepcopy
    from lss.bgm import reference_relations as rr

    relation = {"state": rr.EDIT, "basis": "alignment", "reference": {"reference_sha256": "a" * 64},
        "regions": [
            {"start_seconds": 90., "end_seconds": 110., "reference_sha256": "a" * 64,
             "offset_seconds": 100., "reference_range_seconds": [190., 210.]},
            {"start_seconds": 110., "end_seconds": 150., "reference_sha256": "b" * 64,
             "offset_seconds": -20., "reference_range_seconds": [90., 130.]},
            {"start_seconds": 160., "end_seconds": 170., "reference_sha256": "c" * 64,
             "offset_seconds": 2., "reference_range_seconds": [162., 172.]},
        ],
        "partial_reference_regions": [
            {"start_seconds": 95., "end_seconds": 125., "reference_sha256": "a" * 64,
             "offset_seconds": 30., "reference_range_seconds": [125., 155.]},
            {"start_seconds": 180., "end_seconds": 190., "reference_sha256": "z" * 64,
             "offset_seconds": 1., "reference_range_seconds": [181., 191.]},
        ]}
    parent = {**segment(start=100., end=140.), "reference_relation": relation}
    before = deepcopy(parent)
    parts = cue_scope.split_at_edit_boundaries([parent], [(120., 121.)], 1400.)
    assert [(p["start_seconds"], p["end_seconds"]) for p in parts] == [(100., 120.), (121., 140.)]
    left, right = [p["reference_relation"] for p in parts]
    assert [(r["start_seconds"], r["end_seconds"], r["reference_range_seconds"])
            for r in left["regions"]] == [(100., 110., [200., 210.]), (110., 120., [90., 100.])]
    assert [(r["start_seconds"], r["end_seconds"], r["reference_range_seconds"])
            for r in right["regions"]] == [(121., 140., [101., 120.])]
    assert [(r["start_seconds"], r["end_seconds"], r["reference_range_seconds"])
            for r in left["partial_reference_regions"]] == [(100., 120., [130., 150.])]
    assert [(r["start_seconds"], r["end_seconds"], r["reference_range_seconds"])
            for r in right["partial_reference_regions"]] == [(121., 125., [151., 155.])]
    assert left["measured_scope_seconds"] == right["measured_scope_seconds"] == [100., 140.]
    assert rr.compact(left)["measured_scope_seconds"] == [100., 140.]
    assert parent == before
    assert relation["regions"][0]["reference_range_seconds"] == [190., 210.]


def test_detected_silence_split_keeps_every_nonsilent_frame_in_eligible_pieces(tmp_path):
    import soundfile as sf

    rate = 1000
    clock = np.arange(rate * 640) / rate
    signal = np.column_stack((0.2 * np.sin(2 * np.pi * 37 * clock), 0.15 * np.cos(2 * np.pi * 53 * clock)))
    signal[400300:400700] = 0.
    path = tmp_path / "stereo-source.wav"
    sf.write(path, signal, rate, subtype="FLOAT")
    boundaries = cue_scope.digital_silence_runs(path)
    assert len(boundaries) == 1 and boundaries[0] == pytest.approx((400.3, 400.7))
    parts = cue_scope.split_at_edit_boundaries([segment(start=390., end=415.)], boundaries, 640.)
    kept = np.zeros(len(signal), dtype=bool)
    for part in parts:
        kept[round(part["start_seconds"] * rate):round(part["end_seconds"] * rate)] = True
    wanted = np.zeros(len(signal), dtype=bool)
    wanted[390000:415000] = np.any(signal[390000:415000] != 0., axis=1)
    assert np.all(kept[wanted]), "rounding outwards must not discard notes beside a measured silent edit"
    assert not np.any(kept[400300:400700]), "the two assets must remain separated at the physical silent edit"
    assert all(part["end_seconds"] - part["start_seconds"] >= 1. for part in parts)


def test_boundaries_at_or_outside_the_edges_change_nothing_but_the_silence_itself():
    [item] = cue_scope.split_at_edit_boundaries([segment(start=100.0, end=200.0)], [(99.0, 100.5), (199.4, 200.2)], 1400.0)
    assert (item["start_seconds"], item["end_seconds"]) == (100.5, 199.4)
    [far] = cue_scope.split_at_edit_boundaries([segment(start=100.0, end=200.0)], [(50.0, 51.0), (203.1, 203.6)], 1400.0)
    assert "split_from" not in far   # 境界の数秒前で止まった劇伴を、アイキャッチ等の緩衝として削らない
    held = cue_scope.split_at_edit_boundaries([segment(cue_semantics.UNCERTAIN, ("singing_present",),
                                                       start=100.0, end=200.0)], [(150.0, 151.0)], 1400.0)
    assert len(held) == 1 and "split_from" not in held[0]


def test_music_between_two_edit_boundaries_is_held_as_an_unresolved_insert_and_retained():
    """挿入の区画（CMか、サブタイトル・予告等の番組の挿入か区別できない）は役割未確定の保留。対象外ではない。"""
    boundaries = [(400.0, 400.5), (430.0, 430.5), (800.0, 800.5)]
    items = cue_scope.split_at_edit_boundaries([segment(start=380.0, end=470.0)], boundaries, 1400.0)
    _, result = decided(*items, mixture=quiet_mixture(1400), boundaries=boundaries)
    rows = {(i["start_seconds"], i["end_seconds"]): i["scope"] for i in items}
    insert = rows[(400.5, 430.0)]
    assert (insert["disposition"], insert["reason"]) == ("HELD", "program_insert_role_unresolved")
    assert insert["retention"]["state"] == "retained_internal" and "donor" in insert["retention"]["uses"]
    assert rows[(380.0, 400.0)]["disposition"] == "PUBLISHED" and rows[(430.5, 470.0)]["disposition"] == "PUBLISHED"
    assert result["disposition"] == "PUBLISHED"


def test_short_material_is_not_cut_at_digital_silence():
    [item] = cue_scope.split_at_edit_boundaries([segment(start=5.0, end=25.0)], [(0.0, 5.0), (25.0, 30.0)], 30.0)
    assert "split_from" not in item


def test_no_program_role_observation_turns_wanted_music_into_non_target():
    """番組の役割の観測を全て真にしても、欲しい劇伴は対象外（UNSUPPORTED）にならない。最大でも役割未確定の保留。"""
    mixture = quiet_mixture(1400)
    mixture[:, 524] = 0.9       # Television everywhere
    boundaries = [(0.0, 0.5), (590.0, 590.5), (612.0, 612.5)]
    items = [segment(start=595.0, end=610.0, start_edge="episode_edge")]
    decided(*items, mixture=mixture, boundaries=boundaries)
    assert items[0]["context"]["insert_section"] and items[0]["context"]["program_edge"]
    assert items[0]["context"]["television_tag"] and items[0]["context"]["mid_program_short_music"]
    assert items[0]["scope"]["disposition"] == "HELD" and items[0]["scope"]["reason"] == "program_insert_role_unresolved"
    assert items[0]["scope"]["retention"]["publication"] == "not_public"


def test_held_and_unsupported_segments_record_that_their_audio_is_retained():
    items, _ = decided(segment(cue_semantics.UNCERTAIN, ["music_not_continuous"]),
                       segment(cue_semantics.OTHER_NON_TARGET, ["theme_song_shape_with_singing"], 50.0, 80.0))
    held, unsupported = items
    assert held["scope"]["retention"]["uses"] == ["donor", "identity_evidence", "future_completion", "comparison"]
    assert unsupported["scope"]["disposition"] == "UNSUPPORTED"
    assert unsupported["scope"]["retention"]["state"] == "retained_internal"
