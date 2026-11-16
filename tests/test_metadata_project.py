"""Per-cue tags: official disc/track for released recordings, broadcast order for unreleased pieces."""
from lss.metadata import project as P, resolve as R
from lss.metadata.model import Credit, Release, Track


def album():
    v1 = R.merge_volume([Release("musicbrainz", "u", "作品 オリジナルサウンドトラック Vol.1", tracks=(
        Track(1, 1, "メインテーマ", length_seconds=60, credits=(Credit("Composer / 作曲", "composer", "作曲家"),)),
        Track(1, 2, "空飛ぶ探検隊のアイキャッチ", length_seconds=5)))])
    v2 = R.merge_volume([Release("musicbrainz", "u2", "作品 オリジナルサウンドトラック Vol.2", tracks=(
        Track(1, 1, "決戦", length_seconds=90),))])
    return R.series_album("作品", [v1, v2])


def occ(episode, start, scene=None):
    return {"episode_label": f"第{episode}話", "episode_order": episode, "start_seconds": start, "end_seconds": start + 30,
            "observed_scene": scene, "episode_date": f"2023-04-{episode:02d}"}


SERIES = {"first_date": "2023-04-14", "music": ("作曲家",)}


def test_released_edit_takes_official_disc_track_credits_and_relation():
    a = album()
    track = P.find_official(a, {"album": "作品 オリジナルサウンドトラック Vol.2", "tracknumber": "1", "title": "決戦"})
    assert (track.disc, track.number) == (2, 1)
    pieces = {"p": {"official": {"disc": 1, "number": 1}, "relation_state": "released_recording_edit",
                    "occurrences": [occ(3, 65.0, "バトル"), occ(1, 273.0)]}}
    tags = P.tags_for("p", a, pieces, SERIES)
    assert tags["ALBUM"] == "作品 オリジナル・サウンドトラック" and tags["DATE"] == "2023-04-14"
    assert (tags["DISCNUMBER"], tags["TRACKNUMBER"]) == ("1", "1") and "TOTALDISCS" not in tags
    assert tags["COMPOSER"] == ["作曲家"] and tags["EDIT_OF"] == "メインテーマ" and "RELATION" not in tags
    assert tags["USE_EPISODE"] == ["第1話", "第3話"] and tags["USE_START"] == ["04:33", "01:05"]
    assert tags["USE_END"] == ["05:03", "01:35"] and tags["USE_SCENE"] == ["", "バトル"]
    assert (tags["FIRST_USE_EPISODE"], tags["FIRST_USE_DATE"]) == ("第1話", "2023-04-01")
    assert "SERIES_MUSIC" not in tags and "ARTIST" not in tags


def test_unreleased_pieces_are_on_the_extra_disc_in_broadcast_order_and_renumber():
    a = album()
    pieces = {"late": {"occurrences": [occ(5, 10.0)]}, "early": {"occurrences": [occ(2, 300.0), occ(7, 1.0)]},
              "rel": {"official": {"disc": 2, "number": 1}, "occurrences": [occ(1, 1.0)]}}
    tags = P.tags_for("late", a, pieces, SERIES)
    assert (tags["DISCNUMBER"], tags["TRACKNUMBER"]) == ("3", "2") and "USE_SCENE" not in tags
    assert tags["SERIES_MUSIC"] == ["作曲家"] and "COMPOSER" not in tags
    pieces["earlier"] = {"occurrences": [occ(1, 50.0)]}
    assert P.tags_for("late", a, pieces, SERIES)["TRACKNUMBER"] == "3"
    assert P.tags_for("earlier", a, pieces, SERIES)["TRACKNUMBER"] == "1"


def test_title_only_match_needs_a_unique_track_and_unknown_reference_is_none():
    a = album()
    assert P.find_official(a, {"title": "空飛ぶ探検隊のアイキャッチ"}).number == 2
    assert P.find_official(a, {"title": "存在しない曲"}) is None
    assert P.find_official(a, None) is None


def two_disc_album():
    v = R.merge_volume([Release("vgmdb", "v", "二枚組 オリジナル・サウンドトラック", tracks=(
        Track(1, 1, "一", length_seconds=60), Track(1, 2, "二", length_seconds=70),
        Track(2, 1, "三", length_seconds=80), Track(2, 2, "四", length_seconds=90)))])
    return R.series_album("二枚組", [v])


def test_store_running_number_reference_finds_the_cd_track_by_length():
    a = two_disc_album()
    ref = {"album": "二枚組 オリジナル・サウンドトラック", "tracknumber": "3", "discnumber": "1", "length_seconds": 80.3}
    assert (lambda t: (t.disc, t.number))(P.find_official(a, ref)) == (2, 1)
    # Number 2 fits disc 1 track 2 by both readings; a length that fits neither decides nothing.
    assert P.find_official(a, {**ref, "tracknumber": "2", "length_seconds": 200.0}) is None
    assert P.find_official(a, {"title": "四 (Instrumental)", "length_seconds": 90.0}).number == 2



def test_a_cue_holding_part_of_a_release_stays_its_own_piece_and_says_which_release_plays_where():
    a = album()
    o = occ(1, 600.0)
    o["contains_released"] = [{"title": "決戦", "start_seconds": 620.0, "end_seconds": 651.0}]
    tags = P.tags_for("p", a, {"p": {"occurrences": [o]}}, SERIES)
    assert tags["DISCNUMBER"] == "3" and "EDIT_OF" not in tags
    assert tags["CONTAINS_RELEASED"] == ["決戦"] and tags["CONTAINS_RELEASED_EPISODE"] == ["第1話"]
    assert (tags["CONTAINS_RELEASED_START"], tags["CONTAINS_RELEASED_END"]) == (["10:20"], ["10:51"])



def test_a_program_edit_of_a_release_has_its_own_relation_tag():
    a = album()
    pieces = {"p": {"official": {"disc": 1, "number": 2}, "relation_state": "released_recording_edited_by_program",
                    "occurrences": [occ(1, 1.0)]}}
    tags = P.tags_for("p", a, pieces, SERIES)
    assert tags["PROGRAM_EDIT_OF"] == "空飛ぶ探検隊のアイキャッチ" and "EDIT_OF" not in tags
