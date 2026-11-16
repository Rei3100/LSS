"""語彙に無い言葉の読み方: 訳した日本語・英語の名詞句・時刻の検出の補助（modelは使わない）。"""
from dataclasses import replace

import numpy as np
import pytest

from lss.sound import intent, interpret, pe_frame


@pytest.mark.parametrize("english,phrase", [
    ("The sound of rain hitting the window.", "rain hitting the window"),
    ("It's the sound of a fire.", "fire"),
    ("I heard the glass crack.", "glass crack"),
    ("The beam's sound.", "beam"),
    ("Helicopter sounds.", "helicopter"),
    ("Fireworks in the distance", "fireworks in the distance"),
    ("A vacuum cleaner.", "vacuum cleaner"),
])
def test_translations_become_short_english_noun_phrases(english, phrase) -> None:
    assert interpret.english_phrase(english) == phrase


def test_untranslatable_text_is_not_turned_into_a_prompt() -> None:
    assert interpret.english_phrase("「」") is None
    assert interpret.english_phrase("ガラス") is None
    assert interpret.english_phrase("one two three four five six seven eight nine ten") is None


class _Reader:
    def __init__(self, known):
        self.known = known
        self.calls = []

    def __call__(self, japanese):
        self.calls.append(japanese)
        if japanese not in self.known:
            return None
        concept = intent._resolve(self.known[japanese], open_vocabulary=True)
        if concept.id.startswith("open:"):
            concept = replace(concept, label=japanese)
        return concept, self.known[japanese]


def test_the_lexicon_comes_first_and_the_reader_is_not_asked() -> None:
    reader = _Reader({})
    request = intent.parse_one("雨の音", reader)
    assert request.supported and request.atoms[0].concept.id == "rain" and not reader.calls


def test_a_whole_japanese_phrase_is_read_through_the_reader() -> None:
    reader = _Reader({"雨が窓に当たる音": "rain hitting the window"})
    request = intent.parse_one("雨が窓に当たる音", reader)
    assert request.supported and request.heard_as == ("rain hitting the window",)
    assert request.atoms[0].concept.id == "open:rain hitting the window"
    assert request.label == "雨が窓に当たる音"


def test_a_translation_that_lands_on_the_lexicon_uses_the_lexicon_sound() -> None:
    reader = _Reader({"窓ガラスが粉々になる音": "glass breaking"})
    request = intent.parse_one("窓ガラスが粉々になる音", reader)
    assert request.atoms[0].concept.id == "glass" and request.label == "ガラスの割れる音"


def test_removal_and_union_keep_working_with_read_words() -> None:
    reader = _Reader({"トーキングドラム": "talking drum"})
    removed = intent.parse_one("トーキングドラムなし", reader)
    assert removed.operation == "remove" and removed.atoms[0].concept.prompt == "talking drum"
    union = intent.parse_one("トーキングドラムと雨", reader)
    assert {atom.concept.id for atom in union.atoms} == {"open:talking drum", "rain"}


def test_unread_japanese_stays_unknown_and_is_not_guessed() -> None:
    reader = _Reader({})
    request = intent.parse_one("かめはめ波", reader)
    assert not request.supported and request.unknown == ("かめはめ波",) and reader.calls == ["かめはめ波"]
    without_reader = intent.parse_one("かめはめ波")
    assert not without_reader.supported and without_reader.unknown == ("かめはめ波",)


def test_the_reader_is_not_asked_about_english() -> None:
    reader = _Reader({})
    request = intent.parse_one("squeaky rubber toy", reader)
    assert request.supported and request.atoms[0].concept.id == "open:squeaky rubber toy" and not reader.calls


def test_frame_spans_and_seconds() -> None:
    flags = np.zeros(100, dtype=bool)
    flags[25:50] = True
    flags[75:80] = True
    assert pe_frame.frames_to_spans(flags) == [(1.0, 2.0), (3.0, 3.2)]
    probabilities = np.zeros(100, dtype=np.float32)
    probabilities[30] = 0.9
    assert list(pe_frame.per_second(probabilities, 4)) == pytest.approx([0.0, 0.9, 0.0, 0.0])


@pytest.mark.parametrize("text", ["左チャンネルのギター", "右側の足音", "大きい音", "loud noise", "序盤の雨",
                                  "5分くらいの雨"])
def test_place_loudness_and_vague_times_are_not_read_as_sounds(text) -> None:
    reader = _Reader({text: "rain"})
    request = intent.parse_one(text, reader)
    assert not request.supported and request.reason == "scope_not_supported" and not reader.calls


@pytest.mark.parametrize("text,unit,start,end,label", [
    ("前半の足音", "fraction", 0.0, 0.5, "前半の足音"),
    ("後半のBGM", "fraction", 0.5, None, "後半の音楽"),
    ("最初の30秒のピアノ", "seconds", 0.0, 30.0, "最初の30秒のピアノ"),
    ("最後の10秒の台詞", "seconds", -10.0, None, "最後の10秒の台詞"),
    ("1分から2分の雨", "seconds", 60.0, 120.0, "1分から2分の雨音"),
    ("1分30秒から2分までの雨", "seconds", 90.0, 120.0, "1分30秒から2分までの雨音"),
    ("the first 30 seconds of rain", "seconds", 0.0, 30.0, "最初の30秒の雨音"),
    ("rain from 0:10 to 0:40", "seconds", 10.0, 40.0, "0分10秒から0分40秒の雨音"),
])
def test_time_ranges_are_read_and_change_the_identity(text, unit, start, end, label) -> None:
    request = intent.parse_one(text)
    assert request.supported and request.label == label
    assert (request.window.unit, request.window.start, request.window.end) == (unit, start, end)
    plain = intent.parse_one(request.atoms[0].concept.label)
    assert request.key() != plain.key() and plain.window is None


def test_a_window_keeps_only_its_time() -> None:
    from lss.sound import verify

    rate = 1000
    output = np.ones((10 * rate, 2), dtype=np.float32)
    frames = verify.Frames(output=np.ones(250, dtype=np.float32), parent=np.ones(250, dtype=np.float32))
    start, end = intent.parse_one("前半の足音").window.bounds(10.0)
    out, parent, words = verify.within(output, output.copy(), frames, rate, start, end)
    assert out[: 4 * rate].min() == 1.0 and out[6 * rate:].max() == 0.0 and parent[6 * rate:].max() == 0.0
    assert words.output[:120].min() == 1.0 and words.output[130:].max() == 0.0
    assert intent.parse_one("最後の3秒の足音").window.bounds(10.0) == (7.0, 10.0)


@pytest.mark.parametrize("text", ["sound", "all sounds", "everything", "the noise"])
def test_generic_words_do_not_name_a_sound(text) -> None:
    assert not intent.parse_one(text).supported


# --- 修飾した音・英語の句・裏付け（2026-10-14） ------------------------------------------------


def test_a_modifier_narrows_the_words_but_keeps_the_route_and_the_check() -> None:
    from lss.sound import lexicon, plan

    child = intent.parse_one("子どもの笑い声").atoms[0].concept
    assert child.base_id == "laughter" and child.prompt == "child laughter" and child.label == "子どもの笑い声"
    assert child.audioset == lexicon.BY_ID["laughter"].audioset
    distant = intent.parse_one("遠くの雷").atoms[0].concept
    assert distant.base_id == "thunder" and distant.prompt == "distant thunder"
    route = plan.route(intent.parse_one("遠くの雷"), "cinematic")
    assert route.parent == plan.dnr("effects") and route.expression["prompt"] == "distant thunder"
    assert intent.parse_one("鳥のさえずり").atoms[0].concept.id == "bird"        # 修飾語が既に音の言葉にある
    assert intent.parse_one("激しいドラム").atoms[0].concept.id == "drums"      # stemそのものは修飾で変わらない
    # 男声・女声は専門modelのstemそのもの（修飾はstemの中身を変えない）
    assert intent.parse_one("男性の低い歌声").atoms[0].concept.id == "male_singing"
    route = plan.route(intent.parse_one("女性ボーカル"), "music")
    assert route.expression == plan.voices("female", plan.vocals("vocals", plan.SOURCE))
    assert route.parent == plan.vocals("vocals", plan.SOURCE) and "text" not in plan.engines(route.expression)
    assert intent.parse_one("ドラムのシンバル").atoms[0].concept.id == "cymbal"


def test_japanese_mentions_count_only_free_standing_single_kanji_words() -> None:
    assert [c.id for c in intent.japanese_mentions("雨が窓に当たる音")] == ["rain"]
    assert intent.japanese_mentions("かめはめ波") == []
    assert all(c.id != "lead_vocals" for c in intent.japanese_mentions("歌舞伎の拍子木"))
    assert [c.id for c in intent.japanese_mentions("強い風")] == ["wind"]
    assert intent.japanese_mentions("猫がミャーと鳴く声") == [intent._lookup_sound("猫")]   # 「声」は総称


def test_english_phrases_are_matched_without_plural_differences() -> None:
    assert [c.id for c in intent.english_mentions("bubble of soda")] == ["bubbles"]
    assert [c.id for c in intent.english_mentions("the sparrows")] == ["sparrow"]


def test_a_translation_must_not_bring_a_sound_the_japanese_does_not_have() -> None:
    chosen = interpret.choose("雷が落ちる音", [{"english": "The sound of thunder falling."}])
    assert chosen["verdict"] == "grounded" and chosen["concept"].base_id == "thunder"
    assert chosen["concept"].label == "雷が落ちる音"
    assert interpret.choose("猫が窓を引っかく音", [{"english": "A dog scratching the window."}]) is None
    agreed = interpret.choose("猫がミャーと鳴く声", [{"english": "It's the sound of a cat barking."},
                                                   {"english": "It's the sound of a cat crying."}])
    assert agreed["verdict"] == "agreed" and agreed["concept"].id == "cat"


def test_ungrounded_translations_need_both_judges() -> None:
    weak = [{"english": "Opening cans.", "similarity": {"labse": 0.75, "mminilm": 0.84}}]
    strong = [{"english": "Opening cans.", "similarity": {"labse": 0.75, "mminilm": 0.93}}]
    assert interpret.choose("缶を開ける音", weak) is None
    chosen = interpret.choose("缶を開ける音", strong)
    assert chosen["concept"].id == "open:opening cans" and chosen["concept"].label == "缶を開ける音"


def test_unknown_words_get_close_lexicon_words_as_hints_only() -> None:
    request = intent.parse_one("ギタ")
    assert not request.supported and request.suggestions == ("ギター",)


def test_sounds_heard_anywhere_take_the_measured_parent() -> None:
    from lss.sound import plan

    clap = intent.parse_one("拍手")
    default = plan.route(clap, "cinematic")
    assert default.parent == plan.dnr("effects")
    measured = plan.route(clap, "cinematic", open_parents={"applause": plan.dnr("music")})
    assert measured.parent == plan.dnr("music") and plan.measured(clap.atoms[0].concept)
    castanets = plan.route(intent.parse_one("カスタネット"), "music")
    assert castanets.parent == plan.stems("drums", plan.SOURCE)
