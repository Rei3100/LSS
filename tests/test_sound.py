"""欲しい音: 語彙・要求の読み方・経路・検証・T5のtokenizer・safetensors・Recipe。"""
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from lss.core import recipe
from lss.sound import audioset, intent, lexicon, plan, safetensors_file, t5_text, verify


# --- 語彙 ------------------------------------------------------------------


def test_every_word_points_to_one_concept_and_every_concept_is_checkable() -> None:
    assert len(lexicon.CONCEPTS) >= 100
    for concept in lexicon.CONCEPTS:
        assert concept.family in lexicon.FAMILIES
        assert t5_text.normalized_prompt(concept.prompt) == concept.prompt
        assert all(name in audioset.INDEX for name in concept.audioset), concept.id
        for word in concept.words:
            assert lexicon.lookup(word).id == concept.id


def test_audioset_labels_follow_the_published_class_order() -> None:
    assert len(audioset.LABELS) == 527
    assert audioset.LABELS[0] == "Speech" and audioset.LABELS[137] == "Music"
    assert audioset.INDEX["Female singing"] == 33 and audioset.INDEX["Rapping"] == 36
    assert audioset.INDEX["Background music"] == 267


# --- 要求の読み方 ------------------------------------------------------------


def _one(text: str) -> intent.SoundRequest:
    (request,) = intent.parse(text)
    return request


def test_vocal_means_the_lead_only_and_chorus_harmony_are_the_rest() -> None:
    lead = _one("ボーカル")
    assert [a.concept.id for a in lead.atoms] == ["lead_vocals"] and lead.label == "ボーカル"
    for word in ("コーラス", "ハーモニー", "ハモリ", "バックコーラス"):
        request = _one(word)
        assert [a.concept.id for a in request.atoms] == ["backing_vocals"]
        assert request.label == "コーラス・ハーモニー"
    assert [a.concept.id for a in _one("歌声全部").atoms] == ["singing"]
    removed = _one("ボーカル抜き")
    assert removed.operation == "remove" and removed.label == "ボーカルなし"


def test_union_removal_scope_and_english_are_read() -> None:
    union = _one("ドラムとベース")
    assert [a.concept.id for a in union.atoms] == ["drums", "bass"] and union.label == "ドラムとベース"
    assert _one("drums and bass").key() == union.key()
    assert _one("台詞なし").operation == "remove"
    assert _one("without vocals").operation == "remove"
    # 本来の親と同じ範囲は範囲が無いのと同じ
    assert _one("BGMのピアノ").key() == _one("ピアノ").key() == _one("piano in the music").key()
    assert _one("音楽の雨").scope == "music"
    assert _one("雨の音").atoms[0].concept.id == "rain"
    # 語彙の音を1つ含む英語の句は、その音と同じ経路・確かめ方で、言葉は句のまま
    narrowed = _one("sci-fi laser shots").atoms[0]
    assert not narrowed.open_vocabulary and narrowed.concept.base_id == "laser"
    assert narrowed.concept.prompt == "sci-fi laser shots" and narrowed.concept.audioset == lexicon.BY_ID["laser"].audioset
    open_request = _one("squeaky rubber toy")
    assert open_request.atoms[0].open_vocabulary and open_request.atoms[0].concept.prompt == "squeaky rubber toy"


def test_unknown_japanese_is_not_guessed_into_another_sound() -> None:
    request = _one("かめはめ波")
    assert not request.supported and request.unknown == ("かめはめ波",)
    assert not _one("かめはめ波と雨").supported


def test_entries_split_on_japanese_and_ascii_separators_without_duplicates() -> None:
    assert intent.split_entries("ピアノ、雨の音,ピアノ\nドラム") == ["ピアノ", "雨の音", "ドラム"]
    assert len(intent.parse("ピアノ、piano、BGMのピアノ")) == 1
    assert len(intent.split_entries("、".join(str(i) for i in range(40)))) == intent.MAX_REQUESTS


# --- 経路 ------------------------------------------------------------------


def _route(text: str, kind: str) -> plan.Route:
    return plan.route(_one(text), kind)


def test_routes_use_specialists_and_text_separation_on_narrow_parents() -> None:
    music = plan.dnr("music")
    voices = plan.vocals("vocals", music)
    assert _route("台詞", "cinematic").expression == plan.dnr("speech")
    assert _route("ボーカル", "cinematic").expression == plan.lead(voices)
    assert _route("コーラス", "cinematic").expression == {"node": "subtract", "base": voices, "minus": plan.lead(voices)}
    drums = plan.model("sw6", "drums", music)
    assert _route("ドラム", "cinematic").expression == drums
    assert _route("ストリングス", "cinematic").expression == plan.model("bowed_strings", "strings", music)
    assert _route("スネア", "cinematic").expression == plan.model("drumsep6", "snare", drums)
    assert _route("タム", "cinematic").expression == plan.model("drumsep6", "toms", drums)
    assert _route("シンバル", "cinematic").expression == {
        "node": "sum", "items": [plan.model("drumsep6", "ride", drums), plan.model("drumsep6", "crash", drums)]}
    assert _route("オルガン", "cinematic").expression == plan.model("mega53", "organ", music)
    assert _route("管楽器", "cinematic").expression == plan.model("mega53", "wind", music)
    # 専門modelの無い楽器だけを言葉で取る
    assert _route("琴", "cinematic").expression == plan.text("koto", plan.stems("other", music), instrument="koto")
    assert _route("雨の音", "cinematic").expression == plan.text("rain", plan.dnr("effects"))
    # 曲（台詞の無い音源）では元の音が音楽の親
    assert _route("ドラム", "music").expression == plan.model("sw6", "drums", plan.SOURCE)
    assert _route("音楽", "music").representation == "source"
    removed = _route("台詞なし", "cinematic")
    assert removed.expression == {"node": "subtract", "base": plan.SOURCE, "minus": plan.dnr("speech")}
    assert removed.representation == "remainder" and removed.classes == ()


def test_route_nodes_are_listed_children_first_and_once() -> None:
    expression = _route("ドラムとベース", "cinematic").expression
    names = [plan.canonical(node) for node in plan.nodes(expression)]
    assert len(names) == len(set(names))
    drums = plan.model("sw6", "drums", plan.dnr("music"))
    assert names.index(plan.canonical(plan.dnr("music"))) < names.index(plan.canonical(drums))
    assert plan.engines(expression) == {"dnr", "model"}


def test_a_combination_of_one_models_stems_is_one_inference() -> None:
    """ベース＋ギターは同じ6分類の推論の出力の和（別のmodelの和で重なりや欠けを作らない）。"""
    expression = _route("ベースとギター", "cinematic").expression
    models = {(node["model"], plan.canonical(node["parent"])) for node in plan.nodes(expression) if node["node"] == "model"}
    assert models == {("sw6", plan.canonical(plan.dnr("music")))}


def test_every_specialist_route_names_a_lexicon_instrument_and_a_model_output() -> None:
    from lss.sound import extract, lexicon

    ids = {concept.id for concept in lexicon.CONCEPTS if concept.family == "music_part"}
    for table in (plan._SW6_PARTS, plan._DRUMSEP_PARTS, plan._STRINGS_PARTS, plan._MEGA53_PARTS):
        for base, stem in table.items():
            assert base in ids
            assert plan.specialist(base, plan.SOURCE) is not None
    assert set(plan.MODEL_STEMS) == set(extract.MODELS)
    for model_id, spec in extract.MODELS.items():
        assert "/resolve/" in spec["weights"]["url"] and len(spec["weights"]["sha256"]) == 64
        assert "/resolve/" in spec["config"]["url"] and len(spec["config"]["sha256"]) == 64


# --- 検証 ------------------------------------------------------------------


def _tags(seconds: int, **values) -> np.ndarray:
    tags = np.zeros((seconds, 527), dtype=np.float32)
    for name, (start, end, value) in values.items():
        tags[start:end, audioset.INDEX[name.replace("_", " ")]] = value
    return tags


def _tone(seconds: int, rate: int = 8000, level: float = 0.1, on=None) -> np.ndarray:
    audio = np.full((seconds * rate, 2), level, dtype=np.float32)
    if on is not None:
        mask = np.zeros(seconds, dtype=bool)
        mask[on[0]:on[1]] = True
        audio *= np.repeat(mask, rate)[:, None]
    return audio


def test_presence_needs_the_target_class_heard_in_audible_seconds() -> None:
    parent = _tone(10)
    output = _tone(10, on=(3, 7))
    present = verify.decide(output, parent, 8000, classes=("Rain",), family="sound", tags=_tags(10, Rain=(3, 7, 0.4)))
    assert present.present and present.keep[3:7].all() and not present.keep[:2].any()
    absent = verify.decide(output, parent, 8000, classes=("Rain",), family="sound", tags=_tags(10, Rain=(3, 7, 0.01)))
    assert not absent.present and not absent.keep.any() and "target_class_not_heard" in absent.reasons


def test_speech_evidence_does_not_count_where_singing_is_stronger() -> None:
    parent = _tone(10)
    output = _tone(10)
    tags = _tags(10, Speech=(0, 10, 0.6), Singing=(0, 10, 0.8))
    decision = verify.decide(output, parent, 8000, classes=("Speech",), family="speech", tags=tags)
    assert not decision.present


def test_quiet_leak_is_not_called_speech() -> None:
    parent = _tone(10, level=0.3)
    leak = _tone(10, level=0.3 * 10 ** (-30 / 20))
    decision = verify.decide(leak, parent, 8000, classes=("Speech",), family="speech", tags=_tags(10, Speech=(0, 10, 0.9)))
    assert not decision.present


def _frames(seconds: int, on=None, value: float = 0.9) -> np.ndarray:
    frames = np.full(seconds * 25, 0.01, dtype=np.float32)
    if on is not None:
        frames[int(on[0] * 25):int(on[1] * 25)] = value
    return frames


def test_words_alone_decide_presence_when_no_audioset_class_exists() -> None:
    """語彙に無い言葉: タグのクラスが無くても、言葉の時刻の証拠で「ある」を決め、40 msで残す時刻を決める。"""
    parent = _tone(10)
    output = _tone(10, on=(2, 8))
    frames = verify.Frames(output=_frames(10, on=(3.2, 4.0)), parent=_frames(10, on=(3.2, 4.0)))
    decision = verify.decide(output, parent, 8000, classes=(), family="sound", tags=None, frames=frames)
    assert decision.present and decision.outcome == "present"
    assert decision.keep_frames[int(3.5 * 25)] and not decision.keep_frames[int(7.5 * 25)]
    gated = verify.apply_keep(output, 8000, decision.keep, decision.keep_frames)
    assert np.abs(gated[int(3.5 * 8000)]).max() > 0 and np.abs(gated[int(7.5 * 8000)]).max() == 0


def test_without_words_evidence_an_audible_output_is_not_enough() -> None:
    """分離器は対象が無くても何かを返す: 言葉の証拠が無ければ、音が出ていても「ある」にしない。"""
    parent = _tone(10)
    output = _tone(10)
    frames = verify.Frames(output=_frames(10), parent=_frames(10))
    decision = verify.decide(output, parent, 8000, classes=(), family="sound", tags=None, frames=frames)
    assert not decision.present and decision.outcome == "absent" and not decision.keep.any()


def test_heard_in_the_parent_but_not_in_the_output_is_not_extracted() -> None:
    parent = _tone(10)
    output = _tone(10, level=0.01)
    frames = verify.Frames(output=_frames(10), parent=_frames(10, on=(2, 6)))
    decision = verify.decide(output, parent, 8000, classes=("Rain",), family="sound",
                             tags=_tags(10, Rain=(0, 10, 0.001)), frames=frames)
    assert not decision.present and decision.outcome == "not_extracted"
    assert "heard_in_parent_but_not_extracted" in decision.reasons


def test_either_the_words_or_the_tags_can_confirm_a_lexicon_sound() -> None:
    """語彙の音は、言葉の証拠かタグの証拠のどちらかで「ある」。どちらも無ければ「無い」。"""
    parent = _tone(10)
    output = _tone(10, on=(3, 7))
    heard = verify.Frames(output=_frames(10, on=(3, 7)), parent=_frames(10, on=(3, 7)))
    silent = verify.Frames(output=_frames(10), parent=_frames(10))
    by_words = verify.decide(output, parent, 8000, classes=("Rain",), family="sound",
                             tags=_tags(10, Rain=(3, 7, 0.001)), frames=heard)
    assert by_words.present and "target_heard_by_words" in by_words.reasons
    by_tags = verify.decide(output, parent, 8000, classes=("Rain",), family="sound",
                            tags=_tags(10, Rain=(3, 7, 0.4)), frames=silent)
    assert by_tags.present
    neither = verify.decide(output, parent, 8000, classes=("Rain",), family="sound",
                            tags=_tags(10, Rain=(3, 7, 0.001)), frames=silent)
    assert not neither.present and "target_words_not_heard" in neither.reasons


def test_a_short_sound_needs_only_four_tenths_of_a_second_of_evidence() -> None:
    parent = _tone(10)
    output = _tone(10, on=(3, 5))
    short = verify.Frames(output=_frames(10, on=(3.2, 3.6)), parent=_frames(10, on=(3.2, 3.6)))
    assert verify.decide(output, parent, 8000, classes=(), family="sound", tags=None, frames=short).present
    blip = verify.Frames(output=_frames(10, on=(3.2, 3.4)), parent=_frames(10))
    assert not verify.decide(output, parent, 8000, classes=(), family="sound", tags=None, frames=blip).present


def test_frame_mass_counts_only_confident_frames() -> None:
    assert verify.frame_mass(_frames(10, on=(2, 4), value=0.8)) == pytest.approx(1.6, abs=0.05)
    assert verify.frame_mass(_frames(10)) == 0.0


def test_keep_fades_instead_of_cutting() -> None:
    audio = _tone(4, rate=1000, level=1.0)
    keep = np.array([False, True, True, False])
    gated = verify.apply_keep(audio, 1000, keep)
    assert gated[1500, 0] == pytest.approx(1.0) and gated[200, 0] == 0.0
    edge = gated[990:1010, 0]
    assert 0.0 < edge.min() < 1.0 or 0.0 < edge.max() < 1.0


def test_source_kind_follows_how_often_speech_is_heard() -> None:
    assert verify.source_kind(_tags(100, Speech=(0, 20, 0.7))) == "cinematic"
    assert verify.source_kind(_tags(100, Speech=(0, 3, 0.7))) == "music"


# --- T5 tokenizer --------------------------------------------------------------


def _tokenizer(tmp_path: Path) -> t5_text.UnigramTokenizer:
    vocab = [["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0]]
    vocab += [[piece, score] for piece, score in (
        ("▁", -2.0), ("▁rain", -3.0), ("▁ra", -4.0), ("in", -4.0), ("▁dog", -3.0),
        ("▁bark", -5.0), ("ing", -2.0), ("▁barking", -9.0), ("s", -3.0))]
    vocab += [[f"▁filler{i}", -20.0] for i in range(1000)]
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"unk_id": 2, "vocab": vocab}}), encoding="utf-8")
    return t5_text.UnigramTokenizer(path)


def test_unigram_tokenizer_picks_the_most_likely_segmentation(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    index = {piece: i for piece, (i, _) in tokenizer.pieces.items()}
    assert tokenizer.encode("rain") == [index["▁rain"], 1]
    # "▁bark"+"ing" (-7) beats "▁barking" (-9)
    assert tokenizer.encode("Dog  barking") == [index["▁dog"], index["▁bark"], index["ing"], 1]


def test_prompts_outside_plain_english_are_refused() -> None:
    assert t5_text.normalized_prompt("  Dog   Barking ") == "dog barking"
    for bad in ("ピアノ", "", "a" * 300, "rain!"):
        with pytest.raises(t5_text.TextEncoderError):
            t5_text.normalized_prompt(bad)


# --- safetensors ---------------------------------------------------------------


def _safetensors(path: Path, tensors: dict, header_override: dict | None = None) -> Path:
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": {"float32": "F32", "float16": "F16", "int64": "I64"}[str(array.dtype)],
                        "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    header.update(header_override or {})
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))
    return path


def test_safetensors_round_trip_and_selection(tmp_path: Path) -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.array([1, 2], dtype=np.int64)
    path = _safetensors(tmp_path / "w.safetensors", {"a": a, "b": b})
    loaded = safetensors_file.load(path)
    assert np.array_equal(loaded["a"].numpy(), a) and np.array_equal(loaded["b"].numpy(), b)
    assert list(safetensors_file.load(path, names=["b"])) == ["b"]
    with pytest.raises(safetensors_file.SafetensorsError):
        safetensors_file.load(path, names=["missing"])


def test_safetensors_rejects_overlapping_or_out_of_range_tensors(tmp_path: Path) -> None:
    a = np.zeros(4, dtype=np.float32)
    overlapping = _safetensors(tmp_path / "o.safetensors", {"a": a, "b": a},
                               {"b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]}})
    with pytest.raises(safetensors_file.SafetensorsError):
        safetensors_file.read_header(overlapping)
    outside = _safetensors(tmp_path / "x.safetensors", {"a": a},
                           {"a": {"dtype": "F32", "shape": [8], "data_offsets": [0, 32]}})
    with pytest.raises(safetensors_file.SafetensorsError):
        safetensors_file.read_header(outside)


def _small_model():
    import torch

    class Small(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)
            self.norm = torch.nn.LayerNorm(2)
            # CPUで計算する補助表（重みに含まれない）
            self.register_buffer("table", torch.exp(-torch.arange(4).float() / 3.0), persistent=False)

    return Small()


def test_streamed_loading_equals_load_state_dict_and_keeps_cpu_computed_buffers(tmp_path: Path) -> None:
    import torch

    rng = np.random.default_rng(3)
    weights = {"linear.weight": rng.standard_normal((2, 3)).astype(np.float32),
               "linear.bias": rng.standard_normal(2).astype(np.float32),
               "norm.weight": rng.standard_normal(2).astype(np.float32),
               "norm.bias": rng.standard_normal(2).astype(np.float32)}
    path = _safetensors(tmp_path / "w.safetensors", weights)
    reference = _small_model()
    reference.load_state_dict(safetensors_file.load(path), strict=True)
    streamed = safetensors_file.build_empty(_small_model, device="cpu")
    assert all(parameter.device.type == "cpu" for parameter in streamed.parameters())
    safetensors_file.copy_into(streamed, path, names=list(weights))
    for name, value in reference.state_dict().items():
        assert torch.equal(streamed.state_dict()[name], value), name
    assert torch.equal(streamed.table, reference.table)


def test_streamed_loading_is_strict_about_names_and_shapes(tmp_path: Path) -> None:
    weights = {"linear.weight": np.zeros((2, 3), np.float32), "linear.bias": np.zeros(2, np.float32),
               "norm.weight": np.zeros(2, np.float32), "norm.bias": np.zeros(2, np.float32)}
    model = safetensors_file.build_empty(_small_model, device="cpu")
    missing = _safetensors(tmp_path / "m.safetensors", {k: v for k, v in weights.items() if k != "norm.bias"})
    with pytest.raises(safetensors_file.SafetensorsError):
        safetensors_file.copy_into(model, missing)
    wrong = _safetensors(tmp_path / "s.safetensors", {**weights, "linear.weight": np.zeros((3, 3), np.float32)})
    with pytest.raises(safetensors_file.SafetensorsError):
        safetensors_file.copy_into(model, wrong)


# --- Recipe ----------------------------------------------------------------


def test_wanted_sound_recipe_round_trips_and_detects_tampering() -> None:
    request = _one("ピアノ")
    revision = recipe.wanted_sound_recipe_revision(
        request={**request.identity(), "key": request.key()}, label=request.label,
        planner={"schema_version": 1}, engines={"text": "x"}, verification={"policy_id": "p"},
        input_preparation={"method": "libsndfile_direct"}, serialization={"revision": 3})
    assert revision["recipe_id"].startswith("wanted-sound-v1-")
    assert recipe.validate_recipe_revision(revision) == revision
    tampered = json.loads(json.dumps(revision))
    tampered["publication"]["stems"] = ["ドラム"]
    with pytest.raises(ValueError):
        recipe.validate_recipe_revision(tampered)
    other = recipe.wanted_sound_recipe_revision(
        request={**_one("ドラム").identity(), "key": _one("ドラム").key()}, label="ドラム",
        planner={"schema_version": 1}, engines={"text": "x"}, verification={"policy_id": "p"},
        input_preparation={"method": "libsndfile_direct"}, serialization={"revision": 3})
    assert other["recipe_id"] != revision["recipe_id"]


# --- 楽器（2026-10-14） ----------------------------------------------------------------------------


def _instrument_tags(seconds: int, **values) -> np.ndarray:
    tags = np.zeros((seconds, 527), dtype=np.float32)
    for name, value in values.items():
        tags[:, audioset.INDEX[name.replace("_", " ")]] = value
    return tags


def test_an_instrument_is_distinct_only_where_other_instruments_do_not_dominate() -> None:
    alone_piano = _instrument_tags(4, Piano=0.6)
    distinct, alone = verify.instrument_seconds(alone_piano, "piano")
    assert distinct.all() and alone.all()
    guitar_band = _instrument_tags(4, Guitar=0.7, Piano=0.05)
    distinct, alone = verify.instrument_seconds(guitar_band, "piano")
    assert not distinct.any() and not alone.any()
    both = _instrument_tags(4, Guitar=0.6, Piano=0.3)
    distinct, alone = verify.instrument_seconds(both, "piano")
    assert distinct.all() and not alone.any()          # 区別はできるが、ピアノだけではない
    # 同じ系統（ピアノとエレピ）は相手に数えない
    distinct, _ = verify.instrument_seconds(_instrument_tags(2, Piano=0.4, **{"Electric_piano": 0.9}), "piano")
    assert distinct.all()


def test_the_instrument_mask_has_a_floor_and_keeps_the_parent_where_it_plays_alone() -> None:
    from lss.sound import sam_audio

    rate = 8000
    rng = np.random.default_rng(0)
    parent = rng.standard_normal((4 * rate, 2)).astype(np.float32) * 0.1
    target = np.zeros(4 * rate, dtype=np.float32)          # 言葉の分離が何も取らなかった
    residual = parent.mean(axis=1)
    plain = sam_audio.project(parent, target, residual, rate=rate, device="cpu")
    floored = sam_audio.project(parent, target, residual, rate=rate, device="cpu", floor=0.5)
    whole = sam_audio.project(parent, target, residual, rate=rate, device="cpu", floor=0.5,
                              whole_seconds=np.array([True, False, False, True]))
    assert np.abs(plain).max() < 1e-3
    assert np.allclose(floored[rate:3 * rate], 0.5 * parent[rate:3 * rate], atol=1e-3)
    assert np.allclose(whole[rate // 4:3 * rate // 4], parent[rate // 4:3 * rate // 4], atol=1e-3)
    assert np.allclose(whole[3 * rate + rate // 4:], parent[3 * rate + rate // 4:], atol=1e-3)


def test_model_blocks_cover_every_sample_once_and_long_models_are_split() -> None:
    from lss.sound import extract

    for frames, block, overlap in [(1000, 300, 40), (999, 300, 40), (301, 300, 40), (1000, 1000, 40)]:
        total = np.zeros(frames)
        for start, stop, weights in extract.block_plan(frames, block, overlap):
            total[start:stop] += weights
        np.testing.assert_allclose(total, 1.0, atol=1e-6)
    # Mega 53stem keeps about 75 s of all its outputs per inference; one 25 min episode is many blocks.
    assert extract.block_frames(53, 48000) < 80 * 48000 < 25 * 60 * 48000


def test_a_model_stores_only_the_outputs_the_routes_ask_for(tmp_path: Path, monkeypatch) -> None:
    import soundfile as sf

    from lss.sound import extract

    source = tmp_path / "song.wav"
    sf.write(str(source), np.full((8000, 2), 0.1, dtype=np.float32), 8000, subtype="FLOAT")
    workspace = extract.Workspace(source, tmp_path / "work")
    calls = []

    def fake_msst(arch, ckpt, config, target, parent, keep=None):
        calls.append(sorted(keep))
        return {name: np.full((8000, 2), 0.01, dtype=np.float32) for name in keep}

    monkeypatch.setattr(workspace, "_msst", fake_msst)
    monkeypatch.setattr(extract, "cached_asset", lambda spec, progress=None: tmp_path / spec["file_name"])
    monkeypatch.setattr(extract, "model_config", lambda model_id, progress=None: tmp_path / "config.yaml")
    bass_guitar = {"node": "sum", "items": [plan.model("sw6", "bass", plan.SOURCE),
                                            plan.model("sw6", "guitar", plan.SOURCE)]}
    workspace.prepare([bass_guitar, plan.model("sw6", "piano", plan.SOURCE)])
    assert calls == [["bass", "guitar", "piano"]]   # one inference, three outputs kept
    assert len(list((tmp_path / "work").glob("*.wav"))) == 3   # only the asked stems (sums are made on read)
    np.testing.assert_allclose(workspace.audio(bass_guitar), 0.02, atol=1e-7)
