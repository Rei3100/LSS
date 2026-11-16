"""単話BGM候補の製品入口は技術選択を隠し、保留結果を安全に再利用する。

解析器（SFC）・歌唱回収の4分類・意味判定のタグ付けのGPU推論だけを、入力から決まる偽の出力へ置き換える。
復号・解析用複製・worker側の受領票・区間計画・候補保存・意味判定・公開・完了記録・再利用は製品の実装をそのまま
通す。タグ付けの偽物は既定で「歌のある音楽」と答え、通常のProduct既定に従って公開する。保留試験は独立した
非連続Musicタグのfixture modeを明示する。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mutagen.flac import FLAC
import numpy as np
import pytest
import soundfile as sf

from lss import application
from lss.bgm import cue_semantics, episode_cues, event_tagging, four_stem_analysis, sfc_analysis
from lss.core import audio_identity, runtime, session, worker

RATE = 44100
# Upper bound for waiting on another thread/event. It only prevents a hung test; it is not a performance claim, and
# a loaded machine (parallel validation lanes) can take far longer than 10 s to decode and analyse the episode.
HANG_GUARD_SECONDS = 120


def _source(path: Path, *, rate: int = RATE, music=(5, 25), seconds: int = 30,
            channels: int = 2, silent: bool = False, noise: float = .001) -> Path:
    frames = seconds * rate
    values = np.zeros((frames, channels), dtype=np.float32)
    if not silent:
        rng = np.random.default_rng(0)
        values += rng.normal(0, noise, values.shape).astype(np.float32)
        t = np.arange((music[1] - music[0]) * rate) / rate
        tone = 0.2 * np.sin(2 * np.pi * 220 * t)
        for channel in range(channels):
            values[music[0] * rate:music[1] * rate, channel] += tone * (1.0 if channel == 0 else 0.6)
    sf.write(path, values, rate, subtype="PCM_24")
    return path


class _FakeAnalyzer:
    """GPU推論の代わりに、解析用音声の音楽部分を音楽stemへ、残りを台詞stemへ置く。"""

    def __init__(self, *, route: str = "music", entered=None, release=None):
        self.calls: list[Path] = []
        self.route = route
        self.entered, self.release = entered, release
        # 実際のGPU推論は実行ごとに僅かに揺らぐ。やり直した解析を別の信号にする時に変える。
        self.gain = 1.0
        self.minimum_music_amplitude = .01

    def __call__(self, analysis_path, destination, weights, *, device="cuda:0", progress=None, blocks=None,
                 execution_profile=None):
        self.calls.append(Path(analysis_path))
        if self.entered is not None:
            self.entered.set()
            assert self.release.wait(HANG_GUARD_SECONDS)
        audio, rate = sf.read(str(analysis_path), dtype="float32", always_2d=True)
        loud = np.abs(audio) > self.minimum_music_amplitude
        music = (np.where(loud, audio, 0.0) * self.gain).astype(np.float32)
        rest = (audio - music).astype(np.float32)
        if self.route == "speech":
            music, rest = np.zeros_like(audio), audio
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        stems = {}
        for role, values in (("music", music), ("speech", rest), ("effects", rest * 0.5)):
            path = destination / f"{role}.wav"
            sf.write(str(path), values, rate, subtype="FLOAT")
            stems[role] = {"path": str(path), "sha256": sfc_analysis._sha256(path)}
        return {"analysis_sha256": sfc_analysis._sha256(Path(analysis_path)),
                "frames": int(audio.shape[0]), "channels": int(audio.shape[1]),
                "sample_rate": int(rate), "stems": stems, "attention_modules_patched": 12,
                "blocks": None if blocks is None else sorted(blocks)}


class _LeakAnalyzer:
    """台詞の場面で効果音の一部を音楽の出力へ入れる1回目と、それをもう一度通す2回目を真似る。

    正解の成分（曲・台詞・効果音）を知っていて、1回目の音楽の出力＝曲＋効果音の漏れ。
    2回目は ``second`` に従い、漏れを取る（clean）か、欲しい曲まで弱める（weaken）。
    """

    def __init__(self, tone, dialogue, clicks, *, second="clean"):
        self.tone, self.dialogue, self.clicks = tone, dialogue, clicks
        self.second = second
        self.calls: list[tuple[Path, object]] = []

    def __call__(self, analysis_path, destination, weights, *, device="cuda:0", progress=None, blocks=None,
                 execution_profile=None):
        self.calls.append((Path(analysis_path), blocks))
        audio, rate = sf.read(str(analysis_path), dtype="float32", always_2d=True)
        if Path(analysis_path).name != "music.wav":
            music, speech, effects = self.tone + self.clicks, self.dialogue, 0.2 * self.clicks
        else:
            music = self.tone if self.second == "clean" else 0.6 * self.tone
            speech, effects = np.zeros_like(audio), audio - music
            mask = np.zeros(audio.shape[0], dtype=bool)
            for start, end in (sfc_analysis.block_ranges(audio.shape[0], rate)[i] for i in blocks):
                mask[start:end] = True
            music, effects = music * mask[:, None], effects * mask[:, None]
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        stems = {}
        for role, values in (("music", music), ("speech", speech), ("effects", effects)):
            path = destination / f"{role}.wav"
            sf.write(str(path), np.asarray(values, dtype=np.float32), rate, subtype="FLOAT")
            stems[role] = {"path": str(path), "sha256": sfc_analysis._sha256(path)}
        return {"analysis_sha256": sfc_analysis._sha256(Path(analysis_path)),
                "frames": int(audio.shape[0]), "channels": int(audio.shape[1]),
                "sample_rate": int(rate), "stems": stems, "attention_modules_patched": 12,
                "blocks": None if blocks is None else sorted(blocks)}


def _leaky_episode(path: Path):
    """5〜25秒に小さな曲と台詞、台詞の場面の効果音（0.1秒の音が1秒おき）。"""
    frames = 30 * RATE
    t = np.arange(20 * RATE) / RATE
    tone = np.zeros((frames, 2))
    tone[5 * RATE:25 * RATE] = (0.05 * np.sin(2 * np.pi * 330 * t))[:, None]
    rng = np.random.default_rng(3)
    dialogue = np.zeros((frames, 2))
    dialogue[5 * RATE:25 * RATE] = rng.normal(0, 0.1, (20 * RATE, 1))
    clicks = np.zeros((frames, 2))
    for second in range(6, 25):
        clicks[second * RATE:second * RATE + RATE // 10] = rng.normal(0, 0.3, (RATE // 10, 1))
    sf.write(path, (tone + dialogue + clicks).astype(np.float32), RATE, subtype="FLOAT")
    return tone, dialogue, clicks


class _FakeFourStem:
    """4分類のGPU推論の代わり。``vocals(role, region)`` が区間の ``music_vocals`` を返す。

    既定は歌唱なし（0）。信号のファイル名で、効果音の出力か候補の音楽かを見分ける。
    """

    def __init__(self, vocals=None):
        self.calls: list[tuple[str, list]] = []
        self.vocals = vocals or (lambda role, region: np.zeros_like(region))

    def __call__(self, signal_path, destination, weights, spans, *, hop_seconds, roles=("music_vocals",),
                 device="cuda:0", progress=None):
        signal_path, destination = Path(signal_path), Path(destination)
        role = signal_path.stem
        self.calls.append((role, [list(span) for span in spans]))
        audio, rate = sf.read(str(signal_path), dtype="float32", always_2d=True)
        destination.mkdir(parents=True, exist_ok=True)
        items = []
        for first, last in sorted(tuple(span) for span in spans):
            path = destination / f"music_vocals-{first}-{last}.wav"
            sf.write(str(path), np.asarray(self.vocals(role, audio[first:last]), dtype=np.float32), rate,
                     subtype="FLOAT")
            items.append({"span": [first, last], "path": str(path), "sha256": sfc_analysis._sha256(path)})
        return {"signal_sha256": sfc_analysis._sha256(signal_path), "frames": int(audio.shape[0]), "channels": 2,
                "sample_rate": int(rate), "spans": [item["span"] for item in items],
                "stems": {"music_vocals": items}}


class _FakeTagger:
    """タグ付けのGPU推論の代わり。1秒窓ごとに、信号が鳴っていれば ``music``・``singing`` の確率を返す。

    埋め込みは ``embedding(second)`` が返すベクトル（既定は全窓で同じ＝曲が替わらない）。
    """

    def __init__(self, *, singing=0.6, music=0.9, embedding=None, speech=0.0, noncontinuous_music=False):
        self.singing, self.music, self.speech = singing, music, speech
        self.noncontinuous_music = noncontinuous_music
        self.embedding = embedding or (lambda second: np.ones(event_tagging.EMBEDDING_SIZE))
        self.calls: list[Path] = []

    def __call__(self, signal_path, weights, *, device="cuda:0", progress=None):
        self.calls.append(Path(signal_path))
        audio, rate = sf.read(str(signal_path), dtype="float32", always_2d=True)
        seconds = audio.shape[0] // rate
        probabilities = np.zeros((seconds, event_tagging.CLASS_COUNT), dtype=np.float16)
        embeddings = np.zeros((seconds, event_tagging.EMBEDDING_SIZE), dtype=np.float16)
        for second in range(seconds):
            if np.sqrt(np.mean(audio[second * rate:(second + 1) * rate] ** 2)) > 0.01:
                music_probability = (0.4 if self.noncontinuous_music and second % 4 == 0 else self.music)
                probabilities[second, cue_semantics.MUSIC_CLASS] = music_probability
                probabilities[second, cue_semantics.SINGING_CLASSES[0]] = self.singing
                probabilities[second, 0] = (self.speech(second) if callable(self.speech)
                                            else self.speech)  # AudioSet "Speech"
            embeddings[second] = self.embedding(second)
        return probabilities, embeddings


class _Runtime:
    def to_dict(self):
        return {"runtime_id": "runtime-test-r1", "device": "cuda:0", "provider": sfc_analysis.PROVIDER}


class _Lease:
    def close(self):
        pass


@pytest.fixture
def gpu(monkeypatch, tmp_path):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    # 参照録音の置き場所も試験ごと（既定は空。置く試験は自分で置く）。
    monkeypatch.setenv("LSS_REFERENCES_DIR", str(tmp_path / "References"))
    analyzer = _FakeAnalyzer()
    monkeypatch.setattr(sfc_analysis, "ensure_weights", lambda progress=None: tmp_path / "weights.pth")
    monkeypatch.setattr(sfc_analysis, "separate_stems", lambda *a, **k: analyzer(*a, **k))
    four_stem = _FakeFourStem()
    analyzer.four_stem = four_stem
    monkeypatch.setattr(four_stem_analysis, "ensure_weights", lambda progress=None: tmp_path / "four-stem.ckpt")
    monkeypatch.setattr(four_stem_analysis, "separate_spans", lambda *a, **k: four_stem(*a, **k))
    tagger = _FakeTagger()
    analyzer.tagger = tagger
    monkeypatch.setattr(event_tagging, "ensure_weights", lambda progress=None: tmp_path / "tagger.pth")
    monkeypatch.setattr(event_tagging, "tag_signal", lambda *a, **k: tagger(*a, **k))
    monkeypatch.setattr(runtime, "capture_cuda_runtime", lambda *a, **k: _Runtime())
    monkeypatch.setattr(worker.gpu_lease, "acquire", lambda **_k: _Lease())
    monkeypatch.setattr(worker.vram, "reset_peak_vram", lambda: None)
    monkeypatch.setattr(worker.vram, "peak_vram_mb", lambda: 0.0)
    monkeypatch.setattr(worker.vram, "release_unused_vram", lambda: None)
    return analyzer


def _audio_and_id(path: Path):
    return sf.read(str(path), dtype="float64")[0], FLAC(path)["LSS_CANDIDATE_ID"][0]


def _request(source: Path) -> application.ProcessingRequest:
    return application.ProcessingRequest((source,), application.UserIntent.EXTRACT_BGM_CUES)


def _cue_record(outcome):
    """候補づくりの記録（区間計画・2回目・歌唱回収の採否）。公開した時の根拠の記録とは別。"""
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    return json.loads((outcome.record_path.parent / run["cue_record"]["name"]).read_text(encoding="utf-8"))


def _run(tmp_path, source, **kwargs):
    return application.run_bgm_candidate_request(_request(source), state_root=tmp_path / "state", **kwargs)


def test_sfc_execution_profiles_are_recipe_and_cache_identity_inputs(monkeypatch):
    monkeypatch.delenv("LSS_DEV_SFC_EXECUTION_PROFILE", raising=False)
    legacy = application._bgm_candidate_recipe()
    assert "execution_profile" not in legacy["analyzer"]
    assert legacy["analyzer_id"] == sfc_analysis.analyzer_revision()["analyzer_id"]

    monkeypatch.setenv("LSS_DEV_SFC_EXECUTION_PROFILE", "strict_deterministic")
    strict = application._bgm_candidate_recipe()
    assert strict["analyzer"]["execution_profile"] == "strict_deterministic"

    monkeypatch.setenv("LSS_DEV_SFC_EXECUTION_PROFILE", "strict_deterministic_paced60")
    paced = application._bgm_candidate_recipe()
    assert paced["analyzer"]["execution_profile"] == "strict_deterministic_paced60"
    assert len({legacy["analyzer_id"], strict["analyzer_id"], paced["analyzer_id"]}) == 3
    assert len({legacy["recipe_id"], strict["recipe_id"], paced["recipe_id"]}) == 3
    assert len({application._bgm_candidate_run_key("a" * 64, recipe["recipe_id"])
                for recipe in (legacy, strict, paced)}) == 3


def test_sfc_analysis_receipt_cannot_cross_execution_profiles(tmp_path):
    folder = tmp_path / "sfc"
    folder.mkdir()
    stems = {}
    for name in episode_cues.STEMS:
        path = folder / f"{name}.wav"
        path.write_bytes(name.encode())
        stems[name] = {"path": str(path), "sha256": sfc_analysis._sha256(path)}
    receipt_path = folder / "receipt.json"
    receipt = {"type": "bgm_sfc_analysis_receipt", "analyzer_id": "analyzer-r1",
               "analysis": {"sha256": "a" * 64}, "blocks": None, "runtime_id": "runtime-r1",
               "stems": stems}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert application._verified_analysis(receipt_path, "a" * 64, "analyzer-r1") == receipt
    assert application._verified_analysis(receipt_path, "a" * 64, "analyzer-r1",
                                           execution_profile="strict_deterministic") is None

    receipt["execution_profile"] = "strict_deterministic_paced60"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert application._verified_analysis(receipt_path, "a" * 64, "analyzer-r1",
                                          execution_profile="strict_deterministic_paced60") == receipt
    assert application._verified_analysis(receipt_path, "a" * 64, "analyzer-r1") is None


def test_single_episode_produces_held_audio_with_source_time_and_lineage(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.noncontinuous_music = True

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.HELD, outcome.message
    assert outcome.reused is False and len(gpu.calls) == 1
    assert len(outcome.held_artifacts) == 1 and len(outcome.held_alternative_artifacts) == 1
    assert outcome.outputs == () and outcome.zero_result is None
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert run["source"]["sha256"] == session.input_sha256_of(source)
    assert run["source"]["audio_timeline_origin"]["offset_numerator"] == 0
    assert run["publication_state"] == "not_published" and run["promotion_allowed"] is False
    assert run["analysis"]["runtime_id"] == "runtime-test-r1"
    assert run["analysis"]["weights_sha256"] == sfc_analysis.WEIGHTS["sha256"]
    assert run["analysis"]["retained"] is False
    [cue] = run["held"]
    assert 3 <= cue["start_seconds"] <= 7 and 23 <= cue["end_seconds"] <= 27
    record = _cue_record(outcome)
    assert record["original"]["sha256"] == run["source"]["sha256"]
    assert record["references"]["known_recording_exclusion"] == "not_available"

    [music] = outcome.held_artifacts
    [raw] = outcome.held_alternative_artifacts
    for path, representation in ((music, "separator_music_analysis_signal"),
                                 (raw, "original_mixture_no_op")):
        provenance = json.loads(FLAC(path)["LSS_CANDIDATE_PROVENANCE"][0])
        assert provenance["source_path"] == str(source.resolve())
        assert provenance["publication_state"] == "not_published"
        assert provenance["semantic_role"] == "unresolved_music_like"
        assert provenance["evidence"]["representation"] == representation
        assert provenance["origin"]["source_sha256"] == run["source"]["sha256"]
        assert provenance["origin"]["separation_recipe_id"] == run["analysis"]["analyzer_id"]
        assert provenance["origin"]["activity_recipe_id"] == episode_cues.default_recipe().recipe_id
    provenance = json.loads(FLAC(raw)["LSS_CANDIDATE_PROVENANCE"][0])
    original, _ = sf.read(source, dtype="float64", always_2d=True)
    preserved, _ = sf.read(raw, dtype="float64", always_2d=True)
    shift = provenance["serialization"]["headroom_shift"]
    start, end = provenance["origin"]["start_frame"], provenance["origin"]["end_frame"]
    np.testing.assert_allclose(preserved * 2 ** shift, original[start:end], rtol=0, atol=2 ** -22)
    # 解析用の複製とstemは再生成できる作業物。完了後は残さない。
    assert not list((tmp_path / "state" / "work").glob("decode-bgm-analysis-*"))
    assert run["analysis_scratch"]["kept"] == [] and run["analysis_scratch"]["removed_files"] > 0


def _write_receipt(folder: Path, kind: str, names: tuple[str, ...]) -> tuple[Path, dict]:
    folder.mkdir(parents=True, exist_ok=True)
    listed = {}
    for name in names:
        path = folder / f"{name}.wav"
        path.write_bytes(f"{folder.name}/{name}".encode())
        listed[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = {"type": kind, "stems": listed}
    (folder / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return folder / "receipt.json", receipt


def _analysis_scratch(root: Path, source_sha256: str) -> tuple[Path, dict, dict, list]:
    """このrunが使った解析の作業領域：準備の記録・SFCの受領票・歌唱回収の全長の信号（区間の記録が挙げる）。"""
    scratch = root / "work" / "decode-bgm-analysis-0123456789abcdef"
    scratch.mkdir(parents=True)
    (scratch / "analysis.wav").write_bytes(b"analysis copy")
    preparation = {"source_sha256": source_sha256, "sha256": hashlib.sha256(b"analysis copy").hexdigest()}
    (scratch / "analysis-preparation.json").write_text(json.dumps(preparation), encoding="utf-8")
    receipts = [_write_receipt(scratch / "sfc", "bgm_sfc_analysis_receipt", ("music", "speech", "effects"))]
    recovered = scratch / "music-vocal-recovered.wav"
    recovered.write_bytes(b"recovered")
    cue_record = {"vocal_recovery": {"signal": {"path": str(recovered),
                                                "sha256": hashlib.sha256(b"recovered").hexdigest()}}}
    return scratch, cue_record, preparation, receipts


def test_analysis_scratch_retires_only_files_its_records_prove(tmp_path):
    """解析の作業領域で消すのは、このrunの記録が挙げ、中身が記録のsha256と一致する物とその記録だけ。"""
    source_sha256 = "a" * 64
    scratch, cue_record, preparation, receipts = _analysis_scratch(tmp_path, source_sha256)
    note = scratch / "sfc" / "notes.txt"  # 処理中に置かれた利用者の物
    note.write_text("残して", encoding="utf-8")
    lookalike = scratch / "four-stem-mixture" / "music_vocals-1-2.wav"  # 受領票の無い、LSSの名前に似た物
    lookalike.parent.mkdir()
    lookalike.write_bytes(b"user audio")
    (scratch / "sfc" / "speech.wav").write_bytes(b"changed after the receipt")  # 記録と中身が違う
    # このrunが使っていない、作業領域にあるだけの受領票（別の原本・前の試行）は所有の根拠にならない
    foreign_receipt, _ = _write_receipt(scratch / "sfc-second", "bgm_sfc_analysis_receipt", ("music",))

    report = application._retire_analysis_scratch(scratch, source_sha256, cue_record, preparation=preparation,
                                                  receipts=receipts)

    # 中身の変わった作業物（speech.wav）は、それを説明する受領票と一緒に残す
    assert report["kept"] == ["four-stem-mixture/music_vocals-1-2.wav", "sfc-second/music.wav",
                              "sfc-second/receipt.json", "sfc/notes.txt", "sfc/receipt.json", "sfc/speech.wav"]
    assert note.read_text(encoding="utf-8") == "残して" and lookalike.read_bytes() == b"user audio"
    assert (scratch / "sfc" / "speech.wav").read_bytes() == b"changed after the receipt"
    assert foreign_receipt.is_file() and (scratch / "sfc-second" / "music.wav").is_file()
    for gone in ("analysis.wav", "analysis-preparation.json", "music-vocal-recovered.wav",
                 "sfc/music.wav", "sfc/effects.wav"):
        assert not (scratch / gone).exists(), gone
    assert report["removed_files"] == 5


def test_analysis_scratch_never_retires_the_original_input(tmp_path):
    source_sha256 = "a" * 64
    scratch, cue_record, preparation, receipts = _analysis_scratch(tmp_path, source_sha256)

    report = application._retire_analysis_scratch(scratch, source_sha256, cue_record, preparation=preparation,
                                                  receipts=receipts, keep=(scratch / "analysis.wav",))

    assert report["kept"] == ["analysis-preparation.json", "analysis.wav"]
    assert (scratch / "analysis.wav").read_bytes() == b"analysis copy"
    assert not (scratch / "sfc").exists()


def test_analysis_scratch_keeps_a_record_rewritten_after_the_run_used_it(tmp_path):
    source_sha256 = "a" * 64
    scratch, cue_record, preparation, receipts = _analysis_scratch(tmp_path, source_sha256)
    [(receipt_path, _)] = receipts
    receipt_path.write_text(json.dumps({"type": "bgm_sfc_analysis_receipt", "stems": {}}), encoding="utf-8")

    report = application._retire_analysis_scratch(scratch, source_sha256, cue_record, preparation=preparation,
                                                  receipts=receipts)

    assert report["kept"] == ["sfc/receipt.json"] and receipt_path.is_file()
    assert not (scratch / "sfc" / "music.wav").exists()


def test_analysis_scratch_of_another_source_is_not_claimed_by_its_preparation_record(tmp_path):
    scratch, cue_record, preparation, receipts = _analysis_scratch(tmp_path, "b" * 64)

    report = application._retire_analysis_scratch(scratch, "a" * 64, cue_record, preparation=preparation,
                                                  receipts=receipts)

    assert (scratch / "analysis.wav").read_bytes() == b"analysis copy"
    assert (scratch / "analysis-preparation.json").is_file()
    assert sorted(report["kept"]) == ["analysis-preparation.json", "analysis.wav"]


def test_completed_same_input_is_reused_without_reanalysis(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.noncontinuous_music = True

    first = _run(tmp_path, source)
    second = _run(tmp_path, source)

    assert first.disposition is second.disposition is application.ResultDisposition.HELD
    assert second.reused is True and second.held_artifacts == first.held_artifacts
    assert len(gpu.calls) == 1


def test_raw_candidate_keeps_the_native_sample_rate(tmp_path, gpu):
    source = _source(tmp_path / "episode-48k.wav", rate=48000)
    gpu.tagger.noncontinuous_music = True

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.HELD, outcome.message
    [analysis] = gpu.calls
    [raw] = outcome.held_alternative_artifacts
    [music] = outcome.held_artifacts
    assert sf.info(str(raw)).samplerate == 48000
    assert sf.info(str(music)).samplerate == sfc_analysis.SAMPLE_RATE
    provenance = json.loads(FLAC(raw)["LSS_CANDIDATE_PROVENANCE"][0])
    assert provenance["origin"]["sample_rate"] == 48000
    record = _cue_record(outcome)
    assert record["input_preparation"]["analysis_copy"]["resample"].startswith("scipy.signal.resample_poly")
    assert record["input_preparation"]["analysis_copy"]["decoded_rate"] == 48000


@pytest.mark.parametrize("which", ["held_artifacts", "held_alternative_artifacts"])
def test_missing_candidate_is_not_reported_complete_and_is_rebuilt(tmp_path, gpu, which):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.noncontinuous_music = True
    first = _run(tmp_path, source)
    [candidate] = getattr(first, which)
    [raw] = first.held_alternative_artifacts
    raw_audio, raw_id = _audio_and_id(raw)
    candidate.unlink()

    recovered = _run(tmp_path, source)

    assert recovered.disposition is application.ResultDisposition.HELD
    assert recovered.reused is False and len(gpu.calls) == 2
    assert getattr(recovered, which) == getattr(first, which) and candidate.is_file()
    # 原本から切る候補は同じ音・同じ出典の候補として作り直される。解析信号の候補は解析を
    # やり直すので別の候補になり得るが、来歴は記録と一致する。
    audio, identifier = _audio_and_id(raw)
    assert identifier == raw_id and np.array_equal(audio, raw_audio)
    provenance = json.loads(FLAC(candidate)["LSS_CANDIDATE_PROVENANCE"][0])
    run = json.loads(recovered.record_path.read_text(encoding="utf-8"))
    assert provenance["candidate_id"] in {run["held"][0]["music"]["candidate_asset_id"],
                                          run["held"][0]["raw"]["candidate_asset_id"]}
    assert list((first.record_path.parent / "invalid-runs").glob("*.json"))


def test_damaged_candidate_is_replaced_not_reused(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.noncontinuous_music = True
    first = _run(tmp_path, source)
    [candidate] = first.held_alternative_artifacts
    expected_audio, expected_id = _audio_and_id(candidate)
    candidate.write_bytes(candidate.read_bytes()[:-100])

    recovered = _run(tmp_path, source)

    assert recovered.disposition is application.ResultDisposition.HELD
    audio, identifier = _audio_and_id(candidate)
    assert identifier == expected_id and np.array_equal(audio, expected_audio)


def test_interrupted_run_resumes_from_the_saved_analysis(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.noncontinuous_music = True
    original = episode_cues.run_episode_cues
    monkeypatch.setattr(episode_cues, "run_episode_cues",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk went away")))

    interrupted = _run(tmp_path, source)
    assert interrupted.disposition is application.ResultDisposition.FAILED

    monkeypatch.setattr(episode_cues, "run_episode_cues", original)
    resumed = _run(tmp_path, source)

    assert resumed.disposition is application.ResultDisposition.HELD, resumed.message
    assert len(gpu.calls) == 1


def test_candidates_of_an_attempt_that_never_completed_do_not_block_the_next_run(tmp_path, gpu):
    """終了・停電で完了記録の前に止まった試行の候補は、やり直した解析の候補を妨げない。

    実機の反証: 処理中に終了→再起動すると「保存先に別の候補があります」で失敗し続けた
    （FROZEN-c02c8a2 inbox-from-verifier/evidence/bgm-fresh-quit.json）。
    """
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.noncontinuous_music = True
    first = _run(tmp_path, source)
    assert first.disposition is application.ResultDisposition.HELD, first.message
    # 完了記録を書く前に止まった試行と同じ状態（候補だけ残り、解析の作業領域は無い）。
    first.record_path.unlink()
    [music] = first.held_artifacts
    _, stale_id = _audio_and_id(music)
    # 強制終了で残った候補保存の書きかけ（FLACの頭だけ書けた）。LSSの一時名と先頭のbytesは、LSSが書いた
    # 証拠にならない（利用者の物でも同じ形になりうる）ので残す。毎回別の一時名なので次の保存は妨げない。
    partial = music.parent / ".lss-candidate-killed.tmp"
    partial.write_bytes(music.read_bytes()[:4096])
    partial_bytes = partial.read_bytes()
    gpu.gain = 0.98

    # LSSの物と確かめられない物（利用者のメモ・別の原本の候補）は、完了記録が無くても消さない。
    note = music.parent / "user-owned-listening-notes.txt"
    note.write_text("2:10 のフルートが好き", encoding="utf-8")
    foreign = music.parent / "foreign-candidate.flac"
    foreign.write_bytes(music.read_bytes())
    tags = FLAC(foreign)
    tags["LSS_INPUT_SHA256"] = "0" * 64
    tags.save()
    foreign_bytes = foreign.read_bytes()
    unreadable = music.parent / "cue-099-1s-music.flac"
    unreadable.write_bytes(b"not a flac at all")

    again = _run(tmp_path, source)

    assert again.disposition is application.ResultDisposition.HELD, again.message
    assert len(gpu.calls) == 2
    _, identifier = _audio_and_id(music)
    assert identifier != stale_id
    run = json.loads(again.record_path.read_text(encoding="utf-8"))
    assert run["held"][0]["music"]["candidate_asset_id"] == identifier
    assert partial.read_bytes() == partial_bytes
    assert note.read_text(encoding="utf-8") == "2:10 のフルートが好き"
    assert foreign.read_bytes() == foreign_bytes and unreadable.read_bytes() == b"not a flac at all"
    [receipt_path] = (music.parent.parent / "interrupted-attempts").glob("*.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    removed = {row["path"]: row for row in receipt["removed"]}
    assert removed[music.name]["candidate_id"] == stale_id and removed[music.name]["kind"] == "unreferenced_candidate"
    assert partial.name not in removed
    assert sorted(receipt["untouched"]) == sorted([partial.name, note.name, foreign.name, unreadable.name])
    assert receipt["preserved"] == []


def test_a_candidate_named_by_a_retired_run_record_is_moved_aside_not_deleted(tmp_path, gpu):
    """退避した完了記録（invalid-runs）が挙げる候補は、試行をやり直しても消さずに preserved/ へ移す。"""
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.noncontinuous_music = True
    first = _run(tmp_path, source)
    [raw] = first.held_alternative_artifacts
    raw_bytes = raw.read_bytes()
    _, raw_id = _audio_and_id(raw)
    run_dir = first.record_path.parent
    archive = run_dir / "invalid-runs"
    archive.mkdir()
    shutil.copyfile(first.record_path, archive / "retired.json")
    first.record_path.unlink()

    again = _run(tmp_path, source)

    assert again.disposition is application.ResultDisposition.HELD, again.message
    [receipt_path] = (run_dir / "interrupted-attempts").glob("*.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    kept = {row["candidate_id"]: row for row in receipt["preserved"]}
    assert raw_id in kept
    assert (run_dir / kept[raw_id]["moved_to"]).read_bytes() == raw_bytes
    assert (archive / "retired.json").is_file()


def _staging_root(tmp_path: Path) -> Path:
    from lss.bgm import current_best

    return tmp_path / "LSS" / current_best.STAGING_DIR


def test_quit_before_publication_publishes_nothing_and_the_next_run_completes(tmp_path, gpu, monkeypatch):
    """凍結Design 40a: 処理中に終了した1件は保存されない。作業領域にも何も残さず、次の処理で公開まで終わる。"""
    from lss.bgm import cue_publication, current_best
    from lss.core import quitting

    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    monkeypatch.setattr(quitting, "_REQUESTED", threading.Event())
    publish = cue_publication.publish

    def publish_then_quit(*args, **kwargs):
        record = publish(*args, **kwargs)
        assert Path(record["path"]).is_file() and quitting.writing()
        quitting.request()  # 確認カードの「終了する」が公開cueの書き出し中に届く
        return record

    monkeypatch.setattr(cue_publication, "publish", publish_then_quit)
    stopped = _run(tmp_path, source)

    assert stopped.disposition is application.ResultDisposition.FAILED
    assert stopped.error_code == "job_abandoned", stopped.message
    assert not quitting.writing()
    lss_folder = tmp_path / "LSS"
    assert not _staging_root(tmp_path).exists(), list(_staging_root(tmp_path).rglob("*"))
    assert not (lss_folder / "episode - BGM").exists()
    sha = session.input_sha256_of(source)
    assert not current_best.journal_path(tmp_path / "state", sha).exists()
    assert not list((tmp_path / "state" / "bgm-candidates").glob("*/run/run.json"))

    monkeypatch.setattr(cue_publication, "publish", publish)
    monkeypatch.setattr(quitting, "_REQUESTED", threading.Event())
    resumed = _run(tmp_path, source)

    assert resumed.disposition is application.ResultDisposition.PUBLISHED, resumed.message
    [published] = resumed.outputs
    assert published.name == "episode - BGM 00m05s.flac"
    assert not _staging_root(tmp_path).exists()


def test_a_quit_after_staging_lets_the_publication_finish(tmp_path, gpu, monkeypatch):
    """作業領域へ書き終えた後の終了は、完了記録とCurrentBestの切替を最後まで運ぶ。終了処理はその間、封をしない。"""
    from lss.core import quitting

    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    monkeypatch.setattr(quitting, "_REQUESTED", threading.Event())
    monkeypatch.setattr(quitting, "_sealed", False)
    held_open = []
    switch = application._bgm_current_best

    def quit_before_the_switch(*args, **kwargs):
        quitting.request()  # 確認カードの「終了する」が、書き出しの後・切替の前に届く
        held_open.append(quitting.writing() and not quitting.seal_if_idle())
        return switch(*args, **kwargs)

    monkeypatch.setattr(application, "_bgm_current_best", quit_before_the_switch)
    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    assert held_open == [True]
    assert not quitting.writing(), "公開を運び終えたら終了処理を待たせない"
    assert not _staging_root(tmp_path).exists()


def test_nothing_is_written_into_the_users_folder_after_the_host_sealed(tmp_path, gpu, monkeypatch):
    """終了処理が「書いていない」と判断して封をした後は、利用者のフォルダへ書き始めない。"""
    from lss.core import quitting

    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    monkeypatch.setattr(quitting, "_REQUESTED", threading.Event())
    monkeypatch.setattr(quitting, "_sealed", False)
    stage = application._bgm_publish

    def sealed_just_before_staging(*args, **kwargs):
        quitting.request()
        assert quitting.seal_if_idle()
        return stage(*args, **kwargs)

    monkeypatch.setattr(application, "_bgm_publish", sealed_just_before_staging)
    outcome = _run(tmp_path, source)

    assert outcome.error_code == "job_abandoned"
    assert not (tmp_path / "LSS").exists()


def test_quit_before_any_analysis_sends_no_job_to_the_worker(tmp_path, gpu, monkeypatch):
    from lss.core import quitting

    source = _source(tmp_path / "episode.wav")
    monkeypatch.setattr(quitting, "_REQUESTED", threading.Event())
    sent = []

    def dispatch(job, on_progress=None):
        sent.append(job)
        return {"status": "error", "error_code": "worker_unavailable"}

    quitting.request()
    result = application.bgm_candidate_step(source, dispatch)

    assert result["status"] == "error" and result["error_code"] == "job_abandoned"
    assert sent == []


def test_a_staged_cue_left_by_a_killed_process_is_cleared_before_publication(tmp_path, gpu, monkeypatch):
    """公開の前で強制終了された試行の音（作業領域）は、次の処理が片付けてから書く。利用者の物には触れない。

    LSSの一時名に見える書きかけも、名前や先頭のbytesでは所有を確かめられないので残す（次の公開は妨げない）。
    """
    from lss.bgm import cue_publication

    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    publish = cue_publication.publish
    partials = []

    def publish_then_die(*args, **kwargs):
        record = publish(*args, **kwargs)
        # 試聴用の仕上げの書きかけ（FLACの頭だけ書けた）
        partial = Path(record["path"]).with_name(".lss-listening-killed.tmp")
        partial.write_bytes(Path(record["path"]).read_bytes()[:4096])
        partials.append((partial, partial.read_bytes()))
        raise KeyboardInterrupt  # processの強制終了の代わり（どのexceptにも捕まらない）

    monkeypatch.setattr(cue_publication, "publish", publish_then_die)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, source)
    staged = list(_staging_root(tmp_path).rglob("*.flac"))
    assert len(staged) == 1
    unrelated = staged[0].parent / "notes.txt"
    unrelated.write_text("利用者のメモ", encoding="utf-8")
    lookalike = staged[0].parent / ".lss-my-notes.tmp"  # LSSの一時名に似ているだけの利用者の物
    lookalike.write_text("消さないで", encoding="utf-8")
    other_source = _source(tmp_path / "other.wav", music=(3, 20))
    foreign = staged[0].parent.parent / "0123456789abcdef" / "other - BGM 00m03s.flac"
    foreign.parent.mkdir()
    foreign.write_bytes(staged[0].read_bytes())
    tags = FLAC(foreign)
    tags["LSS_INPUT_SHA256"] = session.input_sha256_of(other_source)
    tags.save()

    monkeypatch.setattr(cue_publication, "publish", publish)
    gpu.gain = 0.98
    resumed = _run(tmp_path, source)

    assert resumed.disposition is application.ResultDisposition.PUBLISHED, resumed.message
    [published] = resumed.outputs
    assert published.name == "episode - BGM 00m05s.flac"
    assert not staged[0].exists()
    [(partial, partial_bytes)] = partials
    assert partial.read_bytes() == partial_bytes
    assert unrelated.read_text(encoding="utf-8") == "利用者のメモ"
    assert lookalike.read_text(encoding="utf-8") == "消さないで"
    assert foreign.is_file()


def test_concurrent_same_episode_joins_completed_run_instead_of_double_processing(
    tmp_path, gpu, monkeypatch,
):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.noncontinuous_music = True
    entered, release = threading.Event(), threading.Event()
    slow = _FakeAnalyzer(entered=entered, release=release)
    monkeypatch.setattr(sfc_analysis, "separate_stems", lambda *a, **k: slow(*a, **k))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(_run, tmp_path, source)
        assert entered.wait(HANG_GUARD_SECONDS)
        second_future = pool.submit(_run, tmp_path, source)
        release.set()
        first = first_future.result(timeout=HANG_GUARD_SECONDS)
        second = second_future.result(timeout=HANG_GUARD_SECONDS)

    assert first.disposition is second.disposition is application.ResultDisposition.HELD
    assert first.held_artifacts == second.held_artifacts
    assert sorted((first.reused, second.reused)) == [False, True]
    assert len(slow.calls) == 1


def test_no_sustained_music_is_held_instead_of_false_zero(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / "episode.wav")
    gpu.route = "speech"

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.HELD
    assert outcome.held_artifacts == () and outcome.zero_result is None
    assert "対象なしとは確定していません" in outcome.message
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed_no_candidate"
    assert application.existing_bgm_candidates(
        source, session.input_sha256_of(source), state_root=tmp_path / "state") is None


def test_exact_silence_is_a_validated_zero_without_analysis(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav", silent=True)

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.ZERO
    assert outcome.held_artifacts == () and gpu.calls == []
    session.validate_zero_result(outcome.zero_result)
    assert application.existing_bgm_candidates(
        source, session.input_sha256_of(source), state_root=tmp_path / "state") == {"k": "none", "n": 0}
    again = _run(tmp_path, source)
    assert again.disposition is application.ResultDisposition.ZERO and again.reused is True


def test_mono_source_is_unsupported_not_failed(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav", channels=1)

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.UNSUPPORTED
    assert outcome.error_code == "unsupported_channels"
    assert gpu.calls == [] and outcome.held_artifacts == ()


def test_gpu_failure_keeps_its_actionable_error_code(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / "episode.wav")

    def unavailable(*_a, **_k):
        raise runtime.RuntimeUnavailableError("CUDAを利用できません token=SECRET")

    monkeypatch.setattr(runtime, "capture_cuda_runtime", unavailable)

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.FAILED
    assert outcome.error_code == "gpu_unavailable"
    assert "SECRET" not in outcome.message and gpu.calls == []


def test_non_bgm_or_multiple_input_request_is_rejected_before_processing(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav")
    other = _source(tmp_path / "other.wav")

    wrong = application.run_bgm_candidate_request(application.ProcessingRequest(
        (source,), application.UserIntent.EXTRACT_ACCOMPANIMENT,
    ), state_root=tmp_path / "state")
    multiple = application.run_bgm_candidate_request(application.ProcessingRequest(
        (source, other), application.UserIntent.EXTRACT_BGM_CUES,
    ), state_root=tmp_path / "state")

    assert wrong.error_code == "intent_mismatch"
    assert multiple.error_code == "bgm_requires_single_local_source"
    assert not (tmp_path / "state" / "bgm-candidates").exists()


def test_screen_step_sends_the_analysis_to_the_resident_worker_protocol(tmp_path, gpu, monkeypatch):
    """画面の工程は解析ジョブをworkerのIPC経路（cmd=bgm_analysis）で送り、その結果で候補を作る。"""
    source = _source(tmp_path / "episode.wav")
    monkeypatch.setenv("LSS_DEV_SFC_EXECUTION_PROFILE", "strict_deterministic_paced60")
    gpu.tagger.noncontinuous_music = True
    sent = []

    def dispatch(job, on_progress=None):
        sent.append(job)
        # IPC越しと同じく、workerの振り分けと保存先の解釈を通す。
        return worker._handle(job, progress=on_progress, **worker._destination_of(job))

    result = application.bgm_candidate_step(source, dispatch, on_progress=lambda _event: None)

    assert result["status"] == "ok" and result["bgm"]["k"] == "held" and result["bgm"]["n"] == 1, result
    assert result["bgm"]["product_result"]["state"] == "HELD"
    [job, tags, mixture_tags] = sent
    assert job["cmd"] == "bgm_analysis" and Path(job["destination_dir"]).is_absolute()
    assert job["execution_profile"] == "strict_deterministic_paced60"
    assert "model_id" not in job and "stem_target" not in job
    # 意味判定のタグ付けも同じworkerの経路へ送る（1回目の音楽の出力と、表現の判定のための原本の解析用複製）。
    assert tags["cmd"] == mixture_tags["cmd"] == "bgm_event_tags" and Path(tags["destination_dir"]).is_absolute()
    assert Path(tags["signal_path"]).name == "music.wav" and Path(mixture_tags["signal_path"]).name == "analysis.wav"


def test_worker_refuses_an_analysis_request_from_another_analyzer_revision(tmp_path, gpu):
    analysis = tmp_path / "analysis.wav"
    sf.write(str(analysis), np.zeros((RATE, 2), dtype=np.float32), RATE, subtype="FLOAT")
    job = {"cmd": "bgm_analysis", "analysis_path": str(analysis),
           "analysis_sha256": sfc_analysis._sha256(analysis), "analyzer_id": "bgm-sfc-analysis-v1-other"}

    result = worker.handle_bgm_analysis_job(job, destination_dir=tmp_path / "out")

    assert result["status"] == "error" and result["error_code"] == "invalid_job"
    assert gpu.calls == []


def test_worker_refuses_an_analysis_request_without_a_destination(tmp_path, gpu):
    analysis = tmp_path / "analysis.wav"
    sf.write(str(analysis), np.zeros((RATE, 2), dtype=np.float32), RATE, subtype="FLOAT")
    job = {"cmd": "bgm_analysis", "analysis_path": str(analysis),
           "analysis_sha256": sfc_analysis._sha256(analysis),
           "analyzer_id": sfc_analysis.analyzer_revision()["analyzer_id"]}

    assert worker._handle(job, **worker._destination_of(job))["error_code"] == "invalid_job"
    assert gpu.calls == []


def test_sfx_leak_is_sent_to_a_second_pass_and_the_cleaner_result_is_adopted(tmp_path, gpu, monkeypatch):
    source = tmp_path / "leaky.wav"
    # 2回目・歌唱回収は処理済みで公開する区間にだけ掛かる。公開される条件（歌が聞こえない）にする。
    gpu.tagger.singing = 0.0
    leak = _LeakAnalyzer(*_leaky_episode(source), second="clean")
    monkeypatch.setattr(sfc_analysis, "separate_stems", lambda *a, **k: leak(*a, **k))

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    assert len(leak.calls) == 2 and leak.calls[1][0].name == "music.wav" and leak.calls[1][1] == [0]
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert run["second_pass"]["status"] == "ran" and run["second_pass"]["adopted_count"] == 1
    [cue] = run["held"]
    assert cue["second_pass"] == {"eligible": True, "adopted": True}
    assert Path(cue["music_first_pass"]["path"]) in outcome.held_alternative_artifacts
    [adopted] = outcome.held_artifacts
    provenance = json.loads(FLAC(adopted)["LSS_CANDIDATE_PROVENANCE"][0])
    assert provenance["evidence"]["representation"] == "separator_music_second_pass"
    assert provenance["origin"]["separation_recipe_id"].endswith(run["second_pass"]["recipe_id"])
    record = _cue_record(outcome)
    [planned] = [c for c in record["plan"]["cues"] if c["disposition"] == "held"]
    assert planned["second_pass"]["adoption"]["adopt"] is True
    assert planned["second_pass"]["adoption"]["reasons"] == []
    # 採った音は、効果音の漏れが取れて曲だけになっている
    start = provenance["origin"]["start_frame"]
    audio, _ = sf.read(str(adopted), dtype="float64", always_2d=True)
    shift = provenance["serialization"]["headroom_shift"]
    assert np.max(np.abs(audio * 2 ** shift - leak.tone[start:start + audio.shape[0]])) < 1e-4
    # Publication must retain the successful second pass too. Restoring raw
    # against only the first pass would undo that cleanup at this boundary.
    for published in run["published"]:
        rendered = application._published_audio(run, published)
        begin = published["start_frame"]
        assert np.max(np.abs(rendered - leak.tone[begin:begin+len(rendered)])) < 1e-4
    assert not list((tmp_path / "state" / "work").glob("decode-bgm-analysis-*"))

    again = _run(tmp_path, source)
    assert again.reused is True and len(leak.calls) == 2


def test_a_second_pass_that_weakens_the_music_is_not_adopted(tmp_path, gpu, monkeypatch):
    source = tmp_path / "leaky.wav"
    # 2回目・歌唱回収は処理済みで公開する区間にだけ掛かる。公開される条件（歌が聞こえない）にする。
    gpu.tagger.singing = 0.0
    leak = _LeakAnalyzer(*_leaky_episode(source), second="weaken")
    monkeypatch.setattr(sfc_analysis, "separate_stems", lambda *a, **k: leak(*a, **k))
    monkeypatch.setenv("LSS_DEV_SFC_EXECUTION_PROFILE", "strict_deterministic")
    sent = []

    def dispatch(job, destination_dir):
        job = {**job, "destination_dir": str(destination_dir)}
        sent.append(job)
        return worker._handle(job, **worker._destination_of(job))

    outcome = _run(tmp_path, source, analyze=dispatch)

    assert outcome.record_path is not None, f"{outcome.error_code}: {outcome.message}"
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert len(leak.calls) == 2 and run["second_pass"]["adopted_count"] == 0
    analysis_jobs = [job for job in sent if job["cmd"] == "bgm_analysis"]
    assert len(analysis_jobs) == 2
    assert all(job["execution_profile"] == "strict_deterministic" for job in analysis_jobs)
    [cue] = run["held"]
    assert cue["second_pass"] == {"eligible": True, "adopted": False} and "music_first_pass" not in cue
    record = _cue_record(outcome)
    [planned] = [c for c in record["plan"]["cues"] if c["disposition"] == "held"]
    assert "music_weakened" in planned["second_pass"]["adoption"]["reasons"]
    provenance = json.loads(FLAC(outcome.held_artifacts[0])["LSS_CANDIDATE_PROVENANCE"][0])
    assert provenance["evidence"]["representation"] == "separator_music_analysis_signal"


def test_music_alone_is_not_sent_to_a_second_pass(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav")

    outcome = _run(tmp_path, source)

    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert len(gpu.calls) == 1 and run["second_pass"]["status"] == "not_needed"
    assert run["held"][0]["second_pass"] == {"eligible": False, "adopted": False}


def test_an_interrupted_run_reuses_the_saved_second_pass(tmp_path, gpu, monkeypatch):
    source = tmp_path / "leaky.wav"
    # 2回目・歌唱回収は処理済みで公開する区間にだけ掛かる。公開される条件（歌が聞こえない）にする。
    gpu.tagger.singing = 0.0
    leak = _LeakAnalyzer(*_leaky_episode(source), second="clean")
    monkeypatch.setattr(sfc_analysis, "separate_stems", lambda *a, **k: leak(*a, **k))
    original = episode_cues._materialize_cut
    monkeypatch.setattr(episode_cues, "_materialize_cut",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk went away")))

    assert _run(tmp_path, source).disposition is application.ResultDisposition.FAILED
    monkeypatch.setattr(episode_cues, "_materialize_cut", original)
    resumed = _run(tmp_path, source)

    assert resumed.disposition is application.ResultDisposition.PUBLISHED
    assert len(leak.calls) == 2


def test_the_worker_accepts_a_second_pass_job_and_refuses_bad_blocks(tmp_path, gpu):
    music = tmp_path / "music.wav"
    sf.write(str(music), np.zeros((RATE, 2), dtype=np.float32), RATE, subtype="FLOAT")
    base = {"cmd": "bgm_analysis", "analysis_path": str(music), "analysis_sha256": sfc_analysis._sha256(music),
            "analyzer_id": sfc_analysis.analyzer_revision()["analyzer_id"], "input_role": "first_pass_music"}

    ok = worker.handle_bgm_analysis_job({**base, "blocks": [0]}, destination_dir=tmp_path / "second")
    receipt = json.loads(Path(ok["receipt"]).read_text(encoding="utf-8"))
    assert ok["status"] == "ok" and receipt["blocks"] == [0] and receipt["input_role"] == "first_pass_music"
    for bad in ({**base, "blocks": []}, {**base, "blocks": [-1]}, {**base, "blocks": [True]},
                {**base, "input_role": "something_else"}):
        assert worker.handle_bgm_analysis_job(bad, destination_dir=tmp_path / "bad")["error_code"] == "invalid_job"


class _SplitSingingAnalyzer:
    """SFCが左右を別々に推論し、同じ歌唱を左はEffects、右はMusicへ入れる1回目を真似る。

    2回目（Musicをもう一度通す）は何も取り除かない。
    """

    def __init__(self):
        self.calls: list[Path] = []

    def __call__(self, analysis_path, destination, weights, *, device="cuda:0", progress=None, blocks=None):
        self.calls.append(Path(analysis_path))
        audio, rate = sf.read(str(analysis_path), dtype="float32", always_2d=True)
        zero = np.zeros_like(audio)
        if Path(analysis_path).name == "music.wav":
            music, speech, effects = audio, zero, zero
        else:
            music, speech, effects = audio * [0.0, 1.0], zero, audio * [1.0, 0.0]
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        stems = {}
        for role, values in (("music", music), ("speech", speech), ("effects", effects)):
            path = destination / f"{role}.wav"
            sf.write(str(path), np.asarray(values, dtype=np.float32), rate, subtype="FLOAT")
            stems[role] = {"path": str(path), "sha256": sfc_analysis._sha256(path)}
        return {"analysis_sha256": sfc_analysis._sha256(Path(analysis_path)),
                "frames": int(audio.shape[0]), "channels": int(audio.shape[1]),
                "sample_rate": int(rate), "stems": stems, "attention_modules_patched": 12,
                "blocks": None if blocks is None else sorted(blocks)}


def _split_singing(monkeypatch, gpu, *, music_has_singing=True):
    split = _SplitSingingAnalyzer()
    monkeypatch.setattr(sfc_analysis, "separate_stems", lambda *a, **k: split(*a, **k))
    # 4分類は効果音の出力と原本（解析用の複製）の中身を歌唱と見なす。候補の音楽の中身は
    # ``music_has_singing`` の時だけ歌唱。
    gpu.four_stem.vocals = (lambda role, region: region if role in ("effects", "analysis") or music_has_singing
                            else np.zeros_like(region))
    return split


def test_singing_split_into_effects_is_added_back_and_the_candidate_before_is_kept(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / "episode.wav")
    # 2回目・歌唱回収は処理済みで公開する区間にだけ掛かる。公開される条件（歌が聞こえない）にする。
    gpu.tagger.singing = 0.0
    _split_singing(monkeypatch, gpu)

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    # 4分類は候補区間の効果音の出力と、足す価値のある候補の原本・候補の音楽だけへ送る（全話へは送らない）
    assert [role for role, _ in gpu.four_stem.calls] == ["effects", "analysis", "music"]
    [(_, spans)] = [call for call in gpu.four_stem.calls if call[0] == "effects"]
    frames = sf.info(str(source)).frames
    assert len(spans) == 1 and spans[0][1] - spans[0][0] < frames
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    vocal = run["vocal_recovery"]
    assert vocal["status"] == "ran" and vocal["adopted_count"] == 1
    assert vocal["four_stem_analyzer_id"] == four_stem_analysis.analyzer_revision(3.0)["analyzer_id"]
    assert set(vocal["runs"]) == {"first_pass_effects", "mixture", "first_pass_music"}
    assert all(item["runtime_id"] == "runtime-test-r1" for item in vocal["runs"].values())
    [cue] = run["held"]
    assert cue["vocal_recovery"] == {"eligible": True, "adopted": True}
    before = Path(cue["music_before_vocal_recovery"]["path"])
    assert before in outcome.held_alternative_artifacts
    [adopted] = outcome.held_artifacts
    provenance = json.loads(FLAC(adopted)["LSS_CANDIDATE_PROVENANCE"][0])
    assert provenance["evidence"]["representation"] == "separator_music_vocal_recovered"
    assert vocal["recipe_id"] in provenance["origin"]["separation_recipe_id"]
    assert provenance["evidence"]["adoption"]["added_over_vocals_db"] == pytest.approx(
        10 * np.log10(1 / 0.36), abs=0.1)
    # 足した候補は原本の歌唱（左右とも）に戻り、足す前の候補は右だけのまま
    start, shift = provenance["origin"]["start_frame"], provenance["serialization"]["headroom_shift"]
    audio, _ = sf.read(str(adopted), dtype="float64", always_2d=True)
    original, _ = sf.read(str(source), dtype="float64", always_2d=True)
    original = original[start:start + audio.shape[0]]
    loud = np.abs(original) > 0.01
    np.testing.assert_allclose((audio * 2 ** shift)[loud], original[loud], atol=1e-4)
    earlier, _ = sf.read(str(before), dtype="float64", always_2d=True)
    assert np.max(np.abs(earlier[:, 0])) < 1e-6 < np.max(np.abs(earlier[:, 1]))
    record = _cue_record(outcome)
    [planned] = [c for c in record["plan"]["cues"] if c["disposition"] == "held"]
    assert planned["vocal_recovery"]["adoption"]["reasons"] == []
    assert not list((tmp_path / "state" / "work").glob("decode-bgm-analysis-*"))

    again = _run(tmp_path, source)
    assert again.reused is True and again.held_artifacts == outcome.held_artifacts
    assert len(gpu.four_stem.calls) == 3


def test_a_voice_where_the_candidate_has_no_singing_is_not_added(tmp_path, gpu, monkeypatch):
    """効果音の出力の「声」が、候補の歌唱と同じ時間に無い（候補は器楽だけ）なら足さない。"""
    source = _source(tmp_path / "episode.wav")
    # 2回目・歌唱回収は処理済みで公開する区間にだけ掛かる。公開される条件（歌が聞こえない）にする。
    gpu.tagger.singing = 0.0
    _split_singing(monkeypatch, gpu, music_has_singing=False)

    outcome = _run(tmp_path, source)
    assert outcome.record_path is not None, outcome.message
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert run["vocal_recovery"]["adopted_count"] == 0
    [cue] = run["held"]
    assert cue["vocal_recovery"] == {"eligible": True, "adopted": False}
    assert "music_before_vocal_recovery" not in cue
    record = _cue_record(outcome)
    [planned] = [c for c in record["plan"]["cues"] if c["disposition"] == "held"]
    assert "nothing_meaningful_to_add" in planned["vocal_recovery"]["adoption"]["reasons"]
    provenance = json.loads(FLAC(outcome.held_artifacts[0])["LSS_CANDIDATE_PROVENANCE"][0])
    assert provenance["evidence"]["representation"] == "separator_music_analysis_signal"


def test_music_without_competing_effects_is_not_sent_to_the_four_stem_analysis(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.noncontinuous_music = True
    split = _SplitSingingAnalyzer()
    # 1回目の効果音の出力が無音（音楽は左右とも音楽の出力へ）
    monkeypatch.setattr(sfc_analysis, "separate_stems",
                        lambda path, *a, **k: split(path, *a, **k) if Path(path).name == "music.wav"
                        else gpu(path, *a, **k))
    gpu.route = "music"

    outcome = _run(tmp_path, source)

    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert outcome.disposition is application.ResultDisposition.HELD
    assert all(cue["vocal_recovery"]["adopted"] is False for cue in run["held"])
    assert all(role != "effects" or all(b - a < sf.info(str(source)).frames for a, b in spans)
               for role, spans in gpu.four_stem.calls)


def test_an_interrupted_run_reuses_the_saved_four_stem_analysis(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / "episode.wav")
    # 2回目・歌唱回収は処理済みで公開する区間にだけ掛かる。公開される条件（歌が聞こえない）にする。
    gpu.tagger.singing = 0.0
    split = _split_singing(monkeypatch, gpu)
    original = episode_cues._materialize_cut
    monkeypatch.setattr(episode_cues, "_materialize_cut",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk went away")))

    assert _run(tmp_path, source).disposition is application.ResultDisposition.FAILED
    monkeypatch.setattr(episode_cues, "_materialize_cut", original)
    resumed = _run(tmp_path, source)

    assert resumed.disposition is application.ResultDisposition.PUBLISHED
    assert len(gpu.four_stem.calls) == 3 and len(split.calls) == 2
    run = json.loads(resumed.record_path.read_text(encoding="utf-8"))
    assert run["vocal_recovery"]["adopted_count"] == 1


def test_the_worker_accepts_a_four_stem_job_and_refuses_bad_requests(tmp_path, gpu):
    effects = tmp_path / "effects.wav"
    sf.write(str(effects), np.zeros((RATE, 2), dtype=np.float32), RATE, subtype="FLOAT")
    base = {"cmd": "bgm_four_stem", "signal_path": str(effects), "signal_sha256": sfc_analysis._sha256(effects),
            "analyzer_id": four_stem_analysis.analyzer_revision(3.0)["analyzer_id"], "spans": [[0, RATE // 2]],
            "hop_seconds": 3.0, "roles": ["music_vocals"], "input_role": "first_pass_effects"}

    ok = worker._handle(base, destination_dir=tmp_path / "four-stem")
    receipt = json.loads(Path(ok["receipt"]).read_text(encoding="utf-8"))
    assert ok["status"] == "ok" and receipt["type"] == "bgm_four_stem_receipt"
    assert receipt["spans"] == [[0, RATE // 2]] and receipt["input_role"] == "first_pass_effects"
    assert receipt["analyzer"]["weights"]["sha256"] == four_stem_analysis.WEIGHTS["sha256"]
    for bad in ({**base, "spans": []}, {**base, "spans": [[5, 1]]}, {**base, "spans": [[0, True]]},
                {**base, "hop_seconds": 0}, {**base, "roles": []},
                {**base, "analyzer_id": "bgm-four-stem-v1-other"}):
        assert worker._handle(bad, destination_dir=tmp_path / "bad")["error_code"] == "invalid_job"
    assert worker._handle(base)["error_code"] == "invalid_job"


def test_wanted_cue_is_published_next_to_the_source_with_provenance(tmp_path, gpu):
    """歌の無い、音楽の止まり目で区切られた曲は欲しい劇伴として保存し、候補はHELDのまま残す。"""
    source = _source(tmp_path / "episode.wav", noise=0.)
    gpu.tagger.singing = 0.0
    gpu.minimum_music_amplitude = 0.0

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    [published] = outcome.outputs
    assert published.parent == tmp_path / "LSS" / "episode - BGM"
    assert published.name == "episode - BGM 00m05s.flac"
    assert len(outcome.held_artifacts) == 1
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed_published_cues" and run["publication_state"] == "published_wanted_cues"
    [item] = run["published"]
    tags = FLAC(published)
    assert tags["LSS_SOURCE_TYPE"] == ["bgm_cue"] and tags["LSS_INPUT_SHA256"] == [run["source"]["sha256"]]
    assert tags["LSS_RECIPE_ID"] == [run["recipe_id"]]
    provenance = json.loads(tags["LSS_BGM_PROVENANCE"][0])
    assert provenance["semantic_class"] == "UNKNOWN_OR_UNRELEASED_WANTED_BGM" and provenance["reasons"] == []
    assert provenance["start_seconds"] == item["start_seconds"] and provenance["end_seconds"] == item["end_seconds"]
    assert provenance["identity_state"] == "unresolved_no_registered_reference"
    # 分離器が取り除く物がほぼ無い区間は原音（原本の区間、原本のsample rate）で公開する。
    assert provenance["representation"] == item["representation"] == "raw_original_mixture"
    assert provenance["route"]["state"] == "RAW" and sf.info(str(published)).samplerate == RATE
    # 公開した音は、選んだ表現の保留候補の音（縮小を戻した値）の同じ区間そのもの。
    [cue] = run["held"]
    assert cue["refined"] is False  # 原音で公開する区間は2回目・歌唱回収へ送らない
    candidate, _ = sf.read(cue["raw"]["path"], dtype="float64", always_2d=True)
    candidate /= 2.0 ** -cue["raw"]["serialization"]["headroom_shift"]
    audio, _ = sf.read(str(published), dtype="float64", always_2d=True)
    audio /= 2.0 ** -provenance["serialization"]["headroom_shift"]
    offset = item["start_frame"] - cue["raw"]["source_start_frame"]
    np.testing.assert_allclose(audio, candidate[offset:offset + audio.shape[0]], rtol=0, atol=2 ** -22)
    semantics = json.loads((outcome.record_path.parent / "semantics" / "cue-semantics.json").read_text("utf-8"))
    assert semantics["counts"]["UNKNOWN_OR_UNRELEASED_WANTED_BGM"] == 1
    assert (outcome.record_path.parent / "semantics" / "embeddings.npy").is_file()


def test_published_cue_is_reused_and_rebuilt_without_inference_when_deleted(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    first = _run(tmp_path, source)
    [published] = first.outputs
    identity = json.loads(first.record_path.read_text("utf-8"))["published"][0]["audio"]["pcm_sha256"]
    published.unlink()

    second = _run(tmp_path, source)

    assert second.disposition is application.ResultDisposition.PUBLISHED and second.reused is True
    assert len(gpu.calls) == 1 and len(gpu.tagger.calls) == 2
    [rebuilt] = second.outputs
    assert rebuilt == published and rebuilt.is_file()
    run = json.loads(second.record_path.read_text("utf-8"))
    assert run["published"][0]["audio"]["pcm_sha256"] == identity
    # 作り直しは再利用と区別して返す（UI_UX §4）。次の実行は作り直しの無い再利用。
    assert first.product_result["state"] == "PUBLISHED"
    assert second.product_result["state"] == "REBUILT" and second.product_result["rebuilt_paths"] == [str(published)]
    third = _run(tmp_path, source)
    assert third.product_result["state"] == "REUSED" and third.product_result["rebuilt_count"] == 0


def test_foreign_edits_to_a_published_cue_are_replaced_by_lss_tags_on_the_next_request(tmp_path, gpu):
    from mutagen.flac import FLAC, Picture
    from lss.output import finalizer

    source = _source(tmp_path / "episode.wav")
    [published] = _run(tmp_path, source).outputs
    provenance = json.loads(FLAC(published)["LSS_BGM_PROVENANCE"][0])
    assert provenance["features"]["singing_present"] is True
    audio = FLAC(published)
    own = {key: audio[key] for key in ("TITLE", "REPLAYGAIN_TRACK_GAIN")}
    audio["TITLE"] = "他のソフトが付けた曲名"
    audio["REPLAYGAIN_TRACK_GAIN"] = "-9.00 dB"
    cover = Picture()
    cover.type, cover.mime, cover.data = 3, "image/png", b"user-cover"
    audio.add_picture(cover)
    audio.save()
    identity = audio_identity.audio_identity(published)

    again = _run(tmp_path, source)

    assert again.disposition is application.ResultDisposition.PUBLISHED and len(gpu.calls) == 1
    audio = FLAC(published)
    assert audio["TITLE"] == own["TITLE"] and audio["REPLAYGAIN_TRACK_GAIN"] == own["REPLAYGAIN_TRACK_GAIN"]
    assert audio.pictures == []
    assert audio_identity.audio_identity(published) == identity
    assert finalizer.receipt(published) is not None
    [asset] = again.product_result["assets"]
    assert asset["title"] == own["TITLE"][0] and "title_owner" not in asset


def test_existing_lookup_reports_published_cues_only_when_they_are_intact(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    outcome = _run(tmp_path, source)
    sha = session.input_sha256_of(source)

    found = application.existing_bgm_candidates(source, sha, state_root=tmp_path / "state")
    assert found["k"] == "done" and found["n"] == 1 and found["files"] == [outcome.outputs[0].name]
    outcome.outputs[0].unlink()
    assert application.existing_bgm_candidates(source, sha, state_root=tmp_path / "state") is None


def test_screen_step_reports_published_cues_and_held_segments(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0

    def dispatch(job, on_progress=None):
        return worker._handle(job, progress=on_progress, **worker._destination_of(job))

    result = application.bgm_candidate_step(source, dispatch)

    assert result["status"] == "ok" and result["bgm"]["k"] == "done" and result["bgm"]["n"] == 1
    assert result["bgm"]["held"] == 0 and result["bgm"]["files"][0] == "episode - BGM 00m05s.flac"


# 参照録音との関係（既発売の連続した抜粋・編集・別版の候補・未知）と参照録音の置き場所は、実音声の照合で
# `tests/test_application_bgm_reference_relations.py` が確かめる。


def test_worker_accepts_an_event_tags_job_and_refuses_bad_requests(tmp_path, gpu):
    music = tmp_path / "music.wav"
    sf.write(str(music), np.zeros((3 * RATE, 2), dtype=np.float32), RATE, subtype="FLOAT")
    base = {"cmd": "bgm_event_tags", "signal_path": str(music), "signal_sha256": sfc_analysis._sha256(music),
            "analyzer_id": event_tagging.analyzer_revision()["analyzer_id"]}

    ok = worker._handle(base, destination_dir=tmp_path / "tags")
    receipt = json.loads(Path(ok["receipt"]).read_text(encoding="utf-8"))
    assert ok["status"] == "ok" and receipt["type"] == "bgm_event_tags_receipt" and receipt["windows"] == 3
    assert receipt["analyzer"]["weights"]["sha256"] == event_tagging.WEIGHTS["sha256"]
    for bad in ({**base, "analyzer_id": "bgm-event-tags-v1-other"}, {**base, "signal_path": "music.wav"}):
        assert worker._handle(bad, destination_dir=tmp_path / "bad")["error_code"] == "invalid_job"
    assert worker._handle(base)["error_code"] == "invalid_job"


def test_same_content_at_another_location_is_published_next_to_that_source(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    first = _run(tmp_path, source)
    other = tmp_path / "moved" / "copy.wav"
    other.parent.mkdir()
    other.write_bytes(source.read_bytes())

    second = _run(tmp_path, other)

    assert second.reused is True and len(gpu.calls) == 1
    [moved] = second.outputs
    assert moved.parent == tmp_path / "moved" / "LSS" / "copy - BGM" and moved.name == "copy - BGM 00m05s.flac"
    assert first.outputs[0].is_file()
    assert FLAC(moved)["TITLE"] == FLAC(first.outputs[0])["TITLE"]
    assert "仮題" in FLAC(moved)["TITLE"][0]


def test_a_failed_republish_does_not_discard_the_verified_run(tmp_path, gpu, monkeypatch):
    from lss.bgm import cue_publication

    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    first = _run(tmp_path, source)
    first.outputs[0].unlink()
    monkeypatch.setattr(cue_publication, "publish",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("folder is read-only")))

    failed = _run(tmp_path, source)

    assert failed.disposition is application.ResultDisposition.FAILED
    assert first.record_path.is_file() and all(path.is_file() for path in first.held_artifacts)
    monkeypatch.undo()
    gpu.tagger.singing = 0.0
    again = _run(tmp_path, source)
    assert again.disposition is application.ResultDisposition.PUBLISHED and again.reused is True
    assert len(gpu.calls) == 1


def test_a_cue_with_dialogue_and_effects_is_published_from_the_refined_music(tmp_path, gpu, monkeypatch):
    """台詞・効果音が重なる区間は処理済み（2回目を採った音）で公開し、原音にしない。"""
    source = tmp_path / "leaky.wav"
    leak = _LeakAnalyzer(*_leaky_episode(source), second="clean")
    monkeypatch.setattr(sfc_analysis, "separate_stems", lambda *a, **k: leak(*a, **k))
    gpu.tagger.singing = 0.0

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    [item] = run["published"]
    assert item["representation"] == "sfc_music_refined" and item["route_state"] == "PROCESSED"
    [cue] = run["held"]
    assert cue["refined"] is True and cue["second_pass"]["adopted"] is True
    provenance = json.loads(FLAC(outcome.outputs[0])["LSS_BGM_PROVENANCE"][0])
    assert "separator_removes_material" in provenance["route"]["reasons"]
    assert run["cost"]["heavy_input_seconds"]["sfc_second_pass"] > 0
    assert run["semantics"]["routes"]["PROCESSED"]["count"] == 1


def test_held_only_candidates_are_not_refined(tmp_path, gpu, monkeypatch):
    """Music分類が途切れる保留候補は、2回目・歌唱回収へ送らない。"""
    source = tmp_path / "leaky.wav"
    gpu.tagger.noncontinuous_music = True
    leak = _LeakAnalyzer(*_leaky_episode(source), second="clean")
    monkeypatch.setattr(sfc_analysis, "separate_stems", lambda *a, **k: leak(*a, **k))

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.HELD, outcome.message
    assert len(leak.calls) == 1  # 1回目だけ
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert all(cue["refined"] is False for cue in run["held"])
    assert run["cost"]["heavy_input_seconds"]["sfc_second_pass"] == 0 and run["cost"]["refined_candidate_seconds"] == 0
    assert gpu.four_stem.calls == []


class _FakeInstrumental:
    """伴奏の既定modelのGPU推論の代わり。入れた区間の振幅を半分にして返す（合成が効いたかが分かる）。"""

    def __init__(self):
        self.calls: list[int] = []

    def __call__(self, arch, mix_path, ckpt_path, config_path, stem_target, store_dir, **_kwargs):
        crop, rate = sf.read(str(mix_path), dtype="float32", always_2d=True)
        self.calls.append(int(crop.shape[0]))
        return 0.5 * crop.T, rate


@pytest.fixture
def instrumental(monkeypatch, tmp_path):
    from lss.core import models_store
    from lss.harness import infer

    fake = _FakeInstrumental()
    for name in ("inst.ckpt", "inst.yaml"):
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(models_store, "ensure_model_cached",
                        lambda *a, **k: (tmp_path / "inst.ckpt", tmp_path / "inst.yaml"))
    monkeypatch.setattr(models_store, "model_sha256", lambda *a, **k: "inst-sha")
    monkeypatch.setattr(infer, "resolve_target_filename", lambda config, target: "Instrumental")
    monkeypatch.setattr(infer, "capture_msst_runtime", lambda **k: _Runtime())
    monkeypatch.setattr(infer, "msst_separate", lambda *a, **k: fake(*a, **k))
    return fake


def _leaky_published(tmp_path, gpu, monkeypatch, *, speech):
    source = tmp_path / "leaky.wav"
    leak = _LeakAnalyzer(*_leaky_episode(source), second="clean")
    monkeypatch.setattr(sfc_analysis, "separate_stems", lambda *a, **k: leak(*a, **k))
    gpu.tagger.singing, gpu.tagger.speech = 0.0, speech
    return source, _run(tmp_path, source)


def test_dialogue_left_in_a_processed_cue_is_removed_with_the_instrumental(tmp_path, gpu, monkeypatch, instrumental):
    """SFCの音楽の出力に台詞が残る処理済みcueは、伴奏との共通成分で公開し、何をしたかを来歴へ残す。"""
    from lss.bgm import cue_publication, dialogue_residual

    source, outcome = _leaky_published(tmp_path, gpu, monkeypatch, speech=0.6)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    assert len(instrumental.calls) == 1  # 公開cueの区間（と文脈）だけ
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    [item] = run["published"]
    assert item["representation"] == "sfc_music_refined"
    residual = item["dialogue_residual"]
    assert residual["eligible"] is True and residual["applied"] is True
    assert residual["speech_on_music"] == pytest.approx(0.6, abs=1e-3)
    assert residual["instrumental"]["model_id"] == application._bgm_candidate_recipe()["dialogue_residual_model"]["model_id"]
    provenance = json.loads(FLAC(outcome.outputs[0])["LSS_BGM_PROVENANCE"][0])
    assert provenance["dialogue_residual"]["applied"] is True and provenance["dialogue_residual"]["model_sha256"] == "inst-sha"
    receipt = json.loads((outcome.record_path.parent / "dialogue-residual" / "receipt.json").read_text("utf-8"))
    assert receipt["type"] == "bgm_instrumental_spans_receipt" and receipt["spans"] == [[item["start_frame"], item["end_frame"]]]
    # 公開した音は、処理済みの音と保存した伴奏の共通成分そのもの。
    [cue] = run["held"]
    composed = cue_publication.compose([cue["music"]], item["start_frame"], item["end_frame"])
    assert residual['scope'] == 'local' and residual['transition'] == 'inside_cosine_shared_v1'
    expected = dialogue_residual.region_consensus(composed, dialogue_residual.read_span(residual["instrumental"]),
                                                         residual['regions'], RATE, dialogue_residual.default_recipe())
    audio, _ = sf.read(str(outcome.outputs[0]), dtype="float64", always_2d=True)
    audio /= 2.0 ** -provenance["serialization"]["headroom_shift"]
    np.testing.assert_allclose(audio, expected, rtol=0, atol=2 ** -20)
    assert np.sum(audio ** 2) < 0.5 * np.sum(composed ** 2)  # 伴奏（半分）の側が選ばれた


def test_a_deleted_dialogue_cleaned_cue_is_rebuilt_without_inference(tmp_path, gpu, monkeypatch, instrumental):
    source, first = _leaky_published(tmp_path, gpu, monkeypatch, speech=0.6)
    identity = json.loads(first.record_path.read_text("utf-8"))["published"][0]["audio"]["pcm_sha256"]
    first.outputs[0].unlink()

    second = _run(tmp_path, source)

    assert second.disposition is application.ResultDisposition.PUBLISHED and second.reused is True
    assert len(instrumental.calls) == 1
    assert json.loads(second.record_path.read_text("utf-8"))["published"][0]["audio"]["pcm_sha256"] == identity


def test_a_missing_required_regional_witness_fails_without_publishing_partial_music(tmp_path, gpu, monkeypatch):
    """A missing repair result cannot silently shorten or publish a wanted cue."""
    monkeypatch.setattr(application, "_bgm_dialogue_residual", lambda *a, **k: {})
    source, outcome = _leaky_published(tmp_path, gpu, monkeypatch, speech=_every_fifth_second)
    assert outcome.disposition is application.ResultDisposition.FAILED
    assert outcome.outputs == ()
    assert not list((source.parent / "LSS").rglob("*.flac"))


def _every_fifth_second(second):
    return 0.6 if second % 5 == 2 else 0.0


def test_dialogue_windows_of_a_low_speech_cue_are_cleaned_only_in_those_windows(tmp_path, gpu, monkeypatch,
                                                                                instrumental):
    """cue平均では対象外でも、台詞の残る1秒窓の時間だけ共通成分にし、その外の音は処理済みのまま（v10）。"""
    from lss.bgm import cue_publication, dialogue_residual

    source, outcome = _leaky_published(tmp_path, gpu, monkeypatch, speech=_every_fifth_second)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    assert len(instrumental.calls) == 1
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    [item] = run["published"]
    residual = item["dialogue_residual"]
    assert residual["eligible"] is False and residual["applied"] is True and residual["scope"] == "local"
    rate = item["sample_rate"]
    assert residual["regions"] and all((item["start_frame"] + low) // rate % 5 == 2 and high - low <= rate
                                       for low, high in residual["regions"])
    provenance = json.loads(FLAC(outcome.outputs[0])["LSS_BGM_PROVENANCE"][0])
    assert provenance["dialogue_residual"]["scope"] == "local"
    assert provenance["dialogue_residual"]["regions"] == residual["regions"]
    [cue] = run["held"]
    composed = cue_publication.compose([cue["music"]], item["start_frame"], item["end_frame"])
    recipe = dialogue_residual.default_recipe()
    expected = dialogue_residual.region_consensus(composed, dialogue_residual.read_span(residual["instrumental"]),
                                                         residual["regions"], rate, recipe)
    audio, _ = sf.read(str(outcome.outputs[0]), dtype="float64", always_2d=True)
    audio /= 2.0 ** -provenance["serialization"]["headroom_shift"]
    np.testing.assert_allclose(audio, expected, rtol=0, atol=2 ** -20)
    untouched = dialogue_residual.local_weight(len(composed), residual["regions"], rate, recipe.local_ramp_seconds) == 0
    assert untouched.any() and not untouched.all()
    np.testing.assert_allclose(audio[untouched], composed[untouched], rtol=0, atol=2 ** -20)

    outcome.outputs[0].unlink()
    again = _run(tmp_path, source)
    assert again.disposition is application.ResultDisposition.PUBLISHED and again.reused is True
    assert len(instrumental.calls) == 1  # 作り直しに推論しない
    assert (json.loads(again.record_path.read_text("utf-8"))["published"][0]["audio"]["pcm_sha256"]
            == item["audio"]["pcm_sha256"])


def test_a_processed_cue_without_dialogue_is_not_sent_to_the_instrumental(tmp_path, gpu, monkeypatch, instrumental):
    _, outcome = _leaky_published(tmp_path, gpu, monkeypatch, speech=0.1)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    assert instrumental.calls == []
    [item] = json.loads(outcome.record_path.read_text("utf-8"))["published"]
    assert item["dialogue_residual"]["eligible"] is False and item["dialogue_residual"]["applied"] is False
    assert not (outcome.record_path.parent / "dialogue-residual").exists()


def test_a_raw_cue_is_never_sent_to_the_instrumental(tmp_path, gpu, instrumental):
    source = _source(tmp_path / "episode.wav", noise=0.)
    gpu.tagger.singing = 0.0
    gpu.minimum_music_amplitude = 0.0

    outcome = _run(tmp_path, source)

    [item] = json.loads(outcome.record_path.read_text("utf-8"))["published"]
    assert item["representation"] == "raw_original_mixture" and item["dialogue_residual"]["state"] == "not_processed"
    assert instrumental.calls == []


def test_coarse_raw_cannot_restore_removed_material_but_can_skip_more_processing(tmp_path,gpu,instrumental):
    source = _source(tmp_path/'episode.wav')
    gpu.tagger.singing = 0.
    outcome = _run(tmp_path,source)
    assert outcome.disposition is application.ResultDisposition.PUBLISHED,outcome.message
    [item] = json.loads(outcome.record_path.read_text('utf8'))['published']
    assert item['representation'] == 'sfc_music_refined'
    provenance=json.loads(FLAC(item['path'])['LSS_BGM_PROVENANCE'][0])
    assert provenance['route']['original_check']['state'] == 'retain_separated_music'
    assert 'original_samples_not_established' in provenance['route']['reasons']
    assert not item['dialogue_residual']['applied']
    assert instrumental.calls == []


def test_a_processed_cue_restores_clean_original_islands_and_replays_them(tmp_path, gpu, monkeypatch):
    """One dirty second cannot force the remaining clean music through processing."""
    from lss.bgm import cue_publication
    from lss.separation import selective

    source = tmp_path/'regional.wav'
    t = np.arange(30*RATE)/RATE
    target = np.zeros((len(t),2),dtype='float32')
    target[5*RATE:25*RATE] = (.2*np.sin(2*np.pi*220*t[5*RATE:25*RATE]))[:,None]
    mix = target.copy()
    mix[12*RATE:13*RATE] += np.random.default_rng(4).normal(0,.2,(RATE,2)).astype('float32')
    sf.write(source,mix,RATE,subtype='FLOAT')
    gpu.tagger.singing = 0.

    def separated(analysis, destination, weights, **kwargs):
        receipt = gpu(analysis,destination,weights,**kwargs)
        # Exact toy components define clean/dirty truth independently of the
        # routing algorithm. The real provider is tested in the packaged lane.
        observed,_ = sf.read(analysis,dtype='float32',always_2d=True)
        values = target + .0005*np.gradient(target, axis=0)
        for role,data in [('music',values),('speech',observed-values),('effects',np.zeros_like(values))]:
            path=Path(receipt['stems'][role]['path'])
            sf.write(path,data,RATE,subtype='FLOAT')
            receipt['stems'][role]['sha256']=sfc_analysis._sha256(path)
        return receipt

    monkeypatch.setattr(sfc_analysis,'separate_stems',separated)
    outcome = _run(tmp_path,source)
    assert outcome.disposition is application.ResultDisposition.PUBLISHED,outcome.message
    run=json.loads(outcome.record_path.read_text('utf-8'))
    [cue]=run['published']
    receipt=cue['selective_composition']
    assert receipt and receipt['plan']['original_regions']
    assert not receipt['native_source_pcm_claim']
    audio=application._published_audio(run,cue)
    original=selective.aligned_read(source,RATE,cue['start_frame'],len(audio))
    w=selective.weight(len(audio),receipt['plan']['original_regions'],RATE)
    assert np.array_equal(audio[w==1],original[w==1])
    assert not any(low <= 12*RATE-cue['start_frame'] < high for low,high in receipt['plan']['original_regions'])
    published=cue['audio']['pcm_sha256']
    outcome.outputs[0].unlink()
    count=len(gpu.calls)
    restored=_run(tmp_path,source)
    assert restored.reused and len(gpu.calls)==count
    assert json.loads(restored.record_path.read_text('utf-8'))['published'][0]['audio']['pcm_sha256']==published


def test_the_worker_refuses_bad_instrumental_span_jobs(tmp_path, gpu, instrumental):
    from lss.bgm import dialogue_residual

    signal = _source(tmp_path / "analysis.wav")
    base = {"cmd": "bgm_instrumental_spans", "signal_path": str(signal),
            "signal_sha256": sfc_analysis._sha256(signal), "model_id": application._bgm_candidate_recipe()[
                "dialogue_residual_model"]["model_id"],
            "recipe_id": dialogue_residual.default_recipe().recipe_id, "spans": [[RATE, 3 * RATE]]}
    ok = worker.handle_bgm_instrumental_spans_job(base, destination_dir=tmp_path / "ok")
    assert ok["status"] == "ok", ok
    for bad in ({"spans": [[3 * RATE, RATE]]}, {"recipe_id": "other"}, {"signal_sha256": "0" * 64},
                {"model_id": "unwa_resurrection_inst"}):
        result = worker.handle_bgm_instrumental_spans_job({**base, **bad}, destination_dir=tmp_path / "bad")
        assert result["status"] == "error", bad
    assert worker.handle_bgm_instrumental_spans_job(base, destination_dir=None)["status"] == "error"


def _changing_embedding(distance: float, at: int = 15):
    """``at`` 秒で編成が変わる曲の埋め込み。前後の平均の距離（1−cos）が ``distance`` になる。"""
    first = np.zeros(event_tagging.EMBEDDING_SIZE)
    first[0] = 1.0
    second = np.zeros(event_tagging.EMBEDDING_SIZE)
    second[0], second[1] = 1.0 - distance, np.sqrt(1.0 - (1.0 - distance) ** 2)
    return lambda s: first if s < at else second


def test_a_cue_whose_extent_is_unresolved_is_published_and_marked(tmp_path, gpu):
    """欲しい劇伴の条件を全て満たし、内側に替わり目の疑いがあるだけの区間は、未確定と明示して公開する。"""
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    gpu.tagger.embedding = _changing_embedding(0.15)  # 疑い0.13以上・分ける0.19未満

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    assert "一曲かどうか未確定" in outcome.message
    [published] = outcome.outputs
    assert published.name == "episode - BGM 00m05s (区切り未確定).flac"
    tags = FLAC(published)
    assert "仮題" in tags["TITLE"][0] and tags["TITLE"][0].endswith(" (区切り未確定)")
    provenance = json.loads(tags["LSS_BGM_PROVENANCE"][0])
    assert provenance["semantic_class"] == "UNCERTAIN" and provenance["reasons"] == ["possible_change_inside"]
    assert provenance["cue_extent"]["state"] == "unresolved"
    assert provenance["supported_scope"]["disposition"] == "PUBLISHED"
    assert provenance["supported_scope"]["integrity"]["integrity_passed"] == 1.0
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert run["published"][0]["cue_extent"]["state"] == "unresolved"
    assert provenance["supported_scope_id"] == run["semantics"]["supported_scope_id"]
    semantics = json.loads((outcome.record_path.parent / "semantics" / "cue-semantics.json").read_text("utf-8"))
    assert semantics["held_count"] == 0 and semantics["supported_scope"]["disposition"] == "PUBLISHED"
    assert semantics["counts"]["UNCERTAIN"] == 1  # 意味判定の観測は変えない（公開の判断だけが対応領域）


def test_a_marked_cue_is_rebuilt_under_the_same_name_without_inference(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    gpu.tagger.embedding = _changing_embedding(0.15)
    first = _run(tmp_path, source)
    [published] = first.outputs
    title_before = FLAC(published)["TITLE"]
    published.unlink()

    second = _run(tmp_path, source)

    assert second.reused is True and len(gpu.calls) == 1
    assert second.outputs == (published,) and published.is_file()
    assert FLAC(published)["TITLE"] == title_before


def test_a_clear_change_is_still_split_and_not_marked(tmp_path, gpu):
    """強い替わり目は今までどおり分ける。分けた区間は一曲として公開する（印を付けない）。"""
    source = _source(tmp_path / "episode.wav", music=(5, 29), seconds=34)
    gpu.tagger.singing = 0.0
    gpu.tagger.embedding = _changing_embedding(0.3, at=17)

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    assert len(outcome.outputs) == 2 and not any("区切り未確定" in path.name for path in outcome.outputs)


def test_sung_music_is_published_and_singing_observation_survives_provenance(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav", noise=0.)
    gpu.minimum_music_amplitude = 0.0

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    [published] = outcome.outputs
    provenance = json.loads(FLAC(published)["LSS_BGM_PROVENANCE"][0])
    assert provenance["features"]["singing_present"] is True
    assert provenance["features"]["singing_fraction"] > 0.1
    semantics = json.loads((outcome.record_path.parent / "semantics" / "cue-semantics.json").read_text("utf-8"))
    assert semantics["supported_scope"]["disposition"] == "PUBLISHED"


def test_a_cue_whose_written_audio_lacks_integrity_is_not_published(tmp_path, gpu, monkeypatch):
    from lss.bgm import cue_scope
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    real = cue_scope.integrity
    monkeypatch.setattr(cue_scope, "integrity",
                        lambda audio, rate: {**real(audio, rate), "integrity_passed": 0.0, "silent": True})

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.HELD and outcome.outputs == ()
    semantics = json.loads((outcome.record_path.parent / "semantics" / "cue-semantics.json").read_text("utf-8"))
    [row] = semantics["supported_scope"]["segments"]
    assert (row["disposition"], row["reason"]) == ("HELD", "below_quality_gate")
    [segment] = semantics["segments"]
    assert segment["scope"]["integrity"]["silent"] is True and segment["route"]["not_published"]
    # r4の約束：保存した音声の無い結果のために利用者のフォルダを作らない（結果は状態として返す）。
    assert not (tmp_path / "LSS").exists()
    assert outcome.product_result["state"] == "HELD" and "report_path" not in outcome.product_result


def test_screen_step_counts_only_unpublished_held_segments(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.singing = 0.0
    gpu.tagger.embedding = _changing_embedding(0.15)

    def dispatch(job, on_progress=None):
        return worker._handle(job, progress=on_progress, **worker._destination_of(job))

    result = application.bgm_candidate_step(source, dispatch)

    assert result["bgm"]["k"] == "done" and result["bgm"]["n"] == 1 and result["bgm"]["held"] == 0
    assert result["bgm"]["files"][0].endswith(" (区切り未確定).flac")
    found = application.existing_bgm_candidates(source, session.input_sha256_of(source),
                                                state_root=tmp_path / "state")
    assert found["held"] == 0 and found["files"] == result["bgm"]["files"]
