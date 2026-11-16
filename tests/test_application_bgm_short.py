"""The ordinary BGM Product path publishes only a positively bounded short cue."""
import hashlib
import json

from mutagen.flac import FLAC
import numpy as np
import soundfile as sf

from lss import application
from tests.test_application_bgm_candidates import RATE, _run, _source, gpu  # noqa: F401


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bounded_four_second_music_publishes_and_foreign_edits_are_restored_without_inference(tmp_path, gpu):
    source = _source(tmp_path / "episode.wav", music=(5, 9), seconds=15, noise=0)
    original_source_hash = _sha256(source)
    gpu.tagger.singing = 0.0

    first = _run(tmp_path, source)

    assert first.disposition is application.ResultDisposition.PUBLISHED, first.message
    assert len(first.outputs) == 1
    run = json.loads(first.record_path.read_text(encoding="utf-8"))
    [published] = run["published"]
    assert published["end_seconds"] - published["start_seconds"] == 4.0
    assert published["provenance"]["short_music"]["confirmed"] is True
    assert published["provenance"]["short_music"]["policy"]["revision"] == "physically-bounded-short-music-v6"
    assert published["provenance"]["short_music"]["original_mixture_authority"] == "none"
    assert published["provenance"]["short_music"]["absence_authority"] == "none"
    tags = FLAC(first.outputs[0])
    provenance = json.loads(tags["LSS_BGM_PROVENANCE"][0])
    assert provenance["short_music"]["confirmed"] is True
    assert provenance["short_music"]["span_seconds"] == [published["start_seconds"], published["end_seconds"]]
    assert _sha256(source) == original_source_hash
    decoded, rate = sf.read(str(first.outputs[0]), dtype="float64", always_2d=True)
    assert rate == RATE and decoded.shape == (4 * RATE, 2)

    tags = FLAC(first.outputs[0])
    own_title = tags["TITLE"]
    tags["TITLE"] = "他のソフトが付けた短い曲名"
    tags.save()
    separator_calls, tagger_calls = len(gpu.calls), len(gpu.tagger.calls)

    second = _run(tmp_path, source)

    assert second.disposition is application.ResultDisposition.PUBLISHED and second.reused is True
    assert second.outputs == first.outputs
    assert FLAC(second.outputs[0])["TITLE"] == own_title
    assert len(gpu.calls) == separator_calls and len(gpu.tagger.calls) == tagger_calls
    assert _sha256(source) == original_source_hash


def test_continuous_music_cut_without_quiet_boundaries_does_not_publish_as_short(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / "continuous.wav", music=(0, 15), seconds=15, noise=0)
    gpu.tagger.singing = 0.0
    # Model a separator that omits a quiet wanted continuation outside a loud
    # four-second body. Source-side Music remains positive throughout; the
    # apparent quiet edges in the separated stem cannot prove a source cue end.
    audio, rate = sf.read(str(source), dtype="float32", always_2d=True)
    t = np.arange(len(audio), dtype=np.float32) / rate
    quiet_continuation = 0.005 * np.sin(2 * np.pi * 220 * t)
    loud_body = 0.2 * np.sin(2 * np.pi * 220 * t)
    audio[:] = quiet_continuation[:, None]
    audio[5*rate:9*rate] = loud_body[5*rate:9*rate, None]
    sf.write(str(source), audio, rate, subtype="FLOAT")
    original_source_hash = _sha256(source)

    def source_music_tagger(signal_path, weights, *, device="cuda:0", progress=None):
        gpu.tagger.calls.append(signal_path)
        frames, rate = sf.info(signal_path).frames, sf.info(signal_path).samplerate
        seconds = frames // rate
        probabilities = np.zeros((seconds, 527), dtype=np.float16)
        probabilities[:, 137] = 0.9
        embeddings = np.ones((seconds, 2048), dtype=np.float16)
        return probabilities, embeddings

    # Keep all source windows Music-positive, including the quiet material the
    # fake separator drops from its music output.
    from lss.bgm import event_tagging
    monkeypatch.setattr(event_tagging, "tag_signal", source_music_tagger)

    outcome = _run(tmp_path, source)

    assert outcome.disposition is not application.ResultDisposition.PUBLISHED
    assert outcome.outputs == ()
    run = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert run["published"] == []
    assert any(not cue.get("short_music", {}).get("confirmed", False) for cue in run["held"])
    assert _sha256(source) == original_source_hash
