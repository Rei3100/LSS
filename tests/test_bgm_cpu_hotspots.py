"""CPU hotspot removals of the BGM job keep every decision, record and output (`cloud-reports/cpu-hotspots.md`).

Each optimisation is pinned here against the behaviour it replaces: the work it skips is work whose answer is
already known, and a real change is still seen.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lss import application
from lss.bgm import listening
from tests import test_bgm_overlap as overlap_proof

gpu = overlap_proof.gpu
instrumental = overlap_proof.instrumental
slow_gpu = overlap_proof.slow_gpu


@pytest.fixture
def fast_gpu(monkeypatch):
    monkeypatch.setattr(overlap_proof, "GPU_SECONDS", 0.0)


def _source_reads(monkeypatch, tmp_path):
    """Full reads of the episode source while the published cues are finished for listening."""
    source = (tmp_path / "case" / "media" / "episode.wav").resolve()
    reads, finishing = [], []
    hashed = application.file_sha256

    def counted(path, *args, **kwargs):
        if finishing and Path(path).resolve() == source:
            reads.append(Path(path))
        return hashed(path, *args, **kwargs)

    finish = listening.finish_outputs

    def observed(*args, **kwargs):
        finishing.append(True)
        try:
            return finish(*args, **kwargs)
        finally:
            finishing.clear()

    monkeypatch.setattr(application, "file_sha256", counted)
    monkeypatch.setattr(listening, "finish_outputs", observed)
    return source, reads, finishing


def test_an_unchanged_source_is_read_once_while_the_cues_are_finished(tmp_path, fast_gpu, slow_gpu, monkeypatch):
    source, reads, _ = _source_reads(monkeypatch, tmp_path)
    case = overlap_proof._run_case(tmp_path, "run", monkeypatch, overlapped=False)
    assert case.outcome.disposition is application.ResultDisposition.PUBLISHED, case.outcome.message
    assert len(case.outcome.outputs) >= 3
    # Before: one full read per finished cue (3 here, 20 on a 24-minute episode).
    assert len(reads) == 1


@pytest.mark.parametrize("change", ["same_size_rewrite", "append"])
def test_a_source_rewritten_while_the_cues_are_finished_still_stops_the_job(tmp_path, fast_gpu, slow_gpu, monkeypatch,
                                                                              change):
    source, reads, finishing = _source_reads(monkeypatch, tmp_path)
    finish = listening.finish
    done = []

    def rewrite_after_first(path, *args, **kwargs):
        result = finish(path, *args, **kwargs)
        if finishing and not done:
            done.append(path)
            data = bytearray(source.read_bytes())
            if change == "append":
                data += b"\0" * 8
            else:
                data[-8:] = bytes(8 - i for i in range(8))
            source.write_bytes(bytes(data))
        return result

    monkeypatch.setattr(listening, "finish", rewrite_after_first)
    case = overlap_proof._run_case(tmp_path, "run", monkeypatch, overlapped=False)
    assert done
    assert case.outcome.disposition is application.ResultDisposition.FAILED, case.outcome.message
    assert "聴取用仕上げの間に原本が変更されました" in case.outcome.message, case.outcome.message
    assert len(reads) == 2  # the matching read before the first cue, the read that saw the change


@pytest.mark.parametrize("subtype,bits", [("PCM_S8", 8), ("PCM_16", 16), ("PCM_24", 24)])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("seed", range(4))
def test_the_int32_hash_derived_from_the_float_decode_is_libsndfiles_own(tmp_path, subtype, bits, channels, seed):
    """`audio_identity` derives the int32 PCM hash `flac_frames.stream_is_exact` compares with ffmpeg from its own
    float64 decode. It must equal the separate libsndfile int32 decode it replaces, sample for sample."""
    import numpy as np
    import soundfile as sf

    from lss.core import audio_identity, flac_frames

    rng = np.random.default_rng(seed)
    frames = int(rng.integers(1, 200_000))
    ints = rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), size=(frames, channels), dtype=np.int64)
    ints[: min(frames, 3)] = [[-(1 << (bits - 1))], [(1 << (bits - 1)) - 1], [0]][: min(frames, 3)]
    if seed == 3:
        ints[:] = 0  # digital silence
    path = tmp_path / "x.flac"
    sf.write(path, (ints << (32 - bits)).astype(np.int32), 44100, subtype=subtype)
    audio_identity._DERIVED_S32.value = None
    identity = audio_identity._decoded_identity(path)
    derived_path, derived = audio_identity._DERIVED_S32.value
    assert derived_path == str(path) and derived == flac_frames._libsndfile_pcm_sha256(path)
    assert identity["frames"] == frames
    expected = flac_frames.stream_is_exact(path)
    assert flac_frames.stream_is_exact(path, libsndfile_s32_sha256=derived) is expected


def test_no_int32_hash_is_derived_for_other_containers(tmp_path):
    import numpy as np
    import soundfile as sf

    from lss.core import audio_identity

    path = tmp_path / "x.wav"
    sf.write(path, np.zeros((1000, 2), np.float32), 44100, subtype="FLOAT")
    audio_identity._DERIVED_S32.value = None
    audio_identity._decoded_identity(path)
    assert audio_identity._DERIVED_S32.value == (str(path), None)


def test_a_float_value_that_is_no_exact_int32_falls_back_to_the_separate_decode(tmp_path, monkeypatch):
    """If a decoder ever returned a value that is no left-aligned integer sample, nothing is derived and
    `stream_is_exact` decodes the int32 stream itself, exactly as before."""
    import numpy as np
    import soundfile as sf

    from lss.core import audio_identity, flac_frames

    path = tmp_path / "x.flac"
    sf.write(path, np.zeros((5000, 2), np.int32), 44100, subtype="PCM_24")
    real = sf.SoundFile.read

    def odd(self, *args, **kwargs):
        block = real(self, *args, **kwargs)
        return block + 2.0 ** -40 if block.size else block

    monkeypatch.setattr(sf.SoundFile, "read", odd)
    audio_identity._DERIVED_S32.value = None
    audio_identity._decoded_identity(path)
    assert audio_identity._DERIVED_S32.value == (str(path), None)
    monkeypatch.undo()
    separate = []
    monkeypatch.setattr(flac_frames, "_libsndfile_pcm_sha256", lambda p: separate.append(p) or None)
    audio_identity._IDENTITIES.clear()
    monkeypatch.setattr(audio_identity, "_decoded_identity",
                        lambda p: (audio_identity._DERIVED_S32.__setattr__("value", (str(p), None)),
                                   {"container": "flac"})[1])
    with pytest.raises(audio_identity.AudioIdentityError):
        audio_identity.audio_identity(path)
    assert separate == [path]
