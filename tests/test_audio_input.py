"""同梱FFmpegを使い、実際の音声形式から保存までを検証する。"""
from pathlib import Path
import subprocess

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.core import audio_input, models_store, session, worker
from lss.harness import infer
from lss.paths import repo_root
from tests.test_worker import _setup, _fake_msst_separate


def transcode(source, destination, codec="aac"):
    subprocess.run(
        [str(repo_root() / "bin" / "ffmpeg.exe"), "-nostdin", "-v", "error", "-n",
         "-i", str(source), "-c:a", codec, "-metadata", "title=入力の情報", str(destination)],
        check=True, capture_output=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW,
    )


@pytest.mark.parametrize("extension,codec", [
    (".wav", "pcm_s16le"), (".aiff", "pcm_s16le"), (".flac", "flac"),
    (".mp3", "libmp3lame"), (".m4a", "aac"), (".aac", "aac"),
    (".ogg", "libvorbis"), (".opus", "libopus"), (".wma", "wmav2"),
])
def test_supported_format_reaches_output_without_changing_source(tmp_path, monkeypatch, extension, codec):
    _setup(tmp_path, monkeypatch)
    seed = tmp_path / "seed.wav"
    t = np.arange(7200) / 48000
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(seed, np.column_stack([wave, wave]), 48000)
    source = tmp_path / f"入力{extension}"
    transcode(seed, source, codec)
    original = source.read_bytes()
    decoded_paths = []

    def separate(*args, **kwargs):
        decoded_paths.append(args[1])
        return _fake_msst_separate(*args, **kwargs)

    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", separate)
    result = worker.handle_job({"mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental"})
    assert result["status"] == "ok", result
    assert len(result["outputs"]) == 1
    for output in result["outputs"]:
        info = sf.info(output)
        assert info.samplerate == 48000 and info.channels == 2
        assert 0 < info.frames < 48000 * 0.4
        tags = FLAC(output)
        assert tags["LSS_INPUT_SHA256"] == [session.input_sha256_of(source)]
        assert "REPLAYGAIN_TRACK_GAIN" not in tags  # 400ms未満では測定値を推測しない。
        if extension in {".m4a", ".wma"}:
            assert tags["TITLE"] == ["入力の情報"]
    assert source.read_bytes() == original
    if extension == ".m4a":
        assert decoded_paths[0] != source
        assert not decoded_paths[0].exists()


def test_multichannel_m4a_is_rejected_before_model_loading(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    seed = tmp_path / "surround.wav"
    sf.write(seed, np.zeros((48000, 6)), 48000)
    source = tmp_path / "surround.m4a"
    transcode(seed, source)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: pytest.fail("モデルを読み込んではいけない"))
    result = worker.handle_job({"mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental"})
    assert result["status"] == "error"
    assert "チャンネル数6" in result["message"]


def test_decode_temporary_file_is_removed_on_failure_in_consumer(tmp_path):
    seed = tmp_path / "seed.wav"
    sf.write(seed, np.zeros((44100, 2)), 44100)
    source = tmp_path / "source.m4a"
    transcode(seed, source)
    with pytest.raises(RuntimeError):
        with audio_input.prepare_input(source) as decoded:
            assert decoded.is_file()
            raise RuntimeError("後続処理の失敗")
    assert not decoded.exists()


def test_decode_temporary_file_uses_requested_work_root(tmp_path):
    seed = tmp_path / "seed.wav"
    sf.write(seed, np.zeros((44100, 2)), 44100)
    source = tmp_path / "source.m4a"
    transcode(seed, source)
    work_root = tmp_path / "state" / "work"

    with audio_input.prepare_input(source, work_root=work_root) as decoded:
        assert decoded.parent.parent == work_root
        assert decoded.parent.name.startswith("scratch-decode-")

    assert list(work_root.iterdir()) == []


def test_invalid_audio_reports_read_error(tmp_path):
    source = tmp_path / "broken.m4a"
    source.write_bytes(b"broken")
    with pytest.raises(audio_input.InputDecodeError, match="音声を読み込めません"):
        with audio_input.prepare_input(source):
            pytest.fail("壊れた入力は処理しない")


def test_decode_timeout_has_specific_error_and_cleans_work_dir(tmp_path, monkeypatch):
    source = tmp_path / "slow.m4a"
    source.write_bytes(b"not directly readable")
    work_root = tmp_path / "state" / "work"

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(audio_input.subprocess, "run", timeout)

    with pytest.raises(audio_input.InputDecodeTimeoutError, match="3秒以内"):
        with audio_input.prepare_input(source, work_root=work_root, decode_timeout=3):
            pytest.fail("タイムアウトした入力は渡さない")

    assert list(work_root.iterdir()) == []


def test_nonpositive_decode_timeout_is_rejected_before_process_start(tmp_path, monkeypatch):
    source = tmp_path / "slow.m4a"
    source.write_bytes(b"not directly readable")
    monkeypatch.setattr(
        audio_input.subprocess, "run",
        lambda *a, **k: pytest.fail("FFmpegを起動しない"),
    )

    with pytest.raises(ValueError, match="0より大きい"):
        with audio_input.prepare_input(source, decode_timeout=0):
            pytest.fail("無効な制限時間では入力を渡さない")
