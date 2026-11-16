"""元動画の音声開始offsetを丸めず保存し、不明な時刻を捏造しない。"""
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from lss.core import audio_timeline as timeline
from lss.core.audio_input import InputDecodeError, InputDecodeTimeoutError


def test_standalone_audio_origin_is_its_first_sample(tmp_path, monkeypatch):
    source = tmp_path / "audio.flac"
    sf.write(source, np.zeros((100, 2)), 48000)
    monkeypatch.setattr(timeline.subprocess, "run", lambda *a, **k: pytest.fail("単独音声には追加復号を行わない"))
    assert timeline.probe_audio_timeline(source).to_dict()["offset_seconds"] == 0


def _setup(tmp_path, monkeypatch, lines, code=0):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"container")
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "ffmpeg.exe").write_bytes(b"test-decoder")
    monkeypatch.setattr(timeline, "resource_root", lambda: tmp_path)
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=code, stderr=lines.encode("utf-8"))

    monkeypatch.setattr(timeline.subprocess, "run", run)
    return source, calls


@pytest.mark.parametrize("pts", [336, -1024, 0])
def test_first_audio_pts_uses_integer_sample_timebase(tmp_path, monkeypatch, pts):
    source, calls = _setup(tmp_path, monkeypatch, f"[Parsed_ashowinfo_0 @ 123] n:0 pts:{pts} pts_time:0.007 fmt:fltp channels:2 rate:48000 nb_samples:648\n")
    result = timeline.probe_audio_timeline(source).to_dict()
    assert result["offset_seconds"] == pts / 48000
    assert result["offset_numerator"] / result["offset_denominator"] == pts / 48000
    args, options = calls[0]
    assert "-copyts" in args
    assert args[args.index("-map") + 1] == "0:a:0"
    assert options["creationflags"] == subprocess.CREATE_NO_WINDOW
    assert len(result["decoder_sha256"]) == 64


@pytest.mark.parametrize("lines", ["", "n:0 pts:336 pts_time:0.007 rate:48000", "[Parsed_ashowinfo_0 @ 1] n:0 pts:1 pts_time:0.1 rate:0", "[Parsed_ashowinfo_0 @ 1] n:0 pts:1 pts_time:0.1 rate:48000\n" * 2])
def test_unresolved_or_ambiguous_origin_is_a_failure(tmp_path, monkeypatch, lines):
    source, _ = _setup(tmp_path, monkeypatch, lines)
    with pytest.raises(InputDecodeError):
        timeline.probe_audio_timeline(source)


def test_timeline_probe_timeout_is_actionable(tmp_path, monkeypatch):
    source, _ = _setup(tmp_path, monkeypatch, "")
    monkeypatch.setattr(timeline.subprocess, "run", lambda args, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(args, kw["timeout"])))
    with pytest.raises(InputDecodeTimeoutError):
        timeline.probe_audio_timeline(source, timeout_seconds=1)
