"""音源の素性と波形。読めた事実だけを持ち、推測で埋めない。"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from lss.ui.media import MediaFacts, clock, facts
from lss.ui.waveform import FLOOR, peaks


@pytest.fixture
def source(tmp_path):
    rate = 44100
    time = np.linspace(0, 3.0, int(rate * 3.0), endpoint=False)
    envelope = np.abs(np.sin(time * 2.0))
    signal = (envelope * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    stereo = np.stack([signal, signal], axis=1)
    path = tmp_path / "take.flac"
    sf.write(str(path), stereo, rate, format="FLAC", subtype="PCM_24")
    return path


@pytest.mark.parametrize(
    "seconds, written",
    [(0, "0:00"), (7, "0:07"), (62, "1:02"), (600, "10:00"), (3725, "1:02:05")],
)
def test_durations_use_one_grammar(seconds, written) -> None:
    """m:ss、1時間超で h:mm:ss。経過も長さも同じ書き方にする。"""
    assert clock(seconds) == written


def test_the_facts_come_from_the_header(source) -> None:
    read = facts(source)
    assert read.duration == "0:03"
    assert read.media_format == "FLAC"
    assert read.sample_rate == "44.1"
    assert read.bit_depth == "24"
    assert read.channels == 2
    assert float(read.megabytes) > 0


def test_an_unreadable_source_reports_nothing(tmp_path) -> None:
    """読めなければ項目を持たない。値を作らない。"""
    assert facts(tmp_path / "missing.wav") == MediaFacts()

    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not audio")
    read = facts(broken)
    assert read.duration == ""
    assert read.sample_rate == ""
    assert read.media_format == "WAV", "拡張子だけは事実として残る"


def test_a_lossy_container_has_no_bit_depth(tmp_path) -> None:
    rate = 44100
    signal = np.zeros((rate, 2), dtype=np.float32)
    path = tmp_path / "take.ogg"
    sf.write(str(path), signal, rate, format="OGG", subtype="VORBIS")
    assert facts(path).bit_depth == ""


def test_peaks_are_measured_not_synthesised(source) -> None:
    """凍結Canvasの合成波形は移植しない。実データのピークだけを描く。"""
    values = peaks(source)
    assert len(values) == 44
    assert max(values) == 100
    assert min(values) >= FLOOR
    assert all(0 <= value <= 100 for value in values)
    # 包絡が違えば形も違う。定数列ではない。
    assert len(set(values)) > 5


def test_peaks_are_empty_when_the_source_cannot_be_read(tmp_path) -> None:
    assert peaks(tmp_path / "missing.flac") == ()
    broken = tmp_path / "broken.flac"
    broken.write_bytes(b"nope")
    assert peaks(broken) == ()


def test_a_silent_source_still_draws_a_band(tmp_path) -> None:
    rate = 44100
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros((rate, 2), dtype=np.float32), rate)
    assert set(peaks(path)) == {FLOOR}


def test_rows_carry_the_facts_into_the_list(source) -> None:
    from lss.ui.presentation import row_for

    row = row_for(1, "take", source_path=str(source), media=facts(source))
    assert row.duration == "0:03"
    assert row.media_format == "FLAC"
    assert row.channels == 2


def test_a_row_without_facts_shows_no_spec_block() -> None:
    from lss.ui.presentation import row_for

    row = row_for(1, "take")
    assert (row.duration, row.media_format, row.sample_rate) == ("", "", "")
    assert row.channels == 0


# --- libsndfileが読めないコンテナ -------------------------------------------

def _encode(source_path, destination, *args):
    """同梱FFmpegで検証用のコンテナを作る。無ければテストを飛ばす。"""
    import subprocess

    from lss.paths import resource_root

    ffmpeg = resource_root() / "bin" / "ffmpeg.exe"
    if not ffmpeg.is_file():
        pytest.skip("同梱FFmpegが無い環境")
    result = subprocess.run(
        [str(ffmpeg), "-nostdin", "-v", "error", "-y", "-i", str(source_path),
         *args, str(destination)], capture_output=True)
    if result.returncode:
        pytest.skip(f"この環境では作成できない: {destination.suffix}")
    return destination


@pytest.fixture
def stereo_wav(tmp_path):
    rate = 44100
    signal = np.zeros((rate * 3, 2), dtype=np.float32)
    path = tmp_path / "src.wav"
    sf.write(str(path), signal, rate)
    return path


@pytest.fixture
def surround_wav(tmp_path):
    rate = 44100
    signal = np.zeros((rate * 3, 6), dtype=np.float32)
    path = tmp_path / "src6.wav"
    sf.write(str(path), signal, rate)
    return path


def _await_probe(path):
    import time

    from lss.ui.media import request

    done = []
    request(path, lambda: done.append(1))
    for _ in range(400):
        if done:
            return
        time.sleep(0.02)
    pytest.fail("素性の調査が終わらない")


def test_a_container_libsndfile_cannot_read_still_gets_its_facts(
    tmp_path, stereo_wav,
) -> None:
    """LSSが受け付ける入力なのに行が空になるのは、読めなかったのではなく穴。"""
    target = _encode(stereo_wav, tmp_path / "take.m4a", "-c:a", "aac")
    with pytest.raises(Exception):
        sf.info(str(target))

    assert facts(target).sample_rate == "", "ヘッダだけでは読めない前提"
    _await_probe(target)
    read = facts(target)
    assert read.duration == "0:03"
    assert read.sample_rate == "44.1"
    assert read.channels == 2
    assert read.media_format == "M4A"


def test_a_multichannel_container_reports_its_channel_count(
    tmp_path, surround_wav,
) -> None:
    """2ch超は処理前に理由を出せる。行が事実を持たないと出せない。"""
    from lss.ui.presentation import row_for

    target = _encode(surround_wav, tmp_path / "take6.m4a", "-c:a", "aac")
    _await_probe(target)
    read = facts(target)
    assert read.channels == 6
    row = row_for(1, "take6", source_path=str(target), media=read)
    assert row.note == "6ch の音源は分離できません。ステレオまで対応しています。"


def test_probing_never_blocks_the_caller(tmp_path, stereo_wav) -> None:
    import time

    from lss.ui.media import request

    target = _encode(stereo_wav, tmp_path / "quick.m4a", "-c:a", "aac")
    start = time.perf_counter()
    request(target)
    assert (time.perf_counter() - start) < 0.25, "呼び出し側を止めてはならない"


def test_an_unreadable_source_is_probed_once_and_never_again(tmp_path) -> None:
    """行の更新は毎回起きる。読めない音源のたびに調査を起こしてはならない。"""
    from lss.ui import media

    broken = tmp_path / "broken.m4a"
    broken.write_bytes(b"\0")

    calls: list[str] = []
    original = media._probe
    media._probe = lambda target: calls.append(str(target)) or ("", 0, 0)
    try:
        for _ in range(30):
            media.request(broken)
        media._probe_queue.join()
    finally:
        media._probe = original
    assert len(calls) == 1, f"同じ音源を{len(calls)}回調べている"

    # 読めなかった事実も覚える。次の更新でまた依頼しない。
    media._probe = lambda target: calls.append(str(target)) or ("", 0, 0)
    try:
        media.request(broken)
        media._probe_queue.join()
    finally:
        media._probe = original
    assert len(calls) == 1


def test_probing_never_runs_more_workers_than_it_declares(tmp_path) -> None:
    """同時に走る調査を絞る。フォルダ登録で資源を食い尽くさない。"""
    import threading

    from lss.ui import media

    sources = []
    for index in range(24):
        path = tmp_path / f"many{index}.m4a"
        path.write_bytes(b"\0")
        sources.append(path)

    peak = [0]
    live = [0]
    guard = threading.Lock()
    release = threading.Event()
    original = media._probe

    def slow(target):
        with guard:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        release.wait(timeout=5)
        with guard:
            live[0] -= 1
        return ("", 0, 0)

    media._probe = slow
    try:
        for path in sources:
            media.request(path)
        # 走り始めるだけの時間を与えてから解放する。
        import time

        time.sleep(0.3)
        release.set()
        media._probe_queue.join()
    finally:
        media._probe = original
    assert peak[0] <= media.PROBE_WORKERS, f"同時に{peak[0]}件走った"
