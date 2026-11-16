"""復号音声の先頭を元メディアの時刻へ結び付ける。"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path
import re
import subprocess

import soundfile as sf

from lss.core.audio_input import InputDecodeError, InputDecodeTimeoutError, DecoderUnavailableError
from lss.paths import resource_root


@dataclass(frozen=True)
class AudioTimelineOrigin:
    numerator: int
    denominator: int
    method: str
    decoder_sha256: str | None = None

    def to_dict(self) -> dict:
        offset = Fraction(self.numerator, self.denominator)
        return {
            "offset_numerator": offset.numerator, "offset_denominator": offset.denominator,
            "offset_seconds": float(offset), "method": self.method,
            "decoder_sha256": self.decoder_sha256,
        }


def probe_audio_timeline(path: Path, *, timeout_seconds: float = 30) -> AudioTimelineOrigin:
    """第1音声streamの最初の復号sampleのPTSを、整数と時間基準で保存する。"""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {path}")
    if timeout_seconds <= 0:
        raise ValueError("時刻確認の制限時間は正数である必要があります")
    try:
        sf.info(path)
    except sf.LibsndfileError:
        pass
    else:
        return AudioTimelineOrigin(0, 1, "standalone_audio_sample_zero")
    ffmpeg = resource_root() / "bin" / "ffmpeg.exe"
    if not ffmpeg.is_file():
        raise DecoderUnavailableError("元メディアの音声時刻の確認に必要なFFmpegがありません")
    command = [
        str(ffmpeg), "-nostdin", "-v", "info", "-copyts", "-i", str(path),
        "-map", "0:a:0", "-vn", "-af", "ashowinfo", "-frames:a", "1", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout_seconds, creationflags=subprocess.CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired as exc:
        raise InputDecodeTimeoutError("元メディアの音声時刻を制限時間内に確認できません") from exc
    if result.returncode:
        raise InputDecodeError("元メディアの音声先頭時刻を確認できません")
    output = result.stderr.decode("utf-8", errors="replace")
    matches = re.findall(r"\[Parsed_ashowinfo_[^\]]+\]\s+n:0\s+pts:(-?\d+)\s+pts_time:\S+.*?\brate:(\d+)\b", output)
    if len(matches) != 1 or int(matches[0][1]) <= 0:
        raise InputDecodeError("元メディアの音声先頭時刻を一意に確認できません")
    # ashowinfoのPTSはsample_rate時間基準。float表示の丸めを使わない。
    pts, sample_rate = map(int, matches[0])
    digest = hashlib.sha256(ffmpeg.read_bytes()).hexdigest()
    return AudioTimelineOrigin(pts, sample_rate, "first_decoded_audio_pts", digest)
