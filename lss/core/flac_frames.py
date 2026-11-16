"""FLACの音のstreamが、宣言した長さで正確に終わるか（後ろに何も付いていないか）を確かめる。

libsndfileはSTREAMINFOの総sample数で読むのを止めるため、streamの後ろに足されたbytes・frameは復号した音の
同一性に現れない。ここでは
1. STREAMINFOの総sample数で終わる最後のframeを、frame headerの番号から特定し、
2. そのframeのsubframe（CONSTANT・VERBATIM・FIXED・LPCとRiceの残差）を読み切って本当の終わりを求め、
   CRC-16が合い、その終わりがファイルの終わりと一致することを求める。
3. ffmpegでstreamの全frameを（宣言の長さで止めずに）復号し、sample数・sample rate・左右の数がSTREAMINFOと、
   sampleの並びがlibsndfileの復号と一致し、FLAC復号器が何も警告しないことを求める
   （後ろへ複製されたframeは余分なsampleとして、frame headerだけ別のsample rate・精度を名乗る改変は
   再生の速さの違い・復号器の警告・sampleの違いとして現れる）。
どれか一つでも確かめられなければFalse（同じ出力とは言わない）。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_BLOCK_SIZES = {1: 192, 2: 576, 3: 1152, 4: 2304, 5: 4608, 8: 256, 9: 512, 10: 1024, 11: 2048, 12: 4096,
                13: 8192, 14: 16384, 15: 32768}
_SAMPLE_SIZES = {1: 8, 2: 12, 4: 16, 5: 20, 6: 24, 7: 32}


def _crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


_CRC16 = []
for _value in range(256):
    _crc = _value << 8
    for _ in range(8):
        _crc = ((_crc << 1) ^ 0x8005) & 0xFFFF if _crc & 0x8000 else (_crc << 1) & 0xFFFF
    _CRC16.append(_crc)


def _crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC16[(crc >> 8) ^ byte]
    return crc


class _Bits:
    def __init__(self, data: bytes, start: int):
        self.value = int.from_bytes(data, "big")
        self.total = len(data) * 8
        self.pos = start * 8

    def read(self, n: int) -> int:
        if n == 0:
            return 0
        if self.pos + n > self.total:
            raise EOFError
        self.pos += n
        return (self.value >> (self.total - self.pos)) & ((1 << n) - 1)

    def unary(self) -> int:
        count = 0
        while self.read(1) == 0:
            count += 1
            if count > 1 << 20:
                raise ValueError("unary run too long")
        return count


def streaminfo(path: Path) -> dict | None:
    with Path(path).open("rb") as stream:
        if stream.read(4) != b"fLaC":
            return None
        header = stream.read(4)
        if len(header) != 4 or header[0] & 0x7F != 0:
            return None
        data = stream.read(34)
    if len(data) != 34:
        return None
    bits = int.from_bytes(data, "big")
    def field(offset, width):
        return (bits >> (34 * 8 - offset - width)) & ((1 << width) - 1)
    return {"min_block": field(0, 16), "max_block": field(16, 16), "sample_rate": field(80, 20),
            "channels": field(100, 3) + 1, "bps": field(103, 5) + 1, "total_samples": field(108, 36)}


def _header(data: bytes, at: int, info: dict) -> dict | None:
    if at + 6 > len(data) or data[at] != 0xFF or data[at + 1] & 0xFE != 0xF8:
        return None
    variable = data[at + 1] & 0x01
    block_code, rate_code = data[at + 2] >> 4, data[at + 2] & 0x0F
    channel_code, size_code = data[at + 3] >> 4, (data[at + 3] >> 1) & 0x07
    if block_code == 0 or rate_code == 0x0F or channel_code > 10 or size_code == 3 or data[at + 3] & 0x01:
        return None
    lead = data[at + 4]
    extra = 0 if lead < 0x80 else next((n for n in range(1, 7) if (lead >> (6 - n)) == (0xFF >> (6 - n)) - 1
                                        and lead & (0x80 >> (n + 1)) == 0), None)
    if extra is None or at + 5 + extra > len(data):
        return None
    number = lead if extra == 0 else lead & (0x3F >> extra)
    for byte in data[at + 5:at + 5 + extra]:
        if byte & 0xC0 != 0x80:
            return None
        number = (number << 6) | (byte & 0x3F)
    pos = at + 5 + extra
    if block_code == 6:
        block = data[pos] + 1 if pos < len(data) else None
        pos += 1
    elif block_code == 7:
        block = int.from_bytes(data[pos:pos + 2], "big") + 1 if pos + 2 <= len(data) else None
        pos += 2
    else:
        block = _BLOCK_SIZES.get(block_code)
    pos += 1 if rate_code == 12 else 2 if rate_code in (13, 14) else 0
    if block is None or pos >= len(data) or _crc8(data[at:pos]) != data[pos]:
        return None
    channels = channel_code + 1 if channel_code < 8 else 2
    bps = info["bps"] if size_code == 0 else _SAMPLE_SIZES[size_code]
    first = number if variable else number * info["max_block"]
    return {"first_sample": first, "block": block, "channels": channels, "channel_code": channel_code,
            "bps": bps, "subframes_at": pos + 1}


def _residual(bits: _Bits, block: int, order: int) -> None:
    method = bits.read(2)
    if method > 1:
        raise ValueError("reserved residual coding")
    param_bits, escape = (4, 0xF) if method == 0 else (5, 0x1F)
    partition_order = bits.read(4)
    partitions = 1 << partition_order
    if block % partitions or (block >> partition_order) < order:
        raise ValueError("bad partition")
    for index in range(partitions):
        count = (block >> partition_order) - (order if index == 0 else 0)
        param = bits.read(param_bits)
        if param == escape:
            bits.read(bits.read(5) * count) if count else bits.read(5)
            continue
        for _ in range(count):
            bits.unary()
            bits.read(param)


def _subframe(bits: _Bits, block: int, bps: int) -> None:
    if bits.read(1):
        raise ValueError("subframe padding bit")
    kind = bits.read(6)
    if bits.read(1):
        bps -= bits.unary() + 1
    if kind == 0:
        bits.read(bps)
    elif kind == 1:
        bits.read(bps * block)
    elif 8 <= kind <= 12:
        order = kind - 8
        bits.read(bps * order)
        _residual(bits, block, order)
    elif kind >= 32:
        order = kind - 31
        bits.read(bps * order)
        precision = bits.read(4) + 1
        if precision == 16:
            raise ValueError("invalid LPC precision")
        bits.read(5)
        bits.read(precision * order)
        _residual(bits, block, order)
    else:
        raise ValueError("reserved subframe type")


def _frame_end(data: bytes, at: int, header: dict) -> int:
    bits = _Bits(data, header["subframes_at"])
    for channel in range(header["channels"]):
        side = ((header["channel_code"] == 8 and channel == 1) or (header["channel_code"] == 9 and channel == 0)
                or (header["channel_code"] == 10 and channel == 1))
        _subframe(bits, header["block"], header["bps"] + (1 if side else 0))
    end = (bits.pos + 7) // 8 + 2
    if end > len(data) or _crc16(data[at:end - 2]) != int.from_bytes(data[end - 2:end], "big"):
        raise ValueError("frame CRC-16 mismatch")
    return end


def last_frame_ends_at_eof(path: Path, info: dict, *, window: int = 1 << 20) -> bool:
    """総sample数で終わる最後のframeを読み切り、その終わりがファイルの終わりと一致するか。"""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        base = max(0, size - window)
        stream.seek(base)
        tail = stream.read()
    total = info["total_samples"]
    at = len(tail) - 6
    while at >= 0:
        at = tail.rfind(b"\xff", 0, at + 1)
        if at < 0:
            return False
        header = _header(tail, at, info)
        if header is not None and header["first_sample"] + header["block"] == total:
            try:
                # 読むのはこのframeから後ろだけ（大きなbufferを毎bit読み直さない）。
                frame = tail[at:]
                return _frame_end(frame, 0, {**header, "subframes_at": header["subframes_at"] - at}) == len(frame)
            except (EOFError, ValueError):
                pass
        at -= 1
    return False


def _libsndfile_pcm_sha256(path: Path) -> str | None:
    """libsndfileが復号したsampleを32bit整数（ffmpegのs32と同じ左詰め）に並べたhash。"""
    import hashlib

    import numpy as np
    import soundfile as sf

    digest = hashlib.sha256()
    try:
        with sf.SoundFile(str(path)) as stream:
            for block in stream.blocks(blocksize=1 << 16, dtype="int32", always_2d=True):
                digest.update(np.ascontiguousarray(block, dtype="<i4").tobytes())
    except (sf.LibsndfileError, RuntimeError, OSError):
        return None
    return digest.hexdigest()


def _wav_format(header: bytes) -> tuple[int, int, int] | None:
    """ffmpegが書いたWAVの先頭から(左右の数, sample rate, dataの始まり)。"""
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        return None
    at, fmt = 12, None
    while at + 8 <= len(header):
        kind, size = header[at:at + 4], int.from_bytes(header[at + 4:at + 8], "little")
        if kind == b"fmt " and at + 16 <= len(header):
            fmt = (int.from_bytes(header[at + 10:at + 12], "little"), int.from_bytes(header[at + 12:at + 16], "little"))
        elif kind == b"data":
            return (*fmt, at + 8) if fmt else None
        at += 8 + size + (size & 1)
    return None


def ffmpeg_decode(path: Path) -> dict | None:
    """ffmpegがstreamの全frameを（宣言の長さで止めずに）復号した結果。

    sample数・sample rate・左右の数・32bit整数sampleのhashと、FLAC復号器の警告。確かめられなければNone。
    """
    import hashlib
    import tempfile

    from lss.paths import resource_root

    ffmpeg = resource_root() / "bin" / "ffmpeg.exe"
    if not ffmpeg.is_file():
        return None
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([str(ffmpeg), "-nostdin", "-v", "warning", "-i", str(path), "-map", "0:a:0",
                                    "-map_metadata", "-1", "-fflags", "+bitexact", "-c:a", "pcm_s32le", "-f", "wav", "-"],
                                   stdout=subprocess.PIPE, stderr=errors,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        head = b""
        while len(head) < 4096:
            chunk = process.stdout.read(4096 - len(head))
            if not chunk:
                break
            head += chunk
        layout = _wav_format(head)
        digest, size = hashlib.sha256(), 0
        if layout is not None:
            digest.update(head[layout[2]:])
            size = len(head) - layout[2]
        for chunk in iter(lambda: process.stdout.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
        code = process.wait()
        errors.seek(0)
        text = errors.read().decode("utf-8", "replace")
    if code != 0 or layout is None:
        return None
    channels, rate, _ = layout
    # 埋め込み画像など音以外のstreamの読み取りの報告は除き、音の復号器と音のstreamの報告だけを数える。
    warnings = [line for line in text.splitlines() if "[flac @" in line or "aist#" in line or "dec:flac" in line]
    return {"frames": size // (4 * channels) if channels else 0, "samplerate": rate, "channels": channels,
            "pcm_s32_sha256": digest.hexdigest(), "warnings": warnings}


def stream_is_exact(path: Path, *, libsndfile_s32_sha256: str | None = None) -> bool:
    """FLACの音のstreamが宣言どおりに終わり、どの復号器でも同じ音（sample・速さ・左右）として再生されるか。

    ``libsndfile_s32_sha256`` is `_libsndfile_pcm_sha256` of the same bytes when the caller already decoded them
    (`audio_identity`); otherwise it is decoded here."""
    info = streaminfo(path)
    if info is None or info["total_samples"] == 0:
        return False
    if not last_frame_ends_at_eof(path, info):
        return False
    decoded = ffmpeg_decode(path)
    return (decoded is not None and not decoded["warnings"] and decoded["frames"] == info["total_samples"]
            and decoded["samplerate"] == info["sample_rate"] and decoded["channels"] == info["channels"]
            and decoded["pcm_s32_sha256"] == (libsndfile_s32_sha256 or _libsndfile_pcm_sha256(path)))
