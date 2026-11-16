"""既知の録音（リリース済みの劇伴など）が番組のどこに現れるかの、正の証拠を得る。

契約は `docs/BGM.md` の「Known OST and identity」。

- 一致は「同じ録音が、この区間に、このずれで現れる」という正の証拠だけを返す。
- 一致しないことは、劇伴が無いことや未知曲であることの証拠にしない（`unresolved` のまま）。
- 同じ主題の別アレンジ・別演奏を同一録音にまとめない。スペクトルのピーク配置が一貫したずれで一致し、
  参照の波形で番組の波形を偶然の水準より大きく説明できる範囲、またはファイル内容SHA-256と
  全復号信号が完全一致した範囲を `known_exact` の根拠にする。どちらも意味役割は証明しない。

手法はスペクトルのピーク対によるランドマーク指紋（台詞や効果音が重なっても一部のピークが残る）と、
候補ごとの波形の説明率。台詞が重なって説明率が絶対値に届かない時は、偶然の水準との比と、説明の後に参照の形の
音楽が残らないこと（`bed_blocks`：別の演奏・別アレンジは同じ音符でも残る）で確かめる。番組の切り貼りで同じ
参照の近いずれが隣にある時は、候補のずれだけが続く範囲で確かめる。しきい値はDevelopment段階の値で、昇格の根拠ではない。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from lss.paths import resource_root

METHOD_VERSION = 3
SAMPLE_RATE = 11025
N_FFT = 1024
HOP = 256
FRAME_SECONDS = HOP / SAMPLE_RATE
_LOW_BIN = int(100 * N_FFT / SAMPLE_RATE)
_HIGH_BIN = int(5000 * N_FFT / SAMPLE_RATE)
_PEAK_FREQ_SIZE = 21
_PEAK_TIME_SIZE = 11
_PEAKS_PER_SECOND = 60
_FAN_PEAKS = 15
_MAX_DT = 48
_MAX_DF = 48
_BANDS = 64

# Development段階の判定値。固定ケースで調整し、昇格時に改めて検証する。
MIN_VOTES = 12
MIN_SECONDS = 6.0
_SEGMENT_GAP_SECONDS = 8.0
_NULL_SHIFTS = 24
# 同じ録音なら、区間のどこでもずれは変わらない。伸縮・別テンポの版はずれが動く。
_WINDOW_SECONDS = 3.0
_WINDOW_TOLERANCE_SECONDS = 1.5
_WINDOW_MIN_VOTES = 6
_MIN_CONSISTENT_WINDOWS = 2
# 波形の検証。参照を短いFIRに通して番組音声を説明できる割合。
_LAG_SAMPLES = 512
_FIR_TAPS = 16
_BLOCK_SAMPLES = 2 * SAMPLE_RATE
_WAVEFORM_NULLS = 4
_ALLOWED_MISSES = 1
MIN_BLOCK_EXPLAINED = 0.15
MIN_RUN_BLOCKS = 3
MIN_EXPLAINED_MARGIN = 0.1
# 台詞の下の波形の検証（`bed_blocks`）。台詞が重なると、同じ録音でも説明率は音楽の割合（-10 dBで約0.09）までしか
# 上がらず、上の絶対値（0.15）に届かない。そこで偶然の水準との比と、参照の形をした音楽が説明の後の残りから
# 消えたか（残存比）で同じ録音を確かめる。別の演奏・別アレンジは同じ音符のスペクトルを持つが波形で説明できず、
# 残りに参照の形の音楽がそのまま残る（残存比 ≈ 1）。合成ベンチマーク（`tests/reference_recall_bench.py`、
# 3シード・+10〜-10 dB・編集7種）で、同じ録音の2秒ブロックの残存比は最大0.78・一続きの使用の中央値は最大0.64、
# 別演奏・別アレンジはブロックで最小0.84・中央値で最小0.96。偶然の位置の説明率は最大0.012、-10 dBの同じ録音の
# 説明率の下位10%は約0.05。しきい値はその間に置いたDevelopment段階の値。
BED_MIN_EXPLAINED = 0.04
BED_NULL_FACTOR = 3.0
BED_MIN_PRESENCE = 0.2
BED_MAX_REMAINING = 0.7
BED_MAX_RUN_REMAINING = 0.6
# 標本単位のずれをブロックごとに追う幅（再生速度のわずかな違い：2秒で±1.5 ms）。
BED_TRACK_SAMPLES = 16
_BED_NFFT = 512
_BED_HOP = 128


class DecodeError(RuntimeError):
    """音声を指紋用に復号できない。"""


_DECODED: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_DECODED_LIMIT = 16


def decode_mono(path: Path, *, start: float | None = None, duration: float | None = None) -> np.ndarray:
    """同梱FFmpegで1ch・11025Hzへ復号する（指紋専用。成果物の音声には使わない）。

    同じファイル（所在・大きさ・更新時刻）と区間の復号は、処理内で1回だけ行う。解析WAVは1話の中で
    cue照合・参照取得・重複判定が何度も読む。返す配列は読み取り専用（bytes由来）で、呼び出し側は
    共有された値を書き換えられない。失敗は残さない。
    """
    try:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        key = (str(resolved), stat.st_size, stat.st_mtime_ns, start, duration)
    except OSError:
        return _decode_mono_uncached(path, start=start, duration=duration)
    cached = _DECODED.get(key)
    if cached is not None:
        _DECODED.move_to_end(key)
        return cached
    samples = _decode_mono_uncached(path, start=start, duration=duration)
    if samples.flags.writeable:
        samples.flags.writeable = False
    stat_after = resolved.stat()
    if (stat_after.st_size, stat_after.st_mtime_ns) == key[1:3]:
        _DECODED[key] = samples
        while len(_DECODED) > _DECODED_LIMIT:
            _DECODED.popitem(last=False)
    return samples


def _decode_mono_uncached(path: Path, *, start: float | None = None, duration: float | None = None) -> np.ndarray:
    command = [str(resource_root() / "bin" / "ffmpeg.exe"), "-nostdin", "-v", "error"]
    if start is not None:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(path)]
    if duration is not None:
        command += ["-t", f"{duration:.3f}"]
    command += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise DecodeError(result.stderr.decode("utf-8", errors="replace")[:500])
    return np.frombuffer(result.stdout, dtype=np.float32)


def decode_stereo_window(path: Path, start_frame: int, frames: int) -> np.ndarray:
    """Bounded stereo analysis on the existing global fingerprint sample grid.

    Trim after resampling, without input seeking or resetting filter phase.
    This observation signal never supplies native original-PCM permission.
    """
    if type(start_frame) is not int or type(frames) is not int or start_frame < 0 or frames <= 0:
        raise ValueError("Stereo observation needs positive bounded source frames")
    command = [str(resource_root()/"bin"/"ffmpeg.exe"), "-nostdin", "-v", "error", "-i", str(path),
               "-map", "0:a:0", "-vn", "-ac", "2", "-af",
               f"aresample={SAMPLE_RATE},atrim=start_sample={start_frame}:end_sample={start_frame+frames}",
               "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, capture_output=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode or len(result.stdout) % 8:
        raise DecodeError(result.stderr.decode("utf8", errors="replace")[:500])
    values = np.frombuffer(result.stdout, np.float32).reshape(-1, 2)
    if not len(values) or not np.isfinite(values).all():
        raise DecodeError("Stereo observation has no finite source samples")
    return values


def spectrogram(samples: np.ndarray) -> np.ndarray:
    """フレーム×周波数の対数振幅（dB）。"""
    if samples.size < N_FFT:
        return np.zeros((0, N_FFT // 2 + 1), dtype=np.float32)
    frames = 1 + (samples.size - N_FFT) // HOP
    window = np.hanning(N_FFT).astype(np.float32)
    out = np.empty((frames, N_FFT // 2 + 1), dtype=np.float32)
    # 標本位置の巨大な整数配列と音声の複製を作らず、同じ窓をviewで読む。
    framed = np.lib.stride_tricks.sliding_window_view(samples, N_FFT)[::HOP]
    for begin in range(0, frames, 4096):
        end = min(frames, begin + 4096)
        magnitude = np.abs(np.fft.rfft(framed[begin:end] * window, axis=1))
        out[begin:end] = 20.0 * np.log10(magnitude + 1e-6)
    return out


def peaks(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """局所最大のピーク（時刻フレーム, 周波数ビン）。1秒あたりの数を抑えて強いものを残す。"""
    from scipy.ndimage import maximum_filter

    if spec.shape[0] == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    band = spec[:, _LOW_BIN:_HIGH_BIN]
    local = maximum_filter(band, size=(_PEAK_TIME_SIZE, _PEAK_FREQ_SIZE), mode="constant", cval=-np.inf)
    floor = np.median(band, axis=1, keepdims=True) + 6.0
    times, bins = np.nonzero((band == local) & (band > floor))
    if times.size == 0:
        return times.astype(np.int32), bins.astype(np.int32)
    strength = band[times, bins] - floor[times, 0]
    per_bucket = _PEAKS_PER_SECOND
    bucket = (times * FRAME_SECONDS).astype(np.int64)  # 1秒ごと
    order = np.lexsort((-strength, bucket))
    times, bins, bucket = times[order], bins[order], bucket[order]
    first = np.searchsorted(bucket, bucket, side="left")
    keep = (np.arange(times.size) - first) < per_bucket
    times, bins = times[keep], bins[keep] + _LOW_BIN
    order = np.lexsort((bins, times))
    return times[order].astype(np.int32), bins[order].astype(np.int32)


def landmarks(times: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ピーク対のハッシュと、起点の時刻フレーム。"""
    hashes, anchors = [], []
    for step in range(1, _FAN_PEAKS + 1):
        if times.size <= step:
            break
        dt = times[step:] - times[:-step]
        df = bins[step:] - bins[:-step]
        valid = (dt >= 1) & (dt <= _MAX_DT) & (np.abs(df) <= _MAX_DF)
        f1 = bins[:-step][valid].astype(np.uint32)
        hashes.append((f1 << 13) | ((df[valid] + _MAX_DF).astype(np.uint32) << 6) | dt[valid].astype(np.uint32))
        anchors.append(times[:-step][valid])
    if not hashes:
        return np.zeros(0, np.uint32), np.zeros(0, np.int32)
    return np.concatenate(hashes), np.concatenate(anchors).astype(np.int32)


def band_profile(spec: np.ndarray) -> np.ndarray:
    """整列の検定に使う、対数間隔の帯域エネルギー（フレームごとに平均を除き正規化）。"""
    edges = np.unique(np.geomspace(_LOW_BIN, _HIGH_BIN, _BANDS + 1).astype(int))
    power = 10.0 ** (spec / 10.0)
    bands = np.stack([power[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    bands = np.log10(bands + 1e-12)
    bands -= bands.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(bands, axis=1, keepdims=True)
    return (bands / np.maximum(norm, 1e-9)).astype(np.float32)


@dataclass
class Reference:
    reference_id: str
    path: str
    sha256: str
    frames: int
    profile: np.ndarray = field(repr=False)


@dataclass
class ReferenceIndex:
    references: list[Reference]
    hashes: np.ndarray = field(repr=False)
    owners: np.ndarray = field(repr=False)
    times: np.ndarray = field(repr=False)
    loader: Callable[[Path], np.ndarray] = field(default=decode_mono, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, paths: list[Path], *, loader=decode_mono, cache_dir: Path | None = None) -> "ReferenceIndex":
        """参照録音の索引。`cache_dir` を与えると、内容のSHA-256と指紋の設定ごとに指紋を保存・再利用する。"""
        items = []
        for path in paths:
            path = Path(path)
            sha256 = _sha256(path)
            cached = _load_fingerprint(cache_dir, sha256) if cache_dir is not None else None
            if cached is None:
                spec = spectrogram(loader(path))
                hashes, times = landmarks(*peaks(spec))
                cached = (hashes, times, band_profile(spec), spec.shape[0])
                if cache_dir is not None:
                    _save_fingerprint(cache_dir, sha256, cached)
            items.append((path.stem, str(path), sha256, cached))
        return cls._from_fingerprints(items, loader)

    @classmethod
    def from_signals(cls, signals: list[tuple[str, str, np.ndarray]]) -> "ReferenceIndex":
        """メモリ上の音声（別の番組の区間など）を参照にする。`(識別子, 同一性, 音声)` の並び。"""
        table = {identity_key: samples for _, identity_key, samples in signals}
        items = []
        for name, identity_key, samples in signals:
            spec = spectrogram(samples)
            hashes, times = landmarks(*peaks(spec))
            items.append((name, identity_key, identity_key, (hashes, times, band_profile(spec), spec.shape[0])))
        return cls._from_fingerprints(items, lambda key: table[str(key)])

    @classmethod
    def _from_fingerprints(cls, items, loader) -> "ReferenceIndex":
        references, all_hashes, all_owners, all_times = [], [], [], []
        for number, (name, location, sha256, (hashes, times, profile, frames)) in enumerate(items):
            references.append(Reference(name, location, sha256, int(frames), profile))
            all_hashes.append(hashes)
            all_owners.append(np.full(hashes.size, number, dtype=np.int32))
            all_times.append(times)
        hashes = np.concatenate(all_hashes) if all_hashes else np.zeros(0, np.uint32)
        order = np.argsort(hashes, kind="stable")
        return cls(references, hashes[order],
                   (np.concatenate(all_owners) if all_owners else np.zeros(0, np.int32))[order],
                   (np.concatenate(all_times) if all_times else np.zeros(0, np.int32))[order],
                   loader)

    def _load(self, location: str) -> np.ndarray:
        try:
            return self.loader(location)
        except (KeyError, TypeError):
            return self.loader(Path(location))

    def samples(self, owner: int) -> np.ndarray:
        """波形の検証に使う参照の音声。直近の数件だけを保持する。"""
        if owner not in self._cache:
            if len(self._cache) >= 4:
                self._cache.pop(next(iter(self._cache)))
            self._cache[owner] = self._load(self.references[owner].path)
        return self._cache[owner]


def fingerprint_settings_id() -> str:
    """指紋の設定と実装の識別子。実装だけ変えた場合も古い保存物は使わない。"""
    settings = {"method": METHOD_VERSION, "rate": SAMPLE_RATE, "n_fft": N_FFT, "hop": HOP,
                "band": [_LOW_BIN, _HIGH_BIN], "peak": [_PEAK_TIME_SIZE, _PEAK_FREQ_SIZE, _PEAKS_PER_SECOND],
                "pairs": [_FAN_PEAKS, _MAX_DT, _MAX_DF], "bands": _BANDS,
                "implementation_sha256": _sha256(Path(__file__))}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def matching_settings_id() -> str:
    """録音照合と候補区間の数値設定・実装を保存結果の再利用に対応付ける。"""
    settings = {name: value for name, value in globals().items()
                if name.isupper() and isinstance(value, (int, float))}
    settings["fingerprint_settings_id"] = fingerprint_settings_id()
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def _fingerprint_path(cache_dir: Path, sha256: str) -> Path:
    return Path(cache_dir) / f"{sha256}-{fingerprint_settings_id()}.npz"


def _load_fingerprint(cache_dir: Path, sha256: str):
    path = _fingerprint_path(cache_dir, sha256)
    try:
        with np.load(path) as data:
            return (data["hashes"].astype(np.uint32), data["times"].astype(np.int32),
                    data["profile"].astype(np.float32), int(data["frames"]))
    except (OSError, KeyError, ValueError):
        return None


def _save_fingerprint(cache_dir: Path, sha256: str, fingerprint) -> None:
    import os
    from tempfile import NamedTemporaryFile

    path = _fingerprint_path(cache_dir, sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes, times, profile, frames = fingerprint
    with NamedTemporaryFile(dir=path.parent, prefix=".fp-", suffix=".npz", delete=False) as stream:
        np.savez(stream, hashes=hashes, times=times, profile=profile, frames=np.int64(frames))
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RecordingMatch:
    reference_id: str
    reference_path: str
    reference_sha256: str
    query_start_seconds: float
    query_end_seconds: float
    reference_start_seconds: float
    offset_seconds: float
    votes: int
    similarity: float
    null_mean: float
    null_std: float
    z: float
    measured_windows: int
    max_window_drift_seconds: float | None
    waveform_explained: float | None
    waveform_null_max: float | None
    waveform_blocks: int
    lag_samples: int | None
    alternative_alignments: int
    identity_state: str
    evidence_kind: str = "landmark_and_waveform"
    # 台詞の下の検証で決めた一致だけが持つ、説明の後に残った参照の形の音楽の割合（中央値）。
    remaining_music: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _aligned_similarity(query: np.ndarray, reference: np.ndarray, q_start: int, q_end: int, offset: int) -> float:
    lo = max(q_start, -offset, 0)
    hi = min(q_end, reference.shape[0] - offset, query.shape[0])
    if hi - lo < 8:
        return float("nan")
    return float(np.mean(np.sum(query[lo:hi] * reference[lo + offset:hi + offset], axis=1)))


def _offset_consistency(times: np.ndarray, offsets: np.ndarray, q_start: int, q_end: int,
                        offset: int) -> tuple[int, int | None]:
    """区間を数秒ごとに分け、各窓で最も多いずれが候補のずれと一致するか。

    返り値は（票が足りて測れた窓の数, 候補からの最大のずれ[フレーム]）。
    """
    width = int(_WINDOW_SECONDS / FRAME_SECONDS)
    tolerance = int(_WINDOW_TOLERANCE_SECONDS / FRAME_SECONDS)
    measured, worst = 0, None
    for begin in range(q_start, max(q_end - width // 2, q_start + 1), width):
        inside = (times >= begin) & (times < begin + width) & (np.abs(offsets - offset) <= tolerance)
        if inside.sum() < _WINDOW_MIN_VOTES:
            continue
        values, counts = np.unique(offsets[inside], return_counts=True)
        # 隣り合うずれ（±1フレーム）はまとめて数える。
        smoothed = np.array([counts[np.abs(values - v) <= 1].sum() for v in values])
        if smoothed.max() < _WINDOW_MIN_VOTES:
            continue
        mode = int(values[np.argmax(smoothed)])
        measured += 1
        worst = abs(mode - offset) if worst is None else max(worst, abs(mode - offset))
    return measured, worst


def _consistent_span(times: np.ndarray, offsets: np.ndarray, q_start: int, q_end: int,
                     offset: int) -> tuple[int, int] | None:
    """番組の編集（切り貼り・ループ）で、同じ参照の近いずれの部分が隣にある時の、候補のずれだけが続く範囲。

    `_offset_consistency` と同じ窓で、最も多いずれが候補から1フレーム以内の窓が続く最長の範囲を、両端で半窓だけ
    広げて返す（境目の位置は波形の検証が決める）。ずれが連続して動く（伸縮・別テンポの）版は、続く窓が短く、
    固定のずれの波形でも説明できないので、ここで同じ録音にはならない。
    """
    width = int(_WINDOW_SECONDS / FRAME_SECONDS)
    tolerance = int(_WINDOW_TOLERANCE_SECONDS / FRAME_SECONDS)
    agree: list[tuple[int, bool]] = []
    for begin in range(q_start, max(q_end - width // 2, q_start + 1), width):
        inside = (times >= begin) & (times < begin + width) & (np.abs(offsets - offset) <= tolerance)
        if inside.sum() < _WINDOW_MIN_VOTES:
            continue
        values, counts = np.unique(offsets[inside], return_counts=True)
        smoothed = np.array([counts[np.abs(values - v) <= 1].sum() for v in values])
        if smoothed.max() < _WINDOW_MIN_VOTES:
            continue
        agree.append((begin, abs(int(values[np.argmax(smoothed)]) - offset) <= 1))
    best, current = None, None
    for begin, ok in agree:
        if ok:
            current = [begin, begin, 1] if current is None else [current[0], begin, current[2] + 1]
            if best is None or current[2] > best[2]:
                best = list(current)
        else:
            current = None
    if best is None or best[2] < _MIN_CONSISTENT_WINDOWS:
        return None
    return max(q_start, best[0] - width // 2), min(q_end, best[1] + width + width // 2)


def _segment(signal: np.ndarray, start: int, end: int) -> np.ndarray:
    """範囲外は無音で埋めた区間。"""
    out = np.zeros(max(0, end - start), dtype=np.float32)
    lo, hi = max(start, 0), min(end, signal.size)
    if hi > lo:
        out[lo - start:hi - start] = signal[lo:hi]
    return out


def _fine_lag(query: np.ndarray, reference: np.ndarray) -> int:
    """フレーム単位のずれを、標本単位へ詰める（相互相関の最大）。"""
    size = 1 << int(np.ceil(np.log2(max(query.size, 2) + 2 * _LAG_SAMPLES)))
    spectrum = np.fft.rfft(query, size) * np.conj(np.fft.rfft(reference, size))
    corr = np.fft.irfft(spectrum, size)
    lags = np.concatenate([np.arange(0, _LAG_SAMPLES + 1), np.arange(-_LAG_SAMPLES, 0)])
    values = np.concatenate([corr[:_LAG_SAMPLES + 1], corr[-_LAG_SAMPLES:]])
    return int(lags[np.argmax(np.abs(values))])


def waveform_blocks(query: np.ndarray, reference: np.ndarray, q_start: int, q_end: int,
                    offset: int, *, fine: bool = True,
                    block_samples: int = _BLOCK_SAMPLES) -> tuple[np.ndarray, np.ndarray, int | None]:
    """参照を短い線形フィルタ（音量・EQ・位相の違いを許す）に通して、番組音声をどれだけ説明できるか。

    同じ録音なら台詞や効果音が重なっていても波形の多くを直接説明でき、別の演奏・別アレンジでは
    ほぼ説明できない。ブロックごとの（開始標本, 説明率）と、使った標本単位のずれを返す。
    無音のブロックは含めない。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    empty = (np.zeros(0, np.int64), np.zeros(0, np.float64))
    y = _segment(query, q_start, q_end).astype(np.float64)
    x = _segment(reference, q_start + offset, q_end + offset).astype(np.float64)
    if y.size < block_samples or not np.any(x):
        return *empty, None
    lag = _fine_lag(y, x) if fine else 0
    x = _segment(reference, q_start + offset - lag, q_end + offset - lag).astype(np.float64)
    half = _FIR_TAPS // 2
    padded = np.pad(x, (half, _FIR_TAPS - half - 1))
    starts, values = [], []
    for begin in range(0, y.size - block_samples + 1, block_samples):
        target = y[begin:begin + block_samples]
        energy = float(target @ target)
        if energy < 1e-6 * block_samples:
            continue
        reference_block = padded[begin:begin + block_samples + _FIR_TAPS - 1]
        if not np.any(reference_block):
            values.append(0.0)
            starts.append(q_start + begin)
            continue
        design = sliding_window_view(reference_block, _FIR_TAPS)
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        values.append(max(0.0, 1.0 - float(residual @ residual) / energy))
        starts.append(q_start + begin)
    return np.asarray(starts, np.int64), np.asarray(values, np.float64), lag


def _bed_magnitude(x: np.ndarray) -> np.ndarray:
    """残存比に使う、圧縮した振幅スペクトログラム（100–5000 Hz）。"""
    from numpy.lib.stride_tricks import sliding_window_view

    frames = sliding_window_view(x, _BED_NFFT)[::_BED_HOP] * np.hanning(_BED_NFFT)
    low, high = int(100 * _BED_NFFT / SAMPLE_RATE), int(5000 * _BED_NFFT / SAMPLE_RATE)
    return np.sqrt(np.abs(np.fft.rfft(frames, axis=1))[:, low:high])


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def bed_blocks(query: np.ndarray, reference: np.ndarray, q_start: int, q_end: int, offset: int, *,
               lag: int = 0, block_samples: int = _BLOCK_SAMPLES, shape_floor: float | None = 0.0) -> dict:
    """台詞・効果音が重なる番組音声での波形の検証。ブロックごとに標本単位のずれを `BED_TRACK_SAMPLES` の範囲で追い、
    参照を短いFIRに通して説明できる割合（`explained`）、参照のスペクトルが番組に現れている度合い（`presence`、
    説明の前の振幅スペクトログラムの相関）、説明の後の残りに参照の形の音楽が残る割合（`remaining`、説明の後と前の
    相関の比）を返す。同じ録音なら説明で参照の形の音楽が消え（残存比が小さい）、別の演奏では消えない。
    無音のブロックは含めない。`shape_floor` 未満の説明率のブロックと、`None` の時は全ブロックで、
    `presence`・`remaining` を測らない（NaN。偶然の水準には説明率だけを使う）。"""
    from numpy.lib.stride_tricks import sliding_window_view

    half = _FIR_TAPS // 2
    track = BED_TRACK_SAMPLES

    def measure(begin: int, current: int):
        target = _segment(query, begin, begin + block_samples).astype(np.float64)
        energy = float(target @ target)
        if energy < 1e-6 * block_samples:
            return None, current
        base = begin + offset - current
        wide = _segment(reference, base - track, base + block_samples + track).astype(np.float64)
        if not np.any(wide):
            return None, current
        current -= int(np.argmax(np.abs(np.correlate(wide, target, "valid")))) - track
        base = begin + offset - current
        source = _segment(reference, base - half, base + block_samples + _FIR_TAPS - half - 1).astype(np.float64)
        design = sliding_window_view(source, _FIR_TAPS)
        # 16×16の正規方程式で解く（`waveform_blocks` と同じ最小二乗。特異な時だけlstsqへ）。
        gram = design.T @ design
        try:
            coefficients = np.linalg.solve(gram + 1e-9 * np.trace(gram) * np.eye(_FIR_TAPS), design.T @ target)
        except np.linalg.LinAlgError:
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        explained = max(0.0, 1.0 - float(residual @ residual) / energy)
        presence = remaining = float("nan")
        if shape_floor is not None and explained >= shape_floor:
            shape = _bed_magnitude(source[half:half + block_samples])
            presence = _correlation(_bed_magnitude(target), shape)
            remaining = (_correlation(_bed_magnitude(residual), shape) / presence if presence > 1e-3
                         else float("inf"))
        return (begin, explained, presence, remaining, current), current

    begins = list(range(q_start, q_end - block_samples + 1, block_samples))
    # 区間全体で詰めたずれは区間の中ほどで正しい。速さのわずかな違いでずれが動いても、中ほどから両側へ追う。
    middle = len(begins) // 2
    rows = []
    current = int(lag)
    for begin in begins[middle:]:
        row, current = measure(begin, current)
        rows.append(row)
    current = int(lag)
    for begin in reversed(begins[:middle]):
        row, current = measure(begin, current)
        rows.append(row)
    rows = sorted(row for row in rows if row is not None)
    columns = list(zip(*rows)) if rows else [[]] * 5
    return {key: np.asarray(column, np.int64 if key in ("starts", "lags") else np.float64)
            for key, column in zip(("starts", "explained", "presence", "remaining", "lags"), columns)}


def bed_run(query: np.ndarray, reference: np.ndarray, q_start: int, q_end: int, offset: int,
            null_offsets: list[int], *, lag: int = 0) -> dict | None:
    """台詞の下で同じ録音だと言える、検証ブロックの続く範囲（1ブロックの落ちを許す）。無ければNone。

    偶然の水準は、同じ区間を参照の別の位置（`null_offsets`、フレーム単位）へずらし、同じ追跡で測った説明率の最大。
    ブロックは、説明率がその `BED_NULL_FACTOR` 倍と `BED_MIN_EXPLAINED` 以上、参照のスペクトルが現れていて
    （`BED_MIN_PRESENCE`）、残存比が `BED_MAX_REMAINING` 以下の時だけ数える。範囲の残存比の中央値は
    `BED_MAX_RUN_REMAINING` 以下でなければならない。
    """
    blocks = bed_blocks(query, reference, q_start, q_end, offset, lag=lag, shape_floor=BED_MIN_EXPLAINED)
    if not blocks["starts"].size:
        return None
    shaped = (blocks["presence"] >= BED_MIN_PRESENCE) & (blocks["remaining"] <= BED_MAX_REMAINING)

    def longest(floor: float):
        good = shaped & (blocks["explained"] >= floor)
        run = exact_run(blocks["starts"], np.where(good, 1.0, 0.0))
        return good, (run if run is not None and run[1] - run[0] + 1 >= MIN_RUN_BLOCKS else None)

    # 偶然の水準は床を上げるだけなので、下限の床で範囲が無ければ測らない。
    if longest(BED_MIN_EXPLAINED)[1] is None:
        return None
    nulls = []
    for candidate in null_offsets[:_WAVEFORM_NULLS]:
        nulls.extend(bed_blocks(query, reference, q_start, q_end, candidate * HOP,
                                shape_floor=None)["explained"].tolist())
    null_max = max(nulls) if nulls else None
    if null_max is None:
        return None
    floor = max(BED_MIN_EXPLAINED, BED_NULL_FACTOR * null_max)
    good, run = longest(floor)
    if run is None:
        return None
    first, last = run
    inside = slice(first, last + 1)
    counted = good[inside]
    remaining = float(np.median(blocks["remaining"][inside][counted]))
    explained = float(np.mean(blocks["explained"][inside]))
    if remaining > BED_MAX_RUN_REMAINING or explained < floor:
        return None
    start = int(blocks["starts"][first])
    end = int(blocks["starts"][last]) + _BLOCK_SAMPLES
    start, end = _trim_bed_edges(query, reference, start, end, offset, blocks, first, last,
                                 float(np.median(blocks["explained"][inside][counted])))
    return {"start": start, "end": end, "explained": explained, "null_max": null_max,
            "blocks": last - first + 1, "remaining": remaining,
            "lag": int(blocks["lags"][first])}


def _trim_bed_edges(query, reference, start, end, offset, blocks, first, last, typical) -> tuple[int, int]:
    """範囲の先頭と末尾のブロックを0.5秒ごとに見直す。台詞の下では説明率が音楽の割合で決まるので、範囲の中ほどの
    説明率の半分に届かない端（別の音楽・台詞だけ）を削る。"""
    def explained(at: int, lag: int) -> float:
        b = bed_blocks(query, reference, at, at + _EDGE_SAMPLES, offset, lag=lag, block_samples=_EDGE_SAMPLES,
                       shape_floor=None)
        return float(b["explained"][0]) if b["explained"].size else 0.0

    head_lag, tail_lag = int(blocks["lags"][first]), int(blocks["lags"][last])
    for at in range(start, min(end, start + _BLOCK_SAMPLES), _EDGE_SAMPLES):
        if explained(at, head_lag) >= 0.5 * typical:
            start = at
            break
    else:
        start = min(end, start + _BLOCK_SAMPLES)
    tail_from = max(start, end - _BLOCK_SAMPLES)
    for at in range(end - _EDGE_SAMPLES, tail_from - 1, -_EDGE_SAMPLES):
        if explained(at, tail_lag) >= 0.5 * typical:
            end = at + _EDGE_SAMPLES
            break
    else:
        end = tail_from
    return start, max(end, start)


_EDGE_SAMPLES = SAMPLE_RATE // 2


def _trim_edges(query: np.ndarray, reference: np.ndarray, start: int, end: int, offset: int) -> tuple[int, int]:
    """一致範囲の先頭と末尾の検証ブロックを0.5秒ごとに見直し、裏付けの無い端を削る。"""
    head_starts, head_values, _ = waveform_blocks(query, reference, start, min(end, start + _BLOCK_SAMPLES), offset,
                                                  fine=False, block_samples=_EDGE_SAMPLES)
    for block_start, value in zip(head_starts, head_values):
        if value >= MIN_BLOCK_EXPLAINED:
            start = int(block_start)
            break
    tail_from = max(start, end - _BLOCK_SAMPLES)
    tail_starts, tail_values, _ = waveform_blocks(query, reference, tail_from, end, offset,
                                                  fine=False, block_samples=_EDGE_SAMPLES)
    for block_start, value in zip(tail_starts[::-1], tail_values[::-1]):
        if value >= MIN_BLOCK_EXPLAINED:
            end = int(block_start) + _EDGE_SAMPLES
            break
    return start, max(end, start)


def exact_run(starts: np.ndarray, values: np.ndarray) -> tuple[int, int] | None:
    """説明率がしきい値を超えるブロックが続く最長の範囲（台詞などで1ブロックだけ落ちるのは許す）。"""
    best, current = None, None
    misses = 0
    for position, value in enumerate(values):
        contiguous = current is not None and starts[position] - starts[position - 1] == _BLOCK_SAMPLES
        if value >= MIN_BLOCK_EXPLAINED:
            if current is None or not contiguous:
                current = [position, position]
            else:
                current[1] = position
            misses = 0
        elif current is not None and contiguous and misses < _ALLOWED_MISSES:
            misses += 1
            continue
        else:
            current, misses = None, 0
        if current is not None and (best is None or current[1] - current[0] > best[1] - best[0]):
            best = (current[0], current[1])
    return best


def _exact_content_matches(query_samples: np.ndarray, index: ReferenceIndex,
                           content_sha256: str | None, offset: float) -> list[RecordingMatch]:
    if content_sha256 is None:
        return []
    if (not isinstance(content_sha256, str) or len(content_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in content_sha256)):
        raise ValueError("queryの内容SHA-256が不正です")
    query = np.asarray(query_samples)
    if (query.ndim != 1 or query.dtype.kind not in "fiu" or not np.isfinite(query).all()
            or query.size < MIN_SECONDS * SAMPLE_RATE or not np.any(query)):
        return []
    found = []
    for owner, reference in enumerate(index.references):
        if reference.sha256 != content_sha256:
            continue
        if not np.array_equal(query, index.samples(owner)):
            # 呼出側のhashと復号信号の不一致を、別の弱い照合根拠で覆わない。
            raise ValueError("queryの内容SHA-256に対応する復号波形が一致しません")
        found.append(RecordingMatch(
            reference.reference_id, reference.path, reference.sha256,
            offset, offset + query.size / SAMPLE_RATE, 0.0, -offset,
            0, 1.0, 0.0, 0.0, 0.0, 0, 0.0, 1.0, None,
            query.size // _BLOCK_SAMPLES, 0, 0, "known_exact",
            "exact_content_sha256_and_decoded_waveform",
        ))
    return found


def match(query_samples: np.ndarray, index: ReferenceIndex, *, query_offset_seconds: float = 0.0,
          rng: np.random.Generator | None = None, query_content_sha256: str | None = None,
          collapse: bool = True) -> list[RecordingMatch]:
    """番組（または区間）の音声と参照の索引から、録音一致の候補と検定結果を返す。

    `known_exact` は、指紋と波形の検証、またはファイル内容SHA-256と復号波形の完全一致。
    後者は6秒以上の音がある全信号に限る。hashを省略した呼出で推定・代用しない。
    `collapse=False` は、同じ参照が同じ区間に別のずれで一致したもの（曲の中の繰り返し・番組の編集）を
    畳まずに全部返す（編集か繰り返しかを呼出側が区間の構造で判断するため）。
    """
    exact_content = _exact_content_matches(query_samples, index, query_content_sha256, query_offset_seconds)
    if exact_content:
        return exact_content
    rng = rng or np.random.default_rng(0)
    query_spec = spectrogram(query_samples)
    q_hashes, q_times = landmarks(*peaks(query_spec))
    if q_hashes.size == 0 or index.hashes.size == 0:
        return []
    left = np.searchsorted(index.hashes, q_hashes, side="left")
    right = np.searchsorted(index.hashes, q_hashes, side="right")
    counts = right - left
    hit = counts > 0
    if not hit.any():
        return []
    q_rep = np.repeat(q_times[hit], counts[hit])
    starts = np.repeat(left[hit], counts[hit])
    within = np.arange(starts.size) - np.repeat(np.cumsum(counts[hit]) - counts[hit], counts[hit])
    positions = starts + within
    owners = index.owners[positions]
    offsets = index.times[positions] - q_rep

    key = owners.astype(np.int64) * (1 << 32) + (offsets.astype(np.int64) + (1 << 31))
    unique, votes = np.unique(key, return_counts=True)
    strong = votes >= MIN_VOTES
    query_profile = band_profile(query_spec)
    results: list[RecordingMatch] = []
    claimed: set[tuple[int, int]] = set()
    owner_hits: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for bin_key in unique[strong][np.argsort(-votes[strong])]:
        owner = int(bin_key // (1 << 32))
        offset = int(bin_key % (1 << 32) - (1 << 31))
        if any((owner, offset + d) in claimed for d in (-1, 0, 1)):
            continue
        claimed.add((owner, offset))
        # 同じ参照の別のずれを試すたびに、全参照の票を走査し直さない。
        if owner not in owner_hits:
            owned = owners == owner
            owner_hits[owner] = (q_rep[owned], offsets[owned])
        owner_times, owner_offsets = owner_hits[owner]
        moments = np.sort(owner_times[np.abs(owner_offsets - offset) <= 1])
        gaps = np.flatnonzero(np.diff(moments) > _SEGMENT_GAP_SECONDS / FRAME_SECONDS)
        for segment in np.split(moments, gaps + 1):
            if segment.size < MIN_VOTES:
                continue
            q_start, q_end = int(segment[0]), int(segment[-1]) + _MAX_DT
            if (q_end - q_start) * FRAME_SECONDS < MIN_SECONDS:
                continue
            reference = index.references[owner]
            similarity = _aligned_similarity(query_profile, reference.profile, q_start, q_end, offset)
            null = []
            span = q_end - q_start
            # 偶然の水準: 同じ区間を、参照の別の位置へずらして同じ類似を測る。
            null_offsets = [int(c) for c in rng.integers(-q_start - span // 2, reference.frames - q_start - span // 2,
                                                          size=_NULL_SHIFTS * 3)
                            if abs(int(c) - offset) * FRAME_SECONDS >= 5]
            for candidate in null_offsets:
                value = _aligned_similarity(query_profile, reference.profile, q_start, q_end, candidate)
                if np.isfinite(value):
                    null.append(value)
                if len(null) >= _NULL_SHIFTS:
                    break
            if not np.isfinite(similarity) or len(null) < 8:
                continue
            null_mean, null_std = float(np.mean(null)), float(np.std(null) + 1e-3)
            z = (similarity - null_mean) / null_std
            windows, drift = _offset_consistency(owner_times, owner_offsets, q_start, q_end, offset)
            # 整列スペクトルの検定（z）は記録するが判定には使わない。持続音の多い曲では、
            # 正しい整列でも別の位置と帯域形状が似て低く出るため（実測）。偶然との区別は波形の検証が担う。
            fingerprint_ok = windows >= _MIN_CONSISTENT_WINDOWS and drift is not None and drift <= 1
            if not fingerprint_ok and windows >= _MIN_CONSISTENT_WINDOWS:
                # 編集点の向こうの、同じ参照の別のずれの窓を除いた範囲だけで確かめる。
                span = _consistent_span(owner_times, owner_offsets, q_start, q_end, offset)
                if span is not None and (span[1] - span[0]) * FRAME_SECONDS >= MIN_SECONDS:
                    q_start, q_end = span
                    windows, drift = _offset_consistency(owner_times, owner_offsets, q_start, q_end, offset)
                    fingerprint_ok = windows >= _MIN_CONSISTENT_WINDOWS and drift is not None and drift <= 1
            explained = null_max = lag = None
            blocks = 0
            exact = False
            kind, remaining = "landmark_and_waveform", None
            match_start, match_end = q_start * HOP, q_end * HOP
            if fingerprint_ok:
                ref_samples = index.samples(owner)
                s_offset = offset * HOP
                starts, values, lag = waveform_blocks(query_samples, ref_samples, q_start * HOP, q_end * HOP, s_offset)
                run = exact_run(starts, values)
                nulls = []
                for candidate in null_offsets[:_WAVEFORM_NULLS]:
                    _, null_values, _ = waveform_blocks(query_samples, ref_samples, q_start * HOP, q_end * HOP,
                                                        candidate * HOP, fine=False)
                    nulls.extend(null_values.tolist())
                null_max = max(nulls) if nulls else None
                if run is not None:
                    first, last = run
                    explained = float(np.mean(values[first:last + 1]))
                    blocks = last - first + 1
                    match_start, match_end = _trim_edges(
                        query_samples, ref_samples, int(starts[first]), int(starts[last]) + _BLOCK_SAMPLES,
                        s_offset - (lag or 0))
                    exact = (blocks >= MIN_RUN_BLOCKS
                             and (null_max is None or explained >= null_max + MIN_EXPLAINED_MARGIN))
                # 台詞の下の検証は、台詞の重なりで上の検証が短く切れた・届かなかった時にも、より長く確かめられる。
                if not exact or blocks < (q_end - q_start) * HOP // _BLOCK_SAMPLES - 1:
                    bed = bed_run(query_samples, ref_samples, q_start * HOP, q_end * HOP, s_offset,
                                  null_offsets, lag=lag or 0)
                    if bed is not None and (not exact or bed["blocks"] > blocks):
                        exact, kind, remaining = True, "landmark_and_bed_waveform", round(bed["remaining"], 4)
                        explained, null_max, blocks = bed["explained"], bed["null_max"], bed["blocks"]
                        match_start, match_end, lag = bed["start"], bed["end"], bed["lag"]
            results.append(RecordingMatch(
                reference.reference_id, reference.path, reference.sha256,
                round(query_offset_seconds + match_start / SAMPLE_RATE, 3),
                round(query_offset_seconds + match_end / SAMPLE_RATE, 3),
                round(max(0, match_start + offset * HOP - (lag or 0)) / SAMPLE_RATE, 3),
                round((offset * HOP - (lag or 0)) / SAMPLE_RATE - query_offset_seconds, 3),
                int(segment.size), round(similarity, 4), round(null_mean, 4), round(null_std, 4), round(z, 2),
                windows, None if drift is None else round(drift * FRAME_SECONDS, 3),
                None if explained is None else round(explained, 4),
                None if null_max is None else round(null_max, 4), blocks, lag, 0,
                "known_exact" if exact else "unresolved", kind, remaining,
            ))
    if not collapse:
        return sorted(results, key=lambda m: (m.query_start_seconds, -m.z))
    return _collapse_alternatives(results)


def _collapse_alternatives(results: list[RecordingMatch]) -> list[RecordingMatch]:
    """同じ参照が同じ区間に複数のずれで一致する（曲の中の繰り返し）ときは、最も説明できる整列を残す。"""
    def strength(m: RecordingMatch):
        return (m.identity_state == "known_exact", m.waveform_explained or -1.0, m.z)

    kept: list[RecordingMatch] = []
    for candidate in sorted(results, key=strength, reverse=True):
        duplicate = None
        for position, existing in enumerate(kept):
            if existing.reference_sha256 != candidate.reference_sha256:
                continue
            overlap = (min(existing.query_end_seconds, candidate.query_end_seconds)
                       - max(existing.query_start_seconds, candidate.query_start_seconds))
            shorter = min(existing.query_end_seconds - existing.query_start_seconds,
                          candidate.query_end_seconds - candidate.query_start_seconds)
            if shorter > 0 and overlap / shorter >= 0.5:
                duplicate = position
                break
        if duplicate is None:
            kept.append(candidate)
        else:
            existing = kept[duplicate]
            kept[duplicate] = replace(existing, alternative_alignments=existing.alternative_alignments + 1)
    kept.sort(key=lambda m: (m.query_start_seconds, -m.z))
    return kept


def match_file(path: Path, index: ReferenceIndex, *, loader=decode_mono) -> list[RecordingMatch]:
    return match(loader(Path(path)), index)


MIN_CANDIDATE_COVERAGE = 0.5


def annotate_candidates(candidates: list[dict], matches: list[RecordingMatch], *,
                        min_coverage: float = MIN_CANDIDATE_COVERAGE) -> list[dict]:
    """M3の候補（`start_seconds`/`end_seconds` は元メディアの時刻）へ、録音一致の証拠を付けた複製を返す。

    候補の区間の `min_coverage` 以上を `known_exact` の一致が覆うときだけ `identity_state` を変える。
    覆う録音が1つなら `audio_version_id` にその参照のSHA-256を入れ、複数の別ファイル（TVサイズと
    フルサイズなど）が覆うときは版を決めずに証拠だけを並べる。一致しない候補は変更しない。
    """
    exact = [m for m in matches if m.identity_state == "known_exact"]
    annotated = []
    for candidate in candidates:
        start, end = float(candidate["start_seconds"]), float(candidate["end_seconds"])
        length = end - start
        evidence = []
        for m in exact:
            overlap = min(end, m.query_end_seconds) - max(start, m.query_start_seconds)
            if length > 0 and overlap / length >= min_coverage:
                evidence.append({
                    "reference_id": m.reference_id, "reference_sha256": m.reference_sha256,
                    "covered_fraction": round(overlap / length, 3),
                    "reference_start_seconds": round(m.reference_start_seconds + max(0.0, start - m.query_start_seconds), 3),
                    "waveform_explained": m.waveform_explained, "method_version": METHOD_VERSION,
                })
        updated = dict(candidate)
        if evidence:
            updated["identity_state"] = "known_exact"
            versions = {item["reference_sha256"] for item in evidence}
            updated["audio_version_id"] = f"sha256:{next(iter(versions))}" if len(versions) == 1 else None
            updated["identity_evidence"] = sorted(evidence, key=lambda item: -item["covered_fraction"])
        annotated.append(updated)
    return annotated


@dataclass(frozen=True)
class SharedOccurrence:
    """2つの番組に、同じ録音の区間が現れること（意味は未確定: 劇伴・OP/ED・回想・効果音を含み得る）。"""

    first_start_seconds: float
    first_end_seconds: float
    second_start_seconds: float
    second_end_seconds: float
    waveform_explained: float | None
    blocks: int

    def to_dict(self) -> dict:
        return asdict(self)


SHARED_CHUNK_SECONDS = 120.0


def shared_recordings(first: np.ndarray, second: np.ndarray, *,
                      chunk_seconds: float = SHARED_CHUNK_SECONDS) -> list[SharedOccurrence]:
    """参照音源が無くても、別の番組に同じ録音が現れる区間を探す（未発売・未知の曲の再登場を含む）。

    1本目を区間に分けて索引にし、2本目を照合する。区間の境目で分かれた一致は、同じずれでつながれば1つにする。

    同じ区間の中に収まる繰り返しは、同じ参照の別整列として畳まれる。区間の切り方を変えた
    呼び出しを重ねると拾えることがある。
    """
    size = int(chunk_seconds * SAMPLE_RATE)
    signals = [(f"{start / SAMPLE_RATE:.3f}", f"first@{start}", first[start:start + size])
               for start in range(0, max(first.size - N_FFT, 1), size)]
    index = ReferenceIndex.from_signals(signals)
    # 区間の同一性はchunkごとの位置で決まる。内容が同じchunkでも別の位置として引く。
    starts = {reference.path: int(reference.path.split("@", 1)[1]) / SAMPLE_RATE
              for reference in index.references}
    found = []
    for m in match(second, index):
        if m.identity_state != "known_exact":
            continue
        if m.reference_path not in starts:
            # この索引から出た一致ではない。位置を決められないので使わない。
            continue
        first_start = starts[m.reference_path] + m.reference_start_seconds
        length = m.query_end_seconds - m.query_start_seconds
        found.append([first_start, first_start + length, m.query_start_seconds, m.query_end_seconds,
                      m.waveform_explained or 0.0, m.waveform_blocks])
    return merge_shared(found)


def merge_shared(found: list[list]) -> list[SharedOccurrence]:
    """区間の境目で分かれた一致（1本目の開始, 終了, 2本目の開始, 終了, 説明率, ブロック数）を、同じずれでつなぐ。"""
    # ずれの順、同じずれの中は時刻の順に並べる。区間を重ねた照合や両方向の照合で重複した一致も隣り合う。
    found = sorted(found, key=lambda item: (round(item[0] - item[2], 1), item[2]))
    merged: list[list] = []
    for item in found:
        target = None
        for last in reversed(merged[-4:]):
            same_offset = abs((item[0] - item[2]) - (last[0] - last[2])) <= 0.05
            if same_offset and item[2] - last[3] <= 2.5 and last[2] - item[3] <= 2.5:
                target = last
                break
        if target is None:
            merged.append(list(item))
            continue
        weight = target[5] + item[5]
        target[4] = (target[4] * target[5] + item[4] * item[5]) / max(weight, 1)
        target[0], target[2] = min(target[0], item[0]), min(target[2], item[2])
        target[1], target[3], target[5] = max(target[1], item[1]), max(target[3], item[3]), weight
    merged.sort(key=lambda item: item[2])
    return [SharedOccurrence(round(a, 3), round(b, 3), round(c, 3), round(d, 3), round(e, 4), int(f))
            for a, b, c, d, e, f in merged]


MIN_SHARED_INSIDE = 0.8


def link_occurrences(candidates: dict[str, list[dict]],
                     shared: dict[tuple[str, str], list[SharedOccurrence]]) -> list[dict]:
    """番組ごとの候補を、同じ録音の区間を含む候補どうし・同じ参照録音と一致した候補どうしでつなぐ。

    つないでも候補を統合・削除しない（版が不確かなものを破壊的にまとめない）。返り値は2件以上のまとまりで、
    各まとまりは構成候補と、つながりの根拠（共通録音の区間、または参照録音のSHA-256）を持つ。
    `candidates` は番組の識別子ごとの候補（`candidate_id`, `start_seconds`, `end_seconds`、任意で
    `identity_evidence`）。`shared` は（1本目, 2本目）の番組の組ごとの `shared_recordings` の結果。
    同じ番組を組にすると、その番組の中の再演どうしをつなぐ。同じ候補の中で繰り返しているだけの
    ものは、別の使用箇所ではないのでまとまりにしない。
    """
    parent: dict[tuple[str, str], tuple[str, str]] = {}
    evidence: list[tuple[tuple[str, str], tuple[str, str], dict]] = []

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def containing(episode: str, start: float, end: float):
        for item in candidates.get(episode, []):
            inside = min(end, float(item["end_seconds"])) - max(start, float(item["start_seconds"]))
            if end > start and inside / (end - start) >= MIN_SHARED_INSIDE:
                yield (episode, str(item["candidate_id"]))

    for (first, second), occurrences in shared.items():
        for occurrence in occurrences:
            for a in containing(first, occurrence.first_start_seconds, occurrence.first_end_seconds):
                for b in containing(second, occurrence.second_start_seconds, occurrence.second_end_seconds):
                    evidence.append((a, b, {"kind": "shared_recording", **occurrence.to_dict()}))

    by_reference: dict[str, list[tuple[str, str]]] = {}
    for episode, items in candidates.items():
        for item in items:
            for known in item.get("identity_evidence") or []:
                by_reference.setdefault(known["reference_sha256"], []).append((episode, str(item["candidate_id"])))
    for sha256, members in by_reference.items():
        for other in members[1:]:
            evidence.append((members[0], other, {"kind": "same_reference_recording", "reference_sha256": sha256}))

    for a, b, _ in evidence:
        parent[find(a)] = find(b)
    groups: dict[tuple[str, str], dict] = {}
    for a, b, detail in evidence:
        root = find(a)
        group = groups.setdefault(root, {"members": set(), "evidence": []})
        group["members"].update({a, b})
        group["evidence"].append({"between": [list(a), list(b)], **detail})
    result = []
    for group in groups.values():
        members = sorted(group["members"])
        if len(members) < 2:
            # 同じ候補の中で繰り返しているだけで、別の使用箇所ではない。
            continue
        key = hashlib.sha256(json.dumps(members, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        result.append({"occurrence_group_id": key, "members": [list(m) for m in members],
                       "evidence": group["evidence"]})
    result.sort(key=lambda g: g["members"])
    return result


MIN_REMAINDER_SECONDS = 3.0


def _clipped_recurring(detail: dict, span_start: float, start: float, end: float) -> dict:
    """候補の端で切った再登場区間に合わせて、相手の番組側の区間も同じだけずらす。

    相手側の時刻を持たない根拠はそのまま返す。無い値を作らない。
    """
    if detail.get("other_start_seconds") is None:
        return detail
    other_start = float(detail["other_start_seconds"]) + (start - span_start)
    return {**detail, "other_start_seconds": round(other_start, 3),
            "other_end_seconds": round(other_start + (end - start), 3)}


def split_candidates_at_matches(candidates: list[dict], matches: list[RecordingMatch], *,
                                recurring: list[tuple[float, float, dict]] = (),
                                min_remainder_seconds: float = MIN_REMAINDER_SECONDS) -> list[dict]:
    """音量だけで作った長い候補を、同じ録音だと裏付けられた区間の境目で分ける。

    既知録音との一致区間は `known_exact` の子候補になる。`recurring` は、参照音源は無いが別の番組にも
    同じ録音が現れる区間（この番組の時刻, 終了, 根拠）で、既知の一致と重ならない部分だけを
    `unresolved` のまま `recurring_evidence` 付きの子候補にする。その間に残る区間は `unresolved` の子候補。
    短い残りも未解決候補として保持し、`short_unresolved_remainder` を付ける。一致した隣の曲へは含めない。
    `min_remainder_seconds` は短い残りの表示用の境界であり、候補を捨てるしきい値ではない。
    区間が無い候補はそのまま返す。
    子候補は `derived_from`（元の候補ID）と元メディアの時刻を持つ。元の候補は変更しない。
    """
    spans = [(m.query_start_seconds, m.query_end_seconds, "known", m)
             for m in matches if m.identity_state == "known_exact"]
    spans.sort(key=lambda item: (item[0], -(item[3].waveform_explained or 0)))
    recurring_spans = sorted(((float(a), float(b), "recurring", detail) for a, b, detail in recurring),
                             key=lambda item: (item[0], item[1]))
    result = []
    for candidate in candidates:
        start, end = float(candidate["start_seconds"]), float(candidate["end_seconds"])
        claimed: list[tuple[float, float, str, object]] = []
        for group in (spans, recurring_spans):
            for span_start, e, kind, payload in group:
                s, e = max(start, span_start), min(end, e)
                if e - s < MIN_SECONDS:
                    continue
                # 既に決めた区間と重なる部分は削り、残りが十分なら使う（既知の一致を優先する）。
                for cs, ce, _, _ in sorted(claimed):
                    if ce <= s or cs >= e:
                        continue
                    if cs - s >= e - ce:
                        e = min(e, cs)
                    else:
                        s = max(s, ce)
                if e - s >= MIN_SECONDS and all(ce <= s + 0.5 or cs >= e - 0.5 for cs, ce, _, _ in claimed):
                    claimed.append((s, e, kind,
                                    _clipped_recurring(payload, span_start, s, e)
                                    if kind == "recurring" else payload))
        if not claimed:
            result.append(dict(candidate))
            continue
        pieces, cursor = [], start
        for s, e, kind, payload in sorted(claimed, key=lambda item: item[0]):
            s = max(s, cursor)
            if s > cursor:
                pieces.append((cursor, s, None, None))
            pieces.append((s, e, kind, payload))
            cursor = e
        if end > cursor:
            pieces.append((cursor, end, None, None))
        for number, (s, e, kind, payload) in enumerate(pieces):
            child = {key: value for key, value in candidate.items()
                     if key not in {"identity_evidence", "identity_state", "audio_version_id", "recurring_evidence",
                                    "short_unresolved_remainder", "dropped_remainders"}}
            child.update({
                "candidate_id": f"{candidate['candidate_id']}/{number:02d}",
                "derived_from": candidate["candidate_id"],
                "start_seconds": s, "end_seconds": e,
                "identity_state": "unresolved", "audio_version_id": None,
            })
            if kind is None and e - s < min_remainder_seconds:
                child["short_unresolved_remainder"] = True
            if kind == "known":
                [child] = annotate_candidates([child], [payload], min_coverage=0.99)
            elif kind == "recurring":
                child["recurring_evidence"] = [payload]
            result.append(child)
    return result


def recurring_spans(shared: dict[tuple[str, str], list[SharedOccurrence]]) -> dict[str, list[tuple[float, float, dict]]]:
    """番組の組ごとの共通録音を、番組ごとの（開始, 終了, 根拠）へ並べ直す。"""
    spans: dict[str, list[tuple[float, float, dict]]] = {}
    for (first, second), occurrences in shared.items():
        for item in occurrences:
            spans.setdefault(first, []).append((item.first_start_seconds, item.first_end_seconds,
                                                {"other_episode": second, "other_start_seconds": item.second_start_seconds,
                                                 "other_end_seconds": item.second_end_seconds,
                                                 "waveform_explained": item.waveform_explained}))
            spans.setdefault(second, []).append((item.second_start_seconds, item.second_end_seconds,
                                                 {"other_episode": first, "other_start_seconds": item.first_start_seconds,
                                                  "other_end_seconds": item.first_end_seconds,
                                                  "waveform_explained": item.waveform_explained}))
    return spans


def novelty_decisions(candidates: list[dict], *, covered_fraction: float = 0.9) -> list[dict]:
    """候補ごとに、新しい成果物として保存する必要があるかの判断材料を返す（候補は変更しない）。

    - `known_recording_exists`: 候補のほぼ全体（`covered_fraction`以上）が手元の参照録音と同じ録音。
      新規の成果物にせず、参照録音と位置を示す（`no_novel_asset` の根拠）。
    - `partly_known`: 一部だけが参照録音と一致する。残りは新規の可能性があるため保留する。
    - `pending`: 既知の録音と一致しない。劇伴かどうか・保存するかは、意味の判定を待つ（未知曲として捨てない）。
    """
    decisions = []
    for candidate in candidates:
        evidence = candidate.get("identity_evidence") or []
        best = max((item.get("covered_fraction") or 0.0 for item in evidence), default=0.0)
        if candidate.get("identity_state") == "known_exact" and best >= covered_fraction:
            decision = "known_recording_exists"
        elif evidence:
            decision = "partly_known"
        else:
            decision = "pending"
        entry = {"candidate_id": candidate["candidate_id"], "decision": decision,
                 "start_seconds": candidate["start_seconds"], "end_seconds": candidate["end_seconds"]}
        if evidence:
            top = max(evidence, key=lambda item: item.get("covered_fraction") or 0.0)
            entry["reference"] = {"reference_id": top.get("reference_id"), "reference_sha256": top["reference_sha256"],
                                  "reference_start_seconds": top.get("reference_start_seconds")}
        if candidate.get("recurring_evidence"):
            entry["recurs_in"] = sorted({item["other_episode"] for item in candidate["recurring_evidence"]})
        decisions.append(entry)
    return decisions


def episode_zero_reason(decisions: list[dict]) -> str | None:
    """全候補が既知の録音で説明できるときだけ `no_novel_asset` を返す。候補が無いことは不在の証拠にしない。"""
    if decisions and all(item["decision"] == "known_recording_exists" for item in decisions):
        return "no_novel_asset"
    return None
