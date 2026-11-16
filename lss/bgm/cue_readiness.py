"""公開前のcueを、参照音源なしで測れる範囲だけ検査する。

`docs/QUALITY_AND_VALIDATION.md` のL0（完全性）とL3（信号忠実度）、および
「Ranking前のHard gate」に対応する。意味的な正しさ（L4）も分離品質（L2）も扱わない。
ここで通ることは「良い劇伴である」ことを意味せず、「壊れていない」ことだけを意味する。

測定としきい値を分ける。`measure` は判断を持たない観測値だけを返し、`evaluate` は
先に宣言したRecipeと突き合わせる。既定のしきい値は持たない。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real

import numpy as np

from lss.output import scan

SCHEMA_VERSION = 1
LEGACY_METHOD = "cue-publication-readiness-v1"
METHOD = "cue-publication-readiness-v2"
SUPPORTED_METHODS = (LEGACY_METHOD, METHOD)
# 帯域は判断ではなく観測として報告する。累積比は音楽の帯域分布、edgeは帯域制限を見る。
BANDWIDTH_FRACTIONS = (0.95, 0.99)
EDGE_DROPS_DB = (40.0, 60.0)
_SILENCE_FLOOR_DBFS = -200.0


class CueReadinessError(ValueError):
    """検査の入力、または先に宣言したRecipeが契約に合わない。"""


def _dbfs(value: float) -> float:
    return _SILENCE_FLOOR_DBFS if value <= 0 else max(_SILENCE_FLOOR_DBFS, 20.0 * math.log10(value))


def _stereo(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise CueReadinessError("(channels, frames)の音声が必要です")
    return values.astype(np.float64, copy=False)


def _bandwidth(samples: np.ndarray, samplerate: int, *, method: str = METHOD) -> dict:
    """累積スペクトルが決まった比率へ達する周波数。帯域が失われた出力を見つける。"""
    # Keep v1 available for exact historical recipe replay. New measurements sum
    # channel powers: downmixing first cancels valid out-of-phase stereo music.
    channels = samples.mean(axis=0, keepdims=True) if method == LEGACY_METHOD else samples
    frames = channels.shape[1]
    size = min(1 << 15, 1 << max(8, int(frames).bit_length() - 1))
    if frames < size:
        size = 1 << max(8, int(frames).bit_length() - 1)
    window = np.hanning(size)
    spectrum = np.zeros(size // 2 + 1)
    blocks = 0
    for start in range(0, frames - size + 1, size):
        powers = np.abs(np.fft.rfft(channels[:, start:start + size] * window, axis=1)) ** 2
        spectrum += powers.mean(axis=0)
        blocks += 1
    if blocks == 0:
        return {f"bandwidth_hz_p{int(fraction * 100)}": None for fraction in BANDWIDTH_FRACTIONS}
    total = spectrum.sum()
    freqs = np.fft.rfftfreq(size, 1.0 / samplerate)
    result = {}
    for fraction in BANDWIDTH_FRACTIONS:
        key = f"bandwidth_hz_p{int(fraction * 100)}"
        if total <= 0:
            result[key] = None
            continue
        cumulative = np.cumsum(spectrum) / total
        index = int(np.searchsorted(cumulative, fraction))
        result[key] = float(freqs[min(index, freqs.size - 1)])
    result.update(_spectral_edges(spectrum, freqs))
    return result


def _spectral_edges(spectrum: np.ndarray, freqs: np.ndarray) -> dict:
    """自分の最大帯域レベルから決めた段だけ下がる最高周波数。

    累積比は音楽の帯域分布を表すだけで、低音寄りの劇伴を帯域制限と読み違える。
    ここでは帯域が切り落とされた出力（低域pass、codec上限など）を見るために、
    スペクトルの上端がどこで落ちるかを測る。
    """
    edges = {f"spectral_edge_hz_{int(drop)}db": None for drop in EDGE_DROPS_DB}
    if spectrum.size < 8 or spectrum.max() <= 0:
        return edges
    width = max(3, (spectrum.size // 128) | 1)
    kernel = np.ones(width) / width
    smoothed = np.convolve(spectrum, kernel, mode="same")
    peak = float(smoothed.max())
    if peak <= 0:
        return edges
    for drop in EDGE_DROPS_DB:
        threshold = peak * 10 ** (-drop / 10)
        above = np.flatnonzero(smoothed >= threshold)
        edges[f"spectral_edge_hz_{int(drop)}db"] = float(freqs[int(above[-1])]) if above.size else None
    return edges


def measure(samples: np.ndarray, samplerate: int, *, method: str = METHOD) -> dict:
    """参照を使わない観測値だけを返す。しきい値も合否も持たない。"""
    if method not in SUPPORTED_METHODS:
        raise CueReadinessError("未対応の公開前検査methodです")
    values = _stereo(samples)
    if type(samplerate) is not int or samplerate <= 0:
        raise CueReadinessError("正のsample rateが必要です")
    finite = bool(np.isfinite(values).all())
    channels, frames = values.shape
    energies = [float(np.sum(values[index] ** 2)) for index in range(channels)]
    silent_channels = [index for index, value in enumerate(energies) if value <= 0]
    if finite:
        # scanは(channels, frames)を受け取る。
        scanned = scan.scan_track(values.astype(np.float32), samplerate)
    else:
        # 非有限値のままloudness走査へ渡さない。壊れている事実だけを残す。
        scanned = {"loudness_lufs": None, "true_peak_dbtp": None, "clipped_samples": None,
                   "sample_peak_linear": None, "dynamic_range_db": None, "loudness_range_lu": None}
    measured = {
        "schema_version": SCHEMA_VERSION, "method": method,
        "frames": int(frames), "samplerate": samplerate, "channels": int(channels),
        "seconds": frames / samplerate, "finite": finite,
        "silent": all(value <= 0 for value in energies),
        "silent_channel_indexes": silent_channels,
        "channel_energy_dbfs": [_dbfs(math.sqrt(value / frames)) for value in energies],
        "loudness_lufs": scanned["loudness_lufs"],
        "true_peak_dbtp": scanned["true_peak_dbtp"],
        "clipped_samples": scanned["clipped_samples"],
        "dynamic_range_db": scanned["dynamic_range_db"],
    }
    if channels == 2 and finite:
        mid, side = (values[0] + values[1]) / 2, (values[0] - values[1]) / 2
        side_energy, mid_energy = float(np.sum(side ** 2)), float(np.sum(mid ** 2))
        measured["side_to_mid_db"] = (None if mid_energy <= 0
                                      else _dbfs(math.sqrt(side_energy / max(mid_energy, 1e-300))))
        both = all(value > 0 for value in energies)
        measured["channel_correlation"] = (
            float(np.dot(values[0], values[1]) / math.sqrt(energies[0] * energies[1]))
            if both else None)
    else:
        measured["side_to_mid_db"] = None
        measured["channel_correlation"] = None
    measured.update(_bandwidth(values, samplerate, method=method) if finite else
                    {**{f"bandwidth_hz_p{int(f * 100)}": None for f in BANDWIDTH_FRACTIONS},
                     **{f"spectral_edge_hz_{int(d)}db": None for d in EDGE_DROPS_DB}})
    return measured


@dataclass(frozen=True)
class CueReadinessRecipe:
    """公開前に満たすべき、参照を使わない条件。判断を持つ項目に既定値を置かない。"""

    required_channels: int
    minimum_loudness_lufs: float
    maximum_true_peak_dbtp: float
    maximum_clipped_samples: int
    minimum_spectral_edge_hz: float
    spectral_edge_drop_db: float
    minimum_side_to_mid_db: float | None
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def to_dict(self) -> dict:
        if self.schema_version != SCHEMA_VERSION or self.method not in SUPPORTED_METHODS:
            raise CueReadinessError("未対応の公開前検査schemaです")
        if type(self.required_channels) is not int or not 1 <= self.required_channels <= 2:
            raise CueReadinessError("required_channelsは1または2である必要があります")
        if type(self.maximum_clipped_samples) is not int or self.maximum_clipped_samples < 0:
            raise CueReadinessError("maximum_clipped_samplesは0以上の整数である必要があります")
        for name in ("minimum_loudness_lufs", "maximum_true_peak_dbtp", "minimum_spectral_edge_hz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise CueReadinessError(f"{name}は有限の数値である必要があります")
        if self.minimum_spectral_edge_hz <= 0:
            raise CueReadinessError("minimum_spectral_edge_hzは0より大きい必要があります")
        if self.spectral_edge_drop_db not in EDGE_DROPS_DB:
            raise CueReadinessError(f"spectral_edge_drop_dbは{list(EDGE_DROPS_DB)}のいずれかです")
        if self.minimum_side_to_mid_db is not None and (
                isinstance(self.minimum_side_to_mid_db, bool)
                or not isinstance(self.minimum_side_to_mid_db, Real)
                or not math.isfinite(self.minimum_side_to_mid_db)):
            raise CueReadinessError("minimum_side_to_mid_dbは有限の数値またはnullです")
        if self.minimum_side_to_mid_db is not None and self.required_channels != 2:
            raise CueReadinessError("左右の広がりの条件はstereoのcueにだけ宣言できます")
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-cue-readiness-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def evaluate(measured: dict, recipe: CueReadinessRecipe) -> dict:
    """観測値を先に宣言した条件へ突き合わせる。失敗は必ず名前と観測値で返す。"""
    recipe.to_dict()
    if (not isinstance(measured, dict) or measured.get("method") != recipe.method
            or measured.get("schema_version") != recipe.schema_version):
        raise CueReadinessError("この検査が作った観測値が必要です")
    failures = []

    def fail(check: str, observed, limit) -> None:
        failures.append({"check": check, "observed": observed, "limit": limit})

    if not measured["finite"]:
        fail("non_finite_samples", False, True)
    if measured["silent"]:
        fail("silent_cue", measured["channel_energy_dbfs"], recipe.minimum_loudness_lufs)
    if measured["channels"] != recipe.required_channels:
        fail("channel_count", measured["channels"], recipe.required_channels)
    if measured["silent_channel_indexes"]:
        fail("silent_channel", measured["silent_channel_indexes"], [])
    loudness = measured["loudness_lufs"]
    if loudness is None or not math.isfinite(loudness) or loudness < recipe.minimum_loudness_lufs:
        fail("loudness_lufs", loudness, recipe.minimum_loudness_lufs)
    peak = measured["true_peak_dbtp"]
    if peak is None or not math.isfinite(peak) or peak > recipe.maximum_true_peak_dbtp:
        fail("true_peak_dbtp", peak, recipe.maximum_true_peak_dbtp)
    clipped = measured["clipped_samples"]
    if clipped is None or clipped > recipe.maximum_clipped_samples:
        fail("clipped_samples", clipped, recipe.maximum_clipped_samples)
    key = f"spectral_edge_hz_{int(recipe.spectral_edge_drop_db)}db"
    edge = measured.get(key)
    if edge is None or edge < recipe.minimum_spectral_edge_hz:
        fail(key, edge, recipe.minimum_spectral_edge_hz)
    if recipe.minimum_side_to_mid_db is not None:
        side = measured["side_to_mid_db"]
        if side is None or side < recipe.minimum_side_to_mid_db:
            # 左右が同じ信号へ潰れたcueを、stereoの成果として公開しない。
            fail("side_to_mid_db", side, recipe.minimum_side_to_mid_db)
    return {"schema_version": SCHEMA_VERSION, "method": recipe.method, "recipe_id": recipe.recipe_id,
            "passed": not failures, "failures": failures,
            "checked": ["non_finite_samples", "silent_cue", "channel_count", "silent_channel",
                        "loudness_lufs", "true_peak_dbtp", "clipped_samples", key]
            + (["side_to_mid_db"] if recipe.minimum_side_to_mid_db is not None else []),
            "not_checked": ["意味的role(L4)", "分離の漏れ・欠落(L2)", "参照音源との一致"]}
