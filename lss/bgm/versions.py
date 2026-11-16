"""同じ楽曲の素材（旋律・和声の進行）を共有する別の録音の証拠。同じ録音かどうかとは別の判断。

`identity.py` は波形まで説明できる同じ録音だけを `known_exact` にする。ここは、キー・テンポ・音色・
ミックスが違っても、和声の並びが長く対応するかを測る（クロマの相互再帰と局所整列、Qmax）。
対応しても同じ録音・同じ曲とは言わない（アレンジ違い・変奏・同じ動機を使う別の曲を含み得る）。
候補を捨てたり、まとめたりする根拠にもしない。

点数は、比べた参照録音の集まりの中での外れ具合（中央値と中央絶対偏差によるz）で判断する。
Development: 作品4つのリリース音源62件で、作品をまたぐ825組の両方向の小さい方のzは最大3.8だった
（`.state/validation/bgm-versions-references-r3.json`）。しきい値はその上に余裕を取って決めた。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

METHOD_VERSION = 1
SAMPLE_RATE = 11025
N_FFT = 4096
HOP = 2205  # 5フレーム/秒。さらに2フレームずつ平均して2.5フレーム/秒で整列する
POOL = 2
_LOW_HZ, _HIGH_HZ = 65.0, 2100.0
KAPPA = 0.1  # 相互再帰: 互いの近傍の上位10%に入る組だけを対応とみなす
GAP_OPEN, GAP_EXTEND = 5.0, 0.5
MIN_COHORT = 20  # 集まりが小さいと中央絶対偏差が小さく、偶然の対応でzが大きく出る（合成で11件ではz 8.3）
# 参照録音どうし（両方向のzの小さい方）と、区間から参照録音への片方向のz。
SHARED_MIN_Z = 4.5
SEGMENT_MIN_Z = 6.0
MIN_SEGMENT_SECONDS = 20.0


def _pitch_class_matrix() -> np.ndarray:
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    matrix = np.zeros((12, freqs.size), dtype=np.float32)
    valid = (freqs >= _LOW_HZ) & (freqs <= _HIGH_HZ)
    midi = 69.0 + 12.0 * np.log2(np.where(valid, freqs, 440.0) / 440.0)
    nearest = np.round(midi)
    weight = np.clip(1.0 - 2.0 * np.abs(midi - nearest), 0.0, 1.0)
    for bin_index in np.flatnonzero(valid):
        matrix[int(nearest[bin_index]) % 12, bin_index] = weight[bin_index]
    return matrix


_PITCH = _pitch_class_matrix()
FRAME_SECONDS = HOP * POOL / SAMPLE_RATE


def chroma(samples: np.ndarray) -> np.ndarray:
    """時間×12のクロマ。各フレームを最大値で正規化し、2フレームずつ平均する（約0.4秒ごと）。"""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < N_FFT:
        return np.zeros((0, 12), dtype=np.float32)
    frames = 1 + (samples.size - N_FFT) // HOP
    window = np.hanning(N_FFT).astype(np.float32)
    out = np.empty((frames, 12), dtype=np.float32)
    for begin in range(0, frames, 512):
        end = min(frames, begin + 512)
        index = np.arange(begin, end)[:, None] * HOP + np.arange(N_FFT)[None, :]
        power = np.abs(np.fft.rfft(samples[index] * window, axis=1)) ** 2
        out[begin:end] = power @ _PITCH.T
    peak = out.max(axis=1, keepdims=True)
    out = np.where(peak > 1e-9, out / np.maximum(peak, 1e-9), 0.0)
    usable = out.shape[0] // POOL * POOL
    return out[:usable].reshape(-1, POOL, 12).mean(axis=1).astype(np.float32)


def _transposition(query: np.ndarray, reference: np.ndarray) -> int:
    """全体のクロマの形が最も合うように参照を回す半音数（Optimal Transposition Index）。"""
    q = query.mean(axis=0)
    r = reference.mean(axis=0)
    q = q / max(float(q.max()), 1e-9)
    r = r / max(float(r.max()), 1e-9)
    return int(np.argmax([np.dot(q, np.roll(r, shift)) for shift in range(12)]))


def _recurrence(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    rotated = np.roll(reference, _transposition(query, reference), axis=1)
    qn = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-9)
    rn = rotated / np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-9)
    distance = 1.0 - qn @ rn.T
    n, m = distance.shape
    k_rows, k_cols = max(1, int(KAPPA * m)), max(1, int(KAPPA * n))
    rows = distance <= np.partition(distance, min(k_rows, m - 1), axis=1)[:, min(k_rows, m - 1):min(k_rows, m - 1) + 1]
    cols = distance <= np.partition(distance, min(k_cols, n - 1), axis=0)[min(k_cols, n - 1):min(k_cols, n - 1) + 1, :]
    silent = (query.sum(axis=1) <= 0)[:, None] | (reference.sum(axis=1) <= 0)[None, :]
    return rows & cols & ~silent


def qmax(query: np.ndarray, reference: np.ndarray) -> float:
    """相互再帰の上で、テンポの揺れ（1:2, 2:1の歩み）を許した最長の対応の強さ。"""
    if query.shape[0] < 2 or reference.shape[0] < 2:
        return 0.0
    recurrence = _recurrence(query, reference)
    n, m = recurrence.shape
    score = np.zeros((n + 2, m + 2), dtype=np.float32)
    hit = np.zeros((n + 2, m + 2), dtype=bool)
    hit[2:, 2:] = recurrence
    best = 0.0
    for i in range(2, n + 2):
        previous = np.maximum(np.maximum(score[i - 1, 1:-1], score[i - 2, 1:-1]), score[i - 1, :-2])
        penalty = np.where(hit[i - 1, 1:-1], GAP_OPEN, GAP_EXTEND)
        score[i, 2:] = np.where(hit[i, 2:], previous + 1.0, np.maximum(0.0, previous - penalty))
        best = max(best, float(score[i, 2:].max()))
    return best


def _robust_z(value: float, cohort: np.ndarray) -> float:
    median = float(np.median(cohort))
    spread = 1.4826 * float(np.median(np.abs(cohort - median)))
    return (value - median) / max(spread, 1.0)


@dataclass(frozen=True)
class SharedMaterial:
    first_id: str
    second_id: str
    state: str  # shared_material_candidate / not_shared / undetermined
    score: float
    z_first_to_second: float | None
    z_second_to_first: float | None
    method_version: int = METHOD_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def library_relations(features: dict[str, np.ndarray]) -> list[SharedMaterial]:
    """参照録音どうしの関係。両方向とも集まりの中で外れて高いときだけ候補にする。"""
    ids = sorted(features)
    scores = np.zeros((len(ids), len(ids)), dtype=np.float64)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i != j:
                scores[i, j] = qmax(features[a], features[b])
    results = []
    enough = len(ids) - 1 >= MIN_COHORT
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            z_ij = z_ji = None
            state = "undetermined"
            if enough:
                z_ij = _robust_z(scores[i, j], np.delete(scores[i], i))
                z_ji = _robust_z(scores[j, i], np.delete(scores[j], j))
                state = "shared_material_candidate" if min(z_ij, z_ji) >= SHARED_MIN_Z else "not_shared"
            results.append(SharedMaterial(ids[i], ids[j], state, round(float(max(scores[i, j], scores[j, i])), 2),
                                          None if z_ij is None else round(z_ij, 2),
                                          None if z_ji is None else round(z_ji, 2)))
    return results


def segment_relations(segment: np.ndarray, references: dict[str, np.ndarray]) -> list[dict]:
    """番組の区間のクロマを参照録音の集まりと比べ、集まりの中で外れて高い参照だけを返す（高い順）。"""
    if segment.shape[0] * FRAME_SECONDS < MIN_SEGMENT_SECONDS or len(references) < MIN_COHORT:
        return []
    ids = sorted(references)
    scores = np.array([qmax(segment, references[reference]) for reference in ids])
    found = []
    for index, reference in enumerate(ids):
        z = _robust_z(scores[index], np.delete(scores, index))
        if z >= SEGMENT_MIN_Z:
            found.append({"reference_id": reference, "score": round(float(scores[index]), 2), "z": round(z, 2),
                          "state": "shared_material_candidate", "method_version": METHOD_VERSION})
    return sorted(found, key=lambda item: -item["z"])
