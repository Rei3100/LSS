"""原本の複数区間から、既知録音参照の小さな速度差を推定するDevelopment Stage。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

import numpy as np
from scipy.signal import correlate, resample_poly

from lss.bgm.identity import RecordingMatch
from lss.bgm.recording_recovery import _stereo


@dataclass(frozen=True)
class ReferenceAlignmentRecipe:
    sample_rate: int = 44100
    speed_numerators: tuple[int, ...] = (1998, 1999, 2000, 2001, 2002)
    speed_denominator: int = 2000
    probe_frames: int = 44100
    probe_count: int = 5
    minimum_probe_frames: int = 512
    maximum_offset_frames: int = 220
    minimum_explained: float = 0.1
    minimum_explained_gain: float = 0.02
    probe_policy: str = "source-nonzero-energy-evenly-spaced-tiles-v1"
    schema_version: int = 1
    method: str = "source-only-common-stereo-speed-and-offset-v1"

    def to_dict(self) -> dict:
        for name in ("sample_rate", "speed_denominator", "probe_frames", "probe_count", "minimum_probe_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("参照整列には正の整数のrate・窓・速度分母が必要です")
        if (not isinstance(self.speed_numerators, (tuple, list)) or not self.speed_numerators
                or any(type(n) is not int or n <= 0 or abs(n / self.speed_denominator - 1) > 0.01
                       for n in self.speed_numerators)
                or len(set(self.speed_numerators)) != len(self.speed_numerators)
                or self.speed_denominator not in self.speed_numerators
                or type(self.maximum_offset_frames) is not int or self.maximum_offset_frames < 0
                or self.probe_count < 3 or self.minimum_probe_frames > self.probe_frames
                or self.probe_policy != "source-nonzero-energy-evenly-spaced-tiles-v1"
                or type(self.schema_version) is not int or self.schema_version != 1
                or self.method != "source-only-common-stereo-speed-and-offset-v1"):
            raise ValueError("固定した速度仮説・共通時間ずれの規則が不正です")
        for value in (self.minimum_explained, self.minimum_explained_gain):
            if type(value) not in (float, int) or not np.isfinite(value) or not 0 < value < 1:
                raise ValueError("整列の存在と改善の判定値が不正です")
        payload = asdict(self)
        payload["speed_numerators"] = list(self.speed_numerators)
        return payload

    @property
    def recipe_id(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "bgm-reference-alignment-v1-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _probes(source, start, length, recipe):
    width = min(recipe.probe_frames, length // recipe.probe_count)
    if width < recipe.minimum_probe_frames:
        return width, []
    positions = list(range(0, length - width + 1, width))
    if positions[-1] != length - width:
        positions.append(length - width)
    # 音がない長い前後を除き、音のある全区間から分散して選ぶ。正解stemは使わない。
    active = [p for p in positions if np.sum(source[:, start + p:start + p + width] ** 2) > 1e-20]
    if len(active) > recipe.probe_count:
        indices = np.linspace(0, len(active) - 1, recipe.probe_count).round().astype(int)
        active = [active[i] for i in indices]
    return width, active


def _score(source, reference, start, reference_start, width, positions, recipe):
    if len(positions) < 3:
        return {"explained": 0.0, "reference_offset_frames": 0, "usable_probes": 0}
    shifts = np.arange(-recipe.maximum_offset_frames, recipe.maximum_offset_frames + 1)
    measured = []
    for position in positions:
        nominal = reference_start + position
        # 範囲外をゼロで補い、それを参照との一致根拠にしない。
        available = (nominal + shifts >= 0) & (nominal + shifts + width <= reference.shape[1])
        if not np.any(available):
            continue
        first, last = np.flatnonzero(available)[[0, -1]]
        values = reference[:, nominal + shifts[first]:nominal + shifts[last] + width]
        query = source[:, start + position:start + position + width]
        # 左右の独立した係数・極性を許す。時間軸と速度仮説は左右で共有する。
        explained = np.zeros(len(shifts))
        source_energy = float(np.sum(query ** 2))
        if source_energy <= 1e-20:
            continue
        for channel in range(2):
            dots = correlate(values[channel], query[channel], mode="valid", method="fft")
            cumulative = np.concatenate(([0.0], np.cumsum(values[channel] ** 2)))
            reference_energy = cumulative[width:] - cumulative[:-width]
            explained[first:last + 1] += dots ** 2 / np.maximum(reference_energy, 1e-20) / source_energy
        measured.append(np.minimum(explained, 1.0))
    if len(measured) < 3:
        return {"explained": 0.0, "reference_offset_frames": 0, "usable_probes": len(measured)}
    scores = np.median(measured, axis=0)
    # 同点では、根拠なく時間軸を動かさない最小のずれを選ぶ。
    order = np.argsort(np.abs(shifts), kind="stable")
    best = int(order[np.argmax(scores[order])])
    return {"explained": float(scores[best]), "reference_offset_frames": int(shifts[best]),
            "usable_probes": len(measured)}


def align_reference(original: np.ndarray, reference: np.ndarray, match: RecordingMatch,
                    recipe: ReferenceAlignmentRecipe) -> tuple[np.ndarray, RecordingMatch, dict]:
    """正の既知録音を原本だけで整列する。原本の音量・左右・時間軸は変更しない。

    正解stem・元Music・採点値は速度の選択に渡さない。これは録音同一性や意味的正解の
    新しい証明ではない。弱い改善・短い区間・無音は元の参照と一致時刻をそのまま返す。
    """
    recipe.to_dict()
    source, ref = map(_stereo, (original, reference))
    times = (match.query_start_seconds, match.query_end_seconds, match.reference_start_seconds)
    if (match.identity_state != "known_exact" or any(type(v) not in (float, int) or not np.isfinite(v) for v in times)):
        raise ValueError("参照整列には正の録音一致と有限の時刻が必要です")
    start, end, ref_start = [round(v * recipe.sample_rate) for v in times]
    if not 0 <= start < end <= source.shape[1] or not 0 <= ref_start < ref.shape[1]:
        raise ValueError("参照整列の時刻が原本と参照の範囲から外れています")
    # 全仮説に共通する観測区間を固定し、仮説ごとに都合のよいprobeへ変えない。
    length = min(end - start, *(math.ceil(ref.shape[1] * n / recipe.speed_denominator)
                              - round(ref_start * n / recipe.speed_denominator)
                              for n in recipe.speed_numerators))
    width, positions = _probes(source, start, length, recipe)
    hypotheses = []
    for numerator in recipe.speed_numerators:
        divisor = math.gcd(numerator, recipe.speed_denominator)
        transformed = ref if numerator == recipe.speed_denominator else resample_poly(
            ref, numerator // divisor, recipe.speed_denominator // divisor, axis=1,
            window=("kaiser", 5.0), padtype="constant")
        shifted_start = round(ref_start * numerator / recipe.speed_denominator)
        score = _score(source, transformed, start, shifted_start, width, positions, recipe)
        hypotheses.append({"speed_numerator": numerator, **score})
    baseline = next(row for row in hypotheses if row["speed_numerator"] == recipe.speed_denominator)
    best = max(hypotheses, key=lambda row: (row["explained"],
                                          -abs(row["speed_numerator"] - recipe.speed_denominator)))
    gain = best["explained"] - baseline["explained"]
    selected = (best["speed_numerator"] != recipe.speed_denominator
                and best["explained"] >= recipe.minimum_explained and gain >= recipe.minimum_explained_gain)
    observations = {"recipe_id": recipe.recipe_id, "publication_allowed": False, "semantic_role": "unresolved",
                    "alignment_from_original_only": True, "speed_correction_used": selected,
                    "speed_numerator": best["speed_numerator"] if selected else recipe.speed_denominator,
                    "speed_denominator": recipe.speed_denominator, "explained_gain": gain,
                    "probe_start_frames": [start + p for p in positions], "probe_frames": width,
                    "hypotheses": hypotheses, "reason": "material_source_alignment_gain" if selected
                    else "retain_reference_without_material_alignment_gain"}
    if not selected:
        return ref, match, observations
    numerator = best["speed_numerator"]
    divisor = math.gcd(numerator, recipe.speed_denominator)
    aligned = resample_poly(ref, numerator // divisor, recipe.speed_denominator // divisor, axis=1,
                            window=("kaiser", 5.0), padtype="constant")
    corrected_start = round(ref_start * numerator / recipe.speed_denominator) + best["reference_offset_frames"]
    # 参照の外側にある端を合成しない。実在する共通区間だけに一致時刻を制限する。
    head = max(0, -corrected_start)
    corrected_end = min(end, start + aligned.shape[1] - corrected_start)
    corrected = replace(match, query_start_seconds=(start + head) / recipe.sample_rate,
                        query_end_seconds=corrected_end / recipe.sample_rate,
                        reference_start_seconds=(corrected_start + head) / recipe.sample_rate)
    observations["reference_offset_frames"] = best["reference_offset_frames"]
    return aligned, corrected, observations
