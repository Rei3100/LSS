"""参照で説明できる残差だけを元Musicへ補うDevelopment Stage。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json

import numpy as np

from lss.bgm.identity import RecordingMatch
from lss.bgm.reference_alignment import ReferenceAlignmentRecipe, align_reference
from lss.bgm.recording_recovery import RecordingRecoveryRecipe, reconstruct_recording


def _fit_recipe() -> RecordingRecoveryRecipe:
    return RecordingRecoveryRecipe(boundary_fade_frames=0, blend=1.0)


@dataclass(frozen=True)
class ReferenceResidualRecipe:
    fit_recipe: RecordingRecoveryRecipe = field(default_factory=_fit_recipe)
    blend: float = 1.0
    boundary_fade_frames: int = 4410
    schema_version: int = 1
    method: str = "matched-reference-projected-source-minus-music"
    alignment_recipe: ReferenceAlignmentRecipe | None = None

    def to_dict(self) -> dict:
        if not isinstance(self.fit_recipe, RecordingRecoveryRecipe):
            raise ValueError("固定した録音係数推定Recipeが必要です")
        self.fit_recipe.to_dict()
        if (self.fit_recipe.blend != 1 or self.fit_recipe.boundary_fade_frames != 0
                or type(self.blend) not in (int, float) or not np.isfinite(self.blend) or not 0 < self.blend <= 1
                or type(self.boundary_fade_frames) is not int or self.boundary_fade_frames < 0
                or type(self.schema_version) is not int or self.schema_version != 1
                or self.method != "matched-reference-projected-source-minus-music"):
            raise ValueError("参照残差回収の規則が不正です")
        if self.alignment_recipe is not None:
            if not isinstance(self.alignment_recipe, ReferenceAlignmentRecipe):
                raise ValueError("固定した参照整列Recipeが必要です")
            self.alignment_recipe.to_dict()
            if self.alignment_recipe.sample_rate != self.fit_recipe.sample_rate:
                raise ValueError("参照整列と回収のrateが一致しません")
        payload = asdict(self)
        if self.alignment_recipe is None:
            # 既存の固定Recipeの同一性は、整列を有効にしない限り変えない。
            payload.pop("alignment_recipe")
        else:
            payload["alignment_recipe"] = self.alignment_recipe.to_dict()
        return payload

    @property
    def recipe_id(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "bgm-reference-residual-v1-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def recover_reference_residual(original: np.ndarray, music: np.ndarray, reference: np.ndarray,
                               match: RecordingMatch, recipe: ReferenceResidualRecipe) -> tuple[np.ndarray, dict]:
    """元Musicを残し、P(original)−P(Music)だけを正の録音一致区間へ加える。

    Pは既存Stageと同じ左右独立の局所FIR係数推定。参照との整列は原本から一度だけ決め、
    元Musicの推定にも同じ整列を使う。参照に無い歌唱等を丸ごと置き換えない。
    原本の台詞・効果音も参照基底へ漏れ得るため、回収の成功や意味的BGMを保証しない。
    呼出側は原本・参照・一致根拠のhash、rateと時間軸を検証する。
    """
    recipe.to_dict()
    source, base, ref = map(np.asarray, (original, music, reference))
    if source.shape != base.shape:
        raise ValueError("原本とMusicの時間軸が一致しません")
    empty = np.zeros_like(base, dtype=np.float64)
    speed_alignment = None
    if recipe.alignment_recipe is not None:
        ref, match, speed_alignment = align_reference(source, ref, match, recipe.alignment_recipe)
    source_projection, alignment = reconstruct_recording(source, empty, ref, match, recipe.fit_recipe)
    start, end = alignment["source_start_frame"], alignment["source_end_frame"]
    rate = recipe.fit_recipe.sample_rate
    aligned_match = replace(match, query_start_seconds=start / rate, query_end_seconds=end / rate,
                            reference_start_seconds=alignment["reference_start_frame"] / rate)
    music_projection, _ = reconstruct_recording(
        base, empty, ref, aligned_match, replace(recipe.fit_recipe, maximum_alignment_refinement_frames=0),
    )
    residual = source_projection[:, start:end] - music_projection[:, start:end]
    length = end - start
    gain = np.full(length, recipe.blend, dtype=np.float64)
    fade = min(recipe.boundary_fade_frames, length // 2)
    if fade:
        ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
        gain[:fade] *= ramp
        gain[-fade:] *= ramp[::-1]
    result = base.astype(np.float64, copy=True)
    result[:, start:end] += residual * gain
    if not np.isfinite(result).all():
        raise ValueError("参照残差回収に非有限値があります")
    observations = {
        "recipe_id": recipe.recipe_id, "artifact_kind": "reference_conditioned_residual_candidate",
        "semantic_role": "unresolved", "publication_allowed": False,
        "source_start_frame": start, "source_end_frame": end,
        "reference_start_frame": alignment["reference_start_frame"],
        "alignment_refinement_frames": alignment["alignment_refinement_frames"],
        "alignment_from_original_only": True, "recovered_energy": float(np.sum(residual ** 2)),
        "reference_sha256": match.reference_sha256,
    }
    if speed_alignment is not None:
        observations["speed_alignment"] = speed_alignment
    return result, observations
