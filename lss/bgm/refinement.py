"""ステレオ混合と分離済みstemを使う、明示的なDevelopment後処理。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import istft, stft


@dataclass(frozen=True)
class SpatialRefinementRecipe:
    """局所空間共分散によるWiener再推定。意味役割は判断しない。"""

    n_fft: int = 2048
    hop_frames: int = 512
    covariance_frames: int = 65
    iterations: int = 1
    relative_regularization: float = 1e-6
    block_frames: int = 262144
    schema_version: int = 1
    method: str = "local-stereo-wiener"
    residual_policy: str = "separate-unexplained-source"

    def validate(self) -> None:
        if self.schema_version != 1 or self.method != "local-stereo-wiener":
            raise ValueError("未対応の空間refinement Recipeです")
        if self.residual_policy != "separate-unexplained-source":
            raise ValueError("未説明の残差を自動でMusicへ戻すことはできません")
        for name in ("n_fft", "hop_frames", "covariance_frames", "iterations", "block_frames"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}は正の整数である必要があります")
        if self.n_fft < 4 or self.n_fft % 2:
            raise ValueError("n_fftは4以上の偶数である必要があります")
        if self.hop_frames > self.n_fft // 2:
            raise ValueError("Hann窓のhop_framesはn_fftの半分以下である必要があります")
        if self.block_frames % self.hop_frames:
            raise ValueError("block_framesはhop_framesの整数倍である必要があります")
        if self.covariance_frames % 2 != 1:
            raise ValueError("covariance_framesは奇数である必要があります")
        value = self.relative_regularization
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or not 0 < value < 1
        ):
            raise ValueError("relative_regularizationは0より大きく1未満の有限値が必要です")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        return f"bgm-spatial-v1-{digest}"


@dataclass(frozen=True)
class SpatialRefinementResult:
    recipe_id: str
    stems: Mapping[str, np.ndarray]
    unexplained: np.ndarray


def _audio(value: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(value)
    if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] == 0:
        raise ValueError(f"{name}は空でない(2, frames)のステレオ配列が必要です")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError(f"{name}は実数音声である必要があります")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}に有限でないsampleがあります")
    return values.astype(np.float64, copy=False)


def _local_mean(value: np.ndarray, frames: int) -> np.ndarray:
    if np.iscomplexobj(value):
        return (
            uniform_filter1d(value.real, frames, axis=1, mode="nearest")
            + 1j * uniform_filter1d(value.imag, frames, axis=1, mode="nearest")
        )
    return uniform_filter1d(value, frames, axis=1, mode="nearest")


def _refine_block(
    mixture: np.ndarray,
    stems: Mapping[str, np.ndarray],
    recipe: SpatialRefinementRecipe,
) -> SpatialRefinementResult:
    """入力を変更せず同じsample数の候補stemを返す。正規化・clip・resampleはしない。

    推定stemの局所空間共分散と時変powerから2x2 Wiener filterを作る。
    raw-sum(stems)も独立したsourceとして扱い、残差をMusicへ強制配分しない。
    各iterationは事前に固定する。結果の良否による本番用停止判断ではない。
    """
    recipe.validate()
    raw = _audio(mixture, "mixture")
    if not stems:
        raise ValueError("refinementには分離済みstemが必要です")
    names = list(stems)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("stem名が不正です")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("stem名が重複しています")
    arrays = [_audio(stems[name], name) for name in names]
    if any(value.shape != raw.shape for value in arrays):
        raise ValueError("全stemは元のステレオ音源と同じsample数が必要です")

    # 短い入力でも窓長を暗黙に変更せず、明示した窓で往復する。
    frames = raw.shape[1]
    padded_frames = max(frames, recipe.n_fft)
    options = dict(
        window="hann", nperseg=recipe.n_fft,
        noverlap=recipe.n_fft - recipe.hop_frames,
        nfft=recipe.n_fft, boundary="zeros", padded=True,
    )

    def transform(value: np.ndarray) -> np.ndarray:
        padded = np.pad(value, ((0, 0), (0, padded_frames - frames)))
        # frequency, time, channel
        return stft(padded, **options)[2].transpose(1, 2, 0)

    x = transform(raw)
    y = [transform(value) for value in arrays]
    y.append(x - sum(y))
    for _ in range(recipe.iterations):
        covariances = []
        for source in y:
            power = np.mean(np.abs(source) ** 2, axis=-1)
            cross = source[..., :, None] * source[..., None, :].conj()
            # E[yy^H] / E[v]はPSDを保つ局所power-weighted空間モデル。
            mean_power = np.maximum(_local_mean(power, recipe.covariance_frames), 0)
            spatial = np.zeros_like(cross)
            np.divide(
                _local_mean(cross, recipe.covariance_frames),
                mean_power[..., None, None], out=spatial,
                where=mean_power[..., None, None] > 0,
            )
            covariances.append(power[..., None, None] * spatial)
        covariance = sum(covariances)
        # 低level入力にも同じfilterを適用するため、正則化は相対値のみ。
        scale = np.maximum(np.trace(covariance, axis1=-2, axis2=-1).real / 2, 0)
        diagonal = recipe.relative_regularization * scale
        covariance[..., 0, 0] += diagonal
        covariance[..., 1, 1] += diagonal
        # silent binのinverseを使わない。singularなstereoもHermitian pinvで扱う。
        inverse = np.linalg.pinv(covariance, rcond=1e-12, hermitian=True)
        filtered_mix = np.einsum("ftcd,ftd->ftc", inverse, x)
        y = [np.einsum("ftcd,ftd->ftc", value, filtered_mix) for value in covariances]
        if any(not np.isfinite(value).all() for value in y):
            raise ValueError("空間refinementに有限でない値が発生しました")

    def restore(value: np.ndarray) -> np.ndarray:
        restored = istft(
            value.transpose(2, 0, 1), window="hann", nperseg=recipe.n_fft,
            noverlap=recipe.n_fft - recipe.hop_frames,
            nfft=recipe.n_fft, input_onesided=True, boundary=True,
        )[1][:, :frames].copy()
        if restored.shape != raw.shape or not np.isfinite(restored).all():
            raise ValueError("空間refinementが元のtimelineを保持していません")
        return restored

    return SpatialRefinementResult(
        recipe_id=recipe.recipe_id,
        stems=dict(zip(names, (restore(value) for value in y[:-1]))),
        unexplained=restore(y[-1]),
    )


def refine_stereo(
    mixture: np.ndarray,
    stems: Mapping[str, np.ndarray],
    recipe: SpatialRefinementRecipe,
) -> SpatialRefinementResult:
    """有界のSTFT blockで候補を生成し、各blockの中央を同じtimelineへ戻す。

    反復と局所共分散の依存範囲を含めたcontextを確保する。クロスフェードで
    不連続を隠さず、hop gridを保った中央cropだけを使用する。
    """
    recipe.validate()
    raw = _audio(mixture, "mixture")
    if not stems:
        raise ValueError("refinementには分離済みstemが必要です")
    names = list(stems)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("stem名が不正です")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("stem名が重複しています")
    arrays = {name: _audio(stems[name], name) for name in names}
    if any(value.shape != raw.shape for value in arrays.values()):
        raise ValueError("全stemは元のステレオ音源と同じsample数が必要です")
    if raw.shape[1] <= recipe.block_frames:
        return _refine_block(raw, arrays, recipe)
    radius = recipe.iterations * (
        recipe.covariance_frames // 2 * recipe.hop_frames + recipe.n_fft
    )
    context = ((radius + recipe.hop_frames - 1) // recipe.hop_frames) * recipe.hop_frames
    outputs = {name: np.empty_like(raw) for name in names}
    unexplained = np.empty_like(raw)
    for start in range(0, raw.shape[1], recipe.block_frames):
        end = min(start + recipe.block_frames, raw.shape[1])
        left, right = max(0, start - context), min(raw.shape[1], end + context)
        result = _refine_block(
            raw[:, left:right], {name: value[:, left:right] for name, value in arrays.items()}, recipe,
        )
        crop = slice(start - left, end - left)
        for name in names:
            outputs[name][:, start:end] = result.stems[name][:, crop]
        unexplained[:, start:end] = result.unexplained[:, crop]
    return SpatialRefinementResult(recipe.recipe_id, outputs, unexplained)
