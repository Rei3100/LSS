"""単話BGM候補の解析器（SFC-Locoformer、DnR事前学習の台詞・音楽・効果音3分類）の製品adapter。

`lss.bgm.episode_cues` が区間を見つけるための解析用3 stemを、製品の常駐workerの中で作る。
開発時は別の実行環境（`.providers/sfc-locoformer`）から `scripts/infer_bgm_sfc.py` で
作っていたものと同じ計算を、同梱したmodel実装（`vendor/sfc_locoformer`、MIT、upstreamの
bytesのまま）で行う。配布物にhydra/aiaccel/lightningは要らない。

固定している実行条件（開発時のadapterと同じ。解析の実測はこの条件で行った）:

- DnR事前学習は `n_chan: 1` のmono modelなので、stereo入力はchannelごとに別推論し、
  元のchannel順で組み直す。joint stereo処理ではない。
- EMA重み（`ema_model.module`）を使い、上流の既定どおり bfloat16 autocast で推論する。
- 自己注意のkernelは `EFFICIENT_ATTENTION`。上流既定のFLASHは現行GPUで選べない。
- 区間の中は、upstreamのCSSと同じ12秒の区間を6秒ずつずらして同じmodelへ通し、重なりをupstreamと同じ
  平坦な平均で繋ぐ。ただしupstreamは6秒ごとに平均する相手が不連続に切り替わり、実番組でその位置の4割に
  段差（クリック）が出るので、区間の両端の ``css_edge_ramp_seconds`` だけ重みを直線で下げる（台形窓、
  `_windowed_css`）。それ以外のsampleはupstreamと同じ値になる（全体を三角窓にすると推定が全sampleで変わり、
  閉じ済みの候補Stageの採否が制御素材で動いた）。
- 長尺は120秒の区間へ前後12秒の文脈を足して推論し、内側を繋ぐ。隣り合う区間の継ぎ目の前後
  ``block_crossfade_seconds`` の半分ずつは、両方の区間の推定（文脈として計算済み）を線形に入れ替える。
  crossfadeせずに繋ぐと、実番組で継ぎ目に二階差分が周囲の中央値の10〜98倍の段差（クリック）が出て、
  公開するcueがそれをまたぐ（`.state/validation/bgm-semantic-disposition-r1/`）。
- 解析器の周波数は44.1kHz固定。元の周波数が違えば解析用の複製を作り、変換を記録する。
  原本の区間（raw/no-op候補）は解析用の複製ではなく元の周波数の復号から切る。

CUDAが無ければ失敗する。CPUや別modelへ無言でfallbackしない。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from math import gcd
from pathlib import Path
from typing import Callable

from lss.bgm import gpu_execution
from lss.paths import resource_root

PROVIDER = "sfc-locoformer"
SAMPLE_RATE = 44100
STEM_ORDER = ("speech", "music", "effects")
UPSTREAM = {
    "repository": "https://github.com/b-sigpro/spectral-feature-compression.git",
    "commit": "bf61e1e37471d7a28bc5727d3d07d6cb8b177370",
    "license": "MIT",
    "vendored_files": [
        "spectral_feature_compression/core/model/bandit_split.py",
        "spectral_feature_compression/core/model/bslocoformer.py",
        "spectral_feature_compression/core/model/crossattn_enc_dec.py",
        "spectral_feature_compression/core/model/enc_dec_base.py",
        "spectral_feature_compression/core/model/model_wrapper.py",
    ],
}
_WEIGHTS_REVISION = "0827e70a9e0c6f21b9b46cb9dfe4f7e4ebd307c5"
_WEIGHTS_DIRECTORY = ("models/dnr/locoformer-medium.enc-crossattn96dim.dec-crossattn96dim."
                      "musical64.learnable-query")
WEIGHTS = {
    "cache_name": "sfc_locoformer_dnr_medium",
    "file_name": "model.pth",
    "url": (f"https://huggingface.co/kohei0209/spectral_feature_compression/resolve/"
            f"{_WEIGHTS_REVISION}/{_WEIGHTS_DIRECTORY}/checkpoints/model.pth"),
    "repository": "kohei0209/spectral_feature_compression",
    "revision": _WEIGHTS_REVISION,
    "size_bytes": 128207890,
    "sha256": "b878ea0ffe29c67401727d51cdaab8db0394178d8e0eafa9d2ca5e241fd472e2",
    "state_prefix": "ema_model.module.",
}
# 配布物の `merged_config.yaml`（sha256 4d0d1ffc…）のmodel節をそのまま書き写したもの。
_BAND = {
    "d_inner": 96, "d_model": 128, "n_chan": 1, "sample_rate": SAMPLE_RATE, "n_fft": 2048,
    "n_bands": 64, "band_config": "musical", "query_type": "learnable", "n_heads": 4,
    "slope": [1, 1, 1, 1], "learnable_slope": False, "learnable_pos_bias": True,
    "mask_outside_bands": False,
}
NETWORK = {
    "wrapper": {"n_fft": 2048, "hop_length": 512, "fs": SAMPLE_RATE, "scaling": True,
                "css_segment_size": 12, "css_shift_size": 6},
    "separator": {
        "n_src": 3, "n_chan": 1, "n_layers": 6, "emb_dim": 128, "norm_type": "rmsgroupnorm",
        "num_groups": 8, "tf_order": "ft", "n_heads": 8, "flash_attention": True,
        "attention_dim": 128, "ffn_type": ["swiglu_conv1d", "swiglu_conv1d"],
        "ffn_hidden_dim": [192, 192], "conv1d_kernel": 8, "conv1d_shift": 1, "dropout": 0.1,
        "eps": 1.0e-5,
    },
    "encoder": dict(_BAND),
    "decoder": {**_BAND, "n_src": 3},
}
INFERENCE = {
    "channel_policy": "independent_channel_inference_recomposed_in_source_order",
    "autocast_dtype": "bfloat16",
    "attention_backend": "EFFICIENT_ATTENTION",
    "css": "adapter_css_same_segments_flat_average_with_edge_ramps",
    "css_edge_ramp_seconds": 0.25,
    "block_seconds": 120.0,
    "block_margin_seconds": 12.0,
    "block_join": "linear_crossfade_across_adjacent_block_contexts",
    "block_crossfade_seconds": 1.0,
    "output_subtype": "FLOAT",
}
ANALYSIS_COPY = {"method": "scipy.signal.resample_poly", "subtype": "FLOAT",
                 "rate": SAMPLE_RATE, "channels": "unchanged"}


class SfcAnalysisError(RuntimeError):
    """解析器の実行条件か出力が契約に合わない。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vendor_root() -> Path:
    return resource_root() / "vendor" / "sfc_locoformer"


def adapter_source() -> Path:
    """凍結時はPYZ内のmoduleに物理パスが無いので、同梱した同じsourceを返す。"""
    module_source = Path(__file__)
    if not module_source.is_file():
        module_source = resource_root() / "runtime-sources" / "lss" / "bgm" / "sfc_analysis.py"
    return module_source


def implementation_artifacts(*, normalize_text: bool = False) -> tuple[tuple[str, str], ...]:
    """``normalize_text`` は改行の違いで変わらないRecipe識別用（解析器の版は生のbytes）。"""
    from lss.core import runtime

    return runtime._provider_artifacts((adapter_source(), vendor_root()), normalize_text=normalize_text)


def _execution_policy_source() -> Path:
    source = Path(gpu_execution.__file__)
    if not source.is_file():
        source = resource_root() / "runtime-sources" / "lss" / "bgm" / "gpu_execution.py"
    return source


def analyzer_revision(execution_profile: str | None = None) -> dict:
    """解析結果へ影響する選択の全て。GPUへ触れずに決まる。"""
    profile = gpu_execution.resolve_execution_profile(execution_profile)
    body = {
        "schema_version": 1, "provider": PROVIDER, "upstream": UPSTREAM,
        "weights": {key: WEIGHTS[key] for key in ("url", "repository", "revision", "size_bytes",
                                                  "sha256", "state_prefix")},
        "network": NETWORK, "inference": INFERENCE, "stem_order": list(STEM_ORDER),
        "sample_rate": SAMPLE_RATE, "analysis_copy": ANALYSIS_COPY,
        "implementation": [{"path": path, "sha256": digest}
                           for path, digest in implementation_artifacts()],
    }
    if profile is not None:
        body["execution_profile"] = profile.name
        body["execution_policy"] = gpu_execution.profile_identity(profile.name)
        from lss.core import runtime
        body["execution_policy_implementation"] = {
            "path": "lss/bgm/gpu_execution.py",
            "sha256": runtime.text_source_sha256(_execution_policy_source()),
        }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**body, "analyzer_id": "bgm-sfc-analysis-v1-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]}


def weights_path() -> Path:
    from lss.harness import fetch

    return fetch.model_cache_dir() / WEIGHTS["cache_name"] / WEIGHTS["file_name"]


def ensure_weights(progress: Callable[[dict], None] | None = None) -> Path:
    """固定した重みを、同じ内容のときだけ使う。無ければ固定revisionから取得する。"""
    from lss.core import models_store
    from lss.harness import fetch

    path = weights_path()
    if fetch.is_cached(path, WEIGHTS["size_bytes"], WEIGHTS["sha256"]):
        return path
    if path.is_file():
        # 内容が違う実体は使えない。残したまま取得すると容量を二重に数える。
        path.unlink()
    models_store.ensure_cache_headroom(WEIGHTS["size_bytes"])

    def report(completed: int, total: int | None) -> None:
        if progress is not None:
            progress({"phase": "downloading_model", "model_id": WEIGHTS["cache_name"],
                      "asset": "weight", "completed_bytes": completed, "total_bytes": total})

    return fetch.download(WEIGHTS["url"], path, expected_size=WEIGHTS["size_bytes"],
                          expected_sha256=WEIGHTS["sha256"], progress=report)


def analysis_copy(decoded: Path, target: Path) -> dict:
    """解析器の周波数へ合わせた複製。すでにその周波数なら値を変えずに複製する。"""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    data, rate = sf.read(str(decoded), dtype="float32", always_2d=True)
    if not np.isfinite(data).all():
        raise SfcAnalysisError("解析する音声に有限でない値があります")
    if rate == SAMPLE_RATE:
        converted, method = data, "none"
    else:
        divisor = gcd(SAMPLE_RATE, rate)
        up, down = SAMPLE_RATE // divisor, rate // divisor
        converted = resample_poly(data, up, down, axis=0).astype(np.float32)
        method = f"scipy.signal.resample_poly up={up} down={down}"
    target = Path(target)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        sf.write(str(temporary), converted, SAMPLE_RATE, subtype="FLOAT", format="WAV")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"decoded_rate": int(rate), "analysis_rate": SAMPLE_RATE, "resample": method,
            "channels": int(data.shape[1]), "frames": int(converted.shape[0]),
            "sha256": _sha256(target)}


def _import_vendored():
    root = str(vendor_root())
    if not vendor_root().is_dir():
        raise SfcAnalysisError(f"解析器の同梱実装がありません: {vendor_root()}")
    if root not in sys.path:
        sys.path.insert(0, root)
    from spectral_feature_compression.core.model.bslocoformer import (  # type: ignore
        BSLocoformer, MultiHeadSelfAttention)
    from spectral_feature_compression.core.model.crossattn_enc_dec import (  # type: ignore
        CrossAttnDecoder, CrossAttnEncoder)
    from spectral_feature_compression.core.model.model_wrapper import ModelWrapper  # type: ignore
    return BSLocoformer, MultiHeadSelfAttention, CrossAttnEncoder, CrossAttnDecoder, ModelWrapper


def load_separator(weights: Path, *, device: str = "cuda:0"):
    """同梱実装へ固定重みを厳密に読み込み、推論条件を固定した分離器を返す。"""
    import torch
    from torch.nn.attention import SDPBackend

    BSLocoformer, MultiHeadSelfAttention, CrossAttnEncoder, CrossAttnDecoder, ModelWrapper = (
        _import_vendored())
    separator = ModelWrapper(
        BSLocoformer(CrossAttnEncoder(**NETWORK["encoder"]), CrossAttnDecoder(**NETWORK["decoder"]),
                     **NETWORK["separator"]),
        **NETWORK["wrapper"],
    )
    state = torch.load(str(weights), map_location="cpu", weights_only=False)
    prefix = WEIGHTS["state_prefix"]
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise SfcAnalysisError("重みに解析器のEMA状態がありません")
    separator.load_state_dict(selected, strict=True)
    separator.to(device).eval()
    patched = 0
    for module in separator.modules():
        if isinstance(module, MultiHeadSelfAttention):
            module.sdpa_kernel = getattr(SDPBackend, INFERENCE["attention_backend"])
            patched += 1
    if not patched:
        raise SfcAnalysisError("自己注意moduleが見つからず、kernelを固定できません")
    if any(parameter.device.type != "cuda" for parameter in separator.parameters()):
        raise SfcAnalysisError("silent CPU fallbackを検出しました")
    return separator, patched


def _windowed_css(separator, wav):
    """upstreamの ``ModelWrapper.css`` と同じ区間の切り方・同じmodel呼び出し・同じ平坦な平均で重ね合わせ、
    区間の両端だけ重みを直線で下げて、平均の相手が替わる位置を連続にする。

    ``wav`` は (1, 1, frames)。区間は ``css_segment_size`` 秒、``css_shift_size`` 秒ずつずらす（upstreamと同じ個数・
    同じ末尾の0詰め）。無音の区間は推論せず0とする（upstreamと同じ）。重みは区間の端から
    ``css_edge_ramp_seconds`` までは0から1へ直線、それ以外は1。重みの和で割るので、端から離れたsampleは
    upstreamの平坦な平均と同じ値になる。
    """
    import torch

    rate = separator.fs
    size = int(round(separator.css_segment_size * rate))
    shift = int(round(separator.css_shift_size * rate))
    frames = wav.shape[-1]
    if frames <= size:
        return separator(wav, ref=None)
    overlap = size - shift
    count = -(-(frames - overlap) // shift)
    positions = torch.arange(size, device=wav.device, dtype=torch.float32)
    ramp = max(1.0, INFERENCE["css_edge_ramp_seconds"] * rate)
    window = (torch.clamp(torch.minimum(positions + 0.5, size - positions - 0.5) / ramp, max=1.0)
              if overlap > 0 else torch.ones_like(positions))
    total = None
    weight = torch.zeros(frames, device=wav.device, dtype=torch.float32)
    for index in range(count):
        start = index * shift
        end = min(start + size, frames)
        segment = wav.new_zeros(wav.shape[:-1] + (size,))
        segment[..., :end - start] = wav[..., start:end]
        if float(segment.abs().sum()) == 0.0:
            estimate = None
        else:
            estimate = separator(segment, ref=None)[..., :size].float()
            if total is None:
                total = torch.zeros(estimate.shape[:-1] + (frames,), device=wav.device, dtype=torch.float32)
            total[..., start:end] += estimate[..., :end - start] * window[:end - start]
        weight[start:end] += window[:end - start]
    if total is None:
        return wav.new_zeros((wav.shape[0], NETWORK["separator"]["n_src"], wav.shape[1], frames))
    return total / weight


def block_ranges(frames: int, rate: int = SAMPLE_RATE) -> list[tuple[int, int]]:
    """推論のブロック格子（各ブロックの内側の [開始, 終了) frame）。前後の文脈はこの外側に足す。"""
    block = int(round(INFERENCE["block_seconds"] * rate))
    return [(start, min(frames, start + block)) for start in range(0, frames, block)]


def blocks_for(spans_seconds, frames: int, rate: int = SAMPLE_RATE) -> list[int]:
    """区間 [開始秒, 終了秒) と重なるブロックの番号。"""
    chosen = set()
    for index, (start, end) in enumerate(block_ranges(frames, rate)):
        for low, high in spans_seconds:
            if low * rate < end and high * rate > start:
                chosen.add(index)
    return sorted(chosen)


def separate_stems(analysis_path: Path, destination: Path, weights: Path, *,
                   device: str = "cuda:0",
                   progress: Callable[[dict], None] | None = None,
                   blocks=None,
                   execution_profile: str | None = None) -> dict:
    """解析用の複製から3 stemを作り、destinationへ確定する。stemごとのsha256を返す。

    ``blocks`` を渡すと、そのブロック番号だけを全体と同じ格子・同じ前後の文脈で推論し、
    他の区間は0にする（音楽の出力のもう一度の解析で、必要なブロックだけを通すため）。
    """
    import numpy as np
    import soundfile as sf
    import torch

    profile = gpu_execution.resolve_execution_profile(execution_profile)
    gpu_execution.assert_profile_active(execution_profile, torch)
    analysis_path, destination = Path(analysis_path).resolve(), Path(destination).resolve()
    before = _sha256(analysis_path)
    audio, rate = sf.read(str(analysis_path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise SfcAnalysisError(f"解析用の音声は{SAMPLE_RATE}Hzである必要があります: {rate}")
    if not np.isfinite(audio).all():
        raise SfcAnalysisError("解析する音声に有限でない値があります")
    separator, patched = load_separator(weights, device=device)
    cadence_events: list[dict] = []
    hook = gpu_execution.attach_forward_cadence(
        separator, torch, execution_profile, cadence_events, progress=progress)
    try:
        autocast = getattr(torch, INFERENCE["autocast_dtype"])
        frames, channels = audio.shape
        margin = int(round(INFERENCE["block_margin_seconds"] * rate))
        segment_frames = separator.css_segment_size * rate
        stems = np.zeros((len(STEM_ORDER), frames, channels), dtype=np.float32)
        grid = block_ranges(frames, rate)
        selected = list(range(len(grid))) if blocks is None else sorted(set(int(b) for b in blocks))
        if any(index < 0 or index >= len(grid) for index in selected):
            raise SfcAnalysisError(f"ブロック番号が範囲外です: {selected}")
        total = channels * len(selected)
        done = 0

        def run(segment: np.ndarray) -> np.ndarray:
            wav = torch.from_numpy(segment.T.copy()).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=autocast):
                value = _windowed_css(separator, wav) if wav.shape[-1] >= segment_frames else separator(wav)
            out = value[0].float().cpu().numpy()
            del wav, value
            torch.cuda.empty_cache()
            return out

        half = int(round(INFERENCE["block_crossfade_seconds"] * rate / 2))
        if not 0 < half <= margin:
            raise SfcAnalysisError("継ぎ目のcrossfadeは文脈の長さ以内である必要があります")
        ramp = ((np.arange(2 * half, dtype=np.float32) + 0.5) / (2 * half))[None, :]
        for channel in range(channels):
            previous = None  # (直前に推論したブロック番号, その推定, その推定の先頭frame)
            for index in selected:
                start, end = grid[index]
                low, high = max(0, start - margin), min(frames, end + margin)
                estimate = run(audio[low:high, channel:channel + 1])
                if estimate.shape != (len(STEM_ORDER), 1, high - low):
                    raise SfcAnalysisError(f"推定の形が不正です: {estimate.shape}")
                stems[:, start:end, channel] = estimate[:, 0, start - low:end - low]
                if (previous is not None and previous[0] == index - 1 and start - half >= 0
                        and start + half <= high):
                    # 継ぎ目 start の前後を、前のブロックの推定から今のブロックの推定へ線形に入れ替える。
                    _, prior, prior_low = previous
                    stems[:, start - half:start + half, channel] = (
                        prior[:, 0, start - half - prior_low:start + half - prior_low] * (1.0 - ramp)
                        + estimate[:, 0, start - half - low:start + half - low] * ramp)
                previous = (index, estimate, low)
                done += 1
                if progress is not None:
                    progress({"phase": "separating", "provider": PROVIDER,
                              "completed_blocks": done, "total_blocks": total})
    finally:
        gpu_execution.remove_forward_hook(hook)
        del separator
        torch.cuda.empty_cache()
    if _sha256(analysis_path) != before:
        raise SfcAnalysisError("推論中に解析用の音声が変わりました")
    destination.mkdir(parents=True, exist_ok=True)
    written = {}
    for index, role in enumerate(STEM_ORDER):
        values = stems[index]
        if values.shape != audio.shape or not np.isfinite(values).all():
            raise SfcAnalysisError(f"{role} stemが不正です")
        target = destination / f"{role}.wav"
        temporary = destination / f".{role}.{os.getpid()}.tmp"
        try:
            sf.write(str(temporary), values, rate, subtype=INFERENCE["output_subtype"], format="WAV")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        written[role] = {"path": str(target), "sha256": _sha256(target)}
    result = {"analysis_sha256": before, "frames": int(frames), "channels": int(channels),
            "blocks": None if blocks is None else selected,
            "sample_rate": int(rate), "stems": written, "attention_modules_patched": patched}
    if profile is not None:
        result["execution_profile"] = profile.name
        result["cadence_events"] = cadence_events
    return result
