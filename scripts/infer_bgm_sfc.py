"""SFC-Locoformer（DnR事前学習）でDialogue/Music/Effectsを推定する専用adapter。

BandIt用adapter（`scripts/infer_bgm_bandit.py`）と同じ契約で動く。入力の同一性を確認し、
CUDAで無言のfallbackをせず、stemごとのWAVとruntime識別子付きのreceiptを残す。

    .providers/sfc-locoformer/.venv/Scripts/python.exe scripts/infer_bgm_sfc.py \
        --model .providers/sfc-locoformer/model_weights/dnr/locoformer-medium... \
        --out .state/validation/<record>/sfc-medium \
        <source.flac> [<source.flac> ...]

固定している実行条件（receiptへ記録する）:

- DnR事前学習は `n_chan: 1` のmono modelなので、stereo入力はchannelごとに別推論し、
  元のchannel順で組み直す。joint stereo処理ではない。
- 上流の既定どおり bfloat16 autocast で推論する。
- 自己注意のkernelは `EFFICIENT_ATTENTION` を使う。上流既定の `FLASH_ATTENTION` は
  head_dim=16 / 現行GPUの組み合わせで「No available kernel」になるため選べない。
  同じ計算を別kernelで行う差であり、無言でCPUや別modelへ落ちることはしない。
- stem順は実測で確定した `est0=speech, est1=music, est2=effects`。
  （music-onlyでest1、dialogue-onlyでest0、sfx-onlyでest2が優勢）
- `mamba_ssm` はこの環境ではstubで、SFC-CA経路は構築しない。
- `--block-seconds` を与えると、長尺入力を前後に文脈を足した区間へ分けて推論し、
  内側だけを繋ぐ。29分の全長を一度に渡すとCUDAのメモリが尽きるため。区間の継ぎ目は
  crossfadeしないので、この出力は**解析用の信号**であり公開する成果物ではない。
  区間処理を使ったかはreceiptへ記録する。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STEM_ORDER = ("speech", "music", "effects")
ATTENTION_BACKEND = "EFFICIENT_ATTENTION"
CHANNEL_POLICY = "independent_channel_inference_recomposed_in_source_order"
AUTOCAST_DTYPE = "bfloat16"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--css-segment-size", type=int, default=None)
    parser.add_argument("--css-shift-size", type=int, default=None)
    parser.add_argument("--block-seconds", type=float, default=None,
                        help="長尺を区間へ分けて推論する。継ぎ目をcrossfadeしないので解析用")
    parser.add_argument("--block-margin-seconds", type=float, default=12.0,
                        help="各区間の前後へ足す文脈。推論後に切り落とす")
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from torch.nn.attention import SDPBackend
    from hydra.utils import instantiate
    from aiaccel.config import load_config
    from spectral_feature_compression.core.model.bslocoformer import MultiHeadSelfAttention

    if not torch.cuda.is_available():
        raise RuntimeError("BGM推論にはCUDAが必要です")

    model_dir = args.model.resolve()
    checkpoint_dir = model_dir / "checkpoints"
    config_path = model_dir / "merged_config.yaml"
    artifacts = {
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "checkpoints": sorted(
            {"path": str(p), "sha256": sha256(p)} for p in checkpoint_dir.iterdir() if p.is_file()
        ),
    }
    config = load_config(config_path)
    if int(config.n_src) != len(STEM_ORDER):
        raise ValueError(f"stem数が想定と違います: {config.n_src}")
    if int(config.n_chan) != 1:
        raise ValueError("このadapterはmono modelのchannelごと推論を前提にしています")

    model = instantiate(config.task).to("cuda")
    model.load_average_checkpoint(checkpoint_dir)
    model.eval()
    separator = model.ema_model.module if model.use_ema_model else model.model
    if args.css_segment_size is not None:
        separator.css_segment_size = args.css_segment_size
    if args.css_shift_size is not None:
        separator.css_shift_size = args.css_shift_size
    patched = 0
    for module in separator.modules():
        if isinstance(module, MultiHeadSelfAttention):
            module.sdpa_kernel = getattr(SDPBackend, ATTENTION_BACKEND)
            patched += 1
    if not patched:
        raise RuntimeError("自己注意moduleが見つからず、kernelを固定できません")
    if any(parameter.device.type != "cuda" for parameter in separator.parameters()):
        raise RuntimeError("silent CPU fallbackを検出しました")

    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for source in args.sources:
        source = source.resolve()
        source_hash = sha256(source)
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        if rate != int(config.sr):
            raise ValueError(f"入力のsample rateがmodelと違います: {rate} != {config.sr}")
        if not np.isfinite(audio).all():
            raise ValueError("入力に有限でない値があります")
        frames, channels = audio.shape
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        stems = np.zeros((len(STEM_ORDER), frames, channels), dtype=np.float32)
        block = int(round(args.block_seconds * rate)) if args.block_seconds else frames
        margin = int(round(args.block_margin_seconds * rate)) if args.block_seconds else 0

        def run(segment: np.ndarray) -> np.ndarray:
            wav = torch.from_numpy(segment.T.copy()).unsqueeze(0).to("cuda")
            with torch.inference_mode(), torch.autocast(
                "cuda", enabled=True, dtype=getattr(torch, AUTOCAST_DTYPE),
            ):
                value = (
                    separator.css(wav, ref=None)
                    if model.css_validation and wav.shape[-1] >= separator.css_segment_size * rate
                    else separator(wav)
                )
            out = value[0].float().cpu().numpy()
            del wav, value
            torch.cuda.empty_cache()
            return out

        for channel in range(channels):
            for start in range(0, frames, block):
                end = min(frames, start + block)
                low = max(0, start - margin)
                high = min(frames, end + margin)
                estimate = run(audio[low:high, channel : channel + 1])
                if estimate.shape != (len(STEM_ORDER), 1, high - low):
                    raise ValueError(f"推定の形が不正です: {estimate.shape}")
                stems[:, start:end, channel] = estimate[:, 0, start - low:end - low]
                del estimate
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        case_root = args.out / source.stem
        case_root.mkdir()
        written = {}
        for index, role in enumerate(STEM_ORDER):
            values = stems[index]
            if values.shape != audio.shape or not np.isfinite(values).all():
                raise ValueError(f"{role} stemが不正です")
            path = case_root / f"{role}.wav"
            sf.write(path, values, rate, subtype="FLOAT")
            written[role] = {"path": str(path.resolve()), "sha256": sha256(path)}
        if sha256(source) != source_hash:
            raise ValueError("推論中にsourceが変わりました")
        records.append({
            "source": {"path": str(source), "sha256": source_hash,
                       "frames": frames, "channels": channels, "sample_rate": rate},
            "stems": written, "seconds": seconds,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        })
        print(json.dumps({"case": source.stem, "seconds": round(seconds, 2)}, ensure_ascii=False), flush=True)
        torch.cuda.empty_cache()

    receipt = {
        "schema_version": 1,
        "status": "COMPLETED_CUDA_WITHOUT_FALLBACK",
        "provider": {
            "name": "sfc-locoformer",
            "repository": "https://github.com/b-sigpro/spectral-feature-compression.git",
            "weights_repo": "kohei0209/spectral_feature_compression",
            "weights_revision": "0827e70a9e0c6f21b9b46cb9dfe4f7e4ebd307c5",
            "license": "MIT",
            "artifacts": artifacts,
        },
        "exact_recipe": {
            "n_src": int(config.n_src), "n_chan": int(config.n_chan), "sample_rate": int(config.sr),
            "n_fft": int(config.n_fft), "hop_length": int(config.hop_length),
            "css_segment_size": separator.css_segment_size,
            "css_shift_size": separator.css_shift_size,
            "stem_order": list(STEM_ORDER),
            "channel_policy": CHANNEL_POLICY,
            "autocast_dtype": AUTOCAST_DTYPE,
            "attention_backend": ATTENTION_BACKEND,
            "attention_modules_patched": patched,
            "block_seconds": args.block_seconds,
            "block_margin_seconds": args.block_margin_seconds if args.block_seconds else None,
            "block_join": "no_crossfade_analysis_signal" if args.block_seconds else None,
            "mamba_ssm": "lss-stub (SFC-CA path does not construct Mamba)",
            "use_ema_model": bool(model.use_ema_model),
        },
        "adapter_sha256": sha256(Path(__file__).resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "numpy": np.__version__,
        },
        "cases": records,
    }
    path = args.out / "receipt.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"receipt: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
