"""言葉で指定した任意の音の分離（SAM Audio base-tv、Meta、SAM License）の製品adapter。

SAM Audioは48kHz monoの音声codec（DACVAE）の潜在空間で、文で指定した対象（target）と残り（residual）を
flow matchingで生成する。生成された波形は元の録音そのものではない（codecの再構成・透かし入り・mono）。
LSSはこれを **そのまま公開しない**。targetとresidualの振幅から時間周波数の比率maskを作り、元の音源の
各channelへ掛けたものだけを使う（mask投影）。公開する音は元の録音の成分の部分集合で、stereoと元の
周波数・帯域を保ち、生成物が新しい音を足すことはない。

固定している実行条件:

- model実装はupstream（facebookresearch/sam-audio のDiT・patcher・rope・align、facebookresearch/dacvae の
  codec）のbytesのまま `vendor/sam_audio` に置き、ここで専用の名前空間へ読み込む。upstreamの
  ``SAMAudio`` 本体（映像・span予測・rankerを含む）は使わず、文で条件付けた分離に要る薄い部分
  （入力の整列、anchor、時刻埋め込み、ODE）を同じ演算・同じパラメータ名でここに書く。
- 重みは ``sam-audio-base-tv`` を bfloat16 にしたsafetensors（SAM Licenseの再配布、匿名取得可）。
  公式 ``facebook/sam-audio-base-tv`` の ``checkpoint.pt`` を bfloat16 へ丸めたものと全1237 tensorが
  一致することを開発時に確かめた。映像encoderの重み（``vision_encoder.*``）は文の条件付けでは
  使わないので読まない。それ以外は **strict** に読む。
- 文の条件付けは `lss.sound.t5_text`（T5-base、upstreamと同じ演算）。
- 入力は左右の平均（mono）を48kHzへ変換したもの。20秒の区間を2秒重ねて並べ、重なりは直線で
  入れ替える。区間ごとの初期noiseは固定seed。ODEはupstream既定の中点法16段。DiTはbfloat16 autocast。
- mask は ``|T|^2 / (|T|^2 + |R|^2)``（STFT 2048点・hop 512・Hann）を元の周波数で計算し、元の
  各channelのSTFTへ掛けて戻す。
- CUDAが無ければ失敗する。CPUや別modelへ無言でfallbackしない。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import threading
import types
from math import gcd
from pathlib import Path
from typing import Callable

from lss.paths import resource_root

PROVIDER = "sam-audio-base-tv"
UPSTREAM = {
    "sam_audio": {"repository": "https://github.com/facebookresearch/sam-audio.git",
                  "commit": "bb4c6999d2677c7402360e426afc01ddfad6dce0", "license": "SAM License",
                  "vendored_files": ["sam_audio/model/transformer.py", "sam_audio/model/patcher.py",
                                     "sam_audio/model/rope.py", "sam_audio/model/align.py"]},
    "dacvae": {"repository": "https://github.com/facebookresearch/dacvae.git",
               "commit": "414c20785fc3a28373073ea8ef7a1316eeeaca6e", "license": "Apache-2.0",
               "vendored_files": ["dacvae/model/dacvae.py", "dacvae/model/base.py", "dacvae/nn/layers.py",
                                  "dacvae/nn/quantize.py", "dacvae/nn/bottleneck.py"]},
    "reimplemented": "SAMAudio text-prompted separate(): align_inputs, EmbedAnchors, SinusoidalEmbedding, "
                     "midpoint ODE (torchdiffeq fixed grid), same parameter names",
}
WEIGHTS = {
    "cache_name": "sam_audio_base_tv",
    "file_name": "sam-audio-base-tv-bf16.safetensors",
    "url": ("https://huggingface.co/AEmotionStudio/sam-audio-models/resolve/"
            "5ee10e39dca0640c7e0c0e9d084b3cc6e26e216f/sam-audio-base-tv-bf16.safetensors"),
    "license": "SAM License (redistribution with license; Meta Platforms)",
    "size_bytes": 3862647436,
    "sha256": "1bdff41444bef343d39ebac40c1534120138f53d779eb161f6cf5eb8b8403f9d",
    "equivalence": {
        "official": "facebook/sam-audio-base-tv@ab0acdd77e90d7d19a2d9bfbeb81cd8042541a5e/checkpoint.pt",
        "relation": "all 1237 tensors equal official.to(torch.bfloat16)",
    },
}
MODEL_CONFIG = {
    "in_channels": 768,
    "audio_codec": {"encoder_dim": 64, "encoder_rates": [2, 8, 10, 12], "latent_dim": 1024, "decoder_dim": 1536,
                    "decoder_rates": [12, 10, 8, 2], "n_codebooks": 16, "codebook_size": 1024,
                    "codebook_dim": 128, "quantizer_dropout": False, "sample_rate": 48000},
    "text_dim": 768,
    "vision_dim": 1024,
    "transformer": {"dim": 2048, "n_heads": 16, "n_layers": 16, "dropout": 0.1, "norm_eps": 1e-05,
                    "qk_norm": True, "fc_bias": False, "ffn_exp": 4, "ffn_dim_multiplier": 1, "multiple_of": 64,
                    "non_linearity": "swiglu", "use_rope": True, "max_positions": 10000,
                    "frequency_embedding_dim": 256, "timestep_non_linearity": "swiglu",
                    "t_block_non_linearity": "silu", "t_block_bias": True, "context_dim": 2048,
                    "context_non_linearity": "swiglu", "context_embedder_dropout": 0.0, "context_norm": False,
                    "out_channels": 256, "in_channels": None},
    "num_anchors": 3,
    "anchor_embedding_dim": 128,
}
SAMPLE_RATE = 48000
# codecは畳み込みだけなので、余白付きの短い区間で符号化・復号して繋いでも値は変わらない（作業memoryを抑える）。
CODEC_BLOCK_SECONDS = 5.0
CODEC_MARGIN_SECONDS = 1.0
INFERENCE = {
    "downmix": "mean_of_channels", "resample": "scipy.signal.resample_poly to 48000 and back",
    "chunk_seconds": 20.0, "overlap_seconds": 2.0, "crossfade": "linear",
    "noise_seed": 0, "ode": {"method": "midpoint", "steps": 16},
    "transformer_autocast": "bfloat16", "codec_precision": "float32",
    "codec_blocks": {"seconds": CODEC_BLOCK_SECONDS, "margin_seconds": CODEC_MARGIN_SECONDS},
    "mask": {"kind": "power_ratio", "power": 2.0, "n_fft": 2048, "hop": 512, "window": "hann"},
}
_SKIPPED_PREFIXES = ("vision_encoder.",)
_LOAD_LOCK = threading.Lock()


class SamAudioError(RuntimeError):
    """分離の実行条件か出力が契約に合わない。"""


def vendor_root() -> Path:
    return resource_root() / "vendor" / "sam_audio"


def adapter_source() -> Path:
    module_source = Path(__file__)
    if not module_source.is_file():
        module_source = resource_root() / "runtime-sources" / "lss" / "sound" / "sam_audio.py"
    return module_source


def implementation_artifacts(*, normalize_text: bool = False) -> tuple[tuple[str, str], ...]:
    from lss.core import runtime
    from lss.sound import t5_text

    text_source = Path(t5_text.__file__)
    if not text_source.is_file():
        text_source = resource_root() / "runtime-sources" / "lss" / "sound" / "t5_text.py"
    return runtime._provider_artifacts((adapter_source(), text_source, vendor_root()), normalize_text=normalize_text)


def engine_revision() -> dict:
    """抽出結果へ影響する選択の全て。GPUへ触れずに決まる。"""
    from lss.sound import t5_text

    body = {
        "schema_version": 1, "provider": PROVIDER, "upstream": UPSTREAM,
        "weights": {key: WEIGHTS[key] for key in ("url", "license", "size_bytes", "sha256")},
        "text_encoder": {"weights": {key: t5_text.WEIGHTS[key] for key in ("url", "size_bytes", "sha256")},
                         "tokenizer": {key: t5_text.TOKENIZER[key] for key in ("url", "size_bytes", "sha256")}},
        "model": MODEL_CONFIG, "inference": INFERENCE,
        "implementation": [{"path": path, "sha256": digest}
                           for path, digest in implementation_artifacts(normalize_text=True)],
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**body, "engine_id": "sam-audio-v1-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]}


def _cached(spec: dict, progress: Callable[[dict], None] | None) -> Path:
    from lss.core import models_store
    from lss.harness import fetch

    path = fetch.model_cache_dir() / spec["cache_name"] / spec["file_name"]
    if fetch.is_cached(path, spec["size_bytes"], spec["sha256"]):
        return path
    if path.is_file():
        path.unlink()
    models_store.ensure_cache_headroom(spec["size_bytes"])

    def report(completed: int, total: int | None) -> None:
        if progress is not None:
            progress({"phase": "downloading_model", "model_id": spec["cache_name"], "asset": spec["file_name"],
                      "completed_bytes": completed, "total_bytes": total})

    return fetch.download(spec["url"], path, expected_size=spec["size_bytes"],
                          expected_sha256=spec["sha256"], progress=report)


def ensure_weights(progress: Callable[[dict], None] | None = None) -> dict[str, Path]:
    """固定した重み一式を、同じ内容のときだけ使う。無ければ固定revisionから取得する。"""
    from lss.sound import t5_text

    return {"model": _cached(WEIGHTS, progress), "text": _cached(t5_text.WEIGHTS, progress),
            "tokenizer": _cached(t5_text.TOKENIZER, progress)}


def required_bytes() -> int:
    from lss.sound import t5_text

    return WEIGHTS["size_bytes"] + t5_text.WEIGHTS["size_bytes"] + t5_text.TOKENIZER["size_bytes"]


def _load_file(name: str, path: Path, package: str | None) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if package is not None:
        module.__package__ = package
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _import_vendored():
    """upstreamのbytesのままのmoduleを、専用の名前空間へ読む。

    upstreamの絶対import（``dacvae.*``、``audiotools``、``huggingface_hub``）は、読み込みの間だけ
    この名前空間のmoduleか推論で使わない部分の代役へ向ける。読み込み後は元に戻す。
    """
    root = vendor_root()
    if not root.is_dir():
        raise SamAudioError(f"SAM Audioの同梱実装がありません: {root}")
    prefix = "_lss_sam_audio"
    if f"{prefix}.loaded" in sys.modules:
        return sys.modules[f"{prefix}.loaded"]
    import torch

    def package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []
        return module

    class _Unused:  # 推論で使わない機能（音声file入出力・重み取得）の代役
        def __init__(self, *args, **kwargs):
            raise SamAudioError("推論では使わない機能が呼ばれました")

    audiotools = types.ModuleType("audiotools")
    audiotools.AudioSignal = _Unused
    audiotools.STFTParams = _Unused
    audiotools_ml = types.ModuleType("audiotools.ml")

    class BaseModel(torch.nn.Module):
        INTERN: list = []
        EXTERN: list = []

    audiotools_ml.BaseModel = BaseModel
    audiotools.ml = audiotools_ml
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = _Unused

    shadowed = ("audiotools", "audiotools.ml", "huggingface_hub", "dacvae", "dacvae.nn", "dacvae.model")
    saved = {name: sys.modules.get(name) for name in shadowed}
    loaded = {}
    try:
        sys.modules.update({"audiotools": audiotools, "audiotools.ml": audiotools_ml, "huggingface_hub": hub})
        for name in ("dacvae", "dacvae.nn", "dacvae.model"):
            sys.modules[name] = package(name)
        sys.modules[f"{prefix}"] = package(prefix)
        sys.modules[f"{prefix}.model"] = package(f"{prefix}.model")
        for name in ("layers", "quantize", "bottleneck"):
            module = _load_file(f"dacvae.nn.{name}", root / "dacvae" / "nn" / f"{name}.py", "dacvae.nn")
            setattr(sys.modules["dacvae.nn"], name, module)
        base = _load_file("dacvae.model.base", root / "dacvae" / "model" / "base.py", "dacvae.model")
        sys.modules["dacvae.model"].base = base
        codec = _load_file("dacvae.model.dacvae", root / "dacvae" / "model" / "dacvae.py", "dacvae.model")
        loaded["DACVAE"] = codec.DACVAE

        config = types.ModuleType(f"{prefix}.model.config")

        class TransformerConfig:
            def __init__(self, **values):
                self.__dict__.update(values)

        config.TransformerConfig = TransformerConfig
        sys.modules[config.__name__] = config
        for name in ("rope", "patcher", "transformer", "align"):
            module = _load_file(f"{prefix}.model.{name}", root / "sam_audio" / "model" / f"{name}.py",
                                f"{prefix}.model")
            loaded[name] = module
        loaded["TransformerConfig"] = TransformerConfig
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for name in [key for key in sys.modules if key.startswith("dacvae.")]:
            sys.modules.pop(name, None)
    result = types.SimpleNamespace(
        DACVAE=loaded["DACVAE"], DiT=loaded["transformer"].DiT,
        AlignModalities=loaded["align"].AlignModalities, TransformerConfig=loaded["TransformerConfig"])
    sys.modules[f"{prefix}.loaded"] = result
    return result


def build_model():
    """upstream ``SAMAudio`` の文条件付き分離に要る部分を、同じパラメータ名で組む。"""
    import torch

    vendored = _import_vendored()
    config = MODEL_CONFIG
    width = config["transformer"]["dim"]

    class SinusoidalEmbedding(torch.nn.Module):
        def __init__(self, dim, theta=10000):
            super().__init__()
            half = dim // 2
            inv_freq = torch.exp(-math.log(theta) * torch.arange(half).float() / half)
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def forward(self, x, pos):
            emb = torch.einsum("i, j -> i j", pos, self.inv_freq)
            return torch.cat((emb.cos(), emb.sin()), dim=-1)

    class EmbedAnchors(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim, out_dim):
            super().__init__()
            self.embed = torch.nn.Embedding(num_embeddings + 1, embedding_dim, padding_idx=num_embeddings)
            self.gate = torch.nn.Parameter(torch.tensor([0.0]))
            self.proj = torch.nn.Linear(embedding_dim, out_dim, bias=False)

        def forward(self, x, anchor_ids, anchor_alignment):
            embs = self.embed(anchor_ids.gather(1, anchor_alignment))
            return x + self.gate.tanh() * self.proj(embs)

    class AudioCodec(torch.nn.Module):
        """upstream ``sam_audio.model.codec.DACVAE``（encoder平均とdecoder）と同じ演算。"""

        def __init__(self):
            super().__init__()
            values = config["audio_codec"]
            model = vendored.DACVAE(**values)
            self.encoder = model.encoder
            self.quantizer = model.quantizer
            self.decoder = model.decoder
            self.hop_length = int(math.prod(values["encoder_rates"]))

        def forward(self, waveform):
            length = waveform.size(-1)
            if length % self.hop_length:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.hop_length - length % self.hop_length), "reflect")
            with torch.backends.cudnn.flags(enabled=False):
                mean, _ = self.quantizer.in_proj(self.encoder(waveform)).chunk(2, dim=1)
            return mean

        def decode(self, frames):
            with torch.backends.cudnn.flags(enabled=False):
                return self.decoder(self.quantizer.out_proj(frames))

    class SamAudioText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_codec = AudioCodec()
            self.transformer = vendored.DiT(vendored.TransformerConfig(**config["transformer"]))
            self.proj = torch.nn.Linear(config["in_channels"], width)
            self.align_masked_video = vendored.AlignModalities(config["vision_dim"], width)
            self.embed_anchors = EmbedAnchors(config["num_anchors"], config["anchor_embedding_dim"], width)
            self.memory_proj = torch.nn.Linear(config["text_dim"], width)
            self.timestep_emb = SinusoidalEmbedding(width)

        def vector_field(self, noisy, time, features, video, text, text_mask, anchor_ids, alignment, pad_mask):
            x = torch.cat([noisy, torch.zeros_like(features), features], dim=2)
            aligned = self.embed_anchors(self.align_masked_video(self.proj(x), video), anchor_ids, alignment)
            timestep = self.timestep_emb(time, pos=time).unsqueeze(1)
            memory = self.memory_proj(text) + timestep
            return self.transformer(aligned, time, padding_mask=pad_mask, memory=memory,
                                    memory_padding_mask=text_mask)

    return SamAudioText()


def load_model(weights: Path, *, device: str = "cuda:0"):
    import torch

    from lss.sound import safetensors_file

    if not torch.cuda.is_available():
        raise SamAudioError("SAM Audioの分離にはCUDAが必要です。CPUへはfallbackしません")
    def to_bfloat16(model) -> None:
        # 配布重みはbfloat16。DiT側はそのまま（autocastと同じ精度）で持ち、codecだけfloat32へ戻す。
        for name, module in model.named_children():
            if name not in {"audio_codec", "timestep_emb"}:
                module.to(torch.bfloat16)

    # parameterは空のままGPUへ置き、重みを1 tensorずつ写す（CPUのmemoryへmodel全体を2重に置かない）。
    model = safetensors_file.build_empty(build_model, device=device, prepare=to_bfloat16)
    names = list(model.state_dict())
    header, _ = safetensors_file.read_header(weights)
    unexpected = [name for name in header if name not in set(names) and not name.startswith(_SKIPPED_PREFIXES)]
    if unexpected:
        raise SamAudioError(f"重みに想定外のtensorがあります: {unexpected[:5]}")
    safetensors_file.copy_into(model, weights, names=names)
    model = model.eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise SamAudioError("silent CPU fallbackを検出しました")
    return model


def _resample(values, source: int, target: int):
    import numpy as np
    from scipy.signal import resample_poly

    if source == target:
        return np.asarray(values, dtype=np.float32)
    divisor = gcd(source, target)
    return resample_poly(values, target // divisor, source // divisor, axis=0).astype(np.float32)


class Separator:
    """読み込んだmodelで、文ごとに target/residual の推定を作り、mask投影をする。"""

    def __init__(self, paths: dict[str, Path], *, device: str = "cuda:0"):
        from lss.sound import t5_text

        with _LOAD_LOCK:
            self.model = load_model(paths["model"], device=device)
            self.text = t5_text.TextConditioner(paths["text"], paths["tokenizer"], device=device)
        self.device = device

    def close(self) -> None:
        import torch

        self.model = None
        self.text = None
        torch.cuda.empty_cache()

    def _blocks(self, frames: int):
        hop = self.model.audio_codec.hop_length
        block = max(1, int(CODEC_BLOCK_SECONDS * SAMPLE_RATE / hop))
        margin = int(CODEC_MARGIN_SECONDS * SAMPLE_RATE / hop)
        for start in range(0, frames, block):
            end = min(frames, start + block)
            yield start, end, max(0, start - margin), min(frames, end + margin)

    def _encode(self, audio):
        """(1, 1, samples) → codecの平均（1, 128, frames）。余白付きの区間ごと。"""
        import torch

        codec = self.model.audio_codec
        hop = codec.hop_length
        total = audio.shape[-1]
        frames = math.ceil(total / hop)
        parts = []
        for start, end, low, high in self._blocks(frames):
            features = codec(audio[..., low * hop:min(total, high * hop)])
            parts.append(features[..., start - low:end - low])
        return torch.cat(parts, dim=-1)

    def _decode(self, latents):
        """(batch, 128, frames) → (batch, 1, frames*hop)。余白付きの区間ごと。"""
        import torch

        codec = self.model.audio_codec
        hop = codec.hop_length
        parts = []
        for start, end, low, high in self._blocks(latents.shape[-1]):
            waves = codec.decode(latents[..., low:high])
            parts.append(waves[..., (start - low) * hop:(end - low) * hop])
        return torch.cat(parts, dim=-1)

    def _separate_chunk(self, mono48, hidden, text_mask, seed: int, spans=()):
        """``spans`` は区間の先頭からの秒（開始, 終了）。その時刻を ``+`` のanchorにする（upstreamの
        ``predict_spans`` と同じ条件付け: anchor ids ``[<null>, <pad>, +]``、区間外は ``<null>``）。"""
        import torch

        model = self.model
        hop = model.audio_codec.hop_length
        size = mono48.shape[-1]
        frames = math.ceil(size / hop)
        audio = mono48.reshape(1, 1, -1)
        with torch.inference_mode():
            features = self._encode(audio).transpose(1, 2)
            features = torch.cat([features, features], dim=2)[:, :frames]
            batch, length, channels = features.shape
            video = features.new_zeros(batch, MODEL_CONFIG["vision_dim"], length)
            alignment = torch.zeros((batch, length), dtype=torch.long, device=self.device)
            if spans:
                anchor_ids = torch.tensor([[0, 3, 1]], device=self.device)  # <null>, <pad>, +
                for start, end in spans:
                    first = max(0, math.ceil(start * SAMPLE_RATE / hop))
                    last = min(length, math.ceil(end * SAMPLE_RATE / hop))
                    if last > first:
                        alignment[:, first:last] = 2
            else:
                anchor_ids = torch.tensor([[0, 3]], device=self.device)  # <null>, <pad>
            pad_mask = torch.ones((batch, length), dtype=torch.bool, device=self.device)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            state = torch.randn(features.shape, generator=generator, device=self.device, dtype=features.dtype)
            steps = INFERENCE["ode"]["steps"]
            step = 1.0 / steps

            def field(t, x):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model.vector_field(x, t.expand(batch), features, video, hidden, text_mask,
                                             anchor_ids, alignment, pad_mask)
                return out.float()

            for index in range(steps):
                t0 = torch.tensor(index * step, device=self.device)
                half = state + field(t0, state) * (step / 2)
                state = state + field(t0 + step / 2, half) * step
            generated = state.transpose(1, 2)
            half_channels = channels // 2
            with torch.random.fork_rng(devices=[torch.device(self.device)]):
                torch.manual_seed(seed)
                waves = self._decode(generated.reshape(2 * batch, half_channels, length)).view(batch, 2, -1)
            target = waves[0, 0, :size].float()
            residual = waves[0, 1, :size].float()
        return target, residual

    def estimate(self, mono, rate: int, prompt: str, *, progress: Callable[[int, int], None] | None = None,
                 spans=()):
        """mono（元の周波数）から、元の周波数の target/residual 推定（numpy float32）を返す。

        ``spans`` は対象が鳴っている時刻（音の先頭からの秒）。与えると各区間の該当時刻をanchorにする。"""
        import numpy as np
        import torch

        hidden, text_mask = self.text([prompt])
        mono48 = torch.from_numpy(_resample(mono, rate, SAMPLE_RATE)).to(self.device)
        total = mono48.shape[-1]
        size = int(INFERENCE["chunk_seconds"] * SAMPLE_RATE)
        ramp = int(INFERENCE["overlap_seconds"] * SAMPLE_RATE)
        hop = size - ramp
        starts = [0] if total <= size else list(range(0, max(1, total - ramp), hop))
        target = torch.zeros(total, device=self.device)
        residual = torch.zeros(total, device=self.device)
        weight = torch.zeros(total, device=self.device)
        for count, start in enumerate(starts, start=1):
            end = min(start + size, total)
            if end - start < SAMPLE_RATE // 10 and count > 1:
                break
            piece = mono48[start:end]
            if float(piece.abs().max()) == 0.0:
                t = torch.zeros_like(piece)
                r = torch.zeros_like(piece)
            else:
                offset = start / SAMPLE_RATE
                local = tuple((max(0.0, a - offset), min((end - start) / SAMPLE_RATE, b - offset))
                              for a, b in spans if b > offset and a < end / SAMPLE_RATE)
                t, r = self._separate_chunk(piece, hidden, text_mask, INFERENCE["noise_seed"] + count - 1, local)
            window = torch.ones(end - start, device=self.device)
            fade = min(ramp, end - start)
            if start > 0:
                window[:fade] = torch.linspace(0.0, 1.0, fade + 2, device=self.device)[1:-1]
            if end < total:
                window[-fade:] = torch.minimum(window[-fade:],
                                               torch.linspace(1.0, 0.0, fade + 2, device=self.device)[1:-1])
            target[start:end] += t * window
            residual[start:end] += r * window
            weight[start:end] += window
            if progress is not None:
                progress(count, len(starts))
            if end >= total:
                break
        weight = weight.clamp_min(1e-6)
        target = (target / weight).cpu().numpy()
        residual = (residual / weight).cpu().numpy()
        length = len(mono)
        out_t = _resample(target, SAMPLE_RATE, rate)[:length]
        out_r = _resample(residual, SAMPLE_RATE, rate)[:length]
        if len(out_t) < length:
            out_t = np.pad(out_t, (0, length - len(out_t)))
            out_r = np.pad(out_r, (0, length - len(out_r)))
        if not (np.isfinite(out_t).all() and np.isfinite(out_r).all()):
            raise SamAudioError("分離の推定に有限でない値があります")
        return out_t, out_r


def power_mask(target, residual, *, device: str = "cuda:0"):
    """target/residual推定（元の周波数のmono）からの比率mask（周波数, frame）。"""
    import torch

    spec = INFERENCE["mask"]
    window = torch.hann_window(spec["n_fft"], device=device)

    def magnitude(values):
        x = torch.from_numpy(values).to(device)
        return torch.stft(x, spec["n_fft"], spec["hop"], window=window, return_complex=True).abs() ** spec["power"]

    t, r = magnitude(target), magnitude(residual)
    return t / (t + r + 1e-10)


def project(parent, target, residual, *, device: str = "cuda:0", block_seconds: float = 30.0, rate: int,
            floor: float = 0.0, whole_seconds=None):
    """長い音でも一定のmemoryで、mask投影（``power_mask`` → ``apply_mask``）と同じ値を作る。

    STFTは窓の中だけの計算なので、区間の前後へ ``n_fft`` の倍の余白を付けて計算し、余白を捨てて繋ぐ。
    ``floor`` はmaskの下限（楽器: 言葉の分離が対象の一部を周波数ごとに落とすのを抑える）。``whole_seconds``
    （秒ごとのbool）が真の秒はmaskを1にする（親にその音しか鳴っていない秒は、親をそのまま使う）。
    """
    import numpy as np
    import torch

    spec = INFERENCE["mask"]
    margin = 2 * spec["n_fft"]
    block = max(int(block_seconds * rate), 4 * spec["n_fft"])
    total = len(parent)
    out = np.zeros_like(parent, dtype=np.float32)
    for start in range(0, total, block):
        end = min(total, start + block)
        low, high = max(0, start - margin), min(total, end + margin)
        mask = power_mask(np.ascontiguousarray(target[low:high]), np.ascontiguousarray(residual[low:high]),
                          device=device)
        if floor > 0.0:
            mask = torch.clamp(mask, min=float(floor))
        if whole_seconds is not None and len(whole_seconds):
            # STFTの各frameの中心の秒（center=True: frame k は low + k*hop）
            centers = (low + np.arange(mask.shape[-1]) * spec["hop"]) // rate
            flags = np.asarray(whole_seconds, dtype=bool)[np.minimum(centers, len(whole_seconds) - 1)]
            keep = torch.from_numpy(flags.astype(np.float32)).to(mask.device)
            mask = mask * (1.0 - keep)[None, :] + keep[None, :]
        piece = apply_mask(np.ascontiguousarray(parent[low:high]), mask, device=device)
        out[start:end] = piece[start - low:end - low]
        del mask
    return out


def apply_mask(audio, mask, *, device: str = "cuda:0"):
    """元の音（frames, channels）の各channelへmaskを掛けて戻す。"""
    import torch

    spec = INFERENCE["mask"]
    window = torch.hann_window(spec["n_fft"], device=device)
    x = torch.from_numpy(audio.T.copy()).to(device)
    stft = torch.stft(x, spec["n_fft"], spec["hop"], window=window, return_complex=True)
    frames = min(stft.shape[-1], mask.shape[-1])
    stft = stft[..., :frames] * mask[None, :, :frames]
    out = torch.istft(stft, spec["n_fft"], spec["hop"], window=window, length=audio.shape[0])
    return out.T.contiguous().cpu().numpy()
