"""言葉で書いた音が、音のどの時刻に聞こえるか（PE-A-Frame、Meta、Apache-2.0）の製品adapter。

PE-A-Frame は音と文を同じ空間へ写す対照学習のmodelで、文には1つの埋め込み、音には40 msごとの埋め込みを
作る。両者の内積を学習済みの尺度と偏りで確率（sigmoid）にすると、「その文の音がその40 msに鳴っている」
確率になる。SAM Audio の上流は、このmodelで対象の時刻（span）を予測して分離の条件（anchor）に使う。

LSSでの役割:

- **語彙に縛られない検証**: AudioSetのクラスに無い言葉（英語の名詞句・訳した日本語）でも、取り出した音に
  その音があるかを確かめられる。
- **時刻の選別**: 特定の効果音・声のように、ある時だけ鳴る音の「鳴っている時刻」を40 ms単位で決める。
- **分離の条件**: SAM Audio へ、対象が鳴っている時刻を ``+`` のanchorとして渡す（上流の ``predict_spans``）。

固定している実行条件（upstream: facebookresearch/perception_models ``core/audio_visual_encoder`` と
transformers の ModernBERT/DAC と同じ演算・同じパラメータ名。製品はtransformersを同梱しない）:

- 重みは ``facebook/pe-a-frame-<size>`` の ``perception_models`` 版 ``model.safetensors``（float32、匿名取得可）。
  全tensorを **strict** に読む（文のencoderは使う22層まで。使わない層とMLMの頭は読まない）。
- 文: ``tokenizer.json`` のbyte-level BPE（NFC、GPT-2の分割規則、``[CLS] ... [SEP]``）。文は
  `lss.sound.t5_text.normalized_prompt` を通したASCIIの英語だけを受ける（その範囲では分割規則が厳密に一致）。
  ModernBERT-large の22層目の出力（``hidden_states[22]``）の先頭token → ``text_head``。
- 音: 左右の平均（mono）を48 kHzへ変換し、hop 1920 の倍数まで反射で詰める。DAC encoder（cuDNN無効、
  float32）→ VAEの平均（128次元）→ ``data_proj`` → 先頭にcls tokenを置いたResNet埋め込み → Qwen3形の
  transformer（RMSNorm・q/kのRMSNorm・交互配置のRoPE θ=20000）→ ``audio_head``。
- 長い音は ``WINDOW_SECONDS`` 秒の窓を ``OVERLAP_SECONDS`` 秒重ねて並べ、重なりは窓の中央に近い方の値を使う
  （各窓の端の文脈不足を避ける）。DAC encoderは畳み込みなので、余白付きの区間で計算して繋ぐ。
- CUDAが無ければ失敗する。CPUや別modelへ無言でfallbackしない。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Callable

PROVIDER = "pe-a-frame"
# base を使う: 制御混合（2 fold）で、時刻のanchorによる分離の改善・有無の判定とも large と同等以上
# （判定はbaseのほうが良い）で、取得 2.5 GB・GPU 2.4 GB（large は 6.1 GB・5.4 GB）。
SIZE = "base"
_REVISIONS = {"large": "40187271298f84e2966d4518c88dde698540c9ad",
              "base": "4852459e815078af79bb0fa63889fb39381e4ded"}
REPOSITORY = "https://huggingface.co/facebook/pe-a-frame-{size}"
WEIGHTS = {
    "large": {"cache_name": "pe_a_frame_large", "file_name": "model.safetensors",
              "size_bytes": 6121569400,
              "sha256": "cb1b7d596f1765e6fe21707f7c78989b06e08bc5692ffa88868ad266cce65660"},
    "base": {"cache_name": "pe_a_frame_base", "file_name": "model.safetensors",
             "size_bytes": 2537856824,
             "sha256": "913733810fbb35d907e2562961ed40f08964bf25f2b1695888c54edb61941b54"},
}
TOKENIZER = {"file_name": "tokenizer.json", "size_bytes": 2132967,  # 両sizeで同じbytes
             "sha256": "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1"}
LICENSE = "Apache-2.0 (facebook/pe-a-frame model card; facebookresearch/perception_models)"
# 文のencoder（ModernBERT-large）。両sizeで共通。
TEXT_CONFIG = {"vocab_size": 50368, "hidden_size": 1024, "intermediate_size": 2624, "num_heads": 16,
               "used_layers": 22, "global_every": 3, "local_window": 128, "global_rope_theta": 160000.0,
               "local_rope_theta": 10000.0, "norm_eps": 1e-5, "cls_id": 50281, "sep_id": 50282,
               "max_tokens": 512}
AUDIO_CONFIGS = {
    "large": {"hidden_size": 1792, "intermediate_size": 4800, "layers": 28, "heads": 14, "head_dim": 128},
    "base": {"hidden_size": 1024, "intermediate_size": 2752, "layers": 16, "heads": 8, "head_dim": 128},
}
DAC_CONFIG = {"encoder_hidden_size": 64, "downsampling_ratios": (2, 8, 10, 12), "hidden_size": 1024,
              "codebook_dim": 128}
SAMPLE_RATE = 48000
HOP = 1920                       # 40 ms
FRAMES_PER_SECOND = SAMPLE_RATE // HOP
OUTPUT_DIM = 1024
RMS_EPS = 1e-5
MAX_POSITIONS = 10000
WINDOW_SECONDS = 10.0
OVERLAP_SECONDS = 2.0
# DAC encoderは畳み込みだけなので、余白付きの短い区間で計算して繋いでも値は変わらない（作業memoryを抑える）。
CODEC_BLOCK_SECONDS = 5.0
CODEC_MARGIN_SECONDS = 1.0
_LOAD_LOCK = threading.Lock()


class FrameDetectorError(RuntimeError):
    """PE-A-Frameの入力・重み・出力が契約に合わない。"""


def weights_spec(size: str = SIZE) -> dict:
    spec = dict(WEIGHTS[size])
    spec["url"] = f"{REPOSITORY.format(size=size)}/resolve/{_REVISIONS[size]}/{spec['file_name']}"
    spec["license"] = LICENSE
    return spec


def tokenizer_spec(size: str = SIZE) -> dict:
    return {"cache_name": WEIGHTS[size]["cache_name"], "file_name": TOKENIZER["file_name"],
            "url": f"{REPOSITORY.format(size=size)}/resolve/{_REVISIONS[size]}/{TOKENIZER['file_name']}",
            "size_bytes": TOKENIZER["size_bytes"], "sha256": TOKENIZER["sha256"], "license": LICENSE}


def adapter_source() -> Path:
    module_source = Path(__file__)
    if module_source.suffix == ".py" and module_source.is_file():
        return module_source
    from lss.paths import resource_root

    return resource_root() / "runtime-sources" / "lss" / "sound" / "pe_frame.py"


def engine_revision(size: str = SIZE) -> dict:
    from lss.core import runtime

    body = {"provider": f"{PROVIDER}-{size}", "weights_sha256": WEIGHTS[size]["sha256"],
            "tokenizer_sha256": TOKENIZER["sha256"], "revision": _REVISIONS[size],
            "window_seconds": WINDOW_SECONDS, "overlap_seconds": OVERLAP_SECONDS,
            "adapter_sha256": runtime.text_source_sha256(adapter_source())}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {**body, "engine_id": "pe-frame-v1-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]}


def ensure_weights(progress: Callable[[dict], None] | None = None, size: str = SIZE) -> dict[str, Path]:
    from lss.sound import extract

    return {"model": extract.cached_asset(weights_spec(size), progress),
            "tokenizer": extract.cached_asset(tokenizer_spec(size), progress)}


def required_bytes(size: str = SIZE) -> int:
    return WEIGHTS[size]["size_bytes"] + TOKENIZER["size_bytes"]


# ---------------------------------------------------------------- tokenizer
@lru_cache(maxsize=1)
def _byte_to_unicode() -> dict[int, str]:
    """GPT-2のbyte → 表示可能な文字の対応（tokenizersのByteLevelと同じ表）。"""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    codes = printable[:]
    extra = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            codes.append(256 + extra)
            extra += 1
    return dict(zip(printable, (chr(code) for code in codes)))


# GPT-2の分割規則。ASCIIの文では \p{L} は [A-Za-z]、\p{N} は [0-9] と同じ。
_PRETOKEN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


class BpeTokenizer:
    """``tokenizer.json``（byte-level BPE）で、PE-A-Frameの文を区切る。"""

    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = data.get("model") or {}
        if model.get("type") != "BPE" or (data.get("pre_tokenizer") or {}).get("type") != "ByteLevel":
            raise FrameDetectorError("PE-A-Frameのtokenizerの形が想定と違います")
        self.vocab = {str(piece): int(index) for piece, index in model["vocab"].items()}
        merges = model["merges"]
        self.ranks = {tuple(merge.split(" ") if isinstance(merge, str) else merge): rank
                      for rank, merge in enumerate(merges)}
        self._cache: dict[str, list[int]] = {}

    def _bpe(self, word: str) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        parts = list(word)
        while len(parts) > 1:
            best = None
            for index in range(len(parts) - 1):
                rank = self.ranks.get((parts[index], parts[index + 1]))
                if rank is not None and (best is None or rank < best[0]):
                    best = (rank, index)
            if best is None:
                break
            index = best[1]
            parts = parts[:index] + [parts[index] + parts[index + 1]] + parts[index + 2:]
        try:
            ids = [self.vocab[piece] for piece in parts]
        except KeyError as exc:
            raise FrameDetectorError(f"tokenizerの語彙に無い部分があります: {exc}") from exc
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        from lss.sound import t5_text

        value = unicodedata.normalize("NFC", t5_text.normalized_prompt(text))
        table = _byte_to_unicode()
        ids = [TEXT_CONFIG["cls_id"]]
        for piece in _PRETOKEN.findall(value):
            ids.extend(self._bpe("".join(table[byte] for byte in piece.encode("utf-8"))))
        ids.append(TEXT_CONFIG["sep_id"])
        if len(ids) > TEXT_CONFIG["max_tokens"]:
            raise FrameDetectorError("文が長すぎます")
        return ids


# ---------------------------------------------------------------- model
def build_model(size: str = SIZE):
    """PE-A-Frameの推論に要る部分（upstreamと同じパラメータ名）。"""
    import torch
    import torch.nn.functional as F
    from torch import nn

    audio = AUDIO_CONFIGS[size]
    text = TEXT_CONFIG

    class RMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            out = x.float() * torch.rsqrt((x.float() * x.float()).mean(-1, keepdim=True) + RMS_EPS)
            return (out * self.weight.float()).type_as(x)

    class Snake1d(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.alpha = nn.Parameter(torch.ones(1, dim, 1))

        def forward(self, x):
            return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)

    class ResidualUnit(nn.Module):
        def __init__(self, dim, dilation):
            super().__init__()
            self.snake1 = Snake1d(dim)
            self.conv1 = nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=(6 * dilation) // 2)
            self.snake2 = Snake1d(dim)
            self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

        def forward(self, x):
            y = self.conv2(self.snake2(self.conv1(self.snake1(x))))
            trim = (x.shape[-1] - y.shape[-1]) // 2
            if trim > 0:
                x = x[..., trim:-trim]
            return x + y

    class EncoderBlock(nn.Module):
        def __init__(self, dim, stride):
            super().__init__()
            self.res_unit1 = ResidualUnit(dim // 2, 1)
            self.res_unit2 = ResidualUnit(dim // 2, 3)
            self.res_unit3 = ResidualUnit(dim // 2, 9)
            self.snake1 = Snake1d(dim // 2)
            self.conv1 = nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride,
                                   padding=math.ceil(stride / 2))

        def forward(self, x):
            return self.conv1(self.snake1(self.res_unit3(self.res_unit2(self.res_unit1(x)))))

    class DacEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            width = DAC_CONFIG["encoder_hidden_size"]
            self.conv1 = nn.Conv1d(1, width, kernel_size=7, padding=3)
            self.block = nn.ModuleList(EncoderBlock(width * 2 ** (index + 1), stride)
                                       for index, stride in enumerate(DAC_CONFIG["downsampling_ratios"]))
            model_dim = width * 2 ** len(DAC_CONFIG["downsampling_ratios"])
            self.snake1 = Snake1d(model_dim)
            self.conv2 = nn.Conv1d(model_dim, DAC_CONFIG["hidden_size"], kernel_size=3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            for block in self.block:
                x = block(x)
            return self.conv2(self.snake1(x))

    class Bottleneck(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Conv1d(DAC_CONFIG["hidden_size"], DAC_CONFIG["codebook_dim"] * 2, kernel_size=1)
            self.out_proj = nn.Conv1d(DAC_CONFIG["codebook_dim"], DAC_CONFIG["hidden_size"], kernel_size=1)

        def forward(self, z):
            mean, _ = self.in_proj(z).chunk(2, dim=1)
            return mean

    class DacEncoderVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = DacEncoder()
            self.bottleneck = Bottleneck()

        def forward(self, waveform):
            with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
                return self.bottleneck(self.encoder(waveform))

    class ConvBlock1d(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.groupnorm = nn.GroupNorm(num_groups=1, num_channels=dim)
            self.project = nn.Conv1d(dim, dim, kernel_size=3, padding="same")

        def forward(self, x):
            return self.project(F.silu(self.groupnorm(x)))

    class ResnetBlock1d(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.block1 = ConvBlock1d(dim)
            self.block2 = ConvBlock1d(dim)

        def forward(self, x):
            return self.block2(self.block1(x)) + x

    class Embeddings(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.resnet_block = ResnetBlock1d(dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        def forward(self, x):
            x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
            return self.resnet_block(x.transpose(1, 2)).transpose(1, 2)

    def interleaved_rope(seqlen, dim, theta, device):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: dim // 2].float() / dim))
        angles = torch.outer(torch.arange(seqlen, device=device).float(), freqs)
        cos, sin = angles.cos(), angles.sin()
        return torch.stack((cos, -sin, sin, cos), dim=-1).view(seqlen, dim // 2, 2, 2)

    def apply_interleaved(x, freqs):
        # x: (B, H, S, D) → (B, H, S, D/2, 1, 2)
        pairs = x.reshape(*x.shape[:-1], -1, 1, 2)
        out = (pairs * freqs[None, None].float()).sum(-1).flatten(3)
        return out.type_as(x)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            dim, heads, head_dim = audio["hidden_size"], audio["heads"], audio["head_dim"]
            self.heads, self.head_dim = heads, head_dim
            self.q_proj = nn.Linear(dim, heads * head_dim, bias=False)
            self.k_proj = nn.Linear(dim, heads * head_dim, bias=False)
            self.v_proj = nn.Linear(dim, heads * head_dim, bias=False)
            self.o_proj = nn.Linear(heads * head_dim, dim, bias=False)
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        def forward(self, x, freqs):
            batch, length, _ = x.shape
            shape = (batch, length, self.heads, self.head_dim)
            q = self.q_norm(self.q_proj(x).view(shape)).transpose(1, 2)
            k = self.k_norm(self.k_proj(x).view(shape)).transpose(1, 2)
            v = self.v_proj(x).view(shape).transpose(1, 2)
            q, k = apply_interleaved(q, freqs), apply_interleaved(k, freqs)
            out = F.scaled_dot_product_attention(q, k, v)
            return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            dim, hidden = audio["hidden_size"], audio["intermediate_size"]
            self.gate_proj = nn.Linear(dim, hidden, bias=False)
            self.up_proj = nn.Linear(dim, hidden, bias=False)
            self.down_proj = nn.Linear(hidden, dim, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
            self.mlp = MLP()
            self.input_layernorm = RMSNorm(audio["hidden_size"])
            self.post_attention_layernorm = RMSNorm(audio["hidden_size"])

        def forward(self, x, freqs):
            x = x + self.self_attn(self.input_layernorm(x), freqs)
            return x + self.mlp(self.post_attention_layernorm(x))

    class AudioTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            dim = audio["hidden_size"]
            self.embeddings = Embeddings(dim)
            self.layers = nn.ModuleList(DecoderLayer() for _ in range(audio["layers"]))
            self.norm = RMSNorm(dim)
            self.output = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            x = self.embeddings(x)
            if x.shape[1] > MAX_POSITIONS:
                raise FrameDetectorError("音の窓が長すぎます")
            freqs = interleaved_rope(x.shape[1], audio["hidden_size"] // audio["heads"],
                                     max(10_000, 2 * MAX_POSITIONS), x.device)
            for layer in self.layers:
                x = layer(x, freqs)
            return self.output(self.norm(x))[:, 1:]

    class AudioModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.data_proj = nn.Linear(DAC_CONFIG["codebook_dim"], audio["hidden_size"])
            self.dac_vae_encoder = DacEncoderVAE()
            self.audio_transformer = AudioTransformer()

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    class TextAttention(nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            dim = text["hidden_size"]
            self.heads = text["num_heads"]
            self.head_dim = dim // self.heads
            self.local = layer_id % text["global_every"] != 0
            self.theta = text["local_rope_theta"] if self.local else text["global_rope_theta"]
            self.Wqkv = nn.Linear(dim, 3 * dim, bias=False)
            self.Wo = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            batch, length, dim = x.shape
            qkv = self.Wqkv(x).view(batch, length, 3, self.heads, self.head_dim)
            q, k, v = qkv.transpose(3, 1).unbind(dim=2)
            inv = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=x.device).float() / self.head_dim))
            angles = torch.outer(torch.arange(length, device=x.device).float(), inv)
            emb = torch.cat((angles, angles), dim=-1)
            cos, sin = emb.cos().to(x.dtype)[None, None], emb.sin().to(x.dtype)[None, None]
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            mask = None
            if self.local:
                position = torch.arange(length, device=x.device)
                mask = (position[None, :] - position[:, None]).abs() <= text["local_window"] // 2
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return self.Wo(out.transpose(1, 2).contiguous().view(batch, length, dim))

    class TextMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.Wi = nn.Linear(text["hidden_size"], text["intermediate_size"] * 2, bias=False)
            self.Wo = nn.Linear(text["intermediate_size"], text["hidden_size"], bias=False)

        def forward(self, x):
            value, gate = self.Wi(x).chunk(2, dim=-1)
            return self.Wo(F.gelu(value) * gate)

    def layer_norm(dim):
        return nn.LayerNorm(dim, eps=text["norm_eps"], bias=False)

    class TextLayer(nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            self.attn_norm = nn.Identity() if layer_id == 0 else layer_norm(text["hidden_size"])
            self.attn = TextAttention(layer_id)
            self.mlp_norm = layer_norm(text["hidden_size"])
            self.mlp = TextMLP()

        def forward(self, x):
            x = x + self.attn(self.attn_norm(x))
            return x + self.mlp(self.mlp_norm(x))

    class TextEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embeddings = nn.Embedding(text["vocab_size"], text["hidden_size"])
            self.norm = layer_norm(text["hidden_size"])

        def forward(self, ids):
            return self.norm(self.tok_embeddings(ids))

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = TextEmbeddings()
            self.layers = nn.ModuleList(TextLayer(index) for index in range(text["used_layers"]))

        def forward(self, ids):
            x = self.embeddings(ids)
            for layer in self.layers:
                x = layer(x)
            return x  # hidden_states[used_layers]（final_normの前）

    class ContrastiveHead(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
            self.proj = nn.Linear(in_dim, OUTPUT_DIM, bias=False)

        def forward(self, x):
            return self.proj(self.layer_norm(x))

    class PEAudioFrame(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_model = TextModel()
            self.audio_model = AudioModel()
            self.text_head = ContrastiveHead(text["hidden_size"])
            self.audio_head = ContrastiveHead(audio["hidden_size"])
            self.logit_scale = nn.Parameter(torch.zeros(1))
            self.logit_bias = nn.Parameter(torch.zeros(1))

    return PEAudioFrame()


def _skipped(name: str) -> bool:
    """推論で使わないtensor（使わない文の層・final_norm・VAEの復号側）。"""
    if name.startswith("text_model.layers."):
        return int(name.split(".")[2]) >= TEXT_CONFIG["used_layers"]
    return name in {"text_model.final_norm.weight"}


def load_model(weights: Path, *, size: str = SIZE, device: str = "cuda:0"):
    """parameterは空のままGPUへ置き、重みを1 tensorずつ写す（CPUのmemoryへmodel全体を置かない）。"""
    import torch

    from lss.sound import safetensors_file

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise FrameDetectorError("PE-A-FrameはCUDAで実行します（CPUへ切り替えません）")
    header, _ = safetensors_file.read_header(Path(weights))
    names = [name for name in header if not _skipped(name)]
    model = safetensors_file.build_empty(lambda: build_model(size), device=device)
    try:
        safetensors_file.copy_into(model, Path(weights), names=names)
    except safetensors_file.SafetensorsError as exc:
        raise FrameDetectorError(f"PE-A-Frameの重みが合いません: {exc}") from exc
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise FrameDetectorError("silent CPU fallbackを検出しました")
    return model.eval()


def _resample(values, source: int, target: int):
    import numpy as np
    from scipy.signal import resample_poly

    if source == target:
        return np.asarray(values, dtype=np.float32)
    divisor = math.gcd(source, target)
    return resample_poly(values, target // divisor, source // divisor).astype(np.float32)


class Detector:
    """読み込んだmodelで、文ごと・40 msごとの「その音が鳴っている」確率を作る。"""

    def __init__(self, paths: dict[str, Path], *, size: str = SIZE, device: str = "cuda:0"):
        with _LOAD_LOCK:
            self.model = load_model(paths["model"], size=size, device=device)
            self.tokenizer = BpeTokenizer(paths["tokenizer"])
        self.device = device
        self.size = size
        self._text_cache: dict[str, object] = {}

    def close(self) -> None:
        import torch

        self.model = None
        self._text_cache.clear()
        torch.cuda.empty_cache()

    def text_embedding(self, prompt: str):
        import torch

        if prompt not in self._text_cache:
            ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            with torch.inference_mode():
                hidden = self.model.text_model(ids)
                self._text_cache[prompt] = self.model.text_head(hidden[:, 0])[0].float()
        return self._text_cache[prompt]

    def codec_features(self, mono48):
        """48 kHz monoのDAC-VAE特徴（frames, 128）。余白付きの区間で計算して繋ぐ。"""
        import numpy as np
        import torch

        wav = torch.from_numpy(np.ascontiguousarray(mono48, dtype=np.float32)).to(self.device)
        if wav.shape[-1] % HOP:
            pad = HOP - wav.shape[-1] % HOP
            if wav.shape[-1] > pad:
                wav = torch.nn.functional.pad(wav[None, None], (0, pad), mode="reflect")[0, 0]
            else:
                wav = torch.nn.functional.pad(wav, (0, pad))
        frames = wav.shape[-1] // HOP
        block = int(CODEC_BLOCK_SECONDS * FRAMES_PER_SECOND)
        margin = int(CODEC_MARGIN_SECONDS * FRAMES_PER_SECOND)
        out = []
        encoder = self.model.audio_model.dac_vae_encoder
        with torch.inference_mode():
            for start in range(0, frames, block):
                end = min(frames, start + block)
                low, high = max(0, start - margin), min(frames, end + margin)
                piece = wav[low * HOP:high * HOP].reshape(1, 1, -1)
                features = encoder(piece)[0].transpose(0, 1)  # (frames, 128)
                out.append(features[start - low:start - low + (end - start)])
        return torch.cat(out, dim=0)

    def _window_embeddings(self, features):
        import torch

        model = self.model.audio_model
        with torch.inference_mode():
            hidden = model.audio_transformer(model.data_proj(features[None]))
            return self.model.audio_head(hidden)[0].float()  # (frames, 1024)

    def frame_probabilities(self, audio, rate: int, prompts: list[str], *,
                            progress: Callable[[int, int], None] | None = None):
        """音（frames[, channels]）に対する、文ごと・40 msごとの確率（numpy, (len(prompts), frames)）。"""
        import numpy as np
        import torch

        mono = audio.mean(axis=1) if getattr(audio, "ndim", 1) == 2 else audio
        if not np.isfinite(mono).all():
            raise FrameDetectorError("音に有限でない値があります")
        mono48 = _resample(mono, rate, SAMPLE_RATE)
        if len(mono48) < HOP:
            mono48 = np.pad(mono48, (0, HOP - len(mono48)))
        probabilities = self.probabilities_from_features(self.codec_features(mono48), prompts, progress=progress)
        expected = int(math.ceil(len(mono) / rate * FRAMES_PER_SECOND))
        probabilities = probabilities[:, :max(1, expected)]
        if not np.isfinite(probabilities).all():
            raise FrameDetectorError("確率に有限でない値があります")
        return probabilities

    def probabilities_from_features(self, features, prompts: list[str], *,
                                    progress: Callable[[int, int], None] | None = None):
        """DAC-VAE特徴（frames, 128）から、文ごと・40 msごとの確率（numpy）。窓を重ねて並べる。"""
        import torch

        frames = features.shape[0]
        texts = torch.stack([self.text_embedding(prompt) for prompt in prompts])  # (P, 1024)
        size = int(WINDOW_SECONDS * FRAMES_PER_SECOND)
        overlap = int(OVERLAP_SECONDS * FRAMES_PER_SECOND)
        hop = size - overlap
        starts = [0] if frames <= size else list(range(0, frames - overlap, hop))
        if starts[-1] + size < frames:
            starts.append(frames - size)
        logits = torch.full((len(prompts), frames), float("nan"), device=self.device)
        centre_distance = torch.full((frames,), float("inf"), device=self.device)
        with torch.inference_mode():
            scale, bias = self.model.logit_scale.float(), self.model.logit_bias.float()
            for count, start in enumerate(starts, start=1):
                end = min(frames, start + size)
                embeddings = self._window_embeddings(features[start:end])
                window_logits = (texts @ embeddings.T) * scale + bias  # (P, frames)
                position = torch.arange(start, end, device=self.device)
                distance = (position.float() - (start + end - 1) / 2).abs()
                better = distance < centre_distance[start:end]
                logits[:, start:end] = torch.where(better[None], window_logits, logits[:, start:end])
                centre_distance[start:end] = torch.where(better, distance, centre_distance[start:end])
                if progress is not None:
                    progress(count, len(starts))
            return logits.sigmoid().cpu().numpy()

    def spans(self, audio, rate: int, prompt: str, *, threshold: float):
        """確率が ``threshold`` を超える区間（秒, 秒）。"""
        probabilities = self.frame_probabilities(audio, rate, [prompt])[0]
        return frames_to_spans(probabilities > threshold)


def frames_to_spans(flags) -> list[tuple[float, float]]:
    spans, start = [], None
    for index, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start / FRAMES_PER_SECOND, index / FRAMES_PER_SECOND))
            start = None
    return spans


def per_second(probabilities, seconds: int):
    """40 msごとの確率から、各秒の最大（seconds,）。"""
    import numpy as np

    values = np.zeros(seconds, dtype=np.float32)
    for index in range(seconds):
        block = probabilities[index * FRAMES_PER_SECOND:(index + 1) * FRAMES_PER_SECOND]
        values[index] = float(block.max()) if len(block) else 0.0
    return values
