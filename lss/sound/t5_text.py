"""SAM Audioの文の条件付けに使うT5-base encoder（google-t5/t5-base、Apache-2.0）の製品adapter。

upstream（SAM Audio）は ``transformers.T5EncoderModel`` と ``AutoTokenizer`` を使う。製品はtransformersを
同梱しないので、同じ演算・同じパラメータ名でencoderをここに書き、配布重み（``model.safetensors``）の
encoder部分を **strict** に読む。tokenizerも配布の ``tokenizer.json``（Unigram）を同じ規則で読む。

固定している条件:

- 文はASCIIの英語（小文字・数字・空白・``'-,&``）だけを受ける。その範囲ではupstreamの正規化
  （Precompiled charsmap）は値を変えない。それ以外の文字はここへ渡す前に `lss.sound.intent` が英語へ直す。
- 前処理はupstreamと同じ「空白で分割し、各語の先頭へ ``▁``」。語ごとにUnigramの最尤分割（Viterbi）。
  末尾へ ``</s>``（id 1）。
- encoderはfloat32（upstreamの既定と同じ）。相対位置bucketは双方向32個・最大距離128。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

REVISION = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
REPOSITORY = "https://huggingface.co/google-t5/t5-base"
WEIGHTS = {
    "cache_name": "t5_base",
    "file_name": "model.safetensors",
    "url": f"{REPOSITORY}/resolve/{REVISION}/model.safetensors",
    "license": "Apache-2.0",
    "size_bytes": 891646390,
    "sha256": "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4",
}
TOKENIZER = {
    "cache_name": "t5_base",
    "file_name": "tokenizer.json",
    "url": f"{REPOSITORY}/resolve/{REVISION}/tokenizer.json",
    "license": "Apache-2.0",
    "size_bytes": 1389353,
    "sha256": "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066",
}
CONFIG = {"d_model": 768, "d_kv": 64, "d_ff": 3072, "num_heads": 12, "num_layers": 12,
          "relative_attention_num_buckets": 32, "relative_attention_max_distance": 128,
          "layer_norm_epsilon": 1e-6, "vocab_size": 32128, "eos_token_id": 1, "max_length": 512}
_ALLOWED = re.compile(r"^[a-z0-9 '\-,&]+$")
_SPACE = "▁"


class TextEncoderError(ValueError):
    """文かtokenizer・重みが契約に合わない。"""


def normalized_prompt(text: str) -> str:
    """モデルへ渡す文の正規形。受けられない文は例外。"""
    if not isinstance(text, str):
        raise TextEncoderError("文が文字列ではありません")
    value = " ".join(text.strip().lower().split())
    if not value or not _ALLOWED.fullmatch(value) or len(value) > 200:
        raise TextEncoderError(f"モデルへ渡せる英語の文ではありません: {text!r}")
    return value


class UnigramTokenizer:
    """``tokenizer.json`` のUnigram語彙で、語ごとの最尤分割をする。"""

    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = data.get("model") or {}
        vocab = model.get("vocab")
        if not isinstance(vocab, list) or len(vocab) < 1000:
            raise TextEncoderError("tokenizerの語彙を読めません")
        self.unk_id = int(model.get("unk_id", 2))
        self.pieces = {}
        for index, (piece, score) in enumerate(vocab):
            self.pieces.setdefault(piece, (index, float(score)))
        self.max_piece = max(len(piece) for piece in self.pieces)
        self.unk_score = min(score for _, score in vocab) - 10.0

    def _word(self, word: str) -> list[int]:
        size = len(word)
        best = [(-math.inf, -1, -1)] * (size + 1)  # (score, start, id)
        best[0] = (0.0, -1, -1)
        for end in range(1, size + 1):
            for start in range(max(0, end - self.max_piece), end):
                if best[start][0] == -math.inf:
                    continue
                found = self.pieces.get(word[start:end])
                if found is not None:
                    score = best[start][0] + found[1]
                    if score > best[end][0]:
                        best[end] = (score, start, found[0])
            if best[end][0] == -math.inf:
                # 語彙に無い1文字は未知語として進める。
                score = best[end - 1][0] + self.unk_score
                best[end] = (score, end - 1, self.unk_id)
        ids = []
        end = size
        while end > 0:
            _, start, token = best[end]
            ids.append(token)
            end = start
        ids.reverse()
        merged = []
        for token in ids:
            if token == self.unk_id and merged and merged[-1] == self.unk_id:
                continue
            merged.append(token)
        return merged

    def encode(self, text: str) -> list[int]:
        value = normalized_prompt(text)
        ids = []
        for word in value.split(" "):
            ids.extend(self._word(_SPACE + word))
        ids.append(CONFIG["eos_token_id"])
        return ids[:CONFIG["max_length"]]


def _relative_buckets(relative, num_buckets: int, max_distance: int):
    import torch

    num_buckets //= 2
    buckets = (relative > 0).to(torch.long) * num_buckets
    relative = torch.abs(relative)
    max_exact = num_buckets // 2
    is_small = relative < max_exact
    large = max_exact + (
        torch.log(relative.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    large = torch.min(large, torch.full_like(large, num_buckets - 1))
    return buckets + torch.where(is_small, relative, large)


def build_encoder():
    """``transformers`` のT5EncoderModel（t5-base）と同じ演算・同じパラメータ名。"""
    import torch
    from torch import nn

    config = CONFIG
    inner = config["num_heads"] * config["d_kv"]

    class LayerNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(config["d_model"]))

        def forward(self, x):
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return self.weight * (x * torch.rsqrt(variance + config["layer_norm_epsilon"]))

    class Attention(nn.Module):
        def __init__(self, has_bias: bool):
            super().__init__()
            self.q = nn.Linear(config["d_model"], inner, bias=False)
            self.k = nn.Linear(config["d_model"], inner, bias=False)
            self.v = nn.Linear(config["d_model"], inner, bias=False)
            self.o = nn.Linear(inner, config["d_model"], bias=False)
            if has_bias:
                self.relative_attention_bias = nn.Embedding(config["relative_attention_num_buckets"], config["num_heads"])

        def position_bias(self, length: int, device):
            positions = torch.arange(length, dtype=torch.long, device=device)
            relative = positions[None, :] - positions[:, None]
            buckets = _relative_buckets(relative, config["relative_attention_num_buckets"],
                                        config["relative_attention_max_distance"])
            return self.relative_attention_bias(buckets).permute(2, 0, 1).unsqueeze(0)

        def forward(self, x, bias):
            batch, length, _ = x.shape

            def heads(t):
                return t.view(batch, length, config["num_heads"], config["d_kv"]).transpose(1, 2)

            q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
            scores = torch.matmul(q, k.transpose(3, 2)) + bias
            weights = torch.softmax(scores.float(), dim=-1).type_as(scores)
            out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, length, inner)
            return self.o(out)

    class SelfAttentionLayer(nn.Module):
        def __init__(self, has_bias: bool):
            super().__init__()
            self.SelfAttention = Attention(has_bias)
            self.layer_norm = LayerNorm()

    class DenseReluDense(nn.Module):
        def __init__(self):
            super().__init__()
            self.wi = nn.Linear(config["d_model"], config["d_ff"], bias=False)
            self.wo = nn.Linear(config["d_ff"], config["d_model"], bias=False)

        def forward(self, x):
            return self.wo(torch.relu(self.wi(x)))

    class FeedForwardLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.DenseReluDense = DenseReluDense()
            self.layer_norm = LayerNorm()

    class Block(nn.Module):
        def __init__(self, has_bias: bool):
            super().__init__()
            self.layer = nn.ModuleList([SelfAttentionLayer(has_bias), FeedForwardLayer()])

        def forward(self, x, bias):
            attention, feed = self.layer
            x = x + attention.SelfAttention(attention.layer_norm(x), bias)
            return x + feed.DenseReluDense(feed.layer_norm(x))

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.ModuleList([Block(index == 0) for index in range(config["num_layers"])])
            self.final_layer_norm = LayerNorm()

    class T5Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Embedding(config["vocab_size"], config["d_model"])
            self.encoder = Stack()

        def forward(self, input_ids, attention_mask):
            x = self.shared(input_ids)
            length = input_ids.shape[1]
            bias = self.encoder.block[0].layer[0].SelfAttention.position_bias(length, input_ids.device)
            mask = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * torch.finfo(x.dtype).min
            bias = bias + mask
            for block in self.encoder.block:
                x = block(x, bias)
            return self.encoder.final_layer_norm(x)

    return T5Encoder()


def load_encoder(weights: Path, *, device: str = "cuda:0"):
    """配布重みのencoder部分を厳密に読む。decoder側のtensorは読まない。"""
    import torch

    from lss.sound import safetensors_file

    # parameterは空のままGPUへ置き、重みを1 tensorずつfloat32で写す（CPUのmemoryへ全体を置かない）。
    model = safetensors_file.build_empty(build_encoder, device=device)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TextEncoderError("文のencoderはfloat32で組みます")
    safetensors_file.copy_into(model, weights, names=list(model.state_dict()))
    return model.eval()


class TextConditioner:
    """文をSAM Audioの条件（hidden state と mask）へ変える。"""

    def __init__(self, weights: Path, tokenizer: Path, *, device: str = "cuda:0"):
        self.tokenizer = UnigramTokenizer(tokenizer)
        self.model = load_encoder(weights, device=device)
        self.device = device

    def __call__(self, texts: list[str]):
        import torch

        encoded = [self.tokenizer.encode(text) for text in texts]
        longest = max(len(ids) for ids in encoded)
        ids = torch.zeros((len(encoded), longest), dtype=torch.long)
        mask = torch.zeros((len(encoded), longest), dtype=torch.long)
        for row, values in enumerate(encoded):
            ids[row, :len(values)] = torch.tensor(values)
            mask[row, :len(values)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        with torch.inference_mode():
            hidden = self.model(ids, mask)
        return hidden, mask.bool()
