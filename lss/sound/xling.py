"""日本語の言葉と英語の訳が同じ意味かを測る、多言語の文の埋め込み（BERT系、sentence-transformers形式）。

`lss.sound.translate` の訳は短い語で意味を外すことがある。訳の候補を使う前に、元の日本語と訳した英語を
同じ空間へ写し、cosine類似度が閾値に届くものだけを使う（届かなければ、その言葉は読めなかったものとして
扱い、推測で別の音を作らない）。

固定している実行条件（upstream: transformers の BertModel / tokenizers の BertWordPiece・Unigram と
sentence-transformers の Pooling/Dense/Normalize と同じ演算・同じパラメータ名。製品はそれらを同梱しない）:

- encoderは BertModel（post-LN、GELU、絶対位置、token type 0）。float32。
- ``labse``: WordPiece（大文字小文字・アクセントはそのまま、CJKの文字は1文字ずつ、記号で区切る）、
  ``[CLS] ... [SEP]``。先頭tokenの出力 → Dense(tanh) → L2正規化。
- ``mminilm``: SentencePiece Unigram（tokenizer.json の語彙と precompiled charsmap）、``<s> ... </s>``。
  token出力の平均。
"""
from __future__ import annotations

import base64
import json
import math
import re
import threading
import unicodedata
from pathlib import Path
from typing import Callable

MODELS = {
    "labse": {
        "repository": "https://huggingface.co/sentence-transformers/LaBSE",
        "revision": "836121a0533e5664b21c7aacc5d22951f2b8b25b", "license": "Apache-2.0",
        "cache_name": "labse",
        "files": {"model": ("model.safetensors", 1883734344,
                            "77d8e1f2dbab6eb5d3c261ce9d3dbf1e3c69e02938c95f934f94f42c22dfa31f"),
                  "tokenizer": ("tokenizer.json", 9621556,
                                "d09046654dc5151fc313613f16768b6ecc1e5e3bddbc9c8aca209a5f23fe3cd3"),
                  "dense": ("2_Dense/model.safetensors", 2362528,
                            "f866c945fa2147ce4940c8ead0b4b20af4217e0f4664bd35f41c2a0967c4fe77")},
        "config": {"hidden": 768, "layers": 12, "heads": 12, "intermediate": 3072, "eps": 1e-12,
                   "vocab": 501153, "positions": 512, "pooling": "cls_dense_tanh_normalize"},
    },
    "mminilm": {
        "repository": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42", "license": "Apache-2.0",
        "cache_name": "paraphrase_multilingual_minilm_l12_v2",
        "files": {"model": ("model.safetensors", 470641600,
                            "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b"),
                  "tokenizer": ("tokenizer.json", 9081518,
                                "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8")},
        "config": {"hidden": 384, "layers": 12, "heads": 12, "intermediate": 1536, "eps": 1e-12,
                   "vocab": 250037, "positions": 512, "pooling": "mean"},
    },
}
MAX_TOKENS = 64
_LOAD_LOCK = threading.Lock()


class JudgeError(RuntimeError):
    """文の埋め込みの入力・重み・出力が契約に合わない。"""


def asset_spec(name: str, role: str) -> dict:
    model = MODELS[name]
    file_name, size, digest = model["files"][role]
    return {"cache_name": model["cache_name"], "file_name": file_name.replace("/", "_"),
            "url": f"{model['repository']}/resolve/{model['revision']}/{file_name}", "size_bytes": size,
            "sha256": digest, "license": model["license"]}


def ensure_assets(name: str, progress: Callable[[dict], None] | None = None) -> dict[str, Path]:
    from lss.sound import extract

    return {role: extract.cached_asset(asset_spec(name, role), progress) for role in MODELS[name]["files"]}


def required_bytes(name: str) -> int:
    return sum(size for _, size, _ in MODELS[name]["files"].values())


# ---------------------------------------------------------------- tokenizers
def _is_cjk(code: int) -> bool:
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F or 0x2B740 <= code <= 0x2B81F or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF or 0x2F800 <= code <= 0x2FA1F)


def _is_punctuation(char: str) -> bool:
    code = ord(char)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126:
        return True
    return unicodedata.category(char).startswith("P")


class WordPiece:
    """BERTのBasicTokenizer（lowercase無し・strip_accents無し）＋ WordPiece（``##`` 継続）。"""

    def __init__(self, data: dict):
        model = data["model"]
        self.vocab = {str(piece): int(index) for piece, index in model["vocab"].items()}
        self.unk = self.vocab[model.get("unk_token", "[UNK]")]
        self.prefix = model.get("continuing_subword_prefix", "##")
        self.max_chars = int(model.get("max_input_chars_per_word", 100))
        self.cls, self.sep = self.vocab["[CLS]"], self.vocab["[SEP]"]

    def _clean(self, text: str) -> str:
        out = []
        for char in text:
            code = ord(char)
            if code == 0 or code == 0xFFFD or (unicodedata.category(char).startswith("C") and char not in "\t\n\r"):
                continue
            out.append(" " if char in "\t\n\r" or unicodedata.category(char) == "Zs" else char)
        return "".join(out)

    def _words(self, text: str) -> list[str]:
        spaced = "".join(f" {char} " if _is_cjk(ord(char)) else char for char in self._clean(text))
        words = []
        for token in spaced.split():
            current = ""
            for char in token:
                if _is_punctuation(char):
                    if current:
                        words.append(current)
                    words.append(char)
                    current = ""
                else:
                    current += char
            if current:
                words.append(current)
        return words

    def encode(self, text: str) -> list[int]:
        ids = [self.cls]
        for word in self._words(text):
            if len(word) > self.max_chars:
                ids.append(self.unk)
                continue
            pieces, start, bad = [], 0, False
            while start < len(word):
                end, found = len(word), None
                while start < end:
                    piece = word[start:end] if start == 0 else self.prefix + word[start:end]
                    if piece in self.vocab:
                        found = self.vocab[piece]
                        break
                    end -= 1
                if found is None:
                    bad = True
                    break
                pieces.append(found)
                start = end
            ids.extend([self.unk] if bad else pieces)
        ids.append(self.sep)
        return ids[:MAX_TOKENS - 1] + ([self.sep] if len(ids) > MAX_TOKENS else [])


class UnigramJson:
    """tokenizers の ``Precompiled`` 正規化 ＋ 空白分割 ＋ ``Metaspace`` ＋ Unigram（XLM-R形）。"""

    def __init__(self, data: dict):
        from lss.sound import translate

        normalizer = data.get("normalizer") or {}
        if normalizer.get("type") != "Precompiled":
            raise JudgeError("想定外の正規化です")
        self.charsmap = translate._CharsMap(base64.b64decode(normalizer["precompiled_charsmap"]))
        model = data["model"]
        self.unk_id = int(model["unk_id"])
        self.ids, self.scores = {}, {}
        for index, (piece, score) in enumerate(model["vocab"]):
            self.ids.setdefault(piece, index)
            self.scores.setdefault(piece, float(score))
        special = {"<s>", "<pad>", "</s>", "<unk>", "<mask>"}
        self.pieces = {piece: score for piece, score in self.scores.items() if piece not in special}
        self.max_piece = max(len(piece) for piece in self.pieces)
        self.unknown_score = min(self.pieces.values()) - 10.0
        self.bos, self.eos = self.ids["<s>"], self.ids["</s>"]

    def _segment(self, value: str) -> list[str]:
        count = len(value)
        best = [(-math.inf, -1, False)] * (count + 1)
        best[0] = (0.0, -1, False)
        for end in range(1, count + 1):
            for start in range(max(0, end - self.max_piece), end):
                if best[start][0] == -math.inf:
                    continue
                score = self.pieces.get(value[start:end])
                if score is not None and best[start][0] + score > best[end][0]:
                    best[end] = (best[start][0] + score, start, False)
            start = end - 1
            if value[start:end] not in self.pieces and best[start][0] != -math.inf:
                if best[start][0] + self.unknown_score > best[end][0]:
                    best[end] = (best[start][0] + self.unknown_score, start, True)
        out, end = [], count
        while end > 0:
            _, start, unknown = best[end]
            out.append((value[start:end], unknown))
            end = start
        out.reverse()
        merged, previous = [], False
        for piece, unknown in out:
            if unknown and previous:
                merged[-1] = (merged[-1][0] + piece, True)
            else:
                merged.append((piece, unknown))
            previous = unknown
        return [piece if not unknown else None for piece, unknown in merged]

    def encode(self, text: str) -> list[int]:
        ids = [self.bos]
        for word in self.charsmap.normalize(text).split():
            for piece in self._segment("▁" + word):
                ids.append(self.unk_id if piece is None else self.ids.get(piece, self.unk_id))
        ids.append(self.eos)
        return ids[:MAX_TOKENS - 1] + ([self.eos] if len(ids) > MAX_TOKENS else [])


# ---------------------------------------------------------------- model
def build_model(name: str):
    import torch
    from torch import nn

    config = MODELS[name]["config"]
    dim, heads = config["hidden"], config["heads"]

    class SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)

    class Dense(nn.Module):
        def __init__(self, inputs, outputs):
            super().__init__()
            self.dense = nn.Linear(inputs, outputs)
            self.LayerNorm = nn.LayerNorm(outputs, eps=config["eps"])

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = SelfAttention()
            self.output = Dense(dim, dim)

        def forward(self, x):
            batch, length, _ = x.shape
            head = dim // heads
            shape = (batch, length, heads, head)
            q = self.self.query(x).view(shape).transpose(1, 2)
            k = self.self.key(x).view(shape).transpose(1, 2)
            v = self.self.value(x).view(shape).transpose(1, 2)
            weights = (q @ k.transpose(-1, -2)) / math.sqrt(head)
            out = (weights.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, length, dim)
            return self.output.LayerNorm(self.output.dense(out) + x)

    class Intermediate(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(dim, config["intermediate"])

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.intermediate = Intermediate()
            self.output = Dense(config["intermediate"], dim)

        def forward(self, x):
            x = self.attention(x)
            hidden = nn.functional.gelu(self.intermediate.dense(x))
            return self.output.LayerNorm(self.output.dense(hidden) + x)

    class Embeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(config["vocab"], dim)
            self.position_embeddings = nn.Embedding(config["positions"], dim)
            self.token_type_embeddings = nn.Embedding(2, dim)
            self.LayerNorm = nn.LayerNorm(dim, eps=config["eps"])
            self.register_buffer("position_ids", torch.arange(config["positions"])[None], persistent=True)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList(Layer() for _ in range(config["layers"]))

    class Pooler(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(dim, dim)

    class Bert(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = Embeddings()
            self.encoder = Encoder()
            self.pooler = Pooler()   # 重みに含まれる（文の埋め込みには使わない）

        def forward(self, ids):
            embeddings = self.embeddings
            positions = embeddings.position_ids[:, :ids.shape[1]]
            x = (embeddings.word_embeddings(ids) + embeddings.position_embeddings(positions)
                 + embeddings.token_type_embeddings(torch.zeros_like(ids)))
            x = embeddings.LayerNorm(x)
            for layer in self.encoder.layer:
                x = layer(x)
            return x

    return Bert()


class Judge:
    """日本語の言葉と英語の文の意味の近さ（cosine類似度）。"""

    def __init__(self, name: str, paths: dict[str, Path], *, device: str = "cuda:0"):
        import torch

        from lss.sound import safetensors_file

        if name not in MODELS:
            raise JudgeError(f"未知のmodelです: {name}")
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise JudgeError("文の埋め込みはCUDAで実行します（CPUへ切り替えません）")
        self.name = name
        data = json.loads(Path(paths["tokenizer"]).read_text(encoding="utf-8"))
        self.tokenizer = WordPiece(data) if name == "labse" else UnigramJson(data)
        with _LOAD_LOCK:
            self.model = safetensors_file.build_empty(lambda: build_model(name), device=device)
            safetensors_file.copy_into(self.model, Path(paths["model"]))
            self.model.eval()
            self.dense = None
            if MODELS[name]["config"]["pooling"] == "cls_dense_tanh_normalize":
                dense = safetensors_file.load(Path(paths["dense"]))
                self.dense = (dense["linear.weight"].to(device), dense["linear.bias"].to(device))
        self.device = device

    def close(self) -> None:
        import torch

        self.model = None
        self.dense = None
        torch.cuda.empty_cache()

    def embed(self, text: str):
        import torch

        ids = torch.tensor([self.tokenizer.encode(text)], device=self.device)
        with torch.inference_mode():
            hidden = self.model(ids)[0]
            if self.dense is not None:
                vector = torch.tanh(hidden[0] @ self.dense[0].T + self.dense[1])
            else:
                vector = hidden.mean(dim=0)
            return torch.nn.functional.normalize(vector.float(), dim=-1)

    def similarity(self, japanese: str, english: str) -> float:
        return float(self.embed(japanese) @ self.embed(english))
