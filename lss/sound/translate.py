"""語彙に無い日本語の「欲しい音」を、英語の名詞句の候補へ訳す（opus-mt-ja-en、Helsinki-NLP、Apache-2.0）。

訳は候補にすぎない。短い語では訳が意味を外す（「サイレン」→ "Silence"、「琵琶」→ "I'll be right back."）ので、
訳した候補は `lss.sound.intent` が別の判定（日本語と英語の意味の近さ）で確かめてから使う。ここは訳すだけで、
推測で別の音へ置き換える判断はしない。

固定している実行条件（upstream: transformers の MarianTokenizer / MarianMTModel / generate と同じ演算・同じ
パラメータ名。製品はtransformers・sentencepieceを同梱しない）:

- 文の正規化は SentencePiece の ``nmt_nfkc``（``source.spm`` に同梱の precompiled charsmap を、そのまま
  double-array の最長一致で引く）。先頭へ ``▁``、連続する空白は1つ、空白は ``▁``。
- 分割は ``source.spm`` のUnigram（Viterbi、未知の文字は1文字ずつ ``<unk>``、連続する未知は1つにまとめる）。
  語片を ``vocab.json`` で番号にし、末尾へ ``</s>``。
- model は Marian（encoder・decoder 6層、post-LN、swish、正弦の位置、埋め込みは√512倍、出力は共有埋め込み＋
  ``final_logits_bias``）。float32。
- 探索は beam 6、長さの罰則 1.0、``<pad>`` は出さない、各段で log-softmax を取り直す（``renormalize_logits``）。
  終了の規則は transformers 4.57 の beam search と同じ（候補の上位 beam 数だけが完了でき、完了の得点は
  生成した長さで割る。実行中の最良を現在の長さで割っても完了の最悪に届かなければ止める）。
"""
from __future__ import annotations

import json
import math
import re
import struct
import threading
import unicodedata
from pathlib import Path
from typing import Callable

REVISION = "0770961a39ba6bd66305b149c3f4110bcafca2e6"
REPOSITORY = "https://huggingface.co/Helsinki-NLP/opus-mt-ja-en"
LICENSE = "Apache-2.0 (Helsinki-NLP/opus-mt-ja-en model card)"
CACHE = "opus_mt_ja_en"
FILES = {
    "model": {"file_name": "pytorch_model.bin", "size_bytes": 303294189,
              "sha256": "ed649116c143fc2d7aea690246f4b2b7caa814e9e00a8d5bbe047822b18de022"},
    "source": {"file_name": "source.spm", "size_bytes": 781853,
               "sha256": "d0b5c3b10b5959f056ff2c86e2f2356129242ff1fc72d3a4a34d6a8c0eee4e57"},
    "target": {"file_name": "target.spm", "size_bytes": 801883,
               "sha256": "35d89c704e270d441fade716a3931d49c43179400682edcca7f180694982a2f6"},
    "vocab": {"file_name": "vocab.json", "size_bytes": 1501341,
              "sha256": "0483c9c2b3deade6d6fbfe149eccf9cd3c44c77b348232030ae2d81400d65b61"},
}
CONFIG = {"d_model": 512, "heads": 8, "ffn": 2048, "encoder_layers": 6, "decoder_layers": 6, "vocab_size": 60716,
          "max_positions": 512, "pad_id": 60715, "eos_id": 0, "unk_id": 1, "decoder_start_id": 60715}
SEARCH = {"num_beams": 6, "length_penalty": 1.0, "max_new_tokens": 32, "renormalize_logits": True}
MAX_SOURCE_TOKENS = 64
_SPACE = "▁"
_LOAD_LOCK = threading.Lock()


class TranslationError(RuntimeError):
    """訳の入力・重み・出力が契約に合わない。"""


def asset_spec(role: str) -> dict:
    spec = dict(FILES[role])
    spec.update({"cache_name": CACHE, "url": f"{REPOSITORY}/resolve/{REVISION}/{spec['file_name']}",
                 "license": LICENSE})
    return spec


def ensure_assets(progress: Callable[[dict], None] | None = None) -> dict[str, Path]:
    from lss.sound import extract

    return {role: extract.cached_asset(asset_spec(role), progress) for role in FILES}


def required_bytes() -> int:
    return sum(spec["size_bytes"] for spec in FILES.values())


# ---------------------------------------------------------------- SentencePiece
def _varint(data: bytes, index: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if index >= len(data):
            raise TranslationError("SentencePieceのmodelが途中で終わっています")
        byte = data[index]
        index += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return result, index


def _fields(data: bytes):
    index = 0
    while index < len(data):
        key, index = _varint(data, index)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, index = _varint(data, index)
        elif wire == 1:
            value, index = data[index:index + 8], index + 8
        elif wire == 2:
            length, index = _varint(data, index)
            value, index = data[index:index + length], index + length
        elif wire == 5:
            value, index = data[index:index + 4], index + 4
        else:
            raise TranslationError(f"SentencePieceのmodelに未対応の形があります: {wire}")
        yield number, value


class _CharsMap:
    """SentencePieceのprecompiled charsmap（double-array trie ＋ 置き換え文字列）。"""

    def __init__(self, blob: bytes):
        if len(blob) < 4:
            raise TranslationError("正規化の表が空です")
        (trie_size,) = struct.unpack("<I", blob[:4])
        if trie_size % 4 or 4 + trie_size > len(blob):
            raise TranslationError("正規化の表の大きさが不正です")
        self.units = struct.unpack(f"<{trie_size // 4}I", blob[4:4 + trie_size])
        self.pool = blob[4 + trie_size:]

    def _longest(self, data: bytes, start: int) -> tuple[int, int] | None:
        units = self.units
        unit = units[0]
        position = (unit >> 10) << ((unit & (1 << 9)) >> 6)
        found = None
        for index in range(start, len(data)):
            byte = data[index]
            position ^= byte
            if position >= len(units):
                break
            unit = units[position]
            if (unit & ((1 << 31) | 0xFF)) != byte:
                break
            position ^= (unit >> 10) << ((unit & (1 << 9)) >> 6)
            if (unit >> 8) & 1:
                found = (index + 1 - start, units[position] & ((1 << 31) - 1))
        return found

    def normalize(self, text: str) -> str:
        data = text.encode("utf-8")
        out = bytearray()
        index = 0
        while index < len(data):
            match = self._longest(data, index)
            if match is not None:
                length, offset = match
                end = self.pool.index(b"\0", offset)
                out += self.pool[offset:end]
                index += length
                continue
            # 表に無い文字はそのまま（UTF-8の1文字）
            first = data[index]
            size = 1 if first < 0x80 else 2 if first >> 5 == 0b110 else 3 if first >> 4 == 0b1110 else 4
            out += data[index:index + size]
            index += size
        return out.decode("utf-8", errors="strict")


class SentencePiece:
    """``*.spm``（Unigram）を読み、正規化と最尤分割をする。"""

    def __init__(self, path: Path):
        data = Path(path).read_bytes()
        self.pieces: dict[str, float] = {}
        self.unknown = "<unk>"
        normalizer = None
        for number, value in _fields(data):
            if number == 1:
                piece, score, kind = None, 0.0, 1
                for field, item in _fields(value):
                    if field == 1:
                        piece = item.decode("utf-8")
                    elif field == 2:
                        (score,) = struct.unpack("<f", item)
                    elif field == 3:
                        kind = item
                if piece is None:
                    raise TranslationError("SentencePieceの語片に文字がありません")
                if kind == 1:          # 通常の語片だけを分割に使う（未知・制御は使わない）
                    self.pieces.setdefault(piece, float(score))
                elif kind == 2:
                    self.unknown = piece
            elif number == 3:
                normalizer = dict(_fields(value))
        if not self.pieces or normalizer is None:
            raise TranslationError("SentencePieceのmodelを読めません")
        name = normalizer.get(1, b"").decode("utf-8")
        if name != "nmt_nfkc":
            raise TranslationError(f"想定外の正規化です: {name}")
        self.charsmap = _CharsMap(normalizer.get(2, b""))
        self.add_dummy_prefix = bool(normalizer.get(3, 1))
        self.remove_extra_whitespaces = bool(normalizer.get(4, 1))
        self.max_piece = max(len(piece) for piece in self.pieces)
        self.unknown_score = min(self.pieces.values()) - 10.0

    def normalized(self, text: str) -> str:
        value = self.charsmap.normalize(text)
        if self.remove_extra_whitespaces:
            value = re.sub(" +", " ", value).strip(" ")
        if not value:
            return ""
        value = value.replace(" ", _SPACE)
        return (_SPACE + value) if self.add_dummy_prefix else value

    def encode(self, text: str) -> list[str]:
        value = self.normalized(text)
        if not value:
            return []
        count = len(value)
        best = [(-math.inf, -1, False)] * (count + 1)
        best[0] = (0.0, -1, False)
        for end in range(1, count + 1):
            for start in range(max(0, end - self.max_piece), end):
                if best[start][0] == -math.inf:
                    continue
                piece = value[start:end]
                score = self.pieces.get(piece)
                if score is not None:
                    total = best[start][0] + score
                    if total > best[end][0]:
                        best[end] = (total, start, False)
            # SentencePieceと同じく、1文字の語片が無い位置にだけ1文字の未知を置く。
            start = end - 1
            if value[start:end] not in self.pieces and best[start][0] != -math.inf:
                total = best[start][0] + self.unknown_score
                if total > best[end][0]:
                    best[end] = (total, start, True)
        pieces = []
        end = count
        while end > 0:
            _, start, unknown = best[end]
            pieces.append((value[start:end], unknown))
            end = start
        pieces.reverse()
        merged: list[str] = []
        previous_unknown = False
        for piece, unknown in pieces:
            if unknown and previous_unknown:
                merged[-1] += piece
            else:
                merged.append(piece)
            previous_unknown = unknown
        return merged


# ---------------------------------------------------------------- model
def build_model():
    import torch
    from torch import nn

    config = CONFIG
    dim, heads = config["d_model"], config["heads"]

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.q_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)

        def forward(self, x, memory=None, mask=None):
            batch, length, _ = x.shape
            source = x if memory is None else memory
            head = dim // heads
            q = (self.q_proj(x) * head ** -0.5).view(batch, length, heads, head).transpose(1, 2)
            k = self.k_proj(source).view(batch, -1, heads, head).transpose(1, 2)
            v = self.v_proj(source).view(batch, -1, heads, head).transpose(1, 2)
            weights = q @ k.transpose(-1, -2)
            if mask is not None:
                weights = weights + mask
            out = weights.softmax(dim=-1) @ v
            return self.out_proj(out.transpose(1, 2).reshape(batch, length, dim))

    class EncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
            self.self_attn_layer_norm = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, config["ffn"])
            self.fc2 = nn.Linear(config["ffn"], dim)
            self.final_layer_norm = nn.LayerNorm(dim)

        def forward(self, x):
            x = self.self_attn_layer_norm(x + self.self_attn(x))
            return self.final_layer_norm(x + self.fc2(nn.functional.silu(self.fc1(x))))

    class DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
            self.self_attn_layer_norm = nn.LayerNorm(dim)
            self.encoder_attn = Attention()
            self.encoder_attn_layer_norm = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, config["ffn"])
            self.fc2 = nn.Linear(config["ffn"], dim)
            self.final_layer_norm = nn.LayerNorm(dim)

        def forward(self, x, memory, causal):
            x = self.self_attn_layer_norm(x + self.self_attn(x, mask=causal))
            x = self.encoder_attn_layer_norm(x + self.encoder_attn(x, memory=memory))
            return self.final_layer_norm(x + self.fc2(nn.functional.silu(self.fc1(x))))

    class Stack(nn.Module):
        def __init__(self, layer, count):
            super().__init__()
            self.embed_tokens = nn.Embedding(config["vocab_size"], dim)
            self.embed_positions = nn.Embedding(config["max_positions"], dim)
            self.layers = nn.ModuleList(layer() for _ in range(count))

    class Marian(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.shared = nn.Embedding(config["vocab_size"], dim)
            self.model.encoder = Stack(EncoderLayer, config["encoder_layers"])
            self.model.decoder = Stack(DecoderLayer, config["decoder_layers"])
            self.register_buffer("final_logits_bias", torch.zeros(1, config["vocab_size"]))

        def encode(self, ids):
            encoder = self.model.encoder
            x = encoder.embed_tokens(ids) * math.sqrt(dim)
            x = x + encoder.embed_positions(torch.arange(ids.shape[1], device=ids.device))
            for layer in encoder.layers:
                x = layer(x)
            return x

        def decode(self, ids, memory):
            decoder = self.model.decoder
            length = ids.shape[1]
            x = decoder.embed_tokens(ids) * math.sqrt(dim)
            x = x + decoder.embed_positions(torch.arange(length, device=ids.device))
            causal = torch.full((length, length), torch.finfo(x.dtype).min, device=ids.device).triu(1)
            for layer in decoder.layers:
                x = layer(x, memory, causal)
            return x[:, -1] @ self.model.shared.weight.T + self.final_logits_bias[0]

    return Marian()


def load_model(path: Path, *, device: str = "cuda:0"):
    """配布の ``pytorch_model.bin``（weights_only）を厳密に読む。"""
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise TranslationError("訳はCUDAで実行します（CPUへ切り替えません）")
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    model = build_model()
    expected = set(model.state_dict())
    if set(state) != expected:
        raise TranslationError(f"訳のmodelの重みの名前が合いません: missing={sorted(expected - set(state))[:5]} "
                               f"unexpected={sorted(set(state) - expected)[:5]}")
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device).eval()


class Translator:
    """日本語の短い句を、英語の候補（最良から順）へ訳す。"""

    def __init__(self, paths: dict[str, Path], *, device: str = "cuda:0"):
        with _LOAD_LOCK:
            self.source = SentencePiece(paths["source"])
            vocabulary = json.loads(Path(paths["vocab"]).read_text(encoding="utf-8"))
            self.ids = {str(piece): int(index) for piece, index in vocabulary.items()}
            self.pieces = {index: piece for piece, index in self.ids.items()}
            self.model = load_model(paths["model"], device=device)
        self.device = device

    def close(self) -> None:
        import torch

        self.model = None
        torch.cuda.empty_cache()

    def source_ids(self, text: str) -> list[int]:
        pieces = self.source.encode(unicodedata.normalize("NFC", text))
        ids = [self.ids.get(piece, CONFIG["unk_id"]) for piece in pieces][:MAX_SOURCE_TOKENS]
        return ids + [CONFIG["eos_id"]]

    def text_of(self, ids) -> str:
        special = {CONFIG["eos_id"], CONFIG["pad_id"]}
        pieces = [self.pieces.get(int(index), "<unk>") for index in ids if int(index) not in special]
        return "".join(pieces).replace(_SPACE, " ").strip()

    def translate(self, text: str, *, candidates: int = 4) -> list[tuple[str, float]]:
        """訳の候補（文, 長さで割ったlog確率）を、良い順に ``candidates`` 個まで。"""
        import torch

        config, search = CONFIG, SEARCH
        beams, penalty = search["num_beams"], search["length_penalty"]
        keep = 2 * beams
        with torch.inference_mode():
            source = torch.tensor([self.source_ids(text)], device=self.device)
            memory = self.model.encode(source)
            running = torch.full((beams, 1), config["decoder_start_id"], dtype=torch.long, device=self.device)
            running_scores = torch.full((beams,), -1.0e9, device=self.device)
            running_scores[0] = 0.0
            finished: list[tuple[float, list[int]]] = []
            max_length = 1 + search["max_new_tokens"]
            for current in range(1, max_length):
                logits = self.model.decode(running, memory.expand(beams, -1, -1)).float()
                log_probs = torch.log_softmax(logits, dim=-1)
                log_probs[:, config["pad_id"]] = -math.inf
                if current == max_length - 1:
                    forced = torch.full_like(log_probs, -math.inf)
                    forced[:, config["eos_id"]] = 0.0
                    log_probs = forced
                if search["renormalize_logits"]:
                    log_probs = torch.log_softmax(log_probs, dim=-1)
                total = (log_probs + running_scores[:, None]).reshape(-1)
                top_scores, top_index = torch.topk(total, keep)
                vocab = log_probs.shape[1]
                origin, token = top_index // vocab, top_index % vocab
                hits = (token == config["eos_id"]) | (current + 1 >= max_length)
                # 完了: 上位 beam 数の中で終わった候補だけ。得点は生成した長さ（終端を含む）で割る。
                if self._improvable(running_scores, finished, current, beams, penalty):
                    for rank in range(beams):
                        if bool(hits[rank]):
                            sequence = running[origin[rank]].tolist()[1:] + [int(token[rank])]
                            finished.append((float(top_scores[rank]) / (current ** penalty), sequence))
                    finished.sort(key=lambda item: -item[0])
                    finished = finished[:beams]
                masked = top_scores + hits.float() * -1.0e9
                order = torch.topk(masked, beams).indices
                running = torch.cat([running[origin[order]], token[order, None]], dim=1)
                running_scores = masked[order]
                if bool(hits.all()) or not self._improvable(running_scores, finished, current + 1, beams, penalty):
                    break
        return [(self.text_of(sequence), score) for score, sequence in finished[:candidates]]

    @staticmethod
    def _improvable(running_scores, finished, length: int, beams: int, penalty: float) -> bool:
        if len(finished) < beams:
            return True
        best_running = float(running_scores.max()) / (max(1, length - 1) ** penalty)
        return best_running > min(score for score, _ in finished)
