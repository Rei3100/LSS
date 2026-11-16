"""利用者が言葉で書いた「欲しい音」を、LSSが扱える要求（SoundRequest）へ読み解く。

1つの入力欄には、読点・カンマ・改行で区切って複数の欲しい音を書ける。区切られた1つが1つの成果物になる。

1つの欲しい音の書き方:

- 語彙の音: ``ピアノ`` / ``雨の音`` / ``drums``
- 合わせた音（1つの成果物）: ``ドラムとベース`` / ``drums + bass``
- 取り除いた音: ``台詞なし`` / ``ボーカル抜き`` / ``ドラム以外`` / ``without vocals``
- 範囲を絞った音: ``BGMのピアノ`` / ``効果音の足音`` / ``piano in the music``
- 修飾した音: ``子どもの笑い声`` / ``遠くの雷`` / ``怪獣の足音``（語彙の修飾語 ``lexicon.MODIFIERS`` ＋ 言葉で分離する
  語彙の音）。元の音と同じ経路・同じ確かめ方で、言葉だけを ``child laughter`` のように絞る。
- 語彙に無い英語の名詞句は、そのまま言葉で分離する（``sound`` 系統）。句が語彙の音を1つだけ含むとき
  （``distant thunder`` / ``rain hitting the window``）は、その音と同じ経路・確かめ方で、言葉は句のまま使う。

語彙に無い日本語は、別の音へ推測で置き換えない。workerは ``interpreter`` （`lss.sound.interpret`）を渡せる:
語彙に無い日本語の句を英語へ訳し、訳が元の日本語と同じ意味だと確かめられたときだけ、その英語の音として読む
（``heard_as`` に訳を残す）。確かめられなければ読めなかった語を ``unknown`` に残し、成果物を作らない理由として
正直に返す。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from lss.sound import lexicon
from lss.sound.lexicon import Concept

SCHEMA_VERSION = 1
MAX_REQUESTS = 12
MAX_TEXT = 120
_SEPARATORS = re.compile(r"[、，,;；\n\r/／|]+")
_UNION = re.compile(r"\s*(?:と|＆|&|\+|＋|・| and | with )\s*")
_REMOVE_SUFFIX = re.compile(r"(?:を)?(?:なし|無し|抜き|ぬき|以外|を除く|除く|を除いた|除いた|を消した|消した|を抜いた|"
                            r"抜いた|カット|除去|を消す|消す|オフ|を消して|消して|を除いて|除いて|を取り除いて|取り除いて|"
                            r"を取り除く|取り除く|を抜いて|抜いて|なしで|抜きで)$")
_REMOVE_PREFIX = re.compile(r"^(?:no |without |everything but |everything except |all but |remove |minus )")
_POLITE_SUFFIX = re.compile(r"(?:の音だけ|の音のみ|だけ|のみ|を抽出(?:して)?|を抜き出(?:す|して)?|を取り出(?:す|して)?|"
                            r"を分離(?:して)?|が欲しい|がほしい|を聞きたい|が聞きたい|を残(?:す|して)|ください|下さい|"
                            r"お願いします|お願い| only| please| sounds?)$")
_POLITE_PREFIX = re.compile(r"^(?:extract |isolate |separate |get |keep |just |only |the |a |an )")
_SCOPE = {"music": ("bgm", "音楽", "曲", "劇伴", "サントラ", "music", "the music", "soundtrack", "score"),
          "effects": ("効果音", "環境音", "se", "sfx", "effects", "the effects", "sound effects"),
          "speech": ("台詞", "セリフ", "声", "speech", "dialogue"),
          "singing": ("歌", "歌声", "ボーカル", "歌声全部", "vocals", "all vocals"),
          "instrumental": ("伴奏", "カラオケ", "インスト", "instrumental")}
_SCOPE_FAMILY = {"music": "music", "effects": "effects", "speech": "speech", "singing": "singing",
                 "instrumental": "instrumental"}


# まだ実現できない指定（位置・音量・時間の範囲として読めなかった時間の言葉）。訳すと英語の句として通って
# しまうので、音の言葉として読まずに「まだ扱えない指定」として返す（指定を黙って無視した成果物を作らない）。
# 読める時間の範囲（「前半の」「1分から2分の」「最初の30秒の」）は `_time_window` が先に取り除く。
_UNSUPPORTED_SCOPE = re.compile(
    r"\d+\s*(?:分|秒|時間|min|mins|minute|minutes|sec|secs|second|seconds)(?![a-z])|前半|後半|冒頭|序盤|終盤|最初の|最後の|"
    r"左チャンネル|右チャンネル|左側の|右側の|左の|右の|\bleft (?:channel|side)\b|\bright (?:channel|side)\b|"
    r"大きい音|大きな音|小さい音|小さな音|うるさい|静かな|\bloud\b|\bquiet\b|\bfirst half\b|\bsecond half\b")
_GENERIC_WORDS = frozenset({"sound", "sounds", "noise", "noises", "audio", "all", "everything", "every", "other",
                            "others", "thing", "things", "something", "stuff", "the", "a", "an", "of", "and", "some",
                            "any", "this", "that", "it", "its", "all the", "whole"})


def scope_not_supported(text: str) -> bool:
    """時間・位置・音量のような、まだ実現できない指定を含むか。"""
    return bool(_UNSUPPORTED_SCOPE.search(lexicon.normalize(text)))


@dataclass(frozen=True)
class Window:
    """欲しい音を探す時間の範囲。``seconds``: 先頭からの秒（start<0は最後からの秒、end=Noneは最後まで）。
    ``fraction``: 全体に対する割合（前半 0〜0.5）。``label`` は利用者の言い方（成果物の名前に使う）。"""
    unit: str
    start: float
    end: float | None
    label: str

    def bounds(self, duration: float) -> tuple[float, float]:
        if self.unit == "fraction":
            return duration * self.start, duration * (1.0 if self.end is None else self.end)
        start = duration + self.start if self.start < 0 else self.start
        end = duration if self.end is None else self.end
        return max(0.0, min(start, duration)), max(0.0, min(end, duration))

    def identity(self) -> dict:
        return {"unit": self.unit, "start": self.start, "end": self.end}


_T = r"(?:(\d+)分(?:(\d+(?:\.\d+)?)秒)?|(\d+(?:\.\d+)?)秒|(\d+):(\d\d(?:\.\d+)?))"
_UNIT = {"秒": 1.0, "分": 60.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0, "s": 1.0,
         "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0}


def _seconds(groups) -> float:
    minutes, minute_seconds, seconds, clock_minutes, clock_seconds = groups
    if minutes is not None:
        return int(minutes) * 60.0 + float(minute_seconds or 0)
    if seconds is not None:
        return float(seconds)
    return int(clock_minutes) * 60.0 + float(clock_seconds)


def _time_window(value: str) -> tuple[Window | None, str]:
    """「前半の足音」「1分から2分のBGM」「最初の30秒のピアノ」「first 30 seconds of rain」の時間の範囲と残り。"""
    match = re.fullmatch(r"(前半|後半)の?(.+)", value)
    if match:
        first = match.group(1) == "前半"
        return Window("fraction", 0.0 if first else 0.5, 0.5 if first else None, match.group(1)), match.group(2)
    match = re.fullmatch(r"(?:the )?(first|second) half of (?:the )?(.+)", value) or \
        re.fullmatch(r"(.+?) in the (first|second) half", value)
    if match:
        part, rest = (match.group(1), match.group(2)) if value.startswith(("first", "second", "the ")) \
            else (match.group(2), match.group(1))
        first = part == "first"
        return Window("fraction", 0.0 if first else 0.5, 0.5 if first else None, "前半" if first else "後半"), rest
    match = re.fullmatch(r"((?:最初|はじめ|冒頭|最後)の(\d+(?:\.\d+)?)(秒|分)(?:間)?)の?(.+)", value)
    if match:
        length = float(match.group(2)) * _UNIT[match.group(3)]
        last = match.group(1).startswith("最後")
        return Window("seconds", -length if last else 0.0, None if last else length, match.group(1)), match.group(4)
    match = re.fullmatch(r"(?:the )?(first|last) (\d+(?:\.\d+)?) ?(seconds?|secs?|s|minutes?|mins?) of (?:the )?(.+)", value)
    if match:
        length = float(match.group(2)) * _UNIT[match.group(3)]
        last = match.group(1) == "last"
        label = f"{'最後' if last else '最初'}の{match.group(2)}{'秒' if _UNIT[match.group(3)] == 1.0 else '分'}"
        return Window("seconds", -length if last else 0.0, None if last else length, label), match.group(4)
    match = re.fullmatch(rf"({_T}から{_T}(?:まで)?)の?(.+)", value)
    if match:
        groups = match.groups()
        start, end = _seconds(groups[1:6]), _seconds(groups[6:11])
        if end > start:
            return Window("seconds", start, end, match.group(1).replace(":", "分")), groups[11]
        return None, value
    match = re.fullmatch(rf"({_T}から(?:最後まで|後|以降)?)の?(.+)", value)
    if match:
        groups = match.groups()
        return Window("seconds", _seconds(groups[1:6]), None, match.group(1).replace(":", "分")), groups[6]
    match = re.fullmatch(rf"({_T}まで)の?(.+)", value)
    if match:
        groups = match.groups()
        end = _seconds(groups[1:6])
        if end > 0:
            return Window("seconds", 0.0, end, match.group(1).replace(":", "分")), groups[6]
    match = re.fullmatch(r"(.+?) (?:from|between) (\d+):(\d\d) (?:to|and|-) (\d+):(\d\d)", value)
    if match:
        start = int(match.group(2)) * 60.0 + int(match.group(3))
        end = int(match.group(4)) * 60.0 + int(match.group(5))
        if end > start:
            label = f"{match.group(2)}分{match.group(3)}秒から{match.group(4)}分{match.group(5)}秒"
            return Window("seconds", start, end, label), match.group(1)
    return None, value


@dataclass(frozen=True)
class Atom:
    """欲しい音を構成する1つの対象。語彙の音か、語彙に無い英語の名詞句。"""
    concept: Concept

    @property
    def open_vocabulary(self) -> bool:
        return self.concept.id.startswith("open:")


@dataclass(frozen=True)
class SoundRequest:
    text: str                       # 利用者が書いたとおり（前後の空白だけ除く）
    operation: str                  # "extract" / "remove"
    atoms: tuple[Atom, ...] = ()
    scope: str | None = None        # 範囲を絞る親（"music" 等）。Noneは対象の系統から決める
    unknown: tuple[str, ...] = field(default_factory=tuple)
    heard_as: tuple[str, ...] = field(default_factory=tuple)   # 訳して読んだ日本語の、英語の読み方
    reason: str = ""                # 読めなかった理由（"unknown_words" / "scope_not_supported"）
    suggestions: tuple[str, ...] = field(default_factory=tuple)   # 読めなかった語に近い語彙の語（読み替えはしない）
    window: Window | None = None    # 探す時間の範囲（「前半の足音」）。Noneは全体

    @property
    def supported(self) -> bool:
        return bool(self.atoms) and not self.unknown

    @property
    def label(self) -> str:
        names = "と".join(atom.concept.label for atom in self.atoms) or self.text
        if self.scope and self.operation == "extract":
            scope_label = lexicon.BY_ID[self.scope].label if self.scope in lexicon.BY_ID else self.scope
            if scope_label != names:
                names = f"{scope_label}の{names}"
        names = f"{names}なし" if self.operation == "remove" else names
        return f"{self.window.label}の{names}" if self.window is not None else names

    def identity(self) -> dict:
        """Recipeへ入る、意味の正規形（表記ゆれでは変わらない）。"""
        body = {"schema_version": SCHEMA_VERSION, "operation": self.operation, "scope": self.scope,
                "atoms": [{"id": atom.concept.id, "family": atom.concept.family, "prompt": atom.concept.prompt}
                          for atom in sorted(self.atoms, key=lambda atom: atom.concept.id)]}
        if self.window is not None:
            body["window"] = self.window.identity()
        return body

    def key(self) -> str:
        canonical = json.dumps(self.identity(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _strip(value: str) -> str:
    previous = None
    value = value.strip()
    while previous != value:
        previous = value
        value = _POLITE_SUFFIX.sub("", value).strip()
        value = _POLITE_PREFIX.sub("", value)
    return value


# 言葉で分離する（promptが結果を決める）系統。修飾した音・英語の句を結び付けられるのはこれらだけ。
# ドラム・ベース・男声・女声は専門modelのstemそのもの（promptを使わない）なので、修飾を黙って捨てないよう除く。
_TEXT_FAMILIES = frozenset({"sound", "speech_part", "singing_part", "music_part"})
_STEM_ONLY = frozenset({"drums", "bass", "male_singing", "female_singing"})


def _text_routed(concept: Concept) -> bool:
    return concept.family in _TEXT_FAMILIES and concept.base_id not in _STEM_ONLY \
        and not concept.id.startswith("open:")


def _narrowed(head: Concept, prompt: str, label: str) -> Concept | None:
    """語彙の音 ``head`` と同じ経路・確かめ方で、言葉だけを ``prompt`` に絞った音。"""
    from lss.sound import t5_text

    try:
        prompt = t5_text.normalized_prompt(prompt)
    except t5_text.TextEncoderError:
        return None
    if prompt == head.prompt:
        return head
    return Concept(id=f"{head.base_id}~{prompt}", label=label, family=head.family, prompt=prompt,
                   words=(), audioset=head.audioset, parent=head.parent)


def label_safe(text: str) -> str:
    """利用者の言葉を、成果物の名前に使える形へ（ファイル名に使えない文字を除く）。"""
    return lexicon.normalize(re.sub(r'[\\/:*?"<>|]+', " ", text)).strip() or text


def _singular(text: str) -> str:
    """英語の語ごとの単数形（照合用。bubbles/bubble、sparrows/sparrow、'sを同じに扱う）。"""
    words = []
    for word in re.split(r"\s+", text.replace("'s ", " ").replace("'s", "")):
        if len(word) > 3 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith(("ches", "shes", "sses", "xes")):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        words.append(word)
    return " ".join(word for word in words if word)


_ENGLISH_WORDS = sorted({(_singular(key), concept) for key, concept in lexicon.WORDS.items()
                         if re.fullmatch(r"[a-z0-9 '\-]+", key)}, key=lambda item: (-len(item[0]), item[1].id))
# 日本語の句の中で、音の種類を言わない総称（「猫が鳴く声」の「声」は台詞ではない）。
_GENERIC_HEADS = frozenset({"声", "音"})
_JAPANESE_WORDS = sorted(((key, concept) for key, concept in lexicon.WORDS.items()
                          if re.search("[぀-ヿ㐀-鿿]", key) and key not in _GENERIC_HEADS),
                         key=lambda item: -len(item[0]))
_KANJI = re.compile("[㐀-鿿豈-﫿]")
# 1文字の漢字の語の直前に来てよい文字: 助詞・区切りと、形容詞・動詞の連体形の語尾（「強い風」「流れる水」）。
_PARTICLES = frozenset("のがをにとではもへや、。・ 　いなるうくすつむぶぐただ")


def english_mentions(phrase: str) -> list[Concept]:
    """英語の句に出てくる語彙の音（長い語から、重ならないように、単数・複数を区別せず）。"""
    text = f" {_singular(lexicon.normalize(phrase))} "
    found: list[Concept] = []
    for key, concept in _ENGLISH_WORDS:
        token = f" {key} "
        while token in text:
            text = text.replace(token, " | ", 1)
            if all(existing.base_id != concept.base_id for existing in found):
                found.append(concept)
    return found


def japanese_mentions(text: str) -> list[Concept]:
    """日本語の句に出てくる語彙の音（長い語から、重ならないように）。1文字の漢字の語（雨・猫等）は、前後が
    漢字でないときだけ数える（「歌舞伎」の「歌」を歌とみなさない）。"""
    value = lexicon.normalize(text)
    taken = [False] * len(value)
    found: list[Concept] = []
    for key, concept in _JAPANESE_WORDS:
        start = value.find(key)
        while start >= 0:
            end = start + len(key)
            free = not any(taken[start:end])
            if free and len(key) == 1 and _KANJI.match(key):
                # 1文字の漢字の語は、独立した語のときだけ（前が句頭・助詞・区切り、後ろが漢字でない）。
                # 「かめはめ波」の「波」、「歌舞伎」の「歌」は語の一部。
                free = (start == 0 or value[start - 1] in _PARTICLES) and \
                       (end == len(value) or not _KANJI.match(value[end]))
            if free:
                for index in range(start, end):
                    taken[index] = True
                if all(existing.base_id != concept.base_id for existing in found):
                    found.append(concept)
            start = value.find(key, start + 1)
    return found


def _open_concept(text: str) -> Concept | None:
    from lss.sound import t5_text

    try:
        prompt = t5_text.normalized_prompt(text)
    except t5_text.TextEncoderError:
        return None
    if all(word in _GENERIC_WORDS for word in re.split(r"[\s,&\-']+", prompt) if word):
        return None   # 「音」「全部」だけでは、どの音かが決まらない
    mentioned = english_mentions(prompt)
    if len(mentioned) == 1 and mentioned[0].base_id in _STEM_ONLY:
        return mentioned[0]   # "heavy bass": 専門modelのstemそのもの（修飾はstemの中身を変えない）
    if len(mentioned) == 1 and _text_routed(mentioned[0]):
        # 語彙の音を1つだけ含む句（"distant thunder"）: その音と同じ経路・確かめ方で、言葉は句のまま。
        return _narrowed(mentioned[0], prompt, prompt)
    return Concept(id=f"open:{prompt}", label=prompt, family="sound", prompt=prompt, words=(prompt,))


_SOUND_SUFFIXES = ("の音色", "の効果音", "の音", "の声", "の鳴き声", "の物音", "音色", "効果音", "の響き", "音",
                   " sound effect", " sound effects", " sound", " sounds", " noise", " noises")


def _lookup_sound(word: str) -> Concept | None:
    found = lexicon.lookup(word)
    if found is not None:
        return found
    for suffix in _SOUND_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix):
            found = lexicon.lookup(word[: -len(suffix)])
            if found is not None:
                return found
    return None


def _modified(word: str) -> Concept | None:
    """「子どもの笑い声」「遠くの雷」「怪獣 足音」: 語彙の修飾語 ＋ 言葉で分離する語彙の音。"""
    value = lexicon.normalize(word)
    for key in sorted(lexicon.MODIFIERS, key=len, reverse=True):
        normalized = lexicon.normalize(key)
        if not value.startswith(normalized):
            continue
        rest = value[len(normalized):].strip()
        rest = re.sub(r"^(?:の|で|な|に|が)\s*", "", rest).strip()
        if not rest:
            continue
        head = _lookup_sound(rest)
        if head is not None and head.base_id in _STEM_ONLY:
            return head   # 「激しいドラム」: 専門modelのstemそのもの（修飾はstemの中身を変えない）
        if head is None or not _text_routed(head):
            continue
        modifier = lexicon.MODIFIERS[key]
        if set(_stems(modifier)) <= set(_stems(head.prompt)):
            return head   # 「鳥のさえずり」: 修飾語が既に音の言葉に入っている
        return _narrowed(head, f"{modifier} {head.prompt}", label_safe(word))
    return _inner_modified(value, word)


_MODIFIER_KEYS = sorted((lexicon.normalize(key) for key in lexicon.MODIFIERS), key=len, reverse=True)


def _inner_modified(value: str, word: str) -> Concept | None:
    """修飾語が句の途中にある形（「男性の低い歌声」「古い木のドア」）: 修飾語を除いた残りが語彙の音なら、
    その音を修飾語で絞る（残りが「男性の歌声」のように修飾込みの語彙の音でもよい）。"""
    from itertools import combinations

    occurrences, taken = [], [False] * len(value)
    for key in _MODIFIER_KEYS:
        index = value.find(key)
        if index < 0 or len(key) < 2 or any(taken[index:index + len(key)]):
            continue
        after = value[index + len(key):]
        following = re.match(r"(?:の|な|で)?", after).group(0)
        if not after[len(following):]:
            continue   # 修飾語の後ろに音の語が無い
        for position in range(index, index + len(key)):
            taken[position] = True
        english = next(lexicon.MODIFIERS[k] for k in lexicon.MODIFIERS if lexicon.normalize(k) == key)
        occurrences.append((index, index + len(key) + len(following), english))
    for size in range(1, min(3, len(occurrences)) + 1):
        for chosen in combinations(sorted(occurrences), size):
            rest, cursor = "", 0
            for start, end, _ in chosen:
                rest, cursor = rest + value[cursor:start], end
            head = _lookup_sound((rest + value[cursor:]).strip())
            if head is None:
                continue
            if head.base_id in _STEM_ONLY:
                return head
            if not _text_routed(head):
                continue
            words = [english for _, _, english in chosen if not set(_stems(english)) <= set(_stems(head.prompt))]
            return _narrowed(head, " ".join(words + [head.prompt]), label_safe(word)) if words else head
    return None


def _stems(text: str) -> list[str]:
    return [re.sub(r"(?:es|s)$", "", word) for word in re.split(r"[\s\-]+", text.lower()) if word]


def _resolve(word: str, *, open_vocabulary: bool = True) -> Concept | None:
    word = word.strip()
    if not word:
        return None
    found = _lookup_sound(word)
    if found is not None:
        return found
    found = _modified(word)
    if found is not None:
        return found
    if open_vocabulary and re.fullmatch(r"[A-Za-z0-9 '\-,&]+", word):
        return _open_concept(word)
    return None


def suggestions(word: str, limit: int = 3) -> tuple[str, ...]:
    """読めなかった語に近い語彙の語（打ち間違い・表記ゆれの手掛かり）。推測で読み替えはしない。"""
    import difflib

    value = lexicon.normalize(word)
    japanese = bool(_JAPANESE.search(value))
    pool = [key for key in lexicon.WORDS if bool(_JAPANESE.search(key)) == japanese]
    close = difflib.get_close_matches(value, pool, n=limit * 3, cutoff=0.6 if japanese else 0.75)
    labels: list[str] = []
    for key in close:
        label = lexicon.WORDS[key].label if japanese else key
        if label not in labels:
            labels.append(label)
    return tuple(labels[:limit])


def _scope_of(word: str) -> str | None:
    value = lexicon.normalize(word)
    for scope, words in _SCOPE.items():
        if value in {lexicon.normalize(w) for w in words}:
            return scope
    return None


_JAPANESE = re.compile("[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uff66-\uff9f]")  # かな・漢字・半角カナ


def _interpret(word: str, interpreter, heard: list[str]) -> Concept | None:
    """語彙に無い日本語を、訳して確かめられたときだけ英語の音として読む。"""
    if interpreter is None or not _JAPANESE.search(word):
        return None
    found = interpreter(word)
    if found is None:
        return None
    concept, english = found
    heard.append(english)
    return concept


def _union(value: str, *, open_vocabulary: bool, interpreter=None,
           heard: list[str] | None = None) -> tuple[list[Concept], list[str]]:
    concepts, unknown = [], []
    for part in [part for part in _UNION.split(value) if part.strip()]:
        word = _strip(part)
        concept = _resolve(word, open_vocabulary=open_vocabulary)
        if concept is None and heard is not None:
            concept = _interpret(word, interpreter, heard)
        if concept is None:
            unknown.append(part.strip())
        elif all(existing.id != concept.id for existing in concepts):
            concepts.append(concept)
    return concepts, unknown


def parse_one(text: str, interpreter=None) -> SoundRequest:
    """1つの欲しい音を読む。時間の範囲（「前半の」「1分から2分の」）を除いた残りを、語彙の完全一致 → 範囲付き →
    合わせた音 → 英語の名詞句 → （``interpreter`` があれば）訳して確かめた日本語、の順に試す。"""
    from dataclasses import replace

    original = " ".join(str(text).split())[:MAX_TEXT]
    window, value = _time_window(_strip(lexicon.normalize(original)))
    request = _parse_sound(original, _strip(value), interpreter)
    return replace(request, window=window) if window is not None and request.supported else request


def _parse_sound(original: str, value: str, interpreter) -> SoundRequest:
    if scope_not_supported(value):
        return SoundRequest(original, "extract", (), None, (original,), reason="scope_not_supported")
    operation = "extract"
    if _REMOVE_PREFIX.match(value):
        operation = "remove"
        value = _REMOVE_PREFIX.sub("", value).strip()
    elif _REMOVE_SUFFIX.search(value) and lexicon.lookup(value) is None:
        operation = "remove"
        value = _REMOVE_SUFFIX.sub("", value).strip()
    value = _strip(value)
    exact = _resolve(value, open_vocabulary=False)
    if exact is not None:
        return SoundRequest(original, operation, (Atom(exact),))
    scope = None
    match = re.fullmatch(r"(.+?)の(.+)", value) or re.fullmatch(r"(.+?) (?:in|from|of) (?:the )?(.+)", value)
    if match:
        # 「AのB」は Aが範囲（BGM・効果音・台詞・歌・伴奏）なら範囲付き、英語は「B in A」。
        head, tail = (match.group(1), match.group(2)) if "の" in value else (match.group(2), match.group(1))
        if _scope_of(head) is not None:
            scope = _scope_of(head)
            value = _strip(tail)
        else:
            # 「ドラムのシンバル」「ベースのウッドベース」: 楽器の中の部位はその部位（経路も部位の親で取る）
            container, part = _lookup_sound(head), _lookup_sound(tail)
            if container is not None and part is not None and container.base_id in _STEM_ONLY \
                    and part.base_id != container.base_id and _text_routed(part) and part.family == "music_part":
                return SoundRequest(original, operation, (Atom(part),))
    concepts, unknown = _union(value, open_vocabulary=False)
    heard: list[str] = []
    if not concepts or unknown:
        whole = _resolve(value)
        if whole is not None:
            concepts, unknown = [whole], []
    if (unknown or not concepts) and interpreter is not None:
        # 句全体を訳すほうが意味を保つ（「雨が窓に当たる音」）。だめなら「と」で分けた部分ごとに。
        whole = _interpret(value, interpreter, heard)
        if whole is not None:
            concepts, unknown = [whole], []
        elif len([part for part in _UNION.split(value) if part.strip()]) > 1:
            heard = []
            concepts, unknown = _union(value, open_vocabulary=True, interpreter=interpreter, heard=heard)
    if unknown or not concepts:
        missing = tuple(unknown or [value])
        hints = tuple(dict.fromkeys(hint for word in missing for hint in suggestions(_strip(word))))[:3]
        return SoundRequest(original, operation, (), scope, missing, reason="unknown_words", suggestions=hints)
    return SoundRequest(original, operation, tuple(Atom(c) for c in concepts), _effective_scope(scope, concepts),
                        heard_as=tuple(heard))


_NATURAL_PARENT = {"speech_part": "speech", "singing_part": "singing", "music_part": "music", "sound": "effects",
                   "speech": "speech", "singing": "music", "lead_vocals": "singing", "backing_vocals": "singing",
                   "instrumental": "music", "music": "music", "effects": "effects"}


def _effective_scope(scope: str | None, concepts) -> str | None:
    """範囲が対象の本来の親（「BGMのピアノ」の音楽等）と同じなら、範囲は無いのと同じ。"""
    if scope is None:
        return None
    if all(_NATURAL_PARENT.get(concept.family) == scope or concept.family == _SCOPE_FAMILY[scope]
           for concept in concepts):
        return None
    return scope


def split_entries(text: str) -> list[str]:
    """入力欄の文字列を、欲しい音ごとに分ける（空・重複は除く）。"""
    entries, seen = [], set()
    for part in _SEPARATORS.split(str(text or "")):
        value = " ".join(part.split())
        if value and lexicon.normalize(value) not in seen:
            seen.add(lexicon.normalize(value))
            entries.append(value[:MAX_TEXT])
    return entries[:MAX_REQUESTS]


def parse(entries, interpreter=None) -> list[SoundRequest]:
    """欲しい音の一覧を読む。同じ意味の要求は1つにまとめる。"""
    if isinstance(entries, str):
        entries = split_entries(entries)
    requests, keys = [], set()
    for entry in list(entries)[:MAX_REQUESTS]:
        request = parse_one(entry, interpreter)
        key = request.key() if request.supported else ("unknown", lexicon.normalize(request.text))
        if key in keys:
            continue
        keys.add(key)
        requests.append(request)
    return requests
