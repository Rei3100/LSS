"""欲しい音の要求を、音源の性質に合わせた抽出の経路（式）へ組む。

経路は小さな式で、同じ式は同じ計算になる（`lss.sound.extract` が節ごとに一度だけ計算して使い回す）。

- ``source``: 元の音（復号したそのもの）
- ``dnr``: 台詞・音楽・効果音の3分類（SFC-Locoformer、BGMと同じ解析器）
- ``vocals``: 歌声全部と伴奏（becruily deux、Instrumentalと同じ既定model）
- ``lead``: 歌声全部の中のリードボーカル（becruily karaoke。コーラス・ハーモニーを含まない）
- ``stems``: 音楽のドラム・ベース・その他・歌声（SCNet XL IHF。語彙の楽器の専門modelが無い音を言葉で取る親）
- ``model``: 専門modelの1つの出力（``MODEL_STEMS``。1つのmodelの全出力は1回の推論でまとめて計算する）
- ``text``: 言葉で指定した音（SAM Audio、親の音へのmask投影）
- ``sum`` / ``subtract``: 合わせた音・取り除いた音

選び方（制御混合と実音声の評価で決めた。`docs/engine/WANTED_SOUND.md`）:

- 台詞・音楽・効果音の全体は、専門の3分類が言葉の分離より良い（台詞 14.0 dB 対 7.8 dB）。
- 「ボーカル」はリードボーカルだけ（コーラス・ハーモニーを含まない）。歌声全部からkaraoke modelでリードを
  取り、バッキング（コーラスとハーモニー）は歌声全部からリードを引いた残り。コーラスとハーモニーを
  互いに分ける学習済みmodelは無く、信号処理の分け方は制御実験で分けない場合より悪かったので、
  両方を含むバッキングとして返す。
- 男声・女声は、歌声全部から男女の専門model（chorus male/female）で分ける。言葉の分離は男声で破綻した
  （制御デュエット: 同時 -6.4 / 交互 -12.0 dB、専門model -2.0 / +1.0 dB。女声 -3.0 / -1.2 対 +2.3 / +0.3）。
- 歌声・ドラム・ベースは専門modelが言葉の分離より良い（MUSDB18 7秒版: 歌声 11.4 対 3.2 dB、
  ドラム 8.0 対 4.2、ベース 6.8 対 -3.5）。専門modelが無い楽器は、ドラム・ベース・歌声を除いた
  「その他」から言葉で取り出す。打楽器の細部はドラムから言葉で取り出す。
- 楽器（SCNetの「その他」から言葉で取る楽器）は、言葉の分離のmaskが楽器の一部を周波数ごとに落とすので
  （実楽曲で「その他」そのもの 8.6 dB に対し 0.5 dB）、その楽器しか鳴っていない秒は親をそのまま使い、
  それ以外はmaskの下限を0.5にする（``instrument``。8.3 dB、他の楽器と重なる曲 -5.7 → -3.3 dB）。
- 特定の効果音・環境音は、台詞があると元の音から直接言葉で取るより、3分類の効果音から取るほうが
  良い（-1.4 dB 対 5.0 dB）。
- 専門modelがある楽器は、言葉の分離ではなく専門modelで取る（STEM）。ドラム・ベース・ギター・ピアノは
  6分類（BS-RoFormer SW 6stem: 同じ推論の出力どうしを足すので、ベース＋ギターに音の重なりや欠けが出ない）、
  キック・スネア・タム・ハイハット・ライド・クラッシュは6分類のドラムをDrumSep（MDX23C 6stem）で分け、
  ストリングスは弦楽器の専門model（gilliaan BowedStrings v2）、その他の楽器はMVSep Mega 53stemの出力を使う。
  専門modelが無い楽器だけを、SCNetの「その他」などから言葉で取る。
- 番組（台詞のある音源）では音楽の親を3分類の音楽にし、曲（台詞の無い音源）では元の音を親にする。
- 言葉の分離（``text``）には、その音が鳴っている時刻を条件（anchor）として渡す（``anchors: "frames"``。
  時刻は `lss.sound.pe_frame` が親の音で測る。upstreamの ``predict_spans`` と同じ使い方）。
- 語彙に無い言葉（英語の名詞句・訳した日本語）は、固定の親を決めない。候補の親（番組なら台詞・音楽・
  効果音、曲なら歌・ドラム・ベース・その他）のうち、その言葉の音が最も聞こえる親を音源ごとに測って選ぶ
  （``open_parents``）。どの親にも聞こえなければ、その音は無い。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace

from lss.sound.intent import SoundRequest

SOURCE = {"node": "source"}
_DRUM_PARTS = frozenset({"kick", "snare", "hihat", "cymbal", "percussion", "timpani", "taiko", "tambourine",
                         "tsuzumi", "big_drum", "maracas", "cowbell", "castanets", "triangle", "congas", "drum_roll",
                         "tabla"})
_BASS_PARTS = frozenset({"double_bass", "synth_bass"})
_VOICE_PARTS = {"male_singing": "male", "female_singing": "female"}

# 専門modelの出力の名前（configの instruments と同じ綴り）。`lss.sound.extract.MODELS` が重みを持つ。
MODEL_STEMS = {
    "sw6": ("bass", "drums", "other", "vocals", "guitar", "piano"),
    "drumsep6": ("kick", "snare", "toms", "hh", "ride", "crash"),
    "bowed_strings": ("strings", "other"),
    "mega53": ("accordion", "acoustic-guitar", "back-vocal", "banjo", "bass", "bassoon", "bells", "bowed_strings",
               "brass", "cello", "clarinet", "congas", "digital-piano", "dobro", "double-bass", "drums",
               "electric-guitar", "flute", "french-horn", "glockenspiel", "guitar", "harmonica", "harp",
               "harpsichord", "hh", "keys", "kick", "lead-vocal", "mandolin", "marimba", "oboe", "organ",
               "percussion", "piano", "saxophone", "sitar", "snare", "strings", "synth", "tambourine", "timpani",
               "toms", "triangle", "trombone", "trumpet", "tuba", "ukulele", "viola", "violin", "vocal", "wind",
               "wind-chimes", "woodwind"),
}
# 語彙の楽器（``lexicon`` のid）→ 専門modelの出力。ここに無い楽器は言葉で取る。
_SW6_PARTS = {"drums": "drums", "bass": "bass", "guitar": "guitar", "piano": "piano"}
_DRUMSEP_PARTS = {"kick": "kick", "snare": "snare", "toms": "toms", "hihat": "hh", "ride_cymbal": "ride", "crash_cymbal": "crash"}
_STRINGS_PARTS = {"strings": "strings"}
_MEGA53_PARTS = {
    "accordion": "accordion", "acoustic_guitar": "acoustic-guitar", "banjo": "banjo", "bassoon": "bassoon",
    "brass": "brass", "cello": "cello", "clarinet": "clarinet", "congas": "congas", "double_bass": "double-bass",
    "electric_guitar": "electric-guitar", "electric_piano": "digital-piano", "flute": "flute",
    "french_horn": "french-horn", "glockenspiel": "glockenspiel", "harmonica": "harmonica", "harp": "harp",
    "harpsichord": "harpsichord", "keyboard": "keys", "mandolin": "mandolin", "oboe": "oboe", "organ": "organ",
    "percussion": "percussion", "saxophone": "saxophone", "sitar": "sitar", "synth": "synth",
    "tambourine": "tambourine", "timpani": "timpani", "triangle": "triangle", "trombone": "trombone",
    "trumpet": "trumpet", "tuba": "tuba", "ukulele": "ukulele", "viola": "viola", "violin": "violin",
    "winds": "wind", "woodwinds": "woodwind",
}


def specialist(base: str, music: dict) -> tuple[dict, dict] | None:
    """専門modelで取れる楽器の式と、その親の式。取れなければ None。"""
    if base in _SW6_PARTS:
        return model("sw6", _SW6_PARTS[base], music), music
    if base in _DRUMSEP_PARTS or base == "cymbal":
        drums = model("sw6", "drums", music)
        if base == "cymbal":
            return {"node": "sum", "items": [model("drumsep6", "ride", drums),
                                             model("drumsep6", "crash", drums)]}, drums
        return model("drumsep6", _DRUMSEP_PARTS[base], drums), drums
    if base in _STRINGS_PARTS:
        return model("bowed_strings", _STRINGS_PARTS[base], music), music
    if base in _MEGA53_PARTS:
        return model("mega53", _MEGA53_PARTS[base], music), music
    return None


def dnr(stem: str) -> dict:
    return {"node": "dnr", "stem": stem}


def vocals(stem: str, parent: dict) -> dict:
    return {"node": "vocals", "stem": stem, "parent": parent}


def lead(parent: dict) -> dict:
    return {"node": "lead", "parent": parent}


def stems(stem: str, parent: dict) -> dict:
    return {"node": "stems", "stem": stem, "parent": parent}


def voices(stem: str, parent: dict) -> dict:
    """歌声全部の中の男声（"male"）・女声（"female"）。"""
    return {"node": "voices", "stem": stem, "parent": parent}


def model(model_id: str, stem: str, parent: dict) -> dict:
    """専門modelの1つの出力（``MODEL_STEMS[model_id]`` の名前）。"""
    if stem not in MODEL_STEMS[model_id]:
        raise ValueError(f"{model_id} に {stem} の出力はありません")
    return {"node": "model", "model": model_id, "stem": stem, "parent": parent}


def text(prompt: str, parent: dict, *, instrument: str | None = None) -> dict:
    """言葉で取る音。``instrument`` は楽器の語彙の音の識別（maskの下限と、その楽器しか鳴っていない秒は親を
    そのまま使う。`lss.sound.verify` の楽器の規則）。"""
    node = {"node": "text", "prompt": prompt, "parent": parent, "anchors": "frames"}
    if instrument is not None:
        node["instrument"] = instrument
    return node


def canonical(expression: dict) -> str:
    return json.dumps(expression, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Route:
    expression: dict
    parent: dict                  # 音量の比較と区間の選別の基準になる親の音
    classes: tuple[str, ...]      # 出力を確かめるAudioSetクラス
    family: str                   # 区間の選別の仕方を決める系統
    representation: str           # "extracted"（親の部分集合）/ "remainder"（取り除いた残り）/ "source"（元のまま）
    prompts: tuple[str, ...] = ()  # 出力を時刻ごとに確かめる言葉（`lss.sound.pe_frame`）
    open_vocabulary: bool = False  # 語彙に無い言葉だけの要求（確かめは言葉の時刻の証拠だけ）


def parents(kind: str) -> dict:
    cinematic = kind == "cinematic"
    return {"music": dnr("music") if cinematic else SOURCE,
            "effects": dnr("effects") if cinematic else SOURCE,
            "speech": dnr("speech")}


def candidate_parents(kind: str) -> list[dict]:
    """親を測って選ぶ音（語彙に無い言葉・``parent="auto"`` の音）を探す親の候補。"""
    if kind == "cinematic":
        return [dnr("speech"), dnr("music"), dnr("effects")]
    return [stems(name, SOURCE) for name in ("vocals", "drums", "bass", "other")]


def measured(concept) -> bool:
    """固定の親を決めず、音源ごとにその音が最も聞こえる親を測って選ぶ音か。"""
    return concept.id.startswith("open:") or concept.parent == "auto"


def _atom(concept, kind: str, scope: str | None, open_parents: dict | None = None) -> tuple[dict, dict]:
    """1つの対象の式と、その親の式。"""
    p = parents(kind)
    music = p["music"]
    if scope is not None:
        scoped = {"music": music, "effects": p["effects"], "speech": p["speech"],
                  "singing": vocals("vocals", music), "instrumental": vocals("instrumental", music)}[scope]
        return text(concept.prompt, scoped), scoped
    if concept.parent == "auto" and concept.id in (open_parents or {}):
        parent = open_parents[concept.id]
        return text(concept.prompt, parent), parent
    family = concept.family
    base = concept.base_id
    if family == "speech":
        return p["speech"], SOURCE
    if family == "speech_part":
        return text(concept.prompt, p["speech"]), p["speech"]
    if family == "music":
        return music, SOURCE
    if family == "effects":
        return dnr("effects"), SOURCE
    if family == "singing":
        return vocals("vocals", music), music
    if family == "lead_vocals":
        sung = vocals("vocals", music)
        return lead(sung), sung
    if family == "backing_vocals":
        sung = vocals("vocals", music)
        return {"node": "subtract", "base": sung, "minus": lead(sung)}, sung
    if family == "singing_part":
        parent = vocals("vocals", music)
        if base in _VOICE_PARTS:
            return voices(_VOICE_PARTS[base], parent), parent
        return text(concept.prompt, parent), parent
    if family == "instrumental":
        return vocals("instrumental", music), music
    if family == "music_part":
        found = specialist(base, music)
        if found is not None:
            return found
        if base in _DRUM_PARTS:
            parent = stems("drums", music)
            return text(concept.prompt, parent), parent
        if base in _BASS_PARTS:
            parent = stems("bass", music)
            return text(concept.prompt, parent), parent
        if base == "choir":
            return text(concept.prompt, music), music
        parent = stems("other", music)
        return text(concept.prompt, parent, instrument=base), parent
    if family == "sound":
        if concept.id.startswith("open:"):
            parent = (open_parents or {}).get(concept.id, SOURCE)
            return text(concept.prompt, parent), parent
        return text(concept.prompt, p["effects"]), p["effects"]
    raise ValueError(f"未対応の系統です: {family}")


def route(request: SoundRequest, kind: str, *, open_parents: dict | None = None) -> Route:
    """``open_parents`` は語彙に無い言葉（``open:``）ごとの、証拠で選んだ親の式。"""
    if not request.supported:
        raise ValueError("読めなかった要求には経路を作りません")
    if kind not in {"cinematic", "music"}:
        raise ValueError(f"未知の音源の性質です: {kind}")
    built = [_atom(atom.concept, kind, request.scope, open_parents) for atom in request.atoms]
    expressions = [expression for expression, _ in built]
    expression = expressions[0] if len(expressions) == 1 else {"node": "sum", "items": expressions}
    classes = tuple(dict.fromkeys(name for atom in request.atoms for name in atom.concept.audioset))
    families = {atom.concept.family for atom in request.atoms}
    family = families.pop() if len(families) == 1 else "mixed"
    if request.operation == "remove":
        # 取り除いた残り。親は元の音、確かめるのは「元の音から何かが残ったこと」だけ。
        return Route({"node": "subtract", "base": SOURCE, "minus": expression}, SOURCE, (), "remainder",
                     "remainder")
    parent = built[0][1] if len(built) == 1 else SOURCE
    representation = "source" if expression == SOURCE else "extracted"
    prompts = tuple(dict.fromkeys(atom.concept.prompt for atom in request.atoms))
    open_vocabulary = all(atom.open_vocabulary for atom in request.atoms)
    return Route(expression, parent, classes, family, representation, prompts, open_vocabulary)


def removal_targets(request: SoundRequest, kind: str, *, open_parents: dict | None = None) -> list[Route]:
    """Verify each removed sound independently; one present atom cannot authorize another.

    Instrument retention floors deliberately include other sounds. They are useful
    for listening extracts, but cannot authorize removing those sounds from source.
    """
    if request.operation != "remove":
        return []
    targets = []
    for atom in request.atoms:
        target = route(replace(request, operation="extract", atoms=(atom,)), kind, open_parents=open_parents)
        if target.expression.get("instrument"):
            expression = {key: value for key, value in target.expression.items() if key != "instrument"}
            target = replace(target, expression=expression)
        targets.append(target)
    return targets


def nodes(expression: dict) -> list[dict]:
    """式に含まれる節（子が先）。"""
    found: list[dict] = []

    def walk(node: dict) -> None:
        for key in ("parent", "base", "minus"):
            if key in node:
                walk(node[key])
        for item in node.get("items", []):
            walk(item)
        if all(canonical(node) != canonical(existing) for existing in found):
            found.append(node)

    walk(expression)
    return found


def engines(expression: dict) -> set[str]:
    return {node["node"] for node in nodes(expression)} & {"dnr", "vocals", "lead", "voices", "stems", "model", "text"}
