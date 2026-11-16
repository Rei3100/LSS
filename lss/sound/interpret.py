"""語彙に無い日本語の「欲しい音」を、訳して、元の日本語で裏付けられたときだけ英語の音として読む（worker側、GPU）。

1. `lss.sound.translate` で日本語の句を英語へ訳す（beam searchの上位候補）。
2. 各候補を英語の名詞句へ整え（"The sound of rain hitting the window." → "rain hitting the window"）、
   英語の句と元の日本語の両方に出てくる語彙の音（`lss.sound.intent.english_mentions` /
   ``japanese_mentions``）を比べる（**裏付け**）。
   - 日本語に語彙の音が出てくる句（「雨が窓に当たる音」の「雨」）: 訳に出てくる語彙の音の集まりが、日本語の
     それと一致する候補だけを使う。訳が日本語に無い音を持ち込んだ候補（「猫が喉を鳴らす音」→ "cat barking"
     の barking＝犬）は捨てる。訳が語彙の語そのもの（「ヘリコプターのローター音」→ "helicopter"）の候補を先に使う。
   - 日本語に語彙の音が1つも無い句: 訳だけでは裏付けが無い。2つの多言語の文の埋め込み（`lss.sound.xling` の
     LaBSE と multilingual MiniLM）がどちらも高い候補だけを使う。訳の誤り（「サイレン」→ "Silence"、
     「歓声」→ "Let's go"）は、どちらかの埋め込みが低いことで捨てる。
3. 使う候補は、語彙の語ならその音、語彙の音を1つだけ含む句ならその音と同じ経路・確かめ方の句、それ以外は
   言葉で分離する音（``open:``）として読む。成果物の名前は利用者の日本語のまま。

どの候補も裏付けられなければ ``None``（その言葉は読めなかったものとして扱い、推測で別の音を作らない）。
実測（開発時の Evidence）: 訳だけをLaBSEの閾値0.80で確かめる旧方式は、正しい訳77件のうち3件しか
通さなかった（翻訳器そのものは上流と一致、判定器の値も一致）。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from lss.sound import lexicon

JUDGES = ("labse", "mminilm")
# 日本語に語彙の音が無い句だけに使う。両方の埋め込みの類似度が届いた候補だけを使う。
UNGROUNDED_MIN_SIMILARITY = {"lexicon": {"labse": 0.60, "mminilm": 0.80},   # 訳が語彙の語そのもの
                             "open": {"labse": 0.70, "mminilm": 0.90}}      # 訳が語彙に無い句
CANDIDATES = 6
MAX_WORDS = 8
_LEADING = re.compile(r"^(?:(?:it|that|this)(?:'s| is| was)\s+(?:like\s+|called\s+)?|i (?:heard|hear)\s+|"
                      r"sounds?\s+like\s+|(?:the\s+|a\s+|an\s+)?sounds?\s+of\s+|the\s+|a\s+|an\s+)+")
_TRAILING = re.compile(r"(?:\s+(?:sounds?|noises?|sound effects?))+$")


def english_phrase(text: str) -> str | None:
    """訳文を、言葉で分離するmodelへ渡す英語の名詞句へ整える。整えられなければNone。"""
    value = text.strip().lower().replace("’", "'")
    value = re.sub(r"[\"“”!?.;:()\[\]]", " ", value)
    value = " ".join(value.split())
    previous = None
    while previous != value:
        previous = value
        value = _LEADING.sub("", value).strip()
        value = _TRAILING.sub("", value).strip()
        value = re.sub(r"'s$", "", value).strip(" ,")
    if not value or not re.fullmatch(r"[a-z0-9 '\-,&]+", value) or len(value.split()) > MAX_WORDS:
        return None
    return value


def _source(name: str) -> Path:
    from lss.paths import resource_root

    module_source = Path(__file__).with_name(name)
    if not module_source.is_file():
        module_source = resource_root() / "runtime-sources" / "lss" / "sound" / name
    return module_source


def revision() -> dict:
    """読み方の版（訳のmodel・判定のmodel・閾値・実装）。同じ版なら同じ日本語は同じ音として読む。"""
    from lss.core import runtime
    from lss.sound import translate, xling

    body = {"schema_version": 2,
            "translator": {"revision": translate.REVISION, "weights_sha256": translate.FILES["model"]["sha256"],
                           "search": translate.SEARCH},
            "judges": [{"model": name, "revision": xling.MODELS[name]["revision"],
                        "weights_sha256": xling.MODELS[name]["files"]["model"][2]} for name in JUDGES],
            "ungrounded_min_similarity": UNGROUNDED_MIN_SIMILARITY, "candidates": CANDIDATES,
            "implementation": [{"path": name, "sha256": runtime.text_source_sha256(_source(name))}
                               for name in ("translate.py", "xling.py", "interpret.py", "intent.py", "lexicon.py")]}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {**body, "interpreter_id": "sound-interpret-v2-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]}


def grounding(japanese: str, phrase: str) -> tuple[str, list[str], list[str]]:
    """訳の候補の裏付け。("grounded" / "contradicted" / "ungrounded", 日本語の音, 英語の音)。"""
    from lss.sound import intent

    said = sorted({concept.base_id for concept in intent.japanese_mentions(japanese)})
    heard = sorted({concept.base_id for concept in intent.english_mentions(phrase)})
    if not said:
        return "ungrounded", said, heard
    if heard and set(heard) <= set(said):
        return "grounded", said, heard     # 訳は日本語に無い音を持ち込んでいない（細部を落とすのは可）
    return ("contradicted" if heard else "unsupported"), said, heard


def choose(japanese: str, candidates: list[dict]) -> dict | None:
    """訳の候補（``english`` と、日本語に語彙の音が無い句なら ``similarity`` {判定model: 値}）から1つを選ぶ。

    裏付けた候補のうち、訳が語彙の語そのものの候補、日本語の音を多く含む候補、順位の高い候補の順。
    裏付けた候補が無くても、上位の候補がそろって日本語の同じ1つの音を含む（「猫がミャーと鳴く声」→
    "cat moaning" / "cat crying" …、細部の訳だけが割れる）ときは、その音として読む（``agreed``）。
    日本語に語彙の音が無い句は、両方の埋め込みが閾値に届いた候補だけ。どれも使えなければNone。"""
    from lss.sound import intent

    usable, shared = [], None
    for rank, candidate in enumerate(candidates):
        phrase = english_phrase(candidate["english"])
        if phrase is None:
            continue
        verdict, said, heard = grounding(japanese, phrase)
        if said:
            common = set(said) & set(heard)
            shared = common if shared is None else shared & common
        concept = intent._resolve(phrase, open_vocabulary=True)
        if concept is None or verdict in ("contradicted", "unsupported"):
            continue
        exact = intent._lookup_sound(phrase) is not None
        if verdict == "ungrounded":
            similarity = candidate.get("similarity") or {}
            floor = UNGROUNDED_MIN_SIMILARITY["lexicon" if exact else "open"]
            if any(similarity.get(name, -1.0) < value for name, value in floor.items()):
                continue
        usable.append((0 if exact else 1, -len(heard), rank, phrase, concept, verdict))
    if usable:
        _, _, _, phrase, concept, verdict = min(usable, key=lambda item: item[:3])
    elif shared is not None and len(shared) == 1 and len(candidates) >= 2:
        concept = lexicon.BY_ID[next(iter(shared))]
        phrase, verdict = concept.prompt, "agreed"
    else:
        return None
    if concept.id.startswith("open:") or "~" in concept.id:
        concept = replace(concept, label=intent.label_safe(japanese))   # 成果物の名前は利用者の言葉
    return {"phrase": phrase, "concept": concept, "verdict": verdict}


class Interpreter:
    """``intent.parse(..., interpreter=...)`` に渡す読み手。modelは初めて要るときに読み、``close`` で手放す。

    ``cache`` を渡すと、同じ版で読んだ結果をそこへ残し、次からはGPUを使わずに同じ読み方を返す
    （成果物の再利用の確認でGPUへ触れない）。``before_gpu`` はmodelを読む直前に呼ぶ（GPUの排他取得）。
    埋め込み（判定）のmodelは、日本語に語彙の音が無い句を訳したときだけ読む。
    """

    def __init__(self, *, progress: Callable[[dict], None] | None = None, device: str = "cuda:0",
                 cache: Path | None = None, before_gpu: Callable[[], None] | None = None):
        self._progress = progress
        self._device = device
        self._translator = None
        self._judges: dict = {}
        self._cache_path = Path(cache) if cache is not None else None
        self._before_gpu = before_gpu or (lambda: None)
        self._revision = None
        self._cache: dict = {}
        self.records: list[dict] = []
        self.unavailable: str | None = None   # 訳のmodelを用意できなかった理由（そのときは訳さない）
        if self._cache_path is not None and self._cache_path.is_file():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._cache = {}

    @property
    def interpreter_id(self) -> str:
        if self._revision is None:
            self._revision = revision()["interpreter_id"]
        return self._revision

    def _remember(self, japanese: str, value) -> None:
        if self._cache_path is None:
            return
        self._cache.setdefault(self.interpreter_id, {})[japanese] = value
        temporary = self._cache_path.with_name(f".{self._cache_path.name}.tmp")
        temporary.write_text(json.dumps(self._cache, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(self._cache_path)

    def _load_translator(self) -> None:
        from lss.sound import translate

        self._before_gpu()
        if self._translator is None:
            self._translator = translate.Translator(translate.ensure_assets(self._progress), device=self._device)

    def _judge(self, name: str):
        from lss.sound import xling

        if name not in self._judges:
            self._before_gpu()
            self._judges[name] = xling.Judge(name, xling.ensure_assets(name, self._progress), device=self._device)
        return self._judges[name]

    def close(self) -> None:
        for engine in [self._translator, *self._judges.values()]:
            if engine is not None:
                engine.close()
        self._translator = None
        self._judges = {}

    def __call__(self, japanese: str):
        cached = self._cache.get(self.interpreter_id, {})
        if japanese in cached:
            value = cached[japanese]
            self.records.append({"text": japanese, "cached": True, "chosen": value})
            if value is None:
                return None
            fields = dict(value["concept"])
            fields["words"], fields["audioset"] = tuple(fields["words"]), tuple(fields["audioset"])
            return lexicon.Concept(**fields), value["phrase"]
        if self.unavailable is not None:
            return None
        try:
            self._load_translator()
        except Exception as exc:  # 訳が用意できなくても、語彙の言葉は処理を続ける（offlineの初回等）
            self.unavailable = f"{type(exc).__name__}: {exc}"[:300]
            self.close()
            return None
        record = {"text": japanese, "candidates": [], "chosen": None}
        self.records.append(record)
        candidates = [{"english": english, "log_probability": round(score, 4)}
                      for english, score in self._translator.translate(japanese, candidates=CANDIDATES)]
        if all(grounding(japanese, english_phrase(item["english"]) or "")[0] == "ungrounded" for item in candidates):
            try:
                for name in JUDGES:
                    judge = self._judge(name)
                    for item in candidates:
                        item.setdefault("similarity", {})[name] = round(judge.similarity(japanese, item["english"]), 4)
            except Exception as exc:
                self.unavailable = f"{type(exc).__name__}: {exc}"[:300]
                self.close()
                return None
        record["candidates"] = candidates
        chosen = choose(japanese, candidates)
        if chosen is None:
            self._remember(japanese, None)
            return None
        concept, phrase = chosen["concept"], chosen["phrase"]
        record["chosen"] = {"phrase": phrase, "concept": concept.id, "verdict": chosen["verdict"]}
        self._remember(japanese, {"concept": asdict(concept), "phrase": phrase, "verdict": chosen["verdict"]})
        return concept, phrase


def label_safe(text: str) -> str:
    """利用者の日本語を、成果物の名前に使える形へ（`lss.sound.intent.label_safe`）。"""
    from lss.sound import intent

    return intent.label_safe(text)
