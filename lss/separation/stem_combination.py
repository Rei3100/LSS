"""モデルステム経路のステムの組み合わせ（例: ボーカル＋ドラム）。

利用者は欲しい組み合わせ（意味的な部品の集合）を宣言し、経路はProductが選ぶ。組み合わせは、
同じ原本・同じ分離passの出力ステムを、サンプル位置を揃えたまま足すRecipeの節である。

重複の規則（二重に数えない）:
- 1つの分離passの出力は、その入力の分割（互いに素）として扱う。
- 部品の中身は内容の単位（``ATOMS``）の集合で表す。要求の和集合を、そのpassの出力のうち
  中身が和集合に含まれるものだけで過不足なく覆えるときに限って成立する。出力を部分的に
  またぐ要求（4分割modelでのピアノ等）は分けられないので作らない。
- 他の要求部品に中身が含まれる要求（伴奏＋ドラム。伴奏は既にドラムを含む）は、
  含む側だけで実現し、含まれた側を ``subsumed`` として記録する。
- 和集合がpassの全出力を覆う組み合わせは原本の写しなので作らない。
- 和集合がそのpassの1本の出力と一致するときは、残りの和（相補の残差の和）ではなく
  その直接の出力を使う（``route == "direct_stem"``）。これは単一ステムの既存経路そのものである。

和は宣言した利得（現行は全部品 1.0）を掛けた float64 の和を、部品の正準順（modelの出力順）で取り、
float32 へ丸めた値である。クリップはしない。公開時の2の冪headroomは既存の仕上げ（writer）が
公開単位で決める。

modelごとの出力の意味（``lss.separation.stem_semantics``）が宣言されているmodelでは、名前の推測の代わりに
宣言を使う（例: カラオケmodelの ``Vocals`` はリードだけ）。宣言の無いmodelの扱いは変わらない。

別々の分離passの組み合わせ（規則 ``ALGEBRA_RULE_ID``）: 同じ復号済み入力を各modelへ渡した出力は、
同じレート・長さ・ゼロ位置のとき標本単位で揃っている（``ALIGNMENT_RULE_ID``）。宣言済みの意味のもとで
符号付き単位利得（+1/−1）の和が要求の中身と過不足なく一致するときだけ作る（例: 合唱 = 歌全体 − リード）。
宣言の無いmodel・揃いを証明できない部品は使わない。直接の出力 → 1passの和 → pass数の少ない代数の順に選ぶ。
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf


RULE_ID = "partition-exact-cover-unity-sum-v1"
ALGEBRA_RULE_ID = "declared-semantics-signed-unity-algebra-v1"
ALIGNMENT_RULE_ID = "same-decoded-input-zero-offset-equal-layout-v1"
CROSS_PASS_ROUTE = "cross_pass_algebra"
# 1つの組み合わせに使う分離passの上限。passが増えるほど各modelの推定誤差が重なる。
MAX_PASSES = 3
TARGET_SEPARATOR = "+"
COMBINED_SUBTYPE = "FLOAT"
BLOCK_FRAMES = 1 << 18

LEAD_VOCALS = "lead_vocals"
BACKING_VOCALS = "backing_vocals"
DRUMS = "drums"
BASS = "bass"
GUITAR = "guitar"
PIANO = "piano"
OTHER = "other_instruments"
# 名前で中身が分かる内容の単位。足りない名前（modelの独自の出力）は不透明な単位として扱う。
ATOMS = (LEAD_VOCALS, BACKING_VOCALS, DRUMS, BASS, GUITAR, PIANO, OTHER)
_VOCALS = frozenset({LEAD_VOCALS, BACKING_VOCALS})

# 出力名・要求名のどちらにも使う、中身が固定の名前。
_NAMED = {
    "vocals": _VOCALS, "vocal": _VOCALS,
    "lead_vocals": frozenset({LEAD_VOCALS}), "lead vocals": frozenset({LEAD_VOCALS}),
    "lead": frozenset({LEAD_VOCALS}),
    "backing_vocals": frozenset({BACKING_VOCALS}), "backing vocals": frozenset({BACKING_VOCALS}),
    "backing": frozenset({BACKING_VOCALS}), "chorus": frozenset({BACKING_VOCALS}),
    "drums": frozenset({DRUMS}), "drum": frozenset({DRUMS}),
    "bass": frozenset({BASS}),
    "guitar": frozenset({GUITAR}),
    "piano": frozenset({PIANO}),
}
# 出力名としては「そのpassの残り全部」を意味する名前。
_COMPLEMENT_OUTPUTS = frozenset({"instrumental", "inst", "accompaniment", "no_vocals", "other"})
# 要求名としての伴奏は、どのpassでも「歌以外の全部」。
_REQUEST_INSTRUMENTAL = frozenset({"instrumental", "inst", "accompaniment"})


class StemCombinationError(ValueError):
    """組み合わせをこのpassでは作れない（Recipeの規則で拒否した）。"""

    code = "stem_combination_unsupported"


class StemCombinationIncompleteError(RuntimeError):
    """部品ステムが欠けている・揃っていない。部分的な組み合わせは作らない（UNKNOWN）。"""

    code = "stem_combination_incomplete"


@dataclass(frozen=True)
class CombinationPlan:
    requested: tuple[str, ...]
    components: tuple[str, ...]
    content: tuple[str, ...]
    subsumed: tuple[str, ...] = ()
    gains: tuple[float, ...] = field(default=())

    @property
    def route(self) -> str:
        return "direct_stem" if len(self.components) == 1 else "sum_of_stems"

    @property
    def label(self) -> str:
        """公開する成果物のパート名（ファイル名と ``LSS_STEM``）。実際に足した部品から決める。"""
        return " + ".join(self.components)

    def recipe_stage(self) -> dict:
        """Recipeの ``stem_sum`` 節。音を決める選択だけを入れる（要求の言い回しは入れない）。"""
        return {
            "stage": "stem_sum",
            "rule": RULE_ID,
            "components": list(self.components),
            "gains": [float(gain) for gain in self.gains],
            "content": list(self.content),
            "label": self.label,
        }

    def to_dict(self) -> dict:
        return {
            "rule": RULE_ID,
            "requested": list(self.requested),
            "components": list(self.components),
            "gains": [float(gain) for gain in self.gains],
            "content": list(self.content),
            "subsumed": list(self.subsumed),
            "route": self.route,
            "label": self.label,
        }


def normalized_request(components) -> tuple[str, ...]:
    """要求部品の正規形（小文字・前後空白なし・重複なし・名前順）。2つ以上が必要。"""
    if isinstance(components, str) or not isinstance(components, (list, tuple)):
        raise StemCombinationError(f"組み合わせの部品が不正です: {components!r}")
    names = []
    for value in components:
        if not isinstance(value, str) or not value.strip() or TARGET_SEPARATOR in value:
            raise StemCombinationError(f"組み合わせの部品名が不正です: {value!r}")
        names.append(" ".join(value.split()).casefold())
    unique = sorted(set(names))
    if len(unique) < 2:
        raise StemCombinationError("組み合わせには異なる部品が2つ以上必要です")
    return tuple(unique)


def target_key(components) -> str:
    """キューと分離ジョブの ``stem_target`` へ載せる組み合わせの表記（例: ``drums+vocals``）。"""
    return TARGET_SEPARATOR.join(normalized_request(components))


def parse_target(stem_target: str) -> tuple[str, ...] | None:
    """``stem_target`` が組み合わせならその部品を返す。単一ステムならNone。"""
    if not isinstance(stem_target, str) or TARGET_SEPARATOR not in stem_target:
        return None
    return normalized_request(stem_target.split(TARGET_SEPARATOR))


def declared_content(expected_stems, semantics: Mapping[str, frozenset[str]]) -> dict[str, frozenset[str]]:
    """宣言済みの意味をmodelの実際の出力名へ当てる。出力の集合が宣言と違えば意味は証明されていない。"""
    stems = list(expected_stems)
    if not stems or len({name.casefold() for name in stems}) != len(stems):
        raise StemCombinationError(f"分離passの出力が空、または重複しています: {stems}")
    by_key = {name.casefold(): atoms for name, atoms in semantics.items()}
    if set(by_key) != {name.casefold() for name in stems}:
        raise StemCombinationError(
            f"modelの出力{stems}が宣言した出力の意味{list(semantics)}と一致しません（意味を証明できません）"
        )
    return {name: frozenset(by_key[name.casefold()]) for name in stems}


def pass_content(expected_stems, semantics: Mapping[str, frozenset[str]] | None = None) -> dict[str, frozenset[str]]:
    """1つの分離passの出力ごとの中身。出力は互いに素（入力の分割）として扱う。"""
    if semantics is not None:
        return declared_content(expected_stems, semantics)
    stems = list(expected_stems)
    if not stems or len({name.casefold() for name in stems}) != len(stems):
        raise StemCombinationError(f"分離passの出力が空、または重複しています: {stems}")
    content: dict[str, frozenset[str]] = {}
    complements = []
    opaque = False
    for name in stems:
        key = name.casefold()
        if key in _NAMED:
            content[name] = _NAMED[key]
        elif key in _COMPLEMENT_OUTPUTS:
            complements.append(name)
        else:
            opaque = True
            content[name] = frozenset({f"stem:{key}"})
    named = [value for value in content.values() if not next(iter(value)).startswith("stem:")]
    covered = frozenset().union(*named) if named else frozenset()
    if sum(len(value) for value in named) != len(covered):
        raise StemCombinationError(f"分離passの出力名の中身が重なっています: {stems}")
    remainder = frozenset(ATOMS) - covered
    if len(complements) == 1 and not opaque and remainder:
        content[complements[0]] = remainder
    else:
        for name in complements:
            content[name] = frozenset({f"stem:{name.casefold()}"})
    return {name: content[name] for name in stems}


def _requested_content(name: str, outputs: dict[str, frozenset[str]], declared: bool = False) -> frozenset[str]:
    # 宣言済みのmodelでは、要求の言葉はその意味で読む（カラオケの出力名 ``Vocals`` は歌全体ではない）。
    if not declared or (name not in _NAMED and name not in _REQUEST_INSTRUMENTAL):
        for output, atoms in outputs.items():
            if output.casefold() == name:
                return atoms
    if name in _NAMED:
        return _NAMED[name]
    if name in _REQUEST_INSTRUMENTAL:
        return frozenset(ATOMS) - _VOCALS
    raise StemCombinationError(f"このpassでは分からない部品です: {name}")


def _subsumed(requested, wanted: dict[str, frozenset[str]]) -> list[str]:
    subsumed = []
    for name in requested:
        others = [atoms for other, atoms in wanted.items() if other != name and other not in subsumed]
        if others and wanted[name] <= frozenset().union(*others):
            subsumed.append(name)
    return subsumed


def plan_combination(components, expected_stems, semantics: Mapping[str, frozenset[str]] | None = None
                     ) -> CombinationPlan:
    """要求を1つの分離passの出力で実現する計画。規則に合わなければ拒否する。

    ``semantics`` はそのmodelの宣言済みの出力の意味（``stem_semantics``）。Noneなら名前で読む。
    """
    requested = normalized_request(components)
    outputs = pass_content(expected_stems, semantics)
    wanted = {name: _requested_content(name, outputs, semantics is not None) for name in requested}
    union = frozenset().union(*wanted.values())
    subsumed = _subsumed(requested, wanted)
    whole = frozenset().union(*outputs.values())
    if union >= whole:
        raise StemCombinationError("組み合わせが原本の全部になります（原本の写しは作りません）")
    selected = []
    for name, atoms in outputs.items():
        if atoms <= union:
            selected.append(name)
        elif atoms & union:
            raise StemCombinationError(
                f"出力{name}は組み合わせの一部だけを含むため、このpassでは分けられません"
            )
    realized = frozenset().union(*(outputs[name] for name in selected)) if selected else frozenset()
    if realized != union:
        missing = ", ".join(sorted(union - realized))
        raise StemCombinationError(f"このpassの出力では作れない部品があります: {missing}")
    return CombinationPlan(
        requested=requested,
        components=tuple(selected),
        content=tuple(sorted(union)),
        subsumed=tuple(subsumed),
        gains=tuple(1.0 for _ in selected),
    )


def _component_file(store_dir: Path, name: str) -> Path:
    matches = [path for path in Path(store_dir).iterdir()
               if path.is_file() and path.stem == name and path.suffix.lower() in {".wav", ".flac"}]
    if len(matches) != 1:
        raise StemCombinationIncompleteError(f"組み合わせの部品{name}の分離出力が一意に揃っていません")
    return matches[0]


def combine_files(store_dir: Path, plan: CombinationPlan, destination: Path) -> dict:
    """部品ステムを宣言した利得で足し、``destination`` へfloat32 WAVで書く。

    部品の長さ・チャンネル数・サンプルレートが1つでも違えば、揃え直さずに拒否する。
    部品の欠落・非有限値でも何も残さない（部分的な組み合わせを公開へ渡さない）。
    """
    if plan.route != "sum_of_stems" or len(plan.gains) != len(plan.components):
        raise StemCombinationError("足し合わせる部品が2つ以上ありません")
    files = [_component_file(store_dir, name) for name in plan.components]
    frames, channels, samplerate, peak = _sum_files(list(plan.components), files, list(plan.gains), destination)
    return {
        **plan.to_dict(),
        "frames": frames, "channels": channels, "samplerate": samplerate,
        "sum_peak": peak,
        "component_sha256": {name: _file_sha256(path) for name, path in zip(plan.components, files)},
    }


def _sum_files(names: list[str], files: list[Path], gains: list[float], destination: Path
               ) -> tuple[int, int, int, float]:
    layouts = []
    for path in files:
        info = sf.info(path)
        layouts.append((info.frames, info.channels, info.samplerate))
    if len(set(layouts)) != 1 or layouts[0][0] <= 0:
        raise StemCombinationIncompleteError(
            f"組み合わせの部品の長さ・チャンネル・サンプルレートが揃っていません: "
            f"{dict(zip(names, layouts))}"
        )
    frames, channels, samplerate = layouts[0]
    destination = Path(destination)
    peak = 0.0
    try:
        with sf.SoundFile(destination, mode="x", samplerate=samplerate, channels=channels,
                          subtype=COMBINED_SUBTYPE, format="WAV") as out:
            sources = [sf.SoundFile(path) for path in files]
            try:
                remaining = frames
                while remaining:
                    count = min(BLOCK_FRAMES, remaining)
                    total = np.zeros((count, channels), dtype=np.float64)
                    for name, source, gain in zip(names, sources, gains):
                        block = source.read(count, dtype="float64", always_2d=True)
                        if block.shape != (count, channels) or not np.isfinite(block).all():
                            raise StemCombinationIncompleteError(f"組み合わせの部品{name}を読み切れません")
                        total += block * gain if gain != 1.0 else block
                    combined = total.astype(np.float32)
                    if not np.isfinite(combined).all():
                        raise StemCombinationIncompleteError("組み合わせの和が有限ではありません")
                    peak = max(peak, float(np.max(np.abs(combined))) if combined.size else 0.0)
                    out.write(combined)
                    remaining -= count
            finally:
                for source in sources:
                    source.close()
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return frames, channels, samplerate, peak


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# --- 別々の分離passの組み合わせ ---------------------------------------------

_LABEL_NAMES = {
    LEAD_VOCALS: "lead vocals", BACKING_VOCALS: "backing vocals", DRUMS: "drums", BASS: "bass",
    GUITAR: "guitar", PIANO: "piano", OTHER: "other",
}


def content_label(content) -> str:
    """中身から決まる成果物のパート名（要求の言い回しに依らない）。例: ``backing vocals + drums``。"""
    atoms = set(content)
    if not atoms or not atoms <= set(ATOMS):
        raise StemCombinationError(f"成果物の中身が不正です: {sorted(atoms)}")
    parts = []
    instrumental = frozenset(ATOMS) - _VOCALS
    if instrumental <= atoms:
        parts.append("instrumental")
        atoms -= instrumental
    if _VOCALS <= atoms:
        parts.append("vocals")
        atoms -= _VOCALS
    parts.extend(_LABEL_NAMES[atom] for atom in ATOMS if atom in atoms)
    return " + ".join(parts)


@dataclass(frozen=True)
class PassCandidate:
    """組み合わせに使える分離pass（1つのmodelを、復号済みの同じ入力へ1回）。"""

    model_id: str
    expected_stems: tuple[str, ...]
    # 宣言済みの出力の意味（``stem_semantics``）。Noneは未宣言（名前で読む・別passとは組まない）。
    semantics: Mapping[str, frozenset[str]] | None = None


@dataclass(frozen=True)
class AlgebraTerm:
    model_id: str
    stem: str
    gain: float
    content: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.model_id}/{self.stem}"


@dataclass(frozen=True)
class AlgebraPlan:
    """別々の分離passの出力を、符号付き単位利得で足す計画（``ALGEBRA_RULE_ID``）。"""

    requested: tuple[str, ...]
    passes: tuple[str, ...]
    terms: tuple[AlgebraTerm, ...]
    content: tuple[str, ...]
    subsumed: tuple[str, ...] = ()

    @property
    def route(self) -> str:
        return CROSS_PASS_ROUTE

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(term.key for term in self.terms)

    @property
    def gains(self) -> tuple[float, ...]:
        return tuple(term.gain for term in self.terms)

    @property
    def label(self) -> str:
        return content_label(self.content)

    def recipe_stage(self) -> dict:
        """Recipeの ``aligned_stem_algebra`` 節。passは番号で指し、各passの中身はprovider節が持つ。"""
        return {
            "stage": "aligned_stem_algebra",
            "rule": ALGEBRA_RULE_ID,
            "alignment": ALIGNMENT_RULE_ID,
            "terms": [{"pass": self.passes.index(term.model_id), "stem": term.stem,
                       "gain": float(term.gain), "content": list(term.content)} for term in self.terms],
            "content": list(self.content),
            "label": self.label,
        }

    def to_dict(self) -> dict:
        return {
            "rule": ALGEBRA_RULE_ID,
            "alignment": ALIGNMENT_RULE_ID,
            "requested": list(self.requested),
            "passes": list(self.passes),
            "terms": [{"model_id": term.model_id, "stem": term.stem, "gain": float(term.gain),
                       "content": list(term.content)} for term in self.terms],
            "components": list(self.components),
            "gains": [float(gain) for gain in self.gains],
            "content": list(self.content),
            "subsumed": list(self.subsumed),
            "route": self.route,
            "label": self.label,
        }


def _request_meaning(name: str) -> frozenset[str]:
    if name in _NAMED:
        return _NAMED[name]
    if name in _REQUEST_INSTRUMENTAL:
        return frozenset(ATOMS) - _VOCALS
    raise StemCombinationError(f"宣言済みの意味では分からない部品です: {name}")


def _vector(atoms) -> tuple[int, ...]:
    return tuple(1 if atom in atoms else 0 for atom in ATOMS)


def _contributions(outputs: dict[str, frozenset[str]]) -> list[tuple[tuple[int, ...], tuple[int, ...], float]]:
    """1つのpassが寄与できる量: 出力の空でない部分集合の和を、+1か−1の一方の符号で。"""
    names = list(outputs)
    found = []
    for size in range(1, len(names) + 1):
        for chosen in combinations(range(len(names)), size):
            atoms = frozenset().union(*(outputs[names[index]] for index in chosen))
            vector = _vector(atoms)
            for gain in (1.0, -1.0):
                found.append((tuple(int(gain) * value for value in vector), chosen, gain))
    return found


def _subtract(left, right) -> tuple[int, ...]:
    return tuple(a - b for a, b in zip(left, right))


def plan_cross_pass(components, candidates, *, first_model: str | None = None,
                    max_passes: int = MAX_PASSES) -> AlgebraPlan:
    """要求を2つ以上の分離passの出力の符号付き和で実現する計画。作れなければ拒否する。

    使うのは意味を宣言済みで、宣言がmodelの実際の出力と一致するpassだけ。各passは出力の和を
    +1か−1のどちらか一方の符号で寄与する。和の中身が要求と単位ごとに過不足なく一致する組から、
    pass数・項の数・負の項の数・候補順の小さいものを選ぶ。``first_model`` は最初のpassを固定する
    （分離ジョブが自分のmodelから同じ計画を選び直すため）。
    """
    requested = normalized_request(components)
    wanted = {name: _request_meaning(name) for name in requested}
    union = frozenset().union(*wanted.values())
    if union >= frozenset(ATOMS):
        raise StemCombinationError("組み合わせが原本の全部になります（原本の写しは作りません）")
    subsumed = _subsumed(requested, wanted)
    target = _vector(union)
    declared = []
    for index, candidate in enumerate(candidates):
        if candidate.semantics is None:
            continue
        try:
            outputs = declared_content(candidate.expected_stems, candidate.semantics)
        except StemCombinationError:
            continue
        declared.append((index, candidate, outputs, _contributions(outputs)))
    for size in range(2, max(2, max_passes) + 1):
        best = None
        for group in combinations(declared, size):
            if first_model is not None and group[0][1].model_id != first_model:
                continue
            last = {}
            for vector, chosen, gain in group[-1][3]:
                last.setdefault(vector, []).append((chosen, gain))
            partial = [((0,) * len(ATOMS), ())]
            for _, _, _, table in group[:-1]:
                partial = [(tuple(a + b for a, b in zip(total, vector)), picks + ((chosen, gain),))
                           for total, picks in partial for vector, chosen, gain in table]
            for total, picks in partial:
                for chosen, gain in last.get(_subtract(target, total), ()):
                    selection = picks + ((chosen, gain),)
                    terms = sum(len(item[0]) for item in selection)
                    negatives = sum(len(item[0]) for item in selection if item[1] < 0)
                    key = (terms, negatives, tuple(item[0] for item in group), selection)
                    if best is None or key < best[0]:
                        best = (key, group, selection)
        if best is not None:
            _, group, selection = best
            terms = []
            for (_, candidate, outputs, _), (chosen, gain) in zip(group, selection):
                names = list(outputs)
                for position in chosen:
                    stem = names[position]
                    terms.append(AlgebraTerm(model_id=candidate.model_id, stem=stem, gain=gain,
                                             content=tuple(atom for atom in ATOMS if atom in outputs[stem])))
            return AlgebraPlan(
                requested=requested,
                passes=tuple(candidate.model_id for _, candidate, _, _ in group),
                terms=tuple(terms),
                content=tuple(sorted(union)),
                subsumed=tuple(subsumed),
            )
    raise StemCombinationError(
        "意味を宣言した分離passの組み合わせでは、要求の中身を過不足なく作れません"
    )


def choose_route(components, candidates) -> tuple[str, CombinationPlan | AlgebraPlan]:
    """要求に使うmodelと経路。直接の出力 → 1passの和（部品の少ない順・候補順） → 別passの代数。"""
    singles = []
    reasons = []
    for index, candidate in enumerate(candidates):
        try:
            plan = plan_combination(components, candidate.expected_stems, candidate.semantics)
        except StemCombinationError as exc:
            reasons.append(f"{candidate.model_id}: {exc}")
            continue
        singles.append((len(plan.components), index, candidate.model_id, plan))
    if singles:
        _, _, model_id, plan = min(singles, key=lambda item: item[:2])
        return model_id, plan
    try:
        plan = plan_cross_pass(components, candidates)
    except StemCombinationError as exc:
        reasons.append(f"別passの組み合わせ: {exc}")
        raise StemCombinationError(" / ".join(reasons)) from exc
    return plan.passes[0], plan


def combine_terms(store_dirs: Mapping[str, Path], plan: AlgebraPlan, destination: Path) -> dict:
    """別々のpassの部品を符号付き単位利得で足し、``destination`` へfloat32 WAVで書く。

    揃い（``ALIGNMENT_RULE_ID``）の最後の確かめとして、部品の長さ・チャンネル・レートが1つでも違えば
    揃え直さずに拒否し、何も残さない。
    """
    if not isinstance(plan, AlgebraPlan) or len(set(plan.passes)) < 2:
        raise StemCombinationError("別passの組み合わせの計画ではありません")
    missing = [model_id for model_id in plan.passes if model_id not in store_dirs]
    if missing:
        raise StemCombinationIncompleteError(f"組み合わせの分離passが実行されていません: {missing}")
    files = [_component_file(store_dirs[term.model_id], term.stem) for term in plan.terms]
    frames, channels, samplerate, peak = _sum_files(
        list(plan.components), files, list(plan.gains), destination)
    return {
        **plan.to_dict(),
        "frames": frames, "channels": channels, "samplerate": samplerate,
        "sum_peak": peak,
        "component_sha256": {key: _file_sha256(path) for key, path in zip(plan.components, files)},
    }
