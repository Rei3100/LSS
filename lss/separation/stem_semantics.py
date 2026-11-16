"""modelごとの出力パートの意味（内容の単位）の宣言表。

``ledger/model_stem_semantics.tsv`` は、台帳のmodelが出す各出力名が何を含むかを内容の単位
（``stem_combination.ATOMS``）で宣言する。例えばカラオケmodelの ``Vocals`` はリードボーカルだけで、
その ``Instrumental`` はバッキングボーカルと楽器の全部である。名前だけでは分からない意味を
データとして持ち、組み合わせの計画は宣言があるmodelでは名前の推測の代わりにこれを使う。

宣言の無いmodelは従来どおり（名前の推測、分からない名前は不透明）。宣言は読み込み時に検証する:
1行1出力、単位は既知のものだけ、同じmodelの出力どうしは重ならない（分離passの出力は互いに素）。
宣言がmodelの実際の出力（設定の出力名）と一致しなければ、そのmodelの意味は証明されていないので
組み合わせを作らない（推測へ戻らない）。採否（``状態``）は台帳 ``models.tsv`` が決め、この表は決めない。
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from lss.harness.tsv_io import load_tsv
from lss.paths import resource_root


SEMANTICS_TSV = resource_root() / "ledger" / "model_stem_semantics.tsv"
COLUMNS = ("model_id", "output", "content", "basis", "note")
CONTENT_SEPARATOR = ","


class InvalidStemSemanticsError(ValueError):
    """出力パートの意味の宣言表が壊れている。"""


def parse_content(value: str) -> frozenset[str]:
    from lss.separation.stem_combination import ATOMS

    atoms = [part.strip() for part in value.split(CONTENT_SEPARATOR)] if isinstance(value, str) else []
    if not atoms or any(not atom for atom in atoms):
        raise InvalidStemSemanticsError(f"出力の中身が空です: {value!r}")
    unknown = [atom for atom in atoms if atom not in ATOMS]
    if unknown:
        raise InvalidStemSemanticsError(f"出力の中身に未知の単位があります: {unknown}")
    if len(set(atoms)) != len(atoms):
        raise InvalidStemSemanticsError(f"出力の中身の単位が重複しています: {value!r}")
    return frozenset(atoms)


def load_semantics(path: Path | None = None) -> dict[str, MappingProxyType]:
    """宣言表を読み、modelごとの {出力名: 中身} を返す（出力は表の順）。

    表が無いことは宣言が無いことではなく配布の破損なので、読めなければ例外にする。
    """
    path = Path(SEMANTICS_TSV if path is None else path)
    if not path.is_file():
        raise InvalidStemSemanticsError(f"出力パートの意味の宣言表がありません: {path}")
    header, rows = load_tsv(path)
    if tuple(header) != COLUMNS:
        raise InvalidStemSemanticsError(f"出力パートの意味の宣言表の列が不正です: {header}")
    declared: dict[str, dict[str, frozenset[str]]] = {}
    for index, row in enumerate(rows, start=2):
        model_id = (row.get("model_id") or "").strip()
        output = (row.get("output") or "").strip()
        if not model_id or not output:
            raise InvalidStemSemanticsError(f"{index}行目にmodel_idまたはoutputがありません: {path}")
        if not (row.get("basis") or "").strip():
            raise InvalidStemSemanticsError(f"{index}行目に宣言の根拠（basis）がありません: {model_id}/{output}")
        try:
            content = parse_content(row.get("content") or "")
        except InvalidStemSemanticsError as exc:
            raise InvalidStemSemanticsError(f"{index}行目 {model_id}/{output}: {exc}") from exc
        outputs = declared.setdefault(model_id, {})
        if any(name.casefold() == output.casefold() for name in outputs):
            raise InvalidStemSemanticsError(f"同じmodelの出力が重複しています: {model_id}/{output}")
        overlap = [name for name, atoms in outputs.items() if atoms & content]
        if overlap:
            raise InvalidStemSemanticsError(
                f"同じmodelの出力の中身が重なっています: {model_id}/{output} と {', '.join(overlap)}"
            )
        outputs[output] = content
    return {model_id: MappingProxyType(dict(outputs)) for model_id, outputs in declared.items()}


def declared_for(model_id: str, semantics: dict | None = None):
    """modelの宣言（無ければNone）。``semantics`` を省くと同梱の表を読む。"""
    table = load_semantics() if semantics is None else semantics
    return table.get(model_id)
