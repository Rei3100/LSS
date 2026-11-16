"""モデル台帳・完了履歴・採点に共通のTSV I/O。保存失敗時は元ファイルを保全する。"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def load_tsv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return header, rows


def save_tsv(path: Path, header: list[str], rows: list[dict]) -> None:
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}-",
                                suffix=".tmp", encoding="utf-8", newline="\n", delete=False) as f:
            temporary_path = Path(f.name)
            writer = csv.DictWriter(
                f, fieldnames=header, delimiter="\t", quoting=csv.QUOTE_NONE,
                escapechar="\\", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
