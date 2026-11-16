"""処理済みの入力の重複を、確かな同一性だけで整理するための報告。読み取りだけを行い、何も削除しない。

- `same_file`: 内容SHA-256が同じ入力が、別の場所から処理されている（完全に同じファイル）。
- `same_recording`: 別のファイル（形式・マスタリング・切り出しが違う）だが、波形の検証まで通って
  同じ録音の同じ部分を含む。範囲と方向（どちらがどちらのどこか）を示し、版を決めつけて統合しない。
- 同一性が確かめられないものは報告しない（不明な版をまとめない）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lss.core import session
from lss.harness.tsv_io import load_tsv


@dataclass(frozen=True)
class InputRecord:
    input_sha256: str
    source_path: str
    recipe_id: str
    outputs: list[str]


def _completed_inputs(state_root: Path | None) -> list[InputRecord]:
    try:
        _, rows = load_tsv(session.completion_ledger_path(state_root))
    except (OSError, ValueError):
        return []
    records = []
    for row in rows:
        if not row.get("input_sha256") or row.get("outcome") not in {"artifacts", "zero"}:
            continue
        try:
            outputs = json.loads(row.get("outputs") or "[]")
        except ValueError:
            outputs = []
        records.append(InputRecord(row["input_sha256"], row.get("source_path") or "", row.get("recipe_id") or "",
                                   [str(item.get("path")) for item in outputs if isinstance(item, dict)]))
    return records


def same_file_groups(state_root: Path | None = None) -> list[dict]:
    """同じ内容の入力が複数の場所から処理された組。"""
    by_sha: dict[str, list[InputRecord]] = {}
    for record in _completed_inputs(state_root):
        by_sha.setdefault(record.input_sha256, []).append(record)
    groups = []
    for sha256, records in by_sha.items():
        locations = sorted({record.source_path for record in records if record.source_path})
        if len(locations) < 2:
            continue
        groups.append({
            "kind": "same_file", "input_sha256": sha256, "source_paths": locations,
            "outputs": sorted({output for record in records for output in record.outputs}),
            "present": [path for path in locations if Path(path).is_file()],
        })
    groups.sort(key=lambda group: group["source_paths"])
    return groups


def same_recording_groups(paths: list[Path], *, cache_dir: Path | None = None, loader=None,
                          min_covered_fraction: float = 0.5) -> list[dict]:
    """別ファイルどうしで、同じ録音の同じ部分が波形の検証まで一致する組。

    各ファイルを参照として索引にし、他のファイルを照合する。両方の長さの `min_covered_fraction` 以上が
    一致したものだけを返す（短い引用や別の長さの編集を同じ曲の重複と扱わない）。
    """
    from lss.bgm import identity

    loader = loader or identity.decode_mono
    files = sorted({Path(path).resolve() for path in paths})
    if len(files) < 2:
        return []
    index = identity.ReferenceIndex.build(files, loader=loader, cache_dir=cache_dir)
    lengths = {reference.sha256: reference.frames * identity.FRAME_SECONDS for reference in index.references}
    by_path = {reference.path: reference for reference in index.references}
    # 静かな部分で一致が途切れても、同じずれの一致をまとめて覆う割合を数える。
    spans: dict[tuple, list] = {}
    for path in files:
        own = by_path[str(path)]
        for m in identity.match(loader(path), index):
            if m.identity_state != "known_exact" or m.reference_sha256 == own.sha256:
                continue
            key = (own.sha256, str(path), m.reference_sha256, m.reference_path, round(m.offset_seconds, 1))
            spans.setdefault(key, []).append(m)
    groups: dict[frozenset, dict] = {}
    for (own_sha, own_path, ref_sha, ref_path, _), items in spans.items():
        intervals = sorted((m.query_start_seconds, m.query_end_seconds) for m in items)
        union, cursor_start, cursor_end = 0.0, None, None
        for a, b in intervals:
            if cursor_end is None or a > cursor_end:
                if cursor_end is not None:
                    union += cursor_end - cursor_start
                cursor_start, cursor_end = a, b
            else:
                cursor_end = max(cursor_end, b)
        if cursor_end is not None:
            union += cursor_end - cursor_start
        # 両方のファイルの大半を覆うときだけ重複とする（TVサイズとフルサイズ、引用は別の編集として残す）。
        covered = min(union / max(lengths[own_sha], 1e-9), union / max(lengths[ref_sha], 1e-9))
        if covered < min_covered_fraction:
            continue
        entry = groups.setdefault(frozenset({own_sha, ref_sha}), {"files": {}, "evidence": []})
        entry["files"][own_sha] = own_path
        entry["files"][ref_sha] = ref_path
        entry["evidence"].append({
            "query": own_path, "reference": ref_path, "offset_seconds": items[0].offset_seconds,
            "spans": [[m.query_start_seconds, m.query_end_seconds] for m in sorted(items, key=lambda m: m.query_start_seconds)],
            "covered_fraction": round(covered, 3),
            "waveform_explained": [m.waveform_explained for m in items],
        })
    result = []
    for entry in groups.values():
        result.append({"kind": "same_recording",
                       "files": [{"sha256": sha, "path": path} for sha, path in sorted(entry["files"].items())],
                       "evidence": entry["evidence"]})
    result.sort(key=lambda group: [item["path"] for item in group["files"]])
    return result
