"""CurrentBestの履歴のファイルが、同じrunの識別子・同じ公開名で音の違う版を取り違えないようにする。

同じRecipeを同じ原本に掛け直すとrunの識別子は同じになり、GPUの推論は別の音を出しうる。``current_best`` の
履歴の場所は「runの識別子＋公開名」だけで決まるため、音の違う二つの版が同じ場所へ退避されると、後の版が先の版に
阻まれて履歴に残らず、rollbackの往復で正確な音へ戻れない。

Archive collision handling remains in this journal wrapper. The alias-safe public-file identity and final
publication verification also protect direct callers in ``current_best.py``; that safety fix changes its
Recipe-bound implementation bytes. Existing run/history records are preserved, never relabelled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from lss.bgm import current_best


def _claimed_by_other_audio(journal: dict, path: str, pcm: str | None) -> bool:
    history = (journal.get("next_record") or {}).get("history") or []
    if any(row.get("archive_path") == path and row.get("pcm_sha256") != pcm
           for snapshot in history for row in snapshot.get("files") or []):
        return True
    return Path(path).is_file() and not current_best._matches(Path(path), pcm)


def separate_history_copies(journal: dict) -> dict:
    """外すファイルの履歴の場所が別の音の版に取られている時、その版だけ音のhashの付いた場所へ移す（journalを直す）。

    同じ音の版は同じ場所を共有する。直すのは、この切り替えで外すファイルの行と、この切り替えで加えた履歴の行だけ。"""
    history = (journal.get("next_record") or {}).get("history") or []
    newest = history[-1]["files"] if history else []
    for row in journal.get("retire") or []:
        path, pcm = row.get("archive_path"), row.get("pcm_sha256")
        if not path or not pcm or not _claimed_by_other_audio(journal, path, pcm):
            continue
        old = Path(path)
        moved = str(old.with_name(f"{old.stem}.{pcm[:8]}{old.suffix}"))
        row["archive_path"] = moved
        for item in newest:
            if (item.get("public_path") == row["public_path"] and item.get("pcm_sha256") == pcm
                    and item.get("archive_path") == path):
                item["archive_path"] = moved
    return journal


def _relocated(snapshot: dict, public_folder: Path | None) -> dict:
    """A history snapshot whose files named another place (a moved library, a copied state) restores beside this
    source instead: the old place is read (its history copy) but never written."""
    if public_folder is None:
        return snapshot
    from lss.bgm import write_targets

    files = []
    for item in snapshot.get("files") or []:
        if write_targets.inside(Path(item["public_path"]), public_folder):
            files.append(item)
        else:
            files.append({**item, "public_path": str(Path(public_folder) / Path(item["public_path"]).name),
                          "relocated_from": item["public_path"]})
    return {**snapshot, "files": files}


def restore_previous(state_root: Path, source_sha256: str, history_root: Path,
                     regenerate: Callable[[dict, Path], None] | None = None, *,
                     public_folder: Path | None = None) -> dict:
    """``current_best.restore_previous`` と同じrollback。履歴の場所だけ ``separate_history_copies`` で直す。

    ``public_folder`` (this source's public folder) keeps every restored file inside it (`_relocated`)."""
    record = current_best.read(state_root, source_sha256)
    if record is None or not record.get("history"):
        raise current_best.CurrentBestError("戻せる前の結果がありません")
    target = _relocated(record["history"][-1], public_folder)
    current = record["current"]
    target_paths = {current_best.public_audio_key(item) for item in target["files"]}
    current_paths = {current_best.public_audio_key(item) for item in current["files"]}
    retire = [item for item in current["files"] if current_best.public_audio_key(item) not in target_paths]
    files = []
    staged = {}
    for item in target["files"]:
        entry = {name: value for name, value in item.items() if name != "archive_path"}
        if current_best.public_audio_key(item) not in current_paths:
            entry["placed_now"] = True
            if item.get("archive_path"):
                staged[item["public_path"]] = Path(item["archive_path"])
        files.append(entry)
    restored = {"files": files, "pinned": True, "adopted_at": current_best._now(), "transition": "restored_previous",
                "evaluated_runs": list(target.get("evaluated_runs") or [])}
    journal = current_best.plan_switch(state_root, source_sha256, restored, staged, history_root, retire=retire,
                                       record=record, previous_current=current, mode="restored_previous",
                                       history=list(record["history"][:-1]))
    return current_best.execute(state_root, separate_history_copies(journal), regenerate)
