"""Where LSS may write the presentation (tags) of a published BGM cue outside the run that publishes it.

``docs/product/IDENTITY_AND_CURRENTBEST.md``: name/path/tag are presentation, and the CurrentBest record is the only
place that says which file is a cue's public file. A presentation refresh (another source's recurrence title, scene
names, the recurrence asset's library) may write a file only when all of these hold; anything else is refused with
a reason the caller records, and nothing is written:

- the target is the CurrentBest row's ``public_path`` itself: absolute, no ``..`` component, a ``.flac`` file, and
  not a symlink (a link could name a file anywhere, e.g. the library a state was copied from);
- the row was not adopted by another state: its ``run_path`` does not lie in another state root's
  ``bgm-candidates``. A state copied from another installation still names that installation's library in its CurrentBest rows; those paths are
  observations, never write targets, until this state processes the source and relocates the cue to its own folder;
- the file is still that cue: its ``LSS_BGM_CUE_KEY`` tag equals the row's ``cue_key`` (a user's file, or another
  cue that took the name, is never retagged).

This module is deliberately outside the BGM Recipe implementation set: it changes no audio or adoption decision.
"""
from __future__ import annotations

import os
from pathlib import Path

CANDIDATES_DIR = "bgm-candidates"


class WriteTargetRefused(ValueError):
    """The CurrentBest row does not name a file this state may write. ``reason`` is a stable token."""

    def __init__(self, reason: str, path: object):
        super().__init__(f"{reason}: {path}")
        self.reason = reason
        self.path = str(path)


def _same(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a)) == os.path.normcase(str(b))


def _resolved(path: Path) -> Path:
    try:
        return Path(os.path.abspath(path)).resolve()
    except (OSError, RuntimeError, ValueError):
        return Path(os.path.abspath(path))


def same_place(a: Path, b: Path) -> bool:
    """``a`` and ``b`` name one place under any spelling (case, junction or symlink)."""
    return _same(Path(os.path.abspath(a)), Path(os.path.abspath(b))) or _same(_resolved(a), _resolved(b))


def _own_candidate_dirs(state_root: Path) -> list[Path]:
    """Every spelling of this state's ``bgm-candidates``: raw, resolved, and through the directory junction a tiered
    installation makes of it (``<state>/bgm-candidates`` -> another volume)."""
    raw = Path(os.path.abspath(state_root))
    return [raw / CANDIDATES_DIR, _resolved(raw) / CANDIDATES_DIR, _resolved(raw / CANDIDATES_DIR)]


def adopted_elsewhere(state_root: Path, row: dict) -> bool:
    """The row's adopting run record lies in *another* state root's ``bgm-candidates`` (a copied/relocated state).

    Real CurrentBest rows always name ``<state>/bgm-candidates/<run>/run/run.json``; a row without such a path
    carries no evidence of another owner and is judged by the other checks alone. A row is elsewhere only when its
    ``bgm-candidates`` matches this state's under none of the spellings (raw, resolved, junction target): a state
    whose ``bgm-candidates`` is a junction to another volume is still this state.
    """
    try:
        run_path = Path(row["run_path"])
    except (KeyError, TypeError):
        return False
    try:
        if not run_path.is_absolute():
            return False
        own = [os.path.normcase(str(p)) for p in _own_candidate_dirs(state_root)]
        seen = False
        for spelled in (run_path, _resolved(run_path)):
            parents = spelled.parents
            if len(parents) >= 4 and parents[2].name.casefold() == CANDIDATES_DIR:
                seen = True
                if os.path.normcase(str(parents[2])) in own:
                    return False
        return seen
    except (ValueError, OSError, RuntimeError):
        return False


def inside(path: Path, folder: Path) -> bool:
    """``path`` is a file directly in ``folder`` (junction/symlink/case spellings of one place agree)."""
    try:
        return same_place(Path(path).parent, folder) or _same(_resolved(path).parent, _resolved(folder))
    except (OSError, RuntimeError, ValueError):
        return False


def checked(state_root: Path, row: dict, cue_key: str | None = None) -> Path:
    """The row's ``public_path`` when this state may write its presentation; ``WriteTargetRefused`` otherwise.

    ``cue_key`` is the key the caller looked the row up by (`current_best.published_files`); default: the row's.
    """
    from mutagen.flac import FLAC

    raw = (row or {}).get("public_path")
    if not raw:
        raise WriteTargetRefused("no_public_path", raw)
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts or path.suffix.casefold() != ".flac":
        raise WriteTargetRefused("not_a_plain_public_path", path)
    if path.is_symlink():
        raise WriteTargetRefused("public_path_is_a_link", path)
    if not path.is_file():
        raise WriteTargetRefused("public_file_missing", path)
    if adopted_elsewhere(state_root, row):
        raise WriteTargetRefused("adopted_by_another_state", path)
    try:
        key = FLAC(path).get("LSS_BGM_CUE_KEY")
    except Exception as exc:  # noqa: BLE001 - an unreadable file is not proven to be the cue
        raise WriteTargetRefused("public_file_unreadable", path) from exc
    if key != [str(cue_key if cue_key is not None else row.get("cue_key"))]:
        raise WriteTargetRefused("public_file_is_not_this_cue", path)
    return path
