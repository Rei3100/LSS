"""完了した処理のRecipe本文を、recipe_idで引ける形で残す。

`recipe_id` はRecipe本文のhashなので、保存した本文は読むたびに本文とIDの一致で
検証できる。成果物のタグを読み直す必要がなく、再生ソフトによるtagの書き戻し、
ファイルの移動、成果物の削除があっても、どの処理で作ったかを失わない。

成果物のタグは外部から書き換えられ得るため、内容が完了時と一致しない成果物の
タグは来歴として信用しない。ここはその代わりになる、こちら側の記録である。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from lss.core import session
from lss.core.recipe import validate_recipe_revision

REGISTRY_DIR = "recipes"
_RECIPE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,190}$")


class RecipeRegistryError(RuntimeError):
    """Recipe本文を、IDと一致する形で保存・復元できない。"""


def registry_dir(state_root: Path | None = None) -> Path:
    return (Path(state_root) if state_root is not None else session.state_dir()) / REGISTRY_DIR


def _path(recipe_id: str, state_root: Path | None = None) -> Path:
    if not isinstance(recipe_id, str) or not _RECIPE_ID.fullmatch(recipe_id):
        raise RecipeRegistryError(f"安全なrecipe_idが必要です: {recipe_id!r}")
    return registry_dir(state_root) / f"{recipe_id}.json"


def record(revision: dict, *, state_root: Path | None = None) -> Path:
    """Recipe本文を保存する。本文とIDが一致しないものは保存しない。"""
    try:
        checked = validate_recipe_revision(revision)
    except ValueError as exc:
        raise RecipeRegistryError(f"Recipe本文が不正です: {exc}") from exc
    path = _path(checked["recipe_id"], state_root)
    if path.is_file():
        existing = json.loads(path.read_bytes())
        if existing != checked:
            # 同じIDで別の本文を上書きしない。IDはhashなので起こり得ない事態である。
            raise RecipeRegistryError(f"同じrecipe_idに別の本文が保存されています: {path}")
        return path
    session._write_json_atomic(path, checked)
    return path


def revision_for(recipe_id: str, *, state_root: Path | None = None) -> dict | None:
    """保存済みのRecipe本文。読めない・IDと一致しないものはNoneを返す。"""
    try:
        path = _path(recipe_id, state_root)
    except RecipeRegistryError:
        return None
    try:
        revision = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    try:
        checked = validate_recipe_revision(revision)
    except ValueError:
        return None
    return checked if checked.get("recipe_id") == recipe_id else None


def processing_identity(recipe_id: str, *, state_root: Path | None = None) -> dict | None:
    """Recipe本文から、処理の同一性に使う項目だけを取り出す。"""
    revision = revision_for(recipe_id, state_root=state_root)
    if revision is None:
        return None
    graph = revision.get("stage_graph")
    if not isinstance(graph, list) or not graph or not isinstance(graph[0], dict):
        return None
    stage = graph[0]
    return {"model_id": stage.get("model_id"), "model_sha256": stage.get("model_sha256"),
            "config_sha256": stage.get("config_sha256")}
