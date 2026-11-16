"""開発ツリーとPyInstaller onedirで、可変データと同梱リソースの基準を分ける。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TEST_MODE_ENV = "LSS_TEST_MODE"
PROTECTED_STATE_ROOTS_ENV = "LSS_PROTECTED_STATE_ROOTS"


class ProtectedStateError(RuntimeError):
    """A test run resolved a state root that belongs to the user's product or to development records."""


def guard_state_root(path: Path) -> Path:
    """In test mode (`LSS_TEST_MODE=1`) refuse a state root inside a protected root, links followed.

    Protected roots are the `os.pathsep`-separated folders of `LSS_PROTECTED_STATE_ROOTS`. Outside test mode
    the path is returned unchanged, so the product itself never takes this branch.
    """
    if os.environ.get(TEST_MODE_ENV) != "1":
        return path
    resolved = os.path.normcase(os.path.realpath(path))
    for raw in os.environ.get(PROTECTED_STATE_ROOTS_ENV, "").split(os.pathsep):
        if not raw.strip():
            continue
        root = os.path.normcase(os.path.realpath(raw.strip()))
        if resolved == root or resolved.startswith(root.rstrip(os.sep) + os.sep):
            raise ProtectedStateError(f"test mode: the state root resolves into a protected folder: {path}")
    return path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """可変データを置く配布ルート。開発時はリポジトリルート。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    """同梱された読み取り用リソースのルート。"""
    if is_frozen():
        bundled = getattr(sys, "_MEIPASS", None)
        return Path(bundled).resolve() if bundled else app_root() / "_internal"
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """既存呼び出し向けの別名。可変データの基準としてのみ使う。"""
    return app_root()
