"""モデル設定YAMLの安全な読み込みと、壊れたキャッシュの1回限りの自己修復。"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml


class ModelConfigLoader(yaml.SafeLoader):
    """MSST設定で使われるtupleだけをSafeLoaderへ追加する。"""


def _construct_python_tuple(loader: ModelConfigLoader, node: yaml.SequenceNode) -> tuple:
    return tuple(loader.construct_sequence(node))


ModelConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)


def load_model_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=ModelConfigLoader)
    if not isinstance(config, dict):
        raise ValueError(f"モデル設定のルートがオブジェクトではありません: {path}")
    return config


def load_cached_config(path: Path, download: Callable[[], Path]) -> dict:
    """設定を取得して読み込み、既存キャッシュの構文破損だけは再取得で修復する。

    禁止されたPythonオブジェクトタグは、再取得で隠さず直ちに拒否する。新規取得した
    設定自体が不正な場合も無限再試行せず、壊れた実体をキャッシュへ残さない。
    """
    was_cached = path.is_file() and path.stat().st_size > 0
    download()
    try:
        return load_model_config(path)
    except yaml.constructor.ConstructorError:
        path.unlink(missing_ok=True)
        raise
    except (OSError, UnicodeError, yaml.YAMLError, ValueError):
        path.unlink(missing_ok=True)
        if not was_cached:
            raise

    download()
    try:
        return load_model_config(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
