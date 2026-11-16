"""Public defaults with an optional machine overlay below LSS_PRIVATE_ROOT.

Overlays live in config/<public stem>.local.json. Test mode reads public defaults
only; malformed configuration is an error, never a reason to use other paths.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class StorageMutationRefused(RuntimeError):
    """Machine scope or protection lists are missing; no files may be removed."""


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return value


def _merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        result[key] = (_merge(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict) else value)
    return result


def load_config(public_path: str | os.PathLike) -> dict:
    """Load defaults and recursively merge an existing absolute-root overlay.

    Dictionaries merge; lists, scalars and null replace. The provenance field is
    assigned by this loader, so JSON cannot claim that a missing overlay exists.
    Relative or unset private roots leave the public defaults in effect.
    """
    public_path = Path(public_path)
    result = _read_object(public_path)
    applied = None
    private_root = os.environ.get("LSS_PRIVATE_ROOT")
    if os.environ.get("LSS_TEST_MODE") != "1" and private_root and Path(private_root).is_absolute():
        overlay = Path(private_root) / "config" / f"{public_path.stem}.local.json"
        if overlay.exists():
            result = _merge(result, _read_object(overlay))
            applied = str(overlay)
    result["_machine_overlay"] = applied
    return result


def require_machine_overlay(policy: dict) -> None:
    """Refuse storage deletion/relocation without machine scope and protections.

    Tests may supply explicit temporary policies with synthetic provenance and
    protections; test mode never authorizes reading the real private overlay.
    """
    if not policy.get("_machine_overlay"):
        raise StorageMutationRefused(
            "Storage mutation refused: no machine overlay was loaded; configure "
            "LSS_PRIVATE_ROOT/config/storage_policy.local.json. Public defaults are read-only.")
    for key in ("never", "protected_packages", "model_stores", "frozen_roots"):
        values = policy.get(key)
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not value.strip() for value in values)):
            raise StorageMutationRefused(f"Storage mutation refused: machine protection list {key!r} is empty or invalid.")
