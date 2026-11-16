"""Check that exact development hot paths resolve to the configured SSD volume.

    python scripts/dev_hot_paths.py [--output PATH] [--model NAME=PATH] [--extra PATH]

Prints one JSON report and exits nonzero if any required path resolves off the
hot volume. Resolution/stat calls apply only to the listed paths; this script
never walks a worktree, model directory, HDD, or output tree and never moves data.
Registry entries without a machine path are listed as unavailable, not checked
as paths. Explicit --model paths remain required checks.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

try:
    from scripts.machine_config import load_config
except ModuleNotFoundError:
    from machine_config import load_config

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "scripts" / "storage_policy.json"
MODEL_REGISTRY = ROOT / "scripts" / "local_llm_models.json"


class HotPathError(RuntimeError):
    """A required development path resolves to a volume other than the hot SSD."""

    def __init__(self, record: dict):
        self.record = record
        allowed = record.get('allowed_volumes', [record['hot_volume']])
        super().__init__(f"{record['label']} resolves to {record['volume']} ({record['resolved']}), "
                         f"outside configured fast volumes {allowed}")


def resolve_path(path: str | os.PathLike) -> str:
    """Resolve symlinks/junctions at this exact path without enumerating its tree."""
    return str(Path(path).expanduser().resolve(strict=False))


def _volume_mount(path: str) -> str:
    """Return the actual Windows volume mount for a resolved path, or a POSIX anchor."""
    if os.name != "nt":
        anchor = Path(path).anchor
        if not anchor:
            raise ValueError(f"path has no volume anchor: {path}")
        return anchor.casefold()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_mount = kernel32.GetVolumePathNameW
    get_mount.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    get_mount.restype = ctypes.c_int
    buf = ctypes.create_unicode_buffer(32768)
    if not get_mount(path, buf, len(buf)):
        raise OSError(ctypes.get_last_error(), f"GetVolumePathNameW failed for {path}")
    get_name = kernel32.GetVolumeNameForVolumeMountPointW
    get_name.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    get_name.restype = ctypes.c_int
    volume_name = ctypes.create_unicode_buffer(128)
    if not get_name(buf.value, volume_name, len(volume_name)):
        # Some removable/network volumes refuse the name API even though the
        # OS resolved their true mount point. Keep the resolved mount as a
        # conservative distinct identity; never borrow the input's drive text.
        return "mount:" + os.path.normcase(os.path.normpath(buf.value)).casefold()
    # Compare the filesystem's volume ID, not its drive-letter alias. This also
    # handles directory mount points that place a subtree on another volume.
    return volume_name.value.casefold()


def require_hot(path: str | os.PathLike, label: str, hot_volume=None, *,
                resolver: Callable = resolve_path, volume_resolver: Callable = _volume_mount) -> dict:
    """Accept the resolved identity of any explicitly configured fast SSD.

    Inject `resolver` and `volume_resolver` in tests to model symlinks/junctions
    and physical volumes on platforms without Windows volume APIs.
    """
    resolved = resolver(path)
    if hot_volume is None:
        policy = load_config(POLICY_PATH)
        if not policy.get('fast_volumes'):
            # Which volumes are fast is a fact about one machine (the overlay names them); the public
            # defaults do not. This is a speed admission, not a protection: record that it was not checked.
            return {"label": str(label), "input": str(path), "resolved": str(resolved),
                    "ok": True, "checked": False, "reason": "no fast volumes are configured for this machine"}
        hot_volume = policy['fast_volumes']
    configured = [hot_volume] if isinstance(hot_volume, str) else list(hot_volume)
    volume = volume_resolver(resolved)
    mounts = [volume_resolver(resolver(v)) for v in configured]
    hot_mount = mounts[0]
    record = {"label": str(label), "input": str(path), "resolved": str(resolved),
              "volume": str(volume), "hot_volume": str(hot_mount),
              "ok": volume in mounts}
    if len(mounts) > 1:
        record['allowed_volumes'] = mounts
    if not record["ok"]:
        raise HotPathError(record)
    return record


def inspect_paths(paths: Iterable[tuple[str, str]], hot_volume: str, *,
                  resolver: Callable = resolve_path, volume_resolver: Callable = _volume_mount) -> dict:
    records = []
    for label, path in paths:
        try:
            records.append(require_hot(path, label, hot_volume, resolver=resolver, volume_resolver=volume_resolver))
        except (HotPathError, OSError, ValueError) as exc:
            if isinstance(exc, HotPathError):
                records.append(exc.record)
            else:
                records.append({"label": str(label), "input": str(path), "resolved": None,
                                "volume": None, "hot_volume": None, "ok": False, "error": str(exc)})
    return {"ok": all(row["ok"] for row in records), "hot_volume_configured": str(hot_volume),
            "checked": len(records), "paths": records}


def _git_common_dir(repo: Path) -> Path:
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-common-dir"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"cannot resolve git common directory for {repo}: {result.stderr.strip()}")
    path = Path(result.stdout.strip())
    return path if path.is_absolute() else repo / path


def default_paths(repo: Path = ROOT, *, prefix: str = sys.prefix, base_prefix: str = sys.base_prefix,
                  temp_dir: str | None = None, environ: dict | None = None,
                  registry_path: Path = MODEL_REGISTRY, policy_path: Path = POLICY_PATH) -> tuple[list[tuple[str, str]], str]:
    """Resolve the bounded default path list; never enumerate directories or model files."""
    env = os.environ if environ is None else environ
    policy = load_config(policy_path)
    hot_volume = policy.get('fast_volumes', str(policy["hot_volume"]))
    paths = [("repository", str(repo)), ("git_common_dir", str(_git_common_dir(repo))),
             ("python_prefix", str(prefix)), ("python_base_prefix", str(base_prefix)),
             ("temp_directory", str(temp_dir if temp_dir is not None else tempfile.gettempdir()))]
    local_app_data = env.get("LOCALAPPDATA")
    if local_app_data:
        paths.append(("local_app_data", str(Path(os.path.expandvars(local_app_data)))))
    stage_root = policy.get("stage", {}).get("root")
    if stage_root:
        paths.append(("model_stage_cache", os.path.expandvars(str(stage_root))))
    if policy.get('llama_runtime_root'):
        paths.append(('local_llm_runtime', policy['llama_runtime_root']))
    if registry_path.is_file():
        registry = load_config(registry_path)
        for name, model in registry.get("models", {}).items():
            if model.get("archive_only"):
                continue
            if not model.get("path"):
                continue  # Public metadata alone never supplies a model location.
            model_path = os.path.expandvars(str(model["path"]))
            paths.append((f"model:{name}", model_path))
    return paths, hot_volume


def _parse_model(value: str) -> tuple[str, str]:
    if "=" in value:
        label, path = value.split("=", 1)
        if label and path:
            return f"model:{label}", path
    raise argparse.ArgumentTypeError("use NAME=PATH")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--model-registry", type=Path, default=MODEL_REGISTRY)
    parser.add_argument("--output", action="append", default=[], metavar="PATH",
                        help="additional required job output path; repeat for each path")
    parser.add_argument("--model", action="append", type=_parse_model, default=[], metavar="NAME=PATH",
                        help="additional required model file/path; repeat as needed")
    parser.add_argument("--extra", action="append", default=[], metavar="PATH",
                        help="additional required development path; repeat as needed")
    args = parser.parse_args(argv)
    paths, hot_volume = default_paths(args.repo, registry_path=args.model_registry, policy_path=args.policy)
    paths.extend((f"output:{i + 1}", path) for i, path in enumerate(args.output))
    paths.extend(args.model)
    paths.extend((f"extra:{i + 1}", path) for i, path in enumerate(args.extra))
    report = inspect_paths(paths, hot_volume)
    registry = load_config(args.model_registry) if args.model_registry.is_file() else {}
    report["unavailable_models"] = [
        {"name": name, "reason": "no machine path configured"}
        for name, model in registry.get("models", {}).items() if not model.get("path")
    ]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
