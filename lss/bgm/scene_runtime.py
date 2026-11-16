"""Which scene-name model backend runs here, and the facts the job needs about it.

- ``native``: whisper.cpp + llama.cpp programs (``scene_native``); works in the frozen app and from source whenever
  its engines and weights are installed (``scene_native.ready()``).
- ``transformers``: torch/bitsandbytes (``scene_models.SceneModels``); source runtime only (the packaged app does not
  contain that stack), kept for parity measurements and as the source-development fallback.

``LSS_SCENE_BACKEND=native|transformers`` forces one (it is then used only if it is available, never silently
swapped). Automatic choice prefers ``native``. A result records its backend inside ``revision()``, which is part of
the retry key of every unresolved attempt in ``scene_job``; a cue that already has a name keeps it whichever backend
produced it, and an unresolved cue is retried when the backend (or an engine/weight hash) differs.
"""
from __future__ import annotations

import importlib.util
import os


def transformers_available() -> bool:
    from lss.paths import is_frozen

    if is_frozen():
        return False
    return all(importlib.util.find_spec(name) is not None for name in ("transformers", "bitsandbytes", "huggingface_hub"))


def native_available() -> bool:
    from lss.bgm import scene_native

    return scene_native.ready()


def backend() -> str | None:
    forced = (os.environ.get("LSS_SCENE_BACKEND") or "").strip().lower()
    if forced == "native":
        return "native" if native_available() else None
    if forced == "transformers":
        return "transformers" if transformers_available() else None
    if forced:
        return None
    if native_available():
        return "native"
    return "transformers" if transformers_available() else None


def bootstrap_supported() -> bool:
    """A frozen installation can prepare the native backend when a cue actually needs it."""
    from lss.paths import is_frozen

    forced = (os.environ.get("LSS_SCENE_BACKEND") or "").strip().lower()
    return is_frozen() and forced in ("", "native")


def ensure_bootstrapped() -> None:
    """Prepare native resources only for an eligible, identity-checked background job."""
    if available():
        return
    if not bootstrap_supported():
        raise RuntimeError("scene_naming_backend_unavailable")
    from lss.bgm import scene_native

    scene_native.ensure_ready()
    if not native_available():
        raise RuntimeError("scene_native_bootstrap_not_ready")


def available() -> bool:
    return backend() is not None


def create_models():
    chosen = backend()
    if chosen == "native":
        from lss.bgm.scene_native import NativeSceneModels

        return NativeSceneModels()
    from lss.bgm.scene_models import SceneModels, job_execution_profile

    return SceneModels(execution_profile=job_execution_profile())


def revision() -> dict:
    """The revision the current backend would record, without loading anything."""
    if backend() == "native":
        from lss.bgm import scene_native

        return scene_native.revision()
    from lss.bgm.scene_models import ASR, LLM

    return {"asr": {k: ASR[k] for k in ("repository", "revision")},
            "llm": {k: LLM[k] for k in ("repository", "revision", "quantization", "thinking")}}


def models_cached() -> bool:
    """Whether the selected backend is ready locally; eligible frozen recovery can prepare native files."""
    chosen = backend()
    if chosen == "native":
        return True
    if chosen is None:
        return False
    from lss.bgm import scene_models

    root = scene_models.model_root()
    return all((root / spec["repository"].replace("/", "--") / spec["revision"] / ".lss-complete.json").is_file()
               for spec in (scene_models.ASR, scene_models.LLM))


def read_only_status() -> dict:
    """Observe readiness without acquisition, model construction, GPU probes or state writes."""
    from lss.paths import is_frozen
    from lss.bgm import scene_native

    report = {"schema_version": 1, "status": "unresolved", "frozen": is_frozen(),
              "available": None, "backend": None, "native_ready": None,
              "native_bootstrap_supported": bootstrap_supported(), "native_problem_codes": []}
    try:
        root = scene_native.native_root()
        manifest = root / scene_native.MANIFEST
        if manifest.is_file() and manifest.stat().st_size > 256 * 1024:
            report["native_problem_codes"] = ["manifest_too_large"]
            return report
        problems = scene_native._manifest_problems(root)
        if (root / "bootstrap-in-progress").exists():
            problems = [*problems, "bootstrap_in_progress"]
        chosen = backend()
        forced = (os.environ.get("LSS_SCENE_BACKEND") or "").strip().lower()
        if ((chosen == "native" and problems)
                or (not problems and forced in ("", "native") and chosen != "native")):
            report["native_problem_codes"] = ["readiness_changed_during_observation"]
            return report
        codes = {str(item).split(":", 1)[0] for item in problems}
        if problems and scene_native._read_json_dict(root / "bootstrap-status.json").get("reason") == \
                scene_native.SupplyMismatch.reason_code:
            # A pinned upstream file is not the pinned bytes: a supply problem, not a transient network error.
            codes.add(scene_native.SupplyMismatch.reason_code)
        report.update(status="observed", available=chosen is not None, backend=chosen,
                      native_ready=not problems, native_problem_codes=sorted(codes))
    except Exception as exc:  # Diagnostics expose no model paths, transcript or exception message.
        report["error_type"] = type(exc).__name__
    return report
