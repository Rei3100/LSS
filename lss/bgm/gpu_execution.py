"""Validated, internal execution-only policies for the BGM SFC worker."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import os
import threading
import time
from collections.abc import Callable, Iterator


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    cadence_seconds: float


_PROFILES = {
    "strict_deterministic": ExecutionProfile("strict_deterministic", 0.0),
    "strict_deterministic_paced60": ExecutionProfile("strict_deterministic_paced60", 60.0),
}
_ENV_LOCK = threading.RLock()


def resolve_execution_profile(value: str | None) -> ExecutionProfile | None:
    """Resolve the narrow allowlist; ``None`` preserves the legacy execution path."""
    if value is None:
        return None
    if not isinstance(value, str) or value not in _PROFILES:
        raise ValueError(f"Unsupported internal SFC execution profile: {value!r}")
    return _PROFILES[value]


def profile_identity(value: str | None) -> dict | None:
    profile = resolve_execution_profile(value)
    if profile is None:
        return None
    return {"execution_profile": profile.name, "deterministic_algorithms": True,
            "cudnn_deterministic": True, "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8", "matmul_allow_tf32": False,
            "cadence_seconds_after_forward": profile.cadence_seconds}


def assert_profile_active(value: str | None, torch) -> None:
    """Fail closed if a selected profile reached SFC without its worker context."""
    if value is None:
        return
    profile = resolve_execution_profile(value)
    effective = _read_effective(torch)
    expected = profile_identity(profile.name)
    for key in ("deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark",
                "cublas_workspace_config", "matmul_allow_tf32"):
        if effective[key] != expected[key]:
            raise RuntimeError(f"Execution profile {profile.name} is not active: {key}={effective[key]!r}")
    if effective["deterministic_warn_only"] is not False:
        raise RuntimeError(
            f"Execution profile {profile.name} is not strict: "
            f"deterministic_warn_only={effective['deterministic_warn_only']!r}"
        )


def _read_effective(torch) -> dict:
    return {
        "torch_version": str(torch.__version__),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }


@contextmanager
def configure_execution(value: str | None, *, torch_loader: Callable[[], object] | None = None) -> Iterator[dict | None]:
    """Apply and record a strict deterministic profile, restoring process state on exit.

    For a selected profile CUBLAS is configured before importing torch or touching CUDA.
    The caller must hold the worker GPU lease for the yielded scope.
    """
    profile = resolve_execution_profile(value)
    if profile is None:
        yield None
        return

    with _ENV_LOCK:
        previous_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch, old = None, None
        try:
            torch = (torch_loader or (lambda: importlib.import_module("torch")))()
            cuda = getattr(torch, "cuda", None)
            cuda_initialized = bool(callable(getattr(cuda, "is_initialized", None)) and cuda.is_initialized())
            if cuda_initialized and previous_env != ":4096:8":
                raise RuntimeError("CUDA was initialized before strict CUBLAS workspace configuration")
            old = {
                "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
                "warn_only": bool(getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)()),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            }
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            effective = _read_effective(torch)
            if (not effective["deterministic_algorithms"] or effective["deterministic_warn_only"]
                    or not effective["cudnn_deterministic"] or effective["cudnn_benchmark"]
                    or effective["cublas_workspace_config"] != ":4096:8"
                    or effective["matmul_allow_tf32"]):
                raise RuntimeError(f"Strict deterministic execution profile did not take effect: {effective}")
            yield {"profile": profile.name, "cadence_seconds_after_forward": profile.cadence_seconds,
                   "effective": effective}
        finally:
            if torch is not None and old is not None:
                torch.use_deterministic_algorithms(old["deterministic"], warn_only=old["warn_only"])
                torch.backends.cudnn.deterministic = old["cudnn_deterministic"]
                torch.backends.cudnn.benchmark = old["cudnn_benchmark"]
                torch.backends.cudnn.allow_tf32 = old["cudnn_allow_tf32"]
                torch.backends.cuda.matmul.allow_tf32 = old["matmul_allow_tf32"]
            if previous_env is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_env


def attach_forward_cadence(model, torch, value: str | None, events: list[dict], *,
                           sleeper: Callable[[float], None] = time.sleep,
                           clock: Callable[[], float] = time.time,
                           monotonic: Callable[[], float] = time.monotonic,
                           progress: Callable[[dict], None] | None = None):
    """Attach a post-forward-only rest for the paced profile; return a removable handle."""
    profile = resolve_execution_profile(value)
    if profile is None or profile.cadence_seconds == 0:
        return None

    def after_forward(_module, _inputs, _output):
        torch.cuda.synchronize()
        # Discard only unreferenced allocator cache; model weights and live tensors remain resident.
        torch.cuda.empty_cache()
        started = clock()
        event = {"completed_at": started, "allocated_bytes": int(torch.cuda.memory_allocated()),
                 "reserved_bytes": int(torch.cuda.memory_reserved()),
                 "sleep_seconds": profile.cadence_seconds}
        events.append(event)
        if progress is not None:
            progress({"phase": "forward_rest", **event})
        rest_started = monotonic()
        while True:
            elapsed = monotonic() - rest_started
            if elapsed >= profile.cadence_seconds:
                break
            sleeper(profile.cadence_seconds - elapsed)
        event["rest_elapsed_seconds"] = elapsed
        event["rest_completed_at"] = clock()

    return model.register_forward_hook(after_forward)


def remove_forward_hook(handle) -> None:
    """Remove a temporary cadence hook without obscuring an inference exception."""
    if handle is not None:
        try:
            handle.remove()
        except Exception:
            # Model is discarded by the inference caller regardless; a removal
            # failure must not mask the original forward or output validation error.
            pass
