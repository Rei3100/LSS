"""推論結果を再現・診断するための実行環境識別とGPU hard gate。"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


class RuntimeUnavailableError(RuntimeError):
    """要求された品質契約を満たす実行環境を利用できない。"""


@dataclass(frozen=True)
class RuntimeRevision:
    provider: str
    device: str
    python_version: str
    platform: str
    torch_version: str
    cuda_build: str
    cudnn_version: str
    gpu_name: str
    gpu_capability: str
    adapter_artifacts: tuple[tuple[str, str], ...]
    dependency_versions: tuple[tuple[str, str], ...] = ()
    native_libraries: tuple[tuple[str, str], ...] = ()

    @property
    def runtime_id(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["adapter_artifacts"] = [
            {"path": path, "sha256": digest}
            for path, digest in self.adapter_artifacts
        ]
        payload["dependency_versions"] = [
            {"name": name, "version": version}
            for name, version in self.dependency_versions
        ]
        payload["native_libraries"] = [
            {"name": name, "version": version}
            for name, version in self.native_libraries
        ]
        payload["runtime_id"] = self.runtime_id
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_source_sha256(path: Path) -> str:
    """改行（CRLF/CR/LF）とUTF-8 BOMの違いで変わらない、source textの内容hash。

    NULを含む・UTF-8として読めないfileは表現の違いを持たない実体として生のbytesでhashする。
    Recipeの識別に使い、checkoutの改行変換（core.autocrlf）で同じ実装が別Recipeにならないようにする。
    """
    data = Path(path).read_bytes()
    if b"\x00" not in data:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            if data.startswith(b"\xef\xbb\xbf"):
                data = data[3:]
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _provider_artifacts(paths: Iterable[Path], *, normalize_text: bool = False) -> tuple[tuple[str, str], ...]:
    """provider配布物を配置場所に依存しない相対名と内容hashで識別する。

    ``normalize_text`` はRecipe識別用（`text_source_sha256`）。実行環境ID（runtime_id）は生のbytesのまま。
    """
    digest_of = text_source_sha256 if normalize_text else _sha256
    records: list[tuple[str, str]] = []
    ignored_parts = frozenset({".git", "__pycache__", ".pytest_cache"})
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path.is_file():
            records.append((path.name, digest_of(path)))
            continue
        if not path.is_dir():
            raise RuntimeUnavailableError(f"provider実装を識別できません: {path}")
        for artifact in sorted(path.rglob("*"), key=lambda item: item.as_posix().casefold()):
            relative = artifact.relative_to(path)
            if any(part in ignored_parts for part in relative.parts) or artifact.suffix.lower() == ".pyc":
                continue
            if artifact.is_symlink():
                raise RuntimeUnavailableError(f"provider実装にlinkを含められません: {artifact}")
            if artifact.is_file():
                records.append((f"{path.name}/{relative.as_posix()}", digest_of(artifact)))
    labels = [label for label, _ in records]
    if len(labels) != len(set(labels)):
        raise RuntimeUnavailableError("provider実装の相対artifact名が重複しています")
    return tuple(sorted(records))


def _dependency_versions() -> tuple[tuple[str, str], ...]:
    installed = {
        ((distribution.metadata.get("Name") or "unknown").strip().casefold(), distribution.version)
        for distribution in metadata.distributions()
    }
    return tuple(sorted(installed))


def capture_cuda_runtime(
    provider: str,
    *,
    device_index: int = 0,
    adapter_paths: Iterable[Path] = (),
    torch_module=None,
) -> RuntimeRevision:
    """CUDA実行を実割当で確認し、出力へ結び付けられる実行環境識別を返す。

    CUDAが見えない、指定GPUが存在しない、または実デバイス割当が失敗した場合は
    CPUへフォールバックせず、推論開始前に失敗する。
    """
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("providerが未指定です")
    if not isinstance(device_index, int) or isinstance(device_index, bool) or device_index < 0:
        raise ValueError("device_indexは0以上の整数である必要があります")

    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeUnavailableError("PyTorch実行環境を読み込めません") from exc

    try:
        available = bool(torch_module.cuda.is_available())
        device_count = int(torch_module.cuda.device_count())
    except Exception as exc:
        raise RuntimeUnavailableError("CUDA実行環境を確認できません") from exc
    if not available:
        raise RuntimeUnavailableError("CUDA GPUを利用できません。CPUへはフォールバックしません")
    if device_index >= device_count:
        raise RuntimeUnavailableError(
            f"CUDAデバイス{device_index}を利用できません（検出数: {device_count}）"
        )

    device = f"cuda:{device_index}"
    try:
        probe = torch_module.empty(1, device=device)
        torch_module.cuda.synchronize(device_index)
        del probe
        gpu_name = str(torch_module.cuda.get_device_name(device_index))
        major, minor = torch_module.cuda.get_device_capability(device_index)
        cudnn_value = torch_module.backends.cudnn.version()
    except Exception as exc:
        raise RuntimeUnavailableError(
            f"CUDAデバイス{device_index}へ実データを割り当てられません"
        ) from exc

    artifacts = _provider_artifacts(adapter_paths)
    native_libraries = []
    try:
        import soundfile as soundfile_module
        native_libraries.append(("libsndfile", str(soundfile_module.__libsndfile_version__)))
    except (ImportError, AttributeError):
        pass

    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    return RuntimeRevision(
        provider=provider.strip(),
        device=device,
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=str(torch_module.__version__),
        cuda_build=str(cuda_version or "unknown"),
        cudnn_version=str(cudnn_value if cudnn_value is not None else "unknown"),
        gpu_name=gpu_name,
        gpu_capability=f"{major}.{minor}",
        adapter_artifacts=artifacts,
        dependency_versions=_dependency_versions(),
        native_libraries=tuple(native_libraries),
    )
