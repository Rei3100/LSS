import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from lss.core.runtime import RuntimeUnavailableError, capture_cuda_runtime


class _FakeCuda:
    def __init__(self, *, available=True, count=1, fail_allocation=False):
        self.available = available
        self.count = count
        self.fail_allocation = fail_allocation
        self.synchronized = []

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def synchronize(self, index):
        self.synchronized.append(index)

    def get_device_name(self, index):
        return "RTX test"

    def get_device_capability(self, index):
        return 12, 0


def _torch(cuda):
    def empty(size, *, device):
        if cuda.fail_allocation:
            raise RuntimeError("allocation failed")
        return (size, device)

    return SimpleNamespace(
        __version__="2.test", cuda=cuda, empty=empty,
        version=SimpleNamespace(cuda="12.test"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 9999)),
    )


def test_capture_cuda_runtime_probes_device_and_hashes_provider_files(tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_bytes(b"adapter revision")
    cuda = _FakeCuda()

    revision = capture_cuda_runtime(
        "msst", adapter_paths=(adapter,), torch_module=_torch(cuda),
    )

    assert revision.device == "cuda:0"
    assert revision.gpu_name == "RTX test"
    assert revision.gpu_capability == "12.0"
    assert revision.adapter_artifacts == (("adapter.py", hashlib.sha256(b"adapter revision").hexdigest()),)
    assert revision.dependency_versions
    assert cuda.synchronized == [0]
    assert revision.to_dict()["runtime_id"] == revision.runtime_id
    assert len(revision.runtime_id) == 64


def test_runtime_id_changes_when_provider_implementation_changes(tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_bytes(b"one")
    first = capture_cuda_runtime("msst", adapter_paths=(adapter,), torch_module=_torch(_FakeCuda()))
    adapter.write_bytes(b"two")
    second = capture_cuda_runtime("msst", adapter_paths=(adapter,), torch_module=_torch(_FakeCuda()))

    assert first.runtime_id != second.runtime_id


def test_runtime_id_covers_nested_provider_implementation(tmp_path):
    provider = tmp_path / "provider"
    nested = provider / "utils" / "model_utils.py"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"one")
    first = capture_cuda_runtime("msst", adapter_paths=(provider,), torch_module=_torch(_FakeCuda()))
    nested.write_bytes(b"two")
    second = capture_cuda_runtime("msst", adapter_paths=(provider,), torch_module=_torch(_FakeCuda()))

    assert first.runtime_id != second.runtime_id
    assert first.adapter_artifacts[0][0] == "provider/utils/model_utils.py"


@pytest.mark.parametrize(
    "cuda,message",
    [
        (_FakeCuda(available=False), "CPUへはフォールバックしません"),
        (_FakeCuda(count=0), "検出数: 0"),
        (_FakeCuda(fail_allocation=True), "実データを割り当てられません"),
    ],
)
def test_capture_cuda_runtime_never_falls_back_to_cpu(cuda, message):
    with pytest.raises(RuntimeUnavailableError, match=message):
        capture_cuda_runtime("msst", torch_module=_torch(cuda))


def test_capture_cuda_runtime_rejects_missing_adapter(tmp_path):
    with pytest.raises(RuntimeUnavailableError, match="provider実装を識別できません"):
        capture_cuda_runtime(
            "msst", adapter_paths=(Path(tmp_path / "missing.py"),),
            torch_module=_torch(_FakeCuda()),
        )
