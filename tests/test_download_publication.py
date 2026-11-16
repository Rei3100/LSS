"""共有違反の再試行と、再試行できない場合の既存モデル保全。"""
from pathlib import Path

import pytest

from lss.harness import fetch


class Response:
    headers = {}
    history = []
    url = "https://example.com/weight.ckpt"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield b"validated new weight"


def test_transient_sharing_violation_retries_the_same_validated_candidate(tmp_path, monkeypatch):
    destination = tmp_path / "weight.ckpt"
    destination.write_bytes(b"old weight")
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: Response())
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)
    replace = Path.replace
    candidates = []

    def publish(path, target):
        candidates.append(path)
        assert path.read_bytes() == b"validated new weight"
        assert destination.read_bytes() == b"old weight"
        if len(candidates) <= 2:
            raise PermissionError("一時的な共有違反")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", publish)

    assert fetch.download(Response.url, destination, expected_size=20) == destination
    assert len(candidates) == 3 and len(set(candidates)) == 1
    assert destination.read_bytes() == b"validated new weight"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("failure", [PermissionError, OSError])
def test_failed_publication_preserves_existing_model_and_cleans_candidate(tmp_path, monkeypatch, failure):
    destination = tmp_path / "weight.ckpt"
    destination.write_bytes(b"old weight")
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: Response())
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)
    calls = []

    def fail(path, target):
        calls.append(path)
        raise failure("保存できません")

    monkeypatch.setattr(Path, "replace", fail)

    with pytest.raises(failure, match="保存できません"):
        fetch.download(Response.url, destination, expected_size=20)

    assert len(calls) == (6 if failure is PermissionError else 1)
    assert destination.read_bytes() == b"old weight"
    assert list(tmp_path.iterdir()) == [destination]
