"""状態領域の一時障害で起動できなくても、pipeを解放して再起動できる。"""
import pytest

from lss.core import worker


@pytest.mark.parametrize("failure_type", [PermissionError, OSError])
def test_startup_state_failure_closes_listener_and_allows_retry(monkeypatch, failure_type):
    occupied = set()
    closed = []
    allow_accept = False

    class Listener:
        def __init__(self, address, **kwargs):
            if address in occupied:
                raise OSError("pipe使用中")
            self.address = address
            occupied.add(address)

        def close(self):
            occupied.remove(self.address)
            closed.append(self.address)

        def accept(self):
            assert allow_accept, "起動状態の確認前に要求を受け付けない"
            raise KeyboardInterrupt

    monkeypatch.setattr(worker.connection, "Listener", Listener)
    monkeypatch.setattr(worker.session, "cleanup_stale_work", lambda: [])
    monkeypatch.setattr(worker.session, "reconcile_publications", lambda: [], raising=False)

    def fail():
        raise failure_type("状態領域が一時的に使えない")

    with monkeypatch.context() as patch:
        patch.setattr(worker.session, "cleanup_stale_work", fail)
        with pytest.raises(failure_type, match="一時的"):
            worker.serve(address="isolated-pipe")
    assert occupied == set()
    allow_accept = True
    worker.serve(address="isolated-pipe")
    assert occupied == set()
    assert closed == ["isolated-pipe", "isolated-pipe"]
