"""未コミットのキュー削除APIにも、同じ保存失敗からの復旧を適用する。"""
import pytest

from tests.test_queue_storage_recovery import test_queue_mutation_is_retryable_after_storage_failure as check_recovery


@pytest.mark.parametrize("boundary", ["flush", "replace"])
def test_remove_is_retryable_after_storage_failure(tmp_path, monkeypatch, boundary):
    check_recovery(tmp_path, monkeypatch, "remove", boundary)
