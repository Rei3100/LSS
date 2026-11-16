"""A test run cannot resolve the state root into the user's product state or the development records."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from lss import paths
from lss.core import models_store, session

ROOT = Path(__file__).resolve().parents[1]


def test_run_is_in_test_mode_with_its_own_state_folder():
    assert os.environ["LSS_TEST_MODE"] == "1"
    state = session.state_dir()
    assert state == Path(os.environ["LSS_STATE_DIR"])
    assert "lss-test-state-" in state.name


def test_run_has_no_network_and_its_own_model_folders():
    if os.environ.get("LSS_TEST_EXTERNAL") == "1":
        pytest.skip("this run opted in to real models and network")
    state = Path(os.environ["LSS_STATE_DIR"])
    from lss.bgm import scene_models
    from lss.harness import fetch

    assert fetch.model_cache_dir().parent == state
    assert scene_models.model_root().parent == state
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    from lss.bgm import scene_job

    assert not scene_job.available()  # no ordinary test starts a real naming child
    child = subprocess.run([sys.executable, "-c", "import os, urllib.request\n"
                            "try:\n urllib.request.urlopen('http://example.com/', timeout=5)\nexcept OSError:\n print('blocked')\n"
                            "print(os.environ['LSS_MODEL_DIR'])"],
                           capture_output=True, text=True, timeout=60)
    assert child.stdout.splitlines() == ["blocked", os.environ["LSS_MODEL_DIR"]]


def test_protected_root_and_its_children_are_refused(monkeypatch, tmp_path):
    protected = tmp_path / "product" / ".state"
    protected.mkdir(parents=True)
    monkeypatch.setenv("LSS_PROTECTED_STATE_ROOTS", str(protected))
    for target in (protected, protected / "work"):
        monkeypatch.setenv("LSS_STATE_DIR", str(target))
        with pytest.raises(paths.ProtectedStateError):
            session.state_dir()
        with pytest.raises(paths.ProtectedStateError):
            models_store.models_tsv_path()
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "product" / ".state-other"))
    assert session.state_dir() == tmp_path / "product" / ".state-other"


@pytest.mark.skipif(sys.platform != "win32", reason="directory junction")
def test_link_into_a_protected_root_is_refused(monkeypatch, tmp_path):
    protected = tmp_path / "records"
    protected.mkdir()
    link = tmp_path / "link"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(protected)], check=True, capture_output=True)
    monkeypatch.setenv("LSS_PROTECTED_STATE_ROOTS", str(protected))
    monkeypatch.setenv("LSS_STATE_DIR", str(link))
    with pytest.raises(paths.ProtectedStateError):
        session.state_dir()


def test_checkout_state_is_refused_when_the_override_is_dropped(monkeypatch):
    """Without `LSS_STATE_DIR` the default is the checkout's own `.state`; a test run must not land there."""
    monkeypatch.delenv("LSS_STATE_DIR")
    assert str(ROOT / ".state") in os.environ["LSS_PROTECTED_STATE_ROOTS"].split(os.pathsep)
    with pytest.raises(paths.ProtectedStateError):
        session.state_dir()


def test_outside_test_mode_the_path_is_returned_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("LSS_PROTECTED_STATE_ROOTS", str(tmp_path))
    monkeypatch.delenv("LSS_TEST_MODE")
    assert paths.guard_state_root(tmp_path / "x") == tmp_path / "x"


def test_collection_time_reference_to_a_protected_root_is_refused(tmp_path):
    """A test module that resolves the state root while it is being collected is stopped as well.

    The module lives inside the checkout so that the root `conftest.py` is the one that guards it."""
    protected = tmp_path / "records"
    protected.mkdir()
    with tempfile.TemporaryDirectory(prefix=".state-guard-tests-", dir=ROOT) as folder:
        module = Path(folder) / "test_collects.py"
        env = {key: value for key, value in os.environ.items()
               if key not in ("LSS_TEST_MODE", "LSS_TEST_STATE_OWNER", "LSS_STATE_DIR", "LSS_PROTECTED_STATE_ROOTS")}
        env.update(LSS_DEV_STATE=str(protected), LSS_STATE_DIR=str(protected), LSS_TEST_PARENT_WATCH="0",
                   PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(ROOT),
                   "-c", str(ROOT / "pytest.ini"), str(module)]
        # the pre-set LSS_STATE_DIR is replaced by the run's own folder before collection ...
        module.write_text("import os\nfrom lss.core import session\nSTATE = session.state_dir()\n"
                          "assert os.environ['LSS_TEST_MODE'] == '1' and 'lss-test-state-' in STATE.name\n"
                          "\n\ndef test_nothing():\n    pass\n", encoding="utf-8")
        ok = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        assert ok.returncode == 0, ok.stdout[-800:]
        # ... and a module that points the state root at the protected folder during collection is refused
        module.write_text("import os\nfrom lss.core import session\n"
                          "os.environ['LSS_STATE_DIR'] = os.environ['LSS_DEV_STATE']\nSTATE = session.state_dir()\n"
                          "\n\ndef test_nothing():\n    pass\n", encoding="utf-8")
        bad = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        assert bad.returncode != 0 and "ProtectedStateError" in bad.stdout, bad.stdout[-800:]
    assert not any(protected.iterdir())
