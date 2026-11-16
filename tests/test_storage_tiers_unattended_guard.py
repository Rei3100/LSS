import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import storage_tiers  # noqa: E402


def test_scheduled_pass_refuses_uncommitted_storage_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(storage_tiers, "uncommitted_storage_code", lambda: ["scripts/storage_tiers.py"])
    monkeypatch.setattr(storage_tiers, "local_state", lambda policy: (_ for _ in ()).throw(AssertionError("acted")))
    assert storage_tiers.scheduled({}, tmp_path) == 0
    assert "unattended run refused" in capsys.readouterr().out


def test_dirty_storage_script_is_detected(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "storage_x.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"], check=True)
    assert storage_tiers.uncommitted_storage_code(tmp_path) == []
    (tmp_path / "scripts" / "storage_x.py").write_text("x = 2\n")
    assert storage_tiers.uncommitted_storage_code(tmp_path) == ["scripts/storage_x.py"]
