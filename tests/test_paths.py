from lss.paths import app_root, repo_root, resource_root


def test_repo_root_points_to_repository_root_in_dev_mode():
    root = repo_root()

    assert (root / "VERSION").exists()
    assert (root / "lss").is_dir()


def test_frozen_paths_separate_app_data_from_bundled_resources(monkeypatch, tmp_path):
    app_dir = tmp_path / "LSS"
    internal_dir = app_dir / "_internal"
    internal_dir.mkdir(parents=True)
    fake_exe = app_dir / "LSS.exe"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys.executable", str(fake_exe))
    monkeypatch.setattr("sys._MEIPASS", str(internal_dir), raising=False)

    assert app_root() == app_dir
    assert repo_root() == app_dir
    assert resource_root() == internal_dir


def test_frozen_resource_root_falls_back_to_internal_dir(monkeypatch, tmp_path):
    fake_exe = tmp_path / "LSS.exe"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys.executable", str(fake_exe))
    monkeypatch.delattr("sys._MEIPASS", raising=False)

    assert resource_root() == tmp_path / "_internal"
