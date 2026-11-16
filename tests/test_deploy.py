import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


def _deploy(
    repository: Path, build: Path, deploy_base: Path, *extra: str,
    failure_phase: str | None = None, bgm_receipt: bool = True,
):
    installed = deploy_base / "LSS" / "LSS.exe"
    if "-Finalize" in extra and bgm_receipt and installed.is_file() and (deploy_base / ".LSS.previous").exists():
        # The real receipt comes from scripts/post_deploy_bgm_check.py; these cases test the swap itself.
        (deploy_base / ".LSS-postdeploy-bgm.json").write_text(json.dumps(
            {"status": "PASS", "exe_sha256": hashlib.sha256(installed.read_bytes()).hexdigest()}), encoding="utf-8")
    environment = os.environ.copy()
    if failure_phase is not None:
        environment["LSS_DEPLOY_TEST_FAIL_PHASE"] = failure_phase
    return subprocess.run(
        [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(repository / "deploy-to-freesoft.ps1"),
            "-BuildOutput", str(build), "-DeployBase", str(deploy_base), *extra,
        ],
        cwd=repository, capture_output=True, text=True, encoding="cp932", errors="replace",
        timeout=30, env=environment,
    )


def test_deploy_replaces_runtime_but_preserves_mutable_ledger(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    build = tmp_path / "build" / "LSS"
    target = tmp_path / "portable" / "LSS"
    (build / "_internal").mkdir(parents=True)
    (build / "ledger").mkdir()
    (build / "LSS.exe").write_bytes(b"new exe")
    (build / "_internal" / "new-runtime.dll").write_bytes(b"new")
    (build / "ledger" / "models.tsv").write_text("test ledger", encoding="utf-8")
    (target / "_internal").mkdir(parents=True)
    (target / "ledger").mkdir()
    (target / "LSS.exe").write_bytes(b"old exe")
    (target / "_internal" / "obsolete.dll").write_bytes(b"old")
    (target / "ledger" / "models.tsv").write_text("runtime metrics", encoding="utf-8")
    (target / "user-local.txt").write_text("preserve me", encoding="utf-8")

    result = _deploy(repository, build, tmp_path / "portable")

    assert result.returncode == 0, result.stderr
    assert (target / "LSS.exe").read_bytes() == b"new exe"
    assert (target / "_internal" / "new-runtime.dll").read_bytes() == b"new"
    assert not (target / "_internal" / "obsolete.dll").exists()
    assert (target / "ledger" / "models.tsv").read_text(encoding="utf-8") == "runtime metrics"
    assert (target / "user-local.txt").read_text(encoding="utf-8") == "preserve me"


def test_deploy_failure_after_stage_leaves_known_good_runtime_untouched(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    build = tmp_path / "build" / "LSS"
    target = tmp_path / "portable" / "LSS"
    (build / "_internal").mkdir(parents=True)
    (build / "LSS.exe").write_bytes(b"new exe")
    (build / "_internal" / "runtime.dll").write_bytes(b"new runtime")
    (target / "_internal").mkdir(parents=True)
    (target / "LSS.exe").write_bytes(b"old exe")
    (target / "_internal" / "runtime.dll").write_bytes(b"old runtime")

    result = _deploy(
        repository, build, tmp_path / "portable", failure_phase="after_stage",
    )

    assert result.returncode != 0
    assert (target / "LSS.exe").read_bytes() == b"old exe"
    assert (target / "_internal" / "runtime.dll").read_bytes() == b"old runtime"
    assert not (tmp_path / "portable" / ".LSS.staging").exists()
    assert not (tmp_path / "portable" / ".LSS-deploy.json").exists()


def test_deploy_failure_after_promote_rolls_back_runtime_and_user_data(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    build = tmp_path / "build" / "LSS"
    target = tmp_path / "portable" / "LSS"
    (build / "_internal").mkdir(parents=True)
    (build / "LSS.exe").write_bytes(b"new exe")
    (build / "_internal" / "runtime.dll").write_bytes(b"new runtime")
    (target / "_internal").mkdir(parents=True)
    (target / "ledger").mkdir()
    (target / "LSS.exe").write_bytes(b"old exe")
    (target / "_internal" / "runtime.dll").write_bytes(b"old runtime")
    (target / "ledger" / "models.tsv").write_text("metrics", encoding="utf-8")

    result = _deploy(
        repository, build, tmp_path / "portable", failure_phase="after_promote",
    )

    assert result.returncode != 0
    assert (target / "LSS.exe").read_bytes() == b"old exe"
    assert (target / "_internal" / "runtime.dll").read_bytes() == b"old runtime"
    assert (target / "ledger" / "models.tsv").read_text(encoding="utf-8") == "metrics"
    assert not (tmp_path / "portable" / ".LSS.previous").exists()


def test_next_deploy_recovers_interrupted_swap_before_promoting_new_build(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    deploy_base = tmp_path / "portable"
    target = deploy_base / "LSS"
    previous = deploy_base / ".LSS.previous"
    stage = deploy_base / ".LSS.staging"
    build = tmp_path / "build" / "LSS"
    for root, exe in ((target, b"interrupted new"), (previous, b"known good"), (build, b"next build")):
        (root / "_internal").mkdir(parents=True)
        (root / "LSS.exe").write_bytes(exe)
        (root / "_internal" / "runtime.dll").write_bytes(exe)
    (target / "ledger").mkdir()
    (target / "ledger" / "models.tsv").write_text("preserved metrics", encoding="utf-8")
    stage.mkdir(exist_ok=True)
    (stage / "partial").write_bytes(b"partial")
    journal = {
        "version": 1,
        "phase": "target_promoted",
        "target": str(target.resolve()),
        "staging": str(stage.resolve()),
        "previous": str(previous.resolve()),
        "runtime_entries": ["LSS.exe", "_internal"],
    }
    (deploy_base / ".LSS-deploy.json").write_text(json.dumps(journal), encoding="utf-8")

    result = _deploy(repository, build, deploy_base)

    assert result.returncode == 0, result.stderr
    assert (target / "LSS.exe").read_bytes() == b"next build"
    assert (target / "ledger" / "models.tsv").read_text(encoding="utf-8") == "preserved metrics"
    assert not previous.exists() and not stage.exists()
    assert not (deploy_base / ".LSS-deploy.json").exists()


def test_deploy_rejects_target_name_path_traversal(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    build = tmp_path / "build"
    build.mkdir()

    result = _deploy(
        repository, build, tmp_path / "portable", "-TargetName", "..\\outside",
    )

    assert result.returncode != 0
    assert not (tmp_path / "outside").exists()


def test_deploy_default_build_output_is_resolved_from_script_location(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    script = tmp_path / "tool" / "deploy-to-freesoft.ps1"
    script.parent.mkdir()
    script.write_text(
        (repository / "deploy-to-freesoft.ps1").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    build = script.parent / "dist" / "LSS"
    build.mkdir(parents=True)
    (build / "LSS.exe").write_bytes(b"default build")

    result = subprocess.run(
        [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
            "-DeployBase", str(tmp_path / "portable"),
        ],
        capture_output=True, text=True, encoding="cp932", errors="replace", timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "portable" / "LSS" / "LSS.exe").read_bytes() == b"default build"


def test_deploy_rejects_build_tree_overlapping_target(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    deploy_base = tmp_path / "portable"
    target = deploy_base / "LSS"
    (target / "_internal").mkdir(parents=True)
    runtime = target / "_internal" / "runtime.dll"
    runtime.write_bytes(b"must survive")
    (target / "LSS.exe").write_bytes(b"same tree")

    result = _deploy(repository, target, deploy_base)

    assert result.returncode != 0
    assert runtime.read_bytes() == b"must survive"


def test_deploy_rejects_build_without_executable(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    build = tmp_path / "build" / "LSS"
    (build / "_internal").mkdir(parents=True)

    result = _deploy(repository, build, tmp_path / "portable")

    assert result.returncode != 0
    assert not (tmp_path / "portable" / "LSS").exists()


def test_deploy_stops_before_swap_when_new_runtime_collides_with_unknown_user_file(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    build = tmp_path / "build" / "LSS"
    target = tmp_path / "portable" / "LSS"
    build.mkdir(parents=True)
    target.mkdir(parents=True)
    (build / "LSS.exe").write_bytes(b"new exe")
    (build / "notes.dat").write_bytes(b"new runtime file")
    (target / "LSS.exe").write_bytes(b"old exe")
    (target / "notes.dat").write_bytes(b"user data")

    result = _deploy(repository, build, tmp_path / "portable")

    assert result.returncode != 0
    assert (target / "LSS.exe").read_bytes() == b"old exe"
    assert (target / "notes.dat").read_bytes() == b"user data"
    assert not (tmp_path / "portable" / ".LSS.previous").exists()


def _directory_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True, text=True, encoding="cp932", errors="replace",
        )
        if result.returncode:
            pytest.skip(f"directory reparse pointを作成できない環境です: {result.stderr}")


def test_deploy_rejects_reparse_target_without_touching_link_destination(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    build = tmp_path / "build" / "LSS"
    deploy_base = tmp_path / "portable"
    outside = tmp_path / "outside"
    build.mkdir(parents=True)
    deploy_base.mkdir()
    outside.mkdir()
    (build / "LSS.exe").write_bytes(b"new exe")
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    _directory_symlink(deploy_base / "LSS", outside)

    result = _deploy(repository, build, deploy_base)

    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_deploy_rejects_reparse_build_root(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    real_build = tmp_path / "real-build"
    real_build.mkdir()
    (real_build / "LSS.exe").write_bytes(b"new exe")
    linked_build = tmp_path / "linked-build"
    _directory_symlink(linked_build, real_build)

    result = _deploy(repository, linked_build, tmp_path / "portable")

    assert result.returncode != 0
    assert not (tmp_path / "portable" / "LSS").exists()


def _pending_case(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    base = tmp_path / "portable"
    build = tmp_path / "build" / "LSS"
    target = base / "LSS"
    for root, exe in ((build, b"candidate"), (target, b"known good")):
        (root / "_internal").mkdir(parents=True)
        (root / "LSS.exe").write_bytes(exe)
        (root / "_internal" / "runtime.dll").write_bytes(exe)
    (target / "ledger").mkdir()
    (target / "ledger" / "models.tsv").write_text("user metrics", encoding="utf-8")
    result = _deploy(repository, build, base, "-KeepPrevious")
    assert result.returncode == 0, result.stderr
    return repository, base, build, target


def test_a_deploy_awaiting_acceptance_keeps_the_known_good_runtime(tmp_path):
    _, base, _, target = _pending_case(tmp_path)
    assert (target / "LSS.exe").read_bytes() == b"candidate"
    assert (target / "ledger" / "models.tsv").read_text(encoding="utf-8") == "user metrics"
    assert (base / ".LSS.previous" / "LSS.exe").read_bytes() == b"known good"
    journal = json.loads((base / ".LSS-deploy.json").read_text(encoding="utf-8"))
    assert journal["phase"] == "awaiting_acceptance"


def test_a_failed_acceptance_rolls_back_to_the_known_good_runtime(tmp_path):
    repository, base, build, target = _pending_case(tmp_path)
    # 受入待ちの間に増えた可変データも、戻した配置へ持ち越す。
    (target / "ledger" / "added.tsv").write_text("added while pending", encoding="utf-8")
    result = _deploy(repository, build, base, "-Rollback")
    assert result.returncode == 0, result.stderr
    assert (target / "LSS.exe").read_bytes() == b"known good"
    assert (target / "_internal" / "runtime.dll").read_bytes() == b"known good"
    assert (target / "ledger" / "models.tsv").read_text(encoding="utf-8") == "user metrics"
    assert (target / "ledger" / "added.tsv").read_text(encoding="utf-8") == "added while pending"
    assert not (base / ".LSS.previous").exists()
    assert not (base / ".LSS-deploy.json").exists()


def test_a_passed_acceptance_finalizes_the_candidate(tmp_path):
    repository, base, build, target = _pending_case(tmp_path)
    result = _deploy(repository, build, base, "-Finalize")
    assert result.returncode == 0, result.stderr
    assert (target / "LSS.exe").read_bytes() == b"candidate"
    assert (target / "ledger" / "models.tsv").read_text(encoding="utf-8") == "user metrics"
    assert not (base / ".LSS.previous").exists()
    assert not (base / ".LSS-deploy.json").exists()


def test_finalize_without_a_pending_deploy_is_refused(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    base = tmp_path / "portable"
    (base / "LSS").mkdir(parents=True)
    (base / "LSS" / "LSS.exe").write_bytes(b"live")
    result = _deploy(repository, tmp_path / "missing-build", base, "-Finalize")
    assert result.returncode != 0
    assert (base / "LSS" / "LSS.exe").read_bytes() == b"live"
    assert sorted(path.name for path in base.iterdir()) == ["LSS"]


def test_an_unconfirmed_deploy_is_rolled_back_before_the_next_one(tmp_path):
    repository, base, _, target = _pending_case(tmp_path)
    next_build = tmp_path / "next" / "LSS"
    (next_build / "_internal").mkdir(parents=True)
    (next_build / "LSS.exe").write_bytes(b"next")
    (next_build / "_internal" / "runtime.dll").write_bytes(b"next")
    result = _deploy(repository, next_build, base)
    assert result.returncode == 0, result.stderr
    assert (target / "LSS.exe").read_bytes() == b"next"
    assert (target / "ledger" / "models.tsv").read_text(encoding="utf-8") == "user metrics"
    assert not (base / ".LSS.previous").exists()


def _junction_case(tmp_path):
    """可変エントリ内のjunctionを持つ既存配置（models直下との.state配下）を作る。"""
    repository = Path(__file__).resolve().parents[1]
    base = tmp_path / "portable"
    build = tmp_path / "build" / "LSS"
    target = base / "LSS"
    for root, exe in ((build, b"candidate"), (target, b"known good")):
        (root / "_internal").mkdir(parents=True)
        (root / "LSS.exe").write_bytes(exe)
        (root / "_internal" / "runtime.dll").write_bytes(exe)
    (target / ".state").mkdir()
    targets = {}
    for name, link in (("models", target / "models"), ("work", target / ".state" / "work")):
        destination = tmp_path / "tier" / name
        destination.mkdir(parents=True)
        (destination / "sentinel.bin").write_bytes(b"sentinel " + name.encode())
        _directory_symlink(link, destination)
        targets[name] = (link, destination)
    stamp = {
        name: (dest / "sentinel.bin").stat().st_mtime_ns for name, (_, dest) in targets.items()
    }
    return repository, base, build, target, targets, stamp


def _assert_links_intact(targets, stamp):
    for name, (link, destination) in targets.items():
        assert link.lstat().st_file_attributes & 0x400  # FILE_ATTRIBUTE_REPARSE_POINT
        assert Path(os.path.realpath(link)) == destination.resolve()
        sentinel = destination / "sentinel.bin"
        assert sentinel.read_bytes() == b"sentinel " + name.encode()
        assert sentinel.stat().st_mtime_ns == stamp[name]
        assert (link / "sentinel.bin").read_bytes() == sentinel.read_bytes()


def test_junctions_inside_preserved_entries_survive_keep_rollback_and_finalize(tmp_path):
    repository, base, build, target, targets, stamp = _junction_case(tmp_path)

    result = _deploy(repository, build, base, "-KeepPrevious")
    assert result.returncode == 0, result.stderr
    assert (target / "LSS.exe").read_bytes() == b"candidate"
    _assert_links_intact(targets, stamp)

    result = _deploy(repository, build, base, "-Rollback")
    assert result.returncode == 0, result.stderr
    assert (target / "LSS.exe").read_bytes() == b"known good"
    _assert_links_intact(targets, stamp)

    result = _deploy(repository, build, base, "-KeepPrevious")
    assert result.returncode == 0, result.stderr
    result = _deploy(repository, build, base, "-Finalize")
    assert result.returncode == 0, result.stderr
    assert (target / "LSS.exe").read_bytes() == b"candidate"
    assert not (base / ".LSS.previous").exists()
    _assert_links_intact(targets, stamp)


def test_junction_inside_a_runtime_entry_is_still_refused(tmp_path):
    repository, base, build, target, targets, stamp = _junction_case(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("keep", encoding="utf-8")
    _directory_symlink(target / "_internal" / "linked", outside)

    result = _deploy(repository, build, base, "-KeepPrevious")

    assert result.returncode != 0
    assert (target / "LSS.exe").read_bytes() == b"known good"
    assert (outside / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    _assert_links_intact(targets, stamp)


def test_finalize_refuses_without_a_passing_bgm_receipt_for_the_installed_exe(tmp_path):
    repository, base, build, target = _pending_case(tmp_path)
    refused = _deploy(repository, build, base, "-Finalize", bgm_receipt=False)
    assert refused.returncode != 0 and (base / ".LSS.previous").is_dir()
    (base / ".LSS-postdeploy-bgm.json").write_text(json.dumps(
        {"status": "PASS", "exe_sha256": hashlib.sha256(b"known good").hexdigest()}), encoding="utf-8")
    assert _deploy(repository, build, base, "-Finalize", bgm_receipt=False).returncode != 0
    assert (base / ".LSS.previous").is_dir()
    assert _deploy(repository, build, base, "-Finalize").returncode == 0
    assert (target / "LSS.exe").read_bytes() == b"candidate" and not (base / ".LSS.previous").exists()
