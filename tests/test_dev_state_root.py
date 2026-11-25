"""Development state stays isolated while historical Evidence is read-only in tests."""
from pathlib import Path

import pytest

from scripts import dev_state


def test_explicit_directory_wins_over_checkout_and_product_state(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    state = tmp_path / "development"
    state.mkdir()
    monkeypatch.setattr(dev_state, "CHECKOUT_ROOT", checkout)
    monkeypatch.setenv("LSS_DEV_STATE", str(state))
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "product"))
    assert dev_state.dev_state_root() == state.resolve()
    assert dev_state.dev_state_root(checkout) == state.resolve()
    assert not (checkout / ".state").exists()
    assert not (tmp_path / "product").exists()


def test_unset_uses_checkout_state_without_creating_it(tmp_path, monkeypatch):
    monkeypatch.delenv("LSS_DEV_STATE", raising=False)
    monkeypatch.setattr(dev_state, "CHECKOUT_ROOT", tmp_path)
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "product"))
    assert dev_state.dev_state_root() == tmp_path.resolve() / ".state"
    assert not (tmp_path / ".state").exists()


def test_unset_preserves_explicit_checkout_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("LSS_DEV_STATE", raising=False)
    monkeypatch.setattr(dev_state, "CHECKOUT_ROOT", tmp_path / "runner")
    checkout = tmp_path / "other-checkout"
    assert dev_state.dev_state_root(checkout) == checkout.resolve() / ".state"
    assert not checkout.exists()


@pytest.mark.parametrize("value", ["", ".", "relative-state"])
def test_relative_or_empty_configuration_is_rejected(tmp_path, monkeypatch, value):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dev_state, "CHECKOUT_ROOT", tmp_path)
    (tmp_path / ".state").mkdir()
    (tmp_path / "relative-state").mkdir()
    monkeypatch.setenv("LSS_DEV_STATE", value)
    with pytest.raises(ValueError, match="LSS_DEV_STATE.*absolute"):
        dev_state.dev_state_root()


def test_missing_directory_is_rejected_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_state, "CHECKOUT_ROOT", tmp_path)
    (tmp_path / ".state").mkdir()
    missing = tmp_path / "missing"
    monkeypatch.setenv("LSS_DEV_STATE", str(missing))
    with pytest.raises(ValueError, match="LSS_DEV_STATE.*existing directory"):
        dev_state.dev_state_root()
    assert not missing.exists()


def test_regular_file_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_state, "CHECKOUT_ROOT", tmp_path)
    path = tmp_path / "file"
    path.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("LSS_DEV_STATE", str(path))
    with pytest.raises(ValueError, match="LSS_DEV_STATE.*existing directory"):
        dev_state.dev_state_root()


def test_external_validation_keeps_dependency_and_product_paths(tmp_path, monkeypatch):
    from scripts import disk_cleanup as disk

    checkout = tmp_path / "checkout"
    product = checkout / ".state"
    (product / "validation").mkdir(parents=True)
    (product / "queue.json").write_text("{}", encoding="utf-8")
    state = tmp_path / "development"
    case = state / "validation" / "case"
    case.mkdir(parents=True)
    (case / "music.wav").write_bytes(b"fixture")
    dependency = state / "validation" / "active-plan.json"
    dependency.write_text('{"input": ".state/validation/case/music.wav"}', encoding="utf-8")
    monkeypatch.setenv("LSS_DEV_STATE", str(state))
    monkeypatch.setattr(disk, "ROOT", checkout)
    monkeypatch.setattr(disk, "tracked_files", lambda prefix=".state": [])
    monkeypatch.syspath_prepend(str(dev_state.CHECKOUT_ROOT / "scripts"))

    rows, unreadable = disk.validation_media_candidates()
    assert unreadable == []
    assert len(rows) == 1
    assert rows[0]["path"] == (case / "music.wav").as_posix()
    assert rows[0]["class"] == "KEEP"
    assert "dependency reference" in rows[0]["reason"]
    assert product / "queue.json" in disk.reference_sources()
    assert (product / "queue.json").read_text(encoding="utf-8") == "{}"
    assert disk._prunable_roots() == (state / "validation", checkout / ".providers", product / "work")


def test_external_state_preserves_git_tracked_file_protection(tmp_path, monkeypatch):
    from scripts import disk_cleanup as disk

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    state = tmp_path / "development"
    state.mkdir()
    monkeypatch.setenv("LSS_DEV_STATE", str(state))
    monkeypatch.setattr(disk, "ROOT", checkout)
    calls = []

    def tracked(command, **kwargs):
        calls.append(command)
        return ".state/validation/local.json\n" if command[2] == str(checkout) else "validation/shared.json\n"

    monkeypatch.setattr(disk.subprocess, "check_output", tracked)
    assert disk.tracked_files(".state/validation") == [
        checkout / ".state/validation/local.json", state / "validation/shared.json",
    ]
    assert calls[1] == ["git", "-C", str(state), "ls-files", "validation"]


def test_explicit_release_state_never_falls_back_to_another_checkout(tmp_path, monkeypatch):
    from scripts import state_installed, state_status

    state = tmp_path / "development"
    state.mkdir()
    checkout = tmp_path / "checkout"
    monkeypatch.setenv("LSS_DEV_STATE", str(state))
    monkeypatch.setattr(state_installed, "ROOT", checkout)

    def refuse_git(*args, **kwargs):
        raise AssertionError("Explicit development state must not discover another checkout")

    monkeypatch.setattr(state_installed.subprocess, "run", refuse_git)
    expected = state / "releases/current-product.json"
    assert state_installed._release_file() == expected
    assert state_status.release_file(checkout) == expected


def test_deletion_is_refused_when_the_state_belongs_to_another_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / ".state").mkdir(parents=True)
    external = tmp_path / "other-checkout-state"
    external.mkdir()
    monkeypatch.setenv("LSS_DEV_STATE", str(external))
    with pytest.raises(SystemExit, match="owns them"):
        dev_state.require_owned_for_deletion(checkout)
    monkeypatch.setenv("LSS_DEV_STATE", str(checkout / ".state"))
    assert dev_state.require_owned_for_deletion(checkout) == (checkout / ".state").resolve()
    monkeypatch.delenv("LSS_DEV_STATE")
    assert dev_state.require_owned_for_deletion(checkout) == (checkout / ".state").resolve()


def test_cleanup_entry_points_refuse_external_state_before_surveying(tmp_path, monkeypatch):
    from scripts import disk_cleanup, storage_retention
    from tests.machine_config_fixtures import machine_policy
    policy = machine_policy(tmp_path)
    monkeypatch.setattr(storage_retention.tiers, "load_policy", lambda: policy)
    external = tmp_path / "other-checkout-state"
    external.mkdir()
    monkeypatch.setenv("LSS_DEV_STATE", str(external))
    monkeypatch.setattr(disk_cleanup, "survey", lambda: pytest.fail("surveyed before refusing"))
    with pytest.raises(SystemExit, match="owns them"):
        disk_cleanup.prune(True, limit=1)
    with pytest.raises(SystemExit, match="owns them"):
        disk_cleanup.retire(True)
    with pytest.raises(SystemExit, match="owns them"):
        storage_retention.validation_gc(policy, disk_cleanup.ROOT, apply=True)


def test_deletion_refuses_case_only_state_redirection(tmp_path, monkeypatch):
    checkout = tmp_path.resolve()
    state = checkout / ".state"
    state.mkdir()
    monkeypatch.delenv("LSS_DEV_STATE", raising=False)
    resolve = Path.resolve

    def resolve_redirected(path, *args, **kwargs):
        if str(path) == str(state):
            return checkout / ".STATE"
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_redirected)
    with pytest.raises(SystemExit, match="owns them"):
        dev_state.require_owned_for_deletion(checkout)


def test_a_protected_development_state_is_not_resolved_in_test_mode(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    real = tmp_path / "real-records"
    real.mkdir()
    monkeypatch.setattr(dev_state, "CHECKOUT_ROOT", checkout)
    monkeypatch.setenv("LSS_DEV_STATE", str(real))
    monkeypatch.setenv("LSS_TEST_MODE", "1")
    monkeypatch.setenv("LSS_PROTECTED_STATE_ROOTS", str(real))
    assert dev_state.dev_state_root() == checkout.resolve() / ".state"
    monkeypatch.setenv("LSS_PROTECTED_STATE_ROOTS", str(tmp_path / "unrelated"))
    assert dev_state.dev_state_root() == real.resolve()


@pytest.mark.parametrize("protected", [None, "", " "])
def test_test_mode_refuses_development_state_without_protected_roots(tmp_path, monkeypatch, protected):
    checkout = tmp_path / "checkout"
    state = checkout / ".state"
    state.mkdir(parents=True)
    monkeypatch.setenv("LSS_DEV_STATE", str(state))
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "test-state"))
    if protected is None:
        monkeypatch.delenv("LSS_PROTECTED_STATE_ROOTS", raising=False)
    else:
        monkeypatch.setenv("LSS_PROTECTED_STATE_ROOTS", protected)

    with pytest.raises(PermissionError, match="protected development state"):
        dev_state.dev_state_root(checkout)


@pytest.mark.parametrize("configured", [False, True])
def test_test_mode_never_falls_back_to_a_protected_checkout_state(tmp_path, monkeypatch, configured):
    checkout = tmp_path / "checkout"
    state = checkout / ".state"
    state.mkdir(parents=True)
    test_state = tmp_path / "test-state"
    test_state.mkdir()
    monkeypatch.setenv("LSS_STATE_DIR", str(test_state))
    monkeypatch.setenv("LSS_PROTECTED_STATE_ROOTS", str(state))
    if configured:
        monkeypatch.setenv("LSS_DEV_STATE", str(state))
    else:
        monkeypatch.delenv("LSS_DEV_STATE", raising=False)

    isolated = dev_state.dev_state_root(checkout)
    assert isolated.is_relative_to(test_state.resolve())
    assert not isolated.exists()
    with pytest.raises(SystemExit, match="owns them"):
        dev_state.require_owned_for_deletion(checkout)


def test_evidence_home_uses_configured_state_parent(tmp_path, monkeypatch):
    home = tmp_path / "evidence-home"
    state = home / ".state"
    state.mkdir(parents=True)
    monkeypatch.setenv("LSS_DEV_STATE", str(state))
    monkeypatch.setattr(dev_state, "CHECKOUT_ROOT", tmp_path / "checkout")
    assert dev_state.dev_evidence_root() == home
    assert dev_state.dev_evidence_root(tmp_path / "other-checkout") == home


def test_evidence_home_defaults_to_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.delenv("LSS_DEV_STATE", raising=False)
    monkeypatch.setattr(dev_state, "CHECKOUT_ROOT", checkout)
    assert dev_state.dev_evidence_root() == checkout.resolve()
    assert dev_state.dev_evidence_root(tmp_path) == tmp_path.resolve()
    assert not (checkout / ".state").exists()


def test_missing_evidence_home_is_rejected_without_creation(tmp_path, monkeypatch):
    home = tmp_path / "missing-home"
    monkeypatch.setenv("LSS_DEV_STATE", str(home / ".state"))
    with pytest.raises(ValueError, match="Evidence home.*existing directory"):
        dev_state.dev_evidence_root(tmp_path)
    assert not home.exists()


def test_evidence_home_must_be_a_directory(tmp_path, monkeypatch):
    home = tmp_path / "file"
    home.write_text("fixture", encoding="utf-8")
    monkeypatch.setenv("LSS_DEV_STATE", str(home / ".state"))
    with pytest.raises(ValueError, match="Evidence home.*existing directory"):
        dev_state.dev_evidence_root()


@pytest.mark.parametrize("value", ["", ".", "relative-state"])
def test_evidence_home_rejects_relative_configuration(monkeypatch, value):
    monkeypatch.setenv("LSS_DEV_STATE", value)
    with pytest.raises(ValueError, match="LSS_DEV_STATE.*absolute"):
        dev_state.dev_evidence_root()


def test_evidence_home_does_not_resolve_configured_state(tmp_path, monkeypatch):
    home = tmp_path / "evidence-home"
    home.mkdir()
    monkeypatch.setenv("LSS_DEV_STATE", str(home / ".state"))

    def refuse_resolve(*args, **kwargs):
        pytest.fail("Evidence home must use the configured parent before resolving links")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", refuse_resolve)
        assert dev_state.dev_evidence_root() == home


def _directory_link(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")


def test_deletion_refuses_an_unconfigured_checkout_state_link(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _directory_link(checkout / ".state", outside)
    monkeypatch.delenv("LSS_DEV_STATE", raising=False)

    with pytest.raises(SystemExit, match="owns them"):
        dev_state.require_owned_for_deletion(checkout)


def test_state_link_keeps_evidence_beside_the_link(tmp_path, monkeypatch):
    home = tmp_path / "evidence-home"
    home.mkdir()
    moved = tmp_path / "moved-state"
    moved.mkdir()
    state = home / ".state"
    _directory_link(state, moved)
    monkeypatch.setenv("LSS_DEV_STATE", str(state))
    monkeypatch.setenv("LSS_TEST_MODE", "0")
    assert dev_state.dev_state_root() == moved.resolve()
    assert dev_state.dev_evidence_root() == home


def test_test_mode_reads_real_evidence_despite_protected_state(tmp_path, monkeypatch):
    state = tmp_path / "evidence-home" / ".state"
    state.mkdir(parents=True)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(dev_state, "CHECKOUT_ROOT", checkout)
    monkeypatch.setenv("LSS_DEV_STATE", str(state))
    monkeypatch.setenv("LSS_TEST_MODE", "1")
    monkeypatch.setenv("LSS_PROTECTED_STATE_ROOTS", str(state))
    assert dev_state.dev_state_root() == checkout.resolve() / ".state"
    assert dev_state.dev_evidence_root() == state.parent


@pytest.mark.parametrize("folder", ["release-evidence", "scene-web-evidence", "ep001-truth",
                                    "efficiency-evidence", "ledger", "boundary-evidence"])
def test_evidence_writes_are_refused_in_test_mode(tmp_path, monkeypatch, folder):
    home = tmp_path / "evidence-home"
    state = home / ".state"
    state.mkdir(parents=True)
    monkeypatch.setenv("LSS_DEV_STATE", str(state))
    monkeypatch.setenv("LSS_TEST_MODE", "1")
    monkeypatch.setenv("LSS_PROTECTED_STATE_ROOTS", str(state))
    target = home / folder / "new" / "receipt.json"
    with pytest.raises(PermissionError, match="only read.*Evidence home"):
        dev_state.require_evidence_writable(target)
    assert not target.parent.exists()
    assert dev_state.require_evidence_writable(tmp_path / "temporary" / folder / "receipt.json") is None


def test_test_mode_protects_checkout_evidence_without_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("LSS_DEV_STATE", raising=False)
    monkeypatch.setattr(dev_state, "CHECKOUT_ROOT", tmp_path)
    monkeypatch.setenv("LSS_TEST_MODE", "1")
    with pytest.raises(PermissionError):
        dev_state.require_evidence_writable(tmp_path / "release-evidence" / "receipt.json")
    with pytest.raises(PermissionError):
        dev_state.require_evidence_writable(tmp_path)


def test_evidence_write_guard_checks_relative_paths(tmp_path, monkeypatch):
    home = tmp_path / "evidence-home"
    home.mkdir()
    monkeypatch.setenv("LSS_DEV_STATE", str(home / ".state"))
    monkeypatch.setenv("LSS_TEST_MODE", "1")
    monkeypatch.chdir(home)
    with pytest.raises(PermissionError):
        dev_state.require_evidence_writable(Path("release-evidence/receipt.json"))
    assert dev_state.require_evidence_writable(Path("../temporary/receipt.json")) is None


def test_evidence_write_guard_checks_links_in_both_directions(tmp_path, monkeypatch):
    home = tmp_path / "evidence-home"
    home.mkdir()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    inward = tmp_path / "alias"
    outward = home / "release-evidence"
    _directory_link(inward, home)
    _directory_link(outward, temporary)
    monkeypatch.setenv("LSS_DEV_STATE", str(home / ".state"))
    monkeypatch.setenv("LSS_TEST_MODE", "1")
    for target in (inward / "ledger/receipt.json", outward / "receipt.json"):
        with pytest.raises(PermissionError):
            dev_state.require_evidence_writable(target)


@pytest.mark.parametrize("mode", [None, "0", "true"])
def test_evidence_writes_are_allowed_outside_test_mode(tmp_path, monkeypatch, mode):
    home = tmp_path / "evidence-home"
    home.mkdir()
    monkeypatch.setenv("LSS_DEV_STATE", str(home / ".state"))
    if mode is None:
        monkeypatch.delenv("LSS_TEST_MODE", raising=False)
    else:
        monkeypatch.setenv("LSS_TEST_MODE", mode)
    assert dev_state.require_evidence_writable(home / "release-evidence/receipt.json") is None


def test_receipt_writer_preserves_real_evidence_and_allows_temporary_output(tmp_path, monkeypatch):
    from scripts import acquire_audio_probe

    state = tmp_path / "evidence-home" / ".state"
    state.mkdir(parents=True)
    receipt = state.parent / "release-evidence" / "receipt.json"
    receipt.parent.mkdir()
    receipt.write_text("historical receipt", encoding="utf-8")
    monkeypatch.setenv("LSS_DEV_STATE", str(state))
    monkeypatch.setenv("LSS_TEST_MODE", "1")
    with pytest.raises(PermissionError):
        acquire_audio_probe.save(receipt, {"status": "changed"})
    assert receipt.read_text(encoding="utf-8") == "historical receipt"
    assert list(receipt.parent.iterdir()) == [receipt]
    temporary = tmp_path / "temporary.json"
    acquire_audio_probe.save(temporary, {"status": "fixture"})
    assert '"fixture"' in temporary.read_text(encoding="utf-8")
