"""Resolve development state and read-only test Evidence independently of Product state."""
from __future__ import annotations

import os
from pathlib import Path

CHECKOUT_ROOT = Path(__file__).resolve().parents[1]


def dev_state_root(checkout_root: Path | None = None) -> Path:
    """Use explicit development state, or the caller's legacy checkout state.

    Callers that already accept a checkout root may supply it for the legacy
    fallback. No directory is created, and explicit configuration never falls
    back when it is invalid.
    """
    value = os.environ.get("LSS_DEV_STATE")
    if value is not None and _protected_in_test(value):
        # Tests never resolve the real development records (conftest protects them);
        # they get the (test-replaced) checkout root instead.
        value = None
    if value is None:
        root = CHECKOUT_ROOT if checkout_root is None else Path(checkout_root)
        return root.resolve() / ".state"
    root = Path(value)
    if not root.is_absolute():
        raise ValueError("LSS_DEV_STATE must be an absolute path to an existing directory")
    if not root.is_dir():
        raise ValueError("LSS_DEV_STATE must point to an existing directory")
    return root.resolve()


def dev_evidence_root(checkout_root: Path | None = None) -> Path:
    """Return the existing Evidence home beside the configured development state.

    Use the configured path's parent before resolving links: moving `.state`
    behind a junction must not move its sibling Evidence. Tests read the same
    historical Evidence even when development state itself is protected. With
    no configuration, use the caller's checkout (or this checkout).
    """
    value = os.environ.get("LSS_DEV_STATE")
    if value is None:
        root = (CHECKOUT_ROOT if checkout_root is None else Path(checkout_root)).resolve()
    else:
        state = Path(value)
        if not state.is_absolute():
            raise ValueError("LSS_DEV_STATE must be an absolute path")
        root = state.parent
    if not root.is_dir():
        raise ValueError("Development Evidence home must be an existing directory")
    return root


def require_evidence_writable(path: Path) -> None:
    """Refuse test mutations inside the real Evidence home, including via links.

    Reading historical Evidence is allowed. Call before creating, changing,
    moving or deleting an Evidence output; temporary outputs outside the real
    home remain writable. Outside test mode this helper does nothing.
    """
    if os.environ.get("LSS_TEST_MODE") != "1":
        return
    home = dev_evidence_root()
    target = Path(path)
    # Check both spellings: a link out of the home is still a protected entry,
    # and a link into the home must not let a temporary-looking path mutate it.
    lexical_home = Path(os.path.abspath(home))
    lexical_target = Path(os.path.abspath(target))
    if (lexical_target.is_relative_to(lexical_home)
            or target.resolve().is_relative_to(home.resolve())):
        raise PermissionError("Tests may only read the real development Evidence home")


def require_owned_for_deletion(checkout_root: Path | None = None) -> Path:
    """Refuse deleting development records from a checkout that does not own them.

    Deletion decides "unreferenced" from the running checkout's own files; records
    kept by another checkout may be referenced only from there, and an unseen
    reference is not an absent one. Deleting is therefore allowed only when the
    development state is this checkout's own `.state`.
    """
    root = (CHECKOUT_ROOT if checkout_root is None else Path(checkout_root)).resolve()
    state = dev_state_root(root)
    if state != root / ".state":
        raise SystemExit("Deleting development records runs only from the checkout that owns them")
    return state


def _protected_in_test(value: str) -> bool:
    if os.environ.get("LSS_TEST_MODE") != "1":
        return False
    try:
        target = Path(value).resolve()
    except (OSError, RuntimeError):
        return True
    for item in os.environ.get("LSS_PROTECTED_STATE_ROOTS", "").split(os.pathsep):
        if item.strip():
            try:
                if Path(item).resolve() == target:
                    return True
            except (OSError, RuntimeError):
                continue
    return False
