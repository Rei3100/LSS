"""Explicit temporary machine scope for tests that exercise storage mutation."""


def machine_policy(tmp_path):
    """Declare synthetic provenance and protections, never load private config."""
    protected = tmp_path / "protected"
    return {
        "_machine_overlay": str(tmp_path / "storage_policy.local.json"),
        "never": [str(protected)],
        "protected_packages": [str(protected / "package")],
        "model_stores": [str(protected / "models")],
        "frozen_roots": [str(protected / "frozen")],
    }
