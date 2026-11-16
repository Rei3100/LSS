"""clean_scratch: finished acceptance/post-deploy runs lose their bulk on the SSD; text Evidence and link targets stay."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from tests.machine_config_fixtures import machine_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import storage_tiers  # noqa: E402


def _policy(tmp_path: Path, runs: Path) -> dict:
    return {**machine_policy(tmp_path), "local_state_root": str(tmp_path / "state"), "storage_budget": {"floor_gb": 0, "default_job_gb": 0},
            "scratch_runs": {"globs": [str(runs / "accept-*")], "min_idle_hours": 12,
                             "keep_suffixes": [".json", ".md", ".log"], "keep_max_mb": 1}}


def _age(path: Path, hours: float) -> None:
    then = time.time() - hours * 3600
    for folder, _dirs, files in os.walk(path):
        for name in files:
            os.utime(Path(folder) / name, (then, then))


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows links")
def test_a_finished_run_keeps_its_evidence_and_never_reaches_through_a_link(tmp_path):
    import _winapi

    runs = tmp_path / "work"
    run = runs / "accept-x"
    (run / "case" / "portable").mkdir(parents=True)
    (run / "verdict.json").write_text("{}", encoding="utf-8")
    (run / "case" / "portable" / "LSS.exe").write_bytes(b"x" * 4096)
    (run / "case" / "huge.log").write_bytes(b"x" * 2_000_000)                    # over keep_max_mb: bulk
    models = tmp_path / "product-models"
    models.mkdir()
    (models / "weights.onnx").write_bytes(b"w" * 1024)
    _winapi.CreateJunction(str(models), str(run / "case" / "models"))
    _age(run, 24)
    _age(models, 24)

    record = storage_tiers.clean_scratch(_policy(tmp_path, runs), apply=True, log=lambda line: None)

    assert record["bytes"] == 4096 + 2_000_000
    assert (run / "verdict.json").is_file()
    assert not (run / "case" / "portable").exists()                           # emptied folders go too
    assert (models / "weights.onnx").is_file() and (run / "case" / "models").exists()


def test_a_run_written_recently_is_left_alone(tmp_path):
    runs = tmp_path / "work"
    run = runs / "accept-y"
    run.mkdir(parents=True)
    (run / "audio.flac").write_bytes(b"a" * 4096)

    record = storage_tiers.clean_scratch(_policy(tmp_path, runs), apply=True, log=lambda line: None)

    assert record["deleted"] == [] and (run / "audio.flac").is_file()


def test_unclassified_models_and_their_caches_stay_with_or_without_profiles(tmp_path, monkeypatch):
    store = tmp_path / "models" / "strata"
    for name in ("qwen36", "qwen36-27b", "coder"):
        (store / name).mkdir(parents=True)
        (store / name / "w.gguf").write_bytes(b"w")
    (store / "qwen36" / ".cache").mkdir()
    (store / "qwen36" / ".cache" / "blob").write_bytes(b"c")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "launch.json").write_text(('{"model": "D' + chr(58) + '\\x\\strata\\qwen36\\w.gguf"}'), encoding="utf-8")
    policy = {"unused_local_models": {"roots": [str(store)], "profile_dirs": [str(profiles)]}}
    monkeypatch.setattr(storage_tiers, "_recent", lambda path, idle: False)

    found = storage_tiers.unused_local_models(policy, [], 0.0)

    assert found == []
    assert (store / "qwen36" / ".cache" / "blob").read_bytes() == b"c"


def test_kept_model_folders_survive_even_when_no_launch_profile_names_them(tmp_path, monkeypatch):
    store = tmp_path / "models" / "strata"
    for name in ("coder", "trial"):
        (store / name).mkdir(parents=True)
        (store / name / "w.gguf").write_bytes(b"w")
    (store / "coder" / ".cache").mkdir()
    (store / "coder" / ".cache" / "blob").write_bytes(b"c")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    policy = {"unused_local_models": {"roots": [str(store)], "profile_dirs": [str(profiles)], "keep": ["coder"]}}
    monkeypatch.setattr(storage_tiers, "_recent", lambda path, idle: False)

    found = storage_tiers.unused_local_models(policy, [], 0.0)

    assert found == []
    assert (store / "coder" / ".cache" / "blob").read_bytes() == b"c"
