"""scripts/installed_episode_batch.py: the cue decisions of a run survive the scratch-run deletion."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "installed_episode_batch", Path(__file__).resolve().parents[1] / "scripts" / "installed_episode_batch.py")
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def test_keep_decisions_copies_run_and_semantics(tmp_path):
    import shutil

    scratch = tmp_path / "scratch" / "run"
    scratch.mkdir(parents=True)

    run_json_path = scratch / "run.json"
    run_json_path.write_text('{"published": []}', encoding="utf-8")

    semantics = scratch / "semantics"
    semantics.mkdir()
    cue_semantics_path = semantics / "cue-semantics.json"
    cue_semantics_path.write_text('{"segments": []}', encoding="utf-8")

    target = tmp_path / "out" / "runs" / "ep150"
    batch.keep_decisions(run_json_path, target)

    copied_run_json = target / "run.json"
    copied_cue_semantics = target / "semantics" / "cue-semantics.json"

    assert copied_run_json.exists()
    assert copied_cue_semantics.exists()
    assert copied_run_json.read_bytes() == run_json_path.read_bytes()
    assert copied_cue_semantics.read_bytes() == cue_semantics_path.read_bytes()

    shutil.rmtree(tmp_path / "scratch")

    assert copied_run_json.exists()
    assert copied_cue_semantics.exists()
