import os
import sys
from pathlib import Path

import lss_gui


def test_runtime_caches_are_created_under_managed_state(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("LSS_STATE_DIR", str(state))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    lss_gui._configure_runtime_caches()

    assert Path(os.environ["NUMBA_CACHE_DIR"]) == state / "runtime-cache" / "numba"
    assert Path(os.environ["MPLCONFIGDIR"]) == state / "runtime-cache" / "matplotlib"
    assert (state / "runtime-cache" / "numba").is_dir()
    assert (state / "runtime-cache" / "matplotlib").is_dir()


def test_runtime_caches_preserve_explicit_locations(tmp_path, monkeypatch):
    numba = tmp_path / "chosen-numba"
    matplotlib = tmp_path / "chosen-matplotlib"
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(numba))
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib))

    lss_gui._configure_runtime_caches()

    assert numba.is_dir()
    assert matplotlib.is_dir()


def test_frozen_runtime_replaces_pyinstaller_matplotlib_cache(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("LSS_STATE_DIR", str(state))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "pyinstaller-temp"))
    monkeypatch.setattr("lss.paths.is_frozen", lambda: True)

    lss_gui._configure_runtime_caches()

    assert Path(os.environ["MPLCONFIGDIR"]) == state / "runtime-cache" / "matplotlib"


def test_frozen_inference_uses_resource_directory_and_restores_cwd(tmp_path, monkeypatch):
    resource = tmp_path / "internal"
    resource.mkdir()
    previous = Path.cwd()
    monkeypatch.setattr("lss.paths.is_frozen", lambda: True)
    monkeypatch.setattr("lss.paths.resource_root", lambda: resource)

    with lss_gui._frozen_inference_directory():
        assert Path.cwd() == resource

    assert Path.cwd() == previous


def test_worker_mode_restores_missing_standard_streams(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    lss_gui._ensure_worker_streams()

    assert sys.stdout is not None and sys.stdout.name == os.devnull
    assert sys.stderr is not None and sys.stderr.name == os.devnull
    print("discarded worker output")


def test_bgm_candidate_mode_writes_a_durable_product_receipt(tmp_path, monkeypatch):
    from lss import application

    source = tmp_path / "episode.wav"
    source.write_bytes(b"source")
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    candidate = tmp_path / "candidate.flac"
    candidate.write_bytes(b"candidate")
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    seen = []

    def run(request):
        seen.append(request)
        return application.ProcessingOutcome(
            disposition=application.ResultDisposition.HELD,
            held_artifacts=(candidate,), evidence_path=evidence, record_path=evidence,
            held_alternative_artifacts=(tmp_path / "original-mixture.flac",),
            recipe_id="recipe-r1", message="候補を保留しました",
        )

    monkeypatch.setattr(application, "run_bgm_candidate_request", run)

    assert lss_gui.main(["--bgm-candidates", str(source)]) == 0
    assert seen[0].intent is application.UserIntent.EXTRACT_BGM_CUES
    receipt = tmp_path / "state" / "bgm-candidate-result.json"
    import json
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert saved["disposition"] == "HELD"
    assert saved["held_artifacts"] == [str(candidate)]
    assert saved["held_alternative_artifacts"] == [str(tmp_path / "original-mixture.flac")]
    assert Path(saved["source"]) == source.resolve()
