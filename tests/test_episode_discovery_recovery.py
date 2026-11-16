"""BGM候補の中断再開は同じ出典と処理に限定し、分離を無駄に繰り返さない。"""
import json
from pathlib import Path

from mutagen.flac import FLAC
import numpy as np
import pytest
import soundfile as sf

from lss.core import session
from lss.validation import episode_discovery as episode


def _setup(tmp_path):
    source = tmp_path / "source.wav"
    samples = np.zeros((4000, 2), dtype=np.float32)
    samples[200:900] = 0.2
    samples[2200:2900] = 0.15
    sf.write(source, samples, 1000, subtype="PCM_24")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "schema_version": 1, "method": "windowed_rms_hysteresis",
        "window_frames": 100, "hop_frames": 50,
        "onset_rms": 0.08, "offset_rms": 0.04,
        "bridge_gap_frames": 0, "minimum_active_frames": 1, "padding_frames": 0,
    }), encoding="utf-8")
    calls = []

    def separate(job, *, destination_dir):
        calls.append(job)
        signal = destination_dir / "signal.flac"
        sf.write(signal, samples, 1000, subtype="PCM_24")
        tags = FLAC(signal)
        tags["LSS_STEM"] = "Instrumental"
        tags.save()
        return {
            "status": "ok", "outputs": [str(signal)],
            "input_sha256": session.input_sha256_of(source),
            "recipe_id": "separation-r1", "runtime_id": "runtime-r1",
        }

    run_dir = tmp_path / "run"

    def run(**kwargs):
        return episode.run_episode_discovery(
            source, [recipe], run_dir, source_id="episode-1", model_id="model",
            separate=separate, **kwargs,
        )

    return source, recipe, run_dir, calls, run


def test_resume_after_partial_candidate_save_keeps_ids_and_prior_files(tmp_path, monkeypatch):
    source, _, run_dir, calls, run = _setup(tmp_path)
    before_source = source.read_bytes()
    materialize = episode.materialize_candidate
    count = 0

    def interrupted(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("候補保存中の中断")
        return materialize(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(episode, "materialize_candidate", interrupted)
        with pytest.raises(OSError, match="中断"):
            run()
    assert not (run_dir / "run.json").exists()
    first = next((run_dir / "candidates").rglob("*.flac"))
    first_bytes = first.read_bytes()
    run_path = run(resume=True)
    assert len(calls) == 1
    assert first.read_bytes() == first_bytes
    assert len(list((run_dir / "candidates").rglob("*.flac"))) == 2
    completed = json.loads(run_path.read_text(encoding="utf-8"))
    assert completed["outcome"]["kind"] == "unresolved_candidates"
    assert completed["outcome"]["candidate_count"] == 2
    assert source.read_bytes() == before_source
    before_run = run_path.read_bytes()
    assert run(resume=True) == run_path
    assert run_path.read_bytes() == before_run
    assert len(calls) == 1


@pytest.mark.parametrize("changed", ["source", "recipe", "signal"])
def test_resume_refuses_changed_inputs_or_prepared_artifact(tmp_path, changed):
    source, recipe, run_dir, calls, run = _setup(tmp_path)
    run_path = run()
    before = run_path.read_bytes()
    if changed == "source":
        sf.write(source, np.zeros((4000, 2)), 1000)
    elif changed == "recipe":
        value = json.loads(recipe.read_text(encoding="utf-8"))
        value["onset_rms"] = 0.09
        recipe.write_text(json.dumps(value), encoding="utf-8")
    else:
        sf.write(run_dir / "artifacts" / "signal.flac", np.zeros((4000, 2)), 1000)
    with pytest.raises(episode.EpisodeDiscoveryError):
        run(resume=True)
    assert len(calls) == 1
    assert run_path.read_bytes() == before


def test_missing_candidate_is_regenerated_without_reseparation(tmp_path):
    _, _, run_dir, calls, run = _setup(tmp_path)
    run_path = run()
    missing = next((run_dir / "candidates").rglob("*.flac"))
    before = missing.read_bytes()
    missing.unlink()
    assert run(resume=True) == run_path
    assert len(calls) == 1
    assert missing.read_bytes() == before
    assert len(list((run_dir / "candidates").rglob("*.flac"))) == 2


def test_resume_does_not_claim_completion_from_empty_activity_manifest(tmp_path):
    _, _, run_dir, calls, run = _setup(tmp_path)
    run_path = run()
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["activity_runs"] = []
    run_path.write_text(json.dumps(payload), encoding="utf-8")
    run(resume=True)
    restored = json.loads(run_path.read_text(encoding="utf-8"))
    assert len(restored["activity_runs"]) == 1
    assert len(calls) == 1


def test_folder_add_ignores_candidate_assets_but_explicit_add_accepts_them(tmp_path):
    source, _, run_dir, _, run = _setup(tmp_path)
    run()
    candidate = next((run_dir / "candidates").rglob("*.flac"))
    assert candidate not in session.iter_input_files([tmp_path])
    assert source in session.iter_input_files([tmp_path])
    assert session.iter_input_files([candidate]) == [candidate]


def test_resume_refuses_existing_directory_without_intent(tmp_path):
    _, _, run_dir, calls, run = _setup(tmp_path)
    run_dir.mkdir()
    unrelated = run_dir / "unrelated.txt"
    unrelated.write_text("保護対象", encoding="utf-8")
    with pytest.raises(episode.EpisodeDiscoveryError, match="記録がありません"):
        run(resume=True)
    assert calls == []
    assert unrelated.read_text(encoding="utf-8") == "保護対象"


def test_no_activity_is_unresolved_and_not_validated_absence(tmp_path):
    _, recipe, _, _, run = _setup(tmp_path)
    value = json.loads(recipe.read_text(encoding="utf-8"))
    value["onset_rms"] = 0.8
    value["offset_rms"] = 0.4
    recipe.write_text(json.dumps(value), encoding="utf-8")
    result = json.loads(run().read_text(encoding="utf-8"))
    assert result["outcome"] == {
        "kind": "unresolved_no_activity_candidates", "candidate_count": 0,
        "publication_state": "not_published",
    }
    assert result["semantic_ground_truth"] is None
    assert "zero_result" not in result["outcome"]


def test_separation_failure_is_not_an_empty_or_unknown_completed_result(tmp_path):
    source, recipe, run_dir, calls, run = _setup(tmp_path)

    def failed(*args, **kwargs):
        return {"status": "error", "error_code": "cuda_required", "message": "GPUを利用できません"}

    with pytest.raises(episode.EpisodeDiscoveryError, match="cuda_required"):
        episode.run_episode_discovery(
            source, [recipe], run_dir, source_id="episode-1", model_id="model", separate=failed,
        )
    assert not (run_dir / "run.json").exists()
    assert not (run_dir / "separation.json").exists()
    assert not list(run_dir.rglob("*.flac"))
    # 同じ入力・処理のまま、GPUが回復した後に再開できる。
    result = json.loads(run(resume=True).read_text(encoding="utf-8"))
    assert result["outcome"]["kind"] == "unresolved_candidates"
    assert len(calls) == 1


def test_newly_validated_silence_is_zero_without_candidate_placeholder(tmp_path):
    from lss.core.presence import inspect_signal

    source, recipe, run_dir, _, _ = _setup(tmp_path)
    sf.write(source, np.zeros((4000, 2)), 1000, subtype="PCM_24")
    zero = inspect_signal(source).zero_result()

    def separate(job, **kwargs):
        return {
            "status": "ok", "outputs": [], "zero_result": zero,
            "input_sha256": session.input_sha256_of(source),
            "recipe_id": "silent-r1", "runtime_id": "runtime-r1",
        }

    run_path = episode.run_episode_discovery(
        source, [recipe], run_dir, source_id="episode-1", model_id="model", separate=separate,
    )
    run = json.loads(run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed_validated_zero_result"
    assert run["outcome"] == {"kind": "zero", **zero}
    assert run["activity_runs"] == []
    assert not (run_dir / "candidates").exists()
