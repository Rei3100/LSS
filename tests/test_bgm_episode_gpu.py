"""実番組を実GPUで処理し、未確定候補の保存中断・再開まで確認する。"""
import json
import os
from pathlib import Path

from mutagen.flac import FLAC
import pytest
import soundfile as sf

from lss.core import catalog, models_store, session, worker
from lss.harness import tsv_io
from lss.validation import episode_discovery as episode


@pytest.mark.skipif(os.environ.get("LSS_RUN_BGM_EPISODE_ACCEPTANCE") != "1", reason="実番組のGPU受入は明示実行")
def test_full_episode_gpu_candidate_save_and_resume(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    predecessor = json.loads(Path(os.environ["LSS_BGM_EPISODE_MANIFEST"]).read_text(encoding="utf-8"))
    source = Path(predecessor["source"]["path"])
    assert source.is_file(), "既存Development番組の原本が必要です"
    source_hash = session.input_sha256_of(source)
    model_id = catalog.default_model()["model_id"]
    header, rows = tsv_io.load_tsv(models_store.MODELS_TSV)
    ledger = tmp_path / "models.tsv"
    tsv_io.save_tsv(ledger, header, rows)
    monkeypatch.setattr(models_store, "MODELS_TSV", ledger)
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    recipe = Path(os.environ["LSS_BGM_EPISODE_ACTIVITY_RECIPE"])
    output_dir = tmp_path / "episode-run"
    separations = []
    phases = set()

    def separate(job, **kwargs):
        def progress(event):
            phase = event["phase"]
            if phase not in phases:
                phases.add(phase)
                print(f"実番組処理: {phase}", flush=True)
        result = worker.handle_job(job, progress=progress, **kwargs)
        separations.append(result)
        assert result["status"] == "ok", result
        assert result["runtime_revision"]["device"] == "cuda:0"
        return result

    materialize = episode.materialize_candidate
    saved = []

    def interrupted(*args, **kwargs):
        if saved:
            raise OSError("受入試験で候補保存を中断")
        result = materialize(*args, **kwargs)
        saved.append(result)
        return result

    def run(**kwargs):
        return episode.run_episode_discovery(
            source, [recipe], output_dir,
            source_id=predecessor["source"]["source_id"], model_id=model_id,
            separate=separate, **kwargs,
        )

    with monkeypatch.context() as patch:
        patch.setattr(episode, "materialize_candidate", interrupted)
        with pytest.raises(OSError, match="候補保存を中断"):
            run()
    first = Path(saved[0]["path"])
    first_sha = session.input_sha256_of(first)
    assert not (output_dir / "run.json").exists()
    run_path = run(resume=True)
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    assert len(separations) == 1, "分離が成功した区間保存の再開ではGPU分離を繰り返さない"
    assert session.input_sha256_of(first) == first_sha
    assert run_record["outcome"]["kind"] == "unresolved_candidates"
    assert run_record["outcome"]["publication_state"] == "not_published"
    detection = json.loads(Path(run_record["activity_runs"][0]["record_path"]).read_text(encoding="utf-8"))
    assert len(detection["candidates"]) > 1
    for candidate in detection["candidates"]:
        artifact = candidate["asset"]
        path = Path(artifact["path"])
        assert session.input_sha256_of(path) == artifact["sha256"]
        assert sf.info(path).frames == candidate["end_frame"] - candidate["start_frame"]
        provenance = json.loads(FLAC(path)["LSS_CANDIDATE_PROVENANCE"][0])
        assert provenance["origin"]["source_sha256"] == source_hash
        assert provenance["semantic_role"] == "unresolved_music_like"
        assert provenance["publication_state"] == "not_published"
    assert session.input_sha256_of(source) == source_hash
    frozen = run_path.read_bytes()
    assert run(resume=True) == run_path
    assert run_path.read_bytes() == frozen
    assert len(separations) == 1
    print(json.dumps({
        "source_sha256": source_hash, "source_id": run_record["source"]["source_id"],
        "candidate_count": len(detection["candidates"]),
        "separation_recipe_id": separations[0]["recipe_id"],
        "runtime_id": separations[0]["runtime_id"],
        "verified": ["full_episode_cuda", "candidate_provenance", "interruption_resume", "idempotent_rerun"],
        "semantic_quality_evaluated": False,
    }, ensure_ascii=False), flush=True)
