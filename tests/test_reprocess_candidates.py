"""処理の更新に対する再処理候補の抽出。台帳と成果物は読むだけで変更しない。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from mutagen.flac import FLAC

from lss.assets import reprocess
from lss.assets.reprocess import ProcessingIdentity
from lss.core import session

CURRENT = ProcessingIdentity("model_new", "n" * 64, "c" * 64)


def _output(path: Path, *, model: str, model_sha: str, config_sha: str) -> Path:
    sf.write(path, np.zeros((100, 2), np.float32), 44100, subtype="PCM_16")
    tags = FLAC(path)
    revision = {"stage_graph": [{"model_id": model, "model_sha256": model_sha, "config_sha256": config_sha}]}
    tags["LSS_RECIPE_REVISION"] = json.dumps(revision)
    tags.save()
    return path


def _source(path: Path, content: bytes) -> tuple[Path, str]:
    path.write_bytes(content)
    return path, session.input_sha256_of(path)


def test_only_results_from_older_validated_processing_become_candidates(tmp_path):
    state = tmp_path / "state"
    old_source, old_sha = _source(tmp_path / "old.flac", b"old")
    new_source, new_sha = _source(tmp_path / "new.flac", b"new")
    config_source, config_sha = _source(tmp_path / "config.flac", b"config")
    legacy_source, legacy_sha = _source(tmp_path / "legacy.flac", b"legacy")
    redone_source, redone_sha = _source(tmp_path / "redone.flac", b"redone")

    def done(source, sha, model, model_sha, config, when, name, recipe="separation-v1-x"):
        output = _output(tmp_path / f"{name}.flac", model=model, model_sha=model_sha, config_sha=config)
        session.mark_completed(sha, recipe, when, source_path=source, outputs=[output], state_root=state)

    done(old_source, old_sha, "model_old", "o" * 64, "c" * 64, "2026-08-01", "o1")
    done(new_source, new_sha, "model_new", "n" * 64, "c" * 64, "2026-09-01", "n1")
    done(config_source, config_sha, "model_new", "n" * 64, "d" * 64, "2026-08-02", "c1")
    legacy_out = tmp_path / "l1.flac"
    sf.write(legacy_out, np.zeros((10, 2), np.float32), 44100, subtype="PCM_16")
    session.mark_completed(legacy_sha, "model_old", "2026-07-01", source_path=legacy_source, outputs=[legacy_out],
                           state_root=state)
    done(redone_source, redone_sha, "model_old", "o" * 64, "c" * 64, "2026-08-03", "r1")
    done(redone_source, redone_sha, "model_new", "n" * 64, "c" * 64, "2026-09-02", "r2", recipe="separation-v1-y")
    old_source.unlink()

    result = reprocess.reprocess_candidates(state_root=state, current=CURRENT, verify_sources=True)

    reasons = {c["input_sha256"]: (c["reason"], c["source_status"]) for c in result["candidates"]}
    assert reasons == {
        old_sha: ("model_updated", "missing"),
        config_sha: ("config_updated", "present"),
        # 初期の台帳はrecipe_idにモデル識別子だけを残している。別モデルであることは分かる。
        legacy_sha: ("model_updated", "present"),
    }
    assert result["counts"] == {"model_updated": 2, "already_current": 2, "config_updated": 1}
    assert reprocess.ProcessingIdentity("model_new", None, None).compare(CURRENT) == "processing_unverifiable"
    assert reprocess.ProcessingIdentity(None, None, None).compare(CURRENT) == "processing_unknown"


def test_missing_sources_point_to_the_recorded_reacquisition(tmp_path, monkeypatch):
    state = tmp_path / "state"
    source, sha = _source(tmp_path / "a.webm", b"acquired")
    output = _output(tmp_path / "a - Instrumental.flac", model="model_old", model_sha="o" * 64, config_sha="c" * 64)
    session.mark_completed(sha, "separation-v1-x", "2026-08-01", source_path=source, outputs=[output], state_root=state)
    source.unlink()
    from lss.acquisition import store

    monkeypatch.setattr(store, "records_for_sha256", lambda digest, root=None: [
        {"content": {"sha256": digest}, "reacquire": {"url": "https://www.youtube.com/watch?v=x"}}])
    [candidate] = reprocess.reprocess_candidates(state_root=state, current=CURRENT)["candidates"]
    assert candidate["source_status"] == "missing_reacquirable"
    assert candidate["reacquire_url"] == "https://www.youtube.com/watch?v=x"
