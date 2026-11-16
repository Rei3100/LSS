import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "discover_bgm_activity.py"
SPEC = importlib.util.spec_from_file_location("discover_bgm_activity", SCRIPT)
discover_bgm_activity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(discover_bgm_activity)


def test_activity_cli_writes_evaluator_compatible_unresolved_candidates(tmp_path):
    audio = tmp_path / "music-like.wav"
    samples = np.zeros((1000, 2), dtype=np.float32)
    samples[200:500] = 0.1
    sf.write(audio, samples, 1000, subtype="FLOAT")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "schema_version": 1,
        "method": "windowed_rms_hysteresis",
        "window_frames": 100,
        "hop_frames": 50,
        "onset_rms": 0.08,
        "offset_rms": 0.04,
        "bridge_gap_frames": 0,
        "minimum_active_frames": 1,
        "padding_frames": 0,
    }), encoding="utf-8")
    output = tmp_path / "detections.jsonl"
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(item) for item in [
        {
            "type": "manifest", "schema_version": 1, "corpus_id": "test",
            "revision": "r1", "description": "test",
        },
        {
            "type": "case", "case_id": "case-1", "description": "test",
            "source": {"path": audio.name, "sha256": digest, "sample_rate": 1000, "frames": 1000},
            "expected_outcome": "wanted_present", "attributes": [],
            "occurrences": [{
                "occurrence_id": "score", "start_frame": 200, "end_frame": 500,
                "semantic_role": "dramatic_score", "wanted": True, "novel": True,
                "identity_state": "unknown", "cue_identity_id": None, "audio_version_id": None,
            }],
        },
    ]) + "\n", encoding="utf-8")

    assert discover_bgm_activity.main([
        str(manifest), str(audio), str(recipe), "case-1", str(output),
    ]) == 0

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["case_id"] == "case-1"
    assert record["corpus_source"]["sha256"] == digest
    assert record["analysis_signal"]["sha256"] == digest
    assert record["activity_recipe"]["recipe_id"]
    assert len(record["activity_implementation"]["artifact_sha256"]) == 64
    assert [(item["start_frame"], item["end_frame"]) for item in record["candidates"]] == [
        (200, 550)
    ]
    assert record["candidates"][0]["semantic_role"] == "unresolved_music_like"
    assert record["candidates"][0]["identity_state"] == "unresolved"
    assert list(tmp_path.glob(f".{output.name}-*.tmp")) == []
