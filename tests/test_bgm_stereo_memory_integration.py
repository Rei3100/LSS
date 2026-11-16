"""Production decision integration: optional memory cannot discard measured stereo music."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss import application as app
from lss.bgm import cue_semantics as cs, cue_representation as cr, episode_cues
from lss.bgm import cue_memory, identity, memory_support, reference_store


@pytest.mark.parametrize("mode", ["no_memory", "no_records", "lookup_failure", "formation_failure", "supported", "reference_failure", "unconfirmed_stereo"])
def test_decide_retains_stereo_evidence_through_memory_routes(tmp_path, monkeypatch, mode):
    n = 40
    music_probability = np.zeros(n, np.float32)
    music_probability[5:35] = 1
    music_probability[15:19] = 0
    rate = 44100
    tone = .1 * np.sin(np.arange(n * rate) * (2 * np.pi * 440 / rate))
    audio = np.column_stack((tone, -tone)).astype(np.float32)
    audio[:5 * rate] = 0
    audio[35 * rate:] = 0
    source = tmp_path / "episode.wav"
    sf.write(source, audio, rate, subtype="FLOAT")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    run_dir = tmp_path / "state" / "runs" / "source" / "run"
    recipe = {"cue_semantics_id": cs.default_recipe().recipe_id,
              "cue_representation_id": cr.default_recipe().recipe_id}
    stems = dict.fromkeys(("music", "speech", "effects"), source)
    roles = []

    def tag(signal, folder, recipe, analyze, role):
        roles.append(role)
        folder.mkdir(parents=True, exist_ok=True)
        count = sf.info(signal).frames // rate
        p = np.zeros((count, 527), np.float32)
        p[:, cs.MUSIC_CLASS] = (0 if mode == "unconfirmed_stereo" else 1) if role == "first_pass_music_stereo_difference" else music_probability
        outputs = {}
        for name, value in (("probabilities", p), ("embeddings", np.tile([1., 0.], (count, 1)))):
            path = folder / (name + ".npy")
            np.save(path, value)
            outputs[name] = {"path": str(path), "sha256": app.file_sha256(path)}
        return {"outputs": outputs}

    monkeypatch.setattr(app, "_tag_job", tag)
    levels = np.full(n, -25.)
    levels[:5] = levels[35:] = -100
    monkeypatch.setattr(episode_cues, "window_shares", lambda *args: (None, levels, None))
    monkeypatch.setattr(cr, "residual_levels", lambda *args: np.zeros(n))
    monkeypatch.setattr(app, "_bgm_published_duplicates", lambda *args: {})
    monkeypatch.setattr(cue_memory, "read", lambda *args: {"members": [] if mode in ("no_memory", "unconfirmed_stereo") else [{}]})

    def decode(*args):
        if mode == "lookup_failure":
            raise OSError("unavailable memory query")
        return np.zeros(n * identity.SAMPLE_RATE, np.float32)

    monkeypatch.setattr(identity, "decode_mono", decode)
    record = {"state": "component_observed", "reference_sha256": "memory-recording", "offset_seconds": 0.,
              "reference_type": "published_cue_memory", "music_corroborated": True,
              "onset": {"track": "memory:recording", "predicted_seconds": 5., "role": "score"},
              "spans": [{"start_seconds": 5., "end_seconds": 35.}],
              "trace_origin": 0., "times": np.arange(5., 35., .25).tolist(),
              "fraction": np.ones(120).tolist(), "control": np.zeros(120).tolist()}
    monkeypatch.setattr(memory_support, "nominate", lambda *args, **kwargs: {"state": "checked", "records": [] if mode == "no_records" else [record]})
    if mode == "formation_failure":
        def fail(*args, **kwargs):
            raise ValueError("optional formation failed")
        monkeypatch.setattr(memory_support, "never_lower", fail)
    if mode == "reference_failure":
        @contextmanager
        def failed_lease(*args, **kwargs):
            raise OSError("reference unavailable")
            yield
        monkeypatch.setattr(reference_store, "lease", failed_lease)
    references = ([{"sha256": "reference"}] if mode == "reference_failure" else [], "digest", [], tmp_path / "cache")
    plan = {"source_seconds": float(n), "cues": [{"cue_id": "cue", "start_seconds": 0., "end_seconds": 40.,
                       "disposition": episode_cues.DISPOSITION_HELD}]}
    result = app._bgm_decide(source, recipe, run_dir, scratch, None, stems, source, plan, references)
    assert "first_pass_music_stereo_difference" in roles
    stereo = result["cue_formation"]["stereo_continuations"]
    if mode == "unconfirmed_stereo":
        assert not stereo[0]["positive_music_seconds"]
        assert not any(item["semantic_class"] == cs.WANTED and item["start_seconds"] < 15
                       and item["end_seconds"] > 19 for item in result["decided"])
        return
    assert stereo[0]["positive_music_seconds"] == list(range(15, 19))
    wanted = [item for item in result["decided"] if item["semantic_class"] == cs.WANTED]
    assert len(wanted) == 1
    # Production waveform refinement uses 100 ms windows around the actual silent edges.
    assert wanted[0]["start_seconds"] == pytest.approx(5., abs=.1)
    assert wanted[0]["end_seconds"] == pytest.approx(35., abs=.1)
    assert "voice_preserving_refinement" in result
    memory = result["cue_formation"]["memory_support"]
    assert memory["state"] == {"no_memory": "no_memory", "lookup_failure": "not_checked",
                                "formation_failure": "formation_failed_first_pass_kept"}.get(mode, "checked")
    if mode == "reference_failure":
        assert result["reference_relations"] is not None
