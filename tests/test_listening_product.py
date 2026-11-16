from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from mutagen.flac import FLAC

from lss import application
from lss.assets.naming import NamingStore
from lss.bgm import cue_publication, listening
from lss.core import audio_identity
from lss.output import finalizer, results
from lss.output.filenames import component


def _cue(tmp_path):
    rate, frames = 44100, 44100 * 5
    t = np.arange(frames) / rate
    signal = np.column_stack((0.2 * np.sin(2 * np.pi * 220 * t), 0.1 * np.sin(2 * np.pi * 420 * t)))
    key = cue_publication.cue_key("a" * 64, "accepted-recipe", 0, frames, rate)
    path = tmp_path / "source - BGM 00m00s.flac"
    cue_publication.publish(signal, rate, path, key=key, title="source BGM 00m00s", album="source BGM",
                            input_sha256="a" * 64, recipe_id="accepted-recipe", version="0.1.0",
                            provenance={"type": "bgm_published_cue", "source": {"sha256": "a" * 64},
                                        "start_frame": 0, "end_frame": frames, "sample_rate": rate})
    return path


def test_official_title_updates_the_same_finished_asset_and_preserves_identity_and_annotation(tmp_path):
    path = _cue(tmp_path)
    root = tmp_path / "state"
    first = listening.finish(path, root)
    assert "仮題" in FLAC(path)["TITLE"][0]
    audio = FLAC(path)
    audio["PLAY_COUNT"] = "7"
    audio.save()
    official = {"official": {"asset_id": first["naming"]["asset_id"], "title": "制御fixtureの確認済み曲名",
                             "verified": True, "relation": "exact_version", "evidence_ref": "test:verified-cue-sheet"}}
    second = listening.finish(path, root, extra_evidence=official)
    assert second["naming"]["kind"] == "official"
    assert second["path"] == first["path"] and second["audio"] == first["audio"]
    assert FLAC(path)["PLAY_COUNT"] == ["7"] and FLAC(path)["LSS_NAMING_OFFICIAL"] == ["1"]
    assert len(list(tmp_path.glob("*.flac"))) == 1
    store = NamingStore(root / listening.DIRECTORY)
    state = store.read(first["naming"]["asset_id"])
    assert len(state["revisions"]) == 2
    monkey = pytest.MonkeyPatch()
    monkey.setattr(finalizer, "scan_track_file", lambda *a, **k: pytest.fail("再走査しない"))
    try:
        assert listening.finish(path, root)["reused"]
        assert FLAC(path)["TITLE"] == ["制御fixtureの確認済み曲名"]
    finally:
        monkey.undo()


def test_presentation_failure_preserves_previous_finished_bytes_and_is_retryable(tmp_path, monkeypatch):
    path = _cue(tmp_path)
    root = tmp_path / "state"
    first = listening.finish(path, root)
    original = path.read_bytes()
    official = {"official": {"asset_id": first["naming"]["asset_id"], "title": "新しい確認済み曲名",
                             "verified": True, "relation": "exact_version", "evidence_ref": "test:verified"}}
    with monkeypatch.context() as patch:
        patch.setattr(finalizer, "finalize_staged", lambda *a, **k: (_ for _ in ()).throw(OSError("injected")))
        with pytest.raises(OSError):
            listening.finish(path, root, extra_evidence=official)
    assert path.read_bytes() == original and not listening.current([path], root)
    assert listening.finish(path, root)["naming"]["kind"] == "official"
    assert listening.current([path], root)



@pytest.mark.parametrize("value", ["CON", "NUL.txt", "A:B?*", "trailing. ", "x" * 250, "🎵" * 120])
def test_windows_filename_is_clean_bounded_and_deterministic(value):
    first = component(value)
    assert first == component(value) and len(first.encode("utf-16-le")) // 2 <= 160
    assert not any(char in first for char in '<>:"/\\|?*') and not first.endswith((" ", "."))


@pytest.mark.parametrize("disposition", ["PUBLISHED", "HELD", "ZERO", "UNSUPPORTED", "FAILED"])
def test_result_projection_never_turns_unknown_or_failure_into_publication(disposition):
    projection = results.project(disposition=disposition, outputs=["accepted.flac"] if disposition == "PUBLISHED" else [],
                                 held_count=2 if disposition == "HELD" else 0)
    assert projection["disposition"] == disposition and projection["state"] == disposition
    assert projection["published_count"] == (1 if disposition == "PUBLISHED" else 0)
    assert projection["known"] == []


def test_result_projects_existing_known_replacement_and_future_improvement_truth():
    record = {"current": {"decisions": {"old": [{"decision": "replace_with_better", "validity": "VALID"},
                                                 {"decision": "keep", "validity": "UNKNOWN"}]}}, "history": [{"old": True}]}
    semantics = {"references": {"known_spans": [{"reference": "recording.flac", "start_seconds": 3, "end_seconds": 8}]},
                 "supported_scope": {"counts": {"HELD": 2, "UNSUPPORTED": 1}}}
    before = deepcopy((record, semantics))
    projection = results.project(disposition="PUBLISHED", outputs=["best.flac"], current_best=record,
                                 semantic_record=semantics, held_count=3)
    assert projection["state"] == "IMPROVED_REPLACED" and projection["known"][0]["state"] == "KNOWN"
    assert projection["held_count"] == 2 and projection["unsupported_count"] == 1
    assert projection["replacement_history_count"] == 1
    assert (record, semantics) == before
    reused = results.project(disposition="PUBLISHED", outputs=["best.flac"], current_best=record, reused=True)
    assert reused["state"] == "REUSED" and reused["replacement_count"] == 0


def test_owned_metadata_repair_uses_currentbest_facts_not_edited_tags(tmp_path):
    path = _cue(tmp_path)
    tags = FLAC(path)
    body = json.loads(tags["LSS_BGM_PROVENANCE"][0])
    accepted = {"key": tags["LSS_BGM_CUE_KEY"][0], "provenance": body, "audio": audio_identity.audio_identity(path),
                "input_sha256": tags["LSS_INPUT_SHA256"][0], "recipe_id": tags["LSS_RECIPE_ID"][0],
                "headroom_shift": 0, "version": "0.1.0", "album": "source BGM"}
    first = listening.finish(path, tmp_path / "state", accepted=accepted)
    tags = FLAC(path)
    tags["LSS_BGM_CUE_KEY"] = "edited-key"
    tags["LSS_INPUT_SHA256"] = "b" * 64
    tags["LSS_RECIPE_ID"] = "edited-recipe"
    tags["LSS_BGM_PROVENANCE"] = "invalid-json"
    tags.save()
    repaired = listening.finish(path, tmp_path / "state", accepted=accepted)
    assert repaired["audio"] == first["audio"] and repaired["naming"]["asset_id"] == first["naming"]["asset_id"]
    assert FLAC(path)["LSS_BGM_CUE_KEY"] == [accepted["key"]]
    assert FLAC(path)["LSS_INPUT_SHA256"] == [accepted["input_sha256"]]
    assert len(list(tmp_path.glob("*.flac"))) == 1


def test_changed_pcm_cannot_be_rebound_as_a_previous_accepted_asset(tmp_path):
    import soundfile as sf
    path = _cue(tmp_path)
    tags = FLAC(path)
    accepted = {"key": tags["LSS_BGM_CUE_KEY"][0], "provenance": json.loads(tags["LSS_BGM_PROVENANCE"][0]),
                "audio": audio_identity.audio_identity(path), "input_sha256": tags["LSS_INPUT_SHA256"][0],
                "recipe_id": tags["LSS_RECIPE_ID"][0]}
    sf.write(path, np.ones((5 * 44100, 2)) * 0.1, 44100, subtype="PCM_24")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="音声が一致"):
        listening.finish(path, tmp_path / "state", accepted=accepted)
    assert path.read_bytes() == before


def test_vocal_type_is_preserved_as_source_provenance_not_as_derived_signal_fact(tmp_path):
    path = _cue(tmp_path)
    finalizer.ensure(path, tags={"TITLE": "source", "VOCAL_TYPE": "ON VOCAL"})
    tags = FLAC(path)
    assert "VOCAL_TYPE" not in tags and tags["LSS_SOURCE_VOCAL_TYPE"] == ["ON VOCAL"]
    assert finalizer.receipt(path) is not None


def test_result_report_is_readable_escaped_atomic_and_preserves_unowned_files(tmp_path):
    from lss.output import result_report
    projection = results.project(disposition="PUBLISHED", outputs=[str(tmp_path / "track.flac")])
    projection["source_sha256"] = "a" * 64
    projection["assets"] = [{"path": projection["paths"][0], "title": "<script>曲名</script>"}]
    saved = result_report.save(tmp_path / "source.wav", "bgm", projection)
    path = Path(saved["report_path"])
    text = path.read_text(encoding="utf-8")
    assert "保存しました" in text and "&lt;script&gt;" in text and "<script>" not in text
    assert "現在確認できません" in text
    assert projection["unsupported_count"] is None and "対応範囲外 未確認" in text
    path.write_text("user-owned document", encoding="utf-8")
    refused = result_report.save(tmp_path / "source.wav", "bgm", projection)
    assert "report_error" in refused and path.read_text() == "user-owned document"


def test_restored_mixed_version_result_uses_currentbest_evaluation_not_first_old_cue(tmp_path, monkeypatch):
    from lss import application
    old_path = tmp_path / "bgm-candidates/old/run/run.json"
    new_path = tmp_path / "bgm-candidates/new/run/run.json"
    new_path.parent.mkdir(parents=True)
    old = {"source": {"sha256": "a" * 64}, "recipe_id": "old-recipe"}
    new = {"source": old["source"], "recipe_id": "new-recipe"}
    new_path.write_text(json.dumps(new), encoding="utf-8")
    calls = []
    def verified(path, source_sha, recipe_id):
        calls.append((path, source_sha, recipe_id))
        return new
    monkeypatch.setattr(application, "_verified_bgm_run", verified)
    record = {"current": {"evaluated_runs": ["new"], "files": [{"run_path": str(old_path)}]}}
    assert application._bgm_presentation_basis(old, old_path, record) == (new, new_path)
    assert calls == [(new_path, "a" * 64, "new-recipe")]
    new_path.unlink()
    assert application._bgm_presentation_basis(old, old_path, record) is None


def test_listening_presentation_does_not_change_the_accepted_bgm_recipe_identity():
    """Listening presentation stays outside the audio/decision recipe.

    The recipe now includes cue formation and selective composition, so a
    historical deployed recipe hash is not the identity of this candidate.
    """
    from lss import application
    recipe = application._bgm_candidate_recipe()
    assert "cue_formation" in recipe and "selective_composition" in recipe
    assert not any("listening" in name or name.startswith("ui/") for name in recipe["implementation"])
