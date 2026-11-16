"""Executable invariants for identity, CurrentBest, revisions and file ownership (cloud invariant audit).

Contracts: docs/product/IDENTITY_AND_CURRENTBEST.md, docs/engine/LISTENING_AND_PRESENTATION.md, AGENTS.md
"Non-negotiable meaning". Every signal is synthesized here; no real media.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.bgm import cue_publication, current_best, listening, recurrence_asset, scene_names, write_targets
from lss.output import finalizer, results

RATE = 44100
SHA = "c" * 64


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stamp(path: Path) -> tuple[str, int]:
    return _digest(path), Path(path).stat().st_mtime_ns


def _published(folder: Path, start: int = 10, *, seconds: int = 4, tone: float = 220.0, sha: str = SHA) -> dict:
    """A receipt-free cue exactly as `cue_publication.publish` writes it (TITLE/ALBUM + LSS identity tags)."""
    frames = RATE * seconds
    t = np.arange(frames) / RATE
    audio = np.column_stack((0.2 * np.sin(2 * np.pi * tone * t), 0.1 * np.sin(2 * np.pi * tone * 1.5 * t)))
    key = cue_publication.cue_key(sha, "recipe", start * RATE, start * RATE + frames, RATE)
    source = folder.parent.parent / "ep.mkv"
    record = cue_publication.publish(
        audio, RATE, folder / cue_publication.file_name(source, start), key=key,
        title=cue_publication.title(source, start), album="ep BGM", input_sha256=sha, recipe_id="recipe",
        version="0.1.0", provenance={"type": "bgm_published_cue", "source": {"sha256": sha},
                                     "start_frame": start * RATE, "end_frame": start * RATE + frames,
                                     "sample_rate": RATE, "start_seconds": float(start),
                                     "end_seconds": float(start + seconds)})
    return {**record, "path": Path(record["path"]), "pcm_sha256": record["audio"]["pcm_sha256"],
            "start_seconds": float(start), "end_seconds": float(start + seconds)}


def _row(state: Path, cue: dict, *, run_key: str = "r" * 64) -> dict:
    run_path = state / "bgm-candidates" / run_key / "run" / "run.json"
    return {"public_path": str(cue["path"]), "pcm_sha256": cue["pcm_sha256"], "cue_key": cue["key"],
            "start_seconds": cue["start_seconds"], "end_seconds": cue["end_seconds"], "run_key": run_key,
            "recipe_id": "recipe", "workflow": "episode-sfc-routed-cues-v13", "run_path": str(run_path),
            "adoption": "first_publication"}


def _record(state: Path, rows: list[dict], sha: str = SHA) -> None:
    current_best._write_json_atomic(current_best.record_path(state, sha), {
        "type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION, "source_sha256": sha,
        "current": {"files": rows, "pinned": False, "evaluated_runs": [rows[0]["run_key"]] if rows else []},
        "history": []})


# ------------------------------------------------------------------ invariant 3: write containment


def test_a_plain_current_public_file_of_this_state_is_a_write_target(tmp_path):
    state = tmp_path / "state"
    cue = _published(tmp_path / "lib" / "LSS" / "ep - BGM")
    assert write_targets.checked(state, _row(state, cue)) == cue["path"]


def test_a_row_adopted_by_another_state_is_never_a_write_target(tmp_path):
    """A copied state's CurrentBest still names the library it came from: an observation, never a target."""
    original, clone = tmp_path / "original-state", tmp_path / "clone-state"
    cue = _published(tmp_path / "library-A" / "LSS" / "ep - BGM")
    with pytest.raises(write_targets.WriteTargetRefused) as refused:
        write_targets.checked(clone, _row(original, cue))
    assert refused.value.reason == "adopted_by_another_state"


def test_a_symlinked_public_path_is_never_a_write_target(tmp_path):
    state = tmp_path / "state"
    real = _published(tmp_path / "user-music")                          # somewhere outside any LSS folder
    link = tmp_path / "lib" / "LSS" / "ep - BGM" / real["path"].name
    link.parent.mkdir(parents=True)
    try:
        os.symlink(real["path"], link)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:      # Windows: creating a symlink needs a privilege
            pytest.skip("symlink creation needs SeCreateSymbolicLinkPrivilege on this Windows account")
        raise
    with pytest.raises(write_targets.WriteTargetRefused) as refused:
        write_targets.checked(state, {**_row(state, real), "public_path": str(link)})
    assert refused.value.reason == "public_path_is_a_link"


def test_a_dotdot_public_path_is_never_a_write_target(tmp_path):
    state = tmp_path / "state"
    cue = _published(tmp_path / "Music")
    escaped = tmp_path / "lib" / "LSS" / "ep - BGM" / ".." / ".." / ".." / "Music" / cue["path"].name
    (tmp_path / "lib" / "LSS" / "ep - BGM").mkdir(parents=True)
    with pytest.raises(write_targets.WriteTargetRefused) as refused:
        write_targets.checked(state, {**_row(state, cue), "public_path": str(escaped)})
    assert refused.value.reason == "not_a_plain_public_path"


def test_a_file_that_is_no_longer_the_cue_is_never_a_write_target(tmp_path):
    state = tmp_path / "state"
    folder = tmp_path / "lib" / "LSS" / "ep - BGM"
    cue = _published(folder)
    other = _published(tmp_path / "elsewhere", start=50, tone=330.0)
    cue["path"].unlink()
    os.replace(other["path"], cue["path"])                              # another cue took the public name
    with pytest.raises(write_targets.WriteTargetRefused) as refused:
        write_targets.checked(state, _row(state, cue))
    assert refused.value.reason == "public_file_is_not_this_cue"


def test_recurrence_refresh_elsewhere_never_writes_a_copied_states_library(tmp_path):
    """`application._refresh_recurrence_elsewhere`: the other source's row names another state's library."""
    from lss import application

    original, clone = tmp_path / "original-state", tmp_path / "clone-state"
    cue = _published(tmp_path / "library-A" / "LSS" / "ep - BGM")
    _record(clone, [_row(original, cue)])                               # copied record, adopted by `original`
    before = _stamp(cue["path"])
    result = application._refresh_recurrence_elsewhere(clone, [{
        "source_sha256": SHA, "asset_id": f"bgm-cue:{cue['key']}", "pcm_sha256": cue["pcm_sha256"],
        "path": str(cue["path"]), "view": None}])
    assert result["refreshed"] == 0 and result["pending"][0]["reason"] == "adopted_by_another_state"
    assert _stamp(cue["path"]) == before


def test_recurrence_asset_is_never_built_beside_a_copied_states_library(tmp_path):
    original, clone = tmp_path / "original-state", tmp_path / "clone-state"
    cue = _published(tmp_path / "library-A" / "LSS" / "ep - BGM")
    _record(clone, [_row(original, cue)])
    member = {"source_sha256": SHA, "cue_key": cue["key"], "pcm_sha256": cue["pcm_sha256"], "path": str(cue["path"])}
    assert recurrence_asset._is_current_public(clone, member) is False
    _record(original, [_row(original, cue)])
    assert recurrence_asset._is_current_public(original, member) is True


# ------------------------------------------------------------------ invariant 4: user-owned tags survive


def test_a_recurrence_asset_rebuild_keeps_foreign_tags_and_rewrites_lss_presentation(tmp_path):
    target = tmp_path / "lib" / "LSS" / "BGM 統合版" / "劇伴#1 統合版.flac"
    tags = {"TITLE": "劇伴#1（統合版）", "ALBUM": "LSS 劇伴（統合版）", "LSS_SOURCE_TYPE": "bgm_recurrence_asset",
            "LSS_STEM": "bgm", recurrence_asset.ASSET_TAG: json.dumps({"label_number": 1})}
    t = np.arange(RATE * 4) / RATE
    first = np.column_stack((0.2 * np.sin(2 * np.pi * 220 * t),) * 2)
    recurrence_asset._publish(target, first, RATE, tags)
    audio = FLAC(target)
    audio["COMMENT"], audio["RATING"], audio["GENRE"], audio["MY_FIELD"] = "my note", "80", "Soundtrack", "kept"
    audio.save()
    rebuilt = np.column_stack((0.2 * np.sin(2 * np.pi * 330 * t),) * 2)
    recurrence_asset._publish(target, rebuilt, RATE, tags)
    after = FLAC(target)
    assert after["COMMENT"] == ["my note"] and after["RATING"] == ["80"] and after["MY_FIELD"] == ["kept"]
    assert "GENRE" not in after                       # a presentation key is LSS's to write
    assert finalizer.receipt(target) is not None


def test_listening_refresh_keeps_foreign_tags_and_rewrites_presentation_on_a_receipt_free_cue(tmp_path):
    state = tmp_path / "state"
    cue = _published(tmp_path / "lib" / "LSS" / "ep - BGM")
    audio = FLAC(cue["path"])
    audio["GENRE"], audio["COMMENT"], audio["RATING"] = "Anime", "note", "100"
    audio.save()
    listening.finish(cue["path"], state, expected_pcm=cue["pcm_sha256"])
    after = FLAC(cue["path"])
    assert "GENRE" not in after and after["COMMENT"] == ["note"] and after["RATING"] == ["100"]
    assert current_best.pcm_sha256(cue["path"]) == cue["pcm_sha256"]       # presentation only, PCM untouched
    # LSS owns the title: an edited TITLE is rewritten by the next refresh, foreign tags stay.
    lss_title = FLAC(cue["path"])["TITLE"]
    audio = FLAC(cue["path"])
    audio["TITLE"] = "私の題名"
    audio.save()
    listening.finish(cue["path"], state, expected_pcm=cue["pcm_sha256"])
    assert FLAC(cue["path"])["TITLE"] == lss_title and FLAC(cue["path"])["COMMENT"] == ["note"]


# ------------------------------------------------------------------ crash points: never a torn public file


def test_a_failure_between_staging_and_replace_leaves_the_public_file_and_no_temporary(tmp_path):
    state = tmp_path / "state"
    cue = _published(tmp_path / "lib" / "LSS" / "ep - BGM")
    before = _stamp(cue["path"])

    def interrupted():
        raise RuntimeError("power cut between temp write and replace")

    with pytest.raises(RuntimeError):
        listening.finish(cue["path"], state, expected_pcm=cue["pcm_sha256"], validate_before_publish=interrupted)
    assert _stamp(cue["path"]) == before
    assert [p.name for p in cue["path"].parent.iterdir()] == [cue["path"].name]


def test_a_refresh_refuses_a_file_whose_audio_is_not_the_current_best(tmp_path):
    state = tmp_path / "state"
    cue = _published(tmp_path / "lib" / "LSS" / "ep - BGM")
    before = _stamp(cue["path"])
    with pytest.raises(finalizer.FinalizationError):
        listening.finish(cue["path"], state, expected_pcm="0" * 64)
    assert _stamp(cue["path"]) == before


def test_a_malformed_flac_named_by_a_current_best_row_stops_safely(tmp_path):
    """An older record (adoption-policy-v2 era) naming a truncated FLAC: refused, nothing written, no crash."""
    state = tmp_path / "state"
    cue = _published(tmp_path / "lib" / "LSS" / "ep - BGM")
    data = cue["path"].read_bytes()
    cue["path"].write_bytes(data[: len(data) // 2])
    before = _stamp(cue["path"])
    row = {**_row(state, cue), "adoption_policy": "bgm-currentbest-adoption-v2"}
    assert current_best._matches(cue["path"], row["pcm_sha256"]) is False
    with pytest.raises(Exception):
        listening.finish(cue["path"], state, expected_pcm=row["pcm_sha256"])
    assert _stamp(cue["path"]) == before


# ------------------------------------------------------------------ invariant 1: latest != CurrentBest; rollback


def test_an_unknown_or_held_new_revision_never_displaces_accepted_audio():
    old = {"public_path": "/x/old.flac", "pcm_sha256": "a", "start_seconds": 0, "end_seconds": 20,
           "workflow": "episode-sfc-routed-cues-v12", "extent": "single_cue"}
    new = {"public_path": "/x/new.flac", "pcm_sha256": "b", "start_seconds": 0, "end_seconds": 25,
           "improvements": [], "extent": None}                        # newer, longer, but extent UNKNOWN
    for state in ("VALID", "UNKNOWN"):
        chosen = current_best.select([old], [{"state": state}], [new])
        assert chosen["keep"] == [old] and chosen["retire"] == [] and chosen["place"] == []
    with pytest.raises(current_best.CurrentBestError):
        current_best.select([old], [{"state": "HELD"}], [new])     # an unknown verdict is not silently "valid"


def test_rollback_restores_the_prior_bytes_and_lineage_and_can_return(tmp_path):
    state, folder = tmp_path / "state", tmp_path / "lib" / "LSS" / "ep - BGM"
    history_root = tmp_path / "lib" / "LSS" / current_best.HISTORY_DIR
    old = _published(folder, start=10)
    old_bytes = old["path"].read_bytes()
    old_row = _row(state, old)
    _record(state, [old_row])
    staged = _published(tmp_path / "lib" / "LSS" / current_best.STAGING_DIR / "n", start=9, seconds=6, tone=240.0)
    new_public = folder / staged["path"].name
    new_row = {**_row(state, staged, run_key="n" * 64), "public_path": str(new_public)}
    record = current_best.read(state, SHA)
    journal = current_best.plan_switch(state, SHA, {"files": [{**new_row, "placed_now": True}], "pinned": False,
                                                    "evaluated_runs": ["n" * 64]},
                                       {str(new_public): staged["path"]}, history_root, retire=[old_row],
                                       record=record, previous_current=record["current"], mode="recipe_supersession")
    current_best.execute(state, journal)
    assert not old["path"].exists() and new_public.is_file()
    from lss.bgm import current_best_lineage

    restored = current_best_lineage.restore_previous(state, SHA, history_root)
    assert old["path"].read_bytes() == old_bytes and not new_public.exists()
    assert restored["current"]["pinned"] is True
    assert [item["cue_key"] for item in restored["current"]["files"]] == [old["key"]]
    back = current_best_lineage.restore_previous(state, SHA, history_root)
    assert [item["cue_key"] for item in back["current"]["files"]] == [staged["key"]]
    assert current_best.pcm_sha256(new_public) == staged["pcm_sha256"]


# ------------------------------------------------------------------ invariant 5: one identity, one title


def _scene_cue(data: dict, asset_id: str, name: str, at: str, *, work: str = "work-1") -> None:
    data["cues"][asset_id] = {"asset_id": asset_id, "work_key": work, "name": name, "recorded_at": at,
                              "pcm_sha256": "p" + asset_id[-1], "source_sha256": "s", "episode": 1}


def test_two_different_recordings_never_get_the_same_fixed_name(tmp_path, monkeypatch):
    data = {"cues": {}, "recordings": {}, "works": {}}
    _scene_cue(data, "bgm-cue:1", "ミズウオ大作戦", "2026-01-01T00:00:00+00:00")
    _scene_cue(data, "bgm-cue:2", "ミズウオ大作戦", "2026-01-02T00:00:00+00:00")
    _scene_cue(data, "bgm-cue:3", "ミズウオ大作戦", "2026-01-03T00:00:00+00:00", work="other-work")
    monkeypatch.setattr(scene_names, "update", lambda state_root, change: change(data) or data)
    monkeypatch.setattr(scene_names, "_m_numbers", lambda data, by_recording: {})
    scene_names.settle(tmp_path)
    names = {key: value["name"] for key, value in data["recordings"].items()}
    assert names["asset:bgm-cue:1"] == "ミズウオ大作戦"                  # the first name stays fixed
    assert names["asset:bgm-cue:2"] == "ミズウオ大作戦 その2"
    assert names["asset:bgm-cue:3"] == "ミズウオ大作戦"                  # another work's album is separate
    scene_names.settle(tmp_path)                                         # settling again renames nothing
    assert {key: value["name"] for key, value in data["recordings"].items()} == names


def test_one_recording_keeps_one_title_across_occurrences(tmp_path, monkeypatch):
    data = {"cues": {}, "recordings": {}, "works": {}}
    _scene_cue(data, "bgm-cue:1", "はじめての挑戦", "2026-01-01T00:00:00+00:00")
    _scene_cue(data, "bgm-cue:2", "別の名前", "2026-01-05T00:00:00+00:00")
    memory = {"labels": {"7": {"members": ["bgm-cue:1", "bgm-cue:2"]}}}
    monkeypatch.setattr(scene_names, "update", lambda state_root, change: change(data) or data)
    monkeypatch.setattr(scene_names, "_m_numbers", lambda data, by_recording: {})
    scene_names.settle(tmp_path, memory)
    assert data["recordings"] == {"label:7": data["recordings"]["label:7"]}
    assert data["recordings"]["label:7"]["name"] == "はじめての挑戦"


def test_a_verified_official_title_is_never_replaced_by_a_provisional_one(tmp_path):
    from lss.assets.naming import NamingStore

    store = NamingStore(tmp_path / "names")
    asset = "bgm-cue:k"
    identity = {"cue_key": "k"}
    official = {"official": {"verified": True, "asset_id": asset, "evidence_ref": "release:1",
                             "relation": "exact_version", "title": "公式の曲名"}}
    store.update(asset, official, identity=identity, artifact={})
    for weaker in ({"context": {"asset_id": asset, "scope": "cue", "evidence_ref": "scene:1", "description": "仮"}},
                   {"acoustic": {"status": "measured", "asset_id": asset, "colour": "low", "motion": "steady"}}, {}):
        state = store.update(asset, weaker, identity=identity, artifact={})
        assert state["revisions"][-1]["display_title"] == "公式の曲名" and len(state["revisions"]) == 1


# ------------------------------------------------------------------ invariant 6: UNKNOWN != ABSENT


def test_an_unrebuildable_accepted_cue_is_projected_as_unknown_not_dropped():
    record = {"current": {"files": [], "unavailable": [
        {"public_path": "/x/a.flac", "cue_key": "k", "start_seconds": 1.0, "end_seconds": 9.0, "state": "UNKNOWN",
         "reason": "regenerate_unavailable"}]}}
    projection = results.project(disposition="PUBLISHED", outputs=(), current_best=record, held_count=None)
    assert projection["unavailable_count"] == 1 and projection["unavailable"][0]["state"] == "UNKNOWN"
    assert any(row["reason"] == "accepted_cue_file_unavailable" for row in projection["future_improvement"])
    assert projection["held_count"] is None                              # unknown held count is not shown as 0


def test_a_later_run_keeps_an_unavailable_cue_unknown_until_its_region_is_covered():
    from lss import application

    current = {"unavailable": [{"public_path": "/x/a.flac", "start_seconds": 0.0, "end_seconds": 10.0},
                               {"public_path": "/x/b.flac", "start_seconds": 30.0, "end_seconds": 40.0}]}
    target = [{"start_seconds": 1.0, "end_seconds": 9.0}]
    assert application._still_unavailable(current, target) == {"unavailable": [current["unavailable"][1]]}
    assert application._still_unavailable(None, target) == {}


def test_a_concurrent_second_naming_writer_is_refused_and_writes_nothing(tmp_path):
    from lss.assets.naming import NamingError, NamingStore, _locked

    store = NamingStore(tmp_path / "names")
    asset, identity = "bgm-cue:k", {"cue_key": "k"}
    acoustic = {"acoustic": {"status": "measured", "asset_id": asset, "colour": "low", "motion": "steady"}}
    store.update(asset, acoustic, identity=identity, artifact={})
    before = store.path_for(asset).read_bytes()
    context = {"context": {"asset_id": asset, "scope": "cue", "evidence_ref": "scene:1", "description": "新しい名"}}
    with _locked(store.path_for(asset).with_suffix(".lock")):
        with pytest.raises(NamingError):
            store.update(asset, context, identity=identity, artifact={})
    assert store.path_for(asset).read_bytes() == before
    with pytest.raises(NamingError):                                     # a title change never changes identity
        store.update(asset, context, identity={"cue_key": "other"}, artifact={})
    assert store.path_for(asset).read_bytes() == before
