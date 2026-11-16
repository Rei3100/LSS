from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.assets import naming
from lss.bgm import cue_publication, provisional_naming
from scripts.name_bgm_assets import run

ASSET = "bgm-cue:one-occurrence"
IDENTITY = {"cue_key": "one-occurrence", "cue_identity": "unresolved",
            "revision_lineage": ["original-r1"], "current_best": "original-r1"}


def official(asset=ASSET, title="公式曲名", **overrides):
    return {"official": {"asset_id": asset, "title": title, "verified": True,
                         "relation": "exact_version", "evidence_ref": "local:verified-cue-sheet", **overrides}}


def test_single_occurrence_fallback_is_honest_stable_and_not_official():
    first = naming.propose(ASSET, {})
    assert first == naming.propose(ASSET, {})
    assert first["kind"] == "provisional_fallback" and not first["is_official"]
    assert "名称未詳" in first["title"] and "仮題" in first["display_title"]
    assert first["display_title"] != naming.propose("another-cue", {})["display_title"]


@pytest.mark.parametrize("changes", [dict(verified=False), dict(asset_id="another"),
                                     dict(relation="same_theme"), dict(evidence_ref="")])
def test_unverified_or_wrong_version_official_title_is_not_authoritative(changes):
    assert naming.propose(ASSET, official(**changes))["kind"] == "provisional_fallback"


def test_verified_official_wins_over_rich_provisional_evidence():
    evidence = {**official(), "context": {"asset_id": ASSET, "scope": "cue",
                                         "description": "場面の曲", "evidence_ref": "local:scene"}}
    result = naming.propose(ASSET, evidence)
    assert result["is_official"] and result["kind"] == "official"
    assert result["display_title"] == "公式曲名"


def test_verified_relation_does_not_claim_official_version_title():
    result = naming.propose(ASSET, {"related": {**official()["official"], "relation": "same_theme"}})
    assert result["kind"] == "provisional_related" and not result["is_official"]
    assert "関連版" in result["title"] and "仮題" in result["display_title"]


def test_episode_text_does_not_invent_cue_scene():
    evidence = {"context": {"asset_id": ASSET, "scope": "episode", "description": "決戦",
                            "evidence_ref": "local:episode-title"}}
    assert naming.propose(ASSET, evidence)["kind"] == "provisional_fallback"
    evidence["context"]["scope"] = "cue"
    assert naming.propose(ASSET, evidence)["title"] == "決戦"


def test_title_upgrade_keeps_identity_currentbest_and_full_history(tmp_path):
    store = naming.NamingStore(tmp_path)
    artifact = {"sha256": "a" * 64, "publication_provenance": {"recipe": "old-recipe"}}
    first = store.update(ASSET, {}, identity=IDENTITY, artifact=artifact)
    initial_bytes = store.path_for(ASSET).read_bytes()
    again = store.update(ASSET, {}, identity=IDENTITY, artifact={"path": "moved"})
    assert first == again and initial_bytes == store.path_for(ASSET).read_bytes()
    richer = {"context": {"asset_id": ASSET, "scope": "cue", "description": "旅の場面",
                          "evidence_ref": "local:scene-caption"}}
    middle = store.update(ASSET, richer, identity=IDENTITY, artifact=artifact)
    latest = store.update(ASSET, official(), identity=IDENTITY, artifact=artifact)
    assert latest["identity"] == first["identity"] == IDENTITY
    assert len(latest["revisions"]) == 3
    assert latest["revisions"][:2] == middle["revisions"]
    assert latest["revisions"][0] == first["revisions"][0]
    assert latest["revisions"][-1]["artifact"] == artifact
    assert store.read(ASSET) == latest
    assert store.update(ASSET, {}, identity=IDENTITY, artifact=artifact) == latest


def test_rename_or_policy_change_only_appends_presentation_revision(tmp_path):
    store = naming.NamingStore(tmp_path)
    first = store.update(ASSET, official(), identity=IDENTITY, artifact={})
    second = store.update(ASSET, official(title="訂正曲名"), identity=IDENTITY, artifact={})
    third = store.update(ASSET, official(title="訂正曲名"), identity=IDENTITY, artifact={}, policy_id="next-policy")
    assert first["identity"] == second["identity"] == third["identity"]
    assert len(third["revisions"]) == 3
    assert third["revisions"][0]["title"] == "公式曲名"


def test_identity_change_is_refused_without_touching_state(tmp_path):
    store = naming.NamingStore(tmp_path)
    store.update(ASSET, {}, identity=IDENTITY, artifact={})
    before = store.path_for(ASSET).read_bytes()
    with pytest.raises(naming.NamingError, match="CurrentBest"):
        store.update(ASSET, official(), identity={**IDENTITY, "current_best": "other"}, artifact={})
    assert store.path_for(ASSET).read_bytes() == before


@pytest.mark.parametrize("part", ["history", "identity", "current"])
def test_corrupt_history_is_not_silently_replaced(tmp_path, part):
    store = naming.NamingStore(tmp_path)
    state = store.update(ASSET, {}, identity=IDENTITY, artifact={})
    if part == "history":
        state["revisions"][0]["title"] = "corrupt"
    elif part == "identity":
        state["identity"]["current_best"] = "corrupt"
    else:
        state["current_revision_id"] = "corrupt"
    store.path_for(ASSET).write_text(json.dumps(state), encoding="utf-8")
    before = store.path_for(ASSET).read_bytes()
    with pytest.raises(naming.NamingError):
        store.update(ASSET, {}, identity=IDENTITY, artifact={})
    assert store.path_for(ASSET).read_bytes() == before


def test_interrupted_write_preserves_prior_title_and_retry_recovers(tmp_path, monkeypatch):
    store = naming.NamingStore(tmp_path)
    first = store.update(ASSET, {}, identity=IDENTITY, artifact={})
    original = naming._write_json_atomic
    def fail(*args):
        raise OSError("simulated disk full")
    monkeypatch.setattr(naming, "_write_json_atomic", fail)
    with pytest.raises(OSError):
        store.update(ASSET, official(), identity=IDENTITY, artifact={})
    assert store.read(ASSET) == first
    monkeypatch.setattr(naming, "_write_json_atomic", original)
    assert len(store.update(ASSET, official(), identity=IDENTITY, artifact={})["revisions"]) == 2


def test_concurrent_writer_refuses_to_overwrite(tmp_path):
    store = naming.NamingStore(tmp_path)
    with naming._locked(store.path_for(ASSET).with_suffix(".lock")):
        with pytest.raises(naming.NamingError, match="書込中"):
            store.update(ASSET, {}, identity=IDENTITY, artifact={})
    assert not store.path_for(ASSET).exists()


def test_killed_writer_releases_lock_and_retry_needs_no_cleanup(tmp_path):
    store = naming.NamingStore(tmp_path)
    store.update(ASSET, {}, identity=IDENTITY, artifact={})
    code = ("import sys; from pathlib import Path; from lss.assets.naming import _locked; "
            "ctx=_locked(Path(sys.argv[1])); ctx.__enter__(); print('locked', flush=True); sys.stdin.read()")
    child = subprocess.Popen([sys.executable, "-u", "-c", code, str(store.path_for(ASSET).with_suffix('.lock'))],
                             cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        assert child.stdout.readline().strip() == b"locked"
        with pytest.raises(naming.NamingError, match="書込中"):
            store.update(ASSET, official(), identity=IDENTITY, artifact={})
    finally:
        child.terminate()
        child.communicate(timeout=10)
    assert len(store.update(ASSET, official(), identity=IDENTITY, artifact={})["revisions"]) == 2


def test_richer_cue_context_survives_later_acoustic_only_processing(tmp_path):
    store = naming.NamingStore(tmp_path)
    context = {"context": {"asset_id": ASSET, "scope": "cue", "description": "旅の場面",
                           "evidence_ref": "local:cue-text"}}
    state = store.update(ASSET, context, identity=IDENTITY, artifact={})
    weaker = {"acoustic": {"asset_id": ASSET, "status": "measured", "colour": "low", "motion": "steady"}}
    assert store.update(ASSET, weaker, identity=IDENTITY, artifact={}) == state
    stronger = store.update(ASSET, official(), identity=IDENTITY, artifact={})
    assert stronger["revisions"][0]["title"] == "旅の場面"


def test_midrange_tone_is_not_described_as_broad_spectrum(tmp_path):
    item = _published(tmp_path, frequency=1000)
    row = provisional_naming.name_published(item["path"], naming.NamingStore(tmp_path / "names"))
    assert "中域が際立つ" in row["title"] and "広い" not in row["title"]


def _published(tmp_path, *, frequency=220, varying=False, silent=False, seconds=10, anti_phase=False):
    rate = 16000
    t = np.arange(rate * seconds) / rate
    amplitude = (0.02 + 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 5)) ** 2) if varying else 0.2
    mono = amplitude * np.sin(2 * np.pi * frequency * t)
    audio = np.stack([mono, -mono if anti_phase else mono], axis=1)
    if silent:
        audio[:] = 0
    provenance = {"type": "bgm_published_cue", "source": {"sha256": "a" * 64},
                  "start_frame": 0, "end_frame": len(audio), "sample_rate": rate,
                  "identity_state": "unresolved_no_registered_reference"}
    path = tmp_path / "cue.flac"
    return cue_publication.publish(audio, rate, path, key="one-occurrence", title="ep BGM 01",
                                   album="ep BGM", provenance=provenance,
                                   input_sha256="a" * 64, recipe_id="old", version="old")


@pytest.mark.parametrize("frequency,colour", [(220, "low"), (3500, "high"), (1000, "middle")])
def test_acoustic_controls_distinguish_frequency_content(tmp_path, frequency, colour):
    item = _published(tmp_path, frequency=frequency)
    observed = provisional_naming.acoustic_evidence(item["path"], ASSET)
    assert observed["status"] == "measured" and observed["colour"] == colour
    assert observed["motion"] == "steady"


def test_acoustic_controls_distinguish_envelope_without_cancelling_stereo(tmp_path):
    item = _published(tmp_path, varying=True, anti_phase=True)
    observed = provisional_naming.acoustic_evidence(item["path"], ASSET)
    assert observed["colour"] == "low" and observed["motion"] == "varying"


@pytest.mark.parametrize("options", [dict(silent=True), dict(seconds=1)])
def test_insufficient_audio_has_fallback_not_musical_fabrication(tmp_path, options):
    item = _published(tmp_path, **options)
    row = provisional_naming.name_published(item["path"], naming.NamingStore(tmp_path / "names"))
    assert row["kind"] == "provisional_fallback"


def test_real_publication_format_to_persisted_naming_playlist_without_asset_mutation(tmp_path):
    item = _published(tmp_path)
    path = tmp_path / "cue.flac"
    before = path.read_bytes()
    report = run([path], tmp_path / "presentation")
    row = report["rows"][0]
    assert row["kind"] == "provisional_descriptive" and "仮題" in row["title"]
    assert path.read_bytes() == before
    assert "LSS_BGM_CUE_KEY" in FLAC(path) and item["sha256"] == provisional_naming.audio_identity.file_digest(path)
    assert row["title"] in (tmp_path / "presentation" / "BGM.m3u8").read_text(encoding="utf-8")
    store = naming.NamingStore(tmp_path / "presentation" / "names")
    first = store.read(ASSET)
    assert first["revisions"][0]["artifact"]["publication_provenance"]["identity_state"].startswith("unresolved")
    renamed = run([path], tmp_path / "presentation", evidence_by_asset={ASSET: official()})
    assert renamed["rows"][0]["is_official"] and renamed["rows"][0]["title"] == "公式曲名"
    assert path.read_bytes() == before
    assert len(store.read(ASSET)["revisions"]) == 2
    assert run([path], tmp_path / "presentation")["rows"][0]["title"] == "公式曲名"


def test_copy_or_move_does_not_change_name_or_create_revision(tmp_path):
    item = _published(tmp_path)
    store = naming.NamingStore(tmp_path / "names")
    first = provisional_naming.name_published(item["path"], store)
    moved = tmp_path / "renamed.flac"
    shutil.copyfile(item["path"], moved)
    again = provisional_naming.name_published(moved, store)
    assert again["title"] == first["title"] and again["revision_id"] == first["revision_id"]
    assert len(store.read(ASSET)["revisions"]) == 1


def test_non_publication_and_inconsistent_provenance_are_rejected(tmp_path):
    path = tmp_path / "raw.flac"
    sf.write(path, np.zeros((100, 2)), 16000)
    with pytest.raises(naming.NamingError):
        provisional_naming.name_published(path, naming.NamingStore(tmp_path / "names"))
    item = _published(tmp_path)
    tags = FLAC(item["path"])
    provenance = json.loads(tags["LSS_BGM_PROVENANCE"][0])
    provenance["end_frame"] += 1
    tags["LSS_BGM_PROVENANCE"] = json.dumps(provenance)
    tags.save()
    with pytest.raises(naming.NamingError, match="時間軸"):
        provisional_naming.name_published(item["path"], naming.NamingStore(tmp_path / "names"))


def test_source_change_during_observation_does_not_publish_name(tmp_path, monkeypatch):
    item = _published(tmp_path)
    store = naming.NamingStore(tmp_path / "names")
    original = provisional_naming.acoustic_evidence
    def changed(path, asset):
        observed = original(path, asset)
        tags = FLAC(path)
        tags["TITLE"] = "changed"
        tags.save()
        return observed
    monkeypatch.setattr(provisional_naming, "acoustic_evidence", changed)
    with pytest.raises(naming.NamingError, match="変更"):
        provisional_naming.name_published(item["path"], store)
    assert not store.path_for(ASSET).exists()


def test_external_evidence_cannot_replace_actual_local_measurement(tmp_path):
    item = _published(tmp_path, frequency=220)
    row = provisional_naming.name_published(
        item["path"], naming.NamingStore(tmp_path / "names"),
        extra_evidence={"acoustic": {"asset_id": ASSET, "status": "measured", "colour": "high", "motion": "varying"}})
    assert "低域が厚い" in row["title"] and "なだらかな" in row["title"]


@pytest.mark.parametrize("bad", ["", "bad\nname", "x" * 201])
def test_title_rejects_playlist_injection_and_unbounded_text(bad):
    with pytest.raises(naming.NamingError):
        naming.propose(ASSET, official(title=bad))
