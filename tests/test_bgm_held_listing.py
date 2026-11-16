"""生成せず一覧を出し、既存入力と同時保存先・遅い入力変更を上書きしない境界。"""
import json
from pathlib import Path

import pytest

from lss.assets import held_catalog
from scripts import list_bgm_held_candidates as cli
from tests.test_application_bgm_reference_residual import bundle, case, execute


def test_listing_keeps_primary_and_two_alternatives_in_one_unpublished_occurrence(bundle,tmp_path):
    outcome = execute(bundle);before = outcome.record_path.read_bytes();output = tmp_path / "listing.json"
    assert cli.main([str(outcome.record_path),"--out",str(output)]) == 0
    payload = json.loads(output.read_bytes())
    assert payload["counts"] == {"held_results":1,"logical_occurrences":1,"held_representations":3,"published_outputs":0}
    cue = payload["results"][0]["cues"][0]
    assert cue["origin"]["time_offset_numerator"] == 7
    assert len(cue["alternatives"]) == 2 and all(item["primary_candidate_id"] == cue["primary_candidate_id"] for item in cue["alternatives"])
    assert payload["publication_allowed"] is False and payload["published_outputs"] == []
    assert outcome.record_path.read_bytes() == before


@pytest.mark.parametrize("target",["record","plan","source","candidate"])
def test_export_refuses_existing_inputs_and_preserves_them(bundle,target):
    outcome = execute(bundle)
    destination = {"record":outcome.record_path,"plan":bundle[1],"source":bundle[0].sources[0],
                   "candidate":outcome.held_artifacts[0]}[target]
    before = destination.read_bytes()
    assert cli.main([str(outcome.record_path),"--out",str(destination)]) == 1
    assert destination.read_bytes() == before


def test_late_input_mutation_leaves_no_export(bundle,tmp_path,monkeypatch):
    outcome = execute(bundle);output = tmp_path / "listing.json";verify = held_catalog.verify
    snapshots = held_catalog.snapshots([outcome.record_path])
    def changed(snapshots):
        audio = outcome.held_artifacts[0];audio.write_bytes(audio.read_bytes()+b"late")
        verify(snapshots)
    monkeypatch.setattr(held_catalog,"verify",changed)
    with pytest.raises(held_catalog.CatalogHeldError):
        cli.export_new(output,cli.listing(snapshots),snapshots)
    assert not output.exists()
    assert not list(tmp_path.glob(".held-listing-*.tmp"))


def test_concurrently_created_export_is_not_replaced(bundle,tmp_path,monkeypatch):
    outcome = execute(bundle);snapshots = held_catalog.snapshots([outcome.record_path]);output = tmp_path / "listing.json"
    newer = b'{"newer_listing":true}\n';verify = held_catalog.verify
    def concurrent(records):
        verify(records);output.write_bytes(newer)
    monkeypatch.setattr(held_catalog,"verify",concurrent)
    with pytest.raises(FileExistsError):
        cli.export_new(output,cli.listing(snapshots),snapshots)
    assert output.read_bytes() == newer


def test_unavailable_reference_is_listed_with_parent_and_declared_evidence(bundle,tmp_path):
    outcome = execute(bundle);outcome.reference_residual_evidence_path.write_bytes(b"broken")
    output = tmp_path / "listing.json"
    assert cli.main([str(outcome.record_path),"--out",str(output)]) == 0
    result = json.loads(output.read_bytes())
    assert result["counts"]["logical_occurrences"] == 1 and result["counts"]["held_representations"] == 2
    assert result["results"][0]["unavailable_evidence"] == ["reference_residual"]
