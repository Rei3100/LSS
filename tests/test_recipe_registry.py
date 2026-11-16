"""Recipe本文をIDで引ける形で残し、成果物のタグに頼らず来歴をたどる。"""
from __future__ import annotations

import json

import pytest

from lss.core import recipe, recipe_registry


def revision(**changes) -> dict:
    base = {"provider": "msst", "arch": "mel_band_roformer", "model_id": "model_x",
            "model_sha256": "m" * 64, "config_sha256": "c" * 64, "runtime_id": "runtime-1",
            "expected_stems": ["Instrumental", "Vocals"], "output_samplerate": 44100,
            "input_preparation": {"resample": "none"},
            "serialization": {"subtype": "PCM_24", "clipping": "none"}}
    return recipe.separation_recipe_revision(**{**base, **changes})


def test_a_recorded_recipe_is_read_back_and_verified_against_its_id(tmp_path):
    body = revision()
    path = recipe_registry.record(body, state_root=tmp_path)
    assert path.name == f"{body['recipe_id']}.json"
    assert recipe_registry.revision_for(body["recipe_id"], state_root=tmp_path) == body
    identity = recipe_registry.processing_identity(body["recipe_id"], state_root=tmp_path)
    assert identity == {"model_id": "model_x", "model_sha256": "m" * 64, "config_sha256": "c" * 64}


def test_recording_the_same_recipe_twice_keeps_one_record(tmp_path):
    body = revision()
    first = recipe_registry.record(body, state_root=tmp_path)
    stamp = first.stat().st_mtime_ns
    assert recipe_registry.record(body, state_root=tmp_path) == first
    assert first.stat().st_mtime_ns == stamp


def test_a_body_that_does_not_match_its_id_is_never_recorded(tmp_path):
    body = revision()
    body["model_id"] = "swapped"
    with pytest.raises(recipe_registry.RecipeRegistryError, match="Recipe本文"):
        recipe_registry.record(body, state_root=tmp_path)
    assert not recipe_registry.registry_dir(tmp_path).exists()


def test_a_tampered_record_is_not_read_back(tmp_path):
    body = revision()
    path = recipe_registry.record(body, state_root=tmp_path)
    stored = json.loads(path.read_bytes())
    stored["stage_graph"][0]["model_id"] = "swapped"
    path.write_text(json.dumps(stored), encoding="utf-8")
    assert recipe_registry.revision_for(body["recipe_id"], state_root=tmp_path) is None
    assert recipe_registry.processing_identity(body["recipe_id"], state_root=tmp_path) is None


def test_a_record_stored_under_another_id_is_not_read_back(tmp_path):
    body = revision()
    recipe_registry.record(body, state_root=tmp_path)
    other = revision(model_id="model_y")
    path = recipe_registry.registry_dir(tmp_path) / f"{other['recipe_id']}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    assert recipe_registry.revision_for(other["recipe_id"], state_root=tmp_path) is None


def test_a_different_body_under_the_same_id_is_refused(tmp_path):
    body = revision()
    path = recipe_registry.record(body, state_root=tmp_path)
    changed = dict(body)
    changed["schema_version"] = body["schema_version"]
    path.write_text(json.dumps({**body, "runtime_id": "other"}), encoding="utf-8")
    with pytest.raises(recipe_registry.RecipeRegistryError, match="別の本文"):
        recipe_registry.record(body, state_root=tmp_path)


@pytest.mark.parametrize("recipe_id", ["", "../escape", "a/b", "x" * 300, None, 3])
def test_unsafe_identifiers_are_refused(tmp_path, recipe_id):
    assert recipe_registry.revision_for(recipe_id, state_root=tmp_path) is None
    assert recipe_registry.processing_identity(recipe_id, state_root=tmp_path) is None


def test_a_missing_record_is_simply_absent(tmp_path):
    assert recipe_registry.revision_for("separation-v1-" + "a" * 64, state_root=tmp_path) is None
