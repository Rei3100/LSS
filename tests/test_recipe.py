import pytest

from lss.core.recipe import (
    separation_recipe_revision, target_absence_recipe_revision, validate_recipe_revision,
)


def _revision(**overrides):
    values = {
        "provider": "msst",
        "arch": "mel_band_roformer",
        "model_id": "model",
        "model_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "runtime_id": "c" * 64,
        "expected_stems": ["Vocals", "Instrumental"],
        "output_samplerate": 44100,
        "input_preparation": {"method": "libsndfile_direct", "policy_revision": 1},
        "serialization": {"container": "FLAC", "subtype": "PCM_24", "policy_revision": 2},
    }
    values.update(overrides)
    return separation_recipe_revision(**values)


@pytest.mark.parametrize("field,value", [
    ("provider", "other"),
    ("arch", "bs_roformer"),
    ("model_sha256", "d" * 64),
    ("config_sha256", "e" * 64),
    ("runtime_id", "f" * 64),
    ("expected_stems", ["Instrumental"]),
    ("published_stems", ["Instrumental"]),
    ("output_samplerate", 48000),
    ("input_preparation", {"method": "ffmpeg_pcm_f32le", "policy_revision": 1}),
    ("serialization", {"container": "FLAC", "subtype": "PCM_16", "policy_revision": 2}),
])
def test_recipe_identity_changes_for_every_output_affecting_choice(field, value):
    assert _revision(**{field: value})["recipe_id"] != _revision()["recipe_id"]


def test_recipe_revision_rejects_body_changed_without_new_identity():
    revision = _revision()
    revision["stage_graph"][0]["model_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="一致しません"):
        validate_recipe_revision(revision)


def test_publication_identity_uses_resolved_names_and_stable_set_order():
    assert _revision(published_stems=["vocals", "instrumental"]) == _revision(
        published_stems=["Instrumental", "Vocals"],
    )
    assert _revision(published_stems=["instrumental"])["publication"] == {
        "stems": ["Instrumental"],
    }


@pytest.mark.parametrize("stems", [[], ["Guitar"], ["Vocals", "vocals"], [None]])
def test_invalid_publication_set_cannot_become_a_recipe(stems):
    with pytest.raises(ValueError, match="公開パート"):
        _revision(published_stems=stems)


def test_target_absence_recipe_round_trips_and_rejects_tampering():
    revision = target_absence_recipe_revision(
        requested_target="Instrumental",
        input_preparation={"method": "libsndfile_direct", "policy_revision": 1},
        validation={"method": "exact_digital_silence", "threshold": "none"},
    )

    assert validate_recipe_revision(revision) == revision
    assert revision["recipe_id"].startswith("target-absence-v1-")
    revision["stage_graph"][0]["requested_target"] = "Vocals"
    with pytest.raises(ValueError, match="一致しません"):
        validate_recipe_revision(revision)


def test_unknown_recipe_capability_is_not_accepted_even_with_separation_prefix():
    revision = _revision()
    revision["capability"] = "unknown"
    with pytest.raises(ValueError, match="capability"):
        validate_recipe_revision(revision)
