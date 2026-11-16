"""Pinned upstream AudioSet class indices used by BGM tagging semantics."""

import numpy as np

from lss.bgm import cue_semantics, dialogue_residual, event_tagging


# AudioSet ontology labels: Google, CC BY-SA 4.0; see THIRD_PARTY_NOTICES.md.
# qiuqiangkong/audioset_tagging_cnn metadata/class_labels_indices.csv at
# d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4 (MIT upstream implementation).
UPSTREAM_PIN = "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4/metadata/class_labels_indices.csv"


def test_named_event_classes_match_pinned_upstream_rows():
    expected = {
        "Speech": 0,
        "Singing": 27,
        "Choir": 28,
        "Yodeling": 29,
        "Chant": 30,
        "Mantra": 31,
        "Male singing": 32,
        "Female singing": 33,
        "Child singing": 34,
        "Synthetic singing": 35,
        "Rapping": 36,
        "Humming": 37,
        "Animal": 72,
        "Music": 137,
        "Vocal music": 254,
        "A capella": 255,
        "Song": 266,
        "Background music": 267,
        "Theme music": 268,
        "Jingle (music)": 269,
        "Soundtrack music": 270,
        "Video game music": 272,
        "Explosion": 426,
        "Sound effect": 504,
    }
    assert {name: event_tagging.CLASSES[name] for name in expected} == expected


def test_musical_voice_set_uses_upstream_ids_and_excludes_video_game_music():
    assert cue_semantics.SINGING_CLASSES == (27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 254, 255)
    assert 272 not in cue_semantics.SINGING_CLASSES


def test_from_arrays_observes_upstream_female_and_synthetic_singing_rows():
    probabilities = np.zeros((2, 527), dtype=np.float32)
    probabilities[0, 33] = 0.8  # Female singing
    probabilities[1, 35] = 0.7  # Synthetic singing
    evidence = cue_semantics.WindowEvidence.from_arrays(
        np.full(2, -20.0), probabilities, np.ones((2, 4), dtype=np.float32)
    )
    np.testing.assert_allclose(evidence.singing_probability, [0.8, 0.7])


def test_dialogue_plan_protects_actual_voice_rows_but_not_video_game_music():
    for class_index in (33, 35, 36, 254):
        probabilities = np.zeros((1, 527), dtype=np.float32)
        probabilities[0, class_index] = 0.9
        plan = dialogue_residual.processing_plan(
            probabilities, 0, 48000, 48000, dialogue_residual.default_recipe()
        )
        assert plan["musical_voice_regions"] == [[0, 48000]]
        assert plan["regions"] == []

    probabilities = np.zeros((1, 527), dtype=np.float32)
    probabilities[0, 272] = 0.9
    plan = dialogue_residual.processing_plan(
        probabilities, 0, 48000, 48000, dialogue_residual.default_recipe()
    )
    assert plan["musical_voice_regions"] == []
