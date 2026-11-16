"""公開するcueごとの表現（原音か処理済みか）の選択（`lss.bgm.cue_representation`）。"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lss.bgm import cue_representation as cr

RECIPE = cr.default_recipe()


def _evidence(residual, speech):
    probabilities = np.zeros((len(residual), 527), dtype=np.float32)
    probabilities[:, cr.SPEECH_CLASS] = speech
    return cr.RoutingEvidence.from_arrays(np.asarray(residual, dtype=float), probabilities)


def test_mixture_that_equals_the_music_is_published_raw():
    route = cr.choose(10, 40, _evidence(np.full(60, -35.0), np.full(60, 0.05)), RECIPE)
    assert route["state"] == cr.STATE_RAW and route["representation"] == cr.RAW and not route["fallback"]


def test_material_the_separator_removes_keeps_the_processed_representation():
    residual = np.full(60, -35.0)
    residual[25] = -3.0  # 1秒だけ台詞・効果音がある
    route = cr.choose(10, 40, _evidence(residual, np.full(60, 0.05)), RECIPE)
    assert route["state"] == cr.STATE_PROCESSED and "separator_removes_material" in route["reasons"]


def test_speech_heard_in_the_mixture_is_not_published_raw():
    speech = np.full(60, 0.05)
    speech[30] = 0.9
    route = cr.choose(10, 40, _evidence(np.full(60, -35.0), speech), RECIPE)
    assert route["state"] == cr.STATE_PROCESSED and "speech_in_mixture" in route["reasons"]


def test_between_the_thresholds_falls_back_to_the_known_good_representation():
    route = cr.choose(10, 40, _evidence(np.full(60, -15.0), np.full(60, 0.05)), RECIPE)
    assert route["state"] == cr.STATE_UNCERTAIN and route["representation"] == cr.KNOWN_GOOD and route["fallback"]


def test_recipe_identity_and_validation():
    assert dataclasses.replace(RECIPE, raw_maximum_residual_db=-25.0).recipe_id != RECIPE.recipe_id
    with pytest.raises(cr.CueRepresentationError):
        dataclasses.replace(RECIPE, processed_minimum_residual_db=-30.0).validate()
    with pytest.raises(cr.CueRepresentationError):
        cr.RoutingEvidence.from_arrays(np.array([np.nan]), np.zeros((1, 527)))
