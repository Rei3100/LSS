"""An optional cleanup stage must not remove a positively observed musical voice."""
from dataclasses import replace

import numpy as np
import pytest

from lss.bgm import cue_semantics as cs


@pytest.mark.parametrize('voice_class', cs.SINGING_CLASSES)
def test_one_observed_sung_window_protects_the_whole_materialization_candidate(voice_class):
    tags = np.zeros((20, 527), dtype=np.float32)
    tags[:, cs.MUSIC_CLASS] = .9
    tags[7, voice_class] = .6
    window = cs.WindowEvidence.from_arrays(np.full(20, -20.), tags, np.ones((20, 8)))
    cues = [{'cue_id': 'sung', 'start_seconds': 5.5, 'end_seconds': 10.2},
            {'cue_id': 'plain', 'start_seconds': 11., 'end_seconds': 19.}]
    selected, receipt = cs.voice_preserving_refinement(cues, {'sung','plain'}, window, cs.default_recipe())
    assert selected == ['plain']
    assert receipt['protected'] == [{'cue_id': 'sung', 'observed_windows': [7],
                                      'reason': 'retain_first_pass_musical_voice'}]
    assert receipt['original_mixture_authority'] == receipt['absence_authority'] == 'none'
    legacy, _ = cs.voice_preserving_refinement(cues, {'sung','plain'}, window,
        replace(cs.default_recipe(), vocal_policy='legacy_exclusion_v1'))
    assert legacy == ['plain', 'sung']


def test_voice_outside_a_candidate_does_not_disable_unrelated_cleanup():
    window = cs.WindowEvidence(np.full(20, -20.), np.full(20, .9), np.r_[.6, np.zeros(19)], np.ones((20,8)))
    selected, receipt = cs.voice_preserving_refinement(
        [{'cue_id': 'plain', 'start_seconds': 5., 'end_seconds': 10.}], {'plain'}, window, cs.default_recipe())
    assert selected == ['plain'] and receipt['protected'] == []
