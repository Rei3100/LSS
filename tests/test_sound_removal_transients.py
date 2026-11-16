"""Short effect removal requires simultaneous evidence, never just short audio."""
import numpy as np
import pytest

from lss.sound import audioset, job, plan, verify

RATE = 1000


def observations(*, n=10, offset=25, parent_offset=None, tag_second=1,
                 target=.1, rival=0., parent_length=100):
    audio = np.full((4 * RATE, 2), .1, np.float32)
    tags = np.zeros((4, 527), np.float32)
    tags[tag_second, audioset.INDEX["Thunder"]] = target
    tags[:, audioset.INDEX["Train"]] = rival
    output = np.zeros(100, np.float32)
    parent = np.zeros(parent_length, np.float32)
    output[offset:offset + n] = .9
    first = offset if parent_offset is None else parent_offset
    parent[first:first + n] = .9
    return audio, tags, verify.Frames(output, parent)


@pytest.mark.parametrize("kwargs,expected", [
    ({}, 10), ({"n": 9}, 9), ({"parent_offset": 50}, 0),
    ({"tag_second": 2}, 0), ({"target": .029}, 0),
    ({"rival": .8}, 0), ({"parent_length": 25}, 0),
    ({"offset": 90, "tag_second": 3}, 10),
])
def test_joint_confirmation_counts_only_observed_coincident_supported_frames(kwargs, expected):
    audio, tags, frames = observations(**kwargs)
    assert verify.removal_coincident_frames(audio, audio, RATE, tags, ("Thunder",), frames) == expected


class Workspace:
    rate = RATE

    def __init__(self, audio):
        self.values = audio

    def audio(self, node):
        return self.values if node == plan.SOURCE else self.values * .5


class Tagger:
    def __init__(self, tags):
        self.tags = tags

    def seconds(self, *_):
        return self.tags.copy()


def route(family="sound", classes=("Thunder",)):
    parent = plan.dnr("effects")
    return plan.Route(plan.text("thunder", parent), parent, classes, family, "extracted", ("thunder",))


@pytest.mark.parametrize("kwargs,removed", [
    ({}, True), ({"n": 9}, False), ({"parent_offset": 50}, False),
    ({"tag_second": 2}, False), ({"target": .029}, False), ({"rival": .8}, False),
])
def test_actual_removal_rescues_only_jointly_confirmed_short_effect(kwargs, removed):
    audio, tags, frames = observations(**kwargs)
    actual, details = job._remove_verified(Workspace(audio), [route()], [frames], Tagger(tags), None)
    assert (details["outcome"] == "removed") is removed
    if removed:
        assert "removal_short_effect_jointly_confirmed" in details["targets"][0]["reasons"]
        assert np.any(actual != audio)
        # The existing 50ms fade is allowed at the supported second boundary.
        np.testing.assert_array_equal(actual[:RATE - 50], audio[:RATE - 50])
        np.testing.assert_array_equal(actual[2 * RATE + 50:], audio[2 * RATE + 50:])


def test_joint_evidence_outside_requested_window_does_not_authorize_removal():
    from lss.sound import intent

    audio, tags, frames = observations()
    window = intent.parse_one("最後の1秒の雷なし").window
    _, details = job._remove_verified(Workspace(audio), [route()], [frames], Tagger(tags), window)
    assert details["outcome"] != "removed"


@pytest.mark.parametrize("family,classes", [("music_part", ("Thunder",)), ("sound", ())])
def test_short_effect_path_does_not_broaden_instruments_or_open_phrases(family, classes):
    audio, tags, frames = observations()
    _, details = job._remove_verified(Workspace(audio), [route(family, classes)], [frames], Tagger(tags), None)
    assert details["outcome"] == "held"


def test_joint_support_still_requires_audibility_relative_to_parent():
    audio, tags, frames = observations()
    assert verify.removal_coincident_frames(audio * 1e-3, audio, RATE, tags, ("Thunder",), frames) == 0
