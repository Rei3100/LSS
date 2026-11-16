"""番組一本の中で同じ録音が二度鳴ることを、参照も別の回も無しに見つける。"""
from __future__ import annotations

import numpy as np
import pytest

from lss.bgm import self_recurrence
from lss.bgm.cue_extent import CueExtentRecipe
from lss.bgm.recurring_extent import read_recurring_extent, recurring_matches
from tests.test_bgm_identity import RATE, _dialogue_and_noise, _music


def _episode(seconds: float, placements) -> np.ndarray:
    """台詞と雑音の下に、指定した時刻へ指定の録音を置いた一本の番組。"""
    mix = _dialogue_and_noise(seconds, seed=7) * 0.3
    for start, signal in placements:
        begin = int(start * RATE)
        mix[begin:begin + signal.size] += signal
    return mix.astype(np.float32)


def test_the_same_recording_played_twice_is_found_without_a_reference():
    cue = _music(11, 25.0)
    mix = _episode(300.0, [(30.0, cue), (200.0, cue)])

    found = self_recurrence.repeated_recordings(mix)

    assert found, "同じ録音の再演が見つからない"
    best = max(found, key=lambda item: item.blocks)
    assert abs(best.first_start_seconds - 30.0) <= 2.0
    assert abs(best.second_start_seconds - 200.0) <= 2.0
    assert best.first_start_seconds < best.second_start_seconds


def test_a_source_without_a_repeat_reports_nothing():
    mix = _episode(300.0, [(30.0, _music(12, 25.0)), (200.0, _music(13, 25.0))])

    assert self_recurrence.repeated_recordings(mix) == []


def test_the_trivial_match_with_itself_is_not_a_repeat():
    mix = _episode(300.0, [(30.0, _music(14, 25.0))])

    assert self_recurrence.repeated_recordings(mix) == []


def test_a_repeat_gives_both_appearances_a_readable_extent():
    cue = _music(15, 25.0)
    mix = _episode(300.0, [(30.0, cue), (200.0, cue)])
    found = self_recurrence.repeated_recordings(mix)
    spans = self_recurrence.spans_for_source(found, source_key="ep1")
    matches = recurring_matches(spans)
    recipe = CueExtentRecipe(minimum_similarity=0.15,
                             minimum_single_recording_coverage=0.8,
                             maximum_offset_spread_seconds=0.5)

    first = [row for row in matches if row["query_start_seconds"] < 100.0]
    second = [row for row in matches if row["query_start_seconds"] >= 100.0]
    assert first and second
    # 二度の登場は同じ録音なので、同じ呼び名になる。
    assert {row["reference_id"] for row in matches} == {row["reference_id"] for row in first}
    start, end = first[0]["query_start_seconds"], first[0]["query_end_seconds"]
    assert read_recurring_extent(start, end, first, recipe)["state"] == "single_cue"


@pytest.mark.parametrize("signal,separation", [
    (np.zeros((2, 100), np.float32), 20.0),
    (np.zeros(0, np.float32), 20.0),
    (np.zeros(100, np.float32), -1.0),
    (np.zeros(100, np.float32), True),
])
def test_a_malformed_request_is_refused(signal, separation):
    with pytest.raises(self_recurrence.SelfRecurrenceError):
        self_recurrence.repeated_recordings(signal, minimum_separation_seconds=separation)


def test_spans_need_a_source_key():
    with pytest.raises(self_recurrence.SelfRecurrenceError):
        self_recurrence.spans_for_source([], source_key=" ")


def test_the_same_repeat_read_with_two_alignments_is_counted_once(monkeypatch):
    from lss.bgm.identity import SharedOccurrence

    pairs = [SharedOccurrence(30.0, 55.0, 200.0, 225.0, 0.4, 4),
             SharedOccurrence(30.2, 55.2, 200.2, 225.2, 0.9, 6),
             SharedOccurrence(200.1, 225.1, 30.1, 55.1, 0.5, 5),
             SharedOccurrence(600.0, 630.0, 900.0, 930.0, 0.3, 3)]
    monkeypatch.setattr(self_recurrence.identity, "shared_recordings",
                        lambda *args, **kwargs: pairs)

    kept = self_recurrence.repeated_recordings(np.zeros(1000, np.float32))

    # 同じ再演の三通りの読み（向き違いを含む）は一件にまとまり、別の再演は残る。
    assert [(item.first_start_seconds, item.blocks) for item in kept] == [(30.2, 6), (600.0, 3)]


def test_a_source_shorter_than_one_chunk_is_searched_once(monkeypatch):
    calls = []
    monkeypatch.setattr(self_recurrence.identity, "shared_recordings",
                        lambda *a, **k: calls.append(1) or [])

    self_recurrence.repeated_recordings(np.zeros(100, np.float32), chunk_seconds=120.0)

    assert len(calls) == 1
