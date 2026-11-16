"""完了後の動作の実行。凍結された4択の配線だけを確かめる。

本当の再起動・終了は検証で起こさない。実行体を差し替えて要求だけを見る。
"""
from __future__ import annotations

import pytest

from lss.ui.completion import Completion, Outcome, command_for
from lss.ui.presentation import WHEN_DONE_LABEL, WhenDone, when_done_choices


def _spy():
    calls: list = []
    return calls, Completion(
        runner=lambda command: (calls.append(command), 0)[1],
        quit_app=lambda: calls.append("quit"),
    )


def test_the_choices_stay_the_frozen_four() -> None:
    assert [choice.value for choice in when_done_choices()] == [
        "none", "quit", "restart", "shutdown",
    ]
    # 廃止されたスリープを復活させない。
    assert "sleep" not in {choice.value for choice in WhenDone}
    assert set(WHEN_DONE_LABEL) == set(WhenDone)


def test_remaining_running_does_nothing() -> None:
    calls, completion = _spy()
    assert completion.perform(WhenDone.NONE) is Outcome.NOTHING
    assert calls == []


def test_quitting_asks_the_application_to_close() -> None:
    calls, completion = _spy()
    assert completion.perform(WhenDone.QUIT) is Outcome.QUIT
    assert calls == ["quit"]


@pytest.mark.parametrize(
    "choice, command",
    [
        (WhenDone.RESTART, ("shutdown", "/r", "/t", "0")),
        (WhenDone.SHUTDOWN, ("shutdown", "/s", "/t", "0")),
    ],
)
def test_power_actions_are_handed_to_the_operating_system(choice, command) -> None:
    """LSSは電源を自前で扱わない。OSへ渡すだけ。"""
    calls, completion = _spy()
    assert completion.perform(choice) is Outcome.HANDED_TO_OS
    assert calls == [command]
    assert command_for(choice) == command


def test_a_refused_power_action_leaves_the_application_running() -> None:
    """権限が無ければ待機のまま残す。黙って落とさない。"""
    completion = Completion(runner=lambda command: 1)
    assert completion.perform(WhenDone.SHUTDOWN) is Outcome.REFUSED

    completion = Completion(runner=_raise)
    completion.reset()
    assert completion.perform(WhenDone.RESTART) is Outcome.REFUSED


def _raise(command):
    raise OSError("shutdown が見つからない")


def test_one_completion_runs_the_action_once() -> None:
    calls, completion = _spy()
    assert completion.perform(WhenDone.SHUTDOWN) is Outcome.HANDED_TO_OS
    assert completion.perform(WhenDone.SHUTDOWN) is Outcome.NOTHING
    assert len(calls) == 1
    completion.reset()
    assert completion.perform(WhenDone.SHUTDOWN) is Outcome.HANDED_TO_OS
    assert len(calls) == 2
