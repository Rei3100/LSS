"""凍結UI状態仕様を、Qtを起動せずに表で検証する。"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from lss.ui import state as ui
from lss.ui.state import (
    HaltRecovery,
    Key,
    Phase,
    PrimaryAction,
    PrimaryKind,
    UiModel,
    UiState,
)


ROOT = Path(__file__).resolve().parents[1]

EMPTY = UiModel(phase=Phase.READY, sources=0, pending=0)
READY = UiModel(phase=Phase.READY, sources=8, pending=8)
READY_NOTHING_PENDING = UiModel(phase=Phase.READY, sources=8, pending=0)
PROCESSING = UiModel(phase=Phase.PROCESSING, sources=8, pending=5)
STOPPING = UiModel(phase=Phase.PROCESSING, sources=8, pending=5, stop_requested=True)
STOPPED = UiModel(phase=Phase.READY, sources=8, pending=5, stopped=True)
HALTED = UiModel(phase=Phase.PROCESSING, sources=8, pending=5, halted=True)
ALL_DONE = UiModel(phase=Phase.PROCESSING, sources=8, pending=0, all_done=True)
COMPLETE = UiModel(phase=Phase.COMPLETE, sources=8, pending=0)

ALL_MODELS = (
    EMPTY, READY, READY_NOTHING_PENDING, PROCESSING, STOPPING,
    STOPPED, HALTED, ALL_DONE, COMPLETE,
)


@pytest.mark.parametrize(
    "model, expected",
    [
        (EMPTY, UiState.EMPTY),
        (READY, UiState.READY),
        (READY_NOTHING_PENDING, UiState.READY),
        (PROCESSING, UiState.PROCESSING),
        (STOPPING, UiState.STOPPING),
        (STOPPED, UiState.STOPPED),
        (HALTED, UiState.HALTED),
        (ALL_DONE, UiState.ALL_DONE),
        (COMPLETE, UiState.COMPLETE),
    ],
)
def test_display_state(model: UiModel, expected: UiState) -> None:
    assert ui.display_state(model) is expected


def test_states_are_exactly_the_frozen_set() -> None:
    assert {value.name for value in UiState} == {
        "EMPTY", "READY", "PROCESSING", "STOPPING",
        "STOPPED", "HALTED", "ALL_DONE", "COMPLETE",
    }


@pytest.mark.parametrize(
    "model, kind, action",
    [
        (EMPTY, PrimaryKind.REFUSING, PrimaryAction.START),
        (READY, PrimaryKind.ACTION, PrimaryAction.START),
        (READY_NOTHING_PENDING, PrimaryKind.REFUSING, PrimaryAction.START),
        (PROCESSING, PrimaryKind.ACTION, PrimaryAction.STOP),
        (STOPPING, PrimaryKind.STATE, None),
        (STOPPED, PrimaryKind.ACTION, PrimaryAction.RESUME),
        (HALTED, PrimaryKind.STATE, None),
        (ALL_DONE, PrimaryKind.STATE, None),
        (COMPLETE, PrimaryKind.ACTION, PrimaryAction.BACK_TO_READY),
    ],
)
def test_primary_control(model: UiModel, kind: PrimaryKind, action: PrimaryAction | None) -> None:
    control = ui.primary_control(model)
    assert control.kind is kind
    assert control.action is action


@pytest.mark.parametrize("model", [STOPPING, HALTED, ALL_DONE])
def test_state_displays_are_not_pressable(model: UiModel) -> None:
    control = ui.primary_control(model)
    assert control.actionable is False
    assert control.responds_to_press is False
    assert ui.activate_primary(model) == model


def test_disabled_start_still_answers_the_press() -> None:
    control = ui.primary_control(EMPTY)
    assert control.actionable is False
    assert control.responds_to_press is True
    assert ui.activate_primary(EMPTY) == EMPTY


def test_start_begins_processing() -> None:
    assert ui.display_state(ui.activate_primary(READY)) is UiState.PROCESSING


def test_resume_continues_the_remaining_work() -> None:
    resumed = ui.activate_primary(STOPPED)
    assert ui.display_state(resumed) is UiState.PROCESSING
    assert resumed.pending == STOPPED.pending
    assert resumed.stopped is False


def test_processing_can_request_a_safe_stop() -> None:
    assert ui.can_request_stop(PROCESSING) is True
    assert ui.display_state(ui.activate_primary(PROCESSING)) is UiState.STOPPING
    assert ui.display_state(ui.request_stop(PROCESSING)) is UiState.STOPPING


def test_stopping_settles_into_stopped_with_work_left() -> None:
    settled = ui.stop_settled(STOPPING)
    assert ui.display_state(settled) is UiState.STOPPED
    assert ui.primary_control(settled).action is PrimaryAction.RESUME


def test_stopping_is_not_actionable_and_offers_no_second_stop() -> None:
    assert ui.can_request_stop(STOPPING) is False
    assert ui.request_stop(STOPPING) == STOPPING
    assert ui.apply_key(STOPPING, Key.ENTER) == STOPPING
    assert ui.apply_key(STOPPING, Key.ESC) == STOPPING


@pytest.mark.parametrize("key", list(Key))
def test_halted_never_becomes_stopping(key: Key) -> None:
    assert ui.can_request_stop(HALTED) is False
    assert ui.request_stop(HALTED) == HALTED
    after = ui.apply_key(HALTED, key)
    assert after == HALTED
    assert ui.display_state(after) is UiState.HALTED


def test_halted_recovers_only_through_retry_and_skip() -> None:
    assert ui.halt_recovery(HALTED) == (HaltRecovery.RETRY, HaltRecovery.SKIP)
    assert ui.display_state(ui.fault_resolved(HALTED)) is UiState.PROCESSING


@pytest.mark.parametrize("model", [m for m in ALL_MODELS if m is not HALTED])
def test_halt_recovery_exists_only_while_halted(model: UiModel) -> None:
    assert ui.halt_recovery(model) == ()


def test_halt_keeps_a_pending_stop_request() -> None:
    halted_after_stop = ui.fault_raised(STOPPING)
    assert ui.display_state(halted_after_stop) is UiState.HALTED
    assert ui.display_state(ui.fault_resolved(halted_after_stop)) is UiState.STOPPING


def test_fault_only_applies_while_processing() -> None:
    assert ui.fault_raised(READY) == READY
    assert ui.fault_raised(COMPLETE) == COMPLETE


def test_queue_drained_reaches_complete_through_all_done() -> None:
    drained = ui.queue_drained(UiModel(phase=Phase.PROCESSING, sources=8, pending=0))
    assert ui.display_state(drained) is UiState.ALL_DONE
    assert ui.display_state(ui.session_completed(drained)) is UiState.COMPLETE


def test_last_source_finishing_during_stop_completes_the_session() -> None:
    drained = ui.queue_drained(UiModel(phase=Phase.PROCESSING, sources=8, pending=0, stop_requested=True))
    assert ui.display_state(drained) is UiState.ALL_DONE
    assert ui.display_state(ui.session_completed(drained)) is UiState.COMPLETE


def test_adding_work_while_all_done_does_not_complete_the_session() -> None:
    """ALL DONE中の追加で全件完了は偽になり、未処理を残したままCOMPLETEへ入らない。"""
    grew = ui.with_counts(ALL_DONE, sources=9, pending=1)
    assert grew.all_done is False
    assert ui.display_state(grew) is UiState.PROCESSING
    assert ui.display_state(ui.session_completed(grew)) is UiState.PROCESSING


def test_queue_is_not_drained_while_work_is_pending() -> None:
    assert ui.queue_drained(PROCESSING) == PROCESSING
    assert ui.queue_drained(STOPPING) == STOPPING


def test_complete_returns_to_ready() -> None:
    back = ui.activate_primary(COMPLETE)
    assert back.phase is Phase.READY
    assert ui.display_state(ui.with_counts(back, sources=0, pending=0)) is UiState.EMPTY


def test_empty_belongs_to_the_ready_side() -> None:
    assert EMPTY.phase is Phase.READY
    assert ui.primary_control(EMPTY).action is PrimaryAction.START


TRANSITIONS = (
    ui.activate_primary, ui.request_stop, ui.stop_settled, ui.fault_raised,
    ui.fault_resolved, ui.queue_drained, ui.session_completed,
    lambda m: ui.apply_key(m, Key.ENTER), lambda m: ui.apply_key(m, Key.ESC),
)

# ユーザーの入力だけで起きる遷移。障害発生・保存完了などの外部事象は含まない。
USER_TRANSITIONS = (
    ui.activate_primary, ui.request_stop,
    lambda m: ui.apply_key(m, Key.ENTER), lambda m: ui.apply_key(m, Key.ESC),
)


def _reachable(model: UiModel) -> set[UiModel]:
    seen = {model}
    frontier = [model]
    while frontier:
        current = frontier.pop()
        for transition in TRANSITIONS:
            following = transition(current)
            if following not in seen:
                seen.add(following)
                frontier.append(following)
    return seen


@pytest.mark.parametrize("model", ALL_MODELS)
def test_user_input_never_drives_halted_into_stopping(model: UiModel) -> None:
    """到達可能な全HALTEDモデルで、操作とキーが停止要求を作らない。

    停止要求を保持したままHALTEDになった場合も含めて見る。
    """
    halted = [m for m in _reachable(model) if ui.display_state(m) is UiState.HALTED]
    for current in halted:
        assert ui.can_request_stop(current) is False
        for transition in USER_TRANSITIONS:
            assert transition(current) == current


@pytest.mark.parametrize("model", ALL_MODELS)
def test_stopping_only_settles_stops_or_finishes(model: UiModel) -> None:
    """STOPPINGの出口は、安全停止の完了・全件完了・障害だけ。"""
    stopping = [m for m in _reachable(model) if ui.display_state(m) is UiState.STOPPING]
    for current in stopping:
        successors = {ui.display_state(transition(current)) for transition in TRANSITIONS}
        assert successors <= {
            UiState.STOPPING, UiState.STOPPED, UiState.ALL_DONE, UiState.HALTED,
        }


@pytest.mark.parametrize("model", ALL_MODELS)
def test_transitions_never_leave_the_frozen_state_set(model: UiModel) -> None:
    # 到達したモデルはすべてUiModelの不変条件を満たし、凍結8状態の内側にとどまる。
    reached = {ui.display_state(m) for m in _reachable(model)}
    assert reached <= set(UiState)
    assert not [m for m in _reachable(model) if m.all_done and m.pending]


@pytest.mark.parametrize("model", ALL_MODELS)
def test_halted_is_never_reachable_from_a_primary_press(model: UiModel) -> None:
    if ui.display_state(model) is UiState.HALTED:
        pytest.skip("HALTED自身は対象外")
    assert ui.display_state(ui.activate_primary(model)) is not UiState.HALTED


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sources": -1},
        {"pending": -1},
        {"sources": 1, "pending": 2},
        {"phase": Phase.READY, "stop_requested": True},
        {"phase": Phase.READY, "halted": True},
        {"phase": Phase.COMPLETE, "all_done": True},
        {"phase": Phase.PROCESSING, "stopped": True},
        {"phase": Phase.PROCESSING, "sources": 2, "pending": 1, "all_done": True},
    ],
)
def test_impossible_models_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        UiModel(**kwargs)


def test_state_layer_exposes_no_forbidden_capability() -> None:
    names = [name for name in dir(ui) if not name.startswith("_")]
    values = [
        member.value
        for name in names
        for member in getattr(getattr(ui, name), "__members__", {}).values()
    ]
    forbidden = re.compile(
        r"vocal|stem|checkpoint|provider|backend|recipe|select|setting|window_mode|"
        r"maximi[sz]e|restore|fullscreen|zoom|percent|\beta\b|sleep|state_jump|needs_wiring",
        re.IGNORECASE,
    )
    assert not [word for word in names + values if forbidden.search(str(word))]


def test_state_does_not_depend_on_qt() -> None:
    source = (ROOT / "lss" / "ui" / "state.py").read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if re.match(r"\s*(import|from)\s", line)]
    assert imports
    assert not [line for line in imports if "PySide6" in line or "Qt" in line]
