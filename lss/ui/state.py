"""凍結UI状態仕様の純粋な実装。Qt非依存。

主状態は READY / PROCESSING / COMPLETE の3つで、付随フラグが表示状態を分ける
（EMPTY・STOPPED・STOPPING・HALTED・ALL DONE）。内部都合の中間状態は増やさない。

主操作は3種類に分かれる。

- ACTION  : 押せる操作（START / RESUME / STOP / BACK TO READY）
- REFUSING: 押せるように見えるが実行できない START/RESUME。押下は拒否として返す
- STATE   : 状態表示。押下・ホバー反応を持たない（STOPPING / ALL DONE / HALTED）

HALTED は停止できる状態ではない。回復はHALT面のRETRY / SKIPだけで、
Enter / Esc は停止要求を発行しない。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class Phase(Enum):
    """主状態。"""

    READY = "ready"
    PROCESSING = "processing"
    COMPLETE = "complete"


class UiState(Enum):
    """表示上の状態。"""

    EMPTY = "empty"
    READY = "ready"
    PROCESSING = "processing"
    STOPPING = "stopping"
    STOPPED = "stopped"
    HALTED = "halted"
    ALL_DONE = "all_done"
    COMPLETE = "complete"


class PrimaryAction(Enum):
    """主操作が実行する行為。"""

    START = "start"
    RESUME = "resume"
    STOP = "stop"
    BACK_TO_READY = "back_to_ready"


class PrimaryKind(Enum):
    ACTION = "action"
    REFUSING = "refusing"
    STATE = "state"


class HaltRecovery(Enum):
    """HALT面の回復操作。ここ以外にHALTEDからの出口はない。"""

    RETRY = "retry"
    SKIP = "skip"


class Key(Enum):
    """主操作に関わるキー。行選択・追加などは描画層が扱う。"""

    ENTER = "enter"
    ESC = "esc"


@dataclass(frozen=True)
class PrimaryControl:
    kind: PrimaryKind
    action: PrimaryAction | None = None

    @property
    def actionable(self) -> bool:
        return self.kind is PrimaryKind.ACTION

    @property
    def responds_to_press(self) -> bool:
        """状態表示は押下・ホバー反応を持たない（偽アフォーダンスを作らない）。"""
        return self.kind is not PrimaryKind.STATE


@dataclass(frozen=True)
class UiModel:
    """UIが持つ最小の状態。件数はSOURCE単位で、出力やステムは数えない。"""

    phase: Phase = Phase.READY
    sources: int = 0  # キュー内のSOURCE総数（履歴は含まない）
    pending: int = 0  # 未処理のSOURCE数
    stopped: bool = False
    stop_requested: bool = False
    halted: bool = False
    all_done: bool = False

    def __post_init__(self) -> None:
        if self.sources < 0 or self.pending < 0:
            raise ValueError("SOURCEの件数は負にできません")
        if self.pending > self.sources:
            raise ValueError("未処理の件数がSOURCE総数を超えています")
        if self.phase is not Phase.PROCESSING and (
            self.stop_requested or self.halted or self.all_done
        ):
            raise ValueError("処理中以外にSTOPPING/HALTED/ALL DONEは存在しません")
        if self.phase is not Phase.READY and self.stopped:
            raise ValueError("stoppedはREADY側の表示状態です")
        if self.all_done and self.pending:
            raise ValueError("未処理が残っている状態で全件完了にはなりません")


def display_state(model: UiModel) -> UiState:
    if model.phase is Phase.COMPLETE:
        return UiState.COMPLETE
    if model.phase is Phase.PROCESSING:
        # HALTEDが最優先。停止要求と同時に成立させない。
        if model.halted:
            return UiState.HALTED
        if model.all_done:
            return UiState.ALL_DONE
        if model.stop_requested:
            return UiState.STOPPING
        return UiState.PROCESSING
    if model.stopped:
        return UiState.STOPPED
    if model.sources == 0:
        return UiState.EMPTY
    return UiState.READY


_STATE_DISPLAY = frozenset({UiState.STOPPING, UiState.ALL_DONE, UiState.HALTED})


def primary_control(model: UiModel) -> PrimaryControl:
    state = display_state(model)
    if state in _STATE_DISPLAY:
        return PrimaryControl(PrimaryKind.STATE)
    if state is UiState.COMPLETE:
        return PrimaryControl(PrimaryKind.ACTION, PrimaryAction.BACK_TO_READY)
    if state is UiState.PROCESSING:
        return PrimaryControl(PrimaryKind.ACTION, PrimaryAction.STOP)
    action = PrimaryAction.RESUME if state is UiState.STOPPED else PrimaryAction.START
    kind = PrimaryKind.ACTION if model.pending > 0 else PrimaryKind.REFUSING
    return PrimaryControl(kind, action)


def halt_recovery(model: UiModel) -> tuple[HaltRecovery, ...]:
    """HALTEDのときだけRETRY / SKIPを持つ。"""
    if display_state(model) is UiState.HALTED:
        return (HaltRecovery.RETRY, HaltRecovery.SKIP)
    return ()


def can_request_stop(model: UiModel) -> bool:
    """安全停止を要求できるのは処理中だけ。HALTED / STOPPINGでは要求しない。"""
    return display_state(model) is UiState.PROCESSING


def _run(model: UiModel) -> UiModel:
    return replace(
        model, phase=Phase.PROCESSING, stopped=False,
        stop_requested=False, halted=False, all_done=False,
    )


def request_stop(model: UiModel) -> UiModel:
    """安全停止の要求。押せない状態では何も起きない。"""
    if not can_request_stop(model):
        return model
    return replace(model, stop_requested=True)


def activate_primary(model: UiModel) -> UiModel:
    """主操作の実行。ACTION以外は状態を変えない。"""
    control = primary_control(model)
    if not control.actionable:
        return model
    if control.action in {PrimaryAction.START, PrimaryAction.RESUME}:
        return _run(model)
    if control.action is PrimaryAction.STOP:
        return replace(model, stop_requested=True)
    return replace(model, phase=Phase.READY, stopped=False)


def apply_key(model: UiModel, key: Key) -> UiModel:
    """Enterは主操作、Escは停止要求。どちらもHALTEDでは何もしない。"""
    if key is Key.ENTER:
        return activate_primary(model)
    return request_stop(model)


def stop_settled(model: UiModel) -> UiModel:
    """進行中の1件が安全な完了点に達し、保存された。"""
    if display_state(model) is not UiState.STOPPING:
        return model
    return replace(model, phase=Phase.READY, stop_requested=False, stopped=True)


def fault_raised(model: UiModel) -> UiModel:
    """セッションを続行できない障害。進捗は止め、偽の進行を続けない。

    停止要求は保持する。障害が起きたことでユーザーの停止意思を捨てない。
    """
    if model.phase is not Phase.PROCESSING:
        return model
    return replace(model, halted=True)


def fault_resolved(model: UiModel) -> UiModel:
    """RETRY / SKIPでHALTから復帰する。"""
    if display_state(model) is not UiState.HALTED:
        return model
    return replace(model, halted=False)


def queue_drained(model: UiModel) -> UiModel:
    """全件が終端に達した。COMPLETE遷移待ちの短い状態。

    停止要求中の最後の1件が終わった場合も、残りがなければ全件完了が真実。
    未処理が残っているうちは全件完了にしない（件数の更新が先）。
    """
    if model.pending or display_state(model) not in {UiState.PROCESSING, UiState.STOPPING}:
        return model
    return replace(model, all_done=True)


def session_completed(model: UiModel) -> UiModel:
    if display_state(model) is not UiState.ALL_DONE:
        return model
    return replace(model, phase=Phase.COMPLETE, all_done=False, stop_requested=False)


def with_counts(model: UiModel, *, sources: int, pending: int) -> UiModel:
    """SOURCEの件数を更新する。行の中身とキュー操作は上位が持つ。

    ALL DONE中の追加は許可されている。未処理が増えれば全件完了は偽になるため、
    未処理を残したままCOMPLETEへ進ませない。
    """
    return replace(
        model, sources=sources, pending=pending,
        all_done=model.all_done and pending == 0,
    )
