"""凍結された文言と行の意味を、Qtを起動せずに検証する。"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from lss.ui import presentation as ui
from lss.ui.contract import HaltCode
from lss.ui.presentation import (
    RowKind,
    Tally,
    halt_card,
    present,
    row_for,
    tally_of,
)
from lss.ui.state import Phase, PrimaryKind, UiModel, UiState, display_state

ROOT = Path(__file__).resolve().parents[1]

EMPTY = UiModel()
READY = UiModel(phase=Phase.READY, sources=8, pending=8)
READY_NONE_PENDING = UiModel(phase=Phase.READY, sources=8, pending=0)
PROCESSING = UiModel(phase=Phase.PROCESSING, sources=8, pending=5)
STOPPING = UiModel(phase=Phase.PROCESSING, sources=8, pending=5, stop_requested=True)
STOPPED = UiModel(phase=Phase.READY, sources=8, pending=5, stopped=True)
HALTED = UiModel(phase=Phase.PROCESSING, sources=8, pending=5, halted=True)
ALL_DONE = UiModel(phase=Phase.PROCESSING, sources=8, pending=0, all_done=True)
COMPLETE = UiModel(phase=Phase.COMPLETE, sources=8, pending=0)

ALL_MODELS = (
    EMPTY, READY, READY_NONE_PENDING, PROCESSING, STOPPING,
    STOPPED, HALTED, ALL_DONE, COMPLETE,
)


def _view(model: UiModel, tally: Tally | None = None):
    return present(model, display_state(model), tally)


@pytest.mark.parametrize(
    "model, word, anchor_state",
    [
        (EMPTY, "IDLE", "NO INPUT"),
        (READY, "READY", "READY"),
        (PROCESSING, "RUNNING", "PROCESSING"),
        (STOPPING, "STOPPING", "STOPPING"),
        (STOPPED, "STOPPED", "STOPPED"),
        (HALTED, "HALTED", "HALTED"),
        (ALL_DONE, "ALL DONE", "ALL DONE"),
        (COMPLETE, "COMPLETE", "SESSION COMPLETE"),
    ],
)
def test_state_words_are_the_frozen_ones(model, word, anchor_state) -> None:
    view = _view(model)
    assert view.status_word == word
    assert view.anchor_state == anchor_state


def test_counts_are_source_based_and_zero_padded() -> None:
    assert _view(EMPTY).anchor_count == "00"
    assert _view(READY).anchor_count == "08"
    # 巨大な数字は「済んだ件数」だけを持ち、全体数は単位側が運ぶ。
    # 1つの文字列に2つの概念を詰めない（凍結Canvasと同じ分け方）。
    assert _view(PROCESSING).anchor_count == "03"
    assert _view(PROCESSING).anchor_unit == "/ 8 SOURCES"
    assert _view(READY).anchor_unit == "SOURCES"


def test_ready_reports_the_pending_source_count_in_japanese() -> None:
    assert _view(READY).status_detail == "未処理 8 件"
    assert _view(STOPPED).status_detail == "未処理 5 件"
    # 「ファイル」で数えない。
    assert "ファイル" not in _view(READY).status_detail


@pytest.mark.parametrize(
    "model, label",
    [
        (EMPTY, "START"),
        (READY, "START"),
        (STOPPED, "RESUME"),
        (PROCESSING, "STOP"),
        (COMPLETE, "BACK TO READY"),
        (STOPPING, "STOPPING"),
        (ALL_DONE, "ALL DONE"),
        (HALTED, "HALTED"),
    ],
)
def test_primary_labels_match_the_frozen_table(model, label) -> None:
    assert _view(model).primary_label == label


def test_ready_is_start_and_complete_is_session_complete() -> None:
    assert _view(READY).primary_label == "START"
    assert _view(COMPLETE).anchor_state == "SESSION COMPLETE"


def test_only_meaningful_primary_notes_are_shown() -> None:
    # EMPTYはDROP HEREが既に言っているのでキャプションを出さない。
    assert _view(EMPTY).primary_note == ""
    assert _view(READY).primary_note == ""
    assert _view(READY_NONE_PENDING).primary_note == "未処理の音源がありません"
    assert _view(PROCESSING).primary_note == "進行中の1件だけ仕上げます"
    assert _view(HALTED).primary_note == "対応は RETRY か SKIP を選びます"


def test_stopping_and_stopped_carry_their_frozen_japanese_line() -> None:
    assert _view(STOPPING).anchor_note == "進行中の1件を仕上げています。"
    assert _view(STOPPED).anchor_note == "処理済みは保存されています。"


@pytest.mark.parametrize(
    "model, title",
    [
        (EMPTY, "INPUT"),
        (READY, "QUEUE"),
        (PROCESSING, "RUNNING"),
        (STOPPING, "RUNNING"),
        (STOPPED, "QUEUE"),
        (HALTED, "RUNNING"),
        (COMPLETE, "SESSION"),
    ],
)
def test_list_title_changes_with_the_composition(model, title) -> None:
    assert _view(model).list_title == title


def test_waiting_is_a_sub_item_not_the_title() -> None:
    assert _view(READY).list_subs == (("WAITING", "8"),)
    assert _view(EMPTY).list_subs == ()


def test_session_tallies_are_independent_items() -> None:
    tally = Tally(saved=5, skipped=2, failed=1)
    subs = _view(COMPLETE, tally).list_subs
    assert subs == (("SAVED", "5"), ("SKIPPED", "2"), ("FAILED", "1"))
    # 0件の集計は並べない。
    assert _view(COMPLETE, Tally(saved=8)).list_subs == (("SAVED", "8"),)


def test_only_empty_shows_the_drop_target() -> None:
    assert _view(EMPTY).shows_drop_target is True
    assert not [m for m in ALL_MODELS if m is not EMPTY and _view(m).shows_drop_target]


def test_only_halted_shows_the_halt_card() -> None:
    assert _view(HALTED).shows_halt_card is True
    assert not [m for m in ALL_MODELS if m is not HALTED and _view(m).shows_halt_card]


def test_adding_is_blocked_only_in_complete() -> None:
    assert _view(COMPLETE).allows_adding is False
    assert all(_view(m).allows_adding for m in ALL_MODELS if m is not COMPLETE)


def test_state_displays_are_not_actionable() -> None:
    for model in (STOPPING, ALL_DONE, HALTED):
        assert _view(model).primary_kind is PrimaryKind.STATE


# --- 行 -------------------------------------------------------------------

SAVED = {"status": "ok", "outputs": ["a.flac"]}
DUPLICATE = {"status": "skipped", "outputs": ["a.flac"]}
ZERO = {
    "status": "skipped", "outputs": [],
    "zero_result": {
        "reason": "no_wanted_score",
        "validation_result": {
            "validator_id": "v", "validator_revision": "1", "hard_gates_passed": True,
        },
    },
}
CHANGED = {"status": "error", "error_code": "input_changed", "message": "x"}
CHANNELS = {"status": "error", "error_code": "unsupported_channels", "message": "x"}


@pytest.mark.parametrize(
    "row, kind, badge",
    [
        (row_for(1, "a"), RowKind.QUEUED, ""),
        (row_for(1, "a", holding=True), RowKind.HOLD, "HOLD"),
        (row_for(1, "a", running=True), RowKind.CURRENT, ""),
        (row_for(1, "a", result=SAVED), RowKind.DONE, "✓ DONE"),
        (row_for(1, "a", result=SAVED, session_complete=True), RowKind.SAVED, "INSTRUMENTAL"),
        (row_for(1, "a", result=DUPLICATE), RowKind.SKIPPED, "SKIPPED"),
        (row_for(1, "a", result=ZERO), RowKind.SKIPPED, "SKIPPED"),
        (row_for(1, "a", result=CHANGED), RowKind.FAILED, "FAILED"),
        (row_for(1, "a", result=CHANNELS), RowKind.UNSUPPORTED, "UNSUPPORTED"),
        (row_for(None, "a", history=True), RowKind.SAVED, "INSTRUMENTAL"),
    ],
)
def test_row_kinds_match_the_frozen_table(row, kind, badge) -> None:
    assert row.kind is kind
    assert row.badge == badge


def test_history_rows_have_no_queue_number() -> None:
    assert row_for(None, "a", history=True).number == "—"
    assert row_for(7, "a").number == "07"


def test_only_recoverable_failures_offer_retry() -> None:
    assert row_for(1, "a", result=CHANGED).retryable is True
    assert row_for(1, "a", result=CHANNELS).retryable is False
    assert row_for(1, "a", result=DUPLICATE).retryable is False


def test_row_notes_are_the_frozen_short_lines() -> None:
    assert row_for(1, "a", result=DUPLICATE).note == "同じ内容がすでに分離済みです。"
    assert row_for(1, "a", result=CHANGED).note == "処理中に元のファイルが変更されました。"
    assert row_for(1, "a", result=CHANNELS).note == "ステレオまで対応しています。"


def test_channel_note_uses_the_measured_count_like_the_frozen_canvas() -> None:
    """凍結Canvasは「Nch は分離できません。」を実測値で書く。定型文だけにしない。"""
    from lss.ui.media import MediaFacts

    row = row_for(1, "a", result=CHANNELS, media=MediaFacts(channels=6))
    assert row.note == "6ch の音源は分離できません。ステレオまで対応しています。"
    # 行の素性を読めないコンテナでは、workerが測った値が唯一の事実。
    measured = {**CHANNELS, "channels": 8}
    assert row_for(1, "a", result=measured).note == (
        "8ch の音源は分離できません。ステレオまで対応しています。"
    )


def test_multichannel_source_states_the_reason_before_it_is_attempted() -> None:
    from lss.ui.media import MediaFacts

    row = row_for(1, "a", media=MediaFacts(channels=6))
    assert row.kind is RowKind.QUEUED
    assert row.note == "6ch の音源は分離できません。ステレオまで対応しています。"
    assert row_for(1, "a", media=MediaFacts(channels=2)).note == ""


def test_each_failure_explains_itself_and_never_borrows_another_reason() -> None:
    """RowKindで文言を選ぶと、別の失敗が別の失敗の説明として出る（実機で観測）。"""
    def note(code: str) -> str:
        return row_for(1, "a", result={
            "status": "error", "error_code": code, "message": "x",
        }).note

    assert note("unsafe_output_serialization") == "分離結果をこの形式で保存できませんでした。"
    assert note("processing_failed") == "この音源の処理が完了しませんでした。"
    assert note("input_decode_failed") == "この音源を読み込めませんでした。"
    assert note("file_not_found") == "この音源が見つかりませんでした。"
    assert note("input_decode_timeout") == "この音源の読み込みが時間内に終わりませんでした。"
    reasons = {
        note(code) for code in (
            "input_changed", "unsafe_output_serialization", "processing_failed",
            "input_decode_failed", "file_not_found", "input_decode_timeout",
        )
    }
    assert len(reasons) == 6
    # ステレオの制約は、チャンネル数が理由のときにだけ言う。
    assert "ステレオ" not in note("unsafe_output_serialization")
    assert "ステレオ" not in note("processing_failed")


def test_tally_counts_sources_not_outputs() -> None:
    rows = [
        row_for(1, "a", result=SAVED),
        row_for(2, "b", result=SAVED, session_complete=True),
        row_for(3, "c", result=DUPLICATE),
        row_for(4, "d", result=CHANGED),
        row_for(5, "e", result=CHANNELS),
        row_for(6, "f"),
    ]
    assert tally_of(rows) == Tally(saved=2, skipped=1, failed=2)


# --- HALTカード -----------------------------------------------------------

@pytest.mark.parametrize(
    "halt, code",
    [
        (HaltCode.DISK_FULL, "DISK FULL"),
        (HaltCode.FETCH_FAILED, "FETCH FAILED"),
        (HaltCode.ENGINE_STOPPED, "ENGINE STOPPED"),
    ],
)
def test_halt_cards_exist_for_every_frozen_code(halt, code) -> None:
    card = halt_card(halt)
    assert card.code == code
    assert card.japanese.endswith("。")
    assert card.recovery == ("RETRY", "SKIP")


def test_halt_metrics_are_only_shown_when_measured() -> None:
    assert halt_card(HaltCode.FETCH_FAILED).metrics == ()
    measured = halt_card(HaltCode.DISK_FULL, metrics=(("空き", "1.2 GB"),))
    assert measured.metrics == (("空き", "1.2 GB"),)


# --- 出荷してはならないもの ------------------------------------------------

def test_presentation_never_exposes_a_forbidden_capability() -> None:
    source = (ROOT / "lss" / "ui" / "presentation.py").read_text(encoding="utf-8")
    forbidden = re.compile(
        r"VOCALS|ボーカル|DOWNLOAD ALL|OPEN FOLDER|STATE JUMP|NEEDS WIRING|"
        r"設定画面|モデル選択|スリープ|\bETA\b|残り時間|パーセント",
        re.IGNORECASE,
    )
    assert not forbidden.findall(source)
    # 進捗の割合を文言として組み立てない（不確定表示はcontractが決める）。
    assert "%" not in "".join(
        line for line in source.splitlines() if "f\"" in line or "f'" in line
    ).replace("{value:02d}", "")


def test_presentation_does_not_depend_on_qt() -> None:
    source = (ROOT / "lss" / "ui" / "presentation.py").read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if re.match(r"\s*(import|from)\s", line)]
    assert imports
    assert not [line for line in imports if "PySide6" in line or "Qt" in line]


# --- 保存先（1つの共通保存先を偽って示さない） -----------------------------

from lss.ui.presentation import save_target  # noqa: E402


def test_a_single_destination_shows_its_real_path() -> None:
    target = save_target([('D' + chr(58) + '\\Music\\Album\\LSS')], saved=False)
    assert target.single is True
    assert target.label == "SAVE TO"
    assert target.root == ('D' + chr(58) + '\\Music\\Album') + "\\"
    assert target.leaf == "LSS"
    assert target.chip == "" and target.note == ""


def test_several_destinations_are_never_shown_as_one() -> None:
    target = save_target([('D' + chr(58) + '\\A\\LSS'), ('D' + chr(58) + '\\B\\LSS'), ('E' + chr(58) + '\\C\\LSS')], saved=False)
    assert target.single is False
    assert target.root == "EACH SOURCE FOLDER\\"
    assert target.leaf == "LSS"
    assert target.chip == "3 FOLDERS"
    assert target.note == "保存先は 3 か所。各音源の隣に LSS フォルダを作ります。"
    # 実パスを1つだけ選んで見せてはいけない。
    assert "D:" not in target.root


def test_completed_sessions_use_the_past_tense_label() -> None:
    one = save_target([('D' + chr(58) + '\\A\\LSS')], saved=True)
    many = save_target([('D' + chr(58) + '\\A\\LSS'), ('D' + chr(58) + '\\B\\LSS')], saved=True)
    assert one.label == "SAVED TO" and many.label == "SAVED TO"
    assert many.note == "2 か所に保存しました。各音源の隣の LSS フォルダです。"


def test_duplicate_destinations_count_once() -> None:
    target = save_target([('D' + chr(58) + '\\A\\LSS'), ('D' + chr(58) + '\\A\\LSS')], saved=False)
    assert target.single is True
    assert target.chip == ""


def test_the_saving_rule_is_visible_even_with_no_sources() -> None:
    target = save_target([], saved=False)
    assert target.leaf == "LSS"
    assert target.chip == ""


def test_a_skipped_halt_source_stays_retryable_with_the_halt_reason() -> None:
    row = row_for(1, "a", result={"status": "error", "error_code": "insufficient_output_space"})
    assert row.kind is RowKind.FAILED
    assert row.retryable is True
    assert row.note == "保存先の空き容量が足りません。"


def test_a_validated_zero_result_does_not_claim_an_earlier_separation() -> None:
    zero = {"status": "ok", "outputs": [], "zero_result": {"validation_result": {"validator_id": "exact_digital_silence"}}}
    row = row_for(1, "a", result=zero)
    assert row.kind is RowKind.SKIPPED
    assert row.note == ""
    assert row_for(1, "a", result={"status": "skipped", "outputs": ["x.flac"]}).note == "同じ内容がすでに分離済みです。"
