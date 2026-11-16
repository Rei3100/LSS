"""表示状態から凍結された文言へ。Qt非依存。

凍結ハンドオフ§3/§4の表をそのまま持つ。英語は状態名・ラベル・計数単位、
日本語は帰結と制約の短句。1つの文字列に複数の概念を詰めない。

件数はSOURCE単位で、出力やステムは数えない。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from lss.ui.contract import (
    Failure,
    HaltCode,
    SourceFailure,
    SourceOutcome,
    classify_result,
)
from lss.ui.state import PrimaryAction, PrimaryKind, UiModel, UiState, primary_control

# 状態名は英語。日本語の名詞は「音源」、計数は「件」。
_STATUS_WORD = {
    UiState.EMPTY: "IDLE",
    UiState.READY: "READY",
    UiState.PROCESSING: "RUNNING",
    UiState.STOPPING: "STOPPING",
    UiState.STOPPED: "STOPPED",
    UiState.HALTED: "HALTED",
    UiState.ALL_DONE: "ALL DONE",
    UiState.COMPLETE: "COMPLETE",
}

_ANCHOR_STATE = {
    UiState.EMPTY: "NO INPUT",
    UiState.READY: "READY",
    UiState.PROCESSING: "PROCESSING",
    UiState.STOPPING: "STOPPING",
    UiState.STOPPED: "STOPPED",
    UiState.HALTED: "HALTED",
    UiState.ALL_DONE: "ALL DONE",
    UiState.COMPLETE: "SESSION COMPLETE",
}

_ANCHOR_NOTE = {
    UiState.STOPPING: "進行中の1件を仕上げています。",
    UiState.STOPPED: "処理済みは保存されています。",
}

_PRIMARY_LABEL = {
    PrimaryAction.START: "START",
    PrimaryAction.RESUME: "RESUME",
    PrimaryAction.STOP: "STOP",
    PrimaryAction.BACK_TO_READY: "BACK TO READY",
}

# 副文は「操作名と状態が言えない帰結・制約」のみ。散文の重複説明を置かない。
_PRIMARY_NOTE = {
    UiState.PROCESSING: "進行中の1件だけ仕上げます",
    UiState.HALTED: "対応は RETRY か SKIP を選びます",
}

# リストの題。状態ごとに構図の重心が変わるので、題も変わる。
_LIST_TITLE = {
    UiState.EMPTY: "INPUT",
    UiState.READY: "QUEUE",
    UiState.PROCESSING: "RUNNING",
    UiState.STOPPING: "RUNNING",
    UiState.STOPPED: "QUEUE",
    UiState.HALTED: "RUNNING",
    UiState.ALL_DONE: "SESSION",
    UiState.COMPLETE: "SESSION",
}


def _list_subs(
    model: UiModel, state: UiState, tally: "Tally | None",
) -> tuple[tuple[str, str], ...]:
    """リストの副題。セッション終了後は集計、それ以外は待ち件数。"""
    if state is UiState.EMPTY:
        return ()
    if state in {UiState.ALL_DONE, UiState.COMPLETE}:
        if tally is None:
            return ()
        subs = [("SAVED", str(tally.saved))]
        if tally.skipped:
            subs.append(("SKIPPED", str(tally.skipped)))
        if tally.failed:
            subs.append(("FAILED", str(tally.failed)))
        return tuple(subs)
    if state in {UiState.PROCESSING, UiState.STOPPING, UiState.HALTED}:
        done = model.sources - model.pending
        return (("DONE", f"{done} / {model.sources}"),)
    return (("WAITING", str(model.pending)),)


def _count(value: int) -> str:
    """アンカーの計数は2桁ゼロ詰め（00 / 08 / 24）。"""
    return f"{value:02d}"


@dataclass(frozen=True)
class Presentation:
    """1つの表示状態が必要とする文言一式。"""

    status_word: str
    status_detail: str
    anchor_count: str
    anchor_unit: str
    anchor_state: str
    anchor_note: str
    list_title: str
    list_subs: tuple[tuple[str, str], ...]
    primary_label: str
    primary_note: str
    primary_kind: PrimaryKind
    shows_drop_target: bool
    shows_halt_card: bool
    allows_adding: bool


@dataclass(frozen=True)
class Tally:
    """セッション集計。1文に詰めず、独立要素として並べる。"""

    saved: int = 0
    skipped: int = 0
    failed: int = 0


def tally_of(rows: "list[SourceRow]") -> Tally:
    kinds = [row.kind for row in rows]
    return Tally(
        saved=sum(1 for kind in kinds if kind in {RowKind.DONE, RowKind.SAVED}),
        skipped=kinds.count(RowKind.SKIPPED),
        failed=sum(
            1 for kind in kinds if kind in {RowKind.FAILED, RowKind.UNSUPPORTED}
        ),
    )


def present(model: UiModel, state: UiState, tally: "Tally | None" = None) -> Presentation:
    """表示状態と件数から、凍結された文言を組み立てる。"""
    done = model.sources - model.pending
    control = primary_control(model)

    anchor_unit = "SOURCES"
    if state in {UiState.PROCESSING, UiState.STOPPING, UiState.HALTED, UiState.ALL_DONE}:
        # 巨大な数字は「終わった件数」。全体数は単位側が運ぶ（1文に詰めない）。
        anchor_count = _count(done)
        anchor_unit = f"/ {model.sources} SOURCES"
        status_detail = f"{done} / {model.sources} 完了"
        if state is UiState.STOPPING:
            status_detail = "停止待ち"
        elif state is UiState.HALTED:
            status_detail = "中断"
        elif state is UiState.ALL_DONE:
            status_detail = ""
    elif state is UiState.EMPTY:
        anchor_count = _count(0)
        status_detail = ""
    elif state in {UiState.READY, UiState.STOPPED}:
        # 巨大な数字はこれから処理する件数。処理済みが残っていても数え直さない。
        anchor_count = _count(model.pending or model.sources)
        status_detail = f"未処理 {model.pending} 件"
    elif state is UiState.COMPLETE:
        saved = tally.saved if tally is not None else model.sources
        anchor_count = _count(saved)
        status_detail = f"{saved} 件 完了"

    # 押せないSTARTには理由を添える。EMPTYはDROP HEREが既に言っているので添えない。
    primary_note = _PRIMARY_NOTE.get(state, "")
    if control.kind is PrimaryKind.REFUSING and state is not UiState.EMPTY:
        primary_note = "未処理の音源がありません"

    return Presentation(
        status_word=_STATUS_WORD[state],
        status_detail=status_detail,
        anchor_count=anchor_count,
        anchor_unit=anchor_unit,
        anchor_state=_ANCHOR_STATE[state],
        anchor_note=_ANCHOR_NOTE.get(state, ""),
        list_title=_LIST_TITLE[state],
        list_subs=_list_subs(model, state, tally),
        primary_label=(
            _PRIMARY_LABEL[control.action] if control.action is not None
            else _STATUS_WORD[state]
        ),
        primary_note=primary_note,
        primary_kind=control.kind,
        shows_drop_target=state is UiState.EMPTY,
        shows_halt_card=state is UiState.HALTED,
        # 追加はCOMPLETEでのみ不可（コントロール非表示）。
        allows_adding=state is not UiState.COMPLETE,
    )


class RowKind(Enum):
    """行の見え方。凍結§3の表と1対1。"""

    QUEUED = "queued"
    CURRENT = "current"  # 処理中。ステージ表示を持ちバッジは持たない
    HOLD = "hold"  # 停止要求中の後続
    DONE = "done"  # セッション中の完了
    SAVED = "saved"  # COMPLETE以降・履歴の保存済み
    SKIPPED = "skipped"
    FAILED = "failed"  # 再試行できる
    UNSUPPORTED = "unsupported"  # その音源では成立しない


# バッジは英語、補足は日本語の短句。両者を1つの文字列に混ぜない。
_ROW_BADGE = {
    RowKind.QUEUED: "",
    RowKind.CURRENT: "",
    RowKind.HOLD: "HOLD",
    RowKind.DONE: "✓ DONE",
    RowKind.SAVED: "INSTRUMENTAL",
    RowKind.SKIPPED: "SKIPPED",
    RowKind.FAILED: "FAILED",
    RowKind.UNSUPPORTED: "UNSUPPORTED",
}

_ROW_NOTE = {
    RowKind.SKIPPED: "同じ内容がすでに分離済みです。",
}

# 補足はerror_codeごとの事実で書く。RowKindは見え方であって理由ではないため、
# 1つのKindへ複数のerror_codeが集まる（FAILED/UNSUPPORTED）。Kindで文言を選ぶと
# 別の失敗が別の失敗の説明として出る。凍結Canvasの2件は文言をそのまま持つ。
_FAILURE_NOTE = {
    "input_changed": "処理中に元のファイルが変更されました。",
    "job_abandoned": "処理中に LSS を終了したため、保存していません。",
    "input_decode_failed": "この音源を読み込めませんでした。",
    "input_decode_timeout": "この音源の読み込みが時間内に終わりませんでした。",
    "file_not_found": "この音源が見つかりませんでした。",
    "processing_failed": "この音源の処理が完了しませんでした。",
    "bgm_candidate_processing_failed": "BGM の候補を作れませんでした。",
    "invalid_bgm_candidate_response": "BGM の候補を作れませんでした。",
    # チャンネル数の文言は実測値を含むため、行の素性から組み立てる（_channel_note）。
    "unsupported_channels": "",
    "unsafe_output_serialization": "分離結果をこの形式で保存できませんでした。",
    "invalid_input_signal": "この音源の音声データを正しく読めません。",
}


def _channel_note(channels: int) -> str:
    """凍結Canvasの規則。2chを超える瞬間だけ、実測チャンネル数で理由を言う。"""
    return f"{channels}ch の音源は分離できません。ステレオまで対応しています。" if channels > 2 else ""


@dataclass(frozen=True)
class SourceRow:
    """一覧の1行。SOURCE単位で、出力やステムは数えない。"""

    number: str  # キュー番号。履歴は "—"
    name: str
    kind: RowKind
    note: str = ""
    retryable: bool = False
    separated: str = ""  # 分離実施日時。相対表記は使わない
    item_id: str = ""  # キュー上の識別子。操作の対象を取り違えないために持つ
    source_path: str = ""  # その行自身の入力パス
    destination: str = ""  # その行自身の保存先（共通保存先を騙らない）
    recipe: str = ""  # 由来の記録
    # 音源の素性。読めたものだけを持ち、無い値は行に出さない（推測で埋めない）。
    duration: str = ""  # m:ss
    media_format: str = ""  # WAV / FLAC など
    sample_rate: str = ""  # kHz
    bit_depth: str = ""  # bit
    megabytes: str = ""  # MB
    channels: int = 0  # 2chを超えるときだけ、使えない理由として前面化する

    @property
    def removable(self) -> bool:
        """処理中の行と履歴は一覧から外さない。"""
        return self.kind not in {RowKind.CURRENT, RowKind.SAVED} and bool(self.item_id)

    @property
    def expandable(self) -> bool:
        """処理中の行は開かない。"""
        return self.kind is not RowKind.CURRENT

    @property
    def badge(self) -> str:
        return _ROW_BADGE[self.kind]


def _kind_from_outcome(outcome: SourceOutcome, failure: Failure | None) -> RowKind:
    if outcome is SourceOutcome.SAVED:
        return RowKind.DONE
    if outcome in {SourceOutcome.ALREADY_SEPARATED, SourceOutcome.NO_TARGET_FOUND}:
        # 検証済みの空出力に出荷UIの面は無い。行としてはSKIPPEDへ寄せる。
        return RowKind.SKIPPED
    if failure is not None and failure.source_failure is SourceFailure.UNSUPPORTED:
        return RowKind.UNSUPPORTED
    return RowKind.FAILED


def row_for(
    number: int | None, name: str, *, result: object = None,
    running: bool = False, holding: bool = False, history: bool = False,
    session_complete: bool = False, item_id: str = "",
    source_path: str = "", destination: str = "", media: object = None,
) -> SourceRow:
    """キューの1件を行の見え方へ翻訳する。結果の意味はcontractが所有する。

    `media` は読めた素性（`lss.ui.media.MediaFacts`）。読めなければ項目を持たない。
    """
    label = "—" if number is None else f"{number:02d}"
    common = {
        "item_id": item_id, "source_path": source_path, "destination": destination,
    }
    if media is not None:
        common.update({
            "duration": media.duration, "media_format": media.media_format,
            "sample_rate": media.sample_rate, "bit_depth": media.bit_depth,
            "megabytes": media.megabytes, "channels": media.channels,
        })
    recipe = ""
    if isinstance(result, dict):
        value = result.get("recipe_id")
        recipe = value if isinstance(value, str) else ""

    channel_note = _channel_note(common.get("channels", 0))

    if history:
        return SourceRow(label, name, RowKind.SAVED, recipe=recipe, **common)
    if running:
        return SourceRow(label, name, RowKind.CURRENT, **common)
    if result is None:
        # 2ch超は処理前から成立しない。凍結Canvasは行の素性から常に理由を出す。
        return SourceRow(
            label, name, RowKind.HOLD if holding else RowKind.QUEUED,
            note=channel_note, **common,
        )

    classified = classify_result(result)
    kind = _kind_from_outcome(classified.outcome, classified.failure)
    # セッション中の完了は「✓ DONE」（シアン）、COMPLETE以降は保存済みの主張へ変わる。
    if kind is RowKind.DONE and session_complete:
        kind = RowKind.SAVED
    # 凍結Canvasの優先順: ①2ch超の実測 → ②error_codeごとの理由 → ③SKIPPEDの短句。
    note = channel_note
    if not note and classified.failure is not None:
        code = result.get("error_code") if isinstance(result, dict) else None
        if code == "unsupported_channels":
            # 行の素性を読めないコンテナ（m4a/aac等）では、実測値はworkerだけが持つ。
            measured = result.get("channels") if isinstance(result, dict) else None
            note = _channel_note(measured if isinstance(measured, int) else 0)
            if not note:
                note = "ステレオまで対応しています。"
        elif isinstance(code, str):
            note = _FAILURE_NOTE.get(code, "")
        if not note and classified.failure.halt is not None:
            # SKIPや再起動で行に残ったHALTの音源は、止めた理由の短句をそのまま持つ。
            note = _HALT_COPY[classified.failure.halt][1]
    if not note and classified.outcome is not SourceOutcome.NO_TARGET_FOUND:
        # 検証済みの空出力は「分離済み」ではない。事実でない短句を付けない。
        note = _ROW_NOTE.get(kind, "")
    return SourceRow(
        label, name, kind, note=note,
        retryable=classified.failure is not None and classified.failure.retryable,
        recipe=recipe, **common,
    )


# HALTカードの文言。凍結された3種だけで、実行時に増やさない。
_HALT_COPY = {
    HaltCode.DISK_FULL: ("DISK FULL", "保存先の空き容量が足りません。"),
    HaltCode.FETCH_FAILED: (
        "FETCH FAILED",
        "処理に必要なデータを取得できませんでした。",
    ),
    HaltCode.ENGINE_STOPPED: (
        "ENGINE STOPPED", "分離エンジンが応答しません。",
    ),
}


@dataclass(frozen=True)
class HaltCard:
    """セッションを止めた条件の提示。回復はRETRY / SKIPだけ。"""

    tag: str
    code: str
    japanese: str
    metrics: tuple[tuple[str, str], ...] = ()
    reference: str = ""

    @property
    def recovery(self) -> tuple[str, str]:
        return ("RETRY", "SKIP")


def halt_card(
    halt: HaltCode, *, metrics: tuple[tuple[str, str], ...] = (), reference: str = "",
) -> HaltCard:
    """HALTの提示。数値は実測値だけを渡す（無ければ出さない）。"""
    code, japanese = _HALT_COPY[halt]
    return HaltCard(
        tag="HALTED", code=code, japanese=japanese,
        metrics=metrics, reference=reference,
    )


class WhenDone(Enum):
    """完了後の動作。凍結された4択で、増やさない。

    ここが持つのは選択の意味だけで、OS操作の実行は含まない。
    """

    NONE = "none"
    QUIT = "quit"
    RESTART = "restart"
    SHUTDOWN = "shutdown"


# ラベルは選択中のチップに出る短句。ツールチップは帰結を1文で言う。
WHEN_DONE_LABEL = {
    WhenDone.NONE: "待機したまま",
    WhenDone.QUIT: "LSS を終了",
    WhenDone.RESTART: "PC を再起動",
    WhenDone.SHUTDOWN: "PC を終了",
}

WHEN_DONE_TIP = {
    WhenDone.NONE: "完了後も待機したまま",
    WhenDone.QUIT: "完了後に LSS を終了",
    WhenDone.RESTART: "完了後に PC を再起動",
    WhenDone.SHUTDOWN: "完了後に PC を終了",
}


def when_done_choices() -> tuple[WhenDone, ...]:
    return tuple(WhenDone)


def shows_when_done(state: UiState) -> bool:
    """完了後の動作はREADYのときだけ出す。

    EMPTYは投入面が主張であり、完了後の動作を先に選ばせない（凍結Canvasと同じ）。
    """
    return state in {UiState.READY, UiState.STOPPED}


@dataclass(frozen=True)
class SaveTarget:
    """保存先の提示。1つの共通保存先を偽って示さないための最小の形。

    保存先はSOURCEごとに導出されるので、1バッチが複数の保存先を持ちうる。
    個別の実パスは行の展開詳細とALREADY SAVEDが持つ。
    """

    label: str  # SAVE TO / SAVED TO
    root: str  # 実パスのときの親。パターン表記のときは接頭辞
    leaf: str  # 強調する末端（LSS）
    chip: str  # 複数のときだけ N FOLDERS
    note: str  # 複数のときだけ日本語1行

    @property
    def single(self) -> bool:
        return not self.chip


_PATTERN_ROOT = "EACH SOURCE FOLDER\\"
_OUTPUT_LEAF = "LSS"


def save_target(destinations, *, saved: bool) -> SaveTarget:
    """保存先の集合から提示を組み立てる。0件でも規則は同じなので隠さない。"""
    unique = []
    for destination in destinations:
        text = str(destination)
        if text not in unique:
            unique.append(text)
    label = "SAVED TO" if saved else "SAVE TO"

    if len(unique) == 1:
        path = unique[0]
        head, _, tail = path.rpartition("\\")
        if not head:
            head, _, tail = path.rpartition("/")
        return SaveTarget(label=label, root=head + "\\" if head else "", leaf=tail or path,
                          chip="", note="")

    if not unique:
        # まだSOURCEが無くても、どこへ保存されるかは常に見える。
        return SaveTarget(label=label, root=_PATTERN_ROOT, leaf=_OUTPUT_LEAF, chip="", note="")

    count = len(unique)
    note = (
        f"{count} か所に保存しました。各音源の隣の LSS フォルダです。" if saved
        else f"保存先は {count} か所。各音源の隣に LSS フォルダを作ります。"
    )
    return SaveTarget(
        label=label, root=_PATTERN_ROOT, leaf=_OUTPUT_LEAF,
        chip=f"{count} FOLDERS", note=note,
    )


def format_separated(value: str) -> str:
    """分離実施日時。年を必ず含む絶対表記にする。

    「今日」「昨日」等の相対表記は耐久メタデータに使わない。
    記録された生成時刻をそのまま読み、現在時刻で置き換えない。
    """
    from datetime import datetime

    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return ""
    if moment.tzinfo is not None:
        moment = moment.astimezone()
    return moment.strftime("%Y-%m-%d %H:%M")


# 履歴の識別子。キューのUUIDと混ざらないよう接頭辞で分ける。
HISTORY_ID_PREFIX = "history:"


def history_id(source_path: str) -> str:
    """履歴行の識別子。選択・展開・文脈メニューはこれを対象にする。"""
    return f"{HISTORY_ID_PREFIX}{source_path}" if source_path else ""


def is_history_id(item_id: str) -> bool:
    return item_id.startswith(HISTORY_ID_PREFIX)


def history_row(name: str, *, separated: str, recipe: str, destination: str,
                source_path: str = "", item_id: str = "",
                media: object = None) -> SourceRow:
    """ALREADY SAVEDの1行。セッション内のキュー番号を持たない。

    キュー番号は持たないが、識別子は持つ。持たないと行を選ぶ・開く・
    右クリックする操作の対象が定まらず、見えているのに操作できない面になる。

    素性（長さ・形式・kHz・bit・MB）はキューの行と同じ文法で持つ。持たないと
    履歴だけが右端の長さも展開のスペックブロックも失い、同じ一覧の中で行の
    語彙が2種類になる。読めなければ項目を持たない（推測で埋めない）。
    """
    facts = {}
    if media is not None:
        facts = {
            "duration": media.duration, "media_format": media.media_format,
            "sample_rate": media.sample_rate, "bit_depth": media.bit_depth,
            "megabytes": media.megabytes, "channels": media.channels,
        }
    return SourceRow(
        "—", name, RowKind.SAVED,
        separated=format_separated(separated), recipe=recipe,
        destination=destination, source_path=source_path,
        item_id=item_id or history_id(source_path), **facts,
    )
