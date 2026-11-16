"""ワーカーの実イベントを、凍結UI意味論へ翻訳する境界。Qt非依存。

ここが扱うのは lss.core.worker と lss.core.jobs が実際に発行する phase / error_code だけで、
UIの文言は持たない。返すのは意味の分類であり、表示語は上位の描画層が所有する。

工程の対応（現行実装の実イベントから導出）:

- PREPARE : 入力の指紋計算・復号・再照合、モデル取得、実行環境確認、既存出力の照合
- SEPARATE: 推論の実行（入力の複製と実行環境の確定を含む）
- SAVE    : 出力の解析・書き出し・公開・完了記録

SEPARATE は現行エンジンから真の進捗を観測できない。`staging_input` のように
測れるイベントが内部に混ざっていても、工程としては常に不確定として扱う。
偽のパーセンテージ・残り時間を作らないための機械的な保証である。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class Stage(Enum):
    """処理中の1件について、行内に表示できる3段の工程。"""

    PREPARE = "prepare"
    SEPARATE = "separate"
    SAVE = "save"


# 工程ではなく、その音源の終端を表すphase。
# ただしworkerがこれを流すのは自身で処理を完走／例外終了した経路だけで、
# 再利用スキップの早期returnとキュー段の失敗では1度も流れない（is_terminalを参照）。
TERMINAL_PHASES = frozenset({"completed", "failed"})

_STAGE_BY_PHASE = {
    "preparing_input": Stage.PREPARE,
    "hashing_input": Stage.PREPARE,
    "verifying_input": Stage.PREPARE,
    "validating_input_signal": Stage.PREPARE,
    "checking_model": Stage.PREPARE,
    "downloading_model": Stage.PREPARE,
    "checking_runtime": Stage.PREPARE,
    "checking_completed_output": Stage.PREPARE,
    "separating": Stage.SEPARATE,
    "runtime_ready": Stage.SEPARATE,
    "staging_input": Stage.SEPARATE,
    # 低負荷プロファイルの推論中、各forwardの直後に挟む休止。推論の一部であり、
    # 測れる量(allocated_bytes等)は進捗ではないのでSEPARATE(不確定)に留める。
    "forward_rest": Stage.SEPARATE,
    "analyzing_outputs": Stage.SAVE,
    "checking_existing_output": Stage.SAVE,
    "writing_outputs": Stage.SAVE,
    "recording_completion": Stage.SAVE,
}

# 同じverifying_inputでも、分離の前後で意味する工程が変わる。分離が終わってから
# 公開までの再照合は、ユーザーから見れば保存側の工程。
_SAVE_VERIFY_CHECKPOINTS = frozenset({"after_separation", "before_publish"})
_PREPARE_VERIFY_CHECKPOINTS = frozenset({"after_decode", "before_reuse", "after_metadata"})

# checkpointはphaseと独立にSAVE/PREPAREを分ける2つ目の軸なので、phaseと同じく
# 実装が増えたことを機械的に検出できるようにしておく（未知値は安全側のPREPAREへ落ちる）。
KNOWN_VERIFY_CHECKPOINTS = _PREPARE_VERIFY_CHECKPOINTS | _SAVE_VERIFY_CHECKPOINTS

# 測れる工程だけが持つ量。単位（バイト・フレーム・ステム数）はUIへ出さず、
# 0.0〜1.0の割合だけを渡す。ステム数などの内部計数を露出させないため。
_MEASURE_KEYS = (
    ("completed_bytes", "total_bytes"),
    ("completed_frames", "total_frames"),
    ("completed_stems", "total_stems"),
)

KNOWN_PHASES = frozenset(_STAGE_BY_PHASE) | TERMINAL_PHASES


@dataclass(frozen=True)
class StageProgress:
    """処理中の1件の現在地。fractionがNoneなら不確定表示。"""

    stage: Stage
    fraction: float | None = None
    awaiting_model_fetch: bool = False

    @property
    def determinate(self) -> bool:
        return self.fraction is not None


def _phase_of(event: object) -> str | None:
    if not isinstance(event, Mapping):
        return None
    phase = event.get("phase")
    return phase if isinstance(phase, str) and phase else None


def _fraction_of(event: Mapping) -> float | None:
    for completed_key, total_key in _MEASURE_KEYS:
        completed, total = event.get(completed_key), event.get(total_key)
        if isinstance(completed, bool) or isinstance(total, bool):
            continue
        if not isinstance(completed, (int, float)) or not isinstance(total, (int, float)):
            continue
        if total <= 0:
            continue
        return min(1.0, max(0.0, float(completed) / float(total)))
    return None


def stage_of(event: object) -> Stage | None:
    """進捗イベントの工程。終端phaseと未知phaseはNone。"""
    phase = _phase_of(event)
    if phase is None or phase not in _STAGE_BY_PHASE:
        return None
    if phase == "verifying_input" and event.get("checkpoint") in _SAVE_VERIFY_CHECKPOINTS:
        return Stage.SAVE
    return _STAGE_BY_PHASE[phase]


def translate_progress(event: object) -> StageProgress | None:
    """進捗イベントを工程表示へ翻訳する。終端phaseと未知phaseはNone。"""
    stage = stage_of(event)
    if stage is None or not isinstance(event, Mapping):
        return None
    fraction = None if stage is Stage.SEPARATE else _fraction_of(event)
    return StageProgress(
        stage=stage,
        fraction=fraction,
        awaiting_model_fetch=_phase_of(event) == "downloading_model",
    )


def is_terminal(event: object) -> bool:
    """この進捗イベントが終端phaseか。

    行が終わったかの判定には使えない。既存出力の再利用・検証済みの空出力は
    PREPARE途中の早期returnで終わり、worker_unavailable等のキュー段の失敗は
    workerに届いていないため、いずれも終端phaseを流さない。
    行の終端はストリームではなく結果（classify_result）で確定させる。
    """
    return _phase_of(event) in TERMINAL_PHASES


class FailureScope(Enum):
    """失敗がその音源だけの問題か、セッションを続けられない問題か。"""

    SOURCE = "source"
    SESSION = "session"


class SourceFailure(Enum):
    """行レベルの失敗の種別。"""

    FAILED = "failed"  # 再試行できる
    UNSUPPORTED = "unsupported"  # その音源では成立しない


class HaltCode(Enum):
    """セッションを止める条件。凍結契約が持つ3種類だけ。"""

    DISK_FULL = "disk_full"
    FETCH_FAILED = "fetch_failed"
    ENGINE_STOPPED = "engine_stopped"


class SourceOutcome(Enum):
    """1件の音源の終端。"""

    SAVED = "saved"
    ALREADY_SEPARATED = "already_separated"
    NO_TARGET_FOUND = "no_target_found"  # 検証済みの空出力。失敗ではない
    FAILED = "failed"


@dataclass(frozen=True)
class Failure:
    scope: FailureScope
    source_failure: SourceFailure | None = None
    halt: HaltCode | None = None
    recognized: bool = True

    @property
    def retryable(self) -> bool:
        """行の再試行が意味を持つか。

        HALTの場ではRETRY/SKIPで回復する。SKIPした後や再起動後に残った行も、環境
        （空き容量・接続・エンジン）が戻れば同じ音源で成立しうるので、行から再試行できる
        （凍結Designでも、分離できない理由を持たない失敗行はRETRYを持つ）。
        """
        return self.source_failure is SourceFailure.FAILED or self.scope is FailureScope.SESSION


def _source(kind: SourceFailure) -> Failure:
    return Failure(scope=FailureScope.SOURCE, source_failure=kind)


def _halt(code: HaltCode) -> Failure:
    return Failure(scope=FailureScope.SESSION, halt=code)


# 現行のworker/jobsが発行するerror_codeだけを明示的に分類する。
# 判定基準は「次の音源でも同じように起きるか」。環境・実行基盤の問題はセッション、
# その音源に固有の問題は行レベル。すべてをHALTへ倒さない。
_FAILURES: dict[str, Failure] = {
    # 行レベル・再試行できる
    "input_changed": _source(SourceFailure.FAILED),
    "input_decode_failed": _source(SourceFailure.FAILED),
    "input_decode_timeout": _source(SourceFailure.FAILED),
    "file_not_found": _source(SourceFailure.FAILED),
    "processing_failed": _source(SourceFailure.FAILED),
    # 処理中に LSS を終了した。その1件は保存していないので、次回に再試行できる
    "job_abandoned": _source(SourceFailure.FAILED),
    # 単話BGM候補の工程。環境起因の失敗はworkerのerror_codeのまま届き、上の区分でHALTする
    "bgm_candidate_processing_failed": _source(SourceFailure.FAILED),
    "invalid_bgm_candidate_response": _source(SourceFailure.FAILED),
    # 解析器の推論・出力がその音源で契約に合わなかった（CUDA・重み・workerの問題は別のcode）
    "bgm_analysis_failed": _source(SourceFailure.FAILED),
    # 欲しい音の分離・確認がその音源で契約に合わなかった（CUDA・重み・workerの問題は別のcode）
    "sound_extraction_failed": _source(SourceFailure.FAILED),
    # 行レベル・その音源では成立しない
    "unsupported_channels": _source(SourceFailure.UNSUPPORTED),
    "unsafe_output_serialization": _source(SourceFailure.UNSUPPORTED),
    # 入力が空・不完全・NaN/Infを含む。同じ音源を再試行しても変わらない
    "invalid_input_signal": _source(SourceFailure.UNSUPPORTED),
    # 欲しい結果の処理経路が無い（その結果はこの構成では作れない）
    "capability_unsupported": _source(SourceFailure.UNSUPPORTED),
    # ステムの組み合わせがこのmodelの出力では規則上作れない（同じ音源で再試行しても変わらない）
    "stem_combination_unsupported": _source(SourceFailure.UNSUPPORTED),
    # 組み合わせの部品ステムが欠けた・揃わなかった。部分的な組み合わせは保存していない
    "stem_combination_incomplete": _source(SourceFailure.FAILED),
    # 保存先・キャッシュの空き容量
    "insufficient_output_space": _halt(HaltCode.DISK_FULL),
    "insufficient_model_cache_space": _halt(HaltCode.DISK_FULL),
    # 処理に必要なファイルの取得
    "invalid_model_source": _halt(HaltCode.FETCH_FAILED),
    "model_download_failed": _halt(HaltCode.FETCH_FAILED),
    "model_download_invalid": _halt(HaltCode.FETCH_FAILED),
    "model_download_too_large": _halt(HaltCode.FETCH_FAILED),
    # 分離エンジンが動かない。品質を落とした代替経路へ倒さない
    "gpu_unavailable": _halt(HaltCode.ENGINE_STOPPED),
    "decoder_unavailable": _halt(HaltCode.ENGINE_STOPPED),
    "worker_unavailable": _halt(HaltCode.ENGINE_STOPPED),
    "worker_shutting_down": _halt(HaltCode.ENGINE_STOPPED),
    "dispatch_failed": _halt(HaltCode.ENGINE_STOPPED),
    "invalid_worker_response": _halt(HaltCode.ENGINE_STOPPED),
    "invalid_separation_output": _halt(HaltCode.ENGINE_STOPPED),
    "invalid_job": _halt(HaltCode.ENGINE_STOPPED),
    "invalid_model_ledger": _halt(HaltCode.ENGINE_STOPPED),
    "invalid_stem_target": _halt(HaltCode.ENGINE_STOPPED),
    "unknown_model": _halt(HaltCode.ENGINE_STOPPED),
    "model_unavailable": _halt(HaltCode.ENGINE_STOPPED),
}

KNOWN_ERROR_CODES = frozenset(_FAILURES)

# 未知のerror_codeは、現行のキュー挙動（その音源を失敗にして次へ進む）に合わせる。
# recognized=Falseで境界の欠落を明示し、静かにHALTへ倒さない。
_UNRECOGNIZED = Failure(
    scope=FailureScope.SOURCE, source_failure=SourceFailure.FAILED, recognized=False,
)


def classify_failure(error_code: object) -> Failure:
    if not isinstance(error_code, str):
        return _UNRECOGNIZED
    return _FAILURES.get(error_code, _UNRECOGNIZED)


@dataclass(frozen=True)
class Result:
    outcome: SourceOutcome
    failure: Failure | None = None


def classify_result(result: object) -> Result:
    """workerの結果を行の終端へ翻訳する。

    現行のworkerは「同一内容が処理済み」と「検証済みの空出力」の両方をskippedで返す。
    後者は正当な結果であり「すでに分離済み」ではないため、別の終端として分けておく。
    出荷UIに空出力の面は無いが、両者を同じ終端へ潰すと表示が事実と食い違う。
    """
    if not isinstance(result, Mapping):
        return Result(SourceOutcome.FAILED, _UNRECOGNIZED)
    status = result.get("status")
    if status in {"ok", "skipped"}:
        # jobs側の検証はok/skippedの双方でzero_resultを認めるため、statusより先に見る。
        if result.get("zero_result") is not None:
            return Result(SourceOutcome.NO_TARGET_FOUND)
        return Result(SourceOutcome.SAVED if status == "ok" else SourceOutcome.ALREADY_SEPARATED)
    if status == "error":
        return Result(SourceOutcome.FAILED, classify_failure(result.get("error_code")))
    return Result(SourceOutcome.FAILED, _UNRECOGNIZED)
