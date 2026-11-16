"""実装とUI意味論の翻訳境界を回帰させない。"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from lss.ui import contract
from lss.ui.contract import (
    FailureScope,
    HaltCode,
    SourceFailure,
    SourceOutcome,
    Stage,
)


ROOT = Path(__file__).resolve().parents[1]
LSS = ROOT / "lss"


def _emitted_phases() -> set[str]:
    """現行実装が実際に発行するphaseを収集する。"""
    phases: set[str] = set()
    for path in sorted(LSS.rglob("*.py")):
        if path.is_relative_to(LSS / "ui"):
            continue
        text = path.read_text(encoding="utf-8")
        phases |= set(re.findall(r'"phase":\s*"([a-z_]+)"', text))
        if "def report(phase" in text:
            phases |= set(re.findall(r'report\(\s*"([a-z_]+)"', text))
    return phases


def _emitted_error_codes() -> set[str]:
    codes: set[str] = set()
    for name in ("core/worker.py", "core/jobs.py"):
        text = (LSS / name).read_text(encoding="utf-8")
        codes |= set(re.findall(r'"error_code":\s*"([a-z_]+)"', text))
        if "def _error_code" in text:
            body = text.split("def _error_code")[1].split("\ndef ")[0]
            codes |= set(re.findall(r'return\s+"([a-z_]+)"', body))
    return codes


def test_every_emitted_phase_is_classified() -> None:
    emitted = _emitted_phases()
    assert emitted, "phaseを1つも収集できていない"
    assert emitted <= contract.KNOWN_PHASES


def _emitted_verify_checkpoints() -> set[str]:
    """verifying_inputに実際に渡されるcheckpointを収集する。"""
    text = (LSS / "core" / "worker.py").read_text(encoding="utf-8")
    return set(re.findall(r'checkpoint="([a-z_]+)"', text))


def test_every_emitted_verify_checkpoint_is_classified() -> None:
    emitted = _emitted_verify_checkpoints()
    assert emitted, "checkpointを1つも収集できていない"
    # 未分類のcheckpointは黙ってPREPAREへ落ち、SAVE→PREPAREの逆行として見える。
    assert emitted <= contract.KNOWN_VERIFY_CHECKPOINTS


def test_every_emitted_error_code_is_classified() -> None:
    emitted = _emitted_error_codes()
    assert emitted, "error_codeを1つも収集できていない"
    assert emitted <= contract.KNOWN_ERROR_CODES


def test_stages_are_exactly_the_three_shown_to_the_user() -> None:
    assert [stage.name for stage in Stage] == ["PREPARE", "SEPARATE", "SAVE"]


@pytest.mark.parametrize(
    "event, stage",
    [
        ({"phase": "preparing_input"}, Stage.PREPARE),
        ({"phase": "hashing_input", "completed_bytes": 1, "total_bytes": 2}, Stage.PREPARE),
        ({"phase": "checking_model", "model_id": "m"}, Stage.PREPARE),
        ({"phase": "downloading_model", "asset": "weight"}, Stage.PREPARE),
        ({"phase": "checking_runtime", "provider": "msst"}, Stage.PREPARE),
        ({"phase": "checking_completed_output", "output": "a.flac"}, Stage.PREPARE),
        ({"phase": "verifying_input", "checkpoint": "after_decode"}, Stage.PREPARE),
        ({"phase": "verifying_input", "checkpoint": "before_reuse"}, Stage.PREPARE),
        ({"phase": "verifying_input", "checkpoint": "after_metadata"}, Stage.PREPARE),
        ({"phase": "separating", "model_id": "m"}, Stage.SEPARATE),
        ({"phase": "runtime_ready", "runtime_id": "r"}, Stage.SEPARATE),
        ({"phase": "staging_input", "completed_bytes": 1, "total_bytes": 4}, Stage.SEPARATE),
        ({"phase": "forward_rest", "allocated_bytes": 1, "reserved_bytes": 2, "sleep_seconds": 1.0}, Stage.SEPARATE),
        ({"phase": "verifying_input", "checkpoint": "after_separation"}, Stage.SAVE),
        ({"phase": "verifying_input", "checkpoint": "before_publish"}, Stage.SAVE),
        ({"phase": "analyzing_outputs", "completed_frames": 1, "total_frames": 2}, Stage.SAVE),
        ({"phase": "checking_existing_output", "stem": "s"}, Stage.SAVE),
        ({"phase": "writing_outputs", "completed_stems": 1, "total_stems": 2}, Stage.SAVE),
        ({"phase": "recording_completion", "output": "a.flac"}, Stage.SAVE),
    ],
)
def test_progress_events_map_to_the_expected_stage(event: dict, stage: Stage) -> None:
    progress = contract.translate_progress(event)
    assert progress is not None
    assert progress.stage is stage


@pytest.mark.parametrize(
    "event",
    [
        {"phase": "separating", "model_id": "m"},
        {"phase": "runtime_ready", "runtime_id": "r"},
        {"phase": "staging_input", "completed_bytes": 3, "total_bytes": 4},
        {"phase": "forward_rest", "completed_bytes": 3, "total_bytes": 4},
    ],
)
def test_separate_never_exposes_measurable_progress(event: dict) -> None:
    progress = contract.translate_progress(event)
    assert progress is not None
    assert progress.stage is Stage.SEPARATE
    assert progress.fraction is None
    assert progress.determinate is False


@pytest.mark.parametrize(
    "event, fraction",
    [
        ({"phase": "hashing_input", "completed_bytes": 1, "total_bytes": 4}, 0.25),
        ({"phase": "downloading_model", "completed_bytes": 5, "total_bytes": 10}, 0.5),
        ({"phase": "analyzing_outputs", "completed_frames": 3, "total_frames": 4}, 0.75),
        ({"phase": "writing_outputs", "completed_stems": 1, "total_stems": 2}, 0.5),
        ({"phase": "recording_completion", "completed_bytes": 9, "total_bytes": 9}, 1.0),
    ],
)
def test_measurable_stages_expose_a_ratio_only(event: dict, fraction: float) -> None:
    progress = contract.translate_progress(event)
    assert progress is not None
    assert progress.fraction == pytest.approx(fraction)
    assert progress.determinate is True


@pytest.mark.parametrize(
    "event",
    [
        {"phase": "preparing_input"},
        {"phase": "checking_model", "model_id": "m"},
        {"phase": "writing_outputs", "completed_stems": 1, "total_stems": 0},
        {"phase": "hashing_input", "completed_bytes": "1", "total_bytes": "2"},
    ],
)
def test_unmeasurable_events_stay_indeterminate(event: dict) -> None:
    progress = contract.translate_progress(event)
    assert progress is not None
    assert progress.fraction is None


def test_fraction_is_clamped_to_the_unit_range() -> None:
    progress = contract.translate_progress(
        {"phase": "hashing_input", "completed_bytes": 12, "total_bytes": 10}
    )
    assert progress is not None
    assert progress.fraction == 1.0


def test_model_fetch_is_the_only_first_run_wait_signal() -> None:
    fetching = contract.translate_progress({"phase": "downloading_model", "asset": "weight"})
    other = contract.translate_progress({"phase": "checking_model", "model_id": "m"})
    assert fetching is not None and fetching.awaiting_model_fetch is True
    assert other is not None and other.awaiting_model_fetch is False


@pytest.mark.parametrize(
    "event",
    [
        {"phase": "completed", "output_count": 1},
        {"phase": "failed", "error_code": "processing_failed", "message": "x"},
    ],
)
def test_terminal_events_are_not_stages(event: dict) -> None:
    assert contract.translate_progress(event) is None
    assert contract.stage_of(event) is None
    assert contract.is_terminal(event) is True


@pytest.mark.parametrize("event", [None, {}, {"phase": ""}, {"phase": "future_phase"}, "x"])
def test_unknown_events_do_not_produce_a_stage(event: object) -> None:
    assert contract.translate_progress(event) is None
    assert contract.is_terminal(event) is False


@pytest.mark.parametrize(
    "error_code, halt",
    [
        ("insufficient_output_space", HaltCode.DISK_FULL),
        ("insufficient_model_cache_space", HaltCode.DISK_FULL),
        ("model_download_failed", HaltCode.FETCH_FAILED),
        ("model_download_invalid", HaltCode.FETCH_FAILED),
        ("model_download_too_large", HaltCode.FETCH_FAILED),
        ("invalid_model_source", HaltCode.FETCH_FAILED),
        ("gpu_unavailable", HaltCode.ENGINE_STOPPED),
        ("worker_unavailable", HaltCode.ENGINE_STOPPED),
        ("dispatch_failed", HaltCode.ENGINE_STOPPED),
        ("invalid_worker_response", HaltCode.ENGINE_STOPPED),
        ("invalid_separation_output", HaltCode.ENGINE_STOPPED),
        ("decoder_unavailable", HaltCode.ENGINE_STOPPED),
        ("unknown_model", HaltCode.ENGINE_STOPPED),
        ("invalid_model_ledger", HaltCode.ENGINE_STOPPED),
    ],
)
def test_environment_failures_halt_the_session(error_code: str, halt: HaltCode) -> None:
    failure = contract.classify_failure(error_code)
    assert failure.scope is FailureScope.SESSION
    assert failure.halt is halt
    assert failure.source_failure is None
    # SKIP後・再起動後に行へ残った音源は、環境が戻れば行から再試行できる。
    assert failure.retryable is True


@pytest.mark.parametrize(
    "error_code, kind, retryable",
    [
        ("input_changed", SourceFailure.FAILED, True),
        ("input_decode_failed", SourceFailure.FAILED, True),
        ("input_decode_timeout", SourceFailure.FAILED, True),
        ("file_not_found", SourceFailure.FAILED, True),
        ("processing_failed", SourceFailure.FAILED, True),
        ("unsupported_channels", SourceFailure.UNSUPPORTED, False),
        ("unsafe_output_serialization", SourceFailure.UNSUPPORTED, False),
    ],
)
def test_source_failures_stay_at_the_row(
    error_code: str, kind: SourceFailure, retryable: bool
) -> None:
    failure = contract.classify_failure(error_code)
    assert failure.scope is FailureScope.SOURCE
    assert failure.source_failure is kind
    assert failure.halt is None
    assert failure.retryable is retryable


def test_not_every_failure_becomes_a_halt() -> None:
    scopes = {contract.classify_failure(code).scope for code in contract.KNOWN_ERROR_CODES}
    assert scopes == {FailureScope.SOURCE, FailureScope.SESSION}


@pytest.mark.parametrize("error_code", [None, "", "future_error", 7])
def test_unknown_failures_stay_at_the_row_and_are_marked(error_code: object) -> None:
    failure = contract.classify_failure(error_code)
    assert failure.scope is FailureScope.SOURCE
    assert failure.halt is None
    assert failure.recognized is False


def test_halt_codes_are_exactly_the_frozen_three() -> None:
    assert {code.name for code in HaltCode} == {"DISK_FULL", "FETCH_FAILED", "ENGINE_STOPPED"}


def test_successful_result_is_saved() -> None:
    result = contract.classify_result({"status": "ok", "outputs": ["a.flac"]})
    assert result.outcome is SourceOutcome.SAVED
    assert result.failure is None


def test_duplicate_and_zero_result_are_different_outcomes() -> None:
    duplicate = contract.classify_result({"status": "skipped", "outputs": ["a.flac"]})
    zero = contract.classify_result({
        "status": "skipped",
        "outputs": [],
        "zero_result": {"reason": "no_target_content", "validation_result": {}},
    })
    assert duplicate.outcome is SourceOutcome.ALREADY_SEPARATED
    assert zero.outcome is SourceOutcome.NO_TARGET_FOUND
    assert zero.failure is None


def test_error_result_carries_its_classification() -> None:
    result = contract.classify_result({
        "status": "error", "error_code": "insufficient_output_space", "message": "x",
    })
    assert result.outcome is SourceOutcome.FAILED
    assert result.failure is not None
    assert result.failure.halt is HaltCode.DISK_FULL


@pytest.mark.parametrize("result", [None, {}, {"status": "weird"}, "x"])
def test_malformed_results_fail_the_row_without_inventing_a_halt(result: object) -> None:
    classified = contract.classify_result(result)
    assert classified.outcome is SourceOutcome.FAILED
    assert classified.failure is not None
    assert classified.failure.scope is FailureScope.SOURCE
    assert classified.failure.recognized is False


def test_row_termination_comes_from_the_result_not_the_stream() -> None:
    """終端phaseが流れない終わり方があるため、行の終端は結果で確定させる。"""
    # 既存出力を再利用して終わる音源が最後に流すphase。終端phaseは流れない。
    last_streamed = {"phase": "checking_completed_output", "output": "a.flac"}
    assert contract.is_terminal(last_streamed) is False
    assert contract.stage_of(last_streamed) is Stage.PREPARE
    assert contract.classify_result(
        {"status": "skipped", "outputs": ["a.flac"]}
    ).outcome is SourceOutcome.ALREADY_SEPARATED
    # workerへ届かなかった失敗も進捗を持たない。HALT判定は結果から入る。
    queue_failure = contract.classify_result(
        {"status": "error", "error_code": "worker_unavailable", "message": "x"}
    )
    assert queue_failure.outcome is SourceOutcome.FAILED
    assert queue_failure.failure is not None
    assert queue_failure.failure.halt is HaltCode.ENGINE_STOPPED


def test_result_outcomes_are_exactly_the_frozen_four() -> None:
    assert {outcome.name for outcome in SourceOutcome} == {
        "SAVED", "ALREADY_SEPARATED", "NO_TARGET_FOUND", "FAILED",
    }


def test_contract_exposes_no_forbidden_shipping_capability() -> None:
    """出荷UIが持たない能力を、翻訳境界の公開語彙から作らない。"""
    names = [name for name in dir(contract) if not name.startswith("_")]
    values = [
        member.value
        for name in names
        for member in getattr(getattr(contract, name), "__members__", {}).values()
    ]
    forbidden = re.compile(
        'vocal|instrumental|percent|\x08eta\x08|remaining|estimate|'
        r"select|setting|preview|reprocess|overwrite",
        re.IGNORECASE,
    )
    assert not [word for word in names + values if forbidden.search(str(word))]


def test_contract_does_not_depend_on_qt() -> None:
    source = (LSS / "ui" / "contract.py").read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if re.match(r"\s*(import|from)\s", line)]
    assert imports
    assert not [line for line in imports if "PySide6" in line or "Qt" in line]
