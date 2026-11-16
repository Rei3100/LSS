"""LSS GUIエントリポイント(PyInstaller Analysisの対象スクリプト)。"""
import os
import sys
from contextlib import contextmanager


def _configure_runtime_caches() -> None:
    """凍結実体でもJIT/描画依存が書けるLSS管理下のcacheを先に選ぶ。"""
    from pathlib import Path

    from lss.paths import app_root, is_frozen

    state_root = Path(os.environ.get("LSS_STATE_DIR", app_root() / ".state"))
    cache_root = state_root / "runtime-cache"
    numba_cache = Path(os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba")))
    if is_frozen():
        # PyInstallerのmatplotlib runtime hookが置く一時directoryは、windowed
        # processから書けない場合がある。凍結版はLSS管理下へ必ず戻す。
        os.environ["MPLCONFIGDIR"] = str(cache_root / "matplotlib")
    matplotlib_cache = Path(os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib")))
    numba_cache.mkdir(parents=True, exist_ok=True)
    matplotlib_cache.mkdir(parents=True, exist_ok=True)


@contextmanager
def _frozen_inference_directory():
    """凍結moduleの相対source pathを、同梱した物理sourceへ解決可能にする。"""
    from lss.paths import is_frozen, resource_root

    if not is_frozen():
        yield
        return
    previous = os.getcwd()
    os.chdir(resource_root())
    try:
        yield
    finally:
        os.chdir(previous)


def _run_bgm_candidates(source: str) -> int:
    """配備実体から単話BGM候補を実行し、次回起動後も読める結果票を残す。"""
    from pathlib import Path

    from lss import application
    from lss.core import session

    source_path = Path(source).resolve()
    request = application.ProcessingRequest(
        (source_path,), application.UserIntent.EXTRACT_BGM_CUES,
    )
    with _frozen_inference_directory():
        outcome = application.run_bgm_candidate_request(request)
    result_path = session.state_dir() / "bgm-candidate-result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "bgm_candidate_entry_result",
        "schema_version": 1,
        "source": str(source_path),
        "disposition": outcome.disposition.value,
        "outputs": [str(path) for path in outcome.outputs],
        "product_result": outcome.product_result,
        "held_artifacts": [str(path) for path in outcome.held_artifacts],
        "held_alternative_artifacts": [
            str(path) for path in outcome.held_alternative_artifacts
        ],
        "evidence_path": str(outcome.evidence_path) if outcome.evidence_path else None,
        "record_path": str(outcome.record_path) if outcome.record_path else None,
        "recipe_id": outcome.recipe_id,
        "zero_result": outcome.zero_result,
        "error_code": outcome.error_code,
        "message": outcome.message,
        "reused": outcome.reused,
    }
    session._write_json_atomic(result_path, payload)
    if sys.stdout is not None:
        if outcome.disposition is application.ResultDisposition.PUBLISHED:
            print(f"BGM cueを{len(outcome.outputs)}件保存しました: {result_path}")
        elif outcome.disposition is application.ResultDisposition.HELD:
            print(f"BGM候補を{len(outcome.held_artifacts)}件保留しました: {result_path}")
        elif outcome.disposition is application.ResultDisposition.ZERO:
            print(f"検証済みの対象なし結果を保存しました: {result_path}")
        else:
            print(f"BGM候補処理に失敗しました: {outcome.message or outcome.error_code}")
    return 0 if outcome.disposition in {
        application.ResultDisposition.PUBLISHED, application.ResultDisposition.HELD,
        application.ResultDisposition.ZERO,
    } else 1


def _restore_previous_bgm(source: str) -> int:
    """原本のBGM公開結果を直前に置き換えた結果へ戻し（rollback）、結果票を残す。もう一度で元へ戻る。"""
    from pathlib import Path

    from lss import application
    from lss.core import session

    source_path = Path(source).resolve()
    outcome = application.restore_previous_bgm_result(source_path)
    result_path = session.state_dir() / "bgm-restore-result.json"
    session._write_json_atomic(result_path, {
        "type": "bgm_restore_previous_result", "schema_version": 1, "source": str(source_path),
        "disposition": outcome.disposition.value, "outputs": [str(path) for path in outcome.outputs],
        "product_result": outcome.product_result,
        "record_path": str(outcome.record_path) if outcome.record_path else None,
        "recipe_id": outcome.recipe_id, "error_code": outcome.error_code, "message": outcome.message})
    if sys.stdout is not None:
        print(f"{outcome.message or outcome.error_code}: {result_path}")
    return 0 if outcome.disposition is not application.ResultDisposition.FAILED else 1


def _run_scene_naming(state_root: str, source: str, source_sha256: str) -> int:
    """Run bounded post-publication naming attempts in a short-lived packaged worker process."""
    import json
    from pathlib import Path

    from lss.bgm import scene_job

    state_path = Path(state_root).resolve()
    source_path = Path(source).resolve()
    with _frozen_inference_directory():
        result = scene_job.run_background_worker(state_path, source_path, source_sha256)
    if sys.stdout is not None:
        print(json.dumps({"state": "completed", **result}, ensure_ascii=False))
    return 0


def _run_scene_naming_recovery(state_root: str) -> int:
    """Run one serialized recovery coordinator for current published scene names."""
    import json
    from pathlib import Path

    from lss.bgm import scene_job

    with _frozen_inference_directory():
        result = scene_job.run_current_best_recovery(Path(state_root))
    if sys.stdout is not None:
        print(json.dumps(result, ensure_ascii=False))
    return 0


def _ensure_worker_streams() -> None:
    """windowedビルドでNoneになる標準ストリームを、ワーカー用に破棄先へ接続する。"""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _scene_job_log_path(state_root: str, source_sha256: str):
    """Return only the fixed per-source log path after validating its filename identity."""
    from pathlib import Path

    from lss.bgm.scene_names import DIRECTORY

    if len(source_sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in source_sha256):
        raise ValueError("scene_naming_source_hash_invalid")
    return Path(state_root).resolve() / DIRECTORY / "jobs" / f"{source_sha256[:16].lower()}.log"


def _stream_needs_scene_log(stream) -> bool:
    if stream is None:
        return True
    try:
        return bool(getattr(stream, "closed", False))
    except Exception:  # noqa: BLE001 - an unusable stream should be rebound
        return True


def _bind_scene_job_log(state_root: str, source_sha256: str) -> None:
    """Bind only missing/closed streams to this source's UTF-8 append-only job log."""
    stdout_missing = _stream_needs_scene_log(sys.stdout)
    stderr_missing = _stream_needs_scene_log(sys.stderr)
    if not stdout_missing and not stderr_missing:
        return
    log_path = _scene_job_log_path(state_root, source_sha256)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    if stdout_missing:
        sys.stdout = stream
    if stderr_missing:
        sys.stderr = stream


def _bind_scene_recovery_log(state_root: str) -> None:
    """Bind missing windowed streams to the coordinator's isolated durable log."""
    from pathlib import Path

    from lss.bgm.scene_job import _recovery_log_path

    if not _stream_needs_scene_log(sys.stdout) and not _stream_needs_scene_log(sys.stderr):
        return
    log_path = _recovery_log_path(Path(state_root).resolve())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    if _stream_needs_scene_log(sys.stdout):
        sys.stdout = stream
    if _stream_needs_scene_log(sys.stderr):
        sys.stderr = stream


def _report_scene_job_failure(exc: Exception) -> None:
    """Best-effort ASCII JSON to an existing usable stderr; diagnostics never escape."""
    try:
        stream = sys.stderr
        if stream is None or bool(getattr(stream, "closed", False)):
            return
        import json

        print(json.dumps({"state": "failed", "operation": "scene_naming",
                          "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=True),
              file=stream, flush=True)
    except Exception:  # noqa: BLE001 - broken or encoding-hostile diagnostics must stay contained
        pass


def _run_scene_naming_status(report_path: str) -> int:
    """Write only the explicitly requested new diagnostic report; never alter user state."""
    import json
    from pathlib import Path
    from lss.bgm import scene_runtime

    result = scene_runtime.read_only_status()
    with Path(report_path).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=True, indent=2)
        stream.write("\n")
    return 0 if result["status"] == "observed" else 1


def main(argv: list[str] | None = None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--scene-naming-status":
        # A windowed EXE can have no stdout. Only an explicit new report file is written;
        # normal cache, worker-log, acquisition and GUI setup are bypassed.
        try:
            if len(args) != 2:
                raise ValueError("scene_naming_status_arguments_invalid")
            return _run_scene_naming_status(args[1])
        except Exception as exc:
            _report_scene_job_failure(exc)
            return 1
    if args and args[0] == "--scene-naming-recovery":
        try:
            if len(args) != 2:
                raise ValueError("scene_naming_recovery_arguments_invalid")
            _bind_scene_recovery_log(args[1])
            _configure_runtime_caches()
            return _run_scene_naming_recovery(args[1])
        except Exception as exc:
            _report_scene_job_failure(exc)
            return 1
    if args and args[0] == "--scene-naming":
        # A windowed executable displays a bootloader error dialog for an uncaught
        # exception. This background job must fail without opening a user window,
        # including cache setup and malformed internal arguments.
        try:
            if len(args) != 4:
                raise ValueError("scene_naming_arguments_invalid")
            if len(args[3]) != 64 or any(char not in "0123456789abcdefABCDEF" for char in args[3]):
                raise ValueError("scene_naming_source_hash_invalid")
            _bind_scene_job_log(args[1], args[3])
            _configure_runtime_caches()
            return _run_scene_naming(args[1], args[2], args[3])
        except Exception as exc:
            _report_scene_job_failure(exc)
            return 1
    # librosaを含む処理moduleより前に設定する。PyInstallerのPYZにはNumbaが
    # source locatorを持てないため、既定cacheのままでは初回JITで停止する。
    _configure_runtime_caches()
    if args and args[0] == "--worker":
        _ensure_worker_streams()
        from lss.core.worker import main as worker_main
        with _frozen_inference_directory():
            return worker_main(args[1:])
    if len(args) == 2 and args[0] == "--bgm-candidates":
        _ensure_worker_streams()
        return _run_bgm_candidates(args[1])
    if len(args) == 2 and args[0] == "--bgm-restore-previous":
        _ensure_worker_streams()
        return _restore_previous_bgm(args[1])
    if (len(args) == 2 and args[0] == "--restore-backup") or len(args) == 1:
        # 保存状態のバックアップからの復元。zipをLSS.exeへドラッグしても同じ入口になる。
        from lss.assets import restore_entry
        target = args[-1]
        if args[0] == "--restore-backup" or restore_entry.is_backup_archive(target):
            from pathlib import Path
            return restore_entry.run(Path(target))
    from lss.ui.app import main as gui_main
    return gui_main()

if __name__ == "__main__":
    raise SystemExit(main())
