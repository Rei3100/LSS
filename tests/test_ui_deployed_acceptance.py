"""配備された画面から、実GPUで複数の実曲を処理する受入。

LSS_RUN_UI_ACCEPTANCE=1で明示実行。配備されたEXEのコピーと同梱リソースを使い、
可変の状態・入力のコピー・成果は一時領域へ置く。画面はOSの本物のウィンドウ・
ダイアログ・トレイのメニュー窓で、操作はキー・ポインタ・ダイアログ入力で行う。
音楽品質の採用評価ではない。

証拠（各状態の画面・工程・処理中の描画間隔）は LSS_UI_ACCEPTANCE_EVIDENCE の
フォルダへ残す（未指定なら一時領域）。
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root


import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.core import catalog, client, session
from lss.harness import fetch, infer, tsv_io


pytestmark = pytest.mark.skipif(
    os.environ.get("LSS_RUN_UI_ACCEPTANCE") != "1",
    reason="配備された画面と実GPU・実曲を使う受入は明示実行",
)

ENTER = dict(key="Enter", code="Enter", vk=13)
ESCAPE = dict(key="Escape", code="Escape", vk=27)
DOWN = dict(key="ArrowDown", code="ArrowDown", vk=40)


def _clip(original: Path, destination: Path, seconds: float) -> None:
    info = sf.info(original)
    frames = min(info.frames, int(seconds * info.samplerate))
    data, rate = sf.read(original, frames=frames, always_2d=True)
    sf.write(destination, data, rate, subtype=info.subtype)


def _queue_statuses(state: Path) -> list:
    try:
        payload = json.loads((state / "queue.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [(Path(item["path"]).name, item["status"]) for item in payload.get("items", [])]


def _portable(tmp_path: Path, request) -> tuple[Path, Path]:
    """配備された実行物のコピーと、同梱リソースへのjunction。大容量のCUDA資産は複製しない。"""
    import _winapi

    deployed = Path(os.environ["LSS_ACCEPTANCE_EXECUTABLE"])
    assert deployed.is_file() and (deployed.parent / "_internal").is_dir()
    # 既に動いている常駐ワーカーへ隔離した画面がつながると、別の状態へ書いてしまう。
    assert not client.is_worker_running(), "既定の接続先でLSSのワーカーが動いています"
    isolated = tmp_path / "portable"
    isolated.mkdir()
    shutil.copyfile(deployed, isolated / "LSS.exe")
    junction = isolated / "_internal"
    _winapi.CreateJunction(str(deployed.parent / "_internal"), str(junction))
    request.addfinalizer(junction.rmdir)
    return deployed, isolated / "LSS.exe"


def _offline_environment(state: Path, models: Path, **extra: str) -> dict[str, str]:
    """開発用Pythonを見えなくし、外部への通信は閉じたプロキシで失敗させる。"""
    environment = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith(("PYTHON", "VIRTUAL_ENV", "CONDA"))
    }
    environment["PATH"] = os.pathsep.join(
        part for part in os.environ.get("PATH", "").split(os.pathsep)
        if part and "python" not in part.lower() and ".venv" not in part.lower())
    environment.update({
        "LSS_MODEL_DIR": str(models), "LSS_STATE_DIR": str(state), "LSS_MIN_FREE_BYTES": "0",
        "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "NO_PROXY": "",
        **extra,
    })
    return environment


def _link_model(models: Path, *assets: Path) -> None:
    for asset in assets:
        target = models / asset.parent.name / asset.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(asset, target)


def _evidence(tmp_path: Path, name: str) -> Path:
    base = os.environ.get("LSS_UI_ACCEPTANCE_EVIDENCE")
    path = Path(base) / name if base else tmp_path / "evidence"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_deployed_screen_processes_songs_with_stop_restart_retry_and_reuse(tmp_path, request):
    from tests.deployed_ui_driver import DeployedApp, wait_for

    repository = Path(__file__).resolve().parents[1]
    evidence = _evidence(tmp_path, "songs")
    deployed, executable = _portable(tmp_path, request)

    default = catalog.default_model()
    checkpoint, config = fetch.model_paths(
        default["model_id"], default["distributor_url"], default["config_url"])
    assert checkpoint.is_file() and config.is_file(), "取得済みの既定モデルが必要です"

    _, corpus = tsv_io.load_tsv(dev_evidence_root(repository) / "ledger" / "test_set.tsv")
    originals = [Path(row["mix_path"]) for row in corpus[:4]]
    inbox = tmp_path / "入力"
    folder = inbox / "まとめて"
    folder.mkdir(parents=True)
    full = inbox / ("実曲A" + originals[0].suffix)
    shutil.copyfile(originals[0], full)
    retry_source = inbox / ("実曲D" + originals[3].suffix)
    _clip(originals[3], retry_source, 30)
    batch = [folder / ("実曲B" + originals[1].suffix), folder / ("実曲C" + originals[2].suffix)]
    _clip(originals[1], batch[0], 40)
    _clip(originals[2], batch[1], 40)
    sources = [full, retry_source, *batch]
    for source in sources:
        assert sf.info(source).channels == 2
    digests = {source: session.input_sha256_of(source) for source in sources}
    originals_before = {path: path.stat().st_mtime_ns for path in originals}

    state = tmp_path / "state"
    models = tmp_path / "models"
    # 初回のモデル取得は閉じたプロキシで失敗させる。
    environment = _offline_environment(state, models)
    if os.environ.get("LSS_UI_ACCEPTANCE_EMPTY_MODEL_DIR") != "1":
        # 設定の取得失敗で登録できない不具合を直す前の実行物向け。重みだけを未取得にする。
        _link_model(models, config)
    app = DeployedApp(executable, environment)
    record: dict = {
        "executable": str(deployed), "executable_sha256": session.input_sha256_of(deployed),
        "sources": {source.name: {"sha256": digests[source], "seconds": round(sf.info(source).duration, 3)}
                    for source in sources},
        "steps": [],
    }

    def note(step: str, **facts) -> None:
        record["steps"].append({"step": step, "t": round(time.monotonic() - began, 1), **facts})

    def row_of(state_view: dict, source: Path) -> str:
        matches = [ident for ident in state_view["queue"]
                   if state_view["songs"][ident].get("path") == str(source)]
        assert len(matches) == 1, (source, state_view["queue"])
        return matches[0]

    def settled(page, phase: str, timeout: float):
        try:
            return wait_for(lambda: (lambda view: view if view and view["phase"] == phase else None)(
                page.state()), timeout, f"{phase}への遷移")
        except (OSError, RuntimeError) as exc:
            # 画面プロセスの終了か、ページの切断かを区別して残す。
            raise AssertionError(
                f"{phase}を待つ間にページとの接続が切れた: exit={app.process.poll()}, "
                f"page_target={app._target('page', 'index.html') is not None}, queue={_queue_statuses(state)}"
            ) from exc

    def expected_output(source: Path) -> Path:
        # 画面が保存物として示す名前（凍結Designの `name - Instrumental.flac`）。
        return source.parent / "LSS" / f"{source.stem} - Instrumental.flac"

    def verify_output(source: Path) -> str:
        saved = expected_output(source)
        assert saved.is_file(), f"画面が示す保存物がありません: {saved}"
        info = sf.info(saved)
        _, rate = infer.expected_output_layout(config, sf.info(source).samplerate)
        assert info.frames == math.ceil(sf.info(source).frames * rate / sf.info(source).samplerate)
        assert info.samplerate == rate and info.channels == 2
        for block in sf.blocks(saved, blocksize=262144, always_2d=True):
            assert np.isfinite(block).all()
        tags = FLAC(saved)
        assert tags["LSS_INPUT_SHA256"] == [digests[source]]
        assert tags["LSS_MODEL_SHA256"] == [default["sha256"]]
        assert list((source.parent / "LSS").glob(f"{source.stem}*")) == [saved], "伴奏以外の成果物がある"
        return session.input_sha256_of(saved)

    def ledger_rows(source: Path) -> list[dict]:
        path = session.completion_ledger_path(state)
        if not path.exists():
            return []
        _, rows = tsv_io.load_tsv(path)
        return [row for row in rows if row.get("source_path") == str(source)]

    began = time.monotonic()
    try:
        # --- 起動1回目: ファイル・フォルダ追加、処理、安全停止 ---
        page = app.launch()
        assert page.state()["phase"] == "ready" and page.state()["queue"] == []
        page.screenshot(evidence / "01-empty.png")
        app.choose_with_key(page, inbox, [full.name, retry_source.name])
        wait_for(lambda: len(page.state()["queue"]) == 2, 20, "ファイル2件の追加")
        app.choose_with_key(page, folder, None)
        view = wait_for(lambda: (lambda v: v if len(v["queue"]) == 4 else None)(page.state()), 20,
                        "フォルダの2件の追加")
        assert [view["songs"][ident]["path"] for ident in view["queue"]] == [str(path) for path in sources]
        assert all(view["st"][ident] == "queued" for ident in view["queue"])
        time.sleep(1.5)
        page.screenshot(evidence / "02-ready-4.png")
        note("added", files=2, folder=2)

        page.key(**ENTER)
        halted = wait_for(lambda: (lambda v: v if v["fault"] else None)(page.state()), 180,
                          "初回のモデル取得失敗でのHALT")
        assert halted["fault"] == "model_download_failed", halted["fault"]
        assert not (inbox / "LSS").exists()
        time.sleep(1.0)
        page.screenshot(evidence / "03a-halt-fetch-failed.png")
        note("first_run_fetch_failed_halt")
        # 接続が回復した代わりに、検証済みの既定モデルを置く。再試行で同じ音源から続ける。
        _link_model(models, checkpoint, config)
        page.click_text("RETRY", "再試行")
        separating = wait_for(lambda: (lambda v: v if v["phase"] == "processing" and v["ph"] == 1 else None)(
            page.state()), 120, "実曲AのSEPARATE")
        assert separating["queue"][separating["ci"]] == row_of(separating, full)
        page.screenshot(evidence / "03-processing-separate.png")
        cadence = page.frame_intervals(240)
        note("separate_cadence", **cadence)
        record["cadence_during_gpu_separation"] = cadence
        page.key(**ESCAPE)
        assert page.state()["stopReq"] is True
        page.screenshot(evidence / "04-stopping.png")
        stopped = settled(page, "ready", 600)
        assert stopped["stopped"] is True
        a_id = row_of(stopped, full)
        assert stopped["st"][a_id] == "done"
        assert all(stopped["st"][row_of(stopped, path)] == "queued" for path in [retry_source, *batch])
        assert not (batch[0].parent / "LSS").exists(), "停止後に次の曲を始めた"
        full_output = verify_output(full)
        page.screenshot(evidence / "05-stopped.png")
        note("stopped_after_current")
        assert app.quit_from_tray_menu() == 0
        wait_for(lambda: not client.is_worker_running(), 60, "終了時の常駐ワーカー停止")
        note("quit_from_tray")

        # --- 起動2回目: 再起動後の続行、失敗曲だけの再試行 ---
        hidden = retry_source.with_name(retry_source.name + ".一時退避")
        retry_source.rename(hidden)
        page = app.launch()
        view = page.state()
        assert [view["songs"][ident]["path"] for ident in view["queue"]][-3:] == [
            str(retry_source), *map(str, batch)]
        page.key(**ENTER)

        def separating_row(source: Path):
            def check():
                current = page.state()
                if current["phase"] == "processing" and not current["fault"] and current["ph"] == 1                         and current["queue"][current["ci"]] == row_of(current, source):
                    return current
                return None
            return check

        # 分離中にワーカーが落ちる: 未完成物を公開せず、同じ音源をやり直す。
        # 接続し直しで自動に続くか、ENGINE STOPPEDで止まってRETRYで続くか、どちらかで着地する。
        wait_for(separating_row(batch[0]), 180, "実曲BのSEPARATE")
        workers = app.worker_pids()
        assert len(workers) == 1, workers
        os.kill(workers[0], 9)

        def recovered_or_halted():
            current = page.state()
            if current["fault"]:
                return ("halted", current)
            if current["st"].get(row_of(current, batch[0])) == "failed":
                return ("row_failed", current)
            pids = app.worker_pids()
            if pids and pids != workers and separating_row(batch[0])():
                return ("reconnected", current)
            return None

        outcome, view = wait_for(recovered_or_halted, 90, "ワーカー停止後の着地")
        assert not expected_output(batch[0]).exists()
        time.sleep(1.0)
        page.screenshot(evidence / "05a-after-worker-kill.png")
        if outcome == "halted":
            assert view["fault"] == "worker_unavailable", view["fault"]
            page.click_text("RETRY", "再試行")
        assert outcome != "row_failed", "ワーカー停止を音源の失敗にした"
        record["worker_kill_outcome"] = outcome
        note("worker_killed_during_separation", outcome=outcome)

        # 分離中に画面のプロセスが落ちる: 起動し直すと未完了の曲から続き、完成物は壊れない。
        wait_for(separating_row(batch[1]), 300, "実曲CのSEPARATE")
        app.kill()
        note("screen_killed_during_separation")
        wait_for(lambda: not app.worker_pids() or expected_output(batch[1]).exists(), 300,
                 "残ったワーカーの仕上げ")
        page = app.launch()
        view = page.state()
        assert view["phase"] == "ready"
        assert all(view["songs"][ident]["path"] != str(batch[0]) for ident in view["queue"])
        assert view["st"][row_of(view, batch[1])] == "queued"
        assert view["st"][row_of(view, retry_source)] == "failed"
        b_completions = ledger_rows(batch[0])
        assert len(b_completions) == 1, b_completions
        b_completion = b_completions[0]
        assert b_completion["outcome"] == "artifacts" and b_completion["input_sha256"] == digests[batch[0]]
        b_output_sha256 = verify_output(batch[0])
        assert [(Path(item["path"]), item["sha256"]) for item in json.loads(b_completion["outputs"])] == [
            (expected_output(batch[0]), b_output_sha256),
        ]
        page.key(**ENTER)
        complete = settled(page, "complete", 600)
        states = {path.name: complete["st"][row_of(complete, path)] for path in (batch[1], retry_source)}
        # 画面が落ちた間に残ったワーカーが仕上げたCは、再開時に再利用（skipped）として着地しうる。
        assert states[batch[1].name] in {"done", "skipped"} and states[retry_source.name] == "failed", states
        record["after_screen_crash_row_states"] = states
        batch_outputs = {path: verify_output(path) for path in batch}
        assert batch_outputs[batch[0]] == b_output_sha256
        assert ledger_rows(batch[0]) == b_completions
        assert not (retry_source.parent / "LSS" / f"{retry_source.stem} - Instrumental.flac").exists()
        time.sleep(1.5)
        page.screenshot(evidence / "06-complete-with-failure.png")
        saved_folder = expected_output(batch[1]).parent
        displayed_folder = page.eval(
            "(() => { const v = window.__lssComponent.renderVals(); return v.resPathRoot + v.resPathLeaf; })()")
        assert displayed_folder == str(saved_folder), displayed_folder
        saved_folder_checks = ["displayed_saved_folder"]
        # Explorer のウィンドウ検出は、試験を実行するデスクトップを列挙する。
        if not app.hidden_desktop:
            app.open_saved_folder_from_tray_menu(saved_folder)
            note("opened_saved_folder_from_tray")
            saved_folder_checks.append("open_saved_folder_from_tray")
        note("resumed_after_restart", failed=[retry_source.name], saved=[path.name for path in batch])

        page.key(**ENTER)
        ready = settled(page, "ready", 30)
        failed_id = row_of(ready, retry_source)
        assert ready["queue"] == [failed_id], "BACK TO READYで失敗曲だけが残る"
        assert ready["hist"] == []
        hidden.rename(retry_source)
        page.key(**DOWN)
        wait_for(lambda: page.state()["selId"] == failed_id, 10, "失敗曲の選択")
        time.sleep(0.6)
        page.screenshot(evidence / "07-failed-row-open.png")
        page.click_text("RETRY", "再試行")
        wait_for(lambda: page.state()["st"].get(failed_id) == "queued", 20, "行の再試行")
        page.key(**ENTER)
        complete = settled(page, "complete", 600)
        assert complete["st"][failed_id] == "done"
        retry_output = verify_output(retry_source)
        time.sleep(1.5)
        page.screenshot(evidence / "08-complete-after-retry.png")
        note("retried_failed_row_only")
        page.key(**ENTER)
        settled(page, "ready", 30)

        # --- 同じ入力の再追加は再利用、保存物を消した入力は作り直す ---
        before_regeneration = ledger_rows(batch[0])
        expected_output(batch[0]).unlink()
        app.choose_with_key(page, inbox, [full.name])
        wait_for(lambda: len(page.state()["queue"]) == 1, 20, "再追加")
        app.choose_with_key(page, folder, None)
        view = wait_for(lambda: (lambda v: v if len(v["queue"]) == 3 else None)(page.state()), 20,
                        "フォルダの再追加")
        reuse_started = time.monotonic()
        page.key(**ENTER)
        complete = settled(page, "complete", 600)
        # 保存済みの再利用は、Designの語彙では保存済み（done）かスキップ（skipped）として着地する。
        reuse_states = {path.name: complete["st"][row_of(complete, path)] for path in [full, *batch]}
        assert reuse_states[batch[0].name] == "done", reuse_states
        assert {reuse_states[full.name], reuse_states[batch[1].name]} <= {"done", "skipped"}, reuse_states
        record["reuse_row_states"] = reuse_states
        assert verify_output(full) == full_output, "再利用で保存物が書き換わった"
        assert verify_output(batch[1]) == batch_outputs[batch[1]]
        verify_output(batch[0])
        record["reuse_session_seconds"] = round(time.monotonic() - reuse_started, 1)
        # 台帳: 実曲Aの記録は1件のまま（再追加で作り直さない）、消した曲だけ記録が更新される。
        assert len([row for row in ledger_rows(full) if row.get("outcome") == "artifacts"]) == 1
        assert ledger_rows(batch[0]) != before_regeneration, "消した保存物を作り直した記録が無い"
        assert verify_output(retry_source) == retry_output
        time.sleep(1.5)
        page.screenshot(evidence / "09-complete-reuse.png")
        note("reuse_and_regeneration")
        page.key(**ENTER)
        settled(page, "ready", 30)
        assert app.quit_from_tray_menu() == 0
        wait_for(lambda: not client.is_worker_running(), 60, "終了時の常駐ワーカー停止")

        for source in sources:
            assert session.input_sha256_of(source) == digests[source], f"入力が変わった: {source}"
        assert {path: path.stat().st_mtime_ns for path in originals} == originals_before
        assert checkpoint.stat().st_nlink >= 2  # 取得済みモデルは複製せず参照した
        record["total_seconds"] = round(time.monotonic() - began, 1)
        record["verified"] = [
            "deployed_screen", "no_dev_python_on_path", "first_run_fetch_failure_halt_retry", "file_dialog_multi_select", "folder_dialog", "real_gpu_songs",
            "stages", "cadence_during_gpu", "safe_stop_after_current", "tray_quit_stops_worker",
            "restart_resume_persisted_queue", "worker_crash_recovery", "screen_crash_resume", "failed_row_only_retry", "complete_and_back_to_ready",
            "displayed_output_exists_and_valid", *saved_folder_checks, "readd_reuse_without_rewrite",
            "deleted_output_regenerated", "sources_immutable",
        ]
    finally:
        (evidence / "record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        app.kill()
        if client.is_worker_running():
            client.request_worker_shutdown()


def test_deployed_screen_separates_disk_full_unsupported_and_zero_results(tmp_path, request):
    """容量不足のHALTとSKIP・再起動後の行の再試行、分離できない入力、検証済みの空出力。"""
    from tests.deployed_ui_driver import DeployedApp, wait_for

    repository = Path(__file__).resolve().parents[1]
    evidence = _evidence(tmp_path, "edges")
    deployed, executable = _portable(tmp_path, request)
    default = catalog.default_model()
    checkpoint, config = fetch.model_paths(
        default["model_id"], default["distributor_url"], default["config_url"])
    assert checkpoint.is_file() and config.is_file(), "取得済みの既定モデルが必要です"
    _, corpus = tsv_io.load_tsv(dev_evidence_root(repository) / "ledger" / "test_set.tsv")

    inbox = tmp_path / "入力"
    inbox.mkdir()
    song = inbox / ("実曲E" + Path(corpus[4]["mix_path"]).suffix)
    _clip(Path(corpus[4]["mix_path"]), song, 20)
    rng = np.random.default_rng(7)
    six = inbox / "六チャンネル.wav"
    sf.write(six, rng.normal(0, 0.1, (44100 * 5, 6)), 44100, subtype="PCM_24")
    silent = inbox / "無音.wav"
    sf.write(silent, np.zeros((44100 * 5, 2)), 44100, subtype="PCM_16")
    broken = inbox / "非有限.wav"
    sf.write(broken, np.full((44100 * 5, 2), np.nan), 44100, subtype="FLOAT")
    inputs = [song, six, silent, broken]
    digests = {path: session.input_sha256_of(path) for path in inputs}

    state = tmp_path / "state"
    models = tmp_path / "models"
    _link_model(models, config)  # 重みだけ未取得。取得に要る容量をキャッシュ上限で不足させる。
    app = DeployedApp(executable, _offline_environment(state, models, LSS_MODEL_CACHE_CAP_BYTES="1"))
    record: dict = {"executable": str(deployed), "executable_sha256": session.input_sha256_of(deployed)}
    output = song.parent / "LSS" / f"{song.stem} - Instrumental.flac"

    def phase_is(page, phase: str, timeout: float):
        return wait_for(lambda: (lambda v: v if v and v["phase"] == phase else None)(page.state()),
                        timeout, f"{phase}への遷移")

    def row_of(view: dict, source: Path) -> str:
        return next(ident for ident in view["queue"] if view["songs"][ident].get("path") == str(source))

    try:
        page = app.launch()
        app.choose_with_key(page, inbox, [song.name])
        wait_for(lambda: len(page.state()["queue"]) == 1, 20, "追加")
        page.key(**ENTER)
        halted = wait_for(lambda: (lambda v: v if v["fault"] else None)(page.state()), 180, "DISK FULL")
        assert halted["fault"] == "insufficient_output_space", halted["fault"]
        time.sleep(1.0)
        page.screenshot(evidence / "01-halt-disk-full.png")
        page.click_text("SKIP", "この音源を飛ばす")
        complete = phase_is(page, "complete", 60)
        song_id = row_of(complete, song)
        assert complete["st"][song_id] == "failed"
        page.key(**ENTER)
        phase_is(page, "ready", 30)
        assert app.quit_from_tray_menu() == 0
        wait_for(lambda: not client.is_worker_running(), 60, "終了時の常駐ワーカー停止")
        assert not output.exists()

        # 空きが戻った想定: 上限を外し、取得済みの重みを置いて起動し直す。
        _link_model(models, checkpoint)
        app.environment = _offline_environment(state, models)
        page = app.launch()
        view = page.state()
        assert view["st"][song_id] == "failed"
        page.key(**DOWN)
        wait_for(lambda: page.state()["selId"] == song_id, 10, "失敗行の選択")
        time.sleep(0.6)
        page.screenshot(evidence / "02-skipped-halt-row-after-restart.png")
        note = page.eval(
            "(() => { const c = window.__lssComponent; const r = c.renderVals().rows.find(x => x.key === "
            + json.dumps(song_id) + "); return [r.note, r.actionEn]; })()")
        assert note == ["保存先の空き容量が足りません。", "RETRY"], note
        page.click_text("RETRY", "再試行")
        wait_for(lambda: page.state()["st"].get(song_id) == "queued", 20, "行の再試行")
        app.choose_with_key(page, inbox, [six.name, silent.name, broken.name])
        wait_for(lambda: len(page.state()["queue"]) == 4, 20, "3件の追加")
        page.key(**ENTER)
        complete = phase_is(page, "complete", 600)
        states = {path.name: complete["st"][row_of(complete, path)] for path in inputs}
        record["row_states"] = states
        assert states == {song.name: "done", six.name: "failed", silent.name: "done", broken.name: "failed"}, states
        silent_id = row_of(complete, silent)
        silent_result = page.eval("window.__lssComponent.state.tst[" + json.dumps(silent_id) + "].inst")
        assert (silent_result["k"], silent_result["n"]) == ("none", 0), silent_result
        assert not complete["songs"][silent_id].get("out")
        saved_count = page.eval(
            "Number(window.__lssComponent.renderVals().resTargets.find(target => target.key === 'inst').num)")
        assert saved_count == 1, saved_count
        rows = page.eval(
            "(() => { const c = window.__lssComponent; const s = c.state; return Object.fromEntries("
            "c.renderVals().rows.map(r => [c.songs()[r.key].path, {badge: r.badge, note: r.note, hasNote: r.hasNote}])); })()")
        record["rows"] = rows
        assert rows[str(six)]["badge"] == "UNSUPPORTED" and rows[str(six)]["note"].startswith("6ch")
        assert rows[str(broken)]["badge"] == "UNSUPPORTED"
        time.sleep(1.5)
        page.screenshot(evidence / "03-complete-edges.png")
        assert output.is_file() and sf.info(output).channels == 2
        assert FLAC(output)["LSS_INPUT_SHA256"] == [digests[song]]
        assert sorted(item.name for item in (inbox / "LSS").iterdir()) == [output.name], "失敗・空出力で成果物を作った"
        page.key(**ENTER)
        phase_is(page, "ready", 30)
        assert app.quit_from_tray_menu() == 0
        wait_for(lambda: not client.is_worker_running(), 60, "終了時の常駐ワーカー停止")
        for path in inputs:
            assert session.input_sha256_of(path) == digests[path]
        record["verified"] = [
            "disk_full_halt", "halt_skip", "skipped_halt_row_retry_after_restart", "unsupported_channels",
            "nonfinite_input", "exact_silence_zero_result", "zero_result_excluded_from_saved_count",
            "no_artifacts_for_failures_or_zero_results",
        ]
    finally:
        (evidence / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        app.kill()
        if client.is_worker_running():
            client.request_worker_shutdown()


@pytest.mark.skipif(os.environ.get("LSS_UI_ACCEPTANCE_DOWNLOAD") != "1",
                    reason="既定モデルを実際に取得する初回受入は、通信と容量を使うので別に明示実行")
def test_deployed_screen_first_run_downloads_the_default_model(tmp_path, request):
    """モデルが何も無い状態から、画面の操作だけで既定モデルを取得して1曲を処理する。"""
    from tests.deployed_ui_driver import DeployedApp, wait_for

    repository = Path(__file__).resolve().parents[1]
    evidence = _evidence(tmp_path, "first-run-download")
    deployed, executable = _portable(tmp_path, request)
    default = catalog.default_model()
    _, corpus = tsv_io.load_tsv(dev_evidence_root(repository) / "ledger" / "test_set.tsv")
    inbox = tmp_path / "入力"
    inbox.mkdir()
    song = inbox / ("実曲F" + Path(corpus[5]["mix_path"]).suffix)
    _clip(Path(corpus[5]["mix_path"]), song, 20)
    state = tmp_path / "state"
    models = tmp_path / "models"
    environment = _offline_environment(state, models)
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"):
        environment[key] = os.environ.get(key, "")
    app = DeployedApp(executable, environment)
    record: dict = {"executable": str(deployed), "executable_sha256": session.input_sha256_of(deployed)}
    try:
        page = app.launch()
        app.choose_with_key(page, inbox, [song.name])
        wait_for(lambda: len(page.state()["queue"]) == 1, 60, "追加")
        began = time.monotonic()
        page.key(**ENTER)
        preparing = wait_for(lambda: page.eval(
            "(() => { const s = window.__lssComponent.state; return s.phase === 'processing' && s.ph === 0"
            " && s.mcached === false ? s : null; })()"), 120, "初回取得のPREPARE")
        assert preparing["fault"] is None
        time.sleep(3)
        page.screenshot(evidence / "01-first-run-prepare.png")
        fractions = []

        def separating():
            view = page.eval("(() => { const s = window.__lssComponent.state; return [s.ph, s.phP, s.fault]; })()")
            assert view[2] is None, view
            if view[0] == 0:
                fractions.append(view[1])
            return view[0] >= 1

        wait_for(separating, 1800, "取得後のSEPARATE", interval=1.0)
        record["download_seconds"] = round(time.monotonic() - began, 1)
        record["prepare_fraction_samples"] = sorted(set(round(value, 2) for value in fractions))[-10:]
        wait_for(lambda: page.state()["phase"] == "complete", 600, "COMPLETE", interval=0.5)
        time.sleep(1.5)
        page.screenshot(evidence / "02-first-run-complete.png")
        checkpoint, config = fetch.model_paths(
            default["model_id"], default["distributor_url"], default["config_url"])
        downloaded = models / checkpoint.parent.name / checkpoint.name
        assert fetch.file_sha256(downloaded) == default["sha256"]
        output = song.parent / "LSS" / f"{song.stem} - Instrumental.flac"
        assert output.is_file() and FLAC(output)["LSS_MODEL_SHA256"] == [default["sha256"]]
        page.key(**ENTER)
        wait_for(lambda: page.state()["phase"] == "ready", 30, "READY")
        assert app.quit_from_tray_menu() == 0
        wait_for(lambda: not client.is_worker_running(), 60, "終了時の常駐ワーカー停止")
        record["verified"] = ["first_run_model_download_from_screen", "downloaded_model_sha256", "processed_after_download"]
    finally:
        (evidence / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        app.kill()
        if client.is_worker_running():
            client.request_worker_shutdown()
