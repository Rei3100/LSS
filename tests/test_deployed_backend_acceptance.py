"""配備された実行物を隔離して、実曲の処理・再起動・回復を確認する。

LSS_RUN_DEPLOYED_ACCEPTANCE=1で明示実行。UIや音楽品質の採用評価ではない。
実行物と同梱リソースは読み取り、可変台帳・入力コピー・成果は一時領域へ置く。
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
import subprocess
import time
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.assets import backup
from lss.core import catalog, client, protocol, recipe, scratch, session
from lss.harness import fetch, infer, tsv_io


pytestmark = pytest.mark.skipif(
    os.environ.get("LSS_RUN_DEPLOYED_ACCEPTANCE") != "1",
    reason="配備された実行物と実GPU・実曲を使う受入は明示実行",
)


def test_deployed_song_restart_reuse_failure_and_regeneration(tmp_path, request):
    import _winapi

    repository = Path(__file__).resolve().parents[1]
    deployed = Path(os.environ["LSS_ACCEPTANCE_EXECUTABLE"])
    assert deployed.is_file(), f"配備された実行物が必要です: {deployed}"
    resources = deployed.parent / "_internal"
    assert resources.is_dir(), f"onedirの同梱リソースが必要です: {resources}"
    executable_sha256 = session.input_sha256_of(deployed)
    isolated = tmp_path / "portable"
    isolated.mkdir()
    executable = isolated / "LSS.exe"
    shutil.copyfile(deployed, executable)
    # 大容量のCUDA資産は複製しない。junctionを外す後処理はリンクだけを対象にする。
    junction = isolated / "_internal"
    _winapi.CreateJunction(str(resources), str(junction))
    request.addfinalizer(junction.rmdir)

    default = catalog.default_model()
    checkpoint, config = fetch.model_paths(
        default["model_id"], default["distributor_url"], default["config_url"],
    )
    assert checkpoint.is_file() and config.is_file(), "取得済みの既定モデルが必要です"
    configured_source = os.environ.get("LSS_BACKEND_ACCEPTANCE_SOURCE")
    if configured_source:
        original = Path(configured_source)
    else:
        _, corpus = tsv_io.load_tsv(dev_evidence_root(repository) / "ledger" / "test_set.tsv")
        original = Path(corpus[0]["mix_path"])
    assert sf.info(original).channels == 2
    source_digest = session.input_sha256_of(original)
    source = tmp_path / ("実曲" + original.suffix)
    shutil.copyfile(original, source)
    state = tmp_path / "state"
    environment = {
        **os.environ, "LSS_MODEL_DIR": str(fetch.model_cache_dir()),
        "LSS_STATE_DIR": str(state), "LSS_MIN_FREE_BYTES": "0",
    }
    address = fr"\\.\pipe\lss-deployed-acceptance-{uuid4().hex}"
    process = None

    def start():
        nonlocal process
        process = subprocess.Popen(
            [str(executable), "--worker", "--address", address],
            cwd=isolated, env=environment, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if client.is_worker_running(address):
                return
            assert process.poll() is None, f"配備版の起動失敗: {process.returncode}"
            time.sleep(0.1)
        pytest.fail("配備版ワーカーが時間内に起動しません")

    def stop():
        nonlocal process
        response = client.request_worker_shutdown(address)
        assert response is not None and response["status"] == "ok", response
        process.wait(timeout=15)
        assert process.returncode == 0
        assert not client.is_worker_running(address)
        process = None

    def dispatch(path, events, *, terminate_at_phase=None):
        # 起動済みの配備版へ直接送る。開発用Pythonワーカーへの自動再起動はしない。
        from multiprocessing import connection

        conn = connection.Client(address, authkey=protocol.AUTHKEY)
        try:
            conn.send({
                "mix_path": str(path), "model_id": default["model_id"],
                "stem_target": "Instrumental", "_stream_progress": True,
            })
            deadline = time.monotonic() + 300
            while True:
                assert conn.poll(max(0, deadline - time.monotonic())), "配備版処理の時間超過"
                response = conn.recv()
                if response.get("type") == "progress":
                    events.append(response)
                    if response["phase"] == terminate_at_phase:
                        # テストが起動した隔離ワーカーだけを強制終了する。
                        process.kill()
                        process.wait(timeout=15)
                        return {"status": "test_forced_termination"}
                else:
                    return response
        finally:
            conn.close()

    def verify(result):
        assert result["status"] in {"ok", "skipped"}, result
        assert len(result["outputs"]) == 1
        saved = Path(result["outputs"][0])
        assert saved.parent == source.parent / "LSS"
        assert "Instrumental" in saved.stem
        info = sf.info(saved)
        _, rate = infer.expected_output_layout(config, sf.info(source).samplerate)
        assert info.frames == math.ceil(sf.info(source).frames * rate / sf.info(source).samplerate)
        assert info.samplerate == rate and info.channels == 2
        for block in sf.blocks(saved, blocksize=65536, always_2d=True):
            assert np.isfinite(block).all()
        tags = FLAC(saved)
        assert tags["LSS_INPUT_SHA256"] == [source_digest]
        assert tags["LSS_MODEL_SHA256"] == [default["sha256"]]
        assert tags["LSS_CONFIG_SHA256"] == [fetch.file_sha256(config)]
        assert tags["LSS_RUNTIME_ID"] == [result["runtime_id"]]
        assert recipe.validate_recipe_revision(result["recipe_revision"])["publication"]["stems"] == ["Instrumental"]
        assert session.is_completed(source_digest, result["recipe_id"], source, state_root=state)
        return saved

    try:
        start()
        crashed = dispatch(source, [], terminate_at_phase="separating")
        assert crashed["status"] == "test_forced_termination"
        assert not list((source.parent / "LSS").glob("*.flac"))
        assert not session.completion_ledger_path(state).exists()
        abandoned = list((state / "work").iterdir())
        assert abandoned, "推論前の強制終了で作業物が残る条件を実際に通す"
        # 回収の権限は名前・更新時刻ではなく、payloadと別に置いた所有記録（lss.core.scratch）だけが持つ。
        owners = scratch._registry(state / "work")
        receipts = sorted(owners.glob("*.json"))
        records = [json.loads(path.read_text(encoding="utf-8")) for path in receipts]
        assert {record["directory"] for record in records} == {path.name for path in abandoned}
        assert all(record["released_at"] is None for record in records)
        # 更新時刻だけが古くても、所有記録の猶予内なら起動時に回収しない。
        expired = time.time() - scratch.ORPHAN_GRACE_SECONDS - 60
        for path in abandoned:
            os.utime(path, (expired, expired))
        start()
        assert all(path.exists() for path in abandoned)
        stop()
        # 所有記録の猶予後は、使用中でないことを確かめて起動時に回収する。
        for path, record in zip(receipts, records):
            path.write_text(json.dumps({**record, "created_at": expired}), encoding="utf-8")
        start()
        assert all(not path.exists() for path in abandoned)
        assert not list(owners.iterdir())
        events = []
        result = dispatch(source, events)
        saved = verify(result)
        assert result["status"] == "ok"
        assert {event["phase"] for event in events} >= {"separating", "completed"}
        saved_digest = session.input_sha256_of(saved)
        stop()
        start()
        # 完了を記録した後の起動では、保存状態の自動バックアップが配備先に作られる。
        automatic = list((isolated / "Backups").glob("lss-state-auto-*.zip"))
        assert automatic, "配備版の起動で保存状態の自動バックアップが作られていません"
        assert "state/completed_recipes.tsv" in {
            item["name"] for item in backup.verify_backup(automatic[-1])["files"]}
        reuse_events = []
        reused = dispatch(source, reuse_events)
        assert reused["status"] == "skipped", reused
        assert verify(reused) == saved
        assert session.input_sha256_of(saved) == saved_digest
        assert "separating" not in {event["phase"] for event in reuse_events}
        missing = tmp_path / "存在しない.wav"
        failed = dispatch(missing, [])
        assert failed["status"] == "error" and failed["error_code"] == "file_not_found"
        assert client.is_worker_running(address)
        silent = tmp_path / "厳密な無音.wav"
        sf.write(silent, np.zeros((44100, 2)), 44100, subtype="FLOAT")
        absent = dispatch(silent, [])
        assert absent["status"] == "ok" and absent["outputs"] == [], absent
        assert absent["zero_result"]["validation_result"]["validator_id"] == "exact_digital_silence"
        assert session.completed_zero_result(
            session.input_sha256_of(silent), absent["recipe_id"], silent, state_root=state,
        ) == absent["zero_result"]
        assert dispatch(silent, [])["status"] == "skipped"
        broken = tmp_path / "非有限入力.wav"
        sf.write(broken, np.full((44100, 2), np.nan), 44100, subtype="FLOAT")
        invalid = dispatch(broken, [])
        assert invalid["status"] == "error" and invalid["error_code"] == "invalid_input_signal", invalid
        saved.unlink()
        # ファイル公開後・完了台帳記録中の異常終了を通す。返答は受け取らない。
        crashed = dispatch(source, [], terminate_at_phase="recording_completion")
        assert crashed["status"] == "test_forced_termination"
        assert saved.is_file()
        published_digest = session.input_sha256_of(saved)
        start()
        recovered_events = []
        recovered = dispatch(source, recovered_events)
        assert recovered["status"] == "skipped", recovered
        verify(recovered)
        assert session.input_sha256_of(saved) == published_digest
        assert "separating" not in {event["phase"] for event in recovered_events}
        assert session.input_sha256_of(source) == source_digest
        assert session.input_sha256_of(original) == source_digest
        assert not list((state / "publications").glob("*.json"))
        # 期限前の強制終了残骸は保留する契約なので、最後の再生成の残骸は許容する。
        print(json.dumps({
            "executable": str(deployed), "executable_sha256": executable_sha256,
            "source_sha256": source_digest, "source_seconds": sf.info(source).duration,
            "runtime_id": result["runtime_id"], "recipe_id": result["recipe_id"],
            "verified": ["packaged_real_gpu", "progress", "restart", "reuse", "failure", "exact_silence", "nonfinite_input_rejection", "deleted_output_regeneration", "forced_stop_before_inference", "expired_work_cleanup", "forced_stop_after_publication", "publication_reconciliation_without_inference"],
        }, ensure_ascii=False))
        stop()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
