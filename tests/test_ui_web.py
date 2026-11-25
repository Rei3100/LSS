"""WebView2で出す凍結Designページと、そこへ渡すバックエンドの事実。

画面の見た目と動きはDesign自身のコードなので、ここで守るのは
「Designを書き換えずに、デモだけを実データへ差し替えている」ことと、
「ページが自分では知りえない事実を正しく渡している」こと。
実ページでの通し操作は WebView2 + DevTools Protocol で別途確認する（LV-UI-001）。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
import json

import pytest

from lss.ui.web import page
from lss.ui.web.session import WebSession


# --- ページの組み立て --------------------------------------------------------


def test_every_patch_matches_the_frozen_design_exactly_once() -> None:
    source = page.CANVAS.read_text(encoding="utf-8")
    for patch in page.PATCHES:
        assert source.count(patch.old) == 1, patch.name


def test_built_page_needs_no_network_and_carries_the_bridge(tmp_path: Path) -> None:
    entry = page.build(tmp_path, initial={"queue": [], "note": "</script>"})
    html = entry.read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in html
    assert "Component = class LSSComponent extends Component" in html
    assert '"debugTools":{"editor":"boolean","default":false' in html.replace("&quot;", '"')
    # 初期データはスクリプトを閉じられない
    assert "</script>\"" not in html and "\\u003c/script>" in html
    scripts = ["react.production.min.js", "react-dom.production.min.js", "runtime.js"]
    positions = [html.index(f'<script src="./{name}"></script>') for name in scripts]
    assert positions == sorted(positions)
    assert (tmp_path / "runtime.js").read_bytes() == page.RUNTIME.read_bytes()
    assert "support.js" not in html
    assert not (tmp_path / "support.js").exists()


def test_built_pages_resolve_declared_assets_inside_the_cache(tmp_path: Path) -> None:
    import re
    from html.parser import HTMLParser
    from urllib.parse import unquote, urlsplit

    class AssetReferences(HTMLParser):
        def handle_starttag(self, tag, attrs):
            assert tag != "base"
            urls.extend(value for name, value in attrs if name in {"src", "href"} and value)

    entry = page.build(tmp_path)
    for path in (entry, tmp_path / "tray_menu.html"):
        html = path.read_text(encoding="utf-8")
        urls: list[str] = []
        AssetReferences().feed(html)
        for css in re.findall(r"<style>(.*?)</style>", html, re.DOTALL):
            urls.extend(re.findall(r"url\(['\"]?([^'\")]+)['\"]?\)", css))
        assert urls
        for url in urls:
            parts = urlsplit(url)
            assert not parts.scheme and not parts.netloc, url
            asset = (path.parent / unquote(parts.path)).resolve()
            assert asset.is_relative_to(tmp_path.resolve()) and asset.is_file(), url


def test_page_rebuilds_over_a_read_only_cache_from_a_read_only_package(
        tmp_path: Path, monkeypatch) -> None:
    """配備された同梱物は読み取り専用。2回目の起動でランタイムの上書きが
    PermissionError になり、アプリが起動しなかった（c92defa の配備後）。"""
    import os
    import shutil
    import stat

    package = tmp_path / "package"
    package.mkdir()
    runtime = package / "runtime.js"
    shutil.copyfile(page.RUNTIME, runtime)
    os.chmod(runtime, stat.S_IREAD)
    monkeypatch.setattr(page, "RUNTIME", runtime)
    web = tmp_path / "state" / "web"
    # 以前の版が残した、属性ごと写した読み取り専用のキャッシュ
    web.mkdir(parents=True)
    for name in ("index.html", "runtime.js", "react.production.min.js", "react-dom.production.min.js"):
        shutil.copy2(runtime, web / name)
        os.chmod(web / name, stat.S_IREAD)
    # LSSが書く名前以外の物（利用者のメモ。読み取り専用も）は、起動のたびのページの書き出しで消さない
    notes = {web / "my-notes.txt": b"keep me", web / "fonts" / "readonly-note.txt": b"keep me too"}
    (web / "fonts").mkdir()
    for path, content in notes.items():
        path.write_bytes(content)
    os.chmod(web / "fonts" / "readonly-note.txt", stat.S_IREAD)
    try:
        for _launch in range(2):
            entry = page.build(web, initial={"queue": []})
            assert entry.is_file()
            generated = [path for path in web.rglob("*") if path.is_file() and path not in notes]
            for path in generated:
                assert os.access(path, os.W_OK), path
            assert (web / "runtime.js").read_bytes() == runtime.read_bytes()
            assert (web / "tray_menu.html").is_file()
            assert {path: path.read_bytes() for path in notes} == notes
            assert not os.access(web / "fonts" / "readonly-note.txt", os.W_OK)
    finally:
        os.chmod(runtime, stat.S_IWRITE | stat.S_IREAD)
        os.chmod(web / "fonts" / "readonly-note.txt", stat.S_IWRITE | stat.S_IREAD)


def test_vendored_react_matches_the_lss_integrity_hashes() -> None:
    from hashlib import sha384

    for name, expected in page._REACT:
        assert sha384((page.WEB_DIR / name).read_bytes()).hexdigest() == expected, name


def test_a_reexported_design_fails_loudly_instead_of_drifting() -> None:
    with pytest.raises(page.PatchError):
        page.apply("<html><head></head><body></body></html>")


# --- セッションの事実 --------------------------------------------------------


def _audio(path: Path, channels: int = 2) -> Path:
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros((4410, channels), dtype="float32"), 44100, subtype="PCM_16")
    return path


class _Worker:
    """ジョブごとに結果を決める偽の分離。1件ずつ手動で進められる。"""

    def __init__(self, outcomes: dict[str, dict]) -> None:
        self.outcomes = outcomes
        self.gate = threading.Semaphore(0)
        self.started: list[str] = []

    def __call__(self, job: dict) -> dict:
        name = Path(job["mix_path"]).stem
        self.started.append(name)
        self.gate.acquire(timeout=10)
        return self.outcomes.get(name, {"status": "ok", "outputs": [job["mix_path"] + ".flac"]})


@pytest.fixture
def state(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


def _wait(predicate, timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("条件が成立しなかった")


def _session(state: Path, worker) -> WebSession:
    return WebSession(dispatch=worker, state_path=state / "queue.json")


def _source_ids(session: WebSession, paths: list[Path]) -> list[str]:
    """Outcome checks identify sources independently of normalized batch order."""
    songs = session.snapshot()["songs"]
    return [next(item_id for item_id, song in songs.items() if song["path"] == str(path))
            for path in paths]


def test_complete_carries_the_authoritative_product_result_and_persists_it(state: Path) -> None:
    source = _audio(state / "one.wav")
    result = {"schema_version": 1, "state": "PUBLISHED", "report_path": str(state / "結果.html"),
              "published_count": 1, "held_count": 0}
    worker = _Worker({"one": {"status": "ok", "outputs": [str(source) + ".flac"], "product_result": result}})
    session = _session(state, worker)
    [item] = session.add([source])
    session.start()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    assert session.snapshot()["tst"][item]["inst"]["result"] == result
    saved = json.loads((state / "queue.json").read_text(encoding="utf-8"))
    assert saved["items"][0]["outcomes"]["inst"]["product_result"] == result


def test_snapshot_speaks_the_design_vocabulary(state: Path) -> None:
    a = _audio(state / "A" / "one.wav")
    b = _audio(state / "B" / "six.wav", channels=6)
    session = _session(state, _Worker({}))
    added = session.add([a, b])
    snap = session.snapshot()
    assert snap["queue"] == added
    assert snap["arrival"]["ids"] == added
    one = snap["songs"][added[0]]
    assert one["num"] == "01" and one["name"] == "one" and one["fmt"] == "WAV"
    assert one["sr"] == "44.1" and one["bit"] == "16" and one["dir"] == str(a.parent)
    assert snap["songs"][added[1]]["ch"] == 6
    assert snap["st"] == {added[0]: "queued", added[1]: "queued"}
    assert snap["fault"] is None and snap["running"] is None


def test_row_failures_carry_badge_note_and_retryability(state: Path) -> None:
    changed = _audio(state / "changed.wav")
    six = _audio(state / "six.wav")
    worker = _Worker({
        "changed": {"status": "error", "error_code": "input_changed", "message": "x"},
        "six": {"status": "error", "error_code": "unsupported_channels", "message": "x", "channels": 6},
    })
    session = _session(state, worker)
    ids = session.add([changed, six])
    session.start()
    worker.gate.release()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    snap = session.snapshot()
    assert snap["st"] == {ids[0]: "failed", ids[1]: "failed"}
    retry_key = snap["songs"][ids[0]]["fail"]
    hard_key = snap["songs"][ids[1]]["hard"]
    assert snap["errs"][retry_key] == ["FAILED", "処理中に元のファイルが変更されました。", True]
    assert snap["errs"][hard_key][0] == "UNSUPPORTED"
    assert snap["errs"][hard_key][1].startswith("6ch の音源は分離できません")
    assert snap["errs"][hard_key][2] is False


def test_session_failure_halts_before_the_next_source_starts(state: Path) -> None:
    first = _audio(state / "first.wav")
    second = _audio(state / "second.wav")
    worker = _Worker({"first": {"status": "error", "error_code": "insufficient_output_space", "message": "x"}})
    session = _session(state, worker)
    ids = session.add([first, second])
    session.start()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "halted")
    snap = session.snapshot()
    assert worker.started == ["first"], "HALTの後に次の音源が始まった"
    assert snap["fault"]["key"] == "insufficient_output_space"
    assert snap["fault"]["item"] == ids[0]
    assert snap["fault"]["metrics"] == [], "実測していない数値を出してはいけない"
    # HALT中は停止要求を受け付けない
    session.request_stop()
    assert session.snapshot()["fault"] is not None

    session.halt_skip()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    snap = session.snapshot()
    assert snap["fault"] is None
    assert snap["st"] == {ids[0]: "failed", ids[1]: "done"}


def test_halt_retry_reruns_the_same_source(state: Path) -> None:
    only = _audio(state / "only.wav")
    worker = _Worker({"only": {"status": "error", "error_code": "worker_unavailable", "message": "x"}})
    session = _session(state, worker)
    session.add([only])
    session.start()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "halted")
    worker.outcomes.clear()
    session.halt_retry()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    assert worker.started == ["only", "only"]


def test_stop_finishes_the_current_source_and_settles_stopped(state: Path) -> None:
    ids_paths = [_audio(state / f"s{i}.wav") for i in range(3)]
    worker = _Worker({})
    session = _session(state, worker)
    ids = session.add(ids_paths)
    session.start()
    _wait(lambda: session.snapshot()["running"] == ids[0])
    session.request_stop()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "stopped")
    snap = session.snapshot()
    assert worker.started == ["s0"]
    assert snap["st"] == {ids[0]: "done", ids[1]: "queued", ids[2]: "queued"}
    # RESUMEは同じセッションの続き
    session.start()
    worker.gate.release()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")


def test_progress_becomes_stage_and_separate_is_never_a_percentage(state: Path) -> None:
    only = _audio(state / "only.wav")
    worker = _Worker({})
    session = _session(state, worker)
    (item,) = session.add([only])
    session.start()
    _wait(lambda: session.snapshot()["running"] == item)
    session._on_progress(item, {"phase": "hashing_input", "completed_bytes": 1, "total_bytes": 4})
    assert session.snapshot()["stage"] == {"k": "PREP", "p": 0.25}
    session._on_progress(item, {"phase": "separating", "completed_frames": 5, "total_frames": 10})
    assert session.snapshot()["stage"] == {"k": "SEPARATE", "p": None}
    session._on_progress(item, {"phase": "writing_outputs", "completed_bytes": 3, "total_bytes": 4})
    assert session.snapshot()["stage"] == {"k": "SAVE", "p": 0.75}
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    assert session.snapshot()["stage"] is None


def test_back_to_ready_keeps_failures_and_resets_the_clock(state: Path) -> None:
    good = _audio(state / "good.wav")
    bad = _audio(state / "bad.wav")
    worker = _Worker({"bad": {"status": "error", "error_code": "processing_failed", "message": "x"}})
    session = _session(state, worker)
    session.add([good, bad])
    ids = _source_ids(session, [good, bad])
    session.start()
    worker.gate.release()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    assert session.snapshot()["elapsed"] >= 0
    session.back_to_ready()
    snap = session.snapshot()
    assert snap["queue"] == [ids[1]]
    assert snap["elapsed"] == 0


def test_recipe_is_the_model_recorded_by_the_result(state: Path) -> None:
    only = _audio(state / "only.wav")
    worker = _Worker({"only": {
        "status": "ok", "outputs": [str(only) + ".flac"], "recipe_id": "separation-v1-x",
        "recipe_revision": {"stage_graph": [{"model_id": "model_from_result"}]},
    }})
    session = _session(state, worker)
    session.add([only])
    session.start()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    assert session.snapshot()["recipe"] == "model_from_result"




def test_ready_does_not_restore_completed_history(state: Path) -> None:
    """保存済みの記録は一覧（ALREADY SAVED）へ戻さない。前回の終端も一覧から外す。"""
    from lss.core.session import completion_ledger_path
    from lss.harness.tsv_io import save_tsv

    source = _audio(state / "Album" / "old.wav")
    ledger = completion_ledger_path()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    save_tsv(ledger, ["input_sha256", "recipe_id", "completed_at", "source_path", "outcome",
                      "zero_reason", "validation_result", "outputs"], [{
        "input_sha256": "0" * 64, "recipe_id": "r", "completed_at": "2025-11-03T21:50:00",
        "source_path": str(source), "outcome": "artifacts", "zero_reason": "",
        "validation_result": "", "outputs": '[{"path": "x.flac"}]',
    }])
    good = _audio(state / "good.wav")
    bad = _audio(state / "bad.wav")
    worker = _Worker({"bad": {"status": "error", "error_code": "processing_failed", "message": "x"}})
    session = _session(state, worker)
    session.add([good, bad])
    ids = _source_ids(session, [good, bad])
    session.start()
    worker.gate.release()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    assert session.snapshot()["hist"] == []
    # 再起動したREADYには、失敗だけが再試行の対象として残る。
    restored = _session(state, _Worker({}))
    snap = restored.snapshot()
    assert snap["hist"] == [] and snap["queue"] == [ids[1]]
    assert snap["st"] == {ids[1]: "failed"}


def test_a_validated_zero_result_is_not_claimed_as_a_prior_separation(state: Path) -> None:
    silent = _audio(state / "silent.wav")
    reused = _audio(state / "reused.wav")
    zero = {"reason": "music_not_wanted", "validation_result": {
        "validator_id": "bgm-semantic-gate", "validator_revision": "r1",
        "hard_gates_passed": True, "evidence": {"music_frames": 100}}}
    worker = _Worker({
        "silent": {"status": "ok", "outputs": [], "zero_result": zero},
        "reused": {"status": "skipped", "outputs": [str(reused) + ".flac"]},
    })
    session = _session(state, worker)
    session.add([silent, reused])
    ids = _source_ids(session, [silent, reused])
    session.start()
    worker.gate.release()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    snap = session.snapshot()
    assert snap["st"] == {ids[0]: "done", ids[1]: "done"}
    assert snap["tst"][ids[0]] == {"inst": {"k": "none", "n": 0}}
    assert snap["tst"][ids[1]] == {"inst": {"k": "reused", "n": 1}}
    assert "out" not in snap["songs"][ids[0]] and snap["songs"][ids[1]]["out"] is True


# --- 欲しい結果（伴奏・BGM）------------------------------------------------


class _Bgm:
    """単話BGM候補工程の偽物。音源名ごとに結果を決め、呼ばれた順を記録する。"""

    def __init__(self, outcomes: dict[str, dict] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.started: list[str] = []

    def __call__(self, path, dispatch, on_progress) -> dict:
        self.started.append(Path(path).stem)
        return self.outcomes.get(Path(path).stem, {"status": "ok", "bgm": {"k": "held", "n": 3}})


def _targets_session(state: Path, worker, bgm=None, lookup=None, **options) -> WebSession:
    return WebSession(dispatch=worker, state_path=state / "queue.json", bgm=bgm or _Bgm(),
                      lookup=lookup or (lambda path, sha: {}), wanted_path=state / "wanted.json",
                      **options)


def test_start_freezes_each_source_and_reports_outcomes_per_result(state: Path) -> None:
    one = _audio(state / "one.wav")
    two = _audio(state / "two.wav")
    worker, bgm = _Worker({}), _Bgm({"two": {"status": "ok", "bgm": {"k": "none", "n": 0}}})
    session = _targets_session(state, worker, bgm)
    ids = session.add([one, two])
    assert session.start({ids[0]: ["bgm"], ids[1]: ["inst", "bgm"]}, ["inst"]) == \
        {"started": True, "work": 2}
    snap = session.snapshot()
    assert snap["wants"] == {ids[0]: ["bgm"], ids[1]: ["inst", "bgm"]}
    assert snap["frozenDef"] == ["inst"]
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    snap = session.snapshot()
    # BGMだけの音源は伴奏の分離を呼ばない。伴奏→BGMの順に1音源ずつ。
    assert worker.started == ["two"] and bgm.started == ["one", "two"]
    assert snap["st"] == {ids[0]: "done", ids[1]: "done"}
    assert snap["tst"][ids[0]] == {"bgm": {"k": "held", "n": 3}}
    assert snap["tst"][ids[1]] == {"inst": {"k": "done", "n": 1}, "bgm": {"k": "none", "n": 0}}
    # 次回のSTARTの既定として覚える
    assert _targets_session(state, _Worker({})).snapshot()["bdef"] == ["inst"]


def test_a_source_added_while_processing_takes_the_frozen_default(state: Path) -> None:
    first = _audio(state / "first.wav")
    late = _audio(state / "late.wav")
    worker = _Worker({})
    session = _targets_session(state, worker)
    (first_id,) = session.add([first])
    session.start({first_id: ["inst"]}, ["inst", "bgm"])
    _wait(lambda: session.snapshot()["running"] == first_id)
    (late_id,) = session.add([late])
    snap = session.snapshot()
    assert snap["arrival"]["ids"] == [late_id]
    # 追加は既定で凍結され、既に凍結した音源は動かない
    assert snap["wants"] == {first_id: ["inst"], late_id: ["inst", "bgm"]}
    worker.gate.release()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    assert session.snapshot()["tst"][late_id] == {
        "inst": {"k": "done", "n": 1}, "bgm": {"k": "held", "n": 3}}


def test_partial_failure_retries_only_the_unresolved_result(state: Path) -> None:
    only = _audio(state / "only.wav")
    worker = _Worker({})
    bgm = _Bgm({"only": {"status": "error", "error_code": "bgm_candidate_processing_failed",
                         "message": "x"}})
    session = _targets_session(state, worker, bgm)
    (item,) = session.add([only])
    session.start({item: ["inst", "bgm"]}, ["inst"])
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    snap = session.snapshot()
    assert snap["st"][item] == "failed"
    assert snap["tst"][item] == {"inst": {"k": "done", "n": 1}, "bgm": {"k": "failed", "n": 0}}
    # 一部だけの失敗は音源全体の失敗ではない（PARTIALの行はDesignが組み立てる）
    assert "fail" not in snap["songs"][item] and "hard" not in snap["songs"][item]
    assert snap["have"][item] == {"inst": 1}
    bgm.outcomes.clear()
    assert session.retry_row(item)
    assert session.start({item: ["inst", "bgm"]}, ["inst"])["work"] == 1
    _wait(lambda: session.snapshot()["settle"]["seq"] == 2)
    snap = session.snapshot()
    assert worker.started == ["only"], "受理済みの伴奏を作り直した"
    assert bgm.started == ["only", "only"]
    assert snap["tst"][item] == {"inst": {"k": "done", "n": 1}, "bgm": {"k": "held", "n": 3}}


def test_all_results_existing_completes_without_processing(state: Path) -> None:
    kept = _audio(state / "kept.wav")
    worker, bgm = _Worker({}), _Bgm()

    def lookup(path, sha):
        return {"inst": {"k": "reused", "n": 1}, "bgm": {"k": "none", "n": 0}}

    session = _targets_session(state, worker, bgm, lookup)
    (item,) = session.add([kept])
    _wait(lambda: session.snapshot()["have"].get(item) == {"inst": 1, "bgm": "none"})
    before = session.snapshot()["settle"]["seq"]
    assert session.start({item: ["inst", "bgm"]}, ["inst", "bgm"]) == {"started": True, "work": 0}
    snap = session.snapshot()
    assert worker.started == [] and bgm.started == [] and not session.running()
    assert snap["settle"] == {"seq": before + 1, "kind": "drained"}
    assert snap["elapsed"] == 0
    assert snap["tst"][item] == {"inst": {"k": "reused", "n": 1}, "bgm": {"k": "reused", "n": 0}}


def test_existing_result_is_not_used_once_the_source_changes(state: Path) -> None:
    import os

    kept = _audio(state / "kept.wav")

    def lookup(path, sha):
        return {"inst": {"k": "reused", "n": 1}}

    session = _targets_session(state, _Worker({}), lookup=lookup)
    (item,) = session.add([kept])
    _wait(lambda: session.snapshot()["have"].get(item) == {"inst": 1})
    _audio(kept, channels=1)
    os.utime(kept, ns=(time.time_ns(), time.time_ns() + 10**9))
    assert session.snapshot()["have"].get(item) is None, "変わった原本に既存の結果を言った"


def test_halt_during_the_first_result_does_not_start_the_second(state: Path) -> None:
    only = _audio(state / "only.wav")
    worker = _Worker({"only": {"status": "error", "error_code": "worker_unavailable", "message": "x"}})
    bgm = _Bgm()
    session = _targets_session(state, worker, bgm)
    (item,) = session.add([only])
    session.start({item: ["inst", "bgm"]}, ["inst"])
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "halted")
    assert bgm.started == []
    session.halt_skip()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    snap = session.snapshot()
    assert snap["tst"][item] == {"inst": {"k": "failed", "n": 0}, "bgm": {"k": "failed"}}
    assert bgm.started == []


def test_forecast_needs_measured_samples_of_the_same_results(state: Path) -> None:
    import numpy as np
    import soundfile as sf

    now = [0.0]
    paths = []
    for index in range(4):
        path = state / f"s{index}.wav"
        sf.write(str(path), np.zeros((44100 * 3, 2), dtype="float32"), 44100, subtype="PCM_16")
        paths.append(path)
    worker = _Worker({})
    session = _targets_session(state, worker, clock=lambda: now[0])
    ids = session.add(paths)
    session.start({i: ["inst"] for i in ids}, ["inst"])
    assert session.snapshot()["forecast"] is None
    for index, item in enumerate(ids[:2]):
        _wait(lambda item=item: session.snapshot()["running"] == item)
        session._on_progress(item, {"phase": "separating"})
        now[0] += 10.0 * (index + 1)
        worker.gate.release()
    _wait(lambda: session.snapshot()["running"] == ids[2])
    session._on_progress(ids[2], {"phase": "separating"})
    forecast = session.snapshot()["forecast"]
    assert forecast is not None and "–" in forecast
    worker.gate.release()
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    assert session.snapshot()["forecast"] is None


def test_existing_results_are_looked_up_again_after_processing(state: Path) -> None:
    """READYで「既存なし」と照会した音源も、処理で結果ができた後は既存として見える。"""
    only = _audio(state / "only.wav")
    made: list[str] = []

    def lookup(path, sha):
        return {"inst": {"k": "reused", "n": 1}} if made else {}

    worker = _Worker({})
    session = _targets_session(state, worker, lookup=lookup)
    (first,) = session.add([only])
    _wait(lambda: session.snapshot() and session._existing._known)
    assert session.snapshot()["have"] == {}
    session.start({first: ["inst"]}, ["inst"])
    made.append("inst")
    worker.gate.release()
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    session.back_to_ready()
    (again,) = session.add([only])
    _wait(lambda: session.snapshot()["have"].get(again) == {"inst": 1})


def test_complete_tells_reuse_and_rebuild_apart_from_a_new_save(state: Path) -> None:
    reused, rebuilt = _audio(state / "reused.wav"), _audio(state / "rebuilt.wav")
    def bgm_result(state_name, **counts):
        return {"status": "ok", "bgm": {"k": "done", "n": 2, "held": 1, "files": ["a.flac", "b.flac"],
                                        "product_result": {"schema_version": 1, "state": state_name,
                                                           "published_count": 2, "held_count": 1,
                                                           "unsupported_count": None, "assets": [{"path": "a"}] * 2,
                                                           "known": [{"state": "KNOWN"}] * 50, **counts}}}
    bgm = _Bgm({"reused": bgm_result("REUSED", rebuilt_count=0),
                "rebuilt": bgm_result("REBUILT", rebuilt_count=1)})
    session = _targets_session(state, _Worker({}), bgm)
    session.add([reused, rebuilt])
    ids = _source_ids(session, [reused, rebuilt])
    session.start({ids[0]: ["bgm"], ids[1]: ["bgm"]}, ["bgm"])
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    tst = session.snapshot()["tst"]
    # 保存済みを使っただけの結果は既存として言い、作り直しは新しい保存として件数を添える。
    assert tst[ids[0]]["bgm"]["k"] == "reused" and tst[ids[1]]["bgm"]["k"] == "done"
    assert tst[ids[1]]["bgm"]["result"]["rebuilt_count"] == 1
    # 未確認の件数は未確認のまま。完了画面へは要る事実だけを渡す（全体は結果票と状態にある）。
    assert tst[ids[0]]["bgm"]["result"]["unsupported_count"] is None
    assert "assets" not in tst[ids[0]]["bgm"]["result"] and "known" not in tst[ids[0]]["bgm"]["result"]


# --- 凍結Design 40a: 終了・トレイ・フォルダ -------------------------------------


def test_title_bar_buttons_are_wired_by_the_design_itself() -> None:
    """40a のタイトルバーは最小化（トレイ）と閉じる（終了）を Design 自身が配線する。"""
    source = page.CANVAS.read_text(encoding="utf-8")
    assert 'onClick="{{ onMin }}" data-tip="min"' in source
    assert 'onClick="{{ onClose }}" data-tip="close"' in source
    bridge = page.BRIDGE.read_text(encoding="utf-8")
    # 擬似デスクトップ・「LSS を終了しました」の足場へ入らず、OS の面へ渡す
    assert "this.lssCall('minimize')" in bridge and "this.lssCall('quit')" in bridge
    assert "win: 'tray'" not in bridge and "win: 'quit'" not in bridge


def test_tray_tip_says_the_state_in_one_line_counting_sources() -> None:
    from lss.ui.web.tray import tray_tip

    snap = {"queue": ["a", "b", "c"], "st": {"a": "done", "b": "failed", "c": "queued"}}
    assert tray_tip({"phase": "ready"}, snap) == "待機中"
    assert tray_tip({"phase": "processing"}, snap) == "処理中（2 / 3）"
    assert tray_tip({"phase": "processing", "stopReq": True}, snap) == "停止待ち"
    assert tray_tip({"phase": "processing", "fault": True}, snap) == "操作が必要です"
    assert tray_tip({"phase": "complete"}, snap) == "完了"


def test_completion_notice_repeats_only_what_complete_counted() -> None:
    from lss.ui.web.tray import completion_notice

    assert completion_notice(["伴奏 6 件", "BGM 9 曲", "要確認 3 件"]) == (
        "SESSION COMPLETE", "伴奏 6 件、BGM 9 曲、要確認 3 件\nクリックで LSS を開く")
    assert completion_notice(None) == ("SESSION COMPLETE", "クリックで LSS を開く")


def test_only_listed_source_folders_can_be_opened(tmp_path: Path) -> None:
    from lss.ui.web.host import folder_to_open

    album = tmp_path / "Album"
    (album / "LSS").mkdir(parents=True)
    plain = tmp_path / "Plain"
    plain.mkdir()
    snap = {"songs": {"a": {"dir": str(album)}, "b": {"dir": str(plain)}}}
    assert folder_to_open(str(album), snap) == album
    assert folder_to_open(str(album / "LSS"), snap) == album / "LSS"
    # まだ何も保存していない音源の LSS は、音源の場所を開く
    assert folder_to_open(str(plain / "LSS"), snap) == plain
    for outside in (str(tmp_path), str(album / "LSS" / "x.exe"), r"C:\Windows", "", None):
        assert folder_to_open(outside, snap) is None


def test_completion_notifies_only_while_hidden_in_the_tray(state: Path) -> None:
    """全画面表示中は画面が既に言っているので通知しない。トレイにいる間だけ1回。"""
    from lss.ui.web.host import Host

    class FakeTray:
        def __init__(self) -> None:
            self.notices: list[tuple[str, str]] = []

        def notify(self, title: str, text: str) -> None:
            self.notices.append((title, text))

    host = Host(session=_session(state, _Worker({})))
    tray = FakeTray()
    host._tray = tray
    host.completion.perform = lambda when_done: when_done
    host.session_complete("none", ["伴奏 2 件"])
    assert tray.notices == []
    host._hidden = True
    host.session_complete("quit", ["伴奏 2 件"])
    assert tray.notices == [], "終了する完了動作では通知しない"
    host.session_complete("none", ["伴奏 2 件", "要確認 1 件"])
    assert tray.notices == [("SESSION COMPLETE", "伴奏 2 件、要確認 1 件\nクリックで LSS を開く")]
