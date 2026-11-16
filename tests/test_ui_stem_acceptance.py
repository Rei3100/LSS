"""配備候補の画面でSTEMを選び、単独と組み合わせのステムが1回の実行でそろって保存されることを確かめる受入。

隠しデスクトップで実行物を動かす（利用者の画面を奪わない）。外部への通信は閉じ、モデルは手元の取得済みの
ものをhard linkで置く。明示実行のみ:

    LSS_RUN_UI_ACCEPTANCE=1 LSS_ACCEPTANCE_EXECUTABLE=<dist\\LSS\\LSS.exe> python -m pytest tests/test_ui_stem_acceptance.py
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root


import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.harness import tsv_io
from tests.test_ui_deployed_acceptance import _clip, _evidence, _offline_environment, _portable

pytestmark = pytest.mark.skipif(os.environ.get("LSS_RUN_UI_ACCEPTANCE") != "1",
                                reason="配備された画面と実GPU・実曲を使う受入は明示実行")

ENTER = dict(key="Enter", code="Enter", vk=13)
# 画面の表示に出してはいけない区切り（中黒、全角コロン、中点の区切り）。
FORBIDDEN = ("・", "：", " · ")


def _link_tree(source: Path, target: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.link(path, destination)


def _unscaled(path: Path) -> np.ndarray:
    """保存の縮小（2のべき乗）を戻した値。組み合わせとその和を同じ尺度で比べる。"""
    values, _ = sf.read(path, dtype="float64", always_2d=True)
    shift = int((FLAC(path).get("LSS_OUTPUT_HEADROOM_SHIFT") or ["0"])[0])
    return values * 2.0 ** shift


def test_stem_singles_and_combinations_are_saved_together_from_one_start(tmp_path, request):
    from tests.deployed_ui_driver import DeployedApp, wait_for

    repository = Path(__file__).resolve().parents[1]
    evidence = _evidence(tmp_path, "stem")
    deployed, executable = _portable(tmp_path, request)
    product_models = Path(os.environ["LSS_ACCEPTANCE_PRODUCT_MODELS"])
    stem_models = Path(os.environ["LSS_ACCEPTANCE_STEM_MODELS"])
    models = tmp_path / "models"
    # STEMの確かめ（鳴っている音の分類）はPANNs Cnn14を読む。オフラインなので取得済みの重みを置く。
    for name in ("becruily_deux_inst", "becruily_karaoke_bs_roformer", "panns_cnn14_audioset"):
        _link_tree(product_models / name, models / name)
    _link_tree(stem_models, models)

    _, corpus = tsv_io.load_tsv(dev_evidence_root(repository) / "ledger" / "test_set.tsv")
    inbox = tmp_path / "入力"
    inbox.mkdir()
    song = inbox / ("実曲S" + Path(corpus[0]["mix_path"]).suffix)
    _clip(Path(corpus[0]["mix_path"]), song, 40)
    state = tmp_path / "state"
    app = DeployedApp(executable, _offline_environment(state, models))
    record: dict = {"executable": str(deployed), "steps": []}
    try:
        page = app.launch()
        view = page.state()
        assert view["phase"] == "ready"
        order = page.eval("window.__lssComponent.T")
        assert order == ["bgm", "inst", "snd"], order          # BGM、INSTRUMENTAL、STEMの順
        labels = page.eval("Array.from(document.querySelectorAll('*')).filter(e => e.children.length === 0)"
                           ".map(e => e.textContent.trim()).filter(t => /^(BGM|INSTRUMENTAL|STEM)$/.test(t))")
        assert labels[:3] == ["BGM", "INSTRUMENTAL", "STEM"], labels
        app.choose_with_key(page, inbox, [song.name])
        wait_for(lambda: len(page.state()["queue"]) == 1, 20, "音源の追加")
        page.key(key="2", code="Digit2", vk=50)                  # INSTRUMENTALを外す
        page.key(key="3", code="Digit3", vk=51)                  # STEMを選ぶ
        for chip in ("+ ベース", "+ ギター"):
            page.click_text(chip)
        page.click_text("組み合わせを追加")
        page.click_text("+ ベース")
        page.click_text("+ ギター")
        page.click_text("ベース+ギター を追加")
        page.click_text("組み合わせを追加")
        page.click_text("+ コーラス")
        page.click_text("+ 伴奏")
        page.click_text("コーラス+伴奏 を追加")
        chosen = page.eval("window.__lssComponent.state.sounds")
        assert chosen.split("、") == ["ベース", "ギター", "ベースとギター", "コーラスと伴奏"], chosen
        time.sleep(1.0)
        page.screenshot(evidence / "01-stem-chosen.png")
        shown = page.eval("document.body.innerText")
        assert not [mark for mark in FORBIDDEN if mark in shown], shown

        page.key(**ENTER)
        done = wait_for(lambda: (lambda v: v if v["phase"] == "ready" and v["allDone"] else None)(page.state()),
                        1800, "STEMの完了")
        page.screenshot(evidence / "02-done.png")
        record["final_state"] = {k: done[k] for k in ("st", "fault", "allDone")}
        shown = page.eval("document.body.innerText")
        assert not [mark for mark in FORBIDDEN if mark in shown], shown

        saved = sorted((song.parent / "LSS").glob("*"))
        record["saved"] = [path.name for path in saved]
        assert not [path for path in saved if path.suffix.lower() == ".html"], "結果HTMLが音楽フォルダにある"
        by_label = {}
        for path in saved:
            for label in ("ベース+ギター", "コーラス+伴奏", "ベース", "ギター"):
                if path.suffix == ".flac" and path.stem.endswith(f" - {label}"):
                    by_label[label] = path
        assert set(by_label) == {"ベース", "ギター", "ベース+ギター", "コーラス+伴奏"}, record["saved"]
        bass, guitar, both = (_unscaled(by_label[name]) for name in ("ベース", "ギター", "ベース+ギター"))
        frames = min(len(bass), len(guitar), len(both))
        difference = float(np.abs(both[:frames] - bass[:frames] - guitar[:frames]).max())
        record["bass_plus_guitar_max_abs_difference"] = difference
        assert difference < 1e-5, difference                     # 同じ推論の和（PCM24の丸めの範囲）
        assert app.quit_from_tray_menu() == 0
    finally:
        (evidence / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        if app.process is not None and app.process.poll() is None:
            app.process.kill()
