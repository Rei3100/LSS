"""Claude Design「LSS System Assets」の図形と、トレイの状態・メニューの中身。

図面の座標は Design のファイル（`design/system-assets/LSS System Assets.dc.html`）を
正とし、ここで持つ図形がそれと一致することを機械的に確かめる。
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import pytest

from lss.ui import system_assets as sa
from lss.ui.web import tray

ROOT = Path(__file__).resolve().parents[1]
DESIGN = (ROOT / "design" / "system-assets" / "LSS System Assets.dc.html").read_text(encoding="utf-8")


def _polygons(svg: str) -> list[tuple[str, str]]:
    return re.findall(r'<polygon points="([^"]+)" fill="([^"]+)"', svg)


def _design_svgs(label_size: int) -> list[str]:
    return re.findall(rf'<svg width="{label_size}" height="{label_size}" viewBox="[^"]+"[^>]*>(.*?)</svg>',
                      DESIGN, flags=re.S)


@pytest.mark.parametrize("size", [16, 20, 24, 32])
def test_app_icon_redraws_match_the_design(size: int) -> None:
    ours = _polygons(sa.app_icon_svg(size))
    assert any(_polygons(svg) == ours for svg in _design_svgs(size))


def test_app_icon_master_matches_the_design() -> None:
    ours = _polygons(sa.app_icon_svg(256))
    assert any(_polygons(svg) == ours for svg in _design_svgs(256))


@pytest.mark.parametrize("state", list(sa.TrayState))
def test_tray_states_at_16px_match_the_design(state: sa.TrayState) -> None:
    ours = _polygons(sa.tray_svg(state, 16))
    assert any(_polygons(svg) == ours for svg in _design_svgs(16)), state


def test_tray_on_light_taskbar_matches_the_design() -> None:
    ours = _polygons(sa.tray_svg(sa.TrayState.PROCESSING, 16, light_taskbar=True))
    assert ours == [("7,2 15,2 13,7 5,7", "#1E7F92"), ("3,9 11,9 9,14 1,14", "#101013")]


@pytest.mark.parametrize("size", [20, 24])
def test_ready_tray_at_dpi_sizes_matches_the_design(size: int) -> None:
    ours = _polygons(sa.tray_svg(sa.TrayState.READY, size))
    assert any(_polygons(svg) == ours for svg in _design_svgs(size))


def test_state_is_carried_by_geometry_not_only_colour() -> None:
    shapes = {state: sa.tray_geometry(state, size) for state in sa.TrayState for size in [16]}
    assert len({shape for shape in shapes.values()}) == len(sa.TrayState)
    for size in sa.TRAY_SIZES:
        ready = sa.tray_geometry(sa.TrayState.READY, size)
        for state in (sa.TrayState.PROCESSING, sa.TrayState.COMPLETE, sa.TrayState.HALTED):
            assert sa.tray_geometry(state, size) != ready


def test_processing_pulse_alternates_offset_four_and_three() -> None:
    offset = lambda frame: (sa.tray_geometry(sa.TrayState.PROCESSING, 16, frame)[0][0][0]
                            - sa.tray_geometry(sa.TrayState.PROCESSING, 16, frame)[1][0][0])
    assert (offset(0), offset(1)) == (4, 3)


def test_complete_opens_to_kerf_four_and_holds() -> None:
    kerf = lambda frame: (sa.tray_geometry(sa.TrayState.COMPLETE, 16, frame)[1][0][1]
                          - sa.tray_geometry(sa.TrayState.COMPLETE, 16, frame)[0][2][1])
    assert [kerf(f) for f in range(3)] == [2, 3, 4]
    assert sa.tray_geometry(sa.TrayState.COMPLETE, 16) == sa.tray_geometry(sa.TrayState.COMPLETE, 16, 2)


def test_cursors_match_the_design_geometry() -> None:
    for cursor, marker in ((sa.Cursor.DEFAULT, "DEFAULT"), (sa.Cursor.INTERACTIVE, "INTERACTIVE"),
                           (sa.Cursor.WORKING, "WORKING")):
        section = DESIGN[DESIGN.index(f">{marker}<"):]
        design = re.search(r'<svg width="32" height="32" viewBox="0 0 32 32">(.*?)</svg>', section, re.S).group(1)
        assert _polygons(sa.cursor_svg(cursor)) == _polygons(design), cursor


def test_every_raster_was_baked(tmp_path: Path) -> None:
    baked = ROOT / "assets" / "system"
    missing = [raster.name for raster in sa.rasters() if not (baked / raster.name).is_file()]
    assert not missing, "scripts/build_system_assets.py を実行して焼き直す"
    for state in sa.TrayState:
        for light in (False, True):
            for frame in range(sa.tray_frames(state)):
                assert (baked / sa.tray_ico_name(state, light, frame)).is_file()


def test_app_icon_file_holds_every_redrawn_size() -> None:
    data = (ROOT / "assets" / "app.ico").read_bytes()
    _reserved, kind, count = struct.unpack("<HHH", data[:6])
    sizes = sorted(struct.unpack("<B", data[6 + 16 * i:7 + 16 * i])[0] or 256 for i in range(count))
    assert kind == 1 and sizes == sorted(sa.APP_ICON_SIZES)


# --- トレイの状態とメニュー ----------------------------------------------------

_SNAPSHOT = {
    "queue": ["a", "b", "c"],
    "st": {"a": "done", "b": "queued", "c": "queued"},
    "songs": {
        "a": {"num": "01", "name": "one", "path": ('D' + chr(58) + '\\Music\\A\\one.flac'), "dir": ('D' + chr(58) + '\\Music\\A')},
        "b": {"num": "02", "name": "two", "path": ('D' + chr(58) + '\\Music\\B\\two.mp3'), "dir": ('D' + chr(58) + '\\Music\\B')},
        "c": {"num": "03", "name": "three", "path": ('D' + chr(58) + '\\Music\\B\\three.wav'), "dir": ('D' + chr(58) + '\\Music\\B')},
    },
    "running": "b",
    "stage": {"k": "SEPARATE", "p": None},
    "fault": None,
}


def _rows(model: dict) -> list[tuple[str, str]]:
    return [(item["en"], item["jp"]) for item in model["items"]]


def test_tray_state_follows_the_screen() -> None:
    assert tray.tray_state({"phase": "ready"}) is sa.TrayState.READY
    assert tray.tray_state({"phase": "processing"}) is sa.TrayState.PROCESSING
    assert tray.tray_state({"phase": "processing", "fault": True}) is sa.TrayState.HALTED
    assert tray.tray_state({"phase": "complete"}) is sa.TrayState.COMPLETE


def test_ready_menu_uses_the_in_app_menu_grammar() -> None:
    model = tray.menu_model({"phase": "ready"}, dict(_SNAPSHOT, running=None))
    assert model["head"] == "LSS" and model["foot"] == "未処理 2 件"
    assert _rows(model) == [("OPEN", "LSS を表示"), ("QUIT", "LSS を終了")]


def test_empty_menu_has_no_count() -> None:
    model = tray.menu_model({"phase": "ready"}, {"queue": [], "st": {}, "songs": {}})
    assert model["foot"] == ""


def test_processing_menu_names_the_source_and_omits_quit() -> None:
    model = tray.menu_model({"phase": "processing"}, _SNAPSHOT)
    assert model["head"] == "two" and model["foot"] == "1 / 3 完了"
    assert _rows(model) == [("OPEN", "LSS を表示"), ("STOP", "停止")]


def test_stopping_menu_offers_no_second_stop() -> None:
    model = tray.menu_model({"phase": "processing", "stopReq": True}, _SNAPSHOT)
    assert _rows(model) == [("OPEN", "LSS を表示")] and model["foot"] == "停止待ち"


def test_complete_menu_and_saved_folders() -> None:
    model = tray.menu_model({"phase": "complete"}, _SNAPSHOT)
    assert model["foot"] == "1 件 完了"
    assert _rows(model) == [("OPEN FOLDER", "保存先を開く"), ("OPEN", "LSS を表示"), ("QUIT", "LSS を終了")]
    assert tray.saved_folders(_SNAPSHOT) == [('D' + chr(58) + '\\Music\\A\\LSS')]


def test_halted_menu_uses_the_halt_card_actions_and_no_codes() -> None:
    snapshot = dict(_SNAPSHOT, running=None, fault={"key": "insufficient_output_space", "item": "b"})
    model = tray.menu_model({"phase": "processing", "fault": True}, snapshot)
    assert model["head"] == "two" and model["foot"] == "中断"
    assert _rows(model) == [("RETRY", "再試行"), ("SKIP", "この音源を飛ばす"),
                            ("OPEN", "LSS を表示"), ("QUIT", "LSS を終了")]
    assert "insufficient" not in str(model)


def test_tray_menu_copies_the_in_app_context_menu_values() -> None:
    app = (ROOT / "design" / "reference" / "LSS App.dc.html").read_text(encoding="utf-8")
    menu = (ROOT / "lss" / "ui" / "web" / "tray_menu.html").read_text(encoding="utf-8")
    for value in ("width: '264px'", "background: 'oklch(0.185 0.006 264)'",
                  "clipPath: 'polygon(0 0,100% 0,98.5% 100%,1.5% 100%)'",
                  "boxShadow: '7px 9px 0 rgba(0,0,0,.45)'", "padding: '9px 14px 7px'",
                  "padding: '8px 14px'", "padding: '8px 13px 9px'",
                  "fontSize: '13.5px', letterSpacing: '1.2px'"):
        assert value in app, value
    for value in ("width:264px", "background:oklch(0.185 0.006 264)",
                  "clip-path:polygon(0 0,100% 0,98.5% 100%,1.5% 100%)",
                  "box-shadow:7px 9px 0 rgba(0,0,0,.45)", "padding:9px 14px 7px",
                  "padding:8px 14px", "padding:8px 13px 9px", "font-size:13.5px;letter-spacing:1.2px"):
        assert value in menu, value


def test_saved_folders_include_reused_results_but_not_zero_results() -> None:
    snapshot = {
        "queue": ["a", "b", "c"], "st": {"a": "skipped", "b": "skipped", "c": "failed"},
        "songs": {"a": {"dir": ('D' + chr(58) + '\\Music\\A')}, "b": {"dir": ('D' + chr(58) + '\\Music\\B'), "zero": True},
                  "c": {"dir": ('D' + chr(58) + '\\Music\\C')}},
    }
    assert tray.saved_folders(snapshot) == [('D' + chr(58) + '\\Music\\A\\LSS')]
    model = tray.menu_model({"phase": "complete"}, snapshot)
    assert _rows(model)[0] == ("OPEN FOLDER", "保存先を開く")
    assert model["foot"] == "0 件 完了"
