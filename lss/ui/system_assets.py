"""OSの面（アプリアイコン・トレイアイコン・カーソル）に出すLSSの図形。Qt非依存。

図形の権威は Claude Design の `design/system-assets/LSS System Assets.dc.html`。
ここは図面の座標をそのまま持ち、Designが描かなかった組み合わせ（20/24 px の
PROCESSING・COMPLETE・HALTED、動きの中間コマ、明るいタスクバー）だけを、Designが
言葉で定めた規則（kerf・横ずれ・傾きの反転）で同じ整数グリッド上に導く。

ラスタ化はしない。SVGの文字列を返し、画素にするのはブラウザ（WebView2）の役目
（`scripts/build_system_assets.py`）。Designの受け入れ基準は描画結果なので、
Designが表示に使ったのと同じ描画系で焼く。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

RED = "#D73431"
CYAN = "#43C7DE"
GOLD = "#E5A82A"
BONE = "#F1F2F4"
INK = "#101013"
BACKING = "#08080A"
# 明るいタスクバーでは cyan だけを暗くする（Design §G TRAY ICON）。
CYAN_ON_LIGHT = "#1E7F92"

Quad = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


def _points(quad: Quad) -> str:
    return " ".join(f"{x},{y}" for x, y in quad)


def _svg(size: int, view: int, polygons: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 {view} {view}">' + "".join(polygons) + "</svg>")


def _poly(quad_or_points, fill: str, extra: str = "") -> str:
    points = quad_or_points if isinstance(quad_or_points, str) else _points(quad_or_points)
    return f'<polygon points="{points}" fill="{fill}"{extra}></polygon>'


# --- B. APPLICATION ICON ------------------------------------------------------

APP_ICON_SIZES = (16, 20, 24, 32, 48, 64, 256)

# 256/64/48 は同じマスター（背後の板つき）。32 と 24 は影を落として kerf を太らせ、
# 20 と 16 は切った四辺形をやめて平行な2本の棒にする。
_APP_MASTER = [
    _poly("104,44 240,44 226,108 80,148", BACKING),
    _poly("52,160 198,120 172,232 36,232", BACKING),
    _poly("94,34 230,34 216,98 70,138", RED),
    _poly("42,150 188,110 162,222 26,222", BONE),
]
_APP_REDRAWN = {
    32: [_poly("12,4 29,4 27,12 9,17", RED), _poly("5,19 23,14 20,28 3,28", BONE)],
    24: [_poly("9,3 22,3 20,9 7,13", RED), _poly("4,15 17,10 15,21 2,21", BONE)],
    20: [_poly("7,3 18,3 17,8 6,11", RED), _poly("3,13 14,10 13,17 2,17", BONE)],
    16: [_poly("6,2 14,2 12,7 4,7", RED), _poly("4,9 12,9 10,14 2,14", BONE)],
}


def app_icon_svg(size: int) -> str:
    if size in _APP_REDRAWN:
        return _svg(size, size, _APP_REDRAWN[size])
    if size in (48, 64, 256):
        return _svg(size, 256, _APP_MASTER)
    raise ValueError(f"アプリアイコンは描き分けたサイズだけ: {size}")


# --- C. SYSTEM TRAY ICON ------------------------------------------------------


COMPLETE_FINAL_FRAME = 2


class TrayState(Enum):
    READY = "ready"
    PROCESSING = "processing"
    COMPLETE = "complete"
    HALTED = "halted"


TRAY_SIZES = (16, 20, 24)
LEAN = 2  # 1本あたりの傾きのずれ（16/20/24 px とも 2 px）

# READY の2本（Design §C DPI SCALING が描いた整数グリッド）。上が accent 面、下が ink 面。
_TRAY_READY: dict[int, tuple[Quad, Quad]] = {
    16: (((6, 2), (14, 2), (12, 7), (4, 7)), ((4, 9), (12, 9), (10, 14), (2, 14))),
    20: (((7, 3), (18, 3), (16, 9), (5, 9)), ((5, 11), (16, 11), (14, 17), (3, 17))),
    24: (((9, 3), (21, 3), (19, 10), (7, 10)), ((5, 13), (17, 13), (15, 20), (3, 20))),
}


def _shift(quad: Quad, dx: int = 0, dy: int = 0) -> Quad:
    return tuple((x + dx, y + dy) for x, y in quad)  # type: ignore[return-value]


def _reverse_lean(quad: Quad) -> Quad:
    """上辺を傾きの分だけ左へ、下辺を右へ。平行な2本が楔になる。"""
    (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = quad
    return ((tlx - LEAN, tly), (trx - LEAN, try_), (brx + LEAN, bry), (blx + LEAN, bly))


def tray_geometry(state: TrayState, size: int, frame: int | None = None) -> tuple[Quad, Quad]:
    """トレイの2本。`frame` は動きのコマで、None は静止形。

    PROCESSING: 0（横ずれ 4 px、静止形）/ 1（3 px）。
    COMPLETE: 0 → 1 → 2 と一度だけ開き、2 が静止形（kerf 4）。
    """
    upper, lower = _TRAY_READY[size]
    if state is TrayState.PROCESSING:
        # 横ずれを広げる。1.2 s ごとに 4 px と 3 px を交互（下の板だけ戻る）。
        upper = _shift(upper, dx=1)
        if not frame:
            lower = _shift(lower, dx=-1)
    elif state is TrayState.COMPLETE:
        opened = COMPLETE_FINAL_FRAME if frame is None else frame
        if opened >= 1:
            upper = _shift(upper, dy=-1)
        if opened >= 2:
            lower = _shift(lower, dy=1)
    elif state is TrayState.HALTED:
        upper = _reverse_lean(upper)
    return upper, lower


def tray_colors(state: TrayState, light_taskbar: bool) -> tuple[str, str]:
    """(上の面, 下の面)。ink 面はタスクバーの明暗に従い、accent 面が状態を運ぶ。"""
    ink = INK if light_taskbar else BONE
    accent = {
        TrayState.READY: ink,
        TrayState.PROCESSING: CYAN_ON_LIGHT if light_taskbar else CYAN,
        TrayState.COMPLETE: GOLD,
        TrayState.HALTED: RED,
    }[state]
    return accent, ink


def tray_svg(state: TrayState, size: int, *, light_taskbar: bool = False,
             frame: int | None = None) -> str:
    upper, lower = tray_geometry(state, size, frame)
    accent, ink = tray_colors(state, light_taskbar)
    return _svg(size, size, [_poly(upper, accent), _poly(lower, ink)])


def tray_frames(state: TrayState) -> int:
    if state is TrayState.PROCESSING:
        return 2
    if state is TrayState.COMPLETE:
        return COMPLETE_FINAL_FRAME + 1
    return 1


# 動き。PROCESSING は 1.2 s の2コマ（補間なし）、COMPLETE は一度だけ開く。
PROCESSING_STEP_SECONDS = 1.2
COMPLETE_STEP_SECONDS = 0.12

# --- D. CURSOR FAMILY ---------------------------------------------------------


class Cursor(Enum):
    DEFAULT = "default"
    INTERACTIVE = "interactive"
    WORKING = "working"


CURSOR_SIZES = {1.0: 32, 1.25: 40, 1.5: 48}

_ARROW_PLANE = "3,3 3,23 9,18 13,28 17,26 13,16 20,16"
_ARROW = "1,1 1,21 7,16 11,26 15,24 11,14 18,14"
_EDGE = ' stroke="#101013" stroke-width="1.4" stroke-linejoin="miter"'
_BADGE_EDGE = ' stroke="#101013" stroke-width="1"'


def cursor_svg(cursor: Cursor, size: int = 32, *, ink_plane: bool = False, frame: int = 0) -> str:
    """32 グリッドの図面を `size` px に。ホットスポットは全種 (0,0)。

    `ink_plane`: 赤・シアン・金の面の上では背後の板を ink にする（色は荷重を持たない）。
    WORKING の `frame` 1 は、バッジの kerf を 4 から 5 px へ開いたコマ。
    """
    plane = INK if ink_plane else RED
    parts = [_poly(_ARROW_PLANE, plane), _poly(_ARROW, BONE, _EDGE)]
    if cursor is Cursor.INTERACTIVE:
        parts += [_poly("20,16 30,16 28,21 18,21", RED, _BADGE_EDGE),
                  _poly("18,23 28,23 26,28 16,28", BONE, _BADGE_EDGE)]
    elif cursor is Cursor.WORKING:
        lower = "17,24 27,24 25,29 15,29" if frame == 0 else "17,25 27,25 25,30 15,30"
        parts += [_poly("21,15 31,15 29,20 19,20", CYAN, _BADGE_EDGE),
                  _poly(lower, CYAN, _BADGE_EDGE)]
    return _svg(size, 32, parts)


def cursor_frames(cursor: Cursor) -> int:
    return 2 if cursor is Cursor.WORKING else 1


WORKING_STEP_SECONDS = 1.0


# --- 焼いたファイルの名前 ------------------------------------------------------


def tray_png_name(state: TrayState, size: int, light: bool, frame: int) -> str:
    return f"tray-{state.value}-{'light' if light else 'dark'}-{size}-f{frame}.png"


def tray_ico_name(state: TrayState, light: bool, frame: int) -> str:
    return f"tray-{state.value}-{'light' if light else 'dark'}-f{frame}.ico"


def cursor_png_name(cursor: Cursor, size: int, ink_plane: bool, frame: int) -> str:
    return f"cursor-{cursor.value}{'-ink' if ink_plane else ''}-{size}-f{frame}.png"


@dataclass(frozen=True)
class Raster:
    """焼く1枚。`svg` を `size` px 四方で透明背景に描く。"""

    name: str
    size: int
    svg: str


def rasters() -> list[Raster]:
    """焼くべき全画像。アプリアイコンの各サイズ、トレイ、カーソル。"""
    out = [Raster(f"app-{size}.png", size, app_icon_svg(size)) for size in APP_ICON_SIZES]
    for state in TrayState:
        for light in (False, True):
            for frame in range(tray_frames(state)):
                for size in TRAY_SIZES:
                    out.append(Raster(tray_png_name(state, size, light, frame), size,
                                      tray_svg(state, size, light_taskbar=light, frame=frame)))
    for cursor in Cursor:
        for ink in (False, True):
            for frame in range(cursor_frames(cursor)):
                for size in CURSOR_SIZES.values():
                    out.append(Raster(cursor_png_name(cursor, size, ink, frame), size,
                                      cursor_svg(cursor, size, ink_plane=ink, frame=frame)))
    return out


def ico_bytes(pngs: list[tuple[int, bytes]]) -> bytes:
    """PNG をそのまま格納した ICO（Vista 以降の形式）。サイズごとに描き分けた画像を束ねる。"""
    import struct

    entries = sorted(pngs)
    header = struct.pack("<HHH", 0, 1, len(entries))
    offset = 6 + 16 * len(entries)
    directory = b""
    blobs = b""
    for size, data in entries:
        edge = 0 if size >= 256 else size
        directory += struct.pack("<BBBBHHII", edge, edge, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
        blobs += data
    return header + directory + blobs
