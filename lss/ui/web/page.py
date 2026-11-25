"""凍結Designのページから、アプリで出すページを組み立てる。

**Designを写し直さない。** 受け入れ基準は Claude Design が作ったUIそのもので、
それ以上でも以下でもない（ユーザーの明示の要求）。Qt Widgets・QMLへの写し直しは
毎回ずれた（LV-UI-001）。ここは `design/reference/LSS App.dc.html` をそのまま使い、
アプリとして出すのに必要な差し替えだけを、名前付きで・1か所ずつ・確実に当てる。

差し替えは全部 `PATCHES` に並べる。どれかが原本の中で「ちょうど1回」見つからなければ
組み立ては失敗する。Designが書き出し直されて前提が崩れたとき、静かにずれた画面を
出さないため。

見た目・動き・文言・構図に触る差し替えはここに置かない。
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from hashlib import sha384
from pathlib import Path


def _root() -> Path:
    frozen = getattr(sys, "_MEIPASS", None)
    return Path(frozen) if frozen else Path(__file__).resolve().parents[3]


ROOT = _root()
REFERENCE = ROOT / "design" / "reference"
CANVAS = REFERENCE / "LSS App.dc.html"
RUNTIME = Path(__file__).with_name("runtime.js")
FONT_DIR = ROOT / "assets" / "fonts"
WEB_DIR = ROOT / "assets" / "web"
BRIDGE = Path(__file__).with_name("bridge.js")
TRAY_MENU = Path(__file__).with_name("tray_menu.html")
SYSTEM_ASSETS = ROOT / "assets" / "system"

# LSS pins the vendored React and ReactDOM 18.3.1 UMD builds by SHA-384.
_REACT = (
    ("react.production.min.js",
     "0c6c8bc40ca3ab47fd48fa557af0fa220ced085967305ea85bf5d01a67def88b19f13a84883adc1e430b288e9f881fd9"),
    ("react-dom.production.min.js",
     "8131b1873db595519834c71d24ecaad3511d8348e19ff736da7b31d24caa3f44f1695e5655db121f57d20d47f96098f5"),
)

# 凍結Designの1画面の参照構図。実機ではこれが収まる倍率で拡大し、画面の縦横比との差は
# フレーム自身が伸びて受ける（2560x1440 / 150% なら 1600x900）。
FRAME_WIDTH, FRAME_HEIGHT = 1440, 900


@dataclass(frozen=True)
class Patch:
    name: str
    why: str
    old: str
    new: str


# 同梱書体。凍結Designが Google Fonts から読む書体を、ネットに頼らず同じ名前で出す。
# Noto Sans JP は 700 だけを同梱している（凍結Designの日本語は 700 中心）。
_FACES = (
    ("Anton", 400, "Anton-Regular.ttf"),
    ("Oswald", 400, "Oswald-Regular.ttf"),
    ("Oswald", 500, "Oswald-Medium.ttf"),
    ("Oswald", 600, "Oswald-SemiBold.ttf"),
    ("Oswald", 700, "Oswald-Bold.ttf"),
    ("IBM Plex Sans", 400, "IBMPlexSans-Regular.ttf"),
    ("IBM Plex Sans", 500, "IBMPlexSans-Medium.ttf"),
    ("IBM Plex Sans", 600, "IBMPlexSans-SemiBold.ttf"),
    ("IBM Plex Sans", 700, "IBMPlexSans-Bold.ttf"),
    ("IBM Plex Mono", 400, "IBMPlexMono-Regular.ttf"),
    ("IBM Plex Mono", 500, "IBMPlexMono-Medium.ttf"),
    ("IBM Plex Mono", 600, "IBMPlexMono-SemiBold.ttf"),
    ("Noto Sans JP", 700, "NotoSansJP-Bold.otf"),
)


def _font_faces() -> str:
    rules = []
    for family, weight, name in _FACES:
        kind = "opentype" if name.endswith(".otf") else "truetype"
        rules.append(
            f"@font-face{{font-family:'{family}';font-weight:{weight};font-style:normal;"
            f"font-display:block;src:url('fonts/{name}') format('{kind}')}}")
    return "\n".join(rules)


# アプリの全画面へ写す。参照構図（1440x900）の拡大はWebView2のページ倍率が担う
# （host）。CSSのzoomで拡大すると、Designがポインタ位置から置くメニューや
# ツールチップの座標系がずれるため使わない。
_FIT = """<style>
html,body{margin:0;height:100%;overflow:hidden;background:oklch(0.13 0.004 264)}
body{display:flex;justify-content:center;align-items:flex-start}
</style>"""


PATCHES: tuple[Patch, ...] = (
    Patch(
        name="debug-bands-off",
        why="STATE JUMP と NEEDS WIRING はDesign側のデバッグ用の帯。Designの props の既定を"
            "切るだけで、テンプレートの `sc-if debugOn` が帯ごと出さなくなる。",
        old="&quot;debugTools&quot;:{&quot;editor&quot;:&quot;boolean&quot;,&quot;default&quot;:true,",
        new="&quot;debugTools&quot;:{&quot;editor&quot;:&quot;boolean&quot;,&quot;default&quot;:false,",
    ),
    Patch(
        name="fonts-local-preconnect",
        why="書体をネットから読まない（オフラインでも同じ書体で出す）。",
        old='<link rel="preconnect" href="https://fonts.googleapis.com">',
        new="",
    ),
    Patch(
        name="fonts-local",
        why="Google Fonts の読み込みを、同梱書体の @font-face（同じファミリ名）へ置き換える。",
        old='<link href="https://fonts.googleapis.com/css2?family=Anton&amp;family=Oswald:'
            'wght@400;500;600;700&amp;family=IBM+Plex+Mono:wght@400;500;600&amp;family='
            'IBM+Plex+Sans:wght@400;500;600;700&amp;family=Noto+Sans+JP:wght@500;700;900'
            '&amp;display=swap" rel="stylesheet">',
        new="<style>\n" + _font_faces() + "\n</style>",
    ),
    Patch(
        name="lss-runtime",
        why="同梱の React、ReactDOM、LSS のランタイムを順に読み込む。",
        old='<script src="./support.js"></script>',
        new='<script>window.__lssPush=function(s){window.__LSS_INITIAL__=s};</script>\n'
            + "\n".join(f'<script src="./{name}"></script>' for name, _digest in _REACT)
            + '\n<script src="./runtime.js"></script>',
    ),
    Patch(
        name="hide-template-until-mount",
        why="ランタイムの起動前に、バインディングを含むテンプレートが表示されるのを防ぐ。",
        old="<x-dc>",
        new="<x-dc hidden>",
    ),
    Patch(
        name="backend-bridge",
        why="デモのデータ・時間・操作を、バックエンドの事実と操作へ差し替える（bridge.js）。"
            "コンポーネントの直後に、同じ名前の派生クラスとして差し込む。",
        old="\n}\n</script>\n</body>",
        new="\n}\n__BRIDGE__\n</script>\n</body>",
    ),
    Patch(
        name="frame-fills-display",
        why="Designのフレームは1440x900固定で、16:9の実画面では左右に帯が出る。フレームを"
            "画面いっぱい（最小は参照構図）にし、差はDesign自身のflex（一覧バンドと余白）が"
            "受ける（凍結ハンドオフ §6.1: 縦横比差は非意味的な余白と一覧バンドが吸収する）。",
        old='<div style="width:1440px;height:900px;overflow:hidden;position:relative;',
        new='<div data-lss-frame="1" style="width:100vw;height:100vh;min-width:1440px;'
            'min-height:900px;overflow:hidden;position:relative;',
    ),
    Patch(
        name="canvas-background",
        why="キャンバスの灰色の地は編集画面のもの。アプリでは構図の外もDesignの背景色にする。",
        old="body{margin:0;background:oklch(0.94 0.004 264)}",
        new="body{margin:0;background:oklch(0.13 0.004 264)}",
    ),
)


class PatchError(RuntimeError):
    pass


def apply(source: str, patches: tuple[Patch, ...] = PATCHES, *,
          initial: dict | None = None) -> str:
    """差し替えを当てる。どれかが原本にちょうど1回無ければ失敗する。

    `initial` はバックエンドの最初の事実。最初の描画からデモではなく実データを出す。
    """
    out = source
    for patch in patches:
        count = out.count(patch.old)
        if count != 1:
            raise PatchError(
                f"差し替え {patch.name!r} の対象が原本に {count} 回ある（1回でなければならない）。"
                f" Designが書き出し直された可能性がある。理由: {patch.why}")
        out = out.replace(patch.old, patch.new)
    bridge = BRIDGE.read_text(encoding="utf-8")
    if "</script" in bridge.lower():
        raise PatchError("bridge.js は </script を含められない")
    out = out.replace("__BRIDGE__", bridge)
    head = _FIT
    if initial is not None:
        data = json.dumps(initial, ensure_ascii=False).replace("<", "\\u003c")
        head += f"\n<script>window.__LSS_INITIAL__={data};</script>"
    marker = "</head>"
    if out.count(marker) != 1:
        raise PatchError("原本の </head> が1回でない")
    return out.replace(marker, head + "\n" + marker)


def _make_writable(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass


def _write(target: Path, text: str) -> None:
    # 配備された同梱物は読み取り専用で、以前の版はその属性ごと写していた（copy2）。前回書いた物が読み取り専用でも
    # 上書きできるよう、書く名前の物だけ属性を外す（c92defa は2回目の起動が PermissionError で止まった）。
    if target.exists():
        _make_writable(target)
    target.write_text(text, encoding="utf-8")


def _copy(source: Path, target: Path) -> None:
    """中身だけを写す。同梱物の読み取り専用属性を、書き直す領域へ持ち込まない。"""
    if target.exists():
        _make_writable(target)
    shutil.copyfile(source, target)


def build(destination: Path, *, initial: dict | None = None) -> Path:
    """アプリで配るページ一式を `destination` へ書き出し、入口のパスを返す。"""
    for name, digest in _REACT:
        if sha384((WEB_DIR / name).read_bytes()).hexdigest() != digest:
            raise PatchError(f"Bundled React integrity check failed for {name}")
    destination.mkdir(parents=True, exist_ok=True)
    page = apply(CANVAS.read_text(encoding="utf-8"), initial=initial)
    entry = destination / "index.html"
    _write(entry, page)
    _copy(RUNTIME, destination / "runtime.js")
    for name, _digest in _REACT:
        _copy(WEB_DIR / name, destination / name)
    fonts = destination / "fonts"
    fonts.mkdir(exist_ok=True)
    for _family, _weight, name in _FACES:
        _copy(FONT_DIR / name, fonts / name)
    # System Assets（Claude Design）: トレイのメニューと、ページで使うカーソル。
    menu = TRAY_MENU.read_text(encoding="utf-8").replace("/*__FONTS__*/", _font_faces())
    _write(destination / "tray_menu.html", menu)
    system = destination / "system"
    system.mkdir(exist_ok=True)
    for cursor in SYSTEM_ASSETS.glob("cursor-*.png"):
        _copy(cursor, system / cursor.name)
    return entry
