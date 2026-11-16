"""GUIの起動。凍結Claude DesignのページをWebView2の全画面ウィンドウで出す。

表示経路の決定と実測は `docs/LOCAL_VALIDATION.md` LV-UI-001。
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from lss.ui.web.host import main as web_main

    return web_main()
