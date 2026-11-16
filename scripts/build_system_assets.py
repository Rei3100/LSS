"""Design の図形（`lss.ui.system_assets`）を WebView2 で画素に焼き、`assets/` へ置く。

Design の受け入れ基準は描画結果なので、Design が表示に使ったのと同じ Chromium 系の
描画で焼く（DevTools Protocol で1枚ずつ、透明背景・DPR 1）。生成物はリポジトリに置き、
アプリの実行時には描画しない。

    .venv\\Scripts\\python scripts\\build_system_assets.py
"""
from __future__ import annotations

import base64
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.ui import system_assets as sa  # noqa: E402

OUT = ROOT / "assets" / "system"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Cdp:
    def __init__(self, port: int) -> None:
        import websocket

        end = time.time() + 60
        while True:
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                tab = next(t for t in tabs if t.get("type") == "page")
                break
            except Exception:
                if time.time() > end:
                    raise
                time.sleep(0.3)
        self._ws = websocket.create_connection(
            tab["webSocketDebuggerUrl"], timeout=30, suppress_origin=True)
        self._id = 0

    def call(self, method: str, **params):
        self._id += 1
        self._ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            message = json.loads(self._ws.recv())
            if message.get("id") == self._id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message["result"]


def _render_all(port: int) -> dict[str, bytes]:
    cdp = _Cdp(port)
    cdp.call("Emulation.setDefaultBackgroundColorOverride", color={"r": 0, "g": 0, "b": 0, "a": 0})
    out: dict[str, bytes] = {}
    for raster in sa.rasters():
        cdp.call("Emulation.setDeviceMetricsOverride", width=raster.size, height=raster.size,
                 deviceScaleFactor=1, mobile=False)
        html = ("<html><body style='margin:0;background:transparent;overflow:hidden'>"
                + raster.svg + "</body></html>")
        cdp.call("Runtime.evaluate", expression=(
            "document.open();document.write(" + json.dumps(html) + ");document.close();"
            "new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))"),
            awaitPromise=True)
        shot = cdp.call("Page.captureScreenshot", format="png", fromSurface=True, clip={
            "x": 0, "y": 0, "width": raster.size, "height": raster.size, "scale": 1})
        out[raster.name] = base64.b64decode(shot["data"])
    return out


def _check(name: str, data: bytes, size: int) -> None:
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(data))
    if image.size != (size, size) or image.mode != "RGBA":
        raise RuntimeError(f"{name}: {image.size} {image.mode}（{size}px RGBA でない）")
    if image.getpixel((0, size - 1))[3] != 0 and not name.startswith("cursor"):
        raise RuntimeError(f"{name}: 背景が透明でない")


def main() -> int:
    import webview

    port = _free_port()
    os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = (
        f"--remote-debugging-port={port} --remote-allow-origins=* "
        "--disable-features=CalculateNativeWinOcclusion")
    result: dict = {}
    window = webview.create_window("LSS assets", html="<html></html>", x=-4000, y=0,
                                   width=320, height=320, frameless=True, transparent=True)

    def work() -> None:
        try:
            result["images"] = _render_all(port)
        except Exception as exc:  # noqa: BLE001 - 生成の失敗を呼び出し元へ返す
            result["error"] = exc
        finally:
            window.destroy()

    window.events.loaded += lambda: threading.Thread(target=work, daemon=True).start()
    webview.start(gui="edgechromium", private_mode=True)
    if "error" in result:
        raise result["error"]
    images: dict[str, bytes] = result["images"]
    sizes = {raster.name: raster.size for raster in sa.rasters()}
    for name, data in images.items():
        _check(name, data, sizes[name])

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in images.items():
        (OUT / name).write_bytes(data)
    (ROOT / "assets" / "app.ico").write_bytes(sa.ico_bytes(
        [(size, images[f"app-{size}.png"]) for size in sa.APP_ICON_SIZES]))
    for state in sa.TrayState:
        for light in (False, True):
            for frame in range(sa.tray_frames(state)):
                (OUT / sa.tray_ico_name(state, light, frame)).write_bytes(sa.ico_bytes(
                    [(size, images[sa.tray_png_name(state, size, light, frame)])
                     for size in sa.TRAY_SIZES]))
    print(f"{len(images)} images -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
