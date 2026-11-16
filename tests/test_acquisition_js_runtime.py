"""YouTube取得用のdenoを、署名とチェックサムを確かめてから導入する。外部へは接続しない。"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pytest

from lss.acquisition import js_runtime


def _release(version: str, exe: bytes, *, sums_digest: str | None = None) -> dict:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("deno.exe", exe)
    archive = buffer.getvalue()
    digest = hashlib.sha256(archive).hexdigest()
    base = f"https://example.test/v{version}/"
    return {
        js_runtime.RELEASE_API: json.dumps({"tag_name": f"v{version}", "assets": [
            {"name": js_runtime.ASSET, "browser_download_url": base + js_runtime.ASSET, "digest": "sha256:" + digest},
            {"name": js_runtime.SUMS, "browser_download_url": base + js_runtime.SUMS},
        ]}).encode(),
        base + js_runtime.ASSET: archive,
        base + js_runtime.SUMS: f"Hash      : {(sums_digest or digest).upper()}\n".encode(),
    }


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(js_runtime, "path_runtime", lambda: None)
    return tmp_path


GOOD_SIGNER = lambda path: ("Valid", "CN=Deno Land Inc., O=Deno Land Inc., L=San Diego, S=California, C=US")  # noqa: E731


def test_verified_runtime_is_installed_inside_the_app_state_and_kept_on_update(state):
    checked = []
    files = _release("2.9.6", b"exe-1")
    result = js_runtime.check_and_install(fetch=files.__getitem__, signer=GOOD_SIGNER,
                                          validate=lambda p, v: checked.append((p.read_bytes(), v)), now=1)
    assert result["status"] == "installed" and checked == [(b"exe-1", "2.9.6")]
    runtime = js_runtime.active()
    assert runtime.kind == "managed" and runtime.version == "2.9.6"
    assert js_runtime.root() in __import__("pathlib").Path(runtime.path).parents

    files = _release("2.9.7", b"exe-2")
    js_runtime.check_and_install(fetch=files.__getitem__, signer=GOOD_SIGNER, validate=lambda p, v: None, now=2)
    state_record = js_runtime.load_state()
    assert state_record["active"]["version"] == "2.9.7" and state_record["previous"]["version"] == "2.9.6"


@pytest.mark.parametrize("case", ["checksum", "unsigned", "other_signer", "validation"])
def test_unverifiable_runtime_is_not_activated(state, case):
    files = _release("2.9.6", b"exe", sums_digest="0" * 64 if case == "checksum" else None)
    signer = {
        "unsigned": lambda p: ("NotSigned", ""),
        "other_signer": lambda p: ("Valid", "CN=Someone Else"),
    }.get(case, GOOD_SIGNER)

    def validate(path, version):
        if case == "validation":
            raise RuntimeError("eval failed")

    result = js_runtime.check_and_install(fetch=files.__getitem__, signer=signer, validate=validate, now=1)
    assert result["status"] == "held"
    assert js_runtime.active() is None
    assert not (js_runtime.root() / "staging").exists()


def test_a_supported_runtime_on_path_is_used_without_installing(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(js_runtime, "path_runtime", lambda: js_runtime.Runtime("C:/deno/deno.exe", "2.8.1", "path"))
    assert js_runtime.ensure(fetch=lambda url: pytest.fail("should not download"), now=1) is None
    assert js_runtime.active().kind == "path"


def test_managed_runtime_is_passed_to_ytdlp(state, monkeypatch):
    from lss.acquisition.store import AuthMethod
    from lss.acquisition.ytdlp import YtDlpBackend

    monkeypatch.setattr(js_runtime, "active", lambda: js_runtime.Runtime("C:/LSS/deno.exe", "2.9.6", "managed"))
    args = YtDlpBackend(("yt-dlp",))._args(AuthMethod())
    assert args[args.index("--js-runtimes") + 1] == "deno:C:/LSS/deno.exe"


def test_windows_code_signature_is_read_by_the_os(tmp_path):
    import shutil
    import sys
    from pathlib import Path

    signed = shutil.which("powershell")
    status, subject = js_runtime.authenticode_signer(Path(signed))
    assert status == "Valid" and "Microsoft" in subject
    unsigned = tmp_path / "copy's [1].exe"
    unsigned.write_bytes(b"MZ not really signed")
    assert js_runtime.authenticode_signer(unsigned)[0] != "Valid"
