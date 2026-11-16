import hashlib
import json
import subprocess

import pytest

from lss.bgm import frame_text


def image(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(b"fake-png-for-provider-test")
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enable_windows(tmp_path, monkeypatch):
    script = tmp_path / "provider.ps1"
    script.write_text("provider test script", encoding="utf-8")
    monkeypatch.setattr(frame_text, "_IS_WINDOWS", True)
    monkeypatch.setattr(frame_text, "_SCRIPT", script)
    return script


def runtime_payload(path, **overrides):
    payload = {"status": "OK", "source_sha256": digest(path),
               "powershell_version": "5.1.26100.9711", "width": 1280,
               "height": 720, "recognized_text": "仮の観測文字", "lines": ["仮の観測文字"],
               "raw_lines": [{"text": "仮の観測文字", "words": []}]}
    payload.update(overrides)
    return payload


def test_observe_returns_uncertain_raw_ocr_observation(tmp_path, monkeypatch):
    path = image(tmp_path)
    enable_windows(tmp_path, monkeypatch)

    def runner(args, timeout):
        assert args[0] == str(frame_text._POWERSHELL)
        assert args[args.index("-Language") + 1] == "ja-JP"
        assert timeout <= 60
        payload = runtime_payload(path)
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    result = frame_text.observe(path, digest(path), _runner=runner)
    assert result["status"] == "OK"
    assert result["evidence_kind"] == "uncertain_ocr_observation"
    assert result["recognized_text"] == "仮の観測文字"
    assert result["source_sha256"] == digest(path)
    assert result["width"] == 1280


def test_observe_rejects_wrong_expected_hash(tmp_path):
    path = image(tmp_path)
    with pytest.raises(frame_text.FrameTextIntegrityError):
        frame_text.observe(path, "0" * 64)


def test_observe_detects_source_tampering_during_run(tmp_path, monkeypatch):
    path = image(tmp_path)
    original = digest(path)
    enable_windows(tmp_path, monkeypatch)

    def runner(args, timeout):
        path.write_bytes(b"changed during OCR")
        payload = {"status": "OK", "source_sha256": original}
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    with pytest.raises(frame_text.FrameTextIntegrityError):
        frame_text.observe(path, original, _runner=runner)


def test_timeout_is_typed_unavailable(tmp_path, monkeypatch):
    path = image(tmp_path)
    enable_windows(tmp_path, monkeypatch)

    def runner(args, timeout):
        raise subprocess.TimeoutExpired(args, timeout)

    result = frame_text.observe(path, digest(path), _runner=runner)
    assert result["status"] == "UNAVAILABLE"
    assert result["reason"] == "timeout"


def test_unsupported_host_and_language_are_typed_unavailable(tmp_path, monkeypatch):
    path = image(tmp_path)
    monkeypatch.setattr(frame_text, "_IS_WINDOWS", False)
    assert frame_text.observe(path, digest(path))["reason"] == "unsupported_runtime"
    enable_windows(tmp_path, monkeypatch)

    def unavailable(args, timeout):
        payload = {"status": "UNAVAILABLE", "reason": "language_unavailable"}
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    assert frame_text.observe(path, digest(path), "en-US", _runner=unavailable)["reason"] == "language_unavailable"
    with pytest.raises(ValueError, match="BCP 47"):
        frame_text.observe(path, digest(path), "not a language")


@pytest.mark.parametrize("stdout", ["[]", "null", "\"scalar\""])
def test_malformed_json_shape_is_typed_unavailable(tmp_path, monkeypatch, stdout):
    path = image(tmp_path)
    enable_windows(tmp_path, monkeypatch)
    result = frame_text.observe(path, digest(path), _runner=lambda args, timeout:
                                subprocess.CompletedProcess(args, 0, stdout, ""))
    assert result["status"] == "UNAVAILABLE"
    assert result["reason"] == "invalid_runtime_output"


def test_non_success_without_source_hash_is_unavailable(tmp_path, monkeypatch):
    path = image(tmp_path)
    enable_windows(tmp_path, monkeypatch)
    def unavailable(args, timeout):
        return subprocess.CompletedProcess(args, 1, json.dumps({"status": "UNAVAILABLE",
                                           "reason": "access_denied"}), "")
    result = frame_text.observe(path, digest(path), _runner=unavailable)
    assert result["status"] == "UNAVAILABLE"
    assert result["reason"] == "access_denied"


@pytest.mark.parametrize("payload, raises", [
    ({"status": "OK"}, True),
    ({"status": "UNAVAILABLE", "source_sha256": "f" * 64}, True),
    ({"status": "OK", "powershell_version": "7.5.0"}, False),
    ({"status": "OK", "width": 0}, False),
    ({"status": "OK", "lines": "not a list"}, False),
    ({"status": "OK", "raw_lines": [{"text": "x", "words": ["bad word shape"]}]}, False),
])
def test_hash_and_success_shape_contracts(tmp_path, monkeypatch, payload, raises):
    path = image(tmp_path)
    enable_windows(tmp_path, monkeypatch)
    output = runtime_payload(path, **payload)
    if payload.get("status") == "OK" and "source_sha256" not in payload and len(payload) == 1:
        output.pop("source_sha256")
    runner = lambda args, timeout: subprocess.CompletedProcess(args, 0, json.dumps(output), "")
    if raises:
        with pytest.raises(frame_text.FrameTextIntegrityError):
            frame_text.observe(path, digest(path), _runner=runner)
    else:
        result = frame_text.observe(path, digest(path), _runner=runner)
        assert result["status"] == "UNAVAILABLE"
        assert result["reason"] == "invalid_runtime_output"


def test_source_or_script_removed_or_changed_during_run_is_integrity_error(tmp_path, monkeypatch):
    path = image(tmp_path)
    script = enable_windows(tmp_path, monkeypatch)

    def remove_source(args, timeout):
        path.unlink()
        return subprocess.CompletedProcess(args, 0, "{}", "")

    with pytest.raises(frame_text.FrameTextIntegrityError, match="source image"):
        frame_text.observe(path, digest(path), _runner=remove_source)

    path.write_bytes(b"restored image")

    def alter_script(args, timeout):
        script.write_text("changed script", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, json.dumps(runtime_payload(path)), "")

    with pytest.raises(frame_text.FrameTextIntegrityError, match="provider script"):
        frame_text.observe(path, digest(path), _runner=alter_script)
