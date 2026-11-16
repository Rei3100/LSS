from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from lss.harness import filters


class _FakeResponse:
    def __init__(self, status_code: int, url: str):
        self.status_code = status_code
        self.url = url


def test_f1_ok_public_url(monkeypatch):
    def fake_head(url, allow_redirects=True, timeout=10.0):
        return _FakeResponse(200, url)

    monkeypatch.setattr(filters.requests, "head", fake_head)
    dl_ok, auth_required = filters.check_f1_anonymous_download("https://huggingface.co/x/resolve/main/y.ckpt")
    assert dl_ok is True
    assert auth_required is False


def test_f1_403_requires_auth(monkeypatch):
    def fake_head(url, allow_redirects=True, timeout=10.0):
        return _FakeResponse(403, url)

    monkeypatch.setattr(filters.requests, "head", fake_head)
    dl_ok, auth_required = filters.check_f1_anonymous_download("https://mvsep.com/download/x")
    assert dl_ok is False
    assert auth_required is True


def test_f1_login_redirect_requires_auth(monkeypatch):
    def fake_head(url, allow_redirects=True, timeout=10.0):
        return _FakeResponse(200, "https://example.com/login?next=/x")

    monkeypatch.setattr(filters.requests, "head", fake_head)
    dl_ok, auth_required = filters.check_f1_anonymous_download("https://example.com/x")
    assert dl_ok is False
    assert auth_required is True


def test_f1_network_error_is_not_dl_ok(monkeypatch):
    def fake_head(url, allow_redirects=True, timeout=10.0):
        raise filters.requests.RequestException("boom")

    monkeypatch.setattr(filters.requests, "head", fake_head)
    dl_ok, auth_required = filters.check_f1_anonymous_download("https://unreachable.example/x")
    assert dl_ok is False
    assert auth_required is False


def test_f2_detects_use_torch_checkpoint(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("training:\n  use_torch_checkpoint: False\n", encoding="utf-8")
    assert filters.check_f2_needs_manual_yaml(cfg) is True


def test_f2_no_manual_key(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("training:\n  batch_size: 4\n", encoding="utf-8")
    assert filters.check_f2_needs_manual_yaml(cfg) is False


@pytest.mark.parametrize(
    "license_name",
    ["", "不明", "MVSEP専用共有", "ライセンス無し", "MIT"],
)
def test_f3_license_flag_default_never_blocks(license_name):
    """既定(個人使用・再配布なし)ではF3はどのライセンス値でも候補を落とさない（T326本人裁定）。"""
    assert filters.check_f3_license_flag(license_name) == "候補"


@pytest.mark.parametrize(
    "license_name,expected",
    [
        ("", "保留"),
        ("不明", "保留"),
        ("MVSEP専用共有", "保留"),
        ("ライセンス無し", "保留"),
        ("MIT", "候補"),
    ],
)
def test_f3_license_flag_when_redistributing(license_name, expected):
    """再配布する場合のみライセンス条項の有無が採否に効く（旧仕様を明示フラグ配下に温存）。"""
    assert filters.check_f3_license_flag(license_name, redistribute=True) == expected


def test_f4_succeeds_on_first_attempt(monkeypatch, tmp_path):
    calls = []

    def fake_audio_separator(model_id, sample_audio, store_dir):
        calls.append("audio-separator")

    def fake_msst(arch, sample_audio, ckpt_path, config_path, store_dir):
        calls.append("msst")
        raise AssertionError("should not reach msst when first attempt succeeds")

    monkeypatch.setattr(filters, "_try_audio_separator", fake_audio_separator)
    monkeypatch.setattr(filters, "_try_msst", fake_msst)

    ok = filters.check_f4_infers_on_path("some_model", "audio-separator", tmp_path / "sample.wav")
    assert ok is True
    assert calls == ["audio-separator"]


def test_f4_runs_only_declared_msst_provider(monkeypatch, tmp_path):
    calls = []

    def unexpected_audio_separator(model_id, sample_audio, store_dir):
        calls.append("audio-separator")
        raise AssertionError("declared provider以外を試してはいけない")

    def succeeding_msst(arch, sample_audio, ckpt_path, config_path, store_dir):
        calls.append("msst")

    monkeypatch.setattr(filters, "_try_audio_separator", unexpected_audio_separator)
    monkeypatch.setattr(filters, "_try_msst", succeeding_msst)

    ok = filters.check_f4_infers_on_path(
        "some_model", "msst", tmp_path / "sample.wav",
        arch="mel_band_roformer", ckpt_path=tmp_path / "m.ckpt", config_path=tmp_path / "c.yaml",
    )
    assert ok is True
    assert calls == ["msst"]


def test_f4_msst_requires_ckpt_and_config(tmp_path):
    """ckpt/configを渡さないMSST呼び出しは失敗する（T326: 旧実装はこれを渡しておらず常に失敗していた）。"""
    with pytest.raises(ValueError):
        filters._try_msst("mel_band_roformer", tmp_path / "sample.wav", None, None, tmp_path / "out")


def test_audio_separator_empty_result_is_not_success(monkeypatch, tmp_path):
    seen = []

    class FakeSeparator:
        def __init__(self, **kwargs):
            seen.append(Path(kwargs["output_dir"]))

        def load_model(self, **kwargs):
            pass

        def separate(self, path):
            return []

    monkeypatch.setattr("audio_separator.separator.Separator", FakeSeparator)

    with pytest.raises(FileNotFoundError, match="分離出力がありません"):
        filters._try_audio_separator("model", tmp_path / "sample.wav", tmp_path / "out")

    assert seen == [tmp_path / "out"]


def test_audio_separator_cannot_report_source_as_output(monkeypatch, tmp_path):
    sample = tmp_path / "sample.wav"
    sf.write(sample, np.zeros((100, 2), dtype=np.float32), 44100)

    class FakeSeparator:
        def __init__(self, **kwargs):
            pass

        def load_model(self, **kwargs):
            pass

        def separate(self, path):
            return [str(sample)]

    monkeypatch.setattr("audio_separator.separator.Separator", FakeSeparator)

    with pytest.raises(ValueError, match="作業領域の外"):
        filters._try_audio_separator("model", sample, tmp_path / "out")


def test_msst_requires_fresh_nonempty_audio_output(monkeypatch, tmp_path):
    sample = tmp_path / "sample.wav"
    sf.write(sample, np.zeros((100, 2), dtype=np.float32), 44100)
    store = tmp_path / "out"

    def proc_folder(args):
        sf.write(Path(args["store_dir"]) / "Instrumental.wav", np.zeros((100, 2)), 44100)

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(proc_folder=proc_folder))

    filters._try_msst("arch", sample, tmp_path / "model.ckpt", tmp_path / "config.yaml", store)


def test_msst_without_output_is_not_success(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(proc_folder=lambda args: None))

    with pytest.raises(FileNotFoundError, match="分離出力がありません"):
        filters._try_msst(
            "arch", tmp_path / "sample.wav", tmp_path / "model.ckpt",
            tmp_path / "config.yaml", tmp_path / "out",
        )


def test_f4_both_fail_returns_false(monkeypatch, tmp_path):
    def failing_audio_separator(model_id, sample_audio, store_dir):
        raise RuntimeError("fail")

    def failing_msst(arch, sample_audio, ckpt_path, config_path, store_dir):
        raise RuntimeError("fail")

    monkeypatch.setattr(filters, "_try_audio_separator", failing_audio_separator)
    monkeypatch.setattr(filters, "_try_msst", failing_msst)

    ok = filters.check_f4_infers_on_path("some_model", "不明", tmp_path / "sample.wav")
    assert ok is False
