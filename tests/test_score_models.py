import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import score_models  # noqa: E402


def _write_tsv(path: Path, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_load_save_roundtrip(tmp_path):
    header = ["model_id", "状態"]
    rows = [{"model_id": "a", "状態": "候補"}, {"model_id": "b", "状態": "除外"}]
    path = tmp_path / "models.tsv"
    _write_tsv(path, header, rows)

    loaded_header, loaded_rows = score_models.load_tsv(path)
    assert loaded_header == header
    assert loaded_rows == rows

    score_models.save_tsv(path, header, loaded_rows)
    reloaded_header, reloaded_rows = score_models.load_tsv(path)
    assert reloaded_header == header
    assert reloaded_rows == rows


def test_run_f1_f3_holds_on_auth_required(monkeypatch):
    rows = [
        {
            "model_id": "gated_model",
            "distributor_url": "https://example.com/gated",
            "dl_ok": "不明",
            "auth_required": "不明",
            "license_name": "MIT",
            "needs_manual_yaml": "不明",
            "状態": "候補",
        }
    ]

    def fake_f1(url, timeout=10.0):
        return False, True

    monkeypatch.setattr(score_models, "check_f1_anonymous_download", fake_f1)

    score_models.run_f1_f3(rows)

    assert rows[0]["状態"] == "保留"
    assert rows[0]["dl_ok"] == "false"
    assert rows[0]["auth_required"] == "true"


def _unclear_license_rows():
    return [
        {
            "model_id": "unclear_license_model",
            "distributor_url": "https://example.com/x",
            "dl_ok": "不明",
            "auth_required": "不明",
            "license_name": "不明",
            "needs_manual_yaml": "不明",
            "状態": "候補",
        }
    ]


def test_run_f1_f3_keeps_candidate_on_unclear_license_by_default(monkeypatch):
    """既定(個人使用・再配布なし)ではライセンス不明でも候補のまま＝T326本人裁定の回帰テスト。"""
    rows = _unclear_license_rows()
    monkeypatch.setattr(score_models, "check_f1_anonymous_download", lambda url, timeout=10.0: (True, False))

    score_models.run_f1_f3(rows)

    assert rows[0]["状態"] == "候補"


def test_run_f1_f3_moves_to_hold_on_license_flag_when_redistributing(monkeypatch):
    rows = _unclear_license_rows()
    monkeypatch.setattr(score_models, "check_f1_anonymous_download", lambda url, timeout=10.0: (True, False))

    score_models.run_f1_f3(rows, redistribute=True)

    assert rows[0]["状態"] == "保留"


def test_run_f1_f3_keeps_candidate_when_all_pass(monkeypatch):
    rows = [
        {
            "model_id": "clean_model",
            "distributor_url": "https://example.com/ok",
            "dl_ok": "不明",
            "auth_required": "不明",
            "license_name": "MIT",
            "needs_manual_yaml": "不明",
            "状態": "候補",
        }
    ]

    monkeypatch.setattr(score_models, "check_f1_anonymous_download", lambda url, timeout=10.0: (True, False))

    score_models.run_f1_f3(rows)

    assert rows[0]["状態"] == "候補"
    assert rows[0]["dl_ok"] == "true"


def test_score_candidate_averages_across_songs(monkeypatch, tmp_path):
    import numpy as np
    import soundfile as sf

    sr = 44100
    ref_wave = (0.1 * np.random.randn(sr, 2)).astype("float32")

    ref_paths = []
    for i in range(2):
        p = tmp_path / f"ref_{i}.wav"
        sf.write(p, ref_wave, sr)
        ref_paths.append(p)

    test_songs = [{"reference_path": str(p), "mix_path": str(p)} for p in ref_paths]

    def fake_separate(model_id, mix_path):
        return ref_wave.T.copy(), sr

    result = score_models.score_candidate("dummy_model", test_songs, fake_separate)

    assert set(result.keys()) == {"sdr", "l1_freq", "aura_mrstft", "bleedless", "fullness"}
    assert result["sdr"] > 40.0  # sf.write/readのPCM量子化誤差はあるが、ほぼ同一信号なので高SDRになるはず


def test_score_candidate_rejects_mismatched_timeline(tmp_path):
    import numpy as np
    import soundfile as sf

    sr = 44100
    ref_wave = (0.1 * np.random.randn(sr, 2)).astype("float32")  # 1秒分
    ref_path = tmp_path / "ref.wav"
    sf.write(ref_path, ref_wave, sr)

    test_songs = [{"reference_path": str(ref_path), "mix_path": str(ref_path)}]

    def fake_separate_shorter(model_id, mix_path):
        return ref_wave.T[:, : sr // 2].copy(), sr  # estimateがreferenceより短い(0.5秒)

    import pytest

    with pytest.raises(ValueError, match="timelineが一致しません"):
        score_models.score_candidate("dummy_model", test_songs, fake_separate_shorter)


def test_score_candidate_resamples_reference_to_estimate_rate(tmp_path):
    """参照音源がestimateと異なるネイティブレート(Hi-Res 48kHz等)でもリサンプルして揃える回帰テスト。

    T319続きで発見した実バグ: レートを揃えずサンプル数だけ揃えると別の時間長を比較することになり、
    5候補全てのSDRが-2.6dB台で横並びという明らかに異常な結果になった。
    """
    import numpy as np
    import soundfile as sf

    ref_sr = 48000
    est_sr = 44100
    duration_sec = 2
    # 440Hzの純音（リサンプル後もSDRが高く保たれることを確認するため単純な信号にする）
    t = np.arange(ref_sr * duration_sec) / ref_sr
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    ref_wave = np.stack([tone, tone], axis=1)
    ref_path = tmp_path / "ref_48k.wav"
    sf.write(ref_path, ref_wave, ref_sr)

    t_est = np.arange(est_sr * duration_sec) / est_sr
    tone_est = (0.1 * np.sin(2 * np.pi * 440 * t_est)).astype("float32")
    estimate = np.stack([tone_est, tone_est], axis=0)

    test_songs = [{"reference_path": str(ref_path), "mix_path": str(ref_path)}]

    def fake_separate(model_id, mix_path):
        return estimate.copy(), est_sr

    result = score_models.score_candidate("dummy_model", test_songs, fake_separate)

    assert result["sdr"] > 20.0  # レートを揃えなければ大幅な負のSDRになっていたはず
