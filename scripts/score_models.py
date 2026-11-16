"""F1→F2→F3→F4→採点→採用条件のCLIパイプライン（D-6・D-9・T310 spec §6 Step3）。

実行: .venv\\Scripts\\python.exe scripts\\score_models.py [--run-f4]

既定ではF1（匿名HTTPS取得可否）のみをmodels.tsvの`状態=候補`行に対して実行し、
結果をmodels.tsvへ書き戻す。F2（手動YAML編集要否）は候補モデルの推論設定ファイルが
ローカルに存在する場合のみ判定する（未取得の候補は`needs_manual_yaml=不明`のまま残す）。
F3（ライセンス旗）は既定で何も落とさない——個人使用専用・再配布なしのため（T326本人裁定）。
再配布する場合のみ`--redistribute`で有効化する。

認証が必要、または一時的に到達不能な候補は除外せず保留にする。取得経路を実装・検証できれば
候補へ戻せるためである。

F4（実機推論確認）はモデルの実ダウンロードと分離1曲分の実行時間を伴うため、`--run-f4`を
明示指定した場合のみ実行する（ダウンロード自体はユーザー許可を要さない＝T325）。
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root


import argparse
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lss.harness.adopt import passes_adoption  # noqa: E402
from lss.harness.filters import (  # noqa: E402
    check_f1_anonymous_download,
    check_f2_needs_manual_yaml,
    check_f3_license_flag,
    check_f4_infers_on_path,
)
from lss.harness.fetch import ensure_model_files  # noqa: E402
from lss.harness.infer import msst_separate  # noqa: E402
from lss.harness.metrics import compute_decision_metrics  # noqa: E402
from lss.harness.tsv_io import load_tsv, save_tsv  # noqa: E402

MODELS_TSV = REPO_ROOT / "ledger" / "models.tsv"
TEST_SET_TSV = dev_evidence_root(REPO_ROOT) / "ledger" / "test_set.tsv"


def run_f1_f3(
    rows: list[dict],
    config_path_for: dict[str, Path] | None = None,
    redistribute: bool = False,
) -> None:
    """状態=候補の行に対しF1(匿名DL可否)・F2(config判明分のみ)・F3(ライセンス旗)を実行し、rowsをin-placeで更新する。

    redistribute=False（既定）ではF3は何も落とさない（個人使用・再配布なしのため＝T326本人裁定）。
    """
    config_path_for = config_path_for or {}

    for row in rows:
        if row["状態"] != "候補":
            continue

        url = row["distributor_url"]
        if url.startswith("http"):
            dl_ok, auth_required = check_f1_anonymous_download(url)
            row["dl_ok"] = "true" if dl_ok else "false"
            row["auth_required"] = "true" if auth_required else "false"
            if not dl_ok:
                row["状態"] = "保留"
                continue

        config_path = config_path_for.get(row["model_id"])
        if config_path is not None:
            row["needs_manual_yaml"] = "true" if check_f2_needs_manual_yaml(config_path) else "false"

        license_state = check_f3_license_flag(row["license_name"], redistribute=redistribute)
        if license_state == "保留":
            row["状態"] = "保留"


def run_f4(rows: list[dict], sample_audio: Path, only: set[str] | None = None) -> None:
    """状態=候補の行に対しF4(実機推論)を実行する。モデルの実ダウンロードを伴う（--run-f4時のみ呼ばれる）。

    audio-separatorはmodel_id（台帳の内部識別子）ではなくdistributor_urlのファイル名そのものを
    model_filenameとして要求するため、infer_tool=audio-separatorの行はURL末尾を渡す
    （T319続き・実測で発覚: model_idをそのまま渡すとValueErrorで即失敗する）。
    """
    for row in rows:
        if row["状態"] != "候補":
            continue
        if only and row["model_id"] not in only:
            continue
        if row["infer_tool"] == "audio-separator":
            model_ref = row["distributor_url"].rsplit("/", 1)[-1]
        else:
            model_ref = row["model_id"]

        ckpt_path = config_path = None
        if row["infer_tool"] == "msst":
            ckpt_url, config_url = row["distributor_url"], row.get("config_url", "")
            if not (ckpt_url.startswith("http") and config_url.startswith("http")):
                print(f"F4スキップ(URL不足): {row['model_id']}")
                continue
            ckpt_path, config_path = ensure_model_files(row["model_id"], ckpt_url, config_url)

        ok = check_f4_infers_on_path(
            model_ref,
            row["infer_tool"],
            sample_audio,
            arch=row["arch"],
            ckpt_path=ckpt_path,
            config_path=config_path,
        )
        if not ok:
            row["状態"] = "除外"


def score_candidate(model_id: str, test_songs: list[dict], separate_fn, device: str = "cpu") -> dict[str, float]:
    """試験曲全曲で3決定変数(+記録専用2つ)を算出し平均を返す。

    separate_fn(model_id, mix_path) -> (np.ndarray, samplerate)（推論結果waveform, shape
    (channels, samples)・実際に書き出されたサンプルレート）を呼び出し側から注入する
    （実推論呼び出しをテスト時にモックできるようにするため）。
    """
    import librosa
    import numpy as np
    import soundfile as sf

    accum: dict[str, list[float]] = {"sdr": [], "l1_freq": [], "aura_mrstft": [], "bleedless": [], "fullness": []}
    for song in test_songs:
        reference, ref_sr = sf.read(song["reference_path"], always_2d=True)
        reference = reference.T.astype("float32")
        estimate, est_sr = separate_fn(model_id, Path(song["mix_path"]))
        # 参照音源はネイティブレート(Hi-Res音源は48kHz等)のまま読むが、MSST出力は
        # config.audio.sample_rate(通常44100)固定。レートを揃えずサンプル数だけで揃えると
        # 実際には別の時間長を比較することになり、SDRが全候補で異常に低くなる実測バグがあった
        # (T319続き発見)。resampleでレートを一致させてから長さを揃える。
        if ref_sr != est_sr:
            reference = librosa.resample(reference, orig_sr=ref_sr, target_sr=est_sr)
        if reference.shape[-1] != estimate.shape[-1]:
            raise ValueError(
                "評価対象のtimelineが一致しません: "
                f"reference_frames={reference.shape[-1]}, estimate_frames={estimate.shape[-1]}, "
                f"samplerate={est_sr}, mix={song['mix_path']}"
            )
        result = compute_decision_metrics(reference, estimate, device=device)
        for key, value in result.items():
            accum[key].append(value)

    return {key: float(sum(values) / len(values)) for key, values in accum.items() if values}


def _audio_seconds(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return info.frames / info.samplerate


def run_scoring(rows: list[dict], test_songs: list[dict], only: set[str] | None = None) -> None:
    """状態=候補の行に対しscore_candidate()で試験曲全曲の3決定変数を実測しrowsへ書き戻す。

    MSST経路専用（T319/T326時点の候補は全てinfer_tool=msst）。RTFはこの全曲実測から
    算出し直す（F4の1曲probeより代表性が高いため上書きする）。
    """
    total_audio_seconds = sum(_audio_seconds(song["mix_path"]) for song in test_songs)

    for row in rows:
        if row["状態"] != "候補":
            continue
        if only and row["model_id"] not in only:
            continue
        if row["infer_tool"] != "msst":
            print(f"採点スキップ(msst専用・未対応): {row['model_id']}")
            continue

        ckpt_path, config_path = ensure_model_files(row["model_id"], row["distributor_url"], row["config_url"])
        store_dir = REPO_ROOT / ".score_probe" / row["model_id"]

        def separate_fn(model_id: str, mix_path: Path, _row=row, _ckpt=ckpt_path, _config=config_path, _store=store_dir):
            _store.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix="score-", dir=_store) as song_store:
                return msst_separate(_row["arch"], mix_path, _ckpt, _config, "Instrumental", Path(song_store))

        start = time.time()
        result = score_candidate(row["model_id"], test_songs, separate_fn)
        elapsed = time.time() - start

        row["sdr"] = f"{result['sdr']:.4f}"
        row["l1_freq"] = f"{result['l1_freq']:.4f}"
        row["aura_mrstft"] = f"{result['aura_mrstft']:.4f}"
        row["bleedless"] = f"{result['bleedless']:.4f}"
        row["fullness"] = f"{result['fullness']:.4f}"
        row["rtf"] = f"{elapsed / total_audio_seconds:.4f}"
        row["test_set_id"] = "T319_9songs_2026-08-14"
        row["last_used_at"] = "2026-08-14"
        print(f"採点完了: {row['model_id']} sdr={result['sdr']:.2f} rtf={row['rtf']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-f4", action="store_true", help="実機推論(F4)まで実行する。実行時間が長いため既定では無効。")
    parser.add_argument(
        "--score", action="store_true", help="試験曲全曲でscore_candidate()を実行し3指標をmodels.tsvへ書き戻す。"
    )
    parser.add_argument(
        "--redistribute",
        action="store_true",
        help="モデル重みを第三者へ再配布する前提でF3(ライセンス旗)を有効にする。既定は個人使用のため無効(T326)。",
    )
    parser.add_argument(
        "--only",
        default="",
        help="F4/採点の対象をmodel_id（カンマ区切り）に限定する。大きい重みを1件ずつ実測する用。",
    )
    args = parser.parse_args()

    header, rows = load_tsv(MODELS_TSV)
    run_f1_f3(rows, redistribute=args.redistribute)

    only = {m.strip() for m in args.only.split(",") if m.strip()}

    if args.run_f4:
        _, test_songs = load_tsv(TEST_SET_TSV)
        if not test_songs:
            raise SystemExit("ledger/test_set.tsv が空です。F4には試験曲が最低1曲必要です。")
        run_f4(rows, Path(test_songs[0]["mix_path"]), only=only or None)

    if args.score:
        _, test_songs = load_tsv(TEST_SET_TSV)
        if not test_songs:
            raise SystemExit("ledger/test_set.tsv が空です。採点には試験曲が最低1曲必要です。")
        run_scoring(rows, test_songs, only=only or None)

    save_tsv(MODELS_TSV, header, rows)

    states = {}
    for row in rows:
        states[row["状態"]] = states.get(row["状態"], 0) + 1
    print(f"models.tsv更新完了: {MODELS_TSV}")
    print("状態内訳:", states)


if __name__ == "__main__":
    main()
