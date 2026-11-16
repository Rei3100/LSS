"""score_candidate()向けの実推論アダプタ（T319続き・MSST経路専用）。

F1-F4は「動くか」だけを見る（lss/harness/filters.py）。ここは3指標採点(D-6②)のために
実際の分離波形を取得する経路——vendor/msstのinference.run_folder()と同じ
`config.training.target_instrument`規約でMSSTの出力ファイル名を解決し、波形として読み戻す。

現時点の候補5件(T319/T326)は全てinfer_tool=msstのため、audio-separator経路は未実装
（必要になったら追加する＝将来の投機的一般化はしない）。
"""
from __future__ import annotations

import sys
import math
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from lss.core.model_config import load_model_config
from lss.core.runtime import RuntimeRevision, RuntimeUnavailableError, capture_cuda_runtime
from lss.paths import resource_root


class InvalidSeparationOutputError(ValueError):
    """分離出力が入力と宣言済みの時間・チャンネル構成に対応していない。"""


def _runtime_adapter_paths(msst_root: Path) -> tuple[Path, Path]:
    """開発時と凍結時のどちらでも、adapterとprovider配布物全体を返す。"""
    module_source = Path(__file__)
    if not module_source.is_file():
        module_source = resource_root() / "runtime-sources" / "lss" / "harness" / "infer.py"
    return module_source, msst_root


def capture_msst_runtime(*, device_index: int = 0) -> RuntimeRevision:
    msst_root = resource_root() / "vendor" / "msst"
    return capture_cuda_runtime(
        "msst", device_index=device_index, adapter_paths=_runtime_adapter_paths(msst_root),
    )


def expected_output_layout(config_path: Path, fallback_samplerate: int) -> tuple[list[str], int]:
    """設定から推論が生成するステム名とサンプルレートを返す。"""
    config = load_model_config(config_path)
    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError(f"モデル設定のtrainingが不正です: {config_path}")
    target = training.get("target_instrument")
    stems = [target] if target else training.get("instruments", [])
    if not isinstance(stems, list) or not stems or not all(isinstance(item, str) and item for item in stems):
        raise ValueError(f"モデル設定の出力ステムが不正です: {config_path}")

    audio = config.get("audio", {})
    samplerate = audio.get("sample_rate", fallback_samplerate) if isinstance(audio, dict) else fallback_samplerate
    if not isinstance(samplerate, int) or samplerate <= 0:
        raise ValueError(f"モデル設定のsample_rateが不正です: {config_path}")
    return list(dict.fromkeys(stems)), samplerate


def resolve_target_filename(config_path: Path, stem_target: str) -> str:
    """configのtraining節からstem_targetに対応する出力ファイル名(拡張子抜き)を返す。

    vendor/msst/utils/model_utils.pyのprefer_target_instrument()と同じ規約:
    - target_instrumentが指定されていれば(num_stems=1の慣例)、それがそのままファイル名。
      例: unwa系/gaboxはtarget_instrument="other"または"Instrumental"で1ファイルのみ出力する。
    - 無ければ(num_stems>=2のduality)、instrumentsのうちstem_targetと大文字小文字無視で
      一致するものを使う。例: becruilyはinstruments=[Vocals, Instrumental]でtarget_instrument無し。
    """
    config = load_model_config(config_path)
    training = config.get("training", {})
    if isinstance(training, dict) and training.get("target_instrument"):
        return training["target_instrument"]
    stems, _ = expected_output_layout(config_path, fallback_samplerate=44100)
    for instrument in stems:
        if instrument.lower() == stem_target.lower():
            return instrument

    raise ValueError(
        f"stem_target={stem_target!r} がconfigのinstrumentsに見つからない: "
        f"{stems}（config={config_path}）"
    )


class ResidentModel:
    """1回のworker実行内で同じmodelを読み込み直さないための入れ物。

    最初の推論で ``inference.load_model`` を1回だけ呼び、以降は同じ(model, config, checkpoint, device)の
    呼び出しにだけ読み込み済みmodelを返す。引数が変わったら例外にし、別modelを黙って使わない。
    """

    def __init__(self) -> None:
        self._loaded = None
        self.load_count = 0

    def handle(self, proc_args: dict):
        from inference import load_model, _loaded_key  # type: ignore
        if self._loaded is None:
            self._loaded = load_model(proc_args)
            self.load_count += 1
        elif self._loaded["key"] != _loaded_key(proc_args):
            raise ValueError("ResidentModelは最初と同じmodel・config・checkpoint・deviceでのみ再利用できます")
        return self._loaded


def msst_separate(
    arch: str,
    mix_path: Path,
    ckpt_path: Path,
    config_path: Path,
    stem_target: str,
    store_dir: Path,
    *, progress: Callable[[dict], None] | None = None,
    load_estimate: bool = True,
    device_index: int = 0,
    runtime_observer: Callable[[RuntimeRevision], None] | None = None,
    runtime_revision: RuntimeRevision | None = None,
    resident_model: "ResidentModel | None" = None,
) -> tuple[np.ndarray, int] | None:
    """MSST経路で1曲を実分離し、stem_targetの波形と実際のサンプルレートを返す。

    proc_folderには専用の一時フォルダへコピーした1ファイルだけを渡す。
    store_dirは実行ごとに空のものを使い、過去の出力を成功結果として読まない。

    戻り値のサンプルレートはconfig.audio.sample_rate（MSSTがlibrosa.load時にリサンプルする値）
    であり、mix_pathの原ファイルのネイティブレート（Hi-Res音源は48kHz等）とは一致しないことがある
    （T319続き実測で発覚: 参照音源をネイティブレートのまま比較しSDRが全候補で異常に低い値
    (-2.6dB台で横並び)になるバグの原因だった。呼び出し側でreferenceをこのレートへ合わせる）。
    """
    target_name = resolve_target_filename(config_path, stem_target)
    store_dir.mkdir(parents=True, exist_ok=True)
    if any(store_dir.iterdir()):
        raise ValueError(f"分離作業フォルダが空ではありません: {store_dir}")

    source_info = sf.info(mix_path)
    expected, output_samplerate = expected_output_layout(config_path, source_info.samplerate)
    # MSSTのlibrosa.load/resample(fix=True)と同じ端数処理を使う。
    # stem同士が同じ長さという条件だけでは、全stemの途中切れを検出できない。
    expected_frames = math.ceil(source_info.frames * (output_samplerate / source_info.samplerate))
    expected_channels = source_info.channels
    config = load_model_config(config_path)
    audio_config = config.get("audio", {})
    if (
        expected_channels == 1 and isinstance(audio_config, dict)
        and audio_config.get("num_channels") == 2
    ):
        # providerが設定に従って行うmono複製。stereo入力のdownmixは許容しない。
        expected_channels = 2

    msst_root = resource_root() / "vendor" / "msst"
    if str(msst_root) not in sys.path:
        sys.path.insert(0, str(msst_root))
    from inference import proc_folder  # type: ignore

    runtime = runtime_revision or capture_msst_runtime(device_index=device_index)
    if runtime.provider != "msst" or runtime.device != f"cuda:{device_index}":
        raise RuntimeUnavailableError("事前確定したRuntimeRevisionがMSST実行条件と一致しません")
    if runtime_observer is not None:
        runtime_observer(runtime)

    def emit(event: dict) -> None:
        if progress is not None:
            try:
                progress(event)
            except Exception:
                # 進捗表示の失敗で推論を中断しない。
                pass

    emit({"phase": "runtime_ready", **runtime.to_dict()})

    def report(completed_bytes: int, total_bytes: int) -> None:
        emit({
            "phase": "staging_input",
            "completed_bytes": completed_bytes,
            "total_bytes": total_bytes,
        })

    # 入力複製もジョブ固有の作業領域へ置き、異常終了時は親ごと回収可能にする。
    with TemporaryDirectory(prefix=".input-", dir=store_dir) as input_dir:
        staged_input = Path(input_dir) / f"input{mix_path.suffix}"
        total_bytes = mix_path.stat().st_size
        completed_bytes = 0
        report(completed_bytes, total_bytes)
        with mix_path.open("rb") as source, staged_input.open("xb") as destination:
            while chunk := source.read(1 << 20):
                destination.write(chunk)
                completed_bytes += len(chunk)
                report(completed_bytes, total_bytes)
        proc_args = {
            "model_type": arch,
            "config_path": str(config_path),
            "start_check_point": str(ckpt_path),
            "input_folder": input_dir,
            "store_dir": str(store_dir),
            # 長い元ファイル名によるWindows MAX_PATH超過を避ける。
            "filename_template": "{instr}",
            # provider既定値の変更や外部引数でCPU実行へ逸脱させない。
            "device_ids": [device_index],
            "force_cpu": False,
        }
        if resident_model is None:
            proc_folder(proc_args)
        else:
            # 同じmodel・config・checkpointの連続呼び出しでは読み込み済みmodelを再利用する（推論は同一）。
            proc_folder(proc_args, loaded=resident_model.handle(proc_args))

    outputs: dict[str, Path] = {}
    for name in expected:
        matches = [
            p for p in store_dir.iterdir()
            if p.is_file() and p.stem == name and p.suffix.lower() in {".wav", ".flac"}
        ]
        if len(matches) != 1:
            raise FileNotFoundError(f"分離出力{name}が一意に揃っていません: {store_dir}")
        info = sf.info(matches[0])
        observed = (info.frames, info.channels, info.samplerate)
        required = (expected_frames, expected_channels, output_samplerate)
        if expected_frames <= 0 or observed != required:
            raise InvalidSeparationOutputError(
                f"分離出力{name}が入力・設定と一致しません: "
                f"frames/channels/samplerate={observed}, expected={required}"
            )
        outputs[name] = matches[0]
    if target_name not in outputs:
        raise FileNotFoundError(
            f"MSST出力に{target_name}が見つかりません: {store_dir}"
        )
    if not load_estimate:
        return None
    estimate, sr = sf.read(outputs[target_name], always_2d=True)
    return estimate.T.astype("float32"), sr
