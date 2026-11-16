"""BGM候補の意味判定と曲の区切りに使う音響イベントのタグ付け（PANNs CNN14、AudioSet 527クラス）の製品adapter。

SFC（台詞・音楽・効果音の3分類）は、音程のある効果音・鳴き声・一部の台詞を音楽の出力へ入れることがある。
一方、4分類（BandIt）は実番組のmixでは劇伴の多くを効果音側へ置く。どちらの分離器の出力も「それが音楽か」の
根拠としては足りないので、別の学習データ（AudioSet）で学習した別familyのタグ付けを、SFCの音楽の出力へかける
（`lss.bgm.cue_semantics`）。タグの確率は「その窓の音が音楽に聞こえるか・歌か・鳴き声か」の観測で、
埋め込み（2048次元）は曲の編成・雰囲気の観測として、前後の違いから曲の替わり目を探すのに使う。

固定している実行条件:

- model実装はupstream（qiuqiangkong/audioset_tagging_cnn・panns_inference、MIT）のCnn14と同じ演算・同じ
  パラメータ名をここで書き、配布重みを **strict** に読む。STFTの基底とmel filterも重みに含まれるので、
  librosaは使わない（upstreamのmodels.pyは未使用のmatplotlibをimportするため同梱しない）。
- 重みは Zenodo 10.5281/zenodo.3987831 の ``Cnn14_mAP=0.431.pth``（CC-BY-4.0）。固定したsize・sha256と
  一致する時だけ使う。
- 入力は44.1kHz stereo。通常は従来の左右平均を32kHzへ変換する。3秒窓の左右平均が
  channel平均powerより10dB以上小さい窓だけ、左右のSTFT powerを平均してから同じlogmel/CNNへ渡す。
  信号の合成・音楽の不在・公開の許可はしない。非該当窓の観測は従来どおり。
- 窓は長さ3秒、原本の絶対時刻の1秒格子（窓 ``i`` の中心は ``i + 0.5`` 秒）。信号の外側は0。
- float32、CUDAのみ。CPUや別modelへ無言でfallbackしない。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from lss.paths import resource_root
from lss.sound import audioset

PROVIDER = "panns-cnn14-audioset"
INPUT_RATE = 44100
MODEL_RATE = 32000
CLASS_COUNT = 527
EMBEDDING_SIZE = 2048
UPSTREAM = {
    "repository": "https://github.com/qiuqiangkong/audioset_tagging_cnn",
    "package": "panns_inference 0.1.1 (models.Cnn14) / torchlibrosa 0.1.0 (Spectrogram, LogmelFilterBank)",
    "license": "MIT",
    "reimplemented": "same operations and parameter names, strict state_dict load",
}
WEIGHTS = {
    "cache_name": "panns_cnn14_audioset",
    "file_name": "Cnn14_mAP=0.431.pth",
    "url": "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
    "record": "10.5281/zenodo.3987831",
    "license": "CC-BY-4.0",
    "size_bytes": 327428481,
    "sha256": "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31",
    "md5": "541141fa2ee191a88f24a3219fff024e",
}
MODEL_CONFIG = {"sample_rate": MODEL_RATE, "window_size": 1024, "hop_size": 320, "mel_bins": 64,
                "fmin": 50, "fmax": 14000, "classes_num": CLASS_COUNT, "center": True, "pad_mode": "reflect",
                "ref": 1.0, "amin": 1e-10, "top_db": None}
INFERENCE = {"window_seconds": 3.0, "hop_seconds": 1.0, "grid": "absolute_source_time_centers_i_plus_half_zero_outside",
             "window_count": "ceil_native_duration_include_partial_v1",
             "partial_batch": "separate_forward_keep_existing_full_batches_v1",
             "downmix": "cancelled_window_stereo_power_v1", "resample": "scipy.signal.resample_poly up=320 down=441",
             "cancellation_power_ratio": .1,
             "batch_windows": 64, "precision": "float32", "outputs": ["clipwise_output", "embedding"],
             "storage": "npy float16"}
STEREO_POWER = "mean_channel_spectral_power_v1"
GUARDED_STEREO = "cancelled_window_stereo_power_v1"
CANCELLATION_POWER_RATIO = .1


def cancelled_windows(stereo):
    """Nominate only severe waveform cancellation, never musical wantedness.

    Input is (windows, 2, frames), including the actual model context. Ordinary
    uncorrelated stereo loses 3 dB on downmix; this requires at least 10 dB.
    Silence and invalid evidence cannot nominate a replacement observation.
    """
    import numpy as np
    x = np.asarray(stereo)
    if x.ndim != 3 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise EventTaggingError("Cancellation evidence needs finite stereo windows")
    power = np.mean(x.astype(np.float64)**2, axis=(1, 2))
    downmix = np.mean(x.astype(np.float64), axis=1)
    mixed_power = np.mean(downmix**2, axis=1)
    return (power > 0) & (mixed_power <= CANCELLATION_POWER_RATIO*power)
# AudioSetの並び（class_labels_indices.csv）のうち、判定で名前を使うもの。
CLASSES = {
    name: audioset.INDEX[name]
    for name in (
        "Speech", "Singing", "Choir", "Yodeling", "Chant", "Mantra", "Male singing", "Female singing",
        "Child singing", "Synthetic singing", "Rapping", "Humming", "Animal", "Music", "Vocal music",
        "A capella", "Song", "Background music", "Theme music", "Jingle (music)", "Soundtrack music",
        "Video game music", "Explosion", "Sound effect",
    )
}


class EventTaggingError(RuntimeError):
    """タグ付けの実行条件か出力が契約に合わない。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_source() -> Path:
    """凍結時はPYZ内のmoduleに物理パスが無いので、同梱した同じsourceを返す。"""
    module_source = Path(__file__)
    if not module_source.is_file():
        module_source = resource_root() / "runtime-sources" / "lss" / "bgm" / "event_tagging.py"
    return module_source


def implementation_artifacts(*, normalize_text: bool = False) -> tuple[tuple[str, str], ...]:
    """``normalize_text`` は改行の違いで変わらないRecipe識別用（解析器の版は生のbytes）。"""
    from lss.core import runtime

    return runtime._provider_artifacts((adapter_source(),), normalize_text=normalize_text)


def analyzer_revision() -> dict:
    """タグ付けの結果へ影響する選択の全て。GPUへ触れずに決まる。"""
    body = {
        "schema_version": 1, "provider": PROVIDER, "upstream": UPSTREAM,
        "weights": {key: WEIGHTS[key] for key in ("url", "record", "license", "size_bytes", "sha256")},
        "model": MODEL_CONFIG, "inference": INFERENCE, "input_rate": INPUT_RATE, "classes": CLASSES,
        "implementation": [{"path": path, "sha256": digest} for path, digest in implementation_artifacts()],
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**body, "analyzer_id": "bgm-event-tags-v1-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]}


def weights_path() -> Path:
    from lss.harness import fetch

    return fetch.model_cache_dir() / WEIGHTS["cache_name"] / WEIGHTS["file_name"]


def ensure_weights(progress: Callable[[dict], None] | None = None) -> Path:
    """固定した重みを、同じ内容のときだけ使う。無ければ固定recordから取得する。"""
    from lss.core import models_store
    from lss.harness import fetch

    path = weights_path()
    if fetch.is_cached(path, WEIGHTS["size_bytes"], WEIGHTS["sha256"]):
        return path
    if path.is_file():
        path.unlink()
    models_store.ensure_cache_headroom(WEIGHTS["size_bytes"])

    def report(completed: int, total: int | None) -> None:
        if progress is not None:
            progress({"phase": "downloading_model", "model_id": WEIGHTS["cache_name"],
                      "asset": "weight", "completed_bytes": completed, "total_bytes": total})

    return fetch.download(WEIGHTS["url"], path, expected_size=WEIGHTS["size_bytes"],
                          expected_sha256=WEIGHTS["sha256"], progress=report)


def build_model():
    """upstreamのCnn14と同じ演算・同じパラメータ名（推論だけ。dropoutとSpecAugmentは推論では働かない）。"""
    import torch
    from torch import nn
    import torch.nn.functional as F

    config = MODEL_CONFIG
    bins = config["window_size"] // 2 + 1

    class _Stft(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_real = nn.Conv1d(1, bins, config["window_size"], stride=config["hop_size"], bias=False)
            self.conv_imag = nn.Conv1d(1, bins, config["window_size"], stride=config["hop_size"], bias=False)

    class _Spectrogram(nn.Module):
        def __init__(self):
            super().__init__()
            self.stft = _Stft()

        def forward(self, x):
            x = F.pad(x[:, None, :], (config["window_size"] // 2, config["window_size"] // 2), mode="reflect")
            real = self.stft.conv_real(x)[:, None, :, :].transpose(2, 3)
            imag = self.stft.conv_imag(x)[:, None, :, :].transpose(2, 3)
            return real ** 2 + imag ** 2

    class _Logmel(nn.Module):
        def __init__(self):
            super().__init__()
            self.melW = nn.Parameter(torch.zeros(bins, config["mel_bins"]), requires_grad=False)

        def forward(self, x):
            mel = torch.matmul(x, self.melW)
            return 10.0 * torch.log10(torch.clamp(mel, min=config["amin"])) - 10.0 * float(
                __import__("math").log10(max(config["amin"], config["ref"])))

    class _ConvBlock(nn.Module):
        def __init__(self, inputs, outputs):
            super().__init__()
            self.conv1 = nn.Conv2d(inputs, outputs, (3, 3), (1, 1), (1, 1), bias=False)
            self.conv2 = nn.Conv2d(outputs, outputs, (3, 3), (1, 1), (1, 1), bias=False)
            self.bn1 = nn.BatchNorm2d(outputs)
            self.bn2 = nn.BatchNorm2d(outputs)

        def forward(self, x, pool):
            x = F.relu_(self.bn1(self.conv1(x)))
            x = F.relu_(self.bn2(self.conv2(x)))
            return F.avg_pool2d(x, kernel_size=pool)

    class Cnn14(nn.Module):
        def __init__(self):
            super().__init__()
            self.spectrogram_extractor = _Spectrogram()
            self.logmel_extractor = _Logmel()
            self.bn0 = nn.BatchNorm2d(config["mel_bins"])
            widths = (1, 64, 128, 256, 512, 1024, 2048)
            for index in range(6):
                setattr(self, f"conv_block{index + 1}", _ConvBlock(widths[index], widths[index + 1]))
            self.fc1 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE, bias=True)
            self.fc_audioset = nn.Linear(EMBEDDING_SIZE, CLASS_COUNT, bias=True)

        def forward(self, waveform):
            # Pool channel powers before log compression. Summing waveforms
            # first erases valid opposite-polarity musical components. The
            # historical (batch, frames) path remains exactly replayable.
            if waveform.ndim == 3:
                batch, channels, frames = waveform.shape
                if channels != 2:
                    raise EventTaggingError("Stereo tagging requires two channels")
                power = self.spectrogram_extractor(waveform.reshape(batch * channels, frames))
                power = power.reshape(batch, channels, *power.shape[1:]).mean(dim=1)
            else:
                power = self.spectrogram_extractor(waveform)
            x = self.logmel_extractor(power)
            x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
            for index in range(1, 6):
                x = getattr(self, f"conv_block{index}")(x, (2, 2))
            x = self.conv_block6(x, (1, 1))
            x = torch.mean(x, dim=3)
            x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
            embedding = F.relu_(self.fc1(x))
            return torch.sigmoid(self.fc_audioset(embedding)), embedding

    return Cnn14()


def load_model(weights: Path, *, device: str = "cuda:0"):
    import torch

    if not torch.cuda.is_available():
        raise EventTaggingError("タグ付けにはCUDAが必要です。CPUへはfallbackしません")
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise EventTaggingError("silent CPU fallbackを検出しました")
    return model


def model_input(signal_path: Path, *, channel_policy: str = "mean_of_channels"):
    """44.1kHz stereoの信号を、モデルの32kHz monoへ。窓の格子は原本の秒に合わせる。"""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(str(signal_path), dtype="float32", always_2d=True)
    if rate != INPUT_RATE or audio.shape[1] != 2:
        raise EventTaggingError(f"{INPUT_RATE}Hz stereoの信号が必要です: {rate}Hz {audio.shape[1]}ch")
    if not np.isfinite(audio).all():
        raise EventTaggingError("タグ付けする信号に有限でない値があります")
    if channel_policy == "mean_of_channels":
        values = resample_poly(audio.mean(axis=1), 320, 441).astype(np.float32)
    elif channel_policy == STEREO_POWER:
        values = resample_poly(audio, 320, 441, axis=0).astype(np.float32).T
    else:
        raise EventTaggingError("Unknown channel policy")
    return values, int(audio.shape[0])


def tag_signal(signal_path: Path, weights: Path, *, device: str = "cuda:0",
               progress: Callable[[dict], None] | None = None,
               channel_policy: str | None = None) -> tuple:
    """Return one observation per intersecting source second, including a partial final second."""
    import numpy as np
    import torch

    policy = channel_policy or INFERENCE["downmix"]
    mono, frames = model_input(signal_path, channel_policy="mean_of_channels" if policy == GUARDED_STEREO else policy)
    stereo_padded = None
    if policy == GUARDED_STEREO:
        stereo, _ = model_input(signal_path, channel_policy=STEREO_POWER)
    seconds = (frames + INPUT_RATE - 1) // INPUT_RATE
    size = int(INFERENCE["window_seconds"] * MODEL_RATE)
    half = size // 2
    padded = np.pad(mono, (half, half + size) if mono.ndim == 1 else ((0, 0), (half, half + size)))
    if policy == GUARDED_STEREO:
        stereo_padded = np.pad(stereo, ((0, 0), (half, half + size)))
    probabilities = np.zeros((seconds, CLASS_COUNT), dtype=np.float16)
    embeddings = np.zeros((seconds, EMBEDDING_SIZE), dtype=np.float16)
    model = load_model(weights, device=device)
    batch = INFERENCE["batch_windows"]
    full_seconds = frames // INPUT_RATE
    batches = [(start, min(full_seconds, start + batch)) for start in range(0, full_seconds, batch)]
    if full_seconds < seconds:
        batches.append((full_seconds, seconds))
    try:
        with torch.inference_mode():
            for start, stop in batches:
                indices = range(start, stop)
                # 窓 i は中心 i+0.5 秒、長さ3秒（paddedの先頭half分だけずれる）
                chunks = np.stack([padded[..., int(round((i + 0.5) * MODEL_RATE)):int(round((i + 0.5) * MODEL_RATE)) + size]
                                   for i in indices])
                clip, embedding = model(torch.from_numpy(chunks).to(device))
                probabilities[start:start + len(chunks)] = clip.float().cpu().numpy()
                embeddings[start:start + len(chunks)] = embedding.float().cpu().numpy()
                if stereo_padded is not None:
                    stereo_chunks = np.stack([stereo_padded[:, int(round((i + .5)*MODEL_RATE)):
                                              int(round((i + .5)*MODEL_RATE))+size] for i in indices])
                    selected = np.flatnonzero(cancelled_windows(stereo_chunks))
                    if len(selected):
                        rescued, rescued_embedding = model(torch.from_numpy(stereo_chunks[selected]).to(device))
                        probabilities[start+selected] = rescued.float().cpu().numpy()
                        embeddings[start+selected] = rescued_embedding.float().cpu().numpy()
                if progress is not None:
                    progress({"phase": "separating", "provider": PROVIDER,
                              "completed_blocks": stop, "total_blocks": seconds})
    finally:
        del model
        torch.cuda.empty_cache()
    if not (np.isfinite(probabilities.astype(np.float32)).all() and np.isfinite(embeddings.astype(np.float32)).all()):
        raise EventTaggingError("タグ付けの出力に有限でない値があります")
    return probabilities, embeddings


def tag_to(signal_path: Path, destination: Path, weights: Path, *, device: str = "cuda:0",
           progress: Callable[[dict], None] | None = None) -> dict:
    """信号をタグ付けし、``probabilities.npy`` と ``embeddings.npy`` を destination へ確定する。"""
    import numpy as np

    signal_path, destination = Path(signal_path).resolve(), Path(destination).resolve()
    before = _sha256(signal_path)
    probabilities, embeddings = tag_signal(signal_path, weights, device=device, progress=progress)
    if _sha256(signal_path) != before:
        raise EventTaggingError("タグ付け中に信号が変わりました")
    destination.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, value in (("probabilities", probabilities), ("embeddings", embeddings)):
        target = destination / f"{name}.npy"
        temporary = destination / f".{name}.{os.getpid()}.tmp.npy"
        try:
            np.save(temporary, value)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        written[name] = {"path": str(target), "sha256": _sha256(target), "shape": list(value.shape)}
    return {"signal_sha256": before, "windows": int(probabilities.shape[0]), "outputs": written}


def load_outputs(outputs: dict) -> tuple:
    """受領票の出力を読み、記録どおりの内容か確かめて返す。"""
    import numpy as np

    loaded = []
    for name in ("probabilities", "embeddings"):
        item = outputs[name]
        path = Path(item["path"])
        if _sha256(path) != item["sha256"]:
            raise EventTaggingError(f"タグ付けの出力が記録と一致しません: {name}")
        loaded.append(np.load(path).astype(np.float32))
    return tuple(loaded)
