"""BGM候補の歌唱回収に使う名前付き4分類（Facing the Music、BandIt split）の製品adapter。

4分類は ``music_instrumental`` / ``music_vocals`` / ``speech`` / ``sfx``。SFC（台詞・音楽・効果音の
3分類）が効果音側へ誤配分した歌唱を、効果音の出力の中から ``music_vocals`` として見つけるために使う
（`lss.bgm.vocal_recovery`）。全話へは通さず、候補区間だけを推論する。

固定している実行条件:

- model実装は公開MIT層（`vendor/facing_the_music`、query-bandit、upstreamのbytesのまま）。
  配布重みのSequential名を同じ演算へ付け替え、全パラメータと固定bufferを照合して厳密に読む。
- 重みは Zenodo 10.5281/zenodo.13327983 の ``bandit-split.ckpt``（CC-BY-NC-4.0。個人の
  ローカル利用に限る）。固定したsize・sha256と一致する時だけ使う。
- 入力は44.1kHz stereo。左右は同じ重みで別々に推論し、元の順で組み直す（joint stereoではない）。
- 6秒のchunkを **原本の絶対時刻に固定した格子**（hop ``hop_seconds``）で並べ、periodic Hann窓で
  重ね合わせる。区間はその格子のchunkだけを推論するので、区間の切り方・前後の文脈・他の区間の
  有無で内側の出力は変わらない。原本の外側は0として扱う。
- float32、CUDAのみ。CPUや別modelへ無言でfallbackしない。
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

from lss.paths import resource_root

PROVIDER = "facing-the-music-bandit-split"
SAMPLE_RATE = 44100
ROLES = ("music_instrumental", "music_vocals", "speech", "sfx")
UPSTREAM = {
    "repository": "https://github.com/kwatcharasupat/query-bandit.git",
    "commit": "79ed5bb75e5c3a40cd319d9d990cee913fc65c26",
    "license": "MIT",
    "vendored_files": [
        "core/models/e2e/bandit/bandsplit.py",
        "core/models/e2e/bandit/maskestim.py",
        "core/models/e2e/bandit/tfmodel.py",
        "core/models/e2e/bandit/utils.py",
    ],
}
WEIGHTS = {
    "cache_name": "facing_the_music_bandit_split",
    "file_name": "bandit-split.ckpt",
    "url": "https://zenodo.org/records/13327983/files/bandit-split.ckpt?download=1",
    "record": "10.5281/zenodo.13327983",
    "license": "CC-BY-NC-4.0",
    "size_bytes": 551004479,
    "sha256": "adfd67196defd0ba9bd02a3d319c398172a56a8febc7fe3ff359730453f960c5",
    "md5": "e9f9054a452cb86b30a6f705266dbd7c",
}
MODEL_CONFIG = {
    "sample_rate": SAMPLE_RATE, "n_fft": 2048, "win_length": 2048, "hop_length": 512,
    "window": "hann", "center": True, "normalized": True, "pad_mode": "constant",
    "band_type": "musical", "n_bands": 64, "emb_dim": 128, "rnn_dim": 256,
    "n_sqm_modules": 8, "rnn_type": "GRU", "bidirectional": True, "mlp_dim": 512,
    "hidden_activation": "Tanh", "complex_mask": True, "use_freq_weights": True,
    "band_feature_layout": "channel_frequency_interleaved_real_imaginary_v1",
    "stereo_policy": "shared_weights_independent_channels_preserve_order_v1",
}
CHUNK_SECONDS = 6.0


class FourStemAnalysisError(RuntimeError):
    """4分類の実行条件か出力が契約に合わない。"""


def inference_config(hop_seconds: float) -> dict:
    """出力へ影響する推論条件。hopはRecipe側で宣言し、ここへ渡す。"""
    chunk = int(round(CHUNK_SECONDS * SAMPLE_RATE))
    hop = int(round(float(hop_seconds) * SAMPLE_RATE))
    if hop <= 0 or chunk % hop or (chunk // hop) % 2:
        raise FourStemAnalysisError("hopはchunkを偶数個に割り切る長さである必要があります")
    return {"chunk_frames": chunk, "hop_frames": hop, "window": "hann_periodic",
            "grid": "absolute_source_time_zero_outside", "normalization": "window_sum",
            "precision": "float32", "channel_policy": MODEL_CONFIG["stereo_policy"],
            "batch_chunks": 4, "output_subtype": "FLOAT"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vendor_root() -> Path:
    return resource_root() / "vendor" / "facing_the_music"


def adapter_source() -> Path:
    """凍結時はPYZ内のmoduleに物理パスが無いので、同梱した同じsourceを返す。"""
    module_source = Path(__file__)
    if not module_source.is_file():
        module_source = resource_root() / "runtime-sources" / "lss" / "bgm" / "four_stem_analysis.py"
    return module_source


def implementation_artifacts(*, normalize_text: bool = False) -> tuple[tuple[str, str], ...]:
    """``normalize_text`` は改行の違いで変わらないRecipe識別用（解析器の版は生のbytes）。"""
    from lss.core import runtime

    return runtime._provider_artifacts((adapter_source(), vendor_root()), normalize_text=normalize_text)


def analyzer_revision(hop_seconds: float) -> dict:
    """解析結果へ影響する選択の全て。GPUへ触れずに決まる。"""
    body = {
        "schema_version": 1, "provider": PROVIDER, "upstream": UPSTREAM,
        "weights": {key: WEIGHTS[key] for key in ("url", "record", "license", "size_bytes", "sha256")},
        "model": MODEL_CONFIG, "inference": inference_config(hop_seconds), "roles": list(ROLES),
        "sample_rate": SAMPLE_RATE,
        "implementation": [{"path": path, "sha256": digest} for path, digest in implementation_artifacts()],
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**body, "analyzer_id": "bgm-four-stem-v1-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]}


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


def band_features(spectrogram, start: int, end: int):
    """(B,C,F,T)から配布構成の(C,F,real/imag)順の特徴を得る。"""
    import torch
    batch, channels, _, times = spectrogram.shape
    band = spectrogram[:, :, start:end, :].permute(0, 3, 1, 2).contiguous()
    return torch.view_as_real(band).reshape(batch, times, channels, -1)


def renamed_state(state: dict) -> dict:
    """配布時のSequential名を同じ演算名へ移し、同じ値の別名だけをまとめる。"""
    result = {}
    for name, value in state.items():
        name = name.removeprefix("model.")
        # 配布時はRNNとパラメータを持たないtransposeを交互にSequentialへ登録。
        # 現行公開層はRNNだけをModuleListへ登録し、forwardで同じtransposeを行う。
        match = re.match(r"^tf_model\.seqband\.(\d+)\.(.*)$", name)
        if match:
            index = int(match[1])
            if index % 2:
                raise ValueError("配布時のtranspose位置に学習パラメータが含まれています")
            name = f"tf_model.seqband.{index // 2}.{match[2]}"
        name = re.sub(r"^(band_split\.norm_fc_modules\.\d+)\.combined\.0\.", r"\1.norm.", name)
        name = re.sub(r"^(band_split\.norm_fc_modules\.\d+)\.combined\.1\.", r"\1.fc.", name)
        for old, new in (("0", "norm"), ("1.0", "hidden.0"), ("2.0", "output.0")):
            name = re.sub(r"^(mask_estim\.[A-Za-z0-9_]+\.norm_mlp\.\d+)\.combined\."
                          + re.escape(old) + r"\.", r"\1." + new + ".", name)
        if name in result:
            previous = result[name]
            if previous is value:
                continue
            import torch
            if (torch.is_tensor(previous) and torch.is_tensor(value)
                    and previous.dtype == value.dtype and previous.shape == value.shape
                    and torch.equal(previous, value)):
                continue
            raise ValueError(f"配布重みの名前変更が複数の値を同じ演算へ接続しています: {name}")
        result[name] = value
    return result


def load_model(provider_root: Path, checkpoint_path: Path):
    """PaSSTを必要としない公開層だけを読み、全パラメータと固定bufferを照合する。"""
    import torch
    import torchaudio as ta
    from torch import nn

    package = "core"
    entry = Path(provider_root).resolve() / "core/__init__.py"
    if package in sys.modules:
        if Path(sys.modules[package].__file__).resolve() != entry:
            raise ValueError("別の公開実装を同じproviderとして再利用できません")
    else:
        spec = importlib.util.spec_from_file_location(package, entry,
                    submodule_search_locations=[str(entry.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    prefix = package + ".models.e2e.bandit."
    BandSplitModule = importlib.import_module(prefix + "bandsplit").BandSplitModule
    Mask = importlib.import_module(prefix + "maskestim").OverlappingMaskEstimationModule
    SeqBand = importlib.import_module(prefix + "tfmodel").SeqBandModellingModule
    Specs = importlib.import_module(prefix + "utils").MusicalBandsplitSpecification

    class ReleasedBandSplit(BandSplitModule):
        def forward(self, x):
            # 公開MIT層のNormFCを使う。配布時のSequential構成では周波数ごとに
            # real/imagが隣り合い、古い公開層のreal群+imag群とは特徴順序が異なる。
            return torch.stack([layer(band_features(x, start, end))
                for (start, end), layer in zip(self.band_specs, self.norm_fc_modules)], dim=1)

    class FourStemBandit(nn.Module):
        def __init__(self):
            super().__init__()
            spectral = dict(n_fft=2048, win_length=2048, hop_length=512,
                            window_fn=torch.hann_window, normalized=True, center=True,
                            pad_mode="constant", onesided=True)
            self.stft = ta.transforms.Spectrogram(**spectral, power=None)
            self.istft = ta.transforms.InverseSpectrogram(**spectral)
            specs = Specs(nfft=2048, fs=44100, n_bands=64)
            bands = specs.get_band_specs()
            self.band_split = ReleasedBandSplit(band_specs=bands, emb_dim=128, in_channel=1,
                require_no_overlap=False, require_no_gap=True,
                normalize_channel_independently=False, treat_channel_as_feature=True)
            self.tf_model = SeqBand(n_modules=8, emb_dim=128, rnn_dim=256,
                                   bidirectional=True, rnn_type="GRU")
            self.mask_estim = nn.ModuleDict({role: Mask(band_specs=bands,
                freq_weights=specs.get_freq_weights(), n_freq=1025, emb_dim=128,
                mlp_dim=512, in_channel=1, hidden_activation="Tanh",
                hidden_activation_kwargs={}, complex_mask=True, use_freq_weights=True)
                for role in ROLES})

        def forward(self, audio):
            if audio.ndim != 3 or audio.shape[1] != 2:
                raise ValueError("固定stereoのチャンクが必要です")
            batch, channels, frames = audio.shape
            x = self.stft(audio.reshape(batch * channels, 1, frames))
            q = self.tf_model(self.band_split(x))
            return {role: self.istft((x * decoder(q)).reshape(x.shape), length=frames)
                    .reshape(batch, channels, frames)
                    for role, decoder in self.mask_estim.items()}

    model = FourStemBandit()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = renamed_state({key: value for key, value in checkpoint["state_dict"].items()
                           if key.startswith("model.")})
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("公開層と配布重みの演算名が不一致: "
             f"不足={sorted(set(expected) - set(state))[:5]}, 余分={sorted(set(state) - set(expected))[:5]}")
    # 窓と帯域の重みは学習対象ではない。違う構成へ重みだけを強制接続しない。
    for key, value in model.named_buffers():
        if key not in state or not torch.equal(value, state[key]):
            raise ValueError(f"公開層とチェックポイントの固定bufferが不一致: {key}")
    model.load_state_dict(state, strict=True)
    return model


def grid_chunks(first: int, last: int, config: dict) -> list[int]:
    """[first, last) の出力に寄与する、絶対格子上のchunk開始位置（負もあり得る）。"""
    chunk, hop = config["chunk_frames"], config["hop_frames"]
    low = (first - chunk) // hop + 1
    return [k * hop for k in range(low, -(-last // hop))]


class FourStemSeparator:
    """重みを一度だけ読み、区間ごとに絶対格子のchunkを推論する（CUDA、fallbackなし）。"""

    def __init__(self, weights: Path, hop_seconds: float, *, device: str = "cuda:0"):
        import torch

        if not torch.cuda.is_available():
            raise FourStemAnalysisError("4分類の解析にはCUDAが必要です。CPUへはfallbackしません")
        if not vendor_root().is_dir():
            raise FourStemAnalysisError(f"4分類の同梱実装がありません: {vendor_root()}")
        self._torch = torch
        self.device = device
        self.config = inference_config(hop_seconds)
        self.model = load_model(vendor_root(), Path(weights)).to(device).eval()
        if any(parameter.device.type != "cuda" for parameter in self.model.parameters()):
            raise FourStemAnalysisError("silent CPU fallbackを検出しました")
        chunk = self.config["chunk_frames"]
        self.window = torch.hann_window(chunk, periodic=True, dtype=torch.float32)

    def separate(self, audio, first: int, last: int, roles: Iterable[str] = ROLES) -> dict:
        """audio（(frames, 2) float32、原本の全長）の [first, last) を推論し、役割ごとの配列を返す。"""
        import numpy as np

        torch = self._torch
        roles = tuple(roles)
        if audio.ndim != 2 or audio.shape[1] != 2 or not 0 <= first < last <= audio.shape[0]:
            raise FourStemAnalysisError("原本の範囲内のstereo区間が必要です")
        chunk, batch_size = self.config["chunk_frames"], self.config["batch_chunks"]
        starts = grid_chunks(first, last, self.config)
        length = last - first
        accumulated = {role: np.zeros((length, 2), dtype=np.float64) for role in roles}
        weight = np.zeros(length, dtype=np.float64)
        window = self.window.numpy().astype(np.float64)
        for index in range(0, len(starts), batch_size):
            group = starts[index:index + batch_size]
            pieces = np.zeros((len(group), 2, chunk), dtype=np.float32)
            for row, start in enumerate(group):
                low, high = max(0, start), min(audio.shape[0], start + chunk)
                if high > low:
                    pieces[row, :, low - start:high - start] = audio[low:high].T
            with torch.inference_mode():
                output = self.model(torch.from_numpy(pieces).to(self.device))
                output = {role: output[role].float().cpu().numpy() for role in roles}
            for row, start in enumerate(group):
                low, high = max(first, start), min(last, start + chunk)
                if high <= low:
                    continue
                w = window[low - start:high - start]
                weight[low - first:high - first] += w
                for role in roles:
                    accumulated[role][low - first:high - first] += (
                        w[:, None] * output[role][row, :, low - start:high - start].T)
        if not (weight > 0).all():
            raise FourStemAnalysisError("格子のchunkが区間を覆っていません")
        result = {role: (value / weight[:, None]).astype(np.float32) for role, value in accumulated.items()}
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FourStemAnalysisError("4分類の出力に有限でない値があります")
        return result

    def close(self) -> None:
        del self.model
        self._torch.cuda.empty_cache()


def separate_spans(signal_path: Path, destination: Path, weights: Path, spans, *, hop_seconds: float,
                   roles: Iterable[str] = ("music_vocals",), device: str = "cuda:0",
                   progress: Callable[[dict], None] | None = None) -> dict:
    """解析用の信号（44.1kHz stereo）の区間 [開始frame, 終了frame) だけを推論し、区間ごとのstemを確定する。

    出力は ``<role>-<開始frame>-<終了frame>.wav``（区間の長さだけ）。全長のファイルは作らない。
    """
    import numpy as np
    import soundfile as sf

    signal_path, destination = Path(signal_path).resolve(), Path(destination).resolve()
    before = _sha256(signal_path)
    audio, rate = sf.read(str(signal_path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or audio.shape[1] != 2:
        raise FourStemAnalysisError(f"{SAMPLE_RATE}Hz stereoの解析用信号が必要です: {rate}Hz {audio.shape[1]}ch")
    if not np.isfinite(audio).all():
        raise FourStemAnalysisError("解析する信号に有限でない値があります")
    spans = sorted((int(a), int(b)) for a, b in spans)
    if not spans or any(not 0 <= a < b <= audio.shape[0] for a, b in spans) or any(
            b1 > a2 for (_, b1), (a2, _) in zip(spans, spans[1:])):
        raise FourStemAnalysisError(f"区間が不正です: {spans}")
    roles = tuple(roles)
    if not roles or len(set(roles)) != len(roles) or any(role not in ROLES for role in roles):
        raise FourStemAnalysisError(f"4分類の役割が不正です: {roles}")
    destination.mkdir(parents=True, exist_ok=True)
    separator = FourStemSeparator(weights, hop_seconds, device=device)
    written: dict[str, list] = {role: [] for role in roles}
    try:
        for done, (first, last) in enumerate(spans, 1):
            for role, value in separator.separate(audio, first, last, roles).items():
                target = destination / f"{role}-{first}-{last}.wav"
                temporary = destination / f".{role}-{first}-{last}.{os.getpid()}.tmp"
                try:
                    sf.write(str(temporary), value, SAMPLE_RATE, subtype="FLOAT", format="WAV")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                written[role].append({"span": [first, last], "path": str(target), "sha256": _sha256(target)})
            if progress is not None:
                progress({"phase": "separating", "provider": PROVIDER,
                          "completed_blocks": done, "total_blocks": len(spans)})
    finally:
        separator.close()
    if _sha256(signal_path) != before:
        raise FourStemAnalysisError("推論中に解析用の信号が変わりました")
    return {"signal_sha256": before, "frames": int(audio.shape[0]), "channels": 2,
            "sample_rate": SAMPLE_RATE, "spans": [[a, b] for a, b in spans], "stems": written}
