"""経路（`lss.sound.plan` の式）を実際の音へ計算する。

節ごとの結果は作業フォルダへfloat32のWAVで置き、同じ節は1つの処理の中で一度だけ計算する。
どの節も元の音と同じ周波数・同じ長さ・同じchannel数へ揃える（分離器が44.1kHzでも、戻して揃える）。
GPUのmodelは種類ごとにまとめて使い、使い終えたら手放す（同時に2つの大きなmodelを載せない）。
"""
from __future__ import annotations

import hashlib
from math import gcd
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from lss.sound import plan

SCNET = {
    "cache_name": "scnet_xl_ihf_4stem",
    "file_name": "model_scnet_ep_36_sdr_10.0891.ckpt",
    "url": ("https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/"
            "model_scnet_ep_36_sdr_10.0891.ckpt"),
    "size_bytes": 214063778,
    "sha256": "ac25975f0f5704f3d1a3c3c251505b7a0f417a22eafe82773440ee4f7e14b74f",
    "license": "MIT repository release (ZFTurbo/Music-Source-Separation-Training v1.0.15); trained on MUSDB18-HQ",
}
SCNET_CONFIG = {
    "cache_name": "scnet_xl_ihf_4stem",
    "file_name": "config_musdb18_scnet_xl_more_wide_v5.yaml",
    "url": ("https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/"
            "config_musdb18_scnet_xl_more_wide_v5.yaml"),
    "size_bytes": 5906,
    "sha256": "54ddcb8aeeae85c8d0e148741e7d87b0fbd38bd21cffa5d59ba124fe1e672f40",
}
SCNET_ARCH = "scnet"
SCNET_STEMS = ("drums", "bass", "other", "vocals")
VOCALS_MODEL_ID = "becruily_deux_inst"
VOCALS_ARCH = "mel_band_roformer"
_KARAOKE_REVISION = "f7849ae934209184dc288d1018cd8a76a7fc8b3c"
KARAOKE = {
    "cache_name": "becruily_karaoke_bs_roformer",
    "file_name": "bs_roformer_karaoke_frazer_becruily.ckpt",
    "url": (f"https://huggingface.co/becruily/bs-roformer-karaoke/resolve/{_KARAOKE_REVISION}/"
            "bs_roformer_karaoke_frazer_becruily.ckpt"),
    "size_bytes": 204436907,
    "sha256": "eb90ee24c1154d83fbcfd27e96182f19e061557cc6e4746953125e08c29389f9",
    "license": "unspecified on the public repository (becruily/bs-roformer-karaoke); local personal use",
}
KARAOKE_CONFIG = {
    "cache_name": "becruily_karaoke_bs_roformer",
    "file_name": "config_karaoke_frazer_becruily.yaml",
    "url": (f"https://huggingface.co/becruily/bs-roformer-karaoke/resolve/{_KARAOKE_REVISION}/"
            "config_karaoke_frazer_becruily.yaml"),
    "size_bytes": 1903,
    "sha256": "1d3b58b473025183d0d3c91e7a444cb5f1418d9f34e8f46b8d2fb10d2cc8ab34",
}
KARAOKE_ARCH = "bs_roformer"
# 男声・女声（歌声全部の中の男性・女性の歌声）。制御したデュエット（MUSDB18の歌声40組）で、言葉の分離
# （SAM "male singing" / "female singing"）より平均で約6 dB良い。交互に歌う区間は分けにくい（model card）。
_CHORUS_REVISION = "bb174dcf93b46ac283390355c5cc9f3cef248aac"
CHORUS = {
    "cache_name": "chorus_male_female_bs_roformer",
    "file_name": "model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt",
    "url": (f"https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/{_CHORUS_REVISION}/"
            "model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt"),
    "size_bytes": 527121477,
    "sha256": "123c00786bdbc6bd462dddb35cd21fd6ae99ab8319f93f63a8abc1012e593d94",
    "license": "CC BY-NC-SA 4.0 (Sucial/Chorus_Male_Female_BS_Roformer, experimental); non-commercial local use",
}
CHORUS_CONFIG = {
    "cache_name": "chorus_male_female_bs_roformer",
    "file_name": "config_chorus_male_female_bs_roformer.yaml",
    "url": (f"https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/{_CHORUS_REVISION}/"
            "config_chorus_male_female_bs_roformer.yaml"),
    "size_bytes": 1877,
    "sha256": "e952a90412ed3e9d2f5b0c5076edf3fa15956657602f1b56581a8e57bd61b199",
}
CHORUS_ARCH = "bs_roformer"
VOICES = ("male", "female")


def _hf(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


_MVSEPLESS = ("noblebarkrr/mvsepless_resources", "283025218e6447db47da1ad18df3c2f67ca8d66d")
_GILLIAAN = ("gilliaan/Stem-Separation-Models", "62fe7d8e363e7285d1d24aaaab87351ebc566bae")
# STEMの専門model（`plan.MODEL_STEMS`）。開発で使った重みとsha256が同じ公開ファイルを、初回に取得する。
# ``overrides`` は取得したconfigへ重ねる推論の設定（開発で使った値）。
MODELS = {
    "sw6": {
        "arch": "bs_roformer", "provider": "bs-roformer-sw-6stem",
        "weights": {"cache_name": "bs_roformer_sw_6stem", "file_name": "bs_6stem_fixed.ckpt",
                    "url": _hf(*_MVSEPLESS, "bs_roformer/bs_6stem_fixed.ckpt"), "size_bytes": 699412152,
                    "sha256": "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e",
                    "license": "no license stated (BS-RoFormer SW 6stem, mvsepless mirror); local personal use"},
        "config": {"cache_name": "bs_roformer_sw_6stem", "file_name": "bs_6stem_fixed_config.yaml",
                   "url": _hf(*_MVSEPLESS, "bs_roformer/bs_6stem_fixed_config.yaml"), "size_bytes": 3530,
                   "sha256": "4678db9430a87ee33e7fad199166928c9adcd322e2df1a812b4bf03726e2a48b"},
    },
    "drumsep6": {
        "arch": "mdx23c", "provider": "drumsep-mdx23c-6stem",
        "weights": {"cache_name": "drumsep_mdx23c_6stem", "file_name": "mdx23c_drumsep_6stem_aufr33_jarredou.ckpt",
                    "url": _hf(*_MVSEPLESS, "mdx23c/mdx23c_drumsep_6stem_aufr33_jarredou.ckpt"),
                    "size_bytes": 437652699,
                    "sha256": "d2a4aa53eb584d21eead358a4e66d1882ad182911be018f052b5da73be9096d0",
                    "license": "no license stated (aufr33 & jarredou DrumSep MDX23C); local personal use"},
        "config": {"cache_name": "drumsep_mdx23c_6stem",
                   "file_name": "mdx23c_drumsep_6stem_aufr33_jarredou_config.yaml",
                   "url": _hf(*_MVSEPLESS, "mdx23c/mdx23c_drumsep_6stem_aufr33_jarredou_config.yaml"),
                   "size_bytes": 1653,
                   "sha256": "bd6f471e3bfdd81ef9d9cd9f57a86e3d5e11b8941a5951db8a3c8585a6b4cfc1"},
        "overrides": {"inference": {"num_overlap": 4}},
    },
    "bowed_strings": {
        "arch": "bs_roformer", "provider": "gilliaan-bowedstrings-v2",
        "weights": {"cache_name": "gilliaan_bowedstrings_v2", "file_name": "gilliaan_bsroformer_bowedstrings_v2.ckpt",
                    "url": _hf(*_GILLIAAN, "BowedStrings/BandSplitRoformer/gilliaan_bsroformer_bowedstrings_v2.ckpt"),
                    "size_bytes": 303413749,
                    "sha256": "e49feede4125404930ea4dd2cd634b8a5850cc9e24dc0565a01ccac466a5bbff",
                    "license": "no license stated (gilliaan/Stem-Separation-Models); local personal use"},
        "config": {"cache_name": "gilliaan_bowedstrings_v2", "file_name": "gilliaan_bsroformer_bowedstrings_v2.yaml",
                   "url": _hf(*_GILLIAAN, "BowedStrings/BandSplitRoformer/gilliaan_bsroformer_bowedstrings_v2.yaml"),
                   "size_bytes": 2431,
                   "sha256": "f32c4491984d6d3656cb078626758bef30ed0367e0d19c50d96392812a6eca74"},
    },
    "mega53": {
        "arch": "bs_roformer", "provider": "mvsep-mega-53stem",
        "weights": {"cache_name": "mvsep_mega_53stem", "file_name": "bs_mega_53stem_full_mvsep.ckpt",
                    "url": _hf(*_MVSEPLESS, "bs_roformer/bs_mega_53stem_full_mvsep.ckpt"),
                    "size_bytes": 1368919887,
                    "sha256": "c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f",
                    "license": "no license stated (MVSep Mega 53stem BS-RoFormer, mvsepless mirror); local personal use"},
        "config": {"cache_name": "mvsep_mega_53stem", "file_name": "bs_mega_53stem_full_mvsep_config.yaml",
                   "url": _hf(*_MVSEPLESS, "bs_roformer/bs_mega_53stem_full_mvsep_config.yaml"),
                   "size_bytes": 4184,
                   "sha256": "7e198062a251587088adb91215a4f44ab59e67bd62fcc805cf54d6e7dfc51103"},
    },
}


def model_revision(model_id: str) -> dict:
    spec = MODELS[model_id]
    return {"arch": spec["arch"], "weights_sha256": spec["weights"]["sha256"],
            "config_sha256": spec["config"]["sha256"], "overrides": spec.get("overrides", {})}


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        out[key] = _merge(out.get(key) or {}, value) if isinstance(value, dict) else value
    return out


def model_config(model_id: str, progress: Callable[[dict], None] | None = None) -> Path:
    """取得したconfig。``overrides`` があれば、それを重ねた写しを同じ場所へ置いて返す。"""
    import yaml

    path = cached_asset(MODELS[model_id]["config"], progress)
    overrides = MODELS[model_id].get("overrides")
    if not overrides:
        return path
    from lss.harness.infer import load_model_config

    text = yaml.safe_dump(_merge(load_model_config(path), overrides), sort_keys=False, allow_unicode=True)
    derived = path.with_name(path.stem + ".lss.yaml")
    if not derived.is_file() or derived.read_text(encoding="utf-8") != text:
        temporary = derived.with_name(f".{derived.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(derived)
    return derived


# 1回の推論で持つ「出力の数×秒」の上限（Mega 53stemは約75秒ごと、6stemは約11分ごと）と、区切りの重なり。
MODEL_STEM_SECONDS = 4000.0
MODEL_BLOCK_OVERLAP = 4.0


def block_frames(stem_count: int, rate: int) -> int:
    return int(MODEL_STEM_SECONDS / max(1, stem_count) * rate)


def block_plan(frames: int, block: int, overlap: int) -> list[tuple[int, int, np.ndarray]]:
    """長さ ``frames`` を、``overlap`` だけ重ねた ``block`` ごとの区切りへ分ける。各区切りの重み（重なりは
    直線のcrossfade）の和は、どの標本でも1。"""
    if frames <= block:
        return [(0, frames, np.ones(frames, dtype=np.float32))]
    overlap = max(1, min(overlap, block // 2))
    hop = block - overlap
    starts = list(range(0, frames - overlap, hop))
    plan_ = []
    for number, start in enumerate(starts):
        stop = min(frames, start + block)
        weights = np.ones(stop - start, dtype=np.float32)
        if number > 0:
            weights[:overlap] = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        if number < len(starts) - 1 and stop - start > overlap:
            weights[-overlap:] = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)
        plan_.append((start, stop, weights))
    return plan_


class ExtractionError(RuntimeError):
    """経路を計算できない。"""


def _resample(values: np.ndarray, source: int, target: int) -> np.ndarray:
    from scipy.signal import resample_poly

    if source == target:
        return values.astype(np.float32, copy=False)
    divisor = gcd(source, target)
    return resample_poly(values, target // divisor, source // divisor, axis=0).astype(np.float32)


def _fit(values: np.ndarray, frames: int, channels: int) -> np.ndarray:
    """長さとchannel数を元の音へ揃える（端数の数sampleだけを詰め・削る）。"""
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[1] != channels:
        values = values.mean(axis=1, keepdims=True) if channels == 1 else np.repeat(values[:, :1], channels, axis=1)
    if len(values) < frames:
        values = np.pad(values, ((0, frames - len(values)), (0, 0)))
    values = values[:frames]
    if not np.isfinite(values).all():
        raise ExtractionError("経路の途中の音に有限でない値があります")
    return np.ascontiguousarray(values, dtype=np.float32)


def _key(node: dict) -> str:
    return hashlib.sha256(plan.canonical(node).encode("utf-8")).hexdigest()[:20]


def cached_asset(spec: dict, progress: Callable[[dict], None] | None = None) -> Path:
    from lss.core import models_store
    from lss.harness import fetch

    path = fetch.model_cache_dir() / spec["cache_name"] / spec["file_name"]
    if fetch.is_cached(path, spec["size_bytes"], spec["sha256"]):
        return path
    if path.is_file():
        path.unlink()
    models_store.ensure_cache_headroom(spec["size_bytes"])

    def report(completed: int, total: int | None) -> None:
        if progress is not None:
            progress({"phase": "downloading_model", "model_id": spec["cache_name"], "asset": spec["file_name"],
                      "completed_bytes": completed, "total_bytes": total})

    return fetch.download(spec["url"], path, expected_size=spec["size_bytes"], expected_sha256=spec["sha256"],
                          progress=report)


class Workspace:
    """1つの音源について、経路の節を計算して作業フォルダへ置く。"""

    def __init__(self, source: Path, folder: Path, *, progress: Callable[[dict], None] | None = None,
                 device: str = "cuda:0", checkpoint: Callable[[], None] | None = None):
        info = sf.info(str(source))
        self.source = Path(source)
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.rate, self.frames, self.channels = int(info.samplerate), int(info.frames), int(info.channels)
        self.device = device
        self._progress = progress
        self._checkpoint = checkpoint or (lambda: None)
        self._paths: dict[str, Path] = {}
        self._sam = None
        self._spans: dict[str, tuple[tuple[float, float], ...]] = {}
        self._whole: dict[str, np.ndarray] = {}     # 楽器: その楽器しか鳴っていない秒（親をそのまま使う）
        self._wanted: dict[tuple[str, str], set[str]] = {}   # 専門model（model, 親）ごとの要る出力
        self.used: dict[str, dict] = {}

    # ---- bookkeeping ----
    def report(self, **details) -> None:
        if self._progress is not None:
            try:
                self._progress({"phase": "separating", **details})
            except Exception:
                pass

    def _store(self, node: dict, values: np.ndarray) -> Path:
        path = self.folder / f"{_key(node)}.wav"
        temporary = path.with_name(f".{path.name}.tmp")
        sf.write(str(temporary), _fit(values, self.frames, self.channels), self.rate, subtype="FLOAT", format="WAV")
        temporary.replace(path)
        self._paths[plan.canonical(node)] = path
        return path

    def path(self, node: dict) -> Path:
        if node == plan.SOURCE:
            return self.source
        key = plan.canonical(node)
        if key not in self._paths:
            self._compute(node)
        return self._paths[key]

    def audio(self, node: dict) -> np.ndarray:
        values, _ = sf.read(str(self.path(node)), dtype="float32", always_2d=True)
        return _fit(values, self.frames, self.channels)

    # ---- engines ----
    def _compute(self, node: dict) -> None:
        kind = node["node"]
        self._checkpoint()
        if kind == "dnr":
            self._dnr()
        elif kind == "vocals":
            self._msst_vocals(node["parent"])
        elif kind == "lead":
            self._msst_lead(node["parent"])
        elif kind == "stems":
            self._msst_stems(node["parent"])
        elif kind == "voices":
            self._msst_voices(node["parent"])
        elif kind == "model":
            key = (node["model"], plan.canonical(node["parent"]))
            self._msst_model(node["model"], node["parent"], self._wanted.get(key, set()) | {node["stem"]})
        elif kind == "text":
            self._text(node)
        elif kind == "sum":
            total = np.zeros((self.frames, self.channels), dtype=np.float32)
            for item in node["items"]:
                total += self.audio(item)
            self._store(node, total)
        elif kind == "subtract":
            self._store(node, self.audio(node["base"]) - self.audio(node["minus"]))
        else:
            raise ExtractionError(f"未知の節です: {kind}")
        if plan.canonical(node) not in self._paths:
            raise ExtractionError(f"節を計算できませんでした: {plan.canonical(node)}")

    def _dnr(self) -> None:
        from lss.bgm import sfc_analysis

        weights = sfc_analysis.ensure_weights(self._progress)
        copy = self.folder / "dnr-input.wav"
        preparation = sfc_analysis.analysis_copy(self.source, copy)
        out = self.folder / "dnr"
        self.report(provider=sfc_analysis.PROVIDER)
        separated = sfc_analysis.separate_stems(copy, out, weights, device=self.device,
                                                progress=lambda event: self.report(**{k: v for k, v in event.items()
                                                                                      if k != "phase"}))
        for role in sfc_analysis.STEM_ORDER:
            values, rate = sf.read(separated["stems"][role]["path"], dtype="float32", always_2d=True)
            self._store(plan.dnr(role), _resample(values, rate, self.rate))
            Path(separated["stems"][role]["path"]).unlink(missing_ok=True)
        copy.unlink(missing_ok=True)
        self.used["dnr"] = {"provider": sfc_analysis.PROVIDER, "analysis_preparation": preparation,
                            "analyzer_id": sfc_analysis.analyzer_revision()["analyzer_id"]}

    def _msst(self, arch: str, ckpt: Path, config: Path, target: str, parent: dict | Path,
              keep: set[str] | None = None) -> dict[str, np.ndarray]:
        """``keep`` があれば、その名前の出力だけを読む（他の出力は作業フォルダと一緒に消える）。"""
        from lss.core.scratch import ScratchDirectory
        from lss.harness import infer, vram

        source = parent if isinstance(parent, Path) else self.path(parent)
        with ScratchDirectory(prefix="msst-", dir=self.folder, purpose="separation_inference") as store:
            infer.msst_separate(arch, source, ckpt, config, target, Path(store),
                                progress=lambda event: self.report(**{k: v for k, v in event.items() if k != "phase"}),
                                load_estimate=False)
            stems = {}
            for item in Path(store).iterdir():
                if (item.is_file() and item.suffix.lower() in {".wav", ".flac"}
                        and (keep is None or item.stem.lower() in keep)):
                    values, rate = sf.read(str(item), dtype="float32", always_2d=True)
                    stems[item.stem.lower()] = _resample(values, rate, self.rate)
        try:
            vram.release_unused_vram()
        except Exception:
            pass
        return stems

    def _msst_vocals(self, parent: dict) -> None:
        from lss.core import catalog, models_store

        row = catalog.get_model(VOCALS_MODEL_ID)
        ckpt, config = models_store.ensure_model_cached(
            row["model_id"], row["distributor_url"], row["config_url"], int(row["size_bytes"]),
            models_tsv_path=models_store.MODELS_TSV, progress=self._progress)
        self.report(provider=VOCALS_MODEL_ID)
        stems = self._msst(VOCALS_ARCH, ckpt, config, "Vocals", parent)
        if "vocals" not in stems or "instrumental" not in stems:
            raise ExtractionError(f"歌声と伴奏の分離出力が揃っていません: {sorted(stems)}")
        self._store(plan.vocals("vocals", parent), stems["vocals"])
        self._store(plan.vocals("instrumental", parent), stems["instrumental"])
        self.used["vocals"] = {"model_id": VOCALS_MODEL_ID, "model_sha256": row.get("sha256", ""),
                               "config_sha256": row.get("config_sha256", "")}

    def _msst_lead(self, parent: dict) -> None:
        ckpt = cached_asset(KARAOKE, self._progress)
        config = cached_asset(KARAOKE_CONFIG, self._progress)
        self.report(provider="becruily-karaoke")
        stems = self._msst(KARAOKE_ARCH, ckpt, config, "Vocals", parent)
        if "vocals" not in stems:
            raise ExtractionError(f"リードボーカルの分離出力がありません: {sorted(stems)}")
        self._store(plan.lead(parent), stems["vocals"])
        self.used["lead"] = {"weights_sha256": KARAOKE["sha256"], "config_sha256": KARAOKE_CONFIG["sha256"]}

    def _msst_stems(self, parent: dict) -> None:
        ckpt = cached_asset(SCNET, self._progress)
        config = cached_asset(SCNET_CONFIG, self._progress)
        self.report(provider="scnet-xl-ihf")
        stems = self._msst(SCNET_ARCH, ckpt, config, "vocals", parent)
        missing = [name for name in SCNET_STEMS if name not in stems]
        if missing:
            raise ExtractionError(f"4分類の出力が揃っていません: {missing}")
        for name in SCNET_STEMS:
            self._store(plan.stems(name, parent), stems[name])
        self.used["stems"] = {"weights_sha256": SCNET["sha256"], "config_sha256": SCNET_CONFIG["sha256"]}

    def _msst_voices(self, parent: dict) -> None:
        ckpt = cached_asset(CHORUS, self._progress)
        config = cached_asset(CHORUS_CONFIG, self._progress)
        self.report(provider="chorus-male-female")
        stems = self._msst(CHORUS_ARCH, ckpt, config, "male", parent)
        missing = [name for name in VOICES if name not in stems]
        if missing:
            raise ExtractionError(f"男声・女声の出力が揃っていません: {missing}")
        for name in VOICES:
            self._store(plan.voices(name, parent), stems[name])
        self.used["voices"] = {"weights_sha256": CHORUS["sha256"], "config_sha256": CHORUS_CONFIG["sha256"]}

    def _msst_model(self, model_id: str, parent: dict, wanted: set[str]) -> None:
        """専門modelを1回だけ推論し、要る出力（``wanted``。`prepare` が全経路から集める）を節として置く。

        出力の数×長さが大きいmodel（Mega 53stemで1話）は全出力を一度に持てないので、時間の区切り
        （``block_seconds``）ごとに推論し、重ねた ``MODEL_BLOCK_OVERLAP`` 秒を直線のcrossfadeでつなぐ。
        区切りの中では全出力が同じ推論なので、出力どうしの和は区切りごとにも同じ推論の和になる。"""
        spec = MODELS[model_id]
        ckpt = cached_asset(spec["weights"], self._progress)
        config = model_config(model_id, self._progress)
        self.report(provider=spec["provider"])
        names = plan.MODEL_STEMS[model_id]
        keep = {name.lower() for name in names if name in wanted}
        block = block_frames(len(names), self.rate)
        if self.frames <= block:
            stems = self._msst(spec["arch"], ckpt, config, names[0], parent, keep)
        else:
            stems = self._msst_blocks(spec["arch"], ckpt, config, names[0], parent, keep, block)
        missing = [name for name in names if name.lower() in keep and name.lower() not in stems]
        if missing:
            raise ExtractionError(f"{model_id} の出力が揃っていません: {missing}")
        for name in names:
            if name.lower() in keep:
                self._store(plan.model(model_id, name, parent), stems[name.lower()])
        self.used.setdefault("models", {})[model_id] = model_revision(model_id)

    def _msst_blocks(self, arch: str, ckpt: Path, config: Path, target: str, parent: dict, keep: set[str],
                     block: int) -> dict[str, np.ndarray]:
        source = self.audio(parent)
        overlap = int(MODEL_BLOCK_OVERLAP * self.rate)
        out = {name: np.zeros((self.frames, self.channels), dtype=np.float32) for name in keep}
        for start, stop, weights in block_plan(self.frames, block, overlap):
            self._checkpoint()
            piece = self.folder / f".block-{start}.wav"
            sf.write(str(piece), source[start:stop], self.rate, subtype="FLOAT", format="WAV")
            try:
                stems = self._msst(arch, ckpt, config, target, piece, keep)
            finally:
                piece.unlink(missing_ok=True)
            for name in keep:
                if name in stems:
                    out[name][start:stop] += _fit(stems[name], stop - start, self.channels) * weights[:, None]
        return out

    def _separator(self):
        from lss.sound import sam_audio

        if self._sam is None:
            paths = sam_audio.ensure_weights(self._progress)
            self.report(provider=sam_audio.PROVIDER, loading=True)
            self._sam = sam_audio.Separator(paths, device=self.device)
            self.used["text"] = {"engine_id": sam_audio.engine_revision()["engine_id"]}
        return self._sam

    def release_text_engine(self) -> None:
        if self._sam is not None:
            self._sam.close()
            self._sam = None

    def _text(self, node: dict) -> None:
        from lss.sound import sam_audio

        parent = self.audio(node["parent"])
        separator = self._separator()
        prompt = node["prompt"]
        if float(np.abs(parent).max(initial=0.0)) == 0.0:
            self._store(node, np.zeros_like(parent))
            return
        spans = self._spans.get(plan.canonical(node), ()) if node.get("anchors") == "frames" else ()
        target, residual = separator.estimate(
            parent.mean(axis=1), self.rate, prompt, spans=spans,
            progress=lambda done, total: self.report(provider=sam_audio.PROVIDER, prompt=prompt,
                                                     completed_blocks=done, total_blocks=total))
        if node.get("instrument"):
            from lss.sound import verify

            projected = sam_audio.project(parent, target, residual, device=self.device, rate=self.rate,
                                          floor=verify.INSTRUMENT_MASK_FLOOR,
                                          whole_seconds=self._whole.get(plan.canonical(node)))
        else:
            projected = sam_audio.project(parent, target, residual, device=self.device, rate=self.rate)
        self._store(node, projected)

    def instrument_tags(self, nodes: list[dict]) -> dict[str, np.ndarray]:
        """楽器の言葉の分離の親（その他）のタグ（秒, 527）。タグ付けは使い終えたら手放す。"""
        from lss.bgm import event_tagging
        from lss.sound import verify

        parents = {plan.canonical(node["parent"]): node["parent"] for node in nodes}
        if not parents:
            return {}
        tagger = verify.Tagger.load(event_tagging.ensure_weights(self._progress), device=self.device)
        try:
            return {key: tagger.seconds(self.audio(parent), self.rate) for key, parent in parents.items()}
        finally:
            tagger.close()

    def frame_probabilities(self, items: list[tuple[dict, str]]) -> dict[tuple[str, str], "np.ndarray"]:
        """(節, 言葉) ごとの、40 msごとの「その言葉の音が鳴っている」確率。検出器は使い終えたら手放す。"""
        from lss.sound import pe_frame

        out: dict[tuple[str, str], np.ndarray] = {}
        wanted: dict[str, list[str]] = {}
        nodes: dict[str, dict] = {}
        for node, prompt in items:
            key = plan.canonical(node)
            nodes[key] = node
            if prompt not in wanted.setdefault(key, []):
                wanted[key].append(prompt)
        if not wanted:
            return out
        self.report(provider=pe_frame.PROVIDER, loading=True)
        detector = pe_frame.Detector(pe_frame.ensure_weights(self._progress), device=self.device)
        self.used["frames"] = {"engine_id": pe_frame.engine_revision()["engine_id"]}
        try:
            for key, prompts in wanted.items():
                self._checkpoint()
                audio = self.audio(nodes[key])
                if float(np.abs(audio).max(initial=0.0)) == 0.0:
                    probabilities = np.zeros((len(prompts), max(1, int(np.ceil(
                        len(audio) / self.rate * pe_frame.FRAMES_PER_SECOND)))), dtype=np.float32)
                else:
                    self.report(provider=pe_frame.PROVIDER, prompts=prompts)
                    probabilities = detector.frame_probabilities(audio, self.rate, prompts)
                for prompt, row in zip(prompts, probabilities):
                    out[(key, prompt)] = row
        finally:
            detector.close()
        return out

    def prepare(self, expressions: list[dict]) -> None:
        """式に要る節を、modelの種類ごとにまとめて計算する。同時に2つの大きなmodelを載せない:
        専門modelの節 → 楽器の親のタグ（その楽器しか鳴っていない秒）→ 言葉の分離に渡す時刻（検出器）→
        言葉の分離（SAM）の順。"""
        from lss.sound import verify

        every = []
        for expression in expressions:
            for node in plan.nodes(expression):
                if all(plan.canonical(node) != plan.canonical(seen) for seen in every):
                    every.append(node)
        for node in every:
            if node["node"] == "model":
                self._wanted.setdefault((node["model"], plan.canonical(node["parent"])), set()).add(node["stem"])
        order = {"dnr": 0, "vocals": 1, "lead": 2, "voices": 2, "stems": 3, "model": 3, "text": 4, "sum": 5,
                 "subtract": 5, "source": -1}
        for node in sorted(every, key=lambda item: order.get(item["node"], 6)):
            if node["node"] in {"dnr", "vocals", "lead", "voices", "stems", "model"}:
                self.path(node)
        texts = [node for node in every if node["node"] == "text" and plan.canonical(node) not in self._paths]
        instruments = [node for node in texts if node.get("instrument")
                       and plan.canonical(node) not in self._whole]
        if instruments:
            tags = self.instrument_tags(instruments)
            for node in instruments:
                _, alone = verify.instrument_seconds(tags[plan.canonical(node["parent"])], node["instrument"])
                self._whole[plan.canonical(node)] = alone
        anchored = [node for node in texts if node.get("anchors") == "frames"
                    and plan.canonical(node) not in self._spans]
        if anchored:
            # 言葉の分離の親は専門modelの節か元の音（計算済み）。検出器とSAMを同時に載せない。
            probabilities = self.frame_probabilities([(node["parent"], node["prompt"]) for node in anchored])
            from lss.sound import pe_frame

            for node in anchored:
                flags = probabilities[(plan.canonical(node["parent"]), node["prompt"])] > verify.SPAN_PROBABILITY
                self._spans[plan.canonical(node)] = tuple(pe_frame.frames_to_spans(flags))
        for node in texts:
            self.path(node)
        self.release_text_engine()
