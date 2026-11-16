"""参照録音が無くても、1話の中の主題歌（OP/ED）らしい候補を見分ける。

主題歌は取り出す必要が無い（`docs/BGM.md`「エピソード／シリーズ単位の処理」）。参照録音が
あれば照合で外れるが、無い場合は形と中身で見分ける。

1. **形**：話の冒頭20%以内か終わり15%以内にあり、60秒以上続き、台詞がほぼ乗らない
   （分離器の音楽の取り分の中央値が0.95以上）。実番組12話で、参照照合で分かっている主題歌は
   劇中で台詞の下に流れた1件と取り分の低い1件を除いて全て当たった。
2. **中身**：形だけでは、終盤の台詞の無い劇伴（corpus-c の歌の無いエンディング用劇伴など）も当たる。
   主題歌には歌があるので、形に当たった候補だけに、歌を独立したクラスとして持つBGM向け
   4分類モデル（Facing the Music）を解析器としてかけ、歌の取り分を測る
   （`singing_share`。固定27ケースで矛盾0件だった帯：0.01以下＝無し、0.05以上＝有り）。

歌があれば主題歌らしいとして候補一覧から外す。**削除はせず**、記録と音声を残す。歌が無い・
判断できない場合は劇伴の候補のまま残す（欲しい曲を永久に失うより余分な候補を選ぶ）。
"""
from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from lss.bgm.singing_share import SingingShareRecipe, classify, measure
from lss.paths import app_root

PROVIDER_ROOT = app_root() / ".providers/facing-the-music/query-bandit"
CHECKPOINT = app_root() / ".providers/facing-the-music/bandit-split.ckpt"
FADER_SOURCE = app_root() / ".providers/bandit/core/utils/audio.py"
FADER = {"window_type": "hann", "chunk_size_second": 6.0, "hop_size_second": 0.5,
         "fs": 44100, "batch_size": 4}
# 固定27ケースで矛盾0件だった帯（`scripts/probe_bgm_episode_singing.py`）。実番組を見て決め直さない。
SINGING = SingingShareRecipe(maximum_absent_share=0.01, minimum_present_share=0.05,
                             minimum_music_dbfs=-70.0)


class ThemeSongError(ValueError):
    """主題歌の見分けの宣言か材料が契約に合わない。"""


@dataclass(frozen=True)
class ThemeShapeRecipe:
    """主題歌らしい形の宣言。"""

    head_fraction: float
    tail_fraction: float
    minimum_seconds: float
    minimum_median_share: float

    def validate(self) -> None:
        for name, low, high in (("head_fraction", 0.0, 1.0), ("tail_fraction", 0.0, 1.0),
                                ("minimum_seconds", 0.0, 600.0), ("minimum_median_share", 0.0, 1.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not low <= float(value) <= high:
                raise ThemeSongError(f"{name}は{low}以上{high}以下である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SHAPE = ThemeShapeRecipe(head_fraction=0.2, tail_fraction=0.15, minimum_seconds=60.0,
                                 minimum_median_share=0.95)


def has_theme_shape(cue: dict, source_seconds: float, recipe: ThemeShapeRecipe = DEFAULT_SHAPE) -> bool:
    recipe.validate()
    if cue["seconds"] < recipe.minimum_seconds or cue["median_share"] < recipe.minimum_median_share:
        return False
    return (cue["start_seconds"] / source_seconds <= recipe.head_fraction
            or cue["end_seconds"] / source_seconds >= 1.0 - recipe.tail_fraction)


class SingingAnalyzer:
    """4分類モデルを一度だけ読み、候補の音声ごとに歌の取り分を測る（CUDA、fallbackなし）。"""

    def __init__(self):
        import torch

        from lss.bgm.provider_padding import reflect_pad_last_axis
        from lss.validation import facing_music

        if not torch.cuda.is_available():
            raise ThemeSongError("歌の解析にはCUDAが必要です。CPUへはfallbackしません")
        if not CHECKPOINT.is_file() or not PROVIDER_ROOT.is_dir() or not FADER_SOURCE.is_file():
            raise ThemeSongError("歌の解析器（Facing the Music）の配布物がありません")
        self._torch = torch
        self._model = facing_music.load_model(PROVIDER_ROOT, CHECKPOINT).to("cuda:0").eval()
        spec = importlib.util.spec_from_file_location("_lss_theme_fader", FADER_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._fader = module.OverlapAddFader(**FADER).to("cuda:0").eval()
        original = module.BaseFader.prepare

        def prepare(fader, audio):
            if not fader.fade_edge_frames or max(fader.edge_frame_pad_sizes) < audio.shape[-1]:
                return original(fader, audio)
            padded = reflect_pad_last_axis(audio, tuple(fader.edge_frame_pad_sizes))
            chunks = max(1, (max(0, padded.shape[-1] - fader.chunk_size) + fader.hop_size - 1)
                         // fader.hop_size + 1)
            total = (chunks - 1) * fader.hop_size + fader.chunk_size
            return torch.nn.functional.pad(padded, (0, total - padded.shape[-1])), chunks

        module.BaseFader.prepare = prepare

    def singing(self, audio_path: Path) -> dict:
        import soundfile as sf

        torch = self._torch
        audio, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        if rate != FADER["fs"] or audio.shape[1] != 2 or not np.isfinite(audio).all():
            raise ThemeSongError(f"44.1kHz stereoの音声が必要です: {audio_path}")
        tensor = torch.from_numpy(audio.T.copy()).unsqueeze(0).to("cuda:0")
        with torch.inference_mode():
            result = self._fader(tensor, self._model)["audio"]
        vocals = result["music_vocals"][0].cpu().numpy()
        instrumental = result["music_instrumental"][0].cpu().numpy()
        observed = measure(vocals, instrumental)
        return {**classify(observed, SINGING), "singing_share": observed.get("singing_share")}
