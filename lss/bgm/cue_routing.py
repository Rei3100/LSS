"""Music側の活動区間を保ち、推定の切替を区間単位に固定するDevelopment stage。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Mapping

import numpy as np

from lss.bgm.routing import DisagreementRoutingRecipe, analyze_stereo


@dataclass(frozen=True)
class CueRoutingRecipe:
    window_recipe: DisagreementRoutingRecipe = field(default_factory=DisagreementRoutingRecipe)
    context_frames: int = 0
    require_energy_increase: bool = False
    schema_version: int = 1
    method: str = "music-activity-cue-persistent-disagreement-routing"
    switch_rule: str = "any-positive-window-center-in-candidate"
    outside_activity: str = "zero-on-common-timeline"

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("未対応のcue routing schemaです")
        if (self.method != "music-activity-cue-persistent-disagreement-routing"
                or self.switch_rule != "any-positive-window-center-in-candidate"
                or self.outside_activity != "zero-on-common-timeline"):
            raise ValueError("未対応のcue routing規則です")
        if not isinstance(self.window_recipe, DisagreementRoutingRecipe):
            raise ValueError("窓ごとの固定routing Recipeが必要です")
        if type(self.context_frames) is not int or self.context_frames < 0:
            raise ValueError("context_framesは0以上の整数である必要があります")
        if type(self.require_energy_increase) is not bool:
            raise ValueError("require_energy_increaseは真偽値である必要があります")
        self.window_recipe.validate()

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "bgm-cue-route-v1-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class CueRoutingResult:
    audio: np.ndarray
    cues: tuple[dict, ...]
    recipe_id: str


def route_cues_stereo(instrumental: np.ndarray, stems: Mapping[str, np.ndarray],
                      intervals: list[tuple[int, int]], recipe: CueRoutingRecipe) -> CueRoutingResult:
    """窓の不一致がある候補は全体をMusicで保持し、歌の途中で切り替えない。

    候補は曲の意味的な境界ではない。誤ったMusic推定や単発の不一致で区間全体を
    切り替えるリスクも持つため、この規則を製品の自動選択に昇格しない。
    """
    recipe.validate()
    windows_evidence = analyze_stereo(instrumental, stems, recipe.window_recipe)
    frames = np.asarray(instrumental).shape[1]
    previous_end = 0
    for start, end in intervals:
        if (type(start) is not int or type(end) is not int or start < previous_end
                or end <= start or end > frames):
            raise ValueError("候補区間はtimeline内の昇順・非重複の整数区間が必要です")
        previous_end = end
    parent, music = np.asarray(instrumental, dtype=np.float64), np.asarray(stems["Music"], dtype=np.float64)
    output = np.zeros_like(parent)
    centers = np.array([row["center_frame"] for row in windows_evidence])
    positives = np.array([row["use_music"] for row in windows_evidence], dtype=bool)
    expanded = []
    for start, end in intervals:
        left, right = max(0, start - recipe.context_frames), min(frames, end + recipe.context_frames)
        if expanded and left <= expanded[-1][1]:
            expanded[-1][1] = max(right, expanded[-1][1])
            expanded[-1][2].append((start, end))
        else:
            expanded.append([left, right, [(start, end)]])
    cues = []
    for start, end, parents in expanded:
        count, windows = 0, 0
        parent_energy, music_energy = 0.0, 0.0
        for parent_start, parent_end in parents:
            lo, hi = np.searchsorted(centers, [parent_start, parent_end])
            count += int(positives[lo:hi].sum())
            windows += int(hi - lo)
            parent_energy += float(np.sum(parent[:, parent_start:parent_end] ** 2))
            music_energy += float(np.sum(music[:, parent_start:parent_end] ** 2))
        if not np.isfinite(parent_energy) or not np.isfinite(music_energy):
            raise ValueError("cue routingのenergy集計が有限範囲を超えました")
        energy_increased = music_energy > parent_energy
        use_music = count > 0 and (not recipe.require_energy_increase or energy_increased)
        output[:, start:end] = (music if use_music else parent)[:, start:end]
        cues.append({"start_frame": start, "end_frame": end, "use_music": use_music,
                     "positive_window_count": count, "window_count": windows,
                     "activity_intervals": parents,
                     "parent_energy": parent_energy, "music_energy": music_energy,
                     "music_energy_increased": energy_increased,
                     "evidence_role": "renderer_disagreement_not_semantic_identity"})
    return CueRoutingResult(output, tuple(cues), recipe.recipe_id)
