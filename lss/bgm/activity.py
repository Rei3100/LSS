"""music-like信号から高recall候補区間を得る版付きactivity stage。"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from lss.validation.occurrence_metrics import DetectedOccurrence


ACTIVITY_SCHEMA_VERSION = 1
ACTIVITY_METHOD = "windowed_rms_hysteresis"
RELATIVE_HYSTERESIS_METHOD = "windowed_rms_relative_hysteresis"
STEM_ENERGY_SHARE_METHOD = "stem_energy_share_hysteresis"
STEM_ENERGY_SHARE_RELATIVE_METHOD = "stem_energy_share_relative_hysteresis"
_RECIPE_KEYS = frozenset({
    "schema_version", "method", "window_frames", "hop_frames", "onset_rms", "offset_rms",
    "bridge_gap_frames", "minimum_active_frames", "padding_frames",
})
_RELATIVE_RECIPE_KEYS = frozenset({
    "schema_version", "method", "window_frames", "hop_frames", "onset_rms",
    "offset_ratio_db", "offset_floor_rms",
    "bridge_gap_frames", "minimum_active_frames", "padding_frames",
})
_STEM_ENERGY_SHARE_RECIPE_KEYS = frozenset({
    "schema_version", "method", "onset_share", "offset_share",
    "onset_music_rms", "offset_music_rms", "window_frames", "hop_frames",
    "bridge_gap_frames", "minimum_active_frames", "padding_frames",
})
_STEM_ENERGY_SHARE_RELATIVE_RECIPE_KEYS = frozenset({
    "schema_version", "method", "onset_share", "onset_music_rms",
    "offset_ratio_db", "offset_floor_rms", "window_frames", "hop_frames",
    "bridge_gap_frames", "minimum_active_frames", "padding_frames",
})


class InvalidActivityRecipeError(ValueError):
    """BGM activity Recipeが再現可能な形になっていない。"""


@dataclass(frozen=True)
class ActivityRecipe:
    window_frames: int
    hop_frames: int
    onset_rms: float
    offset_rms: float
    bridge_gap_frames: int
    minimum_active_frames: int
    padding_frames: int
    schema_version: int = ACTIVITY_SCHEMA_VERSION
    method: str = ACTIVITY_METHOD

    def validate(self) -> None:
        if self.schema_version != ACTIVITY_SCHEMA_VERSION:
            raise InvalidActivityRecipeError(
                f"未対応のactivity schemaです: {self.schema_version}"
            )
        if self.method != ACTIVITY_METHOD:
            raise InvalidActivityRecipeError(f"未対応のactivity methodです: {self.method}")
        for field in ("window_frames", "hop_frames", "minimum_active_frames"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise InvalidActivityRecipeError(f"{field}は正の整数である必要があります")
        for field in ("bridge_gap_frames", "padding_frames"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise InvalidActivityRecipeError(f"{field}は0以上の整数である必要があります")
        for field in ("onset_rms", "offset_rms"):
            value = getattr(self, field)
            if (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not np.isfinite(value) or value <= 0
            ):
                raise InvalidActivityRecipeError(f"{field}は有限な正数である必要があります")
        if self.offset_rms > self.onset_rms:
            raise InvalidActivityRecipeError("offset_rmsはonset_rms以下である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"bgm-activity-rms-v{self.schema_version}-{digest}"


@dataclass(frozen=True)
class RelativeHysteresisRecipe:
    """区間の開始は絶対量、終わりはその区間自身の強さに対する相対量で決める。

    静かな曲では、曲全体が絶対しきい値の近くに来る。固定のoffset_rmsでは同じ一曲が
    途中で切れ、立ち上がりと余韻を失う。ここではonsetの絶対gateを変えずに、検出した
    区間のpeakから一定dB下までを同じ区間として保つ。無音・雑音へ無制限に広がらない
    ように、絶対の下限も併せて宣言する。
    """

    window_frames: int
    hop_frames: int
    onset_rms: float
    offset_ratio_db: float
    offset_floor_rms: float
    bridge_gap_frames: int
    minimum_active_frames: int
    padding_frames: int
    schema_version: int = ACTIVITY_SCHEMA_VERSION
    method: str = RELATIVE_HYSTERESIS_METHOD

    def validate(self) -> None:
        if self.schema_version != ACTIVITY_SCHEMA_VERSION:
            raise InvalidActivityRecipeError(
                f"未対応のactivity schemaです: {self.schema_version}"
            )
        if self.method != RELATIVE_HYSTERESIS_METHOD:
            raise InvalidActivityRecipeError(f"未対応のactivity methodです: {self.method}")
        for field in ("window_frames", "hop_frames", "minimum_active_frames"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise InvalidActivityRecipeError(f"{field}は正の整数である必要があります")
        for field in ("bridge_gap_frames", "padding_frames"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise InvalidActivityRecipeError(f"{field}は0以上の整数である必要があります")
        for field in ("onset_rms", "offset_floor_rms"):
            value = getattr(self, field)
            if (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not np.isfinite(value) or value <= 0
            ):
                raise InvalidActivityRecipeError(f"{field}は有限な正数である必要があります")
        value = self.offset_ratio_db
        if (
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not np.isfinite(value) or value <= 0
        ):
            raise InvalidActivityRecipeError("offset_ratio_dbは有限な正数である必要があります")
        if self.offset_floor_rms > self.onset_rms:
            raise InvalidActivityRecipeError("offset_floor_rmsはonset_rms以下である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def offset_rms_for(self, peak_rms: float) -> float:
        """この区間のpeakに対する終端しきい値。絶対の下限を下回らない。"""
        self.validate()
        relative = float(peak_rms) * 10.0 ** (-self.offset_ratio_db / 20.0)
        return max(float(self.offset_floor_rms), relative)

    @property
    def recipe_id(self) -> str:
        self.validate()
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"bgm-activity-relative-rms-v{self.schema_version}-{digest}"


@dataclass(frozen=True)
class StemEnergyShareRecipe:
    """正規化したMusic/Speech/Effects成果物からmusic-like区間を得る。"""

    onset_share: float
    offset_share: float
    onset_music_rms: float
    offset_music_rms: float
    window_frames: int
    hop_frames: int
    bridge_gap_frames: int
    minimum_active_frames: int
    padding_frames: int
    schema_version: int = ACTIVITY_SCHEMA_VERSION
    method: str = STEM_ENERGY_SHARE_METHOD

    ANALYSIS_STEM: ClassVar[str] = "Music"
    COMPARISON_STEMS: ClassVar[tuple[str, str]] = ("Speech", "Effects")

    def validate(self) -> None:
        if self.schema_version != ACTIVITY_SCHEMA_VERSION:
            raise InvalidActivityRecipeError(
                f"未対応のactivity schemaです: {self.schema_version}"
            )
        if self.method != STEM_ENERGY_SHARE_METHOD:
            raise InvalidActivityRecipeError(f"未対応のactivity methodです: {self.method}")
        for field in ("window_frames", "hop_frames", "minimum_active_frames"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise InvalidActivityRecipeError(f"{field}は正の整数である必要があります")
        for field in ("bridge_gap_frames", "padding_frames"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise InvalidActivityRecipeError(f"{field}は0以上の整数である必要があります")
        for field in ("onset_share", "offset_share"):
            value = getattr(self, field)
            if (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not np.isfinite(value) or value <= 0 or value > 1
            ):
                raise InvalidActivityRecipeError(f"{field}は0より大きく1以下である必要があります")
        for field in ("onset_music_rms", "offset_music_rms"):
            value = getattr(self, field)
            if (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not np.isfinite(value) or value <= 0
            ):
                raise InvalidActivityRecipeError(f"{field}は有限な正数である必要があります")
        if self.offset_share > self.onset_share:
            raise InvalidActivityRecipeError("offset_shareはonset_share以下である必要があります")
        if self.offset_music_rms > self.onset_music_rms:
            raise InvalidActivityRecipeError(
                "offset_music_rmsはonset_music_rms以下である必要があります"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def parameters_dict(self) -> dict[str, int | float]:
        return {
            key: value for key, value in self.to_dict().items()
            if key not in {"schema_version", "method"}
        }

    @property
    def recipe_id(self) -> str:
        self.validate()
        canonical = json.dumps(
            self.parameters_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"bgm-stem-energy-share-v{self.schema_version}-{digest}"

    @property
    def required_stems(self) -> tuple[str, str, str]:
        return (self.ANALYSIS_STEM, *self.COMPARISON_STEMS)


@dataclass(frozen=True)
class RelativeStemEnergyShareRecipe:
    """区間の開始はMusicの絶対量と取り分で決め、続きはMusic自身の強さで決める。

    完成mixでは台詞に合わせてBGMが下げられる。取り分で終わりを決めると、同じ一曲が
    台詞のたびに切れる。ここでは開始の二つのgateは変えず、続きの判断からは取り分を外し、
    区間のpeakから一定dB下までをMusicが残っている証拠として扱う。無音・微小な残留へ
    広がらないように、絶対の下限も宣言する。
    """

    onset_share: float
    onset_music_rms: float
    offset_ratio_db: float
    offset_floor_rms: float
    window_frames: int
    hop_frames: int
    bridge_gap_frames: int
    minimum_active_frames: int
    padding_frames: int
    schema_version: int = ACTIVITY_SCHEMA_VERSION
    method: str = STEM_ENERGY_SHARE_RELATIVE_METHOD

    ANALYSIS_STEM: ClassVar[str] = "Music"
    COMPARISON_STEMS: ClassVar[tuple[str, str]] = ("Speech", "Effects")

    def validate(self) -> None:
        if self.schema_version != ACTIVITY_SCHEMA_VERSION:
            raise InvalidActivityRecipeError(
                f"未対応のactivity schemaです: {self.schema_version}"
            )
        if self.method != STEM_ENERGY_SHARE_RELATIVE_METHOD:
            raise InvalidActivityRecipeError(f"未対応のactivity methodです: {self.method}")
        for field in ("window_frames", "hop_frames", "minimum_active_frames"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise InvalidActivityRecipeError(f"{field}は正の整数である必要があります")
        for field in ("bridge_gap_frames", "padding_frames"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise InvalidActivityRecipeError(f"{field}は0以上の整数である必要があります")
        value = self.onset_share
        if (
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not np.isfinite(value) or value <= 0 or value > 1
        ):
            raise InvalidActivityRecipeError("onset_shareは0より大きく1以下である必要があります")
        for field in ("onset_music_rms", "offset_floor_rms", "offset_ratio_db"):
            value = getattr(self, field)
            if (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not np.isfinite(value) or value <= 0
            ):
                raise InvalidActivityRecipeError(f"{field}は有限な正数である必要があります")
        if self.offset_floor_rms > self.onset_music_rms:
            raise InvalidActivityRecipeError(
                "offset_floor_rmsはonset_music_rms以下である必要があります"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def parameters_dict(self) -> dict[str, int | float]:
        return {
            key: value for key, value in self.to_dict().items()
            if key not in {"schema_version", "method"}
        }

    def offset_rms_for(self, peak_rms: float) -> float:
        self.validate()
        relative = float(peak_rms) * 10.0 ** (-self.offset_ratio_db / 20.0)
        return max(float(self.offset_floor_rms), relative)

    @property
    def recipe_id(self) -> str:
        self.validate()
        canonical = json.dumps(
            self.parameters_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"bgm-stem-energy-share-relative-v{self.schema_version}-{digest}"

    @property
    def required_stems(self) -> tuple[str, str, str]:
        return (self.ANALYSIS_STEM, *self.COMPARISON_STEMS)


DiscoveryRecipe = (ActivityRecipe | RelativeHysteresisRecipe | StemEnergyShareRecipe
                   | RelativeStemEnergyShareRecipe)
# 単一stemを走査するRecipe。区間化の規則だけが違う。
SINGLE_SIGNAL_RECIPES = (ActivityRecipe, RelativeHysteresisRecipe)
# 3 stemの取り分を見るRecipe。
STEM_SHARE_RECIPES = (StemEnergyShareRecipe, RelativeStemEnergyShareRecipe)


@dataclass(frozen=True)
class ActivityCandidate:
    candidate_id: str
    start_frame: int
    end_frame: int
    peak_rms: float


@dataclass(frozen=True)
class ActivityDiscoveryResult:
    recipe_id: str
    sample_rate: int
    source_frames: int
    window_count: int
    candidates: tuple[ActivityCandidate, ...]

    def as_detections(self) -> list[DetectedOccurrence]:
        return [
            DetectedOccurrence(
                candidate_id=item.candidate_id,
                start_frame=item.start_frame,
                end_frame=item.end_frame,
            )
            for item in self.candidates
        ]


def load_activity_recipe(path: Path) -> ActivityRecipe:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidActivityRecipeError(f"activity Recipeを読めません: {path}") from exc
    if not isinstance(raw, dict) or set(raw) != _RECIPE_KEYS:
        missing = sorted(_RECIPE_KEYS - set(raw) if isinstance(raw, dict) else _RECIPE_KEYS)
        extra = sorted(set(raw) - _RECIPE_KEYS if isinstance(raw, dict) else set())
        raise InvalidActivityRecipeError(
            f"activity Recipeのキーが不正です: missing={missing}, extra={extra}"
        )
    recipe = ActivityRecipe(**raw)
    recipe.validate()
    return recipe


def load_relative_hysteresis_recipe(path: Path) -> RelativeHysteresisRecipe:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidActivityRecipeError(f"activity Recipeを読めません: {path}") from exc
    if not isinstance(raw, dict) or set(raw) != _RELATIVE_RECIPE_KEYS:
        missing = sorted(
            _RELATIVE_RECIPE_KEYS - set(raw) if isinstance(raw, dict) else _RELATIVE_RECIPE_KEYS
        )
        extra = sorted(set(raw) - _RELATIVE_RECIPE_KEYS if isinstance(raw, dict) else set())
        raise InvalidActivityRecipeError(
            f"activity Recipeのキーが不正です: missing={missing}, extra={extra}"
        )
    recipe = RelativeHysteresisRecipe(**raw)
    recipe.validate()
    return recipe


def load_stem_energy_share_recipe(path: Path) -> StemEnergyShareRecipe:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidActivityRecipeError(f"activity Recipeを読めません: {path}") from exc
    if not isinstance(raw, dict) or set(raw) != _STEM_ENERGY_SHARE_RECIPE_KEYS:
        missing = sorted(
            _STEM_ENERGY_SHARE_RECIPE_KEYS - set(raw)
            if isinstance(raw, dict) else _STEM_ENERGY_SHARE_RECIPE_KEYS
        )
        extra = sorted(
            set(raw) - _STEM_ENERGY_SHARE_RECIPE_KEYS if isinstance(raw, dict) else set()
        )
        raise InvalidActivityRecipeError(
            f"activity Recipeのキーが不正です: missing={missing}, extra={extra}"
        )
    recipe = StemEnergyShareRecipe(**raw)
    recipe.validate()
    return recipe


def load_relative_stem_energy_share_recipe(path: Path) -> RelativeStemEnergyShareRecipe:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidActivityRecipeError(f"activity Recipeを読めません: {path}") from exc
    keys = _STEM_ENERGY_SHARE_RELATIVE_RECIPE_KEYS
    if not isinstance(raw, dict) or set(raw) != keys:
        missing = sorted(keys - set(raw) if isinstance(raw, dict) else keys)
        extra = sorted(set(raw) - keys if isinstance(raw, dict) else set())
        raise InvalidActivityRecipeError(
            f"activity Recipeのキーが不正です: missing={missing}, extra={extra}"
        )
    recipe = RelativeStemEnergyShareRecipe(**raw)
    recipe.validate()
    return recipe


def load_discovery_recipe(path: Path) -> DiscoveryRecipe:
    """methodで厳密なRecipe schemaへdispatchする。"""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidActivityRecipeError(f"activity Recipeを読めません: {path}") from exc
    method = raw.get("method") if isinstance(raw, dict) else None
    if method == ACTIVITY_METHOD:
        return load_activity_recipe(path)
    if method == RELATIVE_HYSTERESIS_METHOD:
        return load_relative_hysteresis_recipe(path)
    if method == STEM_ENERGY_SHARE_METHOD:
        return load_stem_energy_share_recipe(path)
    if method == STEM_ENERGY_SHARE_RELATIVE_METHOD:
        return load_relative_stem_energy_share_recipe(path)
    raise InvalidActivityRecipeError(f"未対応のactivity methodです: {method}")


def _merge_intervals(
    intervals: list[tuple[int, int, float]], bridge_gap_frames: int,
) -> list[tuple[int, int, float]]:
    merged: list[list[int | float]] = []
    for start, end, peak in intervals:
        if merged and start - int(merged[-1][1]) <= bridge_gap_frames:
            merged[-1][1] = max(int(merged[-1][1]), end)
            merged[-1][2] = max(float(merged[-1][2]), peak)
        else:
            merged.append([start, end, peak])
    return [(int(start), int(end), float(peak)) for start, end, peak in merged]


def _absolute_intervals(
    windows: list[tuple[int, int, float]], recipe: ActivityRecipe,
) -> list[tuple[int, int, float]]:
    """開始も終わりも、先に宣言した絶対RMSで決める。"""
    raw_intervals: list[tuple[int, int, float]] = []
    active_start: int | None = None
    active_end = 0
    peak_rms = 0.0
    for start, end, rms in windows:
        if active_start is None:
            if rms >= recipe.onset_rms:
                active_start = start
                active_end = end
                peak_rms = rms
            continue
        if rms < recipe.offset_rms:
            raw_intervals.append((active_start, active_end, peak_rms))
            active_start = None
            active_end = 0
            peak_rms = 0.0
        else:
            active_end = end
            peak_rms = max(peak_rms, rms)
    if active_start is not None:
        raw_intervals.append((active_start, active_end, peak_rms))
    return raw_intervals


def _relative_intervals(
    windows: list[tuple[int, int, float]], recipe: RelativeHysteresisRecipe,
) -> list[tuple[int, int, float]]:
    """絶対gateで見つけた各区間を、その区間自身のpeakからの相対量まで外へ広げる。

    走査順に依存しないよう、種となる区間を先に全て決めてから左右へ広げる。広がった
    区間が重なる場合は後段の結合規則が一つにまとめる。
    """
    seeds: list[list[int | float]] = []
    for index, (_, _, rms) in enumerate(windows):
        if rms < recipe.onset_rms:
            continue
        if seeds and index == int(seeds[-1][1]) + 1:
            seeds[-1][1] = index
            seeds[-1][2] = max(float(seeds[-1][2]), rms)
        else:
            seeds.append([index, index, rms])

    raw_intervals: list[tuple[int, int, float]] = []
    for first, last, peak in seeds:
        first, last, peak = int(first), int(last), float(peak)
        threshold = recipe.offset_rms_for(peak)
        left = first
        while left > 0 and windows[left - 1][2] >= threshold:
            left -= 1
        right = last
        while right + 1 < len(windows) and windows[right + 1][2] >= threshold:
            right += 1
        raw_intervals.append((windows[left][0], windows[right][1], peak))
    return raw_intervals


def _result_from_windows(
    windows: list[tuple[int, int, float]], source_frames: int,
    sample_rate: int, recipe: ActivityRecipe | RelativeHysteresisRecipe,
) -> ActivityDiscoveryResult:
    """RMS window列を、Recipeが宣言した区間化規則と共通の結合規則で候補へ変換する。"""
    if isinstance(recipe, RelativeHysteresisRecipe):
        raw_intervals = _relative_intervals(windows, recipe)
    else:
        raw_intervals = _absolute_intervals(windows, recipe)

    merged = _merge_intervals(raw_intervals, recipe.bridge_gap_frames)
    retained = [
        (start, end, peak)
        for start, end, peak in merged
        if end - start >= recipe.minimum_active_frames
    ]
    padded = []
    for start, end, peak in retained:
        padded_start = max(0, start - recipe.padding_frames)
        padded_end = min(source_frames, end + recipe.padding_frames)
        if padded and padded_start <= padded[-1][1]:
            previous = padded[-1]
            padded[-1] = (previous[0], max(previous[1], padded_end), max(previous[2], peak))
        else:
            padded.append((padded_start, padded_end, peak))

    recipe_id = recipe.recipe_id
    candidates = tuple(
        ActivityCandidate(
            candidate_id=f"{recipe_id}-{index:04d}-{start}-{end}",
            start_frame=start,
            end_frame=end,
            peak_rms=peak,
        )
        for index, (start, end, peak) in enumerate(padded, start=1)
    )
    return ActivityDiscoveryResult(
        recipe_id=recipe_id,
        sample_rate=sample_rate,
        source_frames=source_frames,
        window_count=len(windows),
        candidates=candidates,
    )


def discover_activity(
    samples: np.ndarray,
    sample_rate: int,
    recipe: ActivityRecipe | RelativeHysteresisRecipe,
) -> ActivityDiscoveryResult:
    """(channels, frames)信号を区間化する。semantic role/identityは未確定のまま返す。"""
    recipe.validate()
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rateは正の整数である必要があります")
    data = np.asarray(samples)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2 or data.shape[0] <= 0 or data.shape[1] <= 0:
        raise ValueError("samplesは空でない(channels, frames)配列である必要があります")
    if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
        raise ValueError("samplesに有限でない値があります")
    source_frames = data.shape[1]

    windows: list[tuple[int, int, float]] = []
    for start in range(0, source_frames, recipe.hop_frames):
        end = min(source_frames, start + recipe.window_frames)
        block = data[:, start:end].astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(np.square(block))))
        windows.append((start, end, rms))

    return _result_from_windows(windows, source_frames, sample_rate, recipe)


def discover_activity_file(
    path: Path, recipe: ActivityRecipe | RelativeHysteresisRecipe,
) -> ActivityDiscoveryResult:
    """長尺信号を全量保持せず、fileから同じwindow規則でactivityを得る。"""
    import soundfile as sf

    recipe.validate()
    path = Path(path)
    with sf.SoundFile(path) as stream:
        sample_rate = stream.samplerate
        source_frames = stream.frames
        channels = stream.channels
        if sample_rate <= 0 or source_frames <= 0 or channels <= 0:
            raise ValueError(f"空または不正な音声です: {path}")
        windows: list[tuple[int, int, float]] = []
        for start in range(0, source_frames, recipe.hop_frames):
            end = min(source_frames, start + recipe.window_frames)
            stream.seek(start)
            data = stream.read(end - start, dtype="float32", always_2d=True)
            if data.shape != (end - start, channels) or not np.isfinite(data).all():
                raise ValueError(f"activity解析中に不正なsampleを読み込みました: {path}")
            block = data.T.astype(np.float64, copy=False)
            rms = float(np.sqrt(np.mean(np.square(block))))
            windows.append((start, end, rms))

    return _result_from_windows(windows, source_frames, sample_rate, recipe)


def discover_stem_energy_share_files(
    stem_paths: Mapping[str, Path],
    recipe: StemEnergyShareRecipe | RelativeStemEnergyShareRecipe,
) -> ActivityDiscoveryResult:
    """正規化した3 stemを逐次走査し、絶対量と相対量の両gateで区間化する。"""
    import soundfile as sf

    recipe.validate()
    normalized: dict[str, Path] = {}
    for stem, path in stem_paths.items():
        key = stem.casefold()
        if key in normalized:
            raise ValueError(f"stem identityが重複しています: {stem}")
        normalized[key] = Path(path)
    missing = [stem for stem in recipe.required_stems if stem.casefold() not in normalized]
    if missing:
        raise ValueError(f"activity解析に必要なstemがありません: {missing}")
    ordered_paths = [normalized[stem.casefold()] for stem in recipe.required_stems]

    with ExitStack() as resources:
        streams = [resources.enter_context(sf.SoundFile(path)) for path in ordered_paths]
        layouts = {(stream.samplerate, stream.frames, stream.channels) for stream in streams}
        if len(layouts) != 1:
            raise ValueError(f"stem layoutが一致しません: {layouts}")
        sample_rate, source_frames, channels = next(iter(layouts))
        if sample_rate <= 0 or source_frames <= 0 or channels <= 0:
            raise ValueError("空または不正なstemがあります")
        windows: list[tuple[int, int, float, float]] = []
        for start in range(0, source_frames, recipe.hop_frames):
            end = min(source_frames, start + recipe.window_frames)
            energies = []
            for path, stream in zip(ordered_paths, streams):
                stream.seek(start)
                data = stream.read(end - start, dtype="float32", always_2d=True)
                if data.shape != (end - start, channels) or not np.isfinite(data).all():
                    raise ValueError(f"activity解析中に不正なsampleを読み込みました: {path}")
                values = data.astype(np.float64, copy=False)
                energies.append(float(np.mean(np.square(values))))
            total_energy = sum(energies)
            music_rms = float(np.sqrt(energies[0]))
            music_share = energies[0] / total_energy if total_energy > 0 else 0.0
            windows.append((start, end, music_rms, music_share))

    if isinstance(recipe, RelativeStemEnergyShareRecipe):
        raw_intervals = _relative_share_intervals(windows, recipe)
    else:
        raw_intervals = _absolute_share_intervals(windows, recipe)

    return _stem_share_result(
        raw_intervals, source_frames, sample_rate, recipe, window_count=len(windows),
    )


def _absolute_share_intervals(
    windows: list[tuple[int, int, float, float]], recipe: StemEnergyShareRecipe,
) -> list[tuple[int, int, float]]:
    """開始も続きも、Musicの絶対量と取り分の両方で決める。"""
    raw_intervals: list[tuple[int, int, float]] = []
    active_start: int | None = None
    active_end = 0
    peak_rms = 0.0
    for start, end, music_rms, music_share in windows:
        if active_start is None:
            if (
                music_rms >= recipe.onset_music_rms
                and music_share >= recipe.onset_share
            ):
                active_start = start
                active_end = end
                peak_rms = music_rms
            continue
        if music_rms < recipe.offset_music_rms or music_share < recipe.offset_share:
            raw_intervals.append((active_start, active_end, peak_rms))
            active_start = None
            active_end = 0
            peak_rms = 0.0
        else:
            active_end = end
            peak_rms = max(peak_rms, music_rms)
    if active_start is not None:
        raw_intervals.append((active_start, active_end, peak_rms))
    return raw_intervals


def _relative_share_intervals(
    windows: list[tuple[int, int, float, float]], recipe: RelativeStemEnergyShareRecipe,
) -> list[tuple[int, int, float]]:
    """取り分は開始だけに使い、続きは区間自身のMusic peakからの相対量で決める。"""
    seeds: list[list[int | float]] = []
    for index, (_, _, music_rms, music_share) in enumerate(windows):
        if music_rms < recipe.onset_music_rms or music_share < recipe.onset_share:
            continue
        if seeds and index == int(seeds[-1][1]) + 1:
            seeds[-1][1] = index
            seeds[-1][2] = max(float(seeds[-1][2]), music_rms)
        else:
            seeds.append([index, index, music_rms])

    raw_intervals: list[tuple[int, int, float]] = []
    for first, last, peak in seeds:
        first, last, peak = int(first), int(last), float(peak)
        threshold = recipe.offset_rms_for(peak)
        left = first
        while left > 0 and windows[left - 1][2] >= threshold:
            left -= 1
        right = last
        while right + 1 < len(windows) and windows[right + 1][2] >= threshold:
            right += 1
        raw_intervals.append((windows[left][0], windows[right][1], peak))
    return raw_intervals


def _stem_share_result(
    raw_intervals: list[tuple[int, int, float]], source_frames: int, sample_rate: int,
    recipe: StemEnergyShareRecipe | RelativeStemEnergyShareRecipe, *, window_count: int,
) -> ActivityDiscoveryResult:
    """区間化規則によらず、結合・最小長・余白の規則は同じものを使う。"""
    merged = _merge_intervals(raw_intervals, recipe.bridge_gap_frames)
    retained = [
        item for item in merged
        if item[1] - item[0] >= recipe.minimum_active_frames
    ]
    padded: list[tuple[int, int, float]] = []
    for start, end, peak in retained:
        item = (
            max(0, start - recipe.padding_frames),
            min(source_frames, end + recipe.padding_frames),
            peak,
        )
        if padded and item[0] <= padded[-1][1]:
            previous = padded[-1]
            padded[-1] = (
                previous[0], max(previous[1], item[1]), max(previous[2], item[2]),
            )
        else:
            padded.append(item)
    recipe_id = recipe.recipe_id
    return ActivityDiscoveryResult(
        recipe_id=recipe_id,
        sample_rate=sample_rate,
        source_frames=source_frames,
        window_count=window_count,
        candidates=tuple(
            ActivityCandidate(
                candidate_id=f"{recipe_id}-{index:04d}",
                start_frame=start,
                end_frame=end,
                peak_rms=peak,
            )
            for index, (start, end, peak) in enumerate(padded, start=1)
        ),
    )
