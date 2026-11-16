"""確かめた既発売録音を部品として、番組で鳴った版（番組の編集を含む）のBGMを組み立てる。

契約は `docs/BGM.md`（Reference/Occurrence、完成mixに無かった音量へ戻さない）と `docs/ACQUISITION.md`
（Referenceを直接の再構成に使うならそのRecipeとして検証する）。

- 材料は、関係判定（`reference_relations`）の領域の地図で「この1秒はこの録音のこの位置で説明し尽くせる」と
  確かめた領域だけ。参照が説明できない領域（番組固有の素材・重ねた別の曲）は元の抽出のまま残す。
- 音量・EQ・位相は番組の音（公開する表現の音）へ窓ごとの短いFIRで合わせる。ダッキングなど番組での音量の変化を
  保ち、参照の音量へは戻さない。左右は別に合わせる。
- 同じ録音の別の位置へ移る編集点は、前後の領域の予測の誤差が最も小さくなる時刻を細かく探し、短い等電力の
  クロスフェードで繋ぐ。抽出の領域との境目も同じクロスフェード。
- 置き換える領域は、関係判定の会計（参照＋台詞・効果音stemの漏れで説明し尽くせ、参照に無い成分が無い）が
  きれいな領域だけ。描いた後も、合わせた予測が番組の音をどれだけ説明するか（fit）を測り、健全でなければ
  置き換えない。参照に無い成分の割合（novelty）は記録する（台詞の漏れを含むので判定には使わない）。
- 来歴（どの領域をどの録音のどこで置き換えたか、合わせ具合、境目）を返す。元の抽出は呼び出し側が保持する。
  この音は抽出そのものではなく、参照を使った再構成である（来歴で区別する）。
- 再構成する理由は、既発売では手に入らない版（番組だけの編集・長さ・繋ぎ・番組固有の素材を含む版・成分の
  違う版）を、手に入る既発売の部分と組み合わせてきれいにすることである。描いた結果が、一つの既発売録音の
  一続きの範囲でほぼ全部できている（番組だけの構造・素材が短い）なら、それは既発売の録音そのものと実質同じで、
  新しい資産として出さない（`availability`）。
- 成分の違う版（`released_recording_component_variant`）は、参照で説明し尽くせる領域だけを置き換え、成分の
  欠けた領域（番組だけの版の部分）は抽出のまま残す。成分の欠けが疑われる領域（absence）は置き換えない。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SCHEMA_VERSION = 3
METHOD = "reference-donor-region-reconstruction-v3"
# 関係の地図で、参照では説明できない（番組だけの素材・別の曲・成分の欠け）と記録した領域。描画は決して置き換えない。
PROTECTED_KINDS = ("unexplained_music", "additional_material", "missing_component")
# 二つの置き換えの間の隙間を橋渡ししてよい領域（窓が編集点を跨いだ同じ録音のセル・判断できない・小さい）。
BRIDGEABLE_KINDS = ("supported", "uncertain", "quiet")
REPRESENTATION = "reference_donor_reconstruction"


class ReconstructionError(ValueError):
    """再構成の計画か材料が契約に合わない。"""


@dataclass(frozen=True)
class ReconstructionRecipe:
    fir_taps: int = 9
    fit_window_seconds: float = 2.0
    fit_hop_seconds: float = 1.0
    ridge: float = 1e-3
    min_region_seconds: float = 2.0       # これより短い置き換えはしない
    min_fit: float = 0.5                  # 描画した領域で、予測が番組の音を説明する割合の下限（健全性）
    region_min_joint: float = 0.6         # 置き換える領域の会計：参照＋漏れで説明できる割合の中央値の下限
    # 置き換える領域の会計：参照に無い成分の割合の中央値の上限。置き換えは抽出した音を参照の予測で上書きするので、
    # 重ねた別の曲（-12dBで中央値0.012〜0.04）を通さない値にする（台詞の漏れのある実の分離の既発売の領域は 0.005前後）。
    region_max_novelty: float = 0.007
    region_cell_max_novelty: float = 0.06  # 置き換える領域の会計：どの1秒でも参照に無い成分がこれ以下（旧0.2は-6dBの重ねを通す）
    region_max_absence: float = 0.012     # 置き換える領域の会計：参照の成分の欠けの中央値の上限
    region_cell_max_absence: float = 0.04  # 置き換える領域の会計：どの1秒でも参照の成分の欠けがこれ以下
    # 番組だけの構造・素材がこれ未満なら、既発売の録音と実質同じ。関係判定のセル（1秒）より短い残りは測れないので
    # 1セル未満だけを許す。v3の「4秒かcueの10%」は、既発売の corpus-c の抜粋の最後の2秒を別の曲へ差し替えた番組の版を
    # 既発売として保留していた（参照の無い時は公開する欲しい音楽を、参照があるだけで消した。
    # `.state/validation/integrated-r2-claude/`）。端の短い差し替え・番組だけの繋ぎは残す価値がある（BGM_AND_AVAILABILITY）。
    unique_min_seconds: float = 1.0
    unique_min_fraction: float = 0.0      # （cueの長さに対する割合。長い方を使う）
    boundary_search_seconds: float = 0.75  # 編集点を探す幅（領域の境目の前後）
    crossfade_seconds: float = 0.03
    lag_search_frames: int = 64           # 出力の標本単位でのずれの詰め
    # Missing field in saved recipes means the historical renderer.
    transition_curve: str = "equal_power_v1"
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD
    relation_authority_policy: str = "scoped_nomination_requires_full_accounting_v1"

    def validate(self) -> None:
        if self.relation_authority_policy != "scoped_nomination_requires_full_accounting_v1":
            raise ReconstructionError("Unsupported relation authority policy")
        if self.transition_curve not in ("equal_power_v1", "raised_cosine_v2"):
            raise ReconstructionError("Unknown reconstruction transition")
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise ReconstructionError("未対応の再構成Recipeです")
        if self.fir_taps % 2 != 1 or not 1 <= self.fir_taps <= 63:
            raise ReconstructionError("fir_tapsは1以上63以下の奇数である必要があります")
        if not 0 < self.fit_hop_seconds <= self.fit_window_seconds / 2:
            raise ReconstructionError("窓とhopの関係が不正です")
        if not 0 < self.min_fit < 1 or not 0 < self.region_max_novelty < self.region_cell_max_novelty < 1:
            raise ReconstructionError("基準は0と1の間である必要があります")
        if (not 0 < self.region_max_absence < self.region_cell_max_absence < 1
                or not 0 <= self.unique_min_fraction < 1 or self.unique_min_seconds < 0):
            raise ReconstructionError("成分の欠け・実質同じの基準が不正です")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-reference-reconstruction-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def default_recipe() -> ReconstructionRecipe:
    return ReconstructionRecipe(transition_curve="raised_cosine_v2")


def clean_donor_regions(regions: list[dict], start_seconds: float, end_seconds: float,
                        recipe: ReconstructionRecipe | None = None) -> dict:
    """Apply the ordinary clean-region donor rules without relation authority gates.

    The caller must establish that the ordinary relation is eligible to use
    reconstruction. This shared pure rule is also used while deciding whether
    scoped observations are needed, so both paths use identical donor criteria.
    """
    recipe = recipe or default_recipe()
    recipe.validate()
    regions = [r for r in regions or []
               if r["end_seconds"] > start_seconds and r["start_seconds"] < end_seconds]
    donors = []
    rejected = []
    for region in regions:
        if region["kind"] != "supported" or region.get("reference_sha256") is None:
            continue
        if (region.get("median_joint") is None or region["median_joint"] < recipe.region_min_joint
                or region["median_novelty"] > recipe.region_max_novelty
                or region["max_novelty"] > recipe.region_cell_max_novelty
                or (region.get("median_absence") is not None
                    and region["median_absence"] > recipe.region_max_absence)
                or (region.get("max_absence") is not None
                    and region["max_absence"] > recipe.region_cell_max_absence)):
            rejected.append({k: region.get(k) for k in ("start_seconds", "end_seconds", "median_joint",
                                                         "median_novelty", "max_novelty", "median_absence",
                                                         "max_absence")})
            continue
        first, last = max(region["start_seconds"], start_seconds), min(region["end_seconds"], end_seconds)
        donors.append({"start_seconds": float(first), "end_seconds": float(last),
                       "reference_sha256": region["reference_sha256"], "offset_seconds": region["offset_seconds"],
                       "origin": region.get("origin")})
    protected = _protected_spans(regions, start_seconds, end_seconds)
    merged = []
    for donor in donors:
        if merged and 0 < donor["start_seconds"] - merged[-1]["end_seconds"] <= 1.0:
            gap = donor["start_seconds"] - merged[-1]["end_seconds"]
            if _bridgeable(regions, merged[-1], donor):
                merged[-1]["end_seconds"] += gap / 2
                donor = {**donor, "start_seconds": donor["start_seconds"] - gap / 2}
        merged.append(dict(donor))
    merged = [d for d in merged if d["end_seconds"] - d["start_seconds"] >= recipe.min_region_seconds]
    for donor in merged:
        donor["protected_spans"] = [list(span) for span in protected]
    return {"donors": merged, "rejected_regions": rejected,
            "protected_spans": [list(span) for span in protected]}


def plan(relation: dict | None, start_seconds: float, end_seconds: float,
         recipe: ReconstructionRecipe | None = None) -> dict:
    """関係の領域の地図から、置き換える領域（録音・ずれ）を決める。置き換えない領域は元の抽出のまま。

    置き換えるのは、参照で説明し尽くせる（supported）と確かめた領域だけ。同じ録音の別の位置へ移る編集点に挟まる
    1秒（窓が両側に跨ぐセル）は、両側の領域で覆う（境目は描く時に細かく決める）。追加の素材がある区間
    （ADDITIONAL）は置き換えない（重ねた音を落とさない）。
    """
    recipe = recipe or default_recipe()
    recipe.validate()
    relation = relation or {}
    if (relation.get("local_nomination") is not None
            or "scoped_local_observations_require_full_accounting"
            in (relation.get("authority_hold_reasons") or [])):
        return {"eligible": False, "reason": "scoped_local_observations_require_full_accounting", "donors": []}
    state = relation.get("state")
    if state not in ("released_recording_edit", "exact_released_recording", "released_recording_component_variant"):
        return {"eligible": False, "reason": f"relation_{state or 'none'}", "donors": []}
    regions = relation.get("regions") or []
    ordinary = clean_donor_regions(regions, start_seconds, end_seconds, recipe)
    merged, rejected = ordinary["donors"], ordinary["rejected_regions"]
    if not merged:
        return {"eligible": False, "reason": "no_clean_supported_region_long_enough", "donors": [],
                "rejected_regions": rejected}
    covered = sum(d["end_seconds"] - d["start_seconds"] for d in merged)
    return {"eligible": True, "reason": None, "donors": merged,
            "covered_seconds": round(covered, 3), "cue_seconds": round(end_seconds - start_seconds, 3),
            "protected_spans": ordinary["protected_spans"],
            "rejected_regions": rejected, "recipe_id": recipe.recipe_id,"recipe":recipe.to_dict()}


def _protected_spans(regions: list[dict], start_seconds: float, end_seconds: float) -> list[tuple[float, float]]:
    """cueの範囲の、参照では説明できないと記録した領域（隣り合うものは一つにまとめる）。"""
    spans: list[list[float]] = []
    for region in sorted(regions, key=lambda r: r["start_seconds"]):
        if region["kind"] not in PROTECTED_KINDS:
            continue
        first, last = max(region["start_seconds"], start_seconds), min(region["end_seconds"], end_seconds)
        if last <= first:
            continue
        if spans and first <= spans[-1][1] + 1e-9:
            spans[-1][1] = max(spans[-1][1], last)
        else:
            spans.append([float(first), float(last)])
    return [(a, b) for a, b in spans]


def _bridgeable(regions: list[dict], left: dict, right: dict) -> bool:
    """二つの置き換えの隙間 [left終了, right開始) を橋渡ししてよいか（隙間の全てが地図にあり、橋渡し可能な種類）。"""
    low, high = left["end_seconds"], right["start_seconds"]
    covered = low
    for region in sorted(regions, key=lambda r: r["start_seconds"]):
        if region["end_seconds"] <= low or region["start_seconds"] >= high:
            continue
        if region["start_seconds"] > covered + 1e-9:
            return False   # 地図に無い隙間
        kind = region["kind"]
        if kind not in BRIDGEABLE_KINDS:
            return False
        if kind == "supported" and region.get("reference_sha256") not in (left["reference_sha256"],
                                                                         right["reference_sha256"]):
            return False   # 別の録音で説明された隙間は、この置き換えの録音の素材ではない
        covered = max(covered, region["end_seconds"])
    return covered >= high - 1e-9


def _fit_channel(base: np.ndarray, reference: np.ndarray, recipe: ReconstructionRecipe, rate: int) -> np.ndarray:
    """参照（長さ = base + taps - 1）を、窓ごとのFIRでbaseへ合わせた予測。窓はハン窓で重ねる。"""
    taps = recipe.fir_taps
    length = base.size
    window = max(taps * 4, int(recipe.fit_window_seconds * rate))
    hop = max(1, int(recipe.fit_hop_seconds * rate))
    predicted = np.zeros(length)
    weights = np.zeros(length)
    full = np.hanning(window + 2)[1:-1]
    for begin in range(-hop, length, hop):
        lo, hi = max(begin, 0), min(begin + window, length)
        if hi - lo < taps:
            continue
        weight = full[lo - begin:hi - begin]
        design = sliding_window_view(reference[lo:hi + taps - 1], taps)
        gram = design.T @ (weight[:, None] * design)
        gram[np.diag_indices_from(gram)] += max(recipe.ridge * float(np.trace(gram)) / taps, 1e-12)
        coefficients = np.linalg.solve(gram, design.T @ (weight * base[lo:hi]))
        predicted[lo:hi] += weight * (design @ coefficients)
        weights[lo:hi] += weight
    return predicted / np.maximum(weights, 1e-12)


def _refine_lag(base: np.ndarray, reference: np.ndarray, start: int, limit: int) -> int:
    """出力の標本単位で、参照の位置の小さなずれを相互相関で詰める（左右の和で）。"""
    length = min(base.shape[1], reference.shape[1] - start - limit)
    if length <= 4 * limit:
        return 0
    y = base.sum(axis=0)[limit:length - limit]
    best, best_lag = -np.inf, 0
    x_full = reference.sum(axis=0)
    for lag in range(-limit, limit + 1):
        x = x_full[start + limit + lag:start + limit + lag + y.size]
        if x.size != y.size:
            continue
        value = float(y @ x)
        if value > best:
            best, best_lag = value, lag
    return best_lag


def _novelty(base: np.ndarray, predicted: np.ndarray, rate: int) -> float:
    """baseのうち、予測の4倍以上の時間周波数成分の割合（参照に無い音）。100Hz〜5kHz。"""
    n = 2048 if rate > 30000 else 1024
    hop = n // 4
    if base.size < n:
        return 0.0
    window = np.hanning(n)

    def power(x):
        return np.abs(np.fft.rfft(sliding_window_view(x, n)[::hop] * window, axis=1)) ** 2

    lo, hi = int(100 * n / rate), int(5000 * n / rate)
    m, p = power(base)[:, lo:hi], power(predicted)[:, lo:hi]
    total = float(m.sum())
    return float(m[m > 4.0 * p].sum() / total) if total > 0 else 0.0


def render(base: np.ndarray, rate: int, cue_start_seconds: float, donors: list[dict], reference_audio,
           recipe: ReconstructionRecipe | None = None, *, protected_spans=()) -> tuple[np.ndarray, dict]:
    """元の表現の音（2ch・`rate`）の置き換える領域を、参照の予測で置き換えた音と、その来歴を返す。

    `cue_start_seconds` は `base` の先頭の番組上の時刻（関係の地図と同じ時間軸）。`reference_audio(sha256)` は
    その参照の2ch・`rate` の音声（`[2, n]`）を返す。基準に届かない領域は置き換えない。
    """
    recipe = recipe or default_recipe()
    recipe.validate()
    base = np.asarray(base, dtype=np.float64)
    if base.ndim != 2 or base.shape[0] != 2:
        raise ReconstructionError("2chの音が必要です")
    length = base.shape[1]
    protected_spans = [list(span) for span in protected_spans]+[list(span) for d in donors for span in d.get("protected_spans",[])]
    taps, half = recipe.fir_taps, recipe.fir_taps // 2
    predictions, rows = [], []
    references: dict[str, np.ndarray] = {}
    for donor in donors:
        sha = donor["reference_sha256"]
        if sha not in references:
            references[sha] = np.asarray(reference_audio(sha), dtype=np.float64)
        reference = references[sha]
        margin = int(recipe.boundary_search_seconds * rate) + int(recipe.crossfade_seconds * rate)
        first = max(0, int(round((donor["start_seconds"] - cue_start_seconds) * rate)) - margin)
        last = min(length, int(round((donor["end_seconds"] - cue_start_seconds) * rate)) + margin)
        if last - first < taps * 8:
            continue
        ref_first = int(round((cue_start_seconds + first / rate + donor["offset_seconds"]) * rate))
        lag = _refine_lag(base[:, first:last], reference, ref_first, recipe.lag_search_frames)
        ref_first += lag
        prediction = np.zeros((2, last - first))
        for channel in range(2):
            segment = np.zeros(last - first + taps - 1)
            lo = ref_first - half
            a, b = max(lo, 0), min(lo + segment.size, reference.shape[1])
            if b > a:
                segment[a - lo:b - lo] = reference[channel, a:b]
            prediction[channel] = _fit_channel(base[channel, first:last], segment, recipe, rate)
        predictions.append((donor, first, last, prediction, lag))
    # どの標本をどの予測で描くか（-1は元の音）。領域の芯は各予測、重なる所は誤差の小さい側の境目で切り替える。
    owner = np.full(length, -1, dtype=np.int64)
    for number, (donor, first, last, prediction, lag) in enumerate(predictions):
        core_first = max(0, int(round((donor["start_seconds"] - cue_start_seconds) * rate)))
        core_last = min(length, int(round((donor["end_seconds"] - cue_start_seconds) * rate)))
        owner[core_first:core_last] = number
    boundaries = []
    for number in range(1, len(predictions)):
        left, right = predictions[number - 1], predictions[number]
        nominal = int(round((right[0]["start_seconds"] - cue_start_seconds) * rate))
        if nominal - int(round((left[0]["end_seconds"] - cue_start_seconds) * rate)) > 1:
            continue  # 間に元の音の領域がある
        search = int(recipe.boundary_search_seconds * rate)
        lo, hi = max(left[1], right[1], nominal - search), min(left[2], right[2], nominal + search)
        if hi - lo < 2:
            continue
        frame = max(1, rate // 100)
        error_left = np.sum((base[:, lo:hi] - left[3][:, lo - left[1]:hi - left[1]]) ** 2, axis=0)
        error_right = np.sum((base[:, lo:hi] - right[3][:, lo - right[1]:hi - right[1]]) ** 2, axis=0)
        cut_points = np.arange(0, hi - lo, frame)
        cumulative_left = np.concatenate([[0.0], np.cumsum(error_left)])
        cumulative_right = np.concatenate([[0.0], np.cumsum(error_right)])
        costs = [cumulative_left[c] + cumulative_right[-1] - cumulative_right[c] for c in cut_points]
        cut = lo + int(cut_points[int(np.argmin(costs))])
        owner[lo:cut] = number - 1
        owner[cut:hi] = number
        boundaries.append({"seconds": round(cue_start_seconds + cut / rate, 4),
                           "nominal_seconds": round(cue_start_seconds + nominal / rate, 4),
                           "from_offset_seconds": left[0]["offset_seconds"],
                           "to_offset_seconds": right[0]["offset_seconds"]})
    # 参照では説明できないと記録した領域は、境目の探索・クロスフェードの幅を含めて、予測の所有にしない。
    protected = np.zeros(length, dtype=bool)
    for span_start, span_end in protected_spans:
        lo_p = max(0, int(np.floor((span_start - cue_start_seconds) * rate)))
        hi_p = min(length, int(np.ceil((span_end - cue_start_seconds) * rate)))
        if hi_p > lo_p:
            protected[lo_p:hi_p] = True
    owner[protected] = -1
    # 予測ごとの基準（置き換える範囲で測る）。届かなければ元の音へ戻す。
    for number, (donor, first, last, prediction, lag) in enumerate(predictions):
        mask = owner == number
        if not mask.any():
            continue
        target = base[:, mask]
        indices = np.flatnonzero(mask)
        predicted = prediction[:, indices - first]
        residual = float(np.sum((target - predicted) ** 2))
        energy = float(np.sum(target ** 2))
        fit = 1.0 - residual / energy if energy > 0 else 0.0
        novelty = _novelty(target.mean(axis=0), predicted.mean(axis=0), rate)
        accepted = fit >= recipe.min_fit
        rows.append({"start_seconds": round(cue_start_seconds + indices[0] / rate, 4),
                     "end_seconds": round(cue_start_seconds + (indices[-1] + 1) / rate, 4),
                     "reference_sha256": donor["reference_sha256"],
                     "reference_start_seconds": round(cue_start_seconds + indices[0] / rate + donor["offset_seconds"]
                                                      + lag / rate, 4),
                     "offset_seconds": round(donor["offset_seconds"] + lag / rate, 6), "lag_frames": int(lag),
                     "fit": round(fit, 4), "novelty": round(novelty, 4),
                     "removed_db": round(10 * np.log10(max(residual, 1e-20) / max(energy, 1e-20)), 2),
                     "accepted": bool(accepted),
                     "reason": None if accepted else "low_fit"})
        if not accepted:
            owner[mask] = -1
    # 描く。所有の変わり目に短い等電力のクロスフェード。
    full_predictions = []
    for donor, first, last, prediction, lag in predictions:
        full = base.copy()
        full[:, first:last] = prediction
        full_predictions.append(full)
    out = base.copy()
    for number, full in enumerate(full_predictions):
        out[:, owner == number] = full[:, owner == number]
    fade = max(1, int(recipe.crossfade_seconds * rate))
    changes = np.flatnonzero(np.diff(owner)) + 1
    for change in changes:
        a, b = int(owner[change - 1]), int(owner[change])
        lo, hi = max(0, change - fade // 2), min(length, change + fade - fade // 2)
        # 保護した領域に接する変わり目は、クロスフェードの全幅を置き換える側の内側へ寄せる（保護した領域は元の音のまま）。
        if protected[change - 1] and a < 0:
            lo, hi = change, min(length, change + fade)
        elif protected[change] and b < 0:
            lo, hi = max(0, change - fade), change
        source_a = base if a < 0 else full_predictions[a]
        source_b = base if b < 0 else full_predictions[b]
        t = np.linspace(0.0, 1.0, hi - lo)
        if recipe.transition_curve == "raised_cosine_v2":
            weight = .5-.5*np.cos(t*np.pi)
            out[:, lo:hi] = (1-weight)*source_a[:, lo:hi]+weight*source_b[:, lo:hi]
        else:
            out[:, lo:hi] = (np.cos(t * np.pi / 2) * source_a[:, lo:hi] + np.sin(t * np.pi / 2) * source_b[:, lo:hi])
    out[:, protected] = base[:, protected]
    if not np.isfinite(out).all():
        raise ReconstructionError("再構成に有限でない値があります")
    replaced = int((owner >= 0).sum())
    provenance = {"schema_version": SCHEMA_VERSION, "method": METHOD, "recipe": recipe.to_dict(),
                  "recipe_id": recipe.recipe_id, "representation": REPRESENTATION,
                  "replaced_seconds": round(replaced / rate, 3), "cue_seconds": round(length / rate, 3),
                  "cue_start_seconds": cue_start_seconds, "cue_end_seconds": cue_start_seconds+length/rate,
                  "protected_spans": [list(s) for s in protected_spans or ()],
                  "donor_regions": rows, "edit_points": boundaries,
                  "statement": "参照を使った再構成：記録した領域は既発売録音を番組の音量・EQへ合わせた音。"
                               "それ以外の領域と元の抽出は番組の音のまま。"}
    return out, provenance


def availability(provenance: dict, recipe: ReconstructionRecipe | None = None, *, distribution=None) -> dict:
    """Suppress only a completely covered, independently distributed recording.

    Interval union prevents duplicate accounting. Every uncovered or protected
    fragment remains potentially valuable, regardless of duration. Legacy
    unique thresholds remain readable in saved recipes but supply no new
    suppression authority. Rounding tolerance is limited to serialized edges.
    """
    from lss.bgm import release_availability
    recipe = recipe or default_recipe()
    recipe.validate()
    cue = float(provenance.get("cue_seconds") or 0.)
    rows = [r for r in provenance.get("donor_regions", []) if r.get("accepted")]
    result = {"independently_available": False, "basis": "episode_unique_structure_or_material",
              "policy": "complete_union_and_checked_distribution_v1", "threshold_seconds": 0.,
              "unique_seconds": cue, "released_spans": 0}
    if not rows or cue <= 0:
        return {**result,"basis":"no_replaced_region"}
    start = provenance.get("cue_start_seconds")
    end = provenance.get("cue_end_seconds")
    if start is None or end is None or not np.isfinite([start,end,cue]).all() or abs(end-start-cue) > .001:
        return {**result,"basis":"cue_extent_not_established"}
    # Count interval unions, not rows. Duplicate and overlapping receipts must
    # never conceal an uncovered intro, internal gap or subsecond alternate tail.
    sha = rows[0]["reference_sha256"]
    offset = rows[0]["offset_seconds"]
    groups = {(r["reference_sha256"],round(r["offset_seconds"],4)) for r in rows}
    intervals = sorted((max(start,r["start_seconds"]),min(end,r["end_seconds"])) for r in rows
                       if r["reference_sha256"] == sha and abs(r["offset_seconds"]-offset) < .0001)
    cursor, covered, holes = start, 0., []
    for a,b in intervals:
        if not np.isfinite([a,b]).all() or b <= a:
            continue
        if a > cursor:
            holes.append([cursor,a])
        covered += max(0.,b-max(a,cursor))
        cursor = max(cursor,b)
    if cursor < end:
        holes.append([cursor,end])
    unique = max(0.,end-start-covered)
    protected = [(max(start,a),min(end,b)) for a,b in provenance.get("protected_spans",[])
                 if a < end and b > start]
    receipt = (distribution or {}).get(sha)
    complete = not holes or all(b-a <= .00011 for a,b in holes)  # receipt rounding only, never an analysis cell
    available = bool(complete and not protected and len(groups)==1 and release_availability.checked(receipt,sha))
    return {**result,"independently_available":available,
            "basis":"one_contiguous_released_span_makes_the_cue" if available else
                    "independent_distribution_unknown" if complete and not release_availability.checked(receipt,sha) else result["basis"],
            "reference_sha256":sha,"offset_seconds":offset,
            "reference_range_seconds":[intervals[0][0]+offset,intervals[-1][1]+offset] if intervals else None,
            "main_span_seconds":covered,"unique_seconds":unique,"uncovered_spans":holes,
            "protected_spans":protected,"released_spans":len(groups),"independent_distribution":receipt}
