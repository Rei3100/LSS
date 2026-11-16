"""公開する劇伴cueに残った台詞を、伴奏抽出の既定modelとの共通成分で取り除く。

SFC（台詞・音楽・効果音の3分類）は台詞の場面で台詞の一部を音楽の出力へ残す。実番組9話の公開cueでは、
秒の約3割でSFCの音楽の出力に台詞が聞こえる（タグ付けのSpeech確率0.3以上）。2回目の解析（`second_pass`）は
効果音の混入を取るためのもので、台詞の残りには効かない。伴奏抽出の既定model（台詞・歌を除く）は台詞を
よく取るが効果音を残す。そこで、両者の時間周波数ごとの小さい振幅を取る（``consensus``）。台詞は伴奏modelが、
効果音はSFCが小さくするので、どちらかが除いた成分は残らない。

1. **試す価値（eligibility）**：製品が意味判定のために既に作ったタグ付け（SFCの1回目の音楽の出力）の
   Speech確率を、cueの窓で平均し、``speech_gate`` 以上のcueだけ。台詞が残らないcueは変えない。
   伴奏modelは歌を除くので、歌の無いことが公開の条件であるcue（`cue_semantics` のWANTED）だけに掛ける。
   平均が gate 未満のcueでも、Speech確率が gate 以上の窓の時間だけは共通成分にする（``local_regions`` /
   ``local_consensus``、Recipe v10）。窓の外は処理済みの音のまま。cue全体や前後へ広げると、台詞の無い所で
   伴奏modelの欠損を劇伴へ持ち込む（`.state/validation/bgm-published-quality-audit-r1/`、
   `.state/validation/bgm-model-dependence-r1/`）。
2. **伴奏**：cueの区間に前後 ``context_seconds`` の文脈を付けた部分だけを伴奏の既定modelへ通し、cueの部分を
   保存する（`run_instrumental_spans`。workerが行う）。保存した音から作るので、作り直しに推論は要らない。
3. **合成**：処理済みの音（1回目か採った2回目、歌唱回収の後）と伴奏の、STFTの係数ごとに振幅の小さい方の
   係数を取る。時間軸・左右・長さは変えない。

実測（事前に固定した判定、`.state/validation/bgm-dialogue-residual-gate-r1/`）：corpus-a の既発売劇伴の区間で、
該当13区間の全てでSFCより録音成分に近づき（中央値+6.7dB、最小+1.7dB）、非該当10区間でも悪化0件。
corpus-c の既存ベンチでも該当8区間の全てで改善。正解・参照・試聴は判定に使わない。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Callable

import numpy as np

SCHEMA_VERSION = 2
METHOD = "dialogue-residual-instrumental-consensus-v1"
SPEECH_CLASS = 0  # AudioSet "Speech"（`event_tagging` の確率の列）
SAMPLE_RATE = 44100


class DialogueResidualError(ValueError):
    """台詞の残りの除去の宣言、または渡された材料が契約に合わない。"""


@dataclass(frozen=True)
class DialogueResidualRecipe:
    """宣言。どれかが変われば識別子が変わる。"""

    speech_gate: float
    context_seconds: float
    n_fft: int
    hop: int
    stem_target: str
    local_dilation_windows: int = 0
    local_ramp_seconds: float = 0.25
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise DialogueResidualError("未対応のschemaです")
        for name, low, high in (("speech_gate", 0.0, 1.0), ("context_seconds", 0.0, 120.0),
                                ("local_ramp_seconds", 0.0, 5.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not low <= float(value) <= high:
                raise DialogueResidualError(f"{name}は{low}以上{high}以下である必要があります")
        value = self.local_dilation_windows
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
            raise DialogueResidualError("local_dilation_windowsは0以上10以下の整数である必要があります")
        for name in ("n_fft", "hop"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 16:
                raise DialogueResidualError(f"{name}は16以上の整数である必要があります")
        if self.hop > self.n_fft:
            raise DialogueResidualError("hopはn_fft以下である必要があります")
        if not isinstance(self.stem_target, str) or not self.stem_target:
            raise DialogueResidualError("stem_targetが未指定です")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-dialogue-residual-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def default_recipe() -> DialogueResidualRecipe:
    """gateと合成は実番組で確かめた値そのもの（`bgm-dialogue-residual-gate-r1`、2026-09-27）。

    cue平均が ``speech_gate`` 未満のcueは、Speech窓そのものだけに掛ける（広げない、境目0.25秒）。前後1窓へ
    広げる版は台詞の無い窓の音を削り、事前固定の基準で不成立（`bgm-model-dependence-r1`、2026-09-28）。
    """
    return DialogueResidualRecipe(speech_gate=0.3, context_seconds=15.0, n_fft=2048, hop=512,
                                  stem_target="Instrumental", local_dilation_windows=0, local_ramp_seconds=0.25)


def speech_on_music(probabilities: np.ndarray, start_seconds: float, end_seconds: float) -> float | None:
    """cueの窓（1秒格子、窓 ``i`` は [i, i+1) 秒）のSpeech確率の平均。窓が無ければNone。"""
    probabilities = np.asarray(probabilities)
    first = max(0, int(math.floor(start_seconds)))
    last = min(probabilities.shape[0], int(math.ceil(end_seconds)))
    if last <= first:
        return None
    return float(np.mean(probabilities[first:last, SPEECH_CLASS].astype(np.float64)))


def eligibility(probabilities: np.ndarray, start_seconds: float, end_seconds: float,
                recipe: DialogueResidualRecipe) -> dict:
    recipe.validate()
    level = speech_on_music(probabilities, start_seconds, end_seconds)
    return {"speech_on_music": None if level is None else round(level, 4),
            "eligible": level is not None and level >= recipe.speech_gate}


def context_span(start_frame: int, end_frame: int, total_frames: int, rate: int,
                 recipe: DialogueResidualRecipe) -> tuple[int, int]:
    """伴奏modelへ入れる範囲（cueの前後に文脈を付け、信号の内側に収める）。"""
    if not 0 <= start_frame < end_frame <= total_frames:
        raise DialogueResidualError("cueの区間が信号の外にあります")
    margin = int(round(recipe.context_seconds * rate))
    return max(0, start_frame - margin), min(total_frames, end_frame + margin)


def _stft(signal: np.ndarray, recipe: DialogueResidualRecipe):
    from scipy.signal import stft

    return [stft(signal[:, channel], nperseg=recipe.n_fft, noverlap=recipe.n_fft - recipe.hop)[2]
            for channel in range(signal.shape[1])]


def consensus(music: np.ndarray, instrumental: np.ndarray, recipe: DialogueResidualRecipe) -> np.ndarray:
    """二つの推定の、STFTの係数ごとに振幅の小さい方の係数から戻した信号。時間軸・左右・長さは ``music`` のまま。"""
    from scipy.signal import istft

    recipe.validate()
    music = np.asarray(music, dtype=np.float64)
    instrumental = np.asarray(instrumental, dtype=np.float64)
    if music.ndim != 2 or music.shape != instrumental.shape or music.shape[0] == 0:
        raise DialogueResidualError("合成する二つの推定の形が一致しません")
    if not (np.isfinite(music).all() and np.isfinite(instrumental).all()):
        raise DialogueResidualError("合成する推定に有限でない値があります")
    frames, channels = music.shape
    out = np.zeros_like(music)
    for channel, (a, b) in enumerate(zip(_stft(instrumental, recipe), _stft(music, recipe))):
        chosen = np.where(np.abs(a) <= np.abs(b), a, b)
        _, restored = istft(chosen, nperseg=recipe.n_fft, noverlap=recipe.n_fft - recipe.hop)
        length = min(restored.size, frames)
        out[:length, channel] = restored[:length]
    if not np.isfinite(out).all():
        raise DialogueResidualError("合成の結果に有限でない値があります")
    return out


def local_regions(probabilities: np.ndarray, start_frame: int, end_frame: int, rate: int,
                  recipe: DialogueResidualRecipe, *, dilation_windows: int | None = None) -> list[list[int]]:
    """cueの中で台詞が残る時間（cue先頭からのframeの [開始, 終了)）。

    タグ付けの1秒窓（窓 ``i`` は原本の [i, i+1) 秒）のうちSpeech確率が ``speech_gate`` 以上の窓を、前後
    ``dilation_windows``（既定は宣言の ``local_dilation_windows``）窓に広げてつなぎ、cueの内側に切る。
    該当が無ければ空。
    """
    recipe.validate()
    if dilation_windows is None:
        dilation_windows = recipe.local_dilation_windows
    probabilities = np.asarray(probabilities)
    if not 0 <= start_frame < end_frame:
        raise DialogueResidualError("cueの区間が不正です")
    first = max(0, start_frame // rate)
    last = min(probabilities.shape[0], -(-end_frame // rate))
    hot = [i for i in range(first, last) if float(probabilities[i, SPEECH_CLASS]) >= recipe.speech_gate]
    regions: list[list[int]] = []
    for i in hot:
        low = max(start_frame, (i - dilation_windows) * rate) - start_frame
        high = min(end_frame, (i + 1 + dilation_windows) * rate) - start_frame
        if high <= low:
            continue
        if regions and low <= regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], high)
        else:
            regions.append([low, high])
    return regions


def local_weight(frames: int, regions, rate: int, ramp_seconds: float) -> np.ndarray:
    """区間の内側で1、区間の外へ ``ramp_seconds`` で直線に0へ下がる重み（長さ ``frames``）。"""
    weight = np.zeros(frames, dtype=np.float64)
    ramp = max(1, int(round(ramp_seconds * rate)))
    distance = np.full(frames, np.inf)
    index = np.arange(frames)
    for low, high in regions:
        if not 0 <= low < high <= frames:
            raise DialogueResidualError("局所の区間がcueの外にあります")
        outside = np.where(index < low, low - index, np.where(index >= high, index - high + 1, 0))
        distance = np.minimum(distance, outside)
    inside = distance == 0
    weight[inside] = 1.0
    near = (distance > 0) & (distance <= ramp)
    weight[near] = 1.0 - distance[near] / (ramp + 1)
    return weight


def processing_plan(probabilities, start_frame, end_frame, rate, recipe):
    """Additional speech cleanup never supplies permission to use the mixture.

    A musical voice conflict retains the separated base and requires a different
    specialist. A low speech score only bypasses this extra cleanup stage.
    """
    from lss.bgm import cue_semantics
    probabilities = np.asarray(probabilities)
    if (probabilities.ndim != 2 or probabilities.shape[1] <= max(cue_semantics.SINGING_CLASSES)
            or not np.isfinite(probabilities).all() or np.any((probabilities < 0)|(probabilities > 1))):
        raise DialogueResidualError("Speech planning needs finite class probabilities")
    protected = []
    safe = probabilities.copy()
    for i in range(max(0,start_frame//rate),min(len(safe),(end_frame+rate-1)//rate)):
        if np.max(safe[i,list(cue_semantics.SINGING_CLASSES)]) >= cue_semantics.default_recipe().singing_probability:
            a,b=max(start_frame,i*rate)-start_frame,min(end_frame,(i+1)*rate)-start_frame
            if protected and protected[-1][1] == a:
                protected[-1][1] = b
            else:
                protected.append([a,b])
            safe[i,SPEECH_CLASS] = 0.
    repair = local_regions(safe,start_frame,end_frame,rate,recipe)
    # Dilation is inference context, never authority to erase a musical voice.
    # Subtract protected windows after expansion as well as before it.
    for lo, hi in protected:
        repair = [[a,b] for first,last in repair
                  for a,b in ((first,min(last,lo)),(max(first,hi),last)) if a < b]
    kept = []
    cursor=0
    for a,b in repair:
        if a>cursor:kept.append([cursor,a])
        cursor=b
    if cursor<end_frame-start_frame:kept.append([cursor,end_frame-start_frame])
    observed_end = min(end_frame, len(probabilities)*rate)
    unknown = [[max(0,observed_end-start_frame),end_frame-start_frame]] if observed_end < end_frame else []
    return {"policy":"separated-base-extra-speech-only-v2","regions":repair,
            "retain_separated_regions":kept,"musical_voice_regions":protected,
            "unknown_regions":unknown,
            "musical_voice_state":"requires_voice_preserving_specialist" if protected else "not_observed",
            "original_mixture_authority":"none","absence_authority":"none"}


def local_consensus(music: np.ndarray, instrumental: np.ndarray, regions, rate: int,
                    recipe: DialogueResidualRecipe) -> np.ndarray:
    """台詞の残る時間だけ ``consensus`` へ置き換え、その外は ``music`` のまま（重みが0のsampleはbit一致）。"""
    music = np.asarray(music, dtype=np.float64)
    if not regions:
        return music
    combined = consensus(music, instrumental, recipe)
    weight = local_weight(music.shape[0], regions, rate, recipe.local_ramp_seconds)[:, None]
    out = music.copy()
    touched = weight[:, 0] > 0
    out[touched] = (1.0 - weight[touched]) * music[touched] + weight[touched] * combined[touched]
    return out


def bounded_local_consensus(music: np.ndarray, instrumental: np.ndarray, regions, rate: int,
                            recipe: DialogueResidualRecipe) -> np.ndarray:
    """Only admitted speech windows; cosine ramps never extend into clean music.

    Keep the legacy renderer above for exact replay of existing accepted assets.
    Compute spectral consensus with the whole available context, then apply it
    only inside these windows, without changing gain or unrelated samples.
    """
    music = np.asarray(music, dtype=np.float64)
    if not regions:
        return music
    combined = consensus(music, instrumental, recipe)
    weight = np.zeros(len(music))
    for low, high in regions:
        if not 0 <= low < high <= len(music):
            raise DialogueResidualError("Speech region is outside the cue")
        weight[low:high] = 1
        fade = min(round(rate * recipe.local_ramp_seconds), (high-low)//2)
        if fade > 1:
            curve = .5-.5*np.cos(np.linspace(0, np.pi, fade))
            if low:
                weight[low:low+fade] = curve
            if high < len(music):
                weight[high-fade:high] = curve[::-1]
    out = music.copy()
    touched = weight > 0
    out[touched] = music[touched] + (combined[touched]-music[touched])*weight[touched, None]
    return out


def region_consensus(music: np.ndarray, instrumental: np.ndarray, spans, rate: int,
                     recipe: DialogueResidualRecipe) -> np.ndarray:
    """Shared support rule for new repairs; legacy renderers stay replayable."""
    from lss.separation import regions
    music = np.asarray(music, dtype=np.float64)
    if not spans:
        return music.copy()
    instrumental = np.asarray(instrumental, dtype=np.float64)
    if instrumental.shape != music.shape:
        raise DialogueResidualError("Repair input is not aligned with music")
    recipe.validate()
    out = music.copy()
    # Keep the whole-cue STFT phase. Context is for estimation, not permission
    # to alter neighboring music. One FFT window is ample overlap on each side.
    for low, high in regions.context_spans(len(music), spans, recipe.n_fft, alignment=recipe.hop):
        weight = regions.inside_weight(len(music), spans, rate, recipe.local_ramp_seconds,
                                       start=low, length=high-low)
        combined = consensus(music[low:high], instrumental[low:high], recipe)
        out[low:high] = regions.blend(music[low:high], combined, weight)
    return out


def inference_regions(start_frame, end_frame, spans, recipe):
    """Output support for regional inference, including the STFT overlap only.

    The separator adds its own context later. This support is not permission to
    alter the adjacent music; composition still uses the original speech spans.
    """
    from lss.separation import regions
    recipe.validate()
    return [(start_frame+a, start_frame+b) for a, b in regions.context_spans(
        end_frame-start_frame, spans, recipe.n_fft, alignment=recipe.hop)]


def assemble_instrumental(base, start_frame, inference_spans):
    """Fill a cue with aligned regional witnesses; keep the scaffold elsewhere."""
    out = np.asarray(base, dtype=np.float64).copy()
    end_frame = start_frame+len(out)
    for (a, b), receipt in sorted(inference_spans.items()):
        lo, hi = max(a, start_frame), min(b, end_frame)
        if hi <= lo:
            continue
        audio = read_span(receipt)
        if len(audio) != b-a or audio.shape[1:] != out.shape[1:]:
            raise DialogueResidualError("Regional witness layout changed")
        out[lo-start_frame:hi-start_frame] = audio[lo-a:hi-a]
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_span(values: np.ndarray, rate: int, target: Path) -> dict:
    """伴奏のcue部分をFLAC PCM_24で保存する。1を超える時だけ2の冪で縮め、その倍率を記録する。"""
    import soundfile as sf

    from lss.output import writer

    values = np.asarray(values, dtype=np.float64)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    shift = writer.headroom_shift_for(peak)
    scale = writer.headroom_scale(shift)
    target = Path(target)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        sf.write(str(temporary), np.clip(values * scale, -1.0, np.nextafter(1.0, 0.0)), rate,
                 subtype="PCM_24", format="FLAC")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "sha256": _sha256(target), "headroom_shift": shift,
            "frames": int(values.shape[0]), "sample_rate": int(rate)}


def read_span(item: dict) -> np.ndarray:
    """保存した伴奏のcue部分を、記録と照合して元の尺度で読む。"""
    import soundfile as sf

    from lss.output import writer

    path = Path(item["path"])
    if _sha256(path) != item["sha256"]:
        raise DialogueResidualError(f"保存した伴奏が記録と一致しません: {path}")
    values, rate = sf.read(str(path), dtype="float64", always_2d=True)
    if rate != int(item["sample_rate"]) or values.shape[0] != int(item["frames"]) or values.shape[1] != 2:
        raise DialogueResidualError(f"保存した伴奏の時間軸・左右が記録と一致しません: {path}")
    return values / writer.headroom_scale(int(item["headroom_shift"]))


def _run_groups(signal_path, destination, groups, sf, shutil, rate, total, separate, progress, writer_pool, pending):
    for index, (low, high, members) in enumerate(groups):
        first, last = members[0][0], members[-1][1]
        crop, _ = sf.read(str(signal_path), start=low, frames=high - low, dtype="float32", always_2d=True)
        work = destination / f".work-{first}-{last}"
        store = work / "store"
        if work.exists():
            shutil.rmtree(work)
        store.mkdir(parents=True)
        crop_path = work / "crop.wav"
        sf.write(str(crop_path), crop, rate, subtype="FLOAT")
        estimate, estimate_rate = separate(crop_path, store)
        estimate = np.asarray(estimate, dtype=np.float64)
        if estimate.ndim == 2 and estimate.shape[0] == 2 and estimate.shape[1] != 2:
            estimate = estimate.T
        if estimate_rate != rate or estimate.shape != (high - low, 2) or not np.isfinite(estimate).all():
            raise DialogueResidualError(
                f"伴奏の出力が信号の時間軸と一致しません: {estimate.shape} @ {estimate_rate}Hz")
        for first, last in members:
            part = estimate[first - low:last - low]
            pending.append(((first, last), low, high, writer_pool.submit(
                write_span, part, rate, destination / f"instrumental-{first}-{last}.flac")))
        shutil.rmtree(work, ignore_errors=True)
        if progress is not None:
            progress({"phase": "separating", "completed_blocks": index + 1, "total_blocks": len(groups)})


def run_instrumental_spans(signal_path: Path, destination: Path, spans, *, separate: Callable,
                           recipe: DialogueResidualRecipe,
                           progress: Callable[[dict], None] | None = None) -> dict:
    """信号の各cue区間を文脈付きで伴奏modelへ通し、cueの部分を保存する。

    ``separate(crop_path, store_dir)`` は伴奏modelの推論（`lss.harness.infer.msst_separate` を包んだもの）で、
    ``(推定 (channels, frames), 周波数)`` を返す。信号と同じ周波数・同じ長さでなければ失敗する。
    """
    import shutil

    import soundfile as sf

    recipe.validate()
    signal_path, destination = Path(signal_path).resolve(), Path(destination).resolve()
    before = _sha256(signal_path)
    info = sf.info(str(signal_path))
    if info.channels != 2:
        raise DialogueResidualError("伴奏へ通す信号は2chである必要があります")
    rate, total = int(info.samplerate), int(info.frames)
    destination.mkdir(parents=True, exist_ok=True)
    items = []
    ordered = sorted(set((int(a), int(b)) for a, b in spans))
    groups = []
    for first, last in ordered:
        low, high = context_span(first, last, total, rate, recipe)
        if groups and low <= groups[-1][1]:
            groups[-1][1] = max(high, groups[-1][1])
            groups[-1][2].append((first, last))
        else:
            groups.append([low, high, [(first, last)]])
    # FLACの保存は次の区間の推論（GPU）と重ねる。保存する音と順序は変えない。
    from concurrent.futures import ThreadPoolExecutor

    writer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dialogue-residual-write")
    pending: list[tuple[tuple[int, int], int, int, object]] = []
    try:
        _run_groups(signal_path, destination, groups, sf, shutil, rate, total, separate, progress, writer_pool, pending)
        for span, low, high, future in pending:
            items.append({"span": [span[0], span[1]], "context": [low, high], **future.result()})
    finally:
        writer_pool.shutdown(wait=True, cancel_futures=True)
    if _sha256(signal_path) != before:
        raise DialogueResidualError("推論中に信号が変わりました")
    return {"signal_sha256": before, "frames": total, "channels": 2, "sample_rate": rate,
            "spans": [item["span"] for item in items], "instrumental": items,
            "inference_groups": [[a, b] for a, b, _ in groups],
            "inference_frames": sum(b-a for a, b, _ in groups),
            "requested_frames": sum(b-a for a, b in ordered)}
