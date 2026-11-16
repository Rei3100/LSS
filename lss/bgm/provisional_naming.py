"""公開済みBGMのローカル命名adapter。音声tag・元run・台帳を書き換えない。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from mutagen.flac import FLAC

from lss.assets.naming import NamingError, NamingStore
from lss.bgm.cue_publication import SOURCE_TYPE
from lss.core import audio_identity

ACOUSTIC_POLICY = {"id": "stereo-power-envelope-v1", "window_seconds": 0.5,
                   "fft_frames": 4096, "active_floor_db": -65,
                   "minimum_windows": 8, "low_hz": 400, "high_hz": 2400,
                   "colour_share": 0.6, "varying_range_db": 10}


def acoustic_evidence(path: Path, asset_id: str) -> dict:
    """左右のpowerを平均。強弱と帯域だけを記述し、楽器・感情・場面は推論しない。"""
    levels, bands = [], []
    with sf.SoundFile(path) as stream:
        rate = stream.samplerate
        block = max(1, round(rate * ACOUSTIC_POLICY["window_seconds"]))
        n_fft = ACOUSTIC_POLICY["fft_frames"]
        if stream.channels != 2:
            raise NamingError("公開BGMの2ch音声が必要です")
        frequencies = np.fft.rfftfreq(n_fft, 1 / rate)
        window = np.hanning(n_fft)
        while True:
            data = stream.read(block, dtype="float64", always_2d=True)
            if len(data) < block:
                break
            if not np.isfinite(data).all():
                raise NamingError("音声に有限でない値があります")
            rms = float(np.sqrt(np.mean(data ** 2)))
            db = 20 * np.log10(max(rms, 1e-15))
            if db < ACOUSTIC_POLICY["active_floor_db"] or len(data) < n_fft:
                continue
            # 各0.5秒窓の中央4096 sample。anti-phase stereoを消さない。
            start = (len(data) - n_fft) // 2
            power = np.mean(np.abs(np.fft.rfft(data[start:start + n_fft] * window[:, None], axis=0)) ** 2, axis=1)
            total = float(power.sum())
            if total <= 0:
                continue
            levels.append(db)
            bands.append([float(power[frequencies < ACOUSTIC_POLICY["low_hz"]].sum() / total),
                          float(power[frequencies >= ACOUSTIC_POLICY["high_hz"]].sum() / total)])
    result = {"asset_id": asset_id, "measurement_policy": ACOUSTIC_POLICY,
              "active_windows": len(levels)}
    if len(levels) < ACOUSTIC_POLICY["minimum_windows"]:
        return {**result, "status": "insufficient", "reason": "insufficient_active_audio"}
    low, high = np.median(bands, axis=0)
    middle = float(np.median([1 - low_share - high_share for low_share, high_share in bands]))
    variation = float(np.percentile(levels, 90) - np.percentile(levels, 10))
    share = ACOUSTIC_POLICY["colour_share"]
    colour = ("low" if low >= share else "high" if high >= share
              else "middle" if middle >= share else "distributed")
    return {**result, "status": "measured", "colour": colour,
            "motion": "varying" if variation >= ACOUSTIC_POLICY["varying_range_db"] else "steady",
            "low_power_share": round(float(low), 6), "high_power_share": round(float(high), 6),
            "middle_power_share": round(middle, 6),
            "envelope_p90_p10_db": round(variation, 6)}


def name_published(path: Path, store: NamingStore, *, extra_evidence: dict | None = None,
                   accepted: dict | None = None) -> dict:
    """現行FLACのcue_keyとprovenanceへ結び付ける。1 Occurrenceだけで実行できる。"""
    path = Path(path).resolve()
    before = audio_identity.file_digest(path)
    try:
        tags = FLAC(path)
        if accepted is not None:
            from lss.bgm.cue_publication import cue_key
            key, provenance = accepted["key"], accepted["provenance"]
            recipe_id, input_sha256 = accepted["recipe_id"], accepted["input_sha256"]
            if key != cue_key(input_sha256, recipe_id, provenance["start_frame"], provenance["end_frame"], provenance["sample_rate"]):
                raise ValueError("CurrentBestのcue同一性と来歴が一致しません")
        else:
            key = tags["LSS_BGM_CUE_KEY"][0]
            provenance = json.loads(tags["LSS_BGM_PROVENANCE"][0])
            recipe_id, input_sha256 = tags["LSS_RECIPE_ID"][0], tags["LSS_INPUT_SHA256"][0]
        if accepted is None and tags.get("LSS_SOURCE_TYPE") != [SOURCE_TYPE]:
            raise ValueError("公開BGMではありません")
        if (not key or provenance.get("type") != "bgm_published_cue"
                or provenance["source"]["sha256"] != input_sha256):
            raise ValueError("公開来歴が一致しません")
        for name in ("start_frame", "end_frame", "sample_rate"):
            if type(provenance[name]) is not int or provenance[name] < 0:
                raise ValueError("時刻範囲が不正です")
        if provenance["end_frame"] <= provenance["start_frame"] or provenance["sample_rate"] <= 0:
            raise ValueError("時刻範囲が不正です")
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise NamingError(f"公開BGMの名称入力が不正です: {path}") from exc
    asset_id = f"bgm-cue:{key}"
    audio = audio_identity.audio_identity(path)
    if accepted is not None and audio["pcm_sha256"] != accepted["audio"]["pcm_sha256"]:
        raise NamingError("CurrentBestと名称入力の音声が一致しません")
    if (audio["frames"] != provenance["end_frame"] - provenance["start_frame"]
            or audio["samplerate"] != provenance["sample_rate"] or audio["channels"] != 2):
        raise NamingError("公開BGMの音声と来歴の時間軸が一致しません")
    if extra_evidence is not None and not isinstance(extra_evidence, dict):
        raise NamingError("追加命名証拠はobjectが必要です")
    previous = store.read(asset_id)
    cached = (previous["revisions"][-1].get("input_evidence") or {}) if previous else {}
    acoustic = cached.get("acoustic") or {}
    if (cached.get("audio_pcm_sha256") != audio["pcm_sha256"]
            or acoustic.get("measurement_policy") != ACOUSTIC_POLICY):
        acoustic = acoustic_evidence(path, asset_id)
    evidence = {**(extra_evidence or {}), "acoustic": acoustic}
    # signalの同一性もevidenceへ入れる。場所やfile tagの違いは名称を変えない。
    evidence["audio_pcm_sha256"] = audio["pcm_sha256"]
    identity = {"cue_key": key, "source_sha256": provenance["source"]["sha256"],
                "start_frame": provenance["start_frame"], "end_frame": provenance["end_frame"],
                "sample_rate": provenance["sample_rate"], "recipe_id": recipe_id}
    artifact = {"path": str(path), "sha256": before, "audio": audio,
                "publication_provenance": provenance}
    if audio_identity.file_digest(path) != before:
        raise NamingError("命名中に公開BGMが変更されました")
    state = store.update(asset_id, evidence, identity=identity, artifact=artifact)
    current = state["revisions"][-1]
    return {"asset_id": asset_id, "path": str(path), "naming_path": str(store.path_for(asset_id)),
            "title": current["display_title"], "base_title": current["title"], "kind": current["kind"],
            "is_official": current["is_official"], "revision_id": current["revision_id"],
            "evidence": current["evidence"], "source_sha256": identity["source_sha256"],
            "source_unchanged": True, "duration_seconds": audio["frames"] / audio["samplerate"]}
