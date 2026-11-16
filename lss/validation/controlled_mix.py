"""実音源componentから、正解区間が自明なBGM controlled corpusを構築する。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import stat
from math import gcd
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Any

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly

from lss.validation.bgm_corpus import load_bgm_corpus


SCHEMA_VERSION = 1
_PLAN_KEYS = frozenset({
    "schema_version", "corpus_id", "revision", "description", "sample_rate", "channels", "cases",
})
_CASE_KEYS = frozenset({
    "case_id", "description", "duration_frames", "expected_outcome", "attributes", "components",
})
_COMPONENT_REQUIRED_KEYS = frozenset({
    "component_id", "path", "source_start_frame", "source_end_frame", "start_frame",
    "gain_db", "fade_frames", "occurrence",
})
_COMPONENT_KEYS = _COMPONENT_REQUIRED_KEYS | {"reference_role", "ducking"}
_DUCKING_KEYS = frozenset({"start_frame", "end_frame", "depth_db", "ramp_frames"})
_OCCURRENCE_KEYS = frozenset({
    "occurrence_id", "semantic_role", "wanted", "novel", "identity_state",
    "cue_identity_id", "audio_version_id",
})
_REFERENCE_ROLES = frozenset({"music", "dialogue", "sfx"})


class InvalidControlledMixError(ValueError):
    """controlled mix planまたは生成信号が検証可能な形になっていない。"""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidControlledMixError(f"JSONキーが重複しています: {key}")
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Any, keys: frozenset[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidControlledMixError(f"{context}はオブジェクトである必要があります")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise InvalidControlledMixError(f"{context}のキーが不正です: missing={missing}, extra={extra}")
    return value


def _required_keys(
    value: Any, required: frozenset[str], allowed: frozenset[str], context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidControlledMixError(f"{context}はオブジェクトである必要があります")
    missing = sorted(required - set(value))
    extra = sorted(set(value) - allowed)
    if missing or extra:
        raise InvalidControlledMixError(f"{context}のキーが不正です: missing={missing}, extra={extra}")
    return value


def _positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InvalidControlledMixError(f"{context}は正の整数である必要があります")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InvalidControlledMixError(f"{context}は0以上の整数である必要があります")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidControlledMixError(f"{context}が未指定です")
    return value.strip()


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidControlledMixError(f"controlled mix planを読めません: {path}") from exc
    plan = _exact_keys(raw, _PLAN_KEYS, "plan")
    if plan["schema_version"] != SCHEMA_VERSION:
        raise InvalidControlledMixError(f"未対応のschemaです: {plan['schema_version']}")
    _text(plan["corpus_id"], "corpus_id")
    _text(plan["revision"], "revision")
    _text(plan["description"], "description")
    _positive_int(plan["sample_rate"], "sample_rate")
    if plan["channels"] not in (1, 2):
        raise InvalidControlledMixError("channelsは1または2である必要があります")
    if not isinstance(plan["cases"], list) or not plan["cases"]:
        raise InvalidControlledMixError("casesは空でない配列である必要があります")
    return plan


def _component_audio(
    raw: dict[str, Any], *, plan_dir: Path, sample_rate: int, channels: int,
    pin_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    component_id = _text(raw["component_id"], "component_id")
    source_path = Path(_text(raw["path"], f"{component_id}.path"))
    if not source_path.is_absolute():
        source_path = plan_dir / source_path
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise InvalidControlledMixError(f"component sourceが存在しません: {source_path}")
    source_start = _nonnegative_int(raw["source_start_frame"], f"{component_id}.source_start_frame")
    source_end = _positive_int(raw["source_end_frame"], f"{component_id}.source_end_frame")
    start_frame = _nonnegative_int(raw["start_frame"], f"{component_id}.start_frame")
    fade_frames = _nonnegative_int(raw["fade_frames"], f"{component_id}.fade_frames")
    gain_db = raw["gain_db"]
    if (
        not isinstance(gain_db, (int, float)) or isinstance(gain_db, bool)
        or not math.isfinite(gain_db)
    ):
        raise InvalidControlledMixError(f"{component_id}.gain_dbは有限な数値である必要があります")
    reference_role = raw.get("reference_role")
    if reference_role is not None and reference_role not in _REFERENCE_ROLES:
        raise InvalidControlledMixError(
            f"{component_id}.reference_roleが不正です: {reference_role!r}"
        )

    # Pathを音声読込後に再openしてhashすると、差し替え時に波形と来歴が別世代に
    # なり得る。選択した実体を管理下へ1回だけコピーし、その固定コピーをhashと
    # 音声読込の共通sourceにする。コピー中の元実体変更もstatでhard failする。
    pin_dir.mkdir(parents=True, exist_ok=True)
    pinned_path: Path | None = None
    try:
        with source_path.open("rb") as source_stream:
            before = os.fstat(source_stream.fileno())
            digest = hashlib.sha256()
            with NamedTemporaryFile(dir=pin_dir, prefix="component-", suffix=".source", delete=False) as pinned:
                pinned_path = Path(pinned.name)
                for chunk in iter(lambda: source_stream.read(1 << 20), b""):
                    digest.update(chunk)
                    pinned.write(chunk)
                pinned.flush()
                os.fsync(pinned.fileno())
            after = os.fstat(source_stream.fileno())
        try:
            path_after = source_path.stat()
        except OSError as exc:
            raise InvalidControlledMixError(
                f"component sourceがコピー中に変更されました: {source_path}"
            ) from exc
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity_fields) or any(
            getattr(after, field) != getattr(path_after, field) for field in identity_fields
        ):
            raise InvalidControlledMixError(
                f"component sourceがコピー中に変更されました: {source_path}"
            )
        if stat.S_ISDIR(before.st_mode):
            raise InvalidControlledMixError(f"component sourceが音声ファイルではありません: {source_path}")

        with sf.SoundFile(pinned_path) as stream:
            if source_end <= source_start or source_end > len(stream):
                raise InvalidControlledMixError(
                    f"{component_id}のsource区間が範囲外です: {source_start}:{source_end}/{len(stream)}"
                )
            source_rate = stream.samplerate
            source_channels = stream.channels
            stream.seek(source_start)
            data = stream.read(source_end - source_start, dtype="float64", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise InvalidControlledMixError(f"component sourceを読めません: {source_path}") from exc
    finally:
        if pinned_path is not None:
            pinned_path.unlink(missing_ok=True)
    if data.shape != (source_end - source_start, source_channels) or not np.isfinite(data).all():
        raise InvalidControlledMixError(f"component sourceが途中で変化または破損しました: {source_path}")
    if source_channels == 1 and channels == 2:
        data = np.repeat(data, 2, axis=1)
    elif source_channels != channels:
        raise InvalidControlledMixError(
            f"{component_id}のchannel変換は未対応です: {source_channels} -> {channels}"
        )
    if source_rate != sample_rate:
        divisor = gcd(source_rate, sample_rate)
        data = resample_poly(data, sample_rate // divisor, source_rate // divisor, axis=0)
    data *= 10.0 ** (float(gain_db) / 20.0)
    if fade_frames:
        fade = min(fade_frames, len(data) // 2)
        if fade:
            ramp = np.arange(fade, dtype=np.float64) / fade
            data[:fade] *= ramp[:, None]
            data[-fade:] *= ramp[::-1, None]
    ducking = _ducking_windows(raw.get("ducking"), component_id=component_id, frames=len(data))
    if ducking:
        data *= _ducking_envelope(ducking, len(data))[:, None]
    provenance = {
        "component_id": component_id,
        "source_path": str(source_path),
        "source_sha256": digest.hexdigest(),
        "source_sample_rate": source_rate,
        "source_channels": source_channels,
        "source_frames": source_end - source_start,
        "source_start_frame": source_start,
        "source_end_frame": source_end,
        "output_start_frame": start_frame,
        "output_end_frame": start_frame + len(data),
        "gain_db": float(gain_db),
        "fade_frames": fade_frames,
        "ducking": ducking,
        "reference_role": reference_role,
    }
    return data, provenance


def _ducking_windows(raw: Any, *, component_id: str, frames: int) -> list[dict[str, Any]]:
    """台詞の下でBGMが下がる区間の指定を検証して返す。位置はcomponent自身の先頭からのframe。"""
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise InvalidControlledMixError(f"{component_id}.duckingは空でない配列である必要があります")
    windows = []
    previous_end = None
    for index, item in enumerate(raw):
        context = f"{component_id}.ducking[{index}]"
        window = _exact_keys(item, _DUCKING_KEYS, context)
        start = _nonnegative_int(window["start_frame"], f"{context}.start_frame")
        end = _positive_int(window["end_frame"], f"{context}.end_frame")
        ramp = _nonnegative_int(window["ramp_frames"], f"{context}.ramp_frames")
        depth = window["depth_db"]
        if (not isinstance(depth, (int, float)) or isinstance(depth, bool)
                or not math.isfinite(depth) or depth > 0):
            raise InvalidControlledMixError(f"{context}.depth_dbは0以下の有限な数値である必要があります")
        if end <= start:
            raise InvalidControlledMixError(f"{context}の区間の順序が不正です")
        if previous_end is not None and start < previous_end:
            raise InvalidControlledMixError(f"{context}が前の区間と重なっています")
        previous_end = end
        windows.append({"start_frame": start, "end_frame": end,
                        "depth_db": float(depth), "ramp_frames": ramp})
    if windows[-1]["end_frame"] > frames:
        raise InvalidControlledMixError(f"{component_id}.duckingがcomponentの範囲を超えています")
    return windows


def _ducking_envelope(windows: list[dict[str, Any]], frames: int) -> np.ndarray:
    """下げ幅の包絡。区間の外は1.0、区間内は指定の深さ、その手前と後ろで直線に移る。"""
    envelope = np.ones(frames, dtype=np.float64)
    for window in windows:
        start, end = window["start_frame"], window["end_frame"]
        ramp = window["ramp_frames"]
        depth = 10.0 ** (window["depth_db"] / 20.0)
        envelope[start:end] = np.minimum(envelope[start:end], depth)
        if ramp:
            head = np.arange(min(ramp, start), 0, -1, dtype=np.float64)
            if head.size:
                values = depth + (1.0 - depth) * (head / ramp)
                envelope[start - head.size:start] = np.minimum(envelope[start - head.size:start], values)
            tail_size = min(ramp, max(frames - end, 0))
            if tail_size:
                values = depth + (1.0 - depth) * (np.arange(1, tail_size + 1, dtype=np.float64) / ramp)
                envelope[end:end + tail_size] = np.minimum(envelope[end:end + tail_size], values)
    return envelope


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_controlled_corpus(plan_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """新規directoryへ音声・construction manifest・BGM corpus manifestを全体公開する。"""
    plan_path = Path(plan_path).resolve()
    output_dir = Path(output_dir).resolve()
    plan = _load_plan(plan_path)
    if output_dir.exists():
        raise FileExistsError(f"出力先は新規directoryである必要があります: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}-", suffix=".tmp"))
    try:
        audio_dir = stage / "audio"
        audio_dir.mkdir()
        pin_dir = stage / ".pinned-sources"
        sample_rate = plan["sample_rate"]
        channels = plan["channels"]
        case_records = []
        corpus_records = [{
            "type": "manifest",
            "schema_version": 1,
            "corpus_id": plan["corpus_id"],
            "revision": plan["revision"],
            "description": plan["description"],
        }]
        case_ids: set[str] = set()
        for case_raw in plan["cases"]:
            case = _exact_keys(case_raw, _CASE_KEYS, "case")
            case_id = _text(case["case_id"], "case_id")
            if case_id in case_ids:
                raise InvalidControlledMixError(f"case_idが重複しています: {case_id}")
            case_ids.add(case_id)
            duration_frames = _positive_int(case["duration_frames"], f"{case_id}.duration_frames")
            if not isinstance(case["components"], list) or not case["components"]:
                raise InvalidControlledMixError(f"{case_id}.componentsは空でない配列である必要があります")
            raw_reference_roles = [item.get("reference_role") for item in case["components"]]
            has_clean_references = any(role is not None for role in raw_reference_roles)
            if has_clean_references and any(role is None for role in raw_reference_roles):
                raise InvalidControlledMixError(
                    f"{case_id}でclean referenceを作る場合は全componentへreference_roleが必要です"
                )
            mix = np.zeros((duration_frames, channels), dtype=np.float64)
            reference_mixes = (
                {role: np.zeros_like(mix) for role in sorted(_REFERENCE_ROLES)}
                if has_clean_references else None
            )
            occurrences = []
            components = []
            component_ids: set[str] = set()
            for component_raw in case["components"]:
                component = _required_keys(
                    component_raw, _COMPONENT_REQUIRED_KEYS, _COMPONENT_KEYS,
                    f"{case_id}.component",
                )
                data, component_record = _component_audio(
                    component, plan_dir=plan_path.parent,
                    sample_rate=sample_rate, channels=channels, pin_dir=pin_dir,
                )
                component_id = component_record["component_id"]
                if component_id in component_ids:
                    raise InvalidControlledMixError(
                        f"{case_id}のcomponent_idが重複しています: {component_id}"
                    )
                component_ids.add(component_id)
                start = component_record["output_start_frame"]
                end = component_record["output_end_frame"]
                if end > duration_frames:
                    raise InvalidControlledMixError(
                        f"{case_id}.{component_id}がcase範囲外です: {start}:{end}/{duration_frames}"
                    )
                mix[start:end] += data
                if reference_mixes is not None:
                    reference_mixes[component_record["reference_role"]][start:end] += data
                occurrence_raw = component["occurrence"]
                if occurrence_raw is not None:
                    occurrence = _exact_keys(
                        occurrence_raw, _OCCURRENCE_KEYS, f"{case_id}.{component_id}.occurrence",
                    )
                    occurrences.append({
                        **occurrence,
                        "start_frame": start,
                        "end_frame": end,
                    })
                component_record["occurrence"] = occurrence_raw
                components.append(component_record)
            if pin_dir.exists() and not any(pin_dir.iterdir()):
                pin_dir.rmdir()
            peak = float(np.max(np.abs(mix)))
            if not math.isfinite(peak) or peak >= 1.0:
                raise InvalidControlledMixError(
                    f"{case_id}はclippingします。gainを明示的に調整してください: peak={peak:.6f}"
                )
            output_path = audio_dir / f"{case_id}.flac"
            sf.write(output_path, mix, sample_rate, subtype="PCM_24", format="FLAC")
            references = {}
            reference_sum_max_abs_error = None
            if reference_mixes is not None:
                reconstructed = sum(reference_mixes.values())
                reference_sum_max_abs_error = float(np.max(np.abs(mix - reconstructed)))
                if reference_sum_max_abs_error != 0.0:
                    raise InvalidControlledMixError(
                        f"{case_id}のclean reference和がmixと一致しません: "
                        f"max_abs_error={reference_sum_max_abs_error}"
                    )
                reference_dir = stage / "references" / case_id
                reference_dir.mkdir(parents=True)
                for role, role_signal in reference_mixes.items():
                    role_peak = float(np.max(np.abs(role_signal)))
                    if not math.isfinite(role_peak) or role_peak >= 1.0:
                        raise InvalidControlledMixError(
                            f"{case_id}.{role} referenceはclippingします: peak={role_peak:.6f}"
                        )
                    reference_path = reference_dir / f"{role}.flac"
                    sf.write(
                        reference_path, role_signal, sample_rate,
                        subtype="PCM_24", format="FLAC",
                    )
                    references[role] = {
                        "path": f"references/{case_id}/{role}.flac",
                        "sha256": _sha256(reference_path),
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "frames": duration_frames,
                        "sample_peak": role_peak,
                    }
            case_records.append({
                "case_id": case_id,
                "output_path": f"audio/{case_id}.flac",
                "output_sha256": _sha256(output_path),
                "sample_rate": sample_rate,
                "channels": channels,
                "frames": duration_frames,
                "sample_peak": peak,
                "components": components,
                "references": references,
                "reference_sum_max_abs_error": reference_sum_max_abs_error,
            })
            attributes = list(case["attributes"])
            if not all(isinstance(item, str) and item.strip() for item in attributes):
                raise InvalidControlledMixError(f"{case_id}.attributesが不正です")
            corpus_records.append({
                "type": "case",
                "case_id": case_id,
                "description": _text(case["description"], f"{case_id}.description"),
                "source": {
                    "path": f"audio/{case_id}.flac",
                    "sha256": case_records[-1]["output_sha256"],
                    "sample_rate": sample_rate,
                    "frames": duration_frames,
                },
                "expected_outcome": case["expected_outcome"],
                "attributes": [*attributes, "controlled_mix"],
                "occurrences": occurrences,
            })

        construction_path = stage / "construction.json"
        _write_json(construction_path, {
            "schema_version": SCHEMA_VERSION,
            "plan_path": str(plan_path),
            "plan_sha256": _sha256(plan_path),
            "builder_sha256": _sha256(Path(__file__)),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "soundfile": sf.__version__,
                "resampler": "scipy.signal.resample_poly-default-v1",
            },
            "cases": case_records,
        })
        construction_sha = _sha256(construction_path)
        for record in corpus_records[1:]:
            record["attributes"].append(f"construction_sha256={construction_sha}")
        manifest_path = stage / "manifest.jsonl"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            for record in corpus_records:
                stream.write(json.dumps(
                    record, ensure_ascii=False, sort_keys=True, allow_nan=False,
                ) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        load_bgm_corpus(manifest_path, verify_sources=True)
        stage.rename(output_dir)
        return output_dir / "manifest.jsonl", output_dir / "construction.json"
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
