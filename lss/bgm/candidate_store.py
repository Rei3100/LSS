"""未確定の曲候補を、単独ファイルでも出典が残る形で安全に保存する。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from mutagen.flac import FLAC
import numpy as np
import soundfile as sf

from lss.core import audio_identity
from lss.output import writer


CANDIDATE_FORMAT_REVISION = 2
HEADROOM_CANDIDATE_FORMAT_REVISION = 3
COPY_BLOCK_FRAMES = 1 << 18


@dataclass(frozen=True)
class CandidateOrigin:
    source_sha256: str
    signal_sha256: str
    separation_recipe_id: str
    runtime_id: str
    activity_recipe_id: str
    start_frame: int
    end_frame: int
    sample_rate: int
    time_offset_numerator: int = 0
    time_offset_denominator: int = 1
    serialization_policy: str = "native_range"

    def validate(self) -> None:
        if (not isinstance(self.serialization_policy, str)
                or self.serialization_policy not in {"native_range", "declared_power_of_two_headroom_v1"}):
            raise ValueError("候補のserialization policyが不正です")
        for field in ("source_sha256", "signal_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"候補の{field}が不正です")
        for field in ("separation_recipe_id", "runtime_id", "activity_recipe_id"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f"候補の{field}がありません")
        for field in ("start_frame", "end_frame", "sample_rate"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"候補の{field}は整数である必要があります")
        if self.start_frame < 0 or self.end_frame <= self.start_frame or self.sample_rate <= 0:
            raise ValueError("候補区間または時間基準が不正です")
        if (
            not isinstance(self.time_offset_numerator, int) or isinstance(self.time_offset_numerator, bool)
            or not isinstance(self.time_offset_denominator, int) or isinstance(self.time_offset_denominator, bool)
            or self.time_offset_denominator <= 0
        ):
            raise ValueError("候補の元メディアの時間基準が不正です")

    @property
    def format_revision(self) -> int:
        return (HEADROOM_CANDIDATE_FORMAT_REVISION if self.serialization_policy != "native_range"
                else CANDIDATE_FORMAT_REVISION)

    def to_dict(self) -> dict:
        payload = asdict(self)
        # 既存v2候補の同一性と来歴を変えない。
        if self.serialization_policy == "native_range":
            payload.pop("serialization_policy")
        return payload

    @property
    def candidate_id(self) -> str:
        self.validate()
        # ファイルの移動やsource_idの表記変更は、候補の同一性を変えない。
        payload = {"format_revision": self.format_revision, **self.to_dict()}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"bgm-candidate-v{self.format_revision}-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_SIGNAL_HASHES: dict[tuple[str, int, int], str] = {}


def signal_sha256(path: Path) -> str:
    """内容SHA-256。同じ所在・大きさ・更新時刻のファイルは1回だけ読む。

    候補の保存は同じ全長の信号（解析stem・原本の復号）から何十も切り出し、切り出しごとに信号が
    変わっていないかを確かめる。実番組1話でこの確認だけで全長stemを272回・111GB読み直していた
    （`.state/validation/bgm-work-avoidance-r1/`）。書き換え・置き換えは大きさか更新時刻を変えるので、
    それらが同じ間は前の結果を使い、変わっていれば読み直す。
    """
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _SIGNAL_HASHES.get(key)
    if cached is None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        cached = digest.hexdigest()
        if len(_SIGNAL_HASHES) > 4096:
            _SIGNAL_HASHES.clear()
        _SIGNAL_HASHES[key] = cached
    return cached


def _sha256(path: Path) -> str:
    return signal_sha256(path)


def _verify_provenance(tags: dict, stored: dict, expected: dict, origin: CandidateOrigin) -> None:
    """表記や所在の変更は許容し、出典・時刻・未確定状態の改変は再利用しない。"""
    if not isinstance(stored, dict):
        raise ValueError("保存済み候補の来歴が不正です")
    for key in ("start_seconds", "end_seconds"):
        if type(stored.get(key)) not in (int, float):
            raise ValueError(f"保存済み候補の来歴が不正です: {key}")
    for key in ("schema_version", "candidate_id", "origin", "start_seconds", "end_seconds", "timebase",
                "semantic_role", "identity_state", "publication_state"):
        if stored.get(key) != expected[key]:
            raise ValueError(f"保存済み候補の来歴が一致しません: {key}")
    try:
        CandidateOrigin(**stored["origin"]).validate()
        # JSONのNaN等を、未使用の追加情報の中にあっても来歴として受け入れない。
        json.dumps(stored, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("保存済み候補の来歴が不正です") from exc
    for key in ("source_id", "source_path"):
        if not isinstance(stored.get(key), str) or not stored[key].strip():
            raise ValueError(f"保存済み候補の来歴が不正です: {key}")
    if not isinstance(stored.get("evidence"), dict):
        raise ValueError("保存済み候補の来歴が不正です: evidence")
    if (tags.get("LSS_SOURCE_TYPE") != ["bgm_candidate"]
            or tags.get("LSS_INPUT_SHA256") != [origin.source_sha256]):
        raise ValueError("保存済み候補の出典タグが一致しません")


def _verify_crop(signal_path: Path, candidate_path: Path, origin: CandidateOrigin) -> None:
    shift = 0
    if origin.serialization_policy != "native_range":
        try:
            shift = int(FLAC(candidate_path)["LSS_OUTPUT_HEADROOM_SHIFT"][0])
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError("候補のheadroom宣言がありません") from exc
        if shift < 0 or shift > 1024:
            raise ValueError("候補のheadroom宣言が不正です")
    scale = writer.headroom_scale(shift)
    with sf.SoundFile(signal_path) as signal, sf.SoundFile(candidate_path) as candidate:
        if (
            signal.samplerate != origin.sample_rate
            or signal.channels != 2
            or candidate.samplerate != origin.sample_rate
            or signal.channels != candidate.channels
            or candidate.frames != origin.end_frame - origin.start_frame
            or origin.end_frame > signal.frames
        ):
            raise ValueError("候補の時間軸・チャンネルが元信号と一致しません")
        if candidate.subtype != writer.serialized_subtype_for(signal.subtype, shift):
            raise ValueError("候補の保存精度が元信号と保存規則に一致しません")
        signal.seek(origin.start_frame)
        quantum = 2.0 ** (-15 if candidate.subtype == "PCM_16" else -23)
        remaining = candidate.frames
        while remaining:
            count = min(remaining, COPY_BLOCK_FRAMES)
            expected = signal.read(count, dtype="float64", always_2d=True) * scale
            actual = candidate.read(count, dtype="float64", always_2d=True)
            if (
                len(expected) != count or actual.shape != expected.shape
                or not np.isfinite(expected).all() or not np.isfinite(actual).all()
                or np.any(np.abs(expected - actual) > quantum)
            ):
                raise ValueError("候補の音声が元区間と保存量子化の範囲で一致しません")
            remaining -= count


def _crop_headroom(signal_path: Path, origin: CandidateOrigin) -> int:
    peak, positive_peak = 0.0, 0.0
    with sf.SoundFile(signal_path) as signal:
        if signal.samplerate != origin.sample_rate or origin.end_frame > signal.frames:
            raise ValueError("候補区間が元信号の時間軸を超えています")
        signal.seek(origin.start_frame)
        remaining = origin.end_frame - origin.start_frame
        while remaining:
            block = signal.read(min(remaining, COPY_BLOCK_FRAMES), dtype="float64", always_2d=True)
            if not len(block) or not np.isfinite(block).all():
                raise ValueError("候補区間をクリップせず完全に保存できません")
            peak = max(peak, float(np.max(np.abs(block))))
            positive_peak = max(positive_peak, float(np.max(block)))
            remaining -= len(block)
    shift = writer.headroom_shift_for(peak)
    # 正の1.0は整数PCMの最大値より大きい。端点もクリップせず保存する。
    if positive_peak * writer.headroom_scale(shift) >= 1.0:
        shift += 1
    return shift


def materialize_candidate(
    signal_path: Path,
    destination: Path,
    *,
    origin: CandidateOrigin,
    source_id: str,
    source_path: Path,
    evidence: dict,
) -> dict:
    """未公開の候補をstagingから確定する。同一候補の再実行は検証して再利用する。"""
    origin.validate()
    signal_path, destination = Path(signal_path).resolve(), Path(destination).resolve()
    if destination.suffix.lower() != ".flac" or destination in {signal_path, Path(source_path).resolve()}:
        raise ValueError("候補の保存先は元信号と異なるFLACである必要があります")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("候補の出典IDがありません")
    if not isinstance(evidence, dict):
        raise ValueError("候補の証拠は辞書である必要があります")
    metadata = {
        "schema_version": origin.format_revision,
        "candidate_id": origin.candidate_id,
        "origin": origin.to_dict(),
        "source_id": source_id,
        "source_path": str(Path(source_path).resolve()),
        "start_seconds": origin.start_frame / origin.sample_rate + origin.time_offset_numerator / origin.time_offset_denominator,
        "end_seconds": origin.end_frame / origin.sample_rate + origin.time_offset_numerator / origin.time_offset_denominator,
        "timebase": "source_media_time_from_decoded_audio_origin",
        "semantic_role": "unresolved_music_like",
        "identity_state": "unresolved",
        "publication_state": "not_published",
        "evidence": evidence,
    }
    if _sha256(signal_path) != origin.signal_sha256:
        raise ValueError("候補の元信号が記録したidentityと一致しません")
    info = sf.info(signal_path)
    if info.channels != 2 or info.samplerate != origin.sample_rate or origin.end_frame > info.frames:
        raise ValueError("候補の元信号は同じ時間軸の完成ステレオである必要があります")
    shift = _crop_headroom(signal_path, origin) if origin.serialization_policy != "native_range" else 0
    scale = writer.headroom_scale(shift)
    if origin.serialization_policy != "native_range":
        metadata["serialization"] = {"range_policy": origin.serialization_policy,
                                     "headroom_shift": shift, "gain": scale,
                                     "headroom_scope": "both_channels_of_candidate",
                                     "clipping": "none", "normalization": "none"}
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    crop_verified = False
    try:
        if not destination.exists():
            with NamedTemporaryFile(dir=destination.parent, prefix=".lss-candidate-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
            with sf.SoundFile(signal_path) as signal:
                if signal.samplerate != origin.sample_rate or origin.end_frame > signal.frames:
                    raise ValueError("候補区間が元信号の時間軸を超えています")
                subtype = writer.serialized_subtype_for(signal.subtype, shift)
                writer.ensure_output_capacity(
                    signal_path, (origin.end_frame - origin.start_frame) / origin.sample_rate,
                    origin.sample_rate, signal.channels, 1, subtype,
                    destination_dir=destination.parent,
                )
                signal.seek(origin.start_frame)
                with sf.SoundFile(temporary, mode="w", samplerate=origin.sample_rate, channels=signal.channels, format="FLAC", subtype=subtype) as output:
                    remaining = origin.end_frame - origin.start_frame
                    while remaining:
                        block = signal.read(min(remaining, COPY_BLOCK_FRAMES), dtype="float64", always_2d=True) * scale
                        if not len(block) or not np.isfinite(block).all() or np.any(block < -1) or np.any(block >= 1):
                            raise ValueError("候補区間をクリップせず完全に保存できません")
                        output.write(block)
                        remaining -= len(block)
            tags = FLAC(temporary)
            tags["LSS_SOURCE_TYPE"] = "bgm_candidate"
            tags["LSS_CANDIDATE_ID"] = origin.candidate_id
            tags["LSS_INPUT_SHA256"] = origin.source_sha256
            tags["LSS_CANDIDATE_PROVENANCE"] = encoded
            if origin.serialization_policy != "native_range":
                tags["LSS_OUTPUT_HEADROOM_SHIFT"] = str(shift)
            tags.save()
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            _verify_crop(signal_path, temporary, origin)
            if _sha256(signal_path) != origin.signal_sha256:
                raise ValueError("候補保存中に元信号が変更されました")
            verified = temporary.stat()
            # Windowsのrenameは既存ファイルを上書きしない。
            try:
                temporary.rename(destination)
                # 検証した直後に名前を変えただけで、大きさ・更新時刻が同じなら同じbytesなので復号し直さない。
                moved = destination.stat()
                crop_verified = (moved.st_size, moved.st_mtime_ns) == (verified.st_size, verified.st_mtime_ns)
            except FileExistsError:
                pass
        tags = FLAC(destination)
        if tags.get("LSS_CANDIDATE_ID") != [origin.candidate_id]:
            raise FileExistsError(f"保存先に別の候補があります: {destination}")
        try:
            provenance = tags["LSS_CANDIDATE_PROVENANCE"]
            if len(provenance) != 1:
                raise ValueError("来歴は単独の記録である必要があります")
            stored = json.loads(provenance[0])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("保存済み候補の来歴が不正です") from exc
        _verify_provenance(tags, stored, metadata, origin)
        if origin.serialization_policy != "native_range" and (
                stored.get("serialization") != metadata["serialization"]
                or tags.get("LSS_OUTPUT_HEADROOM_SHIFT") != [str(shift)]):
            raise ValueError("保存済み候補のheadroom来歴が一致しません")
        if not crop_verified:
            _verify_crop(signal_path, destination, origin)
        if _sha256(signal_path) != origin.signal_sha256:
            raise ValueError("候補の再利用確認中に元信号が変更されました")
        info = sf.info(destination)
        record = {
            "path": str(destination), "sha256": _sha256(destination),
            # 試聴でtagが書き戻されても、同じ音の候補として読み戻せるようにする。
            "audio": audio_identity.audio_identity(destination),
            "candidate_asset_id": origin.candidate_id,
            "sample_rate": info.samplerate, "frames": info.frames,
            "channels": info.channels, "subtype": info.subtype,
            "representation": "noncanonical_staged_candidate",
            "publication_state": "not_published",
            "source_signal_sha256": origin.signal_sha256,
            "source_start_frame": origin.start_frame, "source_end_frame": origin.end_frame,
        }
        if origin.serialization_policy != "native_range":
            record["serialization"] = metadata["serialization"]
        return record
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
