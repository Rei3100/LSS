"""欲しい劇伴と判定した区間（cue）を、保留候補の音声から組み立てて、利用者のフォルダへ保存する。

公開する音は新しく推論しない。`episode_cues` が保存した保留候補の音楽の出力（2回目・歌唱回収を採った候補は
その表現）を、保存時の2の冪の縮小を戻して区間どおりに連結する。候補どうしは隙間なく同じ時間軸で並ぶので、
連結は元の全長の信号の同じ区間と同じ値になる（候補ごとの表現が替わる位置は、保留候補の境目と同じ）。
そのため保存先のファイルが消えても、保留候補が残っていれば推論なしで同じ音を作り直せる。

保存先は ``<原本のフォルダ>/LSS/<原本名> - BGM/``。名前は原本での開始時刻（v12。話の中の順番は使わない）。来歴は FLAC の tag
（原本のsha256・区間・判定・根拠・Recipe）に持たせ、同じcueが既にあれば音を確かめて再利用し、別の内容を
無言で上書きしない。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf
from mutagen.flac import FLAC

from lss.core import audio_identity
from lss.output import writer

SOURCE_TYPE = "bgm_cue"
FOLDER_SUFFIX = " - BGM"
PUBLICATION_POLICY = "held_candidate_music_concatenation_v1"
_BLOCK = 1 << 18


class CuePublicationError(ValueError):
    """公開するcueを組み立てられない・保存できない。"""


def folder_for(source_path: Path) -> Path:
    source_path = Path(source_path)
    return writer.output_dir_for(source_path) / f"{source_path.stem}{FOLDER_SUFFIX}"


def staging_folder(source_path: Path, run_key: str) -> Path:
    """新しいrunの公開cueを検証する作業領域（公開フォルダと同じボリューム）。CurrentBestの切替で公開名へ移す。"""
    from lss.bgm import current_best

    return writer.output_dir_for(Path(source_path)) / current_best.STAGING_DIR / run_key[:16]


def history_root(source_path: Path) -> Path:
    """置き換えた前の版の公開cueを残す履歴（公開フォルダと同じボリューム）。"""
    from lss.bgm import current_best

    return writer.output_dir_for(Path(source_path)) / current_best.HISTORY_DIR


# 一曲かどうか未確定（複数の曲を含みうる）cueの印。名前とtitleで利用者に見えるようにする。
UNRESOLVED_EXTENT_MARK = "区切り未確定"


def _clock(start_seconds: float) -> str:
    total = int(start_seconds)
    return f"{total // 60:02d}m{total % 60:02d}s"


def file_name(source_path: Path, start_seconds: float, *, unresolved_extent: bool = False) -> str:
    """公開名は原本での開始時刻だけで決める（v12）。前に別のcueが加わっても、同じcueの名前は変わらない。

    区間は1秒格子で重ならないので、同じ原本の公開cueの開始秒は重ならない。
    """
    mark = f" ({UNRESOLVED_EXTENT_MARK})" if unresolved_extent else ""
    return f"{Path(source_path).stem} - BGM {_clock(start_seconds)}{mark}.flac"


def title(source_path: Path, start_seconds: float, *, unresolved_extent: bool = False) -> str:
    mark = f" ({UNRESOLVED_EXTENT_MARK})" if unresolved_extent else ""
    return f"{Path(source_path).stem} BGM {_clock(start_seconds)}{mark}"


def _gain(part: dict) -> float:
    serialization = part.get("serialization") or {}
    shift = serialization.get("headroom_shift", 0)
    if isinstance(shift, bool) or not isinstance(shift, int) or not 0 <= shift <= 1024:
        raise CuePublicationError("候補の縮小の記録が不正です")
    return writer.headroom_scale(shift)


def compose(parts: list[dict], start_frame: int, end_frame: int) -> np.ndarray:
    """保留候補の音声（``source_start_frame``/``source_end_frame``/``sample_rate``/縮小の記録を持つ）から
    [start_frame, end_frame) を連結する。候補の中身は記録のsha256で確かめ、隙間・重なりがあれば失敗する。"""
    if not parts or end_frame <= start_frame:
        raise CuePublicationError("公開する区間か、その材料がありません")
    rates = {int(part["sample_rate"]) for part in parts}
    if len(rates) != 1:
        raise CuePublicationError("候補の時間軸が揃っていません")
    ordered = sorted(parts, key=lambda part: part["source_start_frame"])
    pieces, cursor = [], start_frame
    for part in ordered:
        first, last = int(part["source_start_frame"]), int(part["source_end_frame"])
        if last <= cursor or first >= end_frame:
            continue
        if first > cursor:
            raise CuePublicationError("公開する区間の中に候補の無い所があります")
        path = Path(part["path"])
        if not audio_identity.matches_record(path, part):
            raise CuePublicationError(f"候補の音声が記録と一致しません: {path}")
        low, high = cursor, min(end_frame, last)
        with sf.SoundFile(str(path)) as stream:
            if stream.samplerate != int(part["sample_rate"]) or stream.frames != last - first or stream.channels != 2:
                raise CuePublicationError(f"候補の時間軸・左右が記録と一致しません: {path}")
            stream.seek(low - first)
            block = stream.read(high - low, dtype="float64", always_2d=True)
        if block.shape != (high - low, 2):
            raise CuePublicationError(f"候補の音声が短すぎます: {path}")
        pieces.append(block / _gain(part))
        cursor = high
        if cursor >= end_frame:
            break
    if cursor < end_frame:
        raise CuePublicationError("公開する区間の終わりまで候補がありません")
    audio = np.concatenate(pieces)
    if not np.isfinite(audio).all():
        raise CuePublicationError("公開する音に有限でない値があります")
    return audio


def _existing_matches(path: Path, identity: dict, key: str) -> bool:
    try:
        tags = FLAC(path)
    except Exception:
        return False
    if tags.get("LSS_SOURCE_TYPE") != [SOURCE_TYPE] or tags.get("LSS_BGM_CUE_KEY") != [key]:
        return False
    try:
        observed = audio_identity.audio_identity(path)
    except (audio_identity.AudioIdentityError, OSError):
        return False
    # tagの長さは問わない（再生ソフトの書き戻し等）。音のsampleと時間軸・左右・精度が同じなら同じcue。
    return all(observed.get(name) == identity.get(name)
               for name in ("pcm_sha256", "frames", "samplerate", "channels", "subtype"))


def serialization_audio(audio: np.ndarray) -> tuple[np.ndarray, int, float]:
    """The exact existing publication headroom policy, also used by private checks."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    shift = writer.headroom_shift_for(peak)
    scale = writer.headroom_scale(shift)
    scaled = audio * scale
    if np.any(scaled < -1.0) or np.any(scaled >= 1.0):
        # PCMの正の最大値は1未満。ちょうど1になる時だけもう一段縮める（値の比は変えない）。
        shift += 1
        scale = writer.headroom_scale(shift)
        scaled = audio * scale
    return scaled, shift, scale


def _verify_saved_audio(path: Path, audio: np.ndarray, rate: int, scale: float) -> None:
    """Compare saved PCM with the assembled cue without allocating another full cue."""
    with sf.SoundFile(path) as stream:
        if stream.samplerate != rate or (stream.frames, stream.channels) != audio.shape:
            raise CuePublicationError("保存したcueの音が組み立てた音と一致しません")
        tolerance = 2.0 ** -23 / scale + 1e-12
        for first in range(0, len(audio), _BLOCK):
            expected = audio[first:first + _BLOCK]
            written = stream.read(len(expected), dtype="float64", always_2d=True)
            if written.shape != expected.shape or np.any(np.abs(written / scale - expected) > tolerance):
                raise CuePublicationError("保存したcueの音が組み立てた音と一致しません")


def publish(audio: np.ndarray, rate: int, destination: Path, *, key: str, title: str, album: str,
            provenance: dict, input_sha256: str, recipe_id: str, version: str) -> dict:
    """float の音を PCM_24 の FLAC として保存する。値は変えず、1を超える時だけ2の冪で縮めて記録する。

    ``key`` はcueの同一性（原本・Recipe・区間）。同じ名前に同じ ``key`` と同じ音があれば再利用し、
    違う内容なら ``(r2)`` 以降の名前にする。無言で上書きしない。
    """
    destination = Path(destination)
    scaled, shift, scale = serialization_audio(audio)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({**provenance, "serialization": {"headroom_shift": shift, "gain": scale, "subtype": "PCM_24",
                                                      "clipping": "none", "normalization": "none"}},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    temporary = None
    try:
        with NamedTemporaryFile(dir=destination.parent, prefix=".lss-bgm-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        with sf.SoundFile(str(temporary), mode="w", samplerate=rate, channels=2, format="FLAC",
                          subtype="PCM_24") as output:
            for index in range(0, len(scaled), _BLOCK):
                output.write(scaled[index:index + _BLOCK])
        tags = FLAC(temporary)
        tags["TITLE"] = title
        tags["ALBUM"] = album
        tags["LSS_SOURCE_TYPE"] = SOURCE_TYPE
        tags["LSS_STEM"] = "bgm"
        tags["LSS_INPUT_SHA256"] = input_sha256
        tags["LSS_RECIPE_ID"] = recipe_id
        tags["LSS_VERSION"] = version
        tags["LSS_BGM_CUE_KEY"] = key
        tags["LSS_BGM_PROVENANCE"] = body
        tags["LSS_OUTPUT_HEADROOM_SHIFT"] = str(shift)
        tags.save()
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        # 保存した音が、縮小を戻すと組み立てた音と量子化の範囲で一致することを確かめる。
        _verify_saved_audio(temporary, audio, rate, scale)
        identity = audio_identity.audio_identity(temporary)
        target = destination
        suffix = 1
        while True:
            if target.exists():
                if _existing_matches(target, identity, key):
                    break
                suffix += 1
                target = destination.with_name(f"{destination.stem} (r{suffix}){destination.suffix}")
                continue
            try:
                audio_identity.rename_keeping_identity(temporary, target)
                temporary = None
            except FileExistsError:
                continue
            break
        return {"path": str(target), "sha256": audio_identity.file_digest(target),
                "audio": audio_identity.audio_identity(target), "headroom_shift": shift,
                "sample_rate": rate, "frames": int(audio.shape[0]), "channels": 2, "key": key}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cue_key(source_sha256: str, recipe_id: str, start_frame: int, end_frame: int, rate: int) -> str:
    body = json.dumps({"source_sha256": source_sha256, "recipe_id": recipe_id, "start_frame": start_frame,
                       "end_frame": end_frame, "sample_rate": rate}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()
