"""FLAC出力の生成、検証、衝突回避、耐久公開。

store_dir(MSSTがfilename_template="{instr}"で書いた生ステム)から、元音源のフォルダ直下
`LSS\\`へ最終成果物を作る。保存先はSOURCEごとに独立して決まり、同じバッチでも
別のフォルダになり得る。対応コンテナの代表アートワークもFLAC出力へ継承する。
"""
from __future__ import annotations

import json

import base64
import binascii
import math
import os
import shutil
import struct
import threading
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf
from mutagen import File as MutagenFile
from mutagen.flac import FLAC, Picture
from mutagen.id3 import PictureType
from mutagen.mp4 import MP4Cover
from mutagen.oggvorbis import OggVorbis

from lss.core import session
from lss.output.scan import scan_track_file
from lss.output.tags import TagValue, build_tags, provenance_tags
from lss.output import finalizer
from lss.output.filenames import component

OUTPUT_ROOT_NAME = "LSS"
STEM_SOURCE_EXTENSIONS = frozenset({".wav", ".flac"})
SILENCE_THRESHOLD_LINEAR = 10 ** (-60.0 / 20.0)  # -60dBFS(全ステム共通・I27)
OUTPUT_HEADROOM_RATIO = 1.05
OUTPUT_OVERHEAD_BYTES_PER_STEM = 1 << 20
TRIM_SCAN_BLOCK_FRAMES = 1 << 18
OUTPUT_WRITE_BLOCK_FRAMES = 1 << 18
OUTPUT_SERIALIZATION_REVISION = 3

_VORBIS_COMMENT_TYPES = (FLAC, OggVorbis)
_EASY_KEY_MAP = {
    "artist": "ARTIST", "album": "ALBUM", "albumartist": "ALBUMARTIST",
    "date": "DATE", "genre": "GENRE", "tracknumber": "TRACKNUMBER",
    "discnumber": "DISCNUMBER", "title": "TITLE",
}


@dataclass(frozen=True)
class SourceMetadata:
    tags: dict[str, TagValue]
    picture: Picture | None


class InsufficientOutputSpaceError(RuntimeError):
    """最終FLAC一式を安全に一時生成できる空き容量がない。"""


class UnsafeSerializationError(RuntimeError):
    """浮動小数点stemを正規化・clipせず現在の出力形式へ保存できない。"""


def output_dir_for(source_path: Path) -> Path:
    """保存先はSOURCEごとに独立して決まる。同じバッチでも別ディレクトリになり得る。

    出力名が`<SOURCE名> - <パート名>.flac`で衝突しないため、SOURCEごとの
    サブフォルダは作らない。既存の出力は完了台帳が実パスを保持しており、
    この規則の変更では移動も書き換えも行わない。
    """
    return source_path.parent / OUTPUT_ROOT_NAME


def discover_stem_files(store_dir: Path) -> list[Path]:
    if not store_dir.exists():
        return []
    return sorted(
        p for p in store_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in STEM_SOURCE_EXTENSIONS
    )


def select_published_stems(stem_files: list[Path], published_stems: list[str] | None) -> list[Path]:
    """要求された成果物だけを公開対象へ残す。

    store_dirにはRecipeが内部で作った相補stem（duality系のボーカル等）も入る。
    それらはworkerの作業領域と一緒に破棄される途中生成物であって、
    ユーザーが要求した結果ではない。公開集合は呼び出し側が明示する。
    published_stemsがNoneのときは、生成物すべてが要求された成果物である。
    """
    if published_stems is None:
        return stem_files
    wanted = {name.casefold(): name for name in published_stems}
    if len(wanted) != len(published_stems):
        raise ValueError(f"公開するパート名が重複しています: {published_stems}")
    selected: dict[str, Path] = {}
    for path in stem_files:
        key = path.stem.casefold()
        if key not in wanted:
            continue
        if key in selected:
            raise ValueError(f"公開対象のパートが一意ではありません: {wanted[key]}")
        selected[key] = path
    missing = [name for key, name in wanted.items() if key not in selected]
    if missing:
        raise ValueError(f"要求されたパートの分離出力がありません: {', '.join(missing)}")
    return [selected[key] for key in wanted]


def _tag_value(values) -> TagValue | None:
    normalized = [str(value) for value in values]
    if not normalized:
        return None
    return normalized[0] if len(normalized) == 1 else normalized


def read_source_tags(source_path: Path) -> dict[str, TagValue]:
    """継承タグの読み出し。VorbisComment系(FLAC/OGG)はキーがD-8のタグ名とそのまま一致する。

    それ以外(mp3のID3・m4aのMP4タグ等)はmutagenのeasyモードで一般的な項目のみ拾う(best-effort)。
    """
    try:
        audio = MutagenFile(source_path)
    except Exception:
        return {}
    if audio is None:
        return {}

    if isinstance(audio, _VORBIS_COMMENT_TYPES) and audio.tags is not None:
        return {
            key.upper(): value
            for key, values in audio.tags.items()
            if (value := _tag_value(values)) is not None
        }

    try:
        easy = MutagenFile(source_path, easy=True)
    except Exception:
        return {}
    if easy is None or easy.tags is None:
        return {}
    result: dict[str, TagValue] = {}
    for easy_key, target_key in _EASY_KEY_MAP.items():
        values = easy.tags.get(easy_key)
        value = _tag_value(values or [])
        if value is not None:
            result[target_key] = value
    return result


def _picture_from_bytes(
    data: bytes, *, mime: str = "", picture_type: int = PictureType.COVER_FRONT,
    description: str = "",
) -> Picture | None:
    if not data:
        return None
    picture = Picture()
    picture.data = data
    picture.mime = mime or _image_mime(data)
    picture.type = picture_type
    picture.desc = description
    return picture


def _image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _front_picture(pictures: list[Picture]) -> Picture | None:
    if not pictures:
        return None
    return next((picture for picture in pictures if picture.type == PictureType.COVER_FRONT), pictures[0])


def _picture_from_asf(value) -> Picture | None:
    """ASFのWM/Pictureバイト列をFLAC Pictureへ変換する。"""
    data = value.value if hasattr(value, "value") else value
    if not isinstance(data, bytes) or len(data) < 5:
        return None
    picture_type, size = data[0], struct.unpack_from("<I", data, 1)[0]

    def read_utf16z(offset: int) -> tuple[str, int]:
        for end in range(offset, len(data) - 1, 2):
            if data[end:end + 2] == b"\0\0":
                return data[offset:end].decode("utf-16-le", errors="replace"), end + 2
        raise ValueError("WM/Pictureの文字列終端がありません")

    try:
        mime, offset = read_utf16z(5)
        description, offset = read_utf16z(offset)
    except ValueError:
        return None
    image = data[offset:offset + size]
    if len(image) != size:
        return None
    return _picture_from_bytes(image, mime=mime, picture_type=picture_type, description=description)


def read_artwork(source_path: Path) -> Picture | None:
    """対応コンテナの代表アートワークをFLACへ埋め込める形で返す。"""
    try:
        audio = MutagenFile(source_path)
    except Exception:
        return None
    if audio is None:
        return None

    pictures = list(getattr(audio, "pictures", []) or [])
    if pictures:
        return _front_picture(pictures)

    tags = getattr(audio, "tags", None)
    if tags is None:
        return None

    # MP3/WAV/AIFFなどのID3 APIC。
    getall = getattr(tags, "getall", None)
    if callable(getall):
        apic_frames = getall("APIC")
        converted = [
            _picture_from_bytes(
                frame.data, mime=frame.mime, picture_type=frame.type, description=frame.desc,
            )
            for frame in apic_frames
        ]
        picture = _front_picture([item for item in converted if item is not None])
        if picture is not None:
            return picture

    # M4A/MP4のcovr。
    covers = tags.get("covr", [])
    if covers:
        cover = covers[0]
        data = bytes(cover)
        mime = "image/png" if getattr(cover, "imageformat", None) == MP4Cover.FORMAT_PNG else _image_mime(data)
        return _picture_from_bytes(data, mime=mime)

    # Ogg Vorbis/Opusの標準METADATA_BLOCK_PICTURE。
    encoded_pictures = tags.get("metadata_block_picture", [])
    for encoded in encoded_pictures:
        try:
            raw = base64.b64decode(encoded, validate=True)
            return Picture(raw)
        except (binascii.Error, TypeError, ValueError):
            continue

    # 古いVorbisCommentのcoverartも読み出す。
    encoded_covers = tags.get("coverart", [])
    cover_mimes = tags.get("coverartmime", [])
    for index, encoded in enumerate(encoded_covers):
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, TypeError, ValueError):
            continue
        mime = cover_mimes[index] if index < len(cover_mimes) else ""
        return _picture_from_bytes(data, mime=mime)

    # WMA/ASFのWM/Picture。
    for value in tags.get("WM/Picture", []):
        picture = _picture_from_asf(value)
        if picture is not None:
            return picture
    return None


def read_source_metadata(source_path: Path) -> SourceMetadata:
    """処理開始時点の継承タグと代表アートワークを一組のスナップショットにする。

    URLから取得した素材は入れ物にタグを持たないことが多いため、取得時の記録で欠けた項目だけを補う。
    """
    tags = read_source_tags(source_path)
    try:
        from lss.acquisition.store import inherited_tags_for

        for key, value in inherited_tags_for(source_path).items():
            tags.setdefault(key, value)
    except Exception:
        # 取得の記録が読めなくても、分離と保存は元のタグだけで続ける。
        pass
    return SourceMetadata(tags, read_artwork(source_path))


def flac_subtype_for(source_subtype: str) -> str:
    """FLACが扱えるPCM深度へ寄せる(libsndfileのFLACはPCM_16/PCM_24のみ・アップビットはしない)。

    入力がPCM_32/FLOAT等FLAC非対応の深度なら、質を落とさない側のPCM_24へ丸める。
    """
    if source_subtype in ("PCM_16", "PCM_24"):
        return source_subtype
    return "PCM_24"


def headroom_shift_for(peak: float) -> int:
    """|sample|<=1へ収めるのに必要な2の冪の指数(0以上)を返す。

    分離stemは元mixが0dBFS付近まで詰まっているほど1.0を超える(実測1.04〜1.40)。
    これはモデルの正常な出力であり、失敗ではない。倍率を2の冪に限るのは
    IEEE-754上で指数だけが動く=丸め誤差ゼロで完全可逆になるため。
    peakを1.0へ写像しない(=peak正規化ではない)。
    """
    if not math.isfinite(peak) or peak < 0.0:
        raise ValueError(f"sample peakが不正です: {peak}")
    if peak <= 1.0:
        return 0
    shift = max(1, math.ceil(math.log2(peak)))
    # log2の丸めに頼らず、実際に収まることで決める。
    while peak * (2.0 ** -shift) > 1.0:
        shift += 1
    return shift


def headroom_scale(shift: int) -> float:
    if shift < 0:
        raise ValueError(f"headroom shiftが不正です: {shift}")
    return float(2.0 ** -shift)


def serialized_subtype_for(source_subtype: str, headroom_shift: int) -> str:
    """headroomを取った分だけ量子化ステップが粗くならないよう深度を保つ。

    減衰してからPCM_16へ量子化すると、LSBの絶対値が元より大きくなる(=情報が減る)。
    減衰する場合に限りPCM_24へ上げるのは装飾的なアップビットではなく、
    元の絶対量子化ステップ以下を維持するための最小の措置。

    元がPCM_24のときはFLACに上の深度が無く、減衰分(6.02dB/段)だけ量子化SN比が
    下がる。24bitの余裕(約144dB)に対する減衰なので実害は無いが、
    「絶対ステップを必ず維持する」わけではない。
    """
    subtype = flac_subtype_for(source_subtype)
    if headroom_shift > 0 and subtype == "PCM_16":
        return "PCM_24"
    return subtype


def capacity_subtype_for(source_subtype: str) -> str:
    """推論前の容量見積り用。深度昇格が起きても不足しない側へ寄せる。"""
    return serialized_subtype_for(source_subtype, headroom_shift=1)


def serialization_recipe_revision(source_subtype: str) -> dict:
    subtype = flac_subtype_for(source_subtype)
    return {
        "policy_revision": OUTPUT_SERIALIZATION_REVISION,
        "container": "FLAC",
        "subtype": subtype,
        "subtype_policy": "promote_pcm16_to_pcm24_when_headroom_shift",
        # providerが内部でリサンプルする場合、長さはその出力レート基準になる。
        # ここが約束するのは「切らない」ことで、原レートのsample数一致ではない。
        "timeline": "full_stem_length_no_trim",
        "normalization": "none",
        "limiting": "none",
        "clipping": "none",
        "dithering": "libsndfile_default",
        "range_policy": "declared_power_of_two_headroom_v1",
        "headroom_scope": "all_stems_of_publication",
        "post_write_verification": "streaming_quantization_bound_v1",
    }


def estimated_output_bytes(
    duration_seconds: float, samplerate: int, channels: int, stem_count: int, subtype: str,
) -> int:
    """FLAC一式の保守的な事前確保量。非圧縮PCM相当へ5%とファイル単位余白を足す。"""
    if duration_seconds < 0 or samplerate <= 0 or channels <= 0 or stem_count <= 0:
        raise ValueError("出力容量の見積もり条件が不正です")
    bytes_per_sample = 2 if subtype == "PCM_16" else 3
    pcm_bytes = duration_seconds * samplerate * channels * stem_count * bytes_per_sample
    return math.ceil(pcm_bytes * OUTPUT_HEADROOM_RATIO) + stem_count * OUTPUT_OVERHEAD_BYTES_PER_STEM


# One shared reservation ledger for concurrent writers (the BGM CPU lane, `lss.bgm.overlap`). A writer inside
# ``output_reservations()`` keeps the bytes it was granted reserved until its scope ends, so two threads cannot both
# pass the free-space check for the same free bytes. Outside any scope nothing is reserved, and with no reservation
# outstanding the check is exactly the serial one.
_CAPACITY = threading.Condition()
_RESERVED: dict[int, int] = {}
_SCOPE = threading.local()


@contextmanager
def output_reservations():
    """Hold every capacity this thread is granted until the block ends (the write it guards is then on disk)."""
    held = getattr(_SCOPE, "held", None)
    if held is not None:
        yield
        return
    _SCOPE.held = held = []
    try:
        yield
    finally:
        _SCOPE.held = None
        with _CAPACITY:
            for volume, size in held:
                _RESERVED[volume] -= size
                if not _RESERVED[volume]:
                    del _RESERVED[volume]
            _CAPACITY.notify_all()


def ensure_output_capacity(
    source_path: Path, duration_seconds: float, samplerate: int, channels: int,
    stem_count: int, subtype: str, *, destination_dir: Path | None = None,
) -> int:
    needed = estimated_output_bytes(duration_seconds, samplerate, channels, stem_count, subtype)
    capacity_path = Path(destination_dir).resolve() if destination_dir is not None else source_path.parent
    capacity_path.mkdir(parents=True, exist_ok=True)
    held = getattr(_SCOPE, "held", None)
    with _CAPACITY:
        while True:
            free = shutil.disk_usage(capacity_path).free
            if free < needed:
                raise InsufficientOutputSpaceError(
                    f"出力先の空き容量が不足しています: path={capacity_path}, "
                    f"free_bytes={free}, needed_bytes={needed}"
                )
            volume = os.stat(capacity_path).st_dev
            own = sum(size for key, size in held or () if key == volume)
            others = _RESERVED.get(volume, 0) - own
            # Bytes granted to another unfinished writer are not free for this one. Wait for it to finish (its file
            # is then on disk and counted by the free space), never fail where the serial order would have passed.
            if others <= 0 or free - others >= needed:
                break
            _CAPACITY.wait()
        if held is not None:
            held.append((volume, needed))
            _RESERVED[volume] = _RESERVED.get(volume, 0) + needed
    return needed


def common_trim_bounds(stem_waveforms: list[np.ndarray], threshold: float = SILENCE_THRESHOLD_LINEAR) -> tuple[int, int]:
    """全ステムの合算(ミックス)に対して1回だけ無音判定する(D-8「無音トリム」・I27)。

    stem_waveformsは各(channels, frames)で全て同じ長さである前提(同一入力から出た全ステム=I26)。
    戻り値は[start, end)の共通サンプル区間。無音が見つからなければ全長を返す。
    """
    if not stem_waveforms:
        return 0, 0
    # np.stack()は全ステムと同じ総量の巨大な一時配列を作る。長尺音声でも
    # 追加メモリを1ステム分に抑えるため、共通バッファへ逐次加算する。
    mix = np.zeros(stem_waveforms[0].shape, dtype=np.float32)
    for waveform in stem_waveforms:
        if waveform.shape != mix.shape:
            raise ValueError("ステム間の長さ・チャンネル数が一致しません")
        np.add(mix, waveform, out=mix, casting="unsafe")
    energy = np.max(np.abs(mix), axis=0)
    above = np.nonzero(energy > threshold)[0]
    if above.size == 0:
        return 0, mix.shape[-1]
    return int(above[0]), int(above[-1]) + 1


def common_trim_bounds_from_files(
    stem_files: list[Path], threshold: float = SILENCE_THRESHOLD_LINEAR, *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int, int, int, float]:
    """ステム全体を保持せず、チャンク走査で共通トリム境界・音声形式・最大sample peakを返す。

    peakは全ステムを通じた最大値。headroomは公開単位で1つに決めなければならず
    (ステムごとに別倍率だと足し戻した時に元mixへ戻らない=I26の相対関係が壊れる)、
    この走査は既に全ステムを読むためここで一緒に測る。
    """
    if not stem_files:
        raise ValueError("分離出力がありません")
    infos = [sf.info(path) for path in stem_files]
    first_info = infos[0]
    for path, info in zip(stem_files, infos):
        if info.frames <= 0:
            raise ValueError(f"分離出力が空か、不正なサンプルを含んでいます: {path}")
    for path, info in zip(stem_files[1:], infos[1:]):
        if (
            info.frames != first_info.frames or info.samplerate != first_info.samplerate
            or info.channels != first_info.channels
        ):
            raise ValueError(f"ステム間のサンプルレート・長さ・チャンネル数が一致しません: {path}")

    first_above: int | None = None
    last_above: int | None = None
    peak = 0.0
    offset = 0

    def report(completed_frames: int) -> None:
        if progress is not None:
            try:
                progress(completed_frames, first_info.frames)
            except Exception:
                # 進捗表示の失敗で、走査や成果物の保存を中断しない。
                pass

    report(0)
    with ExitStack() as resources:
        streams = [resources.enter_context(sf.SoundFile(path)) for path in stem_files]
        while offset < first_info.frames:
            block_frames = min(TRIM_SCAN_BLOCK_FRAMES, first_info.frames - offset)
            mix = np.zeros((block_frames, first_info.channels), dtype=np.float32)
            for path, stream in zip(stem_files, streams):
                data = stream.read(block_frames, dtype="float32", always_2d=True)
                if data.shape != mix.shape or not np.isfinite(data).all():
                    raise ValueError(f"分離出力が空か、不正なサンプルを含んでいます: {path}")
                if data.size:
                    peak = max(peak, float(np.max(np.abs(data))))
                np.add(mix, data, out=mix, casting="unsafe")
            if not np.isfinite(mix).all():
                raise ValueError("分離出力の合算結果が不正です")
            above = np.nonzero(np.max(np.abs(mix), axis=1) > threshold)[0]
            if above.size:
                if first_above is None:
                    first_above = offset + int(above[0])
                last_above = offset + int(above[-1]) + 1
            offset += block_frames
            report(offset)

    start, end = (0, first_info.frames) if first_above is None else (first_above, last_above)
    return start, end, first_info.samplerate, first_info.channels, peak


def resolve_output_path(dest_dir: Path, source_stem: str, stem_name: str, input_sha256: str, recipe_id: str) -> Path | None:
    """衝突回避(A19)。①入力指紋+レシピIDが両方一致→None(スキップ) ②不一致→(r2)以降の連番。無言上書きはしない。"""
    path, reused = _resolve_output(dest_dir, source_stem, stem_name, input_sha256, recipe_id)
    return None if reused else path


def _audio_matches(
    path: Path, expected_audio: tuple[int, int, int], *,
    progress: Callable[[int, int], None] | None = None,
) -> bool:
    """既存成果物を全量ロードせず、形式と全サンプルの有限性を検証する。"""
    expected_frames, expected_channels, expected_samplerate = expected_audio

    def report(completed_frames: int) -> None:
        if progress is not None:
            try:
                progress(completed_frames, expected_frames)
            except Exception:
                pass

    try:
        info = sf.info(path)
        if (
            info.frames != expected_frames or info.channels != expected_channels
            or info.samplerate != expected_samplerate
        ):
            return False
        completed = 0
        report(0)
        with sf.SoundFile(path) as stream:
            while completed < expected_frames:
                block_frames = min(TRIM_SCAN_BLOCK_FRAMES, expected_frames - completed)
                data = stream.read(block_frames, dtype="float32", always_2d=True)
                if data.shape != (block_frames, expected_channels) or not np.isfinite(data).all():
                    return False
                completed += block_frames
                report(completed)
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def _resolve_output(
    dest_dir: Path, source_stem: str, stem_name: str, input_sha256: str, recipe_id: str,
    reuse_exclusions: frozenset[Path] = frozenset(),
    expected_audio: tuple[int, int, int] | None = None,
    validation_progress: Callable[[int, int], None] | None = None,
    required_tags: dict[str, str] | None = None,
) -> tuple[Path, bool]:
    base_name = component(f"{source_stem} - {stem_name}")
    candidate = dest_dir / f"{base_name}.flac"
    suffix_n = 1
    while candidate.exists():
        existing_tags = read_source_tags(candidate)
        if (
            candidate.resolve() not in reuse_exclusions
            and existing_tags.get("LSS_INPUT_SHA256") == input_sha256
            and existing_tags.get("LSS_RECIPE_ID") == recipe_id
            and all(existing_tags.get(key) == value for key, value in (required_tags or {}).items())
        ):
            reusable = expected_audio is None or _audio_matches(
                candidate, expected_audio, progress=validation_progress,
            )
            if reusable:
                return candidate, True
        suffix_n += 1
        candidate = dest_dir / f"{base_name} (r{suffix_n}).flac"
    return candidate, False


def _ensure_serializable_range(data: np.ndarray, source_path: Path | None = None) -> None:
    if not np.isfinite(data).all():
        raise UnsafeSerializationError(f"分離出力に有限でないsampleがあります: {source_path or '<memory>'}")
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        raise UnsafeSerializationError(
            f"分離出力はPCMへ無変換で保存できる範囲を超えています: "
            f"path={source_path or '<memory>'}, sample_peak={peak:.9f}"
        )


def _verify_serialized_audio(
    source_path: Path, serialized_path: Path, *, start: int, frames: int,
    samplerate: int, channels: int, subtype: str, gain: float = 1.0,
) -> None:
    """書出し後のtimeline・有限性・量子化誤差境界をsourceと逐次比較する。

    宣言済みheadroom倍率`gain`を適用した期待値と比べる。倍率は2の冪のため
    float上で完全一致し、許容誤差は量子化ステップのみで足りる。
    """
    info = sf.info(serialized_path)
    if (
        info.frames != frames or info.samplerate != samplerate or info.channels != channels
        or info.subtype != subtype
    ):
        raise UnsafeSerializationError(f"直列化後の音声layoutが一致しません: {serialized_path}")
    bits = 16 if subtype == "PCM_16" else 24
    tolerance = (1.0 / (2 ** (bits - 1))) * 1.01
    with sf.SoundFile(source_path) as source, sf.SoundFile(serialized_path) as serialized:
        source.seek(start)
        remaining = frames
        while remaining:
            block_frames = min(OUTPUT_WRITE_BLOCK_FRAMES, remaining)
            original = source.read(block_frames, dtype="float64", always_2d=True)
            restored = serialized.read(block_frames, dtype="float64", always_2d=True)
            if original.shape != (block_frames, channels) or restored.shape != original.shape:
                raise UnsafeSerializationError(f"直列化後のsample数が一致しません: {serialized_path}")
            if gain != 1.0:
                original = original * gain
            _ensure_serializable_range(original, source_path)
            if not np.isfinite(restored).all() or float(np.max(np.abs(original - restored))) > tolerance:
                raise UnsafeSerializationError(f"直列化で許容外のsample変化が発生しました: {serialized_path}")
            remaining -= block_frames


def _stage_flac(
    directory: Path, waveform: np.ndarray, samplerate: int, subtype: str,
    tags: dict[str, TagValue], picture: Picture | None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, prefix=".lss-", suffix=".tmp", delete=False) as tmp:
        temporary_path = Path(tmp.name)
    try:
        _ensure_serializable_range(waveform)
        sf.write(temporary_path, waveform.T, samplerate, subtype=subtype, format="FLAC")
        audio = FLAC(temporary_path)
        for key, value in tags.items():
            audio[key] = value
        if picture is not None:
            audio.clear_pictures()
            audio.add_picture(picture)
        audio.save()
        # memory入力も同じhard gateを通す。向きを合わせた一時FLOAT sourceを介して
        # 共通の逐次検証を再利用する必要はないため、ここでは復号値を直接比較する。
        restored, restored_rate = sf.read(temporary_path, dtype="float64", always_2d=True)
        original = np.asarray(waveform).T
        bits = 16 if subtype == "PCM_16" else 24
        tolerance = (1.0 / (2 ** (bits - 1))) * 1.01
        if (
            restored_rate != samplerate or restored.shape != original.shape
            or not np.isfinite(restored).all()
            or float(np.max(np.abs(original - restored))) > tolerance
        ):
            raise UnsafeSerializationError(f"直列化で許容外のsample変化が発生しました: {temporary_path}")
        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_flac(
    path: Path, waveform: np.ndarray, samplerate: int, subtype: str,
    tags: dict[str, TagValue], picture: Picture | None,
) -> None:
    temporary_path = _stage_flac(path.parent, waveform, samplerate, subtype, tags, picture)
    try:
        # Windowsのrenameは既存の出力を上書きせず失敗する。
        temporary_path.rename(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _stage_flac_from_file(
    directory: Path, source_path: Path, *, start: int, frames: int,
    samplerate: int, channels: int, subtype: str,
    tags: dict[str, TagValue], picture: Picture | None, gain: float = 1.0,
) -> Path:
    """生ステムの指定範囲を全量ロードせず、一時FLACへ書き出す。

    `gain`は公開単位で宣言されたheadroom倍率(2の冪)。float32上で指数のみが動くため
    値は変質せず、全ステムへ同じ倍率を掛けることでステム間の相対関係も保たれる。
    """
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, prefix=".lss-", suffix=".tmp", delete=False) as tmp:
        temporary_path = Path(tmp.name)
    try:
        with sf.SoundFile(source_path) as source:
            if source.samplerate != samplerate or source.channels != channels:
                raise ValueError(f"分離出力が走査後に変更されました: {source_path}")
            source.seek(start)
            with sf.SoundFile(
                temporary_path, mode="w", samplerate=samplerate, channels=channels,
                subtype=subtype, format="FLAC",
            ) as destination:
                remaining = frames
                while remaining:
                    block_frames = min(OUTPUT_WRITE_BLOCK_FRAMES, remaining)
                    data = source.read(block_frames, dtype="float32", always_2d=True)
                    if data.shape != (block_frames, channels) or not np.isfinite(data).all():
                        raise ValueError(f"分離出力が走査後に変更されました: {source_path}")
                    if gain != 1.0:
                        data = data * np.float32(gain)
                    _ensure_serializable_range(data, source_path)
                    destination.write(data)
                    remaining -= block_frames
        audio = FLAC(temporary_path)
        for key, value in tags.items():
            audio[key] = value
        if picture is not None:
            audio.clear_pictures()
            audio.add_picture(picture)
        audio.save()
        _verify_serialized_audio(
            source_path, temporary_path, start=start, frames=frames,
            samplerate=samplerate, channels=channels, subtype=subtype, gain=gain,
        )
        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_recipe_output(
    mix_path: Path, store_dir: Path, *,
    recipe_id: str, model: str, model_sha256: str, score: str, version: str,
    input_sha256: str, processed_at: str | None = None,
    include_existing: bool = False, reuse_exclusions: frozenset[Path] = frozenset(),
    source_subtype: str | None = None, source_metadata: SourceMetadata | None = None,
    progress: Callable[[dict], None] | None = None,
    validate_before_publish: Callable[[], None] | None = None,
    trim_common_silence: bool = False,
    publication_state_root: Path | None = None,
    completion_progress: Callable[[Path, int, int], None] | None = None,
    config_sha256: str | None = None,
    runtime_revision: dict | None = None,
    recipe_revision: dict | None = None,
    destination_dir: Path | None = None,
    published_stems: list[str] | None = None,
    selective_receipt: dict | None = None,
    stem_combination_receipt: dict | None = None,
) -> list[Path]:
    """要求された成果物を、最終FLAC(タグ付き・尺揃え)として書き出す。

    published_stemsは、この実行で要求された成果物のパート名。store_dirにある
    それ以外の生ステムはRecipeの内部生成物として扱い、公開も完了記録もしない。
    include_existing=Trueなら再利用した出力も返し、完了記録に全ステムを含められる。
    reuse_exclusionsの出力は保全し、連番の別ファイルへ再生成する。
    共通無音の除去はtimelineを変えるため、Recipeが明示した場合だけ有効にする。
    """
    generated = discover_stem_files(store_dir)
    if not generated:
        raise ValueError(f"分離出力がありません: {store_dir}")
    # 無音判定もheadroom倍率も「公開する一式」に対して決める。破棄する内部生成物の
    # ピークで公開物を余計に減衰させない(serialization革命のheadroom_scopeどおり)。
    stem_files = select_published_stems(generated, published_stems)

    def report(phase: str, **details) -> None:
        if progress is not None:
            try:
                progress({"phase": phase, **details})
            except Exception:
                # 表示側の失敗で、成果物の保存を中断しない。
                pass

    detected_start, detected_end, samplerate, channels, stem_peak = common_trim_bounds_from_files(
        stem_files,
        progress=lambda completed, total: report(
            "analyzing_outputs", completed_frames=completed, total_frames=total,
        ),
    )
    # 分離stemが1.0を超えるのは正常。宣言済みの2の冪headroomで整数PCMへ収める
    # (clip・limit・peak正規化はしない)。倍率は公開単位で1つ。
    shift = headroom_shift_for(stem_peak)
    gain = headroom_scale(shift)
    if trim_common_silence:
        start, end = detected_start, detected_end
    else:
        start, end = 0, sf.info(stem_files[0]).frames

    metadata = source_metadata or read_source_metadata(mix_path)
    source_tags = metadata.tags
    picture = metadata.picture
    subtype = serialized_subtype_for(source_subtype or sf.info(mix_path).subtype, shift)
    when = processed_at or datetime.now(timezone.utc).isoformat()
    dest_dir = Path(destination_dir).resolve() if destination_dir is not None else output_dir_for(mix_path)
    provenance = provenance_tags(
        recipe_id=recipe_id, model=model, model_sha256=model_sha256,
        processed_at=when, input_sha256=input_sha256, score=score, version=version,
        config_sha256=config_sha256, runtime_revision=runtime_revision,
        recipe_revision=recipe_revision, headroom_shift=shift,
    )
    if selective_receipt is not None:
        provenance["LSS_SELECTIVE_COMPOSITION"] = json.dumps(selective_receipt, sort_keys=True, allow_nan=False)
    if stem_combination_receipt is not None:
        # 組み合わせの部品・利得・規則・部品ステムのhash（`lss.separation.stem_combination`）。
        provenance["LSS_STEM_COMBINATION"] = json.dumps(
            stem_combination_receipt, sort_keys=True, ensure_ascii=False, allow_nan=False,
        )
    required_tags = {
        key: value for key, value in provenance.items()
        if key in {
            "LSS_MODEL_SHA256", "LSS_CONFIG_SHA256", "LSS_RUNTIME_ID", "LSS_RECIPE_REVISION",
            "LSS_OUTPUT_HEADROOM_SHIFT",
        }
    }

    written: list[Path] = []
    staged: list[tuple[Path, Path]] = []
    published: list[tuple[Path, str]] = []

    def rollback_published() -> None:
        # A path can be replaced or edited after publication. Only remove the
        # exact bytes this transaction published, as restart recovery does.
        for path, digest in published:
            if path.is_file() and session.input_sha256_of(path) == digest:
                path.unlink(missing_ok=True)

    publication_journal: Path | None = None
    total_stems = len(stem_files)
    report("writing_outputs", completed_stems=0, total_stems=total_stems)
    try:
        # 全ステムを一時ファイルへ完成させてから公開する。2本目以降のタグ付け等が
        # 失敗しても、先に作れたステムだけを最終成果物として残さない。
        for completed_stems, path in enumerate(stem_files, start=1):
            stem_name = path.stem
            out_path, reused = _resolve_output(
                dest_dir, mix_path.stem, stem_name, input_sha256, recipe_id, reuse_exclusions,
                expected_audio=(end - start, channels, samplerate),
                validation_progress=lambda completed, total: report(
                    "checking_existing_output", stem=stem_name,
                    completed_frames=completed, total_frames=total,
                ),
                required_tags=required_tags,
            )
            if reused:
                if include_existing:
                    written.append(out_path)
                report(
                    "writing_outputs", completed_stems=completed_stems,
                    total_stems=total_stems, stem=stem_name, reused=True,
                )
                continue

            from lss.output.tags import inherited_tags, new_tags
            tags = {**inherited_tags(source_tags), **new_tags(stem_name), **provenance}
            temporary_path = _stage_flac_from_file(
                out_path.parent, path, start=start, frames=end - start,
                samplerate=samplerate, channels=channels, subtype=subtype, tags=tags,
                picture=picture, gain=gain,
            )
            staged.append((temporary_path, out_path))
            # 測るのは量子化後の実ファイル。BGMと同じ仕上げ・検証サービスを使う。
            finalizer.finalize_staged(temporary_path, tags=tags, picture=picture, scanner=scan_track_file)
            written.append(out_path)
            report(
                "writing_outputs", completed_stems=completed_stems,
                total_stems=total_stems, stem=stem_name, reused=False,
            )

        if validate_before_publish is not None:
            validate_before_publish()

        if publication_state_root is not None:
            publication_journal = session.prepare_publication(
                input_sha256,
                recipe_id,
                source_path=mix_path,
                outputs=written,
                staged=staged,
                state_root=publication_state_root,
            )

        for temporary_path, out_path in staged:
            digest = session.input_sha256_of(temporary_path)
            temporary_path.rename(out_path)
            published.append((out_path, digest))
        if publication_journal is not None:
            session.complete_publication(
                publication_journal,
                state_root=publication_state_root,
                progress=completion_progress,
            )
        return written
    except OSError:
        if publication_journal is not None:
            # 公開開始後のI/O障害はjournalから再試行する。保存済み成果を消さない。
            raise
        rollback_published()
        raise
    except Exception:
        rollback_published()
        if publication_journal is not None and publication_journal.exists():
            session.abort_publication(publication_journal, state_root=publication_state_root)
        raise
    finally:
        if publication_journal is None or not publication_journal.exists():
            for temporary_path, _ in staged:
                temporary_path.unlink(missing_ok=True)
