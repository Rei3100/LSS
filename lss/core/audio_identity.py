"""復号したsampleそのものの同一性。容器のmetadataを音の変化と混同しない。

外部の再生ソフトが再生履歴・タグを書き戻すと、音を変えずにファイルのbytesが変わる。
byte hashだけで判断すると、検証済みの良い出力を失ったものとして扱い、作り直してしまう。
ここでは復号したsampleの並びと時間軸・左右・精度を別に記録し、両者の差を区別する。
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from lss.core import flac_frames

SCHEMA_VERSION = 1
METHOD = "decoded-stream-float64-v1"
_BLOCK = 1 << 16
# SAME_DECODED_AUDIO_STREAM_CHANGED: 再生ソフト・利用者が可逆で符号化し直した（圧縮率・frameの区切りが違う）
# 出力。音のstreamのbytes長は変わるが、復号sampleの並び・時間軸・左右・精度・容器の種類が記録と同じなので、
# LSSが作った音そのもの。作り直し（推論・(r2)）の理由にしない。metadataだけの変更とは区別して返す。
REUSABLE_STATES = frozenset({"INTACT", "EXTERNAL_METADATA_ONLY_CHANGE", "SAME_DECODED_AUDIO_STREAM_CHANGED"})


class AudioIdentityError(ValueError):
    """復号した音の同一性を求められない。"""


def _read_exact(stream, size: int) -> bytes | None:
    data = stream.read(size)
    return data if len(data) == size else None


def _flac_layout(stream, size: int) -> dict | None:
    """metadata blockを読み飛ばし、音のstreamが始まる位置と残りの長さを返す。"""
    while True:
        header = _read_exact(stream, 4)
        if header is None:
            return None
        stream.seek(int.from_bytes(header[1:], "big"), 1)
        if header[0] & 0x80:
            break
    offset = stream.tell()
    return None if offset > size else {"container": "flac", "audio_offset": offset,
                                       "audio_bytes": size - offset}


def _wav_layout(stream, size: int) -> dict | None:
    """chunkを最後までたどり、data chunkの宣言長を返す。

    後ろへ足された正しいchunkはmetadataの増減として扱うが、解釈できない末尾bytesが
    残る容器は、metadataだけの変更と言えないのでNoneを返す。
    """
    if _read_exact(stream, 4) is None or _read_exact(stream, 4) != b"WAVE":
        return None
    layout = None
    while stream.tell() < size:
        header = _read_exact(stream, 8)
        if header is None:
            return None
        length = int.from_bytes(header[4:], "little")
        offset = stream.tell()
        if length == 0xFFFFFFFF or offset + length > size:
            return None
        if header[:4] == b"data" and layout is None:
            layout = {"container": "wav", "audio_offset": offset, "audio_bytes": length}
        stream.seek(length + (length & 1), 1)
    return layout if stream.tell() in (size, size + 1) else None


def container_layout(path: Path) -> dict | None:
    """音のstreamの位置と長さ。未対応・不整合な容器はNoneとして、原因を断定しない。"""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        magic = _read_exact(stream, 4)
        if magic == b"fLaC":
            return _flac_layout(stream, size)
        if magic == b"RIFF":
            return _wav_layout(stream, size)
    return None


def _decoded_identity(path: Path) -> dict:
    """Decoder observation for diagnostics; does not establish complete FLAC identity."""
    path = Path(path)
    try:
        with sf.SoundFile(path) as stream:
            if stream.channels <= 0 or stream.samplerate <= 0:
                raise AudioIdentityError(f"復号できる音声ではありません: {path}")
            digest = hashlib.sha256()
            # libsndfile decodes FLAC to sample * 2**-(bits-1) as float64 and to sample << (32-bits) as int32, both
            # exact, so the int32 stream `flac_frames.stream_is_exact` compares with ffmpeg is this decode * 2**31.
            # Derived only while every scaled value is an exact int32; otherwise that check decodes on its own.
            s32 = hashlib.sha256() if str(stream.format) == "FLAC" else None
            frames = 0
            while True:
                block = stream.read(_BLOCK, dtype="float64", always_2d=True)
                if block.shape[0] == 0:
                    break
                if not np.isfinite(block).all():
                    raise AudioIdentityError(f"有限でないsampleがあります: {path}")
                digest.update(np.ascontiguousarray(block).tobytes())
                if s32 is not None:
                    scaled = block * 2147483648.0
                    if (np.all(scaled == np.trunc(scaled)) and np.all(scaled >= -2147483648.0)
                            and np.all(scaled <= 2147483647.0)):
                        s32.update(scaled.astype("<i4").tobytes())
                    else:
                        s32 = None
                frames += block.shape[0]
            _DERIVED_S32.value = (str(path), s32.hexdigest() if s32 is not None else None)
            identity = {"schema_version": SCHEMA_VERSION, "method": METHOD,
                         "pcm_sha256": digest.hexdigest(), "frames": frames,
                         "samplerate": int(stream.samplerate), "channels": int(stream.channels),
                         "subtype": str(stream.subtype)}
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioIdentityError(f"音声として復号できません: {path}") from exc
    layout = container_layout(path)
    if layout is not None:
        identity.update(container=layout["container"], audio_bytes=layout["audio_bytes"])
    return identity


# The int32 PCM hash derived by the last `_decoded_identity` of this thread (path, digest or None).
_DERIVED_S32 = threading.local()
_IDENTITIES: dict[tuple[str, int, int], dict] = {}
# (byte SHA-256, size) of a file whose identity was established by decoding -> that identity.
_BY_CONTENT: dict[tuple[str, int], dict] = {}
# The BGM CPU lane (`lss.bgm.overlap`) asks for identities from several threads; every cache access holds this lock.
_IDENTITIES_LOCK = threading.Lock()


def _stamp(path: Path) -> tuple[str, int, int] | None:
    try:
        path = Path(path).resolve()
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_size, stat.st_mtime_ns)


def audio_identity(path: Path) -> dict:
    """Identity of a complete playable stream, excluding tags and metadata.

    Public callers (CurrentBest, listening and publication) must not inherit
    libsndfile's declared-prefix interpretation of malformed FLAC. Diagnostics
    can observe that prefix privately, but it cannot authorize reusable audio.

    The same file (resolved path, size, mtime_ns) is decoded once per process: publication, listening and
    CurrentBest ask for the identity of the same bytes several times. Any rewrite, including a tag write-back,
    changes the size or mtime and is decoded again. Callers get a copy and cannot alter the cache.
    """
    path = Path(path)
    stamp = _stamp(path)
    with _IDENTITIES_LOCK:
        cached = _IDENTITIES.get(stamp) if stamp is not None else None
    if cached is not None:
        return dict(cached)
    # A copy, or a file moved under a new mtime, has the same bytes as one already verified: every decoder reads
    # nothing but the bytes, so their identity and completeness are the same. Look it up by content before decoding.
    content = file_digest(path) if stamp is not None else None
    with _IDENTITIES_LOCK:
        known = _BY_CONTENT.get((content, stamp[1])) if content is not None else None
    if known is not None:
        identity = dict(known)
    else:
        _DERIVED_S32.value = None
        identity = _decoded_identity(path)
        derived = getattr(_DERIVED_S32, "value", None)
        _DERIVED_S32.value = None
        s32 = derived[1] if derived is not None and derived[0] == str(path) else None
        if not stream_complete(path, identity, libsndfile_s32_sha256=s32):
            raise AudioIdentityError(f"FLAC stream is incomplete or decoders disagree: {path}")
    # 復号中に書き換わったファイルの結果は残さない。
    if stamp is not None and _stamp(path) == stamp:
        with _IDENTITIES_LOCK:
            if len(_IDENTITIES) > 4096:
                _IDENTITIES.clear()
            _IDENTITIES[stamp] = dict(identity)
            if content is not None:
                if len(_BY_CONTENT) > 4096:
                    _BY_CONTENT.clear()
                _BY_CONTENT[(content, stamp[1])] = dict(identity)
    return identity


def rename_keeping_identity(source: Path, destination: Path) -> None:
    """``source.rename(destination)``。名前変更後も大きさと更新時刻が同じなら、求め済みの同一性を引き継ぐ。"""
    source, destination = Path(source), Path(destination)
    stamp = _stamp(source)
    with _IDENTITIES_LOCK:
        entry = _IDENTITIES.get(stamp) if stamp is not None else None
    source.rename(destination)
    if entry is not None:
        moved = _stamp(destination)
        if moved is not None and moved[1:] == stamp[1:]:
            with _IDENTITIES_LOCK:
                _IDENTITIES[moved] = dict(entry)


def material_record(path: Path) -> dict:
    """所在・byte hash・（復号できるなら）sample同一性を持つ素材の記録。"""
    path = Path(path).resolve()
    digest = file_digest(path)
    if digest is None:
        raise AudioIdentityError(f"素材を読めません: {path}")
    record = {"path": str(path), "sha256": digest}
    try:
        record["audio"] = audio_identity(path)
    except (AudioIdentityError, OSError):
        pass
    return record


def same_audio(recorded: object, observed: dict) -> bool:
    """記録した同一性と今の音が同じか。記録がない・方式が違うものは同じと言わない。"""
    if (not isinstance(recorded, dict) or recorded.get("schema_version") != SCHEMA_VERSION
            or recorded.get("method") != METHOD):
        return False
    # 音のstreamの長さまで一致するときだけ、差はtag等のmetadataに限られると言える。
    if recorded.get("container") is None or observed.get("container") is None:
        return False
    return all(recorded.get(key) == observed.get(key) for key in
               ("pcm_sha256", "frames", "samplerate", "channels", "subtype",
                "container", "audio_bytes"))


def same_decoded_audio(recorded: object, observed: dict) -> bool:
    """容器の音のstreamのbytesは違っても、復号した音が記録と同じか（可逆の符号化し直し）。

    sampleのhash・frame数・sample rate・左右・精度に加え、容器の種類も記録と同じことを求める（FLACの出力が
    別の容器に変わったものは同じ出力と言わない）。容器を記録していない旧記録・未対応の容器は同じと言わない。"""
    if (not isinstance(recorded, dict) or recorded.get("schema_version") != SCHEMA_VERSION
            or recorded.get("method") != METHOD or not recorded.get("pcm_sha256")):
        return False
    if recorded.get("container") is None or observed.get("container") is None:
        return False
    return all(recorded.get(key) == observed.get(key) for key in
               ("pcm_sha256", "frames", "samplerate", "channels", "subtype", "container"))


def stream_complete(path: Path, observed: dict, *, libsndfile_s32_sha256: str | None = None) -> bool:
    """容器の音のstreamが宣言どおりに終わるか。FLACは宣言の後ろに再生されるframeを置けるので確かめる。

    ``libsndfile_s32_sha256``: libsndfile's int32 decode hash of the same bytes when already known."""
    return observed.get("container") != "flac" or flac_frames.stream_is_exact(
        path, libsndfile_s32_sha256=libsndfile_s32_sha256)


def header_matches(path: Path, recorded: object) -> bool:
    """復号せずに、時間軸・左右・精度が記録と一致するかだけを見る前段の絞り込み。"""
    if not isinstance(recorded, dict):
        return False
    try:
        info = sf.info(str(path))
    except (sf.LibsndfileError, RuntimeError, OSError):
        return False
    return (recorded.get("frames") == info.frames and recorded.get("samplerate") == info.samplerate
            and recorded.get("channels") == info.channels and recorded.get("subtype") == info.subtype)


def file_digest(path: Path) -> str | None:
    """ファイルのbyte hash。読めない場合はNone。"""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(_BLOCK), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def matches_record(path: Path, item: dict) -> bool:
    """記録した{sha256, audio}と同じ素材か。tagだけが書き戻された素材を別物にしない。"""
    if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
        return False
    return diagnose(path, recorded_sha256=item["sha256"], recorded_audio=item.get("audio"),
                    observed_sha256=file_digest(path)) in REUSABLE_STATES


def diagnose(path: Path, *, recorded_sha256: str, recorded_audio: object,
             observed_sha256: str | None) -> str:
    """byte差の原因を、音の変化・容器だけの変化・確かめられないに分ける。"""
    if observed_sha256 == recorded_sha256:
        return "INTACT"
    if not Path(path).is_file():
        return "MISSING"
    if recorded_audio is None:
        # 公開時にsampleの同一性を残していない出力は、bytes差の原因を証明できない。
        return "UNVERIFIABLE_WITHOUT_RECORDED_AUDIO_IDENTITY"
    try:
        observed = _decoded_identity(path)
    except (AudioIdentityError, OSError):
        return "UNREADABLE"
    # 音のstreamの長さが同じでも、FLACの宣言の後ろに再生されるframeを置けば（長さは別の圧縮で詰めて揃う）
    # libsndfileの復号は変わらない。metadataだけの変更と言うにも、streamが宣言どおりに終わることを求める。
    if same_audio(recorded_audio, observed) and stream_complete(path, observed):
        return "EXTERNAL_METADATA_ONLY_CHANGE"
    if not isinstance(recorded_audio, dict):
        return "UNVERIFIABLE_WITHOUT_RECORDED_AUDIO_IDENTITY"
    if (same_decoded_audio(recorded_audio, observed) and observed.get("container") == "flac"
            and not same_audio(recorded_audio, observed) and stream_complete(path, observed)):
        return "SAME_DECODED_AUDIO_STREAM_CHANGED"
    return ("AUDIO_CHANGED" if recorded_audio.get("pcm_sha256") != observed.get("pcm_sha256")
            else "CONTAINER_CHANGE_NOT_PROVEN_METADATA_ONLY")
