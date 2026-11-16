"""音声入力の読み込み。libsndfile非対応形式は同梱FFmpegで一時WAVへ復号する。"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import subprocess

import soundfile as sf

from lss.paths import resource_root
from lss.core import session
from lss.core.scratch import ScratchDirectory

DECODE_TIMEOUT_SECONDS = 600
INPUT_PREPARATION_REVISION = 1


class DecoderUnavailableError(FileNotFoundError):
    """同梱FFmpegが見つからず、入力を復号できない。"""


class InputDecodeError(ValueError):
    """FFmpegが入力を正常なWAVへ復号できない。"""


class InputDecodeTimeoutError(TimeoutError):
    """入力の復号が制限時間内に終わらない。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_preparation_revision(source_path: Path, prepared_path: Path) -> dict:
    """実際に選ばれた復号経路をRecipe identityへ入れる。"""
    common = {
        "policy_revision": INPUT_PREPARATION_REVISION,
        "soundfile_version": sf.__version__,
        "libsndfile_version": sf.__libsndfile_version__,
    }
    if source_path.resolve() == prepared_path.resolve():
        return {**common, "method": "libsndfile_direct"}
    ffmpeg = (resource_root() / "bin" / "ffmpeg.exe").resolve()
    if not ffmpeg.is_file():
        raise DecoderUnavailableError(f"復号に使用したFFmpegを識別できません: {ffmpeg}")
    return {
        **common,
        "method": "ffmpeg_pcm_f32le",
        "ffmpeg_sha256": _sha256(ffmpeg),
        "audio_stream_policy": "map_0_a_0_no_resample_no_downmix_v1",
    }


@contextmanager
def prepare_input(
    path: Path, *, work_root: Path | None = None,
    decode_timeout: float = DECODE_TIMEOUT_SECONDS,
):
    if not path.is_file():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {path}")
    try:
        sf.info(path)
    except sf.LibsndfileError:
        pass
    else:
        yield path
        return

    ffmpeg = resource_root() / "bin" / "ffmpeg.exe"
    if not ffmpeg.is_file():
        raise DecoderUnavailableError(f"この音声の読み込みに必要なFFmpegが見つかりません: {ffmpeg}")
    if decode_timeout <= 0:
        raise ValueError("入力復号の制限時間は0より大きい必要があります")
    with ScratchDirectory(prefix="decode-", purpose="input_decode",
                          dir=work_root if work_root is not None else session.work_dir()) as folder:
        decoded = Path(folder) / "input.wav"
        command = [
            str(ffmpeg), "-nostdin", "-v", "error", "-n", "-i", str(path.resolve()),
            "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-rf64", "auto", str(decoded),
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, timeout=decode_timeout,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except subprocess.TimeoutExpired as exc:
            raise InputDecodeTimeoutError(
                f"音声の読み込みが{decode_timeout:g}秒以内に完了しませんでした: {path}"
            ) from exc
        if result.returncode:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise InputDecodeError(f"音声を読み込めません: {path}\n{message[:1000]}")
        # チャンネル数とサンプルレートは変換せず、検証を通してから推論へ渡す。
        try:
            sf.info(decoded)
        except (OSError, sf.LibsndfileError) as exc:
            raise InputDecodeError(f"復号後の音声を読み込めません: {path}") from exc
        yield decoded
