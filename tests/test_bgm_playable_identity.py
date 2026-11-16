"""Public BGM identity includes the complete stream a player decodes."""
import pytest
import soundfile as sf
import numpy as np

from lss.bgm import current_best, listening
from lss.core import audio_identity, flac_frames
from tests.test_audio_identity import _crc8, _crc16, reencode_flac
from tests.test_listening_product import _cue


def _damage(path, kind):
    info = flac_frames.streaminfo(path)
    blob = bytearray(path.read_bytes())
    start = audio_identity.container_layout(path)["audio_offset"]
    if kind == "extra_frame":
        header = flac_frames._header(bytes(blob), start, info)
        end = flac_frames._frame_end(bytes(blob), start, header)
        path.write_bytes(blob + blob[start:end])
    else:
        while start < len(blob):
            header = flac_frames._header(bytes(blob), start, info)
            end = flac_frames._frame_end(bytes(blob), start, header)
            subframes = header["subframes_at"]
            blob[start + 2] = (blob[start + 2] & 0xF0) | 10  # player sees 48 kHz
            blob[subframes - 1] = _crc8(bytes(blob[start:subframes - 1]))
            blob[end - 2:end] = _crc16(bytes(blob[start:end - 2])).to_bytes(2, "big")
            start = end
        path.write_bytes(blob)


@pytest.mark.parametrize("damage", ["extra_frame", "frame_rate"])
def test_currentbest_and_listening_refuse_unchanged_declared_prefix_with_changed_playable_stream(tmp_path, damage):
    path = _cue(tmp_path)
    expected = audio_identity.audio_identity(path)
    prefix = sf.read(path, dtype="int32")[0]
    _damage(path, damage)
    attacked_bytes = path.read_bytes()
    # Genuine adversarial control: the existing single-decoder identity still agrees.
    np.testing.assert_array_equal(sf.read(path, dtype="int32")[0], prefix)
    assert not flac_frames.stream_is_exact(path)
    assert not current_best._matches(path, expected["pcm_sha256"])
    with pytest.raises(audio_identity.AudioIdentityError, match="FLAC stream"):
        audio_identity.audio_identity(path)
    with pytest.raises(audio_identity.AudioIdentityError, match="FLAC stream"):
        listening.finish(path, tmp_path / "state", expected_pcm=expected["pcm_sha256"])
    assert path.read_bytes() == attacked_bytes


def test_legitimate_public_tag_edit_and_complete_lossless_reencode_keep_currentbest_identity(tmp_path):
    from mutagen.flac import FLAC
    path = _cue(tmp_path)
    expected = audio_identity.audio_identity(path)
    tags = FLAC(path)
    tags["COMMENT"] = ["Foreign annotation"]
    tags.save()
    assert current_best._matches(path, expected["pcm_sha256"])
    reencode_flac(path, level=1.0)
    assert current_best._matches(path, expected["pcm_sha256"])
    assert audio_identity.same_decoded_audio(expected, audio_identity.audio_identity(path))
    assert FLAC(path)["COMMENT"] == ["Foreign annotation"]
