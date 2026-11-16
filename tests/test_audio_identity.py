"""外部の再生ソフトがtagを書き戻しても、検証済みの出力を失ったものとして作り直さない。"""
from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.core import audio_identity, session
from lss.harness import tsv_io


def write_flac(path: Path, seed: int = 0, frames: int = 4096) -> Path:
    rng = np.random.default_rng(seed)
    sf.write(path, rng.standard_normal((frames, 2)) * 0.25, 44100, subtype="PCM_24")
    return path


def add_playback_tag(path: Path, count: str = "3") -> None:
    tags = FLAC(path)
    tags["LSS_TEST_PLAY_COUNT"] = [count]
    tags.save()


def completed(tmp_path, output):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((64, 2)), 44100, subtype="FLOAT")
    session.mark_completed("a" * 64, "recipe-1", source_path=source, outputs=[output],
                           state_root=tmp_path / "state")
    return source


def test_container_metadata_change_keeps_the_verified_output_reusable(tmp_path):
    output = write_flac(tmp_path / "accompaniment.flac")
    source = completed(tmp_path, output)
    before = output.read_bytes()
    add_playback_tag(output)
    assert output.read_bytes() != before
    state = session.describe_completed_outputs("a" * 64, "recipe-1", source, state_root=tmp_path / "state")
    assert state == {output.resolve(): "EXTERNAL_METADATA_ONLY_CHANGE"}
    assert session.check_completed_outputs("a" * 64, "recipe-1", source,
                                           state_root=tmp_path / "state") == {output.resolve(): True}
    assert session.is_completed("a" * 64, "recipe-1", source, state_root=tmp_path / "state")


@pytest.mark.parametrize("damage", ["samples", "timeline", "missing"])
def test_changed_audio_is_not_reused_as_a_completed_output(tmp_path, damage):
    output = write_flac(tmp_path / "accompaniment.flac")
    source = completed(tmp_path, output)
    if damage == "samples":
        write_flac(output, seed=1)
        expected = "AUDIO_CHANGED"
    elif damage == "timeline":
        write_flac(output, frames=4095)
        expected = "AUDIO_CHANGED"
    else:
        output.unlink()
        expected = "MISSING"
    state = session.describe_completed_outputs("a" * 64, "recipe-1", source, state_root=tmp_path / "state")
    assert state == {output.resolve(): expected}
    assert not session.is_completed("a" * 64, "recipe-1", source, state_root=tmp_path / "state")


def test_records_without_audio_identity_do_not_claim_the_cause_of_a_byte_change(tmp_path):
    output = write_flac(tmp_path / "accompaniment.flac")
    source = completed(tmp_path, output)
    ledger = session.completion_ledger_path(tmp_path / "state")
    text = ledger.read_text(encoding="utf-8")
    key = json.dumps('"audio"')[1:-1]
    assert key in text
    ledger.write_text(text.replace(key, json.dumps('"legacy_audio"')[1:-1]), encoding="utf-8")
    add_playback_tag(output)
    state = session.describe_completed_outputs("a" * 64, "recipe-1", source, state_root=tmp_path / "state")
    assert state == {output.resolve(): "UNVERIFIABLE_WITHOUT_RECORDED_AUDIO_IDENTITY"}
    assert not session.is_completed("a" * 64, "recipe-1", source, state_root=tmp_path / "state")


def test_audio_identity_ignores_tags_and_separates_samples_timeline_and_precision(tmp_path):
    first = write_flac(tmp_path / "first.flac")
    identity = audio_identity.audio_identity(first)
    add_playback_tag(first)
    assert audio_identity.audio_identity(first) == identity
    assert identity["frames"] == 4096 and identity["channels"] == 2 and identity["subtype"] == "PCM_24"
    same_samples_other_precision = tmp_path / "second.flac"
    values, rate = sf.read(first, dtype="float64", always_2d=True)
    sf.write(same_samples_other_precision, values, rate, subtype="PCM_16")
    assert not audio_identity.same_audio(identity, audio_identity.audio_identity(same_samples_other_precision))
    with pytest.raises(audio_identity.AudioIdentityError):
        audio_identity.audio_identity(tmp_path / "notes.txt".replace("notes", "absent"))


def test_non_audio_completion_records_stay_usable_without_an_audio_identity(tmp_path):
    output = tmp_path / "report.json"
    output.write_text('{"ok": true}', encoding="utf-8")
    source = completed(tmp_path, output)
    assert session.is_completed("a" * 64, "recipe-1", source, state_root=tmp_path / "state")
    output.write_text('{"ok": false}', encoding="utf-8")
    assert session.describe_completed_outputs("a" * 64, "recipe-1", source, state_root=tmp_path / "state") == {
        output.resolve(): "UNVERIFIABLE_WITHOUT_RECORDED_AUDIO_IDENTITY"}
    assert not session.is_completed("a" * 64, "recipe-1", source, state_root=tmp_path / "state")


def test_bytes_appended_to_the_audio_stream_are_not_treated_as_metadata(tmp_path):
    # Decoded samples are unchanged, but the FLAC stream no longer ends on its last frame: tampering, not a re-encode.
    output = write_flac(tmp_path / "accompaniment.flac")
    source = completed(tmp_path, output)
    with output.open("ab") as stream:
        stream.write(b"appended")
    state = session.describe_completed_outputs("a" * 64, "recipe-1", source, state_root=tmp_path / "state")
    assert state == {output.resolve(): "CONTAINER_CHANGE_NOT_PROVEN_METADATA_ONLY"}
    assert not session.is_completed("a" * 64, "recipe-1", source, state_root=tmp_path / "state")


def _crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def test_exact_flac_stream_accepts_encoder_output_and_rejects_anything_after_the_declared_end(tmp_path):
    # Independent counterexamples on 2918062 (verifier Ve2/Ve3/Ve4): bytes whose CRC-16 continues the last frame,
    # a copy of the last frame, and a copied mid-stream frame were all accepted as the same verified output.
    from lss.core import flac_frames

    for level in (0.0, 0.5, 1.0):
        path = write_flac(tmp_path / f"x{level}.flac", frames=44100 * 3)
        sf.write(path, sf.read(path, dtype="int32")[0], 44100, subtype="PCM_24", compression_level=level)
        add_playback_tag(path)
        assert flac_frames.stream_is_exact(path)
    clean = path.read_bytes()
    junk = b"LSS!" + bytes([0x7F, 0x01])
    forged = tmp_path / "forged.flac"
    forged.write_bytes(clean + junk + _crc16(junk).to_bytes(2, "big"))
    appended = tmp_path / "appended.flac"
    appended.write_bytes(clean + b"late")
    info = flac_frames.streaminfo(path)
    size = len(clean)
    last = next(at for at in range(size - 8, 0, -1)
                if (h := flac_frames._header(clean, at, info)) and h["first_sample"] + h["block"] == info["total_samples"])
    duplicated = tmp_path / "duplicated.flac"
    duplicated.write_bytes(clean + clean[last:])
    for tampered in (forged, appended, duplicated):
        assert sf.read(tampered, dtype="int32")[0].shape == sf.read(path, dtype="int32")[0].shape
        assert not flac_frames.stream_is_exact(tampered), tampered.name


def test_a_tampered_output_with_unchanged_decoded_samples_is_not_reused(tmp_path):
    output = write_flac(tmp_path / "accompaniment.flac", frames=44100 * 2)
    source = completed(tmp_path, output)
    junk = b"LSS!" + bytes([0x7F, 0x01])
    with output.open("ab") as stream:
        stream.write(junk + _crc16(junk).to_bytes(2, "big"))
    state = session.describe_completed_outputs("a" * 64, "recipe-1", source, state_root=tmp_path / "state")
    assert state == {output.resolve(): "CONTAINER_CHANGE_NOT_PROVEN_METADATA_ONLY"}


def test_a_wav_playback_chunk_after_data_keeps_the_declared_stream_length(tmp_path):
    output = tmp_path / "accompaniment.wav"
    sf.write(output, np.arange(400).reshape(200, 2) / 400, 44100, subtype="FLOAT")
    identity = audio_identity.audio_identity(output)
    with output.open("ab") as stream:
        stream.write(b"LIST" + (12).to_bytes(4, "little") + b"INFOIsftplay")
    after = audio_identity.audio_identity(output)
    assert audio_identity.same_audio(identity, after)
    assert after["container"] == "wav" and after["audio_bytes"] == identity["audio_bytes"]
    with output.open("r+b") as stream:
        stream.seek(identity["audio_offset"] if "audio_offset" in identity
                    else audio_identity.container_layout(output)["audio_offset"])
        stream.write(b"\x01\x02\x03\x04")
    assert not audio_identity.same_audio(identity, audio_identity.audio_identity(output))


def test_unsupported_containers_do_not_claim_a_metadata_only_change(tmp_path):
    output = tmp_path / "accompaniment.ogg"
    sf.write(output, np.zeros((2048, 2)), 44100, format="OGG", subtype="VORBIS")
    identity = audio_identity.audio_identity(output)
    assert "container" not in identity and "audio_bytes" not in identity
    assert audio_identity.container_layout(output) is None
    assert not audio_identity.same_audio(identity, audio_identity.audio_identity(output))
    source = completed(tmp_path, output)
    with output.open("ab") as stream:
        stream.write(b"appended")
    assert session.describe_completed_outputs("a" * 64, "recipe-1", source,
        state_root=tmp_path / "state") == {output.resolve(): "CONTAINER_CHANGE_NOT_PROVEN_METADATA_ONLY"}


def legacy_ledger(tmp_path, output):
    """sample同一性を記録していなかった時期の完了記録を作る。"""
    source = completed(tmp_path, output)
    ledger = session.completion_ledger_path(tmp_path / "state")
    header, rows = tsv_io.load_tsv(ledger)
    for row in rows:
        row["outputs"] = json.dumps([{key: value for key, value in item.items() if key != "audio"}
                                     for item in json.loads(row["outputs"])], ensure_ascii=False)
    tsv_io.save_tsv(ledger, header, rows)
    return source, ledger


def test_existing_results_gain_sample_identity_only_while_they_still_match(tmp_path):
    output = write_flac(tmp_path / "accompaniment.flac")
    source, ledger = legacy_ledger(tmp_path, output)
    other = write_flac(tmp_path / "second.flac", seed=5)
    session.mark_completed("b" * 64, "recipe-1", source_path=other, outputs=[other],
                           state_root=tmp_path / "state")
    summary = session.record_missing_audio_identity(state_root=tmp_path / "state")
    assert summary["recorded"] == 1 and summary["already_recorded"] == 1
    assert summary["not_recordable"] == []
    add_playback_tag(output)
    assert session.describe_completed_outputs("a" * 64, "recipe-1", source,
        state_root=tmp_path / "state") == {output.resolve(): "EXTERNAL_METADATA_ONLY_CHANGE"}
    again = session.record_missing_audio_identity(state_root=tmp_path / "state")
    assert again["recorded"] == 0 and again["already_recorded"] == 2


def test_an_output_that_no_longer_matches_is_not_given_a_new_identity(tmp_path):
    output = write_flac(tmp_path / "accompaniment.flac")
    source, ledger = legacy_ledger(tmp_path, output)
    before = ledger.read_bytes()
    write_flac(output, seed=2)
    summary = session.record_missing_audio_identity(state_root=tmp_path / "state")
    assert summary == {"rows": 1, "outputs": 1, "recorded": 0, "already_recorded": 0,
                       "not_recordable": [str(output)], "written": False}
    assert ledger.read_bytes() == before
    assert session.describe_completed_outputs("a" * 64, "recipe-1", source, state_root=tmp_path / "state") == {
        output.resolve(): "UNVERIFIABLE_WITHOUT_RECORDED_AUDIO_IDENTITY"}


def test_a_completion_written_during_the_recording_is_not_overwritten(tmp_path, monkeypatch):
    output = write_flac(tmp_path / "accompaniment.flac")
    source, ledger = legacy_ledger(tmp_path, output)
    other = write_flac(tmp_path / "second.flac", seed=5)
    original = session.audio_identity.audio_identity

    def concurrent(path):
        # 記録の途中で、別の処理が同じ台帳へ完了を書き込む。
        monkeypatch.setattr(session.audio_identity, "audio_identity", original)
        session.mark_completed("c" * 64, "recipe-2", source_path=other, outputs=[other],
                               state_root=tmp_path / "state")
        return original(path)

    monkeypatch.setattr(session.audio_identity, "audio_identity", concurrent)
    with pytest.raises(session.CompletionLedgerChangedError):
        session.record_missing_audio_identity(state_root=tmp_path / "state")
    rows = tsv_io.load_tsv(ledger)[1]
    assert [row["input_sha256"] for row in rows] == ["a" * 64, "c" * 64]


@pytest.mark.parametrize("broken", [[], "recorded", 3, {"schema_version": 99}, {"pcm_sha256": "x"}])
def test_a_damaged_audio_identity_record_never_claims_a_metadata_only_change(tmp_path, broken):
    output = write_flac(tmp_path / "accompaniment.flac")
    add_playback_tag(output)
    state = audio_identity.diagnose(output, recorded_sha256="a" * 64, recorded_audio=broken,
                                    observed_sha256=audio_identity.file_digest(output))
    assert state not in audio_identity.REUSABLE_STATES
    assert not audio_identity.same_audio(broken, audio_identity.audio_identity(output))
    assert not audio_identity.matches_record(output, {"sha256": "a" * 64, "audio": broken})
    assert not audio_identity.matches_record(output, {"sha256": None})


def test_a_directory_or_absent_file_is_never_reusable(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    assert audio_identity.file_digest(folder) is None
    assert not audio_identity.matches_record(folder, {"sha256": "a" * 64})
    assert audio_identity.diagnose(tmp_path / "absent.flac", recorded_sha256="a" * 64,
                                   recorded_audio=None, observed_sha256=None) == "MISSING"


def test_material_record_carries_place_bytes_and_samples(tmp_path):
    output = write_flac(tmp_path / "accompaniment.flac")
    record = audio_identity.material_record(output)
    assert record["path"] == str(output.resolve())
    assert record["sha256"] == audio_identity.file_digest(output)
    assert record["audio"] == audio_identity.audio_identity(output)
    assert audio_identity.matches_record(output, record)
    add_playback_tag(output)
    assert audio_identity.matches_record(output, record)
    notes = tmp_path / "notes.txt"
    notes.write_text("音声ではない素材", encoding="utf-8")
    assert "audio" not in audio_identity.material_record(notes)
    with pytest.raises(audio_identity.AudioIdentityError):
        audio_identity.material_record(tmp_path / "absent.flac")


def test_a_dry_run_counts_without_changing_the_ledger(tmp_path):
    output = write_flac(tmp_path / "accompaniment.flac")
    source, ledger = legacy_ledger(tmp_path, output)
    before = ledger.read_bytes()
    preview = session.record_missing_audio_identity(state_root=tmp_path / "state", dry_run=True)
    assert preview["recorded"] == 1 and preview["written"] is False
    assert ledger.read_bytes() == before
    written = session.record_missing_audio_identity(state_root=tmp_path / "state")
    assert written["recorded"] == 1 and written["written"] is True
    assert ledger.read_bytes() != before
    assert session.record_missing_audio_identity(state_root=tmp_path / "state")["written"] is False


def reencode_flac(path: Path, *, subtype: str = "PCM_24", fmt: str = "FLAC", level: float = 1.0) -> None:
    """Re-encode the same decoded samples with another compression level (what a player or ffmpeg does)."""
    data, rate = sf.read(path, dtype="int32", always_2d=True)
    tags = dict(FLAC(path).tags or {}) if fmt == "FLAC" else {}
    kwargs = {"compression_level": level} if fmt == "FLAC" else {}
    sf.write(path, data, rate, subtype=subtype, format=fmt, **kwargs)
    if tags:
        flac = FLAC(path)
        for key, value in tags.items():
            flac[key] = value
        flac.save()


def test_a_lossless_reencode_of_an_output_is_the_same_audio_and_is_not_rebuilt(tmp_path):
    # Independent counterexample (Codex, d654515 Vc_reencode): a player re-encoded the Instrumental FLAC losslessly;
    # LSS separated again and published an "(r2)" duplicate of the same sound.
    output = write_flac(tmp_path / "accompaniment.flac")
    sf.write(output, sf.read(output, dtype="int32")[0], 44100, subtype="PCM_24", compression_level=0.0)
    source = completed(tmp_path, output)
    before_bytes, before_audio = output.read_bytes(), audio_identity.audio_identity(output)
    reencode_flac(output, level=1.0)
    after_audio = audio_identity.audio_identity(output)
    assert output.read_bytes() != before_bytes and after_audio["audio_bytes"] != before_audio["audio_bytes"]
    assert after_audio["pcm_sha256"] == before_audio["pcm_sha256"]
    assert not audio_identity.same_audio(before_audio, after_audio)  # not a metadata-only change
    assert session.describe_completed_outputs("a" * 64, "recipe-1", source, state_root=tmp_path / "state") == {
        output.resolve(): "SAME_DECODED_AUDIO_STREAM_CHANGED"}
    assert session.is_completed("a" * 64, "recipe-1", source, state_root=tmp_path / "state")


def test_an_extra_frame_padded_to_the_recorded_stream_length_is_not_a_metadata_change(tmp_path):
    # Independent counterexample (Codex, 0fdcdfc equal-length probe): re-encode tighter, append a playable frame and
    # pad with zeros until the stream length equals the record. libsndfile stops at STREAMINFO, so samples, frames
    # and stream length all matched and the output was reused as a metadata-only change; ffmpeg played 4096 more.
    from lss.core import flac_frames

    output = tmp_path / "accompaniment.flac"
    t = np.arange(44100 * 3) / 44100
    tones = np.stack([0.25 * np.sin(2 * np.pi * 440 * t), 0.2 * np.sin(2 * np.pi * 659 * t)], axis=1)
    sf.write(output, tones, 44100, subtype="PCM_24", compression_level=0.0)
    source = completed(tmp_path, output)
    recorded = audio_identity.audio_identity(output)
    reencode_flac(output, level=1.0)
    blob = output.read_bytes()
    start = audio_identity.container_layout(output)["audio_offset"]
    header = flac_frames._header(blob, start, flac_frames.streaminfo(output))
    frame = blob[start:flac_frames._frame_end(blob, start, header)]
    gap = recorded["audio_bytes"] - audio_identity.audio_identity(output)["audio_bytes"]
    assert gap > len(frame)
    output.write_bytes(blob + frame + bytes(gap - len(frame)))
    # Decoder-prefix equality remains useful diagnosis, but cannot be a public identity.
    assert audio_identity.same_audio(recorded, audio_identity._decoded_identity(output))
    with pytest.raises(audio_identity.AudioIdentityError, match="FLAC stream"):
        audio_identity.audio_identity(output)
    state = session.describe_completed_outputs("a" * 64, "recipe-1", source, state_root=tmp_path / "state")
    assert state == {output.resolve(): "CONTAINER_CHANGE_NOT_PROVEN_METADATA_ONLY"}
    assert not session.is_completed("a" * 64, "recipe-1", source, state_root=tmp_path / "state")


def _tones(path: Path, seconds: int = 2) -> Path:
    t = np.arange(44100 * seconds) / 44100
    sf.write(path, np.stack([0.25 * np.sin(2 * np.pi * 440 * t), 0.2 * np.sin(2 * np.pi * 659 * t)], axis=1),
             44100, subtype="PCM_24")
    return path


def _crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


@pytest.mark.parametrize("frames", ["all", "one"])
def test_frame_headers_claiming_another_sample_rate_are_not_the_same_audio(tmp_path, frames):
    # Independent counterexample (Claude verifier, 9a491ef Vr): every frame header rewritten from 44.1 to 48 kHz with
    # CRC-8/CRC-16 recomputed. Same length, STREAMINFO and libsndfile samples unchanged; ffmpeg plays it 8.8 % faster.
    from lss.core import flac_frames

    output = _tones(tmp_path / "accompaniment.flac")
    source = completed(tmp_path, output)
    blob = bytearray(output.read_bytes())
    info = flac_frames.streaminfo(output)
    at, starts = audio_identity.container_layout(output)["audio_offset"], []
    while at < len(blob):
        header = flac_frames._header(bytes(blob), at, info)
        end = flac_frames._frame_end(bytes(blob), at, header)
        starts.append((at, header["subframes_at"], end))
        at = end
    for at, subframes_at, end in (starts if frames == "all" else starts[len(starts) // 2:len(starts) // 2 + 1]):
        blob[at + 2] = (blob[at + 2] & 0xF0) | 10
        blob[subframes_at - 1] = _crc8(bytes(blob[at:subframes_at - 1]))
        blob[end - 2:end] = _crc16(bytes(blob[at:end - 2])).to_bytes(2, "big")
    before = sf.read(output, dtype="int32")[0]
    output.write_bytes(bytes(blob))
    np.testing.assert_array_equal(sf.read(output, dtype="int32")[0], before)
    assert not flac_frames.stream_is_exact(output)
    state = session.describe_completed_outputs("a" * 64, "recipe-1", source, state_root=tmp_path / "state")
    assert state[output.resolve()] not in audio_identity.REUSABLE_STATES


def test_a_hidden_frame_that_decoders_reject_is_not_the_same_audio(tmp_path):
    # Ve6: a CRC-valid frame declaring 16-bit samples and the declared final position, appended after the real last
    # frame. No decoder plays it, but ffmpeg reports the stream as damaged; it is not the output LSS recorded.
    from lss.core import flac_frames

    output = _tones(tmp_path / "accompaniment.flac")
    source = completed(tmp_path, output)
    total = flac_frames.streaminfo(output)["total_samples"]
    first = total - 4096
    head = bytes([0xFF, 0xF9, (12 << 4), (1 << 4) | (4 << 1)])
    extra = next(n for n in range(1, 7) if first < 1 << (5 * n + 6))
    number = bytes([((0xFF << (7 - extra)) & 0xFF) | (first >> (6 * extra))]
                   + [0x80 | ((first >> (6 * (extra - 1 - i))) & 0x3F) for i in range(extra)])
    head += number
    head += bytes([_crc8(head)])
    body = (bytes(1) + (12345).to_bytes(2, "big")) * 2
    frame = head + body
    with output.open("ab") as stream:
        stream.write(frame + _crc16(frame).to_bytes(2, "big"))
    info = flac_frames.streaminfo(output)
    assert flac_frames.last_frame_ends_at_eof(output, info)  # well formed: only a complete decode exposes it
    assert not flac_frames.stream_is_exact(output)
    state = session.describe_completed_outputs("a" * 64, "recipe-1", source, state_root=tmp_path / "state")
    assert state[output.resolve()] not in audio_identity.REUSABLE_STATES


@pytest.mark.parametrize("change", ["precision", "container", "one_sample"])
def test_a_reencode_that_changes_precision_container_or_samples_is_not_the_same_audio(tmp_path, change):
    output = write_flac(tmp_path / "accompaniment.flac")
    source = completed(tmp_path, output)
    if change == "precision":
        reencode_flac(output, subtype="PCM_16")
    elif change == "container":
        data, rate = sf.read(output, dtype="int32", always_2d=True)
        sf.write(output, data, rate, subtype="PCM_24", format="WAV")
    else:
        data, rate = sf.read(output, dtype="int32", always_2d=True)
        data[100, 0] += 256
        sf.write(output, data, rate, subtype="PCM_24", format="FLAC", compression_level=0.0)
    state = session.describe_completed_outputs("a" * 64, "recipe-1", source, state_root=tmp_path / "state")
    assert state[output.resolve()] not in audio_identity.REUSABLE_STATES
    assert not session.is_completed("a" * 64, "recipe-1", source, state_root=tmp_path / "state")


def _count_decodes(monkeypatch):
    calls = []
    real = audio_identity._decoded_identity
    monkeypatch.setattr(audio_identity, "_decoded_identity", lambda p: calls.append(Path(p)) or real(p))
    return calls


def test_identity_is_decoded_once_per_unchanged_file_and_equals_a_fresh_decode(tmp_path, monkeypatch):
    path = write_flac(tmp_path / "a.flac")
    fresh = audio_identity._decoded_identity(path)
    calls = _count_decodes(monkeypatch)
    first = audio_identity.audio_identity(path)
    second = audio_identity.audio_identity(path)
    assert len(calls) == 1
    assert first == second == fresh
    first["pcm_sha256"] = "tampered"
    assert audio_identity.audio_identity(path) == fresh


def test_identity_is_decoded_again_after_any_rewrite(tmp_path, monkeypatch):
    path = write_flac(tmp_path / "a.flac")
    before = audio_identity.audio_identity(path)
    calls = _count_decodes(monkeypatch)
    add_playback_tag(path)
    assert audio_identity.audio_identity(path) == before
    write_flac(path, seed=5)
    changed = audio_identity.audio_identity(path)
    assert len(calls) == 2 and changed["pcm_sha256"] != before["pcm_sha256"]


def test_rename_keeps_identity_only_when_size_and_mtime_are_unchanged(tmp_path, monkeypatch):
    source = write_flac(tmp_path / "a.flac")
    expected = audio_identity.audio_identity(source)
    calls = _count_decodes(monkeypatch)
    audio_identity.rename_keeping_identity(source, tmp_path / "b.flac")
    assert audio_identity.audio_identity(tmp_path / "b.flac") == expected and calls == []
    # A different file moved onto a name with a different mtime is decoded afresh. (Only different bytes: the same
    # bytes under a new mtime are recognised by content, `test_the_same_bytes_under_a_new_stamp_are_not_decoded_again`.)
    other = write_flac(tmp_path / "c.flac", seed=9)
    audio_identity.audio_identity(other)
    import os
    add_playback_tag(other)
    stat = other.stat()
    os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    audio_identity.audio_identity(other)
    assert len(calls) == 2


def test_the_same_bytes_under_a_new_stamp_are_not_decoded_again(tmp_path, monkeypatch):
    """A copy or a move under a new mtime has the bytes of a verified file: the identity is that file's, which is
    exactly what decoding the copy gives (the decoders read nothing but the bytes)."""
    import os
    import shutil

    path = write_flac(tmp_path / "a.flac")
    expected = audio_identity.audio_identity(path)
    calls = _count_decodes(monkeypatch)
    copy = tmp_path / "copy.flac"
    shutil.copyfile(path, copy)
    stat = copy.stat()
    os.utime(copy, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000))
    moved = tmp_path / "moved.flac"
    shutil.move(str(copy), moved)
    observed = audio_identity.audio_identity(moved)
    assert calls == [] and observed == expected
    calls.clear()
    assert observed == audio_identity._decoded_identity(moved)  # what decoding the moved copy itself gives
    calls.clear()
    observed["pcm_sha256"] = "tampered"  # callers get copies; the content cache is not altered through them
    shutil.copyfile(path, tmp_path / "again.flac")
    assert audio_identity.audio_identity(tmp_path / "again.flac") == expected and calls == []


def test_a_copy_with_any_changed_byte_is_decoded_and_judged_afresh(tmp_path, monkeypatch):
    import shutil

    path = write_flac(tmp_path / "a.flac")
    expected = audio_identity.audio_identity(path)
    calls = _count_decodes(monkeypatch)
    tagged = tmp_path / "tagged.flac"
    shutil.copyfile(path, tagged)
    add_playback_tag(tagged)
    assert audio_identity.audio_identity(tagged)["pcm_sha256"] == expected["pcm_sha256"]
    assert len(calls) == 1
    # A truncated copy is not the verified stream: decoding it again finds it incomplete.
    data = path.read_bytes()
    truncated = tmp_path / "truncated.flac"
    truncated.write_bytes(data[:-200])
    with pytest.raises(audio_identity.AudioIdentityError):
        audio_identity.audio_identity(truncated)
    assert len(calls) == 2
