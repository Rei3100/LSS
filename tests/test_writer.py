import base64
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from mutagen import File as MutagenFile
from mutagen.asf import ASFByteArrayAttribute
from mutagen.flac import FLAC
from mutagen.flac import Picture
from mutagen.id3 import APIC, ID3, PictureType
from mutagen.mp4 import MP4Cover

from lss.core import session
from lss.harness import tsv_io
from lss.output import writer
from tests.test_audio_input import transcode


ARTWORK = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _write_flac_source(path: Path, seconds: float = 5.0, samplerate: int = 44100, channels: int = 2) -> np.ndarray:
    frames = int(seconds * samplerate)
    t = np.arange(frames) / samplerate
    wave = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    data = np.tile(wave[:, None], (1, channels))
    sf.write(path, data, samplerate, subtype="PCM_16")
    audio = FLAC(path)
    audio["ARTIST"] = "テスト歌手"
    audio["ALBUM"] = "テストアルバム"
    audio["TITLE"] = "テスト曲"
    audio["TRACK"] = "1"
    picture = Picture()
    picture.type = PictureType.COVER_FRONT
    picture.mime = "image/png"
    picture.data = ARTWORK
    audio.add_picture(picture)
    audio.save()
    return data.T  # (channels, frames)


def _write_stem(path: Path, samplerate: int, frames: int, channels: int = 2, amplitude: float = 0.3, silence_head: int = 0) -> None:
    t = np.arange(frames) / samplerate
    wave = (amplitude * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    if silence_head:
        wave[:silence_head] = 0.0
    data = np.tile(wave[:, None], (1, channels))
    sf.write(path, data, samplerate, subtype="PCM_16")


def _setup_recipe(tmp_path, samplerate=44100, seconds=5.0, silence_head=0):
    mix_path = tmp_path / "曲名.flac"
    _write_flac_source(mix_path, seconds=seconds, samplerate=samplerate)
    frames = int(seconds * samplerate)
    store_dir = tmp_path / "work" / "recipe1"
    store_dir.mkdir(parents=True)
    _write_stem(store_dir / "Vocals.wav", samplerate, frames, silence_head=silence_head)
    _write_stem(store_dir / "Instrumental.wav", samplerate, frames, silence_head=silence_head)
    return mix_path, store_dir


def _call_write(
    mix_path, store_dir, recipe_id="becruily_deux_inst", input_sha256="hash1", **kwargs,
):
    return writer.write_recipe_output(
        mix_path, store_dir,
        recipe_id=recipe_id, model="becruily_deux_inst", model_sha256="modelsha",
        score="sdr=2.14 l1_freq=8.45 aura_mrstft=15.78", version="0.1.0",
        input_sha256=input_sha256, processed_at="2026-08-14T00:00:00+00:00",
        **kwargs,
    )


def test_write_recipe_output_produces_one_file_per_stem(tmp_path):
    mix_path, store_dir = _setup_recipe(tmp_path)
    written = _call_write(mix_path, store_dir)
    assert len(written) == 2
    names = sorted(p.name for p in written)
    assert names == ["曲名 - Instrumental.flac", "曲名 - Vocals.flac"]
    for p in written:
        assert p.parent == mix_path.parent / "LSS"


def test_write_recipe_output_can_publish_to_managed_destination(tmp_path):
    mix_path, store_dir = _setup_recipe(tmp_path)
    destination = tmp_path / "managed" / "artifacts"

    written = _call_write(mix_path, store_dir, destination_dir=destination)

    assert len(written) == 2
    assert all(path.parent == destination.resolve() for path in written)
    assert not writer.output_dir_for(mix_path).exists()


def test_ac10_tag_regulation_full_coverage(tmp_path):
    mix_path, store_dir = _setup_recipe(tmp_path)
    written = _call_write(mix_path, store_dir)

    for path in written:
        audio = FLAC(path)
        assert audio["TITLE"][0] == "テスト曲"  # 未改変(I20/R3)
        for key in (
            "LSS_RECIPE_ID", "LSS_MODEL", "LSS_MODEL_SHA256", "LSS_PROCESSED_AT",
            "LSS_INPUT_SHA256", "LSS_SCORE", "LSS_VERSION",
        ):
            assert audio[key][0], f"{key}が空"
        assert audio["REPLAYGAIN_TRACK_GAIN"][0] != ""
        assert float(audio["REPLAYGAIN_TRACK_PEAK"][0]) != 0.0
        # 元曲はReplayGain/TruePeak系タグを持たない(合成テストfixtureのため)ので、
        # 「出力に再計算値が実在する」ことをもってdiffの代わりとする。
        assert "REPLAYGAIN_TRACK_GAIN" in audio
        assert "ARTIST" in audio and audio["ARTIST"][0] == "テスト歌手"
        assert len(audio.pictures) == 1
        assert audio.pictures[0].mime == "image/png"
        assert audio.pictures[0].data == ARTWORK


def test_write_recipe_output_preserves_multiple_tag_values(tmp_path):
    mix_path, store_dir = _setup_recipe(tmp_path, seconds=0.5)
    source = FLAC(mix_path)
    source["ARTIST"] = ["歌手A", "歌手B"]
    source["GENRE"] = ["Rock", "Live"]
    source.save()

    written = _call_write(mix_path, store_dir)

    for path in written:
        output = FLAC(path)
        assert output["ARTIST"] == ["歌手A", "歌手B"]
        assert output["GENRE"] == ["Rock", "Live"]


@pytest.mark.parametrize("container", ["id3", "mp4", "vorbis", "asf"])
def test_read_artwork_supports_non_flac_metadata(monkeypatch, tmp_path, container):
    if container == "id3":
        tags = ID3()
        tags.add(APIC(mime="image/png", type=PictureType.COVER_FRONT, desc="表紙", data=ARTWORK))
    elif container == "mp4":
        tags = {"covr": [MP4Cover(ARTWORK, imageformat=MP4Cover.FORMAT_PNG)]}
    elif container == "vorbis":
        source = Picture()
        source.type = PictureType.COVER_FRONT
        source.mime = "image/png"
        source.data = ARTWORK
        tags = {"metadata_block_picture": [base64.b64encode(source.write()).decode("ascii")]}
    else:
        raw = (
            bytes([PictureType.COVER_FRONT]) + struct.pack("<I", len(ARTWORK))
            + "image/png\0".encode("utf-16-le") + "表紙\0".encode("utf-16-le") + ARTWORK
        )
        tags = {"WM/Picture": [ASFByteArrayAttribute(raw)]}

    monkeypatch.setattr(writer, "MutagenFile", lambda path: SimpleNamespace(tags=tags))
    picture = writer.read_artwork(tmp_path / "source.audio")

    assert picture is not None
    assert picture.type == PictureType.COVER_FRONT
    assert picture.mime == "image/png"
    assert picture.data == ARTWORK


def test_read_artwork_ignores_malformed_embedded_picture(monkeypatch, tmp_path):
    tags = {"metadata_block_picture": ["not-base64"], "WM/Picture": [b"short"]}
    monkeypatch.setattr(writer, "MutagenFile", lambda path: SimpleNamespace(tags=tags))

    assert writer.read_artwork(tmp_path / "broken.audio") is None


@pytest.mark.parametrize(("extension", "codec", "tag_family"), [
    (".wav", "pcm_s16le", "id3"),
    (".aiff", "pcm_s16le", "id3"),
    (".flac", "flac", "flac"),
    (".mp3", "libmp3lame", "id3"),
    (".m4a", "aac", "mp4"),
    (".ogg", "libvorbis", "vorbis"),
    (".opus", "libopus", "vorbis"),
    (".wma", "wmav2", "asf"),
])
def test_read_artwork_from_supported_container(tmp_path, extension, codec, tag_family):
    seed = tmp_path / "seed.wav"
    sf.write(seed, np.zeros((4800, 2)), 48000)
    source = tmp_path / f"source{extension}"
    transcode(seed, source, codec)
    audio = MutagenFile(source)
    assert audio is not None
    if audio.tags is None:
        audio.add_tags()

    if tag_family == "flac":
        picture = Picture()
        picture.type = PictureType.COVER_FRONT
        picture.mime = "image/png"
        picture.data = ARTWORK
        audio.add_picture(picture)
    elif tag_family == "id3":
        audio.tags.add(APIC(mime="image/png", type=PictureType.COVER_FRONT, desc="表紙", data=ARTWORK))
    elif tag_family == "mp4":
        audio.tags["covr"] = [MP4Cover(ARTWORK, imageformat=MP4Cover.FORMAT_PNG)]
    elif tag_family == "vorbis":
        picture = Picture()
        picture.type = PictureType.COVER_FRONT
        picture.mime = "image/png"
        picture.data = ARTWORK
        audio.tags["metadata_block_picture"] = [base64.b64encode(picture.write()).decode("ascii")]
    else:
        raw = (
            bytes([PictureType.COVER_FRONT]) + struct.pack("<I", len(ARTWORK))
            + "image/png\0".encode("utf-16-le") + "表紙\0".encode("utf-16-le") + ARTWORK
        )
        audio.tags["WM/Picture"] = [ASFByteArrayAttribute(raw)]
    audio.save()

    picture = writer.read_artwork(source)

    assert picture is not None
    assert picture.mime == "image/png"
    assert picture.data == ARTWORK


def test_ac11_all_stems_have_identical_sample_count(tmp_path):
    mix_path, store_dir = _setup_recipe(tmp_path)
    written = _call_write(mix_path, store_dir)
    lengths = {sf.info(p).frames for p in written}
    assert len(lengths) == 1


def test_common_silence_is_preserved_by_default(tmp_path):
    samplerate = 44100
    silence_head = int(0.5 * samplerate)
    mix_path, store_dir = _setup_recipe(tmp_path, samplerate=samplerate, seconds=5.0, silence_head=silence_head)
    written = _call_write(mix_path, store_dir)
    full_frames = int(5.0 * samplerate)
    for p in written:
        assert sf.info(p).frames == full_frames


def test_explicit_recipe_trim_removes_common_leading_silence(tmp_path):
    samplerate = 44100
    silence_head = int(0.5 * samplerate)
    mix_path, store_dir = _setup_recipe(
        tmp_path, samplerate=samplerate, seconds=5.0, silence_head=silence_head,
    )

    written = _call_write(mix_path, store_dir, trim_common_silence=True)

    full_frames = int(5.0 * samplerate)
    assert all(sf.info(path).frames < full_frames for path in written)


def test_ac12_second_run_same_recipe_is_skipped_and_first_file_untouched(tmp_path):
    mix_path, store_dir = _setup_recipe(tmp_path)
    first = _call_write(mix_path, store_dir)
    first_bytes = {p: p.read_bytes() for p in first}

    second = _call_write(mix_path, store_dir)

    assert second == []  # 同一入力+同一レシピ→スキップ(A19①)
    for p, original in first_bytes.items():
        assert p.read_bytes() == original  # 無言上書きしない


def test_ac12_different_recipe_creates_r2_suffixed_files(tmp_path):
    mix_path, store_dir = _setup_recipe(tmp_path)
    first = _call_write(mix_path, store_dir, recipe_id="recipe_a")
    first_bytes = {p: p.read_bytes() for p in first}

    store_dir2 = store_dir.parent / "recipe2"
    store_dir2.mkdir()
    frames = sf.info(store_dir / "Vocals.wav").frames
    _write_stem(store_dir2 / "Vocals.wav", 44100, frames, amplitude=0.6)
    _write_stem(store_dir2 / "Instrumental.wav", 44100, frames, amplitude=0.6)
    second = _call_write(mix_path, store_dir2, recipe_id="recipe_b")

    assert len(second) == 2
    names = sorted(p.name for p in second)
    assert names == ["曲名 - Instrumental (r2).flac", "曲名 - Vocals (r2).flac"]
    for p, original in first_bytes.items():
        assert p.read_bytes() == original


def test_write_recipe_output_no_stem_files_is_error(tmp_path):
    mix_path, _ = _setup_recipe(tmp_path)
    empty_store = tmp_path / "empty"
    empty_store.mkdir()
    with pytest.raises(ValueError, match="分離出力がありません"):
        _call_write(mix_path, empty_store)


@pytest.mark.parametrize("sr,frames,channels", [(48000, 220500, 2), (44100, 100, 2), (44100, 220500, 1)])
def test_inconsistent_stems_fail_before_any_final_output(tmp_path, sr, frames, channels):
    mix_path, store_dir = _setup_recipe(tmp_path)
    _write_stem(store_dir / "Vocals.wav", sr, frames, channels=channels)
    with pytest.raises(ValueError, match="一致しません"):
        _call_write(mix_path, store_dir)
    assert not writer.output_dir_for(mix_path).exists()


def test_tag_write_failure_leaves_no_partial_final_file(tmp_path, monkeypatch):
    mix_path, store_dir = _setup_recipe(tmp_path)
    original_save = FLAC.save

    def fail_save(*args, **kwargs):
        raise OSError("タグ保存失敗")

    monkeypatch.setattr(FLAC, "save", fail_save)
    with pytest.raises(OSError, match="タグ保存失敗"):
        _call_write(mix_path, store_dir)
    assert list(writer.output_dir_for(mix_path).iterdir()) == []
    monkeypatch.setattr(FLAC, "save", original_save)
    assert len(_call_write(mix_path, store_dir)) == 2


def test_second_stem_failure_rolls_back_first_staged_output(tmp_path, monkeypatch):
    mix_path, store_dir = _setup_recipe(tmp_path)
    original_save = FLAC.save
    calls = 0

    def fail_second_save(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("2本目のタグ保存失敗")
        return original_save(self, *args, **kwargs)

    monkeypatch.setattr(FLAC, "save", fail_second_save)

    with pytest.raises(OSError, match="2本目のタグ保存失敗"):
        _call_write(mix_path, store_dir)

    output_dir = writer.output_dir_for(mix_path)
    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []


def test_validation_failure_before_publish_removes_all_staged_outputs(tmp_path):
    mix_path, store_dir = _setup_recipe(tmp_path)

    def reject_publish():
        raise RuntimeError("入力世代が変わった")

    with pytest.raises(RuntimeError, match="入力世代が変わった"):
        _call_write(mix_path, store_dir, validate_before_publish=reject_publish)

    assert list(writer.output_dir_for(mix_path).iterdir()) == []


@pytest.mark.parametrize("invalid_data", [np.empty((0, 2)), np.full((220500, 2), np.nan)])
def test_invalid_samples_fail_before_any_final_output(tmp_path, invalid_data):
    mix_path, store_dir = _setup_recipe(tmp_path)
    sf.write(store_dir / "Vocals.wav", invalid_data, 44100, subtype="FLOAT")
    with pytest.raises(ValueError, match="空か、不正"):
        _call_write(mix_path, store_dir)
    assert not writer.output_dir_for(mix_path).exists()


def test_publish_does_not_replace_existing_output(tmp_path):
    existing = tmp_path / "result.flac"
    existing.write_bytes(b"previous result")
    with pytest.raises(FileExistsError):
        writer._write_flac(existing, np.zeros((2, 44100)), 44100, "PCM_16", {}, None)
    assert existing.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == [existing]


def test_durable_publication_records_completion_and_removes_journal(tmp_path, monkeypatch):
    mix_path, store_dir = _setup_recipe(tmp_path)
    state_root = tmp_path / "state"
    monkeypatch.setenv("LSS_STATE_DIR", str(state_root))

    outputs = _call_write(
        mix_path, store_dir, publication_state_root=state_root,
    )

    assert session.is_completed("hash1", "becruily_deux_inst", mix_path, state_root=state_root)
    assert not session.publication_dir(state_root).exists() or not list(
        session.publication_dir(state_root).glob("*.json")
    )
    assert all(path.is_file() for path in outputs)


def test_completion_io_failure_preserves_outputs_for_recovery(tmp_path, monkeypatch):
    mix_path, store_dir = _setup_recipe(tmp_path)
    state_root = tmp_path / "state"
    monkeypatch.setenv("LSS_STATE_DIR", str(state_root))
    complete = session.complete_publication
    monkeypatch.setattr(
        session, "complete_publication",
        lambda *a, **k: (_ for _ in ()).throw(OSError("ledger failed")),
    )

    with pytest.raises(OSError, match="ledger failed"):
        _call_write(mix_path, store_dir, publication_state_root=state_root)

    destination = writer.output_dir_for(mix_path)
    outputs = list(destination.glob("*.flac"))
    assert len(outputs) == 2
    before = {path: path.read_bytes() for path in outputs}
    assert len(list(session.publication_dir(state_root).glob("*.json"))) == 1
    monkeypatch.setattr(session, "complete_publication", complete)
    assert session.reconcile_publications(state_root)[0]["status"] == "completed"
    assert session.is_completed("hash1", "becruily_deux_inst", mix_path, state_root=state_root)
    assert {path: path.read_bytes() for path in outputs} == before


@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("changed", [False, True])
def test_failed_publication_only_rolls_back_unchanged_owned_audio(tmp_path, monkeypatch, durable, changed):
    mix_path, store_dir = _setup_recipe(tmp_path, seconds=0.5)
    state_root = tmp_path / "state"
    rename = Path.rename
    published = []
    replacement = b"another application's replacement"

    def fail_second_publish(path, target):
        if path.name.startswith(".lss-") and Path(target).suffix == ".flac":
            if published:
                if changed:
                    published[0].write_bytes(replacement)
                raise ValueError("publication interrupted")
            result = rename(path, target)
            published.append(Path(target))
            return result
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_second_publish)
    with pytest.raises(ValueError, match="publication interrupted"):
        _call_write(mix_path, store_dir, publication_state_root=state_root if durable else None)

    assert len(published) == 1
    if changed:
        assert published[0].read_bytes() == replacement
    else:
        assert not published[0].exists()
    assert not list(writer.output_dir_for(mix_path).glob(".lss-*.tmp"))
    assert not list(session.publication_dir(state_root).glob("*.json"))
    assert not session.is_completed("hash1", "becruily_deux_inst", mix_path, state_root=state_root)


def test_interrupted_publish_keeps_remaining_staged_audio_for_recovery(tmp_path, monkeypatch):
    mix_path, store_dir = _setup_recipe(tmp_path)
    state_root = tmp_path / "state"
    rename = Path.rename
    count = 0

    def fail_second_publish(path, target):
        nonlocal count
        if path.name.startswith(".lss-") and Path(target).suffix == ".flac":
            count += 1
            if count == 2:
                raise PermissionError("公開先が使用中")
        return rename(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail_second_publish)
        with pytest.raises(PermissionError, match="公開先が使用中"):
            _call_write(mix_path, store_dir, publication_state_root=state_root)
    destination = writer.output_dir_for(mix_path)
    assert len(list(destination.glob("*.flac"))) == 1
    assert len(list(destination.glob(".lss-*.tmp"))) == 1
    assert session.reconcile_publications(state_root)[0]["status"] == "completed"
    assert len(list(destination.glob("*.flac"))) == 2
    assert not list(destination.glob(".lss-*.tmp"))
    assert session.is_completed("hash1", "becruily_deux_inst", mix_path, state_root=state_root)


def test_truncated_output_with_provenance_tags_is_not_reused(tmp_path):
    mix_path, store_dir = _setup_recipe(tmp_path)
    first = _call_write(mix_path, store_dir)
    broken = first[0]
    original = broken.read_bytes()
    broken.write_bytes(original[:-100])
    assert writer.read_source_tags(broken)["LSS_RECIPE_ID"] == "becruily_deux_inst"
    damaged = broken.read_bytes()
    retry = _call_write(mix_path, store_dir)
    assert len(retry) == 1
    assert "(r2)" in retry[0].name
    assert broken.read_bytes() == damaged


def test_resolve_output_path_first_call_has_no_suffix(tmp_path):
    result = writer.resolve_output_path(tmp_path, "曲名", "Vocals", "hash1", "recipe1")
    assert result == tmp_path / "曲名 - Vocals.flac"


def test_common_trim_bounds_no_silence_returns_full_length():
    waveforms = [np.full((2, 100), 0.5), np.full((2, 100), 0.5)]
    start, end = writer.common_trim_bounds(waveforms)
    assert (start, end) == (0, 100)


def test_common_trim_bounds_empty_list_returns_zero():
    assert writer.common_trim_bounds([]) == (0, 0)


def test_common_trim_bounds_does_not_stack_full_stem_set(monkeypatch):
    monkeypatch.setattr(np, "stack", lambda *a, **k: pytest.fail("全ステムの複製を作らない"))
    waveforms = [np.full((2, 100), 0.25, dtype=np.float32) for _ in range(4)]

    assert writer.common_trim_bounds(waveforms) == (0, 100)


def test_common_trim_bounds_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="一致しません"):
        writer.common_trim_bounds([
            np.zeros((2, 100), dtype=np.float32),
            np.zeros((2, 99), dtype=np.float32),
        ])


def test_file_trim_scan_finds_bounds_across_small_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "TRIM_SCAN_BLOCK_FRAMES", 31)
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    data = np.zeros((100, 2), dtype=np.float32)
    data[35:78] = 0.25
    sf.write(first, data, 44100, subtype="FLOAT")
    sf.write(second, data, 44100, subtype="FLOAT")

    assert writer.common_trim_bounds_from_files([first, second]) == (35, 78, 44100, 2, 0.25)


def test_recipe_output_streams_each_trimmed_stem_without_full_read(tmp_path, monkeypatch):
    mix_path, store_dir = _setup_recipe(tmp_path, seconds=1.0, silence_head=100)
    monkeypatch.setattr(
        writer, "common_trim_bounds",
        lambda *a, **k: pytest.fail("全ステムをRAMへ集める旧経路を使わない"),
    )
    original_scan = writer.scan_track_file
    scans = []

    def bounded_scan(path, *args, **kwargs):
        scans.append((Path(path), kwargs.get("start"), kwargs.get("frames")))
        return original_scan(path, *args, **kwargs)

    monkeypatch.setattr(
        writer.sf, "read", lambda *a, **k: pytest.fail("生ステム全体をsf.readで読まない"),
    )
    monkeypatch.setattr(writer, "scan_track_file", bounded_scan)
    monkeypatch.setattr(writer, "OUTPUT_WRITE_BLOCK_FRAMES", 10000)

    assert len(_call_write(mix_path, store_dir, trim_common_silence=True)) == 2
    assert len(scans) == 2
    assert all(start is None and frames is None and path.suffix == ".tmp" for path, start, frames in scans)


def test_recipe_output_reports_scan_and_per_stem_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "TRIM_SCAN_BLOCK_FRAMES", 10000)
    mix_path, store_dir = _setup_recipe(tmp_path, seconds=1.0)
    events = []

    assert len(_call_write(mix_path, store_dir, progress=events.append)) == 2

    analyzing = [event for event in events if event["phase"] == "analyzing_outputs"]
    writing = [event for event in events if event["phase"] == "writing_outputs"]
    assert analyzing[0] == {
        "phase": "analyzing_outputs", "completed_frames": 0, "total_frames": 44100,
    }
    assert analyzing[-1]["completed_frames"] == analyzing[-1]["total_frames"] == 44100
    assert [event["completed_stems"] for event in writing] == [0, 1, 2]
    assert all(event["total_stems"] == 2 for event in writing)


def test_recipe_output_ignores_progress_consumer_failure(tmp_path):
    mix_path, store_dir = _setup_recipe(tmp_path, seconds=0.1)

    def fail(_event):
        raise RuntimeError("表示を更新できない")

    assert len(_call_write(mix_path, store_dir, progress=fail)) == 2


def test_reusable_outputs_are_stream_validated_without_loading_raw_stems(tmp_path, monkeypatch):
    mix_path, store_dir = _setup_recipe(tmp_path, seconds=1.0)
    first = _call_write(mix_path, store_dir)
    monkeypatch.setattr(writer, "TRIM_SCAN_BLOCK_FRAMES", 10000)
    monkeypatch.setattr(
        writer.sf, "read",
        lambda *a, **k: pytest.fail("再利用時に生ステム全体を読み込まない"),
    )
    events = []

    reused = _call_write(
        mix_path, store_dir, include_existing=True, progress=events.append,
    )

    assert reused == first
    checks = [event for event in events if event["phase"] == "checking_existing_output"]
    assert {event["stem"] for event in checks} == {"Instrumental", "Vocals"}
    for stem in ("Instrumental", "Vocals"):
        stem_events = [event for event in checks if event["stem"] == stem]
        total_frames = stem_events[-1]["total_frames"]
        assert [event["completed_frames"] for event in stem_events] == [
            0, 10000, 20000, 30000, 40000, total_frames,
        ]
        assert stem_events[-1]["completed_frames"] == total_frames


def test_output_capacity_reports_measured_bytes_before_writing(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"")
    monkeypatch.setattr(
        writer.shutil, "disk_usage",
        lambda path: SimpleNamespace(free=1024),
    )

    with pytest.raises(writer.InsufficientOutputSpaceError) as caught:
        writer.ensure_output_capacity(source, 60.0, 44100, 2, 2, "PCM_24")

    message = str(caught.value)
    assert "free_bytes=1024" in message
    assert "needed_bytes=" in message


def test_float_stem_above_normalized_pcm_range_publishes_with_declared_headroom(tmp_path):
    """1.0超の分離stemは正常な出力。宣言済みheadroomで可逆に保存される(clipしない)。"""
    mix_path, store_dir = _setup_recipe(tmp_path, seconds=0.1)
    over = np.linspace(-1.25, 1.25, 4410 * 2, dtype=np.float32).reshape(4410, 2)
    for path in store_dir.iterdir():
        sf.write(path, over, 44100, subtype="FLOAT")

    outputs = _call_write(mix_path, store_dir)

    assert len(outputs) == 2
    for output in outputs:
        assert output.exists()
        tags = FLAC(output)
        assert tags["LSS_OUTPUT_HEADROOM_SHIFT"] == ["1"]
        # 元がPCM_16でも、減衰した分だけ絶対量子化ステップが粗くならないようにする。
        assert sf.info(output).subtype == "PCM_24"
        restored, rate = sf.read(output, dtype="float64", always_2d=True)
        assert rate == 44100
        assert np.max(np.abs(restored)) <= 1.0
        # 2の冪の減衰は完全可逆。元のfloat値は量子化誤差の範囲で復元できる。
        assert np.max(np.abs(restored * 2.0 - over)) <= (2 / (2 ** 23)) * 1.01
        assert float(tags["REPLAYGAIN_TRACK_PEAK"][0]) <= 1.0

    destination = writer.output_dir_for(mix_path)
    assert not list(destination.glob(".lss-*.tmp"))


def test_pcm24_source_above_range_stays_pcm24_and_is_reversible(tmp_path):
    """実機の13件はこの形（48kHz PCM_24・instrumentalが1.0超）。

    FLACにPCM_24より上は無いので深度は上がらない。減衰分だけ量子化SN比は下がるが、
    値は2の冪で完全に戻せる。
    """
    mix_path = tmp_path / "曲名.flac"
    frames = 4410
    timeline = np.arange(frames) / 44100
    wave = (0.4 * np.sin(2 * np.pi * 220 * timeline)).astype(np.float32)
    sf.write(mix_path, np.tile(wave[:, None], (1, 2)), 44100, subtype="PCM_24")
    store_dir = tmp_path / "work" / "recipe1"
    store_dir.mkdir(parents=True)
    over = np.linspace(-1.3, 1.3, frames * 2, dtype=np.float32).reshape(frames, 2)
    for name in ("Instrumental.wav", "Vocals.wav"):
        sf.write(store_dir / name, over, 44100, subtype="FLOAT")

    outputs = _call_write(mix_path, store_dir, source_subtype="PCM_24")

    assert len(outputs) == 2
    for output in outputs:
        assert sf.info(output).subtype == "PCM_24"
        tags = FLAC(output)
        assert tags["LSS_OUTPUT_HEADROOM_SHIFT"] == ["1"]
        restored, _rate = sf.read(output, dtype="float64", always_2d=True)
        assert np.max(np.abs(restored)) <= 1.0
        # 公開バイト列から元のfloatへ戻せる（量子化誤差の範囲）。
        assert np.max(np.abs(restored * 2.0 - over)) <= (2 / (2 ** 23)) * 1.01


def test_all_stems_of_one_publication_share_one_headroom_shift(tmp_path):
    """ステムごとに別倍率にすると足し戻しで元mixへ戻らない。倍率は公開単位で1つ。"""
    mix_path, store_dir = _setup_recipe(tmp_path, seconds=0.1)
    loud = np.full((4410, 2), 1.25, dtype=np.float32)
    quiet = np.full((4410, 2), 0.1, dtype=np.float32)
    sf.write(store_dir / "Instrumental.wav", loud, 44100, subtype="FLOAT")
    sf.write(store_dir / "Vocals.wav", quiet, 44100, subtype="FLOAT")

    outputs = _call_write(mix_path, store_dir)

    shifts = {FLAC(output)["LSS_OUTPUT_HEADROOM_SHIFT"][0] for output in outputs}
    assert shifts == {"1"}
    by_stem = {FLAC(output)["LSS_STEM"][0]: output for output in outputs}
    quiet_restored, _ = sf.read(by_stem["Vocals"], dtype="float64", always_2d=True)
    assert np.max(np.abs(quiet_restored)) == pytest.approx(0.05, abs=1e-5)


def test_non_finite_stem_is_still_rejected_without_partial_output(tmp_path):
    """headroomはNaN/Infを救わない。壊れた推論結果は公開しない。"""
    mix_path, store_dir = _setup_recipe(tmp_path, seconds=0.1)
    broken = np.zeros((4410, 2), dtype=np.float32)
    broken[10] = np.inf
    for path in store_dir.iterdir():
        sf.write(path, broken, 44100, subtype="FLOAT")

    with pytest.raises(ValueError):
        _call_write(mix_path, store_dir)

    destination = writer.output_dir_for(mix_path)
    assert not list(destination.glob("*.flac"))
    assert not list(destination.glob(".lss-*.tmp"))


def test_in_range_stems_keep_source_depth_and_no_headroom_shift(tmp_path):
    """1.0以内なら従来どおり。深度も倍率も変えない。"""
    mix_path, store_dir = _setup_recipe(tmp_path, seconds=0.1)

    outputs = _call_write(mix_path, store_dir)

    for output in outputs:
        assert FLAC(output)["LSS_OUTPUT_HEADROOM_SHIFT"] == ["0"]
        assert sf.info(output).subtype == "PCM_16"


def test_float_stem_inside_range_roundtrips_within_pcm24_quantization(tmp_path):
    mix_path, store_dir = _setup_recipe(tmp_path, seconds=0.1)
    source = np.linspace(-0.999, 0.999, 4410 * 2, dtype=np.float32).reshape(4410, 2)
    for path in store_dir.iterdir():
        sf.write(path, source, 44100, subtype="FLOAT")

    outputs = _call_write(mix_path, store_dir, source_subtype="FLOAT")

    restored, _ = sf.read(outputs[0], dtype="float64", always_2d=True)
    assert np.max(np.abs(restored - source)) <= (1 / (2 ** 23)) * 1.01


def test_each_source_derives_its_own_destination(tmp_path):
    """同じバッチでもSOURCEごとに保存先が独立して決まる。"""
    first = tmp_path / "アルバムA" / "曲1.flac"
    second = tmp_path / "アルバムB" / "曲2.flac"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        _write_flac_source(path, seconds=2.0)

    assert writer.output_dir_for(first) == tmp_path / "アルバムA" / "LSS"
    assert writer.output_dir_for(second) == tmp_path / "アルバムB" / "LSS"
    assert writer.output_dir_for(first) != writer.output_dir_for(second)


def test_sources_sharing_a_folder_do_not_collide(tmp_path):
    """同じフォルダの2つのSOURCEが、同じLSSフォルダへ衝突せず保存される。"""
    written = []
    for name in ("曲1", "曲2"):
        mix_path = tmp_path / f"{name}.flac"
        _write_flac_source(mix_path, seconds=2.0)
        store_dir = tmp_path / "work" / name
        store_dir.mkdir(parents=True)
        _write_stem(store_dir / "Instrumental.wav", 44100, 88200)
        written += _call_write(mix_path, store_dir, input_sha256=f"hash-{name}")

    destination = tmp_path / "LSS"
    assert {p.parent for p in written} == {destination}
    assert sorted(p.name for p in written) == [
        "曲1 - Instrumental.flac", "曲2 - Instrumental.flac",
    ]
    assert len({p.resolve() for p in written}) == 2


def test_existing_outputs_under_the_old_layout_are_left_untouched(tmp_path, monkeypatch):
    """旧レイアウトの成果物は移動も削除も書き換えもされず、台帳の実パスで再利用される。"""
    state_root = tmp_path / "state"
    monkeypatch.setenv("LSS_STATE_DIR", str(state_root))

    mix_path, store_dir = _setup_recipe(tmp_path)
    legacy_dir = mix_path.parent / "LSS" / mix_path.stem
    legacy_dir.mkdir(parents=True)
    legacy = _call_write(mix_path, store_dir, destination_dir=legacy_dir)
    before = {path: path.read_bytes() for path in legacy}
    session.mark_completed(
        "hash1", "becruily_deux_inst", source_path=mix_path, outputs=legacy,
        state_root=state_root, when="2020-01-02T03:04:05+00:00",
    )

    # 新しい規則で保存先を引き直しても、台帳は旧実パスを指したまま再利用できる。
    assert writer.output_dir_for(mix_path) == mix_path.parent / "LSS"
    checked = session.check_completed_outputs(
        "hash1", "becruily_deux_inst", mix_path, state_root=state_root,
    )
    assert set(checked) == {path.resolve() for path in legacy}
    assert all(checked.values())
    assert session.is_completed("hash1", "becruily_deux_inst", mix_path, state_root=state_root)

    for path, data in before.items():
        assert path.exists() and path.read_bytes() == data
    assert legacy_dir.exists()


def test_completion_ledger_keeps_the_original_result_time(tmp_path, monkeypatch):
    """完了時刻は台帳の記録値であり、後から検出した時刻で置き換えない。"""
    state_root = tmp_path / "state"
    monkeypatch.setenv("LSS_STATE_DIR", str(state_root))
    mix_path, store_dir = _setup_recipe(tmp_path)
    written = _call_write(mix_path, store_dir)
    session.mark_completed(
        "hash1", "becruily_deux_inst", source_path=mix_path, outputs=written,
        state_root=state_root, when="2019-05-06T07:08:09+00:00",
    )

    _, rows = tsv_io.load_tsv(session.completion_ledger_path(state_root))
    row = next(r for r in rows if r["input_sha256"] == "hash1")
    assert row["completed_at"] == "2019-05-06T07:08:09+00:00"
    assert row["outcome"] == "artifacts"
    recorded = json.loads(row["outputs"])
    assert {Path(entry["path"]) for entry in recorded} == {p.resolve() for p in written}
    # 実際の生成時刻は出力タグ側にも残る。
    assert FLAC(written[0])["LSS_PROCESSED_AT"][0] == "2026-08-14T00:00:00+00:00"


def test_same_stem_sources_never_overwrite_each_other(tmp_path):
    """同じフォルダに同名異拡張子のSOURCEがあっても、既存出力を上書きしない。"""
    flac_source = tmp_path / "曲.flac"
    _write_flac_source(flac_source, seconds=2.0)
    wav_source = tmp_path / "曲.wav"
    _write_stem(wav_source, 44100, 88200, amplitude=0.2)

    written = []
    for index, source in enumerate((flac_source, wav_source)):
        store_dir = tmp_path / "work" / str(index)
        store_dir.mkdir(parents=True)
        _write_stem(store_dir / "Instrumental.wav", 44100, 88200)
        written += _call_write(source, store_dir, input_sha256=f"hash{index}")

    assert len(written) == 2
    assert len({path.resolve() for path in written}) == 2
    assert {path.parent for path in written} == {tmp_path / "LSS"}
    # 2件目は上書きではなく別名で並ぶ。
    names = sorted(path.name for path in written)
    assert names == ["曲 - Instrumental (r2).flac", "曲 - Instrumental.flac"]
    assert all(path.exists() and path.stat().st_size > 0 for path in written)


def test_only_requested_stems_become_user_facing_results(tmp_path):
    """相補stemはRecipeの内部生成物。要求された成果物だけを公開する。"""
    mix_path, store_dir = _setup_recipe(tmp_path)

    written = _call_write(mix_path, store_dir, published_stems=["Instrumental"])

    assert [p.name for p in written] == ["曲名 - Instrumental.flac"]
    assert sorted(p.name for p in (mix_path.parent / "LSS").iterdir()) == [
        "曲名 - Instrumental.flac",
    ]
    # 内部生成物はstore_dirに残ったまま。workerの作業領域ごと破棄される。
    assert (store_dir / "Vocals.wav").is_file()


def test_completion_record_holds_only_the_requested_result(tmp_path, monkeypatch):
    """公開していないstemを完了台帳へ書かない。履歴が実体より多いと嘘になる。"""
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    mix_path, store_dir = _setup_recipe(tmp_path)

    _call_write(
        mix_path, store_dir, published_stems=["Instrumental"],
        publication_state_root=session.state_dir(),
    )

    _, rows = tsv_io.load_tsv(session.completion_ledger_path())
    outputs = json.loads(rows[-1]["outputs"])
    assert [Path(item["path"]).name for item in outputs] == ["曲名 - Instrumental.flac"]


def test_headroom_is_decided_by_the_published_set_not_by_discarded_stems(tmp_path):
    """破棄する内部生成物のピークで、公開する成果物を余計に減衰させない。"""
    mix_path, store_dir = _setup_recipe(tmp_path, seconds=0.1)
    quiet = np.full((4410, 2), 0.5, dtype=np.float32)
    loud = np.full((4410, 2), 1.9, dtype=np.float32)
    sf.write(store_dir / "Instrumental.wav", quiet, 44100, subtype="FLOAT")
    sf.write(store_dir / "Vocals.wav", loud, 44100, subtype="FLOAT")

    outputs = _call_write(mix_path, store_dir, published_stems=["Instrumental"])

    assert FLAC(outputs[0])["LSS_OUTPUT_HEADROOM_SHIFT"][0] == "0"
    samples, _ = sf.read(outputs[0], dtype="float64", always_2d=True)
    assert np.max(np.abs(samples)) == pytest.approx(0.5, abs=1e-4)


def test_missing_requested_stem_is_not_silently_downgraded(tmp_path):
    mix_path, store_dir = _setup_recipe(tmp_path)
    with pytest.raises(ValueError, match="要求されたパート"):
        _call_write(mix_path, store_dir, published_stems=["Guitar"])
    assert not writer.output_dir_for(mix_path).exists()
