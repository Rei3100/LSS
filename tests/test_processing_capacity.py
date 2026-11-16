from types import SimpleNamespace

import pytest

from lss.core import storage
from lss.output import writer


def test_shared_volume_counts_peak_simultaneous_files_without_double_counting(tmp_path):
    work, output = tmp_path / "work", tmp_path / "outputs"
    work.mkdir()
    required = storage.required_capacity_by_volume(
        work, output, input_copy_bytes=100, raw_stems_bytes=300, final_outputs_bytes=200,
    )
    assert len(required) == 1
    assert next(iter(required.values()))[1] == 500
    assert not output.exists(), "事前見積もりだけで保存先を作らない"


def test_distinct_volumes_reserve_work_and_publication_separately(tmp_path, monkeypatch):
    work, output = tmp_path / "work", tmp_path / "outputs"
    work.mkdir()
    output.mkdir()
    monkeypatch.setattr(storage, "_volume", lambda path: 1 if path == work else 2)
    required = storage.required_capacity_by_volume(
        work, output, input_copy_bytes=100, raw_stems_bytes=300, final_outputs_bytes=200,
    )
    assert required == {1: (work, 400), 2: (output, 200)}


def test_space_that_fits_individual_files_can_still_be_insufficient(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"input" * 100)
    monkeypatch.setattr(writer, "OUTPUT_OVERHEAD_BYTES_PER_STEM", 0)
    # 生FLOAT stereo 2本=1600B、5%余白で1680B。公開1000Bとの併存は2680B。
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=2000))
    with pytest.raises(writer.InsufficientOutputSpaceError, match="needed_bytes=2680"):
        storage.ensure_processing_capacity(
            source, tmp_path, tmp_path, source_frames=100, source_samplerate=44100,
            output_samplerate=44100, channels=2, generated_stem_count=2, final_outputs_bytes=1000,
        )


def test_unpublished_stems_and_resampled_tail_are_included(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"input")
    monkeypatch.setattr(writer, "OUTPUT_OVERHEAD_BYTES_PER_STEM", 0)
    monkeypatch.setattr(writer, "OUTPUT_HEADROOM_RATIO", 1)
    required = storage.ensure_processing_capacity(
        source, tmp_path, tmp_path, source_frames=1, source_samplerate=44100,
        output_samplerate=48000, channels=2, generated_stem_count=2, final_outputs_bytes=0,
    )
    assert next(iter(required.values()))[1] == 2 * 2 * 2 * 4 + 5
