from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from lss import paths
from lss.bgm import scene_names
from lss.core import media_metadata


def test_ffmetadata_parser_preserves_escapes_whitespace_and_multiline_values():
    encoded = (";FFMETADATA1\n"
               "  title  =  keep surrounding spaces  \n"
               "work\\=key=left\\=right \\#tag \\;semi \\\\path\n"
               "description=first line\\\nsecond line with # and = and \\;\n"
               "; global comment\n"
               "[CHAPTER]\nTIMEBASE=1/1000\nTITLE=chapter must not escape\n"
               "[STREAM]\ntitle=stream must not escape\n")

    assert media_metadata.parse_ffmetadata(encoded) == {
        "  title  ": "  keep surrounding spaces  ",
        "work=key": "left=right #tag ;semi \\path",
        "description": "first line\nsecond line with # and = and ;",
    }
    assert media_metadata.parse_ffmetadata((';FFMETADATA1\npath=C' + chr(58) + '\\')) == {"path": ('C' + chr(58) + '\\')}
    assert media_metadata.parse_ffmetadata("not ffmetadata\ntitle=unsafe\n") == {}


def test_frozen_metadata_reader_requires_bundled_ffmpeg_and_emits_global_only_command(monkeypatch, tmp_path):
    bundled = tmp_path / "resources" / "bin" / "ffmpeg.exe"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"test executable marker")
    monkeypatch.setattr(paths, "resource_root", lambda: bundled.parents[1])
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=b";FFMETADATA1\ntitle=global\n[CHAPTER]\ntitle=chapter\n")

    monkeypatch.setattr(media_metadata.subprocess, "run", run)
    assert media_metadata.format_tags(tmp_path / "episode.mkv") == {"title": "global"}
    command, kwargs = calls[0]
    assert command == [str(bundled), "-nostdin", "-v", "error", "-i", str(tmp_path / "episode.mkv"),
                       "-map_metadata", "0", "-f", "ffmetadata", "-"]
    assert kwargs["timeout"] == 60 and kwargs["check"] is True


def test_frozen_metadata_reader_does_not_fall_back_to_path_when_bundle_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "resource_root", lambda: tmp_path / "resources")
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(media_metadata.subprocess, "run", lambda *a, **k: pytest.fail("must not use PATH"))

    with pytest.raises(FileNotFoundError):
        media_metadata.format_tags(tmp_path / "episode.mkv")


def test_development_metadata_reader_uses_path_only_without_a_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "resource_root", lambda: tmp_path / "resources_without_ffmpeg")
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=b";FFMETADATA1\ntitle=dev\n")

    monkeypatch.setattr(media_metadata.subprocess, "run", run)

    assert media_metadata.format_tags(tmp_path / "episode.mkv") == {"title": "dev"}
    assert calls[0][0][0] == "ffmpeg"


def test_source_facts_keeps_filename_fallback_when_media_metadata_cannot_be_read(monkeypatch, tmp_path):
    def corrupt_media(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["ffmpeg"])

    monkeypatch.setattr(media_metadata, "format_tags", corrupt_media)
    facts = scene_names.source_facts(tmp_path / "series-alpha-137.mkv")

    assert facts["title"] == "series-alpha-137"
    assert facts["episode"] == 137 and facts["cast"] == []


def test_actual_stereo_mkv_global_tags_use_bundled_ffmpeg_without_path(tmp_path, monkeypatch):
    fixture = tmp_path / "scene-source-metadata.mkv"
    ffmpeg = paths.resource_root() / "bin" / "ffmpeg.exe"
    drive_example = "".join(("Q", chr(58), chr(92), "旅"))
    story_example = "".join(("Q", chr(58), chr(92), "story"))
    description = ("配信情報 = 特別編; #案内 " + story_example + "\n"
                   + "■キャスト\nハル：声優甲\nカイ：声優乙\n■公式サイト\nhttps://example.jp/series-alpha2023/?x=1#site")
    title = "アニメ「星空冒険団」第137話 = #1; " + drive_example
    subprocess.run([str(ffmpeg), "-nostdin", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=8000:cl=stereo",
                    "-t", "1", "-c:a", "pcm_s16le", "-metadata", "title=" + title,
                    "-metadata", "description=" + description, str(fixture)], check=True, timeout=60)
    expected_source_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
    expected_pcm_sha = hashlib.sha256(bytes(32000)).hexdigest()

    def pcm_witness() -> str:
        decoded = subprocess.run([str(ffmpeg), "-nostdin", "-v", "error", "-i", str(fixture), "-map", "0:a:0",
                                  "-f", "s16le", "-acodec", "pcm_s16le", "-"],
                                 capture_output=True, check=True, timeout=60).stdout
        assert len(decoded) == 32000
        return hashlib.sha256(decoded).hexdigest()

    before_source = hashlib.sha256(fixture.read_bytes()).hexdigest()
    before_pcm = pcm_witness()
    monkeypatch.setenv("PATH", "")
    facts = scene_names.source_facts(fixture)
    after_pcm = pcm_witness()
    after_source = hashlib.sha256(fixture.read_bytes()).hexdigest()

    assert before_source == after_source == expected_source_sha
    assert before_pcm == after_pcm == expected_pcm_sha
    assert facts["title"] == title
    assert facts["episode"] == 137
    assert facts["cast"] == ["ハル", "カイ"]
    assert facts["site"] == "https://example.jp/series-alpha2023/?x=1#site"
    assert facts["description"] == description
