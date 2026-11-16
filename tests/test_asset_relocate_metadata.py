"""再生ソフトがtagを書き戻した成果物を、失われたものとして探し直さない。"""
from __future__ import annotations

import shutil

import numpy as np
import soundfile as sf
from mutagen.flac import FLAC

from lss.assets import relocate
from lss.core import audio_identity, session
from tests.test_reprocess_candidates import _source


def played(path):
    tags = FLAC(path)
    tags["LSS_TEST_PLAY_COUNT"] = ["7"]
    tags.save()
    return path


def accompaniment(path, seed=3, frames=8192):
    rng = np.random.default_rng(seed)
    sf.write(path, rng.standard_normal((frames, 2)) * 0.2, 44100, subtype="PCM_16")
    return path


def completed(tmp_path, state):
    (tmp_path / "music").mkdir()
    source, sha = _source(tmp_path / "music" / "song.flac", b"song")
    output = accompaniment(tmp_path / "music" / "song - Instrumental.flac")
    session.mark_completed(sha, "separation-v1-new", "2026-09-01", source_path=source,
                           outputs=[output], state_root=state)
    return source, sha, output


def test_a_played_output_at_its_recorded_place_is_not_searched_for(tmp_path):
    state = tmp_path / "state"
    _, _, output = completed(tmp_path, state)
    before = output.read_bytes()
    played(output)
    assert output.read_bytes() != before
    assert relocate.find_moved([tmp_path], state_root=state) == []


def test_a_moved_and_played_output_is_found_by_its_samples(tmp_path):
    state = tmp_path / "state"
    _, _, output = completed(tmp_path, state)
    archive = tmp_path / "archive"
    archive.mkdir()
    moved = archive / "別名 - 伴奏.flac"
    shutil.move(output, moved)
    played(moved)
    accompaniment(archive / "unrelated.flac", seed=9)
    moves = relocate.find_moved([archive], state_root=state)
    assert [(move["kind"], move["found_at"]) for move in moves] == [
        ("output", [str(moved.resolve())])]
    relocate.record_relocations(moves, state_root=state)
    assert relocate.current_locations(moves[0]["sha256"], state_root=state) == [moved]
    # 同じ場所で音が変わったら、その所在はもう返さない。
    accompaniment(moved, seed=11)
    assert relocate.current_locations(moves[0]["sha256"], state_root=state) == []


def test_files_whose_samples_differ_are_not_offered_as_relocations(tmp_path):
    state = tmp_path / "state"
    _, _, output = completed(tmp_path, state)
    archive = tmp_path / "archive"
    archive.mkdir()
    moved = archive / "伴奏.flac"
    shutil.move(output, moved)
    accompaniment(moved, seed=13)
    assert relocate.find_moved([archive], state_root=state) == []


def test_only_same_shaped_candidates_are_decoded_during_the_search(tmp_path, monkeypatch):
    state = tmp_path / "state"
    _, _, output = completed(tmp_path, state)
    archive = tmp_path / "archive"
    archive.mkdir()
    shutil.move(output, archive / "伴奏.flac")
    played(archive / "伴奏.flac")
    accompaniment(archive / "long.flac", seed=5, frames=200000)
    accompaniment(archive / "same-shape.flac", seed=17)
    decoded = []
    original = audio_identity.audio_identity
    monkeypatch.setattr(relocate.audio_identity, "audio_identity",
                        lambda path: (decoded.append(path), original(path))[1])
    assert len(relocate.find_moved([archive], state_root=state)) == 1
    # 時間軸・左右・精度が記録と合う候補だけを復号する。長さの違うものは読まない。
    assert set(decoded) == {archive / "伴奏.flac", archive / "same-shape.flac"}
