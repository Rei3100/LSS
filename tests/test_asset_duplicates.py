"""重複の整理は確かな同一性だけで報告し、何も削除しない。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from lss.assets import duplicates
from lss.core import session
from tests.test_bgm_identity import RATE, _dialogue_and_noise, _music


def test_the_same_file_processed_from_two_places_is_reported(tmp_path):
    state = tmp_path / "state"
    first = tmp_path / "a" / "song.flac"
    second = tmp_path / "b" / "song copy.flac"
    other = tmp_path / "other.flac"
    for path, seed in ((first, 1), (second, 1), (other, 2)):
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.random.default_rng(seed).standard_normal((1000, 2)) * 0.1, 44100, subtype="PCM_16")
    for path in (first, second, other):
        output = path.with_name(path.stem + " - Instrumental.flac")
        output.write_bytes(b"out" + path.name.encode())
        session.mark_completed(session.input_sha256_of(path), "separation-v1-x", source_path=path, outputs=[output],
                               state_root=state)
    second.unlink()

    [group] = duplicates.same_file_groups(state)
    assert group["source_paths"] == sorted([str(first.resolve()), str(second.resolve())])
    assert group["present"] == [str(first.resolve())]
    assert len(group["outputs"]) == 2
    assert first.is_file()


def test_different_files_of_the_same_recording_are_grouped_but_quotes_and_others_are_not(tmp_path):
    song = _music(31, 40.0)
    signals = {
        "master": song,
        "quiet_copy": song * 0.4,  # 音量の違う別ファイル（同じ録音）
        "excerpt": song[int(10 * RATE):int(18 * RATE)] + _dialogue_and_noise(8.0) * 0.01,  # 短い引用
        "another": _music(32, 40.0),
        "arrangement": _music(31, 40.0, performance=4),  # 同じ譜面の別演奏
    }
    paths = []
    for name in signals:
        path = tmp_path / f"{name}.flac"
        path.write_bytes(name.encode())
        paths.append(path)
    groups = duplicates.same_recording_groups(paths, loader=lambda p: signals[Path(p).stem])

    assert [sorted(Path(item["path"]).stem for item in group["files"]) for group in groups] == [["master", "quiet_copy"]]
    assert all(item["covered_fraction"] >= 0.5 for item in groups[0]["evidence"])
