import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.core.jobs import SeparationQueue
from lss.core.session import iter_input_files


def _track(path: Path, *, album="Album", track=None, disc=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros((32, 2), dtype="float32"), 44100)
    tags = FLAC(path)
    if album is not None:
        tags["ALBUM"] = album
    if track is not None:
        tags["TRACKNUMBER"] = track
    if disc is not None:
        tags["DISCNUMBER"] = disc
    tags.save()
    return path.resolve()


def test_explicit_and_folder_import_share_natural_order_and_deduplication(tmp_path):
    paths = [_track(tmp_path / f"{n}.flac", album=None) for n in (10, 2, 1)]
    expected = [paths[2], paths[1], paths[0]]
    assert iter_input_files(paths) == expected
    assert iter_input_files([tmp_path]) == expected
    assert iter_input_files([paths[0], tmp_path, paths[1]]) == expected


def test_album_tags_order_numeric_disc_and_track_across_folders(tmp_path):
    third = _track(tmp_path / "alpha" / "A.flac", track="1/2", disc="2/2")
    second = _track(tmp_path / "beta" / "A.flac", track="10/10", disc="1/2")
    first = _track(tmp_path / "beta" / "Z.flac", track="2/10", disc="1/2")
    assert iter_input_files([third, second, first]) == [first, second, third]
    assert iter_input_files([tmp_path]) == [first, second, third]


@pytest.mark.parametrize("track,disc", [
    (None, "1"), ("oops", "1"), ("0", "1"), ("2/1", "1"),
    ("1", "1"), ("2", "0"), ("2", "oops"), (["2", "3"], "1"),
])
def test_incomplete_ambiguous_or_invalid_numbering_uses_natural_fallback(tmp_path, track, disc):
    first = _track(tmp_path / "1.flac", track=track, disc=disc)
    second = _track(tmp_path / "2.flac", track="1", disc="1")
    assert iter_input_files([second, first]) == [first, second]


def test_mixed_albums_do_not_interleave_track_numbers(tmp_path):
    a2 = _track(tmp_path / "A-1.flac", album="A", track="2")
    b1 = _track(tmp_path / "B-2.flac", album="B", track="1")
    a1 = _track(tmp_path / "A-3.flac", album="A", track="1")
    b2 = _track(tmp_path / "B-4.flac", album="B", track="2")
    assert iter_input_files([b2, a2, b1, a1]) == [a1, a2, b1, b2]


def test_corrupt_metadata_does_not_prevent_input_registration(tmp_path):
    paths = [tmp_path / f"{n}.flac" for n in (10, 1, 2)]
    for path in paths:
        path.write_bytes(b"invalid audio")
    assert iter_input_files(paths) == [paths[1], paths[2], paths[0]]


def test_normalized_queue_is_durable_and_appends_late_batch_without_reordering(tmp_path):
    state = tmp_path / "queue.json"
    calls = []
    initial = [tmp_path / "10.wav", tmp_path / "2.wav"]
    late = [tmp_path / "3.wav", tmp_path / "1.wav"]

    def dispatch(job):
        calls.append(Path(job["mix_path"]))
        if len(calls) == 1:
            queue.add(late, "test", "Instrumental")
            assert queue.snapshot()[0].status == "running"
            assert [item.path for item in queue.snapshot()] == [initial[1], initial[0], late[1], late[0]]
            queue.pause()
        return {"status": "ok", "outputs": []}

    queue = SeparationQueue(dispatch, state_path=state)
    queue.add(initial, "test", "Instrumental")
    queue.run()
    restored = SeparationQueue(dispatch, state_path=state)
    assert [item.path for item in restored.snapshot()] == [initial[1], initial[0], late[1], late[0]]
    restored.run()
    assert calls == [initial[1], initial[0], late[1], late[0]]


def test_normalized_queue_keeps_verified_reuse_and_only_dispatches_track_three(tmp_path):
    calls = []
    paths = [tmp_path / f"{n}.wav" for n in (3, 1, 2)]

    def existing(item, target):
        return {"k": "reused", "files": []} if item.path.stem in {"1", "2"} else None

    queue = SeparationQueue(
        lambda job: calls.append(Path(job["mix_path"])) or {"status": "ok", "outputs": []},
        state_path=tmp_path / "queue.json", existing=existing,
    )
    queue.add(paths, "test", "Instrumental", targets=["inst"])
    assert [item.path for item in queue.snapshot()] == [paths[1], paths[2], paths[0]]
    for item in queue.snapshot():
        queue.resolve_existing(item.id)
    assert [item.status for item in queue.snapshot()] == ["completed", "completed", "queued"]
    queue.run()
    assert calls == [paths[0]]
    assert [item.outcomes["inst"]["k"] for item in queue.snapshot()[:2]] == ["reused", "reused"]


def test_restored_legacy_queue_order_is_not_rewritten(tmp_path):
    state = tmp_path / "queue.json"
    queue = SeparationQueue(state_path=state)
    queue.add([tmp_path / "1.wav", tmp_path / "3.wav"], "test", "Instrumental")
    payload = json.loads(state.read_text(encoding="utf-8"))
    payload["items"].reverse()
    state.write_text(json.dumps(payload), encoding="utf-8")
    before = state.read_bytes()
    restored = SeparationQueue(state_path=state)
    assert [item.path.name for item in restored.snapshot()] == ["3.wav", "1.wav"]
    assert state.read_bytes() == before
