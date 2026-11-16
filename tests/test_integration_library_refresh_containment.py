"""Integration: the series-album refresh of other episodes obeys the write-target containment.

Two lines met here: the video-metadata lane writes other sources' series tags at their CurrentBest public file, and
the invariant audit allows a presentation refresh only through ``write_targets.checked``. One rule: a row that names a
file the state may not write (adopted by another state, not this cue, a link) is recorded pending and never written.
"""
from __future__ import annotations

from pathlib import Path

from lss.bgm import current_best, listening, write_targets


def _changed(sha="a" * 64):
    return [{"asset_id": "bgm-cue:k1", "source_sha256": sha, "tags": {"ALBUM": ["x"]}}]


def test_refresh_elsewhere_writes_only_checked_targets(tmp_path, monkeypatch):
    public = tmp_path / "lib" / "cue.flac"
    public.parent.mkdir()
    public.write_bytes(b"x")
    row = {"public_path": str(public), "pcm_sha256": "p" * 64}
    monkeypatch.setattr(current_best, "published_files", lambda state, sha: {"k1": row})
    written = []
    monkeypatch.setattr(listening, "finish", lambda path, state, **kw: written.append(Path(path)))

    def refuse(state, found, cue_key=None):
        raise write_targets.WriteTargetRefused("adopted_by_another_state", found["public_path"])

    monkeypatch.setattr(write_targets, "checked", refuse)
    result = listening._refresh_library_elsewhere(tmp_path, _changed())
    assert written == [] and result["refreshed"] == 0
    assert result["pending"][0]["reason"] == "adopted_by_another_state"

    monkeypatch.setattr(write_targets, "checked", lambda state, found, cue_key=None: Path(found["public_path"]))
    result = listening._refresh_library_elsewhere(tmp_path, _changed())
    assert written == [public] and result["refreshed"] == 1
