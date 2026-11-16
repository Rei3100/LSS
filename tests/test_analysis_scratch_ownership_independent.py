"""Unknown analysis neighbors survive a real interrupted BGM entry and retry."""
from pathlib import Path

import pytest

from lss import application
from lss.bgm import episode_cues
from tests.test_application_bgm_candidates import _run, _source, gpu


def test_interrupted_analysis_retry_preserves_unowned_neighbor(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / 'episode.wav')
    gpu.tagger.noncontinuous_music = True  # held-candidate precondition (test_application_bgm_candidates)
    original = episode_cues.run_episode_cues
    notes = []

    def stop_with_user_note(*args, **kwargs):
        [scratch] = tmp_path.rglob('decode-bgm-analysis-*')
        note = scratch / 'user-owned-listening-notes.txt'
        note.write_text('My listening notes are not generated analysis', encoding='utf-8')
        notes.append((note, note.read_bytes()))
        raise KeyboardInterrupt

    monkeypatch.setattr(episode_cues, 'run_episode_cues', stop_with_user_note)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, source)
    monkeypatch.setattr(episode_cues, 'run_episode_cues', original)
    retried = _run(tmp_path, source)
    assert retried.disposition is application.ResultDisposition.HELD, retried.message
    [(note, original_bytes)] = notes
    assert note.is_file() and note.read_bytes() == original_bytes
