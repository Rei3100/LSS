"""Independent regression (contributed by the Codex verifier, codex/quit-assistance-20261013 89b7e09): missing evidence never authorizes deleting unknown files.

Runs the Product BGM entry twice. Only GPU inference is replaced by the existing
deterministic test fixture; decode, candidates, record validation and retry use
the real implementation. No installed runtime, shared pipe or GPU is accessed.
"""
import hashlib
import shutil

import pytest
from mutagen.flac import FLAC

from lss import application
from tests.test_application_bgm_candidates import _run, _source, gpu


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("damage", ["missing_music", "missing_cue_evidence"])
def test_unverifiable_completed_run_preserves_unowned_state(tmp_path, gpu, damage):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.noncontinuous_music = True  # held-candidate precondition (test_application_bgm_candidates)
    first = _run(tmp_path, source)
    assert first.disposition is application.ResultDisposition.HELD, first.message
    candidates = first.held_artifacts[0].parent
    original_record = first.record_path.read_bytes()

    note = candidates / "user-owned-listening-notes.txt"
    note.write_text("2:10 のフルートが好き", encoding="utf-8")
    foreign = candidates / "foreign-source-candidate.flac"
    shutil.copyfile(first.held_artifacts[0], foreign)
    tags = FLAC(foreign)
    tags["LSS_INPUT_SHA256"] = "0" * 64
    tags.save()
    unreadable = candidates / "unreadable-user-audio.flac"
    unreadable.write_bytes(b"unreadable is not proof of LSS ownership")
    protected = {p: _sha(p) for p in (note, foreign, unreadable)}

    if damage == "missing_music":
        first.held_artifacts[0].unlink()
    else:
        first.evidence_path.unlink()
    gpu.gain = 0.98
    retried = _run(tmp_path, source)
    assert retried.disposition is application.ResultDisposition.HELD, retried.message
    assert len(gpu.calls) == 2
    lost = [str(p) for p, digest in protected.items() if not p.is_file() or _sha(p) != digest]
    assert lost == [], f"Unverifiable completed record deleted/changed unowned files: {lost}"
    archive = first.record_path.parent / "invalid-runs" / (hashlib.sha256(original_record).hexdigest() + ".json")
    assert archive.read_bytes() == original_record
