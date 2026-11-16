"""Normal BGM retry must preserve unowned files with an LSS-like temporary name."""
import hashlib
import shutil

import pytest
from mutagen.flac import FLAC

from lss import application
from tests.test_application_bgm_candidates import _run, _source, gpu


@pytest.mark.parametrize("payload", ["empty", "flac_header", "complete_user_audio"])
def test_retry_preserves_unowned_temporary_lookalike(tmp_path, gpu, payload):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.noncontinuous_music = True  # held-candidate precondition (test_application_bgm_candidates)
    first = _run(tmp_path, source)
    assert first.disposition is application.ResultDisposition.HELD, first.message
    first.record_path.unlink()  # Genuine interrupted state: completed record absent.
    protected = first.held_artifacts[0].parent / ".lss-candidate-user-owned.tmp"
    if payload == "complete_user_audio":
        shutil.copyfile(first.held_artifacts[0], protected)
        tags = FLAC(protected)
        tags.clear()
        tags["TITLE"] = "My independently retained audio"
        tags.save()
    else:
        protected.write_bytes(b"" if payload == "empty" else b"fLaC user-owned recovery notes")
    digest = hashlib.sha256(protected.read_bytes()).hexdigest()
    gpu.gain = 0.98
    retried = _run(tmp_path, source)
    assert retried.disposition is application.ResultDisposition.HELD, retried.message
    assert protected.is_file() and hashlib.sha256(protected.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("prefix", [".lss-bgm-", ".lss-listening-"])
def test_staging_cleanup_preserves_unowned_temporary_audio(tmp_path, prefix):
    from lss.bgm import current_best
    import numpy as np
    import soundfile as sf

    folder = tmp_path / "LSS" / current_best.STAGING_DIR / "run"
    folder.mkdir(parents=True)
    protected = folder / (prefix + "user-owned.tmp")
    sf.write(protected, np.zeros((800, 2)), 8000, format="FLAC")
    tags = FLAC(protected)
    tags["TITLE"] = "User audio without LSS ownership"
    tags.save()
    original = protected.read_bytes()
    application._discard_unpublished_staging(folder, "0" * 64, tmp_path / "state")
    assert protected.is_file() and protected.read_bytes() == original
