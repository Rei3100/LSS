"""短い同一ファイルでも、内容hashと全復号波形の完全一致を正の録音根拠にする。"""
from dataclasses import replace
import json

import numpy as np
import pytest
import soundfile as sf

from lss.bgm import identity, identity_run


@pytest.fixture
def short_recording(tmp_path):
    rate = identity.SAMPLE_RATE
    time = np.arange(int(rate * 6.8)) / rate
    samples = (np.sin(2 * np.pi * 440 * time) * 0.1).astype(np.float32)
    path = tmp_path / "short.wav"
    sf.write(path, np.stack([samples, samples], axis=1), rate, subtype="FLOAT")
    loader = lambda p: sf.read(p, dtype="float32", always_2d=True)[0].mean(axis=1)
    index = identity.ReferenceIndex.build([path], loader=loader)
    return path, loader(path), index


def test_identical_short_content_does_not_need_a_null_offset_space(short_recording, monkeypatch):
    path, samples, index = short_recording
    monkeypatch.setattr(identity, "spectrogram", lambda *args: pytest.fail("完全一致では指紋を再計算しない"))
    [match] = identity.match(samples, index, query_content_sha256=identity._sha256(path), query_offset_seconds=12.007)
    assert match.identity_state == "known_exact" and match.reference_sha256 == identity._sha256(path)
    assert match.evidence_kind == "exact_content_sha256_and_decoded_waveform"
    assert match.query_start_seconds == 12.007
    assert match.query_end_seconds == pytest.approx(12.007 + len(samples) / identity.SAMPLE_RATE)
    assert match.reference_start_seconds == 0 and match.offset_seconds == -12.007
    assert match.waveform_null_max is None and match.votes == 0


def test_same_waveform_without_verified_content_hash_is_not_the_content_proof(short_recording):
    _, samples, index = short_recording
    assert not any(m.evidence_kind == "exact_content_sha256_and_decoded_waveform" for m in identity.match(samples, index))


def test_equal_declared_hash_with_different_waveform_is_an_error(short_recording):
    path, samples, index = short_recording
    changed = samples.copy()
    changed[0] += 0.1
    with pytest.raises(ValueError, match="復号波形"):
        identity.match(changed, index, query_content_sha256=identity._sha256(path))


@pytest.mark.parametrize("proof", ["", "path/to/same-recording", "a" * 63, "G" * 64])
def test_bad_hash_does_not_become_identity(short_recording, proof):
    _, samples, index = short_recording
    with pytest.raises(ValueError, match="SHA"):
        identity.match(samples, index, query_content_sha256=proof)


def test_different_hash_does_not_use_the_content_branch(short_recording):
    _, samples, index = short_recording
    assert not any(m.evidence_kind == "exact_content_sha256_and_decoded_waveform"
                   for m in identity.match(samples, index, query_content_sha256="0" * 64))


@pytest.mark.parametrize("kind", ["silence", "too_short", "nonfinite"])
def test_inaudible_or_incomplete_signal_is_not_a_content_recording_proof(short_recording, kind):
    path, samples, index = short_recording
    query = samples.copy()
    if kind == "silence":
        query[:] = 0
    elif kind == "too_short":
        query = query[:1000]
    else:
        query[0] = np.nan
    # ここは完全一致分岐だけの不変条件。弱い照合へ進んでも、完全一致とは記録しない。
    assert identity._exact_content_matches(query, index, identity._sha256(path), 0) == []


def test_old_serialized_matches_remain_readable_with_their_original_evidence_kind(short_recording):
    path, samples, index = short_recording
    [match] = identity.match(samples, index, query_content_sha256=identity._sha256(path))
    record = replace(match, evidence_kind="landmark_and_waveform").to_dict()
    del record["evidence_kind"]
    assert identity.RecordingMatch(**record).evidence_kind == "landmark_and_waveform"


def test_annotation_passes_the_verified_query_file_hash_to_the_same_matcher(short_recording, tmp_path):
    path, _, _ = short_recording
    sha = identity._sha256(path)
    run = tmp_path / "run.json"
    run.write_text(json.dumps({
        "status": identity_run.ROUTED_STATUS,
        "source": {"source_id": "short", "path": str(path), "sha256": sha},
        "rendering": {"signal": {"path": str(path), "sha256": sha}},
        "renderer_id": "development-only", "candidates": [],
    }), encoding="utf-8")
    before = path.read_bytes(), path.stat().st_mtime_ns
    loader = lambda p: sf.read(p, dtype="float32", always_2d=True)[0].mean(axis=1)

    record_path = identity_run.annotate_run(run, [path], loader=loader)
    payload = json.loads(record_path.read_bytes())
    [match] = payload["matches"]
    assert match["reference_sha256"] == sha and match["identity_state"] == "known_exact"
    assert match["evidence_kind"] == "exact_content_sha256_and_decoded_waveform"
    stamp = record_path.stat().st_mtime_ns
    assert identity_run.annotate_run(run, [path], loader=loader) == record_path
    assert record_path.stat().st_mtime_ns == stamp and (path.read_bytes(), path.stat().st_mtime_ns) == before
