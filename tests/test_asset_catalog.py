"""記録から作り直せる目録で、入力・結果・派生成果・録音・登場区間を区別して引ける。"""
from __future__ import annotations

import json

from lss.assets import catalog
from lss.core import session
from tests.test_reprocess_candidates import _output, _source


def _identity_record(path, source_sha, reference_sha, spans):
    path.write_text(json.dumps({
        "run": {"source_sha256": source_sha},
        "external_evidence_used": [{"kind": "released_reference", "sha256": reference_sha, "path": "C:/ost/cue.flac"}],
        "matches": [{"reference_id": "cue", "reference_sha256": reference_sha, "reference_path": "C:/ost/cue.flac",
                     "query_start_seconds": a, "query_end_seconds": b, "reference_start_seconds": r,
                     "waveform_explained": 0.8, "identity_state": state} for a, b, r, state in spans],
    }), encoding="utf-8")
    return path


def test_catalog_distinguishes_records_and_is_rebuilt_without_changing_them(tmp_path):
    state = tmp_path / "state"
    episode1, sha1 = _source(tmp_path / "ep1.mkv", b"ep1")
    episode2, sha2 = _source(tmp_path / "ep2.mkv", b"ep2")
    output = _output(tmp_path / "ep1 - Instrumental.flac", model="m", model_sha="n" * 64, config_sha="c" * 64)
    session.mark_completed(sha1, "separation-v1-x", "2026-09-01", source_path=episode1, outputs=[output],
                           state_root=state)
    ledger = session.completion_ledger_path(state).read_bytes()
    records = [
        _identity_record(tmp_path / "id1.json", sha1, "d" * 64, [(100.0, 130.0, 5.0, "known_exact"),
                                                                 (300.0, 310.0, 0.0, "unresolved")]),
        _identity_record(tmp_path / "id2.json", sha2, "d" * 64, [(50.0, 70.0, 20.0, "known_exact")]),
    ]

    database = catalog.build(state_root=state, identity_records=records)
    counts = catalog.summary(database=database)
    assert counts == {"sources": 2, "results": 1, "derivatives": 1, "recordings": 1, "occurrences": 2,
                      "recordings_in_multiple_sources": 1}
    occurrences = catalog.occurrences_of("d" * 64, database=database)
    assert {(o["source_sha256"], o["start_seconds"]) for o in occurrences} == {(sha1, 100.0), (sha2, 50.0)}
    assert session.completion_ledger_path(state).read_bytes() == ledger

    # 作り直しは同じ内容になり、失敗しても古い目録を壊さない。
    assert catalog.summary(database=catalog.build(state_root=state, identity_records=records)) == counts
