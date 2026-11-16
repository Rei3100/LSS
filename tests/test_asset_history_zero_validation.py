"""検証根拠のないゼロ件を、保存済みの良い結果より新しい最良結果にしない。"""
import json

import pytest

from lss.assets import history
from lss.core import session
from lss.harness.tsv_io import load_tsv, save_tsv
from tests.test_reprocess_candidates import CURRENT, _source, _output

VALIDATION = {"validator_id": "controlled-presence", "validator_revision": "v1", "hard_gates_passed": True}


def _ledger(tmp_path, *, outputs="[]", reason="no_wanted_target", validation=None):
    state = tmp_path / "state"
    source, sha = _source(tmp_path / "song.flac", b"source")
    output = _output(tmp_path / "old.flac", model="model_old", model_sha="o" * 64, config_sha="c" * 64)
    session.mark_completed(sha, "separation-v1-old", "2026-08-01", source_path=source, outputs=[output], state_root=state)
    path = session.completion_ledger_path(state)
    header, rows = load_tsv(path)
    rows.append({"input_sha256": sha, "recipe_id": "separation-v1-zero", "source_path": str(source.resolve()),
                 "completed_at": "2026-09-01", "outcome": "zero", "zero_reason": reason,
                 "validation_result": json.dumps(validation), "outputs": outputs})
    save_tsv(path, header, rows)
    return state, sha, path


@pytest.mark.parametrize("validation", [None, {}, {**VALIDATION, "hard_gates_passed": False},
    {**VALIDATION, "validator_id": ""}, {**VALIDATION, "validator_revision": ""},
    {**VALIDATION, "extra": float("nan")}, []])
def test_unvalidated_zero_cannot_replace_a_previous_intact_result(tmp_path, validation):
    state, sha, path = _ledger(tmp_path, validation=validation)
    before = path.read_bytes()
    result = history.history(sha, state_root=state, current=CURRENT)
    assert result["current_best"] == 0
    assert result["revisions"][1]["zero_result_valid"] is False
    assert result["revisions"][0]["zero_result_valid"] is None
    assert len(result["revisions"]) == 2 and path.read_bytes() == before


@pytest.mark.parametrize("outputs", ["null", "{}", '[{"path":"unexpected"}]', "not-json"])
def test_zero_with_nonempty_or_invalid_output_record_is_unusable(tmp_path, outputs):
    state, sha, _ = _ledger(tmp_path, outputs=outputs, validation=VALIDATION)
    result = history.history(sha, state_root=state, current=CURRENT)
    assert result["current_best"] == 0 and result["revisions"][1]["zero_result_valid"] is False


def test_unknown_zero_reason_is_unusable(tmp_path):
    state, sha, _ = _ledger(tmp_path, reason="unknown", validation=VALIDATION)
    result = history.history(sha, state_root=state, current=CURRENT)
    assert result["current_best"] == 0 and result["revisions"][1]["zero_result_valid"] is False


def test_valid_zero_remains_a_usable_result_without_placeholder_audio(tmp_path):
    state, sha, path = _ledger(tmp_path, validation=VALIDATION)
    before = path.read_bytes()
    result = history.history(sha, state_root=state, current=CURRENT)
    assert result["current_best"] == 1 and result["revisions"][1]["zero_result_valid"] is True
    assert result["revisions"][1]["outputs"] == []
    assert path.read_bytes() == before
