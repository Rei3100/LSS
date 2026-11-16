"""意味的roleの正解を人手で固定する作業票と、その取り込み。空欄は埋めない。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from lss.validation.bgm_corpus import load_bgm_corpus
from tests.test_application_bgm_identity import routed_result, _execute  # noqa: F401

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "bgm_labeling.py"


@pytest.fixture
def labeling():
    spec = importlib.util.spec_from_file_location("bgm_labeling_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def drafted(labeling, routed_result, tmp_path):
    outcome = _execute(routed_result)
    path = labeling.draft(outcome.record_path, tmp_path / "labels" / "worklist.json")
    return outcome, path, json.loads(path.read_text(encoding="utf-8"))


def label(worklist_path: Path, roles: list[str], wanted: list[bool],
          novel: list[bool | None] | None = None) -> Path:
    worklist = json.loads(worklist_path.read_text(encoding="utf-8"))
    for index, occurrence in enumerate(worklist["occurrences"]):
        occurrence["semantic_role"] = roles[index]
        occurrence["wanted"] = wanted[index]
        occurrence["novel"] = (novel or [None] * len(roles))[index]
    worklist["status"] = labeled_status = "HUMAN_LABELS_COMPLETE"
    assert labeled_status
    worklist_path.write_text(json.dumps(worklist, ensure_ascii=False), encoding="utf-8")
    return worklist_path


def test_a_draft_leaves_every_judgement_empty_and_shows_the_choices(labeling, routed_result, tmp_path):
    outcome, path, worklist = drafted(labeling, routed_result, tmp_path)
    assert worklist["status"] == "AWAITING_HUMAN_LABELS"
    assert len(worklist["occurrences"]) == len(outcome.held_artifacts) == 2
    assert all(item["semantic_role"] is None and item["wanted"] is None
               for item in worklist["occurrences"])
    assert "unresolved_music_like" not in worklist["semantic_role_choices"]
    assert "dramatic_score" in worklist["semantic_role_choices"]
    for item in worklist["occurrences"]:
        assert Path(item["audio_path"]).is_file() and item["seconds"] > 0
    with pytest.raises(labeling.LabelingError, match="既にあります"):
        labeling.draft(outcome.record_path, path)


def test_a_draft_reads_the_held_result_without_changing_anything(labeling, routed_result, tmp_path):
    outcome = _execute(routed_result)
    watched = [Path(outcome.record_path), Path(outcome.evidence_path), *outcome.held_artifacts]
    before = {path: path.read_bytes() for path in watched}
    labeling.draft(outcome.record_path, tmp_path / "worklist.json")
    assert {path: path.read_bytes() for path in watched} == before


@pytest.mark.parametrize("changes,message", [
    ({}, "semantic_role"),
    ({"semantic_role": "unresolved_music_like"}, "semantic_role"),
    ({"semantic_role": "score"}, "semantic_role"),
    ({"semantic_role": "dramatic_score"}, "wanted"),
    ({"semantic_role": "dramatic_score", "wanted": "yes"}, "wanted"),
    ({"semantic_role": "dramatic_score", "wanted": True, "novel": "new"}, "novel"),
    ({"semantic_role": "musical_sfx", "wanted": False, "novel": True}, "novel"),
])
def test_an_unfilled_or_invalid_label_is_refused(labeling, routed_result, tmp_path, changes, message):
    _, path, worklist = drafted(labeling, routed_result, tmp_path)
    worklist["occurrences"][0].update(changes)
    worklist["occurrences"][1].update({"semantic_role": "dramatic_score", "wanted": True})
    path.write_text(json.dumps(worklist, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(labeling.LabelingError, match=message):
        labeling.compile_manifest(path, tmp_path / "corpus.jsonl", corpus_id="c", revision="r1",
                                  case_id="case", description="説明")
    assert not (tmp_path / "corpus.jsonl").exists()


@pytest.mark.parametrize("roles,wanted,novel,outcome", [
    (["dramatic_score", "musical_sfx"], [True, False], [True, None], "wanted_present"),
    (["musical_sfx", "source_music"], [False, False], [None, None], "music_not_wanted"),
    (["dramatic_score", "dramatic_score"], [True, True], [False, False], "no_novel_asset"),
])
def test_filled_labels_become_a_loadable_corpus_with_the_derived_outcome(
        labeling, routed_result, tmp_path, roles, wanted, novel, outcome):
    _, path, _ = drafted(labeling, routed_result, tmp_path)
    label(path, roles, wanted, novel)
    manifest = labeling.compile_manifest(path, tmp_path / "corpus.jsonl", corpus_id="lss_real",
                                         revision="r1", case_id="series-alpha-137", description="実番組")
    corpus = load_bgm_corpus(manifest, verify_sources=True)
    assert len(corpus.cases) == 1
    case = corpus.cases[0]
    assert case.expected_outcome == outcome
    assert [item.semantic_role for item in case.occurrences] == roles
    assert [item.wanted for item in case.occurrences] == wanted
    assert all(item.identity_state == "unresolved" for item in case.occurrences)
    assert "human_labelled" in case.attributes


def test_labels_are_refused_when_the_held_result_changed_after_drafting(labeling, routed_result, tmp_path):
    outcome, path, _ = drafted(labeling, routed_result, tmp_path)
    label(path, ["dramatic_score", "musical_sfx"], [True, False])
    record = Path(outcome.record_path)
    record.write_bytes(record.read_bytes() + b" ")
    with pytest.raises(labeling.LabelingError, match="保留結果"):
        labeling.compile_manifest(path, tmp_path / "corpus.jsonl", corpus_id="c", revision="r1",
                                  case_id="case", description="説明")


def test_labels_are_refused_when_the_analysis_signal_changed(labeling, routed_result, tmp_path):
    _, path, worklist = drafted(labeling, routed_result, tmp_path)
    label(path, ["dramatic_score", "musical_sfx"], [True, False])
    signal = Path(worklist["source"]["path"])
    signal.write_bytes(signal.read_bytes() + b" ")
    with pytest.raises(labeling.LabelingError, match="原本"):
        labeling.compile_manifest(path, tmp_path / "corpus.jsonl", corpus_id="c", revision="r1",
                                  case_id="case", description="説明")


def test_a_worklist_from_somewhere_else_is_refused(labeling, tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"type": "something_else", "occurrences": []}), encoding="utf-8")
    with pytest.raises(labeling.LabelingError, match="作業票"):
        labeling.compile_manifest(other, tmp_path / "corpus.jsonl", corpus_id="c", revision="r1",
                                  case_id="case", description="説明")


def test_candidate_audio_change_is_refused_before_creating_a_corpus(labeling, routed_result, tmp_path):
    _, path, worklist = drafted(labeling, routed_result, tmp_path)
    label(path, ["dramatic_score", "musical_sfx"], [True, False])
    candidate = Path(worklist["occurrences"][0]["audio_path"])
    candidate.write_bytes(candidate.read_bytes() + b"late")
    with pytest.raises(labeling.LabelingError, match="保留結果"):
        labeling.compile_manifest(path, tmp_path / "corpus.jsonl", corpus_id="c", revision="r1",
                                  case_id="case", description="説明")
    assert not (tmp_path / "corpus.jsonl").exists()


def test_duplicate_worklist_keys_are_refused(labeling, routed_result, tmp_path):
    _, path, _ = drafted(labeling, routed_result, tmp_path)
    content = path.read_text(encoding="utf-8")
    content = content.replace(
        '"status": "AWAITING_HUMAN_LABELS",',
        '"status": "AWAITING_HUMAN_LABELS", "status": "HUMAN_LABELS_COMPLETE",', 1)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(labeling.LabelingError, match="重複"):
        labeling.compile_manifest(path, tmp_path / "corpus.jsonl", corpus_id="c", revision="r1",
                                  case_id="case", description="説明")


def test_late_worklist_change_leaves_no_partial_corpus(labeling, routed_result, tmp_path, monkeypatch):
    _, path, _ = drafted(labeling, routed_result, tmp_path)
    label(path, ["dramatic_score", "musical_sfx"], [True, False])
    output = tmp_path / "corpus.jsonl"
    original_load = labeling.load_bgm_corpus
    changed = False

    def change_during_validation(manifest, *, verify_sources=True):
        nonlocal changed
        if not changed:
            changed = True
            path.write_bytes(path.read_bytes() + b"\n")
        return original_load(manifest, verify_sources=verify_sources)

    monkeypatch.setattr(labeling, "load_bgm_corpus", change_during_validation)
    with pytest.raises(labeling.LabelingError, match="作業票"):
        labeling.compile_manifest(path, output, corpus_id="c", revision="r1",
                                  case_id="case", description="説明")
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}-*.tmp"))


def test_recording_identity_field_cannot_be_removed(labeling, routed_result, tmp_path):
    _, path, worklist = drafted(labeling, routed_result, tmp_path)
    label(path, ["dramatic_score", "musical_sfx"], [True, False])
    del worklist["occurrences"][0]["recording_identity"]
    path.write_text(json.dumps(worklist, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(labeling.LabelingError, match="録音照合"):
        labeling.compile_manifest(path, tmp_path / "corpus.jsonl", corpus_id="c", revision="r1",
                                  case_id="case", description="説明")


def test_output_race_cannot_overwrite_a_competing_corpus(labeling, routed_result, tmp_path, monkeypatch):
    _, path, _ = drafted(labeling, routed_result, tmp_path)
    label(path, ["dramatic_score", "musical_sfx"], [True, False])
    output = tmp_path / "corpus.jsonl"
    competitor = b"a corpus written by another process"
    original_rename = Path.rename

    def compete(source, target):
        if Path(target) == output:
            output.write_bytes(competitor)
            raise FileExistsError("corpus appeared")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", compete)
    with pytest.raises(labeling.LabelingError, match="上書きしません"):
        labeling.compile_manifest(path, output, corpus_id="c", revision="r1",
                                  case_id="case", description="説明")
    assert output.read_bytes() == competitor
    assert not list(output.parent.glob(f".{output.name}-*.tmp"))
