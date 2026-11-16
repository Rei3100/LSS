import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from lss.validation.bgm_corpus import InvalidBgmCorpusError, load_bgm_corpus


def _source(tmp_path, *, frames=1000, sample_rate=1000):
    path = tmp_path / "source.wav"
    sf.write(path, np.zeros((frames, 2), dtype=np.float32), sample_rate, subtype="FLOAT")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _meta():
    return {
        "type": "manifest", "schema_version": 1, "corpus_id": "bgm.test", "revision": "r1",
        "description": "試験用BGMコーパス",
    }


def _occurrence(**overrides):
    value = {
        "occurrence_id": "occ-1", "start_frame": 100, "end_frame": 400,
        "semantic_role": "dramatic_score", "wanted": True, "novel": True,
        "identity_state": "unknown", "cue_identity_id": None, "audio_version_id": None,
    }
    value.update(overrides)
    return value


def _case(path, digest, **overrides):
    value = {
        "type": "case", "case_id": "case-1", "description": "台詞下の劇伴",
        "source": {"path": path.name, "sha256": digest, "sample_rate": 1000, "frames": 1000},
        "expected_outcome": "wanted_present", "attributes": ["quiet_score"],
        "occurrences": [_occurrence()],
    }
    value.update(overrides)
    return value


def _write_manifest(tmp_path, *records):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in (_meta(), *records)) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_bgm_corpus_verifies_source_identity_and_layout(tmp_path):
    source, digest = _source(tmp_path)
    manifest = _write_manifest(tmp_path, _case(source, digest))

    corpus = load_bgm_corpus(manifest)

    assert corpus.corpus_id == "bgm.test"
    assert corpus.revision == "r1"
    case = corpus.cases[0]
    assert case.source_path == source.resolve()
    assert case.source_sha256 == digest
    assert case.occurrences[0].semantic_role == "dramatic_score"


def test_source_hash_mismatch_is_rejected(tmp_path):
    source, _ = _source(tmp_path)
    manifest = _write_manifest(tmp_path, _case(source, "0" * 64))

    with pytest.raises(InvalidBgmCorpusError, match="SHA-256が一致しません"):
        load_bgm_corpus(manifest)


def test_source_layout_mismatch_is_rejected(tmp_path):
    source, digest = _source(tmp_path)
    case = _case(source, digest)
    case["source"]["frames"] = 999
    manifest = _write_manifest(tmp_path, case)

    with pytest.raises(InvalidBgmCorpusError, match="layoutがmanifestと一致しません"):
        load_bgm_corpus(manifest)


def test_occurrence_outside_source_is_rejected_before_reading_asset(tmp_path):
    source, digest = _source(tmp_path)
    case = _case(source, digest)
    case["occurrences"] = [_occurrence(end_frame=1001)]
    manifest = _write_manifest(tmp_path, case)

    with pytest.raises(InvalidBgmCorpusError, match="source範囲外"):
        load_bgm_corpus(manifest, verify_sources=False)


def test_duplicate_occurrence_ids_are_rejected(tmp_path):
    source, digest = _source(tmp_path)
    case = _case(source, digest)
    case["occurrences"] = [_occurrence(), _occurrence(start_frame=500, end_frame=600)]
    manifest = _write_manifest(tmp_path, case)

    with pytest.raises(InvalidBgmCorpusError, match="occurrence_idが重複"):
        load_bgm_corpus(manifest, verify_sources=False)


def test_duplicate_case_ids_are_rejected(tmp_path):
    source, digest = _source(tmp_path)
    manifest = _write_manifest(tmp_path, _case(source, digest), _case(source, digest))

    with pytest.raises(InvalidBgmCorpusError, match="case_idが重複"):
        load_bgm_corpus(manifest, verify_sources=False)


@pytest.mark.parametrize(
    "outcome,occurrences,message",
    [
        ("wanted_present", [], "wanted区間がありません"),
        ("no_wanted_score", [_occurrence()], "wanted区間があります"),
        ("music_not_wanted", [], "unwanted区間だけが必要"),
        ("no_novel_asset", [_occurrence(novel=True)], "既知のwanted区間が必要"),
    ],
)
def test_zero_and_publication_outcomes_must_match_occurrences(
    tmp_path, outcome, occurrences, message,
):
    source, digest = _source(tmp_path)
    case = _case(source, digest, expected_outcome=outcome, occurrences=occurrences)
    manifest = _write_manifest(tmp_path, case)

    with pytest.raises(InvalidBgmCorpusError, match=message):
        load_bgm_corpus(manifest, verify_sources=False)


def test_music_not_wanted_case_is_valid(tmp_path):
    source, digest = _source(tmp_path)
    case = _case(
        source, digest, expected_outcome="music_not_wanted",
        occurrences=[_occurrence(semantic_role="musical_sfx", wanted=False, novel=None)],
    )

    corpus = load_bgm_corpus(_write_manifest(tmp_path, case), verify_sources=False)

    assert corpus.cases[0].expected_outcome == "music_not_wanted"


def test_known_exact_requires_cue_and_audio_version_ids(tmp_path):
    source, digest = _source(tmp_path)
    case = _case(source, digest, occurrences=[_occurrence(identity_state="known_exact")])

    with pytest.raises(InvalidBgmCorpusError, match="cue/version識別子が必要"):
        load_bgm_corpus(_write_manifest(tmp_path, case), verify_sources=False)


def test_unknown_manifest_key_is_rejected(tmp_path):
    source, digest = _source(tmp_path)
    case = _case(source, digest, typo_field=True)

    with pytest.raises(InvalidBgmCorpusError, match="未対応のキー"):
        load_bgm_corpus(_write_manifest(tmp_path, case), verify_sources=False)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"type":"manifest","schema_version":1,"corpus_id":"one","corpus_id":"two",'
        '"revision":"r1","description":"test"}\n',
        encoding="utf-8",
    )

    with pytest.raises(InvalidBgmCorpusError, match="JSONキーが重複"):
        load_bgm_corpus(path, verify_sources=False)
