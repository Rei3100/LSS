from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lss.bgm import scene_names, scene_name_spellings


def source(text):
    return {"source_kind": "official_episode", "url": "https://example.org/episodes/",
            "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


@pytest.mark.parametrize("canonical,observed", [("メガソックウサ", "メガソッグーザ"),
                                                ("メガソックウサ", "メガソックーサ"),
                                                ("スーパー", "スウパア")])
def test_general_voicing_and_long_vowels_keep_raw_observation(canonical, observed):
    line = {"text": observed + "が来た", "start": 100., "end": 104.}
    support = scene_name_spellings.witnessed([canonical + "の導き"], [line], [source(canonical)])
    assert len(support) == 1
    assert support[0]["observed_text"] == line["text"] and support[0]["canonical"] == canonical
    assert support[0]["match"] == "inferred_voicing_long_vowel"
    assert support[0]["source"]["text_sha256"] == source(canonical)["text_sha256"]


@pytest.mark.parametrize("canonical,observed,published", [
    ("ウェーノベル", "ラッチョンレラーラッチョン", "ウェーノベル"),
    ("メガソックウサ", "ソックウサ", "メガソックウサ"),
    ("メガソックウサ", "メガノックーサ", "メガソックウサ"),
    ("ダヌ", "タヌ", "ダヌ"),
    ("タヌタヌ", "ダヌダヌ", "タヌタヌ、ダヌダヌ"),
    ("タヌタヌ", "ダヌダヌ", "ダヌダヌ"),
])
def test_no_unrelated_short_partial_ambiguous_or_unpublished_correction(canonical, observed, published):
    assert not scene_name_spellings.witnessed([canonical], [{"text": observed, "start": 1, "end": 2}], [source(published)])


def test_public_text_hash_mismatch_cannot_authorize_a_correction():
    context = {**source("タヌタヌ"), "text": "ダヌダヌ"}
    assert not scene_name_spellings.source_headers([context])
    assert not scene_name_spellings.witnessed(["ダヌダヌ"], [{"text": "タヌタヌ", "start": 1, "end": 2}], [context])


@pytest.mark.parametrize("cue_id,canonical,accepted", [("voiced", "メガソックウサ", True),
                                                       ("long-vowel", "メガソックウサ", True),
                                                       ("unrelated", "ウェーノベル", False)])
def test_retained_orders_recover_spelling_without_reopening_wrong_scene(tmp_path, cue_id, canonical, accepted):
    from tests.test_bgm_scene_names import _saved_replay_cue

    row, original_facts = _saved_replay_cue(cue_id)
    raw = copy.deepcopy(row["heard"])
    facts = {**original_facts, "published_name_sources": [source(canonical)]}
    entry = scene_names.record_cue(tmp_path, row["asset_id"], pcm_sha256=row["pcm_sha256"],
                                   source_sha256=row["source_sha256"], facts=facts,
                                   start_seconds=row["bounds_seconds"][0], end_seconds=row["bounds_seconds"][1],
                                   heard=row["heard"], order=row["old_order"], model={"id": "saved-order"})
    assert bool(entry["name"]) is accepted
    assert entry["heard"] == raw and row["heard"] == raw
    assert bool(entry["published_spelling_support"]) is accepted
    assert facts["cast"] == []


def test_lead_in_name_cannot_support_occurrence_correction(tmp_path):
    facts = {"work_key": "test", "work_title": "test", "cast": [],
             "published_name_sources": [source("タヌタヌ")]}
    entry = scene_names.record_cue(tmp_path, "cue", pcm_sha256="pcm", source_sha256="source", facts=facts,
                                   start_seconds=100., end_seconds=140.,
                                   heard=[{"start": 90., "end": 99., "text": "ダヌダヌが来た"},
                                          {"start": 100., "end": 105., "text": "誰か来た"}],
                                   order={"name": "訪れる影", "use": "タヌタヌの登場", "evidence": "誰か来た"}, model={})
    assert entry["name"] is None and entry["use"] is None
    assert not entry["published_spelling_support"]
