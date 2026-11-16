"""単話BGMの製品入口で、手元の参照録音との関係が出力を正しく変えることを、実音声の照合で確かめる（r4の上）。

GPU推論（分離・4分類・タグ付け・伴奏）だけを `test_application_bgm_candidates` の偽物へ置き換え、
参照録音の照合（復号・指紋・波形の説明率・和声）と、区間の判定・公開・記録・CurrentBestは製品の実装をそのまま通す。
参照録音は通常の使い方（参照録音の置き場所 `References` へ置くだけ）で渡す。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC
from scipy.signal import resample_poly

from lss import application
from lss.bgm import current_best, cue_semantics, identity, reference_relations, references, versions
from tests.test_application_bgm_candidates import RATE, _run, gpu, instrumental  # noqa: F401 - fixtures
from tests.test_bgm_identity import _music
from tests.release_receipts import distribution_receipt
from lss.bgm import release_availability

LOW = identity.SAMPLE_RATE
_RELEASED_FIXTURES = set()


@pytest.fixture(autouse=True)
def _controlled_distribution_truth(monkeypatch):
    _RELEASED_FIXTURES.clear()
    original=release_availability.lookup
    monkeypatch.setattr(release_availability,'lookup',lambda sha,metadata=None:
                        distribution_receipt(sha) if sha in _RELEASED_FIXTURES else original(sha,metadata))


@pytest.fixture(autouse=True)
def _local_instrumental_boundary(instrumental):
    # Extra processing can now follow a rejected RAW nomination. Keep the
    # declared test boundary: inference mocked, actual relations/state retained.
    return instrumental
# Formation and selective processing intentionally change the recipe. Reference
# availability still belongs in the run key, rather than the base recipe.
BGM_RECIPE_ID = application._bgm_candidate_recipe()["recipe_id"]


_EPISODE_ONLY = _music(77, 4.0)   # 番組だけの素材（既発売の録音に無い）


def _recording(seed: int = 1, seconds: float = 40.0) -> np.ndarray:
    return _music(seed, seconds)


def _episode(path: Path, music: np.ndarray, *, at: float = 5.0, seconds: int = 30) -> Path:
    """原本の5秒目から `music`（11025Hzの合成）を置いた2chの番組。曲の外は小さな雑音。"""
    values = np.random.default_rng(0).normal(0, 0.001, (seconds * RATE, 2)).astype(np.float32)
    high = resample_poly(music.astype(np.float64), RATE // LOW, 1).astype(np.float32)
    first = int(at * RATE)
    values[first:first + high.size, 0] += high
    values[first:first + high.size, 1] += 0.8 * high
    sf.write(path, values, RATE, subtype="PCM_24")
    return path


def _library(tmp_path: Path) -> Path:
    """試験の参照録音の置き場所（`gpu` が `LSS_REFERENCES_DIR` に設定した場所）。"""
    return Path(os.environ[references.LIBRARY_ENV])


def _place(tmp_path, recording: np.ndarray, *, name: str = "released.flac", title: str | None = "既発売の曲",
           folder: str = "Series OST", independently_available: bool = True) -> Path:
    """利用者が参照録音の置き場所へ既発売の音源を置く（登録の操作は無い）。"""
    path = _library(tmp_path) / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), recording, LOW, subtype="PCM_16")
    if title is not None:
        tags = FLAC(path)
        tags["title"], tags["version_info"], tags["artist"] = title, "TV Ver.", "作曲者"
        tags["play_count"] = "7"  # 再生ソフトの値は写さない
        tags.save()
    if independently_available:
        _RELEASED_FIXTURES.add(application.file_sha256(path))
    return path


def _semantics(outcome) -> dict:
    return json.loads((outcome.record_path.parent / "semantics" / "cue-semantics.json").read_text("utf-8"))


def _run_record(outcome) -> dict:
    assert outcome.record_path is not None, outcome.message
    return json.loads(outcome.record_path.read_text("utf-8"))


def _exact_case(tmp_path) -> tuple[Path, np.ndarray]:
    recording = _recording()
    return _episode(tmp_path / "episode.wav", recording[int(10 * LOW):int(30 * LOW)]), recording


def test_a_run_without_references_keeps_the_current_base_recipe_identity(tmp_path, gpu):
    recipe = application._bgm_candidate_recipe()
    assert recipe["recipe_id"] == BGM_RECIPE_ID
    assert recipe["recipe_id"] != "bgm-episode-candidates-v13-5a82164decf6f720"
    assert "reference_relations" not in recipe and "reference_policy" not in recipe
    source, _ = _exact_case(tmp_path)
    gpu.tagger.singing = 0.0

    outcome = _run(tmp_path, source)

    key = outcome.record_path.parent.parent.name
    assert key == application._bgm_candidate_run_key(application.file_sha256(source), BGM_RECIPE_ID)
    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    provenance = json.loads(FLAC(outcome.outputs[0])["LSS_BGM_PROVENANCE"][0])
    assert provenance["identity_state"] == "unresolved_no_registered_reference"
    assert "reference_relation" not in provenance
    semantics = _semantics(outcome)
    assert "reference_relations" not in semantics
    assert all("reference_relation" not in s for s in semantics["segments"])
    run = _run_record(outcome)
    assert "references" not in run and "reference_gate_seconds" not in run["cost"]


def test_an_empty_or_missing_reference_library_is_the_same_as_no_references(tmp_path, gpu):
    source, _ = _exact_case(tmp_path)
    gpu.tagger.singing = 0.0
    _library(tmp_path).mkdir(parents=True)
    (_library(tmp_path) / "notes.txt").write_text("音声ではない", encoding="utf-8")

    outcome = _run(tmp_path, source)

    assert outcome.record_path.parent.parent.name == application._bgm_candidate_run_key(
        application.file_sha256(source), BGM_RECIPE_ID)
    assert outcome.disposition is application.ResultDisposition.PUBLISHED


def test_an_exact_released_recording_in_the_library_is_recorded_but_not_extracted_again(tmp_path, gpu):
    source, recording = _exact_case(tmp_path)
    gpu.tagger.singing = 0.0
    reference = _place(tmp_path, recording)

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.HELD, outcome.message
    assert outcome.outputs == () and not (tmp_path / "LSS").exists()
    assert "既発売録音と同じ区間1件" in outcome.message
    semantics = _semantics(outcome)
    assert semantics["counts"]["KNOWN_RELEASED_MUSIC"] == 1
    [segment] = [s for s in semantics["segments"] if s["semantic_class"] == cue_semantics.KNOWN_RELEASED]
    assert segment["reasons"] == [reference_relations.REASON_EXACT]
    assert segment["acoustic_class"] == cue_semantics.WANTED  # 既発売でなければ公開していた
    # 使用箇所は消さない：保留（非公開）として音を残し、理由を名前で残す。
    assert segment["scope"]["disposition"] == "HELD" and segment["scope"]["reason"] == "known_released_recording"
    assert segment["scope"]["retention"]["state"] == "retained_internal"
    record = semantics["reference_relations"]
    assert record["state"] == "checked" and record["reference_count"] == 1
    [occurrence] = record["occurrences"]
    assert occurrence["reference_sha256"] == application.file_sha256(reference)
    assert occurrence["display_title"] == "既発売の曲 [TV Ver.]"
    assert occurrence["tags"] == {"title": "既発売の曲", "version_info": "TV Ver.", "artist": "作曲者"}
    assert occurrence["reference_range_seconds"][0] == pytest.approx(10.0, abs=1.1)
    assert occurrence["extraction"] == "not_created_independent_recording_available"
    assert record["episode_classification"][reference_relations.EXACT]["segments"] == 1
    # 既発売と決めた区間には、2回目・歌唱回収・伴奏を使わない（推論は1回目の分離とタグ付けだけ）。
    run = _run_record(outcome)
    assert len(gpu.calls) == 1 and gpu.four_stem.calls == []
    assert run["cost"]["heavy_input_seconds"]["sfc_second_pass"] == 0
    assert set(run["cost"]["reference_gate_seconds"]) >= {"index_seconds", "match_seconds", "relation_seconds"}
    assert len(run["held"]) == 1 and Path(run["held"][0]["music"]["path"]).is_file()
    # 参照の内容と方針はrunの識別子と記録に入る。参照の出所は置き場所。
    policy = application._bgm_reference_policy()
    assert run["references"]["policy_id"] == policy["policy_id"]
    assert run["references"]["sources"]["library"]["count"] == 1
    assert run["references"]["sources"]["registered_sets"] == 0
    assert outcome.record_path.parent.parent.name == application._bgm_candidate_run_key(
        application.file_sha256(source), BGM_RECIPE_ID, run["references"]["digest"])
    assert run["recipe_id"] == BGM_RECIPE_ID


def test_a_development_reference_set_and_the_library_give_the_same_decision(tmp_path, gpu):
    """登録の経路（開発用の集まり）と置き場所は、同じ内容なら同じ識別子・同じ判定になる（重ねて使わない）。"""
    source, recording = _exact_case(tmp_path)
    gpu.tagger.singing = 0.0
    placed = _place(tmp_path, recording)
    references.add_local("series-a", [placed], state_root=tmp_path / "state")

    paths, digest, missing, sources = application._registered_references(tmp_path / "state")
    assert len(paths) == 1 and missing == [] and sources["registered_sets"] == 1
    assert sources["library"]["count"] == 1
    assert digest == application._registered_references(tmp_path / "state")[1]
    outcome = _run(tmp_path, source)
    assert _semantics(outcome)["counts"]["KNOWN_RELEASED_MUSIC"] == 1


def test_a_shortened_cut_of_a_released_recording_is_recorded_and_not_published(tmp_path, gpu):
    """番組が途中を切って短くしただけの既発売の録音は、その録音そのもの：抽出せず使用箇所として記録する（2026-10-19）。"""
    recording = _recording()
    edit = np.concatenate([recording[:int(8 * LOW)], recording[int(24 * LOW):int(36 * LOW)]])
    source = _episode(tmp_path / "episode.wav", edit)
    gpu.tagger.singing = 0.0
    reference = _place(tmp_path, recording)

    outcome = _run(tmp_path, source)

    assert outcome.outputs == (), outcome.message
    [use] = _semantics(outcome)["reference_relations"]["occurrences"]
    assert use["relation"] == reference_relations.EDIT
    assert use["reference_sha256"] == application.file_sha256(reference)
    assert use["extraction"] == "not_created_independent_recording_available"


def test_an_edit_with_episode_only_material_is_still_published_with_its_relation(tmp_path, gpu):
    recording = _recording()
    edit = np.concatenate([recording[:int(8 * LOW)], _EPISODE_ONLY, recording[int(24 * LOW):int(36 * LOW)]])
    source = _episode(tmp_path / "episode.wav", edit)
    gpu.tagger.singing = 0.0
    reference = _place(tmp_path, recording)

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    [published] = outcome.outputs
    provenance = json.loads(FLAC(published)["LSS_BGM_PROVENANCE"][0])
    assert provenance["identity_state"] == "released_recording_edit"
    relation = provenance["reference_relation"]
    assert relation["state"] == reference_relations.EDIT
    assert relation["reference"]["reference_sha256"] == application.file_sha256(reference)
    assert relation["reference"]["display_title"] == "既発売の曲 [TV Ver.]"
    # The file path never comes from a reference name. The listening title names the verified relation
    # provisionally (time in the episode + "<release title> — 関連版（仮題）"); it is not an official title.
    assert published.name.startswith("episode - BGM ")
    title = FLAC(published)["title"][0]
    assert title.endswith("既発売の曲 [TV Ver.] — 関連版（仮題）") and title[2] == ":"
    assert FLAC(published)["LSS_NAMING_KIND"] == ["provisional_related"]
    assert FLAC(published)["LSS_NAMING_OFFICIAL"] == ["0"]
    assert _semantics(outcome)["reference_relations"]["occurrences"] == []


def test_unregistered_music_over_a_registered_recording_is_still_published(tmp_path, gpu):
    """独立反証 RA-P1-002 の製品入口：登録した録音へ登録していない曲が重なると、参照なしと同じく公開する。

    v1は参照の存在だけで区間全体を既発売にし、公開を0件にしていた。重なった曲は誰の手元にも無い。
    """
    recording = _recording()
    registered = recording[int(10 * LOW):int(30 * LOW)]
    other = _music(55, 20.0)
    other = other * (np.sqrt(np.mean(registered ** 2)) / np.sqrt(np.mean(other ** 2)))
    source = _episode(tmp_path / "episode.wav", (registered + other).astype(np.float32) * 0.7)
    gpu.tagger.singing = 0.0
    reference = _place(tmp_path, recording)

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    semantics = _semantics(outcome)
    assert semantics["counts"].get("KNOWN_RELEASED_MUSIC", 0) == 0
    assert semantics["reference_relations"]["occurrences"] == []
    [published] = outcome.outputs
    provenance = json.loads(FLAC(published)["LSS_BGM_PROVENANCE"][0])
    assert provenance["identity_state"] == "released_recording_with_additional_material"
    relation = provenance["reference_relation"]
    assert relation["state"] == reference_relations.ADDITIONAL
    assert relation["reference"]["reference_sha256"] == application.file_sha256(reference)
    [related] = semantics["reference_relations"]["related_occurrences"]
    assert related["relation"] == reference_relations.ADDITIONAL and related["extraction"] == "normal_decision_kept"


def test_the_work_soundtrack_is_found_acquired_and_used_without_placing_files(tmp_path, gpu, monkeypatch):
    """通常の使い方：利用者は参照音源を置かない。LSSが作品のサウンドトラックを探し、取得し、番組の音声で確かめて使う。

    検索と取得は偽物（ネットワークへは出ない）。照合・関係の判定・保存・退役は製品の実装をそのまま通す。
    """
    from lss.acquisition import service
    from lss.bgm import reference_acquisition, reference_store
    from tests.test_acquisition import FakeBackend

    recording = _recording()
    album = {}
    for number, seed in enumerate((1, 31, 32, 33)):
        path = tmp_path / "remote" / f"t{number}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), _music(seed, 40.0), LOW, subtype="PCM_16")
        _RELEASED_FIXTURES.add(application.file_sha256(path))
        album[f"t{number}"] = path
    backend = FakeBackend(album)
    search_calls = []

    class Search:
        def __init__(self, backend=None):
            pass

        def albums(self, query):
            search_calls.append(query)
            return [{"url": "https://music.example.test/browse/OST"}]

        def tracks(self, url):
            return {"url": url, "title": "テスト番組 オリジナル・サウンドトラック", "id": "OST",
                    "tracks": [{"url": f"https://example.test/t{n}", "id": f"t{n}"} for n in range(4)]}

    monkeypatch.setenv(reference_acquisition.DISCOVERY_ENV, "1")
    monkeypatch.setattr(reference_acquisition, "SearchBackend", Search)
    monkeypatch.setattr(service, "default_backend", lambda: backend)
    monkeypatch.setattr(service, "_ensure_provider", lambda **_: None)
    monkeypatch.setattr(service.random, "uniform", lambda low, high: 0.0)   # 取得の間隔を待たない
    source = _episode(tmp_path / "『テスト番組』第1話「始まり」.wav", recording[int(10 * LOW):int(30 * LOW)])
    gpu.tagger.singing = 0.0
    assert not any(_library(tmp_path).rglob("*.flac")) if _library(tmp_path).exists() else True

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.HELD, outcome.message
    semantics = _semantics(outcome)
    assert semantics["counts"]["KNOWN_RELEASED_MUSIC"] == 1
    [occurrence] = semantics["reference_relations"]["occurrences"]
    state = tmp_path / "state"
    record = reference_store.load(occurrence["reference_sha256"], state)
    assert record["origin"] == reference_store.ORIGIN_ACQUIRED
    assert record["media"]["class"] == reference_store.CLASS_REGENERABLE
    assert record["discovery"]["album_title"] == "テスト番組 オリジナル・サウンドトラック"
    assert record["verification"]["matched_sources"] == [application.file_sha256(source)]
    run = _run_record(outcome)
    assert run["references"]["sources"]["acquired_for_work"] == 4
    assert run["references"]["sources"]["preparation"]["state"] in ("verified", "verified_partial")
    assert run["references"]["sources"]["library"]["count"] == 0   # 利用者は何も置いていない
    assert search_calls   # 作品名（ファイル名から）で探した


def test_an_episode_edit_is_rebuilt_from_the_released_recording_and_regenerated_without_inference(tmp_path, gpu):
    """番組の版（同じ録音の切り貼り）を、確かめた領域だけ既発売録音で組み立てて公開し、来歴と可逆性を残す。"""
    recording = _recording()
    edit = np.concatenate([recording[:int(8 * LOW)], _EPISODE_ONLY, recording[int(24 * LOW):int(36 * LOW)]])
    source = _episode(tmp_path / "episode.wav", edit)
    gpu.tagger.singing = 0.0
    gpu.tagger.speech = 0.9   # 台詞がある場面として処理済みの表現を選ばせる
    reference = _place(tmp_path, recording)

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    [published] = outcome.outputs
    provenance = json.loads(FLAC(published)["LSS_BGM_PROVENANCE"][0])
    assert provenance["reference_relation"]["state"] == reference_relations.EDIT
    assert provenance["representation"] == "reference_donor_reconstruction"
    assert provenance["base_representation"] == "sfc_music_refined"
    reconstruction = provenance["reconstruction"]
    accepted = [row for row in reconstruction["donor_regions"] if row["accepted"]]
    assert accepted and {row["reference_sha256"] for row in accepted} == {application.file_sha256(reference)}
    assert {round(row["offset_seconds"]) for row in accepted} == {-5, 7}   # 4 s of episode-only material between
    # 二つの部分は番組だけの素材（13-17秒）で隔てられる：その素材は抽出の音のまま、参照で置き換えない。
    assert all(row["end_seconds"] <= 13.6 or row["start_seconds"] >= 16.4 for row in accepted), accepted
    assert reconstruction["statement"].startswith("参照を使った再構成")
    # 作り直し：保留候補と記録した計画・参照音源から、推論なしで同じ音になる。
    run = _run_record(outcome)
    [item] = run["published"]
    assert item["representation"] == "reference_donor_reconstruction" and item["reconstruction"]
    audio = application._published_audio(run, item)
    data, _ = sf.read(str(published), dtype="float64", always_2d=True)
    assert audio.shape == data.shape
    assert np.max(np.abs(audio - data)) < 1e-4
    # 元の抽出（置き換える前の表現）も保留候補から作れる（可逆）。
    base = application._published_audio(run, {**item, "reconstruction": None})
    assert base.shape == data.shape and not np.allclose(base, data)


def test_a_planned_reconstruction_that_replaces_nothing_is_not_called_a_reconstruction(tmp_path, gpu, monkeypatch):
    """描いた後の確認で置き換える領域が残らない時は、再構成と名乗らず元の表現のまま公開する（表現の正直さ）。"""
    from lss.bgm import reference_reconstruction

    recording = _recording()
    edit = np.concatenate([recording[:int(8 * LOW)], _EPISODE_ONLY, recording[int(24 * LOW):int(36 * LOW)]])
    source = _episode(tmp_path / "episode.wav", edit)
    gpu.tagger.singing = 0.0
    gpu.tagger.speech = 0.9
    _place(tmp_path, recording)
    strict = reference_reconstruction.ReconstructionRecipe(min_fit=0.999999)
    monkeypatch.setattr(reference_reconstruction, "default_recipe", lambda: strict)

    outcome = _run(tmp_path, source)

    [published] = outcome.outputs
    provenance = json.loads(FLAC(published)["LSS_BGM_PROVENANCE"][0])
    assert provenance["representation"] == "sfc_music_refined"
    assert "reconstruction" not in provenance and "base_representation" not in provenance
    [item] = _run_record(outcome)["published"]
    assert item["representation"] == "sfc_music_refined" and not item.get("reconstruction")
    data, _ = sf.read(str(published), dtype="float64", always_2d=True)
    assert np.max(np.abs(application._published_audio(_run_record(outcome), item) - data)) < 1e-4
    segment = next(s for s in _semantics(outcome)["segments"] if s.get("published"))
    assert segment["route"]["reconstruction_not_applied"]


def test_a_short_tail_from_the_same_recording_is_still_that_recording(tmp_path, gpu):
    """番組の版が、既発売の録音の一続きの範囲＋同じ録音の別の位置の短い尻尾（3秒）。

    尻尾も既発売の録音の一部：繋ぎ直しただけの使用は新しい価値を作らない（2026-10-19、
    `BGM_AND_AVAILABILITY.md`）。v3が消したのは参照があるだけの区間で、ここは全てが録音の部分。
    """
    recording = _recording()
    edit = np.concatenate([recording[:int(24 * LOW)], recording[int(30 * LOW):int(33 * LOW)]])
    source = _episode(tmp_path / "episode.wav", edit, seconds=40)
    gpu.tagger.singing = 0.0
    gpu.tagger.speech = 0.9   # 台詞がある場面として処理済みの表現を選ばせる（再構成の候補になる）
    _place(tmp_path, recording)

    outcome = _run(tmp_path, source)

    assert outcome.outputs == (), outcome.message
    semantics = _semantics(outcome)
    assert semantics["counts"]["KNOWN_RELEASED_MUSIC"] == 1
    [use] = semantics["reference_relations"]["occurrences"]
    assert use["relation"] in (reference_relations.EDIT, reference_relations.EXACT)


def test_a_rebuild_that_would_only_copy_the_released_recording_is_not_published_again(tmp_path, gpu):
    """番組の版が既発売の録音の一続きの範囲だけ（関係判定の1セルに満たない揺れ）：写しを新しく作らない。"""
    recording = _recording()
    source = _episode(tmp_path / "episode.wav", recording[:int(24 * LOW)], seconds=40)
    gpu.tagger.singing = 0.0
    gpu.tagger.speech = 0.9
    _place(tmp_path, recording)

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.HELD, outcome.message
    assert not outcome.outputs
    semantics = _semantics(outcome)
    known = [s for s in semantics["segments"] if s["semantic_class"] == cue_semantics.KNOWN_RELEASED]
    assert known and all(s["scope"]["retention"]["state"] == "retained_internal" for s in known)
    assert semantics["reference_relations"]["occurrences"]


def test_an_episode_version_with_a_layer_removed_is_published_as_a_component_variant(tmp_path, gpu):
    """既発売の録音から一つの層を抜いた番組だけの版。録音が鳴っていても既発売にせず、成分の違う版として公開する。"""
    lead = _music(1, 40.0)
    layer = _music(3, 40.0, harmonics=(1.0,), transpose=-12.0)
    source = _episode(tmp_path / "episode.wav", lead[int(10 * LOW):int(30 * LOW)])
    gpu.tagger.singing = 0.0
    _place(tmp_path, (lead + 0.7 * layer).astype(np.float32))

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    semantics = _semantics(outcome)
    assert semantics["counts"].get("KNOWN_RELEASED_MUSIC", 0) == 0
    assert semantics["reference_relations"]["occurrences"] == []
    [published] = outcome.outputs
    provenance = json.loads(FLAC(published)["LSS_BGM_PROVENANCE"][0])
    assert provenance["identity_state"] == "released_recording_component_variant"
    relation = provenance["reference_relation"]
    assert relation["state"] == reference_relations.COMPONENT
    assert relation["components"]["missing"]["energy_share"] > 0.02
    [related] = semantics["reference_relations"]["related_occurrences"]
    assert related["relation"] == reference_relations.COMPONENT


def test_a_related_version_candidate_is_annotated_but_still_published(tmp_path, gpu):
    """波形では同じ録音と言えない別版（和声の対応だけ）は、既発売にしない（公開を変えない）。"""
    cohort = {f"R{n:02d}": _music(100 + n, 45.0) for n in range(22)}
    for name, signal in cohort.items():
        _place(tmp_path, signal, name=f"{name}.flac", title=None)
    variant = _music(105, 45.0, transpose=2.0, harmonics=(1.0, 0.8, 0.6, 0.3))
    source = _episode(tmp_path / "episode.wav", variant[int(2 * LOW):int(34 * LOW)], seconds=45)
    gpu.tagger.singing = 0.0

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    semantics = _semantics(outcome)
    assert semantics["counts"]["KNOWN_RELEASED_MUSIC"] == 0
    states = {(s.get("reference_relation") or {}).get("state") for s in semantics["segments"]}
    assert reference_relations.EXACT not in states and reference_relations.EDIT not in states
    provenance = json.loads(FLAC(outcome.outputs[0])["LSS_BGM_PROVENANCE"][0])
    assert provenance["identity_state"] in ("related_version_candidate", "unresolved_no_registered_reference_match")
    if provenance["identity_state"] == "related_version_candidate":
        assert provenance["reference_relation"]["candidates"]


def test_a_related_version_from_the_relation_layer_never_suppresses(tmp_path, gpu, monkeypatch):
    """関係の層が別版の候補と言った区間は、公開の判定へ何も足さない（配線の確認）。"""
    source = _episode(tmp_path / "episode.wav", _recording(seed=42, seconds=24.0))
    gpu.tagger.singing = 0.0
    _place(tmp_path, _recording())
    _place(tmp_path, _recording(seed=9), name="another.flac")
    monkeypatch.setattr(versions, "MIN_COHORT", 1)
    monkeypatch.setattr(versions, "segment_relations", lambda segment, refs: [
        {"reference_id": next(iter(refs)), "score": 0.9, "z": 99.0}])

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    provenance = json.loads(FLAC(outcome.outputs[0])["LSS_BGM_PROVENANCE"][0])
    assert provenance["identity_state"] == "related_version_candidate"
    assert provenance["reference_relation"]["state"] == reference_relations.RELATED
    assert _semantics(outcome)["counts"]["KNOWN_RELEASED_MUSIC"] == 0


def test_unknown_music_is_published_even_when_references_are_available(tmp_path, gpu):
    source = _episode(tmp_path / "episode.wav", _recording(seed=42, seconds=20.0))
    gpu.tagger.singing = 0.0
    _place(tmp_path, _recording())

    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    provenance = json.loads(FLAC(outcome.outputs[0])["LSS_BGM_PROVENANCE"][0])
    assert provenance["identity_state"] == "unresolved_no_registered_reference_match"
    assert provenance["reference_relation"]["state"] == reference_relations.UNKNOWN
    assert _semantics(outcome)["counts"]["KNOWN_RELEASED_MUSIC"] == 0


def test_a_failed_reference_check_keeps_every_segment_and_says_it_was_not_checked(tmp_path, gpu, monkeypatch):
    source, recording = _exact_case(tmp_path)
    gpu.tagger.singing = 0.0
    _place(tmp_path, recording)

    def broken(*args, **kwargs):
        raise identity.DecodeError("参照を復号できない")

    monkeypatch.setattr(reference_relations.Gate, "prepare", broken)
    outcome = _run(tmp_path, source)

    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    provenance = json.loads(FLAC(outcome.outputs[0])["LSS_BGM_PROVENANCE"][0])
    assert provenance["identity_state"] == "unresolved_reference_not_checked"
    record = _semantics(outcome)["reference_relations"]
    assert record["state"] == "not_checked" and record["error"] == "DecodeError" and record["occurrences"] == []


def test_reference_names_do_not_change_what_is_extracted(tmp_path, gpu, monkeypatch):
    """参照のタグ・ファイル名・フォルダを変えても、同じ音声からは同じ判定・同じ公開になる（名前と同一性を結合しない）。"""
    recording = _recording()
    edit = np.concatenate([recording[:int(8 * LOW)], recording[int(24 * LOW):int(36 * LOW)]])
    results = []
    for name, title, file_name in (("named", "既発売の曲", "01 既発売の曲.flac"), ("plain", None, "track.flac")):
        folder = tmp_path / name
        folder.mkdir()
        monkeypatch.setenv(references.LIBRARY_ENV, str(folder / "References"))
        source = _episode(folder / "episode.wav", edit)
        gpu.tagger.singing = 0.0
        _place(folder, recording, title=title, name=file_name, folder=name)
        outcome = _run(folder, source)
        run = _run_record(outcome)
        results.append(([(s["start_seconds"], s["end_seconds"], s["semantic_class"],
                          s["reference_relation"]["state"]) for s in _semantics(outcome)["segments"]],
                        [(item["key"], Path(item["path"]).name, item["audio"]["pcm_sha256"])
                         for item in run["published"]],
                        outcome.record_path.parent.parent.name))
    assert results[0][:2] == results[1][:2]


def test_the_library_index_survives_restart_and_is_rebuilt_when_lost(tmp_path, gpu, monkeypatch):
    """置き場所の内容SHA-256の索引は状態領域に残り、次の起動では読み直さない。消えても同じ結果に作り直す。"""
    _, recording = _exact_case(tmp_path)
    _place(tmp_path, recording)
    _place(tmp_path, recording, name="copy of released.flac", folder="other")   # 同じ内容の複製
    state = tmp_path / "state"
    first = application._registered_references(state)
    index = state / references.LIBRARY_INDEX
    assert index.is_file() and len(first[0]) == 1
    assert first[3]["library"]["duplicates"] == 1

    reads = []
    original = references.session.input_sha256_of
    monkeypatch.setattr(references.session, "input_sha256_of", lambda path, *a, **k: reads.append(path)
                        or original(path, *a, **k))
    again = application._registered_references(state)       # 次の起動（同じ状態領域）
    assert reads == [] and again[1] == first[1] and again[0] == first[0]
    index.unlink()
    rebuilt = application._registered_references(state)
    assert len(reads) == 2 and rebuilt[1] == first[1] and index.is_file()


def test_unchanged_references_reuse_the_run_and_give_stable_decisions(tmp_path, gpu):
    source, recording = _exact_case(tmp_path)
    gpu.tagger.singing = 0.0
    _place(tmp_path, recording)

    first = _run(tmp_path, source)
    calls = len(gpu.calls)
    second = _run(tmp_path, source)

    assert second.reused is True and len(gpu.calls) == calls
    assert second.record_path == first.record_path
    assert _semantics(second)["reference_relations"] == _semantics(first)["reference_relations"]
    found = application.existing_bgm_candidates(source, application.file_sha256(source), state_root=tmp_path / "state")
    assert found is None   # 保留だけのrunは既存の受理済み結果ではない（従来どおり、次の実行で読み戻す）


def test_a_reference_change_reconsiders_the_episode_and_reuses_unchanged_reference_work(tmp_path, gpu):
    """参照の集まりが変われば別のrun（その話を考え直す。r4と同じ粒度）。変わらない参照の指紋は作り直さない。
    元の集まりへ戻せば、前のrunを推論なしで再利用する。"""
    source, recording = _exact_case(tmp_path)
    gpu.tagger.singing = 0.0
    _place(tmp_path, recording)
    first = _run(tmp_path, source)
    fingerprints = sorted((tmp_path / "state" / "bgm-reference-fingerprints").glob("*.npz"))
    stamps = {path: path.stat().st_mtime_ns for path in fingerprints}
    calls = len(gpu.calls)

    extra = _place(tmp_path, _recording(seed=9), name="another.flac")
    second = _run(tmp_path, source)

    assert second.record_path != first.record_path and second.reused is False
    assert len(gpu.calls) == calls + 1                      # 話の解析をやり直す（粗い再実行。記録して隠さない）
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in stamps.items())
    assert len(list((tmp_path / "state" / "bgm-reference-fingerprints").glob("*.npz"))) == len(fingerprints) + 1
    assert _semantics(second)["counts"]["KNOWN_RELEASED_MUSIC"] == 1

    extra.unlink()
    third = _run(tmp_path, source)
    assert third.reused is True and third.record_path == first.record_path and len(gpu.calls) == calls + 1


def test_an_old_public_cue_is_kept_when_references_later_show_it_is_a_released_recording(tmp_path, gpu):
    """前から公開しているcueは、参照録音が後から増えて既発売と分かっても外さない（妥当性は変えず、文脈だけ残す）。"""
    source, recording = _exact_case(tmp_path)
    gpu.tagger.singing = 0.0
    before = _run(tmp_path, source)
    assert before.disposition is application.ResultDisposition.PUBLISHED, before.message
    [public] = before.outputs
    pcm = json.loads(FLAC(public)["LSS_BGM_PROVENANCE"][0])

    _place(tmp_path, recording)
    after = _run(tmp_path, source)

    assert after.disposition is application.ResultDisposition.PUBLISHED, after.message
    assert after.outputs == (public,) and public.is_file()
    assert json.loads(FLAC(public)["LSS_BGM_PROVENANCE"][0]) == pcm
    assert _semantics(after)["counts"]["KNOWN_RELEASED_MUSIC"] == 1
    record = current_best.read(tmp_path / "state", application.file_sha256(source))
    [old] = record["current"]["decisions"]["old"]
    assert old["decision"] == "keep" and old["validity"] == "VALID"
    [known] = old["context"]["known_released_recording"]
    assert known["display_title"] == "既発売の曲 [TV Ver.]"
    assert record["current"]["adoption_policy"] == application._BGM_ADOPTION_POLICY
    assert len(record["current"]["evaluated_runs"]) == 2
    # 公開の集まりは変わっていない（外したもの・置いたものが無いので、履歴の切替も作らない。r4と同じ）。
    assert record["history"] == [] and record["current"]["transition"] == "recipe_supersession"
    assert [item["public_path"] for item in record["current"]["files"]] == [str(public)]


def test_the_deployed_worker_creates_the_reference_library_folder_only_when_frozen(tmp_path, monkeypatch):
    from lss import paths
    from lss.core import worker

    target = tmp_path / "app" / "References"
    monkeypatch.setenv(references.LIBRARY_ENV, str(target))
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    worker._ensure_reference_library()
    assert not target.exists()
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    worker._ensure_reference_library()
    assert target.is_dir() and list(target.iterdir()) == []


def test_the_default_library_is_inside_the_app_folder(monkeypatch):
    from lss.paths import app_root

    monkeypatch.delenv(references.LIBRARY_ENV, raising=False)
    assert references.library_root() == app_root() / "References"


def test_the_reference_policy_sources_are_bundled_and_identified():
    """参照の方針の識別子は、凍結版でも同じ実装の内容hashから作る（実体を同梱し、構造の確認にも入れる）。"""
    root = Path(__file__).resolve().parents[1]
    spec = (root / "LSS.spec").read_text(encoding="utf-8")
    build = (root / "build.ps1").read_text(encoding="utf-8-sig")
    policy = application._bgm_reference_policy()
    assert set(policy["implementation"]) == set(application._BGM_REFERENCE_IMPLEMENTATION)
    for name in application._BGM_REFERENCE_IMPLEMENTATION:
        assert f'("lss/{name}", "lss/bgm")' in spec, name
        assert "_internal\lss\\" + name.replace("/", "\\") in build, name
    assert policy["policy_id"].startswith("bgm-reference-policy-v2-")
    assert policy["reference_relations_id"] == reference_relations.default_recipe().recipe_id
