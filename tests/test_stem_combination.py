"""モデルステム経路の組み合わせ: 規則、正確な和、Recipe identity、製品要求の経路選択。"""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss import application
from lss.core import jobs, models_store, recipe
from lss.harness import fetch, tsv_io
from lss.separation import stem_combination as sc
from lss.ui import contract

FOUR = ["drums", "bass", "other", "vocals"]
DUALITY = ["Vocals", "Instrumental"]


# --- 規則 -------------------------------------------------------------------

def test_vocals_and_drums_sum_two_disjoint_outputs_of_one_pass():
    plan = sc.plan_combination(["Vocals", "drums"], FOUR)
    assert plan.components == ("drums", "vocals")  # modelの出力順。要求の順序に依らない
    assert plan.route == "sum_of_stems"
    assert plan.label == "drums + vocals"
    assert plan.gains == (1.0, 1.0)
    assert plan.subsumed == ()
    assert sc.plan_combination(["drums", "vocals"], FOUR) == plan


def test_contained_component_is_counted_once_and_recorded_as_subsumed():
    # 伴奏は既にドラムを含む。ドラムを二重に足さない。
    plan = sc.plan_combination(["instrumental", "drums"], FOUR)
    assert plan.components == ("drums", "bass", "other")
    assert plan.components.count("drums") == 1
    assert plan.subsumed == ("drums",)
    assert plan.route == "sum_of_stems"


def test_direct_complementary_stem_is_preferred_over_sum_of_residuals():
    plan = sc.plan_combination(["instrumental", "drums"], DUALITY)
    assert plan.components == ("Instrumental",)
    assert plan.route == "direct_stem"
    assert plan.subsumed == ("drums",)


@pytest.mark.parametrize("components,stems,needle", [
    (["vocals", "instrumental"], DUALITY, "原本の全部"),
    (["drums", "bass", "other", "vocals"], FOUR, "原本の全部"),
    (["chorus", "instrumental"], DUALITY, "一部だけ"),     # 歌を合唱と主旋律に分けるpassではない
    (["piano", "vocals"], FOUR, "一部だけ"),               # pianoは4分割のotherの一部
    (["kazoo", "vocals"], FOUR, "分からない部品"),
])
def test_combinations_the_pass_cannot_make_are_refused(components, stems, needle):
    with pytest.raises(sc.StemCombinationError, match=needle):
        sc.plan_combination(components, stems)


@pytest.mark.parametrize("components", [["vocals"], ["vocals", "VOCALS"], [], "vocals+drums", ["a+b", "c"], [""]])
def test_a_combination_needs_two_distinct_well_formed_components(components):
    with pytest.raises(sc.StemCombinationError):
        sc.normalized_request(components)


def test_target_key_round_trips_and_single_stems_are_not_combinations():
    key = sc.target_key(["Vocals", " drums "])
    assert key == "drums+vocals"
    assert sc.parse_target(key) == ("drums", "vocals")
    assert sc.parse_target("Instrumental") is None


def test_opaque_model_outputs_are_only_combined_by_their_exact_names():
    stems = ["Vocals", "Kick", "Snare", "Rest"]
    plan = sc.plan_combination(["kick", "snare"], stems)
    assert plan.components == ("Kick", "Snare")
    # 不透明な出力を含むpassでは「残り」の中身を推定しない。
    with pytest.raises(sc.StemCombinationError):
        sc.plan_combination(["drums", "vocals"], stems)


# --- 正確な和 ---------------------------------------------------------------

def _stems(directory: Path, frames=4096, sr=44100, channels=2, seed=7, scale=0.9):
    rng = np.random.default_rng(seed)
    arrays = {}
    for name in FOUR:
        data = (scale * rng.uniform(-1.0, 1.0, (frames, channels))).astype(np.float32)
        sf.write(directory / f"{name}.wav", data, sr, subtype="FLOAT")
        arrays[name] = data
    return arrays


def test_combined_signal_is_the_exact_declared_sum_and_is_not_clipped(tmp_path):
    arrays = _stems(tmp_path)
    plan = sc.plan_combination(["vocals", "drums"], FOUR)
    receipt = sc.combine_files(tmp_path, plan, tmp_path / f"{plan.label}.wav")

    combined, rate = sf.read(tmp_path / "drums + vocals.wav", dtype="float32", always_2d=True)
    expected = (arrays["drums"].astype(np.float64) + arrays["vocals"].astype(np.float64)).astype(np.float32)
    assert rate == 44100
    assert np.array_equal(combined, expected)  # sample-exact, not approximately
    assert float(np.max(np.abs(combined))) > 1.0  # 足した結果が1.0を超えても切らない
    assert receipt["sum_peak"] == float(np.max(np.abs(expected)))
    assert receipt["components"] == ["drums", "vocals"] and receipt["gains"] == [1.0, 1.0]
    assert set(receipt["component_sha256"]) == {"drums", "vocals"}
    assert all(len(value) == 64 for value in receipt["component_sha256"].values())


@pytest.mark.parametrize("mutate", ["frames", "samplerate", "channels"])
def test_misaligned_components_are_refused_and_nothing_is_written(tmp_path, mutate):
    arrays = _stems(tmp_path)
    data = arrays["drums"]
    if mutate == "frames":
        sf.write(tmp_path / "drums.wav", data[:-1], 44100, subtype="FLOAT")
    elif mutate == "samplerate":
        sf.write(tmp_path / "drums.wav", data, 48000, subtype="FLOAT")
    else:
        sf.write(tmp_path / "drums.wav", data[:, :1], 44100, subtype="FLOAT")
    plan = sc.plan_combination(["vocals", "drums"], FOUR)
    destination = tmp_path / f"{plan.label}.wav"
    with pytest.raises(sc.StemCombinationIncompleteError, match="揃っていません"):
        sc.combine_files(tmp_path, plan, destination)
    assert not destination.exists()


def test_missing_or_non_finite_component_leaves_no_partial_combination(tmp_path):
    arrays = _stems(tmp_path)
    plan = sc.plan_combination(["vocals", "drums"], FOUR)
    destination = tmp_path / f"{plan.label}.wav"
    (tmp_path / "drums.wav").unlink()
    with pytest.raises(sc.StemCombinationIncompleteError):
        sc.combine_files(tmp_path, plan, destination)
    assert not destination.exists()

    broken = arrays["drums"].copy()
    broken[-1, 0] = np.nan
    sf.write(tmp_path / "drums.wav", broken, 44100, subtype="FLOAT")
    with pytest.raises(sc.StemCombinationIncompleteError):
        sc.combine_files(tmp_path, plan, destination)
    assert not destination.exists()


# --- Recipe identity --------------------------------------------------------

def _separation(**overrides):
    values = {
        "provider": "msst", "arch": "mel_band_roformer", "model_id": "model_x",
        "model_sha256": "a" * 64, "config_sha256": "b" * 64, "runtime_id": "c" * 64,
        "expected_stems": ["Vocals", "Instrumental"], "output_samplerate": 44100,
        "input_preparation": {"decoder": "test"}, "serialization": {"subtype": "PCM_24"},
    }
    values.update(overrides)
    return values


def test_existing_single_stem_recipe_ids_are_unchanged():
    # 変更前（0d065aab）の実装で計算した値。単一ステムのRecipe identityを動かさない。
    assert recipe.separation_recipe_revision(**_separation(), published_stems=["Instrumental"])["recipe_id"] == (
        "separation-v1-3fa0d4cc3b8433830dc6de55f3bab46990b4f167c13f7f0b3f1d7498b5613d17")
    assert recipe.separation_recipe_revision(
        **_separation(), published_stems=["Vocals"], post_processing={"x": 1})["recipe_id"] == (
        "separation-v1-300586ba27a5944a17cea8e4021a216c717c6c11ee78e9402a2ca6ff93ceb0d4")


def test_combination_recipe_is_stable_and_distinct_from_each_component():
    base = _separation(expected_stems=FOUR)
    first = recipe.stem_combination_recipe_revision(
        **base, combination=sc.plan_combination(["vocals", "drums"], FOUR).recipe_stage())
    again = recipe.stem_combination_recipe_revision(
        **base, combination=sc.plan_combination(["DRUMS", "Vocals"], FOUR).recipe_stage())
    assert first == again
    assert first["recipe_id"].startswith("stem-combination-v1-")
    assert recipe.validate_recipe_revision(first) == first
    singles = {recipe.separation_recipe_revision(**base, published_stems=[name])["recipe_id"]
               for name in ("drums", "vocals")}
    assert first["recipe_id"] not in singles
    other = recipe.stem_combination_recipe_revision(
        **base, combination=sc.plan_combination(["bass", "drums"], FOUR).recipe_stage())
    assert other["recipe_id"] != first["recipe_id"]
    assert first["publication"] == {"stems": ["drums + vocals"]}
    assert first["stage_graph"][1]["components"] == ["drums", "vocals"]


def test_tampered_combination_recipe_is_rejected():
    revision = recipe.stem_combination_recipe_revision(
        **_separation(expected_stems=FOUR),
        combination=sc.plan_combination(["vocals", "drums"], FOUR).recipe_stage())
    tampered = json.loads(json.dumps(revision))
    tampered["stage_graph"][1]["gains"] = [1.0, 2.0]
    with pytest.raises(ValueError):
        recipe.validate_recipe_revision(tampered)
    with pytest.raises(ValueError):
        recipe.stem_combination_recipe_revision(
            **_separation(expected_stems=FOUR),
            combination={**revision["stage_graph"][1], "components": ["drums", "piano"]})


# --- 製品要求 ---------------------------------------------------------------

def _catalog(tmp_path, monkeypatch, models):
    ledger = tmp_path / "models.tsv"
    rows = [{"model_id": model_id, "状態": state, "config_url": f"https://example.test/{model_id}.yaml"}
            for model_id, state, _ in models]
    tsv_io.save_tsv(ledger, list(rows[0]), rows)
    monkeypatch.setattr(models_store, "MODELS_TSV", ledger)
    monkeypatch.setenv("LSS_MODEL_DIR", str(tmp_path / "models"))
    for model_id, _, stems in models:
        config = fetch.model_asset_path(model_id, f"https://example.test/{model_id}.yaml")
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"training:\n  instruments: [{', '.join(stems)}]\n", encoding="utf-8")


def _submit(tmp_path, stems):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    calls = []
    queue = jobs.SeparationQueue(
        lambda job: calls.append(job) or {"status": "ok", "outputs": [str(tmp_path / "r.flac")],
                                          "recipe_id": "stem-combination-v1-x"},
        state_path=tmp_path / "queue.json",
    )
    request = application.ProcessingRequest(
        sources=(source,), intent=application.UserIntent.EXTRACT_STEM_COMBINATION, stems=tuple(stems))
    submission = application.submit(request, queue)
    queue.run()
    return submission, calls


def test_combination_request_chooses_the_model_and_dispatches_one_pass(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch, [("duality", "既定", DUALITY), ("four", "候補", FOUR)])
    submission, calls = _submit(tmp_path, ["vocals", "drums"])
    assert submission.outcome is None and len(submission.queue_item_ids) == 1
    assert [(job["model_id"], job["stem_target"]) for job in calls] == [("four", "drums+vocals")]
    assert submission.stem_combination["route"] == "sum_of_stems"
    assert submission.stem_combination["components"] == ["drums", "vocals"]
    assert submission.stem_combination["model_id"] == "four"


def test_combination_equal_to_a_direct_stem_uses_the_single_stem_route(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch, [("four", "既定", FOUR), ("duality", "候補", DUALITY)])
    submission, calls = _submit(tmp_path, ["instrumental", "drums"])
    # 4分割の3本の和ではなく、相補の直接の出力（伴奏）を選ぶ。
    assert [(job["model_id"], job["stem_target"]) for job in calls] == [("duality", "Instrumental")]
    assert submission.stem_combination["route"] == "direct_stem"
    assert submission.stem_combination["subsumed"] == ["drums"]


def test_combination_no_model_can_make_is_unsupported_without_queueing(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch, [("duality", "既定", DUALITY)])
    submission, calls = _submit(tmp_path, ["vocals", "drums"])
    assert calls == [] and submission.queue_item_ids == ()
    assert submission.outcome.disposition == application.ResultDisposition.UNSUPPORTED
    assert submission.outcome.error_code == "stem_combination_unsupported"


def test_combination_parts_belong_only_to_combination_requests(tmp_path):
    with pytest.raises(ValueError):
        application.ProcessingRequest(sources=(tmp_path,), intent=application.UserIntent.EXTRACT_ACCOMPANIMENT,
                                      stems=("vocals", "drums"))
    with pytest.raises(ValueError):
        application.ProcessingRequest(sources=(tmp_path,), intent=application.UserIntent.EXTRACT_STEM_COMBINATION,
                                      stems=("vocals",))


def test_combination_failures_have_distinct_product_meaning():
    unsupported = application.outcome_from_worker_result(
        {"status": "error", "error_code": "stem_combination_unsupported", "message": "x"})
    incomplete = application.outcome_from_worker_result(
        {"status": "error", "error_code": "stem_combination_incomplete", "message": "x"})
    assert unsupported.disposition == application.ResultDisposition.UNSUPPORTED
    assert incomplete.disposition == application.ResultDisposition.HELD
    assert incomplete.outputs == () and incomplete.held_artifacts == ()
    assert contract.classify_failure("stem_combination_unsupported").recognized
    assert contract.classify_failure("stem_combination_incomplete").recognized
    assert contract.classify_failure("stem_combination_incomplete").scope == contract.FailureScope.SOURCE
