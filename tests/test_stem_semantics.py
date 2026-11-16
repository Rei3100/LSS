"""modelごとの出力の意味と、別々の分離passの組み合わせ（模擬分離器・合成信号）。

宣言は読み込み時に検証され、宣言のあるmodelは名前の推測でなく宣言で読む。宣言の無いmodelと
既存の組み合わせ・単一ステムのRecipe IDは変わらない。別passの組み合わせは宣言済みの意味のもとで
符号付き単位利得の和が要求と一致し、揃い（同じ復号済み入力・同じレート/長さ）が証明できるときだけ作る。
"""
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from lss import application
from lss.core import catalog, flac_frames, gpu_lease, jobs, models_store, recipe, worker
from lss.core.runtime import RuntimeRevision
from lss.harness import fetch, infer, tsv_io
from lss.output import writer
from lss.paths import resource_root
from lss.separation import stem_combination as sc
from lss.separation import stem_semantics

FOUR = ["drums", "bass", "other", "vocals"]
DUALITY = ["Vocals", "Instrumental"]
VOX = frozenset({sc.LEAD_VOCALS, sc.BACKING_VOCALS})
ALL = frozenset(sc.ATOMS)
ALLVOX = {"Vocals": VOX, "Instrumental": ALL - VOX}
KARAOKE = {"Vocals": frozenset({sc.LEAD_VOCALS}), "Instrumental": ALL - {sc.LEAD_VOCALS}}
FOUR_DECLARED = {"drums": frozenset({sc.DRUMS}), "bass": frozenset({sc.BASS}),
                 "other": frozenset({sc.GUITAR, sc.PIANO, sc.OTHER}), "vocals": VOX}


def _table_rows(declared: dict) -> list[dict]:
    return [{"model_id": model_id, "output": output, "content": ",".join(sorted(atoms)),
             "basis": "test", "note": ""}
            for model_id, outputs in declared.items() for output, atoms in outputs.items()]


def _write_table(path: Path, declared: dict) -> Path:
    tsv_io.save_tsv(path, list(stem_semantics.COLUMNS), _table_rows(declared))
    return path


# --- 宣言表の検証 -------------------------------------------------------------

def test_bundled_table_is_valid_and_names_ledger_models():
    table = stem_semantics.load_semantics()
    _, rows = models_store.load_model_ledger(resource_root() / "ledger" / "models.tsv")
    ledger = {row["model_id"]: row for row in rows}
    assert set(table) <= set(ledger)
    assert dict(table["becruily_karaoke_bs_roformer"]) == KARAOKE
    # 採否は台帳が決める。下書きの宣言は通常利用のmodelを増やさない。
    for model_id in ("becruily_karaoke_bs_roformer", "scnet_xl_ihf_4stem", "bs_roformer_sw_6stem"):
        assert ledger[model_id]["状態"] == "保留"


def test_declared_default_model_reads_exactly_as_the_name_rule():
    # 既定modelの宣言は名前の規則と同じ中身。1passの計画（とRecipe）は変わらない。
    declared = stem_semantics.load_semantics()["becruily_deux_inst"]
    assert sc.pass_content(DUALITY, declared) == sc.pass_content(DUALITY)
    for request in (["instrumental", "drums"], ["vocals", "drums"], ["chorus", "instrumental"]):
        try:
            expected = sc.plan_combination(request, DUALITY)
        except sc.StemCombinationError:
            with pytest.raises(sc.StemCombinationError):
                sc.plan_combination(request, DUALITY, declared)
        else:
            assert sc.plan_combination(request, DUALITY, declared) == expected


@pytest.mark.parametrize("rows,needle", [
    ([{"model_id": "m", "output": "Vocals", "content": "lead_vocals,kazoo", "basis": "x", "note": ""}], "未知の単位"),
    ([{"model_id": "m", "output": "Vocals", "content": "", "basis": "x", "note": ""}], "空"),
    ([{"model_id": "m", "output": "Vocals", "content": "lead_vocals,lead_vocals", "basis": "x", "note": ""}], "重複"),
    ([{"model_id": "m", "output": "Vocals", "content": "lead_vocals", "basis": "", "note": ""}], "根拠"),
    ([{"model_id": "m", "output": "Vocals", "content": "lead_vocals", "basis": "x", "note": ""},
      {"model_id": "m", "output": "vocals", "content": "drums", "basis": "x", "note": ""}], "出力が重複"),
    ([{"model_id": "m", "output": "Vocals", "content": "lead_vocals,backing_vocals", "basis": "x", "note": ""},
      {"model_id": "m", "output": "Chorus", "content": "backing_vocals", "basis": "x", "note": ""}], "重なって"),
])
def test_invalid_declarations_are_refused_at_load(tmp_path, rows, needle):
    path = tmp_path / "semantics.tsv"
    tsv_io.save_tsv(path, list(stem_semantics.COLUMNS), rows)
    with pytest.raises(stem_semantics.InvalidStemSemanticsError, match=needle):
        stem_semantics.load_semantics(path)


def test_missing_or_misshapen_table_is_an_error_not_an_empty_declaration(tmp_path):
    with pytest.raises(stem_semantics.InvalidStemSemanticsError):
        stem_semantics.load_semantics(tmp_path / "absent.tsv")
    path = tmp_path / "semantics.tsv"
    tsv_io.save_tsv(path, ["model_id", "output", "content"], [])
    with pytest.raises(stem_semantics.InvalidStemSemanticsError, match="列"):
        stem_semantics.load_semantics(path)


# --- 宣言で読む1passの計画 ------------------------------------------------------

def test_karaoke_declaration_replaces_the_name_heuristic():
    # 名前で読むと Vocals は歌全体。宣言ではリードだけで、伴奏側が合唱と楽器を含む。
    plan = sc.plan_combination(["chorus", "instrumental"], DUALITY, KARAOKE)
    assert plan.components == ("Instrumental",) and plan.route == "direct_stem"
    with pytest.raises(sc.StemCombinationError, match="一部だけ"):
        sc.plan_combination(["chorus", "instrumental"], DUALITY)
    # 要求の「vocals」は歌全体。カラオケの出力名 Vocals（リード）とは一致させない。
    with pytest.raises(sc.StemCombinationError, match="一部だけ"):
        sc.plan_combination(["vocals", "drums"], DUALITY, KARAOKE)
    # 歌全体のmodelではリードだけを分けられない（宣言でも名前でも同じ）。
    with pytest.raises(sc.StemCombinationError, match="一部だけ"):
        sc.plan_combination(["lead", "instrumental"], DUALITY, ALLVOX)


def test_declaration_that_does_not_match_the_model_outputs_is_not_proven():
    with pytest.raises(sc.StemCombinationError, match="一致しません"):
        sc.plan_combination(["chorus", "instrumental"], ["Vocals", "Karaoke"], KARAOKE)


def test_undeclared_models_plan_exactly_as_before():
    for request, stems in ((["vocals", "drums"], FOUR), (["instrumental", "drums"], FOUR),
                           (["instrumental", "drums"], DUALITY), (["kick", "snare"], ["Vocals", "Kick", "Snare", "Rest"])):
        assert sc.plan_combination(request, stems, None) == sc.plan_combination(request, stems)


def _base(**overrides):
    values = {
        "provider": "msst", "arch": "mel_band_roformer", "model_id": "model_x",
        "model_sha256": "a" * 64, "config_sha256": "b" * 64, "runtime_id": "c" * 64,
        "expected_stems": FOUR, "output_samplerate": 44100,
        "input_preparation": {"decoder": "test"}, "serialization": {"subtype": "PCM_24"},
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize("request_,semantics,golden", [
    (["vocals", "drums"], None, "96b5d71098e4665a69d9230eec91f870a62c96b99bd433747f7c43a73cb0ae62"),
    (["instrumental", "drums"], None, "f289df209ad5f591870c29c55487064bc71669a3509c0044414043021dd06f2e"),
    (["bass", "other"], None, "ffefbca9479b6d40954e316299760943486e4fa458659398a4754cd41939b088"),
    # 名前の規則と同じ中身の宣言は、同じRecipeになる。
    (["vocals", "drums"], FOUR_DECLARED, "96b5d71098e4665a69d9230eec91f870a62c96b99bd433747f7c43a73cb0ae62"),
])
def test_existing_combination_recipe_ids_are_unchanged(request_, semantics, golden):
    # 変更前（base/stem-combine）の実装で計算した値。
    stage = sc.plan_combination(request_, FOUR, semantics).recipe_stage()
    assert recipe.stem_combination_recipe_revision(**_base(), combination=stage)["recipe_id"] == (
        f"stem-combination-v1-{golden}")


# --- 別passの計画 ---------------------------------------------------------------

def _candidates(*names):
    known = {"allvox": (DUALITY, ALLVOX), "karaoke": (DUALITY, KARAOKE), "four": (FOUR, FOUR_DECLARED),
             "four_undeclared": (FOUR, None), "allvox_undeclared": (DUALITY, None)}
    return [sc.PassCandidate(name, tuple(known[name][0]), known[name][1]) for name in names]


def test_direct_output_is_preferred_over_derived_algebra():
    model_id, plan = sc.choose_route(["chorus", "instrumental"], _candidates("allvox", "karaoke"))
    assert (model_id, plan.route, plan.components) == ("karaoke", "direct_stem", ("Instrumental",))
    # 1passの和も別passの代数より先。
    model_id, plan = sc.choose_route(["vocals", "drums"], _candidates("allvox", "karaoke", "four"))
    assert (model_id, plan.route, plan.components) == ("four", "sum_of_stems", ("drums", "vocals"))


def test_cross_pass_algebra_is_exact_under_declared_semantics_and_prefers_fewer_passes():
    model_id, plan = sc.choose_route(["chorus", "drums"], _candidates("allvox", "karaoke", "four"))
    assert plan.route == "cross_pass_algebra" and model_id == "karaoke"
    assert plan.passes == ("karaoke", "four")  # 3pass（allvoxも使う）より少ない2pass
    assert [(t.key, t.gain) for t in plan.terms] == [
        ("karaoke/Vocals", -1.0), ("four/drums", 1.0), ("four/vocals", 1.0)]
    total = {atom: sum(t.gain for t in plan.terms if atom in t.content) for atom in sc.ATOMS}
    assert {atom for atom, value in total.items() if value == 1.0} == {sc.BACKING_VOCALS, sc.DRUMS}
    assert all(value in (0.0, 1.0) for value in total.values())
    assert plan.label == "backing vocals + drums"
    assert plan.content == (sc.BACKING_VOCALS, sc.DRUMS)
    # 要求の言い回しは計画を変えない。
    assert sc.choose_route(["backing vocals", "drum"], _candidates("allvox", "karaoke", "four"))[1].terms == plan.terms


def test_first_pass_can_be_pinned_and_chorus_is_vocals_minus_lead():
    # 分離ジョブは自分のmodelを最初のpassとして同じ計画器で選び直す。
    plan = sc.plan_cross_pass(["chorus", "drums"], _candidates("allvox", "karaoke", "four"), first_model="allvox")
    assert plan.passes == ("allvox", "karaoke", "four")
    assert [(t.key, t.gain) for t in plan.terms] == [
        ("allvox/Vocals", 1.0), ("karaoke/Vocals", -1.0), ("four/drums", 1.0)]
    with pytest.raises(sc.StemCombinationError):
        sc.plan_cross_pass(["chorus", "drums"], _candidates("allvox", "karaoke", "four"), first_model="four")


def test_undeclared_models_never_enter_cross_pass_algebra():
    with pytest.raises(sc.StemCombinationError):
        sc.choose_route(["chorus", "drums"], _candidates("allvox", "karaoke", "four_undeclared"))
    with pytest.raises(sc.StemCombinationError):
        sc.plan_cross_pass(["chorus", "drums"], _candidates("allvox_undeclared", "four"))


def test_whole_mix_and_unknown_words_are_refused_across_passes():
    with pytest.raises(sc.StemCombinationError, match="原本の全部"):
        sc.plan_cross_pass(["vocals", "instrumental"], _candidates("allvox", "karaoke"))
    with pytest.raises(sc.StemCombinationError, match="分からない"):
        sc.plan_cross_pass(["kazoo", "drums"], _candidates("karaoke", "four"))


# --- 符号付きの和 ---------------------------------------------------------------

def _write(path: Path, data: np.ndarray, sr=44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sr, subtype="FLOAT")


def _plan():
    return sc.choose_route(["chorus", "drums"], _candidates("allvox", "karaoke", "four"))[1]


def _pass_stems(tmp_path, frames=4096):
    rng = np.random.default_rng(3)
    arrays = {}
    for model_id, stems in (("karaoke", DUALITY), ("four", FOUR)):
        for stem in stems:
            data = (0.8 * rng.uniform(-1.0, 1.0, (frames, 2))).astype(np.float32)
            _write(tmp_path / model_id / f"{stem}.wav", data)
            arrays[f"{model_id}/{stem}"] = data
    return {"karaoke": tmp_path / "karaoke", "four": tmp_path / "four"}, arrays


def test_signed_sum_is_the_exact_declared_algebra(tmp_path):
    dirs, arrays = _pass_stems(tmp_path)
    plan = _plan()
    destination = tmp_path / "out.wav"
    receipt = sc.combine_terms(dirs, plan, destination)
    combined, _ = sf.read(destination, dtype="float32", always_2d=True)
    expected = (arrays["four/drums"].astype(np.float64) + arrays["four/vocals"].astype(np.float64)
                - arrays["karaoke/Vocals"].astype(np.float64)).astype(np.float32)
    assert np.array_equal(combined, expected)
    assert receipt["route"] == "cross_pass_algebra" and receipt["gains"] == [-1.0, 1.0, 1.0]
    assert receipt["passes"] == ["karaoke", "four"]
    assert receipt["alignment"] == sc.ALIGNMENT_RULE_ID and receipt["rule"] == sc.ALGEBRA_RULE_ID
    assert set(receipt["component_sha256"]) == {"karaoke/Vocals", "four/drums", "four/vocals"}


@pytest.mark.parametrize("mutate", ["frames", "samplerate", "channels", "missing_pass", "missing_stem"])
def test_misaligned_or_missing_passes_are_refused_without_a_partial_file(tmp_path, mutate):
    dirs, arrays = _pass_stems(tmp_path)
    vocals = arrays["karaoke/Vocals"]
    if mutate == "frames":
        _write(tmp_path / "karaoke" / "Vocals.wav", vocals[:-1])
    elif mutate == "samplerate":
        _write(tmp_path / "karaoke" / "Vocals.wav", vocals, sr=48000)
    elif mutate == "channels":
        _write(tmp_path / "karaoke" / "Vocals.wav", vocals[:, :1])
    elif mutate == "missing_pass":
        dirs.pop("four")
    else:
        (tmp_path / "four" / "drums.wav").unlink()
    destination = tmp_path / "out.wav"
    with pytest.raises(sc.StemCombinationIncompleteError):
        sc.combine_terms(dirs, _plan(), destination)
    assert not destination.exists()


# --- Recipe identity --------------------------------------------------------------

def _algebra_revision(plan=None):
    plan = plan or _plan()
    passes = [{key: value for key, value in _base(model_id=model_id, expected_stems=list(stems)).items()
               if key not in {"input_preparation", "serialization"}}
              for model_id, stems in (("karaoke", DUALITY), ("four", FOUR))]
    return recipe.stem_algebra_recipe_revision(
        passes=passes, input_preparation={"decoder": "test"}, serialization={"subtype": "PCM_24"},
        algebra=plan.recipe_stage())


def test_algebra_recipe_is_stable_validated_and_its_own_capability():
    first = _algebra_revision()
    assert first == _algebra_revision()
    assert first["recipe_id"].startswith("stem-algebra-v1-")
    assert first["capability"] == "stem_algebra"
    assert recipe.validate_recipe_revision(first) == first
    assert first["publication"] == {"stems": ["backing vocals + drums"]}
    terms = first["stage_graph"][-1]["terms"]
    assert [(t["pass"], t["stem"], t["gain"]) for t in terms] == [(0, "Vocals", -1.0), (1, "drums", 1.0), (1, "vocals", 1.0)]


@pytest.mark.parametrize("tamper", ["gain", "content", "stem", "unused_pass"])
def test_tampered_algebra_recipe_is_rejected(tamper):
    revision = json.loads(json.dumps(_algebra_revision()))
    algebra = revision["stage_graph"][-1]
    if tamper == "gain":
        algebra["terms"][0]["gain"] = 0.5
    elif tamper == "content":
        algebra["terms"][0]["content"] = ["backing_vocals"]
    elif tamper == "stem":
        algebra["terms"][1]["stem"] = "piano"
    else:
        algebra["terms"] = [term for term in algebra["terms"] if term["pass"] == 1]
    with pytest.raises(ValueError):
        recipe.validate_recipe_revision(revision)


# --- 分離ジョブ・製品要求 ------------------------------------------------------------

MODELS = {"allvox": DUALITY, "karaoke": DUALITY, "four": FOUR}
DECLARED = {"allvox": ALLVOX, "karaoke": KARAOKE, "four": FOUR_DECLARED}


def _setup_worker(tmp_path, monkeypatch, rates=None, declared=DECLARED):
    rates = rates or {}
    monkeypatch.setattr(gpu_lease, "acquire", lambda **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    ledger = tmp_path / "models.tsv"
    header = ["model_id", "arch", "distributor_url", "config_url", "size_bytes", "状態",
              "last_used_at", "rtf", "peak_vram_mb", "note"]
    tsv_io.save_tsv(ledger, header, [{
        "model_id": model_id, "arch": "bs_roformer", "distributor_url": f"https://example.com/{model_id}.ckpt",
        "config_url": f"https://example.com/{model_id}.yaml", "size_bytes": "",
        "状態": "既定" if model_id == "allvox" else "候補",
        "last_used_at": "", "rtf": "", "peak_vram_mb": "", "note": "",
    } for model_id in MODELS])
    monkeypatch.setattr(models_store, "MODELS_TSV", ledger)
    monkeypatch.setattr(stem_semantics, "SEMANTICS_TSV", _write_table(tmp_path / "semantics.tsv", declared))
    configs = {}
    for model_id, stems in MODELS.items():
        config = tmp_path / f"{model_id}.yaml"
        config.write_text(f"audio:\n  sample_rate: {rates.get(model_id, 44100)}\n"
                          f"training:\n  instruments: [{', '.join(stems)}]\n", encoding="utf-8")
        configs[model_id] = config
    monkeypatch.setattr(models_store, "ensure_model_cached",
                        lambda model_id, *a, **k: (tmp_path / f"{model_id}.ckpt", configs[model_id]))
    monkeypatch.setattr(catalog, "stem_targets", lambda row: list(MODELS[row["model_id"]]))
    monkeypatch.setattr(infer, "capture_msst_runtime", lambda: RuntimeRevision(
        provider="msst", device="cuda:0", python_version="3.11", platform="test",
        torch_version="test", cuda_build="test", cudnn_version="test",
        gpu_name="test gpu", gpu_capability="12.0", adapter_artifacts=(("provider.py", "a" * 64),),
        dependency_versions=(("torch", "test"),),
    ))

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 29, tzinfo=timezone.utc)

    if not (resource_root() / "bin" / "ffmpeg.exe").is_file():
        monkeypatch.setattr(flac_frames, "stream_is_exact", lambda path: True)
    monkeypatch.setattr(writer, "datetime", _Frozen)


def _source(tmp_path) -> Path:
    path = tmp_path / "source.wav"
    t = np.arange(44100, dtype=np.float64) / 44100
    left = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 3100 * t)
    right = 0.3 * np.sin(2 * np.pi * 330 * t)
    sf.write(path, np.stack([left, right], axis=1).astype(np.float32), 44100, subtype="PCM_24")
    return path


def _component(model_id: str, stem: str, data: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(f"{model_id}/{stem}".encode()))
    return (0.4 * rng.uniform(-1.0, 1.0, data.shape) + 0.3 * np.sign(data)).astype(np.float32)


def _separator(calls):
    def separate(arch, mix_path, ckpt_path, config_path, stem_target, store_dir, **kwargs):
        model_id = Path(ckpt_path).stem
        calls.append((model_id, stem_target, str(mix_path)))
        kwargs["runtime_observer"](kwargs["runtime_revision"])
        data, sr = sf.read(mix_path, always_2d=True, dtype="float32")
        for stem in MODELS[model_id]:
            sf.write(store_dir / f"{stem}.wav", _component(model_id, stem, data), sr, subtype="FLOAT")
    return separate


def test_worker_publishes_the_cross_pass_algebra_with_provenance(tmp_path, monkeypatch):
    _setup_worker(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(infer, "msst_separate", _separator(calls))
    source = _source(tmp_path)

    result = worker.handle_job({"mix_path": str(source), "model_id": "karaoke", "stem_target": "chorus+drums"})

    assert result["status"] == "ok", result
    assert [Path(p).name for p in result["outputs"]] == ["source - backing vocals + drums.flac"]
    assert result["recipe_id"].startswith("stem-algebra-v1-")
    assert [call[:2] for call in calls] == [("karaoke", "Vocals"), ("four", "drums")]
    assert len({call[2] for call in calls}) == 1  # 同じ復号済み入力を両方のpassへ
    published = Path(result["outputs"][0])
    data, _ = sf.read(published, dtype="float64", always_2d=True)
    tags = writer.read_source_tags(published)
    shift = int(tags["LSS_OUTPUT_HEADROOM_SHIFT"])
    mix, _ = sf.read(source, always_2d=True, dtype="float32")
    exact = (_component("four", "drums", mix).astype(np.float64)
             + _component("four", "vocals", mix).astype(np.float64)
             - _component("karaoke", "Vocals", mix).astype(np.float64)).astype(np.float32).astype(np.float64)
    bits = {"PCM_16": 16, "PCM_24": 24}[sf.info(published).subtype]
    assert np.max(np.abs(data * 2.0 ** shift - exact)) <= 2.0 ** shift * 2.0 ** -(bits - 1)
    receipt = json.loads(tags["LSS_STEM_COMBINATION"])
    assert receipt["route"] == "cross_pass_algebra" and receipt["passes"] == ["karaoke", "four"]
    assert receipt["components"] == ["karaoke/Vocals", "four/drums", "four/vocals"]
    assert receipt["gains"] == [-1.0, 1.0, 1.0]
    revision = json.loads(tags["LSS_RECIPE_REVISION"])
    assert revision["capability"] == "stem_algebra"
    assert [stage.get("model_id") for stage in revision["stage_graph"][:-1]] == ["karaoke", "four"]

    again = worker.handle_job({"mix_path": str(source), "model_id": "karaoke", "stem_target": "drums+backing"})
    assert again["status"] == "skipped" and again["recipe_id"] == result["recipe_id"] and len(calls) == 2


def test_worker_refuses_passes_whose_alignment_is_not_proven(tmp_path, monkeypatch):
    _setup_worker(tmp_path, monkeypatch, rates={"four": 48000})
    monkeypatch.setattr(infer, "msst_separate", lambda *a, **k: pytest.fail("分離しない"))
    result = worker.handle_job({"mix_path": str(_source(tmp_path)), "model_id": "karaoke",
                                "stem_target": "chorus+drums"})
    assert result["status"] == "error" and result["error_code"] == "stem_combination_incomplete"
    assert not list(tmp_path.rglob("*.flac"))


def test_worker_never_combines_undeclared_passes(tmp_path, monkeypatch):
    _setup_worker(tmp_path, monkeypatch, declared={"karaoke": KARAOKE, "allvox": ALLVOX})
    monkeypatch.setattr(infer, "msst_separate", lambda *a, **k: pytest.fail("分離しない"))
    result = worker.handle_job({"mix_path": str(_source(tmp_path)), "model_id": "karaoke",
                                "stem_target": "chorus+drums"})
    assert result["status"] == "error" and result["error_code"] == "stem_combination_unsupported"


def _catalog(tmp_path, monkeypatch, models, declared):
    ledger = tmp_path / "models.tsv"
    rows = [{"model_id": model_id, "状態": state, "config_url": f"https://example.test/{model_id}.yaml"}
            for model_id, state, _ in models]
    tsv_io.save_tsv(ledger, list(rows[0]), rows)
    monkeypatch.setattr(models_store, "MODELS_TSV", ledger)
    monkeypatch.setattr(stem_semantics, "SEMANTICS_TSV", _write_table(tmp_path / "semantics.tsv", declared))
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
                                          "recipe_id": "stem-algebra-v1-x"},
        state_path=tmp_path / "queue.json",
    )
    request = application.ProcessingRequest(
        sources=(source,), intent=application.UserIntent.EXTRACT_STEM_COMBINATION, stems=tuple(stems))
    submission = application.submit(request, queue)
    queue.run()
    return submission, calls


def test_requested_combinations_become_reachable_with_declared_models(tmp_path, monkeypatch):
    models = [("allvox", "既定", DUALITY), ("karaoke", "候補", DUALITY), ("four", "候補", FOUR)]
    _catalog(tmp_path, monkeypatch, models, DECLARED)
    submission, calls = _submit(tmp_path, ["chorus", "instrumental"])
    assert [(job["model_id"], job["stem_target"]) for job in calls] == [("karaoke", "Instrumental")]
    assert submission.stem_combination["route"] == "direct_stem"

    (tmp_path / "b").mkdir()
    _catalog(tmp_path / "b", monkeypatch, models, DECLARED)
    submission, calls = _submit(tmp_path / "b", ["vocals", "drums"])
    assert [(job["model_id"], job["stem_target"]) for job in calls] == [("four", "drums+vocals")]

    (tmp_path / "c").mkdir()
    _catalog(tmp_path / "c", monkeypatch, models, DECLARED)
    submission, calls = _submit(tmp_path / "c", ["chorus", "drums"])
    assert [(job["model_id"], job["stem_target"]) for job in calls] == [("karaoke", "chorus+drums")]
    assert submission.stem_combination["route"] == "cross_pass_algebra"
    assert submission.stem_combination["passes"] == ["karaoke", "four"]


def test_without_declarations_the_karaoke_request_stays_unsupported(tmp_path, monkeypatch):
    # 宣言が無ければ、名前で読む2つのduality modelでは合唱を分けられない（従来どおり）。
    _catalog(tmp_path, monkeypatch, [("allvox", "既定", DUALITY), ("karaoke", "候補", DUALITY)], {})
    submission, calls = _submit(tmp_path, ["chorus", "instrumental"])
    assert calls == [] and submission.outcome.error_code == "stem_combination_unsupported"
