"""伴奏抽出の対応範囲と限界。宣言した条件を満たしたこと以上を言わない。"""
from __future__ import annotations

import pytest

from lss.validation.instrumental_scope import (
    InstrumentalScope, InstrumentalScopeError, describe, evaluate,
)


def scope(**changes) -> InstrumentalScope:
    base = {"input_domains": ("finished_stereo_music",), "minimum_snr_db": 6.0,
            "minimum_si_sdr_db": 6.0, "maximum_level_loss_db": 1.0,
            "maximum_vocal_transfer_db": -20.0, "maximum_channel_gain_error_db": 0.5,
            "maximum_side_balance_error_db": 1.0, "maximum_target_absent_output_db": -20.0,
            "allow_spatial_component_loss": False}
    return InstrumentalScope(**{**base, **changes})


def quality(**changes) -> dict:
    fidelity = {"si_sdr_db": {"joint": 20.0}, "scale_dependent_snr_db": 20.0,
                "projection_gain_db": {"joint": -0.1}, "relative_channel_gain_error_db": 0.01,
                "side_to_mid_balance_error_db": 0.05,
                "spatial_component_lost": {"left": False, "right": False, "mid": False, "side": False}}
    base = {"fidelity": fidelity, "target_present": True,
            "target_absent_output_to_vocals_db": None,
            "component_fit": {"vocals_transfer_db": -30.0}}
    for key, value in changes.items():
        if key in fidelity:
            fidelity[key] = value
        else:
            base[key] = value
    return base


def test_a_case_meeting_every_declared_condition_is_inside_the_scope():
    decided = evaluate(quality(), scope(), vocals_present=True)
    assert decided["within_scope"] is True and decided["failures"] == []
    assert decided["scope_id"] == scope().scope_id
    assert "component_fit.vocals_transfer_db" in decided["checked"]
    # 範囲の内側でも、意味的な良さやHoldoutでの一般化は何も言っていない。
    assert any("Holdout" in item for item in decided["not_checked"])


@pytest.mark.parametrize("changes,check", [
    ({"scale_dependent_snr_db": 5.9}, "scale_dependent_snr_db"),
    ({"si_sdr_db": {"joint": 5.0}}, "si_sdr_db.joint"),
    ({"projection_gain_db": {"joint": -1.6}}, "projection_gain_db.joint"),
    ({"relative_channel_gain_error_db": -0.9}, "relative_channel_gain_error_db"),
    ({"side_to_mid_balance_error_db": 1.4}, "side_to_mid_balance_error_db"),
    ({"component_fit": {"vocals_transfer_db": -8.0}}, "component_fit.vocals_transfer_db"),
    ({"spatial_component_lost": {"left": False, "right": False, "mid": False, "side": True}},
     "spatial_component_lost"),
])
def test_every_way_out_of_the_scope_has_a_name(changes, check):
    decided = evaluate(quality(**changes), scope(), vocals_present=True)
    assert [item["check"] for item in decided["failures"]] == [check]
    assert decided["within_scope"] is False


def test_an_input_without_singing_is_not_judged_on_vocal_residue():
    measured = quality(component_fit=None)
    decided = evaluate(measured, scope(), vocals_present=False)
    assert decided["within_scope"] is True
    assert decided["not_applicable"] == ["component_fit.vocals_transfer_db"]
    # 歌唱があるはずなのに測れなかった場合は、残留なしとして通さない。
    unknown = evaluate(measured, scope(), vocals_present=True)
    assert [item["check"] for item in unknown["failures"]] == ["component_fit.vocals_transfer_db"]
    assert evaluate(measured, scope())["within_scope"] is False


def test_a_target_absent_input_is_judged_only_on_what_leaks_out():
    absent = quality(target_present=False, target_absent_output_to_vocals_db=-40.0,
                     component_fit=None)
    decided = evaluate(absent, scope(), vocals_present=True)
    assert decided["within_scope"] is True
    assert decided["checked"] == ["target_absent_output_to_vocals_db"]
    loud = quality(target_present=False, target_absent_output_to_vocals_db=-5.0,
                   component_fit=None)
    assert [item["check"] for item in evaluate(loud, scope())["failures"]] == [
        "target_absent_output_to_vocals_db"]
    missing = quality(target_present=False, target_absent_output_to_vocals_db=None,
                      component_fit=None)
    assert evaluate(missing, scope())["within_scope"] is False


def test_allowing_spatial_loss_moves_it_out_of_the_checked_conditions():
    broken = quality(spatial_component_lost={"left": False, "right": False, "mid": False, "side": True})
    allowed = evaluate(broken, scope(allow_spatial_component_loss=True), vocals_present=True)
    assert allowed["within_scope"] is True
    assert allowed["not_applicable"] == ["spatial_component_lost"]
    assert "spatial_component_lost" not in allowed["checked"]


@pytest.mark.parametrize("changes", [
    {"input_domains": ()}, {"input_domains": ("",)}, {"input_domains": ("a", "a")},
    {"minimum_snr_db": float("nan")}, {"minimum_si_sdr_db": float("inf")},
    {"maximum_level_loss_db": -0.1}, {"maximum_channel_gain_error_db": -1.0},
    {"maximum_side_balance_error_db": -1.0}, {"maximum_vocal_transfer_db": True},
    {"allow_spatial_component_loss": "no"}, {"schema_version": 2}, {"method": "guessed"},
])
def test_an_incomplete_declaration_can_never_judge_a_case(changes):
    broken = scope(**changes)
    with pytest.raises(InstrumentalScopeError):
        broken.to_dict()
    with pytest.raises(InstrumentalScopeError):
        evaluate(quality(), broken, vocals_present=True)


def test_measurements_from_somewhere_else_are_refused():
    with pytest.raises(InstrumentalScopeError):
        evaluate({"scale_dependent_snr_db": 20.0}, scope())
    with pytest.raises(InstrumentalScopeError):
        evaluate({"target_present": True}, scope())
    with pytest.raises(InstrumentalScopeError):
        evaluate(quality(), scope(), vocals_present="yes")


def test_a_description_counts_the_limits_by_reason_and_by_variant():
    cases = [
        {"case_id": "a", "variant": "accompaniment-only", "vocals_present": False,
         "quality": quality(component_fit=None)},
        {"case_id": "b", "variant": "singing-mix", "vocals_present": True,
         "quality": quality(component_fit={"vocals_transfer_db": -8.0})},
        {"case_id": "c", "variant": "singing-mix", "vocals_present": True,
         "quality": quality(component_fit={"vocals_transfer_db": -9.0})},
    ]
    described = describe(cases, scope())
    assert described["case_count"] == 3 and described["within_scope"] == 1
    assert described["reasons"] == {"component_fit.vocals_transfer_db": 2}
    assert described["by_variant"]["singing-mix"] == {"cases": 2, "within_scope": 0}
    assert described["by_variant"]["accompaniment-only"] == {"cases": 1, "within_scope": 1}
    assert described["cases"][0]["not_applicable"] == ["component_fit.vocals_transfer_db"]


def test_duplicate_or_unnamed_cases_are_refused():
    case = {"case_id": "a", "variant": "x", "vocals_present": True, "quality": quality()}
    with pytest.raises(InstrumentalScopeError):
        describe([case, dict(case)], scope())
    with pytest.raises(InstrumentalScopeError):
        describe([{"variant": "x", "quality": quality()}], scope())


def test_the_runner_derives_singing_presence_from_the_corpus_definition(tmp_path, monkeypatch):
    """歌唱の有無はvariant定義の歌gainから決める。名前の思い込みで決めない。"""
    import importlib.util
    import json as json_module

    source = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "describe_instrumental_scope.py"
    spec = importlib.util.spec_from_file_location("describe_instrumental_scope", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.VOCALS_PRESENT == {"accompaniment-only": False, "singing-mix": True,
                                     "vocals-only": True, "quiet-accompaniment": True}

    results = tmp_path / "results"
    results.mkdir()
    for name, variant, vocals in (("a", "accompaniment-only", None),
                                  ("b", "singing-mix", {"vocals_transfer_db": -8.0})):
        (results / f"{name}-result.json").write_text(json_module.dumps(
            {"case_id": name, "variant": variant, "quality": quality(component_fit=vocals)}),
            encoding="utf-8")
    cases = module.load_cases(results)
    assert {case["case_id"]: case["vocals_present"] for case in cases} == {"a": False, "b": True}
    described = describe([{key: value for key, value in case.items() if key != "source"}
                          for case in cases], scope())
    assert described["within_scope"] == 1

    unknown = results / "c-result.json"
    unknown.write_text(json_module.dumps({"case_id": "c", "variant": "mystery",
                                          "quality": quality()}), encoding="utf-8")
    with pytest.raises(InstrumentalScopeError, match="variant"):
        module.load_cases(results)


def test_the_runner_refuses_an_incomplete_scope_declaration(tmp_path):
    import importlib.util
    import json as json_module

    source = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "describe_instrumental_scope.py"
    spec = importlib.util.spec_from_file_location("describe_instrumental_scope_2", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "scope.json"
    path.write_text(json_module.dumps({"minimum_snr_db": 6.0}), encoding="utf-8")
    with pytest.raises(InstrumentalScopeError, match="不完全"):
        module.load_scope(path)
    path.write_text(json_module.dumps({**scope().to_dict(), "minimum_snr_db": None}), encoding="utf-8")
    with pytest.raises(InstrumentalScopeError):
        module.load_scope(path)
