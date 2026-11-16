"""出力へ影響する選択を固定した、機械可読なRecipe revision。"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


SEPARATION_RECIPE_SCHEMA_VERSION = 1
TARGET_ABSENCE_RECIPE_SCHEMA_VERSION = 1
WANTED_SOUND_RECIPE_SCHEMA_VERSION = 1
STEM_COMBINATION_RECIPE_SCHEMA_VERSION = 1
STEM_ALGEBRA_RECIPE_SCHEMA_VERSION = 1


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _provider_stage(
    *, provider: str, arch: str, model_id: str, model_sha256: str, config_sha256: str,
    runtime_id: str, expected_stems: list[str], output_samplerate: int,
) -> dict[str, Any]:
    return {
        "stage": "provider_separation",
        "provider": provider,
        "arch": arch,
        "model_id": model_id,
        "model_sha256": model_sha256,
        "config_sha256": config_sha256,
        "runtime_id": runtime_id,
        "expected_stems": list(expected_stems),
        "output_samplerate": output_samplerate,
    }


def separation_recipe_revision(
    *,
    provider: str,
    arch: str,
    model_id: str,
    model_sha256: str,
    config_sha256: str,
    runtime_id: str,
    expected_stems: list[str],
    output_samplerate: int,
    input_preparation: dict,
    serialization: dict,
    published_stems: list[str] | None = None,
    post_processing: dict | None = None,
) -> dict[str, Any]:
    """現行single-stage分離graphの不変identityを生成する。"""
    # 内部生成物が同じでも、公開対象が違えば別の結果である。公開一式は
    # headroomの計算単位でもあり、完了・zero判定をパート間で共有してはいけない。
    available = {name.casefold(): name for name in expected_stems}
    selected = expected_stems if published_stems is None else published_stems
    if (
        not selected
        or any(not isinstance(name, str) or name.casefold() not in available for name in selected)
        or len({name.casefold() for name in selected}) != len(selected)
    ):
        raise ValueError("Recipeの公開パートが空、不明、または重複しています")
    canonical_stems = [available[key] for key in sorted(name.casefold() for name in selected)]
    revision = {
        "schema_version": SEPARATION_RECIPE_SCHEMA_VERSION,
        "capability": "stem_separation",
        "stage_graph": [_provider_stage(
            provider=provider, arch=arch, model_id=model_id, model_sha256=model_sha256,
            config_sha256=config_sha256, runtime_id=runtime_id, expected_stems=expected_stems,
            output_samplerate=output_samplerate,
        )],
        "input_preparation": deepcopy(input_preparation),
        "serialization": deepcopy(serialization),
        "publication": {"stems": canonical_stems},
    }
    if post_processing is not None:
        revision["stage_graph"].append({"stage": "selective_original_composition",
                                        "revision": deepcopy(post_processing)})
    canonical = _canonical(revision)
    revision["recipe_id"] = (
        f"separation-v{SEPARATION_RECIPE_SCHEMA_VERSION}-"
        f"{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    )
    return revision


def stem_combination_recipe_revision(
    *,
    provider: str,
    arch: str,
    model_id: str,
    model_sha256: str,
    config_sha256: str,
    runtime_id: str,
    expected_stems: list[str],
    output_samplerate: int,
    input_preparation: dict,
    serialization: dict,
    combination: dict,
) -> dict[str, Any]:
    """1つの分離passの出力を足した組み合わせ1つ分のidentity（`lss.separation.stem_combination`）。

    単一ステムのRecipeとは別のcapabilityなので、部品のどの単一ステムとも完了・zero・CurrentBestを
    共有しない。部品・利得・中身・規則が音を決めるので、それだけを入れる。
    """
    _check_combination_stage(combination, expected_stems)
    revision = {
        "schema_version": STEM_COMBINATION_RECIPE_SCHEMA_VERSION,
        "capability": "stem_combination",
        "stage_graph": [
            _provider_stage(
                provider=provider, arch=arch, model_id=model_id, model_sha256=model_sha256,
                config_sha256=config_sha256, runtime_id=runtime_id, expected_stems=expected_stems,
                output_samplerate=output_samplerate,
            ),
            deepcopy(combination),
        ],
        "input_preparation": deepcopy(input_preparation),
        "serialization": deepcopy(serialization),
        "publication": {"stems": [combination["label"]]},
    }
    canonical = _canonical(revision)
    revision["recipe_id"] = (
        f"stem-combination-v{STEM_COMBINATION_RECIPE_SCHEMA_VERSION}-"
        f"{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    )
    return revision


def _check_combination_stage(stage: object, expected_stems: object) -> None:
    if (not isinstance(stage, dict) or stage.get("stage") != "stem_sum"
            or not isinstance(stage.get("rule"), str) or not stage["rule"]
            or not isinstance(stage.get("components"), list) or len(stage["components"]) < 2
            or not isinstance(stage.get("gains"), list)
            or len(stage["gains"]) != len(stage["components"])
            or not isinstance(stage.get("label"), str) or not stage["label"].strip()
            or not isinstance(expected_stems, list)):
        raise ValueError("組み合わせRecipeの節が不正です")
    names = [name for name in stage["components"] if isinstance(name, str)]
    if (len(names) != len(stage["components"]) or len(set(names)) != len(names)
            or any(name not in expected_stems for name in names)
            or any(isinstance(gain, bool) or not isinstance(gain, (int, float)) for gain in stage["gains"])):
        raise ValueError("組み合わせRecipeの部品が分離passの出力と一致しません")


def stem_algebra_recipe_revision(
    *,
    passes: list[dict],
    input_preparation: dict,
    serialization: dict,
    algebra: dict,
) -> dict[str, Any]:
    """別々の分離passの出力を符号付き単位利得で足した組み合わせ1つ分のidentity。

    ``passes`` は各passのprovider節の値（同じ復号済み入力へ各modelを1回）。``algebra`` は
    ``AlgebraPlan.recipe_stage()``。1passの組み合わせ（``stem_combination``）とは別のcapabilityで、
    どの単一ステム・1passの組み合わせとも完了・zero・CurrentBestを共有しない。
    """
    stages = [_provider_stage(**dict(item)) for item in passes]
    _check_algebra_stage(algebra, stages)
    revision = {
        "schema_version": STEM_ALGEBRA_RECIPE_SCHEMA_VERSION,
        "capability": "stem_algebra",
        "stage_graph": [*stages, deepcopy(algebra)],
        "input_preparation": deepcopy(input_preparation),
        "serialization": deepcopy(serialization),
        "publication": {"stems": [algebra["label"]]},
    }
    canonical = _canonical(revision)
    revision["recipe_id"] = (
        f"stem-algebra-v{STEM_ALGEBRA_RECIPE_SCHEMA_VERSION}-"
        f"{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    )
    return revision


def _check_algebra_stage(stage: object, providers: object) -> None:
    if (not isinstance(providers, list) or len(providers) < 2
            or any(not isinstance(item, dict) or item.get("stage") != "provider_separation"
                   or not isinstance(item.get("expected_stems"), list) for item in providers)
            or len({item.get("model_id") for item in providers}) != len(providers)):
        raise ValueError("代数Recipeの分離passが不正です")
    if (not isinstance(stage, dict) or stage.get("stage") != "aligned_stem_algebra"
            or not isinstance(stage.get("rule"), str) or not stage["rule"]
            or not isinstance(stage.get("alignment"), str) or not stage["alignment"]
            or not isinstance(stage.get("terms"), list) or len(stage["terms"]) < 2
            or not isinstance(stage.get("content"), list) or not stage["content"]
            or not isinstance(stage.get("label"), str) or not stage["label"].strip()):
        raise ValueError("代数Recipeの節が不正です")
    used = set()
    seen = set()
    for term in stage["terms"]:
        if not isinstance(term, dict) or set(term) != {"pass", "stem", "gain", "content"}:
            raise ValueError("代数Recipeの項が不正です")
        index, stem, gain = term["pass"], term["stem"], term["gain"]
        if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(providers)
                or not isinstance(stem, str) or stem not in providers[index]["expected_stems"]
                or isinstance(gain, bool) or gain not in (1.0, -1.0)
                or not isinstance(term["content"], list) or not term["content"]
                or (index, stem) in seen):
            raise ValueError("代数Recipeの項が分離passの出力・単位利得と一致しません")
        seen.add((index, stem))
        used.add(index)
    if used != set(range(len(providers))):
        raise ValueError("代数Recipeに項の無い分離passがあります")
    total: dict[str, float] = {}
    for term in stage["terms"]:
        for atom in term["content"]:
            total[atom] = total.get(atom, 0.0) + float(term["gain"])
    if {atom for atom, value in total.items() if value} != set(stage["content"]) or any(
            value not in (0.0, 1.0) for value in total.values()):
        raise ValueError("代数Recipeの項の和が宣言した中身と一致しません")


def target_absence_recipe_revision(
    *, requested_target: str, input_preparation: dict, validation: dict,
) -> dict[str, Any]:
    """推論不要で確定できるtarget absenceのRecipe identity。

    完全無音はmodel、checkpoint、GPU runtimeに依存しない。存在しないmodel artifactを
    Recipeへ捏造せず、復号条件とexact validatorだけを固定する。
    """
    if not isinstance(requested_target, str) or not requested_target.strip():
        raise ValueError("対象なしRecipeには要求targetが必要です")
    if not isinstance(input_preparation, dict) or not isinstance(validation, dict):
        raise ValueError("対象なしRecipeの復号条件または検証条件が不正です")
    revision = {
        "schema_version": TARGET_ABSENCE_RECIPE_SCHEMA_VERSION,
        "capability": "validated_target_absence",
        "stage_graph": [{
            "stage": "exact_signal_presence_validation",
            "requested_target": requested_target,
            "validation": deepcopy(validation),
        }],
        "input_preparation": deepcopy(input_preparation),
        "publication": {"stems": []},
    }
    canonical = _canonical(revision)
    revision["recipe_id"] = (
        f"target-absence-v{TARGET_ABSENCE_RECIPE_SCHEMA_VERSION}-"
        f"{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    )
    return revision


def wanted_sound_recipe_revision(
    *, request: dict, label: str, planner: dict, engines: dict, verification: dict,
    input_preparation: dict, serialization: dict,
) -> dict[str, Any]:
    """言葉で指定した欲しい音1つ分のRecipe identity。

    経路（どの分離器をどの親へ使うか）は音源の性質から決まるので、identityは「決め方」
    （計画の実装と使う分離器・検証の版）で固定する。実際に選んだ経路は成果物の記録に残す。
    """
    if not isinstance(request, dict) or not request.get("atoms"):
        raise ValueError("欲しい音のRecipeには読めた要求が必要です")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("欲しい音のRecipeには成果物の名前が必要です")
    for name, value in (("planner", planner), ("engines", engines), ("verification", verification),
                        ("input_preparation", input_preparation), ("serialization", serialization)):
        if not isinstance(value, dict):
            raise ValueError(f"欲しい音のRecipeの{name}が不正です")
    revision = {
        "schema_version": WANTED_SOUND_RECIPE_SCHEMA_VERSION,
        "capability": "wanted_sound",
        "stage_graph": [{
            "stage": "wanted_sound_extraction",
            "request": deepcopy(request),
            "planner": deepcopy(planner),
            "engines": deepcopy(engines),
            "verification": deepcopy(verification),
        }],
        "input_preparation": deepcopy(input_preparation),
        "serialization": deepcopy(serialization),
        "publication": {"stems": [label]},
    }
    canonical = _canonical(revision)
    revision["recipe_id"] = (
        f"wanted-sound-v{WANTED_SOUND_RECIPE_SCHEMA_VERSION}-"
        f"{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    )
    return revision


def validate_recipe_revision(value: object) -> dict[str, Any]:
    """保存・再利用時にRecipe本文とIDの一致を検証する。"""
    if not isinstance(value, dict) or not isinstance(value.get("recipe_id"), str):
        raise ValueError("Recipe revisionが不正です")
    revision = deepcopy(value)
    recipe_id = revision.pop("recipe_id")
    schema_version = revision.get("schema_version")
    capability = revision.get("capability")
    if capability == "stem_separation":
        supported_schema = SEPARATION_RECIPE_SCHEMA_VERSION
        prefix = "separation"
    elif capability == "stem_combination":
        supported_schema = STEM_COMBINATION_RECIPE_SCHEMA_VERSION
        prefix = "stem-combination"
        graph = revision.get("stage_graph")
        publication = revision.get("publication")
        if (not isinstance(graph, list) or len(graph) != 2 or not isinstance(graph[0], dict)
                or graph[0].get("stage") != "provider_separation"):
            raise ValueError("組み合わせRecipeの構造が不正です")
        _check_combination_stage(graph[1], graph[0].get("expected_stems"))
        if publication != {"stems": [graph[1]["label"]]}:
            raise ValueError("組み合わせRecipeの公開パートが不正です")
    elif capability == "stem_algebra":
        supported_schema = STEM_ALGEBRA_RECIPE_SCHEMA_VERSION
        prefix = "stem-algebra"
        graph = revision.get("stage_graph")
        publication = revision.get("publication")
        if not isinstance(graph, list) or len(graph) < 3:
            raise ValueError("代数Recipeの構造が不正です")
        _check_algebra_stage(graph[-1], graph[:-1])
        if publication != {"stems": [graph[-1]["label"]]}:
            raise ValueError("代数Recipeの公開パートが不正です")
    elif capability == "validated_target_absence":
        supported_schema = TARGET_ABSENCE_RECIPE_SCHEMA_VERSION
        prefix = "target-absence"
        graph = revision.get("stage_graph")
        publication = revision.get("publication")
        if (not isinstance(graph, list) or len(graph) != 1
                or not isinstance(graph[0], dict)
                or graph[0].get("stage") != "exact_signal_presence_validation"
                or not isinstance(graph[0].get("requested_target"), str)
                or not graph[0]["requested_target"].strip()
                or not isinstance(graph[0].get("validation"), dict)
                or publication != {"stems": []}
                or not isinstance(revision.get("input_preparation"), dict)):
            raise ValueError("対象なしRecipeの構造が不正です")
    elif capability == "wanted_sound":
        supported_schema = WANTED_SOUND_RECIPE_SCHEMA_VERSION
        prefix = "wanted-sound"
        graph = revision.get("stage_graph")
        publication = revision.get("publication")
        if (not isinstance(graph, list) or len(graph) != 1 or not isinstance(graph[0], dict)
                or graph[0].get("stage") != "wanted_sound_extraction"
                or not isinstance(graph[0].get("request"), dict)
                or not isinstance(publication, dict) or not isinstance(publication.get("stems"), list)
                or len(publication["stems"]) != 1):
            raise ValueError("欲しい音Recipeの構造が不正です")
    else:
        raise ValueError(f"未対応のRecipe capabilityです: {capability}")
    if schema_version != supported_schema:
        raise ValueError(f"未対応のRecipe schemaです: {schema_version}")
    expected = (
        f"{prefix}-v{schema_version}-"
        f"{hashlib.sha256(_canonical(revision).encode('utf-8')).hexdigest()}"
    )
    if recipe_id != expected:
        raise ValueError("Recipe revision本文とrecipe_idが一致しません")
    return deepcopy(value)
