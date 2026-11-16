"""不確かな分岐で捨てなかったMusic推定を、非破壊の保留候補として別記録に残す。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
from typing import Callable

import soundfile as sf

from lss.bgm.candidate_store import CandidateOrigin, materialize_candidate
from lss.core.session import _write_json_atomic
from lss.core import audio_identity
from lss.harness.fetch import file_sha256


ROOT = Path(__file__).resolve().parents[2]
RULE = "positive_disagreement_windows_without_music_energy_increase"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict:
    # 試聴でtagが書き戻されても、同じ音の素材として確認できるようにする。
    return audio_identity.material_record(path)


def verify(records: list[dict]) -> None:
    for record in records:
        if not audio_identity.matches_record(Path(record["path"]), record):
            raise ValueError(f"保留候補の入力・先行候補の内容が変更されています: {record['path']}")


def freeze(run_path: Path, output: Path) -> Path:
    prior = read_json(run_path)
    parent_plan_path = run_path.parent / "plan.json"
    parent = read_json(parent_plan_path)
    if (prior.get("status") != "completed_development_routed_episode_candidates"
            or prior.get("promotion_allowed") is not False or prior.get("holdout_consumed") is not False
            or prior.get("publication_state") != "not_published"
            or file_sha256(parent_plan_path) != prior["plan_sha256"]):
        raise ValueError("完了した未公開Development候補runが必要です")
    if prior["source"] != parent["source"] or parent["source"]["corpus_role"] != "development":
        raise ValueError("同じDevelopment原本の候補だけを保留します")
    rendering = read_json(run_path.parent / "rendering.json")
    if rendering != prior["rendering"] or len(prior["candidates"]) != len(rendering["cues"]):
        raise ValueError("先行候補とrenderingの対応が一致しません")
    music = parent["stems"]["music"]
    info = sf.info(music["path"])
    if info.samplerate != 44100 or info.channels != 2 or info.frames != parent["frames"]:
        raise ValueError("保留するMusic cacheの時間軸が一致しません")
    inputs = [artifact(run_path), artifact(parent_plan_path), artifact(run_path.parent / "rendering.json"),
              parent["source"], music, rendering["signal"]]
    cases = []
    timeline = parent["audio_timeline_origin"]
    for candidate, cue in zip(prior["candidates"], rendering["cues"]):
        origin = CandidateOrigin(**candidate["origin"])
        origin.validate()
        if (candidate["cue"] != cue or origin.candidate_id != candidate["asset"]["candidate_asset_id"]
                or origin.source_sha256 != parent["source"]["sha256"]
                or origin.signal_sha256 != rendering["signal"]["sha256"]
                or origin.start_frame != cue["start_frame"] or origin.end_frame != cue["end_frame"]):
            raise ValueError("先行候補の単独来歴と区間の対応が一致しません")
        inputs.append(candidate["asset"])
        if cue["positive_window_count"] > 0 and cue["use_music"] is False and cue["music_energy_increased"] is False:
            occurrence = {"source_sha256": origin.source_sha256, "start_frame": origin.start_frame,
                          "end_frame": origin.end_frame, "sample_rate": origin.sample_rate,
                          "time_offset_numerator": timeline["offset_numerator"],
                          "time_offset_denominator": timeline["offset_denominator"]}
            occurrence_id = "bgm-occurrence-v1-" + hashlib.sha256(json.dumps(
                occurrence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            cases.append({"occurrence": occurrence, "occurrence_id": occurrence_id, "cue": cue,
                          "primary_candidate_asset_id": candidate["asset"]["candidate_asset_id"]})
    verify(inputs)
    implementation = {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT / "lss/bgm/candidate_store.py",
    )}
    revision = {"method": "retain_unmodified_music_as_held_alternative", "retention_rule": RULE,
                "upstream_renderer_id": prior["renderer_id"],
                "music_candidate": parent["renderer_revision"]["music_candidate"],
                "implementation": implementation, "serialization_policy": "declared_power_of_two_headroom_v1"}
    renderer_id = "bgm-held-music-v1-" + hashlib.sha256(json.dumps(
        revision, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    runtime_revision = {"source_music": parent["renderer_revision"]["runtime_revision"]["music"],
                        "serialization": {"python": platform.python_version(), "soundfile": sf.__version__,
                                          "libsndfile": sf.__libsndfile_version__}}
    runtime_id = "bgm-held-music-runtime-v1-" + hashlib.sha256(json.dumps(
        runtime_revision, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    plan = output / "plan.json"
    _write_json_atomic(plan, {
        "schema_version": 1, "status": "FROZEN_BEFORE_HELD_ALTERNATIVES", "promotion_allowed": False,
        "holdout_consumed": False, "source": parent["source"], "source_id": parent["source_id"],
        "music": music, "cases": cases, "inputs": inputs, "retention_rule": RULE,
        "renderer_id": renderer_id, "renderer_revision": revision,
        "runtime_id": runtime_id, "runtime_revision": runtime_revision,
        "activity_recipe_id": parent["activity_recipe"]["recipe_id"], "implementation": implementation,
        "upstream_alignment_status": parent["alignment_status"],
        "limits": ["held_alternative_is_not_a_winner_or_semantic_label",
                   "energy_veto_does_not_prove_singing_absence",
                   "original_candidate_run_is_unchanged", "no_source_audio_uploaded"],
    })
    return plan


def run(plan_path: Path, *, on_candidate: Callable[[dict], None] | None = None) -> Path:
    plan = read_json(plan_path)
    if (plan.get("status") != "FROZEN_BEFORE_HELD_ALTERNATIVES" or plan.get("promotion_allowed") is not False
            or plan.get("holdout_consumed") is not False or plan.get("retention_rule") != RULE):
        raise ValueError("固定したDevelopment保留候補の構成が必要です")
    verify(plan["inputs"])
    for relative, expected in plan["implementation"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError("保留候補の固定実装が変更されています")
    plan_hash = file_sha256(plan_path)
    candidates = []
    for case in plan["cases"]:
        origin = CandidateOrigin(**case["occurrence"], signal_sha256=plan["music"]["sha256"],
            separation_recipe_id=plan["renderer_id"], runtime_id=plan["runtime_id"],
            activity_recipe_id=plan["activity_recipe_id"], serialization_policy="declared_power_of_two_headroom_v1")
        saved = materialize_candidate(Path(plan["music"]["path"]), plan_path.parent / "candidates" / f"{origin.candidate_id}.flac",
            origin=origin, source_id=plan["source_id"], source_path=Path(plan["source"]["path"]),
            evidence={"candidate_role": "held_alternative", "hold_reason": RULE,
                      "occurrence_id": case["occurrence_id"],
                      "primary_candidate_asset_id": case["primary_candidate_asset_id"],
                      "renderer_revision": plan["renderer_revision"], "plan_sha256": plan_hash})
        candidates.append({**case, "origin": origin.to_dict(), "asset": saved,
                           "candidate_role": "held_alternative", "selection_state": "unresolved"})
        if on_candidate is not None:
            on_candidate(saved)
    verify(plan["inputs"])
    if file_sha256(plan_path) != plan_hash:
        raise ValueError("保留候補の保存中に固定構成が変更されています")
    for relative, expected in plan["implementation"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError("保留候補の保存中に固定実装が変更されています")
    report = plan_path.parent / "alternatives.json"
    _write_json_atomic(report, {
        "schema_version": 1, "status": "completed_development_held_alternatives",
        "promotion_allowed": False, "holdout_consumed": False, "publication_state": "not_published",
        "plan_sha256": plan_hash, "source": plan["source"], "candidates": candidates,
        "source_hashes_after_verified": True, "original_candidates_unchanged": True,
        "upstream_alignment_status": plan["upstream_alignment_status"],
        "outcome": "held_alternatives" if candidates else "no_eligible_ambiguous_candidates",
        "semantic_ground_truth": None, "selection_state": "unresolved", "limits": plan["limits"],
    })
    return report
