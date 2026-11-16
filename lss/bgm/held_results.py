"""同じrunの代替候補を、勝者を選ばず元候補へ結び付ける読取境界。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mutagen import MutagenError
from mutagen.flac import FLAC
import soundfile as sf

from lss.bgm import candidate_store
from lss.core import audio_identity
from lss.validation.held_bgm_alternatives import RULE


class HeldResultsError(RuntimeError):
    """代替候補を現在の原候補へ安全に結び付けられない。"""


def alternative_view(run_path: Path, report_path: Path, *, locator=None) -> dict:
    """固定した既存Stageの成果を読む。音声・元run・照合記録を変更しない。"""
    try:
        run_path, report_path = Path(run_path).resolve(), Path(report_path).resolve()
        if report_path.name != "alternatives.json":
            raise ValueError("完了した代替候補記録が必要です")
        plan_path = report_path.parent / "plan.json"
        upstream_plan_path = run_path.parent / "plan.json"
        run_bytes, plan_bytes, report_bytes = run_path.read_bytes(), plan_path.read_bytes(), report_path.read_bytes()
        upstream_plan_bytes = upstream_plan_path.read_bytes()
        upstream_plan = json.loads(upstream_plan_bytes)
        upstream, plan, report = (json.loads(value) for value in (run_bytes, plan_bytes, report_bytes))
        if not all(isinstance(value, dict) for value in (upstream, plan, report, upstream_plan)):
            raise ValueError("代替候補または元runの記録が不正です")
        upstream_plan_sha256 = hashlib.sha256(upstream_plan_bytes).hexdigest()
        if (upstream.get("status") != "completed_development_routed_episode_candidates"
                or upstream.get("promotion_allowed") is not False or upstream.get("holdout_consumed") is not False
                or upstream.get("publication_state") != "not_published"
                or upstream.get("plan_sha256") != upstream_plan_sha256
                or plan.get("music") != (upstream_plan.get("stems") or {}).get("music")):
            raise ValueError("代替候補のMusic信号が元runの固定構成へ対応しません")
        plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
        if (plan.get("status") != "FROZEN_BEFORE_HELD_ALTERNATIVES" or plan.get("retention_rule") != RULE
                or plan.get("promotion_allowed") is not False or plan.get("holdout_consumed") is not False
                or not isinstance(plan.get("source"), dict) or plan["source"].get("corpus_role") != "development"
                or plan["source"] != upstream.get("source")
                or (plan.get("renderer_revision") or {}).get("upstream_renderer_id") != upstream.get("renderer_id")):
            raise ValueError("代替候補の固定構成が同じDevelopment原本・処理へ対応しません")
        if (report.get("status") != "completed_development_held_alternatives"
                or type(report.get("schema_version")) is not int or report["schema_version"] != 1
                or report.get("promotion_allowed") is not False or report.get("holdout_consumed") is not False
                or report.get("publication_state") != "not_published" or report.get("source") != plan["source"]
                or report.get("source_hashes_after_verified") is not True
                or report.get("original_candidates_unchanged") is not True
                or report.get("semantic_ground_truth") is not None or report.get("selection_state") != "unresolved"
                or report.get("plan_sha256") != plan_sha256):
            raise ValueError("代替候補の完了・非公開・未解決契約が不正です")
        inputs, cases, candidates = plan.get("inputs"), plan.get("cases"), report.get("candidates")
        if (not isinstance(inputs, list) or any(not isinstance(item, dict) for item in inputs)
                or not isinstance(cases, list) or not isinstance(candidates, list) or len(cases) != len(candidates)):
            raise ValueError("代替候補の固定入力・区間・成果集合が不正です")
        run_sha256 = hashlib.sha256(run_bytes).hexdigest()
        if not any(Path(item["path"]).resolve() == run_path and item["sha256"] == run_sha256 for item in inputs):
            raise ValueError("代替候補が現在の元runの内容hashへ対応しません")
        if not any(Path(item["path"]).resolve() == upstream_plan_path and item["sha256"] == upstream_plan_sha256 for item in inputs):
            raise ValueError("代替候補が元runの固定構成の内容hashへ対応しません")
        expected_outcome = "held_alternatives" if candidates else "no_eligible_ambiguous_candidates"
        if report.get("outcome") != expected_outcome:
            raise ValueError("代替候補の完了結果と件数が一致しません")
        parents = {item["asset"]["candidate_asset_id"]: item for item in upstream["candidates"]}

        def locations(item):
            return locator(item) if locator is not None else (Path(item["path"]).resolve(),)

        def verify_inputs():
            for item in inputs:
                paths = locations(item)
                if not paths or any(not audio_identity.matches_record(path, item) for path in paths):
                    raise ValueError("代替候補の原本・cache・元候補が記録と一致しません")

        verify_inputs()
        artifacts, seen = [], set()
        for case, candidate in zip(cases, candidates):
            if not isinstance(case, dict) or not isinstance(candidate, dict):
                raise ValueError("代替候補の区間記録が不正です")
            primary_id = case["primary_candidate_asset_id"]
            if primary_id not in parents or primary_id in seen:
                raise ValueError("代替候補の親が現在の候補集合へ一意に対応しません")
            seen.add(primary_id)
            parent = parents[primary_id]
            occurrence, cue = case["occurrence"], parent["cue"]
            parent_origin = {**parent["origin"], "time_offset_numerator": parent["origin"].get("time_offset_numerator", 0),
                             "time_offset_denominator": parent["origin"].get("time_offset_denominator", 1)}
            if (case["cue"] != cue or cue.get("positive_window_count", 0) <= 0
                    or cue.get("use_music") is not False or cue.get("music_energy_increased") is not False
                    or any(occurrence[key] != parent_origin[key]
                           for key in ("source_sha256", "start_frame", "end_frame", "sample_rate",
                                       "time_offset_numerator", "time_offset_denominator"))):
                raise ValueError("代替候補が元候補の保留条件または区間へ対応しません")
            occurrence_id = "bgm-occurrence-v1-" + hashlib.sha256(json.dumps(
                occurrence, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode()).hexdigest()
            if case["occurrence_id"] != occurrence_id:
                raise ValueError("代替候補の登場区間IDが一致しません")
            if (any(candidate.get(key) != value for key, value in case.items())
                    or candidate.get("candidate_role") != "held_alternative"
                    or candidate.get("selection_state") != "unresolved"):
                raise ValueError("代替候補の来歴または保留状態が不正です")
            origin = candidate_store.CandidateOrigin(
                **occurrence, signal_sha256=plan["music"]["sha256"], separation_recipe_id=plan["renderer_id"],
                runtime_id=plan["runtime_id"], activity_recipe_id=plan["activity_recipe_id"],
                serialization_policy="declared_power_of_two_headroom_v1",
            )
            origin.validate()
            asset = candidate["asset"]
            if (candidate["origin"] != origin.to_dict() or asset.get("candidate_asset_id") != origin.candidate_id
                    or asset.get("representation") != "noncanonical_staged_candidate"
                    or asset.get("publication_state") != "not_published"
                    or not Path(asset["path"]).resolve().is_relative_to(report_path.parent / "candidates")):
                raise ValueError("代替音声の出典・処理・保存先が一致しません")
            paths = locations(asset)
            if not paths:
                raise ValueError("代替音声の所在を確認できません")
            for path in paths:
                if not audio_identity.matches_record(path, asset):
                    raise ValueError("代替音声の内容hashが一致しません")
                tags = FLAC(path)
                metadata = json.loads(tags["LSS_CANDIDATE_PROVENANCE"][0])
                evidence = metadata.get("evidence") or {}
                if (metadata.get("origin") != origin.to_dict() or metadata.get("candidate_id") != origin.candidate_id
                        or metadata.get("semantic_role") != "unresolved_music_like"
                        or metadata.get("identity_state") != "unresolved"
                        or metadata.get("publication_state") != "not_published"
                        or evidence.get("primary_candidate_asset_id") != primary_id
                        or evidence.get("occurrence_id") != occurrence_id or evidence.get("plan_sha256") != plan_sha256
                        or evidence.get("hold_reason") != RULE):
                    raise ValueError("代替音声の単独来歴が元候補へ対応しません")
                candidate_store._verify_crop(locations(plan["music"])[0], path, origin)
            artifacts.append({"path": str(Path(asset["path"]).resolve()), "sha256": asset["sha256"],
                              **({"audio": asset["audio"]} if isinstance(asset.get("audio"), dict) else {}),
                              "candidate_asset_id": origin.candidate_id, "primary_candidate_asset_id": primary_id,
                              "occurrence_id": occurrence_id})
        verify_inputs()
        if (run_path.read_bytes() != run_bytes or plan_path.read_bytes() != plan_bytes
                or report_path.read_bytes() != report_bytes or upstream_plan_path.read_bytes() != upstream_plan_bytes):
            raise ValueError("代替候補の確認中に元run・固定構成・完了記録が変更されています")
        return {"path": str(report_path), "sha256": hashlib.sha256(report_bytes).hexdigest(),
                "plan": {"path": str(plan_path), "sha256": plan_sha256}, "artifacts": artifacts,
                "semantic_role": "unresolved", "selection_state": "unresolved"}
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError,
            MutagenError, sf.LibsndfileError) as exc:
        raise HeldResultsError(f"現在のBGM保留結果へ代替候補を結び付けられません: {exc}") from exc
