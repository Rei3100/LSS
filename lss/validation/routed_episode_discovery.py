"""検証済みの既存stereo cacheから、方向付き分岐の未公開番組候補を保存する。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile
from typing import Callable

import numpy as np
import soundfile as sf
from mutagen.flac import FLAC

from lss.bgm.activity import discover_stem_energy_share_files, load_stem_energy_share_recipe
from lss.bgm.candidate_store import CandidateOrigin, materialize_candidate
from lss.bgm.cue_routing import CueRoutingRecipe, route_cues_stereo
from lss.bgm.routing import DisagreementRoutingRecipe
from lss.core.audio_timeline import probe_audio_timeline
from lss.core.session import _write_json_atomic
from lss.harness.fetch import file_sha256
from lss.harness.tsv_io import load_tsv
from lss.output.writer import read_source_tags


ROOT = Path(__file__).resolve().parents[2]
RENDERING_RECEIPT_POLICY = "canonical_payload_sha256_v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def rendering_digest(rendered: dict) -> str:
    payload = {key: value for key, value in rendered.items() if key != "payload_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def verify_rendering(rendered: dict, plan_hash: str, root: Path, frames: int) -> None:
    if (rendered.get("plan_sha256") != plan_hash
            or rendered.get("receipt_policy") != RENDERING_RECEIPT_POLICY
            or rendered.get("payload_sha256") != rendering_digest(rendered)):
        raise ValueError("再開するrenderingの構成または区間の完了記録が一致しません")
    if Path(rendered["signal"]["path"]).resolve() != (root / "routed.wav").resolve():
        raise ValueError("再開するrenderingの信号が管理範囲外です")
    info = sf.info(rendered["signal"]["path"])
    if info.samplerate != 44100 or info.channels != 2 or info.frames != frames:
        raise ValueError("再開するrenderingの時間軸が一致しません")
    previous_end = 0
    if not isinstance(rendered["cues"], list):
        raise ValueError("再開するrenderingの区間記録が不正です")
    for cue in rendered["cues"]:
        start, end = cue["start_frame"], cue["end_frame"]
        if (type(start) is not int or type(end) is not int or start < previous_end
                or end <= start or end > frames or type(cue["use_music"]) is not bool):
            raise ValueError("再開するrenderingの区間が時間軸と一致しません")
        previous_end = end
    verify([rendered["signal"]])


def verify(records: list[dict]) -> None:
    for item in records:
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"再利用するsource/cacheの内容が記録と一致しません: {item['path']}")


def read_audio(path: Path, *, restore_headroom: bool = False) -> np.ndarray:
    values, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != 44100 or values.shape[1] != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("再利用する音声は有限な44.1kHz stereoである必要があります")
    if restore_headroom:
        shift = int(FLAC(path)["LSS_OUTPUT_HEADROOM_SHIFT"][0])
        values *= 2.0 ** shift
    return values.T


def alignment_observations(prepared: dict, instrumental: dict, vocals: dict) -> list[dict]:
    """sampleずれを補正せず検出する。sum一致を意味的品質の証明には使わない。"""
    paths = [Path(item["path"]) for item in (prepared, instrumental, vocals)]
    infos = [sf.info(path) for path in paths]
    if any(info.samplerate != 44100 or info.channels != 2 or info.frames != infos[0].frames for info in infos):
        raise ValueError("復号入力と相補stemの時間軸が一致しません")
    frames = infos[0].frames
    width = min(44100, frames)
    starts = sorted({int((frames - width) * fraction) for fraction in (0, 0.25, 0.5, 0.75, 1)})
    gains = [1.0] + [2.0 ** int(FLAC(path)["LSS_OUTPUT_HEADROOM_SHIFT"][0]) for path in paths[1:]]
    observations = []
    for start in starts:
        arrays = [sf.read(path, start=start, frames=width, dtype="float64", always_2d=True)[0] * gain
                  for path, gain in zip(paths, gains)]
        raw, reconstructed = arrays[0], arrays[1] + arrays[2]
        if not np.isfinite(raw).all() or not np.isfinite(reconstructed).all():
            raise ValueError("alignment検査の音声が有限ではありません")
        raw, reconstructed = raw - raw.mean(axis=0), reconstructed - reconstructed.mean(axis=0)
        if np.sum(raw ** 2) == 0 or np.sum(reconstructed ** 2) == 0:
            continue
        maximum_lag = min(32, max(0, (width - 1) // 4))
        correlations = []
        for lag in range(-maximum_lag, maximum_lag + 1):
            left = raw[max(0, lag):min(width, width + lag)]
            right = reconstructed[max(0, -lag):min(width, width - lag)]
            denominator = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
            correlations.append(float(np.sum(left * right) / denominator) if denominator > 0 else 0.0)
        best = int(np.argmax(correlations)) - maximum_lag
        zero = correlations[maximum_lag]
        observations.append({"start_frame": start, "frames": width, "best_lag_frames": best,
                             "zero_lag_correlation": zero, "maximum_lag_frames": maximum_lag})
        if best != 0 or zero < 0.99:
            raise ValueError("renderer間の復号入力に時間ずれまたは不一致があります")
    if not observations:
        raise ValueError("renderer間のalignmentを確認できる非無音区間がありません")
    return observations


def queue_observation(queue_path: Path, queue: dict, source: dict) -> tuple[dict, dict]:
    """現行キューの結果を完了台帳の出力SHAと照合し、観測として読む。"""
    ledger_path = queue_path.parent / "_state" / "completed_recipes.tsv"
    _, completed = load_tsv(ledger_path)
    for item in reversed(queue["items"]):
        result = item.get("result") or {}
        if (item.get("status") not in {"completed", "skipped"}
                or result.get("input_sha256") != source["sha256"]
                or (result.get("runtime_revision") or {}).get("device") != "cuda:0"):
            continue
        rows = [row for row in completed if row["input_sha256"] == source["sha256"]
                and row["recipe_id"] == result["recipe_id"]]
        if not rows:
            raise ValueError("キュー結果に対応する完了台帳の内容SHAがありません")
        expected = {record["path"]: record["sha256"] for record in json.loads(rows[-1]["outputs"])}
        records = []
        for raw_path in result["outputs"]:
            if raw_path not in expected:
                raise ValueError("キュー成果物が完了台帳に記録されていません")
            path = Path(raw_path)
            tags = read_source_tags(path)
            if tags.get("LSS_INPUT_SHA256") != source["sha256"] or tags.get("LSS_RECIPE_ID") != result["recipe_id"]:
                raise ValueError("キュー成果物の単独来歴が結果と一致しません")
            info = sf.info(path)
            records.append({"path": raw_path, "sha256": expected[raw_path], "stem": tags["LSS_STEM"],
                            "frames": info.frames, "sample_rate": info.samplerate, "channels": info.channels})
        return {"status": "completed_unannotated_episode_only_observation", "source": source,
                "separation": {**result, "artifacts": records},
                "cache_origin": {"kind": "completed_queue_item", "item_id": item["id"]}}, artifact(ledger_path)
    raise ValueError("同じDevelopment原本のCUDA完了結果がキューにありません")


def freeze(instrumental_run_path: Path, music_run_path: Path, activity_path: Path,
           output: Path, *, source_id: str) -> Path:
    instrumental_run, music_run = read_json(instrumental_run_path), read_json(music_run_path)
    completion_artifact = None
    if instrumental_run.get("version") == 1 and isinstance(instrumental_run.get("items"), list):
        instrumental_run, completion_artifact = queue_observation(
            instrumental_run_path, instrumental_run, music_run["source"])
    if (instrumental_run.get("status") != "completed_unannotated_episode_only_observation"
            or instrumental_run["source"].get("corpus_role") != "development"
            or (instrumental_run["separation"].get("runtime_revision") or {}).get("device") != "cuda:0"):
        raise ValueError("Development番組の完了したCUDA伴奏cacheが必要です")
    if (music_run.get("status") != "PASSED_REPRESENTATIVE_HARD_SCREEN"
            or music_run.get("hard_failure_count") != 0
            or music_run["runtime"].get("model_device") != "cuda:0"):
        raise ValueError("ハード検査を通ったCUDA Music cacheが必要です")
    source = instrumental_run["source"]
    if source["sha256"] != music_run["source"]["sha256"] or not source_id.strip():
        raise ValueError("両rendererのsourceが一致しません")
    by_stem = {item["stem"]: item for item in instrumental_run["separation"]["artifacts"]}
    if len(by_stem) != len(instrumental_run["separation"]["artifacts"]) or "Instrumental" not in by_stem:
        raise ValueError("伴奏の一意なcacheが必要です")
    if set(music_run["outputs"]) != {"music", "speech", "effects"}:
        raise ValueError("Music/Speech/Effectsの全cacheが必要です")
    records = [artifact(path) for path in (instrumental_run_path, music_run_path, activity_path)]
    records += [source, music_run["plan"], music_run["prepared_input"],
                by_stem["Instrumental"], *music_run["outputs"].values()]
    if "Vocals" in by_stem:
        records.append(by_stem["Vocals"])
    if completion_artifact is not None:
        records.append(completion_artifact)
    verify(records)
    music_plan = read_json(Path(music_run["plan"]["path"]))
    if music_plan["source"].get("corpus_role") != "development" or music_plan["source"]["sha256"] != source["sha256"]:
        raise ValueError("Music cacheの固定構成が同じDevelopment原本ではありません")
    frames = by_stem["Instrumental"]["frames"]
    for item in (music_run["prepared_input"], *music_run["outputs"].values(), *by_stem.values()):
        info = sf.info(item["path"])
        if (info.samplerate != 44100 or info.channels != 2 or info.frames != frames
                or item["frames"] != frames or item["sample_rate"] != 44100 or item["channels"] != 2):
            raise ValueError("全cacheの形式と記録した時間軸が一致しません")
    alignment = (alignment_observations(music_run["prepared_input"], by_stem["Instrumental"], by_stem["Vocals"])
                 if "Vocals" in by_stem else [])
    activity = load_stem_energy_share_recipe(activity_path)
    route = CueRoutingRecipe(context_frames=22050, require_energy_increase=True)
    timeline = probe_audio_timeline(Path(source["path"])).to_dict()
    implementations = {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT / "lss/bgm/cue_routing.py", ROOT / "lss/bgm/routing.py",
        ROOT / "lss/bgm/activity.py", ROOT / "lss/bgm/candidate_store.py", ROOT / "lss/core/audio_timeline.py",
    )}
    runtime_revision = {"instrumental": instrumental_run["separation"]["runtime_revision"],
                        "music": {key: music_run["runtime"].get(key) for key in (
                            "model_device", "gpu", "platform", "python", "pytorch_lightning", "torch", "torch_cuda")},
                        "routing": {"python": platform.python_version(), "numpy": np.__version__,
                                    "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
                                    "platform": platform.platform()}}
    runtime_id = "bgm-cue-runtime-v1-" + hashlib.sha256(json.dumps(
        runtime_revision, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    renderer_revision = {"instrumental_recipe": instrumental_run["separation"]["recipe_revision"],
                         "instrumental_runtime_id": instrumental_run["separation"]["runtime_id"],
                         "runtime_revision": runtime_revision, "runtime_id": runtime_id,
                         "music_candidate": music_plan["candidate"], "music_runner_sha256": music_run["runner"]["sha256"],
                         "cue_recipe": route.to_dict(), "activity_recipe": activity.to_dict(),
                         "implementation": implementations,
                         "serialization_policy": "declared_power_of_two_headroom_v1"}
    renderer_id = "bgm-development-cue-renderer-v1-" + hashlib.sha256(json.dumps(
        renderer_revision, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "plan.json"
    _write_json_atomic(plan_path, {
        "schema_version": 1, "status": "FROZEN_BEFORE_ROUTED_CANDIDATES", "promotion_allowed": False,
        "holdout_consumed": False, "source": source, "source_id": source_id,
        "audio_timeline_origin": timeline, "frames": frames, "records": records,
        "instrumental": by_stem["Instrumental"], "stems": music_run["outputs"],
        "cue_recipe": route.to_dict(), "cue_recipe_id": route.recipe_id,
        "activity_recipe": {**artifact(activity_path), "recipe_id": activity.recipe_id},
        "renderer_id": renderer_id, "renderer_revision": renderer_revision,
        "implementation": implementations, "alignment_observations": alignment,
        "rendering_receipt_policy": RENDERING_RECEIPT_POLICY,
        "alignment_status": "zero_sample_lag_observed" if alignment else "unverified_without_complementary_cache",
        "cache_origin": instrumental_run.get("cache_origin", {"kind": "episode_discovery_run"}),
        "runtime_equivalence": "not_asserted_cached_renderers",
        "limits": ["no_semantic_truth_or_reference_used", "candidate_count_is_not_song_count",
                   "source_music_and_singing_and_musical_effects_remain_unresolved",
                   "alignment_is_integrity_evidence_not_separation_quality"],
    })
    return plan_path


def run(plan_path: Path, *, resume: bool = False,
        on_candidate: Callable[[dict], None] | None = None) -> Path:
    plan = read_json(plan_path)
    if plan.get("status") != "FROZEN_BEFORE_ROUTED_CANDIDATES" or plan.get("promotion_allowed") is not False:
        raise ValueError("固定した未公開Development候補の構成が必要です")
    if plan.get("rendering_receipt_policy") != RENDERING_RECEIPT_POLICY or plan.get("holdout_consumed") is not False:
        raise ValueError("区間の完了記録を固定したDevelopment構成が必要です")
    verify(plan["records"])
    for relative, expected in plan["implementation"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError("固定した候補renderer実装が変更されています")
    root = plan_path.parent
    rendering_path = root / "rendering.json"
    if rendering_path.exists() and not resume:
        raise FileExistsError("既存の候補runは明示的なresumeが必要です")
    plan_hash = file_sha256(plan_path)
    recipe_data = dict(plan["cue_recipe"])
    recipe_data["window_recipe"] = DisagreementRoutingRecipe(**recipe_data["window_recipe"])
    route = CueRoutingRecipe(**recipe_data)
    activity = load_stem_energy_share_recipe(Path(plan["activity_recipe"]["path"]))
    if route.recipe_id != plan["cue_recipe_id"] or activity.recipe_id != plan["activity_recipe"]["recipe_id"]:
        raise ValueError("固定したRecipeの同一性が一致しません")
    if resume and rendering_path.exists():
        rendered = read_json(rendering_path)
        verify_rendering(rendered, plan_hash, root, plan["frames"])
    else:
        paths = {role.title(): Path(item["path"]) for role, item in plan["stems"].items()}
        discovery = discover_stem_energy_share_files(paths, activity)
        intervals = [(candidate.start_frame, candidate.end_frame) for candidate in discovery.candidates]
        parent = read_audio(Path(plan["instrumental"]["path"]), restore_headroom=True)
        stems = {role: read_audio(path) for role, path in paths.items()}
        result = route_cues_stereo(parent, stems, intervals, route)
        if result.audio.shape != (2, plan["frames"]):
            raise ValueError("候補rendererが元のtimelineを保持していません")
        signal_path = root / "routed.wav"
        temporary = None
        try:
            with NamedTemporaryFile(dir=root, prefix=".routed-", suffix=".wav", delete=False) as stream:
                temporary = Path(stream.name)
            sf.write(temporary, result.audio.T, 44100, subtype="DOUBLE")
            temporary.replace(signal_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        rendered = {"plan_sha256": plan_hash, "signal": artifact(signal_path), "cues": list(result.cues),
                    "receipt_policy": RENDERING_RECEIPT_POLICY}
        rendered["payload_sha256"] = rendering_digest(rendered)
        _write_json_atomic(rendering_path, rendered)
    rendering_hash = file_sha256(rendering_path)
    report = root / "run.json"
    if report.exists():
        completed = read_json(report)
        if completed.get("plan_sha256") != plan_hash or completed.get("rendering") != rendered:
            raise ValueError("完了済みの候補と再開する区間記録が一致しません")
    candidates = []
    timeline = plan["audio_timeline_origin"]
    for cue in rendered["cues"]:
        origin = CandidateOrigin(plan["source"]["sha256"], rendered["signal"]["sha256"],
            plan["renderer_id"], plan["renderer_revision"]["runtime_id"], activity.recipe_id,
            cue["start_frame"], cue["end_frame"], 44100, timeline["offset_numerator"], timeline["offset_denominator"],
            serialization_policy="declared_power_of_two_headroom_v1")
        saved = materialize_candidate(Path(rendered["signal"]["path"]), root / "candidates" / f"{origin.candidate_id}.flac",
            origin=origin, source_id=plan["source_id"], source_path=Path(plan["source"]["path"]),
            evidence={"cue": cue, "cue_recipe_id": route.recipe_id, "renderer_revision": plan["renderer_revision"],
                      "plan_sha256": plan_hash, "semantic_status": "unresolved_music_like"})
        candidates.append({"origin": origin.to_dict(), "cue": cue, "asset": saved})
        if on_candidate is not None:
            on_candidate(saved)
    verify(plan["records"] + [rendered["signal"]])
    if file_sha256(rendering_path) != rendering_hash:
        raise ValueError("候補保存中に区間の完了記録が変更されています")
    if file_sha256(plan_path) != plan_hash:
        raise ValueError("候補保存中に固定構成が変更されています")
    for relative, expected in plan["implementation"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError("候補保存中に固定実装が変更されています")
    _write_json_atomic(report, {
        "schema_version": 1, "status": "completed_development_routed_episode_candidates",
        "promotion_allowed": False, "holdout_consumed": False, "publication_state": "not_published",
        "source": plan["source"], "plan_sha256": plan_hash, "renderer_id": plan["renderer_id"],
        "source_hashes_after_verified": True, "rendering": rendered, "candidates": candidates,
        "outcome": "unresolved_candidates" if candidates else "unresolved_no_activity_candidates",
        "semantic_ground_truth": None, "metric_status": "not_scored_without_ground_truth",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "soundfile": sf.__version__},
    })
    return report
