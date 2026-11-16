"""公開4stem重みを固定DevelopmentケースでCUDA実行し、3stem Musicと比較する。"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root

import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
from lss.bgm.provider_padding import reflect_pad_last_axis
from lss.core import gpu_lease
from lss.harness.fetch import file_sha256
from lss.validation import facing_music, reference_residual_episode as guarded, stereo_quality
from lss.validation.bandit_inference_cache import InferenceCache, exclusive_output, validate_plan, verify_source
from scripts.probe_bgm_reference_alignment import GATES, failures

CHECKPOINT_SHA256 = "adfd67196defd0ba9bd02a3d319c398172a56a8febc7fe3ff359730453f960c5"
QUERY_REVISION = "79ed5bb75e5c3a40cd319d9d990cee913fc65c26"
RENDERERS = {"instrumental": ["music_instrumental"],
             "instrumental_and_singing": ["music_instrumental", "music_vocals"]}
FADER = {"window_type": "hann", "chunk_size_second": 6.0,
         "hop_size_second": 0.5, "fs": 44100, "batch_size": 4}


def read(path):
    return json.loads(Path(path).read_bytes())


def art(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def implementation():
    return {str(p.resolve()): file_sha256(p) for p in (
        Path(__file__), ROOT / "lss/validation/facing_music.py",
        ROOT / "lss/validation/bandit_inference_cache.py", ROOT / "lss/bgm/provider_padding.py",
        ROOT / "lss/core/gpu_lease.py", ROOT / "lss/validation/stereo_quality.py",
        ROOT / "lss/validation/stereo_comparison.py", ROOT / "scripts/probe_bgm_reference_alignment.py",
    )}


def freeze(parent_path, output):
    parent = read(parent_path)
    cases = copy.deepcopy(parent["cases"])
    if len(cases) != 33:
        raise ValueError("既存固定33ケースが必要です")
    provider = ROOT / ".providers/facing-the-music/query-bandit"
    checkpoint = ROOT / ".providers/facing-the-music/bandit-split.ckpt"
    if subprocess.check_output(["git", "-C", str(provider), "rev-parse", "HEAD"], text=True).strip() != QUERY_REVISION:
        raise ValueError("固定する公開実装revisionが一致しません")
    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("公式checksum確認済みの4stem重みが必要です")
    fader_path = ROOT / ".providers/bandit/core/utils/audio.py"
    files = {str(p.resolve()): file_sha256(p) for p in (provider / "core").rglob("*.py")}
    files.update({str(p.resolve()): file_sha256(p) for p in (provider / "LICENSE", fader_path, checkpoint)})
    files.update(implementation())
    files[str(parent_path.resolve())] = file_sha256(parent_path)
    for name in ("layout-report-r4.json", "connection-review.json"):
        evidence = dev_state_root(ROOT) / "validation/bgm-facing-music-controls-r1" / name
        files[str(evidence)] = file_sha256(evidence)
    for case in cases:
        verify_source(case)
        for item in (case["reference"], case["stems"]["music"]):
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("既存の参照・3stem Musicが変更されています")
            files[item["path"]] = item["sha256"]
        case["baseline_music"] = case["stems"]["music"]
        case["stems"] = {}
        case.pop("baseline_quality", None)
    plan = {"schema_version": 1, "status": "FROZEN_BEFORE_RESULTS", "promotion_allowed": False,
        "holdout_consumed": False, "sample_rate": 44100, "output_roles": list(facing_music.ROLES),
        "cases": cases, "provider_root": str(provider), "provider_revision": QUERY_REVISION,
        "checkpoint": art(checkpoint), "provider_and_inputs": files,
        "model_config": facing_music.MODEL_CONFIG, "fader": FADER, "fader_source": art(fader_path),
        "renderers": RENDERERS, "gates": GATES,
        "feature_layout_evidence": {
            "previous_adapter_commit": "4fcaf36",
            "diagnostic": art(dev_state_root(ROOT) / "validation/bgm-facing-music-controls-r1/layout-report-r4.json"),
            "review": art(dev_state_root(ROOT) / "validation/bgm-facing-music-controls-r1/connection-review.json"),
            "scope": "公開MIT層の入力特徴順序だけを明示的に修正。別providerの層は取り込まない。"},
        "rights": {"weights": "CC-BY-NC-4.0", "code": "MIT",
            "weights_source": "https://zenodo.org/records/13327983",
            "weights_official_md5": "e9f9054a452cb86b30a6f705266dbd7c",
            "scope": "非商用Development研究。製品への採用・配布を判断しない。"},
        "limits": ["各rendererを全ケースへ同じ方法で適用。参照から勝者やstemを選ばない。",
            "同じ音声の学習データ重複は未確認。意味的BGM精度やtarget-domainへ一般化しない。",
            "配布重みに実行設定はない。公開層へ演算名を対応づけ、全重みと固定bufferを厳密照合する。",
            "今回の特徴順序・固定STFT・チャンク・独立左右処理を記録し、公式評価runtimeと同一とは主張しない。",
            "原本と既存候補を保持。外部音声送信・Holdout・CPU推論fallback・正式公開なし。",
            "入力が完全なゼロなら全4stemをゼロとして保存。その他はCUDAモデルで処理する。"]}
    validate_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "plan.json"
    guarded._write_record(path, plan, lambda: verify(plan))
    return path


def verify(plan):
    if plan["model_config"] != facing_music.MODEL_CONFIG or plan["fader"] != FADER:
        raise ValueError("固定モデル構成・チャンク構成が変更されています")
    for path, digest in plan["provider_and_inputs"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"固定provider・実装・入力の変更: {path}")
    for case in plan["cases"]:
        verify_source(case)


IDENTITY_KEYS = ("output_roles", "provider_revision", "checkpoint",
                 "model_config", "fader", "provider_and_inputs")


def bound_identity(plan, plan_path, output):
    """既存出力のidentityが、この固定planと同じ条件であることを示す。"""
    identity = read(Path(output) / "identity.json")
    if (identity.get("plan_sha256") != file_sha256(plan_path)
            or identity.get("execution_device") != "cuda:0" or identity.get("cpu_fallback") is not False
            or any(identity.get(key) != plan[key] for key in IDENTITY_KEYS)):
        raise ValueError("この固定planのCUDA推論出力ではありません")
    return identity


def pending_cases(cache, plan):
    """再利用できない出力が残ったまま再開せず、隔離の入口へ案内する。"""
    states = {row["case_id"]: row for row in cache.inspect()}
    invalid = [{"case_id": row["case_id"], "state": row["state"], "reason": row["reason"]}
               for row in states.values() if row["state"] not in ("REUSABLE", "MISSING")]
    if invalid:
        raise ValueError("再利用できない推論出力があります。recoverで隔離してから再開してください: "
                         + json.dumps(invalid, ensure_ascii=False))
    return [case for case in plan["cases"] if states[case["case_id"]]["state"] == "MISSING"]


def recover(plan_path, output, quarantine, reason):
    """不正になった推論出力だけを隔離し、残りの再推論をrun --resumeへ渡す。"""
    plan_path, output, quarantine = Path(plan_path), Path(output), Path(quarantine)
    plan = read(plan_path)
    validate_plan(plan)
    digest = file_sha256(plan_path)
    verify(plan)
    identity = bound_identity(plan, plan_path, output)
    with exclusive_output(output, resume=True):
        cache = InferenceCache(output, plan, identity, resume=True)
        record = cache.quarantine_invalid(quarantine, reason=reason)
    if file_sha256(plan_path) != digest:
        raise ValueError("固定planが変更されています")
    return json.dumps({"status": record["status"], "quarantine": str(quarantine.resolve()),
        "quarantined_cases": [row["case_id"] for row in record["cases"]],
        "reinference_required_cases": record["reinference_required_cases"],
        "reusable_cases": record["reusable_cases"], "deleted": []}, ensure_ascii=False)


def run(plan_path, output, resume=False, stop_after=None):
    import torch
    import torchaudio
    plan = read(plan_path)
    validate_plan(plan)
    digest = file_sha256(plan_path)
    def check():
        if file_sha256(plan_path) != digest:
            raise ValueError("固定planが変更されています")
        verify(plan)
    check()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA専用です。CPUへfallbackしません")
    identity = {"plan_sha256": digest, "execution_device": "cuda:0", "cpu_fallback": False,
        "output_roles": plan["output_roles"], "provider_revision": plan["provider_revision"],
        "checkpoint": plan["checkpoint"], "model_config": plan["model_config"], "fader": plan["fader"],
        "provider_and_inputs": plan["provider_and_inputs"], "runtime": {
            "python": sys.version, "torch": torch.__version__, "torchaudio": torchaudio.__version__,
            "numpy": np.__version__, "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0), "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic": torch.are_deterministic_algorithms_enabled()}}
    with exclusive_output(output, resume=resume):
        cache = InferenceCache(output, plan, identity, resume=resume)
        missing = pending_cases(cache, plan)
        if not missing:
            return cache.complete(verify_identity=check)
        with gpu_lease.acquire(on_wait=lambda: print("CUDA所有を待っています", flush=True)):
            model = facing_music.load_model(Path(plan["provider_root"]), Path(plan["checkpoint"]["path"]))
            model = model.to("cuda:0").eval()
            spec = importlib.util.spec_from_file_location("_lss_facing_fader", plan["fader_source"]["path"])
            fader_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fader_module)
            fader = fader_module.OverlapAddFader(**plan["fader"]).to("cuda:0").eval()
            original_prepare = fader_module.BaseFader.prepare
            def prepare(self, audio):
                if not self.fade_edge_frames or max(self.edge_frame_pad_sizes) < audio.shape[-1]:
                    return original_prepare(self, audio)
                padded = reflect_pad_last_axis(audio, tuple(self.edge_frame_pad_sizes))
                chunks = max(1, (max(0, padded.shape[-1] - self.chunk_size) + self.hop_size - 1) // self.hop_size + 1)
                total = (chunks - 1) * self.hop_size + self.chunk_size
                return torch.nn.functional.pad(padded, (0, total - padded.shape[-1])), chunks
            fader_module.BaseFader.prepare = prepare
            def model_fn(audio):
                if audio.device != torch.device("cuda:0") or any(v.device != audio.device
                     for v in list(model.parameters()) + list(model.buffers())):
                    raise RuntimeError("CUDAモデルと入力が同じdeviceにありません")
                result = model(audio)
                if set(result) != set(plan["output_roles"]) or any(v.device != audio.device for v in result.values()):
                    raise RuntimeError("固定4stemのCUDA出力が必要です")
                return result
            try:
                for index, case in enumerate(missing, 1):
                    check()
                    audio, rate = sf.read(case["source"]["path"], dtype="float32", always_2d=True)
                    if rate != 44100 or not np.isfinite(audio).all():
                        raise ValueError("固定stereo入力が不正です")
                    torch.cuda.reset_peak_memory_stats(0)
                    tensor = torch.from_numpy(audio.T.copy()).unsqueeze(0).to("cuda:0")
                    torch.cuda.synchronize(0)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        result = (fader(tensor, model_fn)["audio"] if np.any(audio)
                                  else {role: torch.zeros_like(tensor) for role in plan["output_roles"]})
                    torch.cuda.synchronize(0)
                    seconds = time.perf_counter() - started
                    values = {role: value[0].cpu().numpy().T for role, value in result.items()}
                    record = cache.save(case, values, seconds=seconds,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(0), before_publish=check)
                    print(json.dumps({"case_id": case["case_id"], "seconds": seconds,
                                      "peak_cuda_bytes": record["peak_cuda_bytes"]}), flush=True)
                    if stop_after is not None and index >= stop_after:
                        print("計画的中断。検証済みケースを保持しました。", flush=True)
                        return None
            finally:
                fader_module.BaseFader.prepare = original_prepare
        return cache.complete(verify_identity=check)


def evaluate(plan_path, receipt_path):
    plan, receipt = read(plan_path), read(receipt_path)
    identity = read(receipt_path.parent / "identity.json")
    verify(plan)
    if (receipt["status"] != "COMPLETED_CUDA_WITHOUT_FALLBACK"
            or receipt["plan_sha256"] != file_sha256(plan_path)
            or receipt["output_roles"] != plan["output_roles"]):
        raise ValueError("固定4stemの最終CUDA完了記録が必要です")
    if (identity["provider_and_inputs"] != plan["provider_and_inputs"]
            or identity["model_config"] != plan["model_config"]
            or identity["fader"] != plan["fader"]
            or any(receipt.get(key) != value for key, value in identity.items())):
        raise ValueError("実行同一性と最終完了記録が一致しません")
    rows = {c["case_id"]: c for c in receipt["cases"]}
    if len(receipt["cases"]) != len(plan["cases"]) or set(rows) != {c["case_id"] for c in plan["cases"]}:
        raise ValueError("全固定ケースの一意な完了記録が必要です")
    records = []
    for case in plan["cases"]:
        row = rows[case["case_id"]]
        metadata = read(receipt_path.parent / case["case_id"] / "case.json")
        if (metadata["identity"] != identity
                or row != {key: value for key, value in metadata.items() if key != "identity"}
                or set(row["stems"]) != set(plan["output_roles"])):
            raise ValueError("固定4stemのケース記録と最終完了記録が一致しません")
        if row["source_sha256"] != case["source"]["sha256"] or not row["source_hash_after_verified"]:
            raise ValueError("固定入力の事後検証記録が不一致です")
        signals = {}
        for role, item in row["stems"].items():
            if (Path(item["path"]).resolve() != (receipt_path.parent / case["case_id"] / f"{role}.wav").resolve()
                    or file_sha256(Path(item["path"])) != item["sha256"]):
                raise ValueError("推論stemが変更されています")
            audio, rate = sf.read(item["path"], dtype="float64", always_2d=True)
            if rate != 44100 or audio.shape != (case["source"]["frames"], 2) or not np.isfinite(audio).all():
                raise ValueError("推論stemの左右・有限値・時間軸が不正です")
            signals[role] = audio.T
        baseline, rate = sf.read(case["baseline_music"]["path"], dtype="float64", always_2d=True)
        target, target_rate = sf.read(case["reference"]["path"], dtype="float64", always_2d=True)
        if rate != target_rate or target.shape != baseline.shape:
            raise ValueError("比較する参照と既存Musicの時間軸が不一致です")
        baseline, target = baseline.T, target.T
        absent = not np.any(target)
        before = None if absent else stereo_quality.stereo_quality(baseline, target)
        for name, roles in plan["renderers"].items():
            estimate = sum(signals[role] for role in roles)
            after = None if absent else stereo_quality.stereo_quality(estimate, target)
            bad = failures(before, after, absent, float(np.sum(baseline ** 2)), float(np.sum(estimate ** 2)))
            gain = None if absent else after["scale_dependent_snr_db"] - before["scale_dependent_snr_db"]
            records.append({"case_id": case["case_id"], "renderer": name, "target_absent": absent,
                "wanted_role_policy": case["wanted_role_policy"], "before": before, "after": after,
                "failures": bad, "snr_gain_db": gain,
                "baseline_energy": float(np.sum(baseline ** 2)), "candidate_energy": float(np.sum(estimate ** 2)),
                "material_improvement": gain is not None and gain >= plan["gates"]["meaningful_improvement_db"]})
    summary = [{"renderer": name,
        "regression_cases": [r["case_id"] for r in records if r["renderer"] == name and r["failures"]],
        "materially_improved_cases": [r["case_id"] for r in records if r["renderer"] == name and r["material_improvement"]],
        "decision": "REJECTED_EXACT_CONFIGURATION" if any(r["failures"] for r in records if r["renderer"] == name)
            else "DEVELOPMENT_SURVIVOR_REQUIRES_REAL_DOMAIN_EVIDENCE"} for name in plan["renderers"]]
    report = {"status": "COMPLETED_FIXED_FOUR_STEM_CUDA_DEVELOPMENT_COMPARISON",
        "plan": art(plan_path), "receipt": art(receipt_path), "case_variant_count": len(records),
        "cases": records, "summary": summary, "inputs_after_verified": True,
        "promotion_allowed": False, "holdout_consumed": False, "winner_selected": False,
        "source_upload_used": False, "limits": plan["limits"]}
    output = plan_path.with_name("report.json")
    def check():
        verify(plan)
        for item in (report["plan"], report["receipt"], *[item for row in rows.values() for item in row["stems"].values()]):
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("比較中に固定plan・完了記録・stemが変更されています")
    guarded._write_record(output, report, check)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--parent-plan", type=Path, default=dev_state_root(ROOT) / "validation/bgm-recording-recovery-controls-r3/plan.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int)
    p = sub.add_parser("recover")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--quarantine-dir", type=Path, required=True)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.parent_plan, args.out_dir)
    elif args.command == "run":
        result = run(args.plan, args.out_dir, args.resume, args.stop_after)
    elif args.command == "recover":
        result = recover(args.plan, args.out_dir, args.quarantine_dir, args.reason)
    else:
        result = evaluate(args.plan, args.receipt)
    print(result, flush=True)


if __name__ == "__main__":
    main()
