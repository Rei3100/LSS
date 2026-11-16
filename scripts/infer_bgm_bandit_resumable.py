"""固定BandIt Development推論を、検証済みケースからCUDA専用で再開する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.bgm.provider_padding import padding_revision, reflect_pad_last_axis  # noqa: E402
from lss.bgm.inference_variants import InferenceVariantRecipe, restore_stereo, transform_stereo  # noqa: E402
from lss.core import gpu_lease  # noqa: E402
from lss.core.session import _write_json_atomic  # noqa: E402
from lss.harness.fetch import file_sha256  # noqa: E402
from lss.validation.bandit_inference_cache import (  # noqa: E402
    InferenceCache, exclusive_output, read_json, validate_plan, verify_source,
)


def implementation() -> dict:
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT / "lss/validation/bandit_inference_cache.py",
        ROOT / "lss/bgm/provider_padding.py", ROOT / "lss/core/gpu_lease.py",
        ROOT / "lss/bgm/inference_variants.py",
    )}


def provider_files(provider: dict) -> dict:
    root = Path(provider["root"])
    files = {item["path"]: item["sha256"] for item in provider["artifacts"].values()}
    # 既存の少数artifactだけでなく、モデルを動かすprovider code/configも固定する。
    for directory in ("core", "utils", "configs", "expt"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}:
                relative = path.relative_to(root).as_posix()
                observed = file_sha256(path)
                if relative in files and files[relative] != observed:
                    raise ValueError("固定provider artifactが変更されています")
                files[relative] = observed
    for relative, expected in files.items():
        if file_sha256(root / relative) != expected:
            raise ValueError("固定provider artifactが変更されています")
    return files


def freeze(parent_path: Path, output: Path, *, case_ids: list[str] | None = None) -> Path:
    parent = read_json(parent_path)
    validate_plan(parent)
    cases = parent["cases"]
    if case_ids is not None:
        selected = set(case_ids)
        if len(selected) != len(case_ids) or not selected.issubset({c["case_id"] for c in cases}):
            raise ValueError("元構成に存在する一意なケースだけを固定します")
        cases = [case for case in cases if case["case_id"] in selected]
    for case in cases:
        verify_source(case)
    plan = {**parent, "cases": cases,
            "parent_plan": {"path": str(parent_path.resolve()), "sha256": file_sha256(parent_path)},
            "adapter_sha256": file_sha256(Path(__file__)), "inference_implementation": implementation(),
            "provider_files": provider_files(parent["provider"]),
            "short_input_padding": {"adapter_sha256": file_sha256(Path(__file__)),
                                    "implementation_sha256": file_sha256(ROOT / "lss/bgm/provider_padding.py"),
                                    "policy": padding_revision()},
            "recovery_policy": "validated_complete_case_only_v1"}
    validate_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "plan.json"
    _write_json_atomic(path, plan)
    return path


def verify_fixed(plan_path: Path, plan: dict, expected_hash: str) -> None:
    if file_sha256(plan_path) != expected_hash or implementation() != plan["inference_implementation"]:
        raise ValueError("推論中に固定plan・adapter実装が変更されています")
    if provider_files(plan["provider"]) != plan["provider_files"]:
        raise ValueError("推論中に固定providerの内容が変更されています")
    padding = {"adapter_sha256": file_sha256(Path(__file__)),
               "implementation_sha256": file_sha256(ROOT / "lss/bgm/provider_padding.py"),
               "policy": padding_revision()}
    if plan["adapter_sha256"] != file_sha256(Path(__file__)) or padding != plan["short_input_padding"]:
        raise ValueError("固定adapter・反射境界のidentityが一致しません")


def run(plan_path: Path, output: Path, *, resume: bool = False, stop_after: int | None = None) -> int:
    plan = read_json(plan_path)
    validate_plan(plan)
    if plan.get("recovery_policy") != "validated_complete_case_only_v1":
        raise ValueError("再開用に固定したDevelopment planが必要です")
    plan_hash = file_sha256(plan_path)
    verify_fixed(plan_path, plan, plan_hash)
    for case in plan["cases"]:
        verify_source(case)
    provider_root = Path(plan["provider"]["root"])
    os.environ["PROJECT_ROOT"] = str(provider_root)
    sys.path.insert(0, str(provider_root))
    import numpy as np
    import soundfile as sf
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDAが必要です。CPUへfallbackしません")
    identity = {
        "plan_sha256": plan_hash, "provider": plan["provider"], "provider_files": plan["provider_files"],
        "adapter_sha256": plan["adapter_sha256"], "inference_implementation": plan["inference_implementation"],
        "short_input_padding": plan["short_input_padding"], "execution_device": "cuda:0", "cpu_fallback": False,
        "runtime": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0), "numpy": np.__version__,
                    "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
                    "cudnn": torch.backends.cudnn.version(), "matmul_precision": torch.get_float32_matmul_precision(),
                    "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
    }
    with exclusive_output(output, resume=resume):
        cache = InferenceCache(output, plan, identity, resume=resume)
        missing = [case for case in plan["cases"] if cache.load(case) is None]
        if not missing:
            print(cache.complete(verify_identity=lambda: verify_fixed(plan_path, plan, plan_hash)), flush=True)
            return 0
        # 検証済みcacheだけで完了できるときはモデルを読み込まず、GPU所有も要求しない。
        with gpu_lease.acquire(on_wait=lambda: print("CUDA deviceの所有を待っています", flush=True)):
            import core
            from core.utils.audio import BaseFader, OverlapAddFader
            from utils.config import read_nested_yaml

            config = read_nested_yaml(str(provider_root / plan["provider"]["artifacts"]["experiment"]["path"]))
            inference = read_nested_yaml(str(provider_root / plan["provider"]["artifacts"]["inference"]["path"]))
            model = core.parse_model_config(config["system"]["model"])
            checkpoint = torch.load(provider_root / plan["provider"]["artifacts"]["checkpoint"]["path"],
                                    map_location="cpu", weights_only=True)
            model.load_state_dict({key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()
                                   if key.startswith("model.")}, strict=True)
            model = model.to("cuda:0").eval()
            fader = OverlapAddFader(**inference["fader"]["kwargs"]).to("cuda:0").eval()
            original_prepare = BaseFader.prepare

            def prepare(self, audio):
                if not self.fade_edge_frames or max(self.edge_frame_pad_sizes) < audio.shape[-1]:
                    return original_prepare(self, audio)
                padded = reflect_pad_last_axis(audio, tuple(self.edge_frame_pad_sizes))
                chunks = max(1, (max(0, padded.shape[-1] - self.chunk_size) + self.hop_size - 1) // self.hop_size + 1)
                total = (chunks - 1) * self.hop_size + self.chunk_size
                return torch.nn.functional.pad(padded, (0, total - padded.shape[-1])), chunks

            def model_fn(chunks):
                if chunks.device != torch.device("cuda:0") or any(
                        value.device != torch.device("cuda:0") for value in list(model.parameters()) + list(model.buffers())):
                    raise RuntimeError("CUDAモデル・入力のdeviceが一致しません")
                values = model({"audio": {"mixture": chunks}})[1]["audio"]
                if any(value.device != torch.device("cuda:0") for value in values.values()):
                    raise RuntimeError("モデルがCUDA以外の出力を返しました")
                return values

            BaseFader.prepare = prepare
            try:
                for index, case in enumerate(missing, 1):
                    verify_fixed(plan_path, plan, plan_hash)
                    verify_source(case)
                    variant = (InferenceVariantRecipe(**case["inference_variant"])
                               if "inference_variant" in case else None)
                    audio, _ = sf.read(case["source"]["path"], dtype="float64" if variant else "float32", always_2d=True)
                    if not np.isfinite(audio).all():
                        raise ValueError("BandIt入力に有限でない値があります")
                    model_input = transform_stereo(audio.T, variant).astype(np.float32) if variant else audio.T.copy()
                    if not np.isfinite(model_input).all():
                        raise ValueError("BandIt入力variantを有限なFLOAT精度で表現できません")
                    torch.cuda.reset_peak_memory_stats(0)
                    tensor = torch.from_numpy(model_input).unsqueeze(0).to("cuda:0")
                    torch.cuda.synchronize(0)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        result = fader(tensor, model_fn)["audio"]
                    torch.cuda.synchronize(0)
                    seconds = time.perf_counter() - started
                    values = {role: (restore_stereo(value[0].detach().cpu().numpy(), len(audio), variant).T
                                     if variant else value[0].detach().cpu().numpy().T)
                              for role, value in result.items()}
                    verify_fixed(plan_path, plan, plan_hash)
                    cache.save(case, values, seconds=seconds, peak_cuda_bytes=torch.cuda.max_memory_allocated(0),
                               before_publish=lambda: verify_fixed(plan_path, plan, plan_hash))
                    print(json.dumps({"case_id": case["case_id"], "seconds": seconds, "case_receipt": "validated"}), flush=True)
                    del tensor, result, values, model_input, audio
                    torch.cuda.empty_cache()
                    if stop_after is not None and index >= stop_after:
                        print("ケース完了後に停止しました。全体の完了記録は作成していません", flush=True)
                        return 75
            finally:
                BaseFader.prepare = original_prepare
        print(cache.complete(verify_identity=lambda: verify_fixed(plan_path, plan, plan_hash)), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("parent", type=Path)
    freeze_parser.add_argument("output", type=Path)
    freeze_parser.add_argument("--case", action="append", dest="cases")
    runner = commands.add_parser("run")
    runner.add_argument("plan", type=Path)
    runner.add_argument("output", type=Path)
    runner.add_argument("--resume", action="store_true")
    runner.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.parent, args.output, case_ids=args.cases))
        return 0
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-afterは正のケース数が必要です")
    return run(args.plan, args.output, resume=args.resume, stop_after=args.stop_after)


if __name__ == "__main__":
    raise SystemExit(main())
