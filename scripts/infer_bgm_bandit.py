"""固定Development planの不足stemをBandIt専用CUDA環境で生成する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan["status"] != "FROZEN_BEFORE_RESULTS" or plan["promotion_allowed"]:
        raise ValueError("未昇格の凍結Development planが必要です")
    provider = plan["provider"]
    provider_root = Path(provider["root"])
    for item in provider["artifacts"].values():
        if sha256(provider_root / item["path"]) != item["sha256"]:
            raise ValueError(f"BandIt artifactが固定identityと一致しません: {item['path']}")
    os.environ["PROJECT_ROOT"] = str(provider_root)
    sys.path.insert(0, str(provider_root))
    import numpy as np
    import soundfile as sf
    import torch
    import core
    from core.utils.audio import OverlapAddFader
    from utils.config import read_nested_yaml

    if not torch.cuda.is_available():
        raise RuntimeError("BGM推論にはCUDAが必要です")
    config = read_nested_yaml(str(provider_root / provider["artifacts"]["experiment"]["path"]))
    inference = read_nested_yaml(str(provider_root / provider["artifacts"]["inference"]["path"]))
    model = core.parse_model_config(config["system"]["model"])
    checkpoint = torch.load(
        provider_root / provider["artifacts"]["checkpoint"]["path"],
        map_location="cpu", weights_only=True,
    )
    state = {
        key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    model = model.to("cuda").eval()
    fader = OverlapAddFader(**inference["fader"]["kwargs"]).to("cuda").eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("silent CPU fallbackを検出しました")
    args.output.mkdir(exist_ok=False, parents=True)
    records = []
    for item in plan["cases"]:
        if item["stems"]:
            continue
        source = Path(item["source"]["path"])
        source_hash = sha256(source)
        if source_hash != item["source"]["sha256"]:
            raise ValueError("固定source identityが変わっています")
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        if rate != plan["sample_rate"] or audio.shape != (item["source"]["frames"], 2):
            raise ValueError("BandIt入力のstereo/timelineが不正です")
        if not np.isfinite(audio).all():
            raise ValueError("BandIt入力に有限でない値があります")
        torch.cuda.reset_peak_memory_stats()
        tensor = torch.from_numpy(audio.T.copy()).unsqueeze(0).to("cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = fader(tensor, lambda chunks: model({"audio": {"mixture": chunks}})[1]["audio"])["audio"]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        case_root = args.output / item["case_id"]
        case_root.mkdir()
        stems = {}
        for role in ("music", "speech", "effects"):
            values = output[role][0].detach().cpu().numpy().T
            if values.shape != audio.shape or not np.isfinite(values).all():
                raise ValueError("BandItが不正なstemを返しました")
            path = case_root / f"{role}.wav"
            sf.write(path, values, rate, subtype="FLOAT")
            stems[role] = {"path": str(path.resolve()), "sha256": sha256(path)}
        if sha256(source) != source_hash:
            raise ValueError("推論中にsourceが変わりました")
        records.append({
            "case_id": item["case_id"], "stems": stems,
            "source_sha256": source_hash, "seconds": seconds,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        })
        print(json.dumps({"case_id": item["case_id"], "seconds": seconds}), flush=True)
        del tensor, output
        torch.cuda.empty_cache()
    receipt = {
        "schema_version": 1, "status": "COMPLETED_CUDA_WITHOUT_FALLBACK",
        "plan_sha256": sha256(args.plan), "provider": provider,
        "adapter_sha256": sha256(Path(__file__)),
        "runtime": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0), "numpy": np.__version__},
        "cases": records,
    }
    with (args.output / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
