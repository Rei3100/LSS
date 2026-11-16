"""固定BandIt adapterを再利用し、短い入力の反射境界だけを補う。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.bgm.provider_padding import padding_revision, reflect_pad_last_axis  # noqa: E402
from lss.core import gpu_lease  # noqa: E402
from scripts.infer_bgm_bandit import main, sha256  # noqa: E402


def run():
    if len(sys.argv) != 3:
        raise ValueError("固定planと新規出力先が必要です")
    plan_path, output = Path(sys.argv[1]), Path(sys.argv[2])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    observed = {"adapter_sha256": sha256(Path(__file__)),
                "implementation_sha256": sha256(ROOT / "lss/bgm/provider_padding.py"),
                "policy": padding_revision()}
    if plan.get("short_input_padding") != observed:
        raise ValueError("短い入力の境界処理が固定planと一致しません")
    os.environ["PROJECT_ROOT"] = plan["provider"]["root"]
    sys.path.insert(0, plan["provider"]["root"])
    import torch.nn.functional as functional
    from core.utils.audio import BaseFader

    original = BaseFader.prepare

    def prepare(self, audio):
        if not self.fade_edge_frames or max(self.edge_frame_pad_sizes) < audio.shape[-1]:
            return original(self, audio)
        # fader自身が後でpaddingを除くため、元のframesと各sample位置がそのまま戻る。
        padded = reflect_pad_last_axis(audio, tuple(self.edge_frame_pad_sizes))
        length = padded.shape[-1]
        chunks = max(1, (max(0, length - self.chunk_size) + self.hop_size - 1) // self.hop_size + 1)
        total = (chunks - 1) * self.hop_size + self.chunk_size
        return functional.pad(padded, (0, total - length)), chunks

    BaseFader.prepare = prepare
    try:
        with gpu_lease.acquire():
            result = main()
    finally:
        BaseFader.prepare = original
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if observed != {"adapter_sha256": sha256(Path(__file__)),
                    "implementation_sha256": sha256(ROOT / "lss/bgm/provider_padding.py"),
                    "policy": padding_revision()}:
        raise ValueError("推論途中に短い入力の境界処理が変更されています")
    receipt["short_input_padding"] = observed
    # 既存mainの完了記録へ、このwrapperが使った境界処理のidentityを追加する。
    from lss.core.session import _write_json_atomic
    _write_json_atomic(receipt_path, receipt)
    return result


if __name__ == "__main__":
    raise SystemExit(run())
