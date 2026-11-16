"""音量しきい値を使わず、入力の有限値と完全なデジタル無音を確認する。"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


PRESENCE_POLICY_REVISION = 1
SCAN_BLOCK_FRAMES = 1 << 18


class InvalidInputSignalError(ValueError):
    """入力が空、不完全、またはNaN/Infを含み、正しい処理を保証できない。"""


@dataclass(frozen=True)
class SignalPresence:
    frames: int
    sample_rate: int
    channels: int
    exactly_silent: bool

    def zero_result(self) -> dict:
        if not self.exactly_silent:
            raise ValueError("完全無音でない入力を対象不在と判定できません")
        return {
            "reason": "no_wanted_target",
            "validation_result": {
                "validator_id": "exact_digital_silence",
                "validator_revision": str(PRESENCE_POLICY_REVISION),
                "hard_gates_passed": True,
                "evidence": {
                    "checked_frames": self.frames, "sample_rate": self.sample_rate,
                    "channels": self.channels, "all_samples_exactly_zero": True,
                },
            },
        }


def presence_policy_revision() -> dict:
    return {
        "policy_revision": PRESENCE_POLICY_REVISION,
        "finite_input": "all_decoded_samples",
        "target_absence": "all_decoded_samples_exactly_zero",
        "threshold": "none",
    }


def inspect_signal(path: Path) -> SignalPresence:
    with sf.SoundFile(path) as audio:
        if audio.frames <= 0:
            raise InvalidInputSignalError("入力に音声サンプルがありません")
        exactly_silent = True
        checked = 0
        while checked < audio.frames:
            block = audio.read(min(SCAN_BLOCK_FRAMES, audio.frames - checked), dtype="float64", always_2d=True)
            if not len(block) or block.shape[1] != audio.channels:
                raise InvalidInputSignalError("入力の音声サンプルを最後まで読み取れません")
            if not np.isfinite(block).all():
                raise InvalidInputSignalError("入力の音声にNaNまたは無限大が含まれています")
            if np.any(block != 0):
                exactly_silent = False
            checked += len(block)
        return SignalPresence(checked, audio.samplerate, audio.channels, exactly_silent)
