"""BGM activity候補をsource/Recipe/実装identityと結び付ける証拠record。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from lss.bgm.activity import ACTIVITY_METHOD, ActivityDiscoveryResult, ActivityRecipe
from lss.validation.bgm_corpus import BgmCase


ACTIVITY_DETECTION_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def activity_implementation_revision(method: str = ACTIVITY_METHOD) -> dict:
    path = Path(__file__).parents[1] / "bgm" / "activity.py"
    return {
        "algorithm": method,
        "artifact_sha256": file_sha256(path),
        "numpy_version": np.__version__,
    }


def build_activity_detection_record(
    case: BgmCase,
    analysis_signal_path: Path,
    recipe: ActivityRecipe,
    result: ActivityDiscoveryResult,
    *,
    analysis_signal_sha256: str | None = None,
) -> dict:
    analysis_signal_path = Path(analysis_signal_path).resolve()
    info = sf.info(analysis_signal_path)
    if info.samplerate != case.sample_rate or info.frames != case.frames:
        raise ValueError(f"case {case.case_id}のanalysis signal layoutがcorpusと一致しません")
    if result.sample_rate != info.samplerate or result.source_frames != info.frames:
        raise ValueError(f"case {case.case_id}のactivity結果layoutがanalysis signalと一致しません")
    if result.recipe_id != recipe.recipe_id:
        raise ValueError(f"case {case.case_id}のactivity Recipe identityが一致しません")
    signal_sha = analysis_signal_sha256 or file_sha256(analysis_signal_path)
    if not _SHA256.fullmatch(signal_sha):
        raise ValueError("analysis signal SHA-256が不正です")
    return {
        "type": "activity_detection",
        "schema_version": ACTIVITY_DETECTION_SCHEMA_VERSION,
        "case_id": case.case_id,
        "corpus_source": {
            "sha256": case.source_sha256,
            "sample_rate": case.sample_rate,
            "frames": case.frames,
        },
        "analysis_signal": {
            "sha256": signal_sha,
            "sample_rate": info.samplerate,
            "frames": info.frames,
        },
        "activity_recipe": {
            "recipe_id": recipe.recipe_id,
            "revision": recipe.to_dict(),
        },
        "activity_implementation": activity_implementation_revision(),
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "start_frame": item.start_frame,
                "end_frame": item.end_frame,
                "semantic_role": "unresolved_music_like",
                "identity_state": "unresolved",
                "cue_identity_id": None,
                "audio_version_id": None,
            }
            for item in result.candidates
        ],
    }
