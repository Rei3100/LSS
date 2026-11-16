"""Per-part activity envelopes of one cue, for `lss.metadata.version_parts.decide`.

For a cue FLAC this separates DRUMS, BASS, VOCALS (and OTHER as a witness) with the SCNet XL IHF 4-stem
model through the existing sound-plan code path (`lss.sound.extract.Workspace`, `plan.stems`), and
LEAD_VOCALS with the existing karaoke model (`plan.lead`) only when VOCALS is active. Each envelope is the
frame RMS in dB relative to the cue's own mix RMS (0 dB = the whole mix), hop `HOP_SECONDS`, no overlap.

Results are cached under ``<state>/metadata-library/part-envelopes/<pcm_sha256>-<model_sha16>-<hop>.npz``
(PCM identity from `lss.core.audio_identity`), so a rerun or a second version never re-separates.

OTHER is a witness only: `decide_tags` removes any `OTHER_VER` tag.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import soundfile as sf

from lss.metadata import version_parts

HOP_SECONDS = 0.25
FLOOR_DB = -200.0
STEM_PARTS = {"DRUMS": "drums", "BASS": "bass", "VOCALS": "vocals", "OTHER": "other"}
LEAD = "LEAD_VOCALS"
WITNESS_ONLY = frozenset({"OTHER"})
DIRECTORY = ("metadata-library", "part-envelopes")


class Separator(Protocol):
    """What the envelope code needs from a separation backend (a stub in unit tests)."""
    stem_sha16: str
    lead_sha16: str

    def stems(self, path: Path) -> tuple[dict[str, np.ndarray], int]:
        """lower-case stem name (drums/bass/vocals/other) -> (frames, channels) float array, and the rate."""

    def lead(self, path: Path) -> tuple[np.ndarray, int]:
        """Lead vocals (frames, channels) and the rate."""


def frame_db(stem: np.ndarray, mix_rms: float, rate: int, hop_seconds: float = HOP_SECONDS) -> np.ndarray:
    """Frame RMS of `stem` (all channels pooled) in dB relative to `mix_rms`; silence -> FLOOR_DB."""
    x = np.asarray(stem, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    hop = max(1, int(round(hop_seconds * rate)))
    frames = int(np.ceil(x.shape[0] / hop)) if x.shape[0] else 0
    power = np.square(x).mean(axis=1)
    out = np.full(frames, FLOOR_DB)
    if mix_rms <= 0.0:
        return out
    for i in range(frames):
        rms = float(np.sqrt(power[i * hop:(i + 1) * hop].mean()))
        if rms > 0.0:
            out[i] = max(FLOOR_DB, 20.0 * np.log10(rms / mix_rms))
    return out


def mix_rms(mix: np.ndarray) -> float:
    x = np.asarray(mix, dtype=np.float64)
    return float(np.sqrt(np.square(x).mean())) if x.size else 0.0


def _cache_path(state: Path, pcm_sha256: str, model_sha16: str, hop: float) -> Path:
    return Path(state).joinpath(*DIRECTORY) / f"{pcm_sha256}-{model_sha16}-{hop:g}.npz"


def _load(path: Path, frames: int) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as f:
            data = {k: np.asarray(f[k], dtype=np.float64) for k in f.files if not k.startswith("_")}
    except (OSError, ValueError):
        return None
    if not data or any(v.size != frames for v in data.values()):
        return None
    return data


def _save(path: Path, envelopes: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **{k: v.astype(np.float32) for k, v in envelopes.items()})
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _quantised(env: np.ndarray) -> np.ndarray:
    return env.astype(np.float32).astype(np.float64)


def compute(path: Path, state: Path, separator: Separator, *, hop_seconds: float = HOP_SECONDS,
            identity: dict | None = None, with_lead: bool = True) -> dict[str, np.ndarray]:
    """Envelopes {DRUMS, BASS, VOCALS, OTHER[, LEAD_VOCALS]} for the cue, from the cache when present.

    LEAD_VOCALS is computed (and cached into a second file keyed by the karaoke model) only when VOCALS is
    active, so instrumental cues never pay for the karaoke pass.
    """
    from lss.core.audio_identity import audio_identity

    path = Path(path)
    identity = identity or audio_identity(path)
    pcm = identity["pcm_sha256"]
    rate = int(identity["samplerate"])
    hop = max(1, int(round(hop_seconds * rate)))
    frames = int(np.ceil(int(identity["frames"]) / hop))
    stem_file = _cache_path(state, pcm, separator.stem_sha16, hop_seconds)
    envelopes = _load(stem_file, frames)
    if envelopes is None or any(p not in envelopes for p in STEM_PARTS):
        mix, mix_rate = sf.read(str(path), dtype="float64", always_2d=True)
        reference = mix_rms(mix)
        stems, stem_rate = separator.stems(path)
        envelopes = {part: frame_db(stems[name], reference, stem_rate, hop_seconds)
                     for part, name in STEM_PARTS.items()}
        envelopes = {k: _quantised(v) for k, v in envelopes.items()}
        _save(stem_file, envelopes)
    result = dict(envelopes)
    if with_lead and version_parts._active(version_parts._stats(result["VOCALS"])):
        lead_file = _cache_path(state, pcm, f"{separator.stem_sha16[:8]}{separator.lead_sha16[:8]}", hop_seconds)
        lead = _load(lead_file, frames)
        if lead is None or LEAD not in lead:
            mix, _ = sf.read(str(path), dtype="float64", always_2d=True)
            values, lead_rate = separator.lead(path)
            lead = {LEAD: _quantised(frame_db(values, mix_rms(mix), lead_rate, hop_seconds))}
            _save(lead_file, lead)
        result[LEAD] = lead[LEAD]
    return result


def decide_tags(cue: dict[str, np.ndarray], other: dict[str, np.ndarray] | None = None,
                alignment: version_parts.Alignment | None = None,
                *, hop_seconds: float = HOP_SECONDS) -> tuple[dict, dict]:
    """`version_parts.decide` with OTHER kept as a witness (its `OTHER_VER` tag is never written)."""
    tags, evidence = version_parts.decide(cue, other, alignment, hop_seconds=hop_seconds)
    for part in WITNESS_ONLY:
        tags.pop(version_parts.tag_name(part) or "", None)
    return tags, evidence


class WorkspaceSeparator:
    """The production backend: `lss.sound.extract.Workspace` (SCNet XL IHF 4-stem, becruily karaoke).

    Must run under the GPU lease (production: the worker holds it; development: scripts/gpu_run.py).
    Models come from the existing model store; nothing is downloaded beyond what `cached_asset` already does.
    """

    def __init__(self, scratch: Path, *, device: str = "cuda:0",
                 progress: Callable[[dict], None] | None = None):
        from lss.sound import extract

        self.stem_sha16 = extract.SCNET["sha256"][:16]
        self.lead_sha16 = extract.KARAOKE["sha256"][:16]
        self._scratch = Path(scratch)
        self._device = device
        self._progress = progress
        self._ws = None
        self._ws_path: Path | None = None

    def _workspace(self, path: Path):
        from lss.sound import extract

        if self._ws is None or self._ws_path != Path(path):
            tag = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
            self._ws = extract.Workspace(Path(path), self._scratch / tag, progress=self._progress,
                                         device=self._device)
            self._ws_path = Path(path)
        return self._ws

    def stems(self, path: Path):
        from lss.sound import plan

        ws = self._workspace(path)
        return {name: ws.audio(plan.stems(name, plan.SOURCE)) for name in STEM_PARTS.values()}, ws.rate

    def lead(self, path: Path):
        from lss.sound import plan

        ws = self._workspace(path)
        return ws.audio(plan.lead(plan.stems("vocals", plan.SOURCE))), ws.rate
