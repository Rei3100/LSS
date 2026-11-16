"""What a score order sheet says about the music itself: rhythm, tempo, instrumentation, mode.

Measured per published cue from its own audio and from the per-second AudioSet probabilities the BGM run already
stored for its Music stem (no new inference). Every reading abstains when it is not reliable:
- tempo: onset-strength autocorrelation; a BPM is stated only when the beat is clear (salience >= 0.3) and no
  octave/3:2 alternative is nearly as strong (controlled stretch/resample truth: 87 % right at that salience,
  80 % without; a learned beat tracker scored 31/60 on the same control and is not used);
- rhythm: how many note onsets per second (octave-free), whether the beat is regular, and whether drums play;
- instrumentation: the strongest coarse families heard (percussion, strings, brass, woodwind, piano, plucked,
  mallet, synth, organ) and whether a voice sings;
- mode/key: NNLS note chroma (`music_description.key`).
These are description data (order-sheet direction, tags, M-number sections), never a scene.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SALIENCE_MIN = 0.3
INSTRUMENT_PRESENT = 0.06


TEMPO_RATE, TEMPO_HOP = 22050, 512


def tempo(mono: np.ndarray, rate: int) -> dict:
    """BPM with its salience, or abstention; also onsets per second (octave-free activity).

    The same estimator the controlled truth was measured with (librosa onset strength at 22.05 kHz, hop 512,
    autocorrelation peaks 50-200 BPM with parabolic interpolation).
    """
    from math import gcd

    import librosa
    from scipy.signal import resample_poly

    if rate != TEMPO_RATE:
        divisor = gcd(TEMPO_RATE, rate)
        mono = resample_poly(mono, TEMPO_RATE // divisor, rate // divisor).astype(np.float32)
    envelope = librosa.onset.onset_strength(y=mono, sr=TEMPO_RATE, hop_length=TEMPO_HOP)
    n = len(envelope)
    if n < 200:
        return {"status": "abstained", "reason": "too_short"}
    fps = TEMPO_RATE / TEMPO_HOP
    peaks = librosa.util.peak_pick(envelope, pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=0.3, wait=3)
    onsets_per_second = float(len(peaks) / (len(mono) / TEMPO_RATE))
    e = envelope - envelope.mean()
    acf = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * n)) ** 2)[:n]
    acf = acf / max(acf[0], 1e-9)
    lags = np.arange(int(fps * 60 / 200), int(fps * 60 / 50) + 1)
    lags = lags[lags < n - 2]
    seg = acf[lags]
    local = sorted((i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]),
                   key=lambda i: -seg[i])
    if not local:
        return {"status": "abstained", "reason": "no_beat", "onsets_per_second": round(onsets_per_second, 2)}

    def bpm_of(i):
        lag = lags[i]
        a, b, c = acf[lag - 1], acf[lag], acf[lag + 1]
        d = a - 2 * b + c
        return 60 * fps / (lag + (0.5 * (a - c) / d if abs(d) > 1e-12 else 0.0))
    best = local[0]
    bpm = bpm_of(best)
    salience = float(seg[best])
    rival = next((i for i in local[1:] if abs(bpm_of(i) / bpm - 1) > 0.08), None)
    ambiguous = bool(rival is not None and seg[rival] >= 0.8 * salience and any(
        abs(bpm_of(rival) / bpm / r - 1) < 0.06 for r in (2, 0.5, 3, 1 / 3, 1.5, 2 / 3)))
    found = {"bpm": round(float(bpm), 1), "salience": round(salience, 3), "octave_ambiguous": ambiguous,
             "onsets_per_second": round(onsets_per_second, 2)}
    if salience < SALIENCE_MIN or ambiguous:
        return {**found, "status": "abstained", "reason": "unclear_beat"}
    return {**found, "status": "measured"}


def instrumentation(rows: np.ndarray | None) -> dict | None:
    from lss.bgm import music_description

    if rows is None:
        return None
    try:
        heard = music_description.instruments(rows)
    except music_description.MusicDescriptionError:
        return None
    families = sorted(((v["score"], name) for name, v in heard["families"].items()), reverse=True)
    present = [name for score, name in families if score >= INSTRUMENT_PRESENT][:3]
    return {"present": present, "names": [music_description.FAMILIES[n][0] for n in present],
            "drums": heard["families"]["percussion"]["score"] >= INSTRUMENT_PRESENT,
            "voice": bool(heard["voice_recorded"]), "scores": {n: s for s, n in families}}


def run_rows(run_path: str | Path | None, start_frame: int, end_frame: int, rate: int) -> np.ndarray | None:
    """The run's own per-second probabilities over the cue (sha-checked), or None."""
    from lss.bgm import music_description
    from lss.harness.fetch import file_sha256

    try:
        run = json.loads(Path(run_path).read_text(encoding="utf-8"))
        item = ((run.get("semantics") or {}).get("evidence") or {}).get("probabilities") or {}
        folder = (Path(run_path).parent / "semantics").resolve()
        path = (folder / str(item["name"])).resolve()
        if path.parent != folder or file_sha256(path) != item["sha256"]:
            return None
        return music_description.cue_rows(np.load(path).astype(np.float32), start_frame, end_frame, rate)
    except (OSError, ValueError, KeyError, TypeError, music_description.MusicDescriptionError):
        return None


def measure(path: Path, rows: np.ndarray | None = None) -> dict:
    import soundfile as sf

    from lss.bgm import music_description

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(1)
    try:
        key = music_description.key(mono, rate)
    except music_description.MusicDescriptionError:
        key = {"status": "abstained"}
    hop = max(1, rate // 10)
    blocks = mono[: len(mono) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((blocks ** 2).mean(1)) + 1e-9) if blocks.size else np.zeros(1)
    active = db > db.max() - 40
    swing = float(np.percentile(db[active], 90) - np.percentile(db[active], 10)) if active.any() else 0.0
    return {"key": key, "tempo": tempo(mono, rate), "instrumentation": instrumentation(rows),
            "dynamics_db": round(swing, 1)}


def words(facts: dict) -> str:
    """The measured music in order-sheet words (only what was measured)."""
    out = []
    key = facts.get("key") or {}
    if key.get("status") == "measured":
        out.append("長調（明るめ）" if key["mode"] == "major" else "短調（暗め・切なめ）")
    t = facts.get("tempo") or {}
    if t.get("status") == "measured":
        out.append(f"テンポ約{round(t['bpm'])}・拍がはっきり")
    elif t.get("onsets_per_second") is not None:
        out.append("拍は自由に揺れる")
    rate = t.get("onsets_per_second")
    if rate is not None:
        out.append("音数が多く細かい" if rate > 3.2 else "音数が少なくゆったり" if rate < 1.8 else "音数は中くらい")
    inst = facts.get("instrumentation") or {}
    if inst.get("names"):
        out.append("編成: " + "・".join(inst["names"]) + ("（リズム楽器あり）" if inst.get("drums") else "（リズム楽器なし）"))
    if inst.get("voice"):
        out.append("歌声あり")
    out.append("強弱の起伏が大きい" if facts.get("dynamics_db", 0) > 12 else "強弱はなだらか")
    return "、".join(out)


def tags(facts: dict) -> dict:
    """Player tags only for measured values."""
    found = {}
    t = facts.get("tempo") or {}
    if t.get("status") == "measured":
        found["BPM"] = str(int(round(t["bpm"])))
    key = facts.get("key") or {}
    if key.get("status") == "measured":
        found["INITIALKEY"] = key["tag"]
    return found
