"""公開BGMの音楽としての記述（楽器の大分類と調）。名前と再生ソフトの項目のための証拠で、判定・公開には使わない。

一つのcueだけで決まる（再登場・参照・外部の情報は要らない）。新しい推論はしない：

- 楽器：BGMのrunが判定のために既に求めた、Music stemの1秒ごとのAudioSet確率（PANNs Cnn14、3秒窓・1秒刻み、
  `semantics/probabilities.npy`）を、cueの範囲（解析音声の時計＝`start_frame`/`end_frame`）で平均する。
  一つの楽器名ではなく大分類（金管・弦楽器など）だけを言う：実の公開cueと同じ録音の既発売版では、金管の中の名前
  （French horn・Trombone・Brass instrument）が入れ替わる（`release-evidence/music-description-20261016/`）。
  大分類の確率が `INSTRUMENT_MIN` 未満なら言わない（実の公開cue 71件で約3割が言える）。
- 調：公開したcueの音声そのものの音符のクロマ（倍音を辞書で説明して除いたもの）の平均を、Krumhansl-Kesslerの
  長調・短調の輪郭と相関する。
  相関が `KEY_MIN_CORRELATION` 未満（調性が弱い・雑音）か、1位と2位の差が `KEY_MIN_MARGIN` 未満（相対の長短・
  5度の取り違え）なら言わない。
- 感情（AudioSetの Happy/Sad music など）は実のcueで最大0.2未満の雑音だったので使わない。テンポはオクターブの
  取り違えが半数のcueで決められなかったので使わない（誤ったBPMは黙った誤り）。
"""
from __future__ import annotations

import hashlib
import json
from math import gcd
from pathlib import Path

import numpy as np

from lss.bgm import versions
from lss.sound import audioset

METHOD = "bgm-music-description-v1"

# 実の公開cue 71件（独立検証）：0.2では3件しか言えず、0.1で20件。0.1でも一つ目の大分類は、cueのFLACを直接タグ付けした
# 結果と20/20、同じ録音の既発売版と4/4一致した。二つ目は16/20しか一致しないので、言うのは一つだけ。
INSTRUMENT_MIN = 0.1
INSTRUMENT_NAMED = 1
# 調の閾値は、正解の調が分かる合成（music21のBach合唱曲338曲＋弦楽四重奏40楽章、原調・移調・-20dBの雑音）と
# 雑音・話し声に似た音の反証80本で決めた（`release-evidence/music-description-20261016/`）。
# 下の音符のクロマで：相関0.6・差0.08 → Bach 網羅69%・完全一致95.9%・属調の誤り1.9%、反証で言った調0/80
# （反証の相関の最大0.57）。倍音を含む従来のクロマ（`versions.chroma`）では同じ網羅で完全一致は約80%（属調の誤り約20%）。
KEY_MIN_MARGIN = 0.08
KEY_MIN_CORRELATION = 0.6
# 音符のクロマ（NNLS）：半音ごとの振幅スペクトルを、倍音が s^(h-1) で減る音符の辞書の非負の和で説明し、
# 音符の量だけを音名へ畳む（Mauch & Dixon 2010 の NNLS Chroma の考え方）。倍音（特に第3倍音＝5度上）が
# 音名へ漏れて属調と取り違えるのを防ぐ。
NOTE_FFT = 8192
NOTE_HOP = 2205
NOTE_PARTIALS = 8
NOTE_PARTIAL_DECAY = 0.7
NOTE_RANGE = (33, 96)        # 辞書の音符（A1〜C7）
SPECTRUM_RANGE = (33, 108)   # 半音のスペクトル（倍音の上限まで）
KEY_MIN_FRAMES = 20          # 音符のクロマは0.4秒ごと：8秒未満の音声では調を言わない
SINGING_RECORD_MIN = 0.03    # 記録だけ（題名には使わない）。既発売の歌の9曲中7曲・器楽の6%がこれ以上

# 大分類：AudioSetの名前 → 表示。大分類の点数は、属する名前の平均確率の最大値。
FAMILIES = {
    "brass": ("金管", ("Brass instrument", "French horn", "Trumpet", "Trombone")),
    "strings": ("弦楽器", ("Bowed string instrument", "String section", "Violin, fiddle", "Cello", "Double bass")),
    "woodwind": ("木管", ("Wind instrument, woodwind instrument", "Flute", "Clarinet", "Saxophone")),
    "piano": ("ピアノ", ("Piano", "Electric piano", "Keyboard (musical)", "Harpsichord")),
    "plucked": ("撥弦楽器", ("Guitar", "Electric guitar", "Acoustic guitar", "Plucked string instrument", "Harp",
                         "Mandolin", "Banjo", "Ukulele", "Sitar", "Zither")),
    "percussion": ("打楽器", ("Drum kit", "Drum", "Timpani", "Percussion", "Snare drum", "Bass drum", "Cymbal", "Gong")),
    "mallet": ("鍵盤打楽器", ("Mallet percussion", "Marimba, xylophone", "Glockenspiel", "Vibraphone", "Tubular bells")),
    "synth": ("シンセサイザー", ("Synthesizer", "Sampler")),
    "organ": ("オルガン", ("Organ", "Hammond organ", "Electronic organ")),
}
VOICE = ("Singing", "Choir", "Vocal music", "Female singing", "Male singing", "Child singing")

# Krumhansl-Kessler の輪郭（C を主音とする）。
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# 再生ソフトの INITIALKEY の書き方と、日本語の調名。異名同音は慣用の側。
_KEY_TAG = {"major": ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"),
            "minor": ("Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm")}
_KEY_JA = {"major": ("ハ", "変ニ", "ニ", "変ホ", "ホ", "ヘ", "嬰ヘ", "ト", "変イ", "イ", "変ロ", "ロ"),
           "minor": ("ハ", "嬰ハ", "ニ", "変ホ", "ホ", "ヘ", "嬰ヘ", "ト", "嬰ト", "イ", "変ロ", "ロ")}
_MODE_JA = {"major": "長調", "minor": "短調"}

POLICY = {"method": METHOD, "instrument_min": INSTRUMENT_MIN, "instrument_named": INSTRUMENT_NAMED, "key_min_margin": KEY_MIN_MARGIN,
          "key_min_correlation": KEY_MIN_CORRELATION,
          "key_min_frames": KEY_MIN_FRAMES,
          "note_chroma": {"fft": NOTE_FFT, "hop": NOTE_HOP, "partials": NOTE_PARTIALS, "decay": NOTE_PARTIAL_DECAY,
                          "notes": list(NOTE_RANGE), "spectrum": list(SPECTRUM_RANGE), "solver": "scipy-nnls"}, "singing_record_min": SINGING_RECORD_MIN,
          "families": {name: list(members) for name, (_, members) in FAMILIES.items()},
          "voice": list(VOICE), "key_profiles": "krumhansl-kessler",
          "probabilities": "bgm-first-pass-music-cnn14-3s-1s-mean-over-cue"}
POLICY_ID = "bgm-music-description-" + hashlib.sha256(
    json.dumps(POLICY, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


class MusicDescriptionError(ValueError):
    """記述の材料が契約に合わない。"""


def _index(name: str) -> int:
    try:
        return audioset.INDEX[name]
    except KeyError as exc:  # ラベル表との食い違いは黙って0にしない
        raise MusicDescriptionError(f"AudioSetに無い名前です: {name}") from exc


def instruments(rows: np.ndarray) -> dict:
    """cueの範囲の1秒ごとの確率（秒×527）から、大分類の点数と言える大分類（強い順、最大2つ）。"""
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != len(audioset.LABELS) or rows.shape[0] == 0:
        raise MusicDescriptionError("cueの範囲の確率がありません")
    if not np.isfinite(rows).all():
        raise MusicDescriptionError("確率に有限でない値があります")
    mean = rows.mean(axis=0)
    scores = {}
    for name, (_, members) in FAMILIES.items():
        values = {member: float(mean[_index(member)]) for member in members}
        best = max(values, key=values.get)
        scores[name] = {"score": round(values[best], 4), "strongest": best}
    named = sorted((name for name in scores if scores[name]["score"] >= INSTRUMENT_MIN),
                   key=lambda name: -scores[name]["score"])[:INSTRUMENT_NAMED]
    voice = max(float(mean[_index(name)]) for name in VOICE)
    return {"families": scores, "named": named, "seconds": int(rows.shape[0]),
            "voice": round(voice, 4), "voice_recorded": voice >= SINGING_RECORD_MIN,
            "speech": round(float(mean[_index("Speech")]), 4)}


def _note_dictionary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(NOTE_FFT, 1.0 / versions.SAMPLE_RATE)
    midi = 69.0 + 12.0 * np.log2(np.maximum(freqs, 1e-6) / 440.0)
    pitches = np.arange(SPECTRUM_RANGE[0], SPECTRUM_RANGE[1] + 1)
    to_semitones = np.clip(1.0 - np.abs(midi[None, :] - pitches[:, None]), 0.0, 1.0)
    to_semitones[:, freqs < 25.0] = 0.0
    notes = np.arange(NOTE_RANGE[0], NOTE_RANGE[1] + 1)
    atoms = np.zeros((pitches.size, notes.size))
    for column, note in enumerate(notes):
        for partial in range(1, NOTE_PARTIALS + 1):
            position = note + 12.0 * np.log2(partial) - pitches[0]
            low, fraction = int(np.floor(position)), position - np.floor(position)
            weight = NOTE_PARTIAL_DECAY ** (partial - 1)
            if 0 <= low < pitches.size:
                atoms[low, column] += weight * (1.0 - fraction)
            if 0 <= low + 1 < pitches.size:
                atoms[low + 1, column] += weight * fraction
    atoms /= np.linalg.norm(atoms, axis=0, keepdims=True)
    return to_semitones.astype(np.float32), atoms, notes % 12


def note_chroma(samples: np.ndarray) -> np.ndarray:
    """11025Hz・monoの音声の、音符の量の音名（時間×12、各フレームを最大値で正規化、2フレームずつ平均＝0.4秒ごと）。"""
    from scipy.optimize import nnls

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < NOTE_FFT:
        return np.zeros((0, 12), dtype=np.float32)
    to_semitones, atoms, pitch_class = _note_dictionary()
    window = np.hanning(NOTE_FFT).astype(np.float32)
    count = 1 + (samples.size - NOTE_FFT) // NOTE_HOP
    out = np.zeros((count, 12))
    for begin in range(0, count, 256):
        end = min(count, begin + 256)
        index = np.arange(begin, end)[:, None] * NOTE_HOP + np.arange(NOTE_FFT)[None, :]
        semitones = np.abs(np.fft.rfft(samples[index] * window, axis=1)) @ to_semitones.T
        for row, spectrum in enumerate(semitones):
            if spectrum.max() <= 1e-9:
                continue
            amounts, _ = nnls(atoms, spectrum.astype(np.float64))
            np.add.at(out[begin + row], pitch_class, amounts)
    peak = out.max(axis=1, keepdims=True)
    out = np.where(peak > 0.0, out / np.maximum(peak, 1e-12), 0.0)
    usable = count // 2 * 2
    return out[:usable].reshape(-1, 2, 12).mean(axis=1).astype(np.float32)


def key(samples: np.ndarray, rate: int) -> dict:
    """cueの音声（mono）の調。言えない時は ``status`` が ``abstained``。"""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or rate <= 0 or not np.isfinite(samples).all():
        raise MusicDescriptionError("調を測る音声が不正です")
    if rate != versions.SAMPLE_RATE:
        from scipy.signal import resample_poly

        divisor = gcd(versions.SAMPLE_RATE, rate)
        samples = resample_poly(samples, versions.SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    frames = note_chroma(samples)
    if frames.shape[0] < KEY_MIN_FRAMES:
        return {"status": "abstained", "reason": "too_short", "frames": int(frames.shape[0])}
    profile = frames.mean(axis=0)
    if float(profile.std()) <= 1e-6:
        return {"status": "abstained", "reason": "no_pitch_structure", "frames": int(frames.shape[0])}
    scores = []
    for mode, template in (("major", _MAJOR), ("minor", _MINOR)):
        for tonic in range(12):
            scores.append((float(np.corrcoef(profile, np.roll(template, tonic))[0, 1]), tonic, mode))
    scores.sort(key=lambda item: -item[0])
    (best, tonic, mode), (second, second_tonic, second_mode) = scores[0], scores[1]
    found = {"frames": int(frames.shape[0]), "correlation": round(best, 4), "margin": round(best - second, 4),
             "tonic": tonic, "mode": mode, "tag": _KEY_TAG[mode][tonic],
             "runner_up": _KEY_TAG[second_mode][second_tonic]}
    if best < KEY_MIN_CORRELATION:
        return {**found, "status": "abstained", "reason": "weak_tonality"}
    if best - second < KEY_MIN_MARGIN:
        return {**found, "status": "abstained", "reason": "ambiguous"}
    return {**found, "status": "measured", "name": f"{_KEY_JA[mode][tonic]}{_MODE_JA[mode]}"}


def cue_rows(probabilities: np.ndarray, start_frame: int, end_frame: int, sample_rate: int) -> np.ndarray:
    """解析音声の時計の [start, end) に中心がある1秒窓（中心は i+0.5 秒）。"""
    if sample_rate <= 0 or end_frame <= start_frame or start_frame < 0:
        raise MusicDescriptionError("cueの範囲が不正です")
    first = int(np.ceil(start_frame / sample_rate - 0.5))
    last = int(np.ceil(end_frame / sample_rate - 0.5))
    first, last = max(first, 0), min(last, probabilities.shape[0])
    if last <= first:
        raise MusicDescriptionError("cueの範囲に確率の窓がありません")
    return probabilities[first:last]


def describe(path: Path, rows: np.ndarray | None, *, asset_id: str) -> dict:
    """公開したcueのFLACと、その範囲の確率（無ければ楽器は言わない）から記述の証拠を作る。"""
    import soundfile as sf

    try:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError) as exc:
        raise MusicDescriptionError(f"cueの音声を読めません: {exc}") from exc
    measured_key = key(audio.mean(axis=1), rate)
    rows_state = "absent"
    try:
        heard = instruments(rows) if rows is not None else None
        rows_state = "used" if heard is not None else "absent"
    except MusicDescriptionError:
        heard, rows_state = None, "unusable"   # 範囲の確率が使えない時も調は言える
    named = list(heard["named"]) if heard else []
    found = {"asset_id": asset_id, "policy_id": POLICY_ID, "instrument_rows": rows_state,
             "instruments": heard, "instrument_names": [FAMILIES[name][0] for name in named],
             "key": measured_key}
    text = phrase(found)
    return {**found, "status": "measured" if text else "insufficient", "phrase": text}


def phrase(description: dict) -> str | None:
    """題名の記述部分（例：「金管と弦楽器 — ニ短調」）。言えることが無ければNone。"""
    names = description.get("instrument_names") or []
    measured = (description.get("key") or {}).get("status") == "measured"
    tonality = description["key"]["name"] if measured else None
    if names and tonality:
        return f"{'と'.join(names)} — {tonality}"
    if names:
        return f"{'と'.join(names)}の劇伴"
    if tonality:
        return f"{tonality}の劇伴"
    return None


def key_tag(description: dict | None) -> str | None:
    found = (description or {}).get("key") or {}
    return found.get("tag") if found.get("status") == "measured" else None
