"""取り出した音に、欲しい音が本当にあるかを確かめ、無い区間を静かにする。

分離器は欲しい音が無くても何かを返す。返った音をそのまま「欲しい音」として公開しない。
別のfamilyで学習した音響イベントのタグ付け（PANNs CNN14、AudioSet 527クラス、`lss.bgm.event_tagging`
と同じmodel・同じ重み）で、出力の各秒にその音のクラスが聞こえるかを測る。

- 窓は長さ3秒・1秒格子（窓 ``i`` は ``i`` 秒から ``i+3`` 秒。最後は0詰め）。各秒の証拠は、その秒を含む
  窓の確率の最大。
- 音量: 各秒のRMSが ``ENERGY_FLOOR_DBFS`` 未満、または親の音より ``RELATIVE_FLOOR_DB`` 以上小さい秒は
  「音が無い」（台詞は ``FAMILY_RELATIVE_FLOOR_DB``。元の音で聞こえない大きさの漏れを台詞にしない）。
- 台詞と歌は取り違えやすいので、相手のクラス（``CONTRAST``）のほうが強い秒は証拠にしない。
- 欲しい音がある: 証拠が ``PRESENT_PROBABILITY`` 以上の秒が ``PRESENT_SECONDS`` 秒以上ある
  （AudioSetのクラスが無い対象は、音量の条件だけ）。無ければ成果物を作らない（UNKNOWNでなく、
  この検証規則で確かめた「無い」）。
- 閾値は制御混合（公式サントラの音楽＋LibriSpeechの台詞＋ESC-50の効果音、効果音は3分類の効果音から
  言葉で抽出）で決めた: 正しい言葉 20 件中 14 件を「ある」、無関係な言葉 60 件中 2 件を「ある」とした
  （``docs/engine/WANTED_SOUND.md``）。無関係な言葉にも分離器は親の音をそのまま返すことがあり、
  タグの確認が無いと誤った成果物になる。
- 区間の選別（``gate``）: 特定の効果音・声のように、ある時だけ鳴る対象は、証拠が ``KEEP_PROBABILITY``
  以上の秒（前後1秒を含む）だけを残し、他の秒を ``FADE_SECONDS`` の傾斜で0にする。音楽・台詞全体の
  ように続いて鳴る対象は、音が無い秒だけを0にする。

言葉の時刻の証拠（`lss.sound.pe_frame`、PE-A-Frame）:

- 言葉で分離した音・語彙に無い言葉の音は、要求の言葉そのもので「その音が40 msごとに鳴っているか」を
  出力・親・残り（親−出力）で測る（``Frames``）。AudioSetのクラスが無い言葉も、これで確かめられる。
- ある: 出力の音が聞こえる時刻で、確率 ``FRAME_PRESENT_PROBABILITY`` 以上の40 msが ``FRAME_PRESENT_FRAMES``
  以上（0.4秒）、または（AudioSetのクラスがある言葉は）上のタグの規則。語彙に無い言葉は言葉の証拠だけ。
  制御混合（2 fold、正解40・無関係な語彙120・無関係な自由語40、言葉の分離は時刻のanchor付き）で、
  正解33を「ある」・2を保留・5を見逃し、無関係な語彙8・自由語0を誤って「ある」とした（タグだけでは
  正解31・無関係6。以前は自由語を確かめず、無関係な自由語40件すべてを「ある」にしていた）。
- 取り出せなかった（``not_extracted``）: 出力では「ある」に届かないのに、親では同じ基準で聞こえる。
  これは「無い」ではないので、無いとは記録せず保留にする（UNKNOWN ≠ ABSENT）。
- 時刻の選別: ある時だけ鳴る対象は、40 msの証拠が ``FRAME_KEEP_PROBABILITY`` 以上の時刻（前後
  ``FRAME_KEEP_DILATION_SECONDS`` を含む）も残す（秒の格子より細かく、短い音を切らない）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import gcd

import numpy as np

from lss.sound import audioset

MODEL_RATE = 32000
WINDOW_SECONDS = 3
ENERGY_FLOOR_DBFS = -60.0
RELATIVE_FLOOR_DB = -35.0
# 台詞は元の音に対して聞こえる大きさでなければ台詞の成果物にしない（曲の歌の漏れを台詞と呼ばない）。
FAMILY_RELATIVE_FLOOR_DB = {"speech": -25.0, "speech_part": -25.0}
PRESENT_PROBABILITY = 0.03
PRESENT_SECONDS = 2
KEEP_PROBABILITY = 0.015
KEEP_DILATION_SECONDS = 1
FADE_SECONDS = 0.05
SPEECH_SOURCE_PROBABILITY = 0.5
CINEMATIC_SPEECH_FRACTION = 0.08
GATED_FAMILIES = frozenset({"sound", "speech_part", "singing_part"})
# 言葉の時刻の証拠（PE-A-Frame）。制御混合（音楽＋台詞＋効果音、3分類の効果音から言葉で抽出）で決めた。
SPAN_PROBABILITY = 0.3                 # 言葉の分離へ渡す時刻（upstream predict_spans の既定と同じ）
FRAME_PRESENT_PROBABILITY = 0.8
FRAME_PRESENT_FRAMES = 10
# A brief event needs coincident class and word evidence in parent and output.
# Three consecutive frames prevent isolated probability spikes from authorizing it.
TRANSIENT_PRESENT_FRAMES = 3
FRAME_KEEP_PROBABILITY = 0.3
FRAME_KEEP_DILATION_SECONDS = 0.5
# 取り違えやすい相手。台詞の証拠は歌唱系より話し声系が強い秒だけ、歌の証拠はその逆。
SINGING_CLASSES = ("Singing", "Choir", "Male singing", "Female singing", "Child singing", "Synthetic singing",
                   "Rapping", "Humming", "Vocal music", "A capella")
SPEECH_CLASSES = ("Speech", "Male speech, man speaking", "Female speech, woman speaking",
                  "Child speech, kid speaking", "Conversation", "Narration, monologue")
CONTRAST = {"speech": SINGING_CLASSES, "speech_part": SINGING_CLASSES,
            "singing": SPEECH_CLASSES, "lead_vocals": SPEECH_CLASSES, "backing_vocals": SPEECH_CLASSES,
            "singing_part": SPEECH_CLASSES}
# 楽器（SCNetの「その他」から言葉で取る楽器）。実楽曲（MUSDB18、「その他」が1つの楽器の曲と、別の楽器を
# 重ねた曲）で決めた。頼んだ楽器しか鳴っていない秒（親のタグ）は親をそのまま使い、それ以外はmaskの下限を0.5に
# する（単独 0.5 → 7.8 dB、他の楽器と重なる -5.7 → -3.3 dB、中央値）。有無は親での言葉の証拠で決める
# （``decide(parent_presence=True)``）。タグの比較だけで有無を決めると、頼んだ楽器を取りこぼす（31/42）。
INSTRUMENT_MIN_PROBABILITY = 0.1
INSTRUMENT_RIVAL_RATIO = 0.25
INSTRUMENT_ALONE_PROBABILITY = 0.2
INSTRUMENT_ALONE_RIVAL_RATIO = 0.5
INSTRUMENT_MASK_FLOOR = 0.5
# 楽器の比較で、楽器を名指ししない広いクラス（どの楽器にも付く）は相手に数えない。
BROAD_INSTRUMENT_CLASSES = frozenset({"Music", "Musical instrument", "Orchestra", "Percussion",
                                      "Plucked string instrument", "Bowed string instrument", "Brass instrument",
                                      "Wind instrument, woodwind instrument", "Keyboard (musical)",
                                      "Mallet percussion", "Drum", "Drum kit", "Sound effect", "Electronic music",
                                      "Video game music", "Choir", "Chant"})


def instrument_classes(base_id: str) -> tuple[list[int], list[int]]:
    """楽器の語彙の音の、(自分のクラス, 相手の楽器のクラス) のAudioSetの番号。

    相手は、その他（ドラム・ベース・歌を除いた音）に鳴る旋律・和音の楽器のうち、自分と特定のクラスを共有しない
    もの（「ピアノ」と「エレピ」、「ギター」と「エレキギター」は同じ系統として比べない）。"""
    from lss.sound import lexicon, plan

    def specific(concept) -> set[str]:
        return set(concept.audioset) - BROAD_INSTRUMENT_CLASSES

    asked = lexicon.BY_ID[base_id]
    own = specific(asked) or set(asked.audioset)
    excluded = {"drums", "bass", *plan._DRUM_PARTS, *plan._BASS_PARTS}
    rivals = set()
    for concept in lexicon.CONCEPTS:
        if concept.family != "music_part" or concept.id == base_id or concept.id in excluded:
            continue
        if specific(concept) & own:
            continue   # 同じ系統
        rivals |= specific(concept)
    rivals -= own
    return [audioset.INDEX[name] for name in sorted(own)], [audioset.INDEX[name] for name in sorted(rivals)]


def instrument_seconds(tags: np.ndarray, base_id: str) -> tuple[np.ndarray, np.ndarray]:
    """親（その他）のタグから、(頼んだ楽器が他の楽器に負けない秒, 頼んだ楽器しか鳴っていない秒)。"""
    own, rivals = instrument_classes(base_id)
    mine = tags[:, own].max(axis=1)
    theirs = tags[:, rivals].max(axis=1) if rivals else np.zeros(len(tags), dtype=np.float32)
    distinct = (mine >= INSTRUMENT_MIN_PROBABILITY) & (mine >= INSTRUMENT_RIVAL_RATIO * theirs)
    alone = (mine >= INSTRUMENT_ALONE_PROBABILITY) & (theirs < INSTRUMENT_ALONE_RIVAL_RATIO * mine)
    return distinct, alone


def removal_rivals(classes: tuple[str, ...]) -> list[int]:
    """Unrelated named effects; a sound's own ancestors/children are not rivals.

    A low thunder score cannot authorize removing a strongly tagged train.
    AudioSet's ontology prevents rain/water and dog/bark from competing with
    their own semantic relatives. Music and generic sound-effect tags never
    authorize or veto a specific effect by themselves.
    """
    from lss.sound import audioset_relations, lexicon

    own = set(classes)
    excluded = own | {"Sound effect", "Music", "Speech", "Noise"}
    for name in own:
        excluded.update(audioset_relations.RELATED.get(name, ()))
    rivals = {name for concept in lexicon.CONCEPTS if concept.family == "sound"
              and not (set(concept.audioset) & own) for name in concept.audioset}
    return [audioset.INDEX[name] for name in sorted(rivals - excluded)]


def removal_support(tags: np.ndarray, classes: tuple[str, ...]) -> np.ndarray:
    """Seconds where requested effect evidence dominates unrelated named effects."""
    own = [audioset.INDEX[name] for name in classes]
    rivals = removal_rivals(classes)
    return tags[:, own].max(axis=1) >= tags[:, rivals].max(axis=1) if rivals else np.ones(len(tags), dtype=bool)


def removal_coincident_frames(output: np.ndarray, parent: np.ndarray, rate: int,
                              tags: np.ndarray, classes: tuple[str, ...], frames: "Frames") -> int:
    """Count jointly confirmed word frames inside audible dominant tag seconds.

    This is independent confirmation for a short named effect, not finer tag
    localization: the tagger still supplies three-second-window observations.
    Missing or padded observations never count as confirmation.
    """
    from lss.sound import pe_frame

    out_db, parent_db = second_rms_db(output, rate), second_rms_db(parent, rate)
    count = min(len(out_db), len(parent_db), len(tags))
    own = [audioset.INDEX[name] for name in classes]
    supported = (removal_support(tags[:count], classes)
                 & (tags[:count, own].max(axis=1) >= PRESENT_PROBABILITY)
                 & (out_db[:count] >= ENERGY_FLOOR_DBFS)
                 & (parent_db[:count] >= ENERGY_FLOOR_DBFS)
                 & (out_db[:count] >= parent_db[:count] + RELATIVE_FLOOR_DB))
    supported = np.repeat(supported, pe_frame.FRAMES_PER_SECOND)
    length = min(len(supported), len(frames.output), len(frames.parent))
    return int(np.count_nonzero(supported[:length]
                               & (frames.output[:length] >= FRAME_PRESENT_PROBABILITY)
                               & (frames.parent[:length] >= FRAME_PRESENT_PROBABILITY)))


class VerificationError(RuntimeError):
    """検証の入力か出力が契約に合わない。"""


def policy_revision() -> dict:
    body = {"schema_version": 1, "tagger": "panns-cnn14-audioset (lss.bgm.event_tagging weights)",
            "model_rate": MODEL_RATE, "window_seconds": WINDOW_SECONDS, "energy_floor_dbfs": ENERGY_FLOOR_DBFS,
            "relative_floor_db": RELATIVE_FLOOR_DB, "family_relative_floor_db": FAMILY_RELATIVE_FLOOR_DB,
            "present_probability": PRESENT_PROBABILITY,
            "present_seconds": PRESENT_SECONDS, "keep_probability": KEEP_PROBABILITY,
            "keep_dilation_seconds": KEEP_DILATION_SECONDS, "fade_seconds": FADE_SECONDS,
            "gated_families": sorted(GATED_FAMILIES),
            "contrast": {family: list(names) for family, names in sorted(CONTRAST.items())},
            "speech_source_probability": SPEECH_SOURCE_PROBABILITY,
            "cinematic_speech_fraction": CINEMATIC_SPEECH_FRACTION,
            "frames": {"span_probability": SPAN_PROBABILITY, "present_probability": FRAME_PRESENT_PROBABILITY,
                       "present_frames": FRAME_PRESENT_FRAMES,
                       "transient_present_frames": TRANSIENT_PRESENT_FRAMES,
                       "transient_authority": "extraction_only",
                       "keep_probability": FRAME_KEEP_PROBABILITY,
                       "keep_dilation_seconds": FRAME_KEEP_DILATION_SECONDS},
            "removal": {"short_named_effect": "coincident-parent-output-in-dominant-tag-seconds-v1",
                        "minimum_coincident_frames": FRAME_PRESENT_FRAMES},
            "instruments": {"min_probability": INSTRUMENT_MIN_PROBABILITY, "rival_ratio": INSTRUMENT_RIVAL_RATIO,
                            "alone_probability": INSTRUMENT_ALONE_PROBABILITY,
                            "alone_rival_ratio": INSTRUMENT_ALONE_RIVAL_RATIO,
                            "mask_floor": INSTRUMENT_MASK_FLOOR, "broad_classes": sorted(BROAD_INSTRUMENT_CLASSES)}}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {**body, "policy_id": "sound-verify-v3-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]}


@dataclass
class Tagger:
    model: object
    device: str = "cuda:0"

    @classmethod
    def load(cls, weights, *, device: str = "cuda:0") -> "Tagger":
        from lss.bgm import event_tagging

        return cls(event_tagging.load_model(weights, device=device), device)

    def close(self) -> None:
        import torch

        self.model = None
        torch.cuda.empty_cache()

    def seconds(self, audio: np.ndarray, rate: int) -> np.ndarray:
        """各秒（切り上げ）の、その秒を含む窓の確率の最大 (秒, 527)。"""
        import torch
        from scipy.signal import resample_poly

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        divisor = gcd(MODEL_RATE, rate)
        mono = resample_poly(mono, MODEL_RATE // divisor, rate // divisor).astype(np.float32)
        count = max(1, int(np.ceil(len(mono) / MODEL_RATE)))
        size = WINDOW_SECONDS * MODEL_RATE
        padded = np.pad(mono, (0, count * MODEL_RATE + size))
        windows = np.zeros((count, len(audioset.LABELS)), dtype=np.float32)
        batch = 64
        with torch.inference_mode():
            for start in range(0, count, batch):
                chunk = np.stack([padded[i * MODEL_RATE:i * MODEL_RATE + size]
                                  for i in range(start, min(count, start + batch))])
                probabilities, _ = self.model(torch.from_numpy(chunk).to(self.device))
                windows[start:start + len(chunk)] = probabilities.float().cpu().numpy()
        if not np.isfinite(windows).all():
            raise VerificationError("タグ付けの出力に有限でない値があります")
        per_second = windows.copy()
        for shift in range(1, WINDOW_SECONDS):
            per_second[shift:] = np.maximum(per_second[shift:], windows[:-shift])
        return per_second


def second_rms_db(audio: np.ndarray, rate: int) -> np.ndarray:
    """各秒（切り上げ）のRMS（dBFS、全channel）。"""
    count = max(1, int(np.ceil(len(audio) / rate)))
    values = np.zeros(count, dtype=np.float64)
    flat = audio.reshape(len(audio), -1).astype(np.float64)
    for index in range(count):
        block = flat[index * rate:(index + 1) * rate]
        values[index] = np.mean(block ** 2) if block.size else 0.0
    return 10.0 * np.log10(values + 1e-20)


def source_kind(tags: np.ndarray) -> str:
    """台詞が十分にある音源（番組・映像）は ``cinematic``、それ以外は ``music``。"""
    speech = tags[:, audioset.INDEX["Speech"]]
    fraction = float(np.mean(speech >= SPEECH_SOURCE_PROBABILITY)) if len(speech) else 0.0
    return "cinematic" if fraction >= CINEMATIC_SPEECH_FRACTION else "music"


@dataclass
class Frames:
    """要求の言葉の、40 msごとの確率（言葉が複数なら各時刻の最大）。"""
    output: np.ndarray
    parent: np.ndarray


def frame_mass(probabilities: np.ndarray) -> float:
    """その言葉の音がはっきり鳴っている長さ（秒）。``SPAN_PROBABILITY`` 以上の40 msの確率の和。"""
    from lss.sound import pe_frame

    values = np.asarray(probabilities, dtype=np.float64)
    return float(values[values >= SPAN_PROBABILITY].sum() / pe_frame.FRAMES_PER_SECOND)


def _frames_heard(probabilities: np.ndarray, audible_seconds: np.ndarray) -> bool:
    """音が聞こえる秒の中で、確率が ``FRAME_PRESENT_PROBABILITY`` 以上の40 msが ``FRAME_PRESENT_FRAMES`` 以上。"""
    from lss.sound import pe_frame

    audible = np.repeat(audible_seconds.astype(bool), pe_frame.FRAMES_PER_SECOND)
    length = min(len(audible), len(probabilities))
    return int(((np.asarray(probabilities[:length]) >= FRAME_PRESENT_PROBABILITY) & audible[:length]).sum()) \
        >= FRAME_PRESENT_FRAMES


def frame_seconds(probabilities: np.ndarray, count: int) -> np.ndarray:
    """40 msごとの確率から各秒の最大（count秒）。"""
    from lss.sound import pe_frame

    step = pe_frame.FRAMES_PER_SECOND
    values = np.zeros(count, dtype=np.float32)
    for index in range(count):
        block = probabilities[index * step:(index + 1) * step]
        values[index] = float(block.max()) if len(block) else 0.0
    return values


def _transient_heard(frames: Frames, class_support: np.ndarray) -> bool:
    """Require consecutive, simultaneous parent/output words and audible class support.

    This corroborates extraction only. It grants no open-vocabulary, instrument,
    speech, singing or removal authority and never changes accepted legacy audio.
    """
    from lss.sound import pe_frame

    support = np.repeat(class_support.astype(bool), pe_frame.FRAMES_PER_SECOND)
    length = min(len(support), len(frames.output), len(frames.parent))
    if length < TRANSIENT_PRESENT_FRAMES:
        return False
    flags = (support[:length] & (frames.output[:length] >= FRAME_PRESENT_PROBABILITY)
             & (frames.parent[:length] >= FRAME_PRESENT_PROBABILITY))
    return bool((np.convolve(flags.astype(np.int32), np.ones(TRANSIENT_PRESENT_FRAMES, dtype=np.int32),
                             mode="valid") >= TRANSIENT_PRESENT_FRAMES).any())


@dataclass
class Decision:
    present: bool
    keep: np.ndarray              # 残す秒（bool）
    evidence: np.ndarray          # 各秒の証拠（対象クラスの確率の最大。クラスが無ければnan）
    reasons: list[str]
    outcome: str = ""             # "present" / "absent" / "not_extracted"
    keep_frames: np.ndarray | None = None   # 40 msごとに残す時刻（言葉の時刻の証拠があるとき）
    frame_evidence: np.ndarray | None = None

    def __post_init__(self):
        if not self.outcome:
            self.outcome = "present" if self.present else "absent"

    def summary(self) -> dict:
        finite = self.evidence[np.isfinite(self.evidence)]
        out = {"present": self.present, "outcome": self.outcome, "kept_seconds": int(self.keep.sum()),
               "seconds": int(len(self.keep)),
               "max_evidence": round(float(finite.max()), 4) if finite.size else None,
               "reasons": list(self.reasons)}
        if self.frame_evidence is not None:
            out["max_frame_evidence"] = round(float(self.frame_evidence.max(initial=0.0)), 4)
        return out


def decide(output: np.ndarray, parent: np.ndarray, rate: int, *, classes: tuple[str, ...], family: str,
           tags: np.ndarray | None, frames: Frames | None = None, parent_presence: bool = False,
           allow_brief_extraction: bool = True) -> Decision:
    """出力に欲しい音があるか、どの秒を残すかを決める。``tags`` は出力のタグ（classesが無ければ不要）。

    ``parent_presence``（楽器）: 出力は親の一部（maskの下限・親をそのまま使う秒）なので、有無は親での言葉の
    時刻の証拠で決める（実楽曲で、頼んだ楽器 36/42 → 39/42 を「ある」、無い楽器を「ある」19/52 → 6/52）。

    ``frames`` は要求の言葉の時刻の証拠（出力と親）。あれば「ある」の証拠に加え、取り出せなかったことを
    見分け、残す時刻を40 msで決める。"""
    # Brief corroboration permits exposing a useful extract, never subtracting
    # that material from the source. Removal callers disable it at this boundary.
    out_db = second_rms_db(output, rate)
    parent_db = second_rms_db(parent, rate)
    count = min(len(out_db), len(parent_db))
    out_db, parent_db = out_db[:count], parent_db[:count]
    floor = FAMILY_RELATIVE_FLOOR_DB.get(family, RELATIVE_FLOOR_DB)
    audible = (out_db >= ENERGY_FLOOR_DBFS) & (out_db >= parent_db + floor)
    reasons = []
    if classes:
        if tags is None:
            raise VerificationError("対象クラスの確認にタグが必要です")
        indices = [audioset.INDEX[name] for name in classes]
        evidence = tags[:count, indices].max(axis=1)
        contrast = [audioset.INDEX[name] for name in CONTRAST.get(family, ()) if name not in classes]
        if contrast:
            # 取り違えやすい相手のほうが強い秒は、欲しい音の証拠にしない。
            evidence = np.where(tags[:count, contrast].max(axis=1) >= evidence, 0.0, evidence)
        strong = audible & (evidence >= PRESENT_PROBABILITY)
        present = int(strong.sum()) >= PRESENT_SECONDS
        if not present:
            reasons.append("target_class_not_heard")
    else:
        evidence = np.full(count, np.nan)
        present = int(audible.sum()) >= PRESENT_SECONDS if frames is None else False
    frame_evidence = None
    heard_in_parent = False
    if frames is not None:
        # 言葉で分離した音: 言葉の時刻の証拠、または（クラスがあれば）タグの証拠で「ある」。
        frame_evidence = frame_seconds(frames.output, count)
        heard_by_words = _frames_heard(frames.output, audible)
        present = heard_by_words or (bool(classes) and present)
        if heard_by_words and "target_class_not_heard" in reasons:
            reasons.remove("target_class_not_heard")
            reasons.append("target_heard_by_words")
        if not present:
            reasons.append("target_words_not_heard")
        heard_in_parent = _frames_heard(frames.parent, parent_db >= ENERGY_FLOOR_DBFS)
        if parent_presence:
            present = heard_in_parent
            reasons = ["target_words_heard_in_parent"] if present else ["target_words_not_heard_in_parent"]
        elif allow_brief_extraction and not present and family == "sound" and classes and _transient_heard(frames, strong):
            present = True
            reasons = [reason for reason in reasons if reason not in
                       {"target_class_not_heard", "target_words_not_heard"}]
            reasons.append("transient_corroborated_in_parent_and_output")
    if not audible.any():
        reasons.append("no_audible_output")
        present = False
    if family in GATED_FAMILIES and classes:
        keep = audible & (evidence >= KEEP_PROBABILITY)
        dilated = keep.copy()
        for shift in range(1, KEEP_DILATION_SECONDS + 1):
            dilated[shift:] |= keep[:-shift]
            dilated[:-shift] |= keep[shift:]
        keep = dilated & audible
    else:
        keep = audible
    keep_frames = None
    if frames is not None and family in GATED_FAMILIES:
        from lss.sound import pe_frame

        width = int(round(FRAME_KEEP_DILATION_SECONDS * pe_frame.FRAMES_PER_SECOND))
        flags = frames.output >= FRAME_KEEP_PROBABILITY
        keep_frames = np.convolve(flags.astype(np.float32), np.ones(2 * width + 1, dtype=np.float32),
                                  mode="same") > 0
        # タグの証拠で残す秒も含める（どちらかの証拠がある時刻を残す）。タグのクラスが無い言葉は言葉の証拠だけ。
        # 音が無い秒は残さない。
        tagged = keep if classes else np.zeros(count, dtype=bool)
        per_second = np.repeat(tagged, pe_frame.FRAMES_PER_SECOND)[:len(keep_frames)]
        per_second = np.pad(per_second, (0, max(0, len(keep_frames) - len(per_second))))
        audible_frames = np.repeat(audible, pe_frame.FRAMES_PER_SECOND)[:len(keep_frames)]
        audible_frames = np.pad(audible_frames, (0, max(0, len(keep_frames) - len(audible_frames))))
        keep_frames = (keep_frames | per_second.astype(bool)) & audible_frames.astype(bool)
        keep = np.array([bool(keep_frames[index * pe_frame.FRAMES_PER_SECOND:
                                          (index + 1) * pe_frame.FRAMES_PER_SECOND].any()) for index in range(count)])
    outcome = "present" if present else ("not_extracted" if heard_in_parent else "absent")
    if outcome == "not_extracted":
        reasons.append("heard_in_parent_but_not_extracted")
    if not present:
        keep = np.zeros(count, dtype=bool)
        keep_frames = None if keep_frames is None else np.zeros_like(keep_frames)
    return Decision(present=present, keep=keep, evidence=evidence, reasons=reasons, outcome=outcome,
                    keep_frames=keep_frames, frame_evidence=frame_evidence)


def within(output: np.ndarray, parent: np.ndarray, frames: "Frames | None", rate: int, start: float,
           end: float) -> tuple[np.ndarray, np.ndarray, "Frames | None"]:
    """時間の範囲 [start, end) 秒の外を、境目に ``FADE_SECONDS`` の傾斜を付けて0にする（出力・親・言葉の証拠）。"""
    gain = np.zeros(len(output), dtype=np.float32)
    first, last = int(round(start * rate)), int(round(end * rate))
    gain[max(0, first):max(0, last)] = 1.0
    fade = max(1, int(FADE_SECONDS * rate))
    padded = np.pad(gain, (fade, fade), mode="edge")   # 音の端は傾斜を付けない（範囲が先頭・末尾まで）
    gain = np.convolve(padded, np.ones(fade, dtype=np.float32) / fade, mode="same")[fade:fade + len(gain)]
    gain = gain.astype(np.float32)
    if frames is not None:
        from lss.sound import pe_frame

        def cut(values: np.ndarray) -> np.ndarray:
            index = np.arange(len(values)) / pe_frame.FRAMES_PER_SECOND
            return np.where((index >= start) & (index < end), values, 0.0).astype(np.float32)

        frames = Frames(output=cut(frames.output), parent=cut(frames.parent))
    return output * gain[:, None], parent * gain[:, None], frames


def apply_keep(audio: np.ndarray, rate: int, keep: np.ndarray, keep_frames: np.ndarray | None = None) -> np.ndarray:
    """残す秒（``keep_frames`` があれば40 msの時刻）以外を、境目に短い傾斜を付けて0にする。"""
    gain = np.zeros(len(audio), dtype=np.float32)
    if keep_frames is not None:
        from lss.sound import pe_frame

        for index, flag in enumerate(keep_frames):
            if flag:
                start = int(round(index * rate / pe_frame.FRAMES_PER_SECOND))
                gain[start:int(round((index + 1) * rate / pe_frame.FRAMES_PER_SECOND))] = 1.0
    else:
        for index, flag in enumerate(keep):
            if flag:
                gain[index * rate:(index + 1) * rate] = 1.0
    fade = max(1, int(FADE_SECONDS * rate))
    kernel = np.ones(fade, dtype=np.float32) / fade
    smoothed = np.convolve(gain, kernel, mode="same")
    return (audio * smoothed[:, None] if audio.ndim == 2 else audio * smoothed).astype(np.float32)
