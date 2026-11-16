"""意味判定（`cue_semantics`）の区間を、先に固定した対応領域S3の候補へ写し、公開・保留・対象外を決める。

公開してよいかの最終判断は `supported_scope.classify_candidate` が持つ。このmoduleは、cue_semanticsが
測った観測と理由を対応領域が読む候補の形へ写し、製品が既に作る窓ごとの観測（原本のタグ付け）と
編集境界から、音の中身の安全の観測（`safety`）と番組構造の文脈（`context`）を測る。

S3 mapping for single-episode score and singing, version 1. S2 (v12, bgm-product-repair-r2) repair:

- **番組の役割は文脈であり、意味の真実ではない。** 番組の端・放送挿入のタグ（Television）・番組の中ほどの
  短い音楽・編集境界の両側は ``context`` に記録するだけで、公開を止めない（S2はこれらで本編の劇伴を保留し、
  境界の両側6秒を削っていた。`.state/validation/bgm-product-repair-r3/`）。役割のラベルだけで聞こえる劇伴を
  対象外にしない。
- **編集境界（デジタル無音）の検出は残す。** 完成番組では、CM・告知・アイキャッチ・OP/ED・予告の境目に、全chが
  完全な無音に近い短い区間（``EDIT_SILENCE_DB`` 以下が ``EDIT_SILENCE_SECONDS`` 以上）がある。公開の候補はこの
  境界で分ける（``split_at_edit_boundaries``。境界は音として実際に途切れた所なので、境界をまたぐ一曲は作らない）。
  緩衝の秒数は削らない。
- **境界に挟まれた短い区画（挿入の区画）の音楽は、役割未確定として保留する。** 区画にはCM（対象外）も、
  サブタイトル・アイキャッチ・予告・番組内コーナー（番組の音楽でありうる）も入り、実行時の観測では区別できない
  （Development：corpus-d の単独の区画もCM）。対象外とは言わず ``HELD``（``program_insert_role_unresolved``）にし、
  音は保留候補として系譜付きで残す（再利用・同定・将来の補完・比較の材料）。
- **音の中身の安全の保留は残す。** 原本の窓のうち、非音楽・非人声のイベント音（AudioSetの動物・自然音・物音・
  由来の曖昧な音。Silenceは除く）の確率の最大が ``EVENT_WINDOW_PROBABILITY`` 以上の窓が区間の
  ``EVENT_SCENE_FRACTION`` 以上なら保留（``heavy_nonmusic_event_scene``）。効果音の重い残り・劇伴の削れはこの場面に
  集中し、今の音声の手掛かりでは安全な部分だけを取り出せない（`bgm-product-repair-r2`、`bgm-sfx-mechanism-r1`）。
  この保留は全てのイベント音の窓を数える。窓のイベント音が鈴・鐘の音（``CHIME_CLASSES``）だけで、原本も音楽の出力も
  音楽と言う窓を数えない割引（鉄琴・チェレスタ等の劇伴そのものにタグが付く。ep1 u11）は、保留した候補を描画した音で
  再確認へ指名する判定とその再確認だけが使い、公開の判断は外さない。
- Singing and theme shape are recorded as observations. Wanted music includes sung music and choir. A candidate that continues beyond its edge, is too short or long, has music gaps, matches a known release, or duplicates a prior publication remains held.
  Every non-publication reason is named.
- 書き出す音そのものの完全性（L0）を公開直前に測る。小さい音・帯域の狭い原本の劇伴を落とさないため、
  大きさ・帯域・左右の広がりのしきい値は置かない。
"""
from __future__ import annotations

import numpy as np

from lss.bgm import cue_readiness, cue_semantics
from lss.sound import audioset
from lss.bgm.supported_scope import SupportedScope, classify_candidate, classify_result

SOURCE_DOMAIN = "finished_stereo_program"
# 一曲と言えないことだけが理由の保留。これだけなら範囲未確定のcueとして公開できる。
EXTENT_ONLY_REASONS = frozenset({"possible_change_inside"})
EDGE_REASONS = frozenset({"start_edge_music_continues_outside_candidate",
                          "end_edge_music_continues_outside_candidate"})
PROGRAM_EDGE = "episode_edge"
# 候補の時間軸。区間は1秒格子なのでミリ秒で丸めずに表せる。
TIMEBASE = 1000
MAPPING_VERSION = "cue-semantics-to-scope-s4-short-v3-clipped-reference-provenance"
# AudioSet（PANNs CNN14、527クラス、class_labels_indices.csv）の上位分類のうち、非音楽・非人声のイベント音：
# Animal(72-136)、Natural sounds(283-299)、Sounds of things(300-499)、Source-ambiguous(501-505、Silence 500は除く)。
EVENT_CLASSES = tuple(range(72, 137)) + tuple(range(283, 500)) + (501, 502, 503, 504, 505)
TELEVISION_CLASS = 524
EVENT_WINDOW_PROBABILITY = 0.2
EVENT_SCENE_FRACTION = 0.2
# EVENT_CLASSESのうち、音程のある持続する鈴・鐘の音だけを表すラベル（Doorbellとその子Ding-dong、Onomatopoeiaの
# DingとJingle, tinkle）。同じ音色をAudioSetは音楽側（Bell・Chime・Glockenspiel・Tubular bells等、EVENT_CLASSESの外）
# にも置くので、劇伴の鉄琴・チェレスタ・鈴にこれらが付く。打撃（Clang・Chink, clink）・電子音（Ping・Beep）・
# 電話（Telephone bell ringing・Ringtone）・家畜のCowbellは効果音として明確なので含めない。
CHIME_CLASSES = tuple(audioset.INDEX[name] for name in ("Doorbell", "Ding-dong", "Ding", "Jingle, tinkle"))
# 鈴だけの窓を数えないのは、原本のMusicがこれ以上かつ一回目の音楽の出力のMusicがこれ以上のときだけ。
CHIME_MIXTURE_MUSIC = 0.5
CHIME_STEM_MUSIC = 0.4
# 以下は文脈の観測（公開を止めない）。番組の中ほどの短い音楽・放送挿入のタグの記録に使う。
BROADCAST_PROBABILITY = 0.2
INSERT_MAXIMUM_SECONDS = 20.0
MID_PROGRAM_BAND = (0.35, 0.65)
# 番組の構造（編集境界・挿入の区画）を仮定する素材の最小の長さ。短い素材はデジタル無音で分けない。
MID_PROGRAM_MINIMUM_SECONDS = 600.0
EDIT_SILENCE_DB = -100.0
EDIT_SILENCE_SECONDS = 0.3
EDIT_HOP_SECONDS = 0.1
INSERT_BLOCK_MAXIMUM_SECONDS = 90.0
EDIT_BOUNDARY = "edit_boundary"
SAFETY_METHOD = "runtime-content-observations-v4-chime-discount-rendered-only"
CONTEXT_METHOD = "program-structure-context-v1"
# 公開を止めうる安全の観測は、音の中身の観測だけ（番組の役割のラベルは含めない）。
SAFETY_NAMES = ("heavy_nonmusic_event_scene",)
PROGRAM_INSERT_HOLD = "program_insert_role_unresolved"
# 公開しなかった区間の音は、保留候補として残る（`run["held"]` の原音・処理済みの音声）。その使い道。
RETAINED_USES_HELD = ("donor", "identity_evidence", "future_completion", "comparison")
RETAINED_USES_UNSUPPORTED = ("identity_evidence", "comparison")


def product_scope() -> SupportedScope:
    """Product publication scope for single-episode score and singing."""
    from lss.bgm import short_cues
    return SupportedScope(
        source_domains=(SOURCE_DOMAIN,),
        published_semantic_roles=("dramatic_score", "singing_score"),
        covered_semantic_roles=("dramatic_score", "singing_score"),
        minimum_cue_seconds=10.0, maximum_cue_seconds=150.0,
        short_cue_minimum_seconds=short_cues.POLICY.minimum_seconds,
        short_cue_required_evidence=("self_contained_short_music",),
        required_evidence=("source_provenance", "occurrence_span", "cue_edges"),
        quality_gates={"integrity_passed": 1.0},
        singing_observation_policy="ignore",
        derived_role_policy="accept_declared_ambiguity",
        cue_extent_policy="allow_unresolved_cue_extent",
        publication_allowed=True,
        safety_holds=SAFETY_NAMES,
        program_insert_policy="hold_role_unresolved")


def digital_silence_runs(path) -> list[tuple[float, float]]:
    """原本（解析用複製）の全chが ``EDIT_SILENCE_DB`` 以下の区間（``EDIT_HOP_SECONDS`` 刻み、``EDIT_SILENCE_SECONDS`` 以上）。"""
    import soundfile as sf

    runs: list[tuple[float, float]] = []
    with sf.SoundFile(str(path)) as stream:
        hop = int(round(stream.samplerate * EDIT_HOP_SECONDS))
        levels: list[float] = []
        while True:
            block = stream.read(hop * 600, dtype="float32", always_2d=True)
            if not len(block):
                break
            count = len(block) // hop
            if count:
                frames = block[:count * hop].reshape(count, hop, block.shape[1])
                levels.extend((20 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=(1, 2))) + 1e-12))
                              .tolist())
    low = np.asarray(levels) <= EDIT_SILENCE_DB
    index = 0
    while index < len(low):
        if not low[index]:
            index += 1
            continue
        end = index
        while end < len(low) and low[end]:
            end += 1
        if (end - index) * EDIT_HOP_SECONDS >= EDIT_SILENCE_SECONDS - 1e-9:
            runs.append((round(index * EDIT_HOP_SECONDS, 1), round(end * EDIT_HOP_SECONDS, 1)))
        index = end
    return runs




def split_at_edit_boundaries(segments: list[dict], runs, program_seconds: float) -> list[dict]:
    """公開の候補（欲しい劇伴・一曲かどうかだけ未確定）を編集境界（デジタル無音）で分ける。

    境界は音が実際に途切れた所なので、境界をまたいで一つのcueにしない。分けた端は検出した無音の端に
    置き、``edit_boundary`` にする。整数秒へ外向きに丸めて音のある部分を捨てない。境界の両側を削る緩衝は置かない（アイキャッチ・タイトルの付近の劇伴も
    番組の音楽でありうる。役割は文脈で、削る根拠にしない）。分けた区間は元の意味判定・理由・特徴量を引き継ぎ、
    元の区間を ``split_from`` に残す。1秒未満の切れ端は捨てる（無音の内側）。
    それ以外の区間と、``MID_PROGRAM_MINIMUM_SECONDS`` 未満の短い素材（番組の構造を仮定しない）は変えない。
    """
    out: list[dict] = []
    for segment in segments:
        start, end = float(segment["start_seconds"]), float(segment["end_seconds"])
        if not publication_candidate(segment) or not runs or program_seconds < MID_PROGRAM_MINIMUM_SECONDS:
            out.append(segment)
            continue
        cuts = sorted((float(a), float(b)) for a, b in runs if b > start and a < end)
        if not cuts:
            out.append(segment)
            continue
        pieces, cursor = [], start
        for a, b in cuts:
            pieces.append((cursor, min(end, a)))
            cursor = max(cursor, b)
        pieces.append((cursor, end))
        pieces = [(a, b) for a, b in pieces if b - a >= 1.0]
        if pieces == [(start, end)]:
            out.append(segment)
            continue
        boundaries = [list(run) for run in runs if run[1] >= start - 1.5 and run[0] <= end + 1.5]
        for a, b in pieces:
            piece = {**segment, "start_seconds": a, "end_seconds": b, "core_seconds": [a, b],
                     "split_from": [start, end], "edit_boundaries": boundaries}
            relation = segment.get("reference_relation")
            if relation:
                from lss.bgm import reference_relations
                piece["reference_relation"] = reference_relations.restrict_relation_span(
                    relation, a, b, parent_span=[start, end])
            if a != start:
                piece["start_edge"] = {"kind": EDIT_BOUNDARY}
            if b != end:
                piece["end_edge"] = {"kind": EDIT_BOUNDARY}
            out.append(piece)
    return out


def edit_sections(boundaries, program_seconds: float) -> list[tuple[float, float]]:
    """編集境界に挟まれた区画（前の境界の終わりから次の境界の始まりまで、番組の始まり・終わりを含む）。"""
    points = [(0.0, 0.0), *sorted((float(a), float(b)) for a, b in boundaries),
              (float(program_seconds), float(program_seconds))]
    return [(a1, b0) for (_, a1), (b0, _) in zip(points[:-1], points[1:]) if b0 - a1 > 0.05]


def context(segment: dict, mixture: np.ndarray, boundaries=()) -> dict:
    """区間の番組構造の文脈。公開を止めるのは挿入の区画（役割未確定）だけで、他は記録だけにする。

    ``insert_section``：``MID_PROGRAM_MINIMUM_SECONDS`` 以上の番組で、区間の中心が編集境界に挟まれた
    ``INSERT_BLOCK_MAXIMUM_SECONDS`` 未満の区画にある（CM・告知・アイキャッチ・予告・番組内コーナー等。区別できない）。
    """
    mixture = np.asarray(mixture, dtype=np.float32)
    start, end = float(segment["start_seconds"]), float(segment["end_seconds"])
    program = float(len(mixture))
    first = max(0, int(np.floor(start)))
    last = min(len(mixture), int(np.ceil(end)))
    television = float(mixture[first:last, TELEVISION_CLASS].max()) if last > first else None
    structured = program >= MID_PROGRAM_MINIMUM_SECONDS and bool(boundaries)
    midpoint = (start + end) / 2
    section = next(((a, b) for a, b in edit_sections(boundaries, program) if a - 1.0 <= midpoint < b + 1.0),
                   None) if structured else None
    edges = (segment["start_edge"]["kind"], segment["end_edge"]["kind"])
    middle = (program >= MID_PROGRAM_MINIMUM_SECONDS and end > MID_PROGRAM_BAND[0] * program
              and start < MID_PROGRAM_BAND[1] * program)
    return {"method": CONTEXT_METHOD,
            "edit_section": [round(section[0], 1), round(section[1], 1)] if section else None,
            "insert_section": bool(section is not None and section[1] - section[0] < INSERT_BLOCK_MAXIMUM_SECONDS),
            "program_edge": PROGRAM_EDGE in edges,
            "mid_program_short_music": (end - start) <= INSERT_MAXIMUM_SECONDS and middle,
            "television_max": None if television is None else round(television, 4),
            "television_tag": television is not None and television >= BROADCAST_PROBABILITY}


def _window_music(values, start: int, end: int) -> np.ndarray:
    """窓 [start, end) のMusicの確率。足りない窓・有限でない値は0（鈴の窓を数えない根拠にしない）。"""
    values = np.asarray(values, dtype=np.float64).reshape(-1)[start:end]
    out = np.zeros(end - start)
    out[:len(values)] = np.where(np.isfinite(values), values, 0.0)
    return out


def event_windows(block: np.ndarray, mixture_music=None, stem_music=None) -> tuple[np.ndarray, np.ndarray]:
    """窓ごとのイベント音の有無（数える窓）と、鈴だけなので数えなかった窓。

    ``block`` はイベント音を数えるタグの確率 (窓, 527)。``mixture_music``・``stem_music`` は同じ窓の原本・一回目の
    音楽の出力のMusicの確率で、どちらかが無ければ鈴の窓も数える。
    """
    block = np.asarray(block)
    present = block[:, list(EVENT_CLASSES)] >= EVENT_WINDOW_PROBABILITY
    event = present.any(axis=1)
    if mixture_music is None or stem_music is None:
        return event, np.zeros(len(block), dtype=bool)
    chime = np.isin(np.asarray(EVENT_CLASSES), CHIME_CLASSES)
    only_chime = event & ~present[:, ~chime].any(axis=1)
    musical = ((np.asarray(mixture_music, dtype=np.float64) >= CHIME_MIXTURE_MUSIC)
               & (np.asarray(stem_music, dtype=np.float64) >= CHIME_STEM_MUSIC))
    discounted = only_chime & musical
    return event & ~discounted, discounted


def safety(segment: dict, mixture: np.ndarray, boundaries=(), *, music=None) -> dict:
    """区間の音の中身の安全の観測。``mixture`` は原本（解析用複製）のタグの確率 (窓, 527)、窓 i は [i, i+1) 秒。

    ``event_window_fraction`` と ``heavy_nonmusic_event_scene`` は常に全てのイベント音の窓を数える（公開の判断）。
    ``music`` は ``(原本のMusic, 音楽の出力のMusic)`` の窓ごとの確率（同じ1秒格子）。渡せば、鈴・鐘の音だけの窓で
    両方が音楽と言う窓を数えない割引の観測（``chime_discounted_*``、``event_windows``）を別に添える。割引は
    描画した音での再確認の指名と再確認だけが使い、公開の判断の保留は外さない。
    """
    mixture = np.asarray(mixture, dtype=np.float32)
    if mixture.ndim != 2 or mixture.shape[1] <= max(max(EVENT_CLASSES), TELEVISION_CLASS):
        raise ValueError("原本のタグの確率の形が不正です")
    start = max(0, int(np.floor(float(segment["start_seconds"]))))
    end = min(len(mixture), int(np.ceil(float(segment["end_seconds"]))))
    if end <= start:
        raise ValueError("安全の観測を測る窓がありません")
    block = mixture[start:end]
    if not np.isfinite(block).all():
        raise ValueError("原本のタグの確率に有限でない値があります")
    if music is not None:
        mixture_music, stem_music = (_window_music(values, start, end) for values in music)
    else:
        mixture_music = stem_music = None
    fraction = float(np.mean(event_windows(block)[0]))
    out = {"method": SAFETY_METHOD, "windows": end - start,
           "event_window_fraction": round(fraction, 4),
           "heavy_nonmusic_event_scene": fraction >= EVENT_SCENE_FRACTION}
    if mixture_music is not None:
        event, discounted = event_windows(block, mixture_music, stem_music)
        discounted_fraction = float(np.mean(event))
        out.update({"chime_music_windows_not_counted": int(discounted.sum()),
                    "chime_discounted_event_window_fraction": round(discounted_fraction, 4),
                    "chime_discounted_heavy_nonmusic_event_scene": discounted_fraction >= EVENT_SCENE_FRACTION})
    return out


def cue_extent(segment: dict) -> dict:
    """区間が一曲と言えるか。替わり目の疑いがあれば未確定（複数曲を含みうる）。"""
    doubt = "possible_change_inside" in (segment.get("reasons") or [])
    return {"state": "unresolved" if doubt else "single_cue",
            "reasons": ["possible_change_inside"] if doubt else [],
            "inner_split_distance": (segment.get("features") or {}).get("inner_split_distance")}



def publication_candidate(segment: dict) -> bool:
    """対応領域で公開しうる区間（欲しい劇伴、または一曲かどうかだけが未確定の区間）。重複の照合に使う。"""
    if segment["semantic_class"] == cue_semantics.WANTED:
        return True
    return (segment["semantic_class"] == cue_semantics.UNCERTAIN
            and set(segment.get("reasons") or []) == EXTENT_ONLY_REASONS)



def unresolved_extent(segment: dict) -> bool:
    return (segment.get("scope") or {}).get("cue_extent", {}).get("state") == "unresolved"



def candidate(segment: dict, index: int, source: dict, quality: dict | None) -> dict | None:
    """意味判定の区間1つを対応領域の候補へ写す。音楽に聞こえない区間は候補にしない（None）。"""
    label = segment["semantic_class"]
    if label == cue_semantics.NON_BGM:
        return None
    reasons = set(segment.get("reasons") or [])
    features = segment.get("features") or {}
    singing_present = bool(features.get("singing_present", "singing_present" in reasons))
    start, end = (float(value) for value in segment.get("core_seconds",
                                                        (segment["start_seconds"], segment["end_seconds"])))
    edges = {"start": segment["start_edge"]["kind"], "end": segment["end_edge"]["kind"]}
    item = {
        "candidate_id": f"segment-{index:03d}", "sample_rate": TIMEBASE,
        "start_frame": int(round(start * TIMEBASE)), "end_frame": int(round(end * TIMEBASE)),
        "source_domain": SOURCE_DOMAIN, "identity_state": "unresolved",
        # Keep the existing broad score routing role. A tagger observation can
        # protect musical voices but cannot establish a singing-score identity.
        "semantic_role": "dramatic_score", "cue_extent": cue_extent(segment),
        "observations": {"singing": "present" if singing_present else "absent",
                         "theme_shape": bool(features.get("theme_shape", "theme_song_shape_with_singing" in reasons))},
        "evidence": {"source_provenance": source,
                     "occurrence_span": [int(round(start * TIMEBASE)), int(round(end * TIMEBASE))],
                     # 端の根拠は、音楽の止まり目・強い替わり目・話の端。候補の外へ音楽が続く端は根拠にならない。
                     "cue_edges": None if reasons & EDGE_REASONS else edges},
        "quality": quality,
    }
    from lss.bgm import short_cues
    if short_cues.confirmed(segment):
        item["evidence"]["self_contained_short_music"] = segment["short_music"]
    observed = segment.get("safety")
    if isinstance(observed, dict):
        item["safety"] = {name: observed.get(name) for name in SAFETY_NAMES}
    observed_context = segment.get("context")
    if isinstance(observed_context, dict):
        # 番組構造は文脈。挿入の区画にあることだけを対応領域へ渡す（役割未確定の保留）。
        item["program_insert"] = bool(observed_context.get("insert_section"))
    if label == cue_semantics.OTHER_NON_TARGET:
        item["excluded_non_target"] = "theme_song_shape_with_singing"
        item["observations"] = {"singing": "present"}
    elif label == cue_semantics.KNOWN_RELEASED:
        item["known_release"] = "exact"
    elif label == cue_semantics.DUPLICATE_PUBLISHED:
        found = segment.get("duplicate_of") or [{}]
        item["duplicate_of_published_cue_id"] = str(found[0].get("path") or found[0].get("cue_key")
                                                    or "prior_publication")
    elif label == cue_semantics.UNCERTAIN:
        if "partly_registered_reference" in reasons:
            item["known_release"] = "partial"
        if "music_not_continuous" in reasons:
            item["semantic_role"] = "unresolved_music_like"
    elif label != cue_semantics.WANTED:
        raise ValueError(f"未知の意味判定です: {label!r}")
    return item



def retention(disposition: str, reason: str) -> dict:
    """公開しなかった区間の音の扱い。捨てずに保留候補として残し、使い道を名前で残す（公開と保持は別）。"""
    uses = RETAINED_USES_HELD if disposition == "HELD" else RETAINED_USES_UNSUPPORTED
    return {"state": "retained_internal", "material": "held_candidate_audio", "uses": list(uses),
            "publication": "not_public", "reason": reason}


def decide(segments: list[dict], source: dict, *, mixture: np.ndarray | None = None,
           boundaries=(), quality: dict | None = None) -> dict:
    """区間ごとの公開・保留・対象外を決め、各区間の ``scope`` へ書く。全体の結論も返す。

    ``mixture`` を渡せば各区間の安全の観測と番組構造の文脈を測って ``segment["safety"]``・``segment["context"]`` へ書く（渡さなければ書いてある観測を
    使う）。``quality`` は区間の位置（index）ごとの品質。無い区間は、書き出す音を測る前の判断として完全性を
    満たすと仮定する（公開前に必ず `final` で測り直す）。
    """
    scope = product_scope()
    candidates = []
    for index, segment in enumerate(segments):
        if mixture is not None and segment["semantic_class"] != cue_semantics.NON_BGM:
            segment["safety"] = safety(segment, mixture, boundaries)
            segment["context"] = context(segment, mixture, boundaries)
        measured = (quality or {}).get(index, {"integrity_passed": 1.0})
        item = candidate(segment, index, source, measured)
        previous = segment.get("scope") or {}
        segment["scope"] = None
        if item is None:
            continue
        candidates.append((segment, item))
    result = classify_result([item for _, item in candidates], scope,
                             music_activity_found=bool(candidates))
    for (segment, item), row in zip(candidates, result["candidates"]):
        segment["scope"] = {"scope_id": scope.scope_id, "candidate_id": item["candidate_id"],
                            "disposition": row["disposition"], "reason": row["reason"],
                            "cue_extent": item["cue_extent"],
                            **{name: row[name] for name in ("missing_evidence", "failed_gates", "safety_holds")
                               if name in row},
                            # 書き出す音を測った区間は、その測定を残す。
                            **({"integrity": previous["integrity"]} if "integrity" in previous else {}),
                            **({"retention": retention(row["disposition"], row["reason"])}
                               if row["disposition"] != "PUBLISHED" else {})}
    return result



def publishable(segment: dict) -> bool:
    return (segment.get("scope") or {}).get("disposition") == "PUBLISHED"



def integrity(audio: np.ndarray, rate: int) -> dict:
    """書き出す音（frames, 2）の完全性（L0）。有限・2ch・無音でない・片側が無音でない。"""
    measured = cue_readiness.measure(np.asarray(audio).T, int(rate))
    passed = (measured["finite"] and measured["channels"] == 2 and not measured["silent"]
              and not measured["silent_channel_indexes"])
    return {"integrity_passed": 1.0 if passed else 0.0, "method": measured["method"],
            "finite": measured["finite"], "channels": measured["channels"], "silent": measured["silent"],
            "silent_channel_indexes": measured["silent_channel_indexes"],
            "channel_energy_dbfs": [_finite(value) for value in measured["channel_energy_dbfs"]],
            # 無音などで測れない値は無い（None）として残す。-infを記録へ書かない。
            "true_peak_dbtp": _finite(measured["true_peak_dbtp"]),
            "loudness_lufs": _finite(measured["loudness_lufs"])}



def _finite(value):
    if value is None or isinstance(value, bool):
        return None
    value = float(value)
    return value if np.isfinite(value) else None



def final(segment: dict, index: int, source: dict, measured: dict) -> dict:
    """書き出す音を測った後の最終判断。完全性を欠けば公開しない（保留、理由付き）。"""
    item = candidate(segment, index, source, {"integrity_passed": measured["integrity_passed"]})
    row = classify_candidate(item, product_scope())
    segment["scope"] = {**segment["scope"], "disposition": row["disposition"], "reason": row["reason"],
                        **({"failed_gates": row["failed_gates"]} if "failed_gates" in row else {}),
                        **({"retention": retention(row["disposition"], row["reason"])}
                           if row["disposition"] != "PUBLISHED" else {}),
                        "integrity": measured}
    return segment["scope"]



def recipe() -> dict:
    """Recipeの識別へ入れる宣言。"""
    scope = product_scope()
    return {"scope": scope.to_dict(), "scope_id": scope.scope_id, "mapping": MAPPING_VERSION,
            "extent_only_reasons": sorted(EXTENT_ONLY_REASONS),
            "safety": {"method": SAFETY_METHOD, "event_classes": "audioset 72-136,283-499,501-505",
                       "event_window_probability": EVENT_WINDOW_PROBABILITY,
                       "event_scene_fraction": EVENT_SCENE_FRACTION,
                       "chime_music_discount": {"classes": list(CHIME_CLASSES),
                                                "mixture_music_minimum": CHIME_MIXTURE_MUSIC,
                                                "stem_music_minimum": CHIME_STEM_MUSIC,
                                                "rule": "window whose only event classes are chime classes",
                                                "applies_to": "rendered_content nomination and rendered recheck only; decision hold counts every event window"},
                       "input": "mixture_analysis_copy event tags"},
            "context": {"method": CONTEXT_METHOD, "publication_effect": "none except insert_section",
                        "program_edge": PROGRAM_EDGE, "television_class": TELEVISION_CLASS,
                        "television_tag_probability": BROADCAST_PROBABILITY,
                        "mid_program_short_seconds": INSERT_MAXIMUM_SECONDS,
                        "mid_program_band": list(MID_PROGRAM_BAND),
                        "structured_program_minimum_seconds": MID_PROGRAM_MINIMUM_SECONDS,
                        "insert_section_maximum_seconds": INSERT_BLOCK_MAXIMUM_SECONDS,
                        "insert_section_disposition": f"HELD {PROGRAM_INSERT_HOLD} (retained, not non-target)"},
            "edit_boundaries": {"edit_silence_db": EDIT_SILENCE_DB, "edit_silence_seconds": EDIT_SILENCE_SECONDS,
                                "edit_hop_seconds": EDIT_HOP_SECONDS,
                                "split": "publication candidates split at digital-silence runs, no bumper",
                                "input": "mixture_analysis_copy samples"},
            "retention": {"held": list(RETAINED_USES_HELD), "unsupported": list(RETAINED_USES_UNSUPPORTED)},
            "integrity": cue_readiness.METHOD}
