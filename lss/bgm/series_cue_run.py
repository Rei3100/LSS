"""シリーズの未発売cue経路を、実際の音声に対して実行する。

`series_cues` が宣言を持ち、このモジュールが音声の読み取り・照合・切り出しを行う。
分離器は使わない。原本は読むだけで変えない（切り出しの前後でhashを確かめる）。

手順:

1. 各話の原本mixを読み、既発売録音との一致区間を求めて差し引く対象にする。
   入力はmkv等でよい。復号は製品と同じ入力準備
   （`lss.core.audio_input.prepare_input`、channel数と標本化周波数を変えない）を通す。
2. 話どうしを両方向で照合し、同じ録音の再登場を組として得る。
3. `series_cues.plan_cues` で録音単位へ直し、抽出元と区間を決める。
4. 決まった区間を原本から切り出し、来歴を付けて保存する。
5. 中身の測定を渡せた場合だけ関門を適用する（`series_cues.apply_content_guard`）。

保存するのは**非公開の候補**であり、正式なBGM成果物ではない。意味役割
（欲しい劇伴か）は判定しない。OP/EDは参照録音として渡すことで差し引かれる。
"""
from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from lss.bgm import identity, occurrence_quality, occurrence_stack, series_cues, series_reuse

RECORD_NAME = "series-cues.json"
CANDIDATE_DIRNAME = "candidates"
MUSIC_DIRNAME = "music-estimates"
REPRESENTATION = "noncanonical_staged_candidate"
PUBLICATION_STATE = "not_published"
# 相互説明を測るとき、区間の前後へ足す余裕。照合の下限を割らないため。
MUTUAL_PAD_SECONDS = 0.5
# 同じ話の中の繰り返しを見るとき、これ以上重なっている組は同じ場所なので数えない。
SELF_OVERLAP_SHARE = 0.25
MATCH_CACHE_TYPE = "series_match_cache"
MATCH_CACHE_SCHEMA = 1


class SeriesCueRunError(RuntimeError):
    """シリーズcue経路の実行が、材料か契約の問題で成立しない。"""


@dataclass(frozen=True)
class EpisodeInput:
    """1話ぶんの入力。ラベルは記録の中で話を指す名前。"""

    label: str
    path: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episodes(episodes: Mapping[str, Any] | Iterable[Any], *,
              minimum: int = 2) -> list[EpisodeInput]:
    items: list[EpisodeInput] = []
    if isinstance(episodes, Mapping):
        pairs = episodes.items()
    else:
        pairs = [(item.label, item.path) for item in episodes]
    for label, path in pairs:
        if not isinstance(label, str) or not label.strip():
            raise SeriesCueRunError("話のラベルが未指定です")
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise SeriesCueRunError(f"入力が見つかりません: {resolved}")
        items.append(EpisodeInput(label.strip(), resolved))
    if len({item.label for item in items}) != len(items):
        raise SeriesCueRunError("話のラベルが重複しています")
    if len(items) < minimum:
        raise SeriesCueRunError(
            "話をまたぐ再演を見るには2話以上が必要です。1話で行うなら"
            "`include_within_episode` を宣言してください")
    return items


def _pair_matches(first, second) -> list[identity.SharedOccurrence]:
    """製品経路と同じく両方の向きで照合し、1組へまとめる。"""
    rows: list[list] = []
    for item in identity.shared_recordings(first, second):
        rows.append([item.first_start_seconds, item.first_end_seconds,
                     item.second_start_seconds, item.second_end_seconds,
                     item.waveform_explained or 0.0, item.blocks])
    for item in identity.shared_recordings(second, first):
        rows.append([item.second_start_seconds, item.second_end_seconds,
                     item.first_start_seconds, item.first_end_seconds,
                     item.waveform_explained or 0.0, item.blocks])
    return identity.merge_shared(rows)


def _self_matches(signal) -> list[identity.SharedOccurrence]:
    """同じ話の中で同じ録音が繰り返されている組。自分自身との重なりは数えない。

    1話しか無くても、同じ録音が2回鳴っていれば重ね合わせの材料になる。
    """
    kept: list[list] = []
    for item in identity.shared_recordings(signal, signal):
        first = (item.first_start_seconds, item.first_end_seconds)
        second = (item.second_start_seconds, item.second_end_seconds)
        if series_reuse.overlap_share(first, second) >= SELF_OVERLAP_SHARE:
            continue
        if min(first[1] - first[0], second[1] - second[0]) < identity.MIN_SECONDS:
            continue
        kept.append([first[0], first[1], second[0], second[1],
                     item.waveform_explained or 0.0, item.blocks])
    return identity.merge_shared(kept)


def signal_sha256(signal) -> str:
    """照合に使う信号そのもののhash。復号の仕方が変われば別の信号として扱う。"""
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(signal, dtype=np.float32).tobytes()).hexdigest()


def _match_intent() -> dict:
    """照合結果の再利用を決める条件。照合の実装か数値設定が変われば保存物を使わない。"""
    return {
        "schema_version": MATCH_CACHE_SCHEMA,
        "matching_settings_id": identity.matching_settings_id(),
        "identity_sha256": sha256(Path(identity.__file__)),
        "pairing_sha256": hashlib.sha256("\n".join(
            inspect.getsource(function) for function in (_pair_matches, _self_matches)
        ).encode("utf-8")).hexdigest(),
        "self_overlap_share": SELF_OVERLAP_SHARE,
    }


def _match_cache_path(cache_dir: Path, kind: str, keys: tuple[str, ...], intent: dict) -> Path:
    name = hashlib.sha256(json.dumps(
        {"kind": kind, "signals": list(keys), "intent": intent},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"series-matches-{name}.json"


def _cached_matches(cache_dir: Path | None, kind: str, keys: tuple[str, ...], compute,
                    intent: dict | None = None) -> tuple[list[identity.SharedOccurrence], bool]:
    """同じ信号の組・同じ条件なら保存した照合結果を読み直す。壊れた保存物は使わない。

    話を1話足すと組は増えるが、既にある組の照合はやり直さない。
    返り値の2つ目は保存物を使ったかどうか。
    """
    if cache_dir is None:
        return compute(), False
    intent = _match_intent() if intent is None else intent
    path = _match_cache_path(cache_dir, kind, keys, intent)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("type") == MATCH_CACHE_TYPE and payload.get("kind") == kind
                and payload.get("signals") == list(keys) and payload.get("intent") == intent):
            return [identity.SharedOccurrence(**item) for item in payload["matches"]], True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    found = compute()
    payload = {"type": MATCH_CACHE_TYPE, "kind": kind, "signals": list(keys), "intent": intent,
               "matches": [item.to_dict() for item in found]}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        # 保存できないことは照合の結論を変えない。次回に照合し直すだけ。
        temporary.unlink(missing_ok=True)
    return found, False


def _released_spans(signals, reference_paths, cache_dir):
    spans = {label: [] for label in signals}
    if not reference_paths:
        return spans, 0
    index = identity.ReferenceIndex.build(list(reference_paths), cache_dir=cache_dir)
    for label, signal in signals.items():
        for match in identity.match(signal, index):
            if match.identity_state == "known_exact":
                spans[label].append((match.query_start_seconds, match.query_end_seconds))
    return spans, len(reference_paths)


def mutual_explained(grouped: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, float]:
    """使用箇所ごとに「他の回にどれだけ説明されるか」を測る。

    参照録音を使わずに抽出元を選ぶための値。1回しか無いまとまりは測れないので飛ばす。
    """
    scores: dict[str, float] = {}
    for group in grouped["groups"]:
        occurrences = group.get("occurrences") or []
        if len(occurrences) < 2:
            continue
        segments: dict[str, Any] = {}
        for occurrence in occurrences:
            path = paths.get(occurrence["source_id"])
            if path is None:
                continue
            start = max(0.0, occurrence["start_seconds"] - MUTUAL_PAD_SECONDS)
            duration = (occurrence["end_seconds"] - occurrence["start_seconds"]
                        + 2 * MUTUAL_PAD_SECONDS)
            identifier = series_cues.occurrence_id(occurrence["source_id"],
                                                   occurrence["start_seconds"])
            segments[identifier] = identity.decode_mono(path, start=start, duration=duration)
        if len(segments) < 2:
            continue
        pairwise: dict[str, float] = {}
        for first, second in itertools.permutations(sorted(segments), 2):
            query, reference = segments[first], segments[second]
            if min(query.size, reference.size) < identity.MIN_SECONDS * identity.SAMPLE_RATE:
                pairwise[f"{first}|{second}"] = 0.0
                continue
            found = identity.shared_recordings(reference, query)
            values = [item.waveform_explained or 0.0 for item in found]
            pairwise[f"{first}|{second}"] = float(max(values)) if values else 0.0
        scores.update(occurrence_quality.mutual_explained_scores(pairwise, sorted(segments)))
    return scores


def _cut(path: Path, decoded: Path, span: Mapping[str, Any], target: Path) -> dict[str, Any]:
    """原本から区間を切り出す。`decoded` は製品と同じ入力準備を通した読み取り用の姿。"""
    import numpy as np
    import soundfile as sf

    before = sha256(path)
    with sf.SoundFile(str(decoded)) as stream:
        rate = stream.samplerate
        stream.seek(int(round(span["start_seconds"] * rate)))
        frames = int(round((span["end_seconds"] - span["start_seconds"]) * rate))
        data = stream.read(frames, dtype="float32", always_2d=True)
    if not data.size or not np.isfinite(data).all():
        raise SeriesCueRunError(f"切り出した音声が不正です: {span['recording_key']}")
    sf.write(str(target), data, rate, subtype="PCM_24")
    if sha256(path) != before:
        raise SeriesCueRunError("切り出し中に原本が変わりました")
    return {
        "artifact": {"path": str(target.resolve()), "sha256": sha256(target),
                     "frames": int(data.shape[0]), "channels": int(data.shape[1]),
                     "sample_rate": rate, "representation": REPRESENTATION,
                     "publication_state": PUBLICATION_STATE},
        "source": {"path": str(path), "sha256": before},
    }


def _read_window(decoded: Path, start_seconds: float, frames: int, channels: int):
    """復号済みの姿から、指定の位置・長さで読む。素材の外は無音で埋める。"""
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(str(decoded)) as stream:
        rate = stream.samplerate
        begin = int(round(start_seconds * rate))
        out = np.zeros((frames, channels), dtype=np.float64)
        if begin >= len(stream):
            return out, rate
        stream.seek(max(0, begin))
        data = stream.read(frames + min(0, begin), dtype="float64", always_2d=True)
        if data.shape[1] > channels:
            data = data[:, :channels]
        offset = max(0, -begin)
        length = min(frames - offset, data.shape[0])
        if length > 0:
            out[offset:offset + length, :data.shape[1]] = data[:length]
    return out, rate


def estimate_span_music(
    span: Mapping[str, Any], prepared: Mapping[str, Path],
    stack_recipe: occurrence_stack.OccurrenceStackRecipe, *,
    start_pad_seconds: float,
) -> dict[str, Any] | None:
    """同じ録音の他の回を使って、その区間に鳴っている音楽を推定する。

    他の回が一つも無ければ推定しない（`None`）。分離modelは使わない。
    位置のずれは `occurrence_stack` の位置合わせが詰める。
    """
    import numpy as np
    import soundfile as sf

    others_meta = span.get("other_occurrences") or []
    if not others_meta or span["source_id"] not in prepared:
        return None
    source = prepared[span["source_id"]]
    info = sf.info(str(source))
    rate, channels = int(info.samplerate), int(info.channels)
    frames = int(round((span["end_seconds"] - span["start_seconds"]) * rate))
    if frames < 2:
        return None
    target, _ = _read_window(source, span["start_seconds"], frames, channels)

    others = []
    for item in others_meta:
        label = item["source_id"]
        if label not in prepared:
            continue
        # 対応する位置は、同じ余裕を付けた他の回の始まり。
        window, _ = _read_window(
            prepared[label], item["start_seconds"] - start_pad_seconds, frames, channels)
        if np.any(window):
            others.append(window)
    if not others:
        return None
    result = occurrence_stack.estimate_music(target, others, stack_recipe, sample_rate=rate)
    return {"result": result, "sample_rate": rate,
            "peak": float(np.max(np.abs(result["music"]))) if result["music"].size else 0.0}


def _write_music_estimate(
    span: Mapping[str, Any], prepared: Mapping[str, Path],
    stack_recipe: occurrence_stack.OccurrenceStackRecipe, music_dir: Path, *,
    start_pad_seconds: float,
) -> dict[str, Any] | None:
    """推定した音楽を保存し、来歴を返す。他の回が無ければ保存しない。

    推定は元mixの一部を線形に取り出したもので、振幅は元のまま。丸めや正規化で
    値を変えないよう、浮動小数のWAVで保存する。**解析用の派生物**であり、
    正式なBGM成果物ではない。
    """
    import soundfile as sf

    estimate = estimate_span_music(
        span, prepared, stack_recipe, start_pad_seconds=start_pad_seconds)
    if estimate is None:
        return None
    result = estimate["result"]
    name = (f"{span['recording_key']}-{span['source_id']}-"
            f"{span['start_seconds']:.0f}-music.wav")
    target = music_dir / name
    sf.write(str(target), result["music"], estimate["sample_rate"], subtype="FLOAT")
    return {
        "path": str(target.resolve()), "sha256": sha256(target), "file": name,
        "recipe_id": result["recipe_id"], "recipe": result["recipe"],
        "occurrence_count": result["occurrence_count"],
        "combined_explained": result["combined_explained"],
        "energy_share": result["energy_share"],
        "meets_minimum_energy_share": result["meets_minimum_energy_share"],
        "used": result["used"], "peak": round(estimate["peak"], 6),
        "representation": "noncanonical_analysis_estimate",
        "publication_state": PUBLICATION_STATE,
        "claim_boundary": result["claim_boundary"],
    }


def run_series_cues(
    episodes: Mapping[str, Any] | Iterable[Any],
    recipe: series_cues.SeriesCueRecipe,
    out_dir: Path, *,
    reference_paths: Iterable[Path] | None = None,
    cache_dir: Path | None = None,
    measurements: Iterable[Any] | None = None,
    rank_by_mutual_explained: bool = True,
    stack_recipe: occurrence_stack.OccurrenceStackRecipe | None = None,
) -> Path:
    """シリーズの原本mixから未発売cueの候補を切り出し、記録の場所を返す。"""
    from contextlib import ExitStack

    from lss.core import audio_input

    recipe.validate()
    items = _episodes(episodes, minimum=1 if recipe.include_within_episode else 2)
    paths = {item.label: item.path for item in items}
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    signals = {item.label: identity.decode_mono(item.path) for item in items}
    # 長さは照合に使った信号から取る。素材をもう一度開かない。
    lengths = {label: signal.size / identity.SAMPLE_RATE for label, signal in signals.items()}

    released, reference_count = _released_spans(signals, list(reference_paths or ()), cache_dir)

    signal_keys = {label: signal_sha256(signal) for label, signal in signals.items()}
    intent = _match_intent() if cache_dir is not None else None
    reused = 0
    computed = 0

    def matches(kind: str, *labels: str):
        nonlocal reused, computed
        compute = ((lambda: _self_matches(signals[labels[0]])) if kind == "self"
                   else (lambda: _pair_matches(signals[labels[0]], signals[labels[1]])))
        found, hit = _cached_matches(
            cache_dir, kind, tuple(signal_keys[label] for label in labels), compute, intent)
        reused += 1 if hit else 0
        computed += 0 if hit else 1
        return found

    pairs = []
    within = 0
    if recipe.include_within_episode:
        for label in sorted(paths):
            for match in matches("self", label):
                pairs.append({
                    "first": label, "first_start_seconds": match.first_start_seconds,
                    "first_end_seconds": match.first_end_seconds, "second": label,
                    "second_start_seconds": match.second_start_seconds,
                    "second_end_seconds": match.second_end_seconds,
                    "waveform_explained": min(1.0, max(0.0, match.waveform_explained or 0.0)),
                })
                within += 1
    for first, second in itertools.combinations(sorted(paths), 2):
        for match in matches("pair", first, second):
            pairs.append({
                "first": first, "first_start_seconds": match.first_start_seconds,
                "first_end_seconds": match.first_end_seconds, "second": second,
                "second_start_seconds": match.second_start_seconds,
                "second_end_seconds": match.second_end_seconds,
                "waveform_explained": min(1.0, max(0.0, match.waveform_explained or 0.0)),
            })

    scores = None
    if rank_by_mutual_explained:
        grouped = series_reuse.group_recurrence(pairs, recipe.reuse, released_spans=released)
        scores = mutual_explained(grouped, paths)

    plan = series_cues.plan_cues(
        pairs, recipe, released_spans=released, source_seconds=lengths,
        mutual_explained=scores)

    audio_dir = out / CANDIDATE_DIRNAME
    audio_dir.mkdir(exist_ok=True)
    saved = []
    music_dir = out / MUSIC_DIRNAME
    estimated = 0
    # 同じ話の区間はまとめて切り出す。話ごとの復号を1回で済ませる。
    with ExitStack() as stack:
        prepared: dict[str, Path] = {}

        def ready(label: str) -> Path:
            if label not in prepared:
                prepared[label] = Path(stack.enter_context(
                    audio_input.prepare_input(paths[label])))
            return prepared[label]

        for span in plan["spans"]:
            label = span["source_id"]
            name = f"{span['recording_key']}-{label}-{span['start_seconds']:.0f}.flac"
            entry = {**span, **_cut(paths[label], ready(label), span, audio_dir / name),
                     "file": name}
            if stack_recipe is not None:
                for item in span.get("other_occurrences") or []:
                    if item["source_id"] in paths:
                        ready(item["source_id"])
                music_dir.mkdir(exist_ok=True)
                entry["music_estimate"] = _write_music_estimate(
                    span, prepared, stack_recipe, music_dir,
                    start_pad_seconds=recipe.selection.start_pad_seconds)
                estimated += 1 if entry["music_estimate"] else 0
            else:
                entry["music_estimate"] = None
            entry["marks"] = series_cues.marks_for(
                span.get("explained_median"),
                (entry.get("music_estimate") or {}).get("energy_share"), recipe)
            saved.append(entry)

    # 推定が元mixをほとんど残していない録音は、推定ではなく原本の区間で判定する。
    untrusted = [item["recording_key"] for item in saved
                 if (item.get("music_estimate") or {}).get(
                     "meets_minimum_energy_share") is False]
    marked = [item["recording_key"] for item in saved
              if series_cues.MARK_STOCK_EFFECT in (item.get("marks") or ())]
    screened = series_cues.apply_content_guard(
        plan, recipe, measurements, untrusted_recording_keys=untrusted,
        marked_recording_keys=marked)
    rejected_of = {item["recording_key"]: item["reason"] for item in screened["rejected"]}
    for item in saved:
        item["guard_reason"] = rejected_of.get(item["recording_key"])
        item["guard_state"] = screened["guard_state"]

    payload = {
        "type": "bgm_series_cue_run",
        "schema_version": series_cues.SCHEMA_VERSION,
        "status": "COMPLETED_HELD_CANDIDATES",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recipe_id": plan["recipe_id"],
        "stage_ids": plan["stage_ids"],
        "matching_settings_id": identity.matching_settings_id(),
        "episodes": {label: {"path": str(path), "sha256": sha256(path),
                             "seconds": round(lengths[label], 3)}
                     for label, path in sorted(paths.items())},
        "reference_count": reference_count,
        "released_spans_per_episode": {label: len(spans) for label, spans in sorted(released.items())},
        "occurrence_ranking_basis": plan["occurrence_ranking_basis"],
        "within_episode_pairs": within,
        "match_cache": {"used": cache_dir is not None, "reused": reused, "computed": computed},
        "include_within_episode": recipe.include_within_episode,
        "group_count": plan["grouped"]["group_count"],
        "released_group_count": plan["grouped"]["released_count"],
        "identical_reuse_count": plan["grouped"]["unreleased_identical_reuse"],
        "span_count": plan["span_count"],
        "music_estimate_recipe_id": (None if stack_recipe is None
                                     else stack_recipe.recipe_id),
        "music_estimate_count": estimated,
        "marked_stock_effect_count": sum(
            1 for item in saved if series_cues.MARK_STOCK_EFFECT in (item.get("marks") or ())),
        "music_estimate_trusted_count": sum(
            1 for item in saved
            if (item.get("music_estimate") or {}).get("meets_minimum_energy_share")),
        "guard_state": screened["guard_state"],
        "guard_target": screened["guard_target"],
        "guard_targets_used": screened.get("guard_targets_used"),
        "untrusted_estimate_recording_keys": screened.get("untrusted_recording_keys", untrusted),
        "held_count": screened["held_count"],
        "guard_rejected_count": screened["rejected_count"],
        "publication_state": PUBLICATION_STATE,
        "promotion_allowed": False,
        "holdout_consumed": False,
        "candidates": saved,
        "skipped": plan["selection"]["skipped"],
        "claim_boundary": list(plan["claim_boundary"]) + list(screened["claim_boundary"]),
    }
    record = out / RECORD_NAME
    with record.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def read_result(record_path: Path) -> dict[str, Any]:
    """保存した記録を読み、経路の結果として成立しているか確かめる。"""
    path = Path(record_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SeriesCueRunError(f"シリーズcueの記録を読めません: {path}") from exc
    if not isinstance(payload, dict) or payload.get("type") != "bgm_series_cue_run":
        raise SeriesCueRunError("シリーズcueの記録ではありません")
    if payload.get("publication_state") != PUBLICATION_STATE or payload.get("promotion_allowed"):
        raise SeriesCueRunError("非公開の候補としての記録になっていません")
    for candidate in payload.get("candidates") or []:
        artifact = candidate.get("artifact") or {}
        target = Path(artifact.get("path", ""))
        if not target.is_file() or sha256(target) != artifact.get("sha256"):
            raise SeriesCueRunError(f"候補の音声が記録と一致しません: {target}")
    return payload
