"""複数の番組の候補runをまたいで、同じ録音の区間と候補のつながりを別記録として付ける。

1本ずつの `identity.json`（手元の参照録音との一致）では、リリースされていない劇伴が別の話に
再登場しても分からない。ここは完了した番組runの組ごとに、記録どおりの分離成果物どうしを照合し、
同じ録音の区間（`identity.shared_recordings`）を元メディアの時刻で残す。

- runと `identity.json` は変更しない。出力は指定したフォルダの `series_identity.json` だけ。
- 候補は、既知録音との一致と別の話での再登場の境目で分けた子候補として並べ、判断材料を付ける。
- 候補どうしのつながり（`identity.link_occurrences`）は、統合・削除をしない。
- 同じ入力（runとidentity記録の内容、照合の設定と実装）なら再計算しない。組ごとの照合結果は
  信号のSHA-256の組で保存し、runを足しても既存の組を照合し直さない。
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from lss.bgm import identity, identity_run
from lss.bgm.cue_extent import CueExtentRecipe
from lss.bgm.recurring_extent import read_extents_for_candidates

RECORD_NAME = "series_identity.json"
RECORD_SCHEMA = 1
CHUNK_STEP_RATIO = 0.75
# 候補が一曲かどうかの読み取りを、結果を見る前に固定する。既知録音の一致も、別の回の
# 再登場も、同じ強さ・覆い・ずれの広がりで読む。
CUE_EXTENT_RECIPE = CueExtentRecipe(
    minimum_similarity=identity.MIN_BLOCK_EXPLAINED,
    minimum_single_recording_coverage=0.8,
    maximum_offset_spread_seconds=0.5,
)


class SeriesIdentityError(RuntimeError):
    """番組runの組に共通録音の証拠を付けられない。"""


def _sha256(path: Path) -> str:
    return identity_run._sha256(path)


def _episode(run_path: Path) -> dict:
    run_path = Path(run_path).resolve()
    run_bytes = run_path.read_bytes()
    run = json.loads(run_bytes)
    if not identity_run.is_annotatable(run):
        raise SeriesIdentityError(f"完了した番組の候補runではありません: {run_path}")
    run_sha256 = hashlib.sha256(run_bytes).hexdigest()
    signal_path, signal_kind, signal_sha256 = identity_run._query_signal(run)
    source = run.get("source") or {}
    matches: list[identity.RecordingMatch] = []
    identity_record = run_path.parent / identity_run.RECORD_NAME
    identity_sha256 = None
    if identity_record.is_file():
        try:
            record_bytes = identity_record.read_bytes()
            record = json.loads(record_bytes)
            # 古い条件や別の信号で得た一致から、候補を既知曲扱いしない。
            if identity_run.is_current_annotation(
                    record, run_sha256=run_sha256,
                    query_signal={"kind": signal_kind, "sha256": signal_sha256}):
                identity_run.verify_annotation_inputs(
                    run_path, run, record["intent"], signal_path, signal_sha256,
                )
                matches = [identity.RecordingMatch(**item) for item in record.get("matches") or []]
                identity_sha256 = hashlib.sha256(record_bytes).hexdigest()
        except (OSError, ValueError, TypeError, KeyError, identity_run.IdentityRunError):
            # 任意の付加記録が壊れていても、元の未解決候補と共通録音の探索は保持する。
            matches = []
    return {
        "key": str(source.get("source_id") or source.get("sha256")),
        "run_path": run_path, "run": run, "run_sha256": run_sha256,
        "signal_path": signal_path, "signal_kind": signal_kind, "signal_sha256": signal_sha256,
        "offset": identity_run.timeline_offset(run),
        "matches": matches, "identity_sha256": identity_sha256,
        "identity_intent": record["intent"] if identity_sha256 is not None else None,
    }


def _verify_episodes(episodes: list[dict]) -> None:
    """番組・信号・利用した一致の証拠を、保存前にも確認する。"""
    try:
        for episode in episodes:
            if (_sha256(episode["run_path"]) != episode["run_sha256"]
                    or _sha256(episode["signal_path"]) != episode["signal_sha256"]):
                raise SeriesIdentityError("照合中に番組runまたは信号が変更されています")
            identity_run.candidate_sets(episode["run"])
            if episode["identity_sha256"] is not None:
                path = episode["run_path"].parent / identity_run.RECORD_NAME
                if _sha256(path) != episode["identity_sha256"]:
                    raise SeriesIdentityError("照合中に利用した録音一致の記録が変更されています")
                identity_run.verify_annotation_inputs(
                    episode["run_path"], episode["run"], episode["identity_intent"],
                    episode["signal_path"], episode["signal_sha256"],
                )
    except (OSError, identity_run.IdentityRunError) as exc:
        raise SeriesIdentityError("照合中に入力または候補記録が失われたか変更されています") from exc


def _matching_revision() -> dict:
    return {
        "fingerprint_settings_id": identity.fingerprint_settings_id(),
        "matching_settings_id": identity.matching_settings_id(),
        "method_version": identity.METHOD_VERSION,
        "chunk_step_ratio": CHUNK_STEP_RATIO,
        "implementation_sha256": _sha256(Path(identity.__file__)),
        "stage_sha256": _sha256(Path(__file__)),
        "cue_extent_recipe_id": CUE_EXTENT_RECIPE.recipe_id,
    }


def _pair_cache_path(cache_dir: Path, first: dict, second: dict) -> Path:
    name = hashlib.sha256(
        f"{first['signal_sha256']}|{second['signal_sha256']}|{identity.matching_settings_id()}|"
        f"{_sha256(Path(__file__))}|{CHUNK_STEP_RATIO}".encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"shared-{name}.json"


def _payload_sha256(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return hashlib.sha256(json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _intact(payload: object) -> bool:
    try:
        return isinstance(payload, dict) and payload.get("payload_sha256") == _payload_sha256(payload)
    except (ValueError, TypeError):
        return False


def _pair_intent(first: dict, second: dict) -> dict:
    return {"schema_version": 2, "first_signal_sha256": first["signal_sha256"],
            "second_signal_sha256": second["signal_sha256"], **_matching_revision()}


def _valid_shared(item: identity.SharedOccurrence) -> bool:
    times = [item.first_start_seconds, item.first_end_seconds,
             item.second_start_seconds, item.second_end_seconds]
    return (all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in times)
            and 0 <= times[0] < times[1] and 0 <= times[2] < times[3]
            and abs((times[1] - times[0]) - (times[3] - times[2])) < 1e-6
            and isinstance(item.blocks, int) and not isinstance(item.blocks, bool) and item.blocks > 0
            and isinstance(item.waveform_explained, (int, float))
            and not isinstance(item.waveform_explained, bool)
            and math.isfinite(item.waveform_explained) and 0 <= item.waveform_explained <= 1)


def _read_pair(cache_dir: Path | None, first: dict, second: dict) -> list[identity.SharedOccurrence] | None:
    if cache_dir is None:
        return None
    path = _pair_cache_path(cache_dir, first, second)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (not _intact(payload) or payload.get("type") != "series_pair_cache"
                or payload.get("intent") != _pair_intent(first, second)):
            return None
        items = [identity.SharedOccurrence(**item) for item in payload["occurrences"]]
        return items if all(_valid_shared(item) for item in items) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_pair(cache_dir: Path | None, first: dict, second: dict, found: list[identity.SharedOccurrence]) -> None:
    if cache_dir is None:
        return
    path = _pair_cache_path(cache_dir, first, second)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not all(_valid_shared(item) for item in found):
        raise SeriesIdentityError("共通録音の区間・相関根拠が不正です")
    payload = {"type": "series_pair_cache", "intent": _pair_intent(first, second),
               "occurrences": [item.to_dict() for item in found]}
    payload["payload_sha256"] = _payload_sha256(payload)

    def verify() -> None:
        _verify_episodes([first, second])
        if payload["intent"] != _pair_intent(first, second):
            raise SeriesIdentityError("共通録音cacheの保存中に判定条件が変更されています")

    identity_run._write_annotation(path, payload, verify)


class _Signals:
    """番組の信号を必要なときだけ読み、直近の数本だけを保持する（長い連続番組でもメモリを抑える）。"""

    def __init__(self, episodes: list[dict], loader, keep: int = 3) -> None:
        self._by_sha = {episode["signal_sha256"]: episode for episode in episodes}
        self._loader = loader
        self._keep = keep
        self._cache: dict[str, object] = {}

    def get(self, signal_sha256: str):
        if signal_sha256 not in self._cache:
            if len(self._cache) >= self._keep:
                self._cache.pop(next(iter(self._cache)))
            self._cache[signal_sha256] = self._loader(self._by_sha[signal_sha256]["signal_path"])
        return self._cache[signal_sha256]


def _chunk_fingerprints(episode: dict, signals: _Signals, cache_dir: Path | None) -> list[tuple]:
    """番組の信号を区間に分けた指紋。`(名前, 所在, 識別子, 指紋)` の並びで、内容と設定ごとに保存する。"""
    size = int(identity.SHARED_CHUNK_SECONDS * identity.SAMPLE_RATE)
    items = []
    samples = None
    start = 0
    while True:
        chunk_key = hashlib.sha256(f"{episode['signal_sha256']}@{start}@{size}".encode("utf-8")).hexdigest()
        cached = identity._load_fingerprint(Path(cache_dir), chunk_key) if cache_dir is not None else None
        if cached is None:
            samples = signals.get(episode["signal_sha256"]) if samples is None else samples
            if start >= max(samples.size - identity.N_FFT, 1):
                break
            spec = identity.spectrogram(samples[start:start + size])
            hashes, times = identity.landmarks(*identity.peaks(spec))
            cached = (hashes, times, identity.band_profile(spec), spec.shape[0])
            if cache_dir is not None:
                identity._save_fingerprint(Path(cache_dir), chunk_key, cached)
        elif cached[3] == 0:
            break
        items.append((f"{start / identity.SAMPLE_RATE:.3f}", f"{episode['signal_sha256']}@{start}", chunk_key, cached))
        if cached[3] < size // identity.HOP - identity.N_FFT // identity.HOP:
            break  # 最後の区間
        # 区間の境目で共通録音が短く分かれて取り逃がさないよう、区間を重ねる（重なった一致は同じずれでつなぐ）。
        start += int(size * CHUNK_STEP_RATIO)
    return items


def _shared_pairs(episodes: list[dict], *, cache_dir: Path | None, loader) -> dict[tuple[str, str], list[identity.SharedOccurrence]]:
    """番組の組ごとの共通録音。組を1つずつ照合せず、他の番組の区間をまとめた索引へ各番組を1回だけ照合する。

    番組数nに対して照合はn回で済む。指紋の偏りで向きによって見つかる区間が変わるため、組の両方の向きの
    結果を合わせる。既に保存した組は索引から外す。組の向き（1本目・2本目）は並び順。
    """
    revision = _matching_revision()
    signals = _Signals(episodes, loader)
    fingerprints: dict[str, list[tuple]] = {}
    result: dict[tuple[str, str], list[identity.SharedOccurrence]] = {}
    order = {episode["key"]: position for position, episode in enumerate(episodes)}
    pending_pairs = set()
    for first, second in ((a, b) for i, a in enumerate(episodes) for b in episodes[i + 1:]):
        cached = _read_pair(cache_dir, first, second)
        if cached is None:
            pending_pairs.add((first["key"], second["key"]))
        else:
            result[(first["key"], second["key"])] = cached
    rows: dict[tuple[str, str], list[list]] = {pair: [] for pair in pending_pairs}
    size = int(identity.SHARED_CHUNK_SECONDS * identity.SAMPLE_RATE)

    def chunk_loader(location: str):
        signal_sha256, start = location.rsplit("@", 1)
        return signals.get(signal_sha256)[int(start):int(start) + size]

    for query in episodes:
        others = [e for e in episodes if e is not query
                  and tuple(sorted((query["key"], e["key"]), key=order.get)) in pending_pairs]
        if not others:
            continue
        items, owner_of = [], {}
        for other in others:
            if other["signal_sha256"] not in fingerprints:
                fingerprints[other["signal_sha256"]] = _chunk_fingerprints(other, signals, cache_dir)
            for item in fingerprints[other["signal_sha256"]]:
                owner_of[item[2]] = (other, int(item[1].rsplit("@", 1)[1]) / identity.SAMPLE_RATE)
                items.append(item)
        index = identity.ReferenceIndex._from_fingerprints(items, chunk_loader)
        for m in identity.match(signals.get(query["signal_sha256"]), index):
            if m.identity_state != "known_exact":
                continue
            other, chunk_start = owner_of[m.reference_sha256]
            other_start = chunk_start + m.reference_start_seconds
            span = [m.query_start_seconds, m.query_end_seconds, other_start,
                    other_start + (m.query_end_seconds - m.query_start_seconds)]
            if order[query["key"]] < order[other["key"]]:
                pair, first_span, second_span = (query["key"], other["key"]), span[:2], span[2:]
            else:
                pair, first_span, second_span = (other["key"], query["key"]), span[2:], span[:2]
            rows[pair].append([*first_span, *second_span, m.waveform_explained or 0.0, m.waveform_blocks])
    by_key = {episode["key"]: episode for episode in episodes}
    _verify_episodes(episodes)
    if _matching_revision() != revision:
        raise SeriesIdentityError("照合中に実装または判定条件が変更されています")
    for pair in pending_pairs:
        occurrences = identity.merge_shared(rows[pair])
        _write_pair(cache_dir, by_key[pair[0]], by_key[pair[1]], occurrences)
        result[pair] = occurrences
    return result


def _on_media_timeline(occurrence: identity.SharedOccurrence, first: dict, second: dict) -> identity.SharedOccurrence:
    """照合した信号の時刻を、各番組の元メディアの時刻へ直す。"""
    return identity.SharedOccurrence(
        round(occurrence.first_start_seconds + first["offset"], 3),
        round(occurrence.first_end_seconds + first["offset"], 3),
        round(occurrence.second_start_seconds + second["offset"], 3),
        round(occurrence.second_end_seconds + second["offset"], 3),
        occurrence.waveform_explained, occurrence.blocks)


def annotate_series(run_paths: list[Path], out_dir: Path, *, cache_dir: Path | None = None,
                    loader=identity.decode_mono) -> Path:
    """番組runの集まりに共通録音の証拠を付け、`out_dir/series_identity.json` のパスを返す。"""
    episodes = [_episode(path) for path in sorted({Path(p).resolve() for p in run_paths})]
    keys = [episode["key"] for episode in episodes]
    if len(episodes) < 2:
        raise SeriesIdentityError("共通録音を探すには2本以上の番組runが必要です")
    if len(set(keys)) != len(keys):
        raise SeriesIdentityError("同じ番組のrunが重複しています")
    try:
        candidate_sets = {e["key"]: identity_run.candidate_sets(e["run"]) for e in episodes}
    except identity_run.IdentityRunError as exc:
        raise SeriesIdentityError(str(exc)) from exc
    intent = {
        "schema_version": RECORD_SCHEMA,
        "episodes": [{"key": e["key"], "run_sha256": e["run_sha256"], "identity_sha256": e["identity_sha256"],
                      "signal_sha256": e["signal_sha256"]} for e in episodes],
        **_matching_revision(),
    }
    out_dir = Path(out_dir)
    record_path = out_dir / RECORD_NAME
    if record_path.is_file():
        try:
            cached = json.loads(record_path.read_text(encoding="utf-8"))
            if (_intact(cached) and cached.get("type") == "series_identity_annotation"
                    and cached.get("intent") == intent):
                _verify_episodes(episodes)
                if any(intent.get(key) != value for key, value in _matching_revision().items()):
                    raise SeriesIdentityError("再利用中に実装または判定条件が変更されています")
                return record_path
        except ValueError:
            pass

    by_key = {episode["key"]: episode for episode in episodes}
    shared = {pair: [_on_media_timeline(item, by_key[pair[0]], by_key[pair[1]]) for item in found]
              for pair, found in _shared_pairs(episodes, cache_dir=cache_dir, loader=loader).items()}
    recurring = identity.recurring_spans(shared)

    by_recipe: dict[str, dict[str, list[dict]]] = {}
    decisions: dict[str, dict[str, list[dict]]] = {}
    for episode in episodes:
        sets = candidate_sets[episode["key"]]
        for recipe_id, raw in sets.items():
            pieces = identity.split_candidates_at_matches(raw, episode["matches"],
                                                         recurring=recurring.get(episode["key"], []))
            read_extents_for_candidates(pieces, recurring.get(episode["key"], []),
                                        CUE_EXTENT_RECIPE)
            by_recipe.setdefault(recipe_id, {})[episode["key"]] = pieces
            decisions.setdefault(recipe_id, {})[episode["key"]] = identity.novelty_decisions(pieces)

    payload = {
        "type": "series_identity_annotation",
        "schema_version": RECORD_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "intent": intent,
        "episodes": [{"key": e["key"], "run_path": str(e["run_path"]),
                      "source_sha256": (e["run"].get("source") or {}).get("sha256"), "signal_kind": e["signal_kind"],
                      "known_matches_from_identity_record": e["identity_sha256"] is not None}
                     for e in episodes],
        "evidence_limits": [
            "a shared recording means the same recording is audible in both spans; semantic role is unresolved",
            "occurrence groups link candidates and never merge or delete them",
            "no shared recording is not evidence that a cue is unique or absent",
        ],
        "shared_recordings": [{"first": a, "second": b, "occurrences": [item.to_dict() for item in items]}
                              for (a, b), items in shared.items()],
        "activity_recipes": [
            {"recipe_id": recipe_id, "candidates": candidates, "novelty_decisions": decisions[recipe_id],
             "occurrence_groups": identity.link_occurrences(candidates, shared)}
            for recipe_id, candidates in sorted(by_recipe.items())
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["payload_sha256"] = _payload_sha256(payload)

    def verify() -> None:
        _verify_episodes(episodes)
        if any(intent.get(key) != value for key, value in _matching_revision().items()):
            raise SeriesIdentityError("照合中に実装または判定条件が変更されています")

    identity_run._write_annotation(record_path, payload, verify)
    return record_path
