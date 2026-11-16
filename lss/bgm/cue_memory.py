"""Product-owned memory of published BGM cues: which published cues, across sources, are the same recording.

The single-source BGM path decides each episode on its own. Once several episodes of a series are processed,
the same score recording is published again and again under unrelated provisional titles, each with its own
dialogue/effect residue, and nothing tells the listener that two files are the same piece of music. This module
accumulates the CurrentBest cues that the Product actually published and records verified same-recording
relations between them, so the listening presentation can say so.

- A member is one published CurrentBest cue (asset ``bgm-cue:<cue key>``) with its public path and PCM identity.
  Observing a source replaces that source's member set with its current outputs; replaced members retire.
- A new or changed member is compared with every other present member (other sources and the same source) using
  the existing recording matcher (`identity.match`: landmark fingerprint plus waveform verification). Only a
  ``known_exact`` relation whose waveform explanation is at least ``MIN_EXPLAINED`` becomes a link. Matching two
  separated estimates is the easy direction of that matcher; it never compares with an original mixture.
- Two resolved single cues (one recording each, by the Product's own extent decision) group when they share a
  verified passage of at least ``RESOLVED_MIN_SECONDS`` and ``RESOLVED_MIN_SHARE`` of the shorter cue. A cue whose
  extent is unresolved may contain two recordings: it groups only by more than half of both cues or nearly all of
  the shorter one, and only through links that meet one region of it, so transitivity cannot merge two different
  pieces of music through it.
- The same recording is often played at another tempo (pitch kept): editors fit a score to a scene by time
  stretching it. The landmark matcher needs one constant offset and misses those, so a pair without a constant-
  offset link is checked again from the matcher's own candidate alignments with `stretch_match` (lag tracked
  window by window, one straight lag line = one tempo ratio, cleared against the same tracker's chance level). Such
  a link records its ``rate``; its two ranges have different lengths. It groups like any other link; sample-level
  consumers that need one constant offset (`recurrence_asset`) leave it out.
- A group of two or more members gets a stable number (``劇伴#N``), assigned once and kept.

Nothing here publishes, suppresses, deletes or changes audio, CurrentBest or Cue identity. A relation is
presentation and provenance only: a previously published extraction is not an independently available release,
so recurrence never makes a new occurrence redundant. A missing or changed member file is not evidence of
anything; it is simply not compared. Failure to match leaves the cue as it would be without memory.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

SCHEMA_VERSION = 1
METHOD = "published-cue-recurrence-v2"
DIRECTORY = "bgm-cue-memory"
RECORD_NAME = "memory.json"
LABEL_PREFIX = "劇伴#"
# Waveform share of one cue explained by the other over the matched run. Same recording heard through two
# different separations explains far more than chance (`identity` requires a margin over shifted nulls too).
MIN_EXPLAINED = 0.5
# Grouping when a cue may hold several recordings: more than half of both cues, or nearly all of the shorter one.
MUTUAL_SHARE = 0.5
CONTAINED_SHARE = 0.8
# Two resolved single cues: a verified shared passage this long, and this share of the shorter cue.
RESOLVED_MIN_SECONDS = 10.0
RESOLVED_MIN_SHARE = 0.25
# Links into an unresolved cue meet "one region" of it when their ranges there overlap at least this long.
REGION_OVERLAP_SECONDS = 3.0
# Candidate alignments (by fingerprint votes) from which a tempo-changed passage is tracked, per pair.
STRETCH_SEEDS = 4


class CueMemoryError(RuntimeError):
    """The memory record is unreadable or does not follow this contract."""


def recipe() -> dict:
    """What decides a link. A different value starts the links over (members are kept and re-compared)."""
    from lss.bgm import identity

    return {"method": METHOD, "min_explained": MIN_EXPLAINED, "mutual_share": MUTUAL_SHARE,
            "contained_share": CONTAINED_SHARE, "resolved_min_seconds": RESOLVED_MIN_SECONDS,
            "resolved_min_share": RESOLVED_MIN_SHARE, "region_overlap_seconds": REGION_OVERLAP_SECONDS, "grouping": "resolved-shared-or-unresolved-one-region-v2",
            "matching_settings_id": identity.matching_settings_id(), "stretch": _stretch_recipe()}


def _stretch_recipe() -> dict:
    from lss.bgm import stretch_match as sm

    return {"method": sm.METHOD, "window_seconds": sm.WINDOW_SECONDS, "hop_seconds": sm.HOP_SECONDS,
            "step_search_seconds": sm.STEP_SEARCH_SECONDS, "min_window_explained": sm.MIN_WINDOW_EXPLAINED,
            "max_rate_deviation": sm.MAX_RATE_DEVIATION, "line_tolerance_seconds": sm.LINE_TOLERANCE_SECONDS,
            "min_run_windows": sm.MIN_RUN_WINDOWS, "allowed_misses": sm.ALLOWED_MISSES,
            "min_mean_explained": sm.MIN_MEAN_EXPLAINED, "min_margin": sm.MIN_MARGIN, "seeds": STRETCH_SEEDS}


def recipe_id() -> str:
    body = json.dumps(recipe(), sort_keys=True, separators=(",", ":"))
    return f"bgm-cue-memory-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def root(state_root: Path) -> Path:
    return Path(state_root) / DIRECTORY


def _record_path(state_root: Path) -> Path:
    return root(state_root) / RECORD_NAME


def _empty() -> dict:
    return {"type": "bgm_cue_memory", "schema_version": SCHEMA_VERSION, "recipe_id": recipe_id(),
            "recipe": recipe(), "members": {}, "retired": [], "links": [], "labels": {}, "next_label": 1}


def read(state_root: Path) -> dict:
    path = _record_path(state_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty()
    except (OSError, ValueError) as exc:
        raise CueMemoryError(f"BGMの記憶を読み込めません: {path}") from exc
    if (not isinstance(data, dict) or data.get("type") != "bgm_cue_memory"
            or data.get("schema_version") != SCHEMA_VERSION or not isinstance(data.get("members"), dict)
            or not isinstance(data.get("links"), list) or not isinstance(data.get("labels"), dict)):
        raise CueMemoryError(f"BGMの記憶の形式が違います: {path}")
    if data.get("recipe_id") != recipe_id():
        # Another matching rule: keep who was published, compare everyone again under the current rule.
        data = {**data, "recipe_id": recipe_id(), "recipe": recipe(), "links": [],
                "members": {key: {**value, "compared_with": []} for key, value in data["members"].items()}}
    return data


def _write(state_root: Path, data: dict) -> None:
    path = _record_path(state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
    with NamedTemporaryFile(dir=path.parent, prefix=".lss-memory-", suffix=".tmp", delete=False) as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------- matching

def _fingerprint_path(state_root: Path, pcm_sha256: str) -> Path:
    from lss.bgm import identity

    return root(state_root) / "fingerprints" / f"{pcm_sha256}-{identity.fingerprint_settings_id()}.npz"


def _fingerprint(state_root: Path, pcm_sha256: str, samples: np.ndarray | None, path: Path):
    """Fingerprint keyed by the PCM identity (tags can change the file bytes without changing the music)."""
    from lss.bgm import identity

    place = _fingerprint_path(state_root, pcm_sha256)
    try:
        with np.load(place) as data:
            return (data["hashes"].astype(np.uint32), data["times"].astype(np.int32),
                    data["profile"].astype(np.float32), int(data["frames"]))
    except (OSError, KeyError, ValueError):
        pass
    if samples is None:
        samples = identity.decode_mono(path)
    spec = identity.spectrogram(samples)
    hashes, times = identity.landmarks(*identity.peaks(spec))
    value = (hashes, times, identity.band_profile(spec), int(spec.shape[0]))
    place.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=place.parent, prefix=".fp-", suffix=".npz", delete=False) as stream:
        np.savez(stream, hashes=value[0], times=value[1], profile=value[2], frames=np.int64(value[3]))
        temporary = Path(stream.name)
    try:
        os.replace(temporary, place)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def _still_present(member: dict) -> bool:
    """The member's public file still carries the music it was registered with (tags may have changed)."""
    from lss.core import audio_identity

    path = Path(member["path"])
    try:
        stat = path.stat()
    except OSError:
        return False
    if [stat.st_size, stat.st_mtime_ns] == member.get("file_stat"):
        return True
    try:
        same = audio_identity.audio_identity(path)["pcm_sha256"] == member["pcm_sha256"]
    except Exception:  # noqa: BLE001 - an unreadable file is simply not compared
        return False
    if same:
        member["file_stat"] = [stat.st_size, stat.st_mtime_ns]
    return same


def _union(ranges: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    for low, high in sorted(ranges):
        if end is None or low > end:
            total += high - low
            end = high
        elif high > end:
            total += high - end
            end = high
    return total


def compare(query: dict, query_samples: np.ndarray, others: list[dict], state_root: Path) -> list[dict]:
    """Verified same-recording links from one member to the others (both sides in their own cue clocks)."""
    from lss.bgm import identity

    if not others:
        return []
    items = [(item["asset_id"], item["path"], item["asset_id"],
              _fingerprint(state_root, item["pcm_sha256"], None, Path(item["path"]))) for item in others]
    paths = {item["asset_id"]: item["path"] for item in others}
    index = identity.ReferenceIndex._from_fingerprints(
        [(name, paths[name], sha, fingerprint) for name, _, sha, fingerprint in items],
        lambda location: identity.decode_mono(Path(location)))
    by_id = {item["asset_id"]: item for item in others}
    found: dict[str, list] = {}
    seeds: dict[str, list] = {}
    candidates = identity.match(query_samples, index, collapse=False)
    for match in identity._collapse_alternatives(candidates):
        if (match.identity_state != "known_exact" or match.waveform_explained is None
                or match.waveform_explained < MIN_EXPLAINED):
            continue
        found.setdefault(match.reference_sha256, []).append(match)
    for match in candidates:
        if match.reference_sha256 not in found and match.measured_windows >= 2:
            seeds.setdefault(match.reference_sha256, []).append(match)
    links = _stretched_links(query, query_samples, seeds, by_id, index)
    for other_id, matches in found.items():
        other = by_id[other_id]
        query_ranges = [(m.query_start_seconds, m.query_end_seconds) for m in matches]
        other_ranges = [(m.reference_start_seconds, m.reference_start_seconds + m.query_end_seconds
                         - m.query_start_seconds) for m in matches]
        query_shared = min(_union(query_ranges), query["duration_seconds"])
        other_shared = min(_union(other_ranges), other["duration_seconds"])
        best = max(matches, key=lambda m: (m.query_end_seconds - m.query_start_seconds, m.waveform_explained))
        # Both sides in one order (by this cue's time): range i here is the same music as range i there.
        pairs = sorted(zip(query_ranges, other_ranges))
        links.append({
            "a": query["asset_id"], "b": other_id, "paired": True,
            "a_ranges": [[round(low, 3), round(high, 3)] for (low, high), _ in pairs],
            "b_ranges": [[round(low, 3), round(high, 3)] for _, (low, high) in pairs],
            "a_share": round(query_shared / query["duration_seconds"], 4),
            "b_share": round(other_shared / other["duration_seconds"], 4),
            "offset_seconds": best.offset_seconds, "waveform_explained": best.waveform_explained,
            "waveform_null_max": best.waveform_null_max, "votes": best.votes, "z": best.z,
            "basis": best.evidence_kind, "found_at": _now()})
    return links


def _stretched_links(query: dict, query_samples: np.ndarray, seeds: dict[str, list], by_id: dict,
                     index) -> list[dict]:
    """Links for pairs that share a passage at another tempo (no constant-offset link was verified)."""
    from lss.bgm import identity, stretch_match

    links = []
    for other_id, matches in seeds.items():
        other = by_id[other_id]
        matches = sorted(matches, key=lambda m: -m.votes)[:STRETCH_SEEDS]
        starts = [((m.query_start_seconds + m.query_end_seconds) / 2, m.offset_seconds) for m in matches]
        owner = next(position for position, ref in enumerate(index.references) if ref.sha256 == other_id)
        runs = stretch_match.stretched_matches(query_samples, index.samples(owner), starts,
                                               rate=identity.SAMPLE_RATE)
        if not runs:
            continue
        runs.sort(key=lambda run: run.query_start_seconds)
        query_ranges = [(run.query_start_seconds, run.query_end_seconds) for run in runs]
        other_ranges = [(run.other_start_seconds, run.other_end_seconds) for run in runs]
        best = max(runs, key=lambda run: (run.query_end_seconds - run.query_start_seconds, run.mean_explained))
        links.append({
            "a": query["asset_id"], "b": other_id, "paired": True,
            "a_ranges": [[round(low, 3), round(high, 3)] for low, high in query_ranges],
            "b_ranges": [[round(low, 3), round(high, 3)] for low, high in other_ranges],
            "a_share": round(min(_union(query_ranges), query["duration_seconds"]) / query["duration_seconds"], 4),
            "b_share": round(min(_union(other_ranges), other["duration_seconds"]) / other["duration_seconds"], 4),
            "rate": round(float(np.median([run.rate for run in runs])), 5),
            "offset_seconds": best.lag_at_start_seconds, "waveform_explained": best.mean_explained,
            "waveform_null_max": best.null_max, "votes": matches[0].votes, "z": matches[0].z,
            "basis": best.evidence_kind, "found_at": _now()})
    return links


def groups_link(link: dict, members: dict) -> bool:
    """Whether a link can say the two cues are the same recorded cue (not merely share a passage).

    Two resolved single cues are each one recording by the Product's own extent decision, so a verified shared
    passage of at least ``RESOLVED_MIN_SECONDS`` and ``RESOLVED_MIN_SHARE`` of the shorter one suffices.
    Otherwise (a cue that may contain several recordings) the link must cover more than half of both cues or
    nearly all of the shorter one; `groups` additionally keeps an unresolved cue's links to one region of it.
    """
    a, b = members.get(link["a"]), members.get(link["b"])
    if a is None or b is None:
        return False
    shorter, longer = (a, b) if a["duration_seconds"] <= b["duration_seconds"] else (b, a)
    shorter_share = link["a_share"] if shorter is a else link["b_share"]
    if not a.get("unresolved_extent") and not b.get("unresolved_extent"):
        shared = min(link["a_share"] * a["duration_seconds"], link["b_share"] * b["duration_seconds"])
        return shared >= RESOLVED_MIN_SECONDS and shorter_share >= RESOLVED_MIN_SHARE
    if link["a_share"] > MUTUAL_SHARE and link["b_share"] > MUTUAL_SHARE:
        return True
    return shorter_share >= CONTAINED_SHARE


def _overlaps(first: list, second: list) -> bool:
    """Two sets of ranges in one cue meet over a real passage, not at a boundary."""
    return sum(max(0.0, min(high, other_high) - max(low, other_low))
               for low, high in first for other_low, other_high in second) >= REGION_OVERLAP_SECONDS


def grouping_links(data: dict) -> list[dict]:
    """Links that group. An unresolved cue may hold two recordings, so its links must all meet one region of it:
    the region of its strongest link. A link into a disjoint part of it is kept as evidence but does not group,
    so two different pieces of music can never be merged through one unresolved cue."""
    members = data["members"]
    candidates = [link for link in data["links"] if groups_link(link, members)]
    rejected = set()
    for key, member in members.items():
        if not member.get("unresolved_extent"):
            continue
        own = []
        for number, link in enumerate(candidates):
            if key in (link["a"], link["b"]):
                ranges = link["a_ranges"] if link["a"] == key else link["b_ranges"]
                own.append((-sum(high - low for low, high in ranges), number, ranges))
        if not own:
            continue
        own.sort()
        region = own[0][2]
        rejected.update(number for _, number, ranges in own[1:] if not _overlaps(region, ranges))
    return [link for number, link in enumerate(candidates) if number not in rejected]


def groups(data: dict) -> dict[str, list[str]]:
    """Connected components of grouping links among present members, keyed by their earliest member."""
    members = data["members"]
    parent = {key: key for key in members}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for link in grouping_links(data):
        left, right = find(link["a"]), find(link["b"])
        if left != right:
            parent[right] = left
    components: dict[str, list[str]] = {}
    for key in members:
        components.setdefault(find(key), []).append(key)
    result = {}
    for keys in components.values():
        keys.sort(key=lambda key: (members[key]["order"], key))
        result[keys[0]] = keys
    return result


def _assign_labels(data: dict, components: dict[str, list[str]]) -> dict[str, str]:
    """A stable number per recurring group. Merged groups keep the earliest number; numbers are never reused."""
    labels = data["labels"]
    by_member = {member: number for number, entry in labels.items() for member in entry["members"]}
    assigned = {}
    for anchor, keys in sorted(components.items(), key=lambda item: data["members"][item[0]]["order"]):
        if len(keys) < 2:
            continue
        known = sorted({by_member[key] for key in keys if key in by_member}, key=int)
        number = known[0] if known else str(data["next_label"])
        if not known:
            data["next_label"] += 1
        for other in known[1:]:
            labels[other] = {**labels[other], "merged_into": number, "members": []}
        entry = labels.get(number) or {"created_at": _now()}
        labels[number] = {**entry, "members": sorted(set(entry.get("members") or []) | set(keys)),
                          "anchor": anchor}
        assigned[anchor] = number
    return assigned


# ---------------------------------------------------------------------------------------------------- observe

def _member(asset_id: str, item: dict, order: int) -> dict:
    stat = Path(item["path"]).stat()
    return {"asset_id": asset_id, "cue_key": item["cue_key"], "source_sha256": item["source_sha256"],
            "source_name": item["source_name"], "path": str(item["path"]), "pcm_sha256": item["pcm_sha256"],
            "duration_seconds": item["duration_seconds"], "start_seconds": item["start_seconds"],
            "unresolved_extent": bool(item.get("unresolved_extent")), "order": order,
            "file_stat": [stat.st_size, stat.st_mtime_ns], "registered_at": _now(), "compared_with": []}


def published_item(path: Path, *, accepted: dict | None = None) -> dict:
    """What memory needs from one published CurrentBest file (its identity comes from the accepted record or tags)."""
    from mutagen.flac import FLAC

    from lss.core import audio_identity

    path = Path(path)
    if accepted is not None:
        key, provenance = accepted["key"], accepted["provenance"]
    else:
        tags = FLAC(path)
        key = tags["LSS_BGM_CUE_KEY"][0]
        provenance = json.loads(tags["LSS_BGM_PROVENANCE"][0])
    audio = audio_identity.audio_identity(path)
    source = provenance["source"]
    return {"path": str(path), "cue_key": key, "source_sha256": source["sha256"],
            "source_name": Path(source.get("path") or "").stem, "pcm_sha256": audio["pcm_sha256"],
            "duration_seconds": audio["frames"] / audio["samplerate"],
            "start_seconds": provenance.get("start_seconds"),
            "unresolved_extent": (provenance.get("cue_extent") or {}).get("state") == "unresolved"}


def _sync_source(data: dict, source_sha256: str, items: list[dict]) -> None:
    """Make one source's members exactly its current published cues (replaced/removed ones retire)."""
    members = data["members"]
    current = {f"bgm-cue:{item['cue_key']}": item for item in items}
    for asset_id in [key for key, value in members.items()
                     if value["source_sha256"] == source_sha256 and key not in current]:
        data["retired"].append({**members.pop(asset_id), "retired_at": _now()})
    order = max([value["order"] for value in members.values()] + [0])
    for asset_id, item in current.items():
        existing = members.get(asset_id)
        if existing is not None and existing["pcm_sha256"] == item["pcm_sha256"] and existing["path"] == item["path"]:
            continue
        if existing is not None:
            data["links"] = [link for link in data["links"] if asset_id not in (link["a"], link["b"])]
            for value in members.values():
                # A replaced cue is compared again from both sides.
                value["compared_with"] = [key for key in value["compared_with"] if key != asset_id]
        order += 1
        members[asset_id] = _member(asset_id, item, existing["order"] if existing else order)
    data["links"] = [link for link in data["links"] if link["a"] in members and link["b"] in members]


def _current_best_sources(state_root: Path, skip: str) -> list[tuple[str, list[dict]]]:
    """Other sources' accepted cues, from their CurrentBest records (the authority for what is published).

    A cue that was published before this memory existed joins it here. A record or file that cannot be read,
    or whose audio is no longer what CurrentBest accepted, is left out (not evidence of anything).
    """
    from lss.bgm import current_best

    folder = Path(state_root) / current_best.DIRECTORY
    found = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        if path.name.endswith(".switch.json") or path.stem == skip:
            continue
        try:
            record = current_best.read(state_root, path.stem)
            rows = record["current"]["files"]
        except Exception:  # noqa: BLE001
            continue
        found.append((record.get("updated_at") or "", path.stem, rows))
    found.sort()
    return [(sha, rows) for _, sha, rows in found]


def _accepted_items(data: dict, sha: str, rows: list[dict]) -> list[dict] | None:
    items = []
    for row in sorted(rows, key=lambda row: row.get("start_seconds") or 0):
        member = data["members"].get(f"bgm-cue:{row.get('cue_key')}")
        if member is not None and member["pcm_sha256"] == row.get("pcm_sha256") and member["path"] == row.get("public_path"):
            items.append({name: member[name] for name in ("path", "cue_key", "source_sha256", "source_name",
                                                          "pcm_sha256", "duration_seconds", "start_seconds",
                                                          "unresolved_extent")})
            continue
        try:
            item = published_item(Path(row["public_path"]))
        except Exception:  # noqa: BLE001 - an unreadable accepted file is not compared
            continue
        if item["cue_key"] == row.get("cue_key") and item["pcm_sha256"] == row.get("pcm_sha256")                 and item["source_sha256"] == sha:
            items.append(item)
    return items


def observe(state_root: Path, source_sha256: str, items: list[dict], *, loader=None) -> dict:
    """Register one source's current published cues, compare them with the memory, return the recurrence view.

    ``items`` come from `published_item`. Other sources' accepted cues are synchronised from their CurrentBest
    records first, so cues published before this memory existed take part too. Every pair of present members is
    compared once. Returns ``{"recipe_id", "by_path", "changed_elsewhere", "state"}``: the recurrence of each of
    this source's cues by public path, and the other sources' cues whose recurrence changed (their presentation
    should be refreshed; ``view`` is ``None`` when a cue no longer recurs).
    """
    from lss.assets.naming import _locked
    from lss.bgm import identity

    loader = loader or (lambda path: identity.decode_mono(Path(path)))
    lock = root(state_root) / "memory.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with _locked(lock):
        data = read(state_root)
        before = _views(data)
        for sha, rows in _current_best_sources(state_root, source_sha256):
            _sync_source(data, sha, _accepted_items(data, sha, rows))
        _sync_source(data, source_sha256, items)
        members = data["members"]
        present = {key: value for key, value in members.items() if _still_present(value)}
        for asset_id in sorted(present, key=lambda key: members[key]["order"]):
            query = members[asset_id]
            others = [value for key, value in present.items()
                      if key != asset_id and key not in query["compared_with"]
                      and asset_id not in value["compared_with"]]
            if not others:
                continue
            samples = loader(query["path"])
            _fingerprint(state_root, query["pcm_sha256"], samples, Path(query["path"]))
            data["links"].extend(compare(query, samples, others, state_root))
            query["compared_with"] = sorted({*query["compared_with"], *[value["asset_id"] for value in others]})
        _assign_labels(data, groups(data))
        _write(state_root, data)
        after = _views(data)
    changed = []
    for key in sorted(set(before) | set(after)):
        member = data["members"].get(key)
        if member is None or member["source_sha256"] == source_sha256 or before.get(key) == after.get(key):
            continue
        changed.append({"path": member["path"], "pcm_sha256": member["pcm_sha256"], "view": after.get(key),
                        "source_sha256": member["source_sha256"], "asset_id": key})
    return {"recipe_id": data["recipe_id"], "state": "checked",
            "by_path": {view["path"]: view for view in after.values() if view["source_sha256"] == source_sha256},
            "changed_elsewhere": changed}


def _views(data: dict) -> dict[str, dict]:
    """The recurrence each member's presentation shows (only for members of a recurring group)."""
    members = data["members"]
    by_member = {}
    for number, entry in data["labels"].items():
        if entry.get("merged_into"):
            continue
        for key in entry["members"]:
            by_member[key] = number
    views = {}
    for anchor, keys in groups(data).items():
        if len(keys) < 2:
            continue
        number = next((by_member[key] for key in keys if key in by_member), None)
        if number is None:
            continue
        for key in keys:
            member = members[key]
            others = []
            for other in keys:
                if other == key:
                    continue
                value = members[other]
                link = next((link for link in data["links"] if {link["a"], link["b"]} == {key, other}), None)
                shared = tempo = None
                if link is not None:
                    shared = link["a_share"] if link["a"] == key else link["b_share"]
                    if link.get("rate") is not None:
                        # b advances ``rate`` seconds per second of a: b plays at 1/rate of a's tempo.
                        tempo = round(1.0 / link["rate"] if link["a"] == key else link["rate"], 4)
                others.append({"asset_id": other, "source_sha256": value["source_sha256"],
                               "source_name": value["source_name"], "start_seconds": value["start_seconds"],
                               "duration_seconds": round(value["duration_seconds"], 3), "path": value["path"],
                               "direct": link is not None, "share_of_this_cue": shared, "tempo_ratio": tempo})
            longest = max(keys, key=lambda item: (members[item]["duration_seconds"], -members[item]["order"]))
            views[key] = {"asset_id": key, "path": member["path"], "source_sha256": member["source_sha256"],
                          "label": f"{LABEL_PREFIX}{number}", "label_number": int(number), "anchor": anchor,
                          "group_size": len(keys), "longest": longest == key,
                          "others": sorted(others, key=lambda item: members[item["asset_id"]]["order"])}
    return views


def view_for(state_root: Path, path: Path) -> dict | None:
    """The current recurrence of one public file, read-only (for refreshing another source's presentation)."""
    data = read(state_root)
    return next((view for view in _views(data).values() if Path(view["path"]) == Path(path)), None)


def naming_evidence(view: dict, own_asset_id: str, store) -> dict | None:
    """Presentation evidence shared by a recurring group: one verified related title, else the anchor's description.

    Only titles the naming store already holds for group members are used; recurrence adds no new title source.
    Conflicting related titles inside a group are not propagated.
    """
    titles = set()
    for other in view["others"]:
        state = store.read(other["asset_id"])
        if state is None:
            continue
        current = state["revisions"][-1]
        if current["kind"] in ("official", "provisional_related"):
            related = next((item for item in current["evidence"] if item.get("signal") in
                            ("verified_related_cue", "verified_official_exact_version")), {})
            titles.add(related.get("title") or current["title"])
    if len(titles) == 1:
        title = titles.pop()
        return {"related": {"asset_id": own_asset_id, "title": title, "verified": True,
                            "relation": "same_cue_different_version", "basis": "same_recording_published_cue",
                            "label": view["label"],
                            "evidence_ref": f"cue_memory:{view['label']}:{view['anchor']}"}}
    if titles or view["anchor"] == own_asset_id:
        return None
    anchor = store.read(view["anchor"])
    if anchor is None:
        return None
    current = anchor["revisions"][-1]
    if current["kind"] != "provisional_descriptive":
        return None
    return {"context": {"asset_id": own_asset_id, "scope": "cue", "description": current["title"],
                        "basis": "same_recording_published_cue", "label": view["label"],
                        "evidence_ref": f"cue_memory:{view['label']}:{view['anchor']}"}}


def tag_value(view: dict) -> str:
    """Compact provenance written into the file (``LSS_BGM_RECURRENCE``)."""
    return json.dumps({"method": METHOD, "label": view["label"], "anchor": view["anchor"],
                       "group_size": view["group_size"], "longest": view["longest"],
                       "others": [{key: other[key] for key in ("source_sha256", "source_name", "start_seconds",
                                                                "duration_seconds", "direct", "share_of_this_cue",
                                                                "tempo_ratio")}
                                  for other in view["others"]]},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
