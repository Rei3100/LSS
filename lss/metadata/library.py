"""The series piece ledger: which published cue files are which piece of the series, and where each played.

One ledger per series (``<state>/metadata-library/<series key>/pieces.json``). A member is one published cue
file (asset ``bgm-cue:<key>``) with its occurrence (episode, time span) and, when its recorded reference relation
identifies an official track, that track. Members of one recording (Series Memory group) are one piece. Observing
a source replaces that source's members, so a re-run never duplicates occurrences. The ledger is internal state;
`lss.metadata.project` turns it into tags for every file of a piece.
"""
from __future__ import annotations

import json
from pathlib import Path

from lss.metadata import project

LEDGER = "pieces.json"
ADOPTED = "pieces.adopted.json"
_RELATED = tuple(project.RELATION_TAGS)
MIN_PRESENT_SHARE = 0.5


def _path(state_root: Path, key: str) -> Path:
    from lss.metadata.series_album import DIRECTORY

    return Path(state_root) / DIRECTORY / key / LEDGER


def read(state_root: Path, key: str) -> dict:
    try:
        data = json.loads(_path(state_root, key).read_text(encoding="utf-8"))
        if isinstance(data.get("members"), dict):
            return data
    except (OSError, ValueError, AttributeError):
        pass
    return {"members": {}}


def _write(state_root: Path, key: str, data: dict) -> None:
    path = _path(state_root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _reference(relation: dict, provenance: dict, state_root: Path) -> dict | None:
    """What LSS recorded about the released reference this cue is a version of (album/track/title/length)."""
    if (relation or {}).get("state") not in _RELATED:
        return None
    reference = relation.get("reference") or {}
    sha = reference.get("reference_sha256")
    if not sha:
        return None
    try:
        start = provenance["start_frame"] / provenance["sample_rate"]
        end = provenance["end_frame"] / provenance["sample_rate"]
    except (KeyError, TypeError, ZeroDivisionError):
        return None
    spans = sorted((max(start, r["start_seconds"]), min(end, r["end_seconds"]))
                   for r in relation.get("regions") or []
                   if r.get("reference_sha256") == sha
                   and r.get("kind") in ("supported", "missing_component", "additional_material"))
    # Witnesses can overlap or repeat; only distinct time within this cue supports its identity.
    present, covered_end = 0.0, start
    for first, last in spans:
        present += max(0.0, last - max(first, covered_end))
        covered_end = max(covered_end, last)
    if end <= start or present / (end - start) < MIN_PRESENT_SHARE:
        return None   # a short quote of a release does not make the cue a version of it
    return _reference_info(sha, state_root, reference.get("display_title"))


def _reference_info(sha: str, state_root: Path, display_title: str | None = None) -> dict:
    from lss.bgm import reference_store

    record = reference_store.load(sha, state_root) or {}
    meta = record.get("metadata") or {}
    tags = meta.get("tags") or {}
    return {"sha256": sha, "album": meta.get("album") or tags.get("album"),
            "tracknumber": meta.get("track_number") or tags.get("tracknumber"),
            "discnumber": tags.get("discnumber"), "title": meta.get("display_title") or display_title,
            "length_seconds": (record.get("audio") or {}).get("duration_seconds")}


def _contained(relation: dict, provenance: dict, summary: dict, state_root: Path, version_of: str | None) -> list[dict]:
    """Released recordings verified inside this cue that it is not a version of (episode time, title).

    A cue cut longer than one usage can hold part of a release next to other music; the cue stays its own piece
    and this records, as fact, which release plays where inside it.
    """
    try:
        cue_start = provenance["start_frame"] / provenance["sample_rate"]
        cue_end = provenance["end_frame"] / provenance["sample_rate"]
        offset = float(provenance["start_seconds"]) - cue_start
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return []
    spans: dict[str, list] = {}
    for region in relation.get("regions") or []:
        sha = region.get("reference_sha256")
        if region.get("kind") == "supported" and sha and sha != version_of:
            spans.setdefault(sha, []).append((region["start_seconds"], region["end_seconds"]))
    titles = {c.get("reference_sha256"): c.get("display_title") for c in relation.get("candidates") or []}
    titles.update({(relation.get("reference") or {}).get("reference_sha256"): (relation.get("reference") or {}).get("display_title")})
    out = []
    for sha, ranges in spans.items():
        info = _reference_info(sha, state_root, titles.get(sha))
        official = project.find_official(summary["album"], info)
        title = str(official.title.value) if official is not None else info.get("title")
        if not title:
            continue
        merged = []
        for first, last in sorted((max(cue_start, a), min(cue_end, b)) for a, b in ranges):
            if first >= last:
                continue
            if merged and first <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], last)
            else:
                merged.append([first, last])
        out.extend({"title": title, "start_seconds": first + offset, "end_seconds": last + offset}
                   for first, last in merged)
    return sorted(out, key=lambda c: c["start_seconds"])


def member(path: Path, provenance: dict, asset_id: str, summary: dict, recurrence: dict | None,
           state_root: Path) -> dict:
    """One published cue file as a ledger member (pure apart from reading the reference record)."""
    relation = provenance.get("reference_relation") or {}
    reference = _reference(relation, provenance, state_root)
    official = project.find_official(summary["album"], reference) if reference else None
    episode = summary.get("episode") or {}
    return {"asset_id": asset_id, "path": str(path), "source_sha256": (provenance.get("source") or {}).get("sha256"),
            "piece_id": f"recording:{recurrence['label']}" if recurrence else asset_id,
            "episode_label": episode.get("label"), "episode_order": episode.get("order"),
            "episode_date": episode.get("date"),
            "start_seconds": float(provenance.get("start_seconds") or 0.0),
            "end_seconds": float(provenance.get("end_seconds") or 0.0),
            "observed_scene": None,
            "relation_state": relation.get("state") if official is not None else None,
            "reference_sha256": (reference or {}).get("sha256"),
            "official": None if official is None else {"disc": official.disc, "number": official.number},
            "contains_released": _contained(relation, provenance, summary, state_root,
                                            (reference or {}).get("sha256") if official is not None else None)}


def regrouped(data: dict, labels: dict[str, str]) -> dict:
    """``data`` with every member of a current Series Memory group (``labels``: asset id -> ``劇伴#N``) on that
    group's piece. A member Series Memory no longer groups keeps the piece it had (its own asset when it never had
    one): grouping only ever joins pieces that are the same recording."""
    members = {}
    for asset, m in data["members"].items():
        label = labels.get(asset)
        members[asset] = {**m, "piece_id": f"recording:{label}"} if label else m
    return {**data, "members": members}


def observe(state_root: Path, key: str, source_sha256: str, members: list[dict], *, write: bool = True,
            labels: dict[str, str] | None = None, adopt: str | None = None) -> dict:
    """Replace ``source_sha256``'s members with ``members``; returns the updated ledger (written unless told not).

    ``adopt`` names a provisional ledger (the series key used while the series could not be identified); its
    members join this ledger once, so episodes processed offline are numbered with the rest of their series.
    """
    data = read(state_root, key)
    kept = {}
    if adopt and adopt != key:
        kept.update({asset: m for asset, m in read(state_root, adopt)["members"].items()
                     if m.get("source_sha256") != source_sha256})
    kept.update({asset: m for asset, m in data["members"].items() if m.get("source_sha256") != source_sha256})
    kept.update({m["asset_id"]: m for m in members})
    data = regrouped({**data, "members": kept}, labels or {})
    if write:
        _write(state_root, key, data)
        if adopt and adopt != key and _path(state_root, adopt).is_file():
            # Kept, not deleted: it is the record of what those files were numbered while the series was unknown.
            _path(state_root, adopt).replace(_path(state_root, adopt).with_name(ADOPTED))
    return data


def record_numbering(state_root: Path, key: str, source_sha256: str, was: dict, now: dict, data: dict, *,
                     adopted=()) -> list:
    """Record, as a numbering revision, every file whose disc/track changes (``was``/``now``: asset -> tags).

    Unreleased pieces are numbered in broadcast order, so processing an earlier episode later moves later
    pieces down (accepted by the contract); the move is kept as history, never done silently. Returns the
    changes (empty on an identical re-run, which then writes nothing).
    """
    changes = []
    for asset in sorted(set(was) & set(now)):
        before = (was[asset].get("DISCNUMBER"), was[asset].get("TRACKNUMBER"))
        after = (now[asset].get("DISCNUMBER"), now[asset].get("TRACKNUMBER"))
        if before != after:
            changes.append({"asset_id": asset, "from": {"disc": before[0], "track": before[1]},
                            "to": {"disc": after[0], "track": after[1]}})
    # Files numbered while the series was unknown (provisional ledger) that this ledger has now adopted.
    for asset in sorted(set(adopted) & set(now)):
        changes.append({"asset_id": asset, "from": "provisional", "to": {"disc": now[asset].get("DISCNUMBER"),
                                                                         "track": now[asset].get("TRACKNUMBER")}})
    if changes:
        import time

        revisions = list(data.get("numbering_revisions") or [])
        revisions.append({"revision": len(revisions) + 1, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                          "cause_source_sha256": source_sha256, "changes": changes})
        data["numbering_revisions"] = revisions
        _write(state_root, key, data)
    return changes


def pieces(data: dict) -> dict:
    """Ledger -> ``{piece_id: {"official", "relation_state", "occurrences", "members"}}``.

    Cues identified as (versions of) the same official track are one piece, whatever their Series Memory
    grouping: they take that track's one number and list all their occurrences together.
    """
    official_of = {}
    for asset, m in sorted(data["members"].items()):
        if m.get("official") and m["piece_id"] not in official_of:
            official_of[m["piece_id"]] = m["official"]
    out: dict = {}
    for asset, m in sorted(data["members"].items()):
        official = m.get("official") or official_of.get(m["piece_id"])
        piece_id = f"official:{official['disc']}-{official['number']}" if official else m["piece_id"]
        piece = out.setdefault(piece_id, {"official": None, "relation_state": None, "occurrences": [], "members": []})
        if official and piece["official"] is None:
            piece["official"] = official
        if m.get("official") and piece["relation_state"] is None:
            piece["relation_state"] = m.get("relation_state")
        piece["occurrences"].append({k: m.get(k) for k in ("episode_label", "episode_order", "episode_date",
                                                            "start_seconds", "end_seconds", "observed_scene",
                                                            "contains_released")})
        piece["members"].append(asset)
    return out
