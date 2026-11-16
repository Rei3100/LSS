"""Per-cue music-library tags from the resolved series album and the series piece ledger (pure).

Contract: docs/engine/LISTENING_AND_PRESENTATION.md "Music-library metadata". Inputs are already-decided facts;
nothing here fetches or infers. A cue of an officially released recording takes that recording's disc/track and
credits; an unreleased piece goes on the unreleased disc in broadcast order. Occurrences and relations of the
piece are tags too. TITLE stays with the naming system (verified official title or fixed provisional title).
"""
from __future__ import annotations

from lss.metadata import resolve

# Relation states recorded by `lss.bgm.reference_relations` -> one tag per kind of relation (value: related title).
RELATION_TAGS = {
    "released_recording_edit": "EDIT_OF",
    "released_recording_edited_by_program": "PROGRAM_EDIT_OF",
    "released_recording_component_variant": "COMPONENT_VARIANT_OF",
    "released_recording_with_additional_material": "ADDITIONAL_MATERIAL_ON",
}
FIXED_KEYS = ("ALBUM", "DISCNUMBER", "TRACKNUMBER", "DATE", "M_NUMBER", "SERIES_MUSIC", "FIRST_USE_EPISODE",
              "FIRST_USE_DATE", "USE_EPISODE", "USE_START", "USE_END", "USE_SCENE", "CONTAINS_RELEASED",
              "CONTAINS_RELEASED_EPISODE", "CONTAINS_RELEASED_START", "CONTAINS_RELEASED_END", *RELATION_TAGS.values())


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def find_official(album: dict, reference: dict | None):
    """The official track a released reference recording is, from the reference file's own tags.

    ``reference`` carries what LSS recorded about the reference file (``album``, ``tracknumber``, ``discnumber``,
    ``title``, ``length_seconds``). Reference files usually come from stores, whose numbering may run through a
    multi-disc set as one disc, so a number is read both as (disc, number) and as the running position in the
    volume; the reference's own length must agree (within 2 s) whenever both lengths are known, and a number
    that fits two different tracks decides nothing. Otherwise a unique title match (store suffix removed),
    again length-checked, decides. None when nothing identifies exactly one track.
    """
    if not reference or not album.get("tracks"):
        return None
    length = reference.get("length_seconds")

    def fits(track) -> bool:
        return length is None or track.length_seconds is None or abs(track.length_seconds - length) <= 2.0
    ref_album = resolve.compare_key(reference.get("album") or "")
    try:
        number = int(str(reference.get("tracknumber") or "").split("/")[0])
    except ValueError:
        number = None
    try:
        disc = int(str(reference.get("discnumber") or "1").split("/")[0])
    except ValueError:
        disc = 1
    if ref_album and number is not None:
        volumes = {}
        for track in album["tracks"]:
            volumes.setdefault(track.release_title, []).append(track)
        for title, tracks in volumes.items():
            names = {resolve.compare_key(title)} | {resolve.compare_key(t) for t in (album.get("volume_titles") or {}).get(title, ())}
            if ref_album not in names:
                continue
            ordered = sorted(tracks, key=lambda t: (t.disc, t.number))
            first_disc = ordered[0].disc
            found = {(t.disc, t.number): t for t in ordered if t.disc - first_disc + 1 == disc and t.number == number}
            if disc == 1 and 1 <= number <= len(ordered):
                running = ordered[number - 1]
                found[(running.disc, running.number)] = running
            found = [t for t in found.values() if fits(t)]
            if len(found) == 1:
                return found[0]
    title = resolve.compare_key(resolve.strip_store_suffix(reference.get("title") or ""))
    if title:
        found = [t for t in album["tracks"] if fits(t) and (resolve.compare_key(str(t.title.value)) == title
                 or any(resolve.compare_key(str(v)) == title for v in t.title.raw.values()))]
        if len(found) == 1:
            return found[0]
    return None


def unreleased_order(pieces: dict) -> dict[str, int]:
    """Track numbers of unreleased pieces in broadcast order: first-use episode, then time; ties by piece id."""
    def first_use(item):
        piece_id, piece = item
        keys = [(o.get("episode_order") if o.get("episode_order") is not None else 10 ** 9, o["start_seconds"])
                for o in piece["occurrences"]]
        return (min(keys) if keys else (10 ** 9, 0.0), piece_id)
    unreleased = [(pid, piece) for pid, piece in pieces.items() if not piece.get("official")]
    return {pid: n for n, (pid, _) in enumerate(sorted(unreleased, key=first_use), 1)}


def tags_for(piece_id: str, album: dict, pieces: dict, series: dict | None) -> dict:
    """Tags (name -> str | list[str]) for every file of ``piece_id``; ``pieces`` is the series ledger.

    One tag, one meaning: occurrences are parallel multi-values (``USE_EPISODE``/``USE_START``/``USE_END``/
    ``USE_SCENE``, the scene empty when none was observed); each relation kind has its own tag.
    """
    piece = pieces[piece_id]
    official = piece.get("official")
    tags: dict = {"ALBUM": album["title"]}
    if series and series.get("first_date"):
        tags["DATE"] = series["first_date"]
    if official is not None:
        track = next((t for t in album["tracks"] if (t.disc, t.number) == (official["disc"], official["number"])), None)
        tags["DISCNUMBER"], tags["TRACKNUMBER"] = str(official["disc"]), str(official["number"])
        if track is not None:
            if track.m_number.value:
                tags["M_NUMBER"] = str(track.m_number.value)
            credit, _ = resolve.credit_tags(track.credits)
            tags.update(credit)
            relation = RELATION_TAGS.get(piece.get("relation_state"))
            if relation:
                tags[relation] = str(track.title.value)
    else:
        tags["DISCNUMBER"] = str(album["unreleased_disc"])
        tags["TRACKNUMBER"] = str(unreleased_order(pieces)[piece_id])
        if series and series.get("music"):
            tags["SERIES_MUSIC"] = list(series["music"])
    occurrences = sorted(piece["occurrences"], key=lambda o: (
        o.get("episode_order") if o.get("episode_order") is not None else 10 ** 9, o["start_seconds"]))
    if occurrences:
        tags["USE_EPISODE"] = [o.get("episode_label") or "" for o in occurrences]
        tags["USE_START"] = [clock(o["start_seconds"]) for o in occurrences]
        tags["USE_END"] = [clock(o["end_seconds"]) for o in occurrences]
        if any(o.get("observed_scene") for o in occurrences):
            tags["USE_SCENE"] = [o.get("observed_scene") or "" for o in occurrences]
        contained = [(o, c) for o in occurrences for c in o.get("contains_released") or ()]
        if contained:
            tags["CONTAINS_RELEASED"] = [c["title"] for _, c in contained]
            tags["CONTAINS_RELEASED_EPISODE"] = [o.get("episode_label") or "" for o, _ in contained]
            tags["CONTAINS_RELEASED_START"] = [clock(c["start_seconds"]) for _, c in contained]
            tags["CONTAINS_RELEASED_END"] = [clock(c["end_seconds"]) for _, c in contained]
        first = occurrences[0]
        if first.get("episode_label"):
            tags["FIRST_USE_EPISODE"] = first["episode_label"]
        if first.get("episode_date"):
            tags["FIRST_USE_DATE"] = first["episode_date"]
    return tags
