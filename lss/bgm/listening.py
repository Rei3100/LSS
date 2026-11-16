"""Listening presentation of accepted CurrentBest cues; no reference/semantic decisions live here."""
from __future__ import annotations

import json
import math
from pathlib import Path

from mutagen.flac import FLAC

from lss.assets.naming import NamingStore
from lss.bgm import provisional_naming, scene_names
from lss.core import audio_identity
from lss.output import finalizer

DIRECTORY = "bgm-presentation-names"

# Relations whose waveform evidence says the cue *is* a version of one released recording (not merely a
# harmonic resemblance, and not the released recording itself, which is never published).
_RELATED_NAMING_STATES = ("released_recording_edit", "released_recording_component_variant",
                          "released_recording_with_additional_material")
_RELATED_PRESENT_KINDS = ("supported", "missing_component", "additional_material")
RELATED_MIN_PRESENT_SHARE = 0.5


def related_naming_evidence(asset_id: str, provenance: dict) -> dict | None:
    """Naming evidence from the cue's own recorded reference relation (`reference_relations.compact`).

    Only a waveform-verified relation to one released recording whose title came from that recording's own
    tags, and only when that recording is present in at least half of the cue: a cue that merely contains a
    short quote of a release is not named after it. File names, search results and RELATED (harmony-only)
    candidates never name a cue.
    """
    relation = (provenance or {}).get("reference_relation") or {}
    reference = relation.get("reference") or {}
    if (relation.get("state") not in _RELATED_NAMING_STATES or reference.get("title_source") != "reference_file_tags"
            or not isinstance(reference.get("display_title"), str) or not reference["display_title"].strip()
            or not reference.get("reference_sha256")):
        return None
    # Relation regions use the analysed audio's own clock (frames), not source media time (which adds an offset).
    try:
        start = provenance["start_frame"] / provenance["sample_rate"]
        end = provenance["end_frame"] / provenance["sample_rate"]
    except (KeyError, TypeError, ZeroDivisionError):
        return None
    if end <= start:
        return None
    present = 0.0
    for region in relation.get("regions") or []:
        if region.get("kind") in _RELATED_PRESENT_KINDS and region.get("reference_sha256") == reference["reference_sha256"]:
            present += max(0.0, min(end, region["end_seconds"]) - max(start, region["start_seconds"]))
    share = present / (end - start)
    if share < RELATED_MIN_PRESENT_SHARE:
        return None
    return {"related": {"asset_id": asset_id, "title": reference["display_title"].strip(), "verified": True,
                        "relation": "same_cue_different_version", "reference_relation": relation["state"],
                        "reference_sha256": reference["reference_sha256"], "present_share": round(share, 3),
                        "evidence_ref": f"reference_relation:{relation['state']}:{reference['reference_sha256']}"}}


RECURRENCE_TAG = "LSS_BGM_RECURRENCE"
# The file's own episode in its series album (`library_tags`); provisional time titles carry it.
EPISODE_TAG = "LSS_LIBRARY_EPISODE"


def finish(path: Path, state_root: Path, *, expected_pcm: str | None = None,
           extra_evidence: dict | None = None, validate_before_publish=None, accepted: dict | None = None,
           recurrence: dict | None = None, collection_tags: dict | None = None,
           library_tags: dict | None = None) -> dict:
    """Project a supported published cue into the same physical listening asset, atomically.

    Names stay out of cue identity and public path. Official title upgrades cannot produce
    duplicate assets or change CurrentBest/history. The naming store is LSS's presentation record; LSS writes
    the file's complete tag set (title, presentation, ReplayGain, picture) from it.
    A program's embedded cover does not establish cue/release artwork; LSS asserts no artwork here.
    ``recurrence`` is this cue's view in the published-cue memory (`lss.bgm.cue_memory`): the cue is the same
    recording as other published cues. Its number goes into the provisional title, the group's verified title or
    description becomes naming evidence, and the relation is recorded in ``LSS_BGM_RECURRENCE``.
    """
    path = Path(path)
    audio = audio_identity.audio_identity(path)
    if expected_pcm is not None and audio["pcm_sha256"] != expected_pcm:
        raise finalizer.FinalizationError("CurrentBestと聴取用音声が一致しません")
    store = NamingStore(Path(state_root) / DIRECTORY)
    source = FLAC(path)
    scene_key = ((accepted or {}).get("key") or (source.get("LSS_BGM_CUE_KEY") or [None])[0])
    # Explicit official evidence changes naming precedence, not the independent score/music tags.
    scene = (scene_names.presentation(state_root, f"bgm-cue:{scene_key}", audio["pcm_sha256"])
             if scene_key else None)
    if extra_evidence is None:
        try:
            recorded = (accepted["provenance"] if accepted is not None
                        else json.loads(source["LSS_BGM_PROVENANCE"][0]))
            key = accepted["key"] if accepted is not None else source["LSS_BGM_CUE_KEY"][0]
        except (KeyError, IndexError, TypeError, ValueError):
            recorded, key = None, None
        if recorded is not None and key:
            extra_evidence = related_naming_evidence(f"bgm-cue:{key}", recorded)
            shared = {}
            if recurrence is not None:
                from lss.bgm import cue_memory
                shared = cue_memory.naming_evidence(recurrence, f"bgm-cue:{key}", store) or {}
            # The recording's fixed scene name replaces a group member's sound description; a verified title
            # (the group's or the cue's own relation) still outranks it.
            if scene is not None and scene["evidence"] is not None:
                shared = {**{k: v for k, v in shared.items() if k != "context"}, **scene["evidence"]}
            # The cue's own verified relation outranks what its group says.
            extra_evidence = {**shared, **(extra_evidence or {})} or None
    named = provisional_naming.name_published(path, store,
                                             extra_evidence=extra_evidence, accepted=accepted)
    previous = finalizer.previous_record(path)
    # LSS-owned values: what LSS projected last time, or (before receipts) what the file carries.
    owned = previous["projected_tags"] if previous is not None else {key.upper(): list(value) for key, value in source.items()}
    excluded = {finalizer.RECEIPT_TAG, "LSS_INTEGRATED_LOUDNESS_LUFS", "LSS_TRUE_PEAK_DBTP", "LSS_COLLECTION_METADATA"}
    tags = {key: value for key, value in owned.items()
            if key.startswith("LSS_") and key not in excluded and key != RECURRENCE_TAG
            and not key.startswith(("LSS_LISTENING_", "LSS_NAMING_"))}
    # Library metadata LSS projected before (album, numbers, date, credits, occurrences: `lss.metadata`) stays
    # as it was on a single-file refresh; the collection pass reconciles it. The collection label is contextual
    # source information, never an asserted soundtrack release. Only a receipt proves LSS wrote DATE and credits.
    for key in ("ALBUM", "TRACKNUMBER", "DISCNUMBER"):
        if owned.get(key) or source.get(key):
            tags[key] = owned.get(key) or [str(value) for value in source[key]]
    if previous is not None:
        for key in ("DATE", *sorted(finalizer.metadata_keys(owned))):
            if owned.get(key):
                tags[key] = owned[key]
    if accepted is not None:
        from lss.output.writer import headroom_scale
        shift = accepted["headroom_shift"]
        provenance = {**accepted["provenance"], "serialization": {
            "headroom_shift": shift, "gain": headroom_scale(shift), "subtype": "PCM_24",
            "clipping": "none", "normalization": "none"}}
        tags.update(LSS_SOURCE_TYPE="bgm_cue", LSS_STEM="bgm", LSS_BGM_CUE_KEY=accepted["key"],
                    LSS_INPUT_SHA256=accepted["input_sha256"], LSS_RECIPE_ID=accepted["recipe_id"],
                    LSS_OUTPUT_HEADROOM_SHIFT=str(shift), LSS_VERSION=accepted["version"],
                    LSS_BGM_PROVENANCE=json.dumps(provenance, ensure_ascii=False, sort_keys=True,
                                                  separators=(",", ":"), allow_nan=False), ALBUM=accepted["album"])
    else:
        provenance = json.loads(tags["LSS_BGM_PROVENANCE"][0])
    extent = (provenance.get("cue_extent") or {}).get("state")
    title = named["title"]
    # In a series album (`lss.metadata`) the album is the series, not the episode: a time in the title is only
    # unique with its episode, so the episode goes in front of it.
    episode = (library_tags or {}).get(EPISODE_TAG) or tags.get(EPISODE_TAG)
    episode = (episode[0] if isinstance(episode, list) else episode) or None
    at = f"{episode} " if episode else ""
    basis = (named.get("evidence") or [{}])[0]
    scene_named = basis.get("signal") == "local_cue_context" and basis.get("basis") in ("scene", "music", "context_music")
    if scene_named:
        # A fixed song-like provisional name keeps source time, numbers and labels in separate tags.
        title = f"{named['base_title']}（仮題）"
    elif (scene is not None and not scene.get("name") and named["kind"] == "provisional_descriptive"
          and isinstance(provenance.get("start_seconds"), (int, float))
          and basis.get("signal") in ("local_acoustic_measurement", "insufficient_descriptive_evidence")):
        # The scene was looked for and is not known: the title keeps the time (work and episode are the album)
        # and invents nothing; a sound description is not a scene.
        start = max(0, int(provenance["start_seconds"]))
        title = f"{at}{start // 60:02d}:{start % 60:02d}（仮題）"
    elif not named["is_official"] and isinstance(provenance.get("start_seconds"), (int, float)):
        # A provisional title says where the cue is in the episode (unique within its album) and what it
        # sounds like. The identity token stays in LSS_NAMING_ASSET_ID, not in the listener's title.
        start = max(0, int(provenance["start_seconds"]))
        label = f"・{recurrence['label']}" if recurrence is not None else ""
        title = f"{at}{start // 60:02d}:{start % 60:02d} {named['base_title']}（仮題{label}）"
    elif recurrence is not None:
        title = f"{title}（{recurrence['label']}）"
    if extent == "unresolved":
        from lss.bgm.cue_publication import UNRESOLVED_EXTENT_MARK
        title += f" ({UNRESOLVED_EXTENT_MARK})"
    if recurrence is not None:
        from lss.bgm import cue_memory
        tags[RECURRENCE_TAG] = cue_memory.tag_value(recurrence)
    if scene is not None:
        # M number, order-sheet line and version are their own tags, never mixed into TITLE/TRACKNUMBER.
        tags.update({name: [value] for name, value in scene["tags"].items()})
    tags.update(TITLE=title, LSS_NAMING_KIND=named["kind"], LSS_NAMING_REVISION=named["revision_id"],
                LSS_NAMING_ASSET_ID=named["asset_id"], LSS_NAMING_OFFICIAL="1" if named["is_official"] else "0")
    if collection_tags is not None:
        tags.update(collection_tags)
    if library_tags is not None:
        # The series album projection (`lss.metadata`); an official M number replaces LSS's own in place.
        tags.update(library_tags)
    def still_current():
        if store.read(named["asset_id"])["current_revision_id"] != named["revision_id"]:
            raise finalizer.FinalizationError("仕上げの間に名称の根拠が更新されました")
        if validate_before_publish is not None:
            validate_before_publish()
    result = finalizer.ensure(path, tags=tags, picture=None, validate_before_publish=still_current)
    return {**result, "naming": named, "recurrence": recurrence}


def _collection_numbers(paths, accepted_by_path: dict | None = None) -> dict:
    """Episode playback order from accepted source/time provenance only; no soundtrack/M-number claim."""
    groups = {}
    for path in paths:
        accepted = (accepted_by_path or {}).get(str(path))
        if accepted is not None:
            provenance, source_sha = accepted["provenance"], accepted["input_sha256"]
        else:
            previous = finalizer.previous_record(Path(path))
            tags = previous["projected_tags"] if previous is not None else FLAC(path)
            provenance = json.loads(tags["LSS_BGM_PROVENANCE"][0])
            source_sha = tags["LSS_INPUT_SHA256"][0]
        if not source_sha or provenance["source"]["sha256"] != source_sha:
            raise finalizer.FinalizationError("曲順の原本と来歴が一致しません")
        start = provenance.get("start_seconds")
        known = type(start) in (int, float) and math.isfinite(start)
        groups.setdefault(source_sha, []).append((float(start) if known else None, str(path)))
    result = {}
    for members in groups.values():
        starts = [start for start, _ in members]
        # An unknown or tied position cannot establish a whole collection order. Keep metadata unknown.
        if None in starts or len(set(starts)) != len(starts):
            continue
        for track, (_, path) in enumerate(sorted(members), 1):
            result[path] = {"TRACKNUMBER": str(track), "DISCNUMBER": "1"}
    return result


def _provenance(path: Path, accepted: dict | None) -> tuple[dict, str]:
    if accepted is not None:
        return accepted["provenance"], f"bgm-cue:{accepted['key']}"
    previous = finalizer.previous_record(Path(path))
    tags = previous["projected_tags"] if previous is not None else FLAC(path)
    return json.loads(tags["LSS_BGM_PROVENANCE"][0]), f"bgm-cue:{tags['LSS_BGM_CUE_KEY'][0]}"


def _series_labels(state_root: Path) -> dict[str, str]:
    """asset id -> ``劇伴#N`` of every published cue Series Memory currently groups (`lss.bgm.cue_memory`)."""
    from lss.bgm import cue_memory

    try:
        views = cue_memory._views(cue_memory.read(state_root))
    except Exception:  # noqa: BLE001 - unreadable memory leaves the ledger's own grouping
        return {}
    labels = {}
    for key, view in views.items():
        cue = str(key).removeprefix("bgm-cue:")
        labels[f"bgm-cue:{cue}"] = view["label"]
    return labels


def library_tags(paths, state_root: Path, library: dict, *, accepted_by_path: dict | None = None,
                 recurrence_by_path: dict | None = None, write: bool = True) -> tuple[dict, list[dict]]:
    """(tags for ``paths``, files of other sources whose projection changed) from the series ledger.

    Every file of a piece carries the same occurrences and numbers, so adding an episode can change files of
    other episodes; those are returned as ``{"asset_id", "source_sha256", "tags"}`` for a tag-only refresh of
    that source's CurrentBest file. Members are grouped by the current Series Memory labels, so one recording
    heard in several episodes is one piece with one number. A number that changes on a file already in the
    ledger is recorded as a numbering revision. ``write=False`` computes without recording.
    """
    from lss.metadata import library as ledger, project

    before = ledger.read(state_root, library["key"])
    members, source, mine = [], None, {}
    for path in paths:
        provenance, asset = _provenance(Path(path), (accepted_by_path or {}).get(str(path)))
        source = source or (provenance.get("source") or {}).get("sha256")
        members.append(ledger.member(Path(path), provenance, asset, library,
                                     (recurrence_by_path or {}).get(str(path)), state_root))
        mine[asset] = str(path)
    labels = _series_labels(state_root)
    provisional = library.get("provisional_key")
    adopted = sorted(a for a, m in ledger.read(state_root, provisional)["members"].items()
                     if m.get("source_sha256") != source) if provisional and provisional != library["key"] else []
    data = ledger.observe(state_root, library["key"], source, members, write=write,
                          labels=labels, adopt=provisional) if source else before
    # ``before`` is what was projected last time (its stored grouping), so a regrouped file is refreshed too.
    old, new = ledger.pieces(before), ledger.pieces(data)

    def project_all(pieces, ledger_data):
        out = {}
        for piece_id, piece in pieces.items():
            tags = project.tags_for(piece_id, library["album"], pieces, library.get("series"))
            dynamic = sorted(k for k in tags if k not in finalizer.PRESENTATION_KEYS)
            tags[finalizer.METADATA_KEYS_TAG] = "\n".join(dynamic)
            for asset in piece["members"]:
                own = (ledger_data["members"].get(asset) or {}).get("episode_label")
                out[asset] = {**tags, EPISODE_TAG: own} if own else tags
        return out
    now, was = project_all(new, data), project_all(old, before)
    if write and source:
        ledger.record_numbering(state_root, library["key"], source, was, now, data, adopted=adopted)
    changed = [{"asset_id": asset, "source_sha256": data["members"][asset].get("source_sha256"), "tags": tags}
               for asset, tags in now.items() if asset not in mine and was.get(asset) != tags]
    return {mine[asset]: tags for asset, tags in now.items() if asset in mine}, changed


def _refresh_library_elsewhere(state_root: Path, changed: list[dict]) -> dict:
    """Tag-only refresh of other sources' files whose shared series projection changed.

    The only write target is that source's CurrentBest ``public_path`` (`lss.bgm.current_best.published_files`)
    holding the recorded PCM; the path the ledger remembers is an observation and is never written. A file that
    cannot be refreshed now keeps its tags; the next run of its own source (or of the series) applies them.
    """
    from lss.bgm import current_best, write_targets

    done, pending = 0, []
    for item in changed:
        try:
            cue = str(item["asset_id"]).removeprefix("bgm-cue:")
            row = (current_best.published_files(state_root, item["source_sha256"]).get(cue)
                   if item.get("source_sha256") else None)
            if row is None or not Path(row["public_path"]).is_file():
                pending.append({"asset_id": item["asset_id"], "error": "not_a_current_published_path"})
                continue
            try:  # containment (junction-aware): the same rule as every other presentation refresh
                target = write_targets.checked(state_root, row, cue)
            except write_targets.WriteTargetRefused as refused:
                pending.append({"asset_id": item["asset_id"], "error": "not_a_current_published_path",
                                "reason": refused.reason})
                continue
            finish(target, state_root, expected_pcm=row.get("pcm_sha256"),
                   library_tags=item["tags"])
            done += 1
        except Exception as exc:  # noqa: BLE001 - another source's presentation is never this run's failure
            pending.append({"asset_id": item.get("asset_id"), "error": type(exc).__name__})
    return {"refreshed": done, "pending": pending}


def finish_outputs(paths: tuple[Path, ...], state_root: Path, *, validate_before_publish=None,
                   accepted_by_path: dict | None = None, recurrence_by_path: dict | None = None,
                   library: dict | None = None) -> list[dict]:
    """Finish a source's collection. With ``library`` (`lss.metadata.series_album`) the files join the series
    album; CurrentBest files of other episodes whose shared tags changed are refreshed tag-only."""
    paths = tuple(paths)
    if library is None:
        numbering, mine, others = _collection_numbers(paths, accepted_by_path), {}, []
    else:
        numbering = {}
        mine, others = library_tags(paths, state_root, library, accepted_by_path=accepted_by_path,
                                    recurrence_by_path=recurrence_by_path)
    rows = [finish(path, state_root, validate_before_publish=validate_before_publish,
                   accepted=(accepted_by_path or {}).get(str(path)),
                   recurrence=(recurrence_by_path or {}).get(str(path)),
                   collection_tags=numbering.get(str(path)), library_tags=mine.get(str(path))) for path in paths]
    if others:
        _refresh_library_elsewhere(state_root, others)
    return rows


def describe(paths) -> list[dict]:
    """Read-only projection for the existing-result/COMPLETE path; never rewrites or scans loudness."""
    rows = []
    for path in paths:
        record = finalizer.receipt(Path(path))
        if record is None:
            raise finalizer.FinalizationError("聴取用成果物が未完成です")
        tags = record["projected_tags"]
        rows.append({"path": str(path), "sha256": audio_identity.file_digest(Path(path)), "audio": record["audio"],
                     "finalization": record, "reused": True,
                     "title": (FLAC(Path(path)).get("TITLE") or [None])[0],
                     "naming": {"kind": (tags.get("LSS_NAMING_KIND") or [None])[0],
                                "asset_id": (tags.get("LSS_NAMING_ASSET_ID") or [None])[0]}})
    return rows


def current(paths, state_root: Path, *, collection: bool = False, library: dict | None = None) -> bool:
    """Read-only reuse gate; new naming evidence or a missing receipt requires presentation completion."""
    store = NamingStore(Path(state_root) / DIRECTORY)
    paths = tuple(paths)
    try:
        if library is not None:
            numbering = library_tags(paths, state_root, library, write=False)[0]
        else:
            numbering = _collection_numbers(paths) if collection else {}
    except (OSError, ValueError, KeyError, TypeError):
        return False
    for path in paths:
        try:
            record = finalizer.receipt(Path(path))
            if record is None:
                return False
            tags = record["projected_tags"]
            state = store.read(tags["LSS_NAMING_ASSET_ID"][0])
            if state is None or tags["LSS_NAMING_REVISION"] != [state["current_revision_id"]]:
                return False
            # A renumbered M number or a new order-sheet line also needs the presentation refreshed.
            scene = scene_names.presentation(state_root, tags["LSS_NAMING_ASSET_ID"][0], record["audio"].get("pcm_sha256"))
            # Scene records arrive after initial listening completion. Their name can change without
            # changing the M number or order line, so the naming revision alone is not a freshness gate.
            # Verified release evidence still outranks this local context, just as NamingStore does.
            context = ((scene or {}).get("evidence") or {}).get("context")
            revision = state["revisions"][-1]
            signal = revision["evidence"][0]["signal"]
            if (context is not None
                    and signal not in ("verified_official_exact_version", "verified_related_cue")
                    and (revision.get("input_evidence") or {}).get("context") != context):
                return False
            wanted = (scene or {}).get("tags") or {}
            expected = dict(numbering.get(str(path), {}))
            for name, value in expected.items():
                if tags.get(name) != (list(value) if isinstance(value, list) else [value]):
                    return False
            for name in ("M_NUMBER", "ORDER_MENU", "VERSION", "BPM", "INITIALKEY"):
                if name not in expected and tags.get(name) != ([wanted[name]] if name in wanted else None):
                    return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
    return True
