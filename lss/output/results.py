"""Read-only Product result projection. Run, CurrentBest and completion remain the authorities."""
from __future__ import annotations

def project(*, disposition: str, outputs=(), reused=False, held_count=0, zero_result=None,
            error_code=None, run: dict | None = None, current_best: dict | None = None,
            listening=(), semantic_record: dict | None = None, rebuilt=(), recurrence: dict | None = None) -> dict:
    """``rebuilt`` are published paths that were missing/changed and were regenerated from durable state.

    ``recurrence`` is the published-cue memory's view (`lss.bgm.cue_memory`): which of these cues are the same
    recording as other published cues. Projected as found; ``None`` count means the memory was not checked.
    """
    record = current_best or {}
    rebuilt = [str(path) for path in rebuilt or ()]
    scope_counts = ((semantic_record or {}).get("supported_scope") or {}).get("counts") or {}
    held_count = scope_counts.get("HELD", held_count)
    current = record.get("current") or {}
    decisions = current.get("decisions") or {}
    improvements = [row for row in decisions.get("old") or [] if row.get("decision") == "replace_with_better"]
    state = disposition
    if disposition == "PUBLISHED":
        state = ("REBUILT" if rebuilt else "REUSED") if reused else "IMPROVED_REPLACED" if improvements else "PUBLISHED"
    assets = []
    for item in listening:
        named = item.get("naming") or {}
        receipt = item.get("finalization") or {}
        tags = receipt.get("projected_tags") or {}
        assets.append({"path": item["path"], "sha256": item.get("sha256"),
                       "pcm_sha256": (item.get("audio") or {}).get("pcm_sha256"),
                       "title": item.get("title") or (tags.get("TITLE") or [None])[0],
                       "naming_kind": named.get("kind"),
                       "asset_id": named.get("asset_id"), "listening_policy": (receipt.get("policy") or {}).get("revision"),
                       "loudness_lufs": (receipt.get("measurement") or {}).get("loudness_lufs"),
                       "true_peak_dbtp": (receipt.get("measurement") or {}).get("true_peak_dbtp"),
                       "playback_gain_db": receipt.get("playback_gain_db"), "presentation_reused": item.get("reused")})
    pending = []
    if held_count:
        pending.append({"state": "POSSIBLE_FUTURE_IMPROVEMENT", "reason": "held_evidence_unresolved",
                        "count": held_count, "improvement_guaranteed": False})
    # Accepted cues whose file could not be rebuilt are UNKNOWN (not ABSENT, not invalid): projected, never dropped.
    unavailable = [{"state": "UNKNOWN", **{name: row.get(name) for name in
                                          ("public_path", "cue_key", "start_seconds", "end_seconds", "reason")}}
                   for row in current.get("unavailable") or []]
    if unavailable:
        pending.append({"state": "POSSIBLE_FUTURE_IMPROVEMENT", "reason": "accepted_cue_file_unavailable",
                        "count": len(unavailable), "improvement_guaranteed": False})
    unknown = sum(1 for row in decisions.get("old") or [] if row.get("validity") == "UNKNOWN")
    if unknown:
        pending.append({"state": "POSSIBLE_FUTURE_IMPROVEMENT", "reason": "current_cue_validity_unknown",
                        "count": unknown, "improvement_guaranteed": False})
    known, related = [], []
    # Existing decisions are projected verbatim. No new identity/availability/title inference.
    reference_check = None
    if semantic_record:
        refs = semantic_record.get("references") or {}
        relations = semantic_record.get("reference_relations")
        if isinstance(relations, dict):
            reference_check = relations.get("state")
        if reference_check == "checked":
            # The relation engine's decision is the authority; raw fingerprint hits (`known_spans`) are only the
            # candidates it examined, and a partial match must not be shown as an available recording.
            for span in relations.get("occurrences") or []:
                known.append({"state": "ALREADY_AVAILABLE", "basis": "reference_relation_decision",
                              **{name: span.get(name) for name in
                                 ("start_seconds", "end_seconds", "source_start_seconds", "source_end_seconds",
                                  "relation", "reference_sha256", "display_title", "reference_range_seconds")}})
            # Typed non-EXACT matches carry useful identity evidence without duplicate-release authority.
            # Keep their region maps and normal extraction decisions separate from already-available spans.
            related = [dict(span) for span in relations.get("related_occurrences") or []
                       if span.get("relation") in {"released_recording_edit",
                                                   "released_recording_with_additional_material",
                                                   "released_recording_component_variant"}]
        else:
            for span in refs.get("known_spans") or []:
                known.append({"state": "KNOWN", "basis": "existing_recording_decision", **span})
        duplicates = semantic_record.get("published_duplicates") or {}
        for span in duplicates.get("spans") or []:
            known.append({"state": "ALREADY_AVAILABLE", "basis": "existing_published_duplicate_decision", **span})
    return {"schema_version": 1, "state": state, "disposition": disposition, "reused": bool(reused),
            "source_sha256": ((run or {}).get("source") or {}).get("sha256"),
            "published_count": len(outputs), "paths": [str(path) for path in outputs], "assets": assets,
            "held_count": held_count, "zero_result": zero_result, "error_code": error_code,
            "unsupported_count": scope_counts.get("UNSUPPORTED"),
            "replacement_count": len(improvements) if not reused else 0,
            "replacement_history_count": len(record.get("history") or []),
            "rebuilt_count": len(rebuilt), "rebuilt_paths": rebuilt,
            "unavailable_count": len(unavailable), "unavailable": unavailable,
            "known": known, "known_evidence_available": reference_check == "checked" or bool(known),
            "reference_check": reference_check,
            "related_recordings": related,
            "related_recording_count": len(related) if reference_check == "checked" else None,
            "available_count": (sum(1 for item in known if item["basis"] == "reference_relation_decision")
                                if reference_check == "checked" else None),
            "related_named_count": sum(1 for item in assets if item["naming_kind"] == "provisional_related"),
            **_recurrence(recurrence, outputs),
            "future_improvement": pending}


def _recurrence(memory: dict | None, outputs) -> dict:
    if memory is None:
        return {}
    if memory.get("state") != "checked":
        return {"cue_memory_state": memory.get("state"), "recurring_count": None, "recurrence": []}
    views = memory.get("by_path") or {}
    rows = []
    for path in outputs:
        view = views.get(str(path))
        if view is None:
            continue
        rows.append({"path": str(path), "label": view["label"], "group_size": view["group_size"],
                     "longest": view["longest"],
                     "others": [{name: other.get(name) for name in
                                 ("source_name", "start_seconds", "duration_seconds", "path", "direct",
                                  "tempo_ratio")}
                                for other in view["others"]]})
    elsewhere = memory.get("refreshed_elsewhere") or {}
    return {"cue_memory_state": "checked", "cue_memory_recipe_id": memory.get("recipe_id"),
            "recurring_count": len(rows), "recurrence": rows,
            "recurrence_refreshed_elsewhere": elsewhere.get("refreshed", 0),
            "recurrence_pending_elsewhere": len(elsewhere.get("pending") or []),
            **_recurrence_assets(memory.get("assets"), rows)}


def _recurrence_assets(assets: dict | None, rows: list[dict]) -> dict:
    """Consensus assets (`lss.bgm.recurrence_asset`) of the groups this result's cues belong to."""
    if assets is None:
        return {}
    if assets.get("state") != "checked":
        return {"recurrence_asset_state": assets.get("state"), "recurrence_asset_count": None,
                "recurrence_assets": []}
    labels = {row["label"] for row in rows}
    mine = [{name: item.get(name) for name in ("label", "state", "path", "members", "evidence_seconds",
                                               "consensus_seconds", "replaced_previous")}
            for item in assets.get("assets") or [] if item.get("label") in labels]
    available = [item for item in mine if item["state"] in ("published", "reused", "kept_previous") and item["path"]]
    return {"recurrence_asset_state": "checked", "recurrence_asset_count": len(available),
            "recurrence_asset_updated_count": sum(1 for item in mine if item["state"] == "published"),
            "recurrence_assets": mine}


def worker_result(result: dict) -> dict:
    """Called after the established worker-result validation, not a second completion gate."""
    status, code = result.get("status"), result.get("error_code")
    from lss.core.jobs import _UNSUPPORTED_CODES
    disposition = ("ZERO" if result.get("zero_result") is not None else "PUBLISHED") if status in {"ok", "skipped"} else (
        "UNSUPPORTED" if code in _UNSUPPORTED_CODES else "FAILED")
    return project(disposition=disposition, outputs=result.get("outputs") or [], reused=status == "skipped",
                   zero_result=result.get("zero_result"), error_code=code, listening=result.get("listening") or [],
                   run={"source": {"sha256": result.get("input_sha256")}})
