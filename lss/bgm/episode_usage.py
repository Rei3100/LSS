"""Positive onset nominations from a per-episode music-usage table: none are shipped.

A development snapshot of one fan site's per-episode tables used to be bundled here and read for sources whose
file names matched one series. A shipped result may not lean on what was looked up during development
(AGENTS.md), so nothing is bundled and no file-name pattern selects a work. Callers keep the same contract:
no nominations, normal single-source audio formation unchanged. A table LSS acquires itself at runtime for the
user's own work can be passed in through ``table`` with the same row form.
"""
MAXIMUM_CLOCK_SPREAD = 3.0


def nominate(source, matches, *, reference_metadata=None, table=None):
    if not table:
        return {"state": "no_usage_table", "onsets": []}
    return {"state": "table_not_supported_yet", "onsets": []}
