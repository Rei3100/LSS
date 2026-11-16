"""Explicit independent-release truth for controlled audio fixtures."""
def distribution_receipt(sha):
    return {"state":"confirmed_independent_distribution","reference_sha256":sha,
            "release_id":"controlled-fixture","checked_at":"2026-10-14",
            "catalogue_url":"https://example.org/controlled-release",
            "recording_url":"https://example.org/controlled-recording",
            "basis":"checked_release_and_acquired_recording"}
