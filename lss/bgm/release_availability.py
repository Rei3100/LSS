"""Independent distribution evidence, separate from waveform identity.

Only a byte-addressed, checked distribution receipt can suppress extraction.
Titles, an episode occurrence and missing catalogue entries remain unknown.
The bundled receipts reuse the existing acquired references; no online service
is needed at runtime. Further acquisition receipts can use the same schema.
"""
from pathlib import Path
import json
import re
from urllib.parse import urlsplit

POLICY = "sha-bound-independent-distribution-v1"


def checked(receipt, sha256):
    if not isinstance(receipt, dict):
        return False
    try:
        urls = [receipt["catalogue_url"], receipt["recording_url"]]
        return bool(receipt.get("state") == "confirmed_independent_distribution"
                    and re.fullmatch(r"[0-9a-f]{64}", sha256 or "")
                    and receipt.get("reference_sha256") == sha256
                    and receipt.get("release_id") and receipt.get("checked_at")
                    and all(urlsplit(u).scheme == "https" and urlsplit(u).hostname for u in urls)
                    and receipt.get("basis") == "checked_release_and_acquired_recording")
    except (KeyError, TypeError, ValueError):
        return False


def lookup(sha256, metadata=None):
    """Only a receipt LSS itself recorded with the reference (when it acquired and checked it), or the user's own
    copy in their reference library (`user_copy`), counts.

    No catalogue of receipts researched during development is shipped (AGENTS.md: a shipped result never leans
    on what someone looked up for it); without LSS's own receipt the recording stays unknown and is preserved.
    """
    candidate = (metadata or {}).get("independent_distribution")
    if available(candidate, sha256):
        return dict(candidate)
    return {"state": "independent_distribution_unknown", "reference_sha256": sha256,
            "extraction": "preserve", "catalogue_absence_is_unreleased": False}


_ALBUM_PATH = re.compile(r"/browse/MPREb_[A-Za-z0-9_-]+")
_ALBUM_LIST = re.compile(r"(?:^|&)list=(OLAK5uy_[A-Za-z0-9_-]+)(?:&|$)")
_TRACK_ID = re.compile(r"[A-Za-z0-9_-]{11}")


def from_record(record) -> dict | None:
    """LSSが自分で取得し、目録の収録盤の曲目で確かめた参照の記録から、配布の受領を作る（無ければNone）。

    受領になるのは、発見（`reference_acquisition`）が収録盤の曲目に同じ曲の識別子を見つけて結び付けた
    （`discovery.release`）YouTube Music の公式の盤（`/browse/MPREb_…` か `list=OLAK5uy_…`）の曲を、その曲の
    識別子のまま取得した記録だけ。利用者の音源・名前だけの手掛かり・盤の無い曲の検索結果は受領にしない。
    """
    if not isinstance(record, dict) or record.get("origin") != "acquired":
        return None
    sha256 = record.get("sha256") or ""
    discovery = record.get("discovery") or {}
    release = discovery.get("release") or {}
    track_id = str(discovery.get("track_id") or "")
    provenance = record.get("provenance") or {}
    authority = provenance.get("authority") or {}
    album_url = str(release.get("album_url") or "")
    try:
        parts = urlsplit(album_url)
    except ValueError:
        return None
    if parts.scheme != "https" or parts.hostname != "music.youtube.com":
        return None
    album = _ALBUM_PATH.fullmatch(parts.path or "")
    listed = _ALBUM_LIST.search(parts.query or "")
    release_id = (parts.path.rsplit("/", 1)[-1] if album else listed.group(1) if listed else None)
    if (not release_id or not _TRACK_ID.fullmatch(track_id) or str(authority.get("id") or "") != track_id
            or not re.fullmatch(r"[0-9a-f]{64}", sha256) or not provenance.get("acquired_at")):
        return None
    receipt = {"state": "confirmed_independent_distribution", "reference_sha256": sha256,
               "release_id": release_id, "release_title": release.get("album_title"),
               "catalogue_url": album_url, "recording_url": f"https://music.youtube.com/watch?v={track_id}",
               "checked_at": provenance["acquired_at"], "basis": "checked_release_and_acquired_recording"}
    return receipt if checked(receipt, sha256) else None


USER_COPY = "user_owned_release_copy"


def user_copy(record) -> dict | None:
    """利用者が参照の置き場所へ置いた録音は、利用者がすでに持っている既発売の録音（作り直さない）。

    LSS自身が書いた出力（LSS_* のタグ）は除く。区間がその録音そのものかは、関係の判定が音楽の会計で決める。
    """
    if (not isinstance(record, dict) or record.get("origin") != "user_library"
            or (record.get("metadata") or {}).get("lss_output")
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256") or "")):
        return None
    return {"state": USER_COPY, "reference_sha256": record["sha256"], "basis": "recording_in_users_own_library"}


def available(receipt, sha256) -> bool:
    """抽出しなくてよい根拠：LSSの配布の受領、または利用者の手元の写し（同じ内容の録音だけ）。"""
    if isinstance(receipt, dict) and receipt.get("state") == USER_COPY:
        return receipt.get("reference_sha256") == sha256 and bool(re.fullmatch(r"[0-9a-f]{64}", sha256 or ""))
    return checked(receipt, sha256)
