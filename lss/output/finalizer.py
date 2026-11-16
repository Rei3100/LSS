"""Accepted FLAC -> listening asset. PCM stays unchanged; publication remains journaled by callers.

The embedded receipt links the presentation, measured encoded PCM and upstream provenance.
It deliberately cannot contain the file's own encoded checksum; callers record that after tagging.

Ownership: the tags, ReplayGain values, picture set and library information of a file LSS created are
LSS-owned Product state. LSS writes the complete set from the current naming/measurement record and the
receipt validates all of it; a file whose tags or picture differ from its receipt is not current and is
rewritten. Tags LSS does not write (and files LSS did not create) are never touched.
"""
from __future__ import annotations

import re

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import soundfile as sf
from mutagen.flac import FLAC, Picture
from mutagen import MutagenError

from lss.core import audio_identity
from lss.output.scan import scan_track_file
from lss.output.tags import recalculated_tags

REVISION = "listening-flac-v1"
TARGET_LUFS = -18.0
PLAYBACK_CEILING_DBTP = -1.0
RECEIPT_TAG = "LSS_LISTENING_FINALIZATION"
RECEIPT_SCHEMA_VERSION = 2
POLICY = {"revision": REVISION, "signal_processing": "none", "loudness": "track_replaygain",
          "target_lufs": TARGET_LUFS, "playback_ceiling_dbtp": PLAYBACK_CEILING_DBTP,
          "gain_policy": "target_gain_limited_by_measured_true_peak",
          "true_peak_method": "scipy_resample_poly_4x_estimate",
          "album_gain": "not_inferred"}
PRESENTATION_KEYS = frozenset({"TITLE", "ARTIST", "ALBUM", "ALBUMARTIST", "DATE", "GENRE",
                              "TRACK", "TRACKNUMBER", "DISCNUMBER", "VOCAL_TYPE",
                              "M_NUMBER", "ORDER_MENU", "VERSION", "BPM", "INITIALKEY"})
# Keys the listening measurement writes (`lss.output.tags.recalculated_tags`). Album gain is never LSS-owned.
MEASUREMENT_KEYS = frozenset({"DYNAMIC RANGE", "REPLAYGAIN_ALGORITHM", "REPLAYGAIN_TRACK_GAIN",
                              "REPLAYGAIN_TRACK_PEAK", "REPLAYGAIN_TRACK_RANGE",
                              "TRUEPEAK_SCANNER_CLIPPED_SAMPLES_TRACK", "TRUEPEAK_SCANNER_PEAK_POSITION",
                              "TRUEPEAK_SCANNER_TRACK_PLR"})
WRITABLE_KEYS = PRESENTATION_KEYS | MEASUREMENT_KEYS
# Music-library metadata keys are open-ended (one tag per credited role, `lss.metadata`); the file names the ones
# LSS projected in this tag, so they are LSS-owned presentation like the fixed keys above.
METADATA_KEYS_TAG = "LSS_METADATA_KEYS"
_METADATA_KEY = re.compile(r"^[A-Z0-9][A-Z0-9_ ]{0,63}$")


def metadata_keys(tags) -> frozenset:
    """Dynamic metadata keys a tag mapping declares through ``LSS_METADATA_KEYS``."""
    values = tags.get(METADATA_KEYS_TAG) or tags.get(METADATA_KEYS_TAG.lower()) or []
    values = [values] if isinstance(values, str) else values
    return frozenset(key for value in values for key in str(value).split("\n") if _METADATA_KEY.match(key))




class FinalizationError(ValueError):
    pass


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _values(tags: dict) -> dict[str, list[str]]:
    return {key.upper(): [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]
            for key, value in tags.items() if value is not None and value != "" and value != []}


def picture_identity(picture: Picture | None) -> dict | None:
    if picture is None:
        return None
    return {"sha256": hashlib.sha256(picture.data).hexdigest(), "mime": picture.mime,
            "type": int(picture.type), "description": picture.desc}


def _request(tags: dict, picture: Picture | None) -> dict:
    return {"tags": _values(tags), "artwork": picture_identity(picture), "policy": POLICY}


def _audio(path: Path) -> dict:
    identity = audio_identity.audio_identity(path)
    info = sf.info(path)
    if (info.format != "FLAC" or info.frames <= 0 or identity["frames"] != info.frames
            or info.channels != 2 or info.subtype not in {"PCM_16", "PCM_24"}):
        raise FinalizationError("聴取用音声の形式・長さ・左右が不正です")
    return identity


def _file_tags(audio: FLAC) -> dict[str, list[str]]:
    tags: dict[str, list[str]] = {}
    for key, value in audio.items():
        tags.setdefault(key.upper(), []).extend(str(item) for item in value)
    return tags


def receipt(path: Path, *, tags: dict | None = None, picture: Picture | None = None) -> dict | None:
    """Validate the receipt, LSS's tags/art and the entire PCM. Tags LSS never writes are ignored.

    A record written by an older LSS that left tags or the picture set to someone else (``user_owned`` /
    ``artwork_owner``) is readable but not current: LSS owns the whole set, so the file is rewritten."""
    try:
        audio = FLAC(path)
        record = json.loads(audio[RECEIPT_TAG][0])
        if record.get("policy") != POLICY or record.get("audio") != _audio(path):
            return None
        expected = record["projected_tags"]
        if not isinstance(expected, dict) or "VOCAL_TYPE" in expected:
            return None  # Source vocal attributes no longer describe a separated representation.
        if any(audio.get(key) != value for key, value in expected.items()):
            return None
        if record.get("user_owned") or record.get("artwork_owner", "lss") == "user":
            return None
        pictures = [picture_identity(item) for item in audio.pictures]
        if pictures != ([record["artwork"]] if record["artwork"] else []):
            return None
        if tags is not None and record.get("request_sha256") != _digest(_request(tags, picture)):
            return None
        return record
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, IndexError, MutagenError):
        return None


def previous_record(path: Path) -> dict | None:
    """The last receipt LSS embedded, read leniently (it may no longer validate): the prior tag projection."""
    try:
        record = json.loads(FLAC(path)[RECEIPT_TAG][0])
        if isinstance(record, dict) and isinstance(record.get("projected_tags"), dict):
            return record
    except (OSError, ValueError, KeyError, TypeError, IndexError, MutagenError):
        pass
    return None


def _reusable_measurement(previous: dict | None, audio: dict) -> dict | None:
    """The previous scan of exactly these PCM samples under the same policy, if it is complete."""
    if not previous or previous.get("policy") != POLICY or previous.get("audio") != audio:
        return None
    scan = previous.get("measurement")
    required = ("samplerate", "loudness_lufs", "sample_peak_linear", "loudness_measured", "track_gain_db",
                "loudness_range_lu", "true_peak_dbtp", "true_peak_position_sample", "clipped_samples",
                "plr_db", "dynamic_range_db")
    if not isinstance(scan, dict) or any(name not in scan for name in required):
        return None
    restored = dict(scan)
    for name in ("loudness_lufs", "true_peak_dbtp"):
        if restored[name] is None:
            restored[name] = float("-inf")
    return restored


def finalize_staged(path: Path, *, tags: dict, picture: Picture | None = None, scanner=None,
                    measurement: dict | None = None) -> dict:
    """Complete and verify an unpublished file. No rename and no writes to source/master media.

    LSS writes its complete tag set and picture. ``measurement`` reuses a prior scan of the same PCM samples
    (callers check the audio identity).
    """
    path = Path(path)
    request = _request(tags, picture)
    projected = request["tags"]
    declared = metadata_keys(projected)
    if not all(key in PRESENTATION_KEYS or key.startswith("LSS_") or key in declared for key in projected):
        raise FinalizationError("聴取用タグに未定義の継承項目があります")
    if "VOCAL_TYPE" in projected:
        projected = {**projected, "LSS_SOURCE_VOCAL_TYPE": projected["VOCAL_TYPE"]}
        projected.pop("VOCAL_TYPE")
    before = _audio(path)
    if measurement is None:
        measurement = (scanner or scan_track_file)(path)
    measured = bool(measurement["loudness_measured"])
    peak = measurement["true_peak_dbtp"]
    if measured and (not math.isfinite(measurement["loudness_lufs"]) or not math.isfinite(peak)):
        raise FinalizationError("聴取用音声の音量走査が不正です")
    if not math.isfinite(measurement["sample_peak_linear"]) or measurement["sample_peak_linear"] < 0:
        raise FinalizationError("聴取用音声のピーク走査が不正です")
    listening = recalculated_tags(measurement)
    if not measured or before["frames"] < 3 * before["samplerate"]:
        listening.pop("DYNAMIC RANGE", None)
    if not measured or before["frames"] < 6 * before["samplerate"]:
        listening.pop("REPLAYGAIN_TRACK_RANGE", None)
    requested_gain = TARGET_LUFS - measurement["loudness_lufs"] if measured else None
    playback_gain = min(requested_gain, PLAYBACK_CEILING_DBTP - peak) if measured else None
    # Round downward so the written gain cannot exceed the declared playback ceiling.
    if playback_gain is not None:
        playback_gain = math.floor(playback_gain * 100.0) / 100.0
        listening["REPLAYGAIN_TRACK_GAIN"] = f"{playback_gain:.2f} dB"
        listening["LSS_LISTENING_REQUESTED_GAIN_DB"] = f"{requested_gain:.6f}"
        listening["LSS_LISTENING_GAIN_LIMITED"] = "1" if playback_gain < requested_gain - 0.01 else "0"
        listening["LSS_INTEGRATED_LOUDNESS_LUFS"] = f"{measurement['loudness_lufs']:.6f}"
    if math.isfinite(peak):
        listening["LSS_TRUE_PEAK_DBTP"] = f"{peak:.6f}"
    listening["LSS_LISTENING_POLICY"] = REVISION
    listening["LSS_LISTENING_TARGET_LUFS"] = str(TARGET_LUFS)
    listening["LSS_LISTENING_PLAYBACK_CEILING_DBTP"] = str(PLAYBACK_CEILING_DBTP)
    projected = {**projected, **_values(listening)}
    clean_scan = {key: value if not isinstance(value, float) or math.isfinite(value) else None
                  for key, value in measurement.items()}
    record = {"schema_version": RECEIPT_SCHEMA_VERSION, "policy": POLICY, "audio": before,
              "request_sha256": _digest(request), "projected_tags": projected,
              "artwork": picture_identity(picture), "measurement": clean_scan,
              "requested_gain_db": requested_gain, "playback_gain_db": playback_gain,
              "upstream_provenance_sha256": _digest({k: v for k, v in request['tags'].items()
                                                     if k.startswith('LSS_')})}
    audio = FLAC(path)
    owned_dynamic = metadata_keys(_file_tags(audio)) | declared
    for key in list(audio):
        upper = key.upper()
        if upper.startswith("LSS_") or upper in WRITABLE_KEYS or upper in owned_dynamic:
            del audio[key]
    for key, values in projected.items():
        audio[key] = values
    audio[RECEIPT_TAG] = _json(record)
    audio.clear_pictures()
    if picture is not None:
        audio.add_picture(picture)
    audio.save()
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())
    if receipt(path, tags=tags, picture=picture) != record:
        raise FinalizationError("聴取用音声のタグ・音・来歴の検証に失敗しました")
    return record


def _result(path: Path, record: dict, *, reused: bool) -> dict:
    title = (FLAC(path).get("TITLE") or [None])[0]
    return {"path": str(path), "sha256": audio_identity.file_digest(path), "audio": record["audio"],
            "finalization": record, "reused": reused, "title": title}


def ensure(path: Path, *, tags: dict, picture: Picture | None = None, validate_before_publish=None) -> dict:
    """Upgrade an accepted file atomically, copy-on-write (history may hard-link the previous bytes).

    Valid receipts require no scan/tag rewrite. Failure preserves the old valid asset.
    This is a presentation update, never a new acoustic result or CurrentBest quality decision.
    LSS rewrites its complete tag set and picture; the previous receipt only supplies a reusable measurement.
    """
    path = Path(path)
    current = receipt(path, tags=tags, picture=picture)
    if current is not None:
        return _result(path, current, reused=True)
    original = audio_identity.file_digest(path)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".lss-listening-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        shutil.copyfile(path, temporary)
        measurement = _reusable_measurement(previous_record(temporary), _audio(temporary))
        record = finalize_staged(temporary, tags=tags, picture=picture, measurement=measurement)
        if validate_before_publish is not None:
            validate_before_publish()
        if audio_identity.file_digest(path) != original:
            raise FinalizationError("聴取用仕上げの間に公開音声が変更されました")
        os.replace(temporary, path)
        temporary = None
        return _result(path, record, reused=False)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

