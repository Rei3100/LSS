"""Published-cue memory: the same recording published from different sources is recognised, numbered alike and
named alike; nothing about the audio, publication or CurrentBest changes, and unsupported relations never group."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from mutagen.flac import FLAC

from lss.bgm import cue_memory, cue_publication, listening
from lss.output import result_report, results

RATE = 22050


def _music(seed: int, seconds: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    note = int(0.25 * RATE)
    count = int(seconds * RATE) // note + 1
    t = np.arange(note) / RATE
    envelope = np.minimum(1.0, t / 0.01) * np.exp(-3 * t)
    out = np.zeros(count * note)
    for index in range(count):
        chunk = np.zeros(note)
        for midi in rng.integers(48, 84, size=3):
            frequency = 440.0 * 2 ** ((midi - 69) / 12)
            chunk += np.sin(2 * np.pi * frequency * t) + 0.5 * np.sin(4 * np.pi * frequency * t)
        out[index * note:(index + 1) * note] = chunk * envelope
    out = out[:int(seconds * RATE)]
    return out / np.max(np.abs(out)) * 0.4


def _residue(seconds: float, seed: int) -> np.ndarray:
    """Dialogue-like residue a separator leaves, different in every occurrence."""
    rng = np.random.default_rng(seed)
    size = int(seconds * RATE)
    voice = np.zeros(size)
    position = 0
    while position < size:
        length = int(rng.uniform(0.2, 0.6) * RATE)
        pitch = rng.uniform(100, 250)
        t = np.arange(min(length, size - position)) / RATE
        burst = sum(np.sin(2 * np.pi * pitch * k * t) / k for k in range(1, 12))
        voice[position:position + t.size] = 0.03 * burst * np.hanning(t.size)
        position += length + int(rng.uniform(0.3, 1.2) * RATE)
    return voice


RECORDING = _music(1, 60.0)
OTHER = _music(2, 40.0)


def _cue(folder: Path, source: str, start: float, mono: np.ndarray, *, seed: int, unresolved: bool = False) -> Path:
    audio = mono + _residue(mono.size / RATE, seed)
    stereo = np.stack([audio, audio * 0.9], axis=1)
    sha = (source.encode().hex() * 64)[:64]
    start_frame = int(start * RATE)
    end_frame = start_frame + stereo.shape[0]
    key = cue_publication.cue_key(sha, "recipe", start_frame, end_frame, RATE)
    provenance = {"schema_version": 1, "type": "bgm_published_cue",
                  "source": {"path": str(folder / f"{source}.mkv"), "sha256": sha},
                  "start_frame": start_frame, "end_frame": end_frame, "sample_rate": RATE,
                  "start_seconds": start, "end_seconds": start + stereo.shape[0] / RATE,
                  "cue_extent": {"state": "unresolved" if unresolved else "single_cue"}}
    name = cue_publication.file_name(Path(f"{source}.mkv"), start, unresolved_extent=unresolved)
    record = cue_publication.publish(stereo, RATE, folder / f"{source} - BGM" / name, key=key,
                                     title=cue_publication.title(Path(f"{source}.mkv"), start), album=f"{source} BGM",
                                     provenance=provenance, input_sha256=sha, recipe_id="recipe", version="test")
    return Path(record["path"])


def _observe(state: Path, *paths: Path, source: str | None = None) -> dict:
    items = [cue_memory.published_item(path) for path in paths]
    sha = items[0]["source_sha256"] if items else (source.encode().hex() * 64)[:64]
    return cue_memory.observe(state, sha, items)


@pytest.fixture()
def library(tmp_path):
    first = _cue(tmp_path, "ep01", 100.0, RECORDING[: int(50 * RATE)], seed=11)
    unrelated = _cue(tmp_path, "ep01", 400.0, OTHER, seed=12)
    later = _cue(tmp_path, "ep02", 300.0, RECORDING[int(20 * RATE): int(45 * RATE)], seed=13)
    return tmp_path, first, unrelated, later


def test_same_recording_in_another_episode_is_one_numbered_group(library):
    root, first, unrelated, later = library
    state = root / "state"
    alone = _observe(state, first, unrelated)
    assert alone["by_path"] == {} and alone["changed_elsewhere"] == []
    found = _observe(state, later)
    view = found["by_path"][str(later)]
    assert view["label"] == "劇伴#1" and view["group_size"] == 2 and view["longest"] is False
    assert [other["source_name"] for other in view["others"]] == ["ep01"]
    assert view["others"][0]["start_seconds"] == 100.0
    # The earlier episode's cue joins the same group and must have its presentation refreshed.
    assert [item["path"] for item in found["changed_elsewhere"]] == [str(first)]
    assert found["changed_elsewhere"][0]["view"]["label"] == "劇伴#1"
    assert found["changed_elsewhere"][0]["view"]["longest"] is True
    data = cue_memory.read(state)
    (link,) = data["links"]
    shares = sorted([link["a_share"], link["b_share"]])
    assert shares[0] > 0.4 and shares[1] > 0.9 and link["waveform_explained"] >= cue_memory.MIN_EXPLAINED
    # Observing the same outputs again compares nothing new and changes nothing.
    again = _observe(state, later)
    assert again["by_path"] == found["by_path"] and again["changed_elsewhere"] == []
    assert cue_memory.read(state)["links"] == data["links"]


def test_unrelated_music_never_groups(library):
    root, first, unrelated, later = library
    state = root / "state"
    _observe(state, first, unrelated)
    _observe(state, later)
    views = cue_memory._views(cue_memory.read(state))
    assert f"bgm-cue:{cue_memory.published_item(unrelated)['cue_key']}" not in views


def test_unresolved_cue_with_two_recordings_cannot_merge_them(tmp_path):
    state = tmp_path / "state"
    both = np.concatenate([RECORDING[: int(20 * RATE)], OTHER[: int(20 * RATE)]])
    mixed = _cue(tmp_path, "ep01", 10.0, both, seed=21, unresolved=True)
    first_only = _cue(tmp_path, "ep02", 50.0, RECORDING[: int(20 * RATE)], seed=22)
    second_only = _cue(tmp_path, "ep03", 70.0, OTHER[: int(20 * RATE)], seed=23)
    _observe(state, mixed)
    _observe(state, first_only)
    _observe(state, second_only)
    data = cue_memory.read(state)
    assert len(data["links"]) == 2  # both relations are kept as evidence
    ids = {name: f"bgm-cue:{cue_memory.published_item(path)['cue_key']}"
           for name, path in (("first", first_only), ("second", second_only), ("mixed", mixed))}
    # The unresolved cue joins one of the recordings it holds; the two recordings never share a group.
    (grouped,) = [set(keys) for keys in cue_memory.groups(data).values() if len(keys) > 1]
    assert ids["mixed"] in grouped and not {ids["first"], ids["second"]} <= grouped


def test_retired_or_missing_cues_leave_the_group(library):
    root, first, unrelated, later = library
    state = root / "state"
    _observe(state, first, unrelated)
    _observe(state, later)
    # ep01 reprocessed and no longer publishes the cue: the group dissolves, ep02's label must be removed.
    found = _observe(state, unrelated)
    assert found["by_path"] == {}
    item = cue_memory.published_item(later)
    assert found["changed_elsewhere"] == [{"path": str(later), "pcm_sha256": item["pcm_sha256"], "view": None,
                                           "source_sha256": item["source_sha256"], "asset_id": f"bgm-cue:{item['cue_key']}"}]
    assert cue_memory.read(state)["retired"][0]["path"] == str(first)


def test_a_missing_file_is_not_compared(library):
    root, first, unrelated, later = library
    state = root / "state"
    _observe(state, first)
    first.unlink()
    assert _observe(state, later)["by_path"] == {}
    assert cue_memory.read(state)["links"] == []


def test_listening_names_and_numbers_the_group_and_removes_the_number_when_it_dissolves(library):
    root, first, unrelated, later = library
    state = root / "state"
    _observe(state, first, unrelated)
    listening.finish(first, state)
    found = _observe(state, later)
    finished = listening.finish(later, state, recurrence=found["by_path"][str(later)])
    first_title = None
    for item in found["changed_elsewhere"]:
        refreshed = listening.finish(Path(item["path"]), state, expected_pcm=item["pcm_sha256"], recurrence=item["view"])
        first_title = refreshed["naming"]["base_title"]
    tags_later, tags_first = FLAC(later), FLAC(first)
    assert "（仮題・劇伴#1）" in tags_later["TITLE"][0] and "（仮題・劇伴#1）" in tags_first["TITLE"][0]
    # The later occurrence is described like the first one (the group's anchor), not by its own measurement.
    assert finished["naming"]["base_title"] == first_title
    assert finished["naming"]["kind"] == "provisional_descriptive"
    recorded = json.loads(tags_later[listening.RECURRENCE_TAG][0])
    assert recorded["label"] == "劇伴#1" and recorded["others"][0]["source_name"] == "ep01"
    # The group dissolves (ep01 no longer publishes that cue): ep02's relation and number are removed, and a
    # title another program wrote meanwhile is replaced by LSS's.
    audio = FLAC(later)
    audio["TITLE"] = "他のソフトが付けた名前"
    audio.save()
    dissolved = _observe(state, unrelated)
    assert [item["path"] for item in dissolved["changed_elsewhere"]] == [str(later)]
    for item in dissolved["changed_elsewhere"]:
        listening.finish(Path(item["path"]), state, expected_pcm=item["pcm_sha256"], recurrence=item["view"])
    assert "劇伴#1" not in FLAC(later)["TITLE"][0] and FLAC(later)["TITLE"] != ["他のソフトが付けた名前"]
    assert listening.RECURRENCE_TAG not in FLAC(later)


def test_a_verified_related_title_travels_to_the_same_recording(library):
    root, first, unrelated, later = library
    state = root / "state"
    _observe(state, first, unrelated)
    from lss.assets.naming import NamingStore
    asset = f"bgm-cue:{cue_memory.published_item(first)['cue_key']}"
    related = {"related": {"asset_id": asset, "title": "ゆるさんぽ", "verified": True,
                           "relation": "same_cue_different_version", "evidence_ref": "reference_relation:x"}}
    listening.finish(first, state, extra_evidence=related)
    found = _observe(state, later)
    finished = listening.finish(later, state, recurrence=found["by_path"][str(later)])
    assert finished["naming"]["kind"] == "provisional_related"
    assert FLAC(later)["TITLE"][0].endswith("ゆるさんぽ — 関連版（仮題・劇伴#1）")
    assert NamingStore(state / listening.DIRECTORY).read(asset)["revisions"][-1]["title"] == "ゆるさんぽ — 関連版"


def test_projection_and_report_say_what_recurred(library):
    root, first, unrelated, later = library
    state = root / "state"
    _observe(state, first, unrelated)
    found = _observe(state, later)
    projection = results.project(disposition="PUBLISHED", outputs=[later], recurrence={**found,
                                 "refreshed_elsewhere": {"refreshed": 1, "pending": []}})
    assert projection["recurring_count"] == 1 and projection["recurrence_refreshed_elsewhere"] == 1
    assert projection["recurrence"][0]["others"][0]["source_name"] == "ep01"
    page = result_report.render(Path("ep02.mkv"), "bgm", projection)
    assert "前に保存した曲と同じ録音：1曲" in page and "劇伴#1" in page and "ep01 01:40" in page
    unchecked = results.project(disposition="PUBLISHED", outputs=[later],
                                recurrence={"state": "not_checked", "by_path": {}})
    assert unchecked["recurring_count"] is None  # unknown is not zero
    assert "照合はできませんでした" in result_report.render(Path("ep02.mkv"), "bgm", unchecked)
    assert "recurring_count" not in results.project(disposition="PUBLISHED", outputs=[later])


def test_a_changed_rule_compares_everyone_again(library, monkeypatch):
    root, first, unrelated, later = library
    state = root / "state"
    _observe(state, first, unrelated)
    _observe(state, later)
    from lss.bgm import stretch_match

    monkeypatch.setattr(cue_memory, "MIN_EXPLAINED", 0.99)
    monkeypatch.setattr(stretch_match, "MIN_MEAN_EXPLAINED", 0.99)
    data = cue_memory.read(state)
    assert data["links"] == [] and all(not member["compared_with"] for member in data["members"].values())
    assert _observe(state, later)["by_path"] == {}


def _stretch(signal: np.ndarray, rate: float, segment_seconds: float = 0.25, fade_seconds: float = 0.01) -> np.ndarray:
    """Pitch-preserving tempo change: output time u plays input time u * rate (segments joined by short fades)."""
    segment, fade = int(segment_seconds * RATE), int(fade_seconds * RATE)
    length = int(signal.size / rate)
    out = np.zeros(length + segment + fade)
    norm = np.zeros_like(out)
    window = np.ones(segment + fade)
    window[:fade], window[-fade:] = np.linspace(0, 1, fade), np.linspace(1, 0, fade)
    for start in range(0, length, segment):
        piece = signal[int(start * rate):int(start * rate) + segment + fade]
        if piece.size < segment + fade:
            break
        out[start:start + piece.size] += piece * window
        norm[start:start + piece.size] += window
    return (out / np.maximum(norm, 1e-3))[:length]


def test_the_same_recording_at_another_tempo_groups_and_says_so(tmp_path):
    """Editors fit a score to a scene by time stretching it; a constant offset cannot verify that pair."""
    state = tmp_path / "state"
    first = _cue(tmp_path, "ep01", 100.0, RECORDING[: int(40 * RATE)], seed=21)
    slower = _cue(tmp_path, "ep02", 300.0, _stretch(RECORDING[: int(40 * RATE)], 1 / 1.04)[: int(38 * RATE)], seed=22)
    unrelated = _cue(tmp_path, "ep02", 500.0, OTHER, seed=23)
    _observe(state, first)
    found = _observe(state, slower, unrelated)
    assert str(unrelated) not in found["by_path"]
    view = found["by_path"][str(slower)]
    assert view["label"] == "劇伴#1" and view["group_size"] == 2
    (link,) = cue_memory.read(state)["links"]
    assert link["basis"] == "stretched_lag_path" and abs(link["rate"] - 1.04) < 0.01
    assert link["waveform_explained"] >= link["waveform_null_max"] + 0.1
    # ep02 plays it 4 % slower than ep01: seen from ep02, ep01 is about 4 % faster.
    assert abs(view["others"][0]["tempo_ratio"] - 1.04) < 0.01
    projection = results.project(disposition="PUBLISHED", outputs=[slower], recurrence=found)
    page = result_report.render(Path("ep02.mkv"), "bgm", projection)
    assert "テンポ+4" in page
    # The consensus asset maps the passage linearly at that tempo (`recurrence_asset.warp` lays it on its clock).
    from lss.bgm import recurrence_asset

    pieces = recurrence_asset._pieces(link, link["a"])
    assert pieces and all(abs(ratio - link["rate"]) < 0.01 for *_, ratio in pieces)


def test_cues_published_before_the_memory_join_from_current_best(library):
    """A source processed before this memory existed takes part through its CurrentBest record."""
    from lss.bgm import current_best
    root, first, unrelated, later = library
    state = root / "state"
    rows = [{"public_path": str(path), "pcm_sha256": cue_memory.published_item(path)["pcm_sha256"],
             "cue_key": cue_memory.published_item(path)["cue_key"], "start_seconds": start}
            for path, start in ((first, 100.0), (unrelated, 400.0))]
    sha = cue_memory.published_item(first)["source_sha256"]
    record = current_best.record_path(state, sha)
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
                                  "source_sha256": sha, "current": {"files": rows}, "history": [],
                                  "updated_at": "2026-10-01T00:00:00+00:00"}), encoding="utf-8")
    found = _observe(state, later)
    assert found["by_path"][str(later)]["others"][0]["source_name"] == "ep01"
    assert [item["path"] for item in found["changed_elsewhere"]] == [str(first)]
    # An accepted file whose audio is no longer what CurrentBest accepted is left out.
    rows[0]["pcm_sha256"] = "0" * 64
    record.write_text(json.dumps({"type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
                                  "source_sha256": sha, "current": {"files": rows}, "history": [],
                                  "updated_at": "2026-10-01T00:00:00+00:00"}), encoding="utf-8")
    assert _observe(state, later)["by_path"] == {}


def test_two_resolved_cues_sharing_a_passage_are_one_recording(tmp_path):
    """Resolved single cues that share a verified 20 s passage of a recording group, even at under half."""
    state = tmp_path / "state"
    early = _cue(tmp_path, "ep01", 10.0, RECORDING[: int(45 * RATE)], seed=31)
    late = _cue(tmp_path, "ep02", 20.0, RECORDING[int(25 * RATE): int(60 * RATE)], seed=32)
    short = _cue(tmp_path, "ep03", 30.0, RECORDING[int(40 * RATE): int(47 * RATE)], seed=33)
    _observe(state, early)
    found = _observe(state, late)
    assert found["by_path"][str(late)]["group_size"] == 2
    # A 7-second overlap is a quote, not the same cue: below the declared minimum.
    _observe(state, short)
    assert cue_memory.view_for(state, short) is None
