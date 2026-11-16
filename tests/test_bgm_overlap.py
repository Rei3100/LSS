"""GPU/CPU overlap (`lss.bgm.overlap`) cannot change any output byte, order or record.

The whole Product BGM path runs twice on the same synthetic episode, serially (``LSS_BGM_OVERLAP=0``) and with the
CPU lane, through the real decode, planning, candidate store, semantics, dialogue residual, publication, listening
finish and CurrentBest code. Only the GPU inferences are deterministic fakes (seeded numpy transforms) that also
sleep, so CPU work really runs while a "GPU" stage is busy. Lane tasks get random start delays to shuffle their
order. Every file of both cases is then compared: public FLAC bytes, candidate FLAC bytes, every JSON record
(receipts, ``episode-cues.json``, ``cue-semantics.json``, ``run.json``, CurrentBest, naming store) after removing
only declared timing fields, and the set of files itself.

libsndfile writes the wall-clock time into the PEAK chunk of every FLOAT WAV, so two runs of the same code a
second apart never have byte-identical FLOAT WAVs, nor the hashes and candidate ids derived from them. The
``no_peak_chunk`` fixture turns that chunk off for both runs alike in the byte-identity tests, which leaves byte identity a property of the
code alone (`cloud-reports/gpu-cpu-overlap.md`).
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root


import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss import application
from lss.bgm import episode_cues, event_tagging, four_stem_analysis, overlap, sfc_analysis
from lss.core import quitting
from lss.output import writer
from tests import test_application_bgm_candidates as product

gpu = product.gpu
instrumental = product.instrumental
RATE = product.RATE
# Sleep of every fake GPU call. ``LSS_OVERLAP_PROOF_GPU_SECONDS`` raises it to measure the synthetic wall-clock gain.
GPU_SECONDS = float(os.environ.get("LSS_OVERLAP_PROOF_GPU_SECONDS", "0.4"))
TIMING = re.compile(r"^(elapsed_seconds|stage_seconds|publication_parts|peak_vram_mb|utc|timestamp|duration_seconds|generated_at)$"
                    r"|_at$|_utc$")
HEX = re.compile(r"[0-9a-f]{64}")
CUES = ((5, 27, 330.0), (37, 60, 440.0), (70, 93, 262.0))
SPEECH = ((8, 14), (40, 46), (76, 80))


@pytest.fixture
def no_peak_chunk(monkeypatch):
    from soundfile import _ffi, _snd

    opened = sf.SoundFile.__init__

    def without_peak(self, *args, **kwargs):
        opened(self, *args, **kwargs)
        if "w" in self.mode or "+" in self.mode:
            _snd.sf_command(self._file, 0x1050, _ffi.NULL, 0)  # SFC_SET_ADD_PEAK_CHUNK, false

    monkeypatch.setattr(sf.SoundFile, "__init__", without_peak)


def _episode(path: Path):
    """Three music cues with dialogue and clicks over them; the first-pass music leaks the clicks."""
    frames = 100 * RATE
    rng = np.random.default_rng(11)
    tone = np.zeros((frames, 2))
    dialogue = np.zeros((frames, 2))
    clicks = np.zeros((frames, 2))
    for start, end, frequency in CUES:
        t = np.arange((end - start) * RATE) / RATE
        tone[start * RATE:end * RATE] = (0.05 * np.sin(2 * np.pi * frequency * t))[:, None] * [1.0, 0.7]
        for second in range(start + 1, end - 1, 2):
            clicks[second * RATE:second * RATE + RATE // 10] = rng.normal(0, 0.3, (RATE // 10, 1))
    for start, end in SPEECH:
        dialogue[start * RATE:end * RATE] = rng.normal(0, 0.1, ((end - start) * RATE, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, (tone + dialogue + clicks).astype(np.float32), RATE, subtype="FLOAT")
    return tone, dialogue, clicks


def _slow(function):
    def run(*args, **kwargs):
        time.sleep(GPU_SECONDS)
        return function(*args, **kwargs)
    return run


def _cue_of(second: int) -> int:
    return next((index for index, (start, end, _) in enumerate(CUES) if start <= second < end), len(CUES))


@pytest.fixture
def slow_gpu(gpu, instrumental, monkeypatch):
    """Every fake GPU stage sleeps; the tagger hears speech in the dialogue windows and one song per cue."""
    gpu.tagger.singing = 0.0
    gpu.tagger.speech = lambda second: 0.6 if any(a <= second < b for a, b in SPEECH) else 0.0
    gpu.tagger.embedding = lambda second: np.eye(event_tagging.EMBEDDING_SIZE)[_cue_of(second)]
    monkeypatch.setattr(event_tagging, "tag_signal", _slow(lambda *a, **k: gpu.tagger(*a, **k)))
    monkeypatch.setattr(four_stem_analysis, "separate_spans", _slow(lambda *a, **k: gpu.four_stem(*a, **k)))
    from lss.harness import infer
    monkeypatch.setattr(infer, "msst_separate", _slow(lambda *a, **k: instrumental(*a, **k)))
    return gpu


class Case:
    """One finished run, observed at the shared case path before it is moved aside."""

    def __init__(self, outcome, seconds, snapshot, flac_bytes, leftovers, run, public):
        self.outcome, self.seconds, self.snapshot = outcome, seconds, snapshot
        self.flac_bytes, self.leftovers, self.run, self.public = flac_bytes, leftovers, run, public


def _run_case(tmp_path: Path, name: str, monkeypatch, *, overlapped: bool, seed: int | None = None) -> Case:
    """Run the Product at ``tmp_path/case`` (the same absolute path for every run, so recorded paths and file bytes
    can be compared exactly), observe everything, then move the case aside to ``tmp_path/name``."""
    root = tmp_path / "case"
    source = root / "media" / "episode.wav"
    master = tmp_path / "episode.wav"  # one source file for every run: the same bytes, the same source identity
    if not master.is_file():
        _episode(master)
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(master, source)
    leak = product._LeakAnalyzer(*_episode(tmp_path / "components.wav"), second="clean")
    monkeypatch.setattr(sfc_analysis, "separate_stems", _slow(lambda *a, **k: leak(*a, **k)))
    monkeypatch.setenv("LSS_STATE_DIR", str(root / "state"))
    monkeypatch.setenv("LSS_REFERENCES_DIR", str(root / "References"))
    monkeypatch.setenv(overlap.ENV, "1" if overlapped else "0")
    rng = random.Random(seed)
    lock = threading.Lock()

    def jitter(label):
        with lock:
            pause = rng.uniform(0.0, 0.08)
        time.sleep(pause)

    monkeypatch.setattr(overlap, "_TASK_HOOK", jitter if seed is not None else None)
    started = time.perf_counter()
    outcome = application.run_bgm_candidate_request(
        application.ProcessingRequest((source,), application.UserIntent.EXTRACT_BGM_CUES), state_root=root / "state")
    seconds = time.perf_counter() - started
    assert not [thread for thread in threading.enumerate() if thread.name.startswith("lss-bgm-cpu")]
    run = (json.loads(outcome.record_path.read_text(encoding="utf-8"))
           if outcome.record_path is not None and outcome.record_path.is_file() else None)
    case = Case(outcome, seconds, _snapshot(root), _flac_bytes(root), _leftovers(root), run,
                [Path(path).name for path in outcome.outputs])
    root.rename(tmp_path / name)
    return case


def _normalized(value, root: str):
    if isinstance(value, str):
        value = value.replace(root, "<CASE>")
        if value[:1] in "{[":
            try:
                return _normalized(json.loads(value), root)
            except ValueError:
                pass
        return value
    if isinstance(value, dict):
        value = dict(value)
        if isinstance(value.get("file_stat"), list):
            # (size, mtime_ns) of a public file in the cue memory: the size is content, the mtime is a time.
            value["file_stat"] = value["file_stat"][:1]
        if "removed_bytes" in value and "removed_files" in value:
            # The retired scratch includes worker receipts whose elapsed_seconds have a varying number of digits;
            # the count and every kept path are still compared.
            value.pop("removed_bytes")
        return {key: _normalized(item, root) for key, item in value.items() if not TIMING.search(key)}
    if isinstance(value, list):
        return [_normalized(item, root) for item in value]
    return value


def _flac_view(path: Path, root: str) -> dict:
    """A FLAC's exact audio stream bytes plus its tags (JSON tags normalised like records)."""
    from mutagen.flac import FLAC
    from lss.core import audio_identity

    data = path.read_bytes()
    layout = audio_identity.container_layout(path)
    stream = data[layout["audio_offset"]:]
    tags = {key: [_normalized(item, root) for item in values] for key, values in FLAC(path).tags.as_dict().items()}
    return {"audio_stream_sha256": hashlib.sha256(stream).hexdigest(), "tags": _normalized(tags, root)}


def _snapshot(root: Path) -> dict:
    """Every file of a case: JSON normalised, FLAC as exact audio stream + tags, anything else as bytes.

    Recorded hashes of JSON records (whose timing fields differ by design) are replaced by the hash of their
    normalised content, so a record that differs anywhere else still differs here.
    """
    text = str(root)
    files, aliases = {}, {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".json":
            body = _normalized(json.loads(path.read_text(encoding="utf-8")), text)
            canonical = json.dumps(body, sort_keys=True, ensure_ascii=False)
            aliases[hashlib.sha256(path.read_bytes()).hexdigest()] = (
                "json:" + hashlib.sha256(canonical.encode()).hexdigest())
            files[relative] = body
        elif path.suffix == ".flac":
            files[relative] = _flac_view(path, text)
        else:
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    rendered = json.dumps(files, sort_keys=True, ensure_ascii=False)
    rendered = HEX.sub(lambda match: aliases.get(match.group(0), match.group(0)), rendered)
    return json.loads(rendered)


def _flac_bytes(root: Path) -> dict:
    """Public and candidate FLACs byte for byte (paths relative to the case)."""
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.flac")) if not path.name.startswith(".")}


def _deep(a, b, path: str) -> list[str]:
    if type(a) is not type(b):
        return [f"{path}: {str(a)[:120]!r} != {str(b)[:120]!r}"]
    if isinstance(a, dict):
        return [line for key in sorted(set(a) | set(b))
                for line in ([f"{path}/{key}: only on one side"] if key not in a or key not in b
                             else _deep(a[key], b[key], f"{path}/{key}"))]
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [line for index, (x, y) in enumerate(zip(a, b)) for line in _deep(x, y, f"{path}[{index}]")]
    return [] if a == b else [f"{path}: {str(a)[:120]!r} != {str(b)[:120]!r}"]


def _differences(a: dict, b: dict) -> list[str]:
    out = [f"only in serial: {key}" for key in sorted(set(a) - set(b))]
    out += [f"only in overlapped: {key}" for key in sorted(set(b) - set(a))]
    for key in sorted(set(a) & set(b)):
        if a[key] != b[key]:
            out.append(f"differs: {key}: " + "; ".join(_deep(a[key], b[key], "")[:3]))
    return out


def _public(outcome) -> list[Path]:
    return sorted(Path(path) for path in outcome.outputs)


def test_overlap_changes_no_output_record_or_order(tmp_path, slow_gpu, monkeypatch, no_peak_chunk):
    serial = _run_case(tmp_path, "serial", monkeypatch, overlapped=False)
    assert serial.outcome.disposition is application.ResultDisposition.PUBLISHED, serial.outcome.message
    run = serial.run
    # The episode exercises what the lane touches: several held cues, the second pass, published processed cues
    # with a regional dialogue residual.
    assert len(run["held"]) >= 3 and run["second_pass"]["status"] == "ran"
    assert any(item["dialogue_residual"]["applied"] for item in run["published"])
    assert "overlap_candidate_cpu" not in run["cost"]["stage_seconds"]
    assert len(serial.flac_bytes) >= 2 * len(run["held"]) + len(run["published"])

    timings = []
    for seed in (1, 2, 3):
        case = _run_case(tmp_path, f"overlapped-{seed}", monkeypatch, overlapped=True, seed=seed)
        assert case.outcome.disposition is application.ResultDisposition.PUBLISHED, case.outcome.message
        stages = case.run["cost"]["stage_seconds"]
        assert stages["overlap_candidate_cpu"] > 0 and stages["overlap_publication_cpu"] > 0
        differences = _differences(serial.snapshot, case.snapshot)
        assert not differences, "\n".join(differences)
        # Public and candidate FLACs byte for byte; same public names in the same order.
        assert case.flac_bytes == serial.flac_bytes
        assert case.public == serial.public
        timings.append(case.seconds)
        lane_cpu = stages["overlap_candidate_cpu"] + stages["overlap_publication_cpu"]
    residual = case.snapshot[next(key for key in case.snapshot if key.endswith("dialogue-residual/receipt.json"))]
    print(f"synthetic wall seconds (fake GPU call {GPU_SECONDS} s): serial {serial.seconds:.2f}, overlapped "
          f"{', '.join(f'{t:.2f}' for t in timings)}; lane CPU {lane_cpu:.2f} s; serial stages "
          f"{serial.run['cost']['stage_seconds']}; overlapped stages {stages}; dialogue-residual inference/requested "
          f"frames {residual['inference_frames']}/{residual['requested_frames']}")


def test_the_proof_sees_a_prefetch_that_writes_different_bytes(tmp_path, slow_gpu, monkeypatch, no_peak_chunk):
    """Mutation check: a prefetch whose evidence differs from the serial loop's leaves a different candidate file,
    which the serial loop silently reuses. The snapshot comparison must catch it."""
    serial = _run_case(tmp_path, "serial", monkeypatch, overlapped=False)
    original = episode_cues._materialize_cut

    def mutated(*args, evidence, **kwargs):
        if threading.current_thread().name.startswith("lss-bgm-cpu"):
            evidence = {**evidence, "method": evidence["method"] + "-mutant"}
        return original(*args, evidence=evidence, **kwargs)

    monkeypatch.setattr(episode_cues, "_materialize_cut", mutated)
    case = _run_case(tmp_path, "overlapped", monkeypatch, overlapped=True, seed=4)
    differences = _differences(serial.snapshot, case.snapshot)
    assert any("/run/candidates/" in line for line in differences), differences
    assert case.flac_bytes != serial.flac_bytes


def test_a_failing_lane_task_is_redone_serially_with_the_same_result(tmp_path, slow_gpu, monkeypatch, no_peak_chunk):
    serial = _run_case(tmp_path, "serial", monkeypatch, overlapped=False)
    original = episode_cues._materialize_cut
    from lss.bgm import cue_publication
    composed = cue_publication.compose
    failures = []

    def failing_cut(*args, **kwargs):
        if threading.current_thread().name.startswith("lss-bgm-cpu") and len(failures) < 3:
            failures.append("cut")
            raise OSError("injected lane failure")
        return original(*args, **kwargs)

    def failing_compose(*args, **kwargs):
        if threading.current_thread().name.startswith("lss-bgm-cpu") and "compose" not in failures:
            failures.append("compose")
            raise ValueError("injected lane failure")
        return composed(*args, **kwargs)

    monkeypatch.setattr(episode_cues, "_materialize_cut", failing_cut)
    monkeypatch.setattr(cue_publication, "compose", failing_compose)
    case = _run_case(tmp_path, "overlapped", monkeypatch, overlapped=True, seed=5)

    assert case.outcome.disposition is serial.outcome.disposition is application.ResultDisposition.PUBLISHED
    assert "compose" in failures and "cut" in failures
    differences = _differences(serial.snapshot, case.snapshot)
    assert not differences, "\n".join(differences)
    assert case.flac_bytes == serial.flac_bytes


def _leftovers(root: Path) -> dict:
    run_dirs = list((root / "state" / "bgm-candidates").glob("*/run"))
    files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
                   and path.relative_to(root).as_posix() != "media/episode.wav")
    return {"public": [p for p in files if p.startswith("media/")],
            "run_records": [p for p in files if p.endswith(("run/run.json", "run/episode-cues.json"))],
            "temporary": [p for p in files if p.endswith(".tmp") or Path(p).name.startswith(".lss-")],
            "candidates": sorted(p.split("/run/", 1)[1] for p in files if "/run/candidates/" in p),
            "selective": sorted(p.split("/run/", 1)[1] for p in files if "/run/selective-original/" in p),
            "runs": len(run_dirs)}


@pytest.mark.parametrize("where", ["decide", "publication"])
def test_a_job_that_fails_while_the_lane_works_leaves_no_new_file(tmp_path, slow_gpu, monkeypatch, where):
    """A failure in the GPU stage the lane overlaps (tagging, or the dialogue residual) stops the job. The lane is
    joined inside the cleanup scope; what it created is removed; no record or temporary file is left."""
    from lss.harness import infer

    def fail(*args, **kwargs):
        time.sleep(GPU_SECONDS)
        raise RuntimeError("injected GPU failure")

    def broken(overlapped):
        if where == "decide":
            monkeypatch.setattr(event_tagging, "tag_signal", fail)
        else:
            monkeypatch.setattr(infer, "msst_separate", fail)
        case = _run_case(tmp_path, "overlapped" if overlapped else "serial", monkeypatch, overlapped=overlapped,
                         seed=6 if overlapped else None)
        return case.outcome, case.leftovers

    serial, serial_left = broken(False)
    overlapped, overlapped_left = broken(True)

    assert serial.disposition is overlapped.disposition is application.ResultDisposition.FAILED
    assert overlapped_left["temporary"] == []
    assert overlapped_left["run_records"] == serial_left["run_records"]
    assert not [path for path in overlapped_left["run_records"] if path.endswith("run/run.json")]
    assert set(overlapped_left["candidates"]) <= set(serial_left["candidates"])
    assert set(overlapped_left["selective"]) <= set(serial_left["selective"])
    assert overlapped_left["public"] == serial_left["public"]
    assert not [thread for thread in threading.enumerate() if thread.name.startswith("lss-bgm-cpu")]


def test_a_user_quit_while_the_lane_works_leaves_no_new_file(tmp_path, slow_gpu, monkeypatch):
    calls = []
    real = event_tagging.tag_signal

    def quit_on_second_tagging(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            time.sleep(GPU_SECONDS)
            raise quitting.JobAbandonedError("user quit")
        return real(*args, **kwargs)

    monkeypatch.setattr(event_tagging, "tag_signal", quit_on_second_tagging)
    case = _run_case(tmp_path, "overlapped", monkeypatch, overlapped=True, seed=7)

    assert case.outcome.error_code == "job_abandoned", case.outcome.message
    left = case.leftovers
    assert left["candidates"] == [] and left["run_records"] == [] and left["temporary"] == []
    assert left["public"] == []
    assert not [thread for thread in threading.enumerate() if thread.name.startswith("lss-bgm-cpu")]


def test_the_shared_capacity_check_cannot_be_passed_twice(tmp_path, monkeypatch):
    """Two lane writers, room for one: the second waits until the first's write is done, never both at once."""
    from types import SimpleNamespace

    source = tmp_path / "signal.wav"
    source.write_bytes(b"")
    needed = writer.estimated_output_bytes(10.0, RATE, 2, 1, "PCM_24")
    disk = {"free": int(needed * 1.5)}
    monkeypatch.setattr(writer.shutil, "disk_usage", lambda path: SimpleNamespace(free=disk["free"]))
    inside, overlapping, order = [], [], []
    lock = threading.Lock()

    def write(name):
        writer.ensure_output_capacity(source, 10.0, RATE, 2, 1, "PCM_24", destination_dir=tmp_path)
        with lock:
            inside.append(name)
            overlapping.append(len(inside))
            order.append(name)
        time.sleep(0.2)
        with lock:
            disk["free"] -= needed // 2  # part of the file is on disk; the rest is still reserved
            inside.remove(name)
        return name

    lane = overlap.Lane(2)
    futures = [lane.submit(name, write, name) for name in ("a", "b")]
    lane.close(failed=False)
    assert sorted(future.result() for future in futures) == ["a", "b"]
    assert max(overlapping) == 1 and writer._RESERVED == {}

    # Not enough room even with nothing reserved: refused at once, exactly as serially.
    disk["free"] = needed - 1
    with pytest.raises(writer.InsufficientOutputSpaceError):
        with writer.output_reservations():
            writer.ensure_output_capacity(source, 10.0, RATE, 2, 1, "PCM_24", destination_dir=tmp_path)
    assert writer._RESERVED == {}


def test_without_a_lane_the_capacity_check_is_the_serial_one(tmp_path, monkeypatch):
    from types import SimpleNamespace

    source = tmp_path / "signal.wav"
    source.write_bytes(b"")
    needed = writer.estimated_output_bytes(10.0, RATE, 2, 1, "PCM_24")
    monkeypatch.setattr(writer.shutil, "disk_usage", lambda path: SimpleNamespace(free=needed))
    for _ in range(3):  # nothing is reserved outside a lane scope
        assert writer.ensure_output_capacity(source, 10.0, RATE, 2, 1, "PCM_24", destination_dir=tmp_path) == needed
    assert writer._RESERVED == {}


def test_the_kill_switch_and_worker_count(monkeypatch):
    monkeypatch.delenv(overlap.ENV, raising=False)
    monkeypatch.delenv(overlap.WORKERS_ENV, raising=False)
    assert overlap.enabled() and overlap.workers() == overlap.DEFAULT_WORKERS
    monkeypatch.setenv(overlap.ENV, "0")
    assert not overlap.enabled() and overlap.open_lane() is None
    monkeypatch.setenv(overlap.WORKERS_ENV, "99")
    assert overlap.workers() == overlap.MAXIMUM_WORKERS
    monkeypatch.setenv(overlap.WORKERS_ENV, "x")
    assert overlap.workers() == overlap.DEFAULT_WORKERS


def test_a_failed_lane_removes_only_what_its_tasks_created(tmp_path):
    kept = tmp_path / "kept.flac"
    kept.write_bytes(b"existing")
    started = threading.Event()
    lane = overlap.Lane(1)

    def make(name):
        started.set()
        path = tmp_path / name
        path.write_bytes(b"new")
        lane.created(path)
        time.sleep(0.1)

    lane.submit("one", make, "one.flac")
    lane.submit("two", make, "two.flac")  # never starts: cancelled
    assert started.wait(5)
    lane.close(failed=True)
    assert kept.read_bytes() == b"existing"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["kept.flac"]
    with pytest.raises(RuntimeError):
        lane.submit("late", make, "late.flac")


def _compare_module():
    import importlib.util

    path = dev_evidence_root(Path(__file__).resolve().parents[1]) / "efficiency-evidence" / "compare.py"
    spec = importlib.util.spec_from_file_location("efficiency_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_accept_case(case: Case, root: Path) -> None:
    """Write the ``result.json`` the local acceptance harness writes, for compare.py."""
    from mutagen.flac import FLAC

    rows = []
    for item in case.run["published"]:
        public = root / "media" / "LSS" / "episode - BGM" / Path(item["path"]).name
        tags = {key.upper(): value for key, value in FLAC(public).tags.as_dict().items()}
        rows.append({"name": public.name, "public_path": str(public), "cue_key": tags["LSS_BGM_CUE_KEY"][0],
                     "recorded_pcm_sha256": item["audio"]["pcm_sha256"], "start": item["start_frame"],
                     "end": item["end_frame"], "extent": item.get("cue_extent"), "tags": tags, "user_payload": None})
    (root / "result.json").write_text(json.dumps({
        "case": str(root), "first": {"receipt": {"disposition": case.outcome.disposition.value},
                                     "elapsed_seconds": case.seconds},
        "current_best_after_first": rows}), encoding="utf-8")


def test_compare_accepts_overlap_with_real_peak_stamps_and_fails_on_changed_audio(tmp_path, slow_gpu, monkeypatch):
    """compare.py (the local Windows verifier) on two real runs of this code, PEAK chunks left on: the only
    differences are consistently renamed hashes, so it passes; one changed public sample makes it fail."""
    compare = _compare_module()
    serial = _run_case(tmp_path, "serial", monkeypatch, overlapped=False)
    time.sleep(1.1)  # a different PEAK second
    case = _run_case(tmp_path, "overlapped", monkeypatch, overlapped=True, seed=8)
    for name, observed in (("serial", serial), ("overlapped", case)):
        layout = tmp_path / f"{name}-runs"
        layout.mkdir()
        (tmp_path / name).rename(layout / "case-ep1")
        _as_accept_case(observed, layout / "case-ep1")
    # The PEAK stamps really differ: the raw candidate ids and stems hashes are not equal byte for byte.
    assert case.run["analysis"]["stems"] != serial.run["analysis"]["stems"]

    report, summary = compare.Report(), {}
    compare.compare_case("1", tmp_path / "serial-runs", tmp_path / "overlapped-runs", report, summary)
    assert report.fails == [] and report.passes > 0
    assert any("consistent renaming" in note for note in report.notes)
    assert "overlap_candidate_cpu" in summary["ep1"]["stage_seconds"]

    public = sorted((tmp_path / "overlapped-runs" / "case-ep1" / "media" / "LSS" / "episode - BGM").glob("*.flac"))[0]
    audio, rate = sf.read(public, dtype="int32", always_2d=True)
    audio[rate, 0] += 256
    sf.write(public, audio, rate, subtype="PCM_24")
    report = compare.Report()
    compare.compare_case("1", tmp_path / "serial-runs", tmp_path / "overlapped-runs", report, {})
    assert any("PCM sha256" in line for line in report.fails)


def _stamped_wav(path: Path, samples: np.ndarray, stamp: int) -> str:
    """A FLOAT WAV whose PEAK chunk time stamp is set directly; returns its byte SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, RATE, subtype="FLOAT")
    data = bytearray(path.read_bytes())
    at = data.index(b"PEAK") + 8 + 4  # chunk id + size, then version
    data[at:at + 4] = stamp.to_bytes(4, "little")
    path.write_bytes(bytes(data))
    return hashlib.sha256(bytes(data)).hexdigest()


def _alias_documents(names_a: list[str], names_b: list[str], hashes_a: dict, hashes_b: dict):
    return [("receipt", {"stems": [hashes_a[n] for n in names_a]}, {"stems": [hashes_b[n] for n in names_b]})]


def test_wav_hashes_differing_only_in_the_peak_stamp_are_accepted_by_content(tmp_path):
    compare = _compare_module()
    tone = np.sin(np.linspace(0, 200, RATE, dtype=np.float32))[:, None].astype(np.float32)
    other = tone * 0.5
    a = {"x": _stamped_wav(tmp_path / "a" / "x.wav", tone, 1000), "y": _stamped_wav(tmp_path / "a" / "y.wav", tone, 2000),
         "z": _stamped_wav(tmp_path / "a" / "z.wav", other, 3000)}
    b = {"x": _stamped_wav(tmp_path / "b" / "x.wav", tone, 5000), "y": _stamped_wav(tmp_path / "b" / "y.wav", tone, 5000),
         "z": _stamped_wav(tmp_path / "b" / "z.wav", other, 5000)}
    assert a["x"] != a["y"] and b["x"] == b["y"]  # two stamps in A name one in B: no one-to-one renaming exists
    report = compare.Report()
    docs = _alias_documents(["x", "y", "z"], ["x", "y", "z"], a, b)
    tokens, _ = compare.aliases(docs, report, "t", (tmp_path / "a", tmp_path / "b"))
    assert report.fails == []
    assert compare.diff(compare.aliased(docs[0][1], tokens), compare.aliased(docs[0][2], tokens)) == []
    # Without the files the same documents keep the strict one-to-one rule and say why.
    report = compare.Report()
    compare.aliases(docs, report, "t")
    assert report.fails and "no WAV file" in report.fails[0]


def test_a_one_sample_change_in_an_aliased_wav_fails(tmp_path):
    compare = _compare_module()
    tone = np.sin(np.linspace(0, 200, RATE, dtype=np.float32))[:, None].astype(np.float32)
    changed = tone.copy()
    changed[100, 0] += 1e-3
    a = {"x": _stamped_wav(tmp_path / "a" / "x.wav", tone, 1000)}
    b = {"x": _stamped_wav(tmp_path / "b" / "x.wav", changed, 1000)}
    report = compare.Report()
    compare.aliases(_alias_documents(["x"], ["x"], a, b), report, "t", (tmp_path / "a", tmp_path / "b"))
    assert report.fails and "beyond the PEAK time stamp" in report.fails[0]


def test_retired_scratch_wav_hashes_need_the_same_file_name_on_both_sides(tmp_path):
    compare = _compare_module()
    h = lambda n: hashlib.sha256(str(n).encode()).hexdigest()  # noqa: E731

    def doc(name, digests):
        return {"stems": [{"path": f"C:/gone/{name}-{i}.wav", "sha256": d} for i, d in enumerate(digests)]}

    # Two stamps on one side name one on the other, and no file is left: accepted only through the same file names.
    a, b = doc("v", [h(1), h(2)]), doc("v", [h(3), h(3)])
    report = compare.Report()
    tokens, _ = compare.aliases([("run", a, b)], report, "t", (tmp_path / "a", tmp_path / "b"))
    assert report.fails == []
    assert compare.diff(compare.aliased(a, tokens), compare.aliased(b, tokens)) == []
    report = compare.Report()
    compare.aliases([("run", a, doc("w", [h(3), h(3)]))], report, "t", (tmp_path / "a", tmp_path / "b"))
    assert report.fails and "no WAV file" in report.fails[0]
