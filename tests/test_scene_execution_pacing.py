from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lss.bgm import scene_execution, scene_models


def _clock(cadence):
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    cadence.monotonic = lambda: now[0]
    cadence.sleeper = sleep
    return sleeps


def test_successful_asr_calls_are_paced_once_and_keep_pipeline_grid(monkeypatch, tmp_path):
    calls, cuda_calls, sleeps, held = [], [], [], [True]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="fp16", cuda=SimpleNamespace(
        synchronize=lambda: cuda_calls.append("sync"), empty_cache=lambda: cuda_calls.append("cache"))))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=lambda *a, **k: None))
    model = scene_models.SceneModels(execution_profile="scene_paced60")
    model._asr = lambda *a, **k: calls.append(k) or {"chunks": []}
    model._cadence.sleeper = lambda seconds: (sleeps.append(seconds), setattr(model._cadence, "_test_time", 0))
    times = [0.0]
    model._cadence.monotonic = lambda: times[0]
    model._cadence.sleeper = lambda seconds: (sleeps.append(seconds), times.__setitem__(0, times[0] + seconds))
    ffmpeg = tmp_path / "bin" / "ffmpeg.exe"
    ffmpeg.parent.mkdir()
    ffmpeg.write_bytes(b"fake")
    monkeypatch.setattr("lss.paths.resource_root", lambda: tmp_path)
    monkeypatch.setattr("lss.paths.is_frozen", lambda: True)
    monkeypatch.setattr(scene_models.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=np.zeros(1600, dtype=np.float32).tobytes()))
    model._lease = SimpleNamespace(close=lambda: held.__setitem__(0, False))
    model._gpu = lambda: None
    output = model.transcribe(Path("episode.mkv"), [(0., 1.), (5., 6.)])
    assert output == [] and len(calls) == 2
    assert all(k == {"chunk_length_s": 15, "batch_size": 8, "return_timestamps": True,
                     "generate_kwargs": {"language": "ja", "task": "transcribe"}} for k in calls)
    assert sleeps == [60.0, 60.0] and cuda_calls == ["sync", "cache"] * 2
    assert held[0]
    cadence_events = model.cadence_receipt()["events"]
    assert [e["rest_completed"] for e in cadence_events] == [True, True]
    assert all(e["rest_elapsed_seconds"] >= 60 and
               e["rest_completed_monotonic"] - e["rest_started_monotonic"] >= 60
               for e in cadence_events)
    assert model.last_transcription_receipt["execution_cadence"]["profile"] == "scene_paced60"
    model.release()
    assert not held[0]


def test_successful_llm_generate_paced_without_changing_generation_options(monkeypatch):
    events, times = [], [0.0]
    cuda = SimpleNamespace(synchronize=lambda: events.append("sync"), empty_cache=lambda: events.append("cache"),
                           OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
                           inference_mode=lambda: __import__("contextlib").nullcontext(),
                           ones_like=lambda x: x)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, ones_like=lambda x: x,
                                                               inference_mode=lambda: __import__("contextlib").nullcontext()))
    generated = []

    class Ids:
        shape = (1, 2)
        def to(self, device): return self

    class Generated:
        def __getitem__(self, key): return "tokens"

    model = scene_models.SceneModels(execution_profile="scene_paced60")
    model._llm = SimpleNamespace(device="cuda:0", generate=lambda *a, **kw:
                                 generated.append(kw) or Generated())
    model._tok = SimpleNamespace(apply_chat_template=lambda *a, **k: Ids(),
                                 decode=lambda *a, **k: "answer")
    model._load_llm = lambda: None
    model._cadence.monotonic = lambda: times[0]
    model._cadence.sleeper = lambda seconds: (events.append(seconds), times.__setitem__(0, times[0] + seconds))
    assert model._ask("prompt", 260) == "answer"
    assert len(generated) == 1
    assert {k: v for k, v in generated[0].items() if k != "attention_mask"} == {
        "max_new_tokens": 260, "do_sample": False, "temperature": None, "top_p": None,
        "top_k": None, "max_time": 90}
    assert events == ["sync", "cache", 60.0]
    [event] = model.cadence_receipt()["events"]
    assert event["operation"] == "llm_generate" and event["rest_seconds"] == 60
    assert event["rest_attempted"] and event["rest_completed"]
    assert event["rest_completed_monotonic"] - event["rest_started_monotonic"] >= 60


def test_failed_inference_does_not_rest_and_preserves_original_exception(monkeypatch):
    events = []
    cuda = SimpleNamespace(OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
                           empty_cache=lambda: events.append("cache"), synchronize=lambda: events.append("sync"))
    torch = SimpleNamespace(cuda=cuda, ones_like=lambda x: x,
                            inference_mode=lambda: __import__("contextlib").nullcontext())
    monkeypatch.setitem(sys.modules, "torch", torch)
    expected = RuntimeError("inference failed")
    model = scene_models.SceneModels(execution_profile="scene_paced60")
    class Ids:
        shape = (1, 2)
        def to(self, device): return self
    model._load_llm = lambda: None
    model._tok = SimpleNamespace(apply_chat_template=lambda *a, **k: Ids())
    model._llm = SimpleNamespace(device="cuda:0", generate=lambda *a, **k: (_ for _ in ()).throw(expected))
    with pytest.raises(RuntimeError) as caught:
        model._ask("prompt", 260)
    assert caught.value is expected and events == [] and model.cadence_receipt()["events"] == []


def test_interrupted_rest_retries_until_monotonic_deadline():
    cadence = scene_execution.SceneCadence("scene_paced60")
    now, requests = [0.0], []
    def sleep(seconds):
        requests.append(seconds)
        if len(requests) == 1:
            now[0] += 20
            raise InterruptedError
        now[0] += seconds
    cadence.monotonic = lambda: now[0]
    cadence.sleeper = sleep
    cadence.after_success("asr_pipeline", lambda: None, lambda: None)
    event = cadence.events[0]
    assert requests == [60, 40] and event["rest_completed"]
    assert event["rest_elapsed_seconds"] >= 60
    assert event["rest_completed_monotonic"] - event["rest_started_monotonic"] >= 60


def test_sync_failure_leaves_truthful_incomplete_event_without_rest_start():
    cadence = scene_execution.SceneCadence("scene_paced60")
    _clock(cadence)
    expected = RuntimeError("cuda sync failed")
    with pytest.raises(RuntimeError) as caught:
        cadence.after_success("llm_generate", lambda: (_ for _ in ()).throw(expected),
                              lambda: pytest.fail("cache release must follow sync"))
    event = cadence.events[0]
    assert caught.value is expected
    assert event["rest_attempted"] and not event["synchronize_completed"]
    assert not event["cache_release_completed"] and not event["rest_completed"]
    assert event["rest_started_monotonic"] is None and event["rest_elapsed_seconds"] == 0


def test_default_profile_has_no_event_and_profile_is_allowlisted():
    assert scene_models.SceneModels().cadence_receipt() == {"profile": "legacy", "rest_seconds": 0, "events": []}
    with pytest.raises(scene_execution.ScenePacingError):
        scene_execution.SceneCadence("anything")
