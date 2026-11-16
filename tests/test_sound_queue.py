"""欲しい音（SOUND）: キューの凍結・workerへの依頼・画面のセッション。"""
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.core import jobs
from lss.core.jobs import SeparationQueue
from lss.ui.web.session import WebSession


def _audio(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros((4410, 2), dtype="float32"), 44100, subtype="PCM_16")
    return path


class _Worker:
    def __init__(self, sound: dict | None = None):
        self.jobs: list[dict] = []
        self.sound = sound or {"k": "done", "n": 1, "files": ["a - 雨音.flac"],
                               "items": [{"text": "雨の音", "label": "雨音", "k": "done"}]}

    def __call__(self, job: dict) -> dict:
        self.jobs.append(dict(job))
        if job.get("cmd") == "wanted_sound":
            return {"status": "ok", "sound": self.sound}
        return {"status": "ok", "outputs": [job["mix_path"] + ".flac"]}


def test_sound_target_sends_the_frozen_words_to_the_worker(tmp_path: Path) -> None:
    worker = _Worker()
    queue = SeparationQueue(worker, state_path=tmp_path / "queue.json")
    (item_id,) = queue.add([_audio(tmp_path / "a.wav")])
    queue.freeze({item_id: ["inst", "snd"]}, sounds=["雨の音", "ピアノ"])
    queue.run()
    assert [job.get("cmd") for job in worker.jobs] == [None, "wanted_sound"]
    assert worker.jobs[1]["sounds"] == ["雨の音", "ピアノ"]
    item = queue.snapshot()[0]
    assert item.status == "completed"
    assert item.outcomes["snd"]["k"] == "done" and item.outcomes["snd"]["files"] == ["a - 雨音.flac"]
    # 再起動後も凍結した言葉と結果を持つ
    restored = SeparationQueue(worker, state_path=tmp_path / "queue.json").snapshot()[0]
    assert restored.sounds == ["雨の音", "ピアノ"] and restored.outcomes["snd"]["n"] == 1


def test_sound_without_words_cannot_be_frozen(tmp_path: Path) -> None:
    queue = SeparationQueue(_Worker(), state_path=tmp_path / "queue.json")
    (item_id,) = queue.add([_audio(tmp_path / "a.wav")])
    with pytest.raises(ValueError):
        queue.freeze({item_id: ["snd"]}, sounds=None)
    queue.freeze({item_id: ["inst"]}, sounds=["雨"])
    assert queue.snapshot()[0].sounds is None


def test_an_invalid_worker_answer_is_a_failure_not_a_result(tmp_path: Path) -> None:
    worker = _Worker(sound={"k": "done", "n": -1})
    queue = SeparationQueue(worker, state_path=tmp_path / "queue.json")
    (item_id,) = queue.add([_audio(tmp_path / "a.wav")])
    queue.freeze({item_id: ["snd"]}, sounds=["雨"])
    queue.run()
    item = queue.snapshot()[0]
    assert item.status == "failed" and item.outcomes["snd"]["error_code"] == "invalid_worker_response"


def test_normalized_sounds_split_and_bound_the_words() -> None:
    assert jobs.normalized_sounds("ピアノ、雨の音\n,ピアノ") == ["ピアノ", "雨の音"]
    assert jobs.normalized_sounds(["ドラム", "ベース"]) == ["ドラム", "ベース"]
    assert jobs.normalized_sounds("  ") is None and jobs.normalized_sounds(None) is None
    with pytest.raises(ValueError):
        jobs.normalized_sounds(3)


@pytest.fixture
def state(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


def _wait(predicate, timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("条件が成立しなかった")


def test_session_requires_words_remembers_them_and_reports_each_sound(state: Path) -> None:
    worker = _Worker(sound={"k": "done", "n": 1, "files": ["one - 雨音.flac"], "items": [
        {"text": "雨の音", "label": "雨音", "k": "done"},
        {"text": "かめはめ波", "label": "かめはめ波", "k": "unsupported", "reason": "unknown_words",
         "unknown": ["かめはめ波"]},
        {"text": "雨が窓に当たる音", "label": "雨が窓に当たる音", "k": "held", "reason": "heard_but_not_extracted",
         "heard_as": ["rain hitting the window"]}]})
    session = WebSession(dispatch=worker, state_path=state / "queue.json", lookup=lambda path: {})
    (item_id,) = session.add([_audio(state / "one.wav")])
    assert session.start({item_id: ["snd"]}, ["snd"], "") == {"started": False, "work": 0,
                                                              "reason": "sounds_required"}
    assert session.start({item_id: ["snd"]}, ["snd"], "雨の音、かめはめ波") == {"started": True, "work": 1}
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    snap = session.snapshot()
    assert snap["soundsText"] == "雨の音、かめはめ波"
    assert snap["soundsOf"] == {item_id: ["雨の音", "かめはめ波"]}
    outcome = snap["tst"][item_id]["snd"]
    assert outcome["k"] == "done" and outcome["files"] == ["one - 雨音.flac"]
    assert outcome["items"][1] == {"label": "かめはめ波", "text": "かめはめ波", "k": "unsupported",
                                   "reason": "unknown_words", "unknown": ["かめはめ波"], "heard_as": [],
                                   "suggestions": []}
    assert outcome["items"][2]["k"] == "held" and outcome["items"][2]["heard_as"] == ["rain hitting the window"]
    assert snap["songs"][item_id].get("out") is True
    # 次回の入力欄に同じ言葉を出す
    later = WebSession(dispatch=_Worker(), state_path=state / "queue2.json", lookup=lambda path: {})
    assert later.snapshot()["soundsText"] == "雨の音、かめはめ波"
    assert later.snapshot()["bdef"] == ["snd"]


@pytest.mark.parametrize("status", ["unchanged", "removed"])
def test_removal_disposition_reaches_ui_and_survives_session_reload(state: Path, status: str) -> None:
    worker = _Worker(sound={"k": "done", "n": 1, "files": ["one - 雨音なし.flac"], "items": [
        {"text": "雨なし", "label": "雨音なし", "k": "done", "removal_status": status,
         "decision": {"private_engine_detail": "not a user-facing decision"}}]})
    session = WebSession(dispatch=worker, state_path=state / "queue.json", lookup=lambda path: {})
    (item_id,) = session.add([_audio(state / "one.wav")])
    session.start({item_id: ["snd"]}, ["snd"], "雨なし")
    _wait(lambda: session.snapshot()["settle"]["kind"] == "drained")
    item = session.snapshot()["tst"][item_id]["snd"]["items"][0]
    assert item["removal_status"] == status and "decision" not in item
    restored = SeparationQueue(worker, state_path=state / "queue.json")
    assert restored.snapshot()[0].outcomes["snd"]["items"][0]["removal_status"] == status
