from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from mutagen.flac import FLAC

from lss.bgm import cue_publication, scene_job, scene_names
from lss.core import audio_identity

RATE = 44100
FACTS = {"title": "アニメ「星空冒険団」第{n}話", "work_title": "星空冒険団", "work_key": "work-test",
         "cast": ["ハル", "カイ", "ソラ"]}


@pytest.fixture(autouse=True)
def no_external_network_for_model_fixture_tests(monkeypatch):
    monkeypatch.setenv("LSS_WORK_INFO", "0")


def _cue(folder, source_sha, start, seconds=20, tone=220):
    folder.mkdir(parents=True, exist_ok=True)
    frames = RATE * seconds
    t = np.arange(frames) / RATE
    signal = np.column_stack((0.2 * np.sin(2 * np.pi * tone * t), 0.1 * np.sin(2 * np.pi * tone * 1.5 * t)))
    key = cue_publication.cue_key(source_sha, "recipe", int(start * RATE), int(start * RATE) + frames, RATE)
    path = folder / f"cue {int(start)}.flac"
    cue_publication.publish(signal, RATE, path, key=key, title=path.stem, album="ep BGM", input_sha256=source_sha,
                            recipe_id="recipe", version="0.1.0",
                            provenance={"type": "bgm_published_cue", "source": {"sha256": source_sha},
                                        "start_frame": int(start * RATE), "end_frame": int(start * RATE) + frames,
                                        "sample_rate": RATE, "start_seconds": float(start)})
    return {"path": path, "asset_id": f"bgm-cue:{key}", "pcm_sha256": audio_identity.audio_identity(path)["pcm_sha256"],
            "start": float(start), "end": float(start + seconds)}


def _write_current_best_for_recovery(state, source, source_sha, cues, *, run_key="candidate", requested=True):
    from pathlib import Path
    from lss.bgm import current_best

    if requested:   # the user's BGM job asked for this source's naming (start_in_background)
        scene_job._record_request(Path(state), Path(source), source_sha)

    run_path = Path(state) / "bgm-candidates" / run_key / "run" / "run.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    files = [{"key": cue["asset_id"].removeprefix("bgm-cue:"), "start_seconds": cue["start"],
              "end_seconds": cue["end"], "audio": {"pcm_sha256": cue["pcm_sha256"]}}
             for cue in cues]
    run = {"type": current_best.RUN_TYPE, "status": current_best.PUBLISHED_STATUS,
           "recipe_id": "recipe", "workflow": "episode-sfc-routed-cues-v13",
           "source": {"path": str(source), "sha256": source_sha}, "files": files}
    run_path.write_text(json.dumps(run), encoding="utf-8")
    rows = [current_best.file_entry(run, run_key, run_path, item, cue["path"], adoption="adopted")
            for cue, item in zip(cues, files)]
    current_best._write_json_atomic(current_best.record_path(Path(state), source_sha), {
        "type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
        "source_sha256": source_sha,
        "current": {"files": rows, "pinned": False, "evaluated_runs": [run_key]}, "history": []})


class FakeModels:
    def __init__(self, names):
        self.names, self.calls = list(names), []

    def transcribe(self, source, windows):
        self.calls.append(("transcribe", windows))
        return [{"start": b - 3, "end": b - 1, "text": "ミズウオだ！捕まえよう"} for _, b in windows] + \
               [{"start": 12.0, "end": 15.0, "text": "毎週金曜よる放送中"}]

    def order(self, prompt):
        self.calls.append(("order", prompt))
        return {"name_basis": "scene", "use": "カイがミズウオを捕まえる", "direction": "コミカルに", "category": "コミカル",
                "name": self.names.pop(0), "evidence": "ミズウオだ"}

    def prefix(self, title, description=""):
        self.calls.append(("prefix", title))
        return "LR"

    def revision(self):
        return {"id": "fake"}

    def release(self):
        self.calls.append(("release",))


@pytest.fixture
def library(tmp_path, monkeypatch):
    sources = {"s1": [], "s2": []}

    def published(state_root, sha):
        return sources[sha]

    def facts(source):
        n = 137 if "137" in str(source) else 150
        return {**FACTS, "title": FACTS["title"].format(n=n), "episode": n}
    monkeypatch.setattr(scene_job, "_published", published)
    from lss.bgm import current_best

    monkeypatch.setattr(current_best, "published_files", lambda state_root, sha: {
        row["asset_id"].removeprefix("bgm-cue:"): {"public_path": str(row["path"]), "pcm_sha256": row["pcm_sha256"]}
        for row in sources.get(sha, [])})
    monkeypatch.setattr(scene_names, "source_facts", lambda source, tags=None: facts(source))
    monkeypatch.setattr(scene_job, "sound_words", lambda path: "短調（暗め・切なめ）、強弱はなだらか")
    return sources


def test_a_source_is_named_once_and_its_files_show_the_names(tmp_path, library):
    state = tmp_path / "state"
    library["s1"] += [_cue(tmp_path / "ep137", "s1", 10), _cue(tmp_path / "ep137", "s1", 400, tone=330)]
    models = FakeModels(["ミズウオ大作戦"])
    result = scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models)
    assert result["named"] == 1 and result["unnamed"] == 1      # the 00:10 cue is an announcement
    first, second = (FLAC(c["path"]) for c in library["s1"])
    assert second["TITLE"] == ["ミズウオ大作戦（仮題）"] and second["ORDER_MENU"] == ["コミカルに"]
    scene_entry = scene_names.cue_record(state, library["s1"][1]["asset_id"])
    assert scene_entry["use"] is None and scene_entry["name"] == "ミズウオ大作戦"
    assert scene_entry["proposed"]["use"] == "カイがミズウオを捕まえる"
    assert scene_entry["guards"]["field_guard"]["use"]["tokens"] == ["カイ"]
    assert "カイ" not in "".join(line["text"] for line in scene_entry["heard"])
    assert second["M_NUMBER"][0].startswith("LR") and first["TITLE"] == ["00:10（仮題）"]
    assert ("release",) in models.calls
    # The same audio again: nothing is transcribed or asked.
    again = FakeModels([])
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=again)
    assert again.calls == []


@pytest.mark.parametrize("profile,expected_events", [(None, 0), ("scene_paced60", 1)])
def test_scene_job_environment_profile_reaches_result_and_per_cue_attempt(tmp_path, library, monkeypatch,
                                                                          profile, expected_events):
    from lss.bgm import scene_models

    state = tmp_path / "state"
    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)
    if profile is None:
        monkeypatch.delenv("LSS_SCENE_EXECUTION_PROFILE", raising=False)
        expected_profile = "legacy"
    else:
        monkeypatch.setenv("LSS_SCENE_EXECUTION_PROFILE", profile)
        expected_profile = profile
    constructed = []

    class JobModels(FakeModels):
        last_transcription_receipt = None

        def __init__(self, execution_profile):
            super().__init__(["ミズウオ大作戦"])
            constructed.append(execution_profile)
            self.execution_profile = execution_profile

        def transcribe(self, source, windows):
            self.last_transcription_receipt = {"windows": [{"window_id": 0,
                "source_start_seconds": windows[0][0], "source_end_seconds": windows[0][1]}]}
            return super().transcribe(source, windows)

        def cadence_receipt(self):
            events = ([{"operation": "llm_generate", "rest_seconds": 60, "rest_attempted": True,
                        "rest_completed": True, "rest_elapsed_seconds": 60.01}] if expected_events else [])
            return {"profile": self.execution_profile, "rest_seconds": 60 if expected_events else 0,
                    "events": events}

    monkeypatch.setattr(scene_models, "SceneModels", JobModels)
    result = scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1")
    entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    [attempt] = entry["attempts"]
    assert constructed == [expected_profile]
    assert result["execution_cadence"]["profile"] == expected_profile
    receipt = attempt["identity"]["transcription_receipt"]
    assert receipt["execution_cadence"]["profile"] == expected_profile
    assert len(receipt["execution_cadence"]["events"]) == expected_events


def test_scene_job_rejects_unallowlisted_environment_profile_before_model_construction(tmp_path, library,
                                                                                        monkeypatch):
    from lss.bgm import scene_execution, scene_models

    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)
    monkeypatch.setenv("LSS_SCENE_EXECUTION_PROFILE", "unlisted")
    monkeypatch.setattr(scene_models, "SceneModels", lambda **kwargs: pytest.fail("factory must not run"))
    with pytest.raises(scene_execution.ScenePacingError):
        scene_job.name_source(tmp_path / "state", tmp_path / "series-alpha-137.mkv", "s1")


class SceneUnknown(FakeModels):
    def order(self, prompt):
        self.calls.append(("order", prompt))
        return {"name_basis": "context_music", "name": "しずかな調べ", "direction": "しっとりと",
                "use": None, "category": None, "evidence": None}


def test_unknown_scene_gets_song_like_title_from_episode_and_music_context(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    library["s1"].append(_cue(tmp_path / "ep137", "s1", 400))
    pcm = library["s1"][0]["pcm_sha256"]
    monkeypatch.setattr(scene_job, "_music", lambda cue: {"tempo": {"status": "measured", "bpm": 120}})
    models = SceneUnknown([])
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models)
    tags = FLAC(library["s1"][0]["path"])
    assert tags["TITLE"] == ["しずかな調べ（仮題）"] and tags["BPM"] == ["120"]
    entry = scene_names.read(state)["cues"][library["s1"][0]["asset_id"]]
    assert entry["name_basis"] == "context_music" and entry["use"] is None
    assert entry["name_basis_evidence"]["scene_asserted"] is False
    assert entry["name_basis_evidence"]["source_context"]["episode"] == 137
    assert entry["name_basis_evidence"]["observed_transcript_sha256"]
    assert entry["name_basis_evidence"]["measured_music_sha256"]
    assert entry["name_basis_evidence"]["measured_music_description"] != "測定情報なし"
    assert len([c for c in models.calls if c[0] == "order"]) == 1
    prompt = next(c[1] for c in models.calls if c[0] == "order")
    assert "場面を断定しない歌らしい仮題" in prompt
    assert audio_identity.audio_identity(library["s1"][0]["path"])["pcm_sha256"] == pcm


class QueuedContextNames(FakeModels):
    def transcribe(self, source, windows):
        return []

    def order(self, prompt):
        self.calls.append(("order", prompt))
        return {"name_basis": "context_music", "name": self.names.pop(0), "direction": "しっとりと",
                "use": None, "category": None, "evidence": None}


def _orders(models):
    return [c[1] for c in models.calls if c[0] == "order"]


def test_a_name_another_recording_holds_is_asked_again_once(tmp_path, library):
    state = tmp_path / "state"
    library["s1"] += [_cue(tmp_path / "ep137", "s1", 100), _cue(tmp_path / "ep137", "s1", 400, tone=330)]
    models = QueuedContextNames(["広がる世界の調べ", "広がる世界の調べ", "夜霧の小径"])
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models)
    orders = _orders(models)
    assert len(orders) == 3
    assert "「広がる世界の調べ」はこの作品の別の曲の曲名として既に使われている" in orders[2]
    first, second = (FLAC(c["path"])["TITLE"] for c in library["s1"])
    assert first == ["広がる世界の調べ（仮題）"] and second == ["夜霧の小径（仮題）"]


def test_occurrences_of_one_recording_share_a_name_without_asking_again(tmp_path, library, monkeypatch):
    from lss.bgm import cue_memory

    state = tmp_path / "state"
    cues = [_cue(tmp_path / "ep137", "s1", 100), _cue(tmp_path / "ep137", "s1", 400)]
    library["s1"] += cues
    memory = {"labels": {"5": {"members": [c["asset_id"] for c in cues]}}}
    monkeypatch.setattr(cue_memory, "read", lambda state_root: memory)
    models = QueuedContextNames(["広がる世界の調べ", "広がる世界の調べ"])
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models)
    assert len(_orders(models)) == 2
    assert all(FLAC(c["path"])["TITLE"] == ["広がる世界の調べ（仮題）"] for c in cues)


def test_a_second_clash_keeps_the_numbered_fallback(tmp_path, library):
    state = tmp_path / "state"
    library["s1"] += [_cue(tmp_path / "ep137", "s1", 100), _cue(tmp_path / "ep137", "s1", 400, tone=330)]
    models = QueuedContextNames(["広がる世界の調べ"] * (2 + scene_job.DISTINCT_NAME_ASKS))
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models)
    assert len(_orders(models)) == 2 + scene_job.DISTINCT_NAME_ASKS
    titles = {FLAC(c["path"])["TITLE"][0] for c in library["s1"]}
    assert titles == {"広がる世界の調べ（仮題）", "広がる世界の調べ その2（仮題）"}


def test_a_version_decoration_is_not_a_distinct_name(tmp_path, library):
    state = tmp_path / "state"
    library["s1"] += [_cue(tmp_path / "ep137", "s1", 100), _cue(tmp_path / "ep137", "s1", 400, tone=330)]
    models = QueuedContextNames(["広がる光の調べ", "広がる光の調べ", "広がる光の調べ 〜改訂版〜", "朝もやの小径"])
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models)
    orders = _orders(models)
    assert "版・編・ver・その・番号などの付け足しで区別しない" in orders[2]
    assert "「広がる光の調べ」、「広がる光の調べ 〜改訂版〜」は" in orders[3]
    titles = {FLAC(c["path"])["TITLE"][0] for c in library["s1"]}
    assert titles == {"広がる光の調べ（仮題）", "朝もやの小径（仮題）"}


def test_no_dialogue_still_gets_a_context_music_title_without_scene_assertion(tmp_path, library):
    class NoDialogue(FakeModels):
        def transcribe(self, source, windows):
            return []

        def order(self, prompt):
            self.calls.append(("order", prompt))
            assert '"name_basis": "scene または context_music"' in prompt
            return {"name_basis": "context_music", "name": "第百三十七話の余韻", "direction": None,
                    "use": None, "category": None, "evidence": None}

    state = tmp_path / "state"
    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)
    result = scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=NoDialogue([]))
    assert result["named"] == 1 and FLAC(cue["path"])["TITLE"] == ["第百三十七話の余韻（仮題）"]
    entry = scene_names.read(state)["cues"][cue["asset_id"]]
    assert entry["name"] == "第百三十七話の余韻" and entry["use"] is None
    assert entry["name_basis"] == "context_music" and entry["heard"] == []
    assert entry["name_basis_evidence"]["scene_asserted"] is False
    assert audio_identity.audio_identity(cue["path"])["pcm_sha256"] == cue["pcm_sha256"]


def test_context_music_title_without_measured_music_does_not_claim_sound_features(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)
    monkeypatch.setattr(scene_job, "_music", lambda cue: {})

    class EpisodeContextOnly(FakeModels):
        def transcribe(self, source, windows):
            return []

        def order(self, prompt):
            self.calls.append(("order", prompt))
            assert "曲の音の特徴: 測定情報なし" in prompt
            assert "入力にないテンポ・楽器・リズム等は補わない" in prompt
            return {"name_basis": "context_music", "name": "はじまりの余韻", "direction": None,
                    "use": None, "category": None, "evidence": None}

    models = EpisodeContextOnly([])
    result = scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models)
    assert result["named"] == 1
    entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    assert entry["name"] == "はじまりの余韻" and entry["music"] == {}
    assert entry["name_basis_evidence"]["measured_music_description"] == "測定情報なし"
    assert entry["name_basis_evidence"]["measured_music_sha256"]
    assert entry["direction"] is None


def test_short_mid_episode_stinger_without_dialogue_can_receive_context_title(tmp_path, library):
    state = tmp_path / "state"
    cue = _cue(tmp_path / "ep137", "s1", 640, seconds=10)
    library["s1"].append(cue)

    class ShortStinger(FakeModels):
        def transcribe(self, source, windows):
            self.calls.append(("transcribe", windows))
            return []

        def order(self, prompt):
            self.calls.append(("order", prompt))
            return {"name_basis": "context_music", "name": "第百三十七話の余韻", "direction": None,
                    "use": None, "category": "神秘・不思議", "evidence": None}

    models = ShortStinger([])
    result = scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models)

    assert result["named"] == 1
    assert FLAC(cue["path"])["TITLE"] == ["第百三十七話の余韻（仮題）"]
    entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    assert entry["name_basis"] == "context_music" and entry["use"] is None and entry["heard"] == []
    assert entry["category"] == "神秘・不思議"
    assert entry["name_basis_evidence"]["scene_asserted"] is False


@pytest.mark.parametrize("source_changed", [False, True])
@pytest.mark.parametrize("failure_type", ["runtime", "decoder_exit", "decoder_timeout"])
def test_asr_unavailable_keeps_context_naming_but_never_claims_silence_or_uses_stale_receipts(tmp_path, library, source_changed, failure_type):
    import subprocess

    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)

    class ASRFailure(FakeModels):
        last_transcription_receipt = {"state": "stale_other_source", "windows": [{"window_id": 99}]}

        def transcribe(self, source, windows):
            raise {"runtime": RuntimeError("ASR unavailable"),
                   "decoder_exit": subprocess.CalledProcessError(1, ["ffmpeg"]),
                   "decoder_timeout": subprocess.TimeoutExpired(["ffmpeg"], 600)}[failure_type]

        def order(self, prompt):
            self.calls.append(("order", prompt))
            assert "文字起こしは取得できなかった" in prompt
            assert "台詞がないと確認できたわけではない" in prompt
            return {"name_basis": "context_music", "name": "はじまりの余韻", "direction": None,
                    "use": None, "category": None, "evidence": None}

    models = ASRFailure([])
    validations = []
    def validate():
        validations.append(True)
        if source_changed and len(validations) == 3:
            raise scene_job.SourceIdentityMismatch("source changed after unavailable ASR")
    state = tmp_path / "state"
    if source_changed:
        with pytest.raises(scene_job.SourceIdentityMismatch):
            scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models, validate_source=validate)
        assert scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"]) is None
        assert not any(call[0] == "order" for call in models.calls)
    else:
        result = scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models, validate_source=validate)
        assert result["named"] == 1
        entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
        assert entry["name"] == "はじまりの余韻" and entry["heard"] == [] and entry["use"] is None
        receipt = entry["attempt_identity"]["transcription_receipt"]
        expected_type = {"runtime": "RuntimeError", "decoder_exit": "CalledProcessError",
                         "decoder_timeout": "TimeoutExpired"}[failure_type]
        assert receipt == {"state": "ASR_UNAVAILABLE", "error_type": expected_type, "windows": [],
                           "scope": "transcription_failure_not_silence"}
        assert entry["attempt_identity"]["transcription_window_ids"] == []
        assert audio_identity.audio_identity(cue["path"])["pcm_sha256"] == cue["pcm_sha256"]
    assert len(validations) == 3
    assert ("release",) in models.calls


def test_a_guard_rejected_scene_retries_after_model_revision_then_fixes_first_acceptance(tmp_path, library):
    state = tmp_path / "state"
    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)

    class RevisionModels(FakeModels):
        def __init__(self, model_revision, order):
            super().__init__([])
            self.model_revision, self.order_result = model_revision, order

        def order(self, prompt):
            self.calls.append(("order", prompt))
            return self.order_result

        def revision(self):
            return {"asr": {"revision": "asr-fixed"}, "llm": {"revision": self.model_revision}}

    rejected = RevisionModels("llm-r1", {
        "use": "ウェーノベルの捕獲", "direction": "コミカルに", "category": "コミカル",
        "name": "ウェーノベル大作戦", "evidence": "ミズウオだ",
    })
    first = scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=rejected)
    rejected_entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    assert first["unnamed"] == 1 and rejected_entry["name"] is None
    assert rejected_entry["guards"]["unheard_names"] == ["ウェーノベル"]

    same_revision = RevisionModels("llm-r1", {
        "use": "ミズウオの捕獲", "direction": "コミカルに", "category": "コミカル",
        "name": "ミズウオ大作戦", "evidence": "ミズウオだ",
    })
    assert not scene_job.needs_naming(state, "s1", models=same_revision)
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=same_revision)
    assert same_revision.calls == []

    corrected = RevisionModels("llm-r2", {
        "use": "ミズウオの捕獲", "direction": "コミカルに", "category": "コミカル",
        "name": "ミズウオ大作戦", "evidence": "ミズウオだ",
    })
    second = scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=corrected)
    accepted = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    assert len([call for call in corrected.calls if call[0] == "order"]) == 1
    assert second["named"] == 1 and accepted["name"] == "ミズウオ大作戦"

    changed_again = RevisionModels("llm-r3", {
        "use": "別の場面", "direction": "重々しく", "category": "不穏・怪しい",
        "name": "別の曲名", "evidence": "ミズウオだ",
    })
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=changed_again)
    assert changed_again.calls == []
    assert scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])["name"] == "ミズウオ大作戦"

    attempts = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])["attempts"]
    assert [attempt["status"] for attempt in attempts] == ["guard_rejected", "accepted"]
    assert attempts[0]["proposed"]["name"] == "ウェーノベル大作戦"
    assert attempts[0]["heard"][0]["text"] == "ミズウオだ！捕まえよう"


def test_caught_order_errors_use_persisted_exponential_retry_backoff(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)
    now = [1000.0]
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: now[0])

    class ErrorModels(FakeModels):
        def revision(self):
            return {"asr": {"revision": "asr-fixed"}, "llm": {"revision": "llm-fixed"}}

        def order(self, prompt):
            self.calls.append(("order", prompt))
            now[0] += 301.0  # inference takes longer than the initial retry delay
            raise RuntimeError("temporary model failure")

    first = ErrorModels([])
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=first)
    entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    assert entry["attempt_status"] == "transient_error" and entry["retry_after"] == 1601.0
    assert entry["attempts"][-1]["attempted_at"] == 1000.0 and entry["attempts"][-1]["completed_at"] == 1301.0
    assert not scene_job.needs_naming(state, "s1", models=first)

    same_signature = FakeModels(["ミズウオ大作戦"])
    same_signature.revision = first.revision
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=same_signature)
    assert same_signature.calls == []

    now[0] = 1601.0
    assert scene_job.needs_naming(state, "s1", models=first)
    second = ErrorModels([])
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=second)
    entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    assert entry["attempt_status"] == "transient_error" and entry["retry_after"] == 2502.0
    assert entry["attempts"][-1]["attempted_at"] == 1601.0 and entry["attempts"][-1]["completed_at"] == 1902.0
    assert not scene_job.needs_naming(state, "s1", models=second)

    now[0] = 2502.0
    assert scene_job.needs_naming(state, "s1", models=second)


def test_source_facts_change_retries_same_pcm_when_source_filename_changes(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)
    facts_by_path = {}

    def facts(source, tags=None):
        return {**FACTS, **facts_by_path[str(source)]}

    monkeypatch.setattr(scene_names, "source_facts", facts)
    first_source = tmp_path / "series-alpha-137.mkv"
    second_source = tmp_path / "different-work-150.mkv"
    facts_by_path[str(first_source)] = {"title": "作品A 第137話", "work_title": "作品A", "work_key": "a",
                                        "episode": 137, "cast": ["ハル"], "site": None, "description": ""}
    facts_by_path[str(second_source)] = {"title": "作品B 第150話", "work_title": "作品B", "work_key": "b",
                                         "episode": 150, "cast": ["カイ"], "site": None, "description": ""}

    class Reject(FakeModels):
        def revision(self):
            return {"asr": {"revision": "same"}, "llm": {"revision": "same"}}

        def order(self, prompt):
            self.calls.append(("order", prompt))
            return {"name": None, "use": None, "direction": None, "category": None}

    first = Reject([])
    scene_job.name_source(state, first_source, "s1", models=first)
    assert scene_job.needs_naming(state, "s1", source=second_source, models=first)
    # The same source bytes can acquire different trusted metadata from its path; this is a material input.
    scene_job.name_source(state, second_source, "s1", models=first)
    entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    assert len(entry["attempts"]) == 2
    assert entry["attempts"][0]["identity"]["effective_facts_sha256"] != \
           entry["attempts"][1]["identity"]["effective_facts_sha256"]


def test_background_eligibility_uses_renamed_source_facts_for_same_media(tmp_path, library, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    state = tmp_path / "state"
    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)
    source_facts = {
        "series-alpha-137.mkv": {"title": "作品A 第137話", "work_title": "作品A", "work_key": "a", "episode": 137,
                             "cast": ["ハル"], "site": None, "description": ""},
        "renamed-work-150.mkv": {"title": "作品B 第150話", "work_title": "作品B", "work_key": "b", "episode": 150,
                                  "cast": ["カイ"], "site": None, "description": ""},
    }
    monkeypatch.setattr(scene_names, "source_facts", lambda source, tags=None: {**FACTS, **source_facts[source.name]})

    class Reject(FakeModels):
        def revision(self):
            return {"asr": {"revision": "same"}, "llm": {"revision": "same"}}

        def order(self, prompt):
            return {"name": None, "use": None, "direction": None, "category": None}

    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=Reject([]))
    monkeypatch.setattr(scene_job, "available", lambda: True)
    from lss import paths
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(pid=123))
    result = scene_job.start_in_background(state, tmp_path / "renamed-work-150.mkv", "s1")
    assert result["state"] == "started" and result["pid"] == 123
    assert len(calls) == 1


@pytest.mark.parametrize("cue_state, expected", [("no_published_cues", False), ("missing_entry", True),
                                                  ("accepted", False)])
def test_naming_eligibility_reads_source_facts_only_for_unresolved_existing_cues(
        tmp_path, library, monkeypatch, cue_state, expected):
    state = tmp_path / "state"
    source = tmp_path / "series-alpha-137.mkv"
    fact_reads = []
    monkeypatch.setattr(scene_names, "source_facts", lambda path, tags=None: fact_reads.append(path) or FACTS)
    if cue_state == "no_published_cues":
        library["s1"].clear()
    else:
        cue = _cue(tmp_path / "ep137", "s1", 400)
        library["s1"].append(cue)
        if cue_state == "accepted":
            scene_names.record_cue(
                state, cue["asset_id"], pcm_sha256=cue["pcm_sha256"], source_sha256="s1", facts=FACTS,
                start_seconds=400, end_seconds=420,
                heard=[{"start": 401, "end": 405, "text": "ミズウオだ！捕まえよう"}],
                order={"use": "ミズウオの捕獲", "direction": "コミカルに", "category": "コミカル",
                       "name": "ミズウオ大作戦", "evidence": "ミズウオだ"},
                model={"id": "accepted-test"})

    class RevisionModels(FakeModels):
        def revision(self):
            return {"asr": {"revision": "same"}, "llm": {"revision": "same"}}

    assert scene_job.needs_naming(state, "s1", source=source, models=RevisionModels([])) is expected
    assert fact_reads == []


def _source_backend(monkeypatch):
    """These tests drive the source-only backend with fake models; do not depend on requirements-scene.txt."""
    from lss.bgm import scene_runtime

    monkeypatch.setattr(scene_runtime, "backend", lambda: "transformers")


def test_detached_worker_rechecks_transient_order_failure_at_due_time(tmp_path, library, monkeypatch):
    _source_backend(monkeypatch)
    from lss.bgm import scene_models

    state = tmp_path / "state"
    source = tmp_path / "episode-137.mkv"
    source.write_bytes(b"unchanged source identity")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    cue = _cue(tmp_path / "ep137", source_sha, 400)
    library[source_sha] = [cue]
    now = [1000.0]
    waits, instances, nested = [], [], []
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: now[0])

    class RetryModels(FakeModels):
        def __init__(self, execution_profile="legacy"):
            super().__init__([])
            self.execution_profile = execution_profile
            self.order_count = 0
            self.released = False
            instances.append(self)

        def revision(self):
            return scene_job._current_model_revision(None)

        def order(self, prompt):
            self.order_count += 1
            if len(instances) == 1:
                raise RuntimeError("transient order failure")
            return {"use": "ミズウオの捕獲", "direction": "コミカルに", "category": "コミカル",
                    "name": "ミズウオ大作戦", "evidence": "ミズウオだ"}

        def release(self):
            self.released = True

    monkeypatch.setattr(scene_models, "SceneModels", RetryModels)

    def wait(seconds):
        assert instances[-1].released, "the order model must be released before due-time waiting"
        waits.append(seconds)
        nested.append(scene_job.run_background_worker(state, source, source_sha, wait=wait, clock=lambda: now[0]))
        now[0] += seconds

    result = scene_job.run_background_worker(state, source, source_sha, wait=wait, clock=lambda: now[0])
    entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    assert result["state"] == "complete" and result["attempts"] == 2
    assert waits == [300.0] and nested == [{"state": "already_running", "attempts": 0}]
    assert [attempt["status"] for attempt in entry["attempts"]] == ["transient_error", "accepted"]
    assert [model.released for model in instances] == [True, True]


def test_detached_worker_does_not_schedule_guard_rejected_cues(tmp_path, library, monkeypatch):
    _source_backend(monkeypatch)
    from lss.bgm import scene_models

    state = tmp_path / "state"
    source = tmp_path / "episode-137.mkv"
    source.write_bytes(b"unchanged source identity")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    cue = _cue(tmp_path / "ep137", source_sha, 400)
    library[source_sha] = [cue]
    waits, models = [], []

    class GuardedModels(FakeModels):
        def __init__(self, execution_profile="legacy"):
            super().__init__([])
            self.execution_profile = execution_profile
            models.append(self)

        def revision(self):
            return scene_job._current_model_revision(None)

        def order(self, prompt):
            return {"use": "ウェーノベルの捕獲", "direction": "コミカルに", "category": "コミカル",
                    "name": "ウェーノベル大作戦", "evidence": "ミズウオだ"}

    monkeypatch.setattr(scene_models, "SceneModels", GuardedModels)
    result = scene_job.run_background_worker(state, source, source_sha, wait=waits.append, clock=lambda: 1000.0)
    entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    assert result["state"] == "complete" and result["attempts"] == 1
    assert entry["attempt_status"] == "guard_rejected" and waits == []
    assert len(models) == 1 and models[0].calls[-1] == ("release",)


def test_detached_worker_caps_repeated_transient_attempts_and_persists_next_due_time(tmp_path, library, monkeypatch):
    _source_backend(monkeypatch)
    from lss.bgm import scene_models

    state = tmp_path / "state"
    source = tmp_path / "episode-137.mkv"
    source.write_bytes(b"unchanged source identity")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    cue = _cue(tmp_path / "ep137", source_sha, 400)
    library[source_sha] = [cue]
    now = [1000.0]
    waits, models = [], []
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: now[0])

    class ErrorModels(FakeModels):
        def __init__(self, execution_profile="legacy"):
            super().__init__([])
            self.execution_profile = execution_profile
            models.append(self)

        def revision(self):
            return scene_job._current_model_revision(None)

        def order(self, prompt):
            raise RuntimeError("transient order failure")

    monkeypatch.setattr(scene_models, "SceneModels", ErrorModels)

    def wait(seconds):
        waits.append(seconds)
        now[0] += seconds

    result = scene_job.run_background_worker(state, source, source_sha, wait=wait, clock=lambda: now[0])
    entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    assert result["state"] == "retry_pending" and result["attempts"] == 3
    assert waits == [300.0, 600.0] and now[0] == 1900.0
    assert entry["attempt_status"] == "transient_error" and entry["retry_after"] == 3100.0
    assert len(models) == 3 and all(model.calls[-1] == ("release",) for model in models)

    # A fresh detached invocation must not reset the per-signature budget or reload models.
    exhausted = scene_job.run_background_worker(state, source, source_sha, wait=wait, clock=lambda: 4000.0)
    assert exhausted == {"state": "retry_pending", "attempts": 0, "retry_after": 3100.0}
    assert len(models) == 3 and waits == [300.0, 600.0]


def test_background_trigger_restarts_a_waiter_for_a_future_transient_retry(tmp_path, library, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from lss import paths

    source = tmp_path / "episode-137.mkv"
    source.write_bytes(b"same source after app exit")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    cue = _cue(tmp_path / "ep137", source_sha, 400)
    library[source_sha] = [cue]
    now = [1000.0]
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: now[0])
    monkeypatch.setattr(scene_job, "_current_model_revision", lambda models=None: {"id": "fake"})

    class ErrorModels(FakeModels):
        def order(self, prompt):
            raise RuntimeError("transient order failure")

    scene_job.name_source(tmp_path / "state", source, source_sha, models=ErrorModels([]))
    entry = scene_names.cue_record(tmp_path / "state", cue["asset_id"], cue["pcm_sha256"])
    assert not scene_job.needs_naming(tmp_path / "state", source_sha, source=source, now=now[0])
    assert entry["retry_after"] == 1300.0

    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    monkeypatch.setattr(scene_job, "available", lambda: True)
    launches = []

    def launch(command, **kwargs):
        launches.append(command)
        return SimpleNamespace(pid=321)

    monkeypatch.setattr(subprocess, "Popen", launch)
    started = scene_job.start_in_background(tmp_path / "state", source, source_sha)
    assert started["state"] == "started" and started["pid"] == 321
    assert launches and "lss.bgm.scene_job" in launches[0]


def test_stale_transient_retry_is_ignored_after_current_best_pcm_replacement(tmp_path, library, monkeypatch):
    source = tmp_path / "episode-137.mkv"
    source.write_bytes(b"published source")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    old = _cue(tmp_path / "ep137", source_sha, 400, tone=220)
    library[source_sha] = [old]
    monkeypatch.setattr(scene_job, "_current_model_revision", lambda models=None: {"id": "fake"})
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: 1000.0)

    class ErrorModels(FakeModels):
        def order(self, prompt):
            raise RuntimeError("transient order failure")

    state = tmp_path / "state"
    scene_job.name_source(state, source, source_sha, models=ErrorModels([]))
    assert scene_job._next_transient_retry_after(state, source_sha) == 1300.0

    replacement = _cue(tmp_path / "replacement", source_sha, 400, tone=330)
    assert replacement["asset_id"] == old["asset_id"]
    assert replacement["pcm_sha256"] != old["pcm_sha256"]
    library[source_sha] = [replacement]
    assert scene_names.cue_record(state, replacement["asset_id"], replacement["pcm_sha256"]) is None
    assert scene_job._next_transient_retry_after(state, source_sha) is None


def test_background_worker_lease_is_exclusive_and_released_on_windows(tmp_path):
    source_sha = "a" * 64
    with scene_job._background_worker_lease(tmp_path, source_sha) as acquired:
        assert acquired
        with scene_job._background_worker_lease(tmp_path, source_sha) as duplicate:
            assert not duplicate
    with scene_job._background_worker_lease(tmp_path, source_sha) as reacquired:
        assert reacquired


@pytest.mark.parametrize("frozen", [False, True])
def test_startup_recovery_launches_one_hidden_coordinator(tmp_path, monkeypatch, frozen):
    import subprocess
    import sys
    from types import SimpleNamespace
    from lss import paths
    from lss.bgm import current_best

    state = tmp_path / "state"
    folder = state / current_best.DIRECTORY
    folder.mkdir(parents=True)
    (folder / ("a" * 64 + ".json")).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(scene_job, "available", lambda: True)
    monkeypatch.setattr(paths, "is_frozen", lambda: frozen)
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs:
                        calls.append((command, kwargs)) or SimpleNamespace(pid=992))

    result = scene_job.start_recovery_in_background(state)

    assert result["state"] == "started" and result["pid"] == 992 and len(calls) == 1
    command, kwargs = calls[0]
    if frozen:
        assert command == [sys.executable, "--scene-naming-recovery", str(state.resolve())]
    else:
        assert command == [sys.executable, "-m", "lss.bgm.scene_job", "--recover", str(state.resolve())]
    assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["stdout"].name.endswith("startup-recovery.log")
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with scene_job._background_worker_lease(state, None):
        assert scene_job.start_recovery_in_background(state) == {"state": "already_running"}


def test_startup_recovery_does_not_hash_or_download_when_models_are_not_cached(tmp_path, library, monkeypatch):
    state, source = tmp_path / "state", tmp_path / "episode-137.mkv"
    source.write_bytes(b"finished episode source")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    cue = _cue(tmp_path / "public", source_sha, 400)
    library[source_sha] = [cue]
    _write_current_best_for_recovery(state, source, source_sha, [cue])
    monkeypatch.setattr(scene_job, "_scene_models_cached", lambda: False)
    monkeypatch.setattr(scene_job, "run_background_worker", lambda *args, **kwargs:
                        pytest.fail("uncached startup recovery must not load or fetch models"))
    monkeypatch.setattr(scene_job, "_sha256_file", lambda *a: pytest.fail("uncached startup recovery must not hash"))

    result = scene_job.run_current_best_recovery(state)

    assert result["started"] == 0 and result["skipped"] == 1
    assert scene_job._recovery_log_path(state).is_file()


def test_startup_recovery_retries_a_budgeted_cue_that_becomes_due_during_another_source(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    now = [1000.0]
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: now[0])
    monkeypatch.setattr(scene_job, "_current_model_revision", lambda models=None: {"id": "fake"})
    monkeypatch.setattr(scene_job, "_scene_models_cached", lambda: True)

    def new_source(label):
        source = tmp_path / f"{label}.mkv"
        source.write_bytes(label.encode())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        cue = _cue(tmp_path / f"public-{label}", digest, 400)
        library[digest] = [cue]
        _write_current_best_for_recovery(state, source, digest, [cue], run_key=label)
        return source, digest, cue

    one, two = new_source("one"), new_source("two")
    # The coordinator's deterministic digest order must place the transient source before the slow source.
    first, second = sorted((one, two), key=lambda item: item[1])
    facts_a = scene_names.source_facts(first[0])
    retry_key_a = scene_job._retry_key(state, first[1], first[2], {"id": "fake"}, facts_a)
    calls = []

    def record(cue_row, source_path, digest, status, when):
        facts = scene_names.source_facts(source_path)
        order = ({"use": None, "direction": None, "category": None, "name": None, "error": "RuntimeError"}
                 if status == "transient_error" else
                 {"use": "穏やかな会話", "direction": "しっとりと", "category": "日常・のどか",
                  "name": "おだやかな時間", "evidence": "これはいい"})
        scene_names.record_cue(state, cue_row["asset_id"], pcm_sha256=cue_row["pcm_sha256"],
                               source_sha256=digest, facts=facts, start_seconds=cue_row["start"],
                               end_seconds=cue_row["end"],
                               heard=[{"start": 400, "end": 401, "text": "これはいい"}], order=order,
                               model={"id": "fake"},
                               attempt_identity={"retry_key": retry_key_a} if status == "transient_error" else None,
                               attempted_at=when, completed_at=when)

    def fake_worker(root, path, digest, **kwargs):
        calls.append(digest)
        if digest == first[1]:
            record(first[2], path, digest, "transient_error" if calls.count(digest) == 1 else "accepted", now[0])
        else:
            now[0] += 301.0
            record(second[2], path, digest, "accepted", now[0])
        return {"state": "complete", "attempts": 1}

    monkeypatch.setattr(scene_job, "run_background_worker", fake_worker)
    result = scene_job.run_current_best_recovery(state, clock=lambda: now[0])

    assert calls == [first[1], second[1], first[1]]
    assert result["started"] == 3


def test_startup_recovery_skips_a_changed_recorded_path_for_another_current_path(tmp_path, library, monkeypatch):
    from pathlib import Path
    from lss.bgm import current_best

    state = tmp_path / "state"
    old_source, valid_source = tmp_path / "old.mkv", tmp_path / "valid.mkv"
    old_source.write_bytes(b"recorded source bytes")
    valid_source.write_bytes(b"recorded source bytes")
    digest = hashlib.sha256(valid_source.read_bytes()).hexdigest()
    cue_a = _cue(tmp_path / "public-old", digest, 400)
    cue_b = _cue(tmp_path / "public-valid", digest, 800, tone=330)
    library[digest] = [cue_a, cue_b]

    rows = []
    for run_key, source, cue in (("old", old_source, cue_a), ("valid", valid_source, cue_b)):
        run_path = state / "bgm-candidates" / run_key / "run" / "run.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        item = {"key": cue["asset_id"].removeprefix("bgm-cue:"), "start_seconds": cue["start"],
                "end_seconds": cue["end"], "audio": {"pcm_sha256": cue["pcm_sha256"]}}
        run = {"type": current_best.RUN_TYPE, "status": current_best.PUBLISHED_STATUS,
               "recipe_id": "recipe", "workflow": "episode-sfc-routed-cues-v13",
               "source": {"path": str(source), "sha256": digest}, "files": [item]}
        run_path.write_text(json.dumps(run), encoding="utf-8")
        rows.append(current_best.file_entry(run, run_key, run_path, item, cue["path"], adoption="adopted"))
    old_source.write_bytes(b"replaced bytes at old recorded path")
    scene_job._record_request(state, old_source, digest)   # the job named the old place
    current_best._write_json_atomic(current_best.record_path(state, digest), {
        "type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
        "source_sha256": digest,
        "current": {"files": rows, "pinned": False, "evaluated_runs": ["old", "valid"]}, "history": []})
    monkeypatch.setattr(scene_job, "_scene_models_cached", lambda: True)
    calls = []

    def worker(root, source, source_sha, **kwargs):
        calls.append(Path(source).resolve())
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == source_sha
        return {"state": "complete", "attempts": 1}

    monkeypatch.setattr(scene_job, "run_background_worker", worker)
    result = scene_job.run_current_best_recovery(state)

    assert calls == [valid_source.resolve()]
    assert result["started"] == 1
    log = state / scene_names.DIRECTORY / "jobs" / "startup-recovery.log"
    assert any(event.get("state") == "recorded_path_rejected" and
               event["paths"][0]["reason"] == "recorded_source_identity_mismatch"
               for event in [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()])


def test_startup_recovery_restarts_only_a_current_published_cue(tmp_path, library, monkeypatch):
    from lss.bgm import current_best

    state, source = tmp_path / "state", tmp_path / "episode-137.mkv"
    source.write_bytes(b"finished episode source")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    cue = _cue(tmp_path / "public", source_sha, 400)
    library[source_sha] = [cue]
    run_path = state / "bgm-candidates" / "candidate" / "run" / "run.json"
    run_path.parent.mkdir(parents=True)
    run = {"type": current_best.RUN_TYPE, "status": current_best.PUBLISHED_STATUS,
           "recipe_id": "recipe", "workflow": "episode-sfc-routed-cues-v13",
           "source": {"path": str(source), "sha256": source_sha}, "files": [{
               "key": cue["asset_id"].removeprefix("bgm-cue:"), "start_seconds": cue["start"],
               "end_seconds": cue["end"], "audio": {"pcm_sha256": cue["pcm_sha256"]}}]}
    run_path.write_text(json.dumps(run), encoding="utf-8")
    files = [current_best.file_entry(run, "candidate", run_path, run["files"][0], cue["path"],
                                     adoption="adopted")]
    current_best._write_json_atomic(current_best.record_path(state, source_sha), {
        "type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
        "source_sha256": source_sha,
        "current": {"files": files, "pinned": False, "evaluated_runs": ["candidate"]}, "history": []})

    launches = []
    scene_job._record_request(state, source, source_sha)
    monkeypatch.setattr(scene_job, "_scene_models_cached", lambda: True)
    monkeypatch.setattr(scene_job, "run_background_worker",
                        lambda root, path, digest, **kwargs: launches.append((root, path, digest, kwargs)) or
                        {"state": "complete", "attempts": 1})
    result = scene_job.run_current_best_recovery(state)

    assert result["started"] == 1 and result["skipped"] == 0
    assert launches == [(state.resolve(), source.resolve(), source_sha,
                         {"wait": None, "clock": scene_job._retry_clock, "wait_for_retries": False, "require_request": True})]


def test_startup_recovery_checks_due_sources_before_future_waits_and_skips_ineligible(tmp_path, library, monkeypatch):
    from pathlib import Path
    from lss.bgm import current_best

    state = tmp_path / "state"
    now = [1000.0]
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: now[0])
    monkeypatch.setattr(scene_job, "_current_model_revision", lambda models=None: {"id": "fake"})
    monkeypatch.setattr(scene_job, "_scene_models_cached", lambda: True)

    def make_pending(name, *, tone=220, missing=False):
        source = tmp_path / f"{name}.mkv"
        if not missing:
            source.write_bytes(f"source {name}".encode())
        digest = hashlib.sha256(source.read_bytes()).hexdigest() if not missing else hashlib.sha256(name.encode()).hexdigest()
        cue = _cue(tmp_path / f"public-{name}", digest, 400, tone=tone)
        library[digest] = [cue]
        _write_current_best_for_recovery(state, source, digest, [cue], run_key=name)
        return source, digest, cue

    immediate_source, immediate_sha, immediate_cue = make_pending("immediate")
    future_source, future_sha, future_cue = make_pending("future")
    accepted_source, accepted_sha, accepted_cue = make_pending("accepted")
    guard_source, guard_sha, guard_cue = make_pending("guard")
    exhausted_source, exhausted_sha, exhausted_cue = make_pending("exhausted")
    changed_source, changed_sha, changed_cue = make_pending("changed")
    missing_source, missing_sha, missing_cue = make_pending("missing", missing=True)

    facts = scene_names.source_facts(future_source)
    future_key = scene_job._retry_key(state, future_sha, future_cue, {"id": "fake"}, facts)
    scene_names.record_cue(state, future_cue["asset_id"], pcm_sha256=future_cue["pcm_sha256"],
                           source_sha256=future_sha, facts=facts, start_seconds=future_cue["start"],
                           end_seconds=future_cue["end"], heard=[{"start": 400, "end": 401, "text": "ミズウオだ"}],
                           order={"use": None, "direction": None, "category": None, "name": None,
                                  "error": "RuntimeError"}, model={"id": "fake"},
                           attempt_identity={"retry_key": future_key}, attempted_at=900, completed_at=900)

    accepted_facts = scene_names.source_facts(accepted_source)
    scene_names.record_cue(state, accepted_cue["asset_id"], pcm_sha256=accepted_cue["pcm_sha256"],
                           source_sha256=accepted_sha, facts=accepted_facts,
                           start_seconds=accepted_cue["start"], end_seconds=accepted_cue["end"],
                           heard=[{"start": 400, "end": 401, "text": "これはいい"}],
                           order={"use": "穏やかな会話", "direction": "しっとりと",
                                  "category": "日常・のどか", "name": "おだやかな時間", "evidence": "これはいい"},
                           model={"id": "fake"})

    guard_facts = scene_names.source_facts(guard_source)
    guard_key = scene_job._retry_key(state, guard_sha, guard_cue, {"id": "fake"}, guard_facts)
    scene_names.record_cue(state, guard_cue["asset_id"], pcm_sha256=guard_cue["pcm_sha256"],
                           source_sha256=guard_sha, facts=guard_facts,
                           start_seconds=guard_cue["start"], end_seconds=guard_cue["end"],
                           heard=[{"start": 400, "end": 401, "text": "ミズウオだ"}],
                           order={"use": "ウェーノベルの捕獲", "direction": "コミカルに",
                                  "category": "コミカル", "name": "ウェーノベル大作戦", "evidence": "ミズウオだ"},
                           model={"id": "fake"}, attempt_identity={"retry_key": guard_key})
    assert scene_names.cue_record(state, guard_cue["asset_id"])["attempt_status"] == "guard_rejected"

    exhausted_facts = scene_names.source_facts(exhausted_source)
    exhausted_key = scene_job._retry_key(state, exhausted_sha, exhausted_cue, {"id": "fake"}, exhausted_facts)
    for offset in range(3):
        scene_names.record_cue(state, exhausted_cue["asset_id"], pcm_sha256=exhausted_cue["pcm_sha256"],
                               source_sha256=exhausted_sha, facts=exhausted_facts,
                               start_seconds=exhausted_cue["start"], end_seconds=exhausted_cue["end"],
                               heard=[{"start": 400, "end": 401, "text": "ミズウオだ"}],
                               order={"use": None, "direction": None, "category": None, "name": None,
                                      "error": "RuntimeError"}, model={"id": "fake"},
                               attempt_identity={"retry_key": exhausted_key},
                               attempted_at=900 + offset, completed_at=900 + offset)
    changed_source.write_bytes(b"different source at the recorded location")

    hashed, started = [], []

    def checked_worker(root, path, digest, **kwargs):
        hashed.append(Path(path).resolve())
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != digest:
            raise scene_job.SourceIdentityMismatch("source_identity_mismatch")
        started.append((digest, kwargs))
        return {"state": "retry_pending", "attempts": 0, "retry_after": now[0] + 300}

    monkeypatch.setattr(scene_job, "run_background_worker", checked_worker)

    result = scene_job.run_current_best_recovery(state, wait=lambda _: None, clock=lambda: now[0])

    # The due cue must get its attempt before any wait for the future cue. The changed/missing,
    # accepted, guard-rejected and exhausted sources are never passed to the worker.
    assert [digest for digest, _ in started] == [immediate_sha, future_sha]
    assert started[0][1]["wait_for_retries"] is False
    assert future_sha == started[1][0]
    assert immediate_source.resolve() in hashed and future_source.resolve() in hashed
    assert changed_source.resolve() not in hashed and accepted_source.resolve() not in hashed
    assert guard_source.resolve() not in hashed and exhausted_source.resolve() not in hashed
    assert missing_source.resolve() not in hashed
    # The changed source path is rejected before a worker starts, so only the immediate and
    # future eligible source attempts count as started workers.
    assert result["started"] == 2 and result["skipped"] >= 4
    log = state / scene_names.DIRECTORY / "jobs" / "startup-recovery.log"
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(event.get("source_sha256") == changed_sha and event.get("paths", [{}])[0].get("reason") ==
               "recorded_source_identity_mismatch" for event in events)


def test_a_changed_published_spelling_input_retries_a_guard_rejected_scene(tmp_path, library, monkeypatch):
    from lss.bgm import work_info

    state = tmp_path / "state"
    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)
    context = {"source_kind": "official_episode", "url": "https://example.org/episode/137",
               "text": "第137話", "text_sha256": hashlib.sha256("第137話".encode()).hexdigest(),
               "page_text_sha256": "a" * 64}
    monkeypatch.setattr(scene_job, "_work_info_enabled", lambda: True)
    monkeypatch.setattr(work_info, "naming_context", lambda *args, **kwargs: dict(context))

    class RevisionModels(FakeModels):
        def __init__(self, order):
            super().__init__([])
            self.order_result = order

        def order(self, prompt):
            self.calls.append(("order", prompt))
            return self.order_result

        def revision(self):
            return {"asr": {"revision": "asr-fixed"}, "llm": {"revision": "llm-fixed"}}

    rejected = RevisionModels({"use": "ウェーノベルの捕獲", "direction": "コミカルに", "category": "コミカル",
                               "name": "ウェーノベル大作戦", "evidence": "ミズウオだ"})
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=rejected)
    assert not scene_job.needs_naming(state, "s1", models=rejected)

    cache = state / "bgm-work-info" / "pages"
    cache.mkdir(parents=True)
    unrelated = cache / "unrelated.json"
    unrelated.write_text(json.dumps({"url": "https://other.example/page", "text_sha256": "unrelated"}), encoding="utf-8")
    assert not scene_job.needs_naming(state, "s1", models=rejected)

    from urllib.parse import quote
    context["url"] = "https://ja.wikipedia.org/wiki/" + quote(FACTS["work_title"], safe="")
    page_text = "第137話の追加公開表記"
    cache_path = cache / (hashlib.sha256(context["url"].encode()).hexdigest()[:24] + ".json")
    cache_path.write_text(json.dumps({"url": context["url"], "text": page_text,
        "text_sha256": hashlib.sha256(page_text.encode()).hexdigest(), "first_h1": "第137話", "links": []}),
        encoding="utf-8")
    context.update(text=page_text, text_sha256=hashlib.sha256(page_text.encode()).hexdigest(),
                   page_text_sha256=hashlib.sha256(page_text.encode()).hexdigest())

    corrected = RevisionModels({"use": "ミズウオの捕獲", "direction": "コミカルに", "category": "コミカル",
                                "name": "ミズウオ大作戦", "evidence": "ミズウオだ"})
    assert scene_job.needs_naming(state, "s1", models=corrected)
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=corrected)
    entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    assert len([call for call in corrected.calls if call[0] == "order"]) == 1
    assert entry["name"] == "ミズウオ大作戦"
    assert [attempt["status"] for attempt in entry["attempts"]] == ["guard_rejected", "accepted"]


def test_scene_attempt_persists_only_its_transcription_input_receipt_windows(tmp_path, library):
    state = tmp_path / "state"
    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)

    class ReceiptModels(FakeModels):
        def transcribe(self, source, windows):
            heard = super().transcribe(source, windows)
            self.last_transcription_receipt = {
                "schema": "scene-asr-input-lineage-r1", "source_path": str(source),
                "ffmpeg": {"path": "bundled/ffmpeg.exe", "binary_sha256": "f" * 64,
                           "version": "ffmpeg version fixture"},
                "asr": {"revision": {"repository": "pinned/asr", "revision": "rev1"}},
                "windows": [
                    {"window_id": 4, "source_start_seconds": windows[0][0],
                     "source_end_seconds": windows[0][1], "input": {"sha256": "a" * 64}},
                    {"window_id": 5, "source_start_seconds": 800.0,
                     "source_end_seconds": 820.0, "input": {"sha256": "b" * 64}},
                ],
                "transcript_sha256": "c" * 64,
            }
            return heard

    models = ReceiptModels(["ミズウオ大作戦"])
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models)
    entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    identity = entry["attempts"][-1]["identity"]

    assert identity["transcription_window_ids"] == [4]
    assert identity["transcription_receipt"]["schema"] == "scene-asr-input-lineage-r1"
    assert identity["transcription_receipt"]["transcript_sha256"] == "c" * 64
    assert identity["transcription_receipt"]["windows"] == [models.last_transcription_receipt["windows"][0]]
    assert identity["source_sha256"] == "s1"
    assert entry["heard"] == [{"start": 417.0, "end": 419.0, "text": "ミズウオだ！捕まえよう"}]


@pytest.mark.parametrize("provider", ["official", "wiki"])
def test_normal_naming_acquires_spelling_without_turning_web_words_into_cast(tmp_path, library, monkeypatch, provider):
    from lss.bgm import work_info

    monkeypatch.setenv("LSS_WORK_INFO", "1")
    facts = {**FACTS, "title": "作品 第150話", "work_title": "作品", "episode": 150, "cast": [],
             "description": "公式サイト\nhttps://example.org/anime/" if provider == "official" else ""}
    monkeypatch.setattr(scene_names, "source_facts", lambda source, tags=None: facts)
    pages = {"https://example.org/robots.txt": "User-agent: *\nAllow: /",
             "https://example.org/anime/": '<a href="/anime/episodes/">episodes</a>',
             "https://example.org/anime/episodes/": '<h2>第150話</h2><p>メガソックウサとウェーノベル。</p>',
             "https://ja.wikipedia.org/robots.txt": "User-agent: *\nAllow: /",
             "https://ja.wikipedia.org/wiki/%E4%BD%9C%E5%93%81": '<h1>作品</h1><p>メガソックウサとウェーノベル。</p>'}
    requests = []

    def http(url):
        requests.append(url)
        if url not in pages:
            raise work_info.WorkInfoUnavailable(url)
        return pages[url]

    monkeypatch.setattr(work_info, "_http", http)

    class ObservedModels(FakeModels):
        def transcribe(self, source, windows):
            assert requests, "LSS must acquire the context itself before inference"
            return [{"start": 401., "end": 405., "text": "メガソッグーザの導きだ"}]

        def order(self, prompt):
            assert "公開ページの表記候補" in prompt
            assert "メガソックウサ" in prompt
            assert "一覧（正しい表記。聞き違いはこの表記に直してよい）: \n" in prompt
            return {"name": "導かれる扉の先", "use": "メガソックウサの導き", "evidence": "メガソッグーザの導きだ",
                    "direction": "神秘的に", "category": "神秘・不思議"}

    cue = _cue(tmp_path / "episode", "s1", 400)
    library["s1"].append(cue)
    result = scene_job.name_source(tmp_path / "state", tmp_path / "episode.mkv", "s1", models=ObservedModels([]))
    entry = scene_names.cue_record(tmp_path / "state", cue["asset_id"])
    assert result["named"] == 1 and entry["name"] == "導かれる扉の先"
    assert entry["heard"][0]["text"] == "メガソッグーザの導きだ"
    assert entry["published_spelling_support"][0]["canonical"] == "メガソックウサ"
    assert entry["published_name_sources"][0]["source_kind"] == ("official_episode" if provider == "official" else "wikipedia_work")
    assert not entry["guards"] and facts["cast"] == []
    assert audio_identity.audio_identity(cue["path"])["pcm_sha256"] == cue["pcm_sha256"]


def test_legacy_fixed_music_name_survives_recurrence_without_new_scene_inference(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    a = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(a)
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=FakeModels(["ミズウオ大作戦"]))
    def legacy(data):
        fixed = data["recordings"]["asset:" + a["asset_id"]]
        fixed.update(name="しずかな調べ", basis="music")
    scene_names.update(state, legacy)
    b = _cue(tmp_path / "ep150", "s2", 600)
    library["s2"].append(b)
    memory = {"labels": {"1": {"members": [a["asset_id"], b["asset_id"]]}},
              "members": {a["asset_id"]: {"path": str(a["path"])}, b["asset_id"]: {"path": str(b["path"])}}}
    from lss.bgm import cue_memory
    monkeypatch.setattr(cue_memory, "read", lambda root: memory)
    monkeypatch.setattr(cue_memory, "_views", lambda data: {})
    models = FakeModels([])
    scene_job.name_source(state, tmp_path / "series-alpha-150.mkv", "s2", models=models)
    assert models.calls == []
    assert FLAC(b["path"])["TITLE"] == ["しずかな調べ（仮題）"]
    assert scene_names.read(state)["recordings"]["label:1"]["basis"] == "music"


def test_background_scene_worker_stays_hidden_with_redirected_stdio(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from lss import paths
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    monkeypatch.setattr(scene_job, "available", lambda: True)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *args, **kwargs: True)
    calls = []
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["stdout"].name.endswith(".log")
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(subprocess, "Popen", launch)
    result = scene_job.start_in_background(tmp_path, tmp_path / "episode.mkv", "a" * 64)
    assert result["state"] == "started" and result["pid"] == 123
    kwargs = calls[0][1]
    assert not kwargs["creationflags"] & getattr(subprocess, "DETACHED_PROCESS", 0)
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert kwargs["stderr"] == subprocess.STDOUT and kwargs["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize("context_title", [False, True])
def test_the_same_recording_in_another_episode_keeps_the_first_name_without_inference(tmp_path, library, monkeypatch, context_title):
    state = tmp_path / "state"
    a = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(a)
    first_models = SceneUnknown([]) if context_title else FakeModels(["ミズウオ大作戦"])
    first_name = "しずかな調べ" if context_title else "ミズウオ大作戦"
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=first_models)
    b = _cue(tmp_path / "ep150", "s2", 600)
    library["s2"].append(b)
    memory = {"labels": {"1": {"members": [a["asset_id"], b["asset_id"]]}},
              "members": {a["asset_id"]: {"path": str(a["path"])}, b["asset_id"]: {"path": str(b["path"])}}}
    from lss.bgm import cue_memory
    monkeypatch.setattr(cue_memory, "read", lambda root: memory)
    monkeypatch.setattr(cue_memory, "_views", lambda data: {})
    later = FakeModels([])
    result = scene_job.name_source(state, tmp_path / "series-alpha-150.mkv", "s2", models=later)
    assert later.calls == [] and result["copied"] == 1
    assert FLAC(b["path"])["TITLE"] == [f"{first_name}（仮題）"] and FLAC(a["path"])["TITLE"] == [f"{first_name}（仮題）"]
    assert FLAC(a["path"])["M_NUMBER"] == FLAC(b["path"])["M_NUMBER"]


def test_checked_scene_naming_rejects_initial_source_mismatch_before_any_scene_work(tmp_path, library, monkeypatch):
    source = tmp_path / "series-alpha-137.mkv"
    source.write_bytes(b"published source")
    calls = []
    monkeypatch.setattr(scene_names, "source_facts", lambda *a, **k: calls.append("facts"))
    models = FakeModels([])

    with pytest.raises(scene_job.SourceIdentityMismatch, match="source_identity_mismatch"):
        scene_job.name_source_checked(tmp_path / "state", source, "0" * 64, models=models)

    assert calls == [] and models.calls == []
    assert scene_names.read(tmp_path / "state")["cues"] == {}


def test_checked_scene_naming_rechecks_after_source_facts_before_rekey(tmp_path, library, monkeypatch):
    source = tmp_path / "series-alpha-137.mkv"
    source.write_bytes(b"published source")
    facts_calls, rekey_calls = [], []

    def changing_facts(path, tags=None):
        facts_calls.append(path)
        source.write_bytes(b"replaced during source facts")
        return {**FACTS, "title": FACTS["title"].format(n=137), "episode": 137}

    monkeypatch.setattr(scene_names, "source_facts", changing_facts)
    monkeypatch.setattr(scene_names, "rekey_source", lambda *a: rekey_calls.append(a))
    digest = hashlib.sha256(b"published source").hexdigest()
    library[digest] = []

    with pytest.raises(scene_job.SourceIdentityMismatch, match="source_identity_mismatch"):
        scene_job.name_source_checked(tmp_path / "state", source, digest, models=FakeModels([]))

    assert len(facts_calls) == 1 and rekey_calls == []
    assert scene_names.read(tmp_path / "state")["cues"] == {}


def test_transcribe_source_change_records_nothing_new_and_releases_models(tmp_path, library):
    source = tmp_path / "series-alpha-137.mkv"
    original_bytes = b"stable episode source"
    source.write_bytes(original_bytes)
    digest = hashlib.sha256(original_bytes).hexdigest()
    library[digest] = []
    first = _cue(tmp_path / "ep137", digest, 400)
    library[digest].append(first)
    seeded = FakeModels(["ミズウオ大作戦"])
    scene_job.name_source(tmp_path / "state", source, digest, models=seeded)
    before = scene_names.read(tmp_path / "state")
    first_tags = dict(FLAC(first["path"]))
    first_pcm = audio_identity.audio_identity(first["path"])["pcm_sha256"]

    second = _cue(tmp_path / "ep137", digest, 800, tone=330)
    library[digest].append(second)

    class ChangesDuringTranscribe(FakeModels):
        def transcribe(self, media, windows):
            self.calls.append(("transcribe", windows))
            source.write_bytes(b"replaced while ASR reads")
            return [{"start": 395.0, "end": 399.0, "text": "ミズウオだ！捕まえよう"}]

        def order(self, prompt):
            pytest.fail("a changed source must be rejected before scene inference")

    models = ChangesDuringTranscribe([])
    with pytest.raises(scene_job.SourceIdentityMismatch, match="source_identity_mismatch"):
        scene_job.name_source_checked(tmp_path / "state", source, digest, models=models)

    after = scene_names.read(tmp_path / "state")
    assert set(after["cues"]) == set(before["cues"])
    assert after["cues"] == before["cues"]
    assert dict(FLAC(first["path"])) == first_tags
    assert audio_identity.audio_identity(first["path"])["pcm_sha256"] == first_pcm
    assert audio_identity.audio_identity(second["path"])["pcm_sha256"] == second["pcm_sha256"]
    assert FLAC(second["path"])["TITLE"] == ["cue 800"]
    assert ("release",) in models.calls and not any(call[0] == "order" for call in models.calls)


def test_checked_scene_naming_accepts_matching_source_normally(tmp_path, library):
    source = tmp_path / "series-alpha-137.mkv"
    source.write_bytes(b"verified episode source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cue = _cue(tmp_path / "ep137", digest, 400)
    library[digest] = [cue]
    models = FakeModels(["ミズウオ大作戦"])

    result = scene_job.name_source_checked(tmp_path / "state", source, digest, models=models)

    assert result["named"] == 1
    assert FLAC(cue["path"])["TITLE"] == ["ミズウオ大作戦（仮題）"]
    assert scene_names.read(tmp_path / "state")["cues"][cue["asset_id"]]["source_sha256"] == digest
    assert ("release",) in models.calls


def test_native_unverified_asr_never_reaches_prompt_or_naming_evidence(tmp_path, library):
    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)

    class UnverifiedNative(FakeModels):
        last_transcription_receipt = {"windows": [{"text": "ごめん"}]}

        def revision(self):
            return {"backend": "native-whisper.cpp-llama.cpp"}

        def transcribe(self, source, windows):
            self.calls.append(("transcribe", windows))
            return [{"start": 401, "end": 402, "text": "ごめん"}]

        def order(self, prompt):
            self.calls.append(("order", prompt))
            assert "ごめん" not in prompt
            assert "台詞がないと確認できたわけではない" in prompt
            return {"name_basis": "context_music", "name": "はじまりの余韻", "direction": None,
                    "use": None, "category": None, "evidence": None}

    models = UnverifiedNative([])
    state = tmp_path / "state"
    result = scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models)
    assert result["named"] == 1
    assert not any(call[0] == "transcribe" for call in models.calls)
    entry = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])
    assert entry["heard"] == [] and entry["name_basis"] == "context_music"
    assert entry["use"] is None and entry["name_basis_evidence"]["scene_asserted"] is False
    assert entry["attempt_identity"]["transcription_receipt"] == {
        "state": "ASR_UNVERIFIED", "reason": "native_asr_grounding_unverified", "windows": [],
        "scope": "transcription_unverified_not_silence"}
    assert "ごめん" not in json.dumps(entry, ensure_ascii=False)
    assert audio_identity.audio_identity(cue["path"])["pcm_sha256"] == cue["pcm_sha256"]
    assert ("release",) in models.calls


def test_naming_transcription_other_backend_preserves_words_and_mismatch(tmp_path):
    windows = [(1.0, 2.0)]
    source = tmp_path / "episode.mkv"
    models = FakeModels([])
    heard, receipt = scene_job._naming_transcription(models, source, windows, models.revision())
    assert heard and heard[0]["text"] == "ミズウオだ！捕まえよう"
    assert receipt is None and models.calls == [("transcribe", windows)]

    class Changed(FakeModels):
        def transcribe(self, source, windows):
            raise scene_job.SourceIdentityMismatch("changed")

    with pytest.raises(scene_job.SourceIdentityMismatch):
        scene_job._naming_transcription(Changed([]), source, windows, {"id": "fake"})


def test_refresh_failure_keeps_message_and_ignored_member_path():
    # Case 1: ValueError with Japanese message, member_path None -> no memory_path_ignored
    exc1 = ValueError('公開BGMの音声と来歴の時間軸が一致しません')
    r1 = scene_job._refresh_failure('asset1', exc1, None, Path('/some/path'))
    assert r1 == {"asset_id": "asset1", "error": "ValueError", "message": "公開BGMの音声と来歴の時間軸が一致しません"}

    # Case 2: RuntimeError with 1000-char multi-line message -> message cut to 300, no newline
    long_msg = 'a' * 500 + '\n' + 'b' * 500
    exc2 = RuntimeError(long_msg)
    r2 = scene_job._refresh_failure('asset2', exc2, Path('/other'), Path('/base'))
    assert len(r2["message"]) == 300
    assert '\n' not in r2["message"]
    assert r2["message"] == 'a' * 300

    # Case 3: Exception with empty message -> no 'message' key
    exc3 = ValueError('')
    r3 = scene_job._refresh_failure('asset3', exc3, None, Path('/base'))
    assert "message" not in r3
    assert r3 == {"asset_id": "asset3", "error": "ValueError"}

    # Case 4: member_path equal to path -> no memory_path_ignored key
    exc4 = TypeError('test')
    r4 = scene_job._refresh_failure('asset4', exc4, Path('/same'), Path('/same'))
    assert "memory_path_ignored" not in r4

    # Case 5: member_path different from path -> memory_path_ignored key present
    r5 = scene_job._refresh_failure('asset5', exc4, Path('/diff'), Path('/base'))
    assert r5["memory_path_ignored"] == str(Path('/diff'))


def test_launch_recovery_leaves_a_source_whose_naming_no_user_job_requested(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setattr(scene_job, "_scene_models_cached", lambda: True)
    launched = []
    monkeypatch.setattr(scene_job, "run_background_worker",
                        lambda root, path, digest, **kwargs: launched.append(digest) or {"state": "complete"})
    sources = {}
    for label, requested in (("requested", True), ("library", False)):
        source = tmp_path / f"{label}.mkv"
        source.write_bytes(label.encode())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        cue = _cue(tmp_path / f"public-{label}", digest, 400)
        library[digest] = [cue]
        _write_current_best_for_recovery(state, source, digest, [cue], run_key=label, requested=requested)
        sources[label] = digest
    # Both have unnamed cues (a newer LSS could name them), but only the user's unfinished job is resumed.
    scene_job.run_current_best_recovery(state)
    assert launched == [sources["requested"]]


def test_the_request_record_lives_from_the_user_job_until_its_naming_is_finished(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    library[digest] = [_cue(tmp_path / "public", digest, 400)]
    monkeypatch.setattr(scene_job, "available", lambda: True)
    monkeypatch.setattr(scene_job.subprocess, "Popen", lambda *a, **k: type("P", (), {"pid": 1})())
    assert scene_job.start_in_background(state, source, digest)["state"] == "started"
    assert scene_job._requests(state, digest)
    monkeypatch.setattr(scene_job, "_run_background_worker", lambda *a, **k: {"state": "retry_pending"})
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: 5000.0)
    scene_job.run_background_worker(state, source, digest)
    assert scene_job._requests(state, digest)   # a budgeted retry remains the user's job
    monkeypatch.setattr(scene_job, "_run_background_worker", lambda *a, **k: {"state": "complete"})
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: False)
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: None)
    scene_job.run_background_worker(state, source, digest)
    assert not scene_job._requests(state, digest)


def test_an_exhausted_retry_budget_ends_the_request(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    scene_job._record_request(state, source, digest)
    monkeypatch.setattr(scene_job, "_run_background_worker", lambda *a, **k: {"state": "retry_pending"})
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: False)
    scene_job.run_background_worker(state, source, digest)
    assert not scene_job._requests(state, digest)


def test_a_finishing_worker_covers_a_request_that_arrived_while_nothing_is_left(tmp_path, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    now = [100.0]
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: now[0])
    def worker(*a, **k):
        now[0] = 200.0
        scene_job._record_request(state, source, digest)   # a second job was told already_running
        now[0] = 300.0
        return {"state": "complete"}
    monkeypatch.setattr(scene_job, "_run_background_worker", worker)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: False)
    scene_job.run_background_worker(state, source, digest)
    assert not scene_job._requests(state, digest)


def test_unreadable_naming_state_keeps_the_request(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    library[digest] = [_cue(tmp_path / "public", digest, 400)]
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: 100.0)
    scene_job._record_request(state, source, digest)
    monkeypatch.setattr(scene_job, "_scene_models_cached", lambda: True)
    def unreadable(*a, **k):
        raise OSError("state temporarily locked")
    monkeypatch.setattr(scene_names, "cue_record", unreadable)
    # A launch at a later time would remove the request if the unreadable state counted as "nothing left".
    assert scene_job._eligible_recovery_source(state, digest, [source], now=200.0) is None
    assert scene_job._requests(state, digest)
    monkeypatch.setattr(scene_job, "_run_background_worker", lambda *a, **k: {"state": "current"})
    scene_job.run_background_worker(state, source, digest)   # the worker end asks the same strict question
    assert scene_job._requests(state, digest)


def test_a_waiting_budgeted_retry_keeps_later_requests(tmp_path, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    now = [100.0]
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: now[0])
    def worker(*a, **k):
        now[0] = 200.0
        scene_job._record_request(state, source, digest)
        now[0] = 300.0
        return {"state": "complete"}
    monkeypatch.setattr(scene_job, "_run_background_worker", worker)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: False)
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: 5000.0)   # a cue waits for retry
    scene_job.run_background_worker(state, source, digest)
    assert scene_job._requests(state, digest)


def test_a_finishing_worker_keeps_a_request_that_arrived_after_it_started(tmp_path, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    now = [100.0]
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: now[0])
    def worker(*a, **k):
        now[0] = 200.0
        scene_job._record_request(state, source, digest)   # the user's next job asks again meanwhile
        return {"state": "complete"}
    monkeypatch.setattr(scene_job, "_run_background_worker", worker)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: True)   # the later job's cues still need names
    scene_job.run_background_worker(state, source, digest)
    assert scene_job._requests(state, digest)


def test_launch_recovery_tries_the_requested_place_of_a_moved_source(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    old_place, new_place = tmp_path / "old.mkv", tmp_path / "moved" / "episode.mkv"
    old_place.write_bytes(b"episode")
    digest = hashlib.sha256(old_place.read_bytes()).hexdigest()
    library[digest] = [_cue(tmp_path / "public", digest, 400)]
    new_place.parent.mkdir()
    old_place.rename(new_place)
    scene_job._record_request(state, new_place, digest)
    monkeypatch.setattr(scene_job, "_scene_models_cached", lambda: True)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: True)
    eligible = scene_job._eligible_recovery_source(state, digest, [old_place], now=0.0)
    assert eligible["source"] == new_place


def test_a_worker_that_waited_in_line_does_not_name_a_request_already_finished(tmp_path, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: pytest.fail("finished request re-examined"))
    assert scene_job.run_background_worker(state, source, digest, require_request=True)["state"] == "not_requested"


def test_start_with_nothing_left_to_name_ends_the_requests(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    library[digest] = []
    scene_job._record_request(state, source, digest)
    monkeypatch.setattr(scene_job, "available", lambda: True)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: False)
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: None)
    assert scene_job.start_in_background(state, source, digest)["state"] == "current"
    assert not scene_job._requests(state, digest)


def test_launch_eligibility_ends_a_request_with_nothing_left(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    library[digest] = [_cue(tmp_path / "public", digest, 400)]
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: 100.0)
    monkeypatch.setattr(scene_job, "_scene_models_cached", lambda: True)
    scene_job._record_request(state, source, digest)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: False)
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: None)
    assert scene_job._eligible_recovery_source(state, digest, [source], now=200.0) is None
    assert not scene_job._requests(state, digest)


def test_requests_stay_while_the_sources_bgm_job_holds_its_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: False)
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: None)
    from lss import application as app
    from lss.bgm import current_best
    state = tmp_path / "state"
    digest = "a" * 64
    scene_job._record_request(state, tmp_path / "episode.mkv", digest)
    with app._BgmCandidateRunLock(state / current_best.DIRECTORY / digest, wait_seconds=0):
        scene_job._end_requests(state, digest, None)
        assert scene_job._requests(state, digest)
    scene_job._end_requests(state, digest, None)
    assert not scene_job._requests(state, digest)


def test_launch_tries_the_newest_requested_place_first(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    old_name, new_name = tmp_path / "old-name.mkv", tmp_path / "new-name.mkv"
    old_name.write_bytes(b"episode")
    new_name.write_bytes(b"episode")
    digest = hashlib.sha256(old_name.read_bytes()).hexdigest()
    library[digest] = [_cue(tmp_path / "public", digest, 400)]
    clock = iter([300.0, 100.0])   # the newer request is written first: order must follow requested_at
    monkeypatch.setattr(scene_job, "_retry_clock", lambda: next(clock))
    scene_job._record_request(state, new_name, digest)
    scene_job._record_request(state, old_name, digest)
    monkeypatch.setattr(scene_job, "_scene_models_cached", lambda: True)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: True)
    assert scene_job._eligible_recovery_source(state, digest, [], now=0.0)["source"] == new_name


def test_a_replaced_requested_file_does_not_end_the_request(tmp_path, library, monkeypatch):
    state = tmp_path / "state"
    original, replaced = tmp_path / "original.mkv", tmp_path / "requested.mkv"
    original.write_bytes(b"episode")
    replaced.write_bytes(b"another file now")
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    library[digest] = [_cue(tmp_path / "public", digest, 400)]
    scene_job._record_request(state, replaced, digest)
    monkeypatch.setattr(scene_job, "_scene_models_cached", lambda: True)
    monkeypatch.setattr(scene_job, "needs_naming", lambda root, sha, source=None, **k: source == original)
    eligible = scene_job._eligible_recovery_source(state, digest, [original], now=0.0)
    assert eligible["source"] == original and scene_job._requests(state, digest)
    assert eligible["path_rejections"][0]["reason"] == "recorded_source_identity_mismatch"


def test_the_bgm_job_itself_ends_requests_when_nothing_is_left_to_name(tmp_path, library, monkeypatch):
    from lss import application as app
    from lss.bgm import current_best
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    library[digest] = []
    monkeypatch.setattr(scene_job, "available", lambda: True)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: False)
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: None)
    with app._BgmCandidateRunLock(state / current_best.DIRECTORY / digest, wait_seconds=0):
        scene_job._record_request(state, source, digest)
        assert scene_job.start_in_background(state, source, digest)["state"] == "current"
    assert not scene_job._requests(state, digest)


def test_a_currentbest_switch_in_progress_keeps_the_requests(tmp_path, monkeypatch):
    from lss.bgm import current_best
    state = tmp_path / "state"
    digest = "a" * 64
    scene_job._record_request(state, tmp_path / "episode.mkv", digest)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: False)
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: None)
    journal = current_best.journal_path(state, digest)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text("{}", encoding="utf-8")
    scene_job._end_requests(state, digest, None)
    assert scene_job._requests(state, digest)


def test_a_job_that_starts_during_the_check_keeps_its_request(tmp_path, monkeypatch):
    state = tmp_path / "state"
    digest = "a" * 64
    scene_job._record_request(state, tmp_path / "episode.mkv", digest)
    def check(*a, **k):
        scene_job._record_request(state, tmp_path / "episode.mkv", digest)   # a new job begins meanwhile
        return False
    monkeypatch.setattr(scene_job, "needs_naming", check)
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: None)
    scene_job._end_requests(state, digest, None)
    assert len(scene_job._requests(state, digest)) == 1   # only the request listed before the check ended


def test_a_launch_worker_rechecks_the_request_after_waiting(tmp_path, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    request = scene_job._record_request(state, source, digest)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: False)
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: 50.0)
    def wait(seconds):
        request.unlink()   # the user's job ended the request meanwhile
    result = scene_job.run_background_worker(state, source, digest, wait=wait, clock=lambda: 0.0,
                                             require_request=True)
    assert result["state"] == "not_requested"


def test_a_finishing_worker_leaves_requests_while_the_bgm_job_runs(tmp_path, monkeypatch):
    from lss import application as app
    from lss.bgm import current_best
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: False)
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(scene_job, "_run_background_worker", lambda *a, **k: {"state": "complete"})
    with app._BgmCandidateRunLock(state / current_best.DIRECTORY / digest, wait_seconds=0):
        scene_job._record_request(state, source, digest)   # the job's request, made before the worker started
        scene_job.run_background_worker(state, source, digest)
        assert scene_job._requests(state, digest)


def test_a_launch_worker_rechecks_the_request_after_preparing_models(tmp_path, monkeypatch):
    from lss.bgm import scene_runtime
    state = tmp_path / "state"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"episode")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    request = scene_job._record_request(state, source, digest)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: True)
    monkeypatch.setattr(scene_runtime, "available", lambda: False)
    monkeypatch.setattr(scene_runtime, "bootstrap_supported", lambda: True)
    monkeypatch.setattr(scene_runtime, "ensure_bootstrapped", lambda: request.unlink())   # the job ended it meanwhile
    monkeypatch.setattr(scene_job, "name_source_checked", lambda *a: pytest.fail("ended request was named"))
    result = scene_job.run_background_worker(state, source, digest, require_request=True)
    assert result["state"] == "not_requested"


def test_no_request_ends_while_a_naming_worker_holds_the_source_lease(tmp_path, monkeypatch):
    state = tmp_path / "state"
    digest = "a" * 64
    scene_job._record_request(state, tmp_path / "episode.mkv", digest)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: False)
    monkeypatch.setattr(scene_job, "_pending_retry_with_budget", lambda *a, **k: None)
    with scene_job._background_worker_lease(state, digest) as acquired:
        assert acquired
        scene_job._end_requests(state, digest, None, inside_bgm_job=True)
        assert scene_job._requests(state, digest)
    scene_job._end_requests(state, digest, None, inside_bgm_job=True)
    assert not scene_job._requests(state, digest)
