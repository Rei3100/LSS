"""統合の継ぎ目: 伴奏（選択的な原音合成）・BGM・欲しい音（SOUND）を1つのSTARTで走らせる境界を固定する。

各系統の試験は自分の内側を確かめる。ここは系統の間（Recipeの形・workerの契約・キュー・同梱物）だけを、
CPUと偽の分離器で確かめる。どれも推論せず、GPUにも触れない。
"""
import ast
import json
import re
import threading
from copy import deepcopy
from pathlib import Path

import pytest

from lss import application
from lss.bgm.dialogue_residual import DialogueResidualError
from lss.bgm.event_tagging import EventTaggingError
from lss.bgm.four_stem_analysis import FourStemAnalysisError
from lss.bgm.sfc_analysis import SfcAnalysisError
from lss.core import catalog, jobs, models_store, recipe, recipe_registry, worker
from lss.core.jobs import SeparationQueue
from lss.harness import infer
from lss.separation import selective
from lss.sound import extract, intent, plan, verify
from lss.sound import job as sound_job
from lss.sound.extract import ExtractionError
from lss.sound.sam_audio import SamAudioError
from lss.sound.verify import VerificationError
from scripts import check_runtime_sources as check
from tests.test_sound_job import _source, _Workspace, fake  # noqa: F401  fixture
from tests.test_sound_queue import _audio, _Worker
from tests.test_worker import _fake_msst_separate, _setup, _write_wav
from tests.test_worker_shutdown_boundary import Connection

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = {"method": "libsndfile_direct", "policy_revision": 1}
SERIALIZATION = {"container": "FLAC", "subtype": "PCM_24", "policy_revision": 2}


# --- (a) Recipeの形 -----------------------------------------------------------


def _separation(**changes) -> dict:
    values = {
        "provider": "msst", "arch": "mel_band_roformer", "model_id": "becruily_deux_inst",
        "model_sha256": "a" * 64, "config_sha256": "b" * 64, "runtime_id": "c" * 64,
        "expected_stems": ["Vocals", "Instrumental"], "output_samplerate": 44100,
        "input_preparation": PREPARATION, "serialization": SERIALIZATION,
        "published_stems": ["Instrumental"],
    }
    values.update(changes)
    return recipe.separation_recipe_revision(**values)


def _absence() -> dict:
    return recipe.target_absence_recipe_revision(
        requested_target="Instrumental", input_preparation=PREPARATION,
        validation={"method": "exact_digital_silence", "threshold": "none"})


def _wanted(text: str = "雨の音") -> dict:
    """欲しい音のRecipe。計画と検証の版は実物、分離器の版（重みの全件hash）だけ偽物にする。"""
    (request,) = intent.parse(text)
    return recipe.wanted_sound_recipe_revision(
        request={**request.identity(), "key": request.key()}, label=request.label,
        planner=sound_job.planner_revision(), engines={"fake": 1}, verification=verify.policy_revision(),
        input_preparation=PREPARATION, serialization=SERIALIZATION)


def test_every_recipe_shape_the_three_lines_make_validates_and_round_trips_through_the_registry(tmp_path) -> None:
    plain, composed = _separation(), _separation(post_processing=selective.revision())
    shapes = [("separation-v1-", plain), ("separation-v1-", composed),
              ("target-absence-v1-", _absence()), ("wanted-sound-v1-", _wanted())]
    for prefix, body in shapes:
        assert body["recipe_id"].startswith(prefix)
        assert recipe.validate_recipe_revision(body) == body
        recipe_registry.record(body, state_root=tmp_path)
        assert recipe_registry.revision_for(body["recipe_id"], state_root=tmp_path) == body
    assert len({body["recipe_id"] for _, body in shapes}) == 4
    # 伴奏の2段のRecipeは、段を足しただけの別のRecipe（同じ公開パート）
    assert plain["publication"] == composed["publication"] == {"stems": ["Instrumental"]}


def test_the_selective_stage_follows_the_provider_stage_and_is_part_of_the_identity() -> None:
    post = selective.revision()
    plain, composed = _separation(), _separation(post_processing=post)
    assert len(plain["stage_graph"]) == 1 and len(composed["stage_graph"]) == 2
    # 先頭は今も分離段。来歴の読み手（processing_identity・reprocess）は先頭の段のmodelを読む。
    assert composed["stage_graph"][0]["stage"] == "provider_separation"
    assert composed["stage_graph"][0] == plain["stage_graph"][0]
    assert composed["stage_graph"][1] == {"stage": "selective_original_composition", "revision": post}
    assert post["policy_id"].startswith("selective-v1-")
    assert composed["recipe_id"] != plain["recipe_id"]
    # 実装のbytesや方針が変われば別のRecipe。本文だけ差し替えたものは受け付けない。
    tampered = deepcopy(composed)
    tampered["stage_graph"][1]["revision"]["policy_id"] = "selective-v1-" + "0" * 64
    with pytest.raises(ValueError, match="一致しません"):
        recipe.validate_recipe_revision(tampered)


def test_model_identity_is_read_from_the_provider_stage_and_only_separation_recipes_have_one(tmp_path) -> None:
    expected = {"model_id": "becruily_deux_inst", "model_sha256": "a" * 64, "config_sha256": "b" * 64}
    nothing = {"model_id": None, "model_sha256": None, "config_sha256": None}
    for body, identity in ((_separation(), expected),
                           (_separation(post_processing=selective.revision()), expected),
                           # 対象なし・欲しい音のRecipeにはmodelの同一性が無い。伴奏の「既存の結果」の照会
                           # （application.existing_accompaniment）が、別の系統の完了記録を伴奏と取り違えない。
                           (_absence(), nothing), (_wanted(), nothing)):
        recipe_registry.record(body, state_root=tmp_path)
        assert recipe_registry.processing_identity(body["recipe_id"], state_root=tmp_path) == identity


# --- (b) workerの契約 ---------------------------------------------------------


@pytest.mark.parametrize("error,code", [
    (ExtractionError("x"), "sound_extraction_failed"),
    (SamAudioError("x"), "sound_extraction_failed"),
    (VerificationError("x"), "sound_extraction_failed"),
    (DialogueResidualError("x"), "bgm_analysis_failed"),
    (EventTaggingError("x"), "bgm_analysis_failed"),
    (SfcAnalysisError("x"), "bgm_analysis_failed"),
    (FourStemAnalysisError("x"), "bgm_analysis_failed"),
])
def test_each_lines_exceptions_keep_their_own_error_code(error, code) -> None:
    assert worker._error_code(error) == code


def test_handle_routes_every_command_to_its_own_line(monkeypatch) -> None:
    calls: list[tuple[str, dict, dict]] = []

    def recorder(name: str):
        def handler(job, **kwargs):
            calls.append((name, job, kwargs))
            return {"status": "ok", "via": name}
        return handler

    monkeypatch.setattr(worker, "handle_job", recorder("separate"))
    monkeypatch.setattr(sound_job, "handle_wanted_sound_job", recorder("wanted_sound"))
    bgm = {"bgm_analysis": "handle_bgm_analysis_job", "bgm_four_stem": "handle_bgm_four_stem_job",
           "bgm_event_tags": "handle_bgm_event_tags_job",
           "bgm_instrumental_spans": "handle_bgm_instrumental_spans_job"}
    for cmd, handler in bgm.items():
        monkeypatch.setattr(worker, handler, recorder(cmd))
    progress, destination = (lambda event: None), Path("dest")
    plain = {"mix_path": "a.wav", "model_id": "m", "stem_target": "Instrumental"}
    sound = {"cmd": "wanted_sound", "mix_path": "a.wav", "sounds": ["雨の音"]}

    for job in (sound, plain, {**plain, "cmd": "separate"}, *({"cmd": cmd} for cmd in bgm)):
        assert worker._handle(job, progress=progress, destination_dir=destination)["status"] == "ok"

    assert [(name, job) for name, job, _ in calls] == [
        ("wanted_sound", sound), ("separate", plain), ("separate", {**plain, "cmd": "separate"}),
        *((cmd, {"cmd": cmd}) for cmd in bgm)]
    # 欲しい音のjobだけ、公開の前で止まる確認（終了時の中断）を受け取る。
    assert calls[0][2] == {"progress": progress, "destination_dir": destination,
                           "checkpoint": worker._abandon_checkpoint}
    assert all(kwargs == {"progress": progress, "destination_dir": destination} for _, _, kwargs in calls[1:])


def test_a_wanted_sound_job_over_the_worker_transport_reaches_the_sound_job(monkeypatch, tmp_path) -> None:
    seen = {}

    def sound(job, progress=None, *, destination_dir=None, checkpoint=None):
        seen.update(job=job, destination_dir=destination_dir, checkpoint=checkpoint)
        progress({"phase": "completed", "output_count": 0})
        return {"status": "ok", "sound": {"k": "none", "n": 0}}

    def separate(*args, **kwargs):
        raise AssertionError("欲しい音のjobを伴奏の分離へ渡してはいけない")

    monkeypatch.setattr(sound_job, "handle_wanted_sound_job", sound)
    monkeypatch.setattr(worker, "handle_job", separate)
    job = {"cmd": "wanted_sound", "mix_path": str(tmp_path / "a.wav"), "sounds": ["雨の音"],
           "_stream_progress": True, "destination_dir": str(tmp_path)}
    conn = Connection(job)
    worker._serve_connection(conn, threading.Lock())
    assert conn.responses == [{"type": "progress", "phase": "completed", "output_count": 0},
                              {"status": "ok", "sound": {"k": "none", "n": 0}}]
    assert seen["job"] == job and seen["destination_dir"] == tmp_path
    assert seen["checkpoint"] is worker._abandon_checkpoint
    assert conn.closed


def test_a_sound_failure_in_the_job_is_a_failed_retryable_outcome_in_the_queue(fake: Path, monkeypatch) -> None:
    """経路の計算が失敗したら、workerの `sound_extraction_failed` を、再試行できる「失敗」として記録する。"""

    class Broken(_Workspace):
        def audio(self, node):
            if node == plan.SOURCE:
                return super().audio(node)
            raise ExtractionError("経路の節を計算できません")

    source = _source(fake / "music" / "song.wav")
    worker_fake = _Worker()

    def dispatch(job: dict) -> dict:
        if job.get("cmd") == "wanted_sound":
            return sound_job.handle_wanted_sound_job(job)
        return worker_fake(job)

    queue = SeparationQueue(dispatch, state_path=fake / "queue.json")
    (item_id,) = queue.add([source], "m", "Instrumental")
    queue.freeze({item_id: ["snd"]}, sounds=["雨の音"])
    monkeypatch.setattr(extract, "Workspace", Broken)
    queue.run()
    (item,) = queue.snapshot()
    assert item.status == "failed"
    assert item.outcomes["snd"]["k"] == "failed" and item.outcomes["snd"]["error_code"] == "sound_extraction_failed"
    assert "sound_extraction_failed" not in jobs._UNSUPPORTED_CODES

    monkeypatch.setattr(extract, "Workspace", _Workspace)
    queue.retry(item_id)
    queue.run()
    (item,) = queue.snapshot()
    assert item.status == "completed" and item.outcomes["snd"]["k"] == "done"
    assert item.outcomes["snd"]["files"] == ["song - 雨音.flac"]


# --- (c) 3つの欲しい結果を1つのSTARTで --------------------------------------------

_FAILURES = {"inst": "input_decode_failed", "bgm": "bgm_analysis_failed", "snd": "sound_extraction_failed"}


class _Lines:
    """3系統の偽の処理。呼ばれた順を1つの列へ残し、``fail`` の系統だけエラーを返す。"""

    def __init__(self, fail: str | None = None) -> None:
        self.order: list[str] = []
        self.jobs: list[dict] = []
        self.fail = fail
        self.worker = _Worker()

    def _error(self, target: str) -> dict:
        return {"status": "error", "error_code": _FAILURES[target], "message": "試験の失敗"}

    def dispatch(self, job: dict) -> dict:
        target = "snd" if job.get("cmd") == "wanted_sound" else "inst"
        self.order.append(target)
        self.jobs.append(dict(job))
        return self._error(target) if self.fail == target else self.worker(job)

    def bgm(self, path, run, on_progress) -> dict:
        self.order.append("bgm")
        return self._error("bgm") if self.fail == "bgm" else {"status": "ok", "bgm": {"k": "done", "n": 2}}


def _frozen(tmp_path: Path, lines: _Lines) -> tuple[SeparationQueue, str]:
    queue = SeparationQueue(lines.dispatch, state_path=tmp_path / "queue.json", bgm=lines.bgm)
    (item_id,) = queue.add([_audio(tmp_path / "a.wav")], "m", "Instrumental")
    # 画面の選択順に依らず、凍結する時に表示順（伴奏・BGM・欲しい音）へ揃う。
    queue.freeze({item_id: ["snd", "bgm", "inst"]}, sounds=["雨の音", "ピアノ"])
    return queue, item_id


def test_one_start_with_all_three_targets_runs_them_in_display_order_and_keeps_each_outcome(tmp_path: Path) -> None:
    lines = _Lines()
    queue, _ = _frozen(tmp_path, lines)
    (frozen,) = queue.snapshot()
    assert frozen.targets == ["inst", "bgm", "snd"] and frozen.sounds == ["雨の音", "ピアノ"]
    queue.run()
    assert lines.order == ["inst", "bgm", "snd"]
    assert lines.jobs[1]["cmd"] == "wanted_sound" and lines.jobs[1]["sounds"] == ["雨の音", "ピアノ"]
    (item,) = queue.snapshot()
    assert item.status == "completed" and item.unresolved() == []
    assert item.outcomes["inst"] == {"k": "done", "n": 1}
    assert item.outcomes["bgm"] == {"k": "done", "n": 2}
    assert item.outcomes["snd"]["k"] == "done" and item.outcomes["snd"]["files"] == ["a - 雨音.flac"]
    # 再起動しても、凍結した3つの結果と言葉、結果ごとの到達が残る。
    (restored,) = SeparationQueue(lines.dispatch, state_path=tmp_path / "queue.json", bgm=lines.bgm).snapshot()
    assert restored.targets == ["inst", "bgm", "snd"] and restored.sounds == ["雨の音", "ピアノ"]
    assert restored.outcomes == item.outcomes and restored.status == "completed"


@pytest.mark.parametrize("failing", ["snd", "bgm", "inst"])
def test_a_retry_after_one_failed_target_repeats_only_that_target(tmp_path: Path, failing: str) -> None:
    lines = _Lines(fail=failing)
    queue, item_id = _frozen(tmp_path, lines)
    queue.run()
    # 1つの結果の失敗は、残りの結果を止めない（それぞれが独立の到達）。
    assert lines.order == ["inst", "bgm", "snd"]
    (item,) = queue.snapshot()
    assert item.status == "failed" and item.unresolved() == [failing]
    assert item.outcomes[failing]["k"] == "failed" and item.outcomes[failing]["error_code"] == _FAILURES[failing]
    accepted = {target: deepcopy(outcome) for target, outcome in item.outcomes.items() if target != failing}
    assert set(accepted) == {"inst", "bgm", "snd"} - {failing}
    assert item.result["status"] == ("error" if failing == "inst" else "ok")
    kept_result = deepcopy(item.result)

    lines.fail = None
    lines.order.clear()
    lines.jobs.clear()
    queue.retry(item_id)
    queue.run()
    assert lines.order == [failing]
    if failing == "snd":
        assert lines.jobs[0]["cmd"] == "wanted_sound" and lines.jobs[0]["sounds"] == ["雨の音", "ピアノ"]
    (item,) = queue.snapshot()
    assert item.status == "completed" and item.unresolved() == []
    for target, outcome in accepted.items():
        assert item.outcomes[target] == outcome
    assert item.outcomes[failing]["k"] == "done"
    # 伴奏が受理済みなら、その結果は再試行で消えない。失敗した伴奏は取り直した結果になる。
    assert item.result["status"] == "ok"
    if failing != "inst":
        assert item.result == kept_result


# --- (d) 宣言した差: 欲しい音の「伴奏」は選択的な原音合成を通らない ----------------------------


@pytest.mark.parametrize("kind", ["cinematic", "music"])
def test_the_sound_instrumental_route_is_the_raw_provider_output_without_a_selective_stage(kind: str) -> None:
    """欲しい音の `伴奏` は、伴奏と同じ既定modelの出力を、そのまま親の音の部分集合として返す。
    伴奏の分離にある選択的な原音合成の段は、この経路の式にもRecipeにも無い（宣言した差）。"""
    assert extract.VOCALS_MODEL_ID == catalog.default_model()["model_id"]
    (request,) = intent.parse("伴奏")
    route = plan.route(request, kind)
    assert route.expression == plan.vocals("instrumental", plan.parents(kind)["music"])
    assert route.representation == "extracted"
    assert {node["node"] for node in plan.nodes(route.expression)} <= {"source", "dnr", "vocals"}


def test_the_sound_instrumental_recipe_has_no_selective_stage_and_does_not_depend_on_it() -> None:
    revision = _wanted("伴奏")
    assert [stage["stage"] for stage in revision["stage_graph"]] == ["wanted_sound_extraction"]
    text = json.dumps(revision, ensure_ascii=False)
    assert "selective" not in text and selective.revision()["policy_id"] not in text
    implementation = {item["path"] for item in revision["stage_graph"][0]["planner"]["implementation"]}
    assert implementation == set(sound_job._PLANNER_FILES)
    assert not implementation & {"selective.py", "regions.py"}
    # 経路を組み・計算し・確かめる実装のどこにも、選択的な原音合成は出てこない。
    for path in sorted((ROOT / "lss" / "sound").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "selective" not in source and "separation.regions" not in source, path.name


def test_the_instrumental_separation_recipe_has_the_selective_stage_and_other_parts_do_not(
        tmp_path: Path, monkeypatch) -> None:
    default = catalog.default_model()["model_id"]
    _setup(tmp_path, monkeypatch, model_id=default, state="既定")
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    source = tmp_path / "mix.wav"
    _write_wav(source)
    base = {"mix_path": str(source), "model_id": default}

    instrumental = worker.handle_job({**base, "stem_target": "Instrumental"})
    assert instrumental["status"] == "ok"
    graph = instrumental["recipe_revision"]["stage_graph"]
    assert [stage["stage"] for stage in graph] == ["provider_separation", "selective_original_composition"]
    assert graph[0]["model_id"] == default and graph[1]["revision"] == selective.revision()
    # 2段のRecipeで作った伴奏は、伴奏の「既存の結果」の照会（STARTで再処理を避ける）が今も見つける。
    existing = application.existing_accompaniment(source, instrumental["input_sha256"])
    assert existing is not None and existing["k"] == "reused"
    assert sorted(existing["outputs"]) == sorted(instrumental["outputs"])

    vocals = worker.handle_job({**base, "stem_target": "Vocals"})
    assert vocals["status"] == "ok"
    assert [stage["stage"] for stage in vocals["recipe_revision"]["stage_graph"]] == ["provider_separation"]


# --- (e) 同梱物の一致（静的） ----------------------------------------------------


def _spec_datas() -> list[tuple[str, str]]:
    tree = ast.parse((ROOT / "LSS.spec").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "datas" for t in node.targets):
            return [tuple(entry) for entry in ast.literal_eval(node.value)]
    raise AssertionError("LSS.spec に datas がありません")


def _gated() -> set[str]:
    """build.ps1 の構造スモーク（`$need`）が配布物に要求するパス。"""
    text = (ROOT / "build.ps1").read_text(encoding="utf-8-sig")
    start, end = text.index("$need = @("), text.index("foreach ($r in $need)")
    return {value.replace("\\", "/") for value in re.findall(r'"(_internal\\[^"]+)"', text[start:end])}


def _packaged(source: str, destination: str) -> str:
    """spec の1件（同梱元・同梱先）が、配布物の中で占めるパス。ファイルは同梱先の下、フォルダは同梱先そのもの。"""
    return f"_internal/{destination}" if (ROOT / source).is_dir() else f"_internal/{destination}/{Path(source).name}"


def test_what_the_spec_bundles_for_the_three_lines_is_gated_by_the_build_smoke() -> None:
    datas, gated = _spec_datas(), _gated()
    packaged = {_packaged(source, destination) for source, destination in datas}
    sound = {Path(source).name for source, destination in datas if destination == "runtime-sources/lss/sound"}
    assert sound, "LSS.spec が欲しい音の実行sourceを同梱していません"
    required = {
        "lss/separation/selective.py": "lss/separation", "lss/separation/regions.py": "lss/separation",
        "lss/bgm/episode_usage.py": "lss/bgm", "lss/bgm/release_availability.py": "lss/bgm",
        "lss/bgm/gpu_execution.py": "runtime-sources/lss/bgm",
        "vendor/sam_audio": "vendor/sam_audio",
        **{f"lss/sound/{name}": "runtime-sources/lss/sound" for name in sound},
    }
    for source, destination in required.items():
        assert (source, destination) in datas, f"LSS.spec が同梱していません: {source}"
        assert (ROOT / source).exists(), f"同梱元がありません: {source}"
        assert _packaged(source, destination) in gated, f"build.ps1 の $need が要求していません: {source}"
    assert {path for path in gated if path.startswith("_internal/runtime-sources/lss/sound/")} == {
        f"_internal/runtime-sources/lss/sound/{name}" for name in sound}
    # 逆向きも: 構造スモークが、specの同梱しないsourceを要求していない（配布物が作れなくなる）。
    unbundled = {path for path in gated
                 if path.startswith(("_internal/lss/", "_internal/runtime-sources/")) and path not in packaged}
    assert not unbundled, sorted(unbundled)


def test_the_bundled_sound_sources_are_the_ones_the_runtime_reads_and_the_checker_compares() -> None:
    sound = {Path(source).name for source, destination in _spec_datas() if destination == "runtime-sources/lss/sound"}
    # 凍結した実行は、計画と分離器の版のために、これらを `runtime-sources/lss/sound` から読む。
    assert set(sound_job._PLANNER_FILES) <= sound
    # candidateの検査（同梱物がsourceと同じbytesか）が比べるものと、同梱するものは同じ。
    assert sound == set(check.SOUND_ENGINE_FILES + check.SOUND_PLANNER_FILES)
