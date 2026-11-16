"""常駐workerの「欲しい音」ジョブ（cmd=wanted_sound）。

1つの音源について、言葉で指定された欲しい音を1つずつ成果物にする。

- 欲しい音ごとにRecipe（`lss.core.recipe.wanted_sound_recipe_revision`）を作り、同じ音源・同じRecipeで
  完成済みの成果物があればGPUへ触れずに再利用する。検証して「無い」と確かめたものも記録し、繰り返さない。
- 作るものがあれば、音源の性質（台詞のある番組か、曲か）をタグ付けで確かめ、経路を組み、要る節だけを
  計算する（`lss.sound.extract`）。出力に欲しい音があるかを確かめ（`lss.sound.verify`）、無い秒を静かにして、
  伴奏と同じ書き出し・仕上げ・公開の仕組み（`lss.output.writer`）で ``<音源名> - <欲しい音>.flac`` を保存する。
- 語彙に無い日本語は、訳して元の言葉と同じ意味だと確かめられたときだけ読む（`lss.sound.interpret`。
  読み方は状態の記録へ残し、次からはGPUを使わない）。確かめられなければ成果物を作らず、理由（読めなかった
  語）を返す。推測で別の音を作らない。
- 出力は、その音が聞こえるかを確かめる（AudioSetのタグと、言葉の時刻の証拠）。親には聞こえるのに出力に
  無いときは「無い」と記録せず保留（``held``）にする（無いことと、取り出せなかったことを区別する）。
"""
from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Callable

import soundfile as sf

from lss.core import (audio_input, gpu_lease, presence, recipe, recipe_registry, runtime, session,
                      storage)
from lss.core.scratch import ScratchDirectory
from lss.output import writer
from lss.paths import resource_root
from lss.sound import interpret, intent, plan, verify

WANTED_SOUND = "wanted_sound"
MAX_SOUNDS = intent.MAX_REQUESTS
_PLANNER_FILES = ("intent.py", "lexicon.py", "plan.py", "extract.py", "verify.py", "job.py", "audioset_relations.py")
INTERPRETATIONS = "sound-interpretations.json"


class InvalidSoundJobError(ValueError):
    """欲しい音ジョブの形が契約に合わない。"""


def _source(name: str) -> Path:
    module_source = Path(__file__).with_name(name)
    if not module_source.is_file():
        module_source = resource_root() / "runtime-sources" / "lss" / "sound" / name
    return module_source


def planner_revision() -> dict:
    """要求の読み方・経路の決め方・節の計算の実装（改行の違いでは変わらない）。"""
    files = [{"path": name, "sha256": runtime.text_source_sha256(_source(name))} for name in _PLANNER_FILES]
    return {"schema_version": 1, "implementation": files}


def engines_revision() -> dict:
    """経路で使い得る分離器と検証器の版。重みのsha256と実装の識別で固定する（GPUに触れない）。"""
    from lss.bgm import event_tagging, sfc_analysis
    from lss.core import catalog
    from lss.sound import extract, pe_frame, sam_audio

    row = catalog.get_model(extract.VOCALS_MODEL_ID)
    return {
        "dnr": sfc_analysis.analyzer_revision()["analyzer_id"],
        "vocals": {"model_id": row["model_id"], "sha256": row.get("sha256", ""),
                   "config_sha256": row.get("config_sha256", ""), "arch": extract.VOCALS_ARCH},
        "stems": {"arch": extract.SCNET_ARCH, "weights_sha256": extract.SCNET["sha256"],
                  "config_sha256": extract.SCNET_CONFIG["sha256"]},
        "lead": {"arch": extract.KARAOKE_ARCH, "weights_sha256": extract.KARAOKE["sha256"],
                 "config_sha256": extract.KARAOKE_CONFIG["sha256"]},
        "voices": {"arch": extract.CHORUS_ARCH, "weights_sha256": extract.CHORUS["sha256"],
                   "config_sha256": extract.CHORUS_CONFIG["sha256"]},
        "models": {model_id: extract.model_revision(model_id) for model_id in sorted(extract.MODELS)},
        "text": sam_audio.engine_revision()["engine_id"],
        "frames": pe_frame.engine_revision()["engine_id"],
        "interpreter": interpret.revision()["interpreter_id"],
        "tagger": event_tagging.WEIGHTS["sha256"],
        "msst_runtime": [{"path": path, "sha256": digest} for path, digest in
                         runtime._provider_artifacts((resource_root() / "vendor" / "msst",), normalize_text=True)],
    }


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate(job) -> list[str]:
    if not isinstance(job, dict) or job.get("cmd") != WANTED_SOUND:
        raise InvalidSoundJobError("欲しい音ジョブではありません")
    if not isinstance(job.get("mix_path"), str) or not job["mix_path"].strip():
        raise InvalidSoundJobError("欲しい音ジョブのmix_pathが未指定です")
    sounds = job.get("sounds")
    if (not isinstance(sounds, list) or not sounds or len(sounds) > MAX_SOUNDS
            or not all(isinstance(item, str) and item.strip() for item in sounds)):
        raise InvalidSoundJobError("欲しい音が1つ以上・12個以下の文字列で指定されていません")
    return [item.strip() for item in sounds]


def _item(request: intent.SoundRequest, outcome: str, **values) -> dict:
    item = {"text": request.text, "label": request.label, "k": outcome, **values}
    if request.heard_as:
        item["heard_as"] = list(request.heard_as)
    if request.suggestions:
        item["suggestions"] = list(request.suggestions)
    return item


def _removal_status(paths) -> str | None:
    """Keep a verified no-op visible when reusing an existing removal asset."""
    from mutagen.flac import FLAC

    try:
        statuses = {json.loads(FLAC(path)["LSS_SCORE"][0]).get("removal", {}).get("outcome") for path in paths}
        return statuses.pop() if len(statuses) == 1 else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _aggregate(items: list[dict]) -> dict:
    kinds = [item["k"] for item in items]
    files = [name for item in items for name in item.get("files", [])]
    if "failed" in kinds:
        k = "failed"
    elif "done" in kinds:
        k = "done"
    elif "reused" in kinds:
        k = "reused"
    elif "held" in kinds:
        k = "held"
    elif "none" in kinds:
        k = "none"
    else:
        k = "unsupported"
    return {"k": k, "n": len(files), "files": [Path(name).name for name in files][:500],
            "items": [{key: value for key, value in item.items() if key in
                       {"text", "label", "k", "files", "reason", "unknown", "kind", "route", "decision",
                        "heard_as", "suggestions", "removal_status"}}
                      for item in items]}


def handle_wanted_sound_job(job: dict, progress: Callable[[dict], None] | None = None, *,
                            destination_dir: Path | None = None,
                            checkpoint: Callable[[], None] | None = None) -> dict:
    """欲しい音ジョブを処理する。例外は投げず、workerの結果の形で返す。"""
    from lss.core import worker

    def report(phase: str, **details) -> None:
        if progress is not None:
            try:
                progress({"phase": phase, **details})
            except Exception:
                pass

    checkpoint = checkpoint or (lambda: None)
    resources = ExitStack()
    try:
        report("preparing_input")
        sounds = _validate(job)
        lease = resources.enter_context(_Lease(report))
        reader = interpret.Interpreter(progress=lambda event: report(**event),
                                       cache=session.state_dir() / INTERPRETATIONS, before_gpu=lease.ensure)
        try:
            requests = intent.parse(sounds, reader)
        finally:
            reader.close()
        mix_path = Path(job["mix_path"])
        input_sha256 = session.input_sha256_of(
            mix_path, progress=lambda done, total: report("hashing_input", completed_bytes=done, total_bytes=total))
        decoded = resources.enter_context(audio_input.prepare_input(mix_path, work_root=session.work_dir()))
        session.ensure_input_unchanged(mix_path, input_sha256)
        session.validate_input(decoded)
        info = sf.info(decoded)
        report("validating_input_signal")
        signal = presence.inspect_signal(decoded)
        input_preparation = {**audio_input.input_preparation_revision(mix_path, decoded),
                             "signal_validation": presence.presence_policy_revision()}
        serialization = writer.serialization_recipe_revision(info.subtype)
        planner = planner_revision()
        verification = verify.policy_revision()
        engines = engines_revision()
        items: dict[int, dict] = {}
        pending = []
        for index, request in enumerate(requests):
            if not request.supported:
                reason = request.reason or "unknown_words"
                if reason == "unknown_words" and reader.unavailable is not None:
                    reason = "reader_unavailable"   # 訳のmodelを用意できなかった（言葉が悪いのではない）
                items[index] = _item(request, "unsupported", reason=reason, unknown=list(request.unknown))
                continue
            revision = recipe.wanted_sound_recipe_revision(
                request={**request.identity(), "key": request.key()}, label=request.label, planner=planner,
                engines=engines, verification=verification, input_preparation=input_preparation,
                serialization=serialization)
            recipe_id = revision["recipe_id"]
            zero = session.completed_zero_result(input_sha256, recipe_id, mix_path)
            if zero is not None:
                items[index] = _item(request, "none", reason="verified_absent_reused")
                continue
            checked = session.check_completed_outputs(input_sha256, recipe_id, mix_path)
            if checked and all(checked.values()):
                items[index] = _item(request, "reused", files=[str(path) for path in checked],
                                     removal_status=_removal_status(checked) if request.operation == "remove" else None)
                continue
            pending.append((index, request, revision, checked))
        if pending and signal.exactly_silent:
            for index, request, revision, _ in pending:
                items[index] = _item(request, "none", reason="exact_digital_silence")
                recipe_registry.record(revision, state_root=session.state_dir())
                session.mark_zero_result(input_sha256, revision["recipe_id"], source_path=mix_path,
                                         zero_result=signal.zero_result())
            pending = []
        if pending:
            items.update(_extract(mix_path, decoded, info, input_sha256, pending, engines, report=report,
                                  checkpoint=checkpoint, destination_dir=destination_dir, resources=resources,
                                  lease=lease))
        ordered = [items[index] for index in range(len(requests))]
        outcome = _aggregate(ordered)
        listening = _listening(mix_path, input_sha256, ordered)
        report("completed", output_count=outcome["n"])
        from lss.output import results

        held = sum(1 for item in ordered if item["k"] == "held")
        outputs = [name for item in ordered for name in item.get("files", [])]
        result = {"status": "ok", "sound": outcome, "input_sha256": input_sha256, "listening": listening}
        if outcome["n"]:
            result["product_result"] = results.worker_result({"status": "ok", "outputs": outputs,
                                                              "listening": listening, "input_sha256": input_sha256})
            result["product_result"]["held_count"] = held
        elif held:
            # 聞こえたのに取り出せなかった音がある: 「無い」と記録しない（UNKNOWN と ABSENT を区別する）。
            result["product_result"] = results.project(disposition="HELD", held_count=held,
                                                       run={"source": {"sha256": input_sha256}})
        else:
            result["product_result"] = results.worker_result({
                "status": "ok", "outputs": [], "zero_result": {
                    "reason": "no_wanted_target",
                    "validation_result": {"validator_id": verification["policy_id"], "validator_revision": "1",
                                          "hard_gates_passed": True}},
                "input_sha256": input_sha256})
        return result
    except Exception as exc:  # IPC越しにtracebackを投げない
        code = worker._error_code(exc)
        if isinstance(exc, InvalidSoundJobError):
            code = "invalid_job"
        report("failed", error_code=code, message=str(exc))
        return {"status": "error", "error_code": code, "message": str(exc)}
    finally:
        resources.close()


def _listening(mix_path: Path, input_sha256: str, items: list[dict]) -> list[dict]:
    """公開済み・再利用した成果物の聴取用の仕上げ（伴奏と同じ受領票の確認）。"""
    from mutagen.flac import FLAC

    from lss.core import audio_identity, worker
    from lss.output import finalizer

    finished = []
    for item in items:
        for value in item.get("files", []):
            path = Path(value)
            if not path.is_file():
                continue
            receipt = finalizer.receipt(path)
            if receipt is not None:
                finished.append({"path": str(path), "sha256": audio_identity.file_digest(path),
                                 "audio": receipt["audio"], "finalization": receipt, "reused": True,
                                 "title": (FLAC(path).get("TITLE") or [None])[0]})
                continue
            tags, picture = worker._listening_request(path)
            finished.append(finalizer.ensure(path, tags=tags, picture=picture))
    return finished


def _extract(mix_path: Path, decoded: Path, info, input_sha256: str, pending: list, engines: dict, *,
             report, checkpoint, destination_dir, resources: ExitStack, lease: "_Lease") -> dict[int, dict]:
    """GPUで経路を計算し、確かめて、公開する。"""
    from lss.bgm import event_tagging
    from lss.core import worker
    from lss.harness import vram
    from lss.sound import extract

    out: dict[int, dict] = {}
    source_metadata = writer.read_source_metadata(mix_path)
    final_bytes = writer.ensure_output_capacity(
        mix_path, info.frames / info.samplerate, info.samplerate, info.channels, len(pending),
        writer.capacity_subtype_for(info.subtype), destination_dir=destination_dir)
    storage.ensure_processing_capacity(
        decoded, session.work_dir(),
        destination_dir if destination_dir is not None else writer.output_dir_for(mix_path),
        source_frames=info.frames, source_samplerate=info.samplerate, output_samplerate=info.samplerate,
        channels=max(2, info.channels), generated_stem_count=12, final_outputs_bytes=final_bytes)
    report("checking_runtime", provider="wanted-sound")
    lease.ensure()
    folder = Path(resources.enter_context(ScratchDirectory(prefix=f"{input_sha256[:12]}-sound-",
                                                           dir=session.work_dir(), purpose="separation_inference")))
    workspace = extract.Workspace(decoded, folder / "nodes", progress=lambda event: report(**event),
                                  checkpoint=checkpoint)
    tagger = verify.Tagger.load(event_tagging.ensure_weights(lambda event: report(**event)))
    try:
        source_audio = workspace.audio(plan.SOURCE)
        kind = verify.source_kind(tagger.seconds(source_audio, workspace.rate))
        del source_audio
        tagger.close()
        open_parents, open_evidence = _open_parents(workspace, kind, pending)
        routes = {index: plan.route(request, kind, open_parents=open_parents) for index, request, _, _ in pending}
        removals = {index: plan.removal_targets(request, kind, open_parents=open_parents)
                    for index, request, _, _ in pending if request.operation == "remove"}
        verified_routes = {index: route for index, route in routes.items() if index not in removals}
        verified_routes.update({(index, number): target for index, targets in removals.items()
                                for number, target in enumerate(targets)})
        workspace.prepare([route.expression for route in verified_routes.values()])
        frames = _route_frames(workspace, verified_routes)
        tagger = verify.Tagger.load(event_tagging.ensure_weights())
        for index, request, revision, checked in pending:
            checkpoint()
            route = routes[index]
            removal_details = None
            if index in removals:
                output, removal_details = _remove_verified(
                    workspace, removals[index], [frames.get((index, number)) for number in range(len(removals[index]))],
                    tagger, request.window)
                if output is None:
                    out[index] = _item(request, "held", reason="removal_not_verified", kind=kind,
                                       route=plan.canonical(route.expression), decision=removal_details)
                    continue
                parent = workspace.audio(plan.SOURCE)
                decision = verify.decide(output, parent, workspace.rate, classes=(), family="remainder", tags=None)
            else:
                output, decision = _target_decision(workspace, route, frames.get(index), tagger, request.window)
            details = {"kind": kind, "route": plan.canonical(route.expression), "decision": decision.summary()}
            if removal_details is not None:
                details["decision"]["removal"] = removal_details
            evidence = {cid: open_evidence[cid] for cid in (atom.concept.id for atom in request.atoms)
                        if cid in open_evidence}
            if evidence:
                details["decision"]["parents"] = evidence
            recipe_registry.record(revision, state_root=session.state_dir())
            if decision.outcome == "not_extracted":
                # 親には聞こえるのに取り出せなかった: 無いとは記録しない（次の版の経路で取り直せる）。
                out[index] = _item(request, "held", reason="heard_but_not_extracted", **details)
                continue
            if not decision.present:
                session.mark_zero_result(input_sha256, revision["recipe_id"], source_path=mix_path, zero_result={
                    "reason": "no_wanted_target",
                    "validation_result": {"validator_id": verify.policy_revision()["policy_id"],
                                          "validator_revision": "1", "hard_gates_passed": True,
                                          "source_kind": kind, "route": details["route"],
                                          "decision": decision.summary()}})
                out[index] = _item(request, "none", reason="verified_absent", **details)
                continue
            gated = output if index in removals else verify.apply_keep(
                output, workspace.rate, decision.keep, decision.keep_frames)
            store = folder / f"publish-{index}"
            store.mkdir(parents=True, exist_ok=True)
            label = request.label
            sf.write(str(store / f"{label}.wav"), gated, workspace.rate, subtype="FLOAT", format="WAV")
            report("writing_outputs", stem=label)
            outputs = writer.write_recipe_output(
                mix_path, store, recipe_id=revision["recipe_id"], model="lss-wanted-sound",
                model_sha256=_digest(engines), score=(json.dumps({"route": details["route"], "removal": removal_details},
                    ensure_ascii=False, separators=(",", ":")) if removal_details is not None else f"route={details['route']}"),
                version=worker._lss_version(), input_sha256=input_sha256, include_existing=True,
                reuse_exclusions=frozenset(path for path, valid in (checked or {}).items() if not valid),
                source_subtype=info.subtype, source_metadata=source_metadata,
                validate_before_publish=lambda: (checkpoint(), session.ensure_input_unchanged(mix_path, input_sha256),
                                                 checkpoint()),
                publication_state_root=session.state_dir(), recipe_revision=revision,
                destination_dir=destination_dir, published_stems=[label])
            out[index] = _item(request, "done", files=[str(path) for path in outputs],
                               removal_status=removal_details["outcome"] if removal_details is not None else None, **details)
    finally:
        tagger.close()
        workspace.release_text_engine()
        try:
            vram.release_unused_vram()
        except Exception:
            pass
    return out


def _target_decision(workspace, route, words, tagger, window, *, removal=False):
    import numpy as np

    output = workspace.audio(route.expression)
    parent = workspace.audio(route.parent)
    if window is not None:
        start, end = window.bounds(len(output) / workspace.rate)
        output, parent, words = verify.within(output, parent, words, workspace.rate, start, end)
    tags = tagger.seconds(output, workspace.rate) if route.classes else None
    support = None
    if removal and route.family == "sound" and route.classes:
        support = verify.removal_support(tags, route.classes)
        tags = tags.copy()
        tags[np.ix_(~support, [verify.audioset.INDEX[name] for name in route.classes])] = 0.0
    instrument = route.expression.get("instrument") if route.expression.get("node") == "text" else None
    decision = verify.decide(output, parent, workspace.rate, classes=route.classes, family=route.family,
                             tags=tags, frames=words, parent_presence=bool(instrument) and words is not None,
                             allow_brief_extraction=not removal)
    if removal and words is not None and decision.present:
        # Removing material has a higher cost than exposing an uncertain extract.
        # Require independent AudioSet confirmation as well as the text detector.
        tagged = verify.decide(output, parent, workspace.rate, classes=route.classes,
                               family=route.family, tags=tags) if route.classes else None
        transient = (support is not None and "text" in plan.engines(route.expression)
                     and tagged is not None and not tagged.present
                     and verify.removal_coincident_frames(output, parent, workspace.rate, tags,
                                                          route.classes, words) >= verify.FRAME_PRESENT_FRAMES)
        if transient:
            decision.reasons.append("removal_short_effect_jointly_confirmed")
            # Zero-vs-zero rival ties are not positive support for a new
            # short-effect removal. Keep the existing long-effect path exact.
            support &= tags[:, [verify.audioset.INDEX[name] for name in route.classes]].max(axis=1) \
                >= verify.PRESENT_PROBABILITY
        if tagged is None or (not tagged.present and not transient) or not verify._frames_heard(
                words.output, verify.second_rms_db(output, workspace.rate) >= verify.ENERGY_FLOOR_DBFS):
            decision.present = False
            decision.outcome = "not_extracted"
            decision.reasons.append("removal_needs_independent_target_confirmation")
    if removal and support is not None:
        decision.keep &= support[:len(decision.keep)]
        if decision.keep_frames is not None:
            from lss.sound import pe_frame

            supported_frames = np.repeat(support, pe_frame.FRAMES_PER_SECOND)
            length = min(len(supported_frames), len(decision.keep_frames))
            decision.keep_frames[:length] &= supported_frames[:length]
            decision.keep_frames[length:] = False
    return output, decision


def _remove_verified(workspace, targets, frames, tagger, window):
    """Subtract only independently confirmed targets, preserving the rest of source.

    An absent target is a no-op, never an empty wanted remainder. Conflicting
    parent/output evidence is held. The time window belongs to the removed sound,
    not the full-length remainder. Text estimates with overlapping supports are
    held rather than double-subtracted from the same recorded material.
    """
    import numpy as np

    output = workspace.audio(plan.SOURCE).copy()
    details = {"policy": "verified-removal-v1", "targets": [], "outcome": "unchanged"}
    contributions = []
    for route, words in zip(targets, frames):
        component, decision = _target_decision(workspace, route, words, tagger, window, removal=True)
        evidence = decision.summary()
        evidence["route"] = plan.canonical(route.expression)
        details["targets"].append(evidence)
        if decision.outcome == "not_extracted":
            details["outcome"] = "held"
            return None, details
        if not decision.present:
            continue
        # A text separator cannot authorize removing sound it invented evidence
        # for: the same words must also be confirmed in the recorded parent.
        if words is not None:
            parent = workspace.audio(route.parent)
            if window is not None:
                start, end = window.bounds(len(parent) / workspace.rate)
                _, parent, words = verify.within(parent, parent, words, workspace.rate, start, end)
            if not verify._frames_heard(words.parent, verify.second_rms_db(parent, workspace.rate)
                                        >= verify.ENERGY_FLOOR_DBFS):
                details["outcome"] = "held"
                evidence["reasons"].append("removal_words_not_confirmed_in_parent")
                return None, details
        removed = verify.apply_keep(component, workspace.rate, decision.keep, decision.keep_frames)
        if window is not None:
            start, end = window.bounds(len(removed) / workspace.rate)
            removed[:int(round(start * workspace.rate))] = 0.0
            removed[int(round(end * workspace.rate)):] = 0.0
        for previous, previous_route in contributions:
            shared = (plan.canonical(route.expression) in {plan.canonical(node) for node in plan.nodes(previous_route.expression)}
                      or plan.canonical(previous_route.expression) in {plan.canonical(node) for node in plan.nodes(route.expression)})
            text_overlap = "text" in plan.engines(route.expression) and "text" in plan.engines(previous_route.expression)
            if (shared or text_overlap) and np.any(np.any(removed != 0, axis=1) & np.any(previous != 0, axis=1)):
                details["outcome"] = "held"
                evidence["reasons"].append("overlapping_removal_targets")
                return None, details
        contributions.append((removed, route))
    for removed, _ in contributions:
        output -= removed
    if contributions:
        details["outcome"] = "removed"
    return output, details


def _open_parents(workspace, kind: str, pending: list) -> tuple[dict, dict]:
    """親を測って選ぶ音（語彙に無い言葉・``parent="auto"`` の語彙の音）ごとに、その言葉の音が最も聞こえる親
    （候補の中から）を選ぶ。語彙の音でどの親にも証拠が無ければ、系統で決まる既定の親のまま（選ばない）。"""
    concepts = {atom.concept.id: atom.concept for _, request, _, _ in pending if request.scope is None
                for atom in request.atoms if plan.measured(atom.concept)}
    if not concepts:
        return {}, {}
    candidates = plan.candidate_parents(kind)
    workspace.prepare(candidates)
    probabilities = workspace.frame_probabilities([(parent, concept.prompt) for parent in candidates
                                                   for concept in concepts.values()])
    chosen, evidence = {}, {}
    for cid, concept in concepts.items():
        scored = [(verify.frame_mass(probabilities[(plan.canonical(parent), concept.prompt)]), parent)
                  for parent in candidates]
        best = max(scored, key=lambda item: item[0])
        evidence[cid] = {plan.canonical(parent): round(score, 3) for score, parent in scored}
        if best[0] > 0.0 or cid.startswith("open:"):
            chosen[cid] = best[1]
    return chosen, evidence


def _route_frames(workspace, routes: dict) -> dict[int, "verify.Frames"]:
    """言葉で分離する経路の出力と親で、要求の言葉の時刻の証拠を測る（言葉が複数なら各時刻の最大）。"""
    import numpy as np

    wanted = {index: route for index, route in routes.items()
              if route.prompts and route.representation == "extracted" and "text" in plan.engines(route.expression)}
    items = [(node, prompt) for route in wanted.values() for node in (route.expression, route.parent)
             for prompt in route.prompts]
    if not items:
        return {}
    probabilities = workspace.frame_probabilities(items)
    out = {}
    for index, route in wanted.items():
        def combined(node):
            rows = [probabilities[(plan.canonical(node), prompt)] for prompt in route.prompts]
            length = min(len(row) for row in rows)
            return np.max(np.stack([row[:length] for row in rows]), axis=0)

        out[index] = verify.Frames(output=combined(route.expression), parent=combined(route.parent))
    return out


class _Lease:
    """GPUの排他は、初めてGPUが要るとき（訳・分離）に取り、ジョブの終わりまで持つ。再利用だけならGPUに触れない。"""

    def __init__(self, report):
        self.report = report
        self.lease = None

    def __enter__(self):
        return self

    def ensure(self) -> None:
        if self.lease is None:
            self.lease = gpu_lease.acquire(on_wait=lambda: self.report("checking_runtime", provider="wanted-sound",
                                                                       waiting_for_gpu=True, device_index=0))

    def __exit__(self, *exc):
        if self.lease is not None:
            self.lease.close()
            self.lease = None
        return False
