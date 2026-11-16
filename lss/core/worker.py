"""常駐ワーカー(D-5 L3)。python起動+import(torch/msst等・実測15秒)を1回だけ払い、
以降はIPCで曲を受けて分離する。UIプロセスとは別プロセス(C5)。
"""
from __future__ import annotations

import time
import argparse
import threading
from collections.abc import Callable
from contextlib import ExitStack
from multiprocessing import connection
from multiprocessing.context import AuthenticationError
from pathlib import Path

import soundfile as sf

from lss.core import (audio_identity, audio_input, catalog, gpu_lease, models_store, presence, quitting, recipe,
                      recipe_registry, runtime, session, storage)
from lss.core.protocol import ADDRESS, AUTHKEY, VERSION as PROTOCOL_VERSION
from lss.core.scratch import ScratchDirectory
from lss.harness import fetch, infer, vram
from lss.separation import observation_audio, scene_balance, selective, stem_combination, stem_semantics
from lss.output import writer
from lss.paths import resource_root

_VERSION_FILE = resource_root() / "VERSION"


class InvalidJobError(ValueError):
    """IPCジョブの形またはコマンドが分離契約に合わない。"""


class UnknownModelError(ValueError):
    """指定されたmodel_idが台帳に存在しない。"""


class UnavailableModelError(ValueError):
    """台帳には存在するが、通常利用できない状態のモデルが指定された。"""


class InvalidStemTargetError(ValueError):
    """モデル設定が出力しないパートを要求された。"""


# 画面のprocessのBGM候補の工程と同じ例外（`error_code` は同じ job_abandoned）。
JobAbandonedError = quitting.JobAbandonedError

# 処理中の終了（凍結Design 40a「処理中の 1 件は中断され、保存されません」）。
# GPUの分離そのものは途中で止められないので、分離の前と公開の直前で確かめる。
# 公開を始めた後は止めない（journal が公開を最後まで運ぶ）。
_ABANDON = threading.Event()


def _abandon_checkpoint() -> None:
    if _ABANDON.is_set():
        raise JobAbandonedError("LSS の終了により、処理中の1件を保存せずに中断しました")


def _plan_algebra(combination, model_id: str, expected_stems: list[str], semantics: dict,
                  single_error: Exception):
    """このjobのmodelを最初のpassとする別passの組み合わせ（applicationと同じ計画器・同じ候補順）。

    作れなければ1passでの拒否理由をそのまま返す（従来どおり）。
    """
    candidates = []
    for row in catalog.list_models():
        if row["model_id"] == model_id:
            stems = tuple(expected_stems)
        else:
            try:
                stems = tuple(catalog.stem_targets(row))
            except Exception:
                continue
        candidates.append(stem_combination.PassCandidate(
            model_id=row["model_id"], expected_stems=stems, semantics=semantics.get(row["model_id"])))
    try:
        return stem_combination.plan_cross_pass(combination, candidates, first_model=model_id)
    except stem_combination.StemCombinationError:
        raise single_error from None


def _prepare_extra_passes(plan, model_id: str, semantics: dict, output_samplerate: int,
                          input_samplerate: int, progress) -> list[dict]:
    """後続passのmodelを用意し、揃いと意味を推論の前に確かめる（証明できなければ何も作らない）。"""
    extras = []
    for other in plan.passes:
        if other == model_id:
            continue
        row = catalog.get_model(other)
        size_bytes = int(row["size_bytes"]) if row.get("size_bytes") else None
        ckpt, config = models_store.ensure_model_cached(
            other, row["distributor_url"], row["config_url"], size_bytes,
            models_tsv_path=models_store.MODELS_TSV, progress=progress,
        )
        stems, samplerate = infer.expected_output_layout(config, input_samplerate)
        # 計画後に設定が変わっていれば、宣言した意味はこの出力に当てはまらない。
        stem_combination.declared_content(stems, semantics.get(other) or {})
        if samplerate != output_samplerate:
            raise stem_combination.StemCombinationIncompleteError(
                f"分離passの出力レートが揃いません（{model_id}: {output_samplerate} Hz, {other}: {samplerate} Hz）"
            )
        target = next(term.stem for term in plan.terms if term.model_id == other)
        if target not in stems:
            raise stem_combination.StemCombinationIncompleteError(
                f"組み合わせの部品{other}/{target}をmodelが出力しません")
        extras.append({"row": row, "ckpt": ckpt, "config": config, "expected_stems": list(stems),
                       "target": target})
    return extras


def _validate_job(job: dict) -> None:
    if not isinstance(job, dict):
        raise InvalidJobError("分離ジョブはオブジェクトである必要があります")
    if job.get("cmd", "separate") != "separate":
        raise InvalidJobError(f"未対応のコマンドです: {job.get('cmd')}")
    for key in ("mix_path", "model_id", "stem_target"):
        if not isinstance(job.get(key), str) or not job[key].strip():
            raise InvalidJobError(f"分離ジョブの{key}が未指定です")


def _error_code(exc: Exception) -> str:
    from lss.bgm.dialogue_residual import DialogueResidualError
    from lss.bgm.event_tagging import EventTaggingError
    from lss.bgm.four_stem_analysis import FourStemAnalysisError
    from lss.bgm.sfc_analysis import SfcAnalysisError

    from lss.sound.extract import ExtractionError
    from lss.sound.sam_audio import SamAudioError
    from lss.sound.verify import VerificationError

    if isinstance(exc, (SamAudioError, ExtractionError, VerificationError)):
        return "sound_extraction_failed"
    if isinstance(exc, (SfcAnalysisError, FourStemAnalysisError, EventTaggingError, DialogueResidualError)):
        return "bgm_analysis_failed"
    if isinstance(exc, JobAbandonedError):
        return "job_abandoned"
    if isinstance(exc, InvalidJobError):
        return "invalid_job"
    if isinstance(exc, UnknownModelError):
        return "unknown_model"
    if isinstance(exc, UnavailableModelError):
        return "model_unavailable"
    if isinstance(exc, (stem_combination.StemCombinationError, stem_combination.StemCombinationIncompleteError)):
        return exc.code
    if isinstance(exc, InvalidStemTargetError):
        return "invalid_stem_target"
    if isinstance(exc, models_store.InvalidModelLedgerError):
        return "invalid_model_ledger"
    if isinstance(exc, session.UnsupportedChannelLayoutError):
        return "unsupported_channels"
    if isinstance(exc, session.InputChangedError):
        return "input_changed"
    if isinstance(exc, presence.InvalidInputSignalError):
        return "invalid_input_signal"
    if isinstance(exc, audio_input.DecoderUnavailableError):
        return "decoder_unavailable"
    if isinstance(exc, audio_input.InputDecodeTimeoutError):
        return "input_decode_timeout"
    if isinstance(exc, audio_input.InputDecodeError):
        return "input_decode_failed"
    if isinstance(exc, writer.InsufficientOutputSpaceError):
        return "insufficient_output_space"
    if isinstance(exc, writer.UnsafeSerializationError):
        return "unsafe_output_serialization"
    if isinstance(exc, models_store.InsufficientCacheSpaceError):
        return "insufficient_model_cache_space"
    if isinstance(exc, fetch.InvalidModelSourceError):
        return "invalid_model_source"
    if isinstance(exc, fetch.DownloadTooLargeError):
        return "model_download_too_large"
    if isinstance(exc, fetch.DownloadIntegrityError):
        return "model_download_invalid"
    if isinstance(exc, fetch.DownloadSizeError):
        return "model_download_failed"
    if isinstance(exc, fetch.ModelDownloadError):
        return "model_download_failed"
    if isinstance(exc, (infer.RuntimeUnavailableError, gpu_lease.GpuLeaseUnavailableError)):
        return "gpu_unavailable"
    if isinstance(exc, infer.InvalidSeparationOutputError):
        return "invalid_separation_output"
    if isinstance(exc, FileNotFoundError):
        return "file_not_found"
    return "processing_failed"


def _processing_signature(revision: dict) -> dict:
    """Recipe本文から実行環境の識別（runtime_id）とIDを除いたもの。"""
    signature = {key: value for key, value in revision.items() if key != "recipe_id"}
    signature["stage_graph"] = [{key: value for key, value in stage.items() if key != "runtime_id"}
                                for stage in revision.get("stage_graph") or []]
    return signature


def _runtime_equivalent_completion(input_sha256: str, recipe_revision: dict, mix_path: Path):
    """同じ入力・同じ所在を、同じ処理（model・設定・入力準備・直列化・公開パート）で完了した別の実行環境の結果。

    実行環境だけの違い（OSのbuild、依存の版、それらを含むruntime_id）は品質の更新ではない
    （`lss.assets.reprocess` と同じ判断）。その結果を自分のRecipe IDのまま再利用し、同じ曲を推論し直して
    「(r2)」の複製を作らない。出力が今も同じ音として残っているものだけを使う。
    """
    from lss.harness.tsv_io import load_tsv

    ledger = session.completion_ledger_path()
    if not ledger.exists():
        return None
    _, rows = load_tsv(ledger)
    source_key = str(Path(mix_path).resolve())
    wanted = _processing_signature(recipe_revision)
    seen = {recipe_revision["recipe_id"]}
    for row in reversed(rows):
        previous_id = row.get("recipe_id") or ""
        if row.get("input_sha256") != input_sha256 or row.get("source_path") != source_key or previous_id in seen:
            continue
        seen.add(previous_id)
        previous = recipe_registry.revision_for(previous_id)
        if previous is None or _processing_signature(previous) != wanted:
            continue
        checked = session.check_completed_outputs(input_sha256, previous_id, mix_path)
        if checked and all(checked.values()):
            return previous, checked
    return None


def _equivalent_reuse(input_sha256: str, recipe_revision: dict, runtime_revision: dict, mix_path: Path) -> dict | None:
    found = _runtime_equivalent_completion(input_sha256, recipe_revision, mix_path)
    if found is None:
        return None
    previous, checked = found
    _keep_audio_identity(input_sha256, previous["recipe_id"], mix_path)
    return {"status": "skipped", "input_sha256": input_sha256,
            "recipe_id": previous["recipe_id"], "recipe_revision": previous,
            "runtime_id": previous["stage_graph"][0].get("runtime_id"),
            "outputs": [str(path) for path in checked],
            # 今回の実行環境で求めたRecipe。出力の来歴は作った時のRecipeのまま（偽らない）。
            "equivalent_to": {"recipe_id": recipe_revision["recipe_id"],
                              "runtime_id": runtime_revision["runtime_id"],
                              "basis": "runtime_only_difference"}}


def _input_audio(inference_path: Path) -> dict | None:
    try:
        return audio_identity.audio_identity(inference_path)
    except (audio_identity.AudioIdentityError, OSError):
        return None


def _keep_input_audio(input_sha256: str, recipe_id: str, mix_path: Path, inference_path: Path,
                      known: dict | None = None) -> None:
    """完了の行へ入力の復号sampleの同一性を残す。記録できなくても処理は妨げない。"""
    identity = known or _input_audio(inference_path)
    if identity is None:
        return
    try:
        session.record_input_audio(input_sha256, recipe_id, mix_path, identity)
    except (session.CompletionLedgerChangedError, OSError, ValueError):
        pass


def _same_audio_completion(input_sha256: str, recipe_revision: dict, mix_path: Path, inference_path: Path):
    """原本のbytesは変わったが復号した音は同じ入力（再生ソフトのtag書き戻し等）の、同じ処理の完了結果。

    入力の同一性は今もbyte単位（来歴の ``LSS_INPUT_SHA256``）。tagだけの違いを別の入力として推論し直し、
    「(r2)」の複製を作らないため、完了時に記録した復号sampleの同一性と比べる。記録の無い古い完了は
    確かめられないので使わない。
    """
    rows = [row for row in session.completions_with_input_audio(mix_path)
            if row.get("input_sha256") != input_sha256]
    if not rows:
        return None
    current = _input_audio(inference_path)
    if current is None:
        return None
    wanted = _processing_signature(recipe_revision)
    for row in rows:
        if not session.same_input_audio(row["input_audio"], current):
            continue
        previous = recipe_registry.revision_for(row.get("recipe_id") or "")
        if previous is None or _processing_signature(previous) != wanted:
            continue
        checked = session.check_completed_outputs(row["input_sha256"], previous["recipe_id"], mix_path)
        if checked and all(checked.values()):
            return row["input_sha256"], previous, checked, current
    return None


def _same_audio_reuse(input_sha256: str, recipe_revision: dict, runtime_revision: dict, mix_path: Path,
                      inference_path: Path) -> dict | None:
    found = _same_audio_completion(input_sha256, recipe_revision, mix_path, inference_path)
    if found is None:
        return None
    matched_sha256, previous, checked, current = found
    outputs = list(checked)
    # 今のbytesの入力としても完了を記録する（次の照会・r4でも推論し直さない）。出力の来歴は作った時のまま。
    session.mark_completed(input_sha256, previous["recipe_id"], source_path=mix_path, outputs=outputs)
    _keep_input_audio(input_sha256, previous["recipe_id"], mix_path, inference_path, known=current)
    return {"status": "skipped", "input_sha256": input_sha256,
            "provenance_input_sha256": matched_sha256,
            "recipe_id": previous["recipe_id"], "recipe_revision": previous,
            "runtime_id": previous["stage_graph"][0].get("runtime_id"),
            "outputs": [str(path) for path in outputs],
            "equivalent_to": {"recipe_id": recipe_revision["recipe_id"],
                              "runtime_id": runtime_revision["runtime_id"],
                              "basis": "same_decoded_input_audio",
                              "matched_input_sha256": matched_sha256}}


def _keep_audio_identity(input_sha256: str, recipe_id: str, mix_path: Path) -> None:
    """再利用できると確かめた出力へ、後から復号sampleの同一性を残す。

    再生ソフトがtagを書き戻しても、同じ音として再利用できるようにするための記録。
    記録できなくても再利用は妨げない。
    """
    try:
        session.record_missing_audio_identity(
            input_sha256=input_sha256, recipe_id=recipe_id, source_path=mix_path)
    except (session.CompletionLedgerChangedError, OSError, ValueError):
        pass


def _lss_version() -> str:
    try:
        return _VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "不明"


# 場面の種類の観測。宣言した境目は `docs/PRODUCT.md` の「未対応targetは明示する」に沿う。
SCENE_BALANCE_RECIPE = scene_balance.SceneBalanceRecipe(
    absent_below_db=-40.0, quiet_below_db=-3.0, dominant_above_db=3.0)
# 長い素材ではファイル1件の比が境目付近へ寄る。区間ごとの割合も残す。
SCENE_BALANCE_TIMED_RECIPE = scene_balance.TimedSceneBalanceRecipe(
    window_seconds=10.0, hop_seconds=5.0,
    absent_below_db=-40.0, quiet_below_db=-3.0, dominant_above_db=3.0)


def _scene_balance(mix_path: Path, outputs: list) -> dict | None:
    """原本と取り出したstemだけで、その場面の種類を観測する。

    追加の推論はしない。相手のstemは原本から取り出した分を引いたもので代用する。
    区間ごとに読み合わせるので、長い素材でも持ち回る量が増えない。
    読めない場合は観測を付けない。読めないことを場面の種類にしない。
    """
    if len(outputs) != 1:
        return None
    try:
        shift = 0
        try:
            from mutagen.flac import FLAC

            shift = int(FLAC(str(outputs[0]))["LSS_OUTPUT_HEADROOM_SHIFT"][0])
        except Exception:
            shift = 0
        read = observation_audio.segment_energies(
            Path(mix_path), Path(outputs[0]),
            segment_seconds=SCENE_BALANCE_TIMED_RECIPE.hop_seconds,
            estimate_gain=2.0 ** shift)
        observed = scene_balance.read_balance_from_segments(
            read["segments"], SCENE_BALANCE_TIMED_RECIPE,
            other_energy_of=observation_audio.other_energy)
        return {**observed, "read": read["revision"]}
    except Exception:
        # 観測が取れないことを、場面の種類の結論にしない。
        return None


def _handle_audio_job(
    job: dict, progress: Callable[[dict], None] | None = None, *,
    destination_dir: Path | None = None,
) -> dict:
    def report(phase: str, **details) -> None:
        if progress is not None:
            try:
                progress({"phase": phase, **details})
            except Exception:
                # 表示側の失敗でGPU処理や保存を中断しない。
                pass

    def forward_progress(event: dict) -> None:
        details = dict(event)
        phase = details.pop("phase")
        report(phase, **details)

    resources = ExitStack()
    gpu_touched = False
    device_lease = None
    try:
        report("preparing_input")
        _validate_job(job)
        # 組み合わせ（例: ``drums+vocals``）の表記だけをここで確かめる。中身の規則はmodelの出力で決める。
        combination = stem_combination.parse_target(job["stem_target"])
        _, rows = models_store.load_model_ledger(models_store.MODELS_TSV)
        row = next((r for r in rows if r["model_id"] == job["model_id"]), None)
        if row is None:
            raise UnknownModelError(f"model_idが見つからない: {job['model_id']}")
        if row.get("状態") not in {"既定", "候補"}:
            raise UnavailableModelError(
                f"通常利用できないモデル状態です: {job['model_id']} ({row.get('状態', '未登録')})"
            )

        mix_path = Path(job["mix_path"])
        input_sha256 = session.input_sha256_of(
            mix_path,
            progress=lambda completed, total: report(
                "hashing_input", completed_bytes=completed, total_bytes=total,
            ),
        )
        inference_path = resources.enter_context(
            audio_input.prepare_input(mix_path, work_root=session.work_dir())
        )
        # FFmpeg変換中の差し替えを検出し、旧波形と新しい来歴を混在させない。
        session.ensure_input_unchanged(
            mix_path, input_sha256,
            progress=lambda completed, total: report(
                "verifying_input", checkpoint="after_decode",
                completed_bytes=completed, total_bytes=total,
            ),
        )
        session.validate_input(inference_path)
        info = sf.info(inference_path)
        report("validating_input_signal")
        input_presence = presence.inspect_signal(inference_path)

        # Exact digital silenceはmodelから独立した検証済みabsence。重みの取得、容量予約、
        # GPU runtime初期化を行う前に確定し、不要な初回downloadやdisk-pressure failureを避ける。
        if input_presence.exactly_silent:
            recipe_revision = recipe.target_absence_recipe_revision(
                requested_target=job["stem_target"],
                input_preparation={
                    **audio_input.input_preparation_revision(mix_path, inference_path),
                    "signal_validation": presence.presence_policy_revision(),
                },
                validation=input_presence.zero_result()["validation_result"],
            )
            recipe_id = recipe_revision["recipe_id"]
            session.ensure_input_unchanged(mix_path, input_sha256)
            zero_result = session.completed_zero_result(input_sha256, recipe_id, mix_path)
            if zero_result is not None:
                return {
                    "status": "skipped", "outputs": [], "zero_result": zero_result,
                    "input_sha256": input_sha256, "recipe_id": recipe_id,
                    "recipe_revision": recipe_revision, "runtime_id": recipe_id,
                }
            zero_result = input_presence.zero_result()
            # 完了台帳だけではRecipe本文を復元できない。zeroも通常出力と同じく、
            # 本文を検証可能なregistryへ確定してから完了を記録する。
            recipe_registry.record(recipe_revision, state_root=session.state_dir())
            session.mark_zero_result(
                input_sha256, recipe_id, source_path=mix_path, zero_result=zero_result,
            )
            report("completed", output_count=0, zero_reason=zero_result["reason"])
            return {
                "status": "ok", "outputs": [], "zero_result": zero_result,
                "input_sha256": input_sha256, "recipe_id": recipe_id,
                "recipe_revision": recipe_revision, "runtime_id": recipe_id,
            }

        report("checking_model", model_id=job["model_id"])
        size_bytes = int(row["size_bytes"]) if row.get("size_bytes") else None
        ckpt_path, config_path = models_store.ensure_model_cached(
            job["model_id"], row["distributor_url"], row["config_url"], size_bytes,
            models_tsv_path=models_store.MODELS_TSV,
            progress=forward_progress,
        )

        expected_stems, output_samplerate = infer.expected_output_layout(config_path, info.samplerate)
        combination_plan = None
        algebra_plan = None
        # 別passの組み合わせで、このjobのmodelの後に走らせるpass（row, ckpt, config, 出力, 分離target）。
        extra_passes = []
        separation_target = job["stem_target"]
        if combination is not None:
            # 組み合わせは同じpassの出力の和。部品が1本に縮む要求は単一ステムの経路が受け持つ。
            semantics = stem_semantics.load_semantics()
            try:
                combination_plan = stem_combination.plan_combination(
                    combination, expected_stems, semantics.get(job["model_id"]))
            except stem_combination.StemCombinationError as single_error:
                algebra_plan = _plan_algebra(combination, job["model_id"], expected_stems, semantics, single_error)
            if combination_plan is not None:
                if combination_plan.route != "sum_of_stems":
                    raise stem_combination.StemCombinationError(
                        f"組み合わせが1本の出力{combination_plan.components[0]}と同じです（単一ステムの経路で作ります）"
                    )
                separation_target = combination_plan.components[0]
                published_stems = [combination_plan.label]
            else:
                separation_target = next(term.stem for term in algebra_plan.terms
                                         if term.model_id == job["model_id"])
                extra_passes = _prepare_extra_passes(
                    algebra_plan, job["model_id"], semantics, output_samplerate, info.samplerate,
                    forward_progress,
                )
                published_stems = [algebra_plan.label]
        else:
            if not any(stem.lower() == job["stem_target"].lower() for stem in expected_stems):
                raise InvalidStemTargetError(
                    f"モデルが出力しないパートです: {job['stem_target']} (対応: {', '.join(expected_stems)})"
                )
            # 要求された成果物。モデルが相補stemも作る場合、それは公開対象ではない。
            published_stems = [infer.resolve_target_filename(config_path, job["stem_target"])]

        # 完了/zeroの再利用判定より先に、model/config/provider/runtime/復号/直列化を
        # 固定する。同じmodel_idでも実体や実行graphが変われば別Recipeになる。
        model_sha256 = models_store.model_sha256(job["model_id"], ckpt_path, models_store.MODELS_TSV)
        config_sha256 = fetch.file_sha256(config_path)
        report("checking_runtime", provider="msst")
        runtime_object = infer.capture_msst_runtime()
        gpu_touched = True
        runtime_revision = runtime_object.to_dict()
        separation_revision = dict(
            provider="msst",
            arch=row["arch"],
            model_id=job["model_id"],
            model_sha256=model_sha256,
            config_sha256=config_sha256,
            runtime_id=runtime_revision["runtime_id"],
            expected_stems=expected_stems,
            output_samplerate=output_samplerate,
            input_preparation={
                **audio_input.input_preparation_revision(mix_path, inference_path),
                "signal_validation": presence.presence_policy_revision(),
            },
            serialization=writer.serialization_recipe_revision(info.subtype),
        )
        if combination_plan is not None:
            recipe_revision = recipe.stem_combination_recipe_revision(
                **separation_revision, combination=combination_plan.recipe_stage(),
            )
        elif algebra_plan is not None:
            provider_fields = {key: value for key, value in separation_revision.items()
                               if key not in {"input_preparation", "serialization"}}
            pass_revisions = [provider_fields] + [dict(
                provider_fields, arch=extra["row"]["arch"], model_id=extra["row"]["model_id"],
                model_sha256=models_store.model_sha256(
                    extra["row"]["model_id"], extra["ckpt"], models_store.MODELS_TSV),
                config_sha256=fetch.file_sha256(extra["config"]),
                expected_stems=extra["expected_stems"],
            ) for extra in extra_passes]
            recipe_revision = recipe.stem_algebra_recipe_revision(
                passes=pass_revisions,
                input_preparation=separation_revision["input_preparation"],
                serialization=separation_revision["serialization"],
                algebra=algebra_plan.recipe_stage(),
            )
        else:
            recipe_revision = recipe.separation_recipe_revision(
                **separation_revision,
                published_stems=published_stems,
                post_processing=selective.revision() if job["stem_target"].casefold() == "instrumental" else None,
            )
        recipe_id = recipe_revision["recipe_id"]
        session.ensure_input_unchanged(
            mix_path, input_sha256,
            progress=lambda completed, total: report(
                "verifying_input", checkpoint="before_reuse",
                completed_bytes=completed, total_bytes=total,
            ),
        )
        zero_result = session.completed_zero_result(input_sha256, recipe_id, mix_path)
        if zero_result is not None:
            return {
                "status": "skipped",
                "input_sha256": input_sha256,
                "recipe_id": recipe_id,
                "recipe_revision": recipe_revision,
                "runtime_id": runtime_revision["runtime_id"],
                "outputs": [],
                "zero_result": zero_result,
            }
        checked_outputs = session.check_completed_outputs(
            input_sha256, recipe_id, mix_path,
            progress=lambda path, completed, total: report(
                "checking_completed_output", output=str(path),
                completed_bytes=completed, total_bytes=total,
            ),
        )
        if checked_outputs and all(checked_outputs.values()):
            _keep_audio_identity(input_sha256, recipe_id, mix_path)
            _keep_input_audio(input_sha256, recipe_id, mix_path, inference_path)
            return {
                "status": "skipped", "input_sha256": input_sha256,
                "recipe_id": recipe_id, "recipe_revision": recipe_revision,
                "runtime_id": runtime_revision["runtime_id"],
                "outputs": [str(p) for p in checked_outputs],
            }
        equivalent = _equivalent_reuse(input_sha256, recipe_revision, runtime_revision, mix_path)
        if equivalent is not None:
            _keep_input_audio(input_sha256, equivalent["recipe_id"], mix_path, inference_path)
            return equivalent
        same_audio = _same_audio_reuse(input_sha256, recipe_revision, runtime_revision, mix_path, inference_path)
        if same_audio is not None:
            return same_audio

        # 推論後に元ファイルが変わっても、来歴と継承タグを別世代から混ぜない。
        source_metadata = writer.read_source_metadata(mix_path)
        session.ensure_input_unchanged(
            mix_path, input_sha256,
            progress=lambda completed, total: report(
                "verifying_input", checkpoint="after_metadata",
                completed_bytes=completed, total_bytes=total,
            ),
        )
        final_outputs_bytes = writer.ensure_output_capacity(
            mix_path, info.frames / info.samplerate, output_samplerate, info.channels,
            len(published_stems), writer.capacity_subtype_for(info.subtype),
            destination_dir=destination_dir,
        )
        storage.ensure_processing_capacity(
            inference_path, session.work_dir(),
            destination_dir if destination_dir is not None else writer.output_dir_for(mix_path),
            source_frames=info.frames, source_samplerate=info.samplerate,
            output_samplerate=output_samplerate, channels=max(2, info.channels),
            generated_stem_count=len(expected_stems) + (job["stem_target"].casefold() == "instrumental")
            + (combination_plan is not None or algebra_plan is not None)
            + sum(len(extra["expected_stems"]) for extra in extra_passes),
            final_outputs_bytes=final_outputs_bytes,
        )

        device_lease = gpu_lease.acquire(on_wait=lambda: report(
            "checking_runtime", provider="msst", waiting_for_gpu=True, device_index=0,
        ))
        # 順番待ちの間に原本が変わったり、他ワーカーが同じ要求を完成させたりする。
        session.ensure_input_unchanged(mix_path, input_sha256)
        checked_outputs = session.check_completed_outputs(input_sha256, recipe_id, mix_path)
        if checked_outputs and all(checked_outputs.values()):
            _keep_audio_identity(input_sha256, recipe_id, mix_path)
            return {
                "status": "skipped", "input_sha256": input_sha256,
                "recipe_id": recipe_id, "recipe_revision": recipe_revision,
                "runtime_id": runtime_revision["runtime_id"],
                "outputs": [str(p) for p in checked_outputs],
            }
        vram.reset_peak_vram()
        start = time.time()
        work_root = session.work_dir()
        store_dir = Path(resources.enter_context(
            ScratchDirectory(prefix=f"{input_sha256[:12]}-", dir=work_root,
                             purpose="separation_inference")
        ))
        _abandon_checkpoint()
        report("separating", model_id=job["model_id"])
        runtime_revisions = []
        infer.msst_separate(
            row["arch"], inference_path, ckpt_path, config_path, separation_target, store_dir,
            progress=forward_progress, load_estimate=False,
            runtime_observer=runtime_revisions.append,
            runtime_revision=runtime_object,
        )
        pass_dirs = {job["model_id"]: store_dir}
        for extra in extra_passes:
            # 同じ復号済み入力を各modelへ渡す。出力は入力と同じ位置（ゼロ位置）から始まる。
            extra_dir = Path(resources.enter_context(
                ScratchDirectory(prefix=f"{input_sha256[:12]}-", dir=work_root,
                                 purpose="separation_inference")
            ))
            _abandon_checkpoint()
            report("separating", model_id=extra["row"]["model_id"])
            infer.msst_separate(
                extra["row"]["arch"], inference_path, extra["ckpt"], extra["config"], extra["target"],
                extra_dir, progress=forward_progress, load_estimate=False,
                runtime_observer=runtime_revisions.append,
                runtime_revision=runtime_object,
            )
            pass_dirs[extra["row"]["model_id"]] = extra_dir
        if len(runtime_revisions) != 1 + len(extra_passes):
            raise infer.RuntimeUnavailableError(
                "推論に使用したRuntimeRevisionを一意に確定できません"
            )
        observed_runtime_revision = runtime_revisions[0].to_dict()
        if any(item.to_dict() != observed_runtime_revision for item in runtime_revisions[1:]):
            raise infer.RuntimeUnavailableError("分離passごとのRuntimeRevisionが一致しません")
        if observed_runtime_revision != runtime_revision:
            raise infer.RuntimeUnavailableError("推論時のRuntimeRevisionが事前確定値と一致しません")
        selective_receipt = None
        if job["stem_target"].casefold() == "instrumental":
            targets = [p for p in store_dir.iterdir() if p.stem == published_stems[0]
                       and p.suffix.lower() in {".wav", ".flac"}]
            if len(targets) != 1:
                raise infer.InvalidSeparationOutputError("伴奏の選択的合成の入力が一意ではありません")
            report("analyzing_outputs")
            selective_receipt = selective.process(inference_path, targets[0], store_dir / "composed.tmp")
            if selective_receipt.get("staged_path"):
                Path(selective_receipt.pop("staged_path")).replace(targets[0])
        combination_receipt = None
        if combination_plan is not None:
            report("analyzing_outputs")
            # 部品が1本でも欠けたり揃わなかったりすれば、ここで止まり何も公開しない。
            combination_receipt = stem_combination.combine_files(
                store_dir, combination_plan, store_dir / f"{combination_plan.label}.wav",
            )
        elif algebra_plan is not None:
            report("analyzing_outputs")
            # 揃いの最後の確かめ（長さ・チャンネル・レート）に1つでも外れれば、何も公開しない。
            combination_receipt = stem_combination.combine_terms(
                pass_dirs, algebra_plan, store_dir / f"{algebra_plan.label}.wav",
            )
        elapsed = time.time() - start
        session.ensure_input_unchanged(
            mix_path, input_sha256,
            progress=lambda completed, total: report(
                "verifying_input", checkpoint="after_separation",
                completed_bytes=completed, total_bytes=total,
            ),
        )

        audio_seconds = info.frames / info.samplerate
        rtf = elapsed / audio_seconds if audio_seconds else 0.0
        peak = vram.peak_vram_mb()

        excluded = models_store.record_recipe_metrics(
            job["model_id"], rtf, peak, models_tsv_path=models_store.MODELS_TSV,
            audio_seconds=audio_seconds,
        )
        limits_exceeded = models_store.exceeds_runtime_limits(rtf, peak)

        score = f"sdr={row.get('sdr', '')} l1_freq={row.get('l1_freq', '')} aura_mrstft={row.get('aura_mrstft', '')}"
        # 公開の前にRecipe本文を残す。成果物のタグが後から書き換えられても来歴を失わない。
        recipe_registry.record(recipe_revision, state_root=session.state_dir())
        outputs = writer.write_recipe_output(
            mix_path, store_dir,
            recipe_id=recipe_id, model=job["model_id"], model_sha256=model_sha256,
            score=score, version=_lss_version(), input_sha256=input_sha256,
            include_existing=True,
            reuse_exclusions=frozenset(p for p, valid in checked_outputs.items() if not valid),
            source_subtype=info.subtype,
            source_metadata=source_metadata,
            progress=forward_progress,
            validate_before_publish=lambda: (_abandon_checkpoint(), session.ensure_input_unchanged(
                mix_path, input_sha256,
                progress=lambda completed, total: report(
                    "verifying_input", checkpoint="before_publish",
                    completed_bytes=completed, total_bytes=total,
                ),
            ), _abandon_checkpoint()),
            publication_state_root=session.state_dir(),
            completion_progress=lambda path, completed, total: report(
                "recording_completion", output=str(path),
                completed_bytes=completed, total_bytes=total,
            ),
            config_sha256=config_sha256,
            runtime_revision=runtime_revision,
            recipe_revision=recipe_revision,
            destination_dir=destination_dir,
            published_stems=published_stems,
            selective_receipt=selective_receipt,
            stem_combination_receipt=combination_receipt,
        )

        # 原本へ後からtagが書き戻されても、同じ音の入力として見つけられるようにする。
        _keep_input_audio(input_sha256, recipe_id, mix_path, inference_path)
        report("completed", output_count=len(outputs))
        return {
            "status": "ok",
            "outputs": [str(p) for p in outputs],
            # 観測は推論へ渡した信号と比べる。原本が動画等でも同じ経路で読める。
            "scene_balance": _scene_balance(inference_path, outputs),
            "input_sha256": input_sha256,
            "recipe_id": recipe_id,
            "recipe_revision": recipe_revision,
            "elapsed_seconds": elapsed,
            "rtf": rtf,
            "peak_vram_mb": peak,
            "limits_exceeded": limits_exceeded,
            "excluded": excluded,
            "runtime_id": runtime_revision["runtime_id"],
            "runtime_revision": runtime_revision,
            "config_sha256": config_sha256,
            "selective_composition": selective_receipt,
            **({"stem_combination": combination_receipt} if combination_receipt is not None else {}),
        }
    except Exception as exc:  # IPC越しにtracebackをそのまま投げない=呼び出し側との結合を疎に保つ
        error_code = _error_code(exc)
        report("failed", error_code=error_code, message=str(exc))
        failed = {"status": "error", "error_code": error_code, "message": str(exc)}
        # 実測チャンネル数は表示文言そのものが必要とする事実。文字列から掘り返させない。
        if isinstance(exc, session.UnsupportedChannelLayoutError) and exc.channels:
            failed["channels"] = int(exc.channels)
        return failed
    finally:
        try:
            resources.close()
            if gpu_touched:
                try:
                    vram.release_unused_vram()
                except Exception:
                    # 保存済みの処理結果を、後処理だけの失敗でエラーへ反転させない。
                    pass
        finally:
            if device_lease is not None:
                device_lease.close()


_DERIVED_LISTENING_TAGS = frozenset({"LSS_TRUE_PEAK_DBTP", "LSS_INTEGRATED_LOUDNESS_LUFS", "LSS_SOURCE_VOCAL_TYPE"})


def _listening_request(path: Path):
    """The listening request for an accepted output whose receipt no longer validates.

    Provenance and presentation both come from what LSS projected last time (the previous receipt), not from
    whatever a player or tag editor left in the file; a file published before receipts existed supplies its own.
    The finalizer then writes LSS's complete tag set.
    """
    from mutagen.flac import FLAC
    from lss.output import finalizer

    audio = FLAC(path)
    previous = finalizer.previous_record(path)
    owned = previous["projected_tags"] if previous is not None else {key.upper(): list(values) for key, values in audio.items()}
    tags = {key: values for key, values in owned.items()
            if key.startswith("LSS_") and key != finalizer.RECEIPT_TAG and not key.startswith("LSS_LISTENING_")
            and key not in _DERIVED_LISTENING_TAGS}
    tags.update({key.upper(): values for key, values in owned.items() if key.upper() in finalizer.PRESENTATION_KEYS})
    if "LSS_SOURCE_VOCAL_TYPE" in owned and "VOCAL_TYPE" not in tags:
        # The source vocal attribute stays source provenance (the finalizer stores it under this name).
        tags["VOCAL_TYPE"] = owned["LSS_SOURCE_VOCAL_TYPE"]
    artwork = (previous or {}).get("artwork")
    matching = [item for item in audio.pictures if finalizer.picture_identity(item) == artwork]
    # Before receipts LSS wrote at most the source cover, first; the picture set is LSS's to write.
    picture = matching[0] if matching else (audio.pictures[0] if previous is None and audio.pictures else None)
    return tags, picture


def handle_job(job: dict, progress=None, *, destination_dir: Path | None = None) -> dict:
    """Existing processing/completion authority plus listening presentation of accepted outputs."""
    from mutagen.flac import FLAC
    from lss.core import audio_identity
    from lss.output import finalizer, results

    result = _handle_audio_job(job, progress=progress, destination_dir=destination_dir)
    finished = []
    def validate_source():
        if result.get("input_sha256") and fetch.file_sha256(Path(job["mix_path"])) != result["input_sha256"]:
            raise session.InputChangedError("聴取用仕上げの間に原本が変更されました")
    if result.get("status") in {"ok", "skipped"}:
        try:
            for value in result.get("outputs") or []:
                path = Path(value)
                receipt = finalizer.receipt(path)
                # 来歴は出力を作った時の入力（tagだけ違う原本を同じ音として再利用した時は、その元の入力）。
                provenance = {"LSS_INPUT_SHA256": result.get("provenance_input_sha256") or result.get("input_sha256"),
                              "LSS_RECIPE_ID": result.get("recipe_id")}
                if receipt is not None and any(
                        receipt["projected_tags"].get(tag) != [str(value)]
                        for tag, value in provenance.items() if value):
                    receipt = None
                if receipt is not None:
                    finished.append({"path": str(path), "sha256": audio_identity.file_digest(path),
                                     "audio": receipt["audio"], "finalization": receipt, "reused": True,
                                     "title": (FLAC(path).get("TITLE") or [None])[0]})
                    continue
                tags, picture = _listening_request(path)
                for tag, value in provenance.items():
                    if value:
                        tags[tag] = str(value)
                finished.append(finalizer.ensure(path, tags=tags, picture=picture, validate_before_publish=validate_source))
            validate_source()
        except Exception as exc:
            failed = {"status": "error", "error_code": _error_code(exc), "message": str(exc)}
            return {**failed, "product_result": results.worker_result(failed)}
    # 伴奏は1曲1ファイル。結果は完了画面とsessionへ返し、利用者の曲のフォルダへ結果票を増やさない。
    result["product_result"] = results.worker_result({**result, "listening": finished})
    return result


# 言葉で指定した欲しい音（`lss.sound.job`）。
WANTED_SOUND = "wanted_sound"
BGM_ANALYSIS = "bgm_analysis"
# 1回目は原本の解析用複製、2回目は1回目の音楽の出力（効果音の混入が強い候補だけ、該当ブロック）。
BGM_INPUT_ROLES = ("mixture_analysis_copy", "first_pass_music")


def _validate_bgm_analysis_job(job: dict, destination_dir: Path | None) -> Path:
    if not isinstance(job, dict) or job.get("cmd") != BGM_ANALYSIS:
        raise InvalidJobError("BGM解析ジョブではありません")
    for key in ("analysis_path", "analysis_sha256", "analyzer_id"):
        if not isinstance(job.get(key), str) or not job[key].strip():
            raise InvalidJobError(f"BGM解析ジョブの{key}が未指定です")
    path = Path(job["analysis_path"])
    if not path.is_absolute():
        raise InvalidJobError("BGM解析ジョブの入力は絶対パスである必要があります")
    if destination_dir is None or not Path(destination_dir).is_absolute():
        raise InvalidJobError("BGM解析ジョブの保存先が未指定です")
    blocks = job.get("blocks")
    if blocks is not None and (not isinstance(blocks, list) or not blocks or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in blocks)):
        raise InvalidJobError("BGM解析ジョブのblocksが不正です")
    if job.get("input_role", BGM_INPUT_ROLES[0]) not in BGM_INPUT_ROLES:
        raise InvalidJobError("BGM解析ジョブのinput_roleが不正です")
    return path


def handle_bgm_analysis_job(
    job: dict, progress: Callable[[dict], None] | None = None, *,
    destination_dir: Path | None = None,
) -> dict:
    """単話BGM候補の解析用3 stemを作る（`lss.bgm.sfc_analysis`）。GPUは分離と同じく1件ずつ。

    入力は呼び出し側が作った44.1kHzの解析用複製。公開成果物は作らない。stemと、解析器・
    重み・実行環境の同一性を記録した受領票を保存先へ置く。
    """
    import json

    from lss.bgm import sfc_analysis
    from lss.bgm import gpu_execution

    def report(phase: str, **details) -> None:
        if progress is not None:
            try:
                progress({"phase": phase, **details})
            except Exception:
                pass

    device_lease = None
    gpu_touched = False
    try:
        report("preparing_input")
        analysis_path = _validate_bgm_analysis_job(job, destination_dir)
        execution_profile = job.get("execution_profile")
        try:
            gpu_execution.resolve_execution_profile(execution_profile)
        except ValueError as exc:
            raise InvalidJobError(str(exc)) from exc
        revision = sfc_analysis.analyzer_revision(execution_profile=execution_profile)
        if job["analyzer_id"] != revision["analyzer_id"]:
            # 呼び出し側と別の解析器の結果を、同じRecipeの成果として返さない。
            raise InvalidJobError("BGM解析器の版が呼び出し側と一致しません")
        if fetch.file_sha256(analysis_path) != job["analysis_sha256"]:
            raise session.InputChangedError("BGM解析用の音声が記録と一致しません")
        report("checking_model", model_id=sfc_analysis.WEIGHTS["cache_name"])
        weights = sfc_analysis.ensure_weights(progress=progress)
        if execution_profile is not None:
            # Own the GPU before strict settings or CUDA initialization; keep the lease
            # for every forward and every configured inter-forward rest.
            device_lease = gpu_lease.acquire(on_wait=lambda: report(
                "checking_runtime", provider=sfc_analysis.PROVIDER, waiting_for_gpu=True, device_index=0,
            ))
        with gpu_execution.configure_execution(execution_profile) as execution_record:
            report("checking_runtime", provider=sfc_analysis.PROVIDER)
            runtime_revision = runtime.capture_cuda_runtime(
                sfc_analysis.PROVIDER,
                adapter_paths=(sfc_analysis.adapter_source(), sfc_analysis.vendor_root()),
            ).to_dict()
            gpu_touched = True
            if device_lease is None:
                # Preserve legacy ordering when no execution profile was requested.
                device_lease = gpu_lease.acquire(on_wait=lambda: report(
                    "checking_runtime", provider=sfc_analysis.PROVIDER, waiting_for_gpu=True, device_index=0,
                ))
            vram.reset_peak_vram()
            start = time.time()
            report("separating", provider=sfc_analysis.PROVIDER)
            analysis_options = {"progress": progress, "blocks": job.get("blocks")}
            if execution_profile is not None:
                analysis_options["execution_profile"] = execution_profile
            stems = sfc_analysis.separate_stems(
                analysis_path, Path(destination_dir), weights, **analysis_options)
            if stems["analysis_sha256"] != job["analysis_sha256"]:
                raise session.InputChangedError("BGM解析中に解析用の音声が変わりました")
            receipt = {
                "schema_version": 1, "type": "bgm_sfc_analysis_receipt",
                "status": "COMPLETED_CUDA_WITHOUT_FALLBACK",
                "analyzer": revision, "analyzer_id": revision["analyzer_id"],
                "runtime_revision": runtime_revision, "runtime_id": runtime_revision["runtime_id"],
                "input_role": job.get("input_role", BGM_INPUT_ROLES[0]), "blocks": stems["blocks"],
                "analysis": {"path": str(analysis_path), "sha256": stems["analysis_sha256"],
                             "frames": stems["frames"], "channels": stems["channels"],
                             "sample_rate": stems["sample_rate"]},
                "stems": stems["stems"],
                "attention_modules_patched": stems["attention_modules_patched"],
                "elapsed_seconds": time.time() - start, "peak_vram_mb": vram.peak_vram_mb(),
                "publication_state": "not_published",
            }
            if execution_profile is not None:
                receipt["execution_profile"] = execution_profile
                receipt["execution_policy"] = {**execution_record,
                                                "cadence_events": stems.get("cadence_events", [])}
            receipt_path = Path(destination_dir) / "receipt.json"
            session._write_json_atomic(receipt_path, receipt)
        report("completed", output_count=0)
        return {"status": "ok", "receipt": str(receipt_path), "stems": stems["stems"],
                "analyzer_id": revision["analyzer_id"], "runtime_id": runtime_revision["runtime_id"]}
    except Exception as exc:
        error_code = _error_code(exc)
        report("failed", error_code=error_code, message=str(exc))
        return {"status": "error", "error_code": error_code, "message": str(exc)}
    finally:
        try:
            if gpu_touched:
                try:
                    vram.release_unused_vram()
                except Exception:
                    pass
        finally:
            if device_lease is not None:
                device_lease.close()


BGM_FOUR_STEM = "bgm_four_stem"


def _validate_bgm_four_stem_job(job: dict, destination_dir: Path | None) -> Path:
    if not isinstance(job, dict) or job.get("cmd") != BGM_FOUR_STEM:
        raise InvalidJobError("BGMの4分類ジョブではありません")
    for key in ("signal_path", "signal_sha256", "analyzer_id"):
        if not isinstance(job.get(key), str) or not job[key].strip():
            raise InvalidJobError(f"BGMの4分類ジョブの{key}が未指定です")
    path = Path(job["signal_path"])
    if not path.is_absolute():
        raise InvalidJobError("BGMの4分類ジョブの入力は絶対パスである必要があります")
    if destination_dir is None or not Path(destination_dir).is_absolute():
        raise InvalidJobError("BGMの4分類ジョブの保存先が未指定です")
    spans = job.get("spans")
    if (not isinstance(spans, list) or not spans or any(
            not isinstance(span, list) or len(span) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in span)
            or span[0] >= span[1] for span in spans)):
        raise InvalidJobError("BGMの4分類ジョブのspansが不正です")
    hop = job.get("hop_seconds")
    if isinstance(hop, bool) or not isinstance(hop, (int, float)) or hop <= 0:
        raise InvalidJobError("BGMの4分類ジョブのhop_secondsが不正です")
    roles = job.get("roles")
    if not isinstance(roles, list) or not roles or any(not isinstance(role, str) for role in roles):
        raise InvalidJobError("BGMの4分類ジョブのrolesが不正です")
    return path


def handle_bgm_four_stem_job(
    job: dict, progress: Callable[[dict], None] | None = None, *,
    destination_dir: Path | None = None,
) -> dict:
    """BGM候補の歌唱回収のため、1回目の効果音の出力の候補区間だけを4分類へ通す（`lss.bgm.four_stem_analysis`）。

    公開成果物は作らない。区間外0の全長stemと、解析器・重み・実行環境の同一性を記録した受領票を
    保存先へ置く。GPUは分離と同じく1件ずつ。
    """
    from lss.bgm import four_stem_analysis

    def report(phase: str, **details) -> None:
        if progress is not None:
            try:
                progress({"phase": phase, **details})
            except Exception:
                pass

    device_lease = None
    gpu_touched = False
    try:
        report("preparing_input")
        signal_path = _validate_bgm_four_stem_job(job, destination_dir)
        revision = four_stem_analysis.analyzer_revision(job["hop_seconds"])
        if job["analyzer_id"] != revision["analyzer_id"]:
            raise InvalidJobError("BGMの4分類の版が呼び出し側と一致しません")
        if fetch.file_sha256(signal_path) != job["signal_sha256"]:
            raise session.InputChangedError("BGMの4分類へ渡す信号が記録と一致しません")
        report("checking_model", model_id=four_stem_analysis.WEIGHTS["cache_name"])
        weights = four_stem_analysis.ensure_weights(progress=progress)
        report("checking_runtime", provider=four_stem_analysis.PROVIDER)
        runtime_revision = runtime.capture_cuda_runtime(
            four_stem_analysis.PROVIDER,
            adapter_paths=(four_stem_analysis.adapter_source(), four_stem_analysis.vendor_root()),
        ).to_dict()
        gpu_touched = True
        device_lease = gpu_lease.acquire(on_wait=lambda: report(
            "checking_runtime", provider=four_stem_analysis.PROVIDER, waiting_for_gpu=True, device_index=0,
        ))
        vram.reset_peak_vram()
        start = time.time()
        report("separating", provider=four_stem_analysis.PROVIDER)
        stems = four_stem_analysis.separate_spans(
            signal_path, Path(destination_dir), weights, job["spans"], hop_seconds=job["hop_seconds"],
            roles=job["roles"], progress=progress)
        if stems["signal_sha256"] != job["signal_sha256"]:
            raise session.InputChangedError("BGMの4分類の実行中に信号が変わりました")
        receipt = {
            "schema_version": 1, "type": "bgm_four_stem_receipt",
            "status": "COMPLETED_CUDA_WITHOUT_FALLBACK",
            "analyzer": revision, "analyzer_id": revision["analyzer_id"],
            "runtime_revision": runtime_revision, "runtime_id": runtime_revision["runtime_id"],
            "input_role": job.get("input_role", "first_pass_effects"),
            "spans": stems["spans"], "roles": list(job["roles"]),
            "signal": {"path": str(signal_path), "sha256": stems["signal_sha256"],
                       "frames": stems["frames"], "channels": stems["channels"],
                       "sample_rate": stems["sample_rate"]},
            "stems": stems["stems"],
            "elapsed_seconds": time.time() - start, "peak_vram_mb": vram.peak_vram_mb(),
            "publication_state": "not_published",
        }
        receipt_path = Path(destination_dir) / "receipt.json"
        session._write_json_atomic(receipt_path, receipt)
        report("completed", output_count=0)
        return {"status": "ok", "receipt": str(receipt_path), "stems": stems["stems"],
                "analyzer_id": revision["analyzer_id"], "runtime_id": runtime_revision["runtime_id"]}
    except Exception as exc:
        error_code = _error_code(exc)
        report("failed", error_code=error_code, message=str(exc))
        return {"status": "error", "error_code": error_code, "message": str(exc)}
    finally:
        try:
            if gpu_touched:
                try:
                    vram.release_unused_vram()
                except Exception:
                    pass
        finally:
            if device_lease is not None:
                device_lease.close()


BGM_INSTRUMENTAL_SPANS = "bgm_instrumental_spans"


def _validate_bgm_instrumental_spans_job(job: dict, destination_dir: Path | None) -> Path:
    if not isinstance(job, dict) or job.get("cmd") != BGM_INSTRUMENTAL_SPANS:
        raise InvalidJobError("BGMの伴奏区間ジョブではありません")
    for key in ("signal_path", "signal_sha256", "model_id", "recipe_id"):
        if not isinstance(job.get(key), str) or not job[key].strip():
            raise InvalidJobError(f"BGMの伴奏区間ジョブの{key}が未指定です")
    path = Path(job["signal_path"])
    if not path.is_absolute():
        raise InvalidJobError("BGMの伴奏区間ジョブの入力は絶対パスである必要があります")
    if destination_dir is None or not Path(destination_dir).is_absolute():
        raise InvalidJobError("BGMの伴奏区間ジョブの保存先が未指定です")
    spans = job.get("spans")
    if (not isinstance(spans, list) or not spans or any(
            not isinstance(span, list) or len(span) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in span)
            or span[0] >= span[1] for span in spans)):
        raise InvalidJobError("BGMの伴奏区間ジョブのspansが不正です")
    return path


def handle_bgm_instrumental_spans_job(
    job: dict, progress: Callable[[dict], None] | None = None, *,
    destination_dir: Path | None = None,
) -> dict:
    """公開する劇伴cueの台詞の残りを取るため、cueの区間だけを伴奏の既定modelへ通す（`lss.bgm.dialogue_residual`）。

    公開成果物は作らない。cueの部分の伴奏と、model・設定・実行環境の同一性を記録した受領票を保存先へ置く。
    """
    from lss.bgm import dialogue_residual

    def report(phase: str, **details) -> None:
        if progress is not None:
            try:
                progress({"phase": phase, **details})
            except Exception:
                pass

    device_lease = None
    gpu_touched = False
    try:
        report("preparing_input")
        signal_path = _validate_bgm_instrumental_spans_job(job, destination_dir)
        recipe_ = dialogue_residual.default_recipe()
        if job["recipe_id"] != recipe_.recipe_id:
            raise InvalidJobError("BGMの台詞の残りの除去の版が呼び出し側と一致しません")
        _, rows = models_store.load_model_ledger(models_store.MODELS_TSV)
        row = next((r for r in rows if r["model_id"] == job["model_id"]), None)
        if row is None:
            raise UnknownModelError(f"model_idが見つからない: {job['model_id']}")
        if row.get("状態") != "既定":
            raise UnavailableModelError(f"伴奏の既定modelではありません: {job['model_id']} ({row.get('状態')})")
        if fetch.file_sha256(signal_path) != job["signal_sha256"]:
            raise session.InputChangedError("BGMの伴奏区間へ渡す信号が記録と一致しません")
        report("checking_model", model_id=job["model_id"])
        size_bytes = int(row["size_bytes"]) if row.get("size_bytes") else None
        ckpt_path, config_path = models_store.ensure_model_cached(
            job["model_id"], row["distributor_url"], row["config_url"], size_bytes,
            models_tsv_path=models_store.MODELS_TSV, progress=progress)
        stem_target = infer.resolve_target_filename(config_path, recipe_.stem_target)
        model_sha256 = models_store.model_sha256(job["model_id"], ckpt_path, models_store.MODELS_TSV)
        config_sha256 = fetch.file_sha256(config_path)
        report("checking_runtime", provider="msst")
        runtime_object = infer.capture_msst_runtime()
        gpu_touched = True
        runtime_revision = runtime_object.to_dict()
        device_lease = gpu_lease.acquire(on_wait=lambda: report(
            "checking_runtime", provider="msst", waiting_for_gpu=True, device_index=0,
        ))
        vram.reset_peak_vram()
        start = time.time()
        report("separating", model_id=job["model_id"])

        resident = infer.ResidentModel()

        def separate(crop_path: Path, store_dir: Path):
            return infer.msst_separate(row["arch"], crop_path, ckpt_path, config_path, stem_target, store_dir,
                                       runtime_revision=runtime_object, resident_model=resident)

        result = dialogue_residual.run_instrumental_spans(
            signal_path, Path(destination_dir), job["spans"], separate=separate, recipe=recipe_, progress=progress)
        if result["signal_sha256"] != job["signal_sha256"]:
            raise session.InputChangedError("BGMの伴奏区間の実行中に信号が変わりました")
        receipt = {
            "schema_version": 1, "type": "bgm_instrumental_spans_receipt",
            "status": "COMPLETED_CUDA_WITHOUT_FALLBACK",
            "recipe_id": recipe_.recipe_id, "model_id": job["model_id"], "arch": row["arch"],
            "model_sha256": model_sha256, "config_sha256": config_sha256, "stem_target": stem_target,
            "runtime_revision": runtime_revision, "runtime_id": runtime_revision["runtime_id"],
            "input_role": job.get("input_role", "analysis_copy"),
            "spans": result["spans"],
            "signal": {"path": str(signal_path), "sha256": result["signal_sha256"], "frames": result["frames"],
                       "channels": result["channels"], "sample_rate": result["sample_rate"]},
            "instrumental": result["instrumental"],
            "inference_groups": result.get("inference_groups"),
            "inference_frames": result.get("inference_frames"),
            "requested_frames": result.get("requested_frames"),
            "elapsed_seconds": time.time() - start, "peak_vram_mb": vram.peak_vram_mb(),
            "publication_state": "not_published",
        }
        receipt_path = Path(destination_dir) / "receipt.json"
        session._write_json_atomic(receipt_path, receipt)
        report("completed", output_count=0)
        return {"status": "ok", "receipt": str(receipt_path), "instrumental": result["instrumental"],
                "model_id": job["model_id"], "runtime_id": runtime_revision["runtime_id"]}
    except Exception as exc:
        error_code = _error_code(exc)
        report("failed", error_code=error_code, message=str(exc))
        return {"status": "error", "error_code": error_code, "message": str(exc)}
    finally:
        try:
            if gpu_touched:
                try:
                    vram.release_unused_vram()
                except Exception:
                    pass
        finally:
            if device_lease is not None:
                device_lease.close()


BGM_EVENT_TAGS = "bgm_event_tags"


def _validate_bgm_event_tags_job(job: dict, destination_dir: Path | None) -> Path:
    if not isinstance(job, dict) or job.get("cmd") != BGM_EVENT_TAGS:
        raise InvalidJobError("BGMのタグ付けジョブではありません")
    for key in ("signal_path", "signal_sha256", "analyzer_id"):
        if not isinstance(job.get(key), str) or not job[key].strip():
            raise InvalidJobError(f"BGMのタグ付けジョブの{key}が未指定です")
    path = Path(job["signal_path"])
    if not path.is_absolute():
        raise InvalidJobError("BGMのタグ付けジョブの入力は絶対パスである必要があります")
    if destination_dir is None or not Path(destination_dir).is_absolute():
        raise InvalidJobError("BGMのタグ付けジョブの保存先が未指定です")
    return path


def handle_bgm_event_tags_job(
    job: dict, progress: Callable[[dict], None] | None = None, *,
    destination_dir: Path | None = None,
) -> dict:
    """BGM候補の意味判定のため、1回目の音楽の出力を音響イベントのタグ付けへ通す（`lss.bgm.event_tagging`）。

    公開成果物は作らない。窓ごとのクラス確率と埋め込み、解析器・重み・実行環境の同一性を記録した
    受領票を保存先へ置く。GPUは分離と同じく1件ずつ。
    """
    from lss.bgm import event_tagging

    def report(phase: str, **details) -> None:
        if progress is not None:
            try:
                progress({"phase": phase, **details})
            except Exception:
                pass

    device_lease = None
    gpu_touched = False
    try:
        report("preparing_input")
        signal_path = _validate_bgm_event_tags_job(job, destination_dir)
        revision = event_tagging.analyzer_revision()
        if job["analyzer_id"] != revision["analyzer_id"]:
            raise InvalidJobError("BGMのタグ付けの版が呼び出し側と一致しません")
        if fetch.file_sha256(signal_path) != job["signal_sha256"]:
            raise session.InputChangedError("BGMのタグ付けへ渡す信号が記録と一致しません")
        report("checking_model", model_id=event_tagging.WEIGHTS["cache_name"])
        weights = event_tagging.ensure_weights(progress=progress)
        report("checking_runtime", provider=event_tagging.PROVIDER)
        runtime_revision = runtime.capture_cuda_runtime(
            event_tagging.PROVIDER, adapter_paths=(event_tagging.adapter_source(),),
        ).to_dict()
        gpu_touched = True
        device_lease = gpu_lease.acquire(on_wait=lambda: report(
            "checking_runtime", provider=event_tagging.PROVIDER, waiting_for_gpu=True, device_index=0,
        ))
        vram.reset_peak_vram()
        start = time.time()
        report("separating", provider=event_tagging.PROVIDER)
        tagged = event_tagging.tag_to(signal_path, Path(destination_dir), weights, progress=progress)
        if tagged["signal_sha256"] != job["signal_sha256"]:
            raise session.InputChangedError("BGMのタグ付けの実行中に信号が変わりました")
        receipt = {
            "schema_version": 1, "type": "bgm_event_tags_receipt",
            "status": "COMPLETED_CUDA_WITHOUT_FALLBACK",
            "analyzer": revision, "analyzer_id": revision["analyzer_id"],
            "runtime_revision": runtime_revision, "runtime_id": runtime_revision["runtime_id"],
            "input_role": job.get("input_role", "first_pass_music"),
            "signal": {"path": str(signal_path), "sha256": tagged["signal_sha256"]},
            "windows": tagged["windows"], "outputs": tagged["outputs"],
            "elapsed_seconds": time.time() - start, "peak_vram_mb": vram.peak_vram_mb(),
            "publication_state": "not_published",
        }
        receipt_path = Path(destination_dir) / "receipt.json"
        session._write_json_atomic(receipt_path, receipt)
        report("completed", output_count=0)
        return {"status": "ok", "receipt": str(receipt_path), "outputs": tagged["outputs"],
                "analyzer_id": revision["analyzer_id"], "runtime_id": runtime_revision["runtime_id"]}
    except Exception as exc:
        error_code = _error_code(exc)
        report("failed", error_code=error_code, message=str(exc))
        return {"status": "error", "error_code": error_code, "message": str(exc)}
    finally:
        try:
            if gpu_touched:
                try:
                    vram.release_unused_vram()
                except Exception:
                    pass
        finally:
            if device_lease is not None:
                device_lease.close()


def _handle(job, **kwargs) -> dict:
    """IPCのジョブを種類ごとの処理へ渡す。既定は伴奏等の分離。"""
    if isinstance(job, dict) and job.get("cmd") == BGM_ANALYSIS:
        return handle_bgm_analysis_job(job, **kwargs)
    if isinstance(job, dict) and job.get("cmd") == BGM_FOUR_STEM:
        return handle_bgm_four_stem_job(job, **kwargs)
    if isinstance(job, dict) and job.get("cmd") == BGM_EVENT_TAGS:
        return handle_bgm_event_tags_job(job, **kwargs)
    if isinstance(job, dict) and job.get("cmd") == BGM_INSTRUMENTAL_SPANS:
        return handle_bgm_instrumental_spans_job(job, **kwargs)
    if isinstance(job, dict) and job.get("cmd") == WANTED_SOUND:
        from lss.sound import job as sound_job

        return sound_job.handle_wanted_sound_job(job, checkpoint=_abandon_checkpoint, **kwargs)
    return handle_job(job, **kwargs)


def _wake_listener(address: str, authkey: bytes, attempts: int = 5) -> bool:
    """AF_PIPEのacceptには待ち時間が無いため、自分へ1本繋いで戻す。

    ここで戻せないとacceptが待ち続け、終了要求に応えたまま常駐が残る。
    一度の失敗で諦めない。
    """
    for attempt in range(attempts):
        try:
            conn = connection.Client(address, authkey=authkey)
        except (AuthenticationError, EOFError, OSError):
            time.sleep(0.05 * (attempt + 1))
            continue
        conn.close()
        return True
    return False


def _destination_of(job) -> dict:
    """IPCで指定された保存先（単話BGM候補の内部作業領域）。無指定なら音源の隣のLSS。
    BGM解析ジョブは保存先が必須で、無指定なら拒否する。"""
    value = job.get("destination_dir") if isinstance(job, dict) else None
    if isinstance(value, str) and value and Path(value).is_absolute():
        return {"destination_dir": Path(value)}
    return {}


class _JobConnection:
    """ジョブの接続。処理中の終了（abandon）では、別スレッドから閉じられる。送ると閉じるは同時に行わない。"""

    def __init__(self, conn) -> None:
        self.conn = conn
        self._lock = threading.Lock()
        self._closed = False

    def send(self, message) -> None:
        with self._lock:
            if self._closed:
                raise OSError("ジョブの接続は閉じています")
            self.conn.send(message)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self.conn.close()


_JOBS: set[_JobConnection] = set()
_JOBS_LOCK = threading.Lock()


def _release_job_connections() -> None:
    """処理中の終了（abandon）。実行中・待機中のジョブの接続を閉じ、パイプの名前を空ける。

    待ち受けは名前付きパイプの最初のインスタンスとして作るので、同じ名前の接続が1本でも残ると、終了の直後に
    起動し直した LSS の新しいworkerは待ち受けを作れない（ここの推論が終わるまで数分）。終了した LSS は結果を
    受け取らない（処理中の1件は中断、公開を始めていればこのworkerが最後まで運ぶ）。推論と後始末は続く。
    """
    with _JOBS_LOCK:
        jobs = list(_JOBS)
    for job in jobs:
        try:
            job.close()
        except Exception:  # noqa: BLE001 - 閉じられなくても終了要求への応答は止めない
            pass


def _serve_connection(
    conn, job_lock: threading.Lock, stop: threading.Event | None = None,
    address: str = ADDRESS, authkey: bytes = AUTHKEY,
) -> None:
    try:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if isinstance(job, dict) and job.get("cmd") == "ping":
            try:
                conn.send({"status": "ok", "protocol_version": PROTOCOL_VERSION})
            except (EOFError, OSError):
                pass
            return
        if isinstance(job, dict) and job.get("cmd") == "shutdown":
            # 終了要求は処理中でも即答する。実行中のジョブは公開まで仕上げてから落ちる。
            # 利用者が処理中に終了したとき（abandon）は、公開の前で止めて保存しない。
            busy = job_lock.locked()
            abandon = job.get("abandon") is True
            if abandon:
                _ABANDON.set()
            if stop is not None:
                stop.set()
            try:
                conn.send({
                    "status": "ok", "busy": busy, "protocol_version": PROTOCOL_VERSION,
                })
            except (EOFError, OSError):
                pass
            # 待ち受けを起こす前に、この接続と（処理中の終了なら）ジョブの接続を閉じる。待ち受けが先に閉じて
            # いると、残った同じ名前の接続がパイプを「使用中」に見せ、起こす接続が1回20秒ずつ待たされる
            # （その間、名前を掴んだままで、すぐ起動し直した LSS の新しいworkerが待ち受けを作れない）。
            conn.close()
            if abandon:
                _release_job_connections()
            if stop is not None:
                _wake_listener(address, authkey)
            return
        entry = _JobConnection(conn)
        with _JOBS_LOCK:
            _JOBS.add(entry)
        try:
            # 接続確認は処理中でも受け付けるが、GPU・台帳へのアクセスは1件ずつ行う。
            with job_lock:
                if stop is not None and stop.is_set():
                    result = {
                        "status": "error", "error_code": "worker_shutting_down",
                        "message": "分離ワーカーは終了処理中です。未開始の処理は再開後に実行してください。",
                    }
                elif isinstance(job, dict) and job.get("_stream_progress"):
                    def send_progress(event: dict) -> None:
                        try:
                            entry.send({"type": "progress", **event})
                        except Exception:  # noqa: BLE001
                            # 呼び出し元が閉じても（処理中の終了で閉じても）分離と保存は最後まで完了させる。
                            pass

                    result = _handle(job, progress=send_progress, **_destination_of(job))
                else:
                    result = _handle(job, **_destination_of(job))
            try:
                entry.send(result)
            except Exception:  # noqa: BLE001
                # 呼び出し元が終了しても、保存済みの結果と常駐ワーカーを維持する。
                pass
        finally:
            with _JOBS_LOCK:
                _JOBS.discard(entry)
            entry.close()
    finally:
        conn.close()


def _keep_state_backup() -> None:
    """配備版では、起動のたびに保存状態の自動バックアップを確かめる。

    台帳・キュー・来歴はユーザーの成果を指す唯一の記録で、壊れると成果へ戻れない。
    ユーザーに手動のバックアップを求めない。失敗しても処理の受付は止めない。
    """
    from lss.paths import is_frozen

    if not is_frozen():
        return  # 開発ツリーのワーカーはリポジトリへバックアップを書かない
    try:
        from lss.assets import backup

        backup.ensure_recent_backup()
    except Exception as exc:  # noqa: BLE001 - バックアップの失敗で分離を止めない
        import logging

        logging.getLogger(__name__).warning("保存状態の自動バックアップに失敗しました: %s", exc)


def _ensure_reference_library() -> None:
    """配備版では、参照録音の置き場所（アプリのフォルダ内の `References`）が無ければ空で作る。

    利用者が既発売の音源を置く場所を見つけられるようにするだけ。空なら参照は使わない（従来どおり）。
    """
    from lss.paths import is_frozen

    if not is_frozen():
        return  # 開発ツリーのワーカーはリポジトリへフォルダを作らない
    try:
        from lss.bgm import references

        references.ensure_library_root()
    except Exception as exc:  # noqa: BLE001 - 置き場所を作れなくても処理の受付は止めない
        import logging

        logging.getLogger(__name__).warning("参照録音の置き場所を作れませんでした: %s", exc)


def _retire_idle_reference_media() -> None:
    """LSSが取得した参照音源のうち、知識（指紋・和声・取り直しの所在）を保存済みで、使われていないものを退役させる。

    利用者の音源・取り直せない音源・使用中の音源は消さない。途中で止まった退役の続きもここで終える。
    失敗してもworkerの起動は止めない（次回また行う）。
    """
    try:
        from lss.bgm import reference_store

        state = session.state_dir()
        if (state / reference_store.STORE_DIR).is_dir():
            reference_store.retire_idle(cache_dir=state / "bgm-reference-fingerprints", state_root=state)
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("参照音源の退役を次回へ延期しました: %s", exc)


def _reconcile_bgm_current_best() -> None:
    """中断したBGM公開結果の切替を起動時に完了する。失敗してもworkerの起動は止めない（次の処理でも行う）。"""
    try:
        from lss import application

        application.reconcile_bgm_current_best()
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("BGM公開結果の切替の整合に失敗しました: %s", exc)


LISTEN_RETRY_SECONDS = 20.0


def _listen(address: str, authkey: bytes, *, patience: float = LISTEN_RETRY_SECONDS):
    """待ち受けを作る。同じ名前のパイプが前のworkerに残っている間は、空くまで少し待つ。

    待ち受けは最初のインスタンスとして作るので、前のworkerの接続が残っていると拒否される
    （PermissionError）。前のworkerは処理中の終了で接続をすぐ閉じる。待っても空かなければ起動を諦める
    （画面側は「エンジンが止まった」として再試行できる）。
    """
    deadline = time.monotonic() + patience
    while True:
        try:
            return connection.Listener(address, family="AF_PIPE", authkey=authkey)
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)


def serve(
    address: str = ADDRESS, authkey: bytes = AUTHKEY, *,
    stopped: threading.Event | None = None,
) -> None:
    listener = _listen(address, authkey)
    job_lock = threading.Lock()
    stop = stopped if stopped is not None else threading.Event()
    try:
        # 起動時の状態領域へのアクセスが失敗しても、pipeを占有したままにしない。
        session.cleanup_stale_work()
        session.reconcile_publications()
        _reconcile_bgm_current_best()
        _keep_state_backup()
        _ensure_reference_library()
        _retire_idle_reference_media()
        while not stop.is_set():
            try:
                conn = listener.accept()
            except (AuthenticationError, EOFError, ConnectionError):
                continue
            except OSError as exc:
                if getattr(exc, "winerror", None) in {109, 232, 233}:
                    continue
                raise
            if stop.is_set():
                conn.close()
                break
            threading.Thread(
                target=_serve_connection, args=(conn, job_lock, stop, address, authkey),
                daemon=True,
            ).start()
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
    # 実行中のジョブは公開まで仕上げてから終了する。中途半端な成果物を残さない。
    with job_lock:
        pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LSSの分離ワーカー")
    parser.add_argument("--address", default=ADDRESS, help="ローカル名前付きパイプの接続先")
    args = parser.parse_args(argv)
    try:
        serve(address=args.address)
    except Exception:  # noqa: BLE001
        # 画面の無いworkerの例外は、窓なしビルドでは利用者の画面にエラーの窓を出して居残る。
        # 記録して終了コードで返す（画面側は起動の失敗として扱い、再試行できる）。
        import logging

        logging.getLogger(__name__).exception("分離ワーカーを起動・継続できませんでした")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
