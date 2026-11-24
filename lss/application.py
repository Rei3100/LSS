"""Sourceと欲しい結果から、既存の検証済み処理経路へ接続する製品境界。

通常の呼び出し側はmodel、stem、providerを選ばない。伴奏要求は既存の順次処理へ、
単一ローカル番組のBGM要求は単話候補経路（SFC解析＋`lss.bgm.episode_cues`）へ接続する。
意味を未確定のBGM候補は公開成果にせずHELDとして保存し、保存済み結果を安全に読み戻せる。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Callable, Iterable

from lss.core import audio_identity, catalog, jobs, quitting, session
from lss.harness.fetch import file_sha256

if TYPE_CHECKING:
    from lss.acquisition.service import EntryOutcome


class UserIntent(StrEnum):
    EXTRACT_ACCOMPANIMENT = "extract_accompaniment"
    EXTRACT_BGM_CUES = "extract_bgm_cues"
    # モデルステム経路の組み合わせ（例: ボーカル＋ドラム）。部品は ``ProcessingRequest.stems``。
    EXTRACT_STEM_COMBINATION = "extract_stem_combination"


class ResultDisposition(StrEnum):
    PUBLISHED = "PUBLISHED"
    HELD = "HELD"
    ZERO = "ZERO"
    UNSUPPORTED = "UNSUPPORTED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ProcessingRequest:
    sources: tuple[Path, ...]
    intent: UserIntent
    urls: tuple[str, ...] = field(default=(), repr=False)
    # 組み合わせの要求部品（意味的な名前。例: ("vocals", "drums")）。組み合わせの要求でだけ使う。
    stems: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.sources and not self.urls:
            raise ValueError("処理する入力がありません")
        if any(not isinstance(url, str) or not url.strip() for url in self.urls):
            raise ValueError("取得するURLが空、または文字列ではありません")
        object.__setattr__(self, "sources", tuple(Path(path) for path in self.sources))
        object.__setattr__(self, "urls", tuple(url.strip() for url in self.urls))
        object.__setattr__(self, "intent", UserIntent(self.intent))
        if self.intent == UserIntent.EXTRACT_STEM_COMBINATION:
            from lss.separation import stem_combination

            object.__setattr__(self, "stems", stem_combination.normalized_request(tuple(self.stems)))
        elif self.stems:
            raise ValueError("組み合わせの部品は組み合わせの要求でだけ指定できます")

    @classmethod
    def from_paths(cls, paths: Iterable[Path], intent: UserIntent) -> "ProcessingRequest":
        return cls(sources=tuple(paths), intent=intent)

    @classmethod
    def from_urls(cls, urls: Iterable[str], intent: UserIntent) -> "ProcessingRequest":
        return cls(sources=(), intent=intent, urls=tuple(urls))


@dataclass(frozen=True)
class ProcessingOutcome:
    disposition: ResultDisposition
    outputs: tuple[Path, ...] = ()
    held_artifacts: tuple[Path, ...] = ()
    evidence_path: Path | None = None
    record_path: Path | None = None
    recipe_id: str | None = None
    zero_result: dict | None = None
    error_code: str | None = None
    message: str | None = None
    reused: bool = False
    identity_evidence_path: Path | None = None
    source_locations: tuple[Path, ...] = ()
    held_artifact_locations: dict[Path, tuple[Path, ...]] = field(default_factory=dict)
    held_alternative_artifacts: tuple[Path, ...] = ()
    alternative_evidence_path: Path | None = None
    held_reference_residual_artifacts: tuple[Path, ...] = ()
    reference_residual_evidence_path: Path | None = None
    unavailable_evidence: tuple[str, ...] = ()
    # その処理が対応範囲のどの辺りだったかの観測。品質の判定ではない。
    scene_balance: dict | None = None
    product_result: dict | None = None


@dataclass(frozen=True)
class AcquiredInput:
    """取得項目の製品向け結果。資格情報や取得手段の設定は含めない。"""

    url: str
    source_path: Path | None = None
    source_sha256: str | None = None
    reused: bool = False
    offline_reuse: bool = False
    error_code: str | None = None
    message: str | None = None
    retryable: bool = False


@dataclass(frozen=True)
class Submission:
    intent: UserIntent
    queue_item_ids: tuple[str, ...] = ()
    outcome: ProcessingOutcome | None = None
    acquisitions: tuple[AcquiredInput, ...] = ()
    # 組み合わせの要求で選んだ経路（model、部品、和か直接の出力か、含まれて省いた部品）。
    stem_combination: dict | None = None


@dataclass(frozen=True)
class _ResolvedRequest:
    model_id: str
    stem_target: str
    stem_combination: dict | None = None


# 単話だけから候補を得る製品slice。解析器（SFC 3分類）の音楽の取り分で区間を見つけ、
# 曲の切れ目で分け、音楽が続く区間だけを保留候補にする（`lss.bgm.episode_cues`）。
# 旧来の「伴奏modelの出力の大きさ」で区間を見つける方式は、効果音だけ・台詞＋効果音だけの
# 対象不在素材でも候補を8件出していた（`.state/validation/bgm-cue-boundary-r1/`）。
# この方式は同じ素材で0件と10秒1件、実番組12話で既知の劇伴の秒数の98.5%を覆う
# （`.state/validation/lv-bgm-episode-cues-r1/eval-v2/`）。semantic BGMの正式公開Recipeではない。
# v5: 効果音の出力へ誤配分された歌唱を、候補区間だけ4分類で回収し、安全な時だけ足す
# （`lss.bgm.vocal_recovery`、`.state/validation/bgm-effects-vocal-recovery-r1/`）。
# v6: 保留候補を、別familyのタグ付け（`lss.bgm.event_tagging`）で曲単位へ分け直し、欲しい劇伴と高確信に
# 言える区間だけを公開する（`lss.bgm.cue_semantics`、`lss.bgm.cue_publication`、
# `.state/validation/bgm-semantic-disposition-r1/`）。SFC解析の区間の重なりを三角窓で繋ぐ。
# v7: 公開する区間ごとに表現（原音か処理済みか）を選び、2回目と歌唱回収は処理済みで公開する区間の保留候補に
# だけ掛ける（`lss.bgm.cue_representation`、`.state/validation/bgm-work-avoidance-r1/`）。
# v8: 処理済みで公開するcueのうち、SFCの音楽の出力に台詞が残るcueだけを、伴奏の既定modelとの共通成分にする
# （`lss.bgm.dialogue_residual`、`.state/validation/bgm-dialogue-residual-gate-r1/`）。
# v10: cue平均では対象外のcueも、台詞の残る1秒窓の時間だけ同じ共通成分にする（広げない、境目0.25秒。
# `.state/validation/bgm-model-dependence-r1/`）。
# v11: 公開の可否を、先に固定した対応領域S1（`lss.bgm.cue_scope`・`lss.bgm.supported_scope`）で決める。
# 欲しい劇伴の条件を全て満たし、一曲かどうかだけが未確定の区間も、未確定と明示して公開する。
# 書き出す音の完全性（L0）を欠くcueは公開しない（`.state/validation/bgm-product-capability-sprint-r1/`）。
# v11は独立監査で A4_REPAIR_REQUIRED（`.state/validation/bgm-supported-scope-s1-independent-r1/`）。配備していない。
# v12: 対応領域S2。v11の公開範囲に、実行時に観測できる安全の保留を加える（重い非音楽イベントの場面・番組の
# 端・放送挿入の根拠）。公開名は話の中の開始時刻だけで決め（前に別のcueが入っても名前が変わらない）、
# 同じ原本の前の版の結果はCurrentBestの切替で置き換えて履歴へ残す（`lss.bgm.current_best`）。
# Recipe IDは改行・BOMに依存しない（`.state/validation/bgm-product-repair-r2/`）。
# v13: 対応領域S3。番組の役割（端・放送挿入のタグ・中ほど・境界の付近）は文脈で、公開を止めない。編集境界で分け、
# 挟まれた挿入の区画は役割未確定の保留（音は残す）。CurrentBestはcueごと：旧cueの妥当性をそのcue自身の区間の
# 観測で先に決め、新しいrunが保留でも妥当な旧cueは残し、検証済みの改善だけが妥当な旧cueを置き換える
# （`.state/validation/bgm-product-repair-r3/`）。
_BGM_CANDIDATE_WORKFLOW = "episode-sfc-routed-cues-v13"
# CurrentBestへの採用の方針。runの中身（候補・判定・公開cueの音）は変えないので、Recipe IDとは別に版を持ち、
# CurrentBestの記録へ残す（方針が変わったら、反映済みの原本も推論なしで評価し直す）。
# v1（e633e5c）：重い非音楽イベントの場面・挿入の区画にある前の版のcueを INVALID として外していた。
# v2：妥当性と危険を分ける。INVALID はcue自身の音にある正の根拠（編集境界をまたぐ）だけ。重い非音楽イベントの
#     場面は品質の危険（risk）で、それだけでは前の版のcueを外さない（UNKNOWNのまま残す）。挿入の区画は役割の
#     危険で、CM・配信告知を出さないための保守的な方針として公開から外す（不正とは言わない。履歴に残る）。
#     （`.state/validation/bgm-product-repair-r3-heavy-unknown/`）
# v3：挿入の区画も危険だけ。区画の種類（CMか番組の挿入か）は実行時に決められず、番組の予告・コーナー・終了画面の
#     音楽も入る。区画にあるというだけで前から公開しているcueを外さない（UNKNOWN＋役割の危険として残す）。
#     新しい候補は従来どおり保留し、新しく公開はしない。方針の選び直しは、そのrunを反映する前の集まりから行う
#     （`.state/validation/bgm-product-repair-r4/`）。
# v4: 範囲の決まらない断片を、それを覆う範囲の決まったcueで置き換える（`current_best.EXTENT_IMPROVEMENT`）。
_BGM_ADOPTION_POLICY = "bgm-currentbest-adoption-v4"
_BGM_RECIPE_PREFIX = "bgm-episode-candidates-v13"
_BGM_IDENTITY_METHOD = "text-source-sha256-lf-normalized-v1"
# 解析器のadapterと同梱実装は、解析器の版（analyzer_id）の側で識別する。
_BGM_IMPLEMENTATION = ("bgm/episode_cues.py", "bgm/cue_boundaries.py", "bgm/cue_formation.py", "bgm/stereo_continuity.py", "bgm/recording_support.py", "bgm/occurrence_support.py",
                       "bgm/music_corroboration.py", "bgm/rendered_content.py",
                       "bgm/memory_support.py",
                       "bgm/episode_usage.py", "bgm/candidate_store.py",
                       "bgm/second_pass.py", "bgm/vocal_recovery.py", "bgm/cue_semantics.py",
                       "bgm/cue_publication.py", "bgm/cue_representation.py",
                       "bgm/dialogue_residual.py", "bgm/published_duplicates.py",
                       "bgm/cue_scope.py", "bgm/supported_scope.py", "bgm/cue_readiness.py", "bgm/short_cues.py",
                       "bgm/current_best.py")
# 参照録音との関係（`lss.bgm.reference_relations`）。参照録音が無い時は使わず、Recipe IDにも入れない（参照の無い
# runはr4と同じ識別子・同じ処理）。参照録音がある時だけ、使った参照の内容と、この方針の識別子をrunの識別子へ入れる
# （`_bgm_candidate_run_key`）。方針：区間を参照との関係（同じ録音の連続した抜粋・編集・追加の素材・別版の候補・未知）
# へ分け、一つの録音の一つのずれがMusic stemの音楽を説明し尽くす（台詞・効果音のstemの漏れは差し引く）抜粋（EXACT）
# だけを既発売として保留する（抽出の磨き直し・公開をしない。使用箇所は残す）。参照が鳴っているだけでは消さない。
_BGM_REFERENCE_INTEGRATION = "typed-reference-relations-music-accounting-exact-held-v4-clean-residual-edge-material"
_BGM_REFERENCE_IMPLEMENTATION = ("bgm/reference_relations.py", "bgm/identity.py", "bgm/versions.py",
                                 "bgm/reference_reconstruction.py", "bgm/release_availability.py")
_BGM_RUN_TYPE = "bgm_episode_candidate_product_run"
_BGM_STATUS_HELD = "completed_held_candidates"
_BGM_STATUS_NO_CANDIDATE = "completed_no_candidate"
_BGM_STATUS_ZERO = "completed_validated_zero_result"
_BGM_STATUS_PUBLISHED = "completed_published_cues"


class _BgmCandidateRunLock:
    """同じ決定的runを二重更新せず、後発要求を完了後の再利用へ合流させる。"""

    def __init__(self, product_root: Path, *, wait_seconds: float = 30 * 60) -> None:
        self._path = product_root / ".owner.lock"
        self._stream = None
        self._wait_seconds = wait_seconds

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._wait_seconds
        while True:
            stream = None
            try:
                # Windowsではbyte-range lock中のfileを開く段階がsharing violationに
                # なる構成もある。openとlockを同じ再試行単位にする。
                stream = self._path.open("a+b")
                stream.seek(0)
                if not stream.read(1):
                    stream.write(b"0")
                    stream.flush()
                    os.fsync(stream.fileno())
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:  # pragma: no cover - 製品はWindowsのみ
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if stream is not None:
                    stream.close()
                if time.monotonic() >= deadline:
                    raise RuntimeError("同じBGM候補処理の完了待ちが時間内に終わりませんでした") from exc
                time.sleep(0.1)
        self._stream = stream
        return self

    def __exit__(self, *_exc) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - 製品はWindowsのみ
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)
        except OSError:
            # closeでもOS所有権は解放される。保存済みの成功結果を解放処理だけで
            # FAILEDへ反転させず、次回の同一性検査に委ねる。
            pass
        finally:
            stream.close()


def _archive_outcome(path: Path, previous: bytes, payload: dict) -> bool:
    """変わる前の記録を内容hashの所在へ確定し、同じ内容の再開では増やさない。"""
    try:
        if json.loads(previous) == payload:
            return False
    except (TypeError, ValueError):
        pass
    digest = hashlib.sha256(previous).hexdigest()
    folder = path.parent / "outcome-history" / digest
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "application-outcome.json"
    temporary = None
    try:
        with NamedTemporaryFile(dir=folder, prefix=".outcome-history-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(previous)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != previous:
                raise ValueError("BGM保留履歴が同じ内容hashの記録と一致しません。現在の結果を保持します")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def _write_outcome_atomic(path: Path, payload: dict, validate_before_replace) -> None:
    """検証中の変更では、以前の製品結果を一時ファイルで置き換えない。"""
    temporary = None
    try:
        try:
            previous = path.read_bytes()
        except FileNotFoundError:
            previous = None
        with NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}-", suffix=".tmp",
            encoding="utf-8", newline="\n", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        validate_before_replace()
        if previous is not None:
            archived = _archive_outcome(path, previous, payload)
            # 履歴確定中の別の更新も、古い保存処理から保護する。
            if path.read_bytes() != previous:
                raise ValueError("BGM保留結果が別の処理で更新されました。現在の結果を保持します")
            if archived:
                validate_before_replace()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _resolve_stem_combination(request: ProcessingRequest) -> _ResolvedRequest | ProcessingOutcome:
    """組み合わせを作るmodelと経路を選ぶ。利用者はmodelを選ばない。

    通常利用できるmodel（既定が先）ごとに、出力パートで規則どおり作れるかを確かめ、
    足す部品が最も少ない経路を選ぶ（同数なら既定が先）。要求の中身が1本の出力と一致するmodelでは
    残りの出力の和ではなくその直接の出力を使い、それは単一ステムの既存経路そのものになる。
    出力の意味を宣言したmodel（``ledger/model_stem_semantics.tsv``）は名前の推測でなく宣言で読み、
    1つのpassで作れないときだけ、宣言済みの別々のpassの符号付き和（pass数の少ない順）を使う。
    """
    from lss.separation import stem_combination, stem_semantics

    try:
        rows = catalog.list_models()
        semantics = stem_semantics.load_semantics()
    except Exception as exc:
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED,
            error_code="capability_resolution_failed",
            message=str(exc),
        )
    candidates = []
    reasons = []
    for row in rows:
        try:
            candidates.append(stem_combination.PassCandidate(
                model_id=row["model_id"], expected_stems=tuple(catalog.stem_targets(row)),
                semantics=semantics.get(row["model_id"]),
            ))
        except Exception as exc:
            # 設定を取れないmodelは選ばない。他のmodelで作れるならそれを使う。
            reasons.append(f"{row.get('model_id')}: {exc}")
    try:
        model_id, plan = stem_combination.choose_route(request.stems, candidates)
    except stem_combination.StemCombinationError as exc:
        reasons.append(str(exc))
        return ProcessingOutcome(
            disposition=ResultDisposition.UNSUPPORTED,
            error_code="stem_combination_unsupported",
            message=("この組み合わせを作れる分離modelがありません: "
                     + " / ".join(reasons or ["利用できるmodelがありません"])),
        )
    record = {**plan.to_dict(), "model_id": model_id}
    if plan.route == "direct_stem":
        return _ResolvedRequest(model_id=model_id, stem_target=plan.components[0],
                                stem_combination=record)
    return _ResolvedRequest(model_id=model_id, stem_target=stem_combination.target_key(request.stems),
                            stem_combination=record)


def _resolve(request: ProcessingRequest) -> _ResolvedRequest | ProcessingOutcome:
    if request.intent == UserIntent.EXTRACT_STEM_COMBINATION:
        return _resolve_stem_combination(request)
    if request.intent != UserIntent.EXTRACT_ACCOMPANIMENT:
        return ProcessingOutcome(
            disposition=ResultDisposition.UNSUPPORTED,
            error_code="intent_unsupported",
            message=f"未対応の処理要求です: {request.intent}",
        )
    try:
        row = catalog.default_model()
        stem_target = catalog.requested_stem(row)
    except Exception as exc:
        # 対応済みの製品要求を、設定・台帳・取得の失敗からUNSUPPORTEDへ読み替えない。
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED,
            error_code="capability_resolution_failed",
            message=str(exc),
        )
    return _ResolvedRequest(model_id=row["model_id"], stem_target=stem_target)


def _acquire_inputs(
    urls: tuple[str, ...], acquirer: Callable[[str], Iterable[EntryOutcome]],
) -> tuple[AcquiredInput, ...]:
    from lss.acquisition import store

    results = []
    for url in urls:
        try:
            entries = list(acquirer(url))
            if not entries:
                raise store.AcquisitionFailure(
                    store.FailureKind.UNAVAILABLE, "URLから取得できる素材がありません",
                )
        except Exception as exc:
            # 取得済み素材と他の入力は保つ。台帳・ログイン状態の読取失敗も取得失敗として返す。
            results.append(AcquiredInput(
                url=store.sanitize_url(url),
                error_code=(exc.kind.value if isinstance(exc, store.AcquisitionFailure)
                            else "acquisition_failed"),
                message=store.sanitize_text(str(exc)),
                retryable=isinstance(exc, store.AcquisitionFailure) and exc.retryable,
            ))
            continue
        for entry in entries:
            source = entry.source
            failure = entry.failure
            results.append(AcquiredInput(
                url=store.sanitize_url(entry.requested_url),
                source_path=source.path if source is not None else None,
                source_sha256=source.sha256 if source is not None else None,
                reused=source.reused if source is not None else False,
                offline_reuse=entry.offline_reuse,
                error_code=failure.kind.value if failure is not None else None,
                message=str(failure) if failure is not None else None,
                retryable=failure.retryable if failure is not None else False,
            ))
    return tuple(results)


def submit(
    request: ProcessingRequest, queue: jobs.SeparationQueue, *,
    acquirer: Callable[[str], Iterable[EntryOutcome]] | None = None,
    bgm_runner: Callable[[ProcessingRequest], ProcessingOutcome] | None = None,
) -> Submission:
    """製品要求を対応する固定経路へ接続する。

    URL取得は同期処理なので、画面からは作業スレッドで呼ぶ。省略時は既存の取得serviceが
    LSS専用のログイン状態を使う。認証・進捗・停止の接続には、そのserviceの引数を固定した
    acquirerを渡せる。ローカル入力は取得serviceを呼ばない。単話BGM候補生成も同期処理で、
    伴奏用queueへ登録しない。
    """
    if request.intent == UserIntent.EXTRACT_BGM_CUES:
        if request.urls:
            return Submission(
                intent=request.intent,
                outcome=ProcessingOutcome(
                    disposition=ResultDisposition.UNSUPPORTED,
                    error_code="bgm_requires_single_local_source",
                    message="BGM候補はローカルにある完成済み番組を1本ずつ処理します",
                ),
            )
        if bgm_runner is None:
            bgm_runner = run_bgm_candidate_request
        try:
            outcome = bgm_runner(request)
        except Exception as exc:
            try:
                from lss.acquisition.store import sanitize_text
                message = sanitize_text(str(exc))
            except Exception:
                message = str(exc)
            outcome = ProcessingOutcome(
                disposition=ResultDisposition.FAILED,
                error_code="bgm_candidate_processing_failed",
                message=message,
            )
        if not isinstance(outcome, ProcessingOutcome):
            outcome = ProcessingOutcome(
                disposition=ResultDisposition.FAILED,
                error_code="invalid_bgm_candidate_response",
                message="BGM候補処理が不正な結果を返しました",
            )
        return Submission(intent=request.intent, outcome=outcome)

    resolved = _resolve(request)
    if isinstance(resolved, ProcessingOutcome):
        return Submission(intent=request.intent, outcome=resolved)
    acquisitions = ()
    if request.urls:
        if acquirer is None:
            from lss.acquisition.service import acquire
            acquirer = acquire
        acquisitions = _acquire_inputs(request.urls, acquirer)
    paths = list(request.sources) + [
        entry.source_path for entry in acquisitions if entry.source_path is not None
    ]
    if not paths:
        return Submission(
            intent=request.intent, stem_combination=resolved.stem_combination, acquisitions=acquisitions,
            outcome=ProcessingOutcome(
                disposition=ResultDisposition.FAILED,
                error_code="acquisition_failed",
                message="入力の取得に失敗しました。項目ごとの取得結果を確認してください",
            ),
        )
    try:
        item_ids = tuple(queue.add(
            paths, model_id=resolved.model_id, stem_target=resolved.stem_target,
            include_existing=True,
        ))
    except Exception as exc:
        from lss.acquisition.store import sanitize_text
        return Submission(
            intent=request.intent, stem_combination=resolved.stem_combination, acquisitions=acquisitions,
            outcome=ProcessingOutcome(
                disposition=ResultDisposition.FAILED,
                error_code="queue_registration_failed", message=sanitize_text(str(exc)),
            ),
        )
    if not item_ids:
        return Submission(
            intent=request.intent, stem_combination=resolved.stem_combination, acquisitions=acquisitions,
            outcome=ProcessingOutcome(
                disposition=ResultDisposition.FAILED,
                error_code="no_supported_input",
                message="処理できる入力ファイルがありません",
            ),
        )
    return Submission(intent=request.intent, stem_combination=resolved.stem_combination,
                      queue_item_ids=item_ids, acquisitions=acquisitions)


def _bgm_implementation_sources(root: Path | None = None) -> dict[str, str]:
    """候補の中身を決める実装の内容hash。凍結時も同じ相対位置にsourceを同梱する。

    v12から改行・BOMの違いで変わらないhash（`runtime.text_source_sha256`）。checkoutの改行変換で
    同じ実装が別Recipeにならない。v11以前のRecipe IDは生のbytesのhashで、記録どおりに読める。
    """
    from lss.core import runtime

    root = Path(root) if root is not None else Path(__file__).resolve().parent
    return {name: runtime.text_source_sha256(root / name) for name in _BGM_IMPLEMENTATION}


def _bgm_recipe_identity(body: dict) -> dict:
    """Recipe IDへ入れる本文。解析器の実装は改行に依存しない内容hashへ置き換え、生bytes由来の
    解析器の版（analyzer_id）は除く（解析器の版は受領票の照合にだけ使う）。それ以外は本文と同じ。"""
    from lss.bgm import event_tagging, four_stem_analysis, sfc_analysis

    identity = json.loads(json.dumps(body))
    for key, module in (("analyzer", sfc_analysis), ("event_tagger", event_tagging),
                        ("four_stem_analyzer", four_stem_analysis)):
        identity[key].pop("analyzer_id", None)
        identity[key]["implementation"] = [{"path": path, "sha256": digest} for path, digest
                                           in module.implementation_artifacts(normalize_text=True)]
        identity.pop(f"{key}_id", None)
    return identity


def _bgm_candidate_recipe() -> dict:
    from lss.bgm import (cue_publication, cue_representation, cue_scope, cue_semantics, current_best,
                         dialogue_residual, episode_cues, event_tagging, four_stem_analysis, published_duplicates,
                         second_pass, sfc_analysis, vocal_recovery)
    from lss.core import catalog
    from lss.separation import selective
    from lss.bgm import cue_formation, memory_support, music_corroboration, rendered_content
    from lss.bgm import gpu_execution

    cue_recipe = episode_cues.default_recipe()
    scope = cue_scope.recipe()
    duplicates = published_duplicates.default_recipe()
    semantics = cue_semantics.default_recipe()
    tagger = event_tagging.analyzer_revision()
    # Development-only execution profiles are explicit recipe inputs. An unset variable keeps the
    # legacy analyzer identity; an invalid value fails closed rather than silently reusing legacy work.
    resolved_profile = gpu_execution.resolve_execution_profile(os.environ.get("LSS_DEV_SFC_EXECUTION_PROFILE"))
    execution_profile = resolved_profile.name if resolved_profile is not None else None
    analyzer = sfc_analysis.analyzer_revision(execution_profile=execution_profile)
    refinement = second_pass.default_recipe()
    vocal = vocal_recovery.default_recipe()
    four_stem = four_stem_analysis.analyzer_revision(vocal.hop_seconds)
    dialogue = dialogue_residual.default_recipe()
    instrumental = catalog.default_model()
    body = {
        "workflow": _BGM_CANDIDATE_WORKFLOW,
        "cue_recipe": cue_recipe.to_dict(), "cue_recipe_id": cue_recipe.recipe_id,
        "second_pass": refinement.to_dict(), "second_pass_id": refinement.recipe_id,
        "vocal_recovery": vocal.to_dict(), "vocal_recovery_id": vocal.recipe_id,
        "four_stem_analyzer": four_stem, "four_stem_analyzer_id": four_stem["analyzer_id"],
        "analyzer": analyzer, "analyzer_id": analyzer["analyzer_id"],
        "event_tagger": tagger, "event_tagger_id": tagger["analyzer_id"],
        "cue_semantics": semantics.to_dict(), "cue_semantics_id": semantics.recipe_id,
        "cue_formation": cue_formation.revision(),
        "memory_support": memory_support.revision(),
        "music_corroboration": music_corroboration.revision(),
        "rendered_content_recovery": rendered_content.revision(),
        "cue_representation": cue_representation.default_recipe().to_dict(),
        "cue_representation_id": cue_representation.default_recipe().recipe_id,
        "representations": cue_representation.REPRESENTATIONS,
        "refinement_scope": "held_candidates_of_cues_published_as_processed",
        "dialogue_residual": dialogue.to_dict(), "dialogue_residual_id": dialogue.recipe_id,
        # 伴奏の既定modelが替われば別Recipe（実体の同一性は受領票のsha256で確かめる）。
        "dialogue_residual_model": {"model_id": instrumental["model_id"], "arch": instrumental["arch"],
                                    "ledger_sha256": instrumental.get("sha256") or None},
        # cue平均Speech>=gateのcueはcue全体、それ未満でSpeech窓のあるcueはその窓の時間だけ（v10）。
        "dialogue_residual_scope": "regional_speech_only_with_musical_voice_protection_v3",
        "selective_composition": selective.revision(),
        "selective_nomination": "first_and_selected_music_and_source; processed_speech_regions_veto_raw",
        "selective_source_speech_gate": .2,
        "publication": {"policy": cue_publication.PUBLICATION_POLICY, "format": "FLAC PCM_24",
                        "range": "declared_power_of_two_headroom", "folder": "<source folder>/LSS/<source stem> - BGM",
                        "published_class": cue_semantics.WANTED,
                        "unresolved_extent_mark": cue_publication.UNRESOLVED_EXTENT_MARK},
        # 公開・保留・対象外を決める対応領域（S2）。意味判定の区間を写し、安全の観測を加えて最終判断する。
        "supported_scope": scope, "supported_scope_id": scope["scope_id"],
        # 同じ状態領域で先に公開した別の原本のcueと同じ組み立て済み音声は、重複として公開しない。
        # 使った先行runはrunの記録に残す（runの識別子には入れない。処理済みの話が増えても再利用を壊さない）。
        "published_duplicates": duplicates.to_dict(), "published_duplicates_id": duplicates.recipe_id,
        "published_duplicates_scope": "completed_published_runs_of_other_sources_in_same_state",
        "implementation": _bgm_implementation_sources(),
        # Recipe IDは改行・BOMの違いで変わらない（`_bgm_recipe_identity`）。
        "identity_method": _BGM_IDENTITY_METHOD,
        # 同じ原本の前の版の結果を、このRecipeの検証済み結果が置き換える範囲（`lss.bgm.current_best`）。
        "supersedes": current_best.supersession_declaration(),
        # 登録済みの参照録音の集まりがあれば既発売の同定に使う（無ければ使わない。不一致は根拠にしない）。
        # 使った参照の内容はrunの識別子へ入れる（`_bgm_candidate_run_key`）。
        "known_recording_identity": "registered_reference_sets_when_present",
    }
    canonical = json.dumps(_bgm_recipe_identity(body), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**body, "recipe_id": f"{_BGM_RECIPE_PREFIX}-"
            + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]}


def _bgm_reference_policy() -> dict:
    """参照録音を使う時の方針の宣言と識別子（関係の判定の宣言・統合の版・実装の内容hash）。"""
    from lss.bgm import reference_relations
    from lss.core import runtime

    relations = reference_relations.default_recipe()
    root = Path(__file__).resolve().parent
    body = {"integration": _BGM_REFERENCE_INTEGRATION,
            "reference_relations": relations.to_dict(), "reference_relations_id": relations.recipe_id,
            "implementation": {name: runtime.text_source_sha256(root / name)
                               for name in _BGM_REFERENCE_IMPLEMENTATION}}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**body, "policy_id": "bgm-reference-policy-v2-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]}


def _bgm_candidate_run_key(source_sha256: str, recipe_id: str, references_digest: str | None = None,
                           *, memory_refresh: str | None = None) -> str:
    body = {"source_sha256": source_sha256, "recipe_id": recipe_id}
    if references_digest is not None:
        # 参照録音が無い状態のrunは、参照の宣言を持たない識別子のまま（同じ素材の再利用を変えない）。
        # 参照がある時は、使った参照の内容と、参照との関係の方針の識別子を入れる（方針が変われば別のrun）。
        body["references"] = references_digest
        body["reference_policy"] = _bgm_reference_policy()["policy_id"]
    if memory_refresh is not None:
        body["memory_refresh"] = memory_refresh
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _registered_references(state_root: Path | None, *, source: Path | None = None,
                           source_sha256: str | None = None, ingest: bool = True) -> tuple[list, str | None, list[dict], dict]:
    """使える参照録音：状態領域に登録された集まり（全部）と、参照録音の置き場所（`references.library_root`）の音声、
    およびLSSがこの番組の作品のために取得して確かめた参照（`reference_store`、作品ごと）。

    内容SHA-256で確かめた所在だけを、内容ごとに1つ使う（内容SHA-256の順）。見つからない録音は使わず記録に残す。
    置き場所を読めないことは処理を止める理由にしない（その分は参照が無いのと同じ）。取得した参照は作品に
    結び付いたものだけを使う（別の作品の取得でこの番組のrunの識別子を変えない）。重い音源が退役していても、
    参照の知識（指紋）があれば集まりに入る（波形の確認が要る時に取り直す）。
    戻り値は（参照の知識の記録（`ingest=False` なら内容SHA-256）, 内容の集まりのdigest（無ければNone）,
    見つからない録音, 出所の要約）。
    """
    from lss.bgm import reference_acquisition, reference_store, references

    root = Path(state_root) if state_root is not None else session.state_dir()
    found: dict[str, Path] = {}
    missing: list[dict] = []
    registered = 0
    folder = root / references.DIRECTORY
    if folder.is_dir():
        for record in sorted(folder.glob("*.json")):
            resolution = references.resolve(record.stem, state_root=root)
            # `resolve` は見つけた所在と、その内容SHA-256（出所の辞書のキー）を同じ順で入れる。
            for path, sha256 in zip(resolution.paths, resolution.origins):
                found.setdefault(sha256, path)
                registered += 1
            missing.extend(resolution.missing)
    try:
        scan = references.scan_library(root)
        library = scan.summary()
        for path in scan.paths:
            found.setdefault(scan.sha256[str(path)], path)
    except OSError as exc:
        library = {"policy": references.LIBRARY_POLICY, "root": str(references.library_root()), "count": 0,
                   "error": type(exc).__name__}
    acquired: list[dict] = []
    work = None
    if source is not None:
        try:
            work = reference_acquisition.work_hint(source, source_sha256, root)
            if work is not None:
                acquired = [record for record in reference_store.records(root, work_key=work["work_key"])
                            if record["origin"] == reference_store.ORIGIN_ACQUIRED and record["sha256"] not in found]
        except (OSError, ValueError):
            acquired = []
    sources = {"registered_sets": registered, "library": library,
               "acquired_for_work": len(acquired), "work": work}
    shas = sorted(set(found) | {record["sha256"] for record in acquired})
    if not shas:
        return [], None, missing, sources
    digest = hashlib.sha256(json.dumps(shas).encode("utf-8")).hexdigest()
    if not ingest:
        return shas, digest, missing, sources
    entries = reference_store.sync_user_media([(found[sha], sha) for sha in sorted(found)],
                                              cache_dir=root / "bgm-reference-fingerprints", state_root=root)
    return sorted(entries + acquired, key=lambda record: record["sha256"]), digest, missing, sources


def _reference_preparation_seconds() -> float:
    """番組の判定に使う参照が決まるまで待つ上限（秒）。既定は4分。"""
    try:
        return max(0.0, float(os.environ.get("LSS_REFERENCE_PREPARATION_SECONDS", "240")))
    except ValueError:
        return 240.0


def _ensure_bgm_candidate_recipe(path: Path, recipe: dict) -> None:
    """固定Recipeを競合時にも別内容で上書きせず、完全なfileとして確定する。"""
    encoded = (json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
               + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = path.read_bytes()
    except FileNotFoundError:
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix=".candidate-recipe-", suffix=".tmp",
                                    delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        current = path.read_bytes()
    if current != encoded:
        raise ValueError("同じBGM候補Recipeの保存先に別内容があります")


def _verified_bgm_run(run_path: Path, source_sha256: str, recipe_id: str) -> dict:
    """完了記録と、その記録が指す候補音声を読み直して確かめる。"""
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if (run.get("type") != _BGM_RUN_TYPE or run.get("schema_version") != 1
            or (run.get("source") or {}).get("sha256") != source_sha256
            or run.get("recipe_id") != recipe_id):
        raise ValueError("BGM候補runの原本同一性またはRecipeが一致しません")
    status = run.get("status")
    if status == _BGM_STATUS_ZERO:
        session.validate_zero_result(run.get("zero_result"))
        return run
    if status not in {_BGM_STATUS_HELD, _BGM_STATUS_NO_CANDIDATE, _BGM_STATUS_PUBLISHED}:
        raise ValueError("BGM候補runが完了状態ではありません")
    run_dir = run_path.parent.resolve()
    record = run.get("cue_record") or {}
    record_path = (run_dir / str(record.get("name", ""))).resolve()
    if record_path.parent != run_dir or file_sha256(record_path) != record.get("sha256"):
        raise ValueError("BGM候補の検出根拠が完了記録と一致しません")
    candidates_root = (run_dir / "candidates").resolve()
    cues = run.get("held")
    if not isinstance(cues, list) or (status in {_BGM_STATUS_HELD, _BGM_STATUS_PUBLISHED}) != bool(cues):
        raise ValueError("BGM候補の完了記録と候補区間数が一致しません")
    semantics = run.get("semantics")
    if cues:
        # 意味判定の記録と、その判定に使った窓ごとの観測（タグ・埋め込み・音楽の大きさ）を確かめる。
        if not isinstance(semantics, dict):
            raise ValueError("BGM候補の意味判定の記録がありません")
        semantics_dir = (run_dir / "semantics").resolve()
        for item in [semantics.get("record") or {}, *(semantics.get("evidence") or {}).values()]:
            path = (semantics_dir / str(item.get("name", ""))).resolve()
            if path.parent != semantics_dir or file_sha256(path) != item.get("sha256"):
                raise ValueError("BGM候補の意味判定の根拠が完了記録と一致しません")
        published = run.get("published")
        if not isinstance(published, list) or (status == _BGM_STATUS_PUBLISHED) != bool(published):
            raise ValueError("BGM候補の完了記録と公開cue数が一致しません")
    identifiers: set[str] = set()
    for cue in cues:
        start, end = cue.get("start_seconds"), cue.get("end_seconds")
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not 0 <= start < end):
            raise ValueError("BGM候補に出典時刻がありません")
        for name in ("music", "raw", "music_first_pass", "music_before_vocal_recovery"):
            if name in ("music_first_pass", "music_before_vocal_recovery") and name not in cue:
                continue
            artifact = cue.get(name) or {}
            path = Path(str(artifact.get("path", ""))).resolve()
            identifier = artifact.get("candidate_asset_id")
            if (not path.is_relative_to(candidates_root) or not isinstance(identifier, str)
                    or identifier in identifiers or file_sha256(path) != artifact.get("sha256")):
                raise ValueError("BGM候補音声の同一性または保存範囲が不正です")
            identifiers.add(identifier)
    return run


def _retire_unverifiable_run(run_path: Path) -> None:
    """検証できない完了記録を診断用に退避する（``invalid-runs/<記録のsha256>.json``）。候補には触れない。

    記録を外した後のrunは完了記録の無い試行と同じに扱い、候補の片付けは `_discard_interrupted_attempt` が
    所有と参照を確かめて行う。退避した記録が挙げる候補は消さずに ``preserved/`` へ移し、LSSの物と確かめられない
    物（利用者の物・別の原本の物・読めない物）には触れない。記録の証拠が欠けていることは、候補フォルダの物が
    LSSの物だという根拠にならない。
    """
    previous = run_path.read_bytes()
    archive = run_path.parent / "invalid-runs"
    target = archive / f"{hashlib.sha256(previous).hexdigest()}.json"
    if not target.exists():
        archive.mkdir(parents=True, exist_ok=True)
        temporary = archive / f".{target.name}.tmp"
        temporary.write_bytes(previous)
        os.replace(temporary, target)
    run_path.unlink()


def _candidate_of(path: Path, source_sha256: str) -> str | None:
    """LSSがこの原本から書いた非公開の候補なら、その候補ID。確かめられなければNone。"""
    from mutagen import MutagenError
    from mutagen.flac import FLAC

    try:
        tags = FLAC(path)
    except (MutagenError, OSError, ValueError):
        return None
    identifiers = tags.get("LSS_CANDIDATE_ID") or []
    if (tags.get("LSS_SOURCE_TYPE") != ["bgm_candidate"] or tags.get("LSS_INPUT_SHA256") != [source_sha256]
            or len(identifiers) != 1 or not identifiers[0].startswith("bgm-candidate-v")):
        return None
    return identifiers[0]


def _discard_interrupted_attempt(run_dir: Path, source_sha256: str, state_root: Path) -> None:
    """完了記録の無いrunに残った、前の試行の候補を片付ける（終了・停電等で途中で止まった）。

    完了記録（run.json）の無い試行の候補は、同じRecipeの再実行で作り直せる非公開の内部成果。解析をやり直すと
    同じ名前の別の候補になり得る（GPU推論は実行ごとに僅かに揺らぎ、FLOATのWAVは書いた時刻も持つ）ので、残すと
    次の保存が「保存先に別の候補があります」で止まり続ける。

    フォルダの名前や完了記録が無いことだけを根拠に消さない。消すのは、LSSがこの原本から書いたとtagで確かめられ
    （種類・候補ID・原本）、どの記録からも参照されない候補だけ。退避した完了記録（invalid-runs）や原本の
    CurrentBestの記録が名前か候補IDを挙げる候補は、消さずに ``preserved/`` へ移す。それ以外（利用者の物・
    別の原本の物・読めない物）には触れない。LSSの一時名に見える物（``.lss-candidate-*.tmp``）も、名前や先頭の
    bytesはLSSが書いた証拠にならないので触れない（書きかけは毎回別の一時名で、残っても次の保存を妨げない）。
    行ったことは ``interrupted-attempts/`` へ受領票として残す。
    """
    from lss.bgm import current_best

    candidates_root = run_dir / "candidates"
    if (run_dir / "run.json").exists() or not candidates_root.is_dir():
        return
    references = []
    for record in [*sorted((run_dir / "invalid-runs").glob("*.json")),
                   current_best.record_path(state_root, source_sha256)]:
        try:
            references.append(record.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            continue
    referenced = "\n".join(references)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"-{os.getpid()}"
    receipt = {"type": "bgm_interrupted_attempt_cleanup", "schema_version": 1, "run_dir": str(run_dir),
               "source_sha256": source_sha256, "reason": "no completed run record (run.json)",
               "removed": [], "preserved": [], "untouched": []}
    for path in sorted(candidates_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(candidates_root).as_posix()
        identifier = _candidate_of(path, source_sha256) if path.suffix.lower() == ".flac" else None
        if identifier is None:
            receipt["untouched"].append(relative)
            continue
        row = {"path": relative, "candidate_id": identifier, "sha256": file_sha256(path)}
        if identifier in referenced or path.name in referenced:
            target = run_dir / "preserved" / stamp / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
            receipt["preserved"].append({**row, "moved_to": target.relative_to(run_dir).as_posix()})
        else:
            path.unlink()
            receipt["removed"].append({**row, "kind": "unreferenced_candidate"})
    if receipt["removed"] or receipt["preserved"]:
        session._write_json_atomic(run_dir / "interrupted-attempts" / f"{stamp}.json", receipt)


def _is_staged_cue_of(path: Path, source_sha256: str) -> bool:
    from mutagen import MutagenError
    from mutagen.flac import FLAC

    from lss.bgm import cue_publication

    try:
        tags = FLAC(path)
    except (MutagenError, OSError, ValueError):
        return False
    return (tags.get("LSS_SOURCE_TYPE") == [cue_publication.SOURCE_TYPE]
            and tags.get("LSS_INPUT_SHA256") == [source_sha256])


def _discard_unpublished_staging(staging_dir: Path, source_sha256: str, state_root: Path) -> None:
    """公開前の作業領域に残った、この原本の音を消す（試行が公開の前で止まった）。

    作業領域の音はCurrentBestの切替でだけ公開名へ移る。切替の記録（journal）が無い間、ここに残る音は
    どの結果でもない。LSSが書いたこの原本の公開cue（tagで確かめる）だけを消す。別の原本の音・それ以外の
    ファイル（LSSの一時名に見える ``.lss-bgm-*.tmp`` 等も。名前や先頭のbytesは所有の証拠にならない）には
    触れない。切替の途中（journalあり）なら何もしない。
    """
    from lss.bgm import current_best

    if current_best.journal_path(state_root, source_sha256).is_file():
        return
    root = staging_dir.parent
    if not root.is_dir():
        return
    for folder in [staging_dir, *sorted(path for path in root.iterdir() if path.is_dir() and path != staging_dir)]:
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() == ".flac" and _is_staged_cue_of(path, source_sha256):
                path.unlink(missing_ok=True)
        try:
            folder.rmdir()  # 空になった作業領域だけ
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def _bgm_outcome_base(run: dict, run_path: Path, *, reused: bool,
                 outputs: tuple[Path, ...] | None = None) -> ProcessingOutcome:
    """runの結論を画面へ返す。``outputs`` は原本のCurrentBestの公開中のファイル（前の版のcueを含みうる）。"""
    if run["status"] == _BGM_STATUS_ZERO and not outputs:
        return ProcessingOutcome(
            disposition=ResultDisposition.ZERO, evidence_path=run_path, record_path=run_path,
            recipe_id=run["recipe_id"], zero_result=run["zero_result"], reused=reused,
            message="検証済みの対象なし結果です",
        )
    held = tuple(Path(cue["music"]["path"]) for cue in run["held"])
    # 原本の区間、2回目を採った候補の1回目、歌唱を足した候補の足す前。採らなかった別の表現として残す。
    alternatives = tuple(Path(cue[name]["path"]) for cue in run["held"]
                         for name in ("raw", "music_first_pass", "music_before_vocal_recovery") if name in cue)
    own = [item for item in run.get("published") or [] if item.get("current_best") in (None, "placed")]
    published = outputs if outputs is not None else tuple(Path(item["path"]) for item in own)
    # 参照録音と同じ録音の連続した抜粋（既発売）として抽出・公開しなかった区間。使用箇所として記録してある。
    known = int(((run.get("semantics") or {}).get("counts") or {}).get("KNOWN_RELEASED_MUSIC", 0))
    known_note = f"（手元の既発売録音と同じ区間{known}件は、抽出せず使用箇所として記録しました）" if known else ""
    if published:
        unresolved = sum(1 for item in own if (item.get("cue_extent") or {}).get("state") == "unresolved"
                         and Path(item["path"]) in set(published))
        earlier = len(published) - sum(1 for item in own if Path(item["path"]) in set(published))
        # 番組だけの版（編集・成分の違う版）を、手に入る既発売録音の領域で組み立て直したcue。
        rebuilt = sum(1 for item in own if item.get("representation") == "reference_donor_reconstruction"
                      and Path(item["path"]) in set(published))
        evidence = (run_path.parent / "semantics" / run["semantics"]["record"]["name"]
                    if run.get("semantics") else run_path.parent / run["cue_record"]["name"])
        return ProcessingOutcome(
            disposition=ResultDisposition.PUBLISHED, outputs=published, held_artifacts=held,
            held_alternative_artifacts=alternatives,
            evidence_path=evidence, record_path=run_path,
            recipe_id=run["recipe_id"], reused=reused,
            message=f"欲しい劇伴と判定したcueを{len(published)}件保存しています"
                    + (f"（うち{earlier}件は前の版のまま残したcue）" if earlier > 0 else "")
                    + (f"（うち{unresolved}件は一曲かどうか未確定）" if unresolved else "")
                    + (f"（うち{rebuilt}件は番組だけの版を既発売録音で組み立て直したcue）" if rebuilt else "")
                    + f"（判定を保留した区間{_bgm_held_count(run)}件は非公開のまま残しています）" + known_note,
        )
    return ProcessingOutcome(
        disposition=ResultDisposition.HELD, held_artifacts=held, held_alternative_artifacts=alternatives,
        evidence_path=run_path.parent / run["cue_record"]["name"], record_path=run_path,
        recipe_id=run["recipe_id"], reused=reused,
        message=(f"公開の条件（意味・安全・完全性）を満たさないBGM候補を{len(held)}件保留しました{known_note}" if held else
                 "候補区間は見つかりませんでしたが、対象なしとは確定していません"),
    )


def _accepted_run_item(row: dict) -> dict | None:
    """CurrentBestの行が指す採用runの公開項目（同じcue・同じ音）。runが読めない・その音をもう持たない時はNone。

    Noneは「runの記録から射影できない」であって不正ではない。復元したCurrentBestの行は、採用した後に書き換えられた
    runを指すことがある。その場合も公開ファイル自身のtagが採用時の同一性を持つ。"""
    try:
        authority = json.loads(Path(row["run_path"]).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return next((item for item in authority.get("published") or []
                 if item.get("key") == row.get("cue_key")
                 and (item.get("audio") or {}).get("pcm_sha256") == row.get("pcm_sha256")), None)


def _bgm_cue_memory(state_root: Path, source_sha256: str, outputs, accepted_by_path: dict) -> dict:
    """Register this source's published cues in the published-cue memory and read their recurrence.

    Memory is an optional amplifier of presentation: if it cannot be read or compared, the cues are finished
    exactly as without it and the projection says the check did not happen.
    """
    from lss.bgm import cue_memory

    items, unreadable = [], 0
    for path in outputs:
        try:
            items.append(cue_memory.published_item(Path(path), accepted=accepted_by_path.get(str(path))))
        except Exception:  # noqa: BLE001 - one unreadable cue is not compared; the others still are
            unreadable += 1
    try:
        found = cue_memory.observe(state_root, source_sha256, items)
    except Exception as exc:  # noqa: BLE001 - memory never decides or blocks publication
        return {"state": "not_checked", "error": type(exc).__name__, "by_path": {}, "changed_elsewhere": []}
    return {**found, "unreadable": unreadable}


def _refresh_recurrence_elsewhere(state_root: Path, changed: list[dict]) -> dict:
    """Refresh the presentation of other sources' cues whose recurrence changed (title number, shared name).

    Only a file still carrying the PCM the memory registered is touched, through the same listening finisher
    (LSS rewrites its own complete tag set). A file that cannot be refreshed now keeps its title; its own next run applies it.
    """
    from lss.bgm import current_best, listening, write_targets

    done, pending = 0, []
    for item in changed:
        try:
            # The target is the other source's CurrentBest public_path; the path the memory remembers is only an
            # observation (it may name a library the state was copied from) and is never written.
            row = current_best.published_files(state_root, item["source_sha256"]).get(
                str(item["asset_id"]).removeprefix("bgm-cue:")) if item.get("source_sha256") else None
            if row is None or row.get("pcm_sha256") != item["pcm_sha256"]:
                pending.append({"path": item["path"], "error": "not_a_current_published_path"})
                continue
            try:
                target = write_targets.checked(state_root, row, str(item["asset_id"]).removeprefix("bgm-cue:"))
            except write_targets.WriteTargetRefused as refused:
                pending.append({"path": item["path"], "error": "not_a_current_published_path",
                                "reason": refused.reason})
                continue
            listening.finish(target, state_root, expected_pcm=item["pcm_sha256"], recurrence=item["view"])
            done += 1
        except Exception as exc:  # noqa: BLE001 - another source's presentation is never this run's failure
            pending.append({"path": item["path"], "error": type(exc).__name__})
    return {"refreshed": done, "pending": pending}


def _bgm_recurrence_assets(state_root: Path) -> dict:
    """Build or refresh one consensus asset per recurring group (`lss.bgm.recurrence_asset`).

    An additional library file only: episode cues and their CurrentBest are untouched, and a failure here leaves
    the result exactly as without the asset (the projection says it was not checked).
    """
    from lss.bgm import recurrence_asset

    try:
        return recurrence_asset.update(state_root)
    except Exception as exc:  # noqa: BLE001 - the asset never decides or blocks the episode's publication
        return {"state": "not_checked", "error": type(exc).__name__, "assets": []}


def _bgm_library(source: Path | None, state_root: Path) -> dict | None:
    """The series album LSS acquires for this source (`lss.metadata.series_album`); None leaves the episode's own
    collection presentation. Metadata never blocks or fails a result."""
    if source is None:
        return None
    try:
        from lss.metadata import series_album

        return series_album.for_source(Path(source), state_root)
    except Exception:  # noqa: BLE001 - metadata is presentation only
        return None


def _file_stamp(path: Path) -> tuple[str, int, int, int, int] | None:
    """What any rewrite of a file changes: size, mtime, change time and file id (None if it cannot be read)."""
    try:
        path = Path(path).resolve()
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def _bgm_outcome(run: dict, run_path: Path, *, reused: bool,
                 outputs: tuple[Path, ...] | None = None, source: Path | None = None,
                 operation: str | None = None, rebuilt=()) -> ProcessingOutcome:
    from lss.bgm import collection_metadata, current_best, listening
    from lss.output import results

    root = run_path.parents[3]
    record = current_best.read(root, run["source"]["sha256"])
    basis = _bgm_presentation_basis(run, run_path, record)
    basis_run, basis_path = basis if basis is not None else (None, None)
    outcome = _bgm_outcome_base(*(basis or (run, run_path)), reused=reused, outputs=outputs)
    accepted_by_path = {}
    for row in ((record or {}).get("current") or {}).get("files") or []:
        item = _accepted_run_item(row)
        if item is None:
            # The run that adopted this audio no longer lists it (a restored/older CurrentBest whose run record was
            # rewritten since). The file itself carries the accepted identity; listening reads it from its own tags.
            continue
        accepted_by_path[row["public_path"]] = {**item, "input_sha256": run["source"]["sha256"],
                                               "recipe_id": row["recipe_id"],
                                               "album": collection_metadata.album(source) if source else item["album"]}
    verified_stamps = set()

    def validate_source():
        # Runs before every finished output and once after: re-read the whole source only when its stamp changed
        # since a read that matched (a rewrite changes the size, mtime or change time), like
        # `candidate_store.signal_sha256`. Each call re-read the full episode before (21 times for 20 cues).
        if source is None:
            return
        stamp = _file_stamp(source)
        if stamp is not None and stamp in verified_stamps:
            return
        if file_sha256(source) != run["source"]["sha256"]:
            raise session.InputChangedError("聴取用仕上げの間に原本が変更されました")
        if stamp is not None and _file_stamp(source) == stamp:
            verified_stamps.add(stamp)
    memory = _bgm_cue_memory(root, run["source"]["sha256"], outcome.outputs, accepted_by_path)
    finished = listening.finish_outputs(outcome.outputs, root, validate_before_publish=validate_source,
                                        accepted_by_path=accepted_by_path, recurrence_by_path=memory["by_path"],
                                        library=_bgm_library(source, root))
    validate_source()
    memory["refreshed_elsewhere"] = _refresh_recurrence_elsewhere(root, memory.pop("changed_elsewhere"))
    memory["assets"] = _bgm_recurrence_assets(root) if memory.get("state") == "checked" else None
    semantic_record = _semantic_result_record(basis_run, basis_path) if basis is not None else None
    projection = results.project(
        disposition=outcome.disposition.value, outputs=outcome.outputs, reused=reused,
        held_count=_bgm_held_count(basis_run) if basis is not None else None, zero_result=outcome.zero_result,
        run=run, current_best=record, listening=finished, semantic_record=semantic_record, rebuilt=rebuilt,
        recurrence=memory)
    projection["evaluation_recipe_id"] = basis_run["recipe_id"] if basis is not None else None
    if basis is None:
        outcome = replace(outcome, message=f"公開音声{len(outcome.outputs)}件。保留数と対応範囲外の記録は現在確認できません")
    if operation is not None:
        projection["operation"] = operation
        if operation == "restore_previous":
            projection["state"] = "RESTORED"
            outcome = replace(outcome, message=f"前の保存結果へ戻しました（公開音声{len(outcome.outputs)}件）")
    if outcome.outputs:
        # Scene names are given after publication in their own process; the result never waits for them.
        try:
            from lss.bgm import scene_job

            projection["scene_naming"] = scene_job.start_in_background(
                root, source or Path(run["source"]["path"]), run["source"]["sha256"])["state"]
        except Exception:  # noqa: BLE001 - naming never blocks or fails the BGM result
            projection["scene_naming"] = "not_started"
    # The result stays in LSS state; nothing but audio is written into the user's music folder (user, 2026-11-05).
    return replace(outcome, product_result=projection)


def _bgm_presentation_basis(run: dict, run_path: Path, record: dict | None):
    """Mixed-version CurrentBest uses its latest evaluated run for source-level result counts.

    Restoring the files does not make their first legacy cue the source-level evaluation authority.
    An unavailable/corrupt latest record stays unknown instead of silently showing older counts.
    """
    evaluated = ((record or {}).get("current") or {}).get("evaluated_runs") or []
    if not evaluated or evaluated[-1] == run_path.parents[1].name:
        return run, run_path
    path = run_path.parents[3] / "bgm-candidates" / evaluated[-1] / "run/run.json"
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        verified = _verified_bgm_run(path, run["source"]["sha256"], candidate["recipe_id"])
        return verified, path
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _semantic_result_record(run: dict, run_path: Path) -> dict | None:
    """Optional visibility evidence must still match its authoritative checksum; missing is unknown."""
    item = (run.get("semantics") or {}).get("record") or {}
    folder = (run_path.parent / "semantics").resolve()
    path = (folder / str(item.get("name", ""))).resolve()
    try:
        if path.parent == folder and file_sha256(path) == item.get("sha256"):
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return None


def _existing_bgm_projection(source: Path, run: dict, run_path: Path, files, root: Path, record: dict) -> dict:
    from lss.bgm import listening
    from lss.output import results
    basis = _bgm_presentation_basis(run, run_path, record)
    basis_run, basis_path = basis if basis is not None else (None, None)
    projection = results.project(disposition="PUBLISHED", outputs=files, reused=True,
                                 held_count=_bgm_held_count(basis_run) if basis is not None else None,
                                 run=run, current_best=record, listening=listening.describe(files),
                                 semantic_record=_semantic_result_record(basis_run, basis_path) if basis is not None else None)
    projection["evaluation_recipe_id"] = basis_run["recipe_id"] if basis is not None else None
    return projection


def _bgm_held_count(run: dict) -> int:
    """判定を保留して公開しなかった区間の数（画面の注記用）。v11以前の記録は意味判定の保留数。"""
    semantics = run.get("semantics") or {}
    if isinstance(semantics.get("held_count"), int):
        return semantics["held_count"]
    return int((semantics.get("counts") or {}).get("UNCERTAIN", 0))


def _published_audio(run: dict, item: dict, *, state_root: Path | None = None):
    """公開cueの音を、保留候補の音声と記録した台詞の残りの除去から作り直す（推論しない）。"""
    from lss.bgm import cue_publication, dialogue_residual

    representation = item.get("base_representation") or item.get("representation")
    key = "raw" if representation == "raw_original_mixture" else "music"
    parts = [cue[key] for cue in run["held"] if cue["cue_id"] in item["candidates"]]
    audio = cue_publication.compose(parts, item["start_frame"], item["end_frame"])
    residual = item.get("dialogue_residual") or {}
    if residual.get("applied"):
        # 保存した伴奏と記録した区間から同じ合成をする（推論しない）。
        from lss.bgm import dialogue_residual
        span = dialogue_residual.read_span(residual["instrumental"])
        if residual.get("scope") == "local":
            renderer = (dialogue_residual.region_consensus if residual.get("transition") == "inside_cosine_shared_v1"
                        else dialogue_residual.bounded_local_consensus if residual.get("transition") == "inside_cosine"
                        else dialogue_residual.local_consensus)
            audio = renderer(audio, span, residual["regions"], item["sample_rate"],
                             dialogue_residual.default_recipe())
        else:
            audio = dialogue_residual.consensus(audio, span, dialogue_residual.default_recipe())
    if item.get("selective_composition"):
        from lss.separation import selective
        original = dialogue_residual.read_span(item["selective_composition"]["original"])
        audio = selective.render(original, audio, item["selective_composition"]["plan"])
    if item.get("reconstruction"):
        # 記録した置き換えの計画から同じ音を描く（参照音源が退役していれば取得の台帳から取り直す）。
        saved_recipe = dict(item["reconstruction"].get("recipe") or {})
        saved_recipe.setdefault("transition_curve","equal_power_v1")
        plan = {"donors": item["reconstruction"]["planned_donors"],"recipe":saved_recipe,
                "protected_spans":item["reconstruction"].get("protected_spans",[])}
        audio, _ = _reference_reconstruction(audio, item["sample_rate"], item["start_frame"], plan,state_root=state_root)
    return audio


def _reference_reconstruction(audio, rate: int, start_frame: int, plan: dict, state_root: Path | None = None):
    """計画した領域を、参照音源（2ch・``rate``）から描いて置き換える。参照音源は使用中として退役させない。"""
    import subprocess

    import numpy as np

    from lss.bgm import reference_reconstruction, reference_store
    from lss.paths import resource_root

    root = Path(state_root) if state_root is not None else session.state_dir()
    donors = plan["donors"]
    cache: dict[str, np.ndarray] = {}

    def reference_audio(sha256: str):
        if sha256 not in cache:
            path = reference_store.ensure_media(sha256, state_root=root)
            result = subprocess.run(
                [str(resource_root() / "bin" / "ffmpeg.exe"), "-nostdin", "-v", "error", "-i", str(path),
                 "-map", "0:a:0", "-vn", "-ac", "2", "-ar", str(rate), "-f", "f32le", "pipe:1"],
                capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode:
                raise reference_store.ReferenceMediaUnavailable("参照音源を復号できません")
            cache[sha256] = np.frombuffer(result.stdout, np.float32).reshape(-1, 2).T.astype(np.float64)
        return cache[sha256]

    base = np.asarray(audio, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 2:
        raise ValueError("再構成する音は（frame, 2）である必要があります")
    with reference_store.lease({donor["reference_sha256"] for donor in donors}, root):
        render_recipe = (reference_reconstruction.ReconstructionRecipe(**plan["recipe"]) if plan.get("recipe")
                         else reference_reconstruction.default_recipe())
        rendered, provenance = reference_reconstruction.render(base.T, rate, start_frame / rate, donors,
                                                               reference_audio,recipe=render_recipe,
                                                               protected_spans=plan.get("protected_spans",[]))
    # 作り直しは同じ計画（入力）から同じ音を描く（描画は決定的）。
    provenance["planned_donors"] = [dict(donor) for donor in donors]
    return np.ascontiguousarray(rendered.T), provenance


def _regenerate_published(run: dict, item: dict, destination: Path, source: Path | None = None,
                          *, state_root: Path | None = None) -> dict:
    """公開cueを ``destination`` へ作り直す。音が記録と一致しなければ失敗。"""
    from lss.bgm import collection_metadata, cue_publication

    title, album = item["title"], item["album"]
    if source is not None:
        title, album = _restemmed(item["title"], run, source), collection_metadata.album(Path(source))
    record = cue_publication.publish(
        _published_audio(run, item,state_root=state_root), item["sample_rate"], Path(destination), key=item["key"], title=title,
        album=album, provenance=item["provenance"], input_sha256=run["source"]["sha256"],
        recipe_id=run["recipe_id"], version=item["version"])
    if record["audio"].get("pcm_sha256") != (item.get("audio") or {}).get("pcm_sha256"):
        Path(record["path"]).unlink(missing_ok=True)
        raise ValueError("作り直した公開cueの音が記録と一致しません")
    return record


def _restemmed(text: str, run: dict, source: Path) -> str:
    """記録した名前・titleの原本名の部分を、今処理している原本の名前にする（同じ内容を別の場所・名前で処理した時）。"""
    recorded_stem = Path(str((run.get("source") or {}).get("path", ""))).stem
    if recorded_stem and text.startswith(recorded_stem):
        return Path(source).stem + text[len(recorded_stem):]
    return text


def _public_destination(item: dict, source: Path | None, run: dict | None = None) -> Path:
    """公開名。記録した名前（版ごとの名前の規則のまま）、同じ内容の原本を別の場所から処理した時はその原本の隣に、
    その原本の名前で置く。"""
    from lss.bgm import cue_publication

    recorded = Path(item["destination"])
    if source is None:
        return recorded
    name = _restemmed(recorded.name, run, source) if run is not None else recorded.name
    return cue_publication.folder_for(source) / name


def _run_regenerator(state_root: Path, source_sha256: str, *, rebuilt: list | None = None):
    """CurrentBestの切替・整合で公開cueを失っていた時に、同じ原本のrunの記録から推論なしで作り直す関数。

    ``rebuilt`` へ作り直した公開名を加える（結果の表示で「再利用」と「作り直し」を区別する）。"""
    import tempfile

    from lss.bgm import current_best

    def regenerate(item: dict, public: Path) -> None:
        for run_path in sorted((Path(state_root) / "bgm-candidates").glob("*/run/run.json")):
            try:
                run = json.loads(run_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (run.get("source") or {}).get("sha256") != source_sha256:
                continue
            for published in run.get("published") or []:
                if (published.get("audio") or {}).get("pcm_sha256") != item["pcm_sha256"]:
                    continue
                # 作り直しの置き場は毎回新しく作る（公開フォルダの隣。同じボリュームで置き換えるため）。片付けるのは
                # そこへ置いた1つのファイルと、空になった置き場だけ（既存の同名フォルダや中の物には触れない）。
                Path(public).parent.mkdir(parents=True, exist_ok=True)
                folder = Path(tempfile.mkdtemp(prefix=".lss-regenerate-", dir=Path(public).parent))
                made = folder / Path(public).name
                try:
                    made = Path(_regenerate_published(run, published, made,state_root=state_root)["path"])
                    os.replace(made, public)
                finally:
                    if made.parent == folder:
                        made.unlink(missing_ok=True)
                    try:
                        folder.rmdir()
                    except OSError:
                        pass
                if rebuilt is not None:
                    rebuilt.append(str(public))
                return
        raise current_best.CurrentBestError(f"公開cueを作り直せるrunがありません: {public}")
    return regenerate


def _assess_old_cues(run: dict, run_path: Path, old_files: list[dict]) -> list[dict]:
    """前から公開しているcueの妥当性（``state``）と危険（``risks``）を、このrunが測った同じ原本の観測で、そのcue自身の
    区間について決める。妥当性と危険は別の判断（採用の方針 `_BGM_ADOPTION_POLICY`）。

    ``INVALID`` は、そのcue自身の音にある正の根拠だけで決める（新しいrunの区切り方・範囲の未確定・根拠の欠け・失敗、
    場面や区画の種類から見込まれる危険は根拠にしない）：
    - ``crosses_edit_boundary``：区間の内側に編集境界（デジタル無音）がある。音が途切れた別々の断片（CM・告知等の
      挿入を含みうる）を一つのcueに繋いでいる（cue自身の音の事実）
    危険（``risks``。場面・区画の種類から見込まれる危険で、そのcueが悪いことの証明ではない）：
    - ``quality_risk:heavy_nonmusic_event_scene``：区間が重い非音楽イベントの場面。この場面の抽出は失敗が多いが、
      cueごとに良し悪しを見分ける実行時の根拠は無い。新しい候補は保留にするが、前から公開しているcueは外さない
    - ``role_risk:within_edit_insert_section``：区間が編集境界に挟まれた挿入の区画にある。CMか番組の挿入（予告・
      コーナー・終了画面）か決められない。新しい候補は保留にするが、前から公開しているcueは外さない
    正の根拠も危険も無ければ ``VALID``、危険だけなら ``UNKNOWN``、観測が無ければ ``UNKNOWN``（いずれも不正ではない）。
    番組の端・放送挿入のタグ・中ほど・境界の付近は文脈として残すだけ。
    新しいrunが参照録音と同じ録音の連続した抜粋（既発売）と決めた区間に重なる旧cueも、文脈
    （``known_released_recording``）として記録するだけで、妥当性は変えない（既発売の録音があることは、
    前から公開しているcueの音が不正だという根拠ではない）。
    """
    import numpy as np

    from lss.bgm import cue_scope

    semantics = run.get("semantics") or {}
    folder = run_path.parent / "semantics"
    try:
        record = json.loads((folder / semantics["record"]["name"]).read_text(encoding="utf-8"))
        mixture = np.load(folder / "mixture-probabilities.npy")
        boundaries = [(float(a), float(b)) for a, b in record.get("edit_boundaries") or []]
        if "edit_boundaries" not in record:
            raise KeyError("edit_boundaries")
    except (OSError, ValueError, KeyError, TypeError):
        return [{"state": "UNKNOWN", "basis": ["new_run_has_no_assessable_observations"], "risks": []}
                for _ in old_files]
    relations = record.get("reference_relations") or {}
    occurrences = (relations.get("occurrences") or []) if relations.get("state") == "checked" else []
    out = []
    for item in old_files:
        try:
            start, end = float(item["start_seconds"]), float(item["end_seconds"])
            span = {"start_seconds": start, "end_seconds": end,
                    "start_edge": {"kind": "public_cue_edge"}, "end_edge": {"kind": "public_cue_edge"}}
            observed = cue_scope.safety(span, mixture)
            observed_context = cue_scope.context(span, mixture, boundaries)
        except (KeyError, TypeError, ValueError):
            out.append({"state": "UNKNOWN", "basis": ["cue_span_not_assessable"], "risks": []})
            continue
        basis, risks = [], []
        structured = len(mixture) >= cue_scope.MID_PROGRAM_MINIMUM_SECONDS   # 新しい候補を分ける条件と同じ
        if structured and any(a >= start + 1.0 and b <= end - 1.0 for a, b in boundaries):
            basis.append("crosses_edit_boundary")
        elif observed_context["insert_section"]:
            risks.append("role_risk:within_edit_insert_section")
        if observed["heavy_nonmusic_event_scene"]:
            risks.append("quality_risk:heavy_nonmusic_event_scene")
        known = [{"reference_sha256": o["reference_sha256"], "display_title": o.get("display_title"),
                  "start_seconds": o["start_seconds"], "end_seconds": o["end_seconds"],
                  "reference_range_seconds": o.get("reference_range_seconds")}
                 for o in occurrences
                 if min(end, float(o["end_seconds"])) - max(start, float(o["start_seconds"])) > 0.5]
        if known:
            observed_context = {**observed_context, "known_released_recording": known}
        state = "INVALID" if basis else ("UNKNOWN" if risks else "VALID")
        out.append({"state": state, "basis": basis or (["no_asset_local_invalidation"] if risks
                                                      else ["no_invalidating_observation"]),
                    "risks": risks, "safety": observed, "context": observed_context})
    return out


def _bgm_current_best(root: Path, source: Path, source_sha256: str, key: str, run: dict,
                      run_path: Path, *, rebuilt: list | None = None) -> tuple[dict, Path, tuple[Path, ...] | None]:
    """このrunの判断をCurrentBestへ反映する（cueごと。妥当な旧cueは残し、不正な旧cueは履歴へ）。

    - 中断していた切替は先に続きを行う。
    - 既にこのrunを反映済みなら、公開中のファイルを確かめ、欠けていれば推論なしで作り直す。
    - 利用者が前の結果へ戻した（pinned）原本は、新しい版へ勝手に戻さない。
    - このRecipeが置き換えると宣言していない版（新しい版など）の公開中のcueがあれば触れない。
    - 旧cueの妥当性は `_assess_old_cues`、最良の選択は `current_best.select`。新しいrunが保留・未確定でも
      妥当な旧cueは残る。
    戻り値は、画面と記録に返すrunとその記録の場所、公開中のファイル（CurrentBestが無ければNone）。
    """
    from lss.bgm import cue_publication, current_best, current_best_lineage, write_targets

    regenerate = _run_regenerator(root, source_sha256, rebuilt=rebuilt)
    current_best.reconcile(root, source_sha256, regenerate)
    record = current_best.read(root, source_sha256)
    record = _relocated_to_source(root, source, source_sha256, record, regenerate, rebuilt=rebuilt)
    current = (record or {}).get("current")
    own = (*current_best.SUPERSEDED_WORKFLOWS, _BGM_CANDIDATE_WORKFLOW)
    if current is not None and key not in (current.get("evaluated_runs") or []):
        foreign = [item for item in current["files"] if item.get("workflow") not in own]
        if current.get("pinned") or foreign:
            # 利用者が戻した結果、または置き換えると宣言していない版の結果。触れない。
            return run, run_path, _verified_current_files(root, source, source_sha256, record, regenerate,
                                                          rebuilt=rebuilt)
    reevaluate = (current is not None and key in (current.get("evaluated_runs") or [])
                  and current.get("adoption_policy") != _BGM_ADOPTION_POLICY and not current.get("pinned"))
    if current is not None and key in (current.get("evaluated_runs") or []) and not reevaluate:
        # 反映済み。消えた・音の変わった公開cueを作り直す（tagだけの書き戻しは作り直さない）。
        return run, run_path, _verified_current_files(root, source, source_sha256, record, regenerate,
                                                      rebuilt=rebuilt)
    previous_current = current
    old_files = list((current or {}).get("files") or [])
    if reevaluate:
        # 前の採用の方針でこのrunを反映した原本。反映する前の公開cueの集まりから、今の方針で選び直す（推論しない）。
        # 方針の選び直し・巻き戻しも履歴を残すので、それ（前の方針の結果）ではなく、このrunを反映した切替の前の
        # 集まりを使う（切替の種類を持たない古い記録の履歴も、runを反映した切替の前の集まりとして扱う）。
        before = [h for h in (record.get("history") or []) if h.get("replaced_by_run_key") == key
                  and h.get("replaced_by_transition") not in ("adoption_policy_reevaluation", "restored_previous")]
        old_files = ([{k: v for k, v in item.items() if k != "archive_path"} | (
                        {"archive_path": item["archive_path"]} if item.get("archive_path") else {})
                      for item in before[-1]["files"]] if before
                     else [item for item in old_files if item.get("run_key") != key])
    if current is None:
        for old_key, old_path, old_run in current_best.legacy_current(root, source_sha256, key):
            for item in old_run.get("published") or []:
                pcm = (item.get("audio") or {}).get("pcm_sha256")
                here = _public_destination(item, source, old_run)
                # A legacy run's recorded file outside this source's folder (a moved library or copied state) is
                # evidence that the cue was accepted, never a file to retire or keep: the cue is taken over beside
                # this source (rebuilt there without inference when it is kept), and the old place is not touched.
                inside = write_targets.inside(Path(item["path"]), cue_publication.folder_for(source))
                for candidate in ((Path(item["path"]), here) if inside else (here, Path(item["path"]))):
                    if current_best._matches(candidate, pcm):
                        old_files.append(current_best.file_entry(old_run, old_key, old_path, item,
                                                                 candidate if write_targets.same_place(candidate, here) or inside else here,
                                                                 adoption="legacy_before_current_best"))
                        break
        if old_files:
            previous_current = {"files": old_files, "pinned": False, "transition": "legacy_before_current_best",
                                "evaluated_runs": sorted({item["run_key"] for item in old_files})}
    published = run.get("published") or []
    if not published and not old_files:
        # 公開cueが無く、前から公開しているcueも無い（保留・候補なし・対象なし）。CurrentBestの記録は作らない。
        return run, run_path, None
    for item in old_files:
        if "extent" not in item:
            # 範囲の決まり方を持たない前の記録は、そのcueを公開したrunの記録から読む（読めなければ不明のまま）。
            accepted = _accepted_run_item(item)
            item["extent"] = ((accepted or {}).get("cue_extent") or {}).get("state")
    assessments = _assess_old_cues(run, run_path, old_files)
    staged: dict[str, Path] = {}
    new_files = []
    for item in published:
        public = _public_destination(item, source, run)
        path = Path(item["path"])
        if path != public and path.is_file() and audio_identity.matches_record(path, item):
            staged[str(public)] = path
        new_files.append(current_best.file_entry(run, key, run_path, item, public,
                                                 adoption="first_publication" if not old_files
                                                 else "recipe_supersession"))
    # 危険（場面・区画の種類からの見込み）は妥当性ではない。全ての旧cueを妥当性→最良の順で選び、危険は記録に残す。
    chosen = current_best.select(old_files, assessments, new_files)
    risks = {old["public_path"]: a.get("risks") or [] for old, a in zip(old_files, assessments)}
    decisions_old = [{**row, "risks": risks.get(row["public_path"], [])} for row in chosen["decisions"]]
    target = [*chosen["keep"], *chosen["place"]]
    public_now = {(item["public_path"], item.get("pcm_sha256")) for item in (previous_current or {}).get("files") or []}
    if reevaluate:
        public_now = {(item["public_path"], item.get("pcm_sha256")) for item in current["files"]}
        for item in target:
            if item.get("archive_path"):
                staged[item["public_path"]] = Path(item["archive_path"])
    target_ids = {(item["public_path"], item.get("pcm_sha256")) for item in target}
    retire = [item for item in ((current["files"] if reevaluate else (previous_current or {}).get("files")) or [])
              if (item["public_path"], item.get("pcm_sha256")) not in target_ids]
    next_current = {"files": [{**{k: v for k, v in item.items() if k != "archive_path"},
                               **({"placed_now": True} if (item["public_path"], item.get("pcm_sha256"))
                                  not in public_now else {})} for item in target],
                    "pinned": False, "adopted_at": current_best._now(),
                    "transition": ("adoption_policy_reevaluation" if reevaluate
                                   else "recipe_supersession" if old_files else "first_publication"),
                    "evaluated_runs": list((current or {}).get("evaluated_runs") or []) + ([] if reevaluate else [key]),
                    "adoption_policy": _BGM_ADOPTION_POLICY,
                    # An accepted cue that could not be rebuilt stays UNKNOWN until a cue covers its region again:
                    # a later run never turns it into ABSENT by forgetting it.
                    **_still_unavailable(current, target),
                    "decisions": {"old": decisions_old, "new": chosen["new_decisions"],
                                  "by_run_key": key, "by_recipe_id": run["recipe_id"],
                                  "adoption_policy": _BGM_ADOPTION_POLICY}}
    journal = current_best.plan_switch(root, source_sha256, next_current, staged,
                                       cue_publication.history_root(source), retire=retire,
                                       record=record, previous_current=current if reevaluate else previous_current,
                                       mode=next_current["transition"])
    current_best.execute(root, current_best_lineage.separate_history_copies(journal), regenerate)
    placed = {item["public_path"] for item in chosen["place"]}
    not_placed = {item["public_path"]: item for item in chosen["not_placed"]}
    for item in published:
        public = str(_public_destination(item, source, run))
        if public in placed:
            item.update(path=public, sha256=file_sha256(Path(public)), current_best="placed")
        elif public in not_placed:
            # 同じ領域の妥当な前の版を残したので公開しない。音は保留候補から推論なしで作り直せる。
            staged_file = staged.get(public)
            if staged_file is not None and current_best.STAGING_DIR in staged_file.parts:
                staged_file.unlink(missing_ok=True)
            item.update(current_best={"state": "not_placed", "reason": not_placed[public]["reason"],
                                      "kept": not_placed[public]["kept"]})
    session._write_json_atomic(run_path, run)
    return run, run_path, tuple(Path(item["public_path"]) for item in journal["next_record"]["current"]["files"])


def _still_unavailable(current: dict | None, target: list[dict]) -> dict:
    from lss.bgm import current_best

    rows = [row for row in (current or {}).get("unavailable") or []
            if not any(current_best._overlap(row, item) > current_best.OVERLAP_SECONDS for item in target)]
    return {"unavailable": rows} if rows else {}


def _local_run_path(root: Path, item: dict) -> dict:
    """When a relocated row names another state's run record, name this state's copy of it (if present).

    A row's ``run_path`` says which state adopted it (`lss.bgm.write_targets`). Relocating the cue into this
    source's own folder is this state's adoption of that file, so the row then names this state's run record.
    """
    from lss.bgm import write_targets

    try:
        local = Path(root) / "bgm-candidates" / str(item["run_key"]) / "run" / "run.json"
    except (KeyError, TypeError):
        return {}
    if local.is_file() and not write_targets.same_place(local, Path(str(item.get("run_path") or ""))):
        return {"run_path": str(local), "relocated_from_run_path": item.get("run_path")}
    return {}


def _relocated_to_source(root: Path, source: Path, source_sha256: str, record: dict | None, regenerate,
                         *, rebuilt: list | None = None) -> dict | None:
    """Move a CurrentBest that names files outside this source's own folder to that folder before deciding.

    The same content processed from another place (a moved library, a copied state) must not retire, keep, retag or
    delete the files the record still names elsewhere: those are observations. `_verified_current_files` rebuilds
    the cues beside this source without inference and rewrites the record; the old place is never touched.
    """
    from lss.bgm import cue_publication, current_best, write_targets

    files = ((record or {}).get("current") or {}).get("files") or []
    folder = cue_publication.folder_for(source)
    if not any(not write_targets.inside(Path(item["public_path"]), folder) for item in files):
        return record
    _verified_current_files(root, source, source_sha256, record, regenerate, rebuilt=rebuilt)
    return current_best.read(root, source_sha256)


def _verified_current_files(root: Path, source: Path, source_sha256: str, record: dict,
                            regenerate, *, rebuilt: list | None = None) -> tuple[Path, ...]:
    """公開中のcueのファイルを確かめ、消えた・音の変わったものを推論なしで作り直す。

    同じ内容の原本を別の場所から処理した時は、その原本の隣へ、その原本の名前で作り直し、CurrentBestの所在を
    移す（runは内容で決まり、保存先は所在で決まる。前の場所のファイルには触れない）。
    """
    from lss.bgm import cue_publication, current_best, write_targets

    current = record["current"]
    files, outputs, moved = [], [], False
    for item in current.get("files") or []:
        public = Path(item["public_path"])
        made_from = None
        try:
            run = json.loads(Path(item["run_path"]).read_text(encoding="utf-8"))
            made_from = next(entry for entry in run.get("published") or []
                             if (entry.get("audio") or {}).get("pcm_sha256") == item.get("pcm_sha256"))
            here = _public_destination(made_from, source, run)
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            # Without the run record the cue keeps its recorded name, but still beside *this* source: a recorded
            # location elsewhere (a library the state was copied from) is never written.
            here = cue_publication.folder_for(source) / public.name
        if (not write_targets.same_place(here, public)
                and not write_targets.inside(public, cue_publication.folder_for(source))):
            moved = True
            item = {**item, "relocated_from": str(public), **_local_run_path(root, item)}
            public = here
        if not current_best._matches(public, item.get("pcm_sha256")):
            if public.exists():
                # 音の変わったファイル（利用者の編集等）を無言で上書きしない。履歴の別の場所へ移してから作り直す。
                displaced = (public.parent.parent / current_best.HISTORY_DIR / "displaced"
                             / current_best._now().replace(":", ""))
                displaced.mkdir(parents=True, exist_ok=True)
                os.replace(public, displaced / public.name)
            public.parent.mkdir(parents=True, exist_ok=True)
            if made_from is not None:
                _regenerate_published(run, made_from, public, source,state_root=root)
                if rebuilt is not None:
                    rebuilt.append(str(public))
            else:
                regenerate(item, public)
        files.append({**item, "public_path": str(public)})
        outputs.append(public)
    if moved:
        current_best._write_json_atomic(current_best.record_path(root, source_sha256),
                                        {**record, "current": {**current, "files": files}})
    return tuple(outputs)


def _verified_analysis(receipt_path: Path, analysis_sha256: str, analyzer_id: str, *,
                       blocks: list[int] | None = None,
                       execution_profile: str | None = None) -> dict | None:
    """解析用stemの受領票が、同じ解析用音声・同じ解析器・同じブロックの完全な結果ならそれを返す。"""
    from lss.bgm import episode_cues

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("type") != "bgm_sfc_analysis_receipt"
                or receipt.get("analyzer_id") != analyzer_id
                or (receipt.get("analysis") or {}).get("sha256") != analysis_sha256
                or receipt.get("blocks") != blocks
                or receipt.get("execution_profile") != execution_profile
                or not isinstance(receipt.get("runtime_id"), str)):
            return None
        stems = receipt.get("stems") or {}
        folder = receipt_path.parent.resolve()
        for stem in episode_cues.STEMS:
            item = stems.get(stem) or {}
            path = Path(str(item.get("path", ""))).resolve()
            if path.parent != folder or file_sha256(path) != item.get("sha256"):
                return None
        return receipt
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _verified_four_stem(receipt_path: Path, signal_sha256: str, analyzer_id: str, spans: list) -> dict | None:
    """4分類の受領票が、同じ信号・同じ解析器・同じ区間の完全な結果ならそれを返す（中断再開と再実行で使い回す）。"""
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("type") != "bgm_four_stem_receipt"
                or receipt.get("analyzer_id") != analyzer_id
                or (receipt.get("signal") or {}).get("sha256") != signal_sha256
                or receipt.get("spans") != [list(span) for span in spans]
                or not isinstance(receipt.get("runtime_id"), str)):
            return None
        items = (receipt.get("stems") or {}).get("music_vocals") or []
        folder = receipt_path.parent.resolve()
        if [item.get("span") for item in items] != receipt["spans"]:
            return None
        for item in items:
            path = Path(str(item.get("path", ""))).resolve()
            if path.parent != folder or file_sha256(path) != item.get("sha256"):
                return None
        return receipt
    except (OSError, ValueError, TypeError, AttributeError):
        return None


class _BgmAnalysisFailed(RuntimeError):
    """解析ジョブの構造化失敗を、製品境界で意味を失わず分類できるよう保持する。"""

    def __init__(self, worker_result: dict) -> None:
        self.worker_result = dict(worker_result)
        super().__init__(str(worker_result.get("message") or worker_result.get("error_code")))


def _reference_completion_seconds() -> float:
    """判定の後、作品のアルバムの残りの取得が終わるのを待つ上限（秒）。既定は30分。"""
    try:
        return max(0.0, float(os.environ.get("LSS_REFERENCE_COMPLETION_SECONDS", "1800")))
    except ValueError:
        return 1800.0


def _run_bgm_candidate_request(
    request: ProcessingRequest, *, state_root: Path | None = None, analyze=None,
    memory_refresh: dict | None = None,
) -> ProcessingOutcome:
    """BGM候補を作り、判定の後に届いた作品の参照があれば、その参照でもう一度だけ判定し直す。

    参照の取得は判定を待たせる上限を超えて続くことがある（アルバム全体の取得）。その時の結果は先に公開し、
    取得が終わって参照の集まりが変わっていれば、新しい集まりの run を作って CurrentBest を置き換える
    （発売済みの曲を重複して公開したままにしない）。集まりが同じなら、その run がもうあるので何もしない。
    """
    outcome = _run_bgm_candidate_request_once(request, state_root=state_root, analyze=analyze,
                                              memory_refresh=memory_refresh)
    if outcome.disposition is ResultDisposition.FAILED or len(request.sources) != 1:
        return outcome
    try:
        from lss.bgm import reference_acquisition
        source = request.sources[0].resolve()
        root = Path(state_root).resolve() if state_root is not None else session.state_dir().resolve()
        source_sha256 = file_sha256(source)
        reference_acquisition.wait_active(source, source_sha256, root, _reference_completion_seconds(),
                                          quitting.checkpoint)
        _, references_digest, _, _ = _registered_references(root, source=source, source_sha256=source_sha256,
                                                             ingest=False)
        key = _bgm_candidate_run_key(source_sha256, _bgm_candidate_recipe()["recipe_id"], references_digest,
                                     memory_refresh=memory_refresh["token"] if memory_refresh else None)
        if (root / "bgm-candidates" / key / "run" / "run.json").is_file():
            return outcome
    except quitting.JobAbandonedError:
        return outcome   # 先の結果は公開済み。次の処理で新しい参照の run を作る
    except OSError:
        return outcome
    return _run_bgm_candidate_request_once(request, state_root=state_root, analyze=analyze,
                                           memory_refresh=memory_refresh)


def _run_bgm_candidate_request_once(
    request: ProcessingRequest, *, state_root: Path | None = None, analyze=None,
    memory_refresh: dict | None = None,
) -> ProcessingOutcome:
    """単話だけから、来歴付きの非公開BGM候補を自動生成または安全に再利用する。

    対応領域の判断とCurrentBestに従って公開し、条件を欠く候補は ``HELD`` に残す。完全な無音だけを検証済みの
    ``ZERO`` にし、区間が見つからなかっただけならfalse zeroにしない。解析器、threshold、
    保存先は製品側で決め、呼び出し側へ選択を戻さない。``analyze(job, destination_dir)``
    は解析ジョブを常駐workerへ送る関数で、省略時は同じ処理をこのprocessで行う。
    """
    if request.intent is not UserIntent.EXTRACT_BGM_CUES:
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED, error_code="intent_mismatch",
            message="BGM候補入口にはBGM cue抽出要求が必要です",
        )
    if len(request.sources) != 1 or request.urls:
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED, error_code="bgm_requires_single_local_source",
            message="BGM候補は完成済み番組を1本ずつ処理します",
        )
    source = request.sources[0].resolve()
    try:
        if not source.is_file():
            raise FileNotFoundError(f"入力が見つかりません: {source}")
        source_sha256 = file_sha256(source)
        if memory_refresh is not None and source_sha256 != memory_refresh["source_sha256"]:
            raise session.InputChangedError("BGMを再評価する原本が変更されています")
        recipe = _bgm_candidate_recipe()
        root = Path(state_root).resolve() if state_root is not None else session.state_dir().resolve()
        # 作品の既発売の参照を探して取得・確認する（配備版。ネットワーク・取得が無ければ何もしない）。
        # この番組の判定に使う参照が決まるまで待ち、アルバムの残りの取得は分離と並行して背景で続ける。
        from lss.bgm import reference_acquisition
        preparation = reference_acquisition.Preparation(
            source, source_sha256, root / "bgm-reference-fingerprints", root).start()
        preparation_state = preparation.wait_ready(_reference_preparation_seconds())
        reference_paths, references_digest, references_missing, reference_sources = _registered_references(
            root, source=source, source_sha256=source_sha256)
        reference_sources = {**reference_sources, "preparation": {
            k: preparation_state.get(k) for k in ("state", "wait", "work_key", "title", "basis", "reused",
                                                  "verified_album", "failure_kind", "failure_detail", "retryable",
                                                  "needs_login") if k in preparation_state}}
        key = _bgm_candidate_run_key(source_sha256, recipe["recipe_id"], references_digest,
                                     memory_refresh=memory_refresh["token"] if memory_refresh else None)
        product_root = root / "bgm-candidates" / key
        run_dir = product_root / "run"
        run_path = run_dir / "run.json"
        from lss.bgm import current_best

        # 同じ原本のCurrentBestを扱う処理は、Recipeの版をまたいで一つずつ行う。
        with _BgmCandidateRunLock(root / current_best.DIRECTORY / source_sha256),                 _BgmCandidateRunLock(product_root):
            _ensure_bgm_candidate_recipe(product_root / "recipe.json", recipe)
            rebuilt: list[str] = []
            with quitting.writing_user_folder():
                current_best.reconcile(root, source_sha256, _run_regenerator(root, source_sha256, rebuilt=rebuilt))
            record = current_best.read(root, source_sha256)
            pinned = (record or {}).get("current") or {}
            if pinned.get("pinned") and key not in (pinned.get("evaluated_runs") or []) and pinned.get("files"):
                # 利用者が前の結果へ戻した原本。新しい版で勝手に処理し直さない（推論しない）。
                with quitting.writing_user_folder():
                    outputs = _verified_current_files(root, source, source_sha256, record,
                                                      _run_regenerator(root, source_sha256, rebuilt=rebuilt),
                                                      rebuilt=rebuilt)
                pinned_path = Path(pinned["files"][0]["run_path"])
                pinned_run = json.loads(pinned_path.read_text(encoding="utf-8"))
                from lss.bgm import source_visual_backfill
                source_visual_backfill.ensure(source, source_sha256, pinned_run, pinned_path)
                return _bgm_outcome(pinned_run, pinned_path,
                                    reused=True, outputs=outputs, source=source, rebuilt=rebuilt)
            if run_path.is_file():
                try:
                    verified = _verified_bgm_run(run_path, source_sha256, recipe["recipe_id"])
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    # 消えた・壊れた候補を完了扱いしない。記録と合う候補は残し、作り直す。
                    _retire_unverifiable_run(run_path)
                    verified = None
                if verified is not None:
                    # 完了済みのrun。CurrentBestにし（切替が途中なら続きを行う）、消えた公開cueは推論なしで作り直す。
                    # 失敗しても、検証済みの候補と判定は捨てない（失敗として返す）。
                    with quitting.writing_user_folder():
                        current, current_path, outputs = _bgm_current_best(root, source, source_sha256, key,
                                                                           verified, run_path, rebuilt=rebuilt)
                    from lss.bgm import source_visual_backfill
                    source_visual_backfill.ensure(source, source_sha256, verified, run_path)
                    if current_path != run_path:
                        source_visual_backfill.ensure(source, source_sha256, current, current_path)
                    return _bgm_outcome(current, current_path, reused=True, outputs=outputs, source=source,
                                        rebuilt=rebuilt)
            quitting.checkpoint()
            _discard_interrupted_attempt(run_dir, source_sha256, root)
            run = _process_bgm_candidates(source, source_sha256, recipe, run_dir, key, analyze,
                                          references=(reference_paths, references_digest, references_missing,
                                                      root / "bgm-reference-fingerprints"),
                                          reference_sources=reference_sources)
            if memory_refresh is not None:
                run["memory_refresh"] = memory_refresh
            if file_sha256(source) != source_sha256:
                raise session.InputChangedError("BGM候補の保存中に原本が変更されました")
            session._write_json_atomic(run_path, run)
            verified = _verified_bgm_run(run_path, source_sha256, recipe["recipe_id"])
            try:
                # Before publication: an interruption after it still leaves the user's naming request to resume.
                from lss.bgm import scene_job
                scene_job._record_request(root, source, source_sha256)
            except Exception:  # noqa: BLE001 - naming never blocks or fails the BGM result
                pass
            # 完了記録を書いた後の公開（CurrentBestの切替）は、終了の要求があっても最後まで運ぶ（数秒）。
            with quitting.writing_user_folder():
                current, current_path, outputs = _bgm_current_best(root, source, source_sha256, key, verified,
                                                                   run_path, rebuilt=rebuilt)
            return _bgm_outcome(current, current_path, reused=False, outputs=outputs, source=source,
                                rebuilt=rebuilt)
    except Exception as exc:
        if isinstance(exc, quitting.JobAbandonedError):
            # 利用者が処理中に終了した。失敗ではなく、次の起動で処理待ちへ戻る（画面の順次処理が扱う）。
            return ProcessingOutcome(disposition=ResultDisposition.FAILED, error_code="job_abandoned",
                                     message=str(exc))
        worker_result = getattr(exc, "worker_result", None)
        try:
            from lss.acquisition.store import sanitize_text
            message = sanitize_text(
                str(worker_result.get("message") or exc)
                if isinstance(worker_result, dict) else str(exc)
            )
        except Exception:
            message = str(exc)
        error_code = None
        if isinstance(worker_result, dict) and worker_result.get("status") == "error":
            error_code = worker_result.get("error_code")
        elif isinstance(exc, session.UnsupportedChannelLayoutError):
            error_code = "unsupported_channels"
        if isinstance(error_code, str) and error_code:
            return ProcessingOutcome(
                disposition=(ResultDisposition.UNSUPPORTED if error_code == "unsupported_channels"
                             else ResultDisposition.FAILED),
                error_code=error_code, message=message,
            )
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED,
            error_code="bgm_candidate_processing_failed", message=message,
        )
    finally:
        # 作業領域へ書き終えた公開を、完了記録とCurrentBestの切替まで運び終えた（または失敗した）。
        quitting.release_publication()


def run_bgm_candidate_request(request: ProcessingRequest, *, state_root: Path | None = None, analyze=None) -> ProcessingOutcome:
    """Product entry with a projection for every disposition, including validation/execution failure."""
    outcome = _run_bgm_candidate_request(request, state_root=state_root, analyze=analyze)
    if outcome.disposition is ResultDisposition.PUBLISHED:
        outcome = _refresh_bgm_from_memory(outcome, request, state_root=state_root, analyze=analyze)
    if outcome.product_result is not None:
        return outcome
    from lss.output import results
    return replace(outcome, product_result=results.project(
        disposition=outcome.disposition.value, outputs=outcome.outputs, reused=outcome.reused,
        held_count=len(outcome.held_artifacts), zero_result=outcome.zero_result, error_code=outcome.error_code))


def _refresh_bgm_from_memory(outcome: ProcessingOutcome, request: ProcessingRequest, *,
                             state_root: Path | None = None, analyze=None) -> ProcessingOutcome:
    """Revisit at most one older unresolved source after a normal BGM job grows local memory.

    This is a trigger, never permission to replace an asset. The ordinary source processing, never-lower
    guards, CurrentBest selection, journal, listening metadata and rollback decide the result. A separate
    token-addressed run keeps the previous completed run reusable. Failure of this optional revisit cannot
    turn the successfully saved foreground source into a failure. No recursive refresh or unbounded sweep.
    """
    from lss.bgm import memory_refresh

    root = Path(state_root).resolve() if state_root is not None else session.state_dir().resolve()
    try:
        # Two foreground sources may complete together. One optional revisit owns the planner and receipts.
        with _BgmCandidateRunLock(root / memory_refresh.DIRECTORY, wait_seconds=0):
            quitting.checkpoint()
            plans = memory_refresh.pending(root, recipe_id=_bgm_candidate_recipe()["recipe_id"])
            foreground = {str(p.resolve()) for p in request.sources}
            foreground_sha = (outcome.product_result or {}).get("source_sha256")
            plan = next((p for p in plans if p["source_sha256"] != foreground_sha
                         and str(Path(p["source_path"]).resolve()) not in foreground), None)
            if plan is None or not memory_refresh.claim(root, plan):
                return outcome
            # The nested entry has its own per-source lock; it does not schedule more jobs.
            refreshed = _run_bgm_candidate_request(
                ProcessingRequest.from_paths([Path(plan["source_path"])], UserIntent.EXTRACT_BGM_CUES),
                state_root=root, analyze=analyze, memory_refresh=plan)
            result = {"ok": refreshed.disposition in (ResultDisposition.PUBLISHED, ResultDisposition.HELD, ResultDisposition.ZERO),
                      "disposition": refreshed.disposition.value, "error_code": refreshed.error_code,
                      "outputs": [str(p) for p in refreshed.outputs], "record_path": str(refreshed.record_path or ""),
                      "reused": refreshed.reused, "product_result": refreshed.product_result}
            if refreshed.error_code == "job_abandoned":
                # Leave the claim resumable: a user quit is neither a completed attempt nor a failed witness.
                return outcome
            memory_refresh.finish(root, plan, result)
            projection = dict(outcome.product_result or {})
            projection["related_bgm_refresh"] = {"source": plan["source_path"], **result}
            return replace(outcome, product_result=projection)
    except Exception:  # noqa: BLE001 - optional maintenance cannot invalidate a published foreground asset
        return outcome


def _process_bgm_candidates(source: Path, source_sha256: str, recipe: dict, run_dir: Path,
                            key: str, analyze, *, references=((), None, (), None),
                            reference_sources: dict | None = None) -> dict:
    """原本を復号し、解析用3 stemを作り、候補を保存して完了記録の本文を返す。"""
    import soundfile as sf

    from lss.bgm import episode_cues, overlap, second_pass, sfc_analysis, vocal_recovery
    from lss.core import audio_input, presence
    from lss.core.audio_timeline import probe_audio_timeline

    started = time.perf_counter()
    timing: dict[str, float] = {}

    def mark(name: str, since: float) -> float:
        now = time.perf_counter()
        timing[name] = round(timing.get(name, 0.0) + now - since, 2)
        return now

    timeline = probe_audio_timeline(source).to_dict()
    base = {
        "type": _BGM_RUN_TYPE, "schema_version": 1, "recipe_id": recipe["recipe_id"],
        "workflow": recipe["workflow"],
        "source": {"path": str(source), "sha256": source_sha256, "bytes": source.stat().st_size,
                   "audio_timeline_origin": timeline},
        "publication_state": "not_published", "promotion_allowed": False,
    }
    # 異常終了で残っても、次の起動時に古い作業領域として回収される名前にする。
    scratch = session.work_dir() / f"decode-bgm-analysis-{key[:16]}"
    with audio_input.prepare_input(source, work_root=session.work_dir()) as decoded:
        decoded = Path(decoded)
        quitting.checkpoint()
        if file_sha256(source) != source_sha256:
            raise session.InputChangedError("復号中に原本が変更されました")
        session.validate_input(decoded)
        info = sf.info(str(decoded))
        if info.channels != 2:
            # 対応範囲は完成済み2ch番組。monoを複製して2chのふりをしない。
            raise session.UnsupportedChannelLayoutError(
                f"BGM候補は2chの完成済み番組を対象にします（{info.channels}ch）",
                channels=info.channels)
        signal = presence.inspect_signal(decoded)
        if signal.exactly_silent:
            zero = signal.zero_result()
            session.validate_zero_result(zero)
            return {**base, "status": _BGM_STATUS_ZERO, "zero_result": zero}
        scratch.mkdir(parents=True, exist_ok=True)
        analysis = scratch / "analysis.wav"
        preparation_path = scratch / "analysis-preparation.json"
        try:
            preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
            if (preparation.get("source_sha256") != source_sha256
                    or file_sha256(analysis) != preparation.get("sha256")):
                preparation = None
        except (OSError, ValueError, AttributeError):
            preparation = None
        clock = mark("decode_and_checks", started)
        if preparation is None:
            preparation = {**sfc_analysis.analysis_copy(decoded, analysis),
                           "source_sha256": source_sha256}
            session._write_json_atomic(preparation_path, preparation)
        clock = mark("analysis_copy", clock)
        stems_dir = (scratch / "sfc").resolve()
        receipt = _verified_analysis(stems_dir / "receipt.json", preparation["sha256"],
                                     recipe["analyzer_id"],
                                     execution_profile=recipe["analyzer"].get("execution_profile"))
        if receipt is None:
            job = {"cmd": "bgm_analysis", "analysis_path": str(analysis.resolve()),
                   "analysis_sha256": preparation["sha256"], "analyzer_id": recipe["analyzer_id"]}
            if "execution_profile" in recipe["analyzer"]:
                job["execution_profile"] = recipe["analyzer"]["execution_profile"]
            if analyze is None:
                from lss.core import worker
                result = worker.handle_bgm_analysis_job(job, destination_dir=stems_dir)
            else:
                result = analyze(job, stems_dir)
            if not isinstance(result, dict) or result.get("status") != "ok":
                raise _BgmAnalysisFailed(result if isinstance(result, dict) else {
                    "status": "error", "error_code": "invalid_worker_response",
                    "message": "BGM解析ワーカーから不正な応答が返りました"})
            receipt = _verified_analysis(stems_dir / "receipt.json", preparation["sha256"],
                                         recipe["analyzer_id"],
                                         execution_profile=recipe["analyzer"].get("execution_profile"))
            if receipt is None:
                raise ValueError("BGM解析の結果が受領票と一致しません")
        stems = {stem: Path(receipt["stems"][stem]["path"]) for stem in episode_cues.STEMS}
        clock = mark("sfc_first_pass", clock)
        second_receipts: list[dict] = []

        def refine(blocks: list[int]) -> dict:
            """効果音の混入が強そうな候補を含むブロックだけ、1回目の音楽の出力をもう一度解析する。"""
            first_music = stems["music"].resolve()
            first_sha = file_sha256(first_music)
            second_dir = (scratch / "sfc-second").resolve()
            second = _verified_analysis(second_dir / "receipt.json", first_sha, recipe["analyzer_id"],
                                        blocks=blocks,
                                        execution_profile=recipe["analyzer"].get("execution_profile"))
            if second is None:
                job = {"cmd": "bgm_analysis", "analysis_path": str(first_music),
                       "analysis_sha256": first_sha, "analyzer_id": recipe["analyzer_id"],
                       "blocks": blocks, "input_role": "first_pass_music"}
                if "execution_profile" in recipe["analyzer"]:
                    job["execution_profile"] = recipe["analyzer"]["execution_profile"]
                if analyze is None:
                    from lss.core import worker
                    result = worker.handle_bgm_analysis_job(job, destination_dir=second_dir)
                else:
                    result = analyze(job, second_dir)
                if not isinstance(result, dict) or result.get("status") != "ok":
                    raise _BgmAnalysisFailed(result if isinstance(result, dict) else {
                        "status": "error", "error_code": "invalid_worker_response",
                        "message": "BGM解析ワーカーから不正な応答が返りました"})
                second = _verified_analysis(second_dir / "receipt.json", first_sha, recipe["analyzer_id"],
                                            blocks=blocks,
                                            execution_profile=recipe["analyzer"].get("execution_profile"))
                if second is None:
                    raise ValueError("BGMの2回目の解析の結果が受領票と一致しません")
            second_receipts.append(second)
            return {"music": second["stems"]["music"]["path"], "analysis_id": recipe["analyzer_id"],
                    "runtime_id": second["runtime_id"]}

        vocal_recipe = vocal_recovery.default_recipe()
        four_stem_receipts: dict[str, dict] = {}

        def recover(input_role: str, signal: Path, spans: list) -> dict:
            """候補区間だけを4分類へ通す（歌唱回収）。同じ信号・区間の完了済み結果は使い回す。"""
            signal = Path(signal).resolve()
            signal_sha = file_sha256(signal)
            folder = (scratch / f"four-stem-{input_role}").resolve()
            spans = [list(span) for span in spans]
            done = _verified_four_stem(folder / "receipt.json", signal_sha, recipe["four_stem_analyzer_id"], spans)
            if done is None:
                job = {"cmd": "bgm_four_stem", "signal_path": str(signal), "signal_sha256": signal_sha,
                       "analyzer_id": recipe["four_stem_analyzer_id"], "spans": spans,
                       "hop_seconds": vocal_recipe.hop_seconds, "roles": ["music_vocals"],
                       "input_role": input_role}
                if analyze is None:
                    from lss.core import worker
                    result = worker.handle_bgm_four_stem_job(job, destination_dir=folder)
                else:
                    result = analyze(job, folder)
                if not isinstance(result, dict) or result.get("status") != "ok":
                    raise _BgmAnalysisFailed(result if isinstance(result, dict) else {
                        "status": "error", "error_code": "invalid_worker_response",
                        "message": "BGM解析ワーカーから不正な応答が返りました"})
                done = _verified_four_stem(folder / "receipt.json", signal_sha, recipe["four_stem_analyzer_id"],
                                           spans)
                if done is None:
                    raise ValueError("BGMの4分類の結果が受領票と一致しません")
            four_stem_receipts[input_role] = done
            return {"stems": done["stems"]["music_vocals"], "analysis_id": done["analyzer_id"],
                    "runtime_id": done["runtime_id"]}

        # 区間計画 → 公開する区間と表現の判定 → 処理済みで公開する区間の保留候補だけ磨き直す。
        plan = episode_cues.plan_episode_cues(stems, episode_cues.default_recipe())
        clock = mark("candidate_plan", clock)
        origin = episode_cues.CueOrigin(
            source_path=source, source_sha256=source_sha256,
            source_id=f"source-{source_sha256[:16]}",
            analysis_id=recipe["analyzer_id"], runtime_id=receipt["runtime_id"],
            time_offset_numerator=timeline["offset_numerator"],
            time_offset_denominator=timeline["offset_denominator"])
        # The first-pass music cut and the raw cut of every held cue depend only on the first-pass stems and the
        # decode, which no later GPU stage changes. Write them on the CPU lane while event tagging, the second pass
        # and the four-stem recovery run; `run_episode_cues` then reuses each verified file (the same bytes).
        lane = overlap.open_lane()
        try:
            if lane is not None:
                _prefetch_final_candidates(lane, plan, stems, decoded, run_dir, origin)
            decision = None
            if any(cue["disposition"] == episode_cues.DISPOSITION_HELD for cue in plan["cues"]):
                decision = _bgm_decide(source, recipe, run_dir, scratch, analyze, stems, analysis, plan, references,
                                       source_sha256=source_sha256, native_rate=info.samplerate,
                                       audio_timeline_origin=timeline)
                timing["event_tagging"] = decision["tagging_seconds"]
            clock = mark("semantics_and_routing", clock)
            record_path = episode_cues.run_episode_cues(
                analysis, stems, run_dir, episode_cues.default_recipe(),
                origin=origin,
                raw_source=decoded,
                provenance={
                    "analysis_copy": {name: preparation[name] for name in
                                      ("decoded_rate", "analysis_rate", "resample", "channels",
                                       "frames", "sha256")},
                    "raw_cut_from": "product input preparation (decoded at the native rate)",
                    "analysis_retained": False,
                },
                second_pass_recipe=second_pass.default_recipe(), refine=refine,
                vocal_recovery_recipe=vocal_recipe, recover=recover, work_dir=scratch,
                plan=plan, refine_cue_ids=decision["refine_cue_ids"] if decision else [],
            )
            if lane is not None:
                lane.close(failed=False)
                _retire_unused_prefetch(lane, record_path)
                # CPU seconds the lane spent while other stages ran (wall-clock stages no longer add up to total).
                timing["overlap_candidate_cpu"] = round(lane.busy_seconds, 2)
        except BaseException:
            if lane is not None:
                lane.close(failed=True)
            raise
        clock = mark("refinement_and_candidates", clock)
        if file_sha256(source) != source_sha256:
            raise session.InputChangedError("BGM候補の作成中に原本が変更されました")
    cue_record = json.loads(record_path.read_text(encoding="utf-8"))
    held = []
    for cue in cue_record["plan"]["cues"]:
        if cue["disposition"] != episode_cues.DISPOSITION_HELD:
            continue
        artifacts = cue["artifacts"]
        refinement_decision = cue.get("second_pass") or {}
        entry = {
            "cue_id": cue["cue_id"],
            "start_seconds": cue["start_seconds"], "end_seconds": cue["end_seconds"],
            # 公開cueは保留候補の音声を連結して作る。その材料（区間・縮小・音の同一性）も持つ。
            "music": {name: artifacts["separator_music"][name]
                      for name in ("path", "sha256", "candidate_asset_id", "audio", "source_start_frame",
                                   "source_end_frame", "sample_rate", "serialization")
                      if name in artifacts["separator_music"]},
            "raw": {name: artifacts["raw"][name]
                    for name in ("path", "sha256", "candidate_asset_id", "audio", "source_start_frame",
                                 "source_end_frame", "sample_rate", "serialization")
                    if name in artifacts["raw"]},
            "refined": cue.get("refinement") != "not_routed",
            "second_pass": {"eligible": bool((refinement_decision.get("eligibility") or {}).get("eligible")),
                            "adopted": bool(refinement_decision.get("adopted"))},
            "vocal_recovery": {"eligible": bool(((cue.get("vocal_recovery") or {}).get("eligibility") or {})
                                                .get("eligible")),
                               "adopted": bool((cue.get("vocal_recovery") or {}).get("adopted"))},
        }
        if "separator_music_first_pass" in artifacts:
            entry["music_first_pass"] = {name: artifacts["separator_music_first_pass"][name]
                                         for name in ("path", "sha256", "candidate_asset_id", "audio",
                                                      "source_start_frame", "source_end_frame", "sample_rate", "serialization")
                                         if name in artifacts["separator_music_first_pass"]}
        if "separator_music_before_vocal_recovery" in artifacts:
            entry["music_before_vocal_recovery"] = {
                name: artifacts["separator_music_before_vocal_recovery"][name]
                for name in ("path", "sha256", "candidate_asset_id")}
        held.append(entry)
    refinement = cue_record.get("second_pass") or {}
    vocal = cue_record.get("vocal_recovery") or {}
    semantics, published = None, []
    clock = time.perf_counter()
    if held:
        semantics, published = _bgm_publish(source, source_sha256, recipe, run_dir, held, timeline, decision,
                                            analysis=analysis, analyze=analyze, overlap_timing=timing)
    clock = mark("publication", clock)
    run = {
        **base,
        "status": (_BGM_STATUS_PUBLISHED if published else _BGM_STATUS_HELD) if held else _BGM_STATUS_NO_CANDIDATE,
        "cue_record": {"name": record_path.name, "sha256": file_sha256(record_path)},
        "analysis": {"analyzer_id": recipe["analyzer_id"], "runtime_id": receipt["runtime_id"],
                     "runtime_revision": receipt["runtime_revision"],
                     "weights_sha256": receipt["analyzer"]["weights"]["sha256"],
                     "analysis_sha256": preparation["sha256"],
                     "stems": receipt["stems"], "retained": False,
                     "elapsed_seconds": receipt.get("elapsed_seconds"), "peak_vram_mb": receipt.get("peak_vram_mb")},
        "second_pass": {
            "recipe_id": refinement.get("recipe_id"), "status": refinement.get("status"),
            "eligible_count": refinement.get("eligible_count", 0),
            "adopted_count": refinement.get("adopted_count", 0),
            "blocks": refinement.get("blocks"),
            "runtime_id": refinement.get("runtime_id"),
            "runtime_revision": second_receipts[-1]["runtime_revision"] if second_receipts else None,
            "signal_sha256": (refinement.get("signal") or {}).get("sha256"),
            "stems": second_receipts[-1]["stems"] if second_receipts else None,
            "elapsed_seconds": second_receipts[-1].get("elapsed_seconds") if second_receipts else None,
            "peak_vram_mb": second_receipts[-1].get("peak_vram_mb") if second_receipts else None,
            "retained": False,
        },
        "vocal_recovery": {
            "recipe_id": vocal.get("recipe_id"), "status": vocal.get("status"),
            "eligible_count": vocal.get("eligible_count", 0), "worth_count": vocal.get("worth_count", 0),
            "adopted_count": vocal.get("adopted_count", 0),
            "four_stem_analyzer_id": recipe["four_stem_analyzer_id"],
            "signal_sha256": (vocal.get("signal") or {}).get("sha256"),
            "runs": {role: {"input_role": role, "signal_sha256": item["signal"]["sha256"],
                            "spans": item["spans"], "runtime_id": item["runtime_id"],
                            "runtime_revision": item["runtime_revision"],
                            "weights_sha256": item["analyzer"]["weights"]["sha256"],
                            "stems": item["stems"], "elapsed_seconds": item.get("elapsed_seconds"),
                            "peak_vram_mb": item.get("peak_vram_mb")}
                     for role, item in sorted(four_stem_receipts.items())},
            "retained": False,
        },
        "held": held,
        "claim_boundary": cue_record["claim_boundary"],
    }
    if held:
        run["semantics"] = semantics
        run["published"] = published
        run["publication_state"] = "published_wanted_cues" if published else "not_published"
    # 重い処理へ実際に入れた秒数と、Stageごとの所要（来歴として、どこへ計算を使ったかを残す）。
    duration = preparation["frames"] / sfc_analysis.SAMPLE_RATE
    block_seconds = sfc_analysis.INFERENCE["block_seconds"]
    second_seconds = min(duration, len(refinement.get("blocks") or []) * block_seconds)
    four_stem_seconds = {role: round(sum(b - a for a, b in item["spans"]) / sfc_analysis.SAMPLE_RATE, 1)
                         for role, item in four_stem_receipts.items()}
    timing["total"] = round(time.perf_counter() - started, 2)
    run["cost"] = {
        "source_seconds": round(duration, 1),
        "heavy_input_seconds": {"sfc_first_pass": round(duration, 1), "sfc_second_pass": round(second_seconds, 1),
                                "four_stem": four_stem_seconds,
                                "event_tagging": round(2 * duration + decision.get("stereo_tagging_input_seconds", 0), 1) if decision else 0.0},
        "refined_candidate_seconds": round(sum(c["end_seconds"] - c["start_seconds"] for c in held if c["refined"]), 1),
        "held_candidate_seconds": round(sum(c["end_seconds"] - c["start_seconds"] for c in held), 1),
        "stage_seconds": {name: value for name, value in timing.items() if name != "publication_parts"},
        # 公開の工程の内訳（伴奏model・組み立て・書き出し・仕上げ）。重なって動くので和は公開の秒と一致しない。
        **({"publication_parts": timing["publication_parts"]} if "publication_parts" in timing else {}),
    }
    if references[1] is not None:
        # 参照録音を使ったrun。使った内容の集まり・出所・方針の識別子（runの識別子に入れたもの）を残す。
        policy = _bgm_reference_policy()
        run["references"] = {"digest": references[1], "count": len(references[0]),
                             "missing": list(references[2]), "sources": reference_sources or {},
                             "policy_id": policy["policy_id"], "policy": policy}
        if decision and decision.get("reference_relations") is not None:
            # 参照との照合に使った時間（指紋の索引・復号・照合・関係の判定）。重い推論は使わない。
            run["cost"]["reference_gate_seconds"] = dict(decision["reference_relations"].get("timing") or {})
    # 解析用の複製とstemは再生成できる作業物。記録にはsha256だけを残す。
    used = [(stems_dir / "receipt.json", receipt),
            *((scratch / "sfc-second" / "receipt.json", item) for item in second_receipts),
            *((scratch / f"four-stem-{role}" / "receipt.json", item) for role, item in four_stem_receipts.items())]
    if decision:
        used += [(scratch / "event-tags" / "receipt.json", decision["music_tags"]),
                 (scratch / "event-tags-mixture" / "receipt.json", decision["mixture_tags"])]
        used += [(Path(path), item) for path, item in decision.get("stereo_scratch_receipts", [])]
    run["analysis_scratch"] = _retire_analysis_scratch(scratch, source_sha256, cue_record, preparation=preparation,
                                                       receipts=used, keep=(source,))
    return run


def _prefetch_final_candidates(lane, plan: dict, stems: dict, raw_source: Path, run_dir: Path, origin) -> None:
    """Submit, in plan order, the candidate cuts whose audio no later GPU stage can change.

    Every held cue gets its first-pass music cut (``<stem>-music.flac``; the published music when the second pass
    and vocal recovery are not adopted, ``separator_music_first_pass`` otherwise) and its raw cut. The arguments are
    exactly those of the serial loop in `lss.bgm.episode_cues.run_episode_cues`: candidate_store keys the file by
    name and writes the evidence into it, so any difference would be a different file. The serial loop then finds
    the finished file and takes the verified-reuse path of `candidate_store.materialize_candidate`, which returns
    the same record. If the serial loop reaches a cut first, both writers produce identical bytes and the rename
    keeps one of them (`materialize_candidate` already tolerates a concurrent writer).
    """
    from lss.bgm import episode_cues

    paths = {stem: Path(stems[stem]).resolve() for stem in episode_cues.STEMS}
    raw = Path(raw_source).resolve()
    audio_dir = Path(run_dir).resolve() / "candidates"
    # Hash both full-length signals here, once: the tasks then hit candidate_store's hash cache.
    music_sha = episode_cues._sha256(paths["music"])
    raw_sha = episode_cues._sha256(raw)

    def cut(signal, signal_sha256, start, end, target, evidence):
        target = Path(target)
        existed = target.exists()
        record = episode_cues._materialize_cut(signal, signal_sha256, start, end, target,
                                               origin=origin, recipe_id=plan["recipe_id"], evidence=evidence)
        if not existed:
            lane.created(target)
        return record

    for cue in plan["cues"]:
        if cue["disposition"] != episode_cues.DISPOSITION_HELD:
            continue
        stem = f"{cue['cue_id']}-{cue['start_seconds']:.0f}s"
        measured = {key: cue[key] for key in ("cue_id", "active_fraction", "median_share",
                                              "music_over_speech_db", "music_over_effects_db")}
        span = (cue["start_seconds"], cue["end_seconds"])
        lane.submit(f"{stem}-music", cut, paths["music"], music_sha, *span, audio_dir / f"{stem}-music.flac",
                    {"method": episode_cues.METHOD, "representation": "separator_music_analysis_signal",
                     **measured})
        lane.submit(f"{stem}-raw", cut, raw, raw_sha, *span, audio_dir / f"{stem}-raw.flac",
                    {"method": episode_cues.METHOD, "representation": "original_mixture_no_op",
                     "detected_from_signal_sha256": music_sha, **measured})


def _retire_unused_prefetch(lane, record_path: Path) -> None:
    """A prefetched cut that the completed record does not list would be a file the serial path never wrote."""
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    listed = {str(Path(artifact["path"]).resolve()) for cue in record["plan"]["cues"]
              for artifact in (cue.get("artifacts") or {}).values()}
    for path in lane.created_files():
        if str(path.resolve()) not in listed:
            path.unlink(missing_ok=True)


def _receipt_items(receipt: dict) -> list[dict]:
    """受領票が挙げる出力（``{"path", "sha256", ...}``）。知らない形は何も挙げないものとして扱う。"""
    kind = receipt.get("type")
    if kind == "bgm_four_stem_receipt":
        stems = receipt.get("stems")
        items = stems.get("music_vocals") if isinstance(stems, dict) else None
    else:
        listed = receipt.get({"bgm_sfc_analysis_receipt": "stems", "bgm_event_tags_receipt": "outputs",
                              "bgm_stereo_projection_receipt": "outputs"}.get(kind, ""))
        items = list(listed.values()) if isinstance(listed, dict) else None
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _owned_analysis_files(scratch: Path, source_sha256: str, cue_record: dict, *, preparation: dict | None = None,
                          receipts: Iterable[tuple[Path, dict]] = ()) -> tuple[dict[Path, str], dict[Path, list[Path]]]:
    """解析の作業領域で、このrunが書いた・使ったと自分の記録で確かめられる物。

    返すのは（作業物 → 記録のsha256、記録のファイル → それが挙げる作業物）。根拠にするのは、このrunが使った記録
    だけ：解析用の複製の準備の記録（``preparation``。同じ原本の物）、このrunが検証して使ったworkerの受領票
    （``receipts``：パスと、その時に読んだ内容）、歌唱回収の全長の信号を挙げるこのrunの区間の記録。作業領域に
    あるというだけの記録（別の原本・別の試行・形の分からない物）や、フォルダの名前、記録の無い物は所有の根拠に
    しない。記録のファイルも、このrunが使った内容のままの時だけ自分の物とする。
    """
    root = scratch.resolve()
    artifacts: dict[Path, str] = {}
    records: dict[Path, list[Path]] = {}

    def claim(path, digest) -> Path | None:
        if not isinstance(path, str) or not isinstance(digest, str):
            return None
        path = Path(path).resolve()
        if not path.is_relative_to(root) or path == root:
            return None
        artifacts[path] = digest
        return path

    def unchanged(path: Path, used: dict) -> bool:
        try:
            return json.loads(path.read_text(encoding="utf-8")) == used
        except (OSError, ValueError):
            return False

    if isinstance(preparation, dict) and preparation.get("source_sha256") == source_sha256:
        claimed = claim(str(root / "analysis.wav"), preparation.get("sha256"))
        if unchanged(root / "analysis-preparation.json", preparation):
            records[root / "analysis-preparation.json"] = [claimed] if claimed else []
    for receipt_path, receipt in receipts:
        receipt_path = Path(receipt_path).resolve()
        folder = receipt_path.parent
        if not isinstance(receipt, dict) or folder == root or not folder.is_relative_to(root):
            continue
        claimed = [path for path in (claim(item.get("path"), item.get("sha256")) for item in _receipt_items(receipt)
                                     if Path(str(item.get("path", ""))).resolve().parent == folder) if path]
        if unchanged(receipt_path, receipt):
            records[receipt_path] = claimed
    signal = ((cue_record or {}).get("vocal_recovery") or {}).get("signal") if isinstance(cue_record, dict) else None
    if isinstance(signal, dict):
        claim(signal.get("path"), signal.get("sha256"))
    return artifacts, records


def _retire_analysis_scratch(scratch: Path, source_sha256: str, cue_record: dict, *, preparation: dict | None = None,
                             receipts: Iterable[tuple[Path, dict]] = (), keep: Iterable[Path] = ()) -> dict:
    """runを書き終えた解析の作業領域を片付け、行ったことを返す（runへ記録する）。

    消すのは、このrunの記録で確かめられ（`_owned_analysis_files`）、中身が記録のsha256と一致する作業物と、挙げる
    作業物がもう無い記録だけ。作業領域の名前は決まっていて、前の試行の物も、処理中に置かれた利用者の物も入りうる
    ので、それ以外（記録に無い物・中身の変わった物とそれを説明する記録・link・``keep``（原本））は場所もbytesも
    そのまま残し、空になったフォルダだけを消す。
    """
    import logging
    import stat

    def link(path: Path) -> bool:
        try:
            return path.is_symlink() or bool(
                getattr(path.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except OSError:
            return True

    report = {"removed_files": 0, "removed_bytes": 0, "kept": []}
    if not scratch.is_dir() or link(scratch):
        return report
    root = scratch.resolve()
    artifacts, records = _owned_analysis_files(scratch, source_sha256, cue_record, preparation=preparation,
                                               receipts=receipts)
    protected = {Path(path).resolve() for path in keep}
    found = []
    for path in sorted(scratch.rglob("*")):
        if path.is_dir() and not link(path):
            continue
        resolved = path.resolve()
        if (link(path) or not path.is_file() or not resolved.is_relative_to(root) or resolved in protected
                or (resolved not in artifacts and resolved not in records)):
            report["kept"].append(path.relative_to(scratch).as_posix())
            continue
        found.append((path, resolved))

    def remove(path: Path) -> None:
        size = path.stat().st_size
        path.unlink()
        report["removed_files"] += 1
        report["removed_bytes"] += size

    for path, resolved in found:  # 作業物を先に。中身が記録と違う物は残す
        if resolved in artifacts:
            if file_sha256(path) == artifacts[resolved]:
                remove(path)
            else:
                report["kept"].append(path.relative_to(scratch).as_posix())
    for path, resolved in found:  # 記録は、挙げる作業物が残っていない時だけ（残した物の来歴を失わない）
        if resolved in records and resolved not in artifacts:
            if any(claimed.exists() for claimed in records[resolved]):
                report["kept"].append(path.relative_to(scratch).as_posix())
            else:
                remove(path)
    report["kept"].sort()
    for folder in sorted((p for p in scratch.rglob("*") if p.is_dir() and not link(p)),
                         key=lambda p: len(p.parts), reverse=True):
        if folder.resolve().is_relative_to(root):
            try:
                folder.rmdir()  # 空になったフォルダだけ（linkは消さない）
            except OSError:
                pass
    try:
        scratch.rmdir()
    except OSError:
        pass
    if report["kept"]:
        logging.getLogger(__name__).warning(
            "解析の作業領域に、LSSの物と確かめられないファイルを残しました: %s (%d件)", scratch, len(report["kept"]))
    return report


def _verified_event_tags(receipt_path: Path, signal_sha256: str, analyzer_id: str) -> dict | None:
    """タグ付けの受領票が、同じ信号・同じ解析器の完全な結果ならそれを返す（中断再開と再実行で使い回す）。"""
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("type") != "bgm_event_tags_receipt" or receipt.get("analyzer_id") != analyzer_id
                or (receipt.get("signal") or {}).get("sha256") != signal_sha256
                or not isinstance(receipt.get("runtime_id"), str)):
            return None
        folder = receipt_path.parent.resolve()
        for name in ("probabilities", "embeddings"):
            item = (receipt.get("outputs") or {}).get(name) or {}
            path = Path(str(item.get("path", ""))).resolve()
            if path.parent != folder or file_sha256(path) != item.get("sha256"):
                return None
        return receipt
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _lss_version() -> str:
    try:
        from lss.paths import resource_root
        return (resource_root() / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return "不明"


def _tag_job(signal: Path, folder: Path, recipe: dict, analyze, input_role: str) -> dict:
    """信号を常駐workerのタグ付けへ通す（同じ信号・同じ解析器の完了済み結果は使い回す）。受領票を返す。"""
    signal = Path(signal).resolve()
    signal_sha = file_sha256(signal)
    folder = Path(folder).resolve()
    receipt = _verified_event_tags(folder / "receipt.json", signal_sha, recipe["event_tagger_id"])
    if receipt is None:
        job = {"cmd": "bgm_event_tags", "signal_path": str(signal), "signal_sha256": signal_sha,
               "analyzer_id": recipe["event_tagger_id"], "input_role": input_role}
        if analyze is None:
            from lss.core import worker
            result = worker.handle_bgm_event_tags_job(job, destination_dir=folder)
        else:
            result = analyze(job, folder)
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise _BgmAnalysisFailed(result if isinstance(result, dict) else {
                "status": "error", "error_code": "invalid_worker_response",
                "message": "BGM解析ワーカーから不正な応答が返りました"})
        receipt = _verified_event_tags(folder / "receipt.json", signal_sha, recipe["event_tagger_id"])
        if receipt is None:
            raise ValueError("BGMのタグ付けの結果が受領票と一致しません")
    return receipt


def _bgm_stereo_continuations(candidates, window, semantics_recipe, music, scratch, evidence_dir, recipe, analyze):
    """分類器が途切れた短い区間だけ、左右差に残る音楽を追加で測る。音声は置き換えない。"""
    import shutil
    import tempfile
    import numpy as np
    from lss.bgm import event_tagging, stereo_continuity

    records, receipts = [], []
    seconds = 0
    for index, plan in enumerate(stereo_continuity.plan_holes(candidates, window, semantics_recipe)):
        folder = Path(tempfile.mkdtemp(prefix="stereo-continuity-", dir=scratch))
        try:
            projection = stereo_continuity.project(music, plan, folder / "projection")
            receipts.append((str(folder / "projection" / "receipt.json"), projection))
            tagged = _tag_job(Path(projection["outputs"]["signal"]["path"]), folder / "tags", recipe, analyze,
                              "first_pass_music_stereo_difference")
            receipts.append((str(folder / "tags" / "receipt.json"), tagged))
            probabilities, embeddings = event_tagging.load_outputs(tagged["outputs"])
            measured = stereo_continuity.evaluate(plan, projection, probabilities, embeddings, semantics_recipe)
            target = evidence_dir / f"stereo-continuity-{index}"
            target.mkdir(parents=True, exist_ok=True)
            outputs = {}
            for name in ("probabilities", "embeddings"):
                path = target / f"{name}.npy"
                shutil.copyfile(tagged["outputs"][name]["path"], path)
                outputs[name] = {"path": str(path), "sha256": file_sha256(path)}
            measured["evidence"] = {"projection": projection, "tagging": tagged, "outputs": outputs}
            records.append(measured)
            seconds += probabilities.shape[0]
        except Exception as exc:  # noqa: BLE001 - stereo continuity is an optional amplifier; BaseException still propagates
            # 補助経路が測れなくても、従来の単話回収はそのまま使える。
            records.append({"state": "stereo_continuation_not_checked", "span": plan["gap_seconds"],
                            "error": type(exc).__name__, "identity_authority": "none", "absence_authority": "none"})
    return records, receipts, seconds


def _bgm_memory_support(analysis: Path, music: Path, segments: list[dict], run_dir: Path, source: Path,
                        candidates=(), *, speech: Path | None = None, effects: Path | None = None) -> dict:
    """Component records from the published-cue memory (`lss.bgm.memory_support`); never raises.

    Without memory, or when it cannot be read, the receipt says so and holds no records: the episode is decided
    exactly as without it.
    """
    from lss.bgm import identity, memory_support

    try:
        from lss.bgm import cue_memory

        state_root = run_dir.parents[2]
        if not cue_memory.read(state_root).get("members"):
            return {"state": "no_memory", "records": [], "policy": memory_support.revision()}
        source_sha256 = file_sha256(source)
        query = identity.decode_mono(Path(analysis))
        music_samples = identity.decode_mono(Path(music))
        size = min(len(query), len(music_samples))
        from lss.bgm import cue_semantics

        stretches = [(float(low), float(high)) for low, high, _ in cue_semantics._stretches(list(candidates))]
        def stereo(start: int, frames: int):
            return (identity.decode_stereo_window(Path(analysis), start, frames),
                    identity.decode_stereo_window(Path(music), start, frames))

        def interference():
            result = {}
            for name, path in (("speech", speech), ("effects", effects)):
                if path is None:
                    continue
                try:
                    result[name] = identity.decode_mono(Path(path))
                except (OSError, ValueError, RuntimeError):
                    result[name] = None
            return result

        return memory_support.nominate(query[:size], music_samples[:size], segments, state_root, source_sha256,
                                       stretches=stretches, source_name=Path(source).stem, stereo=stereo,
                                       query_interference_loader=interference)
    except Exception as exc:  # noqa: BLE001 - memory is an optional amplifier; failure keeps the local result
        return {"state": "not_checked", "error": type(exc).__name__, "records": [],
                "policy": memory_support.revision()}


def _bgm_decide(source: Path, recipe: dict, run_dir: Path, scratch: Path, analyze, stems: dict, analysis: Path,
                plan: dict, references, *, source_sha256: str | None = None, native_rate: int | None = None,
                audio_timeline_origin: dict | None = None) -> dict:
    """保留候補を曲単位へ分け直して公開する区間を決め、公開する区間ごとに表現（原音か処理済みか）を選ぶ。

    2回目・歌唱回収の前に行う。磨き直しが要るのは、処理済みの表現で公開する区間の保留候補だけになる。
    判定に使った窓ごとの観測（タグの確率・埋め込み・SFCの音楽の大きさ・原本との差・原本のタグ）はrunへ残す。
    """
    import shutil

    import numpy as np

    from lss.bgm import cue_formation, cue_representation, cue_scope, cue_semantics, episode_cues, episode_usage, event_tagging
    from lss.bgm import memory_support

    semantics_recipe = cue_semantics.default_recipe()
    routing_recipe = cue_representation.default_recipe()
    if (semantics_recipe.recipe_id != recipe["cue_semantics_id"]
            or routing_recipe.recipe_id != recipe["cue_representation_id"]):
        raise ValueError("BGMの意味判定・表現の宣言がRecipeと一致しません")
    started = time.perf_counter()
    music_tags = _tag_job(stems["music"], scratch / "event-tags", recipe, analyze, "first_pass_music")
    mixture_tags = _tag_job(analysis, scratch / "event-tags-mixture", recipe, analyze, "mixture_analysis_copy")
    tagging_seconds = time.perf_counter() - started
    event_tagging.load_outputs(music_tags["outputs"])
    event_tagging.load_outputs(mixture_tags["outputs"])
    _, levels, _ = episode_cues.window_shares(stems, 1.0)
    residual = cue_representation.residual_levels(str(analysis), str(stems["music"]))
    evidence_dir = run_dir / "semantics"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for name, source_path in (("probabilities", music_tags["outputs"]["probabilities"]["path"]),
                              ("embeddings", music_tags["outputs"]["embeddings"]["path"]),
                              ("mixture-probabilities", mixture_tags["outputs"]["probabilities"]["path"])):
        target = evidence_dir / f"{name}.npy"
        shutil.copyfile(source_path, target)
        evidence[name.replace("-", "_")] = {"name": target.name, "sha256": file_sha256(target)}
    for name, values in (("music-levels", levels), ("residual-levels", residual)):
        target = evidence_dir / f"{name}.npy"
        np.save(target, np.asarray(values, dtype=np.float32))
        evidence[name.replace("-", "_")] = {"name": target.name, "sha256": file_sha256(target)}
    window = cue_semantics.WindowEvidence.from_arrays(
        np.load(evidence_dir / "music-levels.npy"), np.load(evidence_dir / "probabilities.npy"),
        np.load(evidence_dir / "embeddings.npy"))
    routing_evidence = cue_representation.RoutingEvidence.from_arrays(
        np.load(evidence_dir / "residual-levels.npy"), np.load(evidence_dir / "mixture-probabilities.npy"))
    reference_paths, references_digest, references_missing, fingerprint_cache = references
    held = [cue for cue in plan["cues"] if cue["disposition"] == episode_cues.DISPOSITION_HELD]
    candidates = [(cue["start_seconds"], cue["end_seconds"]) for cue in held]
    stereo_started = time.perf_counter()
    stereo_continuations, stereo_receipts, stereo_seconds = _bgm_stereo_continuations(
        candidates, window, semantics_recipe, stems["music"], scratch, evidence_dir, recipe, analyze)
    tagging_seconds += time.perf_counter() - stereo_started
    stereo_music = np.zeros(window.count, dtype=bool)
    for continuation in stereo_continuations:
        if continuation.get("state") == "stereo_music_continuation_observed":
            stereo_music[continuation["positive_music_seconds"]] = True
    segments, formation = cue_formation.segment(candidates, window, semantics_recipe, music=stems["music"],
                                               stereo_continuations=stereo_continuations)
    known, relations, gate = [], None, None
    component_music = stereo_music.copy()
    # LSS's own published cues of other sources (the published-cue memory) measure, in this episode's own source
    # and music, how a recording heard here goes on past what the classifier heard. Same component records as a
    # released reference, same cue-formation rules; no table, no donor, no EXACT. Failure keeps the result above.
    memory_records = _bgm_memory_support(analysis, stems["music"], segments, run_dir, source, candidates,
                                         speech=stems.get("speech"), effects=stems.get("effects"))
    formation_memory = {k: v for k, v in memory_records.items() if k != "records"}
    first_pass = (segments, formation)

    def memory_formation():
        """Cue formation with the memory's component records only; on any failure, the first-pass result."""
        if not memory_records["records"]:
            return (*first_pass, stereo_music.copy())
        try:
            from lss.bgm import recording_support

            components = {"records": memory_records["records"], "onsets": [], "memory_support": formation_memory}
            parts, formed = cue_formation.segment(candidates, window, semantics_recipe, music=stems["music"],
                                                  components=components, stereo_continuations=stereo_continuations)
            music_windows = recording_support.confirmed_music_windows(components, window.count) | stereo_music
            parts = memory_support.never_lower(first_pass[0], parts, window, semantics_recipe,
                                               additional_music_windows=music_windows, baseline_music_windows=stereo_music,
                                               observations=formed["observations"])
            parts = memory_support.join_covered(parts, memory_records["records"], window, semantics_recipe,
                                                additional_music_windows=music_windows,
                                                observations=formed["observations"])
            # The memory's music windows act only on parts memory moved, and the whole result is checked once more
            # against the first pass with the windows the final decision will use.
            parts, music_windows = memory_support.final_check(first_pass[0], parts, music_windows, window,
                                                              semantics_recipe, baseline_music_windows=stereo_music,
                                                              observations=formed["observations"])
            return parts, formed, music_windows
        except Exception as exc:  # noqa: BLE001 - memory is an optional amplifier
            formation_memory.update(state="formation_failed_first_pass_kept", error=type(exc).__name__)
            return (*first_pass, stereo_music.copy())

    if not reference_paths:
        segments, formation, component_music = memory_formation()
    if reference_paths:
        from lss.bgm import reference_relations, reference_store

        relation_recipe = reference_relations.default_recipe()
        store_root = Path(fingerprint_cache).parent
        shas = [entry["sha256"] for entry in reference_paths]
        try:
            # 参照の知識（指紋・和声）から照合の索引を作る。重い音源は波形の確認の時だけ読み、退役していれば
            # 取得の台帳から取り直す（取り直せない参照の整列は根拠にしない）。使用中の参照音源は退役させない。
            # 区間を参照との関係ごとに分け、一つの録音の一つのずれが音楽を説明し尽くす片だけを既発売にする
            # （2回目・歌唱回収・台詞の残りの除去・公開の前に決めるので、その片には重い処理を使わない）。
            # 音楽の会計にはMusic stemと、台詞・効果音のstem（Musicへの漏れを差し引く材料）を使う。
            with reference_store.lease(shas, store_root):
                fetched: list[str] = []
                library, skipped = reference_store.build_library(
                    list(reference_paths), cache_dir=fingerprint_cache, state_root=store_root,
                    resolver=reference_store.MediaResolver(state_root=store_root, fetched=fetched, strict=False))
                gate = reference_relations.Gate.prepare(analysis, cache_dir=fingerprint_cache, library=library,
                                                        music=stems["music"],
                                                        interference=[stems["speech"], stems["effects"]],
                                                        recipe=relation_recipe)
                known = gate.known_spans()
                usage = episode_usage.nominate(source, [m.to_dict() for m in gate.matches],
                                              reference_metadata=library.metadata)
                from lss.bgm import recording_support, identity
                components = recording_support.nominate(gate.query, library, usage, identity.SAMPLE_RATE,music=gate.music,
                                                       source_path=analysis,music_path=stems["music"])
                from lss.bgm import occurrence_support
                source_hash = file_sha256(source)
                occurrence = occurrence_support.nominate(gate.query,gate.music,usage,window,source_hash,run_dir.parents[2])
                components["records"].extend(occurrence["records"])
                components["occurrence_memory"]={k:v for k,v in occurrence.items() if k!="records"}
                components["onsets"]=usage.get("onsets",[])
                # 参照の成分とメモリの成分は混ぜない（参照の規則にメモリが割り込んで結果を下げない）。
                # メモリは、結果を下げない時だけ働く一続きの録音の再結合（join_covered）に使う。
                components["memory_support"]=formation_memory
                try:
                    components["occurrence_memory"]["capture"]=occurrence_support.remember(
                        source,source_hash,usage,window,run_dir.parents[2])
                except (OSError,ValueError):
                    components["occurrence_memory"]["capture"]={"state":"memory_unavailable_local_route_retained"}
                component_music = recording_support.confirmed_music_windows(components,window.count) | stereo_music
                segments, formation = cue_formation.segment(
                    candidates, window, semantics_recipe, usage=usage,
                    matches=[m.to_dict() for m in gate.matches], music=stems["music"], components=components,
                    mixture_music=np.load(evidence_dir/"mixture-probabilities.npy")[:,cue_semantics.MUSIC_CLASS],
                    stereo_continuations=stereo_continuations)
                component_music[formation["nominated_music_seconds"]] = True
                try:
                    segments = memory_support.join_covered(segments, memory_records["records"], window,
                                                           semantics_recipe, additional_music_windows=component_music,
                                                           observations=formation["observations"],
                                                           onsets=usage.get("onsets", []))
                except Exception as exc:  # noqa: BLE001 - optional memory step never costs the reference result
                    formation_memory.update(join_state="failed_reference_segments_kept", join_error=type(exc).__name__)
                segments = gate.relate(segments)
                gate.timing["reference_media"] = {"reacquired": sorted(set(fetched)), "skipped": skipped,
                                                  "unavailable": sorted(library.index.loader.unavailable)}
        except Exception as exc:  # noqa: BLE001 - 照合できないことを、区間を消す根拠にしない（全区間UNKNOWN）
            gate, known = None, []
            relations = reference_relations.not_checked(exc, relation_recipe)
            # 参照の窓・区間を混ぜずに、メモリだけの区間形成（無ければ一回目）へ戻す。
            segments, formation, component_music = memory_formation()
    from lss.bgm import music_corroboration
    musical_windows, musical_receipt = music_corroboration.observe(
        segments, window, np.load(evidence_dir / "probabilities.npy"),
        np.load(evidence_dir / "mixture-probabilities.npy"), semantics_recipe)
    component_music |= musical_windows
    formation["musical_subtype_corroboration"] = musical_receipt
    decided = cue_semantics.decide(segments, window, semantics_recipe, source_seconds=float(plan["source_seconds"]),
                                  additional_music_windows=component_music)
    if gate is not None:
        decided = reference_relations.apply(decided, cue_semantics.KNOWN_RELEASED)
    # 完成番組の編集境界（デジタル無音）で公開の候補を分ける。CM・告知等の挿入の区画は対応領域が保留する。
    # 既発売（EXACT）の片は公開の候補ではないので分けない。分けた片は元の区間の参照との関係を引き継ぐ。
    edit_boundaries = cue_scope.digital_silence_runs(analysis)
    decided = cue_scope.split_at_edit_boundaries(decided, edit_boundaries, float(plan["source_seconds"]))
    from lss.bgm import short_cues
    decided = short_cues.apply(decided, window, np.load(evidence_dir / "mixture-probabilities.npy"),
                              stems["music"], semantics_recipe, float(window.count))
    if gate is not None:
        relations = gate.record(decided)
        try:
            # 照合の結果を参照の知識へ残す（どの番組でどの関係として現れたか）。失敗しても判定は変えない。
            from lss.bgm import reference_store

            seen: dict[str, dict] = {}
            for item in decided:
                relation = item.get("reference_relation") or {}
                sha = (relation.get("reference") or {}).get("reference_sha256")
                if sha:
                    seen.setdefault(sha, {}).setdefault(relation["state"], 0)
                    seen[sha][relation["state"]] += 1
            work = None
            from lss.bgm import reference_acquisition
            hint = reference_acquisition.work_hint(source, source_sha256, Path(fingerprint_cache).parent)
            work = (hint or {}).get("work_key")
            reference_store.note_relations([entry["sha256"] for entry in reference_paths],
                                           source_sha256=source_sha256 or "", work_key=work, summary=seen,
                                           state_root=Path(fingerprint_cache).parent)
        except Exception:  # noqa: BLE001
            pass
    duplicates = _bgm_published_duplicates(recipe, run_dir, analysis, decided, source_sha256, fingerprint_cache)
    # 公開・保留・対象外は対応領域（S2）が決める。安全の観測は原本のタグと編集境界から測る。
    # 書き出す音の完全性は公開の直前に測り直す。
    scope_source = {"path": str(source), "sha256": source_sha256}
    first_pass_probabilities = np.load(evidence_dir / "probabilities.npy")
    mixture_probabilities = np.load(evidence_dir / "mixture-probabilities.npy")
    # 公開の判断の安全の観測は原本の全てのイベント音の窓を数える。鈴・鐘の音だけの音楽の窓の割引は、
    # 描画した音での再確認へ回す指名（rendered_content.nominate）と再確認だけが使う。
    cue_scope.decide(decided, scope_source, mixture=mixture_probabilities, boundaries=edit_boundaries)
    from lss.bgm import source_visual_job
    source_visual = source_visual_job.observe(source, source_sha256 or "", audio_timeline_origin,
                                              decided, run_dir)
    from lss.bgm import rendered_content
    rendered_content.nominate(decided, scope_source, first_pass_probabilities, mixture_probabilities)
    refine: set[str] = set()
    for item in decided:
        if not cue_scope.publishable(item) and not rendered_content.nominated(item):
            item["route"] = {"state": cue_representation.STATE_SKIP, "representation": None,
                             "reasons": ["not_published"]}
            continue
        item["route"] = cue_representation.choose(item["start_seconds"], item["end_seconds"], routing_evidence,
                                                  routing_recipe)
        if item["route"]["representation"] == cue_representation.RAW:
            from lss.separation import selective
            import soundfile as sf
            rate = sf.info(stems["music"]).samplerate
            start_frame,end_frame = round(item["start_seconds"]*rate),round(item["end_seconds"]*rate)
            length = end_frame-start_frame
            original = selective.aligned_read(analysis,rate,start_frame,length,dtype="float64")
            first_music = selective.aligned_read(stems["music"],rate,start_frame,length,dtype="float64")
            step = max(1,round(rate*selective.POLICY.window_seconds))
            mixture_speech = routing_evidence.mix_speech_probability
            allowed = np.asarray([bool(len(mixture_speech[(start_frame+i)//rate:
                (start_frame+min(length,i+step)+rate-1)//rate])) and bool(np.all(
                mixture_speech[(start_frame+i)//rate:(start_frame+min(length,i+step)+rate-1)//rate]
                <= recipe["selective_source_speech_gate"])) for i in range(0,length,step)],dtype=bool)
            native_grid = native_rate == rate
            item["route"] = cue_representation.certify_raw(item["route"],original,first_music,rate,
                                                          allowed=allowed,native_grid=native_grid)
        if gate is not None and item["route"]["representation"] == cue_representation.PROCESSED:
            # 番組の版（EDIT）で、参照で説明し尽くせると確かめた領域は、既発売録音を番組の音量・EQへ合わせた音で
            # 置き換える計画を立てる（`reference_reconstruction`）。描くのは公開の時。説明できない領域は抽出のまま。
            from lss.bgm import reference_reconstruction

            planned = reference_reconstruction.plan(item.get("reference_relation"), item["start_seconds"],
                                                    item["end_seconds"])
            if planned["eligible"]:
                item["route"] = {**item["route"], "reconstruction": planned}
        if item["route"]["representation"] == cue_representation.PROCESSED:
            # 処理済みで公開する区間の保留候補だけを、2回目と歌唱回収へ送る。
            refine.update(cue["cue_id"] for cue in held
                          if cue["start_seconds"] < item["end_seconds"] and cue["end_seconds"] > item["start_seconds"])
    refine_ids, voice_preservation = cue_semantics.voice_preserving_refinement(
        held, refine, window, semantics_recipe)
    return {"decided": decided, "refine_cue_ids": refine_ids, "evidence": evidence, "known": known,
            "voice_preserving_refinement": voice_preservation,
            "cue_formation": {**formation, "memory_support": formation_memory},
            "duplicates": duplicates, "reference_relations": relations,
            "scope_source": scope_source, "edit_boundaries": edit_boundaries,
            "window_count": window.count, "music_tags": music_tags, "mixture_tags": mixture_tags,
            "stereo_scratch_receipts": stereo_receipts, "stereo_tagging_input_seconds": stereo_seconds,
            "tagging_seconds": round(tagging_seconds, 2), "references": (references_digest, len(reference_paths),
                                                                         list(references_missing)),
            "semantics_recipe": semantics_recipe, "routing_recipe": routing_recipe,
            "source_visual": source_visual}


def _bgm_published_duplicates(recipe: dict, run_dir: Path, analysis: Path, decided: list[dict],
                              source_sha256: str | None, fingerprint_cache: Path | None) -> dict:
    """公開しうる区間（WANTEDと、一曲かどうかだけが未確定の区間）のうち、同じ状態領域で先に公開した
    別の原本のcueと同じ組み立て済み音声の区間を重複にする。

    照合できない時は何も変えない（従来どおり公開する）。公開を減らす方向だけに働く。
    """
    from lss.bgm import cue_scope, cue_semantics, identity, published_duplicates

    duplicate_recipe = published_duplicates.default_recipe()
    if duplicate_recipe.recipe_id != recipe["published_duplicates_id"]:
        raise ValueError("BGMの重複判定の宣言がRecipeと一致しません")
    wanted = [item for item in decided if cue_scope.publication_candidate(item)]
    record = {"recipe_id": duplicate_recipe.recipe_id, "prior_runs": [], "identical_spans": [], "state": "not_needed"}
    if not wanted or source_sha256 is None:
        return record
    from lss.bgm import current_best

    parts, runs = published_duplicates.prior_publications(
        run_dir.parent.parent, source_sha256, adopted=current_best.adopted_runs(run_dir.parent.parent.parent))
    record["prior_runs"] = runs
    if not parts:
        record["state"] = "no_prior_publication"
        return record
    try:
        spans = published_duplicates.identical_spans(identity.decode_mono(Path(analysis)), parts, duplicate_recipe,
                                                     cache_dir=fingerprint_cache)
    except (OSError, ValueError, identity.DecodeError) as exc:
        record.update({"state": "not_checked", "error": type(exc).__name__})
        return record
    record.update({"state": "checked", "identical_spans": spans})
    for item in wanted:
        found = published_duplicates.duplicate_of(item["start_seconds"], item["end_seconds"], spans, duplicate_recipe)
        if found:
            item["semantic_class"] = cue_semantics.DUPLICATE_PUBLISHED
            item["reasons"] = [published_duplicates.REASON]
            item["duplicate_of"] = found
    return record


def _published_parts(held: list[dict], representation: str, start_seconds: float, end_seconds: float):
    """表現ごとの材料（保留候補の音声）と、その時間軸での公開区間のframe。"""
    from lss.bgm import cue_representation

    key = "raw" if representation == cue_representation.RAW else "music"
    members = [cue for cue in held if cue["start_seconds"] < end_seconds and cue["end_seconds"] > start_seconds]
    parts = [cue[key] for cue in members]
    rate = int(parts[0]["sample_rate"])
    # 区間は1秒格子。保留候補の外（候補の端が秒の途中の時）へははみ出さない。
    start_frame = max(int(round(start_seconds * rate)), min(part["source_start_frame"] for part in parts))
    end_frame = min(int(round(end_seconds * rate)), max(part["source_end_frame"] for part in parts))
    return members, parts, rate, start_frame, end_frame


def _verified_instrumental_spans(receipt_path: Path, signal_sha256: str, recipe: dict, spans: list) -> dict | None:
    """伴奏区間の受領票が、同じ信号・同じ宣言・同じmodel・同じ区間の完全な結果ならそれを返す。"""
    from lss.bgm import dialogue_residual

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("type") != "bgm_instrumental_spans_receipt"
                or receipt.get("recipe_id") != recipe["dialogue_residual_id"]
                or receipt.get("model_id") != recipe["dialogue_residual_model"]["model_id"]
                or (receipt.get("signal") or {}).get("sha256") != signal_sha256
                or receipt.get("spans") != [list(span) for span in spans]
                or not isinstance(receipt.get("runtime_id"), str)):
            return None
        folder = receipt_path.parent.resolve()
        items = receipt.get("instrumental") or []
        if [item.get("span") for item in items] != receipt["spans"]:
            return None
        for item in items:
            if Path(str(item.get("path", ""))).resolve().parent != folder:
                return None
            dialogue_residual.read_span(item)
        return receipt
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return None


def _bgm_dialogue_residual(recipe: dict, run_dir: Path, analysis: Path | None, analyze,
                           candidates: list[tuple[int, int]]) -> dict:
    """台詞が残る処理済みcueの区間だけを伴奏の既定modelへ通し、cueの部分をrunへ保存する（作り直しに使う）。"""
    if not candidates:
        return {}
    if analysis is None:
        raise ValueError("台詞の残りを取るための解析用の信号がありません")
    spans = sorted([int(a), int(b)] for a, b in candidates)
    signal_sha = file_sha256(analysis)
    folder = (run_dir / "dialogue-residual").resolve()
    receipt = _verified_instrumental_spans(folder / "receipt.json", signal_sha, recipe, spans)
    if receipt is None:
        job = {"cmd": "bgm_instrumental_spans", "signal_path": str(Path(analysis).resolve()),
               "signal_sha256": signal_sha, "model_id": recipe["dialogue_residual_model"]["model_id"],
               "recipe_id": recipe["dialogue_residual_id"], "spans": spans, "input_role": "analysis_copy"}
        if analyze is None:
            from lss.core import worker
            result = worker.handle_bgm_instrumental_spans_job(job, destination_dir=folder)
        else:
            result = analyze(job, folder)
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise _BgmAnalysisFailed(result if isinstance(result, dict) else {
                "status": "error", "error_code": "invalid_worker_response",
                "message": "BGM解析ワーカーから不正な応答が返りました"})
        receipt = _verified_instrumental_spans(folder / "receipt.json", signal_sha, recipe, spans)
        if receipt is None:
            raise ValueError("BGMの伴奏区間の結果が受領票と一致しません")
    spans_by_frames = {}
    for item in receipt["instrumental"]:
        spans_by_frames[tuple(item["span"])] = {
            **{name: item[name] for name in ("path", "sha256", "headroom_shift", "frames", "sample_rate")},
            "runtime_id": receipt["runtime_id"], "model_id": receipt["model_id"],
            "model_sha256": receipt["model_sha256"]}
    return spans_by_frames


def _published_identity_state(item: dict, decision: dict) -> str:
    """公開cueの来歴の同一性。参照が無い・照合していない時は従来の値のまま。"""
    if not decision["references"][1]:
        return "unresolved_no_registered_reference"
    relations = decision.get("reference_relations")
    if relations is None:
        return "unresolved_no_registered_reference_match"
    if relations.get("state") != "checked":
        return "unresolved_reference_not_checked"
    from lss.bgm import reference_relations
    return reference_relations.identity_state(item.get("reference_relation"), references_registered=True)


def _published_reference_relation(item: dict, decision: dict) -> dict:
    """照合した時だけ、公開cueの来歴へ参照との関係の要約を入れる（参照が無い時の来歴は変えない）。"""
    if (decision.get("reference_relations") or {}).get("state") != "checked":
        return {}
    from lss.bgm import reference_relations
    return {"reference_relation": reference_relations.compact(item.get("reference_relation"))}


def _relations_with_source_time(decision: dict, offset: float) -> dict:
    """関係の記録の使用箇所へ、元メディアの時刻を入れる。"""
    record = dict(decision["reference_relations"])
    if record.get("state") == "checked":
        from lss.bgm import reference_relations
        record["occurrences"] = reference_relations.occurrences(decision["decided"], time_offset=offset)
    return record


def _bgm_publish(source: Path, source_sha256: str, recipe: dict, run_dir: Path, held: list[dict],
                 timeline: dict, decision: dict, *, analysis: Path | None = None, analyze=None,
                 overlap_timing: dict | None = None) -> tuple[dict, list]:
    """公開してよい区間を、選んだ表現の保留候補の音声から組み立てて保存する。判定の記録も書く。

    処理済みで公開するcueのうち、SFCの音楽の出力に台詞が残るcueは、伴奏の既定modelとの共通成分にする
    （`lss.bgm.dialogue_residual`）。伴奏はcueの区間だけ推論し、runへ保存する（作り直しに推論は要らない）。

    利用者が処理中に終了したら、cueごとの区切りで止め、作業領域へ書いた音を消してから止まる（公開しない）。
    作業領域へ書き終えて終了の要求も無ければ、完了記録とCurrentBestの切替まで終了処理を待たせる
    （`quitting.hold_publication`。解くのは `_run_bgm_candidate_request`）。
    """
    from lss.bgm import cue_publication

    # 公開名は原本の隣の公開フォルダ。新しいrunの音はまず作業領域へ書いて検証し、CurrentBestの切替で
    # 公開名へ移す（`_bgm_current_best`）。前の版の公開cueはその切替で履歴へ移る。
    destination_dir = cue_publication.folder_for(source)
    staging_dir = cue_publication.staging_folder(source, run_dir.parent.name)
    # 前の試行が公開の前で止まって残した音を片付けてから書く（同じ名前の別の音を残さない）。
    with quitting.writing_user_folder():
        _discard_unpublished_staging(staging_dir, source_sha256, run_dir.parents[2])
    return _bgm_publish_staged(source, source_sha256, recipe, run_dir, held, timeline, decision,
                               analysis=analysis, analyze=analyze, destination_dir=destination_dir,
                               staging_dir=staging_dir, overlap_timing=overlap_timing)


def _bgm_publish_staged(source: Path, source_sha256: str, recipe: dict, run_dir: Path, held: list[dict],
                        timeline: dict, decision: dict, *, analysis: Path | None, analyze,
                        destination_dir: Path, staging_dir: Path,
                        overlap_timing: dict | None = None) -> tuple[dict, list]:
    import numpy as np

    from lss.bgm import (cue_publication, cue_representation, cue_scope, cue_semantics, dialogue_residual, overlap,
                         rendered_content)
    from lss.separation import selective

    if recipe.get("selective_composition") != selective.revision():
        raise ValueError("BGM selective composition differs from its Recipe")

    decided = decision["decided"]
    offset = timeline["offset_numerator"] / timeline["offset_denominator"]
    version = _lss_version()
    published = []
    integrity_failed = []
    # 対応領域が公開してよいとした区間（欲しい劇伴と、一曲かどうかだけが未確定の区間）。
    wanted = [item for item in decided if cue_scope.publishable(item) or rendered_content.nominated(item)]
    positions = {id(item): position for position, item in enumerate(decided)}
    measured_integrity: dict[int, dict] = {}
    residual_recipe = dialogue_residual.default_recipe()
    if residual_recipe.recipe_id != recipe["dialogue_residual_id"]:
        raise ValueError("BGMの台詞の残りの除去の宣言がRecipeと一致しません")
    speech = np.load(run_dir / "semantics" / "probabilities.npy")
    mixture_speech = np.load(run_dir / "semantics" / "mixture-probabilities.npy")[:, 0]
    plans = []
    for item in wanted:
        route = item["route"]
        members, parts, rate, start_frame, end_frame = _published_parts(
            held, route["representation"], item["start_seconds"], item["end_seconds"])
        check = {"state": "not_processed", "eligible": False, "regions": []}
        if route["representation"] == cue_representation.PROCESSED:
            check = {"state": "checked", **dialogue_residual.eligibility(
                speech, start_frame / rate, end_frame / rate, residual_recipe), "regions": []}
            # Cue-average speech remains diagnostic; it never expands treatment
            # into clean windows. Context is available to the spectral renderer.
            check.update(dialogue_residual.processing_plan(speech,start_frame,end_frame,rate,residual_recipe))
        plans.append((item, members, parts, rate, start_frame, end_frame, check))
    required = sorted(set(span for plan in plans for span in dialogue_residual.inference_regions(
        plan[4], plan[5], plan[6]["regions"], residual_recipe)))
    # The composition before the dialogue residual (compose, the selective-original plan and its WAV) needs no
    # inference. With the CPU lane on, it runs for the next few cues while the residual model (and later each cue's
    # rendered-content tagging) holds the GPU; results are consumed strictly in plan order below.
    lane = overlap.open_lane()
    pending: dict[int, object] = {}
    lookahead = 2 * overlap.workers()

    def prepare(index):
        item, members, parts, rate, start_frame, end_frame, check = plans[index]
        return _publication_composition(item, members, parts, rate, start_frame, end_frame, check, run_dir,
                                        analysis, mixture_speech, recipe["selective_source_speech_gate"],
                                        created=lane.created if lane is not None else None)

    def schedule(index):
        if lane is not None and index < len(plans) and index not in pending:
            pending[index] = lane.submit(f"publication-{index}", prepare, index)

    try:
        for index in range(lookahead):
            schedule(index)
        parts_clock = time.perf_counter()
        instrumental = _bgm_dialogue_residual(recipe, run_dir, analysis, analyze, required)
        # 公開の工程の内訳（秒）。重なって動く分があるので、和は公開の秒と一致しない。
        publication_parts = {"residual_model": round(time.perf_counter() - parts_clock, 2)}
        parts_clock = time.perf_counter()
        prepared = _bgm_prepare_published(plans, pending, schedule, lookahead, prepare, instrumental, held, run_dir,
                                          analysis, analyze, recipe, decision, positions, measured_integrity,
                                          integrity_failed, residual_recipe, publication_parts)
        publication_parts["prepare"] = round(time.perf_counter() - parts_clock, 2)
        if lane is not None:
            lane.close(failed=False)
            if overlap_timing is not None:
                overlap_timing["overlap_publication_cpu"] = round(lane.busy_seconds, 2)
    except BaseException:
        if lane is not None:
            lane.close(failed=True)
        raise
    with quitting.writing_user_folder():
        try:
            for item, members, audio, rate, start_frame, end_frame, residual, applied, original_composition, reconstruction, route, scope_row in prepared:
                quitting.checkpoint()
                index = len(published) + 1
                unresolved = cue_scope.unresolved_extent(item)
                key = cue_publication.cue_key(source_sha256, recipe["recipe_id"], start_frame, end_frame, rate)
                provenance = {
                    "schema_version": 1, "type": "bgm_published_cue",
                    "source": {"path": str(source), "sha256": source_sha256},
                    "start_frame": start_frame, "end_frame": end_frame, "sample_rate": rate,
                    "start_seconds": round(start_frame / rate + offset, 6), "end_seconds": round(end_frame / rate + offset, 6),
                    "timebase": "source_media_time_from_decoded_audio_origin",
                    "semantic_class": item["semantic_class"], "reasons": item["reasons"], "features": item["features"],
                    "edges": {"start": item["start_edge"], "end": item["end_edge"]},
                    **({"short_music": item["short_music"]} if item.get("short_music") else {}),
                    "representation": (route["representation"] if reconstruction is None
                                       else reconstruction["representation"]),
                    **({"base_representation": route["representation"], "reconstruction": reconstruction}
                       if reconstruction is not None else {}),
                    "route": {name: route[name] for name in ("state", "fallback", "reasons", "features", "original_check")
                              if name in route},
                    "dialogue_residual": {**{name: value for name, value in residual.items() if name != "instrumental"},
                                          **applied},
                    "selective_composition": original_composition,
                    **({"rendered_content_check": item["rendered_content_check"]}
                       if item.get("rendered_content_check") else {}),
                    "identity_state": _published_identity_state(item, decision),
                    **_published_reference_relation(item, decision),
                    "parts": [{"cue_id": cue["cue_id"],
                               "candidate_asset_id": cue["raw" if route["representation"] == cue_representation.RAW
                                                         else "music"]["candidate_asset_id"],
                               "second_pass_adopted": cue["second_pass"]["adopted"],
                               "vocal_recovery_adopted": cue["vocal_recovery"]["adopted"]} for cue in members],
                    "recipe_id": recipe["recipe_id"], "cue_semantics_id": recipe["cue_semantics_id"],
                    "cue_representation_id": recipe["cue_representation_id"],
                    "event_tagger_id": recipe["event_tagger_id"], "analyzer_id": recipe["analyzer_id"],
                    # 公開を決めた対応領域と、その判断（一曲かどうか未確定なら明示する）。
                    "supported_scope": {name: scope_row[name] for name in
                                        ("scope_id", "disposition", "reason", "cue_extent", "integrity")},
                    "cue_extent": scope_row["cue_extent"], "supported_scope_id": recipe["supported_scope_id"],
                }
                from lss.bgm import collection_metadata
                title = cue_publication.title(source, start_frame / rate + offset, unresolved_extent=unresolved)
                name = cue_publication.file_name(source, start_frame / rate + offset, unresolved_extent=unresolved)
                destination = destination_dir / name
                parts_clock = time.perf_counter()
                record = cue_publication.publish(
                    audio, rate, staging_dir / name, key=key, title=title, album=collection_metadata.album(source),
                    provenance=provenance, input_sha256=source_sha256, recipe_id=recipe["recipe_id"], version=version)
                publication_parts["write"] = round(publication_parts.get("write", 0.0)
                                                   + time.perf_counter() - parts_clock, 2)
                parts_clock = time.perf_counter()
                from lss.bgm import listening
                finished = listening.finish(Path(record["path"]), run_dir.parents[2], expected_pcm=record["audio"]["pcm_sha256"])
                publication_parts["finish"] = round(publication_parts.get("finish", 0.0)
                                                    + time.perf_counter() - parts_clock, 2)
                record.update(sha256=finished["sha256"], audio=finished["audio"], finalization=finished["finalization"])
                checked = item.get("rendered_content_check") or {}
                if checked.get("state") == "passed" and record["audio"]["pcm_sha256"] != checked["checked_audio"]["pcm_sha256"]:
                    raise ValueError("Saved cue PCM differs from the rendered content check")
                published.append({**record, "index": index, "start_frame": start_frame, "end_frame": end_frame,
                                  "start_seconds": provenance["start_seconds"], "end_seconds": provenance["end_seconds"],
                                  "representation": provenance["representation"], "route_state": route["state"],
                                  **({"base_representation": route["representation"], "reconstruction": reconstruction}
                                     if reconstruction is not None else {}),
                                  "candidates": [cue["cue_id"] for cue in members], "title": title,
                                  "dialogue_residual": residual, "cue_extent": scope_row["cue_extent"],
                                  "selective_composition": original_composition,
                                  "album": collection_metadata.album(source), "destination": str(destination),
                                  "provenance": provenance, "version": version})
                item["published"] = {"path": record["path"], "index": index}
            quitting.checkpoint()
        except quitting.JobAbandonedError:
            _discard_unpublished_staging(staging_dir, source_sha256, run_dir.parents[2])
            raise
        quitting.hold_publication()
    if overlap_timing is not None:
        overlap_timing["publication_parts"] = publication_parts
    # 書き出す音を測った結果も入れて、区間ごとの公開・保留・対象外と1本の結論を決め直す（記録用）。
    scope_result = cue_scope.decide(decided, decision["scope_source"],
                                    quality={position: {"integrity_passed": value["integrity_passed"]}
                                             for position, value in measured_integrity.items()})
    if sorted(position for position, item in enumerate(decided) if item.get("published")) != sorted(
            position for position, item in enumerate(decided) if cue_scope.publishable(item)):
        raise ValueError("BGMの公開cueが対応領域の判断と一致しません")
    counts = {name: sum(1 for item in decided if item["semantic_class"] == name)
              for name in cue_semantics.SEMANTIC_CLASSES}
    # 画面の注記「非公開のまま残した区間」：対応領域が保留（HELD）・対象外（UNSUPPORTED）にした区間。
    held_count = sum(1 for item in decided if (item.get("scope") or {}).get("disposition") in ("HELD", "UNSUPPORTED"))
    routes = {}
    for state in (cue_representation.STATE_RAW, cue_representation.STATE_PROCESSED, cue_representation.STATE_UNCERTAIN):
        chosen = [item for item in wanted if item.get("published") and item["route"]["state"] == state]
        routes[state] = {"count": len(chosen),
                         "seconds": round(sum(item["end_seconds"] - item["start_seconds"] for item in chosen), 3)}
    music_tags, mixture_tags = decision["music_tags"], decision["mixture_tags"]
    references_digest, reference_count, references_missing = decision["references"]
    body = {
        "type": "bgm_cue_semantics", "schema_version": 2, "recipe": decision["semantics_recipe"].to_dict(),
        "recipe_id": decision["semantics_recipe"].recipe_id,
        "representation_recipe": decision["routing_recipe"].to_dict(),
        "representation_recipe_id": decision["routing_recipe"].recipe_id,
        "source": {"path": str(source), "sha256": source_sha256},
        "cue_formation": decision.get("cue_formation"),
        "voice_preserving_refinement": decision.get("voice_preserving_refinement"),
        "event_tags": {role: {"analyzer_id": item["analyzer_id"], "runtime_id": item["runtime_id"],
                              "runtime_revision": item["runtime_revision"],
                              "weights_sha256": item["analyzer"]["weights"]["sha256"],
                              "signal_sha256": item["signal"]["sha256"], "input_role": role,
                              "elapsed_seconds": item.get("elapsed_seconds"), "peak_vram_mb": item.get("peak_vram_mb")}
                       for role, item in (("first_pass_music", music_tags), ("mixture_analysis_copy", mixture_tags))},
        "references": {"digest": references_digest, "count": reference_count,
                       "missing": references_missing, "known_spans": decision["known"]},
        **({"reference_relations": _relations_with_source_time(decision, offset)}
           if decision.get("reference_relations") is not None else {}),
        "published_duplicates": decision.get("duplicates"),
        "source_visual": decision.get("source_visual"),
        "window_count": decision["window_count"], "segments": decided, "counts": counts, "routes": routes,
        "held_count": held_count, "integrity_failed": integrity_failed,
        "edit_boundaries": [list(run) for run in decision.get("edit_boundaries") or []],
        "supported_scope": {"scope_id": recipe["supported_scope_id"],
                            **{name: scope_result[name] for name in ("disposition", "reason", "counts",
                                                                     "music_activity_found")},
                            "segments": [{"candidate_id": row["candidate_id"], "disposition": row["disposition"],
                                          "reason": row["reason"]} for row in scope_result["candidates"]]},
        "claim_boundary": [
            "公開は対応領域S2（単話の器楽の劇伴）が決めた。タグ付けが音楽と聞き、歌が無く、10〜150秒で、"
            "両端が音楽の止まり目か強い替わり目の区間だけ。それ以外は理由を付けてHELD・対象外のまま残す",
            "原本の非音楽イベント音が密な区間は原則保留。ほかの保留理由がなく、初回分離音のイベント音が少ない場合だけ、"
            "完成候補のPCM24音声を再検査する。イベント音の密度と音楽の存在を確認できた候補だけ公開する。放送挿入の"
            "役割未確定は引き続き保留。これは効果音の残りや劇伴の削れが無いことの保証ではない",
            "内側に替わり目の疑いがある区間は、一曲かどうか未確定（複数の曲を含みうる）と名前・tag・来歴に"
            "明示して公開した。区切りは推定していない",
            "公開前に書き出す音の完全性（有限・2ch・無音でない・片側が無音でない）を測った",
            "原音で公開したのは、分離器が取り除く成分が区間のどこでも小さく、原本に台詞が聞こえない区間だけ。"
            "言い切れない区間は既知良好の処理済みで公開した",
            "参照録音が登録されていなければ、既発売かどうかは判断していない（未発売の根拠にもしていない）",
            "同じ状態領域で先に公開した別の原本のcueと、台詞・効果音まで同じ組み立て済み音声の区間は、"
            "重複として公開しない（最初に処理した原本の分は単話では判別できず公開される）",
            "区間の端は1秒格子。音楽の止まり目は3秒窓のタグで決めている",
            *(["参照録音で、一つの録音の一つのずれで途切れず説明できる片（連続した抜粋）だけを既発売として"
               "抽出・公開しない（使用箇所として記録し、保留候補の音は残す）。同じ録音の編集・成分の違う版・"
               "別版の候補・未知の区間は公開の判定を変えない。参照のタグ・ファイルの名前は記録へ写すだけで、"
               "判定には使っていない"]
              if (decision.get("reference_relations") or {}).get("state") == "checked" else []),
            *(["既発売録音で組み立て直した結果が、一つの既発売録音の一続きの範囲とほぼ同じ（番組だけの構造・"
               "素材が短い）cueは、手に入る録音の写しになるので新しく公開しない（使用箇所として記録する）"]
              if any(item.get("independently_available") for item in decided) else []),
        ],
    }
    record_path = run_dir / "semantics" / "cue-semantics.json"
    session._write_json_atomic(record_path, body)
    return ({"record": {"name": record_path.name, "sha256": file_sha256(record_path)}, "evidence": decision["evidence"],
             "recipe_id": decision["semantics_recipe"].recipe_id,
             "representation_recipe_id": decision["routing_recipe"].recipe_id, "counts": counts, "routes": routes,
             "held_count": held_count, "integrity_failed": integrity_failed,
             "supported_scope_id": recipe["supported_scope_id"],
             "scope_disposition": scope_result["disposition"], "scope_reason": scope_result["reason"],
             "published_count": len(published), "references_digest": references_digest}, published)


def _publication_composition(item: dict, members: list, parts: list, rate: int, start_frame: int, end_frame: int,
                             check: dict, run_dir: Path, analysis: Path | None, mixture_speech, speech_gate: float,
                             *, created=None):
    """A published cue's audio before the dialogue residual: the composed held parts and, for processed cues, the
    selective-original plan (and its WAV under ``selective-original/``). Needs no inference and reads only decided
    inputs, so it may run on the CPU lane (`lss.bgm.overlap`) ahead of the cue's turn. ``created`` is told about a
    file this call wrote that did not exist before."""
    import numpy as np
    import soundfile as sf

    from lss.bgm import cue_publication, cue_representation
    from lss.separation import selective

    route = item["route"]
    audio = cue_publication.compose(parts, start_frame, end_frame)
    original_composition = None
    original = None
    if route["representation"] == cue_representation.PROCESSED and analysis is not None:
        original = selective.aligned_read(analysis, rate, start_frame, len(audio))
        first_parts = [cue.get("music_first_pass") or cue["music"] for cue in members]
        first_music = cue_publication.compose(first_parts, start_frame, end_frame)
        step = max(1, round(rate * selective.POLICY.window_seconds))
        allowed = []
        for low in range(0, len(audio), step):
            first = (start_frame+low)//rate
            last = (start_frame+min(len(audio), low+step)+rate-1)//rate
            values = mixture_speech[first:last]
            allowed.append(last <= len(mixture_speech) and len(values) > 0
                           and bool(np.all(values <= speech_gate))
                           and not any(a < min(len(audio), low+step) and b > low
                                       for a,b in check["regions"]))
        original_plan = selective.plan(original, first_music, rate,
                                       allowed=np.asarray(allowed, dtype=bool), witness=audio)
        if original_plan["original_regions"] or original_plan["silent_regions"]:
            folder = run_dir / "selective-original"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{start_frame}-{end_frame}.wav"
            existed = path.exists()
            sf.write(path, original, rate, subtype="FLOAT")
            if created is not None and not existed:
                created(path)
            original_composition = {"plan": original_plan, "source_grid": "declared_analysis_copy",
                "native_source_pcm_claim": False,
                "original": {"path": str(path), "sha256": file_sha256(path),
                             "frames": len(audio), "sample_rate": rate, "headroom_shift": 0}}
    return audio, original, original_composition


def _bgm_prepare_published(plans, pending, schedule, lookahead, prepare, instrumental, held, run_dir, analysis,
                           analyze, recipe, decision, positions, measured_integrity, integrity_failed,
                           residual_recipe, publication_parts: dict | None = None) -> list:
    """Render every wanted cue internally, in plan order, and finish optional inference before the user-folder
    write lease. Publication consumes these exact prepared arrays only."""
    import numpy as np

    from lss.bgm import cue_semantics, cue_scope, dialogue_residual, overlap, rendered_content
    from lss.separation import selective

    prepared = []
    for index, (item, members, parts, rate, start_frame, end_frame, check) in enumerate(plans):
        quitting.checkpoint()
        route = item["route"]
        future = pending.pop(index, None)
        schedule(index + lookahead)
        prepared_early = overlap.result_or_none(future)
        audio, original, original_composition = prepared_early if prepared_early is not None else prepare(index)
        residual = {**check, "recipe_id": residual_recipe.recipe_id, "applied": False, "scope": None}
        if check["regions"]:
            regional_spans = dialogue_residual.inference_regions(start_frame, end_frame, check["regions"], residual_recipe)
            witnesses = {span: instrumental[span] for span in regional_spans}
            if any(int(span["sample_rate"]) != rate for span in witnesses.values()):
                raise ValueError("伴奏区間の時間軸が公開cueと一致しません")
            assembled = dialogue_residual.assemble_instrumental(audio, start_frame, witnesses)
            span = dialogue_residual.write_span(
                assembled, rate, run_dir / "dialogue-residual" / f"composed-{start_frame}-{end_frame}.flac")
            prototype = next(iter(witnesses.values()))
            span.update({key: prototype[key] for key in ("runtime_id", "model_id", "model_sha256")})
            span["regional_witnesses"] = [{"span": list(k), **v} for k, v in witnesses.items()]
            # Fresh rendering and replay consume the same serialized
            # witness; PCM24 rounding must not change CurrentBest.
            audio = dialogue_residual.region_consensus(audio, dialogue_residual.read_span(span),
                                                       check["regions"], rate, residual_recipe)
            residual.update({"applied": True, "scope": "local", "instrumental": span,
                             "transition": "inside_cosine_shared_v1"})
        if original_composition is not None:
            audio = selective.render(original, audio, original_composition["plan"])
        applied = ({"model_id": residual["instrumental"]["model_id"],
                    "model_sha256": residual["instrumental"]["model_sha256"],
                    "runtime_id": residual["instrumental"]["runtime_id"]} if residual["applied"] else {})
        reconstruction = None
        if route.get("reconstruction"):
            rebuilt, reconstruction = _reference_reconstruction(audio, rate, start_frame, route["reconstruction"],
                                                               state_root=run_dir.parents[2])
            if reconstruction["replaced_seconds"] > 0:
                from lss.bgm import reference_reconstruction

                from lss.bgm import release_availability
                distribution = {d["reference_sha256"]:release_availability.lookup(d["reference_sha256"])
                                for d in reconstruction["donor_regions"] if d.get("accepted")}
                available = reference_reconstruction.availability(reconstruction,distribution=distribution)
                reconstruction["availability"] = available
                if available["independently_available"]:
                    # 描いた結果は、手に入る既発売の録音の一続きの範囲と実質同じ：新しい資産を作らない。
                    # 既発売（EXACT）と同じく公開せず、使用箇所として記録する（保留候補の音は残る）。
                    item.update(acoustic_class=item["semantic_class"], acoustic_reasons=item["reasons"],
                                semantic_class=cue_semantics.KNOWN_RELEASED,
                                reasons=["reconstruction_equals_released_recording"],
                                independently_available=available)
                    item["route"] = route = {**route, "not_published": "independently_available_release",
                                             "reconstruction_check": available}
                    position = positions[id(item)]
                    measured_integrity[position] = cue_scope.integrity(audio, rate)
                    cue_scope.final(item, position, decision["scope_source"], measured_integrity[position])
                    continue
                audio = rebuilt
            else:
                # 描いた後の確認で置き換える領域が残らなかった：再構成と名乗らず、元の表現のまま公開する。
                item["route"] = route = {**route, "reconstruction_not_applied": [
                    {k: row[k] for k in ("start_seconds", "end_seconds", "fit", "reason")}
                    for row in reconstruction["donor_regions"]]}
                reconstruction = None
        if rendered_content.nominated(item):
            folder = run_dir / "rendered-content" / f"{start_frame}-{end_frame}"
            check_clock = time.perf_counter()
            try:
                preparation = rendered_content.prepare(audio, rate, start_frame, folder)
                tagged = _tag_job(Path(preparation["signal"]), folder / "tags", recipe, analyze,
                                  "tentative_serialized_source_heavy_cue")
                from lss.bgm import event_tagging
                tagged_p, tagged_e = event_tagging.load_outputs(tagged["outputs"])
                tagging_evidence = {
                    "signal_sha256": preparation["signal_sha256"], "analyzer_id": tagged["analyzer_id"],
                    "receipt_path": str(folder / "tags" / "receipt.json"),
                    "receipt_sha256": file_sha256(folder / "tags" / "receipt.json")}
                rendered_content.evaluate(item, preparation, tagged_p, tagged_e,
                    np.load(run_dir / "semantics" / "mixture-probabilities.npy"))
                item["rendered_content_check"]["tagging"] = tagging_evidence
            except quitting.JobAbandonedError:
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                item["safety"] = dict(item["rendered_content_check"]["source_safety"])
                item["rendered_content_check"] = {**item["rendered_content_check"], "state": "held",
                    "reason": "rendered_observations_unavailable", "error_type": type(exc).__name__}
            if publication_parts is not None:
                publication_parts["rendered_check"] = round(publication_parts.get("rendered_check", 0.0) + time.perf_counter() - check_clock, 2)
        checked = item.get("rendered_content_check") or {}
        if checked.get("method") == rendered_content.METHOD and checked.get("state") != "passed":
            item["safety"] = dict(checked["source_safety"])
        # 書き出す音そのものの完全性（L0）を測り、対応領域で最終判断する。欠けていれば公開しない。
        position = positions[id(item)]
        measured_integrity[position] = cue_scope.integrity(audio, rate)
        scope_row = cue_scope.final(item, position, decision["scope_source"], measured_integrity[position])
        if scope_row["disposition"] != "PUBLISHED":
            item["route"] = {**route, "not_published": scope_row["reason"]}
            if scope_row["reason"] == "below_quality_gate":
                integrity_failed.append({"start_seconds": item["start_seconds"], "end_seconds": item["end_seconds"],
                                         "failed_gates": scope_row.get("failed_gates")})
            continue
        prepared.append((item, members, audio, rate, start_frame, end_frame, residual, applied,
                         original_composition, reconstruction, route, scope_row))
    return prepared


def reconcile_bgm_current_best(state_root: Path | None = None) -> int:
    """起動時に、中断したBGM公開結果の切替の続きを行う（新版を置けなければ旧版へ戻す）。"""
    from lss.bgm import current_best

    root = Path(state_root).resolve() if state_root is not None else session.state_dir().resolve()
    return current_best.reconcile_all(root, lambda sha: _run_regenerator(root, sha))


def restore_previous_bgm_result(source: Path, *, state_root: Path | None = None) -> ProcessingOutcome:
    """原本のBGM公開結果を、直前に置き換えた結果へ戻す（rollback）。今の結果は履歴へ移る。

    戻した結果は固定され、同じ原本を処理し直しても勝手に新しい版へ戻さない。もう一度呼ぶと、
    直前の結果（戻す前の版）へ戻る。推論はしない（履歴のファイル、無ければ保留候補から作り直す）。
    """
    from lss.bgm import cue_publication, current_best, current_best_lineage

    source = Path(source).resolve()
    try:
        source_sha256 = file_sha256(source)
        root = Path(state_root).resolve() if state_root is not None else session.state_dir().resolve()
        with _BgmCandidateRunLock(root / current_best.DIRECTORY / source_sha256):
            regenerate = _run_regenerator(root, source_sha256)
            current_best.reconcile(root, source_sha256, regenerate)
            _relocated_to_source(root, source, source_sha256, current_best.read(root, source_sha256), regenerate)
            record = current_best_lineage.restore_previous(root, source_sha256,
                                                           cue_publication.history_root(source), regenerate,
                                                           public_folder=cue_publication.folder_for(source))
            files = record["current"]["files"]
            if not files:
                raise current_best.CurrentBestError("戻した結果に公開cueがありません")
            restored_path = Path(files[0]["run_path"])
            run = json.loads(restored_path.read_text(encoding="utf-8"))
            return _bgm_outcome(run, restored_path, reused=True,
                                outputs=tuple(Path(item["public_path"]) for item in files), source=source,
                                operation="restore_previous")
    except Exception as exc:
        return ProcessingOutcome(disposition=ResultDisposition.FAILED, error_code="bgm_restore_failed",
                                 message=str(exc) or type(exc).__name__)


def bgm_candidate_step(
    source: Path, dispatch: Callable[..., dict], on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """画面の順次処理から呼ぶ、1音源のBGM候補工程。worker形式の結果を返す。

    解析器の推論は伴奏と同じ常駐workerへ送り（GPUを握るprocessを増やさない）、保存先は
    BGM候補の内部作業領域に固定する。意味を未確定の候補はHELDのまま公開しない。
    成功時の ``bgm`` は結果ごとの到達で、``done``（検証済みcueを公開）、
    ``held``（候補n区間を保留）、``none``（検証済みの対象なし）を区別する。
    """
    def analyze(job: dict, destination_dir: Path) -> dict:
        # 利用者が処理中に終了したら、次の推論を頼まず、結果待ちの後にも先へ進まない。
        quitting.checkpoint()
        payload = {**job, "destination_dir": str(Path(destination_dir).resolve())}
        try:
            result = dispatch(payload, on_progress=on_progress)
        except (EOFError, OSError, TimeoutError) as exc:
            quitting.checkpoint()
            # 伴奏の順次処理と同じく、workerへ届かないのはエンジンの停止（HALT）。
            # 候補処理の一般的な失敗（行の失敗）へ倒さない。
            return {"status": "error", "error_code": "worker_unavailable", "message": str(exc)}
        quitting.checkpoint()
        return result

    outcome = run_bgm_candidate_request(
        ProcessingRequest.from_paths([Path(source)], UserIntent.EXTRACT_BGM_CUES),
        analyze=analyze,
    )
    if outcome.disposition is ResultDisposition.ZERO:
        return {"status": "ok", "bgm": {"k": "none", "n": 0, "product_result": outcome.product_result}, "reused": outcome.reused}
    if outcome.disposition is ResultDisposition.HELD:
        return {"status": "ok", "bgm": {"k": "held", "n": len(outcome.held_artifacts), "product_result": outcome.product_result},
                "evidence_path": str(outcome.evidence_path or ""), "reused": outcome.reused}
    if outcome.disposition is ResultDisposition.PUBLISHED:
        # 公開したcueの数と、判定を保留して非公開のまま残した区間の数（画面の注記用）。
        held = 0
        try:
            run = json.loads(Path(outcome.record_path).read_text(encoding="utf-8"))
            held = _bgm_held_count(run)
        except (OSError, ValueError, TypeError):
            pass
        return {"status": "ok", "bgm": {"k": "done", "n": len(outcome.outputs), "held": held,
                                        "product_result": outcome.product_result,
                                        "files": [Path(path).name for path in outcome.outputs]},
                "outputs": [str(path) for path in outcome.outputs], "reused": outcome.reused}
    return {"status": "error",
            "error_code": outcome.error_code or "bgm_candidate_processing_failed",
            "message": outcome.message or ""}


def existing_bgm_candidates(source: Path, source_sha256: str, *,
                            state_root: Path | None = None) -> dict | None:
    """同じ原本・同じRecipeの完了済みBGM候補runがあれば、その到達を返す（処理はしない）。

    受理済みとして再利用できるのは、検証済みの対象なし（``none``）と、公開したcueが全て記録どおり
    残っているrun（``done``）だけ。保留候補しか無いrunは公開の基準に届いていないので、既存として扱わず
    次の実行で同じrunを読み戻す。公開cueが消えていれば、次の実行で保留候補から作り直す。
    """
    recipe = _bgm_candidate_recipe()
    root = Path(state_root).resolve() if state_root is not None else session.state_dir().resolve()
    # A saved foreground source can still service an interrupted older-source revisit after restart.
    # Do not let the UI's reuse shortcut hide pending work from the ordinary Product entry.
    try:
        from lss.bgm import memory_refresh
        if any(plan["source_sha256"] != source_sha256
               for plan in memory_refresh.pending(root, recipe_id=recipe["recipe_id"])):
            return None
    except Exception:  # noqa: BLE001 - optional memory cannot disable normal reuse
        pass
    _, references_digest, _, _ = _registered_references(root, source=source, source_sha256=source_sha256,
                                                         ingest=False)
    key = _bgm_candidate_run_key(source_sha256, recipe["recipe_id"], references_digest)
    run_path = root / "bgm-candidates" / key / "run" / "run.json"
    from lss.bgm import current_best

    if current_best.journal_path(root, source_sha256).is_file():
        return None   # 切替の途中。次の処理で続きを行う
    try:
        record = current_best.read(root, source_sha256)
    except current_best.CurrentBestError:
        return None
    if record is not None:
        current = record["current"]
        if not current.get("pinned") and (key not in (current.get("evaluated_runs") or [])
                                          or current.get("adoption_policy") != _BGM_ADOPTION_POLICY):
            # 今のCurrentBestにはこのRecipeの判断が（今の採用の方針で）まだ反映されていない。処理すると反映する。
            return None
        files = [Path(item["public_path"]) for item in current["files"]]
        if not files or not all(current_best._matches(path, item.get("pcm_sha256"))
                                for path, item in zip(files, current["files"])):
            return None
        from lss.bgm import listening
        if not listening.current(files, root, collection=True, library=_bgm_library(source, root)):
            return None
        try:
            adopted_path = run_path if run_path.is_file() else Path(current["files"][0]["run_path"])
            adopted = json.loads(adopted_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        projection = _existing_bgm_projection(source, adopted, adopted_path, files, root, record)
        return {"k": "done", "n": len(files), "outputs": [str(path) for path in files],
                "files": [path.name for path in files], "held": projection["held_count"],
                "product_result": projection}
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (run.get("status") == _BGM_STATUS_ZERO and run.get("type") == _BGM_RUN_TYPE
            and run.get("recipe_id") == recipe["recipe_id"]
            and (run.get("source") or {}).get("sha256") == source_sha256):
        try:
            session.validate_zero_result(run.get("zero_result"))
        except (TypeError, ValueError):
            return None
        return {"k": "none", "n": 0}
    if run.get("status") == _BGM_STATUS_PUBLISHED:
        try:
            verified = _verified_bgm_run(run_path, source_sha256, recipe["recipe_id"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
        from lss.bgm import cue_publication

        outputs = [Path(item["path"]) for item in verified["published"]]
        folder = cue_publication.folder_for(Path(source)).resolve()
        if all(path.is_file() and path.parent.resolve() == folder and audio_identity.matches_record(path, item)
               for path, item in zip(outputs, verified["published"])):
            return {"k": "done", "n": len(outputs), "outputs": [str(path) for path in outputs],
                    "files": [path.name for path in outputs],
                    "held": _bgm_held_count(verified)}
    return None


def existing_accompaniment(source: Path, source_sha256: str, *,
                           state_root: Path | None = None) -> dict | None:
    """同じ原本を既定の伴奏modelで分離した、検証できる保存済み結果があればその到達を返す。

    完了台帳の同じ原本・同じ所在の記録のうち、Recipe本文のmodelが現在の既定と一致し、
    記録した出力が今も同じ音として残っているものだけを使う。処理はしない。
    """
    from lss.core import recipe_registry
    from lss.harness.tsv_io import load_tsv

    model_id = catalog.default_model()["model_id"]
    ledger = session.completion_ledger_path(state_root)
    if not ledger.exists():
        return None
    _, rows = load_tsv(ledger)
    source_key = str(Path(source).resolve())
    for row in reversed(rows):
        if row.get("input_sha256") != source_sha256 or row.get("source_path") != source_key:
            continue
        recipe_id = row.get("recipe_id") or ""
        identity = recipe_registry.processing_identity(recipe_id, state_root=state_root)
        if identity is None or identity.get("model_id") != model_id:
            continue
        if session.completed_zero_result(
            source_sha256, recipe_id, Path(source), state_root=state_root,
        ) is not None:
            return {"k": "none", "n": 0}
        checked = session.check_completed_outputs(
            source_sha256, recipe_id, Path(source), state_root=state_root)
        if checked and all(checked.values()):
            from lss.bgm.listening import describe
            from lss.output import results
            try:
                listening = describe(checked)
            except ValueError:
                return None
            projection = results.worker_result({"status": "skipped", "outputs": [str(p) for p in checked],
                                                "input_sha256": source_sha256, "listening": listening})
            return {"k": "reused", "n": len(checked), "outputs": [str(p) for p in checked],
                    "product_result": projection}
    return None


def outcome_from_worker_result(result: dict) -> ProcessingOutcome:
    """永続化されたworker結果を、backend詳細を漏らさない製品結果へ変換する。"""
    checked = jobs.validated_worker_result(result)
    status = checked["status"]
    if status in {"ok", "skipped"}:
        outputs = tuple(Path(path) for path in checked["outputs"])
        zero_result = checked.get("zero_result")
        if zero_result is not None:
            # validated_worker_resultも検査するが、製品境界の不変条件として明示する。
            session.validate_zero_result(zero_result)
            return ProcessingOutcome(
                disposition=ResultDisposition.ZERO,
                recipe_id=checked.get("recipe_id"),
                zero_result=zero_result,
                reused=status == "skipped",
            )
        return ProcessingOutcome(
            disposition=ResultDisposition.PUBLISHED,
            outputs=outputs,
            recipe_id=checked.get("recipe_id"),
            reused=status == "skipped",
            scene_balance=_scene_balance_observation(checked.get("scene_balance")),
            product_result=checked.get("product_result"),
        )
    error_code = checked["error_code"]
    disposition = (
        ResultDisposition.UNSUPPORTED
        if error_code in {"unsupported_channels", "stem_combination_unsupported"}
        # 部品が欠けた組み合わせは、作れるかどうかが未確定（UNKNOWN）のまま保留する。部分的な音は保存していない。
        else ResultDisposition.HELD if error_code == "stem_combination_incomplete"
        else ResultDisposition.FAILED
    )
    return ProcessingOutcome(
        disposition=disposition,
        error_code=error_code,
        message=checked["message"],
    )


def _scene_balance_observation(value) -> dict | None:
    """worker結果の場面の観測を、記録どおりの形だけ通す。足りない値は作らない。"""
    from lss.separation.scene_balance import STATES

    if not isinstance(value, dict) or value.get("state") not in STATES:
        return None
    if not isinstance(value.get("recipe_id"), str):
        return None
    balance = value.get("balance_db")
    if balance is not None and (isinstance(balance, bool) or not isinstance(balance, (int, float))):
        return None
    observation = {"state": value["state"], "reason": value.get("reason"),
                   "recipe_id": value["recipe_id"], "balance_db": balance}
    timed = value.get("over_time")
    if (isinstance(timed, dict) and isinstance(timed.get("recipe_id"), str)
            and isinstance(timed.get("state_shares"), dict)
            and all(name in STATES and isinstance(share, (int, float))
                    and not isinstance(share, bool)
                    for name, share in timed["state_shares"].items())):
        observation["over_time"] = {
            "recipe_id": timed["recipe_id"],
            "window_count": timed.get("window_count"),
            "state_shares": dict(timed["state_shares"]),
            "median_balance_db": timed.get("median_balance_db")}
    return observation


def outcome_from_item(item: jobs.QueueItem) -> ProcessingOutcome | None:
    """再起動後のqueue項目を含め、終端結果だけを製品結果へ変換する。"""
    if item.status in {"queued", "running"}:
        return None
    if item.result is None:
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED,
            error_code="missing_worker_result",
            message="完了した処理項目にworker結果がありません",
        )
    return outcome_from_worker_result(item.result)


def _read_bgm_plan(plan_path: Path, source_path: Path, *, locator=None) -> tuple[dict, str]:
    content = plan_path.read_bytes()
    plan = json.loads(content)
    if not isinstance(plan, dict):
        raise ValueError("固定BGM planの記録が不正です")
    source = plan.get("source")
    if (plan.get("status") != "FROZEN_BEFORE_ROUTED_CANDIDATES"
            or plan.get("promotion_allowed") is not False
            or plan.get("holdout_consumed") is not False or not isinstance(source, dict)
            or not isinstance(plan.get("renderer_id"), str) or not plan["renderer_id"]):
        raise ValueError("未公開Development候補として固定したplanが必要です")
    if Path(source.get("path", "")).resolve() != source_path:
        raise ValueError("製品要求の入力と固定BGM planの原本が一致しません")
    current_source = locator(source)[0] if locator is not None else source_path
    if file_sha256(current_source) != source.get("sha256"):
        raise ValueError("固定BGM planの原本内容が一致しません")
    return plan, hashlib.sha256(content).hexdigest()


def _read_bgm_result(plan_path: Path, plan: dict, plan_sha256: str,
                     report_path: Path, *, locator=None) -> tuple[dict, str, list[dict]]:
    content = report_path.read_bytes()
    report = json.loads(content)
    if (not isinstance(report, dict) or report_path != plan_path.parent / "run.json"
            or report.get("status") != "completed_development_routed_episode_candidates"
            or report.get("promotion_allowed") is not False or report.get("holdout_consumed") is not False
            or report.get("publication_state") != "not_published"
            or report.get("source_hashes_after_verified") is not True
            or report.get("plan_sha256") != plan_sha256 or report.get("renderer_id") != plan.get("renderer_id")
            or report.get("outcome") not in {"unresolved_candidates", "unresolved_no_activity_candidates"}):
        raise ValueError("BGM Development結果の非公開・同一性契約が不正です")
    planned_source, source = plan["source"], report.get("source")
    if (not isinstance(source, dict)
            or Path(source.get("path", "")).resolve() != Path(planned_source["path"]).resolve()
            or source.get("sha256") != planned_source["sha256"]):
        raise ValueError("BGM Development実行後の原本同一性が一致しません")
    if not isinstance(report.get("candidates"), list):
        raise ValueError("BGM Development候補の記録が不正です")
    held_records = []
    paths = set()
    for candidate in report["candidates"]:
        asset = candidate.get("asset") if isinstance(candidate, dict) else None
        if (not isinstance(asset, dict) or asset.get("representation") != "noncanonical_staged_candidate"
                or asset.get("publication_state") != "not_published"):
            raise ValueError("BGM候補が非公開の保留成果として記録されていません")
        path = Path(asset.get("path", "")).resolve()
        if not path.is_relative_to(plan_path.parent / "candidates") or path in paths:
            raise ValueError("BGM候補の保存先が一致しません")
        item = {"path": str(path), "sha256": asset.get("sha256")}
        if isinstance(asset.get("audio"), dict):
            item["audio"] = asset["audio"]
        current_path = locator(item)[0] if locator is not None else path
        if not asset.get("sha256") or not audio_identity.matches_record(current_path, item):
            raise ValueError("BGM候補の内容SHA-256または保存先が一致しません")
        paths.add(path)
        held_records.append(item)
    if bool(held_records) != (report["outcome"] == "unresolved_candidates"):
        raise ValueError("BGM Development outcomeと候補件数が一致しません")
    return report, hashlib.sha256(content).hexdigest(), held_records


def _verify_bgm_inputs(plan_path: Path, plan_sha256: str, source: dict,
                       report_path: Path, report_sha256: str, held_records: list[dict], *, locations=None) -> None:
    def intact(item):
        paths = locations[item["path"]] if locations is not None else (Path(item["path"]),)
        return bool(paths) and all(audio_identity.matches_record(path, item) for path in paths)

    if (not intact(source) or file_sha256(plan_path) != plan_sha256 or file_sha256(report_path) != report_sha256
            or any(not intact(item) for item in held_records)):
        raise ValueError("BGM製品結果の確認中に原本・plan・根拠・候補が変更されています")


def _bgm_outcome_payload(source: dict, report_path: Path, report: dict,
                         report_sha256: str, held_records: list[dict], *, plan_record: dict | None = None) -> dict:
    payload = {
        "schema_version": 1, "type": "processing_outcome", "intent": UserIntent.EXTRACT_BGM_CUES.value,
        "source": {"path": str(Path(source["path"]).resolve()), "sha256": source["sha256"]},
        "disposition": ResultDisposition.HELD.value, "outputs": [], "held_artifacts": held_records,
        "evidence": {"path": str(report_path), "sha256": report_sha256},
        "recipe_id": report.get("renderer_id"), "publication_allowed": False, "reason": report["outcome"],
    }
    if plan_record is not None:
        payload["plan"] = plan_record
    return payload


def _unavailable_bgm_evidence(record, identity_view, music_view, reference_view) -> tuple[str, ...]:
    return tuple(role for key, role, view in (
        ("recording_identity", "recording_identity", identity_view),
        ("held_alternatives", "music_alternative", music_view),
        ("held_reference_residuals", "reference_residual", reference_view),
    ) if record.get(key) is not None and view is None)


def read_bgm_development_outcome(record_path: Path, *, state_root: Path | None = None) -> ProcessingOutcome:
    """保存済みの保留結果を読み戻す。候補生成・推論・記録更新を行わない。

    原本・固定plan・run・全候補を確認する。現在使えない任意の録音照合は引き継がず、
    未解決の原候補は保つ。移動記録に残る同内容の素材も使い、記録された全所在を返す。
    held_artifactsは候補ごとの先頭の所在、held_artifact_locationsは確認できた全所在を持つ。
    Developmentの候補なしを対象不在のZEROへ変換しない。
    """
    try:
        record_path = Path(record_path).resolve()
        if record_path.name != "application-outcome.json":
            raise ValueError("保存済みのBGM製品結果記録が必要です")
        content = record_path.read_bytes()
        record = json.loads(content)
        if not isinstance(record, dict) or not isinstance(record.get("source"), dict):
            raise ValueError("保存済みのBGM製品結果が不正です")
        plan_record = record.get("plan")
        if plan_record is not None and not isinstance(plan_record, dict):
            raise ValueError("保存済みBGM planの来歴が不正です")
        history_folder = record_path.parent.parent
        legacy_root = record_path.parent
        if (plan_record is None and history_folder.name == "outcome-history"
                and record_path.parent.name == hashlib.sha256(content).hexdigest()):
            # 初期形式の記録はplanを持たない。内容hashを確認した履歴の元runへ戻る。
            legacy_root = history_folder.parent
        plan_path = (Path(plan_record["path"]).resolve() if plan_record is not None
                     else legacy_root / "plan.json")
        is_history = (history_folder == plan_path.parent / "outcome-history"
                      and record_path.parent.name == hashlib.sha256(content).hexdigest())
        if plan_path.parent != record_path.parent and not is_history:
            raise ValueError("保存済みBGM planが処理結果の範囲外です")
        report_path = plan_path.parent / "run.json"
        locations = {}

        def find_locations(item):
            from lss.assets.relocate import current_locations
            path = Path(item["path"])
            found = []
            try:
                if path.is_file() and audio_identity.matches_record(path, item):
                    found = [path.resolve()]
            except OSError:
                pass
            if not found:
                found = sorted({path.resolve() for path in current_locations(item["sha256"], state_root=state_root)})
            if not found:
                raise ValueError(f"保存済みBGM素材が同じ内容で見つかりません: {item['path']}")
            return tuple(found)

        def locate(item):
            found = find_locations(item)
            locations[item["path"]] = found
            return found

        plan, plan_sha256 = _read_bgm_plan(plan_path, Path(record["source"]["path"]).resolve(), locator=locate)
        if plan_record is not None and plan_record != {"path": str(plan_path), "sha256": plan_sha256}:
            raise ValueError("保存済みBGM planの同一性が一致しません")
        report, report_sha256, held_records = _read_bgm_result(plan_path, plan, plan_sha256, report_path, locator=locate)
        expected = _bgm_outcome_payload(plan["source"], report_path, report, report_sha256, held_records,
                                        plan_record=plan_record)
        base = {key: value for key, value in record.items()
                if key not in {"recording_identity", "held_alternatives", "held_reference_residuals"}}
        if (_with_resolved_source(base) != _with_resolved_source(expected)
                or type(record.get("schema_version")) is not int
                or record.get("publication_allowed") is not False):
            raise ValueError("保存済みのBGM製品結果と原本・根拠・保留候補が対応しません")
        identity_evidence_path = None
        saved_identity = record.get("recording_identity")
        if saved_identity is not None:
            from lss.bgm import identity_candidates
            try:
                current_identity = identity_candidates.annotation_view(report_path)
                if current_identity == saved_identity:
                    identity_evidence_path = report_path.parent / "identity.json"
            except identity_candidates.IdentityCandidatesError:
                pass
        alternative_view = None
        alternative_locations = {}

        def locate_alternative(item):
            found = find_locations(item)
            alternative_locations[item["path"]] = found
            return found

        saved_alternatives = record.get("held_alternatives")
        if isinstance(saved_alternatives, dict):
            from lss.bgm import held_results
            try:
                checked = held_results.alternative_view(report_path, Path(saved_alternatives["path"]), locator=locate_alternative)
                if checked == saved_alternatives:
                    alternative_view = checked
            except (held_results.HeldResultsError, KeyError, TypeError, ValueError):
                pass
        reference_view = None
        reference_locations = {}

        def locate_reference(item):
            found = find_locations(item)
            reference_locations[item["path"]] = found
            return found

        saved_reference = record.get("held_reference_residuals")
        if isinstance(saved_reference, dict):
            from lss.validation import reference_residual_episode
            try:
                checked = reference_residual_episode.read_result(
                    report_path, Path(saved_reference["path"]), locator=locate_reference)
                if checked == saved_reference:
                    reference_view = checked
            except (reference_residual_episode.ReferenceResidualResultError, KeyError, TypeError, ValueError):
                pass
        _verify_bgm_inputs(plan_path, plan_sha256, plan["source"], report_path, report_sha256, held_records,
                           locations=locations)
        if identity_evidence_path is not None:
            try:
                if identity_candidates.annotation_view(report_path) != saved_identity:
                    identity_evidence_path = None
            except identity_candidates.IdentityCandidatesError:
                identity_evidence_path = None
        if alternative_view is not None:
            try:
                if held_results.alternative_view(report_path, Path(alternative_view["path"]), locator=locate_alternative) != alternative_view:
                    alternative_view = None
            except held_results.HeldResultsError:
                alternative_view = None
        if reference_view is not None:
            try:
                if reference_residual_episode.read_result(
                        report_path, Path(reference_view["path"]), locator=locate_reference) != reference_view:
                    reference_view = None
            except reference_residual_episode.ReferenceResidualResultError:
                reference_view = None
        if record_path.read_bytes() != content:
            raise ValueError("読戻し中にBGM製品結果記録が変更されています")
        unavailable = _unavailable_bgm_evidence(record, identity_evidence_path, alternative_view, reference_view)
        return ProcessingOutcome(
            disposition=ResultDisposition.HELD, held_artifacts=tuple(locations[item["path"]][0] for item in held_records),
            evidence_path=report_path, record_path=record_path, recipe_id=report.get("renderer_id"),
            identity_evidence_path=identity_evidence_path, reused=True,
            source_locations=locations[plan["source"]["path"]],
            held_artifact_locations={
                **{Path(item["path"]): locations[item["path"]] for item in held_records},
                **({Path(item["path"]): alternative_locations[item["path"]] for item in alternative_view["artifacts"]}
                   if alternative_view is not None else {}),
                **({Path(item["path"]): reference_locations[item["path"]] for item in reference_view["artifacts"]}
                   if reference_view is not None else {}),
            },
            held_alternative_artifacts=tuple(alternative_locations[item["path"]][0] for item in alternative_view["artifacts"])
            if alternative_view is not None else (),
            alternative_evidence_path=Path(alternative_view["path"]) if alternative_view is not None else None,
            held_reference_residual_artifacts=tuple(reference_locations[item["path"]][0]
                                                    for item in reference_view["artifacts"])
            if reference_view is not None else (),
            reference_residual_evidence_path=Path(reference_view["path"]) if reference_view is not None else None,
            unavailable_evidence=unavailable,
            message="一部の保留根拠を確認できません。元候補とその来歴は保持しています"
            if unavailable else "保存済みのDevelopment候補を保留結果として読み戻しました",
        )
    except (ImportError, OSError, TypeError, ValueError, KeyError) as exc:
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED, error_code="invalid_bgm_development_result", message=str(exc),
        )


def _with_resolved_source(payload: dict) -> dict:
    """原本の所在は綴りではなく実体で比べる。素材を移して別名で辿れても同じ結果とする。"""
    source = payload.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        return payload
    try:
        resolved = str(Path(source["path"]).resolve())
    except (OSError, ValueError):
        return payload
    return {**payload, "source": {**source, "path": resolved}}


def _cue_readiness(outcome, report: dict, recipe) -> dict:
    """候補ごとの参照なし検査。Recipeを宣言しない限り、品質は測ったことにしない。"""
    if recipe is None:
        return {}
    import soundfile as sf

    from lss.bgm import cue_readiness

    results = {}
    for candidate, place in zip(report["candidates"], outcome.held_artifacts):
        values, rate = sf.read(str(place), dtype="float64", always_2d=True)
        measured = cue_readiness.measure(values.T, int(rate), method=recipe.method)
        checked = cue_readiness.evaluate(measured, recipe)
        results[candidate["asset"]["candidate_asset_id"]] = {
            "readiness": checked,
            # 対応領域の品質gateは下限だけを見る。合否も測定値として渡す。
            "quality": {**{key: value for key, value in measured.items()
                           if isinstance(value, (int, float)) and not isinstance(value, bool)},
                        "readiness_passed": 1.0 if checked["passed"] else 0.0}}
    return results


def _cue_extents_from_annotation(report: dict, annotation_path: Path) -> dict | None:
    """runへ付いた一致・再演の記録から、候補ごとの「一曲かどうか」を引く。

    子候補は親の候補を切り分けたものなので、親の区間を覆う読み取りだけを引き上げる。
    記録が読めない、または根拠が無い候補には何も付けない。
    """
    from lss.bgm import identity_run
    from lss.bgm.cue_extent import CueExtentError, extent_for_span

    try:
        annotation = json.loads(Path(annotation_path).read_bytes())
        offset = identity_run.timeline_offset(report)
        children = []
        for activity in annotation.get("activity_runs") or []:
            children.extend(activity.get("candidates") or [])
        if not children:
            return None
        extents = {}
        for candidate in report["candidates"]:
            asset, origin = candidate["asset"], candidate["origin"]
            rate = float(origin["sample_rate"])
            reading = extent_for_span(origin["start_frame"] / rate + offset,
                                      origin["end_frame"] / rate + offset, children)
            if reading is not None:
                extents[asset["candidate_asset_id"]] = reading
        return extents or None
    except (OSError, ValueError, KeyError, TypeError, CueExtentError):
        # 付加記録が読めないことを、範囲が決まったことにも決まらないことにもしない。
        return None


def _occurrence_groups_from_annotation(report: dict, annotation_path: Path) -> dict | None:
    """runへ付いた使用箇所のまとまりを、候補IDごとの重複の材料へ直す。

    まとまりは子候補で作られているので、子を含む親の候補へ持ち上げる。一つの親の中で
    繰り返しているだけのものは、別の使用箇所ではないので重複にしない。
    """
    from lss.bgm import identity_run

    try:
        annotation = json.loads(Path(annotation_path).read_bytes())
        offset = identity_run.timeline_offset(report)
        spans = []
        for candidate in report["candidates"]:
            origin = candidate["origin"]
            rate = float(origin["sample_rate"])
            spans.append((candidate["asset"]["candidate_asset_id"],
                          origin["start_frame"] / rate + offset,
                          origin["end_frame"] / rate + offset))
        children = {}
        for activity in annotation.get("activity_runs") or []:
            for child in activity.get("candidates") or []:
                start, end = float(child["start_seconds"]), float(child["end_seconds"])
                for name, low, high in spans:
                    if start >= low - 0.05 and end <= high + 0.05:
                        children[str(child["candidate_id"])] = name
                        break
        groups = {}
        for activity in annotation.get("activity_runs") or []:
            for group in activity.get("occurrence_groups") or []:
                parents = sorted({children[str(member[1])] for member in group["members"]
                                  if str(member[1]) in children})
                if len(parents) < 2:
                    # 同じ候補の中の繰り返しは、別の使用箇所ではない。
                    continue
                for name in parents:
                    groups[name] = {"group_id": group["occurrence_group_id"],
                                    "members": parents, "chosen_candidate_id": None}
        return groups or None
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        # 付加記録が読めないことを、重複が無いことにしない。
        return None


def review_bgm_scope_coverage(record_path: Path, scope, *, source_domain: str,
                              readiness=None, observations: dict | None = None,
                              cue_extents: dict | None = None, role_policy=None,
                              state_root: Path | None = None) -> dict:
    """保存済みの保留結果を、先に固定した対応領域で読み直す。

    候補ごとに公開・保留・未対応を決め、公開しない理由を名前で返す。音声・記録・
    公開状態は一切変更しない。ここで`PUBLISHED`と出ても、実際の公開は行わない。

    `observations` は候補IDごとの観測（歌唱の有無など）。測っていない観測は渡さない。
    `cue_extents` は候補IDごとの「一曲かどうか」の読み取り。根拠のない候補には渡さない。
    `role_policy` を渡すと、記録に役割が無い候補の役割を観測から導く。導いた役割は
    採用policyの結論であり、annotationの事実ではない。観測が決めないものは未解決のまま残る。
    """
    from lss.bgm import identity_candidates
    from lss.bgm.supported_scope import ScopeError, classify_result

    if not isinstance(source_domain, str) or not source_domain.strip():
        raise ScopeError("対象の素材領域を宣言する必要があります")
    outcome = read_bgm_development_outcome(record_path, state_root=state_root)
    if outcome.disposition is not ResultDisposition.HELD:
        return {"status": "NO_READABLE_HELD_RESULT", "publication_performed": False,
                "record_path": str(Path(record_path).resolve()),
                "outcome_disposition": outcome.disposition.value, "message": outcome.message}
    report_path = Path(outcome.evidence_path)
    report = json.loads(report_path.read_bytes())
    identity_states, identity_available = {}, False
    if outcome.identity_evidence_path is not None:
        try:
            identity_states = identity_candidates.candidate_identity_states(
                identity_candidates.annotation_view(report_path))
            identity_available = True
        except identity_candidates.IdentityCandidatesError:
            # 照合根拠を今読めないことを、同一性が無いことにしない。
            identity_states, identity_available = {}, False
    if cue_extents is None and outcome.identity_evidence_path is not None:
        # 指定が無ければ、同じrunへ付いた一致・再演の記録から読み取る。
        # 読み取れない記録は無視する。根拠が無いことを「一曲である」にしない。
        cue_extents = _cue_extents_from_annotation(
            report, Path(outcome.identity_evidence_path))
    occurrence_groups = (
        {} if outcome.identity_evidence_path is None
        else (_occurrence_groups_from_annotation(
            report, Path(outcome.identity_evidence_path)) or {}))
    checked = _cue_readiness(outcome, report, readiness)
    derived_roles = {}
    if role_policy is not None:
        from lss.bgm.role_policy import decide_roles

        derived_roles = decide_roles(
            {candidate["asset"]["candidate_asset_id"]:
                 (observations or {}).get(candidate["asset"]["candidate_asset_id"]) or {}
             for candidate in report["candidates"]}, role_policy)
    candidates = []
    for candidate in report["candidates"]:
        asset, origin = candidate["asset"], candidate["origin"]
        recording_identity = identity_states.get(asset["candidate_asset_id"])
        measured = checked.get(asset["candidate_asset_id"])
        derived = derived_roles.get(asset["candidate_asset_id"])
        candidates.append({
            "candidate_id": asset["candidate_asset_id"],
            "sample_rate": origin["sample_rate"], "start_frame": origin["start_frame"],
            "end_frame": origin["end_frame"], "source_domain": source_domain,
            # 記録の役割を優先し、無ければ宣言したpolicyで観測から導く。推測はしない。
            "semantic_role": (candidate.get("semantic_role")
                              or (derived or {}).get("semantic_role")
                              or "unresolved_music_like"),
            # 記録の役割を使ったときは、policyの導出を判断材料にしない。
            "semantic_role_derivation": None if candidate.get("semantic_role") else derived,
            "identity_state": (recording_identity or {}).get("identity_state", "unresolved"),
            "observations": (observations or {}).get(asset["candidate_asset_id"]) or {},
            "cue_extent": (cue_extents or {}).get(asset["candidate_asset_id"]),
            "occurrence_group": occurrence_groups.get(asset["candidate_asset_id"]),
            "evidence": {"source_provenance": report.get("source"),
                         "occurrence_span": [origin["start_frame"], origin["end_frame"]],
                         "recording_identity": recording_identity},
            "quality": (measured or {}).get("quality", candidate.get("quality"))})
    decided = classify_result(candidates, scope, music_activity_found=bool(candidates))
    for row in decided["candidates"]:
        result = (checked.get(row["candidate_id"]) or {}).get("readiness")
        if result is not None:
            row["readiness"] = result
    return {"status": "REVIEWED_WITHOUT_PUBLISHING", "publication_performed": False,
            "record_path": str(Path(record_path).resolve()),
            "evidence_path": str(report_path), "recording_identity_available": identity_available,
            "cue_readiness_recipe_id": readiness.recipe_id if readiness is not None else None,
            "observed_candidate_count": sum(1 for row in candidates if row["observations"]),
            "cue_extent_candidate_count": sum(1 for row in candidates
                                              if row["cue_extent"] is not None),
            "role_policy_id": None if role_policy is None else role_policy.policy_id,
            "occurrence_group_candidate_count": len(occurrence_groups),
            "derived_role_candidate_count": sum(
                1 for row in derived_roles.values()
                if row["semantic_role"] not in ("unresolved", "unresolved_music_like")),
            "scope": scope.to_dict(), **decided}


def run_bgm_development(
    request: ProcessingRequest,
    frozen_plan_path: Path,
    *,
    resume: bool = False,
    runner=None,
    reference_paths: Iterable[Path] | None = None,
    identity_cache_dir: Path | None = None,
    reference_origins: dict[str, dict] | None = None,
    alternative_run_path: Path | None = None,
    reference_residual_run_path: Path | None = None,
) -> ProcessingOutcome:
    """固定済みのBGM候補Stageを実行し、非公開のHELD結果として返す。

    これは凍結planを使うDevelopment専用入口。通常の ``submit`` は単話だけから得る
    製品候補経路を使い、この実験用plan経路には接続しない。
    """
    if request.intent != UserIntent.EXTRACT_BGM_CUES:
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED,
            error_code="intent_mismatch",
            message="BGM Development入口にはBGM cue抽出要求が必要です",
        )
    if len(request.sources) != 1 or request.urls:
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED,
            error_code="development_bgm_requires_single_source",
            message="BGM Development実行は1入力ずつ固定する必要があります",
        )
    plan_path = Path(frozen_plan_path).resolve()
    source_path = request.sources[0].resolve()
    try:
        # 通常の伴奏入口でBGM分析用numpy/soundfile等を読み込まない。
        from lss.validation import routed_episode_discovery
        if runner is None:
            runner = routed_episode_discovery.run
        plan, plan_sha256 = _read_bgm_plan(plan_path, source_path)
        planned_source = plan["source"]
        report_path = Path(runner(plan_path, resume=resume)).resolve()
        report, report_sha256, held_records = _read_bgm_result(plan_path, plan, plan_sha256, report_path)
        held = [Path(item["path"]) for item in held_records]
        outcome_path = plan_path.parent / "application-outcome.json"
        try:
            previous_bytes = outcome_path.read_bytes()
        except FileNotFoundError:
            previous_bytes = None
        try:
            previous = json.loads(previous_bytes) if previous_bytes is not None else {}
        except (TypeError, ValueError):
            previous = {}
        current_evidence = {"path": str(report_path), "sha256": report_sha256}
        current_source = {"path": str(source_path), "sha256": planned_source["sha256"]}
        prior_views = previous if (isinstance(previous, dict) and previous.get("evidence") == current_evidence
                                  and previous.get("source") == current_source
                                  and previous.get("recipe_id") == report["renderer_id"]) else {}
        recording_identity = None
        identity_evidence_path = None
        if reference_paths is not None or (report_path.parent / "identity.json").is_file():
            from lss.bgm import identity, identity_candidates, identity_run
            if reference_paths is not None:
                try:
                    identity_evidence_path = identity_run.annotate_run(
                        report_path, list(reference_paths), cache_dir=identity_cache_dir,
                        reference_origins=reference_origins,
                    )
                    recording_identity = identity_candidates.annotation_view(report_path)
                except (identity.DecodeError, identity_run.IdentityRunError, identity_candidates.IdentityCandidatesError) as exc:
                    raise ValueError(str(exc)) from exc
            else:
                # 再開時に参照の指定を省略しても、現在の入力に対応する既存根拠を失わない。
                # 古い根拠は現在の判断に使わず、未解決の原候補と復元用の履歴を保つ。
                try:
                    recording_identity = identity_candidates.annotation_view(report_path)
                    identity_evidence_path = report_path.parent / identity_run.RECORD_NAME
                except (identity_run.IdentityRunError, identity_candidates.IdentityCandidatesError):
                    recording_identity = None
        payload = _bgm_outcome_payload(
            planned_source, report_path, report, report_sha256, held_records,
            plan_record={"path": str(plan_path), "sha256": plan_sha256},
        )
        if recording_identity is not None:
            payload["recording_identity"] = recording_identity
        elif isinstance(prior_views.get("recording_identity"), dict):
            payload["recording_identity"] = prior_views["recording_identity"]
        alternatives = None
        inherited_alternatives = False
        saved_alternatives = prior_views.get("held_alternatives")
        if alternative_run_path is None and isinstance(saved_alternatives, dict) and isinstance(saved_alternatives.get("path"), str):
            alternative_run_path = Path(saved_alternatives["path"])
            inherited_alternatives = True
        if alternative_run_path is not None:
            from lss.bgm import held_results
            try:
                alternatives = held_results.alternative_view(report_path, alternative_run_path)
            except held_results.HeldResultsError as exc:
                if not inherited_alternatives:
                    raise ValueError(str(exc)) from exc
            if alternatives is not None:
                payload["held_alternatives"] = alternatives
            elif inherited_alternatives:
                # 所在や任意の根拠の一時的な不利用で、再検討に必要な出典を消さない。
                payload["held_alternatives"] = saved_alternatives
        reference_view = None
        inherited_reference = False
        saved_reference = prior_views.get("held_reference_residuals")
        if (reference_residual_run_path is None and isinstance(saved_reference, dict)
                and isinstance(saved_reference.get("path"), str)):
            reference_residual_run_path = Path(saved_reference["path"])
            inherited_reference = True
        if reference_residual_run_path is not None:
            from lss.validation import reference_residual_episode
            try:
                reference_view = reference_residual_episode.read_result(report_path, reference_residual_run_path)
            except reference_residual_episode.ReferenceResidualResultError:
                if not inherited_reference:
                    raise
            if reference_view is not None:
                payload["held_reference_residuals"] = reference_view
            elif inherited_reference:
                payload["held_reference_residuals"] = saved_reference

        def verify_outcome_inputs() -> None:
            _verify_bgm_inputs(plan_path, plan_sha256, planned_source, report_path, report_sha256, held_records)
            if recording_identity is not None:
                try:
                    current_identity = identity_candidates.annotation_view(report_path)
                except (identity_run.IdentityRunError, identity_candidates.IdentityCandidatesError) as exc:
                    raise ValueError(str(exc)) from exc
                if current_identity != recording_identity:
                    raise ValueError("BGM製品結果の保存中に録音照合の根拠が変更されています")
            if alternatives is not None:
                try:
                    current_alternatives = held_results.alternative_view(report_path, alternative_run_path)
                except held_results.HeldResultsError as exc:
                    raise ValueError(str(exc)) from exc
                if current_alternatives != alternatives:
                    raise ValueError("BGM製品結果の保存中に代替候補の根拠が変更されています")
            if reference_view is not None:
                current_reference = reference_residual_episode.read_result(report_path, reference_residual_run_path)
                if current_reference != reference_view:
                    raise ValueError("BGM製品結果の保存中に参照残差分岐の根拠が変更されています")
            try:
                current_bytes = outcome_path.read_bytes()
            except FileNotFoundError:
                current_bytes = None
            if current_bytes != previous_bytes:
                raise ValueError("BGM保留結果が別の処理で更新されました。現在の結果を保持します")

        _write_outcome_atomic(outcome_path, payload, verify_outcome_inputs)
        unavailable = _unavailable_bgm_evidence(payload, recording_identity, alternatives, reference_view)
        return ProcessingOutcome(
            disposition=ResultDisposition.HELD,
            held_artifacts=tuple(held),
            evidence_path=report_path,
            record_path=outcome_path,
            recipe_id=report.get("renderer_id"),
            identity_evidence_path=identity_evidence_path,
            held_alternative_artifacts=tuple(Path(item["path"]) for item in alternatives["artifacts"])
            if alternatives is not None else (),
            alternative_evidence_path=Path(alternatives["path"]) if alternatives is not None else None,
            held_reference_residual_artifacts=tuple(Path(item["path"]) for item in reference_view["artifacts"])
            if reference_view is not None else (),
            reference_residual_evidence_path=Path(reference_view["path"]) if reference_view is not None else None,
            unavailable_evidence=unavailable,
            message="一部の保留根拠を確認できません。元候補と再検討に必要な来歴を保持しました"
            if unavailable else "Development候補を保留しました。正式なBGM成果物としては公開していません",
        )
    except (ImportError, OSError, TypeError, ValueError, KeyError) as exc:
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED,
            error_code="invalid_bgm_development_result",
            message=str(exc),
        )


def run_bgm_series_development(
    request: ProcessingRequest,
    out_dir: Path,
    *,
    episode_labels: Iterable[str] | None = None,
    recipe=None,
    reference_paths: Iterable[Path] | None = None,
    identity_cache_dir: Path | None = None,
    measurements: Iterable[dict] | None = None,
    rank_by_mutual_explained: bool = True,
    stack_recipe=None,
    runner=None,
) -> ProcessingOutcome:
    """同一シリーズの複数話から、未発売cueの候補を保留結果として取り出す。

    ``run_bgm_development`` が1話ぶんの固定planを実行するのに対し、こちらは**話どうしの
    再演**を材料にする。分離器を使わず、原本mixと参照録音の照合だけで候補を作る。

    ``stack_recipe`` を渡すと、同じ録音の他の回を使って候補ごとの音楽を推定し、解析用の
    派生物として保存する（`lss/bgm/occurrence_stack.py`）。分離modelは使わない。

    これもDevelopment専用入口。通常の ``submit`` は単話だけから得る製品候補経路を使い、
    シリーズ再演を前提とするこの補助経路には接続しない。
    """
    if request.intent != UserIntent.EXTRACT_BGM_CUES:
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED,
            error_code="intent_mismatch",
            message="BGM Development入口にはBGM cue抽出要求が必要です",
        )
    declared = recipe
    if declared is None:
        from lss.bgm import series_cues as _series_cues
        declared = _series_cues.DEFAULT_RECIPE
    minimum = 1 if getattr(declared, "include_within_episode", False) else 2
    if len(request.sources) < minimum or request.urls:
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED,
            error_code="series_bgm_requires_multiple_sources",
            message="話をまたぐ再演を見るには同一シリーズの2話以上をローカル入力で渡す必要があります。"
                    "1話で行うなら `include_within_episode` を宣言したRecipeを渡してください",
        )
    try:
        # 通常の伴奏入口でBGM照合用numpy/soundfile等を読み込まない。
        from lss.bgm import series_cue_run

        sources = [Path(path).resolve() for path in request.sources]
        labels = [str(label).strip() for label in episode_labels] if episode_labels is not None \
            else [path.stem for path in sources]
        if len(labels) != len(sources):
            raise ValueError("話のラベルの数が入力と一致しません")
        episodes = dict(zip(labels, sources))
        if len(episodes) != len(sources):
            raise ValueError("話のラベルが重複しています")

        run = series_cue_run.run_series_cues if runner is None else runner
        record_path = Path(run(
            episodes, declared, Path(out_dir),
            reference_paths=list(reference_paths) if reference_paths is not None else None,
            cache_dir=identity_cache_dir,
            measurements=list(measurements) if measurements is not None else None,
            rank_by_mutual_explained=rank_by_mutual_explained,
            stack_recipe=stack_recipe,
        )).resolve()
        record = series_cue_run.read_result(record_path)
    except (ImportError, OSError, TypeError, ValueError, KeyError,
            RuntimeError) as exc:
        return ProcessingOutcome(
            disposition=ResultDisposition.FAILED,
            error_code="invalid_bgm_series_development_result",
            message=str(exc),
        )

    held = [candidate for candidate in record["candidates"] if not candidate.get("guard_reason")]
    if not held:
        # 対象不在・ゼロ件は正当な結果。候補の音声と来歴は残す。
        return ProcessingOutcome(
            disposition=ResultDisposition.ZERO,
            evidence_path=record_path,
            record_path=record_path,
            recipe_id=record["recipe_id"],
            source_locations=tuple(sources),
            zero_result={
                "reason": "no_unreleased_recurrence_held",
                "span_count": record["span_count"],
                "guard_state": record["guard_state"],
                "guard_rejected_count": record["guard_rejected_count"],
                "released_group_count": record["released_group_count"],
                "identical_reuse_count": record["identical_reuse_count"],
                "claim_boundary": record["claim_boundary"],
            },
            message="同じ録音の再演から取り出せる未発売候補はありませんでした",
        )
    return ProcessingOutcome(
        disposition=ResultDisposition.HELD,
        held_artifacts=tuple(Path(item["artifact"]["path"]) for item in held),
        evidence_path=record_path,
        record_path=record_path,
        recipe_id=record["recipe_id"],
        source_locations=tuple(sources),
        message="Development候補を保留しました。正式なBGM成果物としては公開していません",
    )
