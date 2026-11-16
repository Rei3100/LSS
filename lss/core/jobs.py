"""UIに依存しない順次処理。run()は呼び出し側のバックグラウンドスレッドで実行する。"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event, Lock
from collections.abc import Callable
from uuid import uuid4

from lss.core import catalog, session
from lss.core.client import dispatch_job
from lss.core.session import iter_input_files
from lss.harness import fetch


_DEFAULT_STATE_PATH = object()
_QUEUE_STATE_VERSION = 1
_STATUSES = frozenset({"queued", "running", "completed", "skipped", "failed"})
_ACTIVE_STATUSES = frozenset({"queued", "running"})

# 欲しい結果（意味的target）。表示順でもある。伴奏は既存の分離経路、BGMは単話候補経路、
# 欲しい音は利用者が言葉で指定した音（`lss.sound`）。
INSTRUMENTAL = "inst"
BGM = "bgm"
SOUND = "snd"
TARGETS = (INSTRUMENTAL, BGM, SOUND)
# 結果ごとの到達。done/reused/none は受理済み（再試行で繰り返さない）。held は公開できる
# 基準に届いていない非公開候補で、failed/unsupported と同じく受理済みではない。
ACCEPTED_OUTCOMES = frozenset({"done", "reused", "none"})
_OUTCOME_KINDS = ACCEPTED_OUTCOMES | {"held", "failed", "unsupported"}
# その音源では成立しない失敗。同じ音源で再試行しても変わらない（lss.ui.contractと同じ区分）。
_UNSUPPORTED_CODES = frozenset({
    "unsupported_channels", "unsafe_output_serialization", "invalid_input_signal",
    "capability_unsupported", "stem_combination_unsupported",
})


def normalized_sounds(values) -> list[str] | None:
    """欲しい音（言葉）の一覧。空はNone。12個・各120文字まで、重複は除く。"""
    from lss.sound import intent

    if values is None:
        return None
    if isinstance(values, str):
        entries = intent.split_entries(values)
    elif isinstance(values, (list, tuple)) and all(isinstance(value, str) for value in values):
        entries = intent.split_entries("\n".join(values))
    else:
        raise ValueError(f"欲しい音が不正です: {values!r}")
    return entries or None


def normalized_targets(targets) -> list[str]:
    """欲しい結果の集合を表示順へ正規化する。空・未知を含む集合は受け付けない。"""
    if targets is None:
        return [INSTRUMENTAL]
    if isinstance(targets, str) or not all(target in TARGETS for target in targets):
        raise ValueError(f"欲しい結果が不正です: {targets!r}")
    values = [target for target in TARGETS if target in set(targets)]
    if not values:
        raise ValueError("欲しい結果が1つもありません")
    return values


def outcome_of_worker_result(result: dict) -> dict:
    """伴奏の分離結果を、結果ごとの到達へ翻訳する。"""
    status = result.get("status")
    product = {"product_result": result["product_result"]} if isinstance(result.get("product_result"), dict) else {}
    if status in {"ok", "skipped"}:
        if result.get("zero_result") is not None:
            return {"k": "none", "n": 0, **product}
        return {"k": "done" if status == "ok" else "reused",
                "n": len(result.get("outputs") or []), **product}
    code = result.get("error_code") or "processing_failed"
    return {"k": "unsupported" if code in _UNSUPPORTED_CODES else "failed", "error_code": code, **product}


@dataclass
class QueueItem:
    id: str
    path: Path
    model_id: str
    stem_target: str
    status: str = "queued"
    result: dict | None = None
    progress: dict | None = None
    # STARTで凍結した欲しい結果。Noneは未凍結（旧形式と伴奏だけの登録は伴奏のみ）。
    targets: list[str] | None = None
    # 結果ごとの到達 {target: {"k": ..., "n": ...}}。受理済みは再試行で繰り返さない。
    outcomes: dict | None = None
    # STARTで凍結した欲しい音（言葉）。欲しい結果に SOUND がある時だけ使う。
    sounds: list[str] | None = None

    def wanted(self) -> list[str]:
        return normalized_targets(self.targets)

    def unresolved(self) -> list[str]:
        """まだ受理済みの到達が無い欲しい結果。"""
        outcomes = self.outcomes or {}
        return [target for target in self.wanted()
                if (outcomes.get(target) or {}).get("k") not in ACCEPTED_OUTCOMES]


def validated_worker_result(result) -> dict:
    """worker応答を、保存・製品結果への変換に使える形へ正規化する。"""
    invalid = {
        "status": "error",
        "error_code": "invalid_worker_response",
        "message": "分離ワーカーから不正な応答が返りました",
    }
    if not isinstance(result, dict) or result.get("status") not in {"ok", "skipped", "error"}:
        return invalid
    if result["status"] in {"ok", "skipped"}:
        outputs = result.get("outputs")
        if not isinstance(outputs, list) or not all(
            isinstance(path, str) and path.strip() for path in outputs
        ):
            return invalid
        zero_result = result.get("zero_result")
        if bool(outputs) == (zero_result is not None):
            return invalid
        if zero_result is not None:
            try:
                session.validate_zero_result(zero_result)
            except ValueError:
                return invalid
    elif not (
        isinstance(result.get("error_code"), str) and result["error_code"].strip()
        and isinstance(result.get("message"), str)
    ):
        return invalid
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return invalid
    return deepcopy(result)


def _persisted_result(result: dict | None) -> dict | None:
    """保存するworker結果。実行環境の全記述は畳み、runtime_idだけを残す。

    実行環境の全記述（依存・adapterのhash一覧）は1件で20KBを超え、どの項目でも同じ内容に
    なる。成果物のタグにも残るので、キューの保存では複製しない。項目が増えるたびに
    保存全体が膨らみ、状態が変わるたびの書き直しが重くなるのを避ける。
    """
    if not isinstance(result, dict) or "runtime_revision" not in result:
        return result
    compact = {key: value for key, value in result.items() if key != "runtime_revision"}
    revision = result.get("runtime_revision")
    if "runtime_id" not in compact and isinstance(revision, dict) and revision.get("runtime_id"):
        compact["runtime_id"] = revision["runtime_id"]
    return compact


class SeparationQueue:
    """処理中の追加、失敗後の継続、失敗項目の再試行、ファイル間での一時停止を扱う。

    キューが空になるかpause()されるまでrun()が処理する。再開は再度run()を呼ぶ。
    状態は既定でstate_dir()/queue.jsonへ保存し、再起動時に未完了項目を復旧する。
    """

    def __init__(
        self, dispatch=None, *, state_path=_DEFAULT_STATE_PATH,
        on_progress: Callable[[str, dict], None] | None = None,
        on_result: Callable[[str, dict], None] | None = None,
        existing: Callable[[QueueItem, str], dict | None] | None = None,
        bgm: Callable[..., dict] | None = None,
    ):
        """`existing(item, target)` は既存の結果を照会し、使えるなら到達（reused/none）を返す。
        `bgm(path, dispatch, on_progress)` は単話BGM候補の1工程を実行し、worker形式の
        結果（成功時は ``{"status": "ok", "bgm": 到達}``）を返す。

        `on_result(item_id, result)` は結果ごとの工程が終わるたびに同期で呼ばれる。
        """
        self._dispatch = dispatch or dispatch_job
        self._existing = existing
        self._bgm = bgm
        self._halted = Event()
        self._abandoned = Event()
        self._streams_progress = dispatch is None
        self._on_progress = on_progress
        self._on_result = on_result
        self._items: list[QueueItem] = []
        self._revision = 0
        self._lock = Lock()
        self._runner = Lock()
        self._pause = Event()
        self._state_path = (
            session.state_dir() / "queue.json" if state_path is _DEFAULT_STATE_PATH
            else Path(state_path) if state_path is not None else None
        )
        self._load()

    def _load(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != _QUEUE_STATE_VERSION:
                raise ValueError("対応していないキュー状態です")
            raw_items = payload.get("items")
            if not isinstance(raw_items, list):
                raise ValueError("itemsが配列ではありません")
            restored = []
            seen_ids: set[str] = set()
            recovered_running = False
            for raw in raw_items:
                if not isinstance(raw, dict):
                    raise ValueError("項目がオブジェクトではありません")
                status = raw.get("status")
                if status not in _STATUSES:
                    raise ValueError(f"不明な状態です: {status}")
                item_id = raw.get("id")
                if not isinstance(item_id, str) or not item_id or item_id in seen_ids:
                    raise ValueError(f"項目IDが不正または重複しています: {item_id}")
                seen_ids.add(item_id)
                for key in ("path", "model_id", "stem_target"):
                    if not isinstance(raw.get(key), str) or not raw[key]:
                        raise ValueError(f"項目の{key}が不正です")
                result = deepcopy(raw.get("result"))
                progress = deepcopy(raw.get("progress"))
                targets = raw.get("targets")
                if targets is not None:
                    targets = normalized_targets(targets)
                sounds = normalized_sounds(raw.get("sounds"))
                outcomes = deepcopy(raw.get("outcomes"))
                if outcomes is not None and (
                        not isinstance(outcomes, dict)
                        or any(key not in TARGETS or not isinstance(value, dict)
                               or value.get("k") not in _OUTCOME_KINDS
                               for key, value in outcomes.items())):
                    raise ValueError("項目のoutcomesが不正です")
                if result is not None and not isinstance(result, dict):
                    raise ValueError("項目のresultがオブジェクトではありません")
                if progress is not None and not isinstance(progress, dict):
                    raise ValueError("項目のprogressがオブジェクトではありません")
                if status == "running":
                    status = "queued"
                    recovered_running = True
                    if outcomes is not None:
                        # 中断した工程の到達は確定していない。受理済みだけを持ち越す。
                        outcomes = {key: value for key, value in outcomes.items()
                                    if value.get("k") in ACCEPTED_OUTCOMES} or None
                    if (outcomes or {}).get(INSTRUMENTAL) is None:
                        result = None
                restored.append(QueueItem(
                    id=item_id,
                    path=Path(raw["path"]),
                    model_id=raw["model_id"],
                    stem_target=raw["stem_target"],
                    status=status,
                    result=result,
                    progress=None if recovered_running and raw.get("status") == "running"
                    else progress,
                    targets=targets,
                    outcomes=outcomes,
                    sounds=sounds,
                ))
            self._items = restored
            if recovered_running:
                self._save_locked()
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"キュー状態を読み込めません: {self._state_path}: {exc}") from exc

    def _save_locked(self) -> None:
        # 変更のたびに版を進める。表示側が同じ内容を何度も複製し直さずに済む。
        self._revision += 1
        if self._state_path is None or self._abandoned.is_set():
            # 終了後は書かない。処理中だった1件は保存済みの「処理中」のまま残り、次の起動で処理待ちへ
            # 戻る（受理済みの結果は持ち越す）。すぐ起動し直した次のLSSの記録も上書きしない。
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _QUEUE_STATE_VERSION,
            "items": [
                {**asdict(item), "path": str(item.path), "result": _persisted_result(item.result)}
                for item in self._items
            ],
        }
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w", dir=self._state_path.parent, prefix=f".{self._state_path.name}-",
                suffix=".tmp", encoding="utf-8", newline="\n", delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary_path.replace(self._state_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _persist_locked(self, rollback: Callable[[], None]) -> None:
        """保存できなかった未確定の操作を戻し、同じキューで再試行できるようにする。"""
        try:
            self._save_locked()
        except BaseException:
            rollback()
            raise

    def add(
        self, paths: list[Path], model_id: str | None = None, stem_target: str | None = None,
        *, include_existing: bool = False, targets: list[str] | None = None,
        sounds: list[str] | None = None,
    ) -> list[str]:
        """入力を登録する。include_existing時は同じ処理待ち・処理中項目のIDも返す。

        `targets` を渡すと、その欲しい結果で凍結して登録する（処理中の追加）。`sounds` は
        その時の欲しい音（言葉）。
        """
        if targets is not None:
            targets = normalized_targets(targets)
        sounds = normalized_sounds(sounds)
        files = iter_input_files(paths)
        if not files:
            return []
        if model_id is None or stem_target is None:
            row = catalog.default_model() if model_id is None else catalog.get_model(model_id)
            model_id = row["model_id"]
            # 要求した意味的な結果（伴奏）。モデルが相補stemも作るかどうかとは無関係で、
            # 公開される成果物はここで指定したパートだけになる。
            if stem_target is None:
                try:
                    stem_target = catalog.requested_stem(row)
                except (fetch.ModelDownloadError, fetch.DownloadSizeError):
                    # 初回で設定を取得できなくても登録は妨げない。モデルの出力パートとの照合と
                    # 取得の失敗は、処理時にworkerが返す（画面ではFETCH FAILEDで止まり再試行できる）。
                    stem_target = catalog.INSTRUMENTAL
        added = []
        accepted = []
        with self._lock:
            previous_items = list(self._items)
            # 終了済みの履歴は再投入を妨げない。同じパスの音声が後から編集・差し替え
            # られた場合も、worker側の入力ハッシュで新しい処理として判定できるようにする。
            known = {
                (item.path, item.model_id, item.stem_target): item.id
                for item in self._items if item.status in _ACTIVE_STATUSES
            }
            for path in files:
                key = (path, model_id, stem_target)
                if key in known:
                    if include_existing and known[key] not in accepted:
                        accepted.append(known[key])
                    continue
                item = QueueItem(uuid4().hex, path, model_id, stem_target,
                                 targets=list(targets) if targets is not None else None,
                                 sounds=list(sounds) if sounds is not None and targets is not None
                                 and SOUND in targets else None)
                self._items.append(item)
                known[key] = item.id
                added.append(item.id)
                accepted.append(item.id)
            if added:
                self._persist_locked(lambda: self._items.__setitem__(slice(None), previous_items))
        return accepted

    def snapshot(self) -> list[QueueItem]:
        with self._lock:
            return deepcopy(self._items)

    @property
    def revision(self) -> int:
        """変更のたびに進む版。内容を複製せずに「変わったか」だけを問える。"""
        with self._lock:
            return self._revision

    def remove(self, item_id: str) -> bool:
        """一覧から1件外す。処理中の項目は外さない（保存済みの結果は消さない）。

        キューから消えるのは登録だけで、既に書き出した成果物には触れない。
        """
        with self._lock:
            item = next((item for item in self._items if item.id == item_id), None)
            if item is None or item.status == "running":
                return False
            previous_items = list(self._items)
            self._items.remove(item)
            self._persist_locked(lambda: self._items.__setitem__(slice(None), previous_items))
            return True

    def pause(self) -> None:
        """現在のファイルの保存が終わった後、次のファイルへ進まず停止する。"""
        self._pause.set()

    def halt(self) -> None:
        """セッションを続けられない。現在の音源の残りの結果にも進まず停止する。"""
        self._halted.set()
        self._pause.set()

    def abandon(self) -> None:
        """利用者が処理中に LSS を終了した（凍結Design 40a）。処理中の1件は失敗にせず手放す。

        実行中の工程の応答（中断・終了中の拒否・届かない等）を結果として記録しない。キューの状態も
        以後保存しない。保存済みの記録では処理中のままで、次の起動で処理待ちへ戻る。既に公開した
        結果は完了台帳にあり、次の処理で再利用される（処理し直さない）。
        """
        self._abandoned.set()
        self.halt()

    def freeze(self, wanted: dict[str, list[str]], sounds: list[str] | None = None) -> None:
        """STARTで、処理待ちの項目ごとに欲しい結果（と欲しい音の言葉）を凍結する。処理中・終端の項目は動かさない。"""
        checked = {item_id: normalized_targets(targets) for item_id, targets in wanted.items()}
        words = normalized_sounds(sounds)
        if words is None and any(SOUND in targets for targets in checked.values()):
            raise ValueError("欲しい音を選んだ時は、欲しい音の言葉が必要です")
        with self._lock:
            changed = []
            for item in self._items:
                if item.id in checked and item.status == "queued":
                    changed.append((item, item.targets, item.sounds))
                    item.targets = list(checked[item.id])
                    item.sounds = list(words) if SOUND in checked[item.id] else None
            if changed:
                def rollback() -> None:
                    for item, previous, previous_sounds in changed:
                        item.targets = previous
                        item.sounds = previous_sounds
                self._persist_locked(rollback)

    def resolve_existing(self, item_id: str) -> bool:
        """欲しい結果が全部既存で揃う処理待ちの項目を、処理せずに終端へ確定する。"""
        with self._lock:
            item = next((item for item in self._items if item.id == item_id), None)
            if item is None or item.status != "queued":
                return False
            candidate = deepcopy(item)
        found = {}
        for target in candidate.unresolved():
            outcome = self._existing_outcome(candidate, target)
            if outcome is None:
                return False
            found[target] = outcome
        with self._lock:
            if item.status != "queued":
                return False
            previous = (item.status, item.outcomes, item.progress)
            item.outcomes = {**(item.outcomes or {}), **found}
            item.status = "completed"
            item.progress = {"phase": "completed"}

            def rollback() -> None:
                item.status, item.outcomes, item.progress = previous

            self._persist_locked(rollback)
        return True

    def _existing_outcome(self, item: QueueItem, target: str) -> dict | None:
        if self._existing is None:
            return None
        try:
            outcome = self._existing(item, target)
        except Exception:
            # 照会できないものは「無い」として作り直す。既存と偽らない。
            return None
        if not isinstance(outcome, dict) or outcome.get("k") not in {"reused", "none"}:
            return None
        return dict(outcome)

    def retry(self, item_id: str) -> None:
        with self._lock:
            item = next((item for item in self._items if item.id == item_id), None)
            if item is None:
                raise KeyError(item_id)
            if item.status != "failed":
                raise ValueError("再試行できるのは失敗した項目だけです")
            previous = (item.status, item.result, item.progress, item.outcomes)

            def rollback() -> None:
                item.status, item.result, item.progress, item.outcomes = previous

            item.status = "queued"
            item.progress = None
            # 受理済みの結果は残し、未達の結果だけを次の実行で繰り返す。
            kept = {key: value for key, value in (item.outcomes or {}).items()
                    if value.get("k") in ACCEPTED_OUTCOMES}
            item.outcomes = kept or None
            if kept.get(INSTRUMENTAL) is None:
                item.result = None
            self._persist_locked(rollback)

    def _record_progress(self, item_id: str, event: dict) -> None:
        if not isinstance(event, dict):
            return
        try:
            json.dumps(event, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return
        copied = deepcopy(event)
        with self._lock:
            item = next((candidate for candidate in self._items if candidate.id == item_id), None)
            if item is None or item.status != "running":
                return
            item.progress = copied
        if self._on_progress is not None:
            try:
                self._on_progress(item_id, deepcopy(copied))
            except Exception:
                # 表示側の通知失敗で順次処理を止めない。
                pass

    def _dispatch_instrumental(self, item: QueueItem) -> dict:
        job = {"mix_path": str(item.path), "model_id": item.model_id, "stem_target": item.stem_target}
        try:
            if self._streams_progress:
                result = self._dispatch(
                    job, on_progress=lambda event: self._record_progress(item.id, event),
                )
            else:
                result = self._dispatch(job)
            return validated_worker_result(result)
        except (EOFError, OSError, TimeoutError) as exc:
            return {"status": "error", "error_code": "worker_unavailable", "message": str(exc)}
        except Exception as exc:
            return {"status": "error", "error_code": "dispatch_failed", "message": str(exc)}

    def _dispatch_bgm(self, item: QueueItem) -> tuple[dict, dict]:
        """単話BGM候補の工程。worker形式の結果と、結果ごとの到達を返す。"""
        if self._bgm is None:
            result = {"status": "error", "error_code": "capability_unsupported",
                      "message": "BGM候補の処理経路がありません"}
            return result, {"k": "unsupported", "error_code": "capability_unsupported"}

        def dispatch(job: dict, on_progress=None) -> dict:
            if self._streams_progress:
                return self._dispatch(job, on_progress=on_progress)
            return self._dispatch(job)

        try:
            result = self._bgm(
                item.path, dispatch, lambda event: self._record_progress(item.id, event),
            )
        except (EOFError, OSError, TimeoutError) as exc:
            result = {"status": "error", "error_code": "worker_unavailable", "message": str(exc)}
        except Exception as exc:
            result = {"status": "error", "error_code": "bgm_candidate_processing_failed",
                      "message": str(exc)}
        if not isinstance(result, dict):
            result = {"status": "error", "error_code": "invalid_bgm_candidate_response",
                      "message": "BGM候補処理が不正な結果を返しました"}
        if result.get("status") == "ok":
            outcome = result.get("bgm")
            if (isinstance(outcome, dict) and outcome.get("k") in {"done", "none", "held"}
                    and isinstance(outcome.get("n", 0), int) and outcome.get("n", 0) >= 0):
                return result, dict(outcome)
            result = {"status": "error", "error_code": "invalid_bgm_candidate_response",
                      "message": "BGM候補処理が不正な結果を返しました"}
        code = result.get("error_code") or "bgm_candidate_processing_failed"
        result = {**result, "status": "error", "error_code": code}
        kind = "unsupported" if code in _UNSUPPORTED_CODES else "failed"
        outcome = {"k": kind, "error_code": code}
        if isinstance(result.get("message"), str) and result["message"]:
            # 画面には出さない内部の診断。失敗の理由を後から追えるように残す。
            outcome["message"] = result["message"][:2000]
        return result, outcome

    def _dispatch_sound(self, item: QueueItem) -> tuple[dict, dict]:
        """欲しい音の工程（workerの cmd=wanted_sound）。worker形式の結果と、結果ごとの到達を返す。"""
        if not item.sounds:
            result = {"status": "error", "error_code": "capability_unsupported",
                      "message": "欲しい音の言葉がありません"}
            return result, {"k": "unsupported", "error_code": "capability_unsupported"}
        job = {"cmd": "wanted_sound", "mix_path": str(item.path), "sounds": list(item.sounds)}
        try:
            if self._streams_progress:
                result = self._dispatch(job, on_progress=lambda event: self._record_progress(item.id, event))
            else:
                result = self._dispatch(job)
        except (EOFError, OSError, TimeoutError) as exc:
            result = {"status": "error", "error_code": "worker_unavailable", "message": str(exc)}
        except Exception as exc:
            result = {"status": "error", "error_code": "dispatch_failed", "message": str(exc)}
        if not isinstance(result, dict):
            result = {"status": "error", "error_code": "invalid_worker_response",
                      "message": "欲しい音の処理が不正な結果を返しました"}
        if result.get("status") == "ok":
            outcome = result.get("sound")
            if (isinstance(outcome, dict) and outcome.get("k") in _OUTCOME_KINDS
                    and isinstance(outcome.get("n", 0), int) and not isinstance(outcome.get("n", 0), bool)
                    and outcome.get("n", 0) >= 0):
                kept = {key: deepcopy(outcome[key]) for key in ("k", "n", "files", "items") if key in outcome}
                if isinstance(result.get("product_result"), dict):
                    kept["product_result"] = deepcopy(result["product_result"])
                return result, kept
            result = {"status": "error", "error_code": "invalid_worker_response",
                      "message": "欲しい音の処理が不正な結果を返しました"}
        code = result.get("error_code") or "processing_failed"
        result = {**result, "status": "error", "error_code": code}
        outcome = {"k": "unsupported" if code in _UNSUPPORTED_CODES else "failed", "error_code": code}
        if isinstance(result.get("message"), str) and result["message"]:
            outcome["message"] = result["message"][:2000]
        return result, outcome

    def _notify(self, item_id: str, result: dict) -> None:
        # 終端phaseが1度も流れない経路（再利用スキップ・検証済みの空出力・
        # workerへ届かなかった失敗）があるため、終端は結果として同期で通知する。
        # ここでpause()/halt()すれば次へ進まない。表示側の周期に依存しない。
        if self._on_result is None:
            return
        try:
            self._on_result(item_id, result)
        except Exception:
            # 継続可否を判断できないまま次へ進めない。結果は保存済み。
            self._pause.set()
            raise

    def _run_legacy(self, item: QueueItem) -> None:
        """欲しい結果を凍結していない登録（伴奏だけ）。従来どおり1回の分離で終端を決める。"""
        try:
            result = self._dispatch_instrumental(item)
        except BaseException:
            with self._lock:
                item.status = "queued"
                item.progress = None
                self._save_locked()
            raise
        if self._abandoned.is_set():
            return
        with self._lock:
            item.result = deepcopy(result)
            item.status = {"ok": "completed", "skipped": "skipped", "error": "failed"}[result["status"]]
            if item.progress is None or item.progress.get("phase") not in {"completed", "failed"}:
                item.progress = {"phase": "failed" if item.status == "failed" else "completed"}

            def rollback_result() -> None:
                # 成果物の公開とキュー保存は別transaction。保存に失敗した終端を
                # メモリだけで確定すると再開時に通知を失うため、再回収の対象へ戻す。
                # 公開済み音声はworkerの完了台帳で再利用され、ここでは消さない。
                item.status = "queued"
                item.result = None
                item.progress = None

            self._persist_locked(rollback_result)
            notified = deepcopy(result)
        self._notify(item.id, notified)

    def _run_targets(self, item: QueueItem) -> None:
        """凍結した欲しい結果を、表示順に1つずつ到達させる。受理済みは繰り返さない。

        既存の結果で足りるものは処理しない。HALTなら残りの結果へ進まず止まる。
        STOPは現在の音源の結果を全部仕上げてから止まる（pauseは音源の境界でだけ効く）。
        """
        try:
            for target in item.unresolved():
                if self._abandoned.is_set():
                    return
                if self._halted.is_set():
                    break
                with self._lock:
                    candidate = deepcopy(item)
                outcome = self._existing_outcome(candidate, target)
                result = None
                if outcome is None:
                    if target == INSTRUMENTAL:
                        result = self._dispatch_instrumental(item)
                        outcome = outcome_of_worker_result(result)
                    elif target == SOUND:
                        result, outcome = self._dispatch_sound(item)
                    else:
                        result, outcome = self._dispatch_bgm(item)
                if self._abandoned.is_set():
                    # 終了で中断した工程の応答は結果ではない（失敗として残さない）。
                    return
                with self._lock:
                    previous = (item.outcomes, item.result)
                    item.outcomes = {**(item.outcomes or {}), target: outcome}
                    if target == INSTRUMENTAL and result is not None:
                        item.result = deepcopy(result)

                    def rollback_outcome(previous=previous) -> None:
                        item.outcomes, item.result = previous

                    self._persist_locked(rollback_outcome)
                if result is not None:
                    self._notify(item.id, {**deepcopy(result), "target": target})
        except BaseException:
            with self._lock:
                item.status = "queued"
                item.progress = None
                self._save_locked()
            raise
        if self._abandoned.is_set():
            return
        with self._lock:
            outcomes = item.outcomes or {}
            bad = any(value.get("k") in {"failed", "unsupported"} for value in outcomes.values())
            unfinished = any(target not in outcomes for target in item.wanted())
            item.status = "failed" if bad or unfinished else "completed"
            item.progress = {"phase": "failed" if item.status == "failed" else "completed"}

            def rollback_status() -> None:
                item.status = "queued"
                item.progress = None

            self._persist_locked(rollback_status)

    def run(self) -> None:
        if not self._runner.acquire(blocking=False):
            raise RuntimeError("このキューは処理中です")
        try:
            self._pause.clear()
            self._halted.clear()
            while not self._pause.is_set():
                with self._lock:
                    item = next((item for item in self._items if item.status == "queued"), None)
                    if item is None:
                        return
                    item.status = "running"
                    self._persist_locked(lambda: setattr(item, "status", "queued"))
                    legacy = item.targets is None and not item.outcomes
                if legacy:
                    self._run_legacy(item)
                else:
                    self._run_targets(item)
        finally:
            self._runner.release()
