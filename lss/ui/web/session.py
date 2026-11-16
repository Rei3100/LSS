"""凍結Designのページへ渡す、バックエンドの事実。Qt非依存。

ページ（Claude Designのコンポーネント）は画面の状態機械とモーションを自分で持つ。
ここが持つのは「ページが自分では知りえない事実」だけ。

- 一覧に並ぶSOURCEと、その素性・終端・失敗理由
- SOURCEごとにSTARTで凍結した欲しい結果（伴奏・BGM）と、結果ごとの到達
- 既存の結果（検証できる保存済みの伴奏、検証済みのBGMなし）の照会
- 処理中の1件の工程（PREPARE / SEPARATE / SAVE）と、測れる工程だけの割合
- セッションを止めた条件（HALT）と、実行が止まった理由（着地）
- 実測の経過時間、この曲の経過、推定できるときだけの残り時間、由来（RECIPE）

READYは完了済みの履歴（ALREADY SAVED）を一覧へ戻さない。保存済みの知識は、
同じ音源が並んだときの「既存の結果」としてだけ使う。

事実はJSONにできる辞書（スナップショット）で渡す。変わるたびに `on_change` を呼ぶ。
呼び出し元のスレッドは問わない（キューの実行スレッドからも呼ばれる）。

HALTの判断はキューの実行スレッド上で同期に行う。表示を待ってから止めると、
次のSOURCE（または同じSOURCEの次の結果）が始まってしまうため。
"""
from __future__ import annotations

import json
import queue as _queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

from lss.core import jobs
from lss.core.jobs import QueueItem, SeparationQueue
from lss.ui import contract
from lss.ui.presentation import _FAILURE_NOTE, RowKind, row_for

_ACTIVE = frozenset({"queued", "running"})
_TERMINAL = frozenset({"completed", "skipped", "failed"})

# 凍結Designの工程キー。
_STAGE_KEY = {
    contract.Stage.PREPARE: "PREP",
    contract.Stage.SEPARATE: "SEPARATE",
    contract.Stage.SAVE: "SAVE",
}

# 凍結DesignのFAULTSのキー。文言はDesign側が持つ。
_FAULT_KEY = {
    contract.HaltCode.DISK_FULL: "insufficient_output_space",
    contract.HaltCode.FETCH_FAILED: "model_download_failed",
    contract.HaltCode.ENGINE_STOPPED: "worker_unavailable",
}

_RESULT_VIEW_FIELDS = ("schema_version", "state", "published_count", "held_count", "unsupported_count", "rebuilt_count",
                       "replacement_count", "available_count", "related_recording_count", "related_named_count", "recurring_count",
                       "recurrence_asset_count", "recurrence_asset_updated_count",
                       "report_path")


def _result_view(result: dict) -> dict:
    """Product resultのうち、完了画面が言う事実だけ（全体の記録は保存先の結果票と状態にある）。

    未確認の件数（None）は未確認のまま渡し、0にしない。"""
    return {name: result.get(name) for name in _RESULT_VIEW_FIELDS if name in result}


# 行の終端。Designの `st` の語彙（欲しい結果を凍結していない旧形式の登録）。
_ROW_STATE = {
    RowKind.DONE: "done",
    RowKind.SAVED: "done",
    RowKind.SKIPPED: "skipped",
    RowKind.FAILED: "failed",
    RowKind.UNSUPPORTED: "failed",
}

_BAD = frozenset({"failed", "unsupported"})
# 残り時間を出すのに必要な、同じ結果の組での実測の件数。1件では幅を言えない。
_FORECAST_SAMPLES = 2


def _song(number: str, path: Path, media) -> dict:
    """Designの曲データの形。読めなかった素性は空のまま（推測で埋めない）。"""
    return {
        "num": number,
        "name": path.stem,
        "fmt": media.media_format,
        "dur": media.duration,
        "mb": media.megabytes,
        "sr": media.sample_rate,
        "bit": media.bit_depth,
        "ch": media.channels,
        "dir": str(path.parent),
        "path": str(path),
    }


def _seconds(duration: str) -> float | None:
    """m:ss / h:mm:ss を秒へ。読めなかった長さはNone。"""
    try:
        parts = [int(part) for part in duration.split(":")]
    except (AttributeError, ValueError):
        return None
    if not 2 <= len(parts) <= 3:
        return None
    total = 0
    for part in parts:
        total = total * 60 + part
    return float(total) if total > 0 else None


def default_recipe() -> str:
    try:
        from lss.core import catalog

        return str(catalog.default_model()["model_id"])
    except Exception:
        return ""


def model_cached(model_id: str) -> bool | None:
    """分離モデルの重みが手元にあるか（サイズだけを見る。ハッシュはしない）。

    分からなければNone。初回の注記は、確かに無いと言えるときだけ出す。
    """
    try:
        from lss.core import catalog
        from lss.harness import fetch

        row = catalog.get_model(model_id)
        ckpt_path, _config = fetch.model_paths(
            model_id, row["distributor_url"], row["config_url"])
        size = int(row["size_bytes"]) if str(row.get("size_bytes", "")).isdigit() else None
        return fetch.is_cached(ckpt_path, size)
    except Exception:
        return None


def _default_lookup(path: Path, sha256: str) -> dict[str, dict | None]:
    """既存の結果の照会（処理はしない）。見つからない・照会できないものはNone。"""
    from lss import application

    found: dict[str, dict | None] = {}
    for target, lookup in ((jobs.INSTRUMENTAL, application.existing_accompaniment),
                           (jobs.BGM, application.existing_bgm_candidates)):
        try:
            found[target] = lookup(path, sha256)
        except Exception:
            found[target] = None
    return found


def _default_bgm(path: Path, dispatch, on_progress) -> dict:
    from lss import application

    return application.bgm_candidate_step(path, dispatch, on_progress)


def _signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_size, stat.st_mtime_ns)


class _Existing:
    """既存の結果の照会を、画面を止めずに1本の作業スレッドで行い、署名付きで覚える。

    覚えた結果は、原本と記録された出力の大きさ・更新時刻が変わっていない間だけ使う。
    """

    def __init__(self, lookup, on_found: Callable[[], None]) -> None:
        self._lookup = lookup
        self._on_found = on_found
        self._lock = threading.Lock()
        self._known: dict[str, tuple] = {}
        self._asked: set[tuple] = set()
        self._requests: _queue.Queue = _queue.Queue()
        self._thread: threading.Thread | None = None

    def request(self, path: Path) -> None:
        key = str(path)
        signature = _signature(path)
        if signature is None:
            return
        with self._lock:
            known = self._known.get(key)
            if known is not None and known[0] == signature:
                return
            if (key, signature) in self._asked:
                return
            self._asked.add((key, signature))
            # 積むのと作業スレッドの有無の確認を同じ錠の中で行い、取りこぼさない。
            self._requests.put(path)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._work, name="lss-existing", daemon=True)
                self._thread.start()

    def _work(self) -> None:
        from lss.core.session import input_sha256_of

        while True:
            try:
                path = self._requests.get(timeout=2.0)
            except _queue.Empty:
                with self._lock:
                    if self._requests.empty():
                        self._thread = None
                        return
                continue
            signature = _signature(path)
            if signature is None:
                continue
            try:
                sha256 = input_sha256_of(path)
                found = self._lookup(path, sha256)
            except Exception:
                continue
            if _signature(path) != signature:
                continue
            outputs = []
            for outcome in found.values():
                for output in (outcome or {}).get("outputs", []):
                    outputs.append((output, _signature(Path(output))))
            with self._lock:
                self._known[str(path)] = (signature, found, tuple(outputs))
            if any(found.values()):
                self._on_found()

    def invalidate(self) -> None:
        """結果が確定した（保存・記録が増えた）。照会し直す。"""
        with self._lock:
            self._known.clear()
            self._asked.clear()

    def get(self, path: Path, target: str) -> dict | None:
        """いまも有効な既存の到達。原本か出力が変わっていれば使わない。"""
        with self._lock:
            known = self._known.get(str(path))
        if known is None or _signature(path) != known[0]:
            return None
        if any(_signature(Path(output)) != signature for output, signature in known[2]):
            return None
        outcome = known[1].get(target)
        if outcome is None:
            return None
        return {key: value for key, value in outcome.items() if key != "outputs"}


class WebSession:
    """1つの処理セッション。ページの操作をキューへ、キューの事実をページへ渡す。"""

    def __init__(
        self,
        *,
        dispatch=None,
        state_path=None,
        on_change: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        lookup=None,
        bgm=None,
        wanted_path=None,
    ) -> None:
        """`dispatch`・`state_path`・`lookup`・`bgm`・`wanted_path` は検証用。
        既定は実ワーカー・永続キュー・完了台帳の照会・単話BGM候補経路。"""
        self._lock = threading.RLock()
        self._on_change = on_change
        self._clock = clock
        self._existing = _Existing(lookup or _default_lookup, self._changed)
        options = {} if state_path is None else {"state_path": state_path}
        self._queue = SeparationQueue(
            dispatch, on_progress=self._on_progress, on_result=self._on_result,
            existing=self._existing_outcome, bgm=bgm or _default_bgm, **options,
        )
        # READYは完了済みの履歴を戻さない。前回までの終端（保存・再利用）は一覧から外す。
        # 失敗は再試行の対象なので残す。
        for item in self._queue.snapshot():
            if item.status in {"completed", "skipped"}:
                self._queue.remove(item.id)
        self._stage: contract.StageProgress | None = None
        self._stage_item: str | None = None
        self._fetching = False
        # 初回の長い準備（モデル取得）が起こると確かに言えるか。
        self._first_run = False
        self._halt: contract.HaltCode | None = None
        self._halted_item: str | None = None
        self._halt_seq = 0
        self._stop_requested = False
        self._runner: threading.Thread | None = None
        self._started_at: float | None = None
        self._elapsed = 0.0
        self._settle_seq = 0
        self._settle_kind = ""
        self._arrival_seq = 0
        self._arrival_ids: list[str] = []
        self._recipe = ""
        # 欲しい結果の既定。STARTで凍結した既定は、処理中に追加された音源に使う。
        self._wanted_path = Path(wanted_path) if wanted_path is not None else None
        self._default = self._load_default()
        self._frozen_default: list[str] | None = None
        # 欲しい音（言葉）の共通設定。入力欄の文字列のまま覚え、STARTで項目ごとに凍結する。
        self._sounds_text = self._load_sounds()
        self._frozen_sounds: list[str] | None = None
        # この曲の経過と、残り時間の根拠（実測）。
        self._current: str | None = None
        self._current_at: float | None = None
        self._current_spent = 0.0
        self._processed: dict[str, set[str]] = {}
        self._rates: dict[tuple, list[float]] = {}

    # --- 通知 -------------------------------------------------------------

    def set_listener(self, on_change: Callable[[], None] | None) -> None:
        self._on_change = on_change

    def _changed(self) -> None:
        listener = self._on_change
        if listener is None:
            return
        try:
            listener()
        except Exception:
            # 表示側の失敗で処理を止めない。
            pass

    # --- 欲しい結果の既定 -------------------------------------------------

    def _default_path(self) -> Path:
        if self._wanted_path is not None:
            return self._wanted_path
        from lss.core.session import state_dir

        return state_dir() / "wanted.json"

    def _load_default(self) -> list[str]:
        try:
            value = json.loads(self._default_path().read_text(encoding="utf-8")).get("default")
            return jobs.normalized_targets(value)
        except (OSError, ValueError, AttributeError, TypeError):
            return [jobs.INSTRUMENTAL]

    def _load_sounds(self) -> str:
        try:
            value = json.loads(self._default_path().read_text(encoding="utf-8")).get("sounds")
            return value[:2000] if isinstance(value, str) else ""
        except (OSError, ValueError, AttributeError, TypeError):
            return ""

    def _save_default(self, value: list[str], sounds: str | None = None) -> None:
        path = self._default_path()
        payload = {"default": value}
        text = self._sounds_text if sounds is None else sounds
        if text:
            payload["sounds"] = text
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            # 既定を覚えられなくても処理は進める。次回は伴奏から始まる。
            pass

    # --- 既存の結果 -------------------------------------------------------

    def _existing_outcome(self, item: QueueItem, target: str) -> dict | None:
        found = self._existing.get(Path(item.path), target)
        if found is None:
            return None
        if found.get("k") == "none":
            return {"k": "none", "n": 0, "existing": True}
        reused = {"k": "reused", "n": found.get("n", 1)}
        # 公開済みのBGM cueは、実際のファイル名と保留した区間数も画面へ渡す。
        for key in ("files", "held"):
            if key in found:
                reused[key] = found[key]
        return reused

    @staticmethod
    def _have_value(outcome: dict) -> object:
        """Designの `have` の値。結果の数、または検証済みの無し（'none'）。"""
        if outcome.get("k") == "none":
            return "none"
        return max(1, int(outcome.get("n") or 1))

    # --- スナップショット -------------------------------------------------

    def snapshot(self) -> dict:
        from lss.ui.media import facts
        from lss.ui.media import request as probe_media

        with self._lock:
            items = self._queue.snapshot()
            songs: dict[str, dict] = {}
            errs: dict[str, list] = {}
            st: dict[str, str] = {}
            tst: dict[str, dict] = {}
            have: dict[str, dict] = {}
            wants: dict[str, list[str]] = {}
            sounds_of: dict[str, list[str]] = {}
            queue_ids: list[str] = []
            running: str | None = None
            for number, item in enumerate(items, start=1):
                path = Path(item.path)
                media = facts(path)
                if not media.sample_rate:
                    probe_media(path, self._changed)
                song = _song(f"{number:02d}", path, media)
                queue_ids.append(item.id)
                if item.targets is not None:
                    wants[item.id] = list(item.targets)
                if item.sounds:
                    sounds_of[item.id] = list(item.sounds)
                if item.status == "running":
                    running = item.id
                if item.status in _ACTIVE:
                    self._existing.request(path)
                known = self._have_of(item)
                if known:
                    have[item.id] = known
                if item.status in _TERMINAL and item.targets is not None:
                    st[item.id], tst[item.id] = self._terminal(item, song, errs, media)
                elif item.status in _TERMINAL and item.result is not None:
                    row = row_for(number, path.stem, result=item.result, media=media)
                    state = _ROW_STATE.get(row.kind, "queued")
                    st[item.id] = state
                    if isinstance(item.result, dict) and item.result.get("zero_result") is not None:
                        # 検証済みの空出力。Designの行はSKIPPEDだが「分離済み」とは言わない。
                        song["zero"] = True
                    if state == "failed":
                        key = "e:" + item.id
                        errs[key] = [row.badge, row.note, bool(row.retryable)]
                        song["fail" if row.retryable else "hard"] = key
                else:
                    st[item.id] = "queued"
                songs[item.id] = song

            stage = None
            if running is not None and self._stage is not None and self._stage_item == running:
                fraction = self._stage.fraction
                stage = {"k": _STAGE_KEY[self._stage.stage],
                         "p": None if fraction is None else round(fraction, 4)}

            fault = None
            if self._halt is not None:
                # 数値は実測できたものだけ。現行のworkerは空き・必要量を返さない。
                fault = {"key": _FAULT_KEY[self._halt], "metrics": [], "seq": self._halt_seq,
                         "item": self._halted_item}

            return {
                "queue": queue_ids,
                # READYは完了済みの履歴を一覧へ戻さない。
                "hist": [],
                "songs": songs,
                "st": st,
                "tst": tst,
                "have": have,
                "wants": wants,
                "bdef": list(self._default),
                "frozenDef": list(self._frozen_default) if self._frozen_default else None,
                "soundsText": self._sounds_text,
                "soundsOf": sounds_of,
                "errs": errs,
                "running": running,
                "stage": stage,
                "firstRun": self._first_run or self._fetching,
                "elapsed": int(round(self.elapsed() * 1000)),
                "curElapsed": int(round(self._current_elapsed() * 1000)),
                "forecast": self._forecast(items),
                "clockRunning": self._started_at is not None,
                "fault": fault,
                "settle": {"seq": self._settle_seq, "kind": self._settle_kind},
                "arrival": {"seq": self._arrival_seq, "ids": list(self._arrival_ids)},
                "recipe": self.recipe(items),
            }

    def _have_of(self, item: QueueItem) -> dict:
        """その音源について、いま使える既存の結果（受理済みの到達と照会結果）。"""
        known: dict[str, object] = {}
        for target, outcome in (item.outcomes or {}).items():
            if outcome.get("k") in jobs.ACCEPTED_OUTCOMES:
                known[target] = self._have_value(outcome)
        if item.status in _ACTIVE:
            for target in jobs.TARGETS:
                if target in known:
                    continue
                found = self._existing_outcome(item, target)
                if found is not None:
                    known[target] = self._have_value(found)
        return known

    def _terminal(self, item: QueueItem, song: dict, errs: dict, media) -> tuple[str, dict]:
        """凍結した欲しい結果を持つ項目の終端。行の状態と、結果ごとの到達。"""
        outcomes = item.outcomes or {}
        tst: dict[str, dict] = {}
        for target in item.wanted():
            outcome = outcomes.get(target)
            if outcome is None:
                # HALTのSKIPなどで到達しなかった結果は、作れなかった結果として言う。
                tst[target] = {"k": "failed"}
                continue
            kind = outcome.get("k")
            if outcome.get("existing") and kind == "none":
                kind = "reused"
            result = outcome.get("product_result") if isinstance(outcome.get("product_result"), dict) else None
            if result is not None and kind == "done" and result.get("state") == "REUSED":
                # 何も新しく作らず保存済みの音声を使った（Product resultの判断）。新しい保存と同じ表示にしない。
                kind = "reused"
            tst[target] = {"k": kind, "n": int(outcome.get("n") or 0)}
            if result is not None:
                tst[target]["result"] = _result_view(result)
            if isinstance(outcome.get("files"), list):
                tst[target]["files"] = [str(name) for name in outcome["files"]][:500]
            if isinstance(outcome.get("held"), int) and not isinstance(outcome.get("held"), bool):
                tst[target]["held"] = outcome["held"]
            if target == jobs.SOUND and isinstance(outcome.get("items"), list):
                # 欲しい音ごとの到達（読めなかった語・見つからなかった音を行で言うため）。
                tst[target]["items"] = [
                    {"label": str(entry.get("label", ""))[:120], "text": str(entry.get("text", ""))[:120],
                     "k": str(entry.get("k", "")), "reason": str(entry.get("reason", ""))[:60],
                     "unknown": [str(word)[:60] for word in entry.get("unknown", [])][:6],
                     "heard_as": [str(words)[:80] for words in entry.get("heard_as", [])][:4],
                     "suggestions": [str(word)[:40] for word in entry.get("suggestions", [])][:3],
                     **({"removal_status": entry["removal_status"]}
                        if entry.get("removal_status") in ("unchanged", "removed") else {})}
                    for entry in outcome["items"][:12] if isinstance(entry, dict)]
        if any((tst.get(target) or {}).get("k") in {"done", "reused"} and tst[target]["n"] > 0
               for target in (jobs.INSTRUMENTAL, jobs.BGM, jobs.SOUND)):
            # 公開したBGM cueも、音源の隣のLSSフォルダ（その中の「- BGM」フォルダ）へ保存している。
            song["out"] = True
        bad = [target for target, outcome in tst.items() if outcome["k"] in _BAD]
        if not bad:
            return "done", tst
        accepted = [target for target, outcome in tst.items() if outcome["k"] not in _BAD]
        if not accepted:
            # 音源全体の失敗。理由は最初に失敗した結果のもの。
            key = "e:" + item.id
            errs[key] = self._whole_failure(item, bad[0], media)
            song["fail" if errs[key][2] else "hard"] = key
        return "failed", tst

    def _whole_failure(self, item: QueueItem, target: str, media) -> list:
        outcome = (item.outcomes or {}).get(target) or {}
        if target == jobs.INSTRUMENTAL and isinstance(item.result, dict) \
                and item.result.get("status") == "error":
            row = row_for(None, Path(item.path).stem, result=item.result, media=media)
            return [row.badge, row.note, bool(row.retryable)]
        code = outcome.get("error_code")
        unsupported = outcome.get("k") == "unsupported"
        failure = contract.classify_failure(code)
        note = _FAILURE_NOTE.get(code, "") if isinstance(code, str) else ""
        if unsupported and code == "unsupported_channels":
            note = f"{media.channels}ch の音源は分離できません。ステレオまで対応しています。" \
                if media.channels > 2 else "ステレオまで対応しています。"
        return ["UNSUPPORTED" if unsupported else "FAILED", note,
                not unsupported and failure.retryable]

    def snapshot_json(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False)

    def elapsed(self) -> float:
        with self._lock:
            if self._started_at is None:
                return self._elapsed
            return self._elapsed + (self._clock() - self._started_at)

    def _current_elapsed(self) -> float:
        with self._lock:
            if self._current_at is None:
                return self._current_spent
            return self._current_spent + (self._clock() - self._current_at)

    def elapsed_running(self) -> bool:
        """経過時間が進んでいるか（ページへ毎秒届ける必要があるか）。"""
        with self._lock:
            return self._started_at is not None

    def _forecast(self, items: list[QueueItem]) -> str | None:
        """残り時間の幅。同じ結果の組をこのセッションで実測した件数が足りるときだけ。

        幅は実測した「音源1秒あたりの処理時間」の最小と最大から。長さを読めない音源、
        実測の無い結果の組、初回の準備（取得）が残る間は出さない。
        """
        if self._started_at is None or self._halt is not None or self._first_run \
                or self._fetching:
            return None
        from lss.ui.media import clock, facts

        low = high = 0.0
        remaining = [item for item in items if item.status in _ACTIVE]
        if not remaining:
            return None
        for item in remaining:
            todo = tuple(target for target in item.unresolved()
                         if self._existing_outcome(item, target) is None)
            if not todo:
                continue
            rates = self._rates.get(todo, [])
            seconds = _seconds(facts(Path(item.path)).duration)
            if len(rates) < _FORECAST_SAMPLES or seconds is None:
                return None
            spent = self._current_elapsed() if item.status == "running" else 0.0
            low += max(0.0, seconds * min(rates) - spent)
            high += max(0.0, seconds * max(rates) - spent)
        if high <= 0:
            return None
        low = 10 * int(low // 10)
        high = 10 * int(-(-high // 10))
        return clock(low) + " – " + clock(high)

    def recipe(self, items: list[QueueItem] | None = None) -> str:
        """RECIPE。使用したモデル識別子で、選択肢ではなく由来の記録。

        結果のRecipe revisionに記録された値を優先し、無ければ既定の分離経路のモデル。
        """
        for item in reversed(items if items is not None else self._queue.snapshot()):
            revision = item.result.get("recipe_revision") if isinstance(item.result, dict) else None
            stages = revision.get("stage_graph") if isinstance(revision, dict) else None
            if isinstance(stages, list) and stages and isinstance(stages[0], dict):
                value = stages[0].get("model_id")
                if isinstance(value, str) and value:
                    return value
        if not self._recipe:
            self._recipe = default_recipe()
        return self._recipe

    # --- SOURCEの出入り ---------------------------------------------------

    def add(self, paths: list[Path]) -> list[str]:
        """追加。処理中の追加は、そのセッションで凍結した既定の欲しい結果で受ける。"""
        with self._lock:
            # STOPで凍結は解ける。HALT中（実行は止まっている）も同じセッションの続き。
            frozen = list(self._frozen_default) if self._frozen_default else None
            sounds = list(self._frozen_sounds) if self._frozen_sounds else None
        added = self._queue.add([Path(path) for path in paths], targets=frozen, sounds=sounds)
        if added:
            with self._lock:
                self._arrival_seq += 1
                self._arrival_ids = list(added)
        self._changed()
        return added

    def remove(self, item_id: str) -> bool:
        removed = self._queue.remove(item_id)
        if removed:
            self._changed()
        return removed

    def retry_row(self, item_id: str) -> bool:
        """失敗行を未処理へ戻す。受理済みの結果は残し、未達の結果だけを繰り返す。"""
        try:
            self._queue.retry(item_id)
        except (KeyError, ValueError):
            return False
        self._changed()
        return True

    def source_path(self, item_id: str) -> str:
        for item in self._queue.snapshot():
            if item.id == item_id:
                return str(item.path)
        return ""

    # --- 主操作 -----------------------------------------------------------

    def start(self, wanted: dict | None = None, default: list | None = None, sounds: str | None = None) -> dict:
        """START / RESUME。音源ごとの欲しい結果を凍結して始める。

        `wanted` は {item_id: [target, ...]}（ページが解決した既定または曲ごとの設定）、
        `default` はバッチの既定（処理中に追加された音源に使う）、`sounds` は欲しい音の入力欄の
        文字列（欲しい音を選んだ音源に共通）。欲しい結果が全部既存で揃う音源はここで処理せずに確定し、
        処理が1件も残らなければ実行を始めない。戻り値の ``work`` は処理する音源の数。
        欲しい音を選んだのに言葉が無ければ始めない（``reason: "sounds_required"``）。
        """
        with self._lock:
            if self._halt is not None or self.running():
                return {"started": False, "work": 0}
            pending = [item for item in self._queue.snapshot() if item.status == "queued"]
            if not pending:
                return {"started": False, "work": 0}
            try:
                base = jobs.normalized_targets(default) if default is not None else self._default
                frozen = {}
                for item in pending:
                    value = (wanted or {}).get(item.id)
                    frozen[item.id] = jobs.normalized_targets(value) if value is not None \
                        else list(base)
                text = sounds if isinstance(sounds, str) else (self._sounds_text if sounds is None else "")
                words = jobs.normalized_sounds(text)
            except ValueError:
                return {"started": False, "work": 0}
            if words is None and (jobs.SOUND in base or any(jobs.SOUND in value for value in frozen.values())):
                return {"started": False, "work": 0, "reason": "sounds_required"}
            self._default = list(base)
            self._frozen_default = list(base)
            self._sounds_text = text[:2000]
            self._frozen_sounds = list(words) if words else None
            self._stop_requested = False
            self._queue.freeze(frozen, sounds=words)
        self._save_default(list(base), self._sounds_text)
        for item in pending:
            self._queue.resolve_existing(item.id)
        work = sum(1 for item in self._queue.snapshot() if item.status == "queued")
        if not work:
            with self._lock:
                # 保存済みだけ。処理せずに完了する（偽のPROCESSINGを経ない）。
                self._settle_seq += 1
                self._settle_kind = "drained"
            self._changed()
            return {"started": True, "work": 0}
        with self._lock:
            if self._started_at is None:
                self._started_at = self._clock()
            self._first_run = model_cached(default_recipe()) is False
        self._start_runner()
        return {"started": True, "work": work}

    def request_stop(self) -> None:
        """安全停止。進行中の1件は仕上げ、次のSOURCEは始めない。HALT中は受け付けない。"""
        with self._lock:
            if self._halt is not None:
                return
            self._stop_requested = True
        self._queue.pause()
        self._changed()

    def back_to_ready(self) -> None:
        """COMPLETEからREADYへ。保存・再利用で終わった音源は一覧を離れ、失敗は残る。

        保存済みの知識は既存の結果の照会として残る（履歴一覧は作らない）。
        """
        with self._lock:
            if self._runner is not None and self._runner.is_alive():
                return
            for item in self._queue.snapshot():
                if item.status in {"completed", "skipped"}:
                    self._queue.remove(item.id)
            self._elapsed = 0.0
            self._started_at = None
            self._stage = None
            self._stage_item = None
            self._stop_requested = False
            self._frozen_default = None
            self._frozen_sounds = None
            self._rates.clear()
        self._changed()

    # --- HALTからの回復 ---------------------------------------------------

    def halt_retry(self) -> None:
        """同じ音源から続行する（受理済みの結果は繰り返さない）。"""
        with self._lock:
            item_id = self._halted_item
        if item_id is not None:
            try:
                self._queue.retry(item_id)
            except (KeyError, ValueError):
                pass
        self._resume_after_halt()

    def halt_skip(self) -> None:
        """その音源の未達の結果を失敗のまま、次へ進む。"""
        self._resume_after_halt()

    def _resume_after_halt(self) -> None:
        with self._lock:
            if self._halt is None:
                return
            self._halt = None
            self._halted_item = None
            if self._started_at is None:
                self._started_at = self._clock()
            stopping = self._stop_requested
        if stopping:
            # 停止要求が生きていれば、やり直した1件で止まる（意思を捨てない）。
            self._queue.pause()
        self._start_runner()

    # --- キューからの通知（実行スレッド） ---------------------------------

    def _enter(self, item_id: str) -> None:
        """この曲の経過を、新しい音源に入った時点から測る。"""
        if self._current != item_id:
            self._finish_current(self._queue.snapshot())
            self._current = item_id
            self._current_spent = 0.0
            self._current_at = self._clock()
        elif self._current_at is None:
            self._current_at = self._clock()

    def _on_progress(self, item_id: str, event: dict) -> None:
        progress = contract.translate_progress(event)
        if progress is None:
            return
        with self._lock:
            self._enter(item_id)
            self._stage = progress
            self._stage_item = item_id
            if progress.awaiting_model_fetch:
                self._fetching = True
            elif progress.stage is not contract.Stage.PREPARE:
                self._fetching = False
                self._first_run = False
        self._changed()

    def _on_result(self, item_id: str, result: dict) -> None:
        # 保存・完了記録が増えたかもしれない。「既存なし」の照会結果を使い続けない。
        self._existing.invalidate()
        failure = contract.classify_result(result).failure
        target = result.get("target")
        with self._lock:
            self._stage = None
            self._stage_item = None
            self._fetching = False
            self._first_run = False
            if target is not None and result.get("status") == "ok" and not result.get("reused"):
                self._processed.setdefault(item_id, set()).add(target)
            if (failure is not None and failure.scope is contract.FailureScope.SESSION
                    and failure.halt is not None):
                # 次のSOURCEにも、同じSOURCEの次の結果にも進まない。ここで同期に止める。
                self._queue.halt()
                self._halt = failure.halt
                self._halted_item = item_id
                self._halt_seq += 1
                self._pause_current()
        self._changed()

    def _pause_current(self) -> None:
        if self._current_at is not None:
            self._current_spent += self._clock() - self._current_at
            self._current_at = None

    def _finish_current(self, items: list[QueueItem]) -> None:
        """終わった音源の実測を、残り時間の根拠として覚える。"""
        from lss.ui.media import facts

        item_id = self._current
        if item_id is None:
            return
        item = next((item for item in items if item.id == item_id), None)
        if item is None or item.status in _ACTIVE:
            return
        spent = self._current_spent + (
            self._clock() - self._current_at if self._current_at is not None else 0.0)
        processed = self._processed.pop(item_id, set())
        outcomes = item.outcomes or {}
        clean = processed and not any(
            (outcomes.get(target) or {}).get("k") in _BAD for target in processed)
        seconds = _seconds(facts(Path(item.path)).duration)
        if clean and seconds:
            key = tuple(target for target in jobs.TARGETS if target in processed)
            self._rates.setdefault(key, []).append(spent / seconds)
        self._current = None
        self._current_at = None
        self._current_spent = 0.0

    # --- 実行 -------------------------------------------------------------

    def _start_runner(self) -> None:
        with self._lock:
            if self._runner is not None and self._runner.is_alive():
                self._changed()
                return
            self._runner = threading.Thread(target=self._run, name="lss-queue", daemon=True)
            self._runner.start()
        self._changed()

    def _run(self) -> None:
        try:
            self._queue.run()
        except RuntimeError:
            return
        self._settle()

    def _settle(self) -> None:
        """実行が止まった理由を決める。キューは理由を知らせないので、事実から決める。"""
        with self._lock:
            items = self._queue.snapshot()
            self._finish_current(items)
            remaining = sum(1 for item in items if item.status in _ACTIVE)
            # 凍結Designのfinalizeと同じ優先順: HALT → 停止要求 → 全件終端。
            if self._halt is not None:
                kind = "halted"
            elif self._stop_requested or remaining:
                kind = "stopped"
            else:
                kind = "drained"
            if self._started_at is not None:
                self._elapsed += self._clock() - self._started_at
                self._started_at = None
            if kind == "stopped":
                self._stop_requested = False
                # STOP後のREADYでは欲しい結果を編集し直せる。RESUMEで凍結し直す。
                self._frozen_default = None
                self._frozen_sounds = None
            self._stage = None
            self._stage_item = None
            self._settle_seq += 1
            self._settle_kind = kind
        self._changed()

    def running(self) -> bool:
        runner = self._runner
        return runner is not None and runner.is_alive()

    def shutdown(self, *, abandon: bool = False) -> None:
        """終了時。次のSOURCEは始めない。

        `abandon` は利用者が処理中に終了したとき（凍結Design 40a「処理中の 1 件は中断され、保存されません」）。
        同じ音源の残りの結果（伴奏の後のBGM等）にも進まず、処理中の1件は失敗にせず手放す（次の起動で
        処理待ちへ戻る）。このprocessで動くBGM候補の工程は、利用者のフォルダへ公開する前で止まる。
        """
        if abandon:
            from lss.core import quitting

            quitting.request()
            self._queue.abandon()
        else:
            self._queue.pause()
