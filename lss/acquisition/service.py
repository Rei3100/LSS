"""URLから取得して、ローカル入力と同じ分離キューへ渡す。

1件ずつ順に取得する。複数項目の取得は間隔を空け、混雑を示す失敗では間隔を広げる。
同じ論理ソースの有効な取得物があれば取得し直さない（`force` で意図的に取り直せる）。
"""
from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from lss.acquisition import store
from lss.acquisition.store import AcquiredSource, AcquisitionFailure, AuthMethod, FailureKind
from lss.paths import resource_root

PROBE_TIMEOUT_SECONDS = 120
# 形式の申告サイズは概算のことがある。申告より大きくても置き場所が尽きない余裕を持つ。
SPACE_MARGIN_RATIO = 1.25
SPACE_MARGIN_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Pacing:
    """取得先に配慮した間隔。固定の並列度で押し切らない。"""

    between_items: tuple[float, float] = (3.0, 8.0)
    backoff_start: float = 15.0
    backoff_limit: float = 300.0
    attempts: int = 4


@dataclass
class EntryOutcome:
    requested_url: str
    source: AcquiredSource | None = None
    failure: AcquisitionFailure | None = None
    offline_reuse: bool = False

    @property
    def ok(self) -> bool:
        return self.source is not None


@dataclass
class UrlIntake:
    outcomes: list[EntryOutcome]
    queue_ids: list[str] = field(default_factory=list)


def default_backend():
    from lss.acquisition.ytdlp import YtDlpBackend

    return YtDlpBackend()


def probe_media(path: Path, work: Path) -> dict:
    """取得物に音声ストリームがあり復号できることを確かめ、チャンネル数と標本化周波数を得る。

    完成した2ch音源かどうかの判定は分離の入力検証が行う。ここでは取得物として壊れていないかだけを見る。
    """
    ffmpeg = resource_root() / "bin" / "ffmpeg.exe"
    if not ffmpeg.is_file():
        raise AcquisitionFailure(FailureKind.TOOL_UNAVAILABLE, "取得物の検査に必要なFFmpegが見つかりません")
    work.mkdir(parents=True, exist_ok=True)
    probe = work / "probe.wav"
    probe.unlink(missing_ok=True)
    command = [str(ffmpeg), "-nostdin", "-v", "error", "-y", "-i", str(path), "-map", "0:a:0",
               "-vn", "-t", "5", "-c:a", "pcm_f32le", str(probe)]
    try:
        result = subprocess.run(command, capture_output=True, timeout=PROBE_TIMEOUT_SECONDS,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode or not probe.is_file():
            detail = result.stderr.decode("utf-8", errors="replace").strip()[:500]
            raise AcquisitionFailure(FailureKind.INVALID_MEDIA, f"取得物の音声を復号できません: {detail}")
        try:
            info = sf.info(probe)
        except (OSError, sf.LibsndfileError) as exc:
            raise AcquisitionFailure(FailureKind.INVALID_MEDIA, "取得物の音声を読み込めません") from exc
        if info.frames <= 0:
            raise AcquisitionFailure(FailureKind.INVALID_MEDIA, "取得物に音声サンプルがありません")
    except subprocess.TimeoutExpired as exc:
        raise AcquisitionFailure(FailureKind.INVALID_MEDIA, "取得物の検査が時間内に終わりません") from exc
    finally:
        probe.unlink(missing_ok=True)
    timeline = measure_timeline(path, info.channels, info.samplerate)
    return {"audio_channels": info.channels, "sample_rate": info.samplerate, "timeline": timeline}


TIMELINE_TIMEOUT_SECONDS = 1800
_DURATION = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")


def parse_container_duration(stderr: str) -> float | None:
    """FFmpegが入力の見出しに出すコンテナの長さ（時刻の範囲）。無い・N/Aなら None。"""
    match = _DURATION.search(stderr)
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def measure_timeline(path: Path, channels: int, sample_rate: int) -> dict:
    """全体を復号して標本の長さを測り、コンテナの時刻の長さと並べる。

    分離と区間の時刻は復号した標本の時間軸になる。配信用のfMP4には、フレームを実際より短い
    時間として記録するものがあり（Bilibiliで16秒/34分）、元の再生時刻との対応に効くので来歴に残す。
    途中で復号できなければ取得物として壊れているとみなす。
    """
    ffmpeg = resource_root() / "bin" / "ffmpeg.exe"
    command = [str(ffmpeg), "-nostdin", "-hide_banner", "-v", "info", "-i", str(path), "-map", "0:a:0",
               "-vn", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
    stderr: list[bytes] = []
    total = 0
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
        reader = threading.Thread(target=lambda: stderr.append(process.stderr.read()), daemon=True)
        reader.start()
        deadline = time.monotonic() + TIMELINE_TIMEOUT_SECONDS
        while chunk := process.stdout.read(1 << 20):
            total += len(chunk)
            if time.monotonic() > deadline:
                process.kill()
                raise AcquisitionFailure(FailureKind.INVALID_MEDIA, "取得物の全体の復号が時間内に終わりません")
        code = process.wait()
        reader.join(timeout=10)
    text = b"".join(stderr).decode("utf-8", errors="replace")
    if code:
        detail = " / ".join(line.strip() for line in text.splitlines() if "rror" in line)[:500]
        raise AcquisitionFailure(FailureKind.INVALID_MEDIA, f"取得物の音声を最後まで復号できません: {detail}")
    decoded = total // (2 * channels) / sample_rate
    container = parse_container_duration(text)
    return {
        "decoded_seconds": round(decoded, 6),
        "container_seconds": container,
        "difference_seconds": None if container is None else round(decoded - container, 3),
    }


_TS_PACKET = 188


def detected_container(path: Path) -> str | None:
    """中身から分かるコンテナ。HLSを後処理なしでつなぐと、MPEG-TSが.mp4の名前で残るため。"""
    try:
        with open(path, "rb") as stream:
            head = stream.read(_TS_PACKET * 3 + 1)
    except OSError:
        return None
    if len(head) > _TS_PACKET * 2 and all(head[i] == 0x47 for i in (0, _TS_PACKET, _TS_PACKET * 2)):
        return "mpegts"
    return None


def expected_bytes(record: dict) -> int | None:
    """選ばれた形式の申告サイズ。HLS等で分からなければNone（推測で止めない）。"""
    total = 0
    for item in (record.get("format") or {}).get("selected") or []:
        size = item.get("filesize") or item.get("filesize_approx")
        if not isinstance(size, (int, float)) or size <= 0:
            return None
        total += int(size)
    return total or None


def ensure_space(expected: int | None, work: Path, library: Path) -> None:
    """作業領域と取得ライブラリの空きを、取得を始める前に確かめる。"""
    if expected is None:
        return
    needed = int(expected * SPACE_MARGIN_RATIO) + SPACE_MARGIN_BYTES
    volumes: dict[str, Path] = {}
    for folder in (work, library):
        probe = folder
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        resolved = probe.resolve()
        # 同じボリュームなら作業領域から名前を移すだけで、2倍は要らない。
        volumes.setdefault(os.path.splitdrive(str(resolved))[0].lower() or str(resolved), probe)
    for probe in volumes.values():
        free = shutil.disk_usage(probe).free
        if free < needed:
            raise AcquisitionFailure(
                FailureKind.INSUFFICIENT_SPACE,
                f"取得に必要な空き容量が足りません（必要 {needed} バイト、空き {free} バイト）: {probe}")


def _acquire_one(backend, entry: dict, *, requested_url: str, auth: AuthMethod, force: bool,
                 on_progress, should_stop, library: Path | None,
                 state_root: Path | None) -> AcquiredSource:
    description = backend.describe(entry, auth)
    if not force:
        found = store.find_valid(description.logical_source, state_root)
        if found is not None:
            return store.reuse(*found)
    work = store.work_dir_for(description.logical_source)
    with store.WorkLock(work):
        ensure_space(expected_bytes(description.record), work,
                     library if library is not None else store.library_root())
        started = datetime.now(timezone.utc).isoformat()
        staged = backend.download(description, work, auth, on_progress, should_stop=should_stop)
        if on_progress is not None:
            on_progress({"stage": "verify"})
        media = probe_media(staged, work)
        record = dict(description.record)
        container = detected_container(staged)
        if container == "mpegts" and staged.suffix.lower() != ".ts":
            # バイト列は変えず、中身と一致する拡張子だけを与える。
            renamed = staged.with_suffix(".ts")
            staged.replace(renamed)
            staged = renamed
            record["format"] = {**record.get("format", {}), "container_detected": container,
                                "extension_corrected_to": ".ts"}
        record["location"] = {**record.get("location", {}),
                              "requested_url": store.sanitize_url(requested_url)}
        record["acquired_at"] = started
        record["media"] = media
        record["intent"] = "forced_reacquisition" if force else "acquisition"
        source = store.publish(staged, record=record, folder=description.folder,
                               stem=description.stem, state_root=state_root, library=library)
    shutil.rmtree(work, ignore_errors=True)
    return source


# 取得ツールが古いと起こりやすい失敗。更新を前倒しで確かめ、新しい版が入れば1回だけやり直す。
STALE_TOOL_FAILURES = frozenset({
    FailureKind.EXTRACTOR_FAILED, FailureKind.UNAVAILABLE, FailureKind.TOOL_UNAVAILABLE,
})


def _ensure_provider(*, after_failure: bool = False) -> dict | None:
    from lss.acquisition import provider

    if not provider.environment_allows_updates():
        return None
    from lss.acquisition import js_runtime

    results = []
    for ensure in (provider.ensure, js_runtime.ensure):
        try:
            results.append(ensure(after_failure=after_failure))
        except Exception:  # 更新確認の失敗で、導入済みの版による取得を止めない。
            results.append(None)
    # yt-dlpかdenoのどちらかが新しく入れば、やり直す価値がある。
    installed = [item for item in results if item is not None and item.get("status") == "installed"]
    return installed[0] if installed else next((item for item in results if item is not None), None)


def acquire(url: str, *, backend=None, **options) -> list[EntryOutcome]:
    """URLの各項目を取得する。失敗は項目ごとに返し、他の項目の取得を止めない。

    `backend` を省略すると、検証済みのyt-dlpを使い、必要なら検証付きで更新する。
    オプションは `acquire_with` を参照。
    """
    if backend is not None:
        return acquire_with(url, backend=backend, **options)
    if "auth" not in options:
        # LSSのログインプロファイルでそのサービスへログイン済みなら、そのCookieで取得する。
        from lss.acquisition import login_profile

        options["auth"] = login_profile.auth_for(url, options.get("state_root"))
    _ensure_provider()
    outcomes = acquire_with(url, backend=default_backend(), **options)
    stale = [index for index, outcome in enumerate(outcomes)
             if outcome.failure is not None and outcome.failure.kind in STALE_TOOL_FAILURES]
    if stale:
        result = _ensure_provider(after_failure=True)
        if result is not None and result.get("status") == "installed":
            for index in stale:
                [outcomes[index]] = acquire_with(outcomes[index].requested_url, backend=default_backend(),
                                                 **options)[:1]
    return outcomes


def acquire_with(
    url: str, *, backend, auth: AuthMethod = AuthMethod(), force: bool = False,
    on_progress: Callable[[dict], None] | None = None, pacing: Pacing = Pacing(),
    sleep: Callable[[float], None] = time.sleep, library: Path | None = None,
    state_root: Path | None = None, should_stop: Callable[[], bool] | None = None,
) -> list[EntryOutcome]:
    """指定した取得手段で、URLの各項目を取得する。

    `should_stop` が真になると、進行中の取得を安全に止め、残りの項目は始めない。
    """
    stop_requested = should_stop or (lambda: False)
    try:
        store.cleanup_stale_work()
    except OSError:
        pass

    try:
        entries = _with_backoff(lambda: backend.resolve(url, auth), pacing, sleep)
    except AcquisitionFailure as failure:
        if failure.kind in {FailureKind.NETWORK, FailureKind.RATE_LIMITED} and not force:
            found = store.find_valid_by_url(url, state_root)
            if found is not None:
                return [EntryOutcome(url, store.reuse(*found), offline_reuse=True)]
        return [EntryOutcome(url, failure=failure)]

    outcomes: list[EntryOutcome] = []
    downloaded_before = False
    low, high = pacing.between_items
    for entry in entries:
        entry_url = entry.get("webpage_url") or entry.get("url") or url
        if stop_requested():
            outcomes.append(EntryOutcome(entry_url, failure=AcquisitionFailure(
                FailureKind.CANCELLED, "停止の指示により取得を始めませんでした")))
            continue
        if downloaded_before:
            sleep(random.uniform(low, high))

        def attempt(entry=entry):
            return _acquire_one(backend, entry, requested_url=url, auth=auth, force=force,
                                on_progress=on_progress, should_stop=stop_requested,
                                library=library, state_root=state_root)

        def congested(failure: AcquisitionFailure) -> None:
            nonlocal low, high
            if failure.kind is FailureKind.RATE_LIMITED:
                # 混雑を示されたら、以後の項目の間隔も広げたままにする。
                low, high = (min(max(value * 2, 1.0), pacing.backoff_limit) for value in (low, high))

        try:
            source = _with_backoff(attempt, pacing, sleep, on_retry=congested)
        except AcquisitionFailure as failure:
            congested(failure)
            outcomes.append(EntryOutcome(entry_url, failure=failure))
            downloaded_before = True
            continue
        outcomes.append(EntryOutcome(entry_url, source))
        downloaded_before = downloaded_before or not source.reused
    return outcomes


def _with_backoff(action, pacing: Pacing, sleep, on_retry=None):
    delay = pacing.backoff_start
    for attempt in range(1, max(1, pacing.attempts) + 1):
        try:
            return action()
        except AcquisitionFailure as failure:
            if not failure.retryable or failure.kind is FailureKind.BUSY or attempt >= pacing.attempts:
                raise
            if on_retry is not None:
                on_retry(failure)
            sleep(min(delay, pacing.backoff_limit))
            delay = min(delay * 2, pacing.backoff_limit)
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass
class Reacquisition:
    previous: dict
    outcome: EntryOutcome

    @property
    def identical(self) -> bool:
        """取り直した内容が、以前の取得物と同じバイト列か。違えば派生物の来歴は以前の版を指したまま。"""
        source = self.outcome.source
        return source is not None and source.sha256 == self.previous["content"]["sha256"]


def reacquire(sha256: str, *, auth: AuthMethod | None = None, state_root: Path | None = None,
              **options) -> Reacquisition:
    """記録された所在と認証の種類から、同じソースを取り直す（例: 原本を消した後の再処理）。

    有効な取得物が残っていればそれを使う。プロファイル名は記録しないため、
    既定以外のブラウザプロファイルが必要なら `auth` で渡す。
    """
    records = store.records_for_sha256(sha256, state_root)
    if not records:
        raise KeyError(f"取得の記録がありません: {sha256}")
    previous = max(records, key=lambda r: r.get("acquired_at") or "")
    plan = previous.get("reacquire") or {}
    url = plan.get("url")
    if not url:
        raise ValueError("再取得に必要な所在が記録されていません")
    if auth is None:
        recorded = previous.get("auth") or {}
        if recorded.get("method") == "browser_cookies":
            auth = AuthMethod.browser_cookies(recorded["browser"])
        elif recorded.get("method") == "cookies_file":
            from lss.acquisition import login_profile

            auth = login_profile.auth_for(url, state_root)
        elif recorded.get("method") == "app_login":
            from lss.acquisition import login_profile

            auth = AuthMethod.app_login(login_profile.cookie_profile(state_root))
        else:
            auth = AuthMethod()
    outcomes = acquire(url, auth=auth, state_root=state_root, **options)
    wanted = previous["logical_source"]
    match = next((o for o in outcomes if o.source is not None and o.source.logical_source == wanted), None)
    return Reacquisition(previous, match or outcomes[0])


def enqueue(queue, url: str, **options) -> UrlIntake:
    """取得できた素材を、ローカルのファイル追加と同じ経路で分離キューへ入れる。"""
    outcomes = acquire(url, **options)
    paths = [outcome.source.path for outcome in outcomes if outcome.ok]
    queue_ids = queue.add(paths) if paths else []
    return UrlIntake(outcomes, queue_ids)
