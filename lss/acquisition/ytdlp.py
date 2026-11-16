"""yt-dlpによる取得手段。製品の来歴の形はこのモジュールに依存しない。

- yt-dlpは別プロセスの実行物として呼ぶ。配備版は署名を検証して導入した公式の実行物、
  開発環境では導入がなければ仮想環境のモジュールを使う（`provider.py`）。
- 音声の最良表現を、変換・再多重化せずにそのまま取得する（`--fixup never`、後処理なし）。
- YouTubeはJavaScript実行環境が無いと低品質な形式しか得られないことがあるため、
  黙って質を落とさず、実行環境の不足として失敗させる。
- 取得物は論理ソースごとの作業領域へ落とし、中断した部分ファイルは次回に引き継ぐ。
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from lss.acquisition.store import AcquisitionFailure, AuthMethod, FailureKind, sanitize_text, sanitize_url
from lss.paths import resource_root

logger = logging.getLogger("lss.acquisition")

FORMAT_SELECTOR = "bestaudio/best"
_MAX_WARNINGS = 20
_PROGRESS = "[LSS-PROGRESS] "
_FILE = "[LSS-FILE] "
# 来歴に残すソースのメタデータ。署名付きURL・HTTPヘッダ・Cookieを含む項目は残さない。
_METADATA_FIELDS = (
    "title", "fulltitle", "alt_title", "uploader", "uploader_id", "channel", "channel_id",
    "creators", "artists", "track", "track_number", "album", "album_artists", "release_year",
    "release_date", "upload_date", "timestamp", "duration", "live_status", "availability",
    "playlist_id", "playlist_title", "playlist_index", "series", "season_number",
    "episode", "episode_number", "episode_id",
)
_FORMAT_FIELDS = (
    "format_id", "format", "format_note", "ext", "acodec", "vcodec", "abr", "tbr", "asr",
    "audio_channels", "container", "protocol", "filesize", "filesize_approx", "dynamic_range",
)
# JavaScript実行環境なしでは最良の形式を得られない抽出器。
_NEEDS_JS_RUNTIME = frozenset({"youtube", "youtubemusic"})


@dataclass
class Description:
    """1件のソースについて、取得前に分かる事実。`handle` は手段固有の情報。"""

    logical_source: str
    folder: str
    stem: str
    record: dict
    handle: dict = field(repr=False)


def classify(exc: BaseException | str) -> AcquisitionFailure:
    """yt-dlpの失敗（例外またはERROR行）を、取得契約の失敗種別へ写す。"""
    chain: list[BaseException] = []
    if isinstance(exc, BaseException):
        chain.append(exc)
        exc_info = getattr(exc, "exc_info", None)
        if exc_info and exc_info[1] is not None:
            chain.append(exc_info[1])
        while getattr(chain[-1], "cause", None) is not None and len(chain) < 6:
            chain.append(chain[-1].cause)
    message = str(exc)
    lowered = " ".join([message, *(str(item) for item in chain)]).lower()
    names = {type(item).__name__ for item in chain}

    def has(*texts: str) -> bool:
        return any(text in lowered for text in texts)

    if "CookieLoadError" in names or has("failed to load cookies", "cookies database", "could not copy",
                                         "dpapi", "failed to decrypt"):
        return AcquisitionFailure(FailureKind.AUTH_UNAVAILABLE, message)
    if "UnsupportedError" in names or has("unsupported url"):
        return AcquisitionFailure(FailureKind.UNSUPPORTED_URL, message)
    if "GeoRestrictedError" in names or has("available in your country", "geo restrict", "geo-restrict",
                                             "from your location", "your region"):
        return AcquisitionFailure(FailureKind.REGION_RESTRICTED, message)
    if names & {"ContentTooShortError", "IncompleteRead"} or has(
            "more expected", "incompleteread", "content too short", "bytes, expected"):
        return AcquisitionFailure(FailureKind.INTERRUPTED, message)
    if has("http error 429", "too many requests"):
        return AcquisitionFailure(FailureKind.RATE_LIMITED, message)
    if has("sign in", "login required", "log in", "requires authentication", "members-only",
           "members only", "join this channel", "private video", "--cookies", "premium",
           "age-restricted", "confirm your age"):
        return AcquisitionFailure(FailureKind.AUTH_REQUIRED, message)
    if has("video unavailable", "is unavailable", "has been removed", "does not exist",
           "no longer available", "not found", "http error 404", "drm protected", "is live",
           "live event", "premieres in", "no video formats", "requested format is not available"):
        return AcquisitionFailure(FailureKind.UNAVAILABLE, message)
    if "TransportError" in names or has("timed out", "timeout", "connection", "getaddrinfo",
                                        "name resolution", "unable to download webpage",
                                        "temporary failure", "network", "ssl"):
        return AcquisitionFailure(FailureKind.NETWORK, message)
    return AcquisitionFailure(FailureKind.EXTRACTOR_FAILED, message)


def _failure(exc: BaseException | str, auth: AuthMethod) -> AcquisitionFailure:
    failure = classify(exc)
    if failure.kind is FailureKind.AUTH_REQUIRED and auth.method != "none":
        # 認証情報を渡したうえでの拒否は、ログインのやり直しが必要という別の対処になる。
        return AcquisitionFailure(FailureKind.AUTH_EXPIRED, str(failure))
    return failure


def _location(url) -> str | None:
    return sanitize_url(url) if isinstance(url, str) and url else None


def _pick(info: dict, names) -> dict:
    return {name: info[name] for name in names if info.get(name) is not None}


_VERSION = re.compile(r"^\[debug\] yt-dlp version (?:(?P<channel>\w+)@)?(?P<version>\S+)"
                      r"(?: from (?P<origin>\S+))?(?: \[(?P<head>[0-9a-f]+)\])?(?: \((?P<variant>[^)]+)\))?")
_JS_RUNTIMES = re.compile(r"^\[debug\] JS runtimes: (?P<value>.+)$")
_LIBRARIES = re.compile(r"^\[debug\] Optional libraries: (?P<value>.+)$")


def parse_debug(stderr: str) -> dict:
    """`--verbose` の診断行から、実際に使われたyt-dlpと実行環境の版を得る。"""
    revision: dict = {"name": "yt-dlp", "version": None, "channel": None, "origin": None,
                      "git_head": None, "variant": None, "ejs_version": None, "js_runtime": None}
    for line in stderr.splitlines():
        if match := _VERSION.match(line):
            revision.update(version=match["version"], channel=match["channel"], origin=match["origin"],
                            git_head=match["head"], variant=match["variant"])
        elif match := _JS_RUNTIMES.match(line):
            for item in match["value"].split(","):
                name, _, version = item.strip().partition("-")
                if name and name != "none" and version:
                    revision["js_runtime"] = {"name": name, "version": version}
                    break
        elif match := _LIBRARIES.match(line):
            for item in match["value"].split(","):
                name, _, version = item.strip().rpartition("-")
                if name in {"yt_dlp_ejs", "yt-dlp-ejs"}:
                    revision["ejs_version"] = version
    return revision


@dataclass
class _Run:
    returncode: int
    stdout: list[str]
    stderr: str
    cancelled: bool = False

    def errors(self) -> str:
        # 進捗行の復帰文字でERROR行の本文が次の行へ分かれることがあるため、最初のERROR以降を読む。
        lines = self.stderr.splitlines()
        start = next((i for i, line in enumerate(lines) if line.startswith("ERROR:")), None)
        found = [] if start is None else [
            line.strip() for line in lines[start:]
            if line.strip() and not line.startswith(("[debug]", " ", "Traceback")) and line.strip() != "ERROR:"]
        return "\n".join(found) or self.stderr.strip()[-2000:] or f"yt-dlpが終了コード{self.returncode}で失敗しました"

    def warnings(self) -> list[str]:
        found = [sanitize_text(line) for line in self.stderr.splitlines() if line.startswith("WARNING:")]
        return found[:_MAX_WARNINGS]


class YtDlpBackend:
    name = "yt-dlp"

    def __init__(self, command: Sequence[str] | None = None, *, provider: dict | None = None,
                 extra_args: Sequence[str] = ()) -> None:
        """`command` はyt-dlpの起動列。省略時は導入済みの検証済み実行物（なければ開発用モジュール）。"""
        self._command = tuple(command) if command is not None else None
        self._cookie_copy: Path | None = None
        self._provider = provider
        self._extra = tuple(extra_args)

    # --- 起動 ---------------------------------------------------------------

    def _resolved(self) -> tuple[tuple[str, ...], dict]:
        if self._command is not None:
            return self._command, dict(self._provider or {"kind": "explicit"})
        from lss.acquisition import provider

        current = provider.active()
        return current.command, current.identity()

    def _args(self, auth: AuthMethod) -> list[str]:
        args = ["--ignore-config", "--no-cache-dir", "--encoding", "utf-8", "--color", "no_color",
                "--no-update", "--socket-timeout", "30", "--retries", "5", "--fragment-retries", "5",
                "--extractor-retries", "3"]
        ffmpeg = resource_root() / "bin" / "ffmpeg.exe"
        if ffmpeg.is_file():
            args += ["--ffmpeg-location", str(ffmpeg)]
        try:
            from lss.acquisition import js_runtime

            runtime = js_runtime.active()
        except Exception:
            runtime = None
        if runtime is not None and runtime.kind == "managed":
            # アプリが検証して導入したdenoを使う（PATHの状態に左右されない）。
            args += ["--js-runtimes", f"deno:{runtime.path}"]
        if auth.method == "cookies_file" and self._cookie_copy is not None:
            # 利用者のファイルはyt-dlpが書き戻すので、取得ごとの複製を渡す（元のファイルを変えない）。
            args += ["--cookies", str(self._cookie_copy)]
        if auth.method in {"browser_cookies", "app_login"}:
            args += ["--cookies-from-browser",
                     auth.browser if auth.profile is None else f"{auth.browser}:{auth.profile}"]
        return args + list(self._extra)

    def _run(self, args: list[str], auth: AuthMethod, *,
             on_line: Callable[[str], None] | None = None,
             should_stop: Callable[[], bool] | None = None) -> tuple[_Run, dict]:
        command, identity = self._resolved()
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        holder = None
        if auth.method == "cookies_file":
            import shutil
            import tempfile

            holder = Path(tempfile.mkdtemp(prefix="lss-cookies-"))
            try:
                self._cookie_copy = holder / "cookies.txt"
                shutil.copyfile(auth.profile, self._cookie_copy)
            except OSError as exc:
                shutil.rmtree(holder, ignore_errors=True)
                self._cookie_copy = None
                raise AcquisitionFailure(FailureKind.AUTH_UNAVAILABLE, "指定したCookieファイルを読めません") from exc
        try:
            return self._run_process(command, identity, environment, args, auth, on_line, should_stop)
        finally:
            if holder is not None:
                import shutil

                shutil.rmtree(holder, ignore_errors=True)
                self._cookie_copy = None

    def _run_process(self, command, identity, environment, args, auth, on_line, should_stop):
        try:
            process = subprocess.Popen(
                [*command, *self._args(auth), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            raise AcquisitionFailure(FailureKind.TOOL_UNAVAILABLE, f"yt-dlpを起動できません: {exc}") from exc

        lines: queue.Queue = queue.Queue()
        stderr_chunks: list[bytes] = []

        def read_stdout() -> None:
            for raw in process.stdout:
                lines.put(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
            lines.put(None)

        def read_stderr() -> None:
            stderr_chunks.append(process.stderr.read())

        readers = [threading.Thread(target=read_stdout, daemon=True),
                   threading.Thread(target=read_stderr, daemon=True)]
        for reader in readers:
            reader.start()
        stdout: list[str] = []
        cancelled = False
        while True:
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                if should_stop is not None and should_stop() and process.poll() is None:
                    cancelled = True
                    # 部分ファイルは残る。次回は続きから取得する。
                    process.terminate()
                continue
            if line is None:
                break
            if on_line is not None and (line.startswith(_PROGRESS) or line.startswith(_FILE)):
                on_line(line)
                if should_stop is not None and should_stop() and process.poll() is None:
                    cancelled = True
                    process.terminate()
                if line.startswith(_PROGRESS):
                    continue
            stdout.append(line)
        process.wait()
        for reader in readers:
            reader.join(timeout=5)
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        for warning in _Run(0, [], stderr).warnings():
            logger.warning("yt-dlp: %s", warning)
        return _Run(process.returncode, stdout, stderr, cancelled), identity

    def _json(self, args: list[str], auth: AuthMethod) -> tuple[dict, dict, list[str]]:
        run, identity = self._run(["--verbose", "-J", "-f", FORMAT_SELECTOR, *args], auth)
        if run.returncode:
            raise _failure(run.errors(), auth)
        try:
            info = json.loads("\n".join(run.stdout))
        except ValueError as exc:
            raise AcquisitionFailure(FailureKind.EXTRACTOR_FAILED, "yt-dlpの出力を読めません") from exc
        tool = {**parse_debug(run.stderr), "provider": identity}
        return info, tool, run.warnings()

    # --- 取得の手順 ----------------------------------------------------------

    def resolve(self, url: str, auth: AuthMethod) -> list[dict]:
        """URLを1件以上の取得対象へ展開する。再生リストは各項目の所在だけを得る。

        再生リスト内の動画URLは、その動画だけを指す。
        """
        info, tool, warnings = self._json(["--flat-playlist", "--no-playlist", url], auth)
        if info.get("_type") in {"playlist", "multi_video"}:
            entries = [entry for entry in (info.get("entries") or []) if entry]
            if not entries:
                raise AcquisitionFailure(FailureKind.UNAVAILABLE, "再生リストに取得できる項目がありません")
            return [{**entry, "_lss_playlist": _pick(info, ("id", "title", "extractor_key"))}
                    for entry in entries]
        return [{**info, "_lss_tool": tool, "_lss_warnings": warnings}]

    def describe(self, entry: dict, auth: AuthMethod) -> Description:
        if "_lss_tool" in entry:
            info, tool, warnings = entry, entry["_lss_tool"], entry["_lss_warnings"]
        else:
            target = entry.get("url") or entry.get("webpage_url")
            if not target:
                raise AcquisitionFailure(FailureKind.EXTRACTOR_FAILED, "取得対象の所在が得られません")
            info, tool, warnings = self._json(["--no-playlist", target], auth)
        info = {key: value for key, value in info.items() if not key.startswith("_lss_")}
        if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming", "post_live"}:
            raise AcquisitionFailure(
                FailureKind.UNAVAILABLE, "配信中・配信予定・配信直後の素材は完成した音源として取得しません")

        extractor_key = str(info.get("extractor_key") or info.get("extractor") or "generic")
        source_id = str(info.get("id") or "")
        if not source_id:
            raise AcquisitionFailure(FailureKind.EXTRACTOR_FAILED, "ソースの識別子が得られません")
        if extractor_key.lower() in _NEEDS_JS_RUNTIME and tool.get("js_runtime") is None:
            raise AcquisitionFailure(
                FailureKind.TOOL_UNAVAILABLE,
                "YouTubeの最良の形式を得るJavaScript実行環境（deno）が見つかりません")

        requested = info.get("requested_formats") or [info]
        formats = [_pick(item, _FORMAT_FIELDS) for item in requested]
        playlist = entry.get("_lss_playlist")
        metadata = _pick(info, _METADATA_FIELDS)
        if playlist:
            metadata.setdefault("playlist_id", playlist.get("id"))
            metadata.setdefault("playlist_title", playlist.get("title"))
        record = {
            "logical_source": f"{extractor_key.lower()}:{source_id}",
            "authority": {"extractor_key": extractor_key, "extractor": info.get("extractor"),
                          "id": source_id},
            "location": {"webpage_url": _location(info.get("webpage_url")),
                         "original_url": _location(info.get("original_url"))},
            "auth": auth.provenance(),
            "tool": tool,
            "format": {"selector": FORMAT_SELECTOR, "selected": formats,
                       "transcoded": False, "remuxed": False},
            "metadata": metadata,
            "reacquire": {"url": _location(info.get("webpage_url") or info.get("original_url")),
                          "format_id": info.get("format_id"), "auth_method": auth.method},
            "diagnostics": {"warnings": warnings},
        }
        title = info.get("title") or info.get("track") or source_id
        return Description(record["logical_source"], extractor_key, f"{title} [{source_id}]",
                           record, {"info": info})

    def download(self, description: Description, work: Path, auth: AuthMethod,
                 on_progress: Callable[[dict], None] | None = None, *,
                 should_stop: Callable[[], bool] | None = None) -> Path:
        work.mkdir(parents=True, exist_ok=True)
        # 形式のURLや（認証時は）Cookieを含むため、取得の直後に消す。
        info_path = work / "info.json"
        info_path.write_text(json.dumps(description.handle["info"], ensure_ascii=False), encoding="utf-8")
        found: list[Path] = []

        def on_line(line: str) -> None:
            if line.startswith(_FILE):
                found.append(Path(line[len(_FILE):].strip()))
                return
            if on_progress is None:
                return
            status, downloaded, total, estimate = (line[len(_PROGRESS):].split("|") + ["", "", "", ""])[:4]

            def number(text: str):
                try:
                    return int(float(text))
                except ValueError:
                    return None

            try:
                on_progress({"stage": "download", "status": status, "downloaded_bytes": number(downloaded),
                             "total_bytes": number(total) or number(estimate)})
            except Exception:
                pass

        try:
            run, _identity = self._run([
                "--load-info-json", str(info_path), "-f", FORMAT_SELECTOR,
                "-o", "source.%(ext)s", "-P", f"home:{work}", "-P", f"temp:{work}",
                "--fixup", "never", "--continue", "--no-overwrites", "--no-mtime",
                "--newline", "--progress", "--no-simulate",
                "--progress-template", "download:" + _PROGRESS
                + "%(progress.status)s|%(progress.downloaded_bytes)s|%(progress.total_bytes)s"
                  "|%(progress.total_bytes_estimate)s",
                "--print", "after_move:" + _FILE + "%(filepath)s",
            ], auth, on_line=on_line, should_stop=should_stop)
        finally:
            info_path.unlink(missing_ok=True)
        if run.cancelled:
            raise AcquisitionFailure(FailureKind.CANCELLED, "停止の指示により取得を中断しました")
        if run.returncode:
            raise _failure(run.errors(), auth)
        description.record.setdefault("diagnostics", {}).setdefault("warnings", []).extend(run.warnings())
        if len(found) != 1 or not found[0].is_file():
            raise AcquisitionFailure(FailureKind.INVALID_MEDIA, "取得物のファイルを1つに特定できません")
        return found[0]
