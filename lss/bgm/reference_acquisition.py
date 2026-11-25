"""参照録音の機会の発見・取得・確認を、LSSの取得の層（`lss.acquisition`）で行う。

契約は `docs/ACQUISITION.md`「Referenceの取得」。利用者が既発売の音源を集めて置く手間を、通常の使い方から外す。

既発売の劇伴は「作品名＋サウンドトラック」という題のアルバムだけにあるとは限らない。主題歌のシングルに
劇伴が入っている（題に作品名もサウンドトラックの語も無い）、前の作品・映画のサウンドトラックに入っている、
フルのサウンドトラックより先に数曲だけ出ている、などがある。そこで発見は**曲を単位**にする。

    作品 → 目録の候補（曲の検索・アルバムの検索） → 曲の音声（取得） → この番組の音声と照合
         → 確かめた曲の収録盤（曲目に同じ曲の識別子がある盤） → 盤の残りの曲 → 作品の参照

1. 作品の手掛かり：番組の取得の台帳（作品・再生リストの名前）か、ファイル名から。
2. 候補：YouTube Music の曲の検索（作品名）とアルバムの検索（作品名＋サウンドトラックの語）。検索の順位と
   題は確かめる順番にだけ使い、同一性の根拠にしない。
3. 確認：候補の曲を取得し、この番組の音声と照合する。波形まで一致（known_exact）か、指紋の票が偶然では
   出ない水準で一つのずれに揃う（台詞が重なって波形の一致が切れても、同じ録音の素材が鳴っている）曲だけを
   確かめた曲とする。確かめた曲だけが作品に結び付く。この結び付きは参照の候補を増やすだけで、何かを
   既発売として消す根拠ではない（それは区間ごとの関係判定 `reference_relations` が音楽の会計で決める）。
4. 収録盤：確かめた曲の取得の台帳にある収録盤の名前で盤を探し、曲目に**同じ曲の識別子**がある盤だけを
   収録盤とする（名前の似た盤は使わない）。収録盤の残りの曲を取得して作品へ結び付ける。
5. 記憶：検索・曲目は状態領域に期限つきで残し（同じ問い合わせを繰り返さない）、確かめた曲・確かめられ
   なかった曲（指紋は残る）・収録盤を作品ごとの記録に残す。次の話では、取得済みの指紋を通信なしで先に
   照合する。

通信は取得先ごとに一つずつ、揺らぎのある間隔を空け、混雑の合図で休む（`lss.acquisition.courtesy`）。
休みの間は待ち続けず保留し、次の機会に続きから行う。一回の機会で確認のために取得する曲数には上限がある。
ネットワーク・取得ツール・検索結果が無い時は、何もせずに終わる。単話のBGMの処理を止めない。
取得の認証は通常のURLの取得と同じ：LSSのログインプロファイルでそのサービスへログイン済みならそのCookieを使い
（会員向けの盤も取れる）、そうでなければ認証なし。ブラウザのCookieは集めない。資格情報・期限つきのURLは
記録しない（取得の層の来歴の規則のまま）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import unicodedata
from urllib.parse import quote

import numpy as np

from lss.bgm import identity, reference_store

SCHEMA = 2
OPPORTUNITY_TYPE = "lss_reference_opportunity"
CATALOG_TYPE = "lss_reference_catalog_answer"
DISCOVERY_ENV = "LSS_REFERENCE_DISCOVERY"
METHOD = "track-aware-release-discovery-audio-verified-v2"
QUERY_SUFFIXES = ("オリジナル・サウンドトラック", "サウンドトラック", "OST")
SEARCH_LIMIT = 20          # 検索の結果は上位だけを読む（ページを辿り尽くさない）
SONG_CANDIDATES = 10       # 曲の検索の候補のうち、音声で確かめる上位の数
MAX_ALBUMS = 4             # アルバムの検索の候補のうち、見本で確かめる数
SAMPLE_TRACKS = 6
MAX_TRACKS = 150
VERIFY_BUDGET = 30         # 一回の機会で、確認のために取得する曲の上限（収録盤の残りの取得は別）
RELEASE_LOOKUPS = 3        # 確かめた曲の収録盤を探す時に曲目を読む盤の数
MAX_RELEASES = 4           # 一回の機会で残りの曲を取得する収録盤の数
STRONG_SCORE = 3.0         # 作品名（2）とサウンドトラックの語（1）の両方が題に合う候補
RETRY_AFTER_SECONDS = 7 * 24 * 60 * 60
DETAIL_LIMIT = 200         # 失敗の詳細として記録する長さ（資格情報は元から含まれない）
# ログインの無い状態で、同じ盤の曲が認証（有料会員・会員限定）を求めてこれだけ続き、一曲も取れていなければ、
# 残りの曲を取りに行かない（取れない要求を繰り返さない）。LSSでログインした後の機会には、やり直す。
RESTRICTED_ALBUM_FAILURES = 2
CATALOG_TTL_SECONDS = 7 * 24 * 60 * 60
# 作品の記事（`lss.metadata.sources.wikipedia`）に載る主題歌（OP/ED）も曲の検索で探す。主題歌は作品名の検索や
# サウンドトラックの盤に出ないことが多く、毎話同じ位置で流れて隣の劇伴と一続きになる（照合で外せないと保留になる）。
THEME_KINDS = ("オープニングテーマ", "エンディングテーマ", "主題歌")
THEME_SONGS = 6            # 記事の主題歌のうち、新しい方（記事の後ろ）から探す数
THEME_CANDIDATES = 3       # 主題歌ごとに音声で確かめる検索の上位の数

# 指紋だけの確認（台詞が重なり波形の一致が切れる区間）。一つのずれに揃う票・窓・揺れで、偶然の一致と分ける。
# 値は `.state/validation/reference-intelligence-r2/` の測定（正例と、別の作品の参照の対照）で決めた。
FINGERPRINT_MIN_VOTES = 40
FINGERPRINT_MIN_WINDOWS = 3
FINGERPRINT_MAX_DRIFT_SECONDS = 0.1

# 指紋の票が揃った範囲の、混合での波形の説明（2秒ブロック）。台詞・番組で加えた音が重なる同じ録音
# （corpus-d 第1・4話の挿入歌：上位3ブロック0.08〜0.12、偶然の水準の12〜60倍）と、票は揃うが別の録音
# （合成の規則的な譜面・corpus-c 第10話の予告の区間：偶然と同程度）を分ける。
WAVEFORM_MIN_BLOCK = 0.05
WAVEFORM_MIN_BLOCKS = 3
WAVEFORM_MIN_TOP = 0.05
WAVEFORM_NULL_RATIO = 8.0

TIER_EXACT = "audio_waveform_exact"
TIER_WAVEFORM = "audio_waveform_supported"
PENDING_WAVEFORM = "fingerprint_pending_waveform"

_EPISODE = re.compile(r"(最終[話回]|第\s*[0-9０-９一二三四五六七八九十百千壱弐参肆伍拾]+\s*[話回]|#\s*\d+|episode\s*\d+|ep\.?\s*\d+)",
                      re.IGNORECASE)
_QUOTES = re.compile(r"[「『【\[]([^」』】\]]+)[」』】\]]")
_NOISE = re.compile(r"(公式|見逃し配信|期間限定|一挙配信|配信|official|full|tver|ニコニコ)", re.IGNORECASE)
_SOUNDTRACK_WORDS = ("サウンドトラック", "soundtrack", "ost", "劇伴", "bgm", "music collection", "音楽集")


def enabled() -> bool:
    """配備版では既定で有効、開発ツリーでは明示した時だけ（試験・開発がネットワークへ勝手に出ない）。"""
    value = os.environ.get(DISCOVERY_ENV)
    if value is not None:
        return value.strip().lower() not in ("0", "off", "false", "no", "")
    from lss.paths import is_frozen

    return is_frozen()


def _normal(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"[\s　・･\-‐–—_~～〜!！?？.,、。:：;；'\"“”‘’()（）\[\]【】「」『』]+", "", text)


def work_title(text: str) -> str | None:
    """番組の題から作品名の手掛かりを取る（例：『作品名』第10話… → 作品名）。"""
    text = unicodedata.normalize("NFKC", str(text or ""))
    episode = _EPISODE.search(text)
    head = text[:episode.start()] if episode else text
    for candidate in _QUOTES.findall(head):
        candidate = candidate.strip()
        if candidate and not _NOISE.fullmatch(candidate.replace(" ", "")) and not _EPISODE.search(candidate):
            return candidate
    for candidate in _QUOTES.findall(text):
        candidate = candidate.strip()
        if candidate and not _NOISE.search(candidate) and not _EPISODE.search(candidate):
            return candidate
    head = _QUOTES.sub(" ", head)
    head = _NOISE.sub(" ", head).strip(" -_|")
    return head or None


def theme_songs(source: Path, state_root: Path | None) -> list[dict]:
    """番組の作品の記事に載る主題歌（OP/ED）。新しい方から `THEME_SONGS` 曲。``{"title", "kind", "query"}``。

    記事は作品の情報の取得と同じ（状態領域のcacheを使う）。読めない・載っていなければ空（探さないだけ）。
    検索の語は曲名と歌い手。どの曲が番組で鳴るかは名前ではなく、取得した音声の照合だけが決める。"""
    try:
        from lss.metadata import series_album
        from lss.metadata.sources import wikipedia

        episode = series_album._episode(Path(source), {})
        if episode is None or state_root is None:
            return []
        series = wikipedia.series(episode["work"], episode["subtitle"], episode["kind"], Path(state_root),
                                  episode_number=episode["order"])
    except Exception:  # noqa: BLE001 - 作品の情報が無いことは参照の発見を止めない
        return []
    if series is None:
        return []
    by_kind: dict[str, list[dict]] = {}
    for song in reversed(series.songs):   # 記事は放送順：後ろほど新しい
        kind = next((k for k in THEME_KINDS if k in (song.get("kind") or "")), None)
        if kind is None:
            continue
        singers = [c.name for c in song.get("credits") or () if getattr(c, "role_key", None) == "vocals"]
        query = " ".join([song["title"], *singers[:1]]).strip()
        if query:
            by_kind.setdefault(kind, []).append({"title": song["title"], "kind": song["kind"], "query": query})
    found: list[dict] = []
    # OPとEDを交互に、それぞれ新しい方から（EDの数が多くても今のOPを落とさない）。
    for rank in range(max((len(v) for v in by_kind.values()), default=0)):
        for kind in THEME_KINDS:
            songs = by_kind.get(kind) or []
            if rank < len(songs) and all(item["query"] != songs[rank]["query"] for item in found):
                found.append(songs[rank])
    return found[:THEME_SONGS]


def work_key(title: str) -> str:
    return "work-" + hashlib.sha256(_normal(title).encode("utf-8")).hexdigest()[:20]


def work_hint(source: Path, source_sha256: str | None, state_root: Path | None = None) -> dict | None:
    """番組の作品の手掛かり。取得の台帳（作品・再生リストの名前・題）を先に、無ければファイル名から。"""
    from lss.acquisition import store

    candidates = []
    if source_sha256:
        try:
            records = store.records_for_sha256(source_sha256, state_root)
        except (OSError, ValueError):
            records = []
        for record in sorted(records, key=lambda r: r.get("acquired_at") or "", reverse=True):
            metadata = record.get("metadata") or {}
            for name, basis in (("series", "acquisition_series"), ("playlist_title", "acquisition_playlist"),
                                ("title", "acquisition_title")):
                value = metadata.get(name)
                if isinstance(value, str) and value.strip():
                    title = value.strip() if name == "series" else work_title(value)
                    if title:
                        candidates.append((title, basis))
    title = work_title(Path(source).stem)
    if title:
        candidates.append((title, "file_name"))
    if not candidates:
        return None
    title, basis = candidates[0]
    return {"title": title, "work_key": work_key(title), "basis": basis}


# --- 機会の記録 ------------------------------------------------------------------

def _opportunity_path(key: str, state_root: Path | None) -> Path:
    return reference_store.store_root(state_root) / "opportunities" / f"{key}.json"


def load_opportunity(key: str, state_root: Path | None = None) -> dict | None:
    try:
        payload = json.loads(_opportunity_path(key, state_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) and payload.get("type") == OPPORTUNITY_TYPE else None


def _save_opportunity(payload: dict, state_root: Path | None) -> None:
    reference_store._write(_opportunity_path(payload["work_key"], state_root), payload)


# --- 取得 --------------------------------------------------------------------------

def reference_pacing():
    """参照の取得の再試行：一時的な失敗だけを、間を空けて一度だけやり直す（混雑の合図は礼儀の休みへ回す）。"""
    from lss.acquisition import service

    return service.Pacing(between_items=(3.0, 8.0), backoff_start=30.0, backoff_limit=300.0, attempts=2)


def acquire_into_store(url: str, *, cache_dir: Path, state_root: Path | None = None, work_keys=(),
                       discovery: dict | None = None, backend=None, **options) -> dict:
    """URLの各項目を参照音源として取得し、参照の知識へ入れる。失敗した項目は返すだけ（他を止めない）。

    認証は通常のURLの取得と同じ（LSSのログインプロファイルでログイン済みならそのCookie、無ければ認証なし）。
    """
    from lss.acquisition import service
    from lss.acquisition.store import AuthMethod

    library = reference_store.media_root(state_root)
    if backend is not None:
        options.setdefault("auth", AuthMethod())
        outcomes = service.acquire_with(url, backend=backend, library=library, state_root=state_root, **options)
    else:
        options.setdefault("pacing", reference_pacing())
        outcomes = service.acquire(url, library=library, state_root=state_root, **options)
    ingested, failed, reused = [], [], []
    for outcome in outcomes:
        if not outcome.ok:
            failed.append({"url": outcome.requested_url,
                           "failure": getattr(getattr(outcome.failure, "kind", None), "value", None),
                           "detail": str(outcome.failure or "")[:300]})
            continue
        try:
            record = reference_store.ingest(outcome.source.path, cache_dir=cache_dir, state_root=state_root,
                                            origin=reference_store.ORIGIN_ACQUIRED, work_keys=work_keys,
                                            acquisition=outcome.source.record, discovery=discovery)
        except (OSError, ValueError, identity.DecodeError) as exc:
            failed.append({"url": outcome.requested_url, "failure": f"ingest_{type(exc).__name__}"})
            continue
        ingested.append(record["sha256"])
        if outcome.source.reused:
            reused.append(record["sha256"])
    return {"ingested": ingested, "failed": failed, "reused": reused}


class SearchBackend:
    """yt-dlpで目録を読む（曲の検索・アルバムの検索・アルバムの曲目。取得はしない）。試験では差し替える。

    検索は上位 `SEARCH_LIMIT` 件だけを読む（結果のページを辿り尽くさない）。
    """

    def __init__(self, backend=None):
        from lss.acquisition import service

        self.backend = backend or service.default_backend()

    def _flat(self, url: str, limit: int | None = None) -> dict:
        from lss.acquisition.store import AuthMethod

        args = ["--flat-playlist"]
        if limit:
            args += ["--playlist-end", str(int(limit))]
        info, _tool, _warnings = self.backend._json([*args, url], AuthMethod())
        return info

    def albums(self, query: str) -> list[dict]:
        info = self._flat(f"https://music.youtube.com/search?q={quote(query)}#albums", SEARCH_LIMIT)
        return [{"url": e.get("url") or e.get("webpage_url"), "id": e.get("id")}
                for e in (info.get("entries") or []) if e and (e.get("url") or e.get("webpage_url"))]

    def songs(self, query: str) -> list[dict]:
        info = self._flat(f"https://music.youtube.com/search?q={quote(query)}#songs", SEARCH_LIMIT)
        return [{"url": e.get("url") or e.get("webpage_url"), "id": e.get("id"), "title": e.get("title")}
                for e in (info.get("entries") or []) if e and (e.get("url") or e.get("webpage_url"))]

    def tracks(self, album_url: str) -> dict:
        info = self._flat(album_url)
        entries = [e for e in (info.get("entries") or []) if e]
        return {"title": info.get("title"), "id": info.get("id"), "url": info.get("webpage_url") or album_url,
                "tracks": [{"url": e.get("url") or e.get("webpage_url"), "id": e.get("id"), "title": e.get("title"),
                            "duration": e.get("duration"), "channel": e.get("channel") or e.get("uploader")}
                           for e in entries if (e.get("url") or e.get("webpage_url"))]}


def _signal(failure_kind) -> str | None:
    from lss.acquisition import courtesy
    from lss.acquisition.store import FailureKind

    kind = getattr(failure_kind, "value", failure_kind)
    if kind == FailureKind.RATE_LIMITED.value:
        return courtesy.RATE_LIMITED
    if kind in (FailureKind.NETWORK.value, FailureKind.INTERRUPTED.value):
        return courtesy.TRANSIENT
    return None


class Catalog:
    """目録の問い合わせ。答えは状態領域へ期限つきで残し、期限内は通信しない。通信は礼儀の枠の中で行う。"""

    def __init__(self, search, *, state_root: Path | None, courtesy, now: float, should_stop=lambda: False,
                 ttl: float = CATALOG_TTL_SECONDS, recover=None):
        self.search, self.state_root, self.courtesy = search, state_root, courtesy
        self.now, self.should_stop, self.ttl = now, should_stop, ttl
        # 取得ツールが無い・古い時の回復（検証付きの導入）。成功の見込みがあれば真を返し、問い合わせを1回だけやり直す。
        self.recover = recover
        self.requests = 0
        self.cached = 0

    def _path(self, kind: str, key: str) -> Path:
        digest = hashlib.sha256(f"{kind}\n{key}".encode("utf-8")).hexdigest()[:24]
        return reference_store.store_root(self.state_root) / "catalog" / f"{kind}-{digest}.json"

    def _ask(self, kind: str, key: str, url: str, fetch):
        from lss.acquisition.store import AcquisitionFailure

        path = self._path(kind, key)
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved.get("type") == CATALOG_TYPE and saved.get("kind") == kind and saved.get("key") == key
                    and self.now - float(saved.get("fetched_at_epoch") or 0) < self.ttl):
                self.cached += 1
                return saved["value"]
        except (OSError, ValueError, AttributeError, TypeError):
            pass
        with self.courtesy.slot(url, purpose=f"catalog_{kind}", should_stop=self.should_stop):
            try:
                try:
                    value = fetch()
                except AcquisitionFailure as failure:
                    if self.recover is None or not self.recover(failure):
                        raise
                    value = fetch()
            except AcquisitionFailure as failure:
                self.courtesy.report(url, _signal(failure.kind))
                raise
        self.courtesy.report(url)
        self.requests += 1
        reference_store._write(path, {"type": CATALOG_TYPE, "kind": kind, "key": key, "value": value,
                                      "fetched_at": datetime.now(timezone.utc).isoformat(),
                                      "fetched_at_epoch": self.now})
        return value

    def songs(self, query: str) -> list[dict]:
        if not hasattr(self.search, "songs"):
            return []
        url = f"https://music.youtube.com/search?q={quote(query)}#songs"
        return self._ask("songs", query, url, lambda: self.search.songs(query))

    def albums(self, query: str) -> list[dict]:
        url = f"https://music.youtube.com/search?q={quote(query)}#albums"
        return self._ask("albums", query, url, lambda: self.search.albums(query))

    def tracks(self, album_url: str) -> dict:
        return self._ask("tracks", album_url, album_url, lambda: self.search.tracks(album_url))


def _score(album_title: str | None, title: str) -> float:
    album, work = _normal(album_title), _normal(title)
    score = 0.0
    if work and work in album:
        score += 2.0
    if any(word in album for word in (_normal(w) for w in _SOUNDTRACK_WORDS)):
        score += 1.0
    return score


def _sample(tracks: list[dict], count: int) -> list[dict]:
    if len(tracks) <= count:
        return list(tracks)
    step = (len(tracks) - 1) / (count - 1)
    return [tracks[round(i * step)] for i in range(count)]


def fingerprint_consistent(match) -> bool:
    """指紋の票が一つのずれに揃い、偶然の一致の水準を超える（台詞で波形の一致が切れた同じ録音の候補）。"""
    drift = match.max_window_drift_seconds
    return (match.votes >= FINGERPRINT_MIN_VOTES and match.measured_windows >= FINGERPRINT_MIN_WINDOWS
            and drift is not None and drift <= FINGERPRINT_MAX_DRIFT_SECONDS)


def waveform_support(query_samples, reference_samples, match, rng=None) -> dict:
    """指紋の候補の範囲で、参照の波形が混合をどれだけ直接説明するか（2秒ブロック）と、偶然の位置の水準。

    同じ録音なら台詞・効果音・番組で加えた音が重なっていても、いくつかのブロックで偶然の数十倍を説明する。
    別の録音（同じ作曲者・同じ拍の別の曲・合成の規則的な譜面）は偶然の水準に留まる。
    """
    rng = rng or np.random.default_rng(0)
    rate = identity.SAMPLE_RATE
    start = int(round(match.query_start_seconds * rate))
    end = int(round(match.query_end_seconds * rate))
    offset = int(round(match.offset_seconds * rate))
    _, values, _ = identity.waveform_blocks(query_samples, reference_samples, start, end, offset)
    nulls = []
    for _ in range(12):
        candidate = int(rng.integers(-start, max(1, reference_samples.size - end)))
        if abs(candidate - offset) < 5 * rate:
            continue
        _, null_values, _ = identity.waveform_blocks(query_samples, reference_samples, start, end, candidate,
                                                     fine=False)
        nulls.extend(null_values.tolist())
        if len(nulls) >= 24:
            break
    ordered = np.sort(values)[::-1]
    top = float(np.mean(ordered[:3])) if ordered.size else 0.0
    null_max = float(max(nulls)) if nulls else 0.0
    above = int((values >= WAVEFORM_MIN_BLOCK).sum())
    supported = (ordered.size >= 3 and above >= WAVEFORM_MIN_BLOCKS and top >= WAVEFORM_MIN_TOP
                 and top >= WAVEFORM_NULL_RATIO * null_max)
    return {"blocks": int(ordered.size), "blocks_above": above, "top3": round(top, 4),
            "null_max": round(null_max, 4), "supported": bool(supported)}


def verify(query_samples, records: list[dict], *, cache_dir: Path, state_root: Path | None,
           network: bool = True, acquire_options: dict | None = None) -> dict[str, dict]:
    """参照の知識の記録のうち、この番組の音声で同じ録音が鳴っていると確かめられるもの。

    返り値は内容SHA-256 → 確認の段（`TIER_EXACT`：波形まで一致の連続／`TIER_WAVEFORM`：指紋の票が一つの
    ずれに揃い、その範囲の波形を偶然の水準を大きく超えて説明する）・秒・票。指紋の票だけでは確かめない
    （規則的な拍・和声の別の曲でも票は揃い得る）。
    `network=False` では退役した重い音源を取り直さない。その時は指紋で候補を絞り（`pending_waveform`）、
    波形の確認が要る参照だけを返す（呼び出し側がその曲だけを取り直して確かめる）。
    """
    if not records:
        return {}
    resolver = reference_store.MediaResolver(state_root=state_root, reacquire=network, strict=False,
                                             acquire_options=dict(acquire_options or {}))
    library, _ = reference_store.build_library(records, cache_dir=cache_dir, state_root=state_root,
                                               resolver=resolver)
    found: dict[str, dict] = {}
    owners = {reference.sha256: number for number, reference in enumerate(library.index.references)}
    for match in identity.match(query_samples, library.index, collapse=False):
        sha = match.reference_sha256
        if match.identity_state == "known_exact":
            tier, support = TIER_EXACT, None
        elif fingerprint_consistent(match):
            if sha in resolver.unavailable:
                # 重い音源が手元に無い：指紋の候補として返し、波形はこの曲だけ取り直して確かめる。
                row = found.setdefault(sha, {"tier": PENDING_WAVEFORM, "seconds": 0.0, "votes": 0, "spans": []})
                row["votes"] += int(match.votes)
                continue
            support = waveform_support(query_samples, library.index.samples(owners[sha]), match)
            if not support["supported"]:
                continue
            tier = TIER_WAVEFORM
        else:
            continue
        row = found.setdefault(sha, {"tier": tier, "seconds": 0.0, "votes": 0, "spans": []})
        if row["tier"] == PENDING_WAVEFORM or tier == TIER_EXACT:
            row["tier"] = tier
        row["seconds"] = round(row["seconds"] + match.query_end_seconds - match.query_start_seconds, 1)
        row["votes"] += int(match.votes)
        if len(row["spans"]) < 8:
            row["spans"].append([match.query_start_seconds, match.query_end_seconds, match.offset_seconds, tier]
                                + ([support["top3"], support["null_max"]] if support else []))
    return found


# --- 発見 --------------------------------------------------------------------------

def _new_opportunity(hint: dict, previous: dict | None, source_sha256: str | None, now: float) -> dict:
    keep = previous if previous is not None and previous.get("schema") == SCHEMA else {}
    return {"schema": SCHEMA, "type": OPPORTUNITY_TYPE, "method": METHOD, **hint,
            "attempted_at": datetime.now(timezone.utc).isoformat(), "attempted_at_epoch": now,
            "searched_at_epoch": keep.get("searched_at_epoch"),
            "theme_songs_searched_at_epoch": keep.get("theme_songs_searched_at_epoch"),
            "theme_songs": list(keep.get("theme_songs") or []),
            "checked_sources": sorted(set(keep.get("checked_sources", [])) | {source_sha256}),
            "candidates": {k: dict(v) for k, v in (keep.get("candidates") or {}).items()},
            "albums": {k: dict(v) for k, v in (keep.get("albums") or {}).items()},
            "releases": list(keep.get("releases") or []), "queries": list(keep.get("queries") or []),
            "attempts": int(keep.get("attempts") or 0) + 1, "state": "searching"}


def _candidate(opportunity: dict, track: dict, origin: dict) -> dict:
    key = track.get("id") or track["url"]
    row = opportunity["candidates"].setdefault(key, {"id": track.get("id"), "url": track["url"],
                                                      "title": track.get("title"), "origins": [],
                                                      "state": "pending", "checks": {}})
    if origin not in row["origins"]:
        row["origins"].append(origin)
    if not row.get("title") and track.get("title"):
        row["title"] = track["title"]
    return row


def _priority(row: dict) -> tuple:
    """確かめる順番：曲の検索の順位、作品名の合うアルバムの見本、その他のアルバムの見本、全曲の確認。"""
    best = (9, 999, 999)
    for origin in row["origins"]:
        if origin["kind"] == "song_search":
            best = min(best, (0, origin["rank"], 0))
        elif origin["kind"] == "theme_song_search":
            best = min(best, (0, SONG_CANDIDATES + origin["rank"], origin.get("number", 0)))
        elif origin["kind"] == "album_sample":
            best = min(best, (1 if origin.get("score", 0) >= STRONG_SCORE else 2, origin["album_rank"],
                              origin["position"]))
        elif origin["kind"] == "album_full":
            best = min(best, (3, origin.get("album_rank", 0), origin["position"]))
    return best


def auth_method(url: str, state_root: Path | None, options: dict) -> str:
    """この取得で使う認証の種類（"none" / "app_login" 等）。通常のURLの取得と同じ選び方。"""
    if options.get("auth") is not None:
        return options["auth"].method
    if options.get("backend") is not None:
        return "none"
    from lss.acquisition import login_profile

    try:
        return login_profile.auth_for(url, state_root).method
    except Exception:  # noqa: BLE001 - ログインの状態が読めなければ認証なしとして扱う
        return "none"


def _reopen_for_login(opportunity: dict, method: str) -> int:
    """前の機会はログインが無く、会員向けとして打ち切った曲・盤を、ログインした今はやり直す。"""
    if method == "none" or opportunity.get("auth_method", "none") == method:
        return 0
    reopened = 0
    for row in opportunity["candidates"].values():
        if row["state"] in ("restricted", "restricted_album", "auth_failed", "needs_login") and not row.get("sha256"):
            row["state"] = "pending"
            row.pop("failure", None)
            reopened += 1
    for album_row in opportunity["albums"].values():
        if album_row.get("state") == "restricted":
            album_row["state"] = "release" if album_row.get("verified_track_ids") else "candidate"
        elif album_row.get("state") == "needs_login":
            # 盤の曲目そのものがログインを求めた：ログインした今、検索からやり直す。
            album_row["state"] = "candidate"
            opportunity["searched_at_epoch"] = None
            reopened += 1
        album_row.pop("restricted_failures", None)
        album_row.pop("needs_login", None)
    for row in opportunity["candidates"].values():
        row.pop("needs_login", None)
    return reopened


def _needs_login(opportunity: dict, method: str) -> bool:
    """会員向け・ログインが要るために取れなかった曲・盤がある（ログインが無い時だけ。ログイン済みの拒否は
    `login_attention`）。発見の失敗ではなく、利用者がLSSでログインすれば次の機会に取る。"""
    if method != "none":
        return False
    return (any(row.get("needs_login") or row.get("state") in ("restricted", "restricted_album", "needs_login")
                for row in opportunity["candidates"].values())
            or any(row.get("needs_login") or row.get("state") in ("restricted", "needs_login")
                   for row in opportunity["albums"].values()))


def _known_by_track(state_root: Path | None) -> dict[str, str]:
    """取得済みの参照の知識（取得元の識別子 → 内容SHA-256）。同じ曲を別の作品のために取り直さない。"""
    found = {}
    for record in reference_store.records(state_root):
        authority = (record.get("provenance") or {}).get("authority") or {}
        if authority.get("id") and record.get("origin") == reference_store.ORIGIN_ACQUIRED:
            found.setdefault(str(authority["id"]), record["sha256"])
    return found


# 取得ツールの不在・通信の失敗は、作品の「見つからない」ではない。回数を数えず、次の機会にやり直す。
def _deferrable_kinds() -> frozenset:
    from lss.acquisition.store import FailureKind

    return frozenset({FailureKind.TOOL_UNAVAILABLE, FailureKind.NETWORK, FailureKind.INTERRUPTED,
                      FailureKind.BUSY, FailureKind.RATE_LIMITED})


def _short(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:DETAIL_LIMIT]


def _default_ensure(search, backend):
    """既定の取得手段（配備版のyt-dlp）を使う時だけ、取得と同じ検証付きの導入（署名・SHA-256）を使う。"""
    if search is not None or backend is not None:
        return None
    from lss.acquisition import service

    return service._ensure_provider


def discover(source: Path, *, source_sha256: str | None, cache_dir: Path, state_root: Path | None = None,
             search: SearchBackend | None = None, backend=None, query_samples=None,
             should_stop=lambda: False, now: float | None = None, on_ready=lambda snapshot: None,
             courtesy=None, ensure_provider=None, **acquire_options) -> dict:
    """作品の参照の機会を見つけて取得・確認する。

    - 同じ作品の検索は期限内に繰り返さない（目録の答えも期限つきで残る）。
    - 新しい話では、これまでに取得した候補の指紋を通信なしで先に照合し直す（前の話で鳴らなかった曲が、
      この話で鳴っていれば作品へ結び付ける）。
    - 一回の機会で確認のために取得する曲数には上限があり、残りは次の機会へ回る（保留の続き）。
    - 混雑の合図を受けたら休み（`deferred`）、次の機会に続きから行う。
    取得ツール（yt-dlp）が導入されていない配備版では、最初の目録の問い合わせの前に、URLの取得と同じ検証付きの
    導入（`service._ensure_provider`）を行い、ツールが無い・古い失敗の時は1回だけ導入を確かめてやり直す。
    ツール・通信の失敗は `deferred`（失敗の種類と詳細を残し、回数を数えない）。会員向け・ログインが要る曲・盤は
    `needs_login`（発見の失敗ではない）。
    `on_ready` は、この番組の判定に使う参照が決まった時（収録盤の取得まで、または見つからない・保留の時）に
    呼ぶ。呼び出し側は待てる時間までしか待たず、残りの取得は背景で続いて次の話から使う。
    """
    from lss.acquisition import courtesy as courtesy_module
    from lss.acquisition.courtesy import Deferred
    from lss.acquisition.store import AcquisitionFailure, FailureKind

    hint = work_hint(source, source_sha256, state_root)
    if hint is None:
        return {"state": "no_work_hint"}
    now = time.time() if now is None else now
    courtesy = courtesy or courtesy_module.shared(state_root)
    work = hint["work_key"]
    previous = load_opportunity(work, state_root)
    if previous is not None and previous.get("schema") != SCHEMA:
        # 前の版の記録（アルバム単位）。確かめたアルバムがあればそのまま使う。
        if previous.get("state") == "verified":
            return {**previous, "reused": True}
        previous = None
    opportunity = _new_opportunity(hint, previous, source_sha256, now)
    method = auth_method("https://music.youtube.com/", state_root, {"backend": backend, **acquire_options})
    opportunity["reopened_after_login"] = _reopen_for_login(opportunity, method)
    opportunity["auth_method"] = method
    for row in opportunity["candidates"].values():
        if row["state"] == "acquire_failed" and _signal(row.get("failure")) == courtesy_module.TRANSIENT:
            row["state"] = "pending"   # earlier versions closed a track on a network failure
    clock = time.perf_counter()
    counters = {"verification_downloads": 0, "full_verification_downloads": 0, "release_downloads": 0,
                "known_track_reuse": 0, "local_rematched": 0, "waveform_reacquired": 0}
    opportunity["counters"] = counters
    catalog: Catalog | None = None
    announced = []

    def ready() -> None:
        if not announced:
            announced.append(True)
            on_ready(dict(opportunity))

    def finish(state: str, **extra) -> dict:
        opportunity["state"] = state
        opportunity.update(extra)
        # 作品に結び付かなかった候補の重い音源はすぐ退役させる（指紋は残す。取り直せる）。
        unlinked = [row["sha256"] for row in opportunity["candidates"].values()
                    if row.get("sha256") and row["state"] not in ("verified", "release_track")]
        try:
            opportunity["retired_unverified"] = len(reference_store.retire_now(
                unlinked, cache_dir=cache_dir, state_root=state_root, reason="unverified_candidate"))
        except (OSError, ValueError):
            opportunity["retired_unverified"] = None
        opportunity["needs_login"] = _needs_login(opportunity, method)
        opportunity["catalog"] = ({"requests": catalog.requests, "cached": catalog.cached}
                                  if catalog is not None else {"requests": 0, "cached": 0})
        opportunity["seconds"] = round(time.perf_counter() - clock, 1)
        opportunity["verified_tracks"] = sorted(k for k, r in opportunity["candidates"].items()
                                                if r["state"] == "verified")
        _save_opportunity(opportunity, state_root)
        ready()
        return dict(opportunity)

    def record_check(row: dict, result: dict | None) -> None:
        row["checks"][source_sha256] = result["tier"] if result else "no_match"
        if result:
            if row["state"] != "verified":
                row["state"] = "verified"
                row["verified_by"] = {"source_sha256": source_sha256, **result}
            reference_store.link_work(row["sha256"], work, state_root)
        elif row["state"] in ("pending", "downloaded"):
            row["state"] = "not_verified"

    try:
        if query_samples is None:
            query_samples = identity.decode_mono(Path(source))

        # 1. 手元の知識：これまでに取得した候補（指紋は残る）をこの話の音声で照合し直す。指紋の票が揃った
        #    曲だけ、重い音源を取り直して波形で確かめる（目録の検索・全曲の取り直しはしない）。
        reacquire = {"backend": backend, **acquire_options} if backend is not None else dict(acquire_options)
        reacquire["courtesy"] = courtesy
        known = [(row, reference_store.load(row["sha256"], state_root))
                 for row in opportunity["candidates"].values()
                 if row.get("sha256") and row["checks"].get(source_sha256) in (None, PENDING_WAVEFORM)]
        known = [(row, record) for row, record in known if record is not None]
        if known:
            found = verify(query_samples, [record for _, record in known], cache_dir=cache_dir,
                           state_root=state_root, network=False)
            counters["local_rematched"] = len(known)
            pending = [record for _, record in known
                       if (found.get(record["sha256"]) or {}).get("tier") == PENDING_WAVEFORM]
            if pending:
                counters["waveform_reacquired"] = len(pending)
                confirmed = verify(query_samples, pending, cache_dir=cache_dir, state_root=state_root,
                                   acquire_options=reacquire)
                for record in pending:
                    found[record["sha256"]] = confirmed.get(record["sha256"])
            for row, record in known:
                result = found.get(record["sha256"])
                if result is not None and result["tier"] == PENDING_WAVEFORM:
                    row["checks"][source_sha256] = PENDING_WAVEFORM   # 取り直せなかった：次の機会に確かめる
                    continue
                record_check(row, result)

        def from_theme(key: str, seen: frozenset = frozenset()) -> bool:
            """主題歌から来た候補か：主題歌の検索に出た曲、または主題歌の曲だけから辿った収録盤の曲。"""
            row = opportunity["candidates"][key]
            kinds = {origin["kind"] for origin in row["origins"]}
            if "theme_song_search" in kinds:
                return True
            albums = [origin.get("album_url") for origin in row["origins"] if origin["kind"] == "release_track"]
            if not albums or kinds != {"release_track"}:
                return False
            for url in albums:
                seeds = [other for other in (opportunity["albums"].get(url, {}).get("verified_track_ids") or [])
                         if other != key and other not in seen and other in opportunity["candidates"]]
                if not seeds or not all(from_theme(other, seen | {key}) for other in seeds):
                    return False
            return True

        def verified_any() -> bool:
            """作品の劇伴の曲が確かめられたか。主題歌から来た曲は数えない：主題歌（とその盤の TV size 等）が
            鳴っていることは、劇伴の既発売の盤があるかを何も言わない。"""
            return any(row["state"] == "verified" and not from_theme(key)
                       for key, row in opportunity["candidates"].items())

        if not verified_any():
            for row in opportunity["candidates"].values():
                if row["state"] == "not_needed":
                    row["state"] = "pending"   # 前の版は主題歌の確認だけで盤の見本を閉じた：劇伴がまだなら確かめる

        searched = opportunity.get("searched_at_epoch")
        fresh = searched is not None and now - float(searched) < RETRY_AFTER_SECONDS
        themes_searched = opportunity.get("theme_songs_searched_at_epoch")
        themes_fresh = themes_searched is not None and now - float(themes_searched) < RETRY_AFTER_SECONDS
        undecided = [row for row in opportunity["candidates"].values()
                     if row["state"] in ("pending", "auth_failed")
                     or row["checks"].get(source_sha256) == PENDING_WAVEFORM]
        unreleased = [row for row in opportunity["candidates"].values()
                      if row["state"] == "verified" and not row.get("release_checked")]
        incomplete = [url for url in opportunity["releases"]
                      if opportunity["albums"].get(url, {}).get("state") not in ("acquired", "restricted", "needs_login")]
        if fresh and themes_fresh and not undecided and not unreleased and not incomplete:
            has = any(row["state"] == "verified" for row in opportunity["candidates"].values())
            return finish("verified" if has else "no_verified_release", reused=True)

        ensure = ensure_provider if ensure_provider is not None else _default_ensure(search, backend)
        ensured: list = []

        def recover(failure) -> bool:
            """取得ツールが無い・古い：URLの取得と同じ検証付きの導入を1回だけ行い、問い合わせをやり直す。"""
            if ensure is None or ensured or failure.kind not in (FailureKind.TOOL_UNAVAILABLE,
                                                                 FailureKind.EXTRACTOR_FAILED):
                return False
            ensured.append(True)
            try:
                result = ensure(after_failure=True)
            except Exception:  # noqa: BLE001 - 導入の失敗は元の失敗として扱う（導入の確かめ自体は取得を止めない）
                result = None
            opportunity["provider_ensure"] = (result or {}).get("status") or "none"
            if failure.kind is FailureKind.TOOL_UNAVAILABLE:
                return True
            return result is not None and result.get("status") == "installed"

        if search is None and backend is None and ensure is not None:
            from lss.acquisition import provider

            try:
                provider.active()
            except AcquisitionFailure as missing:
                # 配備版で導入済みの版が無い：最初の目録の問い合わせの前に導入する（検索が取得より先に来る）。
                recover(missing)
        search = search or SearchBackend(backend)
        catalog = Catalog(search, state_root=state_root, courtesy=courtesy, now=now, should_stop=should_stop,
                          recover=recover)
        known_tracks = _known_by_track(state_root)

        def acquire(row: dict, role: str, link: bool, album: dict | None = None) -> str | None:
            if row.get("id") and row["id"] in known_tracks:
                # 別の作品・前の機会で取得済みの曲：知識をそのまま使う（重い音源は要る時に取り直す）。
                row["sha256"] = known_tracks[row["id"]]
                counters["known_track_reuse"] += 1
                if link:
                    reference_store.link_work(row["sha256"], work, state_root)
                if row["state"] == "pending":
                    row["state"] = "downloaded"
                return row["sha256"]
            if album is None:
                # アルバムの候補から来た曲は、その盤を取得の来歴に残す（同一性の根拠ではない）。
                url = next((origin["album_url"] for origin in row["origins"] if origin.get("album_url")), None)
                if url is not None:
                    album = {"url": url, "title": opportunity["albums"].get(url, {}).get("title")}
            discovery = {"method": METHOD, "work_key": work, "role": role, "query_title": hint["title"],
                         "track_id": row.get("id"), "track_title": row.get("title"),
                         **({"album_url": album["url"], "album_title": album.get("title")} if album else {})}
            with courtesy.slot(row["url"], purpose=role, should_stop=should_stop):
                result = acquire_into_store(row["url"], cache_dir=cache_dir, state_root=state_root,
                                            work_keys=(work,) if link else (), discovery=discovery,
                                            backend=backend, **acquire_options)
            failure = next((item["failure"] for item in result["failed"]), None)
            courtesy.report(row["url"], _signal(failure))
            if not result["ingested"]:
                row["failure"] = failure
                row["failure_detail"] = next((item.get("detail") for item in result["failed"]), None)
                if failure in (FailureKind.AUTH_REQUIRED.value, FailureKind.AUTH_EXPIRED.value) and method != "none":
                    # ログインしているのに拒否された（期限切れ・会員でない等）：盤を打ち切らず、次の機会にやり直す。
                    row["state"] = "auth_failed"
                    opportunity["login_attention"] = True
                elif failure == FailureKind.AUTH_REQUIRED.value:
                    # ログインが無く、認証なしでは取れない（有料会員・会員限定）。LSSでログインすれば次の機会に取る。
                    row["state"] = "restricted"
                    row["needs_login"] = True
                    for origin in row["origins"]:
                        album_row = opportunity["albums"].get(origin.get("album_url") or "")
                        if album_row is None:
                            continue
                        album_row["needs_login"] = True
                        album_row["restricted_failures"] = int(album_row.get("restricted_failures") or 0) + 1
                        if (album_row["restricted_failures"] >= RESTRICTED_ALBUM_FAILURES
                                and not album_row.get("acquired_ok")):
                            album_row["state"] = "restricted"
                elif row["state"] == "pending" and _signal(failure) != courtesy_module.TRANSIENT:
                    # A dropped connection says nothing about the track: it stays pending for the next chance.
                    row["state"] = "acquire_failed"
                if _signal(failure) == courtesy_module.RATE_LIMITED:
                    raise Deferred("youtube", courtesy.cooldown_remaining(row["url"]))
                return None
            row["sha256"] = result["ingested"][0]
            for origin in row["origins"]:
                album_row = opportunity["albums"].get(origin.get("album_url") or "")
                if album_row is not None:
                    album_row["acquired_ok"] = int(album_row.get("acquired_ok") or 0) + 1
            if row.get("id"):
                known_tracks[row["id"]] = row["sha256"]
            if row["state"] == "pending":
                row["state"] = "downloaded"
            return row["sha256"]

        def album(url: str) -> dict | None:
            try:
                listed = catalog.tracks(url)
            except AcquisitionFailure as exc:
                if _signal(exc.kind) == courtesy_module.RATE_LIMITED:
                    raise Deferred("youtube", courtesy.cooldown_remaining(url))
                if exc.kind in (FailureKind.TOOL_UNAVAILABLE, FailureKind.NETWORK, FailureKind.INTERRUPTED):
                    raise   # ツール・通信の不調はこの盤だけの問題ではない：保留にして次の機会に続ける
                album_row = opportunity["albums"].setdefault(url, {"url": url})
                if exc.kind in (FailureKind.AUTH_REQUIRED, FailureKind.AUTH_EXPIRED):
                    # 会員向けの盤（曲目を読むのにログインが要る）。失敗ではなく、ログイン待ち。
                    album_row.update(state="needs_login", needs_login=True, failure=exc.kind.value,
                                     failure_detail=_short(exc))
                    return None
                album_row["state"] = "resolve_failed"
                album_row["failure"] = exc.kind.value
                return None
            row = opportunity["albums"].setdefault(url, {"url": url, "state": "candidate"})
            row.update(title=listed.get("title"), track_count=len(listed["tracks"]),
                       track_ids=[t.get("id") for t in listed["tracks"][:MAX_TRACKS]])
            return listed

        def find_release(row: dict) -> None:
            """確かめた曲の収録盤：曲目に同じ曲の識別子がある盤（名前の似た盤は使わない）。"""
            if row.get("release_checked"):
                return
            release = next((url for url, album_row in opportunity["albums"].items()
                            if row.get("id") and row["id"] in (album_row.get("track_ids") or [])), None)
            if release is None and row.get("sha256"):
                record = reference_store.load(row["sha256"], state_root) or {}
                title = (record.get("metadata") or {}).get("album")
                if title:
                    for entry in catalog.albums(title)[:RELEASE_LOOKUPS]:
                        if album(entry["url"]) is None:
                            continue
                        opportunity["albums"][entry["url"]].setdefault("found_via", "release_lookup")
                        if row.get("id") in opportunity["albums"][entry["url"]]["track_ids"]:
                            release = entry["url"]
                            break
            row["release_checked"] = True
            row["release"] = release
            if release is None:
                return
            album_row = opportunity["albums"][release]
            album_row["verified_track_ids"] = sorted(set(album_row.get("verified_track_ids") or [])
                                                     | {row.get("id")})
            album_row["title_names_work"] = _normal(hint["title"]) in _normal(album_row.get("title"))
            if release not in opportunity["releases"]:
                opportunity["releases"].append(release)
                album_row["state"] = "release"

        def in_release(row: dict) -> bool:
            return bool(row.get("id")) and any(row["id"] in (opportunity["albums"].get(url, {}).get("track_ids") or [])
                                               for url in opportunity["releases"])

        def restricted(row: dict) -> bool:
            """曲の検索から来ていない候補で、出どころの盤がすべて認証を求める盤。"""
            albums = [origin.get("album_url") for origin in row["origins"] if origin.get("album_url")]
            return (not any(origin["kind"] in ("song_search", "theme_song_search") for origin in row["origins"])
                    and bool(albums)
                    and all(opportunity["albums"].get(url, {}).get("state") == "restricted" for url in albums))

        def check(rows: list[dict], role: str, budget: str, limit: int, *, full_album: dict | None = None) -> None:
            """一曲ずつ取得して照合する。確かめた曲の収録盤が分かれば、その盤の他の候補は確認に取らない
            （盤ごと取得する）。認証を求める盤の残りの曲は取りに行かない。全曲の確認は、盤が認証で止まれば打ち切る。"""
            for row in rows:
                if should_stop() or counters[budget] >= limit:
                    break
                if full_album is not None and full_album.get("state") == "restricted":
                    break
                if in_release(row) or source_sha256 in row["checks"] or row["state"] == "restricted":
                    continue
                if restricted(row):
                    row["state"] = "restricted_album"
                    continue
                if row.get("sha256") is None:
                    counters[budget] += 1
                    if acquire(row, role, link=False) is None:
                        continue
                record = reference_store.load(row["sha256"], state_root)
                if record is None:
                    continue
                result = verify(query_samples, [record], cache_dir=cache_dir, state_root=state_root,
                                acquire_options=reacquire).get(row["sha256"])
                if result is not None and result["tier"] == PENDING_WAVEFORM:
                    row["checks"][source_sha256] = PENDING_WAVEFORM
                    continue
                record_check(row, result)
                if row["state"] == "verified":
                    find_release(row)

        if not fresh:
            # 2. 候補：曲の検索（作品名）とアルバムの検索（作品名＋サウンドトラックの語）。
            opportunity["queries"] = [{"kind": "songs", "query": hint["title"]}]
            for rank, track in enumerate(catalog.songs(hint["title"])[:SONG_CANDIDATES]):
                _candidate(opportunity, track, {"kind": "song_search", "query": hint["title"], "rank": rank})
            listed: dict[str, dict] = {}
            for suffix in QUERY_SUFFIXES:
                query = f"{hint['title']} {suffix}"
                opportunity["queries"].append({"kind": "albums", "query": query})
                for entry in catalog.albums(query)[:MAX_ALBUMS * 2]:
                    listed.setdefault(entry["url"], entry)
                if len(listed) >= MAX_ALBUMS * 2:
                    break
            ranked = []
            for rank, url in enumerate(list(listed)[:MAX_ALBUMS * 2]):
                if should_stop():
                    break
                found_album = album(url)
                if found_album is None:
                    continue
                score = _score(found_album.get("title"), hint["title"])
                opportunity["albums"][url].update(score=score, search_rank=rank)
                ranked.append((score, rank, url, found_album))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            # 今回の検索に出なかった盤も記録には残す。順位は今回の候補だけに付ける。
            for album_row in opportunity["albums"].values():
                album_row.pop("album_rank", None)
            for album_rank, (score, rank, url, found_album) in enumerate(ranked[:MAX_ALBUMS]):
                opportunity["albums"][url]["album_rank"] = album_rank
                for position, track in enumerate(_sample(found_album["tracks"][:MAX_TRACKS], SAMPLE_TRACKS)):
                    _candidate(opportunity, track, {"kind": "album_sample", "album_url": url,
                                                    "album_rank": album_rank, "score": score,
                                                    "position": position})
            opportunity["searched_at_epoch"] = now
        if not themes_fresh:
            # 主題歌（作品の記事）：作品名の検索とは別の期限で一度探す（前の版で探し終えた作品にも）。
            themes = theme_songs(source, state_root)
            opportunity["theme_songs"] = themes
            for number, song in enumerate(themes):
                if should_stop():
                    break
                opportunity["queries"].append({"kind": "songs", "query": song["query"], "theme": song["kind"]})
                for rank, track in enumerate(catalog.songs(song["query"])[:THEME_CANDIDATES]):
                    _candidate(opportunity, track, {"kind": "theme_song_search", "query": song["query"],
                                                    "theme": song["kind"], "rank": rank, "number": number})
            opportunity["theme_songs_searched_at_epoch"] = now

        # 3. 確認：曲の検索の候補を順位の順に取得して照合する（上限つき）。アルバムの見本は、まだ何も
        #    確かめられていない時と、作品名とサウンドトラックの語が両方合う盤だけ（無駄な取得をしない）。
        for row in opportunity["candidates"].values():
            if row["state"] == "auth_failed":
                row["state"] = "pending"   # ログインしての拒否は一時的なことがある：機会ごとに一度やり直す
        pending = sorted((row for row in opportunity["candidates"].values() if row["state"] == "pending"),
                         key=_priority)
        check([row for row in pending if _priority(row)[0] == 0], "verification_sample",
              "verification_downloads", VERIFY_BUDGET)
        samples = [row for row in pending if 0 < _priority(row)[0] < 3 and row["state"] == "pending"
                   and (not verified_any() or _priority(row)[0] == 1)]
        check(samples, "verification_sample", "verification_downloads", VERIFY_BUDGET)
        if verified_any():
            # 曲の検索で確かめられた：作品名の合わないアルバムの見本は取らない（無駄な取得をしない）。
            for row in pending:
                if row["state"] == "pending" and _priority(row)[0] == 2 and not in_release(row):
                    row["state"] = "not_needed"

        if not verified_any():
            # 作品名とサウンドトラックの語が両方合う盤は、見本が外れただけかもしれない。
            # 題が少しでも合う盤のうち、制限がなく曲目を読める最上位の盤も全曲で確かめる。
            top_found = False
            for url, album_row in sorted(opportunity["albums"].items(),
                                         key=lambda item: item[1].get("album_rank", 99)):
                if should_stop():
                    break
                rank = album_row.get("album_rank")
                if rank is None or album_row.get("state") == "restricted":
                    continue
                score = album_row.get("score", 0)
                is_top = not top_found and score > 0
                if score < STRONG_SCORE and not is_top:
                    continue
                if album_row.get("full_verification"):
                    # 全曲確認を始めた盤は曲目を取得済み。
                    top_found = top_found or is_top
                    continue
                listed_album = album(url)
                if listed_album is None:
                    continue
                album_row["full_verification"] = True
                if is_top:
                    top_found = True
                    if rank != 0:
                        album_row["full_verification_reason"] = "top_restricted"
                rows = [_candidate(opportunity, track, {"kind": "album_full", "album_url": url,
                                                        "album_rank": rank,
                                                        "position": position})
                        for position, track in enumerate(listed_album["tracks"][:MAX_TRACKS])]
                check([row for row in rows if row["state"] == "pending"], "verification_full",
                      "full_verification_downloads", MAX_TRACKS, full_album=album_row)
                if any(row["state"] == "verified" for row in rows):
                    break
        # 4. 収録盤：手元の知識で確かめた曲など、まだ収録盤を探していない確かめた曲。
        for row in [r for r in opportunity["candidates"].values()
                    if r["state"] == "verified" and not r.get("release_checked")]:
            if should_stop():
                break
            find_release(row)

        if not any(row["state"] == "verified" for row in opportunity["candidates"].values()):
            return finish("no_verified_release")
        opportunity["state"] = "verified_partial"
        _save_opportunity(opportunity, state_root)

        # 5. 収録盤の残りの曲を取得して作品へ結び付ける（盤の曲は作品の劇伴として次の話でも使う）。
        complete = True
        for url in opportunity["releases"][:MAX_RELEASES]:
            album_row = opportunity["albums"][url]
            if album_row.get("state") in ("acquired", "restricted"):
                continue
            listed_album = album(url)
            if listed_album is None:
                complete = False
                continue
            failed = 0
            for position, track in enumerate(listed_album["tracks"][:MAX_TRACKS]):
                if should_stop():
                    complete = False
                    break
                row = _candidate(opportunity, track, {"kind": "release_track", "album_url": url,
                                                      "position": position})
                if row.get("sha256") is None and (row["state"] == "restricted"
                                                  or album_row.get("state") == "restricted"):
                    continue   # 認証なしでは取れない曲・盤（取れた曲だけを作品の参照にする）
                if row.get("sha256") is None:
                    counters["release_downloads"] += 1
                    if acquire(row, "release_track", link=True, album=listed_album) is None:
                        failed += 1
                        continue
                reference_store.link_work(row["sha256"], work, state_root,
                                          release={"album_url": url, "album_title": listed_album.get("title")})
                row.setdefault("linked_as", "release_track")
                if row["state"] in ("pending", "downloaded", "not_verified", "acquire_failed"):
                    row["state"] = "release_track"
            album_row["acquired"] = sum(
                1 for t in listed_album["tracks"][:MAX_TRACKS]
                if (opportunity["candidates"].get(t.get("id") or t["url"]) or {}).get("sha256"))
            album_row["failed"] = failed
            if album_row.get("state") == "restricted" or not failed and album_row["acquired"] >= min(
                    len(listed_album["tracks"]), MAX_TRACKS) - sum(
                    1 for track in listed_album["tracks"][:MAX_TRACKS]
                    if (opportunity["candidates"].get(track.get("id") or track["url"]) or {}).get("state")
                    == "restricted"):
                # 取れる曲はすべて取った（認証を求める曲・盤は取らない。次の機会にやり直さない）。
                album_row["state"] = "acquired" if album_row.get("state") != "restricted" else "restricted"
            else:
                complete = False
        if len(opportunity["releases"]) > MAX_RELEASES:
            complete = False
        return finish("verified" if complete else "verified_partial")
    except Deferred as deferred:
        return finish("deferred", deferred={"host": deferred.host, "seconds": round(deferred.seconds, 1)})
    except Exception as exc:  # noqa: BLE001 - 発見の失敗で単話の処理を止めない
        kind = getattr(exc, "kind", None) if isinstance(exc, AcquisitionFailure) else None
        extra = {"error": type(exc).__name__, "failure_detail": _short(exc)}
        if kind is not None:
            extra["failure_kind"] = kind.value
        if kind in (FailureKind.AUTH_REQUIRED, FailureKind.AUTH_EXPIRED):
            # 目録そのものがログインを求めた：発見の失敗ではない。ログインした後の機会にやり直す。
            state = "needs_login"
            opportunity["attempts"] = max(0, opportunity["attempts"] - 1)
        elif kind in _deferrable_kinds():
            # ツールの不在・通信の失敗：「見つからない」と区別し、数える回数を使わず、次の機会に再試行する。
            state = "deferred"
            extra["retryable"] = True
            opportunity["attempts"] = max(0, opportunity["attempts"] - 1)
        else:
            state = "failed"
        try:
            return finish(state, **extra)
        except OSError:
            opportunity.update(state=state, **extra)
            ready()
            return opportunity


_ACTIVE: dict = {}
_ACTIVE_LOCK = threading.Lock()


def wait_active(source: Path, source_sha256: str | None, state_root: Path | None, timeout: float,
                checkpoint=None) -> bool:
    """この番組の作品の参照の準備（アルバムの残りの取得）が動いていれば、終わるまで待つ（上限つき）。

    動いていた準備があれば True。`checkpoint` は待つ間に呼び、終了の要求を例外で伝える。"""
    hint = work_hint(source, source_sha256, state_root)
    key = (hint or {}).get("work_key") or f"source-{source_sha256}"
    with _ACTIVE_LOCK:
        running = _ACTIVE.get(key)
    if running is None or running._thread is None:
        return False
    deadline = time.monotonic() + max(0.0, timeout)
    while running._thread.is_alive() and time.monotonic() < deadline:
        running._thread.join(min(5.0, max(0.0, deadline - time.monotonic())))
        if checkpoint is not None:
            checkpoint()
    return True


@dataclass
class Preparation:
    """BGMの処理と並行して参照の機会を探す（GPUの分離の間にネットワークの待ちを重ねる）。"""

    source: Path
    source_sha256: str | None
    cache_dir: Path
    state_root: Path | None = None
    options: dict = field(default_factory=dict)
    result: dict | None = None
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def start(self) -> "Preparation":
        """同じ作品の準備が既に動いていれば、それに合流する（同じ取得を重ねない）。"""
        self.ready = threading.Event()
        if not enabled():
            self.result = {"state": "disabled"}
            self.ready.set()
            return self
        hint = work_hint(self.source, self.source_sha256, self.state_root)
        key = (hint or {}).get("work_key") or f"source-{self.source_sha256}"
        with _ACTIVE_LOCK:
            running = _ACTIVE.get(key)
            if running is not None and running._thread is not None and running._thread.is_alive():
                return running
            _ACTIVE[key] = self

        def ready(snapshot):
            self.result = snapshot
            self.ready.set()

        def run():
            try:
                self.result = discover(self.source, source_sha256=self.source_sha256, cache_dir=self.cache_dir,
                                       state_root=self.state_root, should_stop=self._stop.is_set,
                                       on_ready=ready, **self.options)
            finally:
                self.ready.set()
                with _ACTIVE_LOCK:
                    if _ACTIVE.get(key) is self:
                        _ACTIVE.pop(key, None)

        self._thread = threading.Thread(target=run, name="lss-reference-preparation", daemon=True)
        self._thread.start()
        return self

    def wait_ready(self, timeout: float) -> dict:
        """この番組の判定に使う参照が決まるまで待つ（待てる時間まで）。残りの取得は背景で続く。"""
        self.ready.wait(max(0.0, timeout))
        state = dict(self.result or {})
        if not self.ready.is_set():
            state["wait"] = "deadline_reached"
        elif self._thread is not None and self._thread.is_alive():
            state["wait"] = "continuing_in_background"
        return state

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
