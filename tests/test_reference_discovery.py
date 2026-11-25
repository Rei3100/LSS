"""作品の参照の発見（曲を単位にした発見・音声の確認・収録盤・記憶）と、外部サービスへの礼儀。

外部サービスへは接続しない（目録と取得は偽の手段）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.acquisition import courtesy as cm
from lss.acquisition.store import AcquisitionFailure, FailureKind
from lss.bgm import identity, reference_acquisition as ra, reference_store as rs
from tests.test_acquisition import NO_WAIT, FakeBackend
from tests.test_bgm_identity import _music

LOW = identity.SAMPLE_RATE


@pytest.fixture
def state(tmp_path, monkeypatch):
    root = tmp_path / "state"
    monkeypatch.setenv("LSS_STATE_DIR", str(root))
    monkeypatch.setenv("LSS_ACQUISITION_DIR", str(tmp_path / "Downloads"))
    return root


def _cache(state: Path) -> Path:
    return state / "bgm-reference-fingerprints"


def _track(path: Path, seed: int, seconds: float = 40.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, _music(seed, seconds), LOW, subtype="PCM_16")
    return path


class Clock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _polite(state=None, clock=None, **policy) -> cm.Courtesy:
    clock = clock or Clock()
    return cm.Courtesy(state, cm.Policy(**{"interval": (0.0, 0.0), **policy}), clock=clock, sleep=clock.sleep)


class AlbumBackend(FakeBackend):
    """取得の台帳に収録盤の名前（YouTube Musicの曲の `album`）を返す偽の取得手段。"""

    def __init__(self, media, albums: dict[str, str], **kwargs):
        super().__init__(media, **kwargs)
        self.albums = albums

    def describe(self, entry, auth):
        description = super().describe(entry, auth)
        if entry["id"] in self.albums:
            description.record["metadata"]["album"] = self.albums[entry["id"]]
        return description


class Catalog:
    """偽の目録：曲の検索・アルバムの検索・曲目。問い合わせを数える。"""

    def __init__(self, songs: dict[str, list[str]] | None = None, albums: dict[str, list[str]] | None = None,
                 listings: dict[str, tuple[str, list[str]]] | None = None):
        self.song_answers = songs or {}
        self.album_answers = albums or {}
        self.listings = listings or {}
        self.calls = []

    def songs(self, query):
        self.calls.append(("songs", query))
        return [{"url": f"https://example.test/{i}", "id": i, "title": f"曲 {i}"}
                for i in self.song_answers.get(query, [])]

    def albums(self, query):
        self.calls.append(("albums", query))
        return [{"url": f"https://music.example.test/browse/{a}", "id": a} for a in self.album_answers.get(query, [])]

    def tracks(self, url):
        self.calls.append(("tracks", url))
        title, ids = self.listings[url.rsplit("/", 1)[-1]]
        return {"title": title, "id": url, "url": url,
                "tracks": [{"url": f"https://example.test/{i}", "id": i, "title": f"曲 {i}"} for i in ids]}


def _episode(path: Path, pieces: list[tuple[int, float, float]], seconds: float = 120.0) -> Path:
    """（曲のseed, 番組の開始秒, 長さ）を並べた番組の音声。"""
    episode = np.zeros(int(seconds * LOW), np.float32)
    for seed, start, length in pieces:
        episode[int(start * LOW):int((start + length) * LOW)] = _music(seed, 40.0)[int(2 * LOW):int((2 + length) * LOW)]
    sf.write(path, episode, LOW, subtype="PCM_16")
    return path


def _linked(state, work_title):
    return {(rec.get("provenance") or {}).get("authority", {}).get("id")
            for rec in rs.records(state, work_key=ra.work_key(work_title))}


def test_track_search_finds_a_release_whose_title_does_not_name_the_work(state, tmp_path):
    """主題歌のシングル（題に作品名もサウンドトラックの語も無い）に入った劇伴を、曲の検索から見つける。"""
    seeds = {"k1": 11, "k2": 12, "k3": 13, "k4": 14, "w1": 21, "w2": 22, "x9": 31}
    media = {i: _track(tmp_path / "remote" / f"{i}.flac", seed) for i, seed in seeds.items()}
    backend = AlbumBackend(media, {"k1": "THEME STARS", "k2": "THEME STARS", "k3": "THEME STARS",
                                   "k4": "THEME STARS", "w1": "別の作品 OST", "w2": "別の作品 OST"})
    catalog = Catalog(
        songs={"番組名": ["x9", "k2"]},
        albums={"番組名 オリジナル・サウンドトラック": ["W"], "THEME STARS": ["Kfake", "K"]},
        listings={"W": ("別の作品 オリジナル・サウンドトラック", ["w1", "w2"]),
                  "Kfake": ("THEME STARS (Remixes)", ["z1"]),
                  "K": ("THEME STARS", ["k1", "k2", "k3", "k4"])})
    source = _episode(tmp_path / "『番組名』第10話「合宿」.wav", [(12, 5.0, 30.0), (13, 60.0, 30.0)])

    result = ra.discover(source, source_sha256="a" * 64, cache_dir=_cache(state), state_root=state,
                         search=catalog, backend=backend, pacing=NO_WAIT, courtesy=_polite())

    assert result["state"] == "verified"
    assert result["verified_tracks"] == ["k2"]
    assert result["candidates"]["k2"]["verified_by"]["tier"] == ra.TIER_EXACT
    release = "https://music.example.test/browse/K"
    assert result["releases"] == [release]
    album = result["albums"][release]
    assert album["title_names_work"] is False and album["found_via"] == "release_lookup"
    assert album["state"] == "acquired" and album["verified_track_ids"] == ["k2"]
    # 名前の似た別の盤（曲目に同じ曲が無い）は収録盤にしない。
    assert "https://music.example.test/browse/Kfake" not in result["releases"]
    # 収録盤の曲（番組でまだ確かめていないk3・k4も）が作品に結び付く。確かめられない候補は結び付かない。
    assert _linked(state, "番組名") == {"k1", "k2", "k3", "k4"}
    assert rs.load(rs._file_sha256(media["x9"]), state)["work_keys"] == []
    assert rs.load(rs._file_sha256(media["x9"]), state)["media"]["state"] == rs.STATE_RETIRED


def test_cross_work_release_track_is_linked_only_by_audio_and_the_next_episode_needs_no_network(state, tmp_path):
    """別の作品（映画）のサウンドトラックの曲が番組で鳴る。次の話は通信なしで、取得済みの指紋だけで照合し直す。"""
    seeds = {"m1": 41, "m2": 42, "m3": 43, "d1": 51}
    media = {i: _track(tmp_path / "remote" / f"{i}.flac", seed) for i, seed in seeds.items()}
    backend = AlbumBackend(media, {"m1": "映画 前作 サウンドトラック", "m2": "映画 前作 サウンドトラック",
                                   "m3": "映画 前作 サウンドトラック"})
    catalog = Catalog(songs={"番組名": ["d1", "m2"]},
                      albums={"映画 前作 サウンドトラック": ["M"]},
                      listings={"M": ("映画 前作 サウンドトラック", ["m1", "m2", "m3"])})
    first = _episode(tmp_path / "『番組名』第1話.wav", [(42, 10.0, 30.0)])
    result = ra.discover(first, source_sha256="b" * 64, cache_dir=_cache(state), state_root=state, search=catalog,
                         backend=backend, pacing=NO_WAIT, courtesy=_polite(), now=1000.0)
    assert result["state"] == "verified" and result["verified_tracks"] == ["m2"]
    assert result["albums"]["https://music.example.test/browse/M"]["title_names_work"] is False
    assert _linked(state, "番組名") == {"m1", "m2", "m3"}
    assert result["candidates"]["d1"]["state"] == "not_verified"
    downloads, calls = list(backend.downloads), list(catalog.calls)

    # 次の話：前の話で鳴らなかった候補 d1 が鳴る。目録は引かず、残った指紋で絞り、票の揃った d1 だけを
    # 取り直して波形で確かめて結び付ける（他の候補・収録盤は取り直さない）。
    second = _episode(tmp_path / "『番組名』第2話.wav", [(51, 20.0, 30.0)])
    again = ra.discover(second, source_sha256="c" * 64, cache_dir=_cache(state), state_root=state, search=catalog,
                        backend=backend, pacing=NO_WAIT, courtesy=_polite(), now=2000.0)
    assert again["candidates"]["d1"]["state"] == "verified"
    assert again["candidates"]["d1"]["verified_by"]["tier"] == ra.TIER_EXACT
    assert again["counters"]["local_rematched"] >= 4 and again["counters"]["waveform_reacquired"] == 1
    assert backend.downloads == downloads + ["d1"] and catalog.calls == calls
    assert "d1" in _linked(state, "番組名")
    downloads = list(backend.downloads)

    # 同じ話をもう一度：何も取得・照会しない。
    third = ra.discover(second, source_sha256="c" * 64, cache_dir=_cache(state), state_root=state,
                        search=catalog, backend=backend, pacing=NO_WAIT, courtesy=_polite(), now=3000.0)
    assert third["reused"] is True and backend.downloads == downloads and catalog.calls == calls


def test_album_search_still_verifies_a_soundtrack_by_audio_before_using_it(state, tmp_path):
    """題の上では先に出る別作品のアルバムと、本当のアルバム（曲の検索の無い目録でも動く）。"""
    wrong = {f"w{n}": _track(tmp_path / "remote" / f"w{n}.flac", 200 + n) for n in range(3)}
    right = {f"r{n}": _track(tmp_path / "remote" / f"r{n}.flac", 100 + n) for n in range(8)}
    backend = FakeBackend({**wrong, **right})

    class AlbumsOnly:
        def __init__(self):
            self.calls = []

        def albums(self, query):
            self.calls.append(query)
            return [{"url": "https://music.example.test/browse/A"}, {"url": "https://music.example.test/browse/B"}]

        def tracks(self, url):
            name = url.rsplit("/", 1)[-1]
            ids = list(wrong) if name == "A" else list(right)
            title = "番組名 オリジナル・サウンドトラック 2" if name == "A" else "番組名 サウンドトラック"
            return {"title": title, "url": url,
                    "tracks": [{"url": f"https://example.test/{i}", "id": i, "title": i} for i in ids]}

    search = AlbumsOnly()
    source = _episode(tmp_path / "『番組名』第1話「始まり」.wav", [(105, 10.0, 30.0)], seconds=90.0)
    result = ra.discover(source, source_sha256="s" * 64, cache_dir=_cache(state), state_root=state, search=search,
                         backend=backend, pacing=NO_WAIT, courtesy=_polite())
    assert result["state"] == "verified" and result["title"] == "番組名" and result["basis"] == "file_name"
    assert result["releases"] == ["https://music.example.test/browse/B"]
    assert result["albums"]["https://music.example.test/browse/B"]["acquired"] == 8
    used = {rec["sha256"] for rec in rs.records(state, work_key=ra.work_key("番組名"))}
    wrong_shas = {rs._file_sha256(p) for p in wrong.values()}
    assert len(used) == 8 and not used & wrong_shas
    assert all(rs.load(sha, state)["media"]["state"] == rs.STATE_RETIRED for sha in wrong_shas)
    queries = list(search.calls)
    again = ra.discover(source, source_sha256="s" * 64, cache_dir=_cache(state), state_root=state, search=search,
                        backend=backend, pacing=NO_WAIT, courtesy=_polite())
    assert again["reused"] is True and search.calls == queries


def test_discovery_without_a_verified_release_adds_nothing_and_never_blocks(state, tmp_path):
    backend = FakeBackend({"w0": _track(tmp_path / "remote" / "w0.flac", 300)})
    catalog = Catalog(songs={"番組名": ["w0"]})
    source = tmp_path / "『番組名』第2話.wav"
    sf.write(source, _music(42, 30.0), LOW, subtype="PCM_16")
    result = ra.discover(source, source_sha256="t" * 64, cache_dir=_cache(state), state_root=state, search=catalog,
                         backend=backend, pacing=NO_WAIT, courtesy=_polite())
    assert result["state"] == "no_verified_release"
    assert rs.records(state, work_key=ra.work_key("番組名")) == []

    class Broken:
        def songs(self, query):
            raise OSError("network unreachable")

    result = ra.discover(tmp_path / "『別の番組』第1話.wav", source_sha256="u" * 64, cache_dir=_cache(state),
                         state_root=state, search=Broken(), courtesy=_polite(),
                         query_samples=np.zeros(LOW, np.float32))
    assert result["state"] == "failed" and result["error"] == "OSError"
    assert ra.discover(tmp_path / "x.wav", source_sha256=None, cache_dir=_cache(state),
                       state_root=state)["state"] in ("no_work_hint", "failed")


def test_a_rate_limit_defers_discovery_and_the_next_chance_resumes_without_redoing_work(state, tmp_path):
    seeds = {"k1": 11, "k2": 12, "k3": 13, "k4": 14}
    media = {i: _track(tmp_path / "remote" / f"{i}.flac", seed) for i, seed in seeds.items()}
    backend = AlbumBackend(media, {i: "盤" for i in seeds},
                           failures={"k3": [FailureKind.RATE_LIMITED] * 3})
    catalog = Catalog(songs={"番組名": ["k2"]}, albums={"盤": ["K"]},
                      listings={"K": ("盤", ["k1", "k2", "k3", "k4"])})
    source = _episode(tmp_path / "『番組名』第3話.wav", [(12, 5.0, 30.0)])
    clock = Clock()
    polite = _polite(state, clock)

    first = ra.discover(source, source_sha256="d" * 64, cache_dir=_cache(state), state_root=state, search=catalog,
                        backend=backend, pacing=NO_WAIT, courtesy=polite, now=clock.now)
    assert first["state"] == "deferred" and first["deferred"]["seconds"] >= 590
    assert backend.downloads == ["k2", "k1"]  # 混雑の合図の後は、休みの間に次の取得へ進まない
    saved = json.loads((state / cm.STATE_FILE).read_text(encoding="utf-8"))
    assert saved["hosts"]["example.test"]["last_signal"]["kind"] == cm.RATE_LIMITED

    # 休みは再起動（別の調整役）でも守る。通信せず保留のまま。
    calls = list(catalog.calls)
    again = ra.discover(source, source_sha256="d" * 64, cache_dir=_cache(state), state_root=state, search=catalog,
                        backend=backend, pacing=NO_WAIT, courtesy=_polite(state, clock), now=clock.now)
    assert again["state"] == "deferred" and backend.downloads == ["k2", "k1"] and catalog.calls == calls

    # 休みの後：取得済みの曲は取り直さず、残りだけ取得して終える。
    clock.now += 3600
    done = ra.discover(source, source_sha256="d" * 64, cache_dir=_cache(state), state_root=state, search=catalog,
                       backend=backend, pacing=NO_WAIT, courtesy=_polite(state, clock), now=clock.now)
    assert done["state"] == "verified"
    assert backend.downloads == ["k2", "k1", "k3", "k4"]
    assert _linked(state, "番組名") == {"k1", "k2", "k3", "k4"}


def test_courtesy_spaces_requests_backs_off_and_honours_retry_after(tmp_path):
    clock = Clock()
    polite = cm.Courtesy(tmp_path, cm.Policy(interval=(4.0, 4.0), cooldown_start=600.0, max_wait=120.0),
                         clock=clock, sleep=clock.sleep)
    url = "https://music.youtube.com/watch?v=abc"
    starts = []
    for _ in range(3):
        with polite.slot(url):
            starts.append(clock.now)
        polite.report(url)
    assert [round(b - a, 3) for a, b in zip(starts, starts[1:])] == [4.0, 4.0]
    assert cm.host_key("https://www.youtube.com/x") == cm.host_key(url) == "youtube"

    # サーバの指示（Retry-After）が休みの初期値より長ければ、それより早く戻らない。
    polite.report(url, cm.RATE_LIMITED, retry_after="1800")
    assert 1790 <= polite.cooldown_remaining(url) <= 1800
    with pytest.raises(cm.Deferred):
        with polite.slot(url):
            pass
    # 別の起動（同じ状態領域）も同じ休みを守る。
    other = cm.Courtesy(tmp_path, cm.Policy(interval=(4.0, 4.0)), clock=clock, sleep=clock.sleep)
    assert other.cooldown_remaining("youtube") > 1700
    clock.now += 1801
    with other.slot(url):
        gap_after = clock.now
    # 混雑の後は間隔を広げる（倍率2）。成功が続けば戻す。
    other.report(url)
    with other.slot(url):
        pass
    assert round(clock.now - gap_after, 3) == 8.0
    for _ in range(10):
        other.report(url)
    assert json.loads((tmp_path / cm.STATE_FILE).read_text(encoding="utf-8"))["hosts"]["youtube"]["factor"] == 1.0


def test_retry_after_values_are_read_as_seconds_or_http_dates():
    assert cm.retry_after_seconds("120") == 120.0
    assert cm.retry_after_seconds("HTTP Error 429: Too Many Requests; Retry-After: 30") == 30.0
    assert cm.retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT", now=1445412480.0 - 60) == 60.0
    assert cm.retry_after_seconds("soon") is None and cm.retry_after_seconds(None) is None


def test_fingerprint_only_verification_needs_votes_windows_and_a_steady_alignment():
    class M:
        def __init__(self, votes, windows, drift):
            self.votes, self.measured_windows, self.max_window_drift_seconds = votes, windows, drift

    assert ra.fingerprint_consistent(M(47, 3, 0.02))  # Synthetic stable support near the production minimum (40).
    assert not ra.fingerprint_consistent(M(18, 1, 0.02))  # 合成例の弱い支持
    assert not ra.fingerprint_consistent(M(400, 9, 1.3))   # 揃わない（ずれが揺れる）
    assert not ra.fingerprint_consistent(M(60, 2, 0.0))


def test_reference_acquisition_uses_the_same_login_choice_as_url_acquisition(state, tmp_path, monkeypatch):
    """参照の取得も通常のURLの取得と同じ：LSSのログインプロファイルでログイン済みならそのCookie（会員向けの盤）。
    認証の種類を参照の取得の側で固定しない（取得の層 `service.acquire` が選ぶ）。"""
    from lss.acquisition import service

    seen = {}

    def acquire(url, **options):
        seen.update(options)
        return []

    monkeypatch.setattr(service, "acquire", acquire)
    ra.acquire_into_store("https://music.youtube.com/watch?v=x", cache_dir=_cache(state), state_root=state)
    assert "auth" not in seen          # service.acquire が login_profile.auth_for で選ぶ
    assert seen["pacing"].attempts == 2


def test_an_album_that_needs_a_paid_or_member_login_is_not_requested_track_by_track(state, tmp_path):
    """有料会員・会員限定の盤（実例：YouTube Music の一部のアニメのサウンドトラック）。資格情報は使わず、
    認証を求められた曲が続いた盤は残りの曲を取りに行かない。発見は見つからないで終わり、処理は止めない。"""
    ids = [f"p{n}" for n in range(8)]
    media = {i: _track(tmp_path / "remote" / f"{i}.flac", 400 + n) for n, i in enumerate(ids)}
    backend = FakeBackend(media, failures={i: [FailureKind.AUTH_REQUIRED] for i in ids})

    class AlbumsOnly:
        def albums(self, query):
            return [{"url": "https://music.example.test/browse/P"}]

        def tracks(self, url):
            return {"title": "番組名 オリジナル・サウンドトラック", "url": url,
                    "tracks": [{"url": f"https://example.test/{i}", "id": i, "title": i} for i in ids]}

    source = _episode(tmp_path / "『番組名』第1話.wav", [(42, 10.0, 30.0)])
    result = ra.discover(source, source_sha256="r" * 64, cache_dir=_cache(state), state_root=state,
                         search=AlbumsOnly(), backend=backend, pacing=NO_WAIT, courtesy=_polite())
    assert result["state"] == "no_verified_release"
    album = result["albums"]["https://music.example.test/browse/P"]
    assert album["state"] == "restricted" and album["restricted_failures"] == ra.RESTRICTED_ALBUM_FAILURES
    attempted = [i for i in ids if not backend.failures[i]]
    assert len(attempted) == ra.RESTRICTED_ALBUM_FAILURES      # 残りの6曲は要求しない
    assert all(result["candidates"][i]["state"] == "restricted" for i in attempted)
    assert backend.downloads == []


@pytest.mark.parametrize(
    ("top_restricted", "runner_up_score", "full_downloads"),
    [
        pytest.param(True, 1.0, 1, id="restricted_top_uses_positive_runner_up"),
        pytest.param(False, 1.0, 2, id="unrestricted_top_keeps_runner_up_sample_only"),
        pytest.param(True, 0.0, 0, id="restricted_top_skips_zero_score_albums"),
    ],
)
def test_full_verification_uses_the_top_available_album(
    state, tmp_path, monkeypatch, top_restricted, runner_up_score, full_downloads,
):
    # 各盤の最初と最後を見本にして、一致する曲は次点の盤の真ん中に置く。
    monkeypatch.setattr(ra, "SAMPLE_TRACKS", 2)
    monkeypatch.setattr(ra, "theme_songs", lambda *_: [])
    top_ids = ["a1", "a2", "a3", "a4"]
    runner_up_ids = ["b1", "b2", "b3"]
    seeds = {track: seed for seed, track in enumerate(top_ids + runner_up_ids, start=400)}
    seeds["b2"] = 42
    media = {track: _track(tmp_path / "remote" / f"{track}.flac", seed) for track, seed in seeds.items()}
    failures = {track: [FailureKind.AUTH_REQUIRED] for track in top_ids} if top_restricted else {}
    backend = FakeBackend(media, failures=failures)
    runner_up_title = "別の作品 サウンドトラック" if runner_up_score else "別の作品の曲集"
    catalog = Catalog(albums={"番組名 オリジナル・サウンドトラック": ["A", "B"]},
                      listings={"A": ("番組名 オリジナル・サウンドトラック", top_ids),
                                "B": (runner_up_title, runner_up_ids)})
    source = _episode(tmp_path / "『番組名』第1話.wav", [(seeds["b2"], 10.0, 30.0)])

    result = ra.discover(source, source_sha256="r" * 64, cache_dir=_cache(state), state_root=state,
                         search=catalog, backend=backend, pacing=NO_WAIT, courtesy=_polite())

    top = result["albums"]["https://music.example.test/browse/A"]
    runner_up = result["albums"]["https://music.example.test/browse/B"]
    assert top["album_rank"] == 0 and (top["state"] == "restricted") == top_restricted
    assert top.get("full_verification", False) is (not top_restricted)
    assert runner_up["album_rank"] == 1 and runner_up["score"] == runner_up_score
    assert result["counters"]["full_verification_downloads"] == full_downloads
    if top_restricted and runner_up_score > 0:
        assert runner_up["full_verification"] is True
        assert runner_up["full_verification_reason"] == "top_restricted"
        assert result["verified_tracks"] == ["b2"]
    else:
        assert "full_verification" not in runner_up
        assert "full_verification_reason" not in runner_up
        assert "b2" not in backend.downloads
        assert result["state"] == "no_verified_release"


def test_full_verification_stops_when_the_album_becomes_restricted_even_with_a_shared_track(
    state, tmp_path, monkeypatch,
):
    # 見本の取得枠は最上位の盤で使い切る。共通の曲は別の盤の見本にも残る。
    monkeypatch.setattr(ra, "SAMPLE_TRACKS", 2)
    monkeypatch.setattr(ra, "VERIFY_BUDGET", 2)
    monkeypatch.setattr(ra, "theme_songs", lambda *_: [])
    top_ids = ["a1", "a2", "a3", "shared", "a5"]
    catalog = Catalog(albums={"番組名 オリジナル・サウンドトラック": ["A", "B"]},
                      listings={"A": ("番組名 オリジナル・サウンドトラック", top_ids),
                                "B": ("別の作品の曲集", ["shared"])})
    backend = FakeBackend({}, failures={"a1": [FailureKind.UNAVAILABLE], "a5": [FailureKind.UNAVAILABLE],
                                        "a2": [FailureKind.AUTH_REQUIRED], "a3": [FailureKind.AUTH_REQUIRED],
                                        "shared": [FailureKind.UNAVAILABLE]})
    source = tmp_path / "『番組名』第1話.wav"

    result = ra.discover(source, source_sha256="r" * 64, cache_dir=_cache(state), state_root=state,
                         search=catalog, backend=backend, pacing=NO_WAIT, courtesy=_polite(),
                         query_samples=np.zeros(LOW, np.float32))

    top_url = "https://music.example.test/browse/A"
    other_url = "https://music.example.test/browse/B"
    top = result["albums"][top_url]
    assert top["full_verification"] is True and top["state"] == "restricted"
    assert result["albums"][other_url]["state"] == "candidate"
    assert result["counters"]["full_verification_downloads"] == 2
    shared = result["candidates"]["shared"]
    assert shared["state"] == "pending"
    assert {(origin["kind"], origin["album_url"]) for origin in shared["origins"]} == {
        ("album_sample", other_url), ("album_full", top_url),
    }
    assert backend.failures["shared"] == [FailureKind.UNAVAILABLE]


def test_search_refresh_drops_the_rank_of_the_old_restricted_top(state, tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "theme_songs", lambda *_: [])
    source = tmp_path / "『番組名』第1話.wav"
    source_sha256 = "r" * 64
    old_url = "https://music.example.test/browse/A"
    hint = ra.work_hint(source, source_sha256, state)
    saved = ra._new_opportunity(hint, None, source_sha256, 1000.0)
    saved.update(state="no_verified_release", searched_at_epoch=1000.0, theme_songs_searched_at_epoch=1000.0,
                 albums={old_url: dict(url=old_url, album_rank=0, score=3.0, state="restricted")})
    ra._save_opportunity(saved, state)
    catalog = Catalog(albums={"番組名 オリジナル・サウンドトラック": ["B", "C"]},
                      listings={"B": ("別の作品 サウンドトラック", ["b1"]),
                                "C": ("別の作品 サウンドトラック", ["c1"])})
    backend = FakeBackend({}, failures={"b1": [FailureKind.UNAVAILABLE], "c1": [FailureKind.UNAVAILABLE]})

    result = ra.discover(source, source_sha256=source_sha256, cache_dir=_cache(state), state_root=state,
                         search=catalog, backend=backend, pacing=NO_WAIT, courtesy=_polite(),
                         query_samples=np.zeros(LOW, np.float32), now=1000.0 + ra.RETRY_AFTER_SECONDS)

    assert result["albums"][old_url]["state"] == "restricted"
    assert "album_rank" not in result["albums"][old_url]
    top = result["albums"]["https://music.example.test/browse/B"]
    assert top["album_rank"] == 0 and top["full_verification"] is True
    assert "full_verification_reason" not in top
    runner_up = result["albums"]["https://music.example.test/browse/C"]
    assert runner_up["album_rank"] == 1 and "full_verification" not in runner_up


def test_a_members_album_skipped_without_login_is_tried_again_after_logging_in(state, tmp_path):
    """ログインが無い時に会員向けとして打ち切った盤は、LSSでログインした後の機会に取り直す（早期の打ち切りを残さない）。"""
    from lss.acquisition.store import AuthMethod

    ids = [f"m{n}" for n in range(6)]
    media = {i: _track(tmp_path / "remote" / f"{i}.flac", 500 + n) for n, i in enumerate(ids)}
    backend = FakeBackend(media, failures={i: [FailureKind.AUTH_REQUIRED] for i in ids[:2]})

    class AlbumsOnly:
        def albums(self, query):
            return [{"url": "https://music.example.test/browse/M"}]

        def tracks(self, url):
            return {"title": "番組名 オリジナル・サウンドトラック", "url": url,
                    "tracks": [{"url": f"https://example.test/{i}", "id": i, "title": i} for i in ids]}

    source = _episode(tmp_path / "『番組名』第1話.wav", [(503, 10.0, 30.0)])
    first = ra.discover(source, source_sha256="v" * 64, cache_dir=_cache(state), state_root=state,
                        search=AlbumsOnly(), backend=backend, pacing=NO_WAIT, courtesy=_polite(), now=1000.0)
    assert first["state"] == "no_verified_release" and first["auth_method"] == "none"
    assert first["albums"]["https://music.example.test/browse/M"]["state"] == "restricted"
    assert backend.downloads == []

    # LSSでログインした（取得の認証が app_login になった）：打ち切った盤をやり直し、番組の曲を確かめて取得する。
    login = AuthMethod.app_login(tmp_path / "profile")
    again = ra.discover(source, source_sha256="v" * 64, cache_dir=_cache(state), state_root=state,
                        search=AlbumsOnly(), backend=backend, pacing=NO_WAIT, courtesy=_polite(), now=2000.0,
                        auth=login)
    assert again["auth_method"] == "app_login" and again["reopened_after_login"] >= 2
    assert again["state"] == "verified" and again["verified_tracks"] == ["m3"]
    assert sorted(backend.downloads) == sorted(ids)


def test_a_network_failure_while_acquiring_a_candidate_is_retried_at_the_next_chance(state, tmp_path):
    """接続の切断・通信の失敗は一時的。取れなかった候補を確かめずに打ち切らない：次の機会に取り直して確かめる
    （打ち切ると「見つからない」が7日ごとの再検索でも続き、作品の参照が二度と見つからない）。"""
    media = {"k2": _track(tmp_path / "remote" / "k2.flac", 12)}
    # NO_WAIT は3回まで試す：この機会の間は通信が戻らない。
    backend = FakeBackend(media, failures={"k2": [FailureKind.NETWORK] * 3})
    catalog = Catalog(songs={"番組名": ["k2"]})
    source = _episode(tmp_path / "『番組名』第4話.wav", [(12, 5.0, 30.0)])

    first = ra.discover(source, source_sha256="n" * 64, cache_dir=_cache(state), state_root=state, search=catalog,
                        backend=backend, pacing=NO_WAIT, courtesy=_polite(), now=1000.0)
    assert first["state"] == "no_verified_release" and backend.downloads == []
    assert first["candidates"]["k2"]["failure"] == FailureKind.NETWORK.value
    assert first["candidates"]["k2"]["state"] == "pending"

    # 一時間後（検索の答えの期限内）：通信は戻った。同じ候補を取り直して確かめる。
    again = ra.discover(source, source_sha256="n" * 64, cache_dir=_cache(state), state_root=state, search=catalog,
                        backend=backend, pacing=NO_WAIT, courtesy=_polite(), now=4600.0)
    assert backend.downloads == ["k2"]
    assert again["verified_tracks"] == ["k2"]


def test_a_candidate_left_failed_by_an_earlier_network_outage_is_reopened(state, tmp_path):
    """前の版が通信の失敗で `acquire_failed` にした候補（状態領域に残る）も、次の機会に取り直す。"""
    media = {"k2": _track(tmp_path / "remote" / "k2.flac", 12)}
    backend = FakeBackend(media)
    catalog = Catalog(songs={"番組名": ["k2"]})
    source = _episode(tmp_path / "『番組名』第5話.wav", [(12, 5.0, 30.0)])
    hint = ra.work_hint(source, "o" * 64, state)
    stale = ra._new_opportunity(hint, None, "o" * 64, 1000.0)
    stale.update(state="no_verified_release", searched_at_epoch=1000.0, candidates={"k2": {
        "id": "k2", "url": "https://example.test/k2", "title": "曲 k2", "state": "acquire_failed",
        "failure": FailureKind.NETWORK.value, "checks": {},
        "origins": [{"kind": "song_search", "query": "番組名", "rank": 0}]}})
    ra._save_opportunity(stale, state)

    again = ra.discover(source, source_sha256="o" * 64, cache_dir=_cache(state), state_root=state, search=catalog,
                        backend=backend, pacing=NO_WAIT, courtesy=_polite(), now=4600.0)
    assert backend.downloads == ["k2"]
    assert again["verified_tracks"] == ["k2"]


# --- 取得ツールの導入・失敗の記録・ログイン待ち ---------------------------------------------------

class ToolSearch:
    """目録の偽物：導入（`installed_by["done"]`）の前は取得ツールが無い失敗を返す。"""

    def __init__(self, installed_by, error=None):
        self.installed_by = installed_by
        self.error = error
        self.calls = []

    def _gate(self, kind, query):
        self.calls.append((kind, query))
        if self.error is not None:
            raise self.error
        if not self.installed_by["done"]:
            raise AcquisitionFailure(FailureKind.TOOL_UNAVAILABLE, "検証済みのyt-dlpが導入されていません")

    def songs(self, query):
        self._gate("songs", query)
        return []   # 候補なし（試験が実際の取得へ出ない）

    def albums(self, query):
        self._gate("albums", query)
        return []

    def tracks(self, url):
        self._gate("tracks", url)
        return {"title": "t", "id": url, "url": url, "tracks": []}


def _show(tmp_path, name="『番組名』第1話.wav"):
    source = tmp_path / name
    sf.write(source, _music(42, 20.0), LOW, subtype="PCM_16")
    return source


SAMPLES = np.zeros(LOW, np.float32)


def _find(state, source, search, ensure=None, **kwargs):
    return ra.discover(source, source_sha256="e" * 64, cache_dir=_cache(state), state_root=state,
                       courtesy=_polite(), ensure_provider=ensure, search=search, query_samples=SAMPLES, **kwargs)


def test_a_packaged_app_without_the_tool_installs_it_before_the_first_catalog_fetch(state, tmp_path, monkeypatch):
    """導入済みのyt-dlpが無い配備版：検索は取得より先に来るので、発見が自分で検証付きの導入を行う。"""
    from lss.acquisition import provider, service

    installed = {"done": False}
    ensured = []

    def ensure(*, after_failure=False):
        ensured.append(after_failure)
        installed["done"] = True
        return {"status": "installed"}

    def missing():
        raise AcquisitionFailure(FailureKind.TOOL_UNAVAILABLE, "検証済みのyt-dlpが導入されていません")

    monkeypatch.setattr(provider, "active", missing)
    monkeypatch.setattr(service, "_ensure_provider", ensure)
    search = ToolSearch(installed)
    monkeypatch.setattr(ra, "SearchBackend", lambda backend=None: search)
    result = ra.discover(_show(tmp_path), source_sha256="e" * 64, cache_dir=_cache(state), state_root=state,
                         courtesy=_polite(), query_samples=SAMPLES)
    assert len(ensured) == 1                       # 導入は1回だけ（問い合わせの前）
    assert result["catalog"]["requests"] > 0 and search.calls[0][0] == "songs"
    assert result["state"] != "failed" and "failure_kind" not in result
    assert result["attempts"] == 1 and result["state"] == "no_verified_release"


def test_a_tool_unavailable_failure_installs_once_and_retries_the_same_fetch(state, tmp_path):
    installed = {"done": False}
    ensured = []

    def ensure(*, after_failure=False):
        ensured.append(after_failure)
        installed["done"] = True
        return {"status": "installed"}

    search = ToolSearch(installed)
    result = _find(state, _show(tmp_path), search, ensure)
    assert ensured == [True]
    assert search.calls[:2] == [("songs", "番組名"), ("songs", "番組名")]   # 失敗した問い合わせを1回やり直す
    assert result["catalog"]["requests"] > 0 and result["provider_ensure"] == "installed"
    assert result["state"] != "failed"


def test_when_the_tool_cannot_be_installed_discovery_is_deferred_without_using_an_attempt(state, tmp_path):
    """導入できない（署名の確認の失敗・通信なし等）：失敗の種類を残し、回数を数えず、次の機会に再試行する。"""
    ensured = []

    def ensure(*, after_failure=False):
        ensured.append(after_failure)
        return None

    source = _show(tmp_path)
    first = _find(state, source, ToolSearch({"done": False}), ensure)
    assert first["state"] == "deferred" and first["failure_kind"] == "tool_unavailable"
    assert first["retryable"] is True and "yt-dlp" in first["failure_detail"]
    assert first["attempts"] == 0 and first["catalog"]["requests"] == 0
    assert first["searched_at_epoch"] is None and ensured == [True]
    saved = ra.load_opportunity(first["work_key"], state)
    assert saved["state"] == "deferred" and saved["failure_kind"] == "tool_unavailable"

    # 次の機会：導入されていれば、作品の検索を続きから行う（回数は1から）。
    second = _find(state, source, ToolSearch({"done": True}), lambda **kw: {"status": "installed"})
    assert second["state"] != "deferred" and second["attempts"] == 1 and second["catalog"]["requests"] > 0


def test_a_network_failure_is_deferred_and_a_real_empty_answer_is_not(state, tmp_path):
    source = _show(tmp_path)
    down = ToolSearch({"done": True}, error=AcquisitionFailure(FailureKind.NETWORK, "getaddrinfo failed"))
    first = _find(state, source, down)
    assert first["state"] == "deferred" and first["failure_kind"] == "network" and first["attempts"] == 0

    second = _find(state, source, Catalog())
    assert second["state"] == "no_verified_release" and second["attempts"] == 1
    assert second["searched_at_epoch"] is not None


def test_an_unclassified_failure_is_still_failed_with_its_detail_recorded(state, tmp_path):
    class Broken:
        def songs(self, query):
            raise AcquisitionFailure(FailureKind.EXTRACTOR_FAILED, "unexpected  extractor\nerror")

    result = _find(state, _show(tmp_path), Broken())
    assert result["state"] == "failed" and result["error"] == "AcquisitionFailure"
    assert result["failure_kind"] == "extractor_failed" and result["failure_detail"] == "unexpected extractor error"
    assert result["attempts"] == 1


def test_music_premium_only_responses_are_needs_login_not_a_discovery_failure(state, tmp_path):
    from lss.acquisition.store import AuthMethod

    premium = AcquisitionFailure(FailureKind.AUTH_REQUIRED, "This video is only available to Music Premium members")

    class PremiumAlbum:
        def songs(self, query):
            return []

        def albums(self, query):
            return [{"url": "https://music.example.test/browse/P"}]

        def tracks(self, url):
            raise premium

    source = _show(tmp_path)
    url = "https://music.example.test/browse/P"
    result = _find(state, source, PremiumAlbum(), now=1000.0)
    album = result["albums"][url]
    assert album["state"] == "needs_login" and album["needs_login"] is True and album["failure"] == "auth_required"
    assert "Premium" in album["failure_detail"]
    assert result["state"] == "no_verified_release" and result["needs_login"] is True
    assert result["auth_method"] == "none"

    # ログインした後の機会：盤を検索からやり直す。
    class Open(PremiumAlbum):
        def tracks(self, url):
            return {"title": "番組名 オリジナル・サウンドトラック", "url": url, "tracks": []}

    again = _find(state, source, Open(), now=2000.0, auth=AuthMethod.app_login(tmp_path / "profile"))
    assert again["reopened_after_login"] >= 1
    assert again["albums"][url]["state"] != "needs_login" and again["needs_login"] is False


def test_a_restricted_track_album_is_reported_as_needing_login(state, tmp_path):
    ids = [f"q{n}" for n in range(4)]
    media = {i: _track(tmp_path / "remote" / f"{i}.flac", 600 + n) for n, i in enumerate(ids)}
    backend = FakeBackend(media, failures={i: [FailureKind.AUTH_REQUIRED] for i in ids})

    class AlbumsOnly:
        def albums(self, query):
            return [{"url": "https://music.example.test/browse/Q"}]

        def tracks(self, url):
            return {"title": "番組名 オリジナル・サウンドトラック", "url": url,
                    "tracks": [{"url": f"https://example.test/{i}", "id": i, "title": i} for i in ids]}

    result = ra.discover(_episode(tmp_path / "『番組名』第1話.wav", [(42, 10.0, 30.0)]), source_sha256="q" * 64,
                         cache_dir=_cache(state), state_root=state, search=AlbumsOnly(), backend=backend,
                         pacing=NO_WAIT, courtesy=_polite())
    assert result["state"] == "no_verified_release" and result["needs_login"] is True
    assert result["albums"]["https://music.example.test/browse/Q"]["needs_login"] is True


def test_a_theme_song_listed_in_the_work_article_is_searched_and_verified_by_audio(state, tmp_path, monkeypatch):
    """作品名の検索にも盤にも出ないED：作品の記事の主題歌から探し、番組の音声と合った時だけ結び付ける。"""
    seeds = {"x9": 31, "e1": 41, "e2": 42, "n1": 51}
    media = {i: _track(tmp_path / "remote" / f"{i}.flac", seed) for i, seed in seeds.items()}
    backend = AlbumBackend(media, {"e1": "主題歌X / B面", "e2": "主題歌X / B面"})
    catalog = Catalog(songs={"番組名": ["x9"], "主題歌X 歌手": ["e1"], "主題歌Y 歌手": ["n1"]},
                      albums={"主題歌X / B面": ["E"]}, listings={"E": ("主題歌X / B面", ["e1", "e2"])})
    monkeypatch.setattr(ra, "theme_songs", lambda source, state_root: [
        {"title": "主題歌X", "kind": "エンディングテーマ", "query": "主題歌X 歌手"},
        {"title": "主題歌Y", "kind": "オープニングテーマ", "query": "主題歌Y 歌手"}])
    source = _episode(tmp_path / "『番組名』第10話「合宿」.wav", [(41, 50.0, 30.0)])

    result = ra.discover(source, source_sha256="a" * 64, cache_dir=_cache(state), state_root=state,
                         search=catalog, backend=backend, pacing=NO_WAIT, courtesy=_polite())

    assert ("songs", "主題歌X 歌手") in catalog.calls
    assert result["verified_tracks"] == ["e1"]
    assert result["releases"] == ["https://music.example.test/browse/E"]
    # 名前で探した曲でも、番組で鳴っていない主題歌（n1）は結び付けない。
    assert _linked(state, "番組名") == {"e1", "e2"}
    assert rs.load(rs._file_sha256(media["n1"]), state)["work_keys"] == []


def test_a_verified_theme_song_does_not_stop_sampling_the_soundtrack_album(state, tmp_path, monkeypatch):
    """主題歌が確かめられても、劇伴の盤（題に作品名の無いサウンドトラック）の見本は確かめる。

    主題歌が鳴っていることは、劇伴の既発売の盤があるかを何も言わない。主題歌から辿った盤の別の曲
    （TV size 等）が鳴っていても同じ。前の版がその見本を `not_needed` で閉じた記録も、劇伴がまだ
    確かめられていなければ次の機会に確かめ直す。
    """
    seeds = {"x9": 31, "e1": 41, "e2": 42, "b1": 61, "b2": 62, "b3": 63}
    media = {i: _track(tmp_path / "remote" / f"{i}.flac", seed) for i, seed in seeds.items()}
    backend = AlbumBackend(media, {"e1": "主題歌X / B面", "e2": "主題歌X / B面",
                                   "b1": "Show Original Soundtrack", "b2": "Show Original Soundtrack",
                                   "b3": "Show Original Soundtrack"})
    catalog = Catalog(songs={"番組名": ["x9"], "主題歌X 歌手": ["e1"]},
                      albums={"番組名 オリジナル・サウンドトラック": ["S"], "主題歌X / B面": ["E"]},
                      listings={"S": ("Show Original Soundtrack", ["b1", "b2", "b3"]),
                                "E": ("主題歌X / B面", ["e1", "e2"])})
    monkeypatch.setattr(ra, "theme_songs", lambda source, state_root: [
        {"title": "主題歌X", "kind": "エンディングテーマ", "query": "主題歌X 歌手"}])
    source = _episode(tmp_path / "『番組名』第10話「合宿」.wav", [(62, 5.0, 30.0), (41, 45.0, 30.0), (42, 85.0, 30.0)])
    kwargs = dict(cache_dir=_cache(state), state_root=state, search=catalog, backend=backend, pacing=NO_WAIT,
                  courtesy=_polite())

    result = ra.discover(source, source_sha256="a" * 64, **kwargs)

    assert {"e1", "b2"} <= set(result["verified_tracks"])
    assert "https://music.example.test/browse/S" in result["releases"]
    assert {"b1", "b2", "b3"} <= _linked(state, "番組名")

    # 前の版の記録：主題歌だけが確かめられ、劇伴の盤の見本は not_needed で閉じられている。
    path = ra._opportunity_path(ra.work_key("番組名"), state)
    saved = json.loads(path.read_text(encoding="utf-8"))
    for key in ("b1", "b2", "b3"):
        saved["candidates"][key].update(state="not_needed", checks={})
        saved["candidates"][key].pop("sha256", None)
    saved["releases"] = [url for url in saved["releases"] if not url.endswith("/S")]
    saved["albums"]["https://music.example.test/browse/S"].pop("state", None)
    path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    again = ra.discover(source, source_sha256="b" * 64, **kwargs)
    assert again["candidates"]["b2"]["state"] == "verified"


def test_an_opportunity_searched_before_theme_songs_existed_searches_them_once(state, tmp_path, monkeypatch):
    seeds = {"x9": 31, "e1": 41}
    media = {i: _track(tmp_path / "remote" / f"{i}.flac", seed) for i, seed in seeds.items()}
    backend = AlbumBackend(media, {})
    catalog = Catalog(songs={"番組名": ["x9"], "主題歌X 歌手": ["e1"]})
    themes = []
    monkeypatch.setattr(ra, "theme_songs", lambda source, state_root: list(themes))
    source = _episode(tmp_path / "『番組名』第10話「合宿」.wav", [(41, 50.0, 30.0)])
    clock = Clock()
    kwargs = dict(cache_dir=_cache(state), state_root=state, search=catalog, backend=backend, pacing=NO_WAIT,
                  courtesy=_polite(clock=clock))
    ra.discover(source, source_sha256="a" * 64, now=clock(), **kwargs)
    # 前の版の記録：主題歌を探した印が無い。
    path = ra._opportunity_path(ra.work_key("番組名"), state)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved.pop("theme_songs_searched_at_epoch")
    path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    themes.append({"title": "主題歌X", "kind": "エンディングテーマ", "query": "主題歌X 歌手"})

    result = ra.discover(source, source_sha256="b" * 64, now=clock() + 60, **kwargs)
    assert "e1" in result["verified_tracks"]
    assert catalog.calls.count(("songs", "番組名")) == 1          # 作品名の検索は期限内なので繰り返さない
    again = ra.discover(source, source_sha256="c" * 64, now=clock() + 120, **kwargs)
    assert again["theme_songs_searched_at_epoch"] == clock() + 60   # 主題歌も期限内は探し直さない
    assert again["state"] in ("verified", "verified_partial")


def test_theme_songs_come_from_the_work_article_newest_first_with_the_singer(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from lss.metadata.model import Credit
    from lss.metadata.sources import wikipedia

    def sung(name):
        return (Credit(role="作曲", role_key="composer", name="作曲者", names=("作曲者",)),
                Credit(role="歌", role_key="vocals", name=name, names=(name,)))

    songs = ({"title": "最初のOP", "kind": "オープニングテーマ", "credits": sung("A")},
             {"title": "挿入の歌", "kind": "挿入歌", "credits": sung("B")},
             {"title": "前のED", "kind": "エンディングテーマ", "credits": sung("C")},
             {"title": "今のED", "kind": "エンディングテーマ", "credits": sung("CUTIE")})
    asked = []

    def series(work, subtitle, kind, state_root, **kwargs):
        asked.append((work, kwargs.get("episode_number")))
        return SimpleNamespace(songs=songs)

    monkeypatch.setattr(wikipedia, "series", series)
    found = ra.theme_songs(tmp_path / "【公式】アニメ「星空冒険団」第145話「驚愕！」.mkv", tmp_path)
    assert [s["query"] for s in found] == ["最初のOP A", "今のED CUTIE", "前のED C"]
    assert asked == [("星空冒険団", 145)]
    monkeypatch.setattr(wikipedia, "series", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert ra.theme_songs(tmp_path / "『番組名』第1話.wav", tmp_path) == []
