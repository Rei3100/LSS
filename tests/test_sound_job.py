"""欲しい音ジョブ: 公開・再利用・検証した「無し」の記録・読めない言葉（分離器とタグ付けは偽物）。"""
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.sound import audioset, intent, interpret, job, plan

RATE = 44100
_REAL_INTERPRETER = interpret.Interpreter


def _source(path: Path, seconds: int = 6, silent: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(seconds * RATE) / RATE
    tone = np.zeros_like(t) if silent else 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), np.stack([tone, tone], axis=1).astype(np.float32), RATE, subtype="PCM_24")
    return path


class _Workspace:
    """経路の節を偽の音で返す。``present`` に入る経路の節だけ音がある。"""

    instances: list = []

    def __init__(self, source, folder, *, progress=None, device="cuda:0", checkpoint=None):
        self.source_audio, self.rate = sf.read(str(source), dtype="float32", always_2d=True)
        self.prepared: list = []
        _Workspace.instances.append(self)

    def audio(self, node):
        if node == plan.SOURCE:
            return self.source_audio
        text = plan.canonical(node)
        if any(word in text for word in _Workspace.absent):
            return np.zeros_like(self.source_audio)
        if any(word in text for word in _Workspace.present):
            return self.source_audio * 0.5
        return np.zeros_like(self.source_audio)

    def prepare(self, expressions):
        self.prepared.extend(expressions)

    def frame_probabilities(self, items):
        """言葉の時刻の証拠: 音がある節で、聞こえる言葉（``words``）だけ高い。"""
        out = {}
        for node, prompt in items:
            audio = self.audio(node)
            frames = int(np.ceil(len(audio) / self.rate * 25))
            heard = float(np.abs(audio).max()) > 0 and prompt in _Workspace.words
            out[(plan.canonical(node), prompt)] = np.full(frames, 0.9 if heard else 0.01, dtype=np.float32)
        return out

    def release_text_engine(self):
        pass


class _Tagger:
    def __init__(self):
        self.closed = False

    def seconds(self, audio, rate):
        count = int(np.ceil(len(audio) / rate))
        tags = np.zeros((count, 527), dtype=np.float32)
        if float(np.abs(audio).max()) > 0:
            for name in _Workspace.heard:
                tags[:, audioset.INDEX[name]] = 0.5
        return tags

    def close(self):
        self.closed = True


class _NoLease:
    def __init__(self, report):
        self.acquired = 0

    def __enter__(self):
        return self

    def ensure(self):
        self.acquired += 1

    def __exit__(self, *exc):
        return False


class _Reader:
    """訳の偽物: ``known`` にある日本語だけを英語の音として読む。"""

    known: dict = {}
    calls: list = []
    unavailable = None

    def __init__(self, **kwargs):
        pass

    def __call__(self, japanese):
        from lss.sound import intent

        _Reader.calls.append(japanese)
        if japanese not in _Reader.known:
            return None
        phrase = _Reader.known[japanese]
        concept = intent._resolve(phrase, open_vocabulary=True)
        if concept.id.startswith("open:"):
            from dataclasses import replace
            concept = replace(concept, label=japanese)
        return concept, phrase

    def close(self):
        pass


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    _Workspace.instances = []
    _Workspace.present = ("\"rain\"",)
    _Workspace.heard = ("Rain", "Music")
    _Workspace.words = ("rain",)
    _Workspace.absent = ()
    _Reader.known, _Reader.calls = {}, []
    from lss.bgm import event_tagging
    from lss.sound import extract, interpret, verify

    monkeypatch.setattr(job, "engines_revision", lambda: {"fake": 1})
    monkeypatch.setattr(job, "_Lease", _NoLease)
    monkeypatch.setattr(interpret, "Interpreter", _Reader)
    monkeypatch.setattr(extract, "Workspace", _Workspace)
    monkeypatch.setattr(event_tagging, "ensure_weights", lambda progress=None: Path("weights"))
    monkeypatch.setattr(verify.Tagger, "load", classmethod(lambda cls, weights, device="cuda:0": _Tagger()))
    return tmp_path


def _run(source: Path, sounds: list[str]) -> dict:
    return job.handle_wanted_sound_job({"cmd": "wanted_sound", "mix_path": str(source), "sounds": sounds})


def test_publishes_present_sounds_records_absence_and_names_unknown_words(fake: Path) -> None:
    source = _source(fake / "music" / "song.wav")
    result = _run(source, ["雨の音", "ピアノ", "かめはめ波"])
    assert result["status"] == "ok", result
    outcome = result["sound"]
    assert outcome["k"] == "done" and outcome["n"] == 1
    assert outcome["files"] == ["song - 雨音.flac"]
    by_text = {item["text"]: item for item in outcome["items"]}
    assert by_text["雨の音"]["k"] == "done"
    assert by_text["ピアノ"]["k"] == "none" and by_text["ピアノ"]["decision"]["present"] is False
    assert by_text["かめはめ波"]["k"] == "unsupported" and by_text["かめはめ波"]["unknown"] == ["かめはめ波"]
    published = source.parent / "LSS" / "song - 雨音.flac"
    assert published.is_file()
    from mutagen.flac import FLAC

    tags = FLAC(published)
    assert tags["LSS_STEM"] == ["雨音"] and tags["LSS_RECIPE_ID"][0].startswith("wanted-sound-v1-")
    assert result["product_result"]["state"] in {"PUBLISHED", "REUSED"}


def test_a_second_run_reuses_published_files_and_verified_absence_without_gpu(fake: Path) -> None:
    source = _source(fake / "music" / "song.wav")
    _run(source, ["雨の音", "ピアノ"])
    created = len(_Workspace.instances)
    again = _run(source, ["雨の音", "ピアノ"])
    assert len(_Workspace.instances) == created
    kinds = {item["text"]: item["k"] for item in again["sound"]["items"]}
    assert kinds == {"雨の音": "reused", "ピアノ": "none"}
    assert again["sound"]["k"] == "reused" and again["sound"]["n"] == 1


def test_a_removed_published_file_is_made_again(fake: Path) -> None:
    source = _source(fake / "music" / "song.wav")
    _run(source, ["雨の音"])
    (source.parent / "LSS" / "song - 雨音.flac").unlink()
    again = _run(source, ["雨の音"])
    assert again["sound"]["items"][0]["k"] == "done"
    assert (source.parent / "LSS" / "song - 雨音.flac").is_file()


def test_exact_silence_is_a_validated_absence(fake: Path) -> None:
    source = _source(fake / "music" / "quiet.wav", silent=True)
    result = _run(source, ["雨の音"])
    assert result["sound"]["k"] == "none" and result["sound"]["items"][0]["reason"] == "exact_digital_silence"
    assert not _Workspace.instances


def test_only_unknown_words_is_unsupported_not_failed(fake: Path) -> None:
    source = _source(fake / "music" / "song.wav")
    result = _run(source, ["かめはめ波"])
    assert result["status"] == "ok" and result["sound"]["k"] == "unsupported"
    assert not _Workspace.instances


@pytest.mark.parametrize("payload", [
    {"cmd": "wanted_sound", "mix_path": "x.wav", "sounds": []},
    {"cmd": "wanted_sound", "mix_path": "x.wav", "sounds": ["a"] * 13},
    {"cmd": "wanted_sound", "sounds": ["雨"]},
])
def test_malformed_jobs_are_refused(fake: Path, payload: dict) -> None:
    result = job.handle_wanted_sound_job(payload)
    assert result["status"] == "error" and result["error_code"] == "invalid_job"


def test_heard_in_the_parent_but_not_extracted_is_held_not_absent(fake: Path) -> None:
    """親には聞こえるのに取り出した音に無い: 「無い」と記録せず保留にし、次の実行で取り直す。"""
    source = _source(fake / "music" / "song.wav")
    _Workspace.present = ("\"stems\"",)   # 親（SCNetのその他）には音がある
    _Workspace.absent = ('"node":"text"',)   # 言葉の分離の出力は無音
    _Workspace.words = ("koto",)             # 専門modelの無い楽器（言葉で取る）
    result = _run(source, ["琴"])
    item = result["sound"]["items"][0]
    assert item["k"] == "held" and item["reason"] == "heard_but_not_extracted"
    assert result["sound"]["k"] == "held" and result["sound"]["n"] == 0
    assert result["product_result"]["state"] == "HELD" and result["product_result"]["held_count"] == 1
    again = _run(source, ["琴"])
    assert again["sound"]["items"][0]["k"] == "held"   # 無いとは記録されていないので、もう一度試す


def test_japanese_outside_the_lexicon_is_read_through_a_checked_translation(fake: Path) -> None:
    """語彙に無い日本語: 確かめられた訳だけを英語の音として読み、成果物の名前は利用者の言葉のまま。"""
    source = _source(fake / "music" / "song.wav")
    _Reader.known = {"雨が窓に当たる音": "rain hitting the window"}
    _Workspace.present = ("rain hitting the window",)
    _Workspace.words = ("rain hitting the window",)
    result = _run(source, ["雨が窓に当たる音", "かめはめ波"])
    by_text = {item["text"]: item for item in result["sound"]["items"]}
    assert by_text["雨が窓に当たる音"]["k"] == "done"
    assert by_text["雨が窓に当たる音"]["heard_as"] == ["rain hitting the window"]
    assert result["sound"]["files"] == ["song - 雨が窓に当たる音.flac"]
    assert by_text["かめはめ波"]["k"] == "unsupported" and by_text["かめはめ波"]["unknown"] == ["かめはめ波"]
    assert "かめはめ波" in _Reader.calls   # 訳を試した上で、確かめられなかった


def test_open_words_choose_the_parent_where_the_sound_is_heard(fake: Path) -> None:
    """語彙に無い言葉は、候補の親のうち、その音が最も聞こえる親から言葉で取り出す。"""
    source = _source(fake / "music" / "song.wav")
    _Workspace.present = ("\"drums\"", "squeaky rubber toy")
    _Workspace.words = ("squeaky rubber toy",)
    result = _run(source, ["squeaky rubber toy"])
    item = result["sound"]["items"][0]
    assert item["k"] == "done"
    assert '"stem":"drums"' in item["route"] and '"prompt":"squeaky rubber toy"' in item["route"]
    parents = item["decision"]["parents"]["open:squeaky rubber toy"]
    assert max(parents, key=parents.get).find('"drums"') >= 0


def test_when_the_reader_cannot_be_prepared_lexicon_words_still_work(fake: Path, monkeypatch) -> None:
    """訳のmodelを用意できない（offlineの初回等）: 語彙の言葉は処理し、訳が要る言葉だけ理由を付けて返す。"""
    class _Offline(_REAL_INTERPRETER):
        def _load_translator(self):
            raise OSError("network unavailable")

    monkeypatch.setattr(interpret, "Interpreter", _Offline)
    source = _source(fake / "music" / "song.wav")
    result = _run(source, ["雨の音", "雨が窓に当たる音"])
    by_text = {item["text"]: item for item in result["sound"]["items"]}
    assert result["status"] == "ok" and by_text["雨の音"]["k"] == "done"
    assert by_text["雨が窓に当たる音"]["k"] == "unsupported"
    assert by_text["雨が窓に当たる音"]["reason"] == "reader_unavailable"


def test_a_time_range_publishes_only_its_time_and_is_its_own_recipe(fake: Path) -> None:
    """「前半の雨の音」: 範囲の外は静か。範囲の無い「雨の音」とは別の成果物・別のRecipe。"""
    source = _source(fake / "music" / "song.wav", seconds=8)
    result = _run(source, ["前半の雨の音", "雨の音"])
    first, whole = result["sound"]["items"]
    assert first["k"] == "done" and whole["k"] == "done"
    assert [Path(name).name for name in first["files"]] == ["song - 前半の雨音.flac"]
    windowed, _ = sf.read(first["files"][0], dtype="float32", always_2d=True)
    plain, _ = sf.read(whole["files"][0], dtype="float32", always_2d=True)
    assert np.abs(windowed[: 3 * RATE]).max() > 0.05 and np.abs(windowed[5 * RATE:]).max() == 0.0
    assert np.abs(plain[5 * RATE:]).max() > 0.05
    assert first["decision"]["kept_seconds"] <= 5 < whole["decision"]["kept_seconds"]


def test_an_instrument_is_decided_by_the_words_heard_in_its_parent(fake: Path) -> None:
    """楽器: 出力は親（その他）の一部なので、有無は親での言葉の証拠で決める。親で聞こえなければ無い。"""
    source = _source(fake / "music" / "song.wav")
    _Workspace.present = ("\"stems\"", "koto")
    _Workspace.heard = ()                       # 出力のタグには何も無くても
    _Workspace.words = ("koto",)                # 親で言葉が聞こえれば、ある
    result = _run(source, ["琴"])               # 専門modelの無い楽器（言葉で取る）
    item = result["sound"]["items"][0]
    assert item["k"] == "done" and item["decision"]["reasons"] == ["target_words_heard_in_parent"]
    assert '"instrument":"koto"' in item["route"]
    other = _source(fake / "music" / "other.wav", seconds=5)
    _Workspace.words = ()                       # 親で聞こえなければ、無い（保留ではない）
    absent = _run(other, ["琴"])["sound"]["items"][0]
    assert absent["k"] == "none" and absent["reason"] == "verified_absent"


def test_absent_removal_preserves_source_and_reuses_instead_of_zero(fake: Path) -> None:
    source = _source(fake / "music" / "song.wav")
    # A separator returns audible, unrelated material for an absent target.
    _Workspace.words = ()
    _Workspace.heard = ("Music",)
    result = _run(source, ["雨なし"])
    item = result["sound"]["items"][0]
    assert item["k"] == "done", result
    assert item["decision"]["removal"]["outcome"] == "unchanged"
    assert item["removal_status"] == "unchanged"
    original, rate = sf.read(source, dtype="float32", always_2d=True)
    output, _ = sf.read(item["files"][0], dtype="float32", always_2d=True)
    np.testing.assert_array_equal(output, original)
    from mutagen.flac import FLAC

    tags = FLAC(item["files"][0])
    tags["TITLE"] = "My retained title"
    tags.save()
    before = Path(item["files"][0]).read_bytes()
    count = len(_Workspace.instances)
    again = _run(source, ["雨なし"])
    assert again["sound"]["items"][0]["k"] == "reused"
    assert again["sound"]["items"][0]["removal_status"] == "unchanged"
    assert len(_Workspace.instances) == count
    assert Path(item["files"][0]).read_bytes() == before


def test_timed_removal_keeps_every_sample_outside_requested_window(fake: Path) -> None:
    source = _source(fake / "music" / "song.wav", seconds=8)
    result = _run(source, ["前半の雨なし"])
    item = result["sound"]["items"][0]
    assert item["k"] == "done", result
    original, _ = sf.read(source, dtype="float32", always_2d=True)
    output, _ = sf.read(item["files"][0], dtype="float32", always_2d=True)
    np.testing.assert_array_equal(output[4 * RATE:], original[4 * RATE:])
    np.testing.assert_allclose(output[RATE:3 * RATE], original[RATE:3 * RATE] * 0.5, atol=2e-7)
    assert item["decision"]["removal"]["outcome"] == "removed"


def test_removal_conflict_is_held_and_retried(fake: Path) -> None:
    source = _source(fake / "music" / "song.wav")
    _Workspace.heard = ("Music",)  # text model alone cannot authorize removal
    result = _run(source, ["雨なし"])
    assert result["sound"]["items"][0]["k"] == "held", result
    assert not list((source.parent / "LSS").glob("*.flac"))
    count = len(_Workspace.instances)
    assert _run(source, ["雨なし"])["sound"]["items"][0]["k"] == "held"
    assert len(_Workspace.instances) > count


def test_union_removal_does_not_let_one_target_authorize_another(fake: Path) -> None:
    source = _source(fake / "music" / "song.wav")
    _Workspace.present = ('"rain"', '"knocking"')
    result = _run(source, ["雨とノックなし"])
    item = result["sound"]["items"][0]
    assert item["k"] == "done", result
    targets = item["decision"]["removal"]["targets"]
    assert [entry["present"] for entry in targets] == [True, False]


def test_overlapping_text_removal_is_held(fake: Path) -> None:
    source = _source(fake / "music" / "song.wav")
    _Workspace.present = ('"rain"', '"thunder"')
    _Workspace.words = ("rain", "thunder")
    _Workspace.heard = ("Rain", "Thunder")
    result = _run(source, ["雨と雷なし"])
    item = result["sound"]["items"][0]
    assert item["k"] == "held", result
    assert item["decision"]["targets"][-1]["reasons"][-1] == "overlapping_removal_targets"


def test_unrelated_dominant_event_cannot_authorize_removal(fake: Path, monkeypatch) -> None:
    source = _source(fake / "music" / "train.wav")
    _Workspace.present = ('"thunder"',)
    _Workspace.words = ("thunder",)  # even the text judge falsely agrees
    original = _Tagger.seconds

    def train_tags(self, audio, rate):
        tags = original(self, audio, rate)
        tags[:, audioset.INDEX["Thunder"]] = 0.05
        tags[:, audioset.INDEX["Train"]] = 0.8
        return tags

    monkeypatch.setattr(_Tagger, "seconds", train_tags)
    item = _run(source, ["雷なし"])["sound"]["items"][0]
    assert item["k"] == "held", item


def test_removal_rivals_exclude_target_ancestors_and_children() -> None:
    from lss.sound import verify

    rivals = {audioset.LABELS[index] for index in verify.removal_rivals(("Dog",))}
    assert "Train" in rivals and "Thunder" in rivals
    assert not ({"Animal", "Domestic animals, pets", "Bark", "Dog"} & rivals)
    assert all("instrument" not in node for node in plan.nodes(
        plan.removal_targets(intent.parse_one("ピアノなし"), "music")[0].expression))
