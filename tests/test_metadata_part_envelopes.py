import numpy as np
import soundfile as sf

from lss.metadata import part_envelopes as pe

RATE = 8000


def _tone(seconds, amp, freq=220.0, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t))[:, None] * np.ones((1, 2))


def test_frame_db_relative_to_mix_and_silence_floor():
    x = _tone(2.0, 0.5)
    mix = pe.mix_rms(x)
    env = pe.frame_db(x, mix, RATE)
    assert env.size == 8 and np.allclose(env, 0.0, atol=0.05)
    half = pe.frame_db(x * 0.1, mix, RATE)
    assert np.allclose(half, -20.0, atol=0.05)
    assert np.all(pe.frame_db(np.zeros_like(x), mix, RATE) == pe.FLOOR_DB)
    assert np.all(pe.frame_db(x, 0.0, RATE) == pe.FLOOR_DB)


def test_frame_db_partial_last_frame_counts():
    x = _tone(1.1, 0.5)
    assert pe.frame_db(x, pe.mix_rms(x), RATE).size == 5


class Stub:
    stem_sha16 = "aaaaaaaaaaaaaaaa"
    lead_sha16 = "bbbbbbbbbbbbbbbb"

    def __init__(self, mix, vocals_gain):
        self.mix, self.vg = mix, vocals_gain
        self.stem_calls = 0
        self.lead_calls = 0

    def stems(self, path):
        self.stem_calls += 1
        z = np.zeros_like(self.mix)
        return {"drums": self.mix * 0.5, "bass": self.mix * 0.001, "other": z, "vocals": self.mix * self.vg}, RATE

    def lead(self, path):
        self.lead_calls += 1
        return self.mix * 0.2, RATE


def _cue(tmp_path, seconds=4.0):
    mix = _tone(seconds, 0.4)
    path = tmp_path / "cue.flac"
    sf.write(str(path), mix, RATE, format="FLAC")
    return path, sf.read(str(path), dtype="float64", always_2d=True)[0]


def test_compute_envelopes_and_cache(tmp_path):
    path, mix = _cue(tmp_path)
    sep = Stub(mix, 0.0)
    env = pe.compute(path, tmp_path / "state", sep)
    assert set(env) == {"DRUMS", "BASS", "VOCALS", "OTHER"}  # no lead: vocals silent
    assert np.allclose(env["DRUMS"], -6.0, atol=0.1)
    assert np.allclose(env["BASS"], -60.0, atol=0.2)
    assert np.all(env["VOCALS"] == pe.FLOOR_DB) and np.all(env["OTHER"] == pe.FLOOR_DB)
    assert sep.stem_calls == 1 and sep.lead_calls == 0
    again = pe.compute(path, tmp_path / "state", sep)
    assert sep.stem_calls == 1  # cache hit
    assert all(np.allclose(again[k], env[k]) for k in env)
    files = list((tmp_path / "state" / "metadata-library" / "part-envelopes").glob("*.npz"))
    assert len(files) == 1 and files[0].name.endswith("-aaaaaaaaaaaaaaaa-0.25.npz")


def test_lead_only_when_vocals_active_and_cached(tmp_path):
    path, mix = _cue(tmp_path)
    sep = Stub(mix, 0.5)
    env = pe.compute(path, tmp_path / "state", sep)
    assert pe.LEAD in env and np.allclose(env[pe.LEAD], -14.0, atol=0.1)
    assert sep.lead_calls == 1
    pe.compute(path, tmp_path / "state", sep)
    assert sep.stem_calls == 1 and sep.lead_calls == 1


def test_cache_key_changes_with_model_and_hop_and_pcm(tmp_path):
    path, mix = _cue(tmp_path)
    s1, s2 = Stub(mix, 0.0), Stub(mix, 0.0)
    s2.stem_sha16 = "cccccccccccccccc"
    pe.compute(path, tmp_path / "st", s1)
    pe.compute(path, tmp_path / "st", s2)
    pe.compute(path, tmp_path / "st", s1, hop_seconds=0.5)
    assert s1.stem_calls == 2 and s2.stem_calls == 1
    other = tmp_path / "other.flac"
    sf.write(str(other), mix * 0.5, RATE, format="FLAC")
    pe.compute(other, tmp_path / "st", s1)
    assert s1.stem_calls == 3
    assert len(list((tmp_path / "st" / "metadata-library" / "part-envelopes").glob("*.npz"))) == 4


def test_corrupt_cache_is_recomputed(tmp_path):
    path, mix = _cue(tmp_path)
    sep = Stub(mix, 0.0)
    pe.compute(path, tmp_path / "st", sep)
    for f in (tmp_path / "st" / "metadata-library" / "part-envelopes").glob("*.npz"):
        f.write_bytes(b"junk")
    pe.compute(path, tmp_path / "st", sep)
    assert sep.stem_calls == 2


def test_other_is_a_witness_never_tagged():
    n = 200
    loud, quiet = np.full(n, -6.0), np.full(n, -60.0)
    tags, ev = pe.decide_tags({"OTHER": loud, "DRUMS": quiet, "BASS": quiet, "VOCALS": quiet})
    assert tags == {} and ev["only"]["part"] == "OTHER"
    tags, _ = pe.decide_tags({"VOCALS": loud, "DRUMS": quiet, "BASS": quiet, "OTHER": quiet})
    assert tags == {"VOCALS_VER": "Only"}
