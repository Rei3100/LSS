"""Partial final seconds are evidence and audio, not an implicit absent tail."""
import numpy as np
import pytest
import soundfile as sf

from lss.bgm import cue_publication, cue_representation as cr, cue_semantics as cs, episode_cues as ec
from tests.test_bgm_episode_cues import _recipe
from tests.test_application_bgm_candidates import gpu

RATE = 44100


def _signals(tmp_path, frames):
    t = np.arange(frames) / RATE
    music = np.column_stack((.2 * np.sin(2 * np.pi * 220 * t), .1 * np.sin(2 * np.pi * 330 * t)))
    paths = {}
    for name in ("music", "speech", "effects"):
        paths[name] = tmp_path / f"{name}.wav"
        sf.write(paths[name], music if name == "music" else np.zeros_like(music), RATE, subtype="FLOAT")
    return paths, music


@pytest.mark.parametrize("frames", [13 * RATE, 13 * RATE + 1, 13 * RATE + RATE * 3 // 4])
def test_levels_and_residuals_cover_exactly_the_observed_seconds(tmp_path, frames):
    paths, _ = _signals(tmp_path, frames)
    shares, levels, _ = ec.window_shares(paths, 1.)
    residual = cr.residual_levels(str(paths["music"]), str(paths["music"]))
    expected = (frames + RATE - 1) // RATE
    assert len(shares) == len(levels) == len(residual) == expected
    assert np.isfinite(levels).all() and np.isfinite(residual).all()
    np.testing.assert_array_equal(shares, np.ones(expected))


def test_planning_segmentation_and_publication_keep_the_native_tail(tmp_path):
    frames = 13 * RATE + RATE * 3 // 4 + 1
    paths, music = _signals(tmp_path, frames)
    duration = frames / RATE
    plan = ec.plan_episode_cues(paths, _recipe(smoothing_windows=1))
    assert plan["source_seconds"] == duration
    [candidate] = plan["cues"]
    assert candidate["end_seconds"] == duration
    _, levels, _ = ec.window_shares(paths, 1.)
    p = np.zeros((len(levels), 527), np.float32)
    p[:, cs.MUSIC_CLASS] = .9
    evidence = cs.WindowEvidence.from_arrays(levels, p, np.ones((len(levels), 8)))
    [segment] = cs.segment([(candidate["start_seconds"], candidate["end_seconds"])], evidence, cs.default_recipe())
    assert segment["end_seconds"] == segment["core_seconds"][1] == duration
    result = cue_publication.publish(music[:round(segment["end_seconds"] * RATE)], RATE, tmp_path / "cue.flac",
        key="tail", title="tail", album="test", provenance={"type": "bgm_published_cue"},
        input_sha256="a" * 64, recipe_id="tail-test", version="development")
    assert result["frames"] == sf.info(result["path"]).frames == frames
    written, _ = sf.read(result["path"], always_2d=True)
    np.testing.assert_allclose(written[-RATE:], music[-RATE:], atol=2. ** -23)


def test_partial_tail_energy_is_not_diluted_by_padding(tmp_path):
    paths, _ = _signals(tmp_path, RATE // 4)
    _, levels, _ = ec.window_shares(paths, 1.)
    data, _ = sf.read(paths["music"], always_2d=True)
    assert levels.shape == (1,)
    assert levels[0] == pytest.approx(10 * np.log10(np.mean(data ** 2)), abs=1e-5)


def test_partial_second_does_not_invent_minimum_short_cue_duration(tmp_path):
    paths, _ = _signals(tmp_path, RATE + RATE // 4)
    recipe = _recipe(smoothing_windows=1, short_seconds=6., short_dominant_share=.9,
                     short_dominant_seconds=2.)
    assert ec.plan_episode_cues(paths, recipe)["cues"] == []


def test_partial_window_presence_uses_actual_time_not_window_count():
    recipe = _recipe(short_seconds=2., short_present_fraction=.8)
    assert ec._complete_short(0, 3, np.array([1., 1., 0.]), np.array([-20., -20., -120.]), recipe,
                              start_seconds=0., end_seconds=2.1)


def test_silent_partial_tail_is_observed_but_not_added_to_the_music(tmp_path):
    frames = 13 * RATE + RATE * 3 // 4
    paths, _ = _signals(tmp_path, frames)
    data, _ = sf.read(paths["music"], always_2d=True)
    data[13 * RATE:] = 0.
    sf.write(paths["music"], data, RATE, subtype="FLOAT")
    plan = ec.plan_episode_cues(paths, _recipe(smoothing_windows=1))
    assert plan["source_seconds"] == frames / RATE
    [candidate] = plan["cues"]
    assert candidate["end_seconds"] == 13.
    _, levels, _ = ec.window_shares(paths, 1.)
    p = np.zeros((len(levels), 527), np.float32)
    p[:, cs.MUSIC_CLASS] = .9  # Contextual classification alone cannot extend into actual silence.
    evidence = cs.WindowEvidence.from_arrays(levels, p, np.ones((len(levels), 8)))
    [segment] = cs.segment([(0., candidate["end_seconds"])], evidence, cs.default_recipe())
    assert len(levels) == 14 and segment["end_seconds"] == 13.


@pytest.mark.parametrize("frames", [RATE // 4, RATE, RATE + 1])
def test_tagger_adapter_emits_a_partial_window_without_model_fallback(tmp_path, monkeypatch, frames):
    import torch
    from lss.bgm import event_tagging as et

    paths, _ = _signals(tmp_path, frames)
    observed = []

    class Model:
        def __call__(self, chunks):
            observed.append(chunks.shape[0])
            return (torch.full((len(chunks), et.CLASS_COUNT), .5),
                    torch.ones((len(chunks), et.EMBEDDING_SIZE)))

    # Only the learned forward pass is replaced; production load_model remains CUDA-only.
    monkeypatch.setattr(et, "load_model", lambda *a, **k: Model())
    p, emb = et.tag_signal(paths["music"], tmp_path / "unused", device="cpu")
    expected = (frames + RATE - 1) // RATE
    assert p.shape == (expected, et.CLASS_COUNT) and emb.shape == (expected, et.EMBEDDING_SIZE)
    assert sum(observed) == expected
    assert observed == ([1, 1] if frames == RATE + 1 else [1])


def test_normal_bgm_job_saves_the_partial_tail_and_reuses_it(tmp_path, monkeypatch, gpu):
    from lss import application as app
    from lss.bgm import event_tagging as et

    frames = 13 * RATE + RATE * 3 // 4 + 1
    paths, _ = _signals(tmp_path, frames)

    def tags(path, weights, **kwargs):
        info = sf.info(path)
        count = (info.frames + info.samplerate - 1) // info.samplerate
        p = np.zeros((count, et.CLASS_COUNT), np.float16)
        p[:, cs.MUSIC_CLASS] = .9
        return p, np.ones((count, et.EMBEDDING_SIZE), np.float16)

    monkeypatch.setattr(et, "tag_signal", tags)
    request = app.ProcessingRequest.from_paths([paths["music"]], app.UserIntent.EXTRACT_BGM_CUES)
    first = app.run_bgm_candidate_request(request, state_root=tmp_path / "state")
    assert first.disposition is app.ResultDisposition.PUBLISHED, first.message
    assert len(first.outputs) == 1
    assert sf.info(first.outputs[0]).frames == frames
    before = first.outputs[0].read_bytes()
    second = app.run_bgm_candidate_request(request, state_root=tmp_path / "state")
    assert second.outputs == first.outputs and second.reused
    assert second.outputs[0].read_bytes() == before
