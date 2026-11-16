"""候補保存後・再利用の最終段階の変更を、番組完成記録へ取り込まない。"""
import json
from pathlib import Path

import pytest

from lss.validation import episode_discovery as episode
from tests.test_episode_discovery_recovery import _setup


@pytest.mark.parametrize("changed", ["recipe", "signal", "first_candidate", "intent", "separation"])
def test_change_after_last_crop_prevents_completion(tmp_path, monkeypatch, changed):
    _, recipe, run_dir, _, run = _setup(tmp_path)
    materialize = episode.materialize_candidate
    assets = []

    def modify_after_last_save(*args, **kwargs):
        asset = materialize(*args, **kwargs)
        assets.append(asset)
        if len(assets) == 2:
            path = {"recipe": recipe, "signal": run_dir / "artifacts" / "signal.flac",
                    "first_candidate": Path(assets[0]["path"]), "intent": run_dir / "intent.json",
                    "separation": run_dir / "separation.json"}[changed]
            # 内容だけの変化でも、完成時に元の入力・成果物へ対応すると扱わない。
            with path.open("ab") as stream:
                stream.write(b" ")
        return asset

    monkeypatch.setattr(episode, "materialize_candidate", modify_after_last_save)
    if changed == "intent":
        # JSONとして有効な空白だけでは構成は変わらない。意味の変化を注入する。
        def mutate_intent(*args, **kwargs):
            asset = modify_after_last_save(*args, **kwargs)
            if len(assets) == 2:
                path = run_dir / "intent.json"
                value = json.loads(path.read_bytes())
                value["source_id"] = "different episode"
                path.write_text(json.dumps(value))
            return asset
        monkeypatch.setattr(episode, "materialize_candidate", mutate_intent)
    with pytest.raises(episode.EpisodeDiscoveryError, match="変更"):
        run()
    assert not (run_dir / "run.json").exists()


def test_change_immediately_after_activity_record_publication_is_detected(tmp_path, monkeypatch):
    _, _, run_dir, _, run = _setup(tmp_path)
    write_json = episode._write_json

    def alter_record(path, value):
        digest = write_json(path, value)
        if path.parent == run_dir / "activity":
            value = {**value, "source_id": "another episode"}
            write_json(path, value)
        return digest

    monkeypatch.setattr(episode, "_write_json", alter_record)
    with pytest.raises(episode.EpisodeDiscoveryError, match="候補記録"):
        run()
    assert not (run_dir / "run.json").exists()


def test_recipe_change_while_reusing_finished_audio_preserves_prior_run(tmp_path, monkeypatch):
    _, recipe, run_dir, calls, run = _setup(tmp_path)
    path = run()
    before = path.read_bytes()
    original = episode._sha256
    candidates = list((run_dir / "candidates").rglob("*.flac"))
    changed = False

    def alter_recipe_after_validation(path):
        nonlocal changed
        result = original(path)
        if Path(path) == candidates[-1] and not changed:
            changed = True
            with recipe.open("ab") as stream:
                stream.write(b" ")
        return result

    monkeypatch.setattr(episode, "_sha256", alter_recipe_after_validation)
    with pytest.raises(episode.EpisodeDiscoveryError, match="Recipe"):
        run(resume=True)
    assert path.read_bytes() == before and len(calls) == 1


def test_file_change_after_import_cannot_be_labelled_as_loaded_new_code(tmp_path, monkeypatch):
    _, _, run_dir, calls, run = _setup(tmp_path)
    original = episode._sha256

    def changed_hash(path):
        if Path(path) == Path(episode.__file__):
            return "0" * 64
        return original(path)

    monkeypatch.setattr(episode, "_sha256", changed_hash)
    with pytest.raises(episode.EpisodeDiscoveryError, match="読み込み済み"):
        run()
    assert calls == [] and not run_dir.exists()


def test_loaded_recipe_is_bound_to_the_bytes_read_before_loading(tmp_path, monkeypatch):
    _, recipe, run_dir, calls, run = _setup(tmp_path)
    load = episode.load_discovery_recipe

    def alter_after_loading(path):
        result = load(path)
        with recipe.open("ab") as stream:
            stream.write(b" ")
        return result

    monkeypatch.setattr(episode, "load_discovery_recipe", alter_after_loading)
    with pytest.raises(episode.EpisodeDiscoveryError, match="読み込み中"):
        run()
    assert calls == [] and not run_dir.exists()


def test_validated_zero_also_requires_unchanged_recipe(tmp_path):
    import numpy as np
    import soundfile as sf
    from lss.core import session
    from lss.core.presence import inspect_signal

    source, recipe, run_dir, _, _ = _setup(tmp_path)
    sf.write(source, np.zeros((4000, 2)), 1000, subtype="PCM_24")
    zero = inspect_signal(source).zero_result()

    def separate(*args, **kwargs):
        with recipe.open("ab") as stream:
            stream.write(b" ")
        return {"status": "ok", "outputs": [], "zero_result": zero,
                "input_sha256": session.input_sha256_of(source), "recipe_id": "silent-r1", "runtime_id": "runtime-r1"}

    with pytest.raises(episode.EpisodeDiscoveryError, match="Recipe"):
        episode.run_episode_discovery(source, [recipe], run_dir, source_id="episode", model_id="model", separate=separate)
    assert not (run_dir / "run.json").exists()
