import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import soundfile as sf

from lss.harness import fetch


@pytest.fixture
def control_script():
    path = Path(__file__).resolve().parents[1] / "scripts/validate_instrumental_controls.py"
    spec = importlib.util.spec_from_file_location("instrumental_controls_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("extension", [False, True])
def test_fixed_linear_cases_keep_stereo_and_common_gain(tmp_path, monkeypatch, control_script, extension):
    archive = tmp_path / "sample.zip"
    tracks = control_script.EXTENSION_TRACKS if extension else control_script.TRACKS
    with zipfile.ZipFile(archive, "w") as stored:
        for _, title in tracks:
            stored.writestr(f"train/{title}.stem.mp4", b"container")
        stored.writestr("test/unused.stem.mp4", b"must not decode")
        stored.writestr("../outside.txt", b"must not extract")
    monkeypatch.setattr(control_script, "ARCHIVE_SHA256", fetch.file_sha256(archive))
    decoded_streams = []
    values = {
        1: [0.5, -0.2], 2: [0.75, 0.1], 3: [0.1, 0.3], 4: [1.0, 0.7],
    }

    def decode(command, **kwargs):
        stream_id = int(command[command.index("-map") + 1].rsplit(":", 1)[-1])
        decoded_streams.append(stream_id)
        sf.write(command[-1], np.tile(values[stream_id], (100, 1)), 44100, subtype="FLOAT")

    monkeypatch.setattr(control_script.subprocess, "run", decode)
    plan = control_script.build(archive, tmp_path / "controls", extension=extension)
    data = json.loads(plan.read_text(encoding="utf-8"))
    assert decoded_streams == [1, 2, 3, 4] * 3
    assert len(data["cases"]) == 12 and data["promotion_allowed"] is False
    assert [source["title"] for source in data["sources"]] == [title for _, title in tracks]
    assert data["case_group"] == ("additional_development_tracks" if extension else "initial_development_tracks")
    assert all(source["common_gain"] == 0.125 for source in data["sources"])
    for case in data["cases"]:
        arrays = {role: sf.read(record["path"], always_2d=True)[0]
                  for role, record in case["artifacts"].items()}
        assert arrays["input"].shape == (100, 2)
        np.testing.assert_allclose(arrays["input"], arrays["target"] + arrays["vocals"], atol=3e-8)
        if case["variant"] == "vocals-only":
            assert np.count_nonzero(arrays["target"]) == 0
        elif case["variant"] == "quiet-accompaniment":
            np.testing.assert_allclose(arrays["target"][0], [0.016875, 0.0025], atol=1e-8)
        elif case["variant"] == "accompaniment-only":
            assert np.count_nonzero(arrays["vocals"]) == 0
            np.testing.assert_allclose(arrays["target"][0], [0.16875, 0.025], atol=1e-8)
    assert not (tmp_path / "outside.txt").exists()
    assert not list((tmp_path / "controls").rglob("unused*"))


def test_changed_archive_is_refused_before_creating_corpus(tmp_path, control_script):
    archive = tmp_path / "unknown.zip"
    archive.write_bytes(b"changed archive")
    output = tmp_path / "controls"
    with pytest.raises(ValueError, match="一致"):
        control_script.build(archive, output)
    assert not output.exists()


def test_changed_fixed_input_is_refused_before_inference(tmp_path, monkeypatch, control_script):
    source = tmp_path / "source.wav"
    source.write_bytes(b"changed audio")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "role": "development_diagnostic_only", "promotion_allowed": False,
        "cases": [{"artifacts": {"input": {"path": str(source), "sha256": "0" * 64}}}],
    }), encoding="utf-8")
    monkeypatch.setattr(control_script.worker, "handle_job", lambda *a, **kw: pytest.fail("固定入力検証が先"))
    with pytest.raises(ValueError, match="変更"):
        control_script.evaluate(plan)
    assert not (tmp_path / "models.tsv").exists()
