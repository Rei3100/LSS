"""An over-long model answer for one cue is an unusable answer for that cue, never a job-wide crash."""
from __future__ import annotations

import pytest

from lss.bgm import scene_job, scene_names
from tests.test_bgm_scene_job import FakeModels, _cue, library  # noqa: F401 - shared fixtures


class RamblingModels(FakeModels):
    def order(self, prompt):
        self.calls.append(("order", prompt))
        if len([c for c in self.calls if c[0] == "order"]) == 1:
            # Greedy decoding repeats this answer on every retry: a 201-character scene description.
            return {"name_basis": "scene", "use": "ヒ" * 201, "direction": None, "category": None,
                    "name": "ミズウオ大作戦", "evidence": "ミズウオだ"}
        return super().order(prompt)


def test_overlong_order_field_is_recorded_as_an_unusable_answer_and_other_cues_continue(tmp_path, library):  # noqa: F811
    state = tmp_path / "state"
    library["s1"] += [_cue(tmp_path / "ep137", "s1", 400, tone=330), _cue(tmp_path / "ep137", "s1", 700, tone=262)]
    models = RamblingModels(["ミズウオ大作戦"])
    result = scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models)
    first = scene_names.cue_record(state, library["s1"][0]["asset_id"])
    second = scene_names.cue_record(state, library["s1"][1]["asset_id"])
    assert first is not None and first["name"] is None       # attempted, recorded, retryable; no crash
    assert second is not None and second["name"] == "ミズウオ大作戦"
    assert result["named"] == 1
    assert ("release",) in models.calls
