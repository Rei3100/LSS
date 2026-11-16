"""A state directory reached through a directory junction is still this state (tiered storage).

On the user's PC ``<state>/bgm-candidates`` is a junction to another volume, so a CurrentBest row's ``run_path`` is
spelled either through the state or through the junction target; both are this state's. A row naming another
state's ``bgm-candidates`` is still elsewhere.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lss.bgm import write_targets

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="directory junctions (mklink /J) exist on Windows only")

RUN = Path("a" * 64) / "run" / "run.json"


def _junction(link: Path, target: Path) -> None:
    target.mkdir(parents=True)
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
    if done.returncode != 0:
        pytest.skip(f"mklink /J unavailable: {done.stdout!r}{done.stderr!r}")


def _row(candidates: Path) -> dict:
    return {"run_path": str(candidates / RUN)}


def test_rows_through_the_state_or_the_junction_target_are_not_elsewhere(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    target = tmp_path / "tier" / "product-state" / "bgm-candidates"
    _junction(state / "bgm-candidates", target)
    assert not write_targets.adopted_elsewhere(state, _row(state / "bgm-candidates"))
    assert not write_targets.adopted_elsewhere(state, _row(target))
    assert not write_targets.adopted_elsewhere(state, _row(Path(str(state).upper()) / "bgm-candidates"))
    assert not write_targets.adopted_elsewhere(Path(str(state).upper()), _row(target))


def test_a_row_naming_another_state_is_elsewhere_in_every_spelling(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    _junction(state / "bgm-candidates", tmp_path / "tier" / "bgm-candidates")
    other = tmp_path / "clone" / "state"
    other.mkdir(parents=True)
    _junction(other / "bgm-candidates", tmp_path / "tier2" / "bgm-candidates")
    assert write_targets.adopted_elsewhere(state, _row(other / "bgm-candidates"))
    assert write_targets.adopted_elsewhere(state, _row(tmp_path / "tier2" / "bgm-candidates"))
    assert write_targets.adopted_elsewhere(other, _row(state / "bgm-candidates"))
    assert write_targets.adopted_elsewhere(other, _row(tmp_path / "tier" / "bgm-candidates"))


def test_a_genuinely_copied_plain_state_is_elsewhere(tmp_path):
    original, clone = tmp_path / "original", tmp_path / "clone"
    assert write_targets.adopted_elsewhere(clone, _row(original / "bgm-candidates"))
    assert not write_targets.adopted_elsewhere(clone, _row(clone / "bgm-candidates"))


def test_inside_and_same_place_agree_across_junction_spellings(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    target = tmp_path / "tier" / "bgm-candidates"
    _junction(state / "bgm-candidates", target)
    assert write_targets.inside(state / "bgm-candidates" / "x.flac", target)
    assert write_targets.inside(target / "x.flac", state / "bgm-candidates")
    assert write_targets.same_place(state / "bgm-candidates" / "r", target / "r")
    assert not write_targets.inside(tmp_path / "other" / "x.flac", target)
