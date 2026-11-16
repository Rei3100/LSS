"""BGM Recipe IDは、改行（LF/CRLF/CR）やUTF-8 BOMのような意味の無い表現の違いで変わらず、実装の中身の違いでは変わる。

独立監査で、同じ実装がcheckoutの改行変換だけで別のRecipe IDになった（`dialogue_residual.py` のCRLF）。
v12からRecipeの識別は `runtime.text_source_sha256` の内容hashを使う。実装を丸ごと一時領域へ複製し、
LFの複製とCRLFの複製を別のprocessで読み込んでRecipe IDを比べる。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lss import application
from lss.core import runtime

ROOT = Path(__file__).resolve().parents[1]
_RECIPE_ID = "from lss import application; print(application._bgm_candidate_recipe()['recipe_id'])"


def test_text_source_digest_ignores_line_endings_and_bom(tmp_path):
    body = "def f():\n    return 1\n"
    variants = {"lf": body.encode(), "crlf": body.replace("\n", "\r\n").encode(),
                "cr": body.replace("\n", "\r").encode(), "bom": b"\xef\xbb\xbf" + body.replace("\n", "\r\n").encode()}
    digests = set()
    for name, data in variants.items():
        (tmp_path / name).write_bytes(data)
        digests.add(runtime.text_source_sha256(tmp_path / name))
    assert len(digests) == 1
    (tmp_path / "changed").write_bytes(b"def f():\n    return 2\n")
    assert runtime.text_source_sha256(tmp_path / "changed") not in digests


def test_binary_artifacts_keep_their_raw_identity(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"\x00\r\n\x01")
    (tmp_path / "b.bin").write_bytes(b"\x00\n\x01")
    assert runtime.text_source_sha256(tmp_path / "a.bin") != runtime.text_source_sha256(tmp_path / "b.bin")
    (tmp_path / "c.bin").write_bytes(b"\xff\xfe\r\n")   # UTF-8として読めない
    assert runtime.text_source_sha256(tmp_path / "c.bin") == runtime._sha256(tmp_path / "c.bin")


def test_runtime_identity_still_uses_raw_bytes(tmp_path):
    (tmp_path / "x.py").write_bytes(b"a\r\n")
    raw = runtime._provider_artifacts([tmp_path / "x.py"])
    normalized = runtime._provider_artifacts([tmp_path / "x.py"], normalize_text=True)
    assert raw != normalized and normalized[0][1] == runtime.text_source_sha256(tmp_path / "x.py")


def _copy_tree(target: Path, *, crlf: bool) -> Path:
    for name in ("lss", "vendor", "ledger"):
        shutil.copytree(ROOT / name, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if crlf:
        for path in target.rglob("*"):
            if path.is_file():
                data = path.read_bytes()
                if b"\x00" in data:
                    continue
                try:
                    data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                path.write_bytes(data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    else:
        for path in target.rglob("*.py"):
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    return target


def _recipe_id(root: Path) -> str:
    result = subprocess.run([sys.executable, "-c", _RECIPE_ID], cwd=root, capture_output=True, text=True,
                            env={"PYTHONPATH": str(root), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
                                 "PATH": __import__("os").environ.get("PATH", "")}, timeout=300)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_the_same_implementation_under_lf_and_crlf_has_one_recipe_id(tmp_path):
    lf = _recipe_id(_copy_tree(tmp_path / "lf", crlf=False))
    crlf = _recipe_id(_copy_tree(tmp_path / "crlf", crlf=True))
    assert lf == crlf == application._bgm_candidate_recipe()["recipe_id"]
    assert lf.startswith("bgm-episode-candidates-v13-")


def test_a_meaningful_change_in_any_recipe_source_changes_the_recipe_id(tmp_path):
    base = _recipe_id(_copy_tree(tmp_path / "base", crlf=False))
    changed = _copy_tree(tmp_path / "changed", crlf=False)
    path = changed / "lss" / "bgm" / "cue_scope.py"
    path.write_text(path.read_text(encoding="utf-8").replace("EVENT_SCENE_FRACTION = 0.2",
                                                             "EVENT_SCENE_FRACTION = 0.25"), encoding="utf-8")
    assert _recipe_id(changed) != base
    analyzer = _copy_tree(tmp_path / "analyzer", crlf=False)
    adapter = analyzer / "lss" / "bgm" / "event_tagging.py"
    adapter.write_text(adapter.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    assert _recipe_id(analyzer) != base


def test_every_recipe_implementation_source_is_bundled_and_identified():
    spec = (ROOT / "LSS.spec").read_text(encoding="utf-8")
    sources = application._bgm_implementation_sources()
    assert set(sources) == set(application._BGM_IMPLEMENTATION)
    for name in application._BGM_IMPLEMENTATION:
        assert f'("lss/{name}", "lss/bgm")' in spec, name


def test_historical_recipe_ids_remain_readable_as_superseded_versions():
    from lss.bgm import current_best

    declaration = application._bgm_candidate_recipe()["supersedes"]
    assert "episode-sfc-routed-cues-v10" in declaration["supersedes_workflows"]
    assert declaration["supersedes_workflows"] == list(current_best.SUPERSEDED_WORKFLOWS)
