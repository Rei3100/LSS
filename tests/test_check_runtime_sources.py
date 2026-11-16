"""candidateの同梱物がソースと同じbytesかを確かめる。改行変換も差として見つける。"""
from __future__ import annotations

from scripts import check_runtime_sources as check


def _tree(root, *, packaged, newline="\n"):
    base = root / "_internal" if packaged else root
    adapter = (base / "runtime-sources" / "lss" / "harness" / "infer.py" if packaged
               else base / "lss" / "harness" / "infer.py")
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(f"print('adapter'){newline}".encode())
    (base / "vendor" / "msst" / "models").mkdir(parents=True)
    (base / "vendor" / "msst" / "inference.py").write_bytes(f"x = 1{newline}".encode())
    (base / "vendor" / "msst" / "models" / "m.py").write_bytes(f"y = 2{newline}".encode())
    sfc_adapter = (base / "runtime-sources" / "lss" / "bgm" / "sfc_analysis.py" if packaged
                   else base / "lss" / "bgm" / "sfc_analysis.py")
    sfc_adapter.parent.mkdir(parents=True)
    sfc_adapter.write_bytes(f"print('sfc'){newline}".encode())
    (base / "vendor" / "sfc_locoformer").mkdir(parents=True)
    (base / "vendor" / "sfc_locoformer" / "model.py").write_bytes(f"z = 3{newline}".encode())
    four_stem = (base / "runtime-sources" / "lss" / "bgm" / "four_stem_analysis.py" if packaged
                 else base / "lss" / "bgm" / "four_stem_analysis.py")
    four_stem.write_bytes(f"print('four stem'){newline}".encode())
    (base / "vendor" / "facing_the_music" / "core").mkdir(parents=True)
    (base / "vendor" / "facing_the_music" / "core" / "bandsplit.py").write_bytes(f"w = 5{newline}".encode())
    tagger = (base / "runtime-sources" / "lss" / "bgm" / "event_tagging.py" if packaged
              else base / "lss" / "bgm" / "event_tagging.py")
    tagger.write_bytes(f"print('event tags'){newline}".encode())
    sound = (base / "runtime-sources" / "lss" / "sound" if packaged else base / "lss" / "sound")
    sound.mkdir(parents=True)
    for name in check.SOUND_ENGINE_FILES + check.SOUND_PLANNER_FILES:
        (sound / name).write_bytes(f"print('{name}'){newline}".encode())
    (base / "vendor" / "sam_audio" / "sam_audio" / "model").mkdir(parents=True)
    (base / "vendor" / "sam_audio" / "sam_audio" / "model" / "transformer.py").write_bytes(f"v = 7{newline}".encode())
    for name in check.SCENE_RESOURCE_FILES:
        target = base / "lss" / "bgm" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"# scene {name}{newline}".encode())
    for name in check.BGM_RESOURCE_FILES:
        target = base / "lss" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"# resource {name}{newline}".encode())
    return root


def test_identical_bytes_pass(tmp_path):
    source = _tree(tmp_path / "src", packaged=False)
    candidate = _tree(tmp_path / "dist", packaged=True)
    assert check.main([str(candidate), "--source", str(source)]) == 0


def test_converted_line_endings_are_caught(tmp_path):
    source = _tree(tmp_path / "src", packaged=False)
    candidate = _tree(tmp_path / "dist", packaged=True, newline="\r\n")
    assert check.main([str(candidate), "--source", str(source)]) == 1
    differing = check.differences(check.runtime_sources(source, packaged=False),
                                  check.runtime_sources(candidate, packaged=True))
    assert "infer.py" in differing and "msst/inference.py" in differing


def test_a_missing_vendor_file_is_caught(tmp_path):
    source = _tree(tmp_path / "src", packaged=False)
    candidate = _tree(tmp_path / "dist", packaged=True)
    (candidate / "_internal" / "vendor" / "msst" / "models" / "m.py").unlink()
    assert check.main([str(candidate), "--source", str(source)]) == 1


def test_the_bgm_analyzer_adapter_and_vendor_are_compared(tmp_path):
    """単話BGM候補の解析器も、同梱物のbytesが解析器の版（analyzer_id）へ入る。"""
    source = _tree(tmp_path / "src", packaged=False)
    candidate = _tree(tmp_path / "dist", packaged=True)
    (candidate / "_internal" / "vendor" / "sfc_locoformer" / "model.py").write_bytes(b"z = 4")
    (candidate / "_internal" / "runtime-sources" / "lss" / "bgm" / "sfc_analysis.py").write_bytes(b"x")
    differing = check.differences(check.runtime_sources(source, packaged=False),
                                  check.runtime_sources(candidate, packaged=True))
    assert differing == ["sfc_analysis.py", "sfc_locoformer/model.py"]
    assert check.main([str(candidate), "--source", str(source)]) == 1


def test_the_four_stem_adapter_and_vendor_are_compared(tmp_path):
    """歌唱回収の4分類も、同梱物のbytesが4分類の版（analyzer_id）へ入る。"""
    source = _tree(tmp_path / "src", packaged=False)
    candidate = _tree(tmp_path / "dist", packaged=True)
    (candidate / "_internal" / "vendor" / "facing_the_music" / "core" / "bandsplit.py").write_bytes(b"w = 6")
    (candidate / "_internal" / "runtime-sources" / "lss" / "bgm" / "four_stem_analysis.py").write_bytes(b"x")
    differing = check.differences(check.runtime_sources(source, packaged=False),
                                  check.runtime_sources(candidate, packaged=True))
    assert differing == ["facing_the_music/core/bandsplit.py", "four_stem_analysis.py"]
    assert check.main([str(candidate), "--source", str(source)]) == 1


def test_the_event_tagging_adapter_is_compared(tmp_path):
    source = _tree(tmp_path / "src", packaged=False)
    candidate = _tree(tmp_path / "dist", packaged=True)
    (candidate / "_internal" / "runtime-sources" / "lss" / "bgm" / "event_tagging.py").write_bytes(b"print('other')")
    assert check.main([str(candidate), "--source", str(source)]) == 1
    assert "event_tagging.py" in check.differences(check.runtime_sources(source, packaged=False),
                                                   check.runtime_sources(candidate, packaged=True))


def test_the_wanted_sound_engine_and_planner_are_compared(tmp_path):
    """欲しい音の分離器（adapter・同梱model実装）と、経路の実装の同梱物も突き合わせる。"""
    source = _tree(tmp_path / "src", packaged=False)
    candidate = _tree(tmp_path / "dist", packaged=True)
    (candidate / "_internal" / "vendor" / "sam_audio" / "sam_audio" / "model" / "transformer.py").write_bytes(b"v = 8")
    (candidate / "_internal" / "runtime-sources" / "lss" / "sound" / "plan.py").write_bytes(b"x")
    differing = check.differences(check.runtime_sources(source, packaged=False),
                                  check.runtime_sources(candidate, packaged=True))
    assert differing == ["lss/sound/plan.py", "sam_audio/sam_audio/model/transformer.py"]
    assert check.main([str(candidate), "--source", str(source)]) == 1


def test_missing_memory_recipe_resource_is_caught_before_packaged_start(tmp_path):
    source = _tree(tmp_path / "src", packaged=False)
    candidate = _tree(tmp_path / "dist", packaged=True)
    (candidate / "_internal/lss/bgm/memory_support.py").unlink()
    differing = check.differences(check.runtime_sources(source, packaged=False),
                                  check.runtime_sources(candidate, packaged=True))
    assert differing == ["lss/bgm/memory_support.py"]
    assert check.main([str(candidate), "--source", str(source)]) == 1


def test_all_bgm_recipe_resources_are_declared_as_physical_spec_data():
    from lss import application

    spec = (check.ROOT / "LSS.spec").read_text(encoding="utf-8")
    resources = set(application._BGM_IMPLEMENTATION + application._BGM_REFERENCE_IMPLEMENTATION)
    assert resources == set(check.BGM_RESOURCE_FILES)
    for name in resources:
        assert f'("lss/{name}", "lss/bgm")' in spec, name


def test_missing_or_different_native_scene_resource_fails(tmp_path):
    source = _tree(tmp_path / "src", packaged=False)
    candidate = _tree(tmp_path / "dist", packaged=True)
    target = candidate / "_internal/lss/bgm/scene_native.py"
    target.unlink()
    assert "lss/bgm/scene_native.py" in check.differences(check.runtime_sources(source, packaged=False),
                                                           check.runtime_sources(candidate, packaged=True))
    assert check.main([str(candidate), "--source", str(source)]) == 1


def _data_tree(root, files):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_missing_tracked_data_file_is_reported(tmp_path):
    tracked = ["lss/acquisition/keys/yt-dlp.asc", "lss/ui/web/bridge.js"]
    _data_tree(tmp_path / "src", {name: b"data" for name in tracked})
    _data_tree(tmp_path / "dist" / "_internal", {"lss/ui/web/bridge.js": b"data"})
    assert check.missing_data_files(tmp_path / "src", tmp_path / "dist", tracked) == [
        "lss/acquisition/keys/yt-dlp.asc"]


def test_changed_data_file_is_reported(tmp_path):
    tracked = ["lss/ui/web/bridge.js"]
    _data_tree(tmp_path / "src", {tracked[0]: b"a\n"})
    _data_tree(tmp_path / "dist" / "_internal", {tracked[0]: b"a\r\n"})
    assert check.missing_data_files(tmp_path / "src", tmp_path / "dist", tracked) == tracked


def test_python_files_and_listed_exclusions_are_not_data(tmp_path):
    tracked = ["lss/core/x.py", "lss/NOTES.md", "lss/bgm/tests/fixture.bin"]
    _data_tree(tmp_path / "src", {name: b"x" for name in tracked})
    (tmp_path / "dist" / "_internal").mkdir(parents=True)
    assert check.missing_data_files(tmp_path / "src", tmp_path / "dist", tracked) == []
    # An exclusion is explicit: with an empty list the same files are required.
    assert check.missing_data_files(tmp_path / "src", tmp_path / "dist", tracked, {}) == [
        "lss/NOTES.md", "lss/bgm/tests/fixture.bin"]


def test_every_exclusion_has_a_reason():
    assert check.NOT_PACKAGED and all(reason.strip() for reason in check.NOT_PACKAGED.values())


def test_real_tracked_data_files_are_all_in_the_spec_or_excluded():
    tracked = check.tracked_data_files(check.ROOT)
    assert tracked is not None
    spec = (check.ROOT / "LSS.spec").read_text(encoding="utf-8")
    import fnmatch
    for name in tracked:
        if any(fnmatch.fnmatchcase(name, pat) for pat in check.NOT_PACKAGED):
            continue
        assert f'"{name}"' in spec, f"{name} is tracked runtime data but not in LSS.spec"
