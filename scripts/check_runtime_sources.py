"""ビルドしたcandidateが、ソースと同じbytesのproviderとadapterを同梱しているか確かめる。

実行環境ID（runtime_id）はprovider配布物とadapterの内容hashを含み、処理IDの一部になる。
改行変換などでbytesが1つでも変わると、同じ処理の過去の成果を再利用できなくなる。
配備の前にcandidateの同梱物をソースと突き合わせ、違えば止める。

    .venv/Scripts/python.exe scripts/check_runtime_sources.py <dist/LSS>
"""
from __future__ import annotations

import argparse
import hashlib
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.core import runtime  # noqa: E402
from lss import application  # noqa: E402


# 欲しい音の抽出器（adapter・文のencoder）と、要求の読み方・経路の実装。
SOUND_ENGINE_FILES = ("sam_audio.py", "t5_text.py", "pe_frame.py", "translate.py", "xling.py", "interpret.py")
SOUND_PLANNER_FILES = ("intent.py", "lexicon.py", "plan.py", "extract.py", "verify.py", "job.py",
                       "audioset_relations.py")
BGM_RESOURCE_FILES = tuple(sorted(set(application._BGM_IMPLEMENTATION + application._BGM_REFERENCE_IMPLEMENTATION)))
SCENE_RESOURCE_FILES = ("scene_job.py", "scene_runtime.py", "scene_native.py", "scene_models.py", "scene_names.py")


def runtime_sources(root: Path, *, packaged: bool) -> dict[str, str]:
    base = root / "_internal" if packaged else root
    sources = base / "runtime-sources" if packaged else base
    artifacts = dict(runtime._provider_artifacts(
        (sources / "lss" / "harness" / "infer.py", base / "vendor" / "msst")))
    # 単話BGM候補の解析器（SFC）。adapterと同梱model実装が解析器の版（analyzer_id）に入る。
    artifacts.update(runtime._provider_artifacts(
        (sources / "lss" / "bgm" / "sfc_analysis.py", base / "vendor" / "sfc_locoformer")))
    # 歌唱回収の4分類。adapterと同梱model実装が4分類の版（analyzer_id）に入る。
    artifacts.update(runtime._provider_artifacts(
        (sources / "lss" / "bgm" / "four_stem_analysis.py", base / "vendor" / "facing_the_music")))
    # 意味判定のタグ付け。adapter（model実装を含む）がタグ付けの版（analyzer_id）に入る。
    artifacts.update(runtime._provider_artifacts((sources / "lss" / "bgm" / "event_tagging.py",)))
    # 欲しい音の言葉による分離（SAM Audio）。adapter・文のencoder・同梱model実装が抽出の版に入り、
    # 要求の読み方と経路の実装（lss/sound）が欲しい音のRecipeに入る。
    artifacts.update(runtime._provider_artifacts(
        tuple(sources / "lss" / "sound" / name for name in SOUND_ENGINE_FILES) + (base / "vendor" / "sam_audio",)))
    artifacts.update({f"lss/sound/{name}": digest for name, digest in runtime._provider_artifacts(
        tuple(sources / "lss" / "sound" / name for name in SOUND_PLANNER_FILES))})
    # BGM computes Recipe identity from these physical resources even when the module executes from PYZ.
    # Checking adapters alone misses a package that imports memory support but cannot identify its source.
    for name in BGM_RESOURCE_FILES:
        path = base / "lss" / name
        artifacts[f"lss/{name}"] = (hashlib.sha256(path.read_bytes()).hexdigest()
                                    if path.is_file() else "MISSING")
    for name in SCENE_RESOURCE_FILES:
        path = base / "lss" / "bgm" / name
        artifacts[f"lss/bgm/{name}"] = (hashlib.sha256(path.read_bytes()).hexdigest()
                                        if path.is_file() else "MISSING")
    return artifacts


def differences(source: dict[str, str], candidate: dict[str, str]) -> list[str]:
    return sorted(name for name in set(source) | set(candidate)
                  if source.get(name) != candidate.get(name))


# Git-tracked non-.py files under lss/ that the package legitimately does not carry. Every other tracked
# runtime data file must be in the package byte-identical: the code reads it beside its module
# (e.g. acquisition/keys/yt-dlp.asc via provider.pinned_keys), and a missing one fails only at run time.
# pattern (fnmatch on the repo-relative posix path) -> reason
NOT_PACKAGED: dict[str, str] = {
    "lss/*.md": "documentation, not read by the program",
    "lss/*/tests/*": "test fixtures; tests do not ship",
    "lss/*/__pycache__/*": "bytecode cache, never tracked content",
}


def tracked_data_files(source: Path) -> list[str] | None:
    """Tracked non-.py files under lss/ of a git work tree, or None when `source` is not one."""
    try:
        out = subprocess.run(["git", "-C", str(source), "ls-files", "-z", "--", "lss"],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    names = [n.decode("utf-8") for n in out.split(b"\0") if n]
    return [n for n in names if not n.endswith(".py")]


def missing_data_files(source: Path, packaged: Path, tracked: list[str],
                       not_packaged: dict[str, str] = NOT_PACKAGED) -> list[str]:
    """Tracked data files absent from, or different in, the package (`packaged` is dist/LSS)."""
    bad = []
    for name in sorted(tracked):
        if name.endswith(".py") or any(fnmatch.fnmatchcase(name, pat) for pat in not_packaged):
            continue
        built = packaged / "_internal" / name
        original = source / name
        if not built.is_file() or not original.is_file() or built.read_bytes() != original.read_bytes():
            bad.append(name)
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="ビルドしたdist/LSS")
    parser.add_argument("--source", type=Path, default=ROOT, help="突き合わせるソースのツリー")
    parser.add_argument("--require-data-files", action="store_true",
                        help="fail when the source is not a git tree, so tracked data files cannot be compared")
    args = parser.parse_args(argv)
    source = runtime_sources(args.source, packaged=False)
    candidate = runtime_sources(args.candidate, packaged=True)
    differing = differences(source, candidate)
    tracked = tracked_data_files(args.source)
    data_missing = missing_data_files(args.source, args.candidate, tracked) if tracked is not None else []
    print(json.dumps({"files": len(source), "differing": differing[:20],
                      "differing_count": len(differing),
                      "data_files": "unchecked" if tracked is None else len(tracked),
                      "data_missing": data_missing}, ensure_ascii=False))
    if tracked is None and args.require_data_files:
        print("source is not a git tree: tracked data files were not compared", file=sys.stderr)
        return 1
    return 1 if differing or data_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
