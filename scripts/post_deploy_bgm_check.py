"""After `deploy-to-freesoft.ps1 -KeepPrevious`: one real BGM job by the installed EXE on the install's own link layout.

    python scripts/post_deploy_bgm_check.py --source <episode.mkv> [--install <LSS_DEPLOY_BASE>/LSS] [--scratch <LSS_WORK_ROOT>/post-deploy]

The installed runtime is copied into a scratch install. Every directory link in the install's top level and in
`.state` is mirrored: `models` points at the same model store (read by the job), state links point at new empty
targets. The source is copied into the scratch, so outputs never land beside the user's media and the user's state
is never touched. The receipt `<DeployBase>\\.LSS-postdeploy-bgm.json` names the installed EXE SHA-256;
`deploy-to-freesoft.ps1 -Finalize` refuses without a PASS receipt for the EXE it would finalize.

Why: release 76b88f4 passed every isolated-state check and was finalized, while every BGM job on the user's real
state failed because `.state\\work` was a junction (`release-evidence/strata-deploy-20261031/ce8d7000/`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

LINK_REFUSAL = "linkを含む"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_link(path: Path) -> bool:
    value = path.lstat()
    return bool(stat.S_ISLNK(value.st_mode)
                or getattr(value, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def junction(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True, timeout=30)


def mirror_layout(install: Path, scratch_install: Path, targets: Path) -> list[dict]:
    """Directory links of the real install, recreated in the scratch install."""
    mirrored = []
    for parent in (install, install / ".state"):
        if not parent.is_dir():
            continue
        for entry in sorted(parent.iterdir()):
            if not is_link(entry):
                continue
            relative = entry.relative_to(install)
            real = Path(os.path.realpath(entry))
            if relative == Path("models"):
                target = real  # the job reads the real model store
            else:
                target = targets / relative.as_posix().replace("/", "__")
                target.mkdir(parents=True, exist_ok=False)
            junction(scratch_install / relative, target)
            mirrored.append({"link": relative.as_posix(), "installed_target": str(real), "check_target": str(target)})
    return mirrored


def entry_result(scratch_install: Path) -> dict | None:
    """`LSS.exe --bgm-candidates` writes its result to the state, not to stdout."""
    path = scratch_install / ".state" / "bgm-candidate-result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def evaluate(run: Path, install: Path, exe_sha: str, original: Path, mirrored: list[dict], exit_code: int,
             seconds: float) -> dict:
    scratch_install, media = run / "LSS", run / "media"
    log = (run / "stdout.log").read_text(encoding="utf-8") + (run / "stderr.log").read_text(encoding="utf-8")
    result = entry_result(scratch_install)
    outputs = [Path(p) for p in (result or {}).get("outputs", [])]
    work = [m for m in mirrored if m["link"] == ".state/work"]
    work_left = sorted(os.listdir(work[0]["check_target"])) if work else []
    failures = []
    if exit_code != 0:
        failures.append(f"exit {exit_code}")
    if not result or result.get("disposition") != "PUBLISHED":
        failures.append(f"disposition {(result or {}).get('disposition')}")
    if not outputs or not all(p.is_file() and p.is_relative_to(media) for p in outputs):
        failures.append("no published FLAC inside the check media folder")
    if LINK_REFUSAL in log:
        failures.append("link refusal in output")
    if work_left:
        failures.append(f"work target not cleaned: {work_left[:5]}")
    source = next(media.glob(f"*{original.suffix}"))
    return {"schema": "lss.post-deploy-bgm/v1", "status": "PASS" if not failures else "FAIL",
            "failures": failures, "exe_sha256": exe_sha, "install": str(install), "source": str(original),
            "source_sha256": sha256(source), "mirrored_links": mirrored, "exit": exit_code, "seconds": seconds,
            "published_cues": len(outputs), "run": str(run)}


def work_root() -> Path:
    """Scratch root for development runs: ``LSS_WORK_ROOT``, else a folder in the user profile."""
    return Path(os.environ.get("LSS_WORK_ROOT") or Path.home() / ".lss-work")


def install_default() -> Path | None:
    """The installed product folder under ``LSS_DEPLOY_BASE``; without it ``--install`` is required."""
    base = os.environ.get("LSS_DEPLOY_BASE")
    return Path(base) / "LSS" if base else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--install", default=install_default(), required=install_default() is None, type=Path)
    parser.add_argument("--scratch", default=work_root() / "post-deploy", type=Path)
    parser.add_argument("--layout", type=Path, help="install whose link layout is mirrored (default: --install); "
                        "lets a frozen candidate package run with the installed model store")
    parser.add_argument("--timeout", default=3600, type=int)
    parser.add_argument("--from-run", type=Path, help="re-evaluate a finished run directory without running LSS")
    args = parser.parse_args(argv)
    install = args.install.resolve()
    exe = install / "LSS.exe"
    exe_sha = sha256(exe)
    if args.from_run:
        previous = json.loads((args.from_run / "receipt.json").read_text(encoding="utf-8"))
        if previous["exe_sha256"] != exe_sha or sha256(args.from_run / "LSS" / "LSS.exe") != exe_sha:
            raise SystemExit("the run was made by a different EXE than the installed one")
        receipt = evaluate(args.from_run, install, exe_sha, Path(previous["source"]), previous["mirrored_links"],
                           previous["exit"], previous["seconds"])
        return finish(receipt, args.from_run, install)
    run = args.scratch / f"{exe_sha[:12]}-{time.strftime('%Y%m%dT%H%M%S')}"
    scratch_install = run / "LSS"
    scratch_install.mkdir(parents=True)
    shutil.copy2(exe, scratch_install / "LSS.exe")
    shutil.copytree(install / "_internal", scratch_install / "_internal")
    for name in (".lss-runtime-manifest.json",):
        if (install / name).is_file():
            shutil.copy2(install / name, scratch_install / name)
    mirrored = mirror_layout((args.layout or install).resolve(), scratch_install, run / "targets")
    media = run / "media"
    media.mkdir()
    source = media / args.source.name
    shutil.copy2(args.source, source)

    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("PYTHON", "VIRTUAL_ENV", "LSS_"))}
    env["LSS_REFERENCES_DIR"] = str(run / "refs")
    started = time.monotonic()
    child = subprocess.run([str(scratch_install / "LSS.exe"), "--bgm-candidates", str(source)], env=env,
                           cwd=scratch_install, creationflags=0x08000000, capture_output=True, timeout=args.timeout)
    stdout = child.stdout.decode("utf-8", "replace")
    stderr = child.stderr.decode("utf-8", "replace")
    (run / "stdout.log").write_text(stdout, encoding="utf-8")
    (run / "stderr.log").write_text(stderr, encoding="utf-8")
    receipt = evaluate(run, install, exe_sha, args.source, mirrored, child.returncode,
                       round(time.monotonic() - started, 1))
    return finish(receipt, run, install)


def finish(receipt: dict, run: Path, install: Path) -> int:
    text = json.dumps(receipt, ensure_ascii=False, indent=2)
    (run / "receipt.json").write_text(text, encoding="utf-8")
    (install.parent / f".{install.name}-postdeploy-bgm.json").write_text(text, encoding="utf-8")
    print(text)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
