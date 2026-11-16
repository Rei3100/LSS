r"""One real BGM job by the installed EXE on a development episode, outputs kept for scoring.

    python scripts/installed_episode_run.py --source <episode.mkv> [--scratch <LSS_WORK_ROOT>/ep-score] [--references <folder of reference recordings>]

The same isolated run as ``post_deploy_bgm_check.py`` (scratch install with the install's link layout, source copied,
user state untouched) but it only writes ``<run>/receipt.json``: the deploy receipt beside the install that
``deploy-to-freesoft.ps1 -Finalize`` reads is left alone. The run directory is kept; score it with
``ep001-truth/score_usages.py`` / ``scripts/episode_score.py``. Start it with ``hidden_run.py --storage-gb 30``;
the Product takes the GPU lease itself, so do not wrap it in ``gpu_run.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import post_deploy_bgm_check as base


def finish(receipt: dict, run: Path, install: Path) -> int:
    text = json.dumps(receipt, ensure_ascii=False, indent=2)
    (run / "receipt.json").write_text(text, encoding="utf-8")
    print(text)
    return 0 if receipt["status"] == "PASS" else 1


def with_references(folder: Path) -> None:
    """Copy the reference recordings into the run's own references folder just before the EXE starts."""
    import shutil
    real = base.subprocess.run

    def run(command, *args, **kwargs):
        if str(command[0]).endswith("LSS.exe"):
            target = Path(kwargs["env"]["LSS_REFERENCES_DIR"])
            target.mkdir(parents=True, exist_ok=True)
            for item in sorted(Path(folder).glob("*")):
                if item.is_file():
                    shutil.copy2(item, target / item.name)
        return real(command, *args, **kwargs)
    base.subprocess.run = run   # other commands pass straight through


if __name__ == "__main__":
    base.finish = finish
    if "--references" in sys.argv:
        at = sys.argv.index("--references")
        with_references(Path(sys.argv[at + 1]))
        del sys.argv[at:at + 2]
    if "--scratch" not in sys.argv:
        sys.argv += ["--scratch", str(base.work_root() / "ep-score")]
    sys.exit(base.main())
