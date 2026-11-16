"""実曲を通常の処理へ通し、結果に付く場面の観測を読む。

伴奏抽出の日常利用版の入力は1曲である。作られたケースと番組1本では読み方が違ったので、
実際の曲でどう出るのかを見る。

通常のworkerをそのまま呼ぶ。観測は追加の推論なしでworkerが付けるものを読むだけで、
ここで測り直さない。参照が無いので品質は測らない。
公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.core import catalog, models_store, session, worker  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
DECISION_RULE = {
    "the_ordinary_worker_produces_the_observation": True,
    "no_quality_is_measured_because_there_is_no_matching_reference": True,
    "nothing_is_published_and_no_default_changes": True,
}
LIMITS = [
    "real_songs_without_a_same_master_reference_so_quality_is_not_measured",
    "a_handful_of_tv_size_songs_is_not_a_corpus",
    "the_observation_inherits_the_errors_of_the_separation_it_reads",
    "no_publication_default_change_or_holdout_use",
]


class SongBalanceProbeError(RuntimeError):
    """実曲の処理が完了しない。"""


def run_probe(songs: list[Path], output: Path) -> Path:
    for path in songs:
        if not path.is_file():
            raise SongBalanceProbeError(f"入力がありません: {path}")
    output.mkdir(parents=True, exist_ok=False)
    ledger = output / "models.tsv"
    shutil.copyfile(models_store.MODELS_TSV, ledger)
    models_store.MODELS_TSV = ledger
    os.environ["LSS_STATE_DIR"] = str(output / "state")
    os.environ["LSS_MIN_FREE_BYTES"] = "0"
    default = catalog.default_model()

    rows = []
    for index, path in enumerate(songs):
        digest = fetch.file_sha256(path)
        result = worker.handle_job(
            {"mix_path": str(path), "model_id": default["model_id"],
             "stem_target": catalog.requested_stem(default)},
            destination_dir=output / "outputs" / f"{index:02d}")
        if result["status"] not in {"ok", "skipped"}:
            raise SongBalanceProbeError(f"分離に失敗しました: {path}: {result}")
        if fetch.file_sha256(path) != digest:
            raise SongBalanceProbeError(f"処理中に入力が変更されています: {path}")
        balance = result.get("scene_balance")
        rows.append({"name": path.name, "source": str(path), "source_sha256": digest,
                     "status": result["status"],
                     "recipe_id": result.get("recipe_id"),
                     "scene_balance": balance})
        state = None if balance is None else balance.get("state")
        shares = None if balance is None else (balance.get("over_time") or {}).get("state_shares")
        print(f"{path.name}: {state} / {shares}", flush=True)

    states = {}
    for row in rows:
        name = (row["scene_balance"] or {}).get("state")
        states[name] = states.get(name, 0) + 1
    result_path = output / "report.json"
    session._write_json_atomic(result_path, {
        "schema_version": SCHEMA_VERSION,
        "type": "scene_balance_songs_report",
        "status": "complete_real_song_description",
        "role": "development_real_material_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "model_id": default["model_id"],
        "file_recipe_id": worker.SCENE_BALANCE_RECIPE.recipe_id,
        "timed_recipe_id": worker.SCENE_BALANCE_TIMED_RECIPE.recipe_id,
        "decision_rule": DECISION_RULE,
        "song_count": len(rows), "states": states,
        "songs": rows,
        "limits": LIMITS,
        "implementation": [{"path": str(Path(__file__)),
                            "sha256": fetch.file_sha256(Path(__file__))}],
    })
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--song", type=Path, action="append", required=True,
                        help="実曲のパス。複数指定する")
    args = parser.parse_args(argv)
    result = run_probe([path.resolve() for path in args.song], args.output.resolve())
    print(f"実曲での観測を保存しました: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
