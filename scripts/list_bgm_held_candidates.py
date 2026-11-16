"""保存済みBGM保留結果の出典・区間・原候補と代替・全所在を、再生成せず一覧にする。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lss.assets import held_catalog


def listing(records):
    results, occurrences, representations = [], set(), set()
    for snapshot in records:
        record = snapshot.record
        cues = {item["candidate_id"]: {"primary_candidate_id": item["candidate_id"],
                "origin": item["origin"], "primary": item, "alternatives": []}
                for item in snapshot.candidates if item["role"] == "primary"}
        for item in snapshot.candidates:
            representations.add((item["candidate_id"], item["content_sha256"]))
            if item["role"] != "primary":
                cues[item["primary_candidate_id"]]["alternatives"].append(item)
                continue
            origin = item["origin"]
            occurrences.add((origin["source_sha256"],origin["start_frame"],origin["end_frame"],origin["sample_rate"],
                             origin.get("time_offset_numerator",0),origin.get("time_offset_denominator",1)))
        results.append({"record_path": str(snapshot.path), "record_sha256": snapshot.digest,
                        "source": record["source"], "source_locations": [str(p) for p in snapshot.source_locations],
                        "disposition": "HELD", "evidence": record["evidence"], "reason": record["reason"],
                        "unavailable_evidence": snapshot.unavailable_evidence, "cues": list(cues.values())})
    return {"status": "HELD", "publication_allowed": False, "published_outputs": [],
            "semantic_ground_truth": None, "winner_selected": False,
            "counts": {"held_results": len(results), "logical_occurrences": len(occurrences),
                       "held_representations": len(representations), "published_outputs": 0},
            "results": results, "claim": "保存済みの未公開候補の出典・区間・現在確認できた所在。意味的BGM正解や勝者は未確定。"}


def export_new(path, payload, snapshots):
    """明示した新規JSONだけを確定する。既存ファイル・入力・来歴を上書きしない。"""
    path = Path(path).resolve()
    if path.suffix.casefold() != ".json" or path.exists():
        raise ValueError("一覧の保存先は未使用の新規JSON pathである必要があります")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w",dir=path.parent,prefix=".held-listing-",suffix=".tmp",
                                encoding="utf-8",newline="\n",delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write("\n");stream.flush();os.fsync(stream.fileno())
        held_catalog.verify(snapshots)
        # 同時に保存先が作られても、linkは既存ファイルを置き換えない。
        os.link(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records",type=Path,nargs="+",help="application-outcome.jsonのpath（複数可）")
    parser.add_argument("--state-root",type=Path,help="移動記録を読む状態領域")
    parser.add_argument("--out",type=Path,help="新規JSONへ一覧を保存する。同名ファイルがあれば失敗する")
    args = parser.parse_args(argv)
    try:
        if args.out is not None and (args.out.exists() or args.out.suffix.casefold() != ".json"):
            raise ValueError("一覧の保存先は未使用の新規JSON pathである必要があります")
        snapshots = held_catalog.snapshots(args.records,state_root=args.state_root)
        payload = listing(snapshots)
        if args.out is None:
            held_catalog.verify(snapshots)
            print(json.dumps(payload,ensure_ascii=False,indent=2))
        else:
            export_new(args.out,payload,snapshots)
            print(json.dumps({"status":"HELD","counts":payload["counts"],"path":str(args.out.resolve())},ensure_ascii=False))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps({"status":"FAILED","error_code":"held_listing_failed",
                          "message":f"保留候補の一覧を確認・保存できません: {exc}"},ensure_ascii=False),file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
