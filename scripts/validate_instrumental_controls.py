"""MUSDBの固定した学習側3曲で、伴奏の保持・歌唱残留・無対象時を実GPU診断する。

公開7秒サンプルのローカルZIPを入力にする。既定変更・昇格・最終評価は行わない。
最初に構成と入力を固定し、その後に推論する。--resumeで固定済みの構成を再開する。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from mutagen.flac import FLAC  # noqa: E402

from lss.core import catalog, models_store, session, worker  # noqa: E402
from lss.harness import fetch  # noqa: E402
from lss.validation.instrumental_quality import instrumental_quality  # noqa: E402


ARCHIVE_SHA256 = "365f66acffc57282e474a4c24b32afec3678258c50b8d45646ac849b0a287df3"
ARCHIVE_URL = "https://github.com/sigsep/sigsep-mus-db/releases/download/v0.4.0/MUSDB18-7-STEMS.zip"
TRACKS = (
    ("nightowl", "A Classic Education - NightOwl"),
    ("easy-tiger", "ANiMAL - Easy Tiger"),
    ("velvet-curtain", "Alexander Ross - Velvet Curtain"),
)
EXTENSION_TRACKS = (
    ("devils-words", "Actions - Devil's Words"),
    ("child", "Aimee Norwich - Child"),
    ("milk-cow-blues", "Angela Thomas Wade - Milk Cow Blues"),
)
VARIANTS = (
    ("accompaniment-only", 1.0, 0.0),
    ("singing-mix", 1.0, 1.0),
    ("vocals-only", 0.0, 1.0),
    ("quiet-accompaniment", 0.1, 1.0),
)


def write_json(path, payload):
    session._write_json_atomic(path, payload)


def build(archive_path: Path, output: Path, *, extension: bool = False) -> Path:
    if fetch.file_sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("サンプルZIPが確認済みの内容と一致しません")
    ffmpeg = ROOT / "bin/ffmpeg.exe"
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    sources, cases = [], []
    with zipfile.ZipFile(archive_path) as archive:
        for track_id, title in EXTENSION_TRACKS if extension else TRACKS:
            member = f"train/{title}.stem.mp4"
            container = inputs / f"{track_id}.stem.mp4"
            # 固定したmemberだけを固定した名前へ保存する。test側は展開・評価しない。
            with archive.open(member) as source, container.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            components = []
            for stream_id in range(1, 5):
                decoded = inputs / f"{track_id}-component-{stream_id}.wav"
                subprocess.run(
                    [str(ffmpeg), "-nostdin", "-v", "error", "-i", str(container),
                     "-map", f"0:a:{stream_id}", "-c:a", "pcm_f32le", str(decoded)],
                    check=True, capture_output=True, timeout=60,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                values, rate = sf.read(decoded, always_2d=True, dtype="float64")
                if rate != 44100 or values.shape[1] != 2 or not np.isfinite(values).all():
                    raise ValueError(f"サンプルcomponentの形式が想定外です: {decoded}")
                components.append(values)
            if len({value.shape for value in components}) != 1:
                raise ValueError(f"componentの時間軸が一致しません: {title}")
            accompaniment = sum(components[:3])
            vocals = components[3]
            if not np.any(accompaniment) or not np.any(vocals):
                raise ValueError(f"固定曲の伴奏か歌唱が無音です: {title}")
            # 公開されたAAC componentの和で入力を定義する。artistic mixのsame-master
            # clean truthとは主張しない。全variantへ同じ2の冪gainを掛ける。
            peak = max(float(np.max(np.abs(accompaniment + vocals))),
                       float(np.max(np.abs(accompaniment))), float(np.max(np.abs(vocals))))
            shift = max(0, int(np.ceil(np.log2(max(peak / 0.5, 1.0)))))
            gain = 2.0 ** -shift
            sources.append({
                "track_id": track_id, "title": title, "member": member,
                "container_sha256": fetch.file_sha256(container), "common_gain": gain,
                "decoded_components": [
                    {"stream_id": index, "sha256": fetch.file_sha256(inputs / f"{track_id}-component-{index}.wav")}
                    for index in range(1, 5)
                ],
            })
            for variant, target_gain, vocals_gain in VARIANTS:
                case_id = f"{track_id}-{variant}"
                target = (accompaniment * gain * target_gain).astype(np.float32)
                unwanted = (vocals * gain * vocals_gain).astype(np.float32)
                mix = target + unwanted
                records = {}
                for role, values in (("input", mix), ("target", target), ("vocals", unwanted)):
                    path = inputs / f"{case_id}-{role}.wav"
                    sf.write(path, values, 44100, subtype="FLOAT")
                    records[role] = {"path": str(path.resolve()), "sha256": fetch.file_sha256(path)}
                cases.append({"case_id": case_id, "variant": variant, "frames": len(mix),
                              "sample_rate": 44100, "artifacts": records})
    plan = output / "plan.json"
    write_json(plan, {
        "schema_version": 1, "role": "development_diagnostic_only", "promotion_allowed": False,
        "case_group": "additional_development_tracks" if extension else "initial_development_tracks",
        "archive_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA256,
        "hash_authority": "observed_download_content_not_publisher_attestation",
        "ffmpeg_sha256": fetch.file_sha256(ffmpeg), "sources": sources, "cases": cases,
        "metric_policy": "uncorrected_stereo_fidelity_plus_diagnostic_two_component_fit_v1",
        "limits": ["public_training_data_not_independent_holdout", "7_second_AAC_components_not_full_song_fidelity",
                   "constructed_linear_mixture_not_finished_commercial_master", "no_automatic_presence_validator_promotion"],
    })
    return plan


def evaluate(plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("role") != "development_diagnostic_only" or plan.get("promotion_allowed") is not False:
        raise ValueError("Development用に固定した構成だけを診断します")
    # 1件目の推論前に、全固定入力とreferenceを検証する。
    for case in plan["cases"]:
        for artifact in case["artifacts"].values():
            if fetch.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError(f"固定入力が変更されました: {artifact['path']}")
    output = plan_path.parent
    private_ledger = output / "models.tsv"
    if not private_ledger.exists():
        shutil.copyfile(models_store.MODELS_TSV, private_ledger)
    models_store.MODELS_TSV = private_ledger
    os.environ["LSS_STATE_DIR"] = str(output / "state")
    os.environ["LSS_MIN_FREE_BYTES"] = "0"
    default = catalog.default_model()
    measurements = []
    for case in plan["cases"]:
        source = Path(case["artifacts"]["input"]["path"])
        target, rate = sf.read(case["artifacts"]["target"]["path"], always_2d=True, dtype="float64")
        vocals, vocals_rate = sf.read(case["artifacts"]["vocals"]["path"], always_2d=True, dtype="float64")
        result = worker.handle_job(
            {"mix_path": str(source), "model_id": default["model_id"], "stem_target": "Instrumental"},
            destination_dir=output / "outputs" / case["case_id"],
        )
        if result["status"] not in {"ok", "skipped"} or len(result["outputs"]) != 1:
            raise RuntimeError(f"固定ケースの分離失敗: {case['case_id']}: {result}")
        saved = Path(result["outputs"][0])
        estimate, saved_rate = sf.read(saved, always_2d=True, dtype="float64")
        if saved_rate != rate or vocals_rate != rate or estimate.shape != target.shape:
            raise ValueError(f"診断時の時間軸・形式不一致: {case['case_id']}")
        shift = int(FLAC(saved)["LSS_OUTPUT_HEADROOM_SHIFT"][0])
        # 宣言済みserialization gainだけを元へ戻す。fit・peak正規化・time補正はしない。
        quality = instrumental_quality((estimate * 2.0 ** shift).T, target.T, vocals.T)
        record = {"case_id": case["case_id"], "variant": case["variant"], "result": result,
                  "output_sha256": fetch.file_sha256(saved), "serialization_shift": shift, "quality": quality}
        measurements.append(record)
        write_json(output / f"{case['case_id']}-result.json", record)
        print(f"固定ケース完了: {case['case_id']}", flush=True)
    report = output / "report.json"
    write_json(report, {
        "schema_version": 1, "role": "development_diagnostic_only", "promotion_allowed": False,
        "default_changed": False, "holdout_consumed": False,
        "plan_sha256": fetch.file_sha256(plan_path), "cases": measurements,
        "metric_implementation": [
            {"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
                ROOT / "lss/validation/instrumental_quality.py", ROOT / "lss/validation/stereo_quality.py", Path(__file__),
            )
        ],
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extension", action="store_true", help="別artistの固定Development3曲を構成する")
    args = parser.parse_args()
    output = args.output.resolve()
    plan = output / "plan.json" if args.resume else build(args.archive.resolve(), output, extension=args.extension)
    if args.build_only:
        print(f"固定構成を保存しました: {plan}")
    else:
        print(f"伴奏の固定ケース診断を保存しました: {evaluate(plan)}")


if __name__ == "__main__":
    main()
