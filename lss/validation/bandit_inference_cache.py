"""固定Development推論を、全stemが検証済みのケースだけ再利用して続ける。"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable

import numpy as np
import soundfile as sf

from lss.core import audio_identity
from lss.core.session import _write_json_atomic
from lss.harness.fetch import file_sha256
from lss.bgm.inference_variants import InferenceVariantRecipe

ROLES = ("music", "speech", "effects")


def output_roles(plan: dict) -> tuple[str, ...]:
    """従来の3stemか、計画と実行同一性で固定した名前付きstemを返す。"""
    if "output_roles" not in plan:
        return ROLES
    roles = plan["output_roles"]
    if (not isinstance(roles, list) or not 1 <= len(roles) <= 32
            or any(not isinstance(role, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", role)
                   for role in roles) or len(set(roles)) != len(roles)):
        raise ValueError("一意で安全な名前を持つ固定output_rolesが必要です")
    return tuple(roles)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_plan(plan: dict) -> None:
    if (plan.get("status") != "FROZEN_BEFORE_RESULTS" or plan.get("promotion_allowed") is not False
            or plan.get("holdout_consumed") is not False or plan.get("sample_rate") != 44100):
        raise ValueError("44.1kHz stereoの未昇格Development planが必要です")
    output_roles(plan)
    cases = plan["cases"]
    ids = [case["case_id"] for case in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError("一意な固定ケースが必要です")
    for case in cases:
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,160}", case["case_id"])
                or case.get("stems") or type(case["source"]["frames"]) is not int
                or case["source"]["frames"] <= 0):
            raise ValueError("安全なケースIDと未推論の正の時間軸が必要です")
        if "inference_variant" in case:
            if not isinstance(case["inference_variant"], dict):
                raise ValueError("固定した入力variantは辞書である必要があります")
            try:
                InferenceVariantRecipe(**case["inference_variant"]).validate()
            except TypeError as exc:
                raise ValueError("固定した入力variantの定義が不正です") from exc


def verify_source(case: dict) -> None:
    source = case["source"]
    if file_sha256(Path(source["path"])) != source["sha256"]:
        raise ValueError("固定sourceの内容が変わっています")
    info = sf.info(source["path"])
    if info.channels != 2 or info.samplerate != 44100 or info.frames != source["frames"]:
        raise ValueError("固定sourceのstereo/timelineが不正です")


def validate_stems(case: dict, records: dict, *, directory: Path, roles=ROLES) -> None:
    if set(records) != set(roles):
        raise ValueError("固定stemすべての完了記録が必要です")
    for role in roles:
        record = records[role]
        path = directory / f"{role}.wav"
        if Path(record["path"]).resolve() != path.resolve():
            raise ValueError("推論stemの所在・内容が完了記録と一致しません")
        if not path.is_file():
            raise FileNotFoundError(path)
        with sf.SoundFile(path) as stream:
            if (stream.channels != 2 or stream.samplerate != 44100
                    or stream.frames != case["source"]["frames"] or stream.subtype != "FLOAT"):
                raise ValueError("推論stemのstereo/timeline/精度が不正です")
            while stream.tell() < stream.frames:
                if not np.isfinite(stream.read(65536, dtype="float32", always_2d=True)).all():
                    raise ValueError("推論stemに有限でない値があります")
        # 外部の再生でtagが書き戻されただけのstemは、同じ推論結果として再利用する。
        if not audio_identity.matches_record(path, record):
            raise ValueError("推論stemの所在・内容が完了記録と一致しません")


@contextmanager
def exclusive_output(output: Path, *, resume: bool):
    """プロセス死亡でもOSが解放する所有権。残ったlockファイルは再開を妨げない。"""
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".owner.lock").open("a+b") as stream:
        try:
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("同じ推論出力を別プロセスが使用しています") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class InferenceCache:
    def __init__(self, output: Path, plan: dict, identity: dict, *, resume: bool):
        validate_plan(plan)
        if identity.get("execution_device") != "cuda:0" or identity.get("cpu_fallback") is not False:
            raise ValueError("CUDA専用でfallbackのない実行条件が必要です")
        if "output_roles" in plan and identity.get("output_roles") != plan["output_roles"]:
            raise ValueError("計画と実行同一性のoutput_rolesが一致しません")
        self.roles = output_roles(plan)
        self.output, self.plan, self.identity = output, plan, identity
        manifest = output / "identity.json"
        if manifest.exists():
            if not resume or read_json(manifest) != identity:
                raise ValueError("再開する推論の構成・provider・実装・runtimeが一致しません")
        else:
            # 未記録の旧出力へidentityを後付けしない。
            if any(path.name != ".owner.lock" for path in output.iterdir()):
                raise ValueError("推論条件の記録がない出力は再利用できません")
            _write_json_atomic(manifest, identity)

    def load(self, case: dict) -> dict | None:
        verify_source(case)
        directory = self.output / case["case_id"]
        if not directory.exists():
            return None
        record = read_json(directory / "case.json")
        if (record["identity"] != self.identity or record["case_id"] != case["case_id"]
                or record["source_sha256"] != case["source"]["sha256"]
                or record.get("source_hash_after_verified") is not True):
            raise ValueError("ケースの固定条件・source完了記録が一致しません")
        expected_variant = case.get("inference_variant")
        if record.get("inference_variant") != expected_variant:
            raise ValueError("ケースの入力variantが固定条件と一致しません")
        if expected_variant is not None and record.get("inference_variant_id") != InferenceVariantRecipe(**expected_variant).recipe_id:
            raise ValueError("ケースの入力variant IDが固定条件と一致しません")
        self._validate_stats(record)
        validate_stems(case, record["stems"], directory=directory, roles=self.roles)
        return record

    @staticmethod
    def _validate_stats(record: dict) -> None:
        if (type(record["seconds"]) not in (int, float) or not math.isfinite(record["seconds"])
                or record["seconds"] < 0 or type(record["peak_cuda_bytes"]) is not int
                or record["peak_cuda_bytes"] <= 0):
            raise ValueError("ケースのCUDA実行記録が不正です")

    def save(self, case: dict, values: dict, *, seconds: float, peak_cuda_bytes: int,
             before_publish: Callable[[], None] | None = None) -> dict:
        verify_source(case)
        final = self.output / case["case_id"]
        if final.exists():
            raise ValueError("既存の推論ケースを上書きしません")
        if set(values) != set(self.roles):
            raise ValueError("固定stemすべての推論値が必要です")
        record = {"schema_version": 2, "case_id": case["case_id"], "identity": self.identity,
                  "source_sha256": case["source"]["sha256"], "source_hash_after_verified": True,
                  "seconds": seconds, "peak_cuda_bytes": peak_cuda_bytes, "stems": {}}
        if "inference_variant" in case:
            record["inference_variant"] = case["inference_variant"]
            record["inference_variant_id"] = InferenceVariantRecipe(**case["inference_variant"]).recipe_id
        self._validate_stats(record)
        staging = Path(tempfile.mkdtemp(prefix=f".partial-{case['case_id']}-", dir=self.output))
        try:
            for role in self.roles:
                audio = np.asarray(values[role])
                if audio.shape != (case["source"]["frames"], 2) or not np.isfinite(audio).all():
                    raise ValueError("BandItが不正なstemを返しました")
                path = staging / f"{role}.wav"
                sf.write(path, audio, 44100, subtype="FLOAT")
                record["stems"][role] = {"path": str((final / path.name).resolve()),
                                         "sha256": file_sha256(path),
                                         # 外部の再生でtagが書き戻されても推論をやり直さない。
                                         "audio": audio_identity.audio_identity(path)}
            staged = {role: {**item, "path": str((staging / f"{role}.wav").resolve())}
                      for role, item in record["stems"].items()}
            validate_stems(case, staged, directory=staging, roles=self.roles)
            verify_source(case)
            _write_json_atomic(staging / "case.json", record)
            if before_publish:
                before_publish()
            verify_source(case)
            os.rename(staging, final)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return record

    def complete(self, *, verify_identity: Callable[[], None]) -> Path:
        verify_identity()
        records = [self.load(case) for case in self.plan["cases"]]
        if any(record is None for record in records):
            raise ValueError("全固定ケースの検証済みCUDA記録がそろっていません")
        verify_identity()
        path = self.output / "receipt.json"
        receipt = {"schema_version": 2, "status": "COMPLETED_CUDA_WITHOUT_FALLBACK",
                   **self.identity, "cases": [{k: v for k, v in record.items() if k != "identity"}
                                              for record in records]}
        if path.exists():
            if read_json(path) != receipt:
                raise ValueError("既存の最終完了記録が固定ケースと一致しません")
        else:
            _write_json_atomic(path, receipt)
        return path

    def inspect(self) -> list[dict]:
        """固定caseを再利用可・未推論・不正へ分類する。良い結果は読むだけで動かさない。"""
        rows = []
        for case in self.plan["cases"]:
            case_id, directory = case["case_id"], self.output / case["case_id"]
            try:
                verify_source(case)
            except Exception as exc:
                rows.append({"case_id": case_id, "state": "INVALID_SOURCE", "reason": _describe(exc)})
                continue
            if not directory.exists():
                rows.append({"case_id": case_id, "state": "MISSING"})
                continue
            try:
                record = self.load(case)
            except Exception as exc:
                rows.append({"case_id": case_id, "state": "INVALID_OUTPUT", "reason": _describe(exc),
                             "files": _directory_digest(directory)})
            else:
                rows.append({"case_id": case_id, "state": "REUSABLE",
                             "stems": {role: item["sha256"] for role, item in record["stems"].items()}})
        return rows

    @staticmethod
    def _relocate(source: Path, target: Path, digest) -> None:
        """削除せず移すだけの回復。移動前後の内容hashが記録と一致することを要求する。"""
        def observed(path: Path):
            return file_sha256(path) if isinstance(digest, str) else _directory_digest(path)
        if target.exists():
            if source.exists():
                raise ValueError(f"隔離元と隔離先の両方に同じ推論出力があります: {source}")
            if observed(target) != digest:
                raise ValueError(f"隔離済みの内容が記録と一致しません: {target}")
            return
        if observed(source) != digest:
            raise ValueError(f"隔離する内容が記録と一致しません: {source}")
        os.rename(source, target)
        if observed(target) != digest:
            raise ValueError(f"隔離後の内容が記録と一致しません: {target}")

    def _freeze_quarantine(self, directory: Path, record_path: Path, *, reason: str) -> dict:
        rows = self.inspect()
        changed = [row["case_id"] for row in rows if row["state"] == "INVALID_SOURCE"]
        if changed:
            # 固定入力が変わった比較は、再推論では同じ条件へ戻せない。
            raise ValueError(f"固定sourceが変わった推論は隔離で回復できません: {changed}")
        cases = [row for row in rows if row["state"] == "INVALID_OUTPUT"]
        if not cases:
            raise ValueError("隔離する不正な推論出力がありません")
        receipt_path = self.output / "receipt.json"
        record = {"schema_version": 1, "status": "FROZEN_BEFORE_MOVE", "deletes_nothing": True,
                  "reason": reason, "output": str(self.output.resolve()), "identity": self.identity,
                  "plan_cases": [case["case_id"] for case in self.plan["cases"]], "cases": cases,
                  "reusable_cases": [row["case_id"] for row in rows if row["state"] == "REUSABLE"],
                  "missing_cases": [row["case_id"] for row in rows if row["state"] == "MISSING"],
                  "receipt": {"path": str(receipt_path.resolve()), "sha256": file_sha256(receipt_path)}
                             if receipt_path.exists() else None}
        directory.mkdir(parents=True)
        _write_json_atomic(record_path, record)
        return record

    def quarantine_invalid(self, directory: Path, *, reason: str) -> dict:
        """不正caseと旧receiptを内容hash付きで隔離し、不足caseだけ再推論できる状態にする。"""
        directory, output = Path(directory).resolve(), self.output.resolve()
        if directory.is_relative_to(output) or output.is_relative_to(directory):
            raise ValueError("隔離先は推論出力の外である必要があります")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("隔離の理由が必要です")
        record_path = directory / "quarantine.json"
        if directory.exists():
            # 移動の途中で落ちた隔離だけを、同じ固定条件と理由で再開する。
            record = read_json(record_path)
            if (record["identity"] != self.identity or record["output"] != str(output)
                    or record["reason"] != reason
                    or record["plan_cases"] != [case["case_id"] for case in self.plan["cases"]]
                    or record["status"] not in ("FROZEN_BEFORE_MOVE", "QUARANTINED_WITHOUT_DELETION")):
                raise ValueError("中断した同じ隔離の記録が必要です")
        else:
            record = self._freeze_quarantine(directory, record_path, reason=reason)
        for entry in record["cases"]:
            self._relocate(self.output / entry["case_id"], directory / entry["case_id"], entry["files"])
        if record["receipt"] is not None:
            self._relocate(self.output / "receipt.json", directory / "receipt.json",
                           record["receipt"]["sha256"])
        quarantined = {entry["case_id"] for entry in record["cases"]}
        after = self.inspect()
        if any(row["state"] != "REUSABLE" and row["state"] != "MISSING" for row in after):
            raise ValueError("隔離しても不正な推論出力が残っています")
        if [row["case_id"] for row in after if row["state"] == "REUSABLE"] != record["reusable_cases"]:
            raise ValueError("隔離で再利用できる推論結果が失われました")
        missing = [case["case_id"] for case in self.plan["cases"]
                   if case["case_id"] in quarantined or case["case_id"] in set(record["missing_cases"])]
        if [row["case_id"] for row in after if row["state"] == "MISSING"] != missing:
            raise ValueError("隔離後に再推論が必要なケースが一致しません")
        if (self.output / "receipt.json").exists():
            raise ValueError("旧receiptが隔離されていません")
        final = {**record, "status": "QUARANTINED_WITHOUT_DELETION",
                 "quarantined_case_count": len(record["cases"]),
                 "reinference_required_cases": missing}
        _write_json_atomic(record_path, final)
        return final


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _directory_digest(directory: Path) -> dict:
    """隔離の前後で同じ内容であることを示す、相対pathごとの内容hash。"""
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"通常ファイルでない推論出力があります: {path}")
        files[path.relative_to(directory).as_posix()] = file_sha256(path)
    if not files:
        raise ValueError(f"内容のない推論出力は隔離できません: {directory}")
    return files
