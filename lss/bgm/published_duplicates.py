"""公開しようとする劇伴の区間が、別の原本で既に公開したcueと同じ組み立て済み音声なら、重複として公開しない。

配信版の冒頭の告知や、話をまたいで編集ごと繰り返す部分（変身の場面など）は、台詞・ナレーション・効果音まで
含めて同じ原本mixが別の話にも現れる。単話の中ではそれが本編の劇伴かどうか決められない（最初に処理した話では
公開されうる）が、同じ状態領域で先に処理した別の原本の公開cueと同じ音なら、新しいcueではない。

- 材料は、同じ状態領域の完了run（別の原本、公開cueあり）の、公開cueを作った保留候補の原本区間の音声。
  指紋は記録済みの内容SHA-256で保存・再利用する（`identity` の指紋cache）。
- 相手の公開cueは、その区間を作った保留候補の原本区間の音声から1本の信号として組み立てて照合する。
- 同じ組み立て済み音声とみなすのは、`identity.match` の ``known_exact`` の一致のうち、波形の説明率が
  ``minimum_explained`` 以上（説明できない成分が約-20dB未満＝符号化の違い程度）のものだけ。同じ録音でも台詞・効果音が違う再使用は、説明できない成分が残るので含めない。
- WANTEDの区間の ``minimum_cover`` 以上をその一致が覆う時だけ重複にする。位置（冒頭何秒）は使わない。
- 照合できない（材料の欠落・破損・復号失敗）時は重複と言わない（従来どおり公開する）。公開を減らす方向だけに働き、
  HELDの区間を公開へ回すことはしない。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
METHOD = "published-cue-identical-assembled-audio-v1"
REASON = "identical_assembled_audio_already_published"


class PublishedDuplicatesError(ValueError):
    """重複判定の宣言か材料が契約に合わない。"""


@dataclass(frozen=True)
class DuplicateRecipe:
    minimum_explained: float = 0.99
    minimum_cover: float = 0.9
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise PublishedDuplicatesError("未対応のschemaです")
        for name, low, high in (("minimum_explained", 0.5, 1.0), ("minimum_cover", 0.5, 1.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= float(value) <= high:
                raise PublishedDuplicatesError(f"{name}は{low}以上{high}以下である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return f"bgm-published-duplicates-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def default_recipe() -> DuplicateRecipe:
    """値は実行前に固定した判定条件（`.state/validation/bgm-residual-gap-rank-r1/criteria.json`）のもの。"""
    return DuplicateRecipe()


@dataclass(frozen=True)
class PriorCue:
    """先に処理した別の原本の公開cue1つ。材料は、その区間を作った保留候補の原本区間の音声。"""

    run_key: str
    source_sha256: str
    index: int
    start_seconds: float                            # その原本の時間軸での公開cueの区間
    end_seconds: float
    members: tuple[tuple[str, str, float], ...]     # (原本区間の音声のパス, 内容SHA-256, その音声の先頭の時刻)

    @property
    def material_id(self) -> str:
        """指紋の保存に使う、材料（音声の内容と切り出す区間）の識別子。"""
        body = json.dumps({"members": [[sha, start] for _, sha, start in self.members],
                           "span": [self.start_seconds, self.end_seconds]}, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()


def prior_publications(candidates_root: Path, source_sha256: str, *,
                       adopted: dict[str, dict[str, set]] | None = None) -> tuple[list[PriorCue], list[str]]:
    """同じ状態領域の完了run（別の原本、公開cueあり）の公開cueを集める。読めないrun・材料の欠けたcueは使わない。

    ``adopted`` は原本ごとのCurrentBestの公開中のcue（`lss.bgm.current_best.adopted_runs`、run→cueの識別子）。
    記録のある原本は、今公開しているcueだけを数える（置き換えて履歴へ移した版・公開しなかった版は公開中ではない）。
    """
    cues: list[PriorCue] = []
    runs: list[str] = []
    for run_path in sorted(Path(candidates_root).glob("*/run/run.json")):
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
            other = (run.get("source") or {}).get("sha256")
            published = run.get("published") or []
            if (run.get("status") != "completed_published_cues" or not isinstance(other, str)
                    or other == source_sha256 or not published):
                continue
            current = (adopted or {}).get(other)
            if current is not None:
                keys = current.get(run_path.parent.parent.name)
                if not keys:
                    continue
                published = [item for item in published if item.get("key") in keys]
            held = {cue["cue_id"]: cue["raw"] for cue in run.get("held") or []}
            found = []
            for item in published:
                rate = int(item["sample_rate"])
                members = []
                for cue_id in item["candidates"]:
                    raw = held[cue_id]
                    if not Path(str(raw["path"])).is_file():
                        raise FileNotFoundError(raw["path"])
                    members.append((str(raw["path"]), str(raw["sha256"]),
                                    int(raw["source_start_frame"]) / int(raw["sample_rate"])))
                found.append(PriorCue(run_key=run_path.parent.parent.name, source_sha256=other,
                                      index=int(item["index"]), start_seconds=int(item["start_frame"]) / rate,
                                      end_seconds=int(item["end_frame"]) / rate,
                                      members=tuple(sorted(members, key=lambda member: member[2]))))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        if found:
            cues.extend(found)
            runs.append(run_path.parent.parent.name)
    return cues, runs


def identical_spans(query_samples, cues: list[PriorCue], recipe: DuplicateRecipe, *,
                    cache_dir: Path | None = None) -> list[dict]:
    """対象の原本mix（11025Hz mono）のうち、別の原本の公開cueと同じ組み立て済み音声の区間。

    相手の公開cueの区間を、その保留候補の原本区間の音声から1本の信号として組み立てて照合する（候補の境目で
    切らない）。材料の内容SHA-256が記録と違う・読めない公開cueは使わない。
    """
    import numpy as np

    from lss.bgm import identity
    from lss.harness.fetch import file_sha256

    recipe.validate()
    rate = identity.SAMPLE_RATE
    by_id = {cue.material_id: cue for cue in cues}
    checked: dict[str, bool] = {}

    def loader(location) -> Any:
        cue = by_id[str(location)]
        out = np.zeros(max(1, int(round((cue.end_seconds - cue.start_seconds) * rate))), dtype=np.float32)
        for path, sha256, part_start in cue.members:
            if path not in checked:
                checked[path] = file_sha256(Path(path)) == sha256
            if not checked[path]:
                raise PublishedDuplicatesError("先に公開したcueの材料が記録と一致しません")
            samples = identity.decode_mono(Path(path))
            offset = int(round((part_start - cue.start_seconds) * rate))
            low, high = max(0, offset), min(out.size, offset + samples.size)
            if high > low:
                out[low:high] = samples[low - offset:high - offset]
        return out

    items = []
    for cue in cues:
        cached = identity._load_fingerprint(cache_dir, cue.material_id) if cache_dir is not None else None
        if cached is None:
            try:
                samples = loader(cue.material_id)
            except (OSError, PublishedDuplicatesError, identity.DecodeError):
                continue  # 読めない・記録と違う材料は使わない（重複の根拠にしない）
            spec = identity.spectrogram(samples)
            hashes, times = identity.landmarks(*identity.peaks(spec))
            cached = (hashes, times, identity.band_profile(spec), spec.shape[0])
            if cache_dir is not None:
                identity._save_fingerprint(cache_dir, cue.material_id, cached)
        items.append((f"{cue.run_key}#{cue.index}", cue.material_id, cue.material_id, cached))
    if not items:
        return []
    index = identity.ReferenceIndex._from_fingerprints(items, loader)
    found = []
    for match in identity.match(query_samples, index):
        explained = match.waveform_explained or 0.0
        if match.identity_state != "known_exact" or explained < recipe.minimum_explained:
            continue
        cue = by_id.get(match.reference_path)
        if cue is None:
            continue
        found.append({"start_seconds": round(match.query_start_seconds, 3),
                      "end_seconds": round(match.query_end_seconds, 3),
                      "prior_run": cue.run_key, "prior_source_sha256": cue.source_sha256,
                      "prior_cue_index": cue.index,
                      "prior_start_seconds": round(cue.start_seconds + match.reference_start_seconds, 3),
                      "waveform_explained": round(float(explained), 4)})
    return found


def covered_fraction(start: float, end: float, spans: Iterable[dict]) -> float:
    if end <= start:
        return 0.0
    covered, cursor = 0.0, start
    for a, b in sorted((max(s["start_seconds"], start), min(s["end_seconds"], end)) for s in spans):
        a = max(a, cursor)
        if b > a:
            covered += b - a
            cursor = b
    return covered / (end - start)


def duplicate_of(start: float, end: float, spans: list[dict], recipe: DuplicateRecipe) -> list[dict] | None:
    """区間が重複なら、その根拠（覆った一致）を返す。そうでなければNone。"""
    inside = [s for s in spans if min(s["end_seconds"], end) - max(s["start_seconds"], start) > 0]
    if inside and covered_fraction(start, end, inside) >= recipe.minimum_cover:
        return inside
    return None
