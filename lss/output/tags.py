"""D-8のタグ3分類+来歴7項目を組み立てる純粋関数群(I/Oはwriter.pyが担う)。

TITLE/TRACKは絶対に改変しない(I20/R3)。状態語(ステム名・来歴)は専用のLSS_*タグへ逃がす。
"""
from __future__ import annotations

import json

from lss.core.recipe import validate_recipe_revision

INHERITED_KEYS = (
    "ARTIST", "ALBUM", "ALBUMARTIST", "DATE", "GENRE", "TRACKNUMBER", "DISCNUMBER", "VOCAL_TYPE",
)
IMMUTABLE_KEYS = ("TITLE", "TRACK")  # I20/R3: 元のまま変更しない

REPLAYGAIN_ALGORITHM_VALUE = "ITU-R BS.1770"
TagValue = str | list[str]


def inherited_tags(source_tags: dict[str, TagValue]) -> dict[str, TagValue]:
    """曲の属性(分離で変わらない)＋TITLE/TRACKをそのままコピーする(I19前半・I21・I20/R3)。"""
    keys = INHERITED_KEYS + IMMUTABLE_KEYS
    return {key: source_tags[key] for key in keys if source_tags.get(key)}


def recalculated_tags(scan: dict) -> dict[str, str]:
    """scan_track()の実測値からD-8「再計算する」タグ群を組み立てる(I23・I24のタグ実例)。

    単位はfoobar2000のタグ実例(I24)に合わせる: GAIN/PEAKはdB/無単位、
    RANGE(LRA)とPLRはLU(Loudness Units)——LRA・PLRは「ラウドネス基準からの差分」を表す単位で
    dBと数値上は等価だが、この2項目だけ実例がLU表記のため合わせる。
    """
    result = {
        "DYNAMIC RANGE": f"{scan['dynamic_range_db']:.0f}",
        "REPLAYGAIN_ALGORITHM": REPLAYGAIN_ALGORITHM_VALUE,
        "REPLAYGAIN_TRACK_GAIN": f"{scan['track_gain_db']:.2f} dB",
        "REPLAYGAIN_TRACK_PEAK": f"{scan['sample_peak_linear']:.6f}",
        "REPLAYGAIN_TRACK_RANGE": f"{scan['loudness_range_lu']:.2f} LU",
        "TRUEPEAK_SCANNER_CLIPPED_SAMPLES_TRACK": str(scan["clipped_samples"]),
        "TRUEPEAK_SCANNER_PEAK_POSITION": _format_position(scan["true_peak_position_sample"], scan.get("samplerate") or 1),
        "TRUEPEAK_SCANNER_TRACK_PLR": f"{scan['plr_db']:.2f} LU",
    }
    if not scan.get("loudness_measured", True):
        for key in ("REPLAYGAIN_ALGORITHM", "REPLAYGAIN_TRACK_GAIN", "REPLAYGAIN_TRACK_RANGE",
                    "TRUEPEAK_SCANNER_TRACK_PLR"):
            result.pop(key)
    return result


def _format_position(sample_index: int, samplerate: int) -> str:
    """ピーク位置を`M:SS.mmm`形式のタイムコードで表す(TruePeak Scanner系コンポーネントの慣例書式)。"""
    total_seconds = sample_index / samplerate
    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60
    return f"{minutes}:{seconds:06.3f}"


def new_tags(stem: str, source_type: str = "separated") -> dict[str, str]:
    """状態語の実体(I19後半「全く同じはNG」)。"""
    return {"LSS_STEM": stem, "LSS_SOURCE_TYPE": source_type}


def provenance_tags(
    *, recipe_id: str, model: str, model_sha256: str, processed_at: str,
    input_sha256: str, score: str, version: str,
    config_sha256: str | None = None,
    runtime_revision: dict | None = None,
    recipe_revision: dict | None = None,
    headroom_shift: int | None = None,
) -> dict[str, str]:
    """基本来歴と、得られた場合は設定・実行環境の完全な機械可読来歴を返す。

    headroom_shiftは直列化時に宣言した2の冪の減衰量(指数)。元のfloat値は
    `2**shift`倍で完全に復元できるため、無音の正規化ではなく可逆な表現選択として記録する。
    """
    result = {
        "LSS_RECIPE_ID": recipe_id,
        "LSS_MODEL": model,
        "LSS_MODEL_SHA256": model_sha256,
        "LSS_PROCESSED_AT": processed_at,
        "LSS_INPUT_SHA256": input_sha256,
        "LSS_SCORE": score,
        "LSS_VERSION": version,
    }
    if headroom_shift is not None:
        if not isinstance(headroom_shift, int) or isinstance(headroom_shift, bool) or headroom_shift < 0:
            raise ValueError(f"headroom_shiftが不正です: {headroom_shift}")
        result["LSS_OUTPUT_HEADROOM_SHIFT"] = str(headroom_shift)
    if config_sha256:
        result["LSS_CONFIG_SHA256"] = config_sha256
    if runtime_revision is not None:
        runtime_id = runtime_revision.get("runtime_id")
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ValueError("runtime_revisionにruntime_idがありません")
        result["LSS_RUNTIME_ID"] = runtime_id
        result["LSS_RUNTIME_REVISION"] = json.dumps(
            runtime_revision, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
    if recipe_revision is not None:
        validated_recipe = validate_recipe_revision(recipe_revision)
        if validated_recipe["recipe_id"] != recipe_id:
            raise ValueError("recipe_revisionとrecipe_idが一致しません")
        result["LSS_RECIPE_REVISION"] = json.dumps(
            validated_recipe, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
    return result


def build_tags(
    *, source_tags: dict[str, TagValue], stem: str, scan: dict, provenance: dict[str, str],
    source_type: str = "separated",
) -> dict[str, TagValue]:
    """3分類＋来歴7項目を1つの出力タグ辞書へ統合する(アートワークはバイナリのため対象外・writer.pyが別途複製)。"""
    tags: dict[str, TagValue] = {}
    tags.update(inherited_tags(source_tags))
    tags.update(recalculated_tags(scan))
    tags.update(new_tags(stem, source_type))
    tags.update(provenance)
    return tags
