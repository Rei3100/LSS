"""採用条件の判定（D-6②・D-7・T310 spec §6 Step3）。

呼び出し側（scripts/score_models.py）は、models.tsvの実測値・フィルタ結果を
次のキーを持つdictへ変換してから渡す:
  - 決定変数: sdr, l1_freq, aura_mrstft（3指標とも試験曲平均）
  - 資源計測: rtf, peak_vram_mb（記録・比較用。品質候補のhard gateにはしない）
  - 互換性結果: f4_ok（exact provider/runtimeで実行できること）
"""
from __future__ import annotations

DECISION_METRICS: tuple[str, ...] = ("sdr", "l1_freq", "aura_mrstft")

# D-7: 単一指標のみ改善→1.0以上／3指標同方向改善→0.3以上。
SINGLE_METRIC_THRESHOLD = 1.0
ALL_METRIC_THRESHOLD = 0.3

# 旧実装との診断互換用の観測基準。採用可否には使わない。
RTF_LIMIT = 3.0
PEAK_VRAM_MB_LIMIT = 12 * 1024


def is_significant(candidate: dict, baseline: dict) -> bool:
    """D-7二段判定: 単一指標のみ改善→1.0以上／3指標同方向改善→0.3以上。

    それ以外（例: ちょうど2指標だけが改善）は「差あり」とはみなさない
    （spec D-7が定義する2ケースのいずれにも該当しないため保守的にFalseを返す）。
    """
    diffs = {m: candidate[m] - baseline[m] for m in DECISION_METRICS}
    improved = {m: d for m, d in diffs.items() if d > 0}

    if len(improved) == len(DECISION_METRICS):
        return all(d >= ALL_METRIC_THRESHOLD for d in improved.values())
    if len(improved) == 1:
        (only_diff,) = improved.values()
        return only_diff >= SINGLE_METRIC_THRESHOLD
    return False


def passes_adoption(candidate_row: dict, baseline_row: dict) -> bool:
    """比較品質とexact runtime互換性を確認する。

    条件1（3指標すべてで既定モデルを下回らない）がまず、他指標が悪化した
    「1指標だけブーストした候補」を機械的に弾く役目を持つ。
    RTF/VRAMは品質同等候補間の比較材料であり、優れた候補を落とす条件ではない。
    匿名取得可否や手動工程の有無も、再現可能な取得/runtimeへ実装できるかとは分けて扱う。
    """
    all_metrics_not_below_baseline = all(
        candidate_row[m] >= baseline_row[m] for m in DECISION_METRICS
    )
    if not all_metrics_not_below_baseline:
        return False

    if not is_significant(candidate_row, baseline_row):
        return False

    if not candidate_row["f4_ok"]:
        return False

    return True
