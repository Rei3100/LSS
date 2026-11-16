"""新しいUIから利用するモデル一覧・既定モデル・出力パートの参照。"""
from lss.core import model_config, models_store
from lss.harness import fetch


def list_models() -> list[dict]:
    _, rows = models_store.load_model_ledger(models_store.MODELS_TSV)
    visible = [row for row in rows if row.get("状態") in {"既定", "候補"}]
    return sorted(visible, key=lambda row: row.get("状態") != "既定")


def default_model() -> dict:
    """台帳で採用済みの既定だけを使う。候補を自動的に既定へ昇格させない。"""
    defaults = [row for row in list_models() if row.get("状態") == "既定"]
    if len(defaults) != 1:
        raise ValueError("モデル台帳に既定モデルが1件登録されている必要があります")
    return defaults[0]


def get_model(model_id: str) -> dict:
    row = next((row for row in list_models() if row.get("model_id") == model_id), None)
    if row is None:
        raise ValueError(f"利用可能なモデルが見つかりません: {model_id}")
    return row


# 現在の製品が満たす意味的な結果。凍結DesignもINSTRUMENTAL(伴奏)だけを示す。
# ここが増えるのは、対応するCapabilityが検証を通ってからで、モデルの出力数ではない。
INSTRUMENTAL = "Instrumental"


def requested_stem(row: dict, target: str = INSTRUMENTAL) -> str:
    """意味的な結果を、そのモデルが実際に出力するパート名へ解決する。

    duality系(Vocals/Instrumental)のモデルでは、要求された成果物は伴奏側だけで、
    相補のボーカルはRecipeが内部で作る途中生成物にすぎない。モデルが何本出すかは
    製品の成果物の本数を決めない。
    """
    targets = stem_targets(row)
    for name in targets:
        if name.casefold() == target.casefold():
            return name
    raise ValueError(
        f"このモデルは{target}を出力しません: {row.get('model_id')} (出力: {', '.join(targets)})"
    )


def stem_targets(row: dict) -> list[str]:
    """設定だけを取得し、モデルが実際に出力するパートを返す。重みは取得しない。"""
    config_url = row.get("config_url", "")
    if not config_url.startswith("https://"):
        raise ValueError(f"モデル設定の取得元が未登録です: {row['model_id']}")
    config_path = fetch.model_asset_path(row["model_id"], config_url)
    config = model_config.load_cached_config(
        config_path, lambda: fetch.download(
            config_url, config_path, max_bytes=fetch.MAX_CONFIG_BYTES,
            expected_sha256=row.get("config_sha256", "").strip().lower() or None,
        ),
    )
    training = config.get("training", {}) if isinstance(config, dict) else {}
    if not isinstance(training, dict):
        raise ValueError("モデルのパート設定が不正です")
    targets = [training["target_instrument"]] if training.get("target_instrument") else training.get("instruments", [])
    if not isinstance(targets, list) or not targets or not all(isinstance(t, str) and t.strip() for t in targets):
        raise ValueError("モデルの出力パートを確認できません")
    return list(dict.fromkeys(targets))
