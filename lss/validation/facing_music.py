"""公開Facing the Music層と名前付き4stem重みのアダプター（製品moduleの再export）。

製品のBGM歌唱回収（`lss.bgm.four_stem_analysis`）と同じ実装を、Development probeからも
同じ名前で使う。処理本体を二重に持たない。
"""
from __future__ import annotations

from lss.bgm.four_stem_analysis import MODEL_CONFIG, ROLES, band_features, load_model, renamed_state

__all__ = ["MODEL_CONFIG", "ROLES", "band_features", "load_model", "renamed_state"]
