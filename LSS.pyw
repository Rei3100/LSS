#!python3.11
"""ダブルクリック起動用ランチャー（コンソール窓を出さない・開発ツリー専用）。"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("lss_gui.py")), run_name="__main__")
