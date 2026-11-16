"""完了後の動作の実行。凍結された4択の配線だけを持つ。

製品としての決定は閉じている（待機のまま／LSS 終了／PC 再起動／PC 終了）。
ここが決めるのは「選ばれた動作をどう実行するか」だけで、選択肢は増やさない。

実行はセッションが COMPLETE に達したときに1回だけ。開発・検証中に本当の
再起動や終了を起こさないよう、実行体は外から差し替えられる。
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from lss.ui.presentation import WhenDone


class Outcome(Enum):
    """実行の結果。UIに新しい面を作らないための最小の語彙。"""

    NOTHING = "nothing"  # 待機のまま。何もしない
    QUIT = "quit"  # LSS を終了した
    HANDED_TO_OS = "handed_to_os"  # OSへ要求した（再起動・終了）
    REFUSED = "refused"  # 権限などで実行できなかった。待機のまま残る


# WindowsのOS操作。LSSは自前で電源を扱わない。
_COMMANDS: dict[WhenDone, tuple[str, ...]] = {
    WhenDone.RESTART: ("shutdown", "/r", "/t", "0"),
    WhenDone.SHUTDOWN: ("shutdown", "/s", "/t", "0"),
}


def command_for(choice: WhenDone) -> tuple[str, ...] | None:
    """その選択がOSへ渡すコマンド。待機と終了はコマンドを持たない。"""
    return _COMMANDS.get(choice)


@dataclass
class Completion:
    """完了後の動作を1回だけ実行する。

    `runner` と `quit_app` を差し替えれば、実際のOS操作なしで検証できる。
    """

    runner: Callable[[tuple[str, ...]], int] | None = None
    quit_app: Callable[[], None] | None = None
    performed: bool = False

    def reset(self) -> None:
        """次のセッションのために実行済みの印を落とす。"""
        self.performed = False

    def perform(self, choice: WhenDone) -> Outcome:
        """選ばれた動作を実行する。同じ完了で二度実行しない。"""
        if self.performed:
            return Outcome.NOTHING
        self.performed = True
        if choice is WhenDone.NONE:
            return Outcome.NOTHING
        if choice is WhenDone.QUIT:
            self._quit()
            return Outcome.QUIT
        command = command_for(choice)
        if command is None:
            return Outcome.NOTHING
        try:
            code = self._run(command)
        except (OSError, ValueError):
            # 権限が無い・コマンドが無い。待機のまま残す（黙って落とさない）。
            return Outcome.REFUSED
        return Outcome.HANDED_TO_OS if code == 0 else Outcome.REFUSED

    def _run(self, command: tuple[str, ...]) -> int:
        if self.runner is not None:
            return self.runner(command)
        return subprocess.run(command, check=False, capture_output=True).returncode

    def _quit(self) -> None:
        if self.quit_app is not None:
            self.quit_app()
            return
        sys.exit(0)  # pragma: no cover - 画面の終了経路を渡されなかったとき
