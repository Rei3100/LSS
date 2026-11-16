"""commit-msg hook: the subject line is English.

Code, commits, Evidence and internal docs are English. Rule: the first non-comment line of the message
(the subject) may contain Japanese kana or kanji only inside a quoted span written with backticks, double quotes or
single quotes, which is how a Japanese target string is named. Anything else is rejected. Body lines are not checked.
Standard library only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

JAPANESE = re.compile("[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]")
QUOTED = re.compile(r"`[^`]*`|\"[^\"]*\"|'[^']*'")


def subject_of(message: str) -> str:
    for line in message.splitlines():
        if line.strip() and not line.startswith("#"):
            return line
    return ""


def violation(message: str) -> str | None:
    """Return the offending subject, or None when the message follows the rule."""
    subject = subject_of(message)
    return subject if JAPANESE.search(QUOTED.sub("", subject)) else None


def main(argv: list[str]) -> int:
    if not argv:
        return 0
    message = Path(argv[0]).read_text(encoding="utf-8", errors="replace")
    bad = violation(message)
    if bad:
        print("commit-msg: the subject line must be English (Japanese is allowed only inside `backticks` or quotes "
              f"naming a target string):\n  {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
