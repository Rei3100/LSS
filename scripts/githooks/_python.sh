# Sourced by the hooks: sets $LSS_PY to the repository's venv python (main checkout's venv from a linked worktree).
ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
COMMON=$(git rev-parse --git-common-dir 2>/dev/null)
MAIN=$(cd "$COMMON/.." 2>/dev/null && pwd)
LSS_PY=
for cand in "$ROOT/.venv/Scripts/python.exe" "$MAIN/.venv/Scripts/python.exe" "$ROOT/.venv/bin/python" "$MAIN/.venv/bin/python"; do
    if [ -x "$cand" ]; then LSS_PY=$cand; break; fi
done
if [ -z "$LSS_PY" ]; then
    LSS_PY=$(command -v python3 || command -v python)
fi
if [ -z "$LSS_PY" ]; then
    echo "lss githook: no python interpreter found (expected .venv/Scripts/python.exe)" >&2
    exit 1
fi
