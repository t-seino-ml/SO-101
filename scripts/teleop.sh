#!/usr/bin/env bash
# Launch SO-101 teleoperation with the settings that tested best on this rig.
#
# Calls the Windows Python directly rather than `uv run`. From a WSL shell `uv` is
# Linux uv: it would delete the Windows .venv and rebuild a Linux one that cannot
# see the COM ports or the DirectShow cameras. Calling the .exe works from both
# WSL and Git Bash.
#
# Override with env vars, e.g. FPS=60 ./scripts/teleop.sh, or pass flags through:
#   ./scripts/teleop.sh --max-relative-target 5
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="./.venv/Scripts/python.exe"
if [ ! -x "$PYTHON" ]; then
  echo "No Windows virtualenv at $PYTHON." >&2
  echo "Rebuild it from PowerShell with 'uv sync' (not from WSL)." >&2
  exit 1
fi

exec "$PYTHON" scripts/teleop_view.py \
  --follower-port "${FOLLOWER_PORT:-COM4}" \
  --leader-port "${LEADER_PORT:-COM3}" \
  --fps "${FPS:-200}" "$@"
