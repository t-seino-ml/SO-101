#!/usr/bin/env bash
# Launch SO-101 teleoperation with the settings that tested best on this rig.
# Override with env vars, e.g. FPS=60 ./teleop.sh, or pass extra flags through:
#   ./teleop.sh --robot.max_relative_target=5
set -euo pipefail
cd "$(dirname "$0")"

exec ./.venv/Scripts/python.exe teleoperate.py \
  --robot.type=so101_follower --robot.port="${FOLLOWER_PORT:-COM4}" --robot.id=follower \
  --teleop.type=so101_leader --teleop.port="${LEADER_PORT:-COM3}" --teleop.id=leader \
  --fps="${FPS:-120}" "$@"
