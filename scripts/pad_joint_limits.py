"""Widen every joint's stored limits slightly, so a resting arm stays inside them.

A servo parked outside its own Min/Max_Position_Limit stops answering the bus
entirely, and the next connect reports it as a missing motor - a confusing symptom
for what is really a joint sitting a couple of degrees too far.

That is easy to arrive at. Calibration records the range by moving each joint to
its ends, so the recorded bounds sit exactly where the arm can rest; then torque
goes off between runs and gravity settles a joint a few ticks past one of them.
Measured here: shoulder_lift resting 19 ticks inside its lower bound, elbow_flex
5 ticks inside its upper one, and the gripper four ticks *outside*.

Padding costs nothing. The limits exist to keep inverse kinematics away from the
hard stops, and a couple of degrees either way does not change that - the stops
themselves are further out than the calibrated range in the first place.

    uv run scripts/pad_joint_limits.py --report
    uv run scripts/pad_joint_limits.py                 # apply the padding
    uv run scripts/pad_joint_limits.py --pad-deg 5
"""

import argparse
import json
from pathlib import Path

from so101.platform import require_windows

require_windows()

from so101.hardware import resolve as resolve_port  # noqa: E402
from so101.hardware.sts3215 import (  # noqa: E402
    Bus,
    JOINT_NAMES,
    MAX_ANGLE_LIMIT,
    MIN_ANGLE_LIMIT,
    PRESENT_POSITION,
    TICKS_PER_DEG,
)

CALIBRATION = (Path.home() / ".cache/huggingface/lerobot/calibration/robots"
               / "so_follower/follower.json")
DEFAULT_PAD_DEG = 3.0
ENCODER_MIN, ENCODER_MAX = 0, 4095


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--pad-deg", type=float, default=DEFAULT_PAD_DEG)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    pad = int(args.pad_deg * TICKS_PER_DEG)
    calibration = {}
    if args.calibration.is_file():
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))

    print(f"  padding by {args.pad_deg:.0f} deg ({pad} ticks) each way\n")
    print(f"  {'joint':<15}{'now':>6}{'before':>14}{'after':>14}{'margin':>9}")

    changed = False
    with Bus(resolve_port([args.port])[0]) as bus:
        for sid, name in JOINT_NAMES.items():
            low = bus.read(sid, MIN_ANGLE_LIMIT, 2)
            high = bus.read(sid, MAX_ANGLE_LIMIT, 2)
            position = bus.read(sid, PRESENT_POSITION, 2)
            if None in (low, high, position):
                print(f"  {name:<15}   could not read")
                continue

            padded = (max(ENCODER_MIN, low - pad), min(ENCODER_MAX, high + pad))
            margin = min(position - padded[0], padded[1] - position)
            print(f"  {name:<15}{position:>6}{f'{low}-{high}':>14}"
                  f"{f'{padded[0]}-{padded[1]}':>14}"
                  f"{margin / TICKS_PER_DEG:>7.1f}d")

            if args.report or padded == (low, high):
                continue
            bus.write(sid, MIN_ANGLE_LIMIT, padded[0], size=2)
            bus.write(sid, MAX_ANGLE_LIMIT, padded[1], size=2)
            if name in calibration:
                calibration[name]["range_min"] = padded[0]
                calibration[name]["range_max"] = padded[1]
            changed = True

    if args.report:
        return
    if changed and calibration:
        args.calibration.write_text(json.dumps(calibration, indent=4) + "\n",
                                    encoding="utf-8")
        print(f"\n  servos and {args.calibration.name} updated")
    elif changed:
        print("\n  servos updated")
    else:
        print("\n  nothing to change")


if __name__ == "__main__":
    main()
