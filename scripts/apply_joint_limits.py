"""Write measured joint limits into the servos, and into LeRobot's calibration.

LeRobot's calibration treats wrist_roll as a continuous joint and gives it the
whole encoder, 0-4095. On this arm it is not: something stops it about 265 degrees
round. A stored range wider than the joint's real travel is worse than none -
inverse kinematics asks for angles the arm cannot reach, the servo leans on its
stop at full load, the overload protection latches, and the next connect reports
the servo as missing.

`easy_calibrate.py` resets the limits every time it runs, so this has to be run
after it, not before.

Limits live in two places and both matter: the servo's own Min/Max_Position_Limit
registers, which stop the hardware, and LeRobot's calibration file, which is what
normalises the angles. Writing only one leaves them disagreeing.

    uv run scripts/apply_joint_limits.py                # apply the measured limits
    uv run scripts/apply_joint_limits.py --report       # show, change nothing
"""

import argparse
import json
import time
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

# Measured by stepping each joint into its stop and watching the load pin at 1000.
# Only joints that genuinely travel less than their calibration claims belong here.
MEASURED_LIMITS = {
    # Downwards the joint stops hard at 1065, so the lower bound sits clear of it.
    # Upwards there is no stop at all - it runs to the encoder's end - so capping
    # it below 4095 only creates a way to end up outside the limits: the joint
    # reached 4091 against a 4084 ceiling and stopped answering the bus.
    "wrist_roll": (1105, 4095),
    # The jaws meet at 1718 and open past 2572 without resistance. Calibration
    # recorded 1726-2369, which is barely half the travel - and a gripper that
    # cannot open far enough cannot get around a 25 mm cube at all.
    #
    # The lower bound sits well clear of the stop rather than just inside it. At
    # 1740 the jaws settled to 1736 once released - four ticks outside - and a
    # servo parked outside its limits stops answering entirely, which reads as a
    # missing motor at the next connect.
    "gripper": (1680, 2600),
}


def _drive_into_range(bus, sid, position, low, high, speed=200):
    """Ramp a joint to the middle of its new range, then release it."""
    from so101.hardware.sts3215 import GOAL_POSITION, TORQUE_ENABLE

    bus.write(sid, GOAL_POSITION, position, size=2)
    bus.torque(sid, True)
    time.sleep(0.15)
    middle = (low + high) // 2
    steps = max(1, abs(middle - position) // 20)
    for step in range(1, steps + 1):
        bus.move_to(sid, position + (middle - position) * step / steps, speed=speed)
        time.sleep(0.05)
    time.sleep(0.8)
    for _ in range(5):
        bus.write(sid, TORQUE_ENABLE, 0)
        if bus.read(sid, TORQUE_ENABLE, 1) == 0:
            break
    return bus.read(sid, PRESENT_POSITION, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--report", action="store_true",
                        help="show what would change and stop")
    args = parser.parse_args()

    by_name = {name: sid for sid, name in JOINT_NAMES.items()}
    print(f"  {'joint':<14}{'servo now':>14}{'measured':>14}{'file now':>14}")

    calibration = {}
    if args.calibration.is_file():
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    else:
        print(f"  {args.calibration} not found; only the servos will change")

    changed_servo, changed_file = [], []
    with Bus(resolve_port([args.port])[0]) as bus:
        for name, (low, high) in MEASURED_LIMITS.items():
            sid = by_name[name]
            servo = (bus.read(sid, MIN_ANGLE_LIMIT, 2), bus.read(sid, MAX_ANGLE_LIMIT, 2))
            entry = calibration.get(name, {})
            stored = (entry.get("range_min"), entry.get("range_max"))
            print(f"  {name:<14}{f'{servo[0]}-{servo[1]}':>14}"
                  f"{f'{low}-{high}':>14}{f'{stored[0]}-{stored[1]}':>14}")

            if args.report:
                continue

            # A joint parked outside its new limits stops answering entirely, and
            # then reads as a missing motor at the next connect. Drive it inside
            # first, while the old wider limits still permit the motion.
            position = bus.read(sid, PRESENT_POSITION, 2)
            if position is not None and not (low <= position <= high):
                print(f"    {name} is at {position}, outside {low}-{high}; "
                      "moving it inside first")
                _drive_into_range(bus, sid, position, low, high)

            if servo != (low, high):
                bus.write(sid, MIN_ANGLE_LIMIT, low, size=2)
                bus.write(sid, MAX_ANGLE_LIMIT, high, size=2)
                changed_servo.append(name)
            if entry and stored != (low, high):
                entry["range_min"], entry["range_max"] = low, high
                changed_file.append(name)

    if args.report:
        return

    if changed_file:
        args.calibration.write_text(json.dumps(calibration, indent=4) + "\n",
                                    encoding="utf-8")

    span = {name: (high - low) / TICKS_PER_DEG
            for name, (low, high) in MEASURED_LIMITS.items()}
    for name in MEASURED_LIMITS:
        print(f"\n  {name}: {span[name]:.0f} deg of travel")
        print(f"    servo registers {'updated' if name in changed_servo else 'already right'}")
        print(f"    calibration file {'updated' if name in changed_file else 'already right'}")

    print("\n  Re-run scripts/calibrate.py: the camera-to-arm homography was fitted")
    print("  against the old normalisation, so forward kinematics reports different")
    print("  positions now.")


if __name__ == "__main__":
    main()
