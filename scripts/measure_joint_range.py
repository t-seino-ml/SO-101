"""Find a joint's real travel by feeling for its hard stops, and record it.

A calibrated range that claims more travel than the joint has is worse than no
calibration: inverse kinematics asks for angles the arm cannot reach, the servo
pushes against its stop at full load, and the overload protection latches - which
then reads as "motor missing" the next time anything connects.

That is exactly what wrist_roll was doing here. Its stored range said a full 360
degrees; it actually stops near -86 on one side.

The joint is walked outwards in small steps and the load watched. A stop shows up
as load climbing while the position no longer changes, at which point the search
backs off immediately rather than leaning on it.

    uv run scripts/measure_joint_range.py wrist_roll
    uv run scripts/measure_joint_range.py wrist_roll --apply
    uv run scripts/measure_joint_range.py --all --dry-run
"""

import argparse
import time

from so101.platform import require_windows

require_windows()

from so101.hardware import resolve as resolve_port  # noqa: E402
from so101.hardware.sts3215 import (  # noqa: E402
    Bus,
    GOAL_POSITION,
    JOINT_NAMES,
    MAX_ANGLE_LIMIT,
    MIN_ANGLE_LIMIT,
    PRESENT_POSITION,
    TICKS_PER_DEG,
    TORQUE_ENABLE,
)

STEP_TICKS = 40           # a full sweep is 4096 ticks; smaller steps take minutes
SETTLE = 0.30             # the joint needs time to actually get there
SEARCH_TIMEOUT = 60.0     # seconds per direction
STALL_LOAD = 450          # of 1023: the joint is pressing on something
STALL_TICKS = 12          # ... and has fallen this far behind the command
BACKOFF_TICKS = 45        # retreat this far once a stop is found
SAFETY_MARGIN = 25        # keep the recorded limit inside the real stop
SPEED = 200
ENCODER_MIN, ENCODER_MAX = 0, 4095


def read_position(bus, sid, attempts=5):
    """Present_Position, retried. A dropped packet is not a missing joint."""
    for _ in range(attempts):
        value = bus.read(sid, PRESENT_POSITION, 2)
        if value is not None:
            return value
    return None


def find_stop(bus, sid, direction, log):
    """Walk one way until the joint stalls. Returns the last free position."""
    position = read_position(bus, sid)
    if position is None:
        raise RuntimeError(f"Cannot read position of ID {sid}")
    last_free = position
    stalled_for = 0
    deadline = time.perf_counter() + SEARCH_TIMEOUT

    while ENCODER_MIN + 5 < position < ENCODER_MAX - 5:
        if time.perf_counter() > deadline:
            log(f"      gave up after {SEARCH_TIMEOUT:.0f}s at {position}")
            return last_free
        target = position + direction * STEP_TICKS
        bus.move_to(sid, target, speed=SPEED)
        time.sleep(SETTLE)
        moved_to = read_position(bus, sid)
        if moved_to is None:
            continue      # a dropped read; try the same step again
        load = abs(bus.read_load(sid) or 0)

        # A stop shows as the joint falling behind its command while the load
        # climbs - not as the position ceasing to change, because the command
        # keeps running ahead of a joint that has already stopped.
        behind = abs(moved_to - target)
        if load > STALL_LOAD and behind > STALL_TICKS:
            stalled_for += 1
            if stalled_for >= 2:
                log(f"      stop at {moved_to} (load {load})")
                # Do not sit on the stop.
                bus.move_to(sid, moved_to - direction * BACKOFF_TICKS, speed=SPEED)
                time.sleep(0.4)
                return last_free
        else:
            stalled_for = 0
            last_free = moved_to
        position = moved_to

    log(f"      reached the encoder end at {position}; no hard stop this way")
    return last_free


def measure(bus, sid, log):
    name = JOINT_NAMES[sid]
    start = read_position(bus, sid)
    stored = (bus.read(sid, MIN_ANGLE_LIMIT, 2), bus.read(sid, MAX_ANGLE_LIMIT, 2))
    log(f"  {name} (ID {sid}) starts at {start}, stored range {stored[0]}-{stored[1]}")

    # Open the limits first, or the servo refuses to move past them and the search
    # measures the stored range back to itself.
    bus.write(sid, MIN_ANGLE_LIMIT, ENCODER_MIN, size=2)
    bus.write(sid, MAX_ANGLE_LIMIT, ENCODER_MAX, size=2)
    bus.write(sid, GOAL_POSITION, start, size=2)
    bus.torque(sid, True)
    time.sleep(0.1)

    log("    searching upwards")
    high = find_stop(bus, sid, +1, log)
    bus.move_to(sid, start, speed=SPEED)
    time.sleep(0.8)
    log("    searching downwards")
    low = find_stop(bus, sid, -1, log)
    bus.move_to(sid, (low + high) // 2, speed=SPEED)
    time.sleep(0.8)

    low, high = low + SAFETY_MARGIN, high - SAFETY_MARGIN
    log(f"    measured {low}-{high} ticks "
        f"({(high - low) / TICKS_PER_DEG:.0f} deg), was "
        f"{(stored[1] - stored[0]) / TICKS_PER_DEG:.0f} deg")
    return stored, (low, high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("joints", nargs="*", default=[],
                        help="joint names, e.g. wrist_roll")
    parser.add_argument("--all", action="store_true", help="every joint")
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--apply", action="store_true",
                        help="write the measured range to the servo")
    args = parser.parse_args()

    by_name = {name: sid for sid, name in JOINT_NAMES.items()}
    if args.all:
        wanted = list(JOINT_NAMES)
    elif args.joints:
        unknown = [j for j in args.joints if j not in by_name]
        if unknown:
            raise SystemExit(f"Unknown joint(s): {unknown}. "
                             f"Choose from {list(by_name)}")
        wanted = [by_name[j] for j in args.joints]
    else:
        raise SystemExit("Name a joint, or pass --all")

    if not args.apply:
        print("  Measuring only. Pass --apply to write the results to the servos.\n")

    results = {}
    with Bus(resolve_port([args.port])[0]) as bus:
        try:
            for sid in wanted:
                stored, measured = measure(bus, sid, print)
                results[sid] = (stored, measured)
                if args.apply:
                    bus.write(sid, MIN_ANGLE_LIMIT, measured[0], size=2)
                    bus.write(sid, MAX_ANGLE_LIMIT, measured[1], size=2)
                    print(f"    written to the servo")
                else:
                    # Put the original limits back; the search widened them.
                    bus.write(sid, MIN_ANGLE_LIMIT, stored[0], size=2)
                    bus.write(sid, MAX_ANGLE_LIMIT, stored[1], size=2)
                print()
        finally:
            for sid in wanted:
                for _ in range(5):
                    bus.write(sid, TORQUE_ENABLE, 0)
                    if bus.read(sid, TORQUE_ENABLE, 1) == 0:
                        break

    print("  joint            stored          measured        change")
    for sid, (stored, measured) in results.items():
        before = (stored[1] - stored[0]) / TICKS_PER_DEG
        after = (measured[1] - measured[0]) / TICKS_PER_DEG
        print(f"  {JOINT_NAMES[sid]:<15} {stored[0]:>5}-{stored[1]:<5} "
              f"{measured[0]:>7}-{measured[1]:<5} {after - before:>+8.0f} deg")

    if args.apply:
        print("\n  The LeRobot calibration file still holds the old ranges. Re-run")
        print("  scripts/easy_calibrate.py to bring it back in step, or the two")
        print("  disagree about what the joint can do.")


if __name__ == "__main__":
    main()
