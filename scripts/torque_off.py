"""Release every servo on an arm, over the raw bus. The second emergency stop.

This exists for the case where the LeRobot process is no longer answering. It
does not import LeRobot, does not connect a robot, and does not read a
calibration: it opens the serial port and writes zero to Torque_Enable, which is
the one thing that has to work when nothing else does.

A Windows COM port is exclusive, so this can only reach the servos once the
process holding the port has let go. If a run is hung, kill it first:

    taskkill /F /IM python.exe        (or Ctrl-C in its terminal)
    uv run scripts/torque_off.py

*** The arm falls when torque is released. *** If it is raised - and the R1
posture holds it straight up, about 400 mm - support it by hand first, or lower
it before running this. Releasing a raised arm is not a safe default; it is the
last resort, and the same warning applies to cutting the power.

    uv run scripts/torque_off.py                 # every port found
    uv run scripts/torque_off.py COM4            # just the follower
    uv run scripts/torque_off.py COM4 --report   # show torque state, change nothing
"""

from so101.platform import require_windows

require_windows()

import argparse  # noqa: E402
import sys  # noqa: E402

from so101.hardware.sts3215 import (  # noqa: E402
    Bus,
    GOAL_POSITION,
    JOINT_NAMES,
    PRESENT_POSITION,
    TORQUE_ENABLE,
    TICKS_PER_DEG,
)

ATTEMPTS = 5


def release(bus, sid, name, report_only):
    """Write Torque_Enable = 0 until the servo confirms it. Returns a line."""
    position = bus.read(sid, PRESENT_POSITION, 2)
    before = bus.read(sid, TORQUE_ENABLE, 1)
    if position is None or before is None:
        return f"  {name:<14} no answer"
    where = f"{position:>5} ticks ({position / TICKS_PER_DEG:>6.1f}d)"
    if report_only:
        return f"  {name:<14} {where}  torque {before}"
    if before == 0:
        return f"  {name:<14} {where}  torque already off"

    # Park the goal where the joint actually is first. A servo whose goal is far
    # from its position will lunge there the moment torque comes back on, and
    # something is going to turn it back on later.
    bus.write(sid, GOAL_POSITION, position, size=2)
    for _ in range(ATTEMPTS):
        bus.write(sid, TORQUE_ENABLE, 0)
        if bus.read(sid, TORQUE_ENABLE, 1) == 0:
            return f"  {name:<14} {where}  torque OFF"
    return f"  {name:<14} {where}  *** STILL ON after {ATTEMPTS} tries ***"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ports", nargs="*", help="e.g. COM4; default is all found")
    parser.add_argument("--report", action="store_true",
                        help="show the torque state and change nothing")
    args = parser.parse_args()

    from so101.hardware import resolve as resolve_port

    if not args.report:
        print("\n  *** The arm falls when torque is released. Support it first "
              "if it is raised. ***\n")

    failed = False
    for port in resolve_port(args.ports):
        print(f"=== {port} ===")
        try:
            with Bus(port) as bus:
                for sid, name in JOINT_NAMES.items():
                    if not bus.ping(sid):
                        print(f"  {name:<14} NO RESPONSE")
                        failed = True
                        continue
                    line = release(bus, sid, name, args.report)
                    print(line)
                    failed = failed or "STILL ON" in line
        except Exception as error:  # noqa: BLE001 - report, do not raise
            print(f"  cannot open {port}: {error}")
            print("  another process is probably still holding it - kill it first")
            failed = True
        print()

    if failed and not args.report:
        print("  Not every servo confirmed. Cut the power, supporting the arm.")
        sys.exit(1)


if __name__ == "__main__":
    main()
