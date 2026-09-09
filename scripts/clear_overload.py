"""Report and clear latched servo protection faults.

A Feetech servo that trips its overload protection keeps answering with the fault
bit set and freezes its load register at the trip value. LeRobot reads such a
reply as "motor missing" and refuses to connect - the symptom is a connect-time
error naming a motor that pings perfectly well through a lower-level tool.

Clearing needs the joint to be out of whatever was straining it, so if a joint
will not clear, move the arm to a relaxed pose by hand and run this again.

    uv run scripts/clear_overload.py            # both arms, report and clear
    uv run scripts/clear_overload.py COM4
    uv run scripts/clear_overload.py --report   # report only, change nothing
"""

import argparse
import time

from so101.platform import require_windows

require_windows()

from so101.hardware import resolve  # noqa: E402
from so101.hardware.sts3215 import (  # noqa: E402
    Bus,
    GOAL_POSITION,
    JOINT_NAMES,
    PRESENT_POSITION,
    TORQUE_ENABLE,
    _packet,
)

FLAGS = ((0x01, "voltage"), (0x02, "angle limit"), (0x04, "overheat"),
         (0x08, "overcurrent"), (0x20, "overload"))


class StatusBus(Bus):
    """A bus that reports the status byte instead of discarding it."""

    def status(self, sid):
        self.ser.reset_input_buffer()
        self.ser.write(_packet(sid, 2, (PRESENT_POSITION, 2)))
        head = self.ser.read(4)
        if len(head) < 4:
            return None
        rest = self.ser.read(head[3])
        return rest[0] if rest else None


def describe(status):
    if status is None:
        return "no reply"
    names = [name for bit, name in FLAGS if status & bit]
    return ", ".join(names) if names else "clean"


def clear(bus, sid):
    """Park the goal on the present position, then toggle torque to reset."""
    position = bus.read(sid, PRESENT_POSITION, 2)
    if position is not None:
        bus.write(sid, GOAL_POSITION, position, size=2)
    for value in (0, 1, 0):
        bus.write(sid, TORQUE_ENABLE, value)
        time.sleep(0.3)
    return bus.status(sid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ports", nargs="*")
    parser.add_argument("--report", action="store_true",
                        help="show the status bytes and change nothing")
    args = parser.parse_args()

    still_faulted = []
    for port in resolve(args.ports):
        print(f"=== {port} ===")
        with StatusBus(port) as bus:
            for sid, name in JOINT_NAMES.items():
                status = bus.status(sid)
                load = bus.read_load(sid)
                line = (f"  ID {sid} {name:<14} status=0x{status:02X} "
                        f"[{describe(status)}]  load={load}")
                if args.report or status in (None, 0):
                    print(line)
                    continue
                after = clear(bus, sid)
                print(f"{line}  ->  0x{after:02X} [{describe(after)}]")
                if after:
                    still_faulted.append(f"{port} {name}")

    if still_faulted:
        print("\nStill faulted: " + ", ".join(still_faulted))
        print("Move those joints to a relaxed pose by hand and run this again.")
        print("If that does not clear it, power-cycle the arm.")
    elif not args.report:
        print("\nAll joints clean.")


if __name__ == "__main__":
    main()
