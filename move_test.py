"""Drive each SO-101 joint through a small sweep and back, one joint at a time.

Usage: python move_test.py COM4 [--amplitude-deg 25] [--ids 1,2,3,4,5,6]
"""
import argparse
import time
from sts3215 import (
    Bus, JOINT_NAMES, TICKS_PER_DEG,
    PRESENT_POSITION, PRESENT_TEMPERATURE,
    MIN_ANGLE_LIMIT, MAX_ANGLE_LIMIT,
)

# The gripper has far less travel than the arm joints, so it gets its own cap.
GRIPPER_ID = 6
GRIPPER_AMPLITUDE_DEG = 10
LOAD_ABORT = 800  # of a 1023 full-scale magnitude
TEMP_ABORT = 60
STEP_DEG = 1.5
STEP_DELAY = 0.03


def ramp(bus, sid, start, target, label):
    """Move in small increments so the joint travels slowly and can be aborted.

    Returns (final_position, peak_load).
    """
    step = STEP_DEG * TICKS_PER_DEG
    n = max(1, int(abs(target - start) / step))
    peak = 0
    for i in range(1, n + 1):
        bus.move_to(sid, start + (target - start) * i / n, speed=300)
        time.sleep(STEP_DELAY)
        load = bus.read_load(sid)
        temp = bus.read(sid, PRESENT_TEMPERATURE, 1)
        if load is not None:
            peak = max(peak, abs(load))
        if load is not None and abs(load) > LOAD_ABORT:
            raise RuntimeError(f"ID {sid} ({label}): load {load} exceeded {LOAD_ABORT}")
        if temp is not None and temp > TEMP_ABORT:
            raise RuntimeError(f"ID {sid} ({label}): temperature {temp}C too high")
    bus.wait_until_reached(sid, target)
    return bus.read(sid, PRESENT_POSITION, 2), peak


def test_joint(bus, sid, amplitude_deg):
    label = JOINT_NAMES[sid]
    start = bus.read(sid, PRESENT_POSITION, 2)
    lo = bus.read(sid, MIN_ANGLE_LIMIT, 2) or 0
    hi = bus.read(sid, MAX_ANGLE_LIMIT, 2) or 4095
    margin = 40
    amp = amplitude_deg * TICKS_PER_DEG
    up = min(start + amp, hi - margin)
    down = max(start - amp, lo + margin)
    print(f"ID {sid} {label:<14} start={start} sweep {down}..{up} "
          f"({(up - down) / TICKS_PER_DEG:.1f} deg)", flush=True)

    bus.torque(sid, True)
    time.sleep(0.05)
    try:
        reached_up, load_up = ramp(bus, sid, start, up, label)
        reached_down, load_down = ramp(bus, sid, up, down, label)
        back, _ = ramp(bus, sid, down, start, label)
    finally:
        bus.torque(sid, False)
    err = abs(back - start) if back is not None else None
    ok = (abs(reached_up - up) <= 25 and abs(reached_down - down) <= 25)
    print(f"   -> up   {reached_up:>5} / {up:>7.0f}  (err {abs(reached_up - up):>3.0f}, peak load {load_up})", flush=True)
    print(f"      down {reached_down:>5} / {down:>7.0f}  (err {abs(reached_down - down):>3.0f}, peak load {load_down})", flush=True)
    print(f"      back {back:>5} / {start:>7.0f}  (err {err})  -> {'OK' if ok else 'SHORTFALL'}", flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("port")
    p.add_argument("--amplitude-deg", type=float, default=25.0)
    p.add_argument("--ids", default="6,5,4,3,2,1")
    args = p.parse_args()

    ids = [int(x) for x in args.ids.split(",")]
    results = {}
    with Bus(args.port) as bus:
        for sid in ids:
            if not bus.ping(sid):
                print(f"ID {sid}: no response, skipped")
                results[sid] = False
                continue
            amp = GRIPPER_AMPLITUDE_DEG if sid == GRIPPER_ID else args.amplitude_deg
            try:
                results[sid] = test_joint(bus, sid, amp)
            except RuntimeError as e:
                print(f"   !! aborted: {e}")
                bus.torque(sid, False)
                results[sid] = False
            time.sleep(0.3)

    passed = sum(results.values())
    print(f"\n{passed}/{len(results)} joints passed on {args.port}")


if __name__ == "__main__":
    main()
