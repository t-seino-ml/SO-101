"""Watch both SO-101 arms and report which one is moved by hand.

The arm that moves is the leader; the untouched one is the follower.
"""
import sys
import time
from so101.hardware import (
    Bus,
    JOINT_NAMES,
    PRESENT_POSITION,
    TICKS_PER_DEG,
    resolve,
)

PORTS = resolve(sys.argv[1:])
GRACE = 8.0
DURATION = 45.0
MOVE_THRESHOLD = 25  # ticks (~2.2 deg), well above encoder noise


def read_positions(bus):
    out = {}
    for sid in JOINT_NAMES:
        pos = bus.read(sid, PRESENT_POSITION, 2)
        if pos is not None:
            out[sid] = pos
    return out


buses = {p: Bus(p) for p in PORTS}
try:
    print(f"Get ready - measuring starts in {GRACE:.0f}s", flush=True)
    time.sleep(GRACE)
    print(f"MOVE THE LEADER ARM NOW ({DURATION:.0f}s)", flush=True)

    span = {p: {sid: [v, v] for sid, v in read_positions(b).items()}
            for p, b in buses.items()}
    end = time.time() + DURATION
    while time.time() < end:
        for p, bus in buses.items():
            for sid, v in read_positions(bus).items():
                lo, hi = span[p][sid]
                span[p][sid] = [min(lo, v), max(hi, v)]
finally:
    for bus in buses.values():
        bus.close()

travel = {}
for p in PORTS:
    print(f"\n=== {p} ===")
    total = 0
    for sid, (lo, hi) in sorted(span[p].items()):
        delta = hi - lo
        total = max(total, delta)
        mark = "  <-- MOVED" if delta > MOVE_THRESHOLD else ""
        print(f"  ID {sid} {JOINT_NAMES[sid]:<14} {lo:>4}..{hi:>4}  "
              f"delta {delta:>4} ({delta / TICKS_PER_DEG:>5.1f} deg){mark}")
    travel[p] = total

moved = [p for p, t in travel.items() if t > MOVE_THRESHOLD]
print()
if len(PORTS) == 1:
    p = PORTS[0]
    print(f"{p}: {'MOVED' if moved else 'NO MOVEMENT DETECTED'} "
          f"(largest travel {travel[p]} ticks)")
elif len(moved) == 1:
    leader = moved[0]
    follower = next(p for p in PORTS if p != leader)
    print(f"LEADER   = {leader}  (moved by hand)")
    print(f"FOLLOWER = {follower}  (stayed still)")
elif not moved:
    print("No movement detected on either arm - nothing to conclude.")
else:
    print(f"Both arms moved ({', '.join(moved)}) - rerun and move only the leader.")
