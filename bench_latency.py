"""Measure sync-read round-trip latency on a servo bus (read-only)."""
import statistics
import sys
import time

import bus_patch  # noqa: F401
from ports import resolve
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
SAMPLES = 300

port = resolve(sys.argv[1:])[0]
motors = {name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100)
          for i, name in enumerate(JOINTS)}
bus = FeetechMotorsBus(port=port, motors=motors)
bus.connect(handshake=False)

samples = []
try:
    for _ in range(SAMPLES):
        t0 = time.perf_counter()
        bus.sync_read("Present_Position", normalize=False)
        samples.append((time.perf_counter() - t0) * 1000)
finally:
    bus.disconnect()

samples.sort()
print(f"{port}: sync_read of {len(JOINTS)} servos, {SAMPLES} samples")
print(f"  min    {samples[0]:6.2f} ms")
print(f"  median {statistics.median(samples):6.2f} ms")
print(f"  p95    {samples[int(len(samples) * 0.95)]:6.2f} ms")
print(f"  max    {samples[-1]:6.2f} ms")
print(f"  -> a read+write loop costs about {2 * statistics.median(samples):.1f} ms")
