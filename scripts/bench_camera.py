"""Measure camera throughput, and what it costs the servo control loop.

The control loop runs at 120 Hz; cameras run at about 30. The question this answers
is whether capturing frames slows the servo loop down.

    uv run scripts/bench_camera.py                      # cameras alone
    uv run scripts/bench_camera.py --port COM4          # cameras + servo loop
    uv run scripts/bench_camera.py --cameras icspring,0
"""

from so101.platform import require_windows

require_windows()

import argparse
import statistics
import time

from so101.camera.capture import CameraSet
from so101.camera.discovery import device_names, resolve
from so101.hardware import bus_patch  # noqa: F401 - serial retries
from so101.hardware import resolve as resolve_port

DURATION = 5.0
CONTROL_HZ = 120


def servo_loop(port, duration, stop_at):
    """Poll every servo as fast as the control loop would, and time each step."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll", "gripper"]
    motors = {name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100)
              for i, name in enumerate(joints)}
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect(handshake=False)

    period = 1.0 / CONTROL_HZ
    steps = []
    try:
        while time.perf_counter() < stop_at:
            t0 = time.perf_counter()
            bus.sync_read("Present_Position", normalize=False)
            steps.append((time.perf_counter() - t0) * 1000)
            slack = period - (time.perf_counter() - t0)
            if slack > 0:
                time.sleep(slack)
    finally:
        bus.disconnect()
    return steps


def report_steps(label, steps):
    steps = sorted(steps)
    print(f"\n{label}: {len(steps)} steps")
    print(f"  median {statistics.median(steps):6.2f} ms")
    print(f"  p95    {steps[int(len(steps) * 0.95)]:6.2f} ms")
    print(f"  max    {steps[-1]:6.2f} ms")
    budget = 1000 / CONTROL_HZ
    over = sum(1 for s in steps if s > budget)
    print(f"  over the {budget:.1f} ms budget for {CONTROL_HZ} Hz: "
          f"{over} ({100 * over / len(steps):.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", default=None,
                        help="comma-separated names or indices (default: all that open)")
    parser.add_argument("--port", default=None,
                        help="servo port to poll concurrently, e.g. COM4")
    parser.add_argument("--seconds", type=float, default=DURATION)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fourcc", default=None, help="e.g. MJPG")
    args = parser.parse_args()

    if args.cameras:
        specs = {spec: spec for spec in args.cameras.split(",")}
    else:
        specs = {name: index for index, name in enumerate(device_names())}
    if not specs:
        raise SystemExit("No cameras found.")

    print(f"Cameras: {', '.join(f'{k} -> index {resolve(v)}' for k, v in specs.items())}")

    with CameraSet.from_specs(specs, width=args.width, height=args.height,
                              fourcc=args.fourcc) as cams:
        for role, stream in cams.streams.items():
            try:
                stream.wait_for_frame(timeout=5.0)
                fourcc, w, h, fps = stream.format
                print(f"  {role}: negotiated {fourcc} {w}x{h} @ {fps:.0f} fps")
            except TimeoutError as e:
                print(f"  {role}: {e}")

        stop_at = time.perf_counter() + args.seconds
        steps = None
        if args.port:
            port = resolve_port([args.port])[0]
            print(f"\nPolling servos on {port} at {CONTROL_HZ} Hz "
                  f"while the cameras stream...")
            steps = servo_loop(port, args.seconds, stop_at)
        else:
            while time.perf_counter() < stop_at:
                time.sleep(0.05)

        print("\nCamera throughput:")
        for stream in cams.streams.values():
            print(f"  {stream}")
            frame = stream.read()
            if frame is not None:
                print(f"      newest frame is {frame.age_ms:.0f} ms old")

    if steps:
        report_steps(f"Servo loop while {len(specs)} camera(s) streamed", steps)


if __name__ == "__main__":
    main()
