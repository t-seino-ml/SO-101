"""Measure what the display rate and the safety clamp cost the control loop.

Logs to an in-memory Rerun recording rather than spawning the viewer, so the
numbers are the loop's own cost. Requires both arms; the follower tracks whatever
the leader is doing, so leave the leader still.

    uv run scripts/bench_teleop.py --seconds 8
"""

import argparse

from so101.platform import require_windows

require_windows()

from so101.camera import load as load_cameras  # noqa: E402
from so101.camera import to_lerobot  # noqa: E402
from so101.hardware import bus_patch  # noqa: F401,E402
from so101.hardware import resolve as resolve_port  # noqa: E402
from so101.hardware import tuning  # noqa: E402
from so101.teleop import run as teleop_run  # noqa: E402

# Logging on every control step is what LeRobot's own teleop_loop does.
EVERY_STEP = 10_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--leader-port", default="COM3")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()

    tuning.install(verbose=False)

    import rerun as rr
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.so_leader import SO101LeaderConfig

    rr.init("bench-teleop")  # memory sink, no viewer

    cameras = to_lerobot(load_cameras())
    leader = make_teleoperator_from_config(SO101LeaderConfig(
        port=resolve_port([args.leader_port])[0], id="leader"))

    cases = [
        ("display every step, clamp 5deg", EVERY_STEP, 5.0),
        ("display every step, no clamp", EVERY_STEP, None),
        ("display 30 Hz, no clamp", 30, None),
        ("no display, no clamp", 0, None),
    ]

    leader.connect()
    try:
        for label, display_hz, clamp in cases:
            follower = make_robot_from_config(SO101FollowerConfig(
                port=resolve_port([args.follower_port])[0], id="follower",
                cameras=cameras, max_relative_target=clamp))
            follower.connect()
            try:
                stats = teleop_run(follower, leader, fps=args.fps,
                                   display_hz=display_hz, display=display_hz > 0,
                                   duration=args.seconds)
            finally:
                follower.disconnect()
            print(f"\n--- {label} ---")
            print(stats.summary(args.fps))
    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()
