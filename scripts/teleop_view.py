"""Teleoperate the SO-101 with both camera views on screen.

Drives the follower from the leader exactly as `teleoperate.py` does, but attaches
the cameras from `cameras.json` and streams them, with the joint positions, to a
Rerun viewer that opens in its own window.

Rerun rather than an OpenCV window: LeRobot pulls in `opencv-python-headless`,
built with GUI support disabled, so `cv2.imshow` cannot open a window. `rerun-sdk`
comes with LeRobot and its viewer is what LeRobot's own `--display_data` uses.

Control and display run at different rates. LeRobot's own loop logs both camera
frames to Rerun on every control step, which at 120 Hz means 345 MB/s and a loop
that cannot keep up; see so101.teleop for what this does instead.

    uv run scripts/teleop_view.py                      # both cameras, 120 Hz
    uv run scripts/teleop_view.py --cameras overhead   # one camera
    uv run scripts/teleop_view.py --check              # connect only, moves nothing
    uv run scripts/teleop_view.py --max-relative-target=5   # gentle first run
"""

from so101.platform import add_venv_scripts_to_path, require_windows

require_windows()
# The Rerun viewer is an executable in the venv's Scripts directory.
add_venv_scripts_to_path()

import argparse

from so101.camera import load as load_cameras
from so101.camera import to_lerobot
from so101.hardware import bus_patch  # noqa: F401 - serial retries
from so101.hardware import resolve as resolve_port
from so101.hardware import tuning
from so101.teleop import DEFAULT_DISPLAY_HZ, DEFAULT_FPS
from so101.teleop import release_torque
from so101.teleop import run as teleop_run

FOLLOWER_PORT = "COM4"
LEADER_PORT = "COM3"


def check(config):
    """Connect, read one observation, and report it. Sends no actions."""
    from lerobot.robots import make_robot_from_config
    from lerobot.teleoperators import make_teleoperator_from_config

    robot = make_robot_from_config(config.robot)
    teleop = make_teleoperator_from_config(config.teleop)
    robot.connect()
    teleop.connect()
    try:
        observation = robot.get_observation()
        action = teleop.get_action()

        print()
        print("Follower observation:")
        for key, value in observation.items():
            shape = getattr(value, "shape", None)
            detail = shape if shape is not None else f"{float(value):.2f}"
            print(f"  {key}: {detail}")

        print()
        print("Leader action:")
        for key, value in action.items():
            print(f"  {key}: {float(value):.2f}")
    finally:
        teleop.disconnect()
        robot.disconnect()

    print()
    print("Everything connected. Run without --check to teleoperate.")


def teleoperate_with(config, display_hz):
    """LeRobot's teleoperate(), but running so101.teleop.run as the loop."""
    from lerobot.robots import make_robot_from_config
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.utils.visualization_utils import init_rerun

    if config.display_data:
        init_rerun(session_name="teleoperation")

    robot = make_robot_from_config(config.robot)
    teleop = make_teleoperator_from_config(config.teleop)
    teleop.connect()
    robot.connect()
    try:
        stats = teleop_run(
            robot=robot,
            teleop=teleop,
            fps=config.fps,
            display_hz=display_hz,
            display=config.display_data,
            duration=config.teleop_time_s,
        )
    finally:
        teleop.disconnect()
        try:
            robot.disconnect()
        except Exception as error:  # noqa: BLE001 - the arm must not stay powered
            print(f"Clean disconnect failed ({error})")
            stuck = release_torque(robot)
            print("Torque released on every joint." if not stuck
                  else f"STILL HOLDING: {', '.join(stuck)} - power the arm down.")
        if config.display_data:
            import rerun as rr

            rr.rerun_shutdown()
    print()
    print(stats.summary(config.fps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--follower-port", default=FOLLOWER_PORT)
    parser.add_argument("--leader-port", default=LEADER_PORT)
    parser.add_argument("--follower-id", default="follower")
    parser.add_argument("--leader-id", default="leader")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--cameras", default=None,
                        help="comma-separated roles from cameras.json (default: all)")
    parser.add_argument("--no-display", action="store_true",
                        help="run the loop without opening the viewer")
    parser.add_argument("--max-relative-target", type=float, default=None,
                        help="cap each step, in degrees. Off by default: it rate "
                             "limits the follower and costs an extra follower read "
                             "per step. Use it on a first run, not afterwards")
    parser.add_argument("--display-hz", type=float, default=DEFAULT_DISPLAY_HZ,
                        help="how often to log frames to the viewer")
    parser.add_argument("--seconds", type=float, default=None,
                        help="stop after this long instead of running until Ctrl+C")
    parser.add_argument("--check", action="store_true",
                        help="connect, read one observation, disconnect - moves nothing")
    args = parser.parse_args()

    tuning.install()

    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
    from lerobot.teleoperators.so_leader import SO101LeaderConfig

    specs = load_cameras()
    if args.cameras:
        wanted = [role.strip() for role in args.cameras.split(",")]
        missing = [role for role in wanted if role not in specs]
        if missing:
            raise SystemExit(f"Not in cameras.json: {', '.join(missing)}. "
                             f"Available: {', '.join(specs) or 'none'}")
        specs = {role: specs[role] for role in wanted}
    if not specs:
        raise SystemExit("No cameras configured. Edit cameras.json, or run "
                         "`uv run scripts/find_cameras.py` to see what is attached.")

    cameras = to_lerobot(specs)
    for role, spec in specs.items():
        print(f"  {role}: {spec.name!r} -> index {cameras[role].index_or_path}, "
              f"{spec.width}x{spec.height} {spec.fourcc or 'default'}")

    config = TeleoperateConfig(
        robot=SO101FollowerConfig(
            port=resolve_port([args.follower_port])[0],
            id=args.follower_id,
            cameras=cameras,
            max_relative_target=args.max_relative_target,
        ),
        teleop=SO101LeaderConfig(
            port=resolve_port([args.leader_port])[0],
            id=args.leader_id,
        ),
        fps=args.fps,
        teleop_time_s=args.seconds,
        display_data=not args.no_display,
    )

    if args.check:
        return check(config)

    print()
    print(f"Teleoperating at {args.fps} Hz, viewer at {args.display_hz:g} Hz. "
          "Ctrl+C to stop.")
    if config.display_data:
        print("A Rerun viewer window opens with the camera streams and joint data.")
    teleoperate_with(config, args.display_hz)


if __name__ == "__main__":
    main()
