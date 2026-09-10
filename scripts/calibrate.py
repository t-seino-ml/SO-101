"""Calibrate the camera against the arm, start to finish, in one command.

Three phases, no other commands to run:

1. Teleoperation, in two steps. First pick up a block: open the gripper with the
   leader's trigger, put a block between the jaws, squeeze, press ENTER when the
   readout says HOLDING. Then rest that block on the table and press ENTER again -
   that second pose is what sets the table height, so it has to be a real touch.
2. Automatic sweep. The arm parks itself, takes a reference frame, then places the
   block at a dozen positions it works out for itself. The detector reads where
   each one lands, forward kinematics says where it really was, and the pixel to
   arm homography is fitted from the pairs.
3. Release. The gripper opens and everything relaxes.

No markers, nothing printed or measured. Re-run it whenever the camera has moved.
The table need not be cleared: the held block is identified by differencing against
the reference frame, so blocks already lying there are ignored.

    uv run scripts/calibrate.py
    uv run scripts/calibrate.py --camera overhead --grid 4x5
    uv run scripts/calibrate.py --no-teleop      # power the gripper, place by hand
"""

import argparse
import time
from pathlib import Path

from so101.platform import require_windows

require_windows()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from so101.camera import CameraSet  # noqa: E402
from so101.hardware import bus_patch  # noqa: F401,E402
from so101.hardware import resolve as resolve_port  # noqa: E402
from so101.hardware import tuning  # noqa: E402
from so101.policy import ArmKinematics, BlockDetector, TableFrame  # noqa: E402
from so101.policy.auto_calibrate import AutoCalibrator, grid_positions  # noqa: E402


# Folded back over the base, clear of the table and of the camera's view of it.
PARK_JOINTS = {"shoulder_pan": 0.0, "shoulder_lift": -95.0, "elbow_flex": 90.0,
               "wrist_flex": 40.0, "wrist_roll": 0.0}
GRIPPER_OPEN = 40.0
GRIPPER_EMPTY = 2.0
GRIPPER_HELD_MARGIN = 4.0     # jaws stopping this far short of empty means held
TELEOP_HZ = 60

STEP_ONE_PROMPT = """
  Step 1 of 2: pick up a block.
  Open the gripper with the leader's trigger, put ONE block between the jaws and
  squeeze - gently. A hard squeeze trips the gripper's overload protection in
  about two seconds, and it is set far lower than the other joints'. Press ENTER
  when the readout says HOLDING.
"""

STEP_TWO_PROMPT = """
  Step 2 of 2: set the table height.
  Lower the block until it just rests on the table, then press ENTER. Every block
  is placed at that z during the sweep, so a pose in mid-air makes the arm either
  press into the table or drop the block from height.
"""

# The camera looks past the arm, so a block gripped deep between the jaws is
# hidden behind them. One run lost all twenty points that way: the arm moved
# correctly every time, the detector just could not see what it was carrying.
MIN_HELD_CONFIDENCE = 0.6


def joints_of(robot):
    return {key.removesuffix(".pos"): float(value)
            for key, value in robot.get_observation().items() if key.endswith(".pos")}


def holding(robot):
    return joints_of(robot)["gripper"] > GRIPPER_EMPTY + GRIPPER_HELD_MARGIN


def teleop_until_enter(robot, teleop, arm, prompt, fps=TELEOP_HZ,
                       stream=None, detector=None):
    """Let the leader drive until ENTER, showing the gripper state and tip height.

    When a camera and detector are supplied, it also reports whether the block in
    the jaws is actually visible - a block gripped too deep disappears behind the
    gripper, and the sweep then has nothing to track.
    """
    from lerobot.utils.utils import enter_pressed, move_cursor_up

    print(prompt)
    period = 1.0 / fps
    last_check = 0.0
    seen = None
    while True:
        started = time.perf_counter()
        robot.send_action(teleop.get_action())
        current = joints_of(robot)
        held = current["gripper"] > GRIPPER_EMPTY + GRIPPER_HELD_MARGIN
        tip = arm.forward({name: current[name] for name in arm.joint_names})

        if detector is not None and started - last_check > 0.5:
            frame = stream.read()
            if frame is not None:
                found = detector.detect(frame.image)
                best = max((d.confidence for d in found), default=0.0)
                seen = (len(found), best)
            last_check = started

        vision = ""
        if seen is not None:
            count, best = seen
            vision = (f"   camera: {count} block(s), best {best:.2f} "
                      + ("VISIBLE" if best >= MIN_HELD_CONFIDENCE
                         else "TOO HIDDEN - grip nearer the tips"))
        print(f"    gripper {current['gripper']:+6.1f} deg  "
              f"{'HOLDING' if held else 'empty  '}   "
              f"tip x={tip[0]:+.3f} y={tip[1]:+.3f} z={tip[2]:+.3f}{vision}   ",
              end="", flush=True)
        if enter_pressed():
            print()
            return held, tip
        print()
        move_cursor_up(1)
        time.sleep(max(0.0, period - (time.perf_counter() - started)))


def power_gripper_and_wait(robot):
    """Fallback when there is no leader: hold the gripper open, then close on ENTER."""
    print("\n  The arm is limp except for the gripper.")
    start = joints_of(robot)
    robot.send_action({**{f"{k}.pos": v for k, v in start.items()},
                       "gripper.pos": GRIPPER_OPEN})
    time.sleep(1.0)
    input("  Put one block between the jaws and press ENTER to close: ")
    current = joints_of(robot)
    robot.send_action({**{f"{k}.pos": v for k, v in current.items()},
                       "gripper.pos": GRIPPER_EMPTY})
    time.sleep(1.0)
    return holding(robot)


def annotate(image, pixels, arm_xy, out_path):
    canvas = image.copy()
    for pixel, position in zip(pixels, arm_xy):
        point = tuple(int(v) for v in pixel)
        cv2.circle(canvas, point, 7, (0, 220, 255), 2)
        cv2.putText(canvas, f"{position[0]:+.2f},{position[1]:+.2f}",
                    (point[0] + 9, point[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 220, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def release(robot):
    current = joints_of(robot)
    robot.send_action({**{f"{k}.pos": v for k, v in current.items()},
                       "gripper.pos": GRIPPER_OPEN})
    time.sleep(0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=None,
                        help="default: the newest trained detector under runs/")
    parser.add_argument("--camera", default="side")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--leader-port", default="COM3")
    parser.add_argument("--no-teleop", action="store_true",
                        help="no leader; power the gripper and place the block by hand")
    parser.add_argument("--grid", default="3x4", help="rows x columns of positions")
    # nargs="+" so a negative lower bound is not mistaken for an option name:
    # argparse reads "--y-range -0.14,0.08" as a missing argument otherwise.
    parser.add_argument("--x-range", nargs="+", default=["0.16,0.28"],
                        help="min,max in metres, e.g. 0.20,0.40")
    parser.add_argument("--y-range", nargs="+", default=["-0.10,0.10"],
                        help="min,max in metres, e.g. -0.14,0.08")
    parser.add_argument("--z-table", type=float, default=None,
                        help="table height in the arm frame; measured if omitted")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()



    def parse_range(parts):
        """Accept "0.2,0.4" or "0.2 0.4", with or without a leading minus."""
        values = [float(v) for part in parts for v in str(part).split(",") if v]
        if len(values) != 2:
            raise SystemExit(f"Expected two numbers for a range, got {parts}")
        return tuple(sorted(values))

    rows, columns = (int(v) for v in args.grid.lower().split("x"))
    x_range = parse_range(args.x_range)
    y_range = parse_range(args.y_range)
    positions = grid_positions(x_range, y_range, (rows, columns))

    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    tuning.install(verbose=False)
    arm = ArmKinematics()
    detector = BlockDetector(weights=args.weights)
    detector.warmup()
    print(f"  detector ready; {len(positions)} positions over "
          f"x {x_range[0]:+.3f}..{x_range[1]:+.3f}  "
          f"y {y_range[0]:+.3f}..{y_range[1]:+.3f}")

    robot = make_robot_from_config(SO101FollowerConfig(
        port=resolve_port([args.follower_port])[0], id="follower"))
    robot.connect()

    teleop = None
    cameras = CameraSet.from_config()
    cameras.start()
    try:
        cameras.wait_for_frames(timeout=25)
        if args.camera not in cameras.streams:
            raise SystemExit(f"No camera role {args.camera!r} in cameras.json")
        stream = cameras.streams[args.camera]
        # -- 1. get a block into the gripper ------------------------------
        measured_z = None
        if args.no_teleop:
            held = power_gripper_and_wait(robot)
            current = {name: value for name, value in joints_of(robot).items()
                       if name in arm.joint_names}
            measured_z = float(arm.forward(current)[2])
        else:
            from lerobot.teleoperators import make_teleoperator_from_config
            from lerobot.teleoperators.so_leader import SO101LeaderConfig

            teleop = make_teleoperator_from_config(SO101LeaderConfig(
                port=resolve_port([args.leader_port])[0], id="leader"))
            teleop.connect()
            held, _ = teleop_until_enter(
                robot, teleop, arm, STEP_ONE_PROMPT,
                stream=stream, detector=detector)
            if not held:
                raise SystemExit("  The gripper is empty. Run again holding a block.")

            held, tip = teleop_until_enter(
                robot, teleop, arm, STEP_TWO_PROMPT,
                stream=stream, detector=detector)
            measured_z = float(tip[2])
            teleop.disconnect()
            teleop = None

        if not held:
            raise SystemExit("  The gripper is empty. Run again and hold a block.")
        print("  block held\n")

        # -- 2. sweep ------------------------------------------------------
        z_table = args.z_table
        if z_table is None:
            z_table = measured_z
            print(f"  table height from the touch: {z_table:.3f} m")

        calibrator = AutoCalibrator(robot, arm, stream, detector, z_table,
                                    park_joints=PARK_JOINTS)
        print("  parking and taking a reference frame")
        reference = calibrator.take_reference()
        print("  visiting positions")
        pixels, arm_xy, skipped = calibrator.visit(positions)
    finally:
        # -- 3. release ----------------------------------------------------
        cameras.stop()
        if teleop is not None:
            teleop.disconnect()
        try:
            release(robot)
        finally:
            robot.disconnect()
        print("\n  gripper released, arm relaxed")

    for position, reason in skipped:
        print(f"    skipped {position}: {reason}")
    if len(pixels) < 4:
        raise SystemExit(f"\n  Only {len(pixels)} usable points; need at least 4.")

    frame = TableFrame.fit(pixels, arm_xy, z_table=z_table, camera=args.camera)
    print(f"\n  {frame}")
    print("  per-point error (mm): " +
          "  ".join(f"{r:.1f}" for r in frame.residuals_mm))
    path = frame.save(args.out) if args.out else frame.save()
    print(f"  saved {path}")
    print("  overlay: " + str(annotate(reference, pixels, arm_xy,
                                       Path("outputs/calibration_points.png"))))

    worst = max(frame.residuals_mm)
    if worst > 15:
        print(f"\n  Worst error is {worst:.0f} mm. Check the overlay: if the marked "
              "points do not spread across the working area, widen --x-range and "
              "--y-range.")


if __name__ == "__main__":
    main()
