"""Pick up the block of a chosen colour, using the side camera and nothing learnt.

This is the coarse-positioning half of the task on its own: the side camera says
which block and where, the arm takes the posture the operator used for that part
of the table, closes, and lifts. Whether the block comes up is a straight answer
to whether the side camera can place the arm accurately enough - one that does
not depend on a policy, or on measuring pixels.

The posture is the part that used to be wrong. Asking inverse kinematics to hold
the gripper vertical puts most of the table out of reach on a five-joint arm; the
demonstrations tilt a median of 23 degrees and reach half again as far. So the
pose comes from the recordings (see so101.policy.approach) and only its position
is adjusted - the height included, since the demonstrated grasp height for that
part of the table is better information than the table plane plus a guess.

    uv run scripts/try_grasp.py --colour green
    uv run scripts/try_grasp.py                    # whichever block is clearest
    uv run scripts/try_grasp.py --colour red --dry-run
"""

import argparse
import time
from pathlib import Path

from so101.platform import require_windows

require_windows()

import numpy as np  # noqa: E402

from so101.camera import CameraSet  # noqa: E402
from so101.hardware import bus_patch  # noqa: F401,E402
from so101.hardware import resolve as resolve_port  # noqa: E402
from so101.hardware import tuning  # noqa: E402
from so101.policy import (  # noqa: E402
    ApproachPoses,
    ArmKinematics,
    BlockDetector,
    TableFrame,
)

HOVER_M = 0.06           # how far above the grasp pose to arrive first
LIFT_M = 0.05            # how far to lift once closed
MIN_CONFIDENCE = 0.5
MOVED_MM = 15.0          # how far a block must shift to count as picked up
SETTLE_S = 1.0
LOAD_ABORT = 700


def joints_of(robot):
    return {key.removesuffix(".pos"): float(value)
            for key, value in robot.get_observation().items()
            if key.endswith(".pos")}


def glide_to(robot, goal, seconds=2.0, fps=30):
    start = joints_of(robot)
    steps = max(1, int(seconds * fps))
    for step in range(1, steps + 1):
        robot.send_action({f"{name}.pos": start[name]
                           + (goal[name] - start[name]) * step / steps
                           for name in goal})
        time.sleep(1.0 / fps)


def strained(robot):
    for name in robot.bus.motors:
        try:
            load = robot.bus.read("Present_Load", name, normalize=False)
        except Exception:  # noqa: BLE001 - a dropped packet is not a fault
            continue
        if abs(load) > LOAD_ABORT:
            return f"{name} load {load}"
    return None


def blocks_on_table(detector, stream, table, margin_mm):
    """Every block the side camera can place, as (colour, position)."""
    found = []
    for detection in detector.detect(stream.read().image.copy()):
        if detection.confidence < MIN_CONFIDENCE:
            continue
        position = table.reach_target(detection.pixel)
        if table.outside_covered_mm(position) > margin_mm:
            continue
        found.append((detection.colour, position, detection.confidence))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colour", default=None,
                        help="which colour to pick up (default: the clearest block)")
    parser.add_argument("--hover", type=float, default=HOVER_M)
    parser.add_argument("--lift", type=float, default=LIFT_M)
    parser.add_argument("--grip", type=float, default=None,
                        help="how far to close, in degrees (default: what the "
                             "nearest demonstrations closed to)")
    parser.add_argument("--open", type=float, default=45.0,
                        help="how wide to open before closing, in degrees")
    parser.add_argument("--margin-mm", type=float, default=30.0)
    parser.add_argument("--side", default="side")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--demos", type=Path, default=Path("data/demos"))
    parser.add_argument("--out", type=Path, default=Path("outputs/grasp"))
    parser.add_argument("--dry-run", action="store_true",
                        help="say what it would do, and move nothing")
    args = parser.parse_args()

    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    tuning.install(verbose=False)
    table = TableFrame.load()
    arm = ArmKinematics()
    approach = ApproachPoses.from_demos(args.demos, kinematics=arm)
    detector = BlockDetector(weights=args.weights)
    detector.warmup()
    print(f"  detector: {detector.weights}")
    print(f"  {approach}")

    cameras = CameraSet.from_config()
    cameras.start()
    robot = make_robot_from_config(SO101FollowerConfig(
        port=resolve_port([args.follower_port])[0], id="follower"))
    robot.connect()
    home = joints_of(robot)

    try:
        cameras.wait_for_frames(timeout=25)
        side = cameras.streams[args.side]
        before = blocks_on_table(detector, side, table, args.margin_mm)
        if not before:
            raise SystemExit("  No blocks the side camera can place.")

        wanted = [entry for entry in before
                  if args.colour is None or entry[0] == args.colour]
        if not wanted:
            raise SystemExit(f"  No {args.colour} block inside the region. "
                             f"Seen: {', '.join(sorted({c for c, _, _ in before}))}")
        colour, target, confidence = max(wanted, key=lambda entry: entry[2])
        print(f"  target: {colour} at x={target[0]:+.3f} y={target[1]:+.3f} "
              f"(confidence {confidence:.2f})")

        # The demonstrated grasp posture for this part of the table, moved
        # sideways onto this block. Its height is kept: what the operator used
        # here is better than the table plane plus a guess at the jaw offset.
        seed, nearest = approach.blend(target)
        height = arm.forward({name: seed[name] for name in arm.joint_names})[2]
        print(f"  nearest demonstration {1000*nearest:.0f} mm away, "
              f"grasping at z={1000*height:.0f} mm")

        at_block = approach.refine(seed, [target[0], target[1], height])
        if at_block is None:
            raise SystemExit("  Cannot bring a demonstrated posture onto that block.")
        above = approach.refine(at_block,
                                [target[0], target[1], height + args.hover])
        lifted = approach.refine(at_block,
                                 [target[0], target[1], height + args.lift])
        if above is None or lifted is None:
            raise SystemExit("  Cannot approach or lift from there.")

        grip = args.grip if args.grip is not None else seed["gripper"]
        print(f"  closing to {grip:.0f} deg (the demonstrations' own grip)")
        if args.dry_run:
            print("\n  dry run; nothing moved")
            return

        print("\n  approaching")
        glide_to(robot, {**above, "gripper": args.open}, seconds=2.5)
        time.sleep(SETTLE_S)
        print("  descending")
        glide_to(robot, {**at_block, "gripper": args.open}, seconds=1.5)
        time.sleep(SETTLE_S)
        hurt = strained(robot)
        if hurt:
            raise SystemExit(f"  {hurt} on the way down; stopping")

        print("  closing")
        glide_to(robot, {**at_block, "gripper": grip}, seconds=1.2)
        time.sleep(SETTLE_S)
        held = joints_of(robot)["gripper"]
        print(f"  gripper settled at {held:.0f} deg "
              f"({'something is between the jaws' if held > grip + 2 else 'closed on nothing'})")

        print("  lifting")
        glide_to(robot, {**lifted, "gripper": grip}, seconds=1.5)
        time.sleep(1.5)

        # A photograph while it is still held: counting detections can be fooled
        # by a change in the light, and this cannot.
        import cv2

        args.out.mkdir(parents=True, exist_ok=True)
        for role, stream in cameras.streams.items():
            cv2.imwrite(str(args.out / f"held_{role}.png"), stream.read().image)
        print(f"  photographed while held: {args.out}")

        after = blocks_on_table(detector, side, table, args.margin_mm)
        still_there = [p for c, p, _ in after if c == colour]
        gone = (not still_there or
                min(1000 * np.linalg.norm(target - p) for p in still_there) > MOVED_MM)
        print(f"\n  the side camera now sees {len(after)} block(s) "
              f"({len(before)} before)")
        print(f"  {'PICKED UP' if gone else 'still on the table'}: the {colour} "
              f"block is {'no longer where it was' if gone else 'where it was'}")
    finally:
        cameras.stop()
        try:
            print("\n  returning to where it started")
            glide_to(robot, {**home, "gripper": args.open}, seconds=2.0)
            time.sleep(0.3)
            glide_to(robot, home, seconds=0.8)
        except Exception as error:  # noqa: BLE001 - always still relax
            print(f"  could not return ({error})")
        finally:
            robot.disconnect()
            print("  arm relaxed")


if __name__ == "__main__":
    main()
