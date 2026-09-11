"""Measure how well the side camera can put the arm over a block.

The grasp policy only has to work from wherever it is handed control, so the
question that decides the whole design is how accurately the side camera can
deliver the arm there. This measures exactly that, and nothing else: no policy,
no grasping.

For each block, the side camera says where it is, the arm hovers over that spot,
and the wrist camera says where the block actually ended up relative to the
gripper. The gap between the two is the error the policy would have to absorb.

Reporting the mean and the spread separately matters, because they mean
different things:

- the **mean** is the fixed offset between the URDF's gripper frame and where
  the wrist camera looks. It is one constant and can simply be subtracted.
- the **spread** is what changes from block to block - mostly parallax, since
  the side camera sees a 25 mm cube from an angle and its detected centre sits
  nearer the top face than the footprint, by an amount that varies across the
  image. A homography cannot represent that, so this part does not go away.

The spread is therefore the number that decides whether coarse positioning from
the side camera is good enough to hand over to a grasp policy.

Two things have to be right for the measurement itself to mean anything.

A block is only measured if no other block of its colour is nearby, because the
block is identified in the wrist view as "the one of that colour nearest the
middle" - which quietly picks the wrong one if its twin is close. The isolation
distance bounds the error the measurement can attribute.

And the pixels-per-millimetre scale is measured at every block rather than once,
by stepping the arm a known distance and watching the block slide across the
view. The wrist camera does not look straight down, so how far a millimetre
carries across the image depends on the arm's pose - a single global scale
silently mixes that variation into the answer.

    uv run scripts/check_reach.py
    uv run scripts/check_reach.py --colour green --blocks 6
    uv run scripts/check_reach.py --hover 0.04 --isolation 80
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

# How high above the table to look from. The poses come from the recordings
# rather than from an upright-gripper solve, which this arm cannot hold: see
# so101.policy.approach.
HOVER_M = 0.05
JOG_M = 0.015            # how far to step when measuring the pixel scale
SETTLE_S = 1.2           # let the servos arrive before believing the camera
MIN_CONFIDENCE = 0.5
ISOLATION_MM = 100.0     # how far the nearest same-coloured block must be
MIN_SCALE_PX_PER_MM = 0.5


def joints_of(robot):
    return {key.removesuffix(".pos"): float(value)
            for key, value in robot.get_observation().items()
            if key.endswith(".pos")}


def glide_to(robot, goal, seconds=2.0, fps=30):
    """Move to a pose gently, in small steps rather than one jump."""
    start = joints_of(robot)
    steps = max(1, int(seconds * fps))
    for step in range(1, steps + 1):
        robot.send_action({f"{name}.pos": start[name]
                           + (goal[name] - start[name]) * step / steps
                           for name in goal})
        time.sleep(1.0 / fps)


def go(robot, arm, pose, seconds=2.0):
    """Move to `pose`, and report where the gripper frame really ended up."""
    goal = dict(joints_of(robot))
    goal.update({name: value for name, value in pose.items()
                 if name != "gripper"})
    glide_to(robot, goal, seconds=seconds)
    time.sleep(SETTLE_S)
    reached = joints_of(robot)
    return arm.forward({name: reached[name] for name in arm.joint_names})


def seen_from_wrist(detector, stream, colour, centre):
    """The target-coloured block nearest the middle of the wrist view."""
    detections = [d for d in detector.detect(stream.read().image.copy())
                  if d.colour == colour and d.confidence >= MIN_CONFIDENCE]
    if not detections:
        return None
    return min(detections, key=lambda d: np.linalg.norm(d.pixel - centre))


def measure(robot, arm, approach, detector, wrist, colour, above, centre, jog):
    """How far the block is from under the gripper, in millimetres.

    The scale is measured here, at this pose, by stepping the arm along each
    table axis in turn and watching the block move across the view.
    """
    base_pose, _ = approach.pose_for(above)
    if base_pose is None:
        return None, "cannot be reached from here"
    base_tip = go(robot, arm, base_pose)
    base = seen_from_wrist(detector, wrist, colour, centre)
    if base is None:
        return None, "not visible from the wrist camera"

    pixels, millimetres = [], []
    for axis in (0, 1):
        step = np.zeros(3)
        step[axis] = jog
        # Refined from the pose already held, so the step stays a small local
        # move instead of jumping to a different posture over the same spot.
        stepped = approach.refine(base_pose, above + step)
        if stepped is None:
            return None, "cannot step far enough to measure the scale"
        tip = go(robot, arm, stepped, seconds=1.0)
        moved = seen_from_wrist(detector, wrist, colour, centre)
        if moved is None:
            return None, "lost sight of the block while measuring the scale"
        pixels.append(moved.pixel - base.pixel)
        millimetres.append(1000 * (tip - base_tip)[:2])

    scale = np.column_stack(pixels) @ np.linalg.inv(np.column_stack(millimetres))
    if min(np.linalg.norm(scale, axis=0)) < MIN_SCALE_PX_PER_MM:
        return None, "the view barely moved; scale not trustworthy"
    return (np.linalg.solve(scale, base.pixel - centre), scale), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colour", default=None,
                        help="restrict to one colour (default: spread over all)")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--hover", type=float, default=HOVER_M,
                        help="height above the table to look from, in metres")
    parser.add_argument("--jog", type=float, default=JOG_M)
    parser.add_argument("--isolation", type=float, default=ISOLATION_MM,
                        help="skip a block with another of its colour this close")
    parser.add_argument("--margin-mm", type=float, default=30.0,
                        help="how far outside the calibrated region to still "
                             "accept a block; the homography extrapolates there")
    parser.add_argument("--side", default="side")
    parser.add_argument("--wrist", default="wrist")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--demos", type=Path, default=Path("data/demos"),
                        help="recordings the approach poses come from")
    args = parser.parse_args()

    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    tuning.install(verbose=False)
    table = TableFrame.load()
    arm = ArmKinematics()
    approach = ApproachPoses.from_demos(args.demos, kinematics=arm)
    print(f"  {approach}")
    detector = BlockDetector(weights=args.weights)      # no frame: pixels only
    detector.warmup()
    print(f"  {table}")

    cameras = CameraSet.from_config()
    cameras.start()
    robot = make_robot_from_config(SO101FollowerConfig(
        port=resolve_port([args.follower_port])[0], id="follower"))
    robot.connect()
    home = joints_of(robot)

    measurements = []
    try:
        cameras.wait_for_frames(timeout=25)
        side = cameras.streams[args.side]
        wrist = cameras.streams[args.wrist]
        # From the frame itself: a camera does not always give the size asked for.
        height, width = wrist.read().image.shape[:2]
        centre = np.array([width / 2, height / 2])

        seen = [(d.colour, table.reach_target(d.pixel))
                for d in detector.detect(side.read().image.copy())
                if d.confidence >= MIN_CONFIDENCE]
        print(f"  side camera sees {len(seen)} block(s)")

        usable = []
        for colour, position in seen:
            outside = table.outside_covered_mm(position)
            if outside > args.margin_mm:
                continue
            if args.colour and colour != args.colour:
                continue
            twins = [1000 * np.linalg.norm(position - other)
                     for other_colour, other in seen
                     if other_colour == colour and other is not position]
            if twins and min(twins) < args.isolation:
                continue
            usable.append((colour, position, outside))
        if not usable:
            raise SystemExit("  No block is both inside the calibrated region "
                             "and far enough from another of its colour.")

        seen_colours = set()
        first = [i for i, entry in enumerate(usable)
                 if not (entry[0] in seen_colours or seen_colours.add(entry[0]))]
        order = first + [i for i in range(len(usable)) if i not in set(first)]
        chosen = [usable[i] for i in order[:args.blocks]]
        print(f"  measuring {len(chosen)}: "
              f"{', '.join(entry[0] for entry in chosen)}\n")

        for index, (colour, position, outside) in enumerate(chosen, 1):
            note = "" if outside <= 0 else f"  ({outside:.0f} mm outside)"
            print(f"  [{index}/{len(chosen)}] {colour} at "
                  f"x={position[0]:+.3f} y={position[1]:+.3f}{note}")
            result, problem = measure(robot, arm, approach, detector, wrist, colour,
                                      position + np.array([0.0, 0.0, args.hover]),
                                      centre, args.jog)
            if problem:
                print(f"      {problem}; skipping")
                continue
            error, scale = result
            measurements.append((colour, error))
            print(f"      scale {np.linalg.norm(scale[:, 0]):.2f} and "
                  f"{np.linalg.norm(scale[:, 1]):.2f} px/mm")
            print(f"      block is {error[0]:+.0f} mm, {error[1]:+.0f} mm from "
                  f"under the gripper  ({np.linalg.norm(error):.0f} mm out)")
    finally:
        cameras.stop()
        try:
            print("\n  returning to where it started")
            glide_to(robot, home, seconds=2.5)
            time.sleep(0.5)
        except Exception as error:  # noqa: BLE001 - always still relax
            print(f"  could not return ({error})")
        finally:
            robot.disconnect()
            print("  arm relaxed")

    if len(measurements) < 3:
        raise SystemExit(f"\n  Only {len(measurements)} measurement(s) - too few "
                         "to separate a fixed offset from the spread.")

    errors = np.array([error for _, error in measurements])
    mean = errors.mean(axis=0)
    residual = np.linalg.norm(errors - mean, axis=1)

    print(f"\n  {len(errors)} block(s) measured")
    print(f"  fixed offset      x={mean[0]:+.0f} mm  y={mean[1]:+.0f} mm"
          f"   ({np.linalg.norm(mean):.0f} mm) - subtractable")
    print(f"  what is left over median {np.median(residual):.0f} mm, "
          f"worst {residual.max():.0f} mm - not subtractable")
    print()
    if residual.max() <= 20:
        print("  Within the few centimetres a grasp policy can absorb. Coarse")
        print("  positioning from the side camera is good enough to hand over.")
    elif residual.max() <= 35:
        print("  Marginal. It would work for blocks near the middle of the")
        print("  calibrated region and fail towards its edges.")
    else:
        print("  Too far out to hand over to a grasp policy as it stands.")
        print("  Closing the loop on the wrist camera would avoid needing this")
        print("  to be accurate at all.")


if __name__ == "__main__":
    main()
