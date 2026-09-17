"""Move the arm above a target and stop there. Nothing descends.

This is the step that tells you whether the calibration is right, separated from
everything that could hurt if it is not. The arm goes to a fixed height above
where perception says the target is, holds, photographs what the wrist camera
sees, and comes back. If it arrives over the wrong spot, no amount of care on
the way down would have fixed it.

    uv run scripts/hover_over.py --target can
    uv run scripts/hover_over.py --target red --height 0.10
    uv run scripts/hover_over.py --target can --dry-run    # no motion at all
    uv run scripts/hover_over.py --target red --all        # every red block

It pauses for ENTER before the first move unless --no-confirm is given, and Ctrl-C
at any point returns the arm to where it started and relaxes it.
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
from so101.policy import ApproachPoses, ArmKinematics, BlockDetector, TableFrame  # noqa: E402
from so101.policy.camera_geometry import CameraGeometry  # noqa: E402
from so101.policy.can_detector import CanDetector  # noqa: E402
from so101.policy.motion import (  # noqa: E402
    Workspace,
    hover_over,
    joints_of,
    relax_to,
    tip_of,
)

HEIGHT_M = 0.08
GRIPPER_OPEN = 45.0
MIN_CONFIDENCE = 0.5
SLOW_S = 3.5          # deliberately unhurried; speed comes later


def targets_from(args, blocks, cans, image, table, geometry):
    """What to hover over, nearest the arm first."""
    if args.target == "can":
        found = cans.detect(image)
        return [("can", can.position, can.confidence) for can in found]

    found = [d for d in blocks.detect(image, colours=[args.target])
             if d.confidence >= MIN_CONFIDENCE]
    found.sort(key=lambda d: float(np.linalg.norm(d.position[:2])))
    return [(d.colour, d.position, d.confidence) for d in found]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True,
                        help="a block colour, or 'can'")
    parser.add_argument("--height", type=float, default=HEIGHT_M,
                        help="metres above the target to stop at")
    parser.add_argument("--all", action="store_true",
                        help="visit every match, not just the nearest")
    parser.add_argument("--seconds", type=float, default=SLOW_S,
                        help="how long each move takes; longer is slower")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what it would do and move nothing")
    parser.add_argument("--no-confirm", action="store_true")
    parser.add_argument("--side", default="side")
    parser.add_argument("--wrist", default="wrist")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--demos", type=Path, default=Path("data/demos"))
    parser.add_argument("--out", type=Path, default=Path("outputs/hover"))
    args = parser.parse_args()

    table = TableFrame.load()
    arm = ArmKinematics()
    approach = ApproachPoses.from_demos(args.demos, kinematics=arm)
    workspace = Workspace()
    blocks = BlockDetector(weights=args.weights, table_frame=table)
    blocks.warmup()

    print(f"  {table}")
    print(f"  {approach}")

    cameras = CameraSet.from_config()
    cameras.start()
    robot = None
    try:
        cameras.wait_for_frames(timeout=25)
        time.sleep(1.0)
        image = cameras.streams[args.side].read().image.copy()
        args.out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out / "00_planned_side.png"), image)

        # The camera geometry is refitted from whatever blocks are in view rather
        # than trusted from file: it is free, and a stale one from before the
        # camera was nudged would be worse than none.
        seen = [d for d in blocks.detect(image) if d.confidence >= 0.6]
        geometry = CameraGeometry.fit(table, seen)
        print(f"  {geometry}")
        cans = CanDetector(table_frame=table, geometry=geometry)
        cans.warmup()

        found = targets_from(args, blocks, cans, image, table, geometry)
        if not found:
            raise SystemExit(f"  nothing matching {args.target!r} in view")
        if not args.all:
            found = found[:1]

        print(f"\n  {len(found)} target(s), {1000*args.height:.0f} mm above each:")
        plan = []
        for label, position, confidence in found:
            above = position + np.array([0.0, 0.0, args.height])
            refused = workspace.rejects(above)
            _, nearest = approach.blend(above)
            note = (f"REFUSED: {refused}" if refused
                    else f"{1000*nearest:.0f} mm from the nearest demonstration")
            print(f"    {label:<8} x={position[0]:+.3f} y={position[1]:+.3f} "
                  f"(conf {confidence:.2f}, "
                  f"{table.outside_covered_mm(position):.0f} mm outside the "
                  f"calibration)  {note}")
            if not refused:
                plan.append((label, position))

        if args.dry_run:
            print("\n  dry run; nothing moved")
            return
        if not plan:
            raise SystemExit("\n  every target was refused; nothing to do")

        if not args.no_confirm:
            print(f"\n  The arm will move to {1000*args.height:.0f} mm above the "
                  f"first target and stop. It will not descend.")
            input("  Press ENTER to move, or Ctrl-C to stop. ")

        tuning.install(verbose=False)
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SO101FollowerConfig

        robot = make_robot_from_config(SO101FollowerConfig(
            port=resolve_port([args.follower_port])[0], id="follower"))
        robot.connect()
        home = joints_of(robot)

        args.out.mkdir(parents=True, exist_ok=True)
        for index, (label, position) in enumerate(plan, 1):
            print(f"\n  --- {index}/{len(plan)}: above the {label} ---")
            pose, problem = hover_over(robot, arm, approach, position,
                                       args.height, gripper=GRIPPER_OPEN,
                                       seconds=args.seconds, workspace=workspace)
            if problem:
                print(f"    {problem}")
                continue

            reached = tip_of(arm, robot)
            wanted = position + np.array([0.0, 0.0, args.height])
            missed = 1000 * (reached - wanted)
            print(f"    asked  x={wanted[0]:+.3f} y={wanted[1]:+.3f} z={wanted[2]:+.3f}")
            print(f"    landed x={reached[0]:+.3f} y={reached[1]:+.3f} z={reached[2]:+.3f}")
            print(f"    off by {missed[0]:+.0f}, {missed[1]:+.0f}, {missed[2]:+.0f} mm "
                  f"({np.linalg.norm(missed):.0f} mm)")

            # Let the frames catch up before saving them. Read straight after the
            # move and the newest frame is still the arm mid-settle, which comes
            # out blurred and is no use for judging where anything is.
            time.sleep(0.6)
            for role, stream in cameras.streams.items():
                path = args.out / f"{index:02d}_{label}_{role}.png"
                cv2.imwrite(str(path), stream.read().image)
            print(f"    photographed to {args.out}")

            if index < len(plan) and not args.no_confirm:
                input("    Press ENTER for the next target, or Ctrl-C to stop. ")

        # Park, then look again with nothing in the way. Comparing this against
        # the frame the move was planned from is the only clean way to ask
        # whether the arm disturbed anything: a frame taken while the arm stands
        # over a block has the arm's shadow on that block's box, and reading a
        # shift off that says more about the occlusion than about the table.
        print("\n  parking to look at the table again")
        relax_to(robot, home, gripper=GRIPPER_OPEN)
        time.sleep(1.2)
        after = cameras.streams[args.side].read().image.copy()
        cv2.imwrite(str(args.out / "99_after_side.png"), after)
        settled = [d for d in blocks.detect(after) if d.confidence >= MIN_CONFIDENCE]
        print(f"  {len(settled)} blocks before, {len(seen)} after")
        for label, position in plan:
            if label == "can":
                continue
            near = [d for d in settled if d.colour == label]
            if not near:
                print(f"    the {label} target has no {label} block near it now")
                continue
            closest = min(near, key=lambda d: np.linalg.norm(d.position[:2]
                                                             - position[:2]))
            gap = 1000 * float(np.linalg.norm(closest.position[:2] - position[:2]))
            print(f"    {label} target x={position[0]:+.3f} y={position[1]:+.3f}"
                  f" -> nearest {label} now {gap:.0f} mm away")

    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        cameras.stop()
        if robot is not None:
            print("\n  returning to where it started")
            failed = relax_to(robot, home, gripper=GRIPPER_OPEN)
            if failed:
                print(f"  could not return ({failed})")
            robot.disconnect()
            print("  arm relaxed")


if __name__ == "__main__":
    main()
