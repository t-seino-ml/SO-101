"""Where a block that the jaws could close on appears, from each height.

The visual servo drives a block to one pixel and then closes. That pixel is only
right at one height, because the wrist camera looks forward as well as down: a
block correctly lined up at 25 mm is 20 mm out by the time the arm is down on it.
Measured here, the apparent error jumped from 17 px to 85 px on one 21 mm
descent. And the obvious fix - line it up at grasping height instead - does not
work either, because that close the wrist camera cannot focus: in a frame taken
down there the detector found the blocks across the table and none of the ones
under the gripper.

So the target pixel has to be known as a function of height. This measures it,
with nothing inferred:

  1. you put one block between the jaws
  2. the arm carries it out over the table and sets it down
  3. it lets go - so the block is now lying exactly where the jaws would have
     closed on it, by construction rather than by calibration
  4. it rises straight up, pausing at each height, and looks

Whatever pixel the block occupies at height h is the servo's target at height h.

The same run also settles the jaw offset, which four separate attempts have
disagreed about by up to 70 mm. The arm's x,y when it released is where the jaws
hold a block; the side camera says where that block is detected; the difference
is the offset, measured on today's hardware in one step.

Clear the other blocks off the table first. One block in view means no chance of
tracking the wrong one, and this takes a minute.

    uv run scripts/teach_servo_target.py
    uv run scripts/teach_servo_target.py --at 0.24,-0.05
    uv run scripts/teach_servo_target.py --show
"""

import argparse
import json
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
from so101.policy.motion import (  # noqa: E402
    OPEN_DEG,
    Workspace,
    glide_to,
    gripper_rotation,
    joints_of,
    move_to,
    relax_to,
    set_gripper,
    tip_of,
)

TARGET_PATH = Path("data/servo_target.json")
HEIGHTS_MM = (0, 10, 20, 30, 40, 60)
SQUEEZE_DEG = 2.0
MIN_CONFIDENCE = 0.4
CARRY_HEIGHT_M = 0.08
# Folded back and up, the same pose scripts/calibrate.py parks at. It matters
# more than it sounds: the first run of this script looked at the block with the
# arm still standing over it - "home" being wherever the arm happened to be when
# a block was put in its jaws, which is right above the table - so the side
# camera saw a block half hidden behind a white gripper. Its box shifted by
# 28 mm, that shift was recorded as the jaw offset, and the arm then spent ten
# attempts reaching confidently for the block next to the one it wanted.
PARK_JOINTS = {"shoulder_pan": 0.0, "shoulder_lift": -95.0, "elbow_flex": 90.0,
               "wrist_flex": 65.0, "wrist_roll": 0.0}


def show_stored():
    if not TARGET_PATH.is_file():
        print(f"  {TARGET_PATH} not found; nothing taught yet")
        return
    data = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    print(f"  taught at x={data['released_at'][0]:+.3f} "
          f"y={data['released_at'][1]:+.3f}, {data['samples']} height(s)")
    offset = data.get("offset_m")
    if offset:
        print(f"  jaw offset  x={1000*offset[0]:+.1f}  y={1000*offset[1]:+.1f} mm")
    print(f"\n  {'height':>8}{'target pixel':>18}{'block width':>13}")
    for entry in data["targets"]:
        where = "%.0f, %.0f" % tuple(entry["pixel"])
        print(f"  {1000*entry['height_m']:6.0f}mm{where:>18}"
              f"{entry['width_px']:11.0f}px")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--at", default=None,
                        help="x,y in metres to set the block down at; "
                             "default: the middle of the calibrated region")
    parser.add_argument("--heights", type=float, nargs="+", default=HEIGHTS_MM,
                        help="heights above the grasp to measure at, in mm")
    parser.add_argument("--carry", type=float, default=CARRY_HEIGHT_M)
    parser.add_argument("--open", type=float, default=OPEN_DEG)
    parser.add_argument("--side", default="side")
    parser.add_argument("--wrist", default="wrist")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--demos", type=Path, default=Path("data/demos"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/servo_target"))
    args = parser.parse_args()

    if args.show:
        return show_stored()

    table = TableFrame.load()
    arm = ArmKinematics()
    approach = ApproachPoses.from_demos(args.demos, kinematics=arm)
    workspace = Workspace()
    detector = BlockDetector(weights=args.weights, table_frame=table)
    detector.warmup()

    if args.at:
        where = np.array([float(v) for v in args.at.split(",")])
    else:
        x0, x1, y0, y1 = table.covered
        where = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    place = np.array([where[0], where[1], table.z_table])
    print(f"  {table}")
    print(f"  the block will be set down at x={place[0]:+.3f} y={place[1]:+.3f}")
    refused = workspace.rejects(place + np.array([0, 0, args.carry]))
    if refused:
        raise SystemExit(f"  that is not somewhere the arm may go: {refused}")

    tuning.install(verbose=False)
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    args.out.mkdir(parents=True, exist_ok=True)
    cameras = CameraSet.from_config()
    cameras.start()
    robot = make_robot_from_config(SO101FollowerConfig(
        port=resolve_port([args.follower_port])[0], id="follower"))
    robot.connect()
    home = joints_of(robot)

    targets, released_at, offset = [], None, None
    release_pose, in_gripper = None, None
    try:
        cameras.wait_for_frames(timeout=25)
        wrist = cameras.streams[args.wrist]
        side = cameras.streams[args.side]

        print("\n  Clear every other block off the table first.")
        set_gripper(robot, args.open, seconds=1.0)
        input("  Put ONE block between the jaws and press ENTER: ")
        set_gripper(robot, SQUEEZE_DEG, seconds=1.2)
        time.sleep(0.8)
        held = joints_of(robot)["gripper"]
        print(f"  jaws at {held:.1f} deg")
        if held < 12:
            raise SystemExit("  nothing seems to be between the jaws")

        print("  carrying it out over the table")
        pose, nearest = approach.pose_for(place, hover=args.carry)
        if pose is None:
            raise SystemExit(f"  cannot reach above there "
                             f"(nearest demonstration {1000*nearest:.0f} mm)")
        glide_to(robot, {name: value for name, value in pose.items()
                         if name != "gripper"}, seconds=4.0)
        time.sleep(1.0)

        print("  setting it down")
        lowered = move_to(robot, arm, approach, joints_of(robot), place,
                          seconds=1.5, workspace=workspace)
        if lowered is None:
            raise SystemExit("  cannot get down to the table there")
        released_at = tip_of(arm, robot)
        release_pose = {name: joints_of(robot)[name] for name in arm.joint_names}
        print(f"  released at x={released_at[0]:+.3f} y={released_at[1]:+.3f} "
              f"z={released_at[2]:+.3f}")
        print(f"  wrist_roll {release_pose['wrist_roll']:+.1f} deg, "
              f"wrist_flex {release_pose['wrist_flex']:+.1f} deg")
        set_gripper(robot, args.open, seconds=1.0)
        time.sleep(0.8)

        # Rise in steps, tallest first: high up the block is in focus and easy to
        # find, and each step down only has to follow it a little way.
        last = None
        for millimetres in sorted(args.heights, reverse=True):
            height = released_at[2] + millimetres / 1000
            moved = move_to(robot, arm, approach, joints_of(robot),
                            np.array([released_at[0], released_at[1], height]),
                            seconds=1.0, workspace=workspace)
            if moved is None:
                print(f"  {millimetres:.0f} mm: cannot get there")
                continue
            time.sleep(0.8)                     # let the frame catch up
            image = wrist.read().image.copy()
            found = [d for d in detector.detect(image)
                     if d.confidence >= MIN_CONFIDENCE]
            if not found:
                print(f"  {millimetres:6.0f} mm: nothing detected "
                      f"(too close to focus?)")
                continue
            block = (min(found, key=lambda d: np.linalg.norm(d.pixel - last))
                     if last is not None else
                     max(found, key=lambda d: d.width_px))
            last = block.pixel
            targets.append({"height_m": millimetres / 1000,
                            "pixel": block.pixel.tolist(),
                            "width_px": float(block.width_px),
                            "colour": block.colour})
            print(f"  {millimetres:6.0f} mm: block at "
                  f"({block.pixel[0]:.0f}, {block.pixel[1]:.0f}), "
                  f"{block.width_px:.0f} px wide")
            cv2.circle(image, tuple(int(v) for v in block.pixel), 6, (60, 220, 60), -1)
            x0, y0, x1, y1 = (int(v) for v in block.box)
            cv2.rectangle(image, (x0, y0), (x1, y1), (60, 220, 60), 2)
            cv2.imwrite(str(args.out / f"h{millimetres:03.0f}.png"), image)

        # And the offset, from the side camera - but only once the arm is
        # genuinely out of the shot. Rising is not enough and neither is going
        # "home", which is wherever the arm was left when the block was handed
        # to it. It has to park.
        print("\n  parking so the block can be seen unobstructed")
        move_to(robot, arm, approach, joints_of(robot),
                np.array([released_at[0], released_at[1],
                          released_at[2] + 0.10]), seconds=1.2,
                workspace=workspace)
        glide_to(robot, {**PARK_JOINTS, "gripper": args.open}, seconds=3.0)
        time.sleep(1.5)
        image = side.read().image.copy()
        cv2.imwrite(str(args.out / "side_after.png"), image)
        seen = [d for d in detector.detect(image) if d.confidence >= MIN_CONFIDENCE]
        if seen:
            block = min(seen, key=lambda d: np.linalg.norm(
                d.position[:2] - released_at[:2]))
            offset = released_at[:2] - block.position[:2]
            print(f"  the block is detected at x={block.position[0]:+.3f} "
                  f"y={block.position[1]:+.3f} ({block.colour})")
            print(f"  the jaws held it at   x={released_at[0]:+.3f} "
                  f"y={released_at[1]:+.3f}")
            print(f"\n  so the jaw offset is x={1000*offset[0]:+.1f} "
                  f"y={1000*offset[1]:+.1f} mm "
                  f"({1000*np.linalg.norm(offset):.1f} mm)")
            # In the base frame that number is only good for this posture. The
            # jaws are fixed to the gripper, so turn it into the gripper's own
            # frame, where it is a constant, and it can be turned back out at
            # whatever posture the next grasp happens to use.
            rotation = gripper_rotation(arm, release_pose)
            in_gripper = rotation.T @ np.append(offset, 0.0)
            print(f"  which in the gripper's own frame is "
                  f"{1000*in_gripper[0]:+.1f}, {1000*in_gripper[1]:+.1f}, "
                  f"{1000*in_gripper[2]:+.1f} mm")
        else:
            print("  no block detected from the side; cannot fit the offset")

    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        cameras.stop()
        failed = relax_to(robot, home, gripper=args.open)
        if failed:
            print(f"  could not return ({failed})")
        robot.disconnect()
        print("  arm relaxed")

    if not targets:
        raise SystemExit("\n  nothing measured")
    targets.sort(key=lambda entry: entry["height_m"])
    TARGET_PATH.write_text(json.dumps({
        "targets": targets,
        "released_at": released_at.tolist(),
        "release_pose": release_pose,
        "offset_m": offset.tolist() if offset is not None else None,
        # The one to use. See the note where it is computed.
        "offset_gripper_m": in_gripper.tolist() if offset is not None else None,
        "samples": len(targets),
        "camera": args.wrist,
    }, indent=2), encoding="utf-8")
    print(f"\n  saved {TARGET_PATH}")
    print(f"\n  {'height':>8}{'target pixel':>18}   how far it moves per mm of descent")
    for first, second in zip(targets, targets[1:]):
        step = 1000 * (second["height_m"] - first["height_m"])
        shift = np.array(second["pixel"]) - np.array(first["pixel"])
        where = "%.0f, %.0f" % tuple(first["pixel"])
        print(f"  {1000*first['height_m']:6.0f}mm{where:>18}"
              f"   {np.linalg.norm(shift)/step:5.1f} px/mm")
    top = targets[-1]
    where = "%.0f, %.0f" % tuple(top["pixel"])
    print(f"  {1000*top['height_m']:6.0f}mm{where:>18}")


if __name__ == "__main__":
    main()
