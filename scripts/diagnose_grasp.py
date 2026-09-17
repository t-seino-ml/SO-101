"""Which link in the chain is broken? Test them one at a time.

Picking a block needs three things to hold, and so far they have only ever been
tested together, which is why six explanations have come and gone:

  A. the jaws can grasp a block sitting where they just put it, from the pose
     they put it in. If this fails, the block moves when it is released, or the
     grip itself is at fault - and nothing further up matters.

  B. that still works after lifting away and coming back. If A passes and B
     fails, the arm does not return to where it was.

  C. it still works when the target comes from the camera instead of from
     memory. If B passes and C fails, the fault is in the detection-to-arm
     mapping, and only there.

The arm sets a block down itself, so every test starts from a position it knows
exactly rather than one it was told.

    uv run scripts/diagnose_grasp.py
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
    GRIP_DEG,
    OPEN_DEG,
    Workspace,
    free_close,
    glide_to,
    holding,
    joints_of,
    move_to,
    relax_to,
    set_gripper,
    tip_of,
)

PARK_JOINTS = {"shoulder_pan": 0.0, "shoulder_lift": -95.0, "elbow_flex": 90.0,
               "wrist_flex": 65.0, "wrist_roll": 0.0}
CARRY_M = 0.08
LIFT_M = 0.06
MIN_CONFIDENCE = 0.4


def report(name, caught, detail=""):
    mark = "PASS" if caught else "FAIL"
    print(f"\n  [{mark}] {name}{'  ' + detail if detail else ''}")
    return caught


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--at", default="0.26,-0.05")
    parser.add_argument("--grip", type=float, default=GRIP_DEG)
    parser.add_argument("--open", type=float, default=OPEN_DEG)
    parser.add_argument("--side", default="side")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--demos", type=Path, default=Path("data/demos"))
    parser.add_argument("--out", type=Path, default=Path("outputs/diagnose"))
    args = parser.parse_args()

    table = TableFrame.load()
    arm = ArmKinematics()
    approach = ApproachPoses.from_demos(args.demos, kinematics=arm)
    workspace = Workspace()
    detector = BlockDetector(table_frame=table)
    detector.warmup()
    where = np.array([float(v) for v in args.at.split(",")])
    place = np.array([where[0], where[1], table.z_table])
    print(f"  {table}")
    print(f"  working at x={place[0]:+.3f} y={place[1]:+.3f}")

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
    results = {}

    try:
        cameras.wait_for_frames(timeout=25)
        side = cameras.streams[args.side]

        print("\n  Clear every other block off the table first.")
        set_gripper(robot, args.open, seconds=1.0)
        input("  Put ONE block between the jaws and press ENTER: ")
        set_gripper(robot, 2.0, seconds=1.2)
        time.sleep(0.8)
        if joints_of(robot)["gripper"] < 12:
            raise SystemExit("  nothing between the jaws")

        pose, _ = approach.pose_for(place, hover=CARRY_M)
        if pose is None:
            raise SystemExit("  cannot reach above there")
        glide_to(robot, {name: value for name, value in pose.items()
                         if name != "gripper"}, seconds=4.0)
        time.sleep(1.0)
        if move_to(robot, arm, approach, joints_of(robot), place, seconds=1.5,
                   workspace=workspace) is None:
            raise SystemExit("  cannot get down to the table")
        released_at = tip_of(arm, robot)
        release_pose = joints_of(robot)
        print(f"  set down at x={released_at[0]:+.3f} y={released_at[1]:+.3f} "
              f"z={released_at[2]:+.3f}")
        set_gripper(robot, args.open, seconds=1.0)
        time.sleep(0.8)

        # -- A: close again without moving at all --------------------------
        # The angle only, for now. What counts as "closed on nothing" cannot be
        # measured down here - the block is right between the jaws, so the
        # measurement grasps it - so that comes later, up in the air, and every
        # angle is judged against it at the end.
        set_gripper(robot, args.grip, seconds=1.0)
        time.sleep(0.8)
        angles = {"A": joints_of(robot)["gripper"]}
        print(f"\n  A  closed to {angles['A']:.1f} deg without moving")
        set_gripper(robot, args.open, seconds=0.8)

        # -- what closing on nothing looks like, with nothing in reach ------
        move_to(robot, arm, approach, joints_of(robot),
                released_at + np.array([0, 0, LIFT_M]), seconds=1.2,
                workspace=workspace)
        time.sleep(0.5)
        free = free_close(robot, grip=args.grip, open_deg=args.open)
        print(f"  closing on nothing up here settles at {free:.1f} deg")

        # -- B: come back to exactly the pose it released from -------------
        glide_to(robot, {name: value for name, value in release_pose.items()
                         if name != "gripper"}, seconds=1.5)
        time.sleep(0.8)
        back_at = tip_of(arm, robot)
        set_gripper(robot, args.grip, seconds=1.0)
        time.sleep(0.8)
        angles["B"] = joints_of(robot)["gripper"]
        results["A"] = report("A  grasp it again without moving",
                              angles["A"] > free + 4,
                              f"jaws {angles['A']:.1f} deg against {free:.1f} on air")
        results["B"] = report(
            "B  lift away and come back to the same pose",
            angles["B"] > free + 4,
            f"jaws {angles['B']:.1f} deg; returned "
            f"{1000*np.linalg.norm(back_at - released_at):.0f} mm from where it "
            f"let go")
        set_gripper(robot, args.open, seconds=0.8)
        if not results["A"]:
            print("       the block is not where the jaws left it, or the grip "
                  "cannot hold it.\n       Nothing above this can work; stopping.")
            return

        # -- C: park, look, and go where the camera says -------------------
        move_to(robot, arm, approach, joints_of(robot),
                released_at + np.array([0, 0, 0.10]), seconds=1.2,
                workspace=workspace)
        glide_to(robot, {**PARK_JOINTS, "gripper": args.open}, seconds=3.0)
        time.sleep(1.5)
        image = side.read().image.copy()
        cv2.imwrite(str(args.out / "parked.png"), image)
        seen = [d for d in detector.detect(image) if d.confidence >= MIN_CONFIDENCE]
        if not seen:
            raise SystemExit("  no block detected from the side")
        block = min(seen, key=lambda d: np.linalg.norm(
            d.position[:2] - released_at[:2]))
        offset = released_at[:2] - block.position[:2]
        print(f"\n  detected at x={block.position[0]:+.3f} y={block.position[1]:+.3f}"
              f"  ({block.colour}, conf {block.confidence:.2f})")
        print(f"  so the offset measures {1000*offset[0]:+.1f}, "
              f"{1000*offset[1]:+.1f} mm ({1000*np.linalg.norm(offset):.1f} mm)")

        target = np.array([block.position[0] + offset[0],
                           block.position[1] + offset[1], released_at[2]])
        pose, _ = approach.pose_for(target, hover=CARRY_M)
        if pose is None:
            raise SystemExit("  cannot reach above the detected block")
        glide_to(robot, {name: value for name, value in pose.items()
                         if name != "gripper"}, seconds=3.0)
        time.sleep(0.8)
        if move_to(robot, arm, approach, joints_of(robot), target, seconds=1.5,
                   workspace=workspace) is None:
            raise SystemExit("  cannot get down to the detected block")
        arrived = tip_of(arm, robot)
        set_gripper(robot, args.grip, seconds=1.0)
        time.sleep(0.8)
        angles["C"] = joints_of(robot)["gripper"]
        results["C"] = report(
            "C  go where the camera says, via the measured offset",
            angles["C"] > free + 4,
            f"jaws {angles['C']:.1f} deg; arrived "
            f"{1000*np.linalg.norm(arrived[:2] - released_at[:2]):.0f} mm from "
            f"where it let go")
        # The postures matter as much as the positions: the jaws hang off the
        # gripper frame, so two poses that put that frame in the same place with
        # the wrist at different angles put the jaws in different places.
        turned = {name: angles_of - release_pose[name]
                  for name, angles_of in joints_of(robot).items()
                  if name in arm.joint_names}
        worst = max(turned.items(), key=lambda item: abs(item[1]))
        print(f"       joints differ from the release pose by at most "
              f"{worst[1]:+.1f} deg ({worst[0]})")
        set_gripper(robot, args.open, seconds=0.8)

    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        cameras.stop()
        relax_to(robot, home, gripper=args.open)
        robot.disconnect()
        print("\n  arm relaxed")

    if results:
        print("\n  " + "  ".join(f"{name}={'pass' if ok else 'FAIL'}"
                                 for name, ok in results.items()))
        if results.get("A") and not results.get("B"):
            print("  The arm does not come back to where it was.")
        elif results.get("B") and not results.get("C"):
            print("  The arm is fine; the camera-to-arm mapping is not.")
        elif all(results.values()):
            print("  Every link holds here. The failures are position-dependent,")
            print("  so the next question is where on the table it breaks down.")


if __name__ == "__main__":
    main()
