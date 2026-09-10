"""Check the calibration by reaching for real blocks, without grasping them.

A small fitting residual is not the same as the arm arriving where a block is.
Two errors survive a good fit:

- Parallax. The side camera sees the block from an angle, so the detection box
  centres on the block's upper face rather than its footprint. Calibration absorbs
  most of this - the touched blocks were seen the same way - but the offset varies
  across the image, and a homography cannot represent that.
- Where the gripper actually is. Forward kinematics reports the URDF's
  gripper_frame_link, which need not be where the jaws close on a block.

So this drives the arm to hover a few millimetres above each detected block, with
the gripper open, and asks you to read off the offset. Nothing is grasped and
nothing is moved, so a bad calibration costs a nudged block at worst.

    uv run scripts/verify_calibration.py                  # every block, in turn
    uv run scripts/verify_calibration.py --colour red     # just the red ones
    uv run scripts/verify_calibration.py --hover 0.02     # hover higher
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
from so101.policy import ArmKinematics, BlockDetector, TableFrame  # noqa: E402


HOVER = 0.012          # metres above the block's reported top
# Pointing the gripper down costs reach: at this table height only 19 of 35
# workspace points solve at 8 cm, against 33 at 3 cm. See so101.policy.pick_place.
APPROACH = 0.035       # travel height between blocks
GRIPPER_OPEN = 40.0
STEP_DEG = 1.5
STEP_DELAY = 0.02
LOAD_ABORT = 700
# The servos run proportional position control, so they settle wherever their
# torque balances gravity and friction - short of the commanded angle. Measured on
# this arm: 15-20 mm of undershoot at the gripper, mostly along x. Commanding the
# target plus the measured shortfall removes it.
#
# Only part of the residual is applied each round. Correcting by the full amount
# overshoots and the error oscillates around +/-8 mm indefinitely; at 0.6 it
# converges monotonically - 19, 7, 3 mm - and at 0.4 it converges too slowly.
SETTLE_ROUNDS = 4
SETTLE_GAIN = 0.6
SETTLE_TOLERANCE_MM = 4.0


def joints_of(robot):
    return {key.removesuffix(".pos"): float(value)
            for key, value in robot.get_observation().items() if key.endswith(".pos")}


def strained(robot):
    """The most loaded joint, if any is past the abort threshold."""
    for name in robot.bus.motors:
        try:
            load = robot.bus.read("Present_Load", name, normalize=False)
        except Exception:  # noqa: BLE001 - a dropped packet is not a fault
            continue
        if abs(load) > LOAD_ABORT:
            return name, load
    return None


def move_joints(robot, target, arm):
    start = joints_of(robot)
    goal = dict(start)
    goal.update(target)
    goal["gripper"] = GRIPPER_OPEN
    travel = max(abs(goal[name] - start[name]) for name in goal)
    steps = max(1, int(travel / STEP_DEG))
    for step in range(1, steps + 1):
        fraction = step / steps
        robot.send_action({f"{name}.pos": start[name]
                           + (goal[name] - start[name]) * fraction for name in goal})
        time.sleep(STEP_DELAY)
        hurt = strained(robot)
        if hurt:
            return False, f"{hurt[0]} load {hurt[1]}"
    time.sleep(0.3)
    return True, ""


def tip_of(robot, arm):
    current = joints_of(robot)
    return arm.forward({name: current[name] for name in arm.joint_names})


def move_to(robot, arm, position, settle=True):
    """Move the gripper frame to `position`, then command away the shortfall."""
    position = np.asarray(position, float)
    current = joints_of(robot)
    seed = {name: current[name] for name in arm.joint_names}
    solution = arm.inverse(position, seed_deg=seed)
    if solution is None:
        return False, "unreachable"
    ok, detail = move_joints(robot, solution, arm)
    if not ok or not settle:
        return ok, detail

    for _ in range(SETTLE_ROUNDS - 1):
        error = position - tip_of(robot, arm)
        if 1000 * float(np.linalg.norm(error)) <= SETTLE_TOLERANCE_MM:
            break
        corrected = arm.inverse(position + SETTLE_GAIN * error, seed_deg={
            name: joints_of(robot)[name] for name in arm.joint_names})
        if corrected is None:
            break
        ok, detail = move_joints(robot, corrected, arm)
        if not ok:
            return ok, detail
    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=None,
                        help="default: the newest trained detector under runs/")
    parser.add_argument("--camera", default="side")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--colour", default=None, help="only check this colour")
    parser.add_argument("--hover", type=float, default=HOVER,
                        help="metres to hover above the block")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many blocks")
    args = parser.parse_args()



    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    tuning.install(verbose=False)
    table = TableFrame.load()
    arm = ArmKinematics()
    detector = BlockDetector(weights=args.weights, table_frame=table)
    detector.warmup()
    print(f"  {table}")

    with CameraSet.from_config() as cameras:
        cameras.wait_for_frames(timeout=25)
        image = cameras.streams[args.camera].read().image.copy()

    blocks = detector.detect(image, colours=[args.colour] if args.colour else None)
    reachable = [b for b in blocks
                 if arm.inverse(b.position + [0, 0, args.hover]) is not None]
    print(f"  {len(blocks)} detected, {len(reachable)} reachable")
    if not reachable:
        raise SystemExit("  Nothing reachable to check.")
    if args.limit:
        reachable = reachable[:args.limit]

    print("\n  The arm will hover just above each block with the gripper open.")
    print("  Look at the gap between the jaws and the block, and say how far off it")
    print("  is. Press ENTER to move to the next one, or 'q' then ENTER to stop.\n")

    robot = make_robot_from_config(SO101FollowerConfig(
        port=resolve_port([args.follower_port])[0], id="follower"))
    robot.connect()
    # Where the arm was before anything moved it, so it can be put back. Leaving
    # it stopped over a block means the next run starts from a pose that may not
    # solve, and it sits in the camera's view of the table.
    home = joints_of(robot)
    print(f"  home pose recorded: "
          + " ".join(f"{k}={v:+.0f}" for k, v in home.items()
                     if k in arm.joint_names))
    try:
        for index, block in enumerate(reachable, 1):
            x, y, z = block.position
            note = ("" if block.trustworthy
                    else f"   {block.extrapolated_mm:.0f}mm OUTSIDE the calibrated "
                         "region - expect it to be off")
            print(f"  [{index}/{len(reachable)}] {block.colour} "
                  f"x={x:+.3f} y={y:+.3f}{note}")

            ok, detail = move_to(robot, arm, [x, y, z + APPROACH])
            if not ok:
                print(f"      approach failed: {detail}")
                continue
            ok, detail = move_to(robot, arm, [x, y, z + args.hover])
            if not ok:
                print(f"      descent failed: {detail}")
                continue

            tip = tip_of(robot, arm)
            wanted = np.array([x, y, z + args.hover])
            error = 1000 * (tip - wanted)
            print(f"      gripper at x={tip[0]:+.3f} y={tip[1]:+.3f} z={tip[2]:+.3f}")
            print(f"      off by x={error[0]:+.0f} y={error[1]:+.0f} "
                  f"z={error[2]:+.0f} mm")
            if input("      ENTER for the next block, q to stop: ").strip().lower() == "q":
                break
            move_to(robot, arm, [x, y, z + APPROACH])
    finally:
        try:
            print()
            print("  returning to the home pose")
            # Lift clear of the table first: travelling home at block height would
            # drag the gripper through whatever is in the way.
            tip = tip_of(robot, arm)
            move_to(robot, arm, [tip[0], tip[1], tip[2] + APPROACH], settle=False)
            move_joints(robot, {name: home[name] for name in arm.joint_names}, arm)
        except Exception as error:  # noqa: BLE001 - always still relax the arm
            print(f"  could not return home ({error})")
        finally:
            current = joints_of(robot)
            robot.send_action({**{f"{k}.pos": v for k, v in current.items()},
                               "gripper.pos": GRIPPER_OPEN})
            time.sleep(0.5)
            robot.disconnect()
            print("  arm relaxed")

    print("\n  If the jaws sat consistently off in one direction, that is a fixed")
    print("  offset worth correcting. If the error grew towards one side of the")
    print("  table, the calibration points did not cover that side.")


if __name__ == "__main__":
    main()
