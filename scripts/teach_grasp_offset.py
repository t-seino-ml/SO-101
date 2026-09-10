"""Measure the gap between where a block is detected and where it must be grasped.

Two errors survive a good homography, and neither shows up in its residuals:

- Parallax. The side camera sees a 25 mm cube from an angle, so the detection box
  centres nearer its top face than its footprint. The offset varies across the
  image, and a homography cannot represent that.
- The gripper frame. Forward kinematics reports the URDF's gripper_frame_link,
  which is not where the jaws actually close on a block.

Both are systematic, so both can be measured and subtracted - but only against
ground truth, and the ground truth here is a human driving the gripper to a pose
that would genuinely grasp the block.

For each block: the detector says where it thinks it is, you drive the jaws around
it with the leader, and the difference is recorded. Do several across the working
area; the average becomes the offset applied to every detection afterwards.

    uv run scripts/teach_grasp_offset.py
    uv run scripts/teach_grasp_offset.py --blocks 6
    uv run scripts/teach_grasp_offset.py --show      # print the stored offset
"""

import argparse
import json
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


OFFSET_PATH = Path("data/grasp_offset.json")
TELEOP_HZ = 60


def joints_of(robot):
    return {key.removesuffix(".pos"): float(value)
            for key, value in robot.get_observation().items() if key.endswith(".pos")}


def teleop_until_enter(robot, teleop, arm, detector, stream, target, fps=TELEOP_HZ):
    """Drive with the leader until ENTER, showing the running offset."""
    from lerobot.utils.utils import enter_pressed, move_cursor_up

    period = 1.0 / fps
    while True:
        started = time.perf_counter()
        robot.send_action(teleop.get_action())
        current = joints_of(robot)
        tip = arm.forward({name: current[name] for name in arm.joint_names})
        delta = 1000 * (tip - target)
        print(f"    gripper x={tip[0]:+.3f} y={tip[1]:+.3f} z={tip[2]:+.3f}   "
              f"offset from the detection: x={delta[0]:+5.0f} y={delta[1]:+5.0f} "
              f"z={delta[2]:+5.0f} mm   ", end="", flush=True)
        if enter_pressed():
            print()
            return tip
        print()
        move_cursor_up(1)
        time.sleep(max(0.0, period - (time.perf_counter() - started)))


def show_stored():
    if not OFFSET_PATH.is_file():
        print(f"  {OFFSET_PATH} not found; no offset has been taught yet")
        return
    data = json.loads(OFFSET_PATH.read_text(encoding="utf-8"))
    offset = data["offset_m"]
    print(f"  offset  x={1000*offset[0]:+.1f}  y={1000*offset[1]:+.1f}  "
          f"z={1000*offset[2]:+.1f} mm   from {data['samples']} sample(s)")
    spread = data.get("spread_mm")
    if spread:
        print(f"  spread  x={spread[0]:.1f}  y={spread[1]:.1f}  z={spread[2]:.1f} mm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=None,
                        help="default: the newest trained detector under runs/")
    parser.add_argument("--camera", default="side")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--leader-port", default="COM3")
    parser.add_argument("--blocks", type=int, default=4,
                        help="how many blocks to teach against")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.show:
        return show_stored()


    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.so_leader import SO101LeaderConfig

    tuning.install(verbose=False)
    table = TableFrame.load()
    arm = ArmKinematics()
    detector = BlockDetector(weights=args.weights, table_frame=table)
    detector.warmup()
    print(f"  {table}")

    cameras = CameraSet.from_config()
    cameras.start()
    robot = make_robot_from_config(SO101FollowerConfig(
        port=resolve_port([args.follower_port])[0], id="follower"))
    teleop = make_teleoperator_from_config(SO101LeaderConfig(
        port=resolve_port([args.leader_port])[0], id="leader"))
    robot.connect()
    teleop.connect()
    home = joints_of(robot)

    samples = []
    try:
        cameras.wait_for_frames(timeout=25)
        stream = cameras.streams[args.camera]
        blocks = [b for b in detector.detect(stream.read().image.copy())
                  if b.trustworthy][:args.blocks]
        if not blocks:
            raise SystemExit("  No blocks inside the calibrated region.")

        print(f"\n  {len(blocks)} block(s) to teach against.")
        print("  For each one: drive the follower with the leader until the jaws")
        print("  are exactly where they would need to be to close on that block -")
        print("  straddling it, at grasping height - then press ENTER.\n")

        for index, block in enumerate(blocks, 1):
            print(f"  [{index}/{len(blocks)}] {block.colour} detected at "
                  f"x={block.position[0]:+.3f} y={block.position[1]:+.3f} "
                  f"z={block.position[2]:+.3f}")
            tip = teleop_until_enter(robot, teleop, arm, detector, stream,
                                     block.position)
            samples.append(tip - block.position)
            print(f"      recorded x={1000*samples[-1][0]:+.0f} "
                  f"y={1000*samples[-1][1]:+.0f} z={1000*samples[-1][2]:+.0f} mm\n")
    finally:
        cameras.stop()
        teleop.disconnect()
        try:
            current = joints_of(robot)
            goal = dict(current)
            goal.update({name: home[name] for name in arm.joint_names})
            robot.send_action({f"{k}.pos": v for k, v in goal.items()})
            time.sleep(1.0)
        finally:
            robot.disconnect()

    if not samples:
        raise SystemExit("  Nothing recorded.")

    samples = np.array(samples)
    offset = samples.mean(axis=0)
    spread = samples.max(axis=0) - samples.min(axis=0)
    print(f"  mean offset  x={1000*offset[0]:+.1f}  y={1000*offset[1]:+.1f}  "
          f"z={1000*offset[2]:+.1f} mm   over {len(samples)} sample(s)")
    print(f"  spread       x={1000*spread[0]:.1f}  y={1000*spread[1]:.1f}  "
          f"z={1000*spread[2]:.1f} mm")
    if len(samples) > 1 and max(spread) > 0.015:
        print("\n  The samples disagree by more than 15 mm, so this is not one fixed")
        print("  offset - it varies across the table, which points at parallax or a")
        print("  calibration that does not cover where those blocks are.")

    OFFSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    OFFSET_PATH.write_text(json.dumps({
        "offset_m": offset.tolist(),
        "samples": len(samples),
        "spread_mm": (1000 * spread).tolist(),
        "camera": args.camera,
    }, indent=2), encoding="utf-8")
    print(f"\n  saved {OFFSET_PATH}")


if __name__ == "__main__":
    main()
