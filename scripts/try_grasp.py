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
HOLDING_DEG = 1.5        # jaws this far wider than their free close are holding
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


def load_jaw_offset(path):
    """Where the jaws close relative to the gripper frame, if it was measured."""
    path = Path(path)
    if not path.is_file():
        print(f"  no {path}; the jaws are assumed to close at the gripper frame,")
        print("  which they do not. Measure it: uv run scripts/fit_jaw_offset.py")
        return np.zeros(3)
    data = json.loads(path.read_text(encoding="utf-8"))
    offset = np.array(data["offset_m"], float)
    print(f"  jaw offset x={1000*offset[0]:+.0f} y={1000*offset[1]:+.0f} "
          f"z={1000*offset[2]:+.0f} mm, from {data['samples']} episode(s)")
    return offset


def attempt(robot, arm, approach, detector, cameras, side, table, home,
            colour, jaw_offset, args):
    """One try at picking a block up. Returns what happened, in a word."""
    import cv2

    before = blocks_on_table(detector, side, table, args.margin_mm)
    wanted = [entry for entry in before
              if colour is None or entry[0] == colour]
    if not wanted:
        print(f"  no {colour} block inside the region "
              f"(seen: {', '.join(sorted({c for c, _, _ in before})) or 'nothing'})")
        return "not on the table"

    colour, target, confidence = max(wanted, key=lambda entry: entry[2])
    print(f"  target: {colour} at x={target[0]:+.3f} y={target[1]:+.3f} "
          f"(confidence {confidence:.2f})")

    # The demonstrated grasp posture for this part of the table, moved sideways
    # onto this block. Its height is kept: what the operator used here says more
    # than the table plane plus a guess at where the jaws sit.
    seed, nearest = approach.blend(target)
    # Where the gripper frame has to be for the jaws to be on the block. The two
    # are not the same point, and the difference - measured over the recordings
    # by fit_jaw_offset - is most of why reaching "to the block" used to put the
    # block outside the jaws.
    grasp_at = target + jaw_offset
    print(f"  nearest demonstration {1000*nearest:.0f} mm away; "
          f"gripper frame to x={grasp_at[0]:+.3f} y={grasp_at[1]:+.3f} "
          f"z={grasp_at[2]:+.3f}")

    at_block = approach.refine(seed, grasp_at)
    above = (None if at_block is None else
             approach.refine(at_block, grasp_at + [0, 0, args.hover]))
    lifted = (None if at_block is None else
              approach.refine(at_block, grasp_at + [0, 0, args.lift]))
    if at_block is None or above is None or lifted is None:
        print("  cannot bring a demonstrated posture onto that block")
        return "unreachable"

    grip = args.grip if args.grip is not None else seed["gripper"]
    if args.dry_run:
        print(f"  would close to {grip:.0f} deg; nothing moved")
        return "dry run"

    print("  approaching")
    glide_to(robot, {**above, "gripper": args.open}, seconds=2.5)
    time.sleep(SETTLE_S)

    # What closing on nothing looks like, measured here rather than assumed. The
    # jaws never quite reach the angle they are told to - friction and a soft
    # position gain leave a degree or two - so falling short of the command is
    # not evidence of holding anything. Closing on air first, in this same pose,
    # gives the number worth comparing against.
    glide_to(robot, {**above, "gripper": grip}, seconds=1.0)
    time.sleep(0.6)
    free = joints_of(robot)["gripper"]
    glide_to(robot, {**above, "gripper": args.open}, seconds=0.8)
    time.sleep(0.4)

    print("  descending")
    glide_to(robot, {**at_block, "gripper": args.open}, seconds=1.5)
    time.sleep(SETTLE_S)
    hurt = strained(robot)
    if hurt:
        print(f"  {hurt} on the way down; backing off")
        glide_to(robot, {**above, "gripper": args.open}, seconds=1.0)
        return "strained"

    # The view the jaws have of the block at the moment before they close: the
    # one picture that says which way, and by how far, an attempt was off.
    out = args.out / colour
    out.mkdir(parents=True, exist_ok=True)
    for role, stream in cameras.streams.items():
        cv2.imwrite(str(out / f"at_block_{role}.png"), stream.read().image)

    glide_to(robot, {**at_block, "gripper": grip}, seconds=1.2)
    time.sleep(SETTLE_S)
    held = joints_of(robot)["gripper"]
    blocked = held > free + HOLDING_DEG
    print(f"  jaws closed to {held:.1f} deg, against {free:.1f} on air"
          f"   ({'holding something' if blocked else 'nothing between them'})")

    glide_to(robot, {**lifted, "gripper": grip}, seconds=1.5)
    time.sleep(1.2)

    # Carry it clear before looking again. Lifted, the arm stands between the
    # side camera and the spot it just left, and a block hidden behind the arm
    # reads exactly like a block that has been picked up - which is how three
    # failures first reported themselves as successes.
    glide_to(robot, {**home, "gripper": grip}, seconds=2.0)
    time.sleep(1.5)
    for role, stream in cameras.streams.items():
        cv2.imwrite(str(out / f"carried_{role}.png"), stream.read().image)

    after = blocks_on_table(detector, side, table, args.margin_mm)
    still_there = [p for c, p, _ in after if c == colour]
    gone = (not still_there or
            min(1000 * np.linalg.norm(target - p) for p in still_there) > MOVED_MM)

    # Both signals have to agree. The jaws alone cannot tell a block from the
    # gripper's own slack, and the camera alone cannot tell a lift from a nudge.
    verdict = ("picked up" if blocked and gone else
               "knocked aside" if gone else "missed")
    print(f"  block left its spot: {'yes' if gone else 'no'}   -> {verdict.upper()}")
    print(f"  photographs in {out}")

    glide_to(robot, {**home, "gripper": args.open}, seconds=1.2)
    time.sleep(0.4)
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colour", nargs="+", default=None,
                        help="which colour(s) to pick up, tried in turn in one "
                             "camera session (default: the clearest block)")
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
    parser.add_argument("--jaw-offset", type=Path,
                        default=Path("data/jaw_offset.json"))
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
    jaw_offset = load_jaw_offset(args.jaw_offset)
    print(f"  {approach}")

    cameras = CameraSet.from_config()
    cameras.start()
    robot = make_robot_from_config(SO101FollowerConfig(
        port=resolve_port([args.follower_port])[0], id="follower"))
    robot.connect()
    home = joints_of(robot)

    verdicts = []
    try:
        cameras.wait_for_frames(timeout=25)
        side = cameras.streams[args.side]
        # One camera session for the whole series. Opening and closing these
        # cameras repeatedly is what wedges them: after a few cycles DirectShow
        # will not reopen them and Media Foundation reports the device
        # invalidated, which takes a physical replug to clear.
        for round_number, colour in enumerate(args.colour or [None], 1):
            label = colour or "whatever is clearest"
            print(f"\n  --- {round_number}/{len(args.colour or [None])}: {label} ---")
            verdicts.append((label, attempt(robot, arm, approach, detector,
                                            cameras, side, table, home,
                                            colour, jaw_offset, args)))
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

    if verdicts:
        print()
        for label, verdict in verdicts:
            print(f"  {label:<12}{verdict}")
        picked = sum(1 for _, verdict in verdicts if verdict == "picked up")
        tried = sum(1 for _, verdict in verdicts
                    if verdict in ("picked up", "knocked aside", "missed"))
        if tried:
            print(f"\n  picked up {picked} of {tried}")


if __name__ == "__main__":
    main()
