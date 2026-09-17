"""Pick a block up, put it back, and say whether it worked. Nothing goes near the can.

Phase 5 on its own: the descent and the grasp, with the drop left out, so a
failure here cannot also be a failure of something else. The block is placed back
where it came from afterwards, which means a success rate can be measured over
many attempts without anyone restocking the table.

This tries open loop first - hover, straight down, close - and no visual servo.
Hovering was measured at 5 mm against a 20 mm block, so it is worth finding out
whether the closed loop is needed at all before paying for it: measuring the
image scale costs four arm movements and several seconds every single pick, and
it is most of why the existing pick_colour.py is slow. If open loop turns out to
miss, Phase 7 puts the servo back, and then it will be there for a reason.

    uv run scripts/pick_block.py --colour red
    uv run scripts/pick_block.py --colour red --trials 10
    uv run scripts/pick_block.py --colour red --dry-run
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
    descend_to,
    jaw_target,
    free_close,
    holding,
    hover_over,
    joints_of,
    lift_by,
    relax_to,
    set_gripper,
    strained,
    tip_of,
)

HOVER_M = 0.08
STAGE_M = 0.025        # pause here on the way down, to check the load
LIFT_M = 0.06
MIN_CONFIDENCE = 0.5
SETTLE_S = 0.8


def clean_look(cameras, detector, side, pause=0.8):
    """A frame with the arm parked, and what is on the table in it."""
    time.sleep(pause)
    image = cameras.streams[side].read().image.copy()
    return image, [d for d in detector.detect(image)
                   if d.confidence >= MIN_CONFIDENCE]


def attempt(robot, arm, approach, detector, cameras, args, workspace,
            jaw_offset, home, trial):
    """One pick, then put it back. Returns (outcome, seconds)."""
    started = time.perf_counter()
    image, seen = clean_look(cameras, detector, args.side)
    wanted = [d for d in seen if d.colour == args.colour]
    if not wanted:
        return "no block of that colour", time.perf_counter() - started

    target = min(wanted, key=lambda d: float(np.linalg.norm(d.position[:2])))
    block = target.position
    print(f"    {args.colour} at x={block[0]:+.3f} y={block[1]:+.3f} "
          f"(conf {target.confidence:.2f}, "
          f"{target.extrapolated_mm:.0f} mm outside the calibration)")

    # Where the jaws end up depends on the posture the solver lands on, so the
    # offset has to be turned into this posture before it means anything. See
    # motion.jaw_target.
    if jaw_offset.shape == (3,) and args.in_gripper_frame:
        grasp_at, _, nearest = jaw_target(arm, approach, block, jaw_offset,
                                          hover=args.hover)
        if grasp_at is None:
            return "cannot work out a posture to reach it from", \
                   time.perf_counter() - started
        print(f"    aiming the gripper frame at x={grasp_at[0]:+.3f} "
              f"y={grasp_at[1]:+.3f} "
              f"({1000*np.linalg.norm(grasp_at[:2]-block[:2]):.0f} mm off the "
              f"block, {1000*nearest:.0f} mm from the nearest demonstration)")
    else:
        grasp_at = block + jaw_offset

    pose, problem = hover_over(robot, arm, approach, grasp_at, args.hover,
                               gripper=args.open, seconds=args.seconds,
                               workspace=workspace)
    if problem:
        return f"hover: {problem}", time.perf_counter() - started

    # What closing on nothing looks like, measured up here where nothing can be
    # between the jaws.
    free = free_close(robot, grip=args.grip, open_deg=args.open)
    print(f"    closing on nothing up here settles at {free:.1f} deg")

    for height in (grasp_at[2] + args.stage, grasp_at[2]):
        pose, problem = descend_to(robot, arm, approach, joints_of(robot), height,
                                   workspace=workspace)
        if problem:
            relax_to(robot, home, gripper=args.open)
            return f"descent: {problem}", time.perf_counter() - started

    landed = tip_of(arm, robot)
    print(f"    asked z={1000*grasp_at[2]:+.0f} mm, reached "
          f"{1000*landed[2]:+.0f} mm; off sideways by "
          f"{1000*np.linalg.norm(landed[:2] - grasp_at[:2]):.0f} mm")

    if args.look_only:
        # Down at grasping height with the jaws still open, so a photograph shows
        # where they actually sit against the block. Closing here would shove the
        # block and destroy the very thing being measured.
        time.sleep(0.8)
        for role, stream in cameras.streams.items():
            path = args.out / f"look_{trial:02d}_{role}.png"
            cv2.imwrite(str(path), stream.read().image)
        print(f"    photographed at grasping height to {args.out}")
        lift_by(robot, arm, approach, joints_of(robot), args.lift,
                workspace=workspace)
        relax_to(robot, home, gripper=args.open)
        return "looked", time.perf_counter() - started

    set_gripper(robot, args.grip, seconds=1.0)
    time.sleep(SETTLE_S)
    held = joints_of(robot)["gripper"]
    caught = holding(robot, free)
    print(f"    jaws closed to {held:.1f} deg against {free:.1f} on air"
          f"   ({'holding something' if caught else 'nothing between them'})")

    pose, problem = lift_by(robot, arm, approach, joints_of(robot), args.lift,
                            workspace=workspace)
    if problem:
        set_gripper(robot, args.open, seconds=0.8)
        relax_to(robot, home, gripper=args.open)
        return f"lift: {problem}", time.perf_counter() - started

    if not caught:
        relax_to(robot, home, gripper=args.open)
        return "missed", time.perf_counter() - started

    # Still holding it after the lift? A block that slips out on the way up was
    # never really gripped, and the jaws closing further is what shows it.
    if not holding(robot, free):
        relax_to(robot, home, gripper=args.open)
        return "dropped on the lift", time.perf_counter() - started

    elapsed = time.perf_counter() - started
    if args.hold:
        print("    holding; leaving it in the jaws")
        return "picked", elapsed

    print("    putting it back")
    for height in (grasp_at[2] + args.stage, grasp_at[2]):
        _, problem = descend_to(robot, arm, approach, joints_of(robot), height,
                                workspace=workspace)
        if problem:
            print(f"    {problem}")
            break
    set_gripper(robot, args.open, seconds=0.8)
    time.sleep(0.4)
    lift_by(robot, arm, approach, joints_of(robot), args.lift, workspace=workspace)
    relax_to(robot, home, gripper=args.open)
    return "picked", elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colour", required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--hover", type=float, default=HOVER_M)
    parser.add_argument("--stage", type=float, default=STAGE_M)
    parser.add_argument("--lift", type=float, default=LIFT_M)
    parser.add_argument("--grip", type=float, default=GRIP_DEG)
    parser.add_argument("--open", type=float, default=OPEN_DEG)
    parser.add_argument("--seconds", type=float, default=2.5)
    parser.add_argument("--hold", action="store_true",
                        help="keep the block in the jaws instead of putting it back")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-confirm", action="store_true")
    parser.add_argument("--side", default="side")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--demos", type=Path, default=Path("data/demos"))
    parser.add_argument("--jaw-offset", type=Path, default=None,
                        help="default: data/grasp_offset.json if it exists, "
                             "else data/jaw_offset.json")
    parser.add_argument("--offset-mm", default=None,
                        help="override the stored offset, as x,y,z in mm "
                             "(e.g. 0,0,0 to aim straight at the detection)")
    parser.add_argument("--roll", type=float, default=None,
                        help="wrist roll to pin the posture at, in degrees; "
                             "default: whatever the offset was taught at")
    parser.add_argument("--base-frame-offset", dest="in_gripper_frame",
                        action="store_false",
                        help="add the offset in the arm's frame, the way it was "
                             "done before the posture problem was found")
    parser.add_argument("--look-only", action="store_true",
                        help="descend and photograph, but never close the jaws - "
                             "for measuring where they actually end up")
    parser.add_argument("--out", type=Path, default=Path("outputs/pick_block"))
    args = parser.parse_args()

    table = TableFrame.load()
    arm = ArmKinematics()
    approach = ApproachPoses.from_demos(args.demos, kinematics=arm)
    workspace = Workspace()
    detector = BlockDetector(weights=args.weights, table_frame=table)
    detector.warmup()
    # Three files claim to hold this offset and they disagree by up to 70 mm, so
    # the order matters. Best first:
    #
    #   servo_target.json   the arm carried a block out, set it down and let go,
    #                       then looked. Where it released IS where the jaws hold
    #                       a block - by construction, not by calibration.
    #   grasp_offset.json   a person drove the jaws onto a block. A measurement,
    #                       but their four samples scattered by 77 mm.
    #   jaw_offset.json     inferred from the recordings by deciding which block
    #                       vanished between two episodes. Scatters by 17 mm with
    #                       outliers to 64 mm; some of those matches are wrong.
    source = args.jaw_offset
    if source is None:
        for candidate in (Path("data/servo_target.json"),
                          Path("data/grasp_offset.json"),
                          Path("data/jaw_offset.json")):
            if candidate.is_file() and json.loads(
                    candidate.read_text(encoding="utf-8")).get("offset_m"):
                source = candidate
                break
        else:
            raise SystemExit("  No offset measured. Run: "
                             "uv run scripts/teach_servo_target.py")
    if args.offset_mm is not None:
        jaw_offset = np.array([float(v) for v in args.offset_mm.split(",")]) / 1000
        source = f"the command line ({args.offset_mm} mm)"
    else:
        stored = json.loads(source.read_text(encoding="utf-8"))
        in_gripper = stored.get("offset_gripper_m")
        if args.roll is None:
            args.roll = (stored.get("release_pose") or {}).get("wrist_roll")
        if in_gripper and args.in_gripper_frame:
            jaw_offset = np.array(in_gripper, float)
            frame = "the gripper's frame"
        else:
            jaw_offset = np.array(stored["offset_m"], float)
            frame = "the arm's frame"
            args.in_gripper_frame = False
        # servo_target.json records x,y only: it measures where the jaws put a
        # block down, and the height that happened at is the table's, not a
        # property of the gripper.
        if len(jaw_offset) == 2:
            jaw_offset = np.append(jaw_offset, 0.0)

    print(f"  {table}")
    print(f"  offset from {source}, in {frame}")
    if args.roll is not None and args.in_gripper_frame:
        print(f"  the wrist roll is pinned at {args.roll:+.1f} deg, the angle it "
              f"was taught at")
    print(f"  jaws sit {1000*jaw_offset[0]:+.0f}, {1000*jaw_offset[1]:+.0f}, "
          f"{1000*jaw_offset[2]:+.0f} mm from the gripper frame")
    print(f"  so a block at the table height is grasped at "
          f"z={1000*(table.z_table + jaw_offset[2]):+.1f} mm")

    args.out.mkdir(parents=True, exist_ok=True)
    cameras = CameraSet.from_config()
    cameras.start()
    robot = None
    results = []
    try:
        cameras.wait_for_frames(timeout=25)
        image, seen = clean_look(cameras, detector, args.side, pause=1.2)
        cv2.imwrite(str(args.out / "before.png"), image)
        matching = [d for d in seen if d.colour == args.colour]
        print(f"\n  {len(seen)} blocks on the table, {len(matching)} of them "
              f"{args.colour}")
        for d in matching:
            print(f"    x={d.position[0]:+.3f} y={d.position[1]:+.3f} "
                  f"conf {d.confidence:.2f}")
        if not matching:
            raise SystemExit(f"  no {args.colour} block to pick")

        if args.dry_run:
            print("\n  dry run; nothing moved")
            return

        if not args.no_confirm:
            print(f"\n  The arm will descend to "
                  f"z={1000*(table.z_table + jaw_offset[2]):+.1f} mm and close.")
            input("  Press ENTER to start, or Ctrl-C to stop. ")

        tuning.install(verbose=False)
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SO101FollowerConfig

        robot = make_robot_from_config(SO101FollowerConfig(
            port=resolve_port([args.follower_port])[0], id="follower"))
        robot.connect()
        home = joints_of(robot)

        for trial in range(1, args.trials + 1):
            print(f"\n  --- attempt {trial}/{args.trials} ---")
            outcome, elapsed = attempt(robot, arm, approach, detector, cameras,
                                       args, workspace, jaw_offset, home, trial)
            print(f"    {outcome} in {elapsed:.1f} s")
            results.append((outcome, elapsed))
            if args.hold:
                break

        image, seen = clean_look(cameras, detector, args.side, pause=1.2)
        cv2.imwrite(str(args.out / "after.png"), image)
        print(f"\n  {len(seen)} blocks on the table afterwards")

    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        cameras.stop()
        if robot is not None:
            failed = relax_to(robot, home, gripper=args.open)
            if failed:
                print(f"  could not return ({failed})")
            robot.disconnect()
            print("  arm relaxed")

    if results:
        picked = sum(1 for outcome, _ in results if outcome == "picked")
        print(f"\n  {picked} of {len(results)} picked "
              f"({100*picked/len(results):.0f}%)")
        times = [seconds for outcome, seconds in results if outcome == "picked"]
        if times:
            print(f"  {sum(times)/len(times):.1f} s per successful pick "
                  f"(pick and replace, no drop)")
        for outcome in sorted({o for o, _ in results}):
            count = sum(1 for o, _ in results if o == outcome)
            print(f"    {outcome:<28}{count}")


if __name__ == "__main__":
    main()
