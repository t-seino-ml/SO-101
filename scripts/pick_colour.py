"""Put the block of a chosen colour into the can.

This is the task end to end, and the first thing that uses both cameras for what
each is good for. The side camera says which block and roughly where; the wrist
camera, looked at every step, says how far off the arm still is and closes the
gap. Nothing here is learnt except the detector.

Doing the last part closed-loop is what makes the rest tolerable. The homography
only has to be right to a few centimetres, the demonstrated grasp poses only have
to be roughly right, and the offset between the URDF's gripper frame and the jaws
does not have to be right at all - the arm looks at the block and moves until it
is where a block is held, so every one of those errors gets absorbed instead of
accumulating. Detection costs 10 ms, so looking every step is free.

Three numbers come out of the recordings rather than from anyone measuring:
where the jaws hold a block in the wrist view (fit_grasp_pixel), where the jaws
are relative to the gripper frame (fit_jaw_offset), and where the can is - the
pose the demonstrations were in when they let go.

    uv run scripts/pick_colour.py --colour green
    uv run scripts/pick_colour.py --colour red --count 3
    uv run scripts/pick_colour.py --colour blue --dry-run
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

HOVER_M = 0.06           # how far above the grasp to arrive first
STAGES_M = (0.06, 0.025)  # and the heights to line the block up at, in turn
JOG_M = 0.025            # how far to step when measuring the pixel scale
SERVO_ROUNDS = 8
SERVO_GAIN = 0.7         # damped, so a noisy detection cannot overshoot
SERVO_TOLERANCE_PX = 18
MAX_STEP_MM = 25.0
SETTLE_ROUNDS = 6        # how many times to correct for the arm falling short
SETTLE_GAIN = 0.8
SETTLE_TOLERANCE_M = 0.002
MIN_SCALE_PX_PER_MM = 0.5
MAX_SCALE_PX_PER_MM = 40.0
MAX_SCALE_SKEW = 4.0     # how lopsided the two measured directions may be
TRACK_PX = 260           # how far the same block may jump between looks
GRIP_DEG = 8.0           # narrower than a block, so the block stops the jaws
HOLDING_DEG = 4.0        # jaws this far wider than their free close are holding
MIN_CONFIDENCE = 0.5
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


def load_json(path, key, default):
    path = Path(path)
    if not path.is_file():
        return None, f"{path} not found"
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.array(data[key], float), data


def tip_of(arm, robot):
    pose = joints_of(robot)
    return arm.forward({name: pose[name] for name in arm.joint_names})


def move_to(robot, arm, approach, pose, position, seconds=1.2,
            tolerance_mm=1.0):
    """Shift the arm to `position`, keeping the posture it is already in.

    The tolerance matters more than it looks. Inverse kinematics stops as soon
    as it is within it, so a loose one lets a 12 mm step land 5 mm short and in
    a different direction - which is fine for reaching and useless for
    measuring, where the step *is* the measurement.
    """
    # Commanding the joints inverse kinematics asks for does not put the
    # gripper where it was asked to be: the follower settles short of its
    # target under its own weight, by most of a centimetre when it moves
    # sideways. So aim, look at where it actually went, and add the miss back
    # into the aim - a couple of rounds of that and the gap is millimetres.
    goal = np.asarray(position, float)
    aim = goal.copy()
    moved = None
    for round_number in range(SETTLE_ROUNDS):
        candidate = approach.refine(pose, aim, tolerance_mm=tolerance_mm)
        if candidate is None:
            return moved
        moved = candidate
        glide_to(robot, {name: value for name, value in moved.items()
                         if name != "gripper"},
                 seconds=seconds if round_number == 0 else 0.5)
        time.sleep(SETTLE_S if round_number == 0 else 0.4)
        missed = goal - tip_of(arm, robot)
        if np.linalg.norm(missed) < SETTLE_TOLERANCE_M:
            break
        aim = aim + SETTLE_GAIN * missed
    return moved


def target_in_view(detector, stream, colour, near, radius=None):
    """The block of this colour nearest `near`, if one is close enough.

    Following one block from frame to frame means asking where it was last, not
    where the jaws are: with two blocks of every colour on the table, "nearest
    to the target" hops between them as the arm moves, and a hop looks exactly
    like the arm having moved a long way.
    """
    seen = [d for d in detector.detect(stream.read().image.copy())
            if d.colour == colour and d.confidence >= MIN_CONFIDENCE]
    if not seen:
        return None
    best = min(seen, key=lambda d: np.linalg.norm(d.pixel - near))
    if radius is not None and np.linalg.norm(best.pixel - near) > radius:
        return None
    return best


def keep_looking(detector, stream, colour, near, radius=None, tries=4):
    """Ask again before concluding the block is gone; one missed frame is one
    missed frame, not a lost block."""
    for _ in range(tries):
        seen = target_in_view(detector, stream, colour, near, radius)
        if seen is not None:
            return seen
        time.sleep(0.15)
    return None


def measure_scale(robot, arm, approach, detector, wrist, colour, pose, where,
                  jog):
    """Pixels per millimetre here, by stepping the arm and watching the block.

    The wrist camera does not look straight down and the arm's posture changes
    as it reaches, so this is a property of where the arm is standing, not of
    the camera. Measuring it in place costs two small moves.
    """
    base = keep_looking(detector, wrist, colour, where)
    if base is None:
        return None, None, "the block is not in the wrist camera's view"
    start = tip_of(arm, robot)

    pixels, millimetres = [], []
    for axis in (0, 1):
        step = np.zeros(3)
        step[axis] = jog
        moved = move_to(robot, arm, approach, pose, start + step, seconds=1.5)
        if moved is None:
            return None, None, "cannot step far enough to measure the scale"
        time.sleep(0.6)                  # the follower trails its target
        travelled = tip_of(arm, robot) - start
        # If the arm did not really go, the step divides by almost nothing and
        # the scale comes out enormous - which is how a 12 mm jog once implied
        # 50 pixels per millimetre.
        # A little vertical drift is tolerable - a centimetre at arm's length
        # changes what a pixel is worth by a few percent - but a lot of it means
        # the arm went somewhere other than where it was sent.
        if abs(travelled[2]) > 0.012:
            move_to(robot, arm, approach, pose, start, seconds=1.0)
            return None, None, (f"the step also moved {1000*travelled[2]:+.0f} mm "
                                "vertically, so it did not go where it was sent")
        if np.linalg.norm(travelled[:2]) < 0.6 * jog:
            move_to(robot, arm, approach, pose, start, seconds=1.0)
            return None, None, (f"the arm moved {1000*np.linalg.norm(travelled):.0f}"
                                f" mm of the {1000*jog:.0f} mm asked for")
        seen = keep_looking(detector, wrist, colour, base.pixel, TRACK_PX)
        if seen is None:
            move_to(robot, arm, approach, pose, start, seconds=1.0)
            return None, None, "lost track of the block while measuring"
        shift = seen.pixel - base.pixel
        print(f"      axis {axis}: arm moved "
              f"{1000*travelled[0]:+.1f}, {1000*travelled[1]:+.1f}, "
              f"{1000*travelled[2]:+.1f} mm -> block moved "
              f"{shift[0]:+.0f}, {shift[1]:+.0f} px")
        pixels.append(shift)
        millimetres.append(1000 * travelled[:2])
        move_to(robot, arm, approach, pose, start, seconds=1.0)
        base = keep_looking(detector, wrist, colour, base.pixel,
                            TRACK_PX) or base
    scale = np.column_stack(pixels) @ np.linalg.inv(np.column_stack(millimetres))
    strength = np.linalg.norm(scale, axis=0)
    if min(strength) < MIN_SCALE_PX_PER_MM or max(strength) > MAX_SCALE_PX_PER_MM:
        return None, None, (f"scale came out {strength[0]:.1f} and "
                            f"{strength[1]:.1f} px/mm, which cannot be right")
    # If the two directions come out nearly parallel in the image, inverting
    # this amplifies a few pixels of detection noise into centimetres of
    # movement, and the correction walks away instead of closing in.
    spread = np.linalg.svd(scale, compute_uv=False)
    if spread[0] / max(spread[1], 1e-9) > MAX_SCALE_SKEW:
        return None, None, (f"the two directions came out {spread[0]/spread[1]:.0f}"
                            " times apart; correcting from that is not stable")
    return scale, base, None


def servo(robot, arm, approach, detector, wrist, colour, pose, where, scale,
          args):
    """Move until the block sits where the jaws hold one. Returns the error."""
    error_px = None
    last = where
    # Which way a correction has to go is worked out from the measurement, and
    # it has been wrong once already - the arm and the image do not agree on
    # what "forward" is as obviously as the algebra suggests. So the first
    # correction is treated as an experiment: if the block ends up further away
    # than it started, the direction is flipped and stays flipped.
    direction = 1.0
    previous = None
    for round_number in range(1, args.rounds + 1):
        # A detector that misses one frame is not a lost block; it is one
        # frame. Ask again before giving up.
        seen = keep_looking(detector, wrist, colour, last,
                            None if round_number == 1 else TRACK_PX)
        if seen is None:
            print("      lost track of the block")
            return None
        last = seen.pixel
        error_px = seen.pixel - where
        distance = float(np.linalg.norm(error_px))
        # The scale was measured by moving the arm and watching the block, so
        # it already reads "arm by this much -> block's pixel by that much".
        # The move wanted is therefore the shift wanted, run back through it -
        # towards the target from the block, not away from it.
        error_mm = np.linalg.solve(scale, where - seen.pixel)
        print(f"      {round_number}: off by {distance:.0f} px "
              f"({error_mm[0]:+.0f}, {error_mm[1]:+.0f} mm)")
        if distance <= args.tolerance:
            return error_px
        if previous is not None and distance > previous + 5 and direction > 0:
            direction = -1.0
            print("      that went the wrong way; correcting the other way")
        previous = distance

        step = direction * args.gain * error_mm
        if np.linalg.norm(step) > MAX_STEP_MM:
            step *= MAX_STEP_MM / np.linalg.norm(step)
        here = tip_of(arm, robot)
        if move_to(robot, arm, approach, pose,
                   here + np.array([step[0] / 1000, step[1] / 1000, 0.0])) is None:
            print("      cannot move any further that way")
            return error_px
    return error_px


def pick_one(robot, arm, approach, detector, cameras, side, wrist, table,
             home, colour, jaw_offset, where, args):
    """One block, from the side camera to the can. Returns what happened."""
    seen = [(d.colour, table.reach_target(d.pixel), d.confidence)
            for d in detector.detect(side.read().image.copy())
            if d.confidence >= MIN_CONFIDENCE]
    wanted = [entry for entry in seen
              if entry[0] == colour
              and table.outside_covered_mm(entry[1]) <= args.margin_mm]
    if not wanted:
        print(f"  no {colour} block the side camera can place "
              f"(it sees {', '.join(sorted({c for c, _, _ in seen})) or 'nothing'})")
        return "not on the table"

    _, block, confidence = max(wanted, key=lambda entry: entry[2])
    print(f"  {colour} at x={block[0]:+.3f} y={block[1]:+.3f} "
          f"(confidence {confidence:.2f})")

    grasp_at = block + jaw_offset
    pose, nearest = approach.pose_for(grasp_at, hover=args.hover)
    # The reaching poses are interpolations between demonstrated ones. Well
    # outside where any demonstration went, the interpolation is a guess: the
    # arm gets there in a posture from which both measuring directions look the
    # same in the image, and correcting from that walks away rather than in.
    if 1000 * nearest > args.max_gap_mm:
        print(f"  nothing was demonstrated near there "
              f"({1000*nearest:.0f} mm to the closest)")
        return "no demonstration nearby"
    if pose is None:
        print(f"  cannot reach above it (nearest demonstration "
              f"{1000*nearest:.0f} mm away)")
        return "unreachable"
    # Not the gripper angle the demonstrations *reached* - that is where the
    # block stopped the jaws, and commanding it closes them onto nothing,
    # because the servo simply arrives there in mid air. Squeeze past it and
    # let the block do the stopping, which is also what makes the difference
    # measurable afterwards.
    grip = args.grip
    if args.dry_run:
        print(f"  would hover, servo, and close to {grip:.0f} deg")
        return "dry run"

    print("  moving over it")
    glide_to(robot, {**pose, "gripper": args.open}, seconds=2.5)
    time.sleep(SETTLE_S)

    # Line the block up more than once, coming down between tries. The pixel the
    # jaws hold a block at was measured at grasping height, and the camera does
    # not look straight down - so an alignment made from six centimetres up does
    # not survive the descent. Each stage is nearer the height the target was
    # measured at, and the correction it needs is smaller than the last.
    above = joints_of(robot)
    # What closing on nothing looks like, measured up here where nothing can be
    # between the jaws. Down at grasping height the jaws are among the other
    # blocks, and one of those can stop them just as well as the target - which
    # makes every attempt look identical whether it caught anything or not.
    # Given as long as the real close gets, because the jaws are still creeping
    # shut after half a second.
    glide_to(robot, {**above, "gripper": grip}, seconds=1.0)
    time.sleep(SETTLE_S)
    free = joints_of(robot)["gripper"]
    glide_to(robot, {**above, "gripper": args.open}, seconds=0.8)
    time.sleep(0.4)
    print(f"  closing on nothing up here settles at {free:.1f} deg")

    for stage, height in enumerate(args.stages, 1):
        if stage > 1:
            print(f"  down to {1000*height:.0f} mm above it")
            lowered = move_to(robot, arm, approach, above,
                              grasp_at + np.array([0.0, 0.0, height]),
                              seconds=1.2)
            if lowered is None:
                print("  cannot get down to that height")
                return "cannot descend"
            above = joints_of(robot)
            hurt = strained(robot)
            if hurt:
                print(f"  {hurt} on the way down")
                return "strained"

        print(f"  measuring the scale at {1000*height:.0f} mm")
        # Nearer the table a millimetre is worth more pixels, so the same step
        # would fling the block past where it can still be recognised as the
        # same one. Shrink it with the height.
        jog = max(0.010, args.jog * height / args.hover)
        scale, _, problem = measure_scale(robot, arm, approach, detector, wrist,
                                          colour, above, where, jog)
        if problem:
            print(f"  {problem}")
            return "no scale"
        print(f"      {np.linalg.norm(scale[:, 0]):.2f} and "
              f"{np.linalg.norm(scale[:, 1]):.2f} px/mm")

        print("  closing in")
        error_px = servo(robot, arm, approach, detector, wrist, colour, above,
                         where, scale, args)
        if error_px is None:
            return "lost it"
        if np.linalg.norm(error_px) > args.tolerance:
            print(f"  stopped {np.linalg.norm(error_px):.0f} px out")
        above = joints_of(robot)

    print("  onto it")
    here = tip_of(arm, robot)
    down = move_to(robot, arm, approach, above,
                   np.array([here[0], here[1], grasp_at[2]]), seconds=1.2)
    if down is None:
        print("  cannot get down to it")
        return "cannot descend"
    landed = tip_of(arm, robot)
    print(f"      asked for z={1000*grasp_at[2]:+.0f} mm, reached "
          f"{1000*landed[2]:+.0f} mm")
    hurt = strained(robot)
    if hurt:
        print(f"  {hurt} on the way down")
        return "strained"

    # The one picture that says whether a miss was sideways or vertical.
    import cv2

    out = Path(args.out) / colour
    out.mkdir(parents=True, exist_ok=True)
    for role, stream in cameras.streams.items():
        cv2.imwrite(str(out / f"at_block_{role}.png"), stream.read().image)

    glide_to(robot, {**down, "gripper": grip}, seconds=1.0)
    time.sleep(SETTLE_S)
    held = joints_of(robot)["gripper"]
    blocked = held > free + HOLDING_DEG
    print(f"  jaws closed to {held:.1f} deg, against {free:.1f} on air"
          f"   ({'holding something' if blocked else 'nothing between them'})")

    print("  lifting")
    lifted = move_to(robot, arm, approach, down,
                     tip_of(arm, robot) + np.array([0.0, 0.0, args.hover]),
                     seconds=1.2)
    if lifted is None:
        glide_to(robot, {**home, "gripper": args.open}, seconds=2.0)
        return "cannot lift"

    if not blocked:
        glide_to(robot, {**home, "gripper": args.open}, seconds=2.0)
        return "missed"

    can = approach.over_the_can()
    print("  carrying to the can")
    glide_to(robot, {**can, "gripper": grip}, seconds=2.5)
    time.sleep(SETTLE_S)
    glide_to(robot, {**can, "gripper": args.open}, seconds=0.8)
    time.sleep(0.6)
    print("  dropped")
    glide_to(robot, {**home, "gripper": args.open}, seconds=2.0)
    return "in the can"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colour", required=True)
    parser.add_argument("--count", type=int, default=1,
                        help="how many blocks of that colour to move")
    parser.add_argument("--hover", type=float, default=HOVER_M)
    parser.add_argument("--stages", type=float, nargs="+", default=STAGES_M,
                        help="heights above the grasp to line the block up at")
    parser.add_argument("--jog", type=float, default=JOG_M)
    parser.add_argument("--rounds", type=int, default=SERVO_ROUNDS)
    parser.add_argument("--gain", type=float, default=SERVO_GAIN)
    parser.add_argument("--tolerance", type=float, default=SERVO_TOLERANCE_PX)
    parser.add_argument("--grip", type=float, default=GRIP_DEG,
                        help="how far to squeeze; a block stops the jaws first")
    parser.add_argument("--open", type=float, default=45.0)
    parser.add_argument("--margin-mm", type=float, default=30.0)
    parser.add_argument("--max-gap-mm", type=float, default=80.0,
                        help="how far from the nearest demonstration to still try")
    parser.add_argument("--side", default="side")
    parser.add_argument("--wrist", default="wrist")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--demos", type=Path, default=Path("data/demos"))
    parser.add_argument("--jaw-offset", type=Path,
                        default=Path("data/jaw_offset.json"))
    parser.add_argument("--grasp-pixel", type=Path,
                        default=Path("data/grasp_pixel.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/pick"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    tuning.install(verbose=False)
    table = TableFrame.load()
    arm = ArmKinematics()
    approach = ApproachPoses.from_demos(args.demos, kinematics=arm)
    detector = BlockDetector(weights=args.weights)
    detector.warmup()

    jaw_offset, _ = load_json(args.jaw_offset, "offset_m", None)
    if jaw_offset is None:
        raise SystemExit("  No jaw offset. Run: uv run scripts/fit_jaw_offset.py")
    where, pixel_data = load_json(args.grasp_pixel, "pixel", None)
    if where is None:
        raise SystemExit("  No grasp pixel. Run: uv run scripts/fit_grasp_pixel.py")

    print(f"  detector: {detector.weights}")
    print(f"  {approach}")
    print(f"  jaws sit {1000*jaw_offset[0]:+.0f}, {1000*jaw_offset[1]:+.0f}, "
          f"{1000*jaw_offset[2]:+.0f} mm from the gripper frame")
    print(f"  and hold a block at pixel ({where[0]:.0f}, {where[1]:.0f})")
    can = approach.over_the_can()
    if can is None:
        raise SystemExit("  The recordings show no release; nowhere to put it.")

    cameras = CameraSet.from_config()
    cameras.start()
    robot = make_robot_from_config(SO101FollowerConfig(
        port=resolve_port([args.follower_port])[0], id="follower"))
    robot.connect()
    home = joints_of(robot)

    results = []
    try:
        cameras.wait_for_frames(timeout=25)
        side = cameras.streams[args.side]
        wrist = cameras.streams[args.wrist]
        for number in range(1, args.count + 1):
            print(f"\n  --- {number}/{args.count} ---")
            results.append(pick_one(robot, arm, approach, detector, cameras,
                                    side, wrist, table, home, args.colour,
                                    jaw_offset, where, args))
            if results[-1] in ("not on the table", "unreachable"):
                break
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

    if results:
        print()
        for number, result in enumerate(results, 1):
            print(f"  {number}. {result}")
        print(f"\n  {sum(1 for r in results if r == 'in the can')} of "
              f"{len(results)} into the can")


if __name__ == "__main__":
    main()
