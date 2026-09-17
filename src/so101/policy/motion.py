"""The moves the task is built out of, and the guards that keep them safe.

These were written inside scripts/pick_colour.py, where they work, and they are
lifted here unchanged in behaviour so that hovering, picking and dropping can be
built from the same pieces instead of three copies drifting apart. The comments
explaining *why* each one looks the way it does are kept with the code, because
every one of them records something that went wrong on this rig.

What is new here is `Workspace`: a check that a target is somewhere the arm
should be asked to go at all, applied before anything moves. Inverse kinematics
will happily solve for a point inside the table, and the load monitor only
notices afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

LOAD_ABORT = 700          # of 1023 full scale, as pick_colour uses
SETTLE_ROUNDS = 6
SETTLE_GAIN = 0.8
SETTLE_TOLERANCE_M = 0.002
SETTLE_S = 1.0
# Held still while lining a target up. Left free, a twenty-millimetre step
# sideways can swing the wrist by sixty degrees - the solver is entitled to
# answer two nearby targets with two quite different postures - and the wrist
# camera swings with it.
FREEZE = ("wrist_roll", "wrist_flex")


@dataclass
class Workspace:
    """Where the arm may be sent. Checked before a move, not after."""

    max_reach_m: float = 0.42
    min_reach_m: float = 0.10
    floor_m: float = -0.015        # nothing below this; the table is at z_table
    ceiling_m: float = 0.30

    def rejects(self, position):
        """Why this target is not allowed, or None if it is."""
        position = np.asarray(position, float)
        reach = float(np.linalg.norm(position[:2]))
        if reach > self.max_reach_m:
            return f"{1000*reach:.0f} mm out, past the {1000*self.max_reach_m:.0f} mm limit"
        if reach < self.min_reach_m:
            return f"{1000*reach:.0f} mm out, inside the {1000*self.min_reach_m:.0f} mm limit"
        if position[2] < self.floor_m:
            return f"z {1000*position[2]:+.0f} mm is below the {1000*self.floor_m:+.0f} mm floor"
        if position[2] > self.ceiling_m:
            return f"z {1000*position[2]:+.0f} mm is above the {1000*self.ceiling_m:+.0f} mm ceiling"
        return None


def gripper_rotation(arm, joints_deg):
    """The gripper frame's orientation, as a 3x3 matrix in the base frame.

    Needed because the jaws hang off the gripper frame rather than sitting on
    it. That offset is a constant in the gripper's own frame; in the base frame,
    where targets are expressed, it turns with the wrist. Measured on this rig:
    two poses that put the gripper frame within 1 mm of the same place had their
    wrists 62 degrees apart, which swings a 20 mm offset by about 20 mm - and a
    grasp tolerates 5.6 mm.

    Five joints and three position constraints leave two degrees of freedom, so
    inverse kinematics is entitled to answer the same point with quite different
    postures, and it does.
    """
    full = arm._to_chain(joints_deg, clamp=False)
    return arm.chain.forward_kinematics(full)[:3, :3]


def posture_for(approach, position, hover=0.0, roll=None):
    """A pose reaching `position`, with the wrist roll pinned if one is given.

    Without the pin the solver is free to answer nearby targets with quite
    different postures - 62 degrees of wrist roll between two poses whose gripper
    frames were 1 mm apart, measured on this rig - and since the jaws hang off
    that frame, the same commanded position then puts them 20 mm apart. Pinning
    the roll to what it was when the offset was measured makes the posture
    repeatable, which is what makes the offset mean anything.

    Returns (pose, nearest demonstration distance).
    """
    pose, nearest = approach.pose_for(position, hover=hover)
    if pose is None or roll is None:
        return pose, nearest
    seeded = dict(pose)
    seeded["wrist_roll"] = roll
    target = np.asarray(position, float) + np.array([0.0, 0.0, hover])
    refined = approach.refine(seeded, target, tolerance_mm=5.0,
                              frozen=("wrist_roll",))
    return (refined if refined is not None else pose), nearest


def jaw_target(arm, approach, block, offset_gripper, hover=0.0, roll=None,
               rounds=3, tolerance_mm=3.0):
    """Where to send the gripper frame so the jaws close on `block`.

    Circular at first sight - where the jaws sit depends on the posture, and the
    posture depends on where the arm is sent - so it is iterated. It converges
    only if the posture varies smoothly with the target, which is why `roll`
    matters: with the wrist free, one iteration can flip the offset's direction
    and the loop settles somewhere confidently wrong. It did exactly that once,
    aiming 18 mm the wrong way.

    Returns (target, pose, nearest demonstration distance).
    """
    block = np.asarray(block, float)
    target = block.copy()
    pose = nearest = None
    for _ in range(rounds):
        pose, nearest = posture_for(approach, target, hover=hover, roll=roll)
        if pose is None:
            return None, None, nearest
        moved = block + gripper_rotation(arm, pose) @ offset_gripper
        settled = np.linalg.norm(moved - target) < tolerance_mm / 1000
        target = moved
        if settled:
            break
    pose, nearest = posture_for(approach, target, hover=hover, roll=roll)
    return target, pose, nearest


def joints_of(robot):
    """The arm's pose, as {joint: degrees}."""
    return {key.removesuffix(".pos"): float(value)
            for key, value in robot.get_observation().items()
            if key.endswith(".pos")}


def tip_of(arm, robot):
    """Where the gripper frame actually is, from the arm's own encoders."""
    pose = joints_of(robot)
    return arm.forward({name: pose[name] for name in arm.joint_names})


def strained(robot):
    """The name and load of any joint pulling too hard, or None."""
    for name in robot.bus.motors:
        try:
            load = robot.bus.read("Present_Load", name, normalize=False)
        except Exception:  # noqa: BLE001 - a dropped packet is not a fault
            continue
        if abs(load) > LOAD_ABORT:
            return f"{name} load {load}"
    return None


def glide_to(robot, goal, seconds=2.0, fps=30):
    """Interpolate to `goal` over `seconds` rather than commanding it outright."""
    start = joints_of(robot)
    steps = max(1, int(seconds * fps))
    for step in range(1, steps + 1):
        robot.send_action({f"{name}.pos": start[name]
                           + (goal[name] - start[name]) * step / steps
                           for name in goal})
        time.sleep(1.0 / fps)


def move_to(robot, arm, approach, pose, position, seconds=1.2,
            tolerance_mm=1.0, frozen=FREEZE, workspace=None, rounds=SETTLE_ROUNDS):
    """Shift the arm to `position`, keeping the posture it is already in.

    Commanding the joints inverse kinematics asks for does not put the gripper
    where it was asked to be: the follower settles short of its target under its
    own weight, by most of a centimetre when it moves sideways. So aim, look at
    where it actually went, and add the miss back into the aim - a couple of
    rounds of that and the gap is millimetres.

    The tolerance matters more than it looks. Inverse kinematics stops as soon as
    it is within it, so a loose one lets a 12 mm step land 5 mm short and in a
    different direction - fine for reaching, useless for measuring.

    Returns the pose reached, or None if the solver could not get there. Raises
    if the target is outside the workspace: that is a caller's mistake, not a
    condition to be handled quietly.
    """
    if workspace is not None:
        refused = workspace.rejects(position)
        if refused is not None:
            raise ValueError(f"target {np.asarray(position)} refused: {refused}")

    goal = np.asarray(position, float)
    aim = goal.copy()
    moved = None
    for round_number in range(rounds):
        candidate = approach.refine(pose, aim, tolerance_mm=tolerance_mm,
                                    frozen=frozen)
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


def hover_over(robot, arm, approach, position, height, gripper=None,
               seconds=2.5, workspace=None, max_gap_mm=80.0, settle=True,
               pose=None, nearest=None):
    """Put the gripper `height` metres above `position`, and stop there.

    Nothing descends. This is the move that has to be right before anything is
    allowed to go down, so it is worth having on its own: if the arm arrives over
    the wrong spot, that is a calibration problem and no amount of care on the
    way down will fix it.

    `settle` runs the aim-look-correct loop afterwards. Without it the follower
    sags under its own weight and arrives low: measured on the first hover here,
    19 mm below an 80 mm target. That is harmless at 80 mm and would not be at
    40 mm, where the can's 35 mm rim is waiting.

    Returns (pose, problem). `problem` is None when it arrived.
    """
    target = np.asarray(position, float) + np.array([0.0, 0.0, height])
    if workspace is not None:
        refused = workspace.rejects(target)
        if refused is not None:
            return None, f"target refused: {refused}"

    # A caller that worked out the target from the posture must hand that same
    # posture in: solving afresh here can land on a different one, and then the
    # offset that produced the target no longer describes where the jaws are.
    if pose is None:
        pose, nearest = approach.pose_for(target)
    # The reaching poses are interpolations between demonstrated ones. Well
    # outside where any demonstration went, the interpolation is a guess.
    if nearest is not None and 1000 * nearest > max_gap_mm:
        return None, (f"nothing was demonstrated near there "
                      f"({1000*nearest:.0f} mm to the closest)")
    if pose is None:
        return None, (f"cannot reach above it (nearest demonstration "
                      f"{1000*nearest:.0f} mm away)")

    command = dict(pose)
    if gripper is not None:
        command["gripper"] = gripper
    glide_to(robot, command, seconds=seconds)
    time.sleep(SETTLE_S)
    hurt = strained(robot)
    if hurt:
        return joints_of(robot), f"{hurt} on the way"

    if settle:
        # Correct the sag, from the pose it is now in rather than the one that
        # was commanded - those differ by the sag, which is the whole point.
        here = joints_of(robot)
        if move_to(robot, arm, approach, here, target, seconds=0.8,
                   workspace=workspace) is None:
            return here, "arrived, but could not correct the sag"
        hurt = strained(robot)
        if hurt:
            return joints_of(robot), f"{hurt} while settling"
    return joints_of(robot), None


GRIP_DEG = 8.0            # squeezed past where a block stops the jaws
OPEN_DEG = 45.0
HOLDING_DEG = 4.0         # jaws this far wider than their free close are holding


def set_gripper(robot, degrees, seconds=1.0):
    """Move the jaws alone, leaving the arm where it is."""
    glide_to(robot, {"gripper": degrees}, seconds=seconds)
    time.sleep(0.3)
    return joints_of(robot)["gripper"]


def free_close(robot, grip=GRIP_DEG, open_deg=OPEN_DEG):
    """What closing on nothing settles at, measured where it is now.

    Worth doing up in the air before every attempt rather than storing a number.
    Down at grasping height the jaws are among the other blocks, and one of those
    can stop them just as well as the target - which makes every attempt look
    identical whether it caught anything or not. Given as long as the real close
    gets, because the jaws are still creeping shut after half a second.
    """
    settled = set_gripper(robot, grip, seconds=1.0)
    time.sleep(SETTLE_S)
    settled = joints_of(robot)["gripper"]
    set_gripper(robot, open_deg, seconds=0.8)
    return settled


def holding(robot, free, margin=HOLDING_DEG):
    """Did the jaws stop on something, or close on air?"""
    return joints_of(robot)["gripper"] > free + margin


def descend_to(robot, arm, approach, pose, height, seconds=1.2, workspace=None):
    """Straight down from where the arm is, to `height`, watching the load.

    Only the height changes: x and y are taken from where the gripper actually
    is, not from where it was aimed, so a descent cannot quietly also correct a
    sideways error that the alignment decided to leave alone.

    Returns (pose, problem).
    """
    here = tip_of(arm, robot)
    target = np.array([here[0], here[1], height])
    if workspace is not None:
        refused = workspace.rejects(target)
        if refused is not None:
            return None, f"refused: {refused}"

    moved = move_to(robot, arm, approach, pose, target, seconds=seconds,
                    workspace=workspace)
    if moved is None:
        return None, "cannot get down to that height"
    hurt = strained(robot)
    if hurt:
        return moved, f"{hurt} on the way down"
    return moved, None


def lift_by(robot, arm, approach, pose, rise, seconds=1.2, workspace=None):
    """Straight up by `rise` metres from where the arm is."""
    here = tip_of(arm, robot)
    target = here + np.array([0.0, 0.0, rise])
    if workspace is not None:
        refused = workspace.rejects(target)
        if refused is not None:
            return None, f"refused: {refused}"
    moved = move_to(robot, arm, approach, pose, target, seconds=seconds,
                    workspace=workspace)
    if moved is None:
        return None, "cannot lift from there"
    return moved, None


def relax_to(robot, home, gripper=None, seconds=2.0):
    """Return to where it started and let go. Never lets an exception through."""
    try:
        goal = dict(home)
        if gripper is not None:
            goal["gripper"] = gripper
        glide_to(robot, goal, seconds=seconds)
        time.sleep(0.3)
        glide_to(robot, home, seconds=0.8)
        return None
    except Exception as error:  # noqa: BLE001 - relaxing must not itself fail
        return str(error)
