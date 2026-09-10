"""Pick a block off the table and drop it in the can.

The motion is scripted rather than learned. Once the detector says where a block
is and the homography turns that into arm coordinates, reaching it is a solved
problem - there is nothing for a policy to discover, and a scripted controller is
deterministic, debuggable, and needs no demonstrations. Later it becomes the thing
that *generates* demonstrations, hundreds of them, for a policy to learn from.

Every motion is a ramp between IK solutions, with load and temperature watched at
each step. shoulder_lift carries the whole arm and will latch its overload
protection if asked to hold a straining pose, so a move that starts loading up is
abandoned rather than pushed through.

Approaches are always from directly above. Coming in sideways knocks over the
blocks that are not being picked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# Heights are tight because pointing the gripper straight down costs reach. At
# this table height the whole workspace solves at 2 cm above it, 33 of 35 points
# at 3-4 cm, and nothing at all at 10 cm - the wrist simply cannot lift a
# downward-pointing gripper that high. The grasp itself solves everywhere.
#
# 3 cm clears a 25 mm cube on the way past while staying inside what the arm can
# actually do.
APPROACH_HEIGHT = 0.035       # metres above the table to travel at
GRASP_CLEARANCE = 0.005       # how far above the reported block position to stop
LIFT_HEIGHT = 0.035
STEP_DEG = 1.5                # per interpolation step, per joint
STEP_DELAY = 0.02
SETTLE_SECONDS = 0.25
LOAD_ABORT = 700              # of 1023 full scale
LOAD_HOLD_LIMIT = 450
TEMP_ABORT = 60

# The servos run proportional position control, so they settle wherever their
# torque balances gravity and friction - short of the commanded angle. Measured on
# this arm: 15-20 mm of undershoot at the gripper, mostly along x. Commanding the
# target plus the measured shortfall removes it.
#
# Only part of the residual is applied each round. Correcting by the full amount
# overshoots and the error oscillates around +/-8 mm indefinitely; at 0.6 it
# converges monotonically - 19, 7, 3 mm - and at 0.4 it converges too slowly to
# finish in the rounds available.
SETTLE_ROUNDS = 4
SETTLE_GAIN = 0.6
SETTLE_TOLERANCE_MM = 4.0

GRIPPER_OPEN = 40.0           # degrees, as LeRobot reports the gripper joint
GRIPPER_CLOSED = 2.0
GRIPPER_EMPTY_MARGIN = 4.0    # closing past this means nothing is held


@dataclass
class PickResult:
    picked: bool
    reason: str = ""
    waypoints: list = field(default_factory=list)

    def __str__(self):
        return ("picked" if self.picked else f"failed: {self.reason}")


class PickPlace:
    """Scripted pick-and-place for one block at a time."""

    def __init__(self, robot, kinematics, drop_position, speed_deg=STEP_DEG,
                 dry_run=False, verbose=True):
        self.robot = robot
        self.arm = kinematics
        self.drop_position = np.asarray(drop_position, float)
        self.speed_deg = speed_deg
        self.dry_run = dry_run
        self.verbose = verbose

    # -- state ------------------------------------------------------------

    def joints(self):
        observation = self.robot.get_observation()
        return {key.removesuffix(".pos"): float(value)
                for key, value in observation.items() if key.endswith(".pos")}

    def arm_joints(self, joints=None):
        joints = self.joints() if joints is None else joints
        return {name: joints[name] for name in self.arm.joint_names}

    def tip(self):
        return self.arm.forward(self.arm_joints())

    def _strained(self):
        """(joint, load) if any joint is loading up, else None."""
        bus = self.robot.bus
        for name in bus.motors:
            try:
                load = bus.read("Present_Load", name, normalize=False)
                temperature = bus.read("Present_Temperature", name, normalize=False)
            except Exception:  # noqa: BLE001 - a dropped packet is not a fault
                continue
            # LeRobot's read already decodes Feetech's sign-magnitude encoding, so
            # this is signed. Masking it as an 11-bit field - correct for the raw
            # register - turns -37 into 987 and makes every motion look overloaded.
            if abs(load) > LOAD_ABORT:
                return name, load
            if temperature > TEMP_ABORT:
                return name, temperature
        return None

    # -- motion -----------------------------------------------------------

    def move_joints(self, target, gripper=None):
        """Ramp from the current pose to `target`, watching load as it goes."""
        start = self.joints()
        goal = dict(start)
        goal.update(target)
        if gripper is not None:
            goal["gripper"] = gripper

        travel = max(abs(goal[name] - start[name]) for name in goal)
        steps = max(1, int(travel / self.speed_deg))
        if self.dry_run:
            return True, f"dry run, {steps} steps"

        for step in range(1, steps + 1):
            fraction = step / steps
            action = {f"{name}.pos": start[name] + (goal[name] - start[name]) * fraction
                      for name in goal}
            self.robot.send_action(action)
            time.sleep(STEP_DELAY)
            strain = self._strained()
            if strain:
                return False, f"{strain[0]} strained ({strain[1]})"
        time.sleep(SETTLE_SECONDS)
        return True, ""

    def move_to(self, position, gripper=None, settle=True):
        """Ramp the gripper frame to a Cartesian position, then close the gap.

        Proportional control leaves the arm short of its target. Rather than raise
        the gain - which makes teleoperation shake - the residual is measured and
        commanded away: ask again for the target plus however far it fell short.
        """
        position = np.asarray(position, float)
        seed = self.arm_joints()
        solution = self.arm.inverse(position, seed_deg=seed)
        if solution is None:
            return False, (f"unreachable: x={position[0]:+.3f} "
                           f"y={position[1]:+.3f} z={position[2]:+.3f}")
        ok, detail = self.move_joints(solution, gripper=gripper)
        if not ok or not settle or self.dry_run:
            return ok, detail

        commanded = dict(solution)
        for _ in range(SETTLE_ROUNDS - 1):
            reached = self.tip()
            error = position - reached
            if 1000 * float(np.linalg.norm(error)) <= SETTLE_TOLERANCE_MM:
                break
            # Aim past the target by the amount it undershot.
            corrected = self.arm.inverse(position + SETTLE_GAIN * error,
                                         seed_deg=self.arm_joints())
            if corrected is None:
                break
            commanded = corrected
            ok, detail = self.move_joints(commanded, gripper=gripper)
            if not ok:
                return ok, detail
        return True, ""

    def set_gripper(self, degrees):
        return self.move_joints({}, gripper=degrees)

    # -- the sequence -----------------------------------------------------

    def _log(self, message):
        if self.verbose:
            print(f"    {message}", flush=True)

    def pick(self, position):
        """Pick the block at `position` (x, y, z on the table) and drop it in the can."""
        position = np.asarray(position, float)
        above = position + [0, 0, APPROACH_HEIGHT]
        grasp = position + [0, 0, GRASP_CLEARANCE]
        lift = position + [0, 0, LIFT_HEIGHT]
        over_can = self.drop_position + [0, 0, APPROACH_HEIGHT]

        stages = [
            ("open the gripper", lambda: self.set_gripper(GRIPPER_OPEN)),
            ("move above the block", lambda: self.move_to(above, GRIPPER_OPEN)),
            ("descend", lambda: self.move_to(grasp, GRIPPER_OPEN)),
            ("close on the block", lambda: self.set_gripper(GRIPPER_CLOSED)),
        ]
        waypoints = []
        for label, action in stages:
            self._log(label)
            ok, detail = action()
            waypoints.append(label)
            if not ok:
                self.recover()
                return PickResult(False, f"{label}: {detail}", waypoints)

        if not self.dry_run and not self.holding():
            self._log("nothing in the gripper")
            self.recover()
            return PickResult(False, "grasp missed", waypoints)

        stages = [
            ("lift", lambda: self.move_to(lift, GRIPPER_CLOSED)),
            ("move over the can", lambda: self.move_to(over_can, GRIPPER_CLOSED)),
            ("release", lambda: self.set_gripper(GRIPPER_OPEN)),
        ]
        for label, action in stages:
            self._log(label)
            ok, detail = action()
            waypoints.append(label)
            if not ok:
                self.recover()
                return PickResult(False, f"{label}: {detail}", waypoints)

        return PickResult(True, "", waypoints)

    def holding(self):
        """Did the gripper stop on something, or close all the way to empty?"""
        return self.joints()["gripper"] > GRIPPER_CLOSED + GRIPPER_EMPTY_MARGIN

    def recover(self):
        """Back off to a safe height so a failure does not leave the arm loaded."""
        if self.dry_run:
            return
        tip = self.tip()
        self.move_to([tip[0], tip[1], max(tip[2], APPROACH_HEIGHT)])
