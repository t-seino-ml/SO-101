"""Move the follower arm through a spread of poses.

Used while capturing background frames: the arm is in every real scene, so the
detector needs to see it in many configurations rather than one. Sweeping it
automatically covers the range more evenly than moving it by hand, and it keeps
the operator's hands out of shot.

Poses are sampled inside each joint's calibrated range with a margin, and the arm
ramps between them, so nothing is commanded into a hard stop.

Holding matters as much as moving. shoulder_lift carries the whole arm, and a
pose that is fine to pass through can trip the servo's overload protection if it
is held there - the load builds over the protection time, not instantly. So the
load is watched while holding too, and the arm retreats to a low-load pose rather
than sitting in a straining one.
"""

from __future__ import annotations

import time

import numpy as np

from ..hardware.sts3215 import (
    Bus,
    JOINT_NAMES,
    MAX_ANGLE_LIMIT,
    MIN_ANGLE_LIMIT,
    PRESENT_POSITION,
    PRESENT_TEMPERATURE,
    TICKS_PER_DEG,
    TORQUE_ENABLE,
)

LIMIT_MARGIN = 0.15       # stay this far inside each end of the calibrated range
STEP_DEG = 2.0
STEP_DELAY = 0.02
LOAD_ABORT = 800          # of 1023 full scale
LOAD_HOLD_LIMIT = 450     # sustained load that must not be held
HOLD_SECONDS = 0.6        # linger briefly so the cameras catch a settled pose
TEMP_ABORT = 60
SPEED = 400
STATUS_OVERLOAD = 0x20


class PoseSweeper:
    """Drives the follower between random reachable poses."""

    def __init__(self, port, seed=0, joints=None):
        self.bus = Bus(port)
        self.rng = np.random.default_rng(seed)
        self.joints = list(joints or JOINT_NAMES)
        self.limits = {}
        for sid in self.joints:
            low = self.bus.read(sid, MIN_ANGLE_LIMIT, 2)
            high = self.bus.read(sid, MAX_ANGLE_LIMIT, 2)
            span = high - low
            self.limits[sid] = (low + LIMIT_MARGIN * span, high - LIMIT_MARGIN * span)
        self.rest_pose = {sid: self.bus.read(sid, PRESENT_POSITION, 2)
                          for sid in self.joints}
        self.aborted = None

    def __enter__(self):
        self.clear_overload()
        for sid in self.joints:
            self.bus.torque(sid, True)
        time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.release()
        self.bus.close()

    # -- protection -------------------------------------------------------

    def clear_overload(self):
        """Clear any latched overload before driving.

        A Feetech servo that has tripped keeps answering with the overload bit set
        and freezes its load register at the trip value, and LeRobot reads such a
        reply as "motor missing". Toggling torque clears it once the joint is no
        longer straining.
        """
        cleared = []
        for sid in self.joints:
            self.bus.write(sid, 42, self.bus.read(sid, PRESENT_POSITION, 2), size=2)
            self.bus.torque(sid, False)
            time.sleep(0.05)
            self.bus.torque(sid, True)
            time.sleep(0.05)
            self.bus.torque(sid, False)
            cleared.append(sid)
        return cleared

    def release(self):
        for sid in self.joints:
            for _ in range(5):
                self.bus.torque(sid, False)
                if self.bus.read(sid, TORQUE_ENABLE, 1) == 0:
                    break

    def _strain(self):
        """(joint, load) for the most heavily loaded joint, or None if all fine."""
        worst = None
        for sid in self.joints:
            load = self.bus.read_load(sid)
            if load is None:
                continue
            if worst is None or abs(load) > abs(worst[1]):
                worst = (sid, load)
            temperature = self.bus.read(sid, PRESENT_TEMPERATURE, 1)
            if temperature is not None and temperature > TEMP_ABORT:
                self.aborted = f"{JOINT_NAMES[sid]} reached {temperature}C"
                return worst
        return worst

    # -- motion -----------------------------------------------------------

    def random_pose(self):
        return {sid: float(self.rng.uniform(*self.limits[sid])) for sid in self.joints}

    def move_to(self, pose, on_step=None):
        """Ramp every joint to `pose` together, aborting on load or heat."""
        start = {sid: self.bus.read(sid, PRESENT_POSITION, 2) for sid in self.joints}
        if any(value is None for value in start.values()):
            return False
        travel = max(abs(pose[sid] - start[sid]) for sid in self.joints)
        steps = max(1, int(travel / (STEP_DEG * TICKS_PER_DEG)))

        for step in range(1, steps + 1):
            fraction = step / steps
            for sid in self.joints:
                self.bus.move_to(sid, start[sid] + (pose[sid] - start[sid]) * fraction,
                                 speed=SPEED)
            time.sleep(STEP_DELAY)
            if on_step is not None:
                on_step()
            worst = self._strain()
            if self.aborted:
                return False
            if worst and abs(worst[1]) > LOAD_ABORT:
                self.aborted = (f"{JOINT_NAMES[worst[0]]} load {worst[1]} "
                                f"exceeded {LOAD_ABORT}")
                return False
        return True

    def hold(self, seconds=HOLD_SECONDS, on_step=None):
        """Sit still briefly, but leave early if the pose is straining a joint."""
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            time.sleep(STEP_DELAY)
            if on_step is not None:
                on_step()
            worst = self._strain()
            if self.aborted:
                return False
            if worst and abs(worst[1]) > LOAD_HOLD_LIMIT:
                # Do not wait for the overload timer to expire; go somewhere easier.
                return False
        return True

    def sweep(self, duration, on_step=None):
        """Keep moving to fresh poses until `duration` elapses. Returns pose count."""
        deadline = time.perf_counter() + duration
        poses = 0
        while time.perf_counter() < deadline:
            if not self.move_to(self.random_pose(), on_step):
                break
            poses += 1
            if not self.hold(on_step=on_step):
                # Straining here: retreat to where the arm started, which was a
                # pose it was resting in happily.
                self.move_to(self.rest_pose, on_step)
                if self.aborted:
                    break
        if self.aborted:
            print(f"  pose sweep stopped: {self.aborted}")
        return poses
