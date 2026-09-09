"""Forward and inverse kinematics for the SO-101 arm.

LeRobot's own RobotKinematics is built on placo, which publishes no Windows
wheels, so this uses ikpy instead: pure Python, and it solves this arm to well
under a millimetre.

Angles are in the degrees LeRobot's SO-101 config reports and accepts
(`use_degrees=True`), so a solution can go straight into `robot.send_action`
without another conversion step. Positions are metres in the arm's base frame:
+x forward from the base, +y left, +z up.

The URDF comes from TheRobotStudio/SO-ARM100 (Simulation/SO101). Fetch it with
scripts/fetch_urdf.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

URDF_PATH = Path("data/urdf/so101_new_calib.urdf")

# ikpy chain order, skipping the two fixed links it inserts at either end.
JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
               "wrist_roll")
ACTIVE_SLICE = slice(1, 6)


class ArmKinematics:
    """FK and IK for the five arm joints. The gripper is not part of the chain."""

    def __init__(self, urdf_path=URDF_PATH):
        urdf_path = Path(urdf_path)
        if not urdf_path.is_file():
            raise FileNotFoundError(
                f"{urdf_path} not found. Run scripts/fetch_urdf.py to download it.")
        from ikpy.chain import Chain

        with warnings.catch_warnings():
            # ikpy warns about the fixed links it adds itself; nothing to act on.
            warnings.simplefilter("ignore", UserWarning)
            self.chain = Chain.from_urdf_file(str(urdf_path),
                                              base_elements=["base_link"])
        self.joint_names = list(JOINT_ORDER)

    # -- conversions ------------------------------------------------------

    def _to_chain(self, joints_deg, clamp=True):
        """Pack a joint dict or sequence into the full ikpy vector, in radians.

        Angles are clamped into the URDF's declared limits. The servos' own
        calibrated travel is wider - the calibration measures the real mechanical
        range while the URDF states a conservative one - so a pose read straight
        off the arm can sit outside the chain's bounds, and scipy refuses to start
        from there ("Initial guess is outside of provided bounds").
        """
        if isinstance(joints_deg, dict):
            values = [joints_deg[name] for name in self.joint_names]
        else:
            values = list(joints_deg)
        full = np.zeros(len(self.chain.links))
        full[ACTIVE_SLICE] = np.deg2rad(values)
        if clamp:
            for index, link in zip(range(*ACTIVE_SLICE.indices(len(self.chain.links))),
                                   self.chain.links[ACTIVE_SLICE]):
                bounds = getattr(link, "bounds", None)
                if bounds is None or bounds[0] is None:
                    continue
                margin = 1e-6
                full[index] = float(np.clip(full[index], bounds[0] + margin,
                                            bounds[1] - margin))
        return full

    def _from_chain(self, full):
        return {name: float(np.rad2deg(value))
                for name, value in zip(self.joint_names, full[ACTIVE_SLICE])}

    # -- kinematics -------------------------------------------------------

    def forward(self, joints_deg):
        """Gripper frame position in metres for the given joint angles.

        Not clamped: this reports where the arm actually is, limits or not.
        """
        return self.chain.forward_kinematics(
            self._to_chain(joints_deg, clamp=False))[:3, 3]

    def inverse(self, position, seed_deg=None, tolerance_mm=2.0):
        """Joint angles that put the gripper frame at `position`.

        `seed_deg` should be the arm's current pose: IK is a local search, so
        seeding it from where the arm already is keeps the solution near the
        current configuration instead of jumping across the workspace.

        Returns None when the solver cannot reach the target within tolerance,
        rather than handing back a pose that quietly misses.
        """
        seed = self._to_chain(seed_deg) if seed_deg is not None else None
        solution = self.chain.inverse_kinematics(np.asarray(position, float),
                                                 initial_position=seed)
        reached = self.chain.forward_kinematics(solution)[:3, 3]
        error_mm = 1000 * float(np.linalg.norm(reached - np.asarray(position, float)))
        if error_mm > tolerance_mm:
            return None
        return self._from_chain(solution)

    def reachable(self, position, seed_deg=None, tolerance_mm=2.0):
        return self.inverse(position, seed_deg, tolerance_mm) is not None

    # -- limits -----------------------------------------------------------

    def joint_limits_deg(self):
        """{joint: (low, high)} in degrees, from the URDF."""
        limits = {}
        for name, link in zip(self.joint_names, self.chain.links[ACTIVE_SLICE]):
            bounds = getattr(link, "bounds", None)
            if bounds is None or bounds[0] is None:
                limits[name] = (-180.0, 180.0)
            else:
                limits[name] = (float(np.rad2deg(bounds[0])),
                                float(np.rad2deg(bounds[1])))
        return limits

    def within_limits(self, joints_deg):
        limits = self.joint_limits_deg()
        return all(limits[name][0] - 1e-6 <= joints_deg[name] <= limits[name][1] + 1e-6
                   for name in self.joint_names)
