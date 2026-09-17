"""Kinematics that end at the jaws, not at the frame the URDF happens to name.

Everything so far has solved for `gripper_frame_link` and then added an offset to
the answer. That is where the project's worst bug lived. The frame sits within
8 mm of the wrist_roll axis, so rolling the wrist barely moves it - inverse
kinematics can pick any roll it likes and still report that it reached the
target - while the jaws, which hang 20 mm off that axis, swing through 58 mm over
the roll's travel. Measured on the real arm: two poses whose gripper frames were
1 mm apart, wrists 62 degrees apart, and the grasp missed.

Adding the offset after the solve cannot fix that, because by then the posture is
already chosen. So the offset becomes part of the chain instead: one more fixed
link on the end, and ikpy solves for the point that actually touches the block.

Two constraints matter beyond position, and they are separate things:

- the tool axis, which is where the gripper points. `orientation_mode="Z"` pins
  it, and on its own it pins nothing about roll: rolling turns the tool about
  its own z, so the axis is invariant to it. Measured across the full roll
  travel, the axis moves 0.000 degrees.
- the roll itself, which has to be frozen rather than constrained. With it
  frozen the arm has four joints for three position constraints plus the axis,
  which is a well-posed problem instead of a two-parameter family of answers.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from .kinematics import ACTIVE_SLICE, ArmKinematics

TCP_PATH = Path("data/tcp.json")
#: Straight down in the base frame. Kept for callers that want it, but see the
#: note in ApproachPoses: with a vertical tool axis this arm has no solution
#: above about 90 mm anywhere on the table, and none at all past 300 mm out.
DOWN = np.array([0.0, 0.0, -1.0])


def load_offset(path=TCP_PATH):
    """The TCP's position in the gripper frame, in metres."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/sim_tcp.py --save")
    return np.array(json.loads(path.read_text(encoding="utf-8"))
                    ["tcp_in_gripper_m"], float)


class ToolKinematics(ArmKinematics):
    """Forward and inverse kinematics of the point the jaws close on."""

    def __init__(self, urdf_path=None, tcp=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if urdf_path is None:
                super().__init__()
            else:
                super().__init__(urdf_path)
        self.tcp = np.asarray(load_offset() if tcp is None else tcp, float)
        self.gripper_chain = self.chain
        self.chain = self._with_tool(self.chain, self.tcp)

    @staticmethod
    def _with_tool(chain, tcp):
        """The same chain with one fixed link on the end, at the TCP."""
        from ikpy.chain import Chain
        from ikpy.link import URDFLink

        tool = URDFLink(name="tcp", origin_translation=np.asarray(tcp, float),
                        origin_orientation=np.zeros(3), joint_type="fixed")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return Chain(list(chain.links) + [tool],
                         active_links_mask=list(chain.active_links_mask) + [False])

    # -- kinematics -------------------------------------------------------

    def forward(self, joints_deg):
        """Where the jaws close, in metres in the base frame."""
        return self.pose(joints_deg)[0]

    def pose(self, joints_deg):
        """(position, 3x3 rotation) of the TCP."""
        matrix = self.chain.forward_kinematics(
            self._to_chain(joints_deg, clamp=False))
        return matrix[:3, 3], matrix[:3, :3]

    def gripper_frame(self, joints_deg):
        """Where `gripper_frame_link` is - what the old code solved for.

        Drops the last element: the tool link is fixed, so its joint value is
        always zero, and the shorter chain will not accept a vector sized for
        the longer one.
        """
        return self.gripper_chain.forward_kinematics(
            self._to_chain(joints_deg, clamp=False)[:-1])[:3, 3]

    def inverse(self, position, seed_deg=None, tolerance_mm=2.0, orientation=None,
                frozen=(), orientation_mode="Z"):
        """Joint angles putting the TCP at `position`.

        `frozen` is the important argument, not `orientation`. Freezing
        wrist_roll is what makes the answer unique enough to be useful; pinning
        the tool axis on its own leaves the roll free, and the roll is what moves
        the jaws.

        Returns None rather than a pose that quietly misses.
        """
        seed = self._to_chain(seed_deg) if seed_deg is not None else None
        kwargs = {} if orientation is None else {
            "target_orientation": np.asarray(orientation, float),
            "orientation_mode": orientation_mode,
        }
        previous = self.chain.active_links_mask
        if frozen:
            if seed is None:
                raise ValueError("freezing a joint needs a seed to freeze it at")
            self.chain.active_links_mask = self.frozen_mask(frozen)
        try:
            solution = self.chain.inverse_kinematics(
                np.asarray(position, float), initial_position=seed, **kwargs)
        finally:
            self.chain.active_links_mask = previous

        reached = self.chain.forward_kinematics(solution)[:3, 3]
        if 1000 * float(np.linalg.norm(reached - np.asarray(position, float))) \
                > tolerance_mm:
            return None
        return self._from_chain(solution)

    def set_limits(self, limits_deg):
        """Replace the URDF's declared joint ranges with measured ones.

        Scoring a solution against limits the solver does not know about
        measures nothing: ikpy clamps to the bounds in its own chain, so answers
        land exactly on the URDF's stops however generous the scoring is. The
        servos' calibrated travel differs in both directions - wrist_flex has
        212 degrees against the URDF's 190, wrist_roll 266 against 320 - so this
        has to reach the chain itself.
        """
        first = ACTIVE_SLICE.indices(len(self.gripper_chain.links))[0]
        for chain in (self.chain, self.gripper_chain):
            for offset, name in enumerate(self.joint_names):
                if name not in limits_deg:
                    continue
                low, high = limits_deg[name]
                chain.links[first + offset].bounds = (np.radians(low),
                                                      np.radians(high))
        return self

    def frozen_mask(self, frozen):
        """The chain's active mask with these joints held still.

        Overridden because the chain is one link longer than the parent's.
        """
        mask = list(self.chain.active_links_mask)
        first = ACTIVE_SLICE.indices(len(self.gripper_chain.links))[0]
        for name in frozen:
            mask[first + self.joint_names.index(name)] = False
        return mask

    def _to_chain(self, joints_deg, clamp=True):
        """Pack a joint dict or sequence into this chain's vector, in radians.

        Written out rather than delegating: the parent sizes its vector from
        `self.chain`, which is now one link longer, so calling it and appending
        gives a vector one too long.

        Angles are clamped into the URDF's limits. The servos' calibrated travel
        is wider than the URDF declares, so a pose read straight off the arm can
        sit outside the chain's bounds and scipy refuses to start from there.
        """
        if isinstance(joints_deg, dict):
            values = [joints_deg[name] for name in self.joint_names]
        else:
            values = list(joints_deg)
        full = np.zeros(len(self.chain.links))
        full[ACTIVE_SLICE] = np.deg2rad(values)
        if clamp:
            for index in range(*ACTIVE_SLICE.indices(len(self.chain.links))):
                bounds = getattr(self.chain.links[index], "bounds", None)
                if bounds is None or bounds[0] is None:
                    continue
                full[index] = float(np.clip(full[index], bounds[0] + 1e-6,
                                            bounds[1] - 1e-6))
        return full

    def _from_chain(self, full):
        return {name: float(np.rad2deg(value))
                for name, value in zip(self.joint_names, full[ACTIVE_SLICE])}

    def tool_axis(self, joints_deg):
        return self.pose(joints_deg)[1][:, 2]

    def __str__(self):
        return (f"ToolKinematics(tcp {1000*self.tcp[0]:+.1f}, "
                f"{1000*self.tcp[1]:+.1f}, {1000*self.tcp[2]:+.1f} mm "
                f"from the gripper frame)")
