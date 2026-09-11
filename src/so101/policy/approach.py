"""Arm poses for reaching a point on the table, taken from the demonstrations.

Asking inverse kinematics for a position with the gripper pointing straight down
looks like the obvious way to reach a block, and it is wrong for this arm. With
five joints, position and orientation cannot both be chosen freely: demanding a
vertical tool axis leaves no solution above about 90 mm anywhere on the table,
and none at all past roughly 300 mm out. Measured against the recordings, the
operator never worked that way either - the median grasp tilts 23 degrees off
vertical, only five of thirty came within 15 degrees, and they reach to 394 mm.

So the poses come from the recordings instead. Each demonstration contributes
the pose it was in at the moment the jaws closed, which is by construction
reachable, clear of the table, and pointing the wrist camera at the block. For a
new target the nearest of those poses are blended by distance and handed to
inverse kinematics as a seed, with the orientation left free - so the solver
refines a posture the arm is known to be able to hold rather than searching for
one from scratch.
"""

import json
from pathlib import Path

import numpy as np

from .kinematics import ArmKinematics

DEFAULT_ROOT = Path("data/demos")
NEIGHBOURS = 3
EPSILON_M = 0.005        # keeps a target sitting exactly on a sample finite


class ApproachPoses:
    """Where to put the arm to reach a given spot on the table."""

    def __init__(self, positions, poses, joint_names, kinematics=None):
        self.positions = np.asarray(positions, float)   # (n, 3) in the arm frame
        self.poses = list(poses)                        # [{joint: degrees}]
        self.joint_names = list(joint_names)
        self.arm = kinematics or ArmKinematics()

    def __len__(self):
        return len(self.poses)

    def __str__(self):
        low, high = self.positions.min(axis=0), self.positions.max(axis=0)
        return (f"ApproachPoses({len(self)} from demonstrations, "
                f"x {1000*low[0]:.0f}..{1000*high[0]:.0f} "
                f"y {1000*low[1]:.0f}..{1000*high[1]:.0f} mm)")

    @classmethod
    def from_demos(cls, root=DEFAULT_ROOT, kinematics=None):
        """One pose per episode: where the arm was when the jaws closed.

        The gripper reaches its narrowest once per episode, on the block. That
        frame is the one worth keeping - it is the pose the whole approach was
        aiming at.
        """
        import pandas as pd

        root = Path(root)
        info = json.loads(Path(root, "meta", "info.json").read_text(encoding="utf-8"))
        names = [name.removesuffix(".pos")
                 for name in info["features"]["action"]["names"]]
        gripper = names.index("gripper")

        frames = pd.concat([pd.read_parquet(path)
                            for path in sorted(Path(root, "data").rglob("*.parquet"))])
        frames = frames.sort_values(["episode_index", "frame_index"])

        arm = kinematics or ArmKinematics()
        positions, poses = [], []
        for _, rows in frames.groupby("episode_index"):
            states = np.stack(rows["observation.state"].to_numpy())
            closed = states[int(np.argmin(states[:, gripper]))]
            pose = {name: float(closed[index]) for index, name in enumerate(names)}
            positions.append(arm.forward({name: pose[name]
                                          for name in arm.joint_names}))
            poses.append(pose)
        return cls(positions, poses, names, arm)

    def blend(self, position, neighbours=NEIGHBOURS):
        """The nearby demonstrated poses, weighted by how near they are.

        Distance is measured in the table plane: two poses over the same spot at
        different heights approach it the same way, and height is what the
        solver adjusts afterwards.
        """
        position = np.asarray(position, float)
        distances = np.linalg.norm(self.positions[:, :2] - position[:2], axis=1)
        nearest = np.argsort(distances)[:max(1, neighbours)]
        weights = 1.0 / (distances[nearest] + EPSILON_M)
        weights /= weights.sum()
        return ({name: float(sum(w * self.poses[i][name]
                                 for w, i in zip(weights, nearest)))
                 for name in self.joint_names},
                float(distances[nearest[0]]))

    def refine(self, pose, position, tolerance_mm=5.0):
        """Nudge an existing pose onto a nearby position, keeping its posture.

        Looking a pose up afresh for a point a couple of centimetres away can
        land on a different set of neighbours and so a different posture, which
        swings the wrist camera even though the gripper barely moved. Seeding
        from the pose already held keeps the motion local.
        """
        solution = self.arm.inverse(
            np.asarray(position, float),
            seed_deg={name: pose[name] for name in self.arm.joint_names},
            tolerance_mm=tolerance_mm,
            orientation=None,
        )
        if solution is None:
            return None
        refined = dict(pose)
        refined.update(solution)
        return refined

    def pose_for(self, position, hover=0.0, neighbours=NEIGHBOURS,
                 tolerance_mm=5.0):
        """Joint angles that put the gripper at `position`, `hover` metres up.

        Returns (pose, how far the nearest demonstration was) or (None, distance)
        when even a demonstrated posture cannot be refined onto the target.
        """
        position = np.asarray(position, float) + np.array([0.0, 0.0, hover])
        seed, nearest = self.blend(position, neighbours)
        solution = self.arm.inverse(
            position,
            seed_deg={name: seed[name] for name in self.arm.joint_names},
            tolerance_mm=tolerance_mm,
            orientation=None,          # five joints cannot also choose this
        )
        if solution is None:
            return None, nearest
        pose = dict(seed)
        pose.update(solution)
        return pose, nearest
