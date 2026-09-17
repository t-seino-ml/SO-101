"""A MuJoCo twin of the rig, for checking geometry without the arm.

The point is not to reproduce the physics. It is to take the questions that cost
thirty seconds each on the real arm - is this pose reachable, does the tool point
the right way, where does the grasp point actually end up, does the arm hit the
table - and answer thousands of them in a few seconds.

Kinematics are deliberately NOT reimplemented here. `so101.policy.kinematics`
solves the URDF with ikpy and is what the real arm uses; the MJCF is generated
from the same Onshape CAD, so MuJoCo is used to check that solver rather than to
replace it, and afterwards for the things ikpy has no opinion about: collision,
contact, and rendering a camera view.

- `model`   load the arm, build a table/block/can scene, read frames back
"""

from .model import SO101Sim, TABLE_THICKNESS_M, BLOCK_SIZE_M, CAN_DIAMETER_M, CAN_HEIGHT_M

__all__ = ["SO101Sim", "TABLE_THICKNESS_M", "BLOCK_SIZE_M",
           "CAN_DIAMETER_M", "CAN_HEIGHT_M"]
