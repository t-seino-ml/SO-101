"""Perception-to-motion for the block task.

- `kinematics`   forward and inverse kinematics via ikpy and the SO-101 URDF
- `table_frame`  homography from camera pixels to the arm's own coordinates
- `auto_calibrate`  fits that homography by watching the arm carry a block
- `detector`     the trained block detector, with this rig's inference settings
- `pick_place`   scripted pick-and-place, which will also generate the episodes
                 a policy is later trained on

The split is deliberate: the detector says where a block is, the homography turns
that into a reachable position, and kinematics gets the arm there. Nothing has to
learn perception and control jointly, which is what makes a small dataset enough.
"""

from .detector import BlockDetector, find_weights, Detection
from .pick_place import PickPlace, PickResult
from .approach import ApproachPoses
from .kinematics import ArmKinematics
from .table_frame import TableFrame

__all__ = ["ArmKinematics", "ApproachPoses",
    "BlockDetector",
    "find_weights", "Detection", "PickPlace",
           "PickResult", "TableFrame"]
