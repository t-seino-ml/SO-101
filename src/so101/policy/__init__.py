"""Where a block is, from the side camera.

- `detector`  the trained block detector, with this rig's inference settings

That is all this package does now. It used to carry the rest of a pick as well -
inverse kinematics, a pixels-to-table homography, a TCP model, visual servoing -
and the exhibition does none of it: a colour picks the nearest taught slot in
pixel space, and the arm replays joint angles a person taught it. The detector
answers "which colour is where on screen", and nothing downstream needs metres.

The removed modules are in the history if that approach is picked up again.
"""

from .detector import BlockDetector, Detection, find_weights

__all__ = ["BlockDetector", "Detection", "find_weights"]
