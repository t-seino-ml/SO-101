"""Training data: block crops, synthetic scenes, and LeRobot datasets.

- `blocks`   cut the coloured cubes out of the source photos, with alpha, so they
             can be composited onto any background
- (planned)  synthetic scene generation for the detector
- (planned)  teleoperation recording and LeRobot dataset handling
"""

from .blocks import COLOR_CLASSES, extract_blocks

__all__ = ["COLOR_CLASSES", "extract_blocks"]
