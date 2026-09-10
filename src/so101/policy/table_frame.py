"""Map overhead camera pixels to positions in the arm's own frame.

The blocks all sit on one flat surface, so a plane-to-plane homography is enough:
no camera intrinsics, no depth, no pose estimation. Eight numbers relate where a
block appears in the image to where the arm has to reach for it.

What gets fitted is pixels to the *robot's* frame, not to some abstract table
frame. That is the mapping actually needed, and it means the calibration captures
the camera's placement relative to the arm in one step rather than composing two
transforms and their two error budgets.

Correspondences come from touching blocks: the detector gives the pixel position
of a block on the table, and forward kinematics gives where the gripper is when
it touches that same block. See scripts/calibrate_table.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CALIBRATION_PATH = Path("data/table_frame.json")
MIN_CORRESPONDENCES = 4      # a homography has 8 degrees of freedom
RANSAC_THRESHOLD_PX = 8.0


class TableFrame:
    """A homography between overhead image pixels and the arm's x/y plane."""

    def __init__(self, matrix, z_table=None, residuals_mm=None, camera=None,
                 covered=None):
        self.matrix = np.asarray(matrix, float).reshape(3, 3)
        self.z_table = z_table
        self.residuals_mm = residuals_mm
        self.camera = camera
        # (x_min, x_max, y_min, y_max) of the points actually visited. A
        # homography degrades quickly outside the region it was fitted over -
        # measured here as a block 17 mm beyond the covered y range landing
        # visibly off while one inside it lined up - so callers need to know
        # when a position is an extrapolation rather than a measurement.
        self.covered = tuple(covered) if covered is not None else None

    # -- fitting ----------------------------------------------------------

    @classmethod
    def fit(cls, pixels, positions, z_table=None, camera=None):
        """Fit from pixel/position pairs. `positions` are (x, y) in metres.

        RANSAC rather than a plain least-squares fit: one mis-touched block would
        otherwise drag the whole mapping, and a bad correspondence is easy to
        produce by hand.
        """
        pixels = np.asarray(pixels, np.float32).reshape(-1, 1, 2)
        positions = np.asarray(positions, np.float32).reshape(-1, 1, 2)
        if len(pixels) < MIN_CORRESPONDENCES:
            raise ValueError(
                f"Need at least {MIN_CORRESPONDENCES} correspondences, got {len(pixels)}")

        matrix, inliers = cv2.findHomography(
            pixels, positions, cv2.RANSAC,
            RANSAC_THRESHOLD_PX * 1e-3)  # threshold is in the output units, metres
        if matrix is None:
            raise ValueError("Homography fit failed; the points may be collinear")

        flat = positions.reshape(-1, 2)
        frame = cls(matrix, z_table=z_table, camera=camera,
                    covered=(float(flat[:, 0].min()), float(flat[:, 0].max()),
                             float(flat[:, 1].min()), float(flat[:, 1].max())))
        frame.residuals_mm = frame.residuals(pixels.reshape(-1, 2), flat)
        frame.inliers = inliers.ravel().astype(bool) if inliers is not None else None
        return frame

    def residuals(self, pixels, positions):
        """Per-point error in millimetres between mapped pixels and known positions."""
        mapped = np.array([self.to_arm(pixel) for pixel in pixels])
        return (1000 * np.linalg.norm(mapped - np.asarray(positions, float),
                                      axis=1)).tolist()

    # -- use --------------------------------------------------------------

    def to_arm(self, pixel):
        """(u, v) in the image -> (x, y) in metres in the arm's frame."""
        point = np.array([pixel[0], pixel[1], 1.0], float)
        mapped = self.matrix @ point
        if abs(mapped[2]) < 1e-12:
            raise ValueError(f"Pixel {pixel} maps to infinity")
        return mapped[:2] / mapped[2]

    def to_pixel(self, position):
        """(x, y) in metres -> (u, v) in the image."""
        point = np.array([position[0], position[1], 1.0], float)
        mapped = np.linalg.inv(self.matrix) @ point
        if abs(mapped[2]) < 1e-12:
            raise ValueError(f"Position {position} maps to infinity")
        return mapped[:2] / mapped[2]

    def outside_covered_mm(self, position):
        """How far a position lies beyond the calibrated region, in millimetres."""
        if self.covered is None:
            return 0.0
        x_min, x_max, y_min, y_max = self.covered
        dx = max(x_min - position[0], position[0] - x_max, 0.0)
        dy = max(y_min - position[1], position[1] - y_max, 0.0)
        return 1000 * float(np.hypot(dx, dy))

    def reach_target(self, pixel, z=None):
        """Full 3D target for IK: the block's position at the table height."""
        x, y = self.to_arm(pixel)
        height = self.z_table if z is None else z
        if height is None:
            raise ValueError("No table height recorded; pass z explicitly")
        return np.array([x, y, height])

    # -- persistence ------------------------------------------------------

    def save(self, path=CALIBRATION_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "matrix": self.matrix.tolist(),
            "z_table": self.z_table,
            "residuals_mm": self.residuals_mm,
            "camera": self.camera,
            "covered": list(self.covered) if self.covered else None,
        }, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path=CALIBRATION_PATH):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/calibrate_table.py first.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["matrix"], data.get("z_table"),
                   data.get("residuals_mm"), data.get("camera"),
                   data.get("covered"))

    def __str__(self):
        if not self.residuals_mm:
            return f"TableFrame(camera={self.camera})"
        worst = max(self.residuals_mm)
        mean = sum(self.residuals_mm) / len(self.residuals_mm)
        region = ""
        if self.covered:
            region = (f", covers x {self.covered[0]:+.3f}..{self.covered[1]:+.3f} "
                      f"y {self.covered[2]:+.3f}..{self.covered[3]:+.3f}")
        return (f"TableFrame(camera={self.camera}, z={self.z_table:.3f}m, "
                f"{len(self.residuals_mm)} points, mean {mean:.1f}mm, "
                f"worst {worst:.1f}mm{region})")
