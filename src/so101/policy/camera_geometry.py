"""Where the camera is, so things that are not flat can still be placed.

The table homography maps one plane. Anything standing above that plane breaks
it: a point h above the table projects to where the line of sight through it
meets the table, which is further from the camera than the thing actually is.
For the blocks that error is a few millimetres and the measured jaw offset has
been quietly absorbing it. For the can, which is 35 mm tall, it is 83 mm - twice
the can's radius, so aiming at the wrong one misses the can completely.

Correcting it needs one number the homography does not carry: where the camera
is. That can be fitted from the blocks already lying on the table, with no
targets and nothing to measure. For each block the bottom of its bounding box
is a point genuinely on the table, and the box centre is a point that is not,
and the two are related by

    Q - P = alpha * (P - C)

with C the camera's position projected onto the table and alpha set by the
height. Writing b = alpha * C makes it linear in (alpha, b), so it is a least
squares fit over however many blocks are in view. Measured on this rig with 11
blocks: residuals of 1.0 mm median, 2.7 mm worst.

C itself comes out badly conditioned, and it is worth being clear about that.
The blocks occupy a patch a couple of hand-spans across and C is metres outside
it, so finding it means extrapolating a direction field a long way. Two fits from
two frames, each matching its blocks to about 1 mm, put the camera 0.6 m apart;
resampling the blocks moves it 41 mm at the median and 110 mm at the 90th
percentile. The reported `height_m` inherits all of that, and additionally rests
on the guess that a block's box centre sits 10 mm up. Do not read it as a
measurement of where the camera is.

None of which matters here, because the correction uses the direction to the
camera and not the distance, and sliding C away along that direction barely
changes it. Measured over the same resamples: the can's centre moves 0.1 mm at
the median and 0.4 mm at the 90th percentile, and the two 0.6 m apart fits put
it within 0.7 mm of each other. The wobble in C is real and it does not reach
the answer.

With C known the can needs no parallax model of its own, because one point on
the can really is on the table - the bottom of its bounding box, which is the
near edge of its base. The centre of the base is one radius further along, in
the direction pointing away from the camera.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GEOMETRY_PATH = Path("data/camera_geometry.json")
MIN_BLOCKS = 4
BLOCK_CENTRE_HEIGHT_M = 0.010     # only used to report a plausible camera height


class CameraGeometry:
    """The camera's position on the table plane, in the arm's frame."""

    def __init__(self, camera, alpha=None, residuals_mm=None, blocks=None):
        self.camera = np.asarray(camera, float).reshape(2)
        self.alpha = alpha
        self.residuals_mm = residuals_mm
        self.blocks = blocks

    # -- fitting ----------------------------------------------------------

    @classmethod
    def fit(cls, table_frame, detections):
        """Fit from block detections. Each needs a `box` in pixels.

        Blocks only: the fit assumes every object is the same height, and it is
        the box base being on the table that makes the sum work at all.
        """
        if len(detections) < MIN_BLOCKS:
            raise ValueError(
                f"Need at least {MIN_BLOCKS} blocks to fit the camera, "
                f"got {len(detections)}")

        rows, shifts, bases = [], [], []
        for detection in detections:
            x0, _, x1, y1 = detection.box
            base = table_frame.to_arm(np.array([(x0 + x1) / 2, y1]))
            shift = table_frame.to_arm(detection.pixel) - base
            bases.append(base)
            shifts.append(shift)
            rows.append([base[0], -1.0, 0.0])
            rows.append([base[1], 0.0, -1.0])
        targets = np.array(shifts).reshape(-1)
        solution, *_ = np.linalg.lstsq(np.array(rows), targets, rcond=None)
        alpha, bx, by = solution
        if abs(alpha) < 1e-9:
            raise ValueError("The blocks show no parallax; cannot place the camera")
        camera = np.array([bx, by]) / alpha

        residuals = [1000 * float(np.linalg.norm(
            base + alpha * (base - camera) - (base + shift)))
            for base, shift in zip(bases, shifts)]
        return cls(camera, float(alpha), residuals, len(detections))

    @property
    def height_m(self):
        """Roughly how far above the table the camera is. Indicative only."""
        if not self.alpha:
            return None
        return BLOCK_CENTRE_HEIGHT_M * (1 + self.alpha) / self.alpha

    # -- use --------------------------------------------------------------

    def away(self, position):
        """Unit vector at `position` pointing directly away from the camera."""
        direction = np.asarray(position, float)[:2] - self.camera
        length = np.linalg.norm(direction)
        if length < 1e-9:
            raise ValueError("That position is under the camera")
        return direction / length

    def on_table(self, table_frame, box, radius=0.0):
        """Where a round thing standing on the table actually is.

        `box` is (x0, y0, x1, y1) in pixels. The bottom edge is taken as the near
        edge of the base - true for a cylinder, near enough for anything round -
        and `radius` steps from there to the middle of the base.
        """
        x0, _, x1, y1 = box
        base = table_frame.to_arm(np.array([(x0 + x1) / 2, y1]))
        if not radius:
            return base
        return base + radius * self.away(base)

    # -- persistence ------------------------------------------------------

    def save(self, path=GEOMETRY_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "camera": self.camera.tolist(),
            "alpha": self.alpha,
            "residuals_mm": self.residuals_mm,
            "blocks": self.blocks,
        }, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path=GEOMETRY_PATH):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Fit it from a frame with blocks in it.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["camera"], data.get("alpha"), data.get("residuals_mm"),
                   data.get("blocks"))

    def __str__(self):
        if not self.residuals_mm:
            return f"CameraGeometry(over x={self.camera[0]:+.3f} y={self.camera[1]:+.3f})"
        worst = max(self.residuals_mm)
        mean = sum(self.residuals_mm) / len(self.residuals_mm)
        return (f"CameraGeometry(over x={self.camera[0]:+.3f} y={self.camera[1]:+.3f}, "
                f"about {1000*self.height_m:.0f}mm up, {self.blocks} blocks, "
                f"mean {mean:.1f}mm, worst {worst:.1f}mm)")
