"""A real camera model: intrinsics, extrinsics, and rays that meet a height.

What this replaces is a plane-to-plane homography, which maps one height and
silently mis-maps every other. Blocks sit at 10 mm, the can's mouth at 35 mm, a
pre-grasp hover at 80 mm; a homography fitted at the table answers all three with
the table's answer, and the error grows with height. Measured on the can, the
gap between its box centre and its base came to 83 mm - twice the can's radius.

With K, R and t the question becomes a ray: a pixel names a line out of the
camera, and the answer is where that line crosses the height the object's feature
sits at. One model, any height, no per-height fudge factors.

Deliberately independent of the arm. Nothing here takes a TCP, a jaw offset or a
gripper pose, and the calibration takes 3D points measured some other way. That
separation is the point: when the homography was fitted from the arm's own
positions it absorbed the gripper's geometry into the camera's, and the two could
never be told apart afterwards - which is exactly how a 10.7 mm disagreement
about where the jaws are ended up unresolvable. Calibrate the camera against
something that is not the robot, and a leftover error belongs to the robot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CAMERA_PATH = Path("data/camera_model.json")


class PinholeCamera:
    """K, R, t - and the ray-plane intersection everything else is built on."""

    def __init__(self, matrix, rotation, translation, distortion=None,
                 size=None, name=None, reprojection_px=None):
        self.matrix = np.asarray(matrix, float).reshape(3, 3)
        #: World-to-camera, as OpenCV's solvePnP returns it.
        self.rotation = np.asarray(rotation, float).reshape(3, 3)
        self.translation = np.asarray(translation, float).reshape(3)
        self.distortion = (None if distortion is None
                           else np.asarray(distortion, float).ravel())
        self.size = tuple(size) if size else None
        self.name = name
        self.reprojection_px = reprojection_px

    # -- where the camera is ----------------------------------------------

    @property
    def centre(self):
        """The camera's own position, in world coordinates."""
        return -self.rotation.T @ self.translation

    # -- projection -------------------------------------------------------

    def project(self, points):
        """World points -> pixels. Accepts one point or many."""
        import cv2

        points = np.asarray(points, float).reshape(-1, 3)
        pixels, _ = cv2.projectPoints(
            points, cv2.Rodrigues(self.rotation)[0], self.translation,
            self.matrix, self.distortion)
        pixels = pixels.reshape(-1, 2)
        return pixels[0] if pixels.shape[0] == 1 else pixels

    def ray(self, pixel):
        """(origin, direction) in world coordinates for a pixel.

        Undistorted first if a distortion model is present, so the ray is the
        one the lens would have made if it were ideal.
        """
        import cv2

        pixel = np.asarray(pixel, float).reshape(1, 1, 2)
        if self.distortion is not None:
            normalised = cv2.undistortPoints(pixel, self.matrix, self.distortion)
        else:
            normalised = cv2.undistortPoints(pixel, self.matrix, None)
        x, y = normalised.reshape(2)
        direction = self.rotation.T @ np.array([x, y, 1.0])
        return self.centre, direction / np.linalg.norm(direction)

    def on_plane(self, pixel, height):
        """Where a pixel's ray crosses the horizontal plane at z = `height`.

        This is the whole point of the model. A block's top face, a can's rim
        and a hover position are three different heights and one camera.
        """
        origin, direction = self.ray(pixel)
        if abs(direction[2]) < 1e-9:
            raise ValueError(f"pixel {pixel} looks along the plane z={height}")
        distance = (height - origin[2]) / direction[2]
        if distance <= 0:
            raise ValueError(f"pixel {pixel} points away from z={height}")
        return origin + distance * direction

    # -- calibration ------------------------------------------------------

    @classmethod
    def calibrate(cls, object_points, image_points, size, name=None,
                  fix_distortion=False):
        """Fit K, distortion, R and t from views of known 3D geometry.

        `object_points` is a list of (n, 3) arrays, one per view, in the
        coordinates the target was measured in; `image_points` the matching
        (n, 2) pixel arrays. A single view is enough for R and t if K is already
        known, but intrinsics need several from different angles.
        """
        import cv2

        object_points = [np.asarray(p, np.float32).reshape(-1, 1, 3)
                         for p in object_points]
        image_points = [np.asarray(p, np.float32).reshape(-1, 1, 2)
                        for p in image_points]
        flags = cv2.CALIB_ZERO_TANGENT_DIST if fix_distortion else 0
        error, matrix, distortion, rotations, translations = cv2.calibrateCamera(
            object_points, image_points, tuple(size), None, None, flags=flags)
        return cls(matrix, cv2.Rodrigues(rotations[0])[0], translations[0],
                   distortion, size, name, float(error))

    @classmethod
    def locate(cls, matrix, distortion, object_points, image_points, size=None,
               name=None):
        """R and t for a camera whose K is already known, from one view."""
        import cv2

        object_points = np.asarray(object_points, np.float32).reshape(-1, 1, 3)
        image_points = np.asarray(image_points, np.float32).reshape(-1, 1, 2)
        ok, rotation, translation = cv2.solvePnP(
            object_points, image_points, np.asarray(matrix, float),
            None if distortion is None else np.asarray(distortion, float),
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError("solvePnP could not place the camera")
        camera = cls(matrix, cv2.Rodrigues(rotation)[0], translation, distortion,
                     size, name)
        camera.reprojection_px = float(np.mean(np.linalg.norm(
            camera.project(object_points.reshape(-1, 3))
            - image_points.reshape(-1, 2), axis=1)))
        return camera

    def residuals_px(self, object_points, image_points):
        """Per-point reprojection error, in pixels."""
        projected = self.project(np.asarray(object_points, float).reshape(-1, 3))
        return np.linalg.norm(
            projected - np.asarray(image_points, float).reshape(-1, 2), axis=1)

    # -- persistence ------------------------------------------------------

    def save(self, path=CAMERA_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "matrix": self.matrix.tolist(),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "distortion": (None if self.distortion is None
                           else self.distortion.tolist()),
            "size": list(self.size) if self.size else None,
            "name": self.name,
            "reprojection_px": self.reprojection_px,
        }, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path=CAMERA_PATH):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/calibrate_camera.py first.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["matrix"], data["rotation"], data["translation"],
                   data.get("distortion"), data.get("size"), data.get("name"),
                   data.get("reprojection_px"))

    def __str__(self):
        centre = self.centre
        focal = (self.matrix[0, 0] + self.matrix[1, 1]) / 2
        error = ("" if self.reprojection_px is None
                 else f", reprojects to {self.reprojection_px:.2f} px")
        return (f"PinholeCamera({self.name or 'unnamed'}, f={focal:.0f} px, "
                f"at x={centre[0]:+.3f} y={centre[1]:+.3f} z={centre[2]:+.3f} m"
                f"{error})")
