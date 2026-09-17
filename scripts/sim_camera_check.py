"""Phase S6/S7: does a calibrated camera beat a homography, and by how much?

The homography in use maps pixels to one plane. Everything the task touches sits
at a different height - a block's box centre around 10 mm, the can's rim at
35 mm, a hover at 80 mm - and the homography answers all of them with the table's
answer. This measures what that costs, against a camera model that knows where
the camera is and intersects a ray with whichever height is asked for.

The calibration here deliberately never touches the arm. Correspondences come
from a planar target of known geometry held at several angles, the way a camera
is normally calibrated, so nothing about the gripper can leak into K, R or t.
That independence is the point: the 10.7 mm disagreement between the CAD TCP and
the one measured on the real arm is only diagnosable if the camera's errors and
the gripper's are estimated separately.

Everything is synthetic - a camera with chosen parameters projects points whose
3D positions are known exactly - so what is being tested is the pipeline, not the
lens. Pixel noise is added at the level the detector actually shows.

    uv run scripts/sim_camera_check.py
    uv run scripts/sim_camera_check.py --noise 1.0
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from so101.policy.camera_model import PinholeCamera

SIZE = (800, 600)
BLOCK_TOP_M = 0.020
CAN_RIM_M = 0.035
HOVER_M = 0.080


def truth_camera():
    """A camera placed roughly where the real side camera is.

    Not a fit to the real one - a stand-in with plausible numbers, so the
    pipeline can be exercised before anyone points a lens at anything.
    """
    focal = 700.0
    matrix = np.array([[focal, 0, SIZE[0] / 2],
                       [0, focal, SIZE[1] / 2],
                       [0, 0, 1.0]])
    centre = np.array([0.46, -0.50, 0.42])
    forward = np.array([0.15, 0.10, 0.0]) - centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])       # world -> camera
    return PinholeCamera(matrix, rotation, -rotation @ centre,
                         distortion=np.array([-0.08, 0.02, 0.0, 0.0, 0.0]),
                         size=SIZE, name="synthetic side")


def target_views(truth, rng, views=9, rows=6, columns=8, pitch=0.025):
    """A planar grid of known geometry, seen at several poses.

    A calibration target, in other words. Its coordinates are its own - the
    corners of a printed board - and it knows nothing about the robot.
    """
    board = np.array([[c * pitch, r * pitch, 0.0]
                      for r in range(rows) for c in range(columns)])
    board -= board.mean(axis=0)
    object_points, image_points = [], []
    for _ in range(views):
        angles = rng.uniform(-0.45, 0.45, 3)
        rotation = _euler(angles)
        placed = board @ rotation.T + np.array([
            rng.uniform(0.15, 0.32), rng.uniform(-0.10, 0.10),
            rng.uniform(0.00, 0.12)])
        pixels = truth.project(placed)
        inside = np.all((pixels > 10) & (pixels < np.array(SIZE) - 10), axis=1)
        if inside.sum() < len(board) * 0.9:
            continue
        object_points.append(board)
        image_points.append(pixels)
    return object_points, image_points


def _euler(angles):
    x, y, z = angles
    cx, sx, cy, sy, cz, sz = (np.cos(x), np.sin(x), np.cos(y), np.sin(y),
                              np.cos(z), np.sin(z))
    return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))


def fit_homography(truth, table_z, rng, noise, points=12):
    """The current approach: pixels to the table plane, fitted on the table."""
    import cv2

    world = np.array([[rng.uniform(0.15, 0.32), rng.uniform(-0.15, 0.15), table_z]
                      for _ in range(points)])
    pixels = truth.project(world) + rng.normal(0, noise, (points, 2))
    matrix, _ = cv2.findHomography(pixels.astype(np.float32),
                                   world[:, :2].astype(np.float32),
                                   cv2.RANSAC, 0.008)
    return matrix


def through_homography(matrix, pixel, table_z):
    point = np.array([pixel[0], pixel[1], 1.0])
    mapped = matrix @ point
    return np.array([mapped[0] / mapped[2], mapped[1] / mapped[2], table_z])


def calibrate_at(truth, noise, seed, table_z):
    """Calibrate from a target at this noise level, and say how far off it is."""
    rng = np.random.default_rng(seed)
    object_points, image_points = target_views(truth, rng)
    noisy = [p + rng.normal(0, noise, p.shape) for p in image_points]
    fitted = PinholeCamera.calibrate(object_points, noisy, SIZE)
    board = np.array([[x, y, table_z]
                      for x in np.linspace(0.14, 0.32, 5)
                      for y in np.linspace(-0.14, 0.14, 5)])
    seen = truth.project(board) + rng.normal(0, noise, (len(board), 2))
    camera = PinholeCamera.locate(fitted.matrix, fitted.distortion, board, seen,
                                  SIZE, "calibrated")
    focal = abs((camera.matrix[0, 0] + camera.matrix[1, 1]) / 2
                - (truth.matrix[0, 0] + truth.matrix[1, 1]) / 2)
    return camera, focal, 1000 * float(np.linalg.norm(
        camera.centre - truth.centre))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.3,
                        help="pixel noise on every detection")
    parser.add_argument("--sweep", nargs="*", type=float,
                        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
                        help="noise levels to run the calibration at; the "
                             "zero-noise row is the one that matters most, "
                             "because anything but a near-exact answer there "
                             "is a fault in the pipeline rather than in the data")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--table-z", type=float, default=-0.0013)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/sim/camera_check.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    truth = truth_camera()
    print(f"  truth: {truth}\n")

    # -- how much of the error is noise, and how much is the pipeline? -----
    print("  calibration against pixel noise\n")
    print(f"  {'noise':>8}{'focal err':>12}{'centre err':>13}"
          f"{'reprojection':>15}{'ray at 35 mm':>15}")
    sensitivity = {}
    for level in args.sweep:
        camera, focal, centre = calibrate_at(truth, level, args.seed,
                                             args.table_z)
        probe = np.array([[0.24, 0.02, args.table_z + CAN_RIM_M]])
        pixel = truth.project(probe)
        ray = 1000 * float(np.linalg.norm(
            camera.on_plane(pixel, probe[0, 2]) - probe[0]))
        sensitivity[level] = {"focal_px": float(focal),
                              "centre_mm": float(centre),
                              "reprojection_px": camera.reprojection_px,
                              "ray_mm": ray}
        print(f"  {level:>6.2f}px{focal:11.2f}px{centre:11.2f}mm"
              f"{camera.reprojection_px:13.3f}px{ray:13.2f}mm")
    print("\n  at zero noise the answer should be exact; anything left there is")
    print("  the pipeline's own error, not the measurement's.\n")

    # -- calibrate, from the target alone ----------------------------------
    object_points, image_points = target_views(truth, rng)
    noisy = [p + rng.normal(0, args.noise, p.shape) for p in image_points]
    fitted = PinholeCamera.calibrate(object_points, noisy, SIZE,
                                     name="calibrated")
    print(f"  calibrated from {len(object_points)} views of a 6x8 board, "
          f"{args.noise:.1f} px noise")
    print(f"  {fitted}")
    focal_error = abs((fitted.matrix[0, 0] + fitted.matrix[1, 1]) / 2
                      - (truth.matrix[0, 0] + truth.matrix[1, 1]) / 2)
    print(f"  focal length off by {focal_error:.1f} px")

    # Extrinsics against the table: one view of points whose 3D positions are
    # known some other way - a printed target lying on it, not the arm.
    board = np.array([[x, y, args.table_z]
                      for x in np.linspace(0.14, 0.32, 5)
                      for y in np.linspace(-0.14, 0.14, 5)])
    seen = truth.project(board) + rng.normal(0, args.noise, (len(board), 2))
    camera = PinholeCamera.locate(fitted.matrix, fitted.distortion, board, seen,
                                  SIZE, "calibrated")
    centre_error = 1000 * np.linalg.norm(camera.centre - truth.centre)
    print(f"  {camera}")
    print(f"  camera centre off by {centre_error:.1f} mm\n")

    # -- the comparison ----------------------------------------------------
    homography = fit_homography(truth, args.table_z, rng, args.noise)
    print(f"  recovering 3D positions at three heights, {args.samples} points "
          f"each\n")
    print(f"  {'feature':<28}{'height':>9}{'homography':>14}{'ray-plane':>13}")
    results = {}
    for label, height in (("block box centre", args.table_z + BLOCK_TOP_M / 2),
                          ("can rim", args.table_z + CAN_RIM_M),
                          ("hover position", args.table_z + HOVER_M)):
        world = np.array([[rng.uniform(0.15, 0.32), rng.uniform(-0.15, 0.15),
                           height] for _ in range(args.samples)])
        pixels = truth.project(world) + rng.normal(0, args.noise,
                                                   (args.samples, 2))
        flat, rays = [], []
        for point, pixel in zip(world, pixels):
            flat.append(1000 * np.linalg.norm(
                through_homography(homography, pixel, args.table_z)[:2]
                - point[:2]))
            rays.append(1000 * np.linalg.norm(
                camera.on_plane(pixel, height) - point))
        results[label] = {"height_mm": 1000 * (height - args.table_z),
                          "homography_median_mm": float(np.median(flat)),
                          "ray_median_mm": float(np.median(rays)),
                          "ray_p95_mm": float(np.percentile(rays, 95))}
        print(f"  {label:<28}{1000*(height-args.table_z):7.0f}mm"
              f"{np.median(flat):12.1f}mm{np.median(rays):11.2f}mm")

    print(f"\n  the homography is fitted on the table, so it is only right there;")
    print(f"  its error grows with the height of whatever is being looked at.")
    print(f"  the ray-plane model is told the height and is right at all of them.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "noise_px": args.noise,
        "focal_error_px": float(focal_error),
        "camera_centre_error_mm": float(centre_error),
        "sensitivity": {str(k): v for k, v in sensitivity.items()},
        "reprojection_px": camera.reprojection_px,
        "by_feature": results,
    }, indent=1), encoding="utf-8")
    camera.save()
    print(f"\n  {args.out}")
    print(f"  data/camera_model.json  (the synthetic one, for wiring tests)")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
