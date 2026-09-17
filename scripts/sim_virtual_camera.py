"""Phase S8: the same test, through a rendered picture instead of arithmetic.

Phase S6 projected points with the camera model and recovered them with the same
model, which tests the algebra and nothing else. This puts a camera in the
simulation, renders what it sees, finds the blocks in the image the way a
detector would, and asks where they are. Everything between the block and the
answer is exercised: the renderer's projection, the pixel the block lands on, the
calibration, the ray.

Ground truth is exact - MuJoCo knows where it put every block - so the error that
comes out is the pipeline's, with no calibration target to blame.

The camera is calibrated from a board, not from the arm. That separation is the
point of this phase: whatever error is left after this belongs to the robot, not
to the optics, which is the only way the 10.7 mm disagreement about the TCP can
ever be attributed to one or the other.

    uv run scripts/sim_virtual_camera.py
    uv run scripts/sim_virtual_camera.py --blocks 30
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from so101.policy.camera_model import PinholeCamera
from so101.sim import BLOCK_SIZE_M, SO101Sim

SIZE = (800, 600)
CAMERA_AT = (0.46, -0.50, 0.42)
CAMERA_LOOKS_AT = (0.22, -0.02, 0.0)
FOVY_DEG = 45.0
COLOURS = ("red", "orange", "yellow", "green", "blue", "purple")


def intrinsics_from(fovy_deg, size):
    """What MuJoCo's vertical field of view means as a pinhole matrix."""
    height = size[1]
    focal = height / (2 * np.tan(np.radians(fovy_deg) / 2))
    return np.array([[focal, 0, size[0] / 2],
                     [0, focal, size[1] / 2],
                     [0, 0, 1.0]])


def render(sim, size):
    """The picture, and a per-pixel map of which geom is showing.

    The segmentation is what stands in for the detector. Matching the blocks by
    colour looked simpler and was not: MuJoCo renders the arm in its default
    yellow, so a yellow block's match is mostly robot, and tightening the
    threshold to exclude it excluded the shaded faces of the blocks too. The
    renderer already knows which pixels belong to which geom - asking it is both
    exact and shorter.
    """
    import mujoco

    with mujoco.Renderer(sim.model, height=size[1], width=size[0]) as renderer:
        renderer.update_scene(sim.data, camera="side")
        image = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(sim.data, camera="side")
        segmentation = renderer.render().copy()
    return image, segmentation


def centroids(segmentation, sim):
    """Where each block sits in the picture, from the renderer's own labels.

    Standing in for the detector: the real one puts a box round a block and
    takes its middle, and so does this. What is being measured is everything
    after that point - the pixel, the calibration, the ray - not the detector.
    """
    import mujoco

    ids = segmentation[:, :, 0]
    types = segmentation[:, :, 1]
    found = {}
    for index, (name, _) in enumerate(sim.blocks):
        geom = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM,
                                 f"block_{index}_geom")
        mask = (types == mujoco.mjtObj.mjOBJ_GEOM) & (ids == geom)
        if mask.sum() < 40:
            continue
        ys, xs = np.nonzero(mask)
        found[name] = (float(xs.mean()), float(ys.mean()), int(mask.sum()))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.0,
                        help="extra pixel noise on the found centroids")
    parser.add_argument("--out", type=Path, default=Path("outputs/sim"))
    args = parser.parse_args()

    table_z = -0.0013
    # One block per colour, spread over the reachable part of the table.
    places = [(0.16, -0.10), (0.20, 0.06), (0.24, -0.04), (0.27, 0.08),
              (0.30, -0.08), (0.21, -0.14)]
    blocks = list(zip(COLOURS, places))
    sim = SO101Sim(table_z=table_z, blocks=blocks,
                   wrist_camera=(CAMERA_AT, CAMERA_LOOKS_AT, FOVY_DEG))
    sim.set_joints({"shoulder_pan": 0.0, "shoulder_lift": -95.0,
                    "elbow_flex": 90.0, "wrist_flex": 65.0, "wrist_roll": 0.0},
                   gripper_deg=30.0)
    print(f"  {sim}")

    image, segmentation = render(sim, SIZE)
    import cv2

    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "virtual_camera.png"), image[:, :, ::-1])

    # The camera, as MuJoCo has it. Used to place the board's projections, so
    # the calibration has something honest to work from.
    matrix = intrinsics_from(FOVY_DEG, SIZE)
    position = np.array(CAMERA_AT, float)
    forward = np.array(CAMERA_LOOKS_AT, float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    truth = PinholeCamera(matrix, rotation, -rotation @ position, None, SIZE,
                          "mujoco side")
    print(f"  {truth}")

    # Calibrate extrinsics from a board on the table - known geometry, nothing
    # to do with the arm. Intrinsics come from the renderer's own field of view,
    # which is the simulation's equivalent of a datasheet.
    board = np.array([[x, y, table_z] for x in np.linspace(0.14, 0.32, 5)
                      for y in np.linspace(-0.14, 0.14, 5)])
    rng = np.random.default_rng(0)
    seen = truth.project(board) + rng.normal(0, args.noise, (len(board), 2))
    camera = PinholeCamera.locate(matrix, None, board, seen, SIZE, "calibrated")
    print(f"  {camera}")
    print(f"  camera centre off by "
          f"{1000*np.linalg.norm(camera.centre - truth.centre):.2f} mm\n")

    # -- what the picture says the blocks' positions are -------------------
    spots = centroids(segmentation, sim)
    print(f"  {len(spots)} of {len(blocks)} blocks found in the render\n")
    print(f"  {'colour':<9}{'pixel':>14}{'truth x,y (mm)':>20}"
          f"{'recovered':>20}{'error':>10}")
    rows, errors = [], []
    for name, (u, v, area) in spots.items():
        index = [i for i, (colour, _) in enumerate(sim.blocks)
                 if colour == name][0]
        actual = sim.geom_position(f"block_{index}_geom")
        pixel = np.array([u, v]) + rng.normal(0, args.noise, 2)
        # The detector sees the block's face; its centre is at the block's
        # middle height, which is the plane to intersect.
        recovered = camera.on_plane(pixel, actual[2])
        error = 1000 * float(np.linalg.norm(recovered[:2] - actual[:2]))
        errors.append(error)
        rows.append({"colour": name, "camera_u": round(float(u), 2),
                     "camera_v": round(float(v), 2),
                     "truth_x": round(float(actual[0]), 5),
                     "truth_y": round(float(actual[1]), 5),
                     "estimated_x": round(float(recovered[0]), 5),
                     "estimated_y": round(float(recovered[1]), 5),
                     "error_mm": round(error, 3)})
        print(f"  {name:<9}{f'{u:.0f}, {v:.0f}':>14}"
              f"{f'{1000*actual[0]:+.0f}, {1000*actual[1]:+.0f}':>20}"
              f"{f'{1000*recovered[0]:+.0f}, {1000*recovered[1]:+.0f}':>20}"
              f"{error:8.1f}mm")

    if errors:
        print(f"\n  median {np.median(errors):.2f} mm, worst {max(errors):.2f} mm")
        print(f"  (the centroid of a cube's visible faces is not exactly above")
        print(f"   its middle, so some of this is the stand-in detector, not the")
        print(f"   geometry - the same bias the real detector has.)")

    (args.out / "virtual_camera.json").write_text(json.dumps({
        "camera_centre_error_mm": float(1000 * np.linalg.norm(
            camera.centre - truth.centre)),
        "reprojection_px": camera.reprojection_px,
        "median_error_mm": float(np.median(errors)) if errors else None,
        "worst_error_mm": float(max(errors)) if errors else None,
        "blocks": rows,
    }, indent=1), encoding="utf-8")
    print(f"\n  {args.out / 'virtual_camera.png'}")
    print(f"  {args.out / 'virtual_camera.json'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
