"""Render the simulated rig, so the geometry can be checked by eye.

Numbers said the two models agree. This says whether the thing they agree about
looks like the actual bench: arm at the near edge, blocks in front of it, the tin
off to one side, and the gripper where the gripper should be.

    uv run scripts/sim_render.py
    uv run scripts/sim_render.py --pose 0,-30,60,40,0 --out outputs/sim/pose.png
"""

import argparse
from pathlib import Path

import numpy as np

from so101.sim import SO101Sim

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
VIEWS = {
    # (azimuth, elevation, distance, lookat)
    "overview": (150.0, -25.0, 0.85, (0.22, 0.0, 0.02)),
    "side":     (180.0, -12.0, 0.75, (0.25, 0.0, 0.02)),
    "top":      (180.0, -85.0, 0.70, (0.25, 0.0, 0.00)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", default="0,-30,60,40,0",
                        help="the five arm joints in degrees")
    parser.add_argument("--gripper", type=float, default=35.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", type=Path, default=Path("outputs/sim"))
    args = parser.parse_args()

    pose = dict(zip(JOINTS, (float(v) for v in args.pose.split(","))))
    sim = SO101Sim(
        blocks=[("red", (0.26, -0.05)), ("orange", (0.30, -0.02)),
                ("yellow", (0.24, 0.02)), ("green", (0.29, 0.06)),
                ("blue", (0.22, -0.10)), ("purple", (0.32, 0.01))],
        can=(0.28, 0.13))
    sim.set_joints(pose, gripper_deg=args.gripper)
    print(f"  {sim}")
    position, rotation = sim.gripper_pose()
    print(f"  gripper frame at {1000*position[0]:+.1f}, {1000*position[1]:+.1f}, "
          f"{1000*position[2]:+.1f} mm, tool axis "
          f"{rotation[0,2]:+.2f}, {rotation[1,2]:+.2f}, {rotation[2,2]:+.2f}")

    import mujoco

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    with mujoco.Renderer(sim.model, height=args.height, width=args.width) as renderer:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        for name, (azimuth, elevation, distance, lookat) in VIEWS.items():
            camera.azimuth, camera.elevation = azimuth, elevation
            camera.distance, camera.lookat = distance, np.array(lookat)
            renderer.update_scene(sim.data, camera=camera)
            frame = renderer.render()
            path = args.out / f"rig_{name}.png"
            _write_png(path, frame)
            written.append(path)

    for path in written:
        print(f"  {path}")


def _write_png(path, rgb):
    """Save without pulling in an image library the project does not need."""
    import cv2

    cv2.imwrite(str(path), rgb[:, :, ::-1])


if __name__ == "__main__":
    main()
