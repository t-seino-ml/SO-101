"""Where does a block held in the jaws appear in the wrist view, today?

data/grasp_pixel.json says (390, 193), measured from the recordings on 10 September.
Everything the visual servo does is aimed at that pixel: it moves the arm until
the block sits there and then closes. So if the wrist camera has shifted on its
bracket since - and the side camera has moved about 2 mm in the same period -
the servo converges beautifully onto the wrong place, which is exactly what it
was seen doing: 10 px of residual error and jaws closing on air.

This re-measures it with no inference and no recordings. The arm is limp except
for the gripper; put a block between the jaws, and the detector says where that
block is in the wrist image. That pixel IS the grasp pixel.

Do it at two or three arm poses if you can: the number should barely move, and
if it does, the camera is loose rather than merely shifted.

    uv run scripts/check_grasp_pixel.py
    uv run scripts/check_grasp_pixel.py --samples 3
    uv run scripts/check_grasp_pixel.py --save        # write the new value
"""

import argparse
import json
import time
from pathlib import Path

from so101.platform import require_windows

require_windows()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from so101.camera import CameraSet  # noqa: E402
from so101.hardware import bus_patch  # noqa: F401,E402
from so101.hardware import resolve as resolve_port  # noqa: E402
from so101.hardware import tuning  # noqa: E402
from so101.policy import ArmKinematics, BlockDetector  # noqa: E402
from so101.policy.motion import joints_of, set_gripper  # noqa: E402

GRASP_PIXEL = Path("data/grasp_pixel.json")
OPEN_DEG = 45.0
SQUEEZE_DEG = 2.0
MIN_CONFIDENCE = 0.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2,
                        help="how many times to hold a block, ideally at "
                             "different arm poses")
    parser.add_argument("--wrist", default="wrist")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--save", action="store_true",
                        help="overwrite data/grasp_pixel.json with what is measured")
    parser.add_argument("--out", type=Path, default=Path("outputs/grasp_pixel"))
    args = parser.parse_args()

    stored = None
    if GRASP_PIXEL.is_file():
        data = json.loads(GRASP_PIXEL.read_text(encoding="utf-8"))
        stored = np.array(data["pixel"], float)
        print(f"  stored grasp pixel: ({stored[0]:.0f}, {stored[1]:.0f}) "
              f"from {data.get('samples', '?')} samples, "
              f"spread {data.get('spread_px', float('nan')):.1f} px")

    arm = ArmKinematics()
    detector = BlockDetector(weights=args.weights)
    detector.warmup()
    args.out.mkdir(parents=True, exist_ok=True)

    tuning.install(verbose=False)
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    cameras = CameraSet.from_config()
    cameras.start()
    robot = make_robot_from_config(SO101FollowerConfig(
        port=resolve_port([args.follower_port])[0], id="follower"))
    robot.connect()

    measured = []
    try:
        cameras.wait_for_frames(timeout=25)
        wrist = cameras.streams[args.wrist]
        print("\n  The arm is limp except for the gripper. Move it by hand to a")
        print("  pose like the one it grasps from, put a block between the jaws,")
        print("  and press ENTER. The gripper will squeeze; nothing else moves.\n")

        for sample in range(1, args.samples + 1):
            set_gripper(robot, OPEN_DEG, seconds=1.0)
            input(f"  [{sample}/{args.samples}] block between the jaws, ENTER: ")
            set_gripper(robot, SQUEEZE_DEG, seconds=1.0)
            time.sleep(1.0)
            held = joints_of(robot)["gripper"]
            time.sleep(0.6)                  # let the frame catch up

            image = wrist.read().image.copy()
            found = [d for d in detector.detect(image)
                     if d.confidence >= MIN_CONFIDENCE]
            if not found:
                print("      nothing detected in the wrist view; try again")
                continue
            # The held block is the one nearest the middle of the jaws, which is
            # roughly where the stored pixel says - but fall back to the biggest,
            # since the held block is much the closest thing to the camera.
            block = max(found, key=lambda d: d.width_px)
            measured.append(block.pixel)
            tip = arm.forward({n: joints_of(robot)[n] for n in arm.joint_names})
            print(f"      jaws at {held:.1f} deg, {block.colour} at pixel "
                  f"({block.pixel[0]:.0f}, {block.pixel[1]:.0f}), "
                  f"{block.width_px:.0f} px wide")
            print(f"      gripper frame at x={tip[0]:+.3f} y={tip[1]:+.3f} "
                  f"z={tip[2]:+.3f}")
            if stored is not None:
                shift = block.pixel - stored
                print(f"      that is {shift[0]:+.0f}, {shift[1]:+.0f} px from "
                      f"the stored pixel ({np.linalg.norm(shift):.0f} px)")

            x0, y0, x1, y1 = (int(v) for v in block.box)
            cv2.rectangle(image, (x0, y0), (x1, y1), (60, 220, 60), 2)
            cv2.circle(image, tuple(int(v) for v in block.pixel), 6, (60, 220, 60), -1)
            if stored is not None:
                cv2.drawMarker(image, tuple(int(v) for v in stored), (0, 0, 255),
                               cv2.MARKER_CROSS, 20, 2)
                cv2.putText(image, "stored", (int(stored[0]) + 10, int(stored[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(str(args.out / f"held_{sample:02d}.png"), image)
            print(f"      {args.out / f'held_{sample:02d}.png'}\n")

        set_gripper(robot, OPEN_DEG, seconds=1.0)
        print("  jaws open; take the block back")
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        cameras.stop()
        robot.disconnect()
        print("  arm relaxed")

    if not measured:
        raise SystemExit("\n  nothing measured")
    points = np.array(measured)
    middle = points.mean(axis=0)
    spread = float(np.linalg.norm(points - middle, axis=1).max())
    print(f"\n  measured grasp pixel: ({middle[0]:.0f}, {middle[1]:.0f})"
          f"   over {len(points)} sample(s), spread {spread:.1f} px")
    if stored is not None:
        shift = middle - stored
        print(f"  the stored value is off by {shift[0]:+.0f}, {shift[1]:+.0f} px "
              f"({np.linalg.norm(shift):.0f} px)")
        print(f"  at roughly 3.3 px per mm that is about "
              f"{np.linalg.norm(shift)/3.3:.0f} mm of aiming error")

    if args.save:
        GRASP_PIXEL.write_text(json.dumps({
            "pixel": middle.tolist(),
            "width": 800, "height": 600,
            "samples": len(points),
            "spread_px": spread,
            "camera": args.wrist,
        }, indent=2), encoding="utf-8")
        print(f"\n  saved {GRASP_PIXEL}")
    else:
        print("\n  not saved; re-run with --save to store it")


if __name__ == "__main__":
    main()
