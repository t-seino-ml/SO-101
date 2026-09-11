"""Find where a block appears in the wrist view when the jaws are around it.

Closing the loop on the wrist camera needs a target: the pixel a block has to be
driven to before the jaws will close on it. It is not the middle of the frame -
the camera is bolted beside the gripper, not down its axis - and it cannot be
derived from the URDF, because the jaws are not where the URDF's gripper frame
is.

The recordings already contain the answer. At the moment the jaws closed in each
demonstration they were around a block, so whatever the wrist camera saw between
them at that frame is, by definition, the target. Thirty episodes give thirty
readings of it.

Because the camera is fixed to the arm, this pixel is a property of the rig and
does not change when the table, the lighting or the external camera does.

    uv run scripts/fit_grasp_pixel.py
    uv run scripts/fit_grasp_pixel.py --show
"""

import argparse
import json
from pathlib import Path

import numpy as np

from so101.policy import ArmKinematics, BlockDetector
from so101.policy.approach import grasp_frame

TARGET_PATH = Path("data/grasp_pixel.json")
MIN_CONFIDENCE = 0.4
MIN_CONFIDENCE = 0.4
# Roughly where the jaws hold a block, read off the recordings. The fit only
# has to refine this; it needs to be in the right neighbourhood, not exact.
SEED_PIXEL = np.array([405.0, 180.0])

def show_stored(path):
    if not path.is_file():
        print(f"  {path} not found; nothing measured yet")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    pixel = data["pixel"]
    print(f"  jaws hold a block at pixel ({pixel[0]:.0f}, {pixel[1]:.0f}) "
          f"of {data['width']}x{data['height']}")
    print(f"  from {data['samples']} episode(s), spread {data['spread_px']:.0f} px")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=Path, default=Path("data/demos"))
    parser.add_argument("--repo-id", default="so101/grasp_block")
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=TARGET_PATH)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.show:
        return show_stored(args.out)

    import pandas as pd
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    detector = BlockDetector(weights=args.weights)
    detector.warmup()
    print(f"  detector: {detector.weights}")

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.demos)
    dataset = LeRobotDataset(args.repo_id, root=args.demos)
    names = [name.removesuffix(".pos") for name in meta.features["action"]["names"]]
    gripper = names.index("gripper")
    camera = f"observation.images.{args.camera}"

    frames = pd.concat([pd.read_parquet(path)
                        for path in sorted(Path(args.demos, "data").rglob("*.parquet"))])
    frames = frames.sort_values(["episode_index", "frame_index"])

    candidates, size = [], None
    arm = ArmKinematics()
    def detections(dataset_index):
        image = dataset[dataset_index][camera].numpy()
        # The dataset holds RGB; the detector was trained on what cv2
        # reads, which is BGR. Handing it RGB swaps red for blue and
        # every colour label with it.
        image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, ::-1]
        return image.shape[1::-1], [d for d in detector.detect(image)
                                    if d.confidence >= MIN_CONFIDENCE]

    print(f"\n  {'episode':>8}{'held':>10}{'pixel':>16}{'off':>9}")
    for episode, rows in frames.groupby("episode_index"):
        states = np.stack(rows["observation.state"].to_numpy())
        tips = np.array([arm.forward({name: float(state[i])
                                      for i, name in enumerate(names)
                                      if name in arm.joint_names})
                         for state in states])
        closed = grasp_frame(states[:, gripper], tips[:, 2], meta.fps)
        start = int(meta.episodes["dataset_from_index"][episode])

        size, at_grasp = detections(start + closed)
        if not at_grasp:
            print(f"  {episode:>8}{chr(45):>10}   nothing detected")
            continue

        # Pick the block nearest where the jaws are known to be. That place is
        # solved for below; here it starts from a rough reading taken off the
        # recordings themselves - the marker lands on the held block in every
        # episode checked - and the fit only has to refine it. Seeding from the
        # middle of the frame instead pulls the choice onto whichever block
        # happens to be central, which is how two earlier attempts at this
        # settled on a block nobody ever grasped.
        candidates.append((int(episode), at_grasp))

    if len(candidates) < 5:
        raise SystemExit("  Too few episodes had anything detected.")

    # One pass from the seed, not a search. Letting the target chase the median
    # of its own choices drifts: with a dozen blocks in every frame there is
    # always a denser cluster elsewhere to fall into, and it found two different
    # ones on two runs. The seed is read off the frames and is good to a block's
    # width, so nearest-to-seed already picks the right block.
    target = SEED_PIXEL.astype(float)
    chosen = [min(near, key=lambda d: np.linalg.norm(d.pixel - target))
              for _, near in candidates]
    pixels = np.array([d.pixel for d in chosen])

    for (episode, _), held in zip(candidates, chosen):
        where = f"{held.pixel[0]:.0f}, {held.pixel[1]:.0f}"
        off = float(np.linalg.norm(held.pixel - target))
        print(f"  {episode:>8}{held.colour:>10}{where:>16}{off:>8.0f}p")

    keep = np.linalg.norm(pixels - target, axis=1) < 120
    if keep.sum() >= 5:
        target = np.median(pixels[keep], axis=0)

    spread = float(np.median(np.linalg.norm(pixels[keep] - target, axis=1)))

    print(f"\n  {int(keep.sum())} of {len(pixels)} readings agree")
    print(f"  jaws hold a block at pixel ({target[0]:.0f}, {target[1]:.0f}) "
          f"of {size[0]}x{size[1]}")
    print(f"  that is {target[0] - size[0]/2:+.0f}, {target[1] - size[1]/2:+.0f} "
          f"from the middle of the frame")
    print(f"  spread {spread:.0f} px")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "pixel": target.tolist(),
        "width": size[0],
        "height": size[1],
        "samples": int(keep.sum()),
        "spread_px": spread,
        "camera": args.camera,
    }, indent=2), encoding="utf-8")
    print(f"\n  saved {args.out}")


if __name__ == "__main__":
    main()
