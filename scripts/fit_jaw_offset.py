"""Measure where the jaws close, relative to where forward kinematics says.

The URDF's gripper frame is not the point the jaws meet on a block, and the gap
between them is why reaching "to the block's position" puts the block outside
the jaws. It is one constant, so it only has to be measured once - but measuring
it needs a pose whose jaws are known to be on a block, and the recordings are
full of those.

Each episode ends with a block in the can, so the block it grasped is the one
that is on the table at the start of that episode and gone by the start of the
next. The side camera sees both starts with the arm parked out of the way. That
gives, for every episode, a grasp pose and the place its jaws closed - and the
difference between that place and forward kinematics is the offset.

Episodes where the operator put blocks back during the reset are dropped: they
show no block disappearing, or several, and either way say nothing.

    uv run scripts/fit_jaw_offset.py
    uv run scripts/fit_jaw_offset.py --show
"""

import argparse
import json
from pathlib import Path

import numpy as np

from so101.policy import ArmKinematics, BlockDetector, TableFrame
from so101.policy.approach import grasp_frame

OFFSET_PATH = Path("data/jaw_offset.json")
SAME_BLOCK_MM = 25.0     # nearer than this at both starts is the same block
MIN_CONFIDENCE = 0.5


def show_stored(path):
    if not path.is_file():
        print(f"  {path} not found; nothing measured yet")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    offset = np.array(data["offset_m"])
    print(f"  offset  x={1000*offset[0]:+.1f}  y={1000*offset[1]:+.1f}  "
          f"z={1000*offset[2]:+.1f} mm   from {data['samples']} episode(s)")
    print(f"  spread  {data['spread_mm']:.1f} mm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=Path, default=Path("data/demos"))
    parser.add_argument("--repo-id", default="so101/grasp_block")
    parser.add_argument("--camera", default="side")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OFFSET_PATH)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.show:
        return show_stored(args.out)

    import pandas as pd
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    table = TableFrame.load()
    arm = ArmKinematics()
    detector = BlockDetector(weights=args.weights, table_frame=table)
    detector.warmup()
    print(f"  detector: {detector.weights}")
    print(f"  {table}")

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.demos)
    dataset = LeRobotDataset(args.repo_id, root=args.demos)
    names = [name.removesuffix(".pos") for name in meta.features["action"]["names"]]
    gripper = names.index("gripper")
    camera = f"observation.images.{args.camera}"

    frames = pd.concat([pd.read_parquet(path)
                        for path in sorted(Path(args.demos, "data").rglob("*.parquet"))])
    frames = frames.sort_values(["episode_index", "frame_index"])

    def table_at_start(episode):
        """What the side camera sees at the start of an episode."""
        index = int(meta.episodes["dataset_from_index"][episode])
        image = dataset[index][camera].numpy()
        # The dataset holds RGB; the detector was trained on what cv2
        # reads, which is BGR. Handing it RGB swaps red for blue and
        # every colour label with it.
        image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, ::-1]
        return [(d.colour, d.position) for d in detector.detect(image)
                if d.confidence >= MIN_CONFIDENCE and d.position is not None]

    grasps = {}
    for episode, rows in frames.groupby("episode_index"):
        states = np.stack(rows["observation.state"].to_numpy())
        tips = np.array([arm.forward({name: float(state[i])
                                      for i, name in enumerate(names)
                                      if name in arm.joint_names})
                         for state in states])
        held = states[grasp_frame(states[:, gripper], tips[:, 2], meta.fps)]
        grasps[int(episode)] = arm.forward(
            {name: float(held[index]) for index, name in enumerate(names)
             if name in arm.joint_names})

    print(f"\n  {'episode':>8}{'grasped':>10}{'offset x':>11}{'y':>8}{'z':>8}")
    offsets = []
    previous = table_at_start(0)
    for episode in range(1, meta.total_episodes):
        current = table_at_start(episode)
        vanished = []
        for colour, position in previous:
            same = [other for other_colour, other in current
                    if other_colour == colour
                    and 1000 * np.linalg.norm(position - other) < SAME_BLOCK_MM]
            if not same:
                vanished.append((colour, position))
        previous = current

        if len(vanished) != 1:
            print(f"  {episode - 1:>8}{len(vanished):>10} - skipped")
            continue
        colour, position = vanished[0]
        offset = grasps[episode - 1] - position
        offsets.append(offset)
        print(f"  {episode - 1:>8}{colour:>10}{1000*offset[0]:>10.0f}mm"
              f"{1000*offset[1]:>7.0f}{1000*offset[2]:>8.0f}")

    if len(offsets) < 3:
        raise SystemExit("\n  Too few episodes showed exactly one block "
                         "disappear; nothing to fit.")

    offsets = np.array(offsets)
    # The median, not the mean: a wrongly matched block puts one sample a long
    # way out and there are not many samples to dilute it.
    offset = np.median(offsets, axis=0)
    spread = float(np.median(np.linalg.norm(offsets - offset, axis=1))) * 1000

    print(f"\n  {len(offsets)} usable episode(s)")
    print(f"  jaws close at  x={1000*offset[0]:+.0f}  y={1000*offset[1]:+.0f}  "
          f"z={1000*offset[2]:+.0f} mm  from the gripper frame")
    print(f"  spread {spread:.0f} mm")
    if spread > 25:
        print("\n  That is too scattered to be one constant. Either the side "
              "camera has\n  moved since these were recorded, or blocks were "
              "put back between\n  episodes and the wrong ones are being matched.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "offset_m": offset.tolist(),
        "samples": len(offsets),
        "spread_mm": spread,
        "camera": args.camera,
    }, indent=2), encoding="utf-8")
    print(f"\n  saved {args.out}")


if __name__ == "__main__":
    main()
