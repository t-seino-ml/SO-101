"""Generate a YOLO detection dataset from the block crops.

Every image is synthetic: the blocks come from the crops in data/blocks, and the
backgrounds, lighting and camera artefacts are sampled fresh each time. Real
camera frames are mixed in when data/backgrounds has any - capture them with
scripts/capture_backgrounds.py.

    uv run scripts/build_dataset.py                    # 4000 train, 400 val
    uv run scripts/build_dataset.py --train 8000 --val 800
    uv run scripts/build_dataset.py --preview 12       # sample images, no dataset
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from so101.dataset.scenes import (
    CLASS_NAMES,
    SCENE_SIZE,
    load_backgrounds,
    load_crops,
    make_scene,
)

DEFAULT_OUT = Path("data/yolo")


def write_split(out_dir, split, count, crops, backgrounds, rng, jpeg_quality=92):
    images = out_dir / "images" / split
    labels = out_dir / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    boxes_written = 0
    for index in range(count):
        scene, boxes = make_scene(crops, rng, SCENE_SIZE, backgrounds)
        stem = f"{split}_{index:06d}"
        cv2.imwrite(str(images / f"{stem}.jpg"), scene,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                 for cls, cx, cy, w, h in boxes]
        (labels / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        boxes_written += len(boxes)
        if count >= 200 and (index + 1) % (count // 10) == 0:
            print(f"    {split}: {index + 1}/{count}", flush=True)
    return boxes_written


def write_yaml(out_dir):
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    text = (f"path: {out_dir.resolve().as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n" + names + "\n")
    path = out_dir / "dataset.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def save_preview(count, crops, backgrounds, rng, out_path):
    """A labelled contact sheet, so the synthesis can be eyeballed before training."""
    cells = []
    for _ in range(count):
        scene, boxes = make_scene(crops, rng, SCENE_SIZE, backgrounds)
        for cls, cx, cy, w, h in boxes:
            x0 = int((cx - w / 2) * SCENE_SIZE[0])
            y0 = int((cy - h / 2) * SCENE_SIZE[1])
            x1 = int((cx + w / 2) * SCENE_SIZE[0])
            y1 = int((cy + h / 2) * SCENE_SIZE[1])
            cv2.rectangle(scene, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(scene, CLASS_NAMES[cls], (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cells.append(cv2.resize(scene, (SCENE_SIZE[0] // 2, SCENE_SIZE[1] // 2)))
    columns = 3
    rows = [np.hstack(cells[i:i + columns]) for i in range(0, len(cells), columns)
            if len(cells[i:i + columns]) == columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--blocks", default="data/blocks")
    parser.add_argument("--backgrounds", default="data/backgrounds")
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--val", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=int, default=0,
                        help="write a labelled contact sheet and stop")
    args = parser.parse_args()

    crops = load_crops(args.blocks)
    counts = {name: len(images) for name, images in crops.items()}
    if not sum(counts.values()):
        raise SystemExit(f"No crops in {args.blocks}. "
                         "Run scripts/extract_blocks.py first.")
    backgrounds = load_backgrounds(args.backgrounds)
    print(f"  crops: {counts}")
    print(f"  real backgrounds: {len(backgrounds)}"
          + ("" if backgrounds else "  (procedural only - run "
                                    "scripts/capture_backgrounds.py to add some)"))

    rng = np.random.default_rng(args.seed)

    if args.preview:
        path = save_preview(args.preview, crops, backgrounds, rng,
                            Path("outputs/dataset_preview.png"))
        print(f"  preview -> {path}")
        return

    if args.out.exists():
        shutil.rmtree(args.out)
    train_boxes = write_split(args.out, "train", args.train, crops, backgrounds, rng)
    val_boxes = write_split(args.out, "val", args.val, crops, backgrounds, rng)
    yaml_path = write_yaml(args.out)

    print(f"\n  train {args.train} images, {train_boxes} boxes "
          f"({train_boxes / max(1, args.train):.1f} per image)")
    print(f"  val   {args.val} images, {val_boxes} boxes")
    print(f"  {yaml_path}")


if __name__ == "__main__":
    main()
