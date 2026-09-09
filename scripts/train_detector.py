"""Train the block detector on the synthetic dataset.

Every training image is synthetic, so the thing to watch is not the validation
score - that measures how well the model fits its own generator - but how it does
on real camera frames. Use scripts/eval_detector.py for that.

Ultralytics applies its own augmentation on top of what the generator already
varies. HSV augmentation is dialled down here: colour *is* the class, and shifting
hue far enough turns a red block into an orange one, which is label noise rather
than robustness. Geometry and scale augmentation stay on.

    uv run scripts/train_detector.py                     # yolo11n, 100 epochs
    uv run scripts/train_detector.py --model yolo11s.pt --epochs 150
    uv run scripts/train_detector.py --device cpu
"""

import argparse
from pathlib import Path

DEFAULT_DATA = Path("data/yolo/dataset.yaml")
DEFAULT_PROJECT = Path("runs")  # Ultralytics adds "detect/" itself


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None,
                        help="cuda device index, or 'cpu'. Default: GPU if present")
    parser.add_argument("--name", default="blocks")
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"{args.data} not found. Run scripts/build_dataset.py first.")

    import torch
    from ultralytics import YOLO

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Training on CPU. This will be slow.")
    else:
        print(f"Training on {torch.cuda.get_device_name(int(device))}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(DEFAULT_PROJECT),
        name=args.name,
        patience=args.patience,
        # The generator already varies lighting and white balance; leave the class
        # signal alone. hsv_h in particular walks red towards orange.
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=180,     # blocks sit at any orientation
        translate=0.15,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        erasing=0.2,
        plots=True,
    )

    best = DEFAULT_PROJECT / "detect" / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    print("Now check it against real frames: uv run scripts/eval_detector.py")


if __name__ == "__main__":
    main()
