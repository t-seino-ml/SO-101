"""Check the detector against real camera frames, and time it.

Validation accuracy on the synthetic set only says the model fits its own
generator. What matters is whether it finds real blocks on a real table, so this
runs live against the cameras, draws what it sees, and reports the inference rate.

Put blocks on the table before running it.

    uv run scripts/eval_detector.py                       # live, both cameras
    uv run scripts/eval_detector.py --seconds 20 --save
    uv run scripts/eval_detector.py --images outputs/vision_test
"""

import argparse
import statistics
import time
from pathlib import Path

from so101.platform import require_windows

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from so101.dataset.scenes import CLASS_NAMES  # noqa: E402

BOX_COLOURS = {
    "red": (60, 60, 220), "orange": (60, 150, 240), "yellow": (60, 220, 240),
    "green": (80, 200, 80), "blue": (220, 120, 60), "purple": (200, 80, 140),
}
DEFAULT_WEIGHTS = Path("runs/detect/blocks/weights/best.pt")


def draw(frame, result, conf_threshold):
    counts = {}
    for box in result.boxes:
        confidence = float(box.conf)
        if confidence < conf_threshold:
            continue
        name = CLASS_NAMES[int(box.cls)]
        counts[name] = counts.get(name, 0) + 1
        x0, y0, x1, y1 = (int(v) for v in box.xyxy[0])
        colour = BOX_COLOURS[name]
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(frame, f"{name} {confidence:.2f}", (x0, max(12, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    return counts


def summarise(counts):
    return ", ".join(f"{name}x{n}" for name, n in sorted(counts.items())) or "nothing"


def run_on_images(model, directory, conf, out_dir):
    paths = [p for p in sorted(Path(directory).glob("*.png"))
             if "detect" not in p.name]
    if not paths:
        raise SystemExit(f"No .png files in {directory}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        frame = cv2.imread(str(path))
        result = model(frame, verbose=False)[0]
        counts = draw(frame, result, conf)
        cv2.imwrite(str(out_dir / f"detect_{path.name}"), frame)
        print(f"  {path.name}: {summarise(counts)}")


def run_live(model, conf, seconds, save, out_dir):
    require_windows("eval_detector.py --live")
    from so101.camera import CameraSet

    latencies = []
    with CameraSet.from_config() as cameras:
        cameras.wait_for_frames(timeout=25)
        deadline = time.perf_counter() + seconds
        last_report = 0.0
        frames = {}
        while time.perf_counter() < deadline:
            for role, frame in cameras.read().items():
                if frame is None:
                    continue
                image = frame.image.copy()
                started = time.perf_counter()
                result = model(image, verbose=False)[0]
                latencies.append((time.perf_counter() - started) * 1000)
                counts = draw(image, result, conf)
                frames[role] = (image, counts)
            now = time.perf_counter()
            if now - last_report > 2.0:
                for role, (_, counts) in frames.items():
                    print(f"  {role}: {summarise(counts)}")
                print()
                last_report = now

        if save:
            out_dir.mkdir(parents=True, exist_ok=True)
            for role, (image, _) in frames.items():
                path = out_dir / f"live_{role}.png"
                cv2.imwrite(str(path), image)
                print(f"  saved {path}")

    if latencies:
        ordered = sorted(latencies)
        print(f"\n  inference over {len(ordered)} frames:")
        print(f"    median {statistics.median(ordered):5.1f} ms   "
              f"p95 {ordered[int(len(ordered) * 0.95)]:5.1f} ms   "
              f"max {ordered[-1]:5.1f} ms")
        print(f"    -> {1000 / statistics.median(ordered):.0f} inferences/s "
              f"against a 20 fps camera")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--images", default=None,
                        help="run on a directory of stills instead of the cameras")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/detections"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"{args.weights} not found. "
                         "Train first: uv run scripts/train_detector.py")

    import torch
    from ultralytics import YOLO

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.weights))
    model.to(device if device == "cpu" else f"cuda:{device}")
    print(f"  {args.weights} on {device}, confidence >= {args.conf}\n")

    # One warm-up: the first inference pays for CUDA context and graph setup.
    model(np.zeros((600, 800, 3), np.uint8), verbose=False)

    if args.images:
        run_on_images(model, args.images, args.conf, args.out)
    else:
        run_live(model, args.conf, args.seconds, args.save, args.out)


if __name__ == "__main__":
    main()
