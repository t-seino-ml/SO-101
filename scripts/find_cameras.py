"""List the attached cameras and save one preview frame from each.

OpenCV indices are not stable across replugs, so identify a camera by the name
shown here and pass that name to the rest of the stack.

    uv run scripts/find_cameras.py                  # list and save previews
    uv run scripts/find_cameras.py --no-preview     # list only, much faster
    uv run scripts/find_cameras.py --out shots/
"""

import argparse
from pathlib import Path

import cv2

from so101.camera.discovery import backend_name, device_names, probe

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "outputs" / "camera_previews"


def safe_name(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="directory for the preview images")
    parser.add_argument("--no-preview", action="store_true",
                        help="only list cameras, do not open them")
    args = parser.parse_args()

    names = device_names()
    print(f"Backend: {backend_name()}")
    if not names:
        print("No device names available; falling back to index probing.")

    limit = len(names) if names else 10
    if args.no_preview:
        for index in range(limit):
            print(f"[{index}] {names[index] if index < len(names) else '?'}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    working = 0
    for index in range(limit):
        info, frame = probe(index)
        print(info)
        if frame is None:
            continue
        working += 1
        path = args.out / f"{index}_{safe_name(info.name)}.png"
        cv2.imwrite(str(path), frame)
        print(f"      saved {path}")

    print(f"\n{working}/{limit} camera(s) delivered a frame.")
    if working:
        print(f"Previews in {args.out} - open them to see which camera is which.")


if __name__ == "__main__":
    main()
