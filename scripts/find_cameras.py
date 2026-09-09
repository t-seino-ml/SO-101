"""List the attached cameras and save one preview frame from each.

OpenCV indices are not stable across replugs, so identify a camera by the name
shown here and pass that name to the rest of the stack.

    uv run scripts/find_cameras.py                  # list and save previews
    uv run scripts/find_cameras.py --no-preview     # list only, much faster
    uv run scripts/find_cameras.py --pairs          # which cameras can run together
    uv run scripts/find_cameras.py --out shots/
"""

from so101.platform import require_windows

require_windows()

import argparse
from pathlib import Path

import cv2

from so101.camera.discovery import BACKEND, backend_name, device_names, probe

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "outputs" / "camera_previews"


def safe_name(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def check_pairs(names):
    """Report which cameras can be open at the same time.

    Two USB 2.0 UVC cameras on one host controller usually cannot coexist: the
    isochronous endpoint reserves bandwidth from the camera's declared maximum
    packet size, not from the compressed data rate, so switching to MJPG does not
    help. The fix is to move one camera to a port on a different controller.
    """
    print()
    print("Which cameras can run at the same time:")
    for first in range(len(names)):
        primary = cv2.VideoCapture(first, BACKEND)
        if not primary.isOpened():
            primary.release()
            print(f"  [{first}] {names[first]}: cannot open at all")
            continue
        works, blocked = [], []
        for second in range(len(names)):
            if second == first:
                continue
            other = cv2.VideoCapture(second, BACKEND)
            (works if other.isOpened() else blocked).append(second)
            other.release()
        primary.release()
        ok = ", ".join(f"[{i}]" for i in works) or "none"
        no = ", ".join(f"[{i}]" for i in blocked) or "none"
        print(f"  [{first}] {names[first]}")
        print(f"        with: {ok}")
        print(f"     blocked: {no}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="directory for the preview images")
    parser.add_argument("--no-preview", action="store_true",
                        help="only list cameras, do not open them")
    parser.add_argument("--pairs", action="store_true",
                        help="test which cameras can be open simultaneously")
    args = parser.parse_args()

    names = device_names()
    print(f"Backend: {backend_name()}")
    if not names:
        print("No device names available; falling back to index probing.")

    if args.pairs:
        return check_pairs(names or [f"index {i}" for i in range(10)])

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
