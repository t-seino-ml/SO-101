"""Capture real, block-free camera frames to mix into the synthetic dataset.

Procedural backgrounds cover a wide range of colours and textures, but contain
none of what the real scene actually holds: the white arm, the metal can, the
table edge, cables, the floor beyond it. A detector that has never seen those is
free to call them blocks. Feeding them in as unlabelled backgrounds teaches the
opposite.

Clear every block off the table first. Anything left in frame becomes an
unlabelled negative, so a stray block would train the detector to ignore blocks.

By default the follower arm sweeps itself through random reachable poses while the
frames are taken, so the arm appears in many configurations rather than one. Pass
--no-move-arm to leave it limp and pose it by hand instead.

    uv run scripts/capture_backgrounds.py                  # 40 frames, arm sweeping
    uv run scripts/capture_backgrounds.py --no-move-arm
    uv run scripts/capture_backgrounds.py --frames 80 --seconds 80
"""

import argparse
import time
from pathlib import Path

from so101.platform import require_windows

require_windows()

import cv2  # noqa: E402

from so101.camera import CameraSet  # noqa: E402
from so101.dataset.poses import PoseSweeper  # noqa: E402
from so101.hardware import resolve as resolve_port  # noqa: E402


def save_frames(cameras, out_dir, index):
    written = 0
    for role, frame in cameras.read().items():
        if frame is None:
            continue
        cv2.imwrite(str(out_dir / f"{role}_{index:04d}.png"), frame.image)
        written += 1
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/backgrounds"))
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--no-move-arm", action="store_true",
                        help="leave the arm limp; pose it by hand instead")
    parser.add_argument("--append", action="store_true",
                        help="keep frames already in the output directory")
    args = parser.parse_args()

    if args.out.exists() and not args.append:
        for existing in args.out.glob("*.png"):
            existing.unlink()
    args.out.mkdir(parents=True, exist_ok=True)
    start_index = len(list(args.out.glob("*.png")))

    print("Clear ALL blocks off the table before this starts.")
    print(f"Capturing {args.frames} frames per camera over {args.seconds:.0f}s.")
    if args.no_move_arm:
        print("The arm stays limp - move it by hand while this runs.\n")
    else:
        print("The arm will move itself. Keep the area clear.\n")

    with CameraSet.from_config() as cameras:
        cameras.wait_for_frames(timeout=25)
        for role, stream in cameras.streams.items():
            fourcc, width, height, _ = stream.format
            print(f"  {role}: {fourcc} {width}x{height}")
        print()

        state = {"written": 0, "index": start_index, "next": time.perf_counter()}
        interval = args.seconds / max(1, args.frames)
        limit = start_index + args.frames

        def maybe_capture():
            now = time.perf_counter()
            if now < state["next"] or state["index"] >= limit:
                return
            state["written"] += save_frames(cameras, args.out, state["index"])
            state["index"] += 1
            state["next"] = now + interval
            taken = state["index"] - start_index
            if taken % max(1, args.frames // 10) == 0:
                print(f"  {taken}/{args.frames}", flush=True)

        if args.no_move_arm:
            while state["index"] < limit:
                maybe_capture()
                time.sleep(0.02)
        else:
            port = resolve_port([args.follower_port])[0]
            with PoseSweeper(port) as sweeper:
                poses = sweeper.sweep(args.seconds, on_step=maybe_capture)
            print(f"\n  arm visited {poses} poses")
            # A pose sweep that aborted early leaves frames uncaptured.
            while state["index"] < limit:
                maybe_capture()
                time.sleep(0.02)

    print(f"\n{state['written']} frames -> {args.out}")
    print("Now rebuild the dataset: uv run scripts/build_dataset.py")


if __name__ == "__main__":
    main()
