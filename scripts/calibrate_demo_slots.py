"""Click where the slots are. The only calibration the exhibition needs.

Tomorrow the camera goes somewhere else. Nothing else moves - the robot, the
slots and the can keep the positions they are marked at - so the taught
trajectories still work and the only thing the camera has to be told again is
which patch of image is which slot.

That is three clicks. No target, no chequerboard, no measuring, no arm.

    uv run scripts/calibrate_demo_slots.py
    uv run scripts/calibrate_demo_slots.py --slots 3 --radius 90
    uv run scripts/calibrate_demo_slots.py --show      # look at the saved one

Left-click each slot centre in order. BACKSPACE undoes the last one, ENTER
saves, ESC quits without saving. The circle drawn round each click is the
acceptance radius: a block detected outside every circle stops the run rather
than being assigned to the nearest.
"""

from so101.platform import require_windows

require_windows()

import argparse  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from so101.demo.slots import CALIBRATION_PATH, DEFAULT_RADIUS_PX, SlotMap  # noqa: E402

WINDOW = "slots - click each centre, ENTER to save, BACKSPACE to undo, ESC to quit"
COLOURS = [(80, 200, 255), (120, 255, 120), (255, 160, 80),
           (255, 120, 255), (120, 220, 255), (200, 200, 120)]


def draw(image, points, wanted, radius, detections=None):
    import cv2

    canvas = image.copy()
    for index, (u, v) in enumerate(points):
        colour = COLOURS[index % len(COLOURS)]
        cv2.circle(canvas, (int(u), int(v)), int(radius), colour, 2)
        cv2.drawMarker(canvas, (int(u), int(v)), colour,
                       cv2.MARKER_CROSS, 18, 2)
        cv2.putText(canvas, f"slot{index + 1}", (int(u) + 10, int(v) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    for detection in detections or []:
        u, v = detection.pixel
        cv2.drawMarker(canvas, (int(u), int(v)), (255, 255, 255),
                       cv2.MARKER_TILTED_CROSS, 14, 1)
        cv2.putText(canvas, detection.colour, (int(u) + 8, int(v) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    remaining = wanted - len(points)
    banner = (f"slot{len(points) + 1} をクリック  (あと {remaining})"
              if remaining > 0 else "ENTER で保存")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="カメラ画像上の Slot 中心をクリックして登録します")
    parser.add_argument("--slots", type=int, default=3, help="Slot の数")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_PX,
                        help="この画素数まで離れていても同じ Slot とみなす")
    parser.add_argument("--camera", default="side")
    parser.add_argument("--detect", action="store_true",
                        help="いま見えているブロックも重ねて表示します")
    parser.add_argument("--show", action="store_true",
                        help="保存済みの内容を表示するだけ")
    parser.add_argument("--out", type=Path, default=CALIBRATION_PATH)
    args = parser.parse_args()

    if args.show:
        print(f"\n  {SlotMap.load(args.out)}\n")
        return

    import cv2
    from so101.camera import CameraSet

    cameras = CameraSet.from_config()
    cameras.start()
    points = []
    try:
        cameras.wait_for_frames(timeout=25)
        stream = cameras.streams[args.camera]
        frame = stream.read()
        image = frame.image.copy()
        height, width = image.shape[:2]
        print(f"\n  {args.camera} カメラ {width}x{height}")

        detections = None
        if args.detect:
            from so101.policy import BlockDetector
            detector = BlockDetector(table_frame=None)
            detector.warmup((height, width))
            detections = detector.detect(image)
            print(f"  いま見えているブロック: "
                  + "、".join(f"{d.colour}({d.pixel[0]:.0f},{d.pixel[1]:.0f})"
                              for d in detections))

        print(f"  Slot を {args.slots} 個、左から順にクリックしてください。")
        print("  BACKSPACE で取り消し、ENTER で保存、ESC で中止。\n")

        def on_mouse(event, x, y, flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < args.slots:
                points.append((float(x), float(y)))
                print(f"    slot{len(points)} = ({x}, {y})")

        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, on_mouse)
        while True:
            # A live frame, so the slots can be clicked while the blocks are
            # being put out rather than against a snapshot of an empty table.
            image = stream.read().image.copy()
            cv2.imshow(WINDOW, draw(image, points, args.slots, args.radius,
                                    detections))
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):
                if len(points) < args.slots:
                    print(f"    あと {args.slots - len(points)} 個です")
                    continue
                break
            if key == 8 and points:
                print(f"    slot{len(points)} を取り消しました")
                points.pop()
            if key == 27:
                print("\n  中止しました。保存していません。")
                return
    finally:
        cameras.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass

    slot_map = SlotMap(
        slots={f"slot{index + 1}": point for index, point in enumerate(points)},
        camera=args.camera, resolution=(width, height),
        acceptance_radius_px=args.radius,
        created=datetime.now().astimezone().isoformat(timespec="seconds"),
        note="clicked on a live frame")

    # How close together they are decides whether "nearest" can be trusted.
    names = list(slot_map.slots)
    closest = min(
        (math_dist(slot_map.slots[a], slot_map.slots[b]), a, b)
        for index, a in enumerate(names) for b in names[index + 1:])
    print(f"\n  いちばん近い 2 つ: {closest[1]} と {closest[2]} が "
          f"{closest[0]:.0f} px")
    if closest[0] < 2 * args.radius:
        print(f"  *** 受け入れ半径 {args.radius:.0f} px の円が重なります。")
        print(f"      Slot をもっと離すか、--radius を {closest[0]/2:.0f} 未満に "
              f"してください ***")

    if args.out.is_file():
        backup = args.out.with_name(
            f"{args.out.stem}.{datetime.now():%Y%m%d-%H%M%S}.json")
        backup.write_bytes(args.out.read_bytes())
        print(f"  以前の内容を {backup} へ退避しました")
    slot_map.save(args.out)
    print(f"  保存しました: {args.out}")
    print(f"  {slot_map}")


def math_dist(a, b):
    import math

    return math.dist(a, b)


if __name__ == "__main__":
    main()
