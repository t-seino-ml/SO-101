"""The exhibition: pick the block of a colour out of its slot and drop it in the can.

Three ways in, and they exist so that a failure in one part does not take the
whole demonstration down in front of an audience.

  Level 1   --color blue            the camera finds the blue block, works out
                                    which slot it is in, and the arm goes there
  Level 2   --color blue --slot 3   the camera confirms blue is visible; a
                                    person says which slot. Survives the slot
                                    calibration being wrong
  Level 3   --slot 3                no camera at all. Survives anything

Level 3 is the one that must always work, so it is the one with the fewest
moving parts: read a taught file, check it against the servos' limits, replay it
while watching the load. Nothing is solved for, nothing is detected, nothing is
learnt.

    uv run scripts/demo_slot_pick.py --slot 1 --dry-run   # plan only, no motion
    uv run scripts/demo_slot_pick.py --slot 1 --step      # confirm each waypoint
    uv run scripts/demo_slot_pick.py --color blue

Every run writes outputs/demo/<timestamp>/ with what it saw, what it chose, and
every waypoint it reached.
"""

from so101.platform import require_windows

require_windows()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from so101.demo.slots import SlotMap, steady_centre  # noqa: E402
from so101.demo.trajectory import (  # noqa: E402
    ARM_JOINTS,
    SLOT_DIR,
    Trajectory,
    Unsafe,
    freeze,
    joints_of,
    play,
    read_limits,
)

SHORT = {"shoulder_pan": "pan", "shoulder_lift": "lift",
         "elbow_flex": "elbow", "wrist_flex": "wflex", "wrist_roll": "wroll"}

OUT_ROOT = Path("outputs/demo")
COLOURS = ("red", "orange", "yellow", "green", "blue", "purple")


class Log:
    """Print, and keep a copy beside the run's own results."""

    def __init__(self, path=None):
        self.file = None if path is None else open(path, "w", encoding="utf-8")

    def __call__(self, line=""):
        print(line)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def look(args, log, record):
    """What the camera says. Returns (slot number or None, refusal or None)."""
    from so101.camera import CameraSet
    from so101.policy import BlockDetector

    slot_map = None
    if args.slot is None:
        slot_map = SlotMap.load(args.slots_file)
        log(f"  {slot_map}")

    cameras = CameraSet.from_config()
    cameras.start()
    try:
        cameras.wait_for_frames(timeout=25)
        stream = cameras.streams[args.camera]
        detector = BlockDetector(weights=args.weights, table_frame=None)
        detector.warmup()

        centre, detail = steady_centre(detector, stream, args.color,
                                       frames=args.frames,
                                       confidence=args.confidence, log=log)
        record["detection"] = detail
        image = stream.read().image.copy()
        record["_image"] = image
        log(f"  {args.color}: {detail['seen_in']}/{detail['frames']} フレームで検出"
            + (f"、中心 ({detail['centre_px'][0]:.0f}, "
               f"{detail['centre_px'][1]:.0f})、ばらつき "
               f"{detail['spread_px']:.0f} px、確信度 {detail['confidence']:.2f}"
               if centre else ""))
        if centre is None:
            return None, detail["why"]
        if args.slot is not None:
            log(f"  Slot は人が指定: slot{args.slot}（Level 2）")
            return args.slot, None

        if slot_map.ambiguous(centre):
            order = slot_map.ranked(centre)
            return None, (f"ブロックが {order[0][0]} と {order[1][0]} の"
                          f"ちょうど中間にあります"
                          f"（{order[0][1]:.0f} px と {order[1][1]:.0f} px）")
        name, distance = slot_map.nearest(centre)
        record["slot_distances_px"] = {n: round(d, 1)
                                       for n, d in slot_map.ranked(centre)}
        if name is None:
            return None, (f"いちばん近い Slot まで {distance:.0f} px あります"
                          f"（受け入れ半径 {slot_map.acceptance_radius_px:.0f} px）。"
                          f"ブロックが Slot の上にありません")
        log(f"  → {name}（中心から {distance:.0f} px）")
        return slot_map.number(name), None
    finally:
        cameras.stop()


def main():
    parser = argparse.ArgumentParser(
        description="色を指定してブロックを取り、固定位置の缶へ入れます")
    parser.add_argument("--color", "--colour", dest="color", choices=COLOURS,
                        help="取る色。省略すると Slot 直接指定（Level 3）")
    parser.add_argument("--slot", type=int,
                        help="Slot を人が指定します。--color と併用で Level 2")
    parser.add_argument("--dry-run", action="store_true",
                        help="見て、選んで、計画を表示するだけ。動かしません")
    parser.add_argument("--step", action="store_true",
                        help="waypoint ごとに ENTER を待ちます（最初の数回向け）")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="教示したときの速さに対する倍率。1 未満で遅くなります")
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--camera", default="side")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--trajectories", type=Path, default=SLOT_DIR)
    parser.add_argument("--slots-file", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    if args.color is None and args.slot is None:
        raise SystemExit(
            "\n  --color か --slot のどちらかは必要です。\n"
            "    --slot 1                Level 3（カメラを使いません）\n"
            "    --color blue --slot 1   Level 2（色だけ確認）\n"
            "    --color blue            Level 1（Slot も自動）\n")
    level = 3 if args.color is None else (2 if args.slot is not None else 1)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = args.color or f"slot{args.slot}"
    out_dir = args.out / f"{stamp}_L{level}_{label}"
    record = {
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "level": level, "requested_colour": args.color,
        "requested_slot": args.slot, "speed": args.speed,
        "outcome": "not started",
    }
    log = Log()
    log(f"\n  展示デモ  Level {level}"
        + (f"  色 {args.color}" if args.color else "")
        + (f"  Slot {args.slot}" if args.slot else ""))

    # -- what to do --------------------------------------------------------
    slot = args.slot
    if args.color is not None:
        slot, refusal = look(args, log, record)
        if refusal is not None:
            log(f"\n  *** 動かしません: {refusal} ***")
            record.update({"outcome": "refused", "why": refusal})
            _write(out_dir, record, log)
            raise SystemExit(1)
    record["slot"] = slot

    path = args.trajectories / f"slot{slot}.json"
    trajectory = Trajectory.load(path)
    log(f"\n  {trajectory}")
    log(f"  教示 {trajectory.created}")
    record["trajectory"] = {"file": str(path), "name": trajectory.name,
                            "created": trajectory.created,
                            "phases": trajectory.phases()}

    from so101.hardware import resolve as resolve_port
    port = resolve_port([args.follower_port])[0]

    log("\n  --- 教示済み軌道を、いまのサーボの限界と突き合わせます ---")
    limits = read_limits(port)
    here = {name: limits[name]["deg"] for name in ARM_JOINTS}
    log("  いまの姿勢: " + "、".join(f"{n} {here[n]:+.1f}" for n in ARM_JOINTS))
    trajectory.check(limits, start=here, log=log)
    record["start_pose_deg"] = {n: round(v, 2) for n, v in here.items()}

    if args.dry_run:
        log("\n  --- 実行する waypoint ---")
        log(f"  {'phase':<12}" + "".join(f"{SHORT[n]:>9}" for n in ARM_JOINTS)
            + f"{'grip':>8}{'秒':>6}")
        for point in trajectory.waypoints:
            log(f"  {point.phase:<12}"
                + "".join(f"{point.joints[n]:>+9.1f}" for n in ARM_JOINTS)
                + (f"{point.gripper:>+8.1f}" if point.gripper is not None
                   else f"{'-':>8}")
                + f"{point.seconds / max(args.speed, 1e-3):>6.1f}")
        log("\n  Dry run です。何も動かしていません。\n")
        record["outcome"] = "dry run"
        _write(out_dir, record, log)
        return

    # -- moving ------------------------------------------------------------
    log("\n  --- 接続する前に ---")
    log("    robot.connect() でトルクが入ります。接続＝始動です。")
    log("    アームの可動範囲に人も物もないこと。")
    log("    緊急停止: Ctrl-C → 別ターミナルで "
        "uv run scripts/torque_off.py COM4 → 電源")
    if input("\n  始めるなら ready と入力（それ以外は中止）: ").strip().lower() \
            != "ready":
        log("  接続前に中止しました。何も動かしていません。")
        record["outcome"] = "declined"
        _write(out_dir, record, log)
        return

    from so101.hardware import bus_patch  # noqa: F401
    from so101.hardware import tuning
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    tuning.install(verbose=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.close()
    log = Log(out_dir / "console.log")

    robot = None
    confirm = None
    if args.step:
        def confirm(point):
            return not input(f"      {point.phase} へ進むなら ENTER: ").strip()

    try:
        robot = make_robot_from_config(SO101FollowerConfig(
            port=port, id="follower"))
        robot.connect()
        log("\n  接続しました。トルクが入っています。")
        began = time.perf_counter()
        record["waypoints"] = play(robot, trajectory, log=log,
                                   speed=args.speed, confirm=confirm,
                                   limits=limits)
        record["seconds"] = round(time.perf_counter() - began, 1)
        record["outcome"] = "done"
        log(f"\n  完了しました（{record['seconds']:.1f} 秒）")
    except Unsafe as error:
        record.update({"outcome": "aborted", "why": str(error)})
        log(f"\n  *** 中断: {error} ***")
        freeze(robot, log)
        log("  アームは保持しています。見てから次を決めてください。")
    except KeyboardInterrupt:
        record.update({"outcome": "aborted", "why": "操作者が中止しました"})
        log("\n  操作者が中止しました")
        freeze(robot, log)
    except Exception as error:  # noqa: BLE001
        record.update({"outcome": "error",
                       "why": f"{type(error).__name__}: {error}"})
        log(f"\n  *** {record['why']} ***")
        freeze(robot, log)
    finally:
        if robot is not None:
            try:
                robot.disconnect()
                log("  切断し、トルクを解放しました")
            except Exception as error:  # noqa: BLE001
                log(f"  切断できません: {error}")
                log("  uv run scripts/torque_off.py COM4")
        _write(out_dir, record, log)
        log.close()
    if record["outcome"] not in ("done", "dry run"):
        raise SystemExit(1)


def _write(out_dir, record, log):
    out_dir.mkdir(parents=True, exist_ok=True)
    image = record.pop("_image", None)
    if image is not None:
        try:
            import cv2

            cv2.imwrite(str(out_dir / "seen.png"), image)
        except Exception:  # noqa: BLE001
            pass
    (out_dir / "run.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    log(f"  記録: {out_dir}")


if __name__ == "__main__":
    main()
