"""Teach one slot's pick by driving the arm with the leader, and save it.

The exhibition replays joint angles a person taught and watched work. This is
where they are taught: drive the follower with the leader arm, put it exactly
where it needs to be, press ENTER, and that pose becomes a waypoint. Nothing is
solved for and nothing is learnt.

The phases are prompted in order, and the order is the shape of a pick:

    HOME       where the arm rests between runs, and returns to
    PREGRASP   above the slot, jaws open, clear of the block
    GRASP      down on the block, jaws still open, straddling it
    CLOSE      the same pose with the jaws shut - taught, not assumed, because
               where the jaws stop on a block is a fact about this gripper
    LIFT       straight up, holding
    TRANSFER   somewhere safe in between, clear of everything
    CAN_ABOVE  over the can, still holding
    DROP       the jaws open - the block falls
    RETURN     back to HOME

    uv run scripts/teach_demo_slot.py --slot 1
    uv run scripts/teach_demo_slot.py --slot 1 --review     # replay, teach nothing
    uv run scripts/teach_demo_slot.py --can                 # the shared can part

*** The leader arm drives the follower the moment this connects. *** Hold the
leader before pressing ENTER at the first prompt, or the follower will jump to
wherever the leader happens to be lying.
"""

from so101.platform import require_windows

require_windows()

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from so101.demo.trajectory import (  # noqa: E402
    ARM_JOINTS,
    SLOT_DIR,
    Trajectory,
    Unsafe,
    Waypoint,
    health,
    joints_of,
    read_limits,
)

#: What to ask for, in order, and what the jaws should be doing at each.
#: `gripper` None means "leave the jaws wherever the leader has them", which is
#: right for the poses where the operator is setting the jaws by hand anyway.
SHORT = {"shoulder_pan": "pan", "shoulder_lift": "lift",
         "elbow_flex": "elbow", "wrist_flex": "wflex", "wrist_roll": "wroll"}

#: What to ask for, in order
PLAN = [
    ("HOME", "アームを休ませる姿勢へ。毎回ここから始まり、ここへ戻ります",
     3.0, None),
    ("PREGRASP", "ブロックの真上。顎は開いたまま、ブロックには触れない高さで",
     3.0, None),
    ("GRASP", "ブロックを挟める位置まで下ろす。顎はまだ開いたまま",
     2.0, None),
    ("CLOSE", "リーダの顎を閉じてブロックを掴む。掴めたところで ENTER",
     1.2, None),
    ("LIFT", "そのまま真上へ持ち上げる", 1.5, None),
    ("TRANSFER", "缶へ向かう途中の、何にも当たらない姿勢", 2.5, None),
    ("CAN_ABOVE", "缶の真上。まだ掴んだまま", 2.5, None),
    ("DROP", "リーダの顎を開いてブロックを落とす。落ちたら ENTER", 1.0, None),
    ("RETURN", "HOME へ戻る姿勢", 3.0, None),
]

CAN_PLAN = [
    ("CAN_ABOVE", "缶の真上。ブロックを掴んだつもりの姿勢で", 2.5, None),
    ("DROP", "顎を開く姿勢", 1.0, None),
    ("RETURN", "HOME へ戻る姿勢", 3.0, None),
]


def drive_until_enter(robot, teleop, label, hint, fps=60):
    """Leader drives follower until ENTER. Returns the follower's pose."""
    from lerobot.utils.utils import enter_pressed, move_cursor_up

    print(f"\n  === {label} ===")
    print(f"  {hint}")
    print("  リーダで動かし、決まったら ENTER。")
    period = 1.0 / fps
    while True:
        started = time.perf_counter()
        robot.send_action(teleop.get_action())
        pose = joints_of(robot)
        load, where, temperature, grip = health(robot)
        print("    " + "  ".join(f"{n[:5]}{pose[n]:+7.1f}" for n in ARM_JOINTS)
              + f"  grip{pose.get('gripper', 0):+6.1f}"
              + f"   腕 {load:>4}({where or '-'})  顎 {grip:>4}  {temperature}C   ",
              end="", flush=True)
        if enter_pressed():
            print()
            return pose
        print()
        move_cursor_up(1)
        time.sleep(max(0.0, period - (time.perf_counter() - started)))


def main():
    parser = argparse.ArgumentParser(
        description="リーダアームで展示用の軌道を教示して保存します")
    parser.add_argument("--slot", type=int, help="教示する Slot 番号")
    parser.add_argument("--can", action="store_true",
                        help="缶へ運ぶ共通部分だけを教示します")
    parser.add_argument("--review", action="store_true",
                        help="保存済みの内容を表示するだけ。実機に接続しません")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--leader-port", default="COM3")
    parser.add_argument("--out", type=Path, default=SLOT_DIR)
    args = parser.parse_args()

    if not args.can and args.slot is None:
        raise SystemExit("\n  --slot N か --can を指定してください\n")
    name = "can" if args.can else f"slot{args.slot}"
    path = args.out / f"{name}.json"

    if args.review:
        trajectory = Trajectory.load(path)
        print(f"\n  {trajectory}")
        print(f"  教示 {trajectory.created}\n")
        print(f"  {'phase':<12}" + "".join(f"{SHORT[n]:>9}" for n in ARM_JOINTS)
              + f"{'grip':>8}{'秒':>6}")
        for point in trajectory.waypoints:
            print(f"  {point.phase:<12}"
                  + "".join(f"{point.joints[n]:>+9.1f}" for n in ARM_JOINTS)
                  + (f"{point.gripper:>+8.1f}" if point.gripper is not None
                     else f"{'-':>8}")
                  + f"{point.seconds:>6.1f}")
        return

    plan = CAN_PLAN if args.can else PLAN
    from so101.hardware import resolve as resolve_port
    follower_port = resolve_port([args.follower_port])[0]
    leader_port = resolve_port([args.leader_port])[0]

    print(f"\n  {name} を教示します。フォロワ {follower_port}、"
          f"リーダ {leader_port}")
    if path.is_file():
        print(f"  *** {path} は既にあります。保存すると上書きします "
              f"（バックアップを取ります）***")

    print("\n  --- 読み取りのみで現在の状態を確認します ---")
    limits = read_limits(follower_port)
    print(f"    {'joint':<16}{'現在角':>9}{'可動範囲':>20}{'torque':>8}{'温度':>7}")
    for joint in ARM_JOINTS + ("gripper",):
        info = limits[joint]
        span = f"{info['min_deg']:+.1f} .. {info['max_deg']:+.1f}"
        print(f"    {joint:<16}{info['deg']:>+9.2f}{span:>20}"
              f"{info['torque']:>8}{info['temperature']:>6}C")

    print("\n  --- 接続する前に ---")
    print("    robot.connect() でフォロワのトルクが入り、その直後から")
    print("    *** リーダの姿勢へ追従します ***。リーダを手で持ってから、")
    print("    フォロワの現在姿勢に近づけた状態で始めてください。")
    print("    アームの可動範囲に人も物もないこと。")
    print("    緊急停止: Ctrl-C → 別ターミナルで "
          "uv run scripts/torque_off.py COM4")
    if input("\n  接続するなら ready と入力（それ以外は中止）: ").strip().lower() \
            != "ready":
        print("  接続前に中止しました。何も動かしていません。")
        return

    from so101.hardware import bus_patch  # noqa: F401
    from so101.hardware import tuning
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.so_leader import SO101LeaderConfig

    tuning.install(verbose=False)
    robot = make_robot_from_config(SO101FollowerConfig(
        port=follower_port, id="follower"))
    teleop = make_teleoperator_from_config(SO101LeaderConfig(
        port=leader_port, id="leader"))

    trajectory = Trajectory(
        name=name, slot=None if args.can else str(args.slot),
        created=datetime.now().astimezone().isoformat(timespec="seconds"),
        note="taught with the leader arm; joint angles only")
    try:
        robot.connect()
        teleop.connect()
        print("\n  接続しました。リーダで動かせます。")
        for phase, hint, seconds, _ in plan:
            pose = drive_until_enter(robot, teleop, phase, hint)
            trajectory.waypoints.append(Waypoint(
                phase=phase,
                joints={n: round(pose[n], 3) for n in ARM_JOINTS},
                gripper=round(pose.get("gripper", 0.0), 3),
                seconds=seconds))
            print(f"    {phase} を記録しました")
    except KeyboardInterrupt:
        print("\n  中止しました。ここまでの waypoint は保存します。")
    finally:
        try:
            teleop.disconnect()
        except Exception:  # noqa: BLE001
            pass
        try:
            robot.disconnect()
            print("  フォロワを切断し、トルクを解放しました")
        except Exception as error:  # noqa: BLE001
            print(f"  切断できませんでした: {error}")
            print("  uv run scripts/torque_off.py COM4")

    if not trajectory.waypoints:
        print("  何も記録されていません。保存しません。")
        return
    if path.is_file():
        backup = path.with_name(
            f"{path.stem}.{datetime.now():%Y%m%d-%H%M%S}.json")
        backup.write_bytes(path.read_bytes())
        print(f"  以前の内容を {backup} へ退避しました")
    trajectory.save(path)
    print(f"\n  {trajectory}")
    print(f"  保存しました: {path}")
    print(f"  確認:  uv run scripts/teach_demo_slot.py "
          f"{'--can' if args.can else f'--slot {args.slot}'} --review")


if __name__ == "__main__":
    main()
