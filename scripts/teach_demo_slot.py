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
    connect,
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


def drive_until_enter(robot, teleop, label, hint, fps=60, patience=5):
    """Leader drives follower until ENTER. Returns the follower's pose.

    Nothing is read back while driving, and that is the point. This loop used to
    print every joint and poll the load and temperature of all six every frame -
    twelve extra single-register reads at 60 Hz, some seven hundred a second on
    top of the leader's read and the follower's write. The bus gave out
    mid-session with "no status packet" and the teaching went with it. The only
    read that has to happen is the one at ENTER.

    A dropped packet is survivable and a lost session is not, so a bad frame is
    retried rather than raised: the leader and the follower are on separate USB
    adapters and either can garble one.
    """
    from lerobot.utils.utils import enter_pressed

    print(f"\n  === {label} ===")
    print(f"  {hint}")
    print("  リーダで動かし、決まったら ENTER。")
    period = 1.0 / fps
    misses = 0
    while True:
        started = time.perf_counter()
        try:
            robot.send_action(teleop.get_action())
            misses = 0
        except Exception as error:  # noqa: BLE001 - one bad packet is not a fault
            misses += 1
            if misses >= patience:
                raise
            print(f"    通信が乱れました（{misses}/{patience}）: "
                  f"{type(error).__name__}")
            time.sleep(0.1)
            continue
        if enter_pressed():
            return joints_of(robot)
        time.sleep(max(0.0, period - (time.perf_counter() - started)))


def connect_leader(teleop, port, attempts=3):
    """Connect the leader, retrying a garbled first reply.

    The SDK reads a ping's answer without checking it arrived whole, so one
    truncated status packet raises IndexError out of SCS_MAKEWORD - below the
    level `so101.hardware.bus_patch` retries at, and fatal. The arms are healthy
    when it happens: diag.py reads all six servos a second later. This is the
    same fragility bus_patch exists for, caught one layer further down.
    """
    for attempt in range(1, attempts + 1):
        try:
            teleop.connect()
            return
        except Exception as error:  # noqa: BLE001 - any failure is worth a retry
            print(f"  リーダ({port}) の接続 {attempt}/{attempts} 回目が失敗: "
                  f"{type(error).__name__}: {error}")
            if attempt == attempts:
                raise SystemExit(
                    f"\n  リーダ({port}) に接続できません。\n"
                    "  電源と USB を確認してください。読み取りだけなら\n"
                    f"    uv run scripts/diag.py {port}\n"
                    "  で応答が見えます。フォロワには接続していません。\n")
            time.sleep(1.5)


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

    # The leader first, and the follower only once the leader is talking. A
    # leader that will not connect is a session that cannot happen, and there
    # is no reason for the follower's torque to have come on to find that out.
    connect_leader(teleop, leader_port)

    trajectory = Trajectory(
        name=name, slot=None if args.can else str(args.slot),
        created=datetime.now().astimezone().isoformat(timespec="seconds"),
        note="taught with the leader arm; joint angles only")
    try:
        connect(robot, follower_port)
        print("\n  接続しました。リーダで動かせます。")
        for phase, hint, seconds, _ in plan:
            pose = drive_until_enter(robot, teleop, phase, hint)
            trajectory.waypoints.append(Waypoint(
                phase=phase,
                joints={n: round(pose[n], 3) for n in ARM_JOINTS},
                gripper=round(pose.get("gripper", 0.0), 3),
                seconds=seconds))
            print(f"    {phase} を記録: "
                  + "  ".join(f"{SHORT[n]} {pose[n]:+.1f}" for n in ARM_JOINTS)
                  + f"  grip {pose.get('gripper', 0.0):+.1f}")
    except KeyboardInterrupt:
        print("\n  中止しました。ここまでの waypoint は保存します。")
    except Exception as error:  # noqa: BLE001 - keep whatever was already taught
        # Losing eight waypoints to one bad packet is the expensive outcome.
        print(f"\n  *** {type(error).__name__}: {error} ***")
        print("  ここまでの waypoint は保存します。続きは教示し直してください。")
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
