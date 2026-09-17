"""Release every servo on an arm, over the raw bus. The second emergency stop.

This exists for the case where the LeRobot process is no longer answering. It
does not import LeRobot, does not connect a robot, and does not read a
calibration: it opens the serial port and writes zero to Torque_Enable, which is
the one thing that has to work when nothing else does.

A Windows COM port is exclusive, so this can only reach the servos once the
process holding the port has let go. If a run is hung, kill it first:

    taskkill /F /IM python.exe        (or Ctrl-C in its terminal)
    uv run scripts/torque_off.py

*** The arm falls when torque is released. *** If it is raised - and the R1
posture holds it straight up, about 400 mm - support it by hand first, or lower
it before running this. Releasing a raised arm is not a safe default; it is the
last resort, and the same warning applies to cutting the power.

    uv run scripts/torque_off.py                 # every port found
    uv run scripts/torque_off.py COM4            # just the follower
    uv run scripts/torque_off.py COM4 --report   # show torque state, change nothing
"""

from so101.platform import require_windows

require_windows()

import argparse  # noqa: E402
import sys  # noqa: E402

from so101.hardware.sts3215 import (  # noqa: E402
    Bus,
    GOAL_POSITION,
    JOINT_NAMES,
    PRESENT_POSITION,
    TORQUE_ENABLE,
    TICKS_PER_DEG,
)

# See the note in check_wrist_flex_operational_range.py: console output is fine
# whatever the code page, but a redirect falls back to cp932. This is the
# emergency script - it must not be the thing that raises.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ATTEMPTS = 5


def release(bus, sid, name, report_only):
    """Write Torque_Enable = 0 until the servo confirms it. Returns a line."""
    position = bus.read(sid, PRESENT_POSITION, 2)
    before = bus.read(sid, TORQUE_ENABLE, 1)
    if position is None or before is None:
        return f"  {name:<14} 応答なし"
    where = f"{position:>5} ticks ({position / TICKS_PER_DEG:>6.1f}d)"
    if report_only:
        return f"  {name:<14} {where}  torque {before}"
    if before == 0:
        return f"  {name:<14} {where}  すでに OFF"

    # Park the goal where the joint actually is first. A servo whose goal is far
    # from its position will lunge there the moment torque comes back on, and
    # something is going to turn it back on later.
    bus.write(sid, GOAL_POSITION, position, size=2)
    for _ in range(ATTEMPTS):
        bus.write(sid, TORQUE_ENABLE, 0)
        if bus.read(sid, TORQUE_ENABLE, 1) == 0:
            return f"  {name:<14} {where}  トルク OFF"
    return f"  {name:<14} {where}  *** {ATTEMPTS} 回試しても OFF になりません ***"


def main():
    parser = argparse.ArgumentParser(
        description="アームの全サーボのトルクを解放します（緊急停止 B 系統）。")
    parser.add_argument("ports", nargs="*",
                        help="例: COM4。省略すると見つかった全ポート")
    parser.add_argument("--report", action="store_true",
                        help="トルクの状態を表示するだけで、何も変更しません")
    args = parser.parse_args()

    from so101.hardware import resolve as resolve_port

    if not args.report:
        print("\n  *** トルクを解放するとアームは落下します。"
              "上がっているなら先に支えてください。 ***\n")

    failed = False
    for port in resolve_port(args.ports):
        print(f"=== {port} ===")
        try:
            with Bus(port) as bus:
                for sid, name in JOINT_NAMES.items():
                    if not bus.ping(sid):
                        print(f"  {name:<14} 応答なし")
                        failed = True
                        continue
                    line = release(bus, sid, name, args.report)
                    print(line)
                    failed = failed or "***" in line
        except Exception as error:  # noqa: BLE001 - report, do not raise
            print(f"  {port} を開けません: {error}")
            print("  別のプロセスがまだ保持しています。先にそれを止めてください:")
            print("    taskkill /F /IM python.exe")
            failed = True
        print()

    if failed and not args.report:
        print("  全サーボの OFF を確認できませんでした。"
              "アームを支えたうえで電源を落としてください。")
        sys.exit(1)


if __name__ == "__main__":
    main()
