"""LeRobot teleoperation, with the serial bus hardened and the follower P gain tunable.

The stock `lerobot-teleoperate` calls `sync_read` with no retries inside its control
loop, so it dies on the first garbled status packet. See bus_patch.py.

LeRobot also pins the follower's P_Coefficient to 16 (half the servo default of 32)
on every connect, "to avoid shakiness". That is the main source of tracking lag, so
this script raises it to 24, which tracks noticeably tighter on this arm without
oscillating. Override with SO101_P_COEFFICIENT - "16" restores LeRobot's value, "32"
is the servo default.

    uv run scripts/teleoperate.py --robot.type=so101_follower --robot.port=COM4 --robot.id=follower \
        --teleop.type=so101_leader --teleop.port=COM3 --teleop.id=leader --fps=120

    SO101_P_COEFFICIENT=32 uv run scripts/teleoperate.py ...   # snappier still, may oscillate
"""

import os

from so101.hardware import bus_patch
from lerobot.robots.so_follower.so_follower import SOFollower

P_COEFFICIENT = os.environ.get("SO101_P_COEFFICIENT", "24")

_configure = SOFollower.configure


def _configure_with_gain(self):
    _configure(self)
    if P_COEFFICIENT is None:
        return
    with self.bus.torque_disabled():
        for motor in self.bus.motors:
            self.bus.write("P_Coefficient", motor, int(P_COEFFICIENT))
    print(f"[teleoperate.py] P_Coefficient set to {P_COEFFICIENT} "
          f"(LeRobot writes 16 on every connect)")


SOFollower.configure = _configure_with_gain

if __name__ == "__main__":
    from lerobot.scripts.lerobot_teleoperate import main

    print(f"[teleoperate.py] serial retries={bus_patch.NUM_RETRY}")
    main()
