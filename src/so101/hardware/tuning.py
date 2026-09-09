"""Follower servo gain override.

LeRobot writes `P_Coefficient = 16` to every follower joint on each connect, half
the servo default of 32, with the comment "to avoid shakiness". That is the main
source of tracking lag on this rig, and because it is rewritten on every connect
it cannot be changed by writing the register beforehand - the override has to hook
`configure()`.

24 tracks noticeably tighter here without oscillating. Set SO101_P_COEFFICIENT to
try another value; "16" restores LeRobot's own setting, "32" is the servo default.
"""

import os

DEFAULT_P_COEFFICIENT = "24"
ENV_VAR = "SO101_P_COEFFICIENT"

_installed = False


def p_coefficient():
    return os.environ.get(ENV_VAR, DEFAULT_P_COEFFICIENT)


def install(verbose=True):
    """Patch SOFollower.configure to apply the gain override. Idempotent."""
    global _installed
    if _installed:
        return
    from lerobot.robots.so_follower.so_follower import SOFollower

    original = SOFollower.configure
    value = p_coefficient()

    def configure(self):
        original(self)
        if value is None:
            return
        with self.bus.torque_disabled():
            for motor in self.bus.motors:
                self.bus.write("P_Coefficient", motor, int(value))
        if verbose:
            print(f"[so101] P_Coefficient set to {value} "
                  f"(LeRobot writes 16 on every connect)")

    SOFollower.configure = configure
    _installed = True
