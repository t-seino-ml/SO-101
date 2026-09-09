"""Servo bus access and LeRobot compatibility fixes for the SO-101.

`bus_patch` is deliberately not re-exported here: importing it monkey-patches
LeRobot's MotorsBus, so it has to be an explicit, visible import.
"""

from .ports import candidates, resolve
from .sts3215 import (
    Bus,
    JOINT_NAMES,
    MAX_ANGLE_LIMIT,
    MIN_ANGLE_LIMIT,
    PRESENT_LOAD,
    PRESENT_POSITION,
    PRESENT_TEMPERATURE,
    PRESENT_VOLTAGE,
    TICKS_PER_DEG,
    TORQUE_ENABLE,
)

__all__ = [
    "Bus", "JOINT_NAMES", "TICKS_PER_DEG", "candidates", "resolve",
    "MIN_ANGLE_LIMIT", "MAX_ANGLE_LIMIT", "PRESENT_POSITION", "PRESENT_LOAD",
    "PRESENT_VOLTAGE", "PRESENT_TEMPERATURE", "TORQUE_ENABLE",
]
