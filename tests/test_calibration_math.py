"""Homing offsets derived from recorded travel must stay inside the encoder range.

This is the failure that makes LeRobot's own calibration throw away a whole
recording: a joint starting near a mechanical stop records a range outside
0-4095. easy_calibrate centres on the midpoint of the travel instead.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

RESOLUTION = 4096
CENTRE = (RESOLUTION - 1) // 2
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]


@pytest.fixture
def bus():
    return SimpleNamespace(
        motors={n: SimpleNamespace(id=i + 1, model="sts3215")
                for i, n in enumerate(JOINTS)},
        model_resolution_table={"sts3215": RESOLUTION},
    )


@pytest.mark.parametrize("mins,maxes", [
    # Travel actually recorded on this rig's follower arm.
    ({"shoulder_pan": 1018, "shoulder_lift": 1698, "elbow_flex": 782,
      "wrist_flex": 742, "gripper": 948},
     {"shoulder_pan": 3601, "shoulder_lift": 4079, "elbow_flex": 2991,
      "wrist_flex": 3027, "gripper": 2433}),
    # Leader arm, whose elbow travel crosses the encoder's zero.
    ({"shoulder_pan": 635, "shoulder_lift": 1330, "elbow_flex": -87,
      "wrist_flex": 370, "gripper": 1040},
     {"shoulder_pan": 3213, "shoulder_lift": 3734, "elbow_flex": 2132,
      "wrist_flex": 2715, "gripper": 2345}),
    # A joint pinned hard against one stop, the case that broke lerobot-calibrate.
    ({"shoulder_pan": 0, "shoulder_lift": 100, "elbow_flex": 2040,
      "wrist_flex": 4000, "gripper": 2047},
     {"shoulder_pan": 4095, "shoulder_lift": 2500, "elbow_flex": 2050,
      "wrist_flex": 4095, "gripper": 3000}),
])
def test_ranges_stay_within_the_encoder(bus, mins, maxes):
    import easy_calibrate as ec

    positions = dict(mins, wrist_roll=2268)
    calibration = ec.build_calibration(bus, mins, maxes, positions)

    for joint, cal in calibration.items():
        assert 0 <= cal.range_min, f"{joint} range_min {cal.range_min} is negative"
        assert cal.range_max <= 4095, f"{joint} range_max {cal.range_max} overflows"

        if joint == ec.FULL_TURN_MOTOR:
            assert (cal.range_min, cal.range_max) == (0, RESOLUTION - 1)
        else:
            travel = maxes[joint] - mins[joint]
            assert cal.range_max - cal.range_min == travel, f"{joint} travel changed"
            midpoint = (cal.range_min + cal.range_max) // 2
            assert abs(midpoint - CENTRE) <= 1, f"{joint} is not centred"
