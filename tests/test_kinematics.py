"""Forward and inverse kinematics must be consistent. No hardware needed."""

import numpy as np
import pytest

from so101.policy.kinematics import URDF_PATH, ArmKinematics

pytestmark = pytest.mark.skipif(
    not URDF_PATH.is_file(),
    reason="URDF not downloaded; run scripts/fetch_urdf.py")


@pytest.fixture(scope="module")
def arm():
    return ArmKinematics()


def test_joint_order_matches_the_robot(arm):
    assert arm.joint_names == ["shoulder_pan", "shoulder_lift", "elbow_flex",
                              "wrist_flex", "wrist_roll"]


def test_limits_are_finite_and_ordered(arm):
    for name, (low, high) in arm.joint_limits_deg().items():
        assert low < high, name
        assert -200 < low and high < 200, name


@pytest.mark.parametrize("joints", [
    dict(shoulder_pan=0, shoulder_lift=0, elbow_flex=0, wrist_flex=0, wrist_roll=0),
    dict(shoulder_pan=25, shoulder_lift=-40, elbow_flex=60, wrist_flex=-20,
         wrist_roll=15),
    dict(shoulder_pan=-30, shoulder_lift=20, elbow_flex=-35, wrist_flex=40,
         wrist_roll=-45),
])
def test_ik_recovers_the_pose_fk_produced(arm, joints):
    """Round trip: FK to a point, IK back, and the point must survive.

    The joint angles need not match - this arm has redundant configurations - but
    the position has to.
    """
    target = arm.forward(joints)
    solution = arm.inverse(target, seed_deg=joints)
    assert solution is not None, f"IK failed for a pose FK produced: {joints}"
    reached = arm.forward(solution)
    assert np.linalg.norm(reached - target) < 2e-3, f"{reached} vs {target}"


def test_unreachable_target_returns_none(arm):
    """Far outside the workspace, IK must say so rather than return a near miss."""
    assert arm.inverse([2.0, 0.0, 0.0]) is None


def test_dict_and_sequence_inputs_agree(arm):
    joints = dict(shoulder_pan=10, shoulder_lift=-15, elbow_flex=25,
                  wrist_flex=-5, wrist_roll=0)
    as_sequence = [joints[name] for name in arm.joint_names]
    assert np.allclose(arm.forward(joints), arm.forward(as_sequence))
