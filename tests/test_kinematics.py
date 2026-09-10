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
def test_unconstrained_ik_recovers_the_pose_fk_produced(arm, joints):
    """Round trip: FK to a point, IK back, and the point must survive.

    Orientation is left free here. An arbitrary pose reaches its point with the
    tool at an arbitrary angle, and asking IK to reproduce the position *and*
    point the tool down is a different, stricter question - one this arm often
    cannot answer, which is the subject of the next test.

    The joint angles need not match either: the arm has redundant configurations.
    """
    target = arm.forward(joints)
    solution = arm.inverse(target, seed_deg=joints, orientation=None)
    assert solution is not None, f"IK failed for a pose FK produced: {joints}"
    reached = arm.forward(solution)
    assert np.linalg.norm(reached - target) < 2e-3, f"{reached} vs {target}"


@pytest.mark.parametrize("position", [
    (0.20, 0.00, 0.02),
    (0.24, -0.08, 0.01),
    (0.26, 0.05, 0.03),
])
def test_grasp_poses_point_the_tool_down(arm, position):
    """A grasp needs the jaws coming down onto the block, not across it.

    This is the default for `inverse`, because a solution that reaches the right
    point with the gripper at 21 degrees off vertical - which is what the
    unconstrained solver returned on this arm - cannot close on a block at all.
    """
    solution = arm.inverse(position)
    assert solution is not None, f"no downward grasp at {position}"
    assert np.linalg.norm(arm.forward(solution) - np.array(position)) < 2e-3
    axis = arm.tool_axis(solution)
    tilt = np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))
    assert tilt < 5, f"tool is {tilt:.1f} degrees off vertical: {axis}"


def test_unreachable_target_returns_none(arm):
    """Far outside the workspace, IK must say so rather than return a near miss."""
    assert arm.inverse([2.0, 0.0, 0.0]) is None


def test_dict_and_sequence_inputs_agree(arm):
    joints = dict(shoulder_pan=10, shoulder_lift=-15, elbow_flex=25,
                  wrist_flex=-5, wrist_roll=0)
    as_sequence = [joints[name] for name in arm.joint_names]
    assert np.allclose(arm.forward(joints), arm.forward(as_sequence))
