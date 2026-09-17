"""The exhibition stack, checked without a robot or a camera.

What matters here is not accuracy - the trajectories are taught, so they are as
accurate as the person who taught them - but that the refusals fire. In front of
an audience the failure that matters is the arm moving when it should not have,
and every one of those paths is here.
"""

import json
import math

import pytest

from so101.demo.slots import SlotMap
from so101.demo.trajectory import (
    ARM_JOINTS,
    Trajectory,
    Unsafe,
    Waypoint,
    play,
)

POSE = {"shoulder_pan": 0.0, "shoulder_lift": -95.0, "elbow_flex": 90.0,
        "wrist_flex": 40.0, "wrist_roll": 0.0}
LIMITS = {name: {"min_deg": -100.0, "max_deg": 100.0, "deg": POSE[name]}
          for name in ARM_JOINTS}


@pytest.fixture
def slot_map():
    return SlotMap(slots={"slot1": (200.0, 300.0), "slot2": (400.0, 300.0),
                          "slot3": (600.0, 300.0)},
                   acceptance_radius_px=90.0)


def test_a_block_in_a_slot_is_assigned_to_it(slot_map):
    assert slot_map.nearest((205.0, 305.0))[0] == "slot1"
    assert slot_map.nearest((596.0, 290.0))[0] == "slot3"


def test_a_block_nowhere_near_a_slot_is_refused(slot_map):
    name, distance = slot_map.nearest((400.0, 520.0))
    assert name is None, "a block off the slots must not be assigned to one"
    assert distance > slot_map.acceptance_radius_px


def test_a_block_between_two_slots_is_ambiguous(slot_map):
    """Halfway between two slots is not 'whichever is a pixel nearer'."""
    assert slot_map.ambiguous((300.0, 300.0))
    assert not slot_map.ambiguous((205.0, 305.0))


def test_the_slot_map_survives_a_round_trip(slot_map, tmp_path):
    path = slot_map.save(tmp_path / "slots.json")
    again = SlotMap.load(path)
    assert again.slots == slot_map.slots
    assert again.acceptance_radius_px == slot_map.acceptance_radius_px
    # The file is meant to be read and edited by a person tomorrow.
    written = json.loads(path.read_text(encoding="utf-8"))
    assert set(written) >= {"camera", "resolution", "slots",
                            "acceptance_radius_px"}


def _trajectory(**overrides):
    joints = dict(POSE, **overrides)
    return Trajectory(name="slot1", slot="1", waypoints=[
        Waypoint("HOME", dict(POSE), gripper=40.0),
        Waypoint("PREGRASP", joints, gripper=40.0),
    ])


def test_a_trajectory_survives_a_round_trip(tmp_path):
    path = _trajectory(shoulder_pan=12.0).save(tmp_path / "slot1.json")
    again = Trajectory.load(path)
    assert again.phases() == ["HOME", "PREGRASP"]
    assert again.waypoints[1].joints["shoulder_pan"] == 12.0
    assert again.waypoints[1].gripper == 40.0


def test_a_waypoint_outside_the_servo_limits_is_refused():
    with pytest.raises(Unsafe):
        _trajectory(shoulder_pan=140.0).check(LIMITS, log=lambda *_: None)


def test_a_trajectory_that_starts_far_from_the_arm_is_refused():
    """A taught file is trusted, but not enough to fling the arm across."""
    far = {name: value + 150.0 for name, value in POSE.items()}
    with pytest.raises(Unsafe):
        _trajectory().check(LIMITS, start=far, log=lambda *_: None)


def test_a_sane_trajectory_passes():
    assert _trajectory(shoulder_pan=12.0).check(
        LIMITS, start=dict(POSE), log=lambda *_: None)


class FakeBus:
    def __init__(self, robot, load=30):
        self.robot, self.load = robot, load

    def read(self, register, motor, normalize=True, num_retry=0):
        return {"Present_Load": self.load, "Present_Temperature": 40}[register]

    def sync_read(self, register, normalize=True, num_retry=0):
        return {name: 2047 for name in self.robot.pose}

    def sync_write(self, register, values, normalize=True, num_retry=0):
        pass


class FakeRobot:
    """Goes where it is told. `lag` leaves it short, as a stuck joint would."""

    def __init__(self, pose, lag=0.0, load=30):
        self.pose = dict(pose, gripper=40.0)
        self.bus = FakeBus(self, load)
        self.lag = lag

    def get_observation(self):
        return {f"{name}.pos": value for name, value in self.pose.items()}

    def send_action(self, action):
        for key, value in action.items():
            name = key.removesuffix(".pos")
            self.pose[name] = float(value) - (self.lag if name in ARM_JOINTS
                                              else 0.0)
        return dict(action)


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    import so101.demo.trajectory as module
    import so101.policy.motion as motion

    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(motion.time, "sleep", lambda seconds: None)


def test_play_walks_the_whole_trajectory():
    robot = FakeRobot(POSE)
    records = play(robot, _trajectory(shoulder_pan=12.0), log=lambda *_: None)
    assert [r["phase"] for r in records] == ["HOME", "PREGRASP"]
    assert robot.pose["shoulder_pan"] == pytest.approx(12.0)


def test_play_stops_when_a_waypoint_is_not_reached():
    """A joint that will not arrive is the fault; it must not carry on."""
    robot = FakeRobot(POSE, lag=9.0)
    with pytest.raises(Unsafe, match="到達"):
        play(robot, _trajectory(shoulder_pan=12.0), log=lambda *_: None)


def test_play_stops_on_load():
    robot = FakeRobot(POSE, load=900)
    with pytest.raises(Unsafe, match="負荷"):
        play(robot, _trajectory(shoulder_pan=12.0), log=lambda *_: None)


def test_play_refuses_a_waypoint_miles_from_the_arm():
    robot = FakeRobot(POSE)
    wild = Trajectory(name="slot1", waypoints=[
        Waypoint("HOME", {name: value + 200.0
                          for name, value in POSE.items()}, gripper=40.0)])
    with pytest.raises(Unsafe, match="離れて"):
        play(robot, wild, log=lambda *_: None)
