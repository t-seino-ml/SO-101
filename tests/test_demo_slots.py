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
    """Goes exactly where it is told."""

    def __init__(self, pose, load=30):
        self.pose = dict(pose, gripper=40.0)
        self.bus = FakeBus(self, load)

    def get_observation(self):
        return {f"{name}.pos": value for name, value in self.pose.items()}

    def send_action(self, action):
        for key, value in action.items():
            self.pose[key.removesuffix(".pos")] = float(value)
        return dict(action)


class BlockedRobot(FakeRobot):
    """A joint that meets something and cannot go past it, whatever is asked.

    Distinct from sagging on purpose, because the two look identical for one
    sample and must be handled oppositely: a sag closes when you aim past it, an
    obstacle does not, and aiming further at an obstacle is leaning on it.
    """

    def __init__(self, pose, joint="elbow_flex", stops_at=85.0, load=30):
        super().__init__(pose, load)
        self.joint, self.stops_at = joint, stops_at

    def send_action(self, action):
        for key, value in action.items():
            name = key.removesuffix(".pos")
            self.pose[name] = (max(float(value), self.stops_at)
                               if name == self.joint else float(value))
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
    robot = BlockedRobot(POSE, stops_at=85.0)
    with pytest.raises(Unsafe):
        play(robot, _trajectory(elbow_flex=60.0), log=lambda *_: None)


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


class SaggingRobot(FakeRobot):
    """Arrives a fixed amount short, like the real one under its own weight.

    The sag is against the *command*, so aiming past the target closes it -
    which is the thing being tested. A joint held by something would not behave
    like this, and `test_play_stops_when_a_waypoint_is_not_reached` covers that.
    """

    def __init__(self, pose, sag=2.6, joint="elbow_flex"):
        super().__init__(pose)
        self.sag, self.joint = sag, joint

    def send_action(self, action):
        for key, value in action.items():
            name = key.removesuffix(".pos")
            self.pose[name] = float(value) - (self.sag if name == self.joint
                                              else 0.0)
        return dict(action)


def test_play_corrects_the_sag_rather_than_accepting_it():
    """2.6 deg short at PREGRASP was a real abort, and the fix is to aim past."""
    robot = SaggingRobot(POSE)
    records = play(robot, _trajectory(elbow_flex=60.0), log=lambda *_: None)

    assert records[-1]["corrections"] >= 1, "it should have aimed again"
    assert abs(records[-1]["worst_error_deg"]) <= 1.5
    assert robot.pose["elbow_flex"] == pytest.approx(60.0, abs=0.5), \
        "the arm must end at the taught angle, not a sag below it"
    # And the correction is where it belongs: in the command, not the record of
    # what was taught.
    assert records[-1]["commanded_after_correction"]["elbow_flex"] > 60.0


def test_the_correction_gives_up_rather_than_leaning_on_something():
    """A joint held by an obstacle is not sagging, and must not be pushed."""
    robot = BlockedRobot(POSE, stops_at=85.0)
    with pytest.raises(Unsafe, match="縮みません"):
        play(robot, _trajectory(elbow_flex=60.0), log=lambda *_: None)
    assert robot.pose["elbow_flex"] == pytest.approx(85.0),         "it must have stopped at the obstacle, not pushed past it"


def _pick():
    """A taught pick: jaws open, stopped by a block at 25.6, opened again."""
    grips = [37.9, 37.9, 37.7, 25.6, 25.6, 25.6, 36.3]
    names = ["HOME", "PREGRASP", "GRASP", "CLOSE", "LIFT", "CAN_ABOVE", "DROP"]
    return Trajectory(name="slot1", waypoints=[
        Waypoint(name, dict(POSE), gripper=grip)
        for name, grip in zip(names, grips)])


def test_the_jaws_are_commanded_past_where_the_block_stopped_them():
    """Commanding exactly where the block is closes onto it, not around it."""
    from so101.demo.trajectory import SQUEEZE_DEG, squeeze_plan

    plan = squeeze_plan(_pick())
    taught = [w.gripper for w in _pick().waypoints]
    assert plan[3] == pytest.approx(taught[3] - SQUEEZE_DEG), "CLOSE must squeeze"
    assert plan[:3] == taught[:3], "the open phases are left alone"


def test_the_squeeze_is_held_all_the_way_to_the_can():
    """Re-commanding the taught angle at LIFT would put the block down again."""
    from so101.demo.trajectory import squeeze_plan

    plan = squeeze_plan(_pick())
    assert plan[4] == plan[3] == plan[5], "LIFT and CAN_ABOVE keep holding"
    assert plan[6] == 36.3, "DROP lets go"


def test_closing_on_air_is_noticed():
    """Jaws that arrive where they were sent had nothing between them."""
    robot = FakeRobot(POSE)          # goes exactly where told: nothing in the way
    records = play(robot, _pick(), log=lambda *_: None)
    assert records[3]["grasped"] is False


def test_a_block_between_the_jaws_reads_as_held():
    class Holding(FakeRobot):
        def send_action(self, action):
            super().send_action(action)
            self.pose["gripper"] = max(self.pose["gripper"], 25.6)
            return dict(action)

    records = play(Holding(POSE), _pick(), log=lambda *_: None)
    assert records[3]["grasped"] is True
    assert records[3]["grip_margin_deg"] == pytest.approx(15.0, abs=0.1)


def test_a_gripped_block_is_not_mistaken_for_a_crash():
    """A firm grip pins the gripper's load. That is success, not a fault."""
    class Gripping(FakeRobot):
        def __init__(self, pose):
            super().__init__(pose)
            self.bus.read = self._read

        def _read(self, register, motor, normalize=True, num_retry=0):
            if register == "Present_Load":
                return 500 if motor == "gripper" else 30
            return 40

        def send_action(self, action):
            super().send_action(action)
            self.pose["gripper"] = max(self.pose["gripper"], 25.6)
            return dict(action)

    records = play(Gripping(POSE), _pick(), log=lambda *_: None)
    assert records[3]["grasped"] is True
    assert records[3]["gripper_load"] == 500
    assert records[3]["load"] == 30, "the arm's load is reported on its own"


def test_an_arm_joint_under_load_still_stops_the_run():
    """Excluding the gripper must not have excluded everything."""
    robot = FakeRobot(POSE, load=900)
    with pytest.raises(Unsafe, match="アーム"):
        play(robot, _trajectory(shoulder_pan=12.0), log=lambda *_: None)


def test_a_big_move_is_given_time_whatever_the_taught_duration_says():
    """Teaching slot 2 went a phase early: an 89 deg return carried DROP's 1 s."""
    from so101.demo.trajectory import MAX_DEG_PER_SECOND

    far = dict(POSE, shoulder_lift=POSE["shoulder_lift"] + 89.0)
    trajectory = Trajectory(name="slot2", waypoints=[
        Waypoint("HOME", dict(POSE), gripper=40.0, seconds=3.0),
        Waypoint("DROP", far, gripper=40.0, seconds=1.0),   # mislabelled
    ])
    records = play(FakeRobot(POSE), trajectory, log=lambda *_: None)
    assert records[1]["seconds"] > records[1]["taught_seconds"]
    assert records[1]["deg_per_second"] <= MAX_DEG_PER_SECOND + 0.5


def test_a_small_move_keeps_its_taught_duration():
    """The cap is a ceiling, not a rewrite: slow taught moves stay slow."""
    trajectory = Trajectory(name="slot1", waypoints=[
        Waypoint("HOME", dict(POSE), gripper=40.0, seconds=3.0),
        Waypoint("CLOSE", dict(POSE, shoulder_pan=1.0), gripper=40.0,
                 seconds=1.2),
    ])
    records = play(FakeRobot(POSE), trajectory, log=lambda *_: None)
    assert records[1]["seconds"] == pytest.approx(1.2)


def test_a_big_move_between_two_taught_poses_is_allowed():
    """Folded HOME to extended PREGRASP is 125 deg on slot 4, and legitimate."""
    far = dict(POSE, shoulder_lift=POSE["shoulder_lift"] + 125.0)
    trajectory = Trajectory(name="slot4", waypoints=[
        Waypoint("HOME", dict(POSE), gripper=40.0),
        Waypoint("PREGRASP", far, gripper=40.0),
    ])
    records = play(FakeRobot(POSE), trajectory, log=lambda *_: None)
    assert len(records) == 2


def test_the_first_waypoint_is_still_held_to_the_tighter_bound():
    """Nothing is known about where the arm is when a run starts."""
    from so101.demo.trajectory import MAX_FIRST_STEP_DEG

    trajectory = Trajectory(name="slot4", waypoints=[
        Waypoint("HOME", dict(POSE), gripper=40.0)])
    stranded = {name: value + MAX_FIRST_STEP_DEG + 10.0
                for name, value in POSE.items()}
    with pytest.raises(Unsafe, match="想定外"):
        play(FakeRobot(stranded), trajectory, log=lambda *_: None)


def test_a_small_taught_opening_still_releases():
    """Slot 5 closed at 24 and opened at 29 - a rise of exactly 5.0.

    Detecting the release as "a rise bigger than 5" missed it by nothing at all,
    squeezed the jaws shut through DROP and RETURN, and tripped the gripper's
    overload protection 27 seconds later.
    """
    from so101.demo.trajectory import squeeze_plan

    grips = [39.0, 39.0, 39.0, 24.0, 24.0, 24.0, 29.0]
    names = ["HOME", "PREGRASP", "GRASP", "CLOSE", "LIFT", "CAN_ABOVE", "DROP"]
    trajectory = Trajectory(name="slot5", waypoints=[
        Waypoint(name, dict(POSE), gripper=grip)
        for name, grip in zip(names, grips)])

    plan = squeeze_plan(trajectory)
    assert plan[3] < 24.0, "CLOSE still squeezes"
    assert plan[6] == 29.0, "DROP must command the taught opening, not a squeeze"


def test_a_releasing_phase_never_squeezes():
    """Belt to the braces: releasing is the one thing that must not be missed."""
    from so101.demo.trajectory import squeeze_plan

    # Taught identically at CLOSE and DROP, which no comparison of the numbers
    # alone could tell apart.
    names = ["GRASP", "CLOSE", "LIFT", "DROP"]
    grips = [40.0, 25.0, 25.0, 25.0]
    trajectory = Trajectory(name="odd", waypoints=[
        Waypoint(name, dict(POSE), gripper=grip)
        for name, grip in zip(names, grips)])
    plan = squeeze_plan(trajectory)
    assert plan[1] < 25.0 and plan[2] < 25.0, "CLOSE and LIFT hold"
    assert plan[3] == 25.0, "DROP does not, whatever the numbers say"
