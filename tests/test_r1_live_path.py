"""Drive R1's live path against a mocked arm, as far as the waypoints.

A dry run exercises the planning and none of the running. That gap is not
theoretical: `--fraction 0.15` passed its dry run and then died on the arm with
a NameError, on a line that only the live path reaches, referring to a constant
that had been deleted two commits earlier. The twin had checked the geometry
and nothing had checked the code.

So this stands in a robot and walks the whole live path: connect, the velocity
writes, every stage, the arrival checks, the summary, the park and the release.
It asserts very little about the numbers - they are the mock's, not the arm's -
and everything about the path being runnable at all.

Both shapes are covered, because the difference between them is where the bug
was: a run that needs a lift-off stage and a run that does not.
"""

import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_wrist_flex_operational_range.py"

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="the script refuses to import off Windows")


@pytest.fixture(scope="module")
def r1():
    spec = importlib.util.spec_from_file_location("r1_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBus:
    """Answers the registers the run reads, and remembers what it was told."""

    def __init__(self, robot):
        self.robot = robot
        self.writes = []
        self.sync_writes = []
        self.torque = 1

    def _ticks(self, motor):
        return int(self.robot.pose[motor] * 4095 / 360 + 2047)

    def read(self, register, motor, normalize=True, num_retry=0):
        if register == "Present_Position":
            return (self.robot.pose[motor] if normalize
                    else self._ticks(motor))
        return {"Goal_Position": self._ticks(motor), "Present_Load": 42,
                "Present_Current": 0, "Present_Temperature": 41,
                "Present_Voltage": 123, "Torque_Enable": self.torque}[register]

    def write(self, register, motor, value, normalize=True, num_retry=0):
        self.writes.append((register, motor, value))

    def sync_read(self, register, normalize=True, num_retry=0):
        assert register == "Present_Position"
        return {m: (self.robot.pose[m] if normalize else self._ticks(m))
                for m in self.robot.pose}

    def sync_write(self, register, values, normalize=True, num_retry=0):
        self.sync_writes.append((register, dict(values)))

    def disable_torque(self, *args, **kwargs):
        self.torque = 0


class FakeRobot:
    """An arm that goes exactly where it is told, immediately."""

    def __init__(self, pose):
        self.pose = dict(pose)
        self.bus = FakeBus(self)
        self.connected = False
        self.actions = 0

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        return {f"{name}.pos": value for name, value in self.pose.items()}

    def send_action(self, action):
        self.actions += 1
        for key, value in action.items():
            self.pose[key.removesuffix(".pos")] = float(value)
        return dict(action)


@pytest.fixture
def bench(monkeypatch, r1):
    """A mocked LeRobot, a mocked arm, and an operator who says yes."""
    robot = FakeRobot({"shoulder_pan": 4.09, "shoulder_lift": -100.97,
                       "elbow_flex": 95.16, "wrist_flex": 10.55,
                       "wrist_roll": -55.30, "gripper": -4.75})

    robots = types.ModuleType("lerobot.robots")
    robots.make_robot_from_config = lambda config: robot
    follower = types.ModuleType("lerobot.robots.so_follower")
    follower.SO101FollowerConfig = lambda **kwargs: kwargs
    tuning = types.ModuleType("so101.hardware.tuning")
    tuning.install = lambda verbose=True: None
    tuning.p_coefficient = lambda: "24"
    for name, module in (("lerobot.robots", robots),
                         ("lerobot.robots.so_follower", follower),
                         ("so101.hardware.tuning", tuning),
                         ("so101.hardware.bus_patch",
                          types.ModuleType("so101.hardware.bus_patch"))):
        monkeypatch.setitem(sys.modules, name, module)

    def operator(prompt=""):
        """Someone who agrees, answering each prompt the way it asks to be answered.

        Not "ready" to everything: the gates take different words on purpose -
        the word for connecting, ENTER for each move, "home" for folding up
        after an abort - and a stand-in that says "ready" to a prompt wanting
        ENTER is read as a refusal, which is what it is.
        """
        if "ready" in prompt:
            return "ready"
        if "home" in prompt:
            return "home"
        return ""

    monkeypatch.setattr(builtins, "input", operator)
    # The logic is what is under test, not the wall clock.
    monkeypatch.setattr(r1.time, "sleep", lambda seconds: None)
    return robot


def _arm_record(pose):
    """What read_arm would have returned for this pose."""
    return {name: {"id": index + 1, "ticks": int(value * 4095 / 360 + 2047),
                   "min_ticks": 841, "max_ticks": 3253, "mid_ticks": 2047.0,
                   "deg": value, "min_deg": -106.02, "max_deg": 106.02,
                   "torque": 0, "temperature": 41, "voltage": 123}
            for index, (name, value) in enumerate(pose.items())}


def _args(r1, tmp_path, **overrides):
    import argparse

    values = {"to": 0.0, "posture_only": True, "dry_run": False,
              "posture": "upright", "hold": 0.2, "repeats": 1,
              "follower_port": "COM4", "out": tmp_path, "fraction": 1.0,
              "measured_clearance": None, "support": None, "no_support": True,
              "where": False}
    values.update(overrides)
    return argparse.Namespace(**values)


def _fly(r1, bench, tmp_path, liftoff, **overrides):
    """Plan and fly one run against the mock. Returns the summary."""
    pose = {name: bench.pose[name] for name in r1.ARM_JOINTS}
    arm = _arm_record(dict(bench.pose))
    args = _args(r1, tmp_path, **overrides)
    table = r1.Table(bench.pose["gripper"], None)
    stages = r1.stages_for(pose, r1.POSTURES["upright"], args.to, args.repeats,
                           args.posture_only, "upright", liftoff, args.fraction)
    clear, clearance = r1.replay(stages, pose, table, r1.Log())
    clearance["liftoff"] = {"needed": bool(liftoff), "using": "elbow_flex -1",
                            "move_deg": -4.0, "start_clearance_mm": 2.9,
                            "reaches_mm": 18.6, "sensitivity_mm_per_deg": {}}
    out_dir = tmp_path / ("liftoff" if liftoff else "direct")
    r1.run(args, arm, r1.POSTURES["upright"], stages, "COM4", out_dir,
           r1.Log(), clearance, None)
    return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))


def test_live_path_without_a_liftoff_stage(r1, bench, tmp_path):
    """The shape that crashed: no lift-off, so the stage indices were wrong."""
    summary = _fly(r1, bench, tmp_path, liftoff=None, fraction=0.15)

    assert summary["aborted"] is None, summary["aborted"]
    assert bench.connected is False, "the run must disconnect at the end"
    assert bench.actions > 0, "nothing was ever commanded"
    assert summary["liftoff"]["liftoff_needed"] is False
    assert summary["liftoff"]["liftoff_waypoints"] == []
    assert summary["upright_reached"] is False, "a 15% run has not arrived"
    assert [stage["kind"] for stage in summary["stages"]] == ["transit"]


def test_live_path_with_a_liftoff_stage(r1, bench, tmp_path):
    """And the shape it was written for, so the fix did not just move the bug."""
    liftoff = {"elbow_flex": bench.pose["elbow_flex"] - 4.0}
    summary = _fly(r1, bench, tmp_path, liftoff=liftoff)

    assert summary["aborted"] is None, summary["aborted"]
    assert summary["liftoff"]["liftoff_needed"] is True
    assert summary["liftoff"]["liftoff_waypoints"], "the waypoints are not recorded"
    assert summary["liftoff"]["liftoff_selected_joint"] == "elbow_flex"
    assert summary["liftoff"]["liftoff_clearance_before_mm"] == 2.9
    assert summary["liftoff"]["liftoff_clearance_after_mm"] == 18.6
    assert summary["upright_reached"] is True
    assert [stage["kind"] for stage in summary["stages"]] == ["transit", "transit"]


def test_the_velocity_limit_is_set_and_put_back(r1, bench, tmp_path):
    """Goal_Velocity is the hardware's own ceiling now that nothing clamps commands."""
    _fly(r1, bench, tmp_path, liftoff=None, fraction=0.15)

    velocity = [value for register, _motor, value in bench.bus.writes
                if register == "Goal_Velocity"]
    assert velocity, "Goal_Velocity was never written"
    assert velocity[0] > 0 and velocity[-1] == 0, \
        "it must be set on the way in and restored on the way out"


def test_no_stale_names_anywhere_in_the_module(r1):
    """The constant whose ghost caused this, and any other like it."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "LIFTOFF_DEG" not in source.replace("LIFTOFF_STEP_DEG", "") \
        .replace("LIFTOFF_MAX_DEG", ""), \
        "the fixed lift-off constant is referenced again"
