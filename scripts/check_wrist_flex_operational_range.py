"""Phase R1: confirm one wrist_flex angle is safe to work at, and record why.

This is not scripts/measure_joint_range.py and must not be confused with it.
That script finds where the joint physically stops, by pressing on the stop
until the load pins. This one never looks for a stop. It asks a different
question, which is the one the planner actually needs answered:

    can the arm be sent to this angle, hold it, and be sent there again -
    with the tracking, the load and the temperature all staying ordinary?

That is an *operational* range, and it is narrower than the travel by design.
Nothing here writes a joint limit anywhere: R1 produces evidence, and the limit
is decided afterwards from the whole set of angles, both directions.

Why wrist_flex first. Over the 902 waypoints of the simulated 15x15 pick
workspace, wrist_flex is the joint nearest its operational limit at 704 of them
- 78%, against 19% for wrist_roll and 1% for everything else. It is the joint
that decides how much of the table can be picked from.

    ONE ANGLE PER RUN. There is deliberately no way to sweep. Each run is
    looked at by a person before the next angle is chosen.

        uv run scripts/check_wrist_flex_operational_range.py --posture-only
        uv run scripts/check_wrist_flex_operational_range.py --to +20
        uv run scripts/check_wrist_flex_operational_range.py --to +20 --dry-run

WHAT THE FIRST ATTEMPT GOT WRONG, because all of it is now load-bearing:

  The run was given max_relative_target = 2 degrees, meaning every command was
  clamped to within 2 degrees of where the joint already was. That caps the
  position error the servo ever sees, which caps the torque it develops. A
  joint carrying any weight then cannot advance, the interpolation walks away
  from it, and the command sits pinned at exactly present + 2 for ever. Measured:
  the tracking error held at -2.00 degrees, to the hundredth, for 4.6 seconds.

  elbow_flex lost that argument during the transit and stopped at +52.8 degrees
  when it had been asked for 0. Nothing checked, so the run called the posture
  reached and swung the wrist. With the elbow at +50 the arm is not standing up,
  it is reaching out over the table: replayed in the twin, that sweep puts the
  lowest point of the gripper 34.8 mm below a table at -8.5 mm. The wrist stopped
  at +34.6 degrees against a load of 132 where gravity asks for 0.6% of stall.
  It was pressing on the table.

  So: no relative clamp; speed comes from the interpolation and Goal_Velocity.
  Every waypoint is confirmed reached before the next is sent. Every phase ends
  with an arrival check across *all* the joints it moved, and a phase that did
  not arrive does not hand over to the next one. Stall detection runs while
  moving, not only while holding - a joint 35 degrees short and motionless is
  the thing to catch, and catching it as a "tracking error" at a phase boundary
  2.5 seconds later was luck, not design.
"""

from so101.platform import require_windows

require_windows()

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

# The operator-facing text is Japanese. Windows writes console output through
# WriteConsoleW, so that prints correctly in a terminal whatever the code page
# is - but redirected to a file or a pipe it falls back to the locale encoding,
# which here is cp932 and cannot carry every character used below. Ask for
# UTF-8, and settle for a replacement character rather than an exception raised
# in the middle of a run that is holding a raised arm.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a text stream, or already fixed
        pass

JOINT = "wrist_flex"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
              "wrist_roll")

#: The angles R1 is allowed to ask for, either way. There is no override flag
#: and none should be added: past here the joint has under 6 degrees left before
#: the servo's own Min/Max_Position_Limit, and that margin is the last thing
#: standing between a planner mistake and a stop.
LIMIT_DEG = 100.0
#: Where the wrist is parked between repeats and before the arm is brought home.
SAFE_DEG = 0.0

POSTURES = {
    # Checked in the twin across the whole -100..+100 sweep: no self-collision,
    # lowest arm point +16.2 mm, TCP never below +76 mm.
    "upright": {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
                "wrist_roll": 0.0},
    # The same, for a bench without 400 mm of headroom. Gravity asks slightly
    # less of shoulder_lift here (13% of stall against 18%).
    "half-up": {"shoulder_pan": 0.0, "shoulder_lift": -20.0, "elbow_flex": 20.0,
                "wrist_roll": 0.0},
}

# -- how the arm is moved -------------------------------------------------
#: How far apart the confirmed waypoints are, on the fastest-moving joint. The
#: arm is walked from one to the next and each is confirmed reached before the
#: next is sent, so this is also how far wrong a single unnoticed step can go.
STEP_DEG = 3.0
#: Within this of a waypoint counts as reached. The servos settle a few tenths
#: short under gravity with no integral term, so demanding much less than this
#: would be demanding something the hardware does not do.
ARRIVE_DEG = 0.8
#: A phase has arrived only when every joint it moved is this close. Looser than
#: a waypoint because it is checked once the arm has stopped and settled.
PHASE_TOLERANCE_DEG = 1.5
#: Longest a single waypoint may take before it counts as not arriving.
WAYPOINT_TIMEOUT_S = 4.0
#: Speed the 30 Hz interpolation asks for between waypoints.
WRIST_SPEED_DEG_S = 8.0
TRANSIT_SPEED_DEG_S = 5.0
#: Getting off the table before anything else moves, with whichever joints
#: actually lift the gripper from the pose the arm is in.
#:
#: Which joints those are is not a constant, and assuming it is would be the
#: same mistake in a new place. From the folded parked pose, shoulder_lift - the
#: joint that raises the arm - pushes the gripper *down*: +10 degrees of it puts
#: the gripper 13.9 mm below the table, while elbow_flex and wrist_flex lift it.
#: Interpolating every joint together does clear, but only because the lifting
#: pair wins the race, and that cancellation is exactly what fails when a joint
#: lags. Measured in the twin: hold elbow_flex still, let shoulder_lift track,
#: and 2% of the way through the transit the gripper is already through the
#: table. elbow_flex is the joint that lagged on the first attempt.
#:
#: So the direction is measured from wherever the arm is, every run, and the
#: joints that turn out to lower the gripper are not commanded until it is
#: clear. These are the candidates and the step the search grows in.
LIFT_CANDIDATES = ("wrist_flex", "elbow_flex", "shoulder_lift")
LIFTOFF_STEP_DEG = 4.0
LIFTOFF_MAX_DEG = 40.0
LIFTOFF_SPEED_DEG_S = 1.5
#: What the arm must have before it is allowed to move at all, and what it must
#: keep afterwards. The first live attempt put the gripper on the table; 2.9 mm
#: of model clearance is not enough to cover CAD error, backlash and sag, so the
#: run refuses to start below this rather than reporting it and carrying on.
CLEARANCE_FLOOR_MM = 10.0
CLEARANCE_WANTED_MM = 15.0
#: What a *measured* start clearance must be, when one is given. Higher than the
#: modelled floor on purpose: a ruler read to the nearest centimetre is worth
#: about +-5 mm, and a start condition should not be one measurement error away
#: from being false.
MEASURED_FLOOR_MM = 30.0

#: Test A's headline criterion, and the one that survives what is known about
#: the model.
#:
#: The model's absolute clearance figures are wrong. Measured against a ruler on
#: 2026-09-17: 17.6 mm out at the parked pose, 45.3 mm out with the arm held up,
#: both in the conservative direction - and two poses do not make "conservative"
#: a property of every pose, only of those two. So an absolute threshold from
#: this model is a number whose error is unknown at the poses in between.
#:
#: What survives is the direction. Whatever the model's offset turns out to be,
#: the arm should not be getting closer to the table than it started, and a
#: first-order geometric fact like which way a link is moving is far more robust
#: to an offset than the height it is moving at. So this is checked over every
#: pose of the path, on the gripper and on the arm as a whole, and it is what
#: Test A is really asking.
APPROACH_TOLERANCE_MM = 1.0

#: The support stays put for the whole run - nobody reaches in to remove it once
#: the torque is on - so the arm has to leave it and then stay away from it. At
#: the start the distance is zero by definition: the arm is resting on the
#: thing. So the same reasoning as the table applies, for the same reason: the
#: criterion is the direction, not the height. Having left, it must clear this
#: much, which is generous because the support's position is measured with a
#: ruler by hand and its edges are where the error is.
SUPPORT_MARGIN_MM = 20.0
SUPPORT_TOLERANCE_MM = 1.0
#: How much of the travel a waypoint must stay inside the servo's own limits by.
LIMIT_MARGIN_DEG = 2.0
#: Written into the servos' Goal_Velocity, and restored to 0 afterwards. With
#: no relative clamp on the commands this is the hardware's own ceiling on how
#: fast anything can happen, so it matters more than it used to. Left at the
#: factory's 0 - which means "no limit" - one stray write is a full-speed move.
GOAL_VELOCITY_DEG_S = 15.0
FPS = 30
#: Deliberately not used. See the module docstring: clamping each command to
#: within a couple of degrees of the present position caps the servo's position
#: error, caps its torque, and stalls any joint carrying weight. Speed is the
#: interpolation's job and Goal_Velocity's; it is not this one's.
MAX_RELATIVE_TARGET = None

# -- when to stop ---------------------------------------------------------
# None of these is a safe level. They are the point past which the run stops
# and a person looks at it. The expected load for the sweep is around 40 of
# 1023, from the 0.117 N.m gravity asks at worst.
LOAD_ABORT = 400              # of 1023 full scale
TEMPERATURE_ABORT_C = 55
COMMS_ABORT = 3               # consecutive failed reads
#: MOVING. A joint still short of its waypoint that has stopped moving is the
#: fault worth catching, and it is not the same thing as a large tracking
#: error: a joint on its way to a target it will reach is behind by design.
#: A joint within this of its waypoint is not judged on speed: it is close
#: enough that the waypoint's own arrival timeout is the right thing to catch,
#: and a joint settling the last half-degree is meant to be slowing down.
STALL_REMAINING_DEG = 2.0
STALL_SPEED_DEG_S = 0.25
#: Speed is measured over this window and the verdict then has to hold for this
#: long again, so a stall is called after about a second. Measuring over one
#: window and then waiting out a second full one took two and a bit.
STALL_WINDOW_S = 0.5
STALL_PERSIST_S = 0.5
#: MOVING. Going the wrong way is never a lag.
RETREAT_DEG = 1.5
RETREAT_WINDOW_S = 0.5
#: HOLDING. Only once a waypoint has been confirmed reached does being away
#: from it mean anything, and then it means a great deal.
HOLD_ERROR_ABORT_DEG = 5.0

SAMPLE_HZ = 20
VELOCITY_WINDOW_S = 0.25
TICKS_PER_DEG = 4095 / 360    # LeRobot's own scale: (ticks - mid) * 360 / 4095
DEFAULT_OUT = Path("outputs/real/r1_wrist_flex")
#: The bodies whose distance to the table is what "clearance" means here. The
#: base is not among them: it *is* the table's definition, so its clearance is
#: zero by construction and would mask every other one.
MOVING_BODIES = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
                 "moving_jaw_so101_v1")


class Abort(RuntimeError):
    """Something crossed a threshold. The run stops; the arm keeps holding."""


@dataclass
class Stage:
    """One named move: its waypoints, and everything needed to fly and judge it.

    Written down rather than worked out at each use. The live path used to read
    `stages[0]` as the lift-off and `stages[2]` as the first sweep, which was
    true only while there was always a lift-off stage; the moment one was not
    needed, that indexing pointed at the wrong move. It also reached into
    `path[-1]["elbow_flex"]` for a joint that is now chosen per pose. None of
    that survives a stage knowing its own name.
    """

    label: str
    path: list          # waypoints, each a dict of joint -> degrees
    kind: str           # "transit" - may not approach the table
                        # "sweep"   - lowers the gripper on purpose, held to the floor
    speed: float        # degrees per second between waypoints
    subject: str        # the joint this stage is about, for the log and the watch
    note: str = ""      # what to tell the operator before it runs


# -------------------------------------------------------------------------
# printing next to a machine
# -------------------------------------------------------------------------

def cells(text):
    """How many terminal columns `text` occupies.

    The operator-facing text is Japanese and the numbers beside it are not. A
    kana or kanji occupies two columns where an ASCII letter occupies one, so
    f-string padding - which counts characters - tears the table apart exactly
    where someone is trying to read a load figure in a hurry.
    """
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1
               for c in text)


def pad(text, columns):
    """Left-align `text` in `columns` terminal columns."""
    return text + " " * max(0, columns - cells(text))


def rpad(text, columns):
    """Right-align `text` in `columns` terminal columns."""
    return " " * max(0, columns - cells(text)) + text


class Log:
    """Print to the screen and to the run's own file, so nothing is only scrollback."""

    def __init__(self, path=None):
        self.file = None if path is None else open(path, "w", encoding="utf-8")

    def __call__(self, line=""):
        print(line)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


# -------------------------------------------------------------------------
# one run at a time
# -------------------------------------------------------------------------

@contextmanager
def only_one(path):
    """Refuse to start while another run holds this lock.

    An OS lock rather than a PID file that is checked and then trusted: the
    lock goes away when the process does, however it dies, so a crashed run
    cannot leave the next one locked out.

    Who holds it is written beside the lock rather than inside it. The locked
    byte cannot be read by anyone else - that is what the lock means - so a
    holder that stored its name there could never be named in the refusal.

    Two runs writing one result file is not a hypothetical on this project.
    """
    import msvcrt

    path.parent.mkdir(parents=True, exist_ok=True)
    note = path.with_suffix(path.suffix + ".holder")
    handle = open(path, "a+", encoding="utf-8")
    try:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            try:
                holder = note.read_text(encoding="utf-8").strip()
            except OSError:
                holder = ""
            # `from None`: the refusal is the message, not a stack trace. The
            # OSError underneath it says nothing a person needs.
            raise SystemExit(
                f"\n  すでに {holder or '別のプロセス'} がこのスクリプトを"
                f"実行中です。\n  ロック: {path}\n"
                "  終了を待つか、そちらを止めてから実行してください。\n"
            ) from None
        note.write_text(
            f"pid {os.getpid()} since "
            f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S}",
            encoding="utf-8")
        yield path
    finally:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        handle.close()
        note.unlink(missing_ok=True)


# -------------------------------------------------------------------------
# reading the arm without waking it
# -------------------------------------------------------------------------

def read_arm(port):
    """Every joint's angle, limit and temperature, over the raw bus.

    Read-only, and before LeRobot is involved: connecting enables torque, and
    the plan has to be built and checked against the twin while the arm is
    still limp.

    Degrees are LeRobot's: zero is the midpoint of the calibrated range, which
    is what `send_action` and `get_observation` both mean by zero.
    """
    from so101.hardware.sts3215 import (
        Bus, JOINT_NAMES, MAX_ANGLE_LIMIT, MIN_ANGLE_LIMIT, PRESENT_POSITION,
        PRESENT_TEMPERATURE, PRESENT_VOLTAGE, TORQUE_ENABLE,
    )

    out = {}
    with Bus(port) as bus:
        for sid, name in JOINT_NAMES.items():
            if not bus.ping(sid):
                raise SystemExit(
                    f"  {name} (ID {sid}) が {port} で応答しません。"
                    "電源とバスを確認してください。")
            low = bus.read(sid, MIN_ANGLE_LIMIT, 2)
            high = bus.read(sid, MAX_ANGLE_LIMIT, 2)
            ticks = bus.read(sid, PRESENT_POSITION, 2)
            if None in (low, high, ticks):
                raise SystemExit(f"  {port} の {name} を読めませんでした")
            mid = (low + high) / 2
            out[name] = {
                "id": sid, "ticks": ticks, "min_ticks": low, "max_ticks": high,
                "mid_ticks": mid,
                "deg": (ticks - mid) / TICKS_PER_DEG,
                "min_deg": (low - mid) / TICKS_PER_DEG,
                "max_deg": (high - mid) / TICKS_PER_DEG,
                "torque": bus.read(sid, TORQUE_ENABLE, 1),
                "temperature": bus.read(sid, PRESENT_TEMPERATURE, 1),
                "voltage": bus.read(sid, PRESENT_VOLTAGE, 1),
            }
    return out


def joints_of(robot):
    """The arm's pose as {joint: degrees}, from the robot's own observation."""
    return {key.removesuffix(".pos"): float(value)
            for key, value in robot.get_observation().items()
            if key.endswith(".pos")}


def ticks_for(deg, arm, joint=JOINT):
    """The raw servo position LeRobot will write for this joint angle."""
    return arm[joint]["mid_ticks"] + deg * TICKS_PER_DEG


# -------------------------------------------------------------------------
# the path, as a list - the same list the twin replays and the arm is sent
# -------------------------------------------------------------------------

def waypoints(start, goal, step_deg=STEP_DEG):
    """Poses from `start` to `goal`, no joint moving more than `step_deg` at a time.

    Computed once, from where the arm is, and then not recomputed. That is what
    makes the twin's replay worth anything: if the waypoints were re-derived
    from wherever the arm had got to, the path actually flown would not be the
    path that was cleared.

    Every pose carries every joint in `goal`, so a joint that is not supposed to
    move is commanded to stay rather than left to the servo's memory.
    """
    moved = {name: goal[name] - start[name] for name in goal}
    biggest = max((abs(v) for v in moved.values()), default=0.0)
    steps = max(1, int(round(biggest / step_deg)))
    return [{name: start[name] + moved[name] * step / steps for name in goal}
            for step in range(1, steps + 1)]


class Support:
    """A box standing on the table, holding the arm up while the torque is off.

    It stays where it is for the whole run - nobody reaches in to take it away
    once the torque is on - so the arm has to leave it and then keep away from
    it, and that has to be checked against the path rather than assumed from
    where it was put.

    Axis-aligned, because a book or a block on a table is, and because two edge
    positions and a height are what a person can actually measure.
    """

    PATH = Path("data/bench_support.json")

    def __init__(self, x0, x1, y0, y1, top, label=""):
        self.x0, self.x1 = sorted((float(x0), float(x1)))
        self.y0, self.y1 = sorted((float(y0), float(y1)))
        self.top = float(top)
        self.label = label

    @classmethod
    def parse(cls, text):
        """`x0,x1,y0,y1,top` in millimetres, in the arm's own frame."""
        parts = [float(v) / 1000 for v in text.split(",")]
        if len(parts) != 5:
            raise SystemExit(
                "\n  --support は mm で 5 つの数値です: x0,x1,y0,y1,top\n"
                "  ベース中心を原点に、x は前方、y は左方、top は机面からの高さ。\n"
                "  例: --support 120,220,-60,60,55\n")
        return cls(*parts)

    def distance(self, points, table_z):
        """Smallest distance from any of `points` to the box, in metres.

        Zero while something is inside it or resting on it, which is the normal
        state at the start of a run: the arm is sitting on this thing.
        """
        import numpy as np

        points = np.asarray(points, float).reshape(-1, 3)
        dx = np.maximum(np.maximum(self.x0 - points[:, 0],
                                   points[:, 0] - self.x1), 0.0)
        dy = np.maximum(np.maximum(self.y0 - points[:, 1],
                                   points[:, 1] - self.y1), 0.0)
        dz = np.maximum(np.maximum(table_z - points[:, 2],
                                   points[:, 2] - (table_z + self.top)), 0.0)
        return float(np.sqrt(dx * dx + dy * dy + dz * dz).min())

    def save(self, path=None):
        path = Path(path or self.PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "x0_m": self.x0, "x1_m": self.x1, "y0_m": self.y0,
            "y1_m": self.y1, "top_m": self.top, "label": self.label,
            "written": datetime.now().astimezone().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path=None):
        path = Path(path or cls.PATH)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        support = cls(data["x0_m"], data["x1_m"], data["y0_m"], data["y1_m"],
                      data["top_m"], data.get("label", ""))
        support.written = data.get("written")
        return support

    def describe(self):
        return (f"x {1000*self.x0:+.0f}..{1000*self.x1:+.0f}、"
                f"y {1000*self.y0:+.0f}..{1000*self.y1:+.0f}、"
                f"机面から高さ {1000*self.top:.0f} mm")

    def as_dict(self):
        return {"x0_mm": round(1000 * self.x0, 1),
                "x1_mm": round(1000 * self.x1, 1),
                "y0_mm": round(1000 * self.y0, 1),
                "y1_mm": round(1000 * self.y1, 1),
                "top_mm": round(1000 * self.top, 1), "label": self.label}


class Table:
    """The table, and how far the arm is from it. One height, one simulator.

    Built once and passed around rather than re-derived: a clearance figure and
    a workspace map that disagree about where the table is are not measuring the
    same thing, and four different heights were in circulation before
    `so101.sim.table_height()` existed.
    """

    def __init__(self, jaws_deg, support=None):
        from so101.sim import SO101Sim, table_height

        self.z = table_height()
        self.jaws = jaws_deg
        self.support = support
        self.sim = SO101Sim(table_z=self.z)
        self.ignore = tuple(f"block_{i}" for i in range(16)) + \
            tuple(f"block_{i}_geom" for i in range(16))
        self._meshes = None

    def _mesh_index(self):
        """(geom index, body, vertices) for every arm mesh, worked out once."""
        import mujoco

        if self._meshes is None:
            self._meshes = []
            for index in range(self.sim.model.ngeom):
                body = mujoco.mj_id2name(
                    self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                    self.sim.model.geom_bodyid[index])
                if body not in MOVING_BODIES:
                    continue
                mesh = self.sim.model.geom_dataid[index]
                if mesh < 0:
                    continue
                start = self.sim.model.mesh_vertadr[mesh]
                count = self.sim.model.mesh_vertnum[mesh]
                self._meshes.append(
                    (index, body, self.sim.model.mesh_vert[start:start + count]))
        return self._meshes

    def survey(self, pose):
        """Everything geometric about one pose, from a single pass over the meshes.

        One pass rather than six: the lowest point, the gripper's own lowest
        point and the distance to the support all want the same vertices in the
        same world positions, and walking the meshes is the whole cost.
        """
        import numpy as np

        self.sim.set_joints(pose, gripper_deg=self.jaws)
        lowest, gripper_lowest = np.inf, np.inf
        nearest = np.inf
        for index, body, vertices in self._mesh_index():
            world = vertices @ np.array(
                self.sim.data.geom_xmat[index]).reshape(3, 3).T \
                + self.sim.data.geom_xpos[index]
            low = float(world[:, 2].min())
            lowest = min(lowest, low)
            if body in ("gripper", "moving_jaw_so101_v1"):
                gripper_lowest = min(gripper_lowest, low)
            if self.support is not None:
                nearest = min(nearest, self.support.distance(world, self.z))
        contacts = {f"{a}/{b}" for a, b, _ in self.sim.collisions(
            ignore=self.ignore) if "table" not in (a, b)}
        return {"gap": lowest - self.z, "gripper_gap": gripper_lowest - self.z,
                "support": None if self.support is None else nearest,
                "contacts": contacts}

    def support_gap(self, pose):
        return self.survey(pose)["support"]

    def gripper_gap(self, pose):
        """The gripper assembly's own clearance, whether or not it is lowest.

        Tracked separately because the global minimum moves to the shoulder - a
        fixed body that never gets any closer - as soon as the wrist comes up,
        and after that the global figure stops saying anything about the part
        that can actually hit the table.
        """
        return self.survey(pose)["gripper_gap"]

    def look(self, pose):
        """(clearance in metres, self-contacts) for a pose."""
        seen = self.survey(pose)
        return seen["gap"], seen["contacts"]

    def gap(self, pose):
        return self.survey(pose)["gap"]


def sensitivities(table, start, probe_deg=2.0):
    """How much each joint raises or lowers the gripper, from where the arm is.

    Measured, not assumed. Which way is up depends on the pose, and the whole
    reason the gripper reached the table was a path built on an assumption about
    that which stopped being true.
    """
    # The gripper's clearance, not the arm's lowest point. Once the wrist is
    # up the lowest point is the shoulder, which no joint moves - so every
    # sensitivity comes out as exactly zero and the table says nothing at all.
    here = table.gripper_gap(start)
    out = {}
    for name in LIFT_CANDIDATES:
        for sign in (+1, -1):
            pose = dict(start)
            pose[name] += sign * probe_deg
            out[(name, sign)] = (table.gripper_gap(pose) - here) / probe_deg
    return out


def plan_liftoff(table, start, upright, arm, log):
    """A first move that only raises the gripper, chosen from where the arm is.

    Returns (goal, detail). `goal` is None when the arm already has the room and
    the straight path to the posture keeps it - there is no virtue in moving
    joints for their own sake.
    """
    floor = CLEARANCE_FLOOR_MM / 1000
    here = table.gap(start)
    detail = {"start_clearance_mm": round(1000 * here, 2),
              "sensitivity_mm_per_deg": {}}

    rates = sensitivities(table, start)
    for (name, sign), rate in sorted(rates.items(), key=lambda kv: -kv[1]):
        detail["sensitivity_mm_per_deg"][f"{name} {sign:+d}"] = round(
            1000 * rate, 2)

    # Does it need one at all? Only if the straight run to the posture would
    # take it below the floor.
    straight = waypoints(start, upright)
    least = min(min(table.gap(_between(start, point, step))
                    for step in (0.25, 0.5, 0.75, 1.0))
                for point in straight)
    least = min(least, here)
    detail["straight_min_clearance_mm"] = round(1000 * least, 2)
    if here >= floor and least >= floor:
        detail["needed"] = False
        return None, detail
    detail["needed"] = True

    # It does. Grow the best lifting direction until there is room, refusing any
    # candidate that dips on the way or touches something new.
    best = max(rates, key=lambda key: rates[key])
    if rates[best] <= 0:
        detail["why"] = "no joint raises the gripper from this pose"
        return None, detail
    detail["using"] = f"{best[0]} {best[1]:+d}"

    limits = (arm[best[0]]["min_deg"] + LIMIT_MARGIN_DEG,
              arm[best[0]]["max_deg"] - LIMIT_MARGIN_DEG)
    wanted = max(floor, CLEARANCE_WANTED_MM / 1000)
    move = 0.0
    while move < LIFTOFF_MAX_DEG:
        move += LIFTOFF_STEP_DEG
        angle = start[best[0]] + best[1] * move
        if not limits[0] <= angle <= limits[1]:
            detail["why"] = (f"{best[0]} reaches its limit at "
                             f"{best[1] * move:+.0f} deg")
            break
        goal = {best[0]: angle}
        path = waypoints(start, goal)
        gaps = [here] + [table.gap(_between(start, point, step))
                         for point in path for step in (0.5, 1.0)]
        contacts = table.look(start)[1]
        offending = set()
        for point in path:
            offending |= table.look({**start, **point})[1] - contacts
        rising = all(b >= a - 1e-6 for a, b in zip(gaps, gaps[1:]))
        if offending or not rising:
            detail["why"] = ("self-collision" if offending
                             else "the clearance dips on the way")
            break
        if gaps[-1] >= wanted:
            detail["move_deg"] = round(best[1] * move, 1)
            detail["reaches_mm"] = round(1000 * gaps[-1], 2)
            return goal, detail
    detail.setdefault("why", "could not reach the wanted clearance")
    return None, detail


def report_liftoff(detail, liftoff, log):
    """Why the lift-off is what it is, in the order a person would ask."""
    log(f"    {pad('開始時のクリアランス', 30)}"
        f"{detail['start_clearance_mm']:+.1f} mm")
    log(f"    {pad('そのまま姿勢へ向かった場合の最小', 30)}"
        f"{detail['straight_min_clearance_mm']:+.1f} mm")
    log("    各関節が 1 deg あたりグリッパを上下させる量"
        "（この姿勢で実測。固定値ではありません）:")
    for name, rate in detail["sensitivity_mm_per_deg"].items():
        arrow = "↑" if rate > 0.05 else ("↓" if rate < -0.05 else " ")
        log(f"      {pad(name, 22)}{rate:>+7.2f} mm/deg  {arrow}")
    if not detail["needed"]:
        log("    → 離陸区間は不要です。すでに余裕があり、姿勢まで下回りません。")
        return
    if liftoff is None:
        log(f"    → *** 離陸経路を作れませんでした: {detail.get('why')} ***")
        return
    log(f"    → {detail['using']} を {detail['move_deg']:+.0f} deg。"
        f"クリアランスは {detail['reaches_mm']:+.1f} mm になります。")
    log("      上げる方向の関節だけを使い、下げる方向のものは余裕が"
        "できるまで指令しません。")


def resolve_support(args, log=None):
    """The support the run should assume: the flag, the saved file, or none.

    Saved rather than retyped every run, because five numbers retyped from
    memory is a way to check the path against a support that is not the one on
    the bench. Printed every run for the same reason.
    """
    if getattr(args, "no_support", False):
        return None
    if getattr(args, "support", None):
        support = Support.parse(args.support)
        support.save()
        if log:
            log(f"  支持物を保存しました: {Support.PATH}")
        return support
    return Support.load()


def where(port, log, support=None):
    """Read-only: where the arm is and how much room it has. Repeat while lifting."""
    arm = read_arm(port)
    start = {name: arm[name]["deg"] for name in ARM_JOINTS}
    table = Table(arm["gripper"]["deg"], support)
    seen = table.survey(start)
    gap, contacts = seen["gap"], seen["contacts"]

    log(f"\n  TABLE_Z {1000*table.z:+.2f} mm（ベース底面）、"
        f"jaws {arm['gripper']['deg']:+.1f} deg")
    log("  " + pad("joint", 16) + rpad("現在角", 9) + rpad("ticks", 8)
        + rpad("限界まで", 12))
    for name in ARM_JOINTS:
        j = arm[name]
        room = min(j["deg"] - j["min_deg"], j["max_deg"] - j["deg"])
        log(f"  {pad(name, 16)}{j['deg']:>+9.2f}{j['ticks']:>8}{room:>9.1f} deg")
    lows = {body: table.sim.lowest_point(bodies=(body,))
            for body in MOVING_BODIES}
    body = min(lows, key=lows.get)
    # The gripper assembly's own lowest point, reported whether or not it is the
    # lowest thing on the arm. It is the feature a person can actually put a
    # ruler under, so it is the one worth quoting at every pose - the global
    # minimum wanders to the shoulder as soon as the wrist comes up, and then
    # the two numbers stop being about the same thing.
    jaws_low = min(lows["gripper"], lows["moving_jaw_so101_v1"]) - table.z
    verdict = ("十分です" if 1000 * gap >= CLEARANCE_WANTED_MM else
               "下限は満たします" if 1000 * gap >= CLEARANCE_FLOOR_MM else
               "*** 足りません ***")
    log(f"\n  机までのクリアランス {1000*gap:+.1f} mm   {verdict}")
    log(f"    アーム全体の最下点は {body}")
    log(f"\n  定規で測るならここ:")
    log(f"    グリッパ組立（指先）の最下点   モデル予測 机面から "
        f"{1000*jaws_low:+.1f} mm")
    log("    これはモデルの予測であって、実測値ではありません。")
    log("    実測との差が、そのまま Sim→Real の差です。")
    log(f"    下限 {CLEARANCE_FLOOR_MM:.0f} mm / 望ましくは "
        f"{CLEARANCE_WANTED_MM:.0f} mm 以上")
    if contacts:
        log(f"    自身に接触しています: {sorted(contacts)}")
    if support is not None:
        log(f"\n  支持物 {support.describe()}")
        log(f"    いまのアームとの距離 {1000*seen['support']:+.1f} mm"
            f"（0 なら乗っています）")
        if getattr(support, "written", None):
            log(f"    {Support.PATH} に {support.written} 付けで保存された値です。")
            log("    実物と違っていれば --support で入れ直してください。")
    if 1000 * gap < CLEARANCE_WANTED_MM:
        log("\n  どちらへ動かせば上がるか（この姿勢で実測）:")
        rates = sensitivities(table, start)
        for (name, sign), rate in sorted(rates.items(),
                                         key=lambda kv: -kv[1])[:3]:
            log(f"    {pad(f'{name} {sign:+d}', 22)}{1000*rate:>+7.2f} mm/deg")
    log("")
    return 1000 * gap


def _between(start, point, fraction):
    """The pose `fraction` of the way from `start` to `point`."""
    pose = dict(start)
    pose.update({n: start[n] + (point[n] - start[n]) * fraction for n in point})
    return pose


def check_path_shape(stages, start, log):
    """Does the waypoint list step smoothly, or does it jump between branches?

    Nothing here solves inverse kinematics, so a configuration flip cannot
    happen by construction - the waypoints are a straight line in joint space.
    That is a claim worth checking rather than asserting: it is exactly the
    property that would be lost if anything later planned these poses instead
    of interpolating them, and the check costs nothing.

    A jump would show as a step larger than the spacing, or as a joint that
    reverses direction partway through a stage.
    """
    ok = True
    log("    " + pad("stage", 30) + rpad("最大の 1 歩", 14)
        + rpad("向きの反転", 12) + "   判定")
    for stage in stages:
        label, path = stage.label, stage.path
        biggest, reversals, where = 0.0, 0, ""
        previous = dict(start)
        directions = {}
        for point in path:
            for name, value in point.items():
                step = value - previous[name]
                if abs(step) > biggest:
                    biggest, where = abs(step), name
                if abs(step) > 1e-6:
                    sign = 1 if step > 0 else -1
                    if directions.get(name, sign) != sign:
                        reversals += 1
                    directions[name] = sign
            previous = dict(previous)
            previous.update(point)
        start = dict(previous)
        good = biggest <= STEP_DEG * 1.05 + 1e-6 and reversals == 0
        ok = ok and good
        log(f"    {pad(label, 30)}{biggest:>9.2f} deg"
            f"{f' ({where})' if where else '':<12}{reversals:>4}"
            f"       {'滑らか' if good else '*** 不連続 ***'}")
    log(f"    （1 歩の上限 {STEP_DEG:.1f} deg、関節空間の直線補間なので"
        f"反転は 0 のはずです）")
    return ok


def check_limits(stages, arm, log):
    """Is every waypoint inside the servos' own limits? Returns True if so."""
    worst = {}
    for stage in stages:
        label, path = stage.label, stage.path
        for index, point in enumerate(path, 1):
            for name, value in point.items():
                low = arm[name]["min_deg"] + LIMIT_MARGIN_DEG
                high = arm[name]["max_deg"] - LIMIT_MARGIN_DEG
                room = min(value - low, high - value)
                if name not in worst or room < worst[name][0]:
                    worst[name] = (room, value, label, index)
    ok = True
    log("    " + pad("joint", 16) + rpad("最も端に近い waypoint", 24)
        + rpad("限界まで", 12) + "   stage")
    for name in ARM_JOINTS:
        if name not in worst:
            continue
        room, value, label, index = worst[name]
        if room < 0:
            ok = False
        log(f"    {pad(name, 16)}{value:>+16.2f} deg{room:>16.2f} deg"
            f"   {label} #{index}"
            + ("" if room >= 0 else "   *** 限界外 ***"))
    log(f"    （ファームウェア限界から {LIMIT_MARGIN_DEG:.1f} deg 内側を要求）")
    return ok


def replay(stages, start, table, log, fine=6):
    """Fly the exact waypoint list in MuJoCo. Returns (clear, report).

    `fine` subdivides each waypoint interval further, because a collision can
    happen between two waypoints as easily as at one.

    Three things are checked and they fail differently.

    A contact that is *new* compared with where the arm already rests is one the
    path created; the contacts the start pose already has are the arm leaning on
    itself as it sits there, and it leaves them on the first move.

    Clearance is judged against the floor everywhere, and additionally against
    the start: no path can change where the arm begins, and the run refuses to
    begin below the floor at all. What a path can still do is make it worse.

    And during the first stage the property that matters is not a threshold but
    a direction: leaving the table, the clearance may only increase.
    """
    def look(pose):
        gap, contacts = table.look(pose)
        return contacts, gap

    here = dict(start)
    baseline, parked = look(here)
    table_z = table.z
    log(f"    TABLE_Z        {1000*table_z:+.2f} mm（ベース底面。so101.sim."
        f"table_height()）")
    log(f"    MIN_CLEARANCE  {CLEARANCE_FLOOR_MM:.1f} mm"
        f"（開始時にも、離陸後にも守る下限）")
    log(f"    開始時のクリアランス {1000*parked:+.1f} mm"
        f"{'' if parked >= CLEARANCE_FLOOR_MM / 1000 else '  *** 不足 ***'}")
    if baseline:
        log(f"    この姿勢はすでに自身に接触しています: {sorted(baseline)}")
    log("")
    log("    " + pad("stage", 30) + pad("種別", 9) + rpad("点", 5)
        + rpad("最小 全体", 11) + rpad("グリッパ", 11) + "   判定")

    opening = table.survey(start)
    start_grip = opening["gripper_gap"]
    start_support = opening["support"]
    tolerance = APPROACH_TOLERANCE_MM / 1000
    floor = CLEARANCE_FLOOR_MM / 1000
    support_tolerance = SUPPORT_TOLERANCE_MM / 1000
    clear = True
    report = {"table_z_mm": round(1000 * table_z, 3),
              "clearance_floor_mm": CLEARANCE_FLOOR_MM,
              "clearance_wanted_mm": CLEARANCE_WANTED_MM,
              "approach_tolerance_mm": APPROACH_TOLERANCE_MM,
              "start_clearance_mm": round(1000 * parked, 2),
              "start_gripper_clearance_mm": round(1000 * start_grip, 2),
              "stages": []}
    for stage in stages:
        label, path, kind = stage.label, stage.path, stage.kind
        least, least_grip = float("inf"), float("inf")
        least_support, support_closed_at = float("inf"), None
        offenders, worst_at, closed_at, below_at = set(), None, None, None
        support_previous = start_support
        previous = dict(here)
        for target in path:
            for step in range(1, fine + 1):
                pose = dict(here)
                pose.update({n: previous[n] + (target[n] - previous[n])
                             * step / fine for n in target})
                seen = table.survey(pose)
                found, gap, grip = seen["contacts"], seen["gap"], seen["gripper_gap"]
                if gap < least:
                    least, worst_at = gap, dict(pose)
                if grip < least_grip:
                    least_grip = grip
                if seen["support"] is not None:
                    near = seen["support"]
                    least_support = min(least_support, near)
                    # Leaving the support is allowed to start at zero - the arm
                    # is on it - but from there the distance may only grow, and
                    # once grown it may not come back.
                    if support_closed_at is None \
                            and near < support_previous - support_tolerance:
                        support_closed_at = (dict(pose), support_previous, near)
                    support_previous = max(support_previous, near)
                # A transit and a sweep fail differently, and judging them
                # alike was wrong: turning the wrist down from the posture
                # lowers the gripper by 120 mm on purpose, and calling that
                # "approaching the table" would refuse the only move the whole
                # phase exists to make. So a transit may not get closer to the
                # table than it started - the criterion that survives the
                # model's unknown offset - and a sweep is held to the floor
                # instead, which is what it is for.
                if kind == "transit":
                    if closed_at is None and (gap < parked - tolerance
                                              or grip < start_grip - tolerance):
                        closed_at = (dict(pose), gap, grip)
                elif below_at is None and min(gap, grip) < floor:
                    below_at = (dict(pose), gap, grip)
                offenders |= (found - baseline)
            previous = dict(previous)
            previous.update(target)
            here.update(target)

        support_ok = support_closed_at is None
        ok = (not offenders and closed_at is None and below_at is None
              and support_ok)
        verdict = ("机へ近づきません" if kind == "transit"
                   else f"{CLEARANCE_FLOOR_MM:.0f} mm を保ちます")
        log(f"    {pad(label, 30)}{pad(kind, 9)}{len(path):>5}"
            f"{1000*least:>11.1f}{1000*least_grip:>11.1f}   "
            f"{verdict if ok else '*** 不可 ***'}")
        report["stages"].append({
            "stage": label, "kind": kind, "waypoints": len(path),
            "min_clearance_mm": round(1000 * least, 2),
            "min_gripper_clearance_mm": round(1000 * least_grip, 2),
            "min_support_distance_mm": (None if table.support is None
                                        else round(1000 * least_support, 2)),
            "never_returns_to_support": None if table.support is None
                                        else support_ok,
            "never_approaches": None if kind != "transit" else closed_at is None,
            "above_floor": None if kind == "transit" else below_at is None,
            "no_new_contacts": not offenders, "ok": bool(ok)})
        if offenders:
            log(f"      新たな自己干渉: {sorted(offenders)}")
            clear = False
        if below_at is not None:
            pose, gap, grip = below_at
            log(f"      下限 {CLEARANCE_FLOOR_MM:.0f} mm を割ります: 全体 "
                f"{1000*gap:+.1f} mm、グリッパ {1000*grip:+.1f} mm。そのときの姿勢:")
            log("        " + "、".join(f"{n} {pose[n]:+.1f}" for n in ARM_JOINTS))
            clear = False
        if support_closed_at is not None:
            pose, before, after = support_closed_at
            log(f"      支持物へ戻ります（{1000*before:.1f} → {1000*after:.1f} mm）。"
                f"そのときの姿勢:")
            log("        " + "、".join(f"{n} {pose[n]:+.1f}" for n in ARM_JOINTS))
            clear = False
        if closed_at is not None:
            pose, gap, grip = closed_at
            log(f"      机へ近づきます: 全体 {1000*parked:+.1f} → "
                f"{1000*gap:+.1f} mm、グリッパ {1000*start_grip:+.1f} → "
                f"{1000*grip:+.1f} mm。そのときの姿勢:")
            log("        " + "、".join(f"{n} {pose[n]:+.1f}" for n in ARM_JOINTS))
            clear = False

    if table.support is not None:
        report["support"] = table.support.as_dict()
        report["start_support_distance_mm"] = round(1000 * start_support, 2)
        report["min_support_distance_mm"] = min(
            s["min_support_distance_mm"] for s in report["stages"])
        report["support_margin_mm"] = SUPPORT_MARGIN_MM
        ends = table.survey({**start, **stages[-1].path[-1]})["support"]
        report["end_support_distance_mm"] = round(1000 * ends, 2)
        report["leaves_support"] = bool(1000 * ends >= SUPPORT_MARGIN_MM)
        if not report["leaves_support"]:
            clear = False
    report["min_clearance_mm"] = min(s["min_clearance_mm"]
                                     for s in report["stages"])
    report["min_gripper_clearance_mm"] = min(s["min_gripper_clearance_mm"]
                                             for s in report["stages"])
    report["never_approaches"] = all(s["never_approaches"]
                                     for s in report["stages"]
                                     if s["never_approaches"] is not None)
    log(f"\n    経路全体の最小   全体 {report['min_clearance_mm']:+.1f} mm、"
        f"グリッパ {report['min_gripper_clearance_mm']:+.1f} mm")
    log(f"    開始時からの悪化   "
        + ("なし（どの姿勢でも机へ近づきません）"
           if report["never_approaches"] else "*** あり ***"))
    if table.support is not None:
        log(f"\n    支持物 {table.support.describe()}")
        log(f"      開始時の距離     {report['start_support_distance_mm']:+.1f} mm"
            f"（0 なら乗っています）")
        log(f"      経路中の最小     {report['min_support_distance_mm']:+.1f} mm")
        log(f"      終了時の距離     {report['end_support_distance_mm']:+.1f} mm"
            f"   要求 {SUPPORT_MARGIN_MM:.0f} mm 以上   "
            f"{'OK' if report['leaves_support'] else '*** 不足 ***'}")
        log("      判定は「離れたあと戻らないこと」と「最後に離れていること」です。")
    return clear, report


def stages_for(start, posture, target_deg, repeats, posture_only,
               posture_name="upright", liftoff=None, fraction=1.0):
    """Every move the run will make, named, as waypoint lists.

    One function, used by the twin and by the arm. When the two disagree about
    what is going to be flown, the twin's verdict is about something else.

    `liftoff` comes from `plan_liftoff`, measured from this start pose. It is
    None when the arm already has the room, and there is no virtue in moving
    joints for their own sake.
    """
    upright = dict(posture, wrist_flex=SAFE_DEG)
    out = []
    lifted = dict(start)
    if liftoff:
        joint = max(liftoff, key=lambda n: abs(liftoff[n] - start[n]))
        out.append(Stage(
            "離陸（机から離す）", waypoints(start, liftoff), "transit",
            LIFTOFF_SPEED_DEG_S, joint,
            note="グリッパを机から離す区間です。ここで動かすのは、この姿勢で"
                 "実際に持ち上がる関節だけで、下げる向きの関節は余裕ができる"
                 "まで一切指令しません。"))
        lifted.update(liftoff)
    if fraction < 1.0:
        # Part of the way and stop. The waypoints are a straight line in joint
        # space, so a fraction of it is the same line - the arm ends up
        # somewhere on the path it would have taken anyway, which is what makes
        # this a rehearsal of the real move rather than a different one.
        upright = {n: lifted[n] + (upright[n] - lifted[n]) * fraction
                   for n in upright}
        label = f"{posture_name} へ {100*fraction:.0f}% だけ"
    else:
        label = f"{posture_name} へ"
    moves = {n: abs(upright[n] - lifted[n]) for n in upright}
    out.append(Stage(label, waypoints(lifted, upright), "transit",
                     TRANSIT_SPEED_DEG_S, max(moves, key=moves.get)))
    if posture_only:
        return out
    here = dict(upright)
    for repeat in range(1, repeats + 1):
        out.append(Stage(f"{repeat} 回目: wrist_flex → {target_deg:+.0f}",
                         waypoints(here, {JOINT: target_deg}), "sweep",
                         WRIST_SPEED_DEG_S, JOINT))
        here[JOINT] = target_deg
        out.append(Stage(f"{repeat} 回目: wrist_flex → {SAFE_DEG:+.0f}",
                         waypoints(here, {JOINT: SAFE_DEG}), "sweep",
                         WRIST_SPEED_DEG_S, JOINT))
        here[JOINT] = SAFE_DEG
    return out


# -------------------------------------------------------------------------
# watching, while it moves
# -------------------------------------------------------------------------

class Watch:
    """Samples the arm and stops the run when something stops being ordinary.

    The three numbers a reader has to be able to tell apart are kept in three
    columns and never mixed, because mixing them is how the first attempt
    reported a 35 degree fault that was really a change of reference:

      nominal_target_deg  where the phase is ultimately going
      waypoint_target_deg the confirmed step being walked to right now
      sent_goal_deg       what went into send_action for this sample

    Every sample is written as it is taken rather than collected and saved at
    the end: the samples worth having most are the ones from the run that did
    not finish.
    """

    FIELDS = ("timestamp", "elapsed_s", "phase", "repeat", "waypoint",
              "subject", "nominal_target_deg", "waypoint_target_deg",
              "sent_goal_deg", "measured_deg", "tracking_error_deg",
              "remaining_to_waypoint_deg", "remaining_to_nominal_deg",
              "velocity_deg_s", "load", "current", "temperature_c",
              "voltage_v", "torque_enable", "hold_time_s", "arrival", "stall",
              "max_joint_error_deg", "worst_joint",
              *(f"{name}_deg" for name in ARM_JOINTS),
              "status", "notes")

    def __init__(self, robot, writer, log):
        self.robot = robot
        self.writer = writer
        self.log = log
        self.started = time.perf_counter()
        self.comms_failures = 0
        self.current_ever_nonzero = False
        self.rows = []
        self.history = []       # (t, {joint: measured})
        self.reset("idle", subject=JOINT, nominal=0.0)

    # -- what the current phase is about ----------------------------------

    def reset(self, phase, subject, nominal, repeat=0):
        self.phase = phase
        self.subject = subject
        self.nominal = nominal
        self.repeat = repeat
        self.waypoint_index = 0
        self.waypoint = {}
        self.sent = {}
        self.stalled_since = None
        self.retreat_since = None

    # -- reading ----------------------------------------------------------

    def read_one(self, name, joint=None, default=None):
        try:
            value = self.robot.bus.read(name, joint or self.subject,
                                        normalize=False)
            self.comms_failures = 0
            return value
        except Exception:  # noqa: BLE001 - a dropped packet is not a fault yet
            self.comms_failures += 1
            if self.comms_failures >= COMMS_ABORT:
                raise Abort(f"{self.comms_failures} 回連続で読み出しに失敗 "
                            f"({name})")
            return default

    def velocity(self, pose, now):
        """Degrees per second of the subject joint, over a short window."""
        older = [h for h in self.history if now - h[0] >= VELOCITY_WINDOW_S]
        if not older:
            return 0.0
        when, then = older[-1]
        span = now - when
        if span <= 0 or self.subject not in then:
            return 0.0
        return (pose[self.subject] - then[self.subject]) / span

    def sample(self, hold_time=None, status="ok", notes="", arrival="",
               judge=True):
        now = time.perf_counter()
        try:
            pose = joints_of(self.robot)
            self.comms_failures = 0
        except Exception as error:  # noqa: BLE001
            self.comms_failures += 1
            if self.comms_failures >= COMMS_ABORT:
                raise Abort(f"{self.comms_failures} 回連続で位置読み出しに失敗 "
                            f"({error})") from error
            return None

        self.history.append((now, pose))
        self.history = [h for h in self.history if now - h[0] <= 3.0]

        load = self.read_one("Present_Load", default=0)
        current = self.read_one("Present_Current")
        temperature = self.read_one("Present_Temperature")
        voltage = self.read_one("Present_Voltage")
        torque = self.read_one("Torque_Enable")
        if current:
            self.current_ever_nonzero = True

        measured = pose[self.subject]
        waypoint_target = self.waypoint.get(self.subject)
        sent = self.sent.get(self.subject)
        errors = {n: pose[n] - v for n, v in self.waypoint.items()}
        worst_joint = (max(errors, key=lambda n: abs(errors[n]))
                       if errors else self.subject)
        stall = self.stall_reason(pose, now)

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"),
            "elapsed_s": round(now - self.started, 3),
            "phase": self.phase, "repeat": self.repeat,
            "waypoint": self.waypoint_index, "subject": self.subject,
            "nominal_target_deg": round(self.nominal, 3),
            "waypoint_target_deg": (None if waypoint_target is None
                                    else round(waypoint_target, 3)),
            "sent_goal_deg": None if sent is None else round(sent, 3),
            "measured_deg": round(measured, 3),
            "tracking_error_deg": (None if sent is None
                                   else round(measured - sent, 3)),
            "remaining_to_waypoint_deg": (None if waypoint_target is None else
                                          round(waypoint_target - measured, 3)),
            "remaining_to_nominal_deg": round(self.nominal - measured, 3),
            "velocity_deg_s": round(self.velocity(pose, now), 3),
            "load": load, "current": current, "temperature_c": temperature,
            "voltage_v": None if voltage is None else voltage / 10,
            "torque_enable": torque,
            "hold_time_s": None if hold_time is None else round(hold_time, 2),
            "arrival": arrival, "stall": stall or "",
            "max_joint_error_deg": (round(abs(errors[worst_joint]), 3)
                                    if errors else None),
            "worst_joint": worst_joint if errors else "",
            **{f"{name}_deg": round(pose[name], 3) for name in ARM_JOINTS},
            "status": status, "notes": notes,
        }
        self.writer.writerow(row)
        self.rows.append(row)
        if judge:
            self.judge(row, stall)
        return row

    # -- deciding ---------------------------------------------------------

    def stall_reason(self, pose, now):
        """Which joint is short of its waypoint and no longer moving, if any.

        This is the MOVING check, and it is not a tracking-error check. A joint
        on its way to a waypoint is behind by design; a joint that has stopped
        while still short of one is not on its way to anywhere.
        """
        if not self.waypoint:
            self.stalled_since = None
            return ""
        stuck = []
        for name, target in self.waypoint.items():
            if abs(target - pose[name]) <= STALL_REMAINING_DEG:
                continue
            older = [h for h in self.history if now - h[0] >= STALL_WINDOW_S]
            if not older:
                continue
            when, then = older[-1]
            span = now - when
            if span > 0 and abs(pose[name] - then[name]) / span < STALL_SPEED_DEG_S:
                stuck.append(f"{name} {abs(target - pose[name]):.1f}deg short")
        if not stuck:
            self.stalled_since = None
            return ""
        if self.stalled_since is None:
            self.stalled_since = now
        if now - self.stalled_since >= STALL_PERSIST_S:
            return ", ".join(stuck)
        return ""

    def judge(self, row, stall):
        load = abs(row["load"] or 0)
        if load > LOAD_ABORT:
            raise Abort(f"負荷 {row['load']} が {LOAD_ABORT} を超えました"
                        f"（この掃引の予測値は 40 前後）")
        temperature = row["temperature_c"]
        if temperature is not None and temperature > TEMPERATURE_ABORT_C:
            raise Abort(f"温度 {temperature} C が {TEMPERATURE_ABORT_C} C を"
                        f"超えました")
        if stall:
            raise Abort(f"目標へ進まなくなりました: {stall}"
                        f"（{STALL_SPEED_DEG_S:.2f} deg/s 未満が "
                        f"{STALL_WINDOW_S + STALL_PERSIST_S:.1f} s 継続）")
        if self.waypoint:
            self.check_retreat(row)
        # HOLDING: only once a waypoint is confirmed reached does distance from
        # it mean anything - and then it means a great deal.
        if self.phase == "hold" and row["sent_goal_deg"] is not None:
            if abs(row["measured_deg"] - row["sent_goal_deg"]) \
                    > HOLD_ERROR_ABORT_DEG:
                raise Abort(
                    f"保持中に指令から {row['measured_deg'] - row['sent_goal_deg']:+.2f} "
                    f"deg 外れました（上限 {HOLD_ERROR_ABORT_DEG:.1f} deg）")

    def check_retreat(self, row):
        """Moving away from the waypoint is never a lag."""
        remaining = row["remaining_to_waypoint_deg"]
        if remaining is None:
            return
        now = time.perf_counter()
        older = [r for r in self.rows
                 if r["remaining_to_waypoint_deg"] is not None
                 and r["waypoint"] == row["waypoint"]
                 and row["elapsed_s"] - r["elapsed_s"] >= RETREAT_WINDOW_S]
        if not older:
            return
        grew = abs(remaining) - abs(older[-1]["remaining_to_waypoint_deg"])
        if grew > RETREAT_DEG:
            raise Abort(f"目標から遠ざかっています: 残り "
                        f"{abs(older[-1]['remaining_to_waypoint_deg']):.1f} → "
                        f"{abs(remaining):.1f} deg")


# -------------------------------------------------------------------------
# moving
# -------------------------------------------------------------------------

def arrived(pose, target, tolerance):
    """(ok, worst joint, its error) for a pose against a target pose."""
    errors = {name: pose[name] - value for name, value in target.items()}
    worst = max(errors, key=lambda n: abs(errors[n]))
    return abs(errors[worst]) <= tolerance, worst, errors[worst]


def walk(robot, watch, path, phase, subject, nominal, repeat, speed, log,
         label=""):
    """Walk the waypoint list, confirming each one before sending the next.

    Between waypoints the command is interpolated at 30 Hz so the servo is never
    handed a step change; at each waypoint the arm is given time to actually get
    there and is asked whether it did. A waypoint that does not arrive stops the
    run - it does not become the next waypoint's starting point, which is how a
    52 degree error walked into a phase that thought it had arrived.
    """
    watch.reset(phase, subject, nominal, repeat)
    for index, target in enumerate(path, 1):
        watch.waypoint_index = index
        watch.waypoint = dict(target)
        start = joints_of(robot)
        move = max((abs(target[n] - start[n]) for n in target), default=0.0)
        steps = max(1, int(round(max(move / speed, 1.0 / FPS) * FPS)))
        for step in range(1, steps + 1):
            watch.sent = {n: start[n] + (target[n] - start[n]) * step / steps
                          for n in target}
            robot.send_action({f"{n}.pos": v for n, v in watch.sent.items()})
            time.sleep(1.0 / FPS)
            if step % max(1, int(FPS / SAMPLE_HZ)) == 0:
                watch.sample()

        # Now it has been asked for the whole step. Give it time to get there,
        # and say so either way.
        watch.sent = dict(target)
        deadline = time.perf_counter() + WAYPOINT_TIMEOUT_S
        while True:
            row = watch.sample()
            pose = joints_of(robot) if row is None else {
                n: row[f"{n}_deg"] for n in ARM_JOINTS}
            ok, worst, error = arrived(pose, target, ARRIVE_DEG)
            if ok:
                watch.sample(arrival="reached")
                break
            if time.perf_counter() > deadline:
                watch.sample(arrival="timeout", status="abort", judge=False)
                raise Abort(
                    f"waypoint {index}/{len(path)} に {WAYPOINT_TIMEOUT_S:.0f} s "
                    f"以内に到達しませんでした: {worst} が {error:+.2f} deg ずれ"
                    f"（許容 {ARRIVE_DEG:.1f} deg）")
            time.sleep(1.0 / SAMPLE_HZ)

    # The phase as a whole, once everything has settled.
    time.sleep(0.4)
    pose = joints_of(robot)
    goal = dict(path[-1])
    ok, worst, error = arrived(pose, goal, PHASE_TOLERANCE_DEG)
    watch.sample(arrival="phase ok" if ok else "phase FAILED")
    log(f"    {pad(label or phase, 30)}到達判定 "
        f"{'OK' if ok else '*** 未到達 ***'}  最悪 {worst} {error:+.2f} deg"
        f"（許容 {PHASE_TOLERANCE_DEG:.1f} deg）")
    for name in sorted(goal):
        log(f"      {pad(name, 16)}目標 {goal[name]:+7.2f}   実測 "
            f"{pose[name]:+7.2f}   差 {pose[name] - goal[name]:+6.2f} deg")
    if not ok:
        raise Abort(f"{label or phase} が到達しませんでした: {worst} が "
                    f"{error:+.2f} deg ずれ（許容 {PHASE_TOLERANCE_DEG:.1f} deg）")
    return pose


def hold_at(robot, watch, target_deg, seconds, repeat, log):
    """Sit at a confirmed waypoint and keep sampling. Returns the samples."""
    watch.reset("hold", JOINT, target_deg, repeat)
    watch.waypoint = {JOINT: target_deg}
    watch.sent = {JOINT: target_deg}
    began = time.perf_counter()
    taken = []
    while time.perf_counter() - began < seconds:
        row = watch.sample(hold_time=time.perf_counter() - began)
        if row is not None:
            taken.append(row)
        time.sleep(1.0 / SAMPLE_HZ)
    return taken


def summarise(rows, label):
    """mean / median / p95 / worst of what matters, over a set of samples."""
    import numpy as np

    if not rows:
        return {"label": label, "samples": 0}
    error = np.array([abs(r["tracking_error_deg"] or 0.0) for r in rows])
    load = np.array([abs(r["load"] or 0) for r in rows])
    measured = np.array([r["measured_deg"] for r in rows])
    temps = [r["temperature_c"] for r in rows if r["temperature_c"] is not None]
    return {
        "label": label, "samples": len(rows),
        "measured_mean_deg": round(float(measured.mean()), 3),
        "measured_drift_deg": round(float(measured.max() - measured.min()), 3),
        "tracking_error_mean_deg": round(float(error.mean()), 3),
        "tracking_error_median_deg": round(float(np.median(error)), 3),
        "tracking_error_p95_deg": round(float(np.percentile(error, 95)), 3),
        "tracking_error_worst_deg": round(float(error.max()), 3),
        "load_mean": round(float(load.mean()), 1),
        "load_p95": round(float(np.percentile(load, 95)), 1),
        "load_worst": int(load.max()),
        "temperature_start_c": temps[0] if temps else None,
        "temperature_end_c": temps[-1] if temps else None,
    }


# -------------------------------------------------------------------------
# the report that comes before anything moves
# -------------------------------------------------------------------------

def plan_report(args, arm, posture, stages, out_dir, lock_path, log, clear,
                clearance):
    wrist = arm[JOINT]
    target = args.to
    upright = dict(posture, wrist_flex=SAFE_DEG)

    log("\n  --- いまアームがいる場所（読み取りのみ。トルクには触れていません）---")
    log("    " + pad("joint", 15) + rpad("現在角", 9) + rpad("ticks", 8)
        + rpad("可動範囲 deg", 20) + rpad("torque", 8) + rpad("温度", 7))
    for name in ARM_JOINTS + ("gripper",):
        j = arm[name]
        travel = f"{j['min_deg']:+.2f} .. {j['max_deg']:+.2f}"
        log(f"    {name:<15}{j['deg']:>+9.2f}{j['ticks']:>8}{travel:>20}"
            f"{j['torque']:>8}{j['temperature']:>6}C")

    log(f"\n  --- 掃引を行う姿勢: {args.posture} ---")
    log("    " + pad("joint", 15) + rpad("現在", 9) + rpad("目標", 9)
        + rpad("移動量", 11))
    for name in ARM_JOINTS:
        log(f"    {name:<15}{arm[name]['deg']:>+9.2f}{upright[name]:>+9.2f}"
            f"{upright[name] - arm[name]['deg']:>+11.2f}")

    log("\n  --- 送信する waypoint 列（ツインが再生したものと同一）---")
    total = 0
    for stage in stages:
        label, path = stage.label, stage.path
        total += len(path)
        biggest = 0.0
        previous = None
        for point in path:
            if previous is not None:
                biggest = max(biggest, max(abs(point[n] - previous[n])
                                           for n in point))
            previous = point
        log(f"    {pad(label, 30)}{len(path):>4} 点   1 点あたり最大 "
            f"{biggest:.2f} deg")
    log(f"    {pad('合計', 30)}{total:>4} 点   "
        f"いずれも到達確認してから次を送ります")

    if not args.posture_only:
        log("\n  --- wrist_flex の目標 ---")
        log(f"    {pad('目標角', 28)}{target:+.2f} deg   "
            f"({ticks_for(target, arm):.0f} ticks)")
        log(f"    {pad('ファームウェア下限', 28)}{wrist['min_deg']:+.2f} deg   "
            f"({wrist['min_ticks']} ticks)   余裕 "
            f"{target - wrist['min_deg']:+.2f} deg")
        log(f"    {pad('ファームウェア上限', 28)}{wrist['max_deg']:+.2f} deg   "
            f"({wrist['max_ticks']} ticks)   余裕 "
            f"{wrist['max_deg'] - target:+.2f} deg")
        log(f"    {pad('R1 の上限', 28)}{LIMIT_DEG:+.2f} deg   "
            f"余裕 {LIMIT_DEG - abs(target):+.2f} deg")
        log("\n    joint 角 → servo 生位置の換算 "
            f"（0 deg = calibration 中点 {wrist['mid_ticks']:.0f} ticks）")
        for deg in sorted({0.0, target / 2, target}):
            log(f"      {deg:+8.2f} deg  →  {ticks_for(deg, arm):8.1f} ticks")

    log("\n  --- 速度と刻み ---")
    log(f"    {pad('waypoint 間隔', 28)}{STEP_DEG:.1f} deg（最速関節基準）")
    log(f"    {pad('waypoint 到達判定', 28)}±{ARRIVE_DEG:.1f} deg 以内、"
        f"最大 {WAYPOINT_TIMEOUT_S:.0f} s 待つ")
    log(f"    {pad('phase 到達判定', 28)}全関節 ±{PHASE_TOLERANCE_DEG:.1f} deg 以内")
    log(f"    {pad('waypoint 間の補間', 28)}{FPS} Hz、"
        f"離陸 {LIFTOFF_SPEED_DEG_S:.1f} / 移動 {TRANSIT_SPEED_DEG_S:.0f} / "
        f"手首 {WRIST_SPEED_DEG_S:.0f} deg/s")
    log(f"    {pad('Goal_Velocity', 28)}{GOAL_VELOCITY_DEG_S:.0f} deg/s "
        f"({GOAL_VELOCITY_DEG_S * TICKS_PER_DEG:.0f} ticks/s)、"
        f"終了時に 0 へ戻します")
    log(f"    {pad('max_relative_target', 28)}使用しません（None）")
    log("      前回はこれが 2.0 deg で、サーボが見る位置偏差を 2 deg に抑え、")
    log("      トルクを頭打ちにして elbow_flex を 52.8 deg 手前で止めました。")
    log("      速度は補間と Goal_Velocity で決めます。")
    if not args.posture_only:
        log(f"    {pad('保持', 28)}{args.hold:.1f} s、{SAMPLE_HZ} Hz で記録")
        log(f"    {pad('反復', 28)}{args.repeats} 回")

    log("\n  --- 机までのクリアランス ---")
    log(f"    {pad('TABLE_Z', 28)}{clearance['table_z_mm']:+.2f} mm"
        f"（ベース底面。so101.sim.table_height() ただ一箇所から）")
    log(f"    {pad('MIN_CLEARANCE', 28)}{clearance['clearance_floor_mm']:.1f} mm"
        f"（開始時にも、離陸後にも守る下限。"
        f"望ましくは {clearance['clearance_wanted_mm']:.0f} mm 以上）")
    measured = clearance.get("measured_clearance_mm")
    log(f"    {pad('開始時（モデル予測）', 28)}"
        f"{clearance['start_clearance_mm']:+.1f} mm"
        f"（グリッパ {clearance['start_gripper_clearance_mm']:+.1f} mm）")
    if measured is None:
        log(f"    {pad('開始時（実測）', 28)}指定なし。モデルの値で判定します")
    else:
        log(f"    {pad('開始時（実測・定規）', 28)}{measured:+.1f} mm"
            f"   下限 {clearance['measured_floor_mm']:.0f} mm   "
            f"{'OK' if clearance.get('start_ok') else '*** 不足 ***'}")
        log("      実測は開始条件の判定にだけ使います。ツインの干渉判定と")
        log("      「机へ近づかない」判定は、この値と無関係に効いています。")
    log(f"    {pad('経路全体の最小', 28)}"
        f"全体 {clearance['min_clearance_mm']:+.1f} mm、"
        f"グリッパ {clearance['min_gripper_clearance_mm']:+.1f} mm")
    log(f"    {pad('机へ近づくか', 28)}"
        + ("どの姿勢でも近づきません" if clearance["never_approaches"]
           else "*** 近づきます — 実行を拒否します ***")
        + f"（許容 {clearance['approach_tolerance_mm']:.1f} mm）")
    for stage in clearance["stages"]:
        log(f"      {pad(stage['stage'], 28)}{pad(stage['kind'], 9)}"
            f"{stage['min_clearance_mm']:>7.1f}{stage['min_gripper_clearance_mm']:>9.1f} mm"
            f"  {'OK' if stage['ok'] else '*** 不可 ***'}")
    log("    モデルの絶対値は実測と 17.6 / 45.3 mm ずれていました（2 姿勢で測定）。")
    log("    どちらも保守側でしたが、2 点では全姿勢がそうだとは言えません。")
    log("    だから判定は絶対値ではなく「開始時より机へ近づかないこと」です。")

    log("\n  --- 走行を止める条件 ---")
    log("    [移動中]")
    log(f"      {pad('前進の停止', 26)}waypoint まで "
        f"{STALL_REMAINING_DEG:.1f} deg 以上残して "
        f"{STALL_SPEED_DEG_S:.2f} deg/s 未満が "
        f"{STALL_WINDOW_S + STALL_PERSIST_S:.1f} s 継続")
    log(f"      {pad('逆行', 26)}残距離が {RETREAT_DEG:.1f} deg 以上増えた")
    log(f"      {pad('waypoint 未到達', 26)}{WAYPOINT_TIMEOUT_S:.0f} s 以内に "
        f"±{ARRIVE_DEG:.1f} deg に入らない")
    log(f"      {pad('', 26)}（残り {STALL_REMAINING_DEG:.1f} deg 未満で"
        f"止まった場合はこちらが捕まえます）")
    log("      追従誤差の大きさそのものでは停止しません。移動中に遅れるのは")
    log("      当然で、止まっていることのほうが異常だからです。")
    log("    [保持中]")
    log(f"      {pad('指令からのずれ', 26)}> {HOLD_ERROR_ABORT_DEG:.1f} deg")
    log("    [いつでも]")
    log(f"      {pad('|Present_Load|', 26)}> {LOAD_ABORT} / 1023"
        f"（この掃引の予測値は 40 前後）")
    log(f"      {pad('温度', 26)}> {TEMPERATURE_ABORT_C} C"
        f"（現在 {wrist['temperature']} C）")
    log(f"      {pad('連続読み出し失敗', 26)}>= {COMMS_ABORT} 回")
    log("    いずれも「ここまでは安全」という意味ではありません。異常を早く")
    log("    止めるための上限です。異音・振動・不自然な動きがあれば、数値が")
    log("    条件を満たしていなくても止めてください。")

    log("\n  --- 条件に触れたとき何が起きるか ---")
    log("    1. 全関節の Present_Position を ticks で読む")
    log("    2. Goal_Position をその Present_Position へ書き換える")
    log("    3. 以降の軌道指令は出しません")
    log("    4. トルクは ON のまま、その場を保持します")
    log("    5. 全関節の状態を表示し、summary.json に保存します")
    log("    そのあと、収納するかどうかは人が決めます（home の入力が必要）。")

    log("\n  --- 書き込み先 ---")
    log(f"    {pad('出力ディレクトリ', 28)}{out_dir}"
        f"{'（dry run のため未作成）' if args.dry_run else ''}")
    log(f"    {pad('ロック', 28)}{lock_path}（このプロセスが保持中）")
    log(f"    {pad('既存の結果', 28)}上書きしません。実行ごとに別ディレクトリです")

    log("\n  --- この計画に対するツインの判定 ---")
    log(f"    {'問題なし' if clear else '*** 干渉あり — 実行を拒否します ***'}")
    return clear


def liftoff_record(stages, detail):
    """What the lift-off actually was this run, built from what was used.

    Not from a constant: there is no constant any more. The stage list says
    whether one was planned and how many waypoints it got; the planner's own
    working says which direction it measured as lifting and what that bought.
    """
    stage = next((s for s in stages
                  if s.kind == "transit" and s.label.startswith("離陸")), None)
    record = {
        "liftoff_needed": bool(stage),
        "liftoff_waypoints": [] if stage is None else
            [{joint: round(value, 3) for joint, value in point.items()}
             for point in stage.path],
        "liftoff_stage_label": None if stage is None else stage.label,
        "liftoff_selected_joint": None,
        "liftoff_direction": None,
        "liftoff_move_deg": None,
        "liftoff_clearance_before_mm": None,
        "liftoff_clearance_after_mm": None,
        "liftoff_speed_deg_s": None if stage is None else stage.speed,
    }
    if detail:
        using = detail.get("using")          # e.g. "elbow_flex -1"
        if using:
            joint, _, sign = using.rpartition(" ")
            record["liftoff_selected_joint"] = joint
            record["liftoff_direction"] = sign
        record["liftoff_move_deg"] = detail.get("move_deg")
        record["liftoff_clearance_before_mm"] = detail.get("start_clearance_mm")
        record["liftoff_clearance_after_mm"] = detail.get("reaches_mm")
        record["liftoff_sensitivity_mm_per_deg"] = detail.get(
            "sensitivity_mm_per_deg")
        record["liftoff_why_not"] = detail.get("why")
    return record


def git_state():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        return {"commit": out.stdout.strip(),
                "dirty": bool(dirty.stdout.strip())}
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None}


# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="wrist_flex の 1 つの角度が実用できるかを確認します。"
                    "意図的に、1 回の実行につき 1 角度だけです。")
    parser.add_argument("--to", type=float, metavar="DEG",
                        help=f"確認する wrist_flex の角度。"
                             f"-{LIMIT_DEG:.0f}..+{LIMIT_DEG:.0f}")
    parser.add_argument("--posture-only", action="store_true",
                        help="姿勢へ行って到達を確認し、戻るだけ。手首は振りません")
    parser.add_argument("--fraction", type=float, default=1.0, metavar="F",
                        help="姿勢までの道のりのうち F だけ進んで止まります"
                             "（0<F<=1）。動作系を短い移動で確かめてから"
                             "全行程に進むためのものです。--posture-only 専用")
    parser.add_argument("--support", metavar="x0,x1,y0,y1,top",
                        help="机に置いたままにする支持物の箱を mm で。"
                             "ベース中心が原点、x は前方、y は左方、"
                             "top は机面からの高さ。例: 120,220,-60,60,55。"
                             f"一度渡すと {Support.PATH} に保存され、"
                             "次回以降は自動で読み込みます")
    parser.add_argument("--no-support", action="store_true",
                        help="保存された支持物を無視します（撤去した場合）")
    parser.add_argument("--measured-clearance", type=float, metavar="MM",
                        help="定規で実測した、グリッパ最下点から机までの mm。"
                             "開始条件の判定にだけ使います。モデルの干渉判定を"
                             "無効化するものではありません")
    parser.add_argument("--where", action="store_true",
                        help="いまの姿勢と机までの余裕を表示するだけ。"
                             "手でアームを持ち上げながら繰り返し実行できます")
    parser.add_argument("--dry-run", action="store_true",
                        help="計画を表示するだけで、何も動かしません")
    parser.add_argument("--posture", choices=sorted(POSTURES), default="upright",
                        help="手首を振るときのアームの姿勢")
    parser.add_argument("--hold", type=float, default=3.0,
                        help="目標角で保持する秒数")
    parser.add_argument("--repeats", type=int, default=2,
                        help="目標角へ行く回数（再現性の確認）")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    support = resolve_support(args, Log() if args.where else None)
    if args.where:
        from so101.hardware import resolve as resolve_port
        where(resolve_port([args.follower_port])[0], Log(), support)
        return
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit("\n  --fraction は 0 より大きく 1 以下にしてください\n")
    if args.fraction < 1.0 and not args.posture_only:
        raise SystemExit(
            "\n  --fraction は --posture-only と一緒にだけ使えます。"
            "手首の掃引を途中で止めても測るものがありません。\n")
    if args.posture_only:
        args.to = 0.0
    elif args.to is None:
        raise SystemExit("\n  --to か --posture-only のどちらかを指定してください\n")
    if abs(args.to) > LIMIT_DEG:
        raise SystemExit(
            f"\n  {args.to:+.1f} deg は R1 が許す ±{LIMIT_DEG:.0f} deg の外側です。"
            f"\n  上書きするフラグはありませんし、ここに足すべきでもありません。"
            f"\n  この先は、サーボ自身の停止位置まで 6 deg を切ります。\n")
    if args.repeats < 1:
        raise SystemExit("  --repeats は 1 以上にしてください")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    what = ("posture" if args.posture_only
            else f"{'pos' if args.to >= 0 else 'neg'}{abs(args.to):.0f}")
    out_dir = args.out / f"{stamp}_{what}"
    lock_path = args.out / ".lock"

    from so101.hardware import resolve as resolve_port
    port = resolve_port([args.follower_port])[0]

    with only_one(lock_path):
        log = Log()
        log(f"\n  Phase R1: "
            + (f"姿勢確認のみ（{args.posture}）" if args.posture_only
               else f"wrist_flex {args.to:+.1f} deg、姿勢 {args.posture!r}")
            + f"、ポート {port}")
        log("  " + ("DRY RUN — 何も動かしません" if args.dry_run
                    else "LIVE — 実機が動きます"))

        arm = read_arm(port)
        posture = POSTURES[args.posture]
        start = {name: arm[name]["deg"] for name in ARM_JOINTS}
        table = Table(arm["gripper"]["deg"], support)
        if support is not None:
            log(f"\n  支持物: {support.describe()}")
            log("  置いたまま実行します。トルク ON 後に手を入れて取り除く運用は")
            log("  しません。全 waypoint の掃引体積との距離を下で検査します。")
        else:
            log("\n  支持物: 設定されていません（--support で指定できます）")

        log("\n  --- ここから机までの余裕を、いまの姿勢について測ります ---")
        liftoff, detail = plan_liftoff(table, start, dict(posture,
                                                          wrist_flex=SAFE_DEG),
                                       arm, log)
        report_liftoff(detail, liftoff, log)

        stages = stages_for(start, posture, args.to, args.repeats,
                            args.posture_only, args.posture, liftoff,
                            args.fraction)

        log("\n  --- 送信する waypoint 列を、そのまま MuJoCo で再生します ---")
        clear, clearance = replay(stages, start, table, log)
        clearance["liftoff"] = detail

        log("\n  --- waypoint 列の連続性（不自然な branch jump がないか）---")
        shape_ok = check_path_shape(stages, start, log)

        log("\n  --- 全 waypoint が現在の joint limit の内側か ---")
        limits_ok = check_limits(stages, arm, log)

        # The start condition, and the one place a ruler is allowed to speak.
        # It never switches off the twin: the collision and approach checks
        # above have already run and their verdict stands whatever is typed
        # here. What a measurement can settle is only whether the arm is far
        # enough from the table to begin, which is a fact about the bench that
        # the model has been shown to get wrong by tens of millimetres.
        measured = args.measured_clearance
        clearance["measured_clearance_mm"] = measured
        clearance["measured_floor_mm"] = MEASURED_FLOOR_MM
        if measured is None:
            # The gripper's own clearance, not the lowest thing on the arm.
            # That is the shoulder, sitting 18.6 mm up wherever the arm is
            # pointed - a constant of the robot's own structure that never
            # moves towards the table, so gating on it gates on nothing. The
            # gripper is the part that can arrive somewhere it should not.
            start_ok = (clearance["start_gripper_clearance_mm"]
                        >= CLEARANCE_FLOOR_MM)
            clearance["start_basis"] = "model (gripper)"
        else:
            start_ok = measured >= MEASURED_FLOOR_MM
            clearance["start_basis"] = "measured"
        clearance["start_ok"] = bool(start_ok)

        ok = plan_report(args, arm, posture, stages, out_dir, lock_path, log,
                         clear and limits_ok and shape_ok, clearance)
        if args.dry_run:
            log("\n  Dry run です。何も動かさず、何も書き込んでいません。\n")
            return
        if support is not None and not clearance.get("leaves_support", True):
            raise SystemExit(
                f"\n  経路の終わりで支持物まで "
                f"{clearance['end_support_distance_mm']:.1f} mm しかありません"
                f"（要求 {SUPPORT_MARGIN_MM:.0f} mm）。\n"
                "  支持物を動かすか、置き方を変えてください。\n")
        if not shape_ok:
            raise SystemExit(
                "\n  waypoint 列が不連続です。実行を拒否します。\n")
        if not limits_ok:
            raise SystemExit(
                "\n  joint limit の外へ出る waypoint があります。実行を拒否します。\n")
        if not start_ok:
            if measured is None:
                raise SystemExit(
                    f"\n  開始時のクリアランス（モデル）が "
                    f"{clearance['start_clearance_mm']:.1f} mm しかありません"
                    f"（下限 {CLEARANCE_FLOOR_MM:.0f} mm）。\n"
                    "  トルクは切れています。人が手を離しても保つ支持を使って\n"
                    "  グリッパを机から離し、定規で測った値を渡してください:\n"
                    "    --measured-clearance <mm>   "
                    f"（{MEASURED_FLOOR_MM:.0f} mm 以上が必要）\n"
                    "  いまの姿勢と予測値:\n"
                    "    uv run scripts/check_wrist_flex_operational_range.py --where\n")
            raise SystemExit(
                f"\n  実測クリアランス {measured:.1f} mm は下限 "
                f"{MEASURED_FLOOR_MM:.0f} mm に届きません。\n"
                "  定規の読みは cm 単位で +-5 mm 程度あるので、開始条件が\n"
                "  測定誤差ひとつでひっくり返らない値にしてあります。\n")
        if not ok:
            raise SystemExit(
                "\n  ツインがこの計画に干渉を報告しました。実行を拒否します。\n")

        run(args, arm, posture, stages, port, out_dir, log, clearance,
            support)


def run(args, arm, posture, stages, port, out_dir, log, clearance,
        support=None):
    """Everything from here on moves the arm."""
    import numpy as np  # noqa: F401 - summarise needs it; fail early if absent

    log("\n  --- 接続する前に ---")
    log("    robot.connect() でトルクが入ります。接続＝始動と考えてください。")
    log("    4 つとも、声に出して確認してください:")
    log("      1. アームの可動範囲に人も物もない")
    log(f"      2. {args.posture!r} 姿勢でアームは約 400 mm の高さに立つ。"
        f"その上方が空いている")
    log("      3. 電源をすぐ落とせる — そして落とせばアームは落下する")
    log("      4. このディレクトリで 2 つ目のターミナルが開いている")
    log("")
    log("    緊急停止は、この順で:")
    log("      A. ここで Ctrl-C        — 指令を止め、その場で保持します")
    log("      B. 2 つ目のターミナル:  uv run scripts/torque_off.py COM4")
    log("         （このプロセスがポートを手放したあとでなければ届きません）")
    log("      C. 電源を落とす         — 先にアームを支えてください")
    # The word, not ENTER. Every other prompt in this run takes ENTER, and this
    # one deliberately does not: it is the moment torque comes on, and a
    # keystroke made out of rhythm should not be able to start the arm.
    if input("\n  接続するなら ready と入力（それ以外は中止）: ").strip() \
            .lower() != "ready":
        log("  接続前に中止しました。何も動かしておらず、何も書いていません。")
        log("  （このプロンプトだけは ready の入力が必要です。ENTER だけなら")
        log("    中止します。これ以降のプロンプトは ENTER で進みます。）")
        return

    from so101.hardware import bus_patch  # noqa: F401  serial retries
    from so101.hardware import tuning
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    # The gain the rest of the stack runs at. LeRobot writes 16 on every
    # connect; measuring tracking against 16 would measure a gain nothing else
    # uses. Recorded in the summary either way.
    tuning.install(verbose=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    log.close()
    log = Log(out_dir / "console.log")
    samples_path = out_dir / "samples.csv"
    handle = open(samples_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=Watch.FIELDS)
    writer.writeheader()

    robot = watch = None
    aborted = None
    home = {name: arm[name]["deg"] for name in ARM_JOINTS}
    goal_velocity = int(GOAL_VELOCITY_DEG_S * TICKS_PER_DEG)
    upright = dict(posture, wrist_flex=SAFE_DEG)
    result = {
        "phase": "R1", "joint": JOINT,
        "test": "posture only" if args.posture_only else f"{args.to:+.0f} deg",
        "target_deg": None if args.posture_only else args.to,
        "direction": ("none" if args.posture_only
                      else "positive" if args.to >= 0 else "negative"),
        "posture": args.posture, "posture_deg": upright,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "port": port, "repeats_requested": args.repeats, "hold_s": args.hold,
        "home_pose_deg": {k: round(v, 2) for k, v in home.items()},
        "servo": {
            "mid_ticks": arm[JOINT]["mid_ticks"],
            "limit_ticks": [arm[JOINT]["min_ticks"], arm[JOINT]["max_ticks"]],
            "limit_deg": [round(arm[JOINT]["min_deg"], 2),
                          round(arm[JOINT]["max_deg"], 2)],
            "target_ticks": (None if args.posture_only
                             else round(ticks_for(args.to, arm), 1)),
        },
        "thresholds": {
            "load_abort": LOAD_ABORT,
            "temperature_abort_c": TEMPERATURE_ABORT_C,
            "comms_abort": COMMS_ABORT,
            "stall_remaining_deg": STALL_REMAINING_DEG,
            "stall_speed_deg_s": STALL_SPEED_DEG_S,
            "stall_window_s": STALL_WINDOW_S,
            "stall_persist_s": STALL_PERSIST_S,
            "retreat_deg": RETREAT_DEG,
            "hold_error_abort_deg": HOLD_ERROR_ABORT_DEG,
            "waypoint_timeout_s": WAYPOINT_TIMEOUT_S,
        },
        "motion": {
            "step_deg": STEP_DEG, "arrive_deg": ARRIVE_DEG,
            "phase_tolerance_deg": PHASE_TOLERANCE_DEG,
            "wrist_speed_deg_s": WRIST_SPEED_DEG_S,
            "transit_speed_deg_s": TRANSIT_SPEED_DEG_S,
            "liftoff_speed_deg_s": LIFTOFF_SPEED_DEG_S,
            "clearance_floor_mm": CLEARANCE_FLOOR_MM,
            "goal_velocity_ticks_s": goal_velocity,
            "max_relative_target_deg": MAX_RELATIVE_TARGET,
            "fps": FPS, "p_coefficient": tuning.p_coefficient(),
        },
        "git": git_state(), "clearance": clearance,
        "support": None if support is None else support.as_dict(),
        "liftoff": liftoff_record(stages, clearance.get("liftoff")),
        "stages": [{"label": stage.label, "kind": stage.kind,
                    "waypoints": len(stage.path), "speed_deg_s": stage.speed,
                    "subject": stage.subject} for stage in stages],
        "arrivals": [], "holds": [],
    }

    try:
        robot = make_robot_from_config(SO101FollowerConfig(
            port=port, id="follower",
            max_relative_target=MAX_RELATIVE_TARGET))
        robot.connect()
        log("\n  接続しました。トルクが入り、アームはその場を保持しています")

        for name in ARM_JOINTS:
            robot.bus.write("Goal_Velocity", name, goal_velocity)
        log(f"  アーム 5 関節の Goal_Velocity を {goal_velocity} ticks/s "
            f"({GOAL_VELOCITY_DEG_S:.0f} deg/s) に設定しました")
        log(f"  max_relative_target は使用していません（{MAX_RELATIVE_TARGET}）")

        watch = Watch(robot, writer, log)
        here = joints_of(robot)
        log("  読み戻し: " + "、".join(f"{n} {here[n]:+.1f}" for n in ARM_JOINTS))

        # Every transit stage, in the order the planner produced them. Not
        # stages[0] and stages[1]: whether there is a lift-off stage at all is
        # decided per pose now, and indexing past it was how a run with no
        # lift-off would have flown the wrong stage at the lift-off's speed.
        transits = [stage for stage in stages if stage.kind == "transit"]
        reached = None
        for number, stage in enumerate(transits, 1):
            log(f"\n  --- {number}/{len(transits)}: {stage.label}、"
                f"{len(stage.path)} 点、{stage.speed:.1f} deg/s ---")
            if stage.note:
                for line in stage.note.split("。"):
                    if line.strip():
                        log(f"  {line.strip()}。")
            if input("  動かすなら ENTER（それ以外は中止）: ").strip():
                raise KeyboardInterrupt
            reached = walk(robot, watch, stage.path, stage.kind, stage.subject,
                           stage.path[-1].get(stage.subject,
                                              here[stage.subject]),
                           0, stage.speed, log, label=stage.label)
            log(f"    {stage.subject}  負荷 {watch.rows[-1]['load']}、"
                f"{watch.rows[-1]['temperature_c']} C")
            if number < len(transits):
                if input("  異常がなければ ENTER で継続（それ以外は中止）: "
                         ).strip():
                    raise KeyboardInterrupt
        result["posture_reached_deg"] = {n: round(reached[n], 2)
                                         for n in ARM_JOINTS}
        result["fraction"] = args.fraction
        result["upright_reached"] = args.fraction >= 1.0
        if args.fraction >= 1.0:
            log("\n  UPRIGHT_REACHED — 全関節が許容範囲内です")
        else:
            log(f"\n  道のりの {100*args.fraction:.0f}% まで到達しました。"
                f"姿勢そのものへはまだ行っていません。")
        result["arrivals"].append({"phase": "upright", "ok": True,
                                   "pose": result["posture_reached_deg"]})

        if not args.posture_only:
            # Taken from the list by kind, in pairs: out then back. Counting
            # from index 2 assumed a lift-off stage was always there in front
            # of them, and it is not.
            sweeps = [stage for stage in stages if stage.kind == "sweep"]
            for repeat in range(1, args.repeats + 1):
                out_stage = sweeps[2 * (repeat - 1)]
                back_stage = sweeps[2 * (repeat - 1) + 1]
                log(f"\n  --- {repeat}/{args.repeats} 回目: "
                    f"{SAFE_DEG:+.0f} → {args.to:+.1f} deg、"
                    f"{len(out_stage.path)} 点 ---")
                if input("  進めるなら ENTER（それ以外は中止）: ").strip():
                    raise KeyboardInterrupt
                walk(robot, watch, out_stage.path, "approach", JOINT, args.to,
                     repeat, out_stage.speed, log, label=out_stage.label)

                taken = hold_at(robot, watch, args.to, args.hold, repeat, log)
                stats = summarise(taken, f"hold {repeat}")
                result["holds"].append(stats)
                log(f"    静定角 {stats['measured_mean_deg']:+.2f} deg"
                    f"（保持中のぶれ {stats['measured_drift_deg']:.2f} deg）")
                log(f"    追従誤差  平均 {stats['tracking_error_mean_deg']:.2f}"
                    f"  中央 {stats['tracking_error_median_deg']:.2f}"
                    f"  p95 {stats['tracking_error_p95_deg']:.2f}"
                    f"  最大 {stats['tracking_error_worst_deg']:.2f} deg")
                log(f"    負荷      平均 {stats['load_mean']:.0f}"
                    f"  p95 {stats['load_p95']:.0f}"
                    f"  最大 {stats['load_worst']} / 1023")
                log(f"    温度      {stats['temperature_start_c']} → "
                    f"{stats['temperature_end_c']} C")

                log(f"    {SAFE_DEG:+.0f} deg へ戻します")
                walk(robot, watch, back_stage.path, "return", JOINT,
                     SAFE_DEG, repeat, back_stage.speed, log,
                     label=back_stage.label)

        result["current_register_ever_nonzero"] = watch.current_ever_nonzero
        if not watch.current_ever_nonzero:
            log("\n  Present_Current は終始 0 でした。このサーボは電流を報告"
                "していないようです。負荷の指標は Present_Load です。")

    except Abort as error:
        aborted = str(error)
        log(f"\n  *** 異常停止: {error} ***")
        result["freeze"] = freeze(robot, log, watch, aborted)
    except KeyboardInterrupt:
        aborted = "操作者が中止しました"
        log("\n  操作者が中止しました")
        result["freeze"] = freeze(robot, log, watch, aborted)
    except Exception as error:  # noqa: BLE001
        aborted = f"{type(error).__name__}: {error}"
        log(f"\n  *** {aborted} ***")
        result["freeze"] = freeze(robot, log, watch, aborted)
    finally:
        result["aborted"] = aborted
        result["finished_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        if robot is not None:
            park(robot, watch, home, log, aborted)
            try:
                robot.disconnect()
                log("  切断しました。トルクを解放しました")
            except Exception as error:  # noqa: BLE001
                log(f"  正常に切断できませんでした: {error}")
                log("  実行してください:  uv run scripts/torque_off.py COM4")
        handle.close()
        verdict(result, args, log)
        (out_dir / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"\n  全サンプル  {samples_path}")
        log(f"  まとめ      {out_dir / 'summary.json'}")
        log.close()


FREEZE_ATTEMPTS = 3
FREEZE_RETRIES = 5


def freeze(robot, log, watch=None, reason=""):
    """Stop by making where the arm *is* the thing it is asked to hold.

    Cutting the interpolation is not stopping. The last Goal_Position is still
    standing in the servo, and a servo goes on pushing towards its goal for as
    long as it has torque - so a joint that was aborted because it had fallen
    behind its command is left leaning on exactly the command it could not
    meet. Whatever stopped it, it keeps pulling against.

    So the goal is overwritten with the present position, all six joints at
    once, before anything is reported or decided. Torque stays on: this is the
    wrong moment to drop a raised arm.

    Ticks rather than degrees, in both directions. A degree round-trip goes
    through the calibration midpoint and back and lands a tick or two out, and
    a tick or two of "hold still" is not holding still.
    """
    if robot is None:
        return {"frozen": False, "why": "never connected"}

    for attempt in range(1, FREEZE_ATTEMPTS + 1):
        try:
            ticks = robot.bus.sync_read("Present_Position", normalize=False,
                                        num_retry=FREEZE_RETRIES)
            robot.bus.sync_write("Goal_Position", ticks, normalize=False,
                                 num_retry=FREEZE_RETRIES)
            break
        except Exception as error:  # noqa: BLE001 - keep trying, then say so
            log(f"  その場保持 {attempt}/{FREEZE_ATTEMPTS} 回目に失敗: {error}")
    else:
        log("\n  *** その場保持に失敗しました — アームがまだ直前の指令へ"
            "向かっている可能性があります ***")
        log("  アームを支えたうえで、もう一方のターミナルで:")
        log("    taskkill /F /IM python.exe")
        log("    uv run scripts/torque_off.py COM4")
        return {"frozen": False, "why": "the bus would not answer"}

    log("  その場保持: 全 6 関節の Goal_Position を Present_Position へ"
        "書き換えました。トルクは ON のまま、いまいる場所を保持します")
    state = {"frozen": True, "reason": reason, "joints": {}}
    log("    " + pad("joint", 15) + rpad("現在角", 9) + rpad("ticks", 8)
        + rpad("指令", 8) + rpad("負荷", 7) + rpad("温度", 7)
        + rpad("電圧", 7) + rpad("torque", 8))
    for name in (*ARM_JOINTS, "gripper"):
        row = {}
        for register, key in (("Present_Position", "ticks"),
                              ("Goal_Position", "goal"),
                              ("Present_Load", "load"),
                              ("Present_Temperature", "temperature_c"),
                              ("Present_Voltage", "voltage"),
                              ("Torque_Enable", "torque_enable")):
            try:
                row[key] = robot.bus.read(register, name, normalize=False,
                                          num_retry=FREEZE_RETRIES)
            except Exception:  # noqa: BLE001
                row[key] = None
        try:
            row["deg"] = round(robot.bus.read("Present_Position", name,
                                              num_retry=FREEZE_RETRIES), 3)
        except Exception:  # noqa: BLE001
            row["deg"] = None
        if row.get("voltage") is not None:
            row["voltage_v"] = row.pop("voltage") / 10
        state["joints"][name] = row
        log(f"    {name:<15}"
            f"{row['deg'] if row['deg'] is not None else float('nan'):>+9.2f}"
            f"{str(row['ticks']):>8}{str(row['goal']):>8}"
            f"{str(row['load']):>7}{str(row['temperature_c']):>6}C"
            f"{row.get('voltage_v', 0):>6.1f}V{str(row['torque_enable']):>8}")

    if watch is not None:
        try:
            watch.sample(status="abort", notes=reason, judge=False)
        except Exception:  # noqa: BLE001 - the record is not worth a second fault
            pass
    log("\n  アームは保持しています。どうするか決める前に、見て、音を"
        "聞いてください。")
    return state


def park(robot, watch, home, log, aborted=None):
    """Wrist to zero, arm back to where it was resting, then let go.

    The order is the point. Releasing torque with the arm straight up drops it
    from 400 mm; the pose it was found in is the one it was already holding
    without any torque at all, so that is where it is put back.
    """
    log("\n  --- 収納 ---")
    log("  アームは上がっています。ここでトルクを切れば落下するので、先に")
    log("  見つけたときの折り畳み姿勢へ戻します。")
    try:
        if aborted:
            # After an abort the arm is holding a pose nobody planned, and the
            # reason it stopped has not been looked at yet. Moving is then the
            # answer that has to be asked for, not the one that happens by
            # pressing ENTER.
            log(f"  この実行は途中で止まりました: {aborted}")
            log("  理由が分かるまで、動かすのは安全ではありません。")
            if input("  畳むなら home と入力（それ以外は保持のまま）: "
                     ).strip().lower() != "home":
                log("  保持したままにします（トルク ON）。電源を切る前に"
                    "下ろしてください。")
                log("  準備ができたら:  uv run scripts/torque_off.py COM4")
                return
        elif input("  戻すなら ENTER、保持したままにするなら hold: "
                   ).strip().lower() == "hold":
            log("  保持したままにします（トルク ON）。電源を切る前に"
                "下ろしてください。")
            log("  準備ができたら:  uv run scripts/torque_off.py COM4")
            return
    except (EOFError, KeyboardInterrupt):
        log("  保持したままにします（トルク ON）。")
        return

    try:
        # Straighten the wrist before folding the arm, so the gripper is not
        # swung through anything on the way down. Walked, like everything else.
        here = joints_of(robot)
        _quiet_walk(robot, waypoints(here, {JOINT: SAFE_DEG}),
                    WRIST_SPEED_DEG_S)
        here = joints_of(robot)
        _quiet_walk(robot, waypoints(here, {n: home[n] for n in ARM_JOINTS}),
                    TRANSIT_SPEED_DEG_S)
        log("  収納しました")
    except Exception as error:  # noqa: BLE001 - parking must not itself fail
        log(f"  正常に収納できませんでした: {error}")
        log("  アームは止まった位置にあり、トルクは ON です。支えたうえで:")
        log("    uv run scripts/torque_off.py COM4")
        return

    for name in ARM_JOINTS:
        try:
            robot.bus.write("Goal_Velocity", name, 0)
        except Exception:  # noqa: BLE001
            pass
    log("  Goal_Velocity を 0 へ戻しました")


def _quiet_walk(robot, path, speed):
    """Walk a waypoint list without sampling or judging - used on the way out."""
    for target in path:
        start = joints_of(robot)
        move = max((abs(target[n] - start[n]) for n in target), default=0.0)
        steps = max(1, int(round(max(move / speed, 1.0 / FPS) * FPS)))
        for step in range(1, steps + 1):
            robot.send_action({
                f"{n}.pos": start[n] + (target[n] - start[n]) * step / steps
                for n in target})
            time.sleep(1.0 / FPS)
        deadline = time.perf_counter() + WAYPOINT_TIMEOUT_S
        while time.perf_counter() < deadline:
            if arrived(joints_of(robot), target, ARRIVE_DEG)[0]:
                break
            time.sleep(1.0 / SAMPLE_HZ)


def verdict(result, args, log):
    """What the run showed. Not a joint limit - R1 does not decide one."""
    log(f"\n  === {result['test']}（{args.posture}）===")
    if result.get("aborted"):
        log(f"  完了しませんでした: {result['aborted']}")
        result["verdict"] = "aborted"
        return
    if args.posture_only:
        if result.get("fraction", 1.0) < 1.0:
            log(f"  道のりの {100*result['fraction']:.0f}% までの試走です。"
                f"UPRIGHT_REACHED はまだ主張しません。")
        log("  UPRIGHT_REACHED: " + ("はい" if result.get("upright_reached")
                                     else "いいえ"))
        for name, value in result.get("posture_reached_deg", {}).items():
            log(f"    {pad(name, 16)}{value:+7.2f} deg")
        result["verdict"] = ("ordinary" if result.get("upright_reached")
                             else "review")
        log("  → " + ("姿勢へ到達し、戻りました" if result["verdict"] == "ordinary"
                      else "確認が必要です"))
        return

    holds = result.get("holds", [])
    if len(holds) < args.repeats:
        log(f"  {args.repeats} 回中 {len(holds)} 回しか完了していません")
        result["verdict"] = "incomplete"
        return

    settled = [h["measured_mean_deg"] for h in holds]
    spread = max(settled) - min(settled)
    worst_error = max(h["tracking_error_worst_deg"] for h in holds)
    worst_load = max(h["load_worst"] for h in holds)
    worst_drift = max(h["measured_drift_deg"] for h in holds)
    result["repeatability"] = {
        "settled_deg": [round(s, 3) for s in settled],
        "spread_deg": round(spread, 3),
        "tracking_error_worst_deg": round(worst_error, 3),
        "load_worst": worst_load,
        "hold_drift_worst_deg": round(worst_drift, 3),
    }
    log(f"  静定角 {'、'.join(f'{s:+.2f}' for s in settled)} deg"
        f" — ばらつき {spread:.2f} deg")
    log(f"  追従誤差の最大 {worst_error:.2f} deg、負荷の最大 {worst_load} / 1023、"
        f"保持中のぶれの最大 {worst_drift:.2f} deg")

    good = (worst_error < 2.0 and worst_load < 200 and spread < 1.0
            and worst_drift < 0.5)
    result["verdict"] = "ordinary" if good else "review"
    log("  → " + ("異常なし" if good else "確認が必要です"))
    log("\n  これは 1 つの角度についての記録であって、運用限界ではありません。")
    log("  限界は、正負どちらも歩き終えてから決めます。")


if __name__ == "__main__":
    main()
