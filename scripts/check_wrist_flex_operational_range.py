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
#: The arm starts folded, resting against itself - the twin reports the
#: shoulder and the lower arm already in contact at the pose it is found in.
#: Coming out of that is the one moment in the run where a joint might be
#: pushing something rather than swinging free, so the first slice of the
#: transit is taken at a crawl and looked at before the rest of it runs.
BREAKOUT_FRACTION = 0.08
BREAKOUT_SPEED_DEG_S = 1.5
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
TABLE_Z_M = -0.0085           # data/table_frame.json, the most real number there is


class Abort(RuntimeError):
    """Something crossed a threshold. The run stops; the arm keeps holding."""


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


def replay(stages, start, log, table_z=TABLE_Z_M, fine=6):
    """Fly the exact waypoint list in MuJoCo. Returns True if it is clear.

    `fine` subdivides each waypoint interval further, because a collision can
    happen between two waypoints as easily as at one.

    Two things are checked and they fail differently. A contact that is *new*
    compared with where the arm already rests is one the path created; the
    contacts the parked pose already has are the arm leaning on itself as it
    sits there, and it leaves them on the first move.
    """
    from so101.sim.model import SO101Sim

    bodies = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
              "moving_jaw_so101_v1")
    ignore = tuple(f"block_{i}" for i in range(16)) + \
        tuple(f"block_{i}_geom" for i in range(16))
    sim = SO101Sim(table_z=table_z)

    def look(pose):
        sim.set_joints(pose, gripper_deg=35.0)
        found = {f"{a}/{b}" for a, b, _ in sim.collisions(ignore=ignore)
                 if "table" not in (a, b)}
        return found, sim.lowest_point(bodies=bodies)

    here = dict(start)
    baseline, _ = look(here)
    if baseline:
        log(f"    いまの休止姿勢はすでに自身に接触しています: {sorted(baseline)}")
        log("    （折り畳まれて寄りかかっている状態です。動き出せば離れます）")

    clear = True
    for label, path in stages:
        lowest, offenders, worst_at = float("inf"), set(), None
        previous = dict(here)
        for target in path:
            for step in range(1, fine + 1):
                pose = dict(previous)
                pose.update({n: previous.get(n, here.get(n, 0.0))
                             + (target[n] - previous.get(n, here.get(n, 0.0)))
                             * step / fine for n in target})
                full = dict(here)
                full.update(pose)
                found, low = look(full)
                if low < lowest:
                    lowest, worst_at = low, dict(full)
                offenders |= (found - baseline)
            previous = dict(previous)
            previous.update(target)
            here.update(target)
        hit_table = lowest <= table_z
        verdict = ("問題なし" if not offenders and not hit_table
                   else "*** 干渉あり ***")
        log(f"    {pad(label, 30)}{len(path):>4} 点   "
            f"最低点 {1000*lowest:+7.1f} mm   {verdict}")
        if offenders:
            log(f"      新たな自己干渉: {sorted(offenders)}")
            clear = False
        if hit_table:
            log(f"      机（{1000*table_z:+.1f} mm）に達します。そのときの姿勢:")
            log("        " + "、".join(f"{n} {worst_at[n]:+.1f}"
                                        for n in ARM_JOINTS if n in worst_at))
            clear = False
    return clear


def stages_for(start, posture, target_deg, repeats, posture_only,
               posture_name="upright"):
    """Every move the run will make, named, as waypoint lists.

    One function, used by the twin and by the arm. When the two disagree about
    what is going to be flown, the twin's verdict is about something else.
    """
    upright = dict(posture, wrist_flex=SAFE_DEG)
    breakout = {n: start[n] + (upright[n] - start[n]) * BREAKOUT_FRACTION
                for n in upright}
    out = [("展開（折り畳みから）", waypoints(start, breakout)),
           (f"{posture_name} へ", waypoints(breakout, upright))]
    if posture_only:
        return out
    here = dict(upright)
    for repeat in range(1, repeats + 1):
        out.append((f"{repeat} 回目: wrist_flex → {target_deg:+.0f}",
                    waypoints(here, {JOINT: target_deg})))
        here[JOINT] = target_deg
        out.append((f"{repeat} 回目: wrist_flex → {SAFE_DEG:+.0f}",
                    waypoints(here, {JOINT: SAFE_DEG})))
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

def plan_report(args, arm, posture, stages, out_dir, lock_path, log, clear):
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
    for label, path in stages:
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
        f"展開 {BREAKOUT_SPEED_DEG_S:.1f} / 移動 {TRANSIT_SPEED_DEG_S:.0f} / "
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
        stages = stages_for(start, posture, args.to, args.repeats,
                            args.posture_only, args.posture)

        log("\n  --- 送信する waypoint 列を、そのまま MuJoCo で再生します ---")
        clear = replay(stages, start, log)

        ok = plan_report(args, arm, posture, stages, out_dir, lock_path, log,
                         clear)
        if args.dry_run:
            log("\n  Dry run です。何も動かさず、何も書き込んでいません。\n")
            return
        if not ok:
            raise SystemExit(
                "\n  ツインがこの計画に干渉を報告しました。実行を拒否します。\n")

        run(args, arm, posture, stages, port, out_dir, log)


def run(args, arm, posture, stages, port, out_dir, log):
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
            "breakout_speed_deg_s": BREAKOUT_SPEED_DEG_S,
            "goal_velocity_ticks_s": goal_velocity,
            "max_relative_target_deg": MAX_RELATIVE_TARGET,
            "fps": FPS, "p_coefficient": tuning.p_coefficient(),
        },
        "git": git_state(), "arrivals": [], "holds": [],
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

        breakout, to_posture = stages[0], stages[1]

        log(f"\n  --- {pad('1. ' + breakout[0], 26)}{len(breakout[1])} 点、"
            f"{BREAKOUT_SPEED_DEG_S:.1f} deg/s ---")
        log("  自身に寄りかかった姿勢から抜け出す区間です。")
        if input("  動かすなら ENTER（それ以外は中止）: ").strip():
            raise KeyboardInterrupt
        walk(robot, watch, breakout[1], "unfold", "shoulder_lift",
             breakout[1][-1]["shoulder_lift"], 0, BREAKOUT_SPEED_DEG_S, log,
             label=breakout[0])
        log(f"    wrist_flex  負荷 {watch.rows[-1]['load']}、"
            f"{watch.rows[-1]['temperature_c']} C")
        if input("  異常がなければ ENTER で継続（それ以外は中止）: ").strip():
            raise KeyboardInterrupt

        log(f"\n  --- {pad('2. ' + to_posture[0], 26)}{len(to_posture[1])} 点、"
            f"{TRANSIT_SPEED_DEG_S:.0f} deg/s ---")
        if input("  動かすなら ENTER（それ以外は中止）: ").strip():
            raise KeyboardInterrupt
        reached = walk(robot, watch, to_posture[1], "transit", "shoulder_lift",
                       upright["shoulder_lift"], 0, TRANSIT_SPEED_DEG_S, log,
                       label=to_posture[0])
        result["posture_reached_deg"] = {n: round(reached[n], 2)
                                         for n in ARM_JOINTS}
        result["upright_reached"] = True
        log("\n  UPRIGHT_REACHED — 全関節が許容範囲内です")
        result["arrivals"].append({"phase": "upright", "ok": True,
                                   "pose": result["posture_reached_deg"]})

        if not args.posture_only:
            for repeat in range(1, args.repeats + 1):
                out_stage = stages[2 + 2 * (repeat - 1)]
                back_stage = stages[3 + 2 * (repeat - 1)]
                log(f"\n  --- {repeat}/{args.repeats} 回目: "
                    f"{SAFE_DEG:+.0f} → {args.to:+.1f} deg、"
                    f"{len(out_stage[1])} 点 ---")
                if input("  進めるなら ENTER（それ以外は中止）: ").strip():
                    raise KeyboardInterrupt
                walk(robot, watch, out_stage[1], "approach", JOINT, args.to,
                     repeat, WRIST_SPEED_DEG_S, log, label=out_stage[0])

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
                walk(robot, watch, back_stage[1], "return", JOINT, SAFE_DEG,
                     repeat, WRIST_SPEED_DEG_S, log, label=back_stage[0])

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
