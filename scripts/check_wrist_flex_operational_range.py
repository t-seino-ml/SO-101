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
that decides how much of the table can be picked from, so how far it really
goes is worth more than any other measurement on this arm.

    ONE ANGLE PER RUN. There is deliberately no way to sweep. Each run is
    looked at by a person before the next angle is chosen.

        uv run scripts/check_wrist_flex_operational_range.py --to +70 --dry-run
        uv run scripts/check_wrist_flex_operational_range.py --to +70

The posture matters as much as the angle. Swung from the folded rest pose the
wrist collides with the shoulder over the whole travel - checked against the
MuJoCo twin, lowest point 24 mm *below* the table - so the arm is first taken
to a posture where the sweep is clear. Straight up: pan 0, lift 0, elbow 0,
roll 0. There the twin puts the lowest arm point at +16 mm for every wrist_flex
from -100 to +100, and gravity asks wrist_flex for 0.117 N.m at worst, about 4%
of what the servo can hold. A load much above that is not the arm working.
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

# -- how fast anything is allowed to happen -------------------------------
#: Joint speed the interpolation asks for. Slow enough to watch, and slow
#: enough that stopping between two commands costs a fraction of a degree.
WRIST_SPEED_DEG_S = 8.0
TRANSIT_SPEED_DEG_S = 5.0
#: The arm starts folded, resting against itself - the twin reports the
#: shoulder and the lower arm already in contact at the pose it is found in.
#: Coming out of that is the one moment in the run where a joint might be
#: pushing something rather than swinging free, so the first slice of the
#: transit is taken at a crawl and looked at before the rest of it runs.
BREAKOUT_FRACTION = 0.08
BREAKOUT_SPEED_DEG_S = 1.5
#: Written into the servos' Goal_Velocity, and restored to 0 afterwards. It is
#: a ceiling under the interpolation rather than the thing that sets the speed:
#: a servo held at its velocity limit lags its command, and a lag is what the
#: tracking check is meant to be reading as a fault. Left at the factory's 0 -
#: which means "no limit" - one dropped step commands the joint at full speed.
GOAL_VELOCITY_DEG_S = 15.0
FPS = 30
#: LeRobot clamps each command this far from where the joint actually is. The
#: interpolation steps are ~0.3 degrees, so this never bites in normal running;
#: it is there for the step that should never have been sent.
MAX_RELATIVE_TARGET_DEG = 2.0

# -- when to stop ---------------------------------------------------------
# None of these is a safe level. They are the point past which the run stops
# and a person looks at it. The expected load for this sweep is around 40 of
# 1023, from the 0.117 N.m gravity asks at worst.
LOAD_ABORT = 400              # of 1023 full scale
TRACKING_ABORT_DEG = 5.0
TRACKING_RATE_ABORT_DEG_S = 2.0
TEMPERATURE_ABORT_C = 55
COMMS_ABORT = 3               # consecutive failed reads
#: During a hold: this much error, with the joint no longer moving, is a joint
#: that has stopped answering its command rather than one still on its way.
STALL_ERROR_DEG = 2.0
STALL_MOVEMENT_DEG = 0.1
STALL_WINDOW_S = 1.0

SAMPLE_HZ = 20
TICKS_PER_DEG = 4095 / 360    # LeRobot's own scale: (ticks - mid) * 360 / 4095
DEFAULT_OUT = Path("outputs/real/r1_wrist_flex")


class Abort(RuntimeError):
    """Something crossed a threshold. The run stops; the arm keeps holding."""


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
                f"\n  Refusing to start: {holder or 'another process'} is "
                f"already running this script.\n  Lock: {path}\n"
                "  Wait for it to finish, or stop it, before running again.\n"
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


# -------------------------------------------------------------------------
# reading the arm without waking it
# -------------------------------------------------------------------------

def read_arm(port):
    """Every joint's angle, limit and temperature, over the raw bus.

    Read-only, and before LeRobot is involved: connecting enables torque, and
    the transit has to be planned and checked against the twin while the arm is
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
                    f"  {name} (ID {sid}) does not answer on {port}. "
                    "Check power and the bus before going further.")
            low = bus.read(sid, MIN_ANGLE_LIMIT, 2)
            high = bus.read(sid, MAX_ANGLE_LIMIT, 2)
            ticks = bus.read(sid, PRESENT_POSITION, 2)
            if None in (low, high, ticks):
                raise SystemExit(f"  could not read {name} on {port}")
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
    return arm[joint]["mid_ticks"] + deg * TICKS_PER_DEG


def transit_times(biggest_deg):
    """(unfolding, the rest) in seconds, for a transit whose largest joint
    moves `biggest_deg`."""
    return (max(3.0, biggest_deg * BREAKOUT_FRACTION / BREAKOUT_SPEED_DEG_S),
            max(6.0, biggest_deg * (1 - BREAKOUT_FRACTION) / TRANSIT_SPEED_DEG_S))


# -------------------------------------------------------------------------
# the twin says whether the plan is clear before the arm is asked to fly it
# -------------------------------------------------------------------------

def sim_check(start_deg, posture, target_deg, log):
    """Fly the whole plan in MuJoCo first. Returns True if it is clear.

    Two things are checked and they fail differently: a contact that is *new*
    compared with where the arm already rests is a collision the plan created,
    while the contacts the parked pose already has are the arm leaning on
    itself as it sits there, and it leaves them as soon as it moves.
    """
    from so101.sim.model import SO101Sim

    table_z = -0.0085      # data/table_frame.json, the most real number there is
    bodies = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
              "moving_jaw_so101_v1")
    ignore = tuple(f"block_{i}" for i in range(16)) + \
        tuple(f"block_{i}_geom" for i in range(16))

    sim = SO101Sim(table_z=table_z)

    def contacts(pose):
        sim.set_joints(pose, gripper_deg=35.0)
        found = {f"{a}/{b}" for a, b, _ in sim.collisions(ignore=ignore)
                 if "table" not in (a, b)}
        return found, sim.lowest_point(bodies=bodies)

    start = {name: start_deg[name] for name in ARM_JOINTS}
    baseline, _ = contacts(start)
    if baseline:
        log(f"    いまの休止姿勢はすでに自身に接触しています: {sorted(baseline)}")
        log("    （折り畳まれて寄りかかっている状態です。動き出せば離れます）")

    stages = []
    goal = dict(posture, wrist_flex=SAFE_DEG)
    stages.append(("姿勢への移動", [
        {n: start[n] + (goal[n] - start[n]) * s / 60 for n in ARM_JOINTS}
        for s in range(61)]))
    stages.append((f"wrist_flex {SAFE_DEG:+.0f} → {target_deg:+.0f}", [
        dict(goal, wrist_flex=SAFE_DEG + (target_deg - SAFE_DEG) * s / 60)
        for s in range(61)]))

    clear = True
    for label, path in stages:
        lowest, offenders = float("inf"), set()
        for pose in path:
            found, low = contacts(pose)
            lowest = min(lowest, low)
            offenders |= (found - baseline)
        verdict = ("問題なし" if not offenders and lowest > table_z
                   else "*** 干渉あり ***")
        log(f"    {pad(label, 28)}最低点 {1000*lowest:+7.1f} mm   {verdict}")
        if offenders:
            log(f"      新たな自己干渉: {sorted(offenders)}")
            clear = False
        if lowest <= table_z:
            log(f"      机（{1000*table_z:+.1f} mm）に達します")
            clear = False
    return clear


# -------------------------------------------------------------------------
# watching, while it moves
# -------------------------------------------------------------------------

class Watch:
    """Samples the joint and stops the run when something stops being ordinary.

    Every sample is written out as it is taken rather than collected and saved
    at the end: the samples worth having most are the ones from the run that
    did not finish.
    """

    FIELDS = ("timestamp", "elapsed_s", "joint", "direction", "phase", "repeat",
              "target_deg", "commanded_deg", "measured_deg",
              "tracking_error_deg", "load", "current", "temperature_c",
              "voltage_v", "torque_enable", "hold_time_s", "status", "notes")

    def __init__(self, robot, writer, log, target_deg):
        self.robot = robot
        self.writer = writer
        self.log = log
        self.target_deg = target_deg
        self.direction = "positive" if target_deg >= 0 else "negative"
        self.started = time.perf_counter()
        self.history = []          # (t, measured, error)
        self.comms_failures = 0
        self.current_ever_nonzero = False
        self.rows = []

    def read_one(self, name, default=None):
        try:
            value = self.robot.bus.read(name, JOINT, normalize=False)
            self.comms_failures = 0
            return value
        except Exception:  # noqa: BLE001 - a dropped packet is not a fault yet
            self.comms_failures += 1
            if self.comms_failures >= COMMS_ABORT:
                raise Abort(f"{self.comms_failures} reads in a row failed "
                            f"({name})")
            return default

    def sample(self, phase, repeat, commanded_deg, hold_time=None,
               status="ok", notes=""):
        now = time.perf_counter()
        try:
            measured = joints_of(self.robot)[JOINT]
            self.comms_failures = 0
        except Exception as error:  # noqa: BLE001
            self.comms_failures += 1
            if self.comms_failures >= COMMS_ABORT:
                raise Abort(f"{self.comms_failures} position reads in a row "
                            f"failed ({error})") from error
            return None

        load = self.read_one("Present_Load", 0)
        current = self.read_one("Present_Current")
        temperature = self.read_one("Present_Temperature")
        voltage = self.read_one("Present_Voltage")
        torque = self.read_one("Torque_Enable")
        if current:
            self.current_ever_nonzero = True

        error_deg = measured - commanded_deg
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "elapsed_s": round(now - self.started, 3),
            "joint": JOINT, "direction": self.direction, "phase": phase,
            "repeat": repeat, "target_deg": round(self.target_deg, 3),
            "commanded_deg": round(commanded_deg, 3),
            "measured_deg": round(measured, 3),
            "tracking_error_deg": round(error_deg, 3),
            "load": load, "current": current,
            "temperature_c": temperature,
            "voltage_v": None if voltage is None else voltage / 10,
            "torque_enable": torque,
            "hold_time_s": None if hold_time is None else round(hold_time, 2),
            "status": status, "notes": notes,
        }
        self.writer.writerow(row)
        self.rows.append(row)
        self.history.append((now, measured, error_deg))
        self.history = [h for h in self.history if now - h[0] <= 2.0]

        self.judge(row, phase)
        return row

    def judge(self, row, phase):
        load = abs(row["load"] or 0)
        if load > LOAD_ABORT:
            raise Abort(f"load {row['load']} past {LOAD_ABORT} "
                        f"(the sweep should sit near 40)")
        temperature = row["temperature_c"]
        if temperature is not None and temperature > TEMPERATURE_ABORT_C:
            raise Abort(f"{temperature} C past {TEMPERATURE_ABORT_C} C")
        error = abs(row["tracking_error_deg"])
        if error > TRACKING_ABORT_DEG:
            raise Abort(f"tracking error {row['tracking_error_deg']:+.2f} deg "
                        f"past {TRACKING_ABORT_DEG} deg")

        # How fast the error is growing, over whatever the last second holds.
        older = [h for h in self.history if row["elapsed_s"] is not None
                 and self.history[-1][0] - h[0] >= 0.5]
        if older:
            span = self.history[-1][0] - older[-1][0]
            grew = abs(self.history[-1][2]) - abs(older[-1][2])
            if span > 0 and grew / span > TRACKING_RATE_ABORT_DEG_S:
                raise Abort(f"tracking error growing at {grew/span:.1f} deg/s "
                            f"past {TRACKING_RATE_ABORT_DEG_S} deg/s")

        # Only while holding: a joint that is off its command and no longer
        # moving has stopped answering. During a move, being behind is normal.
        if phase == "hold" and error > STALL_ERROR_DEG:
            window = [h for h in self.history
                      if self.history[-1][0] - h[0] <= STALL_WINDOW_S]
            if len(window) > 4:
                moved = max(h[1] for h in window) - min(h[1] for h in window)
                if moved < STALL_MOVEMENT_DEG:
                    raise Abort(f"held {error:.2f} deg off its command without "
                                f"moving for {STALL_WINDOW_S:.0f} s")


# -------------------------------------------------------------------------
# moving
# -------------------------------------------------------------------------

def glide(robot, watch, goal, seconds, phase, repeat, log):
    """Interpolate every commanded joint to `goal` over `seconds`, watching.

    Interpolated rather than commanded outright because a Feetech servo with
    Goal_Velocity at the factory's 0 goes at whatever speed it can: one
    Goal_Position write is a full-speed move. Small steps at a steady rate make
    the speed a property of this loop, which can be stopped between any two of
    them.
    """
    start = joints_of(robot)
    steps = max(1, int(seconds * FPS))
    sample_every = max(1, int(FPS / SAMPLE_HZ))
    commanded = dict(start)
    for step in range(1, steps + 1):
        commanded = {name: start[name] + (value - start[name]) * step / steps
                     for name, value in goal.items()}
        sent = robot.send_action({f"{name}.pos": value
                                  for name, value in commanded.items()})
        if step % sample_every == 0 or step == steps:
            watch.sample(phase, repeat,
                         sent.get(f"{JOINT}.pos", commanded.get(JOINT, 0.0)))
        time.sleep(1.0 / FPS)
    return commanded


def hold(robot, watch, commanded_deg, seconds, repeat, log):
    """Sit at the commanded angle and keep sampling. Returns the samples."""
    began = time.perf_counter()
    taken = []
    while time.perf_counter() - began < seconds:
        row = watch.sample("hold", repeat, commanded_deg,
                           hold_time=time.perf_counter() - began)
        if row is not None:
            taken.append(row)
        time.sleep(1.0 / SAMPLE_HZ)
    return taken


def summarise(rows, label):
    """mean / median / p95 / worst of what matters, over a set of samples."""
    import numpy as np

    if not rows:
        return {"label": label, "samples": 0}
    error = np.array([abs(r["tracking_error_deg"]) for r in rows])
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

def plan_report(args, arm, posture, out_dir, lock_path, log, sim_clear):
    wrist = arm[JOINT]
    target = args.to
    ticks = ticks_for(target, arm)
    to_min = target - wrist["min_deg"]
    to_max = wrist["max_deg"] - target
    operational = LIMIT_DEG
    goal = dict(posture, wrist_flex=SAFE_DEG)
    biggest = max(abs(goal[name] - arm[name]["deg"]) for name in goal)
    unfold_s, rest_s = transit_times(biggest)
    wrist_s = max(3.0, abs(target - SAFE_DEG) / WRIST_SPEED_DEG_S)

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
        goal = SAFE_DEG if name == JOINT else posture[name]
        log(f"    {name:<15}{arm[name]['deg']:>+9.2f}{goal:>+9.2f}"
            f"{goal - arm[name]['deg']:>+11.2f}")
    log(f"    単一関節の最大移動量 {biggest:+.2f} deg")

    log("\n  --- wrist_flex の目標 ---")
    log(f"    {pad('目標角', 28)}{target:+.2f} deg   ({ticks:.0f} ticks)")
    log(f"    {pad('ファームウェア下限', 28)}{wrist['min_deg']:+.2f} deg   "
        f"({wrist['min_ticks']} ticks)   余裕 {to_min:+.2f} deg")
    log(f"    {pad('ファームウェア上限', 28)}{wrist['max_deg']:+.2f} deg   "
        f"({wrist['max_ticks']} ticks)   余裕 {to_max:+.2f} deg")
    log(f"    {pad('R1 の上限', 28)}{operational:+.2f} deg   "
        f"余裕 {operational - abs(target):+.2f} deg")
    log(f"    {pad('プランナの運用限界', 28)}{wrist['max_deg'] - 5.0:+.2f} deg"
        f"（実可動域から安全マージン 5 deg を引いたもの）")
    log("                                このスクリプトは書き換えません")

    log("\n  --- 速度と時間 ---")
    log(f"    {pad(f'展開（最初の {100*BREAKOUT_FRACTION:.0f}%）', 28)}"
        f"{unfold_s:.1f} s / {BREAKOUT_SPEED_DEG_S:.1f} deg/s、"
        f"ここでいったん停止して待ちます")
    log(f"    {pad('残りの移動', 28)}{rest_s:.1f} s / "
        f"{TRANSIT_SPEED_DEG_S:.0f} deg/s、{FPS} Hz で補間")
    log(f"    {pad(f'wrist_flex {SAFE_DEG:+.0f} → {target:+.0f}', 28)}"
        f"{wrist_s:.1f} s / {WRIST_SPEED_DEG_S:.0f} deg/s")
    log(f"    {pad('保持', 28)}{args.hold:.1f} s、{SAMPLE_HZ} Hz で記録")
    log(f"    {pad('反復', 28)}{args.repeats} 回"
        f"（間に {SAFE_DEG:+.0f} deg へ戻します）")
    total = (unfold_s + rest_s + args.repeats * (2 * wrist_s + args.hold)
             + unfold_s + rest_s)
    log(f"    {pad('動いている時間の目安', 28)}{total:.0f} s")
    log(f"    {pad('Goal_Velocity 書き込み', 28)}{GOAL_VELOCITY_DEG_S:.0f} deg/s "
        f"({GOAL_VELOCITY_DEG_S * TICKS_PER_DEG:.0f} ticks/s)、終了時に 0 へ戻します")
    log(f"    {pad('max_relative_target', 28)}{MAX_RELATIVE_TARGET_DEG:.1f} deg "
        f"／ 1 指令")

    log("\n  --- 走行を止める条件 ---")
    log(f"    {pad('|Present_Load|', 28)}> {LOAD_ABORT} / 1023   "
        f"（ツインの予測は 40 前後）")
    log(f"    {pad('追従誤差', 28)}> {TRACKING_ABORT_DEG:.1f} deg")
    log(f"    {pad('追従誤差の増加率', 28)}> {TRACKING_RATE_ABORT_DEG_S:.1f} deg/s")
    log(f"    {pad('温度', 28)}> {TEMPERATURE_ABORT_C} C"
        f"（現在 {wrist['temperature']} C）")
    log(f"    {pad('連続した読み出し失敗', 28)}>= {COMMS_ABORT} 回")
    log(f"    {pad('指令から外れたまま静止', 28)}> {STALL_ERROR_DEG:.1f} deg のずれで "
        f"{STALL_MOVEMENT_DEG:.1f} deg も動かない状態が {STALL_WINDOW_S:.0f} s")
    log("    いずれも「ここまでは安全」という意味ではありません。異常を早く")
    log("    止めるための上限です。異音・振動・不自然な動きがあれば、数値が")
    log("    条件を満たしていなくても止めてください。")

    log("\n  --- 条件に触れたとき何が起きるか ---")
    log("    1. 全関節の Present_Position を ticks で読む")
    log("    2. Goal_Position をその Present_Position へ書き換える")
    log("       → アームは「いまいる場所」を保持します。補間を止めるだけでは、")
    log("         直前の指令がサーボに残り、追従できなかったその指令に押し")
    log("         続けることになります。")
    log("    3. 以降の軌道指令は出しません")
    log("    4. トルクは ON のまま（上げた腕を解放すれば落下します）")
    log("    5. 全関節の位置・指令・負荷・温度・電圧・トルク状態を表示し、")
    log("       summary.json に保存します")
    log("    そのうえで問いかけます。異常停止のあと腕を畳むには 'home' の")
    log("    入力が必要です。ENTER だけなら保持したままにします。")
    log("    バスが応答しない場合はその旨を表示し、scripts/torque_off.py を")
    log("    案内します。それが正しい唯一の場面です。")

    log("\n  --- 書き込み先 ---")
    log(f"    {pad('出力ディレクトリ', 28)}{out_dir}"
        f"{'（dry run のため未作成）' if args.dry_run else ''}")
    log(f"    {pad('ロック', 28)}{lock_path}（このプロセスが保持中）")
    log(f"    {pad('既存の結果', 28)}上書きしません。実行ごとに別ディレクトリです")

    log("\n  --- この計画に対するツインの判定 ---")
    log(f"    {'問題なし' if sim_clear else '*** 干渉あり — 実行を拒否します ***'}")
    return sim_clear


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
    parser.add_argument("--to", type=float, required=True, metavar="DEG",
                        help=f"確認する wrist_flex の角度。"
                             f"-{LIMIT_DEG:.0f}..+{LIMIT_DEG:.0f}")
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

    if abs(args.to) > LIMIT_DEG:
        raise SystemExit(
            f"\n  {args.to:+.1f} deg は R1 が許す ±{LIMIT_DEG:.0f} deg の"
            f"外側です。\n  上書きするフラグはありませんし、ここに足すべきでも"
            f"ありません。\n  この先は、サーボ自身の停止位置まで 6 deg を"
            f"切ります。\n")
    if args.repeats < 1:
        raise SystemExit("  --repeats は 1 以上にしてください")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    direction = "pos" if args.to >= 0 else "neg"
    out_dir = args.out / f"{stamp}_{direction}{abs(args.to):.0f}"
    lock_path = args.out / ".lock"

    from so101.hardware import resolve as resolve_port
    port = resolve_port([args.follower_port])[0]

    with only_one(lock_path):
        log = Log()
        log(f"\n  Phase R1: wrist_flex {args.to:+.1f} deg、姿勢 "
            f"{args.posture!r}、ポート {port}")
        log("  " + ("DRY RUN — 何も動かしません" if args.dry_run
                    else "LIVE — 実機が動きます"))

        arm = read_arm(port)
        posture = dict(POSTURES[args.posture])

        log("\n  --- MuJoCo ツインで計画を検査します ---")
        start_deg = {name: arm[name]["deg"] for name in ARM_JOINTS}
        clear = sim_check(start_deg, posture, args.to, log)

        ok = plan_report(args, arm, posture, out_dir, lock_path, log, clear)
        if args.dry_run:
            log("\n  Dry run です。何も動かさず、何も書き込んでいません。\n")
            return
        if not ok:
            raise SystemExit(
                "\n  ツインがこの計画に干渉を報告しました。実行を拒否します。\n")

        run(args, arm, posture, port, out_dir, log)


def run(args, arm, posture, port, out_dir, log):
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

    robot = None
    watch = None
    aborted = None
    home = {name: arm[name]["deg"] for name in ARM_JOINTS}
    goal_velocity = int(GOAL_VELOCITY_DEG_S * TICKS_PER_DEG)
    result = {
        "phase": "R1", "joint": JOINT, "target_deg": args.to,
        "direction": "positive" if args.to >= 0 else "negative",
        "posture": args.posture, "posture_deg": posture,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "port": port, "repeats_requested": args.repeats,
        "hold_s": args.hold,
        "home_pose_deg": {k: round(v, 2) for k, v in home.items()},
        "thresholds": {
            "load_abort": LOAD_ABORT,
            "tracking_abort_deg": TRACKING_ABORT_DEG,
            "tracking_rate_abort_deg_s": TRACKING_RATE_ABORT_DEG_S,
            "temperature_abort_c": TEMPERATURE_ABORT_C,
            "comms_abort": COMMS_ABORT,
        },
        "motion": {
            "wrist_speed_deg_s": WRIST_SPEED_DEG_S,
            "transit_speed_deg_s": TRANSIT_SPEED_DEG_S,
            "goal_velocity_ticks_s": goal_velocity,
            "goal_velocity_deg_s": GOAL_VELOCITY_DEG_S,
            "max_relative_target_deg": MAX_RELATIVE_TARGET_DEG,
            "fps": FPS, "p_coefficient": tuning.p_coefficient(),
        },
        "git": git_state(),
        "holds": [],
    }

    try:
        robot = make_robot_from_config(SO101FollowerConfig(
            port=port, id="follower",
            max_relative_target=MAX_RELATIVE_TARGET_DEG))
        robot.connect()
        log("\n  接続しました。トルクが入り、アームはその場を保持しています")

        for name in ARM_JOINTS:
            robot.bus.write("Goal_Velocity", name, goal_velocity)
        log(f"  アーム 5 関節の Goal_Velocity を {goal_velocity} ticks/s "
            f"({GOAL_VELOCITY_DEG_S:.0f} deg/s) に設定しました")

        watch = Watch(robot, writer, log, args.to)
        here = joints_of(robot)
        log("  読み戻し: "
            + "、".join(f"{n} {here[n]:+.1f}" for n in ARM_JOINTS))

        # -- 1. to the posture --------------------------------------------
        goal = dict(posture, wrist_flex=SAFE_DEG)
        biggest = max(abs(goal[n] - here[n]) for n in goal)
        unfold_s, rest_s = transit_times(biggest)
        log(f"\n  --- {args.posture!r} へ移動します。最大移動量 "
            f"{biggest:.1f} deg ---")
        log(f"  2 段に分けます。最初の {100*BREAKOUT_FRACTION:.0f}% "
            f"({biggest * BREAKOUT_FRACTION:.1f} deg) は "
            f"{BREAKOUT_SPEED_DEG_S:.1f} deg/s。")
        log("  そこが、自身に寄りかかった姿勢から抜け出す区間だからです。")
        log(f"  残りは {TRANSIT_SPEED_DEG_S:.0f} deg/s。")
        if input("  動かすなら ENTER（それ以外は中止）: ").strip():
            raise KeyboardInterrupt

        part = {n: here[n] + (goal[n] - here[n]) * BREAKOUT_FRACTION
                for n in goal}
        glide(robot, watch, part, unfold_s, "unfold", 0, log)
        broke_out = joints_of(robot)
        log("  折り畳みから抜けました: "
            + "、".join(f"{n} {broke_out[n]:+.2f}" for n in ARM_JOINTS))
        log(f"    wrist_flex  負荷 {watch.rows[-1]['load']}、"
            f"{watch.rows[-1]['temperature_c']} C、"
            f"追従誤差 {watch.rows[-1]['tracking_error_deg']:+.2f} deg")
        if input("  異常がなければ ENTER で継続（それ以外は中止）: ").strip():
            raise KeyboardInterrupt

        glide(robot, watch, goal, rest_s, "transit", 0, log)
        arrived = joints_of(robot)
        log("  到着: " + "、".join(f"{n} {arrived[n]:+.2f}" for n in ARM_JOINTS))
        result["posture_reached_deg"] = {n: round(arrived[n], 2)
                                         for n in ARM_JOINTS}

        # -- 2. the angle, more than once ---------------------------------
        wrist_s = max(3.0, abs(args.to - SAFE_DEG) / WRIST_SPEED_DEG_S)
        for repeat in range(1, args.repeats + 1):
            log(f"\n  --- {repeat}/{args.repeats} 回目: "
                f"{SAFE_DEG:+.0f} → {args.to:+.1f} deg を {wrist_s:.1f} s で ---")
            if input("  進めるなら ENTER（それ以外は中止）: ").strip():
                raise KeyboardInterrupt

            glide(robot, watch, {JOINT: args.to}, wrist_s, "approach", repeat, log)
            taken = hold(robot, watch, args.to, args.hold, repeat, log)
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
            glide(robot, watch, {JOINT: SAFE_DEG}, wrist_s, "return", repeat, log)

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
            park(robot, watch, home, goal_velocity, log, aborted)
            try:
                robot.disconnect()
                log("  切断しました。トルクを解放しました")
            except Exception as error:  # noqa: BLE001
                log(f"  正常に切断できませんでした: {error}")
                log("  実行してください:  uv run scripts/torque_off.py COM4")
        handle.close()
        verdict(result, args, log)
        (out_dir / "summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
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
    once, before anything is reported or decided. The arm then holds where it
    actually is rather than where it was last told to be. Torque stays on: this
    is the wrong moment to drop a raised arm, and dropping it is what releasing
    torque means.

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
        if row["voltage"] is not None:
            row["voltage_v"] = row.pop("voltage") / 10
        state["joints"][name] = row
        log(f"    {name:<15}"
            f"{row['deg'] if row['deg'] is not None else float('nan'):>+9.2f}"
            f"{str(row['ticks']):>8}{str(row['goal']):>8}"
            f"{str(row['load']):>7}{str(row['temperature_c']):>6}C"
            f"{row.get('voltage_v', 0):>6.1f}V{str(row['torque_enable']):>8}")

    if watch is not None:
        try:
            watch.sample("frozen", 0, state["joints"][JOINT]["deg"] or 0.0,
                         status="abort", notes=reason)  # noqa: E501
        except Exception:  # noqa: BLE001 - the record is not worth a second fault
            pass
    log("\n  アームは保持しています。どうするか決める前に、見て、音を"
        "聞いてください。")
    return state


def park(robot, watch, home, goal_velocity, log, aborted=None):
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
            answer = input("  畳むなら home と入力（それ以外は保持のまま）: "
                           ).strip().lower()
            if answer != "home":
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
        here = joints_of(robot)
        if watch is not None:
            watch.sample("park", 0, here[JOINT], notes="before parking")
        glide_home = max(8.0, max(abs(home[n] - here[n]) for n in home)
                         / TRANSIT_SPEED_DEG_S)
        # Straighten the wrist before folding the arm, so the gripper is not
        # swung through anything on the way down.
        _quiet_glide(robot, {JOINT: SAFE_DEG}, 4.0)
        _quiet_glide(robot, dict(home), glide_home)
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


def _quiet_glide(robot, goal, seconds):
    """Interpolate without sampling or judging - used on the way out."""
    start = joints_of(robot)
    steps = max(1, int(seconds * FPS))
    for step in range(1, steps + 1):
        robot.send_action({f"{name}.pos": start[name]
                           + (value - start[name]) * step / steps
                           for name, value in goal.items()})
        time.sleep(1.0 / FPS)


def verdict(result, args, log):
    """What the run showed. Not a joint limit - R1 does not decide one."""
    holds = result.get("holds", [])
    side = "正側" if result["direction"] == "positive" else "負側"
    log(f"\n  === wrist_flex {args.to:+.1f} deg（{side}）===")
    if result.get("aborted"):
        log(f"  完了しませんでした: {result['aborted']}")
        result["verdict"] = "aborted"
        return
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
