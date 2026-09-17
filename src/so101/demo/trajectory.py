"""Joint waypoints that were taught on the arm, and a player that watches them.

The exhibition does not solve for anything. A person drove the arm to each pose
with the leader, watched it work, and pressed ENTER; this replays those numbers.
That is a deliberate trade: it gives up reaching arbitrary positions and buys
back the one property an exhibition actually needs, which is that what worked an
hour ago works now.

What it does not give up is watching. A taught pose is only safe while the world
matches the one it was taught in - a block in the wrong place, a hand on the arm,
a can that moved - so every waypoint is confirmed reached before the next is
sent, and the load and temperature are read the whole way. The checks are the
ones R1 arrived at, minus everything to do with inverse kinematics, which the
exhibition stack does not use and must not import.

    PHASES is the shape of a pick. A trajectory does not have to use all of them
    and may repeat them, but naming them means a saved file can be read by
    someone who was not there when it was taught.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

#: The joints a trajectory commands. The gripper is carried separately because
#: it is an event - open, close - rather than a place to be.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
              "wrist_roll")

#: The shape of a pick, named so a file can be read later.
PHASES = ("HOME", "PREGRASP", "GRASP", "CLOSE", "LIFT", "TRANSFER",
          "CAN_ABOVE", "DROP", "OPEN", "RETURN")

DEFAULT_SECONDS = 2.0
DEFAULT_SETTLE_S = 0.35
#: Within this of a waypoint counts as reached, *after* the sag has been
#: corrected for. Commanding a joint angle does not produce that joint angle:
#: the follower settles short under its own weight, with no integral term to
#: close the gap - `so101.policy.motion` says the same thing about Cartesian
#: moves and measured most of a centimetre. Measured here: elbow_flex 2.6 deg
#: short at PREGRASP, arm extended forward.
ARRIVE_DEG = 1.5
ARRIVE_TIMEOUT_S = 4.0
#: How many times to aim again, adding the miss back into the command.
#:
#: This is not only about passing the check. Teaching recorded what the follower
#: *measured* while the leader drove it, which was already one sag below what
#: the leader asked for. Replaying that number sags again, so the arm arrives a
#: sag lower than the pose that was taught and watched working - and at GRASP
#: that is the difference between the jaws going round the block and onto it.
SETTLE_ROUNDS = 3
SETTLE_SECONDS = 0.45
#: The correction may not wander further than this from the taught angle. A
#: joint held by something is not short because it sagged, and adding its error
#: back round after round would just lean on whatever is stopping it.
MAX_CORRECTION_DEG = 8.0
#: Of 1023. `so101.policy.motion` aborts at 700; this is lower because a taught
#: trajectory should meet nothing at all - anything it does meet is a surprise.
LOAD_ABORT = 450
TEMPERATURE_ABORT_C = 55
#: Nothing may be commanded further than this from where the arm is. A taught
#: file is trusted, but not so far that a corrupted number can fling the arm.
MAX_STEP_DEG = 120.0
FPS = 30

SLOT_DIR = Path("data/demo_slots")


class Unsafe(RuntimeError):
    """The trajectory or the arm is not in a state this may be replayed in."""


@dataclass
class Waypoint:
    """One pose, and how to get to it."""

    phase: str
    joints: dict            # joint -> degrees, the five arm joints
    gripper: float | None = None    # degrees, or None to leave the jaws alone
    seconds: float = DEFAULT_SECONDS
    settle_s: float = DEFAULT_SETTLE_S
    note: str = ""

    def command(self):
        """What to hand `send_action`, gripper included when it is set."""
        out = dict(self.joints)
        if self.gripper is not None:
            out["gripper"] = self.gripper
        return out

    @classmethod
    def from_dict(cls, data):
        return cls(phase=data["phase"], joints=dict(data["joints"]),
                   gripper=data.get("gripper"),
                   seconds=data.get("seconds", DEFAULT_SECONDS),
                   settle_s=data.get("settle_s", DEFAULT_SETTLE_S),
                   note=data.get("note", ""))


@dataclass
class Trajectory:
    """A named sequence of taught waypoints, and where it came from."""

    name: str
    waypoints: list = field(default_factory=list)
    slot: str | None = None
    created: str = ""
    taught_by: str = "leader teleoperation"
    note: str = ""

    # -- storage ----------------------------------------------------------

    def save(self, path=None):
        path = Path(path or SLOT_DIR / f"{self.name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "name": self.name, "slot": self.slot,
            "created": self.created or datetime.now().astimezone().isoformat(
                timespec="seconds"),
            "taught_by": self.taught_by, "note": self.note,
            "waypoints": [asdict(point) for point in self.waypoints],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} が見つかりません。先に teach_demo_slot.py で教示してください")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(name=data["name"], slot=data.get("slot"),
                   created=data.get("created", ""),
                   taught_by=data.get("taught_by", ""),
                   note=data.get("note", ""),
                   waypoints=[Waypoint.from_dict(w) for w in data["waypoints"]])

    @classmethod
    def for_slot(cls, slot, directory=None):
        directory = Path(directory or SLOT_DIR)
        return cls.load(directory / f"slot{slot}.json")

    # -- checking ---------------------------------------------------------

    def check(self, arm, start=None, log=print):
        """Is every waypoint inside the servos' limits and reachable from here?

        `arm` is what `read_limits` returns: the servos' own Min/Max, read off
        the hardware rather than taken from a file, because the file is not what
        stops the joint.

        Raises Unsafe rather than returning a flag. A trajectory that fails this
        is not something to carry on with carefully.
        """
        problems = []
        log(f"    {'phase':<12}{'joint':<16}{'角度':>9}{'限界まで':>11}")
        worst = {}
        for point in self.waypoints:
            for name, value in point.joints.items():
                if name not in arm:
                    problems.append(f"{point.phase}: 未知の関節 {name}")
                    continue
                low, high = arm[name]["min_deg"], arm[name]["max_deg"]
                room = min(value - low, high - value)
                if room < 0:
                    problems.append(
                        f"{point.phase}: {name} {value:+.1f} deg は "
                        f"{low:+.1f}..{high:+.1f} の外")
                if name not in worst or room < worst[name][0]:
                    worst[name] = (room, value, point.phase)
        for name in ARM_JOINTS:
            if name in worst:
                room, value, phase = worst[name]
                log(f"    {phase:<12}{name:<16}{value:>+9.2f}{room:>9.1f} deg")

        if start is not None:
            first = self.waypoints[0]
            step = max(abs(first.joints[n] - start[n]) for n in first.joints)
            log(f"    いまの姿勢から最初の waypoint まで {step:.1f} deg")
            if step > MAX_STEP_DEG:
                problems.append(
                    f"最初の waypoint が現在姿勢から {step:.0f} deg 離れています"
                    f"（上限 {MAX_STEP_DEG:.0f}）")

        if problems:
            for problem in problems:
                log(f"    *** {problem}")
            raise Unsafe("; ".join(problems))
        log("    すべて joint limit の内側です")
        return True

    def phases(self):
        return [point.phase for point in self.waypoints]

    def __str__(self):
        return (f"Trajectory({self.name}, {len(self.waypoints)} waypoints: "
                + " → ".join(self.phases()) + ")")


# -------------------------------------------------------------------------
# reading the arm
# -------------------------------------------------------------------------

def read_limits(port):
    """The servos' own Min/Max_Position_Limit, in the degrees LeRobot uses.

    Read from the hardware, not from the calibration file: the file is what
    normalises the angles, and the registers are what actually stop the joint.
    Read-only, and before anything is connected.
    """
    from ..hardware.sts3215 import (
        Bus, JOINT_NAMES, MAX_ANGLE_LIMIT, MIN_ANGLE_LIMIT, PRESENT_POSITION,
        PRESENT_TEMPERATURE, TORQUE_ENABLE,
    )

    out = {}
    with Bus(port) as bus:
        for sid, name in JOINT_NAMES.items():
            if not bus.ping(sid):
                raise Unsafe(f"{name} (ID {sid}) が {port} で応答しません")
            low = bus.read(sid, MIN_ANGLE_LIMIT, 2)
            high = bus.read(sid, MAX_ANGLE_LIMIT, 2)
            ticks = bus.read(sid, PRESENT_POSITION, 2)
            if None in (low, high, ticks):
                raise Unsafe(f"{name} を読めませんでした")
            mid = (low + high) / 2
            out[name] = {
                "id": sid, "ticks": ticks, "mid_ticks": mid,
                "deg": (ticks - mid) * 360 / 4095,
                "min_deg": (low - mid) * 360 / 4095,
                "max_deg": (high - mid) * 360 / 4095,
                "torque": bus.read(sid, TORQUE_ENABLE, 1),
                "temperature": bus.read(sid, PRESENT_TEMPERATURE, 1),
            }
    return out


def joints_of(robot):
    """The arm's pose, as {joint: degrees}."""
    return {key.removesuffix(".pos"): float(value)
            for key, value in robot.get_observation().items()
            if key.endswith(".pos")}


def health(robot, joint=None):
    """(worst |load|, its joint, worst temperature) across the arm."""
    load, where, temperature = 0, None, 0
    for name in (*ARM_JOINTS, "gripper") if joint is None else (joint,):
        try:
            value = abs(robot.bus.read("Present_Load", name, normalize=False))
            if value > load:
                load, where = value, name
            temperature = max(temperature, robot.bus.read(
                "Present_Temperature", name, normalize=False))
        except Exception:  # noqa: BLE001 - a dropped packet is not a fault
            continue
    return load, where, temperature


def freeze(robot, log=print):
    """Hold where the arm is, not where it was last told to be.

    Cutting the motion is not stopping: the last Goal_Position stands in the
    servo and it keeps pushing towards it. Torque stays on - releasing a raised
    arm drops it.
    """
    for attempt in range(1, 4):
        try:
            ticks = robot.bus.sync_read("Present_Position", normalize=False,
                                        num_retry=5)
            robot.bus.sync_write("Goal_Position", ticks, normalize=False,
                                 num_retry=5)
            log("  その場保持: Goal_Position を現在位置へ書き換えました"
                "（トルクは ON のまま）")
            return True
        except Exception as error:  # noqa: BLE001
            log(f"  その場保持 {attempt}/3 回目に失敗: {error}")
    log("  *** その場保持に失敗しました。アームを支えて "
        "uv run scripts/torque_off.py COM4 ***")
    return False


# -------------------------------------------------------------------------
# playing
# -------------------------------------------------------------------------

def _clamp(value, taught, limit):
    """A corrected command, kept near the taught angle and inside the servo."""
    low = taught - MAX_CORRECTION_DEG
    high = taught + MAX_CORRECTION_DEG
    if limit is not None:
        low = max(low, limit["min_deg"] + 1.0)
        high = min(high, limit["max_deg"] - 1.0)
    return max(low, min(high, value))


def play(robot, trajectory, log=print, on_sample=None, speed=1.0,
         arrive_deg=ARRIVE_DEG, confirm=None, limits=None):
    """Replay a taught trajectory, watching the arm the whole way.

    Returns a list of per-waypoint records. Raises Unsafe on anything that
    should stop the run; the caller freezes and decides what to do.

    `confirm` is called before each waypoint with the waypoint; returning False
    stops the run cleanly. Used for the first cautious replays and left out once
    a trajectory is trusted.

    `limits` is what `read_limits` returned, so a sag correction cannot walk a
    command outside the servo's own stops. Optional, and only because the
    correction is already capped at MAX_CORRECTION_DEG from the taught angle.
    """
    from ..policy.motion import glide_to

    records = []
    for index, point in enumerate(trajectory.waypoints, 1):
        here = joints_of(robot)
        step = max((abs(value - here[name])
                    for name, value in point.joints.items()), default=0.0)
        if step > MAX_STEP_DEG:
            raise Unsafe(
                f"waypoint {index} ({point.phase}) は現在姿勢から {step:.0f} deg "
                f"離れています（上限 {MAX_STEP_DEG:.0f}）")

        seconds = max(0.4, point.seconds / max(speed, 1e-3))
        log(f"  [{index}/{len(trajectory.waypoints)}] {point.phase:<10} "
            f"最大 {step:5.1f} deg を {seconds:.1f} s"
            + (f"   gripper → {point.gripper:+.0f}"
               if point.gripper is not None else ""))
        if confirm is not None and not confirm(point):
            raise Unsafe("操作者が中止しました")

        glide_to(robot, point.command(), seconds=seconds, fps=FPS)
        time.sleep(point.settle_s)

        # Aim, look at where it actually went, and add the miss back into the
        # aim. A couple of rounds of that and the joint is where it was taught,
        # rather than a sag below it.
        aim = dict(point.joints)
        corrections = 0
        previous_worst = None
        while True:
            reached = joints_of(robot)
            errors = {name: reached[name] - value
                      for name, value in point.joints.items()}
            worst = max(errors, key=lambda n: abs(errors[n]))
            load, hot_joint, temperature = health(robot)
            if load > LOAD_ABORT:
                raise Unsafe(
                    f"{point.phase}: {hot_joint} の負荷 {load} が "
                    f"{LOAD_ABORT} を超えました")
            if temperature > TEMPERATURE_ABORT_C:
                raise Unsafe(f"{point.phase}: 温度 {temperature} C")
            if abs(errors[worst]) <= arrive_deg:
                break
            if corrections >= SETTLE_ROUNDS:
                raise Unsafe(
                    f"{point.phase}: {corrections} 回補正しても到達せず。"
                    f"{worst} が {errors[worst]:+.2f} deg ずれ"
                    f"（許容 {arrive_deg:.1f}）")
            # Not shrinking means something is holding it, and leaning harder on
            # that is the wrong answer.
            if previous_worst is not None \
                    and abs(errors[worst]) > previous_worst - 0.2:
                raise Unsafe(
                    f"{point.phase}: 補正しても誤差が縮みません"
                    f"（{previous_worst:.2f} → {abs(errors[worst]):.2f} deg）。"
                    f"{worst} が何かに当たっている可能性があります")
            previous_worst = abs(errors[worst])

            corrections += 1
            aim = {name: _clamp(value - errors[name], point.joints[name],
                                limits.get(name) if limits else None)
                   for name, value in aim.items()}
            log(f"      補正 {corrections}/{SETTLE_ROUNDS}: {worst} が "
                f"{errors[worst]:+.2f} deg 手前 → 指令を "
                f"{aim[worst] - point.joints[worst]:+.2f} deg ずらします")
            command = dict(aim)
            if point.gripper is not None:
                command["gripper"] = point.gripper
            glide_to(robot, command, seconds=SETTLE_SECONDS, fps=FPS)
            time.sleep(point.settle_s)

        record = {
            "index": index, "phase": point.phase,
            "commanded": {n: round(v, 2) for n, v in point.joints.items()},
            "reached": {n: round(reached[n], 2) for n in point.joints},
            "worst_joint": worst,
            "worst_error_deg": round(errors[worst], 2),
            "corrections": corrections,
            "commanded_after_correction": {n: round(v, 2)
                                           for n, v in aim.items()},
            "gripper_deg": round(reached.get("gripper", 0.0), 2),
            "load": load, "load_joint": hot_joint,
            "temperature_c": temperature,
            "seconds": round(seconds, 2),
            "at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        }
        records.append(record)
        log(f"      到達 {worst} {errors[worst]:+.2f} deg、負荷 {load}"
            f"（{hot_joint}）、{temperature} C"
            + (f"、補正 {corrections} 回" if corrections else ""))
        if on_sample is not None:
            on_sample(record)
    return records
