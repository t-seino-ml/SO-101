"""Phase S3 and S4: can inverse kinematics hit the TCP, and hit it repeatably?

Two questions, and they are not the same one.

S3 asks whether the solver can reach a pose at all: give it a TCP pose that is
known to be reachable - taken from a random set of joint angles, so a solution
provably exists - and see how far its answer lands from it.

S4 asks the question that actually broke the real arm: whether two runs at the
*same* target give the same grasp. Position-only inverse kinematics on a 5-joint
arm leaves two degrees of freedom, and the solver spends them however it likes.
Measured on the real robot, two poses whose gripper frames were 1 mm apart had
wrists 62 degrees apart, which puts the jaws 20 mm apart. The grasp tolerates
5.6 mm.

So each solver setting is scored twice: how close it gets, and how far apart two
answers to one target can be.

    uv run scripts/sim_ik_roundtrip.py
    uv run scripts/sim_ik_roundtrip.py --samples 400
"""

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

from so101.policy import ArmKinematics
from so101.policy.tcp import ToolKinematics

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# Poses the arm actually works in: reaching out over the table, not folded back
# on itself. Sampling the full joint ranges asks the solver about configurations
# no grasp would ever use.
SAMPLING = {"shoulder_pan": (-45, 45), "shoulder_lift": (-70, -10),
            "elbow_flex": (20, 90), "wrist_flex": (10, 70),
            "wrist_roll": (-40, 40)}

SETTINGS = {
    "position only, roll free": dict(orientation=None, frozen=()),
    "tool axis pinned": dict(orientation="axis", frozen=()),
    "roll frozen": dict(orientation=None, frozen=("wrist_roll",)),
    "tool axis + roll frozen": dict(orientation="axis",
                                    frozen=("wrist_roll",)),
}


#: How the arm was driven before this phase: solve for gripper_frame_link, then
#: add the jaw offset to the answer. The offset is fixed in the gripper's frame,
#: so it has to be rotated into the posture the solver chose - and the posture is
#: only known after the solve, which is the whole problem. Two rounds of that is
#: what scripts/pick_block.py does today.
OLD_WAY = "the old way: solve for the gripper frame, add the offset"


def solve(arm, setting, target, axis, seed):
    kwargs = dict(setting)
    if kwargs.pop("orientation") == "axis":
        kwargs["orientation"] = axis
    return arm.inverse(target, seed_deg=seed, tolerance_mm=50.0, **kwargs)


def solve_old_way(plain, arm, target, seed, rounds=3):
    """Aim the gripper frame so that the TCP lands on `target`, iteratively.

    `plain` really has to be a separate ArmKinematics: its chain ends at
    gripper_frame_link, which is what the old code solved for. Reusing the
    tool chain here solves for the TCP instead and quietly measures nothing -
    the error comes out as exactly the offset's length, which is the giveaway.
    """
    aim = np.asarray(target, float)
    solution = None
    for _ in range(rounds):
        solution = plain.inverse(aim, seed_deg=seed, tolerance_mm=50.0,
                                 orientation=None)
        if solution is None:
            return None
        rotation = arm.pose(solution)[1]
        aim = np.asarray(target, float) - rotation @ arm.tcp
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/sim/ik_roundtrip.csv"))
    args = parser.parse_args()

    arm = ToolKinematics()
    # The chain the old code solved for, ending at gripper_frame_link.
    plain = ArmKinematics()
    print(f"  {arm}\n")

    rng = np.random.default_rng(args.seed)
    rows = []
    for trial in range(args.samples):
        truth = {name: float(rng.uniform(*SAMPLING[name])) for name in JOINTS}
        target, rotation = arm.pose(truth)
        axis = rotation[:, 2]

        # Two seeds per target: a neutral one and a scattered one. If the answer
        # depends on where the solver started, it is not a repeatable grasp.
        seeds = [{name: 0.0 for name in JOINTS},
                 {name: float(rng.uniform(*SAMPLING[name])) for name in JOINTS}]

        for label, setting in (*SETTINGS.items(), (OLD_WAY, None)):
            answers = []
            for index, seed in enumerate(seeds):
                solution = (solve_old_way(plain, arm, target, seed)
                            if setting is None
                            else solve(arm, setting, target, axis, seed))
                if solution is None:
                    rows.append({"trial": trial, "setting": label, "seed": index,
                                 "ik_success": 0})
                    continue
                reached, reached_rotation = arm.pose(solution)
                answers.append((solution, reached))
                rows.append({
                    "trial": trial, "setting": label, "seed": index,
                    "ik_success": 1,
                    "target_x": round(target[0], 5),
                    "target_y": round(target[1], 5),
                    "target_z": round(target[2], 5),
                    "fk_tcp_x": round(reached[0], 5),
                    "fk_tcp_y": round(reached[1], 5),
                    "fk_tcp_z": round(reached[2], 5),
                    "position_error_mm": round(
                        1000 * float(np.linalg.norm(reached - target)), 4),
                    "axis_error_deg": round(float(np.degrees(np.arccos(np.clip(
                        float(reached_rotation[:, 2] @ axis), -1, 1)))), 4),
                    "wrist_roll_deg": round(solution["wrist_roll"], 3),
                })
            if len(answers) == 2:
                (first, _), (second, _) = answers
                # What a second solution to the same target costs in grasp terms.
                rows[-1]["tcp_spread_mm"] = round(1000 * float(np.linalg.norm(
                    arm.forward(first) - arm.forward(second))), 4)
                rows[-1]["roll_spread_deg"] = round(
                    abs(first["wrist_roll"] - second["wrist_roll"]), 3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trial", "setting", "seed", "ik_success", "target_x", "target_y",
              "target_z", "fk_tcp_x", "fk_tcp_y", "fk_tcp_z",
              "position_error_mm", "axis_error_deg", "wrist_roll_deg",
              "tcp_spread_mm", "roll_spread_deg"]
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {args.samples} reachable TCP poses, two seeds each\n")
    header = (f"  {'setting':<26}{'solved':>8}{'median':>10}{'p95':>9}"
              f"{'max':>9}{'axis':>9}{'same target, two seeds':>26}")
    print(header)
    print(f"  {'':<26}{'':>8}{'position error (mm)':>28}{'(deg)':>9}"
          f"{'TCP apart / roll apart':>26}")
    for label in (*SETTINGS, OLD_WAY):
        got = [r for r in rows if r["setting"] == label and r["ik_success"]]
        attempts = [r for r in rows if r["setting"] == label]
        if not got:
            print(f"  {label:<26}{'none':>8}")
            continue
        errors = np.array([r["position_error_mm"] for r in got])
        axes = np.array([r["axis_error_deg"] for r in got])
        spread = np.array([r["tcp_spread_mm"] for r in rows
                           if r["setting"] == label and "tcp_spread_mm" in r])
        rolls = np.array([r["roll_spread_deg"] for r in rows
                          if r["setting"] == label and "roll_spread_deg" in r])
        both = (f"{np.median(spread):.1f} mm / {np.median(rolls):.0f} deg"
                if len(spread) else "-")
        print(f"  {label:<26}{100*len(got)/len(attempts):7.0f}%"
              f"{np.median(errors):10.3f}{np.percentile(errors, 95):9.3f}"
              f"{errors.max():9.2f}{np.median(axes):9.2f}{both:>26}")

    print(f"\n  S3 acceptance: median TCP error < 1 mm")
    print(f"  S4 acceptance: two seeds at one target land < 1 mm apart")
    for label in (*SETTINGS, OLD_WAY):
        got = [r for r in rows if r["setting"] == label and r["ik_success"]]
        spread = [r["tcp_spread_mm"] for r in rows
                  if r["setting"] == label and "tcp_spread_mm" in r]
        if not got or not spread:
            continue
        s3 = np.median([r["position_error_mm"] for r in got]) < 1.0
        s4 = np.median(spread) < 1.0
        print(f"    {label:<26}S3 {'PASS' if s3 else 'FAIL'}   "
              f"S4 {'PASS' if s4 else 'FAIL'}")
    print(f"\n  {args.out}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
