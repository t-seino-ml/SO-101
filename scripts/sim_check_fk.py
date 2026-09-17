"""Phase S1: do the two models of this arm agree?

Everything the simulation is for rests on one assumption - that the MJCF MuJoCo
loads and the URDF ikpy solves describe the same robot. They are generated from
the same Onshape CAD, which is a good reason to expect it and not a measurement
of it. If they disagree, every workspace map and every IK test afterwards is
measuring the gap between two models rather than anything about the arm.

So: drive both to the same joint angles and compare where each says the gripper
frame ended up. The MJCF's `gripperframe` site and the URDF's gripper_frame_link
are placed at identical coordinates, so the two numbers should be the same to
floating point.

    uv run scripts/sim_check_fk.py
    uv run scripts/sim_check_fk.py --samples 2000
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from so101.policy import ArmKinematics
from so101.sim import SO101Sim

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/sim/fk_check.json"))
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arm = ArmKinematics()
    sim = SO101Sim()
    print(f"  {sim}")

    limits = arm.joint_limits_deg()
    print(f"  {'joint':<16}{'urdf limits (deg)':>22}")
    for name in JOINTS:
        low, high = limits[name]
        print(f"  {name:<16}{f'{low:+.1f} .. {high:+.1f}':>22}")

    rng = np.random.default_rng(args.seed)
    rows, errors, axis_errors = [], [], []
    for trial in range(args.samples):
        pose = {name: float(rng.uniform(*limits[name])) for name in JOINTS}
        ik_position = arm.forward(pose)
        ik_axis = arm.tool_axis(pose)

        sim.set_joints(pose)
        # In the URDF's convention, so the two are directly comparable. The MJCF
        # site is turned 90 degrees about y from the URDF link; SO101Sim carries
        # that constant so nothing downstream has to remember it.
        sim_position, sim_rotation = sim.gripper_pose()
        sim_axis = sim_rotation[:, 2]

        error = 1000 * float(np.linalg.norm(ik_position - sim_position))
        axis_error = float(np.degrees(np.arccos(
            np.clip(float(ik_axis @ sim_axis), -1.0, 1.0))))
        errors.append(error)
        axis_errors.append(axis_error)
        rows.append({"trial": trial, **{k: round(v, 4) for k, v in pose.items()},
                     "ik_xyz_mm": [round(1000 * v, 4) for v in ik_position],
                     "sim_xyz_mm": [round(1000 * v, 4) for v in sim_position],
                     "position_error_mm": round(error, 6)})

    errors = np.array(errors)
    axis_errors = np.array(axis_errors)
    print(f"\n  {args.samples} random poses inside the URDF limits\n")
    print(f"  gripper frame position, ikpy against MuJoCo")
    print(f"    median {np.median(errors):8.4f} mm")
    print(f"    p95    {np.percentile(errors, 95):8.4f} mm")
    print(f"    max    {errors.max():8.4f} mm")
    print(f"\n  tool axis direction")
    print(f"    median {np.median(axis_errors):8.4f} deg")
    print(f"    max    {axis_errors.max():8.4f} deg")

    passed = errors.max() < 0.1
    print(f"\n  S1 acceptance: max position error < 0.1 mm  ->  "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        worst = int(np.argmax(errors))
        print(f"  worst at {rows[worst]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "samples": args.samples,
        "median_mm": float(np.median(errors)),
        "p95_mm": float(np.percentile(errors, 95)),
        "max_mm": float(errors.max()),
        "axis_median_deg": float(np.median(axis_errors)),
        "axis_max_deg": float(axis_errors.max()),
        "passed": bool(passed),
        "trials": rows,
    }, indent=1), encoding="utf-8")
    print(f"  {args.out}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
