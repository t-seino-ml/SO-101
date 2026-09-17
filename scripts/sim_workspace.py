"""Phase S5: where on the table can this arm actually pick something up?

Reaching a point is not the same as being able to grasp there. A grasp needs the
tool pointing the right way, the joints clear of their stops, the arm out of the
table, and the solver to find it reliably. This maps that over the table.

Two layers, kept apart on purpose:

  reachability   each point solved on its own, multi-start, seeded from the
                 standard posture and nothing else. Whether a block at some spot
                 can be picked must not depend on where the arm happened to be
                 beforehand - a map that says otherwise is describing a path,
                 not a workspace.

  continuity     the same grid walked in order, each point seeded from the last
                 answer, recording how far the joints travel between neighbours.
                 That is a property of a trajectory through the map rather than
                 of the map, so it is reported separately and never changes a
                 verdict.

The earlier version of this map used one seed per point and classified on how
far the wrist had rolled from a preferred value. Both were wrong: one seed
understates what the arm can reach, and the roll is free by design - it moves
the jaws not at all now that the TCP is what inverse kinematics targets.

    uv run scripts/sim_workspace.py
    uv run scripts/sim_workspace.py --grid 20
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np

from so101.policy.ik_policy import (
    JOINTS,
    PREFERRED,
    TILT_DEG,
    TOLERANCE_MM,
    grasp_difference,
    solve,
    tool_axis,
)
from so101.policy.tcp import ToolKinematics
from so101.sim import BLOCK_SIZE_M, SO101Sim, table_height

LIMIT_MARGIN_DEG = 8.0        # closer than this to a stop counts as marginal
AXIS_TOLERANCE_DEG = 2.0


def evaluate(arm, sim, x, y, z, tilt_deg, limits, reference=None):
    """One grid point, solved on its own terms."""
    target = np.array([x, y, z])
    solution = solve(arm, target, reference=reference, tilt_deg=tilt_deg)
    row = {"target_x": round(x, 4), "target_y": round(y, 4),
           "target_z": round(z, 4), "tilt_deg": tilt_deg}
    if solution is None:
        row.update({"ik_success": 0, "verdict": "unreachable"})
        return row, None

    reached, rotation = arm.pose(solution)
    error = 1000 * float(np.linalg.norm(reached - target))
    wanted = tool_axis(solution["shoulder_pan"], tilt_deg)
    axis_error = float(np.degrees(np.arccos(np.clip(
        float(rotation[:, 2] @ wanted), -1, 1))))
    margin = min(min(solution[name] - limits[name][0],
                     limits[name][1] - solution[name]) for name in JOINTS)

    sim.set_joints(solution, gripper_deg=30.0)
    touching = sim.hits_table()

    # Reached it, jaws pointing the right way, nothing against a stop, not in
    # the table. How far the wrist rolled is deliberately not a condition.
    if error > TOLERANCE_MM or touching:
        verdict = "unreachable"
    elif margin < LIMIT_MARGIN_DEG or axis_error > AXIS_TOLERANCE_DEG:
        verdict = "marginal"
    else:
        verdict = "safe"

    row.update({
        "ik_success": 1,
        "fk_tcp_x": round(reached[0], 5), "fk_tcp_y": round(reached[1], 5),
        "fk_tcp_z": round(reached[2], 5),
        "position_error_mm": round(error, 4),
        "orientation_error_deg": round(axis_error, 4),
        "joint_limit_margin_deg": round(margin, 2),
        "collision": int(touching),
        "verdict": verdict,
        **{name: round(solution[name], 2) for name in JOINTS},
    })
    return row, solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--x-range", nargs=2, type=float, default=[0.10, 0.40])
    parser.add_argument("--y-range", nargs=2, type=float, default=[-0.22, 0.22])
    parser.add_argument("--table-z", type=float, default=None)
    parser.add_argument("--tilt", type=float, default=TILT_DEG)
    parser.add_argument("--out", type=Path, default=Path("outputs/sim"))
    args = parser.parse_args()

    arm = ToolKinematics()
    limits = arm.joint_limits_deg()
    print(f"  {arm}")

    # The table's height, from the one place it now comes from. This used to
    # derive its own figure - the old homography's -8 mm gripper-frame height,
    # plus the TCP's offset below that frame, minus half a block - which put the
    # table somewhere between -1 and -15 mm depending on what data/tcp.json said
    # that week. A workspace map and a clearance check that disagree about where
    # the table is are not measuring the same thing.
    table_z = table_height() if args.table_z is None else args.table_z
    print(f"  table at {1000*table_z:+.1f} mm "
          f"({'the underside of the base' if args.table_z is None else 'given'})")
    z = table_z + BLOCK_SIZE_M / 2
    print(f"  grasping a 20 mm block: TCP at z={1000*z:+.1f} mm\n")

    sim = SO101Sim(table_z=table_z)
    xs = np.linspace(*args.x_range, args.grid)
    ys = np.linspace(*args.y_range, args.grid)

    # -- layer one: reachability, each point on its own --------------------
    rows, solutions = [], {}
    for x in xs:
        for y in ys:
            row, solution = evaluate(arm, sim, float(x), float(y), z,
                                     args.tilt, limits)
            rows.append(row)
            if solution is not None:
                solutions[(round(float(x), 4), round(float(y), 4))] = solution

    verdicts = {v: sum(r["verdict"] == v for r in rows)
                for v in ("safe", "marginal", "unreachable")}
    total = len(rows)
    print(f"  {args.grid}x{args.grid}, multi-start, each point independent\n")
    for name, count in verdicts.items():
        print(f"    {name:<13}{count:>5}  ({100*count/total:.0f}%)")

    got = [r for r in rows if r["ik_success"]]
    if got:
        errors = np.array([r["position_error_mm"] for r in got])
        margins = np.array([r["joint_limit_margin_deg"] for r in got])
        print(f"\n    TCP error      median {np.median(errors):.3f} mm, "
              f"p95 {np.percentile(errors, 95):.3f} mm")
        print(f"    limit margin   median {np.median(margins):.1f} deg, "
              f"worst {margins.min():.1f} deg")
        print(f"    collisions     {sum(r['collision'] for r in got)}")

    print(f"\n  looking down on the table  (# safe, + marginal, . unreachable)")
    print(f"  {'':8}y  " + "".join(f"{1000*y:+5.0f}" for y in ys[::2]))
    grid = {(r["target_x"], r["target_y"]): r["verdict"] for r in rows}
    for x in xs:
        line = "".join({"safe": "#", "marginal": "+", "unreachable": "."}
                       [grid[(round(float(x), 4), round(float(y), 4))]]
                       for y in ys)
        print(f"  x={1000*x:+6.0f}   {line}")

    # -- layer two: what a path across the map costs -----------------------
    # Walked as a snake so consecutive points are genuinely adjacent. This
    # informs trajectory planning; it does not touch a single verdict above.
    print(f"\n  walking the reachable points in order, seeded from the last:")
    previous, steps, breaks = None, [], 0
    for index, x in enumerate(xs):
        for y in (ys if index % 2 == 0 else ys[::-1]):
            key = (round(float(x), 4), round(float(y), 4))
            if key not in solutions:
                previous = None
                continue
            if previous is None:
                previous = solutions[key]
                continue
            nearby = solve(arm, np.array([key[0], key[1], z]),
                           reference=previous, tilt_deg=args.tilt)
            if nearby is None:
                breaks += 1
                previous = None
                continue
            steps.append(grasp_difference(nearby, previous))
            previous = nearby
    if steps:
        print(f"    joint move between neighbours: median "
              f"{np.median(steps):.0f} deg, p95 {np.percentile(steps, 95):.0f}, "
              f"worst {np.max(steps):.0f}")
        print(f"    steps over 90 deg: "
              f"{sum(s > 90 for s in steps)} of {len(steps)}")
    print(f"    (a separate layer: none of this changes a verdict)")

    args.out.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (args.out / "workspace.csv").open("w", newline="",
                                           encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "workspace.json").write_text(json.dumps({
        "tilt_deg": args.tilt, "table_z_m": table_z, "tcp_z_m": z,
        "grid": args.grid, "x_range": args.x_range, "y_range": args.y_range,
        "multi_start": True, "verdicts": verdicts,
        "continuity": {
            "median_deg": float(np.median(steps)) if steps else None,
            "p95_deg": float(np.percentile(steps, 95)) if steps else None,
            "worst_deg": float(np.max(steps)) if steps else None,
            "steps_over_90_deg": int(sum(s > 90 for s in steps)),
            "samples": len(steps), "breaks": breaks,
        },
    }, indent=1), encoding="utf-8")
    print(f"\n  {args.out / 'workspace.csv'}")
    print(f"  {args.out / 'workspace.json'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
