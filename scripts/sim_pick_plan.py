"""Phase S9, regenerated: pick trajectories with the alignment height chosen.

The first pass aligned the tool at z_align_max - the height where a vertical
tool axis *just* solves. That is a boundary, not a working height: by definition
the wrist is on its stop there, and it showed, with a joint-limit margin of
0.7 degrees at the median and nothing rated safe.

This runs the same grid twice. Once aligning at the ceiling, as before, and once
with the height searched for and scored on margin, on how far the joints travel
into and out of the pose, on contact, and on clearance. The two are reported side
by side, because "we changed it and it got better" is not a measurement.

Both runs use the servos' measured travel with a safety margin held back - the
planner does not get to spend the last few degrees the arm physically has.

    uv run scripts/sim_pick_plan.py
    uv run scripts/sim_pick_plan.py --grid 11
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np

from so101.policy.ik_policy import PREFERRED
from so101.policy.pick_plan import (
    SAFETY_MARGIN_DEG,
    PickPlanner,
    WorkspaceMask,
    measured_limits,
)
from so101.policy.tcp import ToolKinematics
from so101.sim import SO101Sim, table_height

MARK = {"safe": "#", "marginal": "+", "unreachable": "."}


def run(arm, limits, table_z, xs, ys, align_at_ceiling, neighbours):
    rows, waypoints = [], {}
    for x in xs:
        for y in ys:
            blocks = [("red", (float(x), float(y)))]
            if neighbours:
                blocks += [("blue", (float(x) + 0.035, float(y))),
                           ("green", (float(x), float(y) + 0.035))]
            sim = SO101Sim(table_z=table_z, blocks=blocks)
            planner = PickPlanner(arm, limits, sim=sim, table_z=table_z)
            record, poses = planner.plan(float(x), float(y),
                                         align_at_ceiling=align_at_ceiling)
            rows.append(record)
            waypoints[f"{x:.3f},{y:.3f}"] = poses
    return rows, waypoints


def summarise(rows):
    counts = {v: sum(r["verdict"] == v for r in rows)
              for v in ("safe", "marginal", "unreachable")}
    done = [r for r in rows if r["verdict"] != "unreachable"]
    if not done:
        return counts, None
    margins = np.array([r["min_operational_margin_deg"] for r in done])
    steps = np.array([r["max_step_deg"] for r in done])
    physical = np.array([r["min_physical_margin_deg"] for r in done])
    return counts, {
        "physical_median": float(np.median(physical)),
        "physical_worst": float(physical.min()),
        "margin_median": float(np.median(margins)),
        "margin_p5": float(np.percentile(margins, 5)),
        "margin_worst": float(margins.min()),
        "step_max": float(steps.max()),
        "step_p95": float(np.percentile(steps, 95)),
        "step_median": float(np.median(steps)),
        "total_median": float(np.median([r["total_motion_deg"] for r in done])),
    }


def draw(rows, xs, ys):
    lookup = {(r["target_x"], r["target_y"]): r["verdict"] for r in rows}
    print(f"  {'':8}y  " + "".join(f"{1000*y:+6.0f}" for y in ys[::2]))
    for x in xs:
        line = "".join(MARK[lookup[(round(float(x), 4), round(float(y), 4))]]
                       for y in ys)
        print(f"  x={1000*x:+6.0f}   {line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=9)
    parser.add_argument("--x-range", nargs=2, type=float, default=[0.12, 0.34])
    parser.add_argument("--y-range", nargs=2, type=float, default=[-0.18, 0.18])
    parser.add_argument("--neighbours", action="store_true", default=True)
    parser.add_argument("--only-chosen", action="store_true",
                        help="skip the align-at-the-ceiling run; it is already "
                             "measured and costs as much again")
    parser.add_argument("--out", type=Path, default=Path("outputs/sim"))
    args = parser.parse_args()

    arm = ToolKinematics()
    limits, found = measured_limits(arm.joint_limits_deg())
    arm.set_limits(limits)
    # Where the table is, from so101.sim - not worked out again here. This
    # block used to be the only place that knew, which is how three other
    # numbers stayed in circulation beside it.
    table_z = table_height()

    print(f"  {arm}")
    print(f"  joint limits: {'measured travel' if found else 'URDF'}, "
          f"planner holds back {SAFETY_MARGIN_DEG:.0f} deg")
    print(f"  table at {1000*table_z:+.1f} mm (the underside of the base)")

    xs = np.linspace(*args.x_range, args.grid)
    ys = np.linspace(*args.y_range, args.grid)
    print(f"  {args.grid}x{args.grid} spots\n")

    results = {}
    modes = [("align height chosen", False)] if args.only_chosen else [
        ("align at the ceiling", True), ("align height chosen", False)]
    for label, at_ceiling in modes:
        rows, waypoints = run(arm, limits, table_z, xs, ys, at_ceiling,
                              args.neighbours)
        counts, stats = summarise(rows)
        results[label] = {"rows": rows, "waypoints": waypoints,
                          "counts": counts, "stats": stats}
        print(f"  === {label} ===")
        for name, count in counts.items():
            print(f"    {name:<13}{count:>4}  ({100*count/len(rows):.0f}%)")
        draw(rows, xs, ys)
        print()

    after = results["align height chosen"]
    before = results.get("align at the ceiling")
    for label in results:
        results[label]["collisions"] = sum(
            p["collisions"] for poses in results[label]["waypoints"].values()
            for p in poses)

    if before is not None:
        print(f"  {'':<26}{'ceiling':>12}{'chosen':>12}")
        for name in ("safe", "marginal", "unreachable"):
            print(f"  {name:<26}{before['counts'][name]:>12}"
                  f"{after['counts'][name]:>12}")
        for key, label in (("margin_median", "operational margin median"),
                           ("margin_p5", "operational margin p5"),
                           ("margin_worst", "operational margin worst"),
                           ("physical_median", "physical margin median"),
                           ("physical_worst", "physical margin worst"),
                           ("step_max", "max joint movement"),
                           ("step_p95", "p95 joint movement"),
                           ("total_median", "total motion median")):
            first = ("-" if not before["stats"]
                     else f"{before['stats'][key]:.1f}")
            second = ("-" if not after["stats"]
                      else f"{after['stats'][key]:.1f}")
            print(f"  {label:<26}{first:>12}{second:>12}")
        print(f"  {'collisions':<26}{before['collisions']:>12}"
              f"{after['collisions']:>12}")
    elif after["stats"]:
        print(f"  operational margin  median {after['stats']['margin_median']:.1f}"
              f"  p5 {after['stats']['margin_p5']:.1f}"
              f"  worst {after['stats']['margin_worst']:.1f} deg")
        print(f"  physical margin     median "
              f"{after['stats']['physical_median']:.1f}"
              f"  worst {after['stats']['physical_worst']:.1f} deg")
        print(f"  joint movement      median {after['stats']['step_median']:.0f}"
              f"  p95 {after['stats']['step_p95']:.0f}"
              f"  worst {after['stats']['step_max']:.0f} deg")
        print(f"  collisions          {after['collisions']}")

    heights = [(r.get("z_align_max_mm"), r.get("z_align_opt_mm"))
               for r in after["rows"]]
    chosen = [h for _, h in heights if h]
    both = [(c, h) for c, h in heights if c and h]
    if chosen:
        print(f"\n  alignment height chosen: "
              f"{min(chosen):.0f}..{max(chosen):.0f}"
              f" mm, median {np.median(chosen):.0f}")
        print(f"  as a fraction of the ceiling: median "
              f"{np.median([h / c for c, h in both]):.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for r in after["rows"] for key in r})
    with (args.out / "pick_plan.csv").open("w", newline="",
                                           encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(after["rows"])
    (args.out / "pick_plan.json").write_text(json.dumps({
        "table_z_m": table_z, "safety_margin_deg": SAFETY_MARGIN_DEG,
        "grid": args.grid, "x_range": args.x_range, "y_range": args.y_range,
        "comparison": {label: {"counts": data["counts"], "stats": data["stats"],
                               "collisions": data["collisions"]}
                       for label, data in results.items()},
        "rows": after["rows"], "waypoints": after["waypoints"],
    }, indent=1), encoding="utf-8")

    mask = WorkspaceMask.from_rows(after["rows"])
    path = mask.save()
    disputed = sum(1 for x in np.linspace(*args.x_range, 40)
                   for y in np.linspace(*args.y_range, 40)
                   if mask.disputed(float(x), float(y)))
    print(f"\n  {mask}")
    print(f"  {100*disputed/1600:.0f}% of the table falls where the grid cells "
          f"disagree, so a lookup there should re-plan")
    print(f"  {args.out / 'pick_plan.csv'}")
    print(f"  {args.out / 'pick_plan.json'}")
    print(f"  {path}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
