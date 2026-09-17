"""Phase S9's answer: the workspace a whole pick can be made in.

Phase S5 asked whether a single pose was reachable and got a clean contiguous
region. A pick is a sequence, and a sequence can fail in ways a pose cannot: the
tool has to be turned vertical somewhere the arm can still hold it, the descent
has to stay vertical all the way down, the lift has to come back up under the
same constraint, and none of it may cross the table or reconfigure the arm
halfway. This reads the trajectory run and says what survives.

    uv run scripts/sim_trajectory_report.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MARK = {"safe": "#", "marginal": "+", "unreachable": "."}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def grid_of(rows, key, fmt):
    xs = sorted({r["target_x"] for r in rows})
    ys = sorted({r["target_y"] for r in rows})
    lookup = {(r["target_x"], r["target_y"]): r for r in rows}
    return xs, ys, lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path,
                        default=Path("outputs/sim/pick_trajectory.json"))
    parser.add_argument("--workspace", type=Path,
                        default=Path("outputs/sim/workspace.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/sim/trajectory_workspace.json"))
    args = parser.parse_args()

    data = load(args.trajectory)
    rows, waypoints = data["rows"], data["waypoints"]
    xs, ys, lookup = grid_of(rows, "verdict", None)
    print(f"  {len(rows)} spots, table at {1000*data['table_z_m']:+.1f} mm, "
          f"grasping at {1000*data['grasp_z_m']:+.1f} mm")
    print(f"  neighbouring blocks placed: {data['neighbours']}\n")

    # -- 1. the trajectory-feasible workspace ------------------------------
    counts = {v: sum(r["verdict"] == v for r in rows)
              for v in ("safe", "marginal", "unreachable")}
    print(f"  trajectory-feasible workspace")
    for name, count in counts.items():
        print(f"    {name:<13}{count:>4}  ({100*count/len(rows):.0f}%)")

    print(f"\n  {'':8}y  " + "".join(f"{1000*y:+6.0f}" for y in ys[::2]))
    for x in xs:
        line = "".join(MARK[lookup[(x, y)]["verdict"]] for y in ys)
        print(f"  x={1000*x:+6.0f}   {line}")

    # -- 3. the alignment ceiling ------------------------------------------
    print(f"\n  z_align_max: the highest point above each spot where the tool")
    print(f"  still solves vertical, in mm above the grasp\n")
    print(f"  {'':8}y  " + "".join(f"{1000*y:+6.0f}" for y in ys[::2]))
    ceilings = []
    for x in xs:
        line = ""
        for y in ys:
            value = lookup[(x, y)].get("z_align_max_mm")
            line += "   -  " if not value else f"{value:>6.0f}"
            if value:
                ceilings.append(value)
        print(f"  x={1000*x:+6.0f}  {line}")
    if ceilings:
        print(f"\n    range {min(ceilings):.0f}..{max(ceilings):.0f} mm, "
              f"median {np.median(ceilings):.0f} mm")

    # -- 4. how the travel height relates to reach -------------------------
    transits = [r["transit_mm"] for r in rows if r.get("transit_mm")]
    if transits:
        chosen = {}
        for r in rows:
            if r.get("transit_mm"):
                chosen[r["transit_mm"]] = chosen.get(r["transit_mm"], 0) + 1
        print(f"\n  travel height actually used (the highest that solved):")
        for height, count in sorted(chosen.items(), reverse=True):
            print(f"    {height:>6.1f} mm   {count:>3} spots")

    # -- 5-8. movement, collisions, limits ---------------------------------
    done = [r for r in rows if r["verdict"] != "unreachable"]
    if done:
        steps = np.array([r["max_step_deg"] for r in done])
        totals = np.array([r["total_motion_deg"] for r in done])
        margins = np.array([r["min_limit_margin_deg"] for r in done])
        axes = np.array([r["max_axis_error_deg"] for r in done
                         if r.get("max_axis_error_deg") is not None])
        print(f"\n  over the {len(done)} spots a trajectory completed at:")
        print(f"    largest single step   median {np.median(steps):.0f} deg, "
              f"p95 {np.percentile(steps, 95):.0f}, worst {steps.max():.0f}")
        print(f"    total joint motion    median {np.median(totals):.0f} deg, "
              f"worst {totals.max():.0f}")
        print(f"    joint limit margin    median {np.median(margins):.1f} deg, "
              f"worst {margins.min():.1f}")
        if len(axes):
            print(f"    tool axis error       median {np.median(axes):.3f} deg, "
                  f"worst {axes.max():.3f}")
    failures = {}
    for r in rows:
        if r["verdict"] == "unreachable":
            reason = r.get("failed_at", "unknown")
            reason = reason.split(",")[0][:48]
            failures[reason] = failures.get(reason, 0) + 1
    print(f"\n  why the rest failed:")
    for reason, count in sorted(failures.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>3}  {reason}")

    collisions = sum(p["collisions"] for poses in waypoints.values()
                     for p in poses)
    below = [p["lowest_point_mm"] for poses in waypoints.values() for p in poses
             if p.get("lowest_point_mm") is not None]
    print(f"\n  collisions recorded at any waypoint: {collisions}")
    if below:
        print(f"  lowest moving part ever got to: {min(below):.1f} mm "
              f"(table at {1000*data['table_z_m']:+.1f})")
    clearances = [p["neighbour_clearance_mm"] for poses in waypoints.values()
                  for p in poses if p.get("neighbour_clearance_mm")]
    if clearances:
        print(f"  nearest a neighbouring block: {min(clearances):.0f} mm "
              f"centre to centre")

    # -- 2. against the point-wise workspace -------------------------------
    if args.workspace.is_file():
        with args.workspace.open(encoding="utf-8") as handle:
            pointwise = list(csv.DictReader(handle))
        graded = {}
        for r in rows:
            nearest = min(pointwise, key=lambda p: (
                (float(p["target_x"]) - r["target_x"]) ** 2
                + (float(p["target_y"]) - r["target_y"]) ** 2))
            gap = np.hypot(float(nearest["target_x"]) - r["target_x"],
                           float(nearest["target_y"]) - r["target_y"])
            if gap > 0.03:
                continue
            key = (nearest["verdict"], r["verdict"])
            graded[key] = graded.get(key, 0) + 1
        print(f"\n  point-wise (S5) against trajectory (S9), nearest spot:")
        print(f"    {'S5':<14}{'S9':<14}{'count':>7}")
        for (before, after), count in sorted(graded.items(),
                                             key=lambda kv: -kv[1]):
            print(f"    {before:<14}{after:<14}{count:>7}")
        kept = sum(c for (b, a), c in graded.items()
                   if b == "safe" and a in ("safe", "marginal"))
        was_safe = sum(c for (b, _), c in graded.items() if b == "safe")
        if was_safe:
            print(f"\n    of the spots S5 called safe, {100*kept/was_safe:.0f}% "
                  f"still have a workable trajectory")

    # -- 9. representative trajectories ------------------------------------
    print(f"\n  representative trajectories")
    picks = {"centre": (np.median(xs), 0.0),
             "left": (np.median(xs), min(ys)),
             "right": (np.median(xs), max(ys)),
             "near": (min(xs), 0.0),
             "far": (max(xs), 0.0)}
    for label, (wanted_x, wanted_y) in picks.items():
        key = min(lookup, key=lambda k: (k[0] - wanted_x) ** 2
                  + (k[1] - wanted_y) ** 2)
        record = lookup[key]
        print(f"\n    {label}: x={1000*key[0]:+.0f} y={1000*key[1]:+.0f} mm"
              f"   {record['verdict']}")
        poses = waypoints.get(f"{key[0]:.3f},{key[1]:.3f}", [])
        if not poses:
            print(f"      {record.get('failed_at', '')}")
            continue
        print(f"      {'waypoint':<11}{'z':>7}{'step':>8}{'margin':>9}"
              f"{'lowest':>9}")
        for p in poses:
            print(f"      {p['waypoint']:<11}{p['z_mm']:>6.0f}mm"
                  f"{p['step_deg']:>7.0f}d{p['joint_limit_margin_deg']:>8.1f}d"
                  f"{p.get('lowest_point_mm', float('nan')):>8.1f}mm")

    args.out.write_text(json.dumps({
        "verdicts": counts,
        "z_align_max_mm": {"min": min(ceilings), "max": max(ceilings),
                           "median": float(np.median(ceilings))}
        if ceilings else None,
        "failures": failures,
        "collisions": collisions,
        "motion": {"max_step_median_deg": float(np.median(steps)),
                   "max_step_p95_deg": float(np.percentile(steps, 95)),
                   "max_step_worst_deg": float(steps.max()),
                   "total_median_deg": float(np.median(totals))} if done else None,
    }, indent=1), encoding="utf-8")
    print(f"\n  {args.out}")


if __name__ == "__main__":
    main()
