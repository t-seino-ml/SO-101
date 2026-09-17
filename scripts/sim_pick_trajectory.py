"""Phase S9: can the arm get to a block, and back up, without anything going wrong?

Phase S5 asked whether one pose was reachable. A pick is five of them in a row,
and the row is what has to work: a point the arm can stand at is useless if it
cannot get there without flipping its elbow through the table on the way.

The shape of the trajectory is not assumed. Probing the heights first showed that
a vertical tool axis stops working somewhere between 40 and 90 mm above the
grasp, depending where on the table you are - so "descend vertically from a safe
height" is not available everywhere, and the approach has to be in stages:

    transit    high, orientation not constrained, just getting there
    align      at the highest point where vertical still solves, tool turned
               vertical - done here, clear of the other blocks, not next to them
    descend    straight down, vertical held, in steps
    grasp      jaws closed
    lift       straight back up

Waypoints after the first are seeded from the previous solution and scored on
what a trajectory actually pays: how far the joints have to move, how close they
end up to a stop, and whether anything is in collision. Solving each in isolation
gives poses that are individually fine and jump 300 degrees between them.

    uv run scripts/sim_pick_trajectory.py
    uv run scripts/sim_pick_trajectory.py --grid 8
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
    grasp_difference,
    seeds,
    solve,
    tool_axis,
)
from so101.policy.tcp import ToolKinematics
from so101.sim import BLOCK_SIZE_M, SO101Sim

TRANSIT_M = 0.12              # how high the arm travels between blocks
DESCENT_STEPS = 3
LIFT_M = 0.06
LIMIT_MARGIN_DEG = 8.0
AXIS_TOLERANCE_DEG = 2.0
JUMP_DEG = 90.0               # a configuration change this large is not safe
MARGINAL_JUMP_DEG = 45.0
GOOD_ENOUGH_DEG = 20.0
CLEARANCE_M = 0.030           # least alignment height worth having
#: Blocks are kept out of the contact test. See SO101Sim.collisions.
BLOCK_BODIES = tuple(f"block_{i}" for i in range(12)) +     tuple(f"block_{i}_geom" for i in range(12))
#: How near a finger may come to a block that is not the target. The jaws are
#: about 27 mm across the outside at the tip, so half of that plus a little.
FINGER_HALF_M = 0.018
#: Everything that moves. The base is left out of the table-clearance check for
#: the obvious reason: it is bolted to the table, so it touches it by design.
MOVING_BODIES = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
                 "moving_jaw_so101_v1")


def measured_limits(urdf_limits):
    """Joint limits from the servos' own calibration, not the URDF's caution.

    The URDF states a conservative range and the calibration measures the real
    mechanical travel, and they differ in both directions: wrist_flex really has
    212 degrees against the URDF's 190, while wrist_roll has 266 against 320.
    That matters here because holding the tool vertical puts wrist_flex on its
    stop over most of the table - on the URDF's stop, which the arm does not
    actually have.

    The centre is taken from the URDF, which fixes where zero is; only the span
    comes from the measurement.
    """
    import os

    root = Path(os.environ.get("USERPROFILE", "~")).expanduser()
    path = (root / ".cache/huggingface/lerobot/calibration/robots"
            / "so_follower/follower.json")
    if not path.is_file():
        return dict(urdf_limits), False
    stored = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name, (low, high) in urdf_limits.items():
        entry = stored.get(name)
        if not isinstance(entry, dict) or "range_min" not in entry:
            out[name] = (low, high)
            continue
        span = (entry["range_max"] - entry["range_min"]) / 4096 * 360
        centre = (low + high) / 2
        out[name] = (centre - span / 2, centre + span / 2)
    return out, True


def align_ceiling(arm, x, y, grasp_z, highest=0.10, resolution=0.005):
    """The highest point above (x, y) where the tool still solves vertical.

    Binary search rather than a scan: the probe showed feasibility is contiguous
    in z - a run of heights that work and then nothing - so the boundary can be
    bracketed instead of walked. Five solves instead of twenty.
    """
    def reaches(height):
        return solve(arm, np.array([x, y, grasp_z + height]),
                     reference=PREFERRED, multi_start=False) is not None

    if not reaches(0.0):
        return None
    low, high = 0.0, highest
    if reaches(high):
        return high
    while high - low > resolution:
        middle = (low + high) / 2
        if reaches(middle):
            low = middle
        else:
            high = middle
    return low


def free_reach(arm, target, reference=None):
    """Reach a point with the tool left free, restarting across the seeds.

    Multi-seed for the same reason every other reachability question here is:
    a single seed failing is not evidence that a point is out of reach, and
    checking the travel height with one while the alignment height had been
    checked with several made the transit look impossible at points the arm
    could plainly get to.
    """
    for seed in seeds(reference or PREFERRED):
        answer = arm.inverse(np.asarray(target, float), seed_deg=seed,
                             tolerance_mm=1.0, orientation=None)
        if answer is not None:
            return answer
    return None


def neighbour_clearance(arm, sim, pose, target_index=0):
    """How close the TCP comes to a block that is not the one being picked.

    A stand-in for a swept-volume check, and honest about being one: it measures
    centre to centre in the table plane, which is what matters for a gripper
    coming straight down. Returns None when there is nothing else on the table.
    """
    tcp = arm.pose(pose)[0]
    nearest = None
    for index in range(len(sim.blocks)):
        if index == target_index:
            continue
        other = sim.geom_position(f"block_{index}_geom")
        gap = float(np.linalg.norm(tcp[:2] - other[:2]))
        nearest = gap if nearest is None else min(nearest, gap)
    return nearest


def step_cost(arm, sim, answer, previous, limits):
    """What a candidate solution costs: movement, margin, and contact.

    The three penalties are deliberately not commensurate - a collision has to
    lose to anything that is not one, and a joint on its stop has to lose to a
    long move. So they are scaled to sit in different decades rather than
    tuned against each other.
    """
    motion = grasp_difference(answer, previous)
    margin = min(min(answer[name] - limits[name][0],
                     limits[name][1] - answer[name]) for name in JOINTS)
    sim.set_joints(answer, gripper_deg=35.0)
    clashes = len(sim.collisions(ignore=BLOCK_BODIES))
    return (motion
            + 10.0 * max(0.0, LIMIT_MARGIN_DEG - margin)
            + 1000.0 * clashes)


def waypoint(arm, sim, target, previous, limits, vertical=True, first=False):
    """Solve one waypoint, preferring answers near the one before it."""
    if vertical:
        cost = (None if first else
                lambda answer: step_cost(arm, sim, answer, previous, limits))
        return solve(arm, target, reference=previous or PREFERRED,
                     cost=cost,
                     good_enough_deg=None if first else GOOD_ENOUGH_DEG)
    # Transit: getting there is all that matters, so the tool is left alone.
    best, best_cost = None, np.inf
    for seed in seeds(previous or PREFERRED):
        answer = arm.inverse(np.asarray(target, float), seed_deg=seed,
                             tolerance_mm=1.0, orientation=None)
        if answer is None:
            continue
        scored = (0.0 if previous is None
                  else step_cost(arm, sim, answer, previous, limits))
        if scored < best_cost:
            best, best_cost = answer, scored
    return best


def trajectory(arm, sim, x, y, grasp_z, limits, table_z=0.0,
               transit_m=TRANSIT_M):
    """The whole pick at one spot. Returns a record of how it went."""
    record = {"target_x": round(x, 4), "target_y": round(y, 4),
              "grasp_z": round(grasp_z, 4)}

    ceiling = align_ceiling(arm, x, y, grasp_z)
    record["z_align_max_mm"] = (None if ceiling is None
                                else round(1000 * ceiling, 1))
    if ceiling is None:
        record.update({"verdict": "unreachable", "failed_at": "grasp height"})
        return record, []
    if ceiling < CLEARANCE_M:
        record.update({"verdict": "unreachable",
                       "failed_at": f"vertical only holds to "
                                    f"{1000*ceiling:.0f} mm, too low to turn "
                                    f"the tool clear of the other blocks"})
        return record, []

    # How high the arm travels is not something to insist on. Asking for a fixed
    # 120 mm put the first waypoint outside the workspace over much of the
    # table - the arm cannot fold that tall close in, or reach that high far
    # out - so take the highest of a few that works, down to the alignment
    # height itself, below which there is nothing to gain.
    transit = None
    for height in (transit_m, 0.10, 0.08, 0.06, ceiling + 0.02, ceiling):
        if height < ceiling:
            continue
        if free_reach(arm, np.array([x, y, grasp_z + height])) is not None:
            transit = height
            break
    if transit is None:
        record.update({"verdict": "unreachable", "failed_at": "transit",
                       "why": "no travel height above the alignment height "
                              "is reachable"})
        return record, []
    record["transit_mm"] = round(1000 * transit, 1)

    # Solve the alignment pose first, even though it is executed second. It is
    # the constrained one - tool vertical - so it has the fewest answers, and
    # letting the transit pose be chosen freely and then demanding vertical
    # afterwards is what produced 275 degree reconfigurations between the two.
    # Constrain first, then pick the travel pose that is nearest to it.
    anchor = solve(arm, np.array([x, y, grasp_z + ceiling]), reference=PREFERRED)
    if anchor is None:
        record.update({"verdict": "unreachable", "failed_at": "align"})
        return record, []

    plan = [("transit", np.array([x, y, grasp_z + transit]), False),
            ("align", np.array([x, y, grasp_z + ceiling]), True)]
    for step in range(1, DESCENT_STEPS + 1):
        height = ceiling * (DESCENT_STEPS - step) / DESCENT_STEPS
        plan.append((f"descend {step}", np.array([x, y, grasp_z + height]), True))
    # Lifting is still a vertical move, so it cannot go above the height where
    # vertical stops solving. Asking for a fixed 60 mm failed wherever the
    # ceiling was lower than that - which is most of the far half of the table.
    plan.append(("lift", np.array([x, y, grasp_z + min(LIFT_M, ceiling)]), True))

    poses, steps, previous = [], [], None
    for index, (name, target, vertical) in enumerate(plan):
        # The travel pose is seeded from the alignment pose it has to hand over
        # to, so the two are neighbours by construction.
        answer = waypoint(arm, sim, target, previous or anchor, limits, vertical,
                          first=False)
        if answer is None:
            record.update({"verdict": "unreachable", "failed_at": name})
            return record, poses
        reached, rotation = arm.pose(answer)
        error = 1000 * float(np.linalg.norm(reached - target))
        axis_error = (float(np.degrees(np.arccos(np.clip(float(
            rotation[:, 2] @ tool_axis(answer["shoulder_pan"], TILT_DEG)),
            -1, 1)))) if vertical else None)
        margin = min(min(answer[name_] - limits[name_][0],
                         limits[name_][1] - answer[name_]) for name_ in JOINTS)
        sim.set_joints(answer, gripper_deg=35.0)
        clashes = [c for c in sim.collisions(ignore=BLOCK_BODIES)
                   if "table" not in (c[0], c[1])]
        # The table is checked by geometry rather than by contact: the gripper's
        # convex hull hangs below its own fingertips, so a contact query calls
        # every grasp a crash. The vertices are the real shape.
        lowest = sim.lowest_point(bodies=MOVING_BODIES)
        if lowest < table_z - 0.001:
            clashes = clashes + [("table", "arm vertices", 1000 * (table_z - lowest))]
        # The blocks are excluded from contact and measured by distance instead;
        # MuJoCo's convex hull of the gripper has no gap between its fingers, so
        # reaching for a block always reads as hitting it. What this catches is
        # the fingers arriving where a *neighbouring* block already is.
        clearance = neighbour_clearance(arm, sim, answer)

        if previous is not None:
            steps.append(grasp_difference(answer, previous))
        poses.append({"waypoint": name, "z_mm": round(1000 * target[2], 1),
                      "position_error_mm": round(error, 3),
                      "axis_error_deg": (None if axis_error is None
                                         else round(axis_error, 3)),
                      "joint_limit_margin_deg": round(margin, 2),
                      "collisions": len(clashes),
                      "lowest_point_mm": round(1000 * lowest, 1),
                      "neighbour_clearance_mm": (None if clearance is None
                                                 else round(1000 * clearance, 1)),
                      "step_deg": round(steps[-1], 1) if steps else 0.0,
                      **{j: round(answer[j], 2) for j in JOINTS}})

        if error > 1.0 or clashes:
            record.update({"verdict": "unreachable", "failed_at": name,
                           "why": ("collision: " + ", ".join(
                               f"{a}/{b}" for a, b, _ in clashes[:2]))
                           if clashes else f"missed by {error:.1f} mm"})
            return record, poses
        previous = answer

    margins = [p["joint_limit_margin_deg"] for p in poses]
    axis_errors = [p["axis_error_deg"] for p in poses
                   if p["axis_error_deg"] is not None]
    record.update({
        "waypoints": len(poses),
        "max_step_deg": round(max(steps), 1) if steps else 0.0,
        "total_motion_deg": round(sum(steps), 1),
        "min_limit_margin_deg": round(min(margins), 2),
        "max_axis_error_deg": round(max(axis_errors), 3) if axis_errors else None,
        "collisions": 0,
    })
    if record["max_step_deg"] > JUMP_DEG or record["min_limit_margin_deg"] < 0:
        record["verdict"] = "unreachable"
        record["failed_at"] = "configuration jump" \
            if record["max_step_deg"] > JUMP_DEG else "joint limit"
    elif (record["min_limit_margin_deg"] < LIMIT_MARGIN_DEG
          or record["max_step_deg"] > MARGINAL_JUMP_DEG
          or (axis_errors and max(axis_errors) > AXIS_TOLERANCE_DEG)):
        record["verdict"] = "marginal"
    else:
        record["verdict"] = "safe"
    return record, poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--x-range", nargs=2, type=float, default=[0.12, 0.34])
    parser.add_argument("--y-range", nargs=2, type=float, default=[-0.18, 0.18])
    parser.add_argument("--table-z", type=float, default=None)
    parser.add_argument("--urdf-limits", action="store_true",
                        help="judge against the URDF's declared joint ranges "
                             "instead of the servos' measured travel")
    parser.add_argument("--neighbours", action="store_true",
                        help="put blocks around the target as well")
    parser.add_argument("--out", type=Path, default=Path("outputs/sim"))
    args = parser.parse_args()

    arm = ToolKinematics()
    limits = arm.joint_limits_deg()
    if not args.urdf_limits:
        limits, found = measured_limits(limits)
        # Into the chain, not just into the scoring: see ToolKinematics.set_limits.
        arm.set_limits(limits)
        print("  joint limits: " + ("the servos' measured travel" if found
                                    else "the URDF's (no calibration found)"))
    # The table is where the arm is bolted to it, so it is where the underside
    # of the base sits - which the model knows exactly. Deriving it instead from
    # the real z_table measurement was circular: that number is under review in
    # its own right, and it moved the table 10 mm every time the TCP changed.
    if args.table_z is None:
        probe = SO101Sim(table_z=-1.0)
        probe.set_joints(PREFERRED, gripper_deg=35.0)
        table_z = probe.lowest_point(bodies=("base",))
    else:
        table_z = args.table_z
    grasp_z = table_z + BLOCK_SIZE_M / 2
    print(f"  {arm}")
    print(f"  table at {1000*table_z:+.1f} mm, grasping at TCP z="
          f"{1000*grasp_z:+.1f} mm")

    xs = np.linspace(*args.x_range, args.grid)
    ys = np.linspace(*args.y_range, args.grid)
    rows, everything = [], {}
    for x in xs:
        for y in ys:
            blocks = [("red", (float(x), float(y)))]
            if args.neighbours:
                blocks += [("blue", (float(x) + 0.035, float(y))),
                           ("green", (float(x), float(y) + 0.035))]
            sim = SO101Sim(table_z=table_z, blocks=blocks)
            record, poses = trajectory(arm, sim, float(x), float(y), grasp_z,
                                       limits, table_z=table_z)
            rows.append(record)
            everything[f"{x:.3f},{y:.3f}"] = poses
            print(f"    x={1000*x:>4.0f} y={1000*y:>+5.0f}  "
                  f"{record['verdict']:<12}"
                  f"align<={record['z_align_max_mm'] or 0:>5.0f}mm  "
                  f"step<={record.get('max_step_deg', 0):>5.1f}deg  "
                  f"{record.get('failed_at', '')}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (args.out / "pick_trajectory.csv").open("w", newline="",
                                                 encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "pick_trajectory.json").write_text(json.dumps({
        "table_z_m": table_z, "grasp_z_m": grasp_z,
        "grid": args.grid, "x_range": args.x_range, "y_range": args.y_range,
        "transit_m": TRANSIT_M, "neighbours": args.neighbours,
        "rows": rows, "waypoints": everything,
    }, indent=1), encoding="utf-8")
    print(f"\n  {args.out / 'pick_trajectory.csv'}")
    print(f"  {args.out / 'pick_trajectory.json'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
