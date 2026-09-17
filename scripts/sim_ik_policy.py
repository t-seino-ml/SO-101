"""Phase S4': settle the standard inverse-kinematics policy.

Phase S4 tested solver settings against reachable poses and found several that
hit the TCP. Phase S5 then showed two of the assumptions behind the favourite
were wrong:

- pinning the tool axis does not pin the roll. Rolling turns the tool about its
  own z, so the same axis is reachable at many rolls; solutions came out 80 to
  163 degrees from the seed's roll.
- pinning the axis *and* freezing the roll is over-constrained. Four joints
  cannot satisfy three position constraints plus two of direction, which is why
  every tilt but zero went unreachable almost everywhere.

What changed the conclusion is that roll no longer matters much. With the TCP in
the chain, a solution 130 degrees rolled from another still puts the grasp point
in the same place to a fifth of a millimetre - and the blocks are cubes, so the
jaws do not care which way round they close. Roll is now a question about the
wrist camera's view and about how far the wrist swings on the way, not about
whether the grasp lands.

So this evaluates the candidate policy - position, tool axis pinned, roll free -
on the three things that do matter:

  reach         how much of the table it can solve for at all
  repeatability the same target from two seeds, measured at the TCP
  continuity    how far the joints move between neighbouring targets, which is
                what a trajectory has to actually execute

    uv run scripts/sim_ik_policy.py
    uv run scripts/sim_ik_policy.py --grid 12
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from so101.policy.tcp import ToolKinematics

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
PREFERRED = {"shoulder_pan": 0.0, "shoulder_lift": -35.0, "elbow_flex": 65.0,
             "wrist_flex": 45.0, "wrist_roll": 0.0}
TOLERANCE_MM = 1.0
GRASP_Z = 0.0087              # TCP height for a 20 mm block on the table


def tool_axis(pan_deg, tilt_deg):
    """Down, tilted forwards by `tilt_deg` in the plane the arm is swung to."""
    pan, tilt = np.radians(pan_deg), np.radians(tilt_deg)
    return np.array([np.sin(tilt) * np.cos(pan),
                     np.sin(tilt) * np.sin(pan),
                     -np.cos(tilt)])


def solve(arm, target, tilt_deg, seed, policy):
    """One solve under a named policy. Returns the joint dict or None."""
    if policy == "position only":
        return arm.inverse(target, seed_deg=seed, tolerance_mm=TOLERANCE_MM,
                           orientation=None)
    if policy == "roll frozen, position only":
        return arm.inverse(target, seed_deg=seed, tolerance_mm=TOLERANCE_MM,
                           orientation=None, frozen=("wrist_roll",))
    # The candidate: the axis has to lie in the plane the arm swings to, and
    # that plane is not known until the arm has been placed, so it is solved
    # for position first and the axis built from the pan that comes back.
    rough = arm.inverse(target, seed_deg=seed, tolerance_mm=20.0,
                        orientation=None)
    pan = seed["shoulder_pan"] if rough is None else rough["shoulder_pan"]
    return arm.inverse(target, seed_deg=rough or seed, tolerance_mm=TOLERANCE_MM,
                       orientation=tool_axis(pan, tilt_deg))


POLICIES = ("position only", "roll frozen, position only",
            "tool axis pinned, roll free")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--x-range", nargs=2, type=float, default=[0.12, 0.34])
    parser.add_argument("--y-range", nargs=2, type=float, default=[-0.18, 0.18])
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/sim/ik_policy.json"))
    args = parser.parse_args()

    arm = ToolKinematics()
    print(f"  {arm}")
    xs = np.linspace(*args.x_range, args.grid)
    ys = np.linspace(*args.y_range, args.grid)
    points = [(float(x), float(y)) for x in xs for y in ys]
    print(f"  {len(points)} points over x {args.x_range[0]:.2f}.."
          f"{args.x_range[1]:.2f}, y {args.y_range[0]:+.2f}.."
          f"{args.y_range[1]:+.2f} m at z={1000*GRASP_Z:+.1f} mm\n")

    # -- which tilt reaches the most of the table? -------------------------
    print(f"  the candidate policy, by how far the tool is tilted\n")
    print(f"  {'tilt':>7}{'solved':>9}{'median err':>13}{'p95 err':>10}"
          f"{'axis err':>11}")
    coverage = {}
    for tilt in range(0, 41, 5):
        solved, errors, axes = 0, [], []
        for x, y in points:
            target = np.array([x, y, GRASP_Z])
            answer = solve(arm, target, float(tilt), PREFERRED,
                           "tool axis pinned, roll free")
            if answer is None:
                continue
            solved += 1
            reached, rotation = arm.pose(answer)
            errors.append(1000 * float(np.linalg.norm(reached - target)))
            wanted = tool_axis(answer["shoulder_pan"], float(tilt))
            axes.append(float(np.degrees(np.arccos(np.clip(
                float(rotation[:, 2] @ wanted), -1, 1)))))
        coverage[tilt] = solved
        if solved:
            print(f"  {tilt:>4} deg{100*solved/len(points):8.0f}%"
                  f"{np.median(errors):12.3f}{np.percentile(errors, 95):10.3f}"
                  f"{np.median(axes):10.3f}")
        else:
            print(f"  {tilt:>4} deg{0:8.0f}%")
    tilt = max(coverage, key=lambda t: coverage[t])
    print(f"\n  taking {tilt} deg: it solves the most of the table\n")

    # -- the three policies, side by side ----------------------------------
    print(f"  {'policy':<30}{'solved':>8}{'TCP err':>10}{'repeat':>9}"
          f"{'tool tilt':>16}{'neighbour step':>16}")
    print(f"  {'':<30}{'':>8}{'median mm':>10}{'mm':>9}"
          f"{'median / spread':>16}{'deg, worst':>16}")
    summary = {}
    for policy in POLICIES:
        errors, repeat, rolls, tilts = [], [], [], []
        previous, steps = None, []
        for x, y in points:
            target = np.array([x, y, GRASP_Z])
            first = solve(arm, target, float(tilt), PREFERRED, policy)
            if first is None:
                previous = None
                continue
            reached, rotation = arm.pose(first)
            errors.append(1000 * float(np.linalg.norm(reached - target)))
            # How far off vertical the tool ends up. A policy that leaves this
            # free has the jaws arriving at a different angle at every point on
            # the table, which is a problem for coming down onto a block among
            # other blocks even when the TCP itself is exact.
            tilts.append(float(np.degrees(np.arccos(
                np.clip(-rotation[2, 2], -1, 1)))))

            # Repeatability: a different seed, same target. What matters is
            # where the TCP lands, not what the joints did to get there.
            scattered = {name: PREFERRED[name] + delta for name, delta
                         in zip(JOINTS, (25.0, 15.0, -20.0, -15.0, 90.0))}
            second = solve(arm, target, float(tilt), scattered, policy)
            if second is not None:
                repeat.append(1000 * float(np.linalg.norm(
                    arm.forward(first) - arm.forward(second))))
                rolls.append(abs(first["wrist_roll"] - second["wrist_roll"]))

            # Continuity: seeded from the neighbour's answer, how far do the
            # joints have to travel? This is what a trajectory pays.
            if previous is not None:
                nearby = solve(arm, target, float(tilt), previous, policy)
                if nearby is not None:
                    steps.append(max(abs(nearby[n] - previous[n])
                                     for n in JOINTS))
            previous = first

        if not errors:
            print(f"  {policy:<30}{'none':>8}")
            continue
        summary[policy] = {
            "solved_pct": 100 * len(errors) / len(points),
            "tcp_error_median_mm": float(np.median(errors)),
            "repeatability_median_mm": float(np.median(repeat)) if repeat else None,
            "roll_spread_median_deg": float(np.median(rolls)) if rolls else None,
            "tool_tilt_median_deg": float(np.median(tilts)),
            "tool_tilt_spread_deg": float(np.ptp(tilts)),
            "neighbour_step_worst_deg": float(np.max(steps)) if steps else None,
            "neighbour_step_median_deg": float(np.median(steps)) if steps else None,
        }
        shown = f"{np.median(tilts):.0f} / {np.ptp(tilts):.0f} deg"
        print(f"  {policy:<30}{100*len(errors)/len(points):7.0f}%"
              f"{np.median(errors):10.3f}"
              f"{np.median(repeat) if repeat else float('nan'):9.2f}"
              f"{shown:>16}"
              f"{np.max(steps) if steps else float('nan'):16.0f}")

    print(f"\n  repeat  = same target from a scattered seed, measured at the TCP")
    print(f"  tool tilt = how far off vertical the jaws come in, across the")
    print(f"              table. Its spread is what a policy leaves uncontrolled.")
    print(f"  neighbour step = worst joint move between adjacent grid points when")
    print(f"                   each is seeded from the last, which is what a")
    print(f"                   trajectory has to execute")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"tilt_deg": tilt, "grasp_z_m": GRASP_Z,
                                    "coverage_by_tilt": coverage,
                                    "policies": summary}, indent=1),
                        encoding="utf-8")
    print(f"\n  {args.out}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
