"""Phase S5.5: is the arm refusing, or is the solver giving up?

Two results from S4' and S5 were taken at face value and should not have been:

  the tilt cliff     a vertical tool axis solved 73% of the table; 5 degrees of
                     tilt solved 12%. A cliff that steep between two nearly
                     identical requests is much more like a solver falling into
                     a local minimum than like a mechanism.

  the 304 degree     adjacent grid points, each seeded from the last, wanted
  neighbour jump     joint moves of up to 304 degrees. Real if the arm has to
                     flip configuration; not real if a second attempt from a
                     different start would have found something nearby.

Three things separate the two explanations:

  multi-start        try several seeds and keep the best answer, rather than
                     concluding from one failure that there is nothing there.
  continuation       seed each target from the previous target's answer, so the
                     solver is asked for a nearby solution rather than any
                     solution.
  wrapping           the jaws are a symmetric pair, so a wrist rolled by 180
                     degrees grips exactly the same way. Two solutions 200
                     degrees apart in wrist_roll are 20 degrees apart as grasps,
                     and measuring the raw difference overstates the problem.

    uv run scripts/sim_ik_multistart.py
    uv run scripts/sim_ik_multistart.py --grid 12
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
GRASP_Z = 0.0087


def tool_axis(pan_deg, tilt_deg):
    pan, tilt = np.radians(pan_deg), np.radians(tilt_deg)
    return np.array([np.sin(tilt) * np.cos(pan),
                     np.sin(tilt) * np.sin(pan),
                     -np.cos(tilt)])


def grasp_difference(a, b):
    """How different two solutions are, as grasps rather than as numbers.

    Every joint counts its plain difference except the wrist roll, where 180
    degrees puts the same pair of jaws back across the same block. Rolling a
    symmetric gripper half a turn is not a different grasp, and counting it as
    one is what turned a manageable wrist move into a 304 degree figure.
    """
    worst = 0.0
    for name in JOINTS:
        delta = abs(a[name] - b[name])
        if name == "wrist_roll":
            delta = min(delta % 180.0, 180.0 - delta % 180.0)
        worst = max(worst, delta)
    return worst


def seeds_for(reference, spread=6):
    """A reference pose first, then the same pose rolled around its travel.

    Roll is the free degree of freedom once position and the tool axis are
    pinned, so that is the direction worth restarting in. The reference comes
    first so that when several seeds work, the nearest one is already at hand.
    """
    yield dict(reference)
    for roll in np.linspace(-150, 150, spread):
        seed = dict(reference)
        seed["wrist_roll"] = float(roll)
        yield seed
    # And one with the elbow the other way, in case the arm has to fold.
    other = dict(reference)
    other["shoulder_lift"], other["elbow_flex"] = -70.0, 90.0
    yield other


def solve(arm, target, tilt_deg, reference, multi_start):
    """Position plus tool axis, roll free. Returns the best solution or None."""
    best, best_cost = None, np.inf
    for index, seed in enumerate(seeds_for(reference)):
        rough = arm.inverse(target, seed_deg=seed, tolerance_mm=20.0,
                            orientation=None)
        pan = seed["shoulder_pan"] if rough is None else rough["shoulder_pan"]
        answer = arm.inverse(target, seed_deg=rough or seed,
                             tolerance_mm=TOLERANCE_MM,
                             orientation=tool_axis(pan, tilt_deg))
        if answer is not None:
            # Among solutions that work, the one closest to where the arm
            # already is. That is what makes a path rather than a sequence of
            # unrelated poses.
            cost = grasp_difference(answer, reference)
            if cost < best_cost:
                best, best_cost = answer, cost
            if not multi_start:
                break
        if not multi_start and index == 0:
            break
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=9)
    parser.add_argument("--x-range", nargs=2, type=float, default=[0.12, 0.34])
    parser.add_argument("--y-range", nargs=2, type=float, default=[-0.18, 0.18])
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/sim/ik_multistart.json"))
    args = parser.parse_args()

    arm = ToolKinematics()
    xs = np.linspace(*args.x_range, args.grid)
    ys = np.linspace(*args.y_range, args.grid)
    # Snake through the grid, so consecutive points are always adjacent and
    # continuation is being given a fair chance.
    points = [(float(x), float(y))
              for i, x in enumerate(xs)
              for y in (ys if i % 2 == 0 else ys[::-1])]
    print(f"  {arm}")
    print(f"  {len(points)} points, walked in a snake so neighbours really "
          f"are neighbours\n")

    # -- does multi-start recover the tilted approaches? -------------------
    print(f"  reach against tool tilt\n")
    print(f"  {'tilt':>7}{'one seed':>12}{'multi-start':>14}")
    coverage = {}
    for tilt in (0, 5, 10, 15, 20, 25, 30, 40):
        counts = []
        for multi in (False, True):
            solved = sum(solve(arm, np.array([x, y, GRASP_Z]), float(tilt),
                               PREFERRED, multi) is not None
                         for x, y in points)
            counts.append(100 * solved / len(points))
        coverage[tilt] = counts
        print(f"  {tilt:>4} deg{counts[0]:11.0f}%{counts[1]:13.0f}%")

    # -- and does continuation tame the neighbour jump? --------------------
    print(f"\n  joint movement between neighbouring points, at 0 deg tilt\n")
    print(f"  {'strategy':<40}{'solved':>9}{'median':>9}{'worst':>9}")
    results = {}
    for label, multi, continuation, wrap in (
            ("one seed, always from PREFERRED", False, False, False),
            ("one seed, continuation", False, True, False),
            ("multi-start, continuation", True, True, False),
            ("multi-start, continuation, wrapped", True, True, True)):
        previous, steps, solved = None, [], 0
        for x, y in points:
            reference = previous if (continuation and previous) else PREFERRED
            answer = solve(arm, np.array([x, y, GRASP_Z]), 0.0, reference, multi)
            if answer is None:
                previous = None
                continue
            solved += 1
            if previous is not None:
                steps.append(grasp_difference(answer, previous) if wrap else
                             max(abs(answer[n] - previous[n]) for n in JOINTS))
            previous = answer
        if not steps:
            print(f"  {label:<40}{'none':>9}")
            continue
        results[label] = {"solved_pct": 100 * solved / len(points),
                          "median_deg": float(np.median(steps)),
                          "worst_deg": float(np.max(steps))}
        print(f"  {label:<40}{100*solved/len(points):8.0f}%"
              f"{np.median(steps):9.0f}{np.max(steps):9.0f}")

    print(f"\n  solved  = how much of the table the strategy reached")
    print(f"  median/worst = degrees the busiest joint has to move between")
    print(f"                 one point and the next")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"coverage_by_tilt": coverage,
                                    "continuity": results}, indent=1),
                        encoding="utf-8")
    print(f"\n  {args.out}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
