"""The standard way this arm is asked to reach somewhere.

Settled in Phase S4' and S5.5, and worth stating in one place because every
part of it was arrived at by getting it wrong first:

  target the TCP      not the gripper frame with an offset added afterwards.
                      The frame sits within 8 mm of the wrist_roll axis, so the
                      solver can roll the wrist freely and still report that it
                      arrived, while the jaws swing 58 mm. With the TCP in the
                      chain, two solutions 130 degrees apart in roll put the
                      grasp point within 0.2 mm of each other.

  tool axis vertical  and vertical specifically. Sweeping the tilt, 0 degrees
                      solves 73% of the table and every angle from 5 to 40
                      solves about 10%. Multi-start does not recover them, so
                      that is the mechanism and not the solver.

  roll left free      pinning it as well is over-constrained - four joints
                      against five constraints - and it no longer matters,
                      because the blocks are cubes and the TCP is the target.

  multi-start         one seed is not evidence of anything. Restarting across
                      the roll's travel cut the median joint move between
                      neighbouring targets from 212 degrees to 28.

Whether to seed from the previous answer is the caller's business, and the two
uses are different: judging whether a point is reachable has to be independent
of what came before it, while moving between points wants exactly that history.
"""

from __future__ import annotations

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
          "wrist_roll")
#: The posture solves start from when there is no history to start from.
PREFERRED = {"shoulder_pan": 0.0, "shoulder_lift": -35.0, "elbow_flex": 65.0,
             "wrist_flex": 45.0, "wrist_roll": 0.0}
TILT_DEG = 0.0
TOLERANCE_MM = 1.0
ROLL_STARTS = 6


def tool_axis(pan_deg, tilt_deg=TILT_DEG):
    """Down, tilted forwards by `tilt_deg` in the plane the arm is swung to.

    The arm's plane, not the target's: with the wrist hanging 61 mm off the
    shoulder's axis the two differ, and asking for a tilt in the wrong one makes
    every angle but zero look unreachable.
    """
    pan, tilt = np.radians(pan_deg), np.radians(tilt_deg)
    return np.array([np.sin(tilt) * np.cos(pan),
                     np.sin(tilt) * np.sin(pan),
                     -np.cos(tilt)])


def grasp_difference(first, second):
    """How far apart two solutions are, taking the worst joint."""
    return max(abs(first[name] - second[name]) for name in JOINTS)


def seeds(reference, starts=ROLL_STARTS):
    """Where to restart from: the reference, then rolled around its travel.

    Roll is the free degree of freedom once the position and the tool axis are
    pinned, so it is the direction worth restarting in. The reference comes
    first, so that when several seeds work the nearest is already to hand.
    """
    yield dict(reference)
    for roll in np.linspace(-150, 150, starts):
        seed = dict(reference)
        seed["wrist_roll"] = float(roll)
        yield seed
    folded = dict(reference)
    folded["shoulder_lift"], folded["elbow_flex"] = -70.0, 90.0
    yield folded


def solve(arm, target, reference=None, tilt_deg=TILT_DEG,
          tolerance_mm=TOLERANCE_MM, multi_start=True, good_enough_deg=None,
          cost=None):
    """Put the TCP at `target`, tool pointing down. Returns joints or None.

    `reference` seeds the search and breaks ties: among the solutions that work,
    the one nearest it is returned. Pass the arm's current pose to get a move it
    can make smoothly; leave it out to ask the question afresh.

    `multi_start=False` stops at the first seed that works, which is the right
    setting for asking whether a point is reachable at all - it still restarts
    after a failure, it just does not keep looking once it has an answer.

    `good_enough_deg` stops early once a solution is found within that far of
    the reference. Walking a trajectory, the first seed usually lands near the
    previous pose, and searching the other seven to confirm it costs as much as
    the whole rest of the step.

    `cost` replaces "nearest the reference" with something the caller cares
    about more - joint-limit margin, say, or whether the arm is in the table.
    """
    reference = PREFERRED if reference is None else reference
    target = np.asarray(target, float)
    cost = cost or (lambda answer: grasp_difference(answer, reference))
    best, best_cost = None, np.inf
    for seed in seeds(reference):
        # The tool axis has to lie in the plane the arm ends up swung to, and
        # that is not known until it has been placed. So: position first, read
        # the pan off it, then solve again with the axis that pan allows.
        rough = arm.inverse(target, seed_deg=seed, tolerance_mm=20.0,
                            orientation=None)
        pan = seed["shoulder_pan"] if rough is None else rough["shoulder_pan"]
        answer = arm.inverse(target, seed_deg=rough or seed,
                             tolerance_mm=tolerance_mm,
                             orientation=tool_axis(pan, tilt_deg))
        if answer is not None:
            scored = cost(answer)
            if scored < best_cost:
                best, best_cost = answer, scored
            if not multi_start:
                break
            if good_enough_deg is not None \
                    and grasp_difference(answer, reference) <= good_enough_deg:
                break
    return best
