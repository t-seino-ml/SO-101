"""Planning a whole pick, and saying where on the table one can be planned.

The same planner runs in simulation and on the arm. That is the point of it
living here rather than in a script: a workspace measured against one planner
and executed by another measures nothing.

A pick is five poses, and the constrained one is not the grasp - it is the
alignment, where the tool has to be turned vertical. How high that can happen
varies from 9 mm to 100 mm above the grasp depending where on the table you are,
and at the very top of that range the wrist is against its stop by definition:
z_align_max is the height at which vertical *just* solves. Planning there is
planning on a cliff edge, so the height is searched for instead, scored on what
actually makes a move safe - margin at the stops, how far the joints travel
either side of it, whether anything is in contact, and how much room is left
over the table and the neighbouring blocks.

Joint limits carry a safety margin. The servos' measured travel is what the arm
can physically do; a planner that uses all of it leaves nothing for the
difference between a model and a machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ik_policy import (
    JOINTS,
    PREFERRED,
    TILT_DEG,
    grasp_difference,
    seeds,
    solve,
    tool_axis,
)

MASK_PATH = Path("data/pick_workspace.json")
#: How far inside the measured travel the planner will go. The measured range is
#: what the joint can reach, not what it should be asked for: a servo on its stop
#: has no authority left, and the model is not the machine to better than a
#: degree or two anyway.
SAFETY_MARGIN_DEG = 5.0
#: Below this *operational* margin, a trajectory that completed is still only
#: marginal. It sits on top of the safety margin rather than replacing it: a
#: pose at the comfort threshold has COMFORTABLE + SAFETY degrees of real travel
#: left. That is two reserves stacked, deliberately, and both are named in every
#: record so the total is never a surprise.
COMFORTABLE_MARGIN_DEG = 8.0
JUMP_DEG = 90.0
MARGINAL_JUMP_DEG = 45.0
GOOD_ENOUGH_DEG = 20.0
DESCENT_STEPS = 3
LIFT_M = 0.06
TRANSIT_M = 0.12
TRANSIT_CHOICES = (0.12, 0.10, 0.08, 0.06)
#: Alignment heights to try, as fractions of where vertical stops solving.
ALIGN_FRACTIONS = (0.45, 0.65, 0.85)
MIN_ALIGN_M = 0.030           # below this there is no room to turn the tool
BLOCK_SIZE_M = 0.020


def with_safety_margin(limits, margin_deg=SAFETY_MARGIN_DEG):
    """Joint ranges pulled in from the measured travel."""
    return {name: (low + margin_deg, high - margin_deg)
            for name, (low, high) in limits.items()}


def measured_limits(urdf_limits, path=None):
    """Joint ranges from the servos' calibration rather than the URDF's caution.

    They differ in both directions - wrist_flex really has 212 degrees against
    the URDF's 190, wrist_roll 266 against 320 - and the difference decides
    whether a vertical tool axis is reachable, so it is not a detail.

    Only the span is taken from the calibration; the centre comes from the URDF,
    which is what fixes where zero is.
    """
    import os

    if path is None:
        path = (Path(os.environ.get("USERPROFILE", "~")).expanduser()
                / ".cache/huggingface/lerobot/calibration/robots"
                / "so_follower/follower.json")
    path = Path(path)
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


class PickPlanner:
    """Plans transit, align, descend, grasp and lift for a spot on the table.

    `sim` is optional. With one, collisions and table clearance are checked
    against real geometry; without, the plan is kinematic only - which is what
    the real arm gets, where there is no simulator to ask.
    """

    def __init__(self, arm, limits, sim=None, table_z=0.0,
                 safety_margin_deg=SAFETY_MARGIN_DEG, tilt_deg=TILT_DEG,
                 block_size_m=BLOCK_SIZE_M):
        self.arm = arm
        #: What the arm can physically do, from the servos' calibration.
        self.measured = dict(limits)
        self.safety_margin_deg = safety_margin_deg
        #: What the planner will use: the above, pulled in once. Note that the
        #: inverse kinematics is given `measured`, not this - so a solution may
        #: come back outside the operational range and be rejected here, which
        #: is deliberate. Constraining the solver as well would apply the margin
        #: twice over.
        self.limits = with_safety_margin(limits, safety_margin_deg)
        self.sim = sim
        self.table_z = table_z
        self.tilt_deg = tilt_deg
        self.block_size = block_size_m
        self.grasp_z = table_z + block_size_m / 2

    # -- pieces -----------------------------------------------------------

    def margin(self, pose):
        """How far the tightest joint is from the limits the planner may use.

        The *operational* limits - the measured travel with the safety margin
        already taken off. So a margin of 0 here means the joint is exactly on
        the operational limit and still has the safety margin of real travel
        left, not that it is on the stop.
        """
        return min(min(pose[name] - self.limits[name][0],
                       self.limits[name][1] - pose[name]) for name in JOINTS)

    def physical_margin(self, pose):
        """How far the tightest joint is from the stop the arm actually has."""
        return min(min(pose[name] - self.measured[name][0],
                       self.measured[name][1] - pose[name]) for name in JOINTS)

    def margins(self, pose):
        """Both, so a record never leaves which basis it is on to the reader.

        The two differ by exactly the safety margin, by construction - and that
        is checked rather than assumed, because the inverse kinematics is given
        the physical limits and only the planner applies the safety one. If the
        difference is ever anything else, the margin has been applied twice
        somewhere.
        """
        operational = self.margin(pose)
        physical = self.physical_margin(pose)
        if abs((physical - operational) - self.safety_margin_deg) > 1e-6:
            raise AssertionError(
                f"the safety margin is applied {physical - operational:.2f} deg "
                f"worth, not {self.safety_margin_deg:.2f} - it is being applied "
                "more than once")
        return operational, physical

    def clashes(self, pose, ignore_blocks=True):
        """(contacts, lowest moving point) for a pose, or (0, None) without a sim."""
        if self.sim is None:
            return 0, None
        from ..sim.model import BLOCK_COLOURS  # noqa: F401  (import guard)

        ignore = ()
        if ignore_blocks:
            ignore = tuple(f"block_{i}" for i in range(16)) + \
                tuple(f"block_{i}_geom" for i in range(16))
        self.sim.set_joints(pose, gripper_deg=35.0)
        contacts = [c for c in self.sim.collisions(ignore=ignore)
                    if "table" not in (c[0], c[1])]
        lowest = self.sim.lowest_point(bodies=(
            "shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
            "moving_jaw_so101_v1"))
        if lowest < self.table_z - 0.001:
            contacts = contacts + [("table", "arm", 1000 * (self.table_z - lowest))]
        return len(contacts), lowest

    def vertical_at(self, x, y, height, reference=None, multi_start=True):
        """A vertical-tool solution `height` above the grasp, or None."""
        return solve(self.arm, np.array([x, y, self.grasp_z + height]),
                     reference=reference or PREFERRED, tilt_deg=self.tilt_deg,
                     multi_start=multi_start)

    def free_at(self, x, y, height, reference=None):
        """Any solution `height` above the grasp, tool unconstrained."""
        for seed in seeds(reference or PREFERRED):
            answer = self.arm.inverse(
                np.array([x, y, self.grasp_z + height]), seed_deg=seed,
                tolerance_mm=1.0, orientation=None)
            if answer is not None:
                return answer
        return None

    def align_ceiling(self, x, y, highest=0.10, resolution=0.005):
        """Where vertical stops solving. A boundary, not a place to work.

        Reported because it describes the column, and never planned at: by
        construction the wrist is on its stop there.
        """
        if self.vertical_at(x, y, 0.0, multi_start=False) is None:
            return None
        low, high = 0.0, highest
        if self.vertical_at(x, y, high, multi_start=False) is not None:
            return high
        while high - low > resolution:
            middle = (low + high) / 2
            if self.vertical_at(x, y, middle, multi_start=False) is not None:
                low = middle
            else:
                high = middle
        return low

    # -- choosing where to align ------------------------------------------

    def choose_align(self, x, y, ceiling, transit=None, fractions=ALIGN_FRACTIONS):
        """The best height to turn the tool vertical at, and why.

        Scored on what makes the move safe rather than on how high it is: room
        at the joint stops, how far the arm has to travel into and out of the
        pose, whether anything is in contact, and how much clearance is left
        over the table. Returns (height, detail) or (None, reason).
        """
        # The travel height is the same for every candidate, so it is settled
        # once here rather than re-solved inside the loop - it was most of the
        # cost of choosing.
        if transit is None:
            transit = self._transit_height(x, y, ceiling)
        candidates = []
        for fraction in fractions:
            height = ceiling * fraction
            if height < MIN_ALIGN_M:
                continue
            # Screening, so a single restart is enough: a candidate that needs
            # every seed to be found is not one worth planning around anyway,
            # and the winner is re-solved properly when the plan is built.
            pose = self.vertical_at(x, y, height, multi_start=False)
            if pose is None:
                continue
            contacts, lowest = self.clashes(pose)
            if contacts:
                continue
            margin = self.margin(pose)
            if margin < 0:
                continue
            # What it costs to get into this pose and out of it again: the arm
            # comes from the travel height and leaves downwards.
            above = self.free_at(x, y, transit, reference=pose)
            below = self.vertical_at(x, y, height * (DESCENT_STEPS - 1)
                                     / DESCENT_STEPS, reference=pose,
                                     multi_start=False)
            if below is None:
                continue
            into = grasp_difference(pose, above) if above else 0.0
            out_of = grasp_difference(below, pose)
            candidates.append({
                "height_m": height, "fraction": fraction, "pose": pose,
                "margin_deg": margin, "into_deg": into, "out_of_deg": out_of,
                "lowest_m": lowest,
                # Margin is what the planner is short of, so it dominates; the
                # travel either side is a tie-break between poses that are all
                # comfortable.
                "score": -margin + 0.25 * max(into, out_of),
            })
        if not candidates:
            return None, "no height between a third and most of the ceiling works"
        best = min(candidates, key=lambda c: c["score"])
        best["considered"] = len(candidates)
        return best["height_m"], best

    def _transit_height(self, x, y, ceiling):
        for height in TRANSIT_CHOICES:
            if height < ceiling:
                continue
            if self.free_at(x, y, height) is not None:
                return height
        return ceiling

    # -- the plan ---------------------------------------------------------

    def plan(self, x, y, align_at_ceiling=False):
        """The whole pick at (x, y). Returns a record; `waypoints` may be short."""
        record = {"target_x": round(float(x), 4), "target_y": round(float(y), 4),
                  "grasp_z": round(self.grasp_z, 4)}
        ceiling = self.align_ceiling(x, y)
        record["z_align_max_mm"] = (None if ceiling is None
                                    else round(1000 * ceiling, 1))
        if ceiling is None:
            record.update({"verdict": "unreachable", "failed_at": "grasp height"})
            return record, []
        if ceiling < MIN_ALIGN_M:
            record.update({"verdict": "unreachable", "failed_at": "align",
                           "why": f"vertical only holds to {1000*ceiling:.0f} mm"})
            return record, []

        transit = self._transit_height(x, y, ceiling)
        record["transit_mm"] = round(1000 * transit, 1)

        if align_at_ceiling:
            align_h, detail = ceiling, {"fraction": 1.0, "considered": 0}
        else:
            align_h, detail = self.choose_align(x, y, ceiling, transit=transit)
            if align_h is None:
                record.update({"verdict": "unreachable", "failed_at": "align",
                               "why": detail})
                return record, []
        record["z_align_opt_mm"] = round(1000 * align_h, 1)
        record["align_fraction"] = round(detail.get("fraction", 0.0), 3)
        record["align_candidates"] = detail.get("considered", 0)

        # The alignment pose is solved first even though it runs second: it is
        # the constrained one, and letting the travel pose be chosen freely and
        # then demanding vertical produced 275 degree reconfigurations.
        anchor = detail.get("pose") or self.vertical_at(x, y, align_h)
        if anchor is None:
            record.update({"verdict": "unreachable", "failed_at": "align"})
            return record, []

        plan = [("transit", transit, False), ("align", align_h, True)]
        for step in range(1, DESCENT_STEPS + 1):
            plan.append((f"descend {step}",
                         align_h * (DESCENT_STEPS - step) / DESCENT_STEPS, True))
        plan.append(("lift", min(LIFT_M, align_h), True))

        poses, steps, previous = [], [], None
        for name, height, vertical in plan:
            reference = previous or anchor
            if vertical:
                answer = solve(
                    self.arm,
                    np.array([x, y, self.grasp_z + height]),
                    reference=reference, tilt_deg=self.tilt_deg,
                    cost=(lambda candidate, ref=reference:
                          self._step_cost(candidate, ref)),
                    good_enough_deg=GOOD_ENOUGH_DEG)
            else:
                answer = self._best_free(x, y, height, reference)
            if answer is None:
                record.update({"verdict": "unreachable", "failed_at": name})
                return record, poses

            reached, rotation = self.arm.pose(answer)
            target = np.array([x, y, self.grasp_z + height])
            error = 1000 * float(np.linalg.norm(reached - target))
            axis_error = (float(np.degrees(np.arccos(np.clip(float(
                rotation[:, 2] @ tool_axis(answer["shoulder_pan"],
                                           self.tilt_deg)), -1, 1))))
                if vertical else None)
            contacts, lowest = self.clashes(answer)
            operational, physical = self.margins(answer)
            if previous is not None:
                steps.append(grasp_difference(answer, previous))
            poses.append({
                "waypoint": name, "z_mm": round(1000 * target[2], 1),
                "position_error_mm": round(error, 3),
                "axis_error_deg": None if axis_error is None
                else round(axis_error, 3),
                "operational_margin_deg": round(operational, 2),
                "physical_margin_deg": round(physical, 2),
                "collisions": contacts,
                "lowest_point_mm": None if lowest is None
                else round(1000 * lowest, 1),
                "step_deg": round(steps[-1], 1) if steps else 0.0,
                **{j: round(answer[j], 2) for j in JOINTS}})
            if error > 1.0 or contacts:
                record.update({"verdict": "unreachable", "failed_at": name,
                               "why": "collision" if contacts
                               else f"missed by {error:.1f} mm"})
                return record, poses
            previous = answer

        margins = [p["operational_margin_deg"] for p in poses]
        physical = [p["physical_margin_deg"] for p in poses]
        axes = [p["axis_error_deg"] for p in poses
                if p["axis_error_deg"] is not None]
        record.update({
            "waypoints": len(poses),
            "max_step_deg": round(max(steps), 1) if steps else 0.0,
            "total_motion_deg": round(sum(steps), 1),
            # Both bases, named. The verdict below is on the operational one.
            "min_operational_margin_deg": round(min(margins), 2),
            "min_physical_margin_deg": round(min(physical), 2),
            "safety_margin_deg": self.safety_margin_deg,
            "max_axis_error_deg": round(max(axes), 3) if axes else None,
            "collisions": 0,
        })
        if record["max_step_deg"] > JUMP_DEG                 or record["min_operational_margin_deg"] < 0:
            record["verdict"] = "unreachable"
            record["failed_at"] = ("configuration jump"
                                   if record["max_step_deg"] > JUMP_DEG
                                   else "joint limit")
        elif (record["min_operational_margin_deg"] < COMFORTABLE_MARGIN_DEG
              or record["max_step_deg"] > MARGINAL_JUMP_DEG):
            record["verdict"] = "marginal"
        else:
            record["verdict"] = "safe"
        return record, poses

    def _step_cost(self, candidate, reference):
        motion = grasp_difference(candidate, reference)
        contacts, _ = self.clashes(candidate)
        return (motion
                + 10.0 * max(0.0, COMFORTABLE_MARGIN_DEG - self.margin(candidate))
                + 1000.0 * contacts)

    def _best_free(self, x, y, height, reference):
        best, best_cost = None, np.inf
        for seed in seeds(reference):
            answer = self.arm.inverse(
                np.array([x, y, self.grasp_z + height]), seed_deg=seed,
                tolerance_mm=1.0, orientation=None)
            if answer is None:
                continue
            cost = self._step_cost(answer, reference)
            if cost < best_cost:
                best, best_cost = answer, cost
        return best


class WorkspaceMask:
    """Which spots on the table a pick can be planned for.

    Built from a grid of planned trajectories and asked about arbitrary points,
    because blocks do not land on grid nodes. A point takes the worst verdict of
    the grid cells around it: the map is coarse, and rounding a boundary in the
    optimistic direction is how an arm ends up reaching somewhere it cannot.
    """

    ORDER = ("safe", "marginal", "unreachable")

    def __init__(self, xs, ys, verdicts, extras=None):
        self.xs = np.asarray(xs, float)
        self.ys = np.asarray(ys, float)
        self.verdicts = verdicts          # {(x index, y index): verdict}
        self.extras = extras or {}

    @classmethod
    def from_rows(cls, rows):
        xs = sorted({r["target_x"] for r in rows})
        ys = sorted({r["target_y"] for r in rows})
        index = {(xs.index(r["target_x"]), ys.index(r["target_y"])):
                 r["verdict"] for r in rows}
        extras = {(xs.index(r["target_x"]), ys.index(r["target_y"])): {
            "z_align_max_mm": r.get("z_align_max_mm"),
            "z_align_opt_mm": r.get("z_align_opt_mm")} for r in rows}
        return cls(xs, ys, index, extras)

    def disputed(self, x, y):
        """Do the grid cells around this point disagree?

        Where they do, the map is not answering the question - it is averaging
        two different answers over a cell wider than the thing being picked. A
        caller should re-plan rather than trust either. Only where all four
        corners agree is a lookup really a lookup.
        """
        corners = self._corners(x, y)
        return len(set(corners)) > 1

    def _corners(self, x, y):
        i = int(np.clip(np.searchsorted(self.xs, x) - 1, 0, len(self.xs) - 2))
        j = int(np.clip(np.searchsorted(self.ys, y) - 1, 0, len(self.ys) - 2))
        return [self.verdicts.get((a, b), "unreachable")
                for a in (i, i + 1) for b in (j, j + 1)]

    def classify(self, x, y, planner=None):
        """SAFE, MARGINAL or UNREACHABLE for any point on the table.

        With a `planner`, a point whose surrounding cells disagree is planned
        for properly instead of being written off. The map is a lookup for the
        easy majority; the boundary is where it is worth the two seconds.
        """
        if planner is not None and self.in_range(x, y) and self.disputed(x, y):
            record, _ = planner.plan(x, y)
            return record["verdict"]
        return self._lookup(x, y)

    def in_range(self, x, y):
        return (self.xs[0] <= x <= self.xs[-1]
                and self.ys[0] <= y <= self.ys[-1])

    def _lookup(self, x, y):
        if not self.in_range(x, y):
            return "unreachable"
        return max(self._corners(x, y), key=self.ORDER.index)

    def align_height(self, x, y):
        """The planned alignment height near a point, in metres, or None."""
        i = int(np.clip(np.searchsorted(self.xs, x) - 1, 0, len(self.xs) - 2))
        j = int(np.clip(np.searchsorted(self.ys, y) - 1, 0, len(self.ys) - 2))
        heights = [self.extras.get((a, b), {}).get("z_align_opt_mm")
                   for a in (i, i + 1) for b in (j, j + 1)]
        heights = [h for h in heights if h]
        return min(heights) / 1000 if heights else None

    def save(self, path=MASK_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "xs": self.xs.tolist(), "ys": self.ys.tolist(),
            "verdicts": {f"{a},{b}": v for (a, b), v in self.verdicts.items()},
            "extras": {f"{a},{b}": v for (a, b), v in self.extras.items()},
        }, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path=MASK_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        verdicts = {tuple(int(v) for v in key.split(",")): value
                    for key, value in data["verdicts"].items()}
        extras = {tuple(int(v) for v in key.split(",")): value
                  for key, value in data.get("extras", {}).items()}
        return cls(data["xs"], data["ys"], verdicts, extras)

    def __str__(self):
        counts = {v: sum(1 for x in self.verdicts.values() if x == v)
                  for v in self.ORDER}
        return (f"WorkspaceMask({len(self.xs)}x{len(self.ys)}, "
                + ", ".join(f"{k} {v}" for k, v in counts.items()) + ")")
