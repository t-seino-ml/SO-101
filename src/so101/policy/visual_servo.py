"""Close the last centimetre with the wrist camera, so nothing has to be exact.

Open loop needs the offset between where a block is detected and where the jaws
have to go. That offset was measured four ways and the four disagreed by up to
70 mm, against a grasp that tolerates 5.6 mm. Worse, it is not even a constant:
it is fixed in the gripper's frame, so it turns as the arm's posture changes, and
the reaching poses are blended from demonstrations - which means two targets a
couple of centimetres apart can be approached in quite different postures and
want offsets 38 mm apart. Measured across three positions: 2.5 mm, 8.5 mm and
38 mm of variation from the one place it was taught.

The wrist camera has none of that trouble. It is bolted to the gripper, so a
block sitting where the jaws could close on it appears at the same pixel whatever
posture the arm is in. That pixel is what `data/servo_target.json` holds, measured
by having the arm set a block down and then look at it from several heights.

Two things the measurement also settled:

- Below about 20 mm the wrist camera cannot focus. In a frame taken at grasping
  height the detector found every block across the table and none of the ones
  under the gripper. So the alignment happens at 25-30 mm and the arm then drops
  straight down, which does not change x or y.
- The target pixel moves a long way with height - from (378, 309) at 20 mm to
  (407, 345) at 60 mm - which is why aligning at one height and grasping at
  another misses.

Colour is not used to follow the block. Phase 1 measured the wrist view reading
blue as purple in 98% of frames and yellow as orange in 71%, because its white
balance runs warm; the side camera gets all six right and has already chosen
which block this is. Here the block is simply the detection nearest to where it
was last seen.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .motion import joints_of, move_to, tip_of

TARGET_PATH = Path("data/servo_target.json")
SERVO_HEIGHT_M = 0.030     # where the camera can still focus, and the target is known
JOG_M = 0.012
MAX_ROUNDS = 4
GAIN = 0.8
TOLERANCE_PX = 12
MAX_STEP_MM = 25.0
TRACK_PX = 220             # how far the same block may jump between looks
MIN_SCALE_PX_PER_MM = 0.5
MAX_SCALE_PX_PER_MM = 40.0
MAX_SCALE_SKEW = 4.0
MIN_CONFIDENCE = 0.35      # low: the wrist view is dim and mislabels freely


class ServoTarget:
    """Where a graspable block appears in the wrist view, by height."""

    def __init__(self, heights, pixels, released_at=None, offset=None):
        order = np.argsort(heights)
        self.heights = np.asarray(heights, float)[order]
        self.pixels = np.asarray(pixels, float)[order]
        self.released_at = released_at
        self.offset = offset

    @classmethod
    def load(cls, path=TARGET_PATH):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/teach_servo_target.py first.")
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data["targets"]
        return cls([e["height_m"] for e in entries],
                   [e["pixel"] for e in entries],
                   data.get("released_at"), data.get("offset_m"))

    def at(self, height):
        """The target pixel at `height` above the grasp, interpolated.

        Outside the measured range this holds the nearest end rather than
        extrapolating: the relation is not linear - the measured steps run
        0.6, 2.0 and 1.6 px/mm - and guessing past the ends is how a servo ends
        up converging confidently onto nowhere.
        """
        height = float(np.clip(height, self.heights[0], self.heights[-1]))
        return np.array([np.interp(height, self.heights, self.pixels[:, axis])
                         for axis in (0, 1)])

    @property
    def lowest(self):
        return float(self.heights[0])

    def __str__(self):
        return (f"ServoTarget({len(self.heights)} heights, "
                f"{1000*self.heights[0]:.0f}..{1000*self.heights[-1]:.0f} mm)")


def find_block(detector, stream, near, radius=None, tries=4, pause=0.12):
    """The detection nearest `near`, asking again before giving up.

    One missed frame is one missed frame, not a lost block. No colour filter:
    see the note in the module docstring.
    """
    for _ in range(tries):
        seen = [d for d in detector.detect(stream.read().image.copy())
                if d.confidence >= MIN_CONFIDENCE]
        if seen:
            best = min(seen, key=lambda d: np.linalg.norm(d.pixel - near))
            if radius is None or np.linalg.norm(best.pixel - near) <= radius:
                return best
        time.sleep(pause)
    return None


def measure_jacobian(robot, arm, approach, detector, wrist, pose, where,
                     jog=JOG_M, workspace=None, log=print):
    """Pixels per millimetre here, by stepping the arm and watching the block.

    A property of where the arm is standing, not of the camera: the wrist does
    not look straight down and the posture changes as the arm reaches, so it is
    measured in place. Two small moves, and the arm is put back afterwards.

    Returns (matrix, problem). The matrix maps arm millimetres to pixels.
    """
    base = find_block(detector, wrist, where, radius=TRACK_PX)
    if base is None:
        return None, "the block is not in the wrist camera's view"
    start = tip_of(arm, robot)

    pixels, millimetres = [], []
    for axis in (0, 1):
        step = np.zeros(3)
        step[axis] = jog
        if move_to(robot, arm, approach, pose, start + step, seconds=1.0,
                   workspace=workspace) is None:
            return None, "cannot step far enough to measure the scale"
        time.sleep(0.5)
        travelled = tip_of(arm, robot) - start
        if abs(travelled[2]) > 0.012:
            move_to(robot, arm, approach, pose, start, seconds=0.8,
                    workspace=workspace)
            return None, (f"the step also moved {1000*travelled[2]:+.0f} mm "
                          "vertically, so it did not go where it was sent")
        if np.linalg.norm(travelled[:2]) < 0.6 * jog:
            move_to(robot, arm, approach, pose, start, seconds=0.8,
                    workspace=workspace)
            return None, (f"the arm moved {1000*np.linalg.norm(travelled):.0f} mm "
                          f"of the {1000*jog:.0f} mm asked for")
        seen = find_block(detector, wrist, base.pixel, TRACK_PX)
        if seen is None:
            move_to(robot, arm, approach, pose, start, seconds=0.8,
                    workspace=workspace)
            return None, "lost track of the block while measuring"
        pixels.append(seen.pixel - base.pixel)
        millimetres.append(1000 * travelled[:2])
        move_to(robot, arm, approach, pose, start, seconds=0.8, workspace=workspace)
        base = find_block(detector, wrist, base.pixel, TRACK_PX) or base

    matrix = np.column_stack(pixels) @ np.linalg.inv(np.column_stack(millimetres))
    strength = np.linalg.norm(matrix, axis=0)
    if min(strength) < MIN_SCALE_PX_PER_MM or max(strength) > MAX_SCALE_PX_PER_MM:
        return None, (f"scale came out {strength[0]:.1f} and {strength[1]:.1f} "
                      "px/mm, which cannot be right")
    # Nearly parallel directions invert into nonsense, and a few pixels of
    # detection noise then become centimetres of movement.
    spread = np.linalg.svd(matrix, compute_uv=False)
    if spread[0] / max(spread[1], 1e-9) > MAX_SCALE_SKEW:
        return None, (f"the two directions came out {spread[0]/spread[1]:.0f} "
                      "times apart; correcting from that is not stable")
    return matrix, None


def align(robot, arm, approach, detector, wrist, pose, target_pixel, jacobian,
          rounds=MAX_ROUNDS, gain=GAIN, tolerance=TOLERANCE_PX, workspace=None,
          log=print):
    """Move until the block sits at `target_pixel`. Returns (error_px, problem).

    Keeps the best position seen. The error usually falls for two or three rounds
    and then, if the scale is a little off, starts to climb; going back to the
    best turns a diverging attempt into a merely imperfect one.
    """
    last = target_pixel
    best, best_at, worse = np.inf, None, 0
    error = None
    for round_number in range(1, rounds + 1):
        seen = find_block(detector, wrist, last,
                          None if round_number == 1 else TRACK_PX)
        if seen is None:
            return error, "lost track of the block"
        last = seen.pixel
        error = seen.pixel - target_pixel
        distance = float(np.linalg.norm(error))
        wanted_mm = np.linalg.solve(jacobian, target_pixel - seen.pixel)
        log(f"      {round_number}: off by {distance:.0f} px "
            f"({wanted_mm[0]:+.0f}, {wanted_mm[1]:+.0f} mm)")

        if distance < best:
            best, best_at, worse = distance, tip_of(arm, robot), 0
        elif distance > best + 5:
            worse += 1
            if worse >= 2:
                log(f"      drifting; going back to the best, {best:.0f} px")
                move_to(robot, arm, approach, pose, best_at, workspace=workspace)
                return np.array([best, 0.0]), None
        if distance <= tolerance:
            return error, None

        step = gain * wanted_mm
        if np.linalg.norm(step) > MAX_STEP_MM:
            step *= MAX_STEP_MM / np.linalg.norm(step)
        here = tip_of(arm, robot)
        if move_to(robot, arm, approach, pose,
                   here + np.array([step[0] / 1000, step[1] / 1000, 0.0]),
                   workspace=workspace) is None:
            return error, "cannot move any further that way"
    return error, None
