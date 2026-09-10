"""Calibrate the camera against the arm by watching the arm move.

No markers and no measuring. The arm holds one block, places it at a series of
positions it computes for itself, and the detector reads where each one lands in
the image. Forward kinematics says where the block actually was, so every visit
yields one pixel-to-arm correspondence, and a homography falls out of the set.

The correspondences come from the arm's own motion, so the procedure re-runs
whenever the camera has moved - it takes about a minute and needs nobody watching.

Identifying which detection is the held block: it is the one that moves.

Each pose is compared with the last. A block lying on the table appears at the
same pixel every time; the carried one appears somewhere new. So the held block is
the detection furthest from where any detection sat in the previous frame, and a
frame where nothing has moved is a failed measurement rather than a reading.

Two earlier approaches did not survive contact:

- Overlap with a frame difference. Boxes of blocks the arm was demonstrably
  carrying overlapped the changed region by only 0.12-0.15 - an axis-aligned box
  around a tilted cube is mostly table - against 0.00 for stationary ones. The two
  populations are too close to separate with a threshold.
- Excluding whatever was visible with the arm parked. The arm parks *holding the
  block*, so the block itself got recorded as furniture and then every subsequent
  sighting of it was discarded. Two usable points out of twenty.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

DIFF_THRESHOLD = 28           # per-pixel intensity change that counts as movement
# Fraction of a detection's box that must fall in the changed region. The box is
# axis-aligned around a cube seen at an angle, so a good chunk of it is table even
# when the block is dead centre - measured overlaps of 0.12-0.15 for blocks the arm
# was demonstrably carrying, against 0.00 for stationary ones. The gap between
# those two populations is what matters, not the absolute value.
DIFF_MIN_OVERLAP = 0.08
MIN_PIXEL_TRAVEL = 6.0        # the held block must visibly move between poses
SETTLE_SECONDS = 0.6          # let the arm stop shaking and the camera catch up
STEP_DEG = 1.5
STEP_DELAY = 0.02
PLACE_CLEARANCE = 0.004       # rest the block this far above the table


def grid_positions(x_range=(0.16, 0.28), y_range=(-0.10, 0.10), shape=(3, 4)):
    """A spread of table positions to visit. Spread matters more than count.

    Points clustered together fit a homography that is accurate there and wrong
    everywhere else, so this covers the working area rather than sampling densely.
    """
    xs = np.linspace(*x_range, shape[0])
    ys = np.linspace(*y_range, shape[1])
    positions = [(float(x), float(y)) for x in xs for y in ys]
    # Serpentine order keeps consecutive moves short.
    ordered = []
    for index, x in enumerate(xs):
        row = [p for p in positions if p[0] == x]
        ordered.extend(row if index % 2 == 0 else row[::-1])
    return ordered


def changed_mask(reference, frame):
    """Where the scene differs from the reference: the arm and what it carries."""
    delta = cv2.absdiff(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    mask = (delta > DIFF_THRESHOLD).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def held_block(detections, mask, expect_colour=None):
    """The detection sitting in the changed region, i.e. the one the arm carries.

    `expect_colour` locks onto the colour of the block actually being carried. The
    arm's own body changes the scene too, so a block already lying on the table can
    fall inside the changed region and win on overlap alone - which is how a single
    visit once reported an orange block while the gripper held a purple one, and
    put a 186 mm outlier into the fit.
    """
    best, best_overlap = None, 0.0
    for detection in detections:
        if expect_colour is not None and detection.colour != expect_colour:
            continue
        x0, y0, x1, y1 = (int(v) for v in detection.box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(mask.shape[1], x1), min(mask.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            continue
        overlap = float(mask[y0:y1, x0:x1].mean())
        if overlap > best_overlap:
            best, best_overlap = detection, overlap
    if best_overlap < DIFF_MIN_OVERLAP:
        return None, best_overlap
    return best, best_overlap


MOVED_RADIUS_PX = 20          # a detection within this of an old one has not moved
MAX_TRAVEL_PX = 260           # one step of the grid never moves a block further


def most_moved(detections, previous_pixels, colour=None):
    """The detection furthest from anything seen last time.

    Returns (detection, distance), or (None, 0) when nothing qualifies.

    Once the carried block's colour is known, only that colour is considered -
    and if none is visible this frame, the answer is "nothing", not "have this
    other block instead". Falling back to the full set is what let a spurious
    purple detection in the corner of the frame stand in for the green block the
    arm was holding, putting two 700 mm outliers into a fit.
    """
    candidates = [d for d in detections
                  if colour is None or d.colour == colour]
    if not candidates:
        return None, 0.0
    if not previous_pixels:
        return max(candidates, key=lambda d: d.confidence), float("inf")
    scored = [(min(float(np.linalg.norm(d.pixel - old)) for old in previous_pixels), d)
              for d in candidates]
    distance, best = max(scored, key=lambda pair: pair[0])
    return best, distance


class AutoCalibrator:
    """Visits positions with a block in the gripper and records where it appears."""

    def __init__(self, robot, kinematics, stream, detector, z_table,
                 park_joints=None, verbose=True):
        self.robot = robot
        self.arm = kinematics
        self.stream = stream
        self.detector = detector
        self.z_table = z_table
        self.park_joints = park_joints
        self.verbose = verbose
        self.reference = None

    # -- plumbing ---------------------------------------------------------

    def _log(self, message):
        if self.verbose:
            print(f"    {message}", flush=True)

    def joints(self):
        observation = self.robot.get_observation()
        return {key.removesuffix(".pos"): float(value)
                for key, value in observation.items() if key.endswith(".pos")}

    def arm_joints(self):
        current = self.joints()
        return {name: current[name] for name in self.arm.joint_names}

    def frame(self):
        time.sleep(SETTLE_SECONDS)
        captured = self.stream.read()
        if captured is None:
            raise RuntimeError("No camera frame")
        return captured.image.copy()

    def move_joints(self, target):
        start = self.joints()
        goal = dict(start)
        goal.update(target)
        travel = max(abs(goal[name] - start[name]) for name in goal)
        for step in range(1, max(1, int(travel / STEP_DEG)) + 1):
            fraction = step / max(1, int(travel / STEP_DEG))
            self.robot.send_action(
                {f"{name}.pos": start[name] + (goal[name] - start[name]) * fraction
                 for name in goal})
            time.sleep(STEP_DELAY)
        return True

    def move_to(self, position):
        solution = self.arm.inverse(position, seed_deg=self.arm_joints())
        if solution is None:
            return False
        return self.move_joints(solution)

    # -- procedure --------------------------------------------------------

    def take_reference(self):
        """Park the arm clear of the table and remember what the scene looks like."""
        if self.park_joints:
            self._log("parking the arm out of the way")
            self.move_joints(self.park_joints)
        self.reference = self.frame()
        return self.reference

    def visit(self, positions):
        """Place the held block at each position; return the correspondences."""
        if self.reference is None:
            raise RuntimeError("Call take_reference() first")

        pixels, arm_xy, skipped = [], [], []
        colour = None
        previous_pixels = [d.pixel for d in self.detector.detect(self.reference)]
        self._log(f"{len(previous_pixels)} block(s) visible with the arm parked")
        for index, (x, y) in enumerate(positions):
            target = np.array([x, y, self.z_table + PLACE_CLEARANCE])
            if not self.move_to(target):
                skipped.append(((x, y), "unreachable"))
                continue

            image = self.frame()
            detections = self.detector.detect(image)
            if not detections:
                skipped.append(((x, y), "nothing detected"))
                continue

            detection, moved = most_moved(detections, previous_pixels, colour)
            previous_pixels = [d.pixel for d in detections]
            if detection is None:
                skipped.append(((x, y), f"no {colour} block visible "
                                        f"({len(detections)} other detection(s))"))
                continue
            if moved < MOVED_RADIUS_PX:
                skipped.append(((x, y), f"nothing moved (best {moved:.0f} px) - "
                                        "the block may have been dropped"))
                continue
            # A jump larger than the grid spacing means this is a different object,
            # not the carried block having travelled.
            if moved != float("inf") and moved > MAX_TRAVEL_PX:
                skipped.append(((x, y), f"implausible jump ({moved:.0f} px) - "
                                        "probably a spurious detection"))
                continue
            if colour is None:
                colour = detection.colour
                self._log(f"carrying a {colour} block")

            # Where the block actually ended up, not where it was asked to go.
            tip = self.arm.forward(self.arm_joints())
            pixels.append(detection.pixel)
            arm_xy.append(tip[:2])
            self._log(f"[{index + 1}/{len(positions)}] asked x={x:+.3f} y={y:+.3f}, "
                      f"reached x={tip[0]:+.3f} y={tip[1]:+.3f}, "
                      f"{detection.colour} at pixel "
                      f"({detection.pixel[0]:.0f}, {detection.pixel[1]:.0f})")

        return np.array(pixels), np.array(arm_xy), skipped
