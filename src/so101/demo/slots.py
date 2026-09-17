"""Where the slots are in the camera image, and which one a block is in.

Pixels, not millimetres. The exhibition never converts a detection into a
position in the arm's frame, and that is the point: converting needs the camera
calibrated against the robot, which is a measurement that has to be redone
whenever the camera moves - and the camera moves tomorrow.

What does not move is the robot, the slots and the can relative to each other.
So the arm's motion comes from taught joint angles, and the camera is only asked
which of three marked places on the table a block is sitting in. That question
survives the camera being put somewhere else with nothing but a re-click of
three points.

A block that is not near any slot gets no answer at all. `acceptance_radius_px`
is what makes "nearest" mean something: without it the nearest slot to a block
on the floor is still a slot.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CALIBRATION_PATH = Path("data/demo_slot_calibration.json")
#: How far from a slot's centre a detection may sit and still be called that
#: slot. A generous default: the slots are hand-marked and a block is 20 mm, so
#: a tight radius refuses good detections. Tightened by measurement, not guess.
DEFAULT_RADIUS_PX = 90.0


@dataclass
class SlotMap:
    """Slot centres in one camera's image, and how near counts as in one."""

    slots: dict                     # "slot1" -> (u, v)
    camera: str = "side"
    resolution: tuple = (800, 600)
    acceptance_radius_px: float = DEFAULT_RADIUS_PX
    created: str = ""
    note: str = ""

    # -- the question this exists to answer --------------------------------

    def nearest(self, pixel):
        """(slot name, distance px) for the slot nearest a detection.

        Returns (None, distance) when nothing is inside the acceptance radius -
        which is an answer, and the one that stops the arm.
        """
        if not self.slots:
            return None, math.inf
        best, best_distance = None, math.inf
        for name, centre in self.slots.items():
            distance = math.dist(pixel, centre)
            if distance < best_distance:
                best, best_distance = name, distance
        if best_distance > self.acceptance_radius_px:
            return None, best_distance
        return best, best_distance

    def ranked(self, pixel):
        """Every slot with its distance, nearest first. For explaining a refusal."""
        return sorted(((name, math.dist(pixel, centre))
                       for name, centre in self.slots.items()),
                      key=lambda pair: pair[1])

    def ambiguous(self, pixel, margin_px=25.0):
        """Are the two nearest slots too close together to choose between?

        A block exactly between two slots should stop the run rather than be
        assigned to whichever is a pixel nearer.
        """
        order = self.ranked(pixel)
        if len(order) < 2:
            return False
        return (order[1][1] - order[0][1]) < margin_px

    def number(self, name):
        """"slot2" -> 2, for finding the trajectory file."""
        return int(str(name).removeprefix("slot"))

    # -- storage -----------------------------------------------------------

    def save(self, path=None):
        path = Path(path or CALIBRATION_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "camera": self.camera,
            "resolution": list(self.resolution),
            "slots": {name: [round(u, 1), round(v, 1)]
                      for name, (u, v) in sorted(self.slots.items())},
            "acceptance_radius_px": self.acceptance_radius_px,
            "created": self.created or datetime.now().astimezone().isoformat(
                timespec="seconds"),
            "note": self.note,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path=None):
        path = Path(path or CALIBRATION_PATH)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} が見つかりません。"
                "先に uv run scripts/calibrate_demo_slots.py を実行してください")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(slots={name: tuple(centre)
                          for name, centre in data["slots"].items()},
                   camera=data.get("camera", "side"),
                   resolution=tuple(data.get("resolution", (800, 600))),
                   acceptance_radius_px=data.get("acceptance_radius_px",
                                                 DEFAULT_RADIUS_PX),
                   created=data.get("created", ""),
                   note=data.get("note", ""))

    def __str__(self):
        where = "、".join(f"{name} ({u:.0f},{v:.0f})"
                          for name, (u, v) in sorted(self.slots.items()))
        return (f"SlotMap({self.camera} {self.resolution[0]}x"
                f"{self.resolution[1]}, 半径 {self.acceptance_radius_px:.0f} px: "
                f"{where})")


def steady_centre(detector, stream, colour, frames=7, confidence=0.5,
                  log=print):
    """The median bbox centre of one colour over several frames.

    One frame is one detection and detections jitter; the median of seven costs
    a tenth of a second and does not move when one of them is odd. It also
    notices when the answer is not stable, which is the case worth refusing on:
    two blocks of the same colour swapping which is "first" between frames looks
    exactly like one block jumping across the table.

    Returns (centre, detail). `centre` is None when there is nothing to act on,
    and `detail["why"]` says what a person should be told.
    """
    import numpy as np

    seen, counts = [], []
    for _ in range(frames):
        image = stream.read().image
        found = [d for d in detector.detect(image, colours=[colour])
                 if d.confidence >= confidence]
        counts.append(len(found))
        if found:
            best = max(found, key=lambda d: d.confidence)
            seen.append((best.pixel[0], best.pixel[1], best.confidence))
    detail = {"frames": frames, "seen_in": len(seen),
              "counts": counts, "colour": colour}
    if not seen:
        detail["why"] = f"{colour} のブロックが見つかりません"
        return None, detail
    if len(seen) < frames * 0.6:
        detail["why"] = (f"{colour} が {frames} フレーム中 {len(seen)} 枚でしか"
                         f"見えません（検出が不安定です）")
        return None, detail
    if max(counts) > 1:
        detail["why"] = (f"{colour} のブロックが {max(counts)} 個見えます。"
                         f"どれを取るか決められません")
        return None, detail

    array = np.array(seen)
    centre = (float(np.median(array[:, 0])), float(np.median(array[:, 1])))
    spread = float(np.max(np.hypot(array[:, 0] - centre[0],
                                   array[:, 1] - centre[1])))
    detail.update({"centre_px": [round(centre[0], 1), round(centre[1], 1)],
                   "spread_px": round(spread, 1),
                   "confidence": round(float(np.median(array[:, 2])), 3)})
    if spread > 60.0:
        detail["why"] = (f"{colour} の検出位置が {spread:.0f} px ばらついています"
                         f"（同じブロックを見ていない可能性）")
        return None, detail
    return centre, detail
