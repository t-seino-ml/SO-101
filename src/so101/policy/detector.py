"""Find blocks in a camera frame, and say where they are for the arm.

Wraps the trained YOLO model with the settings that were measured to work on this
rig, and optionally maps each detection through the table homography so a caller
gets arm coordinates rather than pixels.

Class-agnostic NMS is the important setting. The same block sometimes picks up two
boxes with different colour labels - blue and purple are only fifteen hue units
apart, and the overhead view is dark enough to blur that - and per-class NMS
cannot merge boxes that disagree about the class. Measured on a table holding 12
blocks: the default settings reported 19 detections; class-agnostic NMS reported
exactly 12.

Note that merging the boxes does not fix the colour: it picks the more confident
label. The overhead camera still mislabels blue and purple in dim light, while
the side camera gets all six colours right. See docs/07-vision.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..dataset.scenes import CLASS_NAMES

DEFAULT_WEIGHTS = Path("runs/detect/blocks/weights/best.pt")
CONFIDENCE = 0.4
NMS_IOU = 0.45
AGNOSTIC_NMS = True


def find_weights(explicit=None):
    """Where the detector's weights actually are.

    Ultralytics decides the run directory itself, and where it puts "detect"
    in the path depends on its own settings - so the trained weights do not
    reliably land where the training script said they would. Rather than
    hard-code one guess, take the newest best.pt under runs/.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        return path
    if DEFAULT_WEIGHTS.is_file():
        return DEFAULT_WEIGHTS
    found = sorted(Path("runs").rglob("weights/best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not found:
        raise FileNotFoundError(
            "No trained detector under runs/. "
            "Train one: uv run scripts/train_detector.py")
    return found[0]


@dataclass
class Detection:
    """One block, in pixels and - when a table frame is supplied - in metres."""

    colour: str
    confidence: float
    box: tuple          # (x0, y0, x1, y1) in pixels
    position: np.ndarray | None = None   # (x, y, z) in the arm's frame
    # How far beyond the calibrated region this position sits. Anything above
    # zero is an extrapolation of the homography, and those degrade fast.
    extrapolated_mm: float = 0.0

    @property
    def pixel(self):
        x0, y0, x1, y1 = self.box
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    @property
    def width_px(self):
        return self.box[2] - self.box[0]

    @property
    def trustworthy(self):
        """Is this position a measurement, or the homography guessing?"""
        return self.extrapolated_mm < 1.0

    def __str__(self):
        where = ("" if self.position is None
                 else f" at x={self.position[0]:+.3f} y={self.position[1]:+.3f}")
        warning = ("" if self.trustworthy
                   else f" [{self.extrapolated_mm:.0f}mm outside the calibration]")
        return f"{self.colour} {self.confidence:.2f}{where}{warning}"


class BlockDetector:
    """The trained detector, with this rig's inference settings."""

    def __init__(self, weights=None, device=None, table_frame=None,
                 confidence=CONFIDENCE, nms_iou=NMS_IOU, agnostic=AGNOSTIC_NMS):
        weights = find_weights(weights)
        import torch
        from ultralytics import YOLO

        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = YOLO(str(weights))
        self.model.to(device if device == "cpu" else f"cuda:{device}")
        self.table_frame = table_frame
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.agnostic = agnostic
        self._warm = False

    def warmup(self, size=(600, 800)):
        """Pay the CUDA context and graph setup cost before it matters."""
        if not self._warm:
            self.model(np.zeros((*size, 3), np.uint8), verbose=False)
            self._warm = True

    def detect(self, image, colours=None):
        """Detections in `image`, optionally filtered to certain colours."""
        self.warmup(image.shape[:2])
        result = self.model(image, conf=self.confidence, iou=self.nms_iou,
                            agnostic_nms=self.agnostic, verbose=False)[0]

        wanted = None if colours is None else set(colours)
        detections = []
        for box in result.boxes:
            colour = CLASS_NAMES[int(box.cls)]
            if wanted is not None and colour not in wanted:
                continue
            corners = tuple(float(v) for v in box.xyxy[0])
            detection = Detection(colour, float(box.conf), corners)
            if self.table_frame is not None:
                detection.position = self.table_frame.reach_target(detection.pixel)
                detection.extrapolated_mm = self.table_frame.outside_covered_mm(
                    detection.position)
            detections.append(detection)

        # Nearest first: the arm should clear what is closest before reaching past it.
        if self.table_frame is not None:
            detections.sort(key=lambda d: float(np.linalg.norm(d.position[:2])))
        else:
            detections.sort(key=lambda d: -d.confidence)
        return detections

    def counts(self, image):
        counts = {}
        for detection in self.detect(image):
            counts[detection.colour] = counts.get(detection.colour, 0) + 1
        return counts
