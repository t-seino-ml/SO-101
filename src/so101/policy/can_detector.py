"""Find the can in a camera frame, without having trained anything on it.

The blocks got a detector of their own because six colours had to be told apart
and nothing off the shelf does that. The can needs none of it: there is one can,
it does not need classifying, and an open-vocabulary detector already knows what
a tin looks like. So this is YOLO-World with a text prompt, and the only work was
choosing the prompt and the threshold.

Measured on data/backgrounds/side_*.png, which is two sessions - 40 frames with
the table clear and 30 with the can present - at imgsz 1280:

    prompt             detect%   median conf   false boxes / 40 clear frames
    tin can                87%          0.72        1  at 0.50
    cup                    87%          0.53       38  at 0.25
    round metal tin        83%          0.47        0  at 0.30
    aluminum can           53%          0.28        0  at 0.25
    can                     7%          0.39        0  at 0.25
    container               0%             -        0

"tin can" wins on the margin rather than the rate: it scores the real can around
0.72 while everything it gets wrong sits far below, so a threshold can separate
them. "cup" finds the can just as often and scores half the office furniture the
same way.

The input size matters more than the model size. The can is about 75 px across in
an 800x600 frame and ultralytics resizes to 640 by default, which leaves it
around 60 px; at 1280 the detection rate goes from 80% to 87% and the confident
detections get much more confident. The x-sized model was tried and is worse as
well as five times heavier.

The 13% that are missed are all frames where the arm is parked in front of the
can. That is not a case worth engineering around here - the can is located before
the arm goes anywhere near it, and looked at again from above - but it is the
reason a caller should re-look rather than treat one empty result as "no can".

The threshold is 0.30, not the 0.50 those background frames suggested. Live, with
the can further off and the table full of blocks, it scores about 0.47 against the
0.72 it managed there, so 0.50 found it in 29% of frames where 0.25 found it in
100% of them. A threshold fitted to one distance does not transfer to another.

Confidence cannot be what rejects the wrong things, and it cannot be what ranks
them either. Measured live with both in view, the can on the working table scores
0.37 and another tin on the table behind it scores 0.47 - so taking the most
confident detection reaches for the wrong one. What separates them is where they
are: the working can lands 24 mm outside the calibrated region and 0.31 m from
the arm, the other 316 mm outside it and 0.63 m away, which is past anything this
arm can reach. So the filter is geometric and the requirement is the real one -
the can has to be somewhere the arm can drop a block into.

Where the can is comes from `CameraGeometry`, not from the box centre. The
homography maps the table plane, and the can stands 35 mm above it; the bottom of
its bounding box is the near edge of its base, which is on the table, and the
middle of the base is one radius further away from the camera. See that module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS = Path("weights/yolov8s-worldv2.pt")
PROMPT = "tin can"
CONFIDENCE = 0.30
IMAGE_SIZE = 1280
CAN_RADIUS_M = 0.0425          # 85 mm across
# The can has to be somewhere a block can be dropped into it, which is a shorter
# way of saying it has to be within reach. The demonstrations go out to 394 mm,
# so this leaves a little room beyond them and still excludes the next table.
MAX_REACH_M = 0.42
# A looser second guard, on how far outside the calibrated region a detection may
# sit. The homography degrades past its fitted area, so a can well outside it is
# either mis-placed or not the can.
MAX_OUTSIDE_MM = 250.0


@dataclass
class CanDetection:
    """The can, in pixels and - when a table frame is supplied - in metres."""

    confidence: float
    box: tuple              # (x0, y0, x1, y1) in pixels
    position: np.ndarray | None = None     # (x, y, z) in the arm's frame
    extrapolated_mm: float = 0.0

    @property
    def pixel(self):
        """The middle of the box: the can's opening, near enough."""
        x0, y0, x1, y1 = self.box
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    @property
    def base_pixel(self):
        """The middle of the box's bottom edge: the near edge of the can's base.

        The homography maps the table plane and nothing else, so a point on the
        can's rim - 35 mm up - does not map to where the can is. It maps to where
        the table would be if the line of sight carried on through, which is
        further from the camera; measured here, 83 mm further. This point is on
        the table by construction, so it does not suffer that.

        It is still not the middle of the can: it is the near edge of the base,
        one radius short. `CameraGeometry.on_table` walks that last radius.
        """
        x0, _, x1, y1 = self.box
        return np.array([(x0 + x1) / 2, y1])

    @property
    def reach_m(self):
        """How far the can is from the arm's base, in the table plane."""
        if self.position is None:
            return None
        return float(np.linalg.norm(self.position[:2]))

    @property
    def width_px(self):
        return self.box[2] - self.box[0]

    @property
    def trustworthy(self):
        return self.extrapolated_mm < 1.0

    def __str__(self):
        where = ("" if self.position is None
                 else f" at x={self.position[0]:+.3f} y={self.position[1]:+.3f}")
        warning = ("" if self.trustworthy
                   else f" [{self.extrapolated_mm:.0f}mm outside the calibration]")
        return f"can {self.confidence:.2f}{where}{warning}"


class CanDetector:
    """YOLO-World with the prompt and threshold measured on this rig."""

    def __init__(self, weights=None, device=None, table_frame=None,
                 geometry=None, prompt=PROMPT, confidence=CONFIDENCE,
                 image_size=IMAGE_SIZE, radius=CAN_RADIUS_M,
                 max_reach_m=MAX_REACH_M, max_outside_mm=MAX_OUTSIDE_MM):
        import torch
        from ultralytics import YOLOWorld

        weights = Path(weights) if weights else DEFAULT_WEIGHTS
        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        self.weights = weights
        self.prompt = prompt
        self.confidence = confidence
        self.image_size = image_size
        self.table_frame = table_frame
        # Without it, positions fall back to the base edge - a radius short of
        # the middle, but on the table at least, which the box centre is not.
        self.geometry = geometry
        self.radius = radius
        self.max_reach_m = max_reach_m
        self.max_outside_mm = max_outside_mm

        self.model = YOLOWorld(str(weights))
        # Move before set_classes, not after. set_classes reads the device off
        # the model's own parameters and builds the CLIP text encoder there;
        # called on a CPU-resident model that has since been moved, the tokens
        # and the encoder end up on different devices and the embedding lookup
        # raises "Expected all tensors to be on the same device".
        self.model.to(device if device == "cpu" else f"cuda:{device}")
        self.model.set_classes([prompt])
        self._warm = False

    def warmup(self, size=(600, 800)):
        if not self._warm:
            self.model(np.zeros((*size, 3), np.uint8), imgsz=self.image_size,
                       verbose=False)
            self._warm = True

    def detect(self, image, anywhere=False):
        """The can-like things in `image`, nearest the arm first.

        Not most-confident first. Measured live with two tins in view, the one on
        the working table scored 0.37 and one on the table behind it scored 0.47,
        so ranking by confidence reaches for the wrong tin. Anything out of reach
        is dropped outright, which is what excludes the rest of the office.

        Pass `anywhere` to see what was thrown away, positions and all.
        """
        self.warmup(image.shape[:2])
        result = self.model(image, conf=self.confidence, imgsz=self.image_size,
                            verbose=False)[0]

        found = []
        for box in result.boxes:
            corners = tuple(float(v) for v in box.xyxy[0])
            detection = CanDetection(float(box.conf), corners)
            if self.table_frame is not None:
                detection.position = self.position_of(corners)
                detection.extrapolated_mm = self.table_frame.outside_covered_mm(
                    detection.position)
                if not anywhere and (detection.reach_m > self.max_reach_m
                                     or detection.extrapolated_mm
                                     > self.max_outside_mm):
                    continue
            found.append(detection)
        found.sort(key=lambda d: (d.reach_m if d.position is not None
                                  else -d.confidence))
        return found

    def position_of(self, box):
        """Where the can standing in `box` is, in the arm's frame.

        With a camera geometry this is the middle of the can's base: the bottom
        of the box is the near edge of that base, and the middle is one radius
        further away from the camera. Without one it is that near edge, which is
        at least on the table.
        """
        height = self.table_frame.z_table
        if self.geometry is None:
            x0, _, x1, y1 = box
            x, y = self.table_frame.to_arm(np.array([(x0 + x1) / 2, y1]))
        else:
            x, y = self.geometry.on_table(self.table_frame, box, self.radius)
        return np.array([x, y, height])

    def locate(self, stream, tries=4, pause=0.15):
        """The can, asking again before concluding there is not one.

        One empty frame is one empty frame. The arm standing in front of the can
        accounts for every miss measured on this rig, so a caller that gives up
        on the first look gives up on a can that is there.
        """
        import time

        for _ in range(tries):
            found = self.detect(stream.read().image.copy())
            if found:
                return found[0]
            time.sleep(pause)
        return None
