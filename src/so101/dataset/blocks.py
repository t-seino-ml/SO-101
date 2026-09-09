"""Cut the coloured cubes out of the source photos.

The photos in `img/` are well exposed, shot square-on, and the cubes are painted
far more saturated than the wood behind them, so a saturation threshold separates
them cleanly there. That is emphatically not true of the deployment view - see
docs/07-vision.md - which is exactly why the crops exist: they get composited onto
many different backgrounds and lighting conditions to build a detector that does
not depend on any of this.

Each crop is saved RGBA, alpha from the segmentation mask, so it can be pasted
onto an arbitrary background without a halo.

Colour classes come from the measured hue clusters of the 36 blocks in `img/`:
six colours, six blocks each, with wide gaps between clusters.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# (name, hue_low, hue_high) on OpenCV's 0-179 scale, from the measured clusters.
COLOR_CLASSES = (
    ("red", 0, 4),
    ("orange", 4, 16),
    ("yellow", 20, 36),
    ("green", 38, 70),
    ("blue", 95, 114),
    ("purple", 114, 135),
)

DETECT_SCALE = 4          # segment on a downscaled copy; crop at full resolution
MIN_SATURATION = 120
MIN_VALUE = 50
MIN_AREA_SMALL = 2000     # in downscaled pixels
CROP_MARGIN = 0.12        # fraction of the box, so the cut does not clip corners
# Lab a/b distance from the block's own chroma. Tight enough to shed the cast
# shadow, loose enough to keep a cube's own shaded faces: at 26 whole faces were
# being cut away, which throws away the shape cue the detector needs.
CHROMA_TOLERANCE = 42


def classify(hue):
    for name, low, high in COLOR_CLASSES:
        if low <= hue < high:
            return name
    return None


def _mean_hue(hues):
    """Hue is circular, so average on the unit circle rather than arithmetically."""
    angles = np.deg2rad(hues.astype(float) * 2)
    mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return np.rad2deg(mean) % 360 / 2


def _saturated_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > MIN_SATURATION) & (hsv[:, :, 2] > MIN_VALUE))
    return mask.astype(np.uint8), hsv


def _drop_shadow(bgr, mask):
    """Trim the cast shadow that a saturation threshold drags in with the block.

    Shadowed wood is dark but still saturated, so it survives an S/V threshold and
    ends up glued to the cube. Compositing that onto a different background would
    paste a piece of the original table with it - precisely the leak these crops
    exist to avoid, since the table is a different colour on every rig.

    Shading moves lightness, not chroma, so the cube's own faces stay close to its
    colour in Lab a/b while the wood does not. Take the chroma of the eroded core,
    which is certainly block, and keep only what matches it.
    """
    core = cv2.erode(mask, np.ones((15, 15), np.uint8))
    if core.sum() < 50:
        core = mask
    # Wood is much darker than a lit face; require some lightness too, which
    # separates the shadow without touching the cube's own shading.
    lightness = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    floor = max(20, int(np.percentile(lightness[core.astype(bool)], 5)) - 45)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    reference = np.median(lab[core.astype(bool)][:, 1:], axis=0)
    distance = np.linalg.norm(lab[:, :, 1:] - reference, axis=2)
    keep = (distance < CHROMA_TOLERANCE) & (lightness >= floor)
    return mask & keep.astype(np.uint8)


def _refine(mask):
    """Close gaps, drop specks, keep the largest blob, and fill its holes."""
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def extract_blocks(source_dir="img", out_dir="data/blocks", verbose=True):
    """Write one RGBA crop per block, grouped by colour. Returns per-colour counts."""
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    counts = {name: 0 for name, _, _ in COLOR_CLASSES}
    unclassified = []

    for photo in sorted(source_dir.glob("*.jpg")) + sorted(source_dir.glob("*.png")):
        full = cv2.imread(str(photo))
        if full is None:
            continue
        small = cv2.resize(full, (full.shape[1] // DETECT_SCALE,
                                  full.shape[0] // DETECT_SCALE))
        mask_small, hsv_small = _saturated_mask(small)
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_small)

        for index in range(1, count):
            if stats[index][cv2.CC_STAT_AREA] < MIN_AREA_SMALL:
                continue
            hue = _mean_hue(hsv_small[:, :, 0][labels == index])
            colour = classify(hue)
            if colour is None:
                unclassified.append((photo.name, round(float(hue), 1)))
                continue

            x, y, w, h = (stats[index][cv2.CC_STAT_LEFT], stats[index][cv2.CC_STAT_TOP],
                          stats[index][cv2.CC_STAT_WIDTH], stats[index][cv2.CC_STAT_HEIGHT])
            pad = int(max(w, h) * CROP_MARGIN)
            x0 = max(0, (x - pad) * DETECT_SCALE)
            y0 = max(0, (y - pad) * DETECT_SCALE)
            x1 = min(full.shape[1], (x + w + pad) * DETECT_SCALE)
            y1 = min(full.shape[0], (y + h + pad) * DETECT_SCALE)
            crop = full[y0:y1, x0:x1]

            crop_mask, _ = _saturated_mask(crop)
            crop_mask = _refine(crop_mask)
            if crop_mask is not None:
                crop_mask = _refine(_drop_shadow(crop, crop_mask))
            if crop_mask is None:
                continue

            alpha = cv2.GaussianBlur(crop_mask * 255, (5, 5), 0)
            rgba = np.dstack([crop, alpha])

            counts[colour] += 1
            target = out_dir / colour
            target.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target / f"{photo.stem}_{counts[colour]:02d}.png"), rgba)

    if verbose:
        for name, _, _ in COLOR_CLASSES:
            print(f"  {name:<7} {counts[name]:3d} crops")
        if unclassified:
            print(f"  unclassified hues: {unclassified}")
    return counts
