"""Synthesise detector training scenes from the block crops.

The detector has to survive a table it has never seen: the rig moves, and the
surface underneath it is a different colour and texture every time. Nothing here
may encode "the background is pale wood".

So backgrounds are drawn from a wide procedural distribution - including surfaces
the same colour as the blocks, the case that breaks any colour-only approach -
plus real camera frames when there are any. Lighting, white balance, noise and
blur vary on top. Distractors that are saturated but not cube-shaped are scattered
in, so "coloured blob" alone is not enough to fire the detector.

Block sizes come from measuring the real view: 20-60 px across in an 800x600
frame, so the synthetic range is 16-80 px.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ("red", "orange", "yellow", "green", "blue", "purple")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}

SCENE_SIZE = (800, 600)
BLOCK_PX = (16, 80)          # measured 20-60 in the real view, with headroom
BLOCKS_PER_SCENE = (0, 16)
DISTRACTORS_PER_SCENE = (0, 5)
EDGE_OVERHANG = 0.15         # blocks may hang this far off the frame edge


# --- backgrounds ---------------------------------------------------------

def _solid(size, rng):
    return np.full((size[1], size[0], 3), rng.integers(20, 236, 3), np.uint8)


def _gradient(size, rng):
    first = rng.integers(10, 246, 3).astype(float)
    second = rng.integers(10, 246, 3).astype(float)
    horizontal = rng.random() < 0.5
    ramp = np.linspace(0, 1, size[0] if horizontal else size[1])
    mix = first[None, :] * (1 - ramp[:, None]) + second[None, :] * ramp[:, None]
    if horizontal:
        return np.tile(mix[None, :, :], (size[1], 1, 1)).astype(np.uint8)
    return np.tile(mix[:, None, :], (1, size[0], 1)).astype(np.uint8)


def _streaked(size, rng):
    """Directional grain over an arbitrary base colour - wood-like, but any hue."""
    base = rng.integers(30, 226, 3).astype(float)
    scene = np.tile(base, (size[1], size[0], 1))
    grain = rng.normal(0, 1, (size[1], size[0]))
    grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=rng.uniform(0.6, 3),
                             sigmaY=rng.uniform(10, 60))
    scene += (grain / (np.abs(grain).max() + 1e-6) * rng.uniform(10, 55))[:, :, None]
    matrix = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2),
                                     rng.uniform(-90, 90), 1.3)
    scene = cv2.warpAffine(scene, matrix, size, borderMode=cv2.BORDER_REFLECT)
    return np.clip(scene, 0, 255).astype(np.uint8)


def _mottled(size, rng):
    small = rng.integers(0, 256, (size[1] // 8, size[0] // 8, 3)).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), rng.uniform(1, 4))
    scene = cv2.resize(small, size, interpolation=cv2.INTER_LINEAR)
    tint = rng.integers(20, 236, 3).astype(float)
    weight = rng.uniform(0.55, 0.9)
    return np.clip(scene * (1 - weight) + tint * weight, 0, 255).astype(np.uint8)


BACKGROUND_MAKERS = (_solid, _gradient, _streaked, _streaked, _mottled)


def random_background(size, rng, real_backgrounds=None, real_probability=0.35):
    """A procedural surface, or a real camera frame when some are available."""
    if real_backgrounds is not None and len(real_backgrounds) and \
            rng.random() < real_probability:
        image = real_backgrounds[int(rng.integers(len(real_backgrounds)))]
        if image.shape[1::-1] != tuple(size):
            image = cv2.resize(image, size)
        return image.copy()
    maker = BACKGROUND_MAKERS[int(rng.integers(len(BACKGROUND_MAKERS)))]
    return maker(size, rng)


# --- pasting -------------------------------------------------------------

def _oriented(crop, target_px, rng):
    """Scale, rotate and mildly skew a crop, keeping its alpha."""
    scale = target_px / max(crop.shape[:2])
    resized = cv2.resize(crop, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    height, width = resized.shape[:2]
    side = int(math.hypot(width, height)) + 4
    canvas = np.zeros((side, side, 4), np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top:top + height, left:left + width] = resized

    matrix = cv2.getRotationMatrix2D((side / 2, side / 2), rng.uniform(0, 360), 1.0)
    canvas = cv2.warpAffine(canvas, matrix, (side, side), flags=cv2.INTER_LINEAR)

    # A little perspective: the real cameras look at the table from an angle.
    if rng.random() < 0.6:
        jitter = side * 0.08
        source = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
        target = source + rng.uniform(-jitter, jitter, source.shape).astype(np.float32)
        canvas = cv2.warpPerspective(
            canvas, cv2.getPerspectiveTransform(source, target), (side, side))
    return canvas


def _shade(rgba, rng):
    """Vary each block's own exposure, so its colour is not a fixed constant."""
    bgr = rgba[:, :, :3].astype(np.float32)
    bgr *= rng.uniform(0.55, 1.35)
    bgr += rng.normal(0, rng.uniform(0, 6), bgr.shape)
    rgba[:, :, :3] = np.clip(bgr, 0, 255).astype(np.uint8)
    return rgba


def _alpha_paste(scene, rgba, top_left):
    x, y = top_left
    height, width = rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1 = min(scene.shape[1], x + width)
    y1 = min(scene.shape[0], y + height)
    if x0 >= x1 or y0 >= y1:
        return
    patch = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    region = scene[y0:y1, x0:x1].astype(np.float32)
    scene[y0:y1, x0:x1] = (patch[:, :, :3] * alpha
                           + region * (1 - alpha)).astype(np.uint8)


def _drop_shadow(scene, rgba, top_left, offset, strength):
    """Ground the block. Without a shadow the paste reads as a floating sticker."""
    shadow = np.zeros_like(rgba)
    blurred = cv2.GaussianBlur(rgba[:, :, 3], (0, 0), 3).astype(np.float32)
    shadow[:, :, 3] = (blurred * strength).astype(np.uint8)
    _alpha_paste(scene, shadow, (top_left[0] + offset[0], top_left[1] + offset[1]))


def _distractor(scene, rng):
    """Saturated but not cube-shaped, so shape carries part of the decision."""
    height, width = scene.shape[:2]
    colour = tuple(int(c) for c in rng.integers(0, 256, 3))
    cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
    radius = int(rng.integers(8, 60))
    kind = int(rng.integers(3))
    if kind == 0:
        cv2.circle(scene, (cx, cy), radius, colour, -1)
    elif kind == 1:
        cv2.ellipse(scene, (cx, cy), (radius, max(4, radius // 3)),
                    float(rng.uniform(0, 180)), 0, 360, colour, -1)
    else:
        points = np.array([[cx + int(rng.integers(-radius, radius)),
                            cy + int(rng.integers(-radius, radius))]
                           for _ in range(int(rng.integers(5, 9)))], np.int32)
        cv2.fillPoly(scene, [points], colour)


# --- whole-scene photometrics -------------------------------------------

def photometric(scene, rng):
    """Exposure, contrast, white balance, falloff, noise, blur and JPEG."""
    scene = scene.astype(np.float32)
    scene *= rng.uniform(0.45, 1.5)
    mean = scene.mean()
    scene = (scene - mean) * rng.uniform(0.6, 1.5) + mean
    scene *= rng.uniform(0.82, 1.18, 3)[None, None, :]
    scene = np.clip(scene, 0, 255)
    scene = 255 * (scene / 255) ** rng.uniform(0.7, 1.45)

    if rng.random() < 0.5:
        height, width = scene.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        falloff = 1 - rng.uniform(0.15, 0.6) * (
            ((xx - cx) ** 2 + (yy - cy) ** 2) / (width ** 2 + height ** 2))
        scene *= falloff[:, :, None]

    scene = np.clip(scene, 0, 255)
    if rng.random() < 0.7:
        scene += rng.normal(0, rng.uniform(1, 10), scene.shape)
    scene = np.clip(scene, 0, 255).astype(np.uint8)

    if rng.random() < 0.4:
        scene = cv2.GaussianBlur(scene, (0, 0), rng.uniform(0.4, 1.8))
    if rng.random() < 0.5:
        quality = int(rng.integers(35, 95))
        ok, buffer = cv2.imencode(".jpg", scene, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            scene = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return scene


# --- scene assembly ------------------------------------------------------

def load_crops(blocks_dir="data/blocks"):
    """{colour: [RGBA crop, ...]} from the extracted blocks."""
    crops = {}
    for name in CLASS_NAMES:
        files = sorted(Path(blocks_dir, name).glob("*.png"))
        images = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in files]
        crops[name] = [image for image in images
                       if image is not None and image.shape[2] == 4]
    return crops


def load_backgrounds(backgrounds_dir="data/backgrounds"):
    directory = Path(backgrounds_dir)
    if not directory.is_dir():
        return []
    images = [cv2.imread(str(path)) for path in sorted(directory.glob("*.png"))]
    return [image for image in images if image is not None]


def make_scene(crops, rng, size=SCENE_SIZE, real_backgrounds=None):
    """Return (image, labels), labels as (class_id, cx, cy, w, h) normalised."""
    scene = random_background(size, rng, real_backgrounds)
    for _ in range(int(rng.integers(*DISTRACTORS_PER_SCENE))):
        _distractor(scene, rng)

    shadow_offset = (int(rng.integers(-9, 10)), int(rng.integers(-9, 10)))
    shadow_strength = float(rng.uniform(0.15, 0.6))

    labels = []
    for _ in range(int(rng.integers(*BLOCKS_PER_SCENE))):
        colour = CLASS_NAMES[int(rng.integers(len(CLASS_NAMES)))]
        if not crops[colour]:
            continue
        crop = crops[colour][int(rng.integers(len(crops[colour])))].copy()
        rgba = _shade(_oriented(crop, int(rng.integers(*BLOCK_PX)), rng), rng)

        height, width = rgba.shape[:2]
        x = int(rng.integers(int(-EDGE_OVERHANG * width),
                             max(1, size[0] - int((1 - EDGE_OVERHANG) * width))))
        y = int(rng.integers(int(-EDGE_OVERHANG * height),
                             max(1, size[1] - int((1 - EDGE_OVERHANG) * height))))

        _drop_shadow(scene, rgba, (x, y), shadow_offset, shadow_strength)
        _alpha_paste(scene, rgba, (x, y))

        # Box the visible alpha, not the padded canvas, and clip to the frame.
        visible = rgba[:, :, 3] > 40
        if not visible.any():
            continue
        ys, xs = np.where(visible)
        bx0 = max(0, x + int(xs.min()))
        by0 = max(0, y + int(ys.min()))
        bx1 = min(size[0] - 1, x + int(xs.max()))
        by1 = min(size[1] - 1, y + int(ys.max()))
        if bx1 - bx0 < 6 or by1 - by0 < 6:
            continue
        labels.append((CLASS_IDS[colour],
                       (bx0 + bx1) / 2 / size[0], (by0 + by1) / 2 / size[1],
                       (bx1 - bx0) / size[0], (by1 - by0) / size[1]))

    return photometric(scene, rng), labels
