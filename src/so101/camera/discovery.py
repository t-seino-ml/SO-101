"""Camera discovery with a stable identity.

OpenCV addresses cameras by index, and those indices shift whenever devices are
plugged, unplugged, or enumerated in a different order - the same problem
`hardware.ports` solves for serial ports. DirectShow exposes the device name, which
survives reordering, so cameras are identified by name here and the index is only
resolved at open time.

On Windows the DirectShow enumeration order matches OpenCV's `CAP_DSHOW` indices,
so `device_names()[i]` is the camera OpenCV opens at index `i`. Elsewhere no names
are available and callers fall back to indices.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

import cv2

IS_WINDOWS = sys.platform == "win32"
BACKEND = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY
MAX_INDEX = 10
WARMUP_FRAMES = 5  # the first frames off a UVC camera are often black or stale


def device_names():
    """DirectShow device names, index-aligned with OpenCV. Empty if unavailable."""
    if not IS_WINDOWS:
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return []
    try:
        return list(FilterGraph().get_input_devices())
    except Exception:  # noqa: BLE001 - a COM failure must not break discovery
        return []


@dataclass
class CameraInfo:
    index: int
    name: str
    opened: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0

    @property
    def label(self):
        return f"[{self.index}] {self.name}"

    def __str__(self):
        if not self.opened:
            return f"{self.label} - could not open"
        return f"{self.label} - {self.width}x{self.height} @ {self.fps:.0f} fps"


def probe(index, warmup=WARMUP_FRAMES):
    """Open one camera, read a frame, and report what it actually delivers.

    Returns (CameraInfo, frame or None). The reported size comes from the frame
    itself, not from the capture properties, which lie on plenty of UVC devices.
    """
    names = device_names()
    name = names[index] if index < len(names) else f"OpenCV camera {index}"
    info = CameraInfo(index=index, name=name)

    capture = cv2.VideoCapture(index, BACKEND)
    try:
        if not capture.isOpened():
            return info, None
        frame = None
        for _ in range(warmup):
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
        if frame is None:
            return info, None
        info.opened = True
        info.height, info.width = frame.shape[:2]
        info.fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        return info, frame
    finally:
        capture.release()


def discover(max_index=MAX_INDEX, probe_frames=True):
    """List the cameras present. Probing opens each one, which takes a second or two."""
    names = device_names()
    limit = len(names) if names else max_index

    found = []
    for index in range(limit):
        if probe_frames:
            info, _ = probe(index)
        else:
            name = names[index] if index < len(names) else f"OpenCV camera {index}"
            info = CameraInfo(index=index, name=name)
        found.append(info)
    return found


def resolve(spec):
    """Map a camera spec to a current OpenCV index.

    `spec` is either an index, or a name (or unique substring of one). Names are
    matched case-insensitively so a config can say "icspring" and keep working
    after Windows renumbers the devices.
    """
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return int(spec)

    needle = str(spec).casefold()
    names = device_names()
    matches = [i for i, name in enumerate(names) if needle in name.casefold()]
    if not matches:
        available = ", ".join(f"[{i}] {n}" for i, n in enumerate(names)) or "none"
        raise LookupError(f"No camera matching {spec!r}. Available: {available}")
    if len(matches) > 1:
        listed = ", ".join(f"[{i}] {names[i]}" for i in matches)
        raise LookupError(f"{spec!r} matches more than one camera: {listed}")
    return matches[0]


def open_camera(spec, width=None, height=None, fps=None, fourcc=None):
    """Open a camera by name or index, optionally requesting a stream format.

    FOURCC is set before the frame size: DirectShow picks the stream format first,
    and a camera that offers MJPG only at some resolutions will otherwise stay on
    the uncompressed default. MJPG matters because YUY2 saturates USB 2.0 bandwidth
    long before the camera runs out of frame rate.

    A UVC camera may silently ignore any of this, so check `actual_format()` rather
    than trusting the request.
    """
    index = resolve(spec)
    capture = cv2.VideoCapture(index, BACKEND)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {spec!r} (index {index})")
    if fourcc:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def actual_format(capture):
    """What the camera settled on: (fourcc, width, height, fps)."""
    raw = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4)).strip()
    return (
        fourcc,
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        capture.get(cv2.CAP_PROP_FPS) or 0.0,
    )


def backend_name():
    return "DirectShow" if IS_WINDOWS else f"OpenCV default ({platform.system()})"
