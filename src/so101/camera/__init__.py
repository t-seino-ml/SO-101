"""Camera discovery, capture and role configuration.

Cameras are identified by DirectShow device name rather than by OpenCV index,
because Windows reorders indices - the same problem `hardware.ports` solves for
serial ports. Capture runs on a background thread per camera and publishes only
the newest frame, so the 120 Hz control loop never waits on a 20 fps camera.
"""

from .capture import CameraSet, CameraStream, Frame
from .config import CameraSpec, load, to_lerobot
from .discovery import (
    CameraInfo,
    actual_format,
    device_names,
    discover,
    open_camera,
    probe,
    resolve,
)

__all__ = [
    "CameraInfo", "CameraSet", "CameraSpec", "CameraStream", "Frame",
    "actual_format", "device_names", "discover", "load", "open_camera", "probe",
    "resolve", "to_lerobot",
]
