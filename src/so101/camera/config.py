"""Which camera plays which role on this rig.

Serial ports are auto-detected because any SO-101 bus will do, but a camera's role
is physical: which one looks down at the workspace, which one rides on the wrist.
That mapping cannot be discovered, so it lives in `cameras.json` at the repo root
and has to be edited when the rig changes.

Cameras are named, not indexed. `resolve()` turns a name into whatever index
DirectShow is currently using, so replugging a camera does not silently swap the
overhead and wrist views in a recorded dataset.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .discovery import IS_WINDOWS, resolve

CONFIG_FPATH = Path(__file__).resolve().parents[3] / "cameras.json"


@dataclass
class CameraSpec:
    """One camera's role on this rig."""

    name: str  # a DirectShow device name, or a unique substring of one
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    fourcc: str | None = None
    # log2 seconds, DirectShow convention: -8 is 1/256 s. "auto" calibrates once
    # at startup and then locks, which is what a rig that moves between tables
    # wants. None leaves the camera's own auto-exposure running.
    exposure: float | str | None = None
    auto_wb: bool | None = None
    wb_temperature: int | None = None

    def index(self):
        """The OpenCV index this camera currently sits at."""
        return resolve(self.name)


def load(fpath=None):
    """Read the role -> CameraSpec mapping. Empty if the file does not exist."""
    fpath = Path(fpath) if fpath else CONFIG_FPATH
    if not fpath.is_file():
        return {}
    raw = json.loads(fpath.read_text(encoding="utf-8"))
    return {role: CameraSpec(**spec) for role, spec in raw.items()}


def save(specs, fpath=None):
    fpath = Path(fpath) if fpath else CONFIG_FPATH
    payload = {role: {k: v for k, v in asdict(spec).items() if v is not None}
               for role, spec in specs.items()}
    fpath.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return fpath


def to_lerobot(specs=None):
    """Build the `cameras` dict that LeRobot's robot configs expect.

    Names are resolved to indices here, at build time, so a stale index never ends
    up baked into a saved config.
    """
    from lerobot.cameras.opencv.configuration_opencv import Cv2Backends, OpenCVCameraConfig

    specs = load() if specs is None else specs
    backend = Cv2Backends.DSHOW if IS_WINDOWS else Cv2Backends.ANY
    return {
        role: OpenCVCameraConfig(
            index_or_path=spec.index(),
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
            fourcc=spec.fourcc,
            backend=backend,
        )
        for role, spec in specs.items()
    }
