"""Background camera capture that does not stall the control loop.

`cv2.VideoCapture.read()` blocks until the next frame, so calling it from the
control loop would peg that loop at the camera's frame rate - about 30 fps against
a 120 Hz servo loop. Instead a thread per camera keeps reading and publishes only
the newest frame; the control loop takes whatever is current and never waits.

Dropping intermediate frames is the right trade here. A policy or a teleop step
wants the freshest view, not a backlog of stale ones.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .discovery import WARMUP_S, actual_format, open_camera, resolve


@dataclass
class Frame:
    """A captured frame and when it was captured."""

    image: object  # numpy.ndarray, kept loose so this module need not import numpy
    timestamp: float
    index: int

    @property
    def age_ms(self):
        return (time.perf_counter() - self.timestamp) * 1000


class CameraStream:
    """Reads one camera in a background thread and publishes the latest frame."""

    def __init__(self, spec, width=None, height=None, fps=None, fourcc=None,
                 name=None, warmup_s=WARMUP_S):
        self.spec = spec
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.warmup_s = warmup_s
        self.name = name or str(spec)
        self.format = None  # what the camera settled on, filled in by start()
        self.index = None  # resolved in start(), never from the reader thread

        self._capture = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._frames_read = 0
        self._read_failures = 0
        self._started_at = None

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        if self._thread is not None:
            raise RuntimeError(f"{self.name} is already started")
        # Resolve the name here, on the calling thread. Name lookup goes through
        # DirectShow COM, and two reader threads doing that concurrently deadlocks.
        self.index = resolve(self.spec)
        self._capture = open_camera(self.index, self.width, self.height, self.fps,
                                    self.fourcc)
        self.format = actual_format(self._capture)
        self._stop.clear()
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.name}",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- capture -----------------------------------------------------------

    def _run(self):
        index = self.index
        # Publish nothing until auto-exposure has settled, so the first frame a
        # caller sees is usable rather than washed out.
        settled_at = time.perf_counter() + self.warmup_s
        while not self._stop.is_set():
            ok, image = self._capture.read()
            if not ok or image is None:
                self._read_failures += 1
                # Do not spin on a camera that has gone away.
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            if now < settled_at:
                continue
            frame = Frame(image=image, timestamp=now, index=index)
            with self._lock:
                self._frame = frame
                self._frames_read += 1

    def read(self):
        """The most recent frame, or None if none has arrived yet. Never blocks."""
        with self._lock:
            return self._frame

    def wait_for_frame(self, timeout=5.0):
        """Block until the first frame arrives. Use once, at startup."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
            time.sleep(0.01)
        raise TimeoutError(f"{self.name} produced no frame within {timeout}s")

    # -- stats -------------------------------------------------------------

    @property
    def measured_fps(self):
        if not self._started_at or not self._frames_read:
            return 0.0
        return self._frames_read / (time.perf_counter() - self._started_at)

    @property
    def frames_read(self):
        return self._frames_read

    @property
    def read_failures(self):
        return self._read_failures

    def __str__(self):
        return (f"{self.name}: {self._frames_read} frames, "
                f"{self.measured_fps:.1f} fps, {self._read_failures} failures")


class CameraSet:
    """Several CameraStreams started and stopped together."""

    def __init__(self, streams):
        self.streams = dict(streams)

    @classmethod
    def from_specs(cls, specs, width=None, height=None, fps=None, fourcc=None,
                   warmup_s=WARMUP_S):
        """`specs` maps a role ("overhead", "side") to a camera name or index."""
        return cls({role: CameraStream(spec, width, height, fps, fourcc, name=role,
                                       warmup_s=warmup_s)
                    for role, spec in specs.items()})

    def start(self):
        for stream in self.streams.values():
            stream.start()
        return self

    def stop(self):
        for stream in self.streams.values():
            stream.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def read(self):
        """Latest frame per role. A role maps to None until its first frame lands."""
        return {role: stream.read() for role, stream in self.streams.items()}

    def wait_for_frames(self, timeout=5.0):
        return {role: stream.wait_for_frame(timeout)
                for role, stream in self.streams.items()}
