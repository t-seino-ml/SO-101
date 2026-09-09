"""SO-101 imitation-learning stack.

Subpackages, in the order the project builds them up:

- `hardware`  servo bus, port discovery, LeRobot serial patches (working)
- `camera`    camera discovery and capture (planned)
- `dataset`   teleoperation recording and LeRobot dataset handling (planned)
- `policy`    policy training and on-robot inference (planned)
- `ui`        view components for the integrated app (planned)

See docs/ for the phase-by-phase notes.
"""

__version__ = "0.1.0"
