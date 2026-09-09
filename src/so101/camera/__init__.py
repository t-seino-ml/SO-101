"""Camera discovery, calibration and capture. Phase 2 - not implemented yet.

Intended scope:

- enumerate the attached cameras and keep a stable identity for each one, the way
  `hardware.ports` does for serial ports, so a config does not break when Windows
  renumbers devices
- capture at a fixed resolution and frame rate alongside the 120 Hz control loop,
  without stalling it
- expose frames to `dataset` for recording and to `policy` for inference

LeRobot already ships `lerobot-find-cameras`; start there rather than writing
enumeration from scratch. See docs/02-camera.md.
"""
