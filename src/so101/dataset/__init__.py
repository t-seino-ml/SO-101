"""Teleoperation recording and LeRobot dataset handling. Phase 3 - not implemented yet.

Intended scope:

- record leader/follower joint states plus camera frames into a LeRobot dataset
- inspect and prune episodes before training
- publish to the Hugging Face Hub if the dataset is worth sharing

LeRobot ships `lerobot-record`, `lerobot-dataset-viz` and `lerobot-edit-dataset`;
this package should wrap them with this rig's ports and calibration rather than
reimplementing them. See docs/03-dataset.md.
"""
