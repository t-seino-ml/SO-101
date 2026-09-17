"""The exhibition stack: taught trajectories, replayed exactly.

Kept apart from `so101.policy` on purpose. That package is where the research
lives - inverse kinematics, the TCP, ray-plane geometry, the pick planner - and
all of it is mid-investigation. An exhibition in front of school students is not
the place to find out what a solver does at a pose nobody tried.

So this asks vision only what vision is good at - which colour is where - and
takes every joint angle from a trajectory a person taught and watched succeed on
the actual arm. Nothing here solves for a pose. If the robot, the slots and the
can are put back where they were, the same numbers do the same thing.

- `trajectory`  waypoints, saved and replayed, with the arm watched while it moves
"""

from .trajectory import Trajectory, Waypoint, play

__all__ = ["Trajectory", "Waypoint", "play"]
