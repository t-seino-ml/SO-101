"""Teleoperation loop that keeps control fast and display cheap.

LeRobot's `teleop_loop` does everything at the control rate: it reads the follower
state, reads the leader, sends the action, and logs the whole observation - camera
images included - to Rerun, once per iteration. Two 800x600 RGB frames is 2.9 MB,
so logging them at 120 Hz asks for 345 MB/s. The loop cannot keep up, and the
leader-to-follower delay grows with it.

Here the two run at different rates:

- control, every iteration: read the leader, send to the follower. Two serial
  round trips, about 2 ms.
- display, a few times per frame at most: read the follower state and the newest
  camera frames, and log them.

`max_relative_target` costs an extra follower `sync_read` per step on top of the
rate limit it imposes, which is why it is off unless asked for.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

# Control costs about 1.2 ms per step, so the loop has room well past 120 Hz.
# 200 Hz halves the command latency and still leaves 3.8 ms of slack per step.
DEFAULT_FPS = 200
DEFAULT_DISPLAY_HZ = 30


@dataclass
class LoopStats:
    """Timing for one teleoperation run, in milliseconds."""

    control: list = field(default_factory=list)
    display: list = field(default_factory=list)
    period: list = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    def summary(self, fps):
        elapsed = time.perf_counter() - self.started
        lines = [f"{len(self.period)} steps in {elapsed:.1f}s "
                 f"({len(self.period) / elapsed:.0f} Hz achieved, {fps} Hz asked)"]
        budget = 1000 / fps
        for label, samples in (("control", self.control), ("display", self.display)):
            if not samples:
                continue
            ordered = sorted(samples)
            lines.append(
                f"  {label:<8} median {statistics.median(ordered):5.2f} ms   "
                f"p95 {ordered[int(len(ordered) * 0.95)]:5.2f} ms   "
                f"max {ordered[-1]:6.2f} ms   ({len(ordered)} calls)")
        if self.period:
            ordered = sorted(self.period)
            lines.append(
                f"  {'period':<8} median {statistics.median(ordered):5.2f} ms   "
                f"p95 {ordered[int(len(ordered) * 0.95)]:5.2f} ms   "
                f"max {ordered[-1]:6.2f} ms   (budget {budget:.2f} ms)")
            for threshold in (20, 50):
                stalls = sum(1 for value in ordered if value > threshold)
                if stalls:
                    lines.append(f"  {stalls} step(s) over {threshold} ms "
                                 f"({100 * stalls / len(ordered):.1f}%)")
        return "\n".join(lines)


def run(robot, teleop, fps=DEFAULT_FPS, display_hz=DEFAULT_DISPLAY_HZ,
        display=False, duration=None, compress_images=False, on_stats=None):
    """Drive `robot` from `teleop` until interrupted or `duration` elapses."""
    from lerobot.utils.robot_utils import precise_sleep

    log_rerun_data = None
    if display:
        from lerobot.utils.visualization_utils import log_rerun_data

    period = 1.0 / fps
    display_period = 1.0 / display_hz if display_hz else None
    stats = LoopStats()
    started = stats.started
    next_display = started

    try:
        while True:
            loop_start = time.perf_counter()

            action = teleop.get_action()
            robot.send_action(action)
            stats.control.append((time.perf_counter() - loop_start) * 1000)

            if display and loop_start >= next_display:
                display_start = time.perf_counter()
                log_rerun_data(observation=robot.get_observation(), action=action,
                               compress_images=compress_images)
                stats.display.append((time.perf_counter() - display_start) * 1000)
                # Schedule from now, not from the deadline, so a slow frame does
                # not queue up a burst of catch-up logging.
                next_display = time.perf_counter() + display_period

            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(period - elapsed, 0.0))
            stats.period.append((time.perf_counter() - loop_start) * 1000)

            if duration is not None and time.perf_counter() - started >= duration:
                break
    except KeyboardInterrupt:
        pass

    if on_stats:
        on_stats(stats)
    return stats


def _torque_is_off(bus, motor):
    """Read Torque_Enable without raising on a non-zero status byte.

    A servo that has just been driven hard answers with an error status - a
    latched overload, say - and LeRobot's read() raises on that, which would
    otherwise look like the read failed when it returned a perfectly good value.
    """
    definition = bus.motors[motor]
    address, length = bus.model_ctrl_table[definition.model]["Torque_Enable"]
    try:
        value, _, _ = bus._read(address, length, definition.id, raise_on_error=False)
    except Exception:  # noqa: BLE001 - a genuine comm failure; caller retries
        return False
    return value == 0


def release_torque(robot):
    """Make sure the follower goes limp even if a clean disconnect failed.

    `Robot.disconnect()` disables torque one motor at a time and gives up on the
    first write that errors, leaving every motor after it still holding. But a
    servo answering with a non-zero status byte has usually still applied the
    write, so neither the write nor a normal read can be trusted here: attempt
    every motor, and confirm by reading the register tolerantly.

    Returns the motors still holding, which should be empty.
    """
    bus = getattr(robot, "bus", None)
    if bus is None:
        return []
    still_holding = []
    for motor in bus.motors:
        for _ in range(5):
            try:
                bus.write("Torque_Enable", motor, 0)
            except Exception:  # noqa: BLE001 - the read-back is the real check
                pass
            if _torque_is_off(bus, motor):
                break
        else:
            still_holding.append(motor)
    return still_holding
