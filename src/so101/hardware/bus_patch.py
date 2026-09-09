"""Hardening for LeRobot's Feetech bus over a USB-serial adapter on Windows.

Import this before running any LeRobot entry point that talks to the servos.

1. Serial retries. LeRobot calls `sync_read` with num_retry=0 in its hot paths -
   the calibration recording loop and the 60 Hz teleop loop - so a single corrupted
   status packet raises ConnectionError and kills the run. A Feetech sync read has
   every servo replying back to back, which a USB-serial adapter garbles now and
   then. Every read and write here retries instead of giving up.

2. Range clamping during calibration. `set_half_turn_homings` pins the starting pose
   to encoder 2047 before the range is known, so a joint that starts near a
   mechanical stop records a range outside 0-4095 and fails at the very end.
   Clamping keeps the recording rather than throwing it away. `easy_calibrate.py`
   avoids the problem outright by recording first and centring afterwards.
"""

import os

from lerobot.motors.feetech import feetech
from lerobot.motors.motors_bus import MotorsBus

# Each retry waits out a full serial timeout, so this trades robustness against
# stall length: one dropped packet costs NUM_RETRY x timeout of frozen control.
# Three is enough to survive the occasional garbled status packet without a stall
# long enough to feel.
NUM_RETRY = int(os.environ.get("SO101_SERIAL_RETRIES", "3"))

# LeRobot patches the SDK's setPacketTimeout to add a flat 50 ms margin, so every
# dropped packet freezes control for ~52 ms - long enough to feel as a hitch while
# teleoperating. At 1 Mbaud a six-servo sync read replies in about 0.5 ms, so 50 ms
# is a hundredfold margin. 5 ms still covers servo processing and scheduler jitter.
PACKET_TIMEOUT_MARGIN_MS = float(os.environ.get("SO101_PACKET_TIMEOUT_MS", "5"))
ENCODER_MIN, ENCODER_MAX = 0, 4095

_ping = MotorsBus.ping
_record_ranges = MotorsBus.record_ranges_of_motion
_sync_read = MotorsBus._sync_read
_read = MotorsBus._read
_write = MotorsBus._write


def _sync_read_retrying(self, addr, length, motor_ids, *, num_retry=0, **kwargs):
    return _sync_read(self, addr, length, motor_ids,
                      num_retry=max(num_retry, NUM_RETRY), **kwargs)


def _read_retrying(self, addr, length, motor_id, *, num_retry=0, **kwargs):
    return _read(self, addr, length, motor_id,
                 num_retry=max(num_retry, NUM_RETRY), **kwargs)


def _write_retrying(self, addr, length, motor_id, value, *, num_retry=0, **kwargs):
    return _write(self, addr, length, motor_id, value,
                  num_retry=max(num_retry, NUM_RETRY), **kwargs)


def _ping_retrying(self, motor, num_retry=0, raise_on_error=False):
    """Retry the connect-time roll call.

    `_assert_motors_exist` pings each motor once with no retries and refuses to
    connect if any is missing. One dropped packet out of six therefore aborts the
    whole connection, reporting a servo as absent when it answers perfectly well a
    moment later - which is exactly what happened to shoulder_lift here.
    """
    return _ping(self, motor, num_retry=max(num_retry, NUM_RETRY),
                 raise_on_error=raise_on_error)


def _record_ranges_clamped(self, motors=None, display_values=True):
    mins, maxes = _record_ranges(self, motors, display_values)
    clamped = []
    for edge, values in (("min", mins), ("max", maxes)):
        for motor, value in values.items():
            fixed = max(ENCODER_MIN, min(ENCODER_MAX, value))
            if fixed != value:
                clamped.append(f"{motor} {edge}: {value} -> {fixed}")
                values[motor] = fixed
    if clamped:
        print()
        print("*** WARNING: recorded range fell outside 0-4095 and was clamped ***")
        for line in clamped:
            print(f"    {line}")
        print("    The starting pose was not centred for these joints.")
        print("    Calibration is saved, but redo it from a true mid-range pose.")
        print()
    return mins, maxes


def _set_packet_timeout(self, packet_length):
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * (packet_length + 3.0)
                           + PACKET_TIMEOUT_MARGIN_MS)


# Replace the module-level function; FeetechMotorsBus binds it per PortHandler in
# its constructor, so this has to happen before any bus is created.
feetech.patch_setPacketTimeout = _set_packet_timeout

MotorsBus.ping = _ping_retrying
MotorsBus.record_ranges_of_motion = _record_ranges_clamped
MotorsBus._sync_read = _sync_read_retrying
MotorsBus._read = _read_retrying
MotorsBus._write = _write_retrying
