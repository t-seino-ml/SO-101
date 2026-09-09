"""SO-101 calibration without the "move it to the middle first" step.

LeRobot's own calibration pins whatever pose you start from to encoder value 2047
and only then records the range. Start a joint near a mechanical stop and the far
end of its travel lands outside 0-4095, which fails at the very end and throws away
the whole recording.

This does the same job in the opposite order: record every joint's travel first,
then derive the homing offset from the midpoint of what was actually recorded. Any
starting pose works, and the resulting range is centred on 2047 by construction.

The output is a normal LeRobot calibration file, written to the same path and in the
same format as `lerobot-calibrate`, plus the same values written to the servos.

    uv run scripts/easy_calibrate.py --robot.type=so101_follower --robot.port=COM4 --robot.id=follower
    uv run scripts/easy_calibrate.py --teleop.type=so101_leader  --teleop.port=COM3 --teleop.id=leader
"""

from so101.platform import require_windows

require_windows()

import json
from pathlib import Path

from so101.hardware import bus_patch  # noqa: F401  - installs the serial retry patches
import draccus
from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots import make_robot_from_config
from lerobot.robots.config import RobotConfig
from lerobot.scripts.lerobot_calibrate import CalibrateConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.utils import enter_pressed, init_logging, move_cursor_up

# wrist_roll turns continuously, so it gets the full encoder range instead of a
# recorded one - same convention as LeRobot's own calibration.
FULL_TURN_MOTOR = "wrist_roll"
MIN_TRAVEL = 100  # ticks; below this a joint was clearly never moved
CLEAR_EOL = "\x1b[K"  # a redraw that is shorter than the previous line leaves debris
RECORDING_FPATH = Path(__file__).resolve().parents[1] / "last_recording.json"  # repo root


def record_travel(bus, motor_names):
    positions = bus.sync_read("Present_Position", motor_names, normalize=False)
    mins = dict(positions)
    maxes = dict(positions)

    print(f"Move every joint except '{FULL_TURN_MOTOR}' to both of its end stops.")
    print("Order does not matter, and the starting pose does not matter.")
    print("Press ENTER when every joint shows a TRAVEL of a few hundred ticks...")

    while True:
        positions = bus.sync_read("Present_Position", motor_names, normalize=False)
        mins = {m: min(positions[m], v) for m, v in mins.items()}
        maxes = {m: max(positions[m], v) for m, v in maxes.items()}

        print(CLEAR_EOL)
        print("-" * 50 + CLEAR_EOL)
        print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6} | {'TRAVEL':>6}"
              + CLEAR_EOL)
        for motor in motor_names:
            travel = maxes[motor] - mins[motor]
            flag = "" if travel >= MIN_TRAVEL else "  <- not moved yet"
            print(f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | "
                  f"{maxes[motor]:>6} | {travel:>6}{flag}" + CLEAR_EOL)

        if enter_pressed():
            return mins, maxes

        move_cursor_up(len(motor_names) + 3)


def build_calibration(bus, mins, maxes, positions):
    calibration = {}
    for motor, motor_def in bus.motors.items():
        resolution = bus.model_resolution_table[motor_def.model]
        centre = (resolution - 1) // 2

        if motor == FULL_TURN_MOTOR:
            # Centre on wherever it sits now; the range is the whole turn.
            offset = positions[motor] - centre
            range_min, range_max = 0, resolution - 1
        else:
            # Present_Position = Actual_Position - Homing_Offset, so putting the
            # midpoint of the recorded travel at `centre` keeps both ends in range.
            offset = (mins[motor] + maxes[motor]) // 2 - centre
            range_min = mins[motor] - offset
            range_max = maxes[motor] - offset

        calibration[motor] = MotorCalibration(
            id=motor_def.id,
            drive_mode=0,
            homing_offset=offset,
            range_min=range_min,
            range_max=range_max,
        )
    return calibration


@draccus.wrap()
def run(cfg: CalibrateConfig):
    init_logging()
    device = (make_robot_from_config(cfg.device) if isinstance(cfg.device, RobotConfig)
              else make_teleoperator_from_config(cfg.device))
    device.connect(calibrate=False)
    try:
        bus = device.bus
        bus.disable_torque()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # Clear stored offsets and limits so Present_Position reads raw encoder counts.
        bus.reset_calibration()

        motor_names = [m for m in bus.motors if m != FULL_TURN_MOTOR]
        mins, maxes = record_travel(bus, motor_names)

        # Keep the recording on disk before anything else gets a chance to fail.
        RECORDING_FPATH.write_text(json.dumps({"mins": mins, "maxes": maxes}, indent=2))
        print(f"\nRaw travel recorded to {RECORDING_FPATH}")

        not_moved = [m for m in motor_names if maxes[m] - mins[m] < MIN_TRAVEL]
        if not_moved:
            raise SystemExit(
                f"\nThese joints were not moved far enough: {', '.join(not_moved)}\n"
                "Nothing was written. Re-run and move each of them to both end stops."
            )

        # Read every motor, FULL_TURN_MOTOR included - it is absent from motor_names.
        positions = bus.sync_read("Present_Position", normalize=False)
        calibration = build_calibration(bus, mins, maxes, positions)

        print("\nResulting calibration:")
        print(f"{'NAME':<15} | {'HOMING':>7} | {'RANGE_MIN':>9} | {'RANGE_MAX':>9}")
        for motor, cal in calibration.items():
            print(f"{motor:<15} | {cal.homing_offset:>7} | {cal.range_min:>9} | "
                  f"{cal.range_max:>9}")

        device.calibration = calibration
        bus.write_calibration(calibration)
        device._save_calibration()
        print(f"\nCalibration saved to {device.calibration_fpath}")
    finally:
        device.disconnect()


if __name__ == "__main__":
    run()
