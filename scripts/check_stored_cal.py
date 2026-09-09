"""Show, snapshot and diff the calibration stored inside the servos.

LeRobot's `set_half_turn_homings` calls `reset_calibration()`, which wipes the
Min/Max_Position_Limit and Homing_Offset held in servo EEPROM. A failed calibration
therefore leaves an arm *less* calibrated than before. Take a snapshot first.

    uv run scripts/check_stored_cal.py save     # write servo_baseline.json
    uv run scripts/check_stored_cal.py          # show current values, diffed against it
"""

import json
import sys
from pathlib import Path

from so101.hardware import (
    Bus,
    JOINT_NAMES,
    MAX_ANGLE_LIMIT,
    MIN_ANGLE_LIMIT,
    resolve,
)

HOMING_OFFSET = 31
ENCODER_MAX = 4095
BASELINE_FPATH = Path(__file__).resolve().parents[1] / "servo_baseline.json"  # repo root


def signed11(raw):
    """Feetech Homing_Offset: bit 11 is the sign, bits 0-10 the magnitude."""
    if raw is None:
        return None
    return -(raw & 0x7FF) if raw & 0x800 else raw


def read_state(port):
    state = {}
    with Bus(port) as bus:
        for sid in JOINT_NAMES:
            if not bus.ping(sid):
                continue
            state[str(sid)] = {
                "min": bus.read(sid, MIN_ANGLE_LIMIT, 2),
                "max": bus.read(sid, MAX_ANGLE_LIMIT, 2),
                "homing": signed11(bus.read(sid, HOMING_OFFSET, 2)),
            }
    return state


args = [a for a in sys.argv[1:] if a != "save"]
saving = "save" in sys.argv[1:]
ports = resolve(args)
current = {port: read_state(port) for port in ports}

if saving:
    BASELINE_FPATH.write_text(json.dumps(current, indent=2))
    print(f"Baseline written to {BASELINE_FPATH}")
    raise SystemExit

baseline = json.loads(BASELINE_FPATH.read_text()) if BASELINE_FPATH.is_file() else {}

for port, state in current.items():
    print(f"\n=== {port} ===")
    print(f"{'ID':>2} {'joint':<14} {'min-max':>12} {'homing':>7}  status")
    for sid, values in state.items():
        lo, hi, homing = values["min"], values["max"], values["homing"]
        was = baseline.get(port, {}).get(sid)
        if was is None:
            status = "no baseline"
        elif (was["min"], was["max"]) == (lo, hi):
            status = "unchanged"
        else:
            status = f"CHANGED from {was['min']}-{was['max']}"
        if lo > ENCODER_MAX or hi > ENCODER_MAX or lo >= hi:
            status += " / INVALID"
        name = JOINT_NAMES.get(int(sid), "?")
        print(f"{sid:>2} {name:<14} {f'{lo}-{hi}':>12} {homing:>7}  {status}")

if not baseline:
    print(f"\nNo baseline yet. Run 'uv run scripts/check_stored_cal.py save' before "
          f"calibrating, so a failed run can be spotted.")
