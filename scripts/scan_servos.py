"""Ping every servo on each bus and report its present position.

    uv run scripts/scan_servos.py            # auto-detect ports
    uv run scripts/scan_servos.py COM4
"""

from so101.platform import require_windows

require_windows()

import sys

from so101.hardware import Bus, JOINT_NAMES, PRESENT_POSITION, resolve

BAUD_RATES = (1_000_000, 500_000, 115_200)
SCAN_IDS = range(1, 13)


def scan(port, baudrate):
    with Bus(port, baudrate=baudrate) as bus:
        return [(sid, bus.read(sid, PRESENT_POSITION, 2))
                for sid in SCAN_IDS if bus.ping(sid)]


for port in resolve(sys.argv[1:]):
    for baudrate in BAUD_RATES:
        try:
            found = scan(port, baudrate)
        except Exception as e:  # noqa: BLE001 - report and move to the next port
            print(f"{port}: could not open ({e})")
            break
        if found:
            print(f"{port} @ {baudrate} baud -> {len(found)} servo(s)")
            for sid, pos in found:
                name = JOINT_NAMES.get(sid, "?")
                print(f"   ID {sid:2d} {name:<14} present_position={pos}")
            break
    else:
        print(f"{port}: no servo responded at any baud rate")
