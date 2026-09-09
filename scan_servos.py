"""Ping every servo on each bus and report its present position.

    python scan_servos.py            # auto-detect ports
    python scan_servos.py COM4
"""

import sys

from ports import resolve
from sts3215 import Bus, JOINT_NAMES, PRESENT_POSITION

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
