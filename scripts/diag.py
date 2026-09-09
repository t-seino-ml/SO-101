"""Read-only health report for every servo on both SO-101 arms."""
import sys
from so101.hardware import (
    Bus, JOINT_NAMES, TICKS_PER_DEG,
    PRESENT_POSITION, PRESENT_VOLTAGE, PRESENT_TEMPERATURE,
    TORQUE_ENABLE, MIN_ANGLE_LIMIT, MAX_ANGLE_LIMIT, resolve,
)

for port in resolve(sys.argv[1:]):
    print(f"\n=== {port} ===")
    print(f"{'ID':>2} {'joint':<14} {'pos':>6} {'deg':>7} {'limits':>12} "
          f"{'torque':>6} {'load':>6} {'volt':>5} {'temp':>5}")
    with Bus(port) as bus:
        for sid, name in JOINT_NAMES.items():
            if not bus.ping(sid):
                print(f"{sid:>2} {name:<14} NO RESPONSE")
                continue
            pos = bus.read(sid, PRESENT_POSITION, 2)
            lo = bus.read(sid, MIN_ANGLE_LIMIT, 2)
            hi = bus.read(sid, MAX_ANGLE_LIMIT, 2)
            torque = bus.read(sid, TORQUE_ENABLE, 1)
            load = bus.read_load(sid)
            volt = bus.read(sid, PRESENT_VOLTAGE, 1)
            temp = bus.read(sid, PRESENT_TEMPERATURE, 1)
            print(f"{sid:>2} {name:<14} {pos:>6} {pos / TICKS_PER_DEG:>6.1f}d "
                  f"{f'{lo}-{hi}':>12} {torque:>6} {load:>6} "
                  f"{volt / 10:>4.1f}V {temp:>4}C")
