"""Minimal Feetech STS3215 bus helper (SO-101 servos)."""
import time
import serial

PING, READ, WRITE = 0x01, 0x02, 0x03

# Control table (STS/SMS series)
TORQUE_ENABLE = 40
GOAL_POSITION = 42
GOAL_SPEED = 46
PRESENT_POSITION = 56
PRESENT_SPEED = 58
PRESENT_LOAD = 60
PRESENT_VOLTAGE = 62
PRESENT_TEMPERATURE = 63
MIN_ANGLE_LIMIT = 9
MAX_ANGLE_LIMIT = 11

JOINT_NAMES = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}
TICKS_PER_DEG = 4096 / 360


def _checksum(body):
    return (~sum(body)) & 0xFF


def _packet(sid, inst, params=()):
    body = [sid, len(params) + 2, inst, *params]
    return bytes([0xFF, 0xFF, *body, _checksum(body)])


class Bus:
    def __init__(self, port, baudrate=1_000_000, timeout=0.05):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.port = port

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _txrx(self, sid, inst, params=()):
        self.ser.reset_input_buffer()
        self.ser.write(_packet(sid, inst, params))
        head = self.ser.read(4)
        if len(head) < 4 or head[:2] != b"\xff\xff":
            return None
        rest = self.ser.read(head[3])
        if len(rest) < head[3]:
            return None
        return rest[1:-1]

    def ping(self, sid, attempts=3):
        """Retry: a single garbled status packet is not proof the servo is absent."""
        return any(self._txrx(sid, PING) is not None for _ in range(attempts))

    def read(self, sid, addr, size):
        data = self._txrx(sid, READ, (addr, size))
        if data is None or len(data) < size:
            return None
        return int.from_bytes(data[:size], "little")

    def read_load(self, sid):
        """Present_Load is an 11-bit field: bit 10 is direction, bits 0-9 the magnitude."""
        raw = self.read(sid, PRESENT_LOAD, 2)
        if raw is None:
            return None
        magnitude = raw & 0x3FF
        return -magnitude if raw & 0x400 else magnitude

    def write(self, sid, addr, value, size=1):
        return self._txrx(sid, WRITE, (addr, *value.to_bytes(size, "little")))

    def torque(self, sid, on):
        return self.write(sid, TORQUE_ENABLE, 1 if on else 0)

    def move_to(self, sid, ticks, speed=400):
        self.write(sid, GOAL_SPEED, speed, size=2)
        return self.write(sid, GOAL_POSITION, int(ticks), size=2)

    def wait_until_reached(self, sid, target, tol=8, timeout=3.0):
        end = time.time() + timeout
        while time.time() < end:
            pos = self.read(sid, PRESENT_POSITION, 2)
            if pos is not None and abs(pos - target) <= tol:
                return True
            time.sleep(0.02)
        return False
