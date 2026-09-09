"""List the serial ports that look like an SO-101 servo bus.

    uv run scripts/ports.py
"""

from so101.platform import require_windows

require_windows()

from so101.hardware.ports import describe

if __name__ == "__main__":
    describe()
