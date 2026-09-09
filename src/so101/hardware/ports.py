"""Serial port discovery for the SO-101 servo buses.

The SO-101 control boards use a WCH CH34x USB-serial bridge, so ports from that
vendor are preferred. COM numbers are assigned by the OS and change when the
adapters move to different USB sockets, which is why nothing here hardcodes them.
"""

from serial.tools import list_ports

WCH_VID = 0x1A86


def candidates():
    """Serial ports that look like a servo bus, best guess first."""
    found = [p for p in list_ports.comports() if p.vid == WCH_VID]
    if not found:
        found = list(list_ports.comports())
    return [p.device for p in sorted(found, key=lambda p: p.device)]


def resolve(requested):
    """Use the ports given on the command line, else everything auto-detected."""
    if requested:
        return list(requested)
    found = candidates()
    if not found:
        raise SystemExit(
            "No serial ports found. Check that both arms are powered and plugged in."
        )
    return found


def describe():
    for port in list_ports.comports():
        vid = f"{port.vid:04X}" if port.vid else "----"
        pid = f"{port.pid:04X}" if port.pid else "----"
        print(f"{port.device:<8} {vid}:{pid}  {port.serial_number or '-':<14} "
              f"{port.description}")


if __name__ == "__main__":
    describe()
