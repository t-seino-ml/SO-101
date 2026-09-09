"""LeRobot's stock calibration, with the serial bus hardened.

See bus_patch.py for what is patched and why. Prefer easy_calibrate.py, which also
removes the "move it to the middle of its range first" requirement.

    uv run scripts/calibrate.py --robot.type=so101_follower --robot.port=COM4 --robot.id=follower
    uv run scripts/calibrate.py --teleop.type=so101_leader  --teleop.port=COM3 --teleop.id=leader
"""

from so101.platform import require_windows

require_windows()

from so101.hardware import bus_patch

if __name__ == "__main__":
    from lerobot.scripts.lerobot_calibrate import main

    print(f"[calibrate.py] serial retries={bus_patch.NUM_RETRY}, range clamped to "
          f"{bus_patch.ENCODER_MIN}-{bus_patch.ENCODER_MAX}")
    main()
