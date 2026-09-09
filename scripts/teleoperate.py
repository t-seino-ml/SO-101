"""LeRobot teleoperation with the serial bus hardened and the follower P gain raised.

No cameras and no viewer - see `teleop_view.py` for that. The stock
`lerobot-teleoperate` calls `sync_read` with no retries inside its control loop, so
it dies on the first garbled status packet; see so101.hardware.bus_patch.

    uv run scripts/teleoperate.py --robot.type=so101_follower --robot.port=COM4 \
        --robot.id=follower --teleop.type=so101_leader --teleop.port=COM3 \
        --teleop.id=leader --fps=120
"""

from so101.platform import require_windows

require_windows()

from so101.hardware import bus_patch, tuning

if __name__ == "__main__":
    tuning.install()

    from lerobot.scripts.lerobot_teleoperate import main

    print(f"[teleoperate.py] serial retries={bus_patch.NUM_RETRY}, "
          f"P_Coefficient={tuning.p_coefficient()}")
    main()
