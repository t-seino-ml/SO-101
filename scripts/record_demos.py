"""Record teleoperated demonstrations of grasping a block.

The policy trained on these learns one skill: grasp the block in front of the
gripper. It is not told which colour, and it is given no coordinates - the wrist
camera shows it what is there and it closes the loop visually. Colour selection
and coarse positioning happen outside the policy, from the side camera.

That is why every episode should start with the gripper already near a block, a
few centimetres out, the way the coarse move will leave it at run time. Episodes
that start from across the table teach the policy to search, which is not its job.

Recording uses the same hardened bus as everything else: `lerobot-record` polls
`sync_read` with no retries, so one garbled packet ends a session.

    uv run scripts/record_demos.py --episodes 30
    uv run scripts/record_demos.py --episodes 10 --resume     # add to the set
    uv run scripts/record_demos.py --episode-seconds 25
"""

import argparse
from pathlib import Path

from so101.platform import require_windows

require_windows()

from so101.camera import load as load_cameras  # noqa: E402
from so101.camera import to_lerobot  # noqa: E402
from so101.hardware import bus_patch  # noqa: F401,E402
from so101.hardware import resolve as resolve_port  # noqa: E402
from so101.hardware import tuning  # noqa: E402

DEFAULT_ROOT = Path("data/demos")
DEFAULT_REPO = "so101/grasp_block"
TASK = "grasp the block in front of the gripper and drop it in the can"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--episode-seconds", type=float, default=30.0)
    parser.add_argument("--reset-seconds", type=float, default=15.0,
                        help="pause between episodes, to put a block back")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--leader-port", default="COM3")
    parser.add_argument("--cameras", default=None,
                        help="comma-separated roles (default: all in cameras.json)")
    parser.add_argument("--resume", action="store_true",
                        help="append to an existing dataset")
    parser.add_argument("--display", action="store_true",
                        help="open the Rerun viewer while recording")
    args = parser.parse_args()

    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.scripts.lerobot_record import (
        DatasetRecordConfig,
        RecordConfig,
        record,
    )
    from lerobot.teleoperators.so_leader import SO101LeaderConfig

    tuning.install()

    specs = load_cameras()
    if args.cameras:
        wanted = [role.strip() for role in args.cameras.split(",")]
        missing = [role for role in wanted if role not in specs]
        if missing:
            raise SystemExit(f"Not in cameras.json: {', '.join(missing)}")
        specs = {role: specs[role] for role in wanted}
    cameras = to_lerobot(specs)
    for role, spec in specs.items():
        print(f"  {role}: {spec.name!r} -> index {cameras[role].index_or_path}, "
              f"{spec.width}x{spec.height}")

    config = RecordConfig(
        robot=SO101FollowerConfig(
            port=resolve_port([args.follower_port])[0],
            id="follower",
            cameras=cameras,
        ),
        teleop=SO101LeaderConfig(
            port=resolve_port([args.leader_port])[0],
            id="leader",
        ),
        dataset=DatasetRecordConfig(
            repo_id=args.repo_id,
            single_task=args.task,
            root=args.root,
            fps=args.fps,
            episode_time_s=args.episode_seconds,
            reset_time_s=args.reset_seconds,
            num_episodes=args.episodes,
            video=True,
            # Nothing leaves this machine unless someone asks for it.
            push_to_hub=False,
        ),
        display_data=args.display,
        resume=args.resume,
    )

    print(f"\n  {args.episodes} episodes of {args.episode_seconds:.0f}s, "
          f"{args.reset_seconds:.0f}s between them, at {args.fps} fps")
    print(f"  saving to {args.root}")
    print()
    print("  Each episode: start with the gripper a few centimetres from a block,")
    print("  drive it in with the leader, close on the block, lift, drop it in the")
    print("  can. Vary which block and where it sits between episodes.")
    print()
    print("  Right arrow ends an episode early, left arrow re-records the last one,")
    print("  escape stops the session.")
    print()

    record(config)

    print(f"\n  Recorded to {args.root}")
    print("  Review it:  uv run .venv/Scripts/lerobot-dataset-viz.exe "
          f"--repo-id {args.repo_id} --root {args.root}")


if __name__ == "__main__":
    main()
