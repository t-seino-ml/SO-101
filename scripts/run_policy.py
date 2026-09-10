"""Run the trained grasp policy on the arm.

The policy sees the wrist camera and the joint state, and outputs joint commands.
It knows nothing about colours or coordinates: it grasps whatever is in front of
the gripper. Put the gripper a few centimetres from a block before starting, the
way the demonstrations did.

Safety, since nobody is holding the leader:

- every command is clamped to a maximum step, so a bad prediction cannot fling
  the arm across the table
- load and temperature are watched, and the run stops if a joint starts straining
- ctrl-C, the step limit, or a strained joint all end with the arm relaxed

    uv run scripts/run_policy.py
    uv run scripts/run_policy.py --seconds 20 --max-step 3
    uv run scripts/run_policy.py --checkpoint runs/policy/grasp/checkpoints/020000
"""

import argparse
import time
from pathlib import Path

from so101.platform import require_windows

require_windows()

import numpy as np  # noqa: E402

from so101.camera import load as load_cameras  # noqa: E402
from so101.camera import to_lerobot  # noqa: E402
from so101.hardware import bus_patch  # noqa: F401,E402
from so101.hardware import resolve as resolve_port  # noqa: E402
from so101.hardware import tuning  # noqa: E402

DEFAULT_POLICY = Path("runs/policy/grasp/checkpoints")
MAX_STEP_DEG = 4.0        # per control step, per joint
LOAD_ABORT = 700
TEMP_ABORT = 60
FPS = 30


def newest_checkpoint(directory):
    """The checkpoint the last training run wrote, via its pointer file."""
    directory = Path(directory)
    pointer = directory / "last.txt"
    if pointer.is_file():
        candidate = directory / pointer.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return candidate
    numbered = sorted(p for p in directory.glob("[0-9]*") if p.is_dir())
    if not numbered:
        raise SystemExit(f"No checkpoints in {directory}. Train the policy first.")
    return numbered[-1]


def joints_of(robot):
    return {key.removesuffix(".pos"): float(value)
            for key, value in robot.get_observation().items() if key.endswith(".pos")}


def strained(robot):
    for name in robot.bus.motors:
        try:
            load = robot.bus.read("Present_Load", name, normalize=False)
            temperature = robot.bus.read("Present_Temperature", name, normalize=False)
        except Exception:  # noqa: BLE001 - a dropped packet is not a fault
            continue
        if abs(load) > LOAD_ABORT:
            return f"{name} load {load}"
        if temperature > TEMP_ABORT:
            return f"{name} at {temperature}C"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a specific checkpoint; default is the newest")
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--max-step", type=float, default=MAX_STEP_DEG,
                        help="largest joint move per control step, in degrees")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint or newest_checkpoint(args.policy_dir)
    pretrained = checkpoint / "pretrained_model"
    if not pretrained.is_dir():
        pretrained = checkpoint
    print(f"  policy: {pretrained}")

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    tuning.install()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PreTrainedConfig.from_pretrained(str(pretrained))
    config.pretrained_path = str(pretrained)
    config.device = device
    policy = get_policy_class(config.type).from_pretrained(str(pretrained))
    policy.to(device)
    policy.eval()
    policy.reset()
    print(f"  running on {device}, inputs {list(config.input_features)}")

    specs = load_cameras()
    needed = {key.removeprefix("observation.images.")
              for key in config.input_features
              if key.startswith("observation.images.")}
    specs = {role: spec for role, spec in specs.items() if role in needed}
    missing = needed - set(specs)
    if missing:
        raise SystemExit(f"The policy needs camera(s) {missing}, "
                         "which are not in cameras.json")
    print(f"  cameras: {', '.join(specs)}")

    robot = make_robot_from_config(SO101FollowerConfig(
        port=resolve_port([args.follower_port])[0],
        id="follower",
        cameras=to_lerobot(specs),
    ))
    robot.connect()
    home = joints_of(robot)
    print(f"\n  Running for {args.seconds:.0f}s at {args.fps} Hz, "
          f"steps capped at {args.max_step:.0f} deg. Ctrl-C to stop.\n")

    period = 1.0 / args.fps
    steps = 0
    reason = "finished"
    try:
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            started = time.perf_counter()
            observation = robot.get_observation()
            batch = {}
            for key in config.input_features:
                value = observation[key.removeprefix("observation.")] \
                    if key.removeprefix("observation.") in observation \
                    else observation.get(key)
                batch[key] = value

            with torch.inference_mode():
                action = policy.select_action({
                    key: torch.as_tensor(np.asarray(value)).to(device)
                    for key, value in batch.items() if value is not None
                })
            wanted = action.squeeze(0).cpu().numpy()

            current = joints_of(robot)
            names = list(current)
            capped = {}
            for name, target in zip(names, wanted):
                delta = float(target) - current[name]
                delta = max(-args.max_step, min(args.max_step, delta))
                capped[f"{name}.pos"] = current[name] + delta
            robot.send_action(capped)
            steps += 1

            hurt = strained(robot)
            if hurt:
                reason = f"stopped: {hurt}"
                break
            time.sleep(max(0.0, period - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        print(f"\n  {reason} after {steps} steps")
        try:
            print("  returning to where it started")
            start = joints_of(robot)
            for step in range(1, 41):
                robot.send_action({
                    f"{name}.pos": start[name] + (home[name] - start[name]) * step / 40
                    for name in home})
                time.sleep(0.03)
            time.sleep(0.5)
        except Exception as error:  # noqa: BLE001 - always still relax
            print(f"  could not return ({error})")
        finally:
            robot.disconnect()
            print("  arm relaxed")


if __name__ == "__main__":
    main()
