"""Run the trained grasp policy on the arm.

The policy sees the wrist camera and the joint state, and outputs joint commands.
It knows nothing about colours or coordinates: it grasps whatever is in front of
the gripper. Start it from the pose the demonstrations started from - the gripper
a few centimetres from a block - which `--check` will confirm before moving.

Safety, since nobody is holding the leader:

- commands are clamped to the range the demonstrations actually covered, so the
  arm cannot be sent somewhere no demonstration ever went
- how fast a command may change is capped, measured from the demonstrations
- load and temperature are watched, and the run stops if a joint starts straining
- ctrl-C, the time limit, or a strained joint all end with the arm back where it
  started and relaxed

    uv run scripts/run_policy.py --check          # look before moving anything
    uv run scripts/run_policy.py --seconds 15
    uv run scripts/run_policy.py --checkpoint runs/policy/grasp/checkpoints/010000
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
DEFAULT_ROOT = Path("data/demos")
DEFAULT_REPO = "so101/grasp_block"

# A command may move this far in one step. The demonstrations never moved a joint
# by more than 10.5 deg between frames and stayed under 5 deg at the 99th
# percentile, so this bounds a bad prediction without cutting into a good one.
MAX_STEP_DEG = 8.0
# How far outside the range the demonstrations covered a command may still go.
RANGE_PAD_DEG = 5.0
LOAD_ABORT = 700
TEMP_ABORT = 60


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


def joints_of(observation):
    return {key.removesuffix(".pos"): float(value)
            for key, value in observation.items() if key.endswith(".pos")}


def strained(robot):
    """The name of a joint that is straining, or None."""
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


def report_start_pose(current, starts, names):
    """Whether the arm is standing where the demonstrations started.

    ACT has only ever seen the poses in the demonstrations. Started from
    somewhere else it is extrapolating, and what it does then is not predictable
    from the training loss - so it is worth knowing before the arm moves.
    """
    print(f"\n  {'joint':<15}{'now':>8}{'demos started':>18}")
    outside = []
    for index, name in enumerate(names):
        low, high = starts[:, index].min(), starts[:, index].max()
        ok = low - 3 <= current[name] <= high + 3
        if not ok:
            outside.append(name)
        print(f"  {name:<15}{current[name]:>8.1f}"
              f"{f'{low:.0f} .. {high:.0f}':>18}   {'ok' if ok else '<-- OUTSIDE'}")
    return outside


def episode_starts(root):
    """The first joint state of every recorded episode."""
    import pandas as pd

    frames = pd.concat([pd.read_parquet(path)
                        for path in sorted(Path(root, "data").rglob("*.parquet"))])
    first = frames[frames["frame_index"] == 0].sort_values("episode_index")
    return np.stack(first["observation.state"].to_numpy())


def glide_to(robot, goal, seconds=1.5, fps=30):
    """Move to a pose gently, in small steps rather than one jump."""
    start = joints_of(robot.get_observation())
    steps = max(1, int(seconds * fps))
    for step in range(1, steps + 1):
        robot.send_action({f"{name}.pos": start[name]
                           + (goal[name] - start[name]) * step / steps
                           for name in goal})
        time.sleep(1.0 / fps)


class Trace:
    """What the policy asked for and what the arm did, kept for afterwards.

    A run that ends without a grasp looks the same from the terminal whether the
    policy sat still, reached past the block, or closed on nothing. The only way
    to tell is to keep the numbers and the view the policy had while it decided.
    """

    def __init__(self, directory, names, fps):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.names = names
        self.fps = fps
        self.rows = []
        self.writer = None

    def add(self, observation, wanted, sent, frame):
        state = joints_of(observation)
        self.rows.append((
            [state[name] for name in self.names],
            [float(wanted[f"{name}.pos"]) for name in self.names],
            [float(sent[f"{name}.pos"]) for name in self.names],
        ))
        image = next((value for key, value in frame.items()
                      if key.startswith("observation.images.")), None)
        if image is None:
            return
        if self.writer is None:
            import cv2

            height, width = image.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.directory / "wrist.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        self.writer.write(image[:, :, ::-1])   # the frame is RGB, the writer BGR

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if not self.rows:
            return
        state, wanted, sent = (np.array(column) for column in zip(*self.rows))
        np.savez(self.directory / "trace.npz", names=self.names,
                 state=state, wanted=wanted, sent=sent, fps=self.fps)
        print(f"  trace written to {self.directory}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a specific checkpoint; default is the newest")
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=None,
                        help="default: the rate the demonstrations were recorded at")
    parser.add_argument("--max-step", type=float, default=MAX_STEP_DEG,
                        help="largest change in a joint command per step, in degrees")
    parser.add_argument("--check", action="store_true",
                        help="report the pose and what the policy sees, then stop")
    parser.add_argument("--trace", type=Path, default=None,
                        help="write the run's trajectory and wrist video here")
    parser.add_argument("--to-start", action="store_true",
                        help="glide to the median demonstration start pose first")
    parser.add_argument("--action-steps", type=int, default=None,
                        help="actions executed per camera look (default: as trained)")
    parser.add_argument("--ensemble", type=float, default=None,
                        help="temporal ensembling coefficient, e.g. 0.01; "
                             "queries the policy every step and blends the chunks")
    args = parser.parse_args()

    checkpoint = args.checkpoint or newest_checkpoint(args.policy_dir)
    pretrained = checkpoint / "pretrained_model"
    if not pretrained.is_dir():
        pretrained = checkpoint

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.utils import make_robot_action
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device

    tuning.install()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    fps = args.fps or meta.fps

    config = PreTrainedConfig.from_pretrained(str(pretrained))
    config.pretrained_path = str(pretrained)
    config.device = device
    # How many of the predicted actions to execute before looking again. ACT
    # predicts a whole chunk from one camera frame, and trained with the chunk
    # size as the default it runs open loop for the length of that chunk - over
    # three seconds at 30 Hz, during which nothing it sees can change what it
    # does. Lowering this costs one forward pass per step and nothing else; the
    # weights are unchanged.
    if args.ensemble is not None:
        config.temporal_ensemble_coeff = args.ensemble
        config.n_action_steps = 1
    elif args.action_steps:
        config.n_action_steps = args.action_steps
    policy = make_policy(config, ds_meta=meta)
    # ACT does not normalise inside the model any more - that lives in the
    # processor pipeline, built from the same dataset statistics as training.
    # Feeding the policy raw degrees without it produces confident nonsense.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(pretrained),
        dataset_stats=meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    policy.eval()
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    print(f"  policy: {pretrained}")
    print(f"  {device}, inputs {', '.join(config.input_features)}")
    if config.temporal_ensemble_coeff is not None:
        print(f"  looking every step, ensembling with "
              f"coeff {config.temporal_ensemble_coeff}")
    else:
        print(f"  {config.n_action_steps} actions per camera look "
              f"({config.n_action_steps / fps:.2f}s open loop)")

    # Only the features the policy declared, so the frame carries nothing the
    # policy will not look at and no camera is opened to fill it.
    observed = {key: feature for key, feature in meta.features.items()
                if key in config.input_features}
    needed = {key.removeprefix("observation.images.") for key in observed
              if key.startswith("observation.images.")}
    specs = {role: spec for role, spec in load_cameras().items() if role in needed}
    missing = needed - set(specs)
    if missing:
        raise SystemExit(f"The policy needs camera(s) {missing}, "
                         "which are not in cameras.json")

    names = [name.removesuffix(".pos") for name in meta.features["action"]["names"]]
    low = np.array(meta.stats["action"]["min"]) - RANGE_PAD_DEG
    high = np.array(meta.stats["action"]["max"]) + RANGE_PAD_DEG
    starts = episode_starts(args.root)
    task = meta.tasks.index[0]

    robot = make_robot_from_config(SO101FollowerConfig(
        port=resolve_port([args.follower_port])[0],
        id="follower",
        cameras=to_lerobot(specs),
    ))
    robot.connect()

    if args.to_start:
        # Torque comes off at the end of every run, so the arm settles a little
        # lower each time and the next run begins somewhere else. Two runs are
        # only comparable if they begin in the same place.
        median = dict(zip(names, np.median(starts, axis=0)))
        print("  gliding to the median demonstration start pose")
        glide_to(robot, median, seconds=3.0, fps=fps)
        time.sleep(0.5)

    home = joints_of(robot.get_observation())
    outside = report_start_pose(home, starts, names)
    if outside:
        print(f"\n  {', '.join(outside)} outside anything the demonstrations show.")
        print("  Drive the arm to a demonstration start pose with the leader first:")
        print("  gripper a few centimetres from a block, jaws open, looking down.")
    if args.check or outside:
        robot.disconnect()
        return

    print(f"\n  {args.seconds:.0f}s at {fps} Hz, commands capped at "
          f"{args.max_step:.0f} deg per step. Ctrl-C to stop.\n")

    period = 1.0 / fps
    commanded = dict(home)     # the clamp follows the command, not the servo
    trace = Trace(args.trace, names, fps) if args.trace else None
    steps = 0
    reason = "time limit"
    try:
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            started = time.perf_counter()
            observation = robot.get_observation()
            frame = build_dataset_frame(observed, observation, prefix="observation")
            values = predict_action(
                observation=frame,
                policy=policy,
                device=get_safe_torch_device(config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=config.use_amp,
                task=task,
                robot_type=robot.robot_type,
            )
            wanted = make_robot_action(values, meta.features)

            # Clamp against the last command rather than the measured position.
            # The follower trails its target by tens of degrees during a fast
            # move - that is the servo catching up, not the policy asking for
            # something wild - and clamping against the measurement would let
            # that lag throttle every command.
            safe = {}
            for index, name in enumerate(names):
                key = f"{name}.pos"
                target = float(np.clip(wanted[key], low[index], high[index]))
                step = np.clip(target - commanded[name],
                               -args.max_step, args.max_step)
                commanded[name] = commanded[name] + float(step)
                safe[key] = commanded[name]
            robot.send_action(safe)
            if trace:
                trace.add(observation, wanted, safe, frame)
            steps += 1

            hurt = strained(robot)
            if hurt:
                reason = f"stopped: {hurt}"
                break
            time.sleep(max(0.0, period - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        rate = steps / max(1e-6, args.seconds)
        print(f"\n  {reason} after {steps} steps ({rate:.0f} Hz)")
        if trace:
            trace.close()
        try:
            print("  returning to where it started")
            glide_to(robot, home, seconds=2.0, fps=fps)
            time.sleep(0.5)
        except Exception as error:  # noqa: BLE001 - always still relax
            print(f"  could not return ({error})")
        finally:
            robot.disconnect()
            print("  arm relaxed")


if __name__ == "__main__":
    main()
