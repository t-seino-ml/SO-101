"""Check a trained policy against the recordings, before it touches the arm.

A policy that misbehaves on the arm can be failing in several unrelated ways,
and they are indistinguishable from watching it move. This separates them
without moving anything:

- **replay** feeds recorded observations back and compares the actions with
  what the operator did. Large errors here mean the model, the processors or
  the inference path is wrong - not the robot.
- **ablation** asks the same state with a different episode's image and with a
  blank one. If the action barely changes, the camera is not driving the policy
  and it has memorised a state-to-action mapping, which cannot survive the arm
  being anywhere new.

Small replay errors with a large ablation response mean the policy is sound and
is failing on the arm for the other reason: it only knows the states the
demonstrations passed through, and nothing shows it how to recover from being
slightly elsewhere. That is a data problem, not a training-length one.

Note that every episode was trained on, so replay error measures fit, not
generalisation. It is a floor: a policy that cannot even reproduce its own
training data is broken.

    uv run scripts/check_policy.py
    uv run scripts/check_policy.py --episodes 3 7 14 --plot
    uv run scripts/check_policy.py --checkpoint runs/policy/grasp/checkpoints/020000
"""

import argparse
from pathlib import Path

import numpy as np

DEFAULT_POLICY = Path("runs/policy/grasp/checkpoints")
DEFAULT_ROOT = Path("data/demos")
DEFAULT_REPO = "so101/grasp_block"
SAMPLES_PER_EPISODE = (0.15, 0.35, 0.55, 0.75)


def newest_checkpoint(directory):
    directory = Path(directory)
    pointer = directory / "last.txt"
    if pointer.is_file():
        candidate = directory / pointer.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return candidate
    numbered = sorted(p for p in directory.glob("[0-9]*") if p.is_dir())
    if not numbered:
        raise SystemExit(f"No checkpoints in {directory}.")
    return numbered[-1]


def load(pretrained, meta, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(str(pretrained))
    config.pretrained_path = str(pretrained)
    config.device = device
    config.n_action_steps = 1        # one action per observation, for comparison
    policy = make_policy(config, ds_meta=meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(pretrained),
        dataset_stats=meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    policy.eval()
    return config, policy, preprocessor, postprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="which episodes to check (default: five spread out)")
    parser.add_argument("--plot", action="store_true",
                        help="also draw the first episode's trajectories")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    import torch
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    checkpoint = args.checkpoint or newest_checkpoint(args.policy_dir)
    pretrained = checkpoint / "pretrained_model"
    if not pretrained.is_dir():
        pretrained = checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    names = [name.removesuffix(".pos") for name in meta.features["action"]["names"]]
    config, policy, pre, post = load(pretrained, meta, device)
    camera = next(key for key in config.input_features
                  if key.startswith("observation.images."))
    task = meta.tasks.index[0]
    print(f"  {pretrained}")
    print(f"  {device}, camera {camera.removeprefix('observation.images.')}")

    episodes = args.episodes
    if episodes is None:
        episodes = np.linspace(0, meta.total_episodes - 1, 5).astype(int).tolist()

    def predict(state, image):
        policy.reset()
        pre.reset()
        post.reset()
        batch = {"observation.state": state[None].to(device),
                 camera: image[None].to(device), "task": task}
        with torch.inference_mode():
            action = post(policy.select_action(pre(batch)))
        return action.squeeze(0).cpu().numpy()

    # Replay: the whole of the first episode, action by action.
    first = episodes[0]
    start = meta.episodes["dataset_from_index"][first]
    stop = meta.episodes["dataset_to_index"][first]
    policy.reset()
    pre.reset()
    post.reset()
    predicted, actual = [], []
    for index in range(start, stop):
        item = dataset[index]
        batch = {"observation.state": item["observation.state"][None].to(device),
                 camera: item[camera][None].to(device), "task": task}
        with torch.inference_mode():
            predicted.append(post(policy.select_action(pre(batch)))
                             .squeeze(0).cpu().numpy())
        actual.append(item["action"].numpy())
    predicted, actual = np.array(predicted), np.array(actual)
    replay_error = np.abs(predicted - actual)

    print(f"\n  replay of episode {first} ({len(predicted)} frames)")
    print(f"  {'joint':<15}{'mean':>8}{'p95':>8}{'max':>8}"
          f"{'the joint moved':>18}")
    for index, name in enumerate(names):
        span = actual[:, index].max() - actual[:, index].min()
        print(f"  {name:<15}{replay_error[:, index].mean():>7.1f}d"
              f"{np.percentile(replay_error[:, index], 95):>7.1f}d"
              f"{replay_error[:, index].max():>7.1f}d{span:>17.0f}d")

    # Ablation: the same state, a different view.
    rng = np.random.default_rng(0)
    rows = []
    for episode in episodes:
        start = meta.episodes["dataset_from_index"][episode]
        stop = meta.episodes["dataset_to_index"][episode]
        for fraction in SAMPLES_PER_EPISODE:
            item = dataset[int(start + fraction * (stop - start))]
            state, image = item["observation.state"], item[camera]
            own = predict(state, image)
            other = dataset[int(rng.integers(len(dataset)))][camera]
            rows.append((np.abs(predict(state, other) - own),
                         np.abs(predict(state, torch.zeros_like(image)) - own),
                         np.abs(item["action"].numpy() - own)))
    swapped, blank, error = (np.array(column) for column in zip(*rows))

    print(f"\n  what the camera is worth, over {len(rows)} frames from "
          f"{len(episodes)} episodes")
    print(f"  {'joint':<15}{'error':>10}{'swapped view':>15}{'blank view':>13}")
    for index, name in enumerate(names):
        print(f"  {name:<15}{error[:, index].mean():>9.1f}d"
              f"{swapped[:, index].mean():>14.1f}d{blank[:, index].mean():>12.1f}d")
    print(f"  {'overall':<15}{error.mean():>9.1f}d{swapped.mean():>14.1f}d"
          f"{blank.mean():>12.1f}d")

    print()
    if error.mean() > 5:
        print("  It cannot reproduce its own training data. Something in the "
              "model or\n  the inference path is wrong - look there before the arm.")
    elif swapped.mean() < 2 * error.mean():
        print("  Changing the view barely changes the action: the policy is "
              "going by\n  joint state alone. More of the same data will not "
              "fix that.")
    else:
        print("  The policy fits its data and is driven by the camera. If it "
              "still fails\n  on the arm, it is meeting states the "
              "demonstrations never covered -\n  a shorter task or more varied "
              "demonstrations, not more training steps.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        seconds = np.arange(len(predicted)) / meta.fps
        for index, (name, axis) in enumerate(zip(names, axes.ravel())):
            axis.plot(seconds, actual[:, index], label="operator", linewidth=1.5)
            axis.plot(seconds, predicted[:, index], label="policy", linewidth=1.2)
            axis.set_title(name, fontsize=9)
            axis.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        for axis in axes[-1]:
            axis.set_xlabel("seconds")
        figure.tight_layout()
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"replay_ep{first}.png"
        figure.savefig(path, dpi=110)
        print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
