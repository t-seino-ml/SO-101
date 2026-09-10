"""Train an ACT policy on the recorded grasp demonstrations.

The policy learns one skill: grasp the block in front of the gripper. It sees the
wrist camera and the joint state, and nothing else - no coordinates, no colour.
Which block to go for is decided outside it, by the side camera and a coarse move.

Only the wrist camera is fed in by default. It is bolted to the arm, so its view
does not change when the rig moves to a different table or the external camera is
placed differently - which is the whole reason the policy is driven from it. Adding
the side camera would tie the policy to that camera's placement and undo this.

    uv run scripts/train_policy.py                       # 100k steps
    uv run scripts/train_policy.py --steps 20000         # a quick look first
    uv run scripts/train_policy.py --cameras wrist,side  # if you want both
"""

import argparse
import pathlib
import sys
from pathlib import Path

DEFAULT_ROOT = Path("data/demos")
DEFAULT_REPO = "so101/grasp_block"
DEFAULT_OUT = Path("runs/policy/grasp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--cameras", default="wrist",
                        help="comma-separated camera roles to feed the policy")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="how many future actions ACT predicts at once")
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue the run already in --out, optimizer and all")
    args = parser.parse_args()

    if not (args.root / "meta" / "info.json").is_file():
        raise SystemExit(f"No dataset at {args.root}. "
                         "Record demonstrations first: uv run scripts/record_demos.py")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("  Training on CPU. ACT on CPU takes many hours; expect to wait.")
    else:
        print(f"  Training on {torch.cuda.get_device_name(0)}")

    wanted = [f"observation.images.{role.strip()}"
              for role in args.cameras.split(",")]
    print(f"  policy inputs: {', '.join(wanted)} + observation.state")
    print(f"  {args.steps} steps, batch {args.batch}, "
          f"action chunk {args.chunk_size}")
    print(f"  checkpoints to {args.out}\n")

    # LeRobot derives the policy's inputs from every feature in the dataset, so a
    # recording that holds both cameras would put both into the policy. Narrowing
    # that needs input_features set before the policy is built - make_policy only
    # fills it in when it is empty - and that is not reachable from the CLI.
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import dataset_to_policy_features

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    features = dataset_to_policy_features(meta.features)
    available = [key for key in features if key.startswith("observation.images.")]
    unknown = [key for key in wanted if key not in available]
    if unknown:
        raise SystemExit(f"Not in the dataset: {unknown}. Available: {available}")

    outputs = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    inputs = {k: f for k, f in features.items()
              if k not in outputs
              and (not k.startswith("observation.images.") or k in wanted)}
    dropped = [k for k in available if k not in wanted]
    if dropped:
        print(f"  not shown to the policy: {', '.join(dropped)}")

    resume = []
    if args.resume:
        # Resuming needs the train config the run saved beside its weights, and
        # LeRobot picks the checkpoint up from that path rather than a step
        # number. The optimizer state travels with it, so this is a genuine
        # continuation and not a fresh run warm-started from the weights.
        pointer = args.out / "checkpoints" / "last.txt"
        if not pointer.is_file():
            raise SystemExit(f"Nothing to resume in {args.out}: no checkpoints/last.txt")
        last = args.out / "checkpoints" / pointer.read_text(encoding="utf-8").strip()
        config_path = last / "pretrained_model" / "train_config.json"
        if not config_path.is_file():
            raise SystemExit(f"{config_path} not found")
        print(f"  resuming from {last.name}")
        resume = ["--resume=true", f"--config_path={config_path}"]

    argv = resume + [
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={args.root}",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        f"--output_dir={args.out}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch}",
        f"--save_freq={args.save_every}",
        "--wandb.enable=false",
    ]
    if not args.resume:
        argv[1:1] = ["--policy.type=act",
                     f"--policy.chunk_size={args.chunk_size}",
                     f"--policy.n_action_steps={args.chunk_size}"]
    print("  lerobot-train " + " ".join(argv))
    print()

    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts.lerobot_train import train
    from lerobot.scripts import lerobot_train

    # LeRobot marks the newest checkpoint with a symlink, which Windows refuses to
    # create without administrator rights. It does so *after* saving, so the
    # checkpoint is intact and only the pointer fails - but the exception still
    # ends the run. Write a plain file holding the name instead. Patch the name in
    # the training module, which imported the function directly.
    def _record_last_checkpoint(checkpoint_dir):
        pointer = pathlib.Path(checkpoint_dir).parent / "last.txt"
        pointer.write_text(pathlib.Path(checkpoint_dir).name, encoding="utf-8")

    lerobot_train.update_last_checkpoint = _record_last_checkpoint

    # LeRobot reads a few arguments straight out of sys.argv rather than from
    # the list handed to draccus - --config_path, which resume needs, is one of
    # them - so the process has to look as though it were invoked with them.
    sys.argv = [sys.argv[0]] + argv

    config = draccus.parse(TrainPipelineConfig, args=argv)
    config.policy.input_features = inputs
    config.policy.output_features = outputs
    train(config)

    print(f"\n  Trained. Checkpoints in {args.out}")
    print("  Try it on the arm: uv run scripts/run_policy.py")


if __name__ == "__main__":
    main()
