"""Download the SO-101 URDF used for inverse kinematics.

From TheRobotStudio/SO-ARM100, the upstream hardware repository. It is not
vendored here because it is someone else's file under their own licence; this
fetches it on demand instead.

    uv run scripts/fetch_urdf.py
"""

import argparse
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/"
       "Simulation/SO101/so101_new_calib.urdf")
DEFAULT_OUT = Path("data/urdf/so101_new_calib.urdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out.is_file() and not args.force:
        print(f"  {args.out} already present ({args.out.stat().st_size} bytes)")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {args.url}")
    with urllib.request.urlopen(args.url, timeout=60) as response:
        args.out.write_bytes(response.read())
    print(f"  wrote {args.out} ({args.out.stat().st_size} bytes)")

    from so101.policy.kinematics import ArmKinematics

    arm = ArmKinematics(args.out)
    print(f"  joints: {', '.join(arm.joint_names)}")
    for name, (low, high) in arm.joint_limits_deg().items():
        print(f"    {name:<14} {low:+7.1f} .. {high:+7.1f} deg")


if __name__ == "__main__":
    main()
