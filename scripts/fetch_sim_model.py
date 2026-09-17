"""Download the MuJoCo model of the SO-101, and the meshes both models need.

The URDF this project already solves with came from TheRobotStudio/SO-ARM100.
The same directory holds an MJCF generated from the same Onshape CAD - same
joint names, same origins, and a `gripperframe` site at exactly the coordinates
the URDF's gripper_frame_joint uses. So the simulation and the real arm can share
one kinematic description rather than two that drift apart.

The meshes were never fetched: the URDF references assets/*.stl and none of them
are here, which is fine for ikpy (it reads joints, not geometry) and not fine for
MuJoCo, which needs them for collision.

Nothing is vendored into the repository - these are someone else's files under
their own licence, so they are fetched on demand exactly as scripts/fetch_urdf.py
does for the URDF.

    uv run scripts/fetch_sim_model.py
    uv run scripts/fetch_sim_model.py --force
"""

import argparse
import urllib.error
import urllib.request
from pathlib import Path

BASE = ("https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/"
        "Simulation/SO101/")
MODELS = ("so101_new_calib.xml", "joints_properties.xml")
MESHES = (
    "base_motor_holder_so101_v1.stl",
    "base_so101_v2.stl",
    "motor_holder_so101_base_v1.stl",
    "motor_holder_so101_wrist_v1.stl",
    "moving_jaw_so101_v1.stl",
    "rotation_pitch_so101_v1.stl",
    "sts3215_03a_no_horn_v1.stl",
    "sts3215_03a_v1.stl",
    "under_arm_so101_v1.stl",
    "upper_arm_so101_v1.stl",
    "waveshare_mounting_plate_so101_v2.stl",
    "wrist_roll_follower_so101_v1.stl",
    "wrist_roll_pitch_so101_v2.stl",
)
DEFAULT_ROOT = Path("data/urdf")


def grab(url, out, force):
    if out.is_file() and not force:
        return out.stat().st_size, False
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        out.write_bytes(response.read())
    return out.stat().st_size, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="where the URDF already lives; assets go beside it")
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    total = fetched = 0
    for name in MODELS:
        size, new = grab(args.base + name, args.root / name, args.force)
        total += size
        fetched += new
        print(f"  {'fetched' if new else 'present'}  {args.root / name}  ({size:,} B)")

    missing = []
    for name in MESHES:
        try:
            size, new = grab(args.base + "assets/" + name,
                             args.root / "assets" / name, args.force)
        except urllib.error.HTTPError as error:
            missing.append((name, error.code))
            continue
        total += size
        fetched += new
        print(f"  {'fetched' if new else 'present'}  assets/{name}  ({size:,} B)")

    print(f"\n  {fetched} fetched, {total:,} bytes in {args.root}")
    if missing:
        print("  could not fetch:")
        for name, code in missing:
            print(f"    assets/{name}  HTTP {code}")
        raise SystemExit(1)

    # The MJCF names its meshes relative to meshdir="assets", so it only loads
    # from the directory the assets sit in. Check it parses before anyone builds
    # a scene on top of it.
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.root / MODELS[0]))
    print(f"\n  MuJoCo loads it: {model.nq} dof, {model.nbody} bodies, "
          f"{model.ngeom} geoms, {model.nmesh} meshes")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(model.njnt)]
    print(f"  joints: {', '.join(n for n in names if n)}")
    sites = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
             for i in range(model.nsite)]
    print(f"  sites:  {', '.join(n for n in sites if n)}")


if __name__ == "__main__":
    main()
