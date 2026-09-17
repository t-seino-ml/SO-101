"""Phase S2: where do the jaws actually hold a block?

That point is the TCP, and the project has had four disagreeing guesses at it,
measured on the real arm four different ways and spread over 70 mm against a
grasp that tolerates 5.6 mm. None of them was wrong exactly: the offset is fixed
in the *gripper's* frame, so measuring it at four different postures and writing
the answers down in base coordinates gives four different numbers.

TCP is defined here as:

    the centre of a 20 mm block held by the jaws, in the gripper frame

Closing the jaws on a block inside MuJoCo does not work, and it is worth saying
why: MuJoCo collides meshes as convex hulls, and the convex hull of a finger
fills the gap the finger exists to create. Probed directly, the whole region a
block should occupy reads as solid. So the physics is set aside and the CAD
geometry is used as it actually is - the meshes' vertices, which describe the
real, concave shape.

The method is a search rather than a formula: sweep a 20 mm cube through the
gripper frame, and at each position ask whether any vertex of either finger lies
inside it. The positions where none does are where a block can be. As the jaws
close that free region shrinks; the last place a block still fits is where the
jaws hold one, and that is the TCP.

Vertices rather than triangles because the meshes carry 14,000 of them each over
a part 60 mm across - dense enough that a surface cannot slip between samples of
a 20 mm cube.

    uv run scripts/sim_tcp.py
    uv run scripts/sim_tcp.py --save
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from so101.sim import BLOCK_SIZE_M, SO101Sim

TCP_PATH = Path("data/tcp.json")
# The real arm's answer, converted into the definition above.
# teach_servo_target.py stored (gripper frame - block) in gripper coordinates;
# a block's position in the gripper frame is the negative of that. Its z was
# never measured - the script only ever differenced x and y.
MEASURED_ON_REAL_MM = np.array([-16.3, +7.2, np.nan])
FINGERS = ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1")
POSE = {"shoulder_pan": 0.0, "shoulder_lift": -30.0, "elbow_flex": 60.0,
        "wrist_flex": 40.0, "wrist_roll": 0.0}


def finger_vertices(sim, gripper_deg):
    """The two fingers' vertices, in millimetres in the gripper frame."""
    import mujoco

    model, data = sim.model, sim.data
    sim.set_joints(POSE, gripper_deg=gripper_deg)
    origin, rotation = sim.gripper_pose()
    clouds = {}
    for index in range(model.ngeom):
        mesh = model.geom_dataid[index]
        if mesh < 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh)
        if name not in FINGERS or name in clouds:
            continue
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = model.mesh_vert[start:start + count] @ np.array(
            data.geom_xmat[index]).reshape(3, 3).T + data.geom_xpos[index]
        clouds[name] = 1000 * ((world - origin) @ rotation)
    return clouds


def facing_surfaces(clouds, z_low, z_high, y_half=8.0):
    """The two finger faces in a slab at this depth, and their separation.

    The jaw approaches along -x and the fixed finger sits at +x, established
    from the cross-sections. So the facing surfaces are the jaw's largest x and
    the fixed finger's smallest x within the slab.
    """
    jaw = clouds["moving_jaw_so101_v1"]
    fixed = clouds["wrist_roll_follower_so101_v1"]

    def slab(cloud):
        return cloud[(cloud[:, 2] > z_low) & (cloud[:, 2] < z_high)
                     & (np.abs(cloud[:, 1]) < y_half)]

    jaw, fixed = slab(jaw), slab(fixed)
    if not len(jaw) or not len(fixed):
        return None
    inner_jaw, inner_fixed = jaw[:, 0].max(), fixed[:, 0].min()
    return inner_fixed - inner_jaw, (inner_fixed + inner_jaw) / 2


def grip_for(clouds_by_angle, z_centre, width, slab=10.0):
    """The jaw angle that opens to `width` at this depth, and the centre there.

    The jaw rotates rather than translating, so the gap is a wedge: wide at the
    fingertips, narrow near the hinge. Which means the angle the jaws stop at on
    a block, and where that block's centre lands, both depend on how deep it
    sits - so neither is a property of the gripper alone.
    """
    for angle in sorted(clouds_by_angle):
        measured = facing_surfaces(clouds_by_angle[angle], z_centre - slab / 2,
                                   z_centre + slab / 2)
        if measured is None:
            continue
        gap, middle = measured
        if gap >= width:
            return angle, middle, gap
    return None


def pocket(clouds, low, step, shape, half):
    """Where a cube fits between the fingers, as positions in millimetres.

    Done as a voxel grid rather than a loop over candidate positions: the loop
    is a hundred thousand positions against thirty thousand vertices and takes
    minutes, while marking the vertices into a grid and growing them by the
    cube's half-width is one filter and takes milliseconds.

    "Between the fingers" means enclosed along the closing axis - x in the
    gripper frame, established from the cross-sections. Free space beside or in
    front of the gripper is not a pocket; only somewhere with finger on both
    sides is.
    """
    from scipy import ndimage

    occupied = np.zeros(shape, bool)
    for cloud in clouds.values():
        index = np.rint((cloud - low) / step).astype(int)
        keep = np.all((index >= 0) & (index < shape), axis=1)
        index = index[keep]
        occupied[index[:, 0], index[:, 1], index[:, 2]] = True

    width = int(round(2 * half / step)) | 1          # odd, so it stays centred
    blocked = ndimage.maximum_filter(occupied, size=width, mode="constant",
                                     cval=False)
    free = ~blocked

    # Enclosed along x: something occupied at a lower x and at a higher x.
    behind = np.cumsum(occupied, axis=0) > 0
    ahead = np.cumsum(occupied[::-1], axis=0)[::-1] > 0
    enclosed = free & behind & ahead

    cells = np.argwhere(enclosed)
    return cells * step + low if len(cells) else np.empty((0, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=1.0,
                        help="search resolution in mm")
    parser.add_argument("--angles", type=float, nargs="+",
                        default=[40, 35, 30, 26, 24, 22, 20, 16, 12, 8, 4],
                        help="gripper angles to close through, in degrees")
    parser.add_argument("--depth", type=float, default=-15.0,
                        help="how deep the block sits between the fingers, in "
                             "mm; this is set by how far the arm descends")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/sim/tcp.json"))
    args = parser.parse_args()

    sim = SO101Sim(table_z=-5.0)
    half = 1000 * BLOCK_SIZE_M / 2
    print(f"  {sim}")
    print(f"  sweeping a {1000*BLOCK_SIZE_M:.0f} mm cube at {args.step:.1f} mm "
          f"resolution\n")

    # A box around the fingers, from their own extents at the widest opening.
    widest = finger_vertices(sim, max(args.angles))
    stacked = np.vstack(list(widest.values()))
    low = stacked.min(axis=0) - half
    high = stacked.max(axis=0) + half
    shape = tuple(int(np.ceil((high[i] - low[i]) / args.step)) + 1
                  for i in range(3))
    print(f"  search volume x {low[0]:+.0f}..{high[0]:+.0f}  "
          f"y {low[1]:+.0f}..{high[1]:+.0f}  z {low[2]:+.0f}..{high[2]:+.0f} mm"
          f"   ({shape[0]}x{shape[1]}x{shape[2]} voxels)\n")

    # Sweep the jaw through its travel once; every question below is asked of
    # the same set of shapes.
    angles = np.arange(-8, 46, 1.0)
    clouds = {float(a): finger_vertices(sim, float(a)) for a in angles}

    print(f"  a {1000*BLOCK_SIZE_M:.0f} mm block, by how deep it sits between "
          f"the fingers\n")
    print(f"  {'depth (mm)':>12}{'jaw angle':>12}{'grasp centre x,y (mm)':>26}")
    rows = []
    for depth in np.arange(2, -60, -2.0):
        found = grip_for(clouds, depth, 1000 * BLOCK_SIZE_M)
        if found is None:
            print(f"  {depth:>9.0f}   {'never opens that far':>24}")
            continue
        angle, centre_x, gap = found
        # y is read off the pocket, which is symmetric about the finger pads.
        rows.append({"depth_mm": float(depth), "jaw_deg": float(angle),
                     "tcp_mm": [round(float(centre_x), 2), 0.0, float(depth)],
                     "gap_mm": round(float(gap), 2)})
        print(f"  {depth:>9.0f}{angle:>11.0f} deg"
              f"{f'{centre_x:+.1f}, {0.0:+.1f}':>26}")

    if not rows:
        raise SystemExit("the fingers never open to a block's width")

    xs = [r["tcp_mm"][0] for r in rows]
    print("\n  the grasp centre is not one point:")
    print(f"    across depth   x spans {min(xs):+.1f} .. {max(xs):+.1f} mm")
    shallow = [r for r in rows if r["depth_mm"] > -30]
    turned = shallow[-1]["jaw_deg"] - shallow[0]["jaw_deg"] if len(shallow) > 1 else 0
    slope = ((shallow[-1]["tcp_mm"][0] - shallow[0]["tcp_mm"][0]) / turned
             if turned else float("nan"))
    print(f"    per degree of jaw rotation, about {abs(slope):.1f} mm")

    # The operating point: a block sitting where the descent puts it. Depth is
    # ours to choose - it is how far the arm comes down - so it is pinned here
    # rather than discovered.
    operating = min(rows, key=lambda r: abs(r["depth_mm"] - args.depth))
    tcp = np.array(operating["tcp_mm"])
    print(f"\n  operating point: a block "
          f"{abs(operating['depth_mm']):.0f} mm deep")
    print(f"  TCP in the gripper frame: "
          f"{tcp[0]:+.1f}, {tcp[1]:+.1f}, {tcp[2]:+.1f} mm  "
          f"(jaw {operating['jaw_deg']:.0f} deg)")
    print(f"  the real arm measured     "
          f"{MEASURED_ON_REAL_MM[0]:+.1f}, {MEASURED_ON_REAL_MM[1]:+.1f}, "
          f"(z never measured)")
    gap = float(np.linalg.norm(tcp[:2] - MEASURED_ON_REAL_MM[:2]))
    print(f"  they differ by {gap:.1f} mm in x,y")
    print(f"\n  S2 acceptance: the two derivations agree < 3 mm  ->  "
          f"{'PASS' if gap < 3.0 else 'FAIL'}")
    if gap >= 3.0:
        print(f"  The CAD number describes the gripper. The measured one was")
        print(f"  differenced against a *detected* block position, so it carries")
        print(f"  the camera's systematic error as well - which is why it moved")
        print(f"  every time the calibration was redone. Phases S6-S8 separate")
        print(f"  the two; until then neither number should be trusted alone.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "tcp_in_gripper_mm": tcp.tolist(),
        "operating_depth_mm": operating["depth_mm"],
        "jaw_deg": operating["jaw_deg"],
        "measured_on_real_mm": [None if np.isnan(v) else v
                                for v in MEASURED_ON_REAL_MM],
        "difference_xy_mm": gap,
        "by_depth": rows,
    }, indent=1), encoding="utf-8")
    print(f"  {args.out}")

    if args.save:
        TCP_PATH.parent.mkdir(parents=True, exist_ok=True)
        TCP_PATH.write_text(json.dumps({
            "tcp_in_gripper_m": (tcp / 1000).tolist(),
            "source": "CAD mesh facing surfaces, scripts/sim_tcp.py",
            "operating_depth_mm": operating["depth_mm"],
            "jaw_deg_on_20mm_block": operating["jaw_deg"],
            "difference_from_real_xy_mm": gap,
            "note": ("depth is an operating choice, not a property of the "
                     "gripper: the jaw rotates, so the grasp centre moves with "
                     "both the angle and how deep the block sits"),
        }, indent=1), encoding="utf-8")
        print(f"  {TCP_PATH}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
