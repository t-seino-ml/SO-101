"""Load the SO-101 into MuJoCo, and put a table, some blocks and a can in front of it.

The robot description is the upstream MJCF, fetched by scripts/fetch_sim_model.py
from the same directory the URDF came from and generated from the same CAD. Its
`gripperframe` site sits at exactly the coordinates the URDF's
gripper_frame_joint uses, so the frame this reads back is the same frame
`so101.policy.kinematics` reports - which is what makes comparing them a test of
anything.

The scene is written as MJCF text and composed with the robot at load time rather
than kept as a file on disk. Block and can positions change every experiment, and
a scene that is regenerated from numbers cannot fall out of step with them.

Everything here is measured. Blocks are 20 mm cubes and the can is 85 mm across
and 35 mm tall, both from the bench; the table's height comes from
`table_height()` below, and that function is the only place it comes from.

That last point is the one worth being firm about. Four different table heights
were in circulation at once - -8.5, -20, -15.3 and -2.4 mm - and a clearance
means nothing until it says which of them it was measured against. See
`table_height` for what each of them was and why only one of them is a table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_PATH = Path("data/urdf/so101_new_calib.xml")

BLOCK_SIZE_M = 0.020          # 20 mm cubes, measured
CAN_DIAMETER_M = 0.085        # 85 mm across, measured
CAN_HEIGHT_M = 0.035          # 35 mm tall, measured
TABLE_THICKNESS_M = 0.02

_TABLE_Z = None


def table_height(model_path=MODEL_PATH):
    """Where the table is, in metres in the arm's frame. The one answer.

    The arm is bolted to the table, so the table is the plane its base sits on,
    and the model knows where that is: the lowest vertex of the base mesh. No
    camera, no hand-eye, no homography - the same CAD the URDF and the forward
    kinematics come from, and reproducible to the micron.

    The three numbers this replaces, so nobody reaches for one of them again:

      -8.5 mm   `data/table_frame.json` z_table. Not a table: it is the height
                of the *gripper frame* when a block held in the jaws rests on
                the table, and it was fitted through the old homography, so it
                also carries the camera calibration's error.
      -20 mm    the placeholder this constant used to hold, never measured.
      -15.3 mm  Phase S5's derivation - the -8.5 above, plus the TCP's offset
                below the gripper frame, minus half a block. Built on the first
                one, so it inherits everything the first one carries, and it
                moves whenever data/tcp.json does.

    Using the real figure costs about 6 mm of apparent clearance against the old
    one. That is the right direction to be wrong in.

    This number is fixed by where the arm is bolted, and it is NOT the place to
    absorb a disagreement between the model and the bench. On 2026-09-17 the
    model put the gripper 7.5 mm through the table while the gripper was
    demonstrably not touching it, and the base was confirmed sitting flush. A
    height that is checked with a ruler cannot be the explanation for that; the
    explanation is on the arm's side of the problem - collision geometry, a
    frame, the TCP, a joint zero - and moving this constant would only hide it
    while leaving every other prediction as wrong as it was.
    """
    global _TABLE_Z
    if _TABLE_Z is None:
        # A probe rig with its table far below, so nothing can rest on it and
        # change the answer; then read where the base's own underside is. The
        # pose is irrelevant - the base is a fixed body - but the kinematics
        # have to be run at all, or every geom is still sitting at the origin.
        probe = SO101Sim(model_path=model_path, table_z=-1.0)
        probe.set_joints({name: 0.0 for name in probe.ARM_JOINTS},
                         gripper_deg=0.0)
        _TABLE_Z = float(probe.lowest_point(bodies=("base",)))
    return _TABLE_Z

BLOCK_COLOURS = {
    "red": (0.78, 0.18, 0.16), "orange": (0.85, 0.45, 0.13),
    "yellow": (0.85, 0.68, 0.10), "green": (0.25, 0.55, 0.25),
    "blue": (0.18, 0.38, 0.68), "purple": (0.42, 0.28, 0.58),
}


def _quaternion(matrix):
    """(w, x, y, z) for a rotation matrix, MuJoCo's ordering."""
    trace = np.trace(matrix)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        return np.array([0.25 * scale,
                         (matrix[2, 1] - matrix[1, 2]) / scale,
                         (matrix[0, 2] - matrix[2, 0]) / scale,
                         (matrix[1, 0] - matrix[0, 1]) / scale])
    index = int(np.argmax(np.diag(matrix)))
    a, b, c = (index + 1) % 3, (index + 2) % 3, index
    scale = np.sqrt(1.0 + matrix[c, c] - matrix[a, a] - matrix[b, b]) * 2
    quaternion = np.zeros(4)
    quaternion[0] = (matrix[b, a] - matrix[a, b]) / scale
    quaternion[c + 1] = 0.25 * scale
    quaternion[a + 1] = (matrix[a, c] + matrix[c, a]) / scale
    quaternion[b + 1] = (matrix[b, c] + matrix[c, b]) / scale
    return quaternion


def _scene_xml(model_path, table_z, table_size, blocks, can, wrist_camera):
    """The MJCF for everything that is not the robot, wrapping the robot model."""
    body = []
    body.append(
        f'<geom name="table" type="box" pos="{table_size[0]/2 + 0.05:.4f} 0 '
        f'{table_z - TABLE_THICKNESS_M / 2:.4f}" '
        f'size="{table_size[0]/2:.4f} {table_size[1]/2:.4f} '
        f'{TABLE_THICKNESS_M/2:.4f}" rgba="0.62 0.44 0.29 1" '
        f'friction="1 0.01 0.001"/>')

    for index, (colour, (x, y)) in enumerate(blocks):
        rgb = BLOCK_COLOURS.get(colour, (0.5, 0.5, 0.5))
        half = BLOCK_SIZE_M / 2
        body.append(
            f'<body name="block_{index}" pos="{x:.4f} {y:.4f} {table_z + half:.4f}">'
            f'<freejoint name="block_{index}_free"/>'
            f'<geom name="block_{index}_geom" type="box" '
            f'size="{half:.4f} {half:.4f} {half:.4f}" '
            f'rgba="{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f} 1" mass="0.008" '
            f'friction="1 0.01 0.001"/>'
            f'<site name="block_{index}_centre" pos="0 0 0" size="0.002" '
            f'group="4"/></body>')

    if can is not None:
        x, y = can
        radius = CAN_DIAMETER_M / 2
        wall = 0.002
        # A hollow tin, as four walls and a floor: MuJoCo has no hollow cylinder
        # primitive and a mesh is not worth fetching for a shape this simple.
        body.append(f'<body name="can" pos="{x:.4f} {y:.4f} {table_z:.4f}">')
        body.append(
            f'<geom name="can_floor" type="cylinder" pos="0 0 {wall/2:.4f}" '
            f'size="{radius:.4f} {wall/2:.4f}" rgba="0.72 0.70 0.66 1" '
            f'mass="0.03"/>')
        segments = 24
        for k in range(segments):
            angle = 2 * np.pi * k / segments
            body.append(
                f'<geom name="can_wall_{k}" type="box" '
                f'pos="{radius*np.cos(angle):.4f} {radius*np.sin(angle):.4f} '
                f'{CAN_HEIGHT_M/2:.4f}" '
                f'size="{wall:.4f} {np.pi*radius/segments:.4f} '
                f'{CAN_HEIGHT_M/2:.4f}" '
                f'euler="0 0 {angle:.4f}" rgba="0.72 0.70 0.66 1" mass="0.001"/>')
        body.append(
            f'<site name="can_opening" pos="0 0 {CAN_HEIGHT_M:.4f}" '
            f'size="0.002" group="4"/>')
        body.append("</body>")

    camera = ""
    if wrist_camera is not None:
        # A camera bolted to the world, looking at a point. MuJoCo's "targetbody"
        # would need a body to aim at; giving the orientation directly keeps the
        # pose exactly what the caller asked for, which matters when the whole
        # experiment is about recovering that pose from the picture.
        position, lookat, fovy = wrist_camera
        position = np.asarray(position, float)
        forward = np.asarray(lookat, float) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        # MuJoCo cameras look down their own -z, with +x right and +y up.
        matrix = np.stack([right, up, -forward], axis=1)
        quaternion = _quaternion(matrix)
        camera = (f'<camera name="side" mode="fixed" '
                  f'pos="{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}" '
                  f'quat="{quaternion[0]:.8f} {quaternion[1]:.8f} '
                  f'{quaternion[2]:.8f} {quaternion[3]:.8f}" fovy="{fovy:.4f}"/>')

    return f"""<mujoco model="so101_rig">
  <include file="{model_path.name}"/>
  <statistic center="0.2 0 0.1" extent="0.8"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="150" elevation="-25" offwidth="1280" offheight="960"/>
  </visual>
  <worldbody>
    <light pos="0.3 0 1.2" dir="0 0 -1" directional="true"/>
    {camera}
    {"".join(body)}
  </worldbody>
</mujoco>
"""


class SO101Sim:
    """The arm in MuJoCo, driven in the same degrees the real robot reports."""

    #: The site the upstream MJCF puts where the URDF's gripper_frame_link is.
    GRIPPER_SITE = "gripperframe"
    ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll")
    #: The two models agree on where the gripper frame is - 0.004 mm over 500
    #: random poses - and disagree about which way it faces. The URDF turns its
    #: gripper_frame_link 180 degrees about y from gripper_link and the MJCF
    #: turns its site 90 degrees, leaving a constant -90 degrees about y between
    #: them (measured deviation 1.4e-05 over 200 poses, so: constant). Everything
    #: else here is expressed in the URDF's convention, because that is the one
    #: ikpy solves in and therefore the one the real arm is driven in.
    SITE_TO_URDF = np.array([[0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0]])

    def __init__(self, model_path=MODEL_PATH, table_z=None,
                 table_size=(0.60, 0.80), blocks=(), can=None,
                 wrist_camera=None):
        # None, not a constant default: the height is `table_height()`'s to
        # decide, and a default written here would be a second place to change.
        if table_z is None:
            table_z = table_height(model_path)
        import mujoco

        self.mujoco = mujoco
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} not found. Run scripts/fetch_sim_model.py")
        self.table_z = table_z
        self.blocks = list(blocks)
        self.can = can

        xml = _scene_xml(model_path, table_z, table_size, self.blocks, can,
                         wrist_camera)
        # Written beside the robot model and compiled from there, so both
        # `meshdir="assets"` and the <include> resolve against the right
        # directory. from_xml_string would need every mesh passed in as a virtual
        # asset, which is a lot of plumbing to avoid one temporary file.
        self.model = self._compile(xml, model_path)
        self.data = mujoco.MjData(self.model)
        self._joint_index = {
            name: self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in (*self.ARM_JOINTS, "gripper")}

    @staticmethod
    def _compile(xml, model_path):
        import mujoco

        scene = model_path.parent / "_rig_scene.xml"
        scene.write_text(xml, encoding="utf-8")
        try:
            return mujoco.MjModel.from_xml_path(str(scene))
        finally:
            scene.unlink(missing_ok=True)

    # -- state ------------------------------------------------------------

    def set_joints(self, joints_deg, gripper_deg=None, settle=False):
        """Place the arm at these angles, in the degrees LeRobot reports."""
        for name in self.ARM_JOINTS:
            if name in joints_deg:
                self.data.qpos[self._joint_index[name]] = np.deg2rad(
                    joints_deg[name])
        if gripper_deg is not None:
            self.data.qpos[self._joint_index["gripper"]] = np.deg2rad(gripper_deg)
        if settle:
            self.mujoco.mj_forward(self.model, self.data)
            for _ in range(200):
                self.mujoco.mj_step(self.model, self.data)
        else:
            self.mujoco.mj_kinematics(self.model, self.data)
            self.mujoco.mj_forward(self.model, self.data)
        return self

    def joints_deg(self):
        return {name: float(np.rad2deg(self.data.qpos[index]))
                for name, index in self._joint_index.items()}

    def hold_still(self, gripper_deg=None):
        """Point every actuator at where the arm currently is.

        Without this the position servos drive towards whatever ctrl happens to
        hold - zero - the moment the simulation is stepped, and the arm collapses
        before the thing being measured has happened.
        """
        for name in (*self.ARM_JOINTS, "gripper"):
            index = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if index >= 0:
                self.data.ctrl[index] = self.data.qpos[self._joint_index[name]]
        if gripper_deg is not None:
            self.command_gripper(gripper_deg)
        return self

    def command_gripper(self, degrees):
        index = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        self.data.ctrl[index] = np.deg2rad(degrees)
        return self

    def step(self, steps=1):
        for _ in range(steps):
            self.mujoco.mj_step(self.model, self.data)
        return self

    def weightless(self, off=True):
        """Gravity off, for measuring geometry rather than watching things fall."""
        self.model.opt.gravity[:] = (0, 0, 0) if off else (0, 0, -9.81)
        return self

    # -- free bodies ------------------------------------------------------

    def _free_qpos(self, name):
        body = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY,
                                      name)
        if body < 0:
            raise KeyError(f"no body named {name!r}")
        joint = self.model.body_jntadr[body]
        if joint < 0 or self.model.jnt_type[joint] != self.mujoco.mjtJoint.mjJNT_FREE:
            raise KeyError(f"{name!r} has no free joint")
        return self.model.jnt_qposadr[joint]

    def place_body(self, name, position, quaternion=(1, 0, 0, 0)):
        """Put a free body at a world position, at rest."""
        start = self._free_qpos(name)
        self.data.qpos[start:start + 3] = position
        self.data.qpos[start + 3:start + 7] = quaternion
        velocity = self.model.jnt_dofadr[
            self.model.body_jntadr[
                self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY,
                                       name)]]
        self.data.qvel[velocity:velocity + 6] = 0
        self.mujoco.mj_forward(self.model, self.data)
        return self

    def body_in_gripper(self, name):
        """Where a body sits, expressed in the gripper frame (URDF convention)."""
        origin, rotation = self.gripper_pose()
        position, _ = self.body_pose(name)
        return rotation.T @ (position - origin)

    # -- frames -----------------------------------------------------------

    def site_pose(self, name=GRIPPER_SITE):
        """(position, 3x3 rotation) of a site, in the world frame.

        The robot's base sits at the world origin, so world and the arm's base
        frame are the same thing here - which is what lets these numbers be
        compared directly against so101.policy.kinematics.
        """
        index = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE,
                                       name)
        if index < 0:
            raise KeyError(f"no site named {name!r}")
        return (np.array(self.data.site_xpos[index]),
                np.array(self.data.site_xmat[index]).reshape(3, 3))

    def gripper_pose(self):
        """(position, rotation) of the gripper frame, in the URDF's convention.

        The same numbers so101.policy.kinematics reports for the real arm, so a
        target computed for one can be handed to the other without a conversion
        anyone has to remember.
        """
        position, rotation = self.site_pose(self.GRIPPER_SITE)
        return position, rotation @ self.SITE_TO_URDF

    def body_pose(self, name):
        index = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY,
                                       name)
        if index < 0:
            raise KeyError(f"no body named {name!r}")
        return (np.array(self.data.xpos[index]),
                np.array(self.data.xmat[index]).reshape(3, 3))

    def geom_position(self, name):
        index = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM,
                                       name)
        if index < 0:
            raise KeyError(f"no geom named {name!r}")
        return np.array(self.data.geom_xpos[index])

    # -- contact ----------------------------------------------------------

    #: The arm's own links. Its geoms come from meshes and are unnamed, so
    #: contacts have to be identified by the body they belong to - an earlier
    #: version keyed off geom names, found None for every arm geom, and quietly
    #: reported that the arm never touched anything.
    ARM_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist",
                  "gripper", "moving_jaw_so101_v1")

    def _touching(self, index):
        contact = self.data.contact[index]
        out = []
        for geom in (contact.geom1, contact.geom2):
            body = self.model.geom_bodyid[geom]
            out.append((
                self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY,
                                       body),
                self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM,
                                       geom)))
        return out[0], out[1], -1000 * float(contact.dist)

    def contacts(self, depth_mm=0.0):
        """Every touching pair now, as ((body, geom), (body, geom), depth mm).

        `depth_mm` ignores contacts shallower than that. Resting geometry
        touches at essentially zero depth all the time - blocks on the table,
        the arm's own neighbouring links - and counting those as collisions
        makes every pose look like a crash.
        """
        return [touch for touch in
                (self._touching(index) for index in range(self.data.ncon))
                if touch[2] >= depth_mm]

    def collisions(self, depth_mm=0.5, ignore=()):
        """Contacts that matter: an arm link against the world, or against itself.

        A block resting on the table is not a collision. Nor, unfortunately, is
        the gripper "hitting" a block it is reaching for: MuJoCo collides meshes
        as convex hulls, and the hull of the gripper fills the gap between its
        fingers, so every approach to a block reads as a crash. Phase S2 hit the
        same wall trying to close the jaws in simulation. Blocks therefore have
        to be named in `ignore`, and their clearance measured geometrically
        instead - see scripts/sim_pick_trajectory.py.

        What is left is honest: the table, and the arm against itself.
        """
        ignored = set(ignore)
        found = []
        for (body_a, geom_a), (body_b, geom_b), depth in self.contacts(depth_mm):
            names = {body_a, body_b, geom_a, geom_b} - {None}
            if names & ignored:
                continue
            arm = [body for body in (body_a, body_b) if body in self.ARM_BODIES]
            if not arm:
                continue                       # block on table, block on block
            if len(arm) == 2:
                # Neighbouring links are always in contact at the joint; only
                # links that are not adjacent in the chain count as a clash.
                first, second = (self.ARM_BODIES.index(body) for body in arm)
                if abs(first - second) <= 1:
                    continue
            found.append((geom_a or body_a, geom_b or body_b, depth))
        return found

    def hits_table(self, depth_mm=0.5):
        """Whether any arm link is in the table, by contact.

        Reliable for the upper links and not for the gripper: its convex hull
        swallows the gap between the fingers and bulges below the fingertips, so
        down at grasping height it reports a crash the real gripper does not
        have. Use `lowest_point` there.
        """
        return any("table" in (first, second)
                   for first, second, _ in self.collisions(depth_mm))

    def lowest_point(self, bodies=None):
        """The lowest world z of any arm mesh vertex, in metres.

        Vertices, not hulls. The hull of the gripper is a blunt lump that hangs
        below the fingertips; the vertices are the shape the fingers actually
        are, which is what decides whether they clear the table. Same reasoning
        as Phase S2's grasp geometry - and the same reason contact queries are
        not used for it.
        """
        bodies = set(bodies or self.ARM_BODIES)
        lowest = np.inf
        for index in range(self.model.ngeom):
            body = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[index])
            if body not in bodies:
                continue
            mesh = self.model.geom_dataid[index]
            if mesh < 0:
                continue
            start = self.model.mesh_vertadr[mesh]
            count = self.model.mesh_vertnum[mesh]
            vertices = self.model.mesh_vert[start:start + count]
            world = vertices @ np.array(
                self.data.geom_xmat[index]).reshape(3, 3).T \
                + self.data.geom_xpos[index]
            lowest = min(lowest, float(world[:, 2].min()))
        return lowest

    def __str__(self):
        return (f"SO101Sim({self.model.nbody} bodies, {self.model.ngeom} geoms, "
                f"table z={1000*self.table_z:+.0f} mm, "
                f"{len(self.blocks)} block(s), can={'yes' if self.can else 'no'})")
