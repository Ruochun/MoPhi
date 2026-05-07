"""demo/newton_xlb_dem/demo_claw_newton.py

Newton UR10 robot arm + DEME granular terrain co-simulation demo.

This demo adds a granular terrain pile to the UR10 excavator arm simulation.
It reproduces Newton's ``example_robot_ur10`` example inside MoPhi's
``NewtonXLBDEMCoupler`` framework, demonstrating a single UR10 6-DOF industrial
robot arm holding a fixed ready-to-plow pose above a pile of DEME-managed
ellipsoidal particles.

The robot is mounted on a cylindrical pedestal above the ground plane.  All six
revolute joints are initialized to a fixed target configuration that points the
tooling toward +Y and keeps the arm above the terrain.

An excavator plow mesh (``data/mesh/excavator.obj``) is rigidly attached to the
UR10's end-effector link (``ee_link``).  The OBJ file uses centimetre units; the
mesh is scaled by 0.01 when loaded so it is correctly sized in metres.  The plow
is mounted with a flipped local orientation so the bowl faces downward for
plowing.

DEME granular terrain (Phase 2):
  A pile of ellipsoidal particles is created using DEME, following the approach
  of DEM-Engine's ``DEMdemo_Plow.cpp`` demo.  Particles are ellipsoids with
  semi-axes 2:1:1, sampled layer-by-layer with a Poisson-disk sampler to form
  a compact pile at ground level.  In this phase the terrain is DEME-only — no
  interaction with the Newton excavator yet. Particle–excavator coupling will be
  added in a future phase.

Future phases will add:
  • DEME–Newton coupling: forces from excavator plow on granular particles.
  • XLB (lattice-Boltzmann) fluid flow around the moving arm.

The demo closely mirrors Newton's ``example_robot_ur10`` setup (single arm,
SolverMuJoCo, position control) while wrapping all
physics inside the MoPhi ``NewtonXLBDEMCoupler``.

Prerequisites
-------------
  • Build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON (compiles
    NewtonXLBDEMCoupler and builds the mophi_core Python extension module).
  • pip install newton warp-lang   (Newton rigid-body physics + Warp GPU runtime)
  • pip install deme               (optional — DEME discrete-element solver)

Running
-------
From the MoPhi build tree (after ``cmake --build``):

    python demo/newton_xlb_dem/demo_claw_newton.py

Or from the repository root after installing the mophi package:

    python -m demo.newton_xlb_dem.demo_claw_newton
"""

import os
import sys
import time

import numpy as np

# ─── Import MoPhi ─────────────────────────────────────────────────────────
try:
    import mophi
except ImportError as exc:
    sys.exit(
        "ERROR: Could not import the 'mophi' package.\n"
        "       Build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON and ensure that\n"
        "       the build directory (python/) is on PYTHONPATH.\n"
        f"       ({exc})"
    )

if not hasattr(mophi, "NewtonXLBDEMCoupler"):
    sys.exit(
        "ERROR: mophi.NewtonXLBDEMCoupler is not available.\n"
        "       Re-build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON."
    )

# ─── Import Newton + Warp ──────────────────────────────────────────────────
try:
    import newton
    import warp as wp
    from newton import JointTargetMode
    from newton.selection import ArticulationView

    _newton_available = True
except ImportError:
    _newton_available = False
    print("ERROR: Newton (or warp) is not installed.\n" "       Install with:  pip install newton warp-lang")
    sys.exit(1)

# ─── Import DEME (optional) ─────────────────────────────────────────────────
try:
    import DEME

    _deme_available = True
except ImportError:
    _deme_available = False
    print(
        "INFO: DEME is not installed — granular terrain will be skipped.\n"
        "      Install DEME to enable DEM particle simulation."
    )


# ─── OBJ mesh loader ──────────────────────────────────────────────────────
# Pure-Python parser for Wavefront OBJ files.  Handles the v//vn and v/vt/vn
# face formats used by the excavator plow mesh (all faces are triangles).
def _load_obj_mesh(path: str, scale: float = 1.0):
    """Parse a Wavefront OBJ file and return (vertices, indices) as numpy arrays.

    Args:
        path:  Path to the .obj file.
        scale: Uniform scale factor applied to all vertex coordinates.  Use
               0.01 to convert centimetre OBJ coordinates to metres.

    Returns:
        vertices:  float32 ndarray of shape (N, 3) — vertex positions [m].
        indices:   int32 ndarray of shape (F*3,) — flat triangle index list.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    vertices = []
    indices = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]) * scale, float(parts[2]) * scale, float(parts[3]) * scale])
            elif line.startswith("f "):
                # Each token is v, v/vt, v//vn, or v/vt/vn — take the vertex index only.
                parts = line.split()[1:]
                face_verts = [int(p.split("/")[0]) - 1 for p in parts]  # OBJ uses 1-based indices
                if len(face_verts) == 3:
                    indices.extend(face_verts)
                elif len(face_verts) == 4:
                    # Fan-triangulate quads.
                    indices.extend([face_verts[0], face_verts[1], face_verts[2]])
                    indices.extend([face_verts[0], face_verts[2], face_verts[3]])
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)


# ─── Simulation timing ────────────────────────────────────────────────────
# Physics step sizes are explicit user-facing constants.  Rendering cadence is
# configured independently; collaboration loop counts are derived from these.
WORLD_COUNT = 1  # single UR10 arm

RENDER_FPS = 50
FRAME_DT = 1.0 / RENDER_FPS

# Keep Newton at the same step size used previously in this demo.
NEWTON_DT = 1.0 / 500.0
# DEME runs at an explicitly configured fine step size.
DEME_DT = 1.0e-5

SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError(
        "FRAME_DT must be an integer multiple of NEWTON_DT. "
        f"Got FRAME_DT={FRAME_DT:.12g}, NEWTON_DT={NEWTON_DT:.12g}."
    )

DEME_SUBSTEPS = int(round(NEWTON_DT / DEME_DT))
if not np.isclose(DEME_SUBSTEPS * DEME_DT, NEWTON_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError(
        "NEWTON_DT must be an integer multiple of DEME_DT. " f"Got NEWTON_DT={NEWTON_DT:.12g}, DEME_DT={DEME_DT:.12g}."
    )

SIM_FPS = RENDER_FPS
SIM_DT = NEWTON_DT

# Debug mode: render one frame, then keep the OpenGL window alive for visual
# inspection until the user closes it.
PAUSE_AFTER_FIRST_FRAME = True
NUM_FRAMES = 1 if PAUSE_AFTER_FIRST_FRAME else 250  # default run is ≈ 5 s at 50 Hz

# ─── DEME granular terrain constants ──────────────────────────────────────
# Physics parameters follow DEMdemo_Plow.cpp (projectchrono/DEM-Engine).
# Ellipsoid template with anisotropic semi-axes; scaling sets the physical particle size.
_DEM_PI = 3.1415927
_DEM_TERRAIN_SCALING = 0.03  # metres per template unit

# Unscaled ellipsoid template mass and MOI.
# ellipsoid_2_1_1.csv encodes an elongated ellipsoid in template coordinates.
# The MOI terms follow the standard solid-ellipsoid formula using the two
# orthogonal semi-axes for each principal axis.
# These values replicate DEMdemo_Plow.cpp exactly.
_DEM_TEMPLATE_MASS = 2600.0 * (4.0 / 3.0 * _DEM_PI * 2.0 * 1.0 * 1.0)
_DEM_TEMPLATE_MOI = [
    1.0 / 5.0 * _DEM_TEMPLATE_MASS * (1.0**2 + 2.0**2),  # I_x = (m/5)(sy²+sz²) = (m/5)(1+4)
    1.0 / 5.0 * _DEM_TEMPLATE_MASS * (1.0**2 + 2.0**2),  # I_y = (m/5)(sx²+sz²) = (m/5)(1+4)
    1.0 / 5.0 * _DEM_TEMPLATE_MASS * (1.0**2 + 1.0**2),  # I_z = (m/5)(sx²+sy²) = (m/5)(1+1)
]

# Terrain pile geometry (world-space, z-up).
# Configured to keep a compact pile beneath the arm workspace.
_DEM_WORLD_HS = 1.1  # domain half-size in x and y [m]
_DEM_BOWL_BOT = 0.0  # domain bottom
_DEM_FILL_HW = 1.0  # pile fill half-width in x and y [m]
_DEM_FILL_BOT = _DEM_BOWL_BOT + 3.0 * _DEM_TERRAIN_SCALING  # first layer bottom
_DEM_FILL_H = 5.0  # total pile height [m]
_DEM_LAYER_STEP = 4.5 * _DEM_TERRAIN_SCALING  # vertical spacing between fill layers

# ─── Initialize Warp ───────────────────────────────────────────────────────
print("=== MoPhi UR10 Claw Arm co-simulation demo ===\n")
wp.init()
device = wp.get_device()

# ─── Paths ────────────────────────────────────────────────────────────────
# Resolve the excavator OBJ mesh relative to the repository root so the demo
# can be run from any working directory.
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_DEMO_DIR))
EXCAVATOR_OBJ_PATH = os.path.join(_REPO_ROOT, "data", "mesh", "excavator.obj")

# ─── Load the UR10 robot model ─────────────────────────────────────────────
# newton.utils.download_asset("universal_robots_ur10") downloads the UR10 USD
# from the Newton Assets repository.  The arm is mounted on a cylindrical
# pedestal so its base link clears the ground plane.
print("[Newton] Downloading UR10 robot assets ...")
asset_path = newton.utils.download_asset("universal_robots_ur10")
asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
print(f"[Newton] Assets ready: {asset_path}\n")

print("[Newton] Building UR10 model ...")
pedestal_height = 1.2

# Build a single-arm sub-builder, then replicate it WORLD_COUNT times.
# This matches Newton's example pattern and lets ArticulationView find the
# articulation by the "*ur10*" glob pattern.
ur10_sub = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(ur10_sub)

ur10_sub.add_usd(
    asset_file,
    xform=wp.transform(wp.vec3(0.0, 0.0, pedestal_height)),
    collapse_fixed_joints=False,
    enable_self_collisions=False,
    hide_collision_shapes=True,
)
# Cylindrical pedestal attached to the world frame (fixed base).
ur10_sub.add_shape_cylinder(
    -1,
    xform=wp.transform(wp.vec3(0.0, 0.0, pedestal_height / 2.0)),
    half_height=pedestal_height / 2.0,
    radius=0.08,
)

# ─── Attach the excavator plow mesh to the UR10 end-effector ─────────────
# The plow OBJ is authored in centimetres.
# Scaling by OBJ_CM_TO_M converts it to metres.
# The OBJ coordinate origin is near the mounting bracket of the plow, and the
# blade/scoop extends in the local negative-Z direction.
# Apply a 180° local rotation so the bowl opens downward for plowing.
OBJ_CM_TO_M = 0.01
PLOW_LOCAL_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.pi))

ee_link_body_idx = ur10_sub.body_label.index("/ur10/ee_link")
assert ur10_sub.body_label[ee_link_body_idx] == "/ur10/ee_link", f"Unexpected ee_link body index: {ee_link_body_idx}"

if os.path.isfile(EXCAVATOR_OBJ_PATH):
    print(f"[Plow] Loading excavator mesh from {EXCAVATOR_OBJ_PATH} ...")
    _plow_verts, _plow_indices = _load_obj_mesh(EXCAVATOR_OBJ_PATH, scale=OBJ_CM_TO_M)
    plow_mesh = newton.Mesh(
        _plow_verts,
        _plow_indices,
        compute_inertia=False,  # plow mass is negligible for this demo phase
        is_solid=False,  # treat as a surface shell (open plow geometry)
        color=(0.55, 0.45, 0.35),  # earthy brown — excavator steel colour
    )
    ur10_sub.add_shape_mesh(
        ee_link_body_idx,
        mesh=plow_mesh,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), PLOW_LOCAL_ROT),
    )
    print(
        f"[Plow] Excavator plow attached to body {ee_link_body_idx} "
        f"({ur10_sub.body_label[ee_link_body_idx]})  "
        f"[{len(_plow_verts)} vertices, {len(_plow_indices) // 3} triangles]\n"
    )
else:
    print(f"[Plow] WARNING: excavator OBJ not found at {EXCAVATOR_OBJ_PATH} — plow will be omitted.\n")

# Position control with stiff gains matching Newton's UR10 example.
for i in range(len(ur10_sub.joint_target_ke)):
    ur10_sub.joint_target_ke[i] = 500
    ur10_sub.joint_target_kd[i] = 50
    ur10_sub.joint_target_mode[i] = int(JointTargetMode.POSITION)

# Replicate into a single-world builder and add the ground plane.
builder = newton.ModelBuilder()
builder.replicate(ur10_sub, WORLD_COUNT, spacing=(2.0, 2.0, 0.0))
builder.add_ground_plane()

newton_model = builder.finalize()

# SolverMuJoCo with contacts disabled — the arm doesn't need to collide with
# the ground during this first phase; DEME granular contacts will be added later.
newton_solver = newton.solvers.SolverMuJoCo(
    newton_model,
    disable_contacts=True,
)

print(f"[Newton] UR10 model built: {newton_model.body_count} bodies, {newton_model.joint_count} joints.\n")

# ─── Visualization backend selection ─────────────────────────────────────────
# Set USE_OMNIVERSE_VISUALIZATION = True to write each frame to a USD scene file
# (requires:  pip install usd-core).
# Set USE_OMNIVERSE_VISUALIZATION = False (default) to use Newton's real-time
# OpenGL window via mophi.OpenGLVisualizer.
# The simulation loop body (begin_frame / log_state / end_frame)
# is identical for both backends; only the constructor and close() differ.
USE_OMNIVERSE_VISUALIZATION = False

vis = None
_vis_available = False

if USE_OMNIVERSE_VISUALIZATION:
    vis = mophi.OmniverseVisualizer(output_path="demo_claw_newton.usdc", fps=float(SIM_FPS))
    _vis_available = vis.pxr_available
    if _vis_available:
        print("[USD] pxr (OpenUSD) available — Omniverse USD export enabled.\n")
    else:
        print(
            "[USD] pxr (OpenUSD) is not installed — Omniverse export disabled.\n"
            "      Install with:  pip install usd-core\n"
            "      The simulation will run without visualization."
        )
else:
    try:
        vis = mophi.OpenGLVisualizer(newton_model)
        _vis_available = True
        print("[Viewer] MoPhi OpenGL visualization window opened.\n")
    except Exception as exc:
        vis = None
        _vis_available = False
        print(
            f"[Viewer] Could not open MoPhi OpenGL viewer ({exc}).\n"
            "         The simulation will run without visualization."
        )

# ─── Build the DEME granular terrain ──────────────────────────────────────
# Creates a pile of ellipsoidal particles following the approach of
# DEM-Engine's DEMdemo_Plow.cpp demo.  Particles are sampled layer-by-layer
# with a Poisson-disk sampler and deposited near the ground plane.
# No interaction with the Newton excavator in this phase — DEME runs
# independently.  Particle–excavator coupling will be added in a future phase.
deme_solver = None
_dem_terrain_tracker = None
_dem_num_terrain_particles = 0

if _deme_available:
    print("[DEME] Building granular terrain (ellipsoidal particles) ...")
    try:
        deme_solver = DEME.DEMSolver()
        deme_solver.UseFrictionalHertzianModel()
        deme_solver.SetVerbosity("ERROR")

        mat_walls = deme_solver.LoadMaterial({"E": 1e7, "nu": 0.3, "CoR": 0.3, "mu": 0.5})
        mat_particles = deme_solver.LoadMaterial({"E": 1e7, "nu": 0.3, "CoR": 0.3, "mu": 0.5})
        # Mixed contact properties between wall and particle materials.
        deme_solver.SetMaterialPropertyPair("CoR", mat_walls, mat_particles, 0.3)
        deme_solver.SetMaterialPropertyPair("mu", mat_walls, mat_particles, 0.5)

        # Ellipsoid template (semi-axes 2:1:1) scaled to physical particle size.
        # The CSV file encodes the clump geometry relative to unit sphere radii;
        # Scale() resizes the template to a reasonable physical particle size.
        particle_template = deme_solver.LoadClumpType(
            _DEM_TEMPLATE_MASS,
            _DEM_TEMPLATE_MOI,
            DEME.GetDEMEDataFile("clumps/ellipsoid_2_1_1.csv"),
            mat_particles,
        )
        particle_template.Scale(_DEM_TERRAIN_SCALING)

        # Box domain: closed on bottom and four sides, open at the top so
        # particles can be launched upward by the excavator without being trapped.
        _DEM_BOX_POS_OFF_X, _DEM_BOX_POS_OFF_Y = 0.0, 1.2
        deme_solver.InstructBoxDomainDimension(
            [-_DEM_WORLD_HS + _DEM_BOX_POS_OFF_X, _DEM_WORLD_HS + _DEM_BOX_POS_OFF_X],
            [-_DEM_WORLD_HS + _DEM_BOX_POS_OFF_Y, _DEM_WORLD_HS + _DEM_BOX_POS_OFF_Y],
            [_DEM_BOWL_BOT, _DEM_WORLD_HS * 10.0],
        )
        deme_solver.InstructBoxDomainBoundingBC("top_open", mat_walls)

        # Sample particle positions layer by layer (mirrors DEMdemo_Plow.cpp).
        # PDSampler guarantees spacing proportional to configured particle size, so
        # no two initial particles overlap.
        sampler = DEME.PDSampler(4.0 * _DEM_TERRAIN_SCALING)
        pile_positions = []
        layer_z = 0.0
        while layer_z < _DEM_FILL_H:
            center = [_DEM_BOX_POS_OFF_X, _DEM_BOX_POS_OFF_Y, _DEM_FILL_BOT + layer_z]
            half_ext = [_DEM_FILL_HW, _DEM_FILL_HW, 0.0]
            layer_pts = sampler.SampleBox(center, half_ext)
            pile_positions.extend(layer_pts)
            layer_z += _DEM_LAYER_STEP

        _dem_num_terrain_particles = len(pile_positions)
        num_layers = int(_DEM_FILL_H / _DEM_LAYER_STEP) + 1
        print(f"[DEME] Sampled {_dem_num_terrain_particles} terrain particle(s) " f"in {num_layers} layer(s).")

        the_pile = deme_solver.AddClumps(particle_template, pile_positions)
        the_pile.SetFamily(0)
        _dem_terrain_tracker = deme_solver.Track(the_pile)

        deme_solver.SetGravitationalAcceleration([0.0, 0.0, -9.81])
        deme_solver.SetInitTimeStep(DEME_DT)
        print(f"[DEME] Running at step size {DEME_DT}.\n")
        deme_solver.SetErrorOutAvgContacts(100)
        deme_solver.Initialize()

        print(f"[DEME] Granular terrain initialized " f"({_dem_num_terrain_particles} ellipsoidal particle(s)).\n")
    except Exception as exc:
        print(f"[DEME] Could not build granular terrain ({exc}) — disabling DEME.\n")
        deme_solver = None
        _dem_terrain_tracker = None
        _dem_num_terrain_particles = 0

# ─── Initialize the coupler ────────────────────────────────────────────────
# XLB is passed as None; DEME terrain solver is connected when available
# (no Newton–DEME force coupling yet — particles run independently).
coupler = mophi.NewtonXLBDEMCoupler()
coupler.set_verbosity(mophi.VERBOSITY_INFO)

print("[Coupler] Initializing NewtonXLBDEMCoupler ...")
coupler.initialize(
    newton_model=newton_model,
    newton_solver=newton_solver,
    xlb_simulation=None,
    deme_solver=deme_solver,
    sim_dt=SIM_DT,
)
print("[Coupler] NewtonXLBDEMCoupler initialized.\n")

# ─── Fixed joint target setup (after coupler.initialize()) ───────────────
# ArticulationView provides structured access to the UR10 articulation's DOFs.
# The "*ur10*" glob matches all UR10 instances in the model.
articulation_view = ArticulationView(
    newton_model,
    "*ur10*",
    exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE],
)
assert (
    articulation_view.count == WORLD_COUNT
), f"Expected {WORLD_COUNT} UR10 articulation(s), found {articulation_view.count}"

dof_count = articulation_view.joint_dof_count

# Initialize to a ready-to-plow direction: arm reaches toward +Y and remains
# above the terrain.  Keep any extra DOFs at their current initialized values.
joint_q_target_np = articulation_view.get_attribute("joint_q", coupler.newton_state_0).numpy()
ready_to_plow_q = np.array(
    # UR10 joint order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3.
    [0.5 * np.pi, -1.25, 1.55, -0.30, -0.5 * np.pi, 0.0],
    dtype=np.float32,
)
num_ready_dofs = min(dof_count, len(ready_to_plow_q))
joint_q_target_np[:, 0, :num_ready_dofs] = ready_to_plow_q[:num_ready_dofs]

joint_q_target_wp = wp.array(joint_q_target_np, dtype=wp.float32, device=device)
articulation_view.set_attribute("joint_q", coupler.newton_state_0, joint_q_target_wp)
articulation_view.set_attribute("joint_target_pos", coupler.newton_control, joint_q_target_wp)

print(f"[Control] Applied fixed ready-to-plow joint target for {num_ready_dofs} DOFs.\n")

# ─── Movie recording settings ─────────────────────────────────────────────
# Set SAVE_MOVIE = True to record the rendered simulation frames to a video file.
# Requires: pip install imageio imageio-ffmpeg
SAVE_MOVIE = True
MOVIE_OUTPUT_PATH = "demo_claw_newton.mp4"
MOVIE_FPS = SIM_FPS

_movie_writer = None
if SAVE_MOVIE and _vis_available and not USE_OMNIVERSE_VISUALIZATION:
    try:
        import imageio

        _movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS)
        print(f"[Movie] Recording simulation to '{MOVIE_OUTPUT_PATH}' at {MOVIE_FPS} fps.\n")
    except ImportError:
        print(
            "WARNING: imageio is not installed — movie recording disabled.\n"
            "         Install with:  pip install imageio imageio-ffmpeg"
        )
        SAVE_MOVIE = False

# ─── DEME terrain visualisation arrays ────────────────────────────────────
# Clump template geometry mirrors the ellipsoid_2_1_1.csv data fed to the DEME
# solver (5 component spheres; template-unit values scaled by _DEM_TERRAIN_SCALING).
# CSV columns are x, y, z, r (offset from clump centre in local frame, then radius).
# These arrays mirror the template's component-sphere layout so visualized clumps
# match DEME particle geometry.
_DEM_CLUMP_SPHERE_RADII = np.array([1.00, 0.88, 0.64, 0.88, 0.64], dtype=np.float32) * _DEM_TERRAIN_SCALING
_DEM_CLUMP_SPHERE_OFFSETS = (
    np.array(
        [[0.0, 0.0, 0.00], [0.0, 0.0, 0.86], [0.0, 0.0, 1.44], [0.0, 0.0, -0.86], [0.0, 0.0, -1.44]],
        dtype=np.float32,
    )
    * _DEM_TERRAIN_SCALING
)

_dem_terrain_colors_wp = None
if _deme_available and _dem_terrain_tracker is not None and _vis_available and _dem_num_terrain_particles > 0:
    _dem_terrain_colors_wp = wp.array(
        np.tile([0.76, 0.60, 0.42], (_dem_num_terrain_particles, 1)).astype(np.float32),
        dtype=wp.vec3,
    )
    print(f"[Viewer] {_dem_num_terrain_particles} DEME terrain particle(s) registered for visualisation.\n")

# ─── Co-simulation loop ────────────────────────────────────────────────────
# Each frame:
#   1. For each substep: advance Newton and DEME.
#   2. Visualize Newton arm state and DEME terrain particles.
print(
    f"Running up to {NUM_FRAMES} frame(s) "
    f"(frame_dt={FRAME_DT * 1000:.1f} ms, "
    f"{SIM_SUBSTEPS} Newton substeps × {NEWTON_DT * 1000:.3f} ms, "
    f"{DEME_SUBSTEPS} DEME micro-steps per Newton substep × {DEME_DT * 1000:.3f} ms) ...\n"
)

# Point the camera toward the arm base and terrain pile with a slight downward
# angle so both robot motion and near-ground particle behavior remain visible.
if _vis_available and not USE_OMNIVERSE_VISUALIZATION:
    vis.set_camera(
        pos=wp.vec3(3.0, -3.0, 2.5),
        pitch=-25.0,
        yaw=135.0,
    )

# ─── Static scene overlays: coordinate axes and scale bar ─────────────────
# Log arrows and lines ONCE before the simulation loop.  Newton's ViewerGL
# stores them in a persistent dict and re-renders them every frame; there
# is no need to re-submit static geometry.
#
# Placement rationale:
#   • Coordinate axes are placed in the foreground, clear of the arm and terrain.
#   • The scale bar is also placed in the foreground and slightly elevated so it
#     remains legible and avoids z-fighting with the ground.
if _vis_available:
    # XYZ coordinate-axis arrows at a foreground reference point.
    # X = red, Y = green, Z = blue; all arrows share the same configurable length.
    _AXIS_LEN = 0.5
    _AXIS_OX, _AXIS_OY, _AXIS_OZ = 0.8, -0.9, 0.02  # origin in world space
    _axis_origin = np.full((3, 3), [_AXIS_OX, _AXIS_OY, _AXIS_OZ], dtype=np.float32)
    _axis_tips = np.array(
        [
            [_AXIS_OX + _AXIS_LEN, _AXIS_OY, _AXIS_OZ],
            [_AXIS_OX, _AXIS_OY + _AXIS_LEN, _AXIS_OZ],
            [_AXIS_OX, _AXIS_OY, _AXIS_OZ + _AXIS_LEN],
        ],
        dtype=np.float32,
    )
    _axis_colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    vis.log_arrows(
        "coord_axes",
        wp.array(_axis_origin, dtype=wp.vec3),
        wp.array(_axis_tips, dtype=wp.vec3),
        wp.array(_axis_colors, dtype=wp.vec3),
    )

    # Scale bar in the foreground, raised above the ground to avoid
    # z-fighting.  Endpoint spheres anchor the bar ends
    # and make its extent unambiguous.
    _SB_X0, _SB_X1, _SB_Y, _SB_Z = -0.5, 0.5, -0.9, 0.06  # 1 m along x
    vis.log_lines(
        "scale_bar",
        wp.array([[_SB_X0, _SB_Y, _SB_Z]], dtype=wp.vec3),
        wp.array([[_SB_X1, _SB_Y, _SB_Z]], dtype=wp.vec3),
        (1.0, 1.0, 0.0),  # yellow
    )
    # Endpoint marker spheres so the scale bar endpoints are clearly visible.
    _sb_marker_radius = 0.04  # m
    vis.log_points(
        "scale_bar_markers",
        wp.array([[_SB_X0, _SB_Y, _SB_Z], [_SB_X1, _SB_Y, _SB_Z]], dtype=wp.vec3),
        radii=wp.array([_sb_marker_radius, _sb_marker_radius], dtype=wp.float32),
        colors=wp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=wp.vec3),
    )

sim_time = 0.0

for frame in range(NUM_FRAMES):
    # Stop early if the OpenGL viewer window has been closed by the user.
    # For OmniverseVisualizer, vis.is_running() always returns True.
    if _vis_available and not vis.is_running():
        print(f"\n[Viewer] Window closed by user after frame {frame}.")
        break

    # ── Collaboration loop: counts are derived from explicit dt constants ──
    for _ in range(SIM_SUBSTEPS):
        # ── DEME micro-step loop (runs at explicit DEME_DT) ──
        # Arm control is fixed in this demo phase; DEME advances independently.
        for _ in range(DEME_SUBSTEPS):
            # Advance DEME granular terrain one micro-step (no coupling to Newton yet).
            if deme_solver is not None:
                coupler.step_deme()

        # Advance Newton one substep (uses joint targets from the last DEME micro-step).
        coupler.step_newton()

    sim_time += FRAME_DT

    # ── Visualization ──────────────────────────────────────────────────────
    if _vis_available:
        vis.begin_frame(sim_time)
        vis.log_state(coupler.newton_state_0)

        # Render live DEME terrain particles as clumps (overlapping-sphere assemblies).
        # Each particle's template geometry matches the ellipsoid_2_1_1.csv clump
        # fed to the DEME solver; live positions and orientations come from the tracker.
        if _dem_terrain_tracker is not None and _dem_terrain_colors_wp is not None:
            _terrain_pos_np = np.array(_dem_terrain_tracker.Positions(), dtype=np.float32)
            _terrain_quat_np = np.array(_dem_terrain_tracker.OrientationQuaternions(), dtype=np.float32)
            _dem_terrain_pos_wp = wp.array(_terrain_pos_np, dtype=wp.vec3)
            _dem_terrain_orient_wp = wp.array(_terrain_quat_np, dtype=wp.vec4)
            vis.log_clumps(
                "dem_terrain",
                _dem_terrain_pos_wp,
                _dem_terrain_orient_wp,
                _DEM_CLUMP_SPHERE_RADII,
                _DEM_CLUMP_SPHERE_OFFSETS,
                colors=_dem_terrain_colors_wp,
            )

        vis.end_frame()

        if _movie_writer is not None:
            _movie_writer.append_data(vis.get_frame().numpy())

    # ── Console status (periodic) ──────────────────────────────────────────
    if (frame + 1) % 50 == 0 or frame == 0:
        print(f"  frame {frame + 1:>4}/{NUM_FRAMES}  sim_time={sim_time:.3f} s")

print()

# ─── Debug pause after first rendered frame ──────────────────────────────────
if PAUSE_AFTER_FIRST_FRAME and _vis_available and not USE_OMNIVERSE_VISUALIZATION and vis.is_running():
    print("[Debug] First frame rendered. Inspect the initial configuration; close the viewer window to continue.")
    while vis.is_running():
        vis.begin_frame(sim_time)
        vis.log_state(coupler.newton_state_0)

        if _dem_terrain_tracker is not None and _dem_terrain_colors_wp is not None:
            _terrain_pos_np = np.array(_dem_terrain_tracker.Positions(), dtype=np.float32)
            _terrain_quat_np = np.array(_dem_terrain_tracker.OrientationQuaternions(), dtype=np.float32)
            _dem_terrain_pos_wp = wp.array(_terrain_pos_np, dtype=wp.vec3)
            _dem_terrain_orient_wp = wp.array(_terrain_quat_np, dtype=wp.vec4)
            vis.log_clumps(
                "dem_terrain",
                _dem_terrain_pos_wp,
                _dem_terrain_orient_wp,
                _DEM_CLUMP_SPHERE_RADII,
                _DEM_CLUMP_SPHERE_OFFSETS,
                colors=_dem_terrain_colors_wp,
            )

        vis.end_frame()
        time.sleep(0.1)

# ─── Finalize ─────────────────────────────────────────────────────────────
if _movie_writer is not None:
    _movie_writer.close()
    print(f"[Movie] Saved simulation recording to '{MOVIE_OUTPUT_PATH}'.\n")

if vis is not None:
    vis.close()

print("[Coupler] Finalizing NewtonXLBDEMCoupler ...")
coupler.finalize()
print("[Coupler] NewtonXLBDEMCoupler finalized.\n")

print("Demo completed successfully.")
print(f"  Newton version : {newton.__version__}")
print(f"  Warp  version  : {wp.__version__}")
if _deme_available:
    print(f"  DEME           : installed ({_dem_num_terrain_particles} terrain particles)")
else:
    print("  DEME           : not installed (granular terrain skipped)")
