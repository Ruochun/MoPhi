"""Settle sampled DFC particles, then feed them through a Newton-aligned nozzle.

DEME creates fresh-concrete particles inside the auger contact proxy and settles
them with temporary analytical confinement. Once their kinetic energy remains
low, the confinement is disabled and flow begins in the same solver. Newton
remains stationary and receives no particle forces; G-code motion is disabled
by default.
"""

import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp

import mophi
import newton
import newton.ik as ik
from mophi.couplers.newton_deme import NewtonDEMEParticleExchange
from mophi.utils import (
    load_package_provider,
    parse_gcode,
    sample_grid_between_mesh_surfaces,
    triangulate_spheres,
    write_vtk_file_series,
    write_vtk_polydata,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared_utils.abb_irb6700_printer_utils import build_abb_irb6700_printer  # noqa: E402

DEME = load_package_provider("deme")

# ── User configuration ───────────────────────────────────────────────────────
NEWTON_DT = 1.0 / 500.0
NEWTON_GRAVITY = (0.0, 0.0, 0.0)
RENDER_FPS = 50
WARMUP_SECONDS = 0.5
POST_PATH_HOLD_SECONDS = 0.5
RENDER_SETTLING_PHASE = True
RUN_PARTICLE_DISCHARGE_AFTER_SETTLING = True
RUN_GCODE_MOTION_AFTER_SETTLING = False
GCODE_TIME_SCALE = 1.0
GCODE_RAPID_FEED_RATE = 0.25
GCODE_TO_WORLD_SCALE = (1.0, 1.0, 1.0)

JOINT_TARGET_STIFFNESS = 500.0
JOINT_TARGET_DAMPING = 50.0
AUGER_TARGET_STIFFNESS = 100.0
AUGER_TARGET_DAMPING = 10.0
IK_ITERATIONS = 24
IK_DAMPING = 0.1

# DFC uses the millimetre-tonne-second unit convention of the reference slump
# test. Mesh-local metres are converted into DEME coordinates during setup.
DEME_DT_INITIAL = 1.0e-8
DEME_DT_MAX = 1.0e-5
DEME_DT_GROWTH_FACTOR = 1.01
DEME_DT_GROWTH_STEP_COUNT = 20
DEME_SETTLING_CHECK_INTERVAL = 0.02
DEME_SETTLING_MIN_SECONDS = 0.10
DEME_SETTLING_MAX_SECONDS = 2.0
DEME_SETTLING_KINETIC_ENERGY_THRESHOLD = 0.1
DEME_SETTLING_REQUIRED_LOW_ENERGY_CHECKS = 5
DEME_GRAVITY = (0.0, 0.0, -9810.0)
DEME_ERROR_OUT_VELOCITY = 1.0e7
DEME_PARTICLE_FAMILY = 0
DEME_NOZZLE_PROXY_FAMILY = 10
DEME_NOZZLE_PROXY_PATCH_ANGLE_DEGREES = 20.0
DEME_SETTLING_BOUNDARY_FAMILY = 11
DEME_BUILD_PLATE_FAMILY = 12
ENABLE_DEME_BUILD_PLATE = False
# Match the Chrono MiniSlump grading: aggregate diameters span 2--4 mm,
# surrounded by a 2.5 mm mortar layer in the DFC collision radius.
DEME_AGGREGATE_DIAMETER_MIN = 2.0
DEME_AGGREGATE_DIAMETER_MAX = 4.0
DEME_AGGREGATE_GRADING_EXPONENT = 0.5
DEME_PARTICLE_GRADING_SEED = 406
DEME_MORTAR_LAYER = 2.5
DEME_CONCRETE_DENSITY = 2.072109e-9
DEME_PARTICLE_MASS_DENSITY_SCALE = 0.5
DEME_MAX_PARTICLE_RADIUS = DEME_AGGREGATE_DIAMETER_MAX / 2.0 + DEME_MORTAR_LAYER
DEME_PARTICLE_SPACING = 2.1 * DEME_MAX_PARTICLE_RADIUS
# These bounds clip the upper material chamber in contact-mesh-local metres.
# Local X crosses its side walls; the expanded local Y/Z span fills more of the chamber.
DEME_AUGER_SAMPLE_BOUNDS_MIN = (0.10, -0.09, -0.09)
DEME_AUGER_SAMPLE_BOUNDS_MAX = (0.17, -0.01, 0.045)
DEME_AUGER_SAMPLE_RAY_AXIS = 0
# Keep the largest mortar-coated sphere clear of the initial contact surface.
DEME_AUGER_SAMPLE_CLEARANCE = 0.0055
DEME_BROAD_DOMAIN_X = (-1000.0, 1000.0)
DEME_BROAD_DOMAIN_Y = (-1000.0, 1000.0)
DEME_BROAD_DOMAIN_Z = (-500.0, 1500.0)
DEME_INITIAL_BIN_SIZE = 4.0 * DEME_MAX_PARTICLE_RADIUS
DEME_DISCHARGE_DT_MAX = 1.0e-6
DEME_DISCHARGE_SECONDS = 4.0
DEME_DFC_MATERIAL = {
    "ENm_mat": 7.0e-3,
    "ENa_mat": 100.0,
    "mortar_layer_mat": DEME_MORTAR_LAYER,
    "alpha_mat": 0.25,
    "beta_mat": 0.5,
    "np_mat": 1.0,
    "sgmTmax_mat": 3.0e-4,
    "sgmTau0_mat": 2.0e-5,
    "kappa0_mat": 100.0,
    "eta_inf_mat": 1.25e-5,
    "flocbeta_mat": 0.01,
    "flocTcr_mat": 100.0,
    "flocm_mat": 1.0,
    "lambda_init_mat": 2.0,
}
DEME_TO_WORLD_SCALE = 0.001
DEME_VISUAL_ORIGIN = (2.52, -0.57, 0.08)
DEME_PARTICLE_COLOR = (0.72, 0.42, 0.22)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PRINTER_ASSET_DIRECTORY = Path(__file__).resolve().parent / "data" / "chrono_irb6700_printer"
GCODE_PATH = Path(__file__).resolve().parent / "data" / "dfc_single_bead.gcode"
DFC_FORCE_MODEL_PATH = Path(__file__).resolve().parent / "data" / "DFCModel.cu"
BUILD_PLATE_CENTER = (2.52, -0.57, 0.04)
BUILD_PLATE_HALF_EXTENTS = (0.70, 0.70, 0.04)

USE_OMNIVERSE_VISUALIZATION = False
SAVE_MOVIE = True
SAVE_VTK_TIME_SERIES = True
SAVE_AUGER_CONTACT_PROXY_VTK = True
VTK_PARTICLE_LATITUDE_SEGMENTS = 6
VTK_PARTICLE_LONGITUDE_SEGMENTS = 8
CAMERA_POSITION = (4.7, -4.0, 2.7)
REFERENCE_AXIS_ORIGIN = (1.35, -1.25, 0.02)
REFERENCE_SCALE_BAR_CENTER = (2.45, -1.55, 0.05)
PATH_VISUAL_SAMPLES = 80
PATH_VISUAL_COLOR = (0.15, 0.80, 0.95)
PATH_VISUAL_WIDTH = 0.006
SHOW_NOZZLE_CONTACT_WIREFRAME = True
NOZZLE_CONTACT_WIREFRAME_COLOR = (0.95, 0.20, 0.85)
NOZZLE_CONTACT_WIREFRAME_WIDTH = 0.0015
SHOW_DEME_BUILD_PLATE = True
DEME_BUILD_PLATE_VISUAL_COLOR = (1.0, 0.65, 0.10)
DEME_BUILD_PLATE_VISUAL_WIDTH = 0.004
DEME_BUILD_PLATE_VISUAL_Z_OFFSET = 0.002
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "dfc_3d_printing_gcode.mp4"
USD_OUTPUT_PATH = OUTPUT_DIRECTORY / "dfc_3d_printing_gcode.usdc"
METADATA_OUTPUT_PATH = OUTPUT_DIRECTORY / "run_metadata.json"
VTK_TIME_SERIES_DIRECTORY = OUTPUT_DIRECTORY / "vtk"
VTK_SERIES_PATH = OUTPUT_DIRECTORY / "dfc_3d_printing.vtk.series"
AUGER_CONTACT_PROXY_VTK_PATH = OUTPUT_DIRECTORY / "auger_contact_proxy.vtk"
MOVIE_FPS = RENDER_FPS

# ── Derived constants ────────────────────────────────────────────────────────
FRAME_DT = 1.0 / RENDER_FPS
SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
WARMUP_STEPS = int(round(WARMUP_SECONDS / NEWTON_DT))


def _quat_rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return vector + 2.0 * np.cross(
        quaternion[:3],
        np.cross(quaternion[:3], vector) + quaternion[3] * vector,
    )


def _quat_multiply_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float32,
    )


def _gcode_offset(program, program_time: float) -> np.ndarray:
    """Return the configured world-axis offset from the G-code path origin."""
    path_origin = np.asarray(program.moves[0].start, dtype=np.float32)
    path_position = np.asarray(program.position_at(program_time), dtype=np.float32)
    return (path_position - path_origin) * np.asarray(GCODE_TO_WORLD_SCALE, dtype=np.float32)


def _configure_dfc_solver(solver):
    """Install the shared DFC force law and return its MiniSlump material."""
    solver.SetJitifyClumpTemplates(False)
    solver.SetJitifyMassProperties(False)
    force_model = solver.ReadContactForceModel(str(DFC_FORCE_MODEL_PATH))
    force_model.SetMustHaveMatProp(set())
    force_model.SetMustPairwiseMatProp(set())
    force_model.SetPerContactWildcards(
        {
            "contact_info_step_time",
            "contact_info_lambda",
            "contact_info_strain_x",
            "contact_info_strain_y",
            "contact_info_strain_z",
        }
    )
    force_model.SetPerOwnerWildcards({"ParticleThixoAvg"})
    force_model.DefineCustomModelPrerequisites(DFC_MODEL_PREREQUISITES)
    return solver.LoadMaterial(DEME_DFC_MATERIAL)


def _mini_slump_particle_radii(particle_count: int) -> np.ndarray:
    """Return deterministic mortar-coated radii following MiniSlump grading."""
    quantiles = (np.arange(particle_count, dtype=np.float64) + 0.5) / particle_count
    # Fuller grading describes cumulative aggregate volume. Dividing each size
    # class by particle volume converts it to the corresponding number density.
    number_distribution_power = DEME_AGGREGATE_GRADING_EXPONENT - 3.0
    diameter_min_power = DEME_AGGREGATE_DIAMETER_MIN**number_distribution_power
    diameter_max_power = DEME_AGGREGATE_DIAMETER_MAX**number_distribution_power
    aggregate_diameters = (diameter_min_power + quantiles * (diameter_max_power - diameter_min_power)) ** (
        1.0 / number_distribution_power
    )
    np.random.default_rng(DEME_PARTICLE_GRADING_SEED).shuffle(aggregate_diameters)
    return aggregate_diameters / 2.0 + DEME_MORTAR_LAYER


DFC_MODEL_PREREQUISITES = r"""
void __device__ XdirToDxDyDz(const float3 &Vxdir, const float3 &Vsingular,
                             float3 &Vx, float3 &Vy, float3 &Vz) {
    if (length(Vxdir) < 1e-8) {
        Vx = make_float3(1, 0, 0);
    } else {
        Vx = normalize(Vxdir);
    }
    Vz = cross(Vx, Vsingular);
    if (length(Vz) < 1e-6) {
        float3 fallback;
        if (abs(Vsingular.z) < 0.9)
            fallback = make_float3(0, 0, 1);
        else if (abs(Vsingular.y) < 0.9)
            fallback = make_float3(0, 1, 0);
        else
            fallback = make_float3(1, 0, 0);
        Vz = cross(Vx, fallback);
    }
    Vz = normalize(Vz);
    Vy = cross(Vz, Vx);
}
"""


def _write_vtk_frame(
    path: Path,
    body_transforms: np.ndarray,
    particle_positions: np.ndarray,
    particle_radii: np.ndarray,
) -> None:
    """Export the current printer, build plate, and DEME particles for ParaView."""
    world_vertices = []
    triangles = []
    part_ids = []
    point_part_ids = []
    contact_mesh_flags = []
    deme_build_plate_flags = []
    point_offset = 0
    for part_id, mesh in enumerate(printer.meshes):
        body_transform = body_transforms[mesh.body_index]
        body_position = body_transform[:3]
        body_rotation = body_transform[3:7]
        transformed = np.asarray(
            [body_position + _quat_rotate_xyzw(body_rotation, vertex) for vertex in mesh.vertices_local],
            dtype=np.float64,
        )
        world_vertices.append(transformed)
        point_part_ids.extend([part_id] * len(transformed))
        triangles.append(mesh.triangles + point_offset)
        part_ids.extend([part_id] * len(mesh.triangles))
        contact_mesh_flags.extend([int(mesh.is_nozzle_contact_mesh)] * len(mesh.triangles))
        deme_build_plate_flags.extend([0] * len(mesh.triangles))
        point_offset += len(transformed)

    plate_center = np.asarray(BUILD_PLATE_CENTER, dtype=np.float64)
    hx, hy, hz = BUILD_PLATE_HALF_EXTENTS
    plate_vertices = plate_center + np.asarray(
        [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ]
    )

    plate_triangles = np.asarray(
        [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ],
        dtype=np.int64,
    )
    world_vertices.append(plate_vertices)
    point_part_ids.extend([len(printer.meshes)] * len(plate_vertices))
    triangles.append(plate_triangles + point_offset)
    part_ids.extend([len(printer.meshes)] * len(plate_triangles))
    contact_mesh_flags.extend([0] * len(plate_triangles))
    deme_build_plate_flags.extend([int(ENABLE_DEME_BUILD_PLATE)] * len(plate_triangles))
    mesh_point_count = point_offset + len(plate_vertices)
    # Emit physical sphere surfaces so particles open as spheres in ParaView
    # without requiring a Glyph filter or reader-specific point sizing.
    particle_vertices, particle_triangles, particle_ids = triangulate_spheres(
        particle_positions,
        particle_radii,
        latitude_segments=VTK_PARTICLE_LATITUDE_SEGMENTS,
        longitude_segments=VTK_PARTICLE_LONGITUDE_SEGMENTS,
    )
    world_vertices.append(particle_vertices)
    point_part_ids.extend([-1] * len(particle_vertices))
    triangles.append(particle_triangles + mesh_point_count)
    part_ids.extend([-1] * len(particle_triangles))
    contact_mesh_flags.extend([0] * len(particle_triangles))
    deme_build_plate_flags.extend([0] * len(particle_triangles))
    point_radii = np.concatenate(
        (
            np.zeros(mesh_point_count, dtype=np.float64),
            particle_radii[particle_ids],
        )
    )
    vtk_particle_ids = np.concatenate((np.full(mesh_point_count, -1, dtype=np.int64), particle_ids))
    write_vtk_polydata(
        path,
        np.concatenate(world_vertices),
        np.concatenate(triangles),
        cell_scalars={
            "scene_part_id": part_ids,
            "is_nozzle_contact_mesh": contact_mesh_flags,
            "is_deme_build_plate": deme_build_plate_flags,
        },
        point_scalars={
            "scene_part_id": point_part_ids,
            "particle_id": vtk_particle_ids,
            "particle_radius": point_radii,
        },
        title="MoPhi ABB IRB6700 printer frame",
    )


@wp.kernel
def _transform_nozzle_contact_lines(
    body_q: wp.array(dtype=wp.transform),
    body_index: int,
    local_starts: wp.array(dtype=wp.vec3),
    local_ends: wp.array(dtype=wp.vec3),
    world_starts: wp.array(dtype=wp.vec3),
    world_ends: wp.array(dtype=wp.vec3),
):
    line_index = wp.tid()
    body_transform = body_q[body_index]
    world_starts[line_index] = wp.transform_point(body_transform, local_starts[line_index])
    world_ends[line_index] = wp.transform_point(body_transform, local_ends[line_index])


@wp.kernel
def _transform_deme_particle_positions(
    deme_positions: wp.array(dtype=wp.vec3),
    world_positions: wp.array(dtype=wp.vec3),
    world_origin: wp.vec3,
    scale: float,
):
    particle_index = wp.tid()
    world_positions[particle_index] = world_origin + scale * deme_positions[particle_index]


if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("FRAME_DT must be an integer multiple of NEWTON_DT")
if GCODE_TIME_SCALE <= 0.0:
    raise ValueError("GCODE_TIME_SCALE must be positive")
if PATH_VISUAL_SAMPLES < 2:
    raise ValueError("PATH_VISUAL_SAMPLES must be at least two")
if not GCODE_PATH.is_file():
    mophi.fatal(f"Required G-code path is missing: {GCODE_PATH}")
if not PRINTER_ASSET_DIRECTORY.is_dir():
    mophi.fatal(f"Required ABB printer assets are missing: {PRINTER_ASSET_DIRECTORY}")
if not DFC_FORCE_MODEL_PATH.is_file():
    mophi.fatal(f"Required DFC contact model is missing: {DFC_FORCE_MODEL_PATH}")

mophi.check_newton_warp_mujoco_versions(
    mophi.REQUIRED_NEWTON_VERSION,
    mophi.REQUIRED_WARP_VERSION,
    mophi.REQUIRED_MUJOCO_VERSION,
)

program = parse_gcode(GCODE_PATH, rapid_feed_rate=GCODE_RAPID_FEED_RATE)
if not program.moves:
    mophi.fatal(f"G-code path contains no supported motion: {GCODE_PATH}")
path_seconds = program.duration * GCODE_TIME_SCALE
motion_frames = int(math.ceil(path_seconds / FRAME_DT))
hold_frames = int(round(POST_PATH_HOLD_SECONDS / FRAME_DT))

print("=== MoPhi DFC particle-settling and flow stage ===")
print(f"[G-code] Loaded {len(program.moves)} moves ({program.duration:.3f} s nominal).")
wp.init()
device = wp.get_device()
if not device.is_cuda:
    mophi.fatal("The DFC particle-flow simulation requires a CUDA device.")

print("[Newton] Reconstructing the ABB IRB6700 and auger print head ...")
try:
    printer = build_abb_irb6700_printer(
        PRINTER_ASSET_DIRECTORY,
        JOINT_TARGET_STIFFNESS,
        JOINT_TARGET_DAMPING,
        AUGER_TARGET_STIFFNESS,
        AUGER_TARGET_DAMPING,
    )
except (FileNotFoundError, ValueError) as exc:
    mophi.fatal(str(exc))
builder = printer.builder
nozzle_body_idx = printer.nozzle_body_index
nozzle_position_local = np.asarray(printer.nozzle_position_local, dtype=np.float32)
nozzle_rotation_local = np.asarray(printer.nozzle_rotation_local, dtype=np.float32)
print(
    "[Newton] Nozzle contact mesh: "
    f"{printer.nozzle_contact_vertex_count} vertices, "
    f"{printer.nozzle_contact_triangle_count} triangles, "
    f"{printer.nozzle_contact_edge_count} manifold edges."
)

print("[DEME] Sampling DFC stock inside the auger contact mesh ...")
deme_solver = DEME.DEMSolver([device.ordinal])
if device.ordinal not in deme_solver.GetGPUDeviceIDs():
    mophi.fatal(f"DEME workers {deme_solver.GetGPUDeviceIDs()} do not share Warp CUDA device {device.ordinal}.")
dfc_material = _configure_dfc_solver(deme_solver)
nozzle_body_position_world = np.asarray(printer.nozzle_body_position_world, dtype=np.float64)
nozzle_body_rotation_world = np.asarray(printer.nozzle_body_rotation_world, dtype=np.float64)
nozzle_contact_position_world = nozzle_body_position_world + _quat_rotate_xyzw(
    nozzle_body_rotation_world,
    np.asarray(printer.nozzle_contact_position_local, dtype=np.float64),
)
nozzle_contact_rotation_world = _quat_multiply_xyzw(
    nozzle_body_rotation_world,
    np.asarray(printer.nozzle_contact_rotation_local, dtype=np.float64),
)
contact_surface = mophi.load_obj(str(printer.nozzle_contact_mesh_path))
# Sample in contact-mesh-local coordinates. The ray axis must cross opposing
# inner walls; the explicit bounds clip the open chamber in the other axes.
particle_positions_local = sample_grid_between_mesh_surfaces(
    contact_surface.vertices,
    np.asarray(contact_surface.indices, dtype=np.int64).reshape((-1, 3)),
    DEME_PARTICLE_SPACING * DEME_TO_WORLD_SCALE,
    DEME_AUGER_SAMPLE_BOUNDS_MIN,
    DEME_AUGER_SAMPLE_BOUNDS_MAX,
    ray_axis=DEME_AUGER_SAMPLE_RAY_AXIS,
    clearance=DEME_AUGER_SAMPLE_CLEARANCE,
)
if len(particle_positions_local) == 0:
    mophi.fatal("The configured auger mesh sampling region produced no particles.")
particle_radii = _mini_slump_particle_radii(len(particle_positions_local))
particle_masses = DEME_CONCRETE_DENSITY * DEME_PARTICLE_MASS_DENSITY_SCALE * (4.0 / 3.0) * np.pi * particle_radii**3
# MiniSlump creates one sphere type per graded particle because both radius and
# mass vary across the aggregate distribution.
particle_templates = [
    deme_solver.LoadSphereType(float(mass), float(radius), dfc_material)
    for mass, radius in zip(particle_masses, particle_radii)
]
particle_positions_world = np.asarray(
    [
        nozzle_contact_position_world + _quat_rotate_xyzw(nozzle_contact_rotation_world, position)
        for position in particle_positions_local
    ],
    dtype=np.float64,
)
particle_positions = (particle_positions_world - np.asarray(DEME_VISUAL_ORIGIN, dtype=np.float64)) / DEME_TO_WORLD_SCALE
deme_particles = deme_solver.AddClumps(particle_templates, particle_positions.tolist())
deme_particles.SetFamily(DEME_PARTICLE_FAMILY)
deme_particles.SetOwnerWildcards({"ParticleThixoAvg": [0.0] * len(particle_positions)})
deme_particle_tracker = deme_solver.Track(deme_particles)

# A +X side plane keeps stock near the auger wall while the -Z-facing top
# plane closes the chamber above it. The auger mesh itself supports the stock
# from below; both temporary planes are disabled after settling.
settling_boundary_margin = DEME_AUGER_SAMPLE_CLEARANCE / DEME_TO_WORLD_SCALE
settling_min_x = float(particle_positions[:, 0].min() - settling_boundary_margin)
settling_top_z = float(particle_positions[:, 2].max() + settling_boundary_margin)
deme_settling_boundaries = deme_solver.AddExternalObject()
deme_settling_boundaries.AddPlane((settling_min_x, 0.0, 0.0), (1.0, 0.0, 0.0), dfc_material)
deme_settling_boundaries.AddPlane((0.0, 0.0, settling_top_z), (0.0, 0.0, -1.0), dfc_material)
deme_settling_boundaries.SetFamily(DEME_SETTLING_BOUNDARY_FAMILY)
deme_solver.SetFamilyFixed(DEME_SETTLING_BOUNDARY_FAMILY)

# DEME does not see Newton collision geometry. Add a persistent analytical
# plane at the finite Newton build plate's top surface so discharged material
# accumulates at the same visible height instead of leaving the DEME domain.
build_plate_top_world_z = BUILD_PLATE_CENTER[2] + BUILD_PLATE_HALF_EXTENTS[2]
build_plate_top_deme_z = (build_plate_top_world_z - DEME_VISUAL_ORIGIN[2]) / DEME_TO_WORLD_SCALE
deme_build_plate = None
if ENABLE_DEME_BUILD_PLATE:
    deme_build_plate = deme_solver.AddExternalObject()
    deme_build_plate.AddPlane((0.0, 0.0, build_plate_top_deme_z), (0.0, 0.0, 1.0), dfc_material)
    deme_build_plate.SetFamily(DEME_BUILD_PLATE_FAMILY)
    deme_solver.SetFamilyFixed(DEME_BUILD_PLATE_FAMILY)
    print(
        f"[DEME] Enabled analytical build plate at world z={build_plate_top_world_z:g} m "
        f"(DEME z={build_plate_top_deme_z:g} mm)."
    )
else:
    print("[DEME] Analytical build plate disabled.")

# Register the same inner-surface contact mesh used by Newton.
nozzle_contact_position_deme = (nozzle_contact_position_world - np.asarray(DEME_VISUAL_ORIGIN)) / DEME_TO_WORLD_SCALE
deme_nozzle_proxy = deme_solver.AddWavefrontMeshObject(
    str(printer.nozzle_contact_mesh_path),
    dfc_material,
)
# Partition connected triangles at sharper local-normal changes so DEME can
# treat the curved inner surface as smaller contact patches.
deme_nozzle_patch_count = deme_nozzle_proxy.SplitIntoConvexPatches(DEME_NOZZLE_PROXY_PATCH_ANGLE_DEGREES)
deme_nozzle_proxy.Scale(1.0 / DEME_TO_WORLD_SCALE)
deme_nozzle_proxy.SetInitPos(nozzle_contact_position_deme.tolist())
deme_nozzle_proxy.SetInitQuat(nozzle_contact_rotation_world.tolist())
deme_nozzle_proxy.SetFamily(DEME_NOZZLE_PROXY_FAMILY)
deme_nozzle_tracker = deme_solver.Track(deme_nozzle_proxy)
deme_solver.SetFamilyFixed(DEME_NOZZLE_PROXY_FAMILY)
print(
    f"[DEME] Split the auger contact proxy into {deme_nozzle_patch_count} convex patches "
    f"at {DEME_NOZZLE_PROXY_PATCH_ANGLE_DEGREES:g} degrees."
)
deme_solver.InstructBoxDomainDimension(
    DEME_BROAD_DOMAIN_X,
    DEME_BROAD_DOMAIN_Y,
    DEME_BROAD_DOMAIN_Z,
)
deme_solver.SetGravitationalAcceleration(DEME_GRAVITY)
deme_solver.SetInitTimeStep(DEME_DT_INITIAL)
deme_solver.SetInitBinSize(DEME_INITIAL_BIN_SIZE)
deme_solver.SetErrorOutVelocity(DEME_ERROR_OUT_VELOCITY)
deme_solver.SetErrorOutAngularVelocity(DEME_ERROR_OUT_VELOCITY)
deme_solver.SetErrorOutAvgContacts(1000)
deme_solver.Initialize()
deme_kinetic_energy = deme_solver.CreateInspector("clump_kinetic_energy")
deme_particle_count = len(particle_positions)
deme_particle_exchange = NewtonDEMEParticleExchange()
deme_particle_exchange.initialize(
    deme_solver,
    int(deme_particle_tracker.GetOwnerID()),
    deme_particle_count,
    device,
)
print(
    f"[DEME] Initialized {deme_particle_count} graded DFC spheres inside the confined auger chamber "
    f"(r={particle_radii.min():.3f}--{particle_radii.max():.3f} mm)."
)

builder.add_ground_plane()
plate_cfg = newton.ModelBuilder.ShapeConfig(has_particle_collision=True)
builder.add_shape_box(
    -1,
    xform=wp.transform(wp.vec3(*BUILD_PLATE_CENTER)),
    hx=BUILD_PLATE_HALF_EXTENTS[0],
    hy=BUILD_PLATE_HALF_EXTENTS[1],
    hz=BUILD_PLATE_HALF_EXTENTS[2],
    cfg=plate_cfg,
    label="build_plate",
)

model = builder.finalize()
model.set_gravity(NEWTON_GRAVITY)
solver = newton.solvers.SolverMuJoCo(model, disable_contacts=True)
state_0 = model.state()
state_1 = model.state()
control = model.control()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
wp.copy(control.joint_target_pos, model.joint_q)

initial_nozzle_body_transform = state_0.body_q.numpy()[nozzle_body_idx]
initial_nozzle_body_rotation = initial_nozzle_body_transform[3:7].astype(np.float32)
initial_nozzle_position = initial_nozzle_body_transform[:3] + _quat_rotate_xyzw(
    initial_nozzle_body_rotation,
    nozzle_position_local,
)
initial_nozzle_rotation = _quat_multiply_xyzw(initial_nozzle_body_rotation, nozzle_rotation_local)

position_objective = ik.IKObjectivePosition(
    link_index=nozzle_body_idx,
    link_offset=wp.vec3(*nozzle_position_local),
    target_positions=wp.array([initial_nozzle_position], dtype=wp.vec3, device=device),
)
rotation_objective = ik.IKObjectiveRotation(
    link_index=nozzle_body_idx,
    link_offset_rotation=wp.quat(*nozzle_rotation_local),
    target_rotations=wp.array([initial_nozzle_rotation], dtype=wp.vec4, device=device),
)
joint_limit_objective = ik.IKObjectiveJointLimit(
    joint_limit_lower=model.joint_limit_lower,
    joint_limit_upper=model.joint_limit_upper,
    weight=10.0,
)
ik_joint_q = wp.clone(model.joint_q.reshape((1, model.joint_coord_count)))
ik_solver = ik.IKSolver(
    model=model,
    n_problems=1,
    objectives=[position_objective, rotation_objective, joint_limit_objective],
    lambda_initial=IK_DAMPING,
    jacobian_mode=ik.IKJacobianType.ANALYTIC,
)

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
if SAVE_AUGER_CONTACT_PROXY_VTK:
    # Export the exact DEME/Newton proxy in its initial world pose as a
    # standalone surface for direct inspection in ParaView.
    contact_triangles = np.asarray(contact_surface.indices, dtype=np.int64).reshape((-1, 3))
    contact_vertices_world = np.asarray(
        [
            nozzle_contact_position_world + _quat_rotate_xyzw(nozzle_contact_rotation_world, vertex)
            for vertex in contact_surface.vertices
        ],
        dtype=np.float64,
    )
    write_vtk_polydata(
        AUGER_CONTACT_PROXY_VTK_PATH,
        contact_vertices_world,
        contact_triangles,
        cell_scalars={"is_nozzle_contact_mesh": np.ones(len(contact_triangles), dtype=np.int64)},
        title="MoPhi auger inner-surface contact proxy",
    )
    print(f"[VTK] Auger contact proxy: {AUGER_CONTACT_PROXY_VTK_PATH}")
if SAVE_VTK_TIME_SERIES:
    VTK_TIME_SERIES_DIRECTORY.mkdir(parents=True, exist_ok=True)
if USE_OMNIVERSE_VISUALIZATION:
    vis = mophi.OmniverseVisualizer(output_path=str(USD_OUTPUT_PATH), fps=RENDER_FPS)
else:
    vis = mophi.OpenGLVisualizer(model)
camera_to_nozzle = initial_nozzle_position - np.asarray(CAMERA_POSITION)
camera_yaw = np.degrees(np.arctan2(camera_to_nozzle[1], camera_to_nozzle[0]))
camera_pitch = np.degrees(np.arctan2(camera_to_nozzle[2], np.hypot(camera_to_nozzle[0], camera_to_nozzle[1])))
vis.set_camera(pos=wp.vec3(*CAMERA_POSITION), pitch=camera_pitch, yaw=camera_yaw)
mophi.log_orientation_and_scale_reference(
    vis,
    axis_origin=REFERENCE_AXIS_ORIGIN,
    scale_bar_center=REFERENCE_SCALE_BAR_CENTER,
)

path_times = np.linspace(0.0, program.duration, PATH_VISUAL_SAMPLES)
path_points = np.asarray([initial_nozzle_position + _gcode_offset(program, time) for time in path_times])
path_starts = wp.array(path_points[:-1], dtype=wp.vec3, device=device)
path_ends = wp.array(path_points[1:], dtype=wp.vec3, device=device)
path_colors = wp.full(len(path_points) - 1, PATH_VISUAL_COLOR, dtype=wp.vec3, device=device)

nozzle_contact_local_starts = wp.array(printer.nozzle_contact_line_starts_local, dtype=wp.vec3, device=device)
nozzle_contact_local_ends = wp.array(printer.nozzle_contact_line_ends_local, dtype=wp.vec3, device=device)
nozzle_contact_world_starts = wp.empty_like(nozzle_contact_local_starts)
nozzle_contact_world_ends = wp.empty_like(nozzle_contact_local_ends)
nozzle_contact_colors = wp.full(
    len(printer.nozzle_contact_line_starts_local),
    NOZZLE_CONTACT_WIREFRAME_COLOR,
    dtype=wp.vec3,
    device=device,
)
# The analytical plane is infinite; draw its finite Newton plate footprint and
# center cross slightly above the surface so the marker remains visible.
plate_x, plate_y, _ = BUILD_PLATE_CENTER
plate_hx, plate_hy, _ = BUILD_PLATE_HALF_EXTENTS
plate_visual_z = build_plate_top_world_z + DEME_BUILD_PLATE_VISUAL_Z_OFFSET
deme_build_plate_visual_starts = wp.array(
    [
        (plate_x - plate_hx, plate_y - plate_hy, plate_visual_z),
        (plate_x + plate_hx, plate_y - plate_hy, plate_visual_z),
        (plate_x + plate_hx, plate_y + plate_hy, plate_visual_z),
        (plate_x - plate_hx, plate_y + plate_hy, plate_visual_z),
        (plate_x - plate_hx, plate_y, plate_visual_z),
        (plate_x, plate_y - plate_hy, plate_visual_z),
    ],
    dtype=wp.vec3,
    device=device,
)
deme_build_plate_visual_ends = wp.array(
    [
        (plate_x + plate_hx, plate_y - plate_hy, plate_visual_z),
        (plate_x + plate_hx, plate_y + plate_hy, plate_visual_z),
        (plate_x - plate_hx, plate_y + plate_hy, plate_visual_z),
        (plate_x - plate_hx, plate_y - plate_hy, plate_visual_z),
        (plate_x + plate_hx, plate_y, plate_visual_z),
        (plate_x, plate_y + plate_hy, plate_visual_z),
    ],
    dtype=wp.vec3,
    device=device,
)
deme_build_plate_visual_colors = wp.full(
    len(deme_build_plate_visual_starts),
    DEME_BUILD_PLATE_VISUAL_COLOR,
    dtype=wp.vec3,
    device=device,
)
deme_particle_world_positions = wp.empty(deme_particle_count, dtype=wp.vec3, device=device)
particle_radii_world = particle_radii * DEME_TO_WORLD_SCALE
deme_particle_radii = wp.array(particle_radii_world, dtype=wp.float32, device=device)
deme_particle_colors = wp.full(deme_particle_count, DEME_PARTICLE_COLOR, dtype=wp.vec3, device=device)

movie_writer = None
if SAVE_MOVIE and not USE_OMNIVERSE_VISUALIZATION:
    movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS)
vtk_frames = []


def _set_gcode_target(program_time: float) -> None:
    target_position = initial_nozzle_position + _gcode_offset(program, program_time)
    position_objective.set_target_position(0, wp.vec3(*target_position))
    ik_solver.step(ik_joint_q, ik_joint_q, iterations=IK_ITERATIONS)
    wp.copy(control.joint_target_pos, ik_joint_q.flatten())


def _step_newton() -> None:
    global state_0, state_1
    state_0.clear_forces()
    solver.step(state_0, state_1, control, None, NEWTON_DT)
    state_0, state_1 = state_1, state_0


def _render(frame_time: float) -> None:
    deme_particle_positions, _ = deme_particle_exchange.read_deme_particle_state()
    wp.launch(
        _transform_deme_particle_positions,
        dim=deme_particle_count,
        inputs=[
            deme_particle_positions,
            deme_particle_world_positions,
            wp.vec3(*DEME_VISUAL_ORIGIN),
            DEME_TO_WORLD_SCALE,
        ],
        device=device,
    )
    vis.begin_frame(frame_time)
    vis.log_state(state_0)
    vis.log_points(
        "dfc_particles",
        deme_particle_world_positions,
        radii=deme_particle_radii,
        colors=deme_particle_colors,
    )
    vis.log_lines(
        "gcode_toolpath",
        path_starts,
        path_ends,
        path_colors,
        width=PATH_VISUAL_WIDTH,
    )
    if ENABLE_DEME_BUILD_PLATE and SHOW_DEME_BUILD_PLATE:
        vis.log_lines(
            "deme_analytical_build_plate",
            deme_build_plate_visual_starts,
            deme_build_plate_visual_ends,
            deme_build_plate_visual_colors,
            width=DEME_BUILD_PLATE_VISUAL_WIDTH,
        )
    if SHOW_NOZZLE_CONTACT_WIREFRAME:
        wp.launch(
            _transform_nozzle_contact_lines,
            dim=len(nozzle_contact_local_starts),
            inputs=[
                state_0.body_q,
                nozzle_body_idx,
                nozzle_contact_local_starts,
                nozzle_contact_local_ends,
            ],
            outputs=[nozzle_contact_world_starts, nozzle_contact_world_ends],
            device=device,
        )
        vis.log_lines(
            "nozzle_contact_mesh_wireframe",
            nozzle_contact_world_starts,
            nozzle_contact_world_ends,
            nozzle_contact_colors,
            width=NOZZLE_CONTACT_WIREFRAME_WIDTH,
        )
    vis.end_frame()
    if movie_writer is not None:
        movie_writer.append_data(vis.get_frame().numpy())
    if SAVE_VTK_TIME_SERIES:
        vtk_frame_path = VTK_TIME_SERIES_DIRECTORY / f"frame_{len(vtk_frames):06d}.vtk"
        _write_vtk_frame(
            vtk_frame_path,
            state_0.body_q.numpy(),
            deme_particle_world_positions.numpy(),
            particle_radii_world,
        )
        vtk_frames.append((frame_time, vtk_frame_path))


sim_time = 0.0
frames_completed = 0
settling_frames_completed = 0
settling_checks = 0
settling_low_energy_checks = 0
settling_converged = False
settling_final_kinetic_energy = float("inf")
settling_time = 0.0
discharge_time = 0.0
discharge_frames_completed = 0
discharge_final_kinetic_energy = float("nan")
release_total_contacts = 0
release_nozzle_contacts = 0
try:
    print("[DEME] Settling until the DFC stock maintains low kinetic energy ...")
    next_energy_check_time = DEME_SETTLING_CHECK_INTERVAL
    next_render_time = FRAME_DT
    current_deme_dt = DEME_DT_INITIAL
    while settling_time < DEME_SETTLING_MAX_SECONDS:
        if not vis.is_running():
            break
        chunk_duration = (
            DEME_DT_GROWTH_STEP_COUNT * current_deme_dt
            if current_deme_dt < DEME_DT_MAX
            else DEME_SETTLING_CHECK_INTERVAL
        )
        chunk_duration = min(
            chunk_duration,
            DEME_SETTLING_MAX_SECONDS - settling_time,
            next_energy_check_time - settling_time,
            next_render_time - settling_time,
        )
        deme_solver.DoDynamics(chunk_duration)
        settling_time += chunk_duration
        sim_time += chunk_duration
        if current_deme_dt < DEME_DT_MAX:
            current_deme_dt = min(current_deme_dt * DEME_DT_GROWTH_FACTOR, DEME_DT_MAX)
            deme_solver.UpdateStepSize(current_deme_dt)

        if settling_time + 1.0e-12 >= next_render_time:
            if RENDER_SETTLING_PHASE:
                _render(sim_time)
                settling_frames_completed += 1
                frames_completed += 1
            next_render_time += FRAME_DT

        if settling_time + 1.0e-12 >= next_energy_check_time:
            settling_final_kinetic_energy = float(deme_kinetic_energy.GetValue())
            settling_checks += 1
            if (
                settling_time >= DEME_SETTLING_MIN_SECONDS
                and settling_final_kinetic_energy <= DEME_SETTLING_KINETIC_ENERGY_THRESHOLD
            ):
                settling_low_energy_checks += 1
            else:
                settling_low_energy_checks = 0
            print(
                f"[DEME] settle t={settling_time:.3f} s, dt={current_deme_dt:.3e} s, "
                f"KE={settling_final_kinetic_energy:.6e}, "
                f"low-energy checks={settling_low_energy_checks}/{DEME_SETTLING_REQUIRED_LOW_ENERGY_CHECKS}"
            )
            next_energy_check_time += DEME_SETTLING_CHECK_INTERVAL
            if settling_low_energy_checks >= DEME_SETTLING_REQUIRED_LOW_ENERGY_CHECKS:
                settling_converged = True
                break

    if not settling_converged:
        mophi.fatal(
            "DFC particles did not settle below the configured kinetic-energy threshold "
            f"within {DEME_SETTLING_MAX_SECONDS} s; final KE={settling_final_kinetic_energy:.6e}."
        )
    print(f"[DEME] Settling converged at t={settling_time:.3f} s.")

    if RUN_PARTICLE_DISCHARGE_AFTER_SETTLING:
        print("[DEME] Disabling the settling boundaries and beginning in-place auger flow ...")
        deme_solver.DisableContactBetweenFamilies(DEME_PARTICLE_FAMILY, DEME_SETTLING_BOUNDARY_FAMILY)
        deme_solver.UpdateStepSize(DEME_DT_INITIAL)
        deme_solver.DoDynamicsThenSync(0.0)
        release_total_contacts = deme_solver.GetNumContacts()
        release_nozzle_contacts = len(deme_solver.GetOwnerContactClumps(int(deme_nozzle_tracker.GetOwnerID())))
        print(
            f"[DEME] Released in-place stock with {release_total_contacts} total and "
            f"{release_nozzle_contacts} nozzle contact(s).",
            flush=True,
        )

        # Reintroduce the small DFC timestep when the retaining boundaries
        # disappear, then grow it with the stable settling schedule.
        current_discharge_dt = DEME_DT_INITIAL
        next_discharge_render_time = FRAME_DT
        while discharge_time < DEME_DISCHARGE_SECONDS:
            if not vis.is_running():
                break
            chunk_duration = (
                DEME_DT_GROWTH_STEP_COUNT * current_discharge_dt
                if current_discharge_dt < DEME_DISCHARGE_DT_MAX
                else FRAME_DT
            )
            chunk_duration = min(
                chunk_duration,
                DEME_DISCHARGE_SECONDS - discharge_time,
                next_discharge_render_time - discharge_time,
            )
            deme_solver.DoDynamics(chunk_duration)
            discharge_time += chunk_duration
            sim_time += chunk_duration
            if current_discharge_dt < DEME_DISCHARGE_DT_MAX:
                current_discharge_dt = min(
                    current_discharge_dt * DEME_DT_GROWTH_FACTOR,
                    DEME_DISCHARGE_DT_MAX,
                )
                deme_solver.UpdateStepSize(current_discharge_dt)
            if discharge_time + 1.0e-12 >= next_discharge_render_time:
                _render(sim_time)
                discharge_frames_completed += 1
                frames_completed += 1
                next_discharge_render_time += FRAME_DT
                if discharge_frames_completed % 5 == 0:
                    print(
                        f"[DEME] discharge t={discharge_time:.3f} s, dt={current_discharge_dt:.3e} s, "
                        f"KE={float(deme_kinetic_energy.GetValue()):.6e}",
                        flush=True,
                    )
        discharge_final_kinetic_energy = float(deme_kinetic_energy.GetValue())

    _set_gcode_target(0.0)
    for warmup_step in range(WARMUP_STEPS if RUN_GCODE_MOTION_AFTER_SETTLING else 0):
        _step_newton()
        sim_time += NEWTON_DT
        settling_frame_due = (warmup_step + 1) % SIM_SUBSTEPS == 0 or warmup_step + 1 == WARMUP_STEPS
        if RENDER_SETTLING_PHASE and settling_frame_due:
            if not vis.is_running():
                break
            _render(sim_time)
            settling_frames_completed += 1
            frames_completed += 1

    for frame in range(motion_frames + hold_frames if RUN_GCODE_MOTION_AFTER_SETTLING else 0):
        if not vis.is_running():
            break
        path_time = min((frame + 1) * FRAME_DT / GCODE_TIME_SCALE, program.duration)
        _set_gcode_target(path_time)
        for _ in range(SIM_SUBSTEPS):
            _step_newton()
        sim_time += FRAME_DT
        _render(sim_time)
        frames_completed += 1
finally:
    if movie_writer is not None:
        movie_writer.close()
    vis.close()
    deme_particle_exchange.finalize()
    if SAVE_VTK_TIME_SERIES and vtk_frames:
        write_vtk_file_series(VTK_SERIES_PATH, vtk_frames)

final_nozzle_body_transform = state_0.body_q.numpy()[nozzle_body_idx]
final_nozzle_position = final_nozzle_body_transform[:3] + _quat_rotate_xyzw(
    final_nozzle_body_transform[3:7],
    nozzle_position_local,
)
requested_final_nozzle_position = initial_nozzle_position.copy()
if RUN_GCODE_MOTION_AFTER_SETTLING:
    requested_final_nozzle_position += _gcode_offset(program, program.duration)
final_nozzle_error = float(np.linalg.norm(final_nozzle_position - requested_final_nozzle_position))
METADATA_OUTPUT_PATH.write_text(
    json.dumps(
        {
            "phase": "dfc_settling_and_gravity_fed_nozzle_discharge",
            "printer_model": "chrono_reference_abb_irb6700_with_auger",
            "printer_asset_directory": str(PRINTER_ASSET_DIRECTORY),
            "gcode_path": str(GCODE_PATH),
            "gcode_move_count": len(program.moves),
            "gcode_nominal_duration": program.duration,
            "gcode_time_scale": GCODE_TIME_SCALE,
            "newton_dt": NEWTON_DT,
            "render_fps": RENDER_FPS,
            "ik_iterations": IK_ITERATIONS,
            "frames_completed": frames_completed,
            "settling_frames_completed": settling_frames_completed,
            "render_settling_phase": RENDER_SETTLING_PHASE,
            "requested_final_nozzle_position": requested_final_nozzle_position.tolist(),
            "actual_final_nozzle_position": final_nozzle_position.tolist(),
            "final_nozzle_position_error": final_nozzle_error,
            "nozzle_contact_mesh_visible": SHOW_NOZZLE_CONTACT_WIREFRAME,
            "nozzle_contact_mesh_vertices": printer.nozzle_contact_vertex_count,
            "nozzle_contact_mesh_triangles": printer.nozzle_contact_triangle_count,
            "nozzle_contact_mesh_edges": printer.nozzle_contact_edge_count,
            "nozzle_contact_mesh_closed_two_manifold": False,
            "auger_contact_proxy_vtk_saved": SAVE_AUGER_CONTACT_PROXY_VTK,
            "auger_contact_proxy_vtk_path": (
                str(AUGER_CONTACT_PROXY_VTK_PATH) if SAVE_AUGER_CONTACT_PROXY_VTK else None
            ),
            "vtk_time_series_saved": SAVE_VTK_TIME_SERIES,
            "vtk_time_series_frame_count": len(vtk_frames),
            "vtk_series_path": str(VTK_SERIES_PATH) if SAVE_VTK_TIME_SERIES else None,
            "vtk_scene_part_ids": {
                **{str(index): mesh.label for index, mesh in enumerate(printer.meshes)},
                str(len(printer.meshes)): "build_plate",
            },
            "deme_enabled": True,
            "dfc_enabled": True,
            "deme_particle_count": deme_particle_count,
            "deme_dfc_calibration": "Chrono Demo_DFC_MiniSlump",
            "deme_aggregate_diameter_range": [
                DEME_AGGREGATE_DIAMETER_MIN,
                DEME_AGGREGATE_DIAMETER_MAX,
            ],
            "deme_aggregate_grading_exponent": DEME_AGGREGATE_GRADING_EXPONENT,
            "deme_particle_grading_seed": DEME_PARTICLE_GRADING_SEED,
            "deme_mortar_layer": DEME_MORTAR_LAYER,
            "deme_particle_radius_range": [float(particle_radii.min()), float(particle_radii.max())],
            "deme_concrete_density": DEME_CONCRETE_DENSITY,
            "deme_particle_mass_density_scale": DEME_PARTICLE_MASS_DENSITY_SCALE,
            "deme_dfc_material": DEME_DFC_MATERIAL,
            "deme_settling_enabled": True,
            "deme_settling_converged": settling_converged,
            "deme_settling_time": settling_time,
            "deme_settling_checks": settling_checks,
            "deme_settling_final_kinetic_energy": settling_final_kinetic_energy,
            "deme_settling_kinetic_energy_threshold": DEME_SETTLING_KINETIC_ENERGY_THRESHOLD,
            "deme_confinement_enabled_during_settling": True,
            "deme_confinement_enabled_after_settling": not RUN_PARTICLE_DISCHARGE_AFTER_SETTLING,
            "deme_settling_min_x_plane": settling_min_x,
            "deme_settling_top_z_plane": settling_top_z,
            "deme_build_plate_enabled": ENABLE_DEME_BUILD_PLATE,
            "deme_build_plate_family": DEME_BUILD_PLATE_FAMILY,
            "deme_build_plate_top_world_z": build_plate_top_world_z,
            "deme_build_plate_top_z": build_plate_top_deme_z,
            "deme_build_plate_visible": ENABLE_DEME_BUILD_PLATE and SHOW_DEME_BUILD_PLATE,
            "deme_auger_sample_bounds_min": list(DEME_AUGER_SAMPLE_BOUNDS_MIN),
            "deme_auger_sample_bounds_max": list(DEME_AUGER_SAMPLE_BOUNDS_MAX),
            "deme_auger_sample_ray_axis": DEME_AUGER_SAMPLE_RAY_AXIS,
            "deme_auger_sample_clearance": DEME_AUGER_SAMPLE_CLEARANCE,
            "deme_nozzle_contact_proxy_enabled": True,
            "deme_nozzle_contact_proxy_triangles": int(deme_nozzle_proxy.GetNumTriangles()),
            "deme_nozzle_contact_proxy_patch_angle_degrees": DEME_NOZZLE_PROXY_PATCH_ANGLE_DEGREES,
            "deme_nozzle_contact_proxy_patch_count": deme_nozzle_patch_count,
            "deme_particle_discharge_enabled": RUN_PARTICLE_DISCHARGE_AFTER_SETTLING,
            "deme_particle_discharge_time": discharge_time,
            "deme_particle_discharge_frames": discharge_frames_completed,
            "deme_particle_discharge_final_kinetic_energy": (
                discharge_final_kinetic_energy if RUN_PARTICLE_DISCHARGE_AFTER_SETTLING else None
            ),
            "deme_release_total_contacts": release_total_contacts,
            "deme_release_nozzle_contacts": release_nozzle_contacts,
            "gcode_motion_enabled": RUN_GCODE_MOTION_AFTER_SETTLING,
            "movie_saved": SAVE_MOVIE and not USE_OMNIVERSE_VISUALIZATION,
            "movie_path": str(MOVIE_OUTPUT_PATH) if SAVE_MOVIE and not USE_OMNIVERSE_VISUALIZATION else None,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Completed {frames_completed} frames with {final_nozzle_error:.6f} m final nozzle error.")
if SAVE_MOVIE and not USE_OMNIVERSE_VISUALIZATION:
    print(
        f"Movie contains {settling_frames_completed} settling and "
        f"{discharge_frames_completed} discharge frame(s): {MOVIE_OUTPUT_PATH}"
    )
if SAVE_VTK_TIME_SERIES:
    print(f"VTK time series contains {len(vtk_frames)} frame(s): {VTK_SERIES_PATH}")
print(f"Output: {OUTPUT_DIRECTORY}")
