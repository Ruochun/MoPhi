"""DEME cup-drop, frozen-hand push, and surface-velocity calibration experiments."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import fast_simplification  # Required by trimesh's collision-mesh decimator.
import newton
import numpy as np
import rtree  # Required by trimesh's sampled surface-distance checks.
from scipy.spatial.transform import Rotation
import trimesh
import warp as wp

import mophi
from mophi.couplers.newton_deme import prepare_mesh_principal_frame, write_wavefront_mesh
from mophi.utils.grab_motion import load_motion, load_object_motion
from mophi.utils.package_provider import load_package_provider

DEME = load_package_provider("deme")

# ── Configuration ────────────────────────────────────────────────────────────
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OBJECT_PATH = REPOSITORY_ROOT / "data/grab/s2/mug_drink_2_object.npz"
HAND_PATH = REPOSITORY_ROOT / "data/grab/s2/mug_drink_2_right.npz"
HAND_SOURCE_FRAME = 0  # Freeze one measured hand shape for the controlled push.
CASE = "drop"  # drop, hand_push, slide, or deform_slide
CUDA_DEVICE = 0
DEME_DT = 1.0e-4
DURATION = 0.8
RENDER_FPS = 60.0
DIAGNOSTIC_INTERVAL = 0.005
PROGRESS_INTERVAL = 0.1
SETTLE_TIME = 0.25
MOTION_RAMP_TIME = 0.1
SURFACE_SPEED = 0.08
CUP_MASS = 0.25
COLLISION_FACE_COUNT = 6000  # Zero keeps the original mesh; display and inertia always use it.
GEOMETRY_CHECK_SAMPLES = 2000
GEOMETRY_CHECK_SEED = 0
MAX_PROXY_SURFACE_ERROR = 0.0015
MAX_PROXY_VOLUME_ERROR = 0.03
GRAVITY = 9.81
DROP_HEIGHT = 0.02
REST_GAP = 0.0005
HAND_GAP = 0.003
CONTACT_YOUNG_MODULUS = 1.0e6
CONTACT_POISSON_RATIO = 0.3
CONTACT_RESTITUTION = 0.1
CONTACT_FRICTION = 0.5
PUSH_FLOOR_FRICTION = 0.1
FIXED_OWNER_MASS = 1.0e6  # Keeps the contact reduced mass close to the cup mass.
FIXED_OWNER_MOI = (1.0, 1.0, 1.0)
CUP_FAMILY = 0
MOVING_FAMILY = 1
FLOOR_FAMILY = 2
DOMAIN_X = (-0.7, 0.7)
DOMAIN_Y = (-0.6, 0.6)
DOMAIN_Z = (-0.2, 0.8)
BIN_SIZE = 0.02
MAX_VELOCITY = 5.0
DETECTION_SAFETY_SPEED = 0.2  # Includes prescribed nodal motion in detection bounds.
PLATE_EXTENTS = (0.6, 0.4, 0.02)
FLOOR_VISUAL_EXTENTS = (0.7, 0.5, 0.01)
USE_OMNIVERSE_VISUALIZATION = False
HEADLESS = False
RENDER = True
SAVE_MOVIE = False
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAMERA_POSITION = (0.48, -0.65, 0.38)
CAMERA_TARGET = (0.0, 0.0, 0.07)
AXIS_ORIGIN = (-0.25, 0.18, 0.01)
AXIS_LENGTH = 0.08
SCALE_BAR_CENTER = (0.0, 0.3, 0.01)
SCALE_MARKER_RADIUS = 0.007
MESH_COLORS = {"cup": (0.65, 0.70, 0.75), "surface": (0.91, 0.55, 0.19), "floor": (0.35, 0.40, 0.45)}
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output/demo_grab_contact"
# Acceptance limits are diagnostic tolerances, not material calibration.
MAX_ALLOWED_PENETRATION = 0.004
MIN_PUSH_DISPLACEMENT = 0.002
MIN_SLIDE_SPEED_FRACTION = 0.2
MAX_FRICTIONLESS_SPEED_FRACTION = 0.1
MAX_SETTLED_SPEED = 0.05
MAX_WEIGHT_BALANCE_RELATIVE_ERROR = 0.15


def surface_motion(time_s, speed=SURFACE_SPEED):
    """Smoothly start a constant-speed translation after settling."""
    elapsed = max(0.0, time_s - SETTLE_TIME)
    ramp = min(elapsed, MOTION_RAMP_TIME)
    position = speed * (0.5 * ramp * ramp / MOTION_RAMP_TIME + max(0.0, elapsed - MOTION_RAMP_TIME))
    return position, speed * ramp / MOTION_RAMP_TIME


def make_hand_pusher(hand, frame):
    """Orient a frozen recorded hand with its thin direction along world X."""
    rows = np.flatnonzero(hand["source_frame_indices"] == frame)
    if len(rows) != 1:
        mophi.fatal(f"Source frame {frame} is absent from {hand['path']}")
    vertices = hand["hand_vertices_local"][rows[0]].astype(np.float64)
    _, axes = np.linalg.eigh(vertices.T @ vertices)
    # The shortest principal extent is palm thickness; longest becomes vertical.
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    vertices = vertices @ axes
    vertices -= (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    return vertices.astype(np.float32), hand["hand_faces"]


def prescribe_deme_surface(solver, initial_position, speed):
    """Let DEME evaluate consistent position and velocity at every physics step."""
    elapsed = f"fmaxf(0.f, t - {SETTLE_TIME:.9e}f)"
    ramp = f"fminf({elapsed}, {MOTION_RAMP_TIME:.9e}f)"
    displacement = f"({speed:.9e}f * (0.5f * ({ramp}) * ({ramp}) / {MOTION_RAMP_TIME:.9e}f + fmaxf(0.f, ({elapsed}) - {MOTION_RAMP_TIME:.9e}f)))"
    velocity = f"({speed:.9e}f * ({ramp}) / {MOTION_RAMP_TIME:.9e}f)"
    solver.SetFamilyPrescribedPosition(
        MOVING_FAMILY,
        f"({initial_position[0]:.9g} + {displacement})",
        str(float(initial_position[1])),
        str(float(initial_position[2])),
    )
    solver.SetFamilyPrescribedQuaternion(MOVING_FAMILY)
    solver.SetFamilyPrescribedLinVel(MOVING_FAMILY, velocity, "0", "0")
    solver.SetFamilyPrescribedAngVel(MOVING_FAMILY, "0", "0", "0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("drop", "hand_push", "slide", "deform_slide"), default=CASE)
    parser.add_argument("--object", type=Path, default=OBJECT_PATH)
    parser.add_argument("--hand", type=Path, default=HAND_PATH)
    parser.add_argument("--hand-frame", type=int, default=HAND_SOURCE_FRAME)
    parser.add_argument("--dt", type=float, default=DEME_DT)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--friction", type=float, default=CONTACT_FRICTION)
    parser.add_argument("--speed", type=float, default=SURFACE_SPEED)
    parser.add_argument("--collision-faces", type=int, default=COLLISION_FACE_COUNT)
    parser.add_argument("--headless", action="store_true", default=HEADLESS)
    parser.add_argument("--no-render", action="store_true", default=not RENDER)
    parser.add_argument("--save-movie", action="store_true", default=SAVE_MOVIE)
    parser.add_argument("--usd", action="store_true", default=USE_OMNIVERSE_VISUALIZATION)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--check", action="store_true", help="Fail if the selected calibration acceptance checks fail")
    args = parser.parse_args()
    if not args.object.is_file() or (args.case == "hand_push" and not args.hand.is_file()):
        mophi.fatal("Prepared GRAB assets are missing. See docs/how-to/grab-contact.md.")
    if (
        not np.isfinite([args.dt, args.duration, args.friction, args.speed]).all()
        or args.dt <= 0
        or args.duration <= 0
        or args.friction < 0
        or args.speed <= 0
    ):
        mophi.fatal("Require finite positive dt, duration, speed, and nonnegative friction")
    if args.save_movie and (args.no_render or args.usd):
        mophi.fatal("--save-movie requires the OpenGL renderer")
    if args.collision_faces != 0 and args.collision_faces < 4:
        mophi.fatal("--collision-faces must be zero (full mesh) or at least four")
    # DEME stores h as float32 but accepts a double duration. Using the exact
    # stored h for chunk durations prevents an unintended extra step per call.
    physics_dt = float(np.float32(args.dt))
    if not np.isfinite(physics_dt) or physics_dt <= 0:
        mophi.fatal("--dt is outside DEME's positive float32 range")
    output = args.output_dir / args.case
    output.mkdir(parents=True, exist_ok=True)
    obj = load_object_motion(args.object)
    cup = prepare_mesh_principal_frame(obj["object_vertices_local"], obj["object_faces"], CUP_MASS)
    original = trimesh.Trimesh(cup.vertices, cup.triangle_indices, process=False)
    collision = original
    proxy_error = 0.0
    proxy_volume_error = 0.0
    if 0 < args.collision_faces < len(original.faces):
        collision = original.simplify_quadric_decimation(face_count=args.collision_faces)
        if not collision.is_volume:
            mophi.fatal("Simplified cup collision mesh is not a closed positive volume; increase --collision-faces")
        for a, b in ((original, collision), (collision, original)):
            sample_points = trimesh.sample.sample_surface(a, GEOMETRY_CHECK_SAMPLES, seed=GEOMETRY_CHECK_SEED)[0]
            distances = trimesh.proximity.closest_point(b, sample_points)[1]
            proxy_error = max(proxy_error, float(distances.max()))
        proxy_volume_error = abs(collision.volume / original.volume - 1)
        if proxy_error > MAX_PROXY_SURFACE_ERROR or proxy_volume_error > MAX_PROXY_VOLUME_ERROR:
            mophi.fatal(
                f"Collision proxy error is too large ({proxy_error:g} m, volume fraction {proxy_volume_error:g}); increase --collision-faces"
            )
    collision_vertices = np.asarray(collision.vertices, dtype=np.float32)
    collision_faces = np.asarray(collision.faces, dtype=np.int32)
    cup_rotation = Rotation.from_matrix(cup.principal_to_input)
    cup_centered = cup_rotation.apply(collision_vertices)
    gap = DROP_HEIGHT if args.case == "drop" else REST_GAP
    cup_initial = np.array([0.0, 0.0, gap - cup_centered[:, 2].min()])

    solver = DEME.DEMSolver([CUDA_DEVICE])
    solver.SetVerbosity("WARNING")
    solver.SetMeshUniversalContact(True)
    solver.SetTimeStepSize(physics_dt)
    solver.SetGravitationalAcceleration([0.0, 0.0, -GRAVITY])
    solver.InstructBoxDomainDimension(DOMAIN_X, DOMAIN_Y, DOMAIN_Z)
    solver.SetInitBinSize(BIN_SIZE)
    solver.SetMaxVelocity(MAX_VELOCITY)
    solver.SetExpandSafetyAdder(DETECTION_SAFETY_SPEED)
    material = solver.LoadMaterial(
        {
            "E": CONTACT_YOUNG_MODULUS,
            "nu": CONTACT_POISSON_RATIO,
            "CoR": CONTACT_RESTITUTION,
            "mu": args.friction,
            "Crr": 0.0,
        }
    )
    floor_material = solver.LoadMaterial(
        {
            "E": CONTACT_YOUNG_MODULUS,
            "nu": CONTACT_POISSON_RATIO,
            "CoR": CONTACT_RESTITUTION,
            "mu": PUSH_FLOOR_FRICTION,
            "Crr": 0.0,
        }
    )
    solver.SetMaterialPropertyPair("mu", material, floor_material, PUSH_FLOOR_FRICTION)
    write_wavefront_mesh(output / "cup_principal.obj", collision_vertices, collision_faces)
    cup_mesh = solver.AddWavefrontMeshObject(str(output / "cup_principal.obj"), material, False)
    cup_mesh.SetMass(cup.mass)
    cup_mesh.SetMOI(cup.principal_moi.tolist())
    cup_mesh.SetInitPos(cup_initial.tolist())
    cup_mesh.SetInitQuat(cup_rotation.as_quat().tolist())
    cup_mesh.SetFamily(CUP_FAMILY)
    cup_tracker = solver.Track(cup_mesh)
    geometry = {"cup": (cup.vertices, cup.triangle_indices)}
    surface_tracker = None
    surface_initial = np.zeros(3)
    surface_vertices = None

    if args.case in ("drop", "hand_push"):
        floor = solver.AddBCPlane([0, 0, 0], [0, 0, 1], floor_material)
        floor.SetFamily(FLOOR_FAMILY)
        solver.SetFamilyFixed(FLOOR_FAMILY)
        floor_visual = trimesh.creation.box(extents=FLOOR_VISUAL_EXTENTS)
        floor_visual.apply_translation([0, 0, -FLOOR_VISUAL_EXTENTS[2] / 2])
        geometry["floor"] = (
            np.asarray(floor_visual.vertices, dtype=np.float32),
            np.asarray(floor_visual.faces, dtype=np.int32),
        )
    if args.case != "drop":
        if args.case == "hand_push":
            hand = load_motion(args.hand)
            surface_vertices, surface_faces = make_hand_pusher(hand, args.hand_frame)
            surface_initial = np.array(
                [
                    cup_centered[:, 0].min() - HAND_GAP - surface_vertices[:, 0].max(),
                    0.0,
                    max(cup_initial[2], REST_GAP - surface_vertices[:, 2].min()),
                ]
            )
            solver.DisableContactBetweenFamilies(MOVING_FAMILY, FLOOR_FAMILY)
        else:
            plate = trimesh.creation.box(extents=PLATE_EXTENTS)
            surface_vertices = np.asarray(plate.vertices, dtype=np.float32)
            surface_faces = np.asarray(plate.faces, dtype=np.int32)
            surface_initial[2] = -PLATE_EXTENTS[2] / 2
        write_wavefront_mesh(output / "surface.obj", surface_vertices, surface_faces)
        surface = solver.AddWavefrontMeshObject(str(output / "surface.obj"), material, False)
        surface.SetMass(FIXED_OWNER_MASS)
        surface.SetMOI(FIXED_OWNER_MOI)
        surface.SetInitPos(surface_initial.tolist())
        surface.SetFamily(MOVING_FAMILY)
        surface_tracker = solver.Track(surface)
        geometry["surface"] = (surface_vertices, surface_faces)
        if args.case == "deform_slide":
            # Deliberately leave owner velocity zero to test whether UpdateMesh
            # alone contributes surface velocity to the contact force law.
            solver.SetFamilyFixed(MOVING_FAMILY)
        else:
            prescribe_deme_surface(solver, surface_initial, args.speed)

    print(
        f"Initializing DEME: {args.case}, {len(collision_faces)} collision triangles, dt={args.dt:g}; sampled proxy error={proxy_error:g} m",
        flush=True,
    )
    solver.Initialize()
    print("DEME initialized; starting contact experiment.", flush=True)
    wp.init()
    vis = None
    writer = None
    point_buffers = {}
    face_buffers = {}
    if not args.no_render:
        if args.usd:
            vis = mophi.OmniverseVisualizer(str(output / "contact.usdc"), fps=RENDER_FPS)
            if not vis.pxr_available:
                mophi.fatal("USD export requires: python -m pip install usd-core")
        else:
            model = newton.ModelBuilder(up_axis=newton.Axis.Z).finalize()
            vis = mophi.OpenGLVisualizer(model, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, headless=args.headless)
        direction = np.asarray(CAMERA_TARGET) - CAMERA_POSITION
        vis.set_camera(
            wp.vec3(*CAMERA_POSITION),
            float(np.degrees(np.arctan2(direction[2], np.hypot(direction[0], direction[1])))),
            float(np.degrees(np.arctan2(direction[1], direction[0]))),
        )
        for name, (vertices, faces) in geometry.items():
            point_buffers[name] = wp.array(vertices, dtype=wp.vec3)
            face_buffers[name] = wp.array(faces.ravel(), dtype=wp.int32)
    samples = []
    steps = int(np.ceil(args.duration / args.dt))
    diagnostic_steps = max(1, round(DIAGNOSTIC_INTERVAL / args.dt))
    record_steps = set(range(0, steps + 1, diagnostic_steps)) | {steps}
    render_steps = set(
        np.minimum(steps, np.rint(np.arange(0, args.duration, 1 / RENDER_FPS) / args.dt).astype(int))
    ) | {steps}
    # Rigid prescriptions run inside DEME: synchronize only for output. The
    # vertex-motion diagnostic requires a synchronized update every physics step.
    schedule = range(steps + 1) if args.case == "deform_slide" else sorted(record_steps | render_steps)
    cup_id = cup_tracker.GetOwnerID()
    last_world = None
    completed = False
    next_progress = 0.0
    try:
        if args.save_movie:
            writer = imageio.get_writer(output / "contact.mp4", fps=RENDER_FPS)
        previous_step = 0
        for step in schedule:
            t = step * physics_dt
            if step:
                if args.case == "deform_slide":
                    displacement, _ = surface_motion(t - physics_dt, args.speed)
                    nodes = surface_vertices + np.array([displacement, 0, 0], dtype=np.float32)
                    surface_tracker.UpdateMesh(nodes.tolist())
                solver.DoDynamicsThenSync((step - previous_step) * physics_dt)
            previous_step = step
            record = step in record_steps
            render = vis is not None and step in render_steps
            if record or render:
                position = np.asarray(cup_tracker.Pos())
                quaternion = np.asarray(cup_tracker.OriQ())
                velocity = np.asarray(cup_tracker.Vel())
                angular_velocity = np.asarray(cup_tracker.AngVelGlobal())
                if not np.isfinite(np.r_[position, quaternion, velocity, angular_velocity]).all():
                    mophi.fatal(f"Nonfinite DEME cup state at t={t}")
                last_world = Rotation.from_quat(quaternion).apply(cup.vertices) + position
                collision_world = Rotation.from_quat(quaternion).apply(collision_vertices) + position
                if record:
                    force, torque = solver.GetOwnerContactWrench(cup_id)
                    force = np.asarray(force[0])
                    torque = np.asarray(torque[0])
                    samples.append(
                        np.r_[
                            t,
                            position,
                            quaternion,
                            velocity,
                            angular_velocity,
                            force,
                            torque,
                            collision_world[:, 2].min(),
                        ]
                    )
                    if t + args.dt / 2 >= next_progress:
                        print(f"t={t:.3f} s, cup CoM={position.round(5)}, velocity={velocity.round(5)}", flush=True)
                        next_progress += PROGRESS_INTERVAL
                if render:
                    vis.begin_frame(t)
                    if step == 0:
                        mophi.log_orientation_and_scale_reference(
                            vis,
                            axis_origin=AXIS_ORIGIN,
                            axis_length=AXIS_LENGTH,
                            scale_bar_center=SCALE_BAR_CENTER,
                            marker_radius=SCALE_MARKER_RADIUS,
                        )
                    for name, (vertices, _) in geometry.items():
                        if name == "cup":
                            world = last_world
                        elif name == "surface":
                            surface_local = vertices
                            if args.case == "deform_slide":
                                displacement, _ = surface_motion(max(0, t - physics_dt), args.speed)
                                surface_local = vertices + [displacement, 0, 0]
                            world = (
                                Rotation.from_quat(surface_tracker.OriQ()).apply(surface_local) + surface_tracker.Pos()
                            )
                        else:
                            world = vertices
                        point_buffers[name].assign(np.asarray(world, dtype=np.float32))
                        vis.log_mesh(name, point_buffers[name], face_buffers[name], color=MESH_COLORS[name])
                    vis.end_frame()
                    if writer is not None:
                        writer.append_data(vis.get_frame().numpy())
                    if not vis.is_running():
                        break
            if step == steps:
                completed = True
        if vis is not None and not args.usd and vis.is_running():
            imageio.imwrite(output / "last_frame.png", vis.get_frame().numpy())
    finally:
        if writer is not None:
            writer.close()
        if vis is not None:
            vis.close()

    data = np.asarray(samples)
    np.savez_compressed(
        output / "trajectory.npz",
        time_s=data[:, 0],
        cup_com_world=data[:, 1:4],
        cup_quaternion_xyzw=data[:, 4:8],
        cup_velocity_world=data[:, 8:11],
        cup_angular_velocity_world=data[:, 11:14],
        cup_contact_force_world=data[:, 14:17],
        cup_contact_torque_world=data[:, 17:20],
        cup_min_z=data[:, 20],
        mass_kg=cup.mass,
        principal_moi_kg_m2=cup.principal_moi,
        source_mesh_com=cup.center_of_mass,
        principal_to_source=cup.principal_to_input,
    )
    settled = data[:, 0] >= max(SETTLE_TIME, data[-1, 0] - SETTLE_TIME)
    mean_vx = float(np.mean(data[settled, 8])) if settled.any() else float(data[-1, 8])
    mean_contact_force = np.mean(data[settled, 14:17], axis=0) if settled.any() else data[-1, 14:17]
    checks = {
        "completed": completed,
        "time_accounting": bool(abs(solver.GetSimTime() - previous_step * physics_dt) < physics_dt * 1e-3),
        "floor_penetration_bounded": bool(data[:, 20].min() >= -MAX_ALLOWED_PENETRATION),
        "contact_occurred": bool(np.max(np.linalg.norm(data[:, 14:17], axis=1)) > CUP_MASS * GRAVITY / 2),
    }
    surface_geometry_error = None
    if surface_tracker is not None:
        commanded_shift, _ = surface_motion(max(0, previous_step * physics_dt - physics_dt), args.speed)
        expected_surface = surface_vertices + surface_initial + [commanded_shift, 0, 0]
        actual_surface = np.asarray(surface_tracker.GetMeshNodesGlobal())
        surface_geometry_error = float(np.max(np.abs(actual_surface - expected_surface)))
        checks["surface_geometry_matches_command"] = surface_geometry_error < max(1e-6, 2 * args.speed * physics_dt)
    if args.case == "drop":
        checks["cup_settled"] = bool(np.linalg.norm(data[-1, 8:11]) < MAX_SETTLED_SPEED)
        checks["support_balances_weight"] = bool(
            abs(mean_contact_force[2] / (CUP_MASS * GRAVITY) - 1) < MAX_WEIGHT_BALANCE_RELATIVE_ERROR
        )
        freefall = data[:, 0] < 0.8 * np.sqrt(2 * DROP_HEIGHT / GRAVITY)
        expected_z = data[0, 3] - 0.5 * GRAVITY * data[freefall, 0] ** 2
        freefall_error = float(np.max(np.abs(data[freefall, 3] - expected_z)))
        checks["freefall_matches_gravity"] = bool(
            freefall_error < GRAVITY * args.dt * np.sqrt(2 * DROP_HEIGHT / GRAVITY)
        )
    elif args.case == "hand_push":
        checks["cup_pushed_forward"] = bool(data[-1, 1] - data[0, 1] > MIN_PUSH_DISPLACEMENT)
    elif args.friction == 0:
        checks["frictionless_no_transport"] = bool(abs(mean_vx) < args.speed * MAX_FRICTIONLESS_SPEED_FRACTION)
    else:
        checks["surface_velocity_transferred"] = bool(mean_vx > args.speed * MIN_SLIDE_SPEED_FRACTION)
    summary = {
        "case": args.case,
        "dt_s": physics_dt,
        "duration_s": float(data[-1, 0]),
        "solver_time_s": solver.GetSimTime(),
        "cup_mass_kg": cup.mass,
        "source_object": str(args.object.resolve()),
        "source_hand": str(args.hand.resolve()) if args.case == "hand_push" else None,
        "hand_source_frame": args.hand_frame if args.case == "hand_push" else None,
        "mass_model": "uniform density inside the source closed mesh",
        "cup_collision_triangles": len(collision_faces),
        "cup_visual_triangles": len(cup.triangle_indices),
        "proxy_sampled_surface_error_m": proxy_error,
        "proxy_relative_volume_error": float(proxy_volume_error),
        "friction": args.friction,
        "surface_speed_m_s": args.speed,
        "surface_geometry_error_m": surface_geometry_error,
        "surface_owner_velocity_world_m_s": surface_tracker.Vel() if surface_tracker is not None else None,
        "surface_command_velocity_world_m_s": (
            [surface_motion(previous_step * physics_dt, args.speed)[1], 0.0, 0.0]
            if surface_tracker is not None
            else None
        ),
        "material_young_modulus_Pa": CONTACT_YOUNG_MODULUS,
        "material_poisson_ratio": CONTACT_POISSON_RATIO,
        "restitution": CONTACT_RESTITUTION,
        "push_floor_friction": PUSH_FLOOR_FRICTION,
        "gravity_m_s2": GRAVITY,
        "settle_time_s": SETTLE_TIME,
        "motion_ramp_time_s": MOTION_RAMP_TIME,
        "cup_displacement_x_m": float(data[-1, 1] - data[0, 1]),
        "late_mean_cup_velocity_x_m_s": mean_vx,
        "late_mean_contact_force_N": mean_contact_force.tolist(),
        "minimum_cup_z_m": float(data[:, 20].min()),
        "peak_sampled_contact_force_N": float(np.max(np.linalg.norm(data[:, 14:17], axis=1))),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if args.case == "drop":
        summary["freefall_max_position_error_m"] = freefall_error
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if args.check and not summary["passed"]:
        mophi.fatal(f"Contact calibration failed: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
