"""Shared GRAB grasp experiment: prescribed hand and DEME-driven object.

Scene entry points provide configuration; this module retains the visible
DEME stepping schedule and Newton rendering. Legacy output keys use ``cup``
for the manipulated object, including non-cup scenes.
"""

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
from mophi.couplers.newton_deme import (
    prepare_mesh_principal_frame,
    write_wavefront_mesh,
)
from mophi.utils.geometry.mesh_contact_patches import partition_mesh_contact_patches, partition_mesh_seeded_patches
from mophi.utils.grab.motion import load_motion, load_object_motion, sample_interval
from mophi.utils.package_provider import load_package_provider

DEME = load_package_provider("deme")


def sample_grab_hand(hand, source_time, playback_speed, offset):
    """Linear nodal motion with the exact segment derivative for owner velocity.

    Owner orientation is identity: all rotation/articulation remains in local
    nodes, and only the origin derivative contributes to contact velocity.
    """
    # Interpolate the origin and local vertices separately so their sum is the
    # commanded world surface. Scaling the origin derivative slows contact velocity
    # consistently with playback; local articulation velocity is not supplied.
    left, right, weight = sample_interval(hand["timestamps_s"], source_time)
    origin = (1 - weight) * hand["hand_origin_world"][left] + weight * hand["hand_origin_world"][right]
    nodes = (1 - weight) * hand["hand_vertices_local"][left] + weight * hand["hand_vertices_local"][right]
    velocity = np.zeros(3)
    if right != left:
        velocity = (
            playback_speed
            * (hand["hand_origin_world"][right] - hand["hand_origin_world"][left])
            / (hand["timestamps_s"][right] - hand["timestamps_s"][left])
        )
    return origin + offset, nodes, velocity


def assess_grab_run(data, grasp, config):
    """Distinguish sustained contact-supported lift from mere upward flight."""
    # Diagnostic rows are packed during stepping, then saved as named NPZ arrays.
    # data: time, CoM, xyzw orientation, linear/angular velocity, force, torque, bottom Z.
    # grasp: source time, reference CoM, hand origin/velocity, penetration, transfer errors.
    checks = {}
    checks["hand_geometry_matches_command"] = bool(grasp[:, 11].max() < config.GRAB_TRANSFER_TOLERANCE)
    checks["hand_owner_velocity_matches_command"] = bool(grasp[:, 12].max() < config.GRAB_TRANSFER_TOLERANCE)
    checks["sampled_hand_penetration_bounded"] = bool(grasp[:, 10].max() < config.MAX_ALLOWED_PENETRATION)
    # Require clearance and upward contact support over the final window: a brief
    # ballistic hop alone should not qualify as a supported grasp.
    lift_window = data[:, 0] >= data[-1, 0] - config.GRAB_LIFT_HOLD_INTERVAL
    checks["cup_lifted"] = bool(np.min(data[lift_window, 20]) > config.MIN_GRAB_LIFT)
    checks["lift_supported_by_contact"] = bool(
        checks["cup_lifted"]
        and np.mean(data[lift_window, 16]) > config.CUP_MASS * config.GRAVITY * config.GRAB_SUPPORT_FORCE_FRACTION
    )
    checks["cup_near_recorded_position"] = bool(
        np.linalg.norm(data[-1, 1:4] - grasp[-1, 1:4]) < config.MAX_GRAB_REFERENCE_ERROR
    )
    return checks


def main(config):
    # ── Scene selection and input validation ─────────────────────────────────
    # Each entry point supplies its own scene_config; CLI arguments override only
    # selected settings. Validate before allocating solver resources.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("grab", "cup_rest"), default=config.CASE)
    parser.add_argument("--object", type=Path, default=config.OBJECT_PATH)
    parser.add_argument("--hand", type=Path, default=config.HAND_PATH)
    parser.add_argument("--start-frame", type=int, default=config.GRAB_START_FRAME)
    parser.add_argument("--stop-frame", type=int, default=config.GRAB_STOP_FRAME)
    parser.add_argument("--playback-speed", type=float, default=config.PLAYBACK_SPEED)
    parser.add_argument("--dt", type=float, default=config.DEME_DT)
    parser.add_argument(
        "--duration",
        type=float,
        default=config.DURATION,
        help="Cup-rest duration; grab derives it from source interval and playback speed",
    )
    parser.add_argument("--friction", type=float, default=config.CONTACT_FRICTION)
    parser.add_argument("--bin-size", type=float, default=config.BIN_SIZE)
    parser.add_argument("--compact-rest-domain", action="store_true", default=config.COMPACT_REST_DOMAIN)
    parser.add_argument("--floor-friction", type=float, default=config.FLOOR_FRICTION)
    parser.add_argument(
        "--cup-patches",
        choices=("single", "sectors", "triangles", "connected", "spread"),
        default=config.CUP_PATCH_MODE,
    )
    parser.add_argument(
        "--hand-patches", choices=("single", "triangles", "connected", "spread"), default=config.HAND_PATCH_MODE
    )
    parser.add_argument("--cup-patch-radius", type=float, default=config.CUP_PATCH_RADIUS)
    parser.add_argument("--hand-patch-radius", type=float, default=config.HAND_PATCH_RADIUS)
    parser.add_argument("--patch-normal-angle", type=float, default=config.PATCH_MAX_NORMAL_ANGLE)
    parser.add_argument("--patch-sectors", type=int, default=config.CUP_PATCH_SECTORS)
    parser.add_argument("--collision-faces", type=int, default=config.COLLISION_FACE_COUNT)
    parser.add_argument("--headless", action="store_true", default=config.HEADLESS)
    parser.add_argument("--no-render", action="store_true", default=not config.RENDER)
    parser.add_argument("--save-movie", action="store_true", default=config.SAVE_MOVIE)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIRECTORY)
    parser.add_argument("--check", action="store_true", help="Fail if the selected scene acceptance checks fail")
    args = parser.parse_args()
    if not np.isfinite(args.bin_size) or args.bin_size <= 0:
        mophi.fatal("--bin-size must be finite and positive")
    if args.compact_rest_domain and args.case != "cup_rest":
        mophi.fatal("--compact-rest-domain is only supported for cup_rest")
    if not np.isfinite(args.floor_friction) or args.floor_friction < 0 or args.patch_sectors < 2:
        mophi.fatal("Require nonnegative finite floor friction and at least two patch sectors")
    if not args.object.is_file() or (args.case == "grab" and not args.hand.is_file()):
        mophi.fatal("Prepared GRAB assets are missing. See docs/how-to/grab.md.")
    if (
        not np.isfinite([args.dt, args.duration, args.friction]).all()
        or args.dt <= 0
        or args.duration <= 0
        or args.friction < 0
    ):
        mophi.fatal("Require finite positive dt, duration, and nonnegative friction")
    if args.save_movie and args.no_render:
        mophi.fatal("--save-movie requires the OpenGL renderer")
    if args.collision_faces != 0 and args.collision_faces < 4:
        mophi.fatal("--collision-faces must be zero (full mesh) or at least four")
    # DEME stores h as float32 but accepts a double duration. Using the exact
    # stored h for chunk durations prevents an unintended extra step per call.
    physics_dt = float(np.float32(args.dt))
    if not np.isfinite(physics_dt) or physics_dt <= 0:
        mophi.fatal("--dt is outside DEME's positive float32 range")

    # ── Recorded motion and the two time axes ────────────────────────────────
    # Source timestamps belong to GRAB; simulation time starts at zero and includes
    # a settling hold. The isolated object-rest case never loads a hand trajectory.
    output = args.output_dir / args.case
    output.mkdir(parents=True, exist_ok=True)
    obj = load_object_motion(args.object)
    hand = None
    grab_offset = np.zeros(3)
    grab_samples = []
    maximum_nodal_speed = 0.0
    contact_samples = []
    contact_offsets = [0]
    if args.case == "grab":
        hand = load_motion(args.hand)
        if str(hand["source_sequence"]) != str(obj["source_sequence"]):
            mophi.fatal("Hand and object must originate from the same GRAB sequence")
        if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
            mophi.fatal("--playback-speed must be finite and positive")
        start_time = args.start_frame / float(hand["source_fps"])
        stop_time = args.stop_frame / float(hand["source_fps"])
        if (
            not max(hand["timestamps_s"][0], obj["timestamps_s"][0])
            <= start_time
            < stop_time
            <= min(hand["timestamps_s"][-1], obj["timestamps_s"][-1])
        ):
            mophi.fatal("Require an increasing source-frame interval inside both recordings")
        # Bound speeds using actual moving vertices, including finger articulation,
        # because the owner translation alone understates collision-search motion.
        selected = (hand["timestamps_s"] >= start_time) & (hand["timestamps_s"] <= stop_time)
        world_nodes = hand["hand_vertices_local"][selected] + hand["hand_origin_world"][selected, None, :]
        nodal_velocity = (
            np.diff(world_nodes, axis=0) / np.diff(hand["timestamps_s"][selected])[:, None, None] * args.playback_speed
        )
        maximum_nodal_speed = float(np.linalg.norm(nodal_velocity, axis=2).max())
        if maximum_nodal_speed > config.MAX_VELOCITY:
            mophi.fatal("Selected hand motion exceeds MAX_VELOCITY; reduce playback speed")
        args.duration = config.SETTLE_TIME + (stop_time - start_time) / args.playback_speed
    if args.case == "cup_rest":
        start_time = args.start_frame / float(obj["source_fps"])
        if not obj["timestamps_s"][0] <= start_time <= obj["timestamps_s"][-1]:
            mophi.fatal("--start-frame is outside the object recording")

    # ── Object geometry, mass frame, and collision proxy ─────────────────────
    # DEME expects diagonal inertia about the CoM. Keep the full source mesh in
    # that principal frame for mass properties/rendering; simplify only contact
    # geometry and check its surface/volume error against the original.
    cup = prepare_mesh_principal_frame(obj["object_vertices_local"], obj["object_faces"], config.CUP_MASS)
    original = trimesh.Trimesh(cup.vertices, cup.triangle_indices, process=False)
    collision = original
    proxy_error = 0.0
    proxy_volume_error = 0.0
    if 0 < args.collision_faces < len(original.faces):
        collision = original.simplify_quadric_decimation(face_count=args.collision_faces)
        if not collision.is_volume:
            mophi.fatal("Simplified object collision mesh is not a closed positive volume; increase --collision-faces")
        for a, b in ((original, collision), (collision, original)):
            sample_points = trimesh.sample.sample_surface(
                a, config.GEOMETRY_CHECK_SAMPLES, seed=config.GEOMETRY_CHECK_SEED
            )[0]
            distances = trimesh.proximity.closest_point(b, sample_points)[1]
            proxy_error = max(proxy_error, float(distances.max()))
        proxy_volume_error = abs(collision.volume / original.volume - 1)
        if proxy_error > config.MAX_PROXY_SURFACE_ERROR or proxy_volume_error > config.MAX_PROXY_VOLUME_ERROR:
            mophi.fatal(
                f"Collision proxy error is too large ({proxy_error:g} m, volume fraction {proxy_volume_error:g}); increase --collision-faces"
            )
    collision_vertices = np.asarray(collision.vertices, dtype=np.float32)
    collision_faces = np.asarray(collision.faces, dtype=np.int32)

    # ── Place the recording in the simulation world ─────────────────────────
    # Compose source-object rotation with the principal-frame rotation. Apply one
    # common translation to object and hand: center the scene over a Z=0 table
    # without changing their relative placement. Recorded object motion sets only
    # the initial pose; later recorded poses are used for comparison diagnostics.
    row, _, weight = sample_interval(obj["timestamps_s"], start_time)
    if weight != 0:
        mophi.fatal("Grasp initialization requires an object sample at --start-frame")
    recorded_rotation = Rotation.from_quat(obj["object_quaternion_xyzw"][row])
    recorded_translation = obj["object_translation_world"][row]
    cup_rotation = recorded_rotation * Rotation.from_matrix(cup.principal_to_input)
    recorded_world = recorded_rotation.apply(obj["object_vertices_local"]) + recorded_translation
    grab_offset = np.array(
        [-recorded_world[:, 0].mean(), -recorded_world[:, 1].mean(), config.REST_GAP - recorded_world[:, 2].min()]
    )
    cup_initial = recorded_rotation.apply(cup.center_of_mass) + recorded_translation + grab_offset
    if args.case == "grab":
        domain = np.asarray([config.DOMAIN_X, config.DOMAIN_Y, config.DOMAIN_Z])
        motion_min = world_nodes.min(axis=(0, 1)) + grab_offset
        motion_max = world_nodes.max(axis=(0, 1)) + grab_offset
        if np.any(motion_min < domain[:, 0]) or np.any(motion_max > domain[:, 1]):
            mophi.fatal("Selected hand trajectory leaves the configured DEME domain; adjust DOMAIN_X/Y/Z")

    # ── DEME domain and contact materials ───────────────────────────────────
    # Universal contact handles the triangle surfaces. Collision-search expansion
    # accounts for prescribed hand motion; material parameters set the compliant
    # contact response, with a separate object/table friction coefficient.
    solver = DEME.DEMSolver([config.CUDA_DEVICE])
    solver.SetVerbosity("WARNING")
    solver.SetMeshUniversalContact(True)
    if args.case == "cup_rest":
        solver.SetContactOutputContent(["OWNER", "GEO_ID", "FORCE", "POINT", "NORMAL", "TORQUE"])
    solver.SetTimeStepSize(physics_dt)
    solver.SetGravitationalAcceleration([0.0, 0.0, -config.GRAVITY])
    domain_bounds = (
        (config.REST_DOMAIN_X, config.REST_DOMAIN_Y, config.REST_DOMAIN_Z)
        if args.compact_rest_domain
        else (config.DOMAIN_X, config.DOMAIN_Y, config.DOMAIN_Z)
    )
    solver.InstructBoxDomainDimension(*domain_bounds)
    solver.SetInitBinSize(args.bin_size)
    solver.SetMaxVelocity(config.MAX_VELOCITY)
    solver.SetExpandSafetyAdder(
        max(config.DETECTION_SAFETY_SPEED, config.GRAB_DETECTION_SPEED_FACTOR * maximum_nodal_speed)
    )
    material = solver.LoadMaterial(
        {
            "E": config.CONTACT_YOUNG_MODULUS,
            "nu": config.CONTACT_POISSON_RATIO,
            "CoR": config.CONTACT_RESTITUTION,
            "mu": args.friction,
            "Crr": 0.0,
        }
    )
    floor_material = solver.LoadMaterial(
        {
            "E": config.CONTACT_YOUNG_MODULUS,
            "nu": config.CONTACT_POISSON_RATIO,
            "CoR": config.CONTACT_RESTITUTION,
            "mu": args.floor_friction,
            "Crr": 0.0,
        }
    )
    solver.SetMaterialPropertyPair("mu", material, floor_material, args.floor_friction)

    # ── Dynamic object owner and contact patches ────────────────────────────
    # Patch IDs group triangles for contact evaluation; they do not create separate
    # rigid bodies. All triangles retain one owner, mass, and principal inertia.
    write_wavefront_mesh(output / "cup_principal.obj", collision_vertices, collision_faces)
    cup_mesh = solver.AddWavefrontMeshObject(str(output / "cup_principal.obj"), material, False)
    if args.cup_patches == "spread":
        cup_mesh.SetPatchIDs(
            partition_mesh_seeded_patches(collision_vertices, collision_faces, config.CUP_PATCH_COUNT).tolist()
        )
    elif args.cup_patches == "connected":
        cup_mesh.SetPatchIDs(
            partition_mesh_contact_patches(
                collision_vertices, collision_faces, args.cup_patch_radius, args.patch_normal_angle
            ).tolist()
        )
    elif args.cup_patches == "triangles":
        cup_mesh.SetEachTriangleAsPatch()
    elif args.cup_patches == "sectors":
        # Partition around the source cup's vertical axis without changing triangles.
        centers = Rotation.from_matrix(cup.principal_to_input).apply(collision_vertices[collision_faces].mean(axis=1))
        angles = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2 * np.pi)
        cup_mesh.SetPatchIDs(np.floor(angles / (2 * np.pi) * args.patch_sectors).astype(int).tolist())
    else:
        cup_mesh.SetPatchIDs([0] * len(collision_faces))
    cup_mesh.SetMass(cup.mass)
    cup_mesh.SetMOI(cup.principal_moi.tolist())
    cup_mesh.SetInitPos(cup_initial.tolist())
    cup_mesh.SetInitQuat(cup_rotation.as_quat().tolist())
    cup_mesh.SetFamily(config.CUP_FAMILY)
    cup_tracker = solver.Track(cup_mesh)
    geometry = {"cup": (cup.vertices, cup.triangle_indices)}
    surface_tracker = None
    surface_initial = np.zeros(3)
    surface_vertices = None
    surface = None

    # ── Table physics and display geometry ──────────────────────────────────
    # Contact uses an analytical plane. The box is only its visible representation,
    # with its upper face at the same world height.
    floor = solver.AddBCPlane([0, 0, 0], [0, 0, 1], floor_material)
    floor.SetFamily(config.FLOOR_FAMILY)
    solver.SetFamilyFixed(config.FLOOR_FAMILY)
    floor_visual = trimesh.creation.box(extents=config.FLOOR_VISUAL_EXTENTS)
    floor_visual.apply_translation([0, 0, -config.FLOOR_VISUAL_EXTENTS[2] / 2])
    geometry["floor"] = (
        np.asarray(floor_visual.vertices, dtype=np.float32),
        np.asarray(floor_visual.faces, dtype=np.int32),
    )

    # ── Prescribed deforming hand ────────────────────────────────────────────
    # Build one mesh owner in hand-origin coordinates and assign patches once.
    # Prescribed family motion prevents contact reactions from moving the hand;
    # its mass/inertia values are placeholders. Hand/table contacts are disabled.
    if args.case == "grab":
        surface_initial, surface_vertices, _ = sample_grab_hand(hand, start_time, 0, grab_offset)
        surface_faces = hand["hand_faces"]
        solver.DisableContactBetweenFamilies(config.MOVING_FAMILY, config.FLOOR_FAMILY)
        write_wavefront_mesh(output / "surface.obj", surface_vertices, surface_faces)
        surface = solver.AddWavefrontMeshObject(str(output / "surface.obj"), material, False)
        if args.hand_patches == "spread":
            surface.SetPatchIDs(
                partition_mesh_seeded_patches(surface_vertices, surface_faces, config.HAND_PATCH_COUNT).tolist()
            )
        elif args.hand_patches == "connected":
            surface.SetPatchIDs(
                partition_mesh_contact_patches(
                    surface_vertices, surface_faces, args.hand_patch_radius, args.patch_normal_angle
                ).tolist()
            )
        elif args.hand_patches == "triangles":
            surface.SetEachTriangleAsPatch()
        else:
            surface.SetPatchIDs([0] * len(surface_faces))
        surface.SetMass(config.FIXED_OWNER_MASS)
        surface.SetMOI(config.FIXED_OWNER_MOI)
        surface.SetInitPos(surface_initial.tolist())
        surface.SetFamily(config.MOVING_FAMILY)
        surface_tracker = solver.Track(surface)
        geometry["surface"] = (surface_vertices, surface_faces)
        # Prescribe geometry while retaining owner velocity in the contact law.
        solver.SetFamilyPrescribedPosition(config.MOVING_FAMILY)
        solver.SetFamilyPrescribedQuaternion(config.MOVING_FAMILY)
        solver.SetFamilyPrescribedLinVel(config.MOVING_FAMILY)
        solver.SetFamilyPrescribedAngVel(config.MOVING_FAMILY, "0", "0", "0")

    # Save the actual initial contact geometry and patch IDs for visual inspection
    # before Initialize compiles kernels and allocates DEME runtime storage.
    patches = dict(
        cup_vertices=collision_vertices,
        cup_faces=collision_faces,
        cup_patch_ids=np.asarray(cup_mesh.GetPatchIDs(), dtype=np.int32),
    )
    if surface is not None:
        patches.update(
            surface_vertices=surface_vertices,
            surface_faces=surface_faces,
            surface_patch_ids=np.asarray(surface.GetPatchIDs(), dtype=np.int32),
        )
    np.savez_compressed(output / "patches.npz", **patches)
    print(
        f"Contact patches: object={cup_mesh.GetNumPatches()}, hand={surface.GetNumPatches() if surface else 0}",
        flush=True,
    )
    print("Initializing DEME (kernel compilation can take several minutes).", flush=True)
    solver.Initialize()
    print("DEME initialized; starting the scene.", flush=True)

    # ── Optional Newton rendering ────────────────────────────────────────────
    # Newton supplies a mesh viewer here, not a second physics integration. Reuse
    # Warp vertex buffers while DEME provides the dynamic object pose.
    wp.init()
    vis = None
    writer = None
    buffers = {}
    indices = {}
    if not args.no_render:
        model = newton.ModelBuilder(up_axis=newton.Axis.Z).finalize()
        vis = mophi.OpenGLVisualizer(
            model, width=config.WINDOW_WIDTH, height=config.WINDOW_HEIGHT, headless=args.headless
        )
        direction = np.asarray(config.CAMERA_TARGET) - config.CAMERA_POSITION
        vis.set_camera(
            wp.vec3(*config.CAMERA_POSITION),
            float(np.degrees(np.arctan2(direction[2], np.hypot(direction[0], direction[1])))),
            float(np.degrees(np.arctan2(direction[1], direction[0]))),
        )
        for name, (vertices, faces) in geometry.items():
            buffers[name] = wp.array(vertices, dtype=wp.vec3)
            indices[name] = wp.array(faces.ravel(), dtype=wp.int32)

    # ── Independent diagnostic and rendering schedules ──────────────────────
    # Both schedules select physics-step indices; neither changes DEME dt. Save
    # movie-cadence poses even without a live renderer so replay needs no rerun.
    samples, render_samples = [], []
    steps = int(np.ceil(args.duration / args.dt))
    diagnostic_steps = max(1, round(config.GRAB_DIAGNOSTIC_INTERVAL / args.dt)) if args.case == "grab" else 1
    record_steps = set(range(0, steps + 1, diagnostic_steps)) | {steps}
    render_steps = set(
        np.minimum(steps, np.rint(np.arange(0, args.duration, 1 / config.RENDER_FPS) / args.dt).astype(int))
    ) | {steps}
    cup_id = cup_tracker.GetOwnerID()
    completed = False
    next_progress = 0.0
    try:
        if args.save_movie:
            writer = imageio.get_writer(output / "contact.mp4", fps=config.RENDER_FPS)
        for step in range(steps + 1):
            # Map simulation time to the held/slowed recording, command the hand,
            # then advance DEME. Step zero records initialization without advancing.
            # Local node updates preserve topology and the initial patch IDs.
            t = step * physics_dt
            if args.case == "grab":
                source_time = min(stop_time, start_time + max(0.0, t - config.SETTLE_TIME) * args.playback_speed)
                owner_position, current_nodes, owner_velocity = sample_grab_hand(
                    hand, source_time, args.playback_speed if t >= config.SETTLE_TIME else 0.0, grab_offset
                )
                surface_tracker.SetPos(owner_position.tolist())
                surface_tracker.SetVel(owner_velocity.tolist())
                surface_tracker.UpdateMesh(current_nodes.tolist())
            if step:
                solver.DoDynamicsThenSync(physics_dt)
            record = step in record_steps
            render = step in render_steps
            # Read owner state only at diagnostic/movie samples. Transform the full
            # object mesh for display and the proxy for the table-clearance measure.
            if record or render:
                position = np.asarray(cup_tracker.Pos())
                quaternion = np.asarray(cup_tracker.OriQ())
                velocity = np.asarray(cup_tracker.Vel())
                angular_velocity = np.asarray(cup_tracker.AngVelGlobal())
                if not np.isfinite(np.r_[position, quaternion, velocity, angular_velocity]).all():
                    mophi.fatal(f"Nonfinite DEME object state at {t}")
                rotation = Rotation.from_quat(quaternion)
                world_cup = rotation.apply(cup.vertices) + position
                minimum_z = (rotation.apply(collision_vertices) + position)[:, 2].min()
                if render and args.case == "grab":
                    render_samples.append(np.r_[t, source_time, position, quaternion])
                if record:
                    force, torque = solver.GetOwnerContactWrench(cup_id)
                    samples.append(
                        np.r_[t, position, quaternion, velocity, angular_velocity, force[0], torque[0], minimum_z]
                    )
                    # Isolated rest tests retain individual contact forces/points;
                    # offsets delimit each variable-length contact sample for analysis.
                    if args.case == "cup_rest":
                        info = solver.GetContactDetailedInfo(config.CONTACT_OUTPUT_FORCE_THRESHOLD)
                        owners_a = np.asarray(info.GetAOwner(), dtype=np.int64)
                        count = len(owners_a)
                        if count:
                            if np.any(owners_a != cup_id):
                                mophi.fatal("Cup-rest diagnostic expects owner A to be the cup")
                            contact_samples.append(
                                np.column_stack(
                                    [
                                        np.full(count, t),
                                        owners_a,
                                        info.GetBOwner(),
                                        info.GetAGeo(),
                                        info.GetBGeo(),
                                        info.GetPoint(),
                                        info.GetForce(),
                                        info.GetNormal(),
                                        info.GetTorque(),
                                    ]
                                )
                            )
                        contact_offsets.append(contact_offsets[-1] + count)
                    else:
                        # Compare with recorded CoM in the same world frame. Measure
                        # sampled penetration in the object principal frame (positive
                        # signed distance is inside), and audit the hand commands read
                        # back from DEME. These diagnostics do not modify the motion.
                        a, b, w = sample_interval(obj["timestamps_s"], source_time)
                        centers = (
                            Rotation.from_quat(obj["object_quaternion_xyzw"][[a, b]]).apply(cup.center_of_mass)
                            + obj["object_translation_world"][[a, b]]
                            + grab_offset
                        )
                        reference = (1 - w) * centers[0] + w * centers[1]
                        hand_in_cup = rotation.inv().apply(current_nodes + owner_position - position)
                        penetration = max(0.0, float(trimesh.proximity.signed_distance(collision, hand_in_cup).max()))
                        geometry_error = np.max(
                            np.abs(np.asarray(surface_tracker.GetMeshNodesGlobal()) - (current_nodes + owner_position))
                        )
                        velocity_error = np.max(np.abs(np.asarray(surface_tracker.Vel()) - owner_velocity))
                        grab_samples.append(
                            np.r_[
                                source_time,
                                reference,
                                owner_position,
                                owner_velocity,
                                penetration,
                                geometry_error,
                                velocity_error,
                            ]
                        )
                if t + args.dt / 2 >= next_progress:
                    print(f"t={t:.3f} s, object CoM={position.round(5)}", flush=True)
                    next_progress += config.PROGRESS_INTERVAL
                # Send world-space meshes to the renderer at movie cadence only.
                # Closing the window stops the run; it is not marked completed.
                if render and vis is not None:
                    vis.begin_frame(t)
                    if step == 0:
                        mophi.log_orientation_and_scale_reference(
                            vis,
                            axis_origin=config.AXIS_ORIGIN,
                            axis_length=config.AXIS_LENGTH,
                            scale_bar_center=config.SCALE_BAR_CENTER,
                            marker_radius=config.SCALE_MARKER_RADIUS,
                        )
                    for name, (vertices, _) in geometry.items():
                        world = (
                            world_cup
                            if name == "cup"
                            else current_nodes + owner_position if name == "surface" else vertices
                        )
                        buffers[name].assign(np.asarray(world, dtype=np.float32))
                        vis.log_mesh(name, buffers[name], indices[name], color=config.MESH_COLORS[name])
                    vis.end_frame()
                    if writer is not None:
                        writer.append_data(vis.get_frame().numpy())
                    if not vis.is_running():
                        break
            if step == steps:
                completed = True
        if vis is not None and vis.is_running():
            imageio.imwrite(output / "last_frame.png", vis.get_frame().numpy())
    finally:
        if writer is not None:
            writer.close()
        if vis is not None:
            vis.close()

    # ── Persist replay geometry and named diagnostic arrays ──────────────────
    # Separate movie samples from force/trajectory diagnostics so changing render
    # cadence does not change the diagnostics or require replay interpolation.
    data = np.asarray(samples)
    if args.case == "grab":
        poses = np.asarray(render_samples)
        np.savez_compressed(
            output / "render_poses.npz",
            time_s=poses[:, 0],
            source_time_s=poses[:, 1],
            cup_com_world=poses[:, 2:5],
            cup_quaternion_xyzw=poses[:, 5:9],
            world_translation=grab_offset,
            fps=config.RENDER_FPS,
            cup_vertices_principal=cup.vertices,
            cup_faces=cup.triangle_indices,
        )
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
    if args.case == "cup_rest":
        contacts = np.concatenate(contact_samples) if contact_samples else np.empty((0, 17))
        np.savez_compressed(
            output / "contacts.npz",
            sample_offsets=np.asarray(contact_offsets),
            time_s=contacts[:, 0],
            owner_a=contacts[:, 1].astype(np.int64),
            owner_b=contacts[:, 2].astype(np.int64),
            patch_a=contacts[:, 3].astype(np.int64),
            patch_b=contacts[:, 4].astype(np.int64),
            point_world=contacts[:, 5:8],
            force_on_cup_world=contacts[:, 8:11],
            normal_world=contacts[:, 11:14],
            model_torque_on_cup_world=contacts[:, 14:17],
        )

    # ── Acceptance checks ────────────────────────────────────────────────────
    # Evaluate the completed measurements: rest tests require stable support, while
    # grasp tests require sustained lift and bounded command/geometry errors.
    # Tolerances affect the verdict only; they are not enforced by the solver.
    settled = data[:, 0] >= max(config.SETTLE_TIME, data[-1, 0] - config.SETTLE_TIME)
    mean_vx = float(np.mean(data[settled, 8])) if settled.any() else float(data[-1, 8])
    mean_contact_force = np.mean(data[settled, 14:17], axis=0) if settled.any() else data[-1, 14:17]
    checks = dict(
        completed=completed,
        time_accounting=bool(abs(solver.GetSimTime() - step * physics_dt) < physics_dt * 1e-3),
        floor_penetration_bounded=bool(data[:, 20].min() >= -config.MAX_ALLOWED_PENETRATION),
        contact_occurred=bool(np.max(np.linalg.norm(data[:, 14:17], axis=1)) > config.CUP_MASS * config.GRAVITY / 2),
    )
    if args.case == "cup_rest":
        rotations = Rotation.from_quat(data[:, 4:8])
        angle = np.degrees((rotations * rotations[0].inv()).magnitude())
        checks.update(
            angular_drift_bounded=bool(angle.max() < config.REST_MAX_ANGULAR_DRIFT_DEG),
            angular_speed_settled=bool(np.linalg.norm(data[-1, 11:14]) < config.REST_MAX_ANGULAR_SPEED),
            support_balances_weight=bool(
                abs(mean_contact_force[2] / (config.CUP_MASS * config.GRAVITY) - 1)
                < config.MAX_WEIGHT_BALANCE_RELATIVE_ERROR
            ),
        )
    else:
        grasp = np.asarray(grab_samples)
        np.savez_compressed(
            output / "grasp_diagnostics.npz",
            time_s=data[:, 0],
            source_time_s=grasp[:, 0],
            recorded_cup_com_world=grasp[:, 1:4],
            hand_origin_world=grasp[:, 4:7],
            hand_owner_velocity_world=grasp[:, 7:10],
            sampled_hand_penetration_m=grasp[:, 10],
            hand_geometry_error_m=grasp[:, 11],
            hand_velocity_error_m_s=grasp[:, 12],
        )
        checks.update(assess_grab_run(data, grasp, config))

    # ── Run provenance and verdict ───────────────────────────────────────────
    # Save settings and individual checks before optionally returning a failed
    # --check exit status. A finished simulation can still fail physical criteria.
    summary = {
        "case": args.case,
        "deme_version": DEME.__version__,
        "dt_s": physics_dt,
        "duration_s": float(data[-1, 0]),
        "solver_time_s": solver.GetSimTime(),
        "cup_mass_kg": cup.mass,
        "source_object": str(args.object.resolve()),
        "source_hand": str(args.hand.resolve()) if args.case == "grab" else None,
        "mass_model": "uniform density inside the source closed mesh",
        "cup_collision_triangles": len(collision_faces),
        "bin_size_m": args.bin_size,
        "domain_bounds_m": domain_bounds,
        "cup_patch_mode": args.cup_patches,
        "cup_patch_count": cup_mesh.GetNumPatches(),
        "hand_patch_mode": args.hand_patches if args.case == "grab" else None,
        "surface_patch_count": surface.GetNumPatches() if surface is not None else 0,
        "cup_patch_radius_m": args.cup_patch_radius,
        "hand_patch_radius_m": args.hand_patch_radius,
        "patch_normal_angle_deg": args.patch_normal_angle,
        "recorded_initial_frame": args.start_frame,
        "cup_visual_triangles": len(cup.triangle_indices),
        "proxy_sampled_surface_error_m": proxy_error,
        "proxy_relative_volume_error": float(proxy_volume_error),
        "friction": args.friction,
        "surface_owner_velocity_world_m_s": surface_tracker.Vel() if surface_tracker is not None else None,
        "material_young_modulus_Pa": config.CONTACT_YOUNG_MODULUS,
        "material_poisson_ratio": config.CONTACT_POISSON_RATIO,
        "restitution": config.CONTACT_RESTITUTION,
        "push_floor_friction": args.floor_friction,
        "gravity_m_s2": config.GRAVITY,
        "settle_time_s": config.SETTLE_TIME,
        "cup_displacement_x_m": float(data[-1, 1] - data[0, 1]),
        "late_mean_cup_velocity_x_m_s": mean_vx,
        "late_mean_contact_force_N": mean_contact_force.tolist(),
        "minimum_cup_z_m": float(data[:, 20].min()),
        "peak_sampled_contact_force_N": float(np.max(np.linalg.norm(data[:, 14:17], axis=1))),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if args.case == "cup_rest":
        summary.update(
            maximum_rotation_deg=float(angle.max()),
            final_angular_velocity_world=data[-1, 11:14].tolist(),
            maximum_active_contacts=int(np.diff(contact_offsets).max()),
            diagnostic_interval_s=physics_dt,
        )
    if args.case == "grab":
        summary.update(
            maximum_prescribed_nodal_speed_m_s=maximum_nodal_speed,
            start_frame=args.start_frame,
            stop_frame=args.stop_frame,
            playback_speed=args.playback_speed,
            hand_velocity_model="origin translation only; zero owner angular velocity; prescribed local deformation",
            final_cup_bottom_height_m=float(data[-1, 20]),
            final_recorded_com_error_m=float(np.linalg.norm(data[-1, 1:4] - grasp[-1, 1:4])),
            maximum_sampled_hand_penetration_m=float(grasp[:, 10].max()),
            maximum_hand_geometry_error_m=float(grasp[:, 11].max()),
            maximum_hand_owner_velocity_error_m_s=float(grasp[:, 12].max()),
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if args.check and not summary["passed"]:
        mophi.fatal(f"Scene acceptance checks failed: {output / 'summary.json'}")
