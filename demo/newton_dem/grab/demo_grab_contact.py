"""DEME cup contact, isolated resting-cup diagnostics, and recorded grasp experiments."""

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

# Scenario parameters are shared with the viewer in scene_config.py.
from scene_config import *


def sample_grab_hand(hand, source_time, playback_speed, offset):
    """Linear nodal motion with the exact segment derivative for owner velocity.

    Owner orientation is identity: all rotation/articulation remains in local
    nodes, and only the origin derivative contributes to contact velocity.
    """
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


def assess_grab_run(data, grasp):
    """Distinguish sustained contact-supported lift from mere upward flight."""
    checks = {}
    checks["hand_geometry_matches_command"] = bool(grasp[:, 11].max() < GRAB_TRANSFER_TOLERANCE)
    checks["hand_owner_velocity_matches_command"] = bool(grasp[:, 12].max() < GRAB_TRANSFER_TOLERANCE)
    checks["sampled_hand_penetration_bounded"] = bool(grasp[:, 10].max() < MAX_ALLOWED_PENETRATION)
    lift_window = data[:, 0] >= data[-1, 0] - GRAB_LIFT_HOLD_INTERVAL
    checks["cup_lifted"] = bool(np.min(data[lift_window, 20]) > MIN_GRAB_LIFT)
    checks["lift_supported_by_contact"] = bool(
        checks["cup_lifted"] and np.mean(data[lift_window, 16]) > CUP_MASS * GRAVITY * GRAB_SUPPORT_FORCE_FRACTION
    )
    checks["cup_near_recorded_position"] = bool(
        np.linalg.norm(data[-1, 1:4] - grasp[-1, 1:4]) < MAX_GRAB_REFERENCE_ERROR
    )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("grab", "cup_rest"), default=CASE)
    parser.add_argument("--object", type=Path, default=OBJECT_PATH)
    parser.add_argument("--hand", type=Path, default=HAND_PATH)
    parser.add_argument("--start-frame", type=int, default=GRAB_START_FRAME)
    parser.add_argument("--stop-frame", type=int, default=GRAB_STOP_FRAME)
    parser.add_argument("--playback-speed", type=float, default=PLAYBACK_SPEED)
    parser.add_argument("--dt", type=float, default=DEME_DT)
    parser.add_argument(
        "--duration",
        type=float,
        default=DURATION,
        help="Cup-rest duration; grab derives it from source interval and playback speed",
    )
    parser.add_argument("--friction", type=float, default=CONTACT_FRICTION)
    parser.add_argument("--bin-size", type=float, default=BIN_SIZE)
    parser.add_argument("--compact-rest-domain", action="store_true", default=COMPACT_REST_DOMAIN)
    parser.add_argument("--floor-friction", type=float, default=FLOOR_FRICTION)
    parser.add_argument(
        "--cup-patches", choices=("single", "sectors", "triangles", "connected", "spread"), default=CUP_PATCH_MODE
    )
    parser.add_argument(
        "--hand-patches", choices=("single", "triangles", "connected", "spread"), default=HAND_PATCH_MODE
    )
    parser.add_argument("--cup-patch-radius", type=float, default=CUP_PATCH_RADIUS)
    parser.add_argument("--hand-patch-radius", type=float, default=HAND_PATCH_RADIUS)
    parser.add_argument("--patch-normal-angle", type=float, default=PATCH_MAX_NORMAL_ANGLE)
    parser.add_argument("--patch-sectors", type=int, default=CUP_PATCH_SECTORS)
    parser.add_argument("--collision-faces", type=int, default=COLLISION_FACE_COUNT)
    parser.add_argument("--headless", action="store_true", default=HEADLESS)
    parser.add_argument("--no-render", action="store_true", default=not RENDER)
    parser.add_argument("--save-movie", action="store_true", default=SAVE_MOVIE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
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
        selected = (hand["timestamps_s"] >= start_time) & (hand["timestamps_s"] <= stop_time)
        world_nodes = hand["hand_vertices_local"][selected] + hand["hand_origin_world"][selected, None, :]
        nodal_velocity = (
            np.diff(world_nodes, axis=0) / np.diff(hand["timestamps_s"][selected])[:, None, None] * args.playback_speed
        )
        maximum_nodal_speed = float(np.linalg.norm(nodal_velocity, axis=2).max())
        if maximum_nodal_speed > MAX_VELOCITY:
            mophi.fatal("Selected hand motion exceeds MAX_VELOCITY; reduce playback speed")
        args.duration = SETTLE_TIME + (stop_time - start_time) / args.playback_speed
    if args.case == "cup_rest":
        start_time = args.start_frame / float(obj["source_fps"])
        if not obj["timestamps_s"][0] <= start_time <= obj["timestamps_s"][-1]:
            mophi.fatal("--start-frame is outside the object recording")
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
    row, _, weight = sample_interval(obj["timestamps_s"], start_time)
    if weight != 0:
        mophi.fatal("Grasp initialization requires an object sample at --start-frame")
    recorded_rotation = Rotation.from_quat(obj["object_quaternion_xyzw"][row])
    recorded_translation = obj["object_translation_world"][row]
    cup_rotation = recorded_rotation * Rotation.from_matrix(cup.principal_to_input)
    recorded_world = recorded_rotation.apply(obj["object_vertices_local"]) + recorded_translation
    grab_offset = np.array(
        [-recorded_world[:, 0].mean(), -recorded_world[:, 1].mean(), REST_GAP - recorded_world[:, 2].min()]
    )
    cup_initial = recorded_rotation.apply(cup.center_of_mass) + recorded_translation + grab_offset
    if args.case == "grab":
        domain = np.asarray([DOMAIN_X, DOMAIN_Y, DOMAIN_Z])
        motion_min = world_nodes.min(axis=(0, 1)) + grab_offset
        motion_max = world_nodes.max(axis=(0, 1)) + grab_offset
        if np.any(motion_min < domain[:, 0]) or np.any(motion_max > domain[:, 1]):
            mophi.fatal("Selected hand trajectory leaves the configured DEME domain; adjust DOMAIN_X/Y/Z")

    solver = DEME.DEMSolver([CUDA_DEVICE])
    solver.SetVerbosity("WARNING")
    solver.SetMeshUniversalContact(True)
    if args.case == "cup_rest":
        solver.SetContactOutputContent(["OWNER", "GEO_ID", "FORCE", "POINT", "NORMAL", "TORQUE"])
    solver.SetTimeStepSize(physics_dt)
    solver.SetGravitationalAcceleration([0.0, 0.0, -GRAVITY])
    domain_bounds = (
        (REST_DOMAIN_X, REST_DOMAIN_Y, REST_DOMAIN_Z) if args.compact_rest_domain else (DOMAIN_X, DOMAIN_Y, DOMAIN_Z)
    )
    solver.InstructBoxDomainDimension(*domain_bounds)
    solver.SetInitBinSize(args.bin_size)
    solver.SetMaxVelocity(MAX_VELOCITY)
    solver.SetExpandSafetyAdder(max(DETECTION_SAFETY_SPEED, GRAB_DETECTION_SPEED_FACTOR * maximum_nodal_speed))
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
            "mu": args.floor_friction,
            "Crr": 0.0,
        }
    )
    solver.SetMaterialPropertyPair("mu", material, floor_material, args.floor_friction)
    write_wavefront_mesh(output / "cup_principal.obj", collision_vertices, collision_faces)
    cup_mesh = solver.AddWavefrontMeshObject(str(output / "cup_principal.obj"), material, False)
    if args.cup_patches == "spread":
        cup_mesh.SetPatchIDs(
            partition_mesh_seeded_patches(collision_vertices, collision_faces, CUP_PATCH_COUNT).tolist()
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
    cup_mesh.SetFamily(CUP_FAMILY)
    cup_tracker = solver.Track(cup_mesh)
    geometry = {"cup": (cup.vertices, cup.triangle_indices)}
    surface_tracker = None
    surface_initial = np.zeros(3)
    surface_vertices = None
    surface = None

    floor = solver.AddBCPlane([0, 0, 0], [0, 0, 1], floor_material)
    floor.SetFamily(FLOOR_FAMILY)
    solver.SetFamilyFixed(FLOOR_FAMILY)
    floor_visual = trimesh.creation.box(extents=FLOOR_VISUAL_EXTENTS)
    floor_visual.apply_translation([0, 0, -FLOOR_VISUAL_EXTENTS[2] / 2])
    geometry["floor"] = (
        np.asarray(floor_visual.vertices, dtype=np.float32),
        np.asarray(floor_visual.faces, dtype=np.int32),
    )
    if args.case == "grab":
        surface_initial, surface_vertices, _ = sample_grab_hand(hand, start_time, 0, grab_offset)
        surface_faces = hand["hand_faces"]
        solver.DisableContactBetweenFamilies(MOVING_FAMILY, FLOOR_FAMILY)
        write_wavefront_mesh(output / "surface.obj", surface_vertices, surface_faces)
        surface = solver.AddWavefrontMeshObject(str(output / "surface.obj"), material, False)
        if args.hand_patches == "spread":
            surface.SetPatchIDs(
                partition_mesh_seeded_patches(surface_vertices, surface_faces, HAND_PATCH_COUNT).tolist()
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
        surface.SetMass(FIXED_OWNER_MASS)
        surface.SetMOI(FIXED_OWNER_MOI)
        surface.SetInitPos(surface_initial.tolist())
        surface.SetFamily(MOVING_FAMILY)
        surface_tracker = solver.Track(surface)
        geometry["surface"] = (surface_vertices, surface_faces)
        # Prescribe geometry while retaining owner velocity in the contact law.
        solver.SetFamilyPrescribedPosition(MOVING_FAMILY)
        solver.SetFamilyPrescribedQuaternion(MOVING_FAMILY)
        solver.SetFamilyPrescribedLinVel(MOVING_FAMILY)
        solver.SetFamilyPrescribedAngVel(MOVING_FAMILY, "0", "0", "0")
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
        f"Contact patches: cup={cup_mesh.GetNumPatches()}, hand={surface.GetNumPatches() if surface else 0}", flush=True
    )
    print("Initializing DEME (kernel compilation can take several minutes).", flush=True)
    solver.Initialize()
    print("DEME initialized; starting the scene.", flush=True)
    wp.init()
    vis = None
    writer = None
    buffers = {}
    indices = {}
    if not args.no_render:
        model = newton.ModelBuilder(up_axis=newton.Axis.Z).finalize()
        vis = mophi.OpenGLVisualizer(model, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, headless=args.headless)
        direction = np.asarray(CAMERA_TARGET) - CAMERA_POSITION
        vis.set_camera(
            wp.vec3(*CAMERA_POSITION),
            float(np.degrees(np.arctan2(direction[2], np.hypot(direction[0], direction[1])))),
            float(np.degrees(np.arctan2(direction[1], direction[0]))),
        )
        for name, (vertices, faces) in geometry.items():
            buffers[name] = wp.array(vertices, dtype=wp.vec3)
            indices[name] = wp.array(faces.ravel(), dtype=wp.int32)
    samples, render_samples = [], []
    steps = int(np.ceil(args.duration / args.dt))
    diagnostic_steps = max(1, round(GRAB_DIAGNOSTIC_INTERVAL / args.dt)) if args.case == "grab" else 1
    record_steps = set(range(0, steps + 1, diagnostic_steps)) | {steps}
    render_steps = set(
        np.minimum(steps, np.rint(np.arange(0, args.duration, 1 / RENDER_FPS) / args.dt).astype(int))
    ) | {steps}
    cup_id = cup_tracker.GetOwnerID()
    completed = False
    next_progress = 0.0
    try:
        if args.save_movie:
            writer = imageio.get_writer(output / "contact.mp4", fps=RENDER_FPS)
        for step in range(steps + 1):
            t = step * physics_dt
            if args.case == "grab":
                source_time = min(stop_time, start_time + max(0.0, t - SETTLE_TIME) * args.playback_speed)
                owner_position, current_nodes, owner_velocity = sample_grab_hand(
                    hand, source_time, args.playback_speed if t >= SETTLE_TIME else 0.0, grab_offset
                )
                surface_tracker.SetPos(owner_position.tolist())
                surface_tracker.SetVel(owner_velocity.tolist())
                surface_tracker.UpdateMesh(current_nodes.tolist())
            if step:
                solver.DoDynamicsThenSync(physics_dt)
            record = step in record_steps
            render = step in render_steps
            if record or render:
                position = np.asarray(cup_tracker.Pos())
                quaternion = np.asarray(cup_tracker.OriQ())
                velocity = np.asarray(cup_tracker.Vel())
                angular_velocity = np.asarray(cup_tracker.AngVelGlobal())
                if not np.isfinite(np.r_[position, quaternion, velocity, angular_velocity]).all():
                    mophi.fatal(f"Nonfinite DEME cup state at {t}")
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
                    if args.case == "cup_rest":
                        info = solver.GetContactDetailedInfo(CONTACT_OUTPUT_FORCE_THRESHOLD)
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
                    print(f"t={t:.3f} s, cup CoM={position.round(5)}", flush=True)
                    next_progress += PROGRESS_INTERVAL
                if render and vis is not None:
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
                        world = (
                            world_cup
                            if name == "cup"
                            else current_nodes + owner_position if name == "surface" else vertices
                        )
                        buffers[name].assign(np.asarray(world, dtype=np.float32))
                        vis.log_mesh(name, buffers[name], indices[name], color=MESH_COLORS[name])
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
            fps=RENDER_FPS,
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
    settled = data[:, 0] >= max(SETTLE_TIME, data[-1, 0] - SETTLE_TIME)
    mean_vx = float(np.mean(data[settled, 8])) if settled.any() else float(data[-1, 8])
    mean_contact_force = np.mean(data[settled, 14:17], axis=0) if settled.any() else data[-1, 14:17]
    checks = dict(
        completed=completed,
        time_accounting=bool(abs(solver.GetSimTime() - step * physics_dt) < physics_dt * 1e-3),
        floor_penetration_bounded=bool(data[:, 20].min() >= -MAX_ALLOWED_PENETRATION),
        contact_occurred=bool(np.max(np.linalg.norm(data[:, 14:17], axis=1)) > CUP_MASS * GRAVITY / 2),
    )
    if args.case == "cup_rest":
        rotations = Rotation.from_quat(data[:, 4:8])
        angle = np.degrees((rotations * rotations[0].inv()).magnitude())
        checks.update(
            angular_drift_bounded=bool(angle.max() < REST_MAX_ANGULAR_DRIFT_DEG),
            angular_speed_settled=bool(np.linalg.norm(data[-1, 11:14]) < REST_MAX_ANGULAR_SPEED),
            support_balances_weight=bool(
                abs(mean_contact_force[2] / (CUP_MASS * GRAVITY) - 1) < MAX_WEIGHT_BALANCE_RELATIVE_ERROR
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
        checks.update(assess_grab_run(data, grasp))
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
        "material_young_modulus_Pa": CONTACT_YOUNG_MODULUS,
        "material_poisson_ratio": CONTACT_POISSON_RATIO,
        "restitution": CONTACT_RESTITUTION,
        "push_floor_friction": args.floor_friction,
        "gravity_m_s2": GRAVITY,
        "settle_time_s": SETTLE_TIME,
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


if __name__ == "__main__":
    main()
