"""Build the Chrono-reference ABB IRB6700 concrete printer in Newton."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import mophi
import newton
from newton import JointTargetMode

_BODY_NAMES = (
    "IRB6700-MH3_150-320_IRC5_rev01_BASE_CAD-1",
    "IRB6700_155-285_IRC5_rev03_LINK01_CAD-1",
    "IRB6700_155-285_IRC5_rev01_LINK02_CAD-1",
    "IRB6700_155-285_IRC5_rev02_LINK03_CAD-1",
    "IRB6700_155-285_IRC5_rev01_LINK04_CAD-1",
    "IRB6700-MH3_150-320_IRC5_rev04_LINK05_CAD.step-1",
    "IRB6700_155-285_IRC5_rev01_LINK06_CAD-1",
    "beam_011025-1",
    "Auger_002_RB_out-1",
    "Auger_In-1",
)

_BODY_LABELS = (
    "abb_base",
    "abb_shoulder",
    "abb_upper_arm",
    "abb_elbow",
    "abb_forearm",
    "abb_wrist",
    "abb_end_effector",
    "printer_support_beam",
    "auger_outer_housing",
    "auger_inner_screw",
)

_BODY_COLORS = (
    (0.16, 0.16, 0.18),
    (0.95, 0.34, 0.06),
    (0.95, 0.34, 0.06),
    (0.95, 0.34, 0.06),
    (0.95, 0.34, 0.06),
    (0.95, 0.34, 0.06),
    (0.20, 0.20, 0.22),
    (0.32, 0.34, 0.37),
    (0.20, 0.22, 0.25),
    (0.48, 0.50, 0.52),
)


@dataclass(frozen=True)
class ABBPrinterModel:
    """Newton indices and TCP transform needed by the printing demo."""

    builder: newton.ModelBuilder
    body_indices: dict[str, int]
    nozzle_body_index: int
    nozzle_position_local: tuple[float, float, float]
    nozzle_rotation_local: tuple[float, float, float, float]
    nozzle_position_world: tuple[float, float, float]
    nozzle_rotation_world: tuple[float, float, float, float]


def _quat_multiply(lhs, rhs):
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return np.array(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float64,
    )


def _quat_rotate(quaternion, vector):
    xyz = np.asarray(quaternion[:3], dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + quaternion[3] * vector)


def _transform_multiply(lhs, rhs):
    lhs_position, lhs_rotation = lhs
    rhs_position, rhs_rotation = rhs
    return (
        np.asarray(lhs_position) + _quat_rotate(lhs_rotation, rhs_position),
        _quat_multiply(lhs_rotation, rhs_rotation),
    )


def _transform_inverse(transform):
    position, rotation = transform
    inverse_rotation = np.array((-rotation[0], -rotation[1], -rotation[2], rotation[3]))
    return -_quat_rotate(inverse_rotation, position), inverse_rotation


def _json_transform(value):
    value = value.get("m_csys", value)
    position = value["pos"]
    rotation = value["rot"]
    return (
        np.array((position["x"], position["y"], position["z"]), dtype=np.float64),
        np.array((rotation["e1"], rotation["e2"], rotation["e3"], rotation["e0"]), dtype=np.float64),
    )


def _wp_transform(transform):
    position, rotation = transform
    return wp.transform(wp.vec3(*position), wp.quat(*rotation))


def _to_newton_world(transform):
    # The Chrono assembly uses Y-up. Rotate its world frame about +X so Newton is Z-up.
    chrono_to_newton = (
        np.zeros(3, dtype=np.float64),
        np.array((np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)), dtype=np.float64),
    )
    return _transform_multiply(chrono_to_newton, transform)


def build_abb_irb6700_printer(
    asset_directory: Path,
    joint_stiffness: float,
    joint_damping: float,
    auger_stiffness: float,
    auger_damping: float,
) -> ABBPrinterModel:
    """Load the pinned Chrono CAD export and reconstruct its articulation in Newton."""
    asset_directory = Path(asset_directory)
    assembly_path = asset_directory / "Robot_export_j2.json"
    if not assembly_path.is_file():
        raise FileNotFoundError(f"ABB printer assembly metadata is missing: {assembly_path}")

    assembly = json.loads(assembly_path.read_text(encoding="utf-8"))["sys"]["assembly"]
    bodies_by_name = {body["m_name"]: body for body in assembly["bodies"]}
    missing_bodies = [name for name in _BODY_NAMES if name not in bodies_by_name]
    if missing_bodies:
        raise ValueError(f"ABB printer assembly is missing bodies: {missing_bodies}")

    marker_body = next(body for body in assembly["bodies"] if body["m_name"] == "SLDW_GROUND")
    markers = {
        marker["m_name"]: _to_newton_world(_json_transform(marker["m_csys"])) for marker in marker_body["markers"]
    }
    missing_markers = [f"Link{index:02d}" for index in range(1, 11) if f"Link{index:02d}" not in markers]
    if missing_markers:
        raise ValueError(f"ABB printer assembly is missing frames: {missing_markers}")

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    body_indices = {}
    body_world_transforms = {}

    for body_name, body_label, color in zip(_BODY_NAMES, _BODY_LABELS, _BODY_COLORS, strict=True):
        body = bodies_by_name[body_name]
        body_world = _to_newton_world(_json_transform(body["m_csys"]))
        body_world_transforms[body_name] = body_world
        variables = body["variables"]
        inertia = np.asarray(variables["inertia"]["data"], dtype=np.float64).reshape((3, 3))
        body_index = builder.add_link(
            xform=_wp_transform(body_world),
            com=wp.vec3(0.0),
            mass=variables["mass"],
            inertia=wp.mat33(inertia),
            label=body_label,
            lock_inertia=True,
        )
        body_indices[body_name] = body_index

        shape_record = body["visual_model"]["m_shapes"][0]
        mesh_path = asset_directory / Path(shape_record["first"]["filename"]).name
        if not mesh_path.is_file():
            raise FileNotFoundError(f"ABB printer body mesh is missing: {mesh_path}")
        surface = mophi.load_obj(str(mesh_path))
        mesh = newton.Mesh(
            surface.vertices,
            surface.indices,
            compute_inertia=False,
            is_solid=False,
            color=color,
        )
        shape_transform = _json_transform(body["ref_to_com"])
        collision_enabled = body_name == "Auger_002_RB_out-1"
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=collision_enabled,
            has_particle_collision=collision_enabled,
        )
        builder.add_shape_mesh(
            body_index,
            xform=_wp_transform(shape_transform),
            mesh=mesh,
            cfg=shape_cfg,
            label=f"{body_label}_mesh",
        )

    base_index = body_indices[_BODY_NAMES[0]]
    joint_indices = []
    joint_indices.append(
        builder.add_joint_fixed(
            -1,
            base_index,
            parent_xform=_wp_transform(body_world_transforms[_BODY_NAMES[0]]),
            child_xform=wp.transform_identity(),
            label="abb_world_mount",
        )
    )

    for joint_number in range(1, 7):
        parent_name = _BODY_NAMES[joint_number - 1]
        child_name = _BODY_NAMES[joint_number]
        joint_world = markers[f"Link{joint_number:02d}"]
        parent_joint = _transform_multiply(_transform_inverse(body_world_transforms[parent_name]), joint_world)
        child_joint = _transform_multiply(_transform_inverse(body_world_transforms[child_name]), joint_world)
        joint_indices.append(
            builder.add_joint_revolute(
                body_indices[parent_name],
                body_indices[child_name],
                parent_xform=_wp_transform(parent_joint),
                child_xform=_wp_transform(child_joint),
                axis=wp.vec3(0.0, 0.0, 1.0),
                target_pos=0.0,
                target_ke=joint_stiffness,
                target_kd=joint_damping,
                limit_lower=-np.pi,
                limit_upper=np.pi,
                actuator_mode=JointTargetMode.POSITION,
                label=f"abb_joint_{joint_number}",
            )
        )

    for marker_name, parent_name, child_name, label in (
        ("Link07", _BODY_NAMES[6], _BODY_NAMES[7], "end_effector_to_beam"),
        ("Link08", _BODY_NAMES[7], _BODY_NAMES[8], "beam_to_auger_housing"),
    ):
        joint_world = markers[marker_name]
        parent_joint = _transform_multiply(_transform_inverse(body_world_transforms[parent_name]), joint_world)
        child_joint = _transform_multiply(_transform_inverse(body_world_transforms[child_name]), joint_world)
        joint_indices.append(
            builder.add_joint_fixed(
                body_indices[parent_name],
                body_indices[child_name],
                parent_xform=_wp_transform(parent_joint),
                child_xform=_wp_transform(child_joint),
                label=label,
            )
        )

    outer_name = _BODY_NAMES[8]
    inner_name = _BODY_NAMES[9]
    auger_joint_world = markers["Link09"]
    joint_indices.append(
        builder.add_joint_revolute(
            body_indices[outer_name],
            body_indices[inner_name],
            parent_xform=_wp_transform(
                _transform_multiply(_transform_inverse(body_world_transforms[outer_name]), auger_joint_world)
            ),
            child_xform=_wp_transform(
                _transform_multiply(_transform_inverse(body_world_transforms[inner_name]), auger_joint_world)
            ),
            axis=wp.vec3(0.0, 0.0, 1.0),
            target_pos=0.0,
            target_ke=auger_stiffness,
            target_kd=auger_damping,
            actuator_mode=JointTargetMode.POSITION,
            label="auger_rotation_joint",
        )
    )
    builder.add_articulation(joint_indices, label="abb_irb6700_printer")

    nozzle_world = markers["Link10"]
    nozzle_local = _transform_multiply(_transform_inverse(body_world_transforms[outer_name]), nozzle_world)
    return ABBPrinterModel(
        builder=builder,
        body_indices=body_indices,
        nozzle_body_index=body_indices[outer_name],
        nozzle_position_local=tuple(nozzle_local[0]),
        nozzle_rotation_local=tuple(nozzle_local[1]),
        nozzle_position_world=tuple(nozzle_world[0]),
        nozzle_rotation_world=tuple(nozzle_world[1]),
    )
