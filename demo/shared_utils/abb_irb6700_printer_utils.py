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

_NOZZLE_CONTACT_BODY_NAME = "Auger_002_RB_out-1"
_NOZZLE_CONTACT_MESH_FILENAME = "body_10_1_inner.obj"


@dataclass(frozen=True)
class ABBPrinterMesh:
    """One printer triangle mesh expressed in its Newton body's local frame."""

    label: str
    body_index: int
    vertices_local: np.ndarray
    triangles: np.ndarray
    is_nozzle_contact_mesh: bool


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
    nozzle_body_position_world: tuple[float, float, float]
    nozzle_body_rotation_world: tuple[float, float, float, float]
    nozzle_contact_mesh_path: Path
    nozzle_contact_position_local: tuple[float, float, float]
    nozzle_contact_rotation_local: tuple[float, float, float, float]
    nozzle_contact_line_starts_local: np.ndarray
    nozzle_contact_line_ends_local: np.ndarray
    nozzle_contact_vertex_count: int
    nozzle_contact_triangle_count: int
    nozzle_contact_edge_count: int
    meshes: tuple[ABBPrinterMesh, ...]


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


def _validated_contact_wireframe(surface, shape_transform, *, allow_boundary_edges=False):
    """Validate a triangle contact surface and return its unique body-local edges."""
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    indices = np.asarray(surface.indices, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("The nozzle contact mesh must contain finite three-dimensional vertices.")
    if indices.ndim != 1 or indices.size == 0 or indices.size % 3 != 0:
        raise ValueError("The nozzle contact mesh must contain a non-empty triangle index list.")
    if indices.min() < 0 or indices.max() >= len(vertices):
        raise ValueError("The nozzle contact mesh contains an out-of-range vertex index.")

    triangles = indices.reshape((-1, 3))
    edge_01 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edge_02 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    doubled_areas = np.linalg.norm(
        np.cross(edge_01, edge_02),
        axis=1,
    )
    if np.any(doubled_areas <= 1.0e-12):
        raise ValueError("The nozzle contact mesh contains degenerate triangles.")

    # The CAD OBJ repeats vertices along visual seams. Weld coincident positions
    # for the topology audit and wireframe so every physical edge is drawn once.
    _, welded_indices = np.unique(np.round(vertices, decimals=8), axis=0, return_inverse=True)
    welded_triangles = welded_indices[triangles]
    edges = np.sort(
        np.concatenate(
            (
                welded_triangles[:, (0, 1)],
                welded_triangles[:, (1, 2)],
                welded_triangles[:, (2, 0)],
            )
        ),
        axis=1,
    )
    unique_edges, edge_incidence = np.unique(edges, axis=0, return_counts=True)
    boundary_count = int(np.count_nonzero(edge_incidence == 1))
    nonmanifold_count = int(np.count_nonzero(edge_incidence > 2))
    if nonmanifold_count or (boundary_count and not allow_boundary_edges):
        raise ValueError(
            "The nozzle contact mesh has invalid topology after welding coincident vertices: "
            f"{boundary_count} boundary edge(s), {nonmanifold_count} non-manifold edge(s)."
        )

    representative_indices = np.full(welded_indices.max() + 1, -1, dtype=np.int64)
    representative_indices[welded_indices[::-1]] = np.arange(len(vertices) - 1, -1, -1)
    edge_vertex_indices = representative_indices[unique_edges]
    shape_position, shape_rotation = shape_transform
    body_local_vertices = np.asarray(
        [shape_position + _quat_rotate(shape_rotation, vertex) for vertex in vertices],
        dtype=np.float32,
    )
    return (
        body_local_vertices[edge_vertex_indices[:, 0]],
        body_local_vertices[edge_vertex_indices[:, 1]],
        len(vertices),
        len(triangles),
        len(unique_edges),
    )


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
    nozzle_contact_wireframe = None
    nozzle_contact_mesh_path = None
    nozzle_contact_shape_transform = None
    printer_meshes = []

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
        shape_position, shape_rotation = shape_transform
        vertices_local = np.asarray(
            [shape_position + _quat_rotate(shape_rotation, vertex) for vertex in surface.vertices],
            dtype=np.float64,
        )
        printer_meshes.append(
            ABBPrinterMesh(
                label=body_label,
                body_index=body_index,
                vertices_local=vertices_local,
                triangles=np.asarray(surface.indices, dtype=np.int64).reshape((-1, 3)),
                is_nozzle_contact_mesh=False,
            )
        )
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=False,
            has_particle_collision=False,
        )
        builder.add_shape_mesh(
            body_index,
            xform=_wp_transform(shape_transform),
            mesh=mesh,
            cfg=shape_cfg,
            label=f"{body_label}_mesh",
        )

        if body_name == _NOZZLE_CONTACT_BODY_NAME:
            # Keep the complete CAD housing as the visible shape, but use only
            # its exposed cavity wall for contact. The open surface prevents a
            # cohesive particle from being captured by the thin wall's exterior.
            nozzle_contact_mesh_path = asset_directory / _NOZZLE_CONTACT_MESH_FILENAME
            if not nozzle_contact_mesh_path.is_file():
                raise FileNotFoundError(f"ABB printer nozzle contact mesh is missing: {nozzle_contact_mesh_path}")
            contact_surface = mophi.load_obj(str(nozzle_contact_mesh_path))
            contact_mesh = newton.Mesh(
                contact_surface.vertices,
                contact_surface.indices,
                compute_inertia=False,
                is_solid=False,
                color=color,
            )
            contact_vertices_local = np.asarray(
                [shape_position + _quat_rotate(shape_rotation, vertex) for vertex in contact_surface.vertices],
                dtype=np.float64,
            )
            printer_meshes.append(
                ABBPrinterMesh(
                    label=f"{body_label}_contact_proxy",
                    body_index=body_index,
                    vertices_local=contact_vertices_local,
                    triangles=np.asarray(contact_surface.indices, dtype=np.int64).reshape((-1, 3)),
                    is_nozzle_contact_mesh=True,
                )
            )
            nozzle_contact_wireframe = _validated_contact_wireframe(
                contact_surface, shape_transform, allow_boundary_edges=True
            )
            nozzle_contact_shape_transform = shape_transform
            builder.add_shape_mesh(
                body_index,
                xform=_wp_transform(shape_transform),
                mesh=contact_mesh,
                cfg=newton.ModelBuilder.ShapeConfig(
                    has_shape_collision=True,
                    has_particle_collision=True,
                    is_visible=False,
                ),
                label=f"{body_label}_contact_proxy",
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
    if nozzle_contact_wireframe is None or nozzle_contact_mesh_path is None or nozzle_contact_shape_transform is None:
        raise ValueError("The ABB printer assembly did not provide an outer-auger contact mesh.")
    line_starts, line_ends, vertex_count, triangle_count, edge_count = nozzle_contact_wireframe
    return ABBPrinterModel(
        builder=builder,
        body_indices=body_indices,
        nozzle_body_index=body_indices[outer_name],
        nozzle_position_local=tuple(nozzle_local[0]),
        nozzle_rotation_local=tuple(nozzle_local[1]),
        nozzle_position_world=tuple(nozzle_world[0]),
        nozzle_rotation_world=tuple(nozzle_world[1]),
        nozzle_body_position_world=tuple(body_world_transforms[outer_name][0]),
        nozzle_body_rotation_world=tuple(body_world_transforms[outer_name][1]),
        nozzle_contact_mesh_path=nozzle_contact_mesh_path,
        nozzle_contact_position_local=tuple(nozzle_contact_shape_transform[0]),
        nozzle_contact_rotation_local=tuple(nozzle_contact_shape_transform[1]),
        nozzle_contact_line_starts_local=line_starts,
        nozzle_contact_line_ends_local=line_ends,
        nozzle_contact_vertex_count=vertex_count,
        nozzle_contact_triangle_count=triangle_count,
        nozzle_contact_edge_count=edge_count,
        meshes=tuple(printer_meshes),
    )
