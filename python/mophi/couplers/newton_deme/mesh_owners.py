"""Initialization-time triangle-mesh bridging from Newton bodies to DEME owners."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .contact_coupler import NewtonDEMEOwnerMap


@dataclass(frozen=True)
class NewtonDEMEMeshOwnerSpec:
    """Describe one body-local triangle mesh that DEME owns as a contact proxy."""

    name: str
    newton_body_index: int
    family: int
    mass: float
    principal_moi: tuple[float, float, float]
    vertices: np.ndarray
    triangle_indices: np.ndarray

    def __post_init__(self) -> None:
        if not self.name or Path(self.name).name != self.name:
            raise ValueError(f"DEME mesh owner name must be one non-empty path component, got {self.name!r}.")
        if self.newton_body_index < 0:
            raise ValueError("Newton body indices must be non-negative.")
        if self.mass <= 0.0:
            raise ValueError(f"DEME mesh owner mass must be positive, got {self.mass}.")
        if len(self.principal_moi) != 3 or any(value <= 0.0 for value in self.principal_moi):
            raise ValueError(
                "DEME mesh owner principal MOI must contain three positive values, " f"got {self.principal_moi}."
            )
        vertices = np.asarray(self.vertices, dtype=np.float32)
        triangles = np.asarray(self.triangle_indices, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
            raise ValueError(f"DEME mesh vertices must be a finite (N, 3) array, got shape {vertices.shape}.")
        if triangles.ndim == 1:
            if len(triangles) % 3 != 0:
                raise ValueError("Flat DEME triangle-index arrays must contain a multiple of three entries.")
            triangles = triangles.reshape(-1, 3)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(f"DEME triangle indices must have shape (M, 3), got {triangles.shape}.")
        if len(vertices) == 0 or len(triangles) == 0:
            raise ValueError("DEME mesh owner geometry must contain vertices and triangles.")
        if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
            raise ValueError("DEME mesh owner triangle indices reference vertices outside the mesh.")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangle_indices", triangles)
        object.__setattr__(self, "principal_moi", tuple(float(value) for value in self.principal_moi))


@dataclass(frozen=True)
class NewtonDEMEMeshOwnerBinding:
    """Retain the DEME tracker and Newton association created from one mesh spec."""

    newton_body_index: int
    tracker: object
    mesh_path: Path


def combine_triangle_meshes(
    mesh_parts: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine body-local triangle meshes while adjusting their vertex indices."""
    if not mesh_parts:
        raise ValueError("At least one triangle-mesh part is required.")
    vertices = []
    triangles = []
    vertex_offset = 0
    for part_vertices, part_triangles in mesh_parts:
        part_vertices = np.asarray(part_vertices, dtype=np.float32)
        part_triangles = np.asarray(part_triangles, dtype=np.int32)
        if part_vertices.ndim != 2 or part_vertices.shape[1] != 3:
            raise ValueError(f"Triangle-mesh vertices must have shape (N, 3), got {part_vertices.shape}.")
        if part_triangles.size % 3 != 0:
            raise ValueError("Triangle-mesh indices must contain complete triangles.")
        part_triangles = part_triangles.reshape(-1, 3)
        if np.any(part_triangles < 0) or np.any(part_triangles >= len(part_vertices)):
            raise ValueError("Triangle-mesh indices reference vertices outside their mesh part.")
        vertices.append(part_vertices)
        triangles.append(part_triangles + vertex_offset)
        vertex_offset += len(part_vertices)
    return np.concatenate(vertices), np.concatenate(triangles)


def write_wavefront_mesh(path: Path, vertices: np.ndarray, triangle_indices: np.ndarray) -> None:
    """Write a body-local triangle mesh in the Wavefront OBJ subset DEME consumes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n" for vertex in vertices]
    triangles = np.asarray(triangle_indices, dtype=np.int32).reshape(-1, 3) + 1
    lines.extend(f"f {triangle[0]} {triangle[1]} {triangle[2]}\n" for triangle in triangles)
    path.write_text("".join(lines), encoding="utf-8")


def add_deme_mesh_owners(
    deme_solver,
    material,
    specs: Sequence[NewtonDEMEMeshOwnerSpec],
    initial_newton_body_q: np.ndarray,
    mesh_directory: Path,
) -> list[NewtonDEMEMeshOwnerBinding]:
    """Serialize mesh specs, add DEME owners, and retain their trackers."""
    if not specs:
        raise ValueError("At least one DEME mesh owner specification is required.")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"DEME mesh owner names must be unique within one export, got {names}.")
    body_q = np.asarray(initial_newton_body_q)
    if body_q.ndim != 2 or body_q.shape[1] != 7:
        raise ValueError(f"Newton body transforms must have shape (N, 7), got {body_q.shape}.")
    bindings = []
    for spec in specs:
        if spec.newton_body_index >= len(body_q):
            raise ValueError(
                f"Mesh owner {spec.name!r} references Newton body {spec.newton_body_index}, "
                f"but only {len(body_q)} transforms were supplied."
            )
        mesh_path = Path(mesh_directory) / f"{spec.name}.obj"
        write_wavefront_mesh(mesh_path, spec.vertices, spec.triangle_indices)
        owner = deme_solver.AddWavefrontMeshObject(str(mesh_path), material, False)
        owner.SetInitPos(body_q[spec.newton_body_index, :3].tolist())
        owner.SetInitQuat(body_q[spec.newton_body_index, 3:7].tolist())
        owner.SetFamily(spec.family)
        owner.SetMass(spec.mass)
        owner.SetMOI(list(spec.principal_moi))
        bindings.append(NewtonDEMEMeshOwnerBinding(spec.newton_body_index, deme_solver.Track(owner), mesh_path))
    return bindings


def owner_map_from_mesh_bindings(bindings: Sequence[NewtonDEMEMeshOwnerBinding]) -> NewtonDEMEOwnerMap:
    """Build a runtime owner map after DEME has assigned IDs during initialization."""
    return NewtonDEMEOwnerMap(
        [binding.newton_body_index for binding in bindings],
        [binding.tracker.GetOwnerID() for binding in bindings],
    )
