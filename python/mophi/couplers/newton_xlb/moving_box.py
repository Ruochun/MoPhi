"""GPU-resident moving AABB exchange between Newton scenes and XLB masks."""

from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.kernel
def _clear_bc(mask: wp.array4d(dtype=wp.uint8), x0: int, y0: int, z0: int):
    i, j, k = wp.tid()
    mask[0, x0 + i, y0 + j, z0 + k] = wp.uint8(0)


@wp.kernel
def _stamp_bc(mask: wp.array4d(dtype=wp.uint8), x0: int, y0: int, z0: int, boundary_id: int):
    i, j, k = wp.tid()
    mask[0, x0 + i, y0 + j, z0 + k] = wp.uint8(boundary_id)


@wp.kernel
def _clear_missing(mask: wp.array4d(dtype=wp.bool), direction: int, x0: int, y0: int, z0: int):
    i, j, k = wp.tid()
    mask[direction, x0 + i, y0 + j, z0 + k] = False


@wp.kernel
def _stamp_missing(
    mask: wp.array4d(dtype=wp.bool),
    lattice_velocities: wp.array2d(dtype=wp.int32),
    direction: int,
    x0: int,
    y0: int,
    z0: int,
    x1: int,
    y1: int,
    z1: int,
):
    i, j, k = wp.tid()
    x, y, z = x0 + i, y0 + j, z0 + k
    sx = x - lattice_velocities[direction, 0]
    sy = y - lattice_velocities[direction, 1]
    sz = z - lattice_velocities[direction, 2]
    mask[direction, x, y, z] = sx < x0 or sx > x1 or sy < y0 or sy > y1 or sz < z0 or sz > z1


def world_to_grid_index(world_position, domain_min, domain_max, grid_shape):
    """Map a world point to the nearest interior XLB cell."""
    domain_min = np.asarray(domain_min, dtype=np.float64)
    domain_max = np.asarray(domain_max, dtype=np.float64)
    shape = np.asarray(grid_shape, dtype=np.int64)
    if np.any(domain_max <= domain_min):
        raise ValueError("Every XLB domain maximum must exceed its minimum.")
    if shape.shape != (3,) or np.any(shape < 3):
        raise ValueError("XLB grid_shape must contain three dimensions of at least three cells.")
    fraction = (np.asarray(world_position, dtype=np.float64) - domain_min) / (domain_max - domain_min)
    index = (np.clip(fraction, 0.0, 1.0) * shape).astype(int)
    return np.clip(index, np.ones(3, dtype=int), shape - 2)


def prescribed_box_grid(world_min, world_max, domain_min, domain_max, grid_shape):
    """Return inclusive grid bounds for a world AABB, or ``(None, None)`` when outside."""
    world_min = np.asarray(world_min, dtype=np.float64)
    world_max = np.asarray(world_max, dtype=np.float64)
    domain_min = np.asarray(domain_min, dtype=np.float64)
    domain_max = np.asarray(domain_max, dtype=np.float64)
    if np.any(world_max < world_min):
        raise ValueError("AABB world_max must not be below world_min.")
    if np.any(world_max <= domain_min) or np.any(world_min >= domain_max):
        return None, None
    return (
        world_to_grid_index(np.maximum(world_min, domain_min), domain_min, domain_max, grid_shape),
        world_to_grid_index(np.minimum(world_max, domain_max), domain_min, domain_max, grid_shape),
    )


def update_box_boundary_gpu(
    bc_mask, missing_mask, old_min, old_max, new_min, new_max, boundary_id, lattice_velocities, q
):
    """Clear an old box and stamp a new box into XLB boundary masks on their Warp device."""
    device = bc_mask.device
    if old_min is not None:
        lo, hi = np.asarray(old_min, dtype=int), np.asarray(old_max, dtype=int)
        dims = tuple((hi - lo + 1).tolist())
        wp.launch(_clear_bc, dim=dims, inputs=[bc_mask, *lo.tolist()], device=device)
        for direction in range(q):
            wp.launch(_clear_missing, dim=dims, inputs=[missing_mask, direction, *lo.tolist()], device=device)
    if new_min is not None:
        lo, hi = np.asarray(new_min, dtype=int), np.asarray(new_max, dtype=int)
        dims = tuple((hi - lo + 1).tolist())
        wp.launch(_stamp_bc, dim=dims, inputs=[bc_mask, *lo.tolist(), int(boundary_id)], device=device)
        for direction in range(q):
            wp.launch(
                _stamp_missing,
                dim=dims,
                inputs=[missing_mask, lattice_velocities, direction, *lo.tolist(), *hi.tolist()],
                device=device,
            )
    return new_min, new_max


@dataclass
class MovingBoxBoundary:
    """Keep the previous bounds needed for an incremental XLB mask update."""

    boundary_id: int
    lattice_velocities: object
    q: int
    grid_min: object = None
    grid_max: object = None

    def update(self, bc_mask, missing_mask, grid_min, grid_max):
        self.grid_min, self.grid_max = update_box_boundary_gpu(
            bc_mask,
            missing_mask,
            self.grid_min,
            self.grid_max,
            grid_min,
            grid_max,
            self.boundary_id,
            self.lattice_velocities,
            self.q,
        )
        return self.grid_min, self.grid_max


@wp.kernel
def _stamp_newton_body_bc(
    bc_mask: wp.array4d(dtype=wp.uint8),
    base_bc_mask: wp.array4d(dtype=wp.uint8),
    body_q: wp.array(dtype=wp.transform),
    body_index: int,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    lower_extents: wp.vec3,
    upper_extents: wp.vec3,
    boundary_id: int,
):
    x, y, z = wp.tid()
    center = wp.transform_get_translation(body_q[body_index])
    dims = wp.vec3(float(bc_mask.shape[1]), float(bc_mask.shape[2]), float(bc_mask.shape[3]))
    extent = domain_max - domain_min
    fraction = wp.vec3(
        (center[0] - domain_min[0]) / extent[0],
        (center[1] - domain_min[1]) / extent[1],
        (center[2] - domain_min[2]) / extent[2],
    )
    center_cell = wp.vec3i(
        int(wp.floor(fraction[0] * dims[0])),
        int(wp.floor(fraction[1] * dims[1])),
        int(wp.floor(fraction[2] * dims[2])),
    )
    lo = wp.vec3i(
        int(wp.floor((center[0] - lower_extents[0] - domain_min[0]) / extent[0] * dims[0])),
        int(wp.floor((center[1] - lower_extents[1] - domain_min[1]) / extent[1] * dims[1])),
        int(wp.floor((center[2] - lower_extents[2] - domain_min[2]) / extent[2] * dims[2])),
    )
    hi = wp.vec3i(
        int(wp.floor((center[0] + upper_extents[0] - domain_min[0]) / extent[0] * dims[0])),
        int(wp.floor((center[1] + upper_extents[1] - domain_min[1]) / extent[1] * dims[1])),
        int(wp.floor((center[2] + upper_extents[2] - domain_min[2]) / extent[2] * dims[2])),
    )
    inside = (
        x >= wp.max(1, lo[0])
        and x <= wp.min(int(dims[0]) - 2, hi[0])
        and y >= wp.max(1, lo[1])
        and y <= wp.min(int(dims[1]) - 2, hi[1])
        and z >= wp.max(1, lo[2])
        and z <= wp.min(int(dims[2]) - 2, hi[2])
    )
    bc_mask[0, x, y, z] = wp.uint8(boundary_id) if inside else base_bc_mask[0, x, y, z]


@wp.kernel
def _stamp_newton_body_missing(
    missing_mask: wp.array4d(dtype=wp.bool),
    base_missing_mask: wp.array4d(dtype=wp.bool),
    lattice_velocities: wp.array2d(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_index: int,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    lower_extents: wp.vec3,
    upper_extents: wp.vec3,
):
    direction, x, y, z = wp.tid()
    center = wp.transform_get_translation(body_q[body_index])
    dims = wp.vec3(float(missing_mask.shape[1]), float(missing_mask.shape[2]), float(missing_mask.shape[3]))
    extent = domain_max - domain_min
    fraction = wp.vec3(
        (center[0] - domain_min[0]) / extent[0],
        (center[1] - domain_min[1]) / extent[1],
        (center[2] - domain_min[2]) / extent[2],
    )
    center_cell = wp.vec3i(
        int(wp.floor(fraction[0] * dims[0])),
        int(wp.floor(fraction[1] * dims[1])),
        int(wp.floor(fraction[2] * dims[2])),
    )
    lo = wp.vec3i(
        wp.max(1, int(wp.floor((center[0] - lower_extents[0] - domain_min[0]) / extent[0] * dims[0]))),
        wp.max(1, int(wp.floor((center[1] - lower_extents[1] - domain_min[1]) / extent[1] * dims[1]))),
        wp.max(1, int(wp.floor((center[2] - lower_extents[2] - domain_min[2]) / extent[2] * dims[2]))),
    )
    hi = wp.vec3i(
        wp.min(int(dims[0]) - 2, int(wp.floor((center[0] + upper_extents[0] - domain_min[0]) / extent[0] * dims[0]))),
        wp.min(int(dims[1]) - 2, int(wp.floor((center[1] + upper_extents[1] - domain_min[1]) / extent[1] * dims[1]))),
        wp.min(int(dims[2]) - 2, int(wp.floor((center[2] + upper_extents[2] - domain_min[2]) / extent[2] * dims[2]))),
    )
    inside = x >= lo[0] and x <= hi[0] and y >= lo[1] and y <= hi[1] and z >= lo[2] and z <= hi[2]
    sx = x - lattice_velocities[direction, 0]
    sy = y - lattice_velocities[direction, 1]
    sz = z - lattice_velocities[direction, 2]
    source_outside = sx < lo[0] or sx > hi[0] or sy < lo[1] or sy > hi[1] or sz < lo[2] or sz > hi[2]
    missing_mask[direction, x, y, z] = inside and source_outside or base_missing_mask[direction, x, y, z]


class NewtonBodyBoxBoundary:
    """Update an XLB AABB directly from a Newton GPU body transform array."""

    def __init__(self):
        self.initialized = False

    def initialize(
        self,
        bc_mask,
        missing_mask,
        lattice_velocities,
        body_index,
        domain_min,
        domain_max,
        lower_extents,
        boundary_id,
        upper_extents=None,
        initial_grid_min=None,
        initial_grid_max=None,
    ):
        """Capture immutable base masks and boundary geometry."""
        if self.initialized:
            raise RuntimeError("NewtonBodyBoxBoundary is already initialized.")
        if not bc_mask.device.is_cuda or missing_mask.device != bc_mask.device:
            raise ValueError("Newton--XLB boundary arrays must share one CUDA Warp device.")
        if int(body_index) < 0:
            raise ValueError("Newton body_index must be non-negative.")
        self.bc_mask = bc_mask
        self.missing_mask = missing_mask
        self.base_bc_mask = wp.clone(bc_mask)
        self.base_missing_mask = wp.clone(missing_mask)
        if initial_grid_min is not None:
            update_box_boundary_gpu(
                self.base_bc_mask,
                self.base_missing_mask,
                initial_grid_min,
                initial_grid_max,
                None,
                None,
                boundary_id,
                lattice_velocities,
                int(missing_mask.shape[0]),
            )
        self.lattice_velocities = lattice_velocities
        self.body_index = int(body_index)
        self.domain_min = wp.vec3(*map(float, domain_min))
        self.domain_max = wp.vec3(*map(float, domain_max))
        upper_extents = lower_extents if upper_extents is None else upper_extents
        self.lower_extents = wp.vec3(*map(float, lower_extents))
        self.upper_extents = wp.vec3(*map(float, upper_extents))
        self.boundary_id = int(boundary_id)
        self.device = bc_mask.device
        self.initialized = True

    def update_from_newton(self, body_q):
        """Read ``body_q`` and rewrite both XLB masks without host staging."""
        if not self.initialized:
            raise RuntimeError("NewtonBodyBoxBoundary must be initialized before use.")
        if body_q.device != self.device or body_q.dtype != wp.transform:
            raise ValueError("Newton body_q must be a wp.transform array on the XLB CUDA device.")
        if self.body_index >= len(body_q):
            raise ValueError(f"Newton body index {self.body_index} exceeds body_q length {len(body_q)}.")
        wp.launch(
            _stamp_newton_body_bc,
            dim=tuple(self.bc_mask.shape[1:]),
            inputs=[
                self.bc_mask,
                self.base_bc_mask,
                body_q,
                self.body_index,
                self.domain_min,
                self.domain_max,
                self.lower_extents,
                self.upper_extents,
                self.boundary_id,
            ],
            device=self.device,
        )
        wp.launch(
            _stamp_newton_body_missing,
            dim=tuple(self.missing_mask.shape),
            inputs=[
                self.missing_mask,
                self.base_missing_mask,
                self.lattice_velocities,
                body_q,
                self.body_index,
                self.domain_min,
                self.domain_max,
                self.lower_extents,
                self.upper_extents,
            ],
            device=self.device,
        )

    def finalize(self):
        """Release retained solver-array references."""
        self.__dict__.clear()
        self.initialized = False


@wp.kernel
def _stamp_newton_bodies_bc(
    bc_mask: wp.array4d(dtype=wp.uint8),
    base_bc_mask: wp.array4d(dtype=wp.uint8),
    body_q: wp.array(dtype=wp.transform),
    body_indices: wp.array(dtype=wp.int32),
    half_extents: wp.array(dtype=wp.vec3),
    boundary_ids: wp.array(dtype=wp.int32),
    box_count: int,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
):
    x, y, z = wp.tid()
    dims = wp.vec3(float(bc_mask.shape[1]), float(bc_mask.shape[2]), float(bc_mask.shape[3]))
    domain_extent = domain_max - domain_min
    value = base_bc_mask[0, x, y, z]
    for box_index in range(box_count):
        center = wp.transform_get_translation(body_q[body_indices[box_index]])
        extent = half_extents[box_index]
        lo = wp.vec3i(
            wp.max(1, int(wp.floor((center[0] - extent[0] - domain_min[0]) / domain_extent[0] * dims[0]))),
            wp.max(1, int(wp.floor((center[1] - extent[1] - domain_min[1]) / domain_extent[1] * dims[1]))),
            wp.max(1, int(wp.floor((center[2] - extent[2] - domain_min[2]) / domain_extent[2] * dims[2]))),
        )
        hi = wp.vec3i(
            wp.min(
                int(dims[0]) - 2,
                int(wp.floor((center[0] + extent[0] - domain_min[0]) / domain_extent[0] * dims[0])),
            ),
            wp.min(
                int(dims[1]) - 2,
                int(wp.floor((center[1] + extent[1] - domain_min[1]) / domain_extent[1] * dims[1])),
            ),
            wp.min(
                int(dims[2]) - 2,
                int(wp.floor((center[2] + extent[2] - domain_min[2]) / domain_extent[2] * dims[2])),
            ),
        )
        if x >= lo[0] and x <= hi[0] and y >= lo[1] and y <= hi[1] and z >= lo[2] and z <= hi[2]:
            # Later boxes intentionally win, matching sequential mask stamping.
            value = wp.uint8(boundary_ids[box_index])
    bc_mask[0, x, y, z] = value


@wp.kernel
def _stamp_newton_bodies_missing(
    missing_mask: wp.array4d(dtype=wp.bool),
    base_missing_mask: wp.array4d(dtype=wp.bool),
    lattice_velocities: wp.array2d(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_indices: wp.array(dtype=wp.int32),
    half_extents: wp.array(dtype=wp.vec3),
    box_count: int,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
):
    direction, x, y, z = wp.tid()
    dims = wp.vec3(float(missing_mask.shape[1]), float(missing_mask.shape[2]), float(missing_mask.shape[3]))
    domain_extent = domain_max - domain_min
    value = base_missing_mask[direction, x, y, z]
    for box_index in range(box_count):
        center = wp.transform_get_translation(body_q[body_indices[box_index]])
        extent = half_extents[box_index]
        lo = wp.vec3i(
            wp.max(1, int(wp.floor((center[0] - extent[0] - domain_min[0]) / domain_extent[0] * dims[0]))),
            wp.max(1, int(wp.floor((center[1] - extent[1] - domain_min[1]) / domain_extent[1] * dims[1]))),
            wp.max(1, int(wp.floor((center[2] - extent[2] - domain_min[2]) / domain_extent[2] * dims[2]))),
        )
        hi = wp.vec3i(
            wp.min(
                int(dims[0]) - 2,
                int(wp.floor((center[0] + extent[0] - domain_min[0]) / domain_extent[0] * dims[0])),
            ),
            wp.min(
                int(dims[1]) - 2,
                int(wp.floor((center[1] + extent[1] - domain_min[1]) / domain_extent[1] * dims[1])),
            ),
            wp.min(
                int(dims[2]) - 2,
                int(wp.floor((center[2] + extent[2] - domain_min[2]) / domain_extent[2] * dims[2])),
            ),
        )
        if x >= lo[0] and x <= hi[0] and y >= lo[1] and y <= hi[1] and z >= lo[2] and z <= hi[2]:
            sx = x - lattice_velocities[direction, 0]
            sy = y - lattice_velocities[direction, 1]
            sz = z - lattice_velocities[direction, 2]
            value = sx < lo[0] or sx > hi[0] or sy < lo[1] or sy > hi[1] or sz < lo[2] or sz > hi[2]
    missing_mask[direction, x, y, z] = value


class NewtonBodiesBoxBoundary:
    """Rebuild multiple overlapping XLB AABBs from Newton device transforms."""

    def __init__(self):
        self.initialized = False

    def initialize(
        self,
        bc_mask,
        missing_mask,
        lattice_velocities,
        body_indices,
        half_extents,
        boundary_ids,
        domain_min,
        domain_max,
        initial_grid_bounds=None,
    ):
        """Bind one boundary box per mapped Newton body."""
        if self.initialized:
            raise RuntimeError("NewtonBodiesBoxBoundary is already initialized.")
        body_indices = tuple(int(index) for index in body_indices)
        half_extents = tuple(tuple(map(float, extent)) for extent in half_extents)
        boundary_ids = tuple(int(boundary_id) for boundary_id in boundary_ids)
        if not body_indices or len(body_indices) != len(half_extents) or len(body_indices) != len(boundary_ids):
            raise ValueError("Multi-body XLB boundaries require equally sized, non-empty body/extent/ID mappings.")
        if not bc_mask.device.is_cuda or missing_mask.device != bc_mask.device:
            raise ValueError("Newton--XLB boundary arrays must share one CUDA Warp device.")
        self.bc_mask = bc_mask
        self.missing_mask = missing_mask
        self.base_bc_mask = wp.clone(bc_mask)
        self.base_missing_mask = wp.clone(missing_mask)
        if initial_grid_bounds is not None:
            if len(initial_grid_bounds) != len(body_indices):
                raise ValueError("initial_grid_bounds must contain one entry per boundary box.")
            for (grid_min, grid_max), boundary_id in zip(initial_grid_bounds, boundary_ids):
                update_box_boundary_gpu(
                    self.base_bc_mask,
                    self.base_missing_mask,
                    grid_min,
                    grid_max,
                    None,
                    None,
                    boundary_id,
                    lattice_velocities,
                    int(missing_mask.shape[0]),
                )
        self.body_indices = wp.array(body_indices, dtype=wp.int32, device=bc_mask.device)
        self.box_count = len(body_indices)
        self.maximum_body_index = max(body_indices)
        self.half_extents = wp.array(half_extents, dtype=wp.vec3, device=bc_mask.device)
        self.boundary_ids = wp.array(boundary_ids, dtype=wp.int32, device=bc_mask.device)
        self.lattice_velocities = lattice_velocities
        self.domain_min = wp.vec3(*map(float, domain_min))
        self.domain_max = wp.vec3(*map(float, domain_max))
        self.device = bc_mask.device
        self.initialized = True

    def update_from_newton(self, body_q):
        """Rebuild all mapped boxes directly from Newton's device state."""
        if not self.initialized:
            raise RuntimeError("NewtonBodiesBoxBoundary must be initialized before use.")
        if body_q.device != self.device or body_q.dtype != wp.transform:
            raise ValueError("Newton body_q must be a wp.transform array on the XLB CUDA device.")
        if self.maximum_body_index >= len(body_q):
            raise ValueError("A mapped Newton body index exceeds body_q length.")
        wp.launch(
            _stamp_newton_bodies_bc,
            dim=tuple(self.bc_mask.shape[1:]),
            inputs=[
                self.bc_mask,
                self.base_bc_mask,
                body_q,
                self.body_indices,
                self.half_extents,
                self.boundary_ids,
                self.box_count,
                self.domain_min,
                self.domain_max,
            ],
            device=self.device,
        )
        wp.launch(
            _stamp_newton_bodies_missing,
            dim=tuple(self.missing_mask.shape),
            inputs=[
                self.missing_mask,
                self.base_missing_mask,
                self.lattice_velocities,
                body_q,
                self.body_indices,
                self.half_extents,
                self.box_count,
                self.domain_min,
                self.domain_max,
            ],
            device=self.device,
        )

    def finalize(self):
        """Release retained solver-array references."""
        self.__dict__.clear()
        self.initialized = False
