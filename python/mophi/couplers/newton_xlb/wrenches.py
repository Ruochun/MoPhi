"""GPU-native XLB wrench delivery to Newton body-force arrays."""

from collections.abc import Sequence

import warp as wp


@wp.kernel
def _reduce_halfway_bounce_back_wrench(
    populations: wp.array4d(dtype=wp.float32),
    bc_mask: wp.array4d(dtype=wp.uint8),
    missing_mask: wp.array4d(dtype=wp.bool),
    lattice_velocities: wp.array2d(dtype=wp.int32),
    opposite_indices: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_index: int,
    boundary_id: int,
    domain_min: wp.vec3,
    cell_size: float,
    force_scale: float,
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
):
    direction, i, j, k = wp.tid()
    if bc_mask[0, i, j, k] != wp.uint8(boundary_id) or not missing_mask[direction, i, j, k]:
        return
    cx = float(lattice_velocities[direction, 0])
    cy = float(lattice_velocities[direction, 1])
    cz = float(lattice_velocities[direction, 2])
    lattice_direction = wp.vec3(cx, cy, cz)
    reflected_population = populations[opposite_indices[direction], i, j, k]
    link_force = lattice_direction * (2.0 * reflected_population * force_scale)
    wall_point = (
        domain_min
        + wp.vec3(
            float(i) + 0.5 - 0.5 * cx,
            float(j) + 0.5 - 0.5 * cy,
            float(k) + 0.5 - 0.5 * cz,
        )
        * cell_size
    )
    body_center = wp.transform_get_translation(body_q[body_index])
    link_torque = wp.cross(wall_point - body_center, link_force)
    wp.atomic_add(forces, 0, link_force)
    wp.atomic_add(torques, 0, link_torque)


@wp.kernel
def _scatter_body_wrenches(
    body_indices: wp.array(dtype=wp.int32),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    destination: wp.array(dtype=wp.spatial_vector),
):
    index = wp.tid()
    destination[body_indices[index]] += wp.spatial_vector(forces[index], torques[index])


@wp.kernel
def _apply_nonphysical_wrench_limits(
    raw_forces: wp.array(dtype=wp.vec3),
    raw_torques: wp.array(dtype=wp.vec3),
    feedback_gain: float,
    maximum_force: float,
    maximum_torque: float,
    limited_forces: wp.array(dtype=wp.vec3),
    limited_torques: wp.array(dtype=wp.vec3),
):
    """Numerically moderate an unvalidated wrench; this is not a physical model.

    TODO: Remove this when true moving-boundary treatment and force feedback exist.
    """
    index = wp.tid()
    force = raw_forces[index] * feedback_gain
    torque = raw_torques[index] * feedback_gain
    force_magnitude = wp.length(force)
    torque_magnitude = wp.length(torque)
    if force_magnitude > maximum_force:
        force *= maximum_force / force_magnitude
    if torque_magnitude > maximum_torque:
        torque *= maximum_torque / torque_magnitude
    limited_forces[index] = force
    limited_torques[index] = torque


class NewtonXLBWrenchExchange:
    """Scatter caller-computed XLB wrenches into Newton's GPU body-force array."""

    def __init__(self):
        self.initialized = False

    def initialize(self, newton_model, body_indices: Sequence[int], device):
        """Validate a one-wrench-per-body mapping and allocate its device indices."""
        if self.initialized:
            raise RuntimeError("NewtonXLBWrenchExchange is already initialized.")
        indices = tuple(int(index) for index in body_indices)
        if not indices:
            raise ValueError("Newton--XLB wrench exchange requires at least one body index.")
        if len(set(indices)) != len(indices):
            raise ValueError("Newton--XLB wrench body indices must be unique.")
        if min(indices) < 0 or max(indices) >= int(newton_model.body_count):
            raise ValueError("Newton--XLB wrench mapping contains an out-of-range body index.")
        if not device.is_cuda:
            raise ValueError("Newton--XLB wrench exchange requires a CUDA Warp device.")
        self.newton_model = newton_model
        self.body_indices = wp.array(indices, dtype=wp.int32, device=device)
        self.count = len(indices)
        self.device = device
        # These buffers are only for the explicitly named nonphysical PoC path.
        # Keeping them on-device preserves the GPU-only exchange mechanism.
        # TODO: Remove them when true moving-boundary treatment and force feedback exist.
        self.numerically_limited_forces = wp.empty(self.count, dtype=wp.vec3, device=device)
        self.numerically_limited_torques = wp.empty(self.count, dtype=wp.vec3, device=device)
        self.initialized = True

    def write_xlb_wrenches_to_newton(self, forces, torques, destination, clear_destination=True):
        """Write global force/torque pairs into a Newton ``body_f``-compatible array."""
        if not self.initialized:
            raise RuntimeError("NewtonXLBWrenchExchange must be initialized before use.")
        for array, label in ((forces, "force"), (torques, "torque")):
            if len(array) < self.count or array.dtype != wp.vec3 or array.device != self.device:
                raise ValueError(f"XLB {label} must contain {self.count} wp.vec3 entries on {self.device}.")
        if (
            len(destination) != int(self.newton_model.body_count)
            or destination.dtype != wp.spatial_vector
            or destination.device != self.device
        ):
            raise ValueError("Newton destination must be a body-count wp.spatial_vector array on the shared device.")
        if clear_destination:
            destination.zero_()
        wp.launch(
            _scatter_body_wrenches,
            dim=self.count,
            inputs=[self.body_indices, forces, torques, destination],
            device=self.device,
        )
        return destination

    def write_numerically_limited_xlb_wrenches_to_newton(
        self,
        forces,
        torques,
        destination,
        feedback_gain,
        maximum_force,
        maximum_torque,
        clear_destination=True,
    ):
        """Apply an explicitly nonphysical gain/clamp before writing a PoC wrench.

        This numerical stabilization is provided only for demonstrations whose
        fluid boundary cannot produce a physically valid moving-wall reaction.
        It must not be presented as a hydrodynamic force model.

        TODO: Remove this method when true moving-boundary treatment and force
        feedback are available.
        """
        if feedback_gain <= 0.0:
            raise ValueError("The nonphysical wrench feedback gain must be positive.")
        if maximum_force <= 0.0 or maximum_torque <= 0.0:
            raise ValueError("The nonphysical wrench force and torque limits must be positive.")
        if not self.initialized:
            raise RuntimeError("NewtonXLBWrenchExchange must be initialized before use.")
        for array, label in ((forces, "force"), (torques, "torque")):
            if len(array) < self.count or array.dtype != wp.vec3 or array.device != self.device:
                raise ValueError(f"XLB {label} must contain {self.count} wp.vec3 entries on {self.device}.")
        wp.launch(
            _apply_nonphysical_wrench_limits,
            dim=self.count,
            inputs=[
                forces,
                torques,
                float(feedback_gain),
                float(maximum_force),
                float(maximum_torque),
                self.numerically_limited_forces,
                self.numerically_limited_torques,
            ],
            device=self.device,
        )
        return self.write_xlb_wrenches_to_newton(
            self.numerically_limited_forces,
            self.numerically_limited_torques,
            destination,
            clear_destination=clear_destination,
        )

    def finalize(self):
        """Release model and device-array references."""
        self.__dict__.clear()
        self.initialized = False


class NewtonXLBHalfwayBounceBackWrench:
    """Estimate a body wrench from stationary halfway-bounce-back links.

    This is an intentionally ad-hoc momentum-exchange diagnostic. It does not
    make XLB's stationary-wall boundary scientifically valid for moving bodies.

    TODO: Remove this reducer when true moving-boundary treatment and force
    feedback are available.
    """

    def __init__(self):
        self.initialized = False

    def initialize(
        self,
        body_index,
        boundary_id,
        domain_min,
        cell_size,
        physical_density,
        xlb_dt,
        lattice_velocities,
        opposite_indices,
        device,
    ):
        """Allocate a one-body device wrench and store lattice-to-SI scaling."""
        if self.initialized:
            raise RuntimeError("NewtonXLBHalfwayBounceBackWrench is already initialized.")
        if not device.is_cuda:
            raise ValueError("Halfway-bounce-back wrench reduction requires a CUDA Warp device.")
        if cell_size <= 0.0 or physical_density <= 0.0 or xlb_dt <= 0.0:
            raise ValueError("Cell size, physical density, and XLB dt must be positive.")
        self.body_index = int(body_index)
        self.boundary_id = int(boundary_id)
        self.domain_min = wp.vec3(*domain_min)
        self.cell_size = float(cell_size)
        self.force_scale = float(physical_density) * float(cell_size) ** 4 / float(xlb_dt) ** 2
        self.lattice_velocities = lattice_velocities
        self.opposite_indices = wp.array(opposite_indices, dtype=wp.int32, device=device)
        self.device = device
        self.forces = wp.zeros(1, dtype=wp.vec3, device=device)
        self.torques = wp.zeros(1, dtype=wp.vec3, device=device)
        self.initialized = True

    def reduce(self, populations, bc_mask, missing_mask, body_q):
        """Return the current ad-hoc SI wrench in reusable device arrays.

        TODO: Remove this path when true moving-boundary treatment and force
        feedback are available.
        """
        if not self.initialized:
            raise RuntimeError("NewtonXLBHalfwayBounceBackWrench must be initialized before use.")
        self.forces.zero_()
        self.torques.zero_()
        wp.launch(
            _reduce_halfway_bounce_back_wrench,
            dim=(populations.shape[0], populations.shape[1], populations.shape[2], populations.shape[3]),
            inputs=[
                populations,
                bc_mask,
                missing_mask,
                self.lattice_velocities,
                self.opposite_indices,
                body_q,
                self.body_index,
                self.boundary_id,
                self.domain_min,
                self.cell_size,
                self.force_scale,
                self.forces,
                self.torques,
            ],
            device=self.device,
        )
        return self.forces, self.torques

    def finalize(self):
        """Release device-array references."""
        self.__dict__.clear()
        self.initialized = False
