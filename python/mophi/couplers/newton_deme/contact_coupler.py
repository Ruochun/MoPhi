"""GPU-native Newton--DEME contact-wrench exchange."""

from dataclasses import dataclass
from typing import Sequence

import warp as wp


@dataclass(frozen=True)
class NewtonDEMEOwnerMap:
    """Map Newton rigid bodies to one consecutive span of DEME owners."""

    newton_body_indices: tuple[int, ...]
    deme_owner_ids: tuple[int, ...]

    def __init__(self, newton_body_indices: Sequence[int], deme_owner_ids: Sequence[int]) -> None:
        body_indices = tuple(int(index) for index in newton_body_indices)
        owner_ids = tuple(int(owner_id) for owner_id in deme_owner_ids)
        if not body_indices:
            raise ValueError("A Newton--DEME owner map must contain at least one body.")
        if len(body_indices) != len(owner_ids):
            raise ValueError(
                "Newton body and DEME owner mappings must have equal lengths; "
                f"got {len(body_indices)} and {len(owner_ids)}."
            )
        if any(index < 0 for index in body_indices):
            raise ValueError("Newton body indices must be non-negative.")
        if any(owner_id < 0 for owner_id in owner_ids):
            raise ValueError("DEME owner IDs must be non-negative.")
        expected_owner_ids = tuple(range(owner_ids[0], owner_ids[0] + len(owner_ids)))
        if owner_ids != expected_owner_ids:
            raise ValueError(
                "DEME owners must form one increasing consecutive span for bulk GPU exchange; " f"got {owner_ids}."
            )
        object.__setattr__(self, "newton_body_indices", body_indices)
        object.__setattr__(self, "deme_owner_ids", owner_ids)

    @property
    def first_deme_owner_id(self) -> int:
        return self.deme_owner_ids[0]

    @property
    def owner_count(self) -> int:
        return len(self.deme_owner_ids)


@wp.kernel
def _gather_newton_body_state(
    body_indices: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
):
    owner_offset = wp.tid()
    body_index = body_indices[owner_offset]
    positions[owner_offset] = wp.transform_get_translation(body_q[body_index])
    orientations[owner_offset] = wp.transform_get_rotation(body_q[body_index])
    velocities[owner_offset] = wp.spatial_top(body_qd[body_index])
    angular_velocities[owner_offset] = wp.spatial_bottom(body_qd[body_index])


@wp.kernel
def _accumulate_deme_owner_wrenches(
    owner_offsets_by_body: wp.array(dtype=wp.int32),
    body_owner_starts: wp.array(dtype=wp.int32),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    clear_destination: int,
    body_forces: wp.array(dtype=wp.spatial_vector),
):
    body_index = wp.tid()
    total = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    for grouped_offset in range(body_owner_starts[body_index], body_owner_starts[body_index + 1]):
        owner_offset = owner_offsets_by_body[grouped_offset]
        total += wp.spatial_vector(forces[owner_offset], torques[owner_offset])
    if clear_destination != 0:
        body_forces[body_index] = total
    else:
        body_forces[body_index] += total


@wp.kernel
def _gather_newton_proxy_pose(
    body_indices: wp.array(dtype=wp.int32),
    local_offsets: wp.array(dtype=wp.vec3),
    local_orientations: wp.array(dtype=wp.quat),
    body_q: wp.array(dtype=wp.transform),
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
):
    index = wp.tid()
    transform = body_q[body_indices[index]]
    positions[index] = wp.transform_point(transform, local_offsets[index])
    orientations[index] = wp.transform_get_rotation(transform) * local_orientations[index]


@wp.kernel
def _contact_accelerations_to_newton_forces(
    body_indices: wp.array(dtype=wp.int32),
    orientations: wp.array(dtype=wp.quat),
    masses: wp.array(dtype=wp.float32),
    moments: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    local_angular_accelerations: wp.array(dtype=wp.vec3),
    scale: float,
    body_forces: wp.array(dtype=wp.spatial_vector),
):
    index = wp.tid()
    local_torque = wp.cw_mul(moments[index], local_angular_accelerations[index]) * scale
    force = accelerations[index] * masses[index] * scale
    torque = wp.quat_rotate(orientations[index], local_torque)
    body_forces[body_indices[index]] = wp.spatial_vector(force, torque)


class NewtonDEMEOwnerPoseExchange:
    """Write mapped Newton body poses to a consecutive DEME owner span on-device."""

    _REQUIRED_METHODS = ("SetOwnerPositionFromDevice", "SetOwnerOriQFromDevice")

    def __init__(self) -> None:
        self.initialized = False

    def __del__(self) -> None:
        if self.initialized:
            self.finalize()

    def initialize(
        self,
        newton_model,
        deme_solver,
        owner_map: NewtonDEMEOwnerMap,
        device,
        local_offsets: Sequence[Sequence[float]] | None = None,
        local_orientations: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Bind initialized solvers and allocate reusable pose buffers."""
        if self.initialized:
            raise RuntimeError("NewtonDEMEOwnerPoseExchange is already initialized.")
        if not device.is_cuda:
            raise ValueError("Newton--DEME GPU pose exchange requires a CUDA Warp device.")
        if max(owner_map.newton_body_indices) >= int(newton_model.body_count):
            raise ValueError("Newton--DEME pose mapping contains an out-of-range body index.")
        missing = [name for name in self._REQUIRED_METHODS if not hasattr(deme_solver, name)]
        if missing:
            raise RuntimeError(f"DEMSolver is missing GPU pose exchange methods {missing}.")
        if device.ordinal not in tuple(int(value) for value in deme_solver.GetGPUDeviceIDs()):
            raise ValueError("DEME and Newton must share the selected CUDA device.")

        count = owner_map.owner_count
        local_offsets = [(0.0, 0.0, 0.0)] * count if local_offsets is None else local_offsets
        local_orientations = [(0.0, 0.0, 0.0, 1.0)] * count if local_orientations is None else local_orientations
        if len(local_offsets) != count or len(local_orientations) != count:
            raise ValueError("Pose exchange requires one local transform per DEME owner.")

        self.deme_solver = deme_solver
        self.owner_map = owner_map
        self.device = device
        self._body_indices = wp.array(owner_map.newton_body_indices, dtype=wp.int32, device=device)
        self._local_offsets = wp.array(local_offsets, dtype=wp.vec3, device=device)
        self._local_orientations = wp.array(local_orientations, dtype=wp.quat, device=device)
        self._positions = wp.empty(count, dtype=wp.vec3, device=device)
        self._orientations = wp.empty(count, dtype=wp.quat, device=device)
        self.initialized = True

    def set_deme_owner_pose_from_newton(self, newton_state) -> None:
        """Apply local transforms and write DEME owner poses from Newton device state."""
        if not self.initialized:
            raise RuntimeError("NewtonDEMEOwnerPoseExchange must be initialized before use.")
        count = self.owner_map.owner_count
        wp.launch(
            _gather_newton_proxy_pose,
            dim=count,
            inputs=[
                self._body_indices,
                self._local_offsets,
                self._local_orientations,
                newton_state.body_q,
                self._positions,
                self._orientations,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        first_owner = self.owner_map.first_deme_owner_id
        self.deme_solver.SetOwnerPositionFromDevice(first_owner, self._positions.ptr, self.device.ordinal, count)
        self.deme_solver.SetOwnerOriQFromDevice(first_owner, self._orientations.ptr, self.device.ordinal, count)

    def finalize(self) -> None:
        """Release solver and device-array references."""
        self.deme_solver = None
        self.owner_map = None
        self.device = None
        self._body_indices = None
        self._local_offsets = None
        self._local_orientations = None
        self._positions = None
        self._orientations = None
        self.initialized = False


class NewtonDEMEContactCoupler:
    """Exchange Newton body motion and DEME contact wrenches on one GPU.

    DEME mesh owners must already exist and the DEM solver must already be
    initialized. The caller remains responsible for geometry, materials,
    families, and the relative stepping schedule of both solvers.
    """

    _REQUIRED_DEME_DEVICE_METHODS = (
        "SetOwnerPositionFromDevice",
        "SetOwnerOriQFromDevice",
        "SetOwnerVelocityFromDevice",
        "SetOwnerAngVelGlobalFromDevice",
        "GetOwnerContactWrenchToDevice",
    )

    def __init__(self) -> None:
        self.newton_model = None
        self.deme_solver = None
        self.owner_map = None
        self.device = None
        self._body_indices = None
        self._owner_offsets_by_body = None
        self._body_owner_starts = None
        self._positions = None
        self._orientations = None
        self._velocities = None
        self._angular_velocities = None
        self._forces = None
        self._torques = None
        self.newton_body_forces = None
        self.initialized = False

    def __del__(self) -> None:
        if self.initialized:
            self.finalize()

    def initialize(self, newton_model, deme_solver, owner_map: NewtonDEMEOwnerMap, device) -> None:
        """Bind initialized solvers and allocate the GPU exchange buffers."""
        if self.initialized:
            raise RuntimeError("NewtonDEMEContactCoupler is already initialized.")
        if not device.is_cuda:
            raise ValueError("Newton--DEME GPU contact exchange requires a CUDA Warp device.")
        if max(owner_map.newton_body_indices) >= int(newton_model.body_count):
            raise ValueError(
                f"Newton body mapping contains index {max(owner_map.newton_body_indices)}, "
                f"but the model contains {newton_model.body_count} bodies."
            )

        missing_methods = [name for name in self._REQUIRED_DEME_DEVICE_METHODS if not hasattr(deme_solver, name)]
        if missing_methods:
            raise RuntimeError(
                "GPU-native Newton--DEME contact exchange requires deme3>=3.0.9; "
                f"the DEMSolver is missing {missing_methods}."
            )
        deme_device_ids = tuple(int(device_id) for device_id in deme_solver.GetGPUDeviceIDs())
        if device.ordinal not in deme_device_ids:
            raise ValueError(f"DEME workers {deme_device_ids} do not share Newton's Warp CUDA device {device.ordinal}.")

        self.newton_model = newton_model
        self.deme_solver = deme_solver
        self.owner_map = owner_map
        self.device = device
        owner_count = owner_map.owner_count
        self._body_indices = wp.array(owner_map.newton_body_indices, dtype=wp.int32, device=device)
        owner_offsets_by_body = sorted(
            range(owner_count), key=lambda owner_offset: owner_map.newton_body_indices[owner_offset]
        )
        owners_per_body = [0] * int(newton_model.body_count)
        for body_index in owner_map.newton_body_indices:
            owners_per_body[body_index] += 1
        body_owner_starts = [0]
        for body_owner_count in owners_per_body:
            body_owner_starts.append(body_owner_starts[-1] + body_owner_count)
        self._owner_offsets_by_body = wp.array(owner_offsets_by_body, dtype=wp.int32, device=device)
        self._body_owner_starts = wp.array(body_owner_starts, dtype=wp.int32, device=device)
        self._positions = wp.empty(owner_count, dtype=wp.vec3, device=device)
        self._orientations = wp.empty(owner_count, dtype=wp.quat, device=device)
        self.deme_owner_orientations = self._orientations
        self._velocities = wp.empty(owner_count, dtype=wp.vec3, device=device)
        self._angular_velocities = wp.empty(owner_count, dtype=wp.vec3, device=device)
        self._forces = wp.empty(owner_count, dtype=wp.vec3, device=device)
        self._torques = wp.empty(owner_count, dtype=wp.vec3, device=device)
        self.newton_body_forces = wp.zeros(int(newton_model.body_count), dtype=wp.spatial_vector, device=device)
        self.initialized = True

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("NewtonDEMEContactCoupler must be initialized before use.")

    def set_deme_owner_state_from_newton(self, newton_state) -> None:
        """Upload mapped Newton body pose and velocity to DEME owners."""
        self._require_initialized()
        wp.launch(
            _gather_newton_body_state,
            dim=self.owner_map.owner_count,
            inputs=[
                self._body_indices,
                newton_state.body_q,
                newton_state.body_qd,
                self._positions,
                self._orientations,
                self._velocities,
                self._angular_velocities,
            ],
            device=self.device,
        )
        self._synchronize_newton_to_deme()
        device_ordinal = self.device.ordinal
        first_owner_id = self.owner_map.first_deme_owner_id
        owner_count = self.owner_map.owner_count
        self.deme_solver.SetOwnerPositionFromDevice(first_owner_id, self._positions.ptr, device_ordinal, owner_count)
        self.deme_solver.SetOwnerOriQFromDevice(first_owner_id, self._orientations.ptr, device_ordinal, owner_count)
        self.deme_solver.SetOwnerVelocityFromDevice(first_owner_id, self._velocities.ptr, device_ordinal, owner_count)
        self.deme_solver.SetOwnerAngVelGlobalFromDevice(
            first_owner_id, self._angular_velocities.ptr, device_ordinal, owner_count
        )

    def _synchronize_newton_to_deme(self) -> None:
        """Complete Warp writes before synchronous DEME pointer calls consume them."""
        # This is the single synchronization seam to replace with CUDA
        # stream/event handoff when DEME exposes that capability.
        wp.synchronize_device(self.device)

    def step_deme(self, substeps: int = 1) -> None:
        """Advance DEME independently by the requested number of substeps."""
        self._require_initialized()
        if substeps < 0:
            raise ValueError(f"DEME substeps must be non-negative, got {substeps}.")
        for _ in range(substeps):
            self.deme_solver.DoStepDynamics()

    def get_deme_contact_wrenches_to_device(self, forces=None, torques=None):
        """Write mapped DEME owner wrenches into caller-provided GPU arrays."""
        self._require_initialized()
        owner_count = self.owner_map.owner_count
        forces = self._forces if forces is None else forces
        torques = self._torques if torques is None else torques
        if len(forces) < owner_count or len(torques) < owner_count:
            raise ValueError(
                f"DEME wrench arrays must each hold at least {owner_count} entries; "
                f"got {len(forces)} and {len(torques)}."
            )
        for name, array in (("force", forces), ("torque", torques)):
            if array.device != self.device or array.dtype != wp.vec3:
                raise ValueError(
                    f"DEME {name} output must be a wp.vec3 array on {self.device}; "
                    f"got dtype {array.dtype} on {array.device}."
                )
        self.deme_solver.GetOwnerContactWrenchToDevice(
            forces.ptr,
            torques.ptr,
            owner_count,
            self.device.ordinal,
            self.owner_map.first_deme_owner_id,
            owner_count,
        )
        self._synchronize_deme_to_newton()
        return forces, torques

    def get_deme_contact_accelerations_to_device(self, accelerations, angular_accelerations, angular_frame="global"):
        """Write mapped DEME contact accelerations into caller-owned GPU arrays.

        This raw getter intentionally leaves acceleration-to-wrench semantics to
        the caller, which is useful for diagnostics comparing physical models.
        """
        self._require_initialized()
        if angular_frame not in ("global", "local"):
            raise ValueError(f"angular_frame must be 'global' or 'local', got {angular_frame!r}.")
        angular_method_name = (
            "GetOwnerAngAccGlobalToDevice" if angular_frame == "global" else "GetOwnerAngAccLocalToDevice"
        )
        required_methods = ("GetOwnerAccToDevice", angular_method_name)
        missing = [name for name in required_methods if not hasattr(self.deme_solver, name)]
        if missing:
            raise RuntimeError(f"DEMSolver is missing GPU contact-acceleration getters {missing}.")
        owner_count = self.owner_map.owner_count
        for label, array in (("linear acceleration", accelerations), ("angular acceleration", angular_accelerations)):
            if len(array) < owner_count or array.dtype != wp.vec3 or array.device != self.device:
                raise ValueError(f"DEME {label} output must contain {owner_count} wp.vec3 entries on {self.device}.")
        args = (owner_count, self.device.ordinal, self.owner_map.first_deme_owner_id, owner_count)
        self.deme_solver.GetOwnerAccToDevice(accelerations.ptr, *args)
        getattr(self.deme_solver, angular_method_name)(angular_accelerations.ptr, *args)
        self._synchronize_deme_to_newton()
        return accelerations, angular_accelerations

    def _synchronize_deme_to_newton(self) -> None:
        """Complete DEME writes before Warp consumes their destination buffers."""
        # DEME 3.0.9 device getters are synchronous, so returning from the call
        # is the required handoff today. Keep this seam for future event/stream
        # synchronization without changing the public exchange API.
        pass

    def write_deme_contact_wrenches_to_newton(self, destination=None, clear_destination: bool = True):
        """Gather and accumulate mapped DEME owner wrenches by Newton body."""
        self._require_initialized()
        destination = self.newton_body_forces if destination is None else destination
        if len(destination) != int(self.newton_model.body_count):
            raise ValueError(
                "Newton body-force destination must contain one entry per model body; "
                f"expected {self.newton_model.body_count}, got {len(destination)}."
            )
        if destination.device != self.device or destination.dtype != wp.spatial_vector:
            raise ValueError(
                f"Newton body-force destination must be a wp.spatial_vector array on {self.device}; "
                f"got dtype {destination.dtype} on {destination.device}."
            )
        self.get_deme_contact_wrenches_to_device()
        wp.launch(
            _accumulate_deme_owner_wrenches,
            dim=int(self.newton_model.body_count),
            inputs=[
                self._owner_offsets_by_body,
                self._body_owner_starts,
                self._forces,
                self._torques,
                int(clear_destination),
                destination,
            ],
            device=self.device,
        )
        return destination

    def finalize(self) -> None:
        """Release references to solver-owned objects and exchange buffers."""
        self.newton_model = None
        self.deme_solver = None
        self._body_indices = None
        self._owner_offsets_by_body = None
        self._body_owner_starts = None
        self._positions = None
        self._orientations = None
        self.deme_owner_orientations = None
        self._velocities = None
        self._angular_velocities = None
        self._forces = None
        self._torques = None
        self.newton_body_forces = None
        self.owner_map = None
        self.device = None
        self.initialized = False


class NewtonDEMEContactAccelerationCoupler:
    """Preserve contact-acceleration semantics with GPU-only runtime exchange."""

    _REQUIRED_METHODS = (
        "SetOwnerPositionFromDevice",
        "SetOwnerOriQFromDevice",
        "GetOwnerAccToDevice",
        "GetOwnerAngAccLocalToDevice",
        "GetOwnerMassToDevice",
        "GetOwnerMOIToDevice",
    )

    def __init__(self) -> None:
        self.initialized = False

    def __del__(self) -> None:
        if self.initialized:
            self.finalize()

    def initialize(
        self,
        newton_model,
        deme_solver,
        owner_map: NewtonDEMEOwnerMap,
        local_offsets: Sequence[Sequence[float]],
        device,
        force_scale: float = 1.0,
        local_orientations: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Bind initialized solvers and allocate acceleration-exchange buffers."""
        if self.initialized:
            raise RuntimeError("NewtonDEMEContactAccelerationCoupler is already initialized.")
        if not device.is_cuda:
            raise ValueError("Newton--DEME GPU contact exchange requires a CUDA Warp device.")
        if len(set(owner_map.newton_body_indices)) != owner_map.owner_count:
            raise ValueError("Acceleration feedback currently requires one DEME proxy per Newton body.")
        if len(local_offsets) != owner_map.owner_count:
            raise ValueError("One body-local offset is required per DEME owner.")
        if local_orientations is None:
            local_orientations = [(0.0, 0.0, 0.0, 1.0)] * owner_map.owner_count
        if len(local_orientations) != owner_map.owner_count:
            raise ValueError("One body-local orientation is required per DEME owner.")
        if max(owner_map.newton_body_indices) >= int(newton_model.body_count):
            raise ValueError("Newton--DEME mapping contains an out-of-range body index.")
        missing = [name for name in self._REQUIRED_METHODS if not hasattr(deme_solver, name)]
        if missing:
            raise RuntimeError(f"DEMSolver is missing GPU exchange methods {missing}.")
        if device.ordinal not in tuple(int(value) for value in deme_solver.GetGPUDeviceIDs()):
            raise ValueError("DEME and Newton must share the selected CUDA device.")

        self.newton_model = newton_model
        self.deme_solver = deme_solver
        self.owner_map = owner_map
        self.device = device
        self.force_scale = float(force_scale)
        count = owner_map.owner_count
        self._body_indices = wp.array(owner_map.newton_body_indices, dtype=wp.int32, device=device)
        self._local_offsets = wp.array(local_offsets, dtype=wp.vec3, device=device)
        self._local_orientations = wp.array(local_orientations, dtype=wp.quat, device=device)
        self._positions = wp.empty(count, dtype=wp.vec3, device=device)
        self._orientations = wp.empty(count, dtype=wp.quat, device=device)
        self._accelerations = wp.empty(count, dtype=wp.vec3, device=device)
        self._local_angular_accelerations = wp.empty(count, dtype=wp.vec3, device=device)
        self._masses = wp.empty(count, dtype=wp.float32, device=device)
        self._moments = wp.empty(count, dtype=wp.vec3, device=device)
        self.newton_body_forces = wp.zeros(int(newton_model.body_count), dtype=wp.spatial_vector, device=device)
        args = (count, device.ordinal, owner_map.first_deme_owner_id, count)
        deme_solver.GetOwnerMassToDevice(self._masses.ptr, *args)
        deme_solver.GetOwnerMOIToDevice(self._moments.ptr, *args)
        self.initialized = True

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("NewtonDEMEContactAccelerationCoupler must be initialized before use.")

    def set_deme_owner_pose_from_newton(self, newton_state) -> None:
        """Update offset DEME proxy poses directly from Newton's device state."""
        self._require_initialized()
        wp.launch(
            _gather_newton_proxy_pose,
            dim=self.owner_map.owner_count,
            inputs=[
                self._body_indices,
                self._local_offsets,
                self._local_orientations,
                newton_state.body_q,
                self._positions,
                self._orientations,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        first_owner = self.owner_map.first_deme_owner_id
        count = self.owner_map.owner_count
        self.deme_solver.SetOwnerPositionFromDevice(first_owner, self._positions.ptr, self.device.ordinal, count)
        self.deme_solver.SetOwnerOriQFromDevice(first_owner, self._orientations.ptr, self.device.ordinal, count)

    def step_deme(self, substeps: int = 1) -> None:
        """Advance DEME independently."""
        self._require_initialized()
        if substeps < 0:
            raise ValueError(f"DEME substeps must be non-negative, got {substeps}.")
        for _ in range(substeps):
            self.deme_solver.DoStepDynamics()

    def write_deme_contact_accelerations_to_newton(self):
        """Convert DEME contact accelerations and write Newton body forces on-device."""
        self._require_initialized()
        count = self.owner_map.owner_count
        args = (count, self.device.ordinal, self.owner_map.first_deme_owner_id, count)
        self.deme_solver.GetOwnerAccToDevice(self._accelerations.ptr, *args)
        self.deme_solver.GetOwnerAngAccLocalToDevice(self._local_angular_accelerations.ptr, *args)
        self.newton_body_forces.zero_()
        wp.launch(
            _contact_accelerations_to_newton_forces,
            dim=count,
            inputs=[
                self._body_indices,
                self._orientations,
                self._masses,
                self._moments,
                self._accelerations,
                self._local_angular_accelerations,
                self.force_scale,
                self.newton_body_forces,
            ],
            device=self.device,
        )
        return self.newton_body_forces

    def finalize(self) -> None:
        """Release solver and device-array references."""
        self.__dict__.clear()
        self.initialized = False
