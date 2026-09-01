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
        if len(set(body_indices)) != len(body_indices):
            raise ValueError("Each Newton body may appear only once in a Newton--DEME owner map.")
        if any(index < 0 for index in body_indices):
            raise ValueError("Newton body indices must be non-negative.")
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
def _scatter_deme_owner_wrenches(
    body_indices: wp.array(dtype=wp.int32),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    body_forces: wp.array(dtype=wp.spatial_vector),
):
    owner_offset = wp.tid()
    body_forces[body_indices[owner_offset]] = wp.spatial_vector(forces[owner_offset], torques[owner_offset])


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
        self._positions = wp.empty(owner_count, dtype=wp.vec3, device=device)
        self._orientations = wp.empty(owner_count, dtype=wp.quat, device=device)
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
        # DEME's synchronous pointer APIs do not consume Warp's stream.
        wp.synchronize_device(self.device)
        device_ordinal = self.device.ordinal
        first_owner_id = self.owner_map.first_deme_owner_id
        owner_count = self.owner_map.owner_count
        self.deme_solver.SetOwnerPositionFromDevice(first_owner_id, self._positions.ptr, device_ordinal, owner_count)
        self.deme_solver.SetOwnerOriQFromDevice(first_owner_id, self._orientations.ptr, device_ordinal, owner_count)
        self.deme_solver.SetOwnerVelocityFromDevice(first_owner_id, self._velocities.ptr, device_ordinal, owner_count)
        self.deme_solver.SetOwnerAngVelGlobalFromDevice(
            first_owner_id, self._angular_velocities.ptr, device_ordinal, owner_count
        )

    def step_deme(self, substeps: int = 1) -> None:
        """Advance DEME independently by the requested number of substeps."""
        self._require_initialized()
        if substeps < 0:
            raise ValueError(f"DEME substeps must be non-negative, got {substeps}.")
        for _ in range(substeps):
            self.deme_solver.DoStepDynamics()

    def write_deme_contact_wrenches_to_newton(self) -> None:
        """Replace the mapped entries in ``newton_body_forces`` on the GPU."""
        self._require_initialized()
        owner_count = self.owner_map.owner_count
        self.deme_solver.GetOwnerContactWrenchToDevice(
            self._forces.ptr,
            self._torques.ptr,
            owner_count,
            self.device.ordinal,
            self.owner_map.first_deme_owner_id,
            owner_count,
        )
        self.newton_body_forces.zero_()
        wp.launch(
            _scatter_deme_owner_wrenches,
            dim=owner_count,
            inputs=[self._body_indices, self._forces, self._torques, self.newton_body_forces],
            device=self.device,
        )

    def finalize(self) -> None:
        """Release references to solver-owned objects and exchange buffers."""
        self.newton_model = None
        self.deme_solver = None
        self._body_indices = None
        self._positions = None
        self._orientations = None
        self._velocities = None
        self._angular_velocities = None
        self._forces = None
        self._torques = None
        self.newton_body_forces = None
        self.owner_map = None
        self.device = None
        self.initialized = False
