"""GPU-native DEME particle state made available to Newton-side Warp code."""

import warp as wp


@wp.kernel
def _forces_to_accelerations(
    forces: wp.array(dtype=wp.vec3), masses: wp.array(dtype=wp.float32), accelerations: wp.array(dtype=wp.vec3)
):
    index = wp.tid()
    accelerations[index] = forces[index] / masses[index]


class NewtonDEMEParticleExchange:
    """Read one consecutive DEME particle-owner span into reusable Warp arrays."""

    def __init__(self):
        self.deme_solver = None
        self.first_owner_id = None
        self.owner_count = None
        self.device = None
        self.positions = None
        self.velocities = None
        self.orientations = None
        self.masses = None
        self.linear_accelerations = None
        self.initialized = False

    def initialize(self, deme_solver, first_owner_id: int, owner_count: int, device):
        """Bind an initialized DEMSolver and allocate reusable output arrays."""
        if self.initialized:
            raise RuntimeError("NewtonDEMEParticleExchange is already initialized.")
        if first_owner_id < 0 or owner_count <= 0:
            raise ValueError("DEME particle owner span requires a non-negative start and positive count.")
        if not device.is_cuda:
            raise ValueError("Newton--DEME particle exchange requires a CUDA Warp device.")
        for method in (
            "GetOwnerPositionToDevice",
            "GetOwnerVelocityToDevice",
            "GetOwnerOriQToDevice",
            "GetOwnerMassToDevice",
            "AddOwnerNextStepAccFromDevice",
            "AddOwnerNextStepAngAccFromDevice",
        ):
            if not hasattr(deme_solver, method):
                raise RuntimeError(f"DEMSolver lacks required device method {method}.")
        if device.ordinal not in tuple(int(value) for value in deme_solver.GetGPUDeviceIDs()):
            raise ValueError("DEME and Newton must share the selected CUDA device.")
        self.deme_solver = deme_solver
        self.first_owner_id = int(first_owner_id)
        self.owner_count = int(owner_count)
        self.device = device
        self.positions = wp.empty(owner_count, dtype=wp.vec3, device=device)
        self.velocities = wp.empty(owner_count, dtype=wp.vec3, device=device)
        self.orientations = wp.empty(owner_count, dtype=wp.quat, device=device)
        self.masses = wp.empty(owner_count, dtype=wp.float32, device=device)
        self.linear_accelerations = wp.empty(owner_count, dtype=wp.vec3, device=device)
        deme_solver.GetOwnerMassToDevice(self.masses.ptr, owner_count, device.ordinal, self.first_owner_id, owner_count)
        self.initialized = True

    def read_deme_particle_state(self):
        """Synchronously refresh and return device-resident position and velocity arrays."""
        if not self.initialized:
            raise RuntimeError("NewtonDEMEParticleExchange must be initialized before use.")
        args = (self.owner_count, self.device.ordinal, self.first_owner_id, self.owner_count)
        self.deme_solver.GetOwnerPositionToDevice(self.positions.ptr, *args)
        self.deme_solver.GetOwnerVelocityToDevice(self.velocities.ptr, *args)
        return self.positions, self.velocities

    def read_deme_particle_poses(self):
        """Synchronously refresh and return device-resident positions and orientations."""
        if not self.initialized:
            raise RuntimeError("NewtonDEMEParticleExchange must be initialized before use.")
        args = (self.owner_count, self.device.ordinal, self.first_owner_id, self.owner_count)
        self.deme_solver.GetOwnerPositionToDevice(self.positions.ptr, *args)
        self.deme_solver.GetOwnerOriQToDevice(self.orientations.ptr, *args)
        return self.positions, self.orientations

    def queue_deme_next_step_accelerations(self, linear_accelerations, local_angular_accelerations=None):
        """Queue caller-computed device accelerations for exactly one DEME step."""
        self._validate_vec3_array(linear_accelerations, "linear acceleration")
        wp.synchronize_device(self.device)
        self.deme_solver.AddOwnerNextStepAccFromDevice(
            self.first_owner_id, linear_accelerations.ptr, self.device.ordinal, self.owner_count
        )
        if local_angular_accelerations is not None:
            self._validate_vec3_array(local_angular_accelerations, "local angular acceleration")
            self.deme_solver.AddOwnerNextStepAngAccFromDevice(
                self.first_owner_id, local_angular_accelerations.ptr, self.device.ordinal, self.owner_count
            )

    def queue_deme_next_step_forces(self, global_forces):
        """Convert global device forces using DEME masses and queue their linear accelerations."""
        self._validate_vec3_array(global_forces, "global force")
        wp.launch(
            _forces_to_accelerations,
            dim=self.owner_count,
            inputs=[global_forces, self.masses, self.linear_accelerations],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        self.queue_deme_next_step_accelerations(self.linear_accelerations)

    def _validate_vec3_array(self, array, label):
        if len(array) < self.owner_count or array.dtype != wp.vec3 or array.device != self.device:
            raise ValueError(f"DEME {label} must contain at least {self.owner_count} wp.vec3 entries on {self.device}.")

    def finalize(self):
        """Release solver and device-array references."""
        self.__init__()
