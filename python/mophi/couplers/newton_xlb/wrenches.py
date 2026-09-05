"""GPU-native XLB wrench delivery to Newton body-force arrays."""

from collections.abc import Sequence

import warp as wp


@wp.kernel
def _scatter_body_wrenches(
    body_indices: wp.array(dtype=wp.int32),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    destination: wp.array(dtype=wp.spatial_vector),
):
    index = wp.tid()
    destination[body_indices[index]] += wp.spatial_vector(forces[index], torques[index])


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

    def finalize(self):
        """Release model and device-array references."""
        self.__dict__.clear()
        self.initialized = False
