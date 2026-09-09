"""Deterministic XLB geometry preparation for the ANYmal experiment."""

from dataclasses import dataclass

import numpy as np
import xlb

from mophi.couplers.newton_xlb import (
    XLBStepperState,
    prescribed_box_grid,
    velocity_stencil_array,
    velocity_stencil_to_warp,
)
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.operator.boundary_condition import ExtrapolationOutflowBC, HalfwayBounceBackBC, ZouHeBC
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.precision_policy import Precision, PrecisionPolicy


@dataclass
class AnymalXLBScene:
    """Strongly own every XLB object formerly retained by the demo module."""

    backend: object
    precision: object
    velocity_set: object
    grid: object
    inlet_indices: object
    outlet_indices: object
    wall_indices: object
    robot_indices: object
    inlet_boundary: object
    wall_boundary: object
    outlet_boundary: object
    robot_boundary: object
    robot_grid_min: np.ndarray
    robot_grid_max: np.ndarray
    stepper: object
    runtime: XLBStepperState
    velocity_stencil_numpy: np.ndarray
    velocity_stencil_warp: object
    macroscopic: object
    density_field: object
    velocity_field: object

    def warm_up(self, step_count):
        """Advance the fixed initial scene before coupled motion begins."""
        for _ in range(step_count):
            self.runtime.step()

    def sample_velocity_numpy(self):
        """Sample the current macroscopic velocity in visualization layout."""
        self.density_field, self.velocity_field = self.macroscopic(
            self.runtime.f0,
            self.density_field,
            self.velocity_field,
        )
        velocity = self.velocity_field.numpy().transpose(1, 2, 3, 0).astype(np.float32)
        if not np.all(np.isfinite(velocity)):
            raise RuntimeError("XLB produced a non-finite macroscopic velocity field.")
        if not np.any(velocity):
            raise RuntimeError("XLB produced an entirely zero macroscopic velocity field despite its inlet.")
        return velocity

    def finalize(self):
        """Release the reusable XLB runtime fields."""
        self.runtime.finalize()


def domain_boundary_indices(grid, velocity_dimension):
    """Return the experiment's inlet, outlet, and non-overlapping wall indices."""
    box = grid.bounding_box_indices()
    box_without_edges = grid.bounding_box_indices(remove_edges=True)
    inlet_indices = box_without_edges["back"]
    outlet_indices = box_without_edges["front"]
    wall_indices = [
        box["bottom"][axis] + box["top"][axis] + box["left"][axis] + box["right"][axis]
        for axis in range(velocity_dimension)
    ]
    wall_indices = np.unique(np.array(wall_indices), axis=-1).tolist()
    return inlet_indices, outlet_indices, wall_indices


def initial_robot_box_indices(world_min, world_max, domain_min, domain_max, grid_shape):
    """Return inclusive grid bounds and index lists for the initial robot AABB."""
    grid_min, grid_max = prescribed_box_grid(world_min, world_max, domain_min, domain_max, grid_shape)
    if grid_min is None:
        grid_min = np.array([grid_shape[0] // 2, grid_shape[1] // 2, grid_shape[2] // 2])
        grid_max = grid_min.copy()
    x_indices = np.arange(grid_min[0], grid_max[0] + 1)
    y_indices = np.arange(grid_min[1], grid_max[1] + 1)
    z_indices = np.arange(grid_min[2], grid_max[2] + 1)
    grid_x, grid_y, grid_z = np.meshgrid(x_indices, y_indices, z_indices, indexing="ij")
    indices = [grid_x.flatten().tolist(), grid_y.flatten().tolist(), grid_z.flatten().tolist()]
    return grid_min, grid_max, indices


def build_anymal_xlb_scene(
    grid_shape,
    inlet_speed,
    omega,
    robot_world_min,
    robot_world_max,
    domain_min,
    domain_max,
):
    """Construct and retain the experiment's XLB objects without advancing them."""
    backend = ComputeBackend.WARP
    precision = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q19(precision_policy=precision, compute_backend=backend)
    xlb.init(velocity_set, backend, precision)
    grid = grid_factory(grid_shape, compute_backend=backend)
    inlet_indices, outlet_indices, wall_indices = domain_boundary_indices(grid, velocity_set.d)
    inlet_boundary = ZouHeBC(
        bc_type="velocity",
        prescribed_value=np.array([0.0, inlet_speed, 0.0]),
        indices=inlet_indices,
    )
    wall_boundary = HalfwayBounceBackBC(indices=wall_indices)
    outlet_boundary = ExtrapolationOutflowBC(indices=outlet_indices)
    robot_grid_min, robot_grid_max, robot_indices = initial_robot_box_indices(
        robot_world_min, robot_world_max, domain_min, domain_max, grid_shape
    )
    robot_boundary = HalfwayBounceBackBC(indices=robot_indices)
    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=[wall_boundary, inlet_boundary, outlet_boundary, robot_boundary],
        collision_type="BGK",
    )
    runtime = XLBStepperState().initialize(stepper, omega)
    velocity_stencil_numpy = velocity_stencil_array(velocity_set)
    velocity_stencil_warp = velocity_stencil_to_warp(velocity_set, runtime.bc_mask.device)
    macroscopic = Macroscopic(velocity_set, precision, backend)
    density_field = grid.create_field(cardinality=1, dtype=Precision.FP32)
    velocity_field = grid.create_field(cardinality=3, dtype=Precision.FP32)
    return AnymalXLBScene(
        backend,
        precision,
        velocity_set,
        grid,
        inlet_indices,
        outlet_indices,
        wall_indices,
        robot_indices,
        inlet_boundary,
        wall_boundary,
        outlet_boundary,
        robot_boundary,
        robot_grid_min,
        robot_grid_max,
        stepper,
        runtime,
        velocity_stencil_numpy,
        velocity_stencil_warp,
        macroscopic,
        density_field,
        velocity_field,
    )
