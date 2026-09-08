"""Reusable XLB lattice conversion and independently stepped field state."""

from dataclasses import dataclass

import numpy as np
import warp as wp


def velocity_stencil_array(velocity_set) -> np.ndarray:
    """Return an XLB velocity set as a direction-major ``(q, d)`` integer array."""
    source = velocity_set.c.numpy() if hasattr(velocity_set.c, "numpy") else velocity_set.c
    try:
        values = np.asarray(source, dtype=np.int32)
    except (TypeError, ValueError):
        values = np.empty((0, 0), dtype=np.int32)
    if values.shape not in ((velocity_set.d, velocity_set.q), (velocity_set.q, velocity_set.d)):
        values = np.asarray(
            [
                [int(velocity_set.c[axis, direction]) for direction in range(velocity_set.q)]
                for axis in range(velocity_set.d)
            ],
            dtype=np.int32,
        )
    values = values.T if values.shape[0] == velocity_set.d else values
    if values.shape != (velocity_set.q, velocity_set.d):
        raise ValueError(f"Expected XLB stencil shape {(velocity_set.q, velocity_set.d)}, got {values.shape}.")
    return values


def velocity_stencil_to_warp(velocity_set, device):
    """Create the canonical direction-major XLB stencil on a Warp device."""
    return wp.array(velocity_stencil_array(velocity_set), dtype=wp.int32, device=device)


def bgk_omega_from_lattice_viscosity(lattice_kinematic_viscosity: float) -> float:
    """Convert positive lattice kinematic viscosity to BGK relaxation rate."""
    viscosity = float(lattice_kinematic_viscosity)
    if viscosity <= 0.0:
        raise ValueError("XLB lattice kinematic viscosity must be positive.")
    return 1.0 / (3.0 * viscosity + 0.5)


@dataclass(frozen=True)
class XLBPhysicalScaling:
    """Explicit isotropic SI-to-lattice mapping for an XLB simulation."""

    cell_size: float
    timestep: float
    physical_kinematic_viscosity: float
    physical_density: float = 1.0

    def __post_init__(self):
        if min(self.cell_size, self.timestep, self.physical_kinematic_viscosity, self.physical_density) <= 0.0:
            raise ValueError("XLB physical scaling values must all be positive.")

    @classmethod
    def from_domain(
        cls, domain_min, domain_max, grid_shape, timestep, physical_kinematic_viscosity, physical_density=1.0
    ):
        cell_sizes = (np.asarray(domain_max, dtype=np.float64) - np.asarray(domain_min, dtype=np.float64)) / np.asarray(
            grid_shape, dtype=np.float64
        )
        if np.any(cell_sizes <= 0.0) or not np.allclose(cell_sizes, cell_sizes[0]):
            raise ValueError(f"XLB physical scaling requires positive isotropic cells, got {cell_sizes}.")
        return cls(float(cell_sizes[0]), float(timestep), float(physical_kinematic_viscosity), float(physical_density))

    @property
    def lattice_kinematic_viscosity(self) -> float:
        return self.physical_kinematic_viscosity * self.timestep / self.cell_size**2

    @property
    def omega(self) -> float:
        return bgk_omega_from_lattice_viscosity(self.lattice_kinematic_viscosity)

    @property
    def relaxation_time(self) -> float:
        return 1.0 / self.omega

    def velocity_to_lattice(self, velocity):
        """Convert an SI scalar/vector velocity to lattice units."""
        return np.asarray(velocity) * self.timestep / self.cell_size


class XLBStepperState:
    """Own XLB fields, ping-pong ordering, relaxation rate, and timestep count."""

    def __init__(self):
        self.initialized = False

    def initialize(self, stepper, omega: float, timestep: int = 0):
        if self.initialized:
            raise RuntimeError("XLBStepperState is already initialized.")
        if not 0.0 < float(omega) < 2.0:
            raise ValueError("XLB BGK omega must lie between zero and two.")
        self.stepper = stepper
        self.omega = float(omega)
        self.timestep = int(timestep)
        self.f0, self.f1, self.bc_mask, self.missing_mask = stepper.prepare_fields()
        self.initialized = True
        return self

    def step(self):
        """Advance XLB once and preserve the established population-buffer ordering."""
        if not self.initialized:
            raise RuntimeError("XLBStepperState must be initialized before use.")
        self.f0, self.f1 = self.stepper(self.f0, self.f1, self.bc_mask, self.missing_mask, self.omega, self.timestep)
        self.f0, self.f1 = self.f1, self.f0
        self.timestep += 1
        return self.f0

    def finalize(self):
        self.__dict__.clear()
        self.initialized = False
