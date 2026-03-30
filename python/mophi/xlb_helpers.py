"""mophi.xlb_helpers — Utility functions for XLB (Lattice-Boltzmann) post-processing.

These helpers are solver-agnostic: they operate on raw NumPy velocity arrays and
can be called from within any MoPhi coupler or directly from user scripts and demos.
"""

import numpy as np


def xlb_build_streamlines(
    u,
    domain_min,
    domain_max,
    n_x_seeds=12,
    n_z_seeds=8,
    n_steps=40,
    step_size=0.12,
    seed_y=None,
):
    """Trace arc-length-parameterised streamlines through velocity field *u*.

    Seeds a regular (n_x_seeds × n_z_seeds) grid on the inlet face (y ≈ domain_max)
    and advances each streamline with a normalised Euler step until the path exits
    the domain or the local speed stalls.

    Args:
        u:          (Nx, Ny, Nz, 3) float32 velocity field [m/s or lattice units].
        domain_min: (3,) world-space lower corner of the LBM domain [m].
        domain_max: (3,) world-space upper corner of the LBM domain [m].
        n_x_seeds:  number of seed positions along x.
        n_z_seeds:  number of seed positions along z.
        n_steps:    maximum integration steps per streamline.
        step_size:  arc-length step [m] — controls sample density along lines.
        seed_y:     y-coordinate of the seed plane.  Defaults to domain_max[1] − 0.05
                    (just inside the inlet face at y = domain_max, flow in −y direction).

    Returns:
        pts: (N, 3) float32 — world-space positions along all streamlines.
        spd: (N,)   float32 — velocity magnitude at each sample point.
    """
    nx, ny, nz, _ = u.shape
    if seed_y is None:
        # Default: seed just inside the inlet face (y = domain_max, flow in −y direction).
        seed_y = float(domain_max[1]) - 0.05

    def _interp(pos):
        """Trilinear velocity interpolation at world-space position *pos*."""
        t = (pos - domain_min) / (domain_max - domain_min)
        # Clamp so that the lower-cell index stays at most n-2, keeping (index+1)
        # in-bounds.  The 0.001 epsilon prevents reaching the upper-cell face.
        gx = float(np.clip(t[0] * (nx - 1), 0.0, nx - 1.001))
        gy = float(np.clip(t[1] * (ny - 1), 0.0, ny - 1.001))
        gz = float(np.clip(t[2] * (nz - 1), 0.0, nz - 1.001))
        ix, iy, iz = int(gx), int(gy), int(gz)
        fx, fy, fz = gx - ix, gy - iy, gz - iz
        return (
            u[ix, iy, iz] * (1 - fx) * (1 - fy) * (1 - fz)
            + u[ix + 1, iy, iz] * fx * (1 - fy) * (1 - fz)
            + u[ix, iy + 1, iz] * (1 - fx) * fy * (1 - fz)
            + u[ix, iy, iz + 1] * (1 - fx) * (1 - fy) * fz
            + u[ix + 1, iy + 1, iz] * fx * fy * (1 - fz)
            + u[ix + 1, iy, iz + 1] * fx * (1 - fy) * fz
            + u[ix, iy + 1, iz + 1] * (1 - fx) * fy * fz
            + u[ix + 1, iy + 1, iz + 1] * fx * fy * fz
        )

    # Inset seeds 0.4 m from the x-edges and 0.15 m from the z-edges to avoid
    # the low-speed boundary layers where the no-slip profile approaches zero.
    xs = np.linspace(domain_min[0] + 0.4, domain_max[0] - 0.4, n_x_seeds)
    zs = np.linspace(0.15, domain_max[2] - 0.15, n_z_seeds)

    all_pts: list = []
    all_spd: list = []
    for x0 in xs:
        for z0 in zs:
            pos = np.array([x0, seed_y, z0], dtype=np.float64)
            for _ in range(n_steps):
                if not np.all((pos >= domain_min) & (pos <= domain_max)):
                    break
                vel = _interp(pos).astype(np.float64)
                spd = float(np.linalg.norm(vel))
                if spd < 1e-6:  # stall threshold: treat as zero-velocity region
                    break
                all_pts.append(pos.astype(np.float32).copy())
                all_spd.append(spd)
                pos += step_size * vel / spd  # arc-length step

    if not all_pts:
        return np.zeros((1, 3), dtype=np.float32), np.zeros(1, dtype=np.float32)
    return np.array(all_pts, dtype=np.float32), np.array(all_spd, dtype=np.float32)
