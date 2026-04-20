"""mophi.xlb_helpers — Utility functions for XLB (Lattice-Boltzmann) post-processing.

These helpers are solver-agnostic: they operate on raw NumPy velocity arrays and
can be called from within any MoPhi coupler or directly from user scripts and demos.
"""

import numpy as np


def xlb_make_y_plane_seeds(
    domain_min,
    domain_max,
    n_x_seeds=12,
    n_z_seeds=8,
    seed_y=None,
    x_inset=0.4,
    z_inset=0.15,
):
    """Build a regular grid of seed points on a y = const plane.

    This is the canonical seed-generation helper for flows that enter the domain
    through a face at y ≈ domain_max and travel in the −y direction (e.g. the
    Newton-XLB-DEM demo).  Pass the returned array directly to
    :func:`xlb_build_streamlines` as the *seed_points* argument.

    Args:
        domain_min: (3,) world-space lower corner of the LBM domain [m].
        domain_max: (3,) world-space upper corner of the LBM domain [m].
        n_x_seeds:  number of seed positions along x.
        n_z_seeds:  number of seed positions along z.
        seed_y:     y-coordinate of the seed plane.  Defaults to
                    ``domain_max[1] − 0.05`` (just inside the inlet face).
        x_inset:    distance [m] to inset from the x-edges of the domain.
                    Avoids the near-wall low-speed boundary layers.
        z_inset:    distance [m] to inset from the z-edges of the domain.

    Returns:
        seed_points: (n_x_seeds * n_z_seeds, 3) float64 array of seed positions.
    """
    domain_min = np.asarray(domain_min, dtype=np.float64)
    domain_max = np.asarray(domain_max, dtype=np.float64)
    if seed_y is None:
        seed_y = float(domain_max[1]) - 0.05
    xs = np.linspace(domain_min[0] + x_inset, domain_max[0] - x_inset, n_x_seeds)
    zs = np.linspace(domain_min[2] + z_inset, domain_max[2] - z_inset, n_z_seeds)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    seeds = np.stack([xx.ravel(), np.full(xx.size, seed_y), zz.ravel()], axis=1)
    return seeds.astype(np.float64)


def xlb_build_streamlines(
    u,
    domain_min,
    domain_max,
    seed_points,
    n_steps=40,
    step_size=0.12,
):
    """Trace arc-length-parameterised streamlines through velocity field *u*.

    Advances each seed point with a normalised Euler integration step until the
    path exits the domain or the local speed stalls.  Seeds can be placed
    anywhere in the domain — use :func:`xlb_make_y_plane_seeds` to generate a
    regular y-plane grid, or supply an arbitrary ``(N, 3)`` array to seed from
    any geometry (a sphere surface, a slice, a set of particle positions, etc.).

    Args:
        u:           (Nx, Ny, Nz, 3) float32 velocity field [m/s or lattice units].
        domain_min:  (3,) world-space lower corner of the LBM domain [m].
        domain_max:  (3,) world-space upper corner of the LBM domain [m].
        seed_points: (N, 3) array of world-space seed positions [m].
                     Each row is one starting point for a streamline.
        n_steps:     maximum integration steps per streamline.
        step_size:   arc-length step [m] — controls sample density along lines.

    Returns:
        pts:  (M, 3) float32 — world-space positions along all streamlines.
        spd:  (M,)   float32 — velocity magnitude at each sample point.
        dirs: (M, 3) float32 — unit velocity direction at each sample point.
    """
    nx, ny, nz, _ = u.shape
    domain_min = np.asarray(domain_min, dtype=np.float64)
    domain_max = np.asarray(domain_max, dtype=np.float64)
    seed_points = np.asarray(seed_points, dtype=np.float64)
    if seed_points.ndim == 1:
        seed_points = seed_points[np.newaxis, :]

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

    all_pts: list = []
    all_spd: list = []
    all_dirs: list = []
    for seed in seed_points:
        pos = seed.copy()
        for _ in range(n_steps):
            if not np.all((pos >= domain_min) & (pos <= domain_max)):
                break
            vel = _interp(pos).astype(np.float64)
            spd = float(np.linalg.norm(vel))
            if spd < 1e-6:  # stall threshold: treat as zero-velocity region
                break
            all_pts.append(pos.astype(np.float32).copy())
            all_spd.append(spd)
            all_dirs.append((vel / spd).astype(np.float32))
            pos += step_size * vel / spd  # arc-length step

    if not all_pts:
        return (
            np.zeros((1, 3), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )
    return (
        np.array(all_pts, dtype=np.float32),
        np.array(all_spd, dtype=np.float32),
        np.array(all_dirs, dtype=np.float32),
    )


def xlb_make_streamline_warp_arrays(
    pts,
    spd,
    dirs,
    arrow_interval=5,
    n_shaft=3,
    shaft_step=0.040,
):
    """Convert streamline sample arrays to arrow-glyph Warp arrays for ViewerGL rendering.

    Instead of a uniform chain of dots, each arrow glyph consists of one large
    arrowhead sphere (at the sampled position, pointing in the flow direction)
    followed by ``n_shaft`` progressively smaller shaft spheres trailing
    behind the head along ``-dirs[i]``.  This makes the flow direction
    immediately legible — in the Newton-XLB-DEM demo the arrows should point
    predominantly in the −y direction.

    Colours are mapped from slow → cornflower-blue (0.20, 0.55, 0.85) to
    fast → cyan-white (0.55, 0.90, 1.00); this cool palette contrasts with the
    warm-orange DEM particles without occluding the robot or ground plane.

    Args:
        pts:            (N, 3) float32 — world-space streamline sample positions.
        spd:            (N,)   float32 — velocity magnitude at each sample point.
        dirs:           (N, 3) float32 — unit velocity direction at each sample point,
                        as returned by :func:`xlb_build_streamlines`.
        arrow_interval: stride between sampled arrow positions (default 5 steps ≈ 0.6 m).
        n_shaft:        number of shaft spheres trailing behind the arrowhead (default 3).
        shaft_step:     spacing [m] between consecutive shaft spheres (default 0.04 m).

    Returns:
        pos_wp:    warp.array of warp.vec3 — positions of all glyph spheres.
        radii_wp:  warp.array of warp.float32 — per-sphere radius [m].
        colors_wp: warp.array of warp.vec3 — per-sphere RGB colour.
    """
    import warp as wp  # lazy import: warp is only required when this function is called

    pts = np.asarray(pts, dtype=np.float32)
    spd = np.asarray(spd, dtype=np.float32)
    dirs = np.asarray(dirs, dtype=np.float32)
    spd_max = float(spd.max()) if spd.max() > 0 else 1.0
    t_all = np.clip(spd / spd_max, 0.0, 1.0).astype(np.float32)

    # Arrowhead radius is visibly larger than the shaft so the tip stands out.
    # Shaft radius tapers from near-head to tail for a "bullet" silhouette.
    head_radius = 0.026  # arrowhead sphere radius [m]
    shaft_radius_near = 0.010  # shaft radius adjacent to arrowhead [m]
    shaft_radius_far = 0.005  # shaft radius at the tail end [m]

    arrow_positions: list = []
    arrow_radii: list = []
    arrow_colors: list = []

    for i in range(0, len(pts), arrow_interval):
        pos = pts[i]
        d = dirs[i]  # unit vector pointing in the flow direction
        t = float(t_all[i])
        color = np.array(
            [0.20 + 0.35 * t, 0.55 + 0.35 * t, 0.85 + 0.15 * t],
            dtype=np.float32,
        )

        # Arrowhead: large sphere at the current (leading) streamline position.
        arrow_positions.append(pos)
        arrow_radii.append(head_radius)
        arrow_colors.append(color)

        # Shaft: smaller spheres placed behind the head along -d.
        for k in range(1, n_shaft + 1):
            shaft_pos = (pos - d * shaft_step * k).astype(np.float32)
            # Linearly taper radius from near-head to far-tail.
            # max(..., 1) avoids division by zero when n_shaft=1.
            alpha = (k - 1) / max(n_shaft - 1, 1)
            r = shaft_radius_near + alpha * (shaft_radius_far - shaft_radius_near)
            arrow_positions.append(shaft_pos)
            arrow_radii.append(float(r))
            arrow_colors.append(color)

    if not arrow_positions:
        arrow_positions = [np.zeros(3, dtype=np.float32)]
        arrow_radii = [0.001]
        arrow_colors = [np.zeros(3, dtype=np.float32)]

    pos_np = np.array(arrow_positions, dtype=np.float32)
    radii_np = np.array(arrow_radii, dtype=np.float32)
    colors_np = np.array(arrow_colors, dtype=np.float32)

    pos_wp = wp.array(pos_np, dtype=wp.vec3)
    radii_wp = wp.array(radii_np, dtype=wp.float32)
    colors_wp = wp.array(colors_np, dtype=wp.vec3)
    return pos_wp, radii_wp, colors_wp
