"""MoPhi OpenGL visualizer — wraps Newton's ``ViewerGL`` behind a stable
MoPhi-owned interface.

The class exposes exactly the rendering operations used by MoPhi demos:
opening a real-time OpenGL window, logging a Newton simulation state, logging
arbitrary point clouds (DEM particles, XLB streamlines, …), and recording
frames to an output buffer.

In the future, an ``OmniverseVisualizer`` will offer the same interface backed
by NVIDIA Omniverse instead of Newton's OpenGL renderer.  Demos can switch
backends by changing a single constructor call.
"""

from __future__ import annotations

import numpy as np

from ..utils.vis_utils import rotate_batch


class OpenGLVisualizer:
    """MoPhi visualizer backed by Newton's real-time OpenGL renderer.

    Wraps ``newton.viewer.ViewerGL`` and exposes a stable MoPhi interface so
    that demos are decoupled from Newton's internal viewer API.  All public
    methods forward directly to the underlying ViewerGL instance.

    Usage::

        visualizer = mophi.OpenGLVisualizer(newton_model)

        while visualizer.is_running():
            visualizer.set_camera(pos=..., pitch=..., yaw=...)
            visualizer.begin_frame(sim_time)
            visualizer.log_state(newton_state)
            visualizer.log_points("particles", positions, radii=radii, colors=colors)
            visualizer.end_frame()

        visualizer.close()
    """

    def __init__(self, model) -> None:
        """Open a real-time OpenGL window and register the Newton model.

        Args:
            model: A ``newton.Model`` (or compatible object) that describes the
                   articulated body geometry to be rendered.

        Raises:
            ImportError: if the ``newton`` package is not installed.
            Exception:   if the display cannot be initialized (e.g. headless
                         environment without a virtual framebuffer).
        """
        import newton  # imported here so that the rest of MoPhi works without newton

        self._viewer = newton.viewer.ViewerGL()
        self._viewer.set_model(model)

    # ── Window state ───────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        """Return ``True`` while the viewer window is open.

        Returns ``False`` once the user closes the window, allowing the
        simulation loop to terminate gracefully.
        """
        return self._viewer.is_running()

    # ── Camera ─────────────────────────────────────────────────────────────────

    def set_camera(self, pos, pitch: float, yaw: float) -> None:
        """Position and orient the viewer camera.

        Args:
            pos:   Camera position in world space (e.g. ``warp.vec3``).
            pitch: Vertical tilt in degrees (positive = look down).
            yaw:   Horizontal rotation in degrees.
        """
        self._viewer.set_camera(pos=pos, pitch=pitch, yaw=yaw)

    # ── Per-frame rendering ────────────────────────────────────────────────────

    def begin_frame(self, sim_time: float) -> None:
        """Begin a new visualization frame.

        Must be paired with :meth:`end_frame`.  Call this once per simulation
        step before any ``log_*`` calls.

        Args:
            sim_time: Current simulation time in seconds (used for HUD display
                      and time-stamping the frame buffer).
        """
        self._viewer.begin_frame(sim_time)

    def log_state(self, state) -> None:
        """Log a Newton simulation state for robot/articulated-body rendering.

        Args:
            state: A ``newton.State`` holding the current joint positions and
                   velocities.  Passed directly to the underlying ViewerGL.
        """
        self._viewer.log_state(state)

    def log_points(self, name: str, positions, *, radii=None, colors=None) -> None:
        """Log a named point cloud for rendering as spheres.

        Typical uses include DEM particle positions and XLB flow streamlines.

        Args:
            name:      Unique string identifier for this point set.  The viewer
                       uses it to distinguish separate clouds each frame.
            positions: ``warp.array`` of ``warp.vec3`` world-space positions.
            radii:     ``warp.array`` of ``float`` sphere radii [m].  When
                       ``None`` the viewer uses a default radius.
            colors:    ``warp.array`` of ``warp.vec3`` RGB colours in [0, 1].
                       When ``None`` the viewer uses a default colour.
        """
        self._viewer.log_points(name, positions, radii=radii, colors=colors)

    def log_arrows(self, name: str, starts, ends, colors, *, width: float = 0.01, hidden: bool = False) -> None:
        """Log a named batch of arrows for rendering.

        Typical uses include coordinate-axis indicators and vector-field
        overlays.  Arrow geometry persists between frames; call again only
        when the data changes.

        Forwards to ``ViewerGL.log_arrows`` when available (Newton ≥ 0.4).
        Older Newton builds that lack ``log_arrows`` fall back to
        ``ViewerGL.log_lines`` so the overlays remain visible, just without
        arrowheads.

        Args:
            name:   Unique string identifier for this arrow batch.
            starts: ``warp.array`` of ``warp.vec3`` tail positions.
            ends:   ``warp.array`` of ``warp.vec3`` head (tip) positions.
            colors: ``warp.array`` of ``warp.vec3`` RGB colours, or a single
                    ``(r, g, b)`` tuple/list applied to all arrows.
            width:  Reserved for future world-space width support.
            hidden: When ``True`` the batch is not rendered.
        """
        if hasattr(self._viewer, "log_arrows"):
            self._viewer.log_arrows(name, starts, ends, colors, width=width, hidden=hidden)
        else:
            # Graceful degradation for older Newton builds without log_arrows.
            self._viewer.log_lines(name, starts, ends, colors, width=width, hidden=hidden)

    def log_lines(self, name: str, starts, ends, colors, *, width: float = 0.01, hidden: bool = False) -> None:
        """Log a named batch of line segments for rendering.

        Typical uses include scale bars and grid overlays.  Line geometry
        persists between frames; call again only when the data changes.

        Args:
            name:   Unique string identifier for this line batch.
            starts: ``warp.array`` of ``warp.vec3`` start positions.
            ends:   ``warp.array`` of ``warp.vec3`` end positions.
            colors: ``warp.array`` of ``warp.vec3`` RGB colours, or a single
                    ``(r, g, b)`` tuple/list applied to all lines.
            width:  Reserved for future world-space width support.
            hidden: When ``True`` the batch is not rendered.
        """
        self._viewer.log_lines(name, starts, ends, colors, width=width, hidden=hidden)

    def end_frame(self) -> None:
        """Finalise the current frame and push it to the display."""
        self._viewer.end_frame()

    # ── Frame capture ──────────────────────────────────────────────────────────

    def get_frame(self):
        """Return the rendered frame as a device-resident Warp array.

        The returned array can be transferred to a NumPy array with
        ``.numpy()`` and appended to a video writer, e.g.::

            writer.append_data(visualizer.get_frame().numpy())

        Returns:
            A ``warp.array`` containing the RGBA pixel data of the last
            rendered frame.
        """
        return self._viewer.get_frame()

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the viewer window and release all associated resources."""
        self._viewer.close()

    # ── Primitive shapes ───────────────────────────────────────────────────────

    def log_clumps(
        self,
        name: str,
        centers,
        orientations,
        sphere_radii,
        sphere_offsets,
        *,
        colors=None,
    ) -> None:
        """Log a batch of DEME clumps for rendering as overlapping-sphere assemblies.

        Each clump is defined by a shared template (``sphere_radii``,
        ``sphere_offsets``) plus a per-clump centre and orientation.  The method
        expands the template into world-space sphere positions and forwards all
        of them to :meth:`log_points` in a single call, preserving optional
        per-clump colours.

        Args:
            name:           Unique string identifier for this clump batch.
            centers:        ``(N, 3)`` array of clump centre positions [m].
            orientations:   ``(N, 4)`` array of per-clump xyzw quaternions.
            sphere_radii:   ``(K,)`` array of component sphere radii [m].
            sphere_offsets: ``(K, 3)`` array of component sphere centre offsets
                            from the clump centre in the clump's local frame [m].
            colors:         ``(N, 3)`` per-clump RGB colours in ``[0, 1]``.
                            When ``None`` the viewer uses its default colour.
        """
        import warp as wp

        c_np = centers.numpy() if hasattr(centers, "numpy") else np.asarray(centers, dtype=np.float32)
        q_np = orientations.numpy() if hasattr(orientations, "numpy") else np.asarray(orientations, dtype=np.float32)
        r_np = sphere_radii.numpy() if hasattr(sphere_radii, "numpy") else np.asarray(sphere_radii, dtype=np.float32)
        off_np = (
            sphere_offsets.numpy() if hasattr(sphere_offsets, "numpy") else np.asarray(sphere_offsets, dtype=np.float32)
        )

        if c_np.ndim == 1:
            c_np = c_np.reshape(-1, 3)
        if off_np.ndim == 1:
            off_np = off_np.reshape(-1, 3)

        n_clumps = len(c_np)
        n_spheres = len(r_np)
        if n_clumps == 0 or n_spheres == 0:
            return

        # World positions of every component sphere: (N, K, 3) → (N*K, 3).
        rotated = rotate_batch(q_np, off_np)  # (N, K, 3)
        world_pos = c_np[:, np.newaxis, :] + rotated  # (N, K, 3)
        pos_flat = world_pos.reshape(n_clumps * n_spheres, 3).astype(np.float32)

        # Radii: tile K sphere radii across N clumps → (N*K,).
        radii_flat = np.tile(r_np, n_clumps).astype(np.float32)

        # Colours: repeat each clump colour K times → (N*K, 3).
        colors_wp = None
        if colors is not None:
            col_np = colors.numpy() if hasattr(colors, "numpy") else np.asarray(colors, dtype=np.float32)
            if col_np.ndim == 2 and len(col_np) == n_clumps:
                colors_flat = np.repeat(col_np, n_spheres, axis=0).astype(np.float32)
                colors_wp = wp.array(colors_flat, dtype=wp.vec3)

        self.log_points(
            name,
            wp.array(pos_flat, dtype=wp.vec3),
            radii=wp.array(radii_flat, dtype=wp.float32),
            colors=colors_wp,
        )

    # ── Note ───────────────────────────────────────────────────────────────────
    # rotate_batch (used by log_clumps) is a module-level function imported
    # from mophi.utils.vis_utils — it is not a class method.
