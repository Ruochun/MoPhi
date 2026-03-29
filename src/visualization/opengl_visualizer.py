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


class OpenGLVisualizer:
    """MoPhi visualizer backed by Newton's real-time OpenGL renderer.

    Wraps ``newton.viewer.ViewerGL`` and exposes a stable MoPhi interface so
    that demos are decoupled from Newton's internal viewer API.  All public
    methods forward directly to the underlying ViewerGL instance.

    Usage::

        visualizer = mophi.OpenGLVisualizer()
        visualizer.set_model(newton_model)

        while visualizer.is_running():
            visualizer.set_camera(pos=..., pitch=..., yaw=...)
            visualizer.begin_frame(sim_time)
            visualizer.log_state(newton_state)
            visualizer.log_points("particles", positions, radii=radii, colors=colors)
            visualizer.end_frame()

        visualizer.close()
    """

    def __init__(self) -> None:
        """Open a real-time OpenGL window.

        Raises:
            ImportError: if the ``newton`` package is not installed.
            Exception:   if the display cannot be initialized (e.g. headless
                         environment without a virtual framebuffer).
        """
        import newton  # imported here so that the rest of MoPhi works without newton

        self._viewer = newton.viewer.ViewerGL()

    # ── Model ──────────────────────────────────────────────────────────────────

    def set_model(self, model) -> None:
        """Register the Newton model whose geometry should be rendered.

        Args:
            model: A ``newton.Model`` (or compatible object) that describes the
                   articulated body geometry of the scene.
        """
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
