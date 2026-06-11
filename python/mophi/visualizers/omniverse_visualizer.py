"""MoPhi Omniverse/OpenUSD visualizer — writes co-simulation state to a USDC file.

This module exposes :class:`OmniverseVisualizer`, which implements the same
public interface as :class:`~mophi.OpenGLVisualizer`.  Demos can switch between
real-time OpenGL rendering and offline USD export by changing a single flag:

.. code-block:: python

    USE_OMNIVERSE_VISUALIZATION = False

    if USE_OMNIVERSE_VISUALIZATION:
        vis = mophi.OmniverseVisualizer(output_path="scene.usdc", fps=50.0)
        vis.set_mesh_shapes(mesh_descriptors)  # optional: static UsdGeom.Mesh geometry
    else:
        vis = mophi.OpenGLVisualizer(newton_model)

    while vis.is_running():
        vis.set_camera(pos=..., pitch=..., yaw=...)   # no-op for USD backend
        vis.begin_frame(sim_time)
        vis.log_state(newton_state)
        vis.log_points("particles", positions, radii=radii, colors=colors)
        vis.end_frame()

    vis.close()   # OpenGL closes the window; OmniverseVisualizer saves the USDC

**Solver data → USD mapping**

+------------+------------------------------------------+-----------------------------------+
| Solver     | Internal data                            | USD representation                |
+============+==========================================+===================================+
| Newton     | ``state.body_q[i]`` = 7 floats           | ``UsdGeom.Xform`` per body        |
|            | ``[px,py,pz, qx,qy,qz,qw]`` (xyzw quat) | ``TranslateOp`` + ``OrientOp``    |
|            | Visual mesh descriptors (optional, via   | ``UsdGeom.Mesh`` child prims      |
|            | :meth:`set_mesh_shapes`)                | under each body xform             |
|            | Cube shape descriptors (optional, via    | ``UsdGeom.Cube`` child prims      |
|            | :meth:`set_cube_shapes`)                  | under each body xform             |
+------------+------------------------------------------+-----------------------------------+
| Any solver | Named point cloud: positions + radii     | ``UsdGeom.PointInstancer`` per    |
|            | (e.g. DEM particles, XLB streamlines)   | cloud name; unit-sphere prototype |
|            |                                          | scaled by per-particle radii      |
+------------+------------------------------------------+-----------------------------------+
| DEME       | Clump centers + orientations + template  | ``UsdGeom.PointInstancer``; one   |
|            | sphere radii and offsets                 | sphere instance per component     |
|            |                                          | sphere of each clump              |
+------------+------------------------------------------+-----------------------------------+

The resulting ``.usdc`` file can be opened with:

* ``usdview scene.usdc``                    (CLI, ships with *usd-core*)
* NVIDIA Omniverse Create / Code            (drag-and-drop or File → Open)
* Blender                                   (File → Import → Universal Scene Description)

**Live Omniverse streaming**

The per-frame USD attribute writes are identical for live streaming.  Replace
``Usd.Stage.CreateNew(local_path)`` with the Omniverse live-session API::

    import omni.client
    omni.client.initialize()
    stage = Usd.Stage.CreateNew("omniverse://localhost/Projects/mophi/scene.usd")
"""

from __future__ import annotations

import numpy as np

from ..utils.vis_utils import rotate_batch


class OmniverseVisualizer:
    """MoPhi visualizer backed by OpenUSD — exports co-simulation state to a USDC file.

    Implements the same public interface as :class:`~mophi.OpenGLVisualizer` so
    that demos can switch backends via a single flag.

    :meth:`set_camera` is accepted but ignored (the USD scene has no interactive
    camera; any USD-compatible viewer applies its own camera on playback).
    :meth:`get_frame` always returns ``None`` (no rendered pixel buffer).
    :meth:`is_running` returns ``True`` until :meth:`close` is called, so the
    standard ``while vis.is_running():`` simulation loop works unchanged.
    :meth:`close` saves the USD stage to disk and prints an open-with hint.

    Usage::

        vis = mophi.OmniverseVisualizer(output_path="scene.usdc", fps=50.0)

        while vis.is_running():
            vis.begin_frame(sim_time)
            vis.log_state(newton_state)
            vis.log_points("particles", positions, radii=radii, colors=colors)
            vis.end_frame()

        vis.close()

    Args:
        output_path: Path for the output USD file.  ``.usdc`` = binary (compact);
                     ``.usda`` = ASCII (human-readable for inspection).
        fps:         Frames-per-second used to convert simulation time to USD
                     time codes (e.g. ``sim_time=0.02 s`` at ``fps=50`` → time code 1).
    """

    def __init__(self, output_path: str = "scene.usdc", fps: float = 50.0) -> None:
        self._output_path = output_path
        self._fps = fps
        self._closed = False
        self._current_time: float = 0.0
        self._stage = None
        self._robot_translate_ops: list = []
        self._robot_orient_ops: list = []
        # name → UsdGeom.PointInstancer
        self._point_instancers: dict = {}
        # name → last known point count (used to detect count changes between frames)
        self._instancer_sizes: dict = {}
        # visual mesh descriptors waiting to be baked into the stage
        self._pending_mesh_shapes: list = []

        # box shape descriptors waiting to be baked into the stage
        self._pending_cube_shapes: list = []

        try:
            from pxr import Gf, Usd, UsdGeom, Vt  # noqa: F401

            self._Gf = Gf
            self._Usd = Usd
            self._UsdGeom = UsdGeom
            self._Vt = Vt
            self._pxr_available = True
        except ImportError:
            self._pxr_available = False

    # ── Backend availability ───────────────────────────────────────────────────

    @property
    def pxr_available(self) -> bool:
        """``True`` when the *pxr* (OpenUSD) Python package is importable."""
        return self._pxr_available

    # ── Window state ───────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        """Return ``True`` until :meth:`close` has been called.

        Unlike :class:`~mophi.OpenGLVisualizer`, there is no interactive window
        to close; this method returns ``True`` for the entire lifetime of the
        object, allowing the same ``while vis.is_running():`` loop to work
        regardless of backend.
        """
        return not self._closed

    # ── Camera ─────────────────────────────────────────────────────────────────

    def set_camera(self, pos, pitch: float, yaw: float) -> None:
        """No-op for the USD backend — accepted for API compatibility.

        USD viewers apply their own camera on playback.  Camera parameters are
        intentionally ignored so that the same simulation loop works with both
        :class:`~mophi.OpenGLVisualizer` and :class:`OmniverseVisualizer`.

        Args:
            pos:   Camera position (ignored).
            pitch: Vertical tilt in degrees (ignored).
            yaw:   Horizontal rotation in degrees (ignored).
        """

    # ── Per-frame rendering ────────────────────────────────────────────────────

    def begin_frame(self, sim_time: float) -> None:
        """Begin a new visualization frame.

        Stores *sim_time* for use as the USD time code in subsequent
        :meth:`log_state` / :meth:`log_points` calls within this frame.
        Creates the USD stage on the first invocation.

        Args:
            sim_time: Current simulation time in seconds.
        """
        if not self._pxr_available:
            return
        self._current_time = sim_time
        if self._stage is None:
            self._create_stage()

    def log_state(self, state) -> None:
        """Record a Newton simulation state as USD body transforms.

        Extracts the ``body_q`` Warp array from *state*, converts each row
        ``[px, py, pz, qx, qy, qz, qw]`` (Newton xyzw quaternion) to a USD
        ``TranslateOp`` + ``OrientOp`` sample at the current simulation time.

        The ``/World/Robot`` xform hierarchy is created lazily on the first
        call, sized to match the body count found in *state*.

        Args:
            state: A ``newton.State`` with a ``body_q`` Warp array of shape
                   ``(body_count, 7)`` (or flat ``(body_count * 7,)``).
        """
        if not self._pxr_available or self._stage is None:
            return
        try:
            body_q_np = state.body_q.numpy()
        except Exception:
            return
        if body_q_np.ndim == 1:
            body_q_np = body_q_np.reshape(-1, 7)
        body_count = len(body_q_np)
        if body_count == 0:
            return
        # Lazily create the /World/Robot xform hierarchy on the first call.
        if not self._robot_translate_ops:
            self._ensure_robot_xforms(body_count)
        Gf = self._Gf
        time_code = self._Usd.TimeCode(self._current_time * self._fps)
        n = min(body_count, len(self._robot_translate_ops))
        for i in range(n):
            t = body_q_np[i]
            px, py, pz = float(t[0]), float(t[1]), float(t[2])
            qx, qy, qz, qw = float(t[3]), float(t[4]), float(t[5]), float(t[6])
            # Newton stores quaternions as xyzw; USD requires wxyz (w-first).
            self._robot_translate_ops[i].Set(Gf.Vec3d(px, py, pz), time_code)
            self._robot_orient_ops[i].Set(Gf.Quatf(qw, qx, qy, qz), time_code)

    def set_mesh_shapes(self, mesh_descriptors) -> None:
        """Register static ``UsdGeom.Mesh`` geometry for body xforms in the USD scene.

        Creates one ``UsdGeom.Mesh`` child prim per visual shape descriptor
        under the corresponding ``/World/Robot/Body_NNN`` xform.  Because the
        body xforms are animated every frame by :meth:`log_state`, the meshes
        automatically follow their parent body's motion without any per-frame
        work.

        This method may be called at any time before or after the simulation
        loop starts.  Descriptors registered before the first :meth:`log_state`
        call are stored and applied when :meth:`_ensure_robot_xforms` builds the
        body-xform hierarchy.  Descriptors registered after xforms already exist
        are applied immediately.

        Descriptors with ``mesh=None`` (non-mesh visual shapes) are silently
        skipped.

        Args:
            mesh_descriptors: List of dicts describing mesh shapes.
                Each dict must contain:

                * ``"body_idx"``   – ``int`` row index into ``body_q``
                * ``"mesh"``       – Newton mesh with ``.vertices`` (N×3) and
                  ``.indices`` (3×T flat); ``None`` for non-mesh shapes
                * ``"local_xform"``– ``wp.transform`` (shape pose in body frame)
                * ``"scale"``      – ``[sx, sy, sz]`` scale factors
                * ``"body_name"``  – short link label (used only for debugging)
        """
        self._pending_mesh_shapes = list(mesh_descriptors)
        # If xforms already exist (log_state was already called), apply immediately.
        # Otherwise they will be applied inside _ensure_robot_xforms().
        if self._robot_translate_ops:
            self._apply_mesh_shapes()

    def set_cube_shapes(self, cube_descriptors) -> None:
        """Register static ``UsdGeom.Cube`` geometry for body xforms in the USD scene.

        Creates one ``UsdGeom.Cube`` child prim per descriptor under the
        corresponding ``/World/Robot/Body_NNN`` xform.  Because the body xforms
        are animated every frame by :meth:`log_state`, the cube prims
        automatically follow their parent body's motion without any per-frame
        work.

        This method may be called at any time before or after the simulation
        loop starts.  Descriptors registered before the first :meth:`log_state`
        call are stored and applied when :meth:`_ensure_robot_xforms` builds the
        body-xform hierarchy.  Descriptors registered after xforms already exist
        are applied immediately.

        Args:
            cube_descriptors: List of dicts, one per cube shape.  Each dict must
                contain:

                * ``"body_idx"``    – ``int`` row index into ``body_q``
                * ``"half_extents"``– ``[hx, hy, hz]`` half-sizes in metres
                * ``"local_xform"`` – ``wp.transform`` shape pose in body frame
                  (optional; identity used if absent)
        """
        self._pending_cube_shapes = list(cube_descriptors)
        # If xforms already exist (log_state was already called), apply immediately.
        # Otherwise they will be applied inside _ensure_robot_xforms().
        if self._robot_translate_ops:
            self._apply_cube_shapes()

    def log_points(self, name: str, positions, *, radii=None, colors=None) -> None:
        """Record a named point cloud as a USD ``UsdGeom.PointInstancer``.

        The first call for *name* creates a new ``UsdGeom.PointInstancer`` prim
        at ``/World/Points/<name>`` with a unit-sphere prototype and optional
        per-instance radii (as uniform scale).  Subsequent calls write a new
        time sample for positions.  When the point count changes between frames
        the instancer attributes are updated accordingly.

        The display colour of the prototype sphere is chosen by heuristic:
        names containing ``"stream"`` or ``"xlb"`` are rendered blue; all other
        clouds use orange (DEM particles).

        Args:
            name:      Unique string identifier for this point cloud.
            positions: Warp / NumPy array of shape ``(N, 3)`` with world-space
                       positions ``[x, y, z]`` for each point.
            radii:     Warp / NumPy array of ``float`` per-point radii [m].
                       When ``None`` a default radius of 0.02 m is used.
            colors:    Warp / NumPy array of shape ``(N, 3)`` with per-point
                       RGB colours in ``[0, 1]``.  Stored as a
                       ``primvars:displayColor`` primvar on the instancer
                       (``interpolation = "uniform"``, one colour per instance).
                       When ``None`` the prototype's default colour is used.
        """
        if not self._pxr_available or self._stage is None:
            return

        # ── Convert inputs to numpy ───────────────────────────────────────────
        try:
            pos_np = positions.numpy() if hasattr(positions, "numpy") else np.asarray(positions, dtype=np.float32)
        except Exception:
            return
        if pos_np.ndim == 1:
            pos_np = pos_np.reshape(-1, 3)
        n = len(pos_np)
        if n == 0:
            return

        Gf = self._Gf
        Vt = self._Vt
        UsdGeom = self._UsdGeom
        time_code = self._Usd.TimeCode(self._current_time * self._fps)

        # ── Create PointInstancer lazily on first call ─────────────────────────
        if name not in self._point_instancers:
            safe_name = name.replace(" ", "_").replace("-", "_")
            prim_path = f"/World/Points/{safe_name}"
            UsdGeom.Xform.Define(self._stage, "/World/Points")
            instancer = UsdGeom.PointInstancer.Define(self._stage, prim_path)
            proto_scope = prim_path + "/Prototypes"
            self._stage.DefinePrim(proto_scope, "Scope")
            proto = UsdGeom.Sphere.Define(self._stage, proto_scope + "/UnitSphere")
            proto.GetRadiusAttr().Set(1.0)
            # Colour heuristic: streamlines → blue; others → orange.
            name_lower = name.lower()
            if "stream" in name_lower or "xlb" in name_lower:
                proto_color = Gf.Vec3f(0.20, 0.55, 0.85)
            else:
                proto_color = Gf.Vec3f(0.80, 0.40, 0.10)
            proto.GetDisplayColorAttr().Set(Vt.Vec3fArray([proto_color]))
            instancer.CreatePrototypesRel().AddTarget(proto.GetPath())
            self._point_instancers[name] = instancer
            self._instancer_sizes[name] = 0

        instancer = self._point_instancers[name]
        prev_n = self._instancer_sizes[name]

        # ── Update structural attributes when the point count changes ──────────
        if n != prev_n:
            instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0] * n), time_code)
            instancer.GetOrientationsAttr().Set(Vt.QuathArray([Gf.Quath(1.0, 0.0, 0.0, 0.0)] * n), time_code)
            self._instancer_sizes[name] = n

        # ── Radii → per-instance uniform scale ────────────────────────────────
        _DEFAULT_RADIUS = 0.02
        try:
            if radii is not None:
                r_np = radii.numpy() if hasattr(radii, "numpy") else np.asarray(radii, dtype=np.float32)
                if r_np.ndim == 0:
                    r_np = np.full(n, float(r_np), dtype=np.float32)
            else:
                r_np = np.full(n, _DEFAULT_RADIUS, dtype=np.float32)
            instancer.GetScalesAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(float(r), float(r), float(r)) for r in r_np]),
                time_code,
            )
        except Exception:
            instancer.GetScalesAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(_DEFAULT_RADIUS, _DEFAULT_RADIUS, _DEFAULT_RADIUS)] * n),
                time_code,
            )

        # ── Positions (time-varying) ───────────────────────────────────────────
        instancer.GetPositionsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pos_np]),
            time_code,
        )

        # ── Per-instance colours via primvars:displayColor ─────────────────────
        if colors is not None:
            try:
                from pxr import Sdf, UsdGeom as _UG

                c_np = colors.numpy() if hasattr(colors, "numpy") else np.asarray(colors, dtype=np.float32)
                if c_np.ndim == 2 and len(c_np) == n:
                    pv_api = _UG.PrimvarsAPI(instancer.GetPrim())
                    pv = pv_api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray)
                    pv.SetInterpolation("uniform")
                    pv.Set(
                        Vt.Vec3fArray([Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in c_np]),
                        time_code,
                    )
            except Exception:
                pass

    def log_arrows(self, name: str, starts, ends, colors, *, width: float = 0.01, hidden: bool = False) -> None:
        """Record arrows as colored line segments in the USD scene."""
        self.log_lines(name, starts, ends, colors, width=width, hidden=hidden)

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
        """Record a batch of DEME clumps as USD point instances (overlapping spheres).

        Each clump is expanded into its component spheres (sphere positions
        rotated by the clump orientation and translated to the clump centre).
        The result is forwarded to :meth:`log_points` as a single flat cloud,
        giving each clump ``K`` sphere instances in the USD scene.

        Args:
            name:           Unique string identifier for this clump batch.
            centers:        ``(N, 3)`` array of clump centre positions [m].
            orientations:   ``(N, 4)`` array of per-clump xyzw quaternions.
            sphere_radii:   ``(K,)`` array of component sphere radii [m].
            sphere_offsets: ``(K, 3)`` array of component sphere centre offsets
                            from the clump centre in the clump's local frame [m].
            colors:         ``(N, 3)`` per-clump RGB colours in ``[0, 1]``.
                            When ``None`` the prototype's default colour is used.
        """
        if not self._pxr_available or self._stage is None:
            return

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
        colors_flat = None
        if colors is not None:
            col_np = colors.numpy() if hasattr(colors, "numpy") else np.asarray(colors, dtype=np.float32)
            if col_np.ndim == 2 and len(col_np) == n_clumps:
                colors_flat = np.repeat(col_np, n_spheres, axis=0).astype(np.float32)

        self.log_points(name, pos_flat, radii=radii_flat, colors=colors_flat)

    def log_lines(self, name: str, starts, ends, colors, *, width: float = 0.01, hidden: bool = False) -> None:
        """Record line segments as sampled sphere instances in the USD scene."""
        if hidden or not self._pxr_available:
            return
        if self._stage is None:
            self._create_stage()

        starts_np = starts.numpy() if hasattr(starts, "numpy") else np.asarray(starts, dtype=np.float32)
        ends_np = ends.numpy() if hasattr(ends, "numpy") else np.asarray(ends, dtype=np.float32)
        colors_np = colors.numpy() if hasattr(colors, "numpy") else np.asarray(colors, dtype=np.float32)
        starts_np = np.asarray(starts_np, dtype=np.float32).reshape(-1, 3)
        ends_np = np.asarray(ends_np, dtype=np.float32).reshape(-1, 3)
        colors_np = np.asarray(colors_np, dtype=np.float32).reshape(-1, 3)
        if len(starts_np) == 0:
            return
        if len(colors_np) == 1:
            colors_np = np.repeat(colors_np, len(starts_np), axis=0)

        samples_per_line = 24
        interpolation = np.linspace(0.0, 1.0, samples_per_line, dtype=np.float32)
        positions = (
            starts_np[:, np.newaxis, :] * (1.0 - interpolation[np.newaxis, :, np.newaxis])
            + ends_np[:, np.newaxis, :] * interpolation[np.newaxis, :, np.newaxis]
        ).reshape(-1, 3)
        radii = np.full(len(positions), max(float(width), 0.005), dtype=np.float32)
        sampled_colors = np.repeat(colors_np, samples_per_line, axis=0)
        self.log_points(name, positions, radii=radii, colors=sampled_colors)

    def end_frame(self) -> None:
        """No-op for the USD backend — accepted for API compatibility.

        All time samples for the current frame were already written inline
        during :meth:`log_state` and :meth:`log_points` calls.
        """

    # ── Frame capture ──────────────────────────────────────────────────────────

    def get_frame(self):
        """Return ``None`` — the USD backend has no rendered pixel buffer.

        Accepted for API compatibility with :class:`~mophi.OpenGLVisualizer`.
        Callers should guard movie-writer code with a backend check::

            if not USE_OMNIVERSE and vis.get_frame() is not None:
                writer.append_data(vis.get_frame().numpy())
        """
        return None

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Save the USD stage to *output_path* and release resources.

        Calling this more than once is safe (subsequent calls are no-ops).
        """
        if self._closed:
            return
        self._closed = True
        if self._pxr_available and self._stage is not None:
            self._stage.GetRootLayer().Save()
            print(
                f"[USD] Simulation scene saved to '{self._output_path}'.\n"
                "      Open with:  usdview scene.usdc\n"
                "                  (or drag into NVIDIA Omniverse Create / Code)\n"
                "      To install usdview:  pip install usd-core"
            )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _create_stage(self) -> None:
        """Create the USD stage and define static scene structure."""
        Usd = self._Usd
        UsdGeom = self._UsdGeom
        Gf = self._Gf
        Vt = self._Vt

        self._stage = Usd.Stage.CreateNew(self._output_path)
        self._stage.SetMetadata("metersPerUnit", 1.0)  # SI: metres
        self._stage.SetStartTimeCode(0.0)
        self._stage.SetFramesPerSecond(self._fps)
        UsdGeom.SetStageUpAxis(self._stage, UsdGeom.Tokens.z)  # z-up matches Newton

        # ── /World ────────────────────────────────────────────────────────────
        UsdGeom.Xform.Define(self._stage, "/World")

        # ── /World/Ground (flat quad at z = 0) ────────────────────────────────
        ground = UsdGeom.Mesh.Define(self._stage, "/World/Ground")
        h = 10.0
        ground.GetPointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(-h, -h, 0.0), Gf.Vec3f(h, -h, 0.0), Gf.Vec3f(h, h, 0.0), Gf.Vec3f(-h, h, 0.0)])
        )
        ground.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
        ground.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
        ground.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.70, 0.70, 0.70)]))

        # /World/Robot xforms are created lazily by _ensure_robot_xforms() the
        # first time log_state() is called.
        # /World/Points prim is created lazily by log_points() for each named cloud.

    def _ensure_robot_xforms(self, body_count: int) -> None:
        """Create the ``/World/Robot`` xform hierarchy for *body_count* bodies.

        Called lazily by :meth:`log_state` on its first invocation so that the
        body count is always derived directly from the Newton state array rather
        than from a separate :meth:`set_model` call.
        """
        UsdGeom = self._UsdGeom
        UsdGeom.Xform.Define(self._stage, "/World/Robot")
        for i in range(body_count):
            xf = UsdGeom.Xform.Define(self._stage, f"/World/Robot/Body_{i:03d}")
            self._robot_translate_ops.append(xf.AddTranslateOp())
            self._robot_orient_ops.append(xf.AddOrientOp())
        # Apply any mesh geometry that was registered via set_mesh_shapes()
        # before the first log_state() call.
        if self._pending_mesh_shapes:
            self._apply_mesh_shapes()
        # Apply any cube shapes that were registered via set_cube_shapes()
        # before the first log_state() call.
        if self._pending_cube_shapes:
            self._apply_cube_shapes()

    def _apply_mesh_shapes(self) -> None:
        """Bake static ``UsdGeom.Mesh`` prims from ``_pending_mesh_shapes`` into the stage.

        One child mesh prim is created per visual descriptor under its
        corresponding ``/World/Robot/Body_NNN`` xform.  The local shape
        transform (translation, orientation, scale) is encoded as xformOps
        directly on the mesh prim so the body xform remains a pure animated
        translate+orient.

        Vertices and triangle indices are set as static (time-independent)
        attributes; only the parent body xforms carry time-sampled data.

        After baking, ``_pending_mesh_shapes`` is cleared so meshes are
        not re-created if :meth:`set_mesh_shapes` is accidentally called twice.
        """
        UsdGeom = self._UsdGeom
        Gf = self._Gf
        Vt = self._Vt

        # Track how many Mesh prims have already been added under each body so
        # that each gets a unique, stable prim name (Mesh_00, Mesh_01, …).
        mesh_count_per_body: dict = {}

        for d in self._pending_mesh_shapes:
            mesh = d.get("mesh")
            if mesh is None:
                continue
            b_idx = d["body_idx"]
            k = mesh_count_per_body.get(b_idx, 0)
            mesh_count_per_body[b_idx] = k + 1

            body_path = f"/World/Robot/Body_{b_idx:03d}"
            mesh_path = f"{body_path}/Mesh_{k:02d}"

            usd_mesh = UsdGeom.Mesh.Define(self._stage, mesh_path)

            # ── Geometry ──────────────────────────────────────────────────────
            verts = mesh.vertices
            if hasattr(verts, "numpy"):
                verts = verts.numpy()
            verts = np.asarray(verts, dtype=np.float32)
            if verts.ndim == 1:
                verts = verts.reshape(-1, 3)
            pts = Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
            usd_mesh.GetPointsAttr().Set(pts)

            indices = mesh.indices
            if hasattr(indices, "numpy"):
                indices = indices.numpy()
            indices = np.asarray(indices, dtype=np.int32).flatten()
            n_tris = len(indices) // 3
            usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * n_tris))
            usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices.tolist()))

            # Enable smooth-shading normals so the mesh looks solid in viewers.
            usd_mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

            # ── Local transform: T * R * S (translate outermost, scale innermost)
            # xformOpOrder is applied left-to-right as the composed product, so
            # [T, R, S] yields T * R * S * P for each point P, which is the
            # standard mesh-in-link-frame convention from URDF <visual> tags.
            local_xform = d["local_xform"]
            scale = d["scale"]
            p = local_xform.p  # vec3 translation
            q = local_xform.q  # quaternion in Newton xyzw convention
            xformable = UsdGeom.Xformable(usd_mesh.GetPrim())
            xformable.AddTranslateOp().Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            # Newton xyzw → USD w-first Quatf.
            xformable.AddOrientOp().Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
            xformable.AddScaleOp().Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))

            # ── Display colour (muted steel-blue distinguishes robot from ground)
            usd_mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.60, 0.63, 0.70)]))

        self._pending_mesh_shapes = []

    def _apply_cube_shapes(self) -> None:
        """Bake static ``UsdGeom.Cube`` prims from ``_pending_cube_shapes`` into the stage.

        One child cube prim is created per cube descriptor under its corresponding
        ``/World/Robot/Body_NNN`` xform.  The cube's size is set to 1 and then
        scaled by ``[2*hx, 2*hy, 2*hz]`` to match the box half-extents.  The
        optional local transform is encoded as xformOps so the body xform remains
        a pure animated translate+orient.

        After baking, ``_pending_cube_shapes`` is cleared so cubes are not
        re-created if :meth:`set_cube_shapes` is accidentally called twice.
        """
        UsdGeom = self._UsdGeom
        Gf = self._Gf
        Vt = self._Vt

        # Track how many Cube prims have already been added under each body.
        cube_count_per_body: dict = {}

        for d in self._pending_cube_shapes:
            b_idx = d["body_idx"]
            half_extents = d["half_extents"]
            k = cube_count_per_body.get(b_idx, 0)
            cube_count_per_body[b_idx] = k + 1

            body_path = f"/World/Robot/Body_{b_idx:03d}"
            cube_path = f"{body_path}/Box_{k:02d}"

            usd_cube = UsdGeom.Cube.Define(self._stage, cube_path)
            # Size=1 so the visual dimensions are fully controlled by the scale op below.
            usd_cube.GetSizeAttr().Set(1.0)

            # ── Local transform: T * R * S (translate outermost, scale innermost)
            local_xform = d.get("local_xform")
            xformable = UsdGeom.Xformable(usd_cube.GetPrim())
            if local_xform is not None:
                p = local_xform.p
                q = local_xform.q
                xformable.AddTranslateOp().Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                # Newton xyzw → USD w-first Quatf.
                xformable.AddOrientOp().Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
            # Scale a unit cube to match the box full extents (2 × half-extents).
            hx, hy, hz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
            xformable.AddScaleOp().Set(Gf.Vec3f(2.0 * hx, 2.0 * hy, 2.0 * hz))

            # ── Display colour (warm orange distinguishes boxes from the robot)
            usd_cube.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.85, 0.55, 0.20)]))

        self._pending_cube_shapes = []
