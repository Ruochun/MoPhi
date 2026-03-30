"""MoPhi Omniverse/OpenUSD visualizer — writes co-simulation state to a USDC file.

This module exposes :class:`OmniverseVisualizer`, which implements the same
public interface as :class:`~mophi.OpenGLVisualizer`.  Demos can switch between
real-time OpenGL rendering and offline USD export by changing a single flag:

.. code-block:: python

    USE_OMNIVERSE = False

    if USE_OMNIVERSE_VISUALIZATION:
        vis = mophi.OmniverseVisualizer(output_path="scene.usdc", fps=50.0)
    else:
        vis = mophi.OpenGLVisualizer()

    vis.set_model(newton_model)

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
+------------+------------------------------------------+-----------------------------------+
| Any solver | Named point cloud: positions + radii     | ``UsdGeom.PointInstancer`` per    |
|            | (e.g. DEM particles, XLB streamlines)   | cloud name; unit-sphere prototype |
|            |                                          | scaled by per-particle radii      |
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
        vis.set_model(newton_model)

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
        self._body_count: int = 0
        self._robot_translate_ops: list = []
        self._robot_orient_ops: list = []
        # name → UsdGeom.PointInstancer
        self._point_instancers: dict = {}
        # name → last known point count (used to detect count changes between frames)
        self._instancer_sizes: dict = {}

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

    # ── Model ──────────────────────────────────────────────────────────────────

    def set_model(self, model) -> None:
        """Register the Newton model whose body transforms will be recorded.

        Stores the body count so that the USD scene can be set up with the
        correct number of ``UsdGeom.Xform`` prims the first time
        :meth:`begin_frame` is called.

        Args:
            model: A ``newton.Model`` (or compatible object) with a
                   ``body_count`` attribute.
        """
        if not self._pxr_available:
            return
        self._body_count = int(getattr(model, "body_count", 0))

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

        Args:
            state: A ``newton.State`` with a ``body_q`` Warp array of shape
                   ``(body_count, 7)`` (or flat ``(body_count * 7,)``).
        """
        if not self._pxr_available or self._stage is None:
            return
        if not self._robot_translate_ops:
            return
        try:
            body_q_np = state.body_q.numpy()
        except Exception:
            return
        if body_q_np.ndim == 1:
            body_q_np = body_q_np.reshape(-1, 7)
        Gf = self._Gf
        time_code = self._Usd.TimeCode(self._current_time * self._fps)
        n = min(len(body_q_np), len(self._robot_translate_ops))
        for i in range(n):
            t = body_q_np[i]
            px, py, pz = float(t[0]), float(t[1]), float(t[2])
            qx, qy, qz, qw = float(t[3]), float(t[4]), float(t[5]), float(t[6])
            # Newton stores quaternions as xyzw; USD requires wxyz (w-first).
            self._robot_translate_ops[i].Set(Gf.Vec3d(px, py, pz), time_code)
            self._robot_orient_ops[i].Set(Gf.Quatf(qw, qx, qy, qz), time_code)

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

        # ── /World/Robot : one Xform per Newton body ──────────────────────────
        # Newton body_q row : [px, py, pz,  qx, qy, qz, qw]  (xyzw quaternion)
        # TranslateOp       : Vec3d(px, py, pz)
        # OrientOp          : Quatf(qw, qx, qy, qz)  ← USD convention: w-first
        if self._body_count > 0:
            UsdGeom.Xform.Define(self._stage, "/World/Robot")
            for i in range(self._body_count):
                xf = UsdGeom.Xform.Define(self._stage, f"/World/Robot/Body_{i:03d}")
                self._robot_translate_ops.append(xf.AddTranslateOp())
                self._robot_orient_ops.append(xf.AddOrientOp())

        # /World/Points prim is created lazily by log_points() for each named cloud.
