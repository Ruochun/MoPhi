# MoPhi Visualization

This directory contains MoPhi's backend-agnostic visualization layer.
It is a pure-Python module — no C++ compilation required.

---

## Design philosophy

### 1. Visualizers are solver-agnostic

Neither `OpenGLVisualizer` nor `OmniverseVisualizer` has a hard dependency on
any particular physics solver.  The public API is expressed in terms of
*generic rendering operations* (`begin_frame`, `log_state`, `log_points`,
`end_frame`) rather than in terms of solver-specific types.

This matters for two reasons:

- The current OpenGL backend happens to come from Newton, but it may be
  replaced by a renderer that has nothing to do with Newton in the future.
- The Omniverse/USD backend has no inherent connection to any physics solver
  at all — it simply writes USD time samples for whatever data is handed to
  it.

### 2. Solver-specific data enters only through `log_state`

When a demo needs to render Newton rigid-body state, it calls
`log_state(newton_state)`.  The visualizer inspects the state object and
extracts the data it understands (e.g. `state.body_q` for body transforms).

`log_state` is the *only* method that touches solver-specific data.  All
other methods (`log_points`, `begin_frame`, …) are fully solver-agnostic.

### 3. The Newton model is an init-time argument for OpenGLVisualizer only

Newton's `ViewerGL` requires a `newton.Model` object at construction time in
order to know the articulated-body geometry it will render.  Because this is
a requirement of the *current* OpenGL backend implementation (not of
visualization in general), the model is supplied as a **constructor argument
to `OpenGLVisualizer`** — it is not part of a generic `set_model` interface.

If the OpenGL backend is replaced in the future, this argument is simply
removed from the constructor; nothing else in the API changes.

### 4. `OmniverseVisualizer` needs no model reference

The USD backend infers the body count lazily from the first `log_state` call
(by inspecting `state.body_q.shape`).  It has no reason to hold a reference
to a `newton.Model` object.  The `/World/Robot` xform hierarchy is created on
the first `log_state` call via the private helper `_ensure_robot_xforms`.

### 5. Backends share a stable public interface

Both visualizers expose exactly the same set of public methods so that demos
can switch backends by changing a single constructor call:

```python
USE_OMNIVERSE_VISUALIZATION = False

if USE_OMNIVERSE_VISUALIZATION:
    vis = mophi.OmniverseVisualizer(output_path="scene.usdc", fps=50.0)
else:
    vis = mophi.OpenGLVisualizer(newton_model)

while vis.is_running():
    vis.set_camera(pos=..., pitch=..., yaw=...)
    vis.begin_frame(sim_time)
    vis.log_state(newton_state)
    vis.log_points("particles", positions, radii=radii, colors=colors)
    vis.end_frame()

vis.close()
```

Methods that are meaningless for a particular backend are accepted silently
as no-ops (e.g. `set_camera` on the USD backend).

---

## Public API contract

Every visualizer backend must implement the following methods:

| Method | Signature | Notes |
|--------|-----------|-------|
| `is_running` | `() → bool` | `False` when the user closes the window (OpenGL) or `close()` is called (USD) |
| `set_camera` | `(pos, pitch, yaw)` | No-op is acceptable if the backend has no interactive camera |
| `begin_frame` | `(sim_time: float)` | Called once per simulation step before any `log_*` calls |
| `log_state` | `(state)` | Render/record the rigid-body state from a physics solver |
| `log_points` | `(name, positions, *, radii, colors)` | Render/record a named point cloud (spheres) |
| `log_clumps` | `(name, centers, orientations, sphere_radii, sphere_offsets, *, colors)` | Render/record DEME clumps as overlapping-sphere assemblies; expands template to world-space spheres and delegates to `log_points` |
| `log_arrows` | `(name, starts, ends, colors, *, width, hidden)` | Render/record named arrows (coordinate axes, vector fields); no-op for USD backend |
| `log_lines` | `(name, starts, ends, colors, *, width, hidden)` | Render/record named line segments (scale bars, grids); no-op for USD backend |
| `end_frame` | `()` | Finalize the current frame |
| `get_frame` | `() → array | None` | Return the rendered frame (or `None` if unsupported) |
| `close` | `()` | Release all resources / save the output file |

---

## Backends

### `OpenGLVisualizer`

Wraps Newton's `newton.viewer.ViewerGL` behind the stable MoPhi interface.

**Dependencies**: `newton` Python package (imported lazily inside `__init__`
so that the rest of MoPhi works without it).

If you are running under WSL and the OpenGL window does not appear, confirm that
WSLg GUI support is installed and working before debugging the MoPhi viewer path.

**Constructor**: `OpenGLVisualizer(model)` — the Newton model is required at
construction time because `ViewerGL.set_model()` must be called before the
first frame is rendered.

**`log_state`**: Forwards the Newton state object directly to
`ViewerGL.log_state()`.

### `OmniverseVisualizer`

Writes co-simulation state to a USDC (binary USD) or USDA (ASCII USD) file
using the `pxr` (OpenUSD) library.

**Dependencies**: `pxr` Python package (`pip install usd-core`). The
visualizer degrades gracefully when `pxr` is not installed: all methods
become no-ops and `pxr_available` returns `False`.

**Constructor**: `OmniverseVisualizer(output_path, fps)` — no physics solver
reference required.

**`log_state`**: Extracts `state.body_q` (shape `(N, 7)`, columns
`[px, py, pz, qx, qy, qz, qw]`) and writes a `TranslateOp` + `OrientOp`
time sample for each body.  The `/World/Robot` USD prim hierarchy is created
lazily on the first call, sized to match the actual body count.

**`log_points`**: Creates a `UsdGeom.PointInstancer` prim per named cloud
(lazily on first call) and writes position + scale time samples each frame.

**`log_clumps`**: Expands each clump into its component spheres (offset rotated
by the clump orientation, translated to the clump centre) and forwards the flat
sphere list to `log_points`.  The resulting USD scene contains one
`UsdGeom.PointInstancer` prim with `N × K` sphere instances (N clumps, K
component spheres each).

**Solver data → USD mapping**

| Solver | Internal data | USD representation |
|--------|---------------|--------------------|
| Newton | `state.body_q[i]` = `[px,py,pz, qx,qy,qz,qw]` (xyzw quat) | `UsdGeom.Xform` per body; `TranslateOp` + `OrientOp` |
| Any solver | Named point cloud: positions + radii | `UsdGeom.PointInstancer` per cloud name; unit-sphere prototype scaled uniformly by per-particle radius |
| DEME | Clump centers + orientations + sphere template | `UsdGeom.PointInstancer`; one sphere instance per component sphere of each clump |

The resulting `.usdc` file can be opened with:

- `usdview scene.usdc` (CLI, ships with *usd-core*)
- NVIDIA Omniverse Create / Code (drag-and-drop or File → Open)
- Blender (File → Import → Universal Scene Description)

---

## Adding a new visualizer backend

1. Create `src/visualization/<name>_visualizer.py` (and mirror it under
   `python/mophi/visualizers/<name>_visualizer.py`).
2. Implement all methods listed in the **Public API contract** table above.
3. Add a `try/except ImportError` import in `python/mophi/__init__.py`.
4. Update this README to document the new backend.
