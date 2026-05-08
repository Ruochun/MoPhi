"""demo/feris_newton/demo_feris_newton.py

Demonstrates the FERIS + Newton co-simulation via MoPhi's FERISNewtonCoupler.

Both solvers are managed entirely through the coupler:
  • FERISNewtonCoupler.initialize() accepts the Newton model and solver directly.
  • FERISNewtonCoupler.step() advances FERIS, exchanges coupling data (FERIS node
    positions → Newton geometry; Newton forces → FERIS loads), and advances Newton —
    all in one call.
  • FERISNewtonCoupler.finalize() tears down both solvers.

The demo script only builds the Newton model (describing the rigid-body physics scene)
and then delegates all time-stepping to the coupler.

Prerequisites
-------------
  • Build MoPhi with -DMOPHI_BUILD_FERIS_NEWTON=ON (fetches and builds FERIS,
    compiles FERISNewtonCoupler, and builds the mophi_core Python extension).
  • pip install newton  (or configure with -DMOPHI_FETCH_NEWTON=ON)

Running
-------
From the MoPhi build tree (after ``cmake --build``):

    python demo/feris_newton/demo_feris_newton.py

Or from the repository root after installing the mophi package:

    python -m demo.feris_newton.demo_feris_newton
"""

import sys

# ─── 1. Import MoPhi ─────────────────────────────────────────────────────────
try:
    import mophi
except ImportError as exc:
    sys.exit(
        "ERROR: Could not import the 'mophi' package.\n"
        "       Build MoPhi with -DMOPHI_BUILD_FERIS_NEWTON=ON and ensure that\n"
        "       the build directory (python/) is on PYTHONPATH.\n"
        f"       ({exc})"
    )

if not hasattr(mophi, "FERISNewtonCoupler"):
    sys.exit(
        "ERROR: mophi.FERISNewtonCoupler is not available.\n"
        "       Re-build MoPhi with -DMOPHI_BUILD_FERIS_NEWTON=ON."
    )

# ─── 2. Import Newton ─────────────────────────────────────────────────────────
try:
    import newton
    import warp as wp

    _newton_available = True
except ImportError:
    _newton_available = False
    print(
        "WARNING: Newton (or warp) is not installed.  "
        "The FERIS side will still be demonstrated;\n"
        "         install Newton with:  pip install newton"
    )

# ─── 3. Build the Newton scene (if Newton is available) ───────────────────────
print("=== MoPhi FERIS + Newton co-simulation demo ===\n")

newton_model = None
newton_solver = None

if _newton_available:
    print("[Newton] Setting up a simple double-pendulum simulation ...")

    wp.init()

    builder = newton.ModelBuilder()

    # Two-link pendulum (adapted from newton.examples.basic_pendulum).
    hx, hy, hz = 1.0, 0.1, 0.1

    link_0 = builder.add_link()
    builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)

    link_1 = builder.add_link()
    builder.add_shape_box(link_1, hx=hx, hy=hy, hz=hz)

    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
    j0 = builder.add_joint_revolute(
        parent=-1,
        child=link_0,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
        child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
    )
    j1 = builder.add_joint_revolute(
        parent=link_0,
        child=link_1,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
    )
    builder.add_articulation([j0, j1], label="pendulum")
    builder.add_ground_plane()

    newton_model = builder.finalize()
    newton_solver = newton.solvers.SolverXPBD(newton_model)

    print(f"[Newton] Double-pendulum model built ({newton_model.body_count} bodies).\n")

# ─── 4. Initialize the coupler (passes Newton objects in) ────────────────────
coupler = mophi.FERISNewtonCoupler()

print("[Coupler] Initializing FERISNewtonCoupler ...")
coupler.initialize(
    newton_model=newton_model,
    newton_solver=newton_solver,
    sim_dt=1.0 / 1000.0,
)
print("[Coupler] FERISNewtonCoupler initialized.\n")

# ─── 5. Co-simulation loop ────────────────────────────────────────────────────
num_steps = 5
print(f"Running {num_steps} co-simulation step(s) ...\n")

for i in range(num_steps):
    # A single coupler.step() call advances FERIS, exchanges coupling data
    # (FERIS node positions → Newton; Newton forces → FERIS), and advances Newton.
    coupler.step()
    print(f"  step {i + 1}/{num_steps}  [Coupler] advanced co-simulation step")

print()

# ─── 6. Finalize ──────────────────────────────────────────────────────────────
print("[Coupler] Finalizing FERISNewtonCoupler ...")
coupler.finalize()
print("[Coupler] FERISNewtonCoupler finalized.\n")

print("Demo completed successfully.")
if _newton_available:
    print(f"  Newton version : {newton.__version__}")
    print(f"  Warp  version  : {wp.__version__}")
