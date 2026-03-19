"""demo/tlfea_newton/demo_tlfea_newton.py

Demonstrates TLFEA's FEA solver running alongside a Newton physics session.

TLFEA is accessed through MoPhi's Python bindings (TLFEANewtonCoupler).
Newton is a pure Python GPU-accelerated physics engine (built on NVIDIA Warp)
and is imported directly — no C++ linking is needed.

Prerequisites
-------------
  • Build MoPhi with -DMOPHI_BUILD_TLFEA_NEWTON=ON (fetches and builds TLFEA,
    compiles TLFEANewtonCoupler, and builds the mophi_core Python extension).
  • pip install newton  (or configure with -DMOPHI_FETCH_NEWTON=ON)

Running
-------
From the MoPhi build tree (after `cmake --build`):

    python demo/tlfea_newton/demo_tlfea_newton.py

Or from the repository root after installing the mophi package:

    python -m demo.tlfea_newton.demo_tlfea_newton
"""

import sys

# ─── 1. Import MoPhi ─────────────────────────────────────────────────────────
try:
    import mophi
except ImportError as exc:
    sys.exit(
        "ERROR: Could not import the 'mophi' package.\n"
        "       Build MoPhi with -DMOPHI_BUILD_TLFEA_NEWTON=ON and ensure that\n"
        "       the build directory (python/) is on PYTHONPATH.\n"
        f"       ({exc})"
    )

if not hasattr(mophi, "TLFEANewtonCoupler"):
    sys.exit(
        "ERROR: mophi.TLFEANewtonCoupler is not available.\n"
        "       Re-build MoPhi with -DMOPHI_BUILD_TLFEA_NEWTON=ON."
    )

# ─── 2. Import Newton ─────────────────────────────────────────────────────────
try:
    import newton
    import warp as wp
    _newton_available = True
except ImportError:
    _newton_available = False
    print("WARNING: Newton (or warp) is not installed.  "
          "The TLFEA side will still be demonstrated;\n"
          "         install Newton with:  pip install newton")

# ─── 3. Instantiate TLFEA via MoPhi ──────────────────────────────────────────
print("=== MoPhi TLFEA + Newton co-simulation demo ===\n")

coupler = mophi.TLFEANewtonCoupler()

print("[TLFEA] Initializing TLFEANewtonCoupler ...")
coupler.initialize()
print("[TLFEA] TLFEANewtonCoupler initialized.\n")

# ─── 4. Set up a minimal Newton simulation ────────────────────────────────────
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

    model = builder.finalize()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    # Evaluate forward kinematics once to populate initial state.
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Broad-phase collision contacts used by the solver.
    contacts = model.contacts()

    # Use the XPBD solver (the default solver in Newton's own examples).
    newton_solver = newton.solvers.SolverXPBD(model)

    sim_dt = 1.0 / 1000.0  # 1 ms time step
    print(f"[Newton] Double-pendulum model built ({model.body_count} bodies).\n")

# ─── 5. Co-simulation loop ────────────────────────────────────────────────────
num_steps = 5
print(f"Running {num_steps} co-simulation step(s) ...\n")

for i in range(num_steps):
    step_num = i + 1

    # ── TLFEA step ────────────────────────────────────────────────────────────
    coupler.step()
    print(f"  step {step_num}/{num_steps}  [TLFEA] advanced one FEA step")

    # ── Newton step ───────────────────────────────────────────────────────────
    if _newton_available:
        state_0.clear_forces()
        model.collide(state_0, contacts)
        newton_solver.step(state_0, state_1, control, contacts, sim_dt)
        # Swap states for next iteration.
        state_0, state_1 = state_1, state_0
        print(f"  step {step_num}/{num_steps}  [Newton] advanced one physics step (dt={sim_dt:.4f} s)")

    print()

# ─── 6. Finalize ──────────────────────────────────────────────────────────────
print("[TLFEA] Finalizing TLFEANewtonCoupler ...")
coupler.finalize()
print("[TLFEA] TLFEANewtonCoupler finalized.\n")

print("Demo completed successfully.")
if _newton_available:
    print(f"  Newton version : {newton.__version__}")
    print(f"  Warp  version  : {wp.__version__}")
