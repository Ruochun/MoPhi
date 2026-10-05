# FERIS + Newton

```bash
# Configure: fetch FERIS, install Newton via pip, and build the coupler
cmake -B build-py312 \
      -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
      -DMOPHI_FETCH_FERIS=ON \
      -DMOPHI_FETCH_NEWTON=ON \
      -DMOPHI_BUILD_FERIS_NEWTON=ON
cmake --build build-py312

# Run the FERIS + Newton demo
PYTHONPATH=python python3 demo/feris_newton/demo_feris_newton.py
```

The `FERISNewtonCoupler` bridges C++ (FERIS) and Python (Newton) inside a single
`step()` call. Pass the Newton model and solver at initialization time; the coupler
manages states, forward-kinematics setup, and the per-step coupling data exchange.

**Coupling data exchange** (FERIS ↔ Newton) is handled by two methods on the C++ side:

| Method | Direction | Purpose |
|--------|-----------|---------|
| `get_feris_node_positions()` | FERIS → Newton | Deformed FEA node positions forwarded to Newton geometry |
| `set_feris_node_forces(forces)` | Newton → FERIS | Contact/body forces from Newton applied as FERIS external loads |

Newton (with required Warp and MuJoCo) is installed by MoPhi via pip at configure time when
`-DMOPHI_FETCH_NEWTON=ON`. You can also install it manually beforehand:

```bash
pip install --upgrade newton==1.0.0 warp-lang==1.12.1 mujoco==3.6.0
```
