# Flexible-hand demos

## Run

```bash
# Run 64 downward-facing Allegro hands grasping loose Newton cubes
PYTHONPATH=python python3 demo/newton_xlb_dem/flexible_hand_manipulation/demo_flexible_hand_newton.py
# Run one downward-facing Allegro hand interacting with DEME ellipsoidal clumps
PYTHONPATH=python python3 demo/newton_xlb_dem/flexible_hand_manipulation/demo_flexible_hand_deme.py
```

## Behavior

The flexible-hand Newton demo places 64 downward-facing
Allegro hands in an 8-by-8 laboratory-scale array over individual shallow trays
filled with loose Newton cubes. The flexible-hand DEME demo runs one hand/tray
instance instead, using DEME's HCP sampler to place ellipsoidal clumps above
the tray with small reproducible random initial velocities. DEME reuses
Newton's simplified convex hand contact meshes rather than the fine visual
meshes. These proxies have clump contact disabled during a blocking DEME
settling phase, then switch into an active-contact family for the hand motion.
Newton body poses are mapped to those owners through reusable GPU pose buffers,
and DEME clump poses are read directly into reusable Warp arrays for rendering.
This stage does not feed DEME forces back into Newton. Host copies remain only
for configured diagnostics, final-state metrics, and movie output.
