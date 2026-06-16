# Flexible-hand manipulation demos

This folder hosts articulated-hand manipulation demos that compare a Newton-only
contact workload with a smaller Newton-DEME contact prototype.

## Flexible Hand Newton

`demo_flexible_hand_newton.py` replicates 64 downward-facing Allegro hands.
Each hand closes and rakes through loose Newton cubes inside a shallow tray.
Four repeating grasp profiles vary closure strength, motion amplitude, and
phase. After the grasp phase, prescribed kinematic wrist motion raises every
hand along the world Z axis to demonstrate a grasp-and-lift workload.

Run from the repository root:

```bash
PYTHONPATH=python python3 demo/newton_xlb_dem/flexible_hand_manipulation/demo_flexible_hand_newton.py
```

Generated files are written to
`<repository-root>/output/demo_flexible_hand_newton/`.

## Flexible Hand DEME

`demo_flexible_hand_deme.py` keeps the same laboratory scene and downward-facing
Allegro hand motion, but runs only one hand/tray instance so the DEME contact
path can be debugged. DEME's HCP sampler places ellipsoidal clumps above the
tray location with small reproducible random initial velocities. DEME reuses
Newton's simplified convex Allegro contact meshes rather than the fine visual
meshes. These proxies start in a fixed sleep family with clump contact disabled
during settling, then switch to an active-contact family for the main hand
motion. This coupling remains one-way: DEME contact forces are not fed back
into Newton.

Run from the repository root:

```bash
PYTHONPATH=python python3 demo/newton_xlb_dem/flexible_hand_manipulation/demo_flexible_hand_deme.py
```

Generated files are written to
`<repository-root>/output/demo_flexible_hand_deme/`.
