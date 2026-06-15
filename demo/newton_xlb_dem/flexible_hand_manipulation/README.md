# Flexible-hand manipulation demos

This folder hosts scalable articulated-hand manipulation and contact demos.
The current first stage uses Newton rigid-link Allegro hands; later stages can
replace their collision representation with DEM-Engine-style clump spheres
and add more challenging object shapes and counts.

## Many articulated hands demo

`demo_many_flexible_hands.py` adapts Newton's public Allegro-hand example and
`manipulation_objects/cup` asset into a MoPhi-style contact scaling demo. It
replicates 64 fixed-base hands, 64 enlarged dynamic cups with contact-enabled
rounded handles, and independently drives all 16 finger joints on every hand.

Four repeating grasp profiles vary closure strength, motion amplitude, cup
placement, and cup tilt. Some hands cradle their cups while overclosed,
pulsing, or deliberately open hands can knock down or drop theirs. Cups begin
just outside the fingers to avoid initial penetration, and each compound
handle shares its bowl's display color.

The viewer adds a decorative laboratory with a tiled floor, paneled walls,
work benches, instruments, ceiling light panels, and bright neutral lighting.
These objects exist only in the renderer and do not participate in contact.
Set `SHOW_LABORATORY_ENVIRONMENT = False` to hide them.

Run from the repository root:

```bash
PYTHONPATH=python python3 demo/newton_xlb_dem/flexible_hand_manipulation/demo_many_flexible_hands.py
```

Newton downloads the public `wonik_allegro` and `manipulation_objects/cup`
assets automatically on first run. Generated files are written to
`<repository-root>/output/demo_many_flexible_hands/`.

## Granular-grasp demo

`demo_flexible_hands_granular_grasp.py` replicates 64 downward-facing Allegro
hands. By default, DEME's HCP sampler fills a box above every tray location
with ellipsoidal clumps. Small reproducible random initial velocities disturb
the HCP packing as the particles fall and settle into piles. Four repeating
grasp profiles vary closure strength, motion amplitude, and phase. After the
grasp phase, prescribed kinematic wrist motion raises every hand along the
world Z axis to demonstrate a grasp-and-lift workload.

DEME uses one shared infinite ground plane, allowing particles to spread freely
between the visually rendered trays. For the current runtime test,
`ENABLE_DEME_HAND_MESH_PROXIES = False` by default, so no hand meshes are
exported or loaded into DEME and the clumps do not contact the hands. Enabling
that flag exports the Allegro visual meshes as body-local DEME contact proxies
and synchronizes them from Newton after each Newton step. This coupling remains
one-way: DEME contact forces are not fed back into Newton. Set
`USE_DEME_CLUMPS = False` to use the earlier Newton-cube fallback instead. The
default DEME path requires the `deme` Python package and a DEME-compatible CUDA
GPU.

Run from the repository root:

```bash
PYTHONPATH=python python3 demo/newton_xlb_dem/flexible_hand_manipulation/demo_flexible_hands_granular_grasp.py
```

Generated files are written to
`<repository-root>/output/demo_flexible_hands_granular_grasp/`.
