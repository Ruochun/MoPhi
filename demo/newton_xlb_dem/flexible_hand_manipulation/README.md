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

## Granular-grasp groundwork demo

`demo_flexible_hands_granular_grasp.py` replicates 64 downward-facing Allegro
hands. Each hand closes and rakes through eight loose Newton cubes inside its
own shallow tray. Four repeating grasp profiles vary closure strength, motion
amplitude, and phase so the hands do not all interact with their cube beds in
the same way. The 8-by-8 tray array fills most of the laboratory floor while
leaving visible aisles between neighboring work cells. After the grasp phase,
prescribed kinematic wrist motion raises every hand along the world Z axis to
demonstrate a grasp-and-lift workload.

This first stage uses Newton cubes to establish the hand orientation, per-hand
containers, contact workload, scalable control layout, and lift trajectory.
The `_add_newton_cube_bed()` function intentionally isolates cube creation. A
later stage can replace that block with DEME-managed ellipsoidal clumps like
those used by `demo_claw_newton.py`, then add Newton-DEME contact coupling.

Run from the repository root:

```bash
PYTHONPATH=python python3 demo/newton_xlb_dem/flexible_hand_manipulation/demo_flexible_hands_granular_grasp.py
```

Generated files are written to
`<repository-root>/output/demo_flexible_hands_granular_grasp/`.
