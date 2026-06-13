# Humanoid complex-environment demo

This folder starts from Newton 1.0.0's public
[`example_robot_policy.py`][newton-policy]
Unitree G1 walking baseline. It provides an interactive keyboard-controlled
humanoid walking policy.

`demo_humanoid_complex_environment.py` keeps Newton's policy and simulation
implementation as the baseline. It preserves the G1's policy-trained colliders
for stable ground contact and adds convex contact proxies generated from its
visual body meshes for external-object contact. Lightweight dynamic boxes in
the walking path demonstrate body-object contact, and pressing `B`
interactively spawns additional boxes in front of the robot. It also adds
MoPhi-style configuration, a finite run loop, MP4 generation, run metadata,
and a final-state snapshot.
Generated files are written to
`<repository-root>/output/demo_humanoid_complex_environment/`.

Press `I` in the viewer to walk the G1 forward into the boxes. The boxes are
free Newton bodies and move when struck by the robot's mesh-derived contact
proxies.

Press `B` to spawn a dynamic box in front of the robot. Because Newton models
have fixed topology after finalization, the demo preallocates a configurable
pool of boxes and activates them on demand. Increase
`INTERACTIVE_OBJECT_POOL_SIZE` when testing larger interactive object counts.
Press `P` to reset the robot and return spawned objects to the pool.

The viewer adds a decorative warehouse aisle with racks, crates, walls, and a
floor grid. These objects exist only in the renderer and never participate in
Newton contact. Set `SHOW_WAREHOUSE_ENVIRONMENT = False` to hide them.

## Assets

No manual asset placement is required. On first run, Newton calls
`newton.utils.download_asset("unitree_g1")` and downloads the Unitree G1 model,
walking policy, and policy configuration from the public
[`newton-assets`](https://github.com/newton-physics/newton-assets) repository.
Newton prints the resulting cache directory as `[Assets] Ready at ...`.

Install the downloader and policy-configuration dependencies, then optionally
prefetch the assets:

```bash
pip install GitPython PyYAML
python -c "import newton.utils; print(newton.utils.download_asset('unitree_g1'))"
```

Newton uses `NEWTON_CACHE_PATH` when that environment variable is set.
Otherwise, it stores assets in the platform user cache, normally
`~/.cache/newton/` on Linux. Do not place assets inside the Python
`site-packages` directory.

If a download was interrupted and the cache is missing files such as
`usd/g1_isaac.usd`, download into a fresh cache directory:

```bash
export NEWTON_CACHE_PATH="$HOME/.cache/newton-mophi"
python -c "import newton.utils; print(newton.utils.download_asset('unitree_g1'))"
```

Keep `NEWTON_CACHE_PATH` set when running the demo.

The next stages will add complex terrain and support more challenging
interactive object shapes and counts.

## Swimming idea demo

`demo_humanoid_swimming.py` demonstrates a deliberately simplified underwater
G1 concept. Newton advances the articulated robot with a scripted swimming
stroke and analytic neutral buoyancy, depth hold, drag, and propulsion forces.
XLB simulates and visualizes flow around ten axis-aligned boxes that follow the
torso, pelvis, and moving arm and leg links.

The prescribed stroke is defined in the `PRESCRIBED_SWIM_MOTION` configuration
block. Each entry controls one joint's pose offset, amplitude, and phase, making
the swimming posture easy to replace without editing the simulation loop.

The viewer adds a decorative underwater sandbed, rocks, marine vegetation, and
blue-green sky. These objects exist only in the renderer and do not affect
Newton or XLB. Set `SHOW_UNDERWATER_ENVIRONMENT = False` to hide them.

This is one-way fluid coupling: the robot changes the XLB obstacle masks, but
XLB pressure and momentum are not yet converted into Newton forces. The
analytic underwater forces make the idea visible without claiming full
hydrodynamic fidelity.

Run it with:

```bash
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_swimming.py
```

Generated files are written to
`<repository-root>/output/demo_humanoid_swimming/`.

[newton-policy]: https://github.com/newton-physics/newton/blob/v1.0.0/newton/examples/robot/example_robot_policy.py
