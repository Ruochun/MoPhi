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

No manual asset placement is required. On first run, MoPhi asks Newton to
download the Unitree G1 model, walking policy, and policy configuration from the public
[`newton-assets`](https://github.com/newton-physics/newton-assets) repository.
Newton prints the resulting cache directory as `[Assets] Ready at ...`.

Install the downloader and policy-configuration dependencies:

```bash
pip install GitPython PyYAML
```

Newton uses `NEWTON_CACHE_PATH` when that environment variable is set.
Otherwise, it stores assets in the platform user cache, normally
`~/.cache/newton/` on Linux. Do not place assets inside the Python
`site-packages` directory.

MoPhi validates the model, policy, and configuration after every download. If
Newton returns an incomplete cached sparse checkout, MoPhi automatically calls
Newton's supported force-refresh path once and validates the result again. See
[`docs/newton-assets.md`](../../../docs/newton-assets.md) for the shared cache
handling contract and failure behavior.

If automatic refresh fails because the configured cache root is unwritable or
corrupted, use a new cache root as a **last resort**:

```bash
export NEWTON_CACHE_PATH="$HOME/.cache/newton-mophi-fresh"
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_robot_fighting.py
```

Keep `NEWTON_CACHE_PATH` set when reusing that fresh cache.

The next stages will add complex terrain and support more challenging
interactive object shapes and counts.

## Robot-fighting groundwork demo

`demo_humanoid_robot_fighting.py` places two G1 robots face-to-face and gives
both policies continuous forward commands. Newton's original coarse,
policy-trained collision shapes handle ground and robot-to-robot contact. No
dynamic boxes, warehouse shelves, or other decorative scene objects are added.
The robots' high-resolution visual meshes do not participate in collision.

The demo loads the centered unit-box triangle mesh from `data/mesh/cube.obj`
and scales a copy around every visual robot part, including the torso/head
geometry, arm links, hands, and finger links. Each instance matches the padded
local bounds of its corresponding visual mesh and follows the owning Newton
body every frame. Cyan triangle wireframes identify the first robot; orange
wireframes identify the second. These boxes are not Newton collision shapes and
do not affect physics. Their scaled vertices and the OBJ triangle indices are
retained for later DEME mesh-tracker registration. Set
`SHOW_CONTACT_PROXY_MESHES = False` to hide the overlay, or tune
`CONTACT_PROXY_MESH_PATH`, `CONTACT_PROXY_PADDING`, and
`CONTACT_PROXY_LINE_WIDTH` in the configuration block.

During model construction, the demo nevertheless validates that each visual
mesh exposes vertices and triangle indices and retains its body association,
body-local transform, and scale. It writes vertex/triangle counts to
`output/demo_humanoid_robot_fighting/visual_mesh_manifest.json`. The live mesh objects
and transforms remain available on `HumanoidContactExample.robot_visual_meshes`
for a later DEME tracker integration without changing the robot asset pipeline.

Run it with:

```bash
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_robot_fighting.py
```

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
