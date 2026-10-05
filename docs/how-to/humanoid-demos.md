# Humanoid demos

## Run

```bash
# Run the interactive Newton humanoid walking baseline
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_complex_environment.py
# Run the two-robot Newton-DEME humanoid fighting demo
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_robot_fighting.py
# Run the Newton + XLB humanoid swimming idea demo
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_swimming.py
```

## Newton complex environment

The humanoid complex-environment demo starts from Newton 1.0.0's public
keyboard-controlled Unitree G1 walking-policy example. On first run, Newton
downloads the G1 model, policy, and YAML configuration from the public
`newton-assets` repository; no manual asset placement is required. This first
stage preserves the policy-trained ground colliders, adds convex Newton contact
proxies from the G1's visual body meshes for external-object contact, and
places lightweight dynamic boxes in its walking path. It also adds MoPhi-style
configuration, movie output, run metadata, and a final-state snapshot. Press
`I` in the viewer to walk forward into the boxes and observe body-object
contact. Press `B` to interactively spawn an additional dynamic box in front
of the robot. The demo uses a configurable preallocated object pool because
Newton model topology is fixed after finalization; press `P` to reset the
robot and return spawned objects to the pool.
The viewer also shows a non-physical warehouse aisle; set
`SHOW_WAREHOUSE_ENVIRONMENT = False` to hide the decorative scene.

## Newton–DEME robot fighting

The humanoid robot-fighting demo places two G1 robots face-to-face and commands
both policies forward for five seconds. Newton manages articulated dynamics and
uses its coarse policy colliders for ground contact. DEME resolves cross-robot
mesh contact and feeds each proxy body's force and torque back into Newton;
Newton's duplicate cross-robot collider pairs are disabled. The scene contains
no dynamic boxes, warehouse shelves, or other decorative obstacles. The demo
records a mesh manifest under `output/demo_humanoid_robot_fighting/`, alongside
its movie, metadata, and final-state snapshot.
Set `ROBOT_CONTACT_MODEL` to `"deme"` for this coupled path or `"newton"`
for the original coarse-contact comparison. `SHOW_CONTACT_PROXY_MESHES`
independently turns the wireframe overlay on or off; hiding it does not disable
DEME contact.
It also loads `data/mesh/cube.obj` and draws scaled instances of that triangle
mesh around every visual robot part. The cyan and orange wireframe overlays
follow the two robots' links and show the meshes used for DEME contact. Proxy
components attached to the same Newton body are merged into a body-local OBJ
under `output/demo_humanoid_robot_fighting/deme_contact_proxies/`; configure
the mesh path, padding, line width, colors, or visibility in the demo's
contact-proxy block.
The default script waits until the robots have met, then performs four
alternating punches between 2.05 and 4.85 seconds. Kicks are omitted because
the locomotion controller does not robustly support the required single-leg
balance. `FIGHT_ATTACK_SEQUENCE` stores each attack as
`(start_time, duration, attacker, kind, side)`. The attack trajectory
is added to the learned locomotion target rather than prescribing body poses,
so contact forces and loss of balance remain part of the Newton simulation.
Set `ENABLE_SCRIPTED_FIGHT = False` for the previous approach-only behavior.
The visualization block provides brighter sky/light colors and a closer camera
for viewing strike details; these settings affect rendering only.

## Newton–XLB swimming

The humanoid swimming idea demo places the G1 in an XLB fluid domain without a
ground plane. Newton advances a scripted swimming stroke while analytic
buoyancy, drag, and propulsion keep the robot suspended and moving. Ten
axis-aligned XLB obstacle boxes follow the torso, pelvis, arms, and legs so the
visualized flow responds to the prescribed limb motion. The recurring analytic
force evaluation and multi-body Newton-to-XLB mask exchange stay on the shared
GPU; host copies are limited to initialization, visualization, and output.
Edit the
`PRESCRIBED_SWIM_MOTION` configuration block to change each swimming joint's
pose offset, amplitude, and phase. This first stage is intentionally one-way:
XLB does not yet calculate forces that feed back into Newton.
The underwater sandbed, rocks, vegetation, and background colors are
renderer-only decorations. Set `SHOW_UNDERWATER_ENVIRONMENT = False` to hide
them.
