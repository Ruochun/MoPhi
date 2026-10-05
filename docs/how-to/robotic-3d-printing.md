# Robotic 3D printing demos

## Run

```bash
# Run the robotic 3D-printing co-simulation (Newton arm + DEME spheres)
PYTHONPATH=python python3 demo/newton_dem/robotic_3d_printing/demo_robotic_3d_printing.py
```

## DFC material discharge and printing

`demo_dfc_3d_printing.py` currently prepares and discharges material for the future
DFC printing co-simulation. DEME loads the demo-local `data/DFCModel.cu` fresh-
concrete contact law and uses MoPhi's open-mesh cross-section sampler to place
spheres directly inside the upper auger chamber. Its material calibration and
particle construction follow the Chrono CPU `Demo_DFC_MiniSlump`: aggregate
diameters follow a deterministic 2--4 mm grading, each collision radius includes
the 2.5 mm mortar layer, and particle mass uses the MiniSlump density convention.
A temporary analytical +X side plane keeps the particles near the auger wall,
while a -Z-facing top plane closes the chamber above them. The auger mesh
supports the material from below. DEME settles the particles until their total
kinetic energy remains below the configured threshold, then disables both
temporary planes and begins discharge in the same solver. The DFC coordinates
use the reference model's millimetre-tonne-second convention and are converted
to metres only for visualization.
DEME also owns a permanent upward-facing analytical plane at the top surface of
Newton's visible build plate. This infinite DEME plane remains active through
both phases and catches discharged material at the matching world-space height;
the finite green box remains the Newton representation. Set
`ENABLE_DEME_BUILD_PLATE = False` to omit this DEME contact boundary. With
`SHOW_DEME_BUILD_PLATE = True`, an orange outline and center cross over the
finite plate footprint identify the infinite analytical plane in live and movie
rendering; the marker is hidden whenever physical plate contact is disabled.
The complete CAD housing remains the visual mesh, while Newton and DEME share
the reduced contact proxy so cohesive particles cannot contact the thin wall's
exterior. Each phase starts from a conservative DEME timestep and grows to its
configured phase-specific cap.

The broad DEME domain is only a collision-search extent and creates no physical
world-box boundary.

The demo retains the G-code-driven motion foundation for the later printing
phase. It parses
`robotic_3d_printing/data/dfc_single_bead.gcode`, evaluates the Cartesian path
at the rendering cadence, and uses Newton's inverse-kinematics solver to keep
the ABB IRB6700 auger outlet on that path while preserving its initial
orientation. The Newton articulation is reconstructed from the pinned Chrono
reference demo's JSON metadata and OBJ meshes in
`robotic_3d_printing/data/chrono_irb6700_printer/`. The cyan guide line shows
the requested toolpath. A magenta wireframe overlays the exact inner-surface
triangle mesh registered for particle contact; startup validation rejects
non-finite vertices, degenerate triangles, invalid indices, and non-manifold
edges while deliberately permitting the proxy's open boundary edges.
Before DEME initialization, the demo calls `SplitIntoConvexPatches` on this
proxy with the configured 20-degree local face-normal threshold. The returned
patch count is printed and recorded in `run_metadata.json`.
With `SAVE_AUGER_CONTACT_PROXY_VTK = True`, the same proxy is also written by
itself in its initial world pose to
`output/demo_dfc_3d_printing/auger_contact_proxy.vtk` for direct ParaView
inspection.
Set `RUN_PARTICLE_DISCHARGE_AFTER_SETTLING = False` to stop after settling. Set
`RUN_GCODE_MOTION_AFTER_SETTLING = True` only to exercise the existing Newton
motion after discharge; dynamic Newton-to-DEME proxy pose updates are still a
later step.

With `SAVE_MOVIE = True`, the MP4 contains the settling and subsequent
gravity-fed discharge frames. The default four-second discharge contributes
200 frames at 50 FPS, in addition to the variable-length settling phase. A run
performed with movie recording disabled does not replace an older MP4 already
present in the output directory, so use the final console summary and
`run_metadata.json` to confirm the frame counts for the current run.

Run the motion stage after building the Newton-enabled Python bindings:

```bash
PYTHONPATH=python python3 demo/newton_dem/robotic_3d_printing/demo_dfc_3d_printing.py

# Equivalent convenience target:
cmake --build build-py311 --target demo_dfc_3d_printing
```

Edit `GCODE_PATH`, `GCODE_TIME_SCALE`, and `GCODE_TO_WORLD_SCALE` in the
configuration block to select and place another toolpath. See
[`gcode.md`](../gcode.md) for the supported command subset and parser API. Output
is written beneath `output/demo_dfc_3d_printing/`. With
`SAVE_VTK_TIME_SERIES = True`, the demo writes one numbered VTK snapshot per
rendered frame and a `dfc_3d_printing.vtk.series` manifest containing the complete
ParaView time series. Each frame contains world-space printer CAD meshes, the
build plate, and DEME particle sphere surfaces with radius metadata. See
[`vtk-output.md`](../vtk-output.md) for the snapshot format and cell labels.
The initial stock region is controlled by `DEME_AUGER_SAMPLE_BOUNDS_MIN` and
`DEME_AUGER_SAMPLE_BOUNDS_MAX` in contact-mesh-local metres. See
[`mesh-sampling.md`](../mesh-sampling.md) for the open-mesh sampling method.

## UR10 additive manufacturing

`demo_robotic_3d_printing.py` is a Newton + DEME additive-manufacturing demo
whose sphere population currently exercises DEME's custom
`ForceModelWithCohesion.cu` kernel with `DEME_COHESION = 0.0`. This isolates
the custom-kernel path before cohesion or domain changes are enabled. The model
is stored under `data/force_models/` and loaded explicitly by the demo.
DEME's regular-grid sampler derives that population from a configurable
funnel-local box contained within the bowl. The initial Newton funnel transform
maps those sampled points into world space, rather than relying on an unrelated
global center or a fixed particle count.
Newton downloads its public Universal Robots UR10
asset on first use, then the script attaches the hollow
`data/mesh/funnel.obj` print-head mesh to the wrist. A static build plate and
support frame complete the machine. After settling, the arm moves once from its
starting posture toward a modest lateral endpoint, leaving a nearly straight
particle trail without reversing direction. DEME initializes
a small sphere cloud, six analytical box walls from explicit XYZ ranges, and a
separate analytical plane at the build-plate height. The box's lower Z wall is
below the build plate, avoiding coincident contact boundaries.
Newton's machine stage uses zero gravity so its prescribed arm posture does not
sag during warm-up; DEME retains its own downward gravity for particle settling.
The DEME funnel is always an externally driven contact proxy. It starts at the
Newton funnel's configured initial pose, and every Newton substep updates its
DEME owner pose directly from Newton's device state before DEME advances. There is no independent DEME funnel
trajectory or conditional pose-coupling mode. DEME-to-Newton force feedback is
controlled separately by `ENABLE_DEME_TO_NEWTON_FORCE_FEEDBACK` and defaults to
`False`, so particles do not disturb the prescribed Newton arm motion. When
enabled, `NewtonDEMEContactAccelerationCoupler` retrieves contact acceleration,
angular acceleration, mass, inertia, and orientation into reusable Warp CUDA
arrays and forms the world-space wrench in Newton's external body-force array.
The code marks
a pending correction to shift torque from the DEME funnel frame to the Newton
end-effector frame before force feedback is used for quantitative studies. Particle
positions are likewise retrieved by `NewtonDEMEParticleExchange` directly into
a reusable Warp array for rendering. Recurring Newton–DEME physics exchange no
longer uses tracker host getters or setters. Its generated movie, USD (when selected), and run metadata are written beneath
`output/demo_robotic_3d_printing/`.

The initial settling interval advances both Newton and DEME. With the default
`RENDER_SETTLING_PHASE = True`, it is rendered and recorded at `RENDER_FPS`
before the arm begins its printing trajectory; disabling that flag keeps the
same physics warm-up but omits its frames from the output.

Build and run it directly or through its CMake target:

```bash
cmake -B build-py312 \
      -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
      -DMOPHI_BUILD_NEWTON_XLB_DEM=ON \
      -DMOPHI_BUILD_DEMOS=ON
cmake --build build-py312
PYTHONPATH=python python3 demo/newton_dem/robotic_3d_printing/demo_robotic_3d_printing.py

# Equivalent convenience target (opens the interactive viewer):
cmake --build build-py312 --target demo_robotic_3d_printing
```

Prerequisites are the Newton, Warp, MuJoCo, `deme[cuda12]`, imageio, and
imageio-ffmpeg packages listed in
[`newton-xlb-deme.md`](newton-xlb-deme.md). Newton and both DEME workers must
use the same logical CUDA device for the direct pointer exchange.
