# Python dependencies

The packaged MoPhi wheel declares these Python packages as regular runtime
dependencies because current workflows, demos, visualization, and movie output
use them:

- `torch`
- `GitPython`
- `PyYAML`
- `pycollada`
- `mujoco_warp==3.6.0`
- `pyglet`
- `imageio`
- `imageio-ffmpeg`

The ANYmal multiphysics demo directly requires `torch`. Movie recording paths
require `imageio` and `imageio-ffmpeg`. When installing from a source checkout
instead of a built wheel, install these packages alongside the solver packages
required by the workflow you plan to run.

Solver-specific dependencies and version pins are documented in
[`feris-newton.md`](feris-newton.md) and
[`newton-xlb-deme.md`](newton-xlb-deme.md).
