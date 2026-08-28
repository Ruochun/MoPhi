# G-code support

This document describes and justifies the solver-independent G-code parser in
`python/mophi/utils/gcode.py`. It also explains how demos and tests should use
the parser. The robotic 3D-printing demo does not yet drive its robot from
G-code; that integration is a subsequent milestone.

## Design

The parser converts G-code into `GCodeProgram` and `GCodeMove` records. These
records describe tool-center motion and extrusion state without depending on
Newton, DEME, a particular robot, or a physics timestep.

This separation lets consumers decide how to:

- interpolate the tool path;
- transform G-code coordinates into a simulation world frame;
- solve robot inverse kinematics;
- schedule physics and rendering steps; and
- convert extrusion-axis travel into deposited material.

Lengths are normalized to metres, and feed rates are normalized to metres per
second. Source-line numbers are retained for diagnostics.

The current parser supports:

- `G0` and `G1` rapid and linear motion;
- `G2` and `G3` circular or helical motion using `I`, `J`, and `K` offsets;
- `G17`, `G18`, and `G19` arc working planes;
- `G20` and `G21` inch and millimetre units;
- `G90` and `G91` absolute and incremental coordinates;
- `M82` and `M83` absolute and incremental extrusion;
- `G92` coordinate and extrusion resets; and
- `X`, `Y`, `Z`, `E`, and `F` words.

Unsupported commands are ignored so ordinary slicer metadata does not prevent
path extraction. Radius-form arcs using `R` are not currently supported; arcs
must specify their center with `I`, `J`, or `K`.

## Basic usage

Pass a `pathlib.Path` when reading a file:

```python
from pathlib import Path

from mophi.utils import parse_gcode

program = parse_gcode(
    Path("toolpath.gcode"),
    rapid_feed_rate=1.0,  # G0 speed in m/s
)

print(f"moves: {len(program.moves)}")
print(f"nominal duration: {program.duration:.3f} s")

for move in program.moves:
    print(
        move.command,
        move.start,
        move.end,
        move.feed_rate,
        move.is_extruding,
        move.duration,
    )
```

A string is interpreted as G-code text rather than as a filename:

```python
program = parse_gcode(
    """
    G21
    G90
    M82
    G1 X10 Y5 F600 E2
    G1 X20 E3
    """
)
```

For that example, coordinates and extrusion values expressed in millimetres are
returned in metres, and `F600` becomes `0.01` m/s.

Useful `GCodeMove` fields and properties include:

| Member | Meaning |
|--------|---------|
| `command` | Normalized motion command such as `G0`, `G1`, `G2`, or `G3` |
| `start`, `end` | Cartesian tool-center positions in metres |
| `feed_rate` | Nominal speed in metres per second |
| `extrusion_start`, `extrusion_end` | Extrusion-axis positions in metres |
| `extrusion_delta` | Signed extrusion change during the move |
| `is_extruding` | `True` when the extrusion axis advances |
| `length` | Linear, circular, or helical path length in metres |
| `duration` | Nominal `length / feed_rate` duration in seconds |
| `source_line` | Original G-code line number |

## Running the tests

The tests are in `tests/test_gcode.py`. They cover linear/modal motion, unit
conversion, incremental positioning and extrusion, arc length, rapid feed rate,
and `G92` resets.

Because importing `mophi.utils` first imports the `mophi` package, the active
interpreter must have a compatible `mophi_core` extension. For a source-tree
developer build, configure and build MoPhi with that same interpreter as
described in [how-to.md](how-to.md#source-tree-developer-builds):

```bash
cmake -B build-py312 \
  -DMOPHI_BUILD_PYTHON_BINDINGS=ON \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build-py312 -j
```

Then run:

```bash
PYTHONPATH=python python -m unittest discover \
  -s tests \
  -p 'test_gcode.py' \
  -v
```

If MoPhi is installed as a wheel in the active environment, omit
`PYTHONPATH=python`.

## Current scope

The parser does not yet sample moves at arbitrary times, transform a path into
the build-plate frame, solve robot inverse kinematics, or emit DEME particles.
Those operations belong to trajectory and co-simulation layers built on top of
the parsed program rather than in the parser itself.
