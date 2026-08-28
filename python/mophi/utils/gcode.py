"""Solver-independent parsing of the G-code subset used by printing demos.

All lengths are normalized to metres and feed rates to metres per second. The
parser preserves motion commands instead of constructing a solver-specific
trajectory, leaving interpolation, inverse kinematics, and stepping to callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Iterable

_WORD_PATTERN = re.compile(r"([A-Za-z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)")
_PAREN_COMMENT_PATTERN = re.compile(r"\([^)]*\)")


@dataclass(frozen=True)
class GCodeMove:
    """One tool-center motion, represented in normalized SI units."""

    command: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    feed_rate: float
    extrusion_start: float
    extrusion_end: float
    arc_center_offset: tuple[float, float, float] | None
    arc_plane: str | None
    clockwise: bool | None
    source_line: int

    @property
    def extrusion_delta(self) -> float:
        """Signed extrusion-axis travel during this move, in metres."""
        return self.extrusion_end - self.extrusion_start

    @property
    def is_extruding(self) -> bool:
        """Whether this move advances the extrusion axis."""
        return self.extrusion_delta > 0.0

    @property
    def length(self) -> float:
        """Geometric path length in metres."""
        if self.arc_center_offset is None:
            return math.dist(self.start, self.end)

        plane_axes = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}
        if self.arc_plane not in plane_axes:
            raise ValueError(f"line {self.source_line}: arc has no supported working plane")
        axis_a, axis_b, remaining_axis = plane_axes[self.arc_plane]
        center_a = self.start[axis_a] + self.arc_center_offset[axis_a]
        center_b = self.start[axis_b] + self.arc_center_offset[axis_b]
        start_angle = math.atan2(self.start[axis_b] - center_b, self.start[axis_a] - center_a)
        end_angle = math.atan2(self.end[axis_b] - center_b, self.end[axis_a] - center_a)
        sweep = end_angle - start_angle
        if self.clockwise and sweep >= 0.0:
            sweep -= 2.0 * math.pi
        elif not self.clockwise and sweep <= 0.0:
            sweep += 2.0 * math.pi
        radius = math.hypot(self.start[axis_a] - center_a, self.start[axis_b] - center_b)
        planar_length = abs(sweep) * radius
        return math.hypot(planar_length, self.end[remaining_axis] - self.start[remaining_axis])

    @property
    def duration(self) -> float:
        """Nominal move duration in seconds."""
        if self.feed_rate <= 0.0:
            raise ValueError(f"line {self.source_line}: motion has no positive feed rate")
        return self.length / self.feed_rate


@dataclass(frozen=True)
class GCodeProgram:
    """Parsed motion sequence with solver-neutral SI-valued records."""

    moves: tuple[GCodeMove, ...]

    @property
    def duration(self) -> float:
        return sum(move.duration for move in self.moves)


def parse_gcode(source: str | Path | Iterable[str], *, rapid_feed_rate: float = 1.0) -> GCodeProgram:
    """Parse common printing G-code into a solver-independent motion sequence.

    Supported modal commands are G0/G1, G2/G3 with I/J/K arc centers, G20/G21,
    G90/G91, M82/M83, and F/E/X/Y/Z words. ``rapid_feed_rate`` is in m/s.
    Unsupported commands are ignored so slicer metadata does not prevent path
    extraction; invalid motion state raises an exception.
    """
    if rapid_feed_rate <= 0.0:
        raise ValueError("rapid_feed_rate must be positive")

    if isinstance(source, Path):
        lines = source.read_text(encoding="utf-8").splitlines()
    elif isinstance(source, str):
        lines = source.splitlines()
    else:
        lines = list(source)

    unit_scale = 1.0e-3
    xyz_absolute = True
    extrusion_absolute = True
    position = [0.0, 0.0, 0.0]
    extrusion = 0.0
    feed_rate = 0.0
    modal_motion: int | None = None
    arc_plane = "XY"
    moves: list[GCodeMove] = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = _PAREN_COMMENT_PATTERN.sub("", raw_line).split(";", 1)[0]
        words = [(letter.upper(), float(value)) for letter, value in _WORD_PATTERN.findall(line)]
        if not words:
            continue

        g_codes = [int(value) for letter, value in words if letter == "G" and value.is_integer()]
        m_codes = [int(value) for letter, value in words if letter == "M" and value.is_integer()]
        if 20 in g_codes:
            unit_scale = 0.0254
        if 21 in g_codes:
            unit_scale = 1.0e-3
        if 90 in g_codes:
            xyz_absolute = True
        if 91 in g_codes:
            xyz_absolute = False
        if 17 in g_codes:
            arc_plane = "XY"
        if 18 in g_codes:
            arc_plane = "XZ"
        if 19 in g_codes:
            arc_plane = "YZ"
        if 82 in m_codes:
            extrusion_absolute = True
        if 83 in m_codes:
            extrusion_absolute = False

        line_motion = next((code for code in g_codes if code in (0, 1, 2, 3)), None)
        if line_motion is not None:
            modal_motion = line_motion

        values = {letter: value for letter, value in words if letter not in ("G", "M")}
        if "F" in values:
            feed_rate = values["F"] * unit_scale / 60.0

        if 92 in g_codes:
            for index, axis in enumerate(("X", "Y", "Z")):
                if axis in values:
                    position[index] = values[axis] * unit_scale
            if "E" in values:
                extrusion = values["E"] * unit_scale
            continue

        has_motion_axis = any(axis in values for axis in ("X", "Y", "Z"))
        if modal_motion is None or not has_motion_axis:
            if "E" in values:
                value = values["E"] * unit_scale
                extrusion = value if extrusion_absolute else extrusion + value
            continue

        start = tuple(position)
        end = list(position)
        for index, axis in enumerate(("X", "Y", "Z")):
            if axis in values:
                value = values[axis] * unit_scale
                end[index] = value if xyz_absolute else end[index] + value

        extrusion_start = extrusion
        if "E" in values:
            value = values["E"] * unit_scale
            extrusion = value if extrusion_absolute else extrusion + value

        arc_offset = None
        clockwise = None
        if modal_motion in (2, 3):
            arc_offset = tuple(values.get(axis, 0.0) * unit_scale for axis in ("I", "J", "K"))
            if not any(arc_offset):
                raise ValueError(f"line {line_number}: G2/G3 requires an I/J/K arc-center offset")
            clockwise = modal_motion == 2

        effective_feed = rapid_feed_rate if modal_motion == 0 else feed_rate
        moves.append(
            GCodeMove(
                command=f"G{modal_motion}",
                start=start,
                end=tuple(end),
                feed_rate=effective_feed,
                extrusion_start=extrusion_start,
                extrusion_end=extrusion,
                arc_center_offset=arc_offset,
                arc_plane=arc_plane if arc_offset is not None else None,
                clockwise=clockwise,
                source_line=line_number,
            )
        )
        position = end

    return GCodeProgram(tuple(moves))
