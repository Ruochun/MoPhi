import math
import unittest

from mophi.utils.gcode import parse_gcode


class GCodeParserTest(unittest.TestCase):
    def test_linear_modal_motion_and_extrusion(self):
        program = parse_gcode("G21\nG90\nM82\nG1 X10 Y5 F600 E2\nX20 E3\n")
        self.assertEqual(len(program.moves), 2)
        self.assertEqual(program.moves[0].end, (0.01, 0.005, 0.0))
        self.assertAlmostEqual(program.moves[0].feed_rate, 0.01)
        self.assertAlmostEqual(program.moves[1].extrusion_delta, 0.001)
        self.assertTrue(program.moves[1].is_extruding)

    def test_incremental_coordinates_and_extrusion(self):
        program = parse_gcode("G21 G91 M83\nG1 X2 E1 F60\nX3 E2\n")
        self.assertEqual(program.moves[-1].end, (0.005, 0.0, 0.0))
        self.assertAlmostEqual(program.moves[-1].extrusion_end, 0.003)

    def test_arc_length_and_duration(self):
        program = parse_gcode("G21\nG90\nG1 X10 Y0 F600\nG3 X0 Y10 I-10 J0\n")
        arc = program.moves[-1]
        self.assertAlmostEqual(arc.length, math.pi * 0.01 / 2.0)
        self.assertAlmostEqual(arc.duration, math.pi / 2.0)

    def test_inch_units_and_rapid_feed(self):
        program = parse_gcode("G20\nG0 X1\n", rapid_feed_rate=2.0)
        self.assertEqual(program.moves[0].end, (0.0254, 0.0, 0.0))
        self.assertEqual(program.moves[0].feed_rate, 2.0)

    def test_g92_resets_coordinates_without_creating_motion(self):
        program = parse_gcode("G21\nG1 X10 E2 F60\nG92 X0 E0\nG1 X5 E1\n")
        self.assertEqual(len(program.moves), 2)
        self.assertEqual(program.moves[-1].start, (0.0, 0.0, 0.0))
        self.assertEqual(program.moves[-1].end, (0.005, 0.0, 0.0))
        self.assertAlmostEqual(program.moves[-1].extrusion_delta, 0.001)


if __name__ == "__main__":
    unittest.main()
