import tempfile
import unittest
from pathlib import Path

from generate_scaling_plot_tikz import figure_specs, generate_tikz, validate_panels


class ScalingPlotTests(unittest.TestCase):
    def test_framework_panel_membership(self):
        _, _, panels, legend = figure_specs()[0]
        self.assertEqual([panel[1] for panel in panels], [22000, 1000])
        self.assertEqual(len(panels[0][3]), 6)
        self.assertEqual([name for name, _ in panels[1][3]],
                         ["aix_coll_default", "phydll_default", "phydll_readiness"])
        self.assertEqual(len(legend), 6)

    def test_layouts_have_only_the_three_aix_modes(self):
        _, _, panels, legend = figure_specs()[1]
        self.assertEqual(len(panels), 3)
        for panel, layout in zip(panels, ("default", "affine", "numa")):
            self.assertEqual(panel[1], 250)
            self.assertEqual(panel[3], [(f"{mode}_{layout}", mode) for mode in legend])

    def test_invalid_values_are_rejected(self):
        panel = [("Test", 250, [0, 250], [("curve", "aix_coll")])]
        for value in (None, float("nan"), float("inf"), -1, 251):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_panels([1], {"curve": [value]}, panel)
        with self.assertRaises(ValueError):
            validate_panels([1, 2], {"curve": [100]}, panel)

    def test_invalid_range_does_not_overwrite_output(self):
        name, title, panels, legend = figure_specs()[0]
        data = {key: [900] * 5 for panel in panels for key, _ in panel[3]}
        data["phydll_default"][0] = 1001
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / f"{name}.tex"
            output.write_text("unchanged")
            with self.assertRaises(ValueError):
                generate_tikz([1, 2, 4, 8, 16], data, output, title, panels, legend)
            self.assertEqual(output.read_text(), "unchanged")

    def test_each_panel_clips_its_curves(self):
        for name, title, panels, legend in figure_specs():
            data = {key: [100] * 5 for panel in panels for key, _ in panel[3]}
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / f"{name}.tex"
                generate_tikz([1, 2, 4, 8, 16], data, output, title, panels, legend)
                self.assertEqual(output.read_text().count(r"\clip"), len(panels))
                fragments = list(Path(directory).glob("*.pgf"))
                self.assertEqual(len(fragments), len(panels) + 1)
                for fragment in fragments:
                    text = fragment.read_text()
                    self.assertEqual(text.count(r"\begin{tikzpicture}"), 1)
                    self.assertEqual(text.count(r"\end{tikzpicture}"), 1)
                    self.assertNotIn(r"\documentclass", text)
                    self.assertNotIn("SmartSim curves appear", text)
                    if fragment.stem.endswith("_legend"):
                        self.assertNotIn(r"\clip", text)
                    else:
                        self.assertEqual(text.count(r"\clip"), 1)
                        self.assertIn("rectangle (", text)
                        self.assertIn(",2.1)", text)


if __name__ == "__main__":
    unittest.main()
