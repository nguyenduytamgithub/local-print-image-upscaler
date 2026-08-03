from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np


V4_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4_DIR))

try:
    from deep_raster_v4 import (  # noqa: E402
        axis_output_bounds,
        axis_target_geometry,
        seam_continuity_probe,
        seam_positions,
    )
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    axis_output_bounds = axis_target_geometry = seam_continuity_probe = seam_positions = None


DEEP_RUNTIME_AVAILABLE = axis_output_bounds is not None
REQUIRE_DEEP_RUNTIME = os.environ.get("RESIZE_V4_REQUIRE_DEEP_RUNTIME") == "1"
if REQUIRE_DEEP_RUNTIME and not DEEP_RUNTIME_AVAILABLE:
    raise RuntimeError(
        "V4 deep tests require the V3 CUDA Python environment; refusing to report a skip."
    )


@unittest.skipUnless(DEEP_RUNTIME_AVAILABLE, "V3 CUDA runtime is required")
class DeepRasterGeometryTests(unittest.TestCase):
    def test_fractional_target_scale_uses_stable_integer_bounds(self) -> None:
        self.assertEqual(axis_output_bounds(376, 512, 2.5), (940, 2220))

    def test_overlap_centres_map_to_expected_output_positions(self) -> None:
        self.assertEqual(
            seam_positions([0, 376, 752], tile=512, ratio=2.5, limit=4000),
            [1110, 2050],
        )

    def test_fractional_non_square_target_keeps_axis_ratios_independent(self) -> None:
        # Original 101x67 at x2.5 becomes 252x168 after per-axis pixel rounding;
        # its exact V3 native master is 404x268.
        ratio_x, padded_target_w = axis_target_geometry(404, 512, 252)
        ratio_y, padded_target_h = axis_target_geometry(268, 512, 168)

        self.assertNotEqual(ratio_x, ratio_y)
        self.assertEqual((padded_target_w, padded_target_h), (319, 321))
        self.assertEqual(axis_output_bounds(0, 512, ratio_x), (0, 319))
        self.assertEqual(axis_output_bounds(0, 512, ratio_y), (0, 321))


@unittest.skipUnless(DEEP_RUNTIME_AVAILABLE, "V3 CUDA runtime is required")
class SeamContinuityTests(unittest.TestCase):
    @staticmethod
    def smooth_ramp() -> np.ndarray:
        y, x = np.mgrid[0:100, 0:100]
        ramp = np.clip(x + y, 0, 255).astype(np.uint8)
        return np.repeat(ramp[:, :, None], 3, axis=2)

    def test_smooth_ramp_is_reported_continuous(self) -> None:
        report = seam_continuity_probe(
            self.smooth_ramp(),
            vertical_positions=[50],
            horizontal_positions=[50],
        )
        self.assertTrue(report["likely_continuous"])
        self.assertAlmostEqual(report["vertical"]["median_ratio"], 1.0)
        self.assertAlmostEqual(report["horizontal"]["median_ratio"], 1.0)

    def test_artificial_jump_is_detected(self) -> None:
        image = self.smooth_ramp()
        image[:, 50:, :] = np.clip(image[:, 50:, :].astype(np.int16) + 50, 0, 255)
        report = seam_continuity_probe(
            image,
            vertical_positions=[50],
            horizontal_positions=[],
        )
        self.assertFalse(report["likely_continuous"])
        self.assertGreater(report["vertical"]["max_ratio"], 1.75)


if __name__ == "__main__":
    unittest.main()
