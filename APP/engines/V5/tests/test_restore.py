from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
V5_DIR = PROJECT_ROOT / "APP" / "engines" / "V5"
sys.path.insert(0, str(V5_DIR))

from v5lib.restore import _deterministic_poster_restore, restore_background  # noqa: E402


def linear_to_uint8(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    srgb = np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )
    return np.rint(srgb * 255.0).astype(np.uint8)


def central_mask(height: int = 120, width: int = 160) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[32:88, 58:104] = True
    return mask


def occlude(truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    source = truth.copy()
    source[mask] = np.array((220, 20, 30), dtype=np.uint8)
    return source


class PosterSurfaceSelectionTests(unittest.TestCase):
    def assert_outside_identical(
        self,
        source: np.ndarray,
        restored: np.ndarray,
        footprint: np.ndarray,
    ) -> None:
        self.assertTrue(np.array_equal(source[~footprint], restored[~footprint]))

    def test_flat_field_prefers_constant_model(self) -> None:
        mask = central_mask()
        truth = np.full((120, 160, 3), (244, 240, 230), dtype=np.uint8)
        source = occlude(truth, mask)

        restored, report = _deterministic_poster_restore(source, mask)

        component = report["components"][0]
        self.assertEqual(component["surface_model"], "constant")
        self.assertEqual(component["method"], "robust_flat_colour")
        self.assertTrue(component["surface_confident"])
        self.assertTrue(np.array_equal(restored[mask], truth[mask]))
        self.assert_outside_identical(source, restored, mask)

    def test_affine_linear_light_gradient_is_selected_and_reconstructed(self) -> None:
        height, width = 120, 160
        yy, xx = np.indices((height, width))
        x = xx / (width - 1)
        y = yy / (height - 1)
        truth = linear_to_uint8(
            np.stack(
                (
                    0.25 + 0.40 * x,
                    0.28 + 0.30 * x + 0.03 * y,
                    0.32 + 0.20 * x,
                ),
                axis=2,
            )
        )
        mask = central_mask(height, width)
        source = occlude(truth, mask)

        restored, report = _deterministic_poster_restore(source, mask)

        component = report["components"][0]
        self.assertEqual(component["surface_model"], "affine")
        self.assertEqual(component["method"], "robust_linear_gradient")
        error = np.abs(restored.astype(np.int16) - truth.astype(np.int16))[mask]
        self.assertLessEqual(float(np.percentile(error, 95)), 5.0)
        self.assert_outside_identical(source, restored, mask)

    def test_quadratic_field_requires_quadratic_model(self) -> None:
        height, width = 120, 160
        yy, xx = np.indices((height, width))
        x = xx / (width - 1)
        y = yy / (height - 1)
        truth = linear_to_uint8(
            np.stack(
                (
                    0.22 + 0.45 * (x - 0.5) ** 2,
                    0.25 + 0.35 * (x - 0.5) ** 2 + 0.06 * y,
                    0.30 + 0.25 * (x - 0.5) ** 2,
                ),
                axis=2,
            )
        )
        mask = central_mask(height, width)
        source = occlude(truth, mask)

        restored, report = _deterministic_poster_restore(source, mask)

        component = report["components"][0]
        self.assertEqual(component["surface_model"], "quadratic")
        self.assertEqual(component["method"], "robust_quadratic_gradient")
        error = np.abs(restored.astype(np.int16) - truth.astype(np.int16))[mask]
        self.assertLessEqual(float(np.percentile(error, 95)), 1.0)
        self.assert_outside_identical(source, restored, mask)

    def test_fragmented_repeating_pattern_does_not_collapse_to_one_colour(self) -> None:
        height, width = 120, 160
        yy, xx = np.indices((height, width))
        first = np.array((20, 160, 230), dtype=np.uint8)
        second = np.array((230, 40, 70), dtype=np.uint8)
        truth = np.where((((xx // 3) + (yy // 3)) % 2)[..., None], first, second)
        mask = central_mask(height, width)
        source = occlude(truth, mask)

        restored, report = _deterministic_poster_restore(source, mask)

        component = report["components"][0]
        self.assertFalse(component["surface_confident"])
        self.assertEqual(component["method"], "opencv_telea")
        self.assertGreater(
            component["background_samples"]["selected_fragmentation"],
            0.75,
        )
        self.assert_outside_identical(source, restored, mask)


class PosterStructureRepairTests(unittest.TestCase):
    def test_long_horizontal_rule_is_continued_through_hole(self) -> None:
        mask = central_mask()
        truth = np.full((120, 160, 3), (246, 243, 234), dtype=np.uint8)
        truth[47:55] = np.array((10, 85, 22), dtype=np.uint8)
        source = occlude(truth, mask)

        restored, report = _deterministic_poster_restore(source, mask)

        line_report = report["components"][0]["horizontal_line_repair"]
        self.assertGreater(line_report["restored_pixels"], 0)
        self.assertTrue(set(range(47, 55)).issubset(line_report["restored_rows"]))
        self.assertTrue(np.array_equal(restored[mask], truth[mask]))
        self.assertTrue(np.array_equal(restored[~mask], source[~mask]))

    def test_short_symmetric_accents_are_not_bridged(self) -> None:
        mask = central_mask()
        background = np.full((120, 160, 3), (246, 243, 234), dtype=np.uint8)
        source = background.copy()
        accent = np.array((245, 145, 5), dtype=np.uint8)
        source[48:55, 28:52] = accent
        source[48:55, 110:134] = accent
        source[mask] = np.array((220, 20, 30), dtype=np.uint8)

        restored, report = _deterministic_poster_restore(source, mask)

        line_report = report["components"][0]["horizontal_line_repair"]
        self.assertEqual(line_report["restored_pixels"], 0)
        self.assertTrue(
            np.all(restored[48:55, 70:92] == np.array((246, 243, 234), dtype=np.uint8))
        )
        self.assertTrue(np.array_equal(restored[~mask], source[~mask]))


class RestorationApiTests(unittest.TestCase):
    def test_explicit_lama_failure_is_not_silently_downgraded(self) -> None:
        image = np.full((48, 64, 3), (230, 220, 210), dtype=np.uint8)
        mask = np.zeros((48, 64), dtype=bool)
        mask[16:32, 22:42] = True
        image[mask] = (20, 40, 210)

        with patch("v5lib.restore._lama_restore", side_effect=RuntimeError("broken")):
            with self.assertRaisesRegex(RuntimeError, "requested explicitly"):
                restore_background(
                    image,
                    [mask],
                    mode="lama",
                    progress=lambda _message: None,
                )

    def test_public_api_preserves_every_pixel_outside_reported_footprint(self) -> None:
        image = np.full((96, 128, 3), (242, 238, 228), dtype=np.uint8)
        visible_mask = np.zeros((96, 128), dtype=bool)
        visible_mask[30:66, 46:82] = True
        image[visible_mask] = np.array((210, 30, 30), dtype=np.uint8)

        result = restore_background(
            image,
            [visible_mask],
            mode="poster",
            progress=lambda _message: None,
        )

        self.assertEqual(result.method, "poster")
        self.assertTrue(result.report["outside_footprint_byte_identical"])
        self.assertTrue(
            np.array_equal(
                result.background[~result.removal_footprint],
                image[~result.removal_footprint],
            )
        )


if __name__ == "__main__":
    unittest.main()
