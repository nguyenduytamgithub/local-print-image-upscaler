from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


V7_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7_DIR))

from v7lib.background import (  # noqa: E402
    BackgroundRestoreError,
    restore_text_background,
)


class BackgroundRestoreTests(unittest.TestCase):
    def test_flat_poster_background_is_rebuilt_and_scope_is_exact(self) -> None:
        original = np.full((96, 144, 3), (236, 225, 188), dtype=np.uint8)
        damaged = original.copy()
        mask = np.zeros(original.shape[:2], dtype=bool)
        mask[30:64, 40:104] = True
        damaged[mask] = (188, 20, 20)

        result = restore_text_background(
            damaged,
            mask,
            mode="auto",
            padding=2,
            ring_radius=18,
        )

        np.testing.assert_array_equal(
            result.background[~result.removal_footprint],
            damaged[~result.removal_footprint],
        )
        np.testing.assert_array_equal(
            result.background[result.removal_footprint],
            original[result.removal_footprint],
        )
        self.assertEqual(result.method, "validated_surface_constant")
        self.assertTrue(result.report["outside_footprint_byte_identical"])
        self.assertTrue(result.report["synthesized"])
        self.assertFalse(result.report["recovered_original_pixels_claimed"])

    def test_linear_gradient_is_reconstructed_with_low_error(self) -> None:
        height, width = 120, 180
        yy, xx = np.mgrid[:height, :width]
        original = np.empty((height, width, 3), dtype=np.uint8)
        original[..., 0] = np.rint(64 + xx * 0.35 + yy * 0.11).astype(np.uint8)
        original[..., 1] = np.rint(116 + xx * 0.18 + yy * 0.08).astype(np.uint8)
        original[..., 2] = np.rint(202 - xx * 0.22 + yy * 0.05).astype(np.uint8)
        damaged = original.copy()
        mask = np.zeros((height, width), dtype=bool)
        mask[35:83, 53:131] = True
        damaged[mask] = (10, 10, 10)

        result = restore_text_background(
            damaged,
            [mask],
            mode="auto",
            padding=3,
            ring_radius=28,
        )

        error = np.mean(
            np.abs(
                result.background[result.removal_footprint].astype(np.int16)
                - original[result.removal_footprint].astype(np.int16)
            )
        )
        self.assertLess(error, 1.0)
        self.assertIn(result.method, {"validated_surface_linear", "validated_surface_quadratic"})
        self.assertGreater(result.confidence, 0.9)

    def test_strict_mode_refuses_unvalidated_texture(self) -> None:
        random = np.random.default_rng(1729)
        source = random.integers(0, 256, size=(80, 112, 3), dtype=np.uint8)
        mask = np.zeros(source.shape[:2], dtype=bool)
        mask[24:56, 35:78] = True

        result = restore_text_background(
            source,
            mask,
            mode="strict",
            padding=2,
            ring_radius=16,
            max_surface_mae=0.0,
        )

        np.testing.assert_array_equal(result.background, source)
        self.assertEqual(result.method, "unchanged_unvalidated_context")
        self.assertEqual(result.confidence, 0.0)
        self.assertFalse(result.report["synthesized"])

    def test_opencv_fallback_is_deterministic(self) -> None:
        random = np.random.default_rng(444)
        source = random.integers(0, 256, size=(72, 96, 3), dtype=np.uint8)
        mask = np.zeros(source.shape[:2], dtype=bool)
        mask[25:48, 31:67] = True

        first = restore_text_background(source, mask, mode="opencv", padding=2)
        second = restore_text_background(source, mask, mode="opencv", padding=2)

        np.testing.assert_array_equal(first.background, second.background)
        np.testing.assert_array_equal(first.removal_footprint, second.removal_footprint)
        np.testing.assert_array_equal(
            first.background[~first.removal_footprint], source[~first.removal_footprint]
        )
        self.assertEqual(first.method, "opencv_telea_synthesized")

    def test_explicit_opencv_mode_is_honoured_on_a_flat_background(self) -> None:
        source = np.full((48, 64, 3), (200, 210, 220), dtype=np.uint8)
        mask = np.zeros(source.shape[:2], dtype=bool)
        mask[15:34, 19:45] = True
        source[mask] = (10, 30, 50)

        result = restore_text_background(source, mask, mode="opencv", padding=1)

        self.assertEqual(result.method, "opencv_telea_synthesized")

    def test_disconnected_regions_use_their_own_colour_context(self) -> None:
        original = np.empty((100, 180, 3), dtype=np.uint8)
        original[:, :90] = (242, 231, 197)
        original[:, 90:] = (31, 107, 56)
        damaged = original.copy()
        left = np.zeros(original.shape[:2], dtype=bool)
        right = np.zeros_like(left)
        left[28:68, 20:70] = True
        right[28:68, 110:160] = True
        damaged[left] = (180, 20, 20)
        damaged[right] = (250, 250, 250)

        result = restore_text_background(
            damaged,
            [left, right],
            mode="auto",
            padding=2,
            ring_radius=14,
        )

        np.testing.assert_array_equal(
            result.background[result.removal_footprint],
            original[result.removal_footprint],
        )
        self.assertEqual(result.report["component_count"], 2)
        self.assertEqual(result.method, "validated_surface_constant")

    def test_empty_mask_is_a_lossless_noop(self) -> None:
        source = np.full((20, 30, 3), 91, dtype=np.uint8)
        result = restore_text_background(source, np.zeros((20, 30), dtype=np.uint8))
        np.testing.assert_array_equal(result.background, source)
        self.assertEqual(result.method, "unchanged_empty_mask")
        self.assertFalse(result.report["synthesized"])

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaises(BackgroundRestoreError):
            restore_text_background(
                np.zeros((8, 8, 3), dtype=np.float32),
                np.zeros((8, 8), dtype=bool),
            )
        with self.assertRaises(BackgroundRestoreError):
            restore_text_background(
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((7, 8), dtype=bool),
            )


if __name__ == "__main__":
    unittest.main()
