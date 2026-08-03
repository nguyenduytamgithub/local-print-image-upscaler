from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.upscale_engine import (
    cosine_axis_weight,
    evenly_spaced_starts,
    overlap_sizes,
    padded_source_tensor,
    tile_config_candidates,
)


class TilingTests(unittest.TestCase):
    def test_poster_grid_is_window_aligned(self) -> None:
        starts = evenly_spaced_starts(1280, tile=512, overlap=128)
        self.assertEqual(starts, [0, 384, 768])
        self.assertTrue(all(value % 16 == 0 for value in starts))

    def test_pair_feather_sums_to_one(self) -> None:
        starts = [0, 384]
        _, right = overlap_sizes(starts, 512, 0)
        left, _ = overlap_sizes(starts, 512, 1)
        first = cosine_axis_weight(512, 0, right)[-right:]
        second = cosine_axis_weight(512, left, 0)[:left]
        self.assertTrue(torch.allclose(first + second, torch.ones_like(first), atol=1e-6))
        self.assertGreater(float(first.min()), 0.0)
        self.assertGreater(float(second.min()), 0.0)

    def test_oom_fallback_scales_overlap(self) -> None:
        self.assertEqual(
            tile_config_candidates(512, alignment=16, overlap=128)[:3],
            [(512, 128), (384, 96), (320, 80)],
        )

    def test_tiny_image_can_be_padded_to_full_tile(self) -> None:
        tensor, original = padded_source_tensor(Image.new("RGB", (63, 88), "red"), 512, 16)
        self.assertEqual(original, (63, 88))
        self.assertEqual(tuple(tensor.shape), (1, 3, 512, 512))
        self.assertTrue(torch.isfinite(tensor).all())

    def test_single_pixel_axis_uses_safe_fallback(self) -> None:
        tensor, original = padded_source_tensor(Image.new("RGB", (1, 7), "blue"), 64, 16)
        self.assertEqual(original, (1, 7))
        self.assertEqual(tuple(tensor.shape), (1, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
