from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


V7_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7_DIR))

from v7lib.masks import build_text_mask  # noqa: E402


class TextMaskIsolationTests(unittest.TestCase):
    def test_different_colour_icon_inside_loose_ocr_box_is_not_removed(self) -> None:
        image = np.full((100, 260, 3), (244, 236, 211), dtype=np.uint8)
        icon = np.zeros(image.shape[:2], dtype=np.uint8)
        glyphs = np.zeros_like(icon)

        cv2.circle(icon, (43, 50), 16, 1, thickness=-1)
        for x in range(78, 224, 18):
            cv2.rectangle(glyphs, (x, 34), (x + 10, 67), 1, thickness=-1)

        icon_pixels = icon.astype(bool)
        glyph_pixels = glyphs.astype(bool)
        image[icon_pixels] = (205, 35, 42)  # red icon
        image[glyph_pixels] = (18, 94, 50)  # repeated green text ink

        result = build_text_mask(image, (20, 25, 238, 75))

        glyph_coverage = float(result.mask[glyph_pixels].mean())
        icon_leak = float(result.mask[icon_pixels].mean())
        self.assertGreater(glyph_coverage, 0.90, result.report)
        self.assertLess(icon_leak, 0.02, result.report)
        self.assertGreaterEqual(result.quality, 0.45, result.report)


if __name__ == "__main__":
    unittest.main()
