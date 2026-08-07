from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


V5_DIR = Path(__file__).resolve().parents[1]
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))

from v5lib.matting import (  # noqa: E402
    _component_counter_count,
    _prune_unanchored_alpha,
    _stabilize_source_colour_alpha,
    make_trimap,
)


class ViTMatteTrimapTests(unittest.TestCase):
    def test_source_colour_guard_restores_accent_and_removes_panel_speck(self) -> None:
        shape = (48, 92)
        image = np.full((*shape, 3), (6, 72, 18), dtype=np.uint8)
        mask = np.zeros(shape, dtype=bool)
        mask[14:42, 12:28] = True
        mask[14:42, 42:58] = True
        mask[5:9, 17:24] = True  # detached Vietnamese-style accent
        mask[8, 82] = True  # a seeded panel-colour speck
        mask[3, 80] = True  # isolated one-pixel palette-colour fragment
        image[mask] = (248, 220, 18)
        image[5, 17] = (170, 173, 1)  # darker edge with the same yellow hue
        image[8, 82] = (5, 68, 17)
        alpha = np.zeros(shape, dtype=np.float32)
        alpha[14:42, 12:28] = 0.92
        alpha[14:42, 42:58] = 0.92
        alpha[6:8, 19:22] = 0.8  # model kept only the middle of the accent
        alpha[8, 82] = 1.0
        alpha[3, 80] = 1.0
        sure = np.zeros(shape, dtype=bool)
        sure[18:38, 16:24] = True
        sure[18:38, 46:54] = True
        sure[6, 20] = True
        sure[8, 82] = True
        sure[3, 80] = True

        cleaned, report = _stabilize_source_colour_alpha(
            image, mask, alpha, sure
        )

        self.assertTrue((cleaned[5:9, 17:24] >= 0.5).all(), report)
        self.assertEqual(float(cleaned[8, 82]), 0.0, report)
        self.assertEqual(float(cleaned[3, 80]), 0.0, report)
        self.assertGreater(report["alpha_floor_raised_pixels"], 0)
        self.assertGreaterEqual(
            report["removed_foreign_alpha_component_count"], 1
        )

    def test_source_colour_guard_validates_contract(self) -> None:
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        mask = np.ones((5, 7), dtype=bool)
        alpha = np.ones((5, 7), dtype=np.float32)
        sure = np.zeros((5, 7), dtype=bool)
        with self.assertRaisesRegex(ValueError, "uint8 RGB"):
            _stabilize_source_colour_alpha(image.astype(np.float32), mask, alpha, sure)
        with self.assertRaisesRegex(ValueError, "share a shape"):
            _stabilize_source_colour_alpha(image, mask[:, :-1], alpha, sure)
        bad = alpha.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite values"):
            _stabilize_source_colour_alpha(image, mask, bad, sure)
        outside = sure.copy()
        invalid_mask = mask.copy()
        invalid_mask[0, 0] = False
        outside[0, 0] = True
        with self.assertRaisesRegex(ValueError, "cannot escape"):
            _stabilize_source_colour_alpha(image, invalid_mask, alpha, outside)

    def test_unanchored_alpha_islands_are_pruned_but_detached_accent_survives(self) -> None:
        alpha = np.zeros((14, 24), dtype=np.float32)
        alpha[3:11, 2:8] = 0.8
        alpha[5:9, 3:7] = 1.0
        alpha[1, 12] = 1.0  # legitimate detached accent
        alpha[10:12, 19:21] = 0.42  # model-only island with no ink anchor
        sure_foreground = np.zeros(alpha.shape, dtype=bool)
        sure_foreground[6, 5] = True
        sure_foreground[1, 12] = True

        cleaned, removed_components, removed_pixels = _prune_unanchored_alpha(
            alpha, sure_foreground
        )

        self.assertTrue((cleaned[3:11, 2:8] > 0).all())
        self.assertEqual(float(cleaned[1, 12]), 1.0)
        self.assertFalse(cleaned[10:12, 19:21].any())
        self.assertEqual(removed_components, 1)
        self.assertEqual(removed_pixels, 4)

    def test_o_counter_keeps_a_certain_background_core(self) -> None:
        mask = np.zeros((41, 41), dtype=np.uint8)
        cv2.circle(mask, (20, 20), 15, 1, thickness=-1)
        cv2.circle(mask, (20, 20), 8, 0, thickness=-1)
        binary = mask.astype(bool)

        trimap, sure_foreground = make_trimap(binary, radius=2)

        self.assertEqual(_component_counter_count(binary), 1)
        self.assertEqual(int(trimap[20, 20]), 0)
        self.assertFalse(sure_foreground[20, 20])
        self.assertFalse(sure_foreground[~binary].any())
        # A trimap may mark a narrow unknown band on both sides of the contour,
        # but it must retain certain background safely inside a substantial O.
        counter_core = np.zeros_like(binary)
        cv2.circle(counter_core.view(np.uint8), (20, 20), 4, 1, thickness=-1)
        self.assertTrue((trimap[counter_core] == 0).all())
        self.assertTrue(set(np.unique(trimap)).issubset({0, 128, 255}))

    def test_every_thin_component_retains_sure_foreground(self) -> None:
        mask = np.zeros((36, 52), dtype=bool)
        mask[8:29, 5:13] = True  # glyph body survives ordinary erosion.
        mask[12:13, 22:39] = True  # one-pixel horizontal stroke.
        mask[4, 45] = True  # detached one-pixel Vietnamese-style accent.

        trimap, sure_foreground = make_trimap(mask, radius=3)

        count, labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
        self.assertEqual(count - 1, 3)
        for component_id in range(1, count):
            with self.subTest(component_id=component_id):
                component = labels == component_id
                self.assertTrue(np.logical_and(component, sure_foreground).any())
        self.assertFalse(np.logical_and(sure_foreground, ~mask).any())
        self.assertTrue((trimap[sure_foreground] == 255).all())
        self.assertEqual(int(trimap[4, 45]), 255)

    def test_empty_mask_returns_typed_empty_outputs(self) -> None:
        mask = np.zeros((7, 11), dtype=bool)

        trimap, sure_foreground = make_trimap(mask, radius=1)

        self.assertEqual(trimap.shape, mask.shape)
        self.assertEqual(sure_foreground.shape, mask.shape)
        self.assertEqual(trimap.dtype, np.uint8)
        self.assertEqual(sure_foreground.dtype, np.bool_)
        self.assertFalse(trimap.any())
        self.assertFalse(sure_foreground.any())

    def test_rejects_non_2d_masks(self) -> None:
        for mask in (
            np.zeros(12, dtype=bool),
            np.zeros((4, 5, 1), dtype=bool),
        ):
            with self.subTest(shape=mask.shape):
                with self.assertRaisesRegex(ValueError, "two-dimensional"):
                    make_trimap(mask, radius=1)

    def test_rejects_invalid_radius_values(self) -> None:
        mask = np.ones((5, 5), dtype=bool)
        for radius in (0, -1, True, 1.0, "1", None):
            with self.subTest(radius=radius):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    make_trimap(mask, radius=radius)  # type: ignore[arg-type]

    def test_alpha_pruning_validates_shapes_finiteness_and_anchor_ownership(self) -> None:
        alpha = np.ones((4, 5), dtype=np.float32)
        anchors = np.zeros(alpha.shape, dtype=bool)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            _prune_unanchored_alpha(alpha[..., None], anchors)
        with self.assertRaisesRegex(ValueError, "same shape"):
            _prune_unanchored_alpha(alpha, anchors[:, :-1])
        bad = alpha.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            _prune_unanchored_alpha(bad, anchors)
        alpha[2, 3] = 0.0
        anchors[2, 3] = True
        with self.assertRaisesRegex(ValueError, "positive alpha"):
            _prune_unanchored_alpha(alpha, anchors)


if __name__ == "__main__":
    unittest.main()
