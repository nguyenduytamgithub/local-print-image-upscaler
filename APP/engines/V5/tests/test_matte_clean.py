from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


V5_DIR = Path(__file__).resolve().parents[1]
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))

from v5lib.matte_clean import (  # noqa: E402
    clean_layer_semantic_mask,
    clean_semantic_layer_masks,
    clean_text_layer_mask,
)
from v5lib.model import LayerSpec  # noqa: E402


def _rectangle(
    shape: tuple[int, int], x0: int, y0: int, x1: int, y1: int
) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


class SemanticMatteCleanupTests(unittest.TestCase):
    def test_convex_negative_core_removes_diagonal_multitone_panel_leaks(self) -> None:
        shape = (76, 152)
        image = np.full((*shape, 3), (188, 16, 14), dtype=np.uint8)
        source_ink = np.zeros(shape, dtype=np.uint8)
        cv2.ellipse(source_ink, (42, 42), (24, 28), 0, 0, 360, 1, 8, cv2.LINE_8)
        cv2.ellipse(source_ink, (108, 42), (24, 28), 0, 32, 328, 1, 8, cv2.LINE_8)
        cv2.line(source_ink, (108, 42), (132, 42), 1, 7, cv2.LINE_8)
        ink = source_ink.astype(bool)
        image[ink] = (250, 218, 16)
        proposed = ink.copy()

        # Pick a diagonal-only neighbour inside each glyph's convex envelope.
        # It is just ~sqrt(2) from true ink, so distance-based outline support
        # alone keeps it, yet its source colour is merely a dark panel variation.
        target_points: list[tuple[int, int]] = []
        count, labels = cv2.connectedComponents(source_ink, 8)
        cross = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        square = np.ones((3, 3), dtype=np.uint8)
        for component_id in range(1, count):
            component = labels == component_id
            ys, xs = np.where(component)
            points = np.column_stack((xs, ys)).astype(np.int32)
            hull = cv2.convexHull(points)
            hull_mask = np.zeros(shape, dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 1)
            diagonal = (
                hull_mask.astype(bool)
                & cv2.dilate(component.astype(np.uint8), square).astype(bool)
                & ~cv2.dilate(component.astype(np.uint8), cross).astype(bool)
            )
            candidates = np.argwhere(diagonal)
            if not len(candidates):
                continue
            y, x = map(int, candidates[len(candidates) // 2])
            target_points.append((y, x))
            proposed[y, x] = True
            image[y, x] = (92, 4, 5)

        layer = LayerSpec(
            "multitone-og",
            "Multitone O/G",
            "text_raster",
            proposed,
            0.99,
            metadata={
                "grouping_role": "visual_text_row",
                "visual_bbox": [8, 7, 145, 73],
            },
        )

        cleaned, report = clean_text_layer_mask(layer, image_rgb=image)

        pixel_report = report["negative_space_refinement"]
        self.assertEqual(pixel_report["status"], "refined_colour_topology", report)
        self.assertGreaterEqual(len(target_points), 2)
        for y, x in target_points:
            self.assertFalse(cleaned.mask[y, x], (y, x, pixel_report))
        self.assertGreaterEqual(
            pixel_report["convex_negative_component_count"], len(target_points)
        )
        self.assertGreaterEqual(
            pixel_report["convex_negative_space_pixels"], len(target_points)
        )
        self.assertTrue(cleaned.mask[42, 18])  # O outer stroke remains.
        self.assertTrue(cleaned.mask[42, 120])  # G bar remains.

    def test_colour_topology_removes_o_g_n_negative_space_and_keeps_accent(self) -> None:
        shape = (96, 230)
        background = np.array((8, 92, 30), dtype=np.uint8)
        image = np.empty((*shape, 3), dtype=np.uint8)
        image[:] = background
        ink = (247, 244, 232)
        true_ink = np.zeros(shape, dtype=np.uint8)

        # O: a thick real stroke, but the proposal wrongly owns its counter.
        cv2.ellipse(true_ink, (42, 57), (27, 31), 0, 0, 360, 255, 12, cv2.LINE_AA)
        # G: an open curved stroke plus its horizontal bar.
        cv2.ellipse(true_ink, (112, 57), (27, 31), 0, 35, 330, 255, 12, cv2.LINE_AA)
        cv2.line(true_ink, (112, 57), (137, 57), 255, 11, cv2.LINE_AA)
        # N and a detached Vietnamese-like accent component.
        cv2.line(true_ink, (158, 27), (158, 87), 255, 12, cv2.LINE_AA)
        cv2.line(true_ink, (158, 28), (207, 86), 255, 12, cv2.LINE_AA)
        cv2.line(true_ink, (207, 27), (207, 87), 255, 12, cv2.LINE_AA)
        cv2.ellipse(true_ink, (183, 13), (5, 3), 0, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.ellipse(true_ink, (42, 13), (6, 3), 0, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.line(true_ink, (179, 5), (186, 5), 255, 1, cv2.LINE_8)
        image[true_ink > 24] = ink

        proposed = true_ink > 0
        proposed[8:19, 33:52] = False  # a compact accent missed entirely by SAM
        proposed[5, 179:187] = False  # a valid 8x1 mark, not a long panel rule
        # These are source-background pixels accidentally swallowed by unioned
        # SAM/OCR proposals: O counter, G interior/aperture and both N gaps.
        cv2.ellipse(proposed.view(np.uint8), (42, 57), (17, 20), 0, 0, 360, 1, -1)
        cv2.ellipse(proposed.view(np.uint8), (112, 57), (16, 19), 0, 0, 360, 1, -1)
        proposed[45:70, 130:142] = True
        proposed[31:47, 169:180] = True
        proposed[68:83, 188:199] = True
        layer = LayerSpec(
            "ogn",
            "OGN",
            "text_raster",
            proposed,
            0.99,
            metadata={
                "grouping_role": "ocr_text_line",
                "ocr_expanded_bbox": [5, 4, 224, 92],
            },
        )

        cleaned, report = clean_text_layer_mask(layer, image_rgb=image)

        self.assertEqual(
            report["negative_space_refinement"]["status"],
            "refined_colour_topology",
            report,
        )
        self.assertFalse(cleaned.mask[57, 42])  # O counter
        self.assertFalse(cleaned.mask[42, 112])  # G interior
        self.assertFalse(cleaned.mask[49, 141])  # G aperture beside its real bar
        complement = (~cleaned.mask[4:92, 5:224]).astype(np.uint8)
        _count, labels = cv2.connectedComponents(complement, 4)
        g_inner_label = int(labels[42 - 4, 112 - 5])
        border_labels = set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1])
        self.assertIn(g_inner_label, border_labels)  # G must be open, not a second O.
        self.assertFalse(cleaned.mask[38, 178])  # upper N negative space
        self.assertFalse(cleaned.mask[80, 190])  # lower N negative space
        self.assertTrue(cleaned.mask[57, 20])  # O stroke
        self.assertTrue(cleaned.mask[57, 112 + 12], report)  # G bar
        self.assertTrue(cleaned.mask[13, 183])  # detached accent
        self.assertTrue(cleaned.mask[13, 42])  # entirely missed accent recovered from source
        self.assertTrue(cleaned.mask[5, 182])  # tiny horizontal accent survives aspect filtering
        self.assertTrue(cleaned.mask[55, 158])  # N left stem
        self.assertTrue(cleaned.mask[55, 207])  # N right stem
        self.assertEqual(
            report["negative_space_refinement"]["ink_core_recall"], 1.0
        )
        self.assertGreater(
            report["negative_space_refinement"]["removed_negative_space_pixels"],
            0,
        )
        self.assertGreaterEqual(
            report["negative_space_refinement"]["recovered_missing_accent_component_count"],
            1,
        )

    def test_colour_topology_validates_image_contract(self) -> None:
        mask = _rectangle((20, 30), 5, 5, 25, 15)
        layer = LayerSpec(
            "text",
            "Text",
            "text_raster",
            mask,
            0.9,
            metadata={"grouping_role": "ocr_text_line"},
        )
        with self.assertRaises(ValueError):
            clean_text_layer_mask(layer, image_rgb=np.zeros((20, 30, 3), np.float32))
        with self.assertRaises(ValueError):
            clean_text_layer_mask(layer, image_rgb=np.zeros((19, 30, 3), np.uint8))

    def test_visual_baseline_removes_touching_wide_lobe_but_keeps_q_tail(self) -> None:
        shape = (84, 300)
        image = np.full((*shape, 3), (7, 88, 28), dtype=np.uint8)
        mask = np.zeros(shape, dtype=bool)
        for x0 in (30, 72, 114, 156, 198, 240):
            mask[20:60, x0 : x0 + 20] = True
        # Same-colour bow lobe physically touches the second glyph, while the
        # narrow tail below the fifth glyph represents a legitimate Q-like
        # descender. Connected-component filtering alone cannot separate them.
        mask[58:70, 56:108] = True
        mask[58:70, 204:212] = True
        image[mask] = (250, 218, 15)
        # A nearby different-colour decoration also entered the proposal.
        mask[38:53, 3:14] = True
        image[38:53, 3:14] = (245, 116, 8)
        layer = LayerSpec(
            "visual-row",
            "Visual row",
            "text_raster",
            mask,
            0.95,
            metadata={
                "grouping_role": "visual_text_row",
                "visual_bbox": [0, 4, 296, 78],
            },
        )

        cleaned, report = clean_text_layer_mask(layer, image_rgb=image)

        pixel = report["negative_space_refinement"]
        self.assertEqual(pixel["status"], "refined_colour_topology", report)
        self.assertTrue(cleaned.mask[40, 35])
        self.assertFalse(cleaned.mask[66, 64])  # wide touching lobe removed
        self.assertTrue(cleaned.mask[66, 208])  # narrow Q-like tail retained
        self.assertFalse(cleaned.mask[45, 8])  # separate decoration removed
        self.assertGreaterEqual(pixel["baseline_removed_component_count"], 1)
        self.assertGreater(pixel["baseline_removed_pixels"], 0)
        self.assertGreaterEqual(pixel["removed_nontext_component_count"], 1)

    def test_recovers_exact_component_and_removes_unanchored_islands(self) -> None:
        shape = (24, 32)
        raw = np.zeros(shape, dtype=bool)
        raw[1:3, 1:3] = True  # cc_001: unrelated island before the object.
        raw[6:18, 8:24] = True  # cc_002: intended object.
        raw[20:22, 28:30] = True  # cc_003: unrelated island after it.
        intended = _rectangle(shape, 8, 6, 24, 18)
        expanded = cv2.dilate(
            intended.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)
        ).astype(bool)
        expanded[1:3, 1:3] = True
        expanded[20:22, 28:30] = True
        original_before = expanded.copy()
        raw_before = raw.copy()
        layer = LayerSpec(
            "icon",
            "Icon",
            "object",
            expanded,
            0.99,
            source_ids=[0],
            metadata={
                "grouping_role": "promoted_child_object",
                "source_components": ["sam_0000_cc_002"],
            },
        )

        cleaned, report = clean_layer_semantic_mask(layer, [raw])

        self.assertTrue(np.array_equal(cleaned.mask, intended))
        self.assertFalse(np.logical_and(cleaned.mask, ~raw).any())
        self.assertEqual(report["status"], "cleaned_from_recorded_components")
        self.assertEqual(report["resolved_source_components"], ["sam_0000_cc_002"])
        self.assertGreater(report["removed_unanchored_area"], 0)
        self.assertEqual(report["outside_semantic_source_area"], 0)
        self.assertEqual(report["cleaned_component_count"], 1)
        self.assertTrue(np.array_equal(layer.mask, original_before))
        self.assertTrue(np.array_equal(raw, raw_before))
        self.assertIsNot(cleaned.metadata, layer.metadata)

    def test_text_cleanup_removes_panel_rules_but_keeps_glyphs_and_diacritics(self) -> None:
        shape = (70, 180)
        mask = np.zeros(shape, dtype=bool)
        mask[10:13, 5:175] = True  # panel border above the text
        mask[58:61, 5:175] = True  # panel border below the text
        mask[25:52, 30:45] = True  # first glyph body
        mask[20:23, 34:40] = True  # separate Vietnamese accent
        mask[25:52, 70:88] = True  # second glyph body
        mask[4, 160] = True  # isolated speck
        layer = LayerSpec(
            "text",
            "Text line",
            "text_raster",
            mask,
            0.95,
            metadata={
                "grouping_role": "ocr_text_line",
                "ocr_expanded_bbox": [5, 4, 175, 62],
            },
        )

        cleaned, report = clean_text_layer_mask(layer)

        self.assertFalse(cleaned.mask[10:13, 5:175].any())
        self.assertFalse(cleaned.mask[58:61, 5:175].any())
        self.assertTrue(cleaned.mask[25:52, 30:45].all())
        self.assertTrue(cleaned.mask[20:23, 34:40].all())
        self.assertTrue(cleaned.mask[25:52, 70:88].all())
        self.assertFalse(cleaned.mask[4, 160])
        self.assertEqual(report["removed_frame_component_count"], 2)
        self.assertEqual(report["removed_unanchored_component_count"], 1)

    def test_symmetric_decoration_preserves_each_explicit_component_only(self) -> None:
        shape = (30, 48)
        raw = np.zeros(shape, dtype=bool)
        left = _rectangle(shape, 3, 8, 12, 22)
        middle_noise = _rectangle(shape, 22, 2, 25, 5)
        right = _rectangle(shape, 36, 8, 45, 22)
        raw |= left | middle_noise | right
        proposed = cv2.dilate(
            (left | middle_noise | right).astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
        ).astype(bool)
        layer = LayerSpec(
            "decoration",
            "Decoration",
            "object",
            proposed,
            0.98,
            source_ids=[0],
            metadata={
                "grouping_role": "symmetric_decoration",
                "source_components": ["sam_0000_cc_002", "sam_0000_cc_003"],
            },
        )

        cleaned, report = clean_layer_semantic_mask(layer, np.stack([raw]))

        self.assertTrue(np.array_equal(cleaned.mask, left | right))
        count, _labels = cv2.connectedComponents(cleaned.mask.astype(np.uint8), 8)
        self.assertEqual(count - 1, 2)
        self.assertFalse(np.logical_and(cleaned.mask, middle_noise).any())
        self.assertEqual(report["cleaned_component_count"], 2)
        self.assertEqual(report["outside_semantic_source_area"], 0)

    def test_invalid_provenance_is_a_conservative_no_op(self) -> None:
        shape = (12, 14)
        original = _rectangle(shape, 2, 3, 10, 9)
        layer = LayerSpec(
            "object",
            "Object",
            "object",
            original,
            0.9,
            metadata={
                "grouping_role": "standalone_object",
                "source_components": ["sam_9999_cc_001", "bad-token"],
            },
        )

        cleaned, report = clean_layer_semantic_mask(layer, [original])

        self.assertTrue(np.array_equal(cleaned.mask, original))
        self.assertEqual(report["status"], "skipped_no_resolved_source_component")
        self.assertEqual(
            report["unresolved_source_components"],
            ["sam_9999_cc_001", "bad-token"],
        )
        self.assertFalse(report["changed"])

    def test_batch_keeps_non_object_roles_unchanged_and_reports_totals(self) -> None:
        shape = (18, 24)
        source = _rectangle(shape, 5, 5, 12, 13)
        expanded = cv2.dilate(
            source.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
        ).astype(bool)
        object_layer = LayerSpec(
            "child",
            "Child",
            "object",
            expanded,
            0.99,
            source_ids=[0],
            metadata={
                "grouping_role": "promoted_child_object",
                "source_components": ["sam_0000_cc_001"],
            },
        )
        text_mask = _rectangle(shape, 14, 7, 22, 11)
        text_layer = LayerSpec(
            "text",
            "Text",
            "text_raster",
            text_mask,
            0.95,
            metadata={"grouping_role": "ocr_text_line"},
        )

        cleaned, report = clean_semantic_layer_masks(
            [object_layer, text_layer], [source]
        )

        self.assertTrue(np.array_equal(cleaned[0].mask, source))
        self.assertTrue(np.array_equal(cleaned[1].mask, text_mask))
        self.assertEqual(report["input_layer_count"], 2)
        self.assertEqual(report["cleaned_layer_count"], 1)
        self.assertEqual(report["changed_layer_count"], 1)
        self.assertEqual(report["skipped_layer_count"], 1)
        self.assertGreater(report["removed_unanchored_area"], 0)


if __name__ == "__main__":
    unittest.main()
