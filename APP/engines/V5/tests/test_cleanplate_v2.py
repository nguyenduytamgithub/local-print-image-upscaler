from __future__ import annotations

import unittest

import numpy as np

from v5pro.cleanplate import (
    _robust_global_poster_surface,
    analyse_clean_plate,
    build_clean_plate,
)
from v5pro.schema import AlphaCrop, DocumentGraph, ElementNode


class CleanPlateTests(unittest.TestCase):

    def test_source_remainder_is_not_a_cleanplate_removal_request(self) -> None:
        source = np.full((48, 72, 3), (248, 238, 190), dtype=np.uint8)
        source[15:34, 20:55] = (25, 120, 45)
        surface = np.full_like(source, (248, 238, 190))
        graph = DocumentGraph((72, 48))
        object_alpha = AlphaCrop(20, 15, np.full((19, 35), 255, dtype=np.uint8))
        graph.add_node(
            ElementNode(
                "OBJECT",
                "Object",
                "product",
                object_alpha,
                10,
                full_support=object_alpha,
                review_status="auto_confirmed",
                rgba=np.dstack([source[15:34, 20:55], object_alpha.alpha]),
            )
        )
        remainder_mask = np.full((48, 72), 255, dtype=np.uint8)
        remainder_mask[15:34, 20:55] = 0
        remainder = AlphaCrop(0, 0, remainder_mask)
        graph.add_node(
            ElementNode(
                "BASE_SOURCE_RESIDUAL_0001",
                "Source remainder",
                "unknown",
                remainder,
                -3_000_000,
                full_support=remainder,
                review_status="unresolved",
                rgba=np.dstack([source, remainder.alpha]),
                metadata={"role": "exact_source_remainder_above_clean_base"},
            )
        )

        result = build_clean_plate(
            source, graph, mode="poster", poster_surface_rgb=surface
        )

        self.assertEqual(result.report["requested_union_pixels"], 19 * 35)
        self.assertEqual(result.report["clean_plate_audit"]["grade"], "pass")
        self.assertTrue(np.array_equal(result.background_rgb, surface))

    def test_reconciled_poster_surface_beats_local_ghost_copy(self) -> None:
        height, width = 48, 72
        source = np.full((height, width, 3), (248, 238, 190), dtype=np.uint8)
        source[15:34, 20:55] = (25, 120, 45)
        removal = np.zeros((height, width), dtype=np.uint8)
        removal[13:36, 18:57] = 255
        surface = np.full_like(source, (248, 238, 190))
        result = build_clean_plate(
            source,
            removal,
            mode="poster",
            poster_surface_rgb=surface,
        )
        self.assertEqual(result.method, "reconciled_bottom_poster_surface")
        self.assertTrue(np.array_equal(result.background_rgb[20, 30], surface[20, 30]))
        self.assertTrue(np.array_equal(result.background_rgb[0, 0], source[0, 0]))
    def test_global_poster_surface_recovers_large_occluded_gradient(self) -> None:
        height, width = 120, 180
        yy, xx = np.mgrid[0:height, 0:width]
        base = np.dstack(
            [
                225 + 12 * xx / width + 4 * yy / height,
                218 + 8 * xx / width + 9 * yy / height,
                190 + 6 * xx / width + 10 * yy / height,
            ]
        ).astype(np.uint8)
        source = base.copy()
        footprint = np.zeros((height, width), dtype=bool)
        footprint[24:100, 28:154] = True
        source[footprint] = (250, 250, 250)
        source[45:76, 48:137] = (20, 90, 40)
        recovered, report = _robust_global_poster_surface(source, footprint)
        error = np.abs(recovered.astype(np.int16) - base.astype(np.int16))
        self.assertLessEqual(int(error[footprint].max()), 2)
        self.assertGreater(report["inlier_samples"], 256)
        self.assertTrue(np.array_equal(recovered[~footprint], source[~footprint]))
    def test_linear_poster_surface_is_reconstructed_without_outside_writes(self) -> None:
        height, width = 120, 180
        yy, xx = np.indices((height, width))
        truth = np.stack(
            (
                210 + xx * 25 // width,
                224 + yy * 10 // height,
                205 + xx * 12 // width,
            ),
            axis=2,
        ).astype(np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[34:86, 64:116] = 255
        source = truth.copy()
        source[mask > 0] = np.array((216, 27, 36), dtype=np.uint8)

        result = build_clean_plate(source, mask, mode="poster")

        error = np.abs(result.background.astype(np.int16) - truth.astype(np.int16))
        self.assertLessEqual(float(np.percentile(error[result.removal_footprint], 95)), 3.0)
        self.assertTrue(
            np.array_equal(
                result.background[~result.removal_footprint],
                source[~result.removal_footprint],
            )
        )
        self.assertEqual(result.report["clean_plate_audit"]["grade"], "pass")

    def test_vertical_rule_is_continued_through_removed_object(self) -> None:
        truth = np.full((160, 160, 3), (245, 240, 230), dtype=np.uint8)
        truth[:, 77:83] = np.array((20, 100, 40), dtype=np.uint8)
        mask = np.zeros((160, 160), dtype=np.uint8)
        mask[55:105, 55:105] = 255
        source = truth.copy()
        source[mask > 0] = np.array((220, 30, 30), dtype=np.uint8)

        result = build_clean_plate(source, mask, mode="poster")

        structure = result.report["structure_continuation"]
        self.assertGreater(structure["restored_vertical_pixels"], 0)
        exact = np.all(result.background[60:100, 77:83] == truth[60:100, 77:83], axis=2)
        self.assertGreaterEqual(float(np.mean(exact)), 0.80)

    def test_unchanged_fake_clean_plate_is_reported_as_ghost_failure(self) -> None:
        source = np.full((100, 140, 3), (245, 240, 230), dtype=np.uint8)
        mask = np.zeros((100, 140), dtype=bool)
        mask[30:70, 45:95] = True
        source[mask] = np.array((215, 25, 35), dtype=np.uint8)

        report, heatmap = analyse_clean_plate(source, source.copy(), mask)

        self.assertEqual(report["grade"], "fail")
        self.assertGreater(report["source_ink_retained_ratio"], 0.95)
        self.assertGreater(int(np.percentile(heatmap[mask], 75)), 180)

    def test_empty_removal_is_a_noop_with_explicit_report(self) -> None:
        source = np.full((32, 48, 3), (220, 225, 230), dtype=np.uint8)
        result = build_clean_plate(source, np.zeros((32, 48), dtype=np.uint8))
        self.assertEqual(result.method, "none")
        self.assertTrue(result.report["empty_removal"])
        self.assertTrue(np.array_equal(result.background, source))


if __name__ == "__main__":
    unittest.main()
