from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from v5pro.exporter import render_document
from v5pro.hierarchy import assign_spatial_hierarchy
from v5pro.schema import AlphaCrop, DocumentGraph, ElementNode


class ExporterV2Tests(unittest.TestCase):
    def test_cropped_renderer_recomposes_target_without_foreign_pixels(self) -> None:
        source = np.full((24, 30, 3), 230, dtype=np.uint8)
        source[7:17, 9:22] = (220, 25, 30)
        clean = np.full_like(source, 230)
        alpha = np.full((10, 13), 255, dtype=np.uint8)
        crop = AlphaCrop(9, 7, alpha)
        graph = DocumentGraph((30, 24))
        graph.add_node(
            ElementNode(
                "SHAPE_1",
                "Red shape",
                "badge",
                crop,
                1,
                full_support=crop,
                confidence=1.0,
                review_status="auto_confirmed",
                move_safe=True,
                rgba=np.dstack([source[7:17, 9:22], alpha]),
            )
        )
        result = render_document(graph, source, clean, scale=1.0)
        self.assertEqual(result.report["recomposition_max_abs_error"], 0)
        self.assertEqual(len(result.rendered), 1)
        self.assertEqual(result.rendered[0].bbox, (9, 7, 22, 17))

    def test_scale_has_exact_canvas_and_keeps_crop_bounded(self) -> None:
        source = np.full((8, 10, 3), 240, dtype=np.uint8)
        source[2:6, 3:8] = 40
        target = np.asarray(Image.fromarray(source).resize((20, 16)), dtype=np.uint8)
        crop = AlphaCrop(3, 2, np.full((4, 5), 255, dtype=np.uint8))
        graph = DocumentGraph((10, 8))
        graph.add_node(
            ElementNode(
                "A",
                "A",
                "icon",
                crop,
                1,
                full_support=crop,
                confidence=1.0,
                review_status="auto_confirmed",
                rgba=np.dstack([source[2:6, 3:8], crop.alpha]),
            )
        )
        result = render_document(graph, target, np.full_like(source, 240), scale=2.0)
        self.assertEqual(result.composite.size, (20, 16))
        self.assertEqual(result.rendered[0].bbox, (6, 4, 16, 12))

    def test_occluded_parent_exports_synthesized_support_below_child(self) -> None:
        source = np.full((12, 22, 3), 245, dtype=np.uint8)
        source[5:7, 2:20] = (20, 150, 55)
        source[3:9, 8:14] = (220, 35, 45)
        graph = DocumentGraph((22, 12))
        full_line = np.full((2, 18), 255, dtype=np.uint8)
        visible_line = full_line.copy()
        visible_line[:, 6:12] = 0
        line_visible_crop = AlphaCrop(2, 5, visible_line)
        line_full_crop = AlphaCrop(2, 5, full_line)
        graph.add_node(
            ElementNode(
                "LINE",
                "Line",
                "line",
                line_visible_crop,
                1,
                full_support=line_full_crop,
                confidence=1.0,
                review_status="auto_confirmed",
                move_safe=True,
                occluded=True,
                synthesized_hidden_pixels=True,
                rgba=np.dstack(
                    [np.full((2, 18, 3), (20, 150, 55), np.uint8), full_line]
                ),
            )
        )
        badge = AlphaCrop(8, 3, np.full((6, 6), 255, dtype=np.uint8))
        graph.add_node(
            ElementNode(
                "BADGE",
                "Badge",
                "badge",
                badge,
                2,
                parent_id="LINE",
                full_support=badge,
                confidence=1.0,
                review_status="auto_confirmed",
                move_safe=True,
                rgba=np.dstack([source[3:9, 8:14], badge.alpha]),
            )
        )
        result = render_document(graph, source, np.full_like(source, 245), scale=1.0)
        self.assertEqual(result.report["recomposition_max_abs_error"], 0)
        line_rgba = np.asarray(result.rendered[0].rgba, dtype=np.uint8)
        self.assertTrue(np.array_equal(line_rgba[0, 8, :3], (20, 150, 55)))
        self.assertEqual(int(line_rgba[0, 8, 3]), 255)

    def test_attribution_identifies_cross_group_clean_geometry_last_writer(self) -> None:
        source = np.full((12, 20, 3), 245, dtype=np.uint8)
        source[4:7, 4:16] = (235, 119, 18)
        source[4:7, 7:13] = (20, 35, 50)
        clean = np.full_like(source, 245)
        graph = DocumentGraph((20, 12))

        panel_full = np.full((10, 16), 255, dtype=np.uint8)
        panel_visible = panel_full.copy()
        panel_visible[3:6, 6:12] = 0
        graph.add_node(
            ElementNode(
                "RESIDUAL_PANEL",
                "Earlier residual panel",
                "panel",
                AlphaCrop(1, 1, panel_visible),
                10,
                full_support=AlphaCrop(1, 1, panel_full.copy()),
                confidence=0.5,
                review_status="unresolved",
                rgba=np.dstack(
                    [np.full((10, 16, 3), 245, dtype=np.uint8), panel_full]
                ),
            )
        )
        text_alpha = np.full((3, 6), 255, dtype=np.uint8)
        graph.add_node(
            ElementNode(
                "TEXT",
                "Protected text",
                "text",
                AlphaCrop(7, 4, text_alpha),
                20,
                parent_id="RESIDUAL_PANEL",
                full_support=AlphaCrop(7, 4, text_alpha.copy()),
                confidence=0.99,
                review_status="auto_confirmed",
                rgba=np.dstack([source[4:7, 7:13].copy(), text_alpha.copy()]),
            )
        )
        geometry_full = np.full((3, 12), 255, dtype=np.uint8)
        geometry_hidden = np.zeros_like(geometry_full)
        graph.add_node(
            ElementNode(
                "GEO_FRAME",
                "Later clean frame",
                "frame",
                AlphaCrop(4, 4, geometry_hidden),
                40,
                full_support=AlphaCrop(4, 4, geometry_full.copy()),
                confidence=0.95,
                review_status="auto_confirmed",
                move_safe=True,
                rgba=np.dstack(
                    [
                        np.full((3, 12, 3), (235, 119, 18), dtype=np.uint8),
                        geometry_full.copy(),
                    ]
                ),
                metadata={
                    "geometry_backend": "poster_geometry_v2",
                    "geometry_cleanliness": {
                        "policy": "reference_surface_delta_e_carve_v1",
                        "status": "pass",
                        "reference_type": "constant_colour_rgb",
                    },
                },
            )
        )

        broken = render_document(graph, source, clean, scale=1.0)
        attribution = broken.report["recomposition_attribution"]
        self.assertEqual(attribution["different_pixel_count"], 18)
        self.assertEqual(attribution["diff_bbox"], [7, 4, 13, 7])
        self.assertEqual(attribution["categorical_visible_owner_counts"], {"TEXT": 18})
        self.assertEqual(attribution["last_writer_counts"], {"GEO_FRAME": 18})
        self.assertEqual(
            attribution["last_writer_hidden_full_support_counts"],
            {"GEO_FRAME": 18},
        )
        self.assertEqual(
            attribution["solver_feasibility"]["infeasible_pixel_events"],
            0,
        )
        self.assertEqual(
            attribution["sample_coordinates"][0]["categorical_visible_owner_ids"],
            ["TEXT"],
        )
        self.assertEqual(
            attribution["sample_coordinates"][0]["last_writer_id"],
            "GEO_FRAME",
        )
        self.assertTrue(
            attribution["sample_coordinates"][0][
                "last_writer_uses_hidden_full_support"
            ]
        )

        assign_spatial_hierarchy(graph)
        self.assertEqual(graph.node_map()["TEXT"].parent_id, "GEO_FRAME")
        repaired = render_document(graph, source, clean, scale=1.0)
        self.assertEqual(repaired.report["recomposition_max_abs_error"], 0)
        self.assertEqual(
            repaired.report["recomposition_attribution"]["different_pixel_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
