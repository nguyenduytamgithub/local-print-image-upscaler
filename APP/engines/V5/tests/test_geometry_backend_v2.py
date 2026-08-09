from __future__ import annotations

import unittest

import cv2
import numpy as np

from v5pro.geometry_backend import extract_geometry_layers
from v5pro.inventory import DetectedProposal
from v5pro.schema import AlphaCrop, ProposalRecord


def proposal(
    proposal_id: str,
    kind: str,
    bbox: tuple[int, int, int, int],
    confidence: float = 0.92,
    mask: np.ndarray | None = None,
) -> DetectedProposal:
    hint = AlphaCrop(0, 0, mask) if mask is not None else None
    return DetectedProposal(
        ProposalRecord(proposal_id, "synthetic_test", kind, bbox, confidence),  # type: ignore[arg-type]
        hint,
    )


def rounded_rectangle(
    shape: tuple[int, int], bbox: tuple[int, int, int, int], radius: int
) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = bbox
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (x0 + radius, y0), (x1 - radius - 1, y1 - 1), 255, -1)
    cv2.rectangle(mask, (x0, y0 + radius), (x1 - 1, y1 - radius - 1), 255, -1)
    for x, y in (
        (x0 + radius, y0 + radius),
        (x1 - radius - 1, y0 + radius),
        (x0 + radius, y1 - radius - 1),
        (x1 - radius - 1, y1 - radius - 1),
    ):
        cv2.circle(mask, (x, y), radius, 255, -1, cv2.LINE_AA)
    return mask


class GeometryBackendTests(unittest.TestCase):
    def test_crossing_geometry_does_not_retain_duplicate_hidden_support(self) -> None:
        image = np.full((80, 120, 3), 245, dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (109, 69), (20, 130, 45), 2, cv2.LINE_AA)
        cv2.line(image, (20, 40), (99, 40), (20, 130, 45), 2, cv2.LINE_AA)
        result = extract_geometry_layers(
            image,
            [
                proposal("FRAME_X", "frame", (8, 8, 112, 72), 0.98),
                proposal("LINE_X", "line", (18, 38, 102, 43), 0.96),
            ],
        )
        geometry = [node for node in result.nodes if node.kind in {"frame", "line"}]
        self.assertGreaterEqual(len(geometry), 2)
        first = (geometry[0].full_support or geometry[0].visible_alpha).to_canvas((120, 80)) > 0
        second = (geometry[1].full_support or geometry[1].visible_alpha).to_canvas((120, 80)) > 0
        self.assertEqual(int(np.count_nonzero(first & second)), 0)

    def test_closed_rounded_frame_keeps_hole_and_does_not_take_inner_text(self) -> None:
        shape = (180, 280)
        image = np.full((*shape, 3), (246, 242, 232), dtype=np.uint8)
        outer = rounded_rectangle(shape, (20, 22, 260, 158), 14)
        inner = rounded_rectangle(shape, (27, 29, 253, 151), 9)
        true_frame = (outer > 0) & ~(inner > 0)
        image[true_frame] = np.array((16, 145, 72), dtype=np.uint8)
        protected = np.zeros(shape, dtype=np.uint8)
        cv2.putText(
            image, "SALE", (78, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (16, 145, 72), 4, cv2.LINE_AA
        )
        cv2.putText(
            protected, "SALE", (78, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 6, cv2.LINE_AA
        )

        result = extract_geometry_layers(
            image,
            [proposal("FRAME_A", "frame", (20, 22, 260, 158))],
            protected_alpha=protected,
        )

        self.assertEqual(result.report["counts"]["accepted"], 1)
        node = next(item for item in result.nodes if item.kind == "frame")
        support = node.full_support.to_canvas((shape[1], shape[0])) >= 16
        intersection = np.count_nonzero(support & true_frame)
        union = np.count_nonzero(support | true_frame)
        self.assertGreater(intersection / union, 0.78)
        self.assertEqual(int(np.count_nonzero(support[45:135, 45:235])), 0)
        self.assertEqual(int(np.count_nonzero(support & (protected > 0))), 0)

    def test_attached_text_ribbon_is_removed_from_authoritative_frame_ring(self) -> None:
        shape = (220, 340)
        image = np.full((*shape, 3), (248, 245, 236), dtype=np.uint8)
        outer = rounded_rectangle(shape, (30, 50, 310, 195), 14)
        inner = rounded_rectangle(shape, (37, 57, 303, 188), 9)
        true_frame = (outer > 0) & ~(inner > 0)
        orange = np.array((236, 119, 17), dtype=np.uint8)
        image[true_frame] = orange

        ribbon = np.zeros(shape, dtype=np.uint8)
        ribbon_points = np.array(
            [[30, 25], [220, 25], [207, 42], [220, 58], [30, 58]],
            dtype=np.int32,
        )
        cv2.fillPoly(ribbon, [ribbon_points], 255)
        image[ribbon > 0] = orange
        protected = np.zeros(shape, dtype=np.uint8)
        cv2.putText(
            protected,
            "7 DEAL",
            (44, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            255,
            3,
            cv2.LINE_AA,
        )
        image[protected > 0] = np.array((250, 248, 242), dtype=np.uint8)
        # A separate child selection genuinely crosses the frame stroke.  The
        # clean model must continue underneath this child while excluding the
        # ribbon text, which belongs to the detached header rather than frame.
        cv2.circle(protected, (270, 52), 6, 255, -1, cv2.LINE_AA)

        result = extract_geometry_layers(
            image,
            [proposal("FRAME_RIBBON", "frame", (30, 50, 310, 195), 0.98)],
            protected_alpha=protected,
        )

        self.assertEqual(result.report["counts"]["accepted"], 1)
        panel_report = result.decisions[0].metrics["panel_extraction"]
        self.assertEqual(panel_report["status"], "accepted")
        self.assertGreater(
            panel_report["metrics"][
                "canonical_interior_fit_exclusion_pixel_count"
            ],
            100,
        )
        panel = next(item for item in result.nodes if item.kind == "panel")
        panel_x = 100 - panel.bbox[0]
        panel_y = 57 - panel.bbox[1]
        self.assertGreater(int(panel.rgba[panel_y, panel_x, 3]), 0)
        self.assertLess(
            int(
                np.max(
                    np.abs(
                        panel.rgba[panel_y, panel_x, :3].astype(np.int16)
                        - np.array((248, 245, 236), dtype=np.int16)
                    )
                )
            ),
            10,
        )
        node = next(item for item in result.nodes if item.kind == "frame")
        full = node.full_support.to_canvas((shape[1], shape[0])) >= 16
        visible = node.visible_alpha.to_canvas((shape[1], shape[0])) > 0
        intersection = int(np.count_nonzero(full & true_frame))
        union = int(np.count_nonzero(full | true_frame))
        self.assertGreater(intersection / union, 0.80)

        ribbon_only = (ribbon > 0) & ~true_frame
        retained_ribbon = int(np.count_nonzero(full & ribbon_only))
        self.assertLess(
            retained_ribbon / max(1, int(np.count_nonzero(ribbon_only))),
            0.06,
        )
        self.assertEqual(int(np.count_nonzero(full[70:175, 60:285])), 0)
        self.assertGreater(int(np.count_nonzero(full & (protected > 0))), 20)
        self.assertEqual(int(np.count_nonzero(visible & (protected > 0))), 0)

        topology = node.metadata["support_topology"]
        self.assertEqual(topology["status"], "pass")
        self.assertTrue(topology["structural_background_preserved"])
        self.assertEqual(topology["nonprotected_support_pixel_count_added"], 0)
        evidence = topology["frame_model_evidence"]
        self.assertEqual(evidence["policy"], "dominant_interior_outward_ring_v1")
        self.assertEqual(evidence["status"], "pass")
        self.assertGreater(evidence["removed_attached_content_pixel_count"], 500)
        self.assertGreater(evidence["preserved_structural_hole_pixel_count"], 30000)

    def test_text_shaped_false_frame_stays_unresolved(self) -> None:
        image = np.full((100, 300, 3), (246, 242, 232), dtype=np.uint8)
        cv2.putText(
            image, "BIG SALE", (25, 66), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 30, 35), 4, cv2.LINE_AA
        )
        result = extract_geometry_layers(
            image,
            [proposal("FALSE_FRAME", "frame", (20, 24, 285, 75))],
        )
        self.assertFalse(result.nodes)
        self.assertEqual(result.decisions[0].status, "unresolved")
        self.assertIn("closed", result.decisions[0].reason)

    def test_divider_reconstructs_support_below_protected_badge(self) -> None:
        image = np.full((110, 320, 3), (246, 242, 232), dtype=np.uint8)
        cv2.line(image, (24, 55), (296, 55), (12, 92, 34), 5, cv2.LINE_AA)
        protected = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(protected, (160, 55), 22, 255, -1)
        cv2.circle(image, (160, 55), 22, (235, 128, 22), -1, cv2.LINE_AA)
        result = extract_geometry_layers(
            image,
            [proposal("LINE_A", "line", (22, 51, 299, 60))],
            protected_alpha=protected,
        )
        self.assertEqual(result.report["counts"]["accepted"], 1)
        node = result.nodes[0]
        full = node.full_support.to_canvas((320, 110))
        visible = node.visible_alpha.to_canvas((320, 110))
        self.assertGreater(int(np.count_nonzero(full[53:58, 150:171])), 80)
        self.assertEqual(int(np.count_nonzero(visible[53:58, 150:171])), 0)
        self.assertTrue(node.occluded)
        self.assertTrue(np.array_equal(node.rgba[:, :, 3], node.full_support.alpha))

    def test_ribbon_hint_preserves_notched_outer_shape_and_fills_text_occlusion(self) -> None:
        image = np.full((110, 250, 3), (246, 242, 232), dtype=np.uint8)
        points = np.array(
            [[20, 26], [225, 26], [207, 55], [225, 84], [20, 84], [38, 55]], np.int32
        )
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        image[mask > 0] = np.array((214, 28, 39), dtype=np.uint8)
        protected = np.zeros(mask.shape, dtype=np.uint8)
        cv2.putText(protected, "DEAL", (75, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 3, cv2.LINE_AA)
        image[protected > 0] = np.array((250, 248, 240), dtype=np.uint8)
        result = extract_geometry_layers(
            image,
            [proposal("RIBBON_A", "ribbon", (18, 24, 228, 87), mask=mask)],
            protected_alpha=protected,
        )
        self.assertEqual(result.report["counts"]["accepted"], 1)
        node = result.nodes[0]
        full = node.full_support.to_canvas((250, 110))
        visible = node.visible_alpha.to_canvas((250, 110))
        self.assertGreater(int(full[55, 45]), 0)
        self.assertEqual(int(full[55, 21]), 0, "left notch must remain outside the ribbon")
        self.assertGreater(int(np.count_nonzero(full & protected)), 0)
        self.assertEqual(int(np.count_nonzero(visible & protected)), 0)

    def test_white_card_surface_is_synthesized_below_text_and_product(self) -> None:
        shape = (240, 360)
        image = np.full((*shape, 3), (231, 226, 215), dtype=np.uint8)
        outer = rounded_rectangle(shape, (40, 44, 320, 198), 15)
        inner = rounded_rectangle(shape, (47, 51, 313, 191), 10)
        frame = (outer > 0) & ~(inner > 0)
        image[inner > 0] = np.array((250, 248, 243), dtype=np.uint8)
        image[frame] = np.array((25, 115, 190), dtype=np.uint8)
        protected = np.zeros(shape, dtype=np.uint8)
        cv2.rectangle(protected, (185, 82), (282, 172), 255, -1)
        cv2.rectangle(image, (185, 82), (282, 172), (225, 90, 25), -1)
        cv2.putText(
            protected, "PRICE", (70, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 4, cv2.LINE_AA
        )
        cv2.putText(
            image, "PRICE", (70, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 25, 30), 3, cv2.LINE_AA
        )

        result = extract_geometry_layers(
            image,
            [proposal("CARD_WHITE", "frame", (40, 44, 320, 198))],
            protected_alpha=protected,
        )

        panel = next(item for item in result.nodes if item.kind == "panel")
        frame_node = next(item for item in result.nodes if item.kind == "frame")
        self.assertLess(panel.z_index, frame_node.z_index)
        self.assertLess(panel.z_index, 0, "synthesized surfaces use a global bottom z-band")
        self.assertEqual(set(result.proposals[0].owner_ids), {panel.element_id, frame_node.element_id})
        full = panel.full_support.to_canvas((shape[1], shape[0]))
        visible = panel.visible_alpha.to_canvas((shape[1], shape[0]))
        product = np.zeros(shape, dtype=bool)
        product[82:173, 185:283] = True
        self.assertGreater(int(np.count_nonzero(full[product])), 8000)
        self.assertEqual(int(np.count_nonzero(visible[product])), 0)
        self.assertTrue(panel.occluded)
        # The card layer contains a reconstructed white surface under the
        # independent product, never the orange product pixels themselves.
        local_product = product[
            panel.bbox[1] : panel.bbox[3], panel.bbox[0] : panel.bbox[2]
        ]
        hidden_rgb = panel.rgba[:, :, :3][local_product & (panel.rgba[:, :, 3] > 0)]
        self.assertGreater(len(hidden_rgb), 8000)
        expected_surface = np.array((250, 248, 243), dtype=np.int16)
        self.assertLessEqual(
            float(
                np.percentile(
                    np.abs(hidden_rgb.astype(np.int16) - expected_surface), 95
                )
            ),
            2.0,
        )
        self.assertNotEqual(tuple(int(v) for v in hidden_rgb[len(hidden_rgb) // 2]), (225, 90, 25))
        self.assertEqual(result.decisions[0].metrics["panel_extraction"]["status"], "accepted")
        panel_visible = panel.visible_alpha.to_canvas((shape[1], shape[0])) > 0
        frame_visible = frame_node.visible_alpha.to_canvas((shape[1], shape[0])) > 0
        self.assertEqual(
            int(np.count_nonzero(panel_visible & frame_visible)),
            0,
            "paired panel and frame must never share visible ownership",
        )
        cleanliness = panel.metadata["geometry_cleanliness"]
        self.assertEqual(cleanliness["status"], "pass")
        self.assertEqual(cleanliness["reference_type"], "surface_rgb")
        self.assertGreater(cleanliness["known_child_excluded_pixel_count"], 8000)
        self.assertEqual(cleanliness["carved_destination_role"], "exact_source_remainder_above_clean_base")

    def test_unowned_text_is_carved_from_clean_panel_surface(self) -> None:
        shape = (240, 360)
        clean = np.full((*shape, 3), (231, 226, 215), dtype=np.uint8)
        outer = rounded_rectangle(shape, (40, 44, 320, 198), 15)
        inner = rounded_rectangle(shape, (47, 51, 313, 191), 10)
        clean[inner > 0] = np.array((250, 248, 243), dtype=np.uint8)
        clean[(outer > 0) & ~(inner > 0)] = np.array((25, 115, 190), dtype=np.uint8)
        image = clean.copy()
        dirty_text = np.zeros(shape, dtype=np.uint8)
        cv2.putText(
            dirty_text, "99.000", (90, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 4, cv2.LINE_AA
        )
        cv2.putText(
            image, "99.000", (90, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 40, 50), 4, cv2.LINE_AA
        )

        result = extract_geometry_layers(
            image,
            [proposal("CARD_DIRTY_TEXT", "frame", (40, 44, 320, 198))],
        )

        panel = next(item for item in result.nodes if item.kind == "panel")
        full = panel.full_support.to_canvas((shape[1], shape[0])) > 0
        visible = panel.visible_alpha.to_canvas((shape[1], shape[0])) > 0
        text = dirty_text > 0
        self.assertGreater(int(np.count_nonzero(full & text)), 1000)
        self.assertEqual(int(np.count_nonzero(visible & text)), 0)
        self.assertTrue(panel.move_safe)
        self.assertEqual(panel.review_status, "auto_confirmed")
        cleanliness = panel.metadata["geometry_cleanliness"]
        self.assertEqual(cleanliness["status"], "pass")
        self.assertGreater(cleanliness["carved_pixel_count"], 1000)
        self.assertLessEqual(
            cleanliness["remaining_visible_delta_e76_max"],
            cleanliness["remaining_visible_delta_e76_limit"],
        )
        x0, y0, x1, y1 = panel.bbox
        local_text = text[y0:y1, x0:x1]
        local_support = panel.rgba[:, :, 3] > 0
        observed = panel.rgba[:, :, :3][local_text & local_support]
        self.assertTrue(len(observed))
        self.assertLessEqual(
            int(np.max(np.abs(observed.astype(np.int16) - np.array((250, 248, 243), np.int16)))),
            1,
        )

    def test_large_reference_mismatch_fails_closed(self) -> None:
        shape = (100, 220)
        image = np.full((*shape, 3), (245, 242, 232), dtype=np.uint8)
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (199, 79), 255, -1)
        yy, xx = np.indices(shape)
        first = (mask > 0) & (((xx // 8) + (yy // 8)) % 2 == 0)
        second = (mask > 0) & ~first
        image[first] = np.array((220, 30, 45), dtype=np.uint8)
        image[second] = np.array((30, 90, 210), dtype=np.uint8)

        result = extract_geometry_layers(
            image,
            [proposal("UNSTABLE_RIBBON", "ribbon", (20, 20, 200, 80), mask=mask)],
        )

        node = next(item for item in result.nodes if item.kind == "ribbon")
        cleanliness = node.metadata["geometry_cleanliness"]
        self.assertEqual(cleanliness["status"], "unsafe")
        self.assertFalse(node.move_safe)
        self.assertEqual(node.review_status, "unresolved")
        self.assertGreater(cleanliness["carved_pixel_fraction_of_support"], 0.35)

    def test_affine_coloured_panel_restores_surface_under_children(self) -> None:
        shape = (230, 380)
        image = np.full((*shape, 3), (235, 232, 225), dtype=np.uint8)
        outer = rounded_rectangle(shape, (44, 38, 336, 198), 17)
        inner = rounded_rectangle(shape, (51, 45, 329, 191), 11)
        yy, xx = np.indices(shape)
        truth = np.stack(
            (
                95 + xx * 55 // shape[1],
                165 + xx * 28 // shape[1],
                205 + yy * 18 // shape[0],
            ),
            axis=2,
        ).astype(np.uint8)
        image[inner > 0] = truth[inner > 0]
        image[(outer > 0) & ~(inner > 0)] = np.array((18, 82, 145), dtype=np.uint8)
        protected = np.zeros(shape, dtype=np.uint8)
        cv2.rectangle(protected, (168, 74), (286, 166), 255, -1)
        cv2.rectangle(image, (168, 74), (286, 166), (235, 55, 65), -1)
        cv2.putText(protected, "A", (82, 126), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 6, cv2.LINE_AA)
        cv2.putText(image, "A", (82, 126), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (250, 250, 250), 5, cv2.LINE_AA)

        result = extract_geometry_layers(
            image,
            [proposal("CARD_GRADIENT", "frame", (44, 38, 336, 198))],
            protected_alpha=protected,
        )

        panel = next(item for item in result.nodes if item.kind == "panel")
        surface_report = result.decisions[0].metrics["panel_extraction"]["metrics"]["surface"]
        self.assertEqual(surface_report["model"], "affine")
        x0, y0, x1, y1 = panel.bbox
        local_truth = truth[y0:y1, x0:x1]
        local_protected = protected[y0:y1, x0:x1] > 0
        local_support = panel.rgba[:, :, 3] > 0
        error = np.abs(
            panel.rgba[:, :, :3].astype(np.int16) - local_truth.astype(np.int16)
        )
        self.assertLessEqual(
            float(np.percentile(error[local_protected & local_support], 95)), 3.0
        )

    def test_outer_poster_border_never_becomes_a_panel(self) -> None:
        shape = (200, 300)
        image = np.full((*shape, 3), (246, 242, 232), dtype=np.uint8)
        cv2.rectangle(image, (2, 2), (297, 197), (22, 128, 62), 6, cv2.LINE_AA)
        result = extract_geometry_layers(
            image,
            [proposal("CANVAS_BORDER", "frame", (0, 0, 300, 200))],
        )
        self.assertTrue(any(item.kind == "frame" for item in result.nodes))
        self.assertFalse(any(item.kind == "panel" for item in result.nodes))
        panel = result.decisions[0].metrics["panel_extraction"]
        self.assertEqual(panel["status"], "unresolved")
        self.assertIn("canvas border", panel["reason"])
        self.assertEqual(len(result.proposals[0].owner_ids), 1)

    def test_concentric_frame_proposals_are_rejected_as_duplicate_geometry(self) -> None:
        shape = (220, 340)
        image = np.full((*shape, 3), (230, 225, 215), dtype=np.uint8)
        outer_a = rounded_rectangle(shape, (30, 30, 310, 190), 15)
        inner_a = rounded_rectangle(shape, (35, 35, 305, 185), 11)
        outer_b = rounded_rectangle(shape, (39, 39, 301, 181), 11)
        inner_b = rounded_rectangle(shape, (44, 44, 296, 176), 8)
        second_frame = (outer_b > 0) & ~(inner_b > 0)
        image[inner_a > 0] = np.array((248, 247, 242), dtype=np.uint8)
        image[(outer_a > 0) & ~(inner_a > 0)] = np.array((20, 110, 180), dtype=np.uint8)
        image[second_frame] = np.array((20, 110, 180), dtype=np.uint8)
        protected = second_frame.astype(np.uint8) * 255

        result = extract_geometry_layers(
            image,
            [
                proposal("OUTER_FRAME", "frame", (30, 30, 310, 190)),
                proposal("INNER_FRAME", "frame", (39, 39, 301, 181)),
            ],
            protected_alpha=protected,
        )

        self.assertEqual(sum(item.kind == "panel" for item in result.nodes), 1)
        self.assertEqual(sum(item.kind == "frame" for item in result.nodes), 1)
        statuses = {item.proposal_id: item.status for item in result.decisions}
        self.assertEqual(statuses["OUTER_FRAME"], "accepted")
        self.assertEqual(statuses["INNER_FRAME"], "rejected_duplicate")
        inner = next(item for item in result.decisions if item.proposal_id == "INNER_FRAME")
        self.assertIn("concentric", inner.reason)
        self.assertEqual(result.proposals[1].status, "rejected")

    def test_one_value_protected_alpha_has_zero_visible_parent_ownership(self) -> None:
        image = np.full((100, 300, 3), (245, 241, 230), dtype=np.uint8)
        cv2.line(image, (20, 50), (280, 50), (10, 80, 20), 5, cv2.LINE_AA)
        protected = np.zeros(image.shape[:2], dtype=np.uint8)
        protected[48:53, 140:161] = 1

        result = extract_geometry_layers(
            image,
            [proposal("SOFT_CHILD_EDGE", "line", (19, 46, 282, 55))],
            protected_alpha=protected,
        )

        node = next(item for item in result.nodes if item.kind == "line")
        full = node.full_support.to_canvas((300, 100))
        visible = node.visible_alpha.to_canvas((300, 100))
        protected_support = (protected > 0) & (full > 0)
        self.assertGreater(int(np.count_nonzero(protected_support)), 50)
        self.assertEqual(int(np.count_nonzero(visible[protected > 0])), 0)
        self.assertGreater(int(np.count_nonzero(full[protected > 0])), 50)
        self.assertTrue(np.array_equal(node.rgba[:, :, 3], node.full_support.alpha))

    def test_parallel_hough_proposals_for_same_axis_line_are_deduplicated(self) -> None:
        image = np.full((100, 320, 3), (245, 241, 230), dtype=np.uint8)
        cv2.line(image, (22, 51), (298, 51), (10, 80, 20), 5, cv2.LINE_AA)
        result = extract_geometry_layers(
            image,
            [
                proposal("LINE_PRIMARY", "line", (20, 47, 301, 56), 0.94),
                proposal("LINE_OFFSET", "line", (24, 48, 297, 57), 0.90),
            ],
        )
        self.assertEqual(sum(item.kind == "line" for item in result.nodes), 1)
        self.assertEqual(result.decisions[0].status, "accepted")
        self.assertEqual(result.decisions[1].status, "rejected_duplicate")
        self.assertIn("axis", result.decisions[1].reason)


if __name__ == "__main__":
    unittest.main()
