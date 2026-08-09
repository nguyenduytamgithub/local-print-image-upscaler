from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from v5pro.fusion import (
    extract_semantic_rgba,
    extract_text_rgba,
    fuse_inventory_and_layerd,
    split_raw_layer_components,
)
from v5pro.inventory import DetectedProposal, InventoryResult
from v5pro.layerd_backend import LayerDResult, LayerDRawLayer
from v5pro.schema import AlphaCrop, ProposalRecord


class FusionV2Tests(unittest.TestCase):
    def test_semantic_rgba_unblends_soft_edge(self) -> None:
        background = np.array([240, 245, 235], dtype=np.float32)
        foreground = np.array([190, 35, 20], dtype=np.float32)
        image = np.broadcast_to(background, (48, 48, 3)).copy()
        alpha = np.zeros((24, 24), dtype=np.uint8)
        alpha[2:22, 2:22] = 128
        alpha[4:20, 4:20] = 255
        coverage = alpha.astype(np.float32) / 255.0
        observed = (
            foreground[None, None, :] * coverage[:, :, None]
            + background[None, None, :] * (1.0 - coverage[:, :, None])
        )
        image[12:36, 12:36] = observed
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        support = AlphaCrop(12, 12, alpha)

        rgba, report = extract_semantic_rgba(image, support)

        self.assertTrue(np.array_equal(rgba[:, :, 3], alpha))
        self.assertTrue(np.allclose(rgba[8, 8, :3], foreground, atol=1))
        self.assertTrue(np.allclose(rgba[2, 10, :3], foreground, atol=2))
        self.assertTrue(report["straight_alpha_unblended"])

    def test_text_extraction_preserves_background_holes(self) -> None:
        image = np.full((90, 220, 3), (245, 245, 235), dtype=np.uint8)
        cv2.putText(image, "O G 80", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (15, 30, 120), 5, cv2.LINE_AA)
        hint = np.zeros(image.shape[:2], dtype=np.uint8)
        hint[15:75, 5:215] = 255
        rgba, crop, report = extract_text_rgba(image, (5, 10, 215, 78), layerd_alpha_hint=hint)
        self.assertGreater(report["nonzero_pixels"], 100)
        self.assertLess(report["occupancy"], 0.60)
        # The center of O is background, not a filled bbox.
        self.assertEqual(int(crop.alpha[35, 25]), 0)
        self.assertTrue(np.array_equal(rgba[:, :, 3], crop.alpha))

    def test_text_extraction_removes_edge_cart_and_fails_closed(self) -> None:
        image = np.full((90, 520, 3), (252, 248, 225), dtype=np.uint8)
        cv2.putText(
            image,
            "HANG TIEU DUNG",
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.45,
            (10, 135, 42),
            4,
            cv2.LINE_AA,
        )
        cv2.rectangle(image, (450, 5), (499, 79), (10, 135, 42), -1)
        cv2.rectangle(image, (461, 25), (489, 47), (250, 250, 240), 2)
        cv2.circle(image, (467, 58), 4, (250, 250, 240), -1)
        cv2.circle(image, (485, 58), 4, (250, 250, 240), -1)

        _rgba, crop, report = extract_text_rgba(
            image,
            (5, 5, 500, 80),
            recognized_text="HÀNG TIÊU DÙNG",
        )

        purity = report["text_purity"]
        self.assertEqual(purity["status"], "unsafe")
        self.assertTrue(purity["adjacent_object"])
        self.assertIn("adjacent_non_text_object", purity["reasons"])
        self.assertGreater(np.count_nonzero(crop.alpha[:, :410]), 500)
        self.assertEqual(np.count_nonzero(crop.alpha[:, 440:]), 0)

        # Filtering the icon from the OCR owner must not erase it.  Because it
        # is absent from protected text alpha, LayerD can ledger and own it as
        # a separate component for residual/review handling.
        raw_rgba = np.zeros((90, 520, 4), dtype=np.uint8)
        raw_rgba[5:80, 450:500, :3] = image[5:80, 450:500]
        raw_rgba[5:80, 450:500, 3] = 255
        detected_text = DetectedProposal(
            ProposalRecord(
                "OCR_0001",
                "tesseract_vie_geometry",
                "text",
                (5, 5, 500, 80),
                0.96,
                evidence={"recognized_text": "HÀNG TIÊU DÙNG"},
            )
        )
        fused = fuse_inventory_and_layerd(
            image,
            LayerDResult(
                image.copy(),
                [LayerDRawLayer(raw_rgba, 1, 1)],
                {"backend": "synthetic"},
            ),
            InventoryResult([detected_text], {"ocr": "synthetic"}),
        )
        text_owner = next(item for item in fused.graph.nodes if item.kind == "text")
        raw_record = next(
            item
            for item in fused.graph.proposals
            if item.source == "layerd_iterative_top_layer"
        )
        self.assertEqual(raw_record.status, "assigned")
        self.assertNotEqual(raw_record.owner_ids, [text_owner.element_id])
        raw_owner = fused.graph.node_map()[raw_record.owner_ids[0]]
        self.assertGreaterEqual(raw_owner.bbox[0], 450)
        self.assertGreaterEqual(raw_owner.full_support.nonzero_pixels, 3_700)

    def test_text_extraction_removes_ribbon_hearts_and_second_row(self) -> None:
        image = np.full((120, 620, 3), (253, 250, 230), dtype=np.uint8)
        cv2.rectangle(image, (10, 18), (590, 68), (220, 10, 15), -1)
        cv2.putText(
            image,
            "GIA TOT - TIET KIEM MOI NGAY",
            (75, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        # A trailing heart deliberately touches the OCR-box edge through its
        # right lobe. It is a decoration, not the final glyph.
        cv2.circle(image, (574, 39), 8, (15, 15, 15), -1)
        cv2.circle(image, (584, 39), 8, (15, 15, 15), -1)
        cv2.fillConvexPoly(
            image,
            np.array([[566, 41], [592, 41], [579, 59]], dtype=np.int32),
            (15, 15, 15),
        )
        cv2.line(image, (70, 86), (545, 86), (15, 125, 45), 4, cv2.LINE_AA)

        _rgba, crop, report = extract_text_rgba(
            image,
            (10, 10, 590, 100),
            recognized_text="GIÁ TỐT - TIẾT KIỆM MỖI NGÀY",
        )

        purity = report["text_purity"]
        self.assertEqual(purity["status"], "unsafe")
        self.assertTrue(purity["adjacent_object"])
        self.assertTrue(purity["multi_band"])
        self.assertGreater(np.count_nonzero(crop.alpha[15:55, 55:525]), 500)
        self.assertEqual(np.count_nonzero(crop.alpha[:, 545:]), 0)
        self.assertEqual(np.count_nonzero(crop.alpha[65:, :]), 0)
        # The red ribbon surface itself is not exported as text.
        self.assertEqual(int(crop.alpha[15, 30]), 0)

    def test_clean_vietnamese_text_remains_auto_confirmable(self) -> None:
        image = np.full((100, 500, 3), (248, 246, 232), dtype=np.uint8)
        cv2.putText(
            image,
            "HANG TIEU DUNG",
            (18, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.45,
            (12, 95, 38),
            4,
            cv2.LINE_AA,
        )
        text = DetectedProposal(
            ProposalRecord(
                "OCR_0001",
                "tesseract_vie_geometry",
                "text",
                (10, 10, 480, 82),
                0.96,
                evidence={"recognized_text": "HÀNG TIÊU DÙNG"},
            )
        )

        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(image.copy(), [], {"backend": "synthetic"}),
            InventoryResult([text], {"ocr": "synthetic"}),
        )

        node = next(item for item in result.graph.nodes if item.kind == "text")
        self.assertEqual(node.review_status, "auto_confirmed")
        self.assertTrue(node.move_safe)
        self.assertEqual(node.metadata["text_purity"]["status"], "pass")
        self.assertEqual(node.metadata["text_purity"]["reasons"], [])

    def test_layer_components_are_all_returned(self) -> None:
        rgba = np.zeros((50, 80, 4), dtype=np.uint8)
        rgba[5:15, 5:20] = (255, 0, 0, 255)
        rgba[30:45, 55:75] = (0, 255, 0, 255)
        layer = LayerDRawLayer(rgba, 1, 1)
        components = split_raw_layer_components(layer)
        self.assertEqual(len(components), 2)
        self.assertEqual(sum(item[0].nonzero_pixels for item in components), 450)

    def test_protected_text_holes_do_not_refragment_one_raw_component(self) -> None:
        rgba = np.zeros((40, 90, 4), dtype=np.uint8)
        rgba[10:30, 5:85] = (30, 80, 190, 255)
        protected = np.zeros((40, 90), dtype=np.uint8)
        for x0 in range(12, 78, 10):
            protected[8:32, x0 : x0 + 3] = 255
        components = split_raw_layer_components(
            LayerDRawLayer(rgba, 1, 1),
            subtract_alpha=protected,
        )

        self.assertEqual(len(components), 1)
        self.assertGreater(components[0][2]["remaining_island_count_after_subtraction"], 1)
        self.assertTrue(components[0][2]["topology_preserved_before_subtraction"])

    def test_rejected_fragmented_source_component_exports_final_connected_pieces(self) -> None:
        image = np.full((80, 120, 3), 248, dtype=np.uint8)
        image[36:44, 10:110] = (30, 75, 190)
        semantic_alpha = np.full((10, 8), 255, dtype=np.uint8)
        semantic = DetectedProposal(
            ProposalRecord(
                "SEMANTIC_0001",
                "grounding_dino_plus_sam2_birefnet",
                "product",
                (56, 35, 64, 45),
                0.99,
                evidence={
                    "label": "product package",
                    "auto_extractable": True,
                    "auto_confirmable": True,
                    "sam_iou_score": 0.99,
                    "semantic_policy": {"accepted": True, "kind": "product"},
                    "refinement": {"accepted": True},
                },
            ),
            AlphaCrop(56, 35, semantic_alpha),
        )
        raw_rgba = np.zeros((80, 120, 4), dtype=np.uint8)
        raw_rgba[36:44, 10:110] = (30, 75, 190, 255)

        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(
                image.copy(),
                [LayerDRawLayer(raw_rgba, 1, 1)],
                {"backend": "synthetic"},
            ),
            InventoryResult([semantic], {"semantic": "synthetic"}),
        )

        raw_record = next(
            item
            for item in result.graph.proposals
            if item.source == "layerd_iterative_top_layer"
        )
        self.assertEqual(len(raw_record.owner_ids), 2)
        raw_nodes = [result.graph.node_map()[node_id] for node_id in raw_record.owner_ids]
        self.assertEqual([node.bbox for node in raw_nodes], [(10, 36, 56, 44), (64, 36, 110, 44)])
        self.assertTrue(
            all(node.metadata["final_connected_component"] for node in raw_nodes)
        )
        self.assertTrue(all(node.metadata["final_piece_count"] == 2 for node in raw_nodes))
        self.assertTrue(all(node.review_status == "unresolved" for node in raw_nodes))
        self.assertEqual(raw_record.evidence["atomic_owner_count"], 2)

    def test_fusion_creates_assigned_proposal_for_every_raw_component(self) -> None:
        image = np.full((60, 100, 3), 255, dtype=np.uint8)
        rgba = np.zeros((60, 100, 4), dtype=np.uint8)
        rgba[5:15, 5:20] = (255, 0, 0, 255)
        rgba[30:50, 60:90] = (0, 200, 0, 255)
        layerd = LayerDResult(
            image.copy(),
            [LayerDRawLayer(rgba, 1, 1)],
            {"backend": "synthetic"},
        )
        result = fuse_inventory_and_layerd(
            image,
            layerd,
            InventoryResult([], {"total_proposals": 0}),
        )
        layerd_records = [
            item for item in result.graph.proposals if item.source == "layerd_iterative_top_layer"
        ]
        self.assertEqual(len(layerd_records), 2)
        self.assertTrue(all(item.status == "assigned" and item.owner_ids for item in layerd_records))
        self.assertEqual(result.graph.manifest_record()["multiply_owned_pixels"], 0)

    def test_component_inside_large_frame_bbox_is_not_mislabeled_frame(self) -> None:
        image = np.full((100, 100, 3), 245, dtype=np.uint8)
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[45:51, 46:52] = (25, 90, 180, 255)
        frame = DetectedProposal(
            ProposalRecord(
                "FRAME_0001",
                "opencv_contour_layout",
                "frame",
                (10, 10, 90, 90),
                0.92,
            )
        )
        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(
                image.copy(),
                [LayerDRawLayer(rgba, 1, 1)],
                {"backend": "synthetic"},
            ),
            InventoryResult([frame], {"frames": "synthetic"}),
        )

        raw_nodes = [node for node in result.graph.nodes if node.element_id.startswith("ELEMENT_")]
        self.assertEqual(len(raw_nodes), 1)
        self.assertNotEqual(raw_nodes[0].kind, "frame")
        raw_record = next(
            item for item in result.graph.proposals if item.source == "layerd_iterative_top_layer"
        )
        self.assertNotEqual(raw_record.kind_hint, "frame")
        self.assertEqual(frame.record.status, "unresolved")
        self.assertEqual(frame.record.owner_ids, [])

    def test_missing_diacritic_is_absorbed_into_text_owner(self) -> None:
        image = np.full((60, 120, 3), 248, dtype=np.uint8)
        image[20:30, 30:60] = (25, 40, 150)
        image[14:17, 38:41] = (25, 40, 150)
        text = DetectedProposal(
            ProposalRecord(
                "OCR_0001",
                "tesseract_geometry",
                "text",
                (10, 10, 100, 40),
                0.91,
                evidence={"recognized_text": "ĐIỂM"},
            )
        )
        text_alpha = np.zeros((30, 90), dtype=np.uint8)
        text_alpha[10:20, 20:50] = 255
        text_rgba = np.zeros((30, 90, 4), dtype=np.uint8)
        text_rgba[10:20, 20:50, :3] = (25, 40, 150)
        text_rgba[:, :, 3] = text_alpha
        raw_rgba = np.zeros((60, 120, 4), dtype=np.uint8)
        raw_rgba[14:17, 38:41] = (25, 40, 150, 255)

        with patch(
            "v5pro.fusion.extract_text_rgba",
            return_value=(
                text_rgba,
                AlphaCrop(10, 10, text_alpha),
                {"occupancy": 0.12, "nonzero_pixels": 300, "component_count": 1},
            ),
        ):
            result = fuse_inventory_and_layerd(
                image,
                LayerDResult(
                    image.copy(),
                    [LayerDRawLayer(raw_rgba, 1, 1)],
                    {"backend": "synthetic"},
                ),
                InventoryResult([text], {"ocr": "synthetic"}),
            )

        text_node = next(node for node in result.graph.nodes if node.kind == "text")
        self.assertEqual(result.report["raw_text_absorbed_component_count"], 1)
        self.assertEqual(result.report["raw_cluster_node_count"], 0)
        self.assertLessEqual(text_node.bbox[1], 14)
        self.assertGreater(text_node.full_support.nonzero_pixels, 300)
        local_y, local_x = 14 - text_node.bbox[1], 38 - text_node.bbox[0]
        self.assertGreater(int(text_node.full_support.alpha[local_y, local_x]), 0)
        raw_record = next(
            item for item in result.graph.proposals if item.source == "layerd_iterative_top_layer"
        )
        self.assertEqual(raw_record.owner_ids, [text_node.element_id])
        self.assertTrue(raw_record.evidence["text_mark_absorption"])

    def test_extreme_layerd_speck_is_explicitly_rejected(self) -> None:
        image = np.full((40, 50, 3), 255, dtype=np.uint8)
        rgba = np.zeros((40, 50, 4), dtype=np.uint8)
        rgba[8, 9] = (0, 0, 0, 80)
        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(
                image.copy(),
                [LayerDRawLayer(rgba, 1, 1)],
                {"backend": "synthetic"},
            ),
            InventoryResult([], {"total_proposals": 0}),
        )

        self.assertEqual(result.report["raw_component_count"], 1)
        self.assertEqual(result.report["raw_rejected_speck_count"], 1)
        self.assertEqual(result.report["node_count"], 0)
        record = result.graph.proposals[0]
        self.assertEqual(record.status, "rejected")
        self.assertIn("two supported pixels", record.reason or "")

    def test_incoherent_unknown_cluster_exports_separate_atomic_review_nodes(self) -> None:
        image = np.full((60, 120, 3), 255, dtype=np.uint8)
        rgba = np.zeros((60, 120, 4), dtype=np.uint8)
        for x0 in (10, 18, 26):
            rgba[20:30, x0 : x0 + 5] = (210, 40, 25, 255)
        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(
                image.copy(),
                [LayerDRawLayer(rgba, 1, 1)],
                {"backend": "synthetic"},
            ),
            InventoryResult([], {"total_proposals": 0}),
        )

        records = [
            item for item in result.graph.proposals if item.source == "layerd_iterative_top_layer"
        ]
        self.assertEqual(len(records), 3)
        self.assertEqual(result.report["raw_cluster_node_count"], 1)
        self.assertEqual(result.report["raw_atomic_review_group_count"], 1)
        self.assertEqual(result.report["raw_atomic_review_node_count"], 3)
        self.assertEqual(len({item.owner_ids[0] for item in records}), 3)
        owners = [result.graph.node_map()[item.owner_ids[0]] for item in records]
        self.assertTrue(all(node.review_status == "unresolved" for node in owners))
        self.assertTrue(all(not node.move_safe for node in owners))
        self.assertTrue(
            all(node.metadata["atomic_split_from_rejected_cluster"] for node in owners)
        )
        self.assertEqual(
            len({node.metadata["review_group_id"] for node in owners}),
            1,
        )
        self.assertTrue(all(node.metadata["pixel_union_exported"] is False for node in owners))

    def test_coherent_layerd_line_stays_grouped_but_requires_clean_reference(self) -> None:
        image = np.full((120, 240, 3), 255, dtype=np.uint8)
        rgba = np.zeros((120, 240, 4), dtype=np.uint8)
        # Three same-colour collinear strokes are close enough to describe one
        # useful rule while remaining below every document-span safety limit.
        for x0 in (20, 44, 68):
            rgba[50:54, x0 : x0 + 20] = (30, 75, 190, 255)
        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(
                image.copy(),
                [LayerDRawLayer(rgba, 1, 1)],
                {"backend": "synthetic"},
            ),
            InventoryResult([], {"total_proposals": 0}),
        )

        raw_nodes = [
            node for node in result.graph.nodes if node.element_id.startswith("ELEMENT_")
        ]
        self.assertEqual(len(raw_nodes), 1)
        node = raw_nodes[0]
        self.assertEqual(node.kind, "line")
        self.assertEqual(node.review_status, "unresolved")
        self.assertFalse(node.move_safe)
        self.assertEqual(node.metadata["member_component_count"], 3)
        policy = node.metadata["auto_confirmation_policy"]
        self.assertFalse(policy["eligible"])
        self.assertEqual(policy["decision"], "defer_to_review")
        self.assertEqual(policy["reasons"], ["missing_clean_geometry_reference"])
        self.assertTrue(policy["geometry_reference"]["required"])
        self.assertFalse(policy["geometry_reference"]["available"])
        self.assertTrue(node.metadata["reference_only_deferral"])
        records = [
            item
            for item in result.graph.proposals
            if item.source == "layerd_iterative_top_layer"
        ]
        self.assertEqual(len(records), 3)
        self.assertEqual({tuple(item.owner_ids) for item in records}, {(node.element_id,)})
        self.assertTrue(
            all(item.evidence["cluster_auto_confirmation_eligible"] is False for item in records)
        )
        self.assertTrue(all("clean reference" in (item.reason or "") for item in records))
        self.assertEqual(result.report["raw_cluster_safety_deferral_count"], 1)
        self.assertEqual(result.report["raw_geometry_reference_deferral_count"], 1)
        self.assertEqual(
            result.report["raw_cluster_safety_deferral_reasons"][
                "missing_clean_geometry_reference"
            ],
            1,
        )
        self.assertEqual(result.report["raw_atomic_review_group_count"], 0)
        self.assertEqual(result.report["raw_atomic_review_node_count"], 0)

    def test_document_scale_multi_island_cluster_is_never_auto_confirmed(self) -> None:
        image = np.full((120, 240, 3), 255, dtype=np.uint8)
        rgba = np.zeros((120, 240, 4), dtype=np.uint8)
        # Eighteen individually disconnected marks form one colour-compatible
        # single-linkage chain. Its long aggregate bbox is classified as a
        # line, but topology alone is insufficient evidence that it is one
        # movable design element.
        for index in range(18):
            x0 = 10 + index * 8
            rgba[50:56, x0 : x0 + 5] = (30, 75, 190, 255)
        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(
                image.copy(),
                [LayerDRawLayer(rgba, 1, 1)],
                {"backend": "synthetic"},
            ),
            InventoryResult([], {"total_proposals": 0}),
        )

        raw_nodes = [
            node for node in result.graph.nodes if node.element_id.startswith("ELEMENT_")
        ]
        self.assertEqual(len(raw_nodes), 18)
        self.assertEqual(
            [node.element_id for node in raw_nodes],
            [f"ELEMENT_{index:05d}" for index in range(1, 19)],
        )
        self.assertTrue(all(node.review_status == "unresolved" for node in raw_nodes))
        self.assertTrue(all(not node.move_safe for node in raw_nodes))
        self.assertTrue(all(node.bbox[2] - node.bbox[0] == 5 for node in raw_nodes))
        self.assertTrue(all(node.bbox[3] - node.bbox[1] == 6 for node in raw_nodes))
        self.assertTrue(
            all(node.metadata["atomic_split_from_rejected_cluster"] for node in raw_nodes)
        )
        self.assertTrue(all(node.metadata["pixel_union_exported"] is False for node in raw_nodes))
        self.assertEqual(
            len({node.metadata["review_group_id"] for node in raw_nodes}),
            1,
        )
        policy = raw_nodes[0].metadata["auto_confirmation_policy"]
        self.assertEqual(policy["policy"], "layerd_raw_cluster_fail_closed_v1")
        self.assertFalse(policy["eligible"])
        self.assertEqual(policy["decision"], "defer_to_review")
        self.assertEqual(policy["evidence"]["member_component_count"], 18)
        self.assertGreater(policy["evidence"]["width_fraction"], 0.40)
        self.assertIn("excessive_member_component_count", policy["reasons"])
        self.assertIn("document_scale_width", policy["reasons"])
        self.assertEqual(result.report["raw_cluster_safety_deferral_count"], 1)
        self.assertEqual(result.report["raw_atomic_review_group_count"], 1)
        self.assertEqual(result.report["raw_atomic_review_node_count"], 18)
        # Atomic supports preserve every source matte pixel exactly without any
        # node bbox spanning two unrelated marks.
        atomic_union = np.zeros(rgba.shape[:2], dtype=np.uint8)
        for node in raw_nodes:
            support = node.full_support or node.visible_alpha
            np.maximum(atomic_union, support.to_canvas((240, 120)), out=atomic_union)
        self.assertTrue(np.array_equal(atomic_union, rgba[:, :, 3]))
        self.assertEqual(
            [node.metadata["review_group_member_index"] for node in raw_nodes],
            list(range(1, 19)),
        )
        self.assertTrue(
            all(node.metadata["review_group_member_count"] == 18 for node in raw_nodes)
        )
        records = [
            item
            for item in result.graph.proposals
            if item.source == "layerd_iterative_top_layer"
        ]
        self.assertEqual(len(records), 18)
        self.assertTrue(all(item.status == "assigned" for item in records))
        self.assertEqual(
            {item.owner_ids[0] for item in records},
            {node.element_id for node in raw_nodes},
        )
        self.assertTrue(
            all(
                item.evidence["cluster_auto_confirmation_eligible"] is False
                for item in records
            )
        )
        self.assertTrue(
            all(item.evidence["atomic_split_due_to_rejected_cluster"] for item in records)
        )
        self.assertEqual(result.graph.manifest_record()["multiply_owned_pixels"], 0)

        repeated = fuse_inventory_and_layerd(
            image,
            LayerDResult(
                image.copy(),
                [LayerDRawLayer(rgba.copy(), 1, 1)],
                {"backend": "synthetic"},
            ),
            InventoryResult([], {"total_proposals": 0}),
        )
        self.assertEqual(
            [(node.element_id, node.bbox) for node in repeated.graph.nodes],
            [(node.element_id, node.bbox) for node in result.graph.nodes],
        )
        self.assertEqual(
            [item.owner_ids for item in repeated.graph.proposals],
            [item.owner_ids for item in result.graph.proposals],
        )

    def test_single_layerd_line_is_unresolved_without_clean_reference(self) -> None:
        image = np.full((120, 240, 3), 255, dtype=np.uint8)
        rgba = np.zeros((120, 240, 4), dtype=np.uint8)
        rgba[50:56, 20:190] = (30, 75, 190, 255)
        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(
                image.copy(),
                [LayerDRawLayer(rgba, 1, 1)],
                {"backend": "synthetic"},
            ),
            InventoryResult([], {"total_proposals": 0}),
        )

        node = next(
            node
            for node in result.graph.nodes
            if node.element_id.startswith("ELEMENT_")
        )
        self.assertEqual(node.kind, "line")
        self.assertEqual(node.review_status, "unresolved")
        self.assertFalse(node.move_safe)
        policy = node.metadata["auto_confirmation_policy"]
        self.assertFalse(policy["eligible"])
        self.assertEqual(policy["decision"], "defer_to_review")
        self.assertEqual(policy["evidence"]["member_component_count"], 1)
        self.assertEqual(policy["reasons"], ["missing_clean_geometry_reference"])
        self.assertEqual(
            node.metadata["clean_geometry_reference"],
            {
                "required": True,
                "available": False,
                "reference_type": None,
                "reason": "LayerD/raw geometry has no independently clean surface reference",
            },
        )
        self.assertEqual(result.report["raw_geometry_reference_deferral_count"], 1)

    def test_layerd_frame_is_unresolved_without_clean_reference(self) -> None:
        image = np.full((100, 120, 3), 255, dtype=np.uint8)
        rgba = np.zeros((100, 120, 4), dtype=np.uint8)
        cv2.rectangle(rgba, (20, 15), (99, 84), (30, 75, 190, 255), 4)

        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(
                image.copy(),
                [LayerDRawLayer(rgba, 1, 1)],
                {"backend": "synthetic"},
            ),
            InventoryResult([], {"total_proposals": 0}),
        )

        node = next(
            item
            for item in result.graph.nodes
            if item.element_id.startswith("ELEMENT_")
        )
        self.assertEqual(node.kind, "frame")
        self.assertEqual(node.review_status, "unresolved")
        self.assertFalse(node.move_safe)
        policy = node.metadata["auto_confirmation_policy"]
        self.assertEqual(policy["reasons"], ["missing_clean_geometry_reference"])
        self.assertFalse(policy["geometry_reference"]["available"])
        self.assertEqual(result.report["raw_geometry_reference_deferral_count"], 1)

    def test_semantic_product_becomes_clean_owner_and_is_removed_from_layerd(self) -> None:
        image = np.full((80, 100, 3), 245, dtype=np.uint8)
        alpha = np.zeros((36, 40), dtype=np.uint8)
        cv2.circle(alpha, (20, 18), 16, 255, -1, cv2.LINE_AA)
        image[20:56, 30:70][alpha > 0] = (20, 130, 210)
        semantic = DetectedProposal(
            ProposalRecord(
                "SEMANTIC_0001",
                "grounding_dino_plus_sam2_birefnet",
                "product",
                (30, 20, 70, 56),
                0.92,
                evidence={
                    "label": "product package",
                    "auto_extractable": True,
                    "sam_iou_score": 0.96,
                    "semantic_policy": {"accepted": True},
                },
            ),
            AlphaCrop(30, 20, alpha),
        )
        raw_rgba = np.zeros((80, 100, 4), dtype=np.uint8)
        raw_rgba[20:56, 30:70, :3] = image[20:56, 30:70]
        raw_rgba[20:56, 30:70, 3] = alpha
        layerd = LayerDResult(
            np.full_like(image, 245),
            [LayerDRawLayer(raw_rgba, 1, 1)],
            {"backend": "synthetic"},
        )

        result = fuse_inventory_and_layerd(
            image,
            layerd,
            InventoryResult([semantic], {"semantic": "synthetic"}),
        )

        semantic_nodes = [node for node in result.graph.nodes if node.kind == "product"]
        self.assertEqual(len(semantic_nodes), 1)
        self.assertEqual(result.report["raw_component_count"], 0)
        self.assertEqual(result.report["semantic_node_count"], 1)
        self.assertEqual(semantic.record.status, "assigned")
        self.assertEqual(semantic.record.owner_ids, [semantic_nodes[0].element_id])
        self.assertTrue(np.array_equal(semantic_nodes[0].rgba[:, :, 3], semantic_nodes[0].full_support.alpha))
        # A clean union is safe to move, but a flat raster supplies no
        # independent proof that the union contains exactly one physical item.
        self.assertEqual(semantic_nodes[0].review_status, "unresolved")
        self.assertTrue(semantic_nodes[0].move_safe)
        atomicity = semantic_nodes[0].metadata["semantic_atomicity"]
        self.assertEqual(atomicity["classification"], "atomicity_unverified")
        self.assertFalse(atomicity["atomic_leaf_confirmed"])
        self.assertFalse(atomicity["hidden_instance_pixels_inferred"])
        self.assertEqual(result.report["semantic_atomicity_review_count"], 1)

    def test_clean_compound_product_is_move_safe_but_never_auto_atomic(self) -> None:
        image = np.full((80, 120, 3), 248, dtype=np.uint8)
        alpha = np.zeros((30, 64), dtype=np.uint8)
        cv2.circle(alpha, (19, 15), 14, 255, -1, cv2.LINE_AA)
        cv2.circle(alpha, (45, 15), 14, 255, -1, cv2.LINE_AA)
        image[25:55, 28:92][alpha > 0] = (80, 150, 55)
        semantic = DetectedProposal(
            ProposalRecord(
                "SEMANTIC_0001",
                "grounding_dino_plus_sam2_birefnet",
                "product",
                (28, 25, 92, 55),
                0.96,
                evidence={
                    "label": "rolls of garbage bags",
                    "auto_extractable": True,
                    "auto_confirmable": True,
                    "sam_iou_score": 0.98,
                    "semantic_policy": {"accepted": True, "kind": "product"},
                    "refinement": {
                        "accepted": True,
                        "sam_iou": 0.94,
                        "area_ratio": 1.03,
                    },
                    "semantic_instance_evidence": {
                        "method": "independent_instance_segmentation",
                        "independent": True,
                        "instance_count": 2,
                        "visible_boundary_complete": True,
                    },
                },
            ),
            AlphaCrop(28, 25, alpha),
        )

        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(image.copy(), [], {"backend": "synthetic"}),
            InventoryResult([semantic], {"semantic": "synthetic"}),
        )

        node = next(node for node in result.graph.nodes if node.kind == "product")
        self.assertEqual(node.review_status, "unresolved")
        self.assertTrue(node.move_safe)
        atomicity = node.metadata["semantic_atomicity"]
        self.assertEqual(atomicity["classification"], "compound_subassembly")
        self.assertFalse(atomicity["atomic_leaf_confirmed"])
        self.assertEqual(atomicity["instance_evidence"]["instance_count"], 2)
        self.assertEqual(result.report["semantic_compound_subassembly_count"], 1)

    def test_independently_verified_single_product_may_be_auto_confirmed(self) -> None:
        image = np.full((64, 80, 3), 248, dtype=np.uint8)
        alpha = np.full((18, 22), 255, dtype=np.uint8)
        image[20:38, 25:47] = (180, 70, 45)
        semantic = DetectedProposal(
            ProposalRecord(
                "SEMANTIC_0001",
                "grounding_dino_plus_sam2_birefnet",
                "product",
                (25, 20, 47, 38),
                0.97,
                evidence={
                    "label": "single detergent bottle",
                    "auto_extractable": True,
                    "auto_confirmable": True,
                    "sam_iou_score": 0.98,
                    "semantic_policy": {"accepted": True, "kind": "product"},
                    "refinement": {
                        "accepted": True,
                        "sam_iou": 0.95,
                        "area_ratio": 1.01,
                    },
                    "semantic_instance_evidence": {
                        "method": "independent_instance_segmentation",
                        "independent": True,
                        "instance_count": 1,
                        "visible_boundary_complete": True,
                    },
                },
            ),
            AlphaCrop(25, 20, alpha),
        )

        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(image.copy(), [], {"backend": "synthetic"}),
            InventoryResult([semantic], {"semantic": "synthetic"}),
        )

        node = next(node for node in result.graph.nodes if node.kind == "product")
        self.assertEqual(node.review_status, "auto_confirmed")
        self.assertTrue(node.move_safe)
        self.assertEqual(
            node.metadata["semantic_atomicity"]["classification"],
            "atomic_leaf",
        )
        self.assertTrue(
            node.metadata["semantic_atomicity"]["atomic_leaf_confirmed"]
        )
        self.assertEqual(result.report["semantic_atomicity_review_count"], 0)

    def test_borderline_refinement_is_not_mislabeled_as_atomic_leaf(self) -> None:
        image = np.full((64, 90, 3), 248, dtype=np.uint8)
        alpha = np.full((20, 40), 255, dtype=np.uint8)
        image[20:40, 24:64] = (80, 145, 45)
        semantic = DetectedProposal(
            ProposalRecord(
                "SEMANTIC_0001",
                "grounding_dino_plus_sam2_birefnet",
                "product",
                (24, 20, 64, 40),
                0.96,
                evidence={
                    "label": "roll of garbage bags",
                    "auto_extractable": True,
                    "auto_confirmable": True,
                    "sam_iou_score": 0.98,
                    "semantic_policy": {"accepted": True, "kind": "product"},
                    "refinement": {
                        "accepted": True,
                        "sam_iou": 0.504888,
                        "area_ratio": 1.765065,
                    },
                },
            ),
            AlphaCrop(24, 20, alpha),
        )

        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(image.copy(), [], {"backend": "synthetic"}),
            InventoryResult([semantic], {"semantic": "synthetic"}),
        )

        node = next(node for node in result.graph.nodes if node.kind == "product")
        self.assertEqual(node.review_status, "unresolved")
        self.assertTrue(node.move_safe)
        atomicity = node.metadata["semantic_atomicity"]
        self.assertEqual(
            atomicity["classification"], "compound_or_boundary_ambiguous"
        )
        self.assertTrue(
            atomicity["refinement_boundary_evidence"]["boundary_ambiguous"]
        )
        self.assertFalse(atomicity["hidden_instance_pixels_inferred"])

    def test_failed_birefnet_refinement_is_never_auto_confirmed(self) -> None:
        image = np.full((64, 80, 3), 248, dtype=np.uint8)
        alpha = np.full((20, 24), 255, dtype=np.uint8)
        image[18:38, 22:46] = (35, 120, 210)
        semantic = DetectedProposal(
            ProposalRecord(
                "SEMANTIC_0001",
                "grounding_dino_plus_sam2_birefnet_review",
                "product",
                (22, 18, 46, 38),
                0.99,
                evidence={
                    "label": "product package",
                    "auto_extractable": True,
                    "auto_confirmable": False,
                    "sam_iou_score": 0.99,
                    "semantic_policy": {"accepted": True, "kind": "product"},
                    "refinement": {
                        "accepted": False,
                        "reason": "BiRefNet disagrees with SAM envelope",
                    },
                },
            ),
            AlphaCrop(22, 18, alpha),
        )

        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(image.copy(), [], {"backend": "synthetic"}),
            InventoryResult([semantic], {"semantic": "synthetic"}),
        )

        nodes = [node for node in result.graph.nodes if node.kind == "product"]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].review_status, "unresolved")
        self.assertFalse(nodes[0].move_safe)
        self.assertFalse(nodes[0].metadata["semantic_auto_confirmable"])
        self.assertIn("refinement rejected", " ".join(nodes[0].metadata["semantic_ambiguity_reasons"]))
        self.assertTrue(semantic.record.evidence["requires_manual_review"])
        self.assertIn("manual", (semantic.record.reason or "").lower())
        self.assertEqual(result.report["semantic_refinement_review_count"], 1)
        self.assertEqual(result.report["semantic_ambiguity_owner_ids"], [nodes[0].element_id])

    def test_semantic_kind_mislabel_is_fail_closed(self) -> None:
        image = np.full((64, 80, 3), 248, dtype=np.uint8)
        alpha = np.full((16, 20), 255, dtype=np.uint8)
        image[20:36, 25:45] = (180, 50, 40)
        semantic = DetectedProposal(
            ProposalRecord(
                "SEMANTIC_0001",
                "grounding_dino_plus_sam2_birefnet",
                # Simulate a bad hand-off: DINO/policy said logo, while the
                # materialised proposal was mislabeled as a product.
                "product",
                (25, 20, 45, 36),
                0.99,
                evidence={
                    "label": "brand logo",
                    "auto_extractable": True,
                    "auto_confirmable": True,
                    "sam_iou_score": 0.99,
                    "semantic_policy": {"accepted": True, "kind": "logo"},
                    "refinement": {"accepted": True},
                },
            ),
            AlphaCrop(25, 20, alpha),
        )

        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(image.copy(), [], {"backend": "synthetic"}),
            InventoryResult([semantic], {"semantic": "synthetic"}),
        )

        node = next(node for node in result.graph.nodes if node.kind == "product")
        self.assertEqual(node.review_status, "unresolved")
        self.assertFalse(node.move_safe)
        self.assertFalse(node.metadata["semantic_kind_consistent"])
        self.assertEqual(result.report["semantic_kind_mismatch_review_count"], 1)
        self.assertTrue(semantic.record.evidence["requires_manual_review"])

    def test_failed_refinement_duplicate_downgrades_existing_owner(self) -> None:
        image = np.full((64, 80, 3), 248, dtype=np.uint8)
        image[20:40, 20:44] = (50, 145, 200)
        alpha = np.full((20, 24), 255, dtype=np.uint8)

        def proposal(proposal_id: str, confidence: float, accepted: bool) -> DetectedProposal:
            return DetectedProposal(
                ProposalRecord(
                    proposal_id,
                    "grounding_dino_plus_sam2_birefnet",
                    "product",
                    (20, 20, 44, 40),
                    confidence,
                    evidence={
                        "label": "product package",
                        "auto_extractable": True,
                        "sam_iou_score": 0.99,
                        "semantic_policy": {"accepted": True, "kind": "product"},
                        "refinement": {
                            "accepted": accepted,
                            "reason": "BiRefNet disagrees with SAM envelope" if not accepted else "",
                        },
                    },
                ),
                AlphaCrop(20, 20, alpha.copy()),
            )

        good = proposal("SEMANTIC_0001", 0.99, True)
        failed_duplicate = proposal("SEMANTIC_0002", 0.98, False)
        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(image.copy(), [], {"backend": "synthetic"}),
            InventoryResult([failed_duplicate, good], {"semantic": "synthetic"}),
        )

        nodes = [node for node in result.graph.nodes if node.kind == "product"]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].review_status, "unresolved")
        self.assertFalse(nodes[0].move_safe)
        self.assertEqual(failed_duplicate.record.owner_ids, [nodes[0].element_id])
        self.assertEqual(result.report["semantic_refinement_review_count"], 1)

    def test_duplicate_semantic_mattes_share_one_owner(self) -> None:
        image = np.full((60, 80, 3), 250, dtype=np.uint8)
        alpha = np.full((20, 24), 255, dtype=np.uint8)
        image[15:35, 20:44] = (220, 80, 30)

        def proposal(proposal_id: str, confidence: float) -> DetectedProposal:
            return DetectedProposal(
                ProposalRecord(
                    proposal_id,
                    "grounding_dino_plus_sam2_birefnet",
                    "product",
                    (20, 15, 44, 35),
                    confidence,
                    evidence={
                        "label": "tissue package",
                        "auto_extractable": True,
                        "sam_iou_score": 0.95,
                        "semantic_policy": {"accepted": True},
                    },
                ),
                AlphaCrop(20, 15, alpha.copy()),
            )

        first, second = proposal("SEMANTIC_0001", 0.93), proposal("SEMANTIC_0002", 0.88)
        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(image.copy(), [], {"backend": "synthetic"}),
            InventoryResult([second, first], {"semantic": "synthetic"}),
        )

        product_nodes = [node for node in result.graph.nodes if node.kind == "product"]
        self.assertEqual(len(product_nodes), 1)
        self.assertEqual(first.record.owner_ids, second.record.owner_ids)
        self.assertEqual(result.report["semantic_duplicate_proposal_count"], 1)

    def test_packaging_logo_is_not_exported_twice(self) -> None:
        image = np.full((70, 90, 3), 250, dtype=np.uint8)
        image[10:60, 20:70] = (180, 150, 80)
        product_alpha = np.full((50, 50), 255, dtype=np.uint8)
        logo_alpha = np.full((12, 16), 255, dtype=np.uint8)
        common_evidence = {
            "auto_extractable": True,
            "sam_iou_score": 0.95,
            "semantic_policy": {"accepted": True},
        }
        product = DetectedProposal(
            ProposalRecord(
                "SEMANTIC_0001",
                "grounding_dino_plus_sam2_birefnet",
                "product",
                (20, 10, 70, 60),
                0.91,
                evidence={**common_evidence, "label": "product package"},
            ),
            AlphaCrop(20, 10, product_alpha),
        )
        logo = DetectedProposal(
            ProposalRecord(
                "SEMANTIC_0002",
                "grounding_dino_plus_sam2_birefnet",
                "logo",
                (35, 25, 51, 37),
                0.90,
                evidence={**common_evidence, "label": "brand logo"},
            ),
            AlphaCrop(35, 25, logo_alpha),
        )

        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(image.copy(), [], {"backend": "synthetic"}),
            InventoryResult([logo, product], {"semantic": "synthetic"}),
        )

        self.assertEqual(result.report["semantic_node_count"], 1)
        self.assertEqual(logo.record.owner_ids, product.record.owner_ids)
        self.assertIn("packaging", (logo.record.reason or "").lower())

    def test_dino_only_qr_is_opaque_review_owner_with_quiet_zone(self) -> None:
        image = np.full((100, 120, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (42, 32), (77, 67), (0, 0, 0), -1)
        qr = DetectedProposal(
            ProposalRecord(
                "SEMANTIC_0001",
                "grounding_dino_plus_sam2",
                "qr",
                (42, 32, 78, 68),
                0.72,
                evidence={
                    "label": "QR code",
                    "auto_extractable": False,
                    "semantic_policy": {"accepted": True},
                },
            )
        )

        result = fuse_inventory_and_layerd(
            image,
            LayerDResult(image.copy(), [], {"backend": "synthetic"}),
            InventoryResult([qr], {"semantic": "synthetic"}),
        )

        nodes = [node for node in result.graph.nodes if node.kind == "qr"]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].review_status, "unresolved")
        self.assertFalse(nodes[0].move_safe)
        self.assertTrue(np.all(nodes[0].rgba[:, :, 3] == 255))
        self.assertLess(nodes[0].bbox[0], 42)
        self.assertGreater(nodes[0].bbox[2], 78)
        self.assertEqual(nodes[0].semantic_envelope.bbox, nodes[0].full_support.bbox)
        self.assertTrue(
            np.array_equal(
                nodes[0].semantic_envelope.alpha,
                nodes[0].full_support.alpha,
            )
        )
        self.assertEqual(qr.record.status, "assigned")


if __name__ == "__main__":
    unittest.main()
