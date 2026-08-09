from __future__ import annotations

import unittest

import numpy as np

from v5pro.ownership import enforce_exclusive_ownership
from v5pro.schema import AlphaCrop, DocumentGraph, ElementNode, ProposalRecord


def _node(
    element_id: str,
    kind: str,
    alpha: np.ndarray,
    *,
    left: int = 0,
    top: int = 0,
    z: int = 1,
    source: str = "test",
    full_alpha: np.ndarray | None = None,
) -> ElementNode:
    visible = AlphaCrop(left, top, alpha.copy())
    full = AlphaCrop(left, top, (full_alpha if full_alpha is not None else alpha).copy())
    rgb = np.full((*full.alpha.shape, 3), (30, 120, 50), dtype=np.uint8)
    metadata = {}
    if kind in {"frame", "line"} and source.startswith("opencv"):
        metadata["geometry_backend"] = "poster_geometry_v2"
    if source.startswith("grounding_dino"):
        metadata["semantic_extraction"] = {"test": True}
    return ElementNode(
        element_id,
        element_id,
        kind,  # type: ignore[arg-type]
        visible,
        z,
        full_support=full,
        confidence=0.95,
        review_status="auto_confirmed",
        move_safe=True,
        rgba=np.dstack([rgb, full.alpha]),
        evidence=[{"source": source, "proposal_id": f"P_{element_id}"}],
        metadata=metadata,
    )


class OwnershipV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = np.full((12, 16, 3), 245, dtype=np.uint8)

    def test_panel_under_antialiased_text_has_zero_visible_overlap(self) -> None:
        graph = DocumentGraph((16, 12))
        panel_alpha = np.full((8, 12), 255, dtype=np.uint8)
        text_alpha = np.array(
            [[0, 40, 180, 255, 180, 40], [30, 210, 255, 255, 210, 30]],
            dtype=np.uint8,
        )
        panel = _node("PANEL", "panel", panel_alpha, left=2, top=2, z=1)
        text = _node("TEXT", "text", text_alpha, left=5, top=5, z=2, source="ocr_vietnamese")
        graph.add_node(panel)
        graph.add_node(text)

        report = enforce_exclusive_ownership(graph, self.source)

        panel = graph.node_map()["PANEL"]
        text = graph.node_map()["TEXT"]
        p = panel.visible_alpha.to_canvas(graph.canvas_size)
        t = text.visible_alpha.to_canvas(graph.canvas_size)
        self.assertFalse(np.any((p > 0) & (t > 0)))
        self.assertTrue(np.array_equal(text_alpha, text.visible_alpha.alpha))
        self.assertEqual(panel.full_support.nonzero_pixels, 8 * 12)
        self.assertEqual(panel.rgba[:, :, 3].max(), 255)
        self.assertTrue(panel.occluded)
        self.assertEqual(report["overlap_pixels_after"], 0)

    def test_verified_frame_beats_and_trims_raw_layerd(self) -> None:
        graph = DocumentGraph((16, 12))
        raw = _node(
            "ELEMENT_00001",
            "unknown",
            np.full((4, 9), 255, np.uint8),
            left=2,
            top=4,
            z=999,
            source="layerd_iterative_top_layer",
        )
        frame_alpha = np.zeros((4, 9), dtype=np.uint8)
        frame_alpha[:, :2] = 255
        frame = _node(
            "FRAME",
            "frame",
            frame_alpha,
            left=2,
            top=4,
            z=1,
            source="opencv_layout_frame",
        )
        graph.add_node(raw)
        graph.add_node(frame)

        enforce_exclusive_ownership(graph, self.source)

        raw = graph.node_map()["ELEMENT_00001"]
        r = raw.visible_alpha.to_canvas(graph.canvas_size)
        f = frame.visible_alpha.to_canvas(graph.canvas_size)
        self.assertFalse(np.any((r > 0) & (f > 0)))
        self.assertEqual(raw.full_support.nonzero_pixels, raw.visible_alpha.nonzero_pixels)
        self.assertEqual(raw.visible_alpha.nonzero_pixels, 4 * 7)

    def test_fully_absorbed_raw_node_repairs_proposal_ledger(self) -> None:
        graph = DocumentGraph((16, 12))
        support = np.full((3, 5), 255, dtype=np.uint8)
        raw = _node(
            "ELEMENT_00002",
            "unknown",
            support,
            left=4,
            top=4,
            z=999,
            source="layerd_iterative_top_layer",
        )
        semantic = _node(
            "SEMANTIC_1",
            "product",
            support,
            left=4,
            top=4,
            z=1,
            source="grounding_dino_sam2",
        )
        graph.add_node(raw)
        graph.add_node(semantic)
        semantic.parent_id = raw.element_id
        graph.add_proposal(
            ProposalRecord(
                "RAW_P",
                "layerd_iterative_top_layer",
                "unknown",
                raw.bbox,
                0.45,
                status="assigned",
                owner_ids=[raw.element_id],
            )
        )

        report = enforce_exclusive_ownership(graph, self.source)

        self.assertNotIn(raw.element_id, graph.node_map())
        proposal = graph.proposals[0]
        self.assertEqual(proposal.status, "assigned")
        self.assertEqual(proposal.owner_ids, [semantic.element_id])
        self.assertIn("absorbed", proposal.reason or "")
        self.assertEqual(report["removed_raw_node_count"], 1)
        self.assertEqual(report["reparented_node_count"], 1)
        self.assertIsNone(graph.node_map()[semantic.element_id].parent_id)
        self.assertEqual(report["assigned_dangling_owner_count"], 0)
        graph.validate()

    def test_fully_absorbed_nonraw_residual_is_pruned_and_ledger_reassigned(self) -> None:
        graph = DocumentGraph((16, 12))
        support = np.full((3, 5), 255, dtype=np.uint8)
        residual = _node(
            "RESIDUAL_00172",
            "micro_detail",
            support,
            left=4,
            top=4,
            z=-100,
            source="poster_surface_residual",
        )
        residual.review_status = "unresolved"
        residual.move_safe = False
        semantic = _node(
            "SEMANTIC_1",
            "product",
            support,
            left=4,
            top=4,
            z=1,
            source="grounding_dino_sam2",
        )
        graph.add_node(residual)
        graph.add_node(semantic)
        graph.add_proposal(
            ProposalRecord(
                "RESIDUAL_P",
                "poster_surface_residual",
                "micro_detail",
                residual.bbox,
                0.45,
                status="assigned",
                owner_ids=[residual.element_id],
            )
        )

        report = enforce_exclusive_ownership(graph, self.source)

        self.assertNotIn(residual.element_id, graph.node_map())
        proposal = graph.proposals[0]
        self.assertEqual(proposal.status, "assigned")
        self.assertEqual(proposal.owner_ids, [semantic.element_id])
        self.assertIn("absorbed", proposal.reason or "")
        self.assertEqual(report["removed_raw_node_count"], 0)
        self.assertEqual(report["removed_empty_node_count"], 1)
        self.assertEqual(report["removed_empty_node_ids"], [residual.element_id])
        self.assertEqual(report["assigned_dangling_owner_count"], 0)
        graph.validate()

    def test_final_ownership_max_is_one_across_all_priority_bands(self) -> None:
        graph = DocumentGraph((16, 12))
        graph.add_node(
            _node("PANEL", "panel", np.full((10, 14), 255, np.uint8), left=1, top=1)
        )
        frame_alpha = np.zeros((8, 12), np.uint8)
        frame_alpha[[0, -1], :] = 255
        frame_alpha[:, [0, -1]] = 255
        graph.add_node(
            _node("FRAME", "frame", frame_alpha, left=2, top=2, source="opencv_hough_frame")
        )
        graph.add_node(
            _node(
                "SEMANTIC",
                "icon",
                np.full((4, 4), 255, np.uint8),
                left=5,
                top=4,
                source="grounding_dino_sam2",
            )
        )
        graph.add_node(
            _node(
                "RAW",
                "unknown",
                np.full((6, 7), 255, np.uint8),
                left=3,
                top=3,
                source="layerd_iterative_top_layer",
                z=9999,
            )
        )

        report = enforce_exclusive_ownership(graph, self.source)
        owner_count, _ = graph.ownership_maps()

        self.assertEqual(int(owner_count.max()), 1)
        self.assertEqual(int(np.count_nonzero(owner_count > 1)), 0)
        self.assertEqual(report["max_owner_count_after"], 1)
        for proposal in graph.proposals:
            if proposal.status == "assigned":
                self.assertTrue(proposal.owner_ids)
                self.assertTrue(set(proposal.owner_ids) <= set(graph.node_map()))


if __name__ == "__main__":
    unittest.main()
