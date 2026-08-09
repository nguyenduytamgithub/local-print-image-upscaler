from __future__ import annotations

import unittest

import numpy as np

from v5pro.schema import AlphaCrop, DocumentGraph, ElementNode, ProposalRecord


class SchemaV2Tests(unittest.TestCase):
    def test_crop_is_memory_bounded_and_places_on_canvas(self) -> None:
        alpha = np.array([[0, 128], [255, 0]], dtype=np.uint8)
        crop = AlphaCrop(3, 4, alpha)
        canvas = crop.to_canvas((10, 10))
        self.assertEqual(crop.bbox, (3, 4, 5, 6))
        self.assertEqual(int(canvas[4, 4]), 128)
        self.assertEqual(int(canvas[5, 3]), 255)
        self.assertEqual(int(np.count_nonzero(canvas)), 2)

    def test_ledger_cannot_silently_assign_without_owner(self) -> None:
        with self.assertRaises(ValueError):
            ProposalRecord(
                proposal_id="p1",
                source="ocr",
                kind_hint="text",
                bbox=(0, 0, 5, 5),
                confidence=0.9,
                status="assigned",
            )

    def test_graph_reports_overlap_and_unresolved(self) -> None:
        graph = DocumentGraph((8, 8))
        first = ElementNode(
            "a",
            "A",
            "icon",
            AlphaCrop(1, 1, np.full((3, 3), 255, dtype=np.uint8)),
            1,
            review_status="auto_confirmed",
        )
        second = ElementNode(
            "b",
            "B",
            "icon",
            AlphaCrop(2, 2, np.full((3, 3), 255, dtype=np.uint8)),
            2,
        )
        graph.add_node(first)
        graph.add_node(second)
        graph.add_proposal(
            ProposalRecord(
                "p1", "layerd", "icon", (1, 1, 4, 4), 0.9, "assigned", ["a"]
            )
        )
        graph.add_proposal(ProposalRecord("p2", "edge", "unknown", (6, 6, 7, 7), 0.2))
        record = graph.manifest_record()
        self.assertEqual(record["multiply_owned_pixels"], 4)
        self.assertEqual(record["review_unresolved_node_count"], 1)
        self.assertEqual(record["proposal_accounting"]["unresolved"], 1)

    def test_hierarchy_cycle_fails_closed(self) -> None:
        graph = DocumentGraph((4, 4))
        alpha = np.full((1, 1), 255, dtype=np.uint8)
        graph.add_node(ElementNode("a", "A", "panel", AlphaCrop(0, 0, alpha), 0, "b"))
        graph.add_node(ElementNode("b", "B", "panel", AlphaCrop(1, 1, alpha), 1, "a"))
        with self.assertRaises(RuntimeError):
            graph.validate()


if __name__ == "__main__":
    unittest.main()
