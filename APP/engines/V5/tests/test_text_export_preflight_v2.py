from __future__ import annotations

import unittest

import numpy as np

from v5pro.schema import AlphaCrop, DocumentGraph, ElementNode, ProposalRecord
from v5pro.text_export_preflight import (
    TEXT_EXPORT_PURITY_POLICY,
    enforce_text_export_purity_preflight,
    exact_export_text_purity_report,
)


class TextExportPreflightV2Tests(unittest.TestCase):
    @staticmethod
    def _node(
        alpha: np.ndarray,
        *,
        text: str,
        left: int = 20,
        top: int = 20,
        node_id: str = "TEXT_TEST",
    ) -> ElementNode:
        support = AlphaCrop(left, top, np.ascontiguousarray(alpha, dtype=np.uint8))
        return ElementNode(
            node_id,
            f"TEXT {text}",
            "text",
            support,
            100,
            full_support=support,
            confidence=0.99,
            review_status="auto_confirmed",
            move_safe=True,
            text=text,
            evidence=[{"proposal_id": "OCR_TEST", "source": "tesseract_vie_geometry"}],
            metadata={
                "text_purity": {
                    "policy": "source_mask_text_purity_v1",
                    "status": "pass",
                    "reasons": [],
                }
            },
        )

    @staticmethod
    def _graph(source: np.ndarray, node: ElementNode) -> DocumentGraph:
        graph = DocumentGraph(source.shape[1::-1])
        graph.add_node(node)
        graph.add_proposal(
            ProposalRecord(
                "OCR_TEST",
                "tesseract_vie_geometry",
                "text",
                node.bbox,
                0.99,
                status="assigned",
                owner_ids=[node.element_id],
            )
        )
        return graph

    def test_remote_chromatic_edge_fragment_is_not_auto_safe(self) -> None:
        source = np.full((90, 130, 3), 252, dtype=np.uint8)
        alpha = np.zeros((41, 74), dtype=np.uint8)
        for x0, width in ((20, 5), (28, 5), (36, 5), (44, 6), (53, 6), (62, 5)):
            alpha[3:19, x0 : x0 + width] = 255
            source[23:39, 20 + x0 : 20 + x0 + width] = (25, 25, 25)
        # Reproduces the accepted truoc/TEXT_0037 failure mode: a small,
        # remote, coloured curve touching the lower-left OCR crop edge.
        alpha[35:41, 0:8] = 255
        source[55:61, 20:28] = (174, 196, 118)
        node = self._node(alpha, text="\\ ( GÓI)")
        graph = self._graph(source, node)

        summary = enforce_text_export_purity_preflight(graph, source)

        report = node.metadata["text_export_purity_preflight"]
        self.assertEqual(report["policy"], TEXT_EXPORT_PURITY_POLICY)
        self.assertEqual(report["status"], "unsafe")
        self.assertIn(
            "isolated_out_of_primary_band_edge_component", report["reasons"]
        )
        isolated = report["edge_band_evidence"]["isolated_edge_components"]
        self.assertGreaterEqual(
            isolated[0]["horizontal_gap_from_primary_px"],
            isolated[0]["horizontal_gap_floor_px"],
        )
        self.assertEqual(isolated[0]["edge_contact_axis_count"], 2)
        self.assertTrue(report["foreground_core"]["passed"])
        self.assertEqual(node.review_status, "unresolved")
        self.assertFalse(node.move_safe)
        self.assertEqual(summary["failed_node_ids"], [node.element_id])
        ledger = graph.proposals[0].evidence["text_export_purity_by_owner"]
        self.assertEqual(ledger[node.element_id]["status"], "unsafe")

        # Re-running the final preflight is deterministic and does not append
        # duplicate proposal records.
        second = enforce_text_export_purity_preflight(graph, source)
        self.assertEqual(second["failed_node_ids"], [node.element_id])
        self.assertEqual(second["downgraded_node_ids"], [node.element_id])
        self.assertEqual(
            list(
                graph.proposals[0]
                .evidence["text_export_purity_by_owner"]
                .keys()
            ),
            [node.element_id],
        )

    def test_repeated_gate_preserves_first_downgrade_history_everywhere(self) -> None:
        """The final invocation must not erase an earlier auto->review transition."""

        source = np.full((90, 130, 3), 252, dtype=np.uint8)
        alpha = np.zeros((41, 74), dtype=np.uint8)
        for x0, width in ((20, 5), (28, 5), (36, 5), (44, 6), (53, 6), (62, 5)):
            alpha[3:19, x0 : x0 + width] = 255
            source[23:39, 20 + x0 : 20 + x0 + width] = (25, 25, 25)
        alpha[35:41, 0:8] = 255
        source[55:61, 20:28] = (174, 196, 118)
        node = self._node(alpha, text="6 GOI", node_id="TEXT_HISTORY")
        graph = self._graph(source, node)

        first = enforce_text_export_purity_preflight(graph, source)
        first_snapshot = dict(
            node.metadata["text_export_purity_preflight"]["first_downgrade"]
        )
        self.assertEqual(first["downgraded_node_ids"], [node.element_id])
        self.assertEqual(
            first["records"][0]["review_status_before_first_gate"],
            "auto_confirmed",
        )
        self.assertTrue(first["records"][0]["move_safe_before_first_gate"])
        self.assertIn(
            "isolated_out_of_primary_band_edge_component",
            first_snapshot["reasons"],
        )

        second = enforce_text_export_purity_preflight(graph, source)

        self.assertEqual(second["downgraded_node_ids"], [node.element_id])
        self.assertEqual(second["ever_downgraded_node_ids"], [node.element_id])
        self.assertEqual(
            second["first_downgrade_by_node"][node.element_id], first_snapshot
        )
        repeated_record = second["records"][0]
        self.assertEqual(
            repeated_record["review_status_before_first_gate"], "auto_confirmed"
        )
        self.assertTrue(repeated_record["move_safe_before_first_gate"])
        self.assertEqual(
            repeated_record["review_status_before_this_gate"], "unresolved"
        )
        self.assertFalse(repeated_record["move_safe_before_this_gate"])
        self.assertEqual(repeated_record["first_downgrade"], first_snapshot)
        self.assertEqual(
            repeated_record["action"],
            "retained_prior_downgrade_unresolved_non_move_safe",
        )

        # These are the three independently serialized audit surfaces used by
        # the portable manifest: graph metadata, node metadata and proposal
        # evidence.  All must retain the same immutable first transition.
        manifest = graph.manifest_record()
        nested_summary = manifest["metadata"]["text_export_purity_preflight"]
        nested_node = manifest["nodes"][0]["metadata"][
            "text_export_purity_preflight"
        ]
        nested_proposal = manifest["proposals"][0]["evidence"][
            "text_export_purity_by_owner"
        ][node.element_id]
        self.assertEqual(nested_summary["downgraded_node_ids"], [node.element_id])
        self.assertEqual(nested_node["first_downgrade"], first_snapshot)
        self.assertEqual(nested_proposal["first_downgrade"], first_snapshot)
        self.assertEqual(
            nested_proposal["review_status_before_first_gate"], "auto_confirmed"
        )

    def test_outline_fragments_that_drop_glyph_cores_fail_closed(self) -> None:
        source = np.full((70, 110, 3), 252, dtype=np.uint8)
        alpha = np.zeros((16, 59), dtype=np.uint8)
        for x0 in (2, 13, 24, 35, 46):
            source[23:35, 20 + x0 : 20 + x0 + 8] = (20, 20, 20)
            # Four disconnected corner fragments per expected glyph.  This is
            # the topology of an outline-only palette, not five intact glyphs.
            alpha[3:5, x0 : x0 + 2] = 255
            alpha[3:5, x0 + 6 : x0 + 8] = 255
            alpha[13:15, x0 : x0 + 2] = 255
            alpha[13:15, x0 + 6 : x0 + 8] = 255
        node = self._node(alpha, text="THÙNG")
        graph = self._graph(source, node)

        summary = enforce_text_export_purity_preflight(graph, source)

        report = node.metadata["text_export_purity_preflight"]
        self.assertEqual(report["status"], "unsafe")
        self.assertIn("foreground_core_not_retained", report["reasons"])
        self.assertIn(
            "implausible_fragmented_glyph_topology", report["reasons"]
        )
        self.assertLess(report["foreground_core"]["retained_fraction"], 0.20)
        self.assertTrue(report["topology"]["implausibly_fragmented"])
        self.assertEqual(summary["downgraded_node_ids"], [node.element_id])
        self.assertEqual(node.review_status, "unresolved")
        self.assertFalse(node.move_safe)

    def test_clean_vietnamese_glyphs_and_detached_accents_remain_safe(self) -> None:
        source = np.full((90, 150, 3), 250, dtype=np.uint8)
        alpha = np.zeros((32, 100), dtype=np.uint8)
        for x0 in (5, 17, 29, 41, 53, 65, 77):
            alpha[10:28, x0 : x0 + 7] = 255
            source[30:48, 20 + x0 : 20 + x0 + 7] = (24, 62, 35)
        # Legitimate Vietnamese marks are detached but match their glyph ink;
        # unlike the TEXT_0037 contaminant, they are not chromatic strangers.
        for x0 in (18, 42, 66):
            alpha[3:6, x0 : x0 + 3] = 255
            source[23:26, 20 + x0 : 20 + x0 + 3] = (24, 62, 35)
        node = self._node(alpha, text="ĐẶC BIỆT")
        graph = self._graph(source, node)

        direct = exact_export_text_purity_report(node, source)
        summary = enforce_text_export_purity_preflight(graph, source)

        self.assertEqual(direct["status"], "pass")
        self.assertEqual(direct["reasons"], [])
        self.assertGreaterEqual(direct["foreground_core"]["retained_fraction"], 0.99)
        self.assertFalse(direct["topology"]["implausibly_fragmented"])
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(node.review_status, "auto_confirmed")
        self.assertTrue(node.move_safe)

    def test_large_light_antialiased_vietnamese_mark_above_base_is_safe(self) -> None:
        source = np.full((90, 150, 3), 250, dtype=np.uint8)
        alpha = np.zeros((32, 100), dtype=np.uint8)
        for x0 in (5, 17, 29, 41, 53, 65, 77):
            alpha[11:29, x0 : x0 + 7] = 255
            source[31:49, 20 + x0 : 20 + x0 + 7] = (22, 58, 34)
        # Area 16 is deliberately above this mask's component floor. It also
        # touches the crop top and is much lighter than the base ink, matching
        # a large antialiased Vietnamese mark. Horizontal overlap with its base
        # proves it is not the remote TEXT_0037 contaminant.
        alpha[0:4, 42:46] = 255
        source[20:24, 62:66] = (155, 185, 145)
        node = self._node(alpha, text="ĐẶC BIỆT")

        report = exact_export_text_purity_report(node, source)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["reasons"], [])
        self.assertEqual(
            report["edge_band_evidence"]["isolated_edge_components"], []
        )


if __name__ == "__main__":
    unittest.main()
