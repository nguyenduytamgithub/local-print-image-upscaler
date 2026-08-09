from __future__ import annotations

import unittest

import numpy as np

from v5pro.qa import (
    build_quality_report,
    enforce_geometry_export_topology_preflight,
)
from v5pro.schema import AlphaCrop, DocumentGraph, ElementNode, ProposalRecord


class QAV2Tests(unittest.TestCase):
    def _preflight_geometry(
        self,
        support: np.ndarray,
        *,
        node_id: str,
    ) -> tuple[DocumentGraph, dict[str, object]]:
        alpha = np.where(np.asarray(support) > 0, 255, 0).astype(np.uint8)
        height, width = alpha.shape
        crop = AlphaCrop(0, 0, alpha)
        graph = DocumentGraph((width, height), metadata={"geometry": {}})
        graph.add_node(
            ElementNode(
                node_id,
                f"Frame {node_id}",
                "frame",
                crop,
                1,
                full_support=crop,
                semantic_envelope=crop,
                confidence=0.99,
                review_status="auto_confirmed",
                move_safe=True,
                rgba=np.dstack(
                    [np.full((height, width, 3), 30, dtype=np.uint8), alpha]
                ),
                metadata={"geometry_backend": "poster_geometry_v2"},
            )
        )
        graph.add_proposal(
            ProposalRecord(
                f"P_{node_id}",
                "poster_geometry_v2",
                "frame",
                crop.bbox,
                0.99,
                "assigned",
                [node_id],
            )
        )
        return graph, enforce_geometry_export_topology_preflight(graph)

    def test_export_preflight_downgrades_multicomponent_deep_intrusion_frame(self) -> None:
        support = np.zeros((64, 160), dtype=np.uint8)
        support[2:6, 2:158] = 1
        support[58:62, 2:158] = 1
        support[2:62, 2:6] = 1
        support[2:62, 154:158] = 1
        support[18:48, 58:108] = 1
        graph, report = self._preflight_geometry(
            support,
            node_id="GEO_COMPOUND_FRAME",
        )

        node = graph.node_map()["GEO_COMPOUND_FRAME"]
        topology = node.metadata["exported_full_support_topology"]
        self.assertEqual(report["status"], "review_required")
        self.assertEqual(node.review_status, "unresolved")
        self.assertFalse(node.move_safe)
        self.assertIn(
            "support_is_not_one_coherent_component",
            topology["reasons"],
        )
        self.assertIn("frame_has_large_deep_interior_intrusion", topology["reasons"])
        self.assertEqual(
            report["downgraded_node_ids"],
            ["GEO_COMPOUND_FRAME"],
        )
        ledger = graph.proposals[0].evidence[
            "owner_exported_full_support_topology"
        ]["GEO_COMPOUND_FRAME"]
        self.assertEqual(ledger["status"], "fail")
        self.assertEqual(ledger["review_status_after_gate"], "unresolved")
        self.assertFalse(ledger["move_safe_after_gate"])

    def test_export_preflight_downgrades_frame_without_meaningful_hole(self) -> None:
        support = np.ones((45, 215), dtype=np.uint8)
        graph, report = self._preflight_geometry(
            support,
            node_id="GEO_SOLID_BANNER_AS_FRAME",
        )

        node = graph.node_map()["GEO_SOLID_BANNER_AS_FRAME"]
        topology = node.metadata["exported_full_support_topology"]
        self.assertEqual(report["failed_node_ids"], ["GEO_SOLID_BANNER_AS_FRAME"])
        self.assertEqual(node.review_status, "unresolved")
        self.assertFalse(node.move_safe)
        self.assertIn("frame_missing_meaningful_interior_hole", topology["reasons"])
        self.assertIn("frame_missing_dominant_interior_hole", topology["reasons"])
        self.assertIs(
            graph.metadata["geometry"]["export_topology_preflight"],
            report,
        )

        # A later explicit rejection is terminal; preflight must never revive
        # a non-exported geometry node merely because its topology still fails.
        node.review_status = "rejected"
        skipped = enforce_geometry_export_topology_preflight(graph)
        self.assertEqual(node.review_status, "rejected")
        self.assertFalse(node.move_safe)
        self.assertEqual(
            skipped["skipped_rejected_node_ids"],
            ["GEO_SOLID_BANNER_AS_FRAME"],
        )
        self.assertNotIn("exported_full_support_topology", node.metadata)

    def _graph(self) -> DocumentGraph:
        graph = DocumentGraph((12, 10))
        crop = AlphaCrop(3, 2, np.full((5, 6), 255, dtype=np.uint8))
        graph.add_node(
            ElementNode(
                "N1",
                "Shape",
                "badge",
                crop,
                1,
                full_support=crop,
                semantic_envelope=crop,
                confidence=1.0,
                review_status="auto_confirmed",
                move_safe=True,
                rgba=np.dstack([np.full((5, 6, 3), 20, np.uint8), crop.alpha]),
            )
        )
        graph.add_proposal(
            ProposalRecord("P1", "test", "badge", crop.bbox, 1.0, "assigned", ["N1"])
        )
        return graph

    def test_pass_requires_clean_background_and_exact_composite(self) -> None:
        source = np.full((10, 12, 3), 240, dtype=np.uint8)
        source[2:7, 3:9] = 20
        background = np.full_like(source, 240)
        report = build_quality_report(
            self._graph(),
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
            format_reports={
                "psd": {"created": True, "roundtrip_qa": {"max_abs_error": 0}},
                "ora": {"created": True, "expected_composite_qa": {"max_abs_error": 0}},
            },
        )
        self.assertEqual(report["status"], "PASS")

    def test_source_remainder_is_excluded_from_foreground_ghost_metric(self) -> None:
        graph = self._graph()
        source = np.full((10, 12, 3), 240, dtype=np.uint8)
        source[2:7, 3:9] = 20
        remainder_mask = np.full((10, 12), 255, dtype=np.uint8)
        remainder_mask[2:7, 3:9] = 0
        remainder = AlphaCrop(0, 0, remainder_mask)
        graph.add_node(
            ElementNode(
                "BASE_SOURCE_RESIDUAL_0001",
                "Source remainder",
                "unknown",
                remainder,
                -3_000_000,
                full_support=remainder,
                review_status="auto_confirmed",
                rgba=np.dstack([source, remainder.alpha]),
                metadata={"role": "exact_source_remainder_above_clean_base"},
            )
        )
        report = build_quality_report(
            graph,
            source,
            np.full_like(source, 240),
            source.copy(),
            residual_background_text_regions=0,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["background_ghost"]["opaque_foreground_core_pixels"], 30
        )
        self.assertEqual(
            report["background_ghost"]["byte_identical_core_ratio"], 0.0
        )

    def test_recomposition_is_not_enough_when_ledger_is_unresolved(self) -> None:
        graph = self._graph()
        graph.add_proposal(ProposalRecord("P2", "edge", "unknown", (0, 0, 1, 1), 0.2))
        source = np.full((10, 12, 3), 240, dtype=np.uint8)
        source[2:7, 3:9] = 20
        report = build_quality_report(
            graph,
            source,
            np.full_like(source, 240),
            source.copy(),
            residual_background_text_regions=0,
        )
        self.assertEqual(report["status"], "REVIEW_REQUIRED")
        self.assertIn("proposals are unresolved", " ".join(report["review_reasons"]))

    def test_unaccounted_residual_pixels_fail_closed(self) -> None:
        source = np.full((10, 12, 3), 240, dtype=np.uint8)
        source[2:7, 3:9] = 20
        report = build_quality_report(
            self._graph(),
            source,
            np.full_like(source, 240),
            source.copy(),
            residual_background_text_regions=0,
            residual_report={
                "salient_pixel_accounting": {
                    "total": 30,
                    "assigned": 28,
                    "rejected": 0,
                    "unaccounted": 2,
                },
                "largest_visible_fraction": 0.05,
            },
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("residual pixels", " ".join(report["hard_failures"]))

    def test_documented_psd_limit_is_notice_when_ora_roundtrip_passes(self) -> None:
        source = np.full((10, 12, 3), 240, dtype=np.uint8)
        source[2:7, 3:9] = 20
        report = build_quality_report(
            self._graph(),
            source,
            np.full_like(source, 240),
            source.copy(),
            residual_background_text_regions=0,
            format_reports={
                "psd": {
                    "created": False,
                    "reason": "PSD limit: canvas dimension exceeds 30,000 px",
                },
                "ora": {
                    "created": True,
                    "expected_composite_qa": {"max_abs_error": 0},
                },
            },
        )
        self.assertEqual(report["status"], "PASS")
        self.assertIn("PSD omitted", report["container_notices"][0])

    def test_delivery_scale_recomposition_error_fails_closed(self) -> None:
        source = np.full((10, 12, 3), 240, dtype=np.uint8)
        source[2:7, 3:9] = 20
        report = build_quality_report(
            self._graph(),
            source,
            np.full_like(source, 240),
            source.copy(),
            residual_background_text_regions=0,
            delivery_render_report={"recomposition_max_abs_error": 7},
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("delivery-scale", " ".join(report["hard_failures"]))

    def test_fully_occluded_but_exportable_surface_is_not_empty(self) -> None:
        graph = self._graph()
        hidden = AlphaCrop(0, 0, np.zeros((2, 3), dtype=np.uint8))
        support = AlphaCrop(0, 0, np.full((2, 3), 255, dtype=np.uint8))
        graph.add_node(
            ElementNode(
                "LOWER",
                "Hidden lower rule",
                "line",
                hidden,
                0,
                full_support=support,
                semantic_envelope=support,
                review_status="auto_confirmed",
                occluded=True,
                synthesized_hidden_pixels=True,
                rgba=np.dstack([np.full((2, 3, 3), 240, np.uint8), support.alpha]),
            )
        )
        source = np.full((10, 12, 3), 240, dtype=np.uint8)
        source[2:7, 3:9] = 20
        report = build_quality_report(
            graph,
            source,
            np.full_like(source, 240),
            source.copy(),
            residual_background_text_regions=0,
        )
        self.assertNotIn("LOWER", report["empty_node_ids"])

    def _base_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        source = np.full((10, 12, 3), 240, dtype=np.uint8)
        source[2:7, 3:9] = 20
        return source, np.full_like(source, 240)

    def _geometry_cleanliness(
        self,
        *,
        support: int = 30,
        carved: int = 0,
        children: int = 0,
        remaining: int = 30,
        p95: float = 2.0,
        maximum: float = 3.0,
        limit: float = 10.0,
        high_delta_fraction: float = 0.1,
        max_high_delta_fraction: float = 0.35,
    ) -> dict[str, object]:
        return {
            "policy": "reference_surface_delta_e_carve_v1",
            "status": "pass",
            "reference_type": "constant_colour_rgb",
            "metric_domain": "visible_core_alpha_gte_192_excluding_known_children",
            "carved_destination_role": "exact_source_remainder_above_clean_base",
            "threshold_delta_e76": limit,
            "growth_threshold_delta_e76": limit * 0.75,
            "remaining_visible_delta_e76_limit": limit,
            "dilation_radius_px": 1,
            "support_pixel_count": support,
            "carved_pixel_count": carved,
            "carved_pixel_fraction_of_support": carved / support if support else 0.0,
            "high_delta_seed_fraction_of_support": high_delta_fraction,
            "max_auto_safe_high_delta_fraction": max_high_delta_fraction,
            "known_child_excluded_pixel_count": children,
            "remaining_visible_pixel_count": remaining,
            "remaining_visible_delta_e76_p95": p95,
            "remaining_visible_delta_e76_max": maximum,
        }

    def _text_purity(self, **updates: object) -> dict[str, object]:
        policy: dict[str, object] = {
            "policy": "source_mask_text_purity_v1",
            "status": "pass",
            "reasons": [],
            "giant_component": False,
            "multi_band": False,
            "adjacent_object": False,
            "protected_overlap_pixels": 0,
            "protected_overlap_fraction": 0.0,
            "residual_text_component_count": 3,
            "metrics": {
                "occupancy": 0.32,
                "component_count": 3,
                "largest_component_fraction": 0.55,
                "largest_component_bbox_fraction": 0.28,
                "largest_component_fill": 0.62,
                "row_band_count": 1,
                "primary_band_fraction": 0.95,
                "out_of_band_fraction": 0.05,
                "suspicious_component_count": 0,
            },
        }
        policy.update(updates)
        return policy

    def _text_export_purity(self, **updates: object) -> dict[str, object]:
        policy: dict[str, object] = {
            "policy": "exact_export_text_purity_v1",
            "status": "pass",
            "reasons": [],
            "exact_export_alpha_pixels": 40,
            "foreground_core": {
                "evaluated": True,
                "passed": True,
                "retained_fraction": 1.0,
            },
            "edge_band_evidence": {
                "passed": True,
                "isolated_edge_components": [],
            },
            "topology": {"implausibly_fragmented": False},
        }
        policy.update(updates)
        return policy

    def _geometry_topology_report(
        self,
        kind: str,
        support: np.ndarray,
        *,
        visible: np.ndarray | None = None,
        carved_fraction: float = 0.0,
    ) -> dict[str, object]:
        support_alpha = np.where(np.asarray(support) > 0, 255, 0).astype(np.uint8)
        visible_alpha = (
            support_alpha.copy()
            if visible is None
            else np.where(np.asarray(visible) > 0, 255, 0).astype(np.uint8)
        )
        height, width = support_alpha.shape
        graph = DocumentGraph((width, height))
        support_crop = AlphaCrop(0, 0, support_alpha)
        visible_crop = AlphaCrop(0, 0, visible_alpha)
        support_pixels = int(np.count_nonzero(support_alpha))
        visible_pixels = int(np.count_nonzero(visible_alpha))
        carved_pixels = min(
            support_pixels - visible_pixels,
            int(round(support_pixels * carved_fraction)),
        )
        child_pixels = max(0, support_pixels - carved_pixels - visible_pixels)
        graph.add_node(
            ElementNode(
                "GEOMETRY",
                f"Synthetic {kind}",
                kind,
                visible_crop,
                1,
                full_support=support_crop,
                semantic_envelope=support_crop,
                confidence=1.0,
                review_status="auto_confirmed",
                move_safe=True,
                synthesized_hidden_pixels=bool(carved_pixels),
                rgba=np.dstack(
                    [
                        np.full((height, width, 3), 20, dtype=np.uint8),
                        support_alpha,
                    ]
                ),
                metadata={
                    "geometry_backend": "poster_geometry_v2",
                    "geometry_cleanliness": self._geometry_cleanliness(
                        support=support_pixels,
                        carved=carved_pixels,
                        children=child_pixels,
                        remaining=visible_pixels,
                    ),
                },
            )
        )
        graph.add_proposal(
            ProposalRecord(
                "P_GEOMETRY",
                "poster_geometry_v2",
                kind,
                support_crop.bbox,
                1.0,
                "assigned",
                ["GEOMETRY"],
            )
        )

        if carved_pixels:
            hidden = np.where(
                (support_alpha > 0) & (visible_alpha == 0),
                255,
                0,
            ).astype(np.uint8)
            remainder = AlphaCrop(0, 0, hidden)
            source_for_remainder = np.full(
                (height, width, 3), 240, dtype=np.uint8
            )
            graph.add_node(
                ElementNode(
                    "BASE_SOURCE_RESIDUAL_0001",
                    "Source remainder",
                    "unknown",
                    remainder,
                    2_000_000,
                    full_support=remainder,
                    review_status="auto_confirmed",
                    move_safe=False,
                    rgba=np.dstack([source_for_remainder, hidden]),
                    metadata={"role": "exact_source_remainder_above_clean_base"},
                )
            )

        source = np.full((height, width, 3), 240, dtype=np.uint8)
        source[visible_alpha > 0] = 20
        return build_quality_report(
            graph,
            source,
            np.full_like(source, 240),
            source.copy(),
            residual_background_text_regions=0,
        )

    def test_clean_geometry_reference_policy_allows_auto_safe_node(self) -> None:
        graph = self._graph()
        graph.nodes[0].kind = "panel"
        graph.nodes[0].metadata.update(
            {
                "geometry_backend": "poster_geometry_v2",
                "geometry_cleanliness": self._geometry_cleanliness(),
            }
        )
        source, background = self._base_arrays()

        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["auto_safety_gate"]["contradiction_count"], 0)

    def test_exported_dirty_frame_fails_even_when_visible_alpha_is_empty(self) -> None:
        support = np.zeros((64, 160), dtype=np.uint8)
        support[2:10, 2:96] = 255
        support[24:31, 18:150] = 255
        support[46:60, 110:156] = 255
        report = self._geometry_topology_report(
            "frame",
            support,
            visible=np.zeros_like(support),
            carved_fraction=0.93,
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["layers"][0]["alpha"]["component_count"], 0)
        topology = report["layers"][0]["full_support_topology"]
        self.assertEqual(topology["metrics"]["component_count"], 3)
        self.assertEqual(topology["metrics"]["hole_count"], 0)
        self.assertIn(
            "frame_missing_dominant_interior_hole",
            topology["reasons"],
        )
        flag = report["auto_safety_gate"]["contradictions"][0]["flags"][0]
        self.assertEqual(
            flag["code"],
            "geometry_full_support_topology_rejected_auto_safety",
        )

    def test_exported_clean_frame_passes_when_fully_carved_from_visibility(self) -> None:
        support = np.zeros((28, 96), dtype=np.uint8)
        support[1:5, 1:95] = 255
        support[23:27, 1:95] = 255
        support[1:27, 1:5] = 255
        support[1:27, 91:95] = 255
        report = self._geometry_topology_report(
            "frame",
            support,
            visible=np.zeros_like(support),
            carved_fraction=1.0,
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["auto_safety_gate"]["contradiction_count"], 0)
        topology = report["layers"][0]["full_support_topology"]
        self.assertEqual(topology["status"], "pass")
        self.assertEqual(topology["metrics"]["substantial_component_count"], 1)
        self.assertEqual(topology["metrics"]["meaningful_hole_count"], 1)

    def test_panel_topology_requires_one_solid_filled_surface(self) -> None:
        clean = np.ones((24, 70), dtype=np.uint8)
        clean_report = self._geometry_topology_report("panel", clean)
        self.assertEqual(clean_report["status"], "PASS")

        hollow = clean.copy()
        hollow[5:19, 16:54] = 0
        hollow_report = self._geometry_topology_report("panel", hollow)
        self.assertEqual(hollow_report["status"], "FAIL")
        topology = hollow_report["layers"][0]["full_support_topology"]
        self.assertIn("panel_has_meaningful_interior_holes", topology["reasons"])

    def test_line_topology_rejects_a_large_attached_cross_axis_object(self) -> None:
        clean = np.ones((3, 72), dtype=np.uint8)
        clean_report = self._geometry_topology_report("line", clean)
        self.assertEqual(clean_report["status"], "PASS")

        contaminated = np.zeros((22, 72), dtype=np.uint8)
        contaminated[2:5, :] = 1
        contaminated[2:20, 22:42] = 1
        contaminated_report = self._geometry_topology_report("line", contaminated)
        self.assertEqual(contaminated_report["status"], "FAIL")
        topology = contaminated_report["layers"][0]["full_support_topology"]
        self.assertTrue(
            {
                "line_is_too_thick_for_its_span",
                "line_has_large_cross_axis_intrusion",
            }
            & set(topology["reasons"])
        )

    def test_legacy_geometry_without_cleanliness_policy_fails_closed(self) -> None:
        graph = self._graph()
        graph.nodes[0].kind = "frame"
        graph.nodes[0].metadata["geometry_backend"] = "poster_geometry_v2"
        source, background = self._base_arrays()

        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "missing_geometry_cleanliness_policy",
            {
                flag["code"]
                for flag in report["auto_safety_gate"]["contradictions"][0]["flags"]
            },
        )

    def test_synthesized_geometry_without_cleanliness_policy_fails_closed(self) -> None:
        graph = self._graph()
        graph.nodes[0].kind = "panel"
        graph.nodes[0].synthesized_hidden_pixels = True
        source, background = self._base_arrays()

        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "missing_geometry_cleanliness_policy",
            {
                flag["code"]
                for flag in report["auto_safety_gate"]["contradictions"][0]["flags"]
            },
        )

    def test_residual_and_layerd_geometry_need_explicit_clean_reference(self) -> None:
        source, background = self._base_arrays()
        for evidence_source, metadata in (
            ("poster_surface_residual", {"surface_reconciliation": True}),
            ("layerd_iterative_top_layer", {}),
        ):
            with self.subTest(source=evidence_source):
                graph = self._graph()
                graph.nodes[0].kind = "panel"
                graph.nodes[0].evidence = [
                    {"proposal_id": "P1", "source": evidence_source}
                ]
                graph.nodes[0].metadata.update(metadata)
                report = build_quality_report(
                    graph,
                    source,
                    background,
                    source.copy(),
                    residual_background_text_regions=0,
                )
                self.assertEqual(report["status"], "FAIL")
                self.assertIn(
                    "missing_geometry_cleanliness_policy",
                    {
                        flag["code"]
                        for flag in report["auto_safety_gate"]["contradictions"][0]["flags"]
                    },
                )

    def test_geometry_remaining_contamination_above_limit_fails(self) -> None:
        graph = self._graph()
        graph.nodes[0].kind = "line"
        graph.nodes[0].metadata.update(
            {
                "geometry_backend": "poster_geometry_v2",
                "geometry_cleanliness": self._geometry_cleanliness(
                    p95=10.01,
                    maximum=12.0,
                    limit=10.0,
                ),
            }
        )
        source, background = self._base_arrays()

        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "geometry_remaining_visible_contamination",
            {
                flag["code"]
                for flag in report["auto_safety_gate"]["contradictions"][0]["flags"]
            },
        )

    def test_geometry_carve_accounting_contradiction_fails(self) -> None:
        graph = self._graph()
        graph.nodes[0].kind = "panel"
        policy = self._geometry_cleanliness(carved=31, remaining=0)
        policy["carved_pixel_fraction_of_support"] = 0.5
        graph.nodes[0].metadata.update(
            {
                "geometry_backend": "poster_geometry_v2",
                "geometry_cleanliness": policy,
            }
        )
        source, background = self._base_arrays()

        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "geometry_cleanliness_accounting_contradiction",
            {
                flag["code"]
                for flag in report["auto_safety_gate"]["contradictions"][0]["flags"]
            },
        )

    def test_geometry_high_delta_fraction_cannot_exceed_safe_limit(self) -> None:
        graph = self._graph()
        graph.nodes[0].kind = "panel"
        graph.nodes[0].metadata.update(
            {
                "geometry_backend": "poster_geometry_v2",
                "geometry_cleanliness": self._geometry_cleanliness(
                    high_delta_fraction=0.36,
                    max_high_delta_fraction=0.35,
                ),
            }
        )
        source, background = self._base_arrays()

        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "geometry_cleanliness_accounting_contradiction",
            {
                flag["code"]
                for flag in report["auto_safety_gate"]["contradictions"][0]["flags"]
            },
        )

    def test_reported_geometry_carve_must_exist_in_source_remainder(self) -> None:
        graph = self._graph()
        node = graph.nodes[0]
        node.kind = "panel"
        visible = node.visible_alpha.alpha.copy()
        visible[2, 3] = 0
        node.visible_alpha = AlphaCrop(node.visible_alpha.left, node.visible_alpha.top, visible)
        node.metadata.update(
            {
                "geometry_backend": "poster_geometry_v2",
                "geometry_cleanliness": self._geometry_cleanliness(
                    carved=1,
                    remaining=29,
                ),
            }
        )
        source, background = self._base_arrays()

        missing = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )
        self.assertEqual(missing["status"], "FAIL")
        self.assertIn(
            "geometry_carve_source_remainder_contradiction",
            {
                flag["code"]
                for flag in missing["auto_safety_gate"]["contradictions"][0]["flags"]
            },
        )

        remainder_alpha = np.zeros((10, 12), dtype=np.uint8)
        remainder_alpha[node.visible_alpha.top + 2, node.visible_alpha.left + 3] = 255
        remainder = AlphaCrop(0, 0, remainder_alpha)
        graph.add_node(
            ElementNode(
                "BASE_SOURCE_RESIDUAL_0001",
                "Source remainder",
                "unknown",
                remainder,
                2_000_000,
                full_support=remainder,
                review_status="auto_confirmed",
                move_safe=False,
                rgba=np.dstack([source.copy(), remainder.alpha]),
                metadata={"role": "exact_source_remainder_above_clean_base"},
            )
        )
        clean = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )
        self.assertEqual(clean["status"], "PASS")
        self.assertEqual(clean["auto_safety_gate"]["contradiction_count"], 0)

    def test_clean_text_purity_policy_allows_auto_safe_text(self) -> None:
        graph = self._graph()
        graph.nodes[0].kind = "text"
        graph.nodes[0].metadata["text_purity"] = self._text_purity()
        graph.nodes[0].metadata[
            "text_export_purity_preflight"
        ] = self._text_export_purity()
        source, background = self._base_arrays()

        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["auto_safety_gate"]["contradiction_count"], 0)

    def test_auto_text_without_exact_export_preflight_fails_closed(self) -> None:
        graph = self._graph()
        graph.nodes[0].kind = "text"
        graph.nodes[0].metadata["text_purity"] = self._text_purity()
        source, background = self._base_arrays()

        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "missing_exact_export_text_purity_preflight",
            {
                flag["code"]
                for flag in report["auto_safety_gate"]["contradictions"][0][
                    "flags"
                ]
            },
        )

    def test_legacy_auto_text_without_purity_policy_fails_closed(self) -> None:
        graph = self._graph()
        graph.nodes[0].kind = "price"
        source, background = self._base_arrays()

        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "missing_text_purity_policy",
            {
                flag["code"]
                for flag in report["auto_safety_gate"]["contradictions"][0]["flags"]
            },
        )

    def test_impure_auto_text_is_a_hard_contradiction(self) -> None:
        graph = self._graph()
        graph.nodes[0].kind = "text"
        graph.nodes[0].metadata["text_purity"] = self._text_purity(
            status="unsafe",
            reasons=["adjacent_object_overlap"],
            giant_component=True,
            multi_band=True,
            adjacent_object=True,
            protected_overlap_pixels=4,
            protected_overlap_fraction=0.08,
        )
        source, background = self._base_arrays()

        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "text_purity_rejected_auto_confirmation",
            {
                flag["code"]
                for flag in report["auto_safety_gate"]["contradictions"][0]["flags"]
            },
        )

    def test_failed_refinement_cannot_be_auto_confirmed_or_move_safe(self) -> None:
        graph = self._graph()
        graph.nodes[0].metadata["refinement"] = {
            "accepted": False,
            "reason": "BiRefNet disagrees with SAM envelope",
        }
        source, background = self._base_arrays()
        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )
        self.assertEqual(report["status"], "FAIL")
        gate = report["auto_safety_gate"]
        self.assertEqual(gate["contradiction_count"], 1)
        self.assertEqual(gate["contradictions"][0]["id"], "N1")
        self.assertEqual(
            gate["contradictions"][0]["flags"][0]["code"],
            "failed_refinement_marked_auto_safe",
        )

    def test_semantic_ambiguity_cannot_be_advertised_as_auto_safe(self) -> None:
        graph = self._graph()
        graph.nodes[0].metadata.update(
            {
                "semantic_auto_confirmable": False,
                "semantic_kind_consistent": False,
                "semantic_ambiguity_reasons": ["detector_kind_conflict"],
            }
        )
        source, background = self._base_arrays()
        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "semantic_policy_rejected_auto_confirmation",
            {
                flag["code"]
                for flag in report["auto_safety_gate"]["contradictions"][0]["flags"]
            },
        )

    def test_rejected_layerd_cluster_policy_cannot_be_auto_confirmed(self) -> None:
        graph = self._graph()
        graph.nodes[0].metadata["auto_confirmation_policy"] = {
            "policy": "layerd_raw_cluster_fail_closed_v1",
            "eligible": False,
            "decision": "defer_to_review",
            "reasons": ["sparse_multi_island_support"],
            "evidence": {"member_component_count": 5},
        }
        source, background = self._base_arrays()
        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "consolidation_policy_rejected_auto_confirmation",
            {
                flag["code"]
                for flag in report["auto_safety_gate"]["contradictions"][0]["flags"]
            },
        )

    def test_layerd_defer_decision_is_binding_even_if_eligible_field_is_wrong(self) -> None:
        graph = self._graph()
        graph.nodes[0].metadata["auto_confirmation_policy"] = {
            "policy": "layerd_raw_cluster_fail_closed_v1",
            "eligible": True,
            "decision": "defer_to_review",
            "reasons": [],
        }
        source, background = self._base_arrays()
        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )
        self.assertEqual(report["status"], "FAIL")
        flag = report["auto_safety_gate"]["contradictions"][0]["flags"][0]
        self.assertEqual(flag["decision"], "defer_to_review")

    def test_legacy_large_consolidation_is_fail_closed(self) -> None:
        graph = self._graph()
        graph.nodes[0].metadata.update(
            {
                "member_component_count": 9,
                "consolidation": "nearby colour-compatible islands",
            }
        )
        source, background = self._base_arrays()
        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(
            report["auto_safety_gate"]["contradictions"][0]["flags"][0]["code"],
            "legacy_high_risk_multi_component_consolidation",
        )

    def test_manual_non_move_safe_override_may_retain_failed_refinement(self) -> None:
        graph = self._graph()
        graph.nodes[0].review_status = "user_confirmed"
        graph.nodes[0].move_safe = False
        graph.nodes[0].metadata["refinement"] = {
            "accepted": False,
            "reason": "fallback retained for a human-approved static layer",
        }
        source, background = self._base_arrays()
        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["auto_safety_gate"]["contradiction_count"], 0)

    def test_unresolved_failed_refinement_remains_review_required_not_hard_failure(self) -> None:
        graph = self._graph()
        graph.nodes[0].review_status = "unresolved"
        graph.nodes[0].move_safe = False
        graph.nodes[0].metadata["refinement"] = {"accepted": False}
        source, background = self._base_arrays()
        report = build_quality_report(
            graph,
            source,
            background,
            source.copy(),
            residual_background_text_regions=0,
        )
        self.assertEqual(report["status"], "REVIEW_REQUIRED")
        self.assertEqual(report["auto_safety_gate"]["contradiction_count"], 0)
        self.assertFalse(report["hard_failures"])


if __name__ == "__main__":
    unittest.main()
