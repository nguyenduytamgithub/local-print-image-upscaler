from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from layer_engine_v5_pro import (
    REVIEW_RESUME_SIGNATURE,
    _add_base_residual_layer,
    _apply_compatible_previous_review,
    _protected_alpha,
    _preserve_review_metadata,
    _print_qa_hard_failures,
    _reconcile_base_residual_layers,
    parse_args,
    validate_args,
)
from v5pro.cleanplate import build_clean_plate
from v5pro.exporter import render_document
from v5pro.geometry_backend import extract_geometry_layers
from v5pro.hierarchy import AUTO_ORGANIZATION_SOURCE, assign_organizational_groups
from v5pro.inventory import DetectedProposal
from v5pro.ownership import enforce_exclusive_ownership
from v5pro.review_adapter import prepare_review_bundle
from v5pro.review_server import ReviewSession
from v5pro.schema import AlphaCrop, DocumentGraph, ElementNode, ProposalRecord


class EngineReviewResumeV2Tests(unittest.TestCase):
    def test_fail_console_lists_every_hard_qa_reason_concisely(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            _print_qa_hard_failures(
                {
                    "status": "FAIL",
                    "hard_failures": [
                        "  first hard failure\nwith detail  ",
                        "second hard failure",
                    ],
                }
            )

        text = output.getvalue()
        self.assertIn("LỖI QA CỨNG", text)
        self.assertIn("1. first hard failure with detail", text)
        self.assertIn("2. second hard failure", text)

        output = io.StringIO()
        with redirect_stdout(output):
            _print_qa_hard_failures(
                {"status": "PASS", "hard_failures": ["must stay silent"]}
            )
        self.assertEqual(output.getvalue(), "")

    def _poster_with_base(
        self,
    ) -> tuple[np.ndarray, np.ndarray, DocumentGraph]:
        surface = np.full((48, 72, 3), (248, 238, 190), dtype=np.uint8)
        source = surface.copy()
        source[15:34, 20:55] = (25, 120, 45)
        graph = DocumentGraph(
            (72, 48),
            metadata={"review_resume_signature": REVIEW_RESUME_SIGNATURE},
        )
        crop = AlphaCrop(20, 15, np.full((19, 35), 255, dtype=np.uint8))
        graph.add_node(
            ElementNode(
                "OBJECT",
                "Object",
                "product",
                crop,
                10,
                full_support=crop,
                semantic_envelope=crop,
                confidence=0.95,
                review_status="auto_confirmed",
                rgba=np.dstack([source[15:34, 20:55], crop.alpha]),
            )
        )
        graph.add_proposal(
            ProposalRecord("OBJECT_PROPOSAL", "detector", "product", crop.bbox, 0.95, "assigned", ["OBJECT"])
        )
        _add_base_residual_layer(graph, source, surface)
        return source, surface, graph

    def test_split_remainder_keeps_provenance_and_cleanplate_scope(self) -> None:
        source, surface, graph = self._poster_with_base()
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = prepare_review_bundle(
                graph, source, Path(raw), source_sha256="abc", icc_profile=None
            )
            session = ReviewSession.open(checkpoint)
            session.split_node_by_rectangle(
                "BASE_SOURCE_RESIDUAL_0001", [0, 0, 36, 48]
            )
            restored, _document = _apply_compatible_previous_review(
                graph, checkpoint, source
            )
        enforce_exclusive_ownership(restored, source)
        report = _reconcile_base_residual_layers(restored, source, surface)
        base_nodes = [
            node
            for node in restored.nodes
            if node.metadata.get("role")
            == "exact_source_remainder_above_clean_base"
        ]
        self.assertEqual(len(base_nodes), 2)
        self.assertTrue(report["coverage_exact"])
        clean = build_clean_plate(
            source, restored, mode="poster", poster_surface_rgb=surface
        )
        self.assertEqual(clean.report["requested_union_pixels"], 19 * 35)
        self.assertEqual(clean.report["clean_plate_audit"]["grade"], "pass")
        rendered = render_document(restored, source, clean.background_rgb, scale=1.0)
        self.assertTrue(np.array_equal(np.asarray(rendered.composite), source))

    def test_local_rectangle_proposal_survives_resume_and_trims_remainder(self) -> None:
        source, surface, old_graph = self._poster_with_base()
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = prepare_review_bundle(
                old_graph, source, Path(raw), source_sha256="abc", icc_profile=None
            )
            session = ReviewSession.open(checkpoint)
            state = session.add_rectangular_proposal(
                [2, 2, 12, 10], kind="micro_detail"
            )
            proposal_id = next(
                item["id"]
                for item in state["proposals"]
                if item["id"].startswith("USER_PROPOSAL")
            )
            state = session.accept_proposal(proposal_id)
            user_node_id = next(
                item["id"]
                for item in state["nodes"]
                if item["id"].startswith("USER_USER_PROPOSAL")
            )
            session.accept_node(user_node_id)
            _source2, _surface2, fresh_graph = self._poster_with_base()
            restored, _document = _apply_compatible_previous_review(
                fresh_graph, checkpoint, source
            )
        enforce_exclusive_ownership(restored, source)
        _reconcile_base_residual_layers(restored, source, surface)
        self.assertIn(user_node_id, restored.node_map())
        self.assertTrue(
            any(
                proposal.proposal_id == proposal_id
                and proposal.source == "local_review_rectangle"
                for proposal in restored.proposals
            )
        )
        self.assertEqual(int(restored.ownership_maps()[0].max()), 1)
        rendered = render_document(restored, source, surface, scale=1.0)
        self.assertTrue(np.array_equal(np.asarray(rendered.composite), source))

    def test_delete_false_layer_returns_its_pixels_to_source_remainder(self) -> None:
        source, surface, graph = self._poster_with_base()
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = prepare_review_bundle(
                graph, source, Path(raw), source_sha256="abc", icc_profile=None
            )
            ReviewSession.open(checkpoint).delete_node("OBJECT")
            restored, _document = _apply_compatible_previous_review(
                graph, checkpoint, source
            )
        enforce_exclusive_ownership(restored, source)
        _reconcile_base_residual_layers(restored, source, surface)
        base = next(
            node
            for node in restored.nodes
            if node.metadata.get("role")
            == "exact_source_remainder_above_clean_base"
        )
        base_canvas = base.visible_alpha.to_canvas(restored.canvas_size)
        self.assertTrue(np.all(base_canvas[15:34, 20:55] == 255))
        rendered = render_document(restored, source, surface, scale=1.0)
        self.assertTrue(np.array_equal(np.asarray(rendered.composite), source))

    def test_reemitted_checkpoint_matches_reconciled_graph_and_keeps_valid_group(self) -> None:
        source, surface, graph = self._poster_with_base()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint = prepare_review_bundle(
                graph, source, root, source_sha256="abc", icc_profile=None
            )
            session = ReviewSession.open(checkpoint)
            session.merge_group(
                ["BASE_SOURCE_RESIDUAL_0001", "OBJECT"], name="Artwork"
            )
            review_document = json.loads(checkpoint.read_text(encoding="utf-8"))
            restored, _previous = _apply_compatible_previous_review(
                graph, checkpoint, source
            )
            enforce_exclusive_ownership(restored, source)
            _reconcile_base_residual_layers(restored, source, surface)
            final_checkpoint = prepare_review_bundle(
                restored, source, root, source_sha256="abc", icc_profile=None
            )
            _preserve_review_metadata(final_checkpoint, review_document)
            final = json.loads(final_checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(
            {record["id"] for record in final["nodes"]},
            set(restored.node_map()),
        )
        self.assertEqual(
            final["groups"][0]["member_ids"],
            ["BASE_SOURCE_RESIDUAL_0001", "OBJECT"],
        )

    def test_direct_engine_rejects_fractional_scale(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.png"
            Image.new("RGB", (8, 8), "white").save(source)
            args = parse_args([str(source), "1.5", str(root / "output")])
            with self.assertRaisesRegex(SystemExit, "integer scale"):
                validate_args(args)

    def test_base_residual_is_exclusive_hideable_and_preserves_source(self) -> None:
        surface = np.full((10, 12, 3), 230, dtype=np.uint8)
        source = surface.copy()
        source[1:9, 2:11] = [210, 215, 220]
        source[3:6, 4:8] = [20, 80, 150]
        graph = DocumentGraph((12, 10))
        alpha = AlphaCrop(4, 3, np.full((3, 4), 255, dtype=np.uint8))
        graph.add_node(
            ElementNode(
                "PRODUCT",
                "Product",
                "product",
                alpha,
                10,
                full_support=alpha,
                confidence=0.95,
                review_status="auto_confirmed",
                rgba=np.dstack([source[3:6, 4:8], alpha.alpha]),
            )
        )

        report = _add_base_residual_layer(graph, source, surface)
        remainder = graph.node_map()["BASE_SOURCE_RESIDUAL_0001"]
        remainder_canvas = remainder.visible_alpha.to_canvas(graph.canvas_size)
        self.assertEqual(int(np.count_nonzero(remainder_canvas[3:6, 4:8])), 0)
        self.assertEqual(int(np.max(graph.ownership_maps()[0])), 1)
        self.assertEqual(report["pixel_count"], 108)
        self.assertEqual(graph.proposal_accounting()["assigned"], 1)

        rendered = render_document(graph, source, surface, scale=1.0)
        self.assertTrue(np.array_equal(np.asarray(rendered.composite), source))
        self.assertTrue(np.array_equal(np.asarray(rendered.background), surface))

    def test_dirty_panel_text_moves_to_top_remainder_and_clean_panel_recomposes_exactly(self) -> None:
        height, width = 200, 300
        clean = np.full((height, width, 3), (232, 226, 214), dtype=np.uint8)
        cv2.rectangle(clean, (30, 30), (269, 169), (250, 248, 243), -1, cv2.LINE_8)
        cv2.rectangle(clean, (30, 30), (269, 169), (25, 115, 190), 6, cv2.LINE_AA)
        source = clean.copy()
        dirty_text = np.zeros((height, width), dtype=np.uint8)
        cv2.putText(
            dirty_text, "99.000", (82, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 4, cv2.LINE_AA
        )
        cv2.putText(
            source, "99.000", (82, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 35, 45), 4, cv2.LINE_AA
        )
        detected = DetectedProposal(
            ProposalRecord(
                "DIRTY_PANEL_FRAME",
                "synthetic_test",
                "frame",
                (30, 30, 270, 170),
                0.98,
            )
        )
        geometry = extract_geometry_layers(source, [detected], z_start=100)
        graph = DocumentGraph(
            (width, height),
            nodes=list(geometry.nodes),
            proposals=list(geometry.proposals),
        )
        panel = next(node for node in graph.nodes if node.kind == "panel")
        panel_full = panel.full_support.to_canvas(graph.canvas_size) > 0
        panel_visible = panel.visible_alpha.to_canvas(graph.canvas_size) > 0
        self.assertGreater(int(np.count_nonzero(panel_full & (dirty_text > 0))), 600)
        self.assertEqual(int(np.count_nonzero(panel_visible & (dirty_text > 0))), 0)
        self.assertTrue(panel.move_safe)

        # A raw LayerD guess may still claim the same unrecognised text before
        # categorical arbitration. The geometry reservation must transfer it
        # to the technical remainder rather than let the raw ghost survive.
        ys, xs = np.where(dirty_text > 0)
        left, top, right, bottom = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
        raw_alpha = dirty_text[top:bottom, left:right].copy()
        raw_crop = AlphaCrop(left, top, raw_alpha)
        graph.add_node(
            ElementNode(
                "RAW_DIRTY_TEXT",
                "Raw dirty text guess",
                "unknown",
                raw_crop,
                1,
                full_support=raw_crop,
                confidence=0.4,
                review_status="unresolved",
                move_safe=False,
                rgba=np.dstack([source[top:bottom, left:right].copy(), raw_alpha]),
                evidence=[{"source": "layerd_iterative_top_layer"}],
            )
        )
        graph.add_proposal(
            ProposalRecord(
                "RAW_DIRTY_TEXT_PROPOSAL",
                "layerd_iterative_top_layer",
                "unknown",
                raw_crop.bbox,
                0.4,
                "assigned",
                ["RAW_DIRTY_TEXT"],
            )
        )
        graph.add_node(
            ElementNode(
                "RESIDUAL_DIRTY_PANEL",
                "Unvalidated residual panel",
                "panel",
                raw_crop,
                2,
                full_support=raw_crop,
                confidence=0.5,
                review_status="unresolved",
                move_safe=False,
                rgba=np.dstack([source[top:bottom, left:right].copy(), raw_alpha]),
                evidence=[{"source": "poster_surface_residual"}],
                metadata={
                    "surface_reconciliation": True,
                    "geometry_cleanliness": {
                        "policy": "reference_surface_delta_e_carve_v1",
                        "status": "unsafe",
                        "reference_type": "none",
                    },
                },
            )
        )
        graph.add_proposal(
            ProposalRecord(
                "RESIDUAL_DIRTY_PANEL_PROPOSAL",
                "poster_surface_residual",
                "panel",
                raw_crop.bbox,
                0.5,
                "assigned",
                ["RESIDUAL_DIRTY_PANEL"],
            )
        )
        _add_base_residual_layer(graph, source, clean)
        enforce_exclusive_ownership(graph, source)
        report = _reconcile_base_residual_layers(graph, source, clean)
        self.assertNotIn("RAW_DIRTY_TEXT", graph.node_map())
        self.assertNotIn("RESIDUAL_DIRTY_PANEL", graph.node_map())
        raw_record = next(
            item for item in graph.proposals if item.proposal_id == "RAW_DIRTY_TEXT_PROPOSAL"
        )
        self.assertTrue(raw_record.owner_ids)
        self.assertTrue(
            all(owner.startswith("BASE_SOURCE_RESIDUAL_") for owner in raw_record.owner_ids)
        )
        residual_record = next(
            item
            for item in graph.proposals
            if item.proposal_id == "RESIDUAL_DIRTY_PANEL_PROPOSAL"
        )
        self.assertTrue(residual_record.owner_ids)
        self.assertTrue(
            all(
                owner.startswith("BASE_SOURCE_RESIDUAL_")
                for owner in residual_record.owner_ids
            )
        )
        remainder = next(
            node
            for node in graph.nodes
            if node.metadata.get("role") == "exact_source_remainder_above_clean_base"
        )
        remainder_visible = remainder.visible_alpha.to_canvas(graph.canvas_size) > 0
        self.assertEqual(
            int(np.count_nonzero(remainder_visible & (dirty_text > 0))),
            int(np.count_nonzero(dirty_text > 0)),
        )
        self.assertGreater(
            remainder.z_index,
            max(node.z_index for node in graph.nodes if node is not remainder),
        )
        self.assertTrue(report["coverage_exact"])
        self.assertGreater(report["lower_full_support_overlap_pixels"], 600)
        self.assertEqual(report["visible_overlap_with_extracted_pixels"], 0)
        self.assertEqual(int(np.max(graph.ownership_maps()[0])), 1)

        rendered = render_document(graph, source, clean, scale=1.0)
        self.assertEqual(rendered.report["recomposition_max_abs_error"], 0)
        self.assertTrue(np.array_equal(np.asarray(rendered.composite), source))
        rendered_panel = next(
            layer for layer in rendered.rendered if layer.spec.layer_id == panel.element_id
        )
        panel_png = np.asarray(rendered_panel.rgba)
        x0, y0, x1, y1 = panel.bbox
        local_text = dirty_text[y0:y1, x0:x1] > 0
        panel_pixels = panel_png[:, :, :3][local_text & (panel_png[:, :, 3] > 0)]
        self.assertTrue(len(panel_pixels))
        self.assertLessEqual(
            int(np.max(np.abs(panel_pixels.astype(np.int16) - np.array((250, 248, 243), np.int16)))),
            1,
        )
        self.assertNotEqual(panel.bbox, (0, 0, width, height))

    def test_source_remainder_beats_crossing_geometry_but_not_semantic_child(self) -> None:
        height, width = 20, 30
        clean = np.full((height, width, 3), (242, 239, 232), dtype=np.uint8)
        source = clean.copy()
        source[5:15, 6:24] = (35, 85, 145)
        source[8:11, 10:14] = (210, 45, 30)
        graph = DocumentGraph((width, height))

        full_alpha = np.full((10, 18), 255, dtype=np.uint8)
        carved_alpha = np.zeros_like(full_alpha)
        carved = ElementNode(
            "GEO_CARVED",
            "Clean carved geometry",
            "line",
            AlphaCrop(6, 5, carved_alpha),
            100,
            full_support=AlphaCrop(6, 5, full_alpha.copy()),
            confidence=0.95,
            review_status="auto_confirmed",
            move_safe=True,
            rgba=np.dstack(
                [
                    np.full((10, 18, 3), (60, 155, 75), dtype=np.uint8),
                    full_alpha.copy(),
                ]
            ),
            evidence=[{"source": "opencv_layout_line"}],
            metadata={
                "geometry_backend": "poster_geometry_v2",
                "geometry_cleanliness": {
                    "policy": "reference_surface_delta_e_carve_v1",
                    "status": "pass",
                    "reference_type": "constant_colour_rgb",
                    "carved_pixel_count": int(full_alpha.size),
                },
            },
        )
        crossing = ElementNode(
            "GEO_CROSSING",
            "Crossing verified geometry",
            "frame",
            AlphaCrop(6, 5, full_alpha.copy()),
            200,
            full_support=AlphaCrop(6, 5, full_alpha.copy()),
            confidence=0.95,
            review_status="auto_confirmed",
            move_safe=True,
            rgba=np.dstack(
                [
                    np.full((10, 18, 3), (235, 190, 35), dtype=np.uint8),
                    full_alpha.copy(),
                ]
            ),
            evidence=[{"source": "opencv_layout_frame"}],
            metadata={"geometry_backend": "poster_geometry_v2"},
        )
        child_alpha = np.full((3, 4), 255, dtype=np.uint8)
        child = ElementNode(
            "SEMANTIC_CHILD",
            "Known semantic child",
            "product",
            AlphaCrop(10, 8, child_alpha),
            300,
            full_support=AlphaCrop(10, 8, child_alpha.copy()),
            confidence=0.98,
            review_status="auto_confirmed",
            move_safe=True,
            rgba=np.dstack([source[8:11, 10:14].copy(), child_alpha.copy()]),
            evidence=[{"source": "grounding_dino_sam2"}],
            metadata={"semantic_extraction": {"test": True}},
        )
        graph.add_node(carved)
        graph.add_node(crossing)
        graph.add_node(child)

        initial = _add_base_residual_layer(graph, source, clean)
        self.assertEqual(initial["geometry_hidden_reservation_pixels"], full_alpha.size)
        ownership = enforce_exclusive_ownership(graph, source)
        reconciliation = _reconcile_base_residual_layers(graph, source, clean)

        remainder = graph.node_map()["BASE_SOURCE_RESIDUAL_0001"]
        remainder_mask = remainder.visible_alpha.to_canvas(graph.canvas_size) > 0
        carve_region = np.zeros((height, width), dtype=bool)
        carve_region[5:15, 6:24] = True
        child_region = np.zeros((height, width), dtype=bool)
        child_region[8:11, 10:14] = True
        self.assertEqual(
            int(np.count_nonzero(remainder_mask & carve_region)),
            int(np.count_nonzero(carve_region & ~child_region)),
        )
        self.assertFalse(np.any(remainder_mask & child_region))
        self.assertEqual(
            graph.node_map()["GEO_CROSSING"].visible_alpha.nonzero_pixels,
            0,
        )
        self.assertEqual(
            graph.node_map()["GEO_CROSSING"].full_support.nonzero_pixels,
            full_alpha.size,
        )
        self.assertEqual(
            graph.node_map()["SEMANTIC_CHILD"].visible_alpha.nonzero_pixels,
            child_alpha.size,
        )
        self.assertEqual(ownership["priority_band_node_counts"]["450"], 1)
        self.assertTrue(reconciliation["coverage_exact"])
        self.assertEqual(reconciliation["visible_overlap_with_extracted_pixels"], 0)
        self.assertEqual(int(graph.ownership_maps()[0].max()), 1)

        rendered = render_document(graph, source, clean, scale=1.0)
        self.assertEqual(rendered.report["recomposition_max_abs_error"], 0)
        self.assertTrue(np.array_equal(np.asarray(rendered.composite), source))

    def test_geometry_protection_excludes_raw_layerd_surface_clusters(self) -> None:
        source = np.full((20, 30, 3), 240, dtype=np.uint8)
        graph = DocumentGraph((30, 20))
        raw = AlphaCrop(0, 0, np.full((20, 30), 255, dtype=np.uint8))
        text = AlphaCrop(8, 7, np.full((3, 8), 255, dtype=np.uint8))
        graph.add_node(
            ElementNode(
                "RAW",
                "Raw LayerD cluster",
                "unknown",
                raw,
                1,
                full_support=raw,
                confidence=0.5,
                review_status="unresolved",
                rgba=np.dstack([source, raw.alpha]),
                evidence=[{"source": "layerd_iterative_top_layer"}],
            )
        )
        graph.add_node(
            ElementNode(
                "TEXT",
                "Text",
                "text",
                text,
                2,
                full_support=text,
                confidence=0.9,
                review_status="auto_confirmed",
                rgba=np.dstack([source[7:10, 8:16], text.alpha]),
                evidence=[{"source": "tesseract_vie_geometry"}],
            )
        )
        protected = _protected_alpha(graph)
        self.assertEqual(int(np.count_nonzero(protected)), 24)
        self.assertTrue(np.all(protected[7:10, 8:16] == 255))

    def _graph(self, source: np.ndarray) -> DocumentGraph:
        crop = AlphaCrop(3, 2, np.full((4, 6), 255, dtype=np.uint8))
        graph = DocumentGraph(
            (source.shape[1], source.shape[0]),
            metadata={"review_resume_signature": REVIEW_RESUME_SIGNATURE},
        )
        graph.add_node(
            ElementNode(
                "TEXT_1",
                "Text",
                "text",
                crop,
                1,
                full_support=crop,
                confidence=0.9,
                review_status="unresolved",
                rgba=np.dstack([source[2:6, 3:9], crop.alpha]),
            )
        )
        graph.add_proposal(
            ProposalRecord("OCR_1", "ocr", "text", crop.bbox, 0.9, "assigned", ["TEXT_1"])
        )
        return graph

    def test_resume_keeps_user_decision_and_finished_state(self) -> None:
        source = np.full((12, 16, 3), 235, dtype=np.uint8)
        graph = self._graph(source)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint = prepare_review_bundle(
                graph,
                source,
                root,
                source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                icc_profile=None,
            )
            document = json.loads(checkpoint.read_text(encoding="utf-8"))
            document["nodes"][0]["review_status"] = "user_confirmed"
            document["review"]["finished"] = True
            document["groups"] = [{"id": "G", "name": "Text", "member_ids": ["TEXT_1"]}]
            checkpoint.write_text(json.dumps(document), encoding="utf-8")
            restored, previous = _apply_compatible_previous_review(graph, checkpoint, source)
            self.assertEqual(restored.nodes[0].review_status, "user_confirmed")
            fresh_root = root / "fresh"
            fresh = prepare_review_bundle(
                restored,
                source,
                fresh_root,
                source_sha256="new",
                icc_profile=None,
            )
            _preserve_review_metadata(fresh, previous)
            final = json.loads(fresh.read_text(encoding="utf-8"))
        self.assertTrue(final["review"]["finished"])
        self.assertEqual(final["groups"], [])

    def test_preserve_review_keeps_explicit_group_and_refreshes_auto_folders(self) -> None:
        source = np.full((4, 8, 3), 235, dtype=np.uint8)
        graph = DocumentGraph((8, 4))
        for index in range(6):
            crop = AlphaCrop(index, 1, np.full((1, 1), 255, np.uint8))
            graph.add_node(
                ElementNode(
                    f"ELEMENT_{index}",
                    f"Detail {index}",
                    "unknown",
                    crop,
                    index,
                    full_support=crop,
                    evidence=[{"source": "layerd"}],
                    metadata={"source_iteration": 1},
                )
            )
        explicit = {
            "id": "USER_GROUP_KEEP",
            "name": "Keep exactly",
            "member_ids": ["ELEMENT_2", "ELEMENT_3"],
            "custom": {"user": True},
        }
        graph.metadata["review"] = {"groups": [explicit]}
        assign_organizational_groups(graph)

        previous = {
            "groups": [
                explicit,
                {
                    "id": "AUTO_STALE",
                    "name": "Stale auto folder",
                    "member_ids": ["ELEMENT_0", "ELEMENT_5"],
                    "source": AUTO_ORGANIZATION_SOURCE,
                },
            ],
            "review": {"revision": 7, "finished": False},
        }
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = prepare_review_bundle(
                graph,
                source,
                Path(raw),
                source_sha256="abc",
                icc_profile=None,
            )
            _preserve_review_metadata(checkpoint, previous)
            final = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertEqual(final["groups"][0], explicit)
        self.assertFalse(any(group["id"] == "AUTO_STALE" for group in final["groups"]))
        fresh_auto = [
            group
            for group in final["groups"]
            if group.get("source") == AUTO_ORGANIZATION_SOURCE
        ]
        self.assertEqual(len(fresh_auto), 2)
        self.assertTrue(
            all(
                set(group["member_ids"]).isdisjoint(explicit["member_ids"])
                for group in fresh_auto
            )
        )

    def test_preserve_review_never_resurrects_dissolved_unsafe_group(self) -> None:
        source = np.full((4, 8, 3), 235, dtype=np.uint8)
        graph = DocumentGraph((8, 4))
        parent_crop = AlphaCrop(0, 0, np.full((4, 4), 255, np.uint8))
        child_crop = AlphaCrop(1, 1, np.full((1, 1), 255, np.uint8))
        root_crop = AlphaCrop(6, 1, np.full((1, 1), 255, np.uint8))
        graph.add_node(
            ElementNode(
                "PARENT",
                "Parent",
                "panel",
                parent_crop,
                1,
                full_support=parent_crop,
            )
        )
        graph.add_node(
            ElementNode(
                "CHILD",
                "Child",
                "unknown",
                child_crop,
                2,
                parent_id="PARENT",
                full_support=child_crop,
            )
        )
        graph.add_node(
            ElementNode(
                "ROOT_DETAIL",
                "Root detail",
                "unknown",
                root_crop,
                3,
                full_support=root_crop,
            )
        )
        stale_group = {
            "id": "USER_GROUP_STALE",
            "name": "Cross-parent stale group",
            "member_ids": ["CHILD", "ROOT_DETAIL"],
        }
        graph.metadata["review"] = {"groups": [stale_group]}
        organization = assign_organizational_groups(graph)
        self.assertEqual(organization["dissolved_explicit_group_count"], 1)

        previous = {
            "groups": [stale_group],
            "review": {"revision": 2, "finished": False},
        }
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = prepare_review_bundle(
                graph,
                source,
                Path(raw),
                source_sha256="abc",
                icc_profile=None,
            )
            _preserve_review_metadata(checkpoint, previous)
            final = json.loads(checkpoint.read_text(encoding="utf-8"))
            ReviewSession.open(checkpoint)

        self.assertEqual(final["groups"], [])
        self.assertEqual(final["organization"]["dissolved_explicit_group_count"], 1)

    def test_resume_rejects_another_source_and_changed_inventory(self) -> None:
        source = np.full((12, 16, 3), 235, dtype=np.uint8)
        graph = self._graph(source)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint = prepare_review_bundle(
                graph, source, root, source_sha256="abc", icc_profile=None
            )
            with self.assertRaisesRegex(RuntimeError, "ảnh khác"):
                _apply_compatible_previous_review(graph, checkpoint, source - 1)
            document = json.loads(checkpoint.read_text(encoding="utf-8"))
            document["proposals"] = []
            checkpoint.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "inventory khác"):
                _apply_compatible_previous_review(graph, checkpoint, source)

    def test_resume_rejects_legacy_checkpoint_before_decisions_are_applied(self) -> None:
        source = np.full((12, 16, 3), 235, dtype=np.uint8)
        graph = self._graph(source)
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = prepare_review_bundle(
                graph,
                source,
                Path(raw),
                source_sha256="abc",
                icc_profile=None,
            )
            document = json.loads(checkpoint.read_text(encoding="utf-8"))
            document.pop("review_resume_signature")
            checkpoint.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "older or incompatible"):
                _apply_compatible_previous_review(graph, checkpoint, source)


if __name__ == "__main__":
    unittest.main()
