from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from v5pro.exporter import render_document
from v5pro.hierarchy import (
    AUTO_ORGANIZATION_SOURCE,
    assign_organizational_groups,
    assign_spatial_hierarchy,
)
from v5pro.ownership import ownership_priority
from v5pro.review_adapter import apply_review_checkpoint, prepare_review_bundle
from v5pro.review_server import ReviewSession
from v5pro.schema import AlphaCrop, DocumentGraph, ElementNode, ProposalRecord


class HierarchyReviewAdapterV2Tests(unittest.TestCase):
    def _graph(self) -> tuple[DocumentGraph, np.ndarray]:
        source = np.full((20, 30, 3), 245, dtype=np.uint8)
        graph = DocumentGraph((30, 20))
        frame_alpha = np.zeros((16, 26), dtype=np.uint8)
        frame_alpha[[0, -1], :] = 255
        frame_alpha[:, [0, -1]] = 255
        frame = AlphaCrop(2, 2, frame_alpha)
        text = AlphaCrop(8, 7, np.full((3, 8), 255, dtype=np.uint8))
        for identifier, name, kind, crop, z in (
            ("FRAME", "Frame", "frame", frame, 1),
            ("TEXT", "Text", "text", text, 2),
        ):
            x0, y0, x1, y1 = crop.bbox
            graph.add_node(
                ElementNode(
                    identifier,
                    name,
                    kind,  # type: ignore[arg-type]
                    crop,
                    z,
                    full_support=crop,
                    confidence=1.0,
                    review_status="auto_confirmed",
                    rgba=np.dstack([source[y0:y1, x0:x1], crop.alpha]),
                )
            )
        graph.add_proposal(
            ProposalRecord("P", "test", "text", text.bbox, 1.0, "assigned", ["TEXT"])
        )
        return graph, source

    def test_hierarchy_does_not_union_masks(self) -> None:
        graph, _source = self._graph()
        before = graph.node_map()["FRAME"].visible_alpha.alpha.copy()
        report = assign_spatial_hierarchy(graph)
        self.assertEqual(graph.node_map()["TEXT"].parent_id, "FRAME")
        self.assertTrue(np.array_equal(before, graph.node_map()["FRAME"].visible_alpha.alpha))
        self.assertEqual(report["assignment_count"], 1)

    def test_auto_folders_are_contiguous_and_do_not_change_render(self) -> None:
        width, height = 48, 12
        source = np.full((height, width, 3), 245, dtype=np.uint8)
        graph = DocumentGraph((width, height))
        records = [
            ("RAW_1", "unknown", 1, {"source_iteration": 1}, [{"source": "layerd"}]),
            ("RAW_2", "micro_detail", 2, {"source_iteration": 1}, [{"source": "layerd"}]),
            ("RES_1", "unknown", 3, {}, [{"source": "poster_surface_residual"}]),
            ("RES_2", "micro_detail", 4, {}, [{"source": "poster_surface_residual"}]),
            (
                "GEO_1",
                "line",
                5,
                {"geometry_backend": "poster_geometry_v2"},
                [{"source": "poster_geometry"}],
            ),
            (
                "GEO_2",
                "frame",
                6,
                {"geometry_backend": "poster_geometry_v2"},
                [{"source": "poster_geometry"}],
            ),
            ("SEMANTIC_1", "icon", 7, {}, [{"source": "grounding_dino"}]),
            ("SEMANTIC_2", "product", 8, {}, [{"source": "sam2"}]),
            ("TEXT_1", "text", 9, {}, [{"source": "ocr"}]),
            ("TEXT_2", "price", 10, {}, [{"source": "ocr"}]),
        ]
        for index, (identifier, kind, z_index, metadata, evidence) in enumerate(records):
            alpha = AlphaCrop(index * 4, 3, np.full((3, 3), 255, np.uint8))
            source[3:6, index * 4 : index * 4 + 3] = (20 + index, 60, 100)
            graph.add_node(
                ElementNode(
                    identifier,
                    identifier,
                    kind,  # type: ignore[arg-type]
                    alpha,
                    z_index,
                    full_support=alpha,
                    review_status="unresolved",
                    evidence=evidence,
                    metadata=metadata,
                )
            )

        assign_spatial_hierarchy(graph)
        groups = graph.metadata["review"]["groups"]
        roots = sorted(graph.nodes, key=ownership_priority)
        positions = {node.element_id: index for index, node in enumerate(roots)}
        for group in groups:
            self.assertEqual(group["source"], AUTO_ORGANIZATION_SOURCE)
            members = list(group["member_ids"])
            self.assertEqual(
                [positions[member] for member in members],
                list(range(positions[members[0]], positions[members[-1]] + 1)),
            )
            self.assertEqual({graph.node_map()[member].parent_id for member in members}, {None})

        clean = np.full_like(source, 245)
        grouped = render_document(graph, source, clean, scale=1.0)
        saved_groups = graph.metadata["review"].pop("groups")
        ungrouped = render_document(graph, source, clean, scale=1.0)
        graph.metadata["review"]["groups"] = saved_groups
        self.assertTrue(
            np.array_equal(np.asarray(grouped.composite), np.asarray(ungrouped.composite))
        )
        self.assertEqual(
            [layer.spec.layer_id for layer in grouped.rendered],
            [layer.spec.layer_id for layer in ungrouped.rendered],
        )
        self.assertGreater(len(grouped.user_groups), 0)
        self.assertEqual(ungrouped.user_groups, [])

        with tempfile.TemporaryDirectory() as raw:
            checkpoint = prepare_review_bundle(
                graph,
                source,
                Path(raw),
                source_sha256="abc",
                icc_profile=None,
            )
            document = json.loads(checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(document["groups"], saved_groups)

    def test_explicit_user_group_wins_over_regenerated_auto_folders(self) -> None:
        graph = DocumentGraph((8, 2))
        for index in range(6):
            alpha = AlphaCrop(index, 0, np.full((1, 1), 255, np.uint8))
            graph.add_node(
                ElementNode(
                    f"ELEMENT_{index}",
                    f"Detail {index}",
                    "unknown",
                    alpha,
                    index,
                    full_support=alpha,
                    evidence=[{"source": "layerd"}],
                    metadata={"source_iteration": 1},
                )
            )
        explicit = {
            "id": "USER_GROUP_1",
            "name": "My exact folder",
            "member_ids": ["ELEMENT_2", "ELEMENT_3"],
            "custom_field": {"keep": True},
        }
        graph.metadata["review"] = {"groups": [explicit]}

        report = assign_organizational_groups(graph)

        groups = graph.metadata["review"]["groups"]
        self.assertEqual(groups[0], explicit)
        auto_members = {
            member
            for group in groups[1:]
            for member in group["member_ids"]
        }
        self.assertTrue(auto_members.isdisjoint(explicit["member_ids"]))
        self.assertEqual(report["explicit_group_count"], 1)
        self.assertEqual(report["auto_group_count"], 2)

    def test_unsafe_old_explicit_group_is_dissolved_without_auto_claiming_members(self) -> None:
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
        graph.metadata["review"] = {
            "groups": [
                {
                    "id": "USER_GROUP_OLD",
                    "name": "Old unsafe group",
                    "member_ids": ["CHILD", "ROOT_DETAIL"],
                }
            ]
        }

        report = assign_organizational_groups(graph)

        self.assertEqual(graph.metadata["review"]["groups"], [])
        self.assertEqual(report["dissolved_explicit_group_count"], 1)
        self.assertEqual(
            report["dissolved_explicit_groups"][0]["reason"],
            "members_no_longer_share_one_parent",
        )

    def test_g_like_root_is_collapsed_without_merging_one_thousand_leaves(self) -> None:
        graph = DocumentGraph((1, 1))
        crop = AlphaCrop(0, 0, np.full((1, 1), 255, np.uint8))

        def add_many(prefix: str, count: int, kind: str, metadata, evidence) -> None:
            start = len(graph.nodes)
            for offset in range(count):
                graph.add_node(
                    ElementNode(
                        f"{prefix}_{offset:04d}",
                        f"{prefix} {offset}",
                        kind,  # type: ignore[arg-type]
                        crop,
                        start + offset,
                        full_support=crop,
                        metadata=dict(metadata),
                        evidence=list(evidence),
                    )
                )

        add_many("ELEMENT", 544, "unknown", {"source_iteration": 1}, [{"source": "layerd"}])
        add_many("RESIDUAL", 299, "unknown", {}, [{"source": "poster_surface_residual"}])
        add_many("GEO", 28, "line", {"geometry_backend": "poster_geometry_v2"}, [{"source": "poster_geometry"}])
        add_many("SEMANTIC", 20, "icon", {}, [{"source": "grounding_dino"}])
        add_many("TEXT", 94, "text", {}, [{"source": "ocr"}])
        add_many("SINGLETON", 19, "panel", {}, [{"source": "layout"}])

        report = assign_organizational_groups(graph)

        self.assertEqual(len(graph.nodes), 1004)
        self.assertEqual(report["root_entries_before"], 1004)
        self.assertLessEqual(report["root_entries_after"], 8)
        self.assertEqual(
            sum(len(group["member_ids"]) for group in graph.metadata["review"]["groups"]),
            1004,
        )

    def test_paired_panel_is_below_its_frame(self) -> None:
        graph = DocumentGraph((30, 20))
        panel_crop = AlphaCrop(3, 3, np.full((14, 24), 255, dtype=np.uint8))
        frame_alpha = np.zeros((16, 26), dtype=np.uint8)
        frame_alpha[[0, -1], :] = 255
        frame_alpha[:, [0, -1]] = 255
        frame_crop = AlphaCrop(2, 2, frame_alpha)
        panel = ElementNode(
            "PANEL",
            "Panel",
            "panel",
            panel_crop,
            10,
            full_support=panel_crop,
            review_status="auto_confirmed",
        )
        frame = ElementNode(
            "FRAME",
            "Frame",
            "frame",
            frame_crop,
            11,
            full_support=frame_crop,
            review_status="auto_confirmed",
            metadata={"paired_panel_id": "PANEL"},
        )
        graph.add_node(panel)
        graph.add_node(frame)
        assign_spatial_hierarchy(graph)
        self.assertIsNone(panel.parent_id)
        self.assertEqual(frame.parent_id, "PANEL")

    def test_cross_group_visible_owner_moves_above_last_clean_geometry_writer(self) -> None:
        graph = DocumentGraph((40, 24))
        early_panel_crop = AlphaCrop(
            0, 0, np.full((14, 18), 255, dtype=np.uint8)
        )
        early_panel = ElementNode(
            "EARLY_PANEL",
            "Early panel",
            "panel",
            early_panel_crop,
            10,
            full_support=early_panel_crop,
            review_status="unresolved",
        )
        text_crop = AlphaCrop(10, 6, np.full((4, 12), 255, dtype=np.uint8))
        text = ElementNode(
            "TEXT",
            "Cross-boundary text",
            "text",
            text_crop,
            20,
            full_support=text_crop,
            review_status="auto_confirmed",
        )
        early_clean_alpha = np.full((3, 12), 255, dtype=np.uint8)
        early_hidden_alpha = np.zeros_like(early_clean_alpha)
        early_geometry = ElementNode(
            "EARLY_GEOMETRY",
            "Early nested geometry",
            "line",
            AlphaCrop(3, 7, early_hidden_alpha.copy()),
            30,
            parent_id="EARLY_PANEL",
            full_support=AlphaCrop(3, 7, early_clean_alpha.copy()),
            review_status="auto_confirmed",
            metadata={
                "geometry_backend": "poster_geometry_v2",
                "geometry_cleanliness": {
                    "policy": "reference_surface_delta_e_carve_v1",
                    "status": "pass",
                    "reference_type": "constant_colour_rgb",
                }
            },
        )
        late_clean_alpha = np.full((3, 20), 255, dtype=np.uint8)
        late_hidden_alpha = np.zeros_like(late_clean_alpha)
        late_geometry = ElementNode(
            "LATE_GEOMETRY",
            "Later clean geometry",
            "frame",
            AlphaCrop(14, 7, late_hidden_alpha.copy()),
            40,
            full_support=AlphaCrop(14, 7, late_clean_alpha.copy()),
            review_status="auto_confirmed",
            metadata={
                "geometry_backend": "poster_geometry_v2",
                "geometry_cleanliness": {
                    "policy": "reference_surface_delta_e_carve_v1",
                    "status": "pass",
                    "reference_type": "constant_colour_rgb",
                }
            },
        )
        graph.add_node(early_panel)
        graph.add_node(text)
        graph.add_node(early_geometry)
        graph.add_node(late_geometry)

        report = assign_spatial_hierarchy(graph)

        self.assertEqual(graph.node_map()["TEXT"].parent_id, "LATE_GEOMETRY")
        repairs = report["stack_dependency_repairs"]
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0]["child"], "TEXT")
        self.assertEqual(repairs[0]["parent"], "LATE_GEOMETRY")
        self.assertGreater(repairs[0]["overlap_pixels"], 0)

    def test_technical_source_remainder_is_never_reparented_under_geometry(self) -> None:
        graph = DocumentGraph((20, 12))
        full = np.full((3, 12), 255, dtype=np.uint8)
        hidden = np.zeros_like(full)
        geometry = ElementNode(
            "GEOMETRY",
            "Clean geometry",
            "line",
            AlphaCrop(3, 4, hidden),
            10,
            full_support=AlphaCrop(3, 4, full.copy()),
            review_status="auto_confirmed",
            metadata={
                "geometry_backend": "poster_geometry_v2",
                "geometry_cleanliness": {
                    "policy": "reference_surface_delta_e_carve_v1",
                    "status": "pass",
                    "reference_type": "constant_colour_rgb",
                }
            },
        )
        remainder = ElementNode(
            "BASE_SOURCE_RESIDUAL_0001",
            "Source remainder",
            "unknown",
            AlphaCrop(0, 0, np.full((12, 20), 255, dtype=np.uint8)),
            100,
            full_support=AlphaCrop(
                0, 0, np.full((12, 20), 255, dtype=np.uint8)
            ),
            review_status="unresolved",
            metadata={"role": "exact_source_remainder_above_clean_base"},
        )
        graph.add_node(geometry)
        graph.add_node(remainder)

        assign_spatial_hierarchy(graph)

        self.assertIsNone(graph.node_map()[remainder.element_id].parent_id)

    def test_review_checkpoint_roundtrip_keeps_real_mask(self) -> None:
        graph, source = self._graph()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            digest = hashlib.sha256(source.tobytes()).hexdigest()
            checkpoint = prepare_review_bundle(
                graph, source, root, source_sha256=digest, icc_profile=None
            )
            document = json.loads(checkpoint.read_text(encoding="utf-8"))
            review_source = checkpoint.parent / document["source"]
            self.assertEqual(
                document["source_sha256"],
                hashlib.sha256(review_source.read_bytes()).hexdigest(),
            )
            self.assertEqual(document["original_source_sha256"], digest)
            ReviewSession.open(checkpoint)
            document["nodes"][1]["review_status"] = "user_confirmed"
            document["review"]["finished"] = True
            checkpoint.write_text(json.dumps(document), encoding="utf-8")
            restored = apply_review_checkpoint(graph, checkpoint, source)
            self.assertTrue(
                np.array_equal(
                    restored.node_map()["FRAME"].visible_alpha.alpha,
                    graph.node_map()["FRAME"].visible_alpha.alpha,
                )
            )
            self.assertEqual(restored.node_map()["TEXT"].review_status, "user_confirmed")

    def test_confirmation_preserves_synthesized_support_below_child(self) -> None:
        source = np.full((18, 30, 3), 245, dtype=np.uint8)
        visible = np.full((4, 20), 255, dtype=np.uint8)
        visible[:, 7:13] = 0
        full = np.full((4, 20), 255, dtype=np.uint8)
        visible_crop = AlphaCrop(5, 7, visible)
        full_crop = AlphaCrop(5, 7, full)
        graph = DocumentGraph((30, 18))
        graph.add_node(
            ElementNode(
                "LINE_1",
                "Divider",
                "line",
                visible_crop,
                1,
                full_support=full_crop,
                semantic_envelope=full_crop,
                removal_footprint=full_crop,
                confidence=0.95,
                review_status="unresolved",
                occluded=True,
                synthesized_hidden_pixels=True,
                rgba=np.dstack(
                    [np.full((4, 20, 3), (20, 140, 45), np.uint8), full]
                ),
            )
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint = prepare_review_bundle(
                graph, source, root, source_sha256="abc", icc_profile=None
            )
            document = json.loads(checkpoint.read_text(encoding="utf-8"))
            document["nodes"][0]["review_status"] = "user_confirmed"
            checkpoint.write_text(json.dumps(document), encoding="utf-8")
            restored = apply_review_checkpoint(graph, checkpoint, source)
        node = restored.nodes[0]
        self.assertEqual(node.review_status, "user_confirmed")
        self.assertEqual(node.full_support.nonzero_pixels, 80)
        self.assertEqual(node.visible_alpha.nonzero_pixels, 56)
        self.assertTrue(node.synthesized_hidden_pixels)
        self.assertTrue(np.array_equal(node.rgba[:, :, 3], full))


if __name__ == "__main__":
    unittest.main()
