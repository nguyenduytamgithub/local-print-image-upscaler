from __future__ import annotations

import json
import unittest

import cv2
import numpy as np

from v5pro.residual_backend import reconcile_residual_elements
from v5pro.schema import AlphaCrop, DocumentGraph, ElementNode


def gradient_background(height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float32)
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:, :, 0] = 246.0 + xx / max(1, width - 1) * 4.0
    image[:, :, 1] = 231.0 + yy / max(1, height - 1) * 7.0
    image[:, :, 2] = 174.0 + xx / max(1, width - 1) * 5.0
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def add_owned_child(
    graph: DocumentGraph,
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> ElementNode:
    x0, y0, x1, y1 = bbox
    alpha = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)
    crop = AlphaCrop(x0, y0, alpha)
    rgba = np.dstack([image[y0:y1, x0:x1].copy(), alpha])
    node = ElementNode(
        "EXISTING_CHILD",
        "Existing child",
        "product",
        crop,
        900_000,
        full_support=crop,
        confidence=1.0,
        review_status="auto_confirmed",
        move_safe=True,
        rgba=rgba,
    )
    graph.add_node(node)
    return node


class ResidualBackendV2Tests(unittest.TestCase):
    def test_smooth_gradient_is_surface_not_one_giant_layer(self) -> None:
        image = gradient_background(120, 180)
        graph = DocumentGraph((180, 120))

        result = reconcile_residual_elements(image, graph)

        self.assertEqual(result.nodes, [])
        self.assertEqual(result.report["salient_pixel_accounting"]["unaccounted"], 0)
        self.assertLess(result.report["largest_visible_fraction"], 0.01)
        self.assertGreater(result.report["background"]["coverage"], 0.90)

    def test_flat_panel_synthesizes_below_owned_child_but_visible_alpha_excludes_it(self) -> None:
        image = gradient_background(140, 190)
        cv2.rectangle(image, (28, 28), (161, 112), (248, 248, 244), -1, cv2.LINE_8)
        cv2.rectangle(image, (70, 53), (117, 87), (30, 120, 205), -1, cv2.LINE_8)
        graph = DocumentGraph((190, 140))
        child = add_owned_child(graph, image, (70, 53, 118, 88))

        result = reconcile_residual_elements(image, graph)

        panels = [node for node in result.nodes if node.kind == "panel"]
        self.assertTrue(panels)
        panel = max(panels, key=lambda node: node.full_support.nonzero_pixels)
        self.assertEqual(panel.review_status, "unresolved")
        self.assertFalse(panel.move_safe)
        cleanliness = panel.metadata["geometry_cleanliness"]
        self.assertEqual(cleanliness["status"], "unsafe")
        self.assertEqual(cleanliness["reference_type"], "none")
        self.assertIn("carve_ledger", cleanliness["reason"])
        self.assertTrue(panel.occluded)
        self.assertTrue(panel.synthesized_hidden_pixels)
        px = 90 - panel.bbox[0]
        py = 70 - panel.bbox[1]
        self.assertEqual(int(panel.visible_alpha.alpha[py, px]), 0)
        self.assertEqual(int(panel.full_support.alpha[py, px]), 255)
        self.assertEqual(result.report["salient_pixel_accounting"]["unaccounted"], 0)
        self.assertEqual(child.visible_alpha.nonzero_pixels, 48 * 35)
        self.assertGreaterEqual(result.report["residual_geometry_downgraded_count"], 1)
        self.assertIn(panel.element_id, result.report["residual_geometry_downgraded_node_ids"])

    def test_unowned_ring_keeps_real_center_hole(self) -> None:
        image = gradient_background(140, 180)
        cv2.circle(image, (90, 70), 30, (25, 100, 190), 9, cv2.LINE_8)
        graph = DocumentGraph((180, 140))

        result = reconcile_residual_elements(image, graph)

        containing = [
            node
            for node in result.nodes
            if node.bbox[0] <= 90 < node.bbox[2] and node.bbox[1] <= 70 < node.bbox[3]
        ]
        self.assertTrue(containing)
        ring = max(containing, key=lambda node: node.visible_alpha.nonzero_pixels)
        cx, cy = 90 - ring.bbox[0], 70 - ring.bbox[1]
        self.assertEqual(int(ring.visible_alpha.alpha[cy, cx]), 0)
        self.assertEqual(int((ring.full_support or ring.visible_alpha).alpha[cy, cx]), 0)
        self.assertEqual(result.report["salient_pixel_accounting"]["unaccounted"], 0)

    def test_multiple_residual_elements_are_exclusive_and_fully_ledgers_accounted(self) -> None:
        image = gradient_background(160, 220)
        cv2.rectangle(image, (18, 25), (84, 86), (250, 250, 247), -1, cv2.LINE_8)
        cv2.rectangle(image, (130, 28), (201, 91), (235, 244, 252), -1, cv2.LINE_8)
        cv2.line(image, (25, 130), (195, 130), (20, 130, 65), 5, cv2.LINE_8)
        graph = DocumentGraph((220, 160))

        result = reconcile_residual_elements(image, graph)

        self.assertGreaterEqual(len(result.nodes), 3)
        claimed = np.zeros((160, 220), dtype=bool)
        for node in sorted(result.nodes, key=lambda item: item.z_index, reverse=True):
            canvas = node.visible_alpha.to_canvas((220, 160)) > 0
            self.assertFalse(np.any(claimed & canvas))
            claimed |= canvas
        proposal_ids = [proposal.proposal_id for proposal in result.proposals]
        self.assertEqual(len(proposal_ids), len(set(proposal_ids)))
        self.assertTrue(
            all(
                (proposal.status == "assigned" and proposal.owner_ids)
                or (proposal.status == "rejected" and proposal.reason)
                for proposal in result.proposals
            )
        )
        accounting = result.report["salient_pixel_accounting"]
        self.assertEqual(accounting["total"], accounting["assigned"] + accounting["rejected"])
        self.assertEqual(accounting["unaccounted"], 0)
        self.assertLess(result.report["largest_visible_fraction"], 0.20)
        json.dumps(result.report)

    def test_low_amplitude_background_noise_does_not_become_foreground(self) -> None:
        image = gradient_background(130, 170).astype(np.int16)
        rng = np.random.default_rng(731)
        image += rng.integers(-2, 3, size=image.shape, dtype=np.int16)
        image = np.clip(image, 0, 255).astype(np.uint8)
        graph = DocumentGraph((170, 130))

        result = reconcile_residual_elements(image, graph)

        self.assertEqual(len(result.nodes), 0)
        self.assertEqual(result.report["salient_pixel_accounting"]["unaccounted"], 0)


if __name__ == "__main__":
    unittest.main()
