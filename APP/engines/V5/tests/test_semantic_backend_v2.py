from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from v5pro.semantic_backend import (
    _fallback_soft_alpha,
    _kind_from_label,
    _semantic_policy,
    run_semantic_proposals,
)


class SemanticBackendV2Tests(unittest.TestCase):
    def test_label_taxonomy(self) -> None:
        self.assertEqual(_kind_from_label("product package"), "product")
        self.assertEqual(_kind_from_label("shopping cart icon"), "icon")
        self.assertEqual(_kind_from_label("flower decoration"), "decoration")
        self.assertEqual(_kind_from_label("company logo"), "logo")

    def test_fallback_alpha_is_bounded_and_soft_only_at_edge(self) -> None:
        mask = np.zeros((21, 21), dtype=bool)
        mask[5:16, 5:16] = True
        alpha = _fallback_soft_alpha(mask)
        self.assertEqual(alpha.dtype, np.uint8)
        self.assertEqual(int(alpha[10, 10]), 255)
        self.assertEqual(int(alpha[0, 0]), 0)
        self.assertGreater(int(alpha[5, 10]), 0)

    def test_semantic_policy_defers_giant_false_icon_box(self) -> None:
        accepted, report = _semantic_policy(
            {
                "bbox": (2, 2, 1050, 1488),
                "score": 0.24,
                "label": "customer support headset icon",
            },
            (1054, 1492),
        )
        self.assertFalse(accepted)
        self.assertIn("implausibly large", " ".join(report["reasons"]))

    def test_semantic_policy_accepts_poster_product_box(self) -> None:
        accepted, report = _semantic_policy(
            {
                "bbox": (194, 986, 368, 1121),
                "score": 0.23,
                "label": "pack of tissues",
            },
            (1054, 1492),
        )
        self.assertTrue(accepted)
        self.assertEqual(report["kind"], "product")

    def test_rejected_refinement_keeps_reviewable_mask_but_closes_auto_gate(self) -> None:
        image = np.full((80, 100, 3), 245, dtype=np.uint8)
        mask = np.zeros((80, 100), dtype=bool)
        mask[20:40, 20:35] = True
        fallback = _fallback_soft_alpha(mask)
        detection = {
            "bbox": (20, 20, 35, 40),
            "score": 0.96,
            "label": "product package",
            "alternative_labels": [{"label": "product package", "score": 0.96}],
        }
        with (
            patch(
                "v5pro.semantic_backend._run_dino",
                return_value=([detection], {"model": "synthetic"}),
            ),
            patch(
                "v5pro.semantic_backend._run_sam_boxes",
                return_value=([(mask, 0.99)], {"model": "synthetic"}),
            ),
            patch(
                "v5pro.semantic_backend._refine_with_birefnet",
                return_value=(
                    [
                        (
                            fallback,
                            {
                                "accepted": False,
                                "reason": "BiRefNet disagrees with SAM envelope",
                            },
                        )
                    ],
                    {"model": "synthetic"},
                ),
            ),
        ):
            result = run_semantic_proposals(image, device="cpu", progress=lambda _message: None)

        self.assertEqual(len(result.proposals), 1)
        proposal = result.proposals[0]
        self.assertIsNotNone(proposal.mask_hint)
        self.assertTrue(proposal.record.evidence["auto_extractable"])
        self.assertFalse(proposal.record.evidence["auto_confirmable"])
        self.assertTrue(proposal.record.evidence["requires_manual_review"])
        self.assertIn("refinement", (proposal.record.reason or "").lower())
        self.assertTrue(proposal.record.source.endswith("_review"))
        self.assertEqual(result.report["refinement_review_count"], 1)


if __name__ == "__main__":
    unittest.main()
