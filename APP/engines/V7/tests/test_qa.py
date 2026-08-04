from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest
import unicodedata

import numpy as np
from PIL import Image


V7_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7_DIR))

from v7lib.qa import (  # noqa: E402
    QAThresholds,
    canonical_sha256,
    evaluate_replacement,
    make_determinism_metadata,
    normalize_nfc,
    pixel_sha256,
    save_qa_artifacts,
)


class ReplacementQATests(unittest.TestCase):
    def setUp(self) -> None:
        self.height = 72
        self.width = 112
        x = np.linspace(0, 8, self.width, dtype=np.float32)
        self.background = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.background[..., 0] = np.rint(224 + x).astype(np.uint8)
        self.background[..., 1] = np.rint(206 + x * 0.5).astype(np.uint8)
        self.background[..., 2] = 58

        self.roi = np.zeros((self.height, self.width), dtype=bool)
        self.roi[12:60, 10:102] = True
        self.old = np.zeros_like(self.roi)
        self.old[28:43, 20:38] = True
        self.new = np.zeros_like(self.roi)
        self.new[25:46, 55:84] = True
        self.seams = np.zeros_like(self.roi)

        self.original = self.background.copy()
        self.original[self.old] = (24, 35, 190)
        self.clean_result = self.background.copy()
        self.clean_result[self.new] = (18, 110, 38)
        self.config = {"font": "Noto Sans", "size": 24, "stroke": 0}

    def metadata(self, result: np.ndarray) -> dict[str, object]:
        return {
            "determinism": make_determinism_metadata(
                self.original,
                result,
                seed=17,
                config=self.config,
            )
        }

    def evaluate(self, result: np.ndarray, **overrides: object):
        arguments: dict[str, object] = {
            "roi_mask": self.roi,
            "old_text_mask": self.old,
            "new_text_alpha": self.new,
            "seam_mask": self.seams,
            "target_text": "GẠO CÔNG NGHỆ CHẤT VIỆT",
            "rendered_text": "GẠO CÔNG NGHỆ CHẤT VIỆT",
            "metadata": self.metadata(result),
            "expected_background": self.background,
        }
        arguments.update(overrides)
        return evaluate_replacement(self.original, result, **arguments)

    def test_clean_edit_passes_and_report_is_json_serializable(self) -> None:
        decomposed = unicodedata.normalize("NFD", "GẠO CÔNG NGHỆ CHẤT VIỆT")
        qa = self.evaluate(self.clean_result, rendered_text=decomposed)

        self.assertTrue(qa.passed, qa.report["hard_failures"])
        self.assertEqual(qa.report["status"], "PASS")
        self.assertEqual(qa.overlay.shape, self.clean_result.shape)
        self.assertEqual(qa.overlay.dtype, np.uint8)
        json.dumps(qa.to_dict(), ensure_ascii=False, allow_nan=False)

    def test_one_changed_pixel_outside_roi_is_a_hard_failure(self) -> None:
        damaged = self.clean_result.copy()
        damaged[2, 3] = (0, 0, 0)
        qa = self.evaluate(damaged)

        self.assertFalse(qa.passed)
        self.assertIn("outside_roi_changed", qa.report["hard_failures"])
        metrics = qa.report["checks"]["outside_roi"]["metrics"]
        self.assertEqual(metrics["changed_pixel_count"], 1)
        # Red is blended into the offending pixel on the overlay.
        self.assertGreater(int(qa.overlay[2, 3, 0]), int(qa.overlay[2, 3, 1]))

    def test_retained_old_glyph_is_detected_with_ground_truth(self) -> None:
        ghosted = self.clean_result.copy()
        ghosted[self.old] = self.original[self.old]
        qa = self.evaluate(ghosted)

        self.assertFalse(qa.passed)
        self.assertIn("old_text_ghost_detected", qa.report["hard_failures"])
        metrics = qa.report["checks"]["ghost_residual"]["metrics"]
        self.assertGreater(metrics["gt_residual_coverage"], 0.9)
        ghost_pixel = qa.overlay[34, 28]
        self.assertGreater(int(ghost_pixel[0]), int(ghost_pixel[1]))
        self.assertGreater(int(ghost_pixel[2]), int(ghost_pixel[1]))

    def test_old_edge_detector_works_without_ground_truth(self) -> None:
        ghosted = self.clean_result.copy()
        ghosted[self.old] = self.original[self.old]
        qa = self.evaluate(ghosted, expected_background=None)

        self.assertFalse(qa.report["checks"]["ghost_residual"]["passed"])
        metrics = qa.report["checks"]["ghost_residual"]["metrics"]
        self.assertEqual(metrics["mode"], "no_ground_truth")
        self.assertGreater(metrics["edge_residual_ratio"], 0.5)

    def test_old_edge_detector_accepts_clean_background_without_ground_truth(self) -> None:
        qa = self.evaluate(self.clean_result, expected_background=None)

        self.assertTrue(
            qa.report["checks"]["ghost_residual"]["passed"],
            qa.report["checks"]["ghost_residual"],
        )
        self.assertEqual(
            qa.report["checks"]["ghost_residual"]["metrics"]["mode"],
            "no_ground_truth",
        )

    def test_known_vertical_seam_is_detected(self) -> None:
        result = self.background.copy()
        result[self.new] = (18, 110, 38)
        # Keep the brightness step local to the ROI so outside-ROI QA remains
        # independent from the seam gate.
        right_half = self.roi.copy()
        right_half[:, :48] = False
        result[right_half] = np.clip(
            result[right_half].astype(np.int16) + 35, 0, 255
        ).astype(np.uint8)
        seams = np.zeros_like(self.roi)
        seams[13:59, 48] = True

        qa = self.evaluate(
            result,
            seam_mask=seams,
            expected_background=None,
            metadata=self.metadata(result),
        )

        self.assertFalse(qa.passed)
        self.assertIn("blend_or_tile_seam_detected", qa.report["hard_failures"])
        metrics = qa.report["checks"]["seam"]["metrics"]
        self.assertGreater(metrics["p95_excess"], 20)

    def test_declared_tile_boundary_without_artifact_passes(self) -> None:
        seams = np.zeros_like(self.roi)
        seams[13:59, 48] = True
        qa = self.evaluate(self.clean_result, seam_mask=seams)

        self.assertTrue(qa.report["checks"]["seam"]["passed"])
        self.assertEqual(
            qa.report["checks"]["seam"]["metrics"]["p95_excess"],
            0.0,
        )

    def test_new_alpha_touching_roi_boundary_fails_clipping(self) -> None:
        clipped_alpha = np.zeros_like(self.roi)
        clipped_alpha[12:30, 25:50] = True
        result = self.background.copy()
        result[clipped_alpha] = (18, 110, 38)
        qa = self.evaluate(
            result,
            new_text_alpha=clipped_alpha,
            metadata=self.metadata(result),
        )

        self.assertFalse(qa.passed)
        self.assertIn("new_text_clipped", qa.report["hard_failures"])
        self.assertGreater(
            qa.report["checks"]["clipping"]["metrics"]["clipped_pixel_count"],
            0,
        )

    def test_vietnamese_diacritic_mismatch_is_not_weakened(self) -> None:
        qa = self.evaluate(
            self.clean_result,
            target_text="GẠO CÔNG NGHỆ CHẤT VIỆT",
            rendered_text="GAO CÔNG NGHỆ CHẤT VIỆT",
        )

        self.assertFalse(qa.passed)
        self.assertIn("target_text_mismatch", qa.report["hard_failures"])
        self.assertEqual(normalize_nfc("Đ"), "Đ")
        self.assertNotEqual(normalize_nfc("Đ"), normalize_nfc("D"))

    def test_tampered_determinism_hash_fails(self) -> None:
        metadata = self.metadata(self.clean_result)
        metadata["determinism"]["output_pixel_sha256"] = "0" * 64  # type: ignore[index]
        qa = self.evaluate(self.clean_result, metadata=metadata)

        self.assertFalse(qa.passed)
        self.assertIn("determinism_metadata_invalid", qa.report["hard_failures"])
        errors = qa.report["checks"]["determinism_metadata"]["details"]["errors"]
        self.assertIn("mismatch:output_pixel_sha256", errors)

    def test_bbox_roi_and_pil_inputs_are_supported(self) -> None:
        qa = evaluate_replacement(
            Image.fromarray(self.original, "RGB"),
            Image.fromarray(self.clean_result, "RGB"),
            roi_mask=(10, 12, 102, 60),
            old_text_mask=Image.fromarray(self.old.astype(np.uint8) * 255, "L"),
            new_text_alpha=self.new,
            seam_mask=self.seams,
            target_text="ĐÚNG",
            rendered_text=unicodedata.normalize("NFD", "ĐÚNG"),
            metadata=self.metadata(self.clean_result),
            expected_background=self.background,
        )
        self.assertTrue(qa.passed, qa.report["hard_failures"])

    def test_artifact_writer_round_trips_utf8_json_and_png(self) -> None:
        qa = self.evaluate(self.clean_result)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "qa.json"
            overlay_path = root / "qa-overlay.png"
            save_qa_artifacts(qa, json_path=report_path, overlay_path=overlay_path)

            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            overlay = np.asarray(Image.open(overlay_path).convert("RGB"))
            self.assertEqual(loaded["status"], "PASS")
            self.assertEqual(overlay.shape, self.clean_result.shape)


class HashContractTests(unittest.TestCase):
    def test_canonical_config_hash_is_order_independent(self) -> None:
        first = {"font": "Noto Sans", "effects": {"stroke": 2, "shadow": False}}
        second = {"effects": {"shadow": False, "stroke": 2}, "font": "Noto Sans"}
        self.assertEqual(canonical_sha256(first), canonical_sha256(second))

    def test_pixel_hash_changes_with_shape_and_pixels(self) -> None:
        a = np.zeros((4, 6, 3), dtype=np.uint8)
        b = a.copy()
        b[0, 0] = 1
        c = np.zeros((6, 4, 3), dtype=np.uint8)
        self.assertNotEqual(pixel_sha256(a), pixel_sha256(b))
        self.assertNotEqual(pixel_sha256(a), pixel_sha256(c))

    def test_thresholds_can_allow_a_nonzero_outside_tolerance(self) -> None:
        original = np.zeros((24, 30, 3), dtype=np.uint8)
        result = original.copy()
        result[0, 0] = 1
        roi = np.zeros((24, 30), dtype=bool)
        roi[4:20, 4:26] = True
        old = np.zeros_like(roi)
        old[8:12, 7:11] = True
        new = np.zeros_like(roi)
        new[8:13, 15:21] = True
        determinism = make_determinism_metadata(original, result, seed=1, config={})
        qa = evaluate_replacement(
            original,
            result,
            roi_mask=roi,
            old_text_mask=old,
            new_text_alpha=new,
            seam_mask=np.zeros_like(roi),
            target_text="A",
            rendered_text="A",
            metadata=determinism,
            thresholds=QAThresholds(outside_channel_tolerance=1),
        )
        self.assertTrue(qa.report["checks"]["outside_roi"]["passed"])


if __name__ == "__main__":
    unittest.main()
