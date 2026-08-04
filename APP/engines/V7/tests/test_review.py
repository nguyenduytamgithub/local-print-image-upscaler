from __future__ import annotations

import sys
import unittest
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
V7_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "APP"))
sys.path.insert(0, str(V7_DIR))

from design_repair_v7 import merge_review_document  # noqa: E402
from v7lib.review import build_review_document, resolve_review  # noqa: E402
from v7lib.types import OCRObservation, TextRegion  # noqa: E402


def _region(
    *,
    bbox: tuple[int, int, int, int] = (10, 20, 180, 58),
    text: str = "ĐỒ GIA DỤNG",
    status: str = "yellow",
) -> TextRegion:
    x0, y0, x1, y1 = bbox
    return TextRegion(
        region_id="text_0001",
        bbox=bbox,
        polygon=(
            (float(x0), float(y0)),
            (float(x1), float(y0)),
            (float(x1), float(y1)),
            (float(x0), float(y1)),
        ),
        observations=[OCRObservation("paddle", "original", text, 0.91)],
        selected_text=text,
        proposal=None,
        status=status,  # type: ignore[arg-type]
        critical=False,
        reasons=["review-required"],
    )


class ReviewContractTests(unittest.TestCase):
    def test_matching_source_hash_merges_and_approved_text_is_nfc(self) -> None:
        region = _region()
        current = build_review_document(Path("source.png"), [region])
        source_hash = "a" * 64
        decomposed = unicodedata.normalize("NFD", "ĐỒ GIA DỤNG")
        reviewed_row = dict(current["regions"][0])  # preserve the bound fingerprint
        reviewed_row.update(
            {
                "action": "replace",
                "approved_text": decomposed,
            }
        )
        reviewed = {
            "source_sha256": source_hash.upper(),
            "regions": [reviewed_row],
        }

        merge_review_document(current, reviewed, source_sha256=source_hash)
        replacements, kept, unresolved = resolve_review([region], current)

        self.assertEqual(replacements, {region.region_id: "ĐỒ GIA DỤNG"})
        self.assertTrue(unicodedata.is_normalized("NFC", replacements[region.region_id]))
        self.assertEqual(kept, set())
        self.assertEqual(unresolved, [])

    def test_review_from_a_different_source_hash_is_rejected(self) -> None:
        region = _region()
        current = build_review_document(Path("source.png"), [region])
        reviewed = {
            "source_sha256": "b" * 64,
            "regions": [
                {
                    **current["regions"][0],
                    "action": "keep",
                }
            ],
        }

        with self.assertRaisesRegex(RuntimeError, "different source image"):
            merge_review_document(current, reviewed, source_sha256="a" * 64)

    def test_review_without_source_hash_is_rejected(self) -> None:
        region = _region()
        current = build_review_document(Path("source.png"), [region])
        reviewed = {
            "regions": [
                {
                    **current["regions"][0],
                    "action": "keep",
                }
            ]
        }

        with self.assertRaises(RuntimeError):
            merge_review_document(current, reviewed, source_sha256="a" * 64)

    def test_region_fingerprint_blocks_reordered_or_tampered_approval(self) -> None:
        current = build_review_document(Path("source.png"), [_region()])
        old = build_review_document(
            Path("source.png"),
            [_region(bbox=(410, 220, 590, 262), text="KHÁC VÙNG")],
        )
        current_row = current["regions"][0]
        old_row = old["regions"][0]
        self.assertIn("region_fingerprint", current_row)
        self.assertIn("region_fingerprint", old_row)
        self.assertNotEqual(
            current_row["region_fingerprint"],
            old_row["region_fingerprint"],
        )
        reviewed = {
            "source_sha256": "a" * 64,
            "regions": [
                {
                    **old_row,
                    "action": "replace",
                    "approved_text": "NỘI DUNG ĐÃ DUYỆT CHO VÙNG CŨ",
                }
            ],
        }

        with self.assertRaises(RuntimeError):
            merge_review_document(current, reviewed, source_sha256="a" * 64)

    def test_strict_review_never_auto_approves_a_green_region(self) -> None:
        document = build_review_document(
            Path("source.png"),
            [_region(status="green")],
            review_mode="strict",
        )
        row = document["regions"][0]
        self.assertEqual(row["action"], "pending")
        self.assertEqual(row["approved_text"], "")


if __name__ == "__main__":
    unittest.main()
