from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


V7_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7_DIR))

from v7lib.textnorm import (  # noqa: E402
    ProtectedTextError,
    build_conservative_proposal,
    find_protected_spans,
    normalize_nfc,
    protect_text,
    restore_protected_text,
)
from v7lib.types import OCRObservation, TextRegion  # noqa: E402


class UnicodeNormalizationTests(unittest.TestCase):
    def test_uses_nfc_and_does_not_apply_nfkc(self) -> None:
        self.assertEqual(normalize_nfc("Đo\u0302\u0300"), "Đồ")
        self.assertEqual(normalize_nfc("Ａ①"), "Ａ①")

    def test_non_string_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            normalize_nfc(123)  # type: ignore[arg-type]


class ProtectedSpanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.address = "142X Nguyễn Khoái, Phường Vĩnh Hội, TP. Hồ Chí Minh"
        self.text = (
            "TOP GIA® bán SKU-ABC-123 giá 35.000đ/chai, gọi 0902.111.333, tại "
            + self.address
            + "."
        )

    def test_detects_all_critical_content_kinds_without_overlap(self) -> None:
        spans = find_protected_spans(
            self.text,
            brands=["top gia"],
            addresses=[self.address],
        )
        kinds = {span.kind for span in spans}
        self.assertTrue({"brand", "sku", "price", "phone", "address"} <= kinds)
        self.assertIn("35.000đ/chai", [span.text for span in spans])
        self.assertIn("0902.111.333", [span.text for span in spans])
        for previous, current in zip(spans, spans[1:]):
            self.assertLessEqual(previous.end, current.start)

    def test_generic_number_is_locked_when_it_is_not_a_price_or_phone(self) -> None:
        spans = find_protected_spans("Chai 750ml được cộng 5 điểm, niêm yết 35.000")
        self.assertEqual(
            [(span.kind, span.text) for span in spans],
            [
                ("number", "750ml"),
                ("number", "5 điểm"),
                ("number", "35.000"),
            ],
        )

    def test_mixed_case_sku_after_a_product_code_cue_is_locked_as_one_span(self) -> None:
        source = "Mã ab-123"
        spans = find_protected_spans(source)

        sku_spans = [span for span in spans if span.kind == "sku"]
        self.assertEqual(len(sku_spans), 1)
        self.assertIn("ab-123", sku_spans[0].text)
        proposal = build_conservative_proposal(source, "Mã ax-123", confidence=0.99)
        self.assertFalse(proposal.safe)
        self.assertFalse(proposal.changed)
        self.assertEqual(proposal.proposed_text, source)

    def test_plain_mask_and_restore_is_exact(self) -> None:
        protected = protect_text(
            self.text,
            brands=["TOP GIA"],
            addresses=[self.address],
        )
        self.assertNotEqual(protected.masked_text, protected.source_text)
        self.assertEqual(
            restore_protected_text(protected.masked_text, protected.spans),
            normalize_nfc(self.text),
        )

    def test_t5_mask_uses_existing_sentinels_and_restores_exactly(self) -> None:
        protected = protect_text("Giá 35.000đ, gọi 0902111333", placeholder_style="t5")
        self.assertIn("<extra_id_0>", protected.masked_text)
        self.assertIn("<extra_id_1>", protected.masked_text)
        self.assertEqual(
            restore_protected_text(protected.masked_text, protected.spans),
            "Giá 35.000đ, gọi 0902111333",
        )

    def test_missing_or_reordered_placeholder_fails_closed(self) -> None:
        protected = protect_text("Giá 35.000đ, gọi 0902111333")
        first, second = (span.placeholder for span in protected.spans)
        assert first is not None and second is not None
        with self.assertRaises(ProtectedTextError):
            restore_protected_text(protected.masked_text.replace(first, ""), protected.spans)

        swapped = protected.masked_text.replace(first, "__TEMP__")
        swapped = swapped.replace(second, first).replace("__TEMP__", second)
        with self.assertRaises(ProtectedTextError):
            restore_protected_text(swapped, protected.spans)


class ConservativeProposalTests(unittest.TestCase):
    def test_changed_candidate_is_only_a_review_proposal(self) -> None:
        proposal = build_conservative_proposal(
            "ĐÔ GIA DỤNG 35.000đ",
            "ĐỒ GIA DỤNG 35.000đ",
            confidence=0.91,
            model="unit-test",
        )
        self.assertTrue(proposal.changed)
        self.assertTrue(proposal.requires_approval)
        self.assertTrue(proposal.safe)
        self.assertEqual(proposal.proposed_text, "ĐỒ GIA DỤNG 35.000đ")
        self.assertIn("human-approval-required", proposal.reasons)

    def test_candidate_that_changes_a_price_is_discarded(self) -> None:
        proposal = build_conservative_proposal(
            "Giá 35.000đ",
            "Giá 45.000đ",
            confidence=0.99,
        )
        self.assertFalse(proposal.safe)
        self.assertFalse(proposal.changed)
        self.assertEqual(proposal.proposed_text, "Giá 35.000đ")
        self.assertEqual(proposal.confidence, 0.0)
        self.assertIn("protected-content-mismatch", proposal.reasons)

    def test_candidate_that_invents_an_extra_price_is_discarded(self) -> None:
        proposal = build_conservative_proposal(
            "Giá 35.000đ",
            "Giá 35.000đ hoặc 45.000đ",
            confidence=0.99,
        )
        self.assertFalse(proposal.safe)
        self.assertFalse(proposal.changed)
        self.assertEqual(proposal.proposed_text, "Giá 35.000đ")
        self.assertEqual(proposal.confidence, 0.0)
        self.assertIn("protected-content-mismatch", proposal.reasons)

    def test_nfc_only_change_is_not_reported_as_content_correction(self) -> None:
        source = "Đo\u0302\u0300 gia dụng"
        proposal = build_conservative_proposal(source)
        self.assertEqual(proposal.normalized_text, "Đồ gia dụng")
        self.assertFalse(proposal.changed)
        self.assertIn("nfc-normalized", proposal.reasons)

    def test_text_region_contract_round_trips_through_json(self) -> None:
        proposal = build_conservative_proposal("ĐÔ", "ĐỒ", confidence=0.7)
        region = TextRegion(
            region_id="text-0001",
            bbox=(10, 20, 110, 60),
            polygon=((10.0, 20.0), (110.0, 20.0), (110.0, 60.0), (10.0, 60.0)),
            observations=[OCRObservation("paddle", "original", "ĐÔ", 0.82)],
            selected_text="ĐÔ",
            proposal=proposal,
            status="yellow",
            critical=True,
            reasons=["ocr-disagreement"],
        )
        encoded = json.loads(json.dumps(region.to_dict(), ensure_ascii=False))
        restored = TextRegion.from_dict(encoded)
        self.assertEqual(restored.to_dict(), region.to_dict())
        self.assertEqual(restored.proposal.candidate_text, "ĐỒ")  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
