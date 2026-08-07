from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import threading
import unicodedata
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from PIL import Image


V7_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7_DIR))

from user_review import (  # noqa: E402
    LOOPBACK_HOST,
    ProtectedChangeConfirmationRequired,
    ReviewConflictError,
    ReviewPathError,
    ReviewSession,
    ReviewValidationError,
    create_review_server,
    render_review_html,
    validate_review_document,
)
from v7lib.review import build_review_document, write_review_document  # noqa: E402
from v7lib.types import OCRObservation, TextRegion  # noqa: E402


class ReviewBundle:
    def __init__(
        self,
        root: Path,
        *,
        text: str = "ĐỒ GIA DỤNG",
        suggested: str | None = None,
        status: str = "yellow",
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_path = root / "poster_SOURCE_NORMALIZED.png"
        Image.new("RGB", (320, 160), (246, 238, 211)).save(self.image_path)
        region = TextRegion(
            region_id="text_0001",
            bbox=(38, 42, 282, 104),
            polygon=((38.0, 42.0), (282.0, 42.0), (282.0, 104.0), (38.0, 104.0)),
            observations=[OCRObservation("paddle", "original", text, 0.82)],
            selected_text=text,
            proposal=None,
            status=status,  # type: ignore[arg-type]
            critical=any(character.isdigit() for character in text),
            reasons=["recognition_confidence_below_green_gate"],
        )
        self.document = build_review_document(
            self.image_path,
            [region],
            review_mode="strict",
        )
        self.document["source"] = self.image_path.name
        self.document["source_sha256"] = "a" * 64
        if suggested is not None:
            self.document["regions"][0]["suggested_text"] = suggested  # type: ignore[index]
        self.review_path = root / "TEXT_REVIEW.json"
        write_review_document(self.document, self.review_path)

    @property
    def fingerprint(self) -> str:
        return str(self.document["regions"][0]["region_fingerprint"])  # type: ignore[index]


class ManyReviewBundle:
    def __init__(
        self,
        root: Path,
        *,
        count: int,
        existing_replace: bool = False,
        existing_skip: bool = False,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        columns = 8
        rows = (count + columns - 1) // columns
        self.image_path = root / "catalog_SOURCE_NORMALIZED.png"
        Image.new("RGB", (1_280, max(120, rows * 58 + 20)), (250, 248, 240)).save(
            self.image_path
        )
        regions: list[TextRegion] = []
        for index in range(count):
            column = index % columns
            row = index // columns
            x0, y0 = 10 + column * 158, 10 + row * 58
            x1, y1 = x0 + 140, y0 + 38
            text = "Giá 35.000đ – ĐỒ GIA DỤNG" if index == 2 else f"VÙNG ĐỒ HỌA {index + 1}"
            regions.append(
                TextRegion(
                    region_id=f"text_{index + 1:04d}",
                    bbox=(x0, y0, x1, y1),
                    polygon=(
                        (float(x0), float(y0)),
                        (float(x1), float(y0)),
                        (float(x1), float(y1)),
                        (float(x0), float(y1)),
                    ),
                    observations=[OCRObservation("paddle", "original", text, 0.75)],
                    selected_text=text,
                    proposal=None,
                    status="yellow",
                    critical=index == 2,
                    reasons=["recognition_confidence_below_green_gate"],
                )
            )
        self.document = build_review_document(
            self.image_path,
            regions,
            review_mode="strict",
        )
        self.document["source"] = self.image_path.name
        self.document["source_sha256"] = "b" * 64
        document_rows = self.document["regions"]
        assert isinstance(document_rows, list)
        if existing_replace:
            document_rows[0]["action"] = "replace"
            document_rows[0]["approved_text"] = "NỘI DUNG ĐÃ DUYỆT"
            document_rows[0]["decision_source"] = "existing-user-replace"
        if existing_skip:
            document_rows[1]["action"] = "skip"
            document_rows[1]["approved_text"] = ""
            document_rows[1]["decision_source"] = "existing-user-skip"
        self.review_path = root / "TEXT_REVIEW.json"
        write_review_document(self.document, self.review_path)


class ReviewSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.bundle = ReviewBundle(Path(self.temporary.name) / "bundle")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_state_crop_and_html_are_simple_vietnamese_unicode(self) -> None:
        session = ReviewSession.open(self.bundle.review_path)
        state = session.public_state()
        row = state["regions"][0]  # type: ignore[index]

        self.assertEqual(row["selected_text"], "ĐỒ GIA DỤNG")
        self.assertEqual(row["status_label"], "Cần kiểm tra")
        self.assertEqual(state["summary"]["pending"], 1)  # type: ignore[index]
        crop = session.crop_png(self.bundle.fingerprint)
        self.assertTrue(crop.startswith(b"\x89PNG\r\n\x1a\n"))

        html, nonce = render_review_html(session.token)
        self.assertIn("Giữ nguyên vùng ảnh", html)
        self.assertIn("Thay bằng nội dung trên", html)
        self.assertIn("Bỏ qua vùng này", html)
        self.assertIn("Giữ nguyên tất cả vùng còn lại", html)
        self.assertIn("/api/bulk-keep-pending", html)
        self.assertIn("window.confirm", html)
        self.assertNotIn("bulk-replace", html)
        self.assertIn("Duyệt chữ trước khi phục dựng", html)
        self.assertIn(session.token, html)
        self.assertIn(nonce, html)
        self.assertNotIn(str(self.bundle.review_path), html)

    def test_unicode_replacement_is_nfc_and_saved_atomically(self) -> None:
        session = ReviewSession.open(self.bundle.review_path)
        decomposed = unicodedata.normalize("NFD", "ĐỒ GIA DỤNG MỚI")

        with patch("user_review.os.replace", wraps=os.replace) as replace:
            state = session.apply_decision(
                self.bundle.fingerprint,
                "replace",
                approved_text=decomposed,
            )

        replace.assert_called_once()
        saved = json.loads(self.bundle.review_path.read_text(encoding="utf-8"))
        approved = saved["regions"][0]["approved_text"]
        self.assertEqual(approved, "ĐỒ GIA DỤNG MỚI")
        self.assertTrue(unicodedata.is_normalized("NFC", approved))
        self.assertEqual(saved["regions"][0]["action"], "replace")
        self.assertEqual(state["summary"]["pending"], 0)  # type: ignore[index]
        self.assertEqual(list(self.bundle.root.glob(".TEXT_REVIEW.json.*.tmp")), [])

    def test_numbers_and_mixed_case_sku_never_change_without_confirmation(self) -> None:
        protected = ReviewBundle(
            Path(self.temporary.name) / "protected",
            text="Mã ab-123 giá 35.000đ",
            suggested="Mã ax-123 giá 45.000đ",
        )
        session = ReviewSession.open(protected.review_path)
        before = protected.review_path.read_bytes()

        with self.assertRaises(ProtectedChangeConfirmationRequired) as raised:
            session.apply_decision(
                protected.fingerprint,
                "replace",
                approved_text="Mã ax-123 giá 45.000đ",
            )

        self.assertTrue(
            any("ab-123" in value for value in raised.exception.protected_values)
        )
        self.assertIn("35.000đ", " ".join(raised.exception.protected_values))
        self.assertEqual(protected.review_path.read_bytes(), before)
        session.apply_decision(protected.fingerprint, "keep")
        saved = json.loads(protected.review_path.read_text(encoding="utf-8"))
        row = saved["regions"][0]
        self.assertEqual(row["action"], "keep")
        self.assertEqual(row["approved_text"], "")
        self.assertEqual(row["selected_text"], "Mã ab-123 giá 35.000đ")
        self.assertEqual(row["suggested_text"], "Mã ax-123 giá 45.000đ")

    def test_explicit_confirmation_can_record_an_intentional_protected_change(self) -> None:
        protected = ReviewBundle(
            Path(self.temporary.name) / "confirmed",
            text="SKU Xy-987 giá 35.000đ",
        )
        session = ReviewSession.open(protected.review_path)
        session.apply_decision(
            protected.fingerprint,
            "replace",
            approved_text="SKU Xz-987 giá 45.000đ",
            confirm_protected_change=True,
        )
        saved = json.loads(protected.review_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["regions"][0]["approved_text"], "SKU Xz-987 giá 45.000đ")

    def test_invalid_decisions_do_not_touch_the_file(self) -> None:
        session = ReviewSession.open(self.bundle.review_path)
        before = self.bundle.review_path.read_bytes()
        with self.assertRaises(ReviewValidationError):
            session.apply_decision(self.bundle.fingerprint, "pending")
        with self.assertRaises(ReviewValidationError):
            session.apply_decision("f" * 64, "keep")
        with self.assertRaises(ReviewValidationError):
            session.apply_decision(
                self.bundle.fingerprint,
                "replace",
                approved_text="   ",
            )
        self.assertEqual(self.bundle.review_path.read_bytes(), before)

    def test_external_file_change_is_not_overwritten(self) -> None:
        session = ReviewSession.open(self.bundle.review_path)
        with self.bundle.review_path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        changed = self.bundle.review_path.read_bytes()
        with self.assertRaises(ReviewConflictError):
            session.apply_decision(self.bundle.fingerprint, "keep")
        self.assertEqual(self.bundle.review_path.read_bytes(), changed)

    def test_bulk_keep_changes_only_pending_rows_in_one_atomic_unicode_write(self) -> None:
        bundle = ManyReviewBundle(
            Path(self.temporary.name) / "many",
            count=8,
            existing_replace=True,
            existing_skip=True,
        )
        before = json.loads(bundle.review_path.read_text(encoding="utf-8"))
        session = ReviewSession.open(bundle.review_path)

        with patch("user_review.os.replace", wraps=os.replace) as replace:
            state = session.keep_all_pending()

        replace.assert_called_once()
        saved = json.loads(bundle.review_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["regions"][0], before["regions"][0])
        self.assertEqual(saved["regions"][1], before["regions"][1])
        for index in range(2, 8):
            old_row = before["regions"][index]
            new_row = saved["regions"][index]
            self.assertEqual(new_row["action"], "keep")
            self.assertEqual(new_row["approved_text"], "")
            self.assertEqual(
                new_row["decision_source"],
                "local-user-review-ui-bulk-keep",
            )
            for key, value in old_row.items():
                if key not in {"action", "approved_text", "decision_source"}:
                    self.assertEqual(new_row[key], value)
        protected = saved["regions"][2]
        self.assertEqual(protected["selected_text"], "Giá 35.000đ – ĐỒ GIA DỤNG")
        self.assertEqual(protected["action"], "keep")
        self.assertEqual(state["summary"]["pending"], 0)  # type: ignore[index]
        self.assertEqual(list(bundle.root.glob(".TEXT_REVIEW.json.*.tmp")), [])

    def test_catalog_sized_bulk_keep_handles_135_pending_in_one_write(self) -> None:
        bundle = ManyReviewBundle(
            Path(self.temporary.name) / "catalog-136",
            count=136,
            existing_replace=True,
        )
        session = ReviewSession.open(bundle.review_path)
        self.assertEqual(session.public_state()["summary"]["pending"], 135)  # type: ignore[index]

        with patch("user_review.os.replace", wraps=os.replace) as replace:
            state = session.keep_all_pending()

        replace.assert_called_once()
        self.assertEqual(state["summary"]["pending"], 0)  # type: ignore[index]
        saved = json.loads(bundle.review_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["regions"][0]["action"], "replace")
        self.assertTrue(
            all(row["action"] == "keep" for row in saved["regions"][1:])
        )

    def test_bulk_atomic_failure_preserves_original_document(self) -> None:
        bundle = ManyReviewBundle(
            Path(self.temporary.name) / "bulk-failure",
            count=5,
            existing_replace=True,
        )
        session = ReviewSession.open(bundle.review_path)
        before = bundle.review_path.read_bytes()

        with patch("user_review.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                session.keep_all_pending()

        self.assertEqual(bundle.review_path.read_bytes(), before)
        self.assertEqual(list(bundle.root.glob(".TEXT_REVIEW.json.*.tmp")), [])


class ReviewValidationAndPathTests(unittest.TestCase):
    def test_source_path_cannot_escape_the_review_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = ReviewBundle(root / "bundle")
            outside = root / "outside.png"
            Image.new("RGB", (320, 160), "white").save(outside)
            document = json.loads(bundle.review_path.read_text(encoding="utf-8"))
            document["source"] = "../outside.png"
            write_review_document(document, bundle.review_path)

            with self.assertRaises(ReviewPathError):
                ReviewSession.open(bundle.review_path)

    def test_tampered_fingerprint_and_outside_bbox_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = ReviewBundle(Path(temporary) / "bundle")
            document = json.loads(bundle.review_path.read_text(encoding="utf-8"))
            document["regions"][0]["selected_text"] = "ĐÃ BỊ SỬA"
            with self.assertRaises(ReviewValidationError):
                validate_review_document(document, image_size=(320, 160))

            document = json.loads(bundle.review_path.read_text(encoding="utf-8"))
            document["regions"][0]["bbox"] = [0, 0, 999, 999]
            with self.assertRaises(ReviewValidationError):
                validate_review_document(document, image_size=(320, 160))

    def test_server_refuses_non_loopback_bind_and_uses_random_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = ReviewBundle(Path(temporary) / "bundle")
            with self.assertRaises(ReviewPathError):
                create_review_server(bundle.review_path, host="0.0.0.0")

            handle = create_review_server(bundle.review_path)
            try:
                self.assertEqual(handle.server.server_address[0], LOOPBACK_HOST)
                self.assertIn(LOOPBACK_HOST, handle.url)
                self.assertGreaterEqual(len(handle.session.token), 32)
                self.assertFalse(handle.session.finished.is_set())
                handle.session.mark_finished()
                self.assertTrue(handle.session.finished.is_set())
            finally:
                handle.close()

    def test_http_server_requires_token_and_saves_only_authorized_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = ReviewBundle(Path(temporary) / "bundle")
            handle = create_review_server(bundle.review_path)
            thread = threading.Thread(target=handle.serve_forever, daemon=True)
            thread.start()
            try:
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    urllib.request.urlopen(handle.server.origin + "/", timeout=3)
                self.assertEqual(denied.exception.code, 403)

                with urllib.request.urlopen(handle.url, timeout=3) as response:
                    html = response.read().decode("utf-8")
                    self.assertEqual(response.status, 200)
                    self.assertIn("Content-Security-Policy", response.headers)
                    self.assertIn("Giữ nguyên vùng ảnh", html)

                body = json.dumps(
                    {
                        "region_fingerprint": bundle.fingerprint,
                        "action": "keep",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                request = urllib.request.Request(
                    handle.server.origin + "/api/decision",
                    data=body,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "X-V7-Review-Token": handle.session.token,
                    },
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    state = json.loads(response.read().decode("utf-8"))
                self.assertEqual(state["summary"]["pending"], 0)
                saved = json.loads(bundle.review_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["regions"][0]["action"], "keep")
            finally:
                handle.server.shutdown()
                thread.join(timeout=3)
                handle.close()

    def test_finish_returns_complete_json_before_server_stops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = ReviewBundle(Path(temporary) / "bundle")
            handle = create_review_server(bundle.review_path)

            def serve_and_close() -> None:
                try:
                    handle.serve_forever()
                finally:
                    handle.close()

            thread = threading.Thread(target=serve_and_close, daemon=True)
            thread.start()
            request = urllib.request.Request(
                handle.server.origin + "/api/finish",
                data=b"{}",
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "X-V7-Review-Token": handle.session.token,
                },
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(response.status, 200)

            self.assertEqual(payload["summary"]["total"], 1)
            self.assertTrue(handle.session.finished.is_set())
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())

    def test_bulk_http_requires_token_and_exact_keep_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = ManyReviewBundle(
                Path(temporary) / "bundle",
                count=6,
                existing_replace=True,
                existing_skip=True,
            )
            original_document = copy.deepcopy(bundle.document)
            original_bytes = bundle.review_path.read_bytes()
            handle = create_review_server(bundle.review_path)
            thread = threading.Thread(target=handle.serve_forever, daemon=True)
            thread.start()

            def request(action: str, token: str) -> dict[str, object]:
                body = json.dumps({"action": action}, ensure_ascii=False).encode("utf-8")
                http_request = urllib.request.Request(
                    handle.server.origin + "/api/bulk-keep-pending",
                    data=body,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "X-V7-Review-Token": token,
                    },
                )
                with urllib.request.urlopen(http_request, timeout=3) as response:
                    return json.loads(response.read().decode("utf-8"))

            try:
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    request("keep", "wrong-token")
                self.assertEqual(denied.exception.code, 403)
                self.assertEqual(bundle.review_path.read_bytes(), original_bytes)

                with self.assertRaises(urllib.error.HTTPError) as invalid:
                    request("replace", handle.session.token)
                self.assertEqual(invalid.exception.code, 422)
                self.assertEqual(bundle.review_path.read_bytes(), original_bytes)

                state = request("keep", handle.session.token)
                self.assertEqual(state["summary"]["pending"], 0)  # type: ignore[index]
                saved = json.loads(bundle.review_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["regions"][0], original_document["regions"][0])
                self.assertEqual(saved["regions"][1], original_document["regions"][1])
                for row in saved["regions"][2:]:
                    self.assertEqual(row["action"], "keep")
                    self.assertEqual(row["approved_text"], "")
                    self.assertEqual(
                        row["decision_source"],
                        "local-user-review-ui-bulk-keep",
                    )
            finally:
                handle.server.shutdown()
                thread.join(timeout=3)
                handle.close()

    def test_atomic_failure_preserves_original_and_removes_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = ReviewBundle(Path(temporary) / "bundle")
            session = ReviewSession.open(bundle.review_path)
            before = bundle.review_path.read_bytes()
            with patch("user_review.os.replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    session.apply_decision(bundle.fingerprint, "keep")
            self.assertEqual(bundle.review_path.read_bytes(), before)
            self.assertEqual(list(bundle.root.glob(".TEXT_REVIEW.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
