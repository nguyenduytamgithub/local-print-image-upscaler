"""Simple, local-only review UI for V7 text decisions.

This module intentionally sits beside the rendering engine instead of inside
it.  The browser never edits technical JSON directly: it sends one bounded
decision at a time to a token-protected loopback server, and the server writes
``TEXT_REVIEW.json`` atomically.  OCR text, geometry, fingerprints and model
proposals are immutable through this API.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
import unicodedata
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, quote, urlparse

from PIL import Image, ImageDraw

from v7lib.review import REVIEW_SCHEMA, region_fingerprint
from v7lib.textnorm import find_protected_spans, protected_content_preserved


LOOPBACK_HOST = "127.0.0.1"
MAX_REQUEST_BYTES = 64 * 1024
MAX_APPROVED_TEXT_CHARS = 4_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ACTIONS = {"pending", "replace", "keep", "skip"}
_USER_ACTIONS = _ACTIONS - {"pending"}
_STATUS_LABELS = {
    "green": "Đã đối chiếu tốt",
    "yellow": "Cần kiểm tra",
    "red": "Không đọc chắc chắn",
}
_REASON_LABELS = {
    "protected_or_numeric_content_requires_approval": "Có số, giá hoặc mã cần kiểm tra kỹ.",
    "no_independent_engine_agreement": "Hai bộ đọc chữ chưa đồng ý với nhau.",
    "independent_engine_confidence_below_green_gate": "Độ tin cậy của bộ đọc độc lập còn thấp.",
    "recognition_confidence_below_green_gate": "Độ tin cậy nhận dạng còn thấp.",
    "unreadable_text_requires_user_entry": "Chữ quá mờ; vui lòng nhập nội dung đúng.",
    "language_model_change_requires_approval": "Có đề xuất sửa chữ; chỉ dùng khi bạn xác nhận.",
    "language_proposals_disagree": "Các đề xuất sửa chữ chưa thống nhất.",
    "old_text_mask_failed_quality_gate": "Vùng chữ cũ chưa đủ an toàn để xóa tự động.",
}


class ReviewUIError(RuntimeError):
    """Base error for the local review layer."""


class ReviewValidationError(ReviewUIError):
    """The review document or a submitted decision is invalid."""


class ReviewPathError(ReviewUIError):
    """A requested path escapes the review bundle or is not usable."""


class ReviewConflictError(ReviewUIError):
    """The review file changed on disk while the UI was open."""


class ProtectedChangeConfirmationRequired(ReviewValidationError):
    """A human must explicitly confirm changing a number, price or identifier."""

    def __init__(self, protected_values: list[str]) -> None:
        self.protected_values = protected_values
        joined = ", ".join(protected_values) if protected_values else "dữ liệu quan trọng"
        super().__init__(f"Bạn đang thay đổi số, giá hoặc mã: {joined}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewValidationError(f"Không đọc được file duyệt: {path}") from exc
    if not isinstance(raw, dict):
        raise ReviewValidationError("File duyệt phải là một JSON object.")
    return raw


def _atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
    """Write UTF-8 JSON beside the target, then atomically replace it."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def _resolve_review_path(path: Path | str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise ReviewPathError(f"Không tìm thấy file duyệt: {value}")
    if value.suffix.lower() != ".json":
        raise ReviewPathError("File duyệt phải có phần mở rộng .json.")
    return value


def _resolve_bundle_image(
    review_path: Path,
    document: Mapping[str, object],
    explicit_image: Path | str | None,
) -> Path:
    root = review_path.parent.resolve()
    raw: object = explicit_image if explicit_image is not None else document.get("source")
    if not isinstance(raw, (str, os.PathLike)) or not str(raw).strip():
        raise ReviewPathError("File duyệt không chỉ ra ảnh nguồn trong bundle.")
    supplied = Path(raw).expanduser()
    candidate = supplied if supplied.is_absolute() else root / supplied
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ReviewPathError(f"Không tìm thấy ảnh nguồn của file duyệt: {candidate}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReviewPathError("Ảnh nguồn phải nằm trong cùng bundle với TEXT_REVIEW.json.") from exc
    if not resolved.is_file() or resolved == review_path:
        raise ReviewPathError("Đường dẫn ảnh nguồn không hợp lệ.")
    try:
        with Image.open(resolved) as image:
            image.verify()
    except (OSError, ValueError) as exc:
        raise ReviewPathError(f"Không mở được ảnh nguồn: {resolved}") from exc
    return resolved


def _integer_bbox(value: object, *, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ReviewValidationError("Mỗi vùng duyệt phải có bbox gồm bốn số nguyên.")
    if any(isinstance(item, bool) for item in value):
        raise ReviewValidationError("Tọa độ bbox không hợp lệ.")
    try:
        bbox = tuple(int(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReviewValidationError("Tọa độ bbox không hợp lệ.") from exc
    x0, y0, x1, y1 = bbox
    width, height = image_size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ReviewValidationError(
            f"BBox {bbox} nằm ngoài ảnh {width}x{height}."
        )
    return bbox  # type: ignore[return-value]


def validate_review_document(
    document: Mapping[str, object],
    *,
    image_size: tuple[int, int],
) -> None:
    """Validate all authority-bearing fields before starting a server."""

    if document.get("schema") != REVIEW_SCHEMA:
        raise ReviewValidationError("Sai phiên bản định dạng TEXT_REVIEW.json.")
    source_sha = str(document.get("source_sha256", "")).strip().lower()
    if not _SHA256_RE.fullmatch(source_sha):
        raise ReviewValidationError("TEXT_REVIEW.json thiếu source SHA-256 hợp lệ.")
    rows = document.get("regions")
    if not isinstance(rows, list):
        raise ReviewValidationError("Trường regions phải là một danh sách.")
    fingerprints: set[str] = set()
    identifiers: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ReviewValidationError("Mỗi vùng duyệt phải là một object.")
        region_id = row.get("region_id")
        if not isinstance(region_id, str) or not region_id.strip() or region_id in identifiers:
            raise ReviewValidationError("region_id thiếu hoặc bị trùng.")
        identifiers.add(region_id)
        _integer_bbox(row.get("bbox"), image_size=image_size)
        status = row.get("status")
        if status not in _STATUS_LABELS:
            raise ReviewValidationError(f"Trạng thái vùng {region_id} không hợp lệ.")
        action = row.get("action", "pending")
        if action not in _ACTIONS:
            raise ReviewValidationError(f"Quyết định vùng {region_id} không hợp lệ.")
        for key in ("selected_text", "suggested_text", "approved_text"):
            if row.get(key, "") is not None and not isinstance(row.get(key, ""), str):
                raise ReviewValidationError(f"{key} của vùng {region_id} phải là chuỗi.")
        fingerprint = str(row.get("region_fingerprint", "")).strip().lower()
        if not _SHA256_RE.fullmatch(fingerprint) or fingerprint in fingerprints:
            raise ReviewValidationError("Fingerprint vùng thiếu, sai hoặc bị trùng.")
        try:
            expected = region_fingerprint(row)
        except ValueError as exc:
            raise ReviewValidationError(f"Hình học vùng {region_id} không hợp lệ.") from exc
        if not secrets.compare_digest(fingerprint, expected):
            raise ReviewValidationError(
                f"Vùng {region_id} đã bị thay đổi sau khi tạo fingerprint."
            )
        fingerprints.add(fingerprint)


def _clean_approved_text(value: object) -> str:
    if not isinstance(value, str):
        raise ReviewValidationError("Nội dung thay thế phải là chuỗi Unicode.")
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = normalized.strip()
    if not normalized:
        raise ReviewValidationError("Hãy nhập nội dung đúng trước khi chọn Thay bằng.")
    if len(normalized) > MAX_APPROVED_TEXT_CHARS:
        raise ReviewValidationError(
            f"Nội dung thay thế vượt quá {MAX_APPROVED_TEXT_CHARS} ký tự."
        )
    if any(unicodedata.category(character) == "Cc" and character not in "\n\t" for character in normalized):
        raise ReviewValidationError("Nội dung thay thế chứa ký tự điều khiển không an toàn.")
    return normalized


def _friendly_reasons(row: Mapping[str, object]) -> list[str]:
    raw = row.get("reasons", [])
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for value in raw:
        label = _REASON_LABELS.get(str(value))
        if label and label not in result:
            result.append(label)
    return result


@dataclass(slots=True)
class ReviewSession:
    review_path: Path
    image_path: Path
    document: dict[str, object]
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    finished: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _disk_sha256: str = field(default="", repr=False)
    _image_size: tuple[int, int] = field(default=(0, 0), repr=False)
    _crop_cache: dict[str, bytes] = field(default_factory=dict, repr=False)

    @classmethod
    def open(
        cls,
        review_path: Path | str,
        *,
        image_path: Path | str | None = None,
    ) -> "ReviewSession":
        resolved_review = _resolve_review_path(review_path)
        document = _load_json(resolved_review)
        resolved_image = _resolve_bundle_image(resolved_review, document, image_path)
        with Image.open(resolved_image) as image:
            image.load()
            image_size = image.size
        validate_review_document(document, image_size=image_size)
        return cls(
            review_path=resolved_review,
            image_path=resolved_image,
            document=copy.deepcopy(document),
            _disk_sha256=_sha256_file(resolved_review),
            _image_size=image_size,
        )

    def _rows(self) -> list[dict[str, object]]:
        rows = self.document.get("regions", [])
        assert isinstance(rows, list)
        return [row for row in rows if isinstance(row, dict)]

    def _row(self, fingerprint: str) -> dict[str, object]:
        value = str(fingerprint).strip().lower()
        if not _SHA256_RE.fullmatch(value):
            raise ReviewValidationError("Fingerprint vùng không hợp lệ.")
        for row in self._rows():
            if secrets.compare_digest(str(row.get("region_fingerprint", "")).lower(), value):
                return row
        raise ReviewValidationError("Không tìm thấy vùng duyệt này.")

    def public_state(self) -> dict[str, object]:
        with self._lock:
            regions: list[dict[str, object]] = []
            for row in self._rows():
                fingerprint = str(row["region_fingerprint"])
                selected = str(row.get("selected_text") or "")
                suggested = str(row.get("suggested_text") or selected)
                spans = find_protected_spans(selected)
                action = str(row.get("action", "pending"))
                approved = str(row.get("approved_text") or "")
                regions.append(
                    {
                        "region_id": str(row["region_id"]),
                        "region_fingerprint": fingerprint,
                        "status": str(row["status"]),
                        "status_label": _STATUS_LABELS[str(row["status"])],
                        "selected_text": selected,
                        "suggested_text": suggested,
                        "input_text": approved if action == "replace" and approved else suggested,
                        "action": action,
                        "critical": bool(row.get("critical", False) or spans),
                        "protected_values": [span.text for span in spans],
                        "reasons": _friendly_reasons(row),
                        "crop_url": f"/crop/{quote(fingerprint)}.png?token={quote(self.token)}",
                    }
                )
            pending = sum(item["action"] == "pending" for item in regions)
            return {
                "title": "V7 – Duyệt chữ trước khi phục dựng",
                "regions": regions,
                "summary": {
                    "total": len(regions),
                    "pending": pending,
                    "decided": len(regions) - pending,
                },
            }

    def apply_decision(
        self,
        fingerprint: str,
        action: str,
        *,
        approved_text: object = "",
        confirm_protected_change: bool = False,
    ) -> dict[str, object]:
        normalized_action = str(action).strip().lower()
        if normalized_action not in _USER_ACTIONS:
            raise ReviewValidationError("Chỉ nhận Giữ nguyên, Thay bằng hoặc Bỏ qua.")
        with self._lock:
            if _sha256_file(self.review_path) != self._disk_sha256:
                raise ReviewConflictError(
                    "TEXT_REVIEW.json đã thay đổi ở nơi khác; hãy tải lại trang để tránh ghi đè."
                )
            candidate = copy.deepcopy(self.document)
            rows = candidate.get("regions", [])
            assert isinstance(rows, list)
            row = next(
                (
                    item
                    for item in rows
                    if isinstance(item, dict)
                    and secrets.compare_digest(
                        str(item.get("region_fingerprint", "")).lower(),
                        str(fingerprint).strip().lower(),
                    )
                ),
                None,
            )
            if row is None:
                raise ReviewValidationError("Không tìm thấy vùng duyệt này.")
            # Validate the fingerprint again immediately before a state change.
            if not secrets.compare_digest(
                str(row.get("region_fingerprint", "")).lower(), region_fingerprint(row)
            ):
                raise ReviewValidationError("Vùng duyệt đã bị sửa sau khi tạo fingerprint.")

            if normalized_action == "replace":
                approved = _clean_approved_text(approved_text)
                selected = unicodedata.normalize("NFC", str(row.get("selected_text") or ""))
                spans = find_protected_spans(selected)
                if spans and not protected_content_preserved(approved, spans):
                    if confirm_protected_change is not True:
                        raise ProtectedChangeConfirmationRequired([span.text for span in spans])
                row["approved_text"] = approved
            else:
                row["approved_text"] = ""
            row["action"] = normalized_action
            row["decision_source"] = "local-user-review-ui"
            validate_review_document(candidate, image_size=self._image_size)
            _atomic_write_json(self.review_path, candidate)
            self.document = candidate
            self._disk_sha256 = _sha256_file(self.review_path)
            self._crop_cache.clear()
            return self.public_state()

    def keep_all_pending(self) -> dict[str, object]:
        """Atomically retain the raster for every still-undecided region.

        This is intentionally narrower than a generic bulk-decision API.  It
        cannot approve proposed text, cannot alter an existing decision, and
        cannot change any OCR/protected content field.
        """

        with self._lock:
            if _sha256_file(self.review_path) != self._disk_sha256:
                raise ReviewConflictError(
                    "TEXT_REVIEW.json đã thay đổi ở nơi khác; hãy tải lại trang để tránh ghi đè."
                )
            candidate = copy.deepcopy(self.document)
            rows = candidate.get("regions", [])
            assert isinstance(rows, list)
            changed = 0
            for row in rows:
                if not isinstance(row, dict) or row.get("action", "pending") != "pending":
                    continue
                # Keep means retaining the original raster.  In particular,
                # protected prices/SKUs are not accepted, rewritten or skipped.
                row["action"] = "keep"
                row["approved_text"] = ""
                row["decision_source"] = "local-user-review-ui-bulk-keep"
                changed += 1
            if changed == 0:
                return self.public_state()
            validate_review_document(candidate, image_size=self._image_size)
            _atomic_write_json(self.review_path, candidate)
            self.document = candidate
            self._disk_sha256 = _sha256_file(self.review_path)
            self._crop_cache.clear()
            return self.public_state()

    def crop_png(self, fingerprint: str) -> bytes:
        with self._lock:
            key = str(fingerprint).strip().lower()
            if key in self._crop_cache:
                return self._crop_cache[key]
            row = self._row(key)
            x0, y0, x1, y1 = _integer_bbox(row.get("bbox"), image_size=self._image_size)
            line_height = y1 - y0
            pad = max(16, min(120, int(round(line_height * 0.85))))
            left = max(0, x0 - pad)
            top = max(0, y0 - pad)
            right = min(self._image_size[0], x1 + pad)
            bottom = min(self._image_size[1], y1 + pad)
            with Image.open(self.image_path) as opened:
                opened.load()
                crop = opened.convert("RGB").crop((left, top, right, bottom))
            draw = ImageDraw.Draw(crop)
            box = (x0 - left, y0 - top, x1 - left - 1, y1 - top - 1)
            stroke = max(2, min(5, line_height // 18))
            draw.rectangle(box, outline=(0, 132, 255), width=stroke)
            crop.thumbnail((1200, 520), Image.Resampling.LANCZOS)
            output = BytesIO()
            crop.save(output, format="PNG", compress_level=4)
            value = output.getvalue()
            self._crop_cache[key] = value
            return value

    def mark_finished(self) -> dict[str, object]:
        state = self.public_state()
        self.finished.set()
        return state


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>V7 – Duyệt chữ</title>
  <style nonce="__NONCE__">
    :root { color-scheme: light; font-family: "Segoe UI", Arial, sans-serif; background:#f3f6fb; color:#162033; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; }
    header { background:#14233b; color:white; padding:18px 24px; display:flex; gap:16px; align-items:center; justify-content:space-between; }
    header h1 { margin:0; font-size:22px; }
    #progress { color:#cdd8ea; font-weight:600; }
    main { max-width:1080px; margin:24px auto; padding:0 18px 40px; }
    .card { background:white; border-radius:18px; box-shadow:0 12px 35px #23324d1a; overflow:hidden; }
    .topline { padding:18px 22px; display:flex; align-items:center; gap:12px; border-bottom:1px solid #e5eaf2; }
    .pill { border-radius:999px; padding:6px 11px; font-weight:700; font-size:13px; }
    .green { background:#dff6e8; color:#126238; } .yellow { background:#fff1bd; color:#775600; } .red { background:#ffe0e2; color:#8d1e2a; }
    .crop { background:#202838; min-height:220px; display:flex; align-items:center; justify-content:center; padding:18px; }
    .crop img { max-width:100%; max-height:520px; image-rendering:auto; border-radius:8px; }
    .content { padding:22px; display:grid; gap:16px; }
    .compare { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    .box { padding:14px; border:1px solid #dbe2ee; border-radius:12px; background:#f9fbfe; }
    .label { font-size:12px; color:#60708a; font-weight:700; text-transform:uppercase; letter-spacing:.04em; margin-bottom:7px; }
    .value { font-size:22px; overflow-wrap:anywhere; white-space:pre-wrap; }
    textarea { width:100%; min-height:86px; resize:vertical; border:2px solid #aebbd0; border-radius:12px; padding:12px; font:600 21px/1.35 "Segoe UI",Arial,sans-serif; }
    textarea:focus { outline:3px solid #90c2ff66; border-color:#1477df; }
    .warning { background:#fff7dc; color:#614b00; padding:12px 14px; border-radius:10px; display:none; }
    .reasons { color:#52627a; margin:0; padding-left:20px; }
    .bulk { display:flex; gap:14px; align-items:center; justify-content:space-between; padding:15px; border:2px solid #9ec7ef; border-radius:13px; background:#eef7ff; }
    .bulk-copy { color:#264a70; line-height:1.4; }
    .bulk-copy strong { display:block; color:#173b61; margin-bottom:3px; }
    .actions { display:flex; flex-wrap:wrap; gap:10px; padding-top:4px; }
    button { border:0; border-radius:10px; padding:12px 17px; font-weight:700; font-size:15px; cursor:pointer; }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .keep { background:#e7edf6; color:#253752; } .replace { background:#1477df; color:white; } .skip { background:#fff0f0; color:#942934; }
    .bulk-keep { background:#155fa0; color:white; flex:0 0 auto; }
    .nav { margin-left:auto; background:white; color:#33435c; border:1px solid #cbd5e4; }
    .finish { background:#198754; color:white; }
    #message { min-height:24px; color:#9a2c35; font-weight:600; }
    .empty { padding:50px 24px; text-align:center; }
    @media (max-width:720px) { .compare { grid-template-columns:1fr; } header { align-items:flex-start; flex-direction:column; } .bulk { align-items:stretch; flex-direction:column; } .nav { margin-left:0; } }
  </style>
</head>
<body>
  <header><h1>V7 – Duyệt chữ trước khi phục dựng</h1><div id="progress">Đang tải…</div></header>
  <main>
    <section class="card" id="card">
      <div class="topline"><span class="pill" id="status"></span><strong id="region"></strong></div>
      <div class="crop"><img id="crop" alt="Ảnh cắt vùng chữ cần duyệt"></div>
      <div class="content">
        <div class="compare">
          <div class="box"><div class="label">Chữ máy đọc được</div><div class="value" id="selected"></div></div>
          <div class="box"><div class="label">Đề xuất để tham khảo</div><div class="value" id="suggested"></div></div>
        </div>
        <div><div class="label">Nội dung bạn muốn dùng</div><textarea id="approved" spellcheck="true"></textarea></div>
        <div class="warning" id="warning"></div>
        <ul class="reasons" id="reasons"></ul>
        <div class="bulk">
          <div class="bulk-copy"><strong>Nếu các vùng còn lại đã đúng trên ảnh gốc</strong>Giữ nguyên bitmap, không áp dụng chữ máy đề xuất.</div>
          <button class="bulk-keep" id="bulkKeep" disabled>Giữ nguyên tất cả vùng còn lại</button>
        </div>
        <div id="message"></div>
        <div class="actions">
          <button class="keep" id="keep">Giữ nguyên vùng ảnh</button>
          <button class="replace" id="replace">Thay bằng nội dung trên</button>
          <button class="skip" id="skip">Bỏ qua vùng này</button>
          <button class="nav" id="previous">← Vùng trước</button>
          <button class="nav" id="next">Vùng sau →</button>
          <button class="finish" id="finish">Lưu và hoàn tất</button>
        </div>
      </div>
    </section>
    <section class="card empty" id="empty" hidden><h2>Không có vùng chữ để duyệt</h2><p>Bạn có thể đóng trang này.</p></section>
  </main>
  <script nonce="__NONCE__">
    const TOKEN = __TOKEN_JSON__;
    let state = null, index = 0, busy = false;
    const $ = id => document.getElementById(id);
    const escapeText = value => String(value ?? "");
    async function api(path, body) {
      let response;
      try {
        response = await fetch(path, {
          method: body === undefined ? "GET" : "POST",
          headers: body === undefined ? {"X-V7-Review-Token": TOKEN} : {"Content-Type":"application/json", "X-V7-Review-Token":TOKEN},
          body: body === undefined ? undefined : JSON.stringify(body),
          cache: "no-store"
        });
      } catch (networkError) {
        const error = new Error("Mất kết nối với V7. Các nút duyệt từng vùng được lưu ngay khi bấm; hãy mở lại lệnh Duyệt để kiểm tra phần còn lại.");
        error.cause = networkError;
        throw error;
      }
      const payload = await response.json();
      if (!response.ok) { const error = new Error(payload.message || "Không lưu được quyết định."); error.payload = payload; throw error; }
      return payload;
    }
    function render() {
      const rows = state.regions;
      $("empty").hidden = rows.length !== 0; $("card").hidden = rows.length === 0;
      $("progress").textContent = `${state.summary.decided}/${state.summary.total} vùng đã quyết định · còn ${state.summary.pending}`;
      if (!rows.length) return;
      index = Math.max(0, Math.min(index, rows.length - 1));
      const row = rows[index];
      $("status").className = `pill ${row.status}`; $("status").textContent = row.status_label;
      $("region").textContent = `Vùng ${index + 1}/${rows.length}`;
      $("crop").src = row.crop_url; $("selected").textContent = escapeText(row.selected_text); $("suggested").textContent = escapeText(row.suggested_text);
      $("approved").value = escapeText(row.input_text);
      const warning = $("warning");
      if (row.protected_values.length) { warning.style.display="block"; warning.textContent=`Không tự sửa số/giá/mã: ${row.protected_values.join(", ")}. Nếu cần đổi, hệ thống sẽ hỏi xác nhận lần nữa.`; }
      else warning.style.display="none";
      $("reasons").replaceChildren(...row.reasons.map(value => { const li=document.createElement("li"); li.textContent=value; return li; }));
      $("message").textContent = row.action === "pending" ? "Chưa có quyết định cho vùng này." : `Đã lưu: ${row.action === "replace" ? "Thay bằng" : row.action === "keep" ? "Giữ nguyên" : "Bỏ qua"}.`;
      $("previous").disabled = index === 0; $("next").disabled = index === rows.length - 1;
      $("bulkKeep").disabled = state.summary.pending === 0;
    }
    async function decide(action, confirmed=false) {
      if (busy) return; busy=true; const row=state.regions[index]; $("message").textContent="Đang lưu…";
      try {
        state = await api("/api/decision", {region_fingerprint:row.region_fingerprint, action, approved_text:$("approved").value, confirm_protected_change:confirmed});
        const nextPending = state.regions.findIndex((item, position) => position > index && item.action === "pending");
        if (nextPending >= 0) index=nextPending; render();
      } catch (error) {
        if (error.payload?.code === "protected-change-confirmation-required" && !confirmed) {
          const ok = window.confirm(`${error.message}\n\nChỉ bấm OK nếu bạn đã kiểm tra trực tiếp trên ảnh.`);
          if (ok) { busy=false; return decide(action, true); }
        }
        $("message").textContent=error.message;
      } finally { busy=false; }
    }
    $("keep").onclick=()=>decide("keep"); $("replace").onclick=()=>decide("replace"); $("skip").onclick=()=>decide("skip");
    $("bulkKeep").onclick=async()=>{
      if (busy || !state.summary.pending) return;
      const count=state.summary.pending;
      const ok=window.confirm(`Giữ nguyên ${count} vùng còn lại?\n\nKhông chữ nào được thay thế. Các vùng này sẽ giữ nguyên bitmap gốc, kể cả số, giá và mã hàng.`);
      if (!ok) return;
      busy=true; $("message").textContent="Đang lưu tất cả vùng còn lại…";
      try {
        state=await api("/api/bulk-keep-pending", {action:"keep"});
        render(); $("message").textContent=`Đã giữ nguyên ${count} vùng còn lại. Không có chữ nào bị thay thế.`;
      } catch(error) { $("message").textContent=error.message; }
      finally { busy=false; }
    };
    $("previous").onclick=()=>{index--;render();}; $("next").onclick=()=>{index++;render();};
    $("finish").onclick=async()=>{ if(state.summary.pending && !window.confirm(`Còn ${state.summary.pending} vùng chưa duyệt. Vẫn lưu và dừng?`)) return; try { await api("/api/finish",{}); $("message").textContent="Đã lưu. Bạn có thể đóng trang này."; } catch(error) { $("message").textContent=error.message; } };
    api("/api/state").then(value=>{state=value; const pending=state.regions.findIndex(row=>row.action==="pending"); index=pending<0?0:pending; render();}).catch(error=>{$("message").textContent=error.message;});
  </script>
</body>
</html>"""


def render_review_html(token: str) -> tuple[str, str]:
    """Return HTML and the one-use CSP nonce used by the inline UI assets."""

    nonce = secrets.token_urlsafe(18)
    html = _HTML_TEMPLATE.replace("__NONCE__", nonce).replace(
        "__TOKEN_JSON__", json.dumps(str(token), ensure_ascii=False)
    )
    return html, nonce


class _ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], session: ReviewSession) -> None:
        self.session = session
        super().__init__(address, _ReviewRequestHandler)

    @property
    def origin(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


class _ReviewRequestHandler(BaseHTTPRequestHandler):
    server: _ReviewHTTPServer
    _shutdown_after_response = False

    def finish(self) -> None:
        """Flush the final response before stopping the one-shot review UI."""

        shutdown = self._shutdown_after_response
        try:
            super().finish()
        finally:
            if shutdown:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, _format: str, *args: object) -> None:
        return

    def _secure_headers(self, *, content_type: str, length: int, nonce: str | None = None) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        if nonce:
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; "
                f"style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'; "
                "img-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'none'",
            )

    def _host_is_loopback(self) -> bool:
        expected = f"{LOOPBACK_HOST}:{self.server.server_address[1]}"
        return secrets.compare_digest(self.headers.get("Host", ""), expected)

    def _authorized(self, parsed: Any, *, mutation: bool = False) -> bool:
        if not self._host_is_loopback():
            return False
        header = self.headers.get("X-V7-Review-Token", "")
        query = parse_qs(parsed.query).get("token", [""])[0]
        supplied = header if mutation or header else query
        return bool(supplied) and secrets.compare_digest(supplied, self.server.session.token)

    def _json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._secure_headers(content_type="application/json; charset=utf-8", length=len(data))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: HTTPStatus, code: str, message: str, **extra: object) -> None:
        self._json(status, {"ok": False, "code": code, "message": message, **extra})

    def _body(self) -> dict[str, object]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ReviewValidationError("Thiếu Content-Length.")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ReviewValidationError("Content-Length không hợp lệ.") from exc
        if not 0 <= length <= MAX_REQUEST_BYTES:
            raise ReviewValidationError("Yêu cầu gửi lên quá lớn.")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReviewValidationError("Body phải là JSON UTF-8 hợp lệ.") from exc
        if not isinstance(value, dict):
            raise ReviewValidationError("Body phải là JSON object.")
        return value

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not self._authorized(parsed):
            self._error(HTTPStatus.FORBIDDEN, "forbidden", "Liên kết duyệt không hợp lệ.")
            return
        if parsed.path == "/":
            html, nonce = render_review_html(self.server.session.token)
            data = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self._secure_headers(content_type="text/html; charset=utf-8", length=len(data), nonce=nonce)
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/state":
            self._json(HTTPStatus.OK, self.server.session.public_state())
            return
        match = re.fullmatch(r"/crop/([0-9a-f]{64})\.png", parsed.path)
        if match:
            try:
                data = self.server.session.crop_png(match.group(1))
            except ReviewUIError as exc:
                self._error(HTTPStatus.NOT_FOUND, "crop-not-found", str(exc))
                return
            self.send_response(HTTPStatus.OK)
            self._secure_headers(content_type="image/png", length=len(data))
            self.end_headers()
            self.wfile.write(data)
            return
        self._error(HTTPStatus.NOT_FOUND, "not-found", "Không tìm thấy trang.")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not self._authorized(parsed, mutation=True):
            self._error(HTTPStatus.FORBIDDEN, "forbidden", "Phiên duyệt không hợp lệ.")
            return
        if self.headers.get_content_type() != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "invalid-content-type", "Chỉ nhận JSON.")
            return
        try:
            body = self._body()
            if parsed.path == "/api/decision":
                state = self.server.session.apply_decision(
                    str(body.get("region_fingerprint", "")),
                    str(body.get("action", "")),
                    approved_text=body.get("approved_text", ""),
                    confirm_protected_change=body.get("confirm_protected_change") is True,
                )
                self._json(HTTPStatus.OK, state)
                return
            if parsed.path == "/api/bulk-keep-pending":
                if set(body) != {"action"} or body.get("action") != "keep":
                    raise ReviewValidationError(
                        "Thao tác hàng loạt chỉ nhận Giữ nguyên các vùng còn pending."
                    )
                state = self.server.session.keep_all_pending()
                self._json(HTTPStatus.OK, state)
                return
            if parsed.path == "/api/finish":
                state = self.server.session.mark_finished()
                self._json(HTTPStatus.OK, state)
                self._shutdown_after_response = True
                return
            self._error(HTTPStatus.NOT_FOUND, "not-found", "Không tìm thấy API.")
        except ProtectedChangeConfirmationRequired as exc:
            self._error(
                HTTPStatus.CONFLICT,
                "protected-change-confirmation-required",
                str(exc),
                protected_values=exc.protected_values,
            )
        except ReviewConflictError as exc:
            self._error(HTTPStatus.CONFLICT, "file-conflict", str(exc))
        except ReviewValidationError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid-decision", str(exc))
        except OSError:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "save-failed", "Không ghi được file duyệt.")

    def do_OPTIONS(self) -> None:  # noqa: N802
        # Deliberately no CORS support. A foreign page cannot submit the custom
        # token header because its browser preflight is rejected.
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method-not-allowed", "Không hỗ trợ CORS.")


@dataclass(slots=True)
class ReviewServerHandle:
    server: _ReviewHTTPServer
    session: ReviewSession
    url: str

    def serve_forever(self) -> None:
        self.server.serve_forever(poll_interval=0.20)

    def close(self) -> None:
        self.server.server_close()


def create_review_server(
    review_path: Path | str,
    *,
    image_path: Path | str | None = None,
    host: str = LOOPBACK_HOST,
    port: int = 0,
) -> ReviewServerHandle:
    """Create, but do not start, the token-protected loopback review server."""

    if host != LOOPBACK_HOST:
        raise ReviewPathError("V7 review chỉ được bind vào 127.0.0.1.")
    try:
        selected_port = int(port)
    except (TypeError, ValueError) as exc:
        raise ReviewValidationError("Port không hợp lệ.") from exc
    if not 0 <= selected_port <= 65_535:
        raise ReviewValidationError("Port phải từ 0 đến 65535.")
    session = ReviewSession.open(review_path, image_path=image_path)
    server = _ReviewHTTPServer((host, selected_port), session)
    url = f"{server.origin}/?token={quote(session.token)}"
    return ReviewServerHandle(server=server, session=session, url=url)


def run_review_ui(
    review_path: Path | str,
    *,
    image_path: Path | str | None = None,
    port: int = 0,
    open_browser: Callable[[str], object] | None = None,
) -> dict[str, object]:
    """Open the local page and block until the user chooses “Lưu và hoàn tất”."""

    handle = create_review_server(review_path, image_path=image_path, port=port)
    opener = open_browser or webbrowser.open
    try:
        opener(handle.url)
        handle.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        handle.close()
    return copy.deepcopy(handle.session.document)


__all__ = [
    "LOOPBACK_HOST",
    "ProtectedChangeConfirmationRequired",
    "ReviewConflictError",
    "ReviewPathError",
    "ReviewServerHandle",
    "ReviewSession",
    "ReviewUIError",
    "ReviewValidationError",
    "create_review_server",
    "render_review_html",
    "run_review_ui",
    "validate_review_document",
]
