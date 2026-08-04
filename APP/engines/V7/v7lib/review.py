"""Human-in-the-loop review files and a small native Windows review dialog."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .types import TextRegion


REVIEW_SCHEMA = "local-print-image-upscaler/v7-text-review/1"


def _proposal_text(region: TextRegion) -> str:
    proposal = region.proposal
    if proposal is None:
        return region.selected_text
    return proposal.proposed_text


def region_fingerprint(region: TextRegion | Mapping[str, object]) -> str:
    """Bind a review decision to one OCR geometry/text observation.

    Region ids are intentionally presentation-only: OCR can renumber them when
    another pass inserts a box earlier in reading order.  The review authority
    therefore uses a deterministic digest of the source-space box and the NFC
    OCR text that the reviewer actually saw.
    """

    if isinstance(region, TextRegion):
        bbox_raw: object = region.bbox
        selected_raw: object = region.selected_text
    else:
        bbox_raw = region.get("bbox")
        selected_raw = region.get("selected_text", "")
    if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
        raise ValueError("Review region bbox must contain four integer coordinates.")
    try:
        bbox = [int(value) for value in bbox_raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("Review region bbox must contain four integer coordinates.") from exc
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("Review region bbox is invalid.")
    selected = unicodedata.normalize("NFC", str(selected_raw))
    payload = json.dumps(
        {"bbox": bbox, "selected_text": selected},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"V7_REVIEW_REGION_V1\0" + payload).hexdigest()


def build_review_document(
    source: Path,
    regions: list[TextRegion],
    *,
    ocr_report: dict[str, object] | None = None,
    review_mode: str = "auto",
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for region in regions:
        suggested = _proposal_text(region)
        default_action = (
            "replace" if review_mode != "strict" and region.status == "green" else "pending"
        )
        rows.append(
            {
                **region.to_dict(),
                "region_fingerprint": region_fingerprint(region),
                "suggested_text": suggested,
                "action": default_action,
                "approved_text": region.selected_text if default_action == "replace" else "",
                "instructions": "replace | keep | skip; enter exact approved_text for replace",
            }
        )
    return {
        "schema": REVIEW_SCHEMA,
        "source": str(Path(source).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "green": "independent OCR agreement; rebuild allowed when no language correction is proposed",
            "yellow": "readable but ambiguous/corrected/protected; user approval required",
            "red": "unreadable; user must type the intended text or keep the original pixels",
            "critical": "prices, digits, phone numbers, SKUs and similar fields are never silently changed",
        },
        "ocr": ocr_report or {},
        "regions": rows,
    }


def write_review_document(document: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_review_document(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != REVIEW_SCHEMA or not isinstance(value.get("regions"), list):
        raise ValueError(f"Invalid V7 review file: {path}")
    return value


def resolve_review(
    regions: list[TextRegion],
    document: dict[str, object],
) -> tuple[dict[str, str], set[str], list[str]]:
    """Return replacements, kept region ids and unresolved region ids."""

    known = {item.region_id: item for item in regions}
    replacements: dict[str, str] = {}
    kept: set[str] = set()
    unresolved: list[str] = []
    seen: set[str] = set()
    for row in document.get("regions", []):
        if not isinstance(row, dict):
            continue
        region_id = str(row.get("region_id", ""))
        if region_id not in known or region_id in seen:
            continue
        seen.add(region_id)
        action = str(row.get("action", "pending")).lower().strip()
        if action == "replace":
            text = unicodedata.normalize("NFC", str(row.get("approved_text", "")).strip())
            if text:
                replacements[region_id] = text
            else:
                unresolved.append(region_id)
        elif action in {"keep", "skip"}:
            kept.add(region_id)
        else:
            unresolved.append(region_id)
    unresolved.extend(sorted(set(known) - seen))
    return replacements, kept, sorted(set(unresolved))


def review_in_tk(
    image_path: Path,
    document: dict[str, object],
    output_path: Path,
) -> dict[str, object]:
    """Review ambiguous regions without requiring a browser or cloud service."""

    import tkinter as tk
    from tkinter import messagebox, ttk

    from PIL import ImageTk

    rows = [row for row in document.get("regions", []) if isinstance(row, dict)]
    pending = [row for row in rows if row.get("action") == "pending"]
    if not pending:
        write_review_document(document, output_path)
        return document
    source = Image.open(image_path).convert("RGB")
    root = tk.Tk()
    root.title("V7 – duyệt chữ trước khi phục dựng")
    root.geometry("980x680")
    state = {"index": 0, "photo": None}

    title = ttk.Label(root, text="", font=("Segoe UI", 13, "bold"))
    title.pack(fill="x", padx=16, pady=(14, 6))
    preview = ttk.Label(root)
    preview.pack(fill="both", expand=True, padx=16, pady=6)
    ttk.Label(root, text="Chữ OCR / đề xuất (hãy sửa thành nội dung đúng):").pack(
        anchor="w", padx=16, pady=(8, 2)
    )
    entry_var = tk.StringVar()
    entry = ttk.Entry(root, textvariable=entry_var, font=("Segoe UI", 16))
    entry.pack(fill="x", padx=16, pady=(0, 10))
    reason = ttk.Label(root, text="", wraplength=930)
    reason.pack(fill="x", padx=16, pady=(0, 10))
    controls = ttk.Frame(root)
    controls.pack(fill="x", padx=16, pady=(0, 16))

    def show() -> None:
        row = pending[state["index"]]
        x0, y0, x1, y1 = (int(v) for v in row["bbox"])
        pad = max(12, int((y1 - y0) * 0.7))
        crop = source.crop(
            (max(0, x0 - pad), max(0, y0 - pad), min(source.width, x1 + pad), min(source.height, y1 + pad))
        )
        crop.thumbnail((920, 420), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(crop)
        state["photo"] = photo
        preview.configure(image=photo)
        title.configure(
            text=f"Vùng {state['index'] + 1}/{len(pending)} – {row.get('status', '').upper()}"
        )
        entry_var.set(str(row.get("suggested_text") or row.get("selected_text") or ""))
        reason.configure(text="; ".join(str(v) for v in row.get("reasons", [])))
        entry.focus_set()
        entry.selection_range(0, "end")

    def advance() -> None:
        if state["index"] + 1 >= len(pending):
            write_review_document(document, output_path)
            root.destroy()
        else:
            state["index"] += 1
            show()

    def replace() -> None:
        value = unicodedata.normalize("NFC", entry_var.get().strip())
        if not value:
            messagebox.showerror("Thiếu nội dung", "Hãy nhập chữ đúng hoặc chọn giữ nguyên.")
            return
        pending[state["index"]]["action"] = "replace"
        pending[state["index"]]["approved_text"] = value
        advance()

    def keep() -> None:
        pending[state["index"]]["action"] = "keep"
        pending[state["index"]]["approved_text"] = ""
        advance()

    def cancel() -> None:
        if messagebox.askyesno("Dừng duyệt", "Lưu phần đã duyệt và để phần còn lại chờ xử lý?"):
            write_review_document(document, output_path)
            root.destroy()

    ttk.Button(controls, text="Vẽ lại chữ này", command=replace).pack(side="left")
    ttk.Button(controls, text="Giữ nguyên vùng ảnh", command=keep).pack(side="left", padx=8)
    ttk.Button(controls, text="Lưu và dừng", command=cancel).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", cancel)
    show()
    root.mainloop()
    return load_review_document(output_path)
