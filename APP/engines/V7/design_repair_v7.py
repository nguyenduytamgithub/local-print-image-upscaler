"""V7: reviewable Vietnamese poster text restoration.

The core contract is deliberately stricter than a normal super-resolution
script: OCR and a spelling model may propose text, but only independently
agreed OCR or an explicit review decision is rendered.  Approved old glyphs are
removed at source resolution, the bounded background footprint is reconstructed,
and clean Unicode text is rendered once at the requested final size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

from raster_restore import (
    TRUTH_NOTICE as RASTER_TRUTH_NOTICE,
    RasterRestoreConfig,
    SuperResolutionBackend,
    V3SubprocessBackend,
    restore_raster,
)
from user_review import ReviewUIError, run_review_ui
from v7lib.background import (
    BackgroundRestoreError,
    BackgroundRestoreResult,
    restore_text_background,
)
from v7lib.formats import (
    EditableText,
    build_bundle_manifest,
    export_editable_svg,
    save_png_atomic,
    sha256_file,
    write_manifest_atomic,
)
from v7lib.masks import TextMaskResult, build_text_mask
from v7lib.ocr import OCRPipeline, clean_ocr_text
from v7lib.qa import (
    QAThresholds,
    evaluate_replacement,
    make_determinism_metadata,
    save_qa_artifacts,
)
from v7lib.review import (
    build_review_document,
    load_review_document,
    region_fingerprint,
    resolve_review,
    write_review_document,
)
from v7lib.textnorm import build_conservative_proposal, normalize_nfc
from v7lib.types import LanguageProposal, TextRegion
from v7lib.typography import (
    FontMatch,
    FontRecord,
    TextRender,
    TextStyle,
    TypographyError,
    composite_text,
    discover_windows_fonts,
    estimate_text_style,
    font_variation_axes,
    render_text_layer,
    select_font,
)


Image.MAX_IMAGE_PIXELS = 500_000_000
MIN_SCALE = 1.0
MAX_SCALE = 20.0
LANGUAGE_REVISION = "61596a71696ba360ae828f9db3806610afedf6d3"


def open_review_browser(url: str) -> bool:
    """Open the V7 review page in Chrome when it is installed.

    This is intentionally local-only: ``run_review_ui`` binds to 127.0.0.1 and
    protects the session with a random token.  The generic browser fallback is
    retained for machines that do not have Chrome.
    """

    candidates: list[Path] = []
    for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if base:
            candidates.append(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe")
    for command_name in ("chrome.exe", "chrome"):
        located = shutil.which(command_name)
        if located:
            candidates.append(Path(located))
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        if not candidate.is_file():
            continue
        try:
            subprocess.Popen([str(candidate.resolve()), "--new-window", url])
            return True
        except OSError:
            continue
    opened = bool(webbrowser.open(url, new=1))
    if not opened:
        print(f"  Không tự mở được trình duyệt; hãy mở liên kết cục bộ này: {url}")
    return opened


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7 reviewable design/text restoration")
    parser.add_argument("input", type=Path)
    parser.add_argument("scale", type=float)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--tessdata-dir", type=Path)
    parser.add_argument("--review-mode", choices=("gui", "defer", "auto", "strict"), default="gui")
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--ocr-passes", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--inpaint", choices=("auto", "poster", "opencv", "strict"), default="auto")
    parser.add_argument(
        "--raster-restore",
        choices=("auto", "off", "strong"),
        default="auto",
        help="PRINT_FAITHFUL clean-raster restoration; OCR still uses the original source.",
    )
    parser.add_argument("--no-language-model", action="store_true")
    parser.add_argument("--v3-python", type=Path)
    parser.add_argument("--v3-engine", type=Path)
    parser.add_argument("--language-python", type=Path)
    parser.add_argument("--app-version", default="dev")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, str]:
    source = args.input.resolve()
    output = args.output_dir.resolve()
    if not source.is_file():
        raise SystemExit(f"Input not found: {source}")
    if not MIN_SCALE <= args.scale <= MAX_SCALE or (1.0 < args.scale < 2.0):
        raise SystemExit("V7 scale must be x1 or from x2 through x20.")
    if output.exists():
        raise SystemExit(
            f"Output exists: {output}. V7 refuses to remove it; use the unified launcher "
            "for atomic replacement."
        )
    output.mkdir(parents=True)
    return source, output, args.name or source.stem


def load_source(path: Path) -> tuple[Image.Image, bytes | None, tuple[float, float] | None]:
    with Image.open(path) as opened:
        opened.load()
        image = ImageOps.exif_transpose(opened).convert("RGB")
        profile = opened.info.get("icc_profile")
        dpi_raw = opened.info.get("dpi")
    dpi: tuple[float, float] | None = None
    if isinstance(dpi_raw, tuple) and len(dpi_raw) >= 2:
        values = float(dpi_raw[0]), float(dpi_raw[1])
        if all(math.isfinite(item) and 1 <= item <= 10_000 for item in values):
            dpi = values
    return image, bytes(profile) if profile else None, dpi


def srgb_profile(source_profile: bytes | None) -> bytes:
    if source_profile:
        return source_profile
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _language_snapshot(models_root: Path) -> Path | None:
    snapshots = (
        models_root
        / "huggingface"
        / "models--nrl-ai--vn-spell-correction-base"
        / "snapshots"
    )
    exact = snapshots / LANGUAGE_REVISION
    if all((exact / name).is_file() for name in ("config.json", "model.safetensors", "tokenizer.json")):
        return exact
    return None


def _case_pattern(source: str, candidate: str) -> str:
    letters = [character for character in source if character.isalpha()]
    if letters and all(character.isupper() for character in letters):
        return candidate.upper()
    if source.istitle():
        return candidate.title()
    return candidate


def _model_input(value: str) -> str:
    letters = [character for character in value if character.isalpha()]
    if letters and sum(character.isupper() for character in letters) / len(letters) >= 0.72:
        return value.lower()
    return value


def _context_groups(regions: list[TextRegion]) -> list[list[TextRegion]]:
    groups: list[list[TextRegion]] = []
    current: list[TextRegion] = []
    length = 0
    for region in regions:
        text = region.selected_text or ""
        extra = len(text) + (1 if current else 0)
        if current and (len(current) >= 8 or length + extra > 190):
            groups.append(current)
            current, length = [], 0
        current.append(region)
        length += extra
    if current:
        groups.append(current)
    return groups


def language_proposals(
    regions: list[TextRegion],
    *,
    python: Path | None,
    worker: Path,
    model_dir: Path | None,
    enabled: bool,
) -> dict[str, object]:
    readable = [region for region in regions if (region.selected_text or "").strip()]
    if not enabled:
        return {"enabled": False, "reason": "disabled_by_user", "proposal_count": 0}
    if python is None or not python.is_file() or model_dir is None:
        for region in readable:
            region.reasons.append("local_language_model_unavailable")
        return {
            "enabled": True,
            "available": False,
            "reason": "missing_runtime_or_revision_pinned_model",
            "proposal_count": 0,
        }

    requests: list[dict[str, object]] = []
    groups = _context_groups(readable)
    # Small posters benefit from an individual vote as well as context. Dense
    # catalogues use bounded context chunks to avoid hundreds of GPU calls.
    if len(readable) <= 12:
        for region in readable:
            requests.append(
                {
                    "id": f"individual:{region.region_id}",
                    "op": "propose",
                    "text": _model_input(region.selected_text or ""),
                }
            )
    for index, group in enumerate(groups):
        context = " ".join(region.selected_text or "" for region in group)
        requests.append(
            {
                "id": f"context:{index}",
                "op": "propose",
                "text": _model_input(context),
            }
        )
    payload = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests)
    command = [
        str(python),
        "-B",
        str(worker),
        "--model-dir",
        str(model_dir),
        "--device",
        "auto",
        "--max-length",
        "256",
    ]
    try:
        completed = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        for region in readable:
            region.reasons.append("language_worker_failed")
        return {
            "enabled": True,
            "available": False,
            "error_type": type(exc).__name__,
            "proposal_count": 0,
        }
    responses: dict[str, dict[str, object]] = {}
    for line in completed.stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("id") is not None:
            responses[str(item["id"])] = item
    individual: dict[str, LanguageProposal] = {}
    for region in readable:
        response = responses.get(f"individual:{region.region_id}", {})
        if response.get("ok") and isinstance(response.get("proposal"), dict):
            raw = LanguageProposal.from_dict(response["proposal"])  # type: ignore[arg-type]
            candidate = _case_pattern(region.selected_text or "", raw.proposed_text)
            individual[region.region_id] = build_conservative_proposal(
                region.selected_text or "",
                candidate,
                confidence=raw.confidence,
                reasons=(*raw.reasons, "individual-language-pass"),
                model=raw.model,
            )

    contextual: dict[str, LanguageProposal] = {}
    for index, group in enumerate(groups):
        response = responses.get(f"context:{index}", {})
        if not response.get("ok") or not isinstance(response.get("proposal"), dict):
            continue
        raw = LanguageProposal.from_dict(response["proposal"])  # type: ignore[arg-type]
        original_counts = [len((region.selected_text or "").split()) for region in group]
        proposed_words = raw.proposed_text.split()
        if sum(original_counts) != len(proposed_words):
            for region in group:
                region.reasons.append("context_proposal_could_not_be_aligned")
            continue
        cursor = 0
        for region, count in zip(group, original_counts, strict=True):
            candidate = " ".join(proposed_words[cursor : cursor + count])
            cursor += count
            candidate = _case_pattern(region.selected_text or "", candidate)
            contextual[region.region_id] = build_conservative_proposal(
                region.selected_text or "",
                candidate,
                confidence=raw.confidence,
                reasons=(*raw.reasons, "context-language-pass"),
                model=raw.model,
            )

    changed = 0
    conflicts = 0
    for region in readable:
        first = individual.get(region.region_id)
        second = contextual.get(region.region_id)
        chosen = second or first
        if first and second and first.proposed_text != second.proposed_text:
            conflicts += 1
            region.reasons.append("language_proposals_disagree")
            # Context is useful for Vietnamese accents, but disagreement is a
            # yellow review condition regardless of confidence.
            chosen = second
        region.proposal = chosen
        if chosen and chosen.changed:
            changed += 1
            region.status = "yellow"
            region.reasons.append("language_model_change_requires_approval")
        if chosen and not chosen.safe:
            region.status = "yellow"
            region.reasons.append("unsafe_language_proposal_discarded")
    return {
        "enabled": True,
        "available": completed.returncode == 0,
        "worker_return_code": completed.returncode,
        "model": f"nrl-ai/vn-spell-correction-base@{LANGUAGE_REVISION}",
        "device_policy": "local CUDA when available, otherwise CPU",
        "request_count": len(requests),
        "proposal_count": sum(region.proposal is not None for region in readable),
        "changed_proposal_count": changed,
        "conflict_count": conflicts,
        "stderr_tail": completed.stderr.splitlines()[-8:],
        "policy": "proposals only; every content change requires approval",
    }


def merge_review_document(
    current: dict[str, object],
    reviewed: dict[str, object],
    *,
    source_sha256: str,
) -> None:
    reviewed_sha = str(reviewed.get("source_sha256", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", reviewed_sha):
        raise RuntimeError("The supplied TEXT_REVIEW.json has no valid source SHA-256.")
    if reviewed_sha != source_sha256.lower():
        raise RuntimeError("The supplied TEXT_REVIEW.json belongs to a different source image.")

    decisions: dict[str, dict[str, object]] = {}
    reviewed_rows = reviewed.get("regions", [])
    if not isinstance(reviewed_rows, list):
        raise RuntimeError("The supplied TEXT_REVIEW.json has an invalid regions list.")
    for row in reviewed_rows:
        if not isinstance(row, dict):
            raise RuntimeError("The supplied TEXT_REVIEW.json contains an invalid region row.")
        fingerprint = str(row.get("region_fingerprint", "")).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise RuntimeError("The supplied TEXT_REVIEW.json is missing a valid region fingerprint.")
        try:
            expected = region_fingerprint(row)
        except ValueError as exc:
            raise RuntimeError("The supplied TEXT_REVIEW.json contains invalid region geometry.") from exc
        if fingerprint != expected:
            raise RuntimeError(
                "The supplied TEXT_REVIEW.json region geometry/OCR text was changed after review."
            )
        if fingerprint in decisions:
            raise RuntimeError("The supplied TEXT_REVIEW.json contains duplicate region fingerprints.")
        decisions[fingerprint] = row

    consumed: set[str] = set()
    current_fingerprints: set[str] = set()
    for row in current.get("regions", []):
        if not isinstance(row, dict):
            continue
        fingerprint = region_fingerprint(row)
        if fingerprint in current_fingerprints:
            raise RuntimeError("Current OCR produced duplicate review fingerprints; regenerate review.")
        current_fingerprints.add(fingerprint)
        decision = decisions.get(fingerprint)
        if decision:
            row["action"] = decision.get("action", "pending")
            row["approved_text"] = decision.get("approved_text", "")
            consumed.add(fingerprint)

    stale_decisions = [
        fingerprint
        for fingerprint, row in decisions.items()
        if fingerprint not in consumed
        and str(row.get("action", "pending")).strip().lower() in {"replace", "keep", "skip"}
    ]
    if stale_decisions:
        raise RuntimeError(
            "OCR geometry/text changed after approval; regenerate TEXT_REVIEW.json before applying decisions."
        )


def _row_by_id(document: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(row.get("region_id")): row
        for row in document.get("regions", [])
        if isinstance(row, dict) and row.get("region_id")
    }


def _expanded_box(
    bbox: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
    *,
    scale: float = 1.0,
) -> tuple[int, int, int, int]:
    width, height = canvas_size
    x0, y0, x1, y1 = (int(round(value * scale)) for value in bbox)
    line_height = max(1, y1 - y0)
    line_width = max(1, x1 - x0)
    pad_x = max(1, int(round(min(line_height * 0.10, line_width * 0.04))))
    pad_y = max(1, int(round(line_height * 0.12)))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def _font_preferences(text: str, bbox: tuple[int, int, int, int]) -> tuple[list[str], str]:
    letters = [character for character in text if character.isalpha()]
    uppercase = bool(letters) and sum(character.isupper() for character in letters) / len(letters) > 0.72
    height = bbox[3] - bbox[1]
    if uppercase and height >= 24:
        return ["Arial", "Bahnschrift", "Arial Narrow", "Segoe UI", "Tahoma", "Impact"], "bold"
    return ["Arial", "Bahnschrift", "Segoe UI", "Tahoma", "Times New Roman"], "regular"


def _tight_mask_bbox(
    mask: np.ndarray,
    fallback: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Use the old-glyph pixels when an OCR detector returned a loose rectangle."""

    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if not len(xs):
        return fallback
    tight = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    # A tiny accidental component must not collapse an otherwise useful OCR box.
    if (tight[2] - tight[0]) < 3 or (tight[3] - tight[1]) < 3:
        return fallback
    return tight


def _select_font(
    text: str,
    bbox: tuple[int, int, int, int],
    fonts: tuple[FontRecord, ...],
    *,
    stroke_width_px: int = 0,
) -> tuple[FontMatch, dict[str, object]]:
    """Choose a covered font that also matches the old text footprint geometry.

    Family-name ranking alone consistently picked Arial for condensed poster
    headlines.  Here each preferred family is shaped with RAQM/HarfBuzz in the
    actual target box.  The font whose visible ink best fills both dimensions is
    selected deterministically; this recovers a condensed face when the source
    glyph footprint is tall and narrow.
    """

    families, style = _font_preferences(text, bbox)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    candidates: list[tuple[float, int, FontMatch, dict[str, object]]] = []
    seen: set[str] = set()
    for rank, family in enumerate(families):
        try:
            match = select_font(
                text,
                fonts,
                preferred_families=[family],
                preferred_style=style,
                require_full_coverage=True,
                require_embeddable=False,
            )
        except TypographyError:
            continue
        wanted = family.casefold()
        actual = match.font.family.casefold()
        if not (actual == wanted or wanted in actual or actual in wanted):
            continue
        if match.font.identifier in seen:
            continue
        seen.add(match.font.identifier)
        definitions = font_variation_axes(match.font)
        axis_options: list[tuple[float, ...] | None] = [None]
        if definitions:
            base_axes = [item.default for item in definitions]
            weight_index = next(
                (index for index, item in enumerate(definitions) if item.tag == "wght"),
                None,
            )
            width_index = next(
                (index for index, item in enumerate(definitions) if item.tag == "wdth"),
                None,
            )
            if weight_index is not None:
                weight = definitions[weight_index]
                base_axes[weight_index] = (
                    min(weight.maximum, max(weight.minimum, 700.0))
                    if style == "bold"
                    else weight.default
                )
            if width_index is not None:
                width_axis = definitions[width_index]
                width_values = np.linspace(width_axis.minimum, width_axis.maximum, 11)
                axis_options = []
                for width_value in width_values:
                    values = base_axes.copy()
                    values[width_index] = float(width_value)
                    axis_options.append(tuple(values))
            else:
                axis_options = [tuple(base_axes)]
        for axes in axis_options:
            probe_style = TextStyle(
                font=match.font,
                stroke_width_px=max(0, int(stroke_width_px)),
                horizontal_align="center",
                vertical_align="middle",
                language="vi",
                direction="ltr",
                variation_axes=axes,
            )
            try:
                probe = render_text_layer(
                    text,
                    (0, 0, width, height),
                    probe_style,
                    canvas_size=(width, height),
                    padding_px=0,
                )
            except TypographyError:
                continue
            visible_width = probe.visible_bbox[2] - probe.visible_bbox[0]
            visible_height = probe.visible_bbox[3] - probe.visible_bbox[1]
            width_fill = visible_width / width
            height_fill = visible_height / height
            balanced_fill = min(width_fill, height_fill)
            area_fill = width_fill * height_fill
            geometry_score = balanced_fill * 100.0 + area_fill * 10.0 - rank * 0.05
            axis_report = (
                {
                    definition.tag: round(float(value), 4)
                    for definition, value in zip(definitions, axes, strict=True)
                }
                if axes is not None
                else {}
            )
            candidates.append(
                (
                    geometry_score,
                    -rank,
                    match,
                    {
                        "family": match.font.family,
                        "subfamily": match.font.subfamily,
                        "variation_axes": axis_report,
                        "font_size_px": probe.font_size_px,
                        "visible_size": [visible_width, visible_height],
                        "target_size": [width, height],
                        "width_fill": round(width_fill, 6),
                        "height_fill": round(height_fill, 6),
                        "balanced_fill": round(balanced_fill, 6),
                        "score": round(geometry_score, 6),
                    },
                )
            )
    if candidates:
        selected = max(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
                item[2].font.family.casefold(),
                item[2].font.subfamily.casefold(),
            ),
        )
        return selected[2], {
            "policy": (
                "RAQM-shaped balanced footprint fill across preferred families and "
                "OpenType fvar width/weight grid"
            ),
            "selected": selected[3],
            "candidates": [item[3] for item in sorted(candidates, key=lambda item: item[0], reverse=True)],
        }
    fallback = select_font(
        text,
        fonts,
        preferred_families=families,
        preferred_style=style,
        require_full_coverage=True,
        require_embeddable=False,
    )
    return fallback, {
        "policy": "coverage/style fallback; no preferred family completed the geometry probe",
        "selected": {"family": fallback.font.family, "subfamily": fallback.font.subfamily},
        "candidates": [],
    }


def _render_alpha_to_canvas(rendered: TextRender, canvas_size: tuple[int, int]) -> np.ndarray:
    width, height = canvas_size
    result = np.zeros((height, width), dtype=bool)
    alpha = np.asarray(rendered.rgba.getchannel("A"), dtype=np.uint8) > 0
    x, y = rendered.position
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    copy_w = min(alpha.shape[1] - sx0, width - dx0)
    copy_h = min(alpha.shape[0] - sy0, height - dy0)
    if copy_w > 0 and copy_h > 0:
        result[dy0 : dy0 + copy_h, dx0 : dx0 + copy_w] = alpha[
            sy0 : sy0 + copy_h, sx0 : sx0 + copy_w
        ]
    return result


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _boundary(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(bool)


def render_regions(
    base: Image.Image,
    source_rgb: np.ndarray,
    regions: list[TextRegion],
    replacements: dict[str, str],
    masks: dict[str, TextMaskResult],
    fonts: tuple[FontRecord, ...],
    *,
    scale: float,
    typography_locks: dict[str, TextRender] | None = None,
) -> tuple[Image.Image, np.ndarray, list[dict[str, object]], list[TextRender]]:
    expected_ids = set(replacements)
    if typography_locks is not None and set(typography_locks) != expected_ids:
        raise RuntimeError("Final typography locks do not match the approved replacement regions.")
    result = base.convert("RGBA")
    canvas_size = result.size
    alpha_union = np.zeros((canvas_size[1], canvas_size[0]), dtype=bool)
    records: list[dict[str, object]] = []
    renders: list[TextRender] = []
    for region in regions:
        approved = replacements.get(region.region_id)
        if not approved:
            continue
        mask_result = masks[region.region_id]
        effective_bbox = _tight_mask_bbox(mask_result.mask, region.bbox)
        estimate = estimate_text_style(source_rgb, mask_result.mask, bbox=region.bbox)
        box = _expanded_box(effective_bbox, canvas_size, scale=scale)
        geometry_box = tuple(int(round(value * scale)) for value in effective_bbox)
        locked = typography_locks.get(region.region_id) if typography_locks is not None else None
        lock_report: dict[str, object]
        if locked is not None:
            if normalize_nfc(locked.source_text) != normalize_nfc(approved):
                raise RuntimeError("A typography lock was applied to different approved text.")
            font_record = locked.font
            selected_axis_map = locked.report.get("variation_axes", {})
            axis_definitions = font_variation_axes(font_record)
            if axis_definitions and (
                not isinstance(selected_axis_map, dict)
                or not all(item.tag in selected_axis_map for item in axis_definitions)
            ):
                raise RuntimeError("A variable-font typography lock is missing one or more axes.")
            variation_axes = (
                tuple(float(selected_axis_map[item.tag]) for item in axis_definitions)
                if axis_definitions and isinstance(selected_axis_map, dict)
                else None
            )
            geometry_font_size = max(1, int(round(locked.font_size_px * scale)))
            fill_rgb = locked.fill_rgb
            stroke_rgb = locked.stroke_rgb
            scaled_stroke = max(0, int(round(locked.stroke_width_px * scale)))
            font_geometry = {
                "policy": "locked from the source-scale typography gate",
                "selected": {
                    "family": font_record.family,
                    "subfamily": font_record.subfamily,
                    "variation_axes": selected_axis_map if isinstance(selected_axis_map, dict) else {},
                    "font_size_px": geometry_font_size,
                },
                "candidates": [],
            }
            lock_report = {
                "locked": True,
                "source_font_identifier": font_record.identifier,
                "source_font_size_px": locked.font_size_px,
                "scale": scale,
            }
        else:
            stroke_rgb = estimate.stroke_rgb
            source_stroke = estimate.stroke_width_source_px if stroke_rgb else 0.0
            # Mask antialias fringes can make the local poster background look
            # like a giant outline. Drop a candidate indistinguishable from its
            # outer ring and cap real outlines relative to source line height.
            outer = _dilate(mask_result.mask, 5) & ~_dilate(mask_result.mask, 1)
            if stroke_rgb is not None and outer.any():
                background_rgb = np.median(source_rgb[outer], axis=0)
                if float(np.linalg.norm(background_rgb - np.asarray(stroke_rgb))) < 22.0:
                    stroke_rgb = None
                    source_stroke = 0.0
            line_height = effective_bbox[3] - effective_bbox[1]
            source_stroke = min(source_stroke, max(1.0, line_height * 0.065))
            scaled_stroke = max(0, int(round(source_stroke * scale)))
            match, font_geometry = _select_font(
                approved,
                effective_bbox,
                fonts,
                stroke_width_px=max(0, int(round(source_stroke))),
            )
            font_record = match.font
            selected_geometry = font_geometry.get("selected", {})
            selected_axis_map = (
                selected_geometry.get("variation_axes", {})
                if isinstance(selected_geometry, dict)
                else {}
            )
            axis_definitions = font_variation_axes(font_record)
            variation_axes = (
                tuple(float(selected_axis_map[item.tag]) for item in axis_definitions)
                if axis_definitions
                and isinstance(selected_axis_map, dict)
                and all(item.tag in selected_axis_map for item in axis_definitions)
                else None
            )
            source_font_size = (
                int(selected_geometry.get("font_size_px", 0))
                if isinstance(selected_geometry, dict)
                else 0
            )
            geometry_font_size = max(1, int(round(source_font_size * scale))) if source_font_size else 0
            fill_rgb = estimate.fill_rgb
            lock_report = {"locked": False, "scale": scale}
        style = TextStyle(
            font=font_record,
            font_size_px=geometry_font_size or None,
            fill_rgb=fill_rgb,
            stroke_rgb=stroke_rgb,
            stroke_width_px=scaled_stroke,
            horizontal_align="center",
            vertical_align="middle",
            language="vi",
            direction="ltr",
            variation_axes=variation_axes,
        )
        rendered = render_text_layer(
            approved,
            box,
            style,
            canvas_size=canvas_size,
        )
        result = composite_text(result, rendered)
        alpha_union |= _render_alpha_to_canvas(rendered, canvas_size)
        renders.append(rendered)
        records.append(
            {
                "region_id": region.region_id,
                "approved_text": normalize_nfc(approved),
                "ocr_bbox": list(region.bbox),
                "source_bbox": list(effective_bbox),
                "geometry_target_bbox": list(geometry_box),
                "final_box": list(box),
                "visible_bbox": list(rendered.visible_bbox),
                "font": {
                    "family": rendered.font.family,
                    "subfamily": rendered.font.subfamily,
                    "full_name": rendered.font.full_name,
                    "path": str(rendered.font.path),
                    "face_index": rendered.font.face_index,
                    "embedding_fs_type": rendered.font.embedding.fs_type,
                    "embeddable": rendered.font.embedding.embeddable,
                    "bundled": False,
                    "variation_axes": rendered.report.get("variation_axes", {}),
                },
                "font_size_px": rendered.font_size_px,
                "fill_rgb": list(rendered.fill_rgb),
                "stroke_rgb": list(rendered.stroke_rgb) if rendered.stroke_rgb else None,
                "stroke_width_px": rendered.stroke_width_px,
                "layout_engine": rendered.layout_engine,
                "normalization": rendered.normalization,
                "mask_quality": round(mask_result.quality, 6),
                "style_confidence": round(estimate.confidence, 6),
                "style_report": estimate.report,
                "font_geometry": font_geometry,
                "typography_lock": lock_report,
                "render_report": rendered.report,
            }
        )
    return result.convert("RGB"), alpha_union, records, renders


def typography_geometry_gate(records: list[dict[str, object]]) -> dict[str, object]:
    """Fail closed when rebuilt ink no longer resembles the approved old footprint."""

    checks: list[dict[str, object]] = []
    for record in records:
        target_bbox = record.get("geometry_target_bbox", record["source_bbox"])
        tx0, ty0, tx1, ty1 = (float(value) for value in target_bbox)
        rx0, ry0, rx1, ry1 = (float(value) for value in record["visible_bbox"])
        target_width = max(1.0, tx1 - tx0)
        target_height = max(1.0, ty1 - ty0)
        render_width = max(1.0, rx1 - rx0)
        render_height = max(1.0, ry1 - ry0)
        width_ratio = render_width / target_width
        height_ratio = render_height / target_height
        target_aspect = target_width / target_height
        render_aspect = render_width / render_height
        aspect_log_error = abs(math.log(max(1e-9, render_aspect / target_aspect)))
        center_x_error = abs(((rx0 + rx1) - (tx0 + tx1)) * 0.5) / target_width
        center_y_error = abs(((ry0 + ry1) - (ty0 + ty1)) * 0.5) / target_height
        passed = bool(
            0.78 <= width_ratio <= 1.12
            and 0.78 <= height_ratio <= 1.12
            and aspect_log_error <= 0.18
            and center_x_error <= 0.10
            and center_y_error <= 0.10
        )
        checks.append(
            {
                "region_id": record["region_id"],
                "approved_text": record["approved_text"],
                "passed": passed,
                "target_ink_bbox": [tx0, ty0, tx1, ty1],
                "rendered_ink_bbox": [rx0, ry0, rx1, ry1],
                "width_ratio": round(width_ratio, 6),
                "height_ratio": round(height_ratio, 6),
                "aspect_log_error": round(aspect_log_error, 6),
                "center_x_error": round(center_x_error, 6),
                "center_y_error": round(center_y_error, 6),
                "font_geometry": record.get("font_geometry", {}),
                "status": "PASS" if passed else "REQUIRES_REVIEW",
            }
        )
    failed = [item["region_id"] for item in checks if not bool(item["passed"])]
    return {
        "passed": not failed,
        "policy": (
            "tight old-ink bbox vs RAQM-shaped new ink; no bitmap squashing; "
            "full glyph coverage remains mandatory"
        ),
        "thresholds": {
            "width_ratio": [0.78, 1.12],
            "height_ratio": [0.78, 1.12],
            "aspect_log_error_max": 0.18,
            "center_axis_error_max": 0.10,
        },
        "checks": checks,
        "requires_review_region_ids": failed,
    }


def typography_lock_fidelity(
    source_records: list[dict[str, object]],
    final_records: list[dict[str, object]],
    *,
    scale: float,
) -> dict[str, object]:
    """Prove that final text kept the source-gated face, axes and scaled size."""

    source_by_id = {str(item["region_id"]): item for item in source_records}
    final_by_id = {str(item["region_id"]): item for item in final_records}
    checks: list[dict[str, object]] = []
    for region_id in sorted(set(source_by_id) | set(final_by_id)):
        source = source_by_id.get(region_id)
        final = final_by_id.get(region_id)
        if source is None or final is None:
            checks.append({"region_id": region_id, "passed": False, "reason": "missing_lock_record"})
            continue
        source_font = source.get("font", {})
        final_font = final.get("font", {})
        source_axes = source_font.get("variation_axes", {}) if isinstance(source_font, dict) else {}
        final_axes = final_font.get("variation_axes", {}) if isinstance(final_font, dict) else {}
        expected_size = max(1, int(round(int(source["font_size_px"]) * scale)))
        same_face = bool(
            isinstance(source_font, dict)
            and isinstance(final_font, dict)
            and source_font.get("path") == final_font.get("path")
            and source_font.get("face_index") == final_font.get("face_index")
        )
        same_axes = source_axes == final_axes
        same_scaled_size = int(final["font_size_px"]) == expected_size
        passed = same_face and same_axes and same_scaled_size
        checks.append(
            {
                "region_id": region_id,
                "passed": passed,
                "same_font_face": same_face,
                "same_variation_axes": same_axes,
                "source_font_size_px": int(source["font_size_px"]),
                "expected_final_font_size_px": expected_size,
                "actual_final_font_size_px": int(final["font_size_px"]),
            }
        )
    failed = [str(item["region_id"]) for item in checks if not bool(item["passed"])]
    return {
        "passed": not failed,
        "policy": "final render must preserve the exact source-gated font face and fvar axes",
        "checks": checks,
        "requires_review_region_ids": failed,
    }


def upscale_clean_base(
    clean_source: Image.Image,
    *,
    scale: float,
    python: Path | None,
    engine: Path | None,
    icc_profile: bytes,
) -> tuple[Image.Image, dict[str, object]]:
    final_size = tuple(int(round(value * scale)) for value in clean_source.size)
    if scale == 1:
        return clean_source.copy(), {
            "pipeline": "identity_clean_source_x1",
            "device": "CPU",
            "final_size": list(final_size),
        }
    if python is None or engine is None or not python.is_file() or not engine.is_file():
        raise RuntimeError(
            "V7 x2..x20 requires the verified V3 CUDA runtime. "
            "Run V3 setup or use V7 x1 on a CPU-only machine."
        )
    with tempfile.TemporaryDirectory(prefix="v7_clean_master_") as raw:
        work = Path(raw)
        source_path = work / "clean_source.png"
        output_path = work / "clean_master.png"
        clean_source.save(source_path, format="PNG", icc_profile=icc_profile, compress_level=1)
        command = [
            str(python),
            "-B",
            str(engine),
            str(source_path),
            f"{scale:g}",
            str(output_path),
            "--tile",
            "512",
            "--overlap",
            "128",
            "--force",
        ]
        started = time.perf_counter()
        subprocess.run(command, check=True)
        with Image.open(output_path) as opened:
            opened.load()
            master = opened.convert("RGB")
        if master.size != final_size:
            raise RuntimeError(f"V3 clean master is {master.size}, expected {final_size}.")
        return master, {
            "pipeline": "V3_HIGH_on_clean_text_removed_base",
            "device": "CUDA PyTorch subprocess",
            "final_size": list(final_size),
            "seconds": round(time.perf_counter() - started, 3),
            "policy": "AI upscales only the clean raster base; approved text is rendered afterwards",
        }


def restore_clean_raster_base(
    clean_source: Image.Image,
    *,
    scale: float,
    mode: str,
    v3_python: Path | None,
    v3_engine: Path | None,
    icc_profile: bytes,
    sr_backend: SuperResolutionBackend | None = None,
) -> tuple[Image.Image, dict[str, object], list[str]]:
    """Build the final clean raster without changing OCR/source geometry.

    ``off`` preserves the previous V7 clean-base path. ``auto`` and ``strong``
    use only the fidelity-oriented Swin2SR checkpoint through a single-model V3
    subprocess; neither mode invokes the GAN/frequency-fusion master.
    """

    if mode not in {"auto", "off", "strong"}:
        raise ValueError("raster restore mode must be auto, off or strong")
    if mode == "off":
        legacy, legacy_report = upscale_clean_base(
            clean_source,
            scale=scale,
            python=v3_python,
            engine=v3_engine,
            icc_profile=icc_profile,
        )
        return legacy, {
            "schema": "local-print-image-upscaler/raster-restore-integration/1",
            "mode": "off",
            "profile": "DISABLED_LEGACY_V7_BASE",
            "status": "DISABLED",
            "legacy_upscale": legacy_report,
            "truth_notice": RASTER_TRUTH_NOTICE,
        }, []

    profile_settings: dict[str, object] = {}
    if mode == "strong":
        profile_settings.update(
            {
                "deblur_strength": 0.34,
                "deblur_iterations": 4,
                "denoise_strength": 0.68,
                "detail_strength": 0.34,
            }
        )
    config = RasterRestoreConfig(
        scale=scale,
        enable_deblur=True,
        enable_denoise=True,
        enable_detail=True,
        enable_sr=True,
        enable_sr_x1=True,
        tile=512,
        overlap=128,
        dtype="auto",
        model_key="swin2sr-fidelity",
        **profile_settings,
    )
    backend = sr_backend
    backend_setup_warning: str | None = None
    if backend is None:
        try:
            v3_root = (
                v3_engine.resolve().parent
                if v3_engine is not None
                else Path(__file__).resolve().parent.parent / "V3"
            )
            backend = V3SubprocessBackend.from_local(
                v3_root,
                mode="single",
                verify_selected_model=True,
                python=v3_python,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            backend_setup_warning = (
                f"Local PRINT_FAITHFUL raster backend unavailable ({type(exc).__name__}: {exc}); "
                "using guarded classical/Lanczos fallback."
            )
            backend = None

    restored = restore_raster(
        clean_source,
        config,
        sr_backend=backend,
        progress=lambda message: print(f"  {message}", flush=True),
    )
    result = restored.image.convert("RGB")
    expected_size = tuple(int(round(value * scale)) for value in clean_source.size)
    if result.size != expected_size:
        raise RuntimeError(
            f"Raster restoration returned {result.size}, expected {expected_size}."
        )
    report = dict(restored.report)
    report["integration"] = {
        "mode": mode,
        "profile": "PRINT_FAITHFUL",
        "model_key": "swin2sr-fidelity",
        "backend_mode": "single",
        "gan_or_three_model_fusion": False,
        "ocr_geometry_policy": "OCR and source-scale typography QA use the original source raster",
        "x1_policy": "native Swin2SR x4 prediction then guarded Lanczos back-downsample to x1",
    }
    warnings: list[str] = []
    if backend_setup_warning:
        warnings.append(backend_setup_warning)
    sr_report = report.get("super_resolution", {})
    if isinstance(sr_report, dict) and not bool(sr_report.get("accepted", False)):
        guard_report = sr_report.get("guard", {})
        guard_reasons = (
            guard_report.get("reasons", []) if isinstance(guard_report, dict) else []
        )
        guard_reason = (
            ", ".join(str(item) for item in guard_reasons)
            if isinstance(guard_reasons, list) and guard_reasons
            else None
        )
        reason = str(
            sr_report.get("error")
            or sr_report.get("reason")
            or guard_reason
            or "quality guard rejected SR"
        )
        warnings.append(
            "PRINT_FAITHFUL Swin2SR was not accepted; final raster uses the guarded "
            f"classical/Lanczos fallback. Reason: {reason}"
        )
    report["integration_warnings"] = warnings
    return result, report, warnings


def _box_iou(first: Iterable[int], second: Iterable[int]) -> float:
    ax0, ay0, ax1, ay1 = (int(value) for value in first)
    bx0, by0, bx1, by1 = (int(value) for value in second)
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0, min(ay1, by1) - max(ay0, by0)
    )
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union if union > 0 else 0.0


def ocr_backcheck(
    pipeline: OCRPipeline,
    final_path: Path,
    render_records: list[dict[str, object]],
    renders: list[TextRender],
) -> dict[str, object]:
    if not render_records:
        return {
            "performed": False,
            "passed_count": 0,
            "manual_review_count": 0,
            "reason": "no_rebuilt_text",
        }
    regions, report = pipeline.run(final_path, passes=1)
    isolated_readings: dict[str, list[str]] = {}
    isolated_reports: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="v7_ocr_backcheck_") as raw:
        temporary = Path(raw)
        for chunk_index in range(0, len(renders), 20):
            render_chunk = renders[chunk_index : chunk_index + 20]
            record_chunk = render_records[chunk_index : chunk_index + 20]
            prepared: list[tuple[dict[str, object], Image.Image]] = []
            for record, rendered in zip(record_chunk, render_chunk, strict=True):
                fill = tuple(int(value) for value in record["fill_rgb"])  # type: ignore[arg-type]
                luminance = 0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2]
                background = (255, 255, 255, 255) if luminance < 150 else (0, 0, 0, 255)
                crop = Image.new("RGBA", rendered.rgba.size, background)
                crop.alpha_composite(rendered.rgba)
                crop = crop.convert("RGB").resize(
                    (max(1, crop.width * 3), max(1, crop.height * 3)),
                    Image.Resampling.LANCZOS,
                )
                prepared.append((record, crop))
            width = max((image.width for _, image in prepared), default=1) + 40
            height = sum(image.height + 44 for _, image in prepared) + 20
            sheet = Image.new("RGB", (width, height), (127, 127, 127))
            expected_boxes: list[tuple[str, tuple[int, int, int, int]]] = []
            y = 20
            for record, image in prepared:
                x = 20
                sheet.paste(image, (x, y))
                expected_boxes.append(
                    (
                        str(record["region_id"]),
                        (x, y, x + image.width, y + image.height),
                    )
                )
                y += image.height + 44
            sheet_path = temporary / f"isolated_{chunk_index // 20:03d}.png"
            sheet.save(sheet_path, format="PNG", compress_level=1)
            isolated_regions, isolated_report = pipeline.run(sheet_path, passes=1)
            isolated_reports.append(isolated_report)
            for region_id, expected_box in expected_boxes:
                ranked = sorted(
                    (
                        (_box_iou(expected_box, region.bbox), region)
                        for region in isolated_regions
                        if _box_iou(expected_box, region.bbox) >= 0.16
                    ),
                    key=lambda item: item[0],
                    reverse=True,
                )
                isolated_readings[region_id] = [
                    clean_ocr_text(observation.text)
                    for _, region in ranked[:2]
                    for observation in region.observations
                ]
    checks: list[dict[str, object]] = []
    for record in render_records:
        target = normalize_nfc(str(record["approved_text"]))
        box = record["final_box"]
        matches = sorted(
            (
                (_box_iou(box, region.bbox), region)
                for region in regions
                if _box_iou(box, region.bbox) >= 0.20
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        readings = [
            clean_ocr_text(observation.text)
            for _, region in matches[:2]
            for observation in region.observations
        ]
        isolated = isolated_readings.get(str(record["region_id"]), [])
        final_exact = any(normalize_nfc(value) == target for value in readings)
        isolated_exact = any(normalize_nfc(value) == target for value in isolated)
        exact = isolated_exact
        checks.append(
            {
                "region_id": record["region_id"],
                "target_nfc": target,
                "final_composite_readings": readings,
                "isolated_render_readings": isolated,
                "final_composite_exact": final_exact,
                "isolated_render_exact": isolated_exact,
                "exact_readback": exact,
                "status": "PASS" if exact else "OCR_ADVISORY",
            }
        )
    return {
        "performed": True,
        "policy": (
            "OCR readback is an independent advisory, not a hard text-authority gate. "
            "The hard gate is the exact approved NFC string, verified font coverage, "
            "deterministic shaping/rendering and clipping QA. OCR can still confuse "
            "Vietnamese accents on small raster text. Full-composite OCR is reported "
            "separately because nearby rules and icons can join a line."
        ),
        "hard_gate": False,
        "ocr": report,
        "isolated_ocr_runs": isolated_reports,
        "checks": checks,
        "passed_count": sum(bool(item["exact_readback"]) for item in checks),
        "manual_review_count": sum(not bool(item["exact_readback"]) for item in checks),
        "advisory_count": sum(not bool(item["exact_readback"]) for item in checks),
    }


def make_before_after(source: Image.Image, final: Image.Image) -> Image.Image:
    left = source.copy()
    right = final.copy()
    maximum = 1_600
    target_height = min(max(left.height, right.height), 900)
    left.thumbnail((maximum // 2, target_height), Image.Resampling.LANCZOS)
    right.thumbnail((maximum // 2, target_height), Image.Resampling.LANCZOS)
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width + 8, height), (35, 35, 35))
    canvas.paste(left, (0, (height - left.height) // 2))
    canvas.paste(right, (left.width + 8, (height - right.height) // 2))
    return canvas


def main() -> int:
    args = parse_args()
    source_path, output_dir, name = validate_args(args)
    started = time.perf_counter()
    source_image, source_icc, dpi = load_source(source_path)
    source_rgb = np.asarray(source_image, dtype=np.uint8).copy()
    source_hash = sha256_file(source_path)
    output_icc = srgb_profile(source_icc)
    final_size = tuple(int(round(value * args.scale)) for value in source_image.size)
    tag = f"{args.scale:g}".replace(".", "p")

    source_asset = output_dir / f"{name}_SOURCE_NORMALIZED.png"
    save_png_atomic(source_image, source_asset, icc_profile=output_icc, dpi=dpi)

    print("  V7 bước 1/6: PP-OCRv6 đa lượt + Tesseract tiếng Việt đối chứng...", flush=True)
    ocr = OCRPipeline(
        args.models_root.resolve(),
        tessdata_dir=args.tessdata_dir.resolve() if args.tessdata_dir else None,
    )
    regions, ocr_report = ocr.run(source_asset, passes=args.ocr_passes)
    no_text_detected = len(regions) == 0

    print("  V7 bước 2/6: tạo đề xuất chính tả cục bộ, không tự áp dụng...", flush=True)
    language_report = language_proposals(
        regions,
        python=args.language_python.resolve() if args.language_python else None,
        worker=Path(__file__).resolve().with_name("language_worker.py"),
        model_dir=_language_snapshot(args.models_root.resolve()),
        enabled=not args.no_language_model,
    )

    review_path = output_dir / "TEXT_REVIEW.json"
    review_document = build_review_document(
        source_asset,
        regions,
        ocr_report=ocr_report,
        review_mode=args.review_mode,
    )
    review_document["source"] = source_asset.name
    review_document["source_sha256"] = source_hash
    review_document["source_size"] = list(source_image.size)
    review_document["language"] = language_report
    if args.review_file:
        merge_review_document(
            review_document,
            load_review_document(args.review_file.resolve()),
            source_sha256=source_hash,
        )
    write_review_document(review_document, review_path)
    if args.review_mode == "gui" and not args.review_file:
        print(
            "  Mở trang duyệt chữ đơn giản trong Chrome; file JSON kỹ thuật được giữ ở phía sau...",
            flush=True,
        )
        try:
            review_document = run_review_ui(
                review_path,
                image_path=source_asset,
                open_browser=open_review_browser,
            )
        except ReviewUIError as exc:
            raise RuntimeError(f"Không mở được giao diện duyệt chữ V7: {exc}") from exc
    replacements, kept, unresolved = resolve_review(regions, review_document)

    print("  V7 bước 3/6: tách đúng nét chữ cũ và kiểm tra mask...", flush=True)
    mask_results: dict[str, TextMaskResult] = {}
    rejected_masks: list[str] = []
    rows = _row_by_id(review_document)
    for region in regions:
        if region.region_id not in replacements:
            continue
        result = build_text_mask(source_rgb, region.bbox)
        mask_results[region.region_id] = result
        if result.quality < 0.45:
            rejected_masks.append(region.region_id)
            replacements.pop(region.region_id, None)
            unresolved.append(region.region_id)
            region.reasons.append("old_text_mask_failed_quality_gate")
            row = rows.get(region.region_id)
            if row:
                row["action"] = "pending"
                row["approved_text"] = ""
                row["mask_report"] = result.report
                reasons = list(row.get("reasons", []))
                reasons.append("old_text_mask_failed_quality_gate")
                row["reasons"] = sorted(set(str(item) for item in reasons))
        else:
            row = rows.get(region.region_id)
            if row:
                row["mask_report"] = result.report
    unresolved = sorted(set(unresolved))
    review_document["regions"] = list(rows.values())
    review_document["mask_gate"] = {
        "approved_mask_count": len(replacements),
        "rejected_region_ids": rejected_masks,
        "minimum_quality": 0.45,
    }
    write_review_document(review_document, review_path)

    union_old = np.zeros(source_rgb.shape[:2], dtype=bool)
    for region_id in replacements:
        union_old |= mask_results[region_id].mask
    print("  V7 bước 4/6: gỡ chữ được duyệt và dựng nền trong footprint...", flush=True)
    try:
        restored_rgb = source_rgb.copy()
        removal_footprint = np.zeros(source_rgb.shape[:2], dtype=bool)
        restoration_steps: list[dict[str, object]] = []
        confidences: list[float] = []
        methods: list[str] = []
        # Nearby lines must not become one giant inpaint component. Reconstruct
        # each approved OCR region from its own local context, in reading order.
        for region in regions:
            if region.region_id not in replacements:
                continue
            step = restore_text_background(
                restored_rgb,
                mask_results[region.region_id].mask,
                mode=args.inpaint,
            )
            restored_rgb = step.background
            removal_footprint |= step.removal_footprint
            confidences.append(step.confidence)
            methods.append(step.method)
            restoration_steps.append(
                {
                    "region_id": region.region_id,
                    "method": step.method,
                    "confidence": step.confidence,
                    "report": step.report,
                }
            )
        if not replacements:
            step = restore_text_background(restored_rgb, union_old, mode=args.inpaint)
            restored_rgb = step.background
        outside_identical = bool(
            np.array_equal(restored_rgb[~removal_footprint], source_rgb[~removal_footprint])
        )
        if not outside_identical:
            raise RuntimeError("aggregate restoration changed pixels outside approved footprints")
        restoration = BackgroundRestoreResult(
            background=restored_rgb,
            removal_footprint=removal_footprint,
            method="+".join(methods) if methods else "unchanged_empty_mask",
            confidence=min(confidences) if confidences else 1.0,
            report={
                "method": "per_text_region_sequential",
                "requested_mode": args.inpaint,
                "region_count": len(restoration_steps),
                "removal_pixels": int(removal_footprint.sum()),
                "removal_ratio": round(float(removal_footprint.mean()), 8),
                "outside_footprint_byte_identical": outside_identical,
                "deterministic": True,
                "steps": restoration_steps,
                "truth_notice": (
                    "Hidden source pixels do not exist; each approved text footprint "
                    "is reconstructed from its own visible local context."
                ),
            },
        )
    except BackgroundRestoreError as exc:
        raise RuntimeError(f"V7 background gate failed: {exc}") from exc
    clean_source_image = Image.fromarray(restoration.background, "RGB")

    fonts = discover_windows_fonts()
    if not fonts and replacements:
        raise RuntimeError("No Windows font with Vietnamese glyph coverage was found.")
    source_proof, new_alpha_source, source_render_records, source_renders = render_regions(
        clean_source_image,
        source_rgb,
        regions,
        replacements,
        mask_results,
        fonts,
        scale=1.0,
    )
    source_typography_gate = typography_geometry_gate(source_render_records)

    if replacements:
        target_text = "\u241e".join(replacements[item.region_id] for item in regions if item.region_id in replacements)
        rendered_text = "\u241e".join(
            str(record["approved_text"]) for record in source_render_records
        )
        roi = _dilate(restoration.removal_footprint | new_alpha_source, 3)
        # Intended new glyph edges are not inpaint seams. Exclude their alpha
        # support so the seam gate measures only the reconstructed background.
        seam = _boundary(restoration.removal_footprint) & ~_dilate(new_alpha_source, 3)
        determinism = make_determinism_metadata(
            source_rgb,
            np.asarray(source_proof, dtype=np.uint8),
            config={
                "pipeline": "V7_DESIGN_REPAIR",
                "regions": [
                    {"id": region_id, "text": replacements[region_id]}
                    for region_id in sorted(replacements)
                ],
                "background": restoration.report,
            },
            seed=0,
        )
        qa_result = evaluate_replacement(
            source_rgb,
            np.asarray(source_proof, dtype=np.uint8),
            roi_mask=roi,
            old_text_mask=union_old,
            new_text_alpha=new_alpha_source,
            seam_mask=seam,
            target_text=target_text,
            rendered_text=rendered_text,
            metadata=determinism,
            thresholds=QAThresholds(),
        )
        qa_report: dict[str, object] = qa_result.to_dict()
        qa_overlay = qa_result.overlay
    else:
        qa_report = {
            "passed": True,
            "status": "NOT_APPLICABLE",
            "failures": [],
            "warnings": ["No text region was approved for reconstruction."],
            "checks": {},
        }
        qa_overlay = source_rgb.copy()

    clean_base, raster_restore_report, raster_restore_warnings = restore_clean_raster_base(
        clean_source_image,
        scale=args.scale,
        mode=args.raster_restore,
        v3_python=args.v3_python.resolve() if args.v3_python else None,
        v3_engine=args.v3_engine.resolve() if args.v3_engine else None,
        icc_profile=output_icc,
    )
    if args.raster_restore == "off":
        legacy_upscale = raster_restore_report.get("legacy_upscale", {})
        upscale_report = (
            dict(legacy_upscale) if isinstance(legacy_upscale, dict) else {}
        )
    else:
        sr_summary = raster_restore_report.get("super_resolution", {})
        upscale_report = {
            "pipeline": "V7_PRINT_FAITHFUL_RASTER_RESTORE",
            "status": raster_restore_report.get("status"),
            "final_size": raster_restore_report.get("final_size", list(final_size)),
            "super_resolution": dict(sr_summary) if isinstance(sr_summary, dict) else {},
        }
    print("  V7 bước 5/6: vẽ chữ đã duyệt trực tiếp ở độ phân giải cuối...", flush=True)
    repaired, final_alpha, final_render_records, final_renders = render_regions(
        clean_base,
        source_rgb,
        regions,
        replacements,
        mask_results,
        fonts,
        scale=args.scale,
        typography_locks={
            str(record["region_id"]): rendered
            for record, rendered in zip(source_render_records, source_renders, strict=True)
        },
    )
    final_typography_gate = typography_geometry_gate(final_render_records)
    typography_fidelity = typography_lock_fidelity(
        source_render_records,
        final_render_records,
        scale=args.scale,
    )
    typography_failed_ids = sorted(
        {
            str(region_id)
            for report in (
                source_typography_gate,
                final_typography_gate,
                typography_fidelity,
            )
            for region_id in report.get("requires_review_region_ids", [])
        }
    )
    typography_gate: dict[str, object] = {
        "passed": bool(source_typography_gate.get("passed"))
        and bool(final_typography_gate.get("passed"))
        and bool(typography_fidelity.get("passed")),
        "policy": (
            "source and final geometry must pass; the final face, fvar axes and "
            "scaled font size must exactly match the source-gated render"
        ),
        "source_scale": source_typography_gate,
        "final_scale": final_typography_gate,
        "lock_fidelity": typography_fidelity,
        "requires_review_region_ids": typography_failed_ids,
    }
    clean_path = output_dir / f"{name}_CLEAN_BASE_x{tag}.png"
    final_path = output_dir / f"{name}_REPAIRED_x{tag}.png"
    overlay_path = output_dir / f"{name}_QA_OVERLAY.png"
    before_after_path = output_dir / f"{name}_BEFORE_AFTER.png"
    svg_path = output_dir / f"{name}_TEXT_EDITABLE.svg"
    save_png_atomic(clean_base, clean_path, icc_profile=output_icc, dpi=dpi)
    save_png_atomic(repaired, final_path, icc_profile=output_icc, dpi=dpi)
    save_png_atomic(qa_overlay, overlay_path, icc_profile=output_icc)
    save_png_atomic(make_before_after(source_image, repaired), before_after_path, icc_profile=output_icc)

    print("  V7 bước 6/6: OCR đọc ngược + QA + SVG chữ thật...", flush=True)
    backcheck = ocr_backcheck(ocr, final_path, final_render_records, final_renders)
    editable_texts: list[EditableText] = []
    for record, rendered in zip(final_render_records, final_renders, strict=True):
        variation_map = rendered.report.get("variation_axes", {})
        variation_svg: dict[str, str | int | float] = {}
        if isinstance(variation_map, dict) and variation_map:
            variation_svg["font-variation-settings"] = ", ".join(
                f"'{tag}' {float(value):g}"
                for tag, value in sorted(variation_map.items())
            )
            if "wdth" in variation_map:
                variation_svg["font-stretch"] = f"{float(variation_map['wdth']):g}%"
        selected_weight = (
            int(round(float(variation_map["wght"])))
            if isinstance(variation_map, dict) and "wght" in variation_map
            else (
                "bold"
                if any(
                    token in rendered.font.subfamily.casefold()
                    for token in ("bold", "black", "heavy")
                )
                else "normal"
            )
        )
        editable_texts.append(
            EditableText(
                text=str(record["approved_text"]),
                x=rendered.svg_text_origin[0],
                y=rendered.svg_text_origin[1],
                font_family=rendered.font.family,
                font_size_px=rendered.font_size_px,
                fill=rendered.fill_rgb,
                font_weight=selected_weight,
                stroke=rendered.stroke_rgb,
                stroke_width_px=rendered.stroke_width_px,
                text_anchor="start",
                region_id=str(record["region_id"]),
                review_status="approved",
                extra_attributes=variation_svg,
            )
        )
    svg_report = export_editable_svg(
        svg_path,
        final_size,
        background=clean_path,
        texts=editable_texts,
        metadata={
            "source_sha256": source_hash,
            "review_file": review_path.name,
            "raster_authority": final_path.name,
        },
    )

    sr_status_report = raster_restore_report.get("super_resolution", {})
    raster_prediction_accepted = bool(
        sr_status_report.get("accepted", False)
        if isinstance(sr_status_report, dict)
        else False
    )
    raster_restore_review_required = (
        args.raster_restore != "off" and not raster_prediction_accepted
    )
    if not bool(qa_report.get("passed", False)) or not bool(
        typography_gate.get("passed", False)
    ):
        status = "FAILED_QA"
    elif unresolved or no_text_detected or raster_restore_review_required:
        status = "REVIEW_REQUIRED"
    else:
        status = "PASS"
    ocr_advisories = (
        [
            "Independent OCR did not reproduce every approved accent exactly. "
            "The approved NFC text and font glyph coverage passed hard QA; inspect "
            "BEFORE_AFTER.png as the normal final visual proof."
        ]
        if int(backcheck.get("advisory_count", 0)) > 0
        else []
    )
    qa_document = {
        "schema": "local-print-image-upscaler/v7-qa/1",
        "status": status,
        "source_scale_hard_gates": qa_report,
        "typography_geometry_gate": typography_gate,
        "ocr_backcheck": backcheck,
        "mask_reports": {
            region_id: mask_results[region_id].report for region_id in sorted(mask_results)
        },
        "background": restoration.report,
        "raster_restore": raster_restore_report,
        "raster_restore_review_required": raster_restore_review_required,
        "upscale": upscale_report,
        "unresolved_region_ids": unresolved,
        "no_text_detected": no_text_detected,
        "kept_region_ids": sorted(kept),
        "rejected_mask_region_ids": rejected_masks,
        "final_text_alpha_pixels": int(final_alpha.sum()),
        "advisories": [*raster_restore_warnings, *ocr_advisories],
    }
    qa_path = output_dir / "QA.json"
    qa_path.write_text(json.dumps(qa_document, ensure_ascii=False, indent=2), encoding="utf-8")

    assets = [
        source_asset,
        clean_path,
        final_path,
        overlay_path,
        before_after_path,
        svg_path,
        review_path,
        qa_path,
    ]
    limitations = [
        "Pixels hidden below old text do not exist in a flat bitmap; the clean background is a bounded synthesis.",
        "The SVG contains real editable Unicode text but does not redistribute Windows font files.",
        "The repaired PNG is the raster placement authority; another machine may substitute SVG fonts.",
        RASTER_TRUTH_NOTICE,
    ]
    sr_manifest_accepted = raster_prediction_accepted
    raster_model_manifest: dict[str, object]
    if args.raster_restore == "off":
        raster_model_manifest = {
            "profile": "DISABLED_LEGACY_V7_BASE",
            "model": None,
            "backend_mode": None,
            "gan_or_three_model_fusion": False,
            "accepted": False,
        }
    else:
        raster_model_manifest = {
            "profile": "PRINT_FAITHFUL",
            "model": "Swin2SR real-world PSNR/fidelity x4",
            "backend_mode": "single",
            "gan_or_three_model_fusion": False,
            "accepted": sr_manifest_accepted,
        }
    manifest = build_bundle_manifest(
        pipeline="V7_DESIGN_REPAIR",
        app_version=args.app_version,
        source_name=source_path.name,
        source_sha256=source_hash,
        source_size=source_image.size,
        scale=args.scale,
        final_size=final_size,
        bundle_root=output_dir,
        assets=assets,
        review_required=status != "PASS",
        qa={
            "status": status,
            "report": qa_path.name,
            "ocr_advisory_count": int(backcheck.get("advisory_count", 0)),
            "raster_restore_warning_count": len(raster_restore_warnings),
            "raster_restore_status": raster_restore_report.get("status"),
            "raster_restore_review_required": raster_restore_review_required,
            "typography_review_count": len(
                typography_gate.get("requires_review_region_ids", [])
            ),
        },
        models={
            "ocr": "PaddleOCR 3.7 / PP-OCRv6 medium official local inference model",
            "ocr_vote": "Tesseract 5 Vietnamese tessdata_best when installed",
            "language": language_report,
            "raster_restore": raster_model_manifest,
        },
        limitations=limitations,
        extra={
            "status": status,
            "ocr": ocr_report,
            "language": language_report,
            "review": {
                "mode": args.review_mode,
                "approved_replacement_count": len(replacements),
                "kept_count": len(kept),
                "unresolved_count": len(unresolved),
            },
            "background": restoration.report,
            "raster_restore": raster_restore_report,
            "upscale": upscale_report,
            "typography": final_render_records,
            "svg": svg_report,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    manifest["status"] = status
    manifest["review_required"] = status != "PASS"
    manifest["final_size"] = list(final_size)
    write_manifest_atomic(manifest, output_dir / "manifest.json")

    print(f"  V7 status: {status}")
    for warning in raster_restore_warnings:
        print(f"  Raster restore warning: {warning}")
    if status == "REVIEW_REQUIRED":
        if no_text_detected:
            print("  No text was detected; V7 refuses to claim a repaired final. Inspect the source manually.")
        else:
            print("  Review/edit TEXT_REVIEW.json in the published bundle, then rerun with --review-file.")
    elif status == "FAILED_QA":
        print("  Hard QA failed; inspect QA.json in the published bundle.")
    elif int(backcheck.get("advisory_count", 0)) > 0:
        print("  OCR readback advisory: inspect BEFORE_AFTER.png; hard render QA passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
