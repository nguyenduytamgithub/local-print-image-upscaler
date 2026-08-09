from __future__ import annotations

import math
import unicodedata
from typing import Any

import cv2
import numpy as np

from .schema import DocumentGraph, ElementNode


TEXT_EXPORT_PURITY_POLICY = "exact_export_text_purity_v1"
_ALPHA_PROOF_THRESHOLD = 8
_MIN_CORE_RETENTION = 0.80
_EDGE_COLOUR_DISTANCE_LIMIT = 24.0


def _luminance(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float32)
    return (
        value[:, :, 0] * 0.2126
        + value[:, :, 1] * 0.7152
        + value[:, :, 2] * 0.0722
    )


def _row_bands(binary: np.ndarray) -> list[tuple[int, int, int]]:
    """Return separated alpha bands as ``(top, bottom, pixel_count)``.

    A short vertical closing joins Vietnamese tone marks to their base line,
    while a remote decoration at an OCR-box edge remains a separate band.
    """

    height = int(binary.shape[0])
    if not height or not np.any(binary):
        return []
    present = np.any(binary, axis=1).astype(np.uint8).reshape(-1, 1)
    merge_gap = max(1, min(5, int(round(height * 0.055))))
    kernel = np.ones((merge_gap * 2 + 1, 1), dtype=np.uint8)
    closed = cv2.morphologyEx(present, cv2.MORPH_CLOSE, kernel).ravel() > 0
    bands: list[tuple[int, int, int]] = []
    start: int | None = None
    for row, active in enumerate(np.r_[closed, False]):
        if active and start is None:
            start = row
        elif not active and start is not None:
            bands.append((start, row, int(np.count_nonzero(binary[start:row]))))
            start = None
    return bands


def _ink_unit_count(text: str | None) -> int:
    """Count visible base characters without charging combining tone marks."""

    if not text:
        return 0
    result = 0
    for character in unicodedata.normalize("NFD", str(text)):
        if character.isspace() or unicodedata.combining(character):
            continue
        if unicodedata.category(character).startswith("C"):
            continue
        result += 1
    return result


def _outside_ring_luminance(
    source_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float | None:
    height, width = source_rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    padding = max(3, min(12, int(round(max(x1 - x0, y1 - y0) * 0.15))))
    left = max(0, x0 - padding)
    top = max(0, y0 - padding)
    right = min(width, x1 + padding)
    bottom = min(height, y1 + padding)
    ring = np.ones((bottom - top, right - left), dtype=bool)
    ring[y0 - top : y1 - top, x0 - left : x1 - left] = False
    if not np.any(ring):
        return None
    values = _luminance(source_rgb[top:bottom, left:right])[ring]
    if not values.size:
        return None
    return float(np.median(values))


def _foreground_core_report(
    source_crop: np.ndarray,
    binary: np.ndarray,
    *,
    background_luminance: float | None,
    ink_units: int,
) -> dict[str, Any]:
    """Prove that exact exported alpha still owns decisive source ink cores.

    This deliberately uses only very dark ink against a proven light ring (or
    very light ink against a proven dark ring).  Mid-tone artwork is not
    guessed.  Absence of decisive evidence therefore does not invent a fail;
    it leaves the topology checks to decide automatic safety.
    """

    polarity = "not_decisive"
    threshold: float | None = None
    core = np.zeros(binary.shape, dtype=bool)
    if background_luminance is not None and background_luminance >= 160.0:
        polarity = "dark_on_light"
        threshold = max(24.0, min(96.0, background_luminance * 0.26))
        core = _luminance(source_crop) <= threshold
    elif background_luminance is not None and background_luminance <= 95.0:
        polarity = "light_on_dark"
        threshold = min(
            231.0,
            max(159.0, background_luminance + (255.0 - background_luminance) * 0.74),
        )
        core = _luminance(source_crop) >= threshold

    candidate_pixels = int(np.count_nonzero(core))
    retained_pixels = int(np.count_nonzero(core & binary))
    minimum_evidence_pixels = max(8, min(32, max(1, ink_units) * 2))
    evaluated = candidate_pixels >= minimum_evidence_pixels
    retention = retained_pixels / max(1, candidate_pixels)
    return {
        "background_luminance_median": (
            round(background_luminance, 4)
            if background_luminance is not None
            else None
        ),
        "polarity": polarity,
        "core_luminance_threshold": round(threshold, 4) if threshold is not None else None,
        "candidate_pixels": candidate_pixels,
        "retained_pixels": retained_pixels,
        "retained_fraction": round(retention, 6),
        "minimum_evidence_pixels": minimum_evidence_pixels,
        "minimum_retained_fraction": _MIN_CORE_RETENTION,
        "evaluated": evaluated,
        "passed": bool(not evaluated or retention >= _MIN_CORE_RETENTION),
    }


def _edge_band_report(
    source_crop: np.ndarray,
    binary: np.ndarray,
) -> dict[str, Any]:
    """Find chromatically incompatible fragments outside the glyph row."""

    height, width = binary.shape
    total = int(np.count_nonzero(binary))
    bands = _row_bands(binary)
    primary = max(bands, key=lambda item: item[2]) if bands else None
    records: list[dict[str, Any]] = []
    if primary is not None:
        primary_top, primary_bottom, primary_pixels = primary
        margin = max(1, min(3, int(round(height * 0.04))))
        expanded_top = max(0, primary_top - margin)
        expanded_bottom = min(height, primary_bottom + margin)
        primary_mask = binary.copy()
        primary_mask[:expanded_top] = False
        primary_mask[expanded_bottom:] = False
        source_lab = cv2.cvtColor(source_crop, cv2.COLOR_RGB2LAB).astype(np.float32)
        primary_colours = source_lab[primary_mask]
        if primary_colours.size:
            primary_median = np.median(primary_colours, axis=0)
            primary_columns = np.where(np.any(primary_mask, axis=0))[0]
            primary_left = int(primary_columns.min())
            primary_right = int(primary_columns.max()) + 1
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                binary.astype(np.uint8), 8
            )
            area_floor = max(4, int(math.ceil(total * 0.012)))
            primary_height = max(1, primary_bottom - primary_top)
            gap_floor = max(3, int(round(primary_height * 0.18)))
            horizontal_gap_floor = max(3, int(round(width * 0.04)))
            edge_tolerance = 1
            for component in range(1, count):
                x, y, component_width, component_height, area = (
                    int(value) for value in stats[component]
                )
                bottom = y + component_height
                right = x + component_width
                if area < area_floor:
                    continue
                if y < expanded_bottom and bottom > expanded_top:
                    continue
                gap = (
                    expanded_top - bottom
                    if bottom <= expanded_top
                    else y - expanded_bottom
                )
                touches_left = x <= edge_tolerance
                touches_top = y <= edge_tolerance
                touches_right = right >= width - edge_tolerance
                touches_bottom = bottom >= height - edge_tolerance
                edge_contacts = [
                    side
                    for side, touched in (
                        ("left", touches_left),
                        ("top", touches_top),
                        ("right", touches_right),
                        ("bottom", touches_bottom),
                    )
                    if touched
                ]
                edge_contact_axis_count = int(touches_left or touches_right) + int(
                    touches_top or touches_bottom
                )
                horizontal_gap = (
                    primary_left - right
                    if right <= primary_left
                    else x - primary_right
                    if x >= primary_right
                    else 0
                )
                if (
                    not edge_contacts
                    or gap < gap_floor
                    or horizontal_gap < horizontal_gap_floor
                ):
                    continue
                component_colours = source_lab[labels == component]
                if not component_colours.size:
                    continue
                component_median = np.median(component_colours, axis=0)
                colour_distance = float(np.linalg.norm(component_median - primary_median))
                if colour_distance < _EDGE_COLOUR_DISTANCE_LIMIT:
                    continue
                records.append(
                    {
                        "bbox": [x, y, right, bottom],
                        "pixels": area,
                        "fraction_of_export_alpha": round(area / max(1, total), 6),
                        "gap_from_primary_band_px": gap,
                        "horizontal_gap_from_primary_px": horizontal_gap,
                        "horizontal_gap_floor_px": horizontal_gap_floor,
                        "touches_crop_edge": True,
                        "edge_contacts": edge_contacts,
                        "edge_contact_axis_count": edge_contact_axis_count,
                        "source_lab_distance_from_primary": round(colour_distance, 4),
                    }
                )
        primary_record: list[int] | None = [
            int(primary_top),
            int(primary_bottom),
            int(primary_pixels),
        ]
    else:
        primary_record = None
    return {
        "row_bands": [[int(a), int(b), int(c)] for a, b, c in bands],
        "primary_band": primary_record,
        "colour_distance_limit": _EDGE_COLOUR_DISTANCE_LIMIT,
        "isolated_edge_components": records,
        "passed": not records,
    }


def exact_export_text_purity_report(
    node: ElementNode,
    source_rgb: np.ndarray,
) -> dict[str, Any]:
    """Assess the exact alpha/RGB evidence that PSD/ORA/PNG will receive."""

    if node.kind not in {"text", "price"}:
        raise ValueError("Exact text export purity only accepts text/price nodes.")
    image = np.asarray(source_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Text export purity source must be uint8 RGB.")
    support = node.full_support or node.visible_alpha
    height, width = image.shape[:2]
    support.canvas_slice((width, height))
    x0, y0, x1, y1 = support.bbox
    binary = support.alpha >= _ALPHA_PROOF_THRESHOLD
    source_crop = image[y0:y1, x0:x1]
    text = node.text
    if not text:
        extraction = node.metadata.get("text_extraction")
        if isinstance(extraction, dict):
            raw_text = extraction.get("recognized_text")
            text = str(raw_text) if raw_text is not None else None
    ink_units = _ink_unit_count(text)

    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8
    )
    component_areas = [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, component_count)
    ]
    component_total = len(component_areas)
    median_component_area = (
        float(np.median(component_areas)) if component_areas else 0.0
    )
    micro_count = sum(area <= 4 for area in component_areas)
    micro_fraction = micro_count / max(1, component_total)
    component_limit = max(12, ink_units * 3 + 2) if ink_units else 0

    background_luminance = _outside_ring_luminance(image, support.bbox)
    core = _foreground_core_report(
        source_crop,
        binary,
        background_luminance=background_luminance,
        ink_units=ink_units,
    )
    edge_bands = _edge_band_report(source_crop, binary)
    fragmented_topology = bool(
        ink_units
        and component_total > component_limit
        and (micro_fraction >= 0.20 or median_component_area <= 8.0)
        and (
            not core["evaluated"]
            or float(core["retained_fraction"]) < 0.70
        )
    )

    reasons: list[str] = []
    if not edge_bands["passed"]:
        reasons.append("isolated_out_of_primary_band_edge_component")
    if core["evaluated"] and not core["passed"]:
        reasons.append("foreground_core_not_retained")
    if fragmented_topology:
        reasons.append("implausible_fragmented_glyph_topology")
    if not np.any(binary):
        reasons.append("empty_exact_export_text_alpha")

    return {
        "policy": TEXT_EXPORT_PURITY_POLICY,
        "status": "pass" if not reasons else "unsafe",
        "reasons": reasons,
        "support_source": "full_support" if node.full_support is not None else "visible_alpha",
        "alpha_proof_threshold": _ALPHA_PROOF_THRESHOLD,
        "support_domain": "exact_source_resolution_export_support",
        "support_domain_note": (
            "Support membership is exact before export; a fixed-alpha render solver may "
            "promote alpha values without adding or removing support pixels."
        ),
        "recognized_text": text,
        "recognized_ink_unit_count": ink_units,
        "foreground_core": core,
        "edge_band_evidence": edge_bands,
        "topology": {
            "component_count": component_total,
            "component_limit_for_recognized_text": component_limit,
            "components_per_ink_unit": round(
                component_total / max(1, ink_units), 6
            ),
            "median_component_area": round(median_component_area, 4),
            "micro_component_count": micro_count,
            "micro_component_fraction": round(micro_fraction, 6),
            "implausibly_fragmented": fragmented_topology,
        },
        "exact_export_alpha_pixels": int(np.count_nonzero(binary)),
    }


def _append_review_reason(existing: str | None, addition: str) -> str:
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing}; {addition}"


def _reason_list(value: Any) -> list[str]:
    """Return a stable JSON-safe reason list from persisted gate metadata."""

    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item) for item in value))


def _persisted_first_downgrade(value: Any) -> dict[str, Any] | None:
    """Validate the immutable first-downgrade audit snapshot.

    The export preflight intentionally runs several times around review and
    ownership reconciliation.  A later idempotent pass sees an already
    unresolved node, so transition history cannot be reconstructed from the
    current status.  Persisting this small snapshot on the node makes the
    final graph/manifest truthful without relying on call order.
    """

    if not isinstance(value, dict):
        return None
    reasons = _reason_list(value.get("reasons"))
    if value.get("policy") != TEXT_EXPORT_PURITY_POLICY or not reasons:
        return None
    return {
        "policy": TEXT_EXPORT_PURITY_POLICY,
        "purity_status": str(value.get("purity_status") or "unsafe"),
        "reasons": reasons,
        "review_status_before_gate": str(
            value.get("review_status_before_gate") or "unresolved"
        ),
        "move_safe_before_gate": bool(value.get("move_safe_before_gate", False)),
        "review_status_after_gate": "unresolved",
        "move_safe_after_gate": False,
    }


def enforce_text_export_purity_preflight(
    graph: DocumentGraph,
    source_rgb: np.ndarray,
) -> dict[str, Any]:
    """Fail closed on impure exact text supports before checkpoint/export.

    The preflight never repairs, filters or redraws a glyph.  It records why
    automatic safety is not proven and keeps the observed candidate available
    for human review.
    """

    image = np.asarray(source_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Text export purity source must be uint8 RGB.")
    if graph.canvas_size != image.shape[1::-1]:
        raise ValueError("Text export purity canvas differs from source image.")
    graph.validate()

    ledger_key = "text_export_purity_by_owner"
    for proposal in graph.proposals:
        if isinstance(proposal.evidence, dict):
            proposal.evidence.pop(ledger_key, None)

    owner_proposals: dict[str, list[Any]] = {}
    for proposal in graph.proposals:
        for owner_id in proposal.owner_ids:
            owner_proposals.setdefault(owner_id, []).append(proposal)

    records: list[dict[str, Any]] = []
    downgraded_ids: list[str] = []
    failed_ids: list[str] = []
    first_downgrade_by_node: dict[str, dict[str, Any]] = {}
    for node in sorted(graph.nodes, key=lambda item: (item.z_index, item.element_id)):
        if node.kind not in {"text", "price"} or node.review_status == "rejected":
            continue
        previous_gate = node.metadata.get("text_export_purity_preflight")
        before_status = node.review_status
        before_move_safe = bool(node.move_safe)
        previous_belongs_to_node = bool(
            isinstance(previous_gate, dict)
            and previous_gate.get("node_id") == node.element_id
            and "review_status_before_first_gate" in previous_gate
        )
        if previous_belongs_to_node:
            first_status = str(
                previous_gate.get("review_status_before_first_gate", before_status)
            )
            first_move_safe = bool(
                previous_gate.get("move_safe_before_first_gate", before_move_safe)
            )
            first_gate_status = str(
                previous_gate.get("purity_status_at_first_gate") or "unsafe"
            )
            first_gate_reasons = _reason_list(
                previous_gate.get("reasons_at_first_gate")
            )
            first_downgrade = _persisted_first_downgrade(
                previous_gate.get("first_downgrade")
            )
        else:
            first_status = before_status
            first_move_safe = before_move_safe
            first_gate_status = ""
            first_gate_reasons = []
            first_downgrade = None

        report = exact_export_text_purity_report(node, image)
        if not first_gate_status:
            first_gate_status = str(report["status"])
            first_gate_reasons = list(report["reasons"])
        unsafe = report["status"] != "pass"
        claimed_automatic_safety_now = (
            before_status == "auto_confirmed" or before_move_safe
        )
        if unsafe:
            failed_ids.append(node.element_id)
            node.review_status = "unresolved"
            node.move_safe = False
            if first_downgrade is None and claimed_automatic_safety_now:
                first_downgrade = {
                    "policy": TEXT_EXPORT_PURITY_POLICY,
                    "purity_status": str(report["status"]),
                    "reasons": list(report["reasons"]),
                    "review_status_before_gate": before_status,
                    "move_safe_before_gate": before_move_safe,
                    "review_status_after_gate": "unresolved",
                    "move_safe_after_gate": False,
                }
            previous_review_reason = node.metadata.get("review_reason")
            node.metadata["review_reason"] = _append_review_reason(
                previous_review_reason
                if isinstance(previous_review_reason, str)
                else None,
                (
                    "Mask chữ xuất cuối chưa giữ sạch và đủ lõi chữ gốc; "
                    "cần kiểm tra thủ công, không được tự xác nhận an toàn."
                ),
            )
            if claimed_automatic_safety_now:
                action = "downgraded_to_unresolved_non_move_safe"
            elif first_downgrade is not None:
                action = "retained_prior_downgrade_unresolved_non_move_safe"
            else:
                action = "retained_unresolved_non_move_safe"
        elif first_downgrade is not None:
            action = "purity_passed_status_retained_after_prior_downgrade"
        else:
            action = "purity_passed_status_retained"

        if first_downgrade is not None:
            downgraded_ids.append(node.element_id)
            first_downgrade_by_node[node.element_id] = dict(first_downgrade)

        gate_record = {
            **report,
            "node_id": node.element_id,
            "review_status_before_first_gate": first_status,
            "move_safe_before_first_gate": first_move_safe,
            "purity_status_at_first_gate": first_gate_status,
            "reasons_at_first_gate": first_gate_reasons,
            "review_status_before_this_gate": before_status,
            "move_safe_before_this_gate": before_move_safe,
            "review_status_after_gate": node.review_status,
            "move_safe_after_gate": bool(node.move_safe),
            "ever_downgraded_by_gate": first_downgrade is not None,
            "first_downgrade": first_downgrade,
            "action": action,
        }
        node.metadata["text_export_purity_preflight"] = gate_record

        ledger_record = {
            "policy": TEXT_EXPORT_PURITY_POLICY,
            "status": report["status"],
            "reasons": list(report["reasons"]),
            "review_status_before_first_gate": first_status,
            "move_safe_before_first_gate": first_move_safe,
            "review_status_after_gate": node.review_status,
            "move_safe_after_gate": bool(node.move_safe),
            "ever_downgraded_by_gate": first_downgrade is not None,
            "first_downgrade": first_downgrade,
            "action": action,
        }
        for proposal in owner_proposals.get(node.element_id, []):
            by_owner = proposal.evidence.get(ledger_key)
            if not isinstance(by_owner, dict):
                by_owner = {}
            by_owner[node.element_id] = dict(ledger_record)
            proposal.evidence[ledger_key] = {
                key: by_owner[key] for key in sorted(by_owner)
            }

        records.append(
            {
                "node_id": node.element_id,
                "status": report["status"],
                "reasons": list(report["reasons"]),
                "review_status_before_first_gate": first_status,
                "move_safe_before_first_gate": first_move_safe,
                "purity_status_at_first_gate": first_gate_status,
                "reasons_at_first_gate": first_gate_reasons,
                "review_status_before_this_gate": before_status,
                "move_safe_before_this_gate": before_move_safe,
                "review_status_after_gate": node.review_status,
                "move_safe_after_gate": bool(node.move_safe),
                "ever_downgraded_by_gate": first_downgrade is not None,
                "first_downgrade": first_downgrade,
                "action": action,
                "foreground_core": report["foreground_core"],
                "topology": report["topology"],
                "isolated_edge_components": report["edge_band_evidence"][
                    "isolated_edge_components"
                ],
            }
        )

    summary = {
        "policy": TEXT_EXPORT_PURITY_POLICY,
        "status": "review_required" if failed_ids else "pass",
        "checked_node_count": len(records),
        "failed_node_count": len(failed_ids),
        "failed_node_ids": sorted(failed_ids),
        "downgraded_node_ids": sorted(set(downgraded_ids)),
        "ever_downgraded_node_ids": sorted(set(downgraded_ids)),
        "first_downgrade_by_node": {
            key: first_downgrade_by_node[key]
            for key in sorted(first_downgrade_by_node)
        },
        "records": records,
        "enforcement": (
            "Exact exported text alpha is never filtered or redrawn here. "
            "A failed automatic candidate remains available only as unresolved/non-move-safe."
        ),
    }
    graph.metadata["text_export_purity_preflight"] = summary
    graph.validate()
    return summary
