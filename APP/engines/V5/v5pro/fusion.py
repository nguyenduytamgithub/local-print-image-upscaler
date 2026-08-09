from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from .inventory import DetectedProposal, InventoryResult
from .layerd_backend import LayerDResult, LayerDRawLayer
from .schema import AlphaCrop, Box, DocumentGraph, ElementKind, ElementNode, ProposalRecord


_PRICE_RE = re.compile(
    r"(?:\d[\d., ]{1,}|₫|đ(?:\b|/)|/gói|/thùng|/chai|/túi)",
    re.IGNORECASE,
)
_RAW_GEOMETRY_KINDS: frozenset[str] = frozenset({"panel", "frame", "line"})


@dataclass(slots=True)
class FusionResult:
    graph: DocumentGraph
    background_rgb: np.ndarray
    report: dict[str, Any]


def _intersection(first: Box, second: Box) -> int:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def _box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _semantic_ambiguity_reasons(proposal: ProposalRecord) -> list[str]:
    """Return model disagreements that forbid automatic semantic approval.

    Detector confidence is not evidence that SAM/BiRefNet agreed on the same
    object, nor that the detector label survived the proposal hand-off
    intact.  These are fail-closed review gates, independent of the later
    numeric confidence thresholds.
    """

    evidence = proposal.evidence
    reasons: list[str] = []
    refinement = evidence.get("refinement")
    if isinstance(refinement, dict) and refinement.get("accepted") is False:
        detail = str(refinement.get("reason") or "model disagreement").strip()
        reasons.append(f"semantic refinement rejected: {detail}")

    policy = evidence.get("semantic_policy")
    if isinstance(policy, dict):
        if policy.get("accepted") is False:
            details = policy.get("reasons")
            if isinstance(details, list) and details:
                reasons.append("semantic geometry policy rejected: " + "; ".join(map(str, details)))
            else:
                reasons.append("semantic geometry policy rejected")
        policy_kind = str(policy.get("kind") or "").strip()
        if policy_kind and policy_kind != proposal.kind_hint:
            reasons.append(
                f"semantic label kind {policy_kind!r} disagrees with proposal kind "
                f"{proposal.kind_hint!r}"
            )

    if evidence.get("auto_confirmable") is False and not reasons:
        reasons.append("semantic backend marked candidate as not auto-confirmable")
    if evidence.get("requires_manual_review") is True and not reasons:
        reasons.append("semantic backend requires manual review")

    # Preserve order while preventing duplicate text from repeated adapters.
    return list(dict.fromkeys(reasons))


def _record_semantic_ambiguity(
    proposal: ProposalRecord,
    reasons: list[str],
) -> None:
    if not reasons:
        return
    proposal.evidence["auto_confirmable"] = False
    proposal.evidence["requires_manual_review"] = True
    proposal.evidence["semantic_ambiguity_reasons"] = list(reasons)
    summary = "Manual semantic review required: " + "; ".join(reasons)
    if proposal.reason and summary not in proposal.reason:
        proposal.reason = f"{proposal.reason}; {summary}"
    elif not proposal.reason:
        proposal.reason = summary


def _downgrade_semantic_owner_for_review(
    owner: ElementNode,
    reasons: list[str],
) -> None:
    if not reasons:
        return
    existing = owner.metadata.get("semantic_ambiguity_reasons")
    combined = [str(item) for item in existing] if isinstance(existing, list) else []
    combined.extend(reasons)
    owner.metadata["semantic_ambiguity_reasons"] = list(dict.fromkeys(combined))
    owner.metadata["semantic_auto_confirmable"] = False
    owner.review_status = "unresolved"
    owner.move_safe = False


def _semantic_product_atomicity_policy(proposal: ProposalRecord) -> dict[str, Any]:
    """Decide whether a product matte may claim one physical atomic object.

    A clean foreground matte only proves that its *visible union* can be moved.
    It does not prove that touching or mutually occluding products inside that
    union are independently recoverable from one flattened raster.  V5
    therefore requires explicit, independent instance evidence before a
    product node is automatically described as an atomic leaf.  This policy
    never invents hidden pixels and deliberately keeps whole-group move safety
    separate from semantic/instance confirmation.
    """

    evidence = proposal.evidence
    instance_evidence = evidence.get("semantic_instance_evidence")
    instance_count: int | None = None
    independent = False
    visible_boundary_complete = False
    method: str | None = None
    if isinstance(instance_evidence, dict):
        try:
            parsed_count = int(instance_evidence.get("instance_count"))
            if parsed_count >= 1:
                instance_count = parsed_count
        except (TypeError, ValueError):
            instance_count = None
        independent = instance_evidence.get("independent") is True
        visible_boundary_complete = (
            instance_evidence.get("visible_boundary_complete") is True
        )
        method_value = str(instance_evidence.get("method") or "").strip()
        method = method_value or None

    atomic_leaf_confirmed = bool(
        independent
        and instance_count == 1
        and visible_boundary_complete
        and method
    )
    if independent and instance_count is not None and instance_count > 1:
        classification = "compound_subassembly"
        reasons = [
            f"independent instance evidence reports {instance_count} visible objects"
        ]
    elif atomic_leaf_confirmed:
        classification = "atomic_leaf"
        reasons = []
    else:
        classification = "atomicity_unverified"
        reasons = [
            "one clean matte does not prove one physical instance in a flattened raster"
        ]

    # Borderline SAM/BiRefNet agreement is useful evidence that the accepted
    # visible union may have expanded across a neighbouring/touching product.
    # It is not strong enough to assert a compound, so keep the wording
    # explicitly uncertain while still failing closed for atomicity.
    refinement = evidence.get("refinement")
    refinement_accepted = False
    refinement_iou: float | None = None
    refinement_area_ratio: float | None = None
    if isinstance(refinement, dict):
        refinement_accepted = refinement.get("accepted") is True
        try:
            refinement_iou = float(refinement.get("sam_iou"))
        except (TypeError, ValueError):
            refinement_iou = None
        try:
            refinement_area_ratio = float(refinement.get("area_ratio"))
        except (TypeError, ValueError):
            refinement_area_ratio = None
    boundary_ambiguous = bool(
        not atomic_leaf_confirmed
        and refinement_accepted
        and (
            (refinement_iou is not None and refinement_iou < 0.65)
            or (
                refinement_area_ratio is not None
                and not 0.67 <= refinement_area_ratio <= 1.50
            )
        )
    )
    if boundary_ambiguous and classification == "atomicity_unverified":
        classification = "compound_or_boundary_ambiguous"
        reasons.append(
            "accepted SAM/BiRefNet masks disagree too much to certify one instance"
        )

    return {
        "policy": "semantic_product_atomicity_fail_closed_v1",
        "classification": classification,
        "atomic_leaf_confirmed": atomic_leaf_confirmed,
        "review_required": not atomic_leaf_confirmed,
        "whole_visible_union_only": True,
        "hidden_instance_pixels_inferred": False,
        "instance_evidence": {
            "available": isinstance(instance_evidence, dict),
            "independent": independent,
            "instance_count": instance_count,
            "visible_boundary_complete": visible_boundary_complete,
            "method": method,
        },
        "refinement_boundary_evidence": {
            "accepted": refinement_accepted,
            "sam_iou": refinement_iou,
            "area_ratio": refinement_area_ratio,
            "boundary_ambiguous": boundary_ambiguous,
        },
        "reasons": reasons,
    }


def _bbox_from_mask(mask: np.ndarray, padding: int = 0) -> Box:
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, 0, 0
    height, width = mask.shape
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(width, int(xs.max()) + 1 + padding),
        min(height, int(ys.max()) + 1 + padding),
    )


def _qr_quiet_zone_box(box: Box, canvas_size: tuple[int, int]) -> Box:
    """Expand a DINO QR box enough to retain a printable quiet zone."""

    width, height = canvas_size
    x0, y0, x1, y1 = box
    margin = max(2, int(round(min(x1 - x0, y1 - y0) * 0.07)))
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(width, x1 + margin),
        min(height, y1 + margin),
    )


def _palette_from_ring(image_rgb: np.ndarray, bbox: Box, padding: int) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    ex0, ey0 = max(0, x0 - padding), max(0, y0 - padding)
    ex1, ey1 = min(width, x1 + padding), min(height, y1 + padding)
    crop = image_rgb[ey0:ey1, ex0:ex1]
    ring = np.ones(crop.shape[:2], dtype=bool)
    ring[y0 - ey0 : y1 - ey0, x0 - ex0 : x1 - ex0] = False
    samples = crop[ring]
    if len(samples) < 16:
        inside = image_rgb[y0:y1, x0:x1]
        edge = np.zeros(inside.shape[:2], dtype=bool)
        edge[: min(2, edge.shape[0]), :] = True
        edge[max(0, edge.shape[0] - 2) :, :] = True
        edge[:, : min(2, edge.shape[1])] = True
        edge[:, max(0, edge.shape[1] - 2) :] = True
        samples = inside[edge]
    if not len(samples):
        return np.array([[255, 255, 255]], dtype=np.uint8)
    quantized = (samples // 8).astype(np.uint16)
    packed = quantized[:, 0] * 1024 + quantized[:, 1] * 32 + quantized[:, 2]
    values, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts)[::-1][: min(6, len(values))]
    colors: list[np.ndarray] = []
    for value in values[order]:
        members = samples[packed == value]
        colors.append(np.median(members, axis=0).astype(np.uint8))
    return np.stack(colors)


def _nearest_palette_lab(
    pixels_rgb: np.ndarray, palette_rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = pixels_rgb.shape[:2]
    lab = cv2.cvtColor(pixels_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    palette_image = palette_rgb.reshape(1, -1, 3)
    palette_lab = cv2.cvtColor(palette_image, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    distances = np.linalg.norm(lab[:, :, None, :] - palette_lab[None, None, :, :], axis=3)
    nearest = np.argmin(distances, axis=2)
    minimum = np.take_along_axis(distances, nearest[:, :, None], axis=2)[:, :, 0]
    return minimum, nearest


def _interior_palette_hypotheses(crop_rgb: np.ndarray) -> list[np.ndarray]:
    """Estimate a local text background from the OCR rectangle itself.

    A ring is a good background estimator for ordinary dark text on paper, but
    it is the wrong estimator for text printed *inside* a coloured ribbon.  In
    that case the ring sees the page while the dominant colour inside the OCR
    rectangle is the actual red/green label surface.  Keep only common colour
    bins close to the dominant bin so the glyph colour is never silently added
    to the background palette.
    """

    samples = crop_rgb.reshape(-1, 3)
    if not len(samples):
        return [np.array([[255, 255, 255]], dtype=np.uint8)]
    quantized = (samples // 8).astype(np.uint16)
    packed = quantized[:, 0] * 1024 + quantized[:, 1] * 32 + quantized[:, 2]
    values, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts)[::-1][: min(24, len(values))]
    representatives: list[np.ndarray] = []
    representative_counts: list[int] = []
    for index in order:
        value = values[index]
        members = samples[packed == value]
        representatives.append(np.median(members, axis=0).astype(np.uint8))
        representative_counts.append(int(counts[index]))
    if not representatives:
        return [np.array([[255, 255, 255]], dtype=np.uint8)]

    palette_image = np.stack(representatives).reshape(1, -1, 3)
    palette_lab = cv2.cvtColor(palette_image, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(
        np.float32
    )
    minimum_bin_pixels = max(4, int(round(len(samples) * 0.006)))
    seed_indices: list[int] = []
    for index, (count, lab) in enumerate(zip(representative_counts, palette_lab, strict=True)):
        if count < max(minimum_bin_pixels, int(round(len(samples) * 0.018))):
            continue
        if all(float(np.linalg.norm(lab - palette_lab[seed])) > 24.0 for seed in seed_indices):
            seed_indices.append(index)
        if len(seed_indices) >= 4:
            break
    if not seed_indices:
        seed_indices = [0]

    hypotheses: list[np.ndarray] = []
    for seed in seed_indices:
        selected: list[np.ndarray] = []
        for colour, count, lab in zip(
            representatives, representative_counts, palette_lab, strict=True
        ):
            if count < minimum_bin_pixels:
                continue
            if float(np.linalg.norm(lab - palette_lab[seed])) <= 20.0:
                selected.append(colour)
            if len(selected) >= 8:
                break
        if not selected:
            selected.append(representatives[seed])
        hypotheses.append(np.stack(selected))
    return hypotheses


def _dominant_interior_palette(crop_rgb: np.ndarray) -> np.ndarray:
    """Backward-compatible primary interior palette hypothesis."""

    return _interior_palette_hypotheses(crop_rgb)[0]


def _row_bands(binary: np.ndarray) -> list[tuple[int, int, int]]:
    """Return vertically separated ink bands as ``(top, bottom, pixels)``.

    The small closing radius joins Vietnamese accents to their base glyph but
    deliberately does not join a second caption/icon row accidentally covered
    by an oversized OCR rectangle.
    """

    height = binary.shape[0]
    if height == 0 or not np.any(binary):
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


def _assess_text_mask_purity(
    alpha: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Filter obvious non-text matter and return fail-closed purity evidence.

    This is intentionally conservative.  A clean mask may be auto-confirmed;
    a mask that needed a ribbon, panel, adjacent icon, or second content band
    removed remains ``unsafe`` so a human sees that the OCR rectangle was
    ambiguous even though the exported candidate is cleaner.
    """

    alpha = np.asarray(alpha, dtype=np.uint8)
    binary = alpha >= 8
    height, width = binary.shape
    bbox_area = max(1, width * height)
    total = int(np.count_nonzero(binary))
    reasons: list[str] = []
    filtered = binary.copy()

    bands = _row_bands(binary)
    substantial_band_floor = max(6, int(round(total * 0.045)))
    substantial = [band for band in bands if band[2] >= substantial_band_floor]
    primary_band: tuple[int, int, int] | None = max(bands, key=lambda item: item[2]) if bands else None
    multi_band = len(substantial) > 1
    primary_band_fraction = (
        primary_band[2] / max(1, total) if primary_band is not None else 0.0
    )
    out_of_band_pixels = 0
    if multi_band and primary_band is not None:
        top, bottom, _pixels = primary_band
        margin = max(1, min(4, int(round(height * 0.035))))
        keep_rows = np.zeros(height, dtype=bool)
        keep_rows[max(0, top - margin) : min(height, bottom + margin)] = True
        out_of_band_pixels = int(np.count_nonzero(filtered[~keep_rows]))
        filtered[~keep_rows] = False
        reasons.append("multiple_vertical_content_bands")

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        filtered.astype(np.uint8), 8
    )
    components: list[dict[str, float | int]] = []
    for component in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[component]
        )
        component_bbox_area = max(1, component_width * component_height)
        components.append(
            {
                "label": component,
                "x": x,
                "y": y,
                "width": component_width,
                "height": component_height,
                "area": area,
                "area_fraction": area / max(1, int(np.count_nonzero(filtered))),
                "bbox_fraction": component_bbox_area / bbox_area,
                "fill": area / component_bbox_area,
            }
        )

    component_areas = [int(item["area"]) for item in components]
    median_area = float(np.median(component_areas)) if component_areas else 0.0
    micro_component_ceiling = max(4, int(round(total * 0.0015)))
    micro_component_count = sum(
        int(item["area"]) <= micro_component_ceiling for item in components
    )
    micro_component_fraction = micro_component_count / max(1, len(components))
    # Palette mistakes often leave only antialias/noise outlines: hundreds of
    # tiny disconnected islands instead of one connected component per glyph.
    # Do not let their deceptively low occupancy win candidate selection.
    fragmented_mask = bool(
        len(components) >= max(48, int(round(width * 0.24)))
        and median_area <= max(8.0, total * 0.002)
        and micro_component_fraction >= 0.45
    )
    largest = max(components, key=lambda item: int(item["area"])) if components else None
    largest_area_fraction = float(largest["area_fraction"]) if largest else 0.0
    largest_bbox_fraction = float(largest["bbox_fraction"]) if largest else 0.0
    largest_fill = float(largest["fill"]) if largest else 0.0

    giant_component = bool(
        largest
        and (
            (largest_bbox_fraction >= 0.14 and largest_fill >= 0.28)
            or (largest_area_fraction >= 0.48 and largest_fill >= 0.34)
        )
    )
    suspicious_labels: list[int] = []
    suspicious_boxes: list[tuple[int, int, int, int]] = []
    primary_height = (
        max(1, primary_band[1] - primary_band[0]) if primary_band is not None else max(1, height)
    )
    for item in components:
        area = int(item["area"])
        area_fraction = float(item["area_fraction"])
        component_width = int(item["width"])
        component_height = int(item["height"])
        fill = float(item["fill"])
        oversized_relative_to_glyphs = median_area > 0 and area >= max(18.0, median_area * 4.5)
        block_like = (
            component_width >= max(6, int(round(primary_height * 0.55)))
            and component_height >= max(5, int(round(primary_height * 0.42)))
            and fill >= 0.34
        )
        component_x = int(item["x"])
        component_y = int(item["y"])
        component_right = component_x + component_width
        component_bottom = component_y + component_height
        border_spanning_background = bool(
            (component_x <= 1 and component_right >= width - 1)
            or (component_y <= 1 and component_bottom >= height - 1)
            or float(item["bbox_fraction"]) >= 0.55
        )
        left_neighbour_right = max(
            (
                int(other["x"]) + int(other["width"])
                for other in components
                if int(other["label"]) != int(item["label"])
                and int(other["x"]) + int(other["width"]) <= component_x
                and int(other["area"]) >= max(5, median_area * 0.18)
            ),
            default=component_x,
        )
        right_neighbour_x = min(
            (
                int(other["x"])
                for other in components
                if int(other["label"]) != int(item["label"])
                and int(other["x"]) >= component_right
                and int(other["area"]) >= max(5, median_area * 0.18)
            ),
            default=component_right,
        )
        touches_left = component_x <= 1
        touches_right = component_right >= width - 1
        open_side_gap = (
            component_x - left_neighbour_right
            if touches_right
            else right_neighbour_x - component_right
            if touches_left
            else 0
        )
        edge_isolated_block = bool(
            (touches_left or touches_right)
            and component_height >= max(8, int(round(height * 0.78)))
            and component_width >= max(8, int(round(height * 0.55)))
            and area >= max(30.0, median_area * 2.5)
            and open_side_gap >= max(4, int(round(height * 0.30)))
            and fill >= 0.30
        )
        if (
            area_fraction >= 0.085
            and oversized_relative_to_glyphs
            and block_like
            and not border_spanning_background
        ) or edge_isolated_block:
            suspicious_labels.append(int(item["label"]))
            suspicious_boxes.append(
                (
                    component_x,
                    component_y,
                    component_right,
                    component_bottom,
                )
            )

    adjacent_object = bool(suspicious_labels)
    if giant_component and largest is not None:
        suspicious_labels.append(int(largest["label"]))
        reasons.append("giant_solid_background_component")
    if adjacent_object:
        reasons.append("adjacent_non_text_object")
    if fragmented_mask:
        reasons.append("fragmented_non_glyph_mask")
    suspicious_labels = sorted(set(suspicious_labels))
    if suspicious_labels:
        filtered[np.isin(labels, suspicious_labels)] = False
        # Contrasting details inside an icon/panel may be holes in the giant
        # component rather than members of that component. Remove the whole
        # proven-contaminant envelope so a cart glyph or ribbon highlight is
        # not left behind as fake text after its filled background is removed.
        for sx0, sy0, sx1, sy1 in suspicious_boxes:
            filtered[max(0, sy0 - 1) : min(height, sy1 + 1), max(0, sx0 - 1) : min(width, sx1 + 1)] = False

    # Decorations at an OCR-box edge may be made from several ordinary-sized
    # components (for example a heart plus a ribbon cap).  No single member is
    # anomalously large, but their horizontally overlapping group is isolated
    # from the glyph run.  Detect the group envelope, not a particular icon
    # shape, so its internal highlights cannot leak into the text owner.
    edge_suspicious_component_count = 0
    edge_count, edge_labels, edge_stats, _edge_centroids = (
        cv2.connectedComponentsWithStats(filtered.astype(np.uint8), 8)
    )
    edge_components: list[dict[str, int]] = []
    for component in range(1, edge_count):
        ex, ey, ew, eh, earea = (int(value) for value in edge_stats[component])
        if earea < 2:
            continue
        edge_components.append(
            {
                "label": component,
                "x0": ex,
                "y0": ey,
                "x1": ex + ew,
                "y1": ey + eh,
                "area": earea,
            }
        )
    edge_groups: list[dict[str, Any]] = []
    for item in sorted(edge_components, key=lambda value: (value["x0"], value["x1"])):
        if edge_groups and item["x0"] <= int(edge_groups[-1]["x1"]) + 1:
            group = edge_groups[-1]
            group["x1"] = max(int(group["x1"]), item["x1"])
            group["y0"] = min(int(group["y0"]), item["y0"])
            group["y1"] = max(int(group["y1"]), item["y1"])
            group["area"] = int(group["area"]) + item["area"]
            group["labels"].append(item["label"])
        else:
            edge_groups.append(
                {
                    "x0": item["x0"],
                    "y0": item["y0"],
                    "x1": item["x1"],
                    "y1": item["y1"],
                    "area": item["area"],
                    "labels": [item["label"]],
                }
            )
    post_component_areas = [int(item["area"]) for item in edge_components]
    post_median_area = (
        float(np.median(post_component_areas)) if post_component_areas else 0.0
    )
    significant_group_area = max(18.0, post_median_area * 0.65)
    significant_groups = [
        group for group in edge_groups if int(group["area"]) >= significant_group_area
    ]
    edge_tolerance = max(1, int(round(height * 0.04)))
    edge_group_boxes: list[tuple[int, int, int, int]] = []
    edge_group_labels: list[int] = []
    for group in significant_groups:
        gx0, gy0, gx1, gy1 = (
            int(group["x0"]),
            int(group["y0"]),
            int(group["x1"]),
            int(group["y1"]),
        )
        touches_left = gx0 <= edge_tolerance
        touches_right = gx1 >= width - edge_tolerance
        if touches_left == touches_right:
            continue
        if touches_right:
            neighbour_edge = max(
                (
                    int(other["x1"])
                    for other in significant_groups
                    if other is not group and int(other["x1"]) <= gx0
                ),
                default=gx0,
            )
            open_gap = gx0 - neighbour_edge
        else:
            neighbour_edge = min(
                (
                    int(other["x0"])
                    for other in significant_groups
                    if other is not group and int(other["x0"]) >= gx1
                ),
                default=gx1,
            )
            open_gap = neighbour_edge - gx1
        group_width = gx1 - gx0
        group_height = gy1 - gy0
        if (
            open_gap >= max(4, int(round(height * 0.18)))
            and group_width >= max(5, int(round(height * 0.24)))
            and group_height >= max(5, int(round(height * 0.25)))
        ):
            edge_group_boxes.append((gx0, gy0, gx1, gy1))
            edge_group_labels.extend(int(label) for label in group["labels"])
    if edge_group_boxes:
        adjacent_object = True
        reasons.append("adjacent_non_text_object")
        edge_suspicious_component_count = len(set(edge_group_labels))
        for gx0, gy0, gx1, gy1 in edge_group_boxes:
            filtered[
                max(0, gy0 - 1) : min(height, gy1 + 1),
                max(0, gx0 - 1) : min(width, gx1 + 1),
            ] = False

    # A border-connected panel/background can hide the second row from the
    # first projection pass.  Re-evaluate after contaminant removal so a
    # clean text row does not retain the cart/green rule underneath it.
    post_component_total = int(np.count_nonzero(filtered))
    post_bands = _row_bands(filtered)
    post_substantial_floor = max(6, int(round(post_component_total * 0.045)))
    post_substantial = [
        band for band in post_bands if band[2] >= post_substantial_floor
    ]
    if len(post_substantial) > 1:
        post_primary = max(post_bands, key=lambda item: item[2])
        post_top, post_bottom, post_pixels = post_primary
        margin = max(1, min(4, int(round(height * 0.035))))
        keep_rows = np.zeros(height, dtype=bool)
        keep_rows[
            max(0, post_top - margin) : min(height, post_bottom + margin)
        ] = True
        newly_removed = int(np.count_nonzero(filtered[~keep_rows]))
        filtered[~keep_rows] = False
        out_of_band_pixels += newly_removed
        multi_band = True
        primary_band_fraction = post_pixels / max(1, post_component_total)
        reasons.append("multiple_vertical_content_bands")
    bands_for_report = post_bands if post_bands else bands
    substantial_for_report = post_substantial if post_bands else substantial

    occupancy = total / bbox_area
    if occupancy > 0.72:
        reasons.append("text_mask_occupancy_too_high")
    if total < 2:
        reasons.append("insufficient_text_pixels")
    residual_component_count = int(
        cv2.connectedComponents(filtered.astype(np.uint8), 8)[0] - 1
    )
    if not np.any(filtered):
        reasons.append("no_proven_glyph_pixels_after_purity_filter")

    reasons = list(dict.fromkeys(reasons))
    status = "pass" if not reasons else "unsafe"
    filtered_alpha = alpha.copy()
    filtered_alpha[~filtered] = 0
    report: dict[str, Any] = {
        "policy": "source_mask_text_purity_v1",
        "status": status,
        "reasons": reasons,
        "giant_component": giant_component,
        "multi_band": multi_band,
        "adjacent_object": adjacent_object,
        "fragmented_mask": fragmented_mask,
        "protected_overlap_pixels": 0,
        "protected_overlap_fraction": 0.0,
        "residual_text_component_count": residual_component_count,
        "metrics": {
            "occupancy": round(occupancy, 6),
            "component_count": len(components),
            "largest_component_fraction": round(largest_area_fraction, 6),
            "largest_component_bbox_fraction": round(largest_bbox_fraction, 6),
            "largest_component_fill": round(largest_fill, 6),
            "median_component_area": round(median_area, 4),
            "micro_component_count": micro_component_count,
            "micro_component_fraction": round(micro_component_fraction, 6),
            "row_band_count": len(bands_for_report),
            "substantial_row_band_count": len(substantial_for_report),
            "primary_band_fraction": round(primary_band_fraction, 6),
            "out_of_band_pixels_removed": out_of_band_pixels,
            "out_of_band_fraction": round(out_of_band_pixels / max(1, total), 6),
            "suspicious_component_count": len(suspicious_labels)
            + edge_suspicious_component_count,
            "edge_suspicious_group_count": len(edge_group_boxes),
            "pixels_before_filter": total,
            "pixels_after_filter": int(np.count_nonzero(filtered_alpha)),
        },
    }
    return filtered_alpha, report


def _extract_text_alpha_candidate(
    crop: np.ndarray,
    palette: np.ndarray,
    hint: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    distance, nearest = _nearest_palette_lab(crop, palette)
    scaled = np.clip(distance * 4.0, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_lab = max(5.0, float(otsu) / 4.0)
    strong_lab = max(threshold_lab * 1.65, 13.0)
    support = distance >= threshold_lab
    if hint is not None:
        support &= (hint >= 2) | (distance >= strong_lab)
    occupancy = float(np.mean(support))
    used_hint_fallback = False
    if occupancy > 0.74 and hint is not None:
        support = hint >= 8
        occupancy = float(np.mean(support))
        used_hint_fallback = True
    if not np.any(support):
        alpha = np.zeros(crop.shape[:2], dtype=np.uint8)
    else:
        foreground_distances = distance[support]
        high = max(strong_lab, float(np.percentile(foreground_distances, 78)))
        low = max(2.5, threshold_lab * 0.55)
        alpha_float = np.clip((distance - low) / max(1e-6, high - low), 0.0, 1.0)
        alpha_float[~support] = 0.0
        alpha = np.clip(np.rint(alpha_float * 255.0), 0, 255).astype(np.uint8)
        alpha[np.logical_and(support, distance >= high)] = 255
    alpha, purity = _assess_text_mask_purity(alpha)
    return alpha, nearest, {
        "lab_threshold": round(threshold_lab, 4),
        "occupancy": round(occupancy, 6),
        "used_layerd_hint_fallback": used_hint_fallback,
        "text_purity": purity,
    }


def extract_text_rgba(
    image_rgb: np.ndarray,
    bbox: Box,
    *,
    layerd_alpha_hint: np.ndarray | None = None,
    recognized_text: str | None = None,
) -> tuple[np.ndarray, AlphaCrop, dict[str, Any]]:
    """Extract one logical raster text run without requiring a SAM proposal."""

    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Text bbox escapes canvas: {bbox}")
    padding = max(3, min(12, int(round(max(x1 - x0, y1 - y0) * 0.10))))
    crop = image_rgb[y0:y1, x0:x1]
    hint = (
        np.asarray(layerd_alpha_hint[y0:y1, x0:x1], dtype=np.uint8)
        if layerd_alpha_hint is not None
        else None
    )
    candidates: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
    interior_hypotheses = _interior_palette_hypotheses(crop)
    palette_hypotheses: list[tuple[str, np.ndarray]] = [
        ("outside_ring", _palette_from_ring(image_rgb, bbox, padding))
    ]
    palette_hypotheses.extend(
        (f"dominant_interior_{index}", palette)
        for index, palette in enumerate(interior_hypotheses, 1)
    )
    if len(interior_hypotheses) >= 2:
        palette_hypotheses.append(
            (
                "dominant_interior_mixed_surface",
                np.concatenate((interior_hypotheses[0], interior_hypotheses[1]), axis=0),
            )
        )
    for source, palette in palette_hypotheses:
        alpha, nearest, candidate_report = _extract_text_alpha_candidate(
            crop, palette, hint
        )
        purity = candidate_report["text_purity"]
        candidates.append((source, palette, alpha, nearest, candidate_report))

    def candidate_rank(
        item: tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
    ) -> tuple[int, int, int, int, int, int, float, int]:
        report = item[4]
        purity = report["text_purity"]
        metrics = purity["metrics"]
        return (
            0 if purity["status"] == "pass" else 1,
            1 if int(metrics["pixels_after_filter"]) < 2 else 0,
            1 if purity.get("fragmented_mask") else 0,
            1 if purity["giant_component"] else 0,
            1 if purity["adjacent_object"] else 0,
            len(purity["reasons"]),
            abs(float(report["occupancy"]) - 0.24),
            -int(metrics["pixels_after_filter"]),
        )

    source, palette, alpha, nearest, selected = min(candidates, key=candidate_rank)
    occupancy = float(selected["occupancy"])
    purity = selected["text_purity"]
    background_rgb = palette[nearest]
    a = alpha.astype(np.float32) / 255.0
    denominator = np.maximum(a[:, :, None], 1.0 / 255.0)
    foreground = (
        crop.astype(np.float32) - (1.0 - a[:, :, None]) * background_rgb.astype(np.float32)
    ) / denominator
    foreground = np.clip(np.rint(foreground), 0, 255).astype(np.uint8)
    foreground[a < 1.0 / 255.0] = 0
    rgba = np.dstack([foreground, alpha])
    return rgba, AlphaCrop(x0, y0, alpha), {
        "background_palette_rgb": palette.tolist(),
        "background_palette_source": source,
        "recognized_text": recognized_text,
        "lab_threshold": selected["lab_threshold"],
        "occupancy": round(occupancy, 6),
        "used_layerd_hint_fallback": selected["used_layerd_hint_fallback"],
        "nonzero_pixels": int(np.count_nonzero(alpha)),
        "component_count": int(cv2.connectedComponents((alpha > 0).astype(np.uint8), 8)[0] - 1),
        "text_purity": purity,
        "candidate_evidence": [
            {
                "background_palette_source": candidate[0],
                "background_palette_rgb": candidate[1].tolist(),
                "lab_threshold": candidate[4]["lab_threshold"],
                "occupancy": candidate[4]["occupancy"],
                "text_purity": candidate[4]["text_purity"],
            }
            for candidate in candidates
        ],
    }


def extract_semantic_rgba(
    image_rgb: np.ndarray,
    support: AlphaCrop,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover straight foreground RGB for a semantic soft-alpha asset.

    SAM/BiRefNet supply coverage, not foreground edge colour.  Exporting the
    observed RGB verbatim would bake the old background into antialiased edge
    pixels.  We estimate the local background from a ring and algebraically
    unblend it, while opaque interior pixels remain bit-identical to source.
    """

    height, width = image_rgb.shape[:2]
    support.canvas_slice((width, height))
    x0, y0, x1, y1 = support.bbox
    if not np.any(support.alpha):
        raise ValueError("Semantic support may not be empty.")
    padding = max(4, min(24, int(round(max(support.width, support.height) * 0.06))))
    palette = _palette_from_ring(image_rgb, support.bbox, padding)
    observed = image_rgb[y0:y1, x0:x1]
    _, nearest = _nearest_palette_lab(observed, palette)
    background = palette[nearest]
    alpha = support.alpha
    coverage = alpha.astype(np.float32) / 255.0
    denominator = np.maximum(coverage[:, :, None], 1.0 / 255.0)
    foreground = (
        observed.astype(np.float32)
        - (1.0 - coverage[:, :, None]) * background.astype(np.float32)
    ) / denominator
    foreground = np.clip(np.rint(foreground), 0, 255).astype(np.uint8)
    foreground[alpha == 0] = 0
    rgba = np.dstack([foreground, alpha.copy()])
    return rgba, {
        "background_palette_rgb": palette.tolist(),
        "ring_padding": padding,
        "nonzero_pixels": support.nonzero_pixels,
        "soft_pixel_count": int(np.count_nonzero((alpha > 0) & (alpha < 255))),
        "component_count": int(
            cv2.connectedComponents((alpha > 0).astype(np.uint8), connectivity=8)[0] - 1
        ),
        "straight_alpha_unblended": True,
    }


def _removal_footprint(support: AlphaCrop, canvas_size: tuple[int, int]) -> AlphaCrop:
    """Include the antialias halo when synthesizing the clean plate later."""

    width, height = canvas_size
    radius = max(1, min(6, int(round(min(support.width, support.height) * 0.012))))
    canvas = support.to_canvas(canvas_size)
    footprint = cv2.dilate(
        (canvas >= 2).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)),
    )
    box = _bbox_from_mask(footprint > 0)
    x0, y0, x1, y1 = box
    return AlphaCrop(x0, y0, (footprint[y0:y1, x0:x1] * 255).astype(np.uint8))


def _alpha_overlap(
    first: AlphaCrop,
    second: AlphaCrop,
) -> dict[str, float | int]:
    """Binary overlap statistics without allocating document-sized canvases."""

    x0, y0 = max(first.left, second.left), max(first.top, second.top)
    x1 = min(first.left + first.width, second.left + second.width)
    y1 = min(first.top + first.height, second.top + second.height)
    first_area = first.nonzero_pixels
    second_area = second.nonzero_pixels
    if x1 <= x0 or y1 <= y0:
        intersection = 0
    else:
        first_local = first.alpha[
            y0 - first.top : y1 - first.top,
            x0 - first.left : x1 - first.left,
        ] > 0
        second_local = second.alpha[
            y0 - second.top : y1 - second.top,
            x0 - second.left : x1 - second.left,
        ] > 0
        intersection = int(np.count_nonzero(first_local & second_local))
    union = first_area + second_area - intersection
    return {
        "intersection": intersection,
        "iou": intersection / union if union else 0.0,
        "first_coverage": intersection / max(1, first_area),
        "second_coverage": intersection / max(1, second_area),
        "area_ratio": max(first_area, second_area) / max(1, min(first_area, second_area)),
    }


def split_raw_layer_components(
    layer: LayerDRawLayer,
    *,
    subtract_alpha: np.ndarray | None = None,
) -> list[tuple[AlphaCrop, np.ndarray, dict[str, Any]]]:
    rgba = layer.rgba.copy()
    source_alpha = rgba[:, :, 3].copy()
    if subtract_alpha is not None:
        if subtract_alpha.shape != source_alpha.shape:
            raise ValueError("subtract_alpha canvas shape mismatch")
    # Label the original LayerD topology *before* subtracting higher-priority
    # text/semantic owners. Subtraction can punch hundreds of disconnected
    # holes through one useful LayerD element; relabelling afterwards was the
    # direct cause of the 757-component/877-layer explosion on ``truoc.png``.
    binary = source_alpha > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    components: list[tuple[AlphaCrop, np.ndarray, dict[str, Any]]] = []
    for component in range(1, count):
        x, y, width, height, binary_area = (int(value) for value in stats[component])
        local_label = labels[y : y + height, x : x + width] == component
        local_alpha = np.where(
            local_label,
            source_alpha[y : y + height, x : x + width],
            0,
        ).astype(np.uint8)
        protected_pixels = 0
        if subtract_alpha is not None:
            local_protected = subtract_alpha[y : y + height, x : x + width] > 0
            protected_pixels = int(np.count_nonzero((local_alpha > 0) & local_protected))
            # Pixel ownership is categorical even when coverage is soft. Alpha
            # multiplication would leave a second faint fringe around every
            # protected semantic/text asset.
            local_alpha[local_protected] = 0
        if not np.any(local_alpha):
            continue
        local_rgba = rgba[y : y + height, x : x + width].copy()
        local_rgba[:, :, 3] = local_alpha
        local_rgba[local_alpha == 0, :3] = 0
        tight = AlphaCrop(x, y, local_alpha).tight()
        tx, ty = tight.left - x, tight.top - y
        local_rgba = local_rgba[ty : ty + tight.height, tx : tx + tight.width].copy()
        local_rgba[:, :, 3] = tight.alpha
        remaining_component_count = int(
            cv2.connectedComponents((tight.alpha > 0).astype(np.uint8), 8)[0] - 1
        )
        components.append(
            (
                tight,
                local_rgba,
                {
                    "binary_area": binary_area,
                    "remaining_binary_area": tight.nonzero_pixels,
                    "protected_pixels_removed": protected_pixels,
                    "remaining_island_count_after_subtraction": remaining_component_count,
                    "topology_preserved_before_subtraction": True,
                    "soft_pixel_count": int(
                        np.count_nonzero((tight.alpha > 0) & (tight.alpha < 255))
                    ),
                    "source_iteration": layer.source_iteration,
                    "source_z_index": layer.z_index,
                },
            )
        )
    return components


def _classify_component(
    crop: AlphaCrop,
    rgba: np.ndarray,
    canvas_size: tuple[int, int],
    nearby: Iterable[DetectedProposal],
) -> tuple[ElementKind, float, list[str]]:
    bbox = crop.bbox
    width, height = crop.width, crop.height
    canvas_width, canvas_height = canvas_size
    area = crop.nonzero_pixels
    fill = area / max(1, width * height)
    aspect = max(width / max(1, height), height / max(1, width))
    matches: list[tuple[float, float, float, float, DetectedProposal]] = []
    for proposal in nearby:
        intersection = _intersection(bbox, proposal.record.bbox)
        if not intersection:
            continue
        proposal_coverage = intersection / max(1, _box_area(proposal.record.bbox))
        component_coverage = intersection / max(1, _box_area(bbox))
        union = _box_area(proposal.record.bbox) + _box_area(bbox) - intersection
        iou = intersection / max(1, union)
        # Containment alone is deliberately not a match.  A one-glyph island
        # inside a large card/frame used to inherit the frame label because its
        # own bbox was 100% contained.  Geometry evidence must describe roughly
        # the same extent as the pixels it is labelling.
        extent_agreement = min(
            _box_area(proposal.record.bbox), _box_area(bbox)
        ) / max(1, max(_box_area(proposal.record.bbox), _box_area(bbox)))
        score = max(iou, min(proposal_coverage, component_coverage)) * proposal.record.confidence
        matches.append((score, iou, extent_agreement, component_coverage, proposal))
    matches.sort(key=lambda item: item[0], reverse=True)
    source_ids = [
        item.record.proposal_id
        for score, iou, extent, _coverage, item in matches
        if score >= 0.30 and (iou >= 0.22 or extent >= 0.34)
    ]
    for score, iou, extent, component_coverage, proposal in matches:
        if score < 0.36:
            break
        if proposal.record.kind_hint == "line":
            proposal_width = proposal.record.bbox[2] - proposal.record.bbox[0]
            proposal_height = proposal.record.bbox[3] - proposal.record.bbox[1]
            proposal_aspect = max(
                proposal_width / max(1, proposal_height),
                proposal_height / max(1, proposal_width),
            )
            if (
                aspect >= 7
                and proposal_aspect >= 7
                and component_coverage >= 0.72
                and (iou >= 0.28 or extent >= 0.28)
            ):
                return "line", min(0.96, 0.62 + score * 0.34), source_ids
        if proposal.record.kind_hint == "frame":
            edge_band = max(1, min(8, int(round(min(width, height) * 0.08))))
            binary = crop.alpha > 0
            border = np.zeros_like(binary)
            border[:edge_band, :] = True
            border[-edge_band:, :] = True
            border[:, :edge_band] = True
            border[:, -edge_band:] = True
            border_occupancy = float(np.mean(binary[border])) if np.any(border) else 0.0
            inner_occupancy = float(np.mean(binary[~border])) if np.any(~border) else 0.0
            touches = sum(
                bool(np.any(edge))
                for edge in (
                    binary[:edge_band, :],
                    binary[-edge_band:, :],
                    binary[:, :edge_band],
                    binary[:, -edge_band:],
                )
            )
            frame_shape = (
                touches == 4
                and border_occupancy >= 0.16
                and inner_occupancy <= max(0.20, border_occupancy * 0.72)
                and fill <= 0.42
            )
            if frame_shape and iou >= 0.32 and extent >= 0.38:
                return "frame", min(0.96, 0.63 + score * 0.33), source_ids
    if aspect >= 10 and min(width, height) <= max(18, int(min(canvas_size) * 0.025)):
        return "line", 0.76, source_ids
    bbox_fraction = width * height / max(1, canvas_width * canvas_height)
    if bbox_fraction >= 0.025 and fill <= 0.24:
        binary = crop.alpha > 0
        edge_band = max(1, min(8, int(round(min(width, height) * 0.08))))
        touches = sum(
            bool(np.any(edge))
            for edge in (
                binary[:edge_band, :],
                binary[-edge_band:, :],
                binary[:, :edge_band],
                binary[:, -edge_band:],
            )
        )
        border = np.zeros_like(binary)
        border[:edge_band, :] = True
        border[-edge_band:, :] = True
        border[:, :edge_band] = True
        border[:, -edge_band:] = True
        border_occupancy = float(np.mean(binary[border])) if np.any(border) else 0.0
        inner_occupancy = float(np.mean(binary[~border])) if np.any(~border) else 0.0
        if touches == 4 and border_occupancy >= 0.16 and inner_occupancy <= border_occupancy * 0.72:
            return "frame", 0.68, source_ids
    rgb = rgba[:, :, :3][crop.alpha >= 16]
    if len(rgb):
        quantized = rgb // 24
        unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
        entropy_proxy = unique_colors / max(1.0, math.sqrt(len(rgb)))
    else:
        entropy_proxy = 0.0
    if fill >= 0.55 and entropy_proxy < 0.35:
        return "badge", 0.58, source_ids
    if area <= 12:
        return "micro_detail", 0.35, source_ids
    return "unknown", 0.45, source_ids


@dataclass(slots=True)
class _RawComponent:
    crop: AlphaCrop
    rgba: np.ndarray
    report: dict[str, Any]
    proposal_id: str
    component_index: int


def _crop_median_lab(crop: AlphaCrop, rgba: np.ndarray) -> np.ndarray | None:
    pixels = rgba[:, :, :3][crop.alpha >= 8]
    if not len(pixels):
        return None
    median = np.median(pixels, axis=0).astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(median, cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)


def _node_median_lab(node: ElementNode) -> np.ndarray | None:
    if node.rgba is None:
        return None
    support = node.full_support or node.visible_alpha
    return _crop_median_lab(support, node.rgba)


def _bbox_gap(first: Box, second: Box) -> tuple[int, int]:
    horizontal = max(0, max(first[0], second[0]) - min(first[2], second[2]))
    vertical = max(0, max(first[1], second[1]) - min(first[3], second[3]))
    return horizontal, vertical


def _merge_rgba_parts(
    parts: Iterable[tuple[AlphaCrop, np.ndarray]],
) -> tuple[AlphaCrop, np.ndarray]:
    material = list(parts)
    if not material:
        raise ValueError("At least one RGBA part is required")
    x0 = min(crop.left for crop, _rgba in material)
    y0 = min(crop.top for crop, _rgba in material)
    x1 = max(crop.left + crop.width for crop, _rgba in material)
    y1 = max(crop.top + crop.height for crop, _rgba in material)
    result = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    # Parts are expected to be exclusive after protected-alpha subtraction.
    # In a numerical fringe collision the already written (higher-confidence)
    # pixel wins, avoiding a second blend against an unknown background.
    for crop, rgba in material:
        if rgba.shape[:2] != crop.alpha.shape:
            raise ValueError("RGBA part shape differs from its alpha crop")
        dy, dx = crop.top - y0, crop.left - x0
        target = result[dy : dy + crop.height, dx : dx + crop.width]
        take = (crop.alpha > 0) & (target[:, :, 3] == 0)
        target[take] = rgba[take]
        target[:, :, 3] = np.maximum(target[:, :, 3], crop.alpha)
    support = AlphaCrop(x0, y0, result[:, :, 3].copy()).tight()
    sx0, sy0 = support.left - x0, support.top - y0
    tight_rgba = result[sy0 : sy0 + support.height, sx0 : sx0 + support.width].copy()
    tight_rgba[:, :, 3] = support.alpha
    tight_rgba[support.alpha == 0, :3] = 0
    return support, tight_rgba


def _merge_component_into_node(
    node: ElementNode,
    component: _RawComponent,
    canvas_size: tuple[int, int],
) -> None:
    if node.rgba is None:
        raise ValueError("Cannot absorb a raw component into a node without RGBA")
    old_support = node.full_support or node.visible_alpha
    merged, rgba = _merge_rgba_parts(
        ((old_support, node.rgba), (component.crop, component.rgba))
    )
    node.visible_alpha = merged
    node.full_support = merged
    node.semantic_envelope = merged
    node.removal_footprint = _removal_footprint(merged, canvas_size)
    node.rgba = rgba
    absorbed = node.metadata.setdefault("absorbed_layerd_components", [])
    absorbed.append(component.proposal_id)
    extraction = node.metadata.get("text_extraction")
    if isinstance(extraction, dict):
        extraction["nonzero_pixels_after_absorption"] = merged.nonzero_pixels
        extraction["absorbed_component_count"] = len(absorbed)
        extraction["component_count_after_absorption"] = int(
            cv2.connectedComponents((merged.alpha > 0).astype(np.uint8), 8)[0] - 1
        )
    _refresh_text_purity_after_absorption(node)


def _refresh_text_purity_after_absorption(node: ElementNode) -> None:
    """Re-run the purity gate after a LayerD island joins an OCR owner."""

    if node.kind not in {"text", "price"}:
        return
    support = node.full_support or node.visible_alpha
    _filtered, fresh = _assess_text_mask_purity(support.alpha)
    existing = node.metadata.get("text_purity")
    if not isinstance(existing, dict):
        existing = {
            "policy": "source_mask_text_purity_v1",
            "status": "unsafe",
            "reasons": ["missing_pre_absorption_text_purity_evidence"],
            "giant_component": False,
            "multi_band": False,
            "adjacent_object": False,
            "protected_overlap_pixels": 0,
            "protected_overlap_fraction": 0.0,
            "residual_text_component_count": 0,
            "metrics": {},
        }
    reasons = [str(reason) for reason in existing.get("reasons", [])]
    reasons.extend(str(reason) for reason in fresh.get("reasons", []))
    reasons = list(dict.fromkeys(reasons))
    metrics = dict(existing.get("metrics") or {})
    fresh_metrics = dict(fresh.get("metrics") or {})
    metrics["post_absorption"] = fresh_metrics
    metrics["suspicious_component_count"] = max(
        int(metrics.get("suspicious_component_count", 0)),
        int(fresh_metrics.get("suspicious_component_count", 0)),
    )
    existing.update(
        {
            "policy": "source_mask_text_purity_v1",
            "status": "pass" if not reasons else "unsafe",
            "reasons": reasons,
            "giant_component": bool(existing.get("giant_component"))
            or bool(fresh.get("giant_component")),
            "multi_band": bool(existing.get("multi_band"))
            or bool(fresh.get("multi_band")),
            "adjacent_object": bool(existing.get("adjacent_object"))
            or bool(fresh.get("adjacent_object")),
            "residual_text_component_count": int(
                fresh.get("residual_text_component_count", 0)
            ),
            "metrics": metrics,
        }
    )
    node.metadata["text_purity"] = existing
    extraction = node.metadata.get("text_extraction")
    if isinstance(extraction, dict):
        extraction["text_purity"] = existing
    if existing["status"] != "pass":
        node.review_status = "unresolved"
        node.move_safe = False


def _text_absorption_owner(
    component: _RawComponent,
    text_owners: Iterable[tuple[ProposalRecord, ElementNode]],
) -> ElementNode | None:
    """Return a text owner only for a plausible omitted mark/stroke.

    This gate is intentionally conservative: being inside an OCR rectangle is
    not enough.  The island must be small relative to the logical run, near its
    actual ink, and chromatically compatible.  That preserves Vietnamese marks
    without dragging card borders or product pixels into the text layer.
    """

    component_lab = _crop_median_lab(component.crop, component.rgba)
    best: tuple[float, ElementNode] | None = None
    for proposal, node in text_owners:
        intersection = _intersection(component.crop.bbox, proposal.bbox)
        containment = intersection / max(1, _box_area(component.crop.bbox))
        if containment < 0.82:
            continue
        text_width = max(1, proposal.bbox[2] - proposal.bbox[0])
        text_height = max(1, proposal.bbox[3] - proposal.bbox[1])
        max_mark_area = max(18, int(round(text_width * text_height * 0.055)))
        if component.crop.nonzero_pixels > max_mark_area:
            continue
        if (
            component.crop.width > max(5, int(round(text_width * 0.32)))
            or component.crop.height > max(5, int(round(text_height * 0.58)))
        ):
            continue
        support = node.full_support or node.visible_alpha
        horizontal_gap, vertical_gap = _bbox_gap(component.crop.bbox, support.bbox)
        # Support crops normally retain the OCR bbox, so calculate true ink
        # proximity in canvas coordinates as well.
        px0 = min(component.crop.left, support.left)
        py0 = min(component.crop.top, support.top)
        px1 = max(component.crop.left + component.crop.width, support.left + support.width)
        py1 = max(component.crop.top + component.crop.height, support.top + support.height)
        local_ink = np.zeros((py1 - py0, px1 - px0), dtype=np.uint8)
        sy, sx = support.top - py0, support.left - px0
        local_ink[sy : sy + support.height, sx : sx + support.width] = support.alpha > 0
        radius = max(2, min(12, int(round(text_height * 0.24))))
        near_ink = cv2.dilate(
            local_ink,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)),
        )
        cy, cx = component.crop.top - py0, component.crop.left - px0
        touches_ink = bool(
            np.any(
                near_ink[cy : cy + component.crop.height, cx : cx + component.crop.width]
                & (component.crop.alpha > 0)
            )
        )
        if not touches_ink and (horizontal_gap > radius or vertical_gap > radius):
            continue
        node_lab = _node_median_lab(node)
        color_distance = (
            float(np.linalg.norm(component_lab - node_lab))
            if component_lab is not None and node_lab is not None
            else 0.0
        )
        if color_distance > 30.0:
            continue
        score = containment + (0.5 if touches_ink else 0.0) - color_distance / 100.0
        if best is None or score > best[0]:
            best = (score, node)
    return best[1] if best else None


def _cluster_raw_components(
    components: list[_RawComponent],
    canvas_size: tuple[int, int],
) -> list[list[_RawComponent]]:
    """Cluster nearby islands into useful movable residual elements.

    LayerD's matte often represents each glyph, accent, highlight and shadow as
    a disconnected component.  Exporting those connected components directly
    creates hundreds of unusable pseudo-layers.  This graph groups only nearby,
    colour-compatible islands; distant islands and different colours remain
    independent and every source component retains its own ledger record.
    """

    if not components:
        return []
    canvas_width, canvas_height = canvas_size
    bridge_x = max(4, min(24, int(round(canvas_width * 0.015))))
    bridge_y = max(3, min(18, int(round(canvas_height * 0.009))))
    parents = list(range(len(components)))
    labs = [_crop_median_lab(item.crop, item.rgba) for item in components]

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        a, b = find(first), find(second)
        if a != b:
            parents[b] = a

    ordered = sorted(range(len(components)), key=lambda index: components[index].crop.left)
    for order_index, first_index in enumerate(ordered):
        first = components[first_index].crop
        scan_limit = first.bbox[2] + max(bridge_x * 2, first.height)
        for second_index in ordered[order_index + 1 :]:
            second = components[second_index].crop
            if second.left > scan_limit:
                break
            horizontal_gap, vertical_gap = _bbox_gap(first.bbox, second.bbox)
            vertical_overlap = max(
                0,
                min(first.bbox[3], second.bbox[3])
                - max(first.bbox[1], second.bbox[1]),
            )
            horizontal_overlap = max(
                0,
                min(first.bbox[2], second.bbox[2])
                - max(first.bbox[0], second.bbox[0]),
            )
            same_row = (
                vertical_overlap / max(1, min(first.height, second.height)) >= 0.24
                and horizontal_gap
                <= max(
                    bridge_x,
                    min(38, int(round(min(first.height, second.height) * 1.10))),
                )
            )
            stacked_detail = (
                horizontal_overlap / max(1, min(first.width, second.width)) >= 0.24
                and vertical_gap
                <= max(
                    bridge_y,
                    min(22, int(round(min(first.height, second.height) * 0.60))),
                )
            )
            close = horizontal_gap <= bridge_x and vertical_gap <= bridge_y
            if not (same_row or stacked_detail or close):
                continue
            if labs[first_index] is not None and labs[second_index] is not None:
                color_distance = float(np.linalg.norm(labs[first_index] - labs[second_index]))
                first_area = first.nonzero_pixels
                second_area = second.nonzero_pixels
                small_detail = min(first_area, second_area) <= max(
                    48, int(round(max(first_area, second_area) * 0.12))
                )
                # Small highlights, accent marks and shadows often have a very
                # different palette from the parent artwork but still belong
                # to the same movable element. Large peer islands remain under
                # the stricter colour gate so neighbouring cards do not merge.
                if color_distance > (68.0 if small_detail else 40.0):
                    continue
            union(first_index, second_index)
    groups: dict[int, list[_RawComponent]] = {}
    for index, component in enumerate(components):
        groups.setdefault(find(index), []).append(component)
    return sorted(
        groups.values(),
        key=lambda group: (
            min(item.crop.top for item in group),
            min(item.crop.left for item in group),
        ),
    )


def _raw_cluster_auto_confirmation_policy(
    cluster: list[_RawComponent],
    crop: AlphaCrop,
    canvas_size: tuple[int, int],
    *,
    kind: ElementKind,
    confidence: float,
) -> dict[str, Any]:
    """Return auditable, fail-closed evidence for a LayerD cluster decision.

    ``_cluster_raw_components`` deliberately uses single-linkage grouping so a
    glyph can keep its accent/highlight.  Single linkage can also bridge many
    unrelated islands across a poster.  A detector label is not sufficient
    evidence that such a document-scale aggregate is one movable object.

    The limits below are safety limits, not segmentation parameters.  A clean
    connected component remains eligible for automatic confirmation.  A
    disconnected cluster is deferred when its topology, document-scale span,
    sparsity, or palette dispersion makes contamination plausible.
    """

    canvas_width, canvas_height = canvas_size
    member_count = len(cluster)
    effective_island_count = sum(
        max(1, int(item.report.get("remaining_island_count_after_subtraction", 1)))
        for item in cluster
    )
    canvas_area = max(1, canvas_width * canvas_height)
    bbox_area = max(1, crop.width * crop.height)
    supported_pixels = crop.nonzero_pixels
    member_bbox_area = sum(max(1, item.crop.width * item.crop.height) for item in cluster)
    member_areas = [item.crop.nonzero_pixels for item in cluster]

    labs = [
        lab
        for item in cluster
        if (lab := _crop_median_lab(item.crop, item.rgba)) is not None
    ]
    max_lab_distance = 0.0
    for first_index, first in enumerate(labs):
        for second in labs[first_index + 1 :]:
            max_lab_distance = max(
                max_lab_distance,
                float(np.linalg.norm(first - second)),
            )

    evidence = {
        "member_component_count": member_count,
        "effective_island_count": effective_island_count,
        "subtraction_fragmented_member_count": sum(
            int(item.report.get("remaining_island_count_after_subtraction", 1)) > 1
            for item in cluster
        ),
        "bbox_area_fraction": round(bbox_area / canvas_area, 6),
        "width_fraction": round(crop.width / max(1, canvas_width), 6),
        "height_fraction": round(crop.height / max(1, canvas_height), 6),
        "support_density": round(supported_pixels / bbox_area, 6),
        "component_bbox_packing": round(member_bbox_area / bbox_area, 6),
        "largest_component_fraction": round(
            max(member_areas, default=0) / max(1, sum(member_areas)), 6
        ),
        "max_component_lab_distance": round(max_lab_distance, 3),
        "geometry_kind_requires_clean_reference": kind in _RAW_GEOMETRY_KINDS,
        # LayerD is a foreground decomposition, not a clean-surface oracle.
        # Fusion therefore has no trustworthy reference colour/surface with
        # which to prove that a line/frame/panel matte is free of baked text,
        # prices, icons or product pixels.
        "clean_geometry_reference_available": False,
        "clean_geometry_reference_type": None,
    }
    thresholds = {
        "minimum_classification_confidence": 0.65,
        "maximum_member_components": 8,
        "maximum_effective_islands": 8,
        "maximum_bbox_area_fraction": 0.08,
        "maximum_width_fraction": 0.40,
        "maximum_height_fraction": 0.40,
        "minimum_support_density": 0.035,
        "maximum_component_lab_distance": 48.0,
    }
    reasons: list[str] = []
    if kind in _RAW_GEOMETRY_KINDS:
        reasons.append("missing_clean_geometry_reference")
    if kind == "unknown":
        reasons.append("unknown_classification")
    if confidence < float(thresholds["minimum_classification_confidence"]):
        reasons.append("classification_confidence_below_threshold")

    # A connected matte does not carry the single-linkage contamination risk;
    # its classification still has to satisfy the semantic confidence gate.
    if member_count > 1 or effective_island_count > 1:
        if member_count > int(thresholds["maximum_member_components"]):
            reasons.append("excessive_member_component_count")
        if effective_island_count > int(thresholds["maximum_effective_islands"]):
            reasons.append("excessive_effective_island_count")
        if evidence["bbox_area_fraction"] > thresholds["maximum_bbox_area_fraction"]:
            reasons.append("document_scale_bbox")
        if evidence["width_fraction"] > thresholds["maximum_width_fraction"]:
            reasons.append("document_scale_width")
        if evidence["height_fraction"] > thresholds["maximum_height_fraction"]:
            reasons.append("document_scale_height")
        if evidence["support_density"] < thresholds["minimum_support_density"]:
            reasons.append("sparse_multi_island_support")
        if (
            len(labs) > 1
            and evidence["max_component_lab_distance"]
            > thresholds["maximum_component_lab_distance"]
        ):
            reasons.append("mixed_component_palette")

    eligible = not reasons
    return {
        "policy": "layerd_raw_cluster_fail_closed_v1",
        "eligible": eligible,
        "decision": "auto_confirm" if eligible else "defer_to_review",
        "reasons": reasons,
        "evidence": evidence,
        "thresholds": thresholds,
        "geometry_reference": {
            "required": kind in _RAW_GEOMETRY_KINDS,
            "available": False,
            "reference_type": None,
            "reason": (
                "LayerD/raw geometry has no independently clean surface reference"
                if kind in _RAW_GEOMETRY_KINDS
                else "not_required_for_non_geometry_kind"
            ),
        },
        "rationale": (
            "Disconnected LayerD islands are not movable until their topology, "
            "document span, density and palette all remain inside a conservative "
            "coherence envelope. LayerD line/frame/panel classifications also "
            "remain review-only until an independent clean reference proves the "
            "surface is uncontaminated."
        ),
    }


def _split_raw_component_into_final_pieces(
    component: _RawComponent,
) -> list[_RawComponent]:
    """Return the smallest connected pieces left after protected subtraction.

    A :class:`_RawComponent` follows the topology of the original LayerD
    matte.  Higher-priority text/semantic ownership can punch that topology
    into several disconnected islands.  Keeping those islands together is
    useful only after the cluster coherence policy has accepted the aggregate.
    For a rejected cluster, each final connected island is an independent
    review item so an unrelated speck can never enlarge another layer's bbox.
    """

    binary = component.crop.alpha > 0
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    material: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    for label in range(1, count):
        x, y, width, height, _area = (int(value) for value in stats[label])
        local_membership = labels[y : y + height, x : x + width] == label
        source_alpha = component.crop.alpha[y : y + height, x : x + width]
        alpha = np.where(local_membership, source_alpha, 0).astype(np.uint8)
        if not np.any(alpha):
            continue
        rgba = component.rgba[y : y + height, x : x + width].copy()
        rgba[:, :, 3] = alpha
        rgba[alpha == 0, :3] = 0
        material.append(
            (
                component.crop.top + y,
                component.crop.left + x,
                width,
                height,
                alpha,
                rgba,
            )
        )

    # OpenCV labels are deterministic for one binary raster, but explicitly
    # ordering by document coordinates makes the public node IDs independent
    # of that implementation detail.
    material.sort(key=lambda item: (item[0], item[1], item[3], item[2]))
    piece_count = len(material)
    pieces: list[_RawComponent] = []
    for piece_index, (top, left, _width, _height, alpha, rgba) in enumerate(
        material, 1
    ):
        report = dict(component.report)
        report.update(
            {
                "source_component_proposal_id": component.proposal_id,
                "final_piece_index": piece_index,
                "final_piece_count": piece_count,
                "final_connected_component": True,
                "remaining_island_count_after_atomic_split": 1,
            }
        )
        pieces.append(
            _RawComponent(
                crop=AlphaCrop(left, top, alpha),
                rgba=rgba,
                report=report,
                proposal_id=component.proposal_id,
                component_index=component.component_index,
            )
        )
    return pieces


def _make_visible_ownership(nodes: list[ElementNode], canvas_size: tuple[int, int]) -> None:
    canvas_width, canvas_height = canvas_size
    claimed = np.zeros((canvas_height, canvas_width), dtype=bool)
    for node in sorted(nodes, key=lambda item: item.z_index, reverse=True):
        support = node.full_support or node.visible_alpha
        ys, xs = support.canvas_slice(canvas_size)
        local_claimed = claimed[ys, xs]
        visible = support.alpha.copy()
        visible[local_claimed] = 0
        node.visible_alpha = AlphaCrop(support.left, support.top, visible)
        claimed[ys, xs] |= support.alpha > 0


def _proposal_match_score(proposal: ProposalRecord, node: ElementNode) -> float:
    intersection = _intersection(proposal.bbox, node.bbox)
    if not intersection:
        return 0.0
    return max(
        intersection / max(1, _box_area(proposal.bbox)),
        intersection / max(1, _box_area(node.bbox)),
    )


def fuse_inventory_and_layerd(
    image_rgb: np.ndarray,
    layerd: LayerDResult,
    inventory: InventoryResult,
) -> FusionResult:
    """Build atomic leaves and a complete proposal ledger without hard layer caps."""

    height, width = image_rgb.shape[:2]
    if layerd.background_rgb.shape != image_rgb.shape:
        raise ValueError("LayerD background size differs from source.")
    graph = DocumentGraph((width, height))
    for item in inventory.proposals:
        graph.add_proposal(item.record)

    layerd_alpha_hint = np.zeros((height, width), dtype=np.uint8)
    for raw in layerd.foregrounds_bottom_to_top:
        np.maximum(layerd_alpha_hint, raw.rgba[:, :, 3], out=layerd_alpha_hint)
    protected = np.zeros((height, width), dtype=np.uint8)
    nodes: list[ElementNode] = []
    node_counter = 0

    # QR is a single useful movable asset including its quiet zone. Keeping the
    # tight rectangle opaque avoids broken modules and preserves scanability.
    for detected in [
        item
        for item in inventory.proposals
        if item.record.kind_hint == "qr"
        and not item.record.source.startswith("grounding_dino")
    ]:
        x0, y0, x1, y1 = detected.record.bbox
        alpha = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)
        rgba = np.dstack([image_rgb[y0:y1, x0:x1].copy(), alpha])
        node_counter += 1
        node_id = f"QR_{node_counter:04d}"
        crop = AlphaCrop(x0, y0, alpha)
        node = ElementNode(
            node_id,
            f"QR CODE {node_counter:02d}",
            "qr",
            crop,
            2_000_000 + node_counter,
            semantic_envelope=crop,
            full_support=crop,
            confidence=detected.record.confidence,
            review_status="auto_confirmed" if detected.record.confidence >= 0.95 else "unresolved",
            move_safe=True,
            rgba=rgba,
            evidence=[{"proposal_id": detected.record.proposal_id, "source": detected.record.source}],
        )
        nodes.append(node)
        detected.record.status = "assigned"
        detected.record.owner_ids = [node_id]
        protected[y0:y1, x0:x1] = 255

    # OpenCV can miss a valid but low-resolution QR when decoding fails.  A
    # geometrically plausible DINO QR still becomes an opaque, review-required
    # owner; using a rectangle (not its SAM matte) keeps every module and the
    # quiet zone intact.
    for detected in [
        item
        for item in inventory.proposals
        if item.record.kind_hint == "qr"
        and item.record.source.startswith("grounding_dino")
    ]:
        proposal = detected.record
        policy = proposal.evidence.get("semantic_policy") or {}
        if not bool(policy.get("accepted", False)):
            continue
        dino_box_value = proposal.evidence.get("dino_bbox") or proposal.bbox
        dino_box = tuple(int(value) for value in dino_box_value)
        if len(dino_box) != 4:
            continue
        box_width, box_height = dino_box[2] - dino_box[0], dino_box[3] - dino_box[1]
        aspect = max(box_width / max(1, box_height), box_height / max(1, box_width))
        if box_width < 8 or box_height < 8 or aspect > 1.8:
            proposal.reason = "DINO QR geometry is not square enough; manual review required"
            continue
        qr_box = _qr_quiet_zone_box(dino_box, (width, height))
        duplicate = next(
            (
                node
                for node in nodes
                if node.kind == "qr"
                and _intersection(qr_box, node.bbox)
                / max(1, min(_box_area(qr_box), _box_area(node.bbox)))
                >= 0.60
            ),
            None,
        )
        if duplicate is not None:
            proposal.status = "assigned"
            proposal.owner_ids = [duplicate.element_id]
            proposal.reason = "DINO QR agrees with an existing OpenCV QR owner"
            duplicate.evidence.append(
                {"proposal_id": proposal.proposal_id, "source": proposal.source}
            )
            continue
        x0, y0, x1, y1 = qr_box
        alpha = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)
        crop = AlphaCrop(x0, y0, alpha)
        rgba = np.dstack([image_rgb[y0:y1, x0:x1].copy(), alpha])
        node_counter += 1
        node_id = f"QR_{node_counter:04d}"
        node = ElementNode(
            node_id,
            f"QR CODE REVIEW {node_counter:02d}",
            "qr",
            crop,
            2_000_000 + node_counter,
            # The printable quiet zone is part of the QR asset.  A SAM hint is
            # normally tighter than that zone, so using it as the semantic
            # envelope would make valid opaque QR pixels look like leakage.
            semantic_envelope=crop,
            full_support=crop,
            removal_footprint=_removal_footprint(crop, (width, height)),
            confidence=proposal.confidence,
            review_status="unresolved",
            move_safe=False,
            rgba=rgba,
            evidence=[{"proposal_id": proposal.proposal_id, "source": proposal.source}],
            metadata={
                "quiet_zone_preserved": True,
                "dino_bbox": list(dino_box),
                "review_reason": "OpenCV did not provide an independently decoded QR",
            },
        )
        nodes.append(node)
        proposal.status = "assigned"
        proposal.owner_ids = [node_id]
        proposal.reason = "DINO-only QR owner requires scan/review confirmation"
        protected[y0:y1, x0:x1] = 255

    semantic_kinds = {"product", "logo", "icon", "badge", "decoration", "ribbon"}
    semantic_candidates = [
        item
        for item in inventory.proposals
        if item.record.kind_hint in semantic_kinds
        and item.record.source.startswith("grounding_dino")
        and item.mask_hint is not None
        and bool(item.record.evidence.get("auto_extractable", True))
    ]
    # A product owns branding printed on its packaging.  Processing products
    # first lets a nested logo proposal attach as evidence instead of exporting
    # duplicate pixels as a fake independent logo.
    semantic_priority: dict[ElementKind, int] = {
        "product": 0,
        "ribbon": 1,
        "decoration": 2,
        "badge": 3,
        "logo": 4,
        "icon": 5,
    }
    semantic_candidates.sort(
        key=lambda item: (
            semantic_priority.get(item.record.kind_hint, 99),
            -item.record.confidence,
            item.record.proposal_id,
        )
    )
    semantic_nodes: list[ElementNode] = []
    semantic_duplicates = 0
    semantic_overlap_deferrals = 0
    semantic_empty_failures = 0
    semantic_ambiguity_proposals = 0
    semantic_refinement_reviews = 0
    semantic_kind_mismatch_reviews = 0
    semantic_ambiguity_owner_ids: set[str] = set()
    semantic_atomicity_review_owner_ids: set[str] = set()
    semantic_compound_owner_ids: set[str] = set()
    for detected in semantic_candidates:
        proposal = detected.record
        assert detected.mask_hint is not None
        ambiguity_reasons = _semantic_ambiguity_reasons(proposal)
        if ambiguity_reasons:
            _record_semantic_ambiguity(proposal, ambiguity_reasons)
            semantic_ambiguity_proposals += 1
            refinement = proposal.evidence.get("refinement")
            if isinstance(refinement, dict) and refinement.get("accepted") is False:
                semantic_refinement_reviews += 1
            policy = proposal.evidence.get("semantic_policy")
            if (
                isinstance(policy, dict)
                and str(policy.get("kind") or "").strip()
                and str(policy.get("kind")).strip() != proposal.kind_hint
            ):
                semantic_kind_mismatch_reviews += 1
        original_support = detected.mask_hint.tight()
        if original_support.nonzero_pixels < 4:
            proposal.reason = "Semantic matte contains fewer than four supported pixels"
            semantic_empty_failures += 1
            continue

        duplicate_owner: ElementNode | None = None
        duplicate_report: dict[str, float | int] | None = None
        for owner in semantic_nodes:
            owner_support = owner.full_support or owner.visible_alpha
            overlap = _alpha_overlap(original_support, owner_support)
            same_asset = proposal.kind_hint == owner.kind and (
                float(overlap["iou"]) >= 0.72
                or (
                    min(float(overlap["first_coverage"]), float(overlap["second_coverage"]))
                    >= 0.82
                    and float(overlap["area_ratio"]) <= 2.5
                )
            )
            printed_detail = (
                owner.kind == "product"
                and proposal.kind_hint in {"logo", "icon", "badge"}
                and float(overlap["first_coverage"]) >= 0.90
            )
            if same_asset or printed_detail:
                duplicate_owner = owner
                duplicate_report = overlap
                break
        if duplicate_owner is not None:
            proposal.status = "assigned"
            proposal.owner_ids = [duplicate_owner.element_id]
            duplicate_reason = (
                "Nested packaging detail retained inside product"
                if duplicate_owner.kind == "product" and proposal.kind_hint != "product"
                else "Duplicate semantic matte assigned to existing owner"
            )
            if ambiguity_reasons:
                _downgrade_semantic_owner_for_review(duplicate_owner, ambiguity_reasons)
                semantic_ambiguity_owner_ids.add(duplicate_owner.element_id)
                proposal.reason = f"{duplicate_reason}; manual semantic review required"
            else:
                proposal.reason = duplicate_reason
            proposal.evidence["semantic_deduplication"] = duplicate_report
            duplicate_owner.evidence.append(
                {"proposal_id": proposal.proposal_id, "source": proposal.source}
            )
            semantic_duplicates += 1
            continue

        ys, xs = original_support.canvas_slice((width, height))
        before_pixels = original_support.nonzero_pixels
        remaining_alpha = original_support.alpha.copy()
        remaining_alpha[protected[ys, xs] > 0] = 0
        remaining_pixels = int(np.count_nonzero(remaining_alpha))
        removed_fraction = 1.0 - remaining_pixels / max(1, before_pixels)
        if remaining_pixels < 4 or removed_fraction > 0.35:
            proposal.reason = (
                "Semantic matte overlaps an earlier pixel owner by "
                f"{removed_fraction:.1%}; manual ownership review required"
            )
            proposal.evidence["protected_overlap_fraction"] = round(removed_fraction, 6)
            semantic_overlap_deferrals += 1
            continue
        support = AlphaCrop(
            original_support.left,
            original_support.top,
            remaining_alpha,
        ).tight()
        rgba, extraction = extract_semantic_rgba(image_rgb, support)
        node_counter += 1
        node_id = f"SEMANTIC_{node_counter:05d}"
        label = str(proposal.evidence.get("label") or proposal.kind_hint).strip()
        sam_score = float(proposal.evidence.get("sam_iou_score", proposal.confidence))
        policy = proposal.evidence.get("semantic_policy") or {}
        policy_accepted = bool(policy.get("accepted", True))
        mask_move_safe = (
            not ambiguity_reasons
            and policy_accepted
            and proposal.confidence >= 0.54
            and sam_score >= 0.68
            and removed_fraction <= 0.05
            and extraction["component_count"] <= 64
        )
        atomicity_policy = (
            _semantic_product_atomicity_policy(proposal)
            if proposal.kind_hint == "product"
            else None
        )
        atomicity_confirmed = bool(
            atomicity_policy is None
            or atomicity_policy.get("atomic_leaf_confirmed") is True
        )
        auto = mask_move_safe and atomicity_confirmed
        z_base = {
            "product": 900_000,
            "ribbon": 1_030_000,
            "decoration": 1_040_000,
            "badge": 1_100_000,
            "logo": 1_110_000,
            "icon": 1_120_000,
        }[proposal.kind_hint]
        node = ElementNode(
            node_id,
            f"{proposal.kind_hint.upper()} {label[:56]}",
            proposal.kind_hint,
            support,
            z_base + node_counter,
            semantic_envelope=original_support,
            full_support=support,
            removal_footprint=_removal_footprint(support, (width, height)),
            confidence=proposal.confidence,
            review_status="auto_confirmed" if auto else "unresolved",
            # A clean compound can be moved as one visible group even when its
            # internal physical-instance decomposition still needs review.
            move_safe=mask_move_safe,
            rgba=rgba,
            evidence=[{"proposal_id": proposal.proposal_id, "source": proposal.source}],
            metadata={
                "semantic_extraction": extraction,
                "protected_overlap_fraction": round(removed_fraction, 6),
                "source_detector_bbox": proposal.evidence.get("dino_bbox"),
                "refinement": proposal.evidence.get("refinement"),
                "semantic_ambiguity_reasons": ambiguity_reasons,
                "semantic_auto_confirmable": not ambiguity_reasons,
                "semantic_kind_consistent": not any(
                    reason.startswith("semantic label kind") for reason in ambiguity_reasons
                ),
                **(
                    {
                        "semantic_atomicity": atomicity_policy,
                        "semantic_atomicity_requires_review": not atomicity_confirmed,
                        "move_scope": "whole_visible_union",
                    }
                    if atomicity_policy is not None
                    else {}
                ),
            },
        )
        nodes.append(node)
        semantic_nodes.append(node)
        if ambiguity_reasons:
            semantic_ambiguity_owner_ids.add(node_id)
        if atomicity_policy is not None:
            proposal.evidence["semantic_atomicity"] = atomicity_policy
            if not atomicity_confirmed:
                semantic_atomicity_review_owner_ids.add(node_id)
                atomicity_summary = (
                    "Product matte is move-safe only as one visible union; "
                    "physical instance atomicity requires review"
                )
                proposal.reason = (
                    f"{proposal.reason}; {atomicity_summary}"
                    if proposal.reason
                    else atomicity_summary
                )
            if atomicity_policy.get("classification") == "compound_subassembly":
                semantic_compound_owner_ids.add(node_id)
        proposal.status = "assigned"
        proposal.owner_ids = [node_id]
        ys, xs = support.canvas_slice((width, height))
        protected[ys, xs] = np.maximum(protected[ys, xs], support.alpha)

    text_failures = 0
    text_owners: list[tuple[ProposalRecord, ElementNode]] = []
    for detected in [item for item in inventory.proposals if item.record.kind_hint == "text"]:
        proposal = detected.record
        recognized = str(proposal.evidence.get("recognized_text") or "").strip()
        rgba, crop, extraction = extract_text_rgba(
            image_rgb,
            proposal.bbox,
            layerd_alpha_hint=layerd_alpha_hint,
            recognized_text=recognized or None,
        )
        # Text proposals may overlap (Tesseract line/word variants). The first
        # higher-confidence owner keeps each pixel; later proposals remain in
        # the ledger and go to review if no independent ink remains.
        ys, xs = crop.canvas_slice((width, height))
        local_protected = protected[ys, xs]
        full_alpha = crop.alpha.copy()
        protected_overlap_pixels = int(
            np.count_nonzero((full_alpha > 0) & (local_protected > 0))
        )
        protected_overlap_fraction = protected_overlap_pixels / max(
            1, int(np.count_nonzero(full_alpha))
        )
        full_alpha[local_protected > 0] = 0
        rgba[:, :, 3] = full_alpha
        rgba[full_alpha == 0, :3] = 0
        if not np.any(full_alpha):
            proposal.reason = "OCR geometry overlaps an earlier confirmed QR/text owner; manual review required"
            text_failures += 1
            continue
        crop = AlphaCrop(crop.left, crop.top, full_alpha)
        # The editable crop follows actual ink, not the whole OCR rectangle.
        # The proposal bbox remains in the ledger as the semantic envelope.
        crop, rgba = _merge_rgba_parts(((crop, rgba),))
        node_counter += 1
        node_id = f"TEXT_{node_counter:04d}"
        kind: ElementKind = "price" if _PRICE_RE.search(recognized) else "text"
        purity_source = extraction.get("text_purity")
        if not isinstance(purity_source, dict):
            _ignored, purity_source = _assess_text_mask_purity(crop.alpha)
            purity_source["status"] = "unsafe"
            purity_source["reasons"] = list(
                dict.fromkeys(
                    [
                        *purity_source.get("reasons", []),
                        "missing_text_purity_evidence_from_extractor",
                    ]
                )
            )
        purity = dict(purity_source)
        purity["reasons"] = [str(reason) for reason in purity.get("reasons", [])]
        purity["metrics"] = dict(purity.get("metrics") or {})
        purity["protected_overlap_pixels"] = protected_overlap_pixels
        purity["protected_overlap_fraction"] = round(protected_overlap_fraction, 6)
        purity["residual_text_component_count"] = int(
            cv2.connectedComponents((crop.alpha > 0).astype(np.uint8), 8)[0] - 1
        )
        purity["metrics"]["protected_overlap_pixels"] = protected_overlap_pixels
        purity["metrics"]["protected_overlap_fraction"] = round(
            protected_overlap_fraction, 6
        )
        purity["metrics"]["pixels_after_protected_subtraction"] = crop.nonzero_pixels
        if protected_overlap_pixels:
            purity["reasons"].append("overlaps_existing_protected_owner")
        purity["reasons"] = list(dict.fromkeys(purity["reasons"]))
        purity["status"] = "pass" if not purity["reasons"] else "unsafe"
        extraction["text_purity"] = purity
        extraction["nonzero_pixels_after_protected_subtraction"] = crop.nonzero_pixels
        plausible = 0.002 <= extraction["occupancy"] <= 0.72
        auto = (
            plausible
            and purity["status"] == "pass"
            and proposal.confidence >= 0.35
            and crop.nonzero_pixels >= 2
        )
        node = ElementNode(
            node_id,
            f"{kind.upper()} {recognized[:48] or node_counter}",
            kind,
            crop,
            1_500_000 + node_counter,
            semantic_envelope=crop,
            full_support=crop,
            removal_footprint=_removal_footprint(crop, (width, height)),
            confidence=min(0.98, 0.45 + proposal.confidence * 0.45),
            review_status="auto_confirmed" if auto else "unresolved",
            move_safe=auto,
            text=recognized or None,
            rgba=rgba,
            evidence=[{"proposal_id": proposal.proposal_id, "source": proposal.source}],
            metadata={"text_extraction": extraction, "text_purity": purity},
        )
        nodes.append(node)
        proposal.status = "assigned"
        proposal.owner_ids = [node_id]
        protected[ys, xs] = np.maximum(local_protected, full_alpha)
        text_owners.append((proposal, node))

    raw_component_count = 0
    raw_cluster_node_count = 0
    raw_cluster_safety_deferral_count = 0
    raw_geometry_reference_deferral_count = 0
    raw_cluster_safety_deferral_reasons: dict[str, int] = {}
    raw_atomic_review_node_count = 0
    raw_atomic_review_group_count = 0
    raw_text_absorbed_component_count = 0
    raw_rejected_speck_count = 0
    for raw in layerd.foregrounds_bottom_to_top:
        components = split_raw_layer_components(raw, subtract_alpha=protected)
        pending: list[_RawComponent] = []
        for component_index, (crop, rgba, component_report) in enumerate(components, 1):
            raw_component_count += 1
            layerd_proposal_id = (
                f"LAYERD_I{raw.source_iteration:02d}_C{component_index:05d}"
            )
            raw_component = _RawComponent(
                crop=crop,
                rgba=rgba,
                report=component_report,
                proposal_id=layerd_proposal_id,
                component_index=component_index,
            )
            ledger = ProposalRecord(
                proposal_id=layerd_proposal_id,
                source="layerd_iterative_top_layer",
                kind_hint="unknown",
                bbox=crop.bbox,
                confidence=0.45,
                evidence=component_report,
            )
            # A one/two-pixel LayerD island at document scale is numerical
            # matte noise, not a useful editable element. It is still fully
            # accounted for as an explicit rejected proposal.
            if crop.nonzero_pixels <= 2:
                ledger.status = "rejected"
                ledger.reason = "LayerD matte speck has at most two supported pixels"
                ledger.evidence["smallest_useful_policy"] = "explicit_speck_rejection"
                graph.add_proposal(ledger)
                raw_rejected_speck_count += 1
                continue

            text_owner = _text_absorption_owner(raw_component, text_owners)
            if text_owner is not None:
                _merge_component_into_node(text_owner, raw_component, (width, height))
                text_owner.evidence.append(
                    {
                        "proposal_id": layerd_proposal_id,
                        "source": "layerd_omitted_text_mark",
                    }
                )
                ledger.kind_hint = text_owner.kind
                ledger.confidence = min(0.95, max(0.62, text_owner.confidence))
                ledger.status = "assigned"
                ledger.owner_ids = [text_owner.element_id]
                ledger.reason = "Small colour-compatible island absorbed into OCR text owner"
                ledger.evidence["text_mark_absorption"] = True
                graph.add_proposal(ledger)
                ys, xs = crop.canvas_slice((width, height))
                protected[ys, xs] = np.maximum(protected[ys, xs], crop.alpha)
                raw_text_absorbed_component_count += 1
                continue

            graph.add_proposal(ledger)
            pending.append(raw_component)

        for cluster_index, cluster in enumerate(
            _cluster_raw_components(pending, (width, height)), 1
        ):
            raw_cluster_node_count += 1
            crop, rgba = _merge_rgba_parts(
                (item.crop, item.rgba) for item in cluster
            )
            component_report = {
                "source_iteration": raw.source_iteration,
                "source_z_index": raw.z_index,
                "member_component_count": len(cluster),
                "member_proposal_ids": [item.proposal_id for item in cluster],
                "binary_area": sum(item.crop.nonzero_pixels for item in cluster),
                "soft_pixel_count": sum(
                    int(item.report.get("soft_pixel_count", 0)) for item in cluster
                ),
                "consolidation": (
                    "nearby colour-compatible LayerD islands grouped into one useful residual element"
                ),
            }
            kind, confidence, source_ids = _classify_component(
                crop,
                rgba,
                (width, height),
                inventory.proposals,
            )
            auto_confirmation_policy = _raw_cluster_auto_confirmation_policy(
                cluster,
                crop,
                (width, height),
                kind=kind,
                confidence=confidence,
            )
            if not bool(auto_confirmation_policy["eligible"]):
                raw_cluster_safety_deferral_count += 1
                geometry_reference_missing = (
                    "missing_clean_geometry_reference"
                    in auto_confirmation_policy["reasons"]
                )
                if geometry_reference_missing:
                    raw_geometry_reference_deferral_count += 1
                for reason in auto_confirmation_policy["reasons"]:
                    raw_cluster_safety_deferral_reasons[reason] = (
                        raw_cluster_safety_deferral_reasons.get(reason, 0) + 1
                    )
                non_reference_reasons = [
                    reason
                    for reason in auto_confirmation_policy["reasons"]
                    if reason != "missing_clean_geometry_reference"
                ]
                if geometry_reference_missing and not non_reference_reasons:
                    # The cluster itself is coherent; only cleanliness is
                    # unproven. Preserve the useful line/frame/panel grouping
                    # and exact pixels, but never advertise it as movable.
                    node_counter += 1
                    node_id = f"ELEMENT_{node_counter:05d}"
                    node = ElementNode(
                        node_id,
                        f"{kind.upper()} {node_counter:03d} - clean reference required",
                        kind,
                        crop,
                        raw.z_index * 10_000 + cluster_index,
                        full_support=crop,
                        confidence=confidence,
                        review_status="unresolved",
                        move_safe=False,
                        rgba=rgba,
                        evidence=[
                            *(
                                {
                                    "proposal_id": item.proposal_id,
                                    "source": "layerd_iterative_top_layer",
                                }
                                for item in cluster
                            ),
                            *(
                                {
                                    "proposal_id": source_id,
                                    "source": "inventory_match",
                                }
                                for source_id in source_ids
                            ),
                        ],
                        metadata={
                            **component_report,
                            "auto_confirmation_policy": auto_confirmation_policy,
                            "clean_geometry_reference": auto_confirmation_policy[
                                "geometry_reference"
                            ],
                            "reference_only_deferral": True,
                        },
                    )
                    nodes.append(node)
                    for item in cluster:
                        ledger = next(
                            proposal
                            for proposal in graph.proposals
                            if proposal.proposal_id == item.proposal_id
                        )
                        ledger.kind_hint = kind
                        ledger.confidence = confidence
                        ledger.status = "assigned"
                        ledger.owner_ids = [node_id]
                        ledger.reason = (
                            "LayerD/raw geometry has no independently clean "
                            "reference; grouped pixels retained for manual review"
                        )
                        ledger.evidence["cluster_member_count"] = len(cluster)
                        ledger.evidence["cluster_bbox"] = list(crop.bbox)
                        ledger.evidence["cluster_auto_confirmation_eligible"] = False
                        ledger.evidence["cluster_safety_deferral_reasons"] = list(
                            auto_confirmation_policy["reasons"]
                        )
                        ledger.evidence["clean_geometry_reference"] = dict(
                            auto_confirmation_policy["geometry_reference"]
                        )
                        ledger.evidence["atomic_split_due_to_rejected_cluster"] = False
                    continue
                review_group_id = (
                    f"LAYERD_REVIEW_I{raw.source_iteration:02d}_G{cluster_index:05d}"
                )
                atomic_material: list[_RawComponent] = []
                for item in cluster:
                    atomic_material.extend(_split_raw_component_into_final_pieces(item))
                atomic_material.sort(
                    key=lambda item: (
                        item.crop.top,
                        item.crop.left,
                        item.proposal_id,
                        int(item.report.get("final_piece_index", 0)),
                    )
                )
                if not atomic_material:
                    raise RuntimeError(
                        f"Rejected LayerD cluster {review_group_id} has no atomic pixels."
                    )

                owners_by_proposal: dict[str, list[str]] = {
                    item.proposal_id: [] for item in cluster
                }
                classifications_by_proposal: dict[
                    str, list[tuple[ElementKind, float]]
                ] = {item.proposal_id: [] for item in cluster}
                for atomic_index, item in enumerate(atomic_material, 1):
                    atomic_kind, atomic_confidence, atomic_source_ids = _classify_component(
                        item.crop,
                        item.rgba,
                        (width, height),
                        inventory.proposals,
                    )
                    node_counter += 1
                    node_id = f"ELEMENT_{node_counter:05d}"
                    node = ElementNode(
                        node_id,
                        f"{atomic_kind.upper()} {node_counter:03d} - atomic {atomic_index:03d}",
                        atomic_kind,
                        item.crop,
                        # Atomic siblings are disjoint, so retaining one source
                        # z position preserves the original composite without
                        # inventing an ordering between unrelated pixels.
                        raw.z_index * 10_000 + cluster_index,
                        full_support=item.crop,
                        confidence=atomic_confidence,
                        review_status="unresolved",
                        move_safe=False,
                        rgba=item.rgba,
                        evidence=[
                            {
                                "proposal_id": item.proposal_id,
                                "source": "layerd_iterative_top_layer",
                            },
                            *(
                                {"proposal_id": source_id, "source": "inventory_match"}
                                for source_id in atomic_source_ids
                            ),
                        ],
                        metadata={
                            **item.report,
                            "member_component_count": 1,
                            "source_cluster_member_count": len(cluster),
                            "source_cluster_bbox": list(crop.bbox),
                            "source_cluster_kind": kind,
                            "source_cluster_confidence": round(float(confidence), 6),
                            "atomic_split_from_rejected_cluster": True,
                            "pixel_union_exported": False,
                            "review_group_id": review_group_id,
                            "review_group_member_index": atomic_index,
                            "review_group_member_count": len(atomic_material),
                            "review_group_policy": (
                                "related review items only; pixels remain separate atomic nodes"
                            ),
                            "auto_confirmation_policy": auto_confirmation_policy,
                            "clean_geometry_reference": auto_confirmation_policy[
                                "geometry_reference"
                            ],
                        },
                    )
                    nodes.append(node)
                    owners_by_proposal[item.proposal_id].append(node_id)
                    classifications_by_proposal[item.proposal_id].append(
                        (atomic_kind, atomic_confidence)
                    )

                raw_atomic_review_node_count += len(atomic_material)
                raw_atomic_review_group_count += 1

                for item in cluster:
                    ledger = next(
                        proposal
                        for proposal in graph.proposals
                        if proposal.proposal_id == item.proposal_id
                    )
                    classifications = classifications_by_proposal[item.proposal_id]
                    classified_kinds = {value[0] for value in classifications}
                    ledger.kind_hint = (
                        next(iter(classified_kinds))
                        if len(classified_kinds) == 1
                        else "unknown"
                    )
                    ledger.confidence = max(
                        (value[1] for value in classifications),
                        default=0.45,
                    )
                    ledger.status = "assigned"
                    ledger.owner_ids = owners_by_proposal[item.proposal_id]
                    ledger.reason = (
                        "LayerD/raw geometry has no independently clean reference; "
                        "assigned to separate connected review pieces"
                        if geometry_reference_missing
                        else "Cluster coherence failed; assigned to separate connected review pieces"
                    )
                    ledger.evidence["cluster_member_count"] = len(cluster)
                    ledger.evidence["cluster_bbox"] = list(crop.bbox)
                    ledger.evidence["cluster_auto_confirmation_eligible"] = False
                    ledger.evidence["cluster_safety_deferral_reasons"] = list(
                        auto_confirmation_policy["reasons"]
                    )
                    ledger.evidence["clean_geometry_reference"] = dict(
                        auto_confirmation_policy["geometry_reference"]
                    )
                    ledger.evidence["atomic_split_due_to_rejected_cluster"] = True
                    ledger.evidence["atomic_owner_count"] = len(ledger.owner_ids)
                    ledger.evidence["atomic_review_group_id"] = review_group_id
                continue

            node_counter += 1
            node_id = f"ELEMENT_{node_counter:05d}"
            node = ElementNode(
                node_id,
                f"{kind.upper()} {node_counter:03d}",
                kind,
                crop,
                raw.z_index * 10_000 + cluster_index,
                full_support=crop,
                confidence=confidence,
                review_status="auto_confirmed",
                move_safe=True,
                rgba=rgba,
                evidence=[
                    *(
                        {
                            "proposal_id": item.proposal_id,
                            "source": "layerd_iterative_top_layer",
                        }
                        for item in cluster
                    ),
                    *({"proposal_id": item, "source": "inventory_match"} for item in source_ids),
                ],
                metadata={
                    **component_report,
                    "auto_confirmation_policy": auto_confirmation_policy,
                },
            )
            nodes.append(node)
            for item in cluster:
                ledger = next(
                    proposal
                    for proposal in graph.proposals
                    if proposal.proposal_id == item.proposal_id
                )
                ledger.kind_hint = kind
                ledger.confidence = confidence
                ledger.status = "assigned"
                ledger.owner_ids = [node_id]
                ledger.evidence["cluster_member_count"] = len(cluster)
                ledger.evidence["cluster_bbox"] = list(crop.bbox)
                ledger.evidence["cluster_auto_confirmation_eligible"] = True
                ledger.evidence["cluster_safety_deferral_reasons"] = []

    # Assign every independent inventory proposal to the best atomic node when
    # spatial evidence is strong. Otherwise it stays unresolved and is shown
    # in the review UI; there is no silent threshold discard.
    for proposal in graph.proposals:
        if proposal.status != "unresolved":
            continue
        if proposal.source.startswith("grounding_dino") and not bool(
            proposal.evidence.get("auto_extractable", False)
        ):
            # Geometry/safety-deferred open-vocabulary detections are evidence
            # for the reviewer, not permission to attach a giant false box to
            # whichever LayerD component happens to overlap it.
            continue
        if proposal.kind_hint in {"frame", "line"}:
            # Layout bboxes are only geometric hypotheses. They may be much
            # larger than a contained glyph/island, so only a node already
            # proven to have matching geometry may own them here. The dedicated
            # geometry backend can materialise the remaining proposals later.
            geometry_candidates: list[tuple[float, ElementNode]] = []
            for node in nodes:
                if node.kind != proposal.kind_hint:
                    continue
                intersection = _intersection(proposal.bbox, node.bbox)
                if not intersection:
                    continue
                union = _box_area(proposal.bbox) + _box_area(node.bbox) - intersection
                iou = intersection / max(1, union)
                extent = min(_box_area(proposal.bbox), _box_area(node.bbox)) / max(
                    1, max(_box_area(proposal.bbox), _box_area(node.bbox))
                )
                if iou >= 0.30 and extent >= 0.34:
                    geometry_candidates.append((max(iou, extent), node))
            geometry_candidates.sort(key=lambda item: item[0], reverse=True)
            if geometry_candidates:
                score, owner = geometry_candidates[0]
                proposal.status = "assigned"
                proposal.owner_ids = [owner.element_id]
                proposal.evidence["geometry_assignment_score"] = round(score, 6)
                owner.evidence.append(
                    {"proposal_id": proposal.proposal_id, "source": proposal.source}
                )
            continue
        candidates = sorted(
            ((_proposal_match_score(proposal, node), node) for node in nodes),
            key=lambda item: item[0],
            reverse=True,
        )
        if candidates and candidates[0][0] >= (0.72 if proposal.source.startswith("opencv_residual") else 0.52):
            score, owner = candidates[0]
            proposal.status = "assigned"
            proposal.owner_ids = [owner.element_id]
            proposal.evidence["spatial_assignment_score"] = round(score, 6)
            owner.evidence.append(
                {"proposal_id": proposal.proposal_id, "source": proposal.source}
            )

    _make_visible_ownership(nodes, (width, height))
    for node in nodes:
        graph.add_node(node)
    graph.metadata.update(
        {
            "layerd": layerd.report,
            "inventory_detectors": inventory.detector_reports,
            "ownership_policy": (
                "Visible alpha is exclusive to the topmost nonzero support. "
                "Full support is retained separately for correct occlusion/z-order export."
            ),
        }
    )
    graph.validate()
    report = {
        "node_count": len(nodes),
        "semantic_node_count": len(semantic_nodes),
        "semantic_node_kinds": {
            kind: sum(node.kind == kind for node in semantic_nodes)
            for kind in sorted(semantic_kinds)
        },
        "semantic_duplicate_proposal_count": semantic_duplicates,
        "semantic_overlap_deferral_count": semantic_overlap_deferrals,
        "semantic_empty_failure_count": semantic_empty_failures,
        "semantic_ambiguity_proposal_count": semantic_ambiguity_proposals,
        "semantic_refinement_review_count": semantic_refinement_reviews,
        "semantic_kind_mismatch_review_count": semantic_kind_mismatch_reviews,
        "semantic_ambiguity_owner_ids": sorted(semantic_ambiguity_owner_ids),
        "semantic_atomicity_review_count": len(semantic_atomicity_review_owner_ids),
        "semantic_atomicity_review_owner_ids": sorted(
            semantic_atomicity_review_owner_ids
        ),
        "semantic_compound_subassembly_count": len(semantic_compound_owner_ids),
        "semantic_compound_subassembly_owner_ids": sorted(
            semantic_compound_owner_ids
        ),
        "text_node_count": sum(node.kind in {"text", "price"} for node in nodes),
        "qr_node_count": sum(node.kind == "qr" for node in nodes),
        "raw_component_count": raw_component_count,
        "raw_cluster_node_count": raw_cluster_node_count,
        "raw_cluster_safety_deferral_count": raw_cluster_safety_deferral_count,
        "raw_geometry_reference_deferral_count": raw_geometry_reference_deferral_count,
        "raw_cluster_safety_deferral_reasons": raw_cluster_safety_deferral_reasons,
        "raw_atomic_review_group_count": raw_atomic_review_group_count,
        "raw_atomic_review_node_count": raw_atomic_review_node_count,
        "raw_text_absorbed_component_count": raw_text_absorbed_component_count,
        "raw_rejected_speck_count": raw_rejected_speck_count,
        "text_proposals_without_independent_ink": text_failures,
        "unresolved_node_count": len(graph.unresolved_nodes()),
        "proposal_accounting": graph.proposal_accounting(),
        "protected_pixel_count": int(np.count_nonzero(protected)),
    }
    return FusionResult(graph, layerd.background_rgb.copy(), report)
