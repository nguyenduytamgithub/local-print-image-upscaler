from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from v5lib.formats import save_color_png, write_json

from .schema import DocumentGraph, ElementNode
from .text_export_preflight import TEXT_EXPORT_PURITY_POLICY


_FAILED_POLICY_STATES = {
    "blocked",
    "error",
    "fail",
    "failed",
    "fallback",
    "manual_review",
    "manual_review_required",
    "needs_review",
    "rejected",
    "unsafe",
}

_GEOMETRY_KINDS = {"panel", "frame", "line"}
_GEOMETRY_CLEANLINESS_POLICY = "reference_surface_delta_e_carve_v1"
_GEOMETRY_REFERENCE_TYPES = {"surface_rgb", "constant_colour_rgb"}
_GEOMETRY_METRIC_DOMAIN = "visible_core_alpha_gte_192_excluding_known_children"
_SOURCE_REMAINDER_ROLE = "exact_source_remainder_above_clean_base"
_TEXT_PURITY_POLICY = "source_mask_text_purity_v1"
_POLICY_EPSILON = 1.0e-5


def _normalise_policy_state(value: object) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _node_sources(node: ElementNode) -> tuple[str, ...]:
    return tuple(
        str(item.get("source", "")).strip().casefold()
        for item in node.evidence
        if isinstance(item, dict)
    )


def _is_geometry_cleanliness_subject(node: ElementNode) -> bool:
    """Return whether an inferred geometry-like node needs a clean reference.

    Hand-authored/manual nodes are not forced through an inference policy. The
    gate covers poster geometry, residual surfaces and LayerD geometry guesses,
    including legacy cached nodes created before explicit cleanliness evidence
    existed.
    """

    if node.kind not in _GEOMETRY_KINDS:
        return False
    metadata = node.metadata if isinstance(node.metadata, dict) else {}
    sources = _node_sources(node)
    return bool(
        node.synthesized_hidden_pixels
        or metadata.get("geometry_backend") == "poster_geometry_v2"
        or metadata.get("surface_reconciliation") is True
        or any(
            source.startswith(
                (
                    "poster_geometry",
                    "poster_surface_residual",
                    "layerd",
                    "opencv_",
                    "hough",
                )
            )
            for source in sources
        )
    )


def _source_remainder_union(graph: DocumentGraph) -> np.ndarray:
    width, height = graph.canvas_size
    union = np.zeros((height, width), dtype=bool)
    for node in graph.nodes:
        metadata = node.metadata if isinstance(node.metadata, dict) else {}
        if (
            node.review_status == "rejected"
            or metadata.get("role") != _SOURCE_REMAINDER_ROLE
        ):
            continue
        support = node.full_support or node.visible_alpha
        union |= support.to_canvas(graph.canvas_size) > 0
    return union


def _geometry_cleanliness_flags(
    node: ElementNode,
    *,
    source_remainder_union: np.ndarray | None,
) -> list[dict[str, Any]]:
    if not _is_geometry_cleanliness_subject(node):
        return []
    metadata = node.metadata if isinstance(node.metadata, dict) else {}
    policy = metadata.get("geometry_cleanliness")
    if not isinstance(policy, dict):
        return [
            {
                "code": "missing_geometry_cleanliness_policy",
                "source": "metadata.geometry_cleanliness",
                "geometry_backend": metadata.get("geometry_backend"),
                "node_sources": list(_node_sources(node)),
            }
        ]

    flags: list[dict[str, Any]] = []
    invalid_fields: list[str] = []
    if policy.get("policy") != _GEOMETRY_CLEANLINESS_POLICY:
        invalid_fields.append("policy")
    if _normalise_policy_state(policy.get("status")) != "pass":
        invalid_fields.append("status")
    if policy.get("reference_type") not in _GEOMETRY_REFERENCE_TYPES:
        invalid_fields.append("reference_type")
    if policy.get("metric_domain") != _GEOMETRY_METRIC_DOMAIN:
        invalid_fields.append("metric_domain")
    if policy.get("carved_destination_role") != _SOURCE_REMAINDER_ROLE:
        invalid_fields.append("carved_destination_role")

    numeric_fields = (
        "threshold_delta_e76",
        "growth_threshold_delta_e76",
        "remaining_visible_delta_e76_limit",
        "dilation_radius_px",
        "support_pixel_count",
        "carved_pixel_count",
        "carved_pixel_fraction_of_support",
        "high_delta_seed_fraction_of_support",
        "max_auto_safe_high_delta_fraction",
        "known_child_excluded_pixel_count",
        "remaining_visible_pixel_count",
        "remaining_visible_delta_e76_p95",
        "remaining_visible_delta_e76_max",
    )
    numbers = {key: _finite_number(policy.get(key)) for key in numeric_fields}
    invalid_fields.extend(key for key, value in numbers.items() if value is None)
    nonnegative_fields = set(numeric_fields)
    for key in nonnegative_fields:
        value = numbers.get(key)
        if value is not None and value < 0:
            invalid_fields.append(key)
    fraction = numbers.get("carved_pixel_fraction_of_support")
    if fraction is not None and fraction > 1.0 + _POLICY_EPSILON:
        invalid_fields.append("carved_pixel_fraction_of_support")
    high_delta_fraction = numbers.get("high_delta_seed_fraction_of_support")
    max_high_delta_fraction = numbers.get("max_auto_safe_high_delta_fraction")
    for key, value in (
        ("high_delta_seed_fraction_of_support", high_delta_fraction),
        ("max_auto_safe_high_delta_fraction", max_high_delta_fraction),
    ):
        if value is not None and value > 1.0 + _POLICY_EPSILON:
            invalid_fields.append(key)
    for key in (
        "dilation_radius_px",
        "support_pixel_count",
        "carved_pixel_count",
        "known_child_excluded_pixel_count",
        "remaining_visible_pixel_count",
    ):
        value = numbers.get(key)
        if value is not None and abs(value - round(value)) > _POLICY_EPSILON:
            invalid_fields.append(key)
    if invalid_fields:
        flags.append(
            {
                "code": "invalid_geometry_cleanliness_policy",
                "source": "metadata.geometry_cleanliness",
                "invalid_fields": sorted(set(invalid_fields)),
                "policy": policy.get("policy"),
                "status": policy.get("status"),
            }
        )

    limit = numbers.get("remaining_visible_delta_e76_limit")
    threshold = numbers.get("threshold_delta_e76")
    p95 = numbers.get("remaining_visible_delta_e76_p95")
    maximum = numbers.get("remaining_visible_delta_e76_max")
    if limit is not None and (
        (p95 is not None and p95 > limit + _POLICY_EPSILON)
        or (maximum is not None and maximum > limit + _POLICY_EPSILON)
    ):
        flags.append(
            {
                "code": "geometry_remaining_visible_contamination",
                "source": "metadata.geometry_cleanliness",
                "limit_delta_e76": limit,
                "remaining_visible_delta_e76_p95": p95,
                "remaining_visible_delta_e76_max": maximum,
            }
        )

    contradictions: list[str] = []
    if (
        limit is not None
        and threshold is not None
        and abs(limit - threshold) > _POLICY_EPSILON
    ):
        contradictions.append("remaining_limit_differs_from_reference_threshold")
    support = numbers.get("support_pixel_count")
    carved = numbers.get("carved_pixel_count")
    children = numbers.get("known_child_excluded_pixel_count")
    remaining = numbers.get("remaining_visible_pixel_count")
    if support is not None and carved is not None:
        if support <= 0:
            contradictions.append("support_pixel_count_is_not_positive")
        if carved > support + _POLICY_EPSILON:
            contradictions.append("carved_pixel_count_exceeds_support")
        expected_fraction = carved / support if support > 0 else 0.0
        if fraction is not None and abs(fraction - expected_fraction) > _POLICY_EPSILON:
            contradictions.append("carved_fraction_does_not_match_counts")
    # Remaining-visible is measured only on the opaque core while support also
    # includes antialias pixels, so those counts must not be forced to sum.
    # They still cannot individually exceed the reported source support.
    if support is not None and children is not None and children > support + _POLICY_EPSILON:
        contradictions.append("known_child_exclusion_exceeds_support")
    if support is not None and remaining is not None and remaining > support + _POLICY_EPSILON:
        contradictions.append("remaining_visible_count_exceeds_support")
    if (
        high_delta_fraction is not None
        and max_high_delta_fraction is not None
        and high_delta_fraction > max_high_delta_fraction + _POLICY_EPSILON
    ):
        contradictions.append("high_delta_seed_fraction_exceeds_auto_safe_limit")
    if contradictions:
        flags.append(
            {
                "code": "geometry_cleanliness_accounting_contradiction",
                "source": "metadata.geometry_cleanliness",
                "contradictions": contradictions,
                "support_pixel_count": support,
                "carved_pixel_count": carved,
                "known_child_excluded_pixel_count": children,
                "remaining_visible_pixel_count": remaining,
                "carved_pixel_fraction_of_support": fraction,
            }
        )

    if (
        carved is not None
        and carved > 0
        and source_remainder_union is not None
    ):
        full = (node.full_support or node.visible_alpha).to_canvas(
            (source_remainder_union.shape[1], source_remainder_union.shape[0])
        )
        visible = node.visible_alpha.to_canvas(
            (source_remainder_union.shape[1], source_remainder_union.shape[0])
        )
        # The geometry backend can carve antialiased support down to alpha 64,
        # not only the alpha>=192 metric core.  The exact technical remainder
        # is binary complement-of-visible ownership, so compare every fully
        # removed support pixel rather than undercounting antialias carving.
        hidden_support = (full > 0) & (visible == 0)
        carved_in_remainder = int(np.count_nonzero(hidden_support & source_remainder_union))
        if carved_in_remainder + _POLICY_EPSILON < carved:
            flags.append(
                {
                    "code": "geometry_carve_source_remainder_contradiction",
                    "source": "metadata.geometry_cleanliness+source_remainder",
                    "reported_carved_pixel_count": int(round(carved)),
                    "hidden_core_pixels_in_source_remainder": carved_in_remainder,
                    "destination_role": policy.get("carved_destination_role"),
                }
            )
    return flags


def _geometry_full_support_topology_report(
    node: ElementNode,
) -> dict[str, Any] | None:
    """Validate the shape that is actually exported for editable geometry.

    ``visible_alpha`` is an ownership/compositing mask and can legitimately be
    empty when a synthesized panel, frame or rule is fully occluded.  It is
    therefore unsafe evidence for editability QA.  This audit deliberately
    inspects ``full_support`` (the alpha written to PNG/PSD/ORA) and only falls
    back to visible alpha when the node has no separate synthesized support.

    The checks are topological rather than semantic: a frame must be one
    coherent ring with a dominant interior opening, a panel must be one solid
    filled surface, and a line must be one thin axis-coherent stroke.  Tiny
    antialias islands and pinholes are reported but ignored by the substantial
    component/hole thresholds.
    """

    if node.kind not in _GEOMETRY_KINDS:
        return None

    export_crop = node.full_support or node.visible_alpha
    raw = (export_crop.alpha > 0).astype(np.uint8)
    ys, xs = np.where(raw > 0)
    if not len(xs):
        return {
            "policy": "exported_full_support_topology_v1",
            "status": "fail",
            "alpha_source": (
                "full_support" if node.full_support is not None else "visible_alpha_fallback"
            ),
            "kind": node.kind,
            "reasons": ["empty_export_support"],
            "metrics": {
                "crop_width": int(raw.shape[1]),
                "crop_height": int(raw.shape[0]),
                "support_pixel_count": 0,
                "component_count": 0,
                "substantial_component_count": 0,
                "hole_count": 0,
                "meaningful_hole_count": 0,
            },
        }

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    binary = np.ascontiguousarray(raw[y0:y1, x0:x1])
    height, width = binary.shape
    tight_area = int(height * width)
    support_pixels = int(np.count_nonzero(binary))

    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    component_areas = sorted(
        (int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, component_count)),
        reverse=True,
    )
    component_noise_limit = max(4, int(np.ceil(support_pixels * 0.005)))
    substantial_components = [
        area for area in component_areas if area >= component_noise_limit
    ]
    largest_component_fraction = (
        component_areas[0] / support_pixels if component_areas else 0.0
    )

    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    inverse = (padded == 0).astype(np.uint8)
    inverse_count, inverse_labels, inverse_stats, _inverse_centroids = (
        cv2.connectedComponentsWithStats(inverse, connectivity=4)
    )
    outside_label = int(inverse_labels[0, 0])
    hole_areas = sorted(
        (
            int(inverse_stats[index, cv2.CC_STAT_AREA])
            for index in range(1, inverse_count)
            if index != outside_label
        ),
        reverse=True,
    )
    hole_noise_limit = max(4, int(np.ceil(tight_area * 0.001)))
    meaningful_holes = [area for area in hole_areas if area >= hole_noise_limit]
    total_hole_pixels = int(sum(hole_areas))
    dominant_hole_area = int(hole_areas[0]) if hole_areas else 0
    dominant_hole_fraction = dominant_hole_area / max(1, tight_area)
    dominant_hole_share = dominant_hole_area / max(1, total_hole_pixels)

    min_dimension = min(height, width)
    edge_band = max(2, min(24, int(round(min_dimension * 0.12))))
    rows, columns = np.indices(binary.shape)
    edge_depth = np.minimum.reduce(
        (rows, columns, height - 1 - rows, width - 1 - columns)
    )
    deep_support_pixels = int(
        np.count_nonzero((binary > 0) & (edge_depth > edge_band))
    )
    deep_support_fraction = deep_support_pixels / max(1, support_pixels)

    major_dimension = max(height, width)
    minor_dimension = min_dimension
    aspect_ratio = major_dimension / max(1, minor_dimension)
    if width >= height:
        cross_sections = np.count_nonzero(binary, axis=0)
    else:
        cross_sections = np.count_nonzero(binary, axis=1)
    occupied_cross_sections = cross_sections[cross_sections > 0]
    major_axis_span_fraction = len(occupied_cross_sections) / max(1, major_dimension)
    median_cross_section = (
        float(np.median(occupied_cross_sections))
        if len(occupied_cross_sections)
        else 0.0
    )
    cross_section_outlier_limit = max(4.0, median_cross_section * 2.5)
    cross_section_outlier_fraction = (
        float(np.count_nonzero(cross_sections > cross_section_outlier_limit))
        / max(1, major_dimension)
    )

    reasons: list[str] = []
    substantial_count = len(substantial_components)
    meaningful_hole_count = len(meaningful_holes)
    support_fraction = support_pixels / max(1, tight_area)
    if substantial_count != 1 or largest_component_fraction < 0.95:
        reasons.append("support_is_not_one_coherent_component")

    if node.kind == "frame":
        if meaningful_hole_count < 1:
            reasons.append("frame_missing_meaningful_interior_hole")
        if dominant_hole_fraction < 0.25:
            reasons.append("frame_missing_dominant_interior_hole")
        if meaningful_hole_count > 1 and dominant_hole_share < 0.80:
            reasons.append("frame_has_multiple_competing_interior_holes")
        if (
            min_dimension >= 16
            and deep_support_pixels >= max(32, int(round(support_pixels * 0.05)))
            and deep_support_fraction > 0.18
        ):
            reasons.append("frame_has_large_deep_interior_intrusion")
    elif node.kind == "panel":
        if largest_component_fraction < 0.98:
            reasons.append("panel_has_detached_support")
        if meaningful_hole_count:
            reasons.append("panel_has_meaningful_interior_holes")
        if support_fraction < 0.55:
            reasons.append("panel_is_not_a_coherent_filled_surface")
    else:  # line
        if largest_component_fraction < 0.97:
            reasons.append("line_has_detached_support")
        if meaningful_hole_count:
            reasons.append("line_has_meaningful_interior_holes")
        if major_dimension >= 12 and aspect_ratio < 3.0:
            reasons.append("line_is_not_axis_elongated")
        if support_fraction < 0.25 or major_axis_span_fraction < 0.80:
            reasons.append("line_does_not_form_one_continuous_stroke")
        thickness_limit = max(8, int(np.ceil(major_dimension * 0.08)))
        if major_dimension >= 24 and minor_dimension > thickness_limit:
            reasons.append("line_is_too_thick_for_its_span")
        if cross_section_outlier_fraction > 0.20:
            reasons.append("line_has_large_cross_axis_intrusion")

    metrics: dict[str, Any] = {
        "crop_width": int(raw.shape[1]),
        "crop_height": int(raw.shape[0]),
        "tight_width": width,
        "tight_height": height,
        "support_pixel_count": support_pixels,
        "support_fraction_of_tight_bbox": round(float(support_fraction), 8),
        "component_count": len(component_areas),
        "substantial_component_count": substantial_count,
        "component_noise_limit_pixels": component_noise_limit,
        "component_areas_desc": component_areas[:16],
        "largest_component_fraction": round(float(largest_component_fraction), 8),
        "hole_count": len(hole_areas),
        "meaningful_hole_count": meaningful_hole_count,
        "hole_noise_limit_pixels": hole_noise_limit,
        "hole_areas_desc": hole_areas[:16],
        "dominant_hole_fraction_of_tight_bbox": round(
            float(dominant_hole_fraction), 8
        ),
        "dominant_hole_share": round(float(dominant_hole_share), 8),
        "edge_band_pixels": edge_band,
        "deep_support_pixel_count": deep_support_pixels,
        "deep_support_fraction": round(float(deep_support_fraction), 8),
        "major_to_minor_aspect_ratio": round(float(aspect_ratio), 8),
        "major_axis_span_fraction": round(float(major_axis_span_fraction), 8),
        "median_cross_section_pixels": round(float(median_cross_section), 8),
        "cross_section_outlier_fraction": round(
            float(cross_section_outlier_fraction), 8
        ),
    }
    return {
        "policy": "exported_full_support_topology_v1",
        "status": "fail" if reasons else "pass",
        "alpha_source": (
            "full_support" if node.full_support is not None else "visible_alpha_fallback"
        ),
        "kind": node.kind,
        "reasons": list(dict.fromkeys(reasons)),
        "metrics": metrics,
    }


def geometry_full_support_topology_report(
    node: ElementNode,
) -> dict[str, Any] | None:
    """Public authority for the topology of an exported geometry support.

    Export preflight and final QA must call the same implementation.  Keeping
    this small public entry point avoids a second, subtly different geometry
    classifier at the export boundary.
    """

    return _geometry_full_support_topology_report(node)


def enforce_geometry_export_topology_preflight(
    graph: DocumentGraph,
) -> dict[str, Any]:
    """Fail closed before export when editable geometry is structurally bad.

    Geometry detection can have strong colour/cleanliness evidence while its
    final ``full_support`` still represents a compound object, a detached
    collection, or a solid banner mislabelled as a frame.  Final QA correctly
    rejects those shapes, but discovering the contradiction only after PSD and
    ORA creation is too late.  This preflight evaluates the exact export alpha
    with the final-QA policy and removes automatic safety from every failure.

    Proposal assignment remains truthful: a detector proposal may still be
    assigned to a useful review layer even though that layer is not safe to
    move automatically.  The proposal ledger therefore records the topology
    result as owner evidence instead of falsely changing assignment status.
    """

    graph.validate()
    ledger_key = "owner_exported_full_support_topology"
    for proposal in graph.proposals:
        if isinstance(proposal.evidence, dict):
            proposal.evidence.pop(ledger_key, None)

    owner_proposals: dict[str, list[Any]] = {}
    for proposal in graph.proposals:
        for owner_id in proposal.owner_ids:
            owner_proposals.setdefault(owner_id, []).append(proposal)

    records: list[dict[str, Any]] = []
    passed_ids: list[str] = []
    failed_ids: list[str] = []
    downgraded_ids: list[str] = []
    skipped_rejected_ids: list[str] = []
    for node in sorted(graph.nodes, key=lambda item: (item.z_index, item.element_id)):
        if node.review_status == "rejected":
            if node.kind in _GEOMETRY_KINDS:
                skipped_rejected_ids.append(node.element_id)
            node.metadata.pop("exported_full_support_topology", None)
            continue
        topology = geometry_full_support_topology_report(node)
        if topology is None:
            node.metadata.pop("exported_full_support_topology", None)
            continue

        previous_gate = node.metadata.get("exported_full_support_topology")
        before_status = node.review_status
        before_move_safe = bool(node.move_safe)
        if isinstance(previous_gate, dict):
            first_status = str(
                previous_gate.get("review_status_before_first_gate", before_status)
            )
            first_move_safe = bool(
                previous_gate.get("move_safe_before_first_gate", before_move_safe)
            )
        else:
            first_status = before_status
            first_move_safe = before_move_safe

        failed = topology.get("status") != "pass"
        first_claimed_automatic_safety = (
            first_status == "auto_confirmed" or first_move_safe
        )
        claimed_automatic_safety_now = (
            before_status == "auto_confirmed" or before_move_safe
        )
        if failed:
            node.review_status = "unresolved"
            node.move_safe = False
            failed_ids.append(node.element_id)
            if first_claimed_automatic_safety or claimed_automatic_safety_now:
                downgraded_ids.append(node.element_id)
                action = "downgraded_to_unresolved_non_move_safe"
            else:
                action = "retained_unresolved_non_move_safe"
        else:
            passed_ids.append(node.element_id)
            action = "topology_passed_status_retained"

        gate_record = {
            **topology,
            "node_id": node.element_id,
            "review_status_before_first_gate": first_status,
            "move_safe_before_first_gate": first_move_safe,
            "review_status_before_this_gate": before_status,
            "move_safe_before_this_gate": before_move_safe,
            "review_status_after_gate": node.review_status,
            "move_safe_after_gate": bool(node.move_safe),
            "action": action,
        }
        node.metadata["exported_full_support_topology"] = gate_record

        compact_ledger_record = {
            "policy": topology.get("policy"),
            "status": topology.get("status"),
            "kind": topology.get("kind"),
            "reasons": list(topology.get("reasons", [])),
            "review_status_after_gate": node.review_status,
            "move_safe_after_gate": bool(node.move_safe),
        }
        for proposal in owner_proposals.get(node.element_id, []):
            owner_results = proposal.evidence.get(ledger_key)
            if not isinstance(owner_results, dict):
                owner_results = {}
            owner_results[node.element_id] = dict(compact_ledger_record)
            proposal.evidence[ledger_key] = {
                key: owner_results[key] for key in sorted(owner_results)
            }

        records.append(
            {
                "node_id": node.element_id,
                "name": node.name,
                **compact_ledger_record,
                "action": action,
            }
        )

    summary = {
        "policy": "exported_full_support_topology_v1",
        "status": "review_required" if failed_ids else "pass",
        "checked_node_count": len(records),
        "passed_node_count": len(passed_ids),
        "failed_node_count": len(failed_ids),
        "passed_node_ids": passed_ids,
        "failed_node_ids": failed_ids,
        "downgraded_node_ids": sorted(set(downgraded_ids)),
        "skipped_rejected_node_ids": skipped_rejected_ids,
        "records": records,
        "enforcement": (
            "Every failed geometry support is unresolved and non-move-safe; "
            "proposal assignment is retained with explicit owner topology evidence."
        ),
    }
    graph.metadata["geometry_export_topology_preflight"] = summary
    geometry_metadata = graph.metadata.get("geometry")
    if isinstance(geometry_metadata, dict):
        geometry_metadata["export_topology_preflight"] = summary
    graph.validate()
    return summary


def _geometry_full_support_topology_flags(node: ElementNode) -> list[dict[str, Any]]:
    report = geometry_full_support_topology_report(node)
    if report is None or report["status"] == "pass":
        return []
    return [
        {
            "code": "geometry_full_support_topology_rejected_auto_safety",
            "source": "exported.full_support",
            **report,
        }
    ]


def _text_purity_flags(node: ElementNode) -> list[dict[str, Any]]:
    if node.kind not in {"text", "price"}:
        return []
    metadata = node.metadata if isinstance(node.metadata, dict) else {}
    policy = metadata.get("text_purity")
    if not isinstance(policy, dict):
        return [
            {
                "code": "missing_text_purity_policy",
                "source": "metadata.text_purity",
                "node_sources": list(_node_sources(node)),
            }
        ]

    invalid_fields: list[str] = []
    if policy.get("policy") != _TEXT_PURITY_POLICY:
        invalid_fields.append("policy")
    if _normalise_policy_state(policy.get("status")) != "pass":
        invalid_fields.append("status")
    reasons = policy.get("reasons")
    if not isinstance(reasons, list):
        invalid_fields.append("reasons")
        reasons = []
    for key in ("giant_component", "multi_band", "adjacent_object"):
        if policy.get(key) is not False:
            invalid_fields.append(key)

    count_fields = (
        "protected_overlap_pixels",
        "residual_text_component_count",
    )
    counts = {key: _finite_number(policy.get(key)) for key in count_fields}
    for key, value in counts.items():
        if (
            value is None
            or value < 0
            or abs(value - round(value)) > _POLICY_EPSILON
        ):
            invalid_fields.append(key)
    overlap_fraction = _finite_number(policy.get("protected_overlap_fraction"))
    if (
        overlap_fraction is None
        or overlap_fraction < 0
        or overlap_fraction > 1.0 + _POLICY_EPSILON
    ):
        invalid_fields.append("protected_overlap_fraction")

    metrics = policy.get("metrics")
    metric_fields = (
        "occupancy",
        "component_count",
        "largest_component_fraction",
        "largest_component_bbox_fraction",
        "largest_component_fill",
        "row_band_count",
        "primary_band_fraction",
        "out_of_band_fraction",
        "suspicious_component_count",
    )
    metric_values: dict[str, float | None] = {}
    if not isinstance(metrics, dict):
        invalid_fields.append("metrics")
    else:
        metric_values = {key: _finite_number(metrics.get(key)) for key in metric_fields}
        for key, value in metric_values.items():
            if value is None or value < 0:
                invalid_fields.append(f"metrics.{key}")
        for key in (
            "occupancy",
            "largest_component_fraction",
            "largest_component_bbox_fraction",
            "largest_component_fill",
            "primary_band_fraction",
            "out_of_band_fraction",
        ):
            value = metric_values.get(key)
            if value is not None and value > 1.0 + _POLICY_EPSILON:
                invalid_fields.append(f"metrics.{key}")
        for key in ("component_count", "row_band_count", "suspicious_component_count"):
            value = metric_values.get(key)
            if value is not None and abs(value - round(value)) > _POLICY_EPSILON:
                invalid_fields.append(f"metrics.{key}")

    risk_reasons = [str(item) for item in reasons]
    risk_flags = {
        key: policy.get(key)
        for key in ("giant_component", "multi_band", "adjacent_object")
        if policy.get(key) is True
    }
    protected_overlap = counts.get("protected_overlap_pixels")
    if protected_overlap is not None and protected_overlap > 0:
        risk_reasons.append("protected_overlap_pixels_nonzero")
    if overlap_fraction is not None and overlap_fraction > _POLICY_EPSILON:
        risk_reasons.append("protected_overlap_fraction_nonzero")
    suspicious = metric_values.get("suspicious_component_count")
    if suspicious is not None and suspicious > 0:
        risk_reasons.append("suspicious_component_count_nonzero")

    flags: list[dict[str, Any]] = []
    if invalid_fields:
        flags.append(
            {
                "code": "invalid_text_purity_policy",
                "source": "metadata.text_purity",
                "invalid_fields": sorted(set(invalid_fields)),
                "policy": policy.get("policy"),
                "status": policy.get("status"),
            }
        )
    if risk_reasons or risk_flags:
        flags.append(
            {
                "code": "text_purity_rejected_auto_confirmation",
                "source": "metadata.text_purity",
                "reasons": list(dict.fromkeys(risk_reasons)),
                "risk_flags": risk_flags,
                "protected_overlap_pixels": protected_overlap,
                "protected_overlap_fraction": overlap_fraction,
            }
        )
    export_policy = metadata.get("text_export_purity_preflight")
    if not isinstance(export_policy, dict):
        flags.append(
            {
                "code": "missing_exact_export_text_purity_preflight",
                "source": "metadata.text_export_purity_preflight",
                "node_sources": list(_node_sources(node)),
            }
        )
        return flags

    export_invalid_fields: list[str] = []
    if export_policy.get("policy") != TEXT_EXPORT_PURITY_POLICY:
        export_invalid_fields.append("policy")
    if _normalise_policy_state(export_policy.get("status")) != "pass":
        export_invalid_fields.append("status")
    export_reasons = export_policy.get("reasons")
    if not isinstance(export_reasons, list):
        export_invalid_fields.append("reasons")
        export_reasons = []
    alpha_pixels = _finite_number(export_policy.get("exact_export_alpha_pixels"))
    if (
        alpha_pixels is None
        or alpha_pixels < 1
        or abs(alpha_pixels - round(alpha_pixels)) > _POLICY_EPSILON
    ):
        export_invalid_fields.append("exact_export_alpha_pixels")

    core = export_policy.get("foreground_core")
    if not isinstance(core, dict):
        export_invalid_fields.append("foreground_core")
        core = {}
    elif core.get("passed") is not True:
        export_invalid_fields.append("foreground_core.passed")
    retention = _finite_number(core.get("retained_fraction"))
    if retention is None or retention < 0 or retention > 1.0 + _POLICY_EPSILON:
        export_invalid_fields.append("foreground_core.retained_fraction")

    edge = export_policy.get("edge_band_evidence")
    if not isinstance(edge, dict):
        export_invalid_fields.append("edge_band_evidence")
        edge = {}
    elif edge.get("passed") is not True:
        export_invalid_fields.append("edge_band_evidence.passed")
    isolated = edge.get("isolated_edge_components")
    if not isinstance(isolated, list):
        export_invalid_fields.append("edge_band_evidence.isolated_edge_components")
        isolated = []

    topology = export_policy.get("topology")
    if not isinstance(topology, dict):
        export_invalid_fields.append("topology")
        topology = {}
    elif topology.get("implausibly_fragmented") is not False:
        export_invalid_fields.append("topology.implausibly_fragmented")

    if export_invalid_fields:
        flags.append(
            {
                "code": "invalid_exact_export_text_purity_preflight",
                "source": "metadata.text_export_purity_preflight",
                "invalid_fields": sorted(set(export_invalid_fields)),
                "policy": export_policy.get("policy"),
                "status": export_policy.get("status"),
            }
        )
    export_risks = [str(item) for item in export_reasons]
    if isolated:
        export_risks.append("isolated_edge_components_nonempty")
    if topology.get("implausibly_fragmented") is True:
        export_risks.append("implausibly_fragmented_glyph_topology")
    if core.get("passed") is False:
        export_risks.append("foreground_core_not_retained")
    if export_risks:
        flags.append(
            {
                "code": "exact_export_text_purity_rejected_auto_confirmation",
                "source": "metadata.text_export_purity_preflight",
                "reasons": list(dict.fromkeys(export_risks)),
                "foreground_core": core,
                "isolated_edge_components": isolated,
                "topology": topology,
            }
        )
    return flags


def _node_auto_safety_flags(
    node: ElementNode,
    *,
    source_remainder_union: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Return explicit evidence that contradicts automatic move safety.

    This gate deliberately consumes backend metadata instead of trying to
    re-run segmentation policy in QA.  A fallback matte may still be useful as
    a review candidate, but it cannot be advertised as automatically safe.
    Likewise, a LayerD cluster whose policy rejected automatic confirmation
    remains exportable only as an unresolved review item.
    """

    metadata = node.metadata if isinstance(node.metadata, dict) else {}
    flags = [
        *_geometry_cleanliness_flags(
            node,
            source_remainder_union=source_remainder_union,
        ),
        *_geometry_full_support_topology_flags(node),
        *_text_purity_flags(node),
    ]

    refinement = metadata.get("refinement")
    if isinstance(refinement, dict):
        failed_fields = [
            key
            for key in ("accepted", "passed", "success")
            if key in refinement and refinement.get(key) is False
        ]
        state = _normalise_policy_state(
            refinement.get("status", refinement.get("state", ""))
        )
        review_required = bool(
            refinement.get("requires_review")
            or refinement.get("requires_manual_review")
            or refinement.get("manual_review_required")
        )
        if failed_fields or state in _FAILED_POLICY_STATES or review_required:
            flags.append(
                {
                    "code": "failed_refinement_marked_auto_safe",
                    "source": "metadata.refinement",
                    "failed_fields": failed_fields,
                    "state": state or None,
                    "reason": refinement.get("reason"),
                }
            )

    semantic_reasons = metadata.get("semantic_ambiguity_reasons")
    if not isinstance(semantic_reasons, list):
        semantic_reasons = []
    semantic_blocked = (
        metadata.get("semantic_auto_confirmable") is False
        or metadata.get("semantic_kind_consistent") is False
        or metadata.get("requires_manual_review") is True
        or bool(semantic_reasons)
    )
    if semantic_blocked:
        flags.append(
            {
                "code": "semantic_policy_rejected_auto_confirmation",
                "source": "metadata.semantic_*",
                "semantic_auto_confirmable": metadata.get("semantic_auto_confirmable"),
                "semantic_kind_consistent": metadata.get("semantic_kind_consistent"),
                "requires_manual_review": metadata.get("requires_manual_review"),
                "reasons": [str(item) for item in semantic_reasons],
            }
        )

    cluster_policy = metadata.get("auto_confirmation_policy")
    if isinstance(cluster_policy, dict):
        policy_reasons = cluster_policy.get("reasons")
        if not isinstance(policy_reasons, list):
            policy_reasons = []
        policy_state = _normalise_policy_state(
            cluster_policy.get("status", cluster_policy.get("state", ""))
        )
        policy_decision = _normalise_policy_state(cluster_policy.get("decision", ""))
        policy_blocked = (
            cluster_policy.get("eligible") is False
            or bool(policy_reasons)
            or policy_state in _FAILED_POLICY_STATES
            or policy_decision in {"defer", "defer_to_review", "manual_review"}
        )
        if policy_blocked:
            flags.append(
                {
                    "code": "consolidation_policy_rejected_auto_confirmation",
                    "source": "metadata.auto_confirmation_policy",
                    "policy": cluster_policy.get("policy"),
                    "eligible": cluster_policy.get("eligible"),
                    "state": policy_state or None,
                    "decision": policy_decision or None,
                    "reasons": [str(item) for item in policy_reasons],
                }
            )

    # Compatibility guard for inventories created before the explicit LayerD
    # policy record existed.  More than eight disconnected members exceeds the
    # V1 cluster policy's documented automatic-confirmation ceiling.  This is
    # not a substitute for the backend policy; it prevents old metadata from
    # silently bypassing the release gate.
    member_count = metadata.get(
        "member_component_count", metadata.get("member_segment_count")
    )
    try:
        member_count_int = int(member_count)
    except (TypeError, ValueError):
        member_count_int = 0
    explicitly_consolidated = bool(
        metadata.get("consolidated") or metadata.get("consolidation")
    )
    if (
        not isinstance(cluster_policy, dict)
        and explicitly_consolidated
        and member_count_int > 8
    ):
        flags.append(
            {
                "code": "legacy_high_risk_multi_component_consolidation",
                "source": "metadata.member_component_count",
                "member_count": member_count_int,
                "automatic_member_limit": 8,
            }
        )
    return flags


def _auto_safety_gate(
    graph: DocumentGraph,
    *,
    source_remainder_union: np.ndarray | None = None,
) -> dict[str, Any]:
    if source_remainder_union is None:
        source_remainder_union = _source_remainder_union(graph)
    contradictions: list[dict[str, Any]] = []
    for node in sorted(graph.nodes, key=lambda item: (item.z_index, item.element_id)):
        if node.review_status == "rejected":
            continue
        flags = _node_auto_safety_flags(
            node,
            source_remainder_union=source_remainder_union,
        )
        claims_automatic_safety = node.review_status == "auto_confirmed" or node.move_safe
        if flags and claims_automatic_safety:
            contradictions.append(
                {
                    "id": node.element_id,
                    "name": node.name,
                    "review_status": node.review_status,
                    "move_safe": node.move_safe,
                    "flags": flags,
                }
            )
    return {
        "status": "fail" if contradictions else "pass",
        "contradiction_count": len(contradictions),
        "contradictions": contradictions,
        "policy": (
            "A node cannot be auto_confirmed or move_safe when backend metadata "
            "records failed refinement, semantic ambiguity, rejected/high-risk "
            "multi-component consolidation, unproven/contaminated or malformed "
            "export geometry, or impure text ownership."
        ),
    }


def _canvas_alpha(node: ElementNode, canvas_size: tuple[int, int], *, full: bool = False) -> np.ndarray:
    crop = (node.full_support if full else node.visible_alpha) or node.visible_alpha
    return crop.to_canvas(canvas_size)


def _alpha_component_report(node: ElementNode) -> dict[str, Any]:
    alpha = node.visible_alpha.alpha
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (alpha > 0).astype(np.uint8), 8
    )
    component_areas = sorted((int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)), reverse=True)
    # Hole count is descriptive only: counters in O/G/8 are legitimate and
    # must never be filled merely to reduce this number.
    padded = cv2.copyMakeBorder((alpha > 0).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    inverse = (padded == 0).astype(np.uint8)
    inverse_count, inverse_labels = cv2.connectedComponents(inverse, 4)
    outside_label = int(inverse_labels[0, 0])
    holes = sum(
        index != outside_label and np.any(inverse_labels == index)
        for index in range(1, inverse_count)
    )
    return {
        "component_count": len(component_areas),
        "component_areas_desc": component_areas[:64],
        "micro_components_at_most_4px": sum(area <= 4 for area in component_areas),
        "hole_count": int(holes),
    }


def _envelope_escape_pixels(graph: DocumentGraph) -> tuple[int, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    total = 0
    for node in graph.nodes:
        if node.semantic_envelope is None or node.review_status == "rejected":
            continue
        support = _canvas_alpha(node, graph.canvas_size, full=True) > 0
        envelope = node.semantic_envelope.to_canvas(graph.canvas_size) > 0
        escaped = int(np.count_nonzero(support & ~envelope))
        if escaped:
            records.append({"id": node.element_id, "pixels": escaped})
            total += escaped
    return total, records


def build_quality_report(
    graph: DocumentGraph,
    source_rgb: np.ndarray,
    document_background_rgb: np.ndarray,
    composite_rgb: np.ndarray,
    *,
    residual_background_text_regions: int | None = None,
    format_reports: dict[str, dict[str, object]] | None = None,
    cleanplate_report: dict[str, Any] | None = None,
    residual_report: dict[str, Any] | None = None,
    delivery_render_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    width, height = graph.canvas_size
    expected_shape = (height, width, 3)
    for name, value in (
        ("source", source_rgb),
        ("document background", document_background_rgb),
        ("composite", composite_rgb),
    ):
        array = np.asarray(value)
        if array.dtype != np.uint8 or array.shape != expected_shape:
            raise ValueError(f"{name} must be source-canvas uint8 RGB")
    graph.validate()
    owner_count, visible_union = graph.ownership_maps()
    source_remainder_union = _source_remainder_union(graph)
    auto_safety_gate = _auto_safety_gate(
        graph,
        source_remainder_union=source_remainder_union,
    )
    opaque_union = np.zeros((height, width), dtype=bool)
    empty_nodes: list[str] = []
    layer_details: list[dict[str, Any]] = []
    for node in sorted(graph.nodes, key=lambda item: (item.z_index, item.element_id)):
        if node.review_status == "rejected":
            continue
        canvas_alpha = _canvas_alpha(node, graph.canvas_size)
        # The explicit source remainder is a technical fidelity layer above
        # the clean base, not a claimed foreground object. Including it here
        # would make every quiet background pixel look like a baked-in ghost.
        if node.metadata.get("role") != "exact_source_remainder_above_clean_base":
            opaque_union |= canvas_alpha >= 200
        # A lower panel/rule can be fully occluded in the current composition
        # yet still export a non-empty synthesized full support for editing.
        # Only the actual export alpha being empty is a broken node.
        export_alpha = _canvas_alpha(node, graph.canvas_size, full=True)
        if not np.any(export_alpha):
            empty_nodes.append(node.element_id)
        layer_details.append(
            {
                "id": node.element_id,
                "name": node.name,
                "kind": node.kind,
                "review_status": node.review_status,
                "move_safe": node.move_safe,
                "bbox": list(node.bbox),
                "alpha": _alpha_component_report(node),
                "full_support_topology": geometry_full_support_topology_report(node),
                "auto_safety_flags": _node_auto_safety_flags(
                    node,
                    source_remainder_union=source_remainder_union,
                ),
            }
        )
    envelope_escape, envelope_records = _envelope_escape_pixels(graph)
    identical = np.all(document_background_rgb == source_rgb, axis=2)
    unchanged_opaque_pixels = int(np.count_nonzero(identical & opaque_union))
    opaque_pixels = int(np.count_nonzero(opaque_union))
    unchanged_opaque_ratio = unchanged_opaque_pixels / max(1, opaque_pixels)
    recomposition_error = np.abs(composite_rgb.astype(np.int16) - source_rgb.astype(np.int16))
    accounting = graph.proposal_accounting()
    unresolved_nodes = [node.element_id for node in graph.unresolved_nodes()]
    multiply_owned = int(np.count_nonzero(owner_count > 1))
    format_reports = format_reports or {}
    format_failures: list[str] = []
    format_notices: list[str] = []
    for name, report in format_reports.items():
        created = bool(report.get("created", False))
        if not created:
            reason = str(report.get("reason") or "adapter did not create the container")
            if name.casefold() == "psd":
                # PSD has a documented 30,000 px / 1.6 GB safety ceiling.
                # ORA + cropped PNG layers remain lossless, editable delivery
                # formats and a deliberate PSD skip is not a quality failure.
                format_notices.append(f"PSD omitted: {reason}")
            else:
                format_failures.append(name)
            continue
        roundtrip = report.get("roundtrip_qa", report.get("expected_composite_qa", {}))
        if not isinstance(roundtrip, dict) or int(roundtrip.get("max_abs_error", 0)) > 1:
            format_failures.append(name)

    hard_failures: list[str] = []
    review_reasons: list[str] = []
    if auto_safety_gate["contradiction_count"]:
        hard_failures.append(
            f"{auto_safety_gate['contradiction_count']} layer nodes claim automatic move safety "
            "without clean geometry/text evidence or despite explicit failed/refused backend policy evidence"
        )
    if multiply_owned:
        hard_failures.append(f"{multiply_owned} visible pixels have multiple owners")
    if envelope_escape:
        hard_failures.append(f"{envelope_escape} alpha pixels escape semantic envelopes")
    if empty_nodes:
        hard_failures.append(f"{len(empty_nodes)} exported nodes are empty")
    if format_failures:
        hard_failures.append("container round-trip failed: " + ", ".join(format_failures))
    if accounting["unresolved"]:
        review_reasons.append(f"{accounting['unresolved']} proposals are unresolved")
    if unresolved_nodes:
        review_reasons.append(f"{len(unresolved_nodes)} layer nodes need confirmation")
    if residual_background_text_regions is None:
        review_reasons.append("background OCR residual test was not run")
    elif residual_background_text_regions:
        review_reasons.append(
            f"background still contains {residual_background_text_regions} OCR-like regions"
        )
    if unchanged_opaque_ratio > 0.02:
        review_reasons.append(
            f"{unchanged_opaque_ratio:.2%} of opaque foreground cores are byte-identical in background"
        )
    cleanplate_audit = (cleanplate_report or {}).get("clean_plate_audit", {})
    cleanplate_grade = str(cleanplate_audit.get("grade", "not_run"))
    if cleanplate_report is not None:
        if cleanplate_grade == "fail":
            hard_failures.append("clean-plate residual/seam audit failed")
        elif cleanplate_grade != "pass":
            review_reasons.append(f"clean-plate audit grade is {cleanplate_grade}")
    residual_accounting = (residual_report or {}).get("salient_pixel_accounting", {})
    if residual_report is not None:
        residual_total = int(residual_accounting.get("total", 0))
        residual_assigned = int(residual_accounting.get("assigned", 0))
        residual_rejected = int(residual_accounting.get("rejected", 0))
        residual_unaccounted = int(residual_accounting.get("unaccounted", 0))
        if residual_unaccounted:
            hard_failures.append(
                f"{residual_unaccounted} significant residual pixels are unaccounted"
            )
        if residual_assigned + residual_rejected != residual_total:
            hard_failures.append("residual salient-pixel ledger does not balance")
        largest_fraction = float(residual_report.get("largest_visible_fraction", 0.0))
        if largest_fraction > 0.20:
            review_reasons.append(
                f"largest residual layer covers {largest_fraction:.2%} of the canvas"
            )
    max_error = int(recomposition_error.max()) if recomposition_error.size else 0
    if max_error > 1:
        hard_failures.append(f"recomposition max error is {max_error}, limit is 1")
    delivery_error = int((delivery_render_report or {}).get("recomposition_max_abs_error", 0))
    if delivery_error > 1:
        hard_failures.append(
            f"delivery-scale recomposition max error is {delivery_error}, limit is 1"
        )

    status = "FAIL" if hard_failures else ("REVIEW_REQUIRED" if review_reasons else "PASS")
    return {
        "schema": "V5_EDITABILITY_QA_V2",
        "status": status,
        "truth_policy": (
            "PASS requires complete proposal accounting, no unresolved nodes, exclusive ownership, "
            "clean background evidence, exact container round-trip and <=1 recomposition error. "
            "Recomposition alone never proves that layers are complete or uncontaminated."
        ),
        "hard_failures": hard_failures,
        "review_reasons": review_reasons,
        "proposal_accounting": accounting,
        "node_count": len(graph.nodes),
        "unresolved_node_ids": unresolved_nodes,
        "empty_node_ids": empty_nodes,
        "ownership": {
            "visible_union_pixels": int(np.count_nonzero(visible_union)),
            "multiply_owned_pixels": multiply_owned,
            "unassigned_canvas_pixels": int(np.count_nonzero(~visible_union)),
            "note": "Unassigned canvas includes legitimate background; completeness is governed by proposal ledger.",
        },
        "alpha_envelope": {
            "escaped_pixels": envelope_escape,
            "layers": envelope_records,
        },
        "auto_safety_gate": auto_safety_gate,
        "background_ghost": {
            "opaque_foreground_core_pixels": opaque_pixels,
            "byte_identical_core_pixels": unchanged_opaque_pixels,
            "byte_identical_core_ratio": round(unchanged_opaque_ratio, 8),
            "residual_ocr_region_count": residual_background_text_regions,
            "clean_plate_audit": cleanplate_audit,
        },
        "residual_reconciliation": residual_report,
        "recomposition": {
            "max_abs_error": max_error,
            "mean_abs_error": round(float(recomposition_error.mean()), 8) if recomposition_error.size else 0.0,
            "different_pixels": int(np.count_nonzero(np.any(recomposition_error > 0, axis=2))),
        },
        "containers": format_reports,
        "container_notices": format_notices,
        "delivery_render": delivery_render_report,
        "layers": layer_details,
    }


def write_inventory_files(graph: DocumentGraph, output_dir: Path) -> dict[str, Path]:
    json_path = output_dir / "ELEMENT_INVENTORY.json"
    csv_path = output_dir / "ELEMENT_INVENTORY.csv"
    graph_record = graph.manifest_record()
    write_json(json_path, graph_record)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "id",
                "name",
                "kind",
                "parent_id",
                "z_index",
                "review_status",
                "move_safe",
                "confidence",
                "bbox",
                "text",
            ),
        )
        writer.writeheader()
        for node in sorted(graph.nodes, key=lambda item: (item.z_index, item.element_id)):
            writer.writerow(
                {
                    "id": node.element_id,
                    "name": node.name,
                    "kind": node.kind,
                    "parent_id": node.parent_id or "",
                    "z_index": node.z_index,
                    "review_status": node.review_status,
                    "move_safe": node.move_safe,
                    "confidence": f"{node.confidence:.6f}",
                    "bbox": ",".join(str(value) for value in node.bbox),
                    "text": node.text or "",
                }
            )
    return {"json": json_path, "csv": csv_path}


def _numbered_overlay(source_rgb: np.ndarray, graph: DocumentGraph) -> Image.Image:
    image = Image.fromarray(source_rgb, "RGB").copy()
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=13)
    colours = {
        "text": (0, 175, 255, 210),
        "price": (255, 40, 40, 210),
        "product": (60, 220, 100, 210),
        "frame": (255, 150, 0, 210),
        "line": (255, 210, 0, 210),
        "icon": (190, 70, 255, 210),
        "logo": (190, 70, 255, 210),
    }
    for index, node in enumerate(sorted(graph.nodes, key=lambda item: (item.bbox[1], item.bbox[0])), 1):
        if node.review_status == "rejected":
            continue
        x0, y0, x1, y1 = node.bbox
        colour = colours.get(node.kind, (255, 255, 255, 190))
        draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=colour, width=1)
        label = str(index)
        label_box = draw.textbbox((x0, y0), label, font=font, stroke_width=1)
        draw.rectangle(label_box, fill=(0, 0, 0, 190))
        draw.text((x0, y0), label, fill=colour, font=font, stroke_width=1, stroke_fill=(0, 0, 0, 255))
    return image


def write_qa_artifacts(
    output_dir: Path,
    graph: DocumentGraph,
    source_rgb: np.ndarray,
    document_background_rgb: np.ndarray,
    composite_rgb: np.ndarray,
    report: dict[str, Any],
    *,
    icc_profile: bytes | None,
) -> dict[str, Path]:
    technical = output_dir / "_KY_THUAT"
    technical.mkdir(parents=True, exist_ok=True)
    report_path = technical / "QA_REPORT.json"
    write_json(report_path, report)
    html_path = technical / "QA_REPORT.html"
    status = html.escape(str(report.get("status", "UNKNOWN")))
    reasons = [
        *[str(item) for item in report.get("hard_failures", [])],
        *[str(item) for item in report.get("review_reasons", [])],
    ]
    rows = "".join(f"<li>{html.escape(reason)}</li>" for reason in reasons) or "<li>Không có lỗi.</li>"
    layer_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(layer.get('id', '')))}</td>"
        f"<td>{html.escape(str(layer.get('name', '')))}</td>"
        f"<td>{html.escape(str(layer.get('kind', '')))}</td>"
        f"<td>{html.escape(str(layer.get('review_status', '')))}</td>"
        f"<td>{'Có' if layer.get('move_safe') else 'Chưa'}</td>"
        "</tr>"
        for layer in report.get("layers", [])
        if isinstance(layer, dict)
    )
    html_path.write_text(
        "<!doctype html><html lang='vi'><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width'><title>V5 QA</title><style>body{font:15px Segoe UI,Arial;"
        "max-width:1200px;margin:28px auto;padding:0 18px;color:#172033}h1{margin-bottom:4px}"
        ".status{display:inline-block;padding:8px 14px;border-radius:999px;background:#eef2ff;font-weight:700}"
        "table{border-collapse:collapse;width:100%;margin-top:18px}th,td{border:1px solid #d8deea;"
        "padding:7px;text-align:left}th{background:#f4f6fa}code{background:#f3f5f8;padding:2px 5px}</style>"
        f"<h1>Báo cáo kiểm định layer V5</h1><p class='status'>{status}</p>"
        "<p>PASS chỉ xuất hiện khi không còn vùng chưa xử lý, ownership không chồng lấn, nền không còn "
        "bóng nội dung và PSD/ORA mở lại đúng.</p><h2>Điểm cần chú ý</h2>"
        f"<ul>{rows}</ul><h2>Danh sách layer</h2><table><thead><tr><th>ID</th><th>Tên</th>"
        f"<th>Loại</th><th>Duyệt</th><th>Di chuyển an toàn</th></tr></thead><tbody>{layer_rows}</tbody></table>"
        "</html>",
        encoding="utf-8",
    )
    overlay_path = technical / "NUMBERED_LAYER_OVERLAY.png"
    save_color_png(_numbered_overlay(source_rgb, graph), overlay_path, icc_profile)

    owner_count, visible_union = graph.ownership_maps()
    heat = np.zeros((*visible_union.shape, 3), dtype=np.uint8)
    heat[visible_union] = (30, 180, 80)
    heat[owner_count > 1] = (255, 0, 0)
    ownership_path = technical / "OWNERSHIP_HEATMAP.png"
    save_color_png(Image.fromarray(heat, "RGB"), ownership_path, icc_profile)

    diff = np.abs(composite_rgb.astype(np.int16) - source_rgb.astype(np.int16)).astype(np.uint8)
    diff = np.clip(diff.astype(np.uint16) * 8, 0, 255).astype(np.uint8)
    diff_path = technical / "RECOMPOSITION_DIFF_X8.png"
    save_color_png(Image.fromarray(diff, "RGB"), diff_path, icc_profile)

    residual = np.abs(document_background_rgb.astype(np.int16) - source_rgb.astype(np.int16))
    residual = np.clip(residual.astype(np.uint16) * 3, 0, 255).astype(np.uint8)
    residual_path = technical / "BACKGROUND_CHANGE_X3.png"
    save_color_png(Image.fromarray(residual, "RGB"), residual_path, icc_profile)
    return {
        "report": report_path,
        "report_html": html_path,
        "numbered_overlay": overlay_path,
        "ownership_heatmap": ownership_path,
        "recomposition_diff": diff_path,
        "background_change": residual_path,
    }
