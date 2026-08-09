from __future__ import annotations

"""Conservative poster-geometry reconstruction for V5 Pro.

The inventory stage deliberately emits *proposals*, not masks.  This module
turns only well-supported proposals into editable geometry.  A false negative
is reviewable; a false positive can silently drag text or a product into a
frame layer, so every uncertain case is returned as ``unresolved``.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal

import cv2
import numpy as np

from .inventory import DetectedProposal, InventoryResult
from .schema import AlphaCrop, ElementNode, ProposalRecord


GeometryStatus = Literal["accepted", "unresolved", "rejected_duplicate"]
_SUPPORTED_KINDS = frozenset({"frame", "line", "panel", "ribbon", "badge"})
_CLEANLINESS_POLICY = "reference_surface_delta_e_carve_v1"
_CLEANLINESS_CORE_ALPHA = 192
_CLEANLINESS_GROW_ALPHA = 64
_CLEANLINESS_MAX_CARVED_FRACTION = 0.35
_SUPPORT_TOPOLOGY_POLICY = "protected_geometry_support_completion_v1"


@dataclass(slots=True)
class GeometryDecision:
    proposal_id: str
    status: GeometryStatus
    reason: str
    node_ids: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GeometryResult:
    nodes: list[ElementNode]
    proposals: list[ProposalRecord]
    report: dict[str, Any]
    decisions: list[GeometryDecision] = field(default_factory=list)

    def unresolved_ids(self) -> list[str]:
        return [item.proposal_id for item in self.decisions if item.status == "unresolved"]


@dataclass(slots=True)
class _Candidate:
    alpha: np.ndarray
    confidence: float
    metrics: dict[str, Any]
    colour_rgb: np.ndarray | None = None
    surface_rgb: np.ndarray | None = None
    interior_alpha: np.ndarray | None = None
    interior_fit_exclusion: np.ndarray | None = None


def _validate_rgb(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("geometry input must be a uint8 RGB image")
    return np.ascontiguousarray(image)


def _normalise_protected(protected_alpha: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if protected_alpha is None:
        return np.zeros(shape, dtype=np.uint8)
    value = np.asarray(protected_alpha)
    if value.shape != shape or value.ndim != 2:
        raise ValueError("protected_alpha must match the image height and width")
    if value.dtype == bool:
        return value.astype(np.uint8) * 255
    if value.dtype != np.uint8:
        raise ValueError("protected_alpha must use bool or uint8")
    return np.ascontiguousarray(value)


def _lab(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _delta_e(image_lab: np.ndarray, colour_lab: np.ndarray) -> np.ndarray:
    return np.linalg.norm(image_lab - colour_lab.reshape(1, 1, 3), axis=2)


def _palette(image_rgb: np.ndarray, sample: np.ndarray, limit: int = 12) -> list[np.ndarray]:
    pixels = image_rgb[np.asarray(sample, dtype=bool)]
    if not len(pixels):
        return []
    # Sixteen-value bins are stable across antialias noise and deterministic,
    # unlike a random-initialised k-means palette.
    quantised = (pixels.astype(np.uint16) // 16).astype(np.int16)
    keys, inverse, counts = np.unique(quantised, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(counts)[::-1][:limit]
    result: list[np.ndarray] = []
    for index in order:
        members = pixels[inverse == index]
        result.append(np.median(members, axis=0).astype(np.uint8))
    return result


def _component_contours(binary: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], float, float] | None:
    contours, hierarchy = cv2.findContours(
        np.asarray(binary, dtype=np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None or not contours:
        return None
    hierarchy = hierarchy[0]
    outer_indices = [index for index, item in enumerate(hierarchy) if int(item[3]) < 0]
    if not outer_indices:
        return None
    outer_index = max(outer_indices, key=lambda index: abs(float(cv2.contourArea(contours[index]))))
    outer = contours[outer_index]
    outer_area = abs(float(cv2.contourArea(outer)))
    if outer_area < 4:
        return None
    holes: list[np.ndarray] = []
    child = int(hierarchy[outer_index][2])
    while child >= 0:
        holes.append(contours[child])
        child = int(hierarchy[child][0])
    hole_area = sum(abs(float(cv2.contourArea(item))) for item in holes)
    return outer, holes, outer_area, hole_area


def _smooth_polygon(contour: np.ndarray) -> np.ndarray:
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(0.30, perimeter * 0.0015)
    return cv2.approxPolyDP(contour, epsilon, True)


def _render_shape_alpha(
    shape: tuple[int, int],
    outer: np.ndarray,
    holes: Iterable[np.ndarray],
    *,
    scale: int = 4,
) -> np.ndarray:
    height, width = shape
    high = np.zeros((height * scale, width * scale), dtype=np.uint8)

    def scaled(contour: np.ndarray) -> np.ndarray:
        points = _smooth_polygon(contour).astype(np.int64) * scale
        return points.astype(np.int32)

    cv2.fillPoly(high, [scaled(outer)], 255, lineType=cv2.LINE_AA)
    for hole in holes:
        cv2.fillPoly(high, [scaled(hole)], 0, lineType=cv2.LINE_AA)
    return cv2.resize(high, (width, height), interpolation=cv2.INTER_AREA)


def _render_ring_from_interior(
    shape: tuple[int, int],
    interior: np.ndarray,
    stroke_radius: float,
    *,
    scale: int = 4,
) -> np.ndarray:
    """Render a clean ring outside an authoritative interior contour.

    A colour contour can absorb an attached ribbon and inherit glyph-shaped
    notches from the source bitmap.  The dominant interior hole is much more
    stable: dilating it outward reconstructs the actual frame stroke without
    carrying the attached artwork into the frame layer.
    """

    height, width = shape
    high = np.zeros((height * scale, width * scale), dtype=np.uint8)
    points = _smooth_polygon(interior).astype(np.float64)
    points = np.rint(points * scale).astype(np.int32)
    cv2.fillPoly(high, [points], 255, lineType=cv2.LINE_AA)
    radius = max(1, int(round(float(stroke_radius) * scale)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    expanded = cv2.dilate(high, kernel, iterations=1)
    ring = expanded.copy()
    ring[high > 0] = 0
    return cv2.resize(ring, (width, height), interpolation=cv2.INTER_AREA)


def _regularize_frame_interior(
    shape: tuple[int, int],
    interior: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Canonicalize a ribbon-notched top from clean top/bottom evidence.

    The long straight top run supplies its y baseline; the clean bottom edge
    supplies the rounded-corner profile.  Mirroring that bottom profile onto
    the top fills only downward intrusions such as a joined section ribbon.
    If no dominant straight run exists, the frame is left unresolved rather
    than guessed from a convex hull (which would preserve a visible slope).
    """

    raw = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(raw, [np.asarray(interior, dtype=np.int32)], 1)
    raw_pixels = int(np.count_nonzero(raw))
    raw_area = abs(float(cv2.contourArea(interior)))
    if raw_pixels < 16 or raw_area < 8.0:
        return None, {
            "policy": "top_localized_interior_concavity_v1",
            "status": "unsafe",
            "reason": "interior_is_too_small_for_topology_regularization",
        }
    ix, iy, iw, ih = cv2.boundingRect(interior)
    top_by_column = np.full(shape[1], -1, dtype=np.int32)
    bottom_by_column = np.full(shape[1], -1, dtype=np.int32)
    for column in range(ix, ix + iw):
        rows = np.flatnonzero(raw[:, column] > 0)
        if len(rows):
            top_by_column[column] = int(rows[0])
            bottom_by_column[column] = int(rows[-1])
    margin = max(3, int(round(iw * 0.04)))
    central_columns = np.arange(ix + margin, ix + iw - margin, dtype=np.int32)
    central_columns = central_columns[top_by_column[central_columns] >= 0]
    if len(central_columns) < max(12, int(round(iw * 0.30))):
        return None, {
            "policy": "axis_aligned_top_boundary_completion_v1",
            "status": "unsafe",
            "reason": "insufficient_interior_columns_for_top_boundary_fit",
        }
    top_values = top_by_column[central_columns]
    bottom_values = bottom_by_column[central_columns]
    top_baseline = int(round(float(np.percentile(top_values, 20))))
    bottom_baseline = int(round(float(np.percentile(bottom_values, 80))))
    top_support_fraction = float(np.mean(np.abs(top_values - top_baseline) <= 1))
    bottom_support_fraction = float(
        np.mean(np.abs(bottom_values - bottom_baseline) <= 1)
    )

    expected_top = np.full(shape[1], -1, dtype=np.int32)
    valid_columns = np.flatnonzero(bottom_by_column >= 0)
    expected_top[valid_columns] = top_baseline + np.maximum(
        0, bottom_baseline - bottom_by_column[valid_columns]
    )
    deviations = top_by_column[valid_columns] - expected_top[valid_columns]
    intruded_columns = valid_columns[deviations > 1]
    intrusion_pixels = int(
        np.sum(np.maximum(0, deviations), dtype=np.int64)
    )
    intrusion_span_fraction = (
        float((int(intruded_columns.max()) - int(intruded_columns.min()) + 1) / iw)
        if len(intruded_columns)
        else 0.0
    )
    negative_deviation_fraction = float(np.mean(deviations < -2))
    pre_residual_p95 = float(
        np.percentile(np.abs(deviations), 95) if len(deviations) else 0.0
    )
    report: dict[str, Any] = {
        "policy": "axis_aligned_top_boundary_completion_v1",
        "status": "pass",
        "original_interior_pixel_count": raw_pixels,
        "top_baseline_y_local": top_baseline,
        "bottom_baseline_y_local": bottom_baseline,
        "top_baseline_support_column_fraction": round(top_support_fraction, 8),
        "bottom_baseline_support_column_fraction": round(bottom_support_fraction, 8),
        "pre_regularization_top_residual_p95_px": round(pre_residual_p95, 5),
        "intruded_column_count": int(len(intruded_columns)),
        "intrusion_horizontal_span_fraction": round(intrusion_span_fraction, 8),
        "negative_top_deviation_column_fraction": round(
            negative_deviation_fraction, 8
        ),
    }
    insignificant = intrusion_pixels < max(24, int(round(raw_pixels * 0.0005)))
    if insignificant:
        report.update(
            {
                "model": "observed_axis_aligned_interior",
                "regularized_pixel_count": 0,
                "post_regularization_top_residual_p95_px": round(
                    pre_residual_p95, 5
                ),
            }
        )
        return interior, report
    if (
        top_support_fraction >= 0.20
        and bottom_support_fraction >= 0.30
        and intrusion_span_fraction >= 0.05
        and intrusion_pixels <= int(round(raw_pixels * 0.12))
        and negative_deviation_fraction <= 0.05
    ):
        regularized = raw.copy()
        for column in intruded_columns:
            regularized[
                expected_top[column] : top_by_column[column], column
            ] = 1
        contours, _hierarchy = cv2.findContours(
            regularized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) != 1:
            report.update(
                {
                    "status": "unsafe",
                    "model": "unresolved_top_completion_changed_component_count",
                    "reason": "top completion did not produce one interior component",
                    "regularized_pixel_count": 0,
                }
            )
            return None, report
        fitted = contours[0]
        fitted_mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(fitted_mask, [fitted], 1)
        fitted_top = np.full(shape[1], -1, dtype=np.int32)
        for column in valid_columns:
            rows = np.flatnonzero(fitted_mask[:, column] > 0)
            if len(rows):
                fitted_top[column] = int(rows[0])
        post = fitted_top[valid_columns] - expected_top[valid_columns]
        post_residual_p95 = float(
            np.percentile(np.abs(post), 95) if len(post) else 0.0
        )
        if post_residual_p95 > 1.0:
            report.update(
                {
                    "status": "unsafe",
                    "model": "unresolved_top_completion_residual",
                    "reason": "canonical top boundary residual exceeds one pixel",
                    "regularized_pixel_count": intrusion_pixels,
                    "post_regularization_top_residual_p95_px": round(
                        post_residual_p95, 5
                    ),
                }
            )
            return None, report
        report.update(
            {
                "model": "bottom_mirrored_canonical_top_completion",
                "regularized_pixel_count": intrusion_pixels,
                "post_regularization_top_residual_p95_px": round(
                    post_residual_p95, 5
                ),
            }
        )
        return fitted, report
    report.update(
        {
            "status": "unsafe",
            "model": "unresolved_noncanonical_top_boundary",
            "reason": "frame top has no safe axis-aligned completion evidence",
            "regularized_pixel_count": 0,
            "post_regularization_top_residual_p95_px": round(
                pre_residual_p95, 5
            ),
        }
    )
    return None, report


def _authoritative_frame_ring(
    observed_component: np.ndarray,
    observed_alpha: np.ndarray,
    dominant_interior: np.ndarray,
    protected: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fit a clean frame ring from side/bottom evidence around its main hole."""

    observed = np.asarray(observed_component, dtype=bool)
    protected_binary = np.asarray(protected) > 0
    interior_alpha = _render_shape_alpha(observed.shape, dominant_interior, [])
    interior = interior_alpha >= 128
    if not np.any(interior):
        return None, {
            "policy": "dominant_interior_outward_ring_v1",
            "status": "unsafe",
            "reason": "dominant_interior_is_empty",
        }
    ix, iy, iw, ih = cv2.boundingRect(dominant_interior)
    if iw < 8 or ih < 8:
        return None, {
            "policy": "dominant_interior_outward_ring_v1",
            "status": "unsafe",
            "reason": "dominant_interior_is_too_small",
        }
    outside = (~interior).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    yy, _xx = np.indices(observed.shape)
    # Top edges commonly carry an attached section ribbon.  Estimate the
    # stroke only from the clean side/bottom majority and later validate the
    # reconstructed ring there.
    top_guard = min(observed.shape[0], iy + max(3, int(round(ih * 0.12))))
    clean_side_bottom = yy >= top_guard
    search_radius = int(np.clip(round(min(iw, ih) * 0.08), 4, 24))
    sample_domain = (
        observed
        & ~interior
        & clean_side_bottom
        & ~protected_binary
        & (distance > 0.0)
        & (distance <= float(search_radius))
    )
    samples = distance[sample_domain]
    minimum_samples = max(24, int(round((iw + ih) * 0.06)))
    if len(samples) < minimum_samples:
        return None, {
            "policy": "dominant_interior_outward_ring_v1",
            "status": "unsafe",
            "reason": "insufficient_clean_side_bottom_stroke_samples",
            "sample_pixel_count": int(len(samples)),
            "minimum_sample_pixel_count": minimum_samples,
            "search_radius_px": search_radius,
        }
    validation_domain = (
        clean_side_bottom
        & ~protected_binary
        & ~interior
        & (distance > 0.0)
        & (distance <= float(search_radius))
    )
    observed_validation = observed & validation_domain
    observed_pixels = int(np.count_nonzero(observed_validation))
    best_radius: float | None = None
    best_score = -1.0
    for stroke_radius_candidate in np.arange(0.75, search_radius + 0.001, 0.25):
        model_candidate = validation_domain & (
            distance <= float(stroke_radius_candidate)
        )
        model_count = int(np.count_nonzero(model_candidate))
        if not model_count:
            continue
        intersection = int(np.count_nonzero(model_candidate & observed_validation))
        precision = intersection / model_count
        recall = intersection / max(1, observed_pixels)
        f1 = 2.0 * precision * recall / max(1.0e-9, precision + recall)
        if f1 > best_score + 1.0e-9:
            best_score = f1
            best_radius = float(stroke_radius_candidate)
    stroke_radius = best_radius if best_radius is not None else float("nan")
    if not np.isfinite(stroke_radius) or not 0.75 <= stroke_radius <= search_radius:
        return None, {
            "policy": "dominant_interior_outward_ring_v1",
            "status": "unsafe",
            "reason": "invalid_stroke_radius",
            "stroke_radius_px": stroke_radius,
            "search_radius_px": search_radius,
        }
    ring_alpha = _render_ring_from_interior(
        observed.shape,
        dominant_interior,
        stroke_radius,
    )
    # Supersampled dilation deliberately antialiases past its nominal radius.
    # Clamp that fringe back to the distance-field fit before validating or
    # exporting it; otherwise a one-pixel real border can become a two-pixel
    # synthetic border and the model appears unsupported despite a good fit.
    ring_alpha[
        (distance <= 0.0) | (distance > float(stroke_radius + 0.10))
    ] = 0
    if observed_alpha.shape != observed.shape or observed_alpha.dtype != np.uint8:
        raise ValueError("observed frame alpha must match its component")
    model_ring = ring_alpha >= 32
    expected = model_ring & validation_domain
    intersection = int(np.count_nonzero(expected & observed_validation))
    model_pixels = int(np.count_nonzero(expected))
    model_recall = intersection / max(1, model_pixels)
    observed_precision = intersection / max(1, observed_pixels)
    if (
        model_pixels < minimum_samples
        or model_recall < 0.50
        or observed_precision < 0.58
    ):
        return None, {
            "policy": "dominant_interior_outward_ring_v1",
            "status": "unsafe",
            "reason": "side_bottom_ring_model_does_not_match_observed_frame",
            "stroke_radius_px": round(stroke_radius, 5),
            "sample_pixel_count": int(len(samples)),
            "binary_fit_f1": round(best_score, 6),
            "validation_model_pixel_count": model_pixels,
            "validation_observed_pixel_count": observed_pixels,
            "validation_intersection_pixel_count": intersection,
            "validation_model_recall": round(model_recall, 6),
            "validation_observed_precision": round(observed_precision, 6),
        }
    # Keep the fitted model everywhere so a one-pixel gap in the observed
    # contour cannot turn an exported frame into an open path.  Union only the
    # observed antialiasing that remains close to the fitted stroke; attached
    # ribbons sit farther from the dominant interior and are therefore cut.
    observed_allowed = observed_alpha.copy()
    observed_allowed[distance > float(stroke_radius + 1.5)] = 0
    # The top band is synthesized exclusively from the canonical straight
    # interior model.  Once that boundary has been repaired, reintroducing
    # source pixels here would bake the attached ribbon back into the frame.
    observed_allowed[:top_guard, :] = 0
    np.maximum(ring_alpha, observed_allowed, out=ring_alpha)
    ring = ring_alpha >= 32
    original = observed
    synthesized = ring & ~original
    removed_attachment = original & ~ring
    return ring_alpha, {
        "policy": "dominant_interior_outward_ring_v1",
        "status": "pass",
        "reference": "dominant_interior_hole",
        "stroke_radius_px": round(stroke_radius, 5),
        "search_radius_px": search_radius,
        "sample_pixel_count": int(len(samples)),
        "binary_fit_f1": round(best_score, 6),
        "validation_model_pixel_count": model_pixels,
        "validation_observed_pixel_count": observed_pixels,
        "validation_intersection_pixel_count": intersection,
        "validation_model_recall": round(model_recall, 6),
        "validation_observed_precision": round(observed_precision, 6),
        "original_support_pixel_count": int(np.count_nonzero(original)),
        "reconstructed_support_pixel_count": int(np.count_nonzero(ring)),
        "synthesized_support_pixel_count": int(np.count_nonzero(synthesized)),
        "removed_attached_content_pixel_count": int(np.count_nonzero(removed_attachment)),
        "preserved_structural_hole_pixel_count": int(np.count_nonzero(interior)),
    }


def _complete_protected_support_gaps(
    alpha: np.ndarray,
    protected: np.ndarray,
    *,
    model: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill only child-owned, locally bounded gaps in synthesized geometry."""

    result = np.asarray(alpha, dtype=np.uint8).copy()
    protected_binary = np.asarray(protected) > 0
    support = result >= 16
    ys, xs = np.where(support)
    if not len(xs):
        return result, {
            "policy": _SUPPORT_TOPOLOGY_POLICY,
            "status": "unsafe",
            "model": model,
            "reason": "empty_geometry_support",
        }
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    local_support = support[y0:y1, x0:x1]
    local_protected = protected_binary[y0:y1, x0:x1]
    span = max(x1 - x0, y1 - y0)
    gap_limit = int(np.clip(round(span * 0.04), 6, 64))
    if model == "dominant_interior_outward_ring_v1":
        # The canonical frame is already rendered from a complete interior
        # model. Running directional protected-gap closing afterwards can add
        # child pixels *outside* a rounded corner and create a visible spur.
        # Child overlap remains in full support by construction and is removed
        # from visible ownership later in _make_node.
        return result, {
            "policy": _SUPPORT_TOPOLOGY_POLICY,
            "status": "pass",
            "model": model,
            "completion_mode": "authoritative_frame_model_already_complete",
            "gap_limit_px": gap_limit,
            "detected_protected_gap_pixel_count": 0,
            "synthesized_protected_gap_pixel_count": 0,
            "remaining_detected_protected_gap_pixel_count": 0,
            "nonprotected_support_pixel_count_added": 0,
            "structural_background_preserved": True,
        }

    def detected_gaps(binary: np.ndarray) -> np.ndarray:
        core = binary.astype(np.uint8) * 255
        horizontal = cv2.morphologyEx(
            core,
            cv2.MORPH_CLOSE,
            np.ones((1, gap_limit + 1), np.uint8),
        ) > 0
        vertical = cv2.morphologyEx(
            core,
            cv2.MORPH_CLOSE,
            np.ones((gap_limit + 1, 1), np.uint8),
        ) > 0
        return local_protected & ~binary & (horizontal | vertical)

    before = detected_gaps(local_support)
    completed = before.copy()
    if np.any(completed):
        local = result[y0:y1, x0:x1]
        local[completed] = 255
        local_support = local >= 16
    remaining = detected_gaps(local_support)
    # Completion is strictly constrained to protected child pixels.  This is
    # the explicit guard that preserves genuine geometric holes/counters.
    nonprotected_added = completed & ~local_protected
    report = {
        "policy": _SUPPORT_TOPOLOGY_POLICY,
        "status": "pass" if not np.any(remaining) else "unsafe",
        "model": model,
        "gap_limit_px": gap_limit,
        "detected_protected_gap_pixel_count": int(np.count_nonzero(before)),
        "synthesized_protected_gap_pixel_count": int(np.count_nonzero(completed)),
        "remaining_detected_protected_gap_pixel_count": int(np.count_nonzero(remaining)),
        "nonprotected_support_pixel_count_added": int(np.count_nonzero(nonprotected_added)),
        "structural_background_preserved": bool(not np.any(nonprotected_added)),
    }
    return result, report


def _tight_arrays(
    full_alpha: np.ndarray,
    visible_alpha: np.ndarray,
    image_rgb: np.ndarray,
    *,
    padding: int = 1,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(full_alpha > 0)
    if not len(xs):
        raise ValueError("cannot crop an empty geometry mask")
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(full_alpha.shape[1], int(xs.max()) + 1 + padding)
    y1 = min(full_alpha.shape[0], int(ys.max()) + 1 + padding)
    return (
        x0,
        y0,
        full_alpha[y0:y1, x0:x1].copy(),
        visible_alpha[y0:y1, x0:x1].copy(),
        image_rgb[y0:y1, x0:x1].copy(),
    )


def _reconstruct_rgb(
    source_crop: np.ndarray,
    full_alpha: np.ndarray,
    visible_alpha: np.ndarray,
    colour_rgb: np.ndarray | None,
    surface_crop: np.ndarray | None = None,
) -> np.ndarray:
    support = full_alpha > 0
    hidden = support & (visible_alpha == 0)
    result = source_crop.copy()
    visible = support & (visible_alpha > 0)
    if surface_crop is not None:
        if surface_crop.shape != source_crop.shape or surface_crop.dtype != np.uint8:
            raise ValueError("synthesized panel surface must match the source crop")
        # A geometry node is an editable clean surface, not a copy of the
        # observed bitmap.  Use the fitted reference throughout the reliable
        # alpha support so independently exported PNGs cannot retain text or
        # price ghosts in their RGB channels.
        result[support] = surface_crop[support]
    elif colour_rgb is not None:
        result[support] = np.asarray(colour_rgb, dtype=np.uint8)
    elif hidden.any():
        if np.count_nonzero(visible) >= 16:
            visible_pixels = source_crop[visible]
            visible_float = visible_pixels.astype(np.float32)
            spread = float(
                np.median(np.abs(visible_float - np.median(visible_float, axis=0)))
            )
            if spread <= 12.0:
                result[hidden] = np.median(visible_pixels, axis=0).astype(np.uint8)
            else:
                inpainted = cv2.inpaint(
                    source_crop,
                    hidden.astype(np.uint8) * 255,
                    3,
                    cv2.INPAINT_TELEA,
                )
                result[hidden] = inpainted[hidden]
    return result


def _reference_contamination_carve(
    image_rgb: np.ndarray,
    full_alpha: np.ndarray,
    visible_alpha: np.ndarray,
    protected: np.ndarray,
    candidate: _Candidate,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Carve unexplained source ink out of a synthesized geometry surface.

    The full support and its RGBA remain the clean fitted surface.  Only
    categorical visible ownership is punched, allowing the exact-source
    technical remainder above the extracted layers to retain unknown text or
    price pixels.  Seeds are restricted to the opaque core so legitimate
    antialias edge blends are not mistaken for contamination.
    """

    reference_type: str
    if candidate.surface_rgb is not None:
        reference = np.asarray(candidate.surface_rgb)
        if reference.shape != image_rgb.shape or reference.dtype != np.uint8:
            raise ValueError("candidate surface_rgb must match the uint8 image canvas")
        reference_type = "surface_rgb"
        surface_metrics = candidate.metrics.get("surface", {})
        validation_p95 = (
            float(surface_metrics.get("validation_delta_e76_p95", 0.0))
            if isinstance(surface_metrics, dict)
            else 0.0
        )
        # A visibly clean flat/affine surface needs a tighter gate than the
        # fit-acceptance gate.  The latter proves the model is plausible; this
        # threshold decides which observed ink may remain in the independently
        # exported geometry PNG.  One DeltaE unit above the robust p95 (with a
        # small absolute floor) preserves mild acquisition noise while carving
        # faint glyph/shadow ghosts as well as saturated text.
        threshold = float(np.clip(validation_p95 + 1.0, 6.0, 12.0))
    elif candidate.colour_rgb is not None:
        colour = np.asarray(candidate.colour_rgb, dtype=np.uint8)
        if colour.shape != (3,):
            raise ValueError("candidate colour_rgb must contain exactly three channels")
        reference = np.broadcast_to(colour.reshape(1, 1, 3), image_rgb.shape)
        reference_type = "constant_colour_rgb"
        # Frame extraction may accept a wider palette tolerance than a flat
        # divider.  Respect that evidence while retaining an absolute cap that
        # still catches clearly unrelated source content.
        tolerance = float(candidate.metrics.get("colour_tolerance_lab", 0.0) or 0.0)
        threshold = float(np.clip(max(10.0, tolerance * 0.75), 10.0, 18.0))
    else:
        return visible_alpha.copy(), None

    full = np.asarray(full_alpha, dtype=np.uint8)
    visible = np.asarray(visible_alpha, dtype=np.uint8).copy()
    if full.shape != image_rgb.shape[:2] or visible.shape != full.shape:
        raise ValueError("geometry alpha canvases must match the source image")
    protected_binary = np.asarray(protected) > 0
    support = full > 0
    eligible = (visible > 0) & ~protected_binary
    core = eligible & (full >= _CLEANLINESS_CORE_ALPHA)
    grow_domain = eligible & (full > 0)

    source_lab = _lab(image_rgb)
    reference_lab = _lab(np.ascontiguousarray(reference, dtype=np.uint8))
    delta = np.linalg.norm(source_lab - reference_lab, axis=2)
    growth_threshold = float(max(3.5, threshold * 0.55))
    high_delta_seeds = core & (delta > threshold)
    # Any observed RGB deviation that remains visible would be solved back
    # into the exported geometry PNG by the exact renderer. Route every such
    # pixel to the top SOURCE REMAINDER instead; the geometry itself can then
    # use its clean reference throughout full support. High-DeltaE seeds are
    # still measured separately for fail-closed model validation.
    exact_deviation = eligible & (
        np.any(image_rgb != reference, axis=2) | (full < 255)
    )
    grown = exact_deviation.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    # Two bounded reconstruction steps capture antialiased glyph fringes but
    # cannot flood across a broad textured/gradient geometry candidate.
    for _iteration in range(2):
        expanded = (cv2.dilate(grown.astype(np.uint8), kernel, iterations=1) > 0)
        expanded &= grow_domain & (
            (delta > growth_threshold) | exact_deviation
        )
        if np.array_equal(expanded | grown, grown):
            break
        grown |= expanded
    carved = cv2.dilate(grown.astype(np.uint8), kernel, iterations=1) > 0
    carved &= grow_domain
    visible[carved] = 0

    support_pixels = int(np.count_nonzero(support))
    carved_pixels = int(np.count_nonzero(carved))
    remaining_domain = (visible > 0) & (full >= _CLEANLINESS_CORE_ALPHA) & ~protected_binary
    remaining_delta = delta[remaining_domain]
    remaining_p95 = float(np.percentile(remaining_delta, 95)) if remaining_delta.size else 0.0
    remaining_max = float(np.max(remaining_delta)) if remaining_delta.size else 0.0
    carved_fraction = carved_pixels / max(1, support_pixels)
    high_delta_pixels = int(np.count_nonzero(high_delta_seeds))
    high_delta_fraction = high_delta_pixels / max(1, support_pixels)
    retained_fraction = int(np.count_nonzero(remaining_domain)) / max(1, int(np.count_nonzero(core)))
    status = (
        "pass"
        if high_delta_fraction <= _CLEANLINESS_MAX_CARVED_FRACTION
        and remaining_max <= 1e-5
        else "unsafe"
    )
    report: dict[str, Any] = {
        "policy": _CLEANLINESS_POLICY,
        "status": status,
        "reference_type": reference_type,
        "metric_domain": "visible_core_alpha_gte_192_excluding_known_children",
        "threshold_delta_e76": round(threshold, 5),
        "growth_threshold_delta_e76": round(growth_threshold, 5),
        "remaining_visible_delta_e76_limit": round(threshold, 5),
        "dilation_radius_px": 1,
        "growth_iterations": 2,
        "support_pixel_count": support_pixels,
        "carved_pixel_count": carved_pixels,
        "carved_pixel_fraction_of_support": round(carved_fraction, 8),
        "exact_source_deviation_pixel_count": int(np.count_nonzero(exact_deviation)),
        "high_delta_seed_pixel_count": high_delta_pixels,
        "high_delta_seed_fraction_of_support": round(high_delta_fraction, 8),
        "known_child_excluded_pixel_count": int(np.count_nonzero(support & protected_binary)),
        "preexisting_visible_exclusion_pixel_count": int(np.count_nonzero(support & (visible_alpha == 0))),
        "remaining_visible_pixel_count": int(remaining_delta.size),
        "remaining_visible_delta_e76_p95": round(remaining_p95, 5),
        "remaining_visible_delta_e76_max": round(remaining_max, 5),
        "retained_core_fraction": round(retained_fraction, 8),
        "carved_destination_role": "exact_source_remainder_above_clean_base",
        "max_auto_safe_high_delta_fraction": _CLEANLINESS_MAX_CARVED_FRACTION,
    }
    return visible, report


def _make_node(
    image_rgb: np.ndarray,
    protected: np.ndarray,
    proposal: DetectedProposal,
    candidate: _Candidate,
    element_id: str,
    z_index: int,
    *,
    kind_override: str | None = None,
    ownership_exclusion: np.ndarray | None = None,
    preserve_ownership_exclusion_support: bool = False,
) -> ElementNode:
    full_canvas = candidate.alpha.copy()
    kind = kind_override or proposal.record.kind_hint
    topology_model = (
        str(candidate.metrics.get("frame_topology_model", {}).get("policy"))
        if kind == "frame"
        and isinstance(candidate.metrics.get("frame_topology_model"), dict)
        else "protected_directional_gap_completion"
    )
    full_canvas, support_topology = _complete_protected_support_gaps(
        full_canvas,
        protected,
        model=topology_model,
    )
    frame_model_evidence = candidate.metrics.get("frame_topology_model")
    if kind == "frame" and isinstance(frame_model_evidence, dict):
        # Keep the fitted-model evidence on the node itself, not only in the
        # proposal decision ledger.  QA and downstream consumers must be able
        # to prove that an independently exported frame was synthesized from
        # a dominant interior opening rather than trusting a review label.
        support_topology["frame_model_evidence"] = dict(frame_model_evidence)
    candidate.metrics["support_topology"] = dict(support_topology)
    # Visible ownership is binary even when the supplied child matte is soft.
    # Multiplying by (1-alpha) leaves non-zero parent pixels under antialiased
    # text/product edges, causing two visible owners at threshold 1.  The
    # parent's full support and RGBA still retain the synthesized surface.
    visible_canvas = full_canvas.copy()
    visible_canvas[protected > 0] = 0
    if ownership_exclusion is not None:
        exclusion = np.asarray(ownership_exclusion)
        if exclusion.shape != full_canvas.shape or exclusion.ndim != 2:
            raise ValueError("ownership_exclusion must match the image canvas")
        visible_canvas[exclusion > 0] = 0
        # Overlap with an earlier geometry owner is duplication, not hidden
        # content. Retaining it in full_support lets a later line/frame paint a
        # synthesized stroke back over the actual owner. Only a paired panel
        # deliberately preserves support below its frame/children.
        if not preserve_ownership_exclusion_support:
            full_canvas[exclusion > 0] = 0
    visible_canvas, cleanliness = _reference_contamination_carve(
        image_rgb,
        full_canvas,
        visible_canvas,
        protected,
        candidate,
    )
    if cleanliness is not None:
        candidate.metrics["geometry_cleanliness"] = dict(cleanliness)
    x0, y0, full, visible, source_crop = _tight_arrays(
        full_canvas, visible_canvas, image_rgb
    )
    surface_crop = (
        None
        if candidate.surface_rgb is None
        else candidate.surface_rgb[y0 : y0 + full.shape[0], x0 : x0 + full.shape[1]]
    )
    rgb = _reconstruct_rgb(
        source_crop, full, visible, candidate.colour_rgb, surface_crop
    )
    rgba = np.dstack((rgb, full))
    footprint_binary = cv2.dilate(
        (full > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    )
    footprint = (footprint_binary * 255).astype(np.uint8)
    auto_safe = bool(
        cleanliness is not None
        and cleanliness.get("status") == "pass"
        and support_topology.get("status") == "pass"
    )
    return ElementNode(
        element_id=element_id,
        name=f"{kind.title()} {proposal.record.proposal_id}",
        kind=kind,
        visible_alpha=AlphaCrop(x0, y0, visible),
        full_support=AlphaCrop(x0, y0, full),
        semantic_envelope=AlphaCrop(x0, y0, full.copy()),
        removal_footprint=AlphaCrop(x0, y0, footprint),
        z_index=z_index,
        confidence=candidate.confidence,
        review_status="auto_confirmed" if auto_safe else "unresolved",
        move_safe=auto_safe,
        occluded=bool(np.any((full > 0) & (visible == 0))),
        synthesized_hidden_pixels=bool(np.any((full > 0) & (visible == 0))),
        rgba=rgba,
        evidence=[
            {
                "source": proposal.record.source,
                "proposal_id": proposal.record.proposal_id,
                "geometry_metrics": candidate.metrics,
            }
        ],
        metadata={
            "geometry_backend": "poster_geometry_v2",
            "support_policy": "authoritative_geometry_model_with_protected_child_completion",
            "proposal_bbox": list(proposal.record.bbox),
            "support_topology": dict(support_topology),
            **({"geometry_cleanliness": dict(cleanliness)} if cleanliness is not None else {}),
        },
    )


def _frame_candidate(
    image_rgb: np.ndarray,
    protected: np.ndarray,
    proposal: DetectedProposal,
) -> tuple[_Candidate | None, str, dict[str, Any]]:
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = proposal.record.bbox
    box_width, box_height = x1 - x0, y1 - y0
    if box_width < 12 or box_height < 12:
        return None, "frame proposal is too small for closed-boundary evidence", {}
    if max(box_width / box_height, box_height / box_width) > 16:
        return None, "extreme-aspect proposal has no reliable frame interior", {}
    padding = max(2, min(8, int(round(min(box_width, box_height) * 0.035))))
    cx0, cy0 = max(0, x0 - padding), max(0, y0 - padding)
    cx1, cy1 = min(width, x1 + padding), min(height, y1 + padding)
    crop = image_rgb[cy0:cy1, cx0:cx1]
    crop_lab = _lab(crop)
    local = (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0)
    lx0, ly0, lx1, ly1 = local
    band_width = max(2, min(14, int(round(min(box_width, box_height) * 0.10))))
    band = np.zeros(crop.shape[:2], dtype=bool)
    band[ly0:min(ly1, ly0 + band_width), lx0:lx1] = True
    band[max(ly0, ly1 - band_width):ly1, lx0:lx1] = True
    band[ly0:ly1, lx0:min(lx1, lx0 + band_width)] = True
    band[ly0:ly1, max(lx0, lx1 - band_width):lx1] = True
    best: tuple[
        float,
        np.ndarray,
        np.ndarray,
        list[np.ndarray],
        np.ndarray,
        dict[str, Any],
    ] | None = None
    for colour in _palette(crop, band, 14):
        colour_lab = cv2.cvtColor(colour.reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
        distance = _delta_e(crop_lab, colour_lab)
        for tolerance in (8.0, 13.0, 19.0):
            binary = distance <= tolerance
            allowed = np.zeros_like(binary)
            allowed[max(0, ly0 - 2):min(binary.shape[0], ly1 + 2), max(0, lx0 - 2):min(binary.shape[1], lx1 + 2)] = True
            binary &= allowed
            binary = cv2.morphologyEx(
                binary.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
            ) > 0
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
            for component_id in range(1, count):
                component = labels == component_id
                bx, by, bw, bh, pixels = (int(value) for value in stats[component_id])
                span_x = bw / max(1, box_width)
                span_y = bh / max(1, box_height)
                if span_x < 0.78 or span_y < 0.78:
                    continue
                contours = _component_contours(component)
                if contours is None:
                    continue
                outer, holes, outer_area, hole_area = contours
                hole_ratio = hole_area / max(1.0, outer_area)
                largest_hole_ratio = max(
                    (abs(float(cv2.contourArea(item))) / max(1.0, outer_area) for item in holes),
                    default=0.0,
                )
                ink_ratio = pixels / max(1.0, outer_area)
                # A real outline has one dominant interior.  A panel-coloured
                # background littered with many small text/product holes can
                # have a deceptively high *summed* hole ratio; accepting it
                # would turn most of the poster into one contaminated layer.
                if (
                    hole_ratio < 0.55
                    or largest_hole_ratio < 0.46
                    or not 0.004 <= ink_ratio <= 0.30
                ):
                    continue
                outer_x, outer_y, outer_w, outer_h = cv2.boundingRect(outer)
                bbox_error = (
                    abs(outer_x - lx0) + abs(outer_y - ly0)
                    + abs((outer_x + outer_w) - lx1) + abs((outer_y + outer_h) - ly1)
                ) / max(1.0, 2.0 * (box_width + box_height))
                protected_overlap = float(np.mean(protected[cy0:cy1, cx0:cx1][component] > 0))
                if bbox_error > 0.18 or protected_overlap > 0.12:
                    continue
                score = (
                    0.40 * min(1.0, span_x)
                    + 0.25 * min(1.0, span_y)
                    + 0.15 * min(1.0, hole_ratio / 0.78)
                    + 0.10 * min(1.0, largest_hole_ratio / 0.72)
                    + 0.10 * (1.0 - min(1.0, bbox_error / 0.18))
                )
                metrics = {
                    "span_x": round(span_x, 5),
                    "span_y": round(span_y, 5),
                    "hole_ratio": round(hole_ratio, 5),
                    "largest_hole_ratio": round(largest_hole_ratio, 5),
                    "ink_to_outer_ratio": round(ink_ratio, 5),
                    "bbox_error": round(bbox_error, 5),
                    "protected_overlap": round(protected_overlap, 5),
                    "palette_colour_rgb": colour.tolist(),
                    "colour_tolerance_lab": tolerance,
                }
                if best is None or score > best[0]:
                    best = (score, colour, outer, holes, component.copy(), metrics)
    if best is None:
        return None, "no closed colour-consistent frame with a preserved inner hole", {}
    score, colour, outer, holes, selected_component, metrics = best
    # A frame must retain its dominant interior hole.  Its outer colour
    # component is not authoritative because an attached ribbon can merge
    # into that component and import open glyph notches.  Reconstruct a clean
    # ring from side/bottom stroke evidence around the dominant interior.
    outer_area = abs(float(cv2.contourArea(outer)))
    kept_holes = [
        item for item in holes if abs(float(cv2.contourArea(item))) >= max(4.0, outer_area * 0.006)
    ]
    interior_alpha = np.zeros((height, width), dtype=np.uint8)
    if not kept_holes:
        metrics["frame_topology_model"] = {
            "policy": "dominant_interior_outward_ring_v1",
            "status": "unsafe",
            "reason": "no_authoritative_dominant_interior_hole",
        }
        return None, "frame has no authoritative dominant interior hole", metrics
    dominant_interior = max(
        kept_holes, key=lambda item: abs(float(cv2.contourArea(item)))
    )
    observed_interior_alpha = _render_shape_alpha(
        crop.shape[:2], dominant_interior, []
    )
    dominant_interior, interior_topology = _regularize_frame_interior(
        crop.shape[:2], dominant_interior
    )
    metrics["interior_topology_model"] = interior_topology
    if dominant_interior is None:
        metrics["frame_topology_model"] = {
            "policy": "dominant_interior_outward_ring_v1",
            "status": "unsafe",
            "reason": "authoritative interior top could not be canonicalized",
            "interior_topology": interior_topology,
        }
        return None, "frame interior topology could not be reconstructed safely", metrics
    observed_alpha = _render_shape_alpha(crop.shape[:2], outer, kept_holes)
    local_alpha, topology_model = _authoritative_frame_ring(
        selected_component,
        observed_alpha,
        dominant_interior,
        protected[cy0:cy1, cx0:cx1],
    )
    topology_model["interior_topology"] = interior_topology
    metrics["frame_topology_model"] = topology_model
    if local_alpha is None:
        return None, "frame ring topology could not be reconstructed safely", metrics
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[cy0:cy1, cx0:cx1] = local_alpha
    reconstructed_interior_alpha = _render_shape_alpha(
        crop.shape[:2], dominant_interior, []
    )
    interior_alpha[cy0:cy1, cx0:cx1] = reconstructed_interior_alpha
    interior_fit_exclusion = np.zeros((height, width), dtype=np.uint8)
    added_interior = (
        (reconstructed_interior_alpha >= 16) & (observed_interior_alpha < 16)
    )
    interior_fit_exclusion[cy0:cy1, cx0:cx1][added_interior] = 255
    metrics["interior_surface_fit_exclusion_pixel_count"] = int(
        np.count_nonzero(added_interior)
    )
    confidence = float(np.clip(0.66 + 0.27 * score, 0.0, 0.97))
    if confidence < 0.82:
        return None, "closed frame evidence exists but is below auto-accept confidence", metrics
    return _Candidate(
        alpha,
        confidence,
        metrics,
        colour,
        interior_alpha=interior_alpha,
        interior_fit_exclusion=interior_fit_exclusion,
    ), "accepted", metrics


def _grid_sample_coverage(samples: np.ndarray, support: np.ndarray) -> float:
    ys, xs = np.where(support)
    if not len(xs):
        return 0.0
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    covered = 0
    eligible = 0
    for row in range(4):
        gy0 = y0 + (y1 - y0) * row // 4
        gy1 = y0 + (y1 - y0) * (row + 1) // 4
        for column in range(4):
            gx0 = x0 + (x1 - x0) * column // 4
            gx1 = x0 + (x1 - x0) * (column + 1) // 4
            cell_support = support[gy0:gy1, gx0:gx1]
            possible = int(np.count_nonzero(cell_support))
            if possible < 12:
                continue
            eligible += 1
            present = int(np.count_nonzero(samples[gy0:gy1, gx0:gx1]))
            if present >= max(8, int(round(possible * 0.035))):
                covered += 1
    return covered / eligible if eligible else 0.0


def _lab_residual(truth_rgb: np.ndarray, predicted_rgb: np.ndarray) -> np.ndarray:
    truth = cv2.cvtColor(
        truth_rgb.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)
    predicted = cv2.cvtColor(
        predicted_rgb.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)
    return np.linalg.norm(truth - predicted, axis=1)


def _fit_panel_surface(
    image_rgb: np.ndarray,
    support: np.ndarray,
    protected: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    visible = support & (protected < 16)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 55, 145) > 0
    edge_halo = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    samples = visible & ~edge_halo
    coverage = _grid_sample_coverage(samples, support)
    sample_y, sample_x = np.where(samples)
    support_pixels = int(np.count_nonzero(support))
    sample_ratio = len(sample_x) / max(1, support_pixels)
    if len(sample_x) < 64 or sample_ratio < 0.10 or coverage < 0.42:
        return None, {
            "sample_pixels": int(len(sample_x)),
            "support_pixels": support_pixels,
            "sample_ratio": round(sample_ratio, 6),
            "grid_coverage": round(coverage, 6),
            "surface_status": "insufficient_unoccluded_spatial_evidence",
        }
    # Bound work and make selection deterministic for large section panels.
    if len(sample_x) > 30000:
        indices = np.linspace(0, len(sample_x) - 1, 30000, dtype=np.int64)
        sample_x = sample_x[indices]
        sample_y = sample_y[indices]
    values = image_rgb[sample_y, sample_x].astype(np.float64)
    center_x = float(np.median(sample_x))
    center_y = float(np.median(sample_y))
    scale = max(1.0, float(max(np.ptp(sample_x), np.ptp(sample_y))))
    nx = (sample_x.astype(np.float64) - center_x) / scale
    ny = (sample_y.astype(np.float64) - center_y) / scale
    affine_design = np.column_stack((np.ones(len(nx)), nx, ny))

    constant_colour = np.median(values, axis=0)
    constant_prediction = np.broadcast_to(constant_colour, values.shape)
    constant_residual = _lab_residual(values, np.clip(constant_prediction, 0, 255))

    inliers = np.ones(len(values), dtype=bool)
    coefficients: np.ndarray | None = None
    for _iteration in range(4):
        if np.count_nonzero(inliers) < 48:
            break
        coefficients, *_ = np.linalg.lstsq(
            affine_design[inliers], values[inliers], rcond=None
        )
        prediction = affine_design @ coefficients
        residual = _lab_residual(values, np.clip(prediction, 0, 255))
        cutoff = max(5.0, float(np.percentile(residual[inliers], 82)))
        next_inliers = residual <= cutoff
        if np.array_equal(next_inliers, inliers):
            break
        inliers = next_inliers
    if coefficients is None:
        return None, {"surface_status": "affine_fit_failed"}
    affine_prediction = affine_design @ coefficients
    affine_residual = _lab_residual(values, np.clip(affine_prediction, 0, 255))
    constant_p90 = float(np.percentile(constant_residual, 90))
    affine_p90 = float(np.percentile(affine_residual, 90))
    if constant_p90 <= 7.0 or affine_p90 >= constant_p90 * 0.87:
        model = "constant"
        selected_residual = constant_residual
        selected_coefficients = constant_colour.reshape(1, 3)
    else:
        model = "affine"
        selected_residual = affine_residual
        selected_coefficients = coefficients
    residual_p50 = float(np.percentile(selected_residual, 50))
    residual_p90 = float(np.percentile(selected_residual, 90))
    residual_p95 = float(np.percentile(selected_residual, 95))
    report = {
        "sample_pixels": int(len(values)),
        "support_pixels": support_pixels,
        "sample_ratio": round(sample_ratio, 6),
        "grid_coverage": round(coverage, 6),
        "model": model,
        "constant_validation_delta_e76_p90": round(constant_p90, 5),
        "affine_validation_delta_e76_p90": round(affine_p90, 5),
        "validation_delta_e76_p50": round(residual_p50, 5),
        "validation_delta_e76_p90": round(residual_p90, 5),
        "validation_delta_e76_p95": round(residual_p95, 5),
        "trimmed_outlier_ratio": round(1.0 - float(np.mean(inliers)), 6),
    }
    if residual_p90 > 12.0 or residual_p95 > 22.0:
        report["surface_status"] = "surface_is_not_stable_enough_for_hidden_pixel_synthesis"
        return None, report
    support_y, support_x = np.where(support)
    if model == "constant":
        predicted = np.broadcast_to(selected_coefficients[0], (len(support_x), 3))
    else:
        support_design = np.column_stack(
            (
                np.ones(len(support_x)),
                (support_x.astype(np.float64) - center_x) / scale,
                (support_y.astype(np.float64) - center_y) / scale,
            )
        )
        predicted = support_design @ selected_coefficients
    low = np.maximum(0.0, np.percentile(values[inliers], 1, axis=0) - 6.0)
    high = np.minimum(255.0, np.percentile(values[inliers], 99, axis=0) + 6.0)
    predicted = np.clip(predicted, low, high)
    surface = np.zeros_like(image_rgb)
    surface[support_y, support_x] = np.rint(predicted).astype(np.uint8)
    report["surface_status"] = "accepted"
    return surface, report


def _panel_candidate_from_frame(
    image_rgb: np.ndarray,
    protected: np.ndarray,
    proposal: DetectedProposal,
    frame: _Candidate,
) -> tuple[_Candidate | None, str, dict[str, Any]]:
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = proposal.record.bbox
    box_width, box_height = x1 - x0, y1 - y0
    area_ratio = (box_width * box_height) / max(1.0, width * height)
    width_ratio = box_width / width
    height_ratio = box_height / height
    side_tolerance_x = max(2, int(round(width * 0.015)))
    side_tolerance_y = max(2, int(round(height * 0.015)))
    touched_sides = sum(
        (
            x0 <= side_tolerance_x,
            y0 <= side_tolerance_y,
            x1 >= width - side_tolerance_x,
            y1 >= height - side_tolerance_y,
        )
    )
    guard_metrics = {
        "proposal_area_ratio": round(area_ratio, 6),
        "proposal_width_ratio": round(width_ratio, 6),
        "proposal_height_ratio": round(height_ratio, 6),
        "canvas_sides_touched": int(touched_sides),
    }
    if (
        area_ratio > 0.60
        or touched_sides >= 3
        or (width_ratio >= 0.94 and height_ratio >= 0.82)
    ):
        return None, "outer poster/canvas border is not an independent movable panel", guard_metrics
    outer_area = float(box_width * box_height)
    if frame.interior_alpha is not None and np.any(frame.interior_alpha >= 16):
        alpha = frame.interior_alpha.copy()
        interior_area = float(np.count_nonzero(alpha >= 16))
    else:
        contour_data = _component_contours(frame.alpha >= 32)
        if contour_data is None:
            return None, "accepted frame has no stable panel interior", guard_metrics
        _outer, holes, outer_area, _hole_area = contour_data
        if not holes:
            return None, "frame has no enclosed card surface", guard_metrics
        interior = max(holes, key=lambda item: abs(float(cv2.contourArea(item))))
        interior_area = abs(float(cv2.contourArea(interior)))
        alpha = _render_shape_alpha(frame.alpha.shape, interior, [])
    if interior_area < max(64.0, outer_area * 0.40):
        return None, "enclosed region is too small to be a panel surface", {
            **guard_metrics,
            "interior_to_outer_ratio": round(interior_area / max(1.0, outer_area), 6),
        }
    support = alpha > 0
    protected_ratio = float(np.mean(protected[support] > 0)) if support.any() else 1.0
    fit_protected = protected
    fit_exclusion_pixels = 0
    if frame.interior_fit_exclusion is not None:
        if frame.interior_fit_exclusion.shape != support.shape:
            return None, "frame interior fit exclusion has an invalid shape", guard_metrics
        fit_exclusion = frame.interior_fit_exclusion > 0
        fit_exclusion_pixels = int(np.count_nonzero(fit_exclusion & support))
        fit_protected = np.maximum(
            protected,
            fit_exclusion.astype(np.uint8) * 255,
        )
    surface, surface_report = _fit_panel_surface(
        image_rgb, support, fit_protected
    )
    metrics = {
        **guard_metrics,
        "interior_to_outer_ratio": round(interior_area / max(1.0, outer_area), 6),
        "protected_child_ratio": round(protected_ratio, 6),
        "canonical_interior_fit_exclusion_pixel_count": fit_exclusion_pixels,
        "surface": surface_report,
    }
    if protected_ratio > 0.78:
        return None, "too little visible card surface remains outside protected children", metrics
    if surface is None:
        return None, str(surface_report.get("surface_status", "panel surface fit failed")), metrics
    surface_quality = max(
        0.0, 1.0 - float(surface_report["validation_delta_e76_p90"]) / 16.0
    )
    confidence = float(
        np.clip(
            0.56
            + frame.confidence * 0.16
            + surface_quality * 0.18
            + float(surface_report["grid_coverage"]) * 0.10,
            0.0,
            0.96,
        )
    )
    if confidence < 0.82:
        return None, "panel evidence is below auto-accept confidence", metrics
    metrics["confidence"] = round(confidence, 6)
    return _Candidate(alpha, confidence, metrics, surface_rgb=surface), "accepted", metrics


def _axis_line_candidate(
    image_rgb: np.ndarray,
    protected: np.ndarray,
    proposal: DetectedProposal,
) -> tuple[_Candidate | None, str, dict[str, Any]]:
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = proposal.record.bbox
    box_width, box_height = x1 - x0, y1 - y0
    horizontal = box_width >= box_height
    length = box_width if horizontal else box_height
    thickness_hint = box_height if horizontal else box_width
    if length < 24 or length / max(1, thickness_hint) < 6:
        return None, "proposal is not a sufficiently long axis-aligned divider", {}

    # Transpose vertical proposals so one strict horizontal implementation
    # handles both orientations identically.
    work = image_rgb if horizontal else np.transpose(image_rgb, (1, 0, 2))
    work_protected = protected if horizontal else protected.T
    if horizontal:
        wx0, wy0, wx1, wy1 = x0, y0, x1, y1
    else:
        wx0, wy0, wx1, wy1 = y0, x0, y1, x1
    pad_y = max(5, min(14, int(round(length * 0.012))))
    cx0, cx1 = max(0, wx0 - 2), min(work.shape[1], wx1 + 2)
    center_y = int(round((wy0 + wy1 - 1) / 2.0))
    cy0, cy1 = max(0, center_y - pad_y), min(work.shape[0], center_y + pad_y + 1)
    crop = work[cy0:cy1, cx0:cx1]
    lab = _lab(crop)
    local_x0, local_x1 = wx0 - cx0, wx1 - cx0
    if local_x1 - local_x0 < 16 or crop.shape[0] < 5:
        return None, "line context is clipped too tightly", {}

    row_colours = np.median(lab[:, local_x0:local_x1], axis=1)
    row_dispersion = np.median(
        np.linalg.norm(lab[:, local_x0:local_x1] - row_colours[:, None, :], axis=2), axis=1
    )
    edge_count = max(1, min(3, crop.shape[0] // 4))
    top_reference = np.median(row_colours[:edge_count], axis=0)
    bottom_reference = np.median(row_colours[-edge_count:], axis=0)
    best_row = -1
    best_score = -1.0
    best_metrics: dict[str, Any] = {}
    for row, colour_lab in enumerate(row_colours):
        pixel_distance = np.linalg.norm(
            lab[row, local_x0:local_x1] - colour_lab.reshape(1, 3), axis=1
        )
        coherence = float(np.mean(pixel_distance <= 9.0))
        contrast_top = float(np.linalg.norm(colour_lab - top_reference))
        contrast_bottom = float(np.linalg.norm(colour_lab - bottom_reference))
        contrast = min(contrast_top, contrast_bottom)
        dispersion = float(row_dispersion[row])
        score = coherence * min(1.0, contrast / 20.0) * max(0.0, 1.0 - dispersion / 12.0)
        if score > best_score:
            best_row = row
            best_score = score
            best_metrics = {
                "coherence": coherence,
                "contrast_top_delta_e": contrast_top,
                "contrast_bottom_delta_e": contrast_bottom,
                "dispersion_delta_e": dispersion,
            }
    if best_row < 0:
        return None, "no divider row was found", {}
    line_colour_lab = row_colours[best_row]
    line_colour_rgb = cv2.cvtColor(
        np.clip(line_colour_lab, 0, 255).astype(np.uint8).reshape(1, 1, 3),
        cv2.COLOR_LAB2RGB,
    ).reshape(3)
    distances = np.linalg.norm(row_colours - line_colour_lab.reshape(1, 3), axis=1)
    similar_rows = (distances <= 5.5) & (row_dispersion <= 9.0)
    lo = hi = best_row
    while lo > 0 and similar_rows[lo - 1]:
        lo -= 1
    while hi + 1 < len(similar_rows) and similar_rows[hi + 1]:
        hi += 1
    thickness = hi - lo + 1
    pixel_distance = np.linalg.norm(
        lab[lo : hi + 1, local_x0:local_x1] - line_colour_lab.reshape(1, 1, 3), axis=2
    )
    column_support = np.mean(pixel_distance <= 10.0, axis=0) >= 0.50
    coverage = float(np.mean(column_support))
    support_positions = np.where(column_support)[0]
    protected_view = work_protected[
        cy0 + lo : cy0 + hi + 1,
        wx0:wx1,
    ]
    protected_ratio = float(np.mean(protected_view > 0)) if protected_view.size else 0.0
    best_metrics.update(
        {
            "orientation": "horizontal" if horizontal else "vertical",
            "coverage": round(coverage, 5),
            "thickness_px": thickness,
            "protected_overlap": round(protected_ratio, 5),
            "line_colour_rgb": line_colour_rgb.tolist(),
            "acceptance_score": round(best_score, 5),
        }
    )
    if (
        coverage < 0.70
        or min(best_metrics["contrast_top_delta_e"], best_metrics["contrast_bottom_delta_e"]) < 7.0
        or best_metrics["dispersion_delta_e"] > 8.0
        or protected_ratio > 0.25
        or len(support_positions) < length * 0.65
    ):
        return None, "divider lacks continuous, contrasting, uncontaminated side evidence", best_metrics
    start = wx0 + int(support_positions.min())
    end = wx0 + int(support_positions.max()) + 1
    line_y = cy0 + int(round((lo + hi) / 2.0))
    scale = 4
    high = np.zeros((work.shape[0] * scale, work.shape[1] * scale), dtype=np.uint8)
    cv2.line(
        high,
        (start * scale, line_y * scale),
        ((end - 1) * scale, line_y * scale),
        255,
        max(1, thickness * scale),
        cv2.LINE_AA,
    )
    work_alpha = cv2.resize(
        high, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_AREA
    )
    alpha = work_alpha if horizontal else work_alpha.T
    confidence = float(np.clip(0.65 + 0.18 * coverage + 0.12 * min(1.0, best_score), 0.0, 0.96))
    if confidence < 0.82:
        return None, "divider evidence is below auto-accept confidence", best_metrics
    return _Candidate(alpha, confidence, best_metrics, line_colour_rgb), "accepted", best_metrics


def _filled_shape_candidate(
    image_rgb: np.ndarray,
    protected: np.ndarray,
    proposal: DetectedProposal,
) -> tuple[_Candidate | None, str, dict[str, Any]]:
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = proposal.record.bbox
    box_width, box_height = x1 - x0, y1 - y0
    if box_width < 8 or box_height < 8:
        return None, "filled geometry proposal is too small", {}
    if proposal.mask_hint is None:
        return None, "ribbon/badge/panel requires a segmentation hint or explicit user contour", {}
    hint = proposal.mask_hint.to_canvas((width, height))
    binary = hint >= 32
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if count <= 1:
        return None, "segmentation hint is empty", {}
    # Select the component with the greatest overlap with the proposal box.
    target = np.zeros(binary.shape, dtype=bool)
    target[y0:y1, x0:x1] = True
    component_id = max(
        range(1, count), key=lambda item: int(np.count_nonzero((labels == item) & target))
    )
    component = labels == component_id
    overlap = int(np.count_nonzero(component & target))
    component_area = int(np.count_nonzero(component))
    if overlap < max(16, int(component_area * 0.55)):
        return None, "segmentation hint is not spatially owned by this proposal", {}
    contour_data = _component_contours(component)
    if contour_data is None:
        return None, "segmentation hint has no stable outer contour", {}
    outer, holes, outer_area, _ = contour_data
    solidity = component_area / max(1.0, outer_area)
    bx, by, bw, bh = cv2.boundingRect(outer)
    bbox_iou_intersection = max(0, min(x1, bx + bw) - max(x0, bx)) * max(
        0, min(y1, by + bh) - max(y0, by)
    )
    bbox_iou_union = box_width * box_height + bw * bh - bbox_iou_intersection
    bbox_iou = bbox_iou_intersection / max(1.0, bbox_iou_union)
    kind = proposal.record.kind_hint
    if not 0.42 <= solidity <= 1.12 or bbox_iou < 0.38:
        return None, "segmentation hint is fragmented or does not fit the geometry box", {
            "solidity": round(solidity, 5),
            "bbox_iou": round(bbox_iou, 5),
        }

    # Fill text-shaped holes covered by protected content, but preserve large
    # structural cut-outs/notches.  This provides a clean lower surface when a
    # child text layer is hidden or moved.
    kept_holes: list[np.ndarray] = []
    filled_text_holes = 0
    local_protected = protected
    for hole in holes:
        area = abs(float(cv2.contourArea(hole)))
        hole_mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.fillPoly(hole_mask, [hole], 1)
        protected_fraction = float(np.mean(local_protected[hole_mask > 0] > 0)) if area else 0.0
        if protected_fraction >= 0.18 or area < outer_area * 0.012:
            filled_text_holes += 1
        else:
            kept_holes.append(hole)
    alpha = _render_shape_alpha(binary.shape, outer, kept_holes)
    support = alpha > 0
    visible_pixels = image_rgb[support & (protected == 0)]
    colour = np.median(visible_pixels, axis=0).astype(np.uint8) if len(visible_pixels) else None
    protected_ratio = float(np.mean(protected[support] > 0)) if support.any() else 1.0
    metrics = {
        "solidity": round(solidity, 5),
        "bbox_iou": round(bbox_iou, 5),
        "protected_overlap": round(protected_ratio, 5),
        "preserved_structural_holes": len(kept_holes),
        "filled_text_or_micro_holes": filled_text_holes,
    }
    if protected_ratio > 0.42:
        return None, "too much of the shape is owned by protected child content", metrics
    confidence = float(
        np.clip(0.58 + 0.22 * bbox_iou + 0.12 * min(1.0, solidity) + 0.08 * proposal.record.confidence, 0.0, 0.97)
    )
    if confidence < 0.82:
        return None, "filled geometry evidence is below auto-accept confidence", metrics
    return _Candidate(alpha, confidence, metrics, colour), "accepted", metrics


def _alpha_overlap_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    a = first >= 16
    b = second >= 16
    intersection = int(np.count_nonzero(a & b))
    first_area = int(np.count_nonzero(a))
    second_area = int(np.count_nonzero(b))
    union = int(np.count_nonzero(a | b))
    return {
        "intersection_pixels": intersection,
        "iou": intersection / union if union else 0.0,
        "containment": intersection / min(first_area, second_area)
        if min(first_area, second_area)
        else 0.0,
        "first_area": first_area,
        "second_area": second_area,
    }


def _alpha_iou(first: np.ndarray, second: np.ndarray) -> float:
    return float(_alpha_overlap_metrics(first, second)["iou"])


def _box_duplicate_geometry(
    first_kind: str,
    first_box: tuple[int, int, int, int],
    second_kind: str,
    second_box: tuple[int, int, int, int],
) -> tuple[bool, str | None, dict[str, float]]:
    ax0, ay0, ax1, ay1 = first_box
    bx0, by0, bx1, by1 = second_box
    aw, ah = max(1, ax1 - ax0), max(1, ay1 - ay0)
    bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)
    acx, acy = (ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0
    bcx, bcy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    metrics = {
        "centre_dx": abs(acx - bcx),
        "centre_dy": abs(acy - bcy),
        "width_ratio": min(aw, bw) / max(aw, bw),
        "height_ratio": min(ah, bh) / max(ah, bh),
    }
    if first_kind == second_kind == "frame":
        centre_limit_x = max(4.0, min(aw, bw) * 0.025)
        centre_limit_y = max(4.0, min(ah, bh) * 0.05)
        if (
            metrics["centre_dx"] <= centre_limit_x
            and metrics["centre_dy"] <= centre_limit_y
            and metrics["width_ratio"] >= 0.82
            and metrics["height_ratio"] >= 0.82
        ):
            return True, "near-concentric frame proposal", metrics
    first_horizontal = aw >= ah * 4
    second_horizontal = bw >= bh * 4
    first_vertical = ah >= aw * 4
    second_vertical = bh >= bw * 4
    if first_kind == second_kind == "line" and (
        (first_horizontal and second_horizontal) or (first_vertical and second_vertical)
    ):
        if first_horizontal:
            perpendicular_delta = abs(acy - bcy)
            perpendicular_limit = max(3.0, (ah + bh) * 0.65)
            projected = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            projected_ratio = projected / min(aw, bw)
        else:
            perpendicular_delta = abs(acx - bcx)
            perpendicular_limit = max(3.0, (aw + bw) * 0.65)
            projected = max(0.0, min(ay1, by1) - max(ay0, by0))
            projected_ratio = projected / min(ah, bh)
        metrics["axis_perpendicular_delta"] = perpendicular_delta
        metrics["axis_projected_overlap"] = projected_ratio
        if perpendicular_delta <= perpendicular_limit and projected_ratio >= 0.80:
            return True, "same-axis line proposal", metrics
    return False, None, metrics


def extract_geometry_layers(
    image_rgb: np.ndarray,
    proposals: Iterable[DetectedProposal],
    *,
    protected_alpha: np.ndarray | None = None,
    z_start: int = 100,
) -> GeometryResult:
    """Convert supported poster-layout proposals into clean editable nodes.

    No proposal is silently discarded. Unsupported, ambiguous and duplicated
    proposals remain explicit decisions for the review UI.
    """

    image = _validate_rgb(image_rgb)
    height, width = image.shape[:2]
    protected = _normalise_protected(protected_alpha, (height, width))
    nodes: list[ElementNode] = []
    decisions: list[GeometryDecision] = []
    accepted_geometry: list[
        tuple[np.ndarray, str, tuple[int, int, int, int], str]
    ] = []
    accepted_panel_masks: list[np.ndarray] = []
    geometry_owned = np.zeros((height, width), dtype=np.uint8)

    def register_visible_ownership(node: ElementNode) -> None:
        ys, xs = node.visible_alpha.canvas_slice((width, height))
        np.maximum(
            geometry_owned[ys, xs], node.visible_alpha.alpha, out=geometry_owned[ys, xs]
        )
    proposal_list = list(proposals)
    eligible = [item for item in proposal_list if item.record.kind_hint in _SUPPORTED_KINDS]
    eligible.sort(
        key=lambda item: (
            -float(item.record.confidence),
            -((item.record.bbox[2] - item.record.bbox[0]) * (item.record.bbox[3] - item.record.bbox[1])),
            item.record.proposal_id,
        )
    )
    for proposal in eligible:
        kind = proposal.record.kind_hint
        if kind == "frame":
            candidate, reason, metrics = _frame_candidate(image, protected, proposal)
        elif kind == "line":
            candidate, reason, metrics = _axis_line_candidate(image, protected, proposal)
        else:
            candidate, reason, metrics = _filled_shape_candidate(image, protected, proposal)
        if candidate is None:
            duplicate_box: tuple[str, str, dict[str, float]] | None = None
            for _accepted_alpha, accepted_kind, accepted_box, accepted_id in accepted_geometry:
                box_duplicate, box_reason, box_metrics = _box_duplicate_geometry(
                    kind,
                    proposal.record.bbox,
                    accepted_kind,
                    accepted_box,
                )
                if box_duplicate:
                    duplicate_box = (
                        accepted_id,
                        box_reason or "geometry duplicates a stronger accepted proposal",
                        box_metrics,
                    )
                    break
            if duplicate_box is not None:
                accepted_id, duplicate_reason, duplicate_metrics = duplicate_box
                decisions.append(
                    GeometryDecision(
                        proposal.record.proposal_id,
                        "rejected_duplicate",
                        duplicate_reason,
                        metrics={
                            **metrics,
                            "duplicate_of_proposal_id": accepted_id,
                            "duplicate_policy": "axis_or_concentric_box_geometry_v2",
                            "box_geometry": duplicate_metrics,
                        },
                    )
                )
                continue
            decisions.append(
                GeometryDecision(proposal.record.proposal_id, "unresolved", reason, metrics=metrics)
            )
            continue
        duplicate: tuple[str, str, dict[str, Any]] | None = None
        for accepted_alpha, accepted_kind, accepted_box, accepted_id in accepted_geometry:
            overlap = _alpha_overlap_metrics(candidate.alpha, accepted_alpha)
            box_duplicate, box_reason, box_metrics = _box_duplicate_geometry(
                kind,
                proposal.record.bbox,
                accepted_kind,
                accepted_box,
            )
            if float(overlap["iou"]) >= 0.78:
                duplicate = (
                    accepted_id,
                    "mask IoU duplicates a stronger accepted geometry proposal",
                    {"mask_overlap": overlap, "box_geometry": box_metrics},
                )
                break
            if float(overlap["containment"]) >= 0.78:
                duplicate = (
                    accepted_id,
                    "mask containment duplicates a stronger accepted geometry proposal",
                    {"mask_overlap": overlap, "box_geometry": box_metrics},
                )
                break
            if box_duplicate:
                duplicate = (
                    accepted_id,
                    box_reason or "geometry duplicates a stronger accepted proposal",
                    {"mask_overlap": overlap, "box_geometry": box_metrics},
                )
                break
        if duplicate is not None:
            accepted_id, duplicate_reason, duplicate_metrics = duplicate
            decisions.append(
                GeometryDecision(
                    proposal.record.proposal_id,
                    "rejected_duplicate",
                    duplicate_reason,
                    metrics={
                        **metrics,
                        "duplicate_of_proposal_id": accepted_id,
                        "duplicate_policy": "iou_or_min_area_containment_or_axis_geometry_v2",
                        **duplicate_metrics,
                    },
                )
            )
            continue
        candidate_visible = candidate.alpha.copy()
        candidate_visible[protected > 0] = 0
        candidate_visible[geometry_owned > 0] = 0
        if not np.any(candidate_visible):
            decisions.append(
                GeometryDecision(
                    proposal.record.proposal_id,
                    "rejected_duplicate",
                    "geometry support is entirely owned by stronger accepted geometry",
                    metrics={
                        **metrics,
                        "duplicate_policy": "categorical visible ownership exhaustion",
                    },
                )
            )
            continue
        owner_ids: list[str] = []
        decision_metrics = {**metrics, "confidence": round(candidate.confidence, 6)}
        panel_node: ElementNode | None = None
        if kind == "frame":
            panel_candidate, panel_reason, panel_metrics = _panel_candidate_from_frame(
                image, protected, proposal, candidate
            )
            panel_status = "accepted" if panel_candidate is not None else "unresolved"
            if panel_candidate is not None:
                panel_duplicate_iou = max(
                    (
                        _alpha_iou(panel_candidate.alpha, accepted)
                        for accepted in accepted_panel_masks
                    ),
                    default=0.0,
                )
                if panel_duplicate_iou >= 0.78:
                    panel_metrics = {
                        **panel_metrics,
                        "accepted_panel_iou": round(panel_duplicate_iou, 6),
                    }
                    panel_reason = "panel surface duplicates a stronger accepted card surface"
                    panel_status = "rejected_duplicate"
                    panel_candidate = None
            decision_metrics["panel_extraction"] = {
                "status": panel_status,
                "reason": panel_reason,
                "metrics": panel_metrics,
            }
            if panel_candidate is not None:
                panel_id = f"GEO_{len(nodes) + 1:04d}_PANEL"
                panel_exclusion = np.maximum(geometry_owned, candidate.alpha)
                panel_node = _make_node(
                    image,
                    protected,
                    proposal,
                    panel_candidate,
                    panel_id,
                    # Card surfaces are always bottom material. They must stay
                    # below every frame/line, including an overlapping frame
                    # from another proposal—not only their paired contour.
                    z_start - 1_000_000 + len(nodes),
                    kind_override="panel",
                    ownership_exclusion=panel_exclusion,
                    preserve_ownership_exclusion_support=True,
                )
                panel_node.metadata.update(
                    {
                        "panel_role": "synthesized_card_surface_below_children",
                        "child_occlusion_policy": "protected_alpha_removed_from_visible_support",
                    }
                )
                nodes.append(panel_node)
                owner_ids.append(panel_id)
                accepted_panel_masks.append(panel_candidate.alpha)
        element_id = f"GEO_{len(nodes) + 1:04d}_{kind.upper()}"
        node = _make_node(
            image,
            protected,
            proposal,
            candidate,
            element_id,
            z_start + len(nodes),
            ownership_exclusion=geometry_owned,
        )
        node_cleanliness = node.metadata.get("geometry_cleanliness")
        if isinstance(node_cleanliness, dict):
            decision_metrics["geometry_cleanliness"] = dict(node_cleanliness)
        node_topology = node.metadata.get("support_topology")
        if isinstance(node_topology, dict):
            decision_metrics["support_topology"] = dict(node_topology)
        if panel_node is not None:
            # The panel was proposed from the untrimmed frame contour. Build
            # the actual frame first, then cut panel visible ownership against
            # that final support. Otherwise AA/duplicate trimming can leave a
            # one-pixel synthetic panel outline with no frame above it.
            panel_support = panel_node.full_support or panel_node.visible_alpha
            # Preserve contamination holes already carved by _make_node;
            # rebuilding from full support here would reintroduce the exact
            # source ghosts the clean-surface policy removed.
            panel_visible = panel_node.visible_alpha.alpha.copy()
            pys, pxs = panel_support.canvas_slice((width, height))
            final_exclusion = np.maximum(
                geometry_owned[pys, pxs],
                protected[pys, pxs],
            )
            frame_support = (node.full_support or node.visible_alpha).to_canvas((width, height))
            final_exclusion = np.maximum(final_exclusion, frame_support[pys, pxs])
            panel_visible[final_exclusion > 0] = 0
            panel_node.visible_alpha = AlphaCrop(
                panel_support.left,
                panel_support.top,
                panel_visible,
            )
            panel_node.occluded = bool(
                np.any((panel_support.alpha > 0) & (panel_visible == 0))
            )
            panel_node.synthesized_hidden_pixels = panel_node.occluded
            panel_node.metadata["paired_frame_id"] = element_id
            node.metadata["paired_panel_id"] = panel_node.element_id
            register_visible_ownership(panel_node)
        nodes.append(node)
        owner_ids.append(element_id)
        register_visible_ownership(node)
        accepted_geometry.append(
            (
                candidate.alpha,
                kind,
                proposal.record.bbox,
                proposal.record.proposal_id,
            )
        )
        decisions.append(
            GeometryDecision(
                proposal.record.proposal_id,
                "accepted",
                "geometry has closed/continuous colour and topology evidence",
                owner_ids,
                decision_metrics,
            )
        )
    counts = {
        "eligible": len(eligible),
        "accepted": sum(item.status == "accepted" for item in decisions),
        "unresolved": sum(item.status == "unresolved" for item in decisions),
        "rejected_duplicate": sum(item.status == "rejected_duplicate" for item in decisions),
        "panel_nodes": sum(node.kind == "panel" for node in nodes),
        "frame_nodes": sum(node.kind == "frame" for node in nodes),
        "line_nodes": sum(node.kind == "line" for node in nodes),
        "cleanliness_pass_nodes": sum(
            isinstance(node.metadata.get("geometry_cleanliness"), dict)
            and node.metadata["geometry_cleanliness"].get("status") == "pass"
            for node in nodes
        ),
        "cleanliness_unsafe_nodes": sum(
            isinstance(node.metadata.get("geometry_cleanliness"), dict)
            and node.metadata["geometry_cleanliness"].get("status") != "pass"
            for node in nodes
        ),
        "support_topology_pass_nodes": sum(
            isinstance(node.metadata.get("support_topology"), dict)
            and node.metadata["support_topology"].get("status") == "pass"
            for node in nodes
        ),
        "support_topology_unsafe_nodes": sum(
            not isinstance(node.metadata.get("support_topology"), dict)
            or node.metadata["support_topology"].get("status") != "pass"
            for node in nodes
        ),
    }
    decision_by_id = {item.proposal_id: item for item in decisions}
    updated_proposals: list[ProposalRecord] = []
    for item in proposal_list:
        record = item.record
        decision = decision_by_id.get(record.proposal_id)
        if decision is None:
            updated_proposals.append(replace(record))
        elif decision.status == "accepted":
            updated_proposals.append(
                replace(
                    record,
                    status="assigned",
                    owner_ids=list(decision.node_ids),
                    reason=None,
                    evidence={**record.evidence, "geometry_decision": decision.metrics},
                )
            )
        elif decision.status == "rejected_duplicate":
            updated_proposals.append(
                replace(
                    record,
                    status="rejected",
                    owner_ids=[],
                    reason=decision.reason,
                    evidence={**record.evidence, "geometry_decision": decision.metrics},
                )
            )
        else:
            updated_proposals.append(
                replace(
                    record,
                    status="unresolved",
                    owner_ids=[],
                    reason=decision.reason,
                    evidence={**record.evidence, "geometry_decision": decision.metrics},
                )
            )
    report = {
            "backend": "poster_geometry_v2",
            "policy": "accept only closed frames, continuous contrasting dividers, or explicit filled-shape hints",
            "cleanliness_policy": {
                "name": _CLEANLINESS_POLICY,
                "core_alpha_threshold": _CLEANLINESS_CORE_ALPHA,
                "growth_alpha_threshold": _CLEANLINESS_GROW_ALPHA,
                "max_auto_safe_carved_fraction": _CLEANLINESS_MAX_CARVED_FRACTION,
                "destination_role": "exact_source_remainder_above_clean_base",
                "nodes": [
                    {
                        "node_id": node.element_id,
                        **dict(node.metadata["geometry_cleanliness"]),
                    }
                    for node in nodes
                    if isinstance(node.metadata.get("geometry_cleanliness"), dict)
                ],
            },
            "support_topology_policy": {
                "name": _SUPPORT_TOPOLOGY_POLICY,
                "frame_model": "dominant_interior_outward_ring_v1",
                "structural_hole_guard": (
                    "only protected child gaps may be completed; nonprotected "
                    "background/counters remain outside support"
                ),
                "nodes": [
                    {
                        "node_id": node.element_id,
                        **dict(node.metadata["support_topology"]),
                    }
                    for node in nodes
                    if isinstance(node.metadata.get("support_topology"), dict)
                ],
            },
            "counts": counts,
            "protected_pixels": int(np.count_nonzero(protected)),
            "decisions": [
                {
                    "proposal_id": item.proposal_id,
                    "status": item.status,
                    "reason": item.reason,
                    "node_ids": item.node_ids,
                    "metrics": item.metrics,
                }
                for item in decisions
            ],
        }
    return GeometryResult(
        nodes,
        updated_proposals,
        report,
        decisions,
    )


def build_geometry_nodes(
    image_rgb: np.ndarray,
    inventory: InventoryResult | Iterable[DetectedProposal],
    *,
    protected_alpha: np.ndarray | None = None,
    z_start: int = 100,
) -> GeometryResult:
    """Orchestrator-facing geometry API.

    ``result.proposals`` is a copied ledger with accepted geometry assigned to
    its node ids; unresolved proposals remain unresolved with an explicit
    reason.  The supplied inventory is never mutated.
    """

    proposals = inventory.proposals if isinstance(inventory, InventoryResult) else inventory
    return extract_geometry_layers(
        image_rgb,
        proposals,
        protected_alpha=protected_alpha,
        z_start=z_start,
    )
