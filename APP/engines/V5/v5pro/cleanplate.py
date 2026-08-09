from __future__ import annotations

"""Poster-aware clean-plate synthesis and truthful residual diagnostics."""

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from v5lib.restore import _horizontal_line_repair, photographic_score, restore_background

from .schema import DocumentGraph, union_alpha


@dataclass(slots=True)
class CleanPlateResult:
    background_rgb: np.ndarray
    removal_footprint: np.ndarray
    report: dict[str, Any]
    method: str
    ghost_heatmap: np.ndarray

    @property
    def background(self) -> np.ndarray:
        """Compatibility alias for the legacy V5 restoration result."""

        return self.background_rgb


def _validate_rgb(image_rgb: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be a uint8 RGB image")
    return np.ascontiguousarray(image)


def _normalise_footprint(alpha: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(alpha)
    if value.ndim != 2 or value.shape != shape:
        raise ValueError("removal_alpha must match the source height and width")
    if value.dtype == bool:
        return value.copy()
    if value.dtype != np.uint8:
        raise ValueError("removal_alpha must use bool or uint8")
    # Include antialiased fringes, but do not let a single rounding value turn
    # the entire canvas into a removal request.
    return value >= 4


def _robust_global_poster_surface(
    source_rgb: np.ndarray,
    footprint: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the document-level base surface from visible outer-background evidence.

    Large poster removals frequently leave too little local ring around each
    component.  Telea/local medians then copy white cards or residual letters
    into the clean plate.  The page border normally exposes the true document
    surface, so a robust quadratic fit there is a safer bottom-most layer.
    """

    source = _validate_rgb(source_rgb, "source_rgb")
    height, width = footprint.shape
    yy, xx = np.mgrid[0:height, 0:width]
    border_x = max(8, int(round(width * 0.12)))
    border_y = max(8, int(round(height * 0.08)))
    border = (xx < border_x) | (xx >= width - border_x) | (yy < border_y) | (yy >= height - border_y)
    safe = ~cv2.dilate(footprint.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    samples = border & safe
    if np.count_nonzero(samples) < 256:
        samples = safe
    step = max(1, int(round(min(width, height) / 320)))
    sampled = samples & ((xx % step) == 0) & ((yy % step) == 0)
    sy, sx = np.where(sampled)
    if len(sx) < 64:
        raise RuntimeError("Not enough visible poster background to fit a global surface.")
    x = sx.astype(np.float64) / max(1, width - 1) * 2.0 - 1.0
    y = sy.astype(np.float64) / max(1, height - 1) * 2.0 - 1.0
    design = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    colours = source[sy, sx].astype(np.float64)
    active = np.ones(len(sx), dtype=bool)
    coefficients = np.zeros((design.shape[1], 3), dtype=np.float64)
    residual = np.zeros(len(sx), dtype=np.float64)
    for _ in range(5):
        coefficients, *_ = np.linalg.lstsq(design[active], colours[active], rcond=None)
        prediction = design @ coefficients
        residual = np.linalg.norm(prediction - colours, axis=1)
        # Keep the coherent page surface and discard flowers, dark rules and
        # white card interiors that happen to touch the border.
        limit = max(6.0, float(np.percentile(residual[active], 62)))
        updated = residual <= limit
        if np.array_equal(updated, active) or np.count_nonzero(updated) < 64:
            break
        active = updated
    gx = xx.astype(np.float64) / max(1, width - 1) * 2.0 - 1.0
    gy = yy.astype(np.float64) / max(1, height - 1) * 2.0 - 1.0
    fitted = (
        coefficients[0]
        + gx[:, :, None] * coefficients[1]
        + gy[:, :, None] * coefficients[2]
        + (gx * gx)[:, :, None] * coefficients[3]
        + (gx * gy)[:, :, None] * coefficients[4]
        + (gy * gy)[:, :, None] * coefficients[5]
    )
    result = source.copy()
    result[footprint] = np.clip(np.rint(fitted[footprint]), 0, 255).astype(np.uint8)
    return result, {
        "model": "robust quadratic RGB surface from visible document border",
        "candidate_samples": int(len(sx)),
        "inlier_samples": int(np.count_nonzero(active)),
        "inlier_ratio": round(float(np.mean(active)), 6),
        "inlier_residual_p95": round(float(np.percentile(residual[active], 95)), 5),
    }


def _rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _component_ring(component: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    expanded = cv2.dilate(component.astype(np.uint8), kernel) > 0
    return expanded & ~component


def analyse_clean_plate(
    source_rgb: np.ndarray,
    clean_rgb: np.ndarray,
    removal_footprint: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Estimate ghost/seam risk without pretending hidden ground truth exists.

    The result is a risk audit, not a reference-image quality score.  It asks
    whether anomalous source ink survived inside the declared removal area and
    whether new edge/seam energy is inconsistent with the surrounding field.
    """

    source = _validate_rgb(source_rgb, "source_rgb")
    clean = _validate_rgb(clean_rgb, "clean_rgb")
    if source.shape != clean.shape:
        raise ValueError("source_rgb and clean_rgb must have identical shapes")
    footprint = _normalise_footprint(removal_footprint, source.shape[:2])
    outside_equal = bool(np.array_equal(source[~footprint], clean[~footprint]))
    heatmap = np.zeros(footprint.shape, dtype=np.float32)
    source_lab = _rgb_to_lab(source)
    clean_lab = _rgb_to_lab(clean)
    source_to_clean = np.linalg.norm(source_lab - clean_lab, axis=2)
    source_edges = cv2.Canny(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), 55, 145) > 0
    clean_edges = cv2.Canny(cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY), 55, 145) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(footprint.astype(np.uint8), 8)
    components: list[dict[str, Any]] = []
    weighted_retained = 0.0
    weighted_edge_excess = 0.0
    weighted_seam = 0.0
    total_pixels = int(np.count_nonzero(footprint))
    for component_id in range(1, count):
        component = labels == component_id
        pixels = int(np.count_nonzero(component))
        if not pixels:
            continue
        box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        radius = max(4, min(32, int(round(max(box_width, box_height) * 0.10))))
        ring = _component_ring(component, radius) & ~footprint
        if np.count_nonzero(ring) < 12:
            components.append(
                {
                    "component": component_id,
                    "pixels": pixels,
                    "status": "review",
                    "reason": "insufficient visible ring evidence",
                }
            )
            heatmap[component] = 1.0
            weighted_retained += pixels
            weighted_edge_excess += pixels
            weighted_seam += pixels
            continue
        ring_values = source_lab[ring]
        ring_median = np.median(ring_values, axis=0)
        ring_delta = np.linalg.norm(ring_values - ring_median, axis=1)
        anomaly_threshold = max(7.0, float(np.percentile(ring_delta, 90)) + 3.0)
        source_anomaly = np.linalg.norm(source_lab - ring_median.reshape(1, 1, 3), axis=2)
        anomalous = component & (source_anomaly >= anomaly_threshold)
        retained_similarity = np.exp(-source_to_clean / 4.5)
        anomaly_strength = np.clip(
            (source_anomaly - anomaly_threshold) / max(8.0, anomaly_threshold), 0.0, 1.0
        )
        local_heat = retained_similarity * anomaly_strength
        heatmap[component] = np.maximum(heatmap[component], local_heat[component])
        anomalous_pixels = int(np.count_nonzero(anomalous))
        if anomalous_pixels:
            retained_ratio = float(np.mean(source_to_clean[anomalous] <= 5.0))
        else:
            retained_ratio = 0.0
        ring_edge_density = float(np.mean(clean_edges[ring]))
        inside_edge_density = float(np.mean(clean_edges[component]))
        source_inside_edge_density = float(np.mean(source_edges[component]))
        edge_excess = max(0.0, inside_edge_density - max(0.006, ring_edge_density * 1.65))
        inner_boundary = component & ~(
            cv2.erode(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        )
        outer_boundary = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        outer_boundary &= ~component & ~footprint
        if np.count_nonzero(inner_boundary) and np.count_nonzero(outer_boundary):
            inner_median = np.median(clean_lab[inner_boundary], axis=0)
            outer_median = np.median(clean_lab[outer_boundary], axis=0)
            seam_delta = float(np.linalg.norm(inner_median - outer_median))
        else:
            seam_delta = 99.0
        changed_fraction = float(np.mean(source_to_clean[component] > 3.0))
        weight = pixels / max(1, total_pixels)
        weighted_retained += retained_ratio * pixels
        weighted_edge_excess += min(1.0, edge_excess * 12.0) * pixels
        weighted_seam += min(1.0, seam_delta / 24.0) * pixels
        components.append(
            {
                "component": component_id,
                "pixels": pixels,
                "source_anomalous_pixels": anomalous_pixels,
                "source_ink_retained_ratio": round(retained_ratio, 6),
                "source_edge_density": round(source_inside_edge_density, 6),
                "clean_edge_density": round(inside_edge_density, 6),
                "ring_edge_density": round(ring_edge_density, 6),
                "excess_edge_density": round(edge_excess, 6),
                "boundary_seam_delta_e76": round(seam_delta, 5),
                "changed_fraction": round(changed_fraction, 6),
                "ring_anomaly_threshold_delta_e76": round(anomaly_threshold, 5),
                "weight": round(weight, 6),
            }
        )
    if total_pixels:
        retained = weighted_retained / total_pixels
        edge_risk = weighted_edge_excess / total_pixels
        seam_risk = weighted_seam / total_pixels
    else:
        retained = edge_risk = seam_risk = 0.0
    risk_score = float(np.clip(0.58 * retained + 0.24 * edge_risk + 0.18 * seam_risk, 0.0, 1.0))
    if not outside_equal:
        grade = "fail"
    elif risk_score <= 0.20:
        grade = "pass"
    elif risk_score <= 0.42:
        grade = "review"
    else:
        grade = "fail"
    report = {
        "audit": "clean_plate_ghost_and_seam_risk_v2",
        "grade": grade,
        "risk_score": round(risk_score, 6),
        "source_ink_retained_ratio": round(retained, 6),
        "edge_residual_risk": round(edge_risk, 6),
        "boundary_seam_risk": round(seam_risk, 6),
        "outside_footprint_byte_identical": outside_equal,
        "footprint_pixels": total_pixels,
        "components": components,
        "truth_notice": (
            "No hidden-reference image exists. This measures residual/ghost risk and seams; "
            "it does not claim recovery of the original pixels behind removed objects."
        ),
    }
    return report, np.rint(np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8)


def _repair_axis_structures(
    source: np.ndarray,
    background: np.ndarray,
    footprint: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bridge only axis-aligned structures with evidence on both sides."""

    result = background.copy()
    edges = cv2.Canny(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), 60, 150)
    count, labels, _stats, _ = cv2.connectedComponentsWithStats(footprint.astype(np.uint8), 8)
    horizontal_reports: list[dict[str, Any]] = []
    vertical_reports: list[dict[str, Any]] = []
    for component_id in range(1, count):
        component = labels == component_id
        # The public restore stage deliberately adds a halo around the visible
        # object.  Use a one-pixel eroded core for line-evidence geometry so an
        # antialiased Hough segment ending on that halo is not rejected by a
        # one-pixel endpoint rounding difference.  The full footprint remains
        # the hard write boundary.
        evidence_component = cv2.erode(
            component.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
        ) > 0
        if not evidence_component.any():
            evidence_component = component
        result, horizontal = _horizontal_line_repair(
            source, result, evidence_component, footprint, edges
        )
        transposed, vertical = _horizontal_line_repair(
            np.transpose(source, (1, 0, 2)),
            np.transpose(result, (1, 0, 2)),
            evidence_component.T,
            footprint.T,
            edges.T,
        )
        result = np.transpose(transposed, (1, 0, 2)).copy()
        horizontal_reports.append({"component": component_id, **horizontal})
        vertical_reports.append({"component": component_id, **vertical})
    result[~footprint] = source[~footprint]
    return result, {
        "horizontal": horizontal_reports,
        "vertical": vertical_reports,
        "restored_horizontal_pixels": sum(
            int(item.get("restored_pixels", 0)) for item in horizontal_reports
        ),
        "restored_vertical_pixels": sum(
            int(item.get("restored_pixels", 0)) for item in vertical_reports
        ),
        "policy": "bridge only collinear Hough evidence with matching low-dispersion colours on both sides",
    }


def build_clean_plate(
    source_rgb: np.ndarray,
    graph: DocumentGraph | np.ndarray,
    *,
    mode: str = "auto",
    progress: Callable[[str], None] | None = None,
    poster_surface_rgb: np.ndarray | None = None,
) -> CleanPlateResult:
    """Synthesize a poster clean plate from a union removal alpha.

    The existing audited surface fitter supplies flat/affine/quadratic poster
    fills and a verified LaMa fallback for photographic regions.  This wrapper
    additionally repairs both horizontal and vertical structure and emits a
    residual-risk heatmap for acceptance QA.
    """

    source = _validate_rgb(source_rgb, "source_rgb")
    if isinstance(graph, DocumentGraph):
        if graph.canvas_size != (source.shape[1], source.shape[0]):
            raise ValueError("document graph canvas does not match source_rgb")
        crops = [
            node.removal_footprint or node.full_support or node.visible_alpha
            for node in graph.nodes
            if node.kind != "background"
            and node.review_status != "rejected"
            and node.metadata.get("role")
            != "exact_source_remainder_above_clean_base"
        ]
        requested_alpha = (
            union_alpha(crops, graph.canvas_size)
            if crops
            else np.zeros(source.shape[:2], dtype=np.uint8)
        )
    else:
        requested_alpha = graph
    requested = _normalise_footprint(requested_alpha, source.shape[:2])
    emit = progress or (lambda _message: None)
    if mode not in {"auto", "poster", "lama"}:
        raise ValueError("mode must be 'auto', 'poster', or 'lama'")
    if not requested.any():
        audit, heatmap = analyse_clean_plate(source, source.copy(), requested)
        return CleanPlateResult(
            background_rgb=source.copy(),
            removal_footprint=requested,
            report={
                "requested_method": mode,
                "method": "none",
                "empty_removal": True,
                "clean_plate_audit": audit,
            },
            method="none",
            ghost_heatmap=heatmap,
        )
    restored = restore_background(
        source,
        [requested],
        mode=mode,
        progress=emit,
    )
    repaired, structure_report = _repair_axis_structures(
        source,
        restored.background,
        restored.removal_footprint,
    )
    repaired[~restored.removal_footprint] = source[~restored.removal_footprint]
    audit, heatmap = analyse_clean_plate(
        source, repaired, restored.removal_footprint
    )
    selected_method = restored.method
    reconciled_surface_report: dict[str, Any] | None = None
    reconciled_surface_audit: dict[str, Any] | None = None
    if poster_surface_rgb is not None:
        supplied_surface = _validate_rgb(poster_surface_rgb, "poster_surface_rgb")
        if supplied_surface.shape != source.shape:
            raise ValueError("poster_surface_rgb must match source_rgb")
        # The residual backend derives this bottom-most page surface from
        # boundary-connected, low-gradient evidence and accounts every
        # significant departure as a layer/rejected speck.  Inside the removal
        # footprint it is a more truthful clean plate than local inpainting,
        # which can copy card/text outlines and still score as low residual.
        # This is the true bottom-most editable background, not the exact
        # source composite. Any legitimate source texture/details that differ
        # from it are exported as an explicit low-contrast residual layer by
        # the orchestrator, so no ghost remains baked into the background.
        candidate = supplied_surface.copy()
        # Audit only regions whose hidden surface is actually reconstructed.
        # A full-canvas footprint has no visible surrounding ring and would
        # mechanically score maximum risk even when the supplied page surface
        # is clean. Outside this audit footprint the exact source appearance
        # lives in the explicit base-residual layer.
        audit_candidate = source.copy()
        audit_candidate[restored.removal_footprint] = supplied_surface[
            restored.removal_footprint
        ]
        reconciled_surface_audit, reconciled_heatmap = analyse_clean_plate(
            source, audit_candidate, restored.removal_footprint
        )
        if photographic_score(source) < 0.54:
            repaired = candidate
            audit = reconciled_surface_audit
            heatmap = reconciled_heatmap
            selected_method = "reconciled_bottom_poster_surface"
            structure_report = {
                "policy": (
                    "residual ledger supplies the bottom poster surface; all "
                    "rules/frames remain independent editable layers"
                ),
                "restored_horizontal_pixels": 0,
                "restored_vertical_pixels": 0,
            }
            reconciled_surface_report = {
                "selected": True,
                "policy": (
                    "flat-poster only; full-canvas clean base with exact source "
                    "difference exported as a separate review layer"
                ),
            }
        else:
            reconciled_surface_report = {
                "selected": False,
                "reason": "source classified as photographic",
            }
    global_report: dict[str, Any] | None = None
    global_audit: dict[str, Any] | None = None
    # On graphic posters, a high-risk local fill usually copied card/text ink
    # because the removal component had no clean local ring. Compare it with a
    # document-level border fit and keep the objectively lower ghost-risk plate.
    if poster_surface_rgb is None and photographic_score(source) < 0.54 and (
        audit["grade"] != "pass" or float(np.mean(restored.removal_footprint)) >= 0.24
    ):
        try:
            global_candidate, global_report = _robust_global_poster_surface(
                source, restored.removal_footprint
            )
            global_audit, global_heatmap = analyse_clean_plate(
                source, global_candidate, restored.removal_footprint
            )
            if float(global_audit["risk_score"]) < float(audit["risk_score"]):
                repaired = global_candidate
                audit = global_audit
                heatmap = global_heatmap
                selected_method = "robust_global_poster_surface"
                structure_report = {
                    "policy": (
                        "global clean plate selected; axis rules remain independent layers "
                        "instead of being painted back into the bottom background"
                    ),
                    "restored_horizontal_pixels": 0,
                    "restored_vertical_pixels": 0,
                }
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            global_report = {"failed": True, "error_type": type(exc).__name__}
    report: dict[str, Any] = {
        "backend": "poster_clean_plate_v2",
        "method": selected_method,
        "requested_method": mode,
        "requested_union_pixels": int(np.count_nonzero(requested)),
        "final_footprint_pixels": int(
            source.shape[0] * source.shape[1]
            if reconciled_surface_report and reconciled_surface_report.get("selected")
            else np.count_nonzero(restored.removal_footprint)
        ),
        "base_restoration": restored.report,
        "structure_continuation": structure_report,
        "clean_plate_audit": audit,
        "global_surface_candidate": global_report,
        "global_surface_candidate_audit": global_audit,
        "reconciled_surface_candidate": reconciled_surface_report,
        "reconciled_surface_candidate_audit": reconciled_surface_audit,
        "review_required": audit["grade"] != "pass",
    }
    return CleanPlateResult(
        background_rgb=repaired,
        removal_footprint=(
            np.ones(source.shape[:2], dtype=bool)
            if reconciled_surface_report and reconciled_surface_report.get("selected")
            else restored.removal_footprint.copy()
        ),
        report=report,
        method=selected_method,
        ghost_heatmap=heatmap,
    )
