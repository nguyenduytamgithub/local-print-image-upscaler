from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import dilate_mask


LAMA_MODEL_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"


@dataclass(slots=True)
class RestorationResult:
    background: np.ndarray
    removal_footprint: np.ndarray
    method: str
    report: dict[str, object]


def photographic_score(image_rgb: np.ndarray) -> float:
    """Cheap content classifier used only to choose a conservative fill method."""

    height, width = image_rgb.shape[:2]
    sample = cv2.resize(
        image_rgb,
        (min(width, 384), min(height, 384)),
        interpolation=cv2.INTER_AREA,
    )
    quantized = (sample // 16).reshape(-1, 3)
    unique_ratio = len(np.unique(quantized, axis=0)) / max(1, len(quantized))
    gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    residual = np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 1.2))
    texture = float(np.percentile(residual, 75)) / 32.0
    # Poster text creates strong edges despite a tiny palette, so palette
    # diversity carries more weight than local edge energy. This keeps bold
    # signage out of the photographic LaMa branch.
    return float(np.clip(unique_ratio * 10.0 + texture * 0.25, 0.0, 1.0))


def visible_union(layer_masks: list[np.ndarray]) -> np.ndarray:
    if not layer_masks:
        raise ValueError("At least one layer mask is required.")
    result = np.zeros_like(layer_masks[0], dtype=bool)
    for mask in layer_masks:
        result |= mask
    return result


def make_soft_alpha(mask: np.ndarray, edge_width: float = 1.35) -> np.ndarray:
    """Create an inside-only antialiased matte without expanding ownership.

    A flattened bitmap has background-preblended contour pixels.  Giving the
    innermost boundary a sub-pixel coverage estimate lets the renderer solve a
    decontaminated foreground colour against the reconstructed lower layer.
    Alpha is always exactly zero outside ``mask``; semantic ownership remains
    the hard ceiling.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be a two-dimensional array.")
    if not math.isfinite(float(edge_width)) or edge_width <= 0:
        raise ValueError("edge_width must be a finite positive number.")
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.float32)
    distance = cv2.distanceTransform(
        binary.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    alpha = np.clip(distance / float(edge_width), 0.0, 1.0).astype(np.float32)
    alpha[~binary] = 0.0
    return alpha


def _component_ring(component: np.ndarray, radius: int) -> np.ndarray:
    return dilate_mask(component, radius) & ~dilate_mask(component, 1)


def _robust_spread(pixels: np.ndarray) -> float:
    if not len(pixels):
        return float("inf")
    median = np.median(pixels.astype(np.float32), axis=0)
    mad = np.median(np.abs(pixels.astype(np.float32) - median), axis=0)
    return float((mad * 1.4826).mean())


@dataclass(slots=True)
class _SurfaceFit:
    kind: str
    coefficients: np.ndarray
    normalization: tuple[float, float, float]
    low: np.ndarray
    high: np.ndarray
    validation_median: float
    validation_p90: float
    validation_p95: float
    validation_score: float
    candidates: list[dict[str, object]]


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )


def _linear_to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.rint(_linear_to_srgb(rgb) * 255.0).astype(np.uint8)


def _rgb_rows_to_lab(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3)
    if values.max(initial=0.0) > 1.0:
        values = values / 255.0
    return cv2.cvtColor(values, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def _surface_design(
    xs: np.ndarray,
    ys: np.ndarray,
    normalization: tuple[float, float, float],
    kind: str,
) -> np.ndarray:
    center_x, center_y, scale = normalization
    x = (np.asarray(xs, dtype=np.float64) - center_x) / scale
    y = (np.asarray(ys, dtype=np.float64) - center_y) / scale
    if kind == "constant":
        return np.ones((len(x), 1), dtype=np.float64)
    if kind == "affine":
        return np.column_stack((np.ones_like(x), x, y))
    if kind == "quadratic":
        return np.column_stack((np.ones_like(x), x, y, x * y, x * x, y * y))
    raise ValueError(f"Unknown surface kind: {kind}")


def _fit_surface_coefficients(
    design: np.ndarray,
    values: np.ndarray,
    *,
    iterations: int = 10,
) -> np.ndarray:
    if design.shape[1] == 1:
        return np.median(values, axis=0)[None, :]
    weights = np.ones(len(design), dtype=np.float64)
    ridge = np.eye(design.shape[1], dtype=np.float64) * 1e-6
    ridge[0, 0] = 1e-9
    coefficients = np.zeros((design.shape[1], 3), dtype=np.float64)
    for _ in range(iterations):
        root_weights = np.sqrt(np.maximum(weights, 1e-8))[:, None]
        weighted_design = design * root_weights
        weighted_values = values * root_weights
        normal = weighted_design.T @ weighted_design + ridge
        target = weighted_design.T @ weighted_values
        try:
            coefficients = np.linalg.solve(normal, target)
        except np.linalg.LinAlgError:
            coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
        residual = np.linalg.norm(values - design @ coefficients, axis=1)
        center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - center))) + 1e-8
        cutoff = max(center + 1.345 * scale, 0.002)
        weights = np.where(
            residual <= cutoff,
            1.0,
            cutoff / np.maximum(residual, 1e-8),
        )
    return coefficients


def _predict_surface(
    xs: np.ndarray,
    ys: np.ndarray,
    fit: _SurfaceFit,
) -> np.ndarray:
    design = _surface_design(xs, ys, fit.normalization, fit.kind)
    return np.clip(design @ fit.coefficients, fit.low, fit.high)


def _dominant_background_samples(
    image_rgb: np.ndarray,
    ring: np.ndarray,
    *,
    maximum_samples: int = 20_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    ys, xs = np.where(ring)
    was_subsampled = len(xs) > maximum_samples
    if len(xs) > maximum_samples:
        indices = np.linspace(0, len(xs) - 1, maximum_samples, dtype=np.int64)
        xs, ys = xs[indices], ys[indices]
    if not len(xs):
        return xs, ys, np.empty((0, 3), dtype=np.float64), {
            "ring_samples": 0,
            "selected_samples": 0,
            "selected_ratio": 0.0,
            "cluster_dispersion_delta_e76": None,
            "selected_fragmentation": None,
        }

    rgb = image_rgb[ys, xs]
    lab = _rgb_rows_to_lab(rgb)
    quantized = np.floor(
        (lab + np.array((0.0, 128.0, 128.0), dtype=np.float32))
        / np.array((5.0, 8.0, 8.0), dtype=np.float32)
    ).astype(np.int16)
    _, inverse, counts = np.unique(
        quantized,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    candidate_bins = np.argsort(-counts)[: min(8, len(counts))]
    best_selection: np.ndarray | None = None
    best_dispersion = float("inf")
    best_score = -1.0
    for bin_index in candidate_bins:
        seed_pixels = lab[inverse == bin_index]
        if not len(seed_pixels):
            continue
        center = np.median(seed_pixels, axis=0)
        seed_distance = np.linalg.norm(seed_pixels - center, axis=1)
        radius = float(np.clip(np.percentile(seed_distance, 95) * 2.0 + 2.0, 4.0, 16.0))
        distance = np.linalg.norm(lab - center, axis=1)
        selected = distance <= radius
        dispersion = float(np.median(distance[selected])) if selected.any() else float("inf")
        score = float(selected.sum() / (1.0 + dispersion))
        if score > best_score:
            best_selection = selected
            best_dispersion = dispersion
            best_score = score

    if best_selection is None:
        best_selection = np.ones(len(xs), dtype=bool)
    selected_x = xs[best_selection]
    selected_y = ys[best_selection]
    selected_rgb = image_rgb[selected_y, selected_x].astype(np.float64) / 255.0
    fragmentation: float | None = None
    if len(selected_x) and not was_subsampled:
        selected_mask = np.zeros(ring.shape, dtype=np.uint8)
        selected_mask[selected_y, selected_x] = 1
        eroded = cv2.erode(selected_mask, np.ones((3, 3), dtype=np.uint8)).astype(bool)
        boundary = selected_mask.astype(bool) & ~eroded
        fragmentation = float(boundary.sum() / len(selected_x))
    metadata = {
        "ring_samples": int(len(xs)),
        "selected_samples": int(len(selected_x)),
        "selected_ratio": round(float(len(selected_x) / max(1, len(xs))), 6),
        "cluster_dispersion_delta_e76": (
            None if math.isinf(best_dispersion) else round(best_dispersion, 4)
        ),
        "selected_fragmentation": (
            None if fragmentation is None else round(fragmentation, 6)
        ),
    }
    return selected_x, selected_y, _srgb_to_linear(selected_rgb), metadata


def _spatial_validation_split(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    extent = max(int(np.ptp(xs)), int(np.ptp(ys)), 1)
    block = max(2, int(round(extent / 16.0)))
    x_blocks = xs.astype(np.int64) // block
    y_blocks = ys.astype(np.int64) // block
    hashed = (x_blocks * 73856093) ^ (y_blocks * 19349663)
    validation = hashed % 5 == 0
    if validation.sum() < 8 or (~validation).sum() < 12:
        validation = np.arange(len(xs)) % 5 == 0
    return validation


def _fit_best_surface(
    xs: np.ndarray,
    ys: np.ndarray,
    values_linear: np.ndarray,
    component: np.ndarray,
) -> _SurfaceFit | None:
    if len(xs) < 12:
        return None
    if len(xs) > 8_000:
        indices = np.linspace(0, len(xs) - 1, 8_000, dtype=np.int64)
        xs, ys, values_linear = xs[indices], ys[indices], values_linear[indices]

    component_y, component_x = np.where(component)
    normalization = (
        float((component_x.min() + component_x.max()) / 2.0),
        float((component_y.min() + component_y.max()) / 2.0),
        float(max(np.ptp(component_x), np.ptp(component_y), 1)),
    )
    validation = _spatial_validation_split(xs, ys)
    training = ~validation
    if validation.sum() < 3 or training.sum() < 6:
        validation = np.ones(len(xs), dtype=bool)
        training = np.ones(len(xs), dtype=bool)

    truth_rgb = _linear_to_uint8(values_linear[validation])
    truth_lab = _rgb_rows_to_lab(truth_rgb)
    candidates: list[dict[str, object]] = []
    fitted: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for kind in ("constant", "affine", "quadratic"):
        design = _surface_design(xs[training], ys[training], normalization, kind)
        coefficients = _fit_surface_coefficients(design, values_linear[training])
        low = np.maximum(0.0, np.quantile(values_linear[training], 0.01, axis=0) - 0.01)
        high = np.minimum(1.0, np.quantile(values_linear[training], 0.99, axis=0) + 0.01)
        prediction = np.clip(
            _surface_design(xs[validation], ys[validation], normalization, kind) @ coefficients,
            low,
            high,
        )
        delta_e = np.linalg.norm(
            truth_lab - _rgb_rows_to_lab(_linear_to_uint8(prediction)),
            axis=1,
        )
        median = float(np.percentile(delta_e, 50))
        p90 = float(np.percentile(delta_e, 90))
        p95 = float(np.percentile(delta_e, 95))
        score = median + p90 * 0.25 + p95 * 0.10
        candidates.append(
            {
                "kind": kind,
                "validation_delta_e76_p50": round(median, 4),
                "validation_delta_e76_p90": round(p90, 4),
                "validation_delta_e76_p95": round(p95, 4),
                "selection_score": round(score, 4),
            }
        )
        fitted[kind] = (coefficients, low, high)

    selected_index = 0
    for candidate_index in range(1, len(candidates)):
        current = candidates[selected_index]
        candidate = candidates[candidate_index]
        current_score = float(current["selection_score"])
        candidate_score = float(candidate["selection_score"])
        current_p95 = float(current["validation_delta_e76_p95"])
        candidate_p95 = float(candidate["validation_delta_e76_p95"])
        material_score_gain = candidate_score <= current_score * 0.95 - 0.01
        material_tail_gain = candidate_p95 <= current_p95 - 0.15
        if material_score_gain and material_tail_gain:
            selected_index = candidate_index

    selected = candidates[selected_index]
    kind = str(selected["kind"])
    final_design = _surface_design(xs, ys, normalization, kind)
    final_coefficients = _fit_surface_coefficients(final_design, values_linear)
    final_low = np.maximum(0.0, np.quantile(values_linear, 0.01, axis=0) - 0.01)
    final_high = np.minimum(1.0, np.quantile(values_linear, 0.99, axis=0) + 0.01)
    return _SurfaceFit(
        kind=kind,
        coefficients=final_coefficients,
        normalization=normalization,
        low=final_low,
        high=final_high,
        validation_median=float(selected["validation_delta_e76_p50"]),
        validation_p90=float(selected["validation_delta_e76_p90"]),
        validation_p95=float(selected["validation_delta_e76_p95"]),
        validation_score=float(selected["selection_score"]),
        candidates=candidates,
    )


def _surface_fill(
    canvas: np.ndarray,
    component: np.ndarray,
    fit: _SurfaceFit,
) -> np.ndarray:
    result = canvas.copy()
    fill_y, fill_x = np.where(component)
    result[fill_y, fill_x] = _linear_to_uint8(_predict_surface(fill_x, fill_y, fit))
    return result


def _horizontal_line_repair(
    source: np.ndarray,
    canvas: np.ndarray,
    component: np.ndarray,
    full_footprint: np.ndarray,
    edges: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    component_y, component_x = np.where(component)
    if not len(component_x):
        return canvas, {"candidate_rows": [], "restored_rows": [], "restored_pixels": 0}
    x0, x1 = int(component_x.min()), int(component_x.max()) + 1
    y0, y1 = int(component_y.min()), int(component_y.max()) + 1
    box_width = x1 - x0
    if edges is None:
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 150)
    minimum_length = max(12, int(round(box_width * 0.35)))
    edge_y0 = max(0, y0 - 3)
    edge_y1 = min(source.shape[0], y1 + 3)
    lines = cv2.HoughLinesP(
        edges[edge_y0:edge_y1],
        1,
        np.pi / 180.0,
        threshold=max(10, minimum_length // 2),
        minLineLength=minimum_length,
        maxLineGap=max(4, int(round(box_width * 0.12))),
    )
    left_rows: list[int] = []
    right_rows: list[int] = []
    long_rows: list[int] = []
    very_long = max(float(box_width), source.shape[1] * 0.30)
    side_evidence_length = max(float(minimum_length), box_width * 0.75)
    if lines is not None:
        for line_x0, line_y0, line_x1, line_y1 in np.asarray(lines).reshape(-1, 4):
            line_y0 = int(line_y0) + edge_y0
            line_y1 = int(line_y1) + edge_y0
            dx = int(line_x1) - int(line_x0)
            dy = line_y1 - line_y0
            angle = abs(math.degrees(math.atan2(dy, dx)))
            angle = min(angle, abs(180.0 - angle))
            if angle > 2.0:
                continue
            row = int(round((line_y0 + line_y1) / 2.0))
            if row < y0 - 3 or row >= y1 + 3:
                continue
            start, end = sorted((int(line_x0), int(line_x1)))
            length = math.hypot(dx, dy)
            if length >= side_evidence_length and end <= x0 + 2:
                left_rows.append(row)
            if length >= side_evidence_length and start >= x1 - 2:
                right_rows.append(row)
            if length >= very_long:
                long_rows.append(row)

    confirmed: set[int] = set(long_rows)
    for left_row in left_rows:
        for right_row in right_rows:
            if abs(left_row - right_row) <= 2:
                confirmed.add(int(round((left_row + right_row) / 2.0)))
    candidate_rows = {
        row
        for center in confirmed
        for row in range(max(y0, center - 4), min(y1, center + 5))
    }

    output = canvas.copy()
    restored_rows: set[int] = set()
    restored_pixels = 0
    for row in sorted(candidate_rows):
        positions = np.where(component[row])[0]
        for segment_x in np.split(positions, np.where(np.diff(positions) > 1)[0] + 1):
            if len(segment_x) == 0:
                continue
            segment_start = int(segment_x[0])
            segment_end = int(segment_x[-1]) + 1
            candidate_widths = sorted(
                {
                    4,
                    8,
                    12,
                    min(24, max(4, int(round((segment_end - segment_start) * 0.20)))),
                }
            )
            best: tuple[float, float, np.ndarray, np.ndarray] | None = None
            for width in candidate_widths:
                left_indices = np.arange(max(0, segment_start - width), segment_start)
                right_indices = np.arange(segment_end, min(source.shape[1], segment_end + width))
                left_indices = left_indices[~full_footprint[row, left_indices]]
                right_indices = right_indices[~full_footprint[row, right_indices]]
                if len(left_indices) < 4 or len(right_indices) < 4:
                    continue
                left_lab = _rgb_rows_to_lab(source[row, left_indices])
                right_lab = _rgb_rows_to_lab(source[row, right_indices])
                left_median = np.median(left_lab, axis=0)
                right_median = np.median(right_lab, axis=0)
                agreement = float(np.linalg.norm(left_median - right_median))
                dispersion = max(
                    float(np.median(np.linalg.norm(left_lab - left_median, axis=1))),
                    float(np.median(np.linalg.norm(right_lab - right_median, axis=1))),
                )
                candidate = (agreement, dispersion, left_indices, right_indices)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
            if best is None or best[0] > 12.0 or best[1] > 6.0:
                continue
            _, _, left_indices, right_indices = best
            left_linear = np.median(
                _srgb_to_linear(source[row, left_indices].astype(np.float64) / 255.0),
                axis=0,
            )
            right_linear = np.median(
                _srgb_to_linear(source[row, right_indices].astype(np.float64) / 255.0),
                axis=0,
            )
            blend = np.linspace(0.0, 1.0, len(segment_x), dtype=np.float64)[:, None]
            line = left_linear[None, :] * (1.0 - blend) + right_linear[None, :] * blend
            output[row, segment_x] = _linear_to_uint8(line)
            restored_rows.add(row)
            restored_pixels += len(segment_x)

    return output, {
        "candidate_rows": sorted(int(row) for row in candidate_rows),
        "restored_rows": sorted(int(row) for row in restored_rows),
        "restored_pixels": int(restored_pixels),
        "hough_minimum_length": int(minimum_length),
        "side_evidence_minimum_length": round(float(side_evidence_length), 3),
    }


def _deterministic_poster_restore(
    image_rgb: np.ndarray,
    footprint: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    result = image_rgb.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(footprint.astype(np.uint8), 8)
    methods: list[dict[str, object]] = []
    order = sorted(range(1, count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    telea_seed = cv2.inpaint(image_rgb, footprint.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    poster_edges = cv2.Canny(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), 60, 150)
    for component_index in order:
        component = labels == component_index
        area = int(component.sum())
        bbox_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        bbox_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        radius = max(7, min(48, int(round(max(bbox_width, bbox_height) * 0.08))))
        ring = _component_ring(component, radius) & ~footprint
        ring_pixels = image_rgb[ring]
        spread = _robust_spread(ring_pixels)
        sample_x, sample_y, sample_linear, sample_report = _dominant_background_samples(
            image_rgb,
            ring,
        )
        surface = _fit_best_surface(sample_x, sample_y, sample_linear, component)
        cluster_dispersion = sample_report["cluster_dispersion_delta_e76"]
        cluster_fragmentation = sample_report["selected_fragmentation"]
        surface_confident = bool(
            surface is not None
            and float(sample_report["selected_ratio"]) >= 0.12
            and cluster_dispersion is not None
            and float(cluster_dispersion) <= 8.0
            and (
                cluster_fragmentation is None
                or float(cluster_fragmentation) <= 0.75
            )
            and surface.validation_median <= 6.0
            and surface.validation_p95 <= 22.0
        )
        if surface_confident and surface is not None:
            result = _surface_fill(result, component, surface)
            method = {
                "constant": "robust_flat_colour",
                "affine": "robust_linear_gradient",
                "quadratic": "robust_quadratic_gradient",
            }[surface.kind]
        else:
            result[component] = telea_seed[component]
            method = "opencv_telea"
        result, line_report = _horizontal_line_repair(
            image_rgb,
            result,
            component,
            footprint,
            poster_edges,
        )
        methods.append(
            {
                "component": component_index,
                "pixels": area,
                "ring_spread": round(spread, 3),
                "method": method,
                "surface_model": None if surface is None else surface.kind,
                "surface_confident": surface_confident,
                "surface_validation_delta_e76_p50": (
                    None if surface is None else round(surface.validation_median, 4)
                ),
                "surface_validation_delta_e76_p95": (
                    None if surface is None else round(surface.validation_p95, 4)
                ),
                "surface_candidates": [] if surface is None else surface.candidates,
                "background_samples": sample_report,
                "horizontal_line_repair": line_report,
            }
        )
    result[~footprint] = image_rgb[~footprint]
    return result, {"components": methods}


def _lama_restore(image_rgb: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Run the verified LaMa TorchScript conversion without its stale wrapper deps.

    Tensor conversion and symmetric modulo-8 padding follow the Apache-2.0
    ``simple-lama-inpainting`` reference implementation documented in
    ``APP/engines/V5/SOURCES.md``.
    """

    import torch

    model_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "big-lama.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"LaMa checkpoint is missing: {model_path}")
    digest = hashlib.sha256()
    with model_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != LAMA_MODEL_SHA256:
        raise RuntimeError(f"LaMa checkpoint SHA-256 mismatch: {model_path}")
    height, width = image_rgb.shape[:2]
    padded_height = ((height + 7) // 8) * 8
    padded_width = ((width + 7) // 8) * 8
    image_chw = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    mask_chw = footprint.astype(np.float32)[None, ...]
    image_chw = np.pad(
        image_chw,
        ((0, 0), (0, padded_height - height), (0, padded_width - width)),
        mode="symmetric",
    )
    mask_chw = np.pad(
        mask_chw,
        ((0, 0), (0, padded_height - height), (0, padded_width - width)),
        mode="symmetric",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(model_path), map_location=device)
    try:
        model.eval().to(device)
        image_tensor = torch.from_numpy(image_chw).unsqueeze(0).to(device)
        mask_tensor = (torch.from_numpy(mask_chw).unsqueeze(0).to(device) > 0) * 1
        with torch.inference_mode():
            generated = model(image_tensor, mask_tensor)
        array = generated[0].permute(1, 2, 0).detach().cpu().numpy()
        result = np.clip(array * 255.0, 0, 255).astype(np.uint8)[:height, :width]
        del generated, image_tensor, mask_tensor
        return result
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def restore_background(
    image_rgb: np.ndarray,
    layer_masks: list[np.ndarray],
    *,
    mode: str = "auto",
    progress=print,
) -> RestorationResult:
    """Synthesize only hidden pixels and preserve every pixel outside that footprint."""

    union = visible_union(layer_masks)
    # SAM normally follows the visible object edge, while print artwork often
    # has an outer glow/drop shadow. A wider, bounded footprint prevents those
    # ghosts from remaining in the supposedly clean lower layer.
    radius = max(4, min(18, int(round(min(union.shape) / 150))))
    footprint = dilate_mask(union, radius)
    content_score = photographic_score(image_rgb)
    selected_mode = mode
    if selected_mode == "auto":
        selected_mode = "lama" if content_score >= 0.54 and footprint.mean() < 0.38 else "poster"
    progress(
        "  V5 bước 3/4: tái tạo nền bị che bằng "
        + ("LaMa + ràng buộc vùng thay đổi..." if selected_mode == "lama" else "màu/gradient/Telea bảo toàn poster...")
    )
    fallback: str | None = None
    if selected_mode == "lama":
        try:
            restored = _lama_restore(image_rgb, footprint)
            details: dict[str, object] = {}
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            if mode == "lama":
                raise RuntimeError(
                    "LaMa was requested explicitly but its verified local runtime failed. "
                    "Run APP\\engines\\V5\\setup_v5.ps1 -CheckOnly or use --inpaint poster."
                ) from exc
            fallback = f"LaMa unavailable: {type(exc).__name__}"
            progress(
                "  Cảnh báo V5: LaMa tự động không khả dụng; đang dùng phục hồi poster "
                "và ghi fallback vào manifest."
            )
            selected_mode = "poster"
            restored, details = _deterministic_poster_restore(image_rgb, footprint)
    elif selected_mode == "poster":
        restored, details = _deterministic_poster_restore(image_rgb, footprint)
    else:
        raise ValueError(f"Unknown inpaint mode: {mode}")

    restored = np.array(restored, dtype=np.uint8, copy=True)
    restored[~footprint] = image_rgb[~footprint]
    outside_equal = bool(np.array_equal(restored[~footprint], image_rgb[~footprint]))
    report = {
        "method": selected_mode,
        "requested_method": mode,
        "photographic_score": round(content_score, 5),
        "removal_radius_source_px": radius,
        "removal_footprint_ratio": round(float(footprint.mean()), 6),
        "outside_footprint_byte_identical": outside_equal,
        "fallback": fallback,
        "lama_checkpoint": "big-lama.pt" if selected_mode == "lama" else None,
        "lama_checkpoint_sha256": LAMA_MODEL_SHA256 if selected_mode == "lama" else None,
        "truth_notice": (
            "Pixels hidden in the flat source do not exist. Filled pixels are a plausible "
            "synthesis, not recovery of the original hidden artwork."
        ),
        **details,
    }
    if not outside_equal:
        raise RuntimeError("Background restoration changed pixels outside the declared footprint.")
    return RestorationResult(restored, footprint, selected_mode, report)
