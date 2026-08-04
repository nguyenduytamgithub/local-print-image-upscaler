"""Conservative old-text mask extraction for flat artwork."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


Box = tuple[int, int, int, int]


@dataclass(slots=True)
class TextMaskResult:
    mask: np.ndarray
    roi: Box
    quality: float
    report: dict[str, object]


def _expand_box(box: Box, shape: tuple[int, int], pad: int) -> Box:
    height, width = shape
    x0, y0, x1, y1 = box
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)


def _dominant_lab_centres(values: np.ndarray, maximum: int = 8) -> np.ndarray:
    if not len(values):
        return np.empty((0, 3), dtype=np.float32)
    quantized = np.floor(
        (values + np.array((0.0, 128.0, 128.0), dtype=np.float32))
        / np.array((5.0, 8.0, 8.0), dtype=np.float32)
    ).astype(np.int16)
    _, inverse, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
    centres = []
    for index in np.argsort(-counts)[: min(maximum, len(counts))]:
        centres.append(np.median(values[inverse == index], axis=0))
    return np.asarray(centres, dtype=np.float32)


def _connected_to_seed(candidate: np.ndarray, seed: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    kept = np.zeros_like(candidate, dtype=bool)
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = int(np.logical_and(component, seed).sum())
        if area >= minimum_area and (overlap >= 2 or overlap / max(1, area) >= 0.08):
            kept |= component
    return kept


def _dominant_foreground_seed(
    lab: np.ndarray,
    different: np.ndarray,
    inner: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Choose repeated ink colours instead of every non-background object."""

    eligible = different & inner
    ys, xs = np.where(eligible)
    if len(xs) < 3:
        return eligible, {"foreground_cluster_count": 0, "foreground_seed_pixels": int(len(xs))}
    values = lab[ys, xs]
    quantized = np.floor(
        values / np.array((8.0, 12.0, 12.0), dtype=np.float32)
    ).astype(np.int16)
    _, inverse, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
    candidates: list[tuple[float, np.ndarray, np.ndarray, int]] = []
    inner_width = max(1, int(np.ptp(np.where(inner)[1])) + 1)
    inner_height = max(1, int(np.ptp(np.where(inner)[0])) + 1)
    for index in np.argsort(-counts)[: min(12, len(counts))]:
        members = inverse == index
        if int(members.sum()) < max(3, int(len(values) * 0.008)):
            continue
        centre = np.median(values[members], axis=0)
        within = np.linalg.norm(lab - centre, axis=2) <= 24.0
        selection = within & eligible
        sy, sx = np.where(selection)
        if not len(sx):
            continue
        span_x = (int(sx.max()) - int(sx.min()) + 1) / inner_width
        span_y = (int(sy.max()) - int(sy.min()) + 1) / inner_height
        # Text ink repeats over a line. A nearby icon may be large but usually
        # lacks the same horizontal support inside the OCR rectangle.
        score = float(len(sx)) * (0.35 + 0.65 * span_x) * (0.65 + 0.35 * span_y)
        candidates.append((score, centre, selection, int(len(sx))))
    if not candidates:
        return eligible, {"foreground_cluster_count": 0, "foreground_seed_pixels": int(len(xs))}
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, primary, seed, primary_count = candidates[0]
    accepted_centres = [primary]
    for _, centre, selection, count in candidates[1:]:
        if count >= primary_count * 0.10 and float(np.linalg.norm(centre - primary)) <= 42.0:
            seed |= selection
            accepted_centres.append(centre)
    return seed, {
        "foreground_cluster_count": len(accepted_centres),
        "foreground_seed_pixels": int(seed.sum()),
        "primary_foreground_lab": [round(float(value), 3) for value in primary],
    }


def build_text_mask(image_rgb: np.ndarray, bbox: Box) -> TextMaskResult:
    """Estimate fill, outline, shadow and antialias fringe around one OCR line.

    The function is intentionally bounded by an expanded OCR ROI.  Its quality
    score is a gate: callers must not remove text when the foreground/background
    separation is unstable.
    """

    image = np.asarray(image_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must be uint8 RGB.")
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (int(value) for value in bbox)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid text box {bbox} for {width}x{height} image.")
    line_height = y1 - y0
    pad = max(4, min(48, int(round(line_height * 0.30))))
    roi = _expand_box((x0, y0, x1, y1), (height, width), pad)
    rx0, ry0, rx1, ry1 = roi
    crop = image[ry0:ry1, rx0:rx1]
    crop_h, crop_w = crop.shape[:2]
    inner = np.zeros((crop_h, crop_w), dtype=bool)
    inner[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0] = True

    border_width = max(2, min(pad, int(round(line_height * 0.18))))
    border = np.ones((crop_h, crop_w), dtype=bool)
    if crop_h > border_width * 2 and crop_w > border_width * 2:
        border[border_width:-border_width, border_width:-border_width] = False
    # Exclude the immediate OCR box from background samples; shadows often
    # extend a few pixels past that box, so retain only robust dominant colours.
    border &= ~inner
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    centres = _dominant_lab_centres(lab[border], maximum=4)
    if not len(centres):
        empty = np.zeros((height, width), dtype=bool)
        return TextMaskResult(empty, roi, 0.0, {"failure": "no_background_samples"})
    distances = np.min(
        np.linalg.norm(lab[..., None, :] - centres[None, None, :, :], axis=3), axis=2
    )
    border_distance = distances[border]
    noise = float(np.percentile(border_distance, 90)) if len(border_distance) else 0.0
    threshold = float(np.clip(noise + 7.5, 10.0, 28.0))

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    gradient_threshold = max(24.0, float(np.percentile(gradient[inner], 55)) if inner.any() else 24.0)
    different = distances >= threshold
    edge_support = gradient >= gradient_threshold
    seed, foreground_report = _dominant_foreground_seed(lab, different, inner)
    candidate = cv2.morphologyEx(
        seed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)
    minimum_area = max(2, int(round(line_height * line_height * 0.002)))
    core = _connected_to_seed(candidate, seed, minimum_area)
    # Pull in outline/drop-shadow pixels only next to the supported core.
    halo_radius = max(1, min(3, int(round(line_height * 0.05))))
    halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_radius * 2 + 1,) * 2)
    near_core = cv2.dilate(core.astype(np.uint8), halo_kernel).astype(bool)
    halo = near_core & ((distances >= threshold * 0.48) | edge_support)
    local_mask = core | halo
    fringe_radius = max(1, min(2, int(round(line_height * 0.025))))
    fringe_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fringe_radius * 2 + 1,) * 2)
    local_mask = cv2.dilate(local_mask.astype(np.uint8), fringe_kernel).astype(bool)

    inner_coverage = float(local_mask[inner].mean()) if inner.any() else 0.0
    roi_coverage = float(local_mask.mean())
    component_count = max(0, cv2.connectedComponents(local_mask.astype(np.uint8), 8)[0] - 1)
    # Typical display text occupies roughly 8–75% of its OCR box. Extremely
    # small/large masks are unsafe and are sent to review instead of inpainting.
    # Heavy display faces plus their outline/shadow can legitimately cover most
    # of a tight OCR rectangle. Keep a broad safe band and let border leakage
    # and the downstream ghost/seam gates reject destructive masks.
    if 0.04 <= inner_coverage <= 0.86:
        coverage_score = float(np.clip(1.0 - abs(inner_coverage - 0.48) / 0.70, 0.0, 1.0))
    else:
        coverage_score = 0.0
    border_leak = float(local_mask[border].mean()) if border.any() else 1.0
    leak_score = float(np.clip(1.0 - border_leak / 0.24, 0.0, 1.0))
    component_score = float(np.clip(component_count / max(1.0, len(str(bbox)) * 0.15), 0.35, 1.0))
    quality = float(np.clip(coverage_score * 0.55 + leak_score * 0.35 + component_score * 0.10, 0.0, 1.0))
    if inner_coverage < 0.02 or inner_coverage > 0.90 or roi_coverage > 0.82:
        quality = min(quality, 0.25)

    full = np.zeros((height, width), dtype=bool)
    full[ry0:ry1, rx0:rx1] = local_mask
    return TextMaskResult(
        mask=full,
        roi=roi,
        quality=quality,
        report={
            "bbox": list(bbox),
            "roi": list(roi),
            "line_height": line_height,
            "background_cluster_count": len(centres),
            **foreground_report,
            "lab_distance_threshold": round(threshold, 4),
            "inner_coverage": round(inner_coverage, 6),
            "roi_coverage": round(roi_coverage, 6),
            "border_leak_ratio": round(border_leak, 6),
            "component_count": component_count,
            "quality": round(quality, 6),
            "policy": "bounded colour/edge mask including outline, shadow and antialias fringe",
        },
    )


def build_union_mask(
    image_rgb: np.ndarray, boxes: list[Box], *, minimum_quality: float = 0.45
) -> tuple[np.ndarray, list[TextMaskResult], list[int]]:
    union = np.zeros(image_rgb.shape[:2], dtype=bool)
    results: list[TextMaskResult] = []
    rejected: list[int] = []
    for index, box in enumerate(boxes):
        result = build_text_mask(image_rgb, box)
        results.append(result)
        if result.quality >= minimum_quality:
            union |= result.mask
        else:
            rejected.append(index)
    return union, results, rejected
