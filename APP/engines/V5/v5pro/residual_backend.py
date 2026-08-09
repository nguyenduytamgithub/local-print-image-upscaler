from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from .schema import AlphaCrop, DocumentGraph, ElementKind, ElementNode, ProposalRecord, union_alpha


Progress = Callable[[str], None]


@dataclass(slots=True)
class ResidualResult:
    """Independent reconciliation result; callers decide when to merge it."""

    nodes: list[ElementNode]
    proposals: list[ProposalRecord]
    bottom_surface_rgb: np.ndarray
    background_connected_mask: np.ndarray
    saliency: np.ndarray
    report: dict[str, Any]


@dataclass(slots=True)
class _Segment:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    source_label: int
    area: int
    kind: ElementKind
    confidence: float
    auto_safe: bool
    metrics: dict[str, Any]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _robust_threshold(values: np.ndarray, *, floor: float, sigma: float) -> float:
    material = np.asarray(values, dtype=np.float32)
    if not material.size:
        return floor
    median = float(np.median(material))
    mad = float(np.median(np.abs(material - median))) * 1.4826
    return max(floor, median + sigma * max(0.5, mad))


def _owned_alpha(graph: DocumentGraph) -> np.ndarray:
    return union_alpha(
        [
            node.visible_alpha
            for node in graph.nodes
            if node.review_status != "rejected" and node.visible_alpha.nonzero_pixels
        ],
        graph.canvas_size,
    )


def _connected_page_background(
    image_rgb: np.ndarray,
    owned: np.ndarray,
    *,
    max_side: int = 320,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find the smooth poster surface connected to the document boundary.

    The local-colour flood is intentionally performed at reduced resolution.
    Smooth gradients remain traversable, while panel borders, illustrations and
    text form high-gradient barriers.  This prevents a dominant white card from
    being mistaken for the page background merely because it occupies many
    pixels.
    """

    height, width = image_rgb.shape[:2]
    scale = min(1.0, max_side / max(width, height))
    small_size = (max(16, int(round(width * scale))), max(16, int(round(height * scale))))
    small = cv2.resize(image_rgb, small_size, interpolation=cv2.INTER_AREA)
    owned_small = cv2.resize(
        (owned > 0).astype(np.uint8), small_size, interpolation=cv2.INTER_AREA
    ) >= 1
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness = lab[:, :, 0]
    gx = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    unowned_gradient = gradient[~owned_small]
    gradient_limit = min(
        22.0,
        max(7.0, float(np.percentile(unowned_gradient, 58)) if unowned_gradient.size else 10.0),
    )
    passable = (~owned_small) & (gradient <= gradient_limit)
    sh, sw = passable.shape
    reached = np.zeros((sh, sw), dtype=np.uint8)
    queue: deque[tuple[int, int]] = deque()

    def seed(y: int, x: int) -> None:
        if passable[y, x] and not reached[y, x]:
            reached[y, x] = 1
            queue.append((y, x))

    for x in range(sw):
        seed(0, x)
        seed(sh - 1, x)
    for y in range(sh):
        seed(y, 0)
        seed(y, sw - 1)

    colour_step_limit = 11.5
    while queue:
        y, x = queue.popleft()
        current = lab[y, x]
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not (0 <= ny < sh and 0 <= nx < sw):
                continue
            if reached[ny, nx] or not passable[ny, nx]:
                continue
            if float(np.linalg.norm(lab[ny, nx] - current)) > colour_step_limit:
                continue
            reached[ny, nx] = 1
            queue.append((ny, nx))

    coverage = float(np.mean(reached > 0))
    fallback = False
    if coverage < 0.08:
        # A heavily decorated border can block most seeds.  Use the robust
        # median of valid border pixels, but retain the low-gradient gate.
        border = np.zeros_like(passable)
        band = max(2, int(round(min(sw, sh) * 0.035)))
        border[:band] = True
        border[-band:] = True
        border[:, :band] = True
        border[:, -band:] = True
        samples = lab[border & ~owned_small]
        if len(samples):
            colour = np.median(samples, axis=0)
            distance = np.linalg.norm(lab - colour[None, None, :], axis=2)
            reached = ((distance <= 18.0) & passable).astype(np.uint8)
            coverage = float(np.mean(reached > 0))
            fallback = True

    kernel = np.ones((3, 3), np.uint8)
    reached = cv2.morphologyEx(reached, cv2.MORPH_CLOSE, kernel)
    full = cv2.resize(reached, (width, height), interpolation=cv2.INTER_NEAREST)
    full[owned > 0] = 0
    return ((full > 0).astype(np.uint8) * 255), {
        "reduced_size": [sw, sh],
        "scale": round(scale, 6),
        "gradient_limit": round(gradient_limit, 4),
        "colour_step_limit": colour_step_limit,
        "coverage": round(float(np.mean(full > 0)), 6),
        "border_fallback": fallback,
    }


def _fit_bottom_surface(
    image_rgb: np.ndarray,
    background_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a robust quadratic RGB page surface from connected background."""

    height, width = image_rgb.shape[:2]
    ys, xs = np.where(background_mask > 0)
    if len(xs) < 64:
        median = np.median(image_rgb.reshape(-1, 3), axis=0)
        surface = np.broadcast_to(median, image_rgb.shape).copy()
        return np.clip(np.rint(surface), 0, 255).astype(np.uint8), {
            "method": "global_median_fallback",
            "sample_count": int(len(xs)),
        }
    limit = 120_000
    if len(xs) > limit:
        indices = np.linspace(0, len(xs) - 1, limit, dtype=np.int64)
        ys, xs = ys[indices], xs[indices]
    nx = xs.astype(np.float64) / max(1, width - 1) * 2.0 - 1.0
    ny = ys.astype(np.float64) / max(1, height - 1) * 2.0 - 1.0
    design = np.column_stack(
        [np.ones_like(nx), nx, ny, nx * nx, nx * ny, ny * ny]
    )
    values = image_rgb[ys, xs].astype(np.float64)
    keep = np.ones(len(xs), dtype=bool)
    coefficients = np.zeros((6, 3), dtype=np.float64)
    for _iteration in range(4):
        coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        residual = np.linalg.norm(values - design @ coefficients, axis=1)
        threshold = _robust_threshold(residual[keep], floor=7.0, sigma=2.8)
        updated = residual <= threshold
        if int(np.count_nonzero(updated)) < 48 or np.array_equal(updated, keep):
            break
        keep = updated
    grid_y, grid_x = np.indices((height, width), dtype=np.float64)
    grid_x = grid_x / max(1, width - 1) * 2.0 - 1.0
    grid_y = grid_y / max(1, height - 1) * 2.0 - 1.0
    basis = np.stack(
        [
            np.ones_like(grid_x),
            grid_x,
            grid_y,
            grid_x * grid_x,
            grid_x * grid_y,
            grid_y * grid_y,
        ],
        axis=2,
    )
    surface = basis @ coefficients
    low = np.percentile(values[keep], 1, axis=0) - 12.0
    high = np.percentile(values[keep], 99, axis=0) + 12.0
    surface = np.minimum(np.maximum(surface, low[None, None, :]), high[None, None, :])
    return np.clip(np.rint(surface), 0, 255).astype(np.uint8), {
        "method": "robust_quadratic_page_surface",
        "sample_count": int(len(xs)),
        "inlier_count": int(np.count_nonzero(keep)),
        "coefficients": coefficients.round(6).tolist(),
    }


def _saliency_map(
    image_rgb: np.ndarray,
    surface_rgb: np.ndarray,
    background_connected: np.ndarray,
    owned: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    surface_lab = cv2.cvtColor(surface_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    delta = np.linalg.norm(lab - surface_lab, axis=2)
    lightness = lab[:, :, 0]
    gradient = cv2.magnitude(
        cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3),
    )
    background_values = delta[(background_connected > 0) & (owned == 0)]
    delta_high = _robust_threshold(background_values, floor=12.0, sigma=4.0)
    # Hysteresis floor for pale cards, white-on-cream type and antialias
    # fringes. The old robust-background-derived value rose to ~14 ΔE on the
    # real poster and classified visible white artwork as page texture.
    delta_low = max(5.0, min(8.0, delta_high * 0.42))
    unowned = owned == 0
    unowned_gradient = gradient[unowned]
    gradient_high = max(
        22.0,
        float(np.percentile(unowned_gradient, 76)) if unowned_gradient.size else 22.0,
    )
    outside_page_surface = background_connected == 0
    # Boundary flood is useful for finding the page surface, but a large pale
    # card or low-gradient white glyph can leak through that flood at reduced
    # resolution. A sufficiently large colour departure is therefore salient
    # regardless of the flood label; the weaker edge test still requires being
    # outside the connected page surface. This closes the exact class of
    # low-contrast poster fills that otherwise remain as ghosts in background.
    high_colour_departure = unowned & (delta >= delta_low)
    edge_departure = (
        unowned
        & outside_page_surface
        & (delta >= delta_low)
        & (gradient >= gradient_high)
    )
    strong = high_colour_departure | edge_departure
    # Do not suppress the one-pixel neighbourhood of existing owners. Those
    # are precisely the antialias/fringe pixels that made old V5 cutouts look
    # dirty. Existing alpha itself is excluded above, so any adjacent residual
    # remains categorically separate and reviewable instead of baked in.
    score = np.clip(
        np.maximum(
            (delta - delta_low) / max(1.0, delta_high - delta_low),
            gradient / max(1.0, gradient_high) * 0.45,
        ),
        0.0,
        1.0,
    )
    saliency = np.clip(np.rint(score * 255.0), 0, 255).astype(np.uint8)
    saliency[~strong] = 0
    return saliency, strong.astype(np.uint8), {
        "delta_low": round(delta_low, 4),
        "delta_high": round(delta_high, 4),
        "gradient_high": round(gradient_high, 4),
        "high_colour_departure_pixels": int(np.count_nonzero(high_colour_departure)),
        "edge_departure_pixels": int(np.count_nonzero(edge_departure)),
        "salient_pixels": int(np.count_nonzero(strong)),
    }


def _colour_labels(
    image_rgb: np.ndarray,
    salient: np.ndarray,
    *,
    max_clusters: int = 12,
) -> tuple[np.ndarray, list[list[float]], dict[str, Any]]:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    coords = np.column_stack(np.where(salient > 0))
    labels = np.full(salient.shape, -1, dtype=np.int16)
    if not len(coords):
        return labels, [], {"cluster_count": 0, "sample_count": 0}
    pixels = lab[coords[:, 0], coords[:, 1]]
    cluster_count = max(1, min(max_clusters, int(round(np.sqrt(len(coords) / 3500.0)))))
    if cluster_count == 1:
        labels[coords[:, 0], coords[:, 1]] = 0
        center = np.median(pixels, axis=0)
        return labels, [center.tolist()], {
            "cluster_count": 1,
            "sample_count": int(len(coords)),
        }
    sample_limit = 100_000
    sample = pixels
    if len(sample) > sample_limit:
        indices = np.linspace(0, len(sample) - 1, sample_limit, dtype=np.int64)
        sample = sample[indices]
    cv2.setRNGSeed(731)
    _compactness, _sample_labels, centers = cv2.kmeans(
        sample.astype(np.float32),
        cluster_count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.25),
        4,
        cv2.KMEANS_PP_CENTERS,
    )
    chunk = 120_000
    assigned = np.empty(len(pixels), dtype=np.int16)
    for start in range(0, len(pixels), chunk):
        material = pixels[start : start + chunk]
        distance = np.linalg.norm(material[:, None, :] - centers[None, :, :], axis=2)
        assigned[start : start + chunk] = np.argmin(distance, axis=1).astype(np.int16)
    labels[coords[:, 0], coords[:, 1]] = assigned
    return labels, centers.tolist(), {
        "cluster_count": int(cluster_count),
        "sample_count": int(len(sample)),
    }


def _extract_segments(
    labels: np.ndarray,
    salient: np.ndarray,
    owned: np.ndarray,
    image_rgb: np.ndarray,
    surface_rgb: np.ndarray,
) -> tuple[list[_Segment], list[tuple[np.ndarray, str]], dict[str, Any]]:
    height, width = salient.shape
    canvas_area = height * width
    segments: list[_Segment] = []
    rejected: list[tuple[np.ndarray, str]] = []
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    surface_lab = cv2.cvtColor(surface_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    delta = np.linalg.norm(lab - surface_lab, axis=2)
    next_label = 0
    for colour_label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        mask = (labels == colour_label).astype(np.uint8)
        # Join anti-aliased pixels of one colour family, but never bridge more
        # than a few document pixels or cross into an owned child.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask[owned > 0] = 0
        count, component_labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for component in range(1, count):
            x, y, bw, bh, area = (int(value) for value in stats[component])
            component_mask = component_labels == component
            # Keep only pixels supported by the original saliency family plus
            # one-pixel closure; this bounds any segmentation growth.
            component_mask &= cv2.dilate(
                (salient > 0).astype(np.uint8), np.ones((3, 3), np.uint8)
            ) > 0
            area = int(np.count_nonzero(component_mask))
            if not area:
                continue
            next_label += 1
            if area <= 2:
                rejected.append((component_mask, "residual speck has at most two supported pixels"))
                continue
            box = _bbox(component_mask)
            x0, y0, x1, y1 = box
            bw, bh = x1 - x0, y1 - y0
            bbox_area = max(1, bw * bh)
            bbox_fraction = bbox_area / canvas_area
            available = int(np.count_nonzero(owned[y0:y1, x0:x1] == 0))
            available_fill = float(area / max(1, available))
            fill = area / bbox_area
            aspect = max(bw / max(1, bh), bh / max(1, bw))
            local = component_mask[y0:y1, x0:x1]
            band = max(1, min(8, int(round(min(bw, bh) * 0.08))))
            border = np.zeros_like(local)
            border[:band] = True
            border[-band:] = True
            border[:, :band] = True
            border[:, -band:] = True
            border_occupancy = float(np.mean(local[border])) if np.any(border) else 0.0
            inner_occupancy = float(np.mean(local[~border])) if np.any(~border) else 0.0
            touches_sides = sum(
                bool(np.any(edge))
                for edge in (local[:band], local[-band:], local[:, :band], local[:, -band:])
            )
            pixels_lab = lab[component_mask]
            colour_std = float(np.mean(np.std(pixels_lab, axis=0))) if len(pixels_lab) else 99.0
            delta_median = float(np.median(delta[component_mask]))
            outer = cv2.dilate(component_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            ring = outer & ~component_mask & (owned == 0)
            boundary_contrast = (
                float(
                    np.linalg.norm(
                        np.median(lab[component_mask], axis=0) - np.median(lab[ring], axis=0)
                    )
                )
                if np.any(ring)
                else 0.0
            )
            canvas_touch_count = sum((x0 == 0, y0 == 0, x1 == width, y1 == height))

            kind: ElementKind
            confidence: float
            auto_safe = False
            line_limit = max(12, int(round(min(width, height) * 0.018)))
            frame_shape = (
                touches_sides == 4
                and border_occupancy >= 0.15
                and inner_occupancy <= max(0.20, border_occupancy * 0.65)
                and fill <= 0.40
            )
            if aspect >= 8.0 and min(bw, bh) <= line_limit:
                kind = "line"
                confidence = 0.72
                auto_safe = fill >= 0.42 and colour_std <= 7.5 and boundary_contrast >= 7.0
            elif frame_shape and bbox_fraction >= 0.0015:
                kind = "frame"
                confidence = 0.70
            elif (
                bbox_fraction >= 0.0012
                and aspect <= 15.0
                and available_fill >= 0.52
            ):
                kind = "panel"
                confidence = min(0.92, 0.58 + available_fill * 0.28)
                auto_safe = (
                    available_fill >= 0.78
                    and colour_std <= 5.5
                    and boundary_contrast >= 6.5
                    and bbox_fraction <= 0.46
                    and canvas_touch_count == 0
                )
            elif area <= max(96, int(round(canvas_area * 0.00010))):
                kind = "micro_detail"
                confidence = 0.56
            else:
                kind = "decoration"
                confidence = min(0.82, 0.52 + min(0.24, delta_median / 120.0))

            metrics = {
                "source_colour_label": colour_label,
                "area": area,
                "bbox_fraction": round(bbox_fraction, 7),
                "fill": round(fill, 6),
                "available_fill": round(available_fill, 6),
                "aspect": round(aspect, 4),
                "colour_std_lab": round(colour_std, 4),
                "delta_median_lab": round(delta_median, 4),
                "boundary_contrast_lab": round(boundary_contrast, 4),
                "border_occupancy": round(border_occupancy, 6),
                "inner_occupancy": round(inner_occupancy, 6),
                "touches_bbox_sides": touches_sides,
                "canvas_touch_count": canvas_touch_count,
            }
            if bbox_fraction > 0.48 or (canvas_touch_count >= 2 and bbox_fraction > 0.18):
                rejected.append((component_mask, "canvas-scale smooth region is background leakage"))
                continue
            segments.append(
                _Segment(
                    component_mask,
                    box,
                    next_label,
                    area,
                    kind,
                    confidence,
                    auto_safe,
                    metrics,
                )
            )
    return segments, rejected, {
        "segment_count": len(segments),
        "initial_rejected_count": len(rejected),
    }


def _box_gap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    horizontal_gap = max(0, max(first[0], second[0]) - min(first[2], second[2]))
    vertical_gap = max(0, max(first[1], second[1]) - min(first[3], second[3]))
    horizontal_overlap = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    vertical_overlap = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return horizontal_gap, vertical_gap, horizontal_overlap, vertical_overlap


def _consolidate_segments(
    segments: list[_Segment],
    canvas_size: tuple[int, int],
) -> tuple[list[_Segment], dict[str, Any]]:
    """Turn colour shards into smallest *useful* residual elements.

    K-means deliberately separates colours, which is excellent for excluding a
    flat page surface but can split one icon/banner into many islands.  This
    bounded spatial grouping reconnects nearby non-panel shards while refusing
    any union that would span a large section of the poster.  Panels and proven
    frames stay atomic so they cannot absorb their children.
    """

    if not segments:
        return [], {"input_segment_count": 0, "output_segment_count": 0}
    width, height = canvas_size
    canvas_area = width * height
    fixed = [item for item in segments if item.kind in {"panel", "frame"}]
    movable = [item for item in segments if item.kind not in {"panel", "frame"}]
    movable.sort(key=lambda item: (item.bbox[1], item.bbox[0], -item.area))
    bridge_x = max(4, min(24, int(round(width * 0.012))))
    bridge_y = max(3, min(18, int(round(height * 0.007))))
    max_group_bbox_area = int(round(canvas_area * 0.045))
    max_group_width = int(round(width * 0.62))
    max_group_height = int(round(height * 0.34))
    groups: list[list[_Segment]] = []
    group_boxes: list[tuple[int, int, int, int]] = []

    for segment in movable:
        best_index: int | None = None
        best_distance = float("inf")
        sw = segment.bbox[2] - segment.bbox[0]
        sh = segment.bbox[3] - segment.bbox[1]
        for index, box in enumerate(group_boxes):
            horizontal_gap, vertical_gap, horizontal_overlap, vertical_overlap = _box_gap(
                segment.bbox, box
            )
            box_width, box_height = box[2] - box[0], box[3] - box[1]
            same_row = (
                vertical_overlap / max(1, min(sh, box_height)) >= 0.18
                and horizontal_gap
                <= max(bridge_x, min(36, int(round(min(sh, box_height) * 1.15))))
            )
            stacked = (
                horizontal_overlap / max(1, min(sw, box_width)) >= 0.18
                and vertical_gap
                <= max(bridge_y, min(24, int(round(min(sh, box_height) * 0.70))))
            )
            close = horizontal_gap <= bridge_x and vertical_gap <= bridge_y
            if not (same_row or stacked or close):
                continue
            union_box = (
                min(segment.bbox[0], box[0]),
                min(segment.bbox[1], box[1]),
                max(segment.bbox[2], box[2]),
                max(segment.bbox[3], box[3]),
            )
            union_width = union_box[2] - union_box[0]
            union_height = union_box[3] - union_box[1]
            if (
                union_width * union_height > max_group_bbox_area
                or union_width > max_group_width
                or union_height > max_group_height
            ):
                continue
            distance = float(horizontal_gap + vertical_gap)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            groups.append([segment])
            group_boxes.append(segment.bbox)
            continue
        groups[best_index].append(segment)
        old = group_boxes[best_index]
        group_boxes[best_index] = (
            min(old[0], segment.bbox[0]),
            min(old[1], segment.bbox[1]),
            max(old[2], segment.bbox[2]),
            max(old[3], segment.bbox[3]),
        )

    consolidated: list[_Segment] = list(fixed)
    member_histogram: list[int] = []
    line_limit = max(16, int(round(min(width, height) * 0.030)))
    micro_limit = max(128, int(round(canvas_area * 0.00014)))
    for group_index, group in enumerate(groups, 1):
        member_histogram.append(len(group))
        if len(group) == 1:
            consolidated.append(group[0])
            continue
        mask = np.zeros((height, width), dtype=bool)
        for item in group:
            mask |= item.mask
        box = _bbox(mask)
        bw, bh = box[2] - box[0], box[3] - box[1]
        area = int(np.count_nonzero(mask))
        aspect = max(bw / max(1, bh), bh / max(1, bw))
        fill = area / max(1, bw * bh)
        if aspect >= 7.0 and min(bw, bh) <= line_limit:
            kind: ElementKind = "line"
            confidence = 0.66
        elif area <= micro_limit:
            kind = "micro_detail"
            confidence = 0.52
        else:
            kind = "decoration"
            confidence = 0.58
        metrics = {
            "source_colour_label": -1,
            "area": area,
            "bbox_fraction": round((bw * bh) / max(1, canvas_area), 7),
            "fill": round(fill, 6),
            "available_fill": round(fill, 6),
            "aspect": round(aspect, 4),
            "colour_std_lab": None,
            "delta_median_lab": round(
                float(np.median([item.metrics["delta_median_lab"] for item in group])), 4
            ),
            "boundary_contrast_lab": round(
                float(np.median([item.metrics["boundary_contrast_lab"] for item in group])), 4
            ),
            "border_occupancy": None,
            "inner_occupancy": None,
            "touches_bbox_sides": None,
            "canvas_touch_count": sum(
                (box[0] == 0, box[1] == 0, box[2] == width, box[3] == height)
            ),
            "consolidated": True,
            "member_segment_count": len(group),
            "member_kinds": [item.kind for item in group],
            "member_source_labels": [item.source_label for item in group],
        }
        consolidated.append(
            _Segment(
                mask,
                box,
                100_000 + group_index,
                area,
                kind,
                confidence,
                False,
                metrics,
            )
        )
    consolidated.sort(key=lambda item: (item.bbox[1], item.bbox[0], -item.area))
    return consolidated, {
        "input_segment_count": len(segments),
        "fixed_panel_or_frame_count": len(fixed),
        "spatial_group_count": len(groups),
        "output_segment_count": len(consolidated),
        "components_consolidated": sum(max(0, value - 1) for value in member_histogram),
        "largest_member_count": max(member_histogram, default=0),
        "bridge_x": bridge_x,
        "bridge_y": bridge_y,
        "max_group_bbox_fraction": round(max_group_bbox_area / max(1, canvas_area), 6),
    }


def _panel_envelope(
    mask: np.ndarray,
    owned: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill only child-owned holes; preserve real design holes."""

    source = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(source, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not contours:
        return source, {"child_holes_filled": 0, "design_holes_preserved": 0}
    hierarchy = hierarchy[0]
    envelope = np.zeros_like(source)
    external = [index for index, item in enumerate(hierarchy) if int(item[3]) < 0]
    for index in external:
        cv2.drawContours(envelope, contours, index, 1, -1)
    holes = (envelope > 0) & (source == 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    child_holes = 0
    design_holes = 0
    for component in range(1, count):
        hole = labels == component
        area = int(stats[component, cv2.CC_STAT_AREA])
        owned_fraction = float(np.count_nonzero(hole & (owned > 0))) / max(1, area)
        if owned_fraction >= 0.12:
            child_holes += 1
            continue
        envelope[hole] = 0
        design_holes += 1
    return envelope, {
        "child_holes_filled": child_holes,
        "design_holes_preserved": design_holes,
    }


def _fit_flat_rgb(
    image_rgb: np.ndarray,
    visible_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    x0, y0, x1, y1 = bbox
    pixels = image_rgb[visible_mask]
    if not len(pixels):
        colour = np.array([0, 0, 0], dtype=np.uint8)
    else:
        colour = np.median(pixels, axis=0).astype(np.uint8)
    crop = np.broadcast_to(colour, (y1 - y0, x1 - x0, 3)).copy()
    residual = (
        float(np.median(np.linalg.norm(pixels.astype(np.float32) - colour, axis=1)))
        if len(pixels)
        else 0.0
    )
    return crop, {"median_rgb": colour.tolist(), "median_rgb_residual": round(residual, 4)}


def _alpha_crop(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> AlphaCrop:
    x0, y0, x1, y1 = bbox
    return AlphaCrop(x0, y0, (mask[y0:y1, x0:x1].astype(np.uint8) * 255))


def _removal_footprint(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> AlphaCrop:
    radius = max(1, min(4, int(round(min(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.015))))
    dilated = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)),
    )
    tight = _bbox(dilated > 0)
    return _alpha_crop(dilated > 0, tight)


def reconcile_residual_elements(
    image_rgb: np.ndarray,
    graph: DocumentGraph,
    *,
    z_start: int = 250_000,
    max_colour_clusters: int = 12,
    progress: Progress | None = None,
) -> ResidualResult:
    """Reconcile significant unowned poster content after fusion/geometry.

    This is deliberately a proposal backend, not a silent graph mutation.  It
    models the smooth page surface from boundary-connected pixels, detects
    salient residual regions, partitions them by colour and topology, excludes
    existing children, and returns exclusive nodes plus a complete proposal
    ledger.  Ambiguous content is an unresolved node; only strict flat panels
    and rules are marked move-safe.
    """

    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Residual input must be uint8 RGB")
    height, width = image_rgb.shape[:2]
    if graph.canvas_size != (width, height):
        raise ValueError("Residual graph canvas differs from the source image")
    if max_colour_clusters < 1 or max_colour_clusters > 32:
        raise ValueError("max_colour_clusters must be from 1 through 32")
    emit = progress or (lambda _message: None)
    emit("Residual: estimating the boundary-connected poster surface...")
    owned = _owned_alpha(graph)
    background_connected, background_report = _connected_page_background(image_rgb, owned)
    surface, surface_report = _fit_bottom_surface(image_rgb, background_connected)
    emit("Residual: reconciling saliency outside existing pixel ownership...")
    saliency, salient_binary, saliency_report = _saliency_map(
        image_rgb, surface, background_connected, owned
    )
    labels, centres, colour_report = _colour_labels(
        image_rgb, salient_binary, max_clusters=max_colour_clusters
    )
    segments, initially_rejected, segment_report = _extract_segments(
        labels, salient_binary, owned, image_rgb, surface
    )
    segments, consolidation_report = _consolidate_segments(
        segments, (width, height)
    )

    # Top-like details claim visible pixels before bottom panels.  A panel may
    # synthesize support behind them, but its visible alpha keeps child holes.
    priority = {"micro_detail": 0, "line": 1, "frame": 2, "decoration": 3, "panel": 4}
    segments.sort(
        key=lambda item: (
            priority.get(item.kind, 3),
            item.bbox[1],
            item.bbox[0],
            -item.area,
        )
    )
    existing_ids = set(graph.node_map())
    existing_proposals = {item.proposal_id for item in graph.proposals}
    nodes: list[ElementNode] = []
    proposals: list[ProposalRecord] = []
    claimed = owned > 0
    claimed_owner = np.full((height, width), -1, dtype=np.int32)
    assigned_salient = np.zeros((height, width), dtype=np.uint8)
    rejected_salient = np.zeros((height, width), dtype=np.uint8)
    node_serial = 0
    proposal_serial = 0

    def proposal_id() -> str:
        nonlocal proposal_serial
        while True:
            proposal_serial += 1
            value = f"RESIDUAL_PROPOSAL_{proposal_serial:05d}"
            if value not in existing_proposals:
                existing_proposals.add(value)
                return value

    def node_id() -> str:
        nonlocal node_serial
        while True:
            node_serial += 1
            value = f"RESIDUAL_{node_serial:05d}"
            if value not in existing_ids:
                existing_ids.add(value)
                return value

    for mask, reason in initially_rejected:
        pid = proposal_id()
        box = _bbox(mask)
        proposals.append(
            ProposalRecord(
                pid,
                "poster_surface_residual",
                "micro_detail",
                box,
                0.25,
                status="rejected",
                reason=reason,
                evidence={"pixel_count": int(np.count_nonzero(mask))},
            )
        )
        rejected_salient[mask & (salient_binary > 0)] = 1

    for segment in segments:
        pid = proposal_id()
        raw_mask = segment.mask & (owned == 0)
        visible_mask = raw_mask & ~claimed
        if not np.any(visible_mask):
            owners = claimed_owner[raw_mask & (claimed_owner >= 0)]
            if owners.size:
                owner = nodes[int(np.bincount(owners).argmax())]
                proposals.append(
                    ProposalRecord(
                        pid,
                        "poster_surface_residual",
                        segment.kind,
                        segment.bbox,
                        segment.confidence,
                        status="assigned",
                        owner_ids=[owner.element_id],
                        reason="Residual evidence is already owned by an earlier top element",
                        evidence=segment.metrics,
                    )
                )
            else:
                proposals.append(
                    ProposalRecord(
                        pid,
                        "poster_surface_residual",
                        segment.kind,
                        segment.bbox,
                        0.25,
                        status="rejected",
                        reason="Residual support is entirely covered by existing ownership",
                        evidence=segment.metrics,
                    )
                )
                rejected_salient[raw_mask & (salient_binary > 0)] = 1
            continue

        full_mask = visible_mask.copy()
        envelope_mask = raw_mask.copy()
        hidden_report = {"child_holes_filled": 0, "design_holes_preserved": 0}
        synthesized = False
        if segment.auto_safe and segment.kind == "panel":
            envelope_mask, hidden_report = _panel_envelope(raw_mask, owned | claimed)
            full_mask = envelope_mask > 0
            visible_mask = full_mask & ~claimed
            synthesized = bool(np.any(full_mask & claimed))
        else:
            full_mask = full_mask > 0
            visible_mask = visible_mask > 0
        box = _bbox(full_mask)
        if box == (0, 0, 0, 0):
            continue
        x0, y0, x1, y1 = box
        visible_crop = _alpha_crop(visible_mask, box)
        full_crop = _alpha_crop(full_mask, box)
        envelope_box = _bbox(envelope_mask)
        envelope_crop = _alpha_crop(envelope_mask, envelope_box)
        geometry_requires_clean_reference = segment.kind in {"panel", "frame", "line"}
        candidate_auto_safe = segment.auto_safe and visible_crop.nonzero_pixels >= 3
        # Residual colour segmentation does not carry the reference-vs-source
        # carve ledger required by production geometry. It remains useful as a
        # review candidate, but may not claim automatic move safety while
        # unrecognised text or prices could still be baked into visible pixels.
        auto_safe = candidate_auto_safe and not geometry_requires_clean_reference
        if candidate_auto_safe:
            foreground_rgb, synthesis_report = _fit_flat_rgb(image_rgb, raw_mask, box)
        else:
            foreground_rgb = image_rgb[y0:y1, x0:x1].copy()
            synthesis_report = {"method": "observed_source_rgb_unresolved"}
        rgba = np.dstack([foreground_rgb, full_crop.alpha.copy()])
        rgba[full_crop.alpha == 0, :3] = 0
        nid = node_id()
        z_band = {
            "panel": z_start,
            "frame": z_start + 30_000,
            "line": z_start + 40_000,
            "decoration": z_start + 50_000,
            "micro_detail": z_start + 60_000,
        }.get(segment.kind, z_start + 50_000)
        node = ElementNode(
            nid,
            f"RESIDUAL {segment.kind.upper()} {node_serial:03d}",
            segment.kind,
            visible_crop,
            z_band + node_serial,
            semantic_envelope=envelope_crop,
            full_support=full_crop,
            removal_footprint=_removal_footprint(full_mask, box),
            confidence=segment.confidence,
            review_status="auto_confirmed" if auto_safe else "unresolved",
            move_safe=auto_safe,
            occluded=synthesized,
            synthesized_hidden_pixels=synthesized,
            rgba=rgba,
            evidence=[{"proposal_id": pid, "source": "poster_surface_residual"}],
            metadata={
                **segment.metrics,
                "surface_reconciliation": True,
                "exclusive_child_exclusion": True,
                "hidden_support": hidden_report,
                "rgb_reconstruction": synthesis_report,
                **(
                    {
                        "geometry_cleanliness": {
                            "policy": "reference_surface_delta_e_carve_v1",
                            "status": "unsafe",
                            "reference_type": "none",
                            "reason": (
                                "residual_geometry_has_no_validated_clean_reference_"
                                "and_contamination_carve_ledger"
                            ),
                        },
                        "requires_manual_review": True,
                    }
                    if geometry_requires_clean_reference
                    else {}
                ),
            },
        )
        nodes.append(node)
        proposals.append(
            ProposalRecord(
                pid,
                "poster_surface_residual",
                segment.kind,
                segment.bbox,
                segment.confidence,
                status="assigned",
                owner_ids=[nid],
                reason=(
                    "Strict flat geometry passed automatic move-safety gates"
                    if auto_safe
                    else (
                        "Residual geometry lacks a validated clean reference and "
                        "contamination-carve ledger; manual review is required"
                        if geometry_requires_clean_reference
                        else "Residual is significant but requires visual ownership review"
                    )
                ),
                evidence={
                    **segment.metrics,
                    **(
                        {
                            "automatic_move_safety": "rejected",
                            "automatic_move_safety_reason": (
                                "missing_validated_clean_reference_and_"
                                "contamination_carve_ledger"
                            ),
                        }
                        if geometry_requires_clean_reference
                        else {}
                    ),
                },
            )
        )
        new_index = len(nodes) - 1
        claimed_owner[visible_mask] = new_index
        claimed |= visible_mask
        assigned_salient[visible_mask & (salient_binary > 0)] = 1

    accounted = (assigned_salient > 0) | (rejected_salient > 0)
    leftover = (salient_binary > 0) & ~accounted
    # The segmentation labels cover every salient pixel. Any remainder can only
    # be closure/ownership bookkeeping; account it explicitly rather than hide
    # it in a global catch-all layer.
    count, remainder_labels, stats, _ = cv2.connectedComponentsWithStats(
        leftover.astype(np.uint8), 8
    )
    for component in range(1, count):
        remainder = remainder_labels == component
        area = int(stats[component, cv2.CC_STAT_AREA])
        pid = proposal_id()
        if area <= 2:
            proposals.append(
                ProposalRecord(
                    pid,
                    "poster_surface_residual_reconciliation",
                    "micro_detail",
                    _bbox(remainder),
                    0.20,
                    status="rejected",
                    reason="Final reconciliation speck is below smallest useful element",
                    evidence={"pixel_count": area},
                )
            )
            rejected_salient[remainder] = 1
            continue
        box = _bbox(remainder)
        x0, y0, x1, y1 = box
        crop = _alpha_crop(remainder, box)
        rgba = np.dstack([image_rgb[y0:y1, x0:x1].copy(), crop.alpha.copy()])
        nid = node_id()
        node = ElementNode(
            nid,
            f"RESIDUAL MICRO REVIEW {node_serial:03d}",
            "micro_detail",
            crop,
            z_start + 70_000 + node_serial,
            full_support=crop,
            removal_footprint=_removal_footprint(remainder, box),
            confidence=0.40,
            review_status="unresolved",
            move_safe=False,
            rgba=rgba,
            evidence=[{"proposal_id": pid, "source": "poster_surface_residual_reconciliation"}],
            metadata={"final_reconciliation": True, "pixel_count": area},
        )
        nodes.append(node)
        proposals.append(
            ProposalRecord(
                pid,
                "poster_surface_residual_reconciliation",
                "micro_detail",
                box,
                0.40,
                status="assigned",
                owner_ids=[nid],
                reason="Significant final residual retained for human review",
                evidence={"pixel_count": area},
            )
        )
        assigned_salient[remainder] = 1

    # Assigned ownership takes precedence over a rejected colour-fragment
    # proposal when their one-pixel closures overlap. Keep the pixel ledger
    # mutually exclusive as well as complete.
    rejected_salient[assigned_salient > 0] = 0
    final_accounted = (assigned_salient > 0) | (rejected_salient > 0)
    salient_count = int(np.count_nonzero(salient_binary))
    auto_nodes = sum(node.review_status == "auto_confirmed" for node in nodes)
    downgraded_geometry_nodes = [
        node.element_id
        for node in nodes
        if node.kind in {"panel", "frame", "line"}
        and isinstance(node.metadata.get("geometry_cleanliness"), dict)
        and node.metadata["geometry_cleanliness"].get("status") == "unsafe"
    ]
    report = {
        "backend": "boundary-connected poster surface + Lab saliency residual reconciliation",
        "background": background_report,
        "surface": surface_report,
        "saliency": saliency_report,
        "colour_segmentation": {**colour_report, "centres_lab": centres},
        "segmentation": {
            **segment_report,
            "consolidation": consolidation_report,
        },
        "existing_owned_pixels": int(np.count_nonzero(owned)),
        "node_count": len(nodes),
        "auto_safe_node_count": auto_nodes,
        "unresolved_node_count": len(nodes) - auto_nodes,
        "residual_geometry_cleanliness_policy": (
            "panel/frame/line remain unresolved until a validated clean reference "
            "and reference-surface contamination carve ledger both pass"
        ),
        "residual_geometry_downgraded_count": len(downgraded_geometry_nodes),
        "residual_geometry_downgraded_node_ids": downgraded_geometry_nodes,
        "kind_counts": {
            kind: sum(node.kind == kind for node in nodes)
            for kind in sorted({node.kind for node in nodes})
        },
        "proposal_count": len(proposals),
        "assigned_proposal_count": sum(item.status == "assigned" for item in proposals),
        "rejected_proposal_count": sum(item.status == "rejected" for item in proposals),
        "salient_pixel_accounting": {
            "total": salient_count,
            "assigned": int(np.count_nonzero((assigned_salient > 0) & (salient_binary > 0))),
            "rejected": int(np.count_nonzero((rejected_salient > 0) & (salient_binary > 0))),
            "unaccounted": int(np.count_nonzero((salient_binary > 0) & ~final_accounted)),
        },
        "rejected_reason_counts": dict(
            sorted(
                Counter(
                    item.reason or "unspecified"
                    for item in proposals
                    if item.status == "rejected"
                ).items()
            )
        ),
        "salient_fraction": round(salient_count / max(1, width * height), 7),
        "new_visible_pixel_count": int(np.count_nonzero(claimed & (owned == 0))),
        "largest_visible_fraction": round(
            max((node.visible_alpha.nonzero_pixels for node in nodes), default=0)
            / max(1, width * height),
            7,
        ),
        "limitations": (
            "Residual nodes recover significant unowned raster regions but do not infer semantic "
            "names. Ambiguous decoration/micro details intentionally remain review-required."
        ),
    }
    if report["salient_pixel_accounting"]["unaccounted"]:
        raise RuntimeError("Residual reconciliation left salient pixels unaccounted")
    return ResidualResult(
        nodes,
        proposals,
        surface,
        background_connected,
        saliency,
        report,
    )


__all__ = ["ResidualResult", "reconcile_residual_elements"]
