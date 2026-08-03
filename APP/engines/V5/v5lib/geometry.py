from __future__ import annotations

import math
import re
import unicodedata

import cv2
import numpy as np

from .model import Box


def mask_bbox(mask: np.ndarray) -> Box:
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def box_intersection(a: Box, b: Box) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def box_iou(a: Box, b: Box) -> float:
    intersection = box_intersection(a, b)
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return intersection / union if union else 0.0


def mask_containment(inner: np.ndarray, outer: np.ndarray) -> float:
    area = int(inner.sum())
    return int(np.logical_and(inner, outer).sum()) / area if area else 0.0


def expand_box(box: Box, pixels: int, width: int, height: int) -> Box:
    return (
        max(0, box[0] - pixels),
        max(0, box[1] - pixels),
        min(width, box[2] + pixels),
        min(height, box[3] + pixels),
    )


def box_mask(box: Box, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    result[box[1] : box[3], box[0] : box[2]] = True
    return result


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def close_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def connected_component_cleanup(
    mask: np.ndarray,
    *,
    minimum_area: int,
    keep_largest_if_empty: bool = True,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.where(areas >= minimum_area)[0] + 1
    if not len(keep) and keep_largest_if_empty:
        keep = np.array([int(np.argmax(areas)) + 1])
    return np.isin(labels, keep)


def bbox_distance(a: Box, b: Box) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


def vertical_overlap_ratio(a: Box, b: Box) -> float:
    overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    denominator = max(1, min(a[3] - a[1], b[3] - b[1]))
    return overlap / denominator


def horizontal_gap(a: Box, b: Box) -> int:
    return max(a[0] - b[2], b[0] - a[2], 0)


def safe_name(value: str, fallback: str = "layer") -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", value)
    value = re.sub(r"\s+", " ", value).strip(" ._")
    return value[:80] or fallback
