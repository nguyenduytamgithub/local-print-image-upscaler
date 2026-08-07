"""Deterministic grouping of redundant SAM masks for poster-like artwork.

The module deliberately performs no model loading and no file I/O.  It accepts
source-resolution RGB pixels plus raw SAM masks, then uses connected components,
deduplication, containment, row alignment, colour contrast, proximity and
symmetry to produce a bounded hierarchy of :class:`LayerSpec` objects.

The labels are geometric and positional.  OCR/object detectors may replace the
display names later, but semantic guesses are not required for grouping.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from .model import Detection, LayerSpec, MaskCandidate, TextRegion


POSTER_GROUP_THRESHOLDS: dict[str, float] = {
    "score_floor": 0.90,
    "component_area_fraction_floor": 0.00030,
    "duplicate_iou": 0.82,
    "duplicate_small_coverage": 0.94,
    "duplicate_area_ratio_floor": 0.72,
    "owner_area_fraction_floor": 0.035,
    "owner_card_area_fraction": 0.075,
    "owner_bar_aspect_floor": 3.0,
    "owner_bbox_fill_floor": 0.55,
    "owner_assignment_coverage": 0.72,
    "row_lab_distance": 32.0,
    "row_height_ratio_floor": 0.62,
    "row_vertical_overlap": 0.58,
    "row_gap_in_heights": 0.78,
    "wide_row_component_aspect": 4.2,
    "icon_component_area_fraction_floor": 0.00045,
    "icon_group_area_fraction_floor": 0.0025,
    "standalone_component_area_fraction_floor": 0.0012,
    "icon_bbox_gap_fraction_of_diagonal": 0.023,
    "icon_owner_lab_contrast": 18.0,
    "icon_group_aspect_max": 3.8,
    "icon_min_bbox_side_fraction": 0.048,
    "guided_halo_fraction": 0.0075,
}


@dataclass(slots=True)
class _Candidate:
    raw_id: int
    component_id: int
    x: int
    y: int
    width: int
    height: int
    area: int
    score: float
    crop: np.ndarray
    median_lab: np.ndarray

    @property
    def key(self) -> str:
        return f"sam_{self.raw_id:04d}_cc_{self.component_id:03d}"

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def centre_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def centre_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)

    @property
    def bbox_fill(self) -> float:
        return self.area / max(1, self.width * self.height)


@dataclass(slots=True)
class _Draft:
    token: str
    role: str
    mask: np.ndarray
    members: list[_Candidate]
    score: float
    salience: float
    optional: bool
    parent_token: str | None = None


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[b] = a

    def groups(self) -> list[list[int]]:
        grouped: dict[int, list[int]] = {}
        for index in range(len(self.parent)):
            grouped.setdefault(self.find(index), []).append(index)
        return list(grouped.values())


def _validate_inputs(
    image_rgb: np.ndarray,
    raw_masks: Sequence[np.ndarray] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    max_layers: int,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("image_rgb must have shape (height, width, 3).")
    if image.dtype != np.uint8:
        raise ValueError("image_rgb must use uint8 RGB values.")
    if isinstance(max_layers, bool) or not isinstance(max_layers, int) or max_layers < 1:
        raise ValueError("max_layers must be a positive integer.")

    if isinstance(raw_masks, np.ndarray):
        if raw_masks.ndim == 2:
            masks = [raw_masks]
        elif raw_masks.ndim == 3:
            masks = [raw_masks[index] for index in range(raw_masks.shape[0])]
        else:
            raise ValueError("raw_masks must be a sequence of 2-D masks or an (N,H,W) array.")
    else:
        masks = list(raw_masks)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(masks) != len(score_array):
        raise ValueError("raw_masks and scores must have the same length.")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite.")

    shape = image.shape[:2]
    normalized: list[np.ndarray] = []
    for index, mask in enumerate(masks):
        array = np.asarray(mask)
        if array.ndim > 2:
            array = np.squeeze(array)
        if array.ndim != 2 or array.shape != shape:
            raise ValueError(f"raw_masks[{index}] has shape {array.shape}; expected {shape}.")
        normalized.append(array.astype(bool, copy=False))
    return image, normalized, score_array


def _candidate_full(candidate: _Candidate, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    result[candidate.y : candidate.bottom, candidate.x : candidate.right] = candidate.crop
    return result


def _candidate_union(candidates: Iterable[_Candidate], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        result[candidate.y : candidate.bottom, candidate.x : candidate.right] |= candidate.crop
    return result


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _candidate_bbox(candidates: Sequence[_Candidate]) -> tuple[int, int, int, int]:
    return (
        min(item.x for item in candidates),
        min(item.y for item in candidates),
        max(item.right for item in candidates),
        max(item.bottom for item in candidates),
    )


def _intersection_area(first: _Candidate, second: _Candidate) -> int:
    x0, y0 = max(first.x, second.x), max(first.y, second.y)
    x1, y1 = min(first.right, second.right), min(first.bottom, second.bottom)
    if x1 <= x0 or y1 <= y0:
        return 0
    a = first.crop[y0 - first.y : y1 - first.y, x0 - first.x : x1 - first.x]
    b = second.crop[y0 - second.y : y1 - second.y, x0 - second.x : x1 - second.x]
    return int(np.logical_and(a, b).sum())


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / union if union else 0.0


def _external_fill(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(result, contours, -1, 1, thickness=cv2.FILLED)
    return result.astype(bool)


def _guided_halo(
    mask: np.ndarray,
    image_lab: np.ndarray,
    *,
    radius: int,
    difference: float,
) -> np.ndarray:
    """Expand into pixels unlike the local outer field, retaining outlines/shadows."""

    if radius <= 0 or not mask.any():
        return mask.copy()
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    expanded = cv2.dilate(mask.astype(np.uint8), inner_kernel).astype(bool)
    outer_radius = radius * 2
    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (outer_radius * 2 + 1,) * 2
    )
    outer = cv2.dilate(mask.astype(np.uint8), outer_kernel).astype(bool)
    ring = outer & ~expanded
    if int(ring.sum()) < 16:
        return mask.copy()
    reference = np.median(image_lab[ring], axis=0)
    colour_distance = np.linalg.norm(image_lab - reference, axis=2)
    return mask | (expanded & (colour_distance >= difference))


def _split_components(
    masks: Sequence[np.ndarray],
    scores: np.ndarray,
    image_lab: np.ndarray,
) -> tuple[list[_Candidate], int, int]:
    height, width = image_lab.shape[:2]
    area_floor = max(
        4,
        int(round(POSTER_GROUP_THRESHOLDS["component_area_fraction_floor"] * height * width)),
    )
    candidates: list[_Candidate] = []
    rejected_score = 0
    rejected_area = 0
    for raw_id, (mask, score) in enumerate(zip(masks, scores, strict=True)):
        if float(score) < POSTER_GROUP_THRESHOLDS["score_floor"]:
            rejected_score += 1
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for component_id in range(1, count):
            x, y, component_width, component_height, area = map(int, stats[component_id])
            if area < area_floor:
                rejected_area += 1
                continue
            crop = labels[y : y + component_height, x : x + component_width] == component_id
            median_lab = np.median(
                image_lab[y : y + component_height, x : x + component_width][crop], axis=0
            )
            candidates.append(
                _Candidate(
                    raw_id=raw_id,
                    component_id=component_id,
                    x=x,
                    y=y,
                    width=component_width,
                    height=component_height,
                    area=area,
                    score=float(score),
                    crop=crop,
                    median_lab=median_lab,
                )
            )
    return candidates, rejected_score, rejected_area


def _deduplicate(candidates: Sequence[_Candidate]) -> tuple[list[_Candidate], int]:
    ordered = sorted(
        candidates,
        key=lambda item: (-item.score, -item.area, item.raw_id, item.component_id),
    )
    kept: list[_Candidate] = []
    duplicates = 0
    for candidate in ordered:
        duplicate = False
        for previous in kept:
            intersection = _intersection_area(candidate, previous)
            if not intersection:
                continue
            iou = intersection / (candidate.area + previous.area - intersection)
            coverage = intersection / min(candidate.area, previous.area)
            area_ratio = min(candidate.area, previous.area) / max(candidate.area, previous.area)
            if iou >= POSTER_GROUP_THRESHOLDS["duplicate_iou"] or (
                coverage >= POSTER_GROUP_THRESHOLDS["duplicate_small_coverage"]
                and area_ratio >= POSTER_GROUP_THRESHOLDS["duplicate_area_ratio_floor"]
            ):
                duplicate = True
                duplicates += 1
                break
        if not duplicate:
            kept.append(candidate)
    return kept, duplicates


def _row_related(first: _Candidate, second: _Candidate) -> bool:
    shorter, taller = min(first.height, second.height), max(first.height, second.height)
    if shorter / max(1, taller) < POSTER_GROUP_THRESHOLDS["row_height_ratio_floor"]:
        return False
    vertical_overlap = max(0, min(first.bottom, second.bottom) - max(first.y, second.y))
    if vertical_overlap / max(1, shorter) < POSTER_GROUP_THRESHOLDS["row_vertical_overlap"]:
        return False
    horizontal_gap = max(0, max(first.x, second.x) - min(first.right, second.right))
    if horizontal_gap > POSTER_GROUP_THRESHOLDS["row_gap_in_heights"] * taller:
        return False
    return (
        float(np.linalg.norm(first.median_lab - second.median_lab))
        <= POSTER_GROUP_THRESHOLDS["row_lab_distance"]
    )


def _row_groups(candidates: Sequence[_Candidate]) -> list[list[_Candidate]]:
    if len(candidates) < 2:
        return []
    forest = _DisjointSet(len(candidates))
    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            if _row_related(candidates[first], candidates[second]):
                forest.union(first, second)
    result: list[list[_Candidate]] = []
    for indexes in forest.groups():
        group = [candidates[index] for index in indexes]
        if len(group) < 2:
            continue
        if len(group) >= 4:
            median_width = float(np.median([item.width for item in group]))
            trimmed = [
                item
                for item in group
                if not (item.width > 1.9 * median_width and item.aspect > 1.40)
            ]
            if len(trimmed) >= 2:
                group = trimmed
        x0, y0, x1, y1 = _candidate_bbox(group)
        aspect = (x1 - x0) / max(1, y1 - y0)
        if len(group) >= 3 or aspect >= 1.05:
            result.append(group)
    return result


def _bbox_gap(first: _Candidate, second: _Candidate) -> float:
    dx = max(0, max(first.x, second.x) - min(first.right, second.right))
    dy = max(0, max(first.y, second.y) - min(first.bottom, second.bottom))
    return math.hypot(dx, dy)


def _spatial_groups(
    candidates: Sequence[_Candidate], shape: tuple[int, int]
) -> list[list[_Candidate]]:
    if not candidates:
        return []
    height, width = shape
    maximum_gap = (
        POSTER_GROUP_THRESHOLDS["icon_bbox_gap_fraction_of_diagonal"]
        * math.hypot(height, width)
    )
    forest = _DisjointSet(len(candidates))
    for first in range(len(candidates)):
        a = candidates[first]
        for second in range(first + 1, len(candidates)):
            b = candidates[second]
            if _bbox_gap(a, b) > maximum_gap:
                continue
            x0, y0 = min(a.x, b.x), min(a.y, b.y)
            x1, y1 = max(a.right, b.right), max(a.bottom, b.bottom)
            if x1 - x0 <= 0.43 * width and y1 - y0 <= 0.30 * height:
                forest.union(first, second)
    return [[candidates[index] for index in indexes] for indexes in forest.groups()]


def _owner_drafts(
    candidates: Sequence[_Candidate],
    shape: tuple[int, int],
    image_lab: np.ndarray,
) -> list[_Draft]:
    height, width = shape
    canvas_area = height * width
    radius = max(
        1,
        int(round(POSTER_GROUP_THRESHOLDS["guided_halo_fraction"] * min(shape) * 0.55)),
    )
    proposals: list[_Draft] = []
    for candidate in candidates:
        area_fraction = candidate.area / canvas_area
        large_card = area_fraction >= POSTER_GROUP_THRESHOLDS["owner_card_area_fraction"]
        wide_bar = candidate.aspect >= POSTER_GROUP_THRESHOLDS["owner_bar_aspect_floor"]
        if not (
            area_fraction >= POSTER_GROUP_THRESHOLDS["owner_area_fraction_floor"]
            and (large_card or wide_bar)
            and candidate.bbox_fill >= POSTER_GROUP_THRESHOLDS["owner_bbox_fill_floor"]
        ):
            continue

        members = [candidate]
        core = _candidate_full(candidate, shape)
        if wide_bar:
            for sibling in candidates:
                if sibling.key == candidate.key or sibling.raw_id != candidate.raw_id:
                    continue
                overlap = max(0, min(candidate.bottom, sibling.bottom) - max(candidate.y, sibling.y))
                if overlap / max(1, min(candidate.height, sibling.height)) < 0.72:
                    continue
                if sibling.area / canvas_area < 0.003:
                    continue
                gap = max(0, max(candidate.x, sibling.x) - min(candidate.right, sibling.right))
                if gap <= 0.18 * width:
                    members.append(sibling)
                    core |= _candidate_full(sibling, shape)
            points = np.column_stack(np.nonzero(core))[:, ::-1].astype(np.int32)
            if len(points) >= 3:
                convex = np.zeros(shape, dtype=np.uint8)
                cv2.fillConvexPoly(convex, cv2.convexHull(points), 1)
                core = convex.astype(bool)
        else:
            core = _external_fill(core)
        mask = _guided_halo(core, image_lab, radius=radius, difference=12.0)
        proposals.append(
            _Draft(
                token=f"owner:{candidate.key}",
                role="owner_panel",
                mask=mask,
                members=members,
                score=max(item.score for item in members),
                salience=float(mask.mean()),
                optional=False,
            )
        )

    # Nested/duplicate SAM panels are common; keep the best boundary once.
    accepted: list[_Draft] = []
    for proposal in sorted(
        proposals,
        key=lambda item: (-item.score, -int(item.mask.sum()), item.token),
    ):
        duplicate = False
        for previous in accepted:
            intersection = int(np.logical_and(proposal.mask, previous.mask).sum())
            if not intersection:
                continue
            smaller = min(int(proposal.mask.sum()), int(previous.mask.sum()))
            if _mask_iou(proposal.mask, previous.mask) >= 0.78 or intersection / smaller >= 0.92:
                duplicate = True
                break
        if not duplicate:
            accepted.append(proposal)
    return sorted(accepted, key=lambda item: (_mask_bbox(item.mask)[1], _mask_bbox(item.mask)[0]))


def _candidate_owner(
    candidate: _Candidate, owners: Sequence[_Draft]
) -> tuple[int | None, float]:
    best_index: int | None = None
    best_coverage = 0.0
    for index, owner in enumerate(owners):
        # Candidate crops are already clipped to the shared canvas, so coverage
        # can be measured locally without allocating another full-resolution
        # mask for every candidate/owner pair.
        owner_crop = owner.mask[
            candidate.y : candidate.bottom,
            candidate.x : candidate.right,
        ]
        coverage = float(owner_crop[candidate.crop].sum()) / candidate.area
        if coverage > best_coverage:
            best_index, best_coverage = index, coverage
    if best_coverage < POSTER_GROUP_THRESHOLDS["owner_assignment_coverage"]:
        return None, best_coverage
    return best_index, best_coverage


def _child_drafts(
    candidates: Sequence[_Candidate],
    owners: Sequence[_Draft],
    shape: tuple[int, int],
    image_lab: np.ndarray,
) -> tuple[list[_Draft], dict[str, list[list[str]]]]:
    height, width = shape
    canvas_area = height * width
    owner_source_keys = {item.key for owner in owners for item in owner.members}
    assigned: dict[int, list[_Candidate]] = {index: [] for index in range(len(owners))}
    for candidate in candidates:
        owner_index, _ = _candidate_owner(candidate, owners)
        if owner_index is not None and candidate.key not in owner_source_keys:
            assigned[owner_index].append(candidate)

    radius = max(
        1,
        int(round(POSTER_GROUP_THRESHOLDS["guided_halo_fraction"] * min(shape))),
    )
    drafts: list[_Draft] = []
    row_debug: dict[str, list[list[str]]] = {}
    for owner_index, children in assigned.items():
        owner = owners[owner_index]
        rows = _row_groups(children)
        row_keys = {item.key for row in rows for item in row}
        row_keys.update(
            item.key
            for item in children
            if item.aspect >= POSTER_GROUP_THRESHOLDS["wide_row_component_aspect"]
            and item.height < 0.14 * height
        )
        row_debug[owner.token] = [[item.key for item in row] for row in rows]
        owner_median = np.median(image_lab[owner.mask], axis=0)
        eligible: list[_Candidate] = []
        for item in children:
            if item.key in row_keys:
                continue
            if item.area / canvas_area < POSTER_GROUP_THRESHOLDS[
                "icon_component_area_fraction_floor"
            ]:
                continue
            contrast = float(np.linalg.norm(item.median_lab - owner_median))
            if contrast >= POSTER_GROUP_THRESHOLDS["icon_owner_lab_contrast"]:
                eligible.append(item)

        for group_number, cluster in enumerate(_spatial_groups(eligible, shape), 1):
            total_area = sum(item.area for item in cluster)
            if total_area / canvas_area < POSTER_GROUP_THRESHOLDS[
                "icon_group_area_fraction_floor"
            ]:
                continue
            x0, y0, x1, y1 = _candidate_bbox(cluster)
            aspect = (x1 - x0) / max(1, y1 - y0)
            if not (0.25 <= aspect <= POSTER_GROUP_THRESHOLDS["icon_group_aspect_max"]):
                continue
            if min(x1 - x0, y1 - y0) < (
                POSTER_GROUP_THRESHOLDS["icon_min_bbox_side_fraction"] * min(shape)
            ):
                # A high-confidence but very narrow strip is often one ribbon,
                # handle or stroke from a larger illustration.  Keeping it in
                # the owner is more useful than advertising a partial layer.
                continue
            core = _candidate_union(cluster, shape)
            mask = _guided_halo(core, image_lab, radius=radius, difference=15.0) & owner.mask
            contrast = float(
                np.mean([np.linalg.norm(item.median_lab - owner_median) for item in cluster])
            )
            drafts.append(
                _Draft(
                    token=f"child:{owner.token}:{group_number:03d}",
                    role="promoted_child_object",
                    mask=mask,
                    members=list(cluster),
                    score=max(item.score for item in cluster),
                    salience=float(mask.mean()) * (1.0 + contrast / 64.0),
                    optional=True,
                    parent_token=owner.token,
                )
            )
    return drafts, row_debug


def _merge_symmetric(
    drafts: list[_Draft], shape: tuple[int, int]
) -> list[_Draft]:
    height, width = shape
    used: set[int] = set()
    result: list[_Draft] = []
    for first, draft in enumerate(drafts):
        if first in used or draft.role == "standalone_row":
            if first not in used:
                used.add(first)
                result.append(draft)
            continue
        ax0, ay0, ax1, ay1 = _mask_bbox(draft.mask)
        acx, acy = (ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0
        area_a = int(draft.mask.sum())
        partner: int | None = None
        for second in range(first + 1, len(drafts)):
            other = drafts[second]
            if second in used or other.role == "standalone_row":
                continue
            bx0, by0, bx1, by1 = _mask_bbox(other.mask)
            bcx, bcy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
            area_b = int(other.mask.sum())
            if (
                abs((acx + bcx) - width) <= 0.065 * width
                and abs(acy - bcy) <= 0.055 * height
                and min(area_a, area_b) / max(1, max(area_a, area_b)) >= 0.55
            ):
                partner = second
                break
        if partner is None:
            used.add(first)
            result.append(draft)
            continue
        other = drafts[partner]
        used.update((first, partner))
        members = draft.members + other.members
        merged_mask = draft.mask | other.mask
        result.append(
            _Draft(
                token=f"symmetric:{draft.token}:{other.token}",
                role="symmetric_decoration",
                mask=merged_mask,
                members=members,
                score=max(draft.score, other.score),
                salience=float(merged_mask.mean()),
                optional=True,
            )
        )
    return result


def _standalone_drafts(
    candidates: Sequence[_Candidate],
    owners: Sequence[_Draft],
    shape: tuple[int, int],
    image_lab: np.ndarray,
) -> tuple[list[_Draft], int]:
    height, width = shape
    canvas_area = height * width
    owner_source_keys = {item.key for owner in owners for item in owner.members}
    free = [
        item
        for item in candidates
        if item.key not in owner_source_keys
        and _candidate_owner(item, owners)[0] is None
        and item.area / canvas_area
        >= POSTER_GROUP_THRESHOLDS["standalone_component_area_fraction_floor"]
    ]
    rows = _row_groups(free)
    row_keys = {item.key for row in rows for item in row}
    radius = max(
        1,
        int(round(POSTER_GROUP_THRESHOLDS["guided_halo_fraction"] * min(shape))),
    )
    drafts: list[_Draft] = []
    for row_number, row in enumerate(rows, 1):
        core = _candidate_union(row, shape)
        mask = _guided_halo(core, image_lab, radius=radius, difference=14.0)
        drafts.append(
            _Draft(
                token=f"row:{row_number:03d}",
                role="standalone_row",
                mask=mask,
                members=list(row),
                score=max(item.score for item in row),
                salience=float(mask.mean()),
                optional=False,
            )
        )
    for object_number, cluster in enumerate(
        _spatial_groups([item for item in free if item.key not in row_keys], shape), 1
    ):
        total_area = sum(item.area for item in cluster)
        if total_area / canvas_area < POSTER_GROUP_THRESHOLDS[
            "icon_group_area_fraction_floor"
        ]:
            continue
        core = _candidate_union(cluster, shape)
        mask = _guided_halo(core, image_lab, radius=radius, difference=14.0)
        drafts.append(
            _Draft(
                token=f"standalone:{object_number:03d}",
                role="standalone_object",
                mask=mask,
                members=list(cluster),
                score=max(item.score for item in cluster),
                salience=float(mask.mean()),
                optional=True,
            )
        )

    drafts = _merge_symmetric(drafts, shape)
    # A small mark immediately above and horizontally inside the dominant row
    # is probably an accent/diacritic belonging to that row, not a new layer.
    rows_after_merge = [item for item in drafts if item.role == "standalone_row"]
    if rows_after_merge:
        dominant = max(rows_after_merge, key=lambda item: int(item.mask.sum()))
        hx0, hy0, hx1, hy1 = _mask_bbox(dominant.mask)
        additions: list[_Draft] = []
        for item in drafts:
            if item is dominant or item.role == "standalone_row":
                continue
            x0, y0, x1, y1 = _mask_bbox(item.mask)
            if (
                x1 - x0 < 0.25 * width
                and x0 >= hx0 - 0.03 * width
                and x1 <= hx1 + 0.03 * width
                and (y0 + y1) / 2.0 < (hy0 + hy1) / 2.0
                and y0 <= hy0 + 0.20 * (hy1 - hy0)
                and max(0, hy0 - y1) <= 0.035 * height
                and float(item.mask.mean()) < 0.01
            ):
                additions.append(item)
        if additions:
            for item in additions:
                dominant.mask |= item.mask
                dominant.members.extend(item.members)
                dominant.score = max(dominant.score, item.score)
                dominant.salience = float(dominant.mask.mean())
            drafts = [item for item in drafts if item not in additions]
    return drafts, len(free)


def _position_hint(mask: np.ndarray) -> str:
    height, width = mask.shape
    x0, y0, x1, y1 = _mask_bbox(mask)
    cx = (x0 + x1) / (2.0 * width)
    cy = (y0 + y1) / (2.0 * height)
    horizontal = "left" if cx < 0.38 else "right" if cx > 0.62 else "center"
    if cy < 0.20:
        vertical = "top"
    elif cy < 0.42:
        vertical = "upper"
    elif cy < 0.66:
        vertical = "middle"
    elif cy < 0.86:
        vertical = "lower"
    else:
        vertical = "bottom"
    return f"{vertical}_{horizontal}"


def _select_budget(
    roots: Sequence[_Draft], children: Sequence[_Draft], max_layers: int
) -> tuple[list[_Draft], list[str]]:
    root_priority = {
        "owner_panel": 4,
        "standalone_row": 3,
        "standalone_object": 2,
        "symmetric_decoration": 1,
    }
    ranked_roots = sorted(
        roots,
        key=lambda item: (
            -root_priority.get(item.role, 0),
            -item.salience,
            _mask_bbox(item.mask)[1],
            _mask_bbox(item.mask)[0],
            item.token,
        ),
    )
    kept_roots = ranked_roots[:max_layers]
    kept_tokens = {item.token for item in kept_roots}
    available = max_layers - len(kept_roots)
    eligible_children = sorted(
        (item for item in children if item.parent_token in kept_tokens),
        key=lambda item: (-item.salience, item.token),
    )
    kept_children = eligible_children[:available]
    selected = kept_roots + kept_children
    selected_tokens = {item.token for item in selected}
    dropped = [
        item.token
        for item in list(roots) + list(children)
        if item.token not in selected_tokens
    ]
    return selected, dropped


def _to_layer_specs(
    selected: Sequence[_Draft], all_children: Sequence[_Draft]
) -> list[LayerSpec]:
    selected_tokens = {item.token for item in selected}
    selected_children = [
        item
        for item in all_children
        if item.token in selected_tokens and item.parent_token in selected_tokens
    ]
    child_by_parent: dict[str, list[_Draft]] = {}
    for child in selected_children:
        child_by_parent.setdefault(str(child.parent_token), []).append(child)

    # Parents retain the child footprint.  The V5 renderer can then inpaint the
    # parent under an independently movable child while keeping a real hierarchy.
    for parent in selected:
        for child in child_by_parent.get(parent.token, []):
            parent.mask |= child.mask

    roots = [item for item in selected if item.parent_token is None]
    roots.sort(key=lambda item: (_mask_bbox(item.mask)[1], _mask_bbox(item.mask)[0], item.token))
    selected_children.sort(
        key=lambda item: (
            next((index for index, root in enumerate(roots) if root.token == item.parent_token), 10**6),
            _mask_bbox(item.mask)[1],
            _mask_bbox(item.mask)[0],
            item.token,
        )
    )

    counters = {"owner_panel": 0, "standalone_row": 0, "standalone_object": 0, "symmetric_decoration": 0, "promoted_child_object": 0}
    id_by_token: dict[str, str] = {}
    name_by_token: dict[str, str] = {}
    prefix = {
        "owner_panel": ("poster_panel", "PANEL"),
        "standalone_row": ("poster_row", "ROW GROUP"),
        "standalone_object": ("poster_object", "OBJECT GROUP"),
        "symmetric_decoration": ("poster_decoration", "DECORATION GROUP"),
        "promoted_child_object": ("poster_child", "CHILD OBJECT"),
    }
    for item in roots + selected_children:
        counters[item.role] = counters.get(item.role, 0) + 1
        number = counters[item.role]
        id_prefix, name_prefix = prefix[item.role]
        id_by_token[item.token] = f"{id_prefix}_{number:02d}"
        hint = _position_hint(item.mask).replace("_", " ").upper()
        name_by_token[item.token] = f"{name_prefix} {number:02d} - {hint}"

    layers: list[LayerSpec] = []
    ordered = roots + selected_children
    for display_order, item in enumerate(ordered, 1):
        parent_id = id_by_token.get(str(item.parent_token)) if item.parent_token else None
        category = "detail_group" if item.role == "owner_panel" else "object"
        bbox = _mask_bbox(item.mask)
        height, width = item.mask.shape
        layers.append(
            LayerSpec(
                layer_id=id_by_token[item.token],
                name=name_by_token[item.token],
                category=category,
                mask=item.mask.astype(bool, copy=True),
                score=float(item.score),
                source_ids=sorted({member.raw_id for member in item.members}),
                metadata={
                    "grouping_role": item.role,
                    "layout_hint": _position_hint(item.mask),
                    "parent_id": parent_id,
                    "children": [],
                    "hierarchy_depth": 1 if parent_id else 0,
                    "display_order": display_order,
                    "source_components": sorted({member.key for member in item.members}),
                    "normalized_bbox": [
                        round(bbox[0] / width, 6),
                        round(bbox[1] / height, 6),
                        round(bbox[2] / width, 6),
                        round(bbox[3] / height, 6),
                    ],
                    "optional": bool(item.optional),
                    "salience": round(float(item.salience), 8),
                },
            )
        )
    by_id = {layer.layer_id: layer for layer in layers}
    for layer in layers:
        parent_id = layer.metadata.get("parent_id")
        if parent_id and str(parent_id) in by_id:
            by_id[str(parent_id)].metadata["children"].append(layer.layer_id)
    return layers


def group_poster_layers(
    image_rgb: np.ndarray,
    raw_masks: Sequence[np.ndarray] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    max_layers: int,
) -> tuple[list[LayerSpec], dict]:
    """Group raw SAM masks into a deterministic poster-layer hierarchy.

    Args:
        image_rgb: Source-resolution uint8 RGB array with shape ``(H, W, 3)``.
        raw_masks: Boolean-like SAM masks, each with shape ``(H, W)``.
        scores: SAM quality scores corresponding one-to-one with ``raw_masks``.
        max_layers: Hard maximum number of returned foreground layers.  The
            synthesized/restored background is owned by the calling engine.

    Returns:
        ``(layers, report)`` where ``layers`` contains only foreground
        :class:`LayerSpec` objects and ``report`` is JSON-serializable.
    """

    image, masks, score_array = _validate_inputs(image_rgb, raw_masks, scores, max_layers)
    shape = image.shape[:2]
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    split, rejected_score, rejected_area = _split_components(masks, score_array, image_lab)
    candidates, duplicate_count = _deduplicate(split)
    owners = _owner_drafts(candidates, shape, image_lab)
    children, owner_rows = _child_drafts(candidates, owners, shape, image_lab)
    standalone, free_count = _standalone_drafts(candidates, owners, shape, image_lab)
    selected, dropped = _select_budget(owners + standalone, children, max_layers)
    layers = _to_layer_specs(selected, children)

    covered = np.zeros(shape, dtype=bool)
    top_level = np.zeros(shape, dtype=bool)
    for layer in layers:
        covered |= layer.mask
        if layer.metadata.get("parent_id") is None:
            top_level |= layer.mask
    report: dict[str, object] = {
        "policy": "poster_deterministic_rag_v1",
        "raw_mask_count": len(masks),
        "score_rejected_mask_count": rejected_score,
        "small_component_rejection_count": rejected_area,
        "split_component_count": len(split),
        "deduplicated_candidate_count": len(candidates),
        "duplicate_candidate_count": duplicate_count,
        "owner_panel_count_before_budget": len(owners),
        "standalone_group_count_before_budget": len(standalone),
        "promoted_child_count_before_budget": len(children),
        "free_candidate_count": free_count,
        "selected_layer_count": len(layers),
        "selected_by_role": {
            role: sum(layer.metadata.get("grouping_role") == role for layer in layers)
            for role in sorted({str(layer.metadata.get("grouping_role")) for layer in layers})
        },
        "hierarchy_edges": sum(layer.metadata.get("parent_id") is not None for layer in layers),
        "foreground_coverage_ratio": round(float(covered.mean()), 6),
        "top_level_removal_ratio": round(float(top_level.mean()), 6),
        "max_layers": max_layers,
        "dropped_by_budget": dropped,
        "thresholds": dict(POSTER_GROUP_THRESHOLDS),
        "owner_row_groups": owner_rows,
        "grouping_policy": (
            "Split connected components; remove near-duplicates; detect large panel/card owners; "
            "attach aligned row-like fragments; promote only compact, colour-distinct child groups; "
            "merge free components by row/proximity/symmetry; retain outlines with a local colour-guided halo."
        ),
    }
    return layers, report


@dataclass(slots=True)
class _TextDraft:
    region: TextRegion
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    candidates: list[MaskCandidate]
    match_score: float
    quality: float
    occupancy: float
    seed_support: float
    horizontal_span: float
    protected_removed_fraction: float


@dataclass(slots=True)
class _VisualTextDraft:
    parent_id: str
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    components: list[_Candidate]
    quality: float
    occupancy: float
    contrast: float
    baseline_spread: float


def _copy_layer(layer: LayerSpec) -> LayerSpec:
    return LayerSpec(
        layer_id=layer.layer_id,
        name=layer.name,
        category=layer.category,
        mask=layer.mask.astype(bool, copy=True),
        score=float(layer.score),
        source_ids=list(layer.source_ids),
        label=layer.label,
        text=layer.text,
        metadata=copy.deepcopy(layer.metadata),
        alpha_matte=(
            None
            if layer.alpha_matte is None
            else np.asarray(layer.alpha_matte, dtype=np.float32).copy()
        ),
    )


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _box_intersection(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> int:
    return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _box_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    intersection = _box_intersection(first, second)
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union else 0.0


def _clip_expand_box(
    box: tuple[int, int, int, int],
    padding: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, min(width, int(box[0]) - padding)),
        max(0, min(height, int(box[1]) - padding)),
        max(0, min(width, int(box[2]) + padding)),
        max(0, min(height, int(box[3]) + padding)),
    )


def _candidate_text_match(
    candidate: MaskCandidate, box: tuple[int, int, int, int]
) -> float:
    intersection_box = _box_intersection(candidate.bbox, box)
    if not intersection_box:
        return 0.0
    x0, y0, x1, y1 = box
    intersection = int(candidate.mask[y0:y1, x0:x1].sum())
    if not intersection:
        return 0.0
    inside = intersection / max(1, candidate.area)
    coverage = intersection / max(1, _box_area(box))
    size_balance = min(candidate.area, _box_area(box)) / max(
        candidate.area, _box_area(box)
    )
    return inside * 0.62 + coverage * 0.23 + size_balance * 0.15


def _ocr_contrast_mask(
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    seed: np.ndarray,
) -> np.ndarray:
    """Complete missing glyph pieces inside one OCR line without leaving its box."""

    x0, y0, x1, y1 = box
    roi = image_rgb[y0:y1, x0:x1]
    if roi.size == 0:
        return seed.copy()
    height, width = roi.shape[:2]
    band = max(1, min(height, width) // 10)
    border = np.zeros((height, width), dtype=bool)
    border[:band] = True
    border[-band:] = True
    border[:, :band] = True
    border[:, -band:] = True
    border_pixels = roi[border]
    if not len(border_pixels):
        return seed.copy()

    bins = (border_pixels // 16).astype(np.int16)
    unique, counts = np.unique(bins, axis=0, return_counts=True)
    dominant = unique[int(np.argmax(counts))]
    dominant_pixels = border_pixels[np.all(np.abs(bins - dominant) <= 1, axis=1)]
    background = np.median(dominant_pixels if len(dominant_pixels) else border_pixels, axis=0)
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(
        np.uint8([[np.clip(background, 0, 255)]]), cv2.COLOR_RGB2LAB
    )[0, 0].astype(np.float32)
    distance = np.linalg.norm(roi_lab - background_lab, axis=2)
    border_distance = distance[border]
    threshold = max(14.0, float(np.percentile(border_distance, 78)) + 5.0)
    contrast = distance > threshold
    contrast = cv2.morphologyEx(
        contrast.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)

    local_seed = seed[y0:y1, x0:x1]
    support_radius = max(1, int(round(height * 0.10)))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (support_radius * 2 + 1,) * 2
    )
    seed_support = cv2.dilate(local_seed.astype(np.uint8), support_kernel).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(contrast.astype(np.uint8), 8)
    accepted = np.zeros_like(contrast)
    minimum = max(2, int(round(height * width * 0.00035)))
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        if area < minimum or area > height * width * 0.38:
            continue
        component_mask = labels == component
        touches_seed = bool(np.logical_and(component_mask, seed_support).any())
        glyph_sized = 0.18 * height <= component_height <= height
        if touches_seed or glyph_sized:
            accepted |= component_mask
    result = seed.copy()
    result[y0:y1, x0:x1] |= accepted
    # The OCR box is a hard locality boundary: a SAM mask with distant islands
    # cannot leak those islands into this line layer.
    locality = np.zeros(seed.shape, dtype=bool)
    locality[y0:y1, x0:x1] = True
    return result & locality


def _text_draft(
    image_rgb: np.ndarray,
    candidates: Sequence[MaskCandidate],
    region: TextRegion,
) -> tuple[_TextDraft | None, str | None]:
    height, width = image_rgb.shape[:2]
    canvas_area = height * width
    text = " ".join(str(region.text).split())
    if not text or not any(character.isalnum() for character in text):
        return None, "non_alphanumeric_text"
    if not math.isfinite(float(region.confidence)) or float(region.confidence) < 0:
        return None, "invalid_confidence"
    raw_box = tuple(int(value) for value in region.bbox)
    clipped = _clip_expand_box(raw_box, 0, width, height)
    box_area = _box_area(clipped)
    if clipped[2] - clipped[0] < 3 or clipped[3] - clipped[1] < 3:
        return None, "tiny_box"
    if not 0.00008 <= box_area / canvas_area <= 0.38:
        return None, "implausible_box_area"

    padding = max(2, int(round(min(height, width) * 0.004)))
    box = _clip_expand_box(clipped, padding, width, height)
    ranked = sorted(
        (
            (_candidate_text_match(candidate, box), candidate)
            for candidate in candidates
            if candidate.mask.shape == (height, width)
            and candidate.area / canvas_area < 0.085
        ),
        key=lambda pair: (-pair[0], pair[1].candidate_id),
    )
    matches = [candidate for match, candidate in ranked if match >= 0.43]
    if not matches:
        return None, "no_sam_support"
    best_match = float(ranked[0][0])
    seed = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = box
    for candidate in matches:
        seed[y0:y1, x0:x1] |= candidate.mask[y0:y1, x0:x1]
    if not seed.any():
        return None, "empty_seed"
    completed = _ocr_contrast_mask(image_rgb, box, seed)
    local_area = int(completed[y0:y1, x0:x1].sum())
    occupancy = local_area / max(1, _box_area(box))
    if not 0.015 <= occupancy <= 0.85:
        return None, "implausible_text_occupancy"
    support = int(np.logical_and(completed, seed).sum()) / max(1, int(completed.sum()))
    if support < 0.015:
        return None, "weak_sam_support"
    local_seed = seed[y0:y1, x0:x1]
    seed_y, seed_x = np.where(local_seed)
    horizontal_span = (
        (int(seed_x.max()) - int(seed_x.min()) + 1) / max(1, x1 - x0)
        if len(seed_x)
        else 0.0
    )
    confidence_value = float(region.confidence)
    weak_geometry = support < 0.10 or occupancy < 0.025 or horizontal_span < 0.28
    if confidence_value < 35.0 and weak_geometry:
        return None, "low_confidence_weak_geometry"
    box_width = x1 - x0
    box_height = y1 - y0
    unusually_wide = (
        box_width / max(1, box_height) > 14.0
        or box_width / max(1, width) > 0.85
        or box_area / canvas_area > 0.20
    )
    if unusually_wide and (support < 0.06 or horizontal_span < 0.55):
        return None, "wide_box_weak_geometry"
    confidence = min(1.0, max(0.0, float(region.confidence) / 100.0))
    quality = best_match * 0.58 + confidence * 0.27 + min(1.0, support) * 0.15
    return (
        _TextDraft(
            region=TextRegion(bbox=clipped, text=text, confidence=float(region.confidence)),
            bbox=box,
            mask=completed,
            candidates=matches,
            match_score=best_match,
            quality=float(quality),
            occupancy=float(occupancy),
            seed_support=float(support),
            horizontal_span=float(horizontal_span),
            protected_removed_fraction=0.0,
        ),
        None,
    )


def _subtract_protected_from_text_draft(
    draft: _TextDraft,
    protected_union: np.ndarray,
) -> str | None:
    """Remove already-layered icons/decorations from an OCR text proposal."""

    original_area = int(draft.mask.sum())
    if not original_area or not protected_union.any():
        return None
    cleaned = draft.mask & ~protected_union
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8), 8
    )
    minimum = max(3, int(round(cleaned.size * 0.000003)))
    retained = np.zeros_like(cleaned)
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= minimum:
            retained |= labels == component_id
    retained_area = int(retained.sum())
    if retained_area / max(1, original_area) < 0.12:
        return "protected_overlap_left_weak_text"
    surviving_candidates = [
        candidate
        for candidate in draft.candidates
        if int(np.logical_and(candidate.mask, retained).sum()) / max(1, candidate.area)
        >= 0.08
    ]
    if not surviving_candidates:
        return "protected_overlap_removed_all_sam_support"
    draft.mask = retained
    draft.candidates = surviving_candidates
    draft.protected_removed_fraction = 1.0 - retained_area / max(1, original_area)
    return None


def _deduplicate_text_drafts(
    drafts: Sequence[_TextDraft],
) -> tuple[list[_TextDraft], int]:
    accepted: list[_TextDraft] = []
    duplicate_count = 0
    for draft in sorted(
        drafts,
        key=lambda item: (-item.quality, item.bbox[1], item.bbox[0], item.region.text),
    ):
        duplicate = False
        for previous in accepted:
            intersection = int(np.logical_and(draft.mask, previous.mask).sum())
            if not intersection:
                continue
            smaller = min(int(draft.mask.sum()), int(previous.mask.sum()))
            if (
                _mask_iou(draft.mask, previous.mask) >= 0.55
                or intersection / max(1, smaller) >= 0.86
                or _box_iou(draft.bbox, previous.bbox) >= 0.78
            ):
                duplicate = True
                duplicate_count += 1
                break
        if not duplicate:
            accepted.append(draft)
    return sorted(accepted, key=lambda item: (item.bbox[1], item.bbox[0])), duplicate_count


def _candidate_mask_coverage(candidate: _Candidate, mask: np.ndarray) -> float:
    crop = mask[candidate.y : candidate.bottom, candidate.x : candidate.right]
    return float(crop[candidate.crop].sum()) / max(1, candidate.area)


def _split_visual_components(
    image_lab: np.ndarray,
    candidates: Sequence[MaskCandidate],
) -> list[_Candidate]:
    """Split raw SAM proposals into stable glyph-sized visual components."""

    height, width = image_lab.shape[:2]
    canvas_area = height * width
    area_floor = max(6, int(round(canvas_area * 0.00020)))
    result: list[_Candidate] = []
    for source in candidates:
        if float(source.score) < POSTER_GROUP_THRESHOLDS["score_floor"]:
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            source.mask.astype(np.uint8), 8
        )
        for component_id in range(1, count):
            x, y, component_width, component_height, area = map(
                int, stats[component_id]
            )
            if (
                area < area_floor
                or area / canvas_area >= 0.085
                or component_width < 3
                or component_height < 3
            ):
                continue
            crop = (
                labels[y : y + component_height, x : x + component_width]
                == component_id
            )
            pixels = image_lab[y : y + component_height, x : x + component_width][crop]
            result.append(
                _Candidate(
                    raw_id=int(source.candidate_id),
                    component_id=component_id,
                    x=x,
                    y=y,
                    width=component_width,
                    height=component_height,
                    area=area,
                    score=float(source.score),
                    crop=crop,
                    median_lab=np.median(pixels, axis=0),
                )
            )
    deduplicated, _ = _deduplicate(result)
    return deduplicated


def _visual_row_groups(candidates: Sequence[_Candidate]) -> list[list[_Candidate]]:
    """Merge same-baseline word groups, allowing a colour change within a line."""

    initial = _row_groups(candidates)
    if len(initial) < 2:
        return initial
    forest = _DisjointSet(len(initial))
    for first in range(len(initial)):
        first_box = _candidate_bbox(initial[first])
        first_height = float(np.median([item.height for item in initial[first]]))
        for second in range(first + 1, len(initial)):
            second_box = _candidate_bbox(initial[second])
            second_height = float(np.median([item.height for item in initial[second]]))
            height_ratio = min(first_height, second_height) / max(
                1.0, max(first_height, second_height)
            )
            if height_ratio < 0.58:
                continue
            overlap = max(
                0,
                min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]),
            )
            if overlap / max(1, min(first_box[3] - first_box[1], second_box[3] - second_box[1])) < 0.52:
                continue
            baseline_delta = abs(first_box[3] - second_box[3])
            if baseline_delta > 0.28 * max(first_height, second_height):
                continue
            gap = max(
                0,
                max(first_box[0], second_box[0]) - min(first_box[2], second_box[2]),
            )
            if gap <= 1.65 * max(first_height, second_height):
                # Colour is intentionally ignored here: promotional lines often
                # switch colour between words or between a phrase and a number.
                forest.union(first, second)
    merged: list[list[_Candidate]] = []
    for indexes in forest.groups():
        members = [item for index in indexes for item in initial[index]]
        unique = {(item.raw_id, item.component_id): item for item in members}
        merged.append(sorted(unique.values(), key=lambda item: (item.x, item.y, item.key)))
    return merged


def _visual_plate_box(
    row_box: tuple[int, int, int, int],
    parent: LayerSpec,
    candidates: Sequence[MaskCandidate],
) -> tuple[int, int, int, int] | None:
    """Return a compact flat label behind a row, never the owner panel itself."""

    row_area = _box_area(row_box)
    row_height = max(1, row_box[3] - row_box[1])
    parent_width = max(1, parent.bbox[2] - parent.bbox[0])
    possible: list[tuple[int, tuple[int, int, int, int]]] = []
    for candidate in candidates:
        box = candidate.bbox
        box_width, box_height = box[2] - box[0], box[3] - box[1]
        if (
            candidate.fill_ratio < 0.78
            or box_width / max(1, box_height) < 2.8
            or _box_intersection(row_box, box) / max(1, row_area) < 0.88
            or box_height / row_height > 1.90
            or box_width / parent_width > 0.94
            or candidate.area < row_area * 0.42
        ):
            continue
        possible.append((_box_area(box), box))
    return min(possible, key=lambda item: (item[0], item[1]), default=(0, None))[1]


def _complete_visual_row_seed(
    image_lab: np.ndarray,
    seed: np.ndarray,
    row: Sequence[_Candidate],
    parent: LayerSpec,
    candidates: Sequence[MaskCandidate],
    background_lab: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Recover missed glyphs/accents inside a tight baseline-checked locality."""

    height, width = seed.shape
    row_box = _candidate_bbox(row)
    median_height = float(np.median([item.height for item in row]))
    default_padding = max(3, int(round(min(height, width) * 0.004)))
    plate = _visual_plate_box(row_box, parent, candidates)
    locality_box = plate or _clip_expand_box(
        row_box, default_padding, width, height
    )
    px0, py0, px1, py1 = parent.bbox
    locality_box = (
        max(px0, locality_box[0]),
        max(py0, locality_box[1]),
        min(px1, locality_box[2]),
        min(py1, locality_box[3]),
    )
    x0, y0, x1, y1 = locality_box
    locality = np.zeros_like(seed)
    locality[y0:y1, x0:x1] = True

    # A selected raw SAM mask may carry a detached accent alongside its main
    # connected component. Retain only the part inside this row locality.
    source_by_id = {int(item.candidate_id): item for item in candidates}
    completed = seed.copy()
    for source_id in {item.raw_id for item in row}:
        source = source_by_id.get(int(source_id))
        if source is not None:
            completed |= source.mask & locality
    completed &= locality & parent.mask

    roi_lab = image_lab[y0:y1, x0:x1]
    if not roi_lab.size:
        return completed, locality_box
    distance = np.linalg.norm(roi_lab - background_lab, axis=2)
    contrast = distance >= 18.0
    contrast = cv2.morphologyEx(
        contrast.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)
    local_seed = completed[y0:y1, x0:x1]
    support_radius = max(2, int(round(median_height * 0.18)))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (support_radius * 2 + 1,) * 2
    )
    seed_support = cv2.dilate(local_seed.astype(np.uint8), support_kernel).astype(bool)
    palette = np.asarray([item.median_lab for item in row], dtype=np.float32)
    baseline = float(np.median([item.bottom for item in row]))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        contrast.astype(np.uint8), 8
    )
    accepted = local_seed.copy()
    roi_area = max(1, (x1 - x0) * (y1 - y0))
    for component_id in range(1, count):
        cx, cy, component_width, component_height, area = map(
            int, stats[component_id]
        )
        if area < 3 or area > roi_area * 0.32:
            continue
        component = labels == component_id
        component_pixels = roi_lab[component]
        component_lab = np.median(component_pixels, axis=0)
        if float(np.min(np.linalg.norm(palette - component_lab, axis=1))) > 42.0:
            continue
        touches_seed = bool(np.logical_and(component, seed_support).any())
        global_bottom = y0 + cy + component_height
        baseline_aligned = (
            0.10 * median_height <= component_height <= 1.45 * median_height
            and abs(global_bottom - baseline) <= 0.34 * median_height
        )
        if not (touches_seed or baseline_aligned):
            continue
        touches_border = (
            cx == 0
            or cy == 0
            or cx + component_width == x1 - x0
            or cy + component_height == y1 - y0
        )
        if touches_border and (
            component_width >= 0.72 * (x1 - x0)
            or component_height >= 0.85 * (y1 - y0)
        ):
            # This is the frame/plate or surrounding panel, not a glyph.
            continue
        if touches_border and area > roi_area * 0.12 and not touches_seed:
            continue
        accepted |= component
    completed[y0:y1, x0:x1] = accepted
    return completed & locality & parent.mask, locality_box


def _visual_text_drafts(
    image_rgb: np.ndarray,
    candidates: Sequence[MaskCandidate],
    layers: Sequence[LayerSpec],
) -> tuple[list[_VisualTextDraft], dict[str, int], int]:
    """Find line-shaped SAM groups missed by OCR, strictly inside owner panels."""

    height, width = image_rgb.shape[:2]
    shape = (height, width)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    components = _split_visual_components(image_lab, candidates)
    owners = [
        layer for layer in layers if layer.metadata.get("grouping_role") == "owner_panel"
    ]
    existing_text = [
        layer.mask
        for layer in layers
        if layer.category == "text_raster"
        or layer.metadata.get("text_role") in {"ocr_text_line", "visual_text_row"}
    ]
    protected_layers = [
        layer
        for layer in layers
        if layer.metadata.get("grouping_role")
        in {"promoted_child_object", "standalone_object", "symmetric_decoration"}
    ]
    protected_union = np.zeros(shape, dtype=bool)
    for layer in protected_layers:
        protected_union |= layer.mask

    assigned: dict[str, list[_Candidate]] = {owner.layer_id: [] for owner in owners}
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for component in components:
        if any(_candidate_mask_coverage(component, mask) >= 0.36 for mask in existing_text):
            continue
        if any(
            _candidate_mask_coverage(component, layer.mask) >= 0.48
            for layer in protected_layers
        ):
            continue
        possible: list[tuple[int, LayerSpec]] = []
        for owner in owners:
            coverage = _candidate_mask_coverage(component, owner.mask)
            if coverage >= 0.68:
                possible.append((owner.area, owner))
        if possible:
            assigned[min(possible, key=lambda item: (item[0], item[1].layer_id))[1].layer_id].append(
                component
            )

    drafts: list[_VisualTextDraft] = []
    by_id = {layer.layer_id: layer for layer in owners}
    for parent_id in sorted(assigned):
        parent = by_id[parent_id]
        parent_box = parent.bbox
        parent_width = max(1, parent_box[2] - parent_box[0])
        parent_height = max(1, parent_box[3] - parent_box[1])
        for row in _visual_row_groups(assigned[parent_id]):
            if len(row) < 3:
                reject("too_few_components")
                continue
            x0, y0, x1, y1 = _candidate_bbox(row)
            row_width, row_height = x1 - x0, y1 - y0
            aspect = row_width / max(1, row_height)
            median_height = float(np.median([item.height for item in row]))
            height_ratio = min(item.height for item in row) / max(
                1, max(item.height for item in row)
            )
            maximum_parent_height_fraction = (
                0.78 if parent_width / max(1, parent_height) >= 5.0 else 0.40
            )
            if (
                aspect < 1.65
                or row_width / parent_width < 0.11
                or row_height / parent_height > maximum_parent_height_fraction
                or median_height < max(4.0, parent_height * 0.035)
                or height_ratio < 0.52
            ):
                reject("implausible_row_geometry")
                continue
            baselines = np.asarray([item.bottom for item in row], dtype=np.float32)
            baseline_spread = float(np.std(baselines) / max(1.0, median_height))
            if baseline_spread > 0.28:
                reject("unstable_baseline")
                continue
            horizontal_coverage = sum(item.width for item in row) / max(1, row_width)
            if not 0.28 <= horizontal_coverage <= 1.35:
                reject("implausible_horizontal_coverage")
                continue

            seed = _candidate_union(row, shape)
            occupancy = int(seed[y0:y1, x0:x1].sum()) / max(1, row_width * row_height)
            if not 0.10 <= occupancy <= 0.82:
                reject("implausible_visual_occupancy")
                continue
            contrast_radius = max(2, int(round(median_height * 0.16)))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (contrast_radius * 2 + 1,) * 2
            )
            ring = cv2.dilate(seed.astype(np.uint8), kernel).astype(bool) & ~seed & parent.mask
            if int(ring.sum()) < 16:
                reject("missing_local_background")
                continue
            background_lab = np.median(image_lab[ring], axis=0)
            contrast = float(
                np.median(np.linalg.norm(image_lab[seed] - background_lab, axis=1))
            )
            if contrast < 16.0:
                reject("low_local_contrast")
                continue

            completed_seed, locality_box = _complete_visual_row_seed(
                image_lab,
                seed,
                row,
                parent,
                candidates,
                background_lab,
            )
            halo_radius = max(1, int(round(min(shape) * 0.0035)))
            mask = _guided_halo(
                completed_seed,
                image_lab,
                radius=halo_radius,
                difference=12.0,
            )
            locality = np.zeros(shape, dtype=bool)
            lx0, ly0, lx1, ly1 = locality_box
            locality[ly0:ly1, lx0:lx1] = True
            mask &= locality & parent.mask & ~protected_union
            if not mask.any():
                reject("empty_visual_mask")
                continue
            if any(
                int(np.logical_and(mask, text_mask).sum()) / max(1, int(mask.sum())) >= 0.30
                for text_mask in existing_text
            ):
                reject("overlaps_existing_text")
                continue
            if any(
                _mask_iou(mask, previous.mask) >= 0.48
                or _box_iou(_mask_bbox(mask), previous.bbox) >= 0.76
                for previous in drafts
            ):
                reject("duplicate_visual_row")
                continue

            component_strength = min(1.0, len(row) / 12.0)
            width_strength = min(1.0, row_width / max(1.0, parent_width * 0.68))
            baseline_strength = max(0.0, 1.0 - baseline_spread / 0.28)
            contrast_strength = min(1.0, contrast / 55.0)
            mean_score = float(np.mean([item.score for item in row]))
            quality = (
                mean_score * 0.28
                + component_strength * 0.20
                + width_strength * 0.18
                + baseline_strength * 0.17
                + contrast_strength * 0.17
            )
            drafts.append(
                _VisualTextDraft(
                    parent_id=parent_id,
                    bbox=_mask_bbox(mask),
                    mask=mask,
                    components=row,
                    quality=float(quality),
                    occupancy=float(occupancy),
                    contrast=float(contrast),
                    baseline_spread=float(baseline_spread),
                )
            )
    drafts.sort(key=lambda item: (-item.quality, item.bbox[1], item.bbox[0]))
    return drafts, rejected, len(components)


def _panel_parent_for_text(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    layers: Sequence[LayerSpec],
) -> LayerSpec | None:
    text_area = int(mask.sum())
    box_area = _box_area(box)
    possible: list[LayerSpec] = []
    for layer in layers:
        if layer.metadata.get("grouping_role") != "owner_panel":
            continue
        mask_inside = int(np.logical_and(mask, layer.mask).sum()) / max(1, text_area)
        box_inside = _box_intersection(box, layer.bbox) / max(1, box_area)
        if mask_inside >= 0.55 or box_inside >= 0.88:
            possible.append(layer)
    return min(possible, key=lambda item: item.area, default=None)


def _existing_text_match(mask: np.ndarray, layers: Sequence[LayerSpec]) -> LayerSpec | None:
    area = int(mask.sum())
    possible: list[tuple[float, LayerSpec]] = []
    for layer in layers:
        role = layer.metadata.get("grouping_role")
        if role not in {"standalone_row", "ocr_text_line"} and layer.category != "text_raster":
            continue
        intersection = int(np.logical_and(mask, layer.mask).sum())
        if not intersection:
            continue
        containment = intersection / max(1, min(area, layer.area))
        overlap = _mask_iou(mask, layer.mask)
        if overlap >= 0.30 or containment >= 0.62:
            possible.append((max(overlap, containment), layer))
    return max(possible, key=lambda item: item[0], default=(0.0, None))[1]


def _semantic_hints(
    box: tuple[int, int, int, int], detections: Sequence[Detection]
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for detection in detections:
        intersection = _box_intersection(box, detection.bbox)
        if not intersection:
            continue
        if intersection / max(1, _box_area(box)) < 0.15 and _box_iou(box, detection.bbox) < 0.08:
            continue
        result.append(
            {
                "label": detection.label,
                "score": round(float(detection.score), 6),
                "bbox": list(detection.bbox),
            }
        )
    return sorted(result, key=lambda item: (-float(item["score"]), str(item["label"])))[:4]


def _repair_layer_hierarchy(layers: list[LayerSpec]) -> None:
    by_id = {layer.layer_id: layer for layer in layers}
    for layer in layers:
        parent_id = layer.metadata.get("parent_id")
        if parent_id is not None and str(parent_id) not in by_id:
            layer.metadata["parent_id"] = None
        layer.metadata["children"] = []
    for layer in layers:
        parent_id = layer.metadata.get("parent_id")
        if parent_id is not None:
            by_id[str(parent_id)].metadata["children"].append(layer.layer_id)
    for layer in layers:
        depth = 0
        parent_id = layer.metadata.get("parent_id")
        seen: set[str] = set()
        while parent_id is not None and str(parent_id) in by_id and str(parent_id) not in seen:
            seen.add(str(parent_id))
            depth += 1
            parent_id = by_id[str(parent_id)].metadata.get("parent_id")
        layer.metadata["hierarchy_depth"] = depth


def _cap_existing_layers(
    layers: list[LayerSpec], max_layers: int
) -> tuple[list[LayerSpec], list[str]]:
    if len(layers) <= max_layers:
        return layers, []
    mandatory = [layer for layer in layers if not bool(layer.metadata.get("optional", False))]
    optional = sorted(
        (layer for layer in layers if bool(layer.metadata.get("optional", False))),
        key=lambda item: (
            -float(item.metadata.get("salience", item.area / max(1, item.mask.size))),
            int(item.metadata.get("display_order", 0)),
            item.layer_id,
        ),
    )
    selected = mandatory[:max_layers]
    if len(selected) < max_layers:
        selected.extend(optional[: max_layers - len(selected)])
    selected_ids = {layer.layer_id for layer in selected}
    # A selected child without its parent is less useful than its intact parent.
    selected = [
        layer
        for layer in selected
        if layer.metadata.get("parent_id") is None
        or str(layer.metadata.get("parent_id")) in selected_ids
    ]
    dropped = [layer.layer_id for layer in layers if layer.layer_id not in {x.layer_id for x in selected}]
    _repair_layer_hierarchy(selected)
    return selected, dropped


def add_ocr_text_layers(
    image_rgb: np.ndarray,
    poster_layers: Sequence[LayerSpec],
    candidates: Sequence[MaskCandidate],
    text_regions: Sequence[TextRegion],
    max_layers: int,
    *,
    detections: Sequence[Detection] = (),
) -> tuple[list[LayerSpec], dict]:
    """Add one movable raster layer per supported OCR line to poster hierarchy.

    Existing geometric row layers are upgraded in place instead of duplicated.
    New lines become children of the smallest containing poster panel.  Parent
    masks are unioned with each text mask so the existing V5 parent-cleanup pass
    can reconstruct the panel below text when the line moves or is hidden.

    Grounding-DINO detections are recorded only as metadata hints; they never
    rename panels or change masks.
    """

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_rgb must be a uint8 RGB array with shape (H, W, 3).")
    if isinstance(max_layers, bool) or not isinstance(max_layers, int) or max_layers < 1:
        raise ValueError("max_layers must be a positive integer.")
    shape = image.shape[:2]
    copied = [_copy_layer(layer) for layer in poster_layers]
    if len({layer.layer_id for layer in copied}) != len(copied):
        raise ValueError("poster_layers must have unique layer_id values.")
    for layer in copied:
        if layer.mask.shape != shape:
            raise ValueError(f"Layer {layer.layer_id!r} does not match image shape {shape}.")
    candidate_list = list(candidates)
    for candidate in candidate_list:
        if candidate.mask.shape != shape:
            raise ValueError(
                f"MaskCandidate {candidate.candidate_id} does not match image shape {shape}."
            )

    layers, initially_dropped = _cap_existing_layers(copied, max_layers)
    protected_union = np.zeros(shape, dtype=bool)
    for layer in layers:
        if layer.metadata.get("grouping_role") in {
            "promoted_child_object",
            "standalone_object",
            "symmetric_decoration",
        }:
            protected_union |= layer.mask
    rejection_counts: dict[str, int] = {}
    raw_drafts: list[_TextDraft] = []
    for region in sorted(text_regions, key=lambda item: (item.bbox[1], item.bbox[0], item.text)):
        draft, rejection = _text_draft(image, candidate_list, region)
        if draft is None:
            reason = str(rejection or "unknown")
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        else:
            protected_rejection = _subtract_protected_from_text_draft(
                draft, protected_union
            )
            if protected_rejection is None:
                raw_drafts.append(draft)
            else:
                rejection_counts[protected_rejection] = (
                    rejection_counts.get(protected_rejection, 0) + 1
                )
    drafts, duplicate_regions = _deduplicate_text_drafts(raw_drafts)

    matched_existing = 0
    pending: list[_TextDraft] = []
    annotated_ids: set[str] = set()
    for draft in drafts:
        existing = _existing_text_match(draft.mask, layers)
        if existing is None or existing.layer_id in annotated_ids:
            pending.append(draft)
            continue
        geometric_row = existing.metadata.get("grouping_role") == "standalone_row"
        if geometric_row:
            # OCR geometry is often much wider than stylized display text. The
            # geometric SAM row is already a stronger mask, so OCR annotates it
            # without pulling a neighbouring panel/icon into the layer.
            existing.mask &= ~protected_union
            existing.metadata["ocr_mask_policy"] = "existing_geometry_annotation_only"
            existing.metadata["ocr_source_ids"] = sorted(
                {item.candidate_id for item in draft.candidates}
            )
        else:
            existing.mask |= draft.mask
            existing.source_ids = sorted(
                set(existing.source_ids) | {item.candidate_id for item in draft.candidates}
            )
        existing.category = "text_raster"
        existing.score = max(existing.score, max(item.score for item in draft.candidates))
        existing.text = draft.region.text
        existing.label = "text_line"
        existing.metadata.update(
            {
                "text_role": "ocr_text_line",
                "ocr_bbox": list(draft.region.bbox),
                "ocr_expanded_bbox": list(draft.bbox),
                "ocr_confidence": round(float(draft.region.confidence), 6),
                "ocr_raw_text": draft.region.text,
                "ocr_match_score": round(draft.match_score, 6),
                "ocr_quality": round(draft.quality, 6),
                "ocr_mask_occupancy": round(draft.occupancy, 6),
                "ocr_seed_support": round(draft.seed_support, 6),
                "ocr_horizontal_span": round(draft.horizontal_span, 6),
                "ocr_protected_removed_fraction": round(
                    draft.protected_removed_fraction, 6
                ),
                "semantic_hints": _semantic_hints(draft.bbox, detections),
            }
        )
        parent = _panel_parent_for_text(existing.mask, draft.bbox, layers)
        if parent is not None and parent is not existing:
            existing.metadata["parent_id"] = parent.layer_id
        annotated_ids.add(existing.layer_id)
        matched_existing += 1

    # Text has higher edit value than optional decorations/partial child masks.
    # When necessary, absorb the weakest optional leaf back into its parent.
    capacity = max_layers - len(layers)
    required_slots = max(0, len(pending) - capacity)
    evictable = sorted(
        (
            layer
            for layer in layers
            if bool(layer.metadata.get("optional", False))
            and not layer.metadata.get("children")
            and layer.category != "text_raster"
        ),
        key=lambda item: (
            float(item.metadata.get("salience", item.area / max(1, item.mask.size))),
            -int(item.metadata.get("display_order", 0)),
            item.layer_id,
        ),
    )
    evicted: list[str] = []
    for layer in evictable[:required_slots]:
        layers.remove(layer)
        evicted.append(layer.layer_id)
    _repair_layer_hierarchy(layers)
    capacity = max(0, max_layers - len(layers))
    selected_pending = sorted(
        pending,
        key=lambda item: (-item.quality, item.bbox[1], item.bbox[0], item.region.text),
    )[:capacity]
    selected_pending.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.region.text))
    dropped_by_cap = len(pending) - len(selected_pending)

    used_ids = {layer.layer_id for layer in layers}
    next_text_number = 1
    next_display_order = max(
        (int(layer.metadata.get("display_order", 0)) for layer in layers), default=0
    ) + 1
    for draft in selected_pending:
        while f"poster_text_{next_text_number:02d}" in used_ids:
            next_text_number += 1
        layer_id = f"poster_text_{next_text_number:02d}"
        next_text_number += 1
        parent = _panel_parent_for_text(draft.mask, draft.bbox, layers)
        parent_id = parent.layer_id if parent is not None else None
        hint = _position_hint(draft.mask)
        layer = LayerSpec(
            layer_id=layer_id,
            name=f"TEXT LINE - {hint.replace('_', ' ').upper()}",
            category="text_raster",
            mask=draft.mask.astype(bool, copy=True),
            score=max(item.score for item in draft.candidates),
            source_ids=sorted({item.candidate_id for item in draft.candidates}),
            label="text_line",
            text=draft.region.text,
            metadata={
                "grouping_role": "ocr_text_line",
                "text_role": "ocr_text_line",
                "layout_hint": hint,
                "parent_id": parent_id,
                "children": [],
                "hierarchy_depth": 1 if parent_id else 0,
                "display_order": next_display_order,
                "optional": False,
                "ocr_bbox": list(draft.region.bbox),
                "ocr_expanded_bbox": list(draft.bbox),
                "ocr_confidence": round(float(draft.region.confidence), 6),
                "ocr_raw_text": draft.region.text,
                "ocr_match_score": round(draft.match_score, 6),
                "ocr_quality": round(draft.quality, 6),
                "ocr_mask_occupancy": round(draft.occupancy, 6),
                "ocr_seed_support": round(draft.seed_support, 6),
                "ocr_horizontal_span": round(draft.horizontal_span, 6),
                "ocr_protected_removed_fraction": round(
                    draft.protected_removed_fraction, 6
                ),
                "semantic_hints": _semantic_hints(draft.bbox, detections),
            },
        )
        layers.append(layer)
        used_ids.add(layer_id)
        next_display_order += 1

    _repair_layer_hierarchy(layers)
    visual_drafts, visual_rejections, visual_component_count = _visual_text_drafts(
        image, candidate_list, layers
    )

    # Recover capacity first from a redundant optional leaf. If one more slot
    # is needed for a strong text row, a symmetric decoration may fall back to
    # the background; unique icons/objects are never sacrificed here.
    visual_capacity = max(0, max_layers - len(layers))
    visual_required_slots = max(0, len(visual_drafts) - visual_capacity)
    existing_text_union = np.zeros(shape, dtype=bool)
    for layer in layers:
        if layer.category == "text_raster":
            existing_text_union |= layer.mask
    optional_visual_eviction: list[tuple[int, float, LayerSpec, str]] = []
    strongest_visual_quality = max(
        (draft.quality for draft in visual_drafts), default=0.0
    )
    for layer in layers:
        if (
            not bool(layer.metadata.get("optional", False))
            or layer.metadata.get("children")
            or layer.category == "text_raster"
        ):
            continue
        coverage = int(np.logical_and(layer.mask, existing_text_union).sum()) / max(
            1, layer.area
        )
        role = str(layer.metadata.get("grouping_role"))
        if coverage >= 0.82:
            optional_visual_eviction.append((0, -coverage, layer, "redundant_with_text"))
        elif role == "symmetric_decoration" and strongest_visual_quality >= 0.80:
            optional_visual_eviction.append((1, -coverage, layer, "text_priority_over_decoration"))
    optional_visual_eviction.sort(
        key=lambda item: (item[0], item[1], item[2].layer_id)
    )
    visual_evicted: list[str] = []
    visual_eviction_reasons: dict[str, str] = {}
    for _, _, layer, reason in optional_visual_eviction[:visual_required_slots]:
        layers.remove(layer)
        visual_evicted.append(layer.layer_id)
        visual_eviction_reasons[layer.layer_id] = reason
    _repair_layer_hierarchy(layers)

    visual_capacity = max(0, max_layers - len(layers))
    selected_visual = visual_drafts[:visual_capacity]
    visual_dropped_by_cap = len(visual_drafts) - len(selected_visual)
    selected_visual.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.parent_id))
    for draft in selected_visual:
        while f"poster_text_{next_text_number:02d}" in used_ids:
            next_text_number += 1
        layer_id = f"poster_text_{next_text_number:02d}"
        next_text_number += 1
        hint = _position_hint(draft.mask)
        layer = LayerSpec(
            layer_id=layer_id,
            name=f"TEXT LINE - {hint.replace('_', ' ').upper()}",
            category="text_raster",
            mask=draft.mask.astype(bool, copy=True),
            score=max(item.score for item in draft.components),
            source_ids=sorted({item.raw_id for item in draft.components}),
            label="text_line",
            text=None,
            metadata={
                "grouping_role": "visual_text_row",
                "text_role": "visual_text_row",
                "layout_hint": hint,
                "parent_id": draft.parent_id,
                "children": [],
                "hierarchy_depth": 1,
                "display_order": next_display_order,
                "optional": False,
                "visual_bbox": list(draft.bbox),
                "visual_component_count": len(draft.components),
                "visual_quality": round(draft.quality, 6),
                "visual_mask_occupancy": round(draft.occupancy, 6),
                "visual_lab_contrast": round(draft.contrast, 6),
                "visual_baseline_spread": round(draft.baseline_spread, 6),
                "semantic_hints": _semantic_hints(draft.bbox, detections),
            },
        )
        layers.append(layer)
        used_ids.add(layer_id)
        next_display_order += 1

    _repair_layer_hierarchy(layers)
    text_layers = [
        layer
        for layer in layers
        if layer.category == "text_raster" or layer.metadata.get("text_role") == "ocr_text_line"
    ]
    text_layers.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.layer_id))
    for index, layer in enumerate(text_layers, 1):
        hint = str(layer.metadata.get("layout_hint", _position_hint(layer.mask)))
        layer.name = f"TEXT LINE {index:02d} - {hint.replace('_', ' ').upper()}"
        parent_id = layer.metadata.get("parent_id")
        if parent_id is not None:
            parent = next(
                (item for item in layers if item.layer_id == str(parent_id)),
                None,
            )
            if parent is not None:
                parent.mask |= layer.mask
    _repair_layer_hierarchy(layers)

    report: dict[str, object] = {
        "policy": "poster_ocr_visual_line_hierarchy_v2",
        "input_poster_layer_count": len(poster_layers),
        "input_text_region_count": len(text_regions),
        "input_candidate_count": len(candidate_list),
        "supported_text_proposal_count": len(raw_drafts),
        "duplicate_text_region_count": duplicate_regions,
        "rejected_text_regions": rejection_counts,
        "matched_existing_row_count": matched_existing,
        "created_text_layer_count": len(selected_pending),
        "visual_component_count": visual_component_count,
        "visual_text_proposal_count": len(visual_drafts),
        "rejected_visual_text_rows": visual_rejections,
        "created_visual_text_layer_count": len(selected_visual),
        "visual_text_proposals_dropped_by_cap": visual_dropped_by_cap,
        "optional_layers_evicted_for_visual_text": visual_evicted,
        "visual_text_eviction_reasons": visual_eviction_reasons,
        "selected_text_layer_count": len(text_layers),
        "parented_text_layer_count": sum(
            layer.metadata.get("parent_id") is not None for layer in text_layers
        ),
        "optional_layers_evicted_for_text": evicted,
        "initial_layers_dropped_by_cap": initially_dropped,
        "text_proposals_dropped_by_cap": dropped_by_cap,
        "selected_layer_count": len(layers),
        "max_layers": max_layers,
        "hierarchy_edges": sum(layer.metadata.get("parent_id") is not None for layer in layers),
        "policy_note": (
            "One supported OCR line becomes one raster layer; an overlapping geometric row is upgraded "
            "instead of duplicated. SAM components missed by OCR may form one visual row only when their "
            "baseline, spacing, occupancy, panel containment and local contrast are stable; protected icons "
            "and existing text are excluded. The smallest containing panel becomes parent and unions the "
            "text mask; DINO labels are metadata hints only."
        ),
    }
    return layers, report


__all__ = [
    "POSTER_GROUP_THRESHOLDS",
    "add_ocr_text_layers",
    "group_poster_layers",
]
