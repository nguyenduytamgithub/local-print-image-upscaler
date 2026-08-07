"""Recover clean semantic mattes from the exact SAM components V5 selected.

Poster grouping records component provenance such as ``sam_0025_cc_001`` in
``LayerSpec.metadata["source_components"]``.  The grouping stage may later add
colour-guided halo pixels to the layer mask.  Those pixels are useful as a
proposal, but they are not reliable semantic ownership: a nearby border, word,
or decoration can be swept into the same editable layer.

This module provides a conservative, non-mutating cleanup pass for object-like
layers.  When exact component provenance is available, the recorded connected
components become the semantic matte.  No dilation, closing, colour expansion,
or renderer reconstruction footprint is admitted here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import AbstractSet, Any

import cv2
import numpy as np

from .model import LayerSpec


SOURCE_COMPONENT_PATTERN = re.compile(r"^sam_(\d+)_cc_(\d+)$")

# These roles are assembled directly from SAM connected components by the
# poster grouping path.  Panels and text have different ownership semantics and
# must not be silently changed by this conservative object cleanup pass.
DEFAULT_CLEANABLE_ROLES = frozenset(
    {
        "promoted_child_object",
        "standalone_object",
        "symmetric_decoration",
    }
)

TEXT_ROLES = frozenset({"standalone_row", "ocr_text_line", "visual_text_row"})


def _copy_layer(layer: LayerSpec, mask: np.ndarray | None = None) -> LayerSpec:
    return LayerSpec(
        layer_id=layer.layer_id,
        name=layer.name,
        category=layer.category,
        mask=(layer.mask if mask is None else mask).astype(bool, copy=True),
        score=float(layer.score),
        source_ids=list(layer.source_ids),
        label=layer.label,
        text=layer.text,
        metadata=dict(layer.metadata),
        alpha_matte=(
            None
            if mask is not None or layer.alpha_matte is None
            else np.asarray(layer.alpha_matte, dtype=np.float32).copy()
        ),
    )


def _normalise_raw_masks(
    raw_sam_masks: Sequence[np.ndarray] | np.ndarray,
) -> list[np.ndarray]:
    if isinstance(raw_sam_masks, np.ndarray):
        if raw_sam_masks.ndim == 2:
            values = [raw_sam_masks]
        elif raw_sam_masks.ndim == 3:
            values = [raw_sam_masks[index] for index in range(raw_sam_masks.shape[0])]
        else:
            raise ValueError("raw_sam_masks must be a 2-D mask or an (N,H,W) array.")
    else:
        values = list(raw_sam_masks)

    masks: list[np.ndarray] = []
    for index, value in enumerate(values):
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError(f"raw_sam_masks[{index}] must be two-dimensional.")
        masks.append(array.astype(bool, copy=False))
    return masks


def _component_count(mask: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
    return max(0, int(count) - 1)


class _ComponentResolver:
    """Resolve OpenCV component ids exactly as V5 poster grouping created them."""

    def __init__(self, raw_masks: list[np.ndarray], shape: tuple[int, int]) -> None:
        self.raw_masks = raw_masks
        self.shape = shape
        self._labels: dict[int, tuple[int, np.ndarray]] = {}
        for index, mask in enumerate(raw_masks):
            if mask.shape != shape:
                raise ValueError(
                    f"raw_sam_masks[{index}] has shape {mask.shape}; expected {shape}."
                )

    def resolve(self, reference: str) -> np.ndarray | None:
        match = SOURCE_COMPONENT_PATTERN.fullmatch(reference)
        if match is None:
            return None
        raw_id, component_id = (int(value) for value in match.groups())
        if raw_id < 0 or raw_id >= len(self.raw_masks) or component_id < 1:
            return None
        if raw_id not in self._labels:
            count, labels = cv2.connectedComponents(
                self.raw_masks[raw_id].astype(np.uint8), 8
            )
            self._labels[raw_id] = (int(count), labels)
        count, labels = self._labels[raw_id]
        if component_id >= count:
            return None
        return labels == component_id


def _component_references(layer: LayerSpec) -> list[str]:
    value: Any = layer.metadata.get("source_components", ())
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = []
    # Preserve deterministic source order while ignoring duplicate metadata.
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _clean_with_resolver(
    layer: LayerSpec,
    resolver: _ComponentResolver,
    cleanable_roles: AbstractSet[str],
) -> tuple[LayerSpec, dict[str, object]]:
    original = np.asarray(layer.mask)
    if original.ndim != 2:
        raise ValueError(f"Layer {layer.layer_id!r} mask must be two-dimensional.")
    if original.shape != resolver.shape:
        raise ValueError(
            f"Layer {layer.layer_id!r} mask has shape {original.shape}; "
            f"expected {resolver.shape}."
        )
    original = original.astype(bool, copy=False)
    role = str(layer.metadata.get("grouping_role", ""))
    references = _component_references(layer)
    report: dict[str, object] = {
        "layer_id": layer.layer_id,
        "role": role,
        "policy": "exact_recorded_sam_components_v1",
        "original_area": int(original.sum()),
        "original_component_count": _component_count(original),
        "requested_source_components": references,
        "resolved_source_components": [],
        "unresolved_source_components": [],
    }

    if role not in cleanable_roles:
        report.update(
            {
                "status": "skipped_role",
                "cleaned_area": int(original.sum()),
                "cleaned_component_count": _component_count(original),
                "removed_unanchored_area": 0,
                "restored_source_area": 0,
                "outside_semantic_source_area": 0,
                "changed": False,
            }
        )
        return _copy_layer(layer), report

    semantic_source = np.zeros(resolver.shape, dtype=bool)
    resolved: list[str] = []
    unresolved: list[str] = []
    for reference in references:
        component = resolver.resolve(reference)
        if component is None:
            unresolved.append(reference)
            continue
        semantic_source |= component
        resolved.append(reference)
    report["resolved_source_components"] = resolved
    report["unresolved_source_components"] = unresolved

    if not resolved or not semantic_source.any():
        # Destroying a layer on incomplete provenance would be worse than
        # leaving it for a later/manual refinement pass.
        report.update(
            {
                "status": "skipped_no_resolved_source_component",
                "cleaned_area": int(original.sum()),
                "cleaned_component_count": _component_count(original),
                "removed_unanchored_area": 0,
                "restored_source_area": 0,
                "outside_semantic_source_area": None,
                "changed": False,
            }
        )
        return _copy_layer(layer), report

    # Exact provenance is the anchor.  Replacing the expanded proposal with the
    # recorded components both restores any source pixels lost downstream and
    # removes every island that has no recorded semantic anchor.
    cleaned = semantic_source
    removed = original & ~semantic_source
    restored = semantic_source & ~original
    outside = cleaned & ~semantic_source
    if outside.any():  # Defensive invariant; should be impossible by construction.
        raise RuntimeError(f"Semantic cleanup expanded layer {layer.layer_id!r} outside its source.")

    cleaned_layer = _copy_layer(layer, cleaned)
    report.update(
        {
            "status": "cleaned_from_recorded_components",
            "semantic_source_area": int(semantic_source.sum()),
            "cleaned_area": int(cleaned.sum()),
            "cleaned_component_count": _component_count(cleaned),
            "removed_unanchored_area": int(removed.sum()),
            "restored_source_area": int(restored.sum()),
            "outside_semantic_source_area": int(outside.sum()),
            "changed": not np.array_equal(cleaned, original),
        }
    )
    return cleaned_layer, report


def clean_layer_semantic_mask(
    layer: LayerSpec,
    raw_sam_masks: Sequence[np.ndarray] | np.ndarray,
    *,
    cleanable_roles: AbstractSet[str] = DEFAULT_CLEANABLE_ROLES,
) -> tuple[LayerSpec, dict[str, object]]:
    """Return one cleaned copy of ``layer`` plus a JSON-serialisable report.

    The input ``LayerSpec`` and raw SAM arrays are never mutated.  If a target
    layer has no valid recorded component reference, it is returned unchanged
    and the report records the conservative skip.
    """

    shape = tuple(int(value) for value in np.asarray(layer.mask).shape)
    if len(shape) != 2:
        raise ValueError(f"Layer {layer.layer_id!r} mask must be two-dimensional.")
    resolver = _ComponentResolver(_normalise_raw_masks(raw_sam_masks), shape)
    return _clean_with_resolver(layer, resolver, frozenset(cleanable_roles))


def clean_semantic_layer_masks(
    layers: Sequence[LayerSpec],
    raw_sam_masks: Sequence[np.ndarray] | np.ndarray,
    *,
    cleanable_roles: AbstractSet[str] = DEFAULT_CLEANABLE_ROLES,
) -> tuple[list[LayerSpec], dict[str, object]]:
    """Clean a layer collection with one cached SAM component-label pass."""

    layer_list = list(layers)
    if not layer_list:
        return [], {
            "policy": "exact_recorded_sam_components_v1",
            "input_layer_count": 0,
            "cleaned_layer_count": 0,
            "changed_layer_count": 0,
            "skipped_layer_count": 0,
            "removed_unanchored_area": 0,
            "layers": [],
        }
    first_mask = np.asarray(layer_list[0].mask)
    if first_mask.ndim != 2:
        raise ValueError(f"Layer {layer_list[0].layer_id!r} mask must be two-dimensional.")
    resolver = _ComponentResolver(
        _normalise_raw_masks(raw_sam_masks), tuple(int(value) for value in first_mask.shape)
    )
    roles = frozenset(cleanable_roles)
    cleaned_layers: list[LayerSpec] = []
    reports: list[dict[str, object]] = []
    for layer in layer_list:
        cleaned, report = _clean_with_resolver(layer, resolver, roles)
        cleaned_layers.append(cleaned)
        reports.append(report)

    cleaned_count = sum(
        report["status"] == "cleaned_from_recorded_components" for report in reports
    )
    changed_count = sum(bool(report["changed"]) for report in reports)
    return cleaned_layers, {
        "policy": "exact_recorded_sam_components_v1",
        "input_layer_count": len(layer_list),
        "cleaned_layer_count": cleaned_count,
        "changed_layer_count": changed_count,
        "skipped_layer_count": len(layer_list) - cleaned_count,
        "removed_unanchored_area": sum(
            int(report["removed_unanchored_area"]) for report in reports
        ),
        "layers": reports,
    }


def _metadata_box(layer: LayerSpec) -> tuple[int, int, int, int]:
    for key in ("ocr_expanded_bbox", "ocr_bbox", "visual_bbox"):
        value = layer.metadata.get(key)
        if isinstance(value, Sequence) and len(value) == 4:
            try:
                return tuple(int(item) for item in value)  # type: ignore[return-value]
            except (TypeError, ValueError):
                pass
    return layer.bbox


def _bbox_gap(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> tuple[int, int]:
    horizontal = max(0, max(first[0], second[0]) - min(first[2], second[2]))
    vertical = max(0, max(first[1], second[1]) - min(first[3], second[3]))
    return horizontal, vertical


def _counter_count(mask: np.ndarray) -> int:
    """Count meaningful enclosed background contours in one binary matte."""

    contours, hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return 0
    return sum(
        int(hierarchy[0, index, 3]) >= 0 and cv2.contourArea(contour) >= 2.0
        for index, contour in enumerate(contours)
    )


def _refine_text_negative_space(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    role: str = "",
) -> tuple[np.ndarray, dict[str, object]]:
    """Remove panel-coloured counters/gaps that a unioned SAM seed filled.

    The most background-distant pixels are treated as immutable ink cores.
    Background-coloured proposal pixels are subtracted even inside a connected
    component, while compact source-ink components aligned with a glyph may be
    recovered when SAM missed part of a Vietnamese accent.  This recovers O/0
    counters and G/C apertures without accepting long panel rules.
    """

    rgb = np.asarray(image_rgb)
    original = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be a uint8 RGB array with shape (H,W,3).")
    if rgb.shape[:2] != original.shape:
        raise ValueError(
            f"image_rgb shape {rgb.shape[:2]} does not match text mask {original.shape}."
        )
    x0, y0, x1, y1 = box
    local_mask = original[y0:y1, x0:x1]
    local_rgb = rgb[y0:y1, x0:x1]
    if not local_mask.any() or local_rgb.size == 0:
        return original.copy(), {"status": "skipped_empty_locality"}

    local_height, local_width = local_mask.shape
    pure_text_role = role in {"ocr_text_line", "visual_text_row"}
    ring_radius = max(
        2,
        min(12, int(round(min(local_height, max(8, local_width // 4)) * 0.08))),
    )
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_radius * 2 + 1,) * 2
    )
    ring = cv2.dilate(local_mask.astype(np.uint8), ring_kernel).astype(bool) & ~local_mask
    if int(ring.sum()) < 32:
        ring = ~local_mask
    if int(ring.sum()) < 8:
        return original.copy(), {"status": "skipped_missing_background_samples"}

    lab = cv2.cvtColor(local_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    samples = lab[ring]
    first_median = np.median(samples, axis=0)
    sample_distance = np.linalg.norm(samples - first_median, axis=1)
    core_samples = samples[sample_distance <= np.percentile(sample_distance, 70)]
    background_lab = np.median(core_samples if len(core_samples) else samples, axis=0)
    colour_distance = np.linalg.norm(lab - background_lab, axis=2)
    ring_distance = colour_distance[ring]
    background_threshold = max(13.0, float(np.percentile(ring_distance, 78)) + 4.0)
    foreground_peak = float(np.percentile(colour_distance[local_mask], 92))
    strong_threshold = max(background_threshold, foreground_peak * 0.55)
    strong = local_mask & (colour_distance >= strong_threshold)
    strong_area = int(strong.sum())
    original_area = int(local_mask.sum())
    if strong_area < max(3, int(round(original_area * 0.04))):
        return original.copy(), {
            "status": "skipped_weak_ink_core",
            "background_threshold_lab": round(background_threshold, 4),
            "strong_threshold_lab": round(strong_threshold, 4),
        }

    distance_to_ink = cv2.distanceTransform(
        (~strong).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    occupied_rows = np.flatnonzero(local_mask.any(axis=1))
    glyph_height = int(occupied_rows[-1] - occupied_rows[0] + 1)
    outline_radius = max(1.25, min(2.25, glyph_height * 0.035))
    # Distance alone can preserve a pure panel-coloured bridge when it lies
    # close to both sides of a glyph (the classic closed-G failure).  Require
    # independent pixel evidence too.  The cutoff follows ordinary ring
    # variation but is capped well below the immutable ink threshold, so real
    # antialiased/outlined pixels remain candidates while O counters, G
    # apertures and N negative space matching the panel are subtracted.
    background_cutoff = min(
        strong_threshold * 0.30,
        max(4.0, float(np.percentile(ring_distance, 65)) + 2.0),
    )
    colour_supported = local_mask & (colour_distance >= background_cutoff)
    cleaned_local = colour_supported & (distance_to_ink <= outline_radius)

    # SAM/OCR can omit most of a detached Vietnamese mark even when its source
    # pixels are unambiguous. Recover whole compact source-colour components,
    # not a globally dilated contrast mask. Components already overlapping a
    # glyph core are safe anchors; a fully missed component must align closely
    # above/below such an anchor. Long panel rules are explicitly ineligible.
    source_ink = colour_distance >= strong_threshold
    seed_count, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(
        source_ink.astype(np.uint8), 8
    )
    seed_records: list[
        tuple[int, int, tuple[int, int, int, int], int, bool, np.ndarray]
    ] = []
    anchor_boxes: list[tuple[int, int, int, int]] = []
    anchor_colours: list[np.ndarray] = []
    anchor_ids: set[int] = set()
    for component_id in range(1, seed_count):
        sx, sy, sw, sh, area = map(int, seed_stats[component_id])
        seed_box = (sx, sy, sx + sw, sy + sh)
        overlap = int(np.logical_and(seed_labels == component_id, local_mask).sum())
        seed_component = seed_labels == component_id
        component_colour = np.median(lab[seed_component], axis=0)
        frame_like = (
            sh <= max(4, int(round(glyph_height * 0.10)))
            and (
                sw >= 0.20 * local_width
                or (
                    sw >= max(16, int(round(local_width * 0.04)))
                    and sw / max(1, sh) >= 8.0
                )
            )
        ) or (
            sw <= 3
            and sh >= 0.72 * local_height
            and (sx <= 1 or sx + sw >= local_width - 1)
        )
        seed_records.append(
            (component_id, area, seed_box, overlap, frame_like, component_colour)
        )
        body_geometry = (
            sh >= max(5, int(round(glyph_height * 0.45)))
            and area >= max(8, int(round(original_area * 0.0008)))
            if pure_text_role
            else (
                sh >= max(4, int(round(glyph_height * 0.16)))
                or area >= max(12, int(round(original_area * 0.0015)))
            )
        )
        if (
            not frame_like
            and overlap >= max(3, int(round(area * 0.05)))
            and body_geometry
        ):
            anchor_boxes.append(seed_box)
            anchor_colours.append(component_colour)
            anchor_ids.add(component_id)

    protected_ids: list[int] = []
    recovered_accent_ids: list[int] = []
    maximum_accent_area = max(320, int(round(local_width * local_height * 0.025)))
    ink_similarity_limit = max(28.0, strong_threshold * 0.58)
    for component_id, area, seed_box, overlap, frame_like, component_colour in seed_records:
        if frame_like:
            continue
        if component_id in anchor_ids:
            protected_ids.append(component_id)
            continue
        secondary_area_limit = (
            max(1024, int(round(local_width * local_height * 0.05)))
            if overlap > 0
            else maximum_accent_area
        )
        if area < 2 or area > secondary_area_limit or not anchor_boxes:
            continue
        component_width = seed_box[2] - seed_box[0]
        component_height = seed_box[3] - seed_box[1]
        if component_width >= 16 and component_width / max(1, component_height) >= 8.0:
            continue
        if pure_text_role and component_width > 0.68 * glyph_height:
            continue
        nearest_anchor_colour = min(
            float(np.linalg.norm(component_colour - anchor_colour))
            for anchor_colour in anchor_colours
        )
        if nearest_anchor_colour > ink_similarity_limit:
            continue
        component_center_x = (seed_box[0] + seed_box[2]) / 2.0
        aligned = False
        for anchor in anchor_boxes:
            horizontal_gap, vertical_gap = _bbox_gap(seed_box, anchor)
            center_aligned = (
                anchor[0] - 0.16 * glyph_height
                <= component_center_x
                <= anchor[2] + 0.16 * glyph_height
            )
            if (
                center_aligned
                and horizontal_gap <= 0.12 * glyph_height
                and vertical_gap <= max(4.0, 0.28 * glyph_height)
            ):
                aligned = True
                break
        if aligned:
            protected_ids.append(component_id)
            recovered_accent_ids.append(component_id)

    protected_ink = np.isin(seed_labels, protected_ids) if protected_ids else np.zeros_like(local_mask)
    protected_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    protected_support = cv2.dilate(
        protected_ink.astype(np.uint8), protected_kernel
    ).astype(bool) & (colour_distance >= max(2.0, background_cutoff * 0.65))
    cleaned_local |= protected_support

    # A tiny Vietnamese accent can be a separate component whose antialiasing
    # never reaches the global 92nd-percentile ink threshold. Restore only a
    # fully removed component whose own median remains strongly non-background.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        local_mask.astype(np.uint8), 8
    )
    restored_component_count = 0
    restored_protected = np.zeros_like(local_mask)
    for component_id in range(1, count):
        component = labels == component_id
        if np.logical_and(component, cleaned_local).any():
            continue
        component_area = int(stats[component_id, cv2.CC_STAT_AREA])
        component_median = float(np.median(colour_distance[component]))
        restore_allowed = (
            component_area >= 2 and component_median >= strong_threshold * 0.78
        )
        if restore_allowed and pure_text_role:
            cx, cy, cw, ch, _ = map(int, stats[component_id])
            component_box = (cx, cy, cx + cw, cy + ch)
            component_center_x = (component_box[0] + component_box[2]) / 2.0
            component_colour = np.median(lab[component], axis=0)
            colour_aligned = bool(anchor_colours) and min(
                float(np.linalg.norm(component_colour - anchor_colour))
                for anchor_colour in anchor_colours
            ) <= ink_similarity_limit
            geometry_aligned = any(
                (
                    anchor[0] - 0.16 * glyph_height
                    <= component_center_x
                    <= anchor[2] + 0.16 * glyph_height
                )
                and _bbox_gap(component_box, anchor)[0] <= 0.12 * glyph_height
                and _bbox_gap(component_box, anchor)[1]
                <= max(4.0, 0.28 * glyph_height)
                for anchor in anchor_boxes
            )
            compact = (
                (cw < 16 or cw / max(1, ch) < 8.0)
                and cw <= 0.68 * glyph_height
            )
            restore_allowed = colour_aligned and geometry_aligned and compact
        if restore_allowed:
            cleaned_local |= component
            restored_protected |= component
            restored_component_count += 1

    # Colour thresholds alone cannot reject every counter/aperture pixel on a
    # textured or shaded panel.  A dark-red background variation, for example,
    # can be far from the median red plate yet still be semantically outside a
    # yellow O/G.  Each accepted source-ink component supplies a glyph-local
    # convex envelope.  Anything already owned inside that envelope but not
    # within a one-pixel cross-neighbourhood of real source ink is negative
    # space, unless it is itself protected source ink or a recovered accent.
    # The deliberately narrow 3x3 ellipse catches diagonal one-pixel bridges
    # without eroding the ordinary axial antialias band around a stroke.
    convex_negative_component_count = 0
    convex_negative_space_pixels = 0
    if pure_text_role and anchor_ids:
        ink_neighbourhood_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for component_id in sorted(anchor_ids):
            anchor = seed_labels == component_id
            anchor_y, anchor_x = np.where(anchor)
            if len(anchor_x) < 3:
                continue
            points = np.column_stack((anchor_x, anchor_y)).astype(np.int32)
            hull = cv2.convexHull(points)
            if hull is None or len(hull) < 3:
                continue
            hull_mask = np.zeros_like(cleaned_local, dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 1)
            ink_neighbourhood = cv2.dilate(
                anchor.astype(np.uint8), ink_neighbourhood_kernel
            ).astype(bool)
            negative_core = hull_mask.astype(bool) & ~ink_neighbourhood
            removable = (
                cleaned_local
                & negative_core
                & ~protected_ink
                & ~restored_protected
            )
            removed_area = int(removable.sum())
            if not removed_area:
                continue
            cleaned_local[removable] = False
            convex_negative_component_count += 1
            convex_negative_space_pixels += removed_area

    removed_nontext_component_count = 0
    removed_nontext_component_pixels = 0
    accepted_support = protected_support | restored_protected
    if pure_text_role and accepted_support.any():
        cleaned_count, cleaned_labels = cv2.connectedComponents(
            cleaned_local.astype(np.uint8), 8
        )
        filtered = np.zeros_like(cleaned_local)
        for component_id in range(1, cleaned_count):
            component = cleaned_labels == component_id
            if np.logical_and(component, accepted_support).any():
                filtered |= component
            else:
                removed_nontext_component_count += 1
                removed_nontext_component_pixels += int(component.sum())
        cleaned_local = filtered

    # A same-colour object can physically touch the text and therefore survive
    # connected-component filtering. Visual poster rows expose a stable common
    # baseline across several glyphs: split only the strip below that robust
    # baseline, remove wide lobes/underlines, and keep narrow real descenders
    # such as the tail of Q.
    baseline_removed_component_count = 0
    baseline_removed_pixels = 0
    baseline_source_y: int | None = None
    baseline_rejected = np.zeros_like(cleaned_local)
    if role == "visual_text_row" and len(anchor_boxes) >= 4:
        anchor_bottoms = np.array([anchor[3] for anchor in anchor_boxes], dtype=np.float32)
        baseline_source_y = int(round(float(np.median(anchor_bottoms))))
        if 0 < baseline_source_y < local_height:
            below = np.zeros_like(cleaned_local)
            below[baseline_source_y:] = cleaned_local[baseline_source_y:]
            below_count, below_labels, below_stats, _ = cv2.connectedComponentsWithStats(
                below.astype(np.uint8), 8
            )
            wide_lobe_limit = max(18, int(round(glyph_height * 0.45)))
            wide_rule_limit = max(24, int(round(glyph_height * 0.90)))
            deep_descender_height = max(7, int(round(glyph_height * 0.13)))
            descender_width_limit = max(18, int(round(glyph_height * 0.62)))
            for component_id in range(1, below_count):
                component_width = int(below_stats[component_id, cv2.CC_STAT_WIDTH])
                component_height = int(below_stats[component_id, cv2.CC_STAT_HEIGHT])
                component_area = int(below_stats[component_id, cv2.CC_STAT_AREA])
                wide_lobe = (
                    component_width >= wide_lobe_limit and component_height >= 4
                )
                wide_rule = component_width >= wide_rule_limit
                true_descender = (
                    component_height >= deep_descender_height
                    and component_width <= descender_width_limit
                )
                if (
                    component_area < 24
                    or not (wide_lobe or wide_rule)
                    or true_descender
                ):
                    continue
                component = below_labels == component_id
                cleaned_local[component] = False
                baseline_rejected |= component
                baseline_removed_component_count += 1
                baseline_removed_pixels += component_area

    cleaned_area = int(cleaned_local.sum())
    retained_ratio = cleaned_area / max(1, original_area)
    minimum_retained_ratio = 0.62 if protected_ink.any() else 0.75
    if not cleaned_local.any() or retained_ratio < minimum_retained_ratio:
        return original.copy(), {
            "status": "fallback_excessive_removal",
            "retained_ratio": round(retained_ratio, 6),
            "minimum_retained_ratio": minimum_retained_ratio,
            "background_threshold_lab": round(background_threshold, 4),
            "strong_threshold_lab": round(strong_threshold, 4),
        }
    protected_required = protected_ink & ~baseline_rejected
    protected_missing = int(np.logical_and(protected_required, ~cleaned_local).sum())
    if protected_missing:
        raise RuntimeError(
            f"Text negative-space cleanup lost {protected_missing} protected source-ink pixels."
        )

    result = np.zeros_like(original)
    result[y0:y1, x0:x1] = cleaned_local
    removed = local_mask & ~cleaned_local
    recovered = cleaned_local & ~local_mask
    return result, {
        "status": "refined_colour_topology",
        "background_lab": [round(float(value), 3) for value in background_lab],
        "background_threshold_lab": round(background_threshold, 4),
        "strong_threshold_lab": round(strong_threshold, 4),
        "background_rejection_cutoff_lab": round(background_cutoff, 4),
        "outline_radius_source_px": round(float(outline_radius), 4),
        "ink_core_pixels": int(protected_required.sum()),
        "ink_core_recall": 1.0,
        "removed_negative_space_pixels": int(removed.sum()),
        "removed_fraction": round(float(removed.sum()) / max(1, original_area), 6),
        "retained_ratio": round(retained_ratio, 6),
        "minimum_retained_ratio": minimum_retained_ratio,
        "restored_small_ink_components": restored_component_count,
        "protected_source_ink_component_count": len(protected_ids),
        "recovered_missing_accent_component_count": len(recovered_accent_ids),
        "recovered_source_ink_pixels": int(recovered.sum()),
        "protected_source_ink_recall": 1.0,
        "convex_negative_component_count": convex_negative_component_count,
        "convex_negative_space_pixels": convex_negative_space_pixels,
        "source_ink_component_similarity_limit_lab": round(
            float(ink_similarity_limit), 4
        ),
        "removed_nontext_component_count": removed_nontext_component_count,
        "removed_nontext_component_pixels": removed_nontext_component_pixels,
        "baseline_source_y": baseline_source_y,
        "baseline_removed_component_count": baseline_removed_component_count,
        "baseline_removed_pixels": baseline_removed_pixels,
        "counter_count_before": _counter_count(local_mask),
        "counter_count_after": _counter_count(cleaned_local),
    }


def clean_text_layer_mask(
    layer: LayerSpec,
    image_rgb: np.ndarray | None = None,
) -> tuple[LayerSpec, dict[str, object]]:
    """Remove panel rules and unanchored specks from one raster text matte.

    Text is intentionally allowed to contain many connected components—every
    glyph and Vietnamese diacritic may be separate.  The filter therefore
    rejects only strongly frame-like components and tiny islands that have no
    spatial relationship to a glyph-sized component.
    """

    original = np.asarray(layer.mask, dtype=bool)
    role = str(layer.metadata.get("grouping_role", ""))
    if layer.category != "text_raster" and role not in TEXT_ROLES:
        return _copy_layer(layer), {
            "layer_id": layer.layer_id,
            "status": "skipped_non_text",
            "changed": False,
        }
    height, width = original.shape
    raw_box = _metadata_box(layer)
    x0 = max(0, min(width, raw_box[0]))
    y0 = max(0, min(height, raw_box[1]))
    x1 = max(x0, min(width, raw_box[2]))
    y1 = max(y0, min(height, raw_box[3]))
    locality = np.zeros_like(original)
    locality[y0:y1, x0:x1] = True
    restricted = original & locality
    local_width = max(1, x1 - x0)
    local_height = max(1, y1 - y0)
    local_area = local_width * local_height

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        restricted.astype(np.uint8), 8
    )
    records: list[tuple[int, int, tuple[int, int, int, int], bool]] = []
    frame_ids: set[int] = set()
    for component_id in range(1, count):
        x, y, component_width, component_height, area = map(
            int, stats[component_id]
        )
        box = (x, y, x + component_width, y + component_height)
        horizontal_rule = (
            component_height <= max(4, int(round(0.12 * local_height)))
            and (
                component_width >= 0.20 * local_width
                or component_width / max(1, component_height) >= 8.0
            )
        )
        # Never reject a narrow interior component merely for being tall: I,
        # l and the numeral 1 are legitimate glyphs.  Only a hairline attached
        # to a locality side can be treated as a vertical panel rule.
        vertical_rule = (
            component_width <= 3
            and component_height >= 0.78 * local_height
            and (x <= x0 + 1 or x + component_width >= x1 - 1)
        )
        frame_like = horizontal_rule or vertical_rule
        if frame_like:
            frame_ids.add(component_id)
        records.append((component_id, area, box, frame_like))

    # Glyph bodies are tall or carry meaningful area.  Separate accents and
    # punctuation survive when they sit near one of these anchors.
    anchors = [
        box
        for _component_id, area, box, frame_like in records
        if not frame_like
        and (
            box[3] - box[1] >= max(3, int(round(0.18 * local_height)))
            or area >= max(6, int(round(0.0018 * local_area)))
        )
    ]
    tiny_limit = max(3, int(round(0.00012 * local_area)))
    kept = np.zeros_like(original)
    removed_unanchored_ids: list[int] = []
    for component_id, area, box, frame_like in records:
        if frame_like:
            continue
        near_anchor = any(
            (lambda gap: gap[0] <= 0.18 * local_height and gap[1] <= 0.42 * local_height)(
                _bbox_gap(box, anchor)
            )
            for anchor in anchors
            if anchor != box
        )
        if area < tiny_limit and anchors and not near_anchor:
            removed_unanchored_ids.append(component_id)
            continue
        kept |= labels == component_id

    # Never publish an empty text layer because one unusual font resembled a
    # frame; retaining the strict locality is safer than deleting user content.
    status = "cleaned_text_components"
    if not kept.any() and restricted.any():
        kept = restricted
        frame_ids.clear()
        removed_unanchored_ids.clear()
        status = "fallback_locality_only"
    pixel_report: dict[str, object] = {"status": "not_requested"}
    if image_rgb is not None and kept.any():
        kept, pixel_report = _refine_text_negative_space(
            image_rgb,
            kept,
            (x0, y0, x1, y1),
            role=role,
        )

    report = {
        "layer_id": layer.layer_id,
        "role": role,
        "policy": "text_component_ownership_v1",
        "status": status,
        "locality_bbox": [x0, y0, x1, y1],
        "original_area": int(original.sum()),
        "cleaned_area": int(kept.sum()),
        "original_component_count": max(0, count - 1),
        "cleaned_component_count": _component_count(kept),
        "removed_frame_component_count": len(frame_ids),
        "removed_unanchored_component_count": len(removed_unanchored_ids),
        "removed_area": int((original & ~kept).sum()),
        "negative_space_refinement": pixel_report,
        "changed": not np.array_equal(original, kept),
    }
    return _copy_layer(layer, kept), report


def clean_text_layer_masks(
    layers: Sequence[LayerSpec],
    image_rgb: np.ndarray | None = None,
) -> tuple[list[LayerSpec], dict[str, object]]:
    """Apply conservative glyph-aware cleanup to every text-like layer."""

    cleaned: list[LayerSpec] = []
    reports: list[dict[str, object]] = []
    for layer in layers:
        result, report = clean_text_layer_mask(layer, image_rgb=image_rgb)
        cleaned.append(result)
        if report["status"] != "skipped_non_text":
            reports.append(report)
    return cleaned, {
        "policy": "text_component_ownership_v1",
        "text_layer_count": len(reports),
        "changed_layer_count": sum(bool(item["changed"]) for item in reports),
        "removed_frame_component_count": sum(
            int(item.get("removed_frame_component_count", 0)) for item in reports
        ),
        "removed_unanchored_component_count": sum(
            int(item.get("removed_unanchored_component_count", 0)) for item in reports
        ),
        "removed_area": sum(int(item.get("removed_area", 0)) for item in reports),
        "negative_space_refined_layer_count": sum(
            item.get("negative_space_refinement", {}).get("status")
            == "refined_colour_topology"
            for item in reports
        ),
        "layers": reports,
    }


__all__ = [
    "DEFAULT_CLEANABLE_ROLES",
    "SOURCE_COMPONENT_PATTERN",
    "clean_layer_semantic_mask",
    "clean_semantic_layer_masks",
    "clean_text_layer_mask",
    "clean_text_layer_masks",
]
