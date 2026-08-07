"""Pinned ViTMatte boundary refinement for clean V5 text layers.

Semantic/topology cleanup happens before this module.  ViTMatte receives only a
narrow trimap around that trusted mask and is clipped back inside it, so the
network can estimate fractional edge coverage but cannot fill an O counter,
close a G aperture, invent a remote island, or change wording.
"""

from __future__ import annotations

import gc
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from PIL import Image

from .model import LayerSpec
from .segment import _local_model_snapshot, _verified_file_sha256


VITMATTE_MODEL = "hustvl/vitmatte-small-composition-1k"
VITMATTE_REVISION = "53222614392e8bd24ed804fbd2f9a43c46ac3850"
VITMATTE_WEIGHT_SHA256 = (
    "bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb"
)
REFINED_TEXT_ROLES = frozenset({"standalone_row", "ocr_text_line", "visual_text_row"})


def _copy_with_alpha(
    layer: LayerSpec,
    alpha_matte: np.ndarray,
    semantic_mask: np.ndarray,
) -> LayerSpec:
    return LayerSpec(
        layer_id=layer.layer_id,
        name=layer.name,
        category=layer.category,
        mask=np.asarray(semantic_mask, dtype=bool).copy(),
        score=float(layer.score),
        source_ids=list(layer.source_ids),
        label=layer.label,
        text=layer.text,
        metadata=dict(layer.metadata),
        alpha_matte=np.asarray(alpha_matte, dtype=np.float32).copy(),
    )


def _component_counter_count(mask: np.ndarray) -> int:
    contours, hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return 0
    return sum(
        int(hierarchy[0, index, 3]) >= 0 and cv2.contourArea(contour) >= 2.0
        for index, contour in enumerate(contours)
    )


def make_trimap(mask: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a 0/128/255 trimap and the immutable sure-foreground seed."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional.")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError("radius must be a positive integer.")
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.uint8), np.zeros(binary.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    sure_foreground = cv2.erode(binary.astype(np.uint8), kernel).astype(bool)
    possible_foreground = cv2.dilate(binary.astype(np.uint8), kernel).astype(bool)

    # Keep at least one certain pixel for every thin accent/component.  The
    # maximum of its distance transform is its safest interior location.
    count, labels = cv2.connectedComponents(binary.astype(np.uint8), 8)
    for component_id in range(1, count):
        component = labels == component_id
        if np.logical_and(component, sure_foreground).any():
            continue
        distance = cv2.distanceTransform(
            component.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        maximum = float(distance.max())
        if maximum > 0:
            sure_foreground |= component & (distance >= maximum - 1e-6)

    trimap = np.zeros(binary.shape, dtype=np.uint8)
    trimap[possible_foreground] = 128
    trimap[sure_foreground] = 255
    return trimap, sure_foreground


def _prune_unanchored_alpha(
    alpha: np.ndarray,
    sure_foreground: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Drop predicted alpha islands that contain no trusted foreground seed."""

    matte = np.asarray(alpha, dtype=np.float32)
    anchors = np.asarray(sure_foreground, dtype=bool)
    if matte.ndim != 2 or anchors.ndim != 2:
        raise ValueError("alpha and sure_foreground must be two-dimensional.")
    if matte.shape != anchors.shape:
        raise ValueError("alpha and sure_foreground must have the same shape.")
    if not np.isfinite(matte).all():
        raise ValueError("alpha must contain only finite values.")
    owned = matte > 0
    if np.logical_and(anchors, ~owned).any():
        raise ValueError("sure_foreground must have positive alpha.")
    count, labels = cv2.connectedComponents(owned.astype(np.uint8), 8)
    keep = np.zeros_like(owned)
    removed_components = 0
    removed_pixels = 0
    for component_id in range(1, count):
        component = labels == component_id
        if np.logical_and(component, anchors).any():
            keep |= component
            continue
        removed_components += 1
        removed_pixels += int(component.sum())
    result = matte.copy()
    result[~keep] = 0.0
    return result, removed_components, removed_pixels


def _stabilize_source_colour_alpha(
    image_rgb: np.ndarray,
    clean_mask: np.ndarray,
    alpha: np.ndarray,
    sure_foreground: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Keep source-supported ink and reject tiny foreign threshold islands.

    ViTMatte is deliberately subordinate to the deterministic clean mask, but
    a very thin Vietnamese accent may still lose its outer source pixels while
    a one-pixel panel fragment survives because ``make_trimap`` had to seed
    every component.  Large glyph bodies provide an adaptive Lab palette.
    Clean-mask pixels matching that palette receive a 0.5 alpha floor; small
    threshold components with neither body geometry nor palette evidence are
    removed.  The rule uses only the current source and glyph geometry, never
    wording, fixed colours or poster-specific coordinates.
    """

    rgb = np.asarray(image_rgb)
    binary = np.asarray(clean_mask, dtype=bool)
    matte = np.asarray(alpha, dtype=np.float32)
    anchors = np.asarray(sure_foreground, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be a uint8 RGB array with shape (H,W,3).")
    if binary.ndim != 2 or matte.ndim != 2 or anchors.ndim != 2:
        raise ValueError("clean_mask, alpha and sure_foreground must be two-dimensional.")
    if rgb.shape[:2] != binary.shape or matte.shape != binary.shape or anchors.shape != binary.shape:
        raise ValueError("image_rgb, clean_mask, alpha and sure_foreground must share a shape.")
    if not np.isfinite(matte).all() or (matte < 0).any() or (matte > 1).any():
        raise ValueError("alpha must contain finite values from 0 through 1.")
    if np.logical_and(anchors, ~binary).any():
        raise ValueError("sure_foreground cannot escape the clean mask.")
    if not binary.any():
        return matte.copy(), {
            "status": "skipped_empty_mask",
            "protected_source_colour_pixels": 0,
            "alpha_floor_raised_pixels": 0,
            "removed_foreign_alpha_component_count": 0,
            "removed_foreign_alpha_pixels": 0,
        }

    occupied_rows = np.flatnonzero(binary.any(axis=1))
    glyph_height = int(occupied_rows[-1] - occupied_rows[0] + 1)
    threshold_mask = matte >= 0.5
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        threshold_mask.astype(np.uint8), 8
    )
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ring_radius = max(2, min(8, int(round(glyph_height * 0.08))))
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_radius * 2 + 1,) * 2
    )
    background_ring = cv2.dilate(
        binary.astype(np.uint8), ring_kernel
    ).astype(bool) & ~binary
    if int(background_ring.sum()) < 24:
        background_ring = ~binary
    ring_samples = lab[background_ring]
    first_background = np.median(ring_samples, axis=0)
    first_distance = np.linalg.norm(ring_samples - first_background, axis=1)
    background_core = ring_samples[
        first_distance <= np.percentile(first_distance, 70)
    ]
    background_lab = np.median(
        background_core if len(background_core) else ring_samples, axis=0
    )
    background_distance = np.linalg.norm(lab - background_lab, axis=2)
    ring_distance = background_distance[background_ring]
    background_rejection_cutoff = max(
        8.0, float(np.percentile(ring_distance, 70)) + 2.0
    )
    body_palette_separation = max(
        18.0, float(np.percentile(ring_distance, 85)) + 6.0
    )
    srgb = rgb.astype(np.float32) / 255.0
    linear_rgb = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    background_linear = np.median(linear_rgb[background_ring], axis=0)
    ring_linear_noise = np.linalg.norm(
        linear_rgb[background_ring] - background_linear, axis=1
    )
    compositing_residual_limit = float(
        np.clip(float(np.percentile(ring_linear_noise, 85)) + 0.025, 0.045, 0.10)
    )
    body_records: list[
        tuple[
            tuple[int, int, int, int],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
        ]
    ] = []
    minimum_body_area = max(24, int(round(0.10 * glyph_height * glyph_height)))
    minimum_body_height = max(5, int(round(0.42 * glyph_height)))
    minimum_body_width = max(3, int(round(0.08 * glyph_height)))
    for component_id in range(1, count):
        x, y, width, height, area = map(int, stats[component_id])
        if (
            height < minimum_body_height
            or width < minimum_body_width
            or area < minimum_body_area
        ):
            continue
        component = labels == component_id
        samples = component & anchors
        if int(samples.sum()) < 3:
            samples = component & (matte >= 0.75)
        if int(samples.sum()) < 3:
            continue
        palette = np.median(lab[samples], axis=0)
        if float(np.linalg.norm(palette - background_lab)) < body_palette_separation:
            continue
        foreground_linear = np.median(linear_rgb[samples], axis=0)
        foreground_rgb = np.clip(
            np.median(rgb[samples], axis=0) + 0.5, 0, 255
        ).astype(np.uint8)
        foreground_hsv = cv2.cvtColor(
            foreground_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2HSV
        )[0, 0].astype(np.float32)
        spread = np.linalg.norm(lab[samples] - palette, axis=1)
        colour_limit = float(
            np.clip(float(np.percentile(spread, 90)) + 12.0, 26.0, 48.0)
        )
        body_records.append(
            (
                (x, y, x + width, y + height),
                palette,
                foreground_linear,
                foreground_hsv,
                colour_limit,
            )
        )

    initial_body_palette_count = len(body_records)
    if body_records:
        palette_contrasts = [
            float(np.linalg.norm(record[1] - background_lab))
            for record in body_records
        ]
        relative_contrast_floor = max(
            body_palette_separation, max(palette_contrasts) * 0.42
        )
        body_records = [
            record
            for record, contrast in zip(body_records, palette_contrasts)
            if contrast >= relative_contrast_floor
        ]
    else:
        relative_contrast_floor = body_palette_separation

    if not body_records:
        result = matte.copy()
        result[~binary] = 0.0
        return result, {
            "status": "skipped_no_glyph_body_palette",
            "protected_source_colour_pixels": 0,
            "alpha_floor_raised_pixels": 0,
            "removed_foreign_alpha_component_count": 0,
            "removed_foreign_alpha_pixels": 0,
        }

    palette_supported = np.zeros_like(binary)
    hue_supported = np.zeros_like(binary)
    analytic_supported = np.zeros_like(binary)
    analytic_alpha = np.zeros_like(matte)
    analytic_accent_component_count = 0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    for (
        body_box,
        palette,
        foreground_linear,
        foreground_hsv,
        colour_limit,
    ) in body_records:
        palette_supported |= np.linalg.norm(lab - palette, axis=2) <= colour_limit
        if float(foreground_hsv[1]) >= 72.0:
            hue_delta = np.abs(hsv[..., 0] - float(foreground_hsv[0]))
            hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
            hue_supported |= (
                (hue_delta <= 8.0)
                & (hsv[..., 1] >= 72.0)
                & (hsv[..., 2] >= 48.0)
            )
        direction = foreground_linear - background_linear
        denominator = float(np.dot(direction, direction))
        if denominator <= 1e-8:
            continue
        projection = np.sum(
            (linear_rgb - background_linear) * direction, axis=2
        ) / denominator
        projected = np.clip(projection, 0.0, 1.0)
        reconstructed = (
            background_linear[None, None, :]
            + projected[..., None] * direction[None, None, :]
        )
        residual = np.linalg.norm(linear_rgb - reconstructed, axis=2)
        valid_projection = (
            (projection >= 0.08)
            & (projection <= 1.18)
            & (residual <= compositing_residual_limit)
            & binary
        )
        projection_count, projection_labels, projection_stats, _ = (
            cv2.connectedComponentsWithStats(valid_projection.astype(np.uint8), 8)
        )
        for component_id in range(1, projection_count):
            x, y, width, height, area = map(
                int, projection_stats[component_id]
            )
            component_box = (x, y, x + width, y + height)
            horizontal_gap = max(
                0,
                max(component_box[0], body_box[0])
                - min(component_box[2], body_box[2]),
            )
            vertical_gap = max(
                0,
                max(component_box[1], body_box[1])
                - min(component_box[3], body_box[3]),
            )
            component_center_x = (component_box[0] + component_box[2]) / 2.0
            compact = (
                height <= max(5, int(round(0.38 * glyph_height)))
                and width <= max(8, int(round(0.72 * glyph_height)))
                and area <= max(512, int(round(binary.size * 0.03)))
            )
            aligned = (
                body_box[0] - 0.18 * glyph_height
                <= component_center_x
                <= body_box[2] + 0.18 * glyph_height
                and horizontal_gap <= 0.14 * glyph_height
                and vertical_gap <= max(4.0, 0.32 * glyph_height)
            )
            if not (compact and aligned):
                continue
            component = projection_labels == component_id
            analytic_supported |= component
            analytic_alpha[component] = np.maximum(
                analytic_alpha[component],
                np.clip(projection[component], 0.0, 1.0),
            )
            analytic_accent_component_count += 1
    palette_supported &= background_distance >= background_rejection_cutoff
    analytic_supported &= background_distance >= min(
        8.0, background_rejection_cutoff
    )
    hue_supported &= background_distance >= min(8.0, background_rejection_cutoff)
    protected = binary & (palette_supported | analytic_supported | hue_supported)
    result = matte.copy()
    below_floor = protected & (result < 0.5)
    source_floor = np.maximum(0.5, analytic_alpha)
    result[protected] = np.maximum(result[protected], source_floor[protected])

    direct_background_cutoff = min(8.0, background_rejection_cutoff)
    direct_background = binary & (
        background_distance < direct_background_cutoff
    )
    removed = direct_background.copy()
    result[direct_background] = 0.0
    removed_component_count = max(
        0,
        int(
            cv2.connectedComponents(
                direct_background.astype(np.uint8), 8
            )[0]
        )
        - 1,
    )
    maximum_foreign_area = max(32, int(round(binary.size * 0.003)))
    for threshold in (0.5, 0.25):
        active = result >= threshold
        active_count, active_labels, active_stats, _ = cv2.connectedComponentsWithStats(
            active.astype(np.uint8), 8
        )
        for component_id in range(1, active_count):
            x, y, width, height, area = map(int, active_stats[component_id])
            component = active_labels == component_id
            body_geometry = (
                height >= minimum_body_height
                and width >= minimum_body_width
                and area >= minimum_body_area
            )
            singleton_noise = area == 1 and glyph_height >= 20
            if (
                not singleton_noise
                and (
                    body_geometry
                    or area > maximum_foreign_area
                    or np.logical_and(component, protected).any()
                )
            ):
                continue
            result[component] = 0.0
            halo = cv2.dilate(
                component.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
            ).astype(bool)
            halo &= binary & ~protected & (result < threshold)
            result[halo] = 0.0
            removed |= component | halo
            removed_component_count += 1

    result[~binary] = 0.0
    return result, {
        "status": "stabilized_source_colour_alpha",
        "glyph_body_palette_count": len(body_records),
        "discarded_weak_body_palette_count": (
            initial_body_palette_count - len(body_records)
        ),
        "relative_body_palette_contrast_floor_lab": round(
            float(relative_contrast_floor), 4
        ),
        "background_lab": [round(float(value), 3) for value in background_lab],
        "background_rejection_cutoff_lab": round(
            float(background_rejection_cutoff), 4
        ),
        "direct_background_cutoff_lab": round(float(direct_background_cutoff), 4),
        "compositing_residual_limit_linear_rgb": round(
            compositing_residual_limit, 6
        ),
        "protected_source_colour_pixels": int(protected.sum()),
        "analytic_compositing_support_pixels": int(
            np.logical_and(binary, analytic_supported).sum()
        ),
        "analytic_accent_component_count": analytic_accent_component_count,
        "hue_supported_pixels": int(np.logical_and(binary, hue_supported).sum()),
        "alpha_floor": 0.5,
        "alpha_floor_raised_pixels": int(below_floor.sum()),
        "direct_background_pixels_removed": int(direct_background.sum()),
        "removed_foreign_alpha_component_count": removed_component_count,
        "removed_foreign_alpha_pixels": int(removed.sum()),
    }


def _is_text_layer(layer: LayerSpec) -> bool:
    role = str(layer.metadata.get("grouping_role", ""))
    return layer.category == "text_raster" or role in REFINED_TEXT_ROLES


def refine_text_alpha_mattes(
    image_rgb: np.ndarray,
    layers: Sequence[LayerSpec],
    *,
    progress: Callable[[str], None] = print,
) -> tuple[list[LayerSpec], dict[str, object]]:
    """Estimate source-resolution fractional alpha for each clean text mask."""

    rgb = np.asarray(image_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be a uint8 RGB array with shape (H,W,3).")
    layer_list = list(layers)
    targets = [layer for layer in layer_list if _is_text_layer(layer) and layer.mask.any()]
    if not targets:
        return layer_list, {
            "model": VITMATTE_MODEL,
            "revision": VITMATTE_REVISION,
            "status": "skipped_no_text_layers",
            "refined_layer_count": 0,
            "layers": [],
        }

    import torch
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    snapshot = Path(_local_model_snapshot(VITMATTE_MODEL, VITMATTE_REVISION))
    actual_weight_sha256 = _verified_file_sha256(
        str(snapshot / "model.safetensors"),
        VITMATTE_WEIGHT_SHA256,
        "ViTMatte-S",
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    progress(
        "  V5 bước 3c/4: ViTMatte đang làm mịn alpha trong biên chữ sạch "
        + ("trên GPU..." if device.type == "cuda" else "trên CPU...")
    )
    processor = VitMatteImageProcessor.from_pretrained(
        snapshot, local_files_only=True
    )
    model = VitMatteForImageMatting.from_pretrained(
        snapshot,
        local_files_only=True,
        use_safetensors=True,
    ).eval().to(device)

    refined_by_id: dict[str, LayerSpec] = {}
    records: list[dict[str, object]] = []
    try:
        for layer in targets:
            mask = np.asarray(layer.mask, dtype=bool)
            if mask.shape != rgb.shape[:2]:
                raise ValueError(
                    f"Layer {layer.layer_id!r} mask shape {mask.shape} does not match "
                    f"image {rgb.shape[:2]}."
                )
            x0, y0, x1, y1 = layer.bbox
            glyph_height = max(1, y1 - y0)
            radius = max(2, min(4, int(round(glyph_height * 0.035))))
            padding = max(8, radius * 4)
            left = max(0, x0 - padding)
            top = max(0, y0 - padding)
            right = min(rgb.shape[1], x1 + padding)
            bottom = min(rgb.shape[0], y1 + padding)
            local_rgb = rgb[top:bottom, left:right]
            local_mask = mask[top:bottom, left:right]
            trimap, sure_foreground = make_trimap(local_mask, radius)
            inputs = processor(
                images=Image.fromarray(local_rgb, "RGB"),
                trimaps=Image.fromarray(trimap, "L"),
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            autocast = (
                torch.autocast("cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                local_alpha = model(**inputs).alphas[0, 0].float().cpu().numpy()
            local_alpha = np.clip(
                local_alpha[: local_mask.shape[0], : local_mask.shape[1]], 0.0, 1.0
            )
            # The cleaned topology is authoritative. The model controls only
            # coverage inside it; it can neither grow the outer silhouette nor
            # refill counters/apertures which the colour pass removed.
            local_alpha[~local_mask] = 0.0
            local_alpha[sure_foreground] = 1.0
            local_alpha[local_alpha < 1.0 / 255.0] = 0.0
            role = str(layer.metadata.get("grouping_role", ""))
            if role in {"ocr_text_line", "visual_text_row"}:
                local_alpha, colour_guard_report = _stabilize_source_colour_alpha(
                    local_rgb,
                    local_mask,
                    local_alpha,
                    sure_foreground,
                )
            else:
                colour_guard_report = {
                    "status": "skipped_non_pure_text_role",
                    "protected_source_colour_pixels": 0,
                    "alpha_floor_raised_pixels": 0,
                    "removed_foreign_alpha_component_count": 0,
                    "removed_foreign_alpha_pixels": 0,
                }
            trusted_sure_foreground = sure_foreground & (local_alpha > 0)
            (
                local_alpha,
                removed_alpha_island_count,
                removed_alpha_island_pixels,
            ) = _prune_unanchored_alpha(local_alpha, trusted_sure_foreground)
            # Pruning can sever a sub-threshold bridge and expose a final
            # one-pixel panel island which was connected to a glyph during the
            # first colour pass.  Re-evaluate the now-stable alpha topology;
            # this pass is idempotent for real palette/hue-supported ink.
            if role in {"ocr_text_line", "visual_text_row"}:
                local_alpha, verification_guard_report = (
                    _stabilize_source_colour_alpha(
                        local_rgb,
                        local_mask,
                        local_alpha,
                        trusted_sure_foreground,
                    )
                )
                colour_guard_report["verification_pass"] = verification_guard_report
                trusted_sure_foreground = sure_foreground & (local_alpha > 0)
                (
                    local_alpha,
                    verification_island_count,
                    verification_island_pixels,
                ) = _prune_unanchored_alpha(
                    local_alpha, trusted_sure_foreground
                )
                removed_alpha_island_count += verification_island_count
                removed_alpha_island_pixels += verification_island_pixels
            # Do not remove one-pixel source components merely from their
            # geometry. They can be real punctuation or Vietnamese accents.
            # Renderer-created singletons are handled later with source colour,
            # local-background and lower-layer evidence, after an exact x1
            # preflight render exposes whether required_alpha promoted them.
            removed_quantized_singletons = 0
            trusted_sure_foreground = sure_foreground & (local_alpha > 0)
            rejected_sure_foreground_pixels = int(
                np.logical_and(sure_foreground, ~trusted_sure_foreground).sum()
            )
            alpha = np.zeros(mask.shape, dtype=np.float32)
            alpha[top:bottom, left:right] = local_alpha
            if np.logical_and(alpha > 0, ~mask).any():
                raise RuntimeError(
                    f"ViTMatte escaped clean text topology: {layer.layer_id}"
                )
            if np.logical_and(trusted_sure_foreground, local_alpha < 1.0).any():
                raise RuntimeError(
                    f"ViTMatte clipped an immutable text core: {layer.layer_id}"
                )
            semantic_mask = alpha > 0
            if not semantic_mask.any():
                raise RuntimeError(f"ViTMatte erased text layer: {layer.layer_id}")
            levels = np.unique(np.round(local_alpha * 255.0).astype(np.uint8))
            thresholds = {}
            for threshold in (0.25, 0.5, 0.75):
                threshold_mask = local_alpha >= threshold
                thresholds[str(threshold)] = {
                    "foreground_pixels": int(threshold_mask.sum()),
                    "component_count": max(
                        0,
                        int(
                            cv2.connectedComponents(
                                threshold_mask.astype(np.uint8), 8
                            )[0]
                        )
                        - 1,
                    ),
                    "counter_count": _component_counter_count(threshold_mask),
                }
            refined_by_id[layer.layer_id] = _copy_with_alpha(
                layer,
                alpha,
                semantic_mask,
            )
            records.append(
                {
                    "layer_id": layer.layer_id,
                    "radius_source_px": radius,
                    "crop_bbox": [left, top, right, bottom],
                    "alpha_level_count": int(len(levels)),
                    "soft_alpha_pixels": int(
                        np.logical_and(local_alpha > 0, local_alpha < 1).sum()
                    ),
                    "alpha_outside_clean_mask_pixels": int(
                        np.logical_and(local_alpha > 0, ~local_mask).sum()
                    ),
                    "removed_unanchored_alpha_island_count": removed_alpha_island_count,
                    "removed_unanchored_alpha_island_pixels": removed_alpha_island_pixels,
                    "removed_final_quantized_singleton_count": (
                        removed_quantized_singletons
                    ),
                    "source_colour_guard": colour_guard_report,
                    "semantic_pixels_before_matting": int(mask.sum()),
                    "semantic_pixels_after_matting": int(semantic_mask.sum()),
                    "sure_foreground_recall": 1.0,
                    "rejected_background_like_sure_foreground_pixels": (
                        rejected_sure_foreground_pixels
                    ),
                    "topology_by_alpha_threshold": thresholds,
                }
            )
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = [refined_by_id.get(layer.layer_id, layer) for layer in layer_list]
    return result, {
        "model": VITMATTE_MODEL,
        "revision": VITMATTE_REVISION,
        "model_safetensors_sha256": actual_weight_sha256,
        "model_safetensors_hash_verified_at_runtime": True,
        "model_source": "revision-pinned local cache; no run-time network access",
        "device": str(device),
        "policy": (
            "ViTMatte predicts fractional coverage only inside the colour/topology-cleaned "
            "text mask; sure foreground and all negative space are hard constraints"
        ),
        "refined_layer_count": len(records),
        "layers": records,
    }


__all__ = [
    "REFINED_TEXT_ROLES",
    "VITMATTE_MODEL",
    "VITMATTE_REVISION",
    "VITMATTE_WEIGHT_SHA256",
    "make_trimap",
    "refine_text_alpha_mattes",
]
