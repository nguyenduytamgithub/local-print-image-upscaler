from __future__ import annotations

import hashlib
import io
import json
import math
import os
import posixpath
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import safe_name
from .model import LayerSpec
from .restore import make_soft_alpha


V5_MAX_IMAGE_PIXELS = 500_000_000
Image.MAX_IMAGE_PIXELS = V5_MAX_IMAGE_PIXELS

PSD_MAX_DIMENSION = 30_000
PSD_SAFE_RAW_BYTES = 1_600_000_000
PSD_ROUNDTRIP_MAX_ERROR = 1
ORA_VERSION = "0.0.6"
ORA_MIMETYPE = b"image/openraster"
ORA_XML_MAX_BYTES = 16 * 1024 * 1024
ORA_MAX_MEMBERS = 4096
ORA_ROUNDTRIP_MAX_ERROR = 1
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_CARD_WIDTH = 300
CONTACT_SHEET_CARD_HEIGHT = 250
CONTACT_SHEET_LABEL_HEIGHT = 42
CONTACT_SHEET_PAGE_ROWS = 8
CONTACT_SHEET_PAGE_CAPACITY = CONTACT_SHEET_COLUMNS * CONTACT_SHEET_PAGE_ROWS
CONTACT_SHEET_MAX_DIMENSION = 4096
CONTACT_SHEET_MAX_PIXELS = 16_000_000
CONTACT_SHEET_OVERVIEW_CARDS = 16


@dataclass(slots=True)
class RenderedLayer:
    spec: LayerSpec
    rgba: Image.Image
    alpha: Image.Image
    left: int
    top: int
    source_bbox: tuple[int, int, int, int]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.rgba.width, self.top + self.rgba.height


@dataclass(slots=True)
class _UserGroupPlan:
    group_id: str
    name: str
    members: list[RenderedLayer]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_color_png(
    image: Image.Image,
    target,
    icc_profile: bytes | None,
    *,
    compress_level: int = 4,
) -> None:
    """Save RGB/RGBA artwork with its explicit sRGB document profile."""

    if image.mode not in {"RGB", "RGBA"}:
        raise ValueError(f"RGB ICC only applies to RGB/RGBA artwork, got {image.mode!r}.")
    options: dict[str, object] = {
        "format": "PNG",
        "compress_level": compress_level,
    }
    if icc_profile:
        options["icc_profile"] = icc_profile
    image.save(target, **options)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def resize_rgb(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(image_rgb, "RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8, copy=True)


def resize_alpha(
    mask: np.ndarray,
    size: tuple[int, int],
    *,
    alpha_matte: np.ndarray | None = None,
) -> np.ndarray:
    """Resize a binary or source-resolution fractional matte with Lanczos.

    ``mask`` remains the semantic/topology authority.  A supplied matte can
    describe sub-pixel coverage only inside that mask; malformed or escaping
    values are rejected instead of silently contaminating an editable layer.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional.")
    if alpha_matte is None:
        source = make_soft_alpha(binary)
    else:
        source = np.asarray(alpha_matte, dtype=np.float32)
        if source.ndim != 2 or source.shape != binary.shape:
            raise ValueError(
                f"alpha_matte shape {source.shape} does not match mask {binary.shape}."
            )
        if not np.isfinite(source).all() or (source < 0).any() or (source > 1).any():
            raise ValueError("alpha_matte must contain finite values from 0 through 1.")
        if np.logical_and(source > 0, ~binary).any():
            raise ValueError("alpha_matte cannot own pixels outside its semantic mask.")
    source_alpha = np.clip(source * 255.0 + 0.5, 0, 255).astype(np.uint8)
    image = Image.fromarray(source_alpha, "L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    array = np.array(image, dtype=np.float32) / 255.0
    array[array < 1.0 / 255.0] = 0.0
    return array


def resize_semantic_support(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize ownership without inventing pixels outside the selected region."""

    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, "L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def resize_refined_envelope(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Return the narrow legal envelope for a resampled fractional matte.

    Lanczos needs a one-source-pixel runway to represent fractional coverage
    between samples.  This envelope allows that antialiasing transition while
    still forbidding remote panel lines, shadows, letters, or restoration
    residue from entering a movable layer.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional.")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    envelope = cv2.dilate(binary.astype(np.uint8), kernel).astype(bool)
    return resize_semantic_support(envelope, size)


def serialized_base_alpha_u8(
    spec: LayerSpec,
    size: tuple[int, int],
    *,
    matte_policy: str = "clean",
) -> np.ndarray:
    """Return the exact full-canvas uint8 alpha the renderer starts from.

    This is the single serialization authority for planning and rendering.
    Ordinary semantic layers are intentionally *not* binary: ``make_soft_alpha``
    supplies their antialiased inner edge, then clean delivery clips that edge
    to nearest-resized semantic ownership. Refined mattes instead retain their
    Lanczos coverage only inside the narrow legal resampling envelope.
    """

    if matte_policy not in {"clean", "faithful"}:
        raise ValueError("matte_policy must be 'clean' or 'faithful'.")
    core_alpha = resize_alpha(
        spec.mask,
        size,
        alpha_matte=spec.alpha_matte,
    )
    if matte_policy == "clean":
        if spec.alpha_matte is not None:
            core_alpha *= resize_refined_envelope(spec.mask, size)
        else:
            core_alpha *= resize_semantic_support(spec.mask, size)
    return np.ceil(
        np.maximum(0.0, np.clip(core_alpha, 0.0, 1.0) * 255.0 - 1e-7)
    ).astype(np.uint8)


def _ceil_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return mathematical ceil(numerator / denominator) for integer arrays."""

    return -np.floor_divide(-numerator, denominator)


def _minimum_pillow_alpha_u8(
    target_u8: np.ndarray,
    below_u8: np.ndarray,
) -> np.ndarray:
    """Return the least uint8 alpha able to reproduce every RGB channel.

    Pillow composites one source channel ``F`` over an opaque lower channel
    ``B`` as ``floor((A*F + (255-A)*B + 127) / 255)``.  Solving that integer
    inequality at both legal foreground endpoints gives the exact minimum
    serialized alpha, including Pillow's half-level rounding tolerance.
    """

    target = np.asarray(target_u8, dtype=np.int32)
    below = np.asarray(below_u8, dtype=np.int32)
    if target.shape != below.shape or target.ndim < 1 or target.shape[-1] != 3:
        raise ValueError("target_u8 and below_u8 must be same-shape RGB arrays.")
    lower_target = 255 * target - 127
    upper_target = 255 * target + 127

    brighter_denominator = np.maximum(255 - below, 1)
    brighter = _ceil_divide(
        lower_target - 255 * below,
        brighter_denominator,
    )
    darker_denominator = np.maximum(below, 1)
    darker = _ceil_divide(
        255 * below - upper_target,
        darker_denominator,
    )
    required = np.where(
        target > below,
        brighter,
        np.where(target < below, darker, 0),
    )
    return np.clip(required.max(axis=-1), 0, 255).astype(np.uint8)


def _pillow_foreground_bounds(
    target_u8: np.ndarray,
    below_u8: np.ndarray,
    alpha_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact legal foreground bounds and per-channel feasibility."""

    target = np.asarray(target_u8, dtype=np.int32)
    below = np.asarray(below_u8, dtype=np.int32)
    alpha = np.asarray(alpha_u8, dtype=np.int32)
    if target.shape != below.shape or target.ndim < 1 or target.shape[-1] != 3:
        raise ValueError("target_u8 and below_u8 must be same-shape RGB arrays.")
    if alpha.shape != target.shape[:-1]:
        raise ValueError("alpha_u8 must match the RGB arrays without their channel axis.")

    alpha_channel = alpha[..., None]
    inverse_alpha = 255 - alpha_channel
    safe_alpha = np.maximum(alpha_channel, 1)
    lower_target = 255 * target - 127
    upper_target = 255 * target + 127
    lower_raw = _ceil_divide(
        lower_target - inverse_alpha * below,
        safe_alpha,
    )
    upper_raw = np.floor_divide(
        upper_target - inverse_alpha * below,
        safe_alpha,
    )
    lower = np.clip(lower_raw, 0, 255)
    upper = np.clip(upper_raw, 0, 255)
    feasible = np.logical_and.reduce(
        (lower_raw <= 255, upper_raw >= 0, lower <= upper)
    )

    # At alpha zero the foreground is ignored.  Preserve deterministic bounds
    # while recording that only an already-equal lower channel is feasible.
    transparent = alpha_channel == 0
    feasible = np.where(transparent, target == below, feasible)
    lower = np.where(transparent, target, lower)
    upper = np.where(transparent, target, upper)
    return lower, upper, feasible


def _solve_pillow_foreground_u8(
    target_u8: np.ndarray,
    below_u8: np.ndarray,
    alpha_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose an exact uint8 foreground and return per-pixel feasibility."""

    target = np.asarray(target_u8, dtype=np.int32)
    below = np.asarray(below_u8, dtype=np.int32)
    alpha = np.asarray(alpha_u8, dtype=np.int32)
    lower, upper, feasible_channels = _pillow_foreground_bounds(
        target, below, alpha
    )
    alpha_channel = alpha[..., None]
    inverse_alpha = 255 - alpha_channel
    safe_alpha = np.maximum(alpha_channel, 1)
    # Select the nearest integer to the analytic foreground, then constrain it
    # to the exact interval.  For infeasible pixels this is still the closest
    # bounded approximation, preserving the renderer's diagnostic behavior.
    analytic = (
        255 * target - inverse_alpha * below
    ).astype(np.float64) / safe_alpha
    candidate = np.clip(np.floor(analytic + 0.5), 0, 255).astype(np.int32)
    exact = np.minimum(np.maximum(candidate, lower), upper)
    foreground = np.where(feasible_channels, exact, candidate)
    foreground = np.where(alpha_channel == 0, target, foreground).astype(np.uint8)

    composed = np.floor_divide(
        alpha_channel * foreground.astype(np.int32)
        + inverse_alpha * below
        + 127,
        255,
    )
    if not np.array_equal(composed[feasible_channels], target[feasible_channels]):
        raise RuntimeError("Exact Pillow foreground solver failed its integer invariant.")
    return foreground, feasible_channels.all(axis=-1)


def project_clean_plate_for_alpha(
    master_rgb: np.ndarray,
    clean_candidate_rgb: np.ndarray,
    alpha_u8: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Project a clean plate to the nearest alpha-feasible 8-bit colour.

    Pillow's opaque-destination composition is exactly
    ``Q=floor((A*F + (255-A)*B + 127) / 255)``.  For output target ``T``, the
    legal numerator interval is therefore ``255*T-127`` through ``255*T+127``.
    This function clips each clean-candidate channel to the nearest lower colour
    for which at least one integer foreground exists at the fixed serialized
    alpha.  At alpha 0 the only feasible lower colour is T; at alpha 255 the
    clean candidate is unconstrained.
    """

    target = np.asarray(master_rgb)
    clean = np.asarray(clean_candidate_rgb)
    alpha = np.asarray(alpha_u8)
    if (
        target.dtype != np.uint8
        or clean.dtype != np.uint8
        or target.ndim != 3
        or clean.ndim != 3
        or target.shape != clean.shape
        or target.shape[2] != 3
    ):
        raise ValueError(
            "master_rgb and clean_candidate_rgb must be same-shape uint8 RGB arrays."
        )
    if alpha.dtype != np.uint8 or alpha.ndim != 2 or alpha.shape != target.shape[:2]:
        raise ValueError("alpha_u8 must be a same-canvas two-dimensional uint8 array.")

    target_i = target.astype(np.int32)
    clean_i = clean.astype(np.int32)
    alpha_i = alpha.astype(np.int32)[..., None]
    inverse_alpha = 255 - alpha_i
    safe_inverse_alpha = np.maximum(inverse_alpha, 1)
    lower_target = 255 * target_i - 127
    upper_target = 255 * target_i + 127
    lower = _ceil_divide(
        lower_target - 255 * alpha_i,
        safe_inverse_alpha,
    )
    upper = np.floor_divide(upper_target, safe_inverse_alpha)
    lower = np.clip(lower, 0, 255)
    upper = np.clip(upper, 0, 255)
    opaque = alpha_i == 255
    lower = np.where(opaque, 0, lower)
    upper = np.where(opaque, 255, upper)
    if np.any(lower > upper):  # pragma: no cover - T itself is always feasible
        raise RuntimeError("Computed alpha-feasible colour interval is empty.")
    projected_i = np.minimum(np.maximum(clean_i, lower), upper)
    projected = projected_i.astype(np.uint8)

    foreground, feasible_pixels = _solve_pillow_foreground_u8(
        target, projected, alpha
    )
    composed = np.floor_divide(
        alpha_i * foreground.astype(np.int32)
        + inverse_alpha * projected_i
        + 127,
        255,
    )
    exact_error = np.abs(composed - target_i)
    if not feasible_pixels.all() or exact_error.max() != 0:
        raise RuntimeError("Projected clean plate is not exactly alpha-feasible.")

    changed_channels = projected_i != clean_i
    changed_pixels = np.any(changed_channels, axis=2)
    target_f = target.astype(np.float64) / 255.0
    projected_f = projected.astype(np.float64) / 255.0
    alpha_f = alpha.astype(np.float64) / 255.0
    delta = target_f - projected_f
    required_up = np.where(
        delta > 0,
        delta / np.maximum(1.0 - projected_f, 1.0 / 255.0),
        0.0,
    )
    required_down = np.where(
        delta < 0,
        -delta / np.maximum(projected_f, 1.0 / 255.0),
        0.0,
    )
    required = np.maximum(required_up, required_down).max(axis=2)
    excess = np.maximum(required - alpha_f, 0.0)
    return projected, {
        "policy": "nearest per-channel clean colour inside exact canonical-alpha feasibility interval",
        "serialized_alpha_safety_margin_levels": 0,
        "changed_pixel_count": int(changed_pixels.sum()),
        "changed_channel_count": int(changed_channels.sum()),
        "unchanged_pixel_count": int((~changed_pixels).sum()),
        "alpha_zero_pixel_count": int((alpha == 0).sum()),
        "alpha_opaque_pixel_count": int((alpha == 255).sum()),
        "mean_abs_projection_distance": round(
            float(np.abs(projected_i - clean_i).mean()), 6
        ),
        "max_abs_projection_distance": int(
            np.abs(projected_i - clean_i).max()
        ),
        "maximum_continuous_required_alpha_excess": round(float(excess.max()), 12),
        "maximum_exact_required_alpha_excess_levels": int(
            np.maximum(
                _minimum_pillow_alpha_u8(target, projected).astype(np.int16)
                - alpha.astype(np.int16),
                0,
            ).max()
        ),
        "exact_pillow_recomposition_max_abs_error": int(exact_error.max()),
        "feasibility_gate_passed": True,
    }


def _category_order(category: str) -> int:
    return {"detail_group": 0, "object": 1, "text_raster": 2}.get(category, 0)


def ordered_layer_specs(specs: list[LayerSpec]) -> list[LayerSpec]:
    """Return the canonical container-compatible bottom-to-top render order."""

    base_order = sorted(
        specs,
        key=lambda spec: (
            int(spec.metadata.get("hierarchy_depth", 0)),
            _category_order(spec.category),
            int(spec.metadata.get("display_order", 0)),
        ),
    )
    spec_ids = [spec.layer_id for spec in base_order]
    if len(spec_ids) != len(set(spec_ids)):
        raise RuntimeError("Duplicate V5 layer ids cannot be rendered.")
    by_id = {spec.layer_id: spec for spec in base_order}
    child_specs: dict[str, list[LayerSpec]] = defaultdict(list)
    root_specs: list[LayerSpec] = []
    for spec in base_order:
        parent_id = spec.metadata.get("parent_id")
        if parent_id and str(parent_id) in by_id:
            child_specs[str(parent_id)].append(spec)
        else:
            root_specs.append(spec)

    ordered_specs: list[LayerSpec] = []
    active: set[str] = set()
    emitted: set[str] = set()

    def emit(spec: LayerSpec) -> None:
        if spec.layer_id in active:
            raise RuntimeError(f"Cyclic V5 layer hierarchy at: {spec.layer_id}")
        if spec.layer_id in emitted:
            return
        active.add(spec.layer_id)
        ordered_specs.append(spec)
        for child in child_specs.get(spec.layer_id, []):
            emit(child)
        active.remove(spec.layer_id)
        emitted.add(spec.layer_id)

    for root_spec in root_specs:
        emit(root_spec)
    if len(ordered_specs) != len(base_order):
        raise RuntimeError("V5 hierarchy did not produce a complete render order.")
    return ordered_specs


def render_layers(
    master_rgb: np.ndarray,
    background_rgb: np.ndarray,
    specs: list[LayerSpec],
    *,
    layer_targets: dict[str, np.ndarray] | None = None,
    support_masks: dict[str, np.ndarray] | None = None,
    matte_policy: str = "clean",
) -> tuple[list[RenderedLayer], Image.Image, dict[str, object]]:
    """Render RGBA layers using either clean ownership or legacy faithful effects.

    ``clean`` is the editable-delivery policy: alpha is forbidden outside the
    semantic mask.  ``faithful`` retains V5's former restoration/effect support
    and exists only for backwards-compatible diagnostics.
    """

    if matte_policy not in {"clean", "faithful"}:
        raise ValueError("matte_policy must be 'clean' or 'faithful'.")

    height, width = master_rgb.shape[:2]
    # A container cannot retain hierarchy and also keep children elsewhere in
    # the global z-order. Render parent and descendants contiguously so the PNG
    # preview, PSD groups, and ORA stacks share one bottom-to-top order.
    ordered_specs = ordered_layer_specs(specs)
    # Keep the working composite in the same 8-bit representation that the
    # exported PNG/PSD/ORA layers use. A float-only ideal can report zero error
    # while the actual saved layers differ visibly after alpha quantization.
    current_u8 = np.array(background_rgb, dtype=np.uint8, copy=True)
    rendered: list[RenderedLayer] = []
    for spec in ordered_specs:
        refined_alpha = spec.alpha_matte is not None
        base_alpha_canvas_u8 = serialized_base_alpha_u8(
            spec,
            (width, height),
            matte_policy=matte_policy,
        )
        core_alpha = base_alpha_canvas_u8.astype(np.float32) / 255.0
        semantic_support = resize_semantic_support(spec.mask, (width, height))
        if matte_policy == "clean":
            if refined_alpha:
                # Preserve the continuous ViTMatte/Lanczos edge instead of
                # clipping it into nearest-neighbour xN blocks.  The model was
                # already hard-clipped to the clean source topology; a one-px
                # envelope is the only legal resampling runway.
                support = base_alpha_canvas_u8 > 0
            else:
                # Binary object/panel ownership remains exact and conservative.
                support = semantic_support
        else:
            effect_radius = max(2, min(72, int(round(min(width, height) / 150))))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (effect_radius * 2 + 1, effect_radius * 2 + 1)
            )
            support = cv2.dilate((core_alpha > 0.002).astype(np.uint8), kernel).astype(bool)
            source_support = (support_masks or {}).get(spec.layer_id)
            if source_support is not None:
                source_support_array = np.asarray(source_support, dtype=bool)
                if source_support_array.ndim != 2:
                    raise ValueError(f"Support mask for {spec.layer_id} must be two-dimensional.")
                support_image = Image.fromarray(source_support_array.astype(np.uint8) * 255, "L")
                if support_image.size != (width, height):
                    support_image = support_image.resize((width, height), Image.Resampling.NEAREST)
                support |= np.asarray(support_image, dtype=np.uint8) > 0
        ys, xs = np.where(support)
        if not len(xs):
            continue
        left, top, right, bottom = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
        base_alpha_u8 = base_alpha_canvas_u8[top:bottom, left:right]
        crop_support = support[top:bottom, left:right]
        # Lanczos needs the narrow refined envelope for fractional coverage,
        # but exact-colour correction must not turn its positive ringing lobes
        # into new opaque islands.  Only nearest-resized semantic ownership may
        # receive required_alpha promotion; the envelope keeps base_alpha only.
        promotion_support = (
            semantic_support[top:bottom, left:right]
            if matte_policy == "clean" and refined_alpha
            else crop_support
        )
        below_u8 = current_u8[top:bottom, left:right]
        desired_canvas = (layer_targets or {}).get(spec.layer_id, master_rgb)
        desired_u8 = np.asarray(
            desired_canvas[top:bottom, left:right], dtype=np.uint8
        )
        # The shared serialization helper above is also used by the reverse
        # stack planner. Continuous-alpha bounds are unnecessarily strict near
        # an 8-bit rounding boundary and used to promote a canonical A=1 edge
        # to A=2 even though A=1 could reproduce the output exactly.
        required_alpha_u8 = _minimum_pillow_alpha_u8(desired_u8, below_u8)
        # Required alpha is useful for colour decontamination at the contour,
        # but in clean mode it is allowed only *inside* semantic ownership.
        promotable_required = np.where(
            promotion_support, required_alpha_u8, 0
        ).astype(np.uint8)
        alpha_u8 = np.maximum(base_alpha_u8, promotable_required)
        foreground_u8, _feasible_pixels = _solve_pillow_foreground_u8(
            desired_u8, below_u8, alpha_u8
        )
        rgba = np.dstack(
            (
                foreground_u8,
                alpha_u8,
            )
        )
        rgba_image = Image.fromarray(rgba, "RGBA")
        # Pillow performs the exact same integer alpha-composite operation used
        # when consumers flatten the exported PNG layers. Updating the preview
        # from this result makes it an honest rendering of the deliverables.
        below_rgba = Image.fromarray(
            np.dstack(
                (
                    below_u8,
                    np.full((bottom - top, right - left), 255, dtype=np.uint8),
                )
            ),
            "RGBA",
        )
        below_rgba.alpha_composite(rgba_image)
        current_u8[top:bottom, left:right] = np.asarray(below_rgba, dtype=np.uint8)[..., :3]
        source_bbox = spec.bbox
        rendered.append(
            RenderedLayer(
                spec=spec,
                rgba=rgba_image,
                alpha=Image.fromarray(rgba[..., 3], "L"),
                left=left,
                top=top,
                source_bbox=source_bbox,
            )
        )
    composite = Image.fromarray(current_u8, "RGB")
    error = np.abs(np.array(composite, dtype=np.int16) - master_rgb.astype(np.int16))
    mse = float(np.mean(error.astype(np.float64) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse)
    report = {
        "rendered_layer_count": len(rendered),
        "matte_policy": matte_policy,
        "recomposition_max_abs_error": int(error.max()),
        "recomposition_mean_abs_error": round(float(error.mean()), 6),
        "recomposition_psnr_db": "infinite" if math.isinf(psnr) else round(psnr, 4),
        "edge_policy": (
            "binary mattes stay inside nearest-resized semantic ownership; topology-cleaned "
            "fractional text mattes use Lanczos base alpha inside a one-source-pixel envelope, "
            "while required-alpha promotion is restricted to nearest semantic ownership; edge "
            "colours and the minimum necessary alpha are solved with Pillow's exact integer "
            "source-over equation against the actual 8-bit lower composite"
            if matte_policy == "clean"
            else "legacy semantic core plus restoration/effect support and conservative ceil-to-8-bit "
            "model alpha; required alpha and foreground colours use Pillow's exact integer "
            "source-over equation against the actual 8-bit lower composite"
        ),
        "preview_basis": "actual cropped 8-bit RGBA assets composited bottom-to-top with Pillow",
        "layer_order_policy": "container-compatible depth-first bottom-to-top (parent, then descendants)",
    }
    return rendered, composite, report


def rendered_alpha_canvases(
    rendered: list[RenderedLayer],
    canvas_size: tuple[int, int],
    *,
    refined_only: bool = False,
) -> dict[str, np.ndarray]:
    """Expand cropped serialized alpha into stable full-canvas uint8 references."""

    width, height = canvas_size
    if width <= 0 or height <= 0:
        raise ValueError("canvas_size must contain positive dimensions.")
    result: dict[str, np.ndarray] = {}
    for item in rendered:
        if refined_only and item.spec.alpha_matte is None:
            continue
        layer_id = item.spec.layer_id
        if layer_id in result:
            raise ValueError(f"Duplicate rendered layer id: {layer_id}")
        left, top, right, bottom = item.bbox
        if not (0 <= left <= right <= width and 0 <= top <= bottom <= height):
            raise ValueError(f"Rendered alpha escapes canvas: {layer_id}")
        alpha = np.asarray(item.alpha, dtype=np.uint8)
        if alpha.shape != (bottom - top, right - left):
            raise ValueError(f"Rendered alpha shape mismatch: {layer_id}")
        canvas = np.zeros((height, width), dtype=np.uint8)
        canvas[top:bottom, left:right] = alpha
        result[layer_id] = canvas
    return result


def find_required_alpha_promoted_singletons(
    rendered: list[RenderedLayer],
    canvas_size: tuple[int, int],
    source_rgb: np.ndarray,
    *,
    high_alpha_threshold: int = 128,
    low_source_alpha_threshold: float = 0.5,
    minimum_glyph_height: int = 20,
    background_source_alpha_threshold: float = 0.25,
    local_radius: int = 5,
    local_background_lab_limit: float = 34.0,
    minimum_background_neighbours: int = 4,
    maximum_per_layer: int = 2,
) -> list[dict[str, object]]:
    """Find isolated text pixels promoted by the exact-composition solver.

    The clean renderer may raise a very small, non-zero source matte value to
    opaque alpha when the lower/parent layer has already been cleaned beneath
    that pixel.  That is necessary for exact recomposition, but a lone promoted
    pixel is not a useful editable edge.  Detection deliberately runs at source
    resolution so rendered coordinates map one-to-one back to ``mask`` and
    ``alpha_matte``.

    Real detached punctuation and accents are protected in two independent
    ways: only one-pixel output components whose *source* matte is below 0.5 are
    considered, and the source colour must have a locally connected-looking
    background cluster while not matching a reliable glyph-body palette.  The
    restriction to OCR/visual text rows and reasonably tall glyph groups also
    prevents this topology policy from changing small icons or object layers.
    """

    width, height = canvas_size
    if width <= 0 or height <= 0:
        raise ValueError("canvas_size must contain positive dimensions.")
    if not 1 <= high_alpha_threshold <= 255:
        raise ValueError("high_alpha_threshold must be from 1 through 255.")
    if not 0.0 < low_source_alpha_threshold <= 1.0:
        raise ValueError("low_source_alpha_threshold must be in (0, 1].")
    if minimum_glyph_height < 1:
        raise ValueError("minimum_glyph_height must be positive.")
    rgb = np.asarray(source_rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape != (height, width, 3):
        raise ValueError(
            "source_rgb must be a source-resolution uint8 RGB array matching canvas_size."
        )
    if not 0.0 < background_source_alpha_threshold < low_source_alpha_threshold:
        raise ValueError(
            "background_source_alpha_threshold must be positive and below "
            "low_source_alpha_threshold."
        )
    if local_radius < 2:
        raise ValueError("local_radius must be at least 2 pixels.")
    if local_background_lab_limit <= 0:
        raise ValueError("local_background_lab_limit must be positive.")
    if minimum_background_neighbours < 2:
        raise ValueError("minimum_background_neighbours must be at least 2.")
    if maximum_per_layer < 1:
        raise ValueError("maximum_per_layer must be positive.")

    source_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    pure_text_roles = {"ocr_text_line", "visual_text_row"}
    for item in rendered:
        spec = item.spec
        source_alpha = spec.alpha_matte
        if source_alpha is None:
            continue
        if spec.category != "text_raster":
            continue
        if str(spec.metadata.get("grouping_role", "")) not in pure_text_roles:
            continue
        if spec.bbox[3] - spec.bbox[1] < minimum_glyph_height:
            continue
        if spec.mask.shape != (height, width) or source_alpha.shape != (height, width):
            raise ValueError(
                "Promoted-singleton preflight must run at source resolution; "
                f"layer {spec.layer_id!r} is {spec.mask.shape}, canvas is {(height, width)}."
            )

        # Only substantial >=.75 components define protected text colour.
        # Tiny components cannot vote themselves into the palette, and broad
        # panel-coloured false support is rejected later relative to each
        # candidate's local background estimate.
        glyph_height = spec.bbox[3] - spec.bbox[1]
        strong = np.asarray(source_alpha >= 0.75, dtype=np.uint8)
        strong_count, strong_labels, strong_stats, _ = cv2.connectedComponentsWithStats(
            strong, 8
        )
        minimum_body_area = max(8, int(round(0.018 * glyph_height * glyph_height)))
        body_palettes: list[np.ndarray] = []
        for body_id in range(1, strong_count):
            if int(strong_stats[body_id, cv2.CC_STAT_AREA]) < minimum_body_area:
                continue
            body_palettes.append(np.median(source_lab[strong_labels == body_id], axis=0))

        alpha = np.asarray(item.alpha, dtype=np.uint8)
        high = (alpha >= high_alpha_threshold).astype(np.uint8)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            high, 8
        )
        for component_id in range(1, component_count):
            if int(stats[component_id, cv2.CC_STAT_AREA]) != 1:
                continue
            local_y, local_x = np.argwhere(labels == component_id)[0]
            global_x = int(item.left + int(local_x))
            global_y = int(item.top + int(local_y))
            if not (0 <= global_x < width and 0 <= global_y < height):
                raise RuntimeError(
                    f"Rendered alpha for {spec.layer_id!r} escaped the source canvas."
                )
            base_alpha = float(source_alpha[global_y, global_x])
            # Zero cannot be owned by a source-resolution clean support.  Keep
            # this guard so stale or malformed render records are never used to
            # delete unrelated semantic pixels.
            if not 0.0 < base_alpha < low_source_alpha_threshold:
                continue

            y0 = max(0, global_y - local_radius)
            y1 = min(height, global_y + local_radius + 1)
            x0 = max(0, global_x - local_radius)
            x1 = min(width, global_x + local_radius + 1)
            local_mask = spec.mask[y0:y1, x0:x1]
            local_source_alpha = source_alpha[y0:y1, x0:x1]
            background_evidence = (~local_mask) | (
                local_source_alpha < background_source_alpha_threshold
            )
            background_evidence = np.array(background_evidence, dtype=bool, copy=True)
            background_evidence[global_y - y0, global_x - x0] = False
            evidence_lab = source_lab[y0:y1, x0:x1][background_evidence]
            if len(evidence_lab) < minimum_background_neighbours:
                continue
            candidate_lab = source_lab[global_y, global_x]
            evidence_distance = np.linalg.norm(evidence_lab - candidate_lab, axis=1)
            close = evidence_distance <= local_background_lab_limit
            close_count = int(close.sum())
            if close_count < minimum_background_neighbours:
                continue
            # A mixed edge may contain several colours.  Use only the closest
            # locally supported cluster rather than letting an adjacent yellow
            # glyph or white outline move the background median.
            close_lab = evidence_lab[close]
            if len(close_lab) > 16:
                nearest = np.argsort(evidence_distance[close])[:16]
                close_lab = close_lab[nearest]
            local_background_lab = np.median(close_lab, axis=0)
            background_distance = float(
                np.linalg.norm(candidate_lab - local_background_lab)
            )
            if background_distance > local_background_lab_limit:
                continue

            protected_palette_distance: float | None = None
            if body_palettes:
                body_distance = np.array(
                    [
                        float(np.linalg.norm(candidate_lab - palette))
                        for palette in body_palettes
                    ],
                    dtype=np.float32,
                )
                body_background_contrast = np.array(
                    [
                        float(np.linalg.norm(local_background_lab - palette))
                        for palette in body_palettes
                    ],
                    dtype=np.float32,
                )
                protected_body = body_background_contrast >= 24.0
                if protected_body.any():
                    protected_palette_distance = float(body_distance[protected_body].min())
                    # A low model alpha is not deletion evidence if its source
                    # colour is substantially closer to a trustworthy glyph
                    # palette than to the local background (real punctuation,
                    # diacritics and one-pixel tips fall into this case).
                    if (
                        protected_palette_distance <= 28.0
                        and protected_palette_distance + 6.0 < background_distance
                    ):
                        continue

            candidates[spec.layer_id].append(
                {
                    "layer_id": spec.layer_id,
                    "x": global_x,
                    "y": global_y,
                    "source_alpha": round(base_alpha, 8),
                    "rendered_alpha": int(alpha[local_y, local_x]),
                    "component_area": 1,
                    "local_background_neighbour_count": close_count,
                    "local_background_distance_lab": round(background_distance, 4),
                    "protected_body_palette_distance_lab": (
                        None
                        if protected_palette_distance is None
                        else round(protected_palette_distance, 4)
                    ),
                    "reason": "required_alpha_promoted_low_source_singleton",
                }
            )

    records: list[dict[str, object]] = []
    for layer_id in sorted(candidates):
        ranked = sorted(
            candidates[layer_id],
            key=lambda record: (
                float(record["local_background_distance_lab"]),
                -int(record["local_background_neighbour_count"]),
                int(record["y"]),
                int(record["x"]),
            ),
        )
        records.extend(ranked[:maximum_per_layer])
    records.sort(
        key=lambda record: (str(record["layer_id"]), int(record["y"]), int(record["x"]))
    )
    return records


def prune_required_alpha_promoted_singletons(
    specs: list[LayerSpec],
    records: list[dict[str, object]],
    *,
    low_source_alpha_threshold: float = 0.5,
) -> tuple[list[LayerSpec], dict[str, object]]:
    """Remove verified renderer-promoted singleton support without mutation.

    Callers must rebuild all lower/background targets after this operation.
    That reassigns the source pixel to the parent instead of merely punching a
    hole in the final foreground alpha, preserving exact recomposition.
    """

    if not 0.0 < low_source_alpha_threshold <= 1.0:
        raise ValueError("low_source_alpha_threshold must be in (0, 1].")
    requested: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for record in records:
        try:
            layer_id = str(record["layer_id"])
            x = int(record["x"])
            y = int(record["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed promoted-singleton record.") from exc
        requested[layer_id].add((x, y))

    known_ids = {spec.layer_id for spec in specs}
    unknown_ids = sorted(set(requested) - known_ids)
    if unknown_ids:
        raise ValueError(f"Promoted-singleton records reference unknown layers: {unknown_ids}")

    applied: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    updated: list[LayerSpec] = []
    for spec in specs:
        coordinates = sorted(requested.get(spec.layer_id, set()), key=lambda point: (point[1], point[0]))
        if not coordinates:
            updated.append(spec)
            continue
        if spec.alpha_matte is None:
            raise ValueError(
                f"Promoted-singleton record targets layer {spec.layer_id!r} without alpha_matte."
            )
        mask = np.array(spec.mask, dtype=bool, copy=True)
        alpha_matte = np.array(spec.alpha_matte, dtype=np.float32, copy=True)
        if alpha_matte.shape != mask.shape:
            raise ValueError(f"Layer {spec.layer_id!r} has inconsistent mask/matte shapes.")
        height, width = mask.shape
        for x, y in coordinates:
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"Promoted-singleton coordinate {(x, y)} is outside layer {spec.layer_id!r}."
                )
            base_alpha = float(alpha_matte[y, x])
            if mask[y, x] and 0.0 < base_alpha < low_source_alpha_threshold:
                mask[y, x] = False
                alpha_matte[y, x] = 0.0
                applied.append(
                    {
                        "layer_id": spec.layer_id,
                        "x": x,
                        "y": y,
                        "source_alpha_before": round(base_alpha, 8),
                    }
                )
            else:
                skipped.append(
                    {
                        "layer_id": spec.layer_id,
                        "x": x,
                        "y": y,
                        "source_alpha": round(base_alpha, 8),
                        "mask_owned": bool(mask[y, x]),
                        "reason": "stale_or_protected_source_support",
                    }
                )
        updated.append(
            LayerSpec(
                layer_id=spec.layer_id,
                name=spec.name,
                category=spec.category,
                mask=mask,
                score=spec.score,
                source_ids=list(spec.source_ids),
                label=spec.label,
                text=spec.text,
                metadata=dict(spec.metadata),
                alpha_matte=alpha_matte,
            )
        )

    return updated, {
        "requested_count": sum(len(points) for points in requested.values()),
        "applied_count": len(applied),
        "skipped_count": len(skipped),
        "applied": applied,
        "skipped": skipped,
        "policy": (
            "remove only source-resolution pure-text high-alpha area-1 components "
            "that were promoted from a non-zero source matte below 0.5; rebuild lower targets"
        ),
    }


def _binary_topology_metrics(
    binary: np.ndarray,
    *,
    micro_component_area: int,
) -> dict[str, int]:
    """Measure foreground with 8-connectivity and enclosed background with 4.

    Cropping to the occupied bounds plus an explicit zero border makes the
    counter/hole result independent of the rendered layer crop while retaining
    the complementary FG8/BG4 topology used for raster glyphs.
    """

    foreground = np.asarray(binary, dtype=bool)
    if foreground.ndim != 2:
        raise ValueError("binary topology input must be two-dimensional.")
    if micro_component_area < 1:
        raise ValueError("micro_component_area must be positive.")
    ys, xs = np.where(foreground)
    if not len(xs):
        return {
            "foreground_pixels": 0,
            "foreground_component_count_fg8": 0,
            "micro_component_count": 0,
            "enclosed_background_count_bg4": 0,
        }
    cropped = foreground[
        int(ys.min()) : int(ys.max()) + 1,
        int(xs.min()) : int(xs.max()) + 1,
    ]
    cropped = np.pad(cropped, 1, mode="constant", constant_values=False)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(
        cropped.astype(np.uint8), connectivity=8
    )
    areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]

    background = (~cropped).astype(np.uint8)
    background_count, background_labels, _background_stats, _ = (
        cv2.connectedComponentsWithStats(background, connectivity=4)
    )
    border_labels = set(int(value) for value in background_labels[0, :])
    border_labels.update(int(value) for value in background_labels[-1, :])
    border_labels.update(int(value) for value in background_labels[:, 0])
    border_labels.update(int(value) for value in background_labels[:, -1])
    enclosed_background = sum(
        component_id not in border_labels
        for component_id in range(1, background_count)
    )
    return {
        "foreground_pixels": int(cropped.sum()),
        "foreground_component_count_fg8": max(0, count - 1),
        "micro_component_count": sum(area <= micro_component_area for area in areas),
        "enclosed_background_count_bg4": int(enclosed_background),
    }


def _enclosed_background_labels(foreground: np.ndarray) -> tuple[np.ndarray, int]:
    """Label BG4 holes while forcing one explicit exterior background frame."""

    binary = np.asarray(foreground, dtype=bool)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    background = (~padded).astype(np.uint8)
    count, labels = cv2.connectedComponents(background, connectivity=4)
    exterior = int(labels[0, 0])
    holes = labels.astype(np.int32, copy=True)
    holes[holes == exterior] = 0
    # Input foreground is zero in ``background`` and therefore already label 0.
    hole_ids = np.unique(holes)
    hole_ids = hole_ids[hole_ids > 0]
    if len(hole_ids):
        lookup = np.zeros(int(hole_ids.max()) + 1, dtype=np.int32)
        lookup[hole_ids] = np.arange(1, len(hole_ids) + 1, dtype=np.int32)
        holes = lookup[holes]
    return holes, int(len(hole_ids))


def _label_overlap_maps(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    left_count: int,
    right_count: int,
) -> tuple[list[set[int]], list[set[int]]]:
    """Return spatial overlap relations without scanning once per component."""

    if left_labels.shape != right_labels.shape:
        raise ValueError("Label images must share a shape.")
    left_to_right = [set() for _ in range(left_count + 1)]
    right_to_left = [set() for _ in range(right_count + 1)]
    overlap = (left_labels > 0) & (right_labels > 0)
    if not overlap.any():
        return left_to_right, right_to_left
    multiplier = right_count + 1
    encoded = (
        left_labels[overlap].astype(np.int64) * multiplier
        + right_labels[overlap].astype(np.int64)
    )
    for value in np.unique(encoded):
        left_id = int(value // multiplier)
        right_id = int(value % multiplier)
        left_to_right[left_id].add(right_id)
        right_to_left[right_id].add(left_id)
    return left_to_right, right_to_left


def _spatial_topology_correspondence(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
) -> dict[str, object]:
    """Compare thresholded mattes by spatial FG8/BG4 correspondence.

    Counts alone can be unchanged when two glyphs merge while a new island is
    born, or when an O counter is filled while a C/G aperture closes elsewhere.
    This gate pairs every component and every enclosed background region by
    overlap, so those cancelling failures cannot pass.
    """

    reference = np.asarray(expected, dtype=bool)
    candidate = np.asarray(actual, dtype=bool)
    if reference.ndim != 2 or candidate.ndim != 2 or reference.shape != candidate.shape:
        raise ValueError("expected and actual topology masks must be same-shape 2-D arrays.")
    if not np.isfinite(scale_x) or not np.isfinite(scale_y) or scale_x <= 0 or scale_y <= 0:
        raise ValueError("scale_x and scale_y must be finite and positive.")

    # The comparison is logically full-canvas, but topology outside the union
    # is guaranteed exterior background. Crop once to the occupied union before
    # connected-component labeling to avoid allocating several 5016² label
    # images for every refined layer and threshold.
    hole_core_radius = max(1, int(math.ceil(0.5 * min(scale_x, scale_y))))
    occupied_y, occupied_x = np.where(reference | candidate)
    if not len(occupied_x):
        return {
            "expected_foreground_component_count_fg8": 0,
            "actual_foreground_component_count_fg8": 0,
            "expected_enclosed_background_count_bg4": 0,
            "actual_enclosed_background_count_bg4": 0,
            "orphan_actual_component_pixels": 0,
            "merge_excess": 0,
            "split_excess": 0,
            "hole_core_radius_final_px": hole_core_radius,
            "missing_expected_pixels": 0,
            "orphan_actual_components": 0,
            "merged_actual_components": 0,
            "split_expected_components": 0,
            "missing_expected_holes": 0,
            "new_actual_holes": 0,
            "split_expected_holes": 0,
            "merged_actual_holes": 0,
            "expected_hole_core_intrusion_pixels": 0,
            "passed": True,
        }
    x0, x1 = int(occupied_x.min()), int(occupied_x.max()) + 1
    y0, y1 = int(occupied_y.min()), int(occupied_y.max()) + 1
    reference = reference[y0:y1, x0:x1]
    candidate = candidate[y0:y1, x0:x1]

    expected_padded = np.pad(reference, 1, mode="constant", constant_values=False)
    actual_padded = np.pad(candidate, 1, mode="constant", constant_values=False)
    expected_count, expected_labels, expected_stats, _ = cv2.connectedComponentsWithStats(
        expected_padded.astype(np.uint8), connectivity=8
    )
    actual_count, actual_labels, actual_stats, _ = cv2.connectedComponentsWithStats(
        actual_padded.astype(np.uint8), connectivity=8
    )
    expected_components = max(0, expected_count - 1)
    actual_components = max(0, actual_count - 1)
    expected_to_actual, actual_to_expected = _label_overlap_maps(
        expected_labels,
        actual_labels,
        expected_components,
        actual_components,
    )
    orphan_actual_ids = [
        component_id
        for component_id in range(1, actual_count)
        if not actual_to_expected[component_id]
    ]
    merged_actual_ids = [
        component_id
        for component_id in range(1, actual_count)
        if len(actual_to_expected[component_id]) > 1
    ]
    split_expected_ids = [
        component_id
        for component_id in range(1, expected_count)
        if len(expected_to_actual[component_id]) > 1
    ]
    missing_expected_pixels = int(np.logical_and(reference, ~candidate).sum())

    expected_holes, expected_hole_count = _enclosed_background_labels(reference)
    actual_holes, actual_hole_count = _enclosed_background_labels(candidate)
    expected_hole_to_actual, actual_hole_to_expected = _label_overlap_maps(
        expected_holes,
        actual_holes,
        expected_hole_count,
        actual_hole_count,
    )
    missing_expected_hole_ids = [
        hole_id
        for hole_id in range(1, expected_hole_count + 1)
        if not expected_hole_to_actual[hole_id]
    ]
    new_actual_hole_ids = [
        hole_id
        for hole_id in range(1, actual_hole_count + 1)
        if not actual_hole_to_expected[hole_id]
    ]
    split_expected_hole_ids = [
        hole_id
        for hole_id in range(1, expected_hole_count + 1)
        if len(expected_hole_to_actual[hole_id]) > 1
    ]
    merged_actual_hole_ids = [
        hole_id
        for hole_id in range(1, actual_hole_count + 1)
        if len(actual_hole_to_expected[hole_id]) > 1
    ]

    hole_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (hole_core_radius * 2 + 1, hole_core_radius * 2 + 1),
    )
    expected_hole_core = cv2.erode(
        (expected_holes > 0).astype(np.uint8), hole_kernel
    ).astype(bool)
    # ``expected_holes`` is padded by one pixel, as are the component labels.
    actual_foreground_padded = actual_padded
    hole_core_intrusion = int(
        np.logical_and(expected_hole_core, actual_foreground_padded).sum()
    )

    failures = {
        "missing_expected_pixels": missing_expected_pixels,
        "orphan_actual_components": len(orphan_actual_ids),
        "merged_actual_components": len(merged_actual_ids),
        "split_expected_components": len(split_expected_ids),
        "missing_expected_holes": len(missing_expected_hole_ids),
        "new_actual_holes": len(new_actual_hole_ids),
        "split_expected_holes": len(split_expected_hole_ids),
        "merged_actual_holes": len(merged_actual_hole_ids),
        "expected_hole_core_intrusion_pixels": hole_core_intrusion,
    }
    return {
        "expected_foreground_component_count_fg8": expected_components,
        "actual_foreground_component_count_fg8": actual_components,
        "expected_enclosed_background_count_bg4": expected_hole_count,
        "actual_enclosed_background_count_bg4": actual_hole_count,
        "orphan_actual_component_pixels": int(
            sum(int(actual_stats[index, cv2.CC_STAT_AREA]) for index in orphan_actual_ids)
        ),
        "merge_excess": int(
            sum(len(actual_to_expected[index]) - 1 for index in merged_actual_ids)
        ),
        "split_excess": int(
            sum(len(expected_to_actual[index]) - 1 for index in split_expected_ids)
        ),
        "hole_core_radius_final_px": hole_core_radius,
        **failures,
        "passed": not any(failures.values()),
    }


def matte_quality_report(
    rendered: list[RenderedLayer],
    canvas_size: tuple[int, int],
    *,
    topology_reference: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Audit exported ownership and scaled topology before publication.

    Runtime callers provide the stable source-resolution *rendered* alpha from
    the exact x1 preflight.  That canonical reference already includes valid
    required-alpha corrections, punctuation and counters. Scaling it with
    Lanczos distinguishes real xN topology damage from legitimate x1 solving.
    The raw-matte fallback exists for isolated library tests only.
    """

    width, height = canvas_size
    records: list[dict[str, object]] = []
    outside_total = 0
    outside_exact_total = 0
    topology_checked_layers = 0
    topology_failed_layers = 0
    topology_failed_checks = 0
    for item in rendered:
        semantic = resize_semantic_support(item.spec.mask, canvas_size)
        refined_alpha = item.spec.alpha_matte is not None
        allowed = (
            resize_refined_envelope(item.spec.mask, canvas_size)
            if refined_alpha
            else semantic
        )
        left, top, right, bottom = item.bbox
        if not (0 <= left <= right <= width and 0 <= top <= bottom <= height):
            raise RuntimeError(f"Rendered matte escapes canvas: {item.spec.layer_id}")
        alpha = np.asarray(item.alpha, dtype=np.uint8)
        expected_shape = (bottom - top, right - left)
        if alpha.shape != expected_shape:
            raise RuntimeError(
                f"Rendered alpha shape mismatch for {item.spec.layer_id}: "
                f"{alpha.shape} != {expected_shape}"
            )
        owned = alpha > 0
        high = alpha >= 128
        semantic_crop = semantic[top:bottom, left:right]
        allowed_crop = allowed[top:bottom, left:right]
        outside = int(np.logical_and(owned, ~allowed_crop).sum())
        outside_exact = int(np.logical_and(owned, ~semantic_crop).sum())
        outside_total += outside
        outside_exact_total += outside_exact
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            high.astype(np.uint8), 8
        )
        areas = sorted(
            (int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)),
            reverse=True,
        )
        high_area = int(high.sum())
        detached_area = sum(areas[1:]) if areas else 0
        layer_record: dict[str, object] = {
            "layer_id": item.spec.layer_id,
            "role": str(item.spec.metadata.get("grouping_role", "")),
            "matte_source": "vitmatte_fractional" if refined_alpha else "binary_semantic",
            "nonzero_alpha_pixels": int(owned.sum()),
            "high_alpha_pixels": high_area,
            "soft_alpha_pixels": int(np.logical_and(alpha > 0, alpha < 255).sum()),
            "high_alpha_component_count": max(0, count - 1),
            "detached_high_alpha_ratio": round(detached_area / max(1, high_area), 6),
            "alpha_outside_semantic_pixels": outside,
            "alpha_outside_exact_nearest_semantic_pixels": outside_exact,
        }

        if refined_alpha:
            topology_checked_layers += 1
            source_alpha = np.asarray(item.spec.alpha_matte, dtype=np.float32)
            if source_alpha.shape != item.spec.mask.shape:
                raise RuntimeError(
                    f"Source alpha shape mismatch for topology QA: {item.spec.layer_id}"
                )
            if topology_reference is not None:
                if item.spec.layer_id not in topology_reference:
                    raise ValueError(
                        "Missing canonical topology reference for refined layer: "
                        f"{item.spec.layer_id}"
                    )
                canonical_u8 = np.asarray(
                    topology_reference[item.spec.layer_id]
                )
                if canonical_u8.dtype != np.uint8 or canonical_u8.ndim != 2:
                    raise ValueError(
                        "Canonical topology references must be two-dimensional uint8 alpha."
                    )
                if canonical_u8.shape != item.spec.mask.shape:
                    raise ValueError(
                        f"Canonical topology reference shape {canonical_u8.shape} does not "
                        f"match source mask {item.spec.mask.shape} for {item.spec.layer_id}."
                    )
                reference_image = Image.fromarray(canonical_u8, "L")
                if reference_image.size != canvas_size:
                    reference_image = reference_image.resize(
                        canvas_size, Image.Resampling.LANCZOS
                    )
                expected_u8 = np.array(reference_image, dtype=np.uint8, copy=True)
                expected_u8[~allowed] = 0
                reference_policy = (
                    "Lanczos resize of stable x1 preflight serialized alpha, clipped to legal envelope"
                )
            else:
                reference_alpha = resize_alpha(
                    item.spec.mask,
                    canvas_size,
                    alpha_matte=source_alpha,
                )
                reference_alpha *= allowed
                expected_u8 = np.ceil(
                    np.maximum(0.0, reference_alpha * 255.0 - 1e-7)
                ).astype(np.uint8)
                reference_policy = (
                    "raw source matte fallback (tests only; production supplies canonical x1 alpha)"
                )
            actual_u8 = np.zeros((height, width), dtype=np.uint8)
            actual_u8[top:bottom, left:right] = alpha
            scale_x = width / max(1, item.spec.mask.shape[1])
            scale_y = height / max(1, item.spec.mask.shape[0])
            threshold_records: list[dict[str, object]] = []
            layer_topology_passed = True
            for threshold, actual_level in ((0.25, 64), (0.5, 128), (0.75, 192)):
                correspondence = _spatial_topology_correspondence(
                    expected_u8 >= actual_level,
                    actual_u8 >= actual_level,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                if not correspondence["passed"]:
                    topology_failed_checks += 1
                    layer_topology_passed = False
                threshold_records.append(
                    {
                        "threshold": threshold,
                        "integer_alpha_level": actual_level,
                        **correspondence,
                    }
                )
            if not layer_topology_passed:
                topology_failed_layers += 1
            layer_record["scaled_topology"] = {
                "connectivity": "foreground 8 / enclosed background 4",
                "reference": reference_policy,
                "passed": layer_topology_passed,
                "thresholds": threshold_records,
            }
        records.append(layer_record)
    return {
        "policy": (
            "binary alpha must remain inside semantic ownership; fractional text alpha may use "
            "only a one-source-pixel resampling envelope; every refined layer is spatially "
            "release-gated against its Lanczos-resized stable x1 rendered alpha at exact "
            "levels 64/128/192 using FG8/BG4 components, counters and protected hole cores"
        ),
        "layer_count": len(records),
        "alpha_outside_semantic_pixels": outside_total,
        "alpha_outside_exact_nearest_semantic_pixels": outside_exact_total,
        "ownership_gate_passed": outside_total == 0,
        "topology_checked_layer_count": topology_checked_layers,
        "topology_failed_layer_count": topology_failed_layers,
        "topology_failed_threshold_count": topology_failed_checks,
        "topology_gate_passed": topology_failed_layers == 0,
        "topology_reference_source": (
            "stable_source_resolution_preflight_render"
            if topology_reference is not None
            else "raw_source_matte_fallback"
        ),
        "layers": records,
    }


def _slug_layer(index: int, rendered: RenderedLayer) -> str:
    raw = safe_name(rendered.spec.name, rendered.spec.layer_id)
    asciiish = re.sub(r"[^0-9A-Za-z._ -]+", "_", raw)
    asciiish = re.sub(r"\s+", "_", asciiish).strip("._")[:60] or rendered.spec.layer_id
    return f"{index:02d}_{asciiish}"


def _layer_tree(
    rendered: list[RenderedLayer],
) -> tuple[list[RenderedLayer], dict[str, list[RenderedLayer]]]:
    layer_ids = [item.spec.layer_id for item in rendered]
    if len(layer_ids) != len(set(layer_ids)):
        duplicates = sorted({layer_id for layer_id in layer_ids if layer_ids.count(layer_id) > 1})
        raise RuntimeError(f"Duplicate V5 layer id(s): {', '.join(duplicates)}")
    by_id = {item.spec.layer_id: item for item in rendered}
    children: dict[str, list[RenderedLayer]] = defaultdict(list)
    roots: list[RenderedLayer] = []
    for item in rendered:
        parent_id = item.spec.metadata.get("parent_id")
        if parent_id and str(parent_id) in by_id:
            children[str(parent_id)].append(item)
        else:
            roots.append(item)

    # A cycle would otherwise silently produce no root and omit artwork from
    # every editable container. Unknown parents remain roots so no pixels are
    # lost if optional grouping metadata is incomplete.
    for item in rendered:
        chain: set[str] = set()
        current = item
        while True:
            current_id = current.spec.layer_id
            if current_id in chain:
                raise RuntimeError(f"Cyclic V5 layer hierarchy at: {current_id}")
            chain.add(current_id)
            parent_id = current.spec.metadata.get("parent_id")
            if not parent_id or str(parent_id) not in by_id:
                break
            current = by_id[str(parent_id)]
    return roots, children


def _user_group_layout(
    rendered: list[RenderedLayer],
    roots: list[RenderedLayer],
    children: dict[str, list[RenderedLayer]],
    user_groups: list[dict[str, object]] | None,
) -> tuple[dict[str, _UserGroupPlan], set[str]]:
    """Validate review groups without changing bottom-to-top pixel order."""

    by_id = {item.spec.layer_id: item for item in rendered}
    parent_by_id = {
        item.spec.layer_id: (
            str(item.spec.metadata.get("parent_id"))
            if item.spec.metadata.get("parent_id") in by_id
            else None
        )
        for item in rendered
    }
    first_member: dict[str, _UserGroupPlan] = {}
    grouped_members: set[str] = set()
    seen_group_ids: set[str] = set()
    for raw in user_groups or []:
        if not isinstance(raw, dict):
            raise RuntimeError("V5 review group record is not an object")
        group_id = str(raw.get("id") or "").strip()
        name = str(raw.get("name") or "").strip()
        member_ids = [str(value) for value in raw.get("member_ids", [])]
        if not group_id or group_id in seen_group_ids or not name:
            raise RuntimeError("V5 review group has an invalid/duplicate id or empty name")
        if len(member_ids) < 2 or len(member_ids) != len(set(member_ids)):
            raise RuntimeError(f"V5 review group {group_id} needs distinct members")
        if set(member_ids) - set(by_id):
            raise RuntimeError(f"V5 review group {group_id} references a missing layer")
        if grouped_members.intersection(member_ids):
            raise RuntimeError("A V5 layer may belong to only one user group")
        parents = {parent_by_id[member_id] for member_id in member_ids}
        if len(parents) != 1:
            raise RuntimeError(
                f"V5 review group {group_id} members must share one container parent"
            )
        parent_id = next(iter(parents))
        siblings = roots if parent_id is None else children.get(parent_id, [])
        positions = sorted(
            next(index for index, item in enumerate(siblings) if item.spec.layer_id == member_id)
            for member_id in member_ids
        )
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise RuntimeError(
                f"V5 review group {group_id} members are not contiguous in layer order"
            )
        ordered_members = [siblings[index] for index in positions]
        plan = _UserGroupPlan(group_id, name, ordered_members)
        first_member[ordered_members[0].spec.layer_id] = plan
        grouped_members.update(member_ids)
        seen_group_ids.add(group_id)
    return first_member, grouped_members


def _grouped_siblings(
    siblings: list[RenderedLayer],
    first_member: dict[str, _UserGroupPlan],
    grouped_members: set[str],
) -> list[RenderedLayer | _UserGroupPlan]:
    result: list[RenderedLayer | _UserGroupPlan] = []
    for item in siblings:
        identifier = item.spec.layer_id
        plan = first_member.get(identifier)
        if plan is not None:
            result.append(plan)
        elif identifier not in grouped_members:
            result.append(item)
    return result


def _expected_container_tree(
    background_size: tuple[int, int],
    rendered: list[RenderedLayer],
    user_groups: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Return the exact bottom-to-top tree both PSD and ORA must preserve."""

    width, height = background_size
    roots, children = _layer_tree(rendered)
    first_member, grouped_members = _user_group_layout(
        rendered, roots, children, user_groups
    )

    def pixel(name: str, item: RenderedLayer) -> dict[str, object]:
        return {
            "name": name,
            "kind": "pixel",
            "visible": True,
            "bbox": list(item.bbox),
        }

    def node(item: RenderedLayer) -> dict[str, object]:
        node_children = children.get(item.spec.layer_id, [])
        if not node_children:
            return pixel(item.spec.name, item)
        return {
            "name": f"GROUP - {item.spec.name}",
            "kind": "group",
            "visible": True,
            "children_bottom_to_top": [
                pixel(f"BASE - {item.spec.name}", item),
                *(entry(child) for child in _grouped_siblings(node_children, first_member, grouped_members)),
            ],
        }

    def entry(item: RenderedLayer | _UserGroupPlan) -> dict[str, object]:
        if isinstance(item, _UserGroupPlan):
            return {
                "name": f"USER GROUP - {item.name}",
                "kind": "group",
                "visible": True,
                "children_bottom_to_top": [node(member) for member in item.members],
            }
        return node(item)

    return [
        {
            "name": "00 BACKGROUND - SYNTHESIZED HIDDEN PIXELS",
            "kind": "pixel",
            "visible": True,
            "bbox": [0, 0, width, height],
        },
        *(
            entry(item)
            for item in _grouped_siblings(roots, first_member, grouped_members)
        ),
    ]


def _image_error(actual: Image.Image, expected: Image.Image) -> dict[str, float | int]:
    if actual.size != expected.size:
        raise RuntimeError(
            f"Round-trip canvas mismatch: got {actual.size}, expected {expected.size}"
        )
    actual_array = np.array(actual.convert("RGB"), dtype=np.int16)
    expected_array = np.array(expected.convert("RGB"), dtype=np.int16)
    error = np.abs(actual_array - expected_array)
    return {
        "max_abs_error": int(error.max()) if error.size else 0,
        "mean_abs_error": round(float(error.mean()), 8) if error.size else 0.0,
    }


def export_png_assets(
    output_dir: Path,
    background: Image.Image,
    rendered: list[RenderedLayer],
    *,
    icc_profile: bytes | None = None,
) -> list[dict[str, object]]:
    layers_dir = output_dir / "LAYERS"
    masks_dir = output_dir / "MASKS"
    layers_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    background_path = layers_dir / "00_BACKGROUND_SYNTHESIZED.png"
    save_color_png(background.convert("RGB"), background_path, icc_profile)
    records: list[dict[str, object]] = []
    for index, item in enumerate(rendered, 1):
        slug = _slug_layer(index, item)
        layer_path = layers_dir / f"{slug}.png"
        mask_path = masks_dir / f"{slug}_MASK.png"
        save_color_png(item.rgba.convert("RGBA"), layer_path, icc_profile)
        item.alpha.save(mask_path, format="PNG", compress_level=4)
        records.append(
            {
                "id": item.spec.layer_id,
                "name": item.spec.name,
                "category": item.spec.category,
                "label": item.spec.label,
                "ocr_text": item.spec.text,
                "score": round(item.spec.score, 6),
                "canvas_offset": [item.left, item.top],
                "canvas_bbox": list(item.bbox),
                "source_bbox": list(item.source_bbox),
                "rgba": str(layer_path.relative_to(output_dir)).replace("\\", "/"),
                "mask": str(mask_path.relative_to(output_dir)).replace("\\", "/"),
                "source_sam_mask_ids": item.spec.source_ids,
                "metadata": item.spec.metadata,
            }
        )
    return records


def export_psd(
    path: Path,
    background: Image.Image,
    rendered: list[RenderedLayer],
    expected_composite: Image.Image | None = None,
    *,
    icc_profile: bytes | None = None,
    user_groups: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, PixelLayer
    from psd_tools.constants import BlendMode, Compression, Resource
    from psd_tools.psd.image_resources import ImageResource

    width, height = background.size
    estimated_raw = width * height * 4 + sum(item.rgba.width * item.rgba.height * 4 for item in rendered)
    if path.suffix.lower() != ".psd":
        return {
            "created": False,
            "reason": "PSD adapter requires a .psd path; V5 never writes PSD data under a fake PSB extension",
        }
    if max(width, height) > PSD_MAX_DIMENSION:
        return {"created": False, "reason": "PSD limit: canvas dimension exceeds 30,000 px"}
    if estimated_raw > PSD_SAFE_RAW_BYTES:
        return {
            "created": False,
            "reason": "PSD safety estimate exceeds 1.6 GB; ORA and LAYERS.zip were kept",
            "estimated_raw_bytes": estimated_raw,
        }
    expected_tree = _expected_container_tree(background.size, rendered, user_groups)
    psd = PSDImage.new("RGBA", (width, height), color=(0, 0, 0, 0), depth=8)
    psd.background_color = None
    if icc_profile:
        psd.image_resources.pop(Resource.ICC_UNTAGGED_PROFILE, None)
        psd.image_resources[Resource.ICC_PROFILE] = ImageResource(
            key=Resource.ICC_PROFILE,
            name="",
            data=icc_profile,
        )
    background_layer = PixelLayer.frompil(
        background.convert("RGBA"),
        psd,
        name="Background",
        compression=Compression.RLE,
    )
    # PixelLayer.frompil(name=...) writes the legacy Pascal string directly.
    # Assigning .name writes the authoritative Unicode tagged block and a safe
    # MacRoman fallback, which is required for Vietnamese names.
    background_layer.name = "00 BACKGROUND - SYNTHESIZED HIDDEN PIXELS"
    roots, children = _layer_tree(rendered)
    first_member, grouped_members = _user_group_layout(
        rendered, roots, children, user_groups
    )

    def add_node(item: RenderedLayer, parent) -> None:
        node_children = children.get(item.spec.layer_id, [])
        if node_children:
            group = Group.new(parent, name="Group", open_folder=True)
            group.name = f"GROUP - {item.spec.name}"
            # The group is organizational, not an isolated compositing effect.
            # Pass-through preserves the exact bottom-to-top 8-bit artwork
            # rendering while retaining a useful hierarchy in Photoshop.
            group.blend_mode = BlendMode.PASS_THROUGH
            base = PixelLayer.frompil(
                item.rgba,
                group,
                name="Layer",
                top=item.top,
                left=item.left,
                compression=Compression.RLE,
            )
            base.name = f"BASE - {item.spec.name}"
            for child in _grouped_siblings(
                node_children, first_member, grouped_members
            ):
                add_entry(child, group)
        else:
            layer = PixelLayer.frompil(
                item.rgba,
                parent,
                name="Layer",
                top=item.top,
                left=item.left,
                compression=Compression.RLE,
            )
            layer.name = item.spec.name

    def add_entry(item: RenderedLayer | _UserGroupPlan, parent) -> None:
        if isinstance(item, _UserGroupPlan):
            group = Group.new(parent, name="User Group", open_folder=True)
            group.name = f"USER GROUP - {item.name}"
            group.blend_mode = BlendMode.PASS_THROUGH
            for member in item.members:
                add_node(member, group)
            return
        add_node(item, parent)

    for root_item in _grouped_siblings(roots, first_member, grouped_members):
        add_entry(root_item, psd)
    psd.save(path)
    reopened = PSDImage.open(path)
    if reopened.version != 1:
        raise RuntimeError(
            f"PSD round-trip unexpectedly produced header version {reopened.version}; PSB is not enabled"
        )
    embedded_icc = reopened.image_resources.get_data(Resource.ICC_PROFILE)
    if icc_profile and embedded_icc != icc_profile:
        raise RuntimeError("PSD sRGB ICC profile did not survive round-trip")
    if Resource.ICC_UNTAGGED_PROFILE in reopened.image_resources:
        raise RuntimeError("PSD was incorrectly marked as intentionally untagged")

    def describe(container) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for layer in container:
            if isinstance(layer, Group):
                result.append(
                    {
                        "name": layer.name,
                        "kind": "group",
                        "visible": bool(layer.visible),
                        "children_bottom_to_top": describe(layer),
                    }
                )
            else:
                result.append(
                    {
                        "name": layer.name,
                        "kind": "pixel",
                        "visible": bool(layer.visible),
                        "bbox": list(layer.bbox),
                    }
                )
        return result

    actual_tree = describe(reopened)
    if actual_tree != expected_tree:
        raise RuntimeError(
            "PSD hierarchy/order/name/offset round-trip mismatch: "
            + json.dumps({"expected": expected_tree, "actual": actual_tree}, ensure_ascii=False)
        )

    composite = reopened.composite(force=True, apply_icc=False)
    if composite is None:
        raise RuntimeError("PSD round-trip did not produce a merged composite")
    if expected_composite is None:
        expected_rgba = background.convert("RGBA")

        def flatten_node(item: RenderedLayer) -> None:
            expected_rgba.alpha_composite(item.rgba, dest=(item.left, item.top))
            for child in children.get(item.spec.layer_id, []):
                flatten_node(child)

        for root_item in roots:
            flatten_node(root_item)
        expected_composite = expected_rgba.convert("RGB")
    roundtrip_error = _image_error(composite, expected_composite)
    if roundtrip_error["max_abs_error"] > PSD_ROUNDTRIP_MAX_ERROR:
        raise RuntimeError(
            "PSD merged preview does not reproduce the V5 composite: "
            f"max error {roundtrip_error['max_abs_error']} > {PSD_ROUNDTRIP_MAX_ERROR}"
        )

    descendants = list(reopened.descendants())
    groups = [layer for layer in descendants if isinstance(layer, Group)]
    if any(group.blend_mode != BlendMode.PASS_THROUGH for group in groups):
        raise RuntimeError("PSD group pass-through mode did not survive round-trip")
    pixel_layers = [
        layer
        for layer in descendants
        if not isinstance(layer, Group)
    ]
    user_mask_count = sum(1 for layer in pixel_layers if layer.has_mask())
    if user_mask_count:
        raise RuntimeError(
            "PSD RGBA export unexpectedly converted transparency into user layer masks"
        )
    return {
        "created": True,
        "format": "PSD",
        "header_version": reopened.version,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "canvas": [width, height],
        "estimated_raw_bytes": estimated_raw,
        "top_level_layers_bottom_to_top": [layer.name for layer in reopened],
        "layer_tree_bottom_to_top": actual_tree,
        "pixel_layer_count": len(pixel_layers),
        "group_count": len(groups),
        "group_blend_mode": "pass_through",
        "user_mask_count": user_mask_count,
        "roundtrip_composite_size": list(composite.size),
        "roundtrip_qa": roundtrip_error,
        "icc_profile_sha256": hashlib.sha256(embedded_icc).hexdigest() if embedded_icc else None,
        "icc_profile_bytes": len(embedded_icc) if embedded_icc else 0,
        "format_notice": "All exported artwork layers are raster pixel layers; OCR names are metadata, not editable font objects.",
    }


def _ora_layer_element(item: RenderedLayer, source: str) -> ET.Element:
    return ET.Element(
        "layer",
        {
            "name": item.spec.name,
            "src": source,
            "x": str(item.left),
            "y": str(item.top),
            "visibility": "visible",
            "composite-op": "svg:src-over",
            "opacity": "1.0",
        },
    )


def export_ora(
    path: Path,
    background: Image.Image,
    rendered: list[RenderedLayer],
    composite: Image.Image,
    *,
    icc_profile: bytes | None = None,
    user_groups: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    width, height = background.size
    expected_tree = _expected_container_tree(background.size, rendered, user_groups)
    root = ET.Element(
        "image",
        {
            "version": ORA_VERSION,
            "w": str(width),
            "h": str(height),
            "name": path.stem,
            "xres": "96",
            "yres": "96",
        },
    )
    stack = ET.SubElement(root, "stack")
    stored: list[tuple[str, Image.Image]] = []
    source_by_id: dict[str, str] = {}
    for index, item in enumerate(rendered, 1):
        source = f"data/layer_{index:03d}.png"
        source_by_id[item.spec.layer_id] = source
        stored.append((source, item.rgba))
    roots, children = _layer_tree(rendered)
    first_member, grouped_members = _user_group_layout(
        rendered, roots, children, user_groups
    )

    def add_ora_node(parent_xml: ET.Element, item: RenderedLayer) -> None:
        node_children = children.get(item.spec.layer_id, [])
        if node_children:
            group = ET.SubElement(
                parent_xml,
                "stack",
                {"name": f"GROUP - {item.spec.name}", "isolation": "auto"},
            )
            # Topmost XML child first; base is always at the bottom of its group.
            grouped_children = _grouped_siblings(
                node_children, first_member, grouped_members
            )
            for child in reversed(grouped_children):
                add_ora_entry(group, child)
            base_element = _ora_layer_element(item, source_by_id[item.spec.layer_id])
            base_element.set("name", f"BASE - {item.spec.name}")
            group.append(base_element)
        else:
            parent_xml.append(_ora_layer_element(item, source_by_id[item.spec.layer_id]))

    def add_ora_entry(
        parent_xml: ET.Element,
        item: RenderedLayer | _UserGroupPlan,
    ) -> None:
        if isinstance(item, _UserGroupPlan):
            group = ET.SubElement(
                parent_xml,
                "stack",
                {"name": f"USER GROUP - {item.name}", "isolation": "auto"},
            )
            for member in reversed(item.members):
                add_ora_node(group, member)
            return
        add_ora_node(parent_xml, item)

    # OpenRaster stores the topmost root first.
    grouped_roots = _grouped_siblings(roots, first_member, grouped_members)
    for root_item in reversed(grouped_roots):
        add_ora_entry(stack, root_item)
    background_source = "data/background.png"
    ET.SubElement(
        stack,
        "layer",
        {
            "name": "00 BACKGROUND - SYNTHESIZED HIDDEN PIXELS",
            "src": background_source,
            "x": "0",
            "y": "0",
            "visibility": "visible",
            "composite-op": "svg:src-over",
            "opacity": "1.0",
        },
    )
    stack_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    thumbnail = composite.copy()
    thumbnail.thumbnail((256, 256), Image.Resampling.LANCZOS)
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        _write_zip_bytes(archive, "mimetype", ORA_MIMETYPE, zipfile.ZIP_STORED)
        _write_zip_bytes(archive, "stack.xml", stack_xml, zipfile.ZIP_DEFLATED, 9)
        _write_pil_to_zip(archive, "mergedimage.png", composite, icc_profile)
        _write_pil_to_zip(archive, "Thumbnails/thumbnail.png", thumbnail, icc_profile)
        _write_pil_to_zip(archive, background_source, background, icc_profile)
        for source, image in stored:
            _write_pil_to_zip(archive, source, image, icc_profile)
    validation = validate_ora(
        path,
        expected_tree=expected_tree,
        expected_composite=composite,
        expected_icc_profile=icc_profile,
    )
    return {
        "created": True,
        "format": "OpenRaster",
        "format_version": ORA_VERSION,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **validation,
    }


def _zip_info(name: str, compression: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _write_zip_bytes(
    archive: zipfile.ZipFile,
    name: str,
    data: bytes,
    compression: int,
    compresslevel: int | None = None,
) -> None:
    info = _zip_info(name, compression)
    if compression == zipfile.ZIP_DEFLATED:
        archive.writestr(info, data, compresslevel=compresslevel)
    else:
        archive.writestr(info, data)


def _write_pil_to_zip(
    archive: zipfile.ZipFile,
    name: str,
    image: Image.Image,
    icc_profile: bytes | None = None,
) -> None:

    buffer = io.BytesIO()
    save_color_png(image.convert("RGBA") if image.mode == "RGBA" else image.convert("RGB"), buffer, icc_profile)
    _write_zip_bytes(
        archive,
        name,
        buffer.getvalue(),
        zipfile.ZIP_DEFLATED,
        4,
    )


def _safe_zip_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    member = PurePosixPath(normalized)
    return bool(
        name
        and normalized == name
        and not member.is_absolute()
        and ".." not in member.parts
        and posixpath.normpath(normalized) == normalized
    )


def _multiply_alpha(image: Image.Image, opacity: float) -> Image.Image:
    rgba = image.convert("RGBA")
    if opacity == 1.0:
        return rgba
    alpha = rgba.getchannel("A").point(
        lambda value: max(0, min(255, round(value * opacity)))
    )
    rgba.putalpha(alpha)
    return rgba


def validate_ora(
    path: Path,
    *,
    expected_tree: list[dict[str, object]] | None = None,
    expected_composite: Image.Image | None = None,
    expected_icc_profile: bytes | None = None,
) -> dict[str, object]:
    """Validate the V5 OpenRaster baseline and independently recompose it."""

    with zipfile.ZipFile(path, "r", allowZip64=True) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if not infos or len(infos) > ORA_MAX_MEMBERS:
            raise RuntimeError("Invalid ORA ZIP member count.")
        if len(names) != len(set(names)):
            raise RuntimeError("Invalid ORA: duplicate ZIP member name.")
        if any(not _safe_zip_member(name) for name in names):
            raise RuntimeError("Invalid ORA: unsafe ZIP member path.")
        for info in infos:
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise RuntimeError("Invalid ORA: unsupported ZIP compression method.")
            unix_type = (info.external_attr >> 16) & 0o170000
            if unix_type == 0o120000:
                raise RuntimeError("Invalid ORA: symbolic-link ZIP member.")

        required = {
            "mimetype",
            "stack.xml",
            "mergedimage.png",
            "Thumbnails/thumbnail.png",
        }
        if not required.issubset(names):
            raise RuntimeError("Invalid ORA: required baseline member is missing.")
        if infos[0].filename != "mimetype" or infos[0].compress_type != zipfile.ZIP_STORED:
            raise RuntimeError("Invalid ORA: mimetype must be the first uncompressed member.")
        if archive.read("mimetype") != ORA_MIMETYPE:
            raise RuntimeError("Invalid ORA mimetype value.")
        if archive.getinfo("stack.xml").file_size > ORA_XML_MAX_BYTES:
            raise RuntimeError("Invalid ORA: stack.xml is too large.")

        xml_bytes = archive.read("stack.xml")
        upper_xml = xml_bytes.upper()
        if b"<!DOCTYPE" in upper_xml or b"<!ENTITY" in upper_xml:
            raise RuntimeError("Invalid ORA: DTD and entity declarations are forbidden.")
        try:
            tree = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise RuntimeError(f"Invalid ORA stack.xml: {exc}") from exc
        if tree.tag != "image" or tree.attrib.get("version") != ORA_VERSION:
            raise RuntimeError(f"Invalid ORA: expected image version {ORA_VERSION}.")
        allowed_image_attributes = {"version", "w", "h", "name", "xres", "yres"}
        if set(tree.attrib) - allowed_image_attributes:
            raise RuntimeError("Invalid ORA: unknown image attribute.")
        try:
            width, height = int(tree.attrib["w"]), int(tree.attrib["h"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError("Invalid ORA canvas dimensions.") from exc
        # Some third-party image adapters temporarily disable Pillow's global
        # decompression-bomb limit by setting it to None. ORA validation must
        # remain fail-closed and independent of that mutable process global.
        if width <= 0 or height <= 0 or width * height > V5_MAX_IMAGE_PIXELS:
            raise RuntimeError("Invalid ORA canvas dimensions.")
        has_xres, has_yres = "xres" in tree.attrib, "yres" in tree.attrib
        if has_xres != has_yres:
            raise RuntimeError("Invalid ORA: xres and yres must appear together.")
        if has_xres:
            try:
                if int(tree.attrib["xres"]) < 1 or int(tree.attrib["yres"]) < 1:
                    raise ValueError
            except ValueError as exc:
                raise RuntimeError("Invalid ORA resolution.") from exc
        root_children = list(tree)
        if len(root_children) != 1 or root_children[0].tag != "stack":
            raise RuntimeError("Invalid ORA: image must contain exactly one root stack.")
        root_stack = root_children[0]
        if root_stack.attrib:
            raise RuntimeError("Invalid ORA: root stack must omit optional attributes.")

        png_cache: dict[str, Image.Image] = {}
        png_modes: dict[str, str] = {}
        png_icc_sha256: dict[str, str | None] = {}

        def load_png(source: str) -> Image.Image:
            if source in png_cache:
                return png_cache[source].copy()
            if source not in names:
                raise RuntimeError(f"Invalid ORA layer source: {source}")
            data = archive.read(source)
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError(f"Invalid ORA PNG member: {source}")
            try:
                with Image.open(io.BytesIO(data)) as image:
                    if image.format != "PNG" or image.info.get("interlace", 0) not in {0, None}:
                        raise RuntimeError(f"Invalid ORA PNG encoding: {source}")
                    image.load()
                    if image.mode not in {"RGB", "RGBA"}:
                        raise RuntimeError(
                            f"Unsupported V5 ORA PNG mode {image.mode!r}: {source}"
                        )
                    png_modes[source] = image.mode
                    embedded_icc = image.info.get("icc_profile")
                    if expected_icc_profile is not None and embedded_icc != expected_icc_profile:
                        raise RuntimeError(f"ORA PNG member lost the sRGB ICC profile: {source}")
                    png_icc_sha256[source] = (
                        hashlib.sha256(embedded_icc).hexdigest() if embedded_icc else None
                    )
                    decoded = image.convert("RGBA")
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Invalid ORA PNG member: {source}") from exc
            png_cache[source] = decoded
            return decoded.copy()

        referenced_sources: set[str] = set()

        def common(element: ET.Element, *, is_stack: bool) -> tuple[str, float, bool]:
            allowed = {"name", "opacity", "visibility", "composite-op"}
            if is_stack:
                allowed.add("isolation")
            else:
                allowed.update({"src", "x", "y"})
            if set(element.attrib) - allowed:
                raise RuntimeError(f"Invalid ORA attributes on {element.tag}.")
            try:
                opacity = float(element.attrib.get("opacity", "1"))
            except ValueError as exc:
                raise RuntimeError("Invalid ORA opacity.") from exc
            if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
                raise RuntimeError("Invalid ORA opacity.")
            visibility = element.attrib.get("visibility", "visible")
            if visibility not in {"visible", "hidden"}:
                raise RuntimeError("Invalid ORA visibility.")
            if element.attrib.get("composite-op", "svg:src-over") != "svg:src-over":
                raise RuntimeError("V5 ORA validator only accepts svg:src-over.")
            return element.attrib.get("name", ""), opacity, visibility == "visible"

        def render_stack(
            element: ET.Element,
            *,
            backdrop: Image.Image | None = None,
        ) -> tuple[Image.Image, list[dict[str, object]]]:
            canvas = (
                backdrop.copy()
                if backdrop is not None
                else Image.new("RGBA", (width, height), (0, 0, 0, 0))
            )
            summaries: list[dict[str, object]] = []
            xml_children = list(element)
            if not xml_children:
                raise RuntimeError("Invalid ORA: empty stack.")
            # stack.xml lists topmost first; compositing and manifest order are
            # bottom-to-top.
            for child in reversed(xml_children):
                if child.tag == "layer":
                    name, opacity, visible = common(child, is_stack=False)
                    source = child.attrib.get("src")
                    if not source or not _safe_zip_member(source) or not source.startswith("data/"):
                        raise RuntimeError(f"Invalid ORA layer source: {source}")
                    if Path(source).suffix != ".png":
                        raise RuntimeError("V5 ORA layers must be lowercase PNG files.")
                    try:
                        x = int(child.attrib.get("x", "0"))
                        y = int(child.attrib.get("y", "0"))
                    except ValueError as exc:
                        raise RuntimeError("Invalid ORA layer offset.") from exc
                    layer_image = load_png(source)
                    referenced_sources.add(source)
                    if visible:
                        canvas.alpha_composite(_multiply_alpha(layer_image, opacity), dest=(x, y))
                    summaries.append(
                        {
                            "name": name,
                            "kind": "pixel",
                            "visible": visible,
                            "bbox": [x, y, x + layer_image.width, y + layer_image.height],
                        }
                    )
                elif child.tag == "stack":
                    name, opacity, visible = common(child, is_stack=True)
                    if "x" in child.attrib or "y" in child.attrib:
                        raise RuntimeError("OpenRaster 0.0.6 forbids stack x/y offsets.")
                    isolation = child.attrib.get("isolation", "isolate")
                    if isolation not in {"auto", "isolate"}:
                        raise RuntimeError("Invalid ORA stack isolation.")
                    if isolation == "auto":
                        # V5 groups are organizational pass-through folders.
                        # Rendering directly into the existing backdrop avoids
                        # an extra alpha-quantization boundary.
                        if opacity != 1.0:
                            raise RuntimeError(
                                "V5 ORA pass-through groups require opacity 1.0."
                            )
                        if visible:
                            canvas, group_children = render_stack(child, backdrop=canvas)
                        else:
                            _, group_children = render_stack(child)
                    else:
                        group_image, group_children = render_stack(child)
                        if visible:
                            canvas.alpha_composite(_multiply_alpha(group_image, opacity))
                    summaries.append(
                        {
                            "name": name,
                            "kind": "group",
                            "visible": visible,
                            "children_bottom_to_top": group_children,
                        }
                    )
                else:
                    raise RuntimeError(f"Invalid ORA stack child: {child.tag}")
            return canvas, summaries

        recomposed, actual_tree = render_stack(root_stack)
        if expected_tree is not None and actual_tree != expected_tree:
            raise RuntimeError(
                "ORA hierarchy/order/name/offset round-trip mismatch: "
                + json.dumps({"expected": expected_tree, "actual": actual_tree}, ensure_ascii=False)
            )
        data_members = {
            name for name in names if name.startswith("data/") and not name.endswith("/")
        }
        if data_members != referenced_sources:
            raise RuntimeError("Invalid ORA: data/ members and stack.xml references differ.")

        merged = load_png("mergedimage.png")
        if merged.size != (width, height):
            raise RuntimeError("Invalid ORA: mergedimage.png has the wrong canvas size.")
        recomposition_error = _image_error(recomposed, merged)
        if recomposition_error["max_abs_error"] > ORA_ROUNDTRIP_MAX_ERROR:
            raise RuntimeError(
                "ORA mergedimage.png does not match the layer stack: "
                f"max error {recomposition_error['max_abs_error']} > {ORA_ROUNDTRIP_MAX_ERROR}"
            )
        expected_error = None
        if expected_composite is not None:
            expected_error = _image_error(recomposed, expected_composite)
            if expected_error["max_abs_error"] > ORA_ROUNDTRIP_MAX_ERROR:
                raise RuntimeError(
                    "ORA layer stack does not reproduce the V5 composite: "
                    f"max error {expected_error['max_abs_error']} > {ORA_ROUNDTRIP_MAX_ERROR}"
                )

        thumbnail = load_png("Thumbnails/thumbnail.png")
        if (
            thumbnail.width > 256
            or thumbnail.height > 256
            or thumbnail.width > width
            or thumbnail.height > height
        ):
            raise RuntimeError("Invalid ORA thumbnail dimensions.")
        return {
            "validated": True,
            "canvas": [width, height],
            "zip64_allowed": True,
            "zip64_used": any(info.extract_version >= 45 for info in infos),
            "mimetype_first_uncompressed": True,
            "member_count": len(infos),
            "referenced_layer_png_count": len(referenced_sources),
            "layer_tree_bottom_to_top": actual_tree,
            "recomposition_qa": recomposition_error,
            "expected_composite_qa": expected_error,
            "png_modes": png_modes,
            "png_icc_sha256": png_icc_sha256,
        }


def _contact_sheet_checker() -> Image.Image:
    checker = Image.new(
        "RGB",
        (CONTACT_SHEET_CARD_WIDTH, CONTACT_SHEET_CARD_HEIGHT),
        (220, 220, 220),
    )
    draw = ImageDraw.Draw(checker)
    for cy in range(0, CONTACT_SHEET_CARD_HEIGHT, 20):
        for cx in range(0, CONTACT_SHEET_CARD_WIDTH, 20):
            if (cx // 20 + cy // 20) % 2:
                draw.rectangle((cx, cy, cx + 19, cy + 19), fill=(180, 180, 180))
    return checker


def _render_contact_sheet_grid(
    cards: list[tuple[str, Image.Image]],
    *,
    heading: tuple[str, str] | None = None,
) -> Image.Image:
    if not cards:
        raise ValueError("A V5 contact sheet needs at least one card.")
    header_h = 72 if heading else 0
    rows = math.ceil(len(cards) / CONTACT_SHEET_COLUMNS)
    width = CONTACT_SHEET_COLUMNS * CONTACT_SHEET_CARD_WIDTH
    height = header_h + rows * (CONTACT_SHEET_CARD_HEIGHT + CONTACT_SHEET_LABEL_HEIGHT)
    if (
        width > CONTACT_SHEET_MAX_DIMENSION
        or height > CONTACT_SHEET_MAX_DIMENSION
        or width * height > CONTACT_SHEET_MAX_PIXELS
    ):
        raise RuntimeError(
            f"V5 contact sheet page {width}x{height} exceeds its bounded image policy."
        )
    sheet = Image.new("RGB", (width, height), (38, 38, 42))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    if heading:
        draw.text((12, 10), heading[0], fill="white", font=font)
        draw.text((12, 38), heading[1], fill=(205, 205, 210), font=font)
    checker_template = _contact_sheet_checker()
    for index, (name, artwork) in enumerate(cards):
        column, row = index % CONTACT_SHEET_COLUMNS, index // CONTACT_SHEET_COLUMNS
        x = column * CONTACT_SHEET_CARD_WIDTH
        y = header_h + row * (CONTACT_SHEET_CARD_HEIGHT + CONTACT_SHEET_LABEL_HEIGHT)
        checker = checker_template.copy()
        thumbnail = artwork.copy()
        thumbnail.thumbnail(
            (CONTACT_SHEET_CARD_WIDTH - 16, CONTACT_SHEET_CARD_HEIGHT - 16),
            Image.Resampling.LANCZOS,
        )
        checker.paste(
            thumbnail,
            (
                (CONTACT_SHEET_CARD_WIDTH - thumbnail.width) // 2,
                (CONTACT_SHEET_CARD_HEIGHT - thumbnail.height) // 2,
            ),
            thumbnail if thumbnail.mode == "RGBA" else None,
        )
        sheet.paste(checker, (x, y))
        draw.text(
            (x + 8, y + CONTACT_SHEET_CARD_HEIGHT + 8),
            name[:34],
            fill="white",
            font=font,
        )
    return sheet


def _contact_sheet_overview_cards(
    cards: list[tuple[str, Image.Image]],
) -> list[tuple[str, Image.Image]]:
    if len(cards) <= CONTACT_SHEET_OVERVIEW_CARDS:
        indices = list(range(len(cards)))
    else:
        last = len(cards) - 1
        indices = [
            round(index * last / (CONTACT_SHEET_OVERVIEW_CARDS - 1))
            for index in range(CONTACT_SHEET_OVERVIEW_CARDS)
        ]
    digits = max(3, len(str(len(cards))))
    return [
        (
            f"{index + 1:0{digits}d}/{len(cards):0{digits}d} {cards[index][0]}",
            cards[index][1],
        )
        for index in indices
    ]


def _clear_contact_sheet_pages(directory: Path) -> None:
    if not directory.is_dir():
        return
    for page in directory.glob("page_*.png"):
        if page.is_file():
            page.unlink()
    try:
        directory.rmdir()
    except OSError:
        # Preserve any unrelated user file rather than recursively deleting it.
        pass


def export_contact_sheet(
    path: Path,
    background: Image.Image,
    rendered: list[RenderedLayer],
    *,
    icc_profile: bytes | None = None,
) -> None:
    """Write one legacy-sized sheet for small jobs or a bounded cover plus pages."""

    cards: list[tuple[str, Image.Image]] = [("BACKGROUND", background.convert("RGBA"))]
    cards.extend((item.spec.name, item.rgba) for item in rendered)
    pages_dir = path.parent / "CONTACT_SHEETS"
    _clear_contact_sheet_pages(pages_dir)
    if len(cards) <= CONTACT_SHEET_PAGE_CAPACITY:
        save_color_png(_render_contact_sheet_grid(cards), path, icc_profile)
        return

    page_count = math.ceil(len(cards) / CONTACT_SHEET_PAGE_CAPACITY)
    page_digits = max(3, len(str(page_count)))
    pages_dir.mkdir(parents=True, exist_ok=True)
    for page_index, start in enumerate(
        range(0, len(cards), CONTACT_SHEET_PAGE_CAPACITY), 1
    ):
        stop = min(start + CONTACT_SHEET_PAGE_CAPACITY, len(cards))
        page = _render_contact_sheet_grid(
            cards[start:stop],
            heading=(
                f"V5 CONTACT SHEET - PAGE {page_index:0{page_digits}d}/{page_count:0{page_digits}d}",
                f"CARDS {start + 1}-{stop} OF {len(cards)}",
            ),
        )
        save_color_png(
            page,
            pages_dir / f"page_{page_index:0{page_digits}d}.png",
            icc_profile,
        )
    overview = _render_contact_sheet_grid(
        _contact_sheet_overview_cards(cards),
        heading=(
            "V5 CONTACT SHEET - BOUNDED OVERVIEW",
            f"{len(cards)} CARDS; DETAILS: CONTACT_SHEETS/"
            f"page_{1:0{page_digits}d}.png ... page_{page_count:0{page_digits}d}.png",
        ),
    )
    save_color_png(overview, path, icc_profile)


def package_layers_zip(path: Path, output_dir: Path, included: list[Path]) -> dict[str, object]:
    root = output_dir.resolve()
    members: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in included:
        resolved = item.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"V5 ZIP member escapes the bundle: {item}") from exc
        name = str(relative).replace("\\", "/")
        if not resolved.is_file() or not name or name.startswith("../"):
            raise RuntimeError(f"Invalid V5 ZIP member: {item}")
        if name in seen:
            raise RuntimeError(f"Duplicate V5 ZIP member: {name}")
        seen.add(name)
        members.append((name, resolved))
    members.sort(key=lambda value: value[0])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, item in members:
            archive.write(item, name)
    return {
        "created": True,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "member_count": len(members),
        "members": [name for name, _item in members],
    }


def asset_records(output_dir: Path, *, exclude: set[Path] | None = None) -> list[dict[str, object]]:
    excluded = {path.resolve() for path in (exclude or set())}
    records = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.resolve() in excluded:
            continue
        records.append(
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records
