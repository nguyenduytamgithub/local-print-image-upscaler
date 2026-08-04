"""Deterministic background reconstruction for V7 text replacement.

The functions in this module are deliberately conservative.  They only alter a
declared removal footprint and preserve every pixel outside that footprint
byte-for-byte.  Smooth poster backgrounds are reconstructed with a robust
polynomial surface.  Textured regions may opt into OpenCV Telea inpainting, but
the report always identifies that result as synthesized rather than recovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


class BackgroundRestoreError(ValueError):
    """Raised when an image or mask cannot be processed safely."""


@dataclass(slots=True)
class BackgroundRestoreResult:
    """One background reconstruction and its auditable footprint."""

    background: np.ndarray
    removal_footprint: np.ndarray
    method: str
    confidence: float
    report: dict[str, object]


@dataclass(slots=True)
class _SurfaceCandidate:
    name: str
    degree: int
    coefficients: np.ndarray
    validation_mae: float
    score: float
    retained_samples: int


def _validate_rgb(image_rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(image_rgb)
    if value.ndim != 3 or value.shape[2] != 3:
        raise BackgroundRestoreError("image_rgb must have shape HxWx3.")
    if value.dtype != np.uint8:
        raise BackgroundRestoreError("image_rgb must use uint8 RGB pixels.")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise BackgroundRestoreError("image_rgb cannot be empty.")
    return value


def _normalise_masks(
    masks: np.ndarray | Sequence[np.ndarray] | Iterable[np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    if isinstance(masks, np.ndarray):
        items = [masks]
    else:
        items = list(masks)
    if not items:
        return np.zeros(shape, dtype=bool)
    union = np.zeros(shape, dtype=bool)
    for item in items:
        value = np.asarray(item)
        if value.shape != shape:
            raise BackgroundRestoreError(
                f"mask shape {value.shape} does not match image shape {shape}."
            )
        if value.ndim != 2:
            raise BackgroundRestoreError("each removal mask must be two-dimensional.")
        union |= value.astype(bool, copy=False)
    return union


def _automatic_padding(mask: np.ndarray) -> int:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    heights = [int(stats[index, cv2.CC_STAT_HEIGHT]) for index in range(1, count)]
    if not heights:
        return 0
    typical_height = float(np.median(heights))
    return int(np.clip(round(typical_height * 0.10), 2, 18))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _surface_features(
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
    degree: int,
) -> np.ndarray:
    # Coordinates are normalized around the image centre.  This keeps the
    # least-squares system stable even for large print images.
    x = (xs.astype(np.float64) + 0.5) / max(1, width) * 2.0 - 1.0
    y = (ys.astype(np.float64) + 0.5) / max(1, height) * 2.0 - 1.0
    columns = [np.ones_like(x)]
    if degree >= 1:
        columns.extend((x, y))
    if degree >= 2:
        columns.extend((x * y, x * x, y * y))
    return np.column_stack(columns)


def _deterministic_sample(
    ys: np.ndarray,
    xs: np.ndarray,
    values: np.ndarray,
    maximum: int = 24_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(xs) <= maximum:
        return ys, xs, values
    # Even spacing is reproducible and retains samples across the whole ring.
    indices = np.linspace(0, len(xs) - 1, maximum, dtype=np.int64)
    return ys[indices], xs[indices], values[indices]


def _dominant_surface_context(
    ys: np.ndarray,
    xs: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Reject nearby logos/ink when one poster surface dominates the ring."""

    if len(xs) < 64:
        return ys, xs, colors, {
            "dominant_context_used": False,
            "dominant_context_ratio": 1.0,
        }
    lab = cv2.cvtColor(
        colors.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)
    quantized = np.floor(
        lab / np.array((8.0, 12.0, 12.0), dtype=np.float32)
    ).astype(np.int16)
    _, inverse, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
    best: tuple[float, np.ndarray] | None = None
    total_span_x = max(1, int(xs.max()) - int(xs.min()) + 1)
    total_span_y = max(1, int(ys.max()) - int(ys.min()) + 1)
    for index in np.argsort(-counts)[: min(10, len(counts))]:
        members = inverse == index
        centre = np.median(lab[members], axis=0)
        seed_distance = np.linalg.norm(lab[members] - centre, axis=1)
        radius = float(np.clip(np.percentile(seed_distance, 95) * 2.0 + 3.0, 6.0, 24.0))
        selected = np.linalg.norm(lab - centre, axis=1) <= radius
        count = int(selected.sum())
        if count < 24:
            continue
        sx, sy = xs[selected], ys[selected]
        span = 0.5 * (
            (int(sx.max()) - int(sx.min()) + 1) / total_span_x
            + (int(sy.max()) - int(sy.min()) + 1) / total_span_y
        )
        dispersion = float(np.median(np.linalg.norm(lab[selected] - centre, axis=1)))
        score = count * (0.6 + 0.4 * span) / (1.0 + dispersion)
        if best is None or score > best[0]:
            best = score, selected
    if best is None:
        return ys, xs, colors, {
            "dominant_context_used": False,
            "dominant_context_ratio": 1.0,
        }
    selected = best[1]
    ratio = float(selected.mean())
    # A minority colour may be a panel, icon or text rather than the hidden
    # surface. Only trust a cluster that accounts for a substantial ring share.
    if ratio < 0.34:
        return ys, xs, colors, {
            "dominant_context_used": False,
            "dominant_context_ratio": round(ratio, 6),
        }
    return ys[selected], xs[selected], colors[selected], {
        "dominant_context_used": True,
        "dominant_context_ratio": round(ratio, 6),
        "dominant_context_samples": int(selected.sum()),
    }


def _fit_candidate(
    name: str,
    degree: int,
    xs: np.ndarray,
    ys: np.ndarray,
    colors: np.ndarray,
    width: int,
    height: int,
) -> _SurfaceCandidate | None:
    design = _surface_features(xs, ys, width, height, degree)
    required = design.shape[1] * 4
    if len(xs) < required:
        return None

    # Spatial checkerboard validation avoids rewarding a model that merely
    # memorizes one side of a text box.
    validation = ((xs // 7 + ys // 7) % 5) == 0
    if int(validation.sum()) < design.shape[1] * 2:
        validation = (np.arange(len(xs)) % 5) == 0
    training = ~validation
    if int(training.sum()) < required or not validation.any():
        return None

    train_design = design[training]
    train_colors = colors[training].astype(np.float64)
    try:
        coefficients, *_ = np.linalg.lstsq(train_design, train_colors, rcond=None)
    except np.linalg.LinAlgError:
        return None

    # One robust refit removes nearby letters/icons that leaked into the ring.
    residual = np.linalg.norm(train_design @ coefficients - train_colors, axis=1)
    cutoff = max(4.0, float(np.percentile(residual, 82.0)))
    retained = residual <= cutoff
    if int(retained.sum()) >= required:
        try:
            coefficients, *_ = np.linalg.lstsq(
                train_design[retained], train_colors[retained], rcond=None
            )
        except np.linalg.LinAlgError:
            return None

    predicted = design[validation] @ coefficients
    mae = float(np.mean(np.abs(predicted - colors[validation])))
    # Prefer the simpler surface unless additional terms visibly help.
    complexity_penalty = {0: 0.0, 1: 0.20, 2: 0.65}[degree]
    return _SurfaceCandidate(
        name=name,
        degree=degree,
        coefficients=coefficients,
        validation_mae=mae,
        score=mae + complexity_penalty,
        retained_samples=int(retained.sum()),
    )


def _surface_restore(
    image_rgb: np.ndarray,
    footprint: np.ndarray,
    ring_radius: int,
    excluded_context: np.ndarray | None = None,
) -> tuple[np.ndarray | None, _SurfaceCandidate | None, dict[str, object]]:
    height, width = footprint.shape
    excluded = footprint if excluded_context is None else excluded_context
    ring = _dilate(footprint, ring_radius) & ~excluded
    ys, xs = np.where(ring)
    if len(xs) < 24:
        return None, None, {"ring_samples": int(len(xs)), "reason": "insufficient_context"}

    colors = image_rgb[ys, xs]
    ys, xs, colors = _deterministic_sample(ys, xs, colors)
    ys, xs, colors, context_report = _dominant_surface_context(ys, xs, colors)
    candidates = [
        candidate
        for candidate in (
            _fit_candidate("constant", 0, xs, ys, colors, width, height),
            _fit_candidate("linear", 1, xs, ys, colors, width, height),
            _fit_candidate("quadratic", 2, xs, ys, colors, width, height),
        )
        if candidate is not None
    ]
    if not candidates:
        return None, None, {"ring_samples": int(len(xs)), "reason": "surface_fit_failed"}
    selected = min(candidates, key=lambda item: (item.score, item.degree))

    target_y, target_x = np.where(footprint)
    target_design = _surface_features(
        target_x, target_y, width, height, selected.degree
    )
    predicted = np.clip(
        np.rint(target_design @ selected.coefficients), 0, 255
    ).astype(np.uint8)
    return predicted, selected, {
        "ring_samples": int(len(xs)),
        **context_report,
        "surface_candidates": [
            {
                "name": item.name,
                "validation_mae": round(item.validation_mae, 5),
                "score": round(item.score, 5),
                "retained_samples": item.retained_samples,
            }
            for item in candidates
        ],
    }


def restore_text_background(
    image_rgb: np.ndarray,
    masks: np.ndarray | Sequence[np.ndarray] | Iterable[np.ndarray],
    *,
    mode: str = "auto",
    padding: int | None = None,
    ring_radius: int | None = None,
    max_surface_mae: float = 18.0,
) -> BackgroundRestoreResult:
    """Remove text/graphics from a poster while preserving untouched pixels.

    ``mode`` accepts ``auto``, ``poster``/``surface``, ``opencv`` or ``strict``.
    ``strict`` refuses to synthesize when a smooth surface cannot be validated.
    OpenCV inpainting is deterministic here but is still an inferred texture;
    callers should retain the returned report in the V7 manifest.
    """

    source = _validate_rgb(image_rgb)
    if mode not in {"auto", "poster", "surface", "opencv", "strict"}:
        raise BackgroundRestoreError(
            "mode must be auto, poster, surface, opencv or strict."
        )
    union = _normalise_masks(masks, source.shape[:2])
    if not union.any():
        unchanged = source.copy()
        return BackgroundRestoreResult(
            unchanged,
            union,
            "unchanged_empty_mask",
            1.0,
            {
                "requested_mode": mode,
                "synthesized": False,
                "recovered_original_pixels_claimed": False,
                "outside_footprint_byte_identical": True,
                "removal_pixels": 0,
            },
        )

    chosen_padding = _automatic_padding(union) if padding is None else int(padding)
    if not 0 <= chosen_padding <= 128:
        raise BackgroundRestoreError("padding must be from 0 through 128 pixels.")
    footprint = _dilate(union, chosen_padding)
    chosen_ring = (
        int(np.clip(chosen_padding * 3 + 8, 12, 96))
        if ring_radius is None
        else int(ring_radius)
    )
    if not 2 <= chosen_ring <= 256:
        raise BackgroundRestoreError("ring_radius must be from 2 through 256 pixels.")
    threshold = float(max_surface_mae)
    if not np.isfinite(threshold) or threshold < 0:
        raise BackgroundRestoreError("max_surface_mae must be finite and non-negative.")

    # Fit each disconnected removal region against its own neighbourhood.
    # A single global colour surface is incorrect for posters containing, for
    # example, white text on a green panel and red text on a cream panel.
    component_count, labels = cv2.connectedComponents(
        footprint.astype(np.uint8), connectivity=8
    )
    result = source.copy()
    inferred: np.ndarray | None = None
    component_reports: list[dict[str, object]] = []
    methods: list[str] = []
    weighted_confidence = 0.0
    accepted_values: list[bool] = []
    validation_values: list[float] = []
    synthesized_any = False

    for component_index in range(1, component_count):
        component = labels == component_index
        component_pixels = int(component.sum())
        predicted, candidate, diagnostics = _surface_restore(
            source,
            component,
            chosen_ring,
            excluded_context=footprint,
        )
        accepted = (
            predicted is not None
            and candidate is not None
            and candidate.validation_mae <= threshold
        )
        accepted_values.append(bool(accepted))
        if candidate is not None:
            validation_values.append(float(candidate.validation_mae))

        # Explicit OpenCV mode must be honoured even on a smooth background.
        # Other modes prefer the validated deterministic surface where allowed.
        if mode == "opencv":
            component_method = "opencv_telea_synthesized"
            component_confidence = 0.35
        elif mode in {"poster", "surface"} and predicted is not None:
            component_method = (
                f"poster_surface_{candidate.name}" if candidate else "poster_surface"
            )
            component_confidence = (
                max(0.05, 1.0 - float(candidate.validation_mae) / 40.0)
                if candidate
                else 0.1
            )
            result[component] = predicted
        elif accepted:
            assert predicted is not None and candidate is not None
            component_method = f"validated_surface_{candidate.name}"
            component_confidence = max(0.1, 1.0 - candidate.validation_mae / 30.0)
            result[component] = predicted
        elif mode == "strict":
            component_method = "unchanged_unvalidated_context"
            component_confidence = 0.0
        else:
            component_method = "opencv_telea_synthesized"
            component_confidence = 0.25

        if component_method == "opencv_telea_synthesized":
            if inferred is None:
                inpaint_mask = footprint.astype(np.uint8) * 255
                radius = float(np.clip(max(2, chosen_padding), 2, 12))
                inferred = cv2.inpaint(source, inpaint_mask, radius, cv2.INPAINT_TELEA)
            result[component] = inferred[component]

        synthesized = component_method != "unchanged_unvalidated_context"
        synthesized_any |= synthesized
        methods.append(component_method)
        weighted_confidence += component_confidence * component_pixels
        component_reports.append(
            {
                "component": component_index,
                "pixels": component_pixels,
                "method": component_method,
                "confidence": round(float(component_confidence), 6),
                "surface_auto_accepted": bool(accepted),
                "surface_validation_mae": (
                    round(float(candidate.validation_mae), 5)
                    if candidate is not None
                    else None
                ),
                **diagnostics,
            }
        )

    unique_methods = set(methods)
    if len(unique_methods) == 1:
        method = methods[0]
    elif all(item.startswith("validated_surface_") for item in methods):
        method = "componentwise_validated_surface"
    elif all(item.startswith("poster_surface_") for item in methods):
        method = "componentwise_poster_surface"
    else:
        method = "componentwise_mixed"
    confidence = weighted_confidence / max(1, int(footprint.sum()))

    # A binary replacement edge is visible even when the fitted colour is only
    # a few values away from the real poster. The source mask already has
    # padding around the glyph core, so feather only that outer padding while
    # keeping the old ink interior fully synthesized.
    feather_width = min(5, max(0, chosen_padding))
    if synthesized_any and feather_width > 0:
        distance = cv2.distanceTransform(footprint.astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.clip(distance / float(feather_width), 0.0, 1.0)[..., None]
        blended = np.rint(
            source.astype(np.float32) * (1.0 - alpha)
            + result.astype(np.float32) * alpha
        ).astype(np.uint8)
        result[footprint] = blended[footprint]

    # This invariant is central to V7: replacement cannot silently alter good
    # artwork around the declared text region.
    result[~footprint] = source[~footprint]
    outside_identical = bool(np.array_equal(result[~footprint], source[~footprint]))
    if not outside_identical:  # Defensive; should be unreachable.
        raise RuntimeError("background reconstruction changed pixels outside its footprint")

    validation_mae = max(validation_values) if validation_values else None
    report: dict[str, object] = {
        "requested_mode": mode,
        "method": method,
        "synthesized": synthesized_any,
        "recovered_original_pixels_claimed": False,
        "padding_source_px": chosen_padding,
        "ring_radius_source_px": chosen_ring,
        "feather_width_source_px": feather_width,
        "source_mask_pixels": int(union.sum()),
        "removal_pixels": int(footprint.sum()),
        "removal_ratio": round(float(footprint.mean()), 8),
        "component_count": component_count - 1,
        "surface_auto_accepted": bool(accepted_values and all(accepted_values)),
        "surface_auto_accepted_components": int(sum(accepted_values)),
        "surface_validation_mae": (
            round(float(validation_mae), 5) if validation_mae is not None else None
        ),
        "max_surface_mae": threshold,
        "outside_footprint_byte_identical": outside_identical,
        "deterministic": True,
        "components": component_reports,
    }
    return BackgroundRestoreResult(
        background=result,
        removal_footprint=footprint,
        method=method,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        report=report,
    )


# A shorter alias is convenient for the V7 engine and mirrors V5's naming.
restore_background = restore_text_background


__all__ = [
    "BackgroundRestoreError",
    "BackgroundRestoreResult",
    "restore_background",
    "restore_text_background",
]
