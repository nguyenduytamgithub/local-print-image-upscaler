"""Deterministic quality gates for V7 poster text replacement.

The module deliberately has no model dependency.  It uses masks and provenance
already produced by the replacement engine to detect the failures that must not
reach a print deliverable: edits outside the declared ROI, remnants of the old
glyphs, blend/tile seams, clipped new glyphs, incorrect Unicode text, and
incomplete determinism metadata.

All numeric values returned by :func:`evaluate_replacement` are plain Python
objects and are therefore JSON serialisable.  The diagnostic overlay is kept as
a separate RGB ``uint8`` NumPy array.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata

import cv2
import numpy as np
from PIL import Image


ArrayImage = np.ndarray | Image.Image
ArrayMask = np.ndarray | Image.Image | Sequence[int]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class QAThresholds:
    """Hard-gate defaults for a strict, local text edit.

    Thresholds are intentionally conservative.  A production corpus can tune
    them, but a changed Vietnamese diacritic or an edit outside the ROI must
    remain a hard failure and must never be hidden by an aggregate score.
    """

    outside_channel_tolerance: int = 0
    outside_changed_ratio_max: float = 0.0001
    outside_roi_margin_px: int = 0
    ghost_edge_ratio_max: float = 0.05
    ghost_gt_coverage_max: float = 0.005
    ghost_pixel_delta: int = 5
    ghost_exclusion_radius_px: int = 2
    seam_p95_excess_max: float = 4.0
    seam_zscore_max: float = 4.0
    clipping_margin_px: int = 1
    require_determinism_metadata: bool = True


@dataclass(frozen=True)
class QAResult:
    """A JSON-friendly report paired with a same-size RGB diagnostic overlay."""

    report: dict[str, Any]
    overlay: np.ndarray

    @property
    def passed(self) -> bool:
        return bool(self.report["passed"])

    def to_dict(self) -> dict[str, Any]:
        """Return the report; the overlay is intentionally not embedded."""

        return self.report


def normalize_nfc(text: str) -> str:
    """Normalize text without weakening case, punctuation, spaces, or accents."""

    if not isinstance(text, str):
        raise TypeError("text must be str")
    return unicodedata.normalize("NFC", text)


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value with stable ordering for deterministic configurations."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pixel_sha256(image: ArrayImage) -> str:
    """Hash normalized RGB pixels together with their shape."""

    array = _as_rgb_u8(image)
    digest = hashlib.sha256()
    digest.update(b"V7-RGB-U8\0")
    digest.update(f"{array.shape[0]}x{array.shape[1]}\0".encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def make_determinism_metadata(
    original: ArrayImage,
    result: ArrayImage,
    *,
    seed: int,
    config: Any,
) -> dict[str, Any]:
    """Create the canonical metadata block expected by the QA gate."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return {
        "seed": seed,
        "input_pixel_sha256": pixel_sha256(original),
        "config_sha256": canonical_sha256(config),
        "output_pixel_sha256": pixel_sha256(result),
    }


def evaluate_replacement(
    original: ArrayImage,
    result: ArrayImage,
    *,
    roi_mask: ArrayMask,
    old_text_mask: ArrayMask,
    new_text_alpha: ArrayMask,
    seam_mask: ArrayMask,
    target_text: str,
    rendered_text: str,
    metadata: Mapping[str, Any],
    expected_background: ArrayImage | None = None,
    thresholds: QAThresholds | None = None,
) -> QAResult:
    """Run all hard quality gates for one text replacement.

    Masks must describe the source canvas. ``old_text_mask`` should be a glyph
    mask rather than only an OCR rectangle. ``new_text_alpha`` must include all
    intended effects (fill, stroke, glow, and shadow); otherwise a legitimate
    effect may be misclassified as an old-text ghost.

    ``expected_background`` is optional.  Synthetic tests should supply it for
    direct residual measurement.  Real images can omit it, in which case a
    conservative old-edge/background-ring detector is used.

    Determinism fields may be at the top level of ``metadata`` or in a nested
    ``metadata["determinism"]`` mapping.  Required fields are ``seed``,
    ``input_pixel_sha256``, ``config_sha256``, and ``output_pixel_sha256``.
    """

    limits = thresholds or QAThresholds()
    source = _as_rgb_u8(original)
    output = _as_rgb_u8(result)
    height, width = source.shape[:2]
    shape = (height, width)

    roi = _as_mask(roi_mask, shape, "roi_mask")
    old_mask = _as_mask(old_text_mask, shape, "old_text_mask")
    new_mask = _as_mask(new_text_alpha, shape, "new_text_alpha")
    seams = _as_mask(seam_mask, shape, "seam_mask")

    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    warnings: list[str] = []

    def record(
        name: str,
        passed: bool,
        failure_code: str,
        *,
        metrics: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "passed": bool(passed),
            "metrics": _plain_dict(metrics or {}),
        }
        if details:
            item["details"] = _plain_dict(details)
        checks[name] = item
        if not passed:
            failures.append(failure_code)

    canvas_matches = source.shape == output.shape
    record(
        "canvas",
        canvas_matches,
        "canvas_mismatch",
        details={
            "original_shape": list(source.shape),
            "result_shape": list(output.shape),
        },
    )

    target_nfc = normalize_nfc(target_text)
    rendered_nfc = normalize_nfc(rendered_text)
    record(
        "exact_nfc_target",
        target_nfc == rendered_nfc,
        "target_text_mismatch",
        metrics={"target_length": len(target_nfc), "rendered_length": len(rendered_nfc)},
        details={
            "target_nfc": target_nfc,
            "rendered_nfc": rendered_nfc,
            "target_was_nfc": target_text == target_nfc,
            "rendered_was_nfc": rendered_text == rendered_nfc,
        },
    )

    input_hash = pixel_sha256(source)
    output_hash = pixel_sha256(output)
    determinism_passed, determinism_metrics, determinism_details = _check_determinism(
        metadata,
        input_hash=input_hash,
        output_hash=output_hash,
        required=limits.require_determinism_metadata,
    )
    record(
        "determinism_metadata",
        determinism_passed,
        "determinism_metadata_invalid",
        metrics=determinism_metrics,
        details=determinism_details,
    )

    # A mismatched canvas cannot be compared pixel-for-pixel.  Continue with
    # metadata/text checks, mark all spatial gates failed, and return a useful
    # red overlay on the source canvas rather than raising midway through QA.
    if not canvas_matches:
        for name, code in (
            ("outside_roi", "outside_roi_not_evaluable"),
            ("ghost_residual", "ghost_not_evaluable"),
            ("seam", "seam_not_evaluable"),
            ("clipping", "clipping_not_evaluable"),
        ):
            record(name, False, code, details={"reason": "canvas_mismatch"})
        overlay = source.copy()
        overlay = _tint(overlay, np.ones(shape, dtype=bool), (255, 0, 0), 0.55)
        return _finish_report(checks, failures, warnings, limits, overlay)

    difference = np.max(
        np.abs(output.astype(np.int16) - source.astype(np.int16)), axis=2
    ).astype(np.uint8)
    allowed_roi = _dilate(roi, limits.outside_roi_margin_px)
    outside = ~allowed_roi
    changed_outside = outside & (difference > limits.outside_channel_tolerance)
    outside_count = int(np.count_nonzero(outside))
    changed_outside_count = int(np.count_nonzero(changed_outside))
    outside_ratio = changed_outside_count / max(1, outside_count)
    outside_values = difference[outside]
    record(
        "outside_roi",
        outside_ratio <= limits.outside_changed_ratio_max,
        "outside_roi_changed",
        metrics={
            "outside_pixel_count": outside_count,
            "changed_pixel_count": changed_outside_count,
            "changed_ratio": outside_ratio,
            "mean_abs_channel_max": _mean(outside_values),
            "p95_abs_channel_max": _percentile(outside_values, 95),
            "max_abs_channel": _max(outside_values),
        },
    )

    clipping_bad, clipping_metrics = _clipping_mask(
        roi,
        new_mask,
        margin=limits.clipping_margin_px,
    )
    clipping_passed = (
        clipping_metrics["new_alpha_pixel_count"] > 0
        and clipping_metrics["clipped_pixel_count"] == 0
    )
    record(
        "clipping",
        clipping_passed,
        "new_text_clipped",
        metrics=clipping_metrics,
    )

    erased = old_mask & ~_dilate(new_mask, limits.ghost_exclusion_radius_px)
    ghost_candidates, ghost_metrics, ghost_passed = _ghost_check(
        source,
        output,
        old_mask,
        new_mask,
        erased,
        expected_background=expected_background,
        limits=limits,
    )
    record(
        "ghost_residual",
        ghost_passed,
        "old_text_ghost_detected",
        metrics=ghost_metrics,
    )

    seam_candidates, seam_metrics, seam_passed = _seam_check(
        source,
        output,
        seams,
        limits=limits,
    )
    record(
        "seam",
        seam_passed,
        "blend_or_tile_seam_detected",
        metrics=seam_metrics,
    )

    overlay = output.copy()
    overlay = _tint(overlay, changed_outside, (255, 0, 0), 0.65)
    overlay = _tint(overlay, ghost_candidates, (255, 0, 255), 0.65)
    overlay = _tint(overlay, seam_candidates, (0, 220, 255), 0.65)
    overlay = _tint(overlay, clipping_bad, (255, 210, 0), 0.72)
    overlay[_mask_boundary(roi)] = (0, 255, 0)

    return _finish_report(checks, failures, warnings, limits, overlay)


def save_qa_artifacts(
    result: QAResult,
    *,
    json_path: str | Path,
    overlay_path: str | Path,
) -> None:
    """Write an indented UTF-8 report and an RGB PNG diagnostic overlay."""

    report_target = Path(json_path)
    overlay_target = Path(overlay_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    overlay_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Image.fromarray(result.overlay, "RGB").save(overlay_target, format="PNG")


def _finish_report(
    checks: Mapping[str, Mapping[str, Any]],
    failures: Sequence[str],
    warnings: Sequence[str],
    limits: QAThresholds,
    overlay: np.ndarray,
) -> QAResult:
    passed = not failures
    report: dict[str, Any] = {
        "schema_version": "v7.qa/1",
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "hard_failures": list(dict.fromkeys(failures)),
        "warnings": list(dict.fromkeys(warnings)),
        "checks": dict(checks),
        "thresholds": _plain_dict(asdict(limits)),
        "overlay_legend_rgb": {
            "outside_roi_change": [255, 0, 0],
            "old_text_ghost": [255, 0, 255],
            "seam": [0, 220, 255],
            "clipping": [255, 210, 0],
            "roi_boundary": [0, 255, 0],
        },
    }
    # This assertion protects the public contract during development.
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return QAResult(report=report, overlay=np.ascontiguousarray(overlay, dtype=np.uint8))


def _check_determinism(
    metadata: Mapping[str, Any],
    *,
    input_hash: str,
    output_hash: str,
    required: bool,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    if not isinstance(metadata, Mapping):
        return False, {}, {"errors": ["metadata_not_mapping"]}

    block: Any = metadata.get("determinism", metadata)
    if not isinstance(block, Mapping):
        return False, {}, {"errors": ["determinism_not_mapping"]}

    required_fields = (
        "seed",
        "input_pixel_sha256",
        "config_sha256",
        "output_pixel_sha256",
    )
    errors: list[str] = []
    missing = [field for field in required_fields if field not in block]
    if required and missing:
        errors.extend(f"missing:{field}" for field in missing)

    seed = block.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        errors.append("invalid:seed")

    for field in ("input_pixel_sha256", "config_sha256", "output_pixel_sha256"):
        value = block.get(field)
        if value is not None and (
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
        ):
            errors.append(f"invalid:{field}")

    if block.get("input_pixel_sha256") not in (None, input_hash):
        errors.append("mismatch:input_pixel_sha256")
    if block.get("output_pixel_sha256") not in (None, output_hash):
        errors.append("mismatch:output_pixel_sha256")

    try:
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        errors.append("metadata_not_json_serializable")

    return (
        not errors,
        {
            "required": required,
            "seed_present": seed is not None,
            "input_hash_matches": block.get("input_pixel_sha256") == input_hash,
            "output_hash_matches": block.get("output_pixel_sha256") == output_hash,
        },
        {
            "errors": errors,
            "computed_input_pixel_sha256": input_hash,
            "computed_output_pixel_sha256": output_hash,
            "config_sha256": block.get("config_sha256"),
            "seed": seed,
        },
    )


def _ghost_check(
    original: np.ndarray,
    result: np.ndarray,
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    erased: np.ndarray,
    *,
    expected_background: ArrayImage | None,
    limits: QAThresholds,
) -> tuple[np.ndarray, dict[str, Any], bool]:
    erased_count = int(np.count_nonzero(erased))
    if erased_count == 0:
        return (
            np.zeros_like(erased),
            {
                "mode": "not_applicable",
                "erased_pixel_count": 0,
                "edge_residual_ratio": 0.0,
                "gt_residual_coverage": 0.0,
            },
            True,
        )

    gray_original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    grad_original = _gradient_magnitude(gray_original)
    grad_result = _gradient_magnitude(gray_result)

    old_boundary = _mask_boundary(old_mask)
    source_values = grad_original[old_mask]
    source_threshold = max(8.0, _percentile(source_values, 60))
    template = erased & _dilate(old_boundary | (grad_original >= source_threshold), 1)

    ring_radius = max(4, limits.ghost_exclusion_radius_px * 3)
    background_ring = _dilate(old_mask, ring_radius) & ~_dilate(
        old_mask, max(1, limits.ghost_exclusion_radius_px)
    )
    background_ring &= ~new_mask
    ring_values = grad_result[background_ring]
    ring_median, ring_mad = _median_mad(ring_values)
    output_edge_threshold = max(8.0, ring_median + 3.0 * 1.4826 * ring_mad)

    edge_candidates = template & (grad_result > output_edge_threshold)
    template_count = int(np.count_nonzero(template))
    edge_ratio = int(np.count_nonzero(edge_candidates)) / max(1, template_count)

    gt_candidates = np.zeros_like(erased)
    gt_coverage = 0.0
    mode = "no_ground_truth"
    if expected_background is not None:
        expected = _as_rgb_u8(expected_background)
        if expected.shape != result.shape:
            raise ValueError(
                f"expected_background shape {expected.shape} does not match {result.shape}"
            )
        residual = np.max(
            np.abs(result.astype(np.int16) - expected.astype(np.int16)), axis=2
        )
        gt_candidates = erased & (residual > limits.ghost_pixel_delta)
        gt_coverage = int(np.count_nonzero(gt_candidates)) / erased_count
        mode = "ground_truth"

    passed = edge_ratio <= limits.ghost_edge_ratio_max
    if expected_background is not None:
        passed = passed and gt_coverage <= limits.ghost_gt_coverage_max

    candidates = edge_candidates | gt_candidates
    return (
        candidates,
        {
            "mode": mode,
            "erased_pixel_count": erased_count,
            "edge_template_pixel_count": template_count,
            "edge_residual_pixel_count": int(np.count_nonzero(edge_candidates)),
            "edge_residual_ratio": edge_ratio,
            "background_edge_median": ring_median,
            "background_edge_mad": ring_mad,
            "edge_detection_threshold": output_edge_threshold,
            "gt_residual_pixel_count": int(np.count_nonzero(gt_candidates)),
            "gt_residual_coverage": gt_coverage,
        },
        passed,
    )


def _seam_check(
    original: np.ndarray,
    result: np.ndarray,
    seam_mask: np.ndarray,
    *,
    limits: QAThresholds,
) -> tuple[np.ndarray, dict[str, Any], bool]:
    seam_count = int(np.count_nonzero(seam_mask))
    if seam_count == 0:
        return (
            np.zeros_like(seam_mask),
            {
                "mode": "not_applicable",
                "seam_pixel_count": 0,
                "median_excess": 0.0,
                "p95_excess": 0.0,
                "zscore": 0.0,
            },
            True,
        )

    gray_original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    contrast_original = _local_range(gray_original)
    contrast_result = _local_range(gray_result)
    excess = np.maximum(0.0, contrast_result - contrast_original)

    band = _dilate(seam_mask, 1)
    seam_values = excess[band]
    median_excess = _median(seam_values)
    p95_excess = _percentile(seam_values, 95)

    reference = _dilate(seam_mask, 4) & ~_dilate(seam_mask, 1)
    reference_values = excess[reference]
    reference_median, reference_mad = _median_mad(reference_values)
    zscore = max(
        0.0,
        (median_excess - reference_median) / (1.4826 * reference_mad + 1.0),
    )

    candidate_threshold = max(
        limits.seam_p95_excess_max,
        reference_median + 4.0 * 1.4826 * reference_mad,
    )
    candidates = band & (excess > candidate_threshold)
    passed = (
        p95_excess <= limits.seam_p95_excess_max
        and zscore <= limits.seam_zscore_max
    )
    return (
        candidates,
        {
            "mode": "known_boundaries",
            "seam_pixel_count": seam_count,
            "analysis_band_pixel_count": int(np.count_nonzero(band)),
            "median_excess": median_excess,
            "p95_excess": p95_excess,
            "reference_median": reference_median,
            "reference_mad": reference_mad,
            "zscore": zscore,
            "candidate_pixel_count": int(np.count_nonzero(candidates)),
        },
        passed,
    )


def _clipping_mask(
    roi: np.ndarray,
    new_alpha: np.ndarray,
    *,
    margin: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    safe = _erode(roi, margin)
    bad = new_alpha & ~safe
    alpha_count = int(np.count_nonzero(new_alpha))
    bad_count = int(np.count_nonzero(bad))
    return bad, {
        "new_alpha_pixel_count": alpha_count,
        "clipped_pixel_count": bad_count,
        "clipped_ratio": bad_count / max(1, alpha_count),
        "required_margin_px": margin,
    }


def _as_rgb_u8(image: ArrayImage) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(f"image must be HxW, HxWx1, HxWx3, or HxWx4; got {array.shape}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        # QA compares visible RGB content.  Alpha integrity is a separate format
        # gate at export time and must not be silently composited here.
        array = array[..., :3]

    if np.issubdtype(array.dtype, np.floating):
        finite = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and float(np.max(finite)) <= 1.0:
            finite = finite * 255.0
        array = np.rint(finite)
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _as_mask(mask: ArrayMask, shape: tuple[int, int], name: str) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"))
    elif isinstance(mask, Sequence) and not isinstance(mask, (str, bytes, np.ndarray)):
        if len(mask) != 4:
            raise ValueError(f"{name} bbox must be (x0, y0, x1, y1)")
        x0, y0, x1, y1 = (int(value) for value in mask)
        if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
            raise ValueError(f"{name} bbox is outside canvas: {(x0, y0, x1, y1)}")
        array = np.zeros(shape, dtype=bool)
        array[y0:y1, x0:x1] = True
        return array
    else:
        array = np.asarray(mask)
        if array.ndim == 3:
            array = np.max(array, axis=2)

    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match canvas {shape}")
    return np.ascontiguousarray(array > 0, dtype=bool)


def _kernel(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("morphology radius must be non-negative")
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    return cv2.dilate(mask.astype(np.uint8), _kernel(radius), iterations=1) > 0


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    return cv2.erode(mask.astype(np.uint8), _kernel(radius), iterations=1) > 0


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~_erode(mask, 1)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(x, y)


def _local_range(gray: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    high = cv2.dilate(gray, kernel).astype(np.float32)
    low = cv2.erode(gray, kernel).astype(np.float32)
    return high - low


def _tint(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    if not np.any(mask):
        return image
    output = image.copy()
    base = output[mask].astype(np.float32)
    tint = np.asarray(color, dtype=np.float32)
    output[mask] = np.rint(base * (1.0 - alpha) + tint * alpha).astype(np.uint8)
    return output


def _median_mad(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values.astype(np.float64) - median)))
    return median, mad


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else 0.0


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def _max(values: np.ndarray) -> int:
    return int(np.max(values)) if values.size else 0


def _plain_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain(item) for key, item in value.items()}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_dict(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "QAResult",
    "QAThresholds",
    "canonical_sha256",
    "evaluate_replacement",
    "make_determinism_metadata",
    "normalize_nfc",
    "pixel_sha256",
    "save_qa_artifacts",
]
