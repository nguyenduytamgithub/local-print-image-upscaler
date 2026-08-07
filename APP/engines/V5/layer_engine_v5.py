"""V5 smart layer extraction engine.

This engine never claims to recover hidden pixels from a flat bitmap. It uses
official SAM 2.1 masks, semantic/OCR grouping, conservative background
restoration, and open editable containers with a machine-readable audit trail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from v5lib.formats import (
    asset_records,
    export_contact_sheet,
    export_ora,
    export_png_assets,
    export_psd,
    find_required_alpha_promoted_singletons,
    matte_quality_report,
    ordered_layer_specs,
    package_layers_zip,
    prune_required_alpha_promoted_singletons,
    project_clean_plate_for_alpha,
    render_layers,
    rendered_alpha_canvases,
    resize_rgb,
    resize_semantic_support,
    save_color_png,
    serialized_base_alpha_u8,
    sha256_file,
    write_json,
)
from v5lib.matte_clean import clean_semantic_layer_masks, clean_text_layer_masks
from v5lib.matting import refine_text_alpha_mattes
from v5lib.model import LayerSpec
from v5lib.poster_group import add_ocr_text_layers, group_poster_layers
from v5lib.restore import photographic_score, restore_background
from v5lib.segment import (
    build_mask_candidates,
    detect_text_regions,
    generate_sam_candidates,
    run_object_detection,
    segmentation_overlay,
    select_smart_layers,
)


Image.MAX_IMAGE_PIXELS = 500_000_000
MIN_SCALE = 1.0
MAX_SCALE = 20.0
MAX_RECOMPOSITION_ERROR = 1
OPEN_CONTAINER_SAFE_RAW_BYTES = 12_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5: flat bitmap to smart editable raster layers")
    parser.add_argument("input", type=Path)
    parser.add_argument("scale", type=float)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--master", type=Path, help="Optional V3 AI master at exact final size")
    parser.add_argument("--name", default=None)
    parser.add_argument("--max-layers", type=int, default=24)
    parser.add_argument("--inpaint", choices=("auto", "poster", "lama"), default="auto")
    parser.add_argument("--no-semantic", action="store_true", help="Skip Grounding DINO naming cues")
    parser.add_argument("--app-version", default="dev")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, str]:
    source = args.input.resolve()
    target = args.output_dir.resolve()
    if not source.is_file():
        raise SystemExit(f"Input not found: {source}")
    if not MIN_SCALE <= args.scale <= MAX_SCALE:
        raise SystemExit(f"Scale must be from x{MIN_SCALE:g} through x{MAX_SCALE:g}.")
    if not 4 <= args.max_layers <= 60:
        raise SystemExit("--max-layers must be from 4 through 60.")
    if target.exists():
        raise SystemExit(
            f"Output exists: {target}. V5 refuses to delete an existing directory; "
            "use the unified launcher for atomic replacement."
        )
    target.mkdir(parents=True)
    name = args.name or source.stem
    return source, target, name


def load_rgb(path: Path) -> tuple[Image.Image, bytes | None]:
    with Image.open(path) as image:
        image.load()
        profile = image.info.get("icc_profile")
        return image.convert("RGB"), bytes(profile) if profile else None


def fallback_srgb_profile() -> bytes:
    """Create an explicit sRGB tag only when the normalized input lost its ICC bytes."""

    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    return profile.tobytes()


def assert_portable_bundle_manifest(value: object, staging_dir: Path) -> None:
    """Reject dead launcher staging paths before manifest.json enters LAYERS.zip."""

    strings: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    staging_tokens = {
        str(staging_dir),
        str(staging_dir).replace("\\", "/"),
        "job_v5_",
        ".new-",
    }
    dead = sorted(
        text
        for text in strings
        if any(token and token in text for token in staging_tokens)
    )
    if dead:
        raise RuntimeError(
            "Portable V5 manifest contains launcher staging path(s): "
            + json.dumps(dead[:8], ensure_ascii=False)
        )


def reserve_promoted_singleton_budget(
    records: list[dict[str, object]],
    already_removed: dict[str, int],
    *,
    maximum_per_layer: int = 2,
) -> dict[str, int]:
    """Reserve a hard cumulative auto-removal budget for each text layer."""

    if maximum_per_layer < 1:
        raise ValueError("maximum_per_layer must be positive.")
    updated = {str(layer_id): int(count) for layer_id, count in already_removed.items()}
    if any(count < 0 for count in updated.values()):
        raise ValueError("already_removed counts cannot be negative.")
    for record in records:
        try:
            layer_id = str(record["layer_id"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Promoted-singleton budget record is malformed.") from exc
        updated[layer_id] = updated.get(layer_id, 0) + 1
        if updated[layer_id] > maximum_per_layer:
            raise RuntimeError(
                "V5 found more renderer-promoted singleton candidates than the "
                f"safe cumulative cap ({maximum_per_layer}) for layer {layer_id!r}; "
                "automatic cleanup stopped and no bundle was published."
            )
    return updated


def summarize_topology_failures(
    matte_qa: dict[str, object],
    *,
    maximum_records: int = 12,
) -> list[str]:
    """Return compact release-gate diagnostics with exact failing relations."""

    if maximum_records < 1:
        raise ValueError("maximum_records must be positive.")
    fields = (
        "missing_expected_pixels",
        "orphan_actual_components",
        "orphan_actual_component_pixels",
        "merged_actual_components",
        "merge_excess",
        "split_expected_components",
        "split_excess",
        "missing_expected_holes",
        "new_actual_holes",
        "split_expected_holes",
        "merged_actual_holes",
        "expected_hole_core_intrusion_pixels",
    )
    summaries: list[str] = []
    for layer in matte_qa.get("layers", []):
        if not isinstance(layer, dict):
            continue
        topology = layer.get("scaled_topology")
        if not isinstance(topology, dict) or topology.get("passed", True):
            continue
        for threshold in topology.get("thresholds", []):
            if not isinstance(threshold, dict) or threshold.get("passed", True):
                continue
            failures = [
                f"{field}={int(threshold[field])}"
                for field in fields
                if int(threshold.get(field, 0)) != 0
            ]
            summaries.append(
                f"{layer.get('layer_id', '?')}@{threshold.get('integer_alpha_level', '?')}:"
                + (",".join(failures) if failures else "unspecified_topology_failure")
            )
            if len(summaries) >= maximum_records:
                return summaries
    return summaries


def apply_canonical_refined_alpha(
    layers: list[LayerSpec],
    topology_reference: dict[str, np.ndarray],
) -> list[LayerSpec]:
    """Promote stable x1 serialized alpha to the immutable xN source of truth."""

    updated: list[LayerSpec] = []
    for layer in layers:
        if layer.alpha_matte is None:
            updated.append(layer)
            continue
        if layer.layer_id not in topology_reference:
            raise ValueError(
                f"Missing canonical refined alpha for layer: {layer.layer_id}"
            )
        canonical = np.asarray(topology_reference[layer.layer_id])
        if canonical.dtype != np.uint8 or canonical.ndim != 2:
            raise ValueError("Canonical refined alpha must be two-dimensional uint8.")
        if canonical.shape != layer.mask.shape:
            raise ValueError(
                f"Canonical refined alpha shape {canonical.shape} does not match "
                f"mask {layer.mask.shape} for {layer.layer_id}."
            )
        if np.logical_and(canonical > 0, ~layer.mask).any():
            raise ValueError(
                f"Canonical refined alpha escapes semantic mask: {layer.layer_id}"
            )
        updated.append(
            LayerSpec(
                layer_id=layer.layer_id,
                name=layer.name,
                category=layer.category,
                mask=np.array(layer.mask, dtype=bool, copy=True),
                score=float(layer.score),
                source_ids=list(layer.source_ids),
                label=layer.label,
                text=layer.text,
                metadata=dict(layer.metadata),
                alpha_matte=canonical.astype(np.float32) / 255.0,
            )
        )
    return updated


def plan_fixed_alpha_stack(
    master_rgb: np.ndarray,
    background_candidate_rgb: np.ndarray,
    layers: list[LayerSpec],
    alpha_canvases: dict[str, np.ndarray],
    *,
    preferred_layer_targets: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Plan every lower prefix backwards without changing serialized alpha.

    A clean background (or cleaned parent) can be too different from the
    requested master for a translucent edge to reconstruct.  Solving each
    layer independently in forward order is unsafe when layers overlap: alpha
    owned by a later child cannot help an earlier parent.  This planner starts
    from the exact final master and walks the real PSD/ORA z-order backwards.
    For each layer it projects only the immediately lower prefix into the
    colour interval that *that layer's own uint8 alpha* can reconstruct.

    The returned target for a layer is the planned prefix immediately after
    that layer.  Therefore a normal bottom-to-top renderer can reproduce every
    planned prefix while preserving holes and antialiased contour coverage.
    """

    master = np.asarray(master_rgb)
    background_candidate = np.asarray(background_candidate_rgb)
    if (
        master.dtype != np.uint8
        or background_candidate.dtype != np.uint8
        or master.ndim != 3
        or master.shape[2] != 3
        or background_candidate.shape != master.shape
    ):
        raise ValueError(
            "master_rgb and background_candidate_rgb must be same-shape uint8 RGB arrays."
        )
    preferred_targets = preferred_layer_targets or {}
    unknown_targets = sorted(set(preferred_targets) - {layer.layer_id for layer in layers})
    if unknown_targets:
        raise ValueError(
            "Preferred targets reference unknown layer ids: " + ", ".join(unknown_targets)
        )

    ordered = ordered_layer_specs(layers)
    height, width = master.shape[:2]
    shape = (height, width)
    preferred_prefixes: list[np.ndarray] = [
        np.array(background_candidate, dtype=np.uint8, copy=True)
    ]
    for layer in ordered:
        alpha = alpha_canvases.get(layer.layer_id)
        if alpha is None:
            raise ValueError(f"Missing serialized alpha canvas for {layer.layer_id!r}.")
        alpha_array = np.asarray(alpha)
        if alpha_array.dtype != np.uint8 or alpha_array.shape != shape:
            raise ValueError(
                f"Serialized alpha for {layer.layer_id!r} must be canvas-size uint8."
            )
        target = np.asarray(preferred_targets.get(layer.layer_id, master))
        if target.dtype != np.uint8 or target.shape != master.shape:
            raise ValueError(
                f"Preferred target for {layer.layer_id!r} must be canvas-size uint8 RGB."
            )
        preferred_after = preferred_prefixes[-1].copy()
        support = alpha_array > 0
        preferred_after[support] = target[support]
        preferred_prefixes.append(preferred_after)

    # Reuse the preferred-prefix storage in place. During the descending walk,
    # index ``i`` is still its untouched clean preference while ``i + 1`` has
    # already become the exact planned upper prefix. This avoids retaining two
    # full RGB canvases per layer on large print jobs.
    planned_prefixes = preferred_prefixes
    planned_prefixes[-1] = np.array(master, dtype=np.uint8, copy=True)
    reverse_reports: list[dict[str, object]] = []
    for index in range(len(ordered) - 1, -1, -1):
        layer = ordered[index]
        upper_target = planned_prefixes[index + 1]
        lower, projection = project_clean_plate_for_alpha(
            upper_target,
            preferred_prefixes[index],
            alpha_canvases[layer.layer_id],
        )
        planned_prefixes[index] = lower
        reverse_reports.append(
            {
                "layer_id": layer.layer_id,
                "z_index_bottom_to_top": index,
                **projection,
            }
        )

    planned_background = planned_prefixes[0]
    layer_targets = {
        layer.layer_id: planned_prefixes[index + 1]
        for index, layer in enumerate(ordered)
    }
    ordered_reports = list(reversed(reverse_reports))
    return planned_background, layer_targets, {
        "policy": (
            "exact uint8 alpha-feasible projection solved backwards over the canonical "
            "PSD/ORA z-order; each step uses only that layer's own serialized alpha"
        ),
        "layer_count": len(ordered),
        "changed_pixel_count_sum": sum(
            int(record["changed_pixel_count"]) for record in ordered_reports
        ),
        "all_steps_feasible": all(
            bool(record.get("feasibility_gate_passed", False))
            for record in ordered_reports
        ),
        "layers": ordered_reports,
    }


def main() -> int:
    args = parse_args()
    source, output_dir, name = validate_args(args)
    started = time.perf_counter()
    source_image, source_icc = load_rgb(source)
    source_rgb = np.array(source_image, dtype=np.uint8, copy=True)
    final_size = tuple(int(round(value * args.scale)) for value in source_image.size)
    if args.master:
        master_path = args.master.resolve()
        if not master_path.is_file():
            raise RuntimeError(f"V3 master not found: {master_path}")
        master_image, master_icc = load_rgb(master_path)
        if master_image.size != final_size:
            raise RuntimeError(f"V3 master is {master_image.size}, expected {final_size}.")
        master_policy = "V3 AI master supplied by unified launcher"
    else:
        master_image = source_image.resize(final_size, Image.Resampling.LANCZOS)
        master_icc = source_icc
        master_path = source
        master_policy = "Lanczos fallback; unified launcher normally supplies V3 for scale > 1"
    master_rgb = np.array(master_image, dtype=np.uint8, copy=True)
    output_icc = master_icc or source_icc or fallback_srgb_profile()
    output_icc_sha256 = hashlib.sha256(output_icc).hexdigest()

    masks, scores, sam_report = generate_sam_candidates(source_image)
    candidates = build_mask_candidates(source_rgb, masks, scores)

    text_regions, ocr_report = detect_text_regions(source)
    detections = []
    semantic_report: dict[str, object]
    if args.no_semantic:
        semantic_report = {"disabled": True, "accepted_detection_count": 0}
    else:
        try:
            detections, semantic_report = run_object_detection(source_image)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            semantic_report = {
                "failed": True,
                "error_type": type(exc).__name__,
                "accepted_detection_count": 0,
                "notice": "Layer extraction continued with SAM and OCR grouping.",
            }

    print("  V5 bước 2b/4: đang gộp mask thành các khối thao tác hợp lý...", flush=True)
    content_score = photographic_score(source_rgb)
    if content_score < 0.54:
        layers, grouping_report = group_poster_layers(
            source_rgb,
            masks,
            scores,
            args.max_layers,
        )
        layers, text_layering_report = add_ocr_text_layers(
            source_rgb,
            layers,
            candidates,
            text_regions,
            args.max_layers,
            detections=detections,
        )
        grouping_report.update(
            {
                "selected_layer_count": len(layers),
                "hierarchy_edges": sum(
                    layer.metadata.get("parent_id") is not None for layer in layers
                ),
                "ocr_layering": text_layering_report,
                "content_class": "poster_or_graphic",
                "photographic_score": round(content_score, 5),
                "semantic_policy": (
                    "Poster geometry determines masks; OCR and Grounding DINO are metadata hints only."
                ),
            }
        )
    else:
        layers, grouping_report = select_smart_layers(
            source_rgb,
            candidates,
            text_regions,
            detections,
            max_layers=args.max_layers,
        )
        grouping_report.update(
            {
                "content_class": "photographic_or_textured",
                "photographic_score": round(content_score, 5),
            }
        )
    if not layers:
        raise RuntimeError(
            "V5 did not find a stable movable region. Keep the original image and try another source."
        )

    # The grouping stage deliberately explores halos and nearby colour regions.
    # Editable ownership must be stricter: recover the exact recorded SAM
    # components for objects, then strip panel rules/unanchored specks from text.
    layers, object_matte_report = clean_semantic_layer_masks(layers, masks)
    layers, text_matte_report = clean_text_layer_masks(layers, image_rgb=source_rgb)
    layers, vitmatte_report = refine_text_alpha_mattes(
        source_rgb,
        layers,
        progress=lambda message: print(message, flush=True),
    )
    matte_cleanup_report = {
        "delivery_policy": (
            "colour/topology-clean semantic ownership followed by pinned ViTMatte "
            "fractional coverage constrained inside that topology"
        ),
        "objects": object_matte_report,
        "text": text_matte_report,
        "text_boundary_refinement": vitmatte_report,
    }
    del masks, scores, candidates

    by_id = {layer.layer_id: layer for layer in layers}

    def layer_cleanup_support(layer: LayerSpec, size: tuple[int, int]) -> np.ndarray:
        """Return pixels that lower layers may replace with synthesized colour.

        Fractional text can render in a one-source-pixel Lanczos envelope, but
        that envelope is antialias coverage rather than semantic ownership.
        Keeping the master on lower layers outside nearest-resized ownership
        prevents resampling lobes from requiring opaque alpha promotion.
        """

        return resize_semantic_support(layer.mask, size)

    def serialized_alpha_cache(size: tuple[int, int]) -> dict[str, np.ndarray]:
        return {
            layer.layer_id: serialized_base_alpha_u8(
                layer,
                size,
                matte_policy="clean",
            )
            for layer in layers
        }

    def descendants(layer_id: str) -> list[LayerSpec]:
        result: list[LayerSpec] = []
        queue = list(by_id[layer_id].metadata.get("children", []))
        while queue:
            child_id = str(queue.pop(0))
            child = by_id.get(child_id)
            if child is None:
                continue
            result.append(child)
            queue.extend(child.metadata.get("children", []))
        return result

    def prepare_source_stack():
        """Build the intentionally unprojected x1 detector clean plates.

        Required-alpha promotion is evidence for the narrowly scoped speck
        detector.  Alpha-feasible projection happens only after that detector
        converges, so the historical three bad singleton candidates remain
        observable and removable.
        """

        top_level_masks = [
            layer.mask for layer in layers if layer.metadata.get("parent_id") is None
        ]
        restoration_pass = restore_background(
            source_rgb,
            top_level_masks,
            mode=args.inpaint,
        )
        background_candidate = np.array(
            restoration_pass.background, dtype=np.uint8, copy=True
        )
        background_source = background_candidate
        background_semantic_source = np.zeros(source_rgb.shape[:2], dtype=bool)
        for layer in layers:
            if layer.metadata.get("parent_id") is None:
                background_semantic_source |= layer_cleanup_support(
                    layer, source_image.size
                )
        # Detector preflight intentionally keeps the old full semantic clean
        # plate so renderer-promoted background specks remain visible.
        background_source[~background_semantic_source] = source_rgb[
            ~background_semantic_source
        ]

        source_targets: dict[str, np.ndarray] = {}
        cleaned_sources: dict[str, np.ndarray] = {}
        cleanup_records: list[dict[str, object]] = []
        for layer in layers:
            child_layers = descendants(layer.layer_id)
            if not child_layers:
                continue
            cleaned = restore_background(
                source_rgb,
                [child.mask for child in child_layers],
                mode="poster",
                progress=lambda _message: None,
            )
            cleaned_source = np.array(cleaned.background, dtype=np.uint8, copy=True)
            child_union = np.zeros(source_rgb.shape[:2], dtype=bool)
            replace_source = np.zeros(source_rgb.shape[:2], dtype=bool)
            for child in child_layers:
                child_union |= child.mask
                replace_source |= layer_cleanup_support(child, source_image.size)
            target_for_layer = source_rgb.copy()
            target_for_layer[replace_source] = cleaned_source[replace_source]
            source_targets[layer.layer_id] = target_for_layer
            cleaned_sources[layer.layer_id] = cleaned_source
            cleanup_records.append(
                {
                    "layer_id": layer.layer_id,
                    "descendant_masks_removed": len(child_layers),
                    "cleanup_footprint_ratio": round(
                        float(cleaned.removal_footprint.mean()), 6
                    ),
                    "published_semantic_ratio": round(float(child_union.mean()), 6),
                }
            )
        return (
            restoration_pass,
            background_source,
            source_targets,
            cleaned_sources,
            cleanup_records,
        )

    # The exact-colour solver may promote a low, background-like source alpha
    # to 255 after ViTMatte. Detect that only at x1, transfer the pixel out of
    # the child mask, and rebuild every lower target so no ghost is left behind.
    preflight_passes: list[dict[str, object]] = []
    maximum_preflight_prune_passes = 3
    maximum_promoted_singletons_per_layer = 2
    removed_promoted_singletons_by_layer: dict[str, int] = {}
    for preflight_index in range(maximum_preflight_prune_passes + 1):
        (
            restoration,
            background_source,
            source_layer_targets,
            cleaned_layer_sources,
            hierarchy_cleanup,
        ) = prepare_source_stack()
        (
            preflight_rendered,
            preflight_composite,
            preflight_composition,
        ) = render_layers(
            source_rgb,
            background_source,
            layers,
            layer_targets=source_layer_targets,
            matte_policy="clean",
        )
        if int(preflight_composition["recomposition_max_abs_error"]) > MAX_RECOMPOSITION_ERROR:
            raise RuntimeError(
                "V5 source-resolution ownership preflight could not reproduce the source "
                "within the release error limit."
            )
        promoted = find_required_alpha_promoted_singletons(
            preflight_rendered,
            source_image.size,
            source_rgb,
        )
        pass_record: dict[str, object] = {
            "pass": preflight_index + 1,
            "candidate_count": len(promoted),
            "candidates": promoted,
            "recomposition_max_abs_error": int(
                preflight_composition["recomposition_max_abs_error"]
            ),
        }
        preflight_passes.append(pass_record)
        if not promoted:
            del preflight_rendered, preflight_composite
            break
        del preflight_rendered, preflight_composite
        if preflight_index >= maximum_preflight_prune_passes:
            raise RuntimeError(
                "V5 promoted-singleton cleanup did not converge safely; no bundle was published."
            )
        reserved_budget = reserve_promoted_singleton_budget(
            promoted,
            removed_promoted_singletons_by_layer,
            maximum_per_layer=maximum_promoted_singletons_per_layer,
        )
        layers, prune_report = prune_required_alpha_promoted_singletons(
            layers,
            promoted,
        )
        pass_record["prune"] = prune_report
        if int(prune_report["applied_count"]) != len(promoted):
            raise RuntimeError(
                "V5 promoted-singleton cleanup became stale during preflight; "
                "no bundle was published."
            )
        removed_promoted_singletons_by_layer = reserved_budget
        by_id = {layer.layer_id: layer for layer in layers}
    else:  # pragma: no cover - loop either converges or raises above
        raise RuntimeError("V5 promoted-singleton preflight did not terminate.")

    # Detector is now stable. Build a separate canonical delivery render whose
    # lower prefixes are projected backwards against the raw, pruned x1 matte.
    # This preserves exact topology instead of canonizing renderer promotions
    # caused by an over-clean background or parent plate.
    source_alpha_canvases = serialized_alpha_cache(source_image.size)
    (
        canonical_background,
        canonical_layer_targets,
        canonical_projection_report,
    ) = plan_fixed_alpha_stack(
        source_rgb,
        background_source,
        layers,
        source_alpha_canvases,
        preferred_layer_targets=source_layer_targets,
    )
    if not canonical_projection_report["all_steps_feasible"]:
        raise RuntimeError(
            "V5 reverse fixed-alpha canonical planning found an infeasible prefix; "
            "no bundle was published."
        )
    (
        canonical_rendered,
        canonical_composite,
        canonical_composition,
    ) = render_layers(
        source_rgb,
        canonical_background,
        layers,
        layer_targets=canonical_layer_targets,
        matte_policy="clean",
    )
    if int(canonical_composition["recomposition_max_abs_error"]) > MAX_RECOMPOSITION_ERROR:
        raise RuntimeError(
            "V5 alpha-feasible source canonical render could not reproduce the source "
            "within the release error limit."
        )
    canonical_alpha_canvases = rendered_alpha_canvases(
        canonical_rendered,
        source_image.size,
    )
    for layer in layers:
        actual_alpha = canonical_alpha_canvases.get(layer.layer_id)
        expected_alpha = source_alpha_canvases[layer.layer_id]
        if actual_alpha is None or not np.array_equal(actual_alpha, expected_alpha):
            raise RuntimeError(
                "V5 source canonical render changed fixed serialized alpha for "
                f"{layer.layer_id!r}; no bundle was published."
            )
    topology_reference = {
        layer.layer_id: canonical_alpha_canvases[layer.layer_id]
        for layer in layers
        if layer.alpha_matte is not None
    }
    del (
        canonical_rendered,
        canonical_composite,
        canonical_background,
        canonical_layer_targets,
        canonical_alpha_canvases,
        source_alpha_canvases,
        source_layer_targets,
        background_source,
    )
    layers = apply_canonical_refined_alpha(layers, topology_reference)
    by_id = {layer.layer_id: layer for layer in layers}
    matte_cleanup_report["canonical_scaled_alpha"] = {
        "policy": (
            "after the unprojected speck detector converges, a separate reverse-planned "
            "x1 delivery render must preserve the raw pruned serialized alpha byte-for-byte; "
            "that stable alpha replaces raw ViTMatte coverage for all later scaling while "
            "semantic masks and hierarchy metadata remain unchanged"
        ),
        "refined_layer_count": len(topology_reference),
        "source_size": list(source_image.size),
        "source_recomposition_max_abs_error": int(
            canonical_composition["recomposition_max_abs_error"]
        ),
    }

    promoted_singleton_report = {
        "policy": (
            "unprojected source-resolution detector preflight; remove only renderer-promoted "
            "area-1 pure-text alpha with low source matte plus local background colour "
            "evidence, then rebuild parent/background ownership before canonical planning"
        ),
        "stable": not bool(preflight_passes[-1]["candidate_count"]),
        "pass_count": len(preflight_passes),
        "removed_count": sum(
            int(record.get("prune", {}).get("applied_count", 0))
            for record in preflight_passes
        ),
        "maximum_automatic_removals_per_layer": (
            maximum_promoted_singletons_per_layer
        ),
        "removed_by_layer": removed_promoted_singletons_by_layer,
        "passes": preflight_passes,
    }
    matte_cleanup_report["render_promoted_singletons"] = promoted_singleton_report

    matte_cleanup_report["alpha_feasible_clean_plate_projection"] = {
        "source_canonical": canonical_projection_report,
        "final_delivery": None,
    }

    background_candidate = resize_rgb(restoration.background, final_size)
    background_semantic_final = np.zeros(master_rgb.shape[:2], dtype=bool)
    for layer in layers:
        if layer.metadata.get("parent_id") is None:
            background_semantic_final |= layer_cleanup_support(layer, final_size)
    # Shadows, glows, panel borders and adjacent artwork do not belong to a
    # movable object merely because the inpainting solver needed a wider work
    # footprint. Keep every non-semantic pixel on the lower/background layer.
    background_candidate[~background_semantic_final] = master_rgb[
        ~background_semantic_final
    ]
    restoration.report["published_replacement_policy"] = (
        "synthesized pixels published only inside top-level semantic ownership; "
        "restoration work radius never expands an editable alpha matte; exact "
        "uint8 feasibility may retain master colour where translucent coverage "
        "cannot reconstruct the fully cleaned candidate"
    )

    preferred_layer_targets: dict[str, np.ndarray] = {}
    for layer_id, cleaned_source in cleaned_layer_sources.items():
        cleaned_final = resize_rgb(cleaned_source, final_size)
        replace = np.zeros(master_rgb.shape[:2], dtype=bool)
        for child in descendants(layer_id):
            replace |= layer_cleanup_support(child, final_size)
        target_for_layer = master_rgb.copy()
        target_for_layer[replace] = cleaned_final[replace]
        preferred_layer_targets[layer_id] = target_for_layer

    final_alpha_canvases = serialized_alpha_cache(final_size)
    background_rgb, layer_targets, final_projection_report = plan_fixed_alpha_stack(
        master_rgb,
        background_candidate,
        layers,
        final_alpha_canvases,
        preferred_layer_targets=preferred_layer_targets,
    )
    del background_candidate, preferred_layer_targets
    if not final_projection_report["all_steps_feasible"]:
        raise RuntimeError(
            "V5 reverse fixed-alpha clean-plate planning found an infeasible prefix; "
            "no bundle was published."
        )
    matte_cleanup_report["alpha_feasible_clean_plate_projection"][
        "final_delivery"
    ] = final_projection_report
    background_image = Image.fromarray(background_rgb, "RGB")

    print("  V5 bước 4/4: đang dựng PNG layer, PSD, OpenRaster và kiểm định...", flush=True)
    rendered, composite, composition_report = render_layers(
        master_rgb,
        background_rgb,
        layers,
        layer_targets=layer_targets,
        matte_policy="clean",
    )
    composition_report["release_max_abs_error"] = MAX_RECOMPOSITION_ERROR
    composition_report["release_gate_passed"] = (
        int(composition_report["recomposition_max_abs_error"]) <= MAX_RECOMPOSITION_ERROR
    )
    if not composition_report["release_gate_passed"]:
        raise RuntimeError(
            "V5 layer stack cannot reproduce the selected master safely: "
            f"max error {composition_report['recomposition_max_abs_error']} > "
            f"{MAX_RECOMPOSITION_ERROR}. No bundle was published."
        )
    delivered_alpha_canvases = rendered_alpha_canvases(rendered, final_size)
    for layer in layers:
        actual_alpha = delivered_alpha_canvases.get(layer.layer_id)
        expected_alpha = final_alpha_canvases[layer.layer_id]
        if actual_alpha is None or not np.array_equal(actual_alpha, expected_alpha):
            raise RuntimeError(
                "V5 final renderer changed canonical fixed alpha for "
                f"{layer.layer_id!r}; no bundle was published."
            )
    composition_report["fixed_alpha_gate_passed"] = True
    composition_report["fixed_alpha_checked_layer_count"] = len(layers)
    composition_report["fixed_alpha_policy"] = (
        "every delivered PNG alpha canvas must equal the reverse-planner's "
        "serialized canonical alpha byte-for-byte"
    )
    del delivered_alpha_canvases, final_alpha_canvases, layer_targets
    matte_qa = matte_quality_report(
        rendered,
        final_size,
        topology_reference=topology_reference,
    )
    if not matte_qa["ownership_gate_passed"]:
        raise RuntimeError(
            "V5 clean-matte ownership failed: "
            f"{matte_qa['alpha_outside_semantic_pixels']} alpha pixels lie outside "
            "their semantic masks. No bundle was published."
        )
    if not matte_qa["topology_gate_passed"]:
        failure_details = summarize_topology_failures(matte_qa)
        raise RuntimeError(
            "V5 scaled text-matte topology diverged from its Lanczos source-alpha "
            "reference: "
            + " | ".join(failure_details)
            + ". No bundle was published."
        )
    final_pixels = final_size[0] * final_size[1]
    cropped_layer_raw_bytes = sum(
        item.rgba.width * item.rgba.height * 5 for item in rendered
    )
    open_container_raw_bytes = final_pixels * 3 + cropped_layer_raw_bytes
    projected_export_disk_bytes = (
        open_container_raw_bytes * 3 + final_pixels * 8 + 512 * 1024**2
    )
    free_export_disk_bytes = shutil.disk_usage(output_dir).free
    render_resource_report = {
        "background_rgb_raw_bytes": final_pixels * 3,
        "cropped_rgba_and_mask_raw_bytes": cropped_layer_raw_bytes,
        "open_container_raw_bytes": open_container_raw_bytes,
        "hard_open_container_raw_bytes": OPEN_CONTAINER_SAFE_RAW_BYTES,
        "projected_remaining_export_disk_bytes": projected_export_disk_bytes,
        "free_disk_bytes_before_export": free_export_disk_bytes,
    }
    if open_container_raw_bytes > OPEN_CONTAINER_SAFE_RAW_BYTES:
        raise RuntimeError(
            "V5 editable containers exceed the 12 GB raw-layer safety cap. "
            "Reduce n or --max-layers; no bundle was published."
        )
    if projected_export_disk_bytes > free_export_disk_bytes * 0.75:
        raise RuntimeError(
            "V5 does not have enough free disk for the actual cropped layers plus ORA/ZIP export: "
            f"need about {projected_export_disk_bytes / 1024**3:.2f} GiB, "
            f"free {free_export_disk_bytes / 1024**3:.2f} GiB. "
            "Reduce n or --max-layers; no bundle was published."
        )
    scale_token = f"{args.scale:g}".replace(".", "p")
    prefix = f"{name}_V5_x{scale_token}"
    preview_path = output_dir / f"{prefix}_PREVIEW.png"
    overlay_path = output_dir / f"{prefix}_LAYER_MAP.png"
    psd_path = output_dir / f"{prefix}_EDITABLE.psd"
    ora_path = output_dir / f"{prefix}_MASTER.ora"
    contact_path = output_dir / f"{prefix}_CONTACT_SHEET.png"
    zip_path = output_dir / f"{prefix}_LAYERS.zip"
    guide_path = output_dir / "00_HUONG_DAN_V5.txt"
    text_path = output_dir / "TEXT_OCR.json"
    manifest_path = output_dir / "manifest.json"

    guide_path.write_text(
        "V5 - MỞ FILE NÀO?\n"
        "==================\n\n"
        "1. Photoshop / Photopea / Canva: mở file *_EDITABLE.psd.\n"
        "2. Krita / GIMP: mở file *_MASTER.ora.\n"
        "3. LAYERS: các chi tiết PNG nền trong suốt đã cắt.\n"
        "4. MASKS: mặt nạ trắng/đen để kiểm tra biên cắt; không phải ảnh thành phẩm.\n"
        "5. *_CONTACT_SHEET.png: xem nhanh tất cả layer trên nền caro.\n"
        "6. TEXT_OCR.json và manifest.json: hồ sơ kỹ thuật, người dùng thường không cần sửa.\n\n"
        "Layer chữ đã được khoét khoảng âm O/G/N, bảo vệ dấu tiếng Việt và làm mịn alpha bằng "
        "ViTMatte cục bộ. Hãy xem MASKS ở 100% nếu cần kiểm tra biên.\n\n"
        "V5 không thể khôi phục layer gốc đã mất trong ảnh phẳng. Nền bị che là nền tổng hợp; "
        "hãy tắt/bật từng layer và xem ở 100% trước khi in.\n",
        encoding="utf-8",
    )

    save_color_png(composite, preview_path, output_icc)
    save_color_png(segmentation_overlay(source_rgb, layers), overlay_path, output_icc)
    layer_records = export_png_assets(
        output_dir,
        background_image,
        rendered,
        icc_profile=output_icc,
    )
    export_contact_sheet(
        contact_path,
        background_image,
        rendered,
        icc_profile=output_icc,
    )
    write_json(
        text_path,
        {
            "notice": "OCR is a search/name aid only. Text pixels remain raster; fonts are not reconstructed.",
            "regions": [
                {
                    "bbox": list(region.bbox),
                    "text": region.text,
                    "confidence": round(region.confidence, 3),
                }
                for region in text_regions
            ],
        },
    )
    psd_report = export_psd(
        psd_path,
        background_image,
        rendered,
        expected_composite=composite,
        icc_profile=output_icc,
    )
    psd_report["path"] = psd_path.name if psd_report.get("created") else None
    ora_report = export_ora(
        ora_path,
        background_image,
        rendered,
        composite,
        icc_profile=output_icc,
    )
    ora_report["path"] = ora_path.name
    elapsed = round(time.perf_counter() - started, 3)

    manifest: dict[str, object] = {
        "pipeline": "V5_SMART_EDITABLE_LAYERS",
        "schema_version": 1,
        "app_version": args.app_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        # Bundle manifests must remain valid after the launcher's atomic move;
        # absolute job_v5_* and .new-* staging paths would immediately die.
        "source": source.name,
        "path_policy": "bundle-relative outputs; external provenance recorded by basename and SHA-256",
        "source_sha256": sha256_file(source),
        "source_size": list(source_image.size),
        "scale": args.scale,
        "final_size": list(final_size),
        "master": {
            "name": master_path.name,
            "sha256": sha256_file(master_path),
            "policy": master_policy,
        },
        "color_management": {
            "working_space": "sRGB",
            "icc_profile_sha256": output_icc_sha256,
            "icc_profile_bytes": len(output_icc),
            "profile_source": (
                "selected master embedded profile"
                if master_icc
                else "normalized source embedded profile" if source_icc else "Pillow LittleCMS sRGB fallback"
            ),
            "embedded_in": "RGB/RGBA PNG artwork, OpenRaster PNG artwork and PSD document resource",
            "mask_policy": "grayscale alpha masks remain untagged; RGB ICC does not apply to alpha values",
        },
        "models": {
            "sam2": sam_report,
            "grounding_dino": semantic_report,
            "ocr": ocr_report,
            "vitmatte": vitmatte_report,
        },
        "grouping": grouping_report,
        "matte_cleanup": matte_cleanup_report,
        "background_restoration": restoration.report,
        "hierarchy_cleanup": hierarchy_cleanup,
        "composition_qa": composition_report,
        "matte_qa": matte_qa,
        "render_resource_qa": render_resource_report,
        "layers": layer_records,
        "formats": {"psd": psd_report, "openraster": ora_report},
        "limitations": [
            "A flat bitmap contains no original layer graph; V5 infers useful raster groups.",
            "Pixels hidden behind an extracted object cannot be recovered exactly; the background is synthesized.",
            "OCR labels do not reconstruct the original font or create editable text objects.",
            "Very complex overlaps may need manual mask refinement in Photoshop, Krita, Photopea or Canva.",
        ],
        "total_seconds": elapsed,
    }
    assert_portable_bundle_manifest(manifest, output_dir)
    # LAYERS.zip is the portable raw-asset handoff, not a redundant archive of
    # the PSD, ORA and large QA previews already beside it in the bundle.
    archive_member_policy = (
        "00_HUONG_DAN_V5.txt + manifest + TEXT_OCR.json + "
        "cropped LAYERS/*.png + MASKS/*.png"
    )
    portable_manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
    portable_manifest["formats"] = {
        "layers_zip": {
            "member_policy": archive_member_policy,
            "manifest_scope": (
                "portable layer assets only; the outer bundle manifest adds PSD/ORA and ZIP hashes"
            ),
        }
    }
    write_json(manifest_path, portable_manifest)
    included = [
        manifest_path,
        guide_path,
        text_path,
        *sorted((output_dir / "LAYERS").glob("*.png")),
        *sorted((output_dir / "MASKS").glob("*.png")),
    ]
    zip_report = package_layers_zip(zip_path, output_dir, included)
    zip_report["member_policy"] = archive_member_policy
    manifest["formats"]["layers_zip"] = zip_report  # type: ignore[index]
    manifest["assets"] = asset_records(output_dir, exclude={manifest_path})
    write_json(manifest_path, manifest)

    print("\nV5 HOÀN TẤT", flush=True)
    print(f"  PSD chỉnh sửa : {psd_path.name if psd_report.get('created') else psd_report.get('reason')}")
    print(f"  ORA mở chuẩn  : {ora_path.name}")
    print(f"  ZIP layer PNG : {zip_path.name}")
    print(f"  Bản xem        : {preview_path.name}")
    print(f"  Số layer       : {len(rendered)} + nền")
    print(f"  Thời gian      : {elapsed:.1f} giây")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("V5 stopped by user.", file=sys.stderr)
        raise SystemExit(130)
