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
    package_layers_zip,
    render_layers,
    resize_rgb,
    save_color_png,
    sha256_file,
    write_json,
)
from v5lib.geometry import dilate_mask
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
    del masks, scores, candidates
    if not layers:
        raise RuntimeError(
            "V5 did not find a stable movable region. Keep the original image and try another source."
        )

    by_id = {layer.layer_id: layer for layer in layers}
    top_level_masks = [
        layer.mask for layer in layers if layer.metadata.get("parent_id") is None
    ]
    restoration = restore_background(
        source_rgb,
        top_level_masks,
        mode=args.inpaint,
    )
    background_rgb = resize_rgb(restoration.background, final_size)
    footprint_image = Image.fromarray(
        restoration.removal_footprint.astype(np.uint8) * 255,
        "L",
    )
    if footprint_image.size != final_size:
        footprint_image = footprint_image.resize(final_size, Image.Resampling.NEAREST)
    background_footprint_final = np.array(footprint_image, dtype=np.uint8) > 0
    # Lanczos can otherwise bleed synthesized pixels outside the declared
    # removal footprint.  The final clean base must equal the selected master
    # byte-for-byte everywhere V5 did not explicitly replace hidden content.
    background_rgb[~background_footprint_final] = master_rgb[~background_footprint_final]
    background_image = Image.fromarray(background_rgb, "RGB")

    restoration_radius = int(restoration.report["removal_radius_source_px"])
    support_masks: dict[str, np.ndarray] = {
        layer.layer_id: dilate_mask(layer.mask, restoration_radius) for layer in layers
    }

    def descendants(layer_id: str) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        queue = list(by_id[layer_id].metadata.get("children", []))
        while queue:
            child_id = str(queue.pop(0))
            child = by_id.get(child_id)
            if child is None:
                continue
            result.append(child.mask)
            queue.extend(child.metadata.get("children", []))
        return result

    layer_targets: dict[str, np.ndarray] = {}
    hierarchy_cleanup: list[dict[str, object]] = []
    for layer in layers:
        child_masks = descendants(layer.layer_id)
        if not child_masks:
            continue
        cleaned = restore_background(
            source_rgb,
            child_masks,
            mode="poster",
            progress=lambda _message: None,
        )
        cleaned_final = resize_rgb(cleaned.background, final_size)
        footprint_final = Image.fromarray(cleaned.removal_footprint.astype(np.uint8) * 255, "L")
        if footprint_final.size != final_size:
            footprint_final = footprint_final.resize(final_size, Image.Resampling.NEAREST)
        replace = np.array(footprint_final, dtype=np.uint8) > 0
        target_for_layer = master_rgb.copy()
        target_for_layer[replace] = cleaned_final[replace]
        layer_targets[layer.layer_id] = target_for_layer
        # The parent needs enough drawable support to place the synthesized
        # clean plate beneath its descendants.  Each descendant already has
        # its own same-radius support mask for reconstructing the visible edge,
        # glow or shadow when the stack is flattened.
        support_masks[layer.layer_id] |= cleaned.removal_footprint
        hierarchy_cleanup.append(
            {
                "layer_id": layer.layer_id,
                "descendant_masks_removed": len(child_masks),
                "cleanup_footprint_ratio": round(float(cleaned.removal_footprint.mean()), 6),
            }
        )

    print("  V5 bước 4/4: đang dựng PNG layer, PSD, OpenRaster và kiểm định...", flush=True)
    rendered, composite, composition_report = render_layers(
        master_rgb,
        background_rgb,
        layers,
        layer_targets=layer_targets,
        support_masks=support_masks,
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
    text_path = output_dir / "TEXT_OCR.json"
    manifest_path = output_dir / "manifest.json"

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
        "models": {"sam2": sam_report, "grounding_dino": semantic_report, "ocr": ocr_report},
        "grouping": grouping_report,
        "background_restoration": restoration.report,
        "hierarchy_cleanup": hierarchy_cleanup,
        "composition_qa": composition_report,
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
    archive_member_policy = "manifest + TEXT_OCR.json + cropped LAYERS/*.png + MASKS/*.png"
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
