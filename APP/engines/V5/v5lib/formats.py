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


Image.MAX_IMAGE_PIXELS = 500_000_000

PSD_MAX_DIMENSION = 30_000
PSD_SAFE_RAW_BYTES = 1_600_000_000
PSD_ROUNDTRIP_MAX_ERROR = 1
ORA_VERSION = "0.0.6"
ORA_MIMETYPE = b"image/openraster"
ORA_XML_MAX_BYTES = 16 * 1024 * 1024
ORA_MAX_MEMBERS = 4096
ORA_ROUNDTRIP_MAX_ERROR = 1
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


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


def resize_alpha(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    source_alpha = np.clip(make_soft_alpha(mask) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    image = Image.fromarray(source_alpha, "L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    array = np.array(image, dtype=np.float32) / 255.0
    array[array < 1.0 / 255.0] = 0.0
    return array


def _category_order(category: str) -> int:
    return {"detail_group": 0, "object": 1, "text_raster": 2}.get(category, 0)


def render_layers(
    master_rgb: np.ndarray,
    background_rgb: np.ndarray,
    specs: list[LayerSpec],
    *,
    layer_targets: dict[str, np.ndarray] | None = None,
    support_masks: dict[str, np.ndarray] | None = None,
) -> tuple[list[RenderedLayer], Image.Image, dict[str, object]]:
    """Render bottom-to-top RGBA layers and solve edge colours against what is below."""

    height, width = master_rgb.shape[:2]
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

    # A container cannot retain hierarchy and also keep children elsewhere in
    # the global z-order. Render parent and descendants contiguously so the PNG
    # preview, PSD groups, and ORA stacks share one bottom-to-top order.
    for root_spec in root_specs:
        emit(root_spec)
    if len(ordered_specs) != len(base_order):
        raise RuntimeError("V5 hierarchy did not produce a complete render order.")
    # Keep the working composite in the same 8-bit representation that the
    # exported PNG/PSD/ORA layers use. A float-only ideal can report zero error
    # while the actual saved layers differ visibly after alpha quantization.
    current_u8 = np.array(background_rgb, dtype=np.uint8, copy=True)
    target = master_rgb.astype(np.float32) / 255.0
    rendered: list[RenderedLayer] = []
    for spec in ordered_specs:
        core_alpha = resize_alpha(spec.mask, (width, height))
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
        base_alpha = core_alpha[top:bottom, left:right]
        crop_support = support[top:bottom, left:right]
        below_u8 = current_u8[top:bottom, left:right]
        below = below_u8.astype(np.float32) / 255.0
        desired_canvas = (layer_targets or {}).get(spec.layer_id, master_rgb)
        desired = desired_canvas[top:bottom, left:right].astype(np.float32) / 255.0
        delta = desired - below
        required_up = np.where(
            delta > 0,
            delta / np.maximum(1.0 - below, 1.0 / 255.0),
            0.0,
        )
        required_down = np.where(
            delta < 0,
            -delta / np.maximum(below, 1.0 / 255.0),
            0.0,
        )
        required_alpha = np.clip(
            np.maximum(required_up, required_down).max(axis=2), 0.0, 1.0
        )
        required_alpha[np.max(np.abs(delta), axis=2) < 0.5 / 255.0] = 0.0
        ideal_alpha = np.maximum(base_alpha, required_alpha * crop_support)
        # Ceil conservatively so quantization never drops below the minimum
        # alpha required to represent the requested delta.
        alpha_u8 = np.ceil(
            np.maximum(0.0, np.clip(ideal_alpha, 0.0, 1.0) * 255.0 - 1e-7)
        ).astype(np.uint8)
        crop_alpha = alpha_u8.astype(np.float32) / 255.0
        divisor = np.maximum(crop_alpha[..., None], 1.0 / 255.0)
        foreground = (desired - (1.0 - crop_alpha[..., None]) * below) / divisor
        foreground = np.clip(foreground, 0.0, 1.0)
        foreground_u8 = np.clip(foreground * 255.0 + 0.5, 0, 255).astype(np.uint8)
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
        "recomposition_max_abs_error": int(error.max()),
        "recomposition_mean_abs_error": round(float(error.mean()), 6),
        "recomposition_psnr_db": "infinite" if math.isinf(psnr) else round(psnr, 4),
        "edge_policy": (
            "semantic core alpha plus the exact source-space restoration footprint and conservative "
            "ceil-to-8-bit effect/shadow alpha; "
            "foreground colours solved against and recomposited into the actual 8-bit lower composite"
        ),
        "preview_basis": "actual cropped 8-bit RGBA assets composited bottom-to-top with Pillow",
        "layer_order_policy": "container-compatible depth-first bottom-to-top (parent, then descendants)",
    }
    return rendered, composite, report


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


def _expected_container_tree(
    background_size: tuple[int, int], rendered: list[RenderedLayer]
) -> list[dict[str, object]]:
    """Return the exact bottom-to-top tree both PSD and ORA must preserve."""

    width, height = background_size
    roots, children = _layer_tree(rendered)

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
                *(node(child) for child in node_children),
            ],
        }

    return [
        {
            "name": "00 BACKGROUND - SYNTHESIZED HIDDEN PIXELS",
            "kind": "pixel",
            "visible": True,
            "bbox": [0, 0, width, height],
        },
        *(node(item) for item in roots),
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
    expected_tree = _expected_container_tree(background.size, rendered)
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
            for child in node_children:
                add_node(child, group)
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

    for root_item in roots:
        add_node(root_item, psd)
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
) -> dict[str, object]:
    width, height = background.size
    expected_tree = _expected_container_tree(background.size, rendered)
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

    def add_ora_node(parent_xml: ET.Element, item: RenderedLayer) -> None:
        node_children = children.get(item.spec.layer_id, [])
        if node_children:
            group = ET.SubElement(
                parent_xml,
                "stack",
                {"name": f"GROUP - {item.spec.name}", "isolation": "auto"},
            )
            # Topmost XML child first; base is always at the bottom of its group.
            for child in reversed(node_children):
                add_ora_node(group, child)
            base_element = _ora_layer_element(item, source_by_id[item.spec.layer_id])
            base_element.set("name", f"BASE - {item.spec.name}")
            group.append(base_element)
        else:
            parent_xml.append(_ora_layer_element(item, source_by_id[item.spec.layer_id]))

    # OpenRaster stores the topmost root first.
    for root_item in reversed(roots):
        add_ora_node(stack, root_item)
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
        if width <= 0 or height <= 0 or width * height > Image.MAX_IMAGE_PIXELS:
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


def export_contact_sheet(
    path: Path,
    background: Image.Image,
    rendered: list[RenderedLayer],
    *,
    icc_profile: bytes | None = None,
) -> None:
    cards: list[tuple[str, Image.Image]] = [("BACKGROUND", background.convert("RGBA"))]
    cards.extend((item.spec.name, item.rgba) for item in rendered)
    columns = 4
    card_w, card_h, label_h = 300, 250, 42
    rows = math.ceil(len(cards) / columns)
    sheet = Image.new("RGB", (columns * card_w, rows * (card_h + label_h)), (38, 38, 42))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, (name, artwork) in enumerate(cards):
        column, row = index % columns, index // columns
        x, y = column * card_w, row * (card_h + label_h)
        checker = Image.new("RGB", (card_w, card_h), (220, 220, 220))
        checker_draw = ImageDraw.Draw(checker)
        for cy in range(0, card_h, 20):
            for cx in range(0, card_w, 20):
                if (cx // 20 + cy // 20) % 2:
                    checker_draw.rectangle((cx, cy, cx + 19, cy + 19), fill=(180, 180, 180))
        thumbnail = artwork.copy()
        thumbnail.thumbnail((card_w - 16, card_h - 16), Image.Resampling.LANCZOS)
        checker.paste(
            thumbnail,
            ((card_w - thumbnail.width) // 2, (card_h - thumbnail.height) // 2),
            thumbnail if thumbnail.mode == "RGBA" else None,
        )
        sheet.paste(checker, (x, y))
        draw.text((x + 8, y + card_h + 8), name[:34], fill="white", font=font)
    save_color_png(sheet, path, icc_profile)


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
