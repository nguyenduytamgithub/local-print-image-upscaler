"""Truthful V7 PNG, editable SVG and manifest output helpers.

An SVG containing an embedded bitmap is reported as mixed raster/vector.  V7
does not call a base64 image a vector conversion, and approved text remains a
real ``<text>`` element instead of a decorative bitmap or an uneditable trace.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageCms, UnidentifiedImageError


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
XML_NS = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


class FormatError(RuntimeError):
    """Raised when an output cannot be represented or validated honestly."""


Colour = str | tuple[int, int, int] | tuple[int, int, int, int]
RasterSource = Image.Image | np.ndarray | Path | str


@dataclass(frozen=True, slots=True)
class EditableText:
    text: str
    x: float
    y: float
    font_family: str
    font_size_px: float
    fill: Colour = (0, 0, 0)
    font_weight: str | int = "normal"
    font_style: str = "normal"
    stroke: Colour | None = None
    stroke_width_px: float = 0.0
    text_anchor: str = "start"
    line_height: float = 1.2
    rotation_degrees: float = 0.0
    language: str = "vi"
    direction: str = "ltr"
    region_id: str | None = None
    review_status: str = "approved"
    extra_attributes: Mapping[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VectorPrimitive:
    kind: str
    attributes: Mapping[str, str | int | float]
    primitive_id: str | None = None


@dataclass(frozen=True, slots=True)
class RasterLayer:
    source: RasterSource
    x: float = 0.0
    y: float = 0.0
    width: float | None = None
    height: float | None = None
    opacity: float = 1.0
    layer_id: str | None = None


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standard_srgb_profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _as_pil_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.copy()
    value = np.asarray(image)
    if value.dtype != np.uint8:
        raise FormatError("image arrays must use uint8 pixels.")
    if value.ndim == 2:
        return Image.fromarray(value, "L")
    if value.ndim == 3 and value.shape[2] in {3, 4}:
        return Image.fromarray(value, "RGB" if value.shape[2] == 3 else "RGBA")
    raise FormatError("image arrays must have shape HxW, HxWx3 or HxWx4.")


def save_png_atomic(
    image: Image.Image | np.ndarray,
    target: Path | str,
    *,
    icc_profile: bytes | None = None,
    dpi: tuple[float, float] | None = None,
    compress_level: int = 3,
) -> dict[str, object]:
    """Write, reopen and atomically publish a real PNG."""

    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not 0 <= int(compress_level) <= 9:
        raise FormatError("PNG compress_level must be from 0 through 9.")
    value = _as_pil_image(image)
    if value.mode not in {"RGB", "RGBA", "L", "LA"}:
        value = value.convert("RGBA" if "A" in value.getbands() else "RGB")
    profile = None if value.mode in {"L", "LA"} else (icc_profile or standard_srgb_profile())
    temporary = path.with_name(f".{path.name}.new-{uuid.uuid4().hex}")
    save_options: dict[str, object] = {
        "format": "PNG",
        "compress_level": int(compress_level),
    }
    if profile is not None:
        save_options["icc_profile"] = profile
    if dpi is not None:
        x_dpi, y_dpi = float(dpi[0]), float(dpi[1])
        if not all(math.isfinite(item) and 1.0 <= item <= 10_000.0 for item in (x_dpi, y_dpi)):
            raise FormatError("PNG DPI values must be finite and from 1 through 10,000.")
        save_options["dpi"] = (x_dpi, y_dpi)
    try:
        value.save(temporary, **save_options)
        with Image.open(temporary) as reopened:
            reopened.load()
            if reopened.size != value.size:
                raise FormatError(
                    f"PNG round-trip size {reopened.size} does not match {value.size}."
                )
            if reopened.format != "PNG":
                raise FormatError("published file is not a PNG container.")
            if profile is not None and reopened.info.get("icc_profile") != profile:
                raise FormatError("PNG round-trip lost or changed the sRGB ICC profile.")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "name": path.name,
        "format": "PNG",
        "size": list(value.size),
        "mode": value.mode,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "icc_profile_embedded": profile is not None,
    }


def _colour(value: Colour) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or any(character in stripped for character in '<>"\''):
            raise FormatError(f"unsafe or empty SVG colour: {value!r}")
        return stripped
    channels = tuple(int(channel) for channel in value)
    if len(channels) not in {3, 4} or any(channel < 0 or channel > 255 for channel in channels):
        raise FormatError("SVG colour tuples must be RGB/RGBA bytes.")
    if len(channels) == 3 or channels[3] == 255:
        return f"rgb({channels[0]},{channels[1]},{channels[2]})"
    return f"rgba({channels[0]},{channels[1]},{channels[2]},{channels[3] / 255:.6f})"


def _number(value: str | int | float) -> str:
    if isinstance(value, str):
        return value
    number = float(value)
    if not math.isfinite(number):
        raise FormatError("SVG numeric attributes must be finite.")
    return f"{number:.8g}"


def _safe_extra_attribute(
    key: object,
    value: str | int | float,
    *,
    element: str,
) -> tuple[str, str]:
    """Allow declarative SVG attributes but reject active/external content."""

    name = str(key).strip()
    folded = name.casefold()
    if (
        not name
        or any(character.isspace() or character in '<>"\'' for character in name)
        or folded.startswith("on")
        or folded in {"href", "xlink:href", "style", "xmlns", "xmlns:xlink"}
    ):
        raise FormatError(f"unsafe SVG {element} attribute: {name or key!r}")
    rendered = _number(value)
    lowered = rendered.casefold().replace(" ", "")
    if (
        any(ord(character) < 32 and character not in "\t\n\r" for character in rendered)
        or "javascript:" in lowered
        or "url(" in lowered
        or "data:text/html" in lowered
    ):
        raise FormatError(f"unsafe SVG {element} attribute value for {name!r}")
    return name, rendered


def _raster_png_bytes(source: RasterSource) -> tuple[bytes, tuple[int, int]]:
    if isinstance(source, (str, Path)):
        try:
            with Image.open(Path(source)) as image:
                image.load()
                value = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise FormatError(f"cannot open SVG raster layer: {source}") from exc
    else:
        value = _as_pil_image(source)
        if value.mode not in {"RGB", "RGBA"}:
            value = value.convert("RGBA" if "A" in value.getbands() else "RGB")
    stream = io.BytesIO()
    value.save(stream, format="PNG", compress_level=3)
    return stream.getvalue(), value.size


def _append_raster(
    parent: ET.Element,
    layer: RasterLayer,
    index: int,
    canvas_size: tuple[int, int],
) -> None:
    payload, natural_size = _raster_png_bytes(layer.source)
    width = float(layer.width if layer.width is not None else natural_size[0])
    height = float(layer.height if layer.height is not None else natural_size[1])
    if width <= 0 or height <= 0 or not 0.0 <= float(layer.opacity) <= 1.0:
        raise FormatError("SVG raster dimensions/opacity are invalid.")
    element = ET.SubElement(
        parent,
        f"{{{SVG_NS}}}image",
        {
            "id": layer.layer_id or f"raster-{index + 1}",
            "x": _number(layer.x),
            "y": _number(layer.y),
            "width": _number(width),
            "height": _number(height),
            "opacity": _number(layer.opacity),
            "preserveAspectRatio": "none",
            "data-content-kind": "embedded-raster-png",
        },
    )
    uri = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
    element.set("href", uri)
    element.set(f"{{{XLINK_NS}}}href", uri)
    if layer.width is None and layer.height is None and natural_size != canvas_size:
        element.set("data-natural-size", f"{natural_size[0]}x{natural_size[1]}")


def _append_primitive(parent: ET.Element, primitive: VectorPrimitive, index: int) -> None:
    kind = primitive.kind.casefold()
    if kind not in {"rect", "line", "ellipse", "circle", "path", "polygon", "polyline"}:
        raise FormatError(f"unsupported SVG primitive: {primitive.kind}")
    attributes = dict(
        _safe_extra_attribute(key, value, element="primitive")
        for key, value in primitive.attributes.items()
    )
    attributes.setdefault("id", primitive.primitive_id or f"primitive-{index + 1}")
    attributes["data-content-kind"] = "editable-vector-primitive"
    ET.SubElement(parent, f"{{{SVG_NS}}}{kind}", attributes)


def _append_text(parent: ET.Element, item: EditableText, index: int) -> None:
    if not item.text:
        raise FormatError("editable SVG text cannot be empty.")
    if item.font_size_px <= 0 or item.line_height <= 0:
        raise FormatError("editable SVG font size and line height must be positive.")
    if item.text_anchor not in {"start", "middle", "end"}:
        raise FormatError("SVG text_anchor must be start, middle or end.")
    attributes = {
        "id": item.region_id or f"text-{index + 1}",
        "x": _number(item.x),
        "y": _number(item.y),
        "font-family": item.font_family,
        "font-size": _number(item.font_size_px),
        "font-weight": str(item.font_weight),
        "font-style": item.font_style,
        "fill": _colour(item.fill),
        "text-anchor": item.text_anchor,
        "direction": item.direction,
        "lang": item.language,
        "data-content-kind": "editable-unicode-text",
        "data-review-status": item.review_status,
        "data-font-embedded": "false",
        f"{{{XML_NS}}}space": "preserve",
    }
    if item.stroke is not None and item.stroke_width_px > 0:
        attributes.update(
            {
                "stroke": _colour(item.stroke),
                "stroke-width": _number(item.stroke_width_px),
                "paint-order": "stroke fill",
                "stroke-linejoin": "round",
            }
        )
    if item.rotation_degrees:
        attributes["transform"] = (
            f"rotate({_number(item.rotation_degrees)} {_number(item.x)} {_number(item.y)})"
        )
    for key, value in item.extra_attributes.items():
        safe_key, safe_value = _safe_extra_attribute(key, value, element="text")
        attributes[safe_key] = safe_value
    element = ET.SubElement(parent, f"{{{SVG_NS}}}text", attributes)
    lines = item.text.split("\n")
    if len(lines) == 1:
        element.text = item.text
        return
    for line_index, line in enumerate(lines):
        tspan = ET.SubElement(
            element,
            f"{{{SVG_NS}}}tspan",
            {
                "x": _number(item.x),
                "dy": "0" if line_index == 0 else _number(item.font_size_px * item.line_height),
            },
        )
        tspan.text = line


def validate_editable_svg(path: Path | str) -> dict[str, object]:
    source = Path(path)
    try:
        tree = ET.parse(source)
    except (ET.ParseError, OSError) as exc:
        raise FormatError(f"SVG cannot be reopened: {source}") from exc
    root = tree.getroot()
    if root.tag != f"{{{SVG_NS}}}svg":
        raise FormatError("output root is not an SVG element.")
    images = root.findall(f".//{{{SVG_NS}}}image")
    texts = root.findall(f".//{{{SVG_NS}}}text")
    primitive_names = ("rect", "line", "ellipse", "circle", "path", "polygon", "polyline")
    primitives = [
        element
        for name in primitive_names
        for element in root.findall(f".//{{{SVG_NS}}}{name}")
    ]
    for image in images:
        href = image.get("href") or image.get(f"{{{XLINK_NS}}}href")
        if not href or not href.startswith("data:image/png;base64,"):
            raise FormatError("SVG raster layer is not a portable embedded PNG.")
    for text in texts:
        if text.get("data-content-kind") != "editable-unicode-text":
            raise FormatError("SVG contains an undeclared/non-editable text element.")
    return {
        "format": "SVG 2 mixed-content master",
        "raster_image_count": len(images),
        "editable_text_count": len(texts),
        "vector_primitive_count": len(primitives),
        "full_vector": len(images) == 0,
        "mixed_raster_vector": len(images) > 0 and (len(texts) + len(primitives)) > 0,
        "font_files_embedded": 0,
        "truthful_content_declarations": True,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def export_editable_svg(
    target: Path | str,
    canvas_size: tuple[int, int],
    *,
    background: RasterSource | None = None,
    raster_layers: Sequence[RasterLayer] = (),
    texts: Sequence[EditableText] = (),
    primitives: Sequence[VectorPrimitive] = (),
    metadata: Mapping[str, object] | None = None,
    title: str = "V7 editable repair master",
) -> dict[str, object]:
    """Export real editable text/primitives and declare raster content honestly."""

    width, height = (int(canvas_size[0]), int(canvas_size[1]))
    if width < 1 or height < 1:
        raise FormatError("SVG canvas must have positive dimensions.")
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rasters = ([RasterLayer(background, width=width, height=height, layer_id="background-raster")]
                   if background is not None else []) + list(raster_layers)
    claim = (
        "mixed_raster_vector_editable"
        if all_rasters and (texts or primitives)
        else "raster_container"
        if all_rasters
        else "editable_vector_content"
    )
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
            "version": "2.0",
            "data-v7-content-claim": claim,
        },
    )
    ET.SubElement(root, f"{{{SVG_NS}}}title").text = title
    metadata_value: dict[str, object] = {
        "pipeline": "V7_DESIGN_REPAIR",
        "content_claim": claim,
        "raster_layers": len(all_rasters),
        "editable_text_objects": len(texts),
        "vector_primitives": len(primitives),
        "font_files_embedded": 0,
        "notice": (
            "Embedded images remain raster. Approved text is a real editable Unicode SVG element; "
            "its named font must be installed on the editing machine."
        ),
    }
    if metadata:
        metadata_value["user_metadata"] = dict(metadata)
    ET.SubElement(root, f"{{{SVG_NS}}}metadata").text = json.dumps(
        metadata_value, ensure_ascii=False, sort_keys=True, allow_nan=False
    )

    if all_rasters:
        group = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id": "raster-layers"})
        for index, layer in enumerate(all_rasters):
            _append_raster(group, layer, index, (width, height))
    if primitives:
        group = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id": "vector-primitives"})
        for index, primitive in enumerate(primitives):
            _append_primitive(group, primitive, index)
    if texts:
        group = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id": "editable-text"})
        for index, text in enumerate(texts):
            _append_text(group, text, index)

    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    temporary = path.with_name(f".{path.name}.new-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(payload)
        report = validate_editable_svg(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    # Recompute against the published path; temporary names must never leak.
    report = validate_editable_svg(path)
    report.update(
        {
            "path": str(path),
            "name": path.name,
            "canvas_size": [width, height],
            "content_claim": claim,
        }
    )
    return report


def manifest_asset_record(path: Path | str, bundle_root: Path | str) -> dict[str, object]:
    asset = Path(path).resolve()
    root = Path(bundle_root).resolve()
    try:
        relative = asset.relative_to(root)
    except ValueError as exc:
        raise FormatError(f"manifest asset is outside its bundle: {asset}") from exc
    if not asset.is_file() or asset.stat().st_size == 0:
        raise FormatError(f"manifest asset is missing or empty: {asset}")
    return {
        "path": relative.as_posix(),
        "bytes": asset.stat().st_size,
        "sha256": sha256_file(asset),
    }


def build_bundle_manifest(
    *,
    pipeline: str,
    app_version: str,
    source_name: str,
    source_sha256: str,
    source_size: tuple[int, int],
    scale: float,
    final_size: tuple[int, int],
    bundle_root: Path | str,
    assets: Sequence[Path | str],
    review_required: bool,
    qa: Mapping[str, object] | None = None,
    models: Mapping[str, object] | None = None,
    limitations: Sequence[str] = (),
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a portable, relative-path manifest for an atomic V7 bundle."""

    if not pipeline or not source_name or not source_sha256:
        raise FormatError("pipeline, source name and source SHA-256 are required.")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise FormatError("manifest scale must be finite and positive.")
    source_dimensions = (int(source_size[0]), int(source_size[1]))
    final_dimensions = (int(final_size[0]), int(final_size[1]))
    if min(*source_dimensions, *final_dimensions) < 1:
        raise FormatError("manifest source and final dimensions must be positive.")
    root = Path(bundle_root)
    manifest: dict[str, object] = {
        "pipeline": pipeline,
        "schema_version": 1,
        "app_version": app_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": source_name,
            "sha256": source_sha256,
            "size": list(source_dimensions),
        },
        "scale": float(scale),
        "final_size": list(final_dimensions),
        "review_required": bool(review_required),
        "path_policy": "bundle-relative assets only",
        "assets": [manifest_asset_record(path, root) for path in assets],
        "qa": dict(qa or {}),
        "models": dict(models or {}),
        "limitations": list(limitations),
    }
    if extra:
        manifest["details"] = dict(extra)
    # Validate serialisability now, before the caller reaches publication.
    json.dumps(manifest, ensure_ascii=False, allow_nan=False)
    return manifest


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def write_manifest_atomic(
    manifest: Mapping[str, object],
    target: Path | str,
) -> dict[str, object]:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(manifest),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        default=_json_default,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.new-{uuid.uuid4().hex}")
    try:
        temporary.write_text(payload, encoding="utf-8")
        reopened = json.loads(temporary.read_text(encoding="utf-8"))
        if not isinstance(reopened, dict):
            raise FormatError("manifest root must be a JSON object.")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


# Backward-friendly aliases for an engine that prefers shorter verbs.
save_color_png = save_png_atomic
write_json_atomic = write_manifest_atomic


__all__ = [
    "EditableText",
    "FormatError",
    "RasterLayer",
    "VectorPrimitive",
    "build_bundle_manifest",
    "export_editable_svg",
    "manifest_asset_record",
    "save_color_png",
    "save_png_atomic",
    "sha256_file",
    "standard_srgb_profile",
    "validate_editable_svg",
    "write_json_atomic",
    "write_manifest_atomic",
]
