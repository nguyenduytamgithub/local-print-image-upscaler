"""V4 PRINT: build an honest hybrid or full-vector master for large-format print.

Hybrid mode combines a disclosed V3 AI raster layer with real vector paths.
Full-vector mode remains available for flat artwork where posterisation is safe.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as element_tree
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pikepdf
import vtracer
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError


Image.MAX_IMAGE_PIXELS = 500_000_000
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
element_tree.register_namespace("", SVG_NS)
element_tree.register_namespace("xlink", XLINK_NS)

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parents[1]
ROOT_DIR = APP_DIR.parent
SCRIPT_EXPORT_PDFX4 = HERE / "scribus_export_pdfx4.py"

TRACE_SCALE = 2
MAX_SOURCE_MEGAPIXELS = 12.0
MAX_OUTPUT_MEGAPIXELS = 500.0
MAX_PATHS = 150_000
MAX_COMMANDS = 2_000_000
MAX_SVG_BYTES = 256 * 1024 * 1024
MAX_PDF_PAGE_MM = 5_000.0
PDF_POINTS_PER_MM = 72.0 / 25.4
PDF_BOX_TOLERANCE_PT = 1.0

DEFAULT_OUTPUT_PROFILE = "ISO Coated v2 300% (basICColor)"
DEFAULT_RGB_PROFILE = "sRGB IEC61966-2.1"

GMIC_PARAMS = {
    "command": "upscale_smart",
    "width": "200%",
    "height": "200%",
    "depth": "100%",
    "smoothness": 1.0,
    "anisotropy": 0.75,
    "sharpening": 25.0,
}

VTRACER_PARAMS = {
    "colormode": "color",
    "hierarchical": "stacked",
    "mode": "spline",
    "filter_speckle": 5,
    "color_precision": 8,
    "layer_difference": 4,
    "corner_threshold": 60,
    "length_threshold": 5.0,
    "max_iterations": 10,
    "splice_threshold": 45,
    "path_precision": 5,
}

QA_LIMITS = {
    "min_psnr_db": 22.0,
    "min_edge_f1": 0.90,
    "max_abs_error_p95": 35.0,
}

HYBRID_SOURCE_QA_LIMITS = {
    "min_psnr_db": 30.0,
    "min_edge_f1": 0.97,
    "max_abs_error_p95": 16.0,
}

HYBRID_RETENTION_LIMITS = {
    "min_psnr_db": 35.0,
    "min_edge_f1": 0.98,
    "max_abs_error_p95": 8.0,
}

# Direct V3 -> V4 acceptance gate. These are conservative engineering
# thresholds, not an ISO printing standard. A restoration must retain the V3
# content and improve at least two independent sharpness indicators; one noisy
# metric alone is not enough to claim that V4 is better.
COMPARATIVE_QA_LIMITS = {
    "min_ssim_mean": 0.985,
    "min_edge_f1": 0.98,
    "min_acutance_retention_ratio": 0.98,
    "min_gradient_p95_retention_ratio": 0.90,
    "min_acutance_ratio": 1.003,
    "min_tenengrad_ratio": 1.01,
    "min_laplacian_variance_ratio": 1.05,
    "min_gradient_p95_ratio": 1.01,
    "min_crop_gate_pass_fraction": 0.75,
    "max_acutance_ratio": 1.25,
    "max_tenengrad_ratio": 1.30,
    "max_laplacian_variance_ratio": 1.50,
    "max_gradient_p95_ratio": 1.25,
    "max_edge_overshoot_p99_255": 8.0,
    "max_abs_difference_p95": 12.0,
}
COMPARATIVE_QA_FULL_FRAME_MAX_PIXELS = 9_437_184
COMPARATIVE_QA_TILE_SIZE = 384
COMPARATIVE_QA_GRID_SIZE = 8
GAUSSIAN_TRUNCATE = 3.0
UNIFORM_RESTORATION_CANDIDATES = (
    (0.0, 0.06, 1.5),
    (0.10, 0.10, 1.5),
    (0.075, 0.06, 1.5),
    (0.05, 0.07, 1.0),
    (0.025, 0.06, 1.5),
)
RESTORATION_STRIPE_HEIGHT = 256
DEEP_MATERIAL_MIN_CROP_GAIN = 0.02


class V4Error(RuntimeError):
    """Expected V4 validation or toolchain error."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V4 PRINT hybrid/vector reconstruction")
    parser.add_argument("input", type=Path)
    parser.add_argument("scale", type=float, help="PNG proof scale from x2 to x20")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--name", required=True, help="Safe output filename stem")
    parser.add_argument("--width-mm", type=float, help="Intended final print width in millimetres")
    parser.add_argument("--bleed-mm", type=float, default=0.0)
    parser.add_argument("--profile-name", default=DEFAULT_OUTPUT_PROFILE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--raster-base", type=Path, help="V3 AI raster layer for hybrid print mode")
    mode.add_argument("--full-vector", action="store_true", help="Forbid raster layers")
    parser.add_argument(
        "--v3-baseline",
        type=Path,
        help="Validated V3 native baseline used for direct V3-to-V4 quality gating",
    )
    parser.add_argument(
        "--vector-opacity",
        type=float,
        default=0.0,
        help="Visible vector overlay in hybrid mode; 0 keeps the edit layer non-printing.",
    )
    parser.add_argument("--allow-huge", action="store_true")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def find_pinned_tool(
    bundled: Path,
    fallbacks: list[Path],
    label: str,
    version_arguments: list[str],
    version_pattern: str,
) -> Path:
    if bundled.is_file():
        return bundled.resolve()
    incompatible: list[str] = []
    for candidate in fallbacks:
        if not candidate.is_file():
            continue
        try:
            completed = subprocess.run(
                [str(candidate), *version_arguments],
                text=True,
                capture_output=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            incompatible.append(str(candidate))
            continue
        version_output = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode == 0 and re.search(version_pattern, version_output):
            return candidate.resolve()
        incompatible.append(str(candidate))
    detail = f" Incompatible candidates: {', '.join(incompatible)}." if incompatible else ""
    raise V4Error(
        f"Missing compatible {label}; run V4 setup/doctor before processing images.{detail}"
    )


def find_gmic() -> Path:
    bundled = APP_DIR / "shared" / "tools" / "gmic" / "gmic-4.0.2-cli-win64" / "gmic.exe"
    if bundled.is_file():
        return bundled.resolve()
    fallbacks = [
        APP_DIR / "shared" / "tools" / "gmic" / "gmic.exe",
    ]
    system = shutil.which("gmic")
    if system:
        fallbacks.append(Path(system))
    return find_pinned_tool(
        bundled,
        fallbacks,
        "G'MIC 4.0.2",
        ["-version"],
        r"(?<![\d.])4\.0\.2(?![\d.])",
    )


def find_resvg() -> Path:
    bundled = APP_DIR / "shared" / "tools" / "resvg" / "0.47.0" / "resvg.exe"
    if bundled.is_file():
        return bundled.resolve()
    fallbacks = [
        APP_DIR / "shared" / "tools" / "resvg" / "resvg.exe",
    ]
    system = shutil.which("resvg")
    if system:
        fallbacks.append(Path(system))
    return find_pinned_tool(
        bundled,
        fallbacks,
        "resvg 0.47.0",
        ["--version"],
        r"(?<![\d.])0\.47\.0(?![\d.])",
    )


def find_scribus() -> Path:
    bundled = APP_DIR / "shared" / "tools" / "scribus" / "1.6.6" / "Scribus.exe"
    if bundled.is_file():
        return bundled.resolve()
    fallbacks: list[Path] = []
    system = shutil.which("scribus") or shutil.which("Scribus")
    if system:
        fallbacks.append(Path(system))
    fallbacks.extend(
        [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Scribus 1.6.6" / "Scribus.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Scribus 1.6" / "Scribus.exe",
        ]
    )
    return find_pinned_tool(
        bundled,
        fallbacks,
        "Scribus 1.6.6 with PDF/X-4 support",
        ["--version"],
        r"(?<![\d.])1\.6\.6(?![\d.])",
    )


def standard_srgb_profile() -> bytes:
    candidates = [
        APP_DIR / "shared" / "tools" / "scribus" / "1.6.6" / "share" / "profiles" / "sRGB.icm",
        Path(r"C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_bytes()
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def inspect_and_normalize_source(source: Path, normalized: Path) -> dict[str, Any]:
    try:
        with Image.open(source) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise V4Error("Animated or multi-frame input is not supported by V4.")
            image.seek(0)
            source_mode = image.mode
            embedded_icc = image.info.get("icc_profile")
            dpi_value = image.info.get("dpi")
            image = ImageOps.exif_transpose(image)
            image.load()

            has_alpha = (
                image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info
            )
            alpha = image.convert("RGBA").getchannel("A") if has_alpha else None
            colour = image.convert("RGB") if has_alpha else image
            colour_conversion = "untagged_assumed_srgb"
            if embedded_icc:
                try:
                    rgb = ImageCms.profileToProfile(
                        colour,
                        ImageCms.ImageCmsProfile(BytesIO(embedded_icc)),
                        ImageCms.createProfile("sRGB"),
                        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                        outputMode="RGB",
                    )
                    colour_conversion = "embedded_icc_to_srgb"
                except (ImageCms.PyCMSError, OSError, ValueError):
                    rgb = colour.convert("RGB")
                    colour_conversion = "invalid_icc_fallback_to_srgb_assumption"
            else:
                rgb = colour.convert("RGB")
            if alpha is not None:
                rgb = Image.composite(rgb, Image.new("RGB", rgb.size, "white"), alpha)
            source_size = rgb.size
            save_args: dict[str, Any] = {
                "format": "PNG",
                "compress_level": 3,
                "icc_profile": standard_srgb_profile(),
            }
            if dpi_value:
                save_args["dpi"] = dpi_value
            rgb.save(normalized, **save_args)
    except (UnidentifiedImageError, OSError) as exc:
        raise V4Error(f"Cannot read input image: {source}") from exc

    dpi_x = 96.0
    if dpi_value and len(dpi_value) >= 1:
        candidate = float(dpi_value[0])
        if 10.0 <= candidate <= 2400.0:
            dpi_x = candidate
    return {
        "size": source_size,
        "mode": source_mode,
        "dpi_x": dpi_x,
        "input_icc_present": bool(embedded_icc),
        "colour_conversion": colour_conversion,
        "alpha_composited_on_white": bool(alpha is not None),
        "exif_orientation_applied": True,
    }


def preprocess_smart2x(gmic: Path, normalized: Path, smart2x: Path) -> dict[str, Any]:
    started = time.perf_counter()
    command = [
        str(gmic),
        str(normalized),
        "-upscale_smart",
        "200%,200%,100%,1,0.75,25",
        "-cut",
        "0,255",
        "-output",
        str(smart2x),
    ]
    subprocess.run(command, check=True)
    with Image.open(normalized) as source_image, Image.open(smart2x) as result:
        expected = (source_image.width * TRACE_SCALE, source_image.height * TRACE_SCALE)
        if result.size != expected:
            raise V4Error(f"G'MIC produced {result.size}; expected {expected}.")
    return {"seconds": round(time.perf_counter() - started, 3), "command": command}


def trace_full_vector(smart2x: Path, raw_svg: Path) -> dict[str, Any]:
    started = time.perf_counter()
    vtracer.convert_image_to_svg_py(str(smart2x), str(raw_svg), **VTRACER_PARAMS)
    if not raw_svg.is_file() or raw_svg.stat().st_size == 0:
        raise V4Error("VTracer did not produce an SVG.")
    return {"seconds": round(time.perf_counter() - started, 3)}


def normalize_svg(
    raw_svg: Path,
    master_svg: Path,
    source_size: tuple[int, int],
    page_width_mm: float,
    page_height_mm: float,
    intended_width_mm: float,
    print_scale_denominator: int,
    title: str,
    raster_base: Path | None = None,
    vector_opacity: float = 1.0,
    raster_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hybrid = raster_base is not None
    if not (0.0 <= vector_opacity <= 1.0):
        raise V4Error("Vector opacity must be between 0 and 1.")
    if raster_provenance is not None and not hybrid:
        raise V4Error("Raster provenance is valid only for a hybrid SVG.")

    provenance = dict(raster_provenance or {})
    raster_selection = str(provenance.get("selection", "v4_deep_candidate"))
    hybrid_modes = {
        "v4_deep_candidate": (
            "hybrid-deep-ai-vector",
            "V4_DEEP_AI_PRINT_LAYER",
            (
                "V4 Deep Print master: disclosed two-pass AI raster layer plus an "
                "independent G'MIC/VTracer vector editing layer. The edit layer is "
                "non-printing by default because automatic trace can posterize gradients."
            ),
        ),
        "v4_uniform_restoration": (
            "hybrid-v4-uniform-restoration-vector",
            "V4_UNIFORM_RESTORATION_PRINT_LAYER",
            (
                "V4 uniform restoration print master: a spatially uniform, QA-selected "
                "V3/Deep blend with a finite-kernel V3 unsharp term, plus an independent "
                "G'MIC/VTracer vector editing layer. No local masks or patchwork are used."
            ),
        ),
        "v4_uniform_v3_usm": (
            "hybrid-v4-uniform-v3-usm-vector",
            "V4_UNIFORM_V3_USM_PRINT_LAYER",
            (
                "V4 uniform guarded-USM print master derived from the validated V3 AI "
                "master. The Deep raster was retained only as a diagnostic ablation and "
                "is not visible, mixed or embedded in the print layer."
            ),
        ),
        "v3_baseline_fallback": (
            "hybrid-v3-baseline-fallback-vector",
            "V3_BASELINE_FALLBACK_PRINT_LAYER",
            (
                "V3 baseline fallback print master: direct native-resolution QA rejected "
                "the newer candidate, so the validated V3 raster is the sole raster source "
                "and is uniformly resized, with an independent G'MIC/VTracer vector editing layer."
            ),
        ),
        "v4_candidate_unverified": (
            "hybrid-v4-candidate-unverified-vector",
            "V4_UNVERIFIED_CANDIDATE_PRINT_LAYER",
            (
                "Unverified hybrid print master: no validated V3 baseline was available "
                "for direct comparison. The raster is not claimed to improve on V3."
            ),
        ),
    }
    if hybrid and raster_selection not in hybrid_modes:
        raise V4Error(f"Unknown hybrid raster provenance selection: {raster_selection}")
    if hybrid:
        svg_mode, raster_layer_id, hybrid_description = hybrid_modes[raster_selection]
    else:
        svg_mode = "full-vector"
        raster_layer_id = ""
        hybrid_description = ""

    raw_root = element_tree.parse(raw_svg).getroot()
    root_attributes = {
        "version": "1.1",
        "width": f"{page_width_mm:.6f}mm",
        "height": f"{page_height_mm:.6f}mm",
        "viewBox": f"0 0 {source_size[0]} {source_size[1]}",
        "preserveAspectRatio": "xMidYMid meet",
        "shape-rendering": "geometricPrecision",
        "data-v4-mode": svg_mode,
        "data-intended-width-mm": f"{intended_width_mm:.6f}",
        "data-print-scale": f"1:{print_scale_denominator}",
    }
    if hybrid:
        root_attributes["data-raster-selection"] = raster_selection
    new_root = element_tree.Element(
        f"{{{SVG_NS}}}svg",
        root_attributes,
    )
    title_node = element_tree.SubElement(new_root, f"{{{SVG_NS}}}title")
    title_node.text = (
        f"{title} [V3 baseline fallback selected]"
        if hybrid and raster_selection == "v3_baseline_fallback"
        else title
    )
    description = element_tree.SubElement(new_root, f"{{{SVG_NS}}}desc")
    description.text = (
        hybrid_description
        if hybrid
        else (
            "V4 full-vector reconstruction: G'MIC Smart Upscale 2x and "
            "VTracer stacked splines. Contains no embedded raster image."
        )
    )
    metadata = element_tree.SubElement(new_root, f"{{{SVG_NS}}}metadata")
    metadata.text = json.dumps(
        {
            "engine": "V4_PRINT",
            "mode": svg_mode,
            "trace_scale": TRACE_SCALE,
            "vector_opacity": vector_opacity,
            "print_scale": f"1:{print_scale_denominator}",
            "intended_width_mm": intended_width_mm,
            "raster_selection": raster_selection if hybrid else None,
            "raster_provenance": provenance if hybrid else None,
        },
        separators=(",", ":"),
        default=str,
    )

    raster_info: dict[str, Any] | None = None
    if raster_base is not None:
        raster_base = raster_base.resolve()
        if not raster_base.is_file():
            raise V4Error(f"Hybrid raster base does not exist: {raster_base}")
        with Image.open(raster_base) as image:
            image.load()
            raster_size = image.size
            raster_mode = image.mode
        if raster_size[0] * source_size[1] != raster_size[1] * source_size[0]:
            raise V4Error(
                f"Hybrid raster aspect ratio {raster_size} does not match source {source_size}."
            )
        encoded = base64.b64encode(raster_base.read_bytes()).decode("ascii")
        element_tree.SubElement(
            new_root,
            f"{{{SVG_NS}}}image",
            {
                "id": raster_layer_id,
                "x": "0",
                "y": "0",
                "width": str(source_size[0]),
                "height": str(source_size[1]),
                "preserveAspectRatio": "none",
                f"{{{XLINK_NS}}}href": f"data:image/png;base64,{encoded}",
                "data-pixel-width": str(raster_size[0]),
                "data-pixel-height": str(raster_size[1]),
                "data-raster-selection": raster_selection,
            },
        )
        raster_info = {
            "id": raster_layer_id,
            "pixel_size": list(raster_size),
            "mode": raster_mode,
            "sha256": sha256_file(raster_base),
            "bytes": raster_base.stat().st_size,
            "selection": raster_selection,
            "provenance": provenance,
        }

    group_attributes = {
        "id": "V4_FULL_VECTOR_PATHS",
        "transform": f"scale({1 / TRACE_SCALE:g})",
    }
    if hybrid:
        group_attributes["opacity"] = f"{vector_opacity:g}"
    drawing_group = element_tree.SubElement(
        new_root,
        f"{{{SVG_NS}}}g",
        group_attributes,
    )
    for child in list(raw_root):
        name = local_name(child.tag)
        if name in {"title", "desc", "metadata"}:
            continue
        if name == "defs":
            new_root.insert(3, child)
        else:
            drawing_group.append(child)

    tree = element_tree.ElementTree(new_root)
    element_tree.indent(tree, space="  ")
    tree.write(master_svg, encoding="utf-8", xml_declaration=True)

    parsed = element_tree.parse(master_svg).getroot()
    image_count = sum(1 for item in parsed.iter() if local_name(item.tag) == "image")
    expected_images = 1 if hybrid else 0
    if image_count != expected_images:
        raise V4Error(
            f"V4 master contains {image_count} raster image(s); expected {expected_images}."
        )
    for item in parsed.iter():
        if local_name(item.tag) != "image":
            continue
        href = item.get("href") or item.get(f"{{{XLINK_NS}}}href") or ""
        if not href.startswith("data:image/png;base64,"):
            raise V4Error("V4 SVG contains an external or non-PNG raster reference.")
    paths = [item for item in parsed.iter() if local_name(item.tag) == "path"]
    command_count = sum(len(re.findall(r"[MmLlHhVvCcSsQqTtAaZz]", item.get("d", ""))) for item in paths)
    colours = {item.get("fill") for item in paths if item.get("fill")}
    size_bytes = master_svg.stat().st_size
    if len(paths) == 0:
        raise V4Error("V4 master contains no vector path.")
    if len(paths) > MAX_PATHS:
        raise V4Error(f"Vector path limit exceeded: {len(paths):,} > {MAX_PATHS:,}.")
    if command_count > MAX_COMMANDS:
        raise V4Error(f"Vector command limit exceeded: {command_count:,} > {MAX_COMMANDS:,}.")
    if size_bytes > MAX_SVG_BYTES:
        raise V4Error(f"SVG size limit exceeded: {size_bytes / 1024**2:.1f} MiB.")
    return {
        "path_count": len(paths),
        "command_count": command_count,
        "fill_colour_count": len(colours),
        "embedded_image_count": image_count,
        "mode": svg_mode,
        "vector_opacity": vector_opacity,
        "raster_layer": raster_info,
        "size_bytes": size_bytes,
    }


def render_svg(resvg: Path, svg: Path, png: Path, size: tuple[int, int]) -> dict[str, Any]:
    started = time.perf_counter()
    command = [
        str(resvg),
        "--quiet",
        "--shape-rendering",
        "geometricPrecision",
        "-w",
        str(size[0]),
        "-h",
        str(size[1]),
        str(svg),
        str(png),
    ]
    subprocess.run(command, check=True)
    with Image.open(png) as image:
        image.load()
        if image.size != size:
            raise V4Error(f"resvg produced {image.size}; expected {size}.")
        rgb = image.convert("RGB")
        temporary = png.with_name(png.name + ".icc.png")
        rgb.save(
            temporary,
            format="PNG",
            compress_level=6,
            icc_profile=standard_srgb_profile(),
            dpi=(96.0, 96.0),
        )
    os.replace(temporary, png)
    return {"seconds": round(time.perf_counter() - started, 3), "command": command}


def rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode in {"RGBA", "LA"}:
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(white, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def edge_f1(source: np.ndarray, result: np.ndarray) -> dict[str, float]:
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    source_edges = cv2.Canny(source_gray, 50, 150, L2gradient=True) > 0
    result_edges = cv2.Canny(result_gray, 50, 150, L2gradient=True) > 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    source_near = cv2.dilate(source_edges.astype(np.uint8), kernel, iterations=1) > 0
    result_near = cv2.dilate(result_edges.astype(np.uint8), kernel, iterations=1) > 0
    result_total = int(result_edges.sum())
    source_total = int(source_edges.sum())
    if source_total == 0 and result_total == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "source_count": 0,
            "result_count": 0,
        }
    precision = (
        1.0
        if result_total == 0
        else float((result_edges & source_near).sum() / result_total)
    )
    recall = (
        1.0
        if source_total == 0
        else float((source_edges & result_near).sum() / source_total)
    )
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "source_count": source_total,
        "result_count": result_total,
    }


def write_resized_rgb(source: Path, target: Path, size: tuple[int, int]) -> None:
    """Create the deterministic V3 fallback at the requested proof dimensions."""

    with Image.open(source) as image:
        image.load()
        if image.mode in {"RGBA", "LA"}:
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(white, rgba).convert("RGB")
        else:
            rgb = image.convert("RGB")
        if rgb.size != size:
            rgb = rgb.resize(size, Image.Resampling.LANCZOS)
        rgb.save(
            target,
            format="PNG",
            compress_level=4,
            icc_profile=standard_srgb_profile(),
            dpi=(96.0, 96.0),
        )


def gaussian_kernel_spec(
    sigma_x: float,
    sigma_y: float,
    truncate: float = GAUSSIAN_TRUNCATE,
) -> dict[str, Any]:
    """Return an explicit finite anisotropic Gaussian support."""

    if not all(math.isfinite(value) and value > 0.0 for value in (sigma_x, sigma_y)):
        raise V4Error("Gaussian sigma values must be finite and greater than zero.")
    if not math.isfinite(truncate) or truncate < 2.0 or truncate > 6.0:
        raise V4Error("Gaussian truncation must be finite and between 2 and 6 sigma.")
    radius_x = max(1, int(math.ceil(sigma_x * truncate)))
    radius_y = max(1, int(math.ceil(sigma_y * truncate)))
    return {
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "truncate": float(truncate),
        "radius_x": radius_x,
        "radius_y": radius_y,
        "kernel_width": 2 * radius_x + 1,
        "kernel_height": 2 * radius_y + 1,
    }


def uniform_restoration_rgb(
    v3_rgb: np.ndarray,
    deep_rgb: np.ndarray,
    *,
    deep_weight: float,
    unsharp_amount: float,
    sigma_x: float,
    sigma_y: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the globally uniform restoration formula in encoded sRGB RGB."""

    if v3_rgb.shape != deep_rgb.shape or v3_rgb.ndim != 3 or v3_rgb.shape[2] != 3:
        raise V4Error(
            f"Uniform restoration shape mismatch: {v3_rgb.shape} vs {deep_rgb.shape}."
        )
    if not math.isfinite(deep_weight) or not 0.0 <= deep_weight <= 1.0:
        raise V4Error("Deep restoration weight must be between zero and one.")
    if not math.isfinite(unsharp_amount) or not 0.0 <= unsharp_amount <= 1.0:
        raise V4Error("Unsharp amount must be between zero and one.")
    kernel = gaussian_kernel_spec(sigma_x, sigma_y)
    v3_float = v3_rgb.astype(np.float32)
    deep_float = deep_rgb.astype(np.float32) if deep_weight > 0.0 else v3_float
    blurred = cv2.GaussianBlur(
        v3_float,
        (kernel["kernel_width"], kernel["kernel_height"]),
        sigmaX=kernel["sigma_x"],
        sigmaY=kernel["sigma_y"],
        borderType=cv2.BORDER_REFLECT_101,
    )
    restored = np.clip(
        np.rint(
            v3_float * (1.0 - deep_weight)
            + deep_float * deep_weight
            + unsharp_amount * (v3_float - blurred)
        ),
        0.0,
        255.0,
    ).astype(np.uint8)
    return restored, kernel


def write_uniform_restoration(
    v3_source: Path,
    deep_source: Path,
    target: Path,
    size: tuple[int, int],
    *,
    deep_weight: float,
    unsharp_amount: float,
    sigma_native: float,
    stripe_height: int = RESTORATION_STRIPE_HEIGHT,
) -> dict[str, Any]:
    """Render one candidate with bounded stripe floats and a disk-backed output."""

    if stripe_height < 32:
        raise V4Error("Restoration stripe height must be at least 32 pixels.")

    def opened_rgb(image: Image.Image) -> Image.Image:
        image.load()
        if image.mode in {"RGBA", "LA"}:
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(white, rgba).convert("RGB")
        return image.convert("RGB")

    started = time.perf_counter()
    pixel_store = target.with_suffix(target.suffix + ".pixels.tmp")
    output_map: np.memmap | None = None
    with Image.open(v3_source) as v3_image:
        v3_rgb = opened_rgb(v3_image)
        if deep_weight > 0.0:
            with Image.open(deep_source) as deep_image:
                deep_rgb = opened_rgb(deep_image)
        else:
            deep_rgb = v3_rgb
        native_size = v3_rgb.size
        scale_x = size[0] / native_size[0]
        scale_y = size[1] / native_size[1]
        sigma_x = sigma_native * scale_x
        sigma_y = sigma_native * scale_y
        kernel = gaussian_kernel_spec(sigma_x, sigma_y)
        if v3_rgb.size != size:
            v3_rgb = v3_rgb.resize(size, Image.Resampling.LANCZOS)
        if deep_weight == 0.0:
            deep_rgb = v3_rgb
        elif deep_rgb.size != size:
            deep_rgb = deep_rgb.resize(size, Image.Resampling.LANCZOS)
        v3_array = np.asarray(v3_rgb, dtype=np.uint8)
        deep_array = np.asarray(deep_rgb, dtype=np.uint8)
        try:
            output_map = np.memmap(
                pixel_store,
                dtype=np.uint8,
                mode="w+",
                shape=(size[1], size[0], 3),
            )
            for y_start in range(0, size[1], stripe_height):
                y_stop = min(size[1], y_start + stripe_height)
                guard_start = max(0, y_start - kernel["radius_y"])
                guard_stop = min(size[1], y_stop + kernel["radius_y"])
                restored_guard, _ = uniform_restoration_rgb(
                    v3_array[guard_start:guard_stop],
                    deep_array[guard_start:guard_stop],
                    deep_weight=deep_weight,
                    unsharp_amount=unsharp_amount,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                )
                core_start = y_start - guard_start
                core_stop = core_start + (y_stop - y_start)
                output_map[y_start:y_stop] = restored_guard[core_start:core_stop]
            output_map.flush()
            output_image = Image.fromarray(np.asarray(output_map), mode="RGB")
            output_image.save(
                target,
                format="PNG",
                compress_level=4,
                icc_profile=standard_srgb_profile(),
                dpi=(96.0, 96.0),
            )
            output_image.close()
        finally:
            if output_map is not None:
                output_map.flush()
                del output_map
            pixel_store.unlink(missing_ok=True)
    return {
        "method": (
            "uniform_global_guarded_v3_usm_srgb"
            if deep_weight == 0.0
            else "uniform_global_v3_deep_unsharp_srgb"
        ),
        "formula": (
            "clip(rint(v3+unsharp_amount*(v3-GaussianBlur(v3))),0,255)"
            if deep_weight == 0.0
            else (
                "clip(rint(v3*(1-deep_weight)+deep*deep_weight+"
                "unsharp_amount*(v3-GaussianBlur(v3))),0,255)"
            )
        ),
        "v3_weight": round(1.0 - deep_weight, 4),
        "deep_weight": round(deep_weight, 4),
        "deep_pixels_used": deep_weight > 0.0,
        "unsharp_amount": round(unsharp_amount, 4),
        "sigma_native": float(sigma_native),
        "native_size": list(native_size),
        "proof_scale_x": round(scale_x, 8),
        "proof_scale_y": round(scale_y, 8),
        "gaussian": kernel,
        "stripe_height": stripe_height,
        "spatially_uniform": True,
        "size": list(size),
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "seconds": round(time.perf_counter() - started, 3),
    }


def discover_v3_native(raster_base: Path, source_sha256: str) -> dict[str, Any] | None:
    """Resolve the exact validated V3 x4 parent recorded by V4 Deep."""

    sidecar = raster_base.with_suffix(raster_base.suffix + ".json")
    if not sidecar.is_file():
        return None
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise V4Error(f"Cannot read V4 Deep provenance: {sidecar}") from exc

    recorded_sources = {
        str(metadata.get(key, "")).lower()
        for key in ("source_sha256", "deep_input_sha256")
        if metadata.get(key)
    }
    if recorded_sources and source_sha256.lower() not in recorded_sources:
        raise V4Error("V4 Deep raster and current source do not share the same provenance.")

    raw_path = metadata.get("v3_native") or metadata.get("native_path")
    if not raw_path:
        return None
    native = Path(str(raw_path)).resolve()
    if not native.is_file():
        raise V4Error(f"Recorded V3 baseline is missing: {native}")
    actual_sha = sha256_file(native)
    expected_sha = str(
        metadata.get("v3_native_sha256") or metadata.get("native_sha256") or ""
    ).lower()
    if expected_sha and actual_sha.lower() != expected_sha:
        raise V4Error("Recorded V3 baseline failed its SHA-256 integrity check.")
    return {
        "path": native,
        "sha256": actual_sha,
        "sidecar": sidecar.resolve(),
        "pipeline": metadata.get("pipeline"),
    }


def comparative_restoration_metrics(
    v3_rgb: np.ndarray,
    v4_rgb: np.ndarray,
    limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Judge V4 against V3 directly; source fidelity is only a separate diagnostic."""

    limits = limits or COMPARATIVE_QA_LIMITS
    if v3_rgb.shape != v4_rgb.shape or v3_rgb.ndim != 3 or v3_rgb.shape[2] != 3:
        raise V4Error(f"Comparative QA shape mismatch: {v3_rgb.shape} vs {v4_rgb.shape}.")

    v3_luma = cv2.cvtColor(v3_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    v4_luma = cv2.cvtColor(v4_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    c1 = 0.01**2
    c2 = 0.03**2
    mu_v3 = cv2.GaussianBlur(v3_luma, (11, 11), 1.5)
    mu_v4 = cv2.GaussianBlur(v4_luma, (11, 11), 1.5)
    variance_v3 = np.maximum(
        cv2.GaussianBlur(v3_luma * v3_luma, (11, 11), 1.5) - mu_v3 * mu_v3,
        0.0,
    )
    variance_v4 = np.maximum(
        cv2.GaussianBlur(v4_luma * v4_luma, (11, 11), 1.5) - mu_v4 * mu_v4,
        0.0,
    )
    covariance = (
        cv2.GaussianBlur(v3_luma * v4_luma, (11, 11), 1.5) - mu_v3 * mu_v4
    )
    ssim_map = (
        (2.0 * mu_v3 * mu_v4 + c1) * (2.0 * covariance + c2)
    ) / (
        (mu_v3 * mu_v3 + mu_v4 * mu_v4 + c1)
        * (variance_v3 + variance_v4 + c2)
        + 1e-12
    )

    def gradient_magnitude(luma: np.ndarray) -> np.ndarray:
        gradient_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gradient_x, gradient_y)

    gradient_v3 = gradient_magnitude(v3_luma)
    gradient_v4 = gradient_magnitude(v4_luma)

    def stable_ratio(candidate: float, baseline: float, epsilon: float = 1e-12) -> float:
        if baseline <= epsilon:
            return 1.0 if candidate <= epsilon else float("inf")
        return candidate / baseline

    edge_threshold = max(float(np.percentile(gradient_v3, 75)), 4.0 / 255.0)
    edge_band = cv2.dilate(
        (gradient_v3 >= edge_threshold).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ) > 0
    if not np.any(edge_band):
        edge_band = np.ones(v3_luma.shape, dtype=bool)

    acutance_v3 = float(np.mean(gradient_v3[edge_band]))
    acutance_v4 = float(np.mean(gradient_v4[edge_band]))
    acutance_ratio = stable_ratio(acutance_v4, acutance_v3, 1e-8)
    tenengrad_v3 = float(np.mean(np.square(gradient_v3)))
    tenengrad_v4 = float(np.mean(np.square(gradient_v4)))
    tenengrad_ratio = stable_ratio(tenengrad_v4, tenengrad_v3)
    laplacian_v3 = float(np.var(cv2.Laplacian(v3_luma, cv2.CV_32F, ksize=3)))
    laplacian_v4 = float(np.var(cv2.Laplacian(v4_luma, cv2.CV_32F, ksize=3)))
    laplacian_ratio = stable_ratio(laplacian_v4, laplacian_v3)
    gradient_p95_v3 = float(np.percentile(gradient_v3, 95))
    gradient_p95_v4 = float(np.percentile(gradient_v4, 95))
    gradient_p95_ratio = stable_ratio(gradient_p95_v4, gradient_p95_v3, 1e-8)

    local_kernel = np.ones((5, 5), dtype=np.uint8)
    local_min = cv2.erode(v3_luma, local_kernel)
    local_max = cv2.dilate(v3_luma, local_kernel)
    overshoot = (
        np.maximum(local_min - v4_luma, 0.0) + np.maximum(v4_luma - local_max, 0.0)
    ) * 255.0
    edge_overshoot = overshoot[edge_band]
    difference = np.abs(v3_rgb.astype(np.int16) - v4_rgb.astype(np.int16))
    edge = edge_f1(v3_rgb, v4_rgb)

    metrics = {
        "ssim_mean": round(float(np.mean(ssim_map)), 6),
        "ssim_p05": round(float(np.percentile(ssim_map, 5)), 6),
        "edge_f1": round(edge["f1"], 6),
        "edge_precision": round(edge["precision"], 6),
        "edge_recall": round(edge["recall"], 6),
        "edge_pixel_fraction_v3": round(
            float(edge["source_count"]) / v3_luma.size,
            8,
        ),
        "edge_pixel_fraction_v4": round(
            float(edge["result_count"]) / v4_luma.size,
            8,
        ),
        "acutance_v3": round(acutance_v3, 6),
        "acutance_v4": round(acutance_v4, 6),
        "acutance_ratio": round(acutance_ratio, 6) if math.isfinite(acutance_ratio) else "inf",
        "sharpness_gain_percent": (
            round((acutance_ratio - 1.0) * 100.0, 3)
            if math.isfinite(acutance_ratio)
            else "inf"
        ),
        "tenengrad_v3": round(tenengrad_v3, 6),
        "tenengrad_v4": round(tenengrad_v4, 6),
        "tenengrad_ratio": round(tenengrad_ratio, 6),
        "laplacian_variance_v3": round(laplacian_v3, 6),
        "laplacian_variance_v4": round(laplacian_v4, 6),
        "laplacian_variance_ratio": round(laplacian_ratio, 6),
        "gradient_p95_v3": round(gradient_p95_v3, 6),
        "gradient_p95_v4": round(gradient_p95_v4, 6),
        "gradient_p95_ratio": round(gradient_p95_ratio, 6),
        "edge_overshoot_mean_255": round(float(np.mean(edge_overshoot)), 4),
        "edge_overshoot_p99_255": round(float(np.percentile(edge_overshoot, 99)), 4),
        "edge_overshoot_fraction_gt2": round(float(np.mean(edge_overshoot > 2.0)), 6),
        "abs_difference_p95": round(float(np.percentile(difference, 95)), 4),
    }

    retention_failures: list[str] = []
    if metrics["ssim_mean"] < limits["min_ssim_mean"]:
        retention_failures.append(
            f"SSIM {metrics['ssim_mean']:.4f} < {limits['min_ssim_mean']}"
        )
    if edge["f1"] < limits["min_edge_f1"]:
        retention_failures.append(f"edge F1 {edge['f1']:.4f} < {limits['min_edge_f1']}")
    if acutance_ratio < limits["min_acutance_retention_ratio"]:
        retention_failures.append(
            f"acutance retention {acutance_ratio:.4f} < "
            f"{limits['min_acutance_retention_ratio']}"
        )
    if gradient_p95_ratio < limits["min_gradient_p95_retention_ratio"]:
        retention_failures.append(
            f"gradient p95 retention {gradient_p95_ratio:.4f} < "
            f"{limits['min_gradient_p95_retention_ratio']}"
        )
    if not math.isfinite(acutance_ratio) or acutance_ratio > limits["max_acutance_ratio"]:
        retention_failures.append(
            f"acutance ratio {acutance_ratio:.4g} > {limits['max_acutance_ratio']}"
        )
    if tenengrad_ratio > limits["max_tenengrad_ratio"]:
        retention_failures.append(
            f"Tenengrad ratio {tenengrad_ratio:.4f} > {limits['max_tenengrad_ratio']}"
        )
    if laplacian_ratio > limits["max_laplacian_variance_ratio"]:
        retention_failures.append(
            "Laplacian variance ratio "
            f"{laplacian_ratio:.4f} > {limits['max_laplacian_variance_ratio']}"
        )
    if gradient_p95_ratio > limits["max_gradient_p95_ratio"]:
        retention_failures.append(
            f"gradient p95 ratio {gradient_p95_ratio:.4f} > "
            f"{limits['max_gradient_p95_ratio']}"
        )
    if metrics["edge_overshoot_p99_255"] > limits["max_edge_overshoot_p99_255"]:
        retention_failures.append(
            "edge overshoot p99 "
            f"{metrics['edge_overshoot_p99_255']:.2f} > "
            f"{limits['max_edge_overshoot_p99_255']}"
        )
    if metrics["abs_difference_p95"] > limits["max_abs_difference_p95"]:
        retention_failures.append(
            f"difference p95 {metrics['abs_difference_p95']:.1f} > "
            f"{limits['max_abs_difference_p95']}"
        )

    edge_strength_checks = {
        "acutance": (
            math.isfinite(acutance_ratio)
            and acutance_ratio >= limits["min_acutance_ratio"]
        ),
        "gradient_p95": (
            math.isfinite(gradient_p95_ratio)
            and gradient_p95_ratio >= limits["min_gradient_p95_ratio"]
        ),
    }
    energy_detail_checks = {
        "tenengrad": (
            math.isfinite(tenengrad_ratio)
            and tenengrad_ratio >= limits["min_tenengrad_ratio"]
        ),
        "laplacian_variance": (
            math.isfinite(laplacian_ratio)
            and laplacian_ratio >= limits["min_laplacian_variance_ratio"]
        ),
    }
    edge_strength_passed = any(edge_strength_checks.values())
    energy_detail_passed = any(energy_detail_checks.values())
    improvement_failures: list[str] = []
    if not edge_strength_passed:
        improvement_failures.append(
            "no edge-strength indicator passed (requires acutance or gradient p95)"
        )
    if not energy_detail_passed:
        improvement_failures.append(
            "no energy/detail indicator passed (requires Tenengrad or Laplacian variance)"
        )
    safe = not retention_failures
    meaningfully_sharper = edge_strength_passed and energy_detail_passed
    claim_v4_better = safe and meaningfully_sharper
    return {
        **metrics,
        "limits": limits,
        "clarity_checks": {
            "edge_strength": edge_strength_checks,
            "energy_detail": energy_detail_checks,
        },
        "edge_strength_passed": edge_strength_passed,
        "energy_detail_passed": energy_detail_passed,
        "clarity_score": sum(edge_strength_checks.values())
        + sum(energy_detail_checks.values()),
        "clarity_required": (
            "at least one edge-strength and one energy/detail indicator"
        ),
        "retention_passed": safe,
        "meaningfully_sharper": meaningfully_sharper,
        "claim_v4_better": claim_v4_better,
        "passed": claim_v4_better,
        "decision": "use_v4" if claim_v4_better else "fallback_v3",
        "retention_failures": retention_failures,
        "improvement_failures": improvement_failures,
    }


def _sample_axis_origins(length: int, crop: int, grid_size: int) -> list[int]:
    if length <= crop:
        return [0]
    count = min(grid_size, max(2, math.ceil(length / crop)))
    return sorted(
        {
            int(round(index * (length - crop) / (count - 1)))
            for index in range(count)
        }
    )


def _covered_axis_length(origins: list[int], crop: int) -> int:
    covered = 0
    right = -1
    for left in sorted(origins):
        next_right = left + crop
        if left >= right:
            covered += crop
        elif next_right > right:
            covered += next_right - right
        right = max(right, next_right)
    return covered


def native_comparison_samples(
    v3_path: Path,
    deep_path: Path,
    *,
    full_frame_max_pixels: int = COMPARATIVE_QA_FULL_FRAME_MAX_PIXELS,
    tile_size: int = COMPARATIVE_QA_TILE_SIZE,
    grid_size: int = COMPARATIVE_QA_GRID_SIZE,
    halo_radius: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load independent native-x4 crops; no metric ever sees crop-join seams."""

    if (
        full_frame_max_pixels < 1
        or tile_size < 32
        or grid_size < 1
        or halo_radius < 0
    ):
        raise V4Error("Invalid comparative QA sampling configuration.")

    def opened_rgb(image: Image.Image) -> Image.Image:
        image.load()
        if image.mode in {"RGBA", "LA"}:
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(white, rgba).convert("RGB")
        return image.convert("RGB")

    with Image.open(v3_path) as v3_image, Image.open(deep_path) as deep_image:
        v3_rgb = opened_rgb(v3_image)
        deep_rgb = opened_rgb(deep_image)
        native_size = v3_rgb.size
        deep_size = deep_rgb.size
        if not math.isclose(
            native_size[0] / native_size[1],
            deep_size[0] / deep_size[1],
            rel_tol=0.001,
            abs_tol=0.001,
        ):
            raise V4Error(
                f"V3 native and Deep candidate aspect ratios differ: {native_size} vs {deep_size}."
            )

        native_pixels = native_size[0] * native_size[1]
        if native_pixels <= full_frame_max_pixels:
            deep_native = deep_rgb.resize(native_size, Image.Resampling.LANCZOS)
            samples = [
                {
                    "index": 0,
                    "box": [0, 0, native_size[0], native_size[1]],
                    "v3": np.asarray(v3_rgb, dtype=np.uint8),
                    "deep": np.asarray(deep_native, dtype=np.uint8),
                    "v3_guard": np.asarray(v3_rgb, dtype=np.uint8),
                    "deep_guard": np.asarray(deep_native, dtype=np.uint8),
                    "core_slice": [0, native_size[1], 0, native_size[0]],
                }
            ]
            sampling = {
                "strategy": "full_frame_at_v3_native_resolution",
                "native_reference_size": list(native_size),
                "raw_deep_size": list(deep_size),
                "evaluation_size": list(native_size),
                "crop_size": list(native_size),
                "crop_grid": [1, 1],
                "crop_count": 1,
                "halo_radius_native": 0,
                "crop_boxes": [[0, 0, native_size[0], native_size[1]]],
                "evaluated_pixel_count": native_pixels,
                "native_area_coverage": 1.0,
                "metrics_computed_per_crop": True,
                "crop_join_seams_in_metrics": False,
                "source_scale_downsample_used": False,
            }
            return samples, sampling

        crop_width = min(tile_size, native_size[0])
        crop_height = min(tile_size, native_size[1])
        x_origins = _sample_axis_origins(native_size[0], crop_width, grid_size)
        y_origins = _sample_axis_origins(native_size[1], crop_height, grid_size)
        deep_scale_x = deep_size[0] / native_size[0]
        deep_scale_y = deep_size[1] / native_size[1]
        samples: list[dict[str, Any]] = []
        for row, y in enumerate(y_origins):
            for column, x in enumerate(x_origins):
                guard_left = max(0, x - halo_radius)
                guard_top = max(0, y - halo_radius)
                guard_right = min(native_size[0], x + crop_width + halo_radius)
                guard_bottom = min(native_size[1], y + crop_height + halo_radius)
                guard_width = guard_right - guard_left
                guard_height = guard_bottom - guard_top
                v3_guard = v3_rgb.crop(
                    (guard_left, guard_top, guard_right, guard_bottom)
                )
                deep_guard = deep_rgb.resize(
                    (guard_width, guard_height),
                    Image.Resampling.LANCZOS,
                    box=(
                        guard_left * deep_scale_x,
                        guard_top * deep_scale_y,
                        guard_right * deep_scale_x,
                        guard_bottom * deep_scale_y,
                    ),
                )
                v3_guard_array = np.asarray(v3_guard, dtype=np.uint8).copy()
                deep_guard_array = np.asarray(deep_guard, dtype=np.uint8).copy()
                core_left = x - guard_left
                core_top = y - guard_top
                core_right = core_left + crop_width
                core_bottom = core_top + crop_height
                samples.append(
                    {
                        "index": row * len(x_origins) + column,
                        "box": [x, y, x + crop_width, y + crop_height],
                        "v3": v3_guard_array[
                            core_top:core_bottom,
                            core_left:core_right,
                        ].copy(),
                        "deep": deep_guard_array[
                            core_top:core_bottom,
                            core_left:core_right,
                        ].copy(),
                        "v3_guard": v3_guard_array,
                        "deep_guard": deep_guard_array,
                        "core_slice": [
                            core_top,
                            core_bottom,
                            core_left,
                            core_right,
                        ],
                    }
                )

        covered_width = _covered_axis_length(x_origins, crop_width)
        covered_height = _covered_axis_length(y_origins, crop_height)
        sampling = {
            "strategy": "independent_stratified_crops_at_v3_native_resolution",
            "native_reference_size": list(native_size),
            "raw_deep_size": list(deep_size),
            "evaluation_size": [crop_width, crop_height],
            "crop_size": [crop_width, crop_height],
            "crop_grid": [len(x_origins), len(y_origins)],
            "crop_count": len(samples),
            "halo_radius_native": halo_radius,
            "crop_boxes": [sample["box"] for sample in samples],
            "evaluated_pixel_count": len(samples) * crop_width * crop_height,
            "native_area_coverage": round(
                covered_width * covered_height / native_pixels,
                6,
            ),
            "metrics_computed_per_crop": True,
            "crop_join_seams_in_metrics": False,
            "source_scale_downsample_used": False,
        }
        return samples, sampling


def aggregate_comparative_crop_metrics(
    crop_reports: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate independent crop metrics with conservative quantiles and coverage."""

    limits = limits or COMPARATIVE_QA_LIMITS
    if not crop_reports or len(crop_reports) != len(samples):
        raise V4Error("Comparative crop reports and sample descriptors do not align.")

    metric_keys = (
        "ssim_mean",
        "ssim_p05",
        "edge_f1",
        "edge_precision",
        "edge_recall",
        "edge_pixel_fraction_v3",
        "edge_pixel_fraction_v4",
        "acutance_v3",
        "acutance_v4",
        "acutance_ratio",
        "tenengrad_v3",
        "tenengrad_v4",
        "tenengrad_ratio",
        "laplacian_variance_v3",
        "laplacian_variance_v4",
        "laplacian_variance_ratio",
        "gradient_p95_v3",
        "gradient_p95_v4",
        "gradient_p95_ratio",
        "edge_overshoot_mean_255",
        "edge_overshoot_p99_255",
        "edge_overshoot_fraction_gt2",
        "abs_difference_p95",
    )

    summaries: dict[str, dict[str, Any]] = {}
    for key in metric_keys:
        finite_values: list[float] = []
        nonfinite_count = 0
        for report in crop_reports:
            value = report.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                nonfinite_count += 1
            elif math.isfinite(float(value)):
                finite_values.append(float(value))
            else:
                nonfinite_count += 1
        if finite_values:
            values = np.asarray(finite_values, dtype=np.float64)
            summaries[key] = {
                "min": round(float(np.min(values)), 6),
                "p10": round(float(np.percentile(values, 10)), 6),
                "median": round(float(np.median(values)), 6),
                "p90": round(float(np.percentile(values, 90)), 6),
                "max": round(float(np.max(values)), 6),
                "finite_count": len(finite_values),
                "nonfinite_count": nonfinite_count,
            }
        else:
            summaries[key] = {
                "min": None,
                "p10": None,
                "median": None,
                "p90": None,
                "max": None,
                "finite_count": 0,
                "nonfinite_count": nonfinite_count,
            }

    def statistic(key: str, name: str, default: float) -> float:
        value = summaries[key][name]
        return default if value is None else float(value)

    crop_count = len(crop_reports)
    retention_fraction = sum(
        bool(report["retention_passed"]) for report in crop_reports
    ) / crop_count
    edge_strength_fraction = sum(
        bool(report["edge_strength_passed"]) for report in crop_reports
    ) / crop_count
    energy_detail_fraction = sum(
        bool(report["energy_detail_passed"]) for report in crop_reports
    ) / crop_count
    joint_improvement_fraction = sum(
        bool(report["edge_strength_passed"] and report["energy_detail_passed"])
        for report in crop_reports
    ) / crop_count
    full_gate_fraction = sum(
        bool(
            report["retention_passed"]
            and report["edge_strength_passed"]
            and report["energy_detail_passed"]
        )
        for report in crop_reports
    ) / crop_count

    acutance_ratio = statistic("acutance_ratio", "median", float("inf"))
    tenengrad_ratio = statistic("tenengrad_ratio", "median", float("inf"))
    laplacian_ratio = statistic(
        "laplacian_variance_ratio",
        "median",
        float("inf"),
    )
    gradient_p95_ratio = statistic("gradient_p95_ratio", "median", float("inf"))
    edge_strength_checks = {
        "acutance": (
            math.isfinite(acutance_ratio)
            and acutance_ratio >= limits["min_acutance_ratio"]
        ),
        "gradient_p95": (
            math.isfinite(gradient_p95_ratio)
            and gradient_p95_ratio >= limits["min_gradient_p95_ratio"]
        ),
    }
    energy_detail_checks = {
        "tenengrad": (
            math.isfinite(tenengrad_ratio)
            and tenengrad_ratio >= limits["min_tenengrad_ratio"]
        ),
        "laplacian_variance": (
            math.isfinite(laplacian_ratio)
            and laplacian_ratio >= limits["min_laplacian_variance_ratio"]
        ),
    }

    required_fraction = limits["min_crop_gate_pass_fraction"]
    retention_failures: list[str] = []
    ssim_p10 = statistic("ssim_mean", "p10", -float("inf"))
    edge_f1_p10 = statistic("edge_f1", "p10", -float("inf"))
    acutance_p10 = statistic("acutance_ratio", "p10", -float("inf"))
    acutance_p90 = statistic("acutance_ratio", "p90", float("inf"))
    gradient_p10 = statistic("gradient_p95_ratio", "p10", -float("inf"))
    gradient_p90 = statistic("gradient_p95_ratio", "p90", float("inf"))
    tenengrad_p90 = statistic("tenengrad_ratio", "p90", float("inf"))
    laplacian_p90 = statistic(
        "laplacian_variance_ratio",
        "p90",
        float("inf"),
    )
    overshoot_p90 = statistic(
        "edge_overshoot_p99_255",
        "p90",
        float("inf"),
    )
    difference_p90 = statistic("abs_difference_p95", "p90", float("inf"))
    if ssim_p10 < limits["min_ssim_mean"]:
        retention_failures.append(
            f"crop SSIM p10 {ssim_p10:.4f} < {limits['min_ssim_mean']}"
        )
    if edge_f1_p10 < limits["min_edge_f1"]:
        retention_failures.append(
            f"crop edge F1 p10 {edge_f1_p10:.4f} < {limits['min_edge_f1']}"
        )
    if acutance_p10 < limits["min_acutance_retention_ratio"]:
        retention_failures.append(
            f"crop acutance p10 {acutance_p10:.4f} < "
            f"{limits['min_acutance_retention_ratio']}"
        )
    if acutance_p90 > limits["max_acutance_ratio"]:
        retention_failures.append(
            f"crop acutance p90 {acutance_p90:.4f} > {limits['max_acutance_ratio']}"
        )
    if gradient_p10 < limits["min_gradient_p95_retention_ratio"]:
        retention_failures.append(
            f"crop gradient-p95 p10 {gradient_p10:.4f} < "
            f"{limits['min_gradient_p95_retention_ratio']}"
        )
    if gradient_p90 > limits["max_gradient_p95_ratio"]:
        retention_failures.append(
            f"crop gradient-p95 p90 {gradient_p90:.4f} > "
            f"{limits['max_gradient_p95_ratio']}"
        )
    if tenengrad_p90 > limits["max_tenengrad_ratio"]:
        retention_failures.append(
            f"crop Tenengrad p90 {tenengrad_p90:.4f} > {limits['max_tenengrad_ratio']}"
        )
    if laplacian_p90 > limits["max_laplacian_variance_ratio"]:
        retention_failures.append(
            f"crop Laplacian p90 {laplacian_p90:.4f} > "
            f"{limits['max_laplacian_variance_ratio']}"
        )
    if overshoot_p90 > limits["max_edge_overshoot_p99_255"]:
        retention_failures.append(
            f"crop overshoot-p99 p90 {overshoot_p90:.2f} > "
            f"{limits['max_edge_overshoot_p99_255']}"
        )
    if difference_p90 > limits["max_abs_difference_p95"]:
        retention_failures.append(
            f"crop difference-p95 p90 {difference_p90:.1f} > "
            f"{limits['max_abs_difference_p95']}"
        )
    if retention_fraction < required_fraction:
        retention_failures.append(
            f"retention passed in {retention_fraction:.1%} crops < {required_fraction:.0%}"
        )

    edge_strength_passed = any(edge_strength_checks.values())
    energy_detail_passed = any(energy_detail_checks.values())
    improvement_failures: list[str] = []
    if not edge_strength_passed:
        improvement_failures.append(
            "median crop has no edge-strength gain (acutance or gradient p95)"
        )
    if not energy_detail_passed:
        improvement_failures.append(
            "median crop has no energy/detail gain (Tenengrad or Laplacian variance)"
        )
    if joint_improvement_fraction < required_fraction:
        improvement_failures.append(
            f"both improvement groups passed in {joint_improvement_fraction:.1%} crops "
            f"< {required_fraction:.0%}"
        )

    spatial_failures: list[str] = []
    if full_gate_fraction < required_fraction:
        spatial_failures.append(
            f"retention + both improvement groups passed together in "
            f"{full_gate_fraction:.1%} crops < {required_fraction:.0%}"
        )

    safe = not retention_failures
    meaningfully_sharper = not improvement_failures
    spatially_consistent = not spatial_failures
    claim_v4_better = safe and meaningfully_sharper and spatially_consistent

    ranked = sorted(
        zip(samples, crop_reports),
        key=lambda item: (
            bool(item[1]["claim_v4_better"]),
            item[1]["ssim_mean"],
            item[1]["edge_f1"],
        ),
    )
    worst_crops = []
    for sample, report in ranked[: min(5, crop_count)]:
        worst_crops.append(
            {
                "index": sample["index"],
                "box": sample["box"],
                "retention_passed": report["retention_passed"],
                "edge_strength_passed": report["edge_strength_passed"],
                "energy_detail_passed": report["energy_detail_passed"],
                "full_gate_passed": report["claim_v4_better"],
                "ssim_mean": report["ssim_mean"],
                "edge_f1": report["edge_f1"],
                "acutance_ratio": report["acutance_ratio"],
                "gradient_p95_ratio": report["gradient_p95_ratio"],
                "tenengrad_ratio": report["tenengrad_ratio"],
                "laplacian_variance_ratio": report["laplacian_variance_ratio"],
                "retention_failures": report["retention_failures"],
                "improvement_failures": report["improvement_failures"],
            }
        )

    metrics = {
        "ssim_mean": round(ssim_p10, 6),
        "ssim_p05": round(statistic("ssim_p05", "p10", -float("inf")), 6),
        "edge_f1": round(edge_f1_p10, 6),
        "edge_precision": round(
            statistic("edge_precision", "p10", -float("inf")),
            6,
        ),
        "edge_recall": round(
            statistic("edge_recall", "p10", -float("inf")),
            6,
        ),
        "edge_pixel_fraction_v3": round(
            statistic("edge_pixel_fraction_v3", "median", 0.0),
            8,
        ),
        "edge_pixel_fraction_v4": round(
            statistic("edge_pixel_fraction_v4", "median", 0.0),
            8,
        ),
        "acutance_v3": round(statistic("acutance_v3", "median", 0.0), 6),
        "acutance_v4": round(statistic("acutance_v4", "median", 0.0), 6),
        "acutance_ratio": round(acutance_ratio, 6),
        "sharpness_gain_percent": round((acutance_ratio - 1.0) * 100.0, 3),
        "tenengrad_v3": round(statistic("tenengrad_v3", "median", 0.0), 6),
        "tenengrad_v4": round(statistic("tenengrad_v4", "median", 0.0), 6),
        "tenengrad_ratio": round(tenengrad_ratio, 6),
        "laplacian_variance_v3": round(
            statistic("laplacian_variance_v3", "median", 0.0),
            6,
        ),
        "laplacian_variance_v4": round(
            statistic("laplacian_variance_v4", "median", 0.0),
            6,
        ),
        "laplacian_variance_ratio": round(laplacian_ratio, 6),
        "gradient_p95_v3": round(
            statistic("gradient_p95_v3", "median", 0.0),
            6,
        ),
        "gradient_p95_v4": round(
            statistic("gradient_p95_v4", "median", 0.0),
            6,
        ),
        "gradient_p95_ratio": round(gradient_p95_ratio, 6),
        "edge_overshoot_mean_255": round(
            statistic("edge_overshoot_mean_255", "p90", float("inf")),
            4,
        ),
        "edge_overshoot_p99_255": round(overshoot_p90, 4),
        "edge_overshoot_fraction_gt2": round(
            statistic("edge_overshoot_fraction_gt2", "p90", float("inf")),
            6,
        ),
        "abs_difference_p95": round(difference_p90, 4),
    }
    return {
        **metrics,
        "limits": limits,
        "clarity_checks": {
            "edge_strength": edge_strength_checks,
            "energy_detail": energy_detail_checks,
        },
        "edge_strength_passed": edge_strength_passed,
        "energy_detail_passed": energy_detail_passed,
        "clarity_score": sum(edge_strength_checks.values())
        + sum(energy_detail_checks.values()),
        "clarity_required": (
            "at least one edge-strength and one energy/detail indicator"
        ),
        "crop_pass_fractions": {
            "retention": round(retention_fraction, 6),
            "edge_strength": round(edge_strength_fraction, 6),
            "energy_detail": round(energy_detail_fraction, 6),
            "both_improvement_groups": round(joint_improvement_fraction, 6),
            "retention_and_both_groups": round(full_gate_fraction, 6),
            "required": required_fraction,
        },
        "crop_metric_quantiles": summaries,
        "worst_crops": worst_crops,
        "aggregation": {
            "metrics_computed_independently_per_crop": True,
            "crop_join_seams_in_metrics": False,
            "retention_low_quantile": "p10",
            "retention_high_quantile": "p90",
            "improvement_center": "median",
            "spatial_majority_rule": f">={required_fraction:.0%} crops",
        },
        "retention_passed": safe,
        "meaningfully_sharper": meaningfully_sharper,
        "spatially_consistent": spatially_consistent,
        "claim_v4_better": claim_v4_better,
        "passed": claim_v4_better,
        "decision": "use_v4" if claim_v4_better else "fallback_v3",
        "retention_failures": retention_failures,
        "improvement_failures": improvement_failures,
        "spatial_failures": spatial_failures,
    }


def compare_v3_v4_paths(
    v3_path: Path,
    v4_path: Path,
    limits: dict[str, float] | None = None,
    *,
    restoration_candidates: tuple[
        tuple[float, float, float], ...
    ] = UNIFORM_RESTORATION_CANDIDATES,
    full_frame_max_pixels: int = COMPARATIVE_QA_FULL_FRAME_MAX_PIXELS,
    tile_size: int = COMPARATIVE_QA_TILE_SIZE,
    grid_size: int = COMPARATIVE_QA_GRID_SIZE,
) -> dict[str, Any]:
    """Select the strongest safe globally uniform restoration at V3-native scale."""

    limits = limits or COMPARATIVE_QA_LIMITS
    if not restoration_candidates:
        raise V4Error("Uniform restoration candidate ladder cannot be empty.")
    for deep_weight, unsharp_amount, sigma_native in restoration_candidates:
        if not 0.0 <= deep_weight < 1.0:
            raise V4Error("Restoration Deep weights must be at least zero and below one.")
        if not 0.0 <= unsharp_amount <= 1.0:
            raise V4Error("Restoration unsharp amounts must be between zero and one.")
        gaussian_kernel_spec(sigma_native, sigma_native)

    max_halo = max(
        gaussian_kernel_spec(sigma_native, sigma_native)["radius_y"]
        for _, _, sigma_native in restoration_candidates
    )

    samples, sampling = native_comparison_samples(
        v3_path,
        v4_path,
        full_frame_max_pixels=full_frame_max_pixels,
        tile_size=tile_size,
        grid_size=grid_size,
        halo_radius=max_halo,
    )
    raw_crop_reports = [
        comparative_restoration_metrics(sample["v3"], sample["deep"], limits)
        for sample in samples
    ]
    raw_deep_diagnostic = aggregate_comparative_crop_metrics(
        raw_crop_reports,
        samples,
        limits,
    )
    raw_deep_diagnostic.update(
        {
            "selection_eligible": False,
            "role": "diagnostic_only_before_uniform_restoration",
            "decision": "diagnostic_only",
        }
    )

    attempts: list[dict[str, Any]] = []
    for candidate_index, (deep_weight, unsharp_amount, sigma_native) in enumerate(
        restoration_candidates
    ):
        native_kernel = gaussian_kernel_spec(sigma_native, sigma_native)
        crop_reports = []
        for sample in samples:
            restored_guard, _ = uniform_restoration_rgb(
                sample["v3_guard"],
                sample["deep_guard"],
                deep_weight=deep_weight,
                unsharp_amount=unsharp_amount,
                sigma_x=sigma_native,
                sigma_y=sigma_native,
            )
            core_top, core_bottom, core_left, core_right = sample["core_slice"]
            restored = restored_guard[
                core_top:core_bottom,
                core_left:core_right,
            ]
            crop_reports.append(
                comparative_restoration_metrics(sample["v3"], restored, limits)
            )
        attempt = aggregate_comparative_crop_metrics(crop_reports, samples, limits)
        attempt.update(
            {
                "candidate": (
                    "uniform_global_guarded_v3_usm_control"
                    if deep_weight == 0.0
                    else "uniform_global_v3_deep_unsharp_restoration"
                ),
                "candidate_index": candidate_index,
                "v3_weight": round(1.0 - deep_weight, 4),
                "deep_weight": round(deep_weight, 4),
                "unsharp_amount": round(unsharp_amount, 4),
                "sigma_native": float(sigma_native),
                "gaussian_native": native_kernel,
                "halo_radius_native": max_halo,
                "spatially_uniform": True,
            }
        )
        attempts.append(attempt)

    def quality_rank(attempt: dict[str, Any]) -> tuple[float, ...]:
        fractions = attempt["crop_pass_fractions"]
        return (
            1.0 if attempt["claim_v4_better"] else 0.0,
            float(fractions["retention_and_both_groups"]),
            float(fractions["retention"]),
            float(fractions["edge_strength"]),
            float(fractions["energy_detail"]),
            float(attempt["ssim_mean"]),
            -float(attempt["abs_difference_p95"]),
        )

    ranked_attempts = sorted(attempts, key=quality_rank, reverse=True)
    control = next(
        (attempt for attempt in attempts if float(attempt["deep_weight"]) == 0.0),
        None,
    )
    best_deep = next(
        (
            attempt
            for attempt in ranked_attempts
            if float(attempt["deep_weight"]) > 0.0
            and attempt["claim_v4_better"]
        ),
        None,
    )
    material_crop_gain = max(
        DEEP_MATERIAL_MIN_CROP_GAIN,
        2.0 / len(samples),
    )
    selected: dict[str, Any] | None = None
    deep_ablation_reason: str
    if control is not None and control["claim_v4_better"]:
        selected = control
        deep_ablation_reason = "no Deep candidate materially beat the passing V3+USM control"
        if best_deep is not None:
            control_fractions = control["crop_pass_fractions"]
            deep_fractions = best_deep["crop_pass_fractions"]
            full_gate_gain = (
                float(deep_fractions["retention_and_both_groups"])
                - float(control_fractions["retention_and_both_groups"])
            )
            retention_not_worse = float(deep_fractions["retention"]) >= float(
                control_fractions["retention"]
            )
            if full_gate_gain >= material_crop_gain and retention_not_worse:
                selected = best_deep
                deep_ablation_reason = (
                    "Deep candidate materially beat the V3+USM control"
                )
    elif best_deep is not None:
        selected = best_deep
        deep_ablation_reason = (
            "V3+USM control did not pass; selected the best passing Deep candidate"
        )
    else:
        deep_ablation_reason = "neither V3+USM control nor any Deep candidate passed"
    ranking = []
    for rank, attempt in enumerate(ranked_attempts, start=1):
        fractions = attempt["crop_pass_fractions"]
        ranking.append(
            {
                "rank": rank,
                "candidate_index": attempt["candidate_index"],
                "deep_weight": attempt["deep_weight"],
                "unsharp_amount": attempt["unsharp_amount"],
                "sigma_native": attempt["sigma_native"],
                "passed": attempt["claim_v4_better"],
                "full_gate_fraction": fractions["retention_and_both_groups"],
                "retention_fraction": fractions["retention"],
                "edge_strength_fraction": fractions["edge_strength"],
                "energy_detail_fraction": fractions["energy_detail"],
                "ssim_p10": attempt["ssim_mean"],
                "difference_p95_p90": attempt["abs_difference_p95"],
            }
        )

    decisive = dict(selected if selected is not None else ranked_attempts[0])
    selected_uses_deep = bool(
        selected is not None and float(selected["deep_weight"]) > 0.0
    )
    decisive.update(
        {
            "claim_v4_better": selected is not None,
            "passed": selected is not None,
            "decision": (
                (
                    "use_v4_uniform_restoration"
                    if selected_uses_deep
                    else "use_v4_uniform_v3_usm_control"
                )
                if selected is not None
                else "fallback_v3"
            ),
            "selected_deep_weight": (
                selected["deep_weight"] if selected is not None else None
            ),
            "selected_v3_weight": (
                selected["v3_weight"] if selected is not None else None
            ),
            "selected_unsharp_amount": (
                selected["unsharp_amount"] if selected is not None else None
            ),
            "selected_sigma_native": (
                selected["sigma_native"] if selected is not None else None
            ),
            "selected_candidate_index": (
                selected["candidate_index"] if selected is not None else None
            ),
            "restoration_ladder": [
                {
                    "deep_weight": deep_weight,
                    "unsharp_amount": unsharp_amount,
                    "sigma_native": sigma_native,
                }
                for deep_weight, unsharp_amount, sigma_native in restoration_candidates
            ],
            "attempted_restorations": attempts,
            "restoration_ranking": ranking,
            "deep_ablation": {
                "control_candidate_index": (
                    control["candidate_index"] if control is not None else None
                ),
                "control_passed": (
                    control["claim_v4_better"] if control is not None else None
                ),
                "control_full_gate_fraction": (
                    control["crop_pass_fractions"]["retention_and_both_groups"]
                    if control is not None
                    else None
                ),
                "best_deep_candidate_index": (
                    best_deep["candidate_index"] if best_deep is not None else None
                ),
                "best_deep_full_gate_fraction": (
                    best_deep["crop_pass_fractions"]["retention_and_both_groups"]
                    if best_deep is not None
                    else None
                ),
                "material_full_gate_gain_required": round(material_crop_gain, 6),
                "selected_uses_deep": selected_uses_deep,
                "reason": deep_ablation_reason,
            },
            "selection_policy": (
                "highest joint full-gate crop fraction, then retention, edge coverage, "
                "energy coverage, SSIM and lower difference; a passing Deep candidate "
                "must also materially beat the V3+USM ablation control"
            ),
            "raw_deep_diagnostic": raw_deep_diagnostic,
            "method": (
                "Uniform global V3 guarded-USM control and optional Deep ablations with "
                "finite Gaussian support, halo-guarded independent V3-native crops, "
                "p10/p90 retention bounds and a 75% joint spatial gate; no crop-join "
                "seams or source/x1 downsample"
            ),
            **sampling,
            "v3_path": str(v3_path.resolve()),
            "v3_sha256": sha256_file(v3_path),
            "raw_deep_path": str(v4_path.resolve()),
            "raw_deep_sha256": sha256_file(v4_path),
        }
    )
    return decisive


def verify_rendered_restoration(
    v3_path: Path,
    rendered_path: Path,
    limits: dict[str, float] | None = None,
    *,
    full_frame_max_pixels: int = COMPARATIVE_QA_FULL_FRAME_MAX_PIXELS,
    tile_size: int = COMPARATIVE_QA_TILE_SIZE,
    grid_size: int = COMPARATIVE_QA_GRID_SIZE,
) -> dict[str, Any]:
    """Re-run the authoritative native crop gate on the actual encoded proof raster."""

    limits = limits or COMPARATIVE_QA_LIMITS
    samples, sampling = native_comparison_samples(
        v3_path,
        rendered_path,
        full_frame_max_pixels=full_frame_max_pixels,
        tile_size=tile_size,
        grid_size=grid_size,
        halo_radius=0,
    )
    reports = [
        comparative_restoration_metrics(sample["v3"], sample["deep"], limits)
        for sample in samples
    ]
    verification = aggregate_comparative_crop_metrics(reports, samples, limits)
    verification.update(
        {
            **sampling,
            "method": "authoritative_native_crop_gate_on_actual_rendered_png",
            "actual_render_verified": True,
            "v3_path": str(v3_path.resolve()),
            "v3_sha256": sha256_file(v3_path),
            "rendered_path": str(rendered_path.resolve()),
            "rendered_sha256": sha256_file(rendered_path),
        }
    )
    return verification


def decide_rendered_candidate(
    pre_render_qa: dict[str, Any],
    render_verification: dict[str, Any] | None,
) -> dict[str, Any]:
    """Make the final accept/fallback decision from the actual encoded raster gate."""

    if not pre_render_qa.get("claim_v4_better"):
        return {
            "accepted": False,
            "render_attempted": False,
            "decision": "fallback_v3",
            "reason": "no_simulated_restoration_passed",
        }
    if render_verification is None:
        raise V4Error("A simulated restoration pass requires actual render verification.")
    accepted = bool(render_verification.get("claim_v4_better"))
    return {
        "accepted": accepted,
        "render_attempted": True,
        "decision": (
            pre_render_qa["decision"]
            if accepted
            else "fallback_v3_after_render_verification"
        ),
        "reason": (
            "actual_render_gate_passed"
            if accepted
            else "actual_render_verification_failed"
        ),
    }


def visual_metrics(
    source_png: Path,
    proof_png: Path,
    limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    limits = limits or QA_LIMITS
    source = rgb_array(source_png)
    proof = rgb_array(proof_png)
    if source.shape != proof.shape:
        raise V4Error(f"QA image shape mismatch: {source.shape} vs {proof.shape}.")
    difference = np.abs(source.astype(np.int16) - proof.astype(np.int16))
    mse = float(np.mean(np.square(source.astype(np.float32) - proof.astype(np.float32))))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0**2) / mse)
    edge = edge_f1(source, proof)
    metrics = {
        "mae": round(float(difference.mean()), 4),
        "abs_error_p95": round(float(np.percentile(difference, 95)), 4),
        "psnr_db": round(psnr, 4) if math.isfinite(psnr) else "inf",
        "edge_precision": round(edge["precision"], 6),
        "edge_recall": round(edge["recall"], 6),
        "edge_f1": round(edge["f1"], 6),
    }
    failures = []
    if psnr < limits["min_psnr_db"]:
        failures.append(f"PSNR {psnr:.3f} < {limits['min_psnr_db']}")
    if edge["f1"] < limits["min_edge_f1"]:
        failures.append(f"edge F1 {edge['f1']:.4f} < {limits['min_edge_f1']}")
    if metrics["abs_error_p95"] > limits["max_abs_error_p95"]:
        failures.append(
            f"p95 {metrics['abs_error_p95']:.1f} > {limits['max_abs_error_p95']}"
        )
    metrics["limits"] = limits
    metrics["passed"] = not failures
    metrics["failures"] = failures
    if failures:
        raise V4Error("V4 visual QA failed: " + "; ".join(failures))
    return metrics


def write_scribus_prefs(directory: Path, output_profile_name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "scribus150.rc"
    escaped_profile = (
        output_profile_name.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    )
    target.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE SCRIBUSRC>\n"
        "<SCRIBUSRC VERSION=\"1.5.0\">\n"
        "  <ColorManagement SoftProofOn=\"0\" SoftProofFullOn=\"0\" "
        "ColorManagementActive=\"1\" GamutCheck=\"0\" BlackPoint=\"1\" "
        "DefaultMonitorProfile=\"sRGB display profile (ICC v2.2)\" "
        f"DefaultPrinterProfile=\"{escaped_profile}\" "
        "DefaultImageRGBProfile=\"sRGB display profile (ICC v2.2)\" "
        f"DefaultImageCMYKProfile=\"{escaped_profile}\" "
        "DefaultSolidColorRGBProfile=\"sRGB display profile (ICC v2.2)\" "
        f"DefaultSolorColorCMYKProfile=\"{escaped_profile}\" "
        "DefaultIntentColors=\"1\" DefaultIntentImages=\"0\"/>\n"
        "</SCRIBUSRC>\n",
        encoding="utf-8",
    )
    return target


def export_pdfx4(
    scribus: Path,
    master_svg: Path,
    pdf_path: Path,
    work_dir: Path,
    page_width_mm: float,
    page_height_mm: float,
    bleed_mm: float,
    output_profile_name: str,
    title: str,
) -> dict[str, Any]:
    prefs_dir = work_dir / "scribus_prefs"
    write_scribus_prefs(prefs_dir, output_profile_name)
    config_path = work_dir / "scribus_job.json"
    report_path = work_dir / "scribus_report.json"
    sla_path = work_dir / "print_master.sla"
    config = {
        "svg": str(master_svg.resolve()),
        "sla": str(sla_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "report": str(report_path.resolve()),
        "page_width_mm": page_width_mm,
        "page_height_mm": page_height_mm,
        "bleed_mm": bleed_mm,
        "rgb_profile_name": DEFAULT_RGB_PROFILE,
        "output_profile_name": output_profile_name,
        "title": title,
        "author": "Local Print Image Upscaler",
        "description": "V4 PDF/X-4 print master",
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    environment = os.environ.copy()
    environment["RESIZE_V4_SCRIBUS_JOB"] = str(config_path.resolve())
    command = [
        str(scribus),
        "-g",
        "-ns",
        "-pr",
        str(prefs_dir.resolve()),
        "-py",
        str(SCRIPT_EXPORT_PDFX4.resolve()),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=scribus.parent,
        env=environment,
        text=True,
        capture_output=True,
        timeout=900,
    )
    if completed.returncode != 0:
        raise V4Error(
            "Scribus PDF/X-4 export failed. "
            f"exit={completed.returncode}; stderr={completed.stderr[-2000:]}"
        )
    if not report_path.is_file():
        raise V4Error("Scribus exited without an export report.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "ok" or not pdf_path.is_file():
        raise V4Error("Scribus did not create a valid PDF/X-4: " + json.dumps(report))
    return {
        "seconds": round(time.perf_counter() - started, 3),
        "command": command,
        "report": report,
        "stderr_tail": completed.stderr[-1000:],
    }


def stream_path_operator_count(stream: pikepdf.Stream) -> int:
    try:
        data = stream.read_bytes()
    except pikepdf.PdfError:
        return 0
    return len(re.findall(rb"(?<!\S)(?:m|l|c|v|y|h|re)(?!\S)", data))


def scan_pdf_resources(resources: Any, seen: set[tuple[int, int]]) -> dict[str, int]:
    counts = {"images": 0, "forms": 0, "path_operators": 0, "fonts": 0}
    if not resources or "/XObject" not in resources:
        if resources and "/Font" in resources:
            counts["fonts"] += len(resources["/Font"])
        return counts
    if "/Font" in resources:
        counts["fonts"] += len(resources["/Font"])
    for item in resources["/XObject"].values():
        identity = tuple(item.objgen)
        if identity != (0, 0) and identity in seen:
            continue
        if identity != (0, 0):
            seen.add(identity)
        subtype = str(item.get("/Subtype", ""))
        if subtype == "/Image":
            counts["images"] += 1
        elif subtype == "/Form":
            counts["forms"] += 1
            counts["path_operators"] += stream_path_operator_count(item)
            nested = scan_pdf_resources(item.get("/Resources", {}), seen)
            for key in counts:
                counts[key] += nested[key]
    return counts


def _pdf_box_points(box: Any, label: str) -> tuple[float, float, float, float]:
    try:
        coordinates = tuple(float(value) for value in box)
    except (TypeError, ValueError, OverflowError, pikepdf.PdfError) as exc:
        raise V4Error(f"PDF {label} is not a numeric rectangle.") from exc
    if len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
        raise V4Error(f"PDF {label} is not a finite four-coordinate rectangle.")
    left, bottom, right, top = coordinates
    if right <= left or top <= bottom:
        raise V4Error(f"PDF {label} has non-positive width or height: {coordinates!r}.")
    return left, bottom, right, top


def _box_contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> bool:
    return (
        outer[0] <= inner[0] + PDF_BOX_TOLERANCE_PT
        and outer[1] <= inner[1] + PDF_BOX_TOLERANCE_PT
        and outer[2] >= inner[2] - PDF_BOX_TOLERANCE_PT
        and outer[3] >= inner[3] - PDF_BOX_TOLERANCE_PT
    )


def validate_pdf_page_geometry(
    media_box: Any,
    trim_box: Any,
    bleed_box: Any,
    *,
    expected_page_width_mm: float | None = None,
    expected_page_height_mm: float | None = None,
    expected_bleed_mm: float | None = None,
) -> dict[str, Any]:
    """Validate real PDF rectangles, including the scaled page bleed."""
    media = _pdf_box_points(media_box, "MediaBox")
    trim = _pdf_box_points(trim_box, "TrimBox")
    bleed = _pdf_box_points(bleed_box, "BleedBox")
    if not _box_contains(media, bleed):
        raise V4Error("PDF MediaBox does not contain BleedBox.")
    if not _box_contains(bleed, trim):
        raise V4Error("PDF BleedBox does not contain TrimBox.")

    def dimensions(box: tuple[float, float, float, float]) -> tuple[float, float]:
        return box[2] - box[0], box[3] - box[1]

    media_size = dimensions(media)
    trim_size = dimensions(trim)
    bleed_size = dimensions(bleed)
    maximum_points = MAX_PDF_PAGE_MM * PDF_POINTS_PER_MM
    for label, size in (("MediaBox", media_size), ("BleedBox", bleed_size)):
        if max(size) > maximum_points + PDF_BOX_TOLERANCE_PT:
            raise V4Error(
                f"PDF {label} exceeds {MAX_PDF_PAGE_MM:g} mm: "
                f"{size[0] / PDF_POINTS_PER_MM:.3f} x "
                f"{size[1] / PDF_POINTS_PER_MM:.3f} mm."
            )

    actual_bleed_points = {
        "left": trim[0] - bleed[0],
        "bottom": trim[1] - bleed[1],
        "right": bleed[2] - trim[2],
        "top": bleed[3] - trim[3],
    }
    expected = (
        expected_page_width_mm,
        expected_page_height_mm,
        expected_bleed_mm,
    )
    if any(value is not None for value in expected) and not all(
        value is not None for value in expected
    ):
        raise V4Error("Expected PDF width, height and bleed must be supplied together.")
    if all(value is not None for value in expected):
        width_mm = float(expected_page_width_mm)
        height_mm = float(expected_page_height_mm)
        page_bleed_mm = float(expected_bleed_mm)
        if (
            not all(math.isfinite(value) for value in (width_mm, height_mm, page_bleed_mm))
            or width_mm <= 0
            or height_mm <= 0
            or page_bleed_mm < 0
        ):
            raise V4Error("Expected PDF page geometry is invalid.")

        expected_trim = (
            width_mm * PDF_POINTS_PER_MM,
            height_mm * PDF_POINTS_PER_MM,
        )
        expected_outer = (
            (width_mm + 2.0 * page_bleed_mm) * PDF_POINTS_PER_MM,
            (height_mm + 2.0 * page_bleed_mm) * PDF_POINTS_PER_MM,
        )

        def require_dimensions(
            label: str,
            actual: tuple[float, float],
            wanted: tuple[float, float],
        ) -> None:
            if any(
                abs(actual_value - wanted_value) > PDF_BOX_TOLERANCE_PT
                for actual_value, wanted_value in zip(actual, wanted)
            ):
                raise V4Error(
                    f"PDF {label} is {actual[0] / PDF_POINTS_PER_MM:.3f} x "
                    f"{actual[1] / PDF_POINTS_PER_MM:.3f} mm; expected "
                    f"{wanted[0] / PDF_POINTS_PER_MM:.3f} x "
                    f"{wanted[1] / PDF_POINTS_PER_MM:.3f} mm."
                )

        require_dimensions("TrimBox", trim_size, expected_trim)
        require_dimensions("BleedBox", bleed_size, expected_outer)
        require_dimensions("MediaBox", media_size, expected_outer)
        wanted_bleed_points = page_bleed_mm * PDF_POINTS_PER_MM
        for side, actual_points in actual_bleed_points.items():
            if abs(actual_points - wanted_bleed_points) > PDF_BOX_TOLERANCE_PT:
                raise V4Error(
                    f"PDF {side} bleed is {actual_points / PDF_POINTS_PER_MM:.3f} mm; "
                    f"expected {page_bleed_mm:.3f} mm."
                )

    def size_mm(size: tuple[float, float]) -> list[float]:
        return [round(value / PDF_POINTS_PER_MM, 6) for value in size]

    return {
        "media_box": list(media),
        "trim_box": list(trim),
        "bleed_box": list(bleed),
        "media_size_mm": size_mm(media_size),
        "trim_size_mm": size_mm(trim_size),
        "bleed_size_mm": size_mm(bleed_size),
        "actual_bleed_mm": {
            side: round(value / PDF_POINTS_PER_MM, 6)
            for side, value in actual_bleed_points.items()
        },
        "geometry_matches_expected": all(value is not None for value in expected),
    }


def validate_pdfx4(
    pdf_path: Path,
    *,
    expect_hybrid: bool = False,
    expected_page_width_mm: float | None = None,
    expected_page_height_mm: float | None = None,
    expected_bleed_mm: float | None = None,
) -> dict[str, Any]:
    try:
        pdf = pikepdf.open(pdf_path)
    except pikepdf.PdfError as exc:
        raise V4Error(f"Cannot reopen generated PDF: {exc}") from exc
    with pdf:
        if pdf.is_encrypted:
            raise V4Error("PDF/X-4 must not be encrypted.")
        version = str(pdf.pdf_version)
        if tuple(int(value) for value in version.split(".")[:2]) < (1, 6):
            raise V4Error(f"PDF version {version} is below PDF/X-4 base PDF 1.6.")
        pdfx_version = str(pdf.docinfo.get("/GTS_PDFXVersion", ""))
        if "PDF/X-4" not in pdfx_version:
            raise V4Error(f"Missing PDF/X-4 identification: {pdfx_version!r}.")
        if "/OutputIntents" not in pdf.Root or not pdf.Root.OutputIntents:
            raise V4Error("PDF/X-4 is missing OutputIntents.")
        output_intent = pdf.Root.OutputIntents[0]
        if str(output_intent.get("/S", "")) != "/GTS_PDFX":
            raise V4Error("PDF output intent subtype is not /GTS_PDFX.")
        if "/DestOutputProfile" not in output_intent:
            raise V4Error("PDF/X-4 output intent has no embedded ICC profile.")
        profile = output_intent.DestOutputProfile
        profile_bytes = profile.read_bytes()
        profile_components = int(profile.get("/N", 0))
        if profile_components not in {1, 3, 4} or len(profile_bytes) < 1024:
            raise V4Error("Embedded Output Intent ICC profile is invalid or empty.")
        if len(pdf.pages) != 1:
            raise V4Error(f"V4 expected one PDF page; found {len(pdf.pages)}.")
        page = pdf.pages[0]
        for required_box in ("/MediaBox", "/TrimBox", "/BleedBox"):
            if required_box not in page.obj:
                raise V4Error(f"PDF/X-4 page is missing {required_box}.")
        page_geometry = validate_pdf_page_geometry(
            page.obj.MediaBox,
            page.obj.TrimBox,
            page.obj.BleedBox,
            expected_page_width_mm=expected_page_width_mm,
            expected_page_height_mm=expected_page_height_mm,
            expected_bleed_mm=expected_bleed_mm,
        )
        resources = page.obj.get("/Resources", {})
        counts = scan_pdf_resources(resources, set())
        contents = page.obj.get("/Contents")
        if isinstance(contents, pikepdf.Array):
            counts["path_operators"] += sum(stream_path_operator_count(item) for item in contents)
        elif isinstance(contents, pikepdf.Stream):
            counts["path_operators"] += stream_path_operator_count(contents)
        if expect_hybrid and counts["images"] < 1:
            raise V4Error("Hybrid PDF is missing its disclosed raster detail layer.")
        if not expect_hybrid and counts["images"] != 0:
            raise V4Error(
                f"Full-vector PDF contains {counts['images']} raster image XObject(s)."
            )
        if counts["path_operators"] == 0:
            raise V4Error("PDF contains no vector path painting operations.")
        metadata_bytes = b""
        if "/Metadata" in pdf.Root:
            metadata_bytes = pdf.Root.Metadata.read_bytes()
        if b"PDF/X-4" not in metadata_bytes:
            raise V4Error("PDF XMP metadata does not identify PDF/X-4.")
        return {
            "pdf_version": version,
            "gts_pdfx_version": pdfx_version,
            "output_condition": str(output_intent.get("/OutputCondition", "")),
            "output_condition_identifier": str(
                output_intent.get("/OutputConditionIdentifier", "")
            ),
            "icc_components": profile_components,
            "icc_bytes": len(profile_bytes),
            "icc_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            **page_geometry,
            **counts,
            "xmp_pdfx4": True,
            "encrypted": False,
        }


def choose_print_geometry(
    source_size: tuple[int, int],
    source_dpi: float,
    intended_width_mm: float | None,
    bleed_mm: float = 0.0,
) -> dict[str, Any]:
    if source_size[0] <= 0 or source_size[1] <= 0 or source_dpi <= 0:
        raise V4Error("Source dimensions and DPI must be greater than zero.")
    if not math.isfinite(bleed_mm) or bleed_mm < 0:
        raise V4Error("Print bleed must be a finite non-negative value.")
    natural_width_mm = source_size[0] / source_dpi * 25.4
    final_width_mm = intended_width_mm if intended_width_mm is not None else natural_width_mm
    if not math.isfinite(final_width_mm) or final_width_mm <= 0:
        raise V4Error("Print width must be greater than zero.")
    final_height_mm = final_width_mm * source_size[1] / source_size[0]
    longest_outer_side_mm = max(final_width_mm, final_height_mm) + 2.0 * bleed_mm
    denominator = max(1, math.ceil(longest_outer_side_mm / MAX_PDF_PAGE_MM))
    page_width_mm = final_width_mm / denominator
    page_height_mm = final_height_mm / denominator
    page_bleed_mm = bleed_mm / denominator
    return {
        "natural_width_mm": natural_width_mm,
        "intended_width_mm": final_width_mm,
        "intended_height_mm": final_height_mm,
        "finished_bleed_mm": bleed_mm,
        "page_width_mm": page_width_mm,
        "page_height_mm": page_height_mm,
        "page_bleed_mm": page_bleed_mm,
        "page_outer_width_mm": page_width_mm + 2.0 * page_bleed_mm,
        "page_outer_height_mm": page_height_mm + 2.0 * page_bleed_mm,
        "print_scale_denominator": denominator,
        "print_scale": f"1:{denominator}",
    }


def safe_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value).strip(" .")
    return cleaned or "image"


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    raster_base = args.raster_base.resolve() if args.raster_base is not None else None
    baseline_arg = getattr(args, "v3_baseline", None)
    explicit_v3_baseline = baseline_arg.resolve() if baseline_arg is not None else None
    hybrid = raster_base is not None
    if not source.is_file():
        raise V4Error(f"Input does not exist: {source}")
    if not (2.0 <= args.scale <= 20.0):
        raise V4Error("V4 PNG proof scale must be between x2 and x20.")
    if args.bleed_mm < 0 or args.bleed_mm > 100:
        raise V4Error("Bleed must be between 0 and 100 mm.")
    if not (0.0 <= args.vector_opacity <= 0.5):
        raise V4Error("Hybrid vector opacity must be between 0 and 0.5.")
    if raster_base is not None and not raster_base.is_file():
        raise V4Error(f"Hybrid raster base does not exist: {raster_base}")
    if explicit_v3_baseline is not None and not hybrid:
        raise V4Error("--v3-baseline is valid only with --raster-base hybrid mode.")
    if explicit_v3_baseline is not None and not explicit_v3_baseline.is_file():
        raise V4Error(f"V3 baseline does not exist: {explicit_v3_baseline}")

    output_dir.mkdir(parents=True, exist_ok=True)
    gmic = find_gmic()
    resvg = find_resvg()
    scribus = find_scribus()
    if not SCRIPT_EXPORT_PDFX4.is_file():
        raise V4Error(f"Missing Scribus export script: {SCRIPT_EXPORT_PDFX4}")

    name = safe_name(args.name)
    scale_tag = f"{args.scale:g}".replace(".", "p")
    master_svg = output_dir / f"{name}_EDITABLE.svg"
    print_pdf = output_dir / f"{name}_PRINT_PDFX4.pdf"
    proof_png = output_dir / f"{name}_PREVIEW_x{scale_tag}.png"
    manifest_path = output_dir / "manifest.json"

    total_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="v4_work_", dir=output_dir.parent) as raw_temp:
        work_dir = Path(raw_temp)
        normalized = work_dir / "source_rgb.png"
        smart2x = work_dir / "smart2x.png"
        raw_svg = work_dir / "raw_trace.svg"
        qa_png = work_dir / "qa_source_size.png"

        source_info = inspect_and_normalize_source(source, normalized)
        source_size = tuple(source_info["size"])
        source_mp = source_size[0] * source_size[1] / 1_000_000
        if source_mp > MAX_SOURCE_MEGAPIXELS and not args.allow_huge:
            raise V4Error(
                f"V4 full-vector source is {source_mp:.1f} MP; current safe limit is "
                f"{MAX_SOURCE_MEGAPIXELS:g} MP. Use --allow-huge only after a smaller proof passes."
            )
        proof_size = tuple(int(round(value * args.scale)) for value in source_size)
        proof_mp = proof_size[0] * proof_size[1] / 1_000_000
        if proof_mp > MAX_OUTPUT_MEGAPIXELS and not args.allow_huge:
            raise V4Error(
                f"PNG proof would be {proof_size[0]}x{proof_size[1]} ({proof_mp:.1f} MP)."
            )
        geometry = choose_print_geometry(
            source_size,
            source_info["dpi_x"],
            args.width_mm,
            args.bleed_mm,
        )
        if hybrid:
            with Image.open(raster_base) as base_image:
                base_image.load()
                if base_image.size != proof_size:
                    raise V4Error(
                        f"Hybrid raster base is {base_image.size}; expected proof size {proof_size}."
                    )

        selected_raster_base = raster_base
        selected_raster_provenance: dict[str, Any] | None = None
        comparative_qa: dict[str, Any] | None = None
        if hybrid:
            source_sha256 = sha256_file(source)
            if explicit_v3_baseline is not None:
                baseline_info: dict[str, Any] | None = {
                    "path": explicit_v3_baseline,
                    "sha256": sha256_file(explicit_v3_baseline),
                    "sidecar": None,
                    "pipeline": "explicit_cli_baseline",
                }
            else:
                baseline_info = discover_v3_native(raster_base, source_sha256)

            if baseline_info is not None:
                v3_baseline = Path(baseline_info["path"])
                with Image.open(v3_baseline) as baseline_image:
                    baseline_image.load()
                    baseline_size = baseline_image.size
                source_ratio = source_size[0] / source_size[1]
                baseline_ratio = baseline_size[0] / baseline_size[1]
                if not math.isclose(source_ratio, baseline_ratio, rel_tol=0.001, abs_tol=0.001):
                    raise V4Error(
                        "V3 baseline aspect ratio does not match the current source: "
                        f"{baseline_size} vs {source_size}."
                    )

                comparative_qa = compare_v3_v4_paths(
                    v3_baseline,
                    raster_base,
                    COMPARATIVE_QA_LIMITS,
                )
                comparative_qa.update(
                    {
                        "baseline_discovery": baseline_info["pipeline"],
                        "baseline_sidecar": (
                            str(baseline_info["sidecar"])
                            if baseline_info["sidecar"] is not None
                            else None
                        ),
                        "baseline_native_size": list(baseline_size),
                    }
                )
                pre_render_passed = bool(comparative_qa["claim_v4_better"])
                restoration_accepted = False
                render_verification: dict[str, Any] | None = None
                final_render_decision = (
                    None
                    if pre_render_passed
                    else decide_rendered_candidate(comparative_qa, None)
                )
                if pre_render_passed:
                    selected_deep_weight = float(comparative_qa["selected_deep_weight"])
                    selected_unsharp_amount = float(
                        comparative_qa["selected_unsharp_amount"]
                    )
                    selected_sigma_native = float(
                        comparative_qa["selected_sigma_native"]
                    )
                    selected_uses_deep = selected_deep_weight > 0.0
                    restored_raster = work_dir / "v4_uniform_restoration_at_proof_size.png"
                    restoration_render = write_uniform_restoration(
                        v3_baseline,
                        raster_base,
                        restored_raster,
                        proof_size,
                        deep_weight=selected_deep_weight,
                        unsharp_amount=selected_unsharp_amount,
                        sigma_native=selected_sigma_native,
                    )
                    render_verification = verify_rendered_restoration(
                        v3_baseline,
                        restored_raster,
                        COMPARATIVE_QA_LIMITS,
                    )
                    final_render_decision = decide_rendered_candidate(
                        comparative_qa,
                        render_verification,
                    )
                    restoration_accepted = final_render_decision["accepted"]
                    comparative_qa.update(
                        {
                            "pre_render_candidate_passed": True,
                            "actual_render_verification": render_verification,
                            "actual_render_verified_and_passed": restoration_accepted,
                            "restoration_render": restoration_render,
                            "claim_v4_better": restoration_accepted,
                            "passed": restoration_accepted,
                            "decision": final_render_decision["decision"],
                        }
                    )
                    if restoration_accepted:
                        selected_raster_base = restored_raster
                        raster_selection = (
                            "v4_uniform_restoration"
                            if selected_uses_deep
                            else "v4_uniform_v3_usm"
                        )
                        selected_raster_provenance = {
                            "selection": raster_selection,
                            "selected_pipeline": (
                                "V4_UNIFORM_RESTORATION"
                                if selected_uses_deep
                                else "V4_UNIFORM_GUARDED_USM_FROM_V3"
                            ),
                            "method": restoration_render["method"],
                            "formula": restoration_render["formula"],
                            "spatially_uniform": True,
                            "deep_visible": selected_uses_deep,
                            "raw_deep_role": (
                                "uniform_blend_input"
                                if selected_uses_deep
                                else "diagnostic_ablation_only_not_visible"
                            ),
                            "v3_weight": comparative_qa["selected_v3_weight"],
                            "deep_weight": comparative_qa["selected_deep_weight"],
                            "unsharp_amount": comparative_qa[
                                "selected_unsharp_amount"
                            ],
                            "sigma_native": comparative_qa["selected_sigma_native"],
                            "gaussian_proof": restoration_render["gaussian"],
                            "proof_scale_x": restoration_render["proof_scale_x"],
                            "proof_scale_y": restoration_render["proof_scale_y"],
                            "v3_source_path": str(v3_baseline.resolve()),
                            "v3_source_sha256": baseline_info["sha256"],
                            "raw_deep_path": str(raster_base.resolve()),
                            "raw_deep_sha256": comparative_qa["raw_deep_sha256"],
                            "embedded_restoration_sha256": restoration_render["sha256"],
                            "actual_render_verification_sha256": render_verification[
                                "rendered_sha256"
                            ],
                            "actual_render_gate_passed": True,
                            "actual_render_crop_pass_fractions": render_verification[
                                "crop_pass_fractions"
                            ],
                            "quality_gate_decision": comparative_qa["decision"],
                            "deep_ablation": comparative_qa["deep_ablation"],
                        }
                        comparative_qa.update(
                            {
                                "selected_layer": raster_selection,
                                "fallback_applied": False,
                                "final_output_not_worse_than_v3": True,
                            }
                        )
                        if selected_uses_deep:
                            qa_selection = (
                                f"selected {selected_deep_weight:.1%} Deep, "
                                f"USM {selected_unsharp_amount:g}, sigma-native "
                                f"{selected_sigma_native:g}"
                            )
                        else:
                            qa_selection = (
                                "selected the V3 guarded-USM control and rejected the "
                                f"Deep ablation; USM {selected_unsharp_amount:g}, "
                                f"sigma-native {selected_sigma_native:g}"
                            )
                        print(
                            "[QA] Actual rendered restoration passed native-x4 verification; "
                            f"{qa_selection}.",
                            flush=True,
                        )

                if not restoration_accepted:
                    fallback_reason = final_render_decision["reason"]
                    v3_fallback = work_dir / "v3_fallback_at_proof_size.png"
                    write_resized_rgb(v3_baseline, v3_fallback, proof_size)
                    selected_raster_base = v3_fallback
                    selected_raster_provenance = {
                        "selection": "v3_baseline_fallback",
                        "selected_pipeline": "V3_HIGH_NATIVE_BASELINE",
                        "candidate_rejected": True,
                        "v3_source_path": str(v3_baseline.resolve()),
                        "v3_source_sha256": baseline_info["sha256"],
                        "raw_deep_path": str(raster_base.resolve()),
                        "raw_deep_sha256": comparative_qa["raw_deep_sha256"],
                        "embedded_fallback_sha256": sha256_file(v3_fallback),
                        "quality_gate_decision": final_render_decision["decision"],
                        "fallback_reason": fallback_reason,
                        "actual_render_gate_passed": (
                            render_verification["claim_v4_better"]
                            if render_verification is not None
                            else None
                        ),
                    }
                    comparative_qa.update(
                        {
                            "claim_v4_better": False,
                            "passed": False,
                            "pre_render_candidate_passed": pre_render_passed,
                            "actual_render_verification": render_verification,
                            "actual_render_verified_and_passed": False,
                            "decision": final_render_decision["decision"],
                            "selected_layer": "v3_baseline_fallback",
                            "fallback_applied": True,
                            "fallback_reason": fallback_reason,
                            "fallback_sha256": sha256_file(v3_fallback),
                            "final_output_not_worse_than_v3": True,
                        }
                    )
                    print(
                        "[QA] Uniform restoration was not accepted by the complete gate; "
                        "selecting the deterministic V3 fallback.",
                        flush=True,
                    )
            else:
                comparative_qa = {
                    "passed": False,
                    "claim_v4_better": False,
                    "retention_passed": None,
                    "meaningfully_sharper": None,
                    "decision": "unverified_candidate",
                    "selected_layer": "v4_candidate_unverified",
                    "fallback_applied": False,
                    "final_output_not_worse_than_v3": None,
                    "reason": (
                        "No V3 baseline was supplied and the V4 provenance sidecar did not "
                        "identify one; V4 superiority is not claimed."
                    ),
                    "limits": COMPARATIVE_QA_LIMITS,
                }
                selected_raster_provenance = {
                    "selection": "v4_candidate_unverified",
                    "selected_pipeline": "UNVERIFIED_RASTER_CANDIDATE",
                    "raw_candidate_path": str(raster_base.resolve()),
                    "raw_candidate_sha256": sha256_file(raster_base),
                    "quality_gate_decision": "unverified_candidate",
                }
                print(
                    "[QA] No V3 baseline found; keeping the candidate without a V4-better claim.",
                    flush=True,
                )

        print("[1/6] G'MIC Smart Upscale 2x for sub-pixel tracing...", flush=True)
        preprocess = preprocess_smart2x(gmic, normalized, smart2x)
        print("[2/6] VTracer stacked spline reconstruction...", flush=True)
        trace = trace_full_vector(smart2x, raw_svg)
        print(
            "[3/6] Building "
            + (
                "the selected raster master with an independent vector edit layer..."
                if hybrid
                else "a raster-free SVG master..."
            ),
            flush=True,
        )
        vector = normalize_svg(
            raw_svg,
            master_svg,
            source_size,
            geometry["page_width_mm"],
            geometry["page_height_mm"],
            geometry["intended_width_mm"],
            geometry["print_scale_denominator"],
            f"{name} - V4 {'hybrid' if hybrid else 'full-vector'} editable master",
            raster_base=selected_raster_base,
            vector_opacity=args.vector_opacity if hybrid else 1.0,
            raster_provenance=selected_raster_provenance,
        )
        print("[4/6] Rendering deterministic SVG proofs with resvg...", flush=True)
        proof_render = render_svg(resvg, master_svg, proof_png, proof_size)
        qa_started = time.perf_counter()
        with Image.open(proof_png) as proof_image:
            proof_image.load()
            proof_image.convert("RGB").resize(
                source_size, Image.Resampling.LANCZOS
            ).save(qa_png, format="PNG", compress_level=4)
        qa_render = {
            "seconds": round(time.perf_counter() - qa_started, 3),
            "method": "Lanczos downsample from the final xN proof",
        }
        metrics = visual_metrics(
            normalized,
            qa_png,
            HYBRID_SOURCE_QA_LIMITS if hybrid else QA_LIMITS,
        )
        metrics.update(
            {
                "advisory_only": hybrid,
                "gate_role": (
                    "source-fidelity diagnostic; direct V3/V4 QA is authoritative"
                    if hybrid
                    else "full-vector source-fidelity gate"
                ),
            }
        )
        retention_metrics: dict[str, Any] | None = None
        if hybrid:
            base_qa_png = work_dir / "hybrid_base_source_size.png"
            with Image.open(selected_raster_base) as base_image:
                base_image.load()
                base_image.convert("RGB").resize(
                    source_size, Image.Resampling.LANCZOS
                ).save(base_qa_png, format="PNG", compress_level=4)
            retention_metrics = visual_metrics(
                base_qa_png,
                qa_png,
                HYBRID_RETENTION_LIMITS,
            )
        print("[5/6] Exporting ISO PDF/X-4 with Scribus and embedded Output Intent...", flush=True)
        page_bleed_mm = geometry["page_bleed_mm"]
        pdf_export = export_pdfx4(
            scribus,
            master_svg,
            print_pdf,
            work_dir,
            geometry["page_width_mm"],
            geometry["page_height_mm"],
            page_bleed_mm,
            args.profile_name,
            (
                f"{name} - V4 PDF/X-4 {'hybrid' if hybrid else 'vector'} print master "
                f"({geometry['intended_width_mm']:g} x {geometry['intended_height_mm']:g} mm "
                f"finished, {geometry['print_scale']})"
            ),
        )
        placement = pdf_export.get("report", {}).get("placement")
        if not isinstance(placement, dict):
            raise V4Error("Scribus export report is missing artwork placement QA.")
        pdf_placement_qa = {
            **placement,
            "method": "Scribus post-import object geometry against trim plus bleed",
            "passed": True,
        }
        print("[6/6] Structural self-check (not a PDF/X certification)...", flush=True)
        pdf_qa = validate_pdfx4(
            print_pdf,
            expect_hybrid=hybrid,
            expected_page_width_mm=geometry["page_width_mm"],
            expected_page_height_mm=geometry["page_height_mm"],
            expected_bleed_mm=page_bleed_mm,
        )

        if hybrid and comparative_qa is not None:
            if comparative_qa["claim_v4_better"]:
                if selected_raster_provenance["selection"] == "v4_uniform_v3_usm":
                    quality_claim = (
                        "A spatially uniform guarded-USM restoration of the validated V3 AI "
                        "master passed simulated native-x4 selection and an authoritative "
                        "second gate on the actual rendered PNG. The Deep ablation was rejected "
                        "and is not visible or embedded."
                    )
                else:
                    quality_claim = (
                        "A spatially uniform V3/Deep/unsharp restoration passed simulated "
                        "native-x4 selection and an authoritative second gate on the actual "
                        "rendered PNG; it is selected with real vector edit paths."
                    )
            elif comparative_qa["fallback_applied"]:
                quality_claim = (
                    "No V4 quality-superiority claim: the candidate failed the direct "
                    "V3-to-V4 gate, so the validated V3 baseline was selected with real "
                    "vector edit paths."
                )
            else:
                quality_claim = (
                    "No V4 quality-superiority claim: no validated V3 baseline was available "
                    "for direct comparison; the candidate remains explicitly unverified."
                )
        else:
            quality_claim = "SVG and PDF contain real vector paths and zero embedded raster images."

        if hybrid:
            selection = selected_raster_provenance["selection"]
            pipeline = {
                "v4_uniform_restoration": "V4_UNIFORM_RESTORATION_PRINT_HYBRID",
                "v4_uniform_v3_usm": "V4_UNIFORM_V3_USM_PRINT_HYBRID",
                "v3_baseline_fallback": "V4_PRINT_HYBRID_V3_FALLBACK",
                "v4_candidate_unverified": "V4_PRINT_HYBRID_UNVERIFIED_RASTER",
            }[selection]
        else:
            pipeline = "V4_PRINT_FULL_VECTOR"

        config = {
            "pipeline": pipeline,
            "config_version": 8,
            "trace_scale": TRACE_SCALE,
            "vector_opacity": args.vector_opacity if hybrid else 1.0,
            "gmic": GMIC_PARAMS,
            "vtracer": VTRACER_PARAMS,
            "qa_limits": HYBRID_SOURCE_QA_LIMITS if hybrid else QA_LIMITS,
            "hybrid_retention_limits": HYBRID_RETENTION_LIMITS if hybrid else None,
            "comparative_qa_limits": COMPARATIVE_QA_LIMITS if hybrid else None,
            "uniform_restoration_ladder": (
                [
                    {
                        "deep_weight": deep_weight,
                        "unsharp_amount": unsharp_amount,
                        "sigma_native": sigma_native,
                    }
                    for deep_weight, unsharp_amount, sigma_native in (
                        UNIFORM_RESTORATION_CANDIDATES
                    )
                ]
                if hybrid
                else None
            ),
            "gaussian_truncate": GAUSSIAN_TRUNCATE if hybrid else None,
            "restoration_stripe_height": RESTORATION_STRIPE_HEIGHT if hybrid else None,
            "deep_material_min_crop_gain": (
                DEEP_MATERIAL_MIN_CROP_GAIN if hybrid else None
            ),
            "comparative_native_sampling": (
                {
                    "full_frame_max_pixels": COMPARATIVE_QA_FULL_FRAME_MAX_PIXELS,
                    "tile_size": COMPARATIVE_QA_TILE_SIZE,
                    "grid_size": COMPARATIVE_QA_GRID_SIZE,
                }
                if hybrid
                else None
            ),
            "output_profile_name": args.profile_name,
        }
        manifest = {
            "pipeline": pipeline,
            "claim": quality_claim,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "source_sha256": sha256_file(source),
            "source_size": list(source_size),
            "source_mode": source_info["mode"],
            "source_dpi_x": source_info["dpi_x"],
            "source_normalization": {
                "input_icc_present": source_info["input_icc_present"],
                "colour_conversion": source_info["colour_conversion"],
                "alpha_composited_on_white": source_info["alpha_composited_on_white"],
                "exif_orientation_applied": source_info["exif_orientation_applied"],
                "output_space": "sRGB",
            },
            "normalized_sha256": sha256_file(normalized),
            "proof_scale": args.scale,
            "proof_size": list(proof_size),
            "print_geometry": geometry,
            "finished_bleed_mm": args.bleed_mm,
            "page_bleed_mm": page_bleed_mm,
            "config": config,
            "config_sha256": sha256_json(config),
            "tools": {
                "python": sys.version,
                "gmic_path": str(gmic),
                "gmic_sha256": sha256_file(gmic),
                "resvg_path": str(resvg),
                "resvg_sha256": sha256_file(resvg),
                "scribus_path": str(scribus),
                "scribus_sha256": sha256_file(scribus),
                "vtracer": importlib.metadata.version("vtracer"),
                "opencv": cv2.__version__,
                "pillow": importlib.metadata.version("Pillow"),
                "pikepdf": importlib.metadata.version("pikepdf"),
            },
            "timings": {
                "preprocess": preprocess["seconds"],
                "trace": trace["seconds"],
                "qa_render": qa_render["seconds"],
                "proof_render": proof_render["seconds"],
                "pdf_export": pdf_export["seconds"],
                "total": round(time.perf_counter() - total_started, 3),
            },
            "vector": vector,
            "selected_raster_provenance": selected_raster_provenance,
            "visual_qa": metrics,
            "hybrid_retention_qa": retention_metrics,
            "comparative_v3_v4_qa": comparative_qa,
            "pdf_placement_qa": pdf_placement_qa,
            "pdfx4_qa": pdf_qa,
            "structural_qa_scope": (
                "Self-check only: PDF 1.6/PDF-X metadata, OutputIntent ICC, boxes, "
                "encryption and object structure. Final conformity requires an external "
                "PDF/X-4 preflight such as Acrobat/callas/GWG Sign & Display."
            ),
            "outputs": {
                "svg": {"name": master_svg.name, "sha256": sha256_file(master_svg), "bytes": master_svg.stat().st_size},
                "pdf": {"name": print_pdf.name, "sha256": sha256_file(print_pdf), "bytes": print_pdf.stat().st_size},
                "png": {"name": proof_png.name, "sha256": sha256_file(proof_png), "bytes": proof_png.stat().st_size},
            },
            "print_note": (
                "PDF/X-4 uses the named Output Intent above. For PVC/tarpaulin or a specific RIP, "
                "the print shop's own ICC profile and requested scale are authoritative. "
                "The bundled ISO Coated profile is generic and is not a press proof for banner media."
            ),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return manifest


def main() -> int:
    args = build_parser().parse_args()
    manifest = run(args)
    print("\nV4 COMPLETE", flush=True)
    for kind, payload in manifest["outputs"].items():
        print(f"  {kind.upper()}: {args.output_dir.resolve() / payload['name']}", flush=True)
    print(
        f"  QA: PSNR {manifest['visual_qa']['psnr_db']} dB, "
        f"edge F1 {manifest['visual_qa']['edge_f1']}, "
        f"{manifest['vector']['path_count']:,} paths",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (V4Error, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"\nV4 ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
