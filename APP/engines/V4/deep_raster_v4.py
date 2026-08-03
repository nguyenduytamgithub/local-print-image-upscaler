"""V4 Deep Print raster stage.

The input is a validated V3 native x4 master. A second HAT inference pass is
performed tile by tile, but each neural x4 prediction is resampled and blended
directly into the requested final xN canvas. This keeps HAT's restoration gain
without allocating or saving an unnecessary full x16 intermediate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageCms


Image.MAX_IMAGE_PIXELS = 500_000_000
HERE = Path(__file__).resolve().parent
V3_DIR = HERE.parent / "V3"
V3_SRC = V3_DIR / "src"
sys.path.insert(0, str(V3_SRC))

from upscale_engine import (  # noqa: E402
    MODEL_SPECS,
    cosine_axis_weight,
    evenly_spaced_starts,
    is_bfloat16_compatibility_error,
    is_cuda_oom,
    load_descriptor,
    overlap_sizes,
    padded_source_tensor,
    tile_config_candidates,
)


MODEL_KEY = "hat-sharper"
DEEP_CONFIG_VERSION = 1
MAX_OUTPUT_MEGAPIXELS = 500.0


class DeepRasterError(RuntimeError):
    """Expected V4 deep-raster validation or runtime error."""


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="V4 recursive HAT print restoration")
    command.add_argument("source", type=Path, help="Original source image")
    command.add_argument("v3_native", type=Path, help="Validated V3 native x4 PNG")
    command.add_argument("scale", type=float, help="Final scale relative to original")
    command.add_argument("output", type=Path)
    command.add_argument("--tile", type=int, default=512)
    command.add_argument("--overlap", type=int, default=128)
    command.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="auto")
    command.add_argument("--allow-huge", action="store_true")
    command.add_argument("--force", action="store_true")
    return command


def sha256_lower(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standard_srgb_profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def save_png_atomic(
    array: np.ndarray,
    target: Path,
    *,
    icc_profile: bytes | None,
    dpi: tuple[float, float] | None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part.png")
    image = Image.fromarray(array, mode="RGB")
    options: dict[str, object] = {
        "format": "PNG",
        "compress_level": 4,
        "icc_profile": icc_profile or standard_srgb_profile(),
    }
    if dpi:
        options["dpi"] = dpi
    image.save(temporary, **options)
    os.replace(temporary, target)


def axis_output_bounds(start: int, tile: int, ratio: float) -> tuple[int, int]:
    return round(start * ratio), round((start + tile) * ratio)


def axis_target_geometry(
    source_length: int,
    padded_length: int,
    target_length: int,
) -> tuple[float, int]:
    """Return one axis' exact scale ratio and padded output length.

    Final width and height are rounded to integer pixels independently.  Their
    ratios can therefore differ slightly for non-square images at fractional
    scales and must not be collapsed into one averaged ratio.
    """

    if source_length <= 0 or padded_length <= 0 or target_length <= 0:
        raise DeepRasterError("Source, padded, and target axis lengths must be positive.")
    ratio = target_length / source_length
    if ratio > 5.0:
        raise DeepRasterError(f"Unsupported V3-to-final axis ratio: {ratio:g}.")
    return ratio, round(padded_length * ratio)


def seam_positions(starts: list[int], tile: int, ratio: float, limit: int) -> list[int]:
    positions = [
        round(((left + tile + right) / 2.0) * ratio)
        for left, right in zip(starts, starts[1:])
    ]
    return [position for position in positions if 1 <= position < limit]


def seam_continuity_probe(
    array: np.ndarray,
    *,
    vertical_positions: list[int],
    horizontal_positions: list[int],
) -> dict[str, object]:
    """Look for systematic one-pixel jumps at the centres of tile overlaps.

    This is an advisory continuity probe, not a perceptual quality score. Natural
    artwork can contain an edge at a sampled coordinate, so the result is logged
    rather than used as a hard rejection gate.
    """

    offsets = (-30, -20, -10, 10, 20, 30)

    def boundary_step(axis: int, position: int) -> float:
        if axis == 1:
            left = array[:, position - 1, :].astype(np.int16)
            right = array[:, position, :].astype(np.int16)
        else:
            left = array[position - 1, :, :].astype(np.int16)
            right = array[position, :, :].astype(np.int16)
        return float(np.abs(right - left).mean())

    def one_axis(axis: int, positions: list[int], limit: int) -> dict[str, object]:
        rows: list[dict[str, float | int]] = []
        for position in positions:
            neighbours = [
                boundary_step(axis, position + offset)
                for offset in offsets
                if 1 <= position + offset < limit
            ]
            local_median = float(np.median(neighbours)) if neighbours else 0.0
            centre = boundary_step(axis, position)
            ratio = centre / max(1e-9, local_median)
            rows.append(
                {
                    "position": position,
                    "mean_step": round(centre, 6),
                    "local_median": round(local_median, 6),
                    "ratio": round(ratio, 6),
                }
            )
        ratios = [float(row["ratio"]) for row in rows]
        return {
            "count": len(rows),
            "median_ratio": round(float(np.median(ratios)), 6) if ratios else 1.0,
            "max_ratio": round(max(ratios), 6) if ratios else 1.0,
            "samples": rows,
        }

    vertical = one_axis(1, vertical_positions, array.shape[1])
    horizontal = one_axis(0, horizontal_positions, array.shape[0])
    likely_continuous = (
        float(vertical["median_ratio"]) <= 1.15
        and float(horizontal["median_ratio"]) <= 1.15
        and float(vertical["max_ratio"]) <= 1.75
        and float(horizontal["max_ratio"]) <= 1.75
    )
    return {
        "method": "one-pixel boundary step versus six nearby boundaries",
        "advisory_only": True,
        "likely_continuous": likely_continuous,
        "vertical": vertical,
        "horizontal": horizontal,
    }


def recursive_hat_once(
    image: Image.Image,
    target_size: tuple[int, int],
    *,
    model_path: Path,
    tile: int,
    overlap: int,
    dtype_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    spec = MODEL_SPECS[MODEL_KEY]
    descriptor, dtype, use_amp_bfloat16, dtype_label = load_descriptor(
        model_path, dtype_name, spec
    )
    if int(descriptor.scale) != 4:
        raise DeepRasterError(f"HAT model scale is x{descriptor.scale}, expected x4.")

    source, (source_w, source_h) = padded_source_tensor(image, tile, spec.alignment)
    padded_h, padded_w = source.shape[-2:]
    ratio_x, padded_target_w = axis_target_geometry(
        source_w, padded_w, target_size[0]
    )
    ratio_y, padded_target_h = axis_target_geometry(
        source_h, padded_h, target_size[1]
    )
    output = torch.zeros((3, padded_target_h, padded_target_w), dtype=torch.float32)
    weights = torch.zeros((1, padded_target_h, padded_target_w), dtype=torch.float32)
    x_starts = evenly_spaced_starts(padded_w, tile, overlap)
    y_starts = evenly_spaced_starts(padded_h, tile, overlap)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    total = len(x_starts) * len(y_starts)
    done = 0

    with torch.inference_mode():
        for yi, top in enumerate(y_starts):
            top_overlap, bottom_overlap = overlap_sizes(y_starts, tile, yi)
            out_top, out_bottom = axis_output_bounds(top, tile, ratio_y)
            patch_h = out_bottom - out_top
            top_blend = round(top_overlap * ratio_y)
            bottom_blend = round(bottom_overlap * ratio_y)
            wy = cosine_axis_weight(patch_h, top_blend, bottom_blend)
            for xi, left in enumerate(x_starts):
                left_overlap, right_overlap = overlap_sizes(x_starts, tile, xi)
                out_left, out_right = axis_output_bounds(left, tile, ratio_x)
                patch_w = out_right - out_left
                left_blend = round(left_overlap * ratio_x)
                right_blend = round(right_overlap * ratio_x)
                wx = cosine_axis_weight(patch_w, left_blend, right_blend)
                blend = wy[:, None] * wx[None, :]

                patch = source[:, :, top : top + tile, left : left + tile].contiguous()
                patch = patch.to(device="cuda:0", dtype=dtype, non_blocking=False)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=use_amp_bfloat16,
                ):
                    prediction = descriptor(patch)
                    if tuple(prediction.shape[-2:]) != (patch_h, patch_w):
                        prediction = functional.interpolate(
                            prediction,
                            size=(patch_h, patch_w),
                            mode="bicubic",
                            align_corners=False,
                            antialias=True,
                        )
                if not torch.isfinite(prediction).all():
                    raise DeepRasterError("HAT produced NaN or infinity.")
                prediction = prediction[0].float().clamp_(0.0, 1.0).cpu()
                output[:, out_top:out_bottom, out_left:out_right] += prediction * blend
                weights[:, out_top:out_bottom, out_left:out_right] += blend
                del patch, prediction, blend
                done += 1
                print(f"  deep tile {done}/{total}", flush=True)

    output.div_(weights.clamp_min_(1e-6))
    output = output[:, : target_size[1], : target_size[0]]
    array = (
        output.permute(1, 2, 0)
        .mul_(255.0)
        .round_()
        .clamp_(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    metadata = {
        "tile": tile,
        "overlap": overlap,
        "tile_count": total,
        "dtype": dtype_label,
        "v3_to_final_ratio_x": round(ratio_x, 6),
        "v3_to_final_ratio_y": round(ratio_y, 6),
        "seconds": round(time.perf_counter() - started, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "padded_v3_size": [padded_w, padded_h],
        "vertical_seam_positions": seam_positions(
            x_starts, tile, ratio_x, target_size[0]
        ),
        "horizontal_seam_positions": seam_positions(
            y_starts, tile, ratio_y, target_size[1]
        ),
    }
    del descriptor, source, output, weights
    gc.collect()
    torch.cuda.empty_cache()
    return array, metadata


def recursive_hat_with_fallback(
    image: Image.Image,
    target_size: tuple[int, int],
    *,
    model_path: Path,
    requested_tile: int,
    overlap: int,
    dtype_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    spec = MODEL_SPECS[MODEL_KEY]
    errors: list[str] = []
    dtype_attempts = [dtype_name, "fp32"] if dtype_name == "auto" else [dtype_name]
    for effective_dtype in dtype_attempts:
        for tile, effective_overlap in tile_config_candidates(
            requested_tile, spec.alignment, overlap
        ):
            print(
                f"Trying V4 Deep HAT: tile={tile}, overlap={effective_overlap}, "
                f"dtype={effective_dtype}",
                flush=True,
            )
            try:
                result, metadata = recursive_hat_once(
                    image,
                    target_size,
                    model_path=model_path,
                    tile=tile,
                    overlap=effective_overlap,
                    dtype_name=effective_dtype,
                )
                if effective_dtype != dtype_name:
                    metadata["dtype_fallback_from"] = dtype_name
                return result, metadata
            except RuntimeError as error:
                if effective_dtype == "auto" and is_bfloat16_compatibility_error(error):
                    errors.append(f"BF16 compatibility: {error}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                if not is_cuda_oom(error):
                    raise
                errors.append(f"dtype {effective_dtype}, tile {tile}: {error}")
                print("CUDA OOM; retrying a smaller tile.", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
    raise DeepRasterError("All V4 Deep tile sizes failed: " + " | ".join(errors))


def main() -> int:
    args = parser().parse_args()
    source_path = args.source.resolve()
    native_path = args.v3_native.resolve()
    output_path = args.output.resolve()
    if not source_path.is_file() or not native_path.is_file():
        raise DeepRasterError("Source or V3 native master is missing.")
    if output_path.exists() and not args.force:
        raise DeepRasterError(f"Output exists: {output_path}")
    if not (2.0 <= args.scale <= 20.0):
        raise DeepRasterError("Final scale must be between x2 and x20.")
    if not torch.cuda.is_available():
        raise DeepRasterError("V4 Deep Print requires an NVIDIA CUDA GPU.")

    with Image.open(source_path) as source_image:
        source_image.load()
        source_size = source_image.size
        icc_profile = source_image.info.get("icc_profile")
        dpi = source_image.info.get("dpi")
    with Image.open(native_path) as native_image:
        native_image.load()
        native_size = native_image.size
        image = native_image.convert("RGB")
    expected_native = (source_size[0] * 4, source_size[1] * 4)
    if native_size != expected_native:
        raise DeepRasterError(
            f"V3 native master is {native_size}; expected exact x4 {expected_native}."
        )
    target_size = tuple(round(value * args.scale) for value in source_size)
    megapixels = target_size[0] * target_size[1] / 1_000_000
    if megapixels > MAX_OUTPUT_MEGAPIXELS and not args.allow_huge:
        raise DeepRasterError(
            f"V4 Deep output is {megapixels:.1f} MP; use --allow-huge only after a proof passes."
        )

    spec = MODEL_SPECS[MODEL_KEY]
    model_path = V3_DIR / "models" / spec.filename
    if not model_path.is_file():
        raise DeepRasterError(f"Missing HAT model: {model_path}")
    if args.tile < 256 or args.overlap <= 0 or args.overlap * 2 > args.tile:
        raise DeepRasterError("Use tile >=256 and overlap no greater than half the tile.")

    started = time.perf_counter()
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"V4 Deep source: V3 x4 {native_size[0]}x{native_size[1]}", flush=True)
    print(f"V4 Deep target: x{args.scale:g} {target_size[0]}x{target_size[1]}", flush=True)
    result, run = recursive_hat_with_fallback(
        image,
        target_size,
        model_path=model_path,
        requested_tile=args.tile,
        overlap=args.overlap,
        dtype_name=args.dtype,
    )
    run["seam_continuity_qa"] = seam_continuity_probe(
        result,
        vertical_positions=list(run.pop("vertical_seam_positions")),
        horizontal_positions=list(run.pop("horizontal_seam_positions")),
    )
    save_png_atomic(result, output_path, icc_profile=icc_profile, dpi=dpi)
    del result
    with Image.open(output_path) as check:
        check.load()
        if check.size != target_size:
            raise DeepRasterError(f"Saved output is {check.size}; expected {target_size}.")

    manifest = {
        "pipeline": "V4_DEEP_RECURSIVE_HAT",
        "config_version": DEEP_CONFIG_VERSION,
        "engine_sha256": sha256_lower(Path(__file__).resolve()),
        "policy": (
            "validated V3 native x4 master followed by a second uniform HAT pass; "
            "neural predictions are antialiased and blended directly into final xN"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "source_sha256": sha256_lower(source_path),
        "source_size": list(source_size),
        "v3_native": str(native_path),
        "v3_native_sha256": sha256_lower(native_path),
        "v3_native_size": list(native_size),
        "model": spec.label,
        "model_path": str(model_path),
        "model_sha256": sha256_lower(model_path),
        "final_scale": args.scale,
        "final_size": list(target_size),
        "final_megapixels": round(megapixels, 3),
        "output": str(output_path),
        "output_sha256": sha256_lower(output_path),
        "run": run,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "spandrel": version("spandrel"),
        "python": platform.python_version(),
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"V4 DEEP DONE: {output_path}", flush=True)
    print(f"MANIFEST: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DeepRasterError, RuntimeError) as error:
        print(f"V4 DEEP ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error
