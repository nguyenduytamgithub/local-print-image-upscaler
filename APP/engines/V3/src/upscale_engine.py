"""GPU tiled super-resolution engine used by AI_V3_LAB.

The engine intentionally applies one model uniformly to the whole image. Tiles
are only a memory-management detail: overlapping results are merged with smooth
cosine weights so that a tile border cannot become a visible hard seam.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageCms
from spandrel import ModelLoader


Image.MAX_IMAGE_PIXELS = 500_000_000


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    family: str
    label: str
    default_tile: int
    alignment: int
    native_scale: int = 4


MODEL_SPECS = {
    "hat-sharper": ModelSpec(
        key="hat-sharper",
        filename="Real_HAT_GAN_sharper.pth",
        family="HAT",
        label="HAT Real GAN x4 - sharper perceptual",
        default_tile=512,
        alignment=16,
    ),
    "swin2sr-fidelity": ModelSpec(
        key="swin2sr-fidelity",
        filename="Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth",
        family="Swin2SR",
        label="Swin2SR real-world x4 - PSNR/fidelity",
        default_tile=512,
        alignment=8,
    ),
    "realesrgan-detail": ModelSpec(
        key="realesrgan-detail",
        filename="RealESRGAN_x4plus.pth",
        family="RealESRGAN",
        label="Real-ESRGAN x4plus - PyTorch detail",
        default_tile=512,
        alignment=4,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def choose_dtype(requested: str, descriptor: object) -> torch.dtype:
    if requested == "fp32":
        return torch.float32
    if requested == "fp16":
        if not getattr(descriptor, "supports_half", False):
            raise RuntimeError("This model does not declare FP16 support.")
        return torch.float16
    if requested == "bf16":
        if not getattr(descriptor, "supports_bfloat16", False):
            raise RuntimeError("This model does not declare BF16 support.")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("This GPU does not provide native BF16 support.")
        return torch.bfloat16
    if requested != "auto":
        raise ValueError(f"Unknown dtype: {requested}")
    if getattr(descriptor, "supports_half", False):
        return torch.float16
    if (
        getattr(descriptor, "supports_bfloat16", False)
        and torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        return torch.bfloat16
    return torch.float32


def evenly_spaced_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Return fixed-size tile starts while distributing the final remainder.

    Distributing the remainder avoids one unusually wide overlap at the last
    tile, which can otherwise make that strip look different from the rest.
    """

    if length <= tile:
        return [0]
    stride = tile - overlap
    intervals = math.ceil((length - tile) / stride)
    span = length - tile
    return [round(index * span / intervals) for index in range(intervals + 1)]


def overlap_sizes(starts: list[int], tile: int, index: int) -> tuple[int, int]:
    left = 0 if index == 0 else starts[index - 1] + tile - starts[index]
    right = 0 if index == len(starts) - 1 else starts[index] + tile - starts[index + 1]
    return max(0, left), max(0, right)


def cosine_axis_weight(length: int, left: int, right: int) -> torch.Tensor:
    weight = torch.ones(length, dtype=torch.float32)
    if left:
        phase = (torch.arange(left, dtype=torch.float32) + 0.5) / left
        weight[:left] *= 0.5 - 0.5 * torch.cos(math.pi * phase)
    if right:
        phase = (torch.arange(right, dtype=torch.float32) + 0.5) / right
        weight[-right:] *= torch.flip(0.5 - 0.5 * torch.cos(math.pi * phase), dims=(0,))
    return weight.clamp_min_(1e-6)


def pad_right_bottom_safely(
    tensor: torch.Tensor,
    pad_right: int,
    pad_bottom: int,
) -> torch.Tensor:
    """Mirror-pad large gaps in legal steps; replicate only for a 1-pixel axis.

    PyTorch reflection padding requires each padding amount to be smaller than
    the current axis. Tiny inputs therefore cannot jump directly to a 512px
    tile. Repeated reflection preserves a natural boundary while progressively
    growing the tensor until the remaining padding is legal.
    """

    remaining_right = pad_right
    remaining_bottom = pad_bottom
    while remaining_right or remaining_bottom:
        current_h, current_w = tensor.shape[-2:]
        step_right = min(remaining_right, max(0, current_w - 1))
        step_bottom = min(remaining_bottom, max(0, current_h - 1))
        if step_right == 0 and step_bottom == 0:
            return functional.pad(
                tensor,
                (0, remaining_right, 0, remaining_bottom),
                mode="replicate",
            )
        tensor = functional.pad(tensor, (0, step_right, 0, step_bottom), mode="reflect")
        remaining_right -= step_right
        remaining_bottom -= step_bottom
    return tensor


def padded_source_tensor(image: Image.Image, tile: int, alignment: int) -> tuple[torch.Tensor, tuple[int, int]]:
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0)
    source_h, source_w = tensor.shape[-2:]
    padded_h = max(tile, math.ceil(source_h / alignment) * alignment)
    padded_w = max(tile, math.ceil(source_w / alignment) * alignment)
    pad_bottom = padded_h - source_h
    pad_right = padded_w - source_w
    if pad_bottom or pad_right:
        tensor = pad_right_bottom_safely(tensor, pad_right, pad_bottom)
    return tensor, (source_w, source_h)


def tile_config_candidates(requested: int, alignment: int, overlap: int) -> list[tuple[int, int]]:
    values = [requested, 384, 320, 256, 192, 160, 128]
    candidates: list[tuple[int, int]] = []
    overlap_ratio = overlap / requested
    for value in values:
        value = value - value % alignment
        fallback_overlap = round((value * overlap_ratio) / alignment) * alignment
        fallback_overlap = max(alignment, min(fallback_overlap, value // 2))
        pair = (value, fallback_overlap)
        if value <= requested and value >= 64 and pair not in candidates:
            candidates.append(pair)
    if not candidates:
        raise ValueError("Tile must be at least twice the overlap and respect model alignment.")
    return candidates


def is_cuda_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text and ("cuda" in text or "gpu" in text)


def is_bfloat16_compatibility_error(error: BaseException) -> bool:
    text = str(error).lower()
    return "bfloat16" in text and (
        "expected scalar type" in text
        or "not implemented" in text
        or "unsupported" in text
    )


def load_descriptor(model_path: Path, dtype_name: str, spec: ModelSpec):
    descriptor = ModelLoader().load_from_file(model_path)
    use_amp_bfloat16 = False
    # Swin2SR constructs a float32 attention mask during forward. Converting all
    # parameters to BF16 therefore causes a mixed matmul error in Spandrel 0.4.2.
    # CUDA autocast keeps that mask safe while accelerating supported operations.
    if (
        spec.family == "Swin2SR"
        and dtype_name in {"auto", "bf16"}
        and descriptor.supports_bfloat16
        and torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        dtype = torch.float32
        use_amp_bfloat16 = True
    else:
        dtype = choose_dtype(dtype_name, descriptor)
    descriptor.eval()
    descriptor.to(device=torch.device("cuda:0"), dtype=dtype)
    dtype_label = "amp-bf16" if use_amp_bfloat16 else str(dtype).removeprefix("torch.")
    return descriptor, dtype, use_amp_bfloat16, dtype_label


def upscale_tiled_once(
    image: Image.Image,
    model_path: Path,
    spec: ModelSpec,
    tile: int,
    overlap: int,
    dtype_name: str,
    progress: Callable[[str], None] = print,
) -> tuple[np.ndarray, dict[str, object]]:
    descriptor, dtype, use_amp_bfloat16, dtype_label = load_descriptor(model_path, dtype_name, spec)
    if int(descriptor.scale) != spec.native_scale:
        raise RuntimeError(f"Model scale is x{descriptor.scale}, expected x{spec.native_scale}.")

    source, (source_w, source_h) = padded_source_tensor(image, tile, spec.alignment)
    padded_h, padded_w = source.shape[-2:]
    x_starts = evenly_spaced_starts(padded_w, tile, overlap)
    y_starts = evenly_spaced_starts(padded_h, tile, overlap)
    scale = spec.native_scale
    output = torch.zeros((3, padded_h * scale, padded_w * scale), dtype=torch.float32)
    weights = torch.zeros((1, padded_h * scale, padded_w * scale), dtype=torch.float32)

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
            wy = cosine_axis_weight(tile * scale, top_overlap * scale, bottom_overlap * scale)
            for xi, left in enumerate(x_starts):
                left_overlap, right_overlap = overlap_sizes(x_starts, tile, xi)
                wx = cosine_axis_weight(tile * scale, left_overlap * scale, right_overlap * scale)
                blend = wy[:, None] * wx[None, :]

                patch = source[:, :, top : top + tile, left : left + tile].contiguous()
                patch = patch.to(device="cuda:0", dtype=dtype, non_blocking=False)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=use_amp_bfloat16,
                ):
                    prediction = descriptor(patch)
                if not torch.isfinite(prediction).all():
                    raise RuntimeError("Model output contains NaN or infinity.")
                expected = (tile * scale, tile * scale)
                if tuple(prediction.shape[-2:]) != expected:
                    raise RuntimeError(
                        f"Unexpected model tile output {tuple(prediction.shape[-2:])}; expected {expected}."
                    )
                prediction = prediction[0].float().clamp_(0.0, 1.0).cpu()
                out_top = top * scale
                out_left = left * scale
                out_bottom = out_top + tile * scale
                out_right = out_left + tile * scale
                output[:, out_top:out_bottom, out_left:out_right] += prediction * blend
                weights[:, out_top:out_bottom, out_left:out_right] += blend
                del patch, prediction, blend

                done += 1
                progress(f"  tile {done}/{total}")

    output.div_(weights.clamp_min_(1e-6))
    output = output[:, : source_h * scale, : source_w * scale]
    array = (
        output.permute(1, 2, 0)
        .mul_(255.0)
        .round_()
        .clamp_(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    elapsed = time.perf_counter() - started
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    metadata = {
        "tile": tile,
        "overlap": overlap,
        "tile_count": total,
        "dtype": dtype_label,
        "seconds": round(elapsed, 3),
        "peak_vram_gib": round(peak_vram, 3),
        "padded_size": [padded_w, padded_h],
    }
    del descriptor, source, output, weights
    gc.collect()
    torch.cuda.empty_cache()
    return array, metadata


def upscale_tiled_with_fallback(
    image: Image.Image,
    model_path: Path,
    spec: ModelSpec,
    requested_tile: int,
    overlap: int,
    dtype_name: str,
    progress: Callable[[str], None] = print,
) -> tuple[np.ndarray, dict[str, object]]:
    errors: list[str] = []
    dtype_attempts = [dtype_name]
    if dtype_name == "auto":
        # Some transformer checkpoints advertise BF16 support but construct a
        # float32 attention mask at runtime. A real one-tile forward pass is the
        # authoritative compatibility test, so auto mode retries FP32 cleanly.
        dtype_attempts.append("fp32")
    for effective_dtype in dtype_attempts:
        for tile, effective_overlap in tile_config_candidates(requested_tile, spec.alignment, overlap):
            progress(
                f"Trying {spec.label}: tile={tile}, overlap={effective_overlap}, dtype={effective_dtype}"
            )
            try:
                result, metadata = upscale_tiled_once(
                    image=image,
                    model_path=model_path,
                    spec=spec,
                    tile=tile,
                    overlap=effective_overlap,
                    dtype_name=effective_dtype,
                    progress=progress,
                )
                if effective_dtype != dtype_name:
                    metadata["dtype_fallback_from"] = dtype_name
                return result, metadata
            except RuntimeError as error:
                if effective_dtype == "auto" and is_bfloat16_compatibility_error(error):
                    errors.append(f"BF16 compatibility: {error}")
                    progress("BF16 forward is incompatible for this checkpoint; retrying safely in FP32.")
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                if not is_cuda_oom(error):
                    raise
                errors.append(f"dtype {effective_dtype}, tile {tile}: {error}")
                progress(f"CUDA OOM at tile={tile}; retrying smaller tile.")
                gc.collect()
                torch.cuda.empty_cache()
    raise RuntimeError("All tile sizes ran out of VRAM: " + " | ".join(errors))


def standard_srgb_profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def save_png_atomic(array: np.ndarray, path: Path, icc_profile: bytes | None, dpi: tuple[float, float] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part.png")
    image = Image.fromarray(array, mode="RGB")
    save_args: dict[str, object] = {
        "format": "PNG",
        "compress_level": 6,
        "icc_profile": icc_profile or standard_srgb_profile(),
    }
    if dpi:
        save_args["dpi"] = dpi
    image.save(temporary, **save_args)
    os.replace(temporary, path)


def resize_png_lanczos(source: Path, target: Path, size: tuple[int, int]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part.png")
    with Image.open(source) as image:
        icc = image.info.get("icc_profile") or standard_srgb_profile()
        dpi = image.info.get("dpi")
        resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)
        args: dict[str, object] = {
            "format": "PNG",
            "compress_level": 6,
            "icc_profile": icc,
        }
        if dpi:
            args["dpi"] = dpi
        resized.save(temporary, **args)
    os.replace(temporary, target)
