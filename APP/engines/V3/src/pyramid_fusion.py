"""Uniform three-model frequency fusion for print-oriented SR.

Low frequencies (color/large shapes) come from a fidelity transformer, middle
frequencies from HAT, and the finest detail from the sharp V2 result. The same
formula is applied to every pixel; there are no semantic masks or repaired
patches.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from blend_outputs import linear_to_srgb, srgb_to_linear
from upscale_engine import sha256_file, standard_srgb_profile


Image.MAX_IMAGE_PIXELS = 500_000_000
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def read_rgb(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        metadata = {
            "size": image.size,
            "icc_profile": image.info.get("icc_profile") or standard_srgb_profile(),
            "dpi": image.info.get("dpi"),
        }
    return rgb, metadata


def luminance(rgb_u8: np.ndarray) -> np.ndarray:
    return np.sum(srgb_to_linear(rgb_u8.astype(np.float32) / 255.0) * LUMA, axis=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path, help="Swin/fidelity native x4 PNG")
    parser.add_argument("middle", type=Path, help="HAT native x4 PNG")
    parser.add_argument("detail", type=Path, help="V2/strong-detail native x4 PNG")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sigma-high", type=float, default=1.25)
    parser.add_argument("--sigma-low", type=float, default=6.0)
    parser.add_argument("--middle-gain", type=float, default=1.0)
    parser.add_argument("--detail-gain", type=float, default=1.0)
    parser.add_argument(
        "--limiter-margin",
        type=float,
        default=0.01,
        help="Maximum luma excursion beyond the three model predictions.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = [args.base.resolve(), args.middle.resolve(), args.detail.resolve()]
    output = args.output.resolve()
    if output in set(paths):
        raise SystemExit("Fusion output must be a new file.")
    if output.exists() and not args.force:
        raise SystemExit("Output exists; add --force to replace this V3 fusion.")
    if not (0 < args.sigma_high < args.sigma_low):
        raise SystemExit("Require 0 < --sigma-high < --sigma-low.")

    base_u8, metadata = read_rgb(paths[0])
    middle_u8, middle_metadata = read_rgb(paths[1])
    detail_u8, detail_metadata = read_rgb(paths[2])
    if metadata["size"] != middle_metadata["size"] or metadata["size"] != detail_metadata["size"]:
        raise SystemExit("All native x4 inputs must have identical dimensions.")

    print("Converting model outputs to linear-light luminance...")
    base_linear = srgb_to_linear(base_u8.astype(np.float32) / 255.0)
    base_y = np.sum(base_linear * LUMA, axis=2)
    middle_y = luminance(middle_u8)
    detail_y = luminance(detail_u8)
    del base_u8, middle_u8, detail_u8

    print("Building low/middle/high frequency bands...")
    low_base = cv2.GaussianBlur(base_y, (0, 0), args.sigma_low, borderType=cv2.BORDER_REFLECT_101)
    low_middle = cv2.GaussianBlur(middle_y, (0, 0), args.sigma_low, borderType=cv2.BORDER_REFLECT_101)
    high_middle = cv2.GaussianBlur(middle_y, (0, 0), args.sigma_high, borderType=cv2.BORDER_REFLECT_101)
    smooth_detail = cv2.GaussianBlur(detail_y, (0, 0), args.sigma_high, borderType=cv2.BORDER_REFLECT_101)
    target_y = (
        low_base
        + args.middle_gain * (high_middle - low_middle)
        + args.detail_gain * (detail_y - smooth_detail)
    )

    lower = np.minimum(np.minimum(base_y, middle_y), detail_y) - args.limiter_margin
    upper = np.maximum(np.maximum(base_y, middle_y), detail_y) + args.limiter_margin
    target_y = np.clip(target_y, np.maximum(0.0, lower), np.minimum(1.0, upper))
    ratio = (target_y + 1e-5) / (base_y + 1e-5)
    fused_linear = base_linear * ratio[:, :, None]
    dark = base_y < 1e-4
    if np.any(dark):
        fused_linear[dark] = target_y[dark, None]
    clipped_fraction = float(np.mean((fused_linear < 0.0) | (fused_linear > 1.0)))
    fused = linear_to_srgb(np.clip(fused_linear, 0.0, 1.0))
    fused_u8 = np.clip(np.rint(fused * 255.0), 0, 255).astype(np.uint8)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part.png")
    save_args: dict[str, object] = {
        "format": "PNG",
        "compress_level": 6,
        "icc_profile": metadata["icc_profile"],
    }
    if metadata["dpi"]:
        save_args["dpi"] = metadata["dpi"]
    Image.fromarray(fused_u8, mode="RGB").save(temporary, **save_args)
    os.replace(temporary, output)

    manifest = {
        "pipeline": "AI_V3_LAB uniform three-band fusion",
        "formula": "Swin low frequencies + HAT middle frequencies + V2 high frequencies; Swin chroma",
        "base": str(paths[0]),
        "base_sha256": sha256_file(paths[0]),
        "middle": str(paths[1]),
        "middle_sha256": sha256_file(paths[1]),
        "detail": str(paths[2]),
        "detail_sha256": sha256_file(paths[2]),
        "sigma_high": args.sigma_high,
        "sigma_low": args.sigma_low,
        "middle_gain": args.middle_gain,
        "detail_gain": args.detail_gain,
        "limiter_margin": args.limiter_margin,
        "clipped_channel_fraction": clipped_fraction,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "size": list(metadata["size"]),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"DONE: {output}")
    print(f"Clipped channel fraction: {clipped_fraction:.6%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
