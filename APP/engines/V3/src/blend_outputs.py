"""Uniform whole-image fusion of a fidelity SR output and a sharper SR output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from upscale_engine import sha256_file, standard_srgb_profile


Image.MAX_IMAGE_PIXELS = 500_000_000


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1 / 2.4) - 0.055)


def blend_rows(base: np.ndarray, sharp: np.ndarray, alpha: float, mode: str) -> np.ndarray:
    base_linear = srgb_to_linear(base.astype(np.float32) / 255.0)
    sharp_linear = srgb_to_linear(sharp.astype(np.float32) / 255.0)
    if mode == "rgb":
        result = (1.0 - alpha) * base_linear + alpha * sharp_linear
    elif mode == "luma":
        coefficients = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        base_y = np.sum(base_linear * coefficients, axis=2)
        sharp_y = np.sum(sharp_linear * coefficients, axis=2)
        target_y = (1.0 - alpha) * base_y + alpha * sharp_y
        ratio = (target_y + 1e-5) / (base_y + 1e-5)
        result = base_linear * ratio[:, :, None]
        very_dark = base_y < 1e-4
        if np.any(very_dark):
            result[very_dark] = (
                (1.0 - alpha) * base_linear[very_dark] + alpha * sharp_linear[very_dark]
            )
    else:
        raise ValueError(mode)
    result = linear_to_srgb(np.clip(result, 0.0, 1.0))
    return np.clip(np.rint(result * 255.0), 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blend HAT detail uniformly into a Swin2SR fidelity base in linear light."
    )
    parser.add_argument("base", type=Path, help="Fidelity/base x4 PNG")
    parser.add_argument("sharp", type=Path, help="Sharper/perceptual x4 PNG")
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.25, help="Sharper contribution, from 0 through 1")
    parser.add_argument("--mode", choices=("luma", "rgb"), default="luma")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_path = args.base.resolve()
    sharp_path = args.sharp.resolve()
    output_path = args.output.resolve()
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("--alpha must be from 0 through 1.")
    if output_path.exists() and not args.force:
        raise SystemExit("Output exists; add --force to replace this V3 blend.")
    if output_path in {base_path, sharp_path}:
        raise SystemExit("Blend output must be a new file.")

    with Image.open(base_path) as base_image, Image.open(sharp_path) as sharp_image:
        if base_image.size != sharp_image.size:
            raise SystemExit(f"Input sizes differ: {base_image.size} vs {sharp_image.size}")
        base = np.asarray(base_image.convert("RGB"), dtype=np.uint8)
        sharp = np.asarray(sharp_image.convert("RGB"), dtype=np.uint8)
        icc = base_image.info.get("icc_profile") or standard_srgb_profile()
        dpi = base_image.info.get("dpi")

    output = np.empty_like(base)
    rows_per_chunk = 256
    for top in range(0, base.shape[0], rows_per_chunk):
        bottom = min(base.shape[0], top + rows_per_chunk)
        output[top:bottom] = blend_rows(base[top:bottom], sharp[top:bottom], args.alpha, args.mode)
        print(f"rows {bottom}/{base.shape[0]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part.png")
    save_args: dict[str, object] = {
        "format": "PNG",
        "compress_level": 6,
        "icc_profile": icc,
    }
    if dpi:
        save_args["dpi"] = dpi
    Image.fromarray(output, mode="RGB").save(temporary, **save_args)
    os.replace(temporary, output_path)
    manifest = {
        "pipeline": "AI_V3_LAB uniform global fusion",
        "base": str(base_path),
        "base_sha256": sha256_file(base_path),
        "sharp": str(sharp_path),
        "sharp_sha256": sha256_file(sharp_path),
        "mode": args.mode,
        "sharper_alpha": args.alpha,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "size": list(reversed(output.shape[:2])),
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"DONE: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
