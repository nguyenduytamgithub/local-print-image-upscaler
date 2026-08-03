"""AI V3: fidelity-first local GPU upscaling for print graphics."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch
from PIL import Image

from src.upscale_engine import (
    MODEL_SPECS,
    resize_png_lanczos,
    save_png_atomic,
    sha256_file,
    upscale_tiled_with_fallback,
)


Image.MAX_IMAGE_PIXELS = 500_000_000


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="AI V3 local CUDA upscale: one neural x4 pass, then Lanczos only when final scale exceeds x4."
    )
    command.add_argument("input", type=Path)
    command.add_argument("scale", type=float, help="Final scale. Use 4 for native AI QA or 10 for print master.")
    command.add_argument("output", nargs="?", type=Path)
    command.add_argument("--model", choices=sorted(MODEL_SPECS), default="hat-fidelity")
    command.add_argument("--tile", type=int, help="Input tile size; default is model-specific.")
    command.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Shared pixels between neighboring tiles. 128 means about 64 px context per side.",
    )
    command.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="auto")
    command.add_argument("--force", action="store_true", help="Replace an existing V3 output.")
    return command


def main() -> int:
    args = parser().parse_args()
    here = Path(__file__).resolve().parent
    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"Input not found: {source}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is not available in the AI_V3_LAB environment.")
    if not (4 <= args.scale <= 20):
        raise SystemExit("V3 accepts final scales from x4 through x20.")

    spec = MODEL_SPECS[args.model]
    tile = args.tile or spec.default_tile
    if tile % spec.alignment:
        raise SystemExit(f"--tile must be divisible by {spec.alignment} for {spec.family}.")
    if args.overlap % spec.alignment:
        raise SystemExit(f"--overlap must be divisible by {spec.alignment} for {spec.family}.")
    if args.overlap <= 0 or args.overlap * 2 > tile:
        raise SystemExit("--overlap must be positive and no greater than half of --tile.")

    model_path = here / "models" / spec.filename
    if not model_path.is_file():
        raise SystemExit(f"Model checkpoint is missing: {model_path}")

    scale_tag = f"{args.scale:g}".replace(".", "p")
    if args.output:
        target = args.output.resolve()
    else:
        target = here / "output" / f"{source.stem}_ai_v3_{spec.key}_x{scale_tag}.png"
    native = target if abs(args.scale - spec.native_scale) < 1e-9 else target.with_name(
        f"{target.stem}_AI_NATIVE_X4.png"
    )
    outputs = {target, native}
    if source in outputs:
        raise SystemExit("Output must not overwrite the source image.")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise SystemExit("Output already exists; add --force to replace only this V3 result: " + str(existing[0]))

    with Image.open(source) as opened:
        opened.load()
        image = opened.convert("RGB")
        source_size = image.size
        icc_profile = opened.info.get("icc_profile")
        dpi = opened.info.get("dpi")

    final_size = tuple(round(value * args.scale) for value in source_size)
    final_megapixels = final_size[0] * final_size[1] / 1_000_000
    started = time.perf_counter()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {spec.label}")
    print(f"Source: {source_size[0]}x{source_size[1]}")
    print("Stage 1/3: uniform neural x4 inference...")
    result, run = upscale_tiled_with_fallback(
        image=image,
        model_path=model_path,
        spec=spec,
        requested_tile=tile,
        overlap=args.overlap,
        dtype_name=args.dtype,
    )
    save_png_atomic(result, native, icc_profile=icc_profile, dpi=dpi)
    del result

    if target != native:
        print(f"Stage 2/3: Lanczos x4 -> x{args.scale:g}; no second AI pass...")
        resize_png_lanczos(native, target, final_size)
    else:
        print("Stage 2/3: native x4 is the requested final scale.")

    print("Stage 3/3: verify dimensions and write manifest...")
    with Image.open(native) as check:
        expected_native = tuple(value * spec.native_scale for value in source_size)
        if check.size != expected_native:
            raise RuntimeError(f"Native output has size {check.size}, expected {expected_native}.")
    with Image.open(target) as check:
        if check.size != final_size:
            raise RuntimeError(f"Final output has size {check.size}, expected {final_size}.")

    manifest = {
        "pipeline": "AI_V3_LAB",
        "policy": "one uniform neural x4 pass; Lanczos only for the remaining scale",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_size": list(source_size),
        "model_key": spec.key,
        "model_label": spec.label,
        "model_file": str(model_path),
        "model_sha256": sha256_file(model_path),
        "native_scale": spec.native_scale,
        "final_scale": args.scale,
        "native_output": str(native),
        "native_sha256": sha256_file(native),
        "final_output": str(target),
        "final_sha256": sha256_file(target),
        "final_size": list(final_size),
        "final_megapixels": round(final_megapixels, 3),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "spandrel": version("spandrel"),
        "python": platform.python_version(),
        "run": run,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path = target.with_suffix(target.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"DONE: {target}")
    if native != target:
        print(f"NATIVE AI x4: {native}")
    print(f"Final: {final_size[0]}x{final_size[1]} ({final_megapixels:.1f} MP)")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        raise SystemExit(130)
