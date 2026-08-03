"""One-command AI V3 master pipeline.

Runs three complementary official x4 models on the whole image, fuses their
frequency bands uniformly, and uses Lanczos only for the remaining print scale.
Temporary model outputs are automatically removed after a successful run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

from src.upscale_engine import resize_png_lanczos, sha256_file


Image.MAX_IMAGE_PIXELS = 500_000_000
NATIVE_SCALE = 4
MIN_FINAL_SCALE = 2
MAX_FINAL_SCALE = 20


def run(command: list[str]) -> None:
    print("\n> " + " ".join(f'\"{part}\"' if " " in part else part for part in command), flush=True)
    subprocess.run(command, check=True)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI V3 MASTER: Swin low band + HAT middle band + Real-ESRGAN detail, uniformly over the image."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("scale", type=float, help="Final scale from x2 through x20; x4 or x10 is recommended.")
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--keep-components",
        action="store_true",
        help="Also retain the three individual x4 model outputs for research.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"Input not found: {source}")
    if not (MIN_FINAL_SCALE <= args.scale <= MAX_FINAL_SCALE):
        raise SystemExit("Final scale must be from x2 through x20.")
    if args.tile < 256 or args.overlap <= 0 or args.overlap * 2 > args.tile:
        raise SystemExit("Use tile >= 256 and overlap no greater than half the tile.")

    scale_tag = f"{args.scale:g}".replace(".", "p")
    target = (
        args.output.resolve()
        if args.output
        else here / "output" / f"{source.stem}_AI_V3_MASTER_x{scale_tag}.png"
    )
    native = target if abs(args.scale - NATIVE_SCALE) < 1e-9 else target.with_name(
        f"{target.stem}_AI_NATIVE_X4.png"
    )
    if source in {target, native}:
        raise SystemExit("Output must not overwrite the source.")
    for path in {target, native}:
        if path.exists() and not args.force:
            raise SystemExit(f"Output exists: {path}. Add --force to replace only this V3 output.")

    with Image.open(source) as image:
        source_size = image.size
    final_size = tuple(round(value * args.scale) for value in source_size)
    python = str(Path(sys.executable).resolve())
    started = time.perf_counter()
    work_root = here / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="v3_run_", dir=work_root) as temporary_raw:
        temporary = Path(temporary_raw)
        swin = temporary / "01_swin_fidelity_x4.png"
        hat = temporary / "02_hat_sharper_x4.png"
        detail = temporary / "03_realesrgan_detail_x4.png"
        fused = temporary / "04_v3_master_fused_x4.png"

        jobs = (
            ("swin2sr-fidelity", swin),
            ("hat-sharper", hat),
            ("realesrgan-detail", detail),
        )
        for index, (model, output) in enumerate(jobs, 1):
            print(f"\nMODEL {index}/3: {model}", flush=True)
            run(
                [
                    python,
                    "-B",
                    str(here / "upsize_ai_v3.py"),
                    str(source),
                    "4",
                    str(output),
                    "--model",
                    model,
                    "--tile",
                    str(args.tile),
                    "--overlap",
                    str(args.overlap),
                    "--dtype",
                    "auto",
                    "--force",
                ]
            )

        print("\nFUSION: low=Swin, middle=HAT, detail=Real-ESRGAN", flush=True)
        run(
            [
                python,
                "-B",
                str(here / "src" / "pyramid_fusion.py"),
                str(swin),
                str(hat),
                str(detail),
                str(fused),
                "--sigma-high",
                "1.25",
                "--sigma-low",
                "6",
                "--middle-gain",
                "1.0",
                "--detail-gain",
                "1.0",
                "--limiter-margin",
                "0.008",
                "--force",
            ]
        )
        atomic_copy(fused, native)

        component_hashes = {
            "swin2sr_fidelity": sha256_file(swin),
            "hat_sharper": sha256_file(hat),
            "realesrgan_detail": sha256_file(detail),
        }
        if args.keep_components:
            component_dir = target.parent / f"{target.stem}_COMPONENTS_X4"
            component_dir.mkdir(parents=True, exist_ok=True)
            for path in (swin, hat, detail):
                atomic_copy(path, component_dir / path.name)

    if target != native:
        print(f"\nFINAL RESIZE: neural x4 -> x{args.scale:g} with Lanczos (no second AI pass)", flush=True)
        resize_png_lanczos(native, target, final_size)

    with Image.open(target) as image:
        if image.size != final_size:
            raise RuntimeError(f"Final output is {image.size}, expected {final_size}.")
    manifest = {
        "pipeline": "AI_V3_MASTER",
        "policy": "three uniform neural x4 outputs fused by frequency; Lanczos only beyond x4",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_size": list(source_size),
        "models": {
            "low": "Swin2SR Realworld PSNR x4",
            "middle": "HAT Real GAN sharper x4",
            "detail": "Real-ESRGAN x4plus PyTorch",
        },
        "component_sha256": component_hashes,
        "fusion": {
            "sigma_high": 1.25,
            "sigma_low": 6.0,
            "middle_gain": 1.0,
            "detail_gain": 1.0,
            "limiter_margin": 0.008,
            "color_base": "Swin2SR linear-light chromaticity",
        },
        "tile": args.tile,
        "overlap": args.overlap,
        "native_scale": NATIVE_SCALE,
        "native_output": str(native),
        "native_sha256": sha256_file(native),
        "final_scale": args.scale,
        "final_output": str(target),
        "final_sha256": sha256_file(target),
        "final_size": list(final_size),
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path = target.with_suffix(target.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDONE: {target}")
    if target != native:
        print(f"NATIVE AI MASTER x4: {native}")
    print(f"MANIFEST: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"A V3 stage failed with exit code {error.returncode}.") from error
