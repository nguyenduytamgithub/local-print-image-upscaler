"""Fast local upscale engine used by the unified RESIZE launcher.

Real-ESRGAN produces one neural x4 master on the GPU. Final scales other
than x4 are resized once from that master with Lanczos. The public launcher
renders into a temporary job directory, so this engine never touches INPUT.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image


Image.MAX_IMAGE_PIXELS = 500_000_000
AI_NATIVE_SCALE = 4
MIN_FINAL_SCALE = 2
MAX_FINAL_SCALE = 20
MODELS = {
    "sharp": "realesrgan-x4plus",
    "graphic": "realesrgan-x4plus-anime",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V2 FAST: Real-ESRGAN x4 tren GPU, sau do Lanczos den ty le cuoi."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("scale", type=float, help="He so cuoi tu x2 den x20")
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--preset", choices=sorted(MODELS), default="sharp")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--final-sharpening", type=float, default=0.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Cho phep ket qua lon hon 500 megapixel.",
    )
    parser.add_argument("--discard-native-x4", action="store_true")
    return parser


def find_realesrgan(here: Path) -> Path:
    candidates = [
        here / "tools" / "realesrgan-ncnn-vulkan-20220424" / "realesrgan-ncnn-vulkan.exe",
        here / "tools" / "realesrgan-ncnn-vulkan.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit("Thieu Real-ESRGAN portable trong APP\\engines\\V2\\tools.")


def find_gmic(here: Path) -> Path:
    # APP/engines/V2 -> APP
    app_dir = here.parents[1]
    candidates = [
        app_dir / "shared" / "tools" / "gmic" / "gmic-4.0.2-cli-win64" / "gmic.exe",
        app_dir / "shared" / "tools" / "gmic" / "gmic.exe",
        here / "tools" / "gmic" / "gmic.exe",
    ]
    system = shutil.which("gmic")
    if system:
        candidates.insert(0, Path(system))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit("Thieu G'MIC trong APP\\shared\\tools\\gmic.")


def percent(value: float) -> str:
    return f"{value * 100:.8g}%"


def main() -> int:
    args = build_parser().parse_args()
    here = Path(__file__).resolve().parent
    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"Khong tim thay anh: {source}")
    if not (MIN_FINAL_SCALE <= args.scale <= MAX_FINAL_SCALE):
        raise SystemExit("V2 nhan he so tu x2 den x20; khuyen dung x4 hoac x10.")
    if args.tile < 32:
        raise SystemExit("--tile phai tu 32 tro len.")

    with Image.open(source) as image:
        source_size = image.size
    final_size = tuple(int(round(value * args.scale)) for value in source_size)
    megapixels = final_size[0] * final_size[1] / 1_000_000
    if megapixels > 500 and not args.force:
        raise SystemExit(
            f"Anh cuoi {megapixels:.1f} MP; dung --force chi khi ban chac chan."
        )

    if args.output:
        target = args.output.resolve()
    else:
        scale_tag = f"{args.scale:g}".replace(".", "p")
        target = here / "output" / f"{source.stem}_V2_FAST_x{scale_tag}.png"
    if target == source:
        raise SystemExit("Output khong duoc trung input.")
    target.parent.mkdir(parents=True, exist_ok=True)
    native = target.with_name(f"{target.stem}_AI_NATIVE_X4.png")

    exe = find_realesrgan(here)
    model_dir = exe.parent / "models"
    command = [
        str(exe),
        "-i", str(source),
        "-o", str(native),
        "-n", MODELS[args.preset],
        "-s", str(AI_NATIVE_SCALE),
        "-t", str(args.tile),
        "-m", str(model_dir),
        "-g", str(args.gpu),
        "-j", "1:2:2",
        "-f", "png",
        "-v",
    ]
    print(f"[1/3] V2 FAST: AI native x4, preset={args.preset}, tile={args.tile}...", flush=True)
    subprocess.run(command, check=True)

    expected_native = tuple(value * AI_NATIVE_SCALE for value in source_size)
    with Image.open(native) as image:
        if image.size != expected_native:
            raise SystemExit(f"Sai kich thuoc AI x4: {image.size}, can {expected_native}")

    if abs(args.scale - AI_NATIVE_SCALE) < 1e-9:
        if target.exists():
            target.unlink()
        native.replace(target)
    else:
        print(f"[2/3] Lanczos x4 -> x{args.scale:g}...", flush=True)
        remainder = args.scale / AI_NATIVE_SCALE
        resize = [
            str(find_gmic(here)),
            str(native),
            "-resize", f"{percent(remainder)},{percent(remainder)},100%,100%,6",
        ]
        if args.final_sharpening > 0:
            resize += ["-sharpen", f"{args.final_sharpening:g}"]
        resize += ["-cut", "0,255", "-output", str(target)]
        subprocess.run(resize, check=True)

    print("[3/3] Kiem tra output...", flush=True)
    with Image.open(target) as image:
        if image.size != final_size:
            raise SystemExit(f"Sai kich thuoc cuoi: {image.size}, can {final_size}")
    if args.discard_native_x4 and native.exists():
        native.unlink()

    print(f"DONE: {target}", flush=True)
    print(f"Kich thuoc: {final_size[0]}x{final_size[1]} ({megapixels:.1f} MP)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Cong cu ngoai chay loi, ma {exc.returncode}") from exc
