"""Single public command for the local V2 through V7 image engines."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import math
import msvcrt
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError


Image.MAX_IMAGE_PIXELS = 500_000_000
MIN_SCALE = 2.0
MAX_SCALE = 20.0
MAX_MEGAPIXELS = 500.0
HARD_RASTER_MEGAPIXELS = 750.0
V4_SOFT_SOURCE_MEGAPIXELS = 12.0
V4_HARD_SOURCE_MEGAPIXELS = 64.0
V4_HARD_NATIVE_MEGAPIXELS = HARD_RASTER_MEGAPIXELS
V4_HARD_OUTPUT_MEGAPIXELS = HARD_RASTER_MEGAPIXELS
V5_SOFT_OUTPUT_MEGAPIXELS = 120.0
V5_HARD_OUTPUT_MEGAPIXELS = 300.0
V7_SOFT_OUTPUT_MEGAPIXELS = 120.0
V7_HARD_OUTPUT_MEGAPIXELS = 300.0
MIN_SOURCE_DPI = 10.0
MAX_SOURCE_DPI = 2_400.0
V3_CACHE_CONFIG_VERSION = 2
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
INPUT_DIR = ROOT_DIR / "INPUT"
OUTPUT_DIR = ROOT_DIR / "OUTPUT"
WORK_DIR = APP_DIR / "work"
PYTHON = APP_DIR / "engines" / "V3" / ".venv" / "Scripts" / "python.exe"
PYTHON_V4 = APP_DIR / "engines" / "V4" / ".venv" / "Scripts" / "python.exe"
PYTHON_V5_LOCAL = APP_DIR / "engines" / "V5" / ".venv" / "Scripts" / "python.exe"
PYTHON_V5 = PYTHON_V5_LOCAL if PYTHON_V5_LOCAL.is_file() else PYTHON
PYTHON_V7 = APP_DIR / "engines" / "V7" / ".venv" / "Scripts" / "python.exe"
V2_ENGINE = APP_DIR / "engines" / "V2" / "upsize_ai_v2.py"
V3_ENGINE = APP_DIR / "engines" / "V3" / "upsize_ai_v3_master.py"
V4_ENGINE = APP_DIR / "engines" / "V4" / "upsize_vector_v4.py"
V4_DEEP_ENGINE = APP_DIR / "engines" / "V4" / "deep_raster_v4.py"
V5_ENGINE = APP_DIR / "engines" / "V5" / "layer_engine_v5_pro.py"
V5_ENGINE_DIR = V5_ENGINE.parent
V7_ENGINE = APP_DIR / "engines" / "V7" / "design_repair_v7.py"
V7_ENGINE_DIR = V7_ENGINE.parent
V7_MODELS = APP_DIR / "engines" / "V7" / "models"
V7_TESSDATA = V7_MODELS / "tessdata"
V4_DEEP_MODEL = APP_DIR / "engines" / "V3" / "models" / "Real_HAT_GAN_sharper.pth"
V4_DEEP_CONFIG_VERSION = 1
V3_MODEL_FILES = (
    APP_DIR / "engines" / "V3" / "models" / "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth",
    APP_DIR / "engines" / "V3" / "models" / "Real_HAT_GAN_sharper.pth",
    APP_DIR / "engines" / "V3" / "models" / "RealESRGAN_x4plus.pth",
)
V3_ENGINE_FILES = (
    V3_ENGINE,
    APP_DIR / "engines" / "V3" / "upsize_ai_v3.py",
    APP_DIR / "engines" / "V3" / "src" / "upscale_engine.py",
    APP_DIR / "engines" / "V3" / "src" / "pyramid_fusion.py",
    APP_DIR / "engines" / "V3" / "src" / "blend_outputs.py",
)
VERSION_FILE = ROOT_DIR / "VERSION"
V5_REVIEW_SCHEMA = "V5_LAYER_REVIEW_V2"
V5_REVIEW_RESUME_SIGNATURE = "V5_PRO_INVENTORY_2026_08_09_K"
MAX_REVIEW_ROUTING_BYTES = 64 * 1024 * 1024
V5_FAILURE_DIAGNOSTIC_MAX_FILE_BYTES = 64 * 1024 * 1024
V5_FAILURE_DIAGNOSTIC_MAX_TOTAL_BYTES = 128 * 1024 * 1024

# V7 remains executable as a standalone script and therefore its internal
# modules use top-level imports (``v7lib``).  Put only that engine directory on
# the launcher import path so the shared resolver can serve old and new bundles.
if str(V7_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(V7_ENGINE_DIR))

from bundle_layout import (  # noqa: E402
    TECHNICAL_DIR_NAME,
    BundleLayoutError,
    arrange_bundle,
    resolve_bundle_manifest,
    resolve_bundle_paths,
)


def read_app_version() -> str:
    try:
        value = VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "dev"
    return value or "dev"


APP_VERSION = read_app_version()


class UserError(RuntimeError):
    """Expected command/input error with a short user-facing message."""


@lru_cache(maxsize=4)
def python_runtime_has_cuda(python_path: str) -> bool:
    try:
        completed = subprocess.run(
            [
                python_path,
                "-c",
                "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
            ],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def print_help() -> None:
    print(f"Local Print Image Upscaler v{APP_VERSION}\n")
    print(
        f"""
UPSCALE ẢNH GPU

1. Bỏ ảnh vào:
   {INPUT_DIR}

2. Dùng V2 nhanh:
   .\\upscale <ten-anh> <n>

3. Dùng V3 chất lượng cao:
   .\\upscale high <ten-anh> <n>

4. Dùng V4 Print: guarded-USM/Deep ablation có gate native-x4 + PDF/X-4:
   .\\upscale print <ten-anh> <n> [--width-mm <khổ-rộng-mm>]

5. Dùng V4 toàn vector cho logo/đồ họa phẳng (có thể posterize ảnh chụp):
   .\\upscale vector <ten-anh> <n> [--width-mm <khổ-rộng-mm>]

6. Dùng V5 Pro tách ảnh phẳng thành layer có kiểm kê và duyệt trực quan:
   .\\upscale layers <file-hoặc-thư-mục> <n> [--detail exhaustive|grouped]
                   [--review gui|defer|auto|strict]
                   [--inpaint auto|poster|lama] [--no-semantic] [--allow-huge]

7. Dùng V7 làm rõ toàn ảnh và phục dựng chữ hỏng/mờ/sai, có bước duyệt nội dung:
   .\\upscale repair <file-hoặc-thư-mục> <n>
                   [--review gui|defer|auto|strict]
                   [--review-file <TEXT_REVIEW.json>]
                   [--ocr-passes 1..3] [--no-language-model]
                   [--inpaint auto|poster|opencv|strict] [--allow-huge]

8. Mở giao diện duyệt V5 hoặc V7 trong Chrome, không cần sửa JSON bằng tay:
   .\\upscale review <bundle-hoặc-file-duyệt.json>
   .\\upscale duyet  <bundle-hoặc-file-duyệt.json>

Đường dẫn là thư mục thì chương trình chạy tất cả ảnh trong thư mục và
các thư mục con với cùng một hệ số n.

Ví dụ:
   .\\upscale poster.png 4
   .\\upscale "bang quang cao.jpg" 10
   .\\upscale high "bang quang cao.jpg" 10
   .\\upscale "D:\\BO ANH" 10
   .\\upscale high "D:\\BO ANH" 10
   .\\upscale print poster.png 10 --width-mm 3000
   .\\upscale print "D:\\BO ANH" 4 --width-mm 4000
   .\\upscale layers poster.png 1
   .\\upscale layers "D:\\BO ANH" 4
   .\\upscale repair poster.png 1
   .\\upscale repair poster.png 4 --review defer
   .\\upscale repair poster.png 4 --review-file "D:\\TEXT_REVIEW.json"
   .\\upscale repair poster.png 4 --review strict --review-file "D:\\TEXT_REVIEW.json"
   .\\upscale review "D:\\OUTPUT\\V5_LAYERS\\poster_V5_LAYERS_x1"
   .\\upscale review "D:\\OUTPUT\\V7_REPAIR\\poster_V7_REPAIR_x4"

Kết quả V2: {OUTPUT_DIR / 'V2_FAST'}
Kết quả V3: {OUTPUT_DIR / 'V3_HIGH'}
Kết quả V4 Print : {OUTPUT_DIR / 'V4_PRINT'}
Kết quả V4 Vector: {OUTPUT_DIR / 'V4_VECTOR'}
Kết quả V5 Layer : {OUTPUT_DIR / 'V5_LAYERS'}
Kết quả V7 Repair: {OUTPUT_DIR / 'V7_REPAIR'}

Mỗi bundle V4 có 3 file thật:
   *_EDITABLE.svg       raster được QA + path vector opacity 0 để chỉnh sửa
   *_PRINT_PDFX4.pdf    PDF/X-4 có ICC, lấy hình in từ raster được QA
   *_PREVIEW_xN.png     bản PNG cùng raster để xem nhanh

Mỗi bundle V5 luôn có OpenRaster (.ora), ZIP layer PNG/mask, bản xem và manifest.
PSD chỉ có khi canvas không quá 30.000 px/cạnh và dự toán layer thô không quá 1,6 GB.
PSD/ORA là layer raster RGB thật; OCR chỉ đặt tên hỗ trợ, không giả làm font chỉnh sửa.
Phần nền vốn bị vật thể che được tái tạo hợp lý và được ghi rõ là nền suy đoán.
V5 là tài liệu trung gian để sửa, không phải PDF/X hoặc CMYK giao in và không cam kết giữ DPI/khổ vật lý.

V7 phục dựng raster trên toàn ảnh bằng Swin2SR fidelity chạy cục bộ trên GPU, không dùng
GAN/fusion ba model. Với n=1, model vẫn suy luận native x4 rồi downsample có kiểm soát về
kích thước gốc; với n>=2, ảnh được xuất lớn theo n. Nếu model lỗi hoặc QA từ chối, báo cáo
trong _KY_THUAT ghi rõ fallback thay vì âm thầm gọi ảnh mờ là đã phục dựng. Pixel AI là dự
đoán hợp lý, không phải bằng chứng cho chi tiết vốn đã mất.
Chữ được duyệt sẽ bị gỡ khỏi bitmap nguồn, nền trong đúng vùng đó được dựng lại, rồi chữ
Unicode sạch được vẽ trực tiếp ở kích thước cuối.
Giá, SĐT, mã hàng, địa chỉ và mọi thay đổi chính tả luôn cần duyệt. Nếu còn vùng
chưa duyệt hoặc QA thất bại, bundle ghi rõ REVIEW_REQUIRED/FAILED_QA và không tự
nhận là file giao in. Khi chạy cả thư mục, V7 dùng --review defer để không mở hàng loạt cửa sổ.
Mở từng bundle bằng .\\upscale review, bấm lưu trên giao diện rồi chạy lại cùng lệnh thư mục;
V7 sẽ tự nạp đúng review cũ, không cần mở hay sửa JSON bằng tay.
--review strict bắt buộc quyết định rõ cho mọi vùng; review được khóa bằng SHA-256 và fingerprint.

Nếu khổ thành phẩm cộng bleed vượt 5.000 mm, PDF V4 tự chọn tỷ lệ 1:d nhỏ nhất và ghi rõ
trong báo cáo kỹ thuật. Hãy thay ICC mặc định bằng profile của nhà in khi họ cung cấp.

n là hệ số chiều rộng và chiều cao, từ 2 đến 20; riêng V5/V7 nhận cả n=1. V7 n=1 giữ nguyên
kích thước file nhưng vẫn dùng GPU để phục dựng raster; n>=2 vừa phục dựng vừa làm lớn.
V2/V3/V4 thường dùng n=4 hoặc n=10; V5 nên dùng n=1 để sửa nhẹ hoặc n=4 khi cần canvas layer lớn.
Cùng một lệnh chạy lại sẽ thay kết quả cũ
một cách an toàn sau khi file mới đã render và kiểm tra xong.
""".strip()
    )


def parse_command(argv: list[str]) -> tuple[str, str, float, bool, dict[str, object]]:
    if not argv or any(value.lower() in {"help", "-h", "--help", "/?"} for value in argv):
        print_help()
        raise SystemExit(0)

    allow_huge = False
    positional: list[str] = []
    v4_options: dict[str, object] = {
        "width_mm": None,
        "bleed_mm": 0.0,
        "profile_name": "ISO Coated v2 300% (basICColor)",
        "max_layers": 24,
        "detail": "exhaustive",
        "inpaint": "auto",
        "semantic": True,
        "review": "gui",
        "review_file": None,
        "language_model": True,
        "ocr_passes": 3,
    }
    v4_specific_option = False
    v5_specific_option = False
    v5_v7_specific_option = False
    v7_specific_option = False
    v5_v7_review_option = False
    index = 0
    while index < len(argv):
        value = argv[index]
        lowered = value.lower()
        if lowered == "--allow-huge":
            allow_huge = True
        elif lowered == "--no-semantic":
            v4_options["semantic"] = False
            v5_specific_option = True
        elif lowered == "--no-language-model":
            v4_options["language_model"] = False
            v7_specific_option = True
        elif lowered in {"--review", "--review-file", "--ocr-passes"}:
            if index + 1 >= len(argv):
                raise UserError(f"Thiếu giá trị sau {value}.")
            raw = argv[index + 1]
            if lowered == "--review":
                if raw.lower() not in {"gui", "defer", "auto", "strict"}:
                    raise UserError("--review chỉ nhận gui, defer, auto hoặc strict.")
                v4_options["review"] = raw.lower()
                v5_v7_review_option = True
            elif lowered == "--review-file":
                if not raw.strip():
                    raise UserError("--review-file không được để trống.")
                v4_options["review_file"] = raw
            else:
                try:
                    number = int(raw)
                except ValueError as exc:
                    raise UserError(f"Giá trị không hợp lệ cho {value}: {raw}") from exc
                if not 1 <= number <= 3:
                    raise UserError("--ocr-passes phải từ 1 đến 3.")
                v4_options["ocr_passes"] = number
            if lowered != "--review":
                v7_specific_option = True
            index += 1
        elif lowered == "--detail":
            if index + 1 >= len(argv):
                raise UserError(f"Thiếu giá trị sau {value}.")
            raw = argv[index + 1].lower()
            if raw not in {"exhaustive", "grouped"}:
                raise UserError("--detail chỉ nhận exhaustive hoặc grouped.")
            v4_options["detail"] = raw
            v5_specific_option = True
            index += 1
        elif lowered == "--max-layers":
            if index + 1 >= len(argv):
                raise UserError(f"Thiếu giá trị sau {value}.")
            raw = argv[index + 1]
            try:
                number = int(raw)
            except ValueError as exc:
                raise UserError(f"Giá trị không hợp lệ cho {value}: {raw}") from exc
            if not 4 <= number <= 60:
                raise UserError("--max-layers phải từ 4 đến 60.")
            v4_options["max_layers"] = number
            v5_specific_option = True
            index += 1
        elif lowered == "--inpaint":
            if index + 1 >= len(argv):
                raise UserError(f"Thiếu giá trị sau {value}.")
            raw = argv[index + 1].lower()
            if raw not in {"auto", "poster", "lama", "opencv", "strict"}:
                raise UserError("--inpaint chỉ nhận auto, poster, lama, opencv hoặc strict.")
            v4_options["inpaint"] = raw
            v5_v7_specific_option = True
            index += 1
        elif lowered in {"--width-mm", "--bleed-mm", "--profile-name"}:
            if index + 1 >= len(argv):
                raise UserError(f"Thiếu giá trị sau {value}.")
            raw = argv[index + 1]
            if lowered == "--profile-name":
                if not raw.strip():
                    raise UserError("Tên ICC profile không được để trống.")
                v4_options["profile_name"] = raw.strip()
            else:
                try:
                    number = float(raw.replace(",", "."))
                except ValueError as exc:
                    raise UserError(f"Giá trị không hợp lệ cho {value}: {raw}") from exc
                v4_options["width_mm" if lowered == "--width-mm" else "bleed_mm"] = number
            v4_specific_option = True
            index += 1
        elif value.startswith("--"):
            raise UserError(f"Tùy chọn không hỗ trợ: {value}")
        else:
            positional.append(value)
        index += 1

    mode = "V2_FAST"
    if positional and positional[0].lower() in {"high", "v3"}:
        mode = "V3_HIGH"
        positional.pop(0)
    elif positional and positional[0].lower() in {"print", "v4"}:
        mode = "V4_PRINT"
        positional.pop(0)
    elif positional and positional[0].lower() == "vector":
        mode = "V4_VECTOR"
        positional.pop(0)
    elif positional and positional[0].lower() in {"layers", "layer", "v5"}:
        mode = "V5_LAYERS"
        positional.pop(0)
    elif positional and positional[0].lower() in {"repair", "v7"}:
        mode = "V7_REPAIR"
        positional.pop(0)
    elif positional and positional[0].lower() in {"fast", "v2"}:
        positional.pop(0)

    if mode not in {"V4_PRINT", "V4_VECTOR"} and v4_specific_option:
        raise UserError("--width-mm, --bleed-mm và --profile-name chỉ dùng với chế độ print/V4.")
    if mode != "V5_LAYERS" and v5_specific_option:
        raise UserError("--detail, --max-layers và --no-semantic chỉ dùng với chế độ layers/V5.")
    if mode not in {"V5_LAYERS", "V7_REPAIR"} and v5_v7_specific_option:
        raise UserError("--inpaint chỉ dùng với chế độ layers/V5 hoặc repair/V7.")
    if mode != "V7_REPAIR" and v7_specific_option:
        raise UserError(
            "--review, --review-file, --ocr-passes và --no-language-model chỉ dùng với repair/V7."
        )
    if mode not in {"V5_LAYERS", "V7_REPAIR"} and v5_v7_review_option:
        raise UserError("--review chỉ dùng với layers/V5 hoặc repair/V7.")
    if mode == "V5_LAYERS" and v4_options["inpaint"] not in {"auto", "poster", "lama"}:
        raise UserError("V5 --inpaint chỉ nhận auto, poster hoặc lama.")
    if mode == "V7_REPAIR" and v4_options["inpaint"] not in {
        "auto",
        "poster",
        "opencv",
        "strict",
    }:
        raise UserError("V7 --inpaint chỉ nhận auto, poster, opencv hoặc strict.")

    if len(positional) != 2:
        raise UserError("Sai cú pháp. Gõ .\\upscale để xem ví dụ.")
    file_token, scale_token = positional
    try:
        scale = float(scale_token.replace(",", "."))
    except ValueError as exc:
        raise UserError(f"Hệ số không hợp lệ: {scale_token}") from exc
    if mode == "V5_LAYERS" and not scale.is_integer():
        raise UserError(
            "V5 layers requires an integer scale x1..x20 for stable editable masks; "
            "separate at x1, then upscale the edited composite for fractional sizing."
        )
    minimum_scale = 1.0 if mode in {"V5_LAYERS", "V7_REPAIR"} else MIN_SCALE
    if mode == "V7_REPAIR" and 1.0 < scale < 2.0:
        raise UserError("V7 chỉ nhận đúng x1 hoặc từ x2 đến x20; không nhận hệ số nằm giữa.")
    if not (minimum_scale <= scale <= MAX_SCALE):
        raise UserError(
            f"Hệ số phải từ x{minimum_scale:g} đến x{MAX_SCALE:g}. "
            "x100 tạo lượng pixel quá lớn và không làm ảnh có thêm chi tiết thật."
        )
    width_mm = v4_options["width_mm"]
    bleed_mm = v4_options["bleed_mm"]
    if width_mm is not None and not (10.0 <= float(width_mm) <= 100_000.0):
        raise UserError("--width-mm phải từ 10 đến 100.000 mm.")
    if not (0.0 <= float(bleed_mm) <= 100.0):
        raise UserError("--bleed-mm phải từ 0 đến 100 mm.")
    return mode, file_token, scale, allow_huge, v4_options


def resolve_source(token: str) -> Path:
    supplied = Path(token).expanduser()
    candidates: list[Path] = []
    if supplied.is_absolute():
        candidates.append(supplied)
    else:
        candidates.extend((INPUT_DIR / supplied, ROOT_DIR / supplied))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise UserError(
                    "Chỉ nhận PNG, JPG/JPEG, WebP, BMP hoặc TIFF một khung hình."
                )
            return resolved

    if not supplied.suffix and not supplied.is_absolute():
        matches = [
            (INPUT_DIR / f"{token}{extension}").resolve()
            for extension in SUPPORTED_EXTENSIONS
            if (INPUT_DIR / f"{token}{extension}").is_file()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(path.name for path in matches)
            raise UserError(f"Có nhiều ảnh cùng tên ({names}); hãy gõ đầy đủ phần mở rộng.")

    raise UserError(f"Không tìm thấy ảnh trong INPUT: {token}")


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def collect_batch_images(directory: Path) -> list[Path]:
    resolved = directory.resolve()
    if resolved == ROOT_DIR or is_within(resolved, APP_DIR) or is_within(resolved, OUTPUT_DIR):
        raise UserError("Không được dùng RESIZE, APP hoặc OUTPUT làm thư mục nguồn batch.")

    images: list[Path] = []
    for candidate in resolved.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        candidate_resolved = candidate.resolve()
        if is_within(candidate_resolved, APP_DIR) or is_within(candidate_resolved, OUTPUT_DIR):
            continue
        images.append(candidate_resolved)
    images.sort(key=lambda path: str(path.relative_to(resolved)).casefold())
    if not images:
        raise UserError(f"Không có ảnh được hỗ trợ trong thư mục: {resolved}")
    return images


def safe_folder_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value).strip(" .")
    return cleaned or "BATCH"


def canonical_path_identity(path: Path) -> str:
    """Return a stable Windows path identity without reading file contents."""

    canonical = os.path.normcase(str(Path(path).resolve())).replace("\\", "/")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _same_canonical_path(raw: object, expected: Path) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return False
    try:
        return canonical_path_identity(Path(raw)) == canonical_path_identity(expected)
    except (OSError, RuntimeError, ValueError):
        return False


def _read_bundle_manifest(bundle: Path) -> dict[str, object] | None:
    reference = Path(bundle)
    try:
        if reference.is_file():
            manifest_path = reference.resolve(strict=True)
        else:
            if reference.name.casefold() == TECHNICAL_DIR_NAME.casefold():
                reference = reference.parent
            manifest_path = resolve_bundle_manifest(reference)
        if manifest_path is None:
            return None
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (BundleLayoutError, OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def find_chrome_executable() -> Path | None:
    """Find Chrome without invoking a shell or trusting an arbitrary command."""

    candidates: list[Path] = []
    for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if base:
            candidates.append(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe")
    for command_name in ("chrome.exe", "chrome"):
        located = shutil.which(command_name)
        if located:
            candidates.append(Path(located))
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    return None


def open_review_browser(url: str) -> bool:
    """Prefer Google Chrome, then use the registered system browser."""

    chrome = find_chrome_executable()
    if chrome is not None:
        try:
            subprocess.Popen([str(chrome), "--new-window", url])
            return True
        except OSError:
            pass
    opened = bool(webbrowser.open(url, new=1))
    if not opened:
        print(f"Không tự mở được trình duyệt. Hãy mở liên kết cục bộ này: {url}")
    return opened


def _read_review_schema(path: Path) -> str | None:
    """Read only enough JSON to route a review file; model code is never loaded."""

    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError:
        return None
    if (
        not resolved.is_file()
        or resolved.suffix.casefold() != ".json"
        or stat.st_size > MAX_REVIEW_ROUTING_BYTES
    ):
        return None
    try:
        document = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    schema = document.get("schema")
    return schema if isinstance(schema, str) else None


def _v5_bundle_root(checkpoint: Path) -> Path:
    parent = checkpoint.parent
    return parent.parent if parent.name.casefold() == TECHNICAL_DIR_NAME.casefold() else parent


def _read_v5_review_manifest(checkpoint: Path) -> dict[str, object] | None:
    """Read optional V5 metadata and reject a contradictory bundle manifest."""

    bundle = _v5_bundle_root(checkpoint)
    manifest_candidates = (
        bundle / "manifest.json",
        bundle / TECHNICAL_DIR_NAME / "PORTABLE_MANIFEST.json",
    )
    metadata: dict[str, object] | None = None
    for manifest_path in manifest_candidates:
        if not manifest_path.is_file():
            continue
        metadata = _read_bundle_manifest(manifest_path)
        if metadata is None:
            raise UserError(f"Manifest V5 bị hỏng hoặc không đọc được: {manifest_path}")
        if metadata.get("pipeline") != "V5_SMART_EDITABLE_LAYERS":
            raise UserError(
                "Checkpoint mang schema V5 nhưng manifest cùng bundle không phải V5 SMART EDITABLE LAYERS."
            )
        declared = metadata.get("review_checkpoint")
        if isinstance(declared, str) and declared.strip():
            relative = Path(declared.replace("/", os.sep))
            if relative.is_absolute() or ".." in relative.parts:
                raise UserError("Manifest V5 chứa đường dẫn checkpoint không an toàn.")
            try:
                declared_path = (bundle / relative).resolve(strict=True)
            except OSError as exc:
                raise UserError("Manifest V5 chỉ tới checkpoint không tồn tại.") from exc
            if declared_path != checkpoint:
                raise UserError(
                    "File đã chọn không phải LAYER_REVIEW.json được manifest V5 chỉ định."
                )
        break
    return metadata


def resolve_v5_review_target(
    raw_target: str | os.PathLike[str],
) -> tuple[Path, dict[str, object] | None] | None:
    """Resolve a V5 checkpoint from an output bundle, _KY_THUAT, or JSON path."""

    supplied = Path(raw_target).expanduser()
    if supplied.is_symlink():
        raise UserError(f"Không mở review qua symbolic link: {supplied}")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError:
        return None

    if resolved.is_file():
        candidates = [resolved]
    elif resolved.is_dir():
        candidates = []
        if resolved.name.casefold() == TECHNICAL_DIR_NAME.casefold():
            candidates.append(resolved / "LAYER_REVIEW.json")
        candidates.extend(
            [
                resolved / TECHNICAL_DIR_NAME / "LAYER_REVIEW.json",
                resolved / "LAYER_REVIEW.json",
            ]
        )
    else:
        return None

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        identity = os.path.normcase(str(candidate))
        if identity in seen:
            continue
        seen.add(identity)
        if not candidate.is_file():
            continue
        if candidate.is_symlink() or candidate.parent.is_symlink():
            raise UserError(f"Không mở checkpoint V5 qua symbolic link: {candidate}")
        unique_candidates.append(candidate.resolve(strict=True))

    matches = [
        candidate
        for candidate in unique_candidates
        if _read_review_schema(candidate) == V5_REVIEW_SCHEMA
    ]
    if len(matches) > 1:
        raise UserError(
            "Đường dẫn chứa nhiều checkpoint V5; hãy chỉ rõ đúng file _KY_THUAT\\LAYER_REVIEW.json."
        )
    if not matches:
        # A bundle that explicitly identifies itself as V5 must not silently
        # fall through to V7 when its checkpoint is damaged or missing.
        if resolved.is_file():
            bundle = _v5_bundle_root(resolved)
        else:
            bundle = (
                resolved.parent
                if resolved.name.casefold() == TECHNICAL_DIR_NAME.casefold()
                else resolved
            )
        if bundle.is_dir():
            manifest_path = bundle / "manifest.json"
            metadata = _read_bundle_manifest(manifest_path) if manifest_path.is_file() else None
            if metadata is not None and metadata.get("pipeline") == "V5_SMART_EDITABLE_LAYERS":
                raise UserError(
                    "Bundle V5 thiếu LAYER_REVIEW.json hợp lệ (schema V5_LAYER_REVIEW_V2)."
                )
        return None

    checkpoint = matches[0]
    metadata = _read_v5_review_manifest(checkpoint)
    return checkpoint, metadata


def _run_v5_review_ui(review_path: Path) -> dict[str, object]:
    if str(V5_ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(V5_ENGINE_DIR))
    try:
        from v5pro.review_server import ReviewUIError, run_review_ui
    except ImportError as exc:
        raise UserError(
            "Thiếu thành phần giao diện duyệt V5; hãy kiểm tra APP\\engines\\V5\\v5pro."
        ) from exc
    try:
        return run_review_ui(review_path, open_browser=open_review_browser)
    except ReviewUIError as exc:
        raise UserError(f"Không mở được giao diện duyệt layer V5: {exc}") from exc


def _v5_rerun_command(metadata: dict[str, object] | None) -> str | None:
    if metadata is None:
        return None
    source = metadata.get("original_source")
    raw_scale = metadata.get("scale")
    if not isinstance(source, str) or not source.strip() or "\x00" in source:
        return None
    if isinstance(raw_scale, bool):
        return None
    try:
        scale = float(raw_scale)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(scale) or not (scale == 1.0 or 2.0 <= scale <= MAX_SCALE):
        return None
    # This is printed for copy/paste into PowerShell. Refuse values that can
    # interpolate or break out of the quoted source argument.
    unsafe_characters = '&|<>^%!"$' + chr(96) + "\r\n"
    if any(character in source for character in unsafe_characters):
        return None

    arguments = [".\\upscale", "layers", f'"{source}"', f"{scale:g}"]
    launcher_options = metadata.get("launcher_options")
    options = launcher_options if isinstance(launcher_options, dict) else {}
    detail = str(options.get("detail", metadata.get("detail", "exhaustive")))
    if detail in {"exhaustive", "grouped"}:
        arguments.extend(["--detail", detail])
    arguments.extend(["--review", "gui"])
    inpaint = str(options.get("inpaint", "auto"))
    if inpaint in {"auto", "poster", "lama"}:
        arguments.extend(["--inpaint", inpaint])
    if options.get("semantic") is False:
        arguments.append("--no-semantic")
    if options.get("allow_huge") is True:
        arguments.append("--allow-huge")
    return " ".join(arguments)


def _run_resolved_v5_review(
    review_path: Path,
    metadata: dict[str, object] | None,
) -> int:
    print("\nDUYỆT LAYER V5")
    print(f"  File quyết định : {review_path}")
    print("  Trình duyệt     : ưu tiên Google Chrome")
    print("  Cách lưu        : mỗi thao tác được ghi ngay, không cần sửa JSON")
    _run_v5_review_ui(review_path)

    print("\nĐÃ LƯU CHECKPOINT V5 — PSD/ORA HIỆN TẠI CHƯA THAY ĐỔI")
    print("  Muốn tạo lại bộ layer theo lần duyệt, bắt buộc chạy lại lệnh layers.")
    rerun = _v5_rerun_command(metadata)
    if rerun is not None:
        print("  Lệnh chạy lại an toàn cho đúng ảnh và hệ số:")
        print(f"  {rerun}")
    else:
        print("  Không đủ metadata an toàn để tự in lệnh; hãy chạy lại lệnh layers ban đầu với --review gui.")
    print(
        "  Lưu ý trung thực: checkpoint đã lưu sẽ chỉ được áp dụng khi lệnh layers chạy lại "
        "và xác minh nó còn khớp ảnh/kiểm kê; không có bước nào tự sửa PSD/ORA cũ."
    )
    return 0


def run_v5_review_command(raw_target: str | os.PathLike[str]) -> int:
    resolved = resolve_v5_review_target(raw_target)
    if resolved is None:
        raise UserError("Đường dẫn không phải checkpoint/bundle V5_LAYER_REVIEW_V2.")
    return _run_resolved_v5_review(*resolved)


def run_review_command(raw_target: str | os.PathLike[str]) -> int:
    """Auto-route V5 layer checkpoints while retaining the V7 review flow."""

    supplied = Path(raw_target).expanduser()
    if not supplied.exists():
        raise UserError(f"Không tìm thấy bundle hoặc file duyệt V5/V7: {supplied}")
    resolved_v5 = resolve_v5_review_target(raw_target)
    if resolved_v5 is not None:
        return _run_resolved_v5_review(*resolved_v5)
    return run_v7_review_command(raw_target)


def _validated_review_bundle(
    bundle: Path,
) -> tuple[Path, Path, dict[str, object]]:
    try:
        paths = resolve_bundle_paths(bundle)
    except BundleLayoutError as exc:
        raise UserError(f"Bundle V7 không an toàn hoặc bị trùng file: {exc}") from exc
    if paths.manifest is None or paths.review is None or paths.source is None:
        raise UserError(
            "Bundle V7 thiếu manifest, TEXT_REVIEW.json hoặc ảnh SOURCE; không thể mở duyệt."
        )
    metadata = _read_bundle_manifest(paths.manifest)
    if metadata is None or metadata.get("pipeline") != "V7_DESIGN_REPAIR":
        raise UserError("Manifest không chứng minh đây là bundle V7 DESIGN REPAIR.")
    return paths.review, paths.source, metadata


def resolve_v7_review_target(
    raw_target: str | os.PathLike[str],
) -> tuple[Path, Path | None, dict[str, object] | None]:
    """Resolve a legacy/new bundle or an explicitly supplied review JSON."""

    supplied = Path(raw_target).expanduser()
    if supplied.is_symlink():
        raise UserError(f"Không mở review qua symbolic link: {supplied}")
    if supplied.is_dir():
        review, source, metadata = _validated_review_bundle(supplied)
        return review, source, metadata
    try:
        review_path = supplied.resolve(strict=True)
    except OSError as exc:
        raise UserError(f"Không tìm thấy bundle hoặc file duyệt V7: {supplied}") from exc
    if not review_path.is_file() or review_path.suffix.casefold() != ".json":
        raise UserError("Lệnh review/duyet cần một bundle V7 hoặc file JSON duyệt.")

    parent = review_path.parent
    probable_bundle = parent.parent if parent.name.casefold() == TECHNICAL_DIR_NAME.casefold() else parent
    has_bundle_manifest = any(
        candidate.is_file()
        for candidate in (
            probable_bundle / "manifest.json",
            probable_bundle / TECHNICAL_DIR_NAME / "manifest.json",
        )
    )
    if has_bundle_manifest:
        canonical_review, source, metadata = _validated_review_bundle(probable_bundle)
        if canonical_review != review_path:
            raise UserError(
                "File JSON đã chọn không phải TEXT_REVIEW.json được manifest của bundle chỉ định."
            )
        return canonical_review, source, metadata
    # A standalone review file is allowed.  user_review.py will still require
    # its source image to resolve inside this same directory.
    return review_path, None, None


def _run_v7_review_ui(review_path: Path, image_path: Path | None) -> dict[str, object]:
    try:
        from user_review import ReviewUIError, run_review_ui
    except ImportError as exc:
        raise UserError(
            "Thiếu thành phần giao diện duyệt V7; hãy kiểm tra APP\\engines\\V7."
        ) from exc
    try:
        return run_review_ui(
            review_path,
            image_path=image_path,
            open_browser=open_review_browser,
        )
    except ReviewUIError as exc:
        raise UserError(f"Không mở được giao diện duyệt V7: {exc}") from exc


def _v7_rerun_command(
    metadata: dict[str, object] | None,
    review_path: Path,
) -> str | None:
    if metadata is None:
        return None
    is_batch = isinstance(metadata.get("batch_root"), str)
    source = metadata.get("batch_root") if is_batch else metadata.get("original_source")
    raw_scale = metadata.get("scale")
    if not isinstance(source, str) or not source.strip() or "\x00" in source:
        return None
    if isinstance(raw_scale, bool):
        return None
    try:
        scale = float(raw_scale)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(scale) or not (scale == 1.0 or 2.0 <= scale <= MAX_SCALE):
        return None
    # This text is copied into cmd.exe by a human. Refuse to print path data
    # containing command metacharacters instead of turning manifest text into
    # a copy/paste injection vector.
    review_text = str(review_path)
    if any(character in source + review_text for character in '&|<>^%!"\r\n'):
        return None
    arguments = [".\\upscale", "repair", f'"{source}"', f"{scale:g}"]
    launcher_options = metadata.get("launcher_options")
    options = launcher_options if isinstance(launcher_options, dict) else {}
    if is_batch:
        arguments.extend(["--review", "defer"])
    else:
        review_mode = str(options.get("review_mode", "gui"))
        if review_mode not in {"gui", "defer", "auto", "strict"}:
            return None
        arguments.extend(["--review", review_mode, "--review-file", f'"{review_text}"'])
    raw_passes = options.get("ocr_passes", 3)
    if not isinstance(raw_passes, bool):
        try:
            ocr_passes = int(raw_passes)
        except (TypeError, ValueError, OverflowError):
            ocr_passes = 3
        if 1 <= ocr_passes <= 3:
            arguments.extend(["--ocr-passes", str(ocr_passes)])
    inpaint = str(options.get("inpaint", "auto"))
    if inpaint in {"auto", "poster", "opencv", "strict"}:
        arguments.extend(["--inpaint", inpaint])
    if options.get("language_model") is False:
        arguments.append("--no-language-model")
    if options.get("allow_huge") is True:
        arguments.append("--allow-huge")
    return " ".join(arguments)


def run_v7_review_command(raw_target: str | os.PathLike[str]) -> int:
    """Open the human review UI and print, but never execute, the rerun command."""

    review_path, image_path, metadata = resolve_v7_review_target(raw_target)
    print("\nDUYỆT CHỮ V7")
    print(f"  File quyết định : {review_path}")
    if image_path is not None:
        print(f"  Ảnh đối chiếu   : {image_path}")
    print("  Trình duyệt     : ưu tiên Google Chrome")
    _run_v7_review_ui(review_path, image_path)

    print("\nĐÃ LƯU PHẦN DUYỆT — CHƯA TỰ CHẠY LẠI ENGINE")
    rerun = _v7_rerun_command(metadata, review_path)
    if rerun is not None:
        print("  Chạy đúng lệnh sau để dựng lại ảnh với nội dung vừa duyệt:")
        print(f"  {rerun}")
    else:
        print("  Bundle cũ chưa có original_source/scale; hãy chạy lại lệnh repair ban đầu")
        print(f'  và thêm: --review-file "{review_path}"')
    return 0


def _v7_bundle_owned_by(
    bundle: Path,
    source: Path,
    *,
    batch_root: Path | None,
) -> bool:
    manifest = _read_bundle_manifest(bundle)
    if manifest is None or not _same_canonical_path(manifest.get("original_source"), source):
        return False
    if batch_root is not None and not _same_canonical_path(manifest.get("batch_root"), batch_root):
        return False
    return True


def select_v7_target_directory(
    relative_dir: Path,
    result_stem: str,
    tag: str,
    source: Path,
    *,
    batch_root: Path | None,
) -> Path:
    """Reuse a legacy readable name only when its manifest proves ownership."""

    parent = OUTPUT_DIR / "V7_REPAIR" / relative_dir
    legacy = parent / f"{result_stem}_V7_REPAIR_x{tag}"
    if not legacy.exists() or _v7_bundle_owned_by(legacy, source, batch_root=batch_root):
        return legacy
    suffix = canonical_path_identity(source)[:10]
    collision_safe = parent / f"{bounded_output_stem(f'{result_stem}_{suffix}')}_V7_REPAIR_x{tag}"
    if collision_safe.exists() and not _v7_bundle_owned_by(
        collision_safe,
        source,
        batch_root=batch_root,
    ):
        raise UserError(
            f"Output V7 đã tồn tại nhưng thuộc nguồn khác hoặc thiếu manifest: {collision_safe}"
        )
    return collision_safe


def select_v7_batch_group(directory: Path, tag: str) -> Path:
    """Keep equally named source folders in separate, deterministic output groups."""

    base_name = f"{safe_folder_name(directory.name)}_x{tag}"
    legacy = OUTPUT_DIR / "V7_REPAIR" / base_name
    if not legacy.exists():
        return Path(base_name)
    manifests = [
        value
        for path in legacy.rglob("manifest.json")
        if (value := _read_bundle_manifest(path)) is not None
    ]
    if manifests and all(_same_canonical_path(value.get("batch_root"), directory) for value in manifests):
        return Path(base_name)
    safe_name = f"{base_name}_{canonical_path_identity(directory)[:10]}"
    safe_group = OUTPUT_DIR / "V7_REPAIR" / safe_name
    if safe_group.exists():
        safe_manifests = [
            value
            for path in safe_group.rglob("manifest.json")
            if (value := _read_bundle_manifest(path)) is not None
        ]
        if not safe_manifests or not all(
            _same_canonical_path(value.get("batch_root"), directory) for value in safe_manifests
        ):
            raise UserError(
                f"Nhóm output V7 đã tồn tại nhưng không chứng minh đúng thư mục nguồn: {safe_group}"
            )
    return Path(safe_name)


def bounded_output_stem(value: str, maximum: int = 44) -> str:
    """Keep nested atomic bundle paths below legacy Windows MAX_PATH limits."""

    cleaned = safe_folder_name(value)
    if len(cleaned) <= maximum:
        return cleaned
    suffix = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[: maximum - 9]}_{suffix}"


def batch_entries(directory: Path, images: list[Path]) -> list[tuple[Path, Path, str]]:
    relatives = [image.relative_to(directory) for image in images]
    counts = Counter((str(path.parent).casefold(), path.stem.casefold()) for path in relatives)
    used: set[tuple[str, str]] = set()
    entries: list[tuple[Path, Path, str]] = []
    for image, relative in zip(images, relatives, strict=True):
        stem = relative.stem
        key = (str(relative.parent).casefold(), stem.casefold())
        if counts[key] > 1:
            stem = f"{stem}_{relative.suffix.lstrip('.').lower()}"
        unique_key = (str(relative.parent).casefold(), stem.casefold())
        if unique_key in used:
            suffix = hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:8]
            stem = f"{stem}_{suffix}"
            unique_key = (str(relative.parent).casefold(), stem.casefold())
        used.add(unique_key)
        entries.append((image, relative.parent, stem))
    return entries


def inspect_image(path: Path) -> tuple[tuple[int, int], str, bytes | None]:
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise UserError("Ảnh nhiều khung hình/ảnh động không được hỗ trợ.")
            image.seek(0)
            icc_profile = image.info.get("icc_profile")
            source_mode = image.mode
            oriented = ImageOps.exif_transpose(image)
            oriented.load()
            return oriented.size, source_mode, icc_profile
    except (UnidentifiedImageError, OSError) as exc:
        raise UserError(f"Không mở được ảnh: {path.name}") from exc


def srgb_profile_bytes() -> bytes:
    candidates = (
        APP_DIR / "shared" / "tools" / "scribus" / "1.6.6" / "share" / "profiles" / "sRGB.icm",
        Path(r"C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_bytes()
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def canonical_source_dpi(
    raw_dpi: object,
    *,
    swap_axes: bool = False,
) -> tuple[float, float] | None:
    """Return safe isotropic DPI for the staged, orientation-corrected image.

    V4 derives natural print width from horizontal DPI and preserves pixel aspect
    ratio. Valid anisotropic metadata is therefore deliberately normalised to the
    oriented horizontal DPI on both axes; malformed or implausible metadata is
    omitted so the V4 engine uses its documented 96 DPI fallback.
    """

    try:
        if isinstance(raw_dpi, (tuple, list)):
            if not raw_dpi:
                return None
            oriented_horizontal = (
                raw_dpi[1] if swap_axes and len(raw_dpi) > 1 else raw_dpi[0]
            )
        else:
            oriented_horizontal = raw_dpi
        horizontal = float(oriented_horizontal)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(horizontal):
        return None
    if not MIN_SOURCE_DPI <= horizontal <= MAX_SOURCE_DPI:
        return None
    return horizontal, horizontal


def stage_input(source: Path, job_dir: Path, _icc_profile: bytes | None = None) -> Path:
    """Create the one colour-safe input consumed by every engine."""

    staged = job_dir / "input.png"
    with Image.open(source) as image:
        image.seek(0)
        embedded_icc = image.info.get("icc_profile")
        source_dpi = image.info.get("dpi")
        try:
            orientation = int(image.getexif().get(274, 1))
        except (AttributeError, TypeError, ValueError):
            orientation = 1
        staged_dpi = canonical_source_dpi(
            source_dpi,
            swap_axes=orientation in {5, 6, 7, 8},
        )
        image = ImageOps.exif_transpose(image)
        image.load()

        has_alpha = image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info
        alpha = image.convert("RGBA").getchannel("A") if has_alpha else None
        colour = image.convert("RGB") if has_alpha else image

        if embedded_icc:
            try:
                source_profile = ImageCms.ImageCmsProfile(BytesIO(embedded_icc))
                destination_profile = ImageCms.createProfile("sRGB")
                rgb = ImageCms.profileToProfile(
                    colour,
                    source_profile,
                    destination_profile,
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    outputMode="RGB",
                )
            except (ImageCms.PyCMSError, OSError, ValueError):
                rgb = colour.convert("RGB")
        else:
            rgb = colour.convert("RGB")

        if alpha is not None:
            rgb = Image.composite(rgb, Image.new("RGB", rgb.size, "white"), alpha)
        save_options: dict[str, object] = {
            "format": "PNG",
            "compress_level": 3,
            "icc_profile": srgb_profile_bytes(),
        }
        if staged_dpi is not None:
            save_options["dpi"] = staged_dpi
        rgb.save(staged, **save_options)
    return staged


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_pixel_sha256(path: Path) -> str:
    """Hash canonical pixel content, not volatile PNG/ICC container metadata."""

    digest = hashlib.sha256()
    digest.update(b"RESIZE_CANONICAL_SRGB_WHITE_V1\0")
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        digest.update(f"{rgb.width}x{rgb.height}:RGB\0".encode("ascii"))
        rows_per_chunk = 256
        for top in range(0, rgb.height, rows_per_chunk):
            bottom = min(rgb.height, top + rows_per_chunk)
            digest.update(rgb.crop((0, top, rgb.width, bottom)).tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def current_v3_cache_signature() -> dict[str, object]:
    missing = [path for path in (*V3_ENGINE_FILES, *V3_MODEL_FILES) if not path.is_file()]
    if missing:
        raise UserError(f"Thiếu thành phần V3: {missing[0]}")
    signature: dict[str, object] = {
        "cache_config_version": V3_CACHE_CONFIG_VERSION,
        "engine_sha256": {path.name: sha256_file(path) for path in V3_ENGINE_FILES},
        "model_sha256": {path.name: sha256_file(path) for path in V3_MODEL_FILES},
        "run_config": {"scale": 4, "tile": 512, "overlap": 128, "force": True},
    }
    signature["signature_sha256"] = json_sha256(signature)
    return signature


def available_physical_memory() -> int | None:
    """Return currently available RAM on Windows, or None if unavailable."""

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    try:
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available_physical)
    except (AttributeError, OSError):
        pass
    return None


def validate_standard_resource_plan(
    source_size: tuple[int, int],
    final_size: tuple[int, int],
    *,
    allow_huge: bool,
) -> dict[str, float]:
    """Enforce a non-bypassable Pillow-safe cap for V2/V3 raster files."""

    source_pixels = source_size[0] * source_size[1]
    native_mp = source_pixels * 16 / 1_000_000
    output_mp = final_size[0] * final_size[1] / 1_000_000
    if native_mp > HARD_RASTER_MEGAPIXELS or output_mp > HARD_RASTER_MEGAPIXELS:
        raise UserError(
            "Kế hoạch raster vượt giới hạn an toàn cứng "
            f"({native_mp:.1f} MP master, {output_mp:.1f} MP output; "
            f"tối đa {HARD_RASTER_MEGAPIXELS:g} MP)."
        )
    if output_mp > MAX_MEGAPIXELS and not allow_huge:
        raise UserError(
            f"Kết quả sẽ là {final_size[0]}x{final_size[1]} ({output_mp:.1f} MP). "
            "Nếu đã tính đúng và máy in chấp nhận, thêm --allow-huge."
        )
    return {
        "native_megapixels": round(native_mp, 3),
        "output_megapixels": round(output_mp, 3),
    }


def validate_v4_resource_plan(
    source_size: tuple[int, int],
    proof_size: tuple[int, int],
    *,
    full_vector: bool,
    allow_huge: bool,
) -> dict[str, float]:
    source_pixels = source_size[0] * source_size[1]
    proof_pixels = proof_size[0] * proof_size[1]
    native_pixels = source_pixels * (4 if full_vector else 16)
    source_mp = source_pixels / 1_000_000
    native_mp = native_pixels / 1_000_000
    proof_mp = proof_pixels / 1_000_000

    peak_ram_bytes = max(native_pixels * 24, proof_pixels * 20) + 2 * 1024**3
    working_disk_bytes = native_pixels * 5 + proof_pixels * 12 + 1024**3
    estimates = {
        "source_megapixels": round(source_mp, 3),
        "native_megapixels": round(native_mp, 3),
        "output_megapixels": round(proof_mp, 3),
        "estimated_peak_ram_gib": round(peak_ram_bytes / 1024**3, 2),
        "estimated_working_disk_gib": round(working_disk_bytes / 1024**3, 2),
    }

    if (
        source_mp > V4_HARD_SOURCE_MEGAPIXELS
        or native_mp > V4_HARD_NATIVE_MEGAPIXELS
        or proof_mp > V4_HARD_OUTPUT_MEGAPIXELS
    ):
        raise UserError(
            "Kế hoạch V4 vượt giới hạn an toàn cứng "
            f"({source_mp:.1f} MP nguồn, {native_mp:.1f} MP master, {proof_mp:.1f} MP output)."
        )

    exceeds_soft_limit = source_mp > V4_SOFT_SOURCE_MEGAPIXELS or proof_mp > MAX_MEGAPIXELS
    if exceeds_soft_limit and not allow_huge:
        raise UserError(
            f"V4 ước tính cần {estimates['estimated_peak_ram_gib']:.2f} GiB RAM và "
            f"{estimates['estimated_working_disk_gib']:.2f} GiB ổ đĩa tạm. "
            "Kiểm tra kỹ rồi thêm --allow-huge nếu muốn chạy."
        )

    free_disk = shutil.disk_usage(APP_DIR).free
    if working_disk_bytes > free_disk * 0.8:
        raise UserError(
            f"Không đủ dung lượng tạm: ước tính {working_disk_bytes / 1024**3:.2f} GiB, "
            f"hiện trống {free_disk / 1024**3:.2f} GiB."
        )
    free_ram = available_physical_memory()
    if free_ram is not None and peak_ram_bytes > free_ram * 0.85:
        raise UserError(
            f"Không đủ RAM khả dụng: ước tính {peak_ram_bytes / 1024**3:.2f} GiB, "
            f"hiện khả dụng {free_ram / 1024**3:.2f} GiB."
        )
    return estimates


def validate_v5_resource_plan(
    source_size: tuple[int, int],
    final_size: tuple[int, int],
    *,
    allow_huge: bool,
    max_layers: int = 24,
) -> dict[str, float | str]:
    # ``max_layers`` is retained only for callers from older releases. V5 Pro
    # streams tight crops and never drops an element to satisfy a layer cap.
    _legacy_max_layers = max_layers
    source_pixels = source_size[0] * source_size[1]
    output_pixels = final_size[0] * final_size[1]
    source_mp = source_pixels / 1_000_000
    output_mp = output_pixels / 1_000_000
    if source_mp > V4_HARD_SOURCE_MEGAPIXELS or output_mp > V5_HARD_OUTPUT_MEGAPIXELS:
        raise UserError(
            "Kế hoạch V5 vượt giới hạn layer an toàn cứng "
            f"({source_mp:.1f} MP nguồn, {output_mp:.1f} MP canvas; "
            f"tối đa {V5_HARD_OUTPUT_MEGAPIXELS:g} MP đầu ra)."
        )
    # V5 Pro runs DINO, SAM2, BiRefNet and LayerD sequentially and exports one
    # tight crop at a time. Peak memory therefore depends mainly on canvas and
    # the largest model, not an artificial number of full-canvas layer masks.
    estimated_peak_ram = (
        source_pixels * 620
        + output_pixels * 46
        + 5 * 1024**3
    )
    estimated_disk = (
        source_pixels * 180
        + output_pixels * 70
        + 3 * 1024**3
    )
    minimum_disk_before_segmentation = output_pixels * 20 + 2 * 1024**3
    if output_mp > V5_SOFT_OUTPUT_MEGAPIXELS and not allow_huge:
        raise UserError(
            f"Bundle layer V5 sẽ có canvas {final_size[0]}x{final_size[1]} ({output_mp:.1f} MP) "
            "và có thể rất nặng. Nên tách/chỉnh ở x1 hoặc x4; nếu vẫn cần, thêm --allow-huge."
        )
    free_disk = shutil.disk_usage(APP_DIR).free
    # Do not reject a sparse poster from the impossible all-layers-full-canvas
    # disk upper bound. The engine measures actual crop support after
    # segmentation and hard-fails before export if the real ORA/ZIP plan will
    # not fit. This lower bound only protects the unavoidable canvas/work files.
    if minimum_disk_before_segmentation > free_disk * 0.8:
        raise UserError(
            "Không đủ dung lượng tối thiểu cho canvas V5 trước khi tách lớp: "
            f"cần {minimum_disk_before_segmentation / 1024**3:.2f} GiB, "
            f"hiện trống {free_disk / 1024**3:.2f} GiB."
        )
    free_ram = available_physical_memory()
    if free_ram is not None and estimated_peak_ram > free_ram * 0.85:
        raise UserError(
            f"Không đủ RAM khả dụng cho V5 Pro: ước tính {estimated_peak_ram / 1024**3:.2f} GiB, "
            f"hiện khả dụng {free_ram / 1024**3:.2f} GiB. Hãy giảm n."
        )
    return {
        "source_megapixels": round(source_mp, 3),
        "output_megapixels": round(output_mp, 3),
        "estimated_peak_ram_gib": round(estimated_peak_ram / 1024**3, 2),
        "estimated_working_disk_gib": round(estimated_disk / 1024**3, 2),
        "minimum_working_disk_gib": round(minimum_disk_before_segmentation / 1024**3, 2),
        "layer_policy": "exhaustive tight-crop streaming; no hard layer cap",
        "legacy_max_layers_ignored": float(_legacy_max_layers),
    }


def validate_v7_resource_plan(
    source_size: tuple[int, int],
    final_size: tuple[int, int],
    *,
    allow_huge: bool,
) -> dict[str, float]:
    """Bound OCR variants, clean master, masks and final print raster together."""

    source_pixels = source_size[0] * source_size[1]
    output_pixels = final_size[0] * final_size[1]
    source_mp = source_pixels / 1_000_000
    output_mp = output_pixels / 1_000_000
    if source_mp > V4_HARD_SOURCE_MEGAPIXELS or output_mp > V7_HARD_OUTPUT_MEGAPIXELS:
        raise UserError(
            "Kế hoạch V7 vượt giới hạn phục dựng an toàn cứng "
            f"({source_mp:.1f} MP nguồn, {output_mp:.1f} MP đầu ra; "
            f"tối đa {V7_HARD_OUTPUT_MEGAPIXELS:g} MP đầu ra)."
        )
    estimated_peak_ram = source_pixels * 150 + output_pixels * 24 + 4 * 1024**3
    estimated_disk = source_pixels * 30 + output_pixels * 14 + 1024**3
    if output_mp > V7_SOFT_OUTPUT_MEGAPIXELS and not allow_huge:
        raise UserError(
            f"V7 sẽ tạo ảnh {final_size[0]}x{final_size[1]} ({output_mp:.1f} MP). "
            "Hãy kiểm tra khổ in rồi thêm --allow-huge nếu thật sự cần."
        )
    free_disk = shutil.disk_usage(APP_DIR).free
    if estimated_disk > free_disk * 0.8:
        raise UserError(
            f"Không đủ dung lượng làm việc cho V7: ước tính {estimated_disk / 1024**3:.2f} GiB, "
            f"hiện trống {free_disk / 1024**3:.2f} GiB."
        )
    free_ram = available_physical_memory()
    if free_ram is not None and estimated_peak_ram > free_ram * 0.85:
        raise UserError(
            f"Không đủ RAM khả dụng cho V7: ước tính {estimated_peak_ram / 1024**3:.2f} GiB, "
            f"hiện khả dụng {free_ram / 1024**3:.2f} GiB."
        )
    return {
        "source_megapixels": round(source_mp, 3),
        "output_megapixels": round(output_mp, 3),
        "estimated_peak_ram_gib": round(estimated_peak_ram / 1024**3, 2),
        "estimated_working_disk_gib": round(estimated_disk / 1024**3, 2),
    }


def find_cached_v3_native(source_sha256: str, normalized_sha256: str) -> Path | None:
    signature = current_v3_cache_signature()
    cache_dir = APP_DIR / "masters" / "V4_AI_BASE"
    canonical_cache = cache_dir / f"{normalized_sha256}_NATIVE_x4.png"
    legacy_cache = cache_dir / f"{source_sha256}_NATIVE_x4.png"
    private_candidates = [canonical_cache]
    if legacy_cache != canonical_cache:
        private_candidates.append(legacy_cache)
    if cache_dir.is_dir():
        private_candidates.extend(
            sidecar.with_suffix("")
            for sidecar in cache_dir.glob("*_NATIVE_x4.png.json")
        )

    seen: set[str] = set()
    for private_cache in private_candidates:
        cache_key = str(private_cache.resolve()).casefold()
        if cache_key in seen:
            continue
        seen.add(cache_key)
        private_manifest = private_cache.with_suffix(private_cache.suffix + ".json")
        if not private_cache.is_file() or not private_manifest.is_file():
            continue
        try:
            metadata = json.loads(private_manifest.read_text(encoding="utf-8"))
            if (
                metadata.get("pipeline") == "V4_V3_NATIVE_CACHE"
                and (
                    metadata.get("canonical_pixel_sha256")
                    or metadata.get("normalized_stage_sha256")
                )
                == normalized_sha256
                and metadata.get("v3_cache_signature") == signature
                and metadata.get("native_sha256") == sha256_file(private_cache)
            ):
                return private_cache
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    manifest_root = APP_DIR / "manifests" / "V3_HIGH"
    if not manifest_root.is_dir():
        return None
    for candidate in manifest_root.rglob("*.json"):
        try:
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
            if metadata.get("normalized_stage_sha256") != normalized_sha256:
                continue
            if metadata.get("v3_cache_signature") != signature:
                continue
            native = Path(str(metadata["native_output"]))
            expected_sha = str(metadata.get("native_sha256", ""))
            if expected_sha and native.is_file() and sha256_file(native) == expected_sha:
                return native
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return None


def find_cached_v4_deep(
    canonical_pixel_sha256: str,
    v3_native_sha256: str,
    scale: float,
    *,
    legacy_source_sha256: str | None = None,
) -> tuple[Path, dict[str, object]] | None:
    cache_dir = APP_DIR / "masters" / "V4_DEEP"
    scale_token = scale_tag(scale)
    canonical_target = cache_dir / f"{canonical_pixel_sha256}_DEEP_x{scale_token}.png"
    candidates = [canonical_target]
    if legacy_source_sha256:
        legacy_target = cache_dir / f"{legacy_source_sha256}_DEEP_x{scale_token}.png"
        if legacy_target != canonical_target:
            candidates.append(legacy_target)
    if cache_dir.is_dir():
        candidates.extend(
            sidecar.with_suffix("")
            for sidecar in cache_dir.glob(f"*_DEEP_x{scale_token}.png.json")
        )
    if not V4_DEEP_ENGINE.is_file() or not V4_DEEP_MODEL.is_file():
        return None
    engine_sha256 = sha256_file(V4_DEEP_ENGINE)
    model_sha256 = sha256_file(V4_DEEP_MODEL)
    seen: set[str] = set()
    for target in candidates:
        target_key = str(target.resolve()).casefold()
        if target_key in seen:
            continue
        seen.add(target_key)
        manifest_path = target.with_suffix(target.suffix + ".json")
        if not target.is_file() or not manifest_path.is_file():
            continue
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            recorded_identity = metadata.get("canonical_pixel_sha256")
            if recorded_identity is not None and recorded_identity != canonical_pixel_sha256:
                continue
            if metadata.get("pipeline") != "V4_DEEP_RECURSIVE_HAT":
                continue
            if metadata.get("config_version") != V4_DEEP_CONFIG_VERSION:
                continue
            if metadata.get("engine_sha256") != engine_sha256:
                continue
            if str(metadata.get("model_sha256", "")).lower() != model_sha256:
                continue
            if metadata.get("v3_native_sha256") != v3_native_sha256:
                continue
            if not abs(float(metadata.get("final_scale")) - scale) < 1e-9:
                continue
            if metadata.get("output_sha256") != sha256_file(target):
                continue
            return target, metadata
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def prepare_v5_ai_master(
    source: Path,
    staged_input: Path,
    job_dir: Path,
    scale: float,
) -> tuple[Path, dict[str, object]]:
    """Reuse the audited V3 native x4 master, then resize once to V5's canvas."""

    if abs(scale - 1.0) < 1e-9:
        return staged_input, {
            "policy": "source-resolution layer editing",
            "native_scale": 1,
            "reused_v3_cache": False,
        }
    missing_v3 = [path for path in (*V3_ENGINE_FILES, *V3_MODEL_FILES) if not path.is_file()]
    if missing_v3:
        raise UserError(
            "V5 x1 vẫn dùng được, nhưng n>1 cần bộ model V3 để làm nét các layer. "
            f"Đang thiếu: {missing_v3[0]}"
        )
    source_sha = sha256_file(source)
    normalized_sha = canonical_pixel_sha256(staged_input)
    native = find_cached_v3_native(source_sha, normalized_sha)
    reused = native is not None
    if native is None:
        if not python_runtime_has_cuda(str(PYTHON_V5)):
            raise UserError(
                "V5 n>1 cần NVIDIA CUDA để tạo master V3 lần đầu trên máy này. "
                "Máy CPU vẫn tách layer bằng n=1; hoặc cài runtime CUDA/checkpoint V3 rồi chạy lại."
            )
        generated = job_dir / "v5_v3_native_x4.png"
        print("  Master V5: chưa có cache; đang chạy V3 GPU native x4 một lần...", flush=True)
        subprocess.run(
            [
                str(PYTHON_V5),
                "-B",
                str(V3_ENGINE),
                str(staged_input),
                "4",
                str(generated),
                "--tile",
                "512",
                "--overlap",
                "128",
                "--force",
            ],
            check=True,
            cwd=ROOT_DIR,
        )
        cache_target = APP_DIR / "masters" / "V4_AI_BASE" / f"{normalized_sha}_NATIVE_x4.png"
        atomic_install(generated, cache_target)
        with Image.open(cache_target) as native_image:
            native_image.load()
            native_size = list(native_image.size)
        write_json_atomic(
            {
                "pipeline": "V4_V3_NATIVE_CACHE",
                "cache_version": V3_CACHE_CONFIG_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(source),
                "source_sha256": source_sha,
                "normalized_stage_sha256": normalized_sha,
                "canonical_pixel_sha256": normalized_sha,
                "cache_key_type": "canonical_srgb_pixels_v1",
                "v3_cache_signature": current_v3_cache_signature(),
                "native_sha256": sha256_file(cache_target),
                "native_size": native_size,
            },
            cache_target.with_suffix(cache_target.suffix + ".json"),
        )
        native = cache_target
    else:
        print(f"  Master V5: tái sử dụng V3 native x4 đã kiểm định: {native}", flush=True)

    final_size = tuple(int(round(value * scale)) for value in inspect_image(staged_input)[0])
    if abs(scale - 4.0) < 1e-9:
        master = native
    else:
        master = job_dir / f"v5_ai_master_x{scale_tag(scale)}.png"
        with Image.open(native) as image:
            image.load()
            resized = image.convert("RGB").resize(final_size, Image.Resampling.LANCZOS)
            resized.save(master, format="PNG", compress_level=3, icc_profile=srgb_profile_bytes())
    return master, {
        "policy": "V3 fused neural native x4; one Lanczos resize only when requested scale differs from x4",
        "native_scale": 4,
        "native_path": str(native),
        "native_sha256": sha256_file(native),
        "reused_v3_cache": reused,
        "final_master_sha256": sha256_file(master),
    }


def prepare_v4_deep_base(
    source: Path,
    staged_input: Path,
    job_dir: Path,
    proof_size: tuple[int, int],
    scale: float,
    allow_huge: bool,
) -> tuple[Path, dict[str, object]]:
    """Create or reuse the Deep raster evaluated as a V4 ablation candidate."""
    source_sha = sha256_file(source)
    normalized_sha = canonical_pixel_sha256(staged_input)
    native = find_cached_v3_native(source_sha, normalized_sha)
    reused_v3 = native is not None
    if native is None:
        if not PYTHON.is_file() or not V3_ENGINE.is_file():
            raise UserError("V4 PRINT cần engine V3 để tạo lớp AI; engine V3 đang bị thiếu.")
        generated_native = job_dir / "v3_ai_native_x4.png"
        command = [
            str(PYTHON),
            "-B",
            str(V3_ENGINE),
            str(staged_input),
            "4",
            str(generated_native),
            "--tile",
            "512",
            "--overlap",
            "128",
            "--force",
        ]
        print("  AI tầng 1: chưa có cache; đang chạy V3 GPU native x4...", flush=True)
        subprocess.run(command, check=True, cwd=ROOT_DIR)
        cache_target = (
            APP_DIR
            / "masters"
            / "V4_AI_BASE"
            / f"{normalized_sha}_NATIVE_x4.png"
        )
        atomic_install(generated_native, cache_target)
        with Image.open(cache_target) as native_image:
            native_image.load()
            native_size = list(native_image.size)
        write_json_atomic(
            {
                "pipeline": "V4_V3_NATIVE_CACHE",
                "cache_version": V3_CACHE_CONFIG_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(source),
                "source_sha256": source_sha,
                "normalized_stage_sha256": normalized_sha,
                "canonical_pixel_sha256": normalized_sha,
                "cache_key_type": "canonical_srgb_pixels_v1",
                "v3_cache_signature": current_v3_cache_signature(),
                "native_sha256": sha256_file(cache_target),
                "native_size": native_size,
            },
            cache_target.with_suffix(cache_target.suffix + ".json"),
        )
        native = cache_target
    else:
        print(f"  AI tầng 1: tái sử dụng master V3 x4 đã kiểm định: {native}", flush=True)

    native_sha = sha256_file(native)
    cached = find_cached_v4_deep(
        normalized_sha,
        native_sha,
        scale,
        legacy_source_sha256=source_sha,
    )
    reused_deep = cached is not None
    if cached is not None:
        raster_base, deep_manifest = cached
        print(f"  AI tầng 2: tái sử dụng V4 Deep HAT x{scale:g}: {raster_base}", flush=True)
    else:
        if not PYTHON.is_file() or not V4_DEEP_ENGINE.is_file():
            raise UserError("Thiếu engine V4 Deep hoặc môi trường CUDA V3.")
        generated_deep = job_dir / f"v4_deep_x{scale_tag(scale)}.png"
        command = [
            str(PYTHON),
            "-B",
            str(V4_DEEP_ENGINE),
            str(staged_input),
            str(native),
            f"{scale:g}",
            str(generated_deep),
            "--tile",
            "512",
            "--overlap",
            "128",
            "--dtype",
            "auto",
            "--force",
        ]
        if allow_huge:
            command.append("--allow-huge")
        print(
            "  AI tầng 2: đang chạy HAT phục hồi lần hai trên toàn ảnh; "
            "đây là bước V4 chất lượng cao và sẽ lâu...",
            flush=True,
        )
        subprocess.run(command, check=True, cwd=ROOT_DIR)
        generated_manifest = generated_deep.with_suffix(generated_deep.suffix + ".json")
        if not generated_manifest.is_file():
            raise RuntimeError("V4 Deep không tạo manifest kiểm định.")
        deep_manifest = json.loads(generated_manifest.read_text(encoding="utf-8"))
        deep_manifest["deep_input"] = deep_manifest.get("source")
        deep_manifest["deep_input_sha256"] = deep_manifest.get("source_sha256")
        deep_manifest["source"] = str(source)
        deep_manifest["source_sha256"] = source_sha
        deep_manifest["canonical_pixel_sha256"] = normalized_sha
        deep_manifest["cache_key_type"] = "canonical_srgb_pixels_v1"
        cache_target = (
            APP_DIR
            / "masters"
            / "V4_DEEP"
            / f"{normalized_sha}_DEEP_x{scale_tag(scale)}.png"
        )
        atomic_install(generated_deep, cache_target)
        deep_manifest["output"] = str(cache_target)
        deep_manifest["output_sha256"] = sha256_file(cache_target)
        write_json_atomic(
            deep_manifest, cache_target.with_suffix(cache_target.suffix + ".json")
        )
        raster_base = cache_target

    with Image.open(raster_base) as image:
        image.load()
        if image.size != proof_size:
            raise RuntimeError(f"V4 Deep tạo {image.size}, cần đúng {proof_size}.")
    return raster_base, {
        "pipeline": "V4_DEEP_RECURSIVE_HAT",
        "native_path": str(native),
        "native_sha256": native_sha,
        "deep_path": str(raster_base),
        "deep_sha256": sha256_file(raster_base),
        "deep_size": list(proof_size),
        "deep_scale": scale,
        "normalized_stage_sha256": normalized_sha,
        "canonical_pixel_sha256": normalized_sha,
        "cache_key_type": "canonical_srgb_pixels_v1",
        "v3_cache_signature": current_v3_cache_signature(),
        "deep_run": deep_manifest.get("run"),
        "reused_v3_cache": reused_v3,
        "reused_deep_cache": reused_deep,
    }


def scale_tag(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_install(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".new")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def write_json_atomic(data: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".new")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


DIRECTORY_PUBLISH_SCHEMA = "local-print-image-upscaler/directory-publish/1"


def _directory_publish_journal(target: Path) -> Path:
    return target.parent / f".{target.name}.publish.json"


def _validated_publish_backup(target: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("Directory publish journal has no backup path.")
    backup = Path(raw).resolve()
    parent = target.parent.resolve()
    if backup.parent != parent or not re.fullmatch(
        rf"\.{re.escape(target.name)}\.old-[0-9a-f]{{32}}",
        backup.name,
    ):
        raise RuntimeError("Directory publish journal points outside its target namespace.")
    return backup


def recover_interrupted_directory_publish(target: Path) -> None:
    """Recover only a target-specific, journaled interrupted directory swap."""

    journal = _directory_publish_journal(target)
    if not journal.exists():
        return
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Invalid directory publish journal: {journal}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != DIRECTORY_PUBLISH_SCHEMA:
        raise RuntimeError(f"Invalid directory publish journal: {journal}")
    if not _same_canonical_path(payload.get("target"), target):
        raise RuntimeError(f"Directory publish journal belongs to another target: {journal}")
    backup = _validated_publish_backup(target, payload.get("backup"))
    phase = str(payload.get("phase", ""))

    if target.exists():
        if backup.exists():
            if not backup.is_dir():
                raise RuntimeError(f"Unsafe non-directory publish backup: {backup}")
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                print(
                    f"CẢNH BÁO: bundle chính đã an toàn; chưa dọn được backup {backup}: {exc}",
                    file=sys.stderr,
                )
                return
        journal.unlink(missing_ok=True)
        return
    if backup.exists():
        if not backup.is_dir():
            raise RuntimeError(f"Unsafe non-directory publish backup: {backup}")
        os.replace(backup, target)
        journal.unlink(missing_ok=True)
        return
    if phase == "prepared":
        journal.unlink(missing_ok=True)
        return
    raise RuntimeError(
        f"Ambiguous interrupted publish; target and verified backup are both missing: {journal}"
    )


def atomic_install_directory(source: Path, target: Path) -> None:
    """Publish a validated bundle with rollback and next-run crash recovery."""

    target.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    target = target.resolve()
    if not source.is_dir() or source.parent != target.parent:
        raise RuntimeError("Directory publish source must be a staging directory beside its target.")
    recover_interrupted_directory_publish(target)
    backup = target.with_name(f".{target.name}.old-{uuid.uuid4().hex}")
    journal = _directory_publish_journal(target)
    state: dict[str, object] = {
        "schema": DIRECTORY_PUBLISH_SCHEMA,
        "target": str(target),
        "source": str(source),
        "backup": str(backup),
        "phase": "prepared",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(state, journal)
    had_previous = target.exists()
    if had_previous:
        os.replace(target, backup)
        state["phase"] = "old_moved"
        try:
            write_json_atomic(state, journal)
        except BaseException:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            journal.unlink(missing_ok=True)
            raise
    try:
        os.replace(source, target)
    except BaseException:
        if had_previous and backup.exists() and not target.exists():
            os.replace(backup, target)
        journal.unlink(missing_ok=True)
        raise
    state["phase"] = "published"
    try:
        write_json_atomic(state, journal)
    except OSError as exc:
        print(
            f"CẢNH BÁO: bundle đã publish; chưa cập nhật được journal {journal}: {exc}",
            file=sys.stderr,
        )
    if backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            print(
                f"CẢNH BÁO: bundle mới đã publish; backup sẽ được dọn ở lượt sau {backup}: {exc}",
                file=sys.stderr,
            )
            return
    journal.unlink(missing_ok=True)


def _is_link_or_reparse_point(path: Path) -> bool:
    """Return True for symlinks and Windows junction/other reparse entries."""

    metadata = path.lstat()
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _assert_no_reparse_points_below(root: Path) -> None:
    """Refuse recursive deletion when any entry could redirect outside ``root``."""

    if _is_link_or_reparse_point(root):
        raise RuntimeError(f"Refusing to remove V5 staging link/reparse point: {root}")

    def fail_closed(error: OSError) -> None:
        raise RuntimeError(
            f"Cannot fully validate V5 staging before cleanup: {error.filename or root}"
        ) from error

    for current_raw, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=fail_closed,
        followlinks=False,
    ):
        current = Path(current_raw)
        for name in (*directory_names, *file_names):
            candidate = current / name
            try:
                redirected = _is_link_or_reparse_point(candidate)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"V5 staging changed while it was being validated: {candidate}"
                ) from exc
            if redirected:
                raise RuntimeError(
                    f"Refusing to remove V5 staging containing link/reparse point: {candidate}"
                )


def cleanup_stale_v5_output_staging(
    target: Path,
    *,
    active_staging: Path | None = None,
) -> list[Path]:
    """Remove only abandoned launcher staging directories for one V5 target.

    The public launcher calls this while holding ``gpu_job_lock`` and before the
    new staging directory exists.  ``active_staging`` is still excluded so this
    helper remains fail-safe if its lifecycle is changed later.
    """

    declared_target = Path(target).expanduser()
    declared_parent = Path(os.path.abspath(declared_target.parent))
    try:
        parent = declared_target.parent.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot validate the V5 output parent before staging cleanup: {declared_target.parent}"
        ) from exc
    if os.path.normcase(str(declared_parent)) != os.path.normcase(str(parent)):
        raise RuntimeError(
            f"Refusing V5 staging cleanup through a redirected output parent: {declared_target.parent}"
        )
    if not parent.is_dir() or _is_link_or_reparse_point(parent):
        raise RuntimeError(f"Unsafe V5 output parent for staging cleanup: {parent}")

    pattern = re.compile(
        rf"\.{re.escape(declared_target.name)}\.new-[0-9a-f]{{32}}",
        flags=re.ASCII,
    )
    active_name: str | None = None
    if active_staging is not None:
        active = Path(active_staging)
        active_parent = Path(os.path.abspath(active.parent))
        if (
            os.path.normcase(str(active_parent)) != os.path.normcase(str(parent))
            or pattern.fullmatch(active.name) is None
        ):
            raise RuntimeError("Active V5 staging path is outside the target namespace.")
        active_name = os.path.normcase(active.name)

    removed: list[Path] = []
    for candidate in parent.iterdir():
        if pattern.fullmatch(candidate.name) is None:
            continue
        if active_name is not None and os.path.normcase(candidate.name) == active_name:
            continue
        try:
            if _is_link_or_reparse_point(candidate):
                raise RuntimeError(
                    f"Refusing to remove V5 staging link/reparse point: {candidate}"
                )
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            # Another filesystem actor removed it after enumeration.  The GPU
            # lock prevents another launcher job, so there is nothing to clean.
            continue
        if (
            resolved.parent != parent
            or resolved.name != candidate.name
            or not resolved.is_dir()
        ):
            raise RuntimeError(f"Unsafe V5 staging candidate: {candidate}")
        _assert_no_reparse_points_below(resolved)
        if _is_link_or_reparse_point(resolved):
            raise RuntimeError(f"V5 staging changed before cleanup: {resolved}")
        shutil.rmtree(resolved)
        removed.append(resolved)
    return removed


@contextlib.contextmanager
def gpu_job_lock():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = WORK_DIR / "upscale.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise UserError("Đang có một lệnh upscale khác sử dụng GPU. Hãy đợi lệnh đó xong.") from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def run_v4_job(
    mode: str,
    source: Path,
    scale: float,
    allow_huge: bool,
    v4_options: dict[str, object],
    *,
    output_subdir: Path | None = None,
    output_stem: str | None = None,
    batch_root: Path | None = None,
) -> Path:
    if not PYTHON_V4.is_file():
        raise UserError(
            f"Thiếu môi trường V4: {PYTHON_V4}. Hãy chạy APP\\engines\\V4\\setup_v4.ps1."
        )
    if not V4_ENGINE.is_file():
        raise UserError(f"Thiếu engine V4: {V4_ENGINE}")

    source_size, source_mode, icc_profile = inspect_image(source)
    proof_size = tuple(int(round(value * scale)) for value in source_size)
    megapixels = proof_size[0] * proof_size[1] / 1_000_000
    full_vector = mode == "V4_VECTOR"
    resource_plan = validate_v4_resource_plan(
        source_size,
        proof_size,
        full_vector=full_vector,
        allow_huge=allow_huge,
    )

    relative_dir = output_subdir or Path()
    result_stem = output_stem or source.stem
    file_tag = "V4_VECTOR" if full_vector else "V4_PRINT"
    target_dir = OUTPUT_DIR / mode / relative_dir / f"{result_stem}_{file_tag}"
    report_path = (
        APP_DIR
        / "manifests"
        / mode
        / relative_dir
        / f"{result_stem}_{file_tag}_x{scale_tag(scale)}.json"
    )
    target_svg = target_dir / f"{result_stem}_EDITABLE.svg"
    target_pdf = target_dir / f"{result_stem}_PRINT_PDFX4.pdf"
    target_png = target_dir / f"{result_stem}_PREVIEW_x{scale_tag(scale)}.png"

    width_mm = v4_options.get("width_mm")
    bleed_mm = float(v4_options.get("bleed_mm", 0.0))
    profile_name = str(
        v4_options.get("profile_name", "ISO Coated v2 300% (basICColor)")
    )

    print("\nTHÔNG TIN LỆNH")
    print(
        "  Chế độ : "
        + (
            "V4 VECTOR (toàn path, dành cho đồ họa phẳng)"
            if full_vector
            else "V4 DEEP PRINT (guarded-USM/Deep ablation + gate native-x4 + PDF/X-4)"
        )
    )
    print(f"  Input  : {source}")
    print(f"  Nguồn  : {source_size[0]}x{source_size[1]} px, {source_mode}")
    print(f"  Preview: {proof_size[0]}x{proof_size[1]} px ({megapixels:.1f} MP)")
    print(
        "  Tài nguyên ước tính: "
        f"{resource_plan['estimated_peak_ram_gib']:.2f} GiB RAM, "
        f"{resource_plan['estimated_working_disk_gib']:.2f} GiB ổ tạm"
    )
    if width_mm is not None:
        height_mm = float(width_mm) * source_size[1] / source_size[0]
        print(f"  Khổ in : {float(width_mm):g} x {height_mm:g} mm")
    else:
        print("  Khổ in : theo DPI nguồn (có thể đặt bằng --width-mm)")
    print(f"  Output : {target_dir}\n", flush=True)

    started = time.perf_counter()
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_bundle = target_dir.parent / f".{target_dir.name}.new-{uuid.uuid4().hex}"
    try:
        with tempfile.TemporaryDirectory(prefix="job_v4_", dir=WORK_DIR) as temporary_raw:
            job_dir = Path(temporary_raw)
            staged_input = stage_input(source, job_dir, icc_profile)
            canonical_input_sha256 = canonical_pixel_sha256(staged_input)
            raster_base: Path | None = None
            ai_base_info: dict[str, object] | None = None
            if not full_vector:
                raster_base, ai_base_info = prepare_v4_deep_base(
                    source,
                    staged_input,
                    job_dir,
                    proof_size,
                    scale,
                    allow_huge,
                )
            command = [
                str(PYTHON_V4),
                "-B",
                str(V4_ENGINE),
                str(staged_input),
                f"{scale:g}",
                str(staging_bundle),
                "--name",
                result_stem,
                "--bleed-mm",
                f"{bleed_mm:g}",
                "--profile-name",
                profile_name,
            ]
            if raster_base is not None:
                command.extend(
                    [
                        "--raster-base",
                        str(raster_base),
                        "--v3-baseline",
                        str(ai_base_info["native_path"]),
                    ]
                )
            else:
                command.append("--full-vector")
            if width_mm is not None:
                command.extend(["--width-mm", f"{float(width_mm):g}"])
            if allow_huge:
                command.append("--allow-huge")

            subprocess.run(command, check=True, cwd=ROOT_DIR)
            engine_manifest_path = staging_bundle / "manifest.json"
            if not engine_manifest_path.is_file():
                raise RuntimeError("V4 không tạo báo cáo kiểm định.")
            metadata = json.loads(engine_manifest_path.read_text(encoding="utf-8"))
            if not metadata.get("visual_qa", {}).get("passed"):
                raise RuntimeError("V4 không vượt kiểm tra chất lượng hình ảnh.")
            if (
                not full_vector
                and not metadata.get("hybrid_retention_qa", {}).get("passed")
            ):
                raise RuntimeError("V4 PRINT làm suy giảm raster đã được chọn vượt ngưỡng an toàn.")
            if not full_vector:
                comparative = metadata.get("comparative_v3_v4_qa")
                if not isinstance(comparative, dict):
                    raise RuntimeError("V4 PRINT thiếu cổng so sánh trực tiếp V3/V4.")
                if comparative.get("final_output_not_worse_than_v3") is not True:
                    raise RuntimeError("V4 PRINT chưa chứng minh đầu ra cuối không kém V3.")
            svg_images = int(metadata.get("vector", {}).get("embedded_image_count", -1))
            pdf_images = int(metadata.get("pdfx4_qa", {}).get("images", -1))
            if full_vector and (svg_images != 0 or pdf_images != 0):
                raise RuntimeError("Chế độ V4 VECTOR còn chứa ảnh raster nhúng.")
            if not full_vector and (svg_images < 1 or pdf_images < 1):
                raise RuntimeError("V4 PRINT thiếu lớp raster đã được QA và khai báo provenance.")
            if metadata.get("pdfx4_qa", {}).get("gts_pdfx_version") != "PDF/X-4":
                raise RuntimeError("PDF V4 không được nhận diện là PDF/X-4.")
            if metadata.get("pdf_placement_qa", {}).get("passed") is not True:
                raise RuntimeError("PDF V4 chưa chứng minh artwork phủ đúng trang in và bleed.")

            staged_files = {
                kind: staging_bundle / payload["name"]
                for kind, payload in metadata.get("outputs", {}).items()
            }
            if set(staged_files) != {"svg", "pdf", "png"}:
                raise RuntimeError("Bundle V4 không đủ SVG, PDF và PNG.")
            for kind, path in staged_files.items():
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"File V4 {kind.upper()} bị thiếu hoặc rỗng.")
            with Image.open(staged_files["png"]) as proof:
                proof.load()
                if proof.size != proof_size:
                    raise RuntimeError(f"V4 tạo preview {proof.size}, cần {proof_size}.")

            total_seconds = round(time.perf_counter() - started, 3)
            engine_manifest_path.unlink()
            atomic_install_directory(staging_bundle, target_dir)

            metadata.update(
                {
                    "launcher": "Local Print Image Upscaler unified command",
                    "app_version": APP_VERSION,
                    "source": str(source),
                    "source_sha256": sha256_file(source),
                    "source_size": list(source_size),
                    "canonical_input_sha256": canonical_input_sha256,
                    "resource_plan": resource_plan,
                    "final_bundle": str(target_dir),
                    "ai_base": ai_base_info,
                    "launcher_total_seconds": total_seconds,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            for kind, final_path in {
                "svg": target_svg,
                "pdf": target_pdf,
                "png": target_png,
            }.items():
                metadata["outputs"][kind].update(
                    {
                        "path": str(final_path),
                        "sha256": sha256_file(final_path),
                        "bytes": final_path.stat().st_size,
                    }
                )
            if batch_root is not None:
                metadata["batch_root"] = str(batch_root)
                metadata["batch_relative_source"] = str(source.relative_to(batch_root))
            write_json_atomic(metadata, report_path)
            report_prefix = f"{result_stem}_{file_tag}_x"
            for stale_report in report_path.parent.glob("*.json"):
                if stale_report != report_path and stale_report.name.startswith(report_prefix):
                    stale_report.unlink()
    finally:
        if staging_bundle.exists():
            shutil.rmtree(staging_bundle)

    print("\nHOÀN TẤT V4")
    print(f"  SVG chỉnh sửa : {target_svg}")
    print(f"  PDF giao in   : {target_pdf}")
    print(f"  PNG xem nhanh : {target_png}")
    print(f"  Thời gian     : {total_seconds:.1f} giây")
    return target_dir


def _build_v5_engine_command(
    *,
    staged_input: Path,
    scale: float,
    staging_bundle: Path,
    master: Path,
    result_stem: str,
    options: dict[str, object],
    prior_review_file: Path | None,
) -> list[str]:
    command = [
        str(PYTHON_V5),
        "-B",
        str(V5_ENGINE),
        str(staged_input),
        f"{scale:g}",
        str(staging_bundle),
        "--master",
        str(master),
        "--name",
        result_stem,
        "--detail",
        str(options.get("detail", "exhaustive")),
        "--review-mode",
        str(options.get("review", "gui")),
        "--inpaint",
        str(options.get("inpaint", "auto")),
        "--app-version",
        APP_VERSION,
    ]
    if not bool(options.get("semantic", True)):
        command.append("--no-semantic")
    if prior_review_file is not None:
        command.extend(["--review-file", str(prior_review_file)])
    return command


def _v5_resume_metadata_matches(
    metadata: dict[str, object] | None,
    source: Path,
    scale: float,
) -> bool:
    """Reject legacy/incompatible checkpoints before launching costly V5 inference."""

    if not metadata:
        return False
    if metadata.get("engine_generation") != "V5_PRO_EXHAUSTIVE_V2":
        return False
    if metadata.get("review_resume_signature") != V5_REVIEW_RESUME_SIGNATURE:
        return False
    raw_scale = metadata.get("scale")
    if isinstance(raw_scale, bool):
        return False
    try:
        if not math.isclose(float(raw_scale), float(scale), rel_tol=0.0, abs_tol=1e-9):
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        _same_canonical_path(metadata.get("original_source"), source)
        and metadata.get("original_source_sha256") == sha256_file(source)
    )


def _validate_v5_publish_manifest(
    metadata: dict[str, object],
    final_size: tuple[int, int],
) -> None:
    """Allow review-needed bundles, but never atomically publish a hard FAIL."""

    if metadata.get("pipeline") != "V5_SMART_EDITABLE_LAYERS":
        raise RuntimeError("Manifest V5 has the wrong pipeline.")
    if metadata.get("final_size") != list(final_size):
        raise RuntimeError("Manifest V5 has the wrong final canvas size.")
    grouping = metadata.get("grouping")
    selected = (
        int(grouping.get("selected_layer_count", 0))
        if isinstance(grouping, dict)
        else 0
    )
    if selected < 1:
        raise RuntimeError("V5 produced no valid foreground/editable layer.")
    qa_status = str(metadata.get("qa_status") or "")
    if qa_status == "FAIL":
        raise RuntimeError(
            "V5 hard QA failed; staging will not replace the last good output."
        )
    if qa_status not in {"PASS", "REVIEW_REQUIRED"}:
        raise RuntimeError(f"V5 manifest has an invalid QA status: {qa_status or 'missing'}")


def _v5_failure_diagnostic_target(result_stem: str) -> Path:
    safe_stem = re.sub(r"[\x00-\x1f]+", "_", safe_folder_name(result_stem))
    safe_stem = safe_stem.strip(" ._")[:80] or "image"
    work_root = WORK_DIR.resolve()
    target = (work_root / f"v5_last_failure_{safe_stem}").resolve()
    if target.parent != work_root or target.name != f"v5_last_failure_{safe_stem}":
        raise RuntimeError("Unsafe V5 failure diagnostic target.")
    return target


def _preserve_v5_failure_diagnostics(
    staging_bundle: Path,
    result_stem: str,
) -> Path | None:
    """Atomically retain only small QA evidence from a hard-failed V5 run."""

    staging = staging_bundle.resolve()
    manifest_source = staging / "manifest.json"
    if not staging.is_dir() or not manifest_source.is_file():
        return None
    if manifest_source.stat().st_size > V5_FAILURE_DIAGNOSTIC_MAX_FILE_BYTES:
        raise RuntimeError("Required V5 diagnostic exceeds the per-file safety cap: manifest.json")
    try:
        manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or str(manifest.get("qa_status") or "") != "FAIL":
        return None

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    work_root = WORK_DIR.resolve()
    target = _v5_failure_diagnostic_target(result_stem)
    if target.exists() and not target.is_dir():
        raise RuntimeError(f"Unsafe non-directory V5 diagnostic target: {target}")
    temporary = work_root / f".{target.name}.new-{uuid.uuid4().hex}"
    if temporary.parent.resolve() != work_root or not re.fullmatch(
        rf"\.{re.escape(target.name)}\.new-[0-9a-f]{{32}}",
        temporary.name,
    ):
        raise RuntimeError("Unsafe V5 diagnostic staging path.")
    required_relatives = (
        Path("manifest.json"),
        Path(TECHNICAL_DIR_NAME) / "QA_REPORT.json",
        Path(TECHNICAL_DIR_NAME) / "QA_REPORT.html",
    )
    optional_relatives = (
        Path(TECHNICAL_DIR_NAME) / "RECOMPOSITION_DIFF_X8.png",
        Path(TECHNICAL_DIR_NAME) / "LAYER_REVIEW.json",
        Path(TECHNICAL_DIR_NAME) / "SOURCE_FOR_REVIEW.png",
        Path(TECHNICAL_DIR_NAME) / "BACKGROUND_FOR_OCR_QA.png",
    )
    copied = 0
    copied_bytes = 0
    try:
        temporary.mkdir(parents=False, exist_ok=False)
        for relative in (*required_relatives, *optional_relatives):
            source = staging / relative
            if not source.is_file():
                continue
            size = source.stat().st_size
            required = relative in required_relatives
            if size > V5_FAILURE_DIAGNOSTIC_MAX_FILE_BYTES:
                if required:
                    raise RuntimeError(
                        f"Required V5 diagnostic exceeds the per-file safety cap: {relative}"
                    )
                continue
            if copied_bytes + size > V5_FAILURE_DIAGNOSTIC_MAX_TOTAL_BYTES:
                if required:
                    raise RuntimeError(
                        f"Required V5 diagnostics exceed the total safety cap at: {relative}"
                    )
                continue
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            actual_size = destination.stat().st_size
            if actual_size > V5_FAILURE_DIAGNOSTIC_MAX_FILE_BYTES:
                destination.unlink(missing_ok=True)
                if required:
                    raise RuntimeError(
                        f"Required V5 diagnostic grew beyond the safety cap: {relative}"
                    )
                continue
            if copied_bytes + actual_size > V5_FAILURE_DIAGNOSTIC_MAX_TOTAL_BYTES:
                destination.unlink(missing_ok=True)
                if required:
                    raise RuntimeError(
                        f"Required V5 diagnostics grew beyond the total safety cap at: {relative}"
                    )
                continue
            copied_bytes += actual_size
            copied += 1
        if not copied:
            return None
        atomic_install_directory(temporary, target)
        return target
    finally:
        if temporary.exists():
            resolved_temporary = temporary.resolve()
            if resolved_temporary.parent != work_root:
                raise RuntimeError("Refusing to remove unsafe V5 diagnostic staging path.")
            shutil.rmtree(resolved_temporary)


def run_v5_job(
    source: Path,
    scale: float,
    allow_huge: bool,
    options: dict[str, object],
    *,
    output_subdir: Path | None = None,
    output_stem: str | None = None,
    batch_root: Path | None = None,
) -> Path:
    if not PYTHON_V5.is_file():
        raise UserError(
            "Thiếu môi trường V5. Hãy chạy APP\\engines\\V5\\setup_v5.ps1 một lần."
        )
    if not V5_ENGINE.is_file():
        raise UserError(f"Thiếu engine V5: {V5_ENGINE}")
    source_size, source_mode, icc_profile = inspect_image(source)
    final_size = tuple(int(round(value * scale)) for value in source_size)
    resource_plan = validate_v5_resource_plan(
        source_size,
        final_size,
        allow_huge=allow_huge,
        max_layers=int(options.get("max_layers", 24)),
    )
    relative_dir = output_subdir or Path()
    result_stem = output_stem or source.stem
    tag = scale_tag(scale)
    target_dir = OUTPUT_DIR / "V5_LAYERS" / relative_dir / f"{result_stem}_V5_LAYERS_x{tag}"
    report_path = (
        APP_DIR
        / "manifests"
        / "V5_LAYERS"
        / relative_dir
        / f"{result_stem}_V5_LAYERS_x{tag}.json"
    )
    prior_review_file: Path | None = None
    if target_dir.is_dir():
        try:
            prior = resolve_v5_review_target(target_dir)
        except UserError as exc:
            print(
                f"CẢNH BÁO: không tái sử dụng checkpoint V5 cũ vì bundle không hợp lệ: {exc}",
                file=sys.stderr,
            )
        else:
            if prior is not None:
                candidate, prior_metadata = prior
                if _v5_resume_metadata_matches(prior_metadata, source, scale):
                    prior_review_file = candidate
                else:
                    print(
                        "CẢNH BÁO: checkpoint V5 cũ không khớp chính xác ảnh/hệ số hiện tại; "
                        "engine sẽ tạo kiểm kê mới thay vì nhập quyết định sai.",
                        file=sys.stderr,
                    )

    print("\nTHÔNG TIN LỆNH")
    print("  Chế độ : V5 PRO EXHAUSTIVE LAYERS (inventory + ownership + review + clean plate)")
    print(f"  Input  : {source}")
    print(f"  Nguồn  : {source_size[0]}x{source_size[1]} px, {source_mode}")
    print(f"  Canvas : {final_size[0]}x{final_size[1]} px ({resource_plan['output_megapixels']:.1f} MP)")
    print("  Layer  : không cắt theo số lượng; mọi proposal đều được gán, từ chối có lý do hoặc đưa ra duyệt")
    print(f"  Duyệt  : {options.get('review', 'gui')}")
    if prior_review_file is not None:
        print(f"  Resume : nạp checkpoint đã duyệt và xác minh lại từ {prior_review_file}")
    print(
        "  Tài nguyên ước tính bảo thủ: "
        f"{resource_plan['estimated_peak_ram_gib']:.2f} GiB RAM, "
        f"{resource_plan['estimated_working_disk_gib']:.2f} GiB ổ tạm"
    )
    print(f"  Output : {target_dir}\n", flush=True)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_bundle = target_dir.parent / f".{target_dir.name}.new-{uuid.uuid4().hex}"
    removed_staging = cleanup_stale_v5_output_staging(
        target_dir,
        active_staging=staging_bundle,
    )
    if removed_staging:
        suffix = "directory" if len(removed_staging) == 1 else "directories"
        print(f"  Cleaned {len(removed_staging)} abandoned V5 staging {suffix}.")
    started = time.perf_counter()
    failure_diagnostic: Path | None = None
    failure_diagnostic_attempted = False

    def preserve_failure_diagnostic_once() -> None:
        nonlocal failure_diagnostic, failure_diagnostic_attempted
        if failure_diagnostic_attempted:
            return
        failure_diagnostic_attempted = True
        try:
            failure_diagnostic = _preserve_v5_failure_diagnostics(
                staging_bundle,
                result_stem,
            )
        except (OSError, RuntimeError) as exc:
            print(
                f"CẢNH BÁO: không giữ được chẩn đoán V5 FAIL tối giản: {exc}",
                file=sys.stderr,
            )
            return
        if failure_diagnostic is not None:
            print(
                f"  Chẩn đoán V5 FAIL đã giữ tại: {failure_diagnostic}",
                file=sys.stderr,
                flush=True,
            )

    try:
        with tempfile.TemporaryDirectory(prefix="job_v5_", dir=WORK_DIR) as temporary_raw:
            job_dir = Path(temporary_raw)
            staged_input = stage_input(source, job_dir, icc_profile)
            normalized_sha = canonical_pixel_sha256(staged_input)
            master, master_info = prepare_v5_ai_master(source, staged_input, job_dir, scale)
            command = _build_v5_engine_command(
                staged_input=staged_input,
                scale=scale,
                staging_bundle=staging_bundle,
                master=master,
                result_stem=result_stem,
                options=options,
                prior_review_file=prior_review_file,
            )
            subprocess.run(command, check=True, cwd=ROOT_DIR)

            engine_manifest_path = staging_bundle / "manifest.json"
            if not engine_manifest_path.is_file():
                raise RuntimeError("V5 không tạo manifest kiểm định.")
            metadata = json.loads(engine_manifest_path.read_text(encoding="utf-8"))
            if str(metadata.get("qa_status") or "") == "FAIL":
                preserve_failure_diagnostic_once()
            _validate_v5_publish_manifest(metadata, final_size)
            if metadata.get("pipeline") != "V5_SMART_EDITABLE_LAYERS":
                raise RuntimeError("Manifest V5 sai pipeline.")
            if metadata.get("final_size") != list(final_size):
                raise RuntimeError("Manifest V5 sai kích thước canvas.")
            if int(metadata.get("grouping", {}).get("selected_layer_count", 0)) < 1:
                raise RuntimeError("V5 không có layer foreground hợp lệ.")
            required_patterns = ("*_MASTER.ora", "*_LAYERS.zip", "*_PREVIEW.png")
            for pattern in required_patterns:
                if not any(staging_bundle.glob(pattern)):
                    raise RuntimeError(f"V5 thiếu đầu ra bắt buộc: {pattern}")
            metadata.update(
                {
                    "launcher": "Local Print Image Upscaler unified command",
                    "launcher_app_version": APP_VERSION,
                    "original_source": str(source),
                    "original_source_sha256": sha256_file(source),
                    "source_path_identity_sha256": canonical_path_identity(source),
                    "normalized_stage_sha256": normalized_sha,
                    "launcher_master": master_info,
                    "launcher_options": {
                        "detail": str(options.get("detail", "exhaustive")),
                        "review_mode": str(options.get("review", "gui")),
                        "inpaint": str(options.get("inpaint", "auto")),
                        "semantic": bool(options.get("semantic", True)),
                        "allow_huge": bool(allow_huge),
                    },
                    "resource_plan": resource_plan,
                    "final_bundle": str(target_dir),
                    "launcher_total_seconds": round(time.perf_counter() - started, 3),
                }
            )
            if batch_root is not None:
                metadata["batch_root"] = str(batch_root)
                metadata["batch_root_path_identity_sha256"] = canonical_path_identity(batch_root)
                metadata["batch_relative_source"] = str(source.relative_to(batch_root))
            write_json_atomic(metadata, engine_manifest_path)
            atomic_install_directory(staging_bundle, target_dir)
            try:
                write_json_atomic(metadata, report_path)
            except OSError as exc:
                print(
                    f"CẢNH BÁO: bundle đã publish; không ghi được bản sao manifest {report_path}: {exc}",
                    file=sys.stderr,
                )
    except BaseException:
        preserve_failure_diagnostic_once()
        raise
    finally:
        if staging_bundle.exists():
            shutil.rmtree(staging_bundle)

    total_seconds = time.perf_counter() - started
    print("\nHOÀN TẤT V5")
    print(f"  Mở bundle tại : {target_dir}")
    print(f"  PSD chỉnh sửa : {next(target_dir.glob('*_EDITABLE.psd'), 'đã bỏ qua do giới hạn PSD')}")
    print(f"  ORA mở chuẩn  : {next(target_dir.glob('*_MASTER.ora'))}")
    print(f"  Thời gian     : {total_seconds:.1f} giây")
    return target_dir


def run_v7_job(
    source: Path,
    scale: float,
    allow_huge: bool,
    options: dict[str, object],
    *,
    output_subdir: Path | None = None,
    output_stem: str | None = None,
    batch_root: Path | None = None,
) -> Path:
    if not PYTHON_V7.is_file():
        raise UserError(
            f"Thiếu môi trường V7: {PYTHON_V7}. "
            "Hãy chạy APP\\engines\\V7\\setup_v7.ps1 một lần."
        )
    if not V7_ENGINE.is_file():
        raise UserError(f"Thiếu engine V7: {V7_ENGINE}")
    source_size, source_mode, icc_profile = inspect_image(source)
    final_size = tuple(int(round(value * scale)) for value in source_size)
    resource_plan = validate_v7_resource_plan(
        source_size,
        final_size,
        allow_huge=allow_huge,
    )
    relative_dir = output_subdir or Path()
    result_stem = bounded_output_stem(output_stem or source.stem)
    tag = scale_tag(scale)
    target_dir = select_v7_target_directory(
        relative_dir,
        result_stem,
        tag,
        source,
        batch_root=batch_root,
    )
    report_path = (
        APP_DIR
        / "manifests"
        / "V7_REPAIR"
        / relative_dir
        / f"{target_dir.name}.json"
    )
    review_mode = str(options.get("review", "gui"))
    review_file_raw = options.get("review_file")
    review_file: Path | None = None
    if review_file_raw:
        review_file = Path(str(review_file_raw)).expanduser().resolve()
        if not review_file.is_file():
            raise UserError(f"Không tìm thấy file duyệt V7: {review_file}")
    prior_batch_review: Path | None = None
    if batch_root is not None and review_file is None and target_dir.exists():
        try:
            previous_paths = resolve_bundle_paths(target_dir)
        except BundleLayoutError as exc:
            raise UserError(
                f"Bundle V7 batch cũ không an toàn; giữ nguyên và không ghi đè: {target_dir}: {exc}"
            ) from exc
        prior_batch_review = previous_paths.review
        if prior_batch_review is None or previous_paths.manifest is None:
            raise UserError(
                f"Bundle V7 batch cũ thiếu review/manifest; giữ nguyên và không ghi đè: {target_dir}"
            )
        try:
            prior_review_data = json.loads(prior_batch_review.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise UserError(f"TEXT_REVIEW.json cũ bị hỏng; không ghi đè: {prior_batch_review}") from exc
        if not isinstance(prior_review_data, dict) or not isinstance(
            prior_review_data.get("regions"), list
        ):
            raise UserError(f"TEXT_REVIEW.json cũ sai cấu trúc; không ghi đè: {prior_batch_review}")
    tessdata_dir = (
        V7_TESSDATA
        if (V7_TESSDATA / "vie.traineddata").is_file()
        else APP_DIR / "engines" / "V5" / "models" / "tessdata"
    )

    print("\nTHÔNG TIN LỆNH")
    print("  Chế độ : V7 RESTORE (làm rõ toàn ảnh + OCR/duyệt + gỡ/vẽ lại chữ)")
    print(f"  Input  : {source}")
    print(f"  Nguồn  : {source_size[0]}x{source_size[1]} px, {source_mode}")
    print(
        f"  Đích   : {final_size[0]}x{final_size[1]} px "
        f"({resource_plan['output_megapixels']:.1f} MP)"
    )
    print(f"  Duyệt  : {review_mode}{' + file đã duyệt' if review_file else ''}")
    if prior_batch_review is not None:
        print(f"  Review : tự nạp an toàn từ {prior_batch_review}")
    print(
        "  Tài nguyên ước tính bảo thủ: "
        f"{resource_plan['estimated_peak_ram_gib']:.2f} GiB RAM, "
        f"{resource_plan['estimated_working_disk_gib']:.2f} GiB ổ tạm"
    )
    print(f"  Output : {target_dir}\n", flush=True)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_bundle = target_dir.parent / f".{target_dir.name}.new-{uuid.uuid4().hex}"
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="job_v7_", dir=WORK_DIR) as temporary_raw:
            job_dir = Path(temporary_raw)
            staged_input = stage_input(source, job_dir, icc_profile)
            normalized_sha = canonical_pixel_sha256(staged_input)
            effective_review_file = review_file
            if prior_batch_review is not None:
                review_snapshot = job_dir / "previous_TEXT_REVIEW.json"
                shutil.copy2(prior_batch_review, review_snapshot)
                effective_review_file = review_snapshot
            command = [
                str(PYTHON_V7),
                "-B",
                str(V7_ENGINE),
                str(staged_input),
                f"{scale:g}",
                str(staging_bundle),
                "--name",
                result_stem,
                "--models-root",
                str(V7_MODELS),
                "--tessdata-dir",
                str(tessdata_dir),
                "--review-mode",
                review_mode,
                "--ocr-passes",
                str(int(options.get("ocr_passes", 3))),
                "--inpaint",
                str(options.get("inpaint", "auto")),
                "--v3-python",
                str(PYTHON),
                "--v3-engine",
                str(V3_ENGINE),
                "--language-python",
                str(PYTHON),
                "--app-version",
                APP_VERSION,
            ]
            if effective_review_file is not None:
                command.extend(["--review-file", str(effective_review_file)])
            if not bool(options.get("language_model", True)):
                command.append("--no-language-model")
            subprocess.run(command, check=True, cwd=ROOT_DIR)

            try:
                engine_paths = resolve_bundle_paths(staging_bundle)
            except BundleLayoutError as exc:
                raise RuntimeError(f"Bundle kỹ thuật V7 không an toàn: {exc}") from exc
            engine_manifest_path = engine_paths.manifest
            if engine_manifest_path is None:
                raise RuntimeError("V7 không tạo manifest kiểm định.")
            metadata = json.loads(engine_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise RuntimeError("Manifest V7 phải là một JSON object.")
            if metadata.get("pipeline") != "V7_DESIGN_REPAIR":
                raise RuntimeError("Manifest V7 sai pipeline.")
            if metadata.get("final_size") != list(final_size):
                raise RuntimeError("Manifest V7 sai kích thước đầu ra.")
            try:
                engine_scale = float(metadata.get("scale"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("Manifest V7 thiếu hệ số scale hợp lệ.") from exc
            if not math.isfinite(engine_scale) or not math.isclose(
                engine_scale,
                scale,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError("Manifest V7 ghi sai hệ số scale.")
            status = metadata.get("status")
            if status not in {"PASS", "REVIEW_REQUIRED", "FAILED_QA"}:
                raise RuntimeError("Manifest V7 thiếu trạng thái QA rõ ràng.")
            required_engine_files = {
                "ảnh phục dựng": engine_paths.result,
                "ảnh so sánh": engine_paths.comparison,
                "ảnh nguồn chuẩn hóa": engine_paths.source,
                "nền sạch": engine_paths.clean,
                "QA overlay": engine_paths.overlay,
                "TEXT_REVIEW.json": engine_paths.review,
                "QA.json": engine_paths.qa,
            }
            missing_engine_files = [
                label for label, path in required_engine_files.items() if path is None
            ]
            if missing_engine_files:
                raise RuntimeError(
                    "V7 thiếu đầu ra bắt buộc: " + ", ".join(missing_engine_files)
                )
            final_review_path = target_dir / TECHNICAL_DIR_NAME / "TEXT_REVIEW.json"
            launcher_options = {
                "review_mode": review_mode,
                "ocr_passes": int(options.get("ocr_passes", 3)),
                "inpaint": str(options.get("inpaint", "auto")),
                "language_model": bool(options.get("language_model", True)),
                "allow_huge": bool(allow_huge),
            }
            rerun_argv = [
                "upscale",
                "repair",
                str(batch_root if batch_root is not None else source),
                f"{scale:g}",
                "--review",
                "defer" if batch_root is not None else review_mode,
            ]
            if batch_root is None:
                rerun_argv.extend(["--review-file", str(final_review_path)])
            rerun_argv.extend(
                [
                    "--ocr-passes",
                    str(launcher_options["ocr_passes"]),
                    "--inpaint",
                    str(launcher_options["inpaint"]),
                ]
            )
            if not launcher_options["language_model"]:
                rerun_argv.append("--no-language-model")
            if launcher_options["allow_huge"]:
                rerun_argv.append("--allow-huge")
            metadata.update(
                {
                    "launcher": "Local Print Image Upscaler unified command",
                    "launcher_app_version": APP_VERSION,
                    "original_source": str(source),
                    "original_source_sha256": sha256_file(source),
                    "source_path_identity_sha256": canonical_path_identity(source),
                    "normalized_stage_sha256": normalized_sha,
                    "resource_plan": resource_plan,
                    "final_bundle": str(target_dir),
                    "scale": float(scale),
                    "launcher_paths": {
                        "result": (
                            "01_KET_QUA_DA_DAT.png"
                            if status == "PASS"
                            else "01_XEM_TRUOC_CAN_DUYET.png"
                        ),
                        "review": f"{TECHNICAL_DIR_NAME}/TEXT_REVIEW.json",
                        "qa": f"{TECHNICAL_DIR_NAME}/QA.json",
                        "manifest": f"{TECHNICAL_DIR_NAME}/manifest.json",
                    },
                    "launcher_options": launcher_options,
                    "review_argv": ["upscale", "review", str(target_dir)],
                    "rerun_argv": rerun_argv,
                    "launcher_total_seconds": round(time.perf_counter() - started, 3),
                }
            )
            if batch_root is not None:
                metadata["batch_root"] = str(batch_root)
                metadata["batch_root_path_identity_sha256"] = canonical_path_identity(batch_root)
                metadata["batch_relative_source"] = str(source.relative_to(batch_root))
            write_json_atomic(metadata, engine_manifest_path)
            try:
                arranged_paths = arrange_bundle(staging_bundle, str(status))
            except BundleLayoutError as exc:
                raise RuntimeError(f"Không thể chuẩn hóa bundle V7: {exc}") from exc
            if arranged_paths.manifest is None:
                raise RuntimeError("Bundle V7 đã sắp xếp nhưng thiếu manifest kỹ thuật.")
            metadata = json.loads(arranged_paths.manifest.read_text(encoding="utf-8"))
            atomic_install_directory(staging_bundle, target_dir)
            try:
                write_json_atomic(metadata, report_path)
            except OSError as exc:
                print(
                    f"CẢNH BÁO: bundle đã publish; không ghi được bản sao manifest {report_path}: {exc}",
                    file=sys.stderr,
                )
    finally:
        if staging_bundle.exists():
            shutil.rmtree(staging_bundle)

    total_seconds = time.perf_counter() - started
    final_paths = resolve_bundle_paths(target_dir)
    if final_paths.manifest is None or final_paths.result is None:
        raise RuntimeError("Bundle V7 đã publish nhưng resolver không tìm thấy kết quả.")
    manifest = json.loads(final_paths.manifest.read_text(encoding="utf-8"))
    print("\nHOÀN TẤT V7")
    print(f"  Trạng thái    : {manifest['status']}")
    print(f"  Ảnh phục dựng : {final_paths.result}")
    print(f"  File duyệt    : {final_paths.review}")
    print(f"  Báo cáo QA    : {final_paths.qa}")
    print(f"  Thời gian     : {total_seconds:.1f} giây")
    if manifest["status"] != "PASS":
        print(f"  Duyệt dễ dàng : .\\upscale review \"{target_dir}\"")
        print("  Lưu ý         : chưa được gọi là bản giao in; cần duyệt chữ rồi chạy lại repair.")
    elif int(manifest.get("qa", {}).get("ocr_advisory_count", 0)) > 0:
        print("  Lưu ý         : QA cứng đạt; OCR đọc ngược còn cảnh báo dấu. Xem 02_SO_SANH.png.")
    return target_dir


def run_job(
    mode: str,
    source: Path,
    scale: float,
    allow_huge: bool,
    v4_options: dict[str, object] | None = None,
    *,
    output_subdir: Path | None = None,
    output_stem: str | None = None,
    batch_root: Path | None = None,
) -> Path:
    if mode == "V7_REPAIR":
        return run_v7_job(
            source,
            scale,
            allow_huge,
            v4_options or {},
            output_subdir=output_subdir,
            output_stem=output_stem,
            batch_root=batch_root,
        )
    if mode == "V5_LAYERS":
        return run_v5_job(
            source,
            scale,
            allow_huge,
            v4_options or {},
            output_subdir=output_subdir,
            output_stem=output_stem,
            batch_root=batch_root,
        )
    if mode in {"V4_PRINT", "V4_VECTOR"}:
        return run_v4_job(
            mode,
            source,
            scale,
            allow_huge,
            v4_options or {},
            output_subdir=output_subdir,
            output_stem=output_stem,
            batch_root=batch_root,
        )
    if not PYTHON.is_file():
        raise UserError(f"Thiếu Python nội bộ: {PYTHON}")
    if not V2_ENGINE.is_file() or not V3_ENGINE.is_file():
        raise UserError("Thiếu engine V2 hoặc V3 trong APP.")

    source_size, source_mode, icc_profile = inspect_image(source)
    final_size = tuple(int(round(value * scale)) for value in source_size)
    standard_plan = validate_standard_resource_plan(
        source_size,
        final_size,
        allow_huge=allow_huge,
    )
    megapixels = standard_plan["output_megapixels"]

    tag = scale_tag(scale)
    file_tag = "V2_FAST" if mode == "V2_FAST" else "V3_HIGH"
    relative_dir = output_subdir or Path()
    result_stem = output_stem or source.stem
    target = OUTPUT_DIR / mode / relative_dir / f"{result_stem}_{file_tag}_x{tag}.png"
    master = APP_DIR / "masters" / mode / relative_dir / f"{result_stem}_{file_tag}_NATIVE_x4.png"
    manifest = APP_DIR / "manifests" / mode / relative_dir / f"{result_stem}_{file_tag}_x{tag}.json"

    print("\nTHÔNG TIN LỆNH")
    print(f"  Chế độ : {'V2 FAST (nhanh)' if mode == 'V2_FAST' else 'V3 HIGH (chất lượng cao)'}")
    print(f"  Input  : {source}")
    print(f"  Nguồn  : {source_size[0]}x{source_size[1]} px, {source_mode}")
    print(f"  Đích   : {final_size[0]}x{final_size[1]} px ({megapixels:.1f} MP)")
    print(f"  Output : {target}\n", flush=True)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="job_", dir=WORK_DIR) as temporary_raw:
        job_dir = Path(temporary_raw)
        staged_input = stage_input(source, job_dir, icc_profile)
        staged_output = job_dir / "result.png"
        if mode == "V2_FAST":
            command = [
                str(PYTHON), "-B", str(V2_ENGINE), str(staged_input), f"{scale:g}",
                str(staged_output), "--preset", "sharp", "--tile", "512",
            ]
        else:
            command = [
                str(PYTHON), "-B", str(V3_ENGINE), str(staged_input), f"{scale:g}",
                str(staged_output), "--tile", "512", "--overlap", "128", "--force",
            ]
        if allow_huge and mode == "V2_FAST":
            command.append("--force")

        subprocess.run(command, check=True, cwd=ROOT_DIR)
        staged_native = staged_output.with_name(f"{staged_output.stem}_AI_NATIVE_X4.png")
        staged_manifest = staged_output.with_suffix(staged_output.suffix + ".json")

        with Image.open(staged_output) as result:
            result.load()
            if result.size != final_size:
                raise RuntimeError(f"Engine tao {result.size}, can {final_size}.")

        actual_master = target
        if staged_native.is_file():
            atomic_install(staged_native, master)
            actual_master = master
        atomic_install(staged_output, target)

        total_seconds = round(time.perf_counter() - started, 3)
        if mode == "V3_HIGH" and staged_manifest.is_file():
            metadata = json.loads(staged_manifest.read_text(encoding="utf-8"))
        else:
            metadata = {
                "pipeline": "V2_FAST",
                "policy": "Real-ESRGAN x4plus once; Lanczos only for the requested final scale",
                "tile": 512,
            }
        metadata.update(
            {
                "launcher": "Local Print Image Upscaler unified command",
                "app_version": APP_VERSION,
                "source": str(source),
                "source_sha256": sha256_file(source),
                "source_size": list(source_size),
                "normalized_stage_sha256": canonical_pixel_sha256(staged_input),
                "native_output": str(actual_master),
                "native_sha256": sha256_file(actual_master),
                "final_scale": scale,
                "final_output": str(target),
                "final_sha256": sha256_file(target),
                "final_size": list(final_size),
                "launcher_total_seconds": total_seconds,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        if mode == "V3_HIGH":
            metadata["v3_cache_signature"] = current_v3_cache_signature()
        if batch_root is not None:
            metadata["batch_root"] = str(batch_root)
            metadata["batch_relative_source"] = str(source.relative_to(batch_root))
        write_json_atomic(metadata, manifest)

    print("\nHOÀN TẤT")
    print(f"  Ảnh cần mở/in: {target}")
    print(f"  Thời gian    : {total_seconds:.1f} giây")
    if actual_master != target:
        print("  Master x4 và báo cáo kỹ thuật đã được cất gọn trong APP.")
    return target


def run_batch(
    mode: str,
    directory: Path,
    scale: float,
    allow_huge: bool,
    v4_options: dict[str, object],
) -> int:
    if mode == "V7_REPAIR":
        if not PYTHON_V7.is_file() or not V7_ENGINE.is_file():
            raise UserError("Thiếu Python hoặc engine V7; batch chưa thể chạy.")
        if v4_options.get("review_file"):
            raise UserError(
                "--review-file chỉ áp dụng cho một ảnh. Với thư mục, chạy lượt đầu "
                "--review defer rồi dùng .\\upscale review <bundle> cho từng kết quả."
            )
        if v4_options.get("review") == "gui":
            v4_options = dict(v4_options)
            v4_options["review"] = "defer"
    elif mode == "V5_LAYERS":
        if not PYTHON_V5.is_file() or not V5_ENGINE.is_file():
            raise UserError("Thiếu Python CUDA hoặc engine V5; batch chưa thể chạy.")
        if v4_options.get("review") == "gui":
            v4_options = dict(v4_options)
            v4_options["review"] = "defer"
    elif mode in {"V4_PRINT", "V4_VECTOR"}:
        if not PYTHON_V4.is_file() or not V4_ENGINE.is_file():
            raise UserError("Thiếu Python hoặc engine V4 trong APP; batch chưa thể chạy.")
    elif not PYTHON.is_file() or not V2_ENGINE.is_file() or not V3_ENGINE.is_file():
        raise UserError("Thiếu Python hoặc engine V2/V3 trong APP; batch chưa thể chạy.")
    images = collect_batch_images(directory)
    tag = scale_tag(scale)
    batch_group = (
        select_v7_batch_group(directory, tag)
        if mode == "V7_REPAIR"
        else Path(f"{safe_folder_name(directory.name)}_x{tag}")
    )
    entries = batch_entries(directory, images)
    batch_output = OUTPUT_DIR / mode / batch_group
    succeeded: list[Path] = []
    failed: list[tuple[Path, str]] = []
    started = time.perf_counter()

    print("\nCHẠY CẢ THƯ MỤC")
    print(f"  Nguồn      : {directory}")
    print(f"  Số ảnh     : {len(entries)}")
    print(f"  Cùng hệ số : x{scale:g}")
    print(f"  Kết quả    : {batch_output}")
    print("  Cách chạy  : tuần tự từng ảnh để GPU không bị tranh VRAM\n")

    for index, (source, relative_parent, result_stem) in enumerate(entries, 1):
        print(f"\n{'=' * 72}")
        print(f"ẢNH {index}/{len(entries)}: {source.relative_to(directory)}")
        print(f"{'=' * 72}", flush=True)
        try:
            result = run_job(
                mode,
                source,
                scale,
                allow_huge,
                v4_options,
                output_subdir=batch_group / relative_parent,
                output_stem=result_stem,
                batch_root=directory,
            )
            succeeded.append(result)
        except KeyboardInterrupt:
            raise
        except (UserError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
            failed.append((source, str(exc)))
            print(f"\nBỎ QUA ẢNH LỖI: {source.name}: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 72)
    print("TỔNG KẾT THƯ MỤC")
    print(f"  Thành công : {len(succeeded)}/{len(entries)}")
    print(f"  Thất bại   : {len(failed)}")
    print(f"  Thời gian  : {elapsed:.1f} giây")
    print(f"  Mở tại     : {batch_output}")
    if failed:
        print("\nCác ảnh lỗi:")
        for source, error in failed:
            print(f"  - {source}: {error}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_console()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0].casefold() in {"review", "duyet"}:
        if len(arguments) != 2:
            raise UserError(
                "Cú pháp: .\\upscale review <bundle-hoặc-file-duyệt.json>"
            )
        return run_review_command(arguments[1])
    mode, file_token, scale, allow_huge, v4_options = parse_command(arguments)
    source = resolve_source(file_token)
    with gpu_job_lock():
        if source.is_dir():
            return run_batch(mode, source, scale, allow_huge, v4_options)
        run_job(mode, source, scale, allow_huge, v4_options)
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserError as exc:
        print(f"\nLỖI: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except subprocess.CalledProcessError as exc:
        print(f"\nLỖI: engine dừng với mã {exc.returncode}. Kết quả cũ (nếu có) vẫn được giữ.", file=sys.stderr)
        raise SystemExit(exc.returncode or 1) from exc
    except OSError as exc:
        print(f"\nLỖI HỆ THỐNG/Ổ ĐĨA: {exc}. Batch đã dừng để tránh lỗi lặp lại.", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\nĐã dừng theo yêu cầu. Thư mục tạm của lệnh này sẽ được dọn.", file=sys.stderr)
        raise SystemExit(130)
