"""Single public command for the local V2 FAST and V3 HIGH upscalers."""

from __future__ import annotations

import contextlib
import hashlib
import json
import msvcrt
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError


Image.MAX_IMAGE_PIXELS = 500_000_000
MIN_SCALE = 2.0
MAX_SCALE = 20.0
MAX_MEGAPIXELS = 500.0
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
DIRECT_ENGINE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
INPUT_DIR = ROOT_DIR / "INPUT"
OUTPUT_DIR = ROOT_DIR / "OUTPUT"
WORK_DIR = APP_DIR / "work"
PYTHON = APP_DIR / "engines" / "V3" / ".venv" / "Scripts" / "python.exe"
V2_ENGINE = APP_DIR / "engines" / "V2" / "upsize_ai_v2.py"
V3_ENGINE = APP_DIR / "engines" / "V3" / "upsize_ai_v3_master.py"
VERSION_FILE = ROOT_DIR / "VERSION"


def read_app_version() -> str:
    try:
        value = VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "dev"
    return value or "dev"


APP_VERSION = read_app_version()


class UserError(RuntimeError):
    """Expected command/input error with a short user-facing message."""


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

Đường dẫn là thư mục thì chương trình chạy tất cả ảnh trong thư mục và
các thư mục con với cùng một hệ số n.

Ví dụ:
   .\\upscale poster.png 4
   .\\upscale "bang quang cao.jpg" 10
   .\\upscale high "bang quang cao.jpg" 10
   .\\upscale "D:\\BO ANH" 10
   .\\upscale high "D:\\BO ANH" 10

Kết quả V2: {OUTPUT_DIR / 'V2_FAST'}
Kết quả V3: {OUTPUT_DIR / 'V3_HIGH'}

n là hệ số chiều rộng và chiều cao, từ 2 đến 20.
Khuyên dùng n=4 hoặc n=10. Cùng một lệnh chạy lại sẽ thay kết quả cũ
một cách an toàn sau khi file mới đã render và kiểm tra xong.
""".strip()
    )


def parse_command(argv: list[str]) -> tuple[str, str, float, bool]:
    if not argv or any(value.lower() in {"help", "-h", "--help", "/?"} for value in argv):
        print_help()
        raise SystemExit(0)

    allow_huge = False
    positional: list[str] = []
    for value in argv:
        lowered = value.lower()
        if lowered == "--allow-huge":
            allow_huge = True
        elif value.startswith("--"):
            raise UserError(f"Tùy chọn không hỗ trợ: {value}")
        else:
            positional.append(value)

    mode = "V2_FAST"
    if positional and positional[0].lower() in {"high", "v3"}:
        mode = "V3_HIGH"
        positional.pop(0)
    elif positional and positional[0].lower() in {"fast", "v2"}:
        positional.pop(0)

    if len(positional) != 2:
        raise UserError("Sai cú pháp. Gõ .\\upscale để xem ví dụ.")
    file_token, scale_token = positional
    try:
        scale = float(scale_token.replace(",", "."))
    except ValueError as exc:
        raise UserError(f"Hệ số không hợp lệ: {scale_token}") from exc
    if not (MIN_SCALE <= scale <= MAX_SCALE):
        raise UserError(
            f"Hệ số phải từ x{MIN_SCALE:g} đến x{MAX_SCALE:g}. "
            "x100 tạo lượng pixel quá lớn và không làm ảnh có thêm chi tiết thật."
        )
    return mode, file_token, scale, allow_huge


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
            image.load()
            return image.size, image.mode, image.info.get("icc_profile")
    except (UnidentifiedImageError, OSError) as exc:
        raise UserError(f"Không mở được ảnh: {path.name}") from exc


def stage_input(source: Path, job_dir: Path, icc_profile: bytes | None) -> Path:
    suffix = source.suffix.lower()
    if suffix in DIRECT_ENGINE_EXTENSIONS:
        staged = job_dir / f"input{suffix}"
        shutil.copy2(source, staged)
        return staged

    staged = job_dir / "input.png"
    with Image.open(source) as image:
        image.seek(0)
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        save_options = {"icc_profile": icc_profile} if icc_profile else {}
        image.save(staged, format="PNG", **save_options)
    return staged


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


def run_job(
    mode: str,
    source: Path,
    scale: float,
    allow_huge: bool,
    *,
    output_subdir: Path | None = None,
    output_stem: str | None = None,
    batch_root: Path | None = None,
) -> Path:
    if not PYTHON.is_file():
        raise UserError(f"Thiếu Python nội bộ: {PYTHON}")
    if not V2_ENGINE.is_file() or not V3_ENGINE.is_file():
        raise UserError("Thiếu engine V2 hoặc V3 trong APP.")

    source_size, source_mode, icc_profile = inspect_image(source)
    final_size = tuple(int(round(value * scale)) for value in source_size)
    megapixels = final_size[0] * final_size[1] / 1_000_000
    if megapixels > MAX_MEGAPIXELS and not allow_huge:
        raise UserError(
            f"Kết quả sẽ là {final_size[0]}x{final_size[1]} ({megapixels:.1f} MP). "
            "Nếu đã tính đúng và máy in chấp nhận, thêm --allow-huge."
        )

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
                "normalized_stage_sha256": sha256_file(staged_input),
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


def run_batch(mode: str, directory: Path, scale: float, allow_huge: bool) -> int:
    if not PYTHON.is_file() or not V2_ENGINE.is_file() or not V3_ENGINE.is_file():
        raise UserError("Thiếu Python hoặc engine V2/V3 trong APP; batch chưa thể chạy.")
    images = collect_batch_images(directory)
    tag = scale_tag(scale)
    batch_group = Path(f"{safe_folder_name(directory.name)}_x{tag}")
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
    mode, file_token, scale, allow_huge = parse_command(list(sys.argv[1:] if argv is None else argv))
    source = resolve_source(file_token)
    with gpu_job_lock():
        if source.is_dir():
            return run_batch(mode, source, scale, allow_huge)
        run_job(mode, source, scale, allow_huge)
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
