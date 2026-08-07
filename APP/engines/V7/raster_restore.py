"""Conservative raster restoration with an optional local V3 CUDA SR backend.

This module is deliberately independent from the V7 text-repair entry point.
It restores only evidence already present in the raster through bounded
deconvolution, denoising and observed high-frequency enhancement.  Optional
super-resolution is delegated to the existing V3 runtime/checkpoints, which
keeps Torch/CUDA and PaddleOCR in separate Python environments.

Neural SR predicts a plausible high-resolution raster.  It is not evidence of
details that were absent from the source, and every result report says so.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = 500_000_000
ImageInput = Image.Image | np.ndarray
Progress = Callable[[str], None]
LUMA_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
SPATIAL_SHARPNESS_GRID = 4
SPATIAL_MAX_MEASURE_TILE_SIDE = 512
SPATIAL_MIN_SOURCE_TILE_SIDE = 16
SPATIAL_MIN_SOURCE_TILE_AREA = 256
SPATIAL_EDGE_MAGNITUDE_FLOOR = 2.0 / 255.0
SPATIAL_NOISE_EDGE_MULTIPLIER = 3.0
SPATIAL_BASELINE_EDGE_SCORE_FLOOR = 0.006
SPATIAL_SOURCE_DYNAMIC_RANGE_FLOOR = 4.0 / 255.0
SPATIAL_MIN_EDGE_PIXEL_FRACTION = 0.005
SPATIAL_MIN_EDGE_PIXEL_COUNT = 12
SPATIAL_SOFTENED_RATIO = 0.90
SPATIAL_SHARPENED_RATIO = 1.05
SPATIAL_MIN_ELIGIBLE_TILES = 4
SPATIAL_MIN_NON_SOFTENED_COVERAGE = 0.75
SPATIAL_MIN_MEDIAN_RATIO = 0.95
SPATIAL_MIN_P10_RATIO = 0.70
V3_MODEL_FILES: Mapping[str, str] = {
    "hat-sharper": "Real_HAT_GAN_sharper.pth",
    "swin2sr-fidelity": "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth",
    "realesrgan-detail": "RealESRGAN_x4plus.pth",
}
TRUTH_NOTICE = (
    "Deblur, denoise and detail stages are bounded transformations of observed source pixels. "
    "Neural super-resolution predicts plausible pixels; it cannot prove or recover unknowable "
    "detail that is absent from the source."
)


class RasterRestoreError(RuntimeError):
    """Raised for an invalid contract or an explicitly strict SR failure."""


@dataclass(frozen=True, slots=True)
class RasterRestoreConfig:
    """Configuration for :func:`restore_raster`.

    The 512/128 default is the already-tested V3 setting for a 12 GB RTX 3060.
    V3 owns its additional CUDA OOM fallback to progressively smaller tiles.
    """

    scale: float = 4.0
    enable_deblur: bool = True
    enable_denoise: bool = True
    enable_detail: bool = True
    enable_sr: bool = True
    enable_sr_x1: bool = False
    deblur_strength: float = 0.22
    deblur_sigma: float = 1.0
    deblur_iterations: int = 3
    denoise_strength: float = 0.55
    noise_trigger: float = 0.006
    detail_strength: float = 0.24
    detail_sigma: float = 0.85
    tile: int = 512
    overlap: int = 128
    dtype: str = "auto"
    model_key: str = "swin2sr-fidelity"
    strict_sr: bool = False
    stage_max_mean_delta: float = 0.045
    stage_max_excursion_p99: float = 0.09
    stage_max_clip_increase: float = 0.025
    sr_max_roundtrip_mae: float = 0.09
    sr_max_sharpness_ratio: float = 5.0
    sr_max_noise_ratio: float = 5.0
    sr_max_excursion_p99: float = 0.20
    sr_max_clip_increase: float = 0.08

    def __post_init__(self) -> None:
        finite = {
            "scale": self.scale,
            "deblur_strength": self.deblur_strength,
            "deblur_sigma": self.deblur_sigma,
            "denoise_strength": self.denoise_strength,
            "noise_trigger": self.noise_trigger,
            "detail_strength": self.detail_strength,
            "detail_sigma": self.detail_sigma,
            "stage_max_mean_delta": self.stage_max_mean_delta,
            "stage_max_excursion_p99": self.stage_max_excursion_p99,
            "stage_max_clip_increase": self.stage_max_clip_increase,
            "sr_max_roundtrip_mae": self.sr_max_roundtrip_mae,
            "sr_max_sharpness_ratio": self.sr_max_sharpness_ratio,
            "sr_max_noise_ratio": self.sr_max_noise_ratio,
            "sr_max_excursion_p99": self.sr_max_excursion_p99,
            "sr_max_clip_increase": self.sr_max_clip_increase,
        }
        if any(not math.isfinite(float(value)) for value in finite.values()):
            raise ValueError("raster restoration numeric settings must be finite")
        if not 1.0 <= float(self.scale) <= 20.0:
            raise ValueError("scale must be from x1 through x20")
        if not 0.0 <= self.deblur_strength <= 1.0:
            raise ValueError("deblur_strength must be from 0 through 1")
        if not 0.0 <= self.denoise_strength <= 1.0:
            raise ValueError("denoise_strength must be from 0 through 1")
        if not 0.0 <= self.detail_strength <= 1.0:
            raise ValueError("detail_strength must be from 0 through 1")
        if self.deblur_sigma <= 0 or self.detail_sigma <= 0:
            raise ValueError("deblur/detail sigma must be positive")
        if not 1 <= int(self.deblur_iterations) <= 8:
            raise ValueError("deblur_iterations must be from 1 through 8")
        if int(self.tile) != self.tile or int(self.overlap) != self.overlap:
            raise ValueError("tile and overlap must be integers")
        if self.tile < 128 or self.overlap < 8 or self.overlap * 2 > self.tile:
            raise ValueError("tile must be >=128 and overlap must be 8..tile/2")
        if self.dtype not in {"auto", "fp16", "bf16", "fp32"}:
            raise ValueError("dtype must be auto, fp16, bf16 or fp32")
        if self.model_key not in V3_MODEL_FILES:
            raise ValueError(f"unknown V3 model key: {self.model_key}")
        positive_limits = (
            self.stage_max_mean_delta,
            self.stage_max_excursion_p99,
            self.sr_max_roundtrip_mae,
            self.sr_max_sharpness_ratio,
            self.sr_max_noise_ratio,
            self.sr_max_excursion_p99,
        )
        if any(value <= 0 for value in positive_limits):
            raise ValueError("quality guardrail limits must be positive")


@dataclass(frozen=True, slots=True)
class SRInferenceResult:
    image: ImageInput
    report: Mapping[str, object] = field(default_factory=dict)


class SuperResolutionBackend(Protocol):
    """Minimal backend contract used by the restoration pipeline."""

    native_scale: int
    name: str

    def upscale(
        self,
        image: Image.Image,
        *,
        model_key: str,
        tile: int,
        overlap: int,
        dtype: str,
        progress: Progress,
    ) -> SRInferenceResult:
        """Return one native-scale RGB prediction and auditable metadata."""


@dataclass(frozen=True, slots=True)
class LocalV3Assets:
    root: Path
    python: Path
    single_engine: Path
    master_engine: Path
    models: Mapping[str, Path]
    expected_sha256: Mapping[str, str]
    verified_sha256: Mapping[str, str]
    missing: tuple[str, ...]

    @property
    def single_available(self) -> bool:
        blockers = {
            item
            for item in self.missing
            if item in {"python", "single_engine"}
            or item.startswith("model:")
            or item.startswith("hash:")
        }
        return self.python.is_file() and self.single_engine.is_file() and not blockers

    @property
    def master_available(self) -> bool:
        return (
            self.single_available
            and self.master_engine.is_file()
            and "master_engine" not in self.missing
        )

    def to_report(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "python": str(self.python),
            "single_engine": str(self.single_engine),
            "master_engine": str(self.master_engine),
            "models": {key: str(value) for key, value in sorted(self.models.items())},
            "expected_sha256": dict(sorted(self.expected_sha256.items())),
            "verified_sha256": dict(sorted(self.verified_sha256.items())),
            "missing": list(self.missing),
            "single_available": self.single_available,
            "master_available": self.master_available,
        }


@dataclass(slots=True)
class RasterRestoreResult:
    image: Image.Image
    report: dict[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _pixel_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def probe_local_v3_assets(
    v3_root: Path | str | None = None,
    *,
    verify_hashes: bool = False,
) -> LocalV3Assets:
    """Inspect local V3 runtime/checkpoints without importing Torch or using a network."""

    root = (
        Path(v3_root).resolve()
        if v3_root is not None
        else (Path(__file__).resolve().parent.parent / "V3").resolve()
    )
    python = root / ".venv" / "Scripts" / "python.exe"
    single_engine = root / "upsize_ai_v3.py"
    master_engine = root / "upsize_ai_v3_master.py"
    model_paths = {key: root / "models" / name for key, name in V3_MODEL_FILES.items()}
    expected_by_filename: dict[str, str] = {}
    manifest = root / "MODEL_SHA256SUMS.txt"
    if manifest.is_file():
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            pieces = line.split(maxsplit=1)
            if len(pieces) == 2:
                expected_by_filename[Path(pieces[1]).name] = pieces[0].upper()
    expected = {
        key: expected_by_filename[V3_MODEL_FILES[key]]
        for key in V3_MODEL_FILES
        if V3_MODEL_FILES[key] in expected_by_filename
    }
    missing: list[str] = []
    for label, path in (
        ("python", python),
        ("single_engine", single_engine),
        ("master_engine", master_engine),
        *((f"model:{key}", path) for key, path in model_paths.items()),
    ):
        if not path.is_file():
            missing.append(label)
    verified: dict[str, str] = {}
    if verify_hashes:
        for key, path in model_paths.items():
            if not path.is_file():
                continue
            actual = _sha256_file(path)
            verified[key] = actual
            wanted = expected.get(key)
            if wanted is None or actual != wanted:
                missing.append(f"hash:{key}")
    return LocalV3Assets(
        root=root,
        python=python,
        single_engine=single_engine,
        master_engine=master_engine,
        models=model_paths,
        expected_sha256=expected,
        verified_sha256=verified,
        missing=tuple(sorted(set(missing))),
    )


@dataclass(slots=True)
class V3SubprocessBackend:
    """Run the already-installed V3 CUDA engine without importing it into V7."""

    assets: LocalV3Assets
    mode: str = "single"
    timeout_seconds: int = 7200
    verify_selected_model: bool = True
    native_scale: int = field(init=False, default=4)
    name: str = field(init=False, default="local-v3-cuda-subprocess")

    @classmethod
    def from_local(
        cls,
        v3_root: Path | str | None = None,
        *,
        mode: str = "single",
        verify_selected_model: bool = True,
        python: Path | str | None = None,
    ) -> "V3SubprocessBackend":
        assets = probe_local_v3_assets(v3_root, verify_hashes=False)
        if python is not None:
            override = Path(python).resolve()
            assets = replace(
                assets,
                python=override,
                missing=tuple(item for item in assets.missing if item != "python"),
            )
        return cls(
            assets,
            mode=mode,
            verify_selected_model=verify_selected_model,
        )

    def __post_init__(self) -> None:
        if self.mode not in {"single", "master"}:
            raise ValueError("V3 backend mode must be single or master")
        if int(self.timeout_seconds) < 1:
            raise ValueError("timeout_seconds must be positive")

    def build_command(
        self,
        source: Path,
        target: Path,
        *,
        model_key: str,
        tile: int,
        overlap: int,
        dtype: str,
    ) -> list[str]:
        if model_key not in V3_MODEL_FILES:
            raise RasterRestoreError(f"unknown V3 model key: {model_key}")
        engine = self.assets.master_engine if self.mode == "master" else self.assets.single_engine
        command = [
            str(self.assets.python),
            "-B",
            str(engine),
            str(source),
            "4",
            str(target),
            "--tile",
            str(tile),
            "--overlap",
            str(overlap),
        ]
        if self.mode == "single":
            command.extend(("--model", model_key, "--dtype", dtype))
        command.append("--force")
        return command

    def _validate_assets(self, model_key: str) -> None:
        required = [self.assets.python]
        required.append(
            self.assets.master_engine if self.mode == "master" else self.assets.single_engine
        )
        if self.mode == "master":
            required.extend(self.assets.models.values())
        else:
            required.append(self.assets.models[model_key])
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RasterRestoreError("local V3 asset is missing: " + ", ".join(missing))
        if self.verify_selected_model:
            keys: Sequence[str] = tuple(V3_MODEL_FILES) if self.mode == "master" else (model_key,)
            for key in keys:
                wanted = self.assets.expected_sha256.get(key)
                if not wanted:
                    raise RasterRestoreError(f"V3 hash manifest has no entry for {key}")
                actual = _sha256_file(self.assets.models[key])
                if actual != wanted:
                    raise RasterRestoreError(f"V3 checkpoint SHA-256 mismatch for {key}")

    def upscale(
        self,
        image: Image.Image,
        *,
        model_key: str,
        tile: int,
        overlap: int,
        dtype: str,
        progress: Progress,
    ) -> SRInferenceResult:
        self._validate_assets(model_key)
        with tempfile.TemporaryDirectory(prefix="v7_raster_sr_") as raw:
            work = Path(raw)
            source = work / "source.png"
            target = work / "native_x4.png"
            image.convert("RGB").save(source, format="PNG", compress_level=1)
            command = self.build_command(
                source,
                target,
                model_key=model_key,
                tile=tile,
                overlap=overlap,
                dtype=dtype,
            )
            progress(
                f"V3 CUDA {self.mode}: model={model_key}, tile={tile}, overlap={overlap}"
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                }
            )
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=environment,
            )
            if completed.returncode != 0:
                tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-12:]
                raise RasterRestoreError(
                    f"V3 CUDA stage failed with exit code {completed.returncode}: "
                    + " | ".join(tail)
                )
            if not target.is_file():
                raise RasterRestoreError("V3 CUDA stage reported success without an output PNG")
            with Image.open(target) as opened:
                opened.load()
                result = opened.convert("RGB").copy()
            expected = (image.width * self.native_scale, image.height * self.native_scale)
            if result.size != expected:
                raise RasterRestoreError(
                    f"V3 native output is {result.size}, expected {expected}"
                )
            manifest_path = target.with_suffix(target.suffix + ".json")
            manifest: dict[str, object] = {}
            if manifest_path.is_file():
                try:
                    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        manifest = parsed
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    manifest = {"warning": "V3 manifest could not be parsed"}
            report = {
                "backend": self.name,
                "mode": self.mode,
                "model_key": model_key,
                "tile": tile,
                "overlap": overlap,
                "dtype": dtype,
                "native_scale": self.native_scale,
                "v3_manifest": manifest,
                "stdout_tail": (completed.stdout or "").splitlines()[-12:],
            }
            return SRInferenceResult(result, report)


def _normalise_input(image: ImageInput) -> tuple[np.ndarray, np.ndarray | None]:
    alpha: np.ndarray | None = None
    if isinstance(image, Image.Image):
        if "A" in image.getbands():
            alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[:, :, 3].copy()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    else:
        value = np.asarray(image)
        if value.dtype != np.uint8:
            raise RasterRestoreError("raster arrays must use uint8 pixels")
        if value.ndim == 2:
            rgb = np.repeat(value[:, :, None], 3, axis=2)
        elif value.ndim == 3 and value.shape[2] in {3, 4}:
            rgb = value[:, :, :3].copy()
            if value.shape[2] == 4:
                alpha = value[:, :, 3].copy()
        else:
            raise RasterRestoreError("raster arrays must be HxW, HxWx3 or HxWx4")
    if rgb.shape[0] < 2 or rgb.shape[1] < 2:
        raise RasterRestoreError("raster input must be at least 2x2 pixels")
    return np.ascontiguousarray(rgb), alpha


def _resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (rgb.shape[1], rgb.shape[0]) == size:
        return rgb.copy()
    return np.asarray(
        Image.fromarray(rgb, "RGB").resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0),
        dtype=np.uint8,
    ).copy()


def _resize_alpha(alpha: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (alpha.shape[1], alpha.shape[0]) == size:
        return alpha.copy()
    return np.asarray(
        Image.fromarray(alpha, "L").resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0),
        dtype=np.uint8,
    ).copy()


def identity_baseline(image: ImageInput, scale: float = 1.0) -> Image.Image:
    """Return a byte-identical x1 image or a single Lanczos resize baseline."""

    if not math.isfinite(float(scale)) or not 1.0 <= float(scale) <= 20.0:
        raise ValueError("scale must be finite and from x1 through x20")
    rgb, alpha = _normalise_input(image)
    size = (int(round(rgb.shape[1] * scale)), int(round(rgb.shape[0] * scale)))
    resized = _resize_rgb(rgb, size)
    if alpha is None:
        return Image.fromarray(resized, "RGB")
    resized_alpha = _resize_alpha(alpha, size)
    return Image.fromarray(np.dstack((resized, resized_alpha)), "RGBA")


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.sum(rgb.astype(np.float32) * (LUMA_WEIGHTS / 255.0), axis=2)


def measure_raster_metrics(image: ImageInput) -> dict[str, float]:
    """Measure scale-local sharpness, noise, halo proxy and clipping."""

    rgb, _alpha = _normalise_input(image)
    y = _luma(rgb)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101) / 8.0
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101) / 8.0
    gradient = np.sqrt(gx * gx + gy * gy)
    # A poster may contain one narrow edge in a large flat field, so an ordinary
    # p90 can be exactly zero.  Mean energy in the strongest five percent keeps
    # sparse glyph/line edges measurable without rewarding every noisy pixel.
    sharp_cutoff = float(np.percentile(gradient, 95))
    sharp_sample = gradient[gradient >= sharp_cutoff]
    sharpness = float(np.mean(sharp_sample)) if len(sharp_sample) else 0.0
    smooth = cv2.GaussianBlur(y, (0, 0), 0.9, borderType=cv2.BORDER_REFLECT_101)
    residual = y - smooth
    flat_limit = float(np.percentile(gradient, 45))
    flat = gradient <= flat_limit
    sample = residual[flat] if np.count_nonzero(flat) >= 32 else residual.ravel()
    centre = float(np.median(sample))
    noise_sigma = float(1.4826 * np.median(np.abs(sample - centre)))
    edge_limit = max(0.01, float(np.percentile(gradient, 82)))
    edge = gradient >= edge_limit
    inner = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    outer = cv2.dilate(edge.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~inner
    high = np.abs(y - cv2.GaussianBlur(y, (0, 0), 1.4, borderType=cv2.BORDER_REFLECT_101))
    halo_score = float(np.percentile(high[outer], 95)) if np.any(outer) else 0.0
    clipped = np.mean((rgb <= 1) | (rgb >= 254))
    dynamic_range = float(np.percentile(y, 99) - np.percentile(y, 1))
    return {
        "sharpness_edge_mean": round(sharpness, 8),
        "noise_sigma": round(noise_sigma, 8),
        "halo_score": round(halo_score, 8),
        "clipped_channel_fraction": round(float(clipped), 8),
        "dynamic_range_p01_p99": round(dynamic_range, 8),
    }


def _replace_luma(rgb: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    source = rgb.astype(np.float32) / 255.0
    source_y = np.sum(source * LUMA_WEIGHTS, axis=2)
    ratio = (target_y + 1e-5) / (source_y + 1e-5)
    output = source * ratio[:, :, None]
    dark = source_y < 1e-4
    if np.any(dark):
        output[dark] = target_y[dark, None]
    return np.clip(np.rint(np.clip(output, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def _deblur_candidate(rgb: np.ndarray, config: RasterRestoreConfig) -> np.ndarray:
    """Bounded Richardson-Lucy luminance deconvolution with a Gaussian PSF."""

    observed = _luma(rgb)
    latent = np.clip(observed.copy(), 1e-4, 1.0)
    for _ in range(int(config.deblur_iterations)):
        blurred = cv2.GaussianBlur(
            latent,
            (0, 0),
            config.deblur_sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        ratio = observed / np.maximum(blurred, 1e-4)
        correction = cv2.GaussianBlur(
            ratio,
            (0, 0),
            config.deblur_sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        latent = np.clip(latent * correction, 0.0, 1.0)
    target = observed + config.deblur_strength * (latent - observed)
    local_min = cv2.erode(observed, np.ones((5, 5), np.uint8)) - 0.045
    local_max = cv2.dilate(observed, np.ones((5, 5), np.uint8)) + 0.045
    target = np.clip(target, np.maximum(0.0, local_min), np.minimum(1.0, local_max))
    return _replace_luma(rgb, target)


def _denoise_candidate(rgb: np.ndarray, config: RasterRestoreConfig, noise: float) -> np.ndarray:
    h = float(np.clip(noise * 255.0 * 1.35, 2.0, 9.0))
    denoised = cv2.fastNlMeansDenoisingColored(rgb, None, h, h * 0.8, 7, 21)
    mixed = cv2.addWeighted(
        rgb,
        1.0 - config.denoise_strength,
        denoised,
        config.denoise_strength,
        0.0,
    )
    return np.asarray(mixed, dtype=np.uint8)


def _detail_candidate(rgb: np.ndarray, config: RasterRestoreConfig, noise: float) -> np.ndarray:
    y = _luma(rgb)
    smooth = cv2.GaussianBlur(
        y,
        (0, 0),
        config.detail_sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    high = np.clip(y - smooth, -0.045, 0.045)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient = np.sqrt(gx * gx + gy * gy)
    lower = max(noise * 2.5, float(np.percentile(gradient, 55)))
    upper = max(lower + 1e-5, float(np.percentile(gradient, 92)))
    edge_weight = np.clip((gradient - lower) / (upper - lower), 0.0, 1.0)
    target = np.clip(y + config.detail_strength * high * edge_weight, 0.0, 1.0)
    return _replace_luma(rgb, target)


def _safe_ratio(after: float, before: float, *, floor: float = 0.0) -> float:
    denominator = max(before, floor)
    if denominator <= 1e-9:
        return 1.0 if after <= 1e-9 else float("inf")
    return after / denominator


def _comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    before = measure_raster_metrics(reference)
    after = measure_raster_metrics(candidate)
    reference_y = _luma(reference)
    candidate_y = _luma(candidate)
    local_min = cv2.erode(reference_y, np.ones((5, 5), np.uint8))
    local_max = cv2.dilate(reference_y, np.ones((5, 5), np.uint8))
    excursion = np.maximum(candidate_y - local_max, local_min - candidate_y)
    excursion = np.maximum(excursion, 0.0)
    absolute = np.abs(candidate.astype(np.float32) - reference.astype(np.float32)) / 255.0
    # A fidelity SR model often maps an already-near-white paper/background
    # (for example code values 248..253) to legal white 255.  Counting that as
    # newly destroyed highlight detail rejects clean catalog/poster results.
    # The destructive case is a channel with observed midtone information that
    # the candidate collapses to either endpoint.  Keep the legacy aggregate
    # clipping delta below for diagnostics, but quality gates use this
    # source-aware measure instead.
    reference_midtone = (reference >= 8) & (reference <= 247)
    candidate_clipped = (candidate <= 1) | (candidate >= 254)
    newly_clipped_midtone = reference_midtone & candidate_clipped
    midtone_count = int(np.count_nonzero(reference_midtone))
    return {
        "sharpness_ratio": round(
            _safe_ratio(after["sharpness_edge_mean"], before["sharpness_edge_mean"]), 8
        ),
        # Sub-byte model texture on a mathematically flat Lanczos baseline must
        # not become an infinite ratio.  1/1024 luma is below one 8-bit code
        # value yet still lets meaningful noise growth trigger the guard.
        "noise_ratio": round(
            _safe_ratio(
                after["noise_sigma"],
                before["noise_sigma"],
                floor=1.0 / 1024.0,
            ),
            8,
        ),
        "mean_abs_delta": round(float(np.mean(absolute)), 8),
        "p99_abs_delta": round(float(np.percentile(absolute, 99)), 8),
        "local_excursion_p99": round(float(np.percentile(excursion, 99)), 8),
        "local_excursion_fraction_2pct": round(float(np.mean(excursion > 0.02)), 8),
        "clip_fraction_increase": round(
            max(
                0.0,
                after["clipped_channel_fraction"] - before["clipped_channel_fraction"],
            ),
            8,
        ),
        "newly_clipped_midtone_fraction": round(
            float(np.mean(newly_clipped_midtone)),
            8,
        ),
        "newly_clipped_midtone_rate": round(
            float(np.count_nonzero(newly_clipped_midtone) / max(midtone_count, 1)),
            8,
        ),
        "reference_midtone_channel_fraction": round(
            float(np.mean(reference_midtone)),
            8,
        ),
        "halo_score_increase": round(
            max(0.0, after["halo_score"] - before["halo_score"]), 8
        ),
    }


def _spatial_tile_analysis(
    rgb: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    """Return noise-aware structural gradients and observed-edge support."""

    height, width = rgb.shape[:2]
    empty = np.zeros((height, width), dtype=np.float32)
    if min(height, width) < SPATIAL_MIN_SOURCE_TILE_SIDE:
        return (
            {
                "structural_sharpness_score": 0.0,
                "noise_sigma": 0.0,
                "edge_threshold": round(SPATIAL_EDGE_MAGNITUDE_FLOOR, 8),
                "edge_pixel_fraction": 0.0,
                "edge_pixel_count": 0,
                "minimum_edge_pixel_count": SPATIAL_MIN_EDGE_PIXEL_COUNT,
                "dynamic_range_p05_p95": 0.0,
                "measured_width": width,
                "measured_height": height,
            },
            empty,
            empty.astype(bool),
        )
    measured = rgb
    if max(height, width) > SPATIAL_MAX_MEASURE_TILE_SIDE:
        shrink = SPATIAL_MAX_MEASURE_TILE_SIDE / max(height, width)
        measured = cv2.resize(
            rgb,
            (
                max(SPATIAL_MIN_SOURCE_TILE_SIDE, int(round(width * shrink))),
                max(SPATIAL_MIN_SOURCE_TILE_SIDE, int(round(height * shrink))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    y = _luma(measured)
    noise_smooth = cv2.GaussianBlur(
        y,
        (0, 0),
        0.9,
        borderType=cv2.BORDER_REFLECT_101,
    )
    residual = y - noise_smooth
    structure = cv2.GaussianBlur(
        y,
        (0, 0),
        0.8,
        borderType=cv2.BORDER_REFLECT_101,
    )
    gx = cv2.Sobel(
        structure,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
        borderType=cv2.BORDER_REFLECT_101,
    ) / 8.0
    gy = cv2.Sobel(
        structure,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
        borderType=cv2.BORDER_REFLECT_101,
    ) / 8.0
    gradient = np.sqrt(gx * gx + gy * gy)
    # Estimate noise away from structural edges so glyph strokes do not inflate
    # the threshold that is meant to distinguish them from random texture.
    flat_limit = float(np.percentile(gradient, 45))
    flat = gradient <= flat_limit
    noise_sample = residual[flat] if np.count_nonzero(flat) >= 32 else residual.ravel()
    noise_centre = float(np.median(noise_sample))
    noise_sigma = float(1.4826 * np.median(np.abs(noise_sample - noise_centre)))
    edge_threshold = max(
        SPATIAL_EDGE_MAGNITUDE_FLOOR,
        SPATIAL_NOISE_EDGE_MULTIPLIER * noise_sigma,
    )
    support = gradient > edge_threshold
    # Grid borders are measurement boundaries, not observed image edges.
    support[:2, :] = False
    support[-2:, :] = False
    support[:, :2] = False
    support[:, -2:] = False
    inner_gradient = gradient[2:-2, 2:-2]
    strongest_count = max(1, int(math.ceil(inner_gradient.size * 0.10)))
    strongest = np.partition(
        inner_gradient.ravel(),
        inner_gradient.size - strongest_count,
    )[-strongest_count:]
    minimum_edge_count = max(
        SPATIAL_MIN_EDGE_PIXEL_COUNT,
        int(math.ceil(inner_gradient.size * SPATIAL_MIN_EDGE_PIXEL_FRACTION)),
    )
    edge_count = int(np.count_nonzero(support))
    return (
        {
            "structural_sharpness_score": round(float(np.mean(strongest)), 8),
            "noise_sigma": round(noise_sigma, 8),
            "edge_threshold": round(edge_threshold, 8),
            "edge_pixel_fraction": round(
                float(edge_count / max(inner_gradient.size, 1)),
                8,
            ),
            "edge_pixel_count": edge_count,
            "minimum_edge_pixel_count": minimum_edge_count,
            "dynamic_range_p05_p95": round(
                float(np.percentile(y, 95) - np.percentile(y, 5)),
                8,
            ),
            "measured_width": int(measured.shape[1]),
            "measured_height": int(measured.shape[0]),
        },
        gradient,
        support,
    )


def _winsorized_mean(values: np.ndarray, proportion: float = 0.10) -> float:
    flat = np.asarray(values, dtype=np.float32).ravel()
    if flat.size == 0:
        return 0.0
    if flat.size < 4:
        return float(np.mean(flat))
    lower, upper = np.quantile(flat, (proportion, 1.0 - proportion))
    return float(np.mean(np.clip(flat, lower, upper)))


def _grid_bounds(length: int, index: int) -> tuple[int, int]:
    start = int(round(index * length / SPATIAL_SHARPNESS_GRID))
    end = int(round((index + 1) * length / SPATIAL_SHARPNESS_GRID))
    return start, end


def _spatial_sharpness_coverage(
    source: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    """Measure whether observed edge detail is preserved across a 4x4 grid.

    Eligibility comes from the source raster, while the sharpness ratio compares
    the candidate with the same-size Lanczos working baseline.  Flat or tiny
    source tiles therefore neither reward nor penalise a neural prediction.
    """

    thresholds: dict[str, object] = {
        "grid": [SPATIAL_SHARPNESS_GRID, SPATIAL_SHARPNESS_GRID],
        "maximum_measurement_tile_side_px": SPATIAL_MAX_MEASURE_TILE_SIDE,
        "minimum_source_tile_side_px": SPATIAL_MIN_SOURCE_TILE_SIDE,
        "minimum_source_tile_area_px": SPATIAL_MIN_SOURCE_TILE_AREA,
        "edge_magnitude_floor": SPATIAL_EDGE_MAGNITUDE_FLOOR,
        "noise_edge_multiplier": SPATIAL_NOISE_EDGE_MULTIPLIER,
        "baseline_edge_score_floor": SPATIAL_BASELINE_EDGE_SCORE_FLOOR,
        "source_dynamic_range_floor": SPATIAL_SOURCE_DYNAMIC_RANGE_FLOOR,
        "minimum_edge_pixel_fraction": SPATIAL_MIN_EDGE_PIXEL_FRACTION,
        "minimum_edge_pixel_count": SPATIAL_MIN_EDGE_PIXEL_COUNT,
        "preserved_at_or_above_ratio": SPATIAL_SOFTENED_RATIO,
        "softened_below_ratio": SPATIAL_SOFTENED_RATIO,
        "sharpened_at_or_above_ratio": SPATIAL_SHARPENED_RATIO,
        "minimum_eligible_tiles_for_rejection": SPATIAL_MIN_ELIGIBLE_TILES,
        "minimum_non_softened_coverage": SPATIAL_MIN_NON_SOFTENED_COVERAGE,
        "minimum_median_ratio": SPATIAL_MIN_MEDIAN_RATIO,
        "minimum_p10_ratio": SPATIAL_MIN_P10_RATIO,
        "paired_support_policy": (
            "noise-qualified source edge positions mapped to output; 3x3 max-filtered "
            "baseline/candidate gradients; 10% winsorized mean"
        ),
    }
    tiles: list[dict[str, object]] = []
    ratios: list[float] = []
    for row in range(SPATIAL_SHARPNESS_GRID):
        sy0, sy1 = _grid_bounds(source.shape[0], row)
        by0, by1 = _grid_bounds(baseline.shape[0], row)
        for column in range(SPATIAL_SHARPNESS_GRID):
            sx0, sx1 = _grid_bounds(source.shape[1], column)
            bx0, bx1 = _grid_bounds(baseline.shape[1], column)
            source_tile = source[sy0:sy1, sx0:sx1]
            baseline_tile = baseline[by0:by1, bx0:bx1]
            candidate_tile = candidate[by0:by1, bx0:bx1]
            source_stats, _source_gradient, source_support = _spatial_tile_analysis(
                source_tile
            )
            baseline_stats, baseline_gradient, _baseline_support = (
                _spatial_tile_analysis(baseline_tile)
            )
            candidate_stats, candidate_gradient, _candidate_support = (
                _spatial_tile_analysis(candidate_tile)
            )
            if source_support.size and baseline_gradient.size:
                mapped_source_support = cv2.resize(
                    source_support.astype(np.uint8),
                    (baseline_gradient.shape[1], baseline_gradient.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                kernel = np.ones((3, 3), np.uint8)
                baseline_aligned = cv2.dilate(baseline_gradient, kernel)
                candidate_aligned = cv2.dilate(candidate_gradient, kernel)
            else:
                mapped_source_support = np.zeros_like(baseline_gradient, dtype=bool)
                baseline_aligned = baseline_gradient
                candidate_aligned = candidate_gradient
            support_pixel_count = int(np.count_nonzero(mapped_source_support))
            baseline_support_score = _winsorized_mean(
                baseline_aligned[mapped_source_support]
            )
            candidate_support_score = _winsorized_mean(
                candidate_aligned[mapped_source_support]
            )
            baseline_stats["source_position_support_score"] = round(
                baseline_support_score,
                8,
            )
            candidate_stats["source_position_support_score"] = round(
                candidate_support_score,
                8,
            )
            source_height, source_width = source_tile.shape[:2]
            eligibility_reasons: list[str] = []
            if min(source_width, source_height) < SPATIAL_MIN_SOURCE_TILE_SIDE:
                eligibility_reasons.append("source_tile_too_small")
            if source_width * source_height < SPATIAL_MIN_SOURCE_TILE_AREA:
                eligibility_reasons.append("source_tile_area_too_small")
            if (
                float(source_stats["dynamic_range_p05_p95"])
                < SPATIAL_SOURCE_DYNAMIC_RANGE_FLOOR
            ):
                eligibility_reasons.append("source_dynamic_range_too_low")
            if (
                int(source_stats["edge_pixel_count"])
                < int(source_stats["minimum_edge_pixel_count"])
            ):
                eligibility_reasons.append("too_few_observed_edge_pixels")
            if support_pixel_count == 0:
                eligibility_reasons.append("no_mapped_source_edge_support")
            if baseline_support_score < SPATIAL_BASELINE_EDGE_SCORE_FLOOR:
                eligibility_reasons.append("baseline_edge_energy_too_low")
            eligible = not eligibility_reasons
            ratio: float | None = None
            classification = "ineligible"
            if eligible:
                ratio = round(
                    _safe_ratio(
                        candidate_support_score,
                        baseline_support_score,
                        floor=SPATIAL_BASELINE_EDGE_SCORE_FLOOR,
                    ),
                    8,
                )
                ratios.append(ratio)
                if ratio < SPATIAL_SOFTENED_RATIO:
                    classification = "softened"
                elif ratio >= SPATIAL_SHARPENED_RATIO:
                    classification = "sharpened"
                else:
                    classification = "preserved"
            tiles.append(
                {
                    "row": row,
                    "column": column,
                    "source_bounds": [sx0, sy0, sx1, sy1],
                    "output_bounds": [bx0, by0, bx1, by1],
                    "eligible": eligible,
                    "eligibility_reasons": eligibility_reasons,
                    "source_support_pixel_count": int(
                        source_stats["edge_pixel_count"]
                    ),
                    "mapped_support_pixel_count": support_pixel_count,
                    "source": source_stats,
                    "baseline": baseline_stats,
                    "candidate": candidate_stats,
                    "sharpness_ratio": ratio,
                    "classification": classification,
                }
            )
    eligible_count = len(ratios)
    softened_count = sum(value < SPATIAL_SOFTENED_RATIO for value in ratios)
    sharpened_count = sum(value >= SPATIAL_SHARPENED_RATIO for value in ratios)
    non_softened_count = eligible_count - softened_count
    applicable = eligible_count >= SPATIAL_MIN_ELIGIBLE_TILES
    non_softened_coverage = (
        non_softened_count / eligible_count if eligible_count else 1.0
    )
    sharpened_coverage = sharpened_count / eligible_count if eligible_count else 0.0
    softened_coverage = softened_count / eligible_count if eligible_count else 0.0
    median_ratio = float(np.median(ratios)) if ratios else None
    p10_ratio = float(np.percentile(ratios, 10)) if ratios else None
    failure_reasons: list[str] = []
    if applicable:
        if non_softened_coverage < SPATIAL_MIN_NON_SOFTENED_COVERAGE:
            failure_reasons.append("non_softened_coverage_below_threshold")
        if median_ratio is not None and median_ratio < SPATIAL_MIN_MEDIAN_RATIO:
            failure_reasons.append("median_spatial_ratio_below_threshold")
        if p10_ratio is not None and p10_ratio < SPATIAL_MIN_P10_RATIO:
            failure_reasons.append("p10_spatial_ratio_below_threshold")
    passed = not failure_reasons
    if not applicable:
        reason = "insufficient_observed_edge_tiles_for_spatial_rejection"
    elif passed:
        reason = "observed_edge_coverage_preserved"
    else:
        reason = "material_softening_across_spatially_distributed_observed_edges"
    ignored_by_reason: dict[str, int] = {}
    for tile in tiles:
        if bool(tile["eligible"]):
            continue
        for item in tile["eligibility_reasons"]:  # type: ignore[union-attr]
            key = str(item)
            ignored_by_reason[key] = ignored_by_reason.get(key, 0) + 1
    return {
        "passed": passed,
        "applicable": applicable,
        "reason": reason,
        "failure_reasons": failure_reasons,
        "eligible_tile_count": eligible_count,
        "ineligible_tile_count": len(tiles) - eligible_count,
        "ignored_by_reason": dict(sorted(ignored_by_reason.items())),
        "non_softened_tile_count": non_softened_count,
        "softened_tile_count": softened_count,
        "sharpened_tile_count": sharpened_count,
        "non_softened_coverage": round(non_softened_coverage, 8),
        "softened_coverage": round(softened_coverage, 8),
        "sharpened_coverage": round(sharpened_coverage, 8),
        "median_ratio": round(median_ratio, 8) if median_ratio is not None else None,
        "p10_ratio": round(p10_ratio, 8) if p10_ratio is not None else None,
        "thresholds": thresholds,
        "tiles": tiles,
    }


def _guard_stage(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    config: RasterRestoreConfig,
) -> tuple[bool, dict[str, object]]:
    if candidate.dtype != np.uint8 or candidate.shape != reference.shape:
        return False, {"passed": False, "reasons": ["invalid_stage_output"]}
    comparison = _comparison(reference, candidate)
    reasons: list[str] = []
    if comparison["p99_abs_delta"] <= 1.0 / 255.0:
        reasons.append("no_material_change")
    if comparison["mean_abs_delta"] > config.stage_max_mean_delta:
        reasons.append("mean_change_too_large")
    if comparison["local_excursion_p99"] > config.stage_max_excursion_p99:
        reasons.append("halo_or_overshoot_excursion")
    if comparison["newly_clipped_midtone_fraction"] > config.stage_max_clip_increase:
        reasons.append("midtone_information_clipped")
    sharpness_ratio = comparison["sharpness_ratio"]
    noise_ratio = comparison["noise_ratio"]
    if name == "deblur":
        if sharpness_ratio < 0.98:
            reasons.append("sharpness_did_not_improve")
        if sharpness_ratio > 2.2:
            reasons.append("sharpness_gain_implausibly_high")
        if noise_ratio > 1.8:
            reasons.append("noise_amplified")
    elif name == "denoise":
        if noise_ratio > 1.03:
            reasons.append("noise_did_not_decrease")
        if sharpness_ratio < 0.70:
            reasons.append("edge_detail_lost")
    elif name == "detail":
        if sharpness_ratio < 0.98:
            reasons.append("detail_did_not_improve")
        if sharpness_ratio > 1.75:
            reasons.append("detail_gain_implausibly_high")
        if noise_ratio > 1.55:
            reasons.append("noise_amplified")
    passed = not reasons
    return passed, {
        "passed": passed,
        "reasons": reasons,
        "before": measure_raster_metrics(reference),
        "after": measure_raster_metrics(candidate),
        "comparison": comparison,
    }


def _run_classical_stage(
    name: str,
    current: np.ndarray,
    enabled: bool,
    operation: Callable[[], np.ndarray],
    config: RasterRestoreConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    if not enabled:
        return current, {"name": name, "applied": False, "accepted": False, "reason": "disabled"}
    try:
        candidate = operation()
        accepted, guard = _guard_stage(name, current, candidate, config)
    except (cv2.error, MemoryError, RasterRestoreError, ValueError) as exc:
        return current, {
            "name": name,
            "applied": True,
            "accepted": False,
            "reason": "stage_error_fallback",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return (candidate if accepted else current), {
        "name": name,
        "applied": True,
        "accepted": accepted,
        "reason": "guard_pass" if accepted else "guard_rejected_fallback",
        "guard": guard,
    }


def _normalise_backend_result(value: SRInferenceResult | tuple[ImageInput, Mapping[str, object]]) -> SRInferenceResult:
    if isinstance(value, SRInferenceResult):
        return value
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
        return SRInferenceResult(value[0], value[1])
    raise RasterRestoreError("SR backend returned an invalid result contract")


def _guard_sr(
    source: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    config: RasterRestoreConfig,
) -> tuple[bool, dict[str, object]]:
    if candidate.dtype != np.uint8 or candidate.shape != baseline.shape:
        return False, {"passed": False, "reasons": ["invalid_sr_output_shape_or_dtype"]}
    comparison = _comparison(baseline, candidate)
    spatial_coverage = _spatial_sharpness_coverage(source, baseline, candidate)
    roundtrip = _resize_rgb(candidate, (source.shape[1], source.shape[0]))
    roundtrip_mae = float(
        np.mean(np.abs(roundtrip.astype(np.float32) - source.astype(np.float32))) / 255.0
    )
    reasons: list[str] = []
    if comparison["sharpness_ratio"] < 0.70:
        reasons.append("sr_lost_sharpness_vs_lanczos")
    if comparison["sharpness_ratio"] > config.sr_max_sharpness_ratio:
        reasons.append("sr_sharpness_implausibly_high")
    if comparison["noise_ratio"] > config.sr_max_noise_ratio:
        reasons.append("sr_noise_implausibly_high")
    if comparison["local_excursion_p99"] > config.sr_max_excursion_p99:
        reasons.append("sr_halo_or_overshoot_excursion")
    if comparison["newly_clipped_midtone_fraction"] > config.sr_max_clip_increase:
        reasons.append("sr_midtone_information_clipped")
    if not bool(spatial_coverage["passed"]):
        reasons.append("sr_spatial_sharpness_coverage_too_patchy")
    if roundtrip_mae > config.sr_max_roundtrip_mae:
        reasons.append("sr_roundtrip_inconsistent_with_source")
    passed = not reasons
    return passed, {
        "passed": passed,
        "reasons": reasons,
        "baseline_metrics": measure_raster_metrics(baseline),
        "candidate_metrics": measure_raster_metrics(candidate),
        "comparison": comparison,
        "spatial_sharpness_coverage": spatial_coverage,
        "roundtrip_mae": round(roundtrip_mae, 8),
        "roundtrip_limit": config.sr_max_roundtrip_mae,
    }


def restore_raster(
    image: ImageInput,
    config: RasterRestoreConfig | None = None,
    *,
    sr_backend: SuperResolutionBackend | None = None,
    progress: Progress | None = None,
) -> RasterRestoreResult:
    """Restore one raster with stage-level rollback and an SR baseline fallback.

    Classical stages run at source resolution.  A backend, when supplied, must
    return one native-resolution SR image.  Any backend error or QA rejection
    falls back to a single Lanczos resize of the last accepted classical image,
    unless ``strict_sr`` explicitly asks the caller to handle the failure.
    """

    settings = config or RasterRestoreConfig()
    notify = progress or (lambda _message: None)
    original, alpha = _normalise_input(image)
    source_size = (original.shape[1], original.shape[0])
    final_size = (
        int(round(source_size[0] * settings.scale)),
        int(round(source_size[1] * settings.scale)),
    )
    identity = _resize_rgb(original, final_size)
    current = original.copy()
    stages: list[dict[str, object]] = []

    notify("Raster restore 1/4: bounded luminance deblur")
    current, stage = _run_classical_stage(
        "deblur",
        current,
        settings.enable_deblur and settings.deblur_strength > 0,
        lambda: _deblur_candidate(current, settings),
        settings,
    )
    stages.append(stage)

    notify("Raster restore 2/4: noise-aware denoise")
    current_metrics = measure_raster_metrics(current)
    denoise_enabled = (
        settings.enable_denoise
        and settings.denoise_strength > 0
        and current_metrics["noise_sigma"] >= settings.noise_trigger
    )
    current, stage = _run_classical_stage(
        "denoise",
        current,
        denoise_enabled,
        lambda: _denoise_candidate(current, settings, current_metrics["noise_sigma"]),
        settings,
    )
    if settings.enable_denoise and not denoise_enabled:
        stage["reason"] = "noise_below_trigger"
    stages.append(stage)

    notify("Raster restore 3/4: observed-detail enhancement")
    detail_noise = measure_raster_metrics(current)["noise_sigma"]
    current, stage = _run_classical_stage(
        "detail",
        current,
        settings.enable_detail and settings.detail_strength > 0,
        lambda: _detail_candidate(current, settings, detail_noise),
        settings,
    )
    stages.append(stage)

    working_baseline = _resize_rgb(current, final_size)
    sr_report: dict[str, object] = {
        "attempted": False,
        "accepted": False,
        "fallback": "working_lanczos_baseline",
        "tile": settings.tile,
        "overlap": settings.overlap,
        "gpu_memory_policy": (
            "512px tiles with 128px cosine-blended overlap; V3 retries smaller tiles on CUDA OOM"
        ),
    }
    final = working_baseline
    used_neural_prediction = False
    sr_requested = settings.enable_sr and (
        settings.scale > 1.0 or settings.enable_sr_x1
    )
    if sr_requested and sr_backend is not None:
        notify("Raster restore 4/4: local tiled CUDA super-resolution")
        sr_report["attempted"] = True
        try:
            backend_value = sr_backend.upscale(
                Image.fromarray(current, "RGB"),
                model_key=settings.model_key,
                tile=settings.tile,
                overlap=settings.overlap,
                dtype=settings.dtype,
                progress=notify,
            )
            inference = _normalise_backend_result(backend_value)
            native_rgb, _native_alpha = _normalise_input(inference.image)
            native_scale = int(getattr(sr_backend, "native_scale", 4))
            expected_native = (source_size[0] * native_scale, source_size[1] * native_scale)
            if (native_rgb.shape[1], native_rgb.shape[0]) != expected_native:
                raise RasterRestoreError(
                    f"SR backend returned {(native_rgb.shape[1], native_rgb.shape[0])}, "
                    f"expected {expected_native}"
                )
            candidate = _resize_rgb(native_rgb, final_size)
            accepted, guard = _guard_sr(current, working_baseline, candidate, settings)
            sr_report.update(
                {
                    "backend": str(getattr(sr_backend, "name", type(sr_backend).__name__)),
                    "native_scale": native_scale,
                    "backend_report": dict(inference.report),
                    "guard": guard,
                    "accepted": accepted,
                    "fallback": None if accepted else "working_lanczos_baseline",
                }
            )
            if accepted:
                final = candidate
                used_neural_prediction = True
        except Exception as exc:
            if settings.strict_sr:
                raise RasterRestoreError("strict local SR stage failed") from exc
            sr_report.update(
                {
                    "accepted": False,
                    "fallback": "working_lanczos_baseline",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    elif sr_requested:
        sr_report["reason"] = "no_sr_backend_supplied"
    else:
        sr_report["reason"] = "sr_disabled_or_x1"

    accepted_classical = [str(item["name"]) for item in stages if bool(item.get("accepted"))]
    if used_neural_prediction:
        status = "RESTORED_WITH_LOCAL_SR_PREDICTION"
    elif accepted_classical:
        status = "RESTORED_CLASSICAL_LANCZOS_BASELINE"
    elif settings.scale == 1.0:
        status = "IDENTITY"
        final = identity
    else:
        status = "LANCZOS_BASELINE"
        final = identity

    if alpha is not None:
        final_alpha = _resize_alpha(alpha, final_size)
        output = Image.fromarray(np.dstack((final, final_alpha)), "RGBA")
    else:
        output = Image.fromarray(final, "RGB")
    report: dict[str, object] = {
        "schema": "local-print-image-upscaler/raster-restore/1",
        "status": status,
        "source_size": list(source_size),
        "final_size": list(final_size),
        "scale": settings.scale,
        "enable_sr_x1": settings.enable_sr_x1,
        "source_metrics": measure_raster_metrics(original),
        "accepted_source_metrics": measure_raster_metrics(current),
        "final_metrics": measure_raster_metrics(final),
        "classical_stages": stages,
        "accepted_classical_stages": accepted_classical,
        "super_resolution": sr_report,
        "baseline": {
            "identity_at_x1_or_single_lanczos_resize": True,
            "identity_pixel_sha256": _pixel_sha256(identity),
            "working_pixel_sha256": _pixel_sha256(working_baseline),
        },
        "output_pixel_sha256": _pixel_sha256(final),
        "used_neural_prediction": used_neural_prediction,
        "detail_provenance": {
            "deblur": "bounded Gaussian-PSF luminance deconvolution",
            "denoise": "observed-pixel non-local means",
            "detail": "bounded observed high-frequency residual",
            "sr": "local V3 model prediction" if used_neural_prediction else "not accepted",
        },
        "fallback_safe": True,
        "truth_notice": TRUTH_NOTICE,
    }
    return RasterRestoreResult(output, report)


__all__ = [
    "LocalV3Assets",
    "RasterRestoreConfig",
    "RasterRestoreError",
    "RasterRestoreResult",
    "SRInferenceResult",
    "SuperResolutionBackend",
    "TRUTH_NOTICE",
    "V3SubprocessBackend",
    "identity_baseline",
    "measure_raster_metrics",
    "probe_local_v3_assets",
    "restore_raster",
]
