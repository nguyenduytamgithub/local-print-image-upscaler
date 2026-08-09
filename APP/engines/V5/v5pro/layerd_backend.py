from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image


LAYERD_REPO = "cyberagent/layerd-birefnet"
LAYERD_REVISION = "679f743cd001fb5d6360e59e8e1904678c5fa734"
LAYERD_WEIGHT_SHA256 = "28f8acba2736067bf2eb8152f2d3ce75a388dd4e4e57f068b30760a8fe1c44d0"


@dataclass(slots=True)
class LayerDRawLayer:
    rgba: np.ndarray
    z_index: int
    source_iteration: int

    def __post_init__(self) -> None:
        value = np.asarray(self.rgba)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 4:
            raise ValueError("LayerD raw layer must be uint8 RGBA.")
        self.rgba = np.ascontiguousarray(value)


@dataclass(slots=True)
class LayerDResult:
    background_rgb: np.ndarray
    foregrounds_bottom_to_top: list[LayerDRawLayer]
    report: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_layerd_snapshot(*, local_files_only: bool = True) -> Path:
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=LAYERD_REPO,
                revision=LAYERD_REVISION,
                local_files_only=local_files_only,
                allow_patterns=("*.json", "*.safetensors", "*.py"),
            )
        )
    except Exception as exc:
        action = "local verification" if local_files_only else "download"
        raise RuntimeError(
            f"LayerD model {action} failed for {LAYERD_REPO}@{LAYERD_REVISION}: {exc}"
        ) from exc
    weight_path = snapshot / "model.safetensors"
    if not weight_path.is_file():
        raise RuntimeError(f"LayerD weight is missing: {weight_path}")
    actual = sha256_file(weight_path)
    if actual.lower() != LAYERD_WEIGHT_SHA256:
        raise RuntimeError(
            "LayerD weight SHA-256 mismatch: "
            f"expected {LAYERD_WEIGHT_SHA256}, got {actual}"
        )
    return snapshot


def _import_vendored_layerd(vendor_root: Path):
    source_root = vendor_root / "src"
    if not (source_root / "layerd" / "models" / "layerd.py").is_file():
        raise RuntimeError(f"Vendored LayerD source is incomplete: {source_root}")
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from layerd import LayerD

    return LayerD


def run_layerd(
    image_rgb: np.ndarray,
    *,
    vendor_root: Path,
    lama_model: Path,
    device: str,
    max_iterations: int = 8,
    process_size: int = 1024,
) -> LayerDResult:
    """Run the official graphic-design decomposition core from pinned local assets."""

    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("LayerD input must be uint8 RGB.")
    if not 1 <= max_iterations <= 32:
        raise ValueError("LayerD max_iterations must be from 1 through 32.")
    if not 512 <= process_size <= 2048:
        raise ValueError("LayerD process_size must be from 512 through 2048.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("LayerD device must be cpu or cuda.")
    if not lama_model.is_file():
        raise RuntimeError(f"Verified LaMa model is missing: {lama_model}")
    snapshot = resolve_layerd_snapshot(local_files_only=True)
    os.environ["LAMA_MODEL"] = str(lama_model.resolve())
    LayerD = _import_vendored_layerd(vendor_root)

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot use the NVIDIA GPU.")
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = LayerD(
        matting_hf_card=str(snapshot),
        matting_process_size=(process_size, process_size),
        use_unblend=True,
        bg_refine=True,
        fg_refine=True,
        fg_refine_num_colors=4,
        bg_refine_num_colors=16,
        kernel_scale=0.008,
        device=device,
    )
    loaded_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    layers = model.decompose(Image.fromarray(image_rgb, "RGB"), max_iterations=max_iterations)
    inference_seconds = time.perf_counter() - inference_started
    if not layers:
        raise RuntimeError("LayerD returned no background layer.")
    background = np.asarray(layers[0].convert("RGB"), dtype=np.uint8).copy()
    foregrounds: list[LayerDRawLayer] = []
    # LayerD returns final background followed by foregrounds in compositing
    # order (bottom to top). The extraction iteration therefore runs in the
    # opposite direction.
    count = len(layers) - 1
    for index, layer in enumerate(layers[1:], 1):
        rgba = np.asarray(layer.convert("RGBA"), dtype=np.uint8).copy()
        foregrounds.append(
            LayerDRawLayer(
                rgba=rgba,
                z_index=index,
                source_iteration=count - index + 1,
            )
        )
    peak_vram = 0
    if device == "cuda":
        torch.cuda.synchronize()
        peak_vram = int(torch.cuda.max_memory_allocated())
    alpha_coverages = [
        round(float(np.count_nonzero(item.rgba[:, :, 3])) / image_rgb.shape[0] / image_rgb.shape[1], 8)
        for item in foregrounds
    ]
    return LayerDResult(
        background_rgb=background,
        foregrounds_bottom_to_top=foregrounds,
        report={
            "backend": "CyberAgentAILab LayerD ICCV 2025",
            "upstream_commit": "21aef937a0371614adb4d961f52d02409cb8ecc7",
            "model_repo": LAYERD_REPO,
            "model_revision": LAYERD_REVISION,
            "model_weight_sha256": LAYERD_WEIGHT_SHA256,
            "model_hash_verified": True,
            "device": device,
            "process_size": process_size,
            "requested_max_iterations": max_iterations,
            "completed_foreground_iterations": len(foregrounds),
            "alpha_coverages_bottom_to_top": alpha_coverages,
            "model_load_seconds": round(loaded_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "peak_vram_bytes": peak_vram,
            "limitations": (
                "LayerD is an ill-posed reconstruction model; tiny text and ambiguous "
                "granularity require the independent inventory and review gates."
            ),
        },
    )
