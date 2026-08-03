from __future__ import annotations

import csv
import gc
import hashlib
import io
import math
import shutil
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .geometry import (
    box_area,
    box_intersection,
    box_iou,
    box_mask,
    close_mask,
    connected_component_cleanup,
    expand_box,
    horizontal_gap,
    mask_bbox,
    mask_containment,
    mask_iou,
    safe_name,
    vertical_overlap_ratio,
)
from .model import Detection, LayerSpec, MaskCandidate, TextRegion


SAM_MODEL = "facebook/sam2.1-hiera-base-plus"
SAM_REVISION = "b7320756a13354e7530a63935656d35b2f91a290"
SAM_WEIGHT_SHA256 = "2012733a0de5d03efd1bba550a2847c4551be9ef2e0d497c83074df66189f780"
DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DINO_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
DINO_WEIGHT_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
TESSDATA_BEST_REVISION = "e2aad9b983032bb1beff9133104a67cdbb87ca4d"
TESSDATA_VIE_SHA256 = "b6b49293d95d0b6dbd8780174627e82c75be957b6f4ed9862155540d6b00bb45"
TESSDATA_VIE = Path(__file__).resolve().parents[1] / "models" / "tessdata" / "vie.traineddata"

DEFAULT_DETECTION_LABELS = (
    "person",
    "face",
    "animal",
    "vehicle",
    "building",
    "logo",
    "product",
    "package",
    "bottle",
    "food",
    "plant",
    "furniture",
    "hand",
    "gift",
    "label",
    "badge",
    "icon",
    "illustration",
    "text block",
)


def _cuda_device() -> int:
    import torch

    return 0 if torch.cuda.is_available() else -1


def _release_accelerator() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _local_model_snapshot(repo_id: str, revision: str) -> str:
    """Resolve an already verified snapshot without contacting the Hub at run time."""

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
            allow_patterns=("*.json", "*.safetensors", "vocab.txt"),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Pinned model is not available locally: {repo_id}@{revision}. "
            "Run APP\\engines\\V5\\setup_v5.ps1 once while online."
        ) from exc


@lru_cache(maxsize=4)
def _verified_file_sha256(path_text: str, expected_sha256: str, label: str) -> str:
    path = Path(path_text)
    if not path.is_file():
        raise RuntimeError(f"Pinned {label} artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Pinned {label} artifact failed SHA-256 verification: {path}. "
            "Run APP\\engines\\V5\\setup_v5.ps1 again before processing artwork."
        )
    return actual


def generate_sam_candidates(
    image: Image.Image,
    *,
    points_per_crop: int = 32,
    progress=print,
) -> tuple[list[np.ndarray], list[float], dict[str, object]]:
    """Run official SAM 2.1 automatic mask generation at source resolution."""

    import torch
    from transformers import pipeline

    device = _cuda_device()
    progress(
        "  V5 bước 1/4: SAM 2.1 đang tìm biên và vùng ảnh "
        + ("trên GPU..." if device >= 0 else "trên CPU (sẽ chậm)...")
    )
    snapshot = _local_model_snapshot(SAM_MODEL, SAM_REVISION)
    actual_weight_sha256 = _verified_file_sha256(
        str(Path(snapshot) / "model.safetensors"),
        SAM_WEIGHT_SHA256,
        "SAM 2.1",
    )
    generator = pipeline(
        "mask-generation",
        model=snapshot,
        device=device,
        dtype=torch.float32,
    )
    try:
        result = generator(
            image,
            points_per_crop=points_per_crop,
            points_per_batch=points_per_crop,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crops_n_layers=0,
        )
        masks = [item.detach().to("cpu").numpy().astype(bool) for item in result["masks"]]
        raw_scores = result["scores"]
        if hasattr(raw_scores, "detach"):
            raw_scores = raw_scores.detach().to("cpu").tolist()
        scores = [float(value) for value in raw_scores]
    finally:
        del generator
        _release_accelerator()
    return masks, scores, {
        "model": SAM_MODEL,
        "revision": SAM_REVISION,
        "model_safetensors_sha256": actual_weight_sha256,
        "model_safetensors_hash_verified_at_runtime": True,
        "model_source": "revision-pinned local cache; no run-time network access",
        "device": "cuda" if device >= 0 else "cpu",
        "points_per_crop": points_per_crop,
        "pred_iou_thresh": 0.86,
        "stability_score_thresh": 0.92,
        "raw_mask_count": len(masks),
    }


def run_object_detection(
    image: Image.Image,
    *,
    progress=print,
) -> tuple[list[Detection], dict[str, object]]:
    """Use Grounding DINO only as a naming/grouping cue; SAM remains the mask source."""

    import torch
    from transformers import pipeline

    device = _cuda_device()
    progress("  V5 bước 2/4: Grounding DINO đang nhận diện các khối có nghĩa...")
    snapshot = _local_model_snapshot(DINO_MODEL, DINO_REVISION)
    actual_weight_sha256 = _verified_file_sha256(
        str(Path(snapshot) / "model.safetensors"),
        DINO_WEIGHT_SHA256,
        "Grounding DINO",
    )
    detector = pipeline(
        "zero-shot-object-detection",
        model=snapshot,
        device=device,
        dtype=torch.float32,
    )
    try:
        raw = detector(
            image,
            candidate_labels=list(DEFAULT_DETECTION_LABELS),
            threshold=0.15,
        )
    finally:
        del detector
        _release_accelerator()

    width, height = image.size
    canvas_area = width * height
    detections: list[Detection] = []
    for item in raw:
        box_value = item.get("box", {})
        box = (
            max(0, int(round(float(box_value.get("xmin", 0))))),
            max(0, int(round(float(box_value.get("ymin", 0))))),
            min(width, int(round(float(box_value.get("xmax", width))))),
            min(height, int(round(float(box_value.get("ymax", height))))),
        )
        if box_area(box) <= 16 or box_area(box) / canvas_area > 0.55:
            continue
        detection = Detection(
            bbox=box,
            label=str(item.get("label", "object")).strip().lower(),
            score=float(item.get("score", 0.0)),
        )
        duplicate = next(
            (
                previous
                for previous in detections
                if previous.label == detection.label and box_iou(previous.bbox, box) > 0.72
            ),
            None,
        )
        if duplicate is None:
            detections.append(detection)
        elif detection.score > duplicate.score:
            detections.remove(duplicate)
            detections.append(detection)
    detections.sort(key=lambda item: item.score, reverse=True)
    return detections[:24], {
        "model": DINO_MODEL,
        "revision": DINO_REVISION,
        "model_safetensors_sha256": actual_weight_sha256,
        "model_safetensors_hash_verified_at_runtime": True,
        "model_source": "revision-pinned local cache; no run-time network access",
        "device": "cuda" if device >= 0 else "cpu",
        "threshold": 0.15,
        "accepted_detection_count": min(24, len(detections)),
    }


def _find_tesseract() -> Path | None:
    discovered = shutil.which("tesseract")
    candidates = (
        Path(discovered) if discovered else None,
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    )
    return next((path for path in candidates if path and path.is_file()), None)


def detect_text_regions(image_path: Path) -> tuple[list[TextRegion], dict[str, object]]:
    """Obtain line geometry from Tesseract; text remains raster, never fake editable text."""

    executable = _find_tesseract()
    if executable is None:
        return [], {"engine": "tesseract", "available": False, "regions": 0}
    local_vietnamese = TESSDATA_VIE.is_file()
    actual_vietnamese_sha256 = None
    if local_vietnamese:
        actual_vietnamese_sha256 = _verified_file_sha256(
            str(TESSDATA_VIE),
            TESSDATA_VIE_SHA256,
            "Vietnamese tessdata_best",
        )
    command = [
        str(executable),
        str(image_path),
        "stdout",
        "--psm",
        "11",
        "--oem",
        "1",
    ]
    if local_vietnamese:
        command.extend(["--tessdata-dir", str(TESSDATA_VIE.parent), "-l", "vie"])
    else:
        command.extend(["-l", "eng"])
    # Request TSV directly instead of the named ``tsv`` config. A private,
    # pinned tessdata directory contains language data but intentionally does
    # not copy machine-specific Tesseract config files.
    command.extend(["-c", "tessedit_create_tsv=1"])
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError):
        return [], {"engine": "tesseract", "available": True, "regions": 0, "failed": True}

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    reader = csv.DictReader(io.StringIO(completed.stdout), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            confidence = -1
        if not text or confidence < 0:
            continue
        key = (row.get("block_num", "0"), row.get("par_num", "0"), row.get("line_num", "0"))
        grouped[key].append(row)

    regions: list[TextRegion] = []
    for words in grouped.values():
        left = min(int(row["left"]) for row in words)
        top = min(int(row["top"]) for row in words)
        right = max(int(row["left"]) + int(row["width"]) for row in words)
        bottom = max(int(row["top"]) + int(row["height"]) for row in words)
        confidences = [float(row["conf"]) for row in words]
        regions.append(
            TextRegion(
                bbox=(left, top, right, bottom),
                text=" ".join(row["text"].strip() for row in words),
                confidence=sum(confidences) / len(confidences),
            )
        )
    regions.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return regions, {
        "engine": "tesseract",
        "available": True,
        "version": _tesseract_version(executable),
        "language": "vie geometry pass" if local_vietnamese else "eng geometry fallback",
        "vietnamese_model_revision": TESSDATA_BEST_REVISION if local_vietnamese else None,
        "vietnamese_model_sha256": actual_vietnamese_sha256,
        "vietnamese_model_hash_verified_at_runtime": bool(local_vietnamese),
        "regions": len(regions),
        "notice": "OCR text is metadata only; exported artwork remains raster pixels.",
    }


def _tesseract_version(executable: Path) -> str:
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        return result.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def build_mask_candidates(
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    scores: list[float],
) -> list[MaskCandidate]:
    height, width = image_rgb.shape[:2]
    canvas_area = height * width
    candidates: list[MaskCandidate] = []
    for index, (mask, score) in enumerate(zip(masks, scores, strict=True)):
        area = int(mask.sum())
        ratio = area / canvas_area
        if ratio < 0.00012 or ratio > 0.72:
            continue
        bbox = mask_bbox(mask)
        if bbox[2] - bbox[0] < 3 or bbox[3] - bbox[1] < 3:
            continue
        fill = area / max(1, box_area(bbox))
        pixels = image_rgb[mask]
        mean = tuple(float(value) for value in pixels.mean(axis=0))
        std = float(pixels.astype(np.float32).std(axis=0).mean())
        candidate = MaskCandidate(index, mask, float(score), bbox, area, fill, mean, std)
        if _is_near_duplicate(candidate, candidates):
            continue
        candidates.append(candidate)
    return candidates


def _is_near_duplicate(candidate: MaskCandidate, accepted: list[MaskCandidate]) -> bool:
    for previous in reversed(accepted):
        area_ratio = max(candidate.area, previous.area) / max(1, min(candidate.area, previous.area))
        if area_ratio > 1.35:
            continue
        if box_iou(candidate.bbox, previous.bbox) < 0.52:
            continue
        if mask_iou(candidate.mask, previous.mask) > 0.82:
            return True
    return False


def _candidate_matches_box(candidate: MaskCandidate, box: tuple[int, int, int, int]) -> float:
    intersection_box = box_intersection(candidate.bbox, box)
    if not intersection_box:
        return 0.0
    x0, y0, x1, y1 = box
    intersection = int(candidate.mask[y0:y1, x0:x1].sum())
    inside = intersection / max(1, candidate.area)
    coverage = intersection / max(1, box_area(box))
    size_balance = min(candidate.area, box_area(box)) / max(candidate.area, box_area(box))
    return inside * 0.62 + coverage * 0.23 + size_balance * 0.15


def _background_like(candidate: MaskCandidate, canvas_area: int) -> bool:
    ratio = candidate.area / canvas_area
    return (
        (ratio > 0.17)
        or (ratio > 0.025 and candidate.fill_ratio > 0.72 and candidate.std_rgb < 34.0)
    )


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[b] = a


def _should_join(a: MaskCandidate, b: MaskCandidate, shape: tuple[int, int]) -> bool:
    height, width = shape
    ah, bh = a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1]
    aw, bw = a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0]
    colour_distance = math.dist(a.mean_rgb, b.mean_rgb)
    if mask_containment(a.mask, b.mask) > 0.86 or mask_containment(b.mask, a.mask) > 0.86:
        return True
    same_row = (
        vertical_overlap_ratio(a.bbox, b.bbox) > 0.42
        and max(ah, bh) / max(1, min(ah, bh)) < 3.2
        and horizontal_gap(a.bbox, b.bbox) < 1.35 * max(ah, bh)
        and colour_distance < 82.0
    )
    if same_row:
        return True
    close_object_parts = (
        box_iou(a.bbox, b.bbox) > 0.08
        and max(aw, bw) < width * 0.32
        and max(ah, bh) < height * 0.32
        and colour_distance < 105.0
    )
    return close_object_parts


def _layer_from_candidates(
    members: list[MaskCandidate],
    *,
    layer_id: str,
    name: str,
    category: str,
    shape: tuple[int, int],
    label: str | None = None,
    text: str | None = None,
) -> LayerSpec:
    merged = np.zeros(shape, dtype=bool)
    for member in members:
        merged |= member.mask
    minimum_component = max(4, int(merged.size * 0.000015))
    merged = connected_component_cleanup(
        close_mask(merged, 1), minimum_area=minimum_component, keep_largest_if_empty=True
    )
    return LayerSpec(
        layer_id=layer_id,
        name=safe_name(name, layer_id),
        category=category,
        mask=merged,
        score=max(member.score for member in members),
        source_ids=[member.candidate_id for member in members],
        label=label,
        text=text,
    )


def _text_contrast_mask(
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    seed: np.ndarray,
) -> np.ndarray:
    """Complete missing glyph pieces using the dominant colour around an OCR line."""

    x0, y0, x1, y1 = box
    roi = image_rgb[y0:y1, x0:x1]
    if roi.size == 0:
        return seed
    h, w = roi.shape[:2]
    band = max(2, min(h, w) // 10)
    border = np.zeros((h, w), dtype=bool)
    border[:band] = True
    border[-band:] = True
    border[:, :band] = True
    border[:, -band:] = True
    border_pixels = roi[border]
    if not len(border_pixels):
        return seed
    # The most frequent coarse border colour is more robust than a mean when a
    # glyph touches one side of its OCR box.
    bins = (border_pixels // 16).astype(np.int16)
    unique, counts = np.unique(bins, axis=0, return_counts=True)
    dominant_bin = unique[int(np.argmax(counts))]
    dominant_pixels = border_pixels[np.all(np.abs(bins - dominant_bin) <= 1, axis=1)]
    background = np.median(dominant_pixels if len(dominant_pixels) else border_pixels, axis=0)
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(
        np.uint8([[np.clip(background, 0, 255)]]), cv2.COLOR_RGB2LAB
    )[0, 0].astype(np.float32)
    distance = np.linalg.norm(roi_lab - background_lab, axis=2)
    border_distance = distance[border]
    threshold = max(14.0, float(np.percentile(border_distance, 78)) + 5.0)
    contrast = distance > threshold
    contrast = cv2.morphologyEx(
        contrast.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)
    expanded = seed.copy()
    local_seed = seed[y0:y1, x0:x1]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(contrast.astype(np.uint8), 8)
    accepted = np.zeros_like(contrast)
    minimum = max(3, int(h * w * 0.00035))
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < minimum:
            continue
        component_mask = labels == component
        intersects_seed = bool(np.logical_and(component_mask, local_seed).any())
        if intersects_seed or area < h * w * 0.28:
            accepted |= component_mask
    expanded[y0:y1, x0:x1] |= accepted
    return expanded


def _suppress_duplicate_layers(layers: list[LayerSpec]) -> list[LayerSpec]:
    accepted: list[LayerSpec] = []
    for layer in layers:
        replace_index: int | None = None
        reject = False
        for index, previous in enumerate(accepted):
            overlap = mask_iou(layer.mask, previous.mask)
            containment_a = mask_containment(layer.mask, previous.mask)
            containment_b = mask_containment(previous.mask, layer.mask)
            # Text nested in a panel/object is intentional and later becomes a child.
            if "text_raster" in {layer.category, previous.category} and layer.category != previous.category:
                continue
            similar_nontext = (
                layer.category != "text_raster"
                and previous.category != "text_raster"
                and overlap > 0.42
                and max(layer.area, previous.area) / max(1, min(layer.area, previous.area)) < 2.2
            )
            same_category_duplicate = (
                layer.category == previous.category
                and (overlap > 0.58 or min(containment_a, containment_b) > 0.88)
            )
            if not (similar_nontext or same_category_duplicate):
                continue
            layer_quality = layer.area * layer.score
            previous_quality = previous.area * previous.score
            if layer_quality > previous_quality:
                replace_index = index
            else:
                reject = True
            break
        if reject:
            continue
        if replace_index is not None:
            accepted[replace_index] = layer
        else:
            accepted.append(layer)
    return accepted


def _assign_hierarchy(layers: list[LayerSpec]) -> None:
    by_id = {layer.layer_id: layer for layer in layers}
    for child in layers:
        possible: list[LayerSpec] = []
        for parent in layers:
            if parent is child or parent.category == "text_raster":
                continue
            parent_box_area = box_area(parent.bbox)
            child_box_area = box_area(child.bbox)
            if parent.area < child.area * 1.05 and parent_box_area < child_box_area * 1.08:
                continue
            mask_inside = mask_containment(child.mask, parent.mask)
            box_inside = box_intersection(child.bbox, parent.bbox) / max(1, child_box_area)
            box_scale = parent_box_area / max(1, child_box_area)
            if mask_inside >= 0.62 or (box_inside >= 0.88 and box_scale <= 14.0):
                possible.append(parent)
        parent = min(possible, key=lambda item: item.area, default=None)
        child.metadata["parent_id"] = parent.layer_id if parent else None
    for layer in layers:
        layer.metadata["children"] = [
            child.layer_id for child in layers if child.metadata.get("parent_id") == layer.layer_id
        ]
        depth = 0
        parent_id = layer.metadata.get("parent_id")
        seen: set[str] = set()
        while parent_id and parent_id in by_id and parent_id not in seen:
            seen.add(str(parent_id))
            depth += 1
            parent_id = by_id[str(parent_id)].metadata.get("parent_id")
        layer.metadata["hierarchy_depth"] = depth
    # A parent base must remain opaque under every child. SAM often represents
    # a panel as "panel colour minus lettering"; unioning descendants closes
    # those semantic holes so moving a child reveals the cleaned parent.
    for child in sorted(
        layers,
        key=lambda item: int(item.metadata.get("hierarchy_depth", 0)),
        reverse=True,
    ):
        parent_id = child.metadata.get("parent_id")
        if parent_id and str(parent_id) in by_id:
            parent = by_id[str(parent_id)]
            parent.mask = close_mask(parent.mask | child.mask, 1)
def select_smart_layers(
    image_rgb: np.ndarray,
    candidates: list[MaskCandidate],
    text_regions: list[TextRegion],
    detections: list[Detection],
    *,
    max_layers: int = 24,
) -> tuple[list[LayerSpec], dict[str, object]]:
    """Turn hierarchical SAM proposals into a small set of useful movable regions."""

    height, width = image_rgb.shape[:2]
    shape = (height, width)
    canvas_area = height * width
    used_ids: set[int] = set()
    layers: list[LayerSpec] = []

    # OCR geometry has first priority: letters and accents become one line/block,
    # not dozens of unmanageable glyph fragments.
    for region_index, region in enumerate(text_regions, 1):
        padded = expand_box(region.bbox, max(3, int(min(width, height) * 0.006)), width, height)
        matches = [
            candidate
            for candidate in candidates
            if candidate.candidate_id not in used_ids
            and candidate.area / canvas_area < 0.085
            and _candidate_matches_box(candidate, padded) > 0.57
        ]
        if not matches:
            continue
        layer = _layer_from_candidates(
            matches,
            layer_id=f"text_{region_index:02d}",
            name=f"TEXT {region_index:02d} - {region.text}",
            category="text_raster",
            shape=shape,
            text=region.text,
        )
        layer.mask = _text_contrast_mask(image_rgb, padded, layer.mask)
        if layer.area / canvas_area < 0.00012:
            continue
        layers.append(layer)
        used_ids.update(layer.source_ids)

    # DINO gives semantic names and boxes; choose the best SAM boundary for each box.
    object_number = 0
    for detection in detections:
        ranked = sorted(
            (
                (_candidate_matches_box(candidate, detection.bbox), candidate)
                for candidate in candidates
                if candidate.candidate_id not in used_ids
                and candidate.area / canvas_area < 0.22
                and not _background_like(candidate, canvas_area)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.43:
            continue
        best_score, best = ranked[0]
        duplicate = any(mask_iou(best.mask, layer.mask) > 0.62 for layer in layers)
        if duplicate:
            continue
        object_number += 1
        layer = _layer_from_candidates(
            [best],
            layer_id=f"object_{object_number:02d}",
            name=f"OBJECT {object_number:02d} - {detection.label}",
            category="object",
            shape=shape,
            label=detection.label,
        )
        layer.metadata.update(
            {
                "detection_score": detection.score,
                "sam_box_match": best_score,
                "detection_box": list(detection.bbox),
            }
        )
        layers.append(layer)
        used_ids.add(best.candidate_id)

    # Residual SAM regions are clustered by row, colour, proximity and containment.
    residual = [
        candidate
        for candidate in candidates
        if candidate.candidate_id not in used_ids
        and 0.00018 <= candidate.area / canvas_area <= 0.09
        and not _background_like(candidate, canvas_area)
    ]
    residual.sort(key=lambda item: (item.area, item.score), reverse=True)
    residual = residual[:90]
    forest = _UnionFind(len(residual))
    for first in range(len(residual)):
        for second in range(first + 1, len(residual)):
            if _should_join(residual[first], residual[second], shape):
                forest.union(first, second)
    groups: dict[int, list[MaskCandidate]] = defaultdict(list)
    for index, candidate in enumerate(residual):
        groups[forest.find(index)].append(candidate)

    detail_number = 0
    group_values = sorted(groups.values(), key=lambda group: sum(item.area for item in group), reverse=True)
    for group in group_values:
        if len(layers) >= max_layers:
            break
        detail_number += 1
        layer = _layer_from_candidates(
            group,
            layer_id=f"detail_{detail_number:02d}",
            name=f"DETAIL {detail_number:02d}",
            category="detail_group",
            shape=shape,
        )
        ratio = layer.area / canvas_area
        if ratio < 0.00022 or ratio > 0.14:
            continue
        if any(mask_containment(layer.mask, previous.mask) > 0.82 for previous in layers):
            continue
        layers.append(layer)

    # Prefer important, readable groups and enforce a deterministic cap.
    category_priority = {"text_raster": 3, "object": 2, "detail_group": 1}
    layers = _suppress_duplicate_layers(layers)
    layers.sort(
        key=lambda item: (
            category_priority.get(item.category, 0),
            math.sqrt(item.area / canvas_area) * item.score,
        ),
        reverse=True,
    )
    layers = layers[:max_layers]
    _assign_hierarchy(layers)
    for index, layer in enumerate(layers, 1):
        layer.metadata["display_order"] = index

    covered = np.zeros(shape, dtype=bool)
    for layer in layers:
        covered |= layer.mask
    top_level = np.zeros(shape, dtype=bool)
    for layer in layers:
        if layer.metadata.get("parent_id") is None:
            top_level |= layer.mask
    report = {
        "candidate_count_after_filtering": len(candidates),
        "selected_layer_count": len(layers),
        "selected_by_category": {
            category: sum(layer.category == category for layer in layers)
            for category in sorted({layer.category for layer in layers})
        },
        "foreground_coverage_ratio": round(float(covered.mean()), 6),
        "top_level_removal_ratio_before_dilation": round(float(top_level.mean()), 6),
        "hierarchy_edges": sum(layer.metadata.get("parent_id") is not None for layer in layers),
        "max_layers": max_layers,
        "grouping_policy": (
            "OCR lines first; Grounding DINO semantic boxes second; residual SAM masks are "
            "merged by containment, row alignment, colour and proximity; flat large panels stay in background."
        ),
    }
    return layers, report


def segmentation_overlay(image_rgb: np.ndarray, layers: list[LayerSpec]) -> Image.Image:
    palette = np.array(
        [
            (0, 220, 255),
            (255, 80, 80),
            (120, 255, 80),
            (255, 190, 30),
            (190, 80, 255),
            (60, 160, 255),
        ],
        dtype=np.float32,
    )
    overlay = image_rgb.astype(np.float32).copy()
    for index, layer in enumerate(layers):
        colour = palette[index % len(palette)]
        boundary = cv2.morphologyEx(
            layer.mask.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), np.uint8),
        ).astype(bool)
        overlay[layer.mask] = overlay[layer.mask] * 0.84 + colour * 0.16
        overlay[boundary] = colour
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), "RGB")
