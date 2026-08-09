from __future__ import annotations

import gc
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

from .inventory import DetectedProposal
from .schema import AlphaCrop, Box, ElementKind, ProposalRecord


DINO_REPO = "IDEA-Research/grounding-dino-tiny"
DINO_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
DINO_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
SAM_REPO = "facebook/sam2.1-hiera-base-plus"
SAM_REVISION = "b7320756a13354e7530a63935656d35b2f91a290"
SAM_SHA256 = "2012733a0de5d03efd1bba550a2847c4551be9ef2e0d497c83074df66189f780"
BIREF_REPO = "ZhengPeng7/BiRefNet_HR-matting"
BIREF_REVISION = "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f"
BIREF_SHA256 = "a5a4de698739ea5e0e8bbab28e1b293dde95092b87a442d566cbc585c53cef55"

SEMANTIC_LABELS = (
    # Concrete retail-poster phrases outperform generic words such as "box"
    # and "bag", which GroundingDINO otherwise tends to attach to whole cards.
    "facial tissue package",
    "tissue box",
    "pack of tissues",
    "toilet paper package",
    "toilet paper roll",
    "wet wipes package",
    "paper towel roll",
    "bottle of household cleaner",
    "household product package",
    "roll of garbage bags",
    "trash bin",
    "brand logo",
    "flower decoration",
    "leaf decoration",
    "plant decoration",
    "QR code",
    "shopping cart icon",
    "gift box icon",
    "delivery truck icon",
    "shield icon",
    "customer support headset icon",
    "points badge",
    "ribbon banner",
    "illustration",
)


@dataclass(slots=True)
class SemanticResult:
    proposals: list[DetectedProposal]
    report: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(repo: str, revision: str, expected_sha256: str, *, remote_code: bool = False) -> Path:
    patterns = ["*.json", "*.safetensors", "vocab.txt"]
    if remote_code:
        patterns.append("*.py")
    try:
        path = Path(
            snapshot_download(
                repo_id=repo,
                revision=revision,
                local_files_only=True,
                allow_patterns=tuple(patterns),
            )
        )
    except Exception as exc:
        # huggingface_hub >= 1.0 can raise IncompleteSnapshotError when a
        # deliberately filtered download lacks unrelated repository files,
        # even though every runtime file is present. Resolve the immutable
        # revision directly from the cache in that case; the safetensors hash
        # below remains the authority.
        from huggingface_hub import constants

        cache_name = "models--" + repo.replace("/", "--")
        cached = Path(constants.HF_HUB_CACHE) / cache_name / "snapshots" / revision
        if not cached.is_dir():
            raise RuntimeError(f"Pinned model is unavailable locally: {repo}@{revision}") from exc
        path = cached
    weight = path / "model.safetensors"
    if not weight.is_file() or _sha256(weight).lower() != expected_sha256:
        raise RuntimeError(f"Pinned model failed SHA-256 verification: {repo}@{revision}")
    return path


def _release_cuda(*objects: object) -> None:
    del objects
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _box_iou(first: Box, second: Box) -> float:
    ix0, iy0 = max(first[0], second[0]), max(first[1], second[1])
    ix1, iy1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _kind_from_label(label: str) -> ElementKind:
    value = label.lower()
    if "qr" in value:
        return "qr"
    if "logo" in value:
        return "logo"
    if "badge" in value:
        return "badge"
    if "ribbon" in value or "banner" in value:
        return "ribbon"
    if "flower" in value or "plant" in value or "illustration" in value:
        return "decoration"
    if "icon" in value:
        return "icon"
    return "product"


def _semantic_policy(
    detection: dict[str, Any], canvas_size: tuple[int, int]
) -> tuple[bool, dict[str, Any]]:
    """Reject obviously implausible open-vocabulary boxes before segmentation.

    Low-score detections are still retained in the proposal ledger for review;
    this gate only decides whether a box is allowed to create/protect pixels.
    """

    canvas_width, canvas_height = canvas_size
    x0, y0, x1, y1 = detection["bbox"]
    box_width, box_height = x1 - x0, y1 - y0
    area_fraction = box_width * box_height / max(1, canvas_width * canvas_height)
    aspect = max(box_width / max(1, box_height), box_height / max(1, box_width))
    kind = _kind_from_label(str(detection["label"]))
    # (minimum score, minimum area fraction, maximum area fraction, max aspect)
    limits: dict[ElementKind, tuple[float, float, float, float]] = {
        "product": (0.070, 0.00020, 0.060, 7.0),
        "logo": (0.065, 0.00008, 0.040, 8.0),
        "qr": (0.065, 0.00015, 0.045, 2.4),
        "icon": (0.050, 0.00005, 0.012, 4.0),
        "badge": (0.050, 0.00008, 0.025, 3.0),
        "ribbon": (0.050, 0.00020, 0.090, 14.0),
        "decoration": (0.050, 0.00008, 0.140, 14.0),
    }
    minimum_score, minimum_area, maximum_area, maximum_aspect = limits.get(
        kind, (0.10, 0.0001, 0.05, 8.0)
    )
    reasons: list[str] = []
    if float(detection["score"]) < minimum_score:
        reasons.append("detector score below kind-specific floor")
    if area_fraction < minimum_area:
        reasons.append("box too small for semantic segmentation")
    if area_fraction > maximum_area:
        reasons.append("box covers implausibly large canvas fraction")
    if aspect > maximum_aspect:
        reasons.append("box aspect ratio is implausible for label")
    report = {
        "accepted": not reasons,
        "kind": kind,
        "area_fraction": round(area_fraction, 8),
        "aspect_ratio": round(aspect, 6),
        "minimum_score": minimum_score,
        "area_range": [minimum_area, maximum_area],
        "maximum_aspect_ratio": maximum_aspect,
        "reasons": reasons,
    }
    return not reasons, report


def _localize_qr_candidate(
    image_rgb: np.ndarray, candidate: dict[str, Any]
) -> dict[str, Any] | None:
    """Confirm QR structure and replace a loose DINO context box with modules."""

    x0, y0, x1, y1 = candidate["bbox"]
    crop = image_rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    try:
        found, points = cv2.QRCodeDetector().detect(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    except cv2.error:
        return None
    if not found or points is None:
        return None
    vertices = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    local_x0 = max(0, int(math.floor(float(vertices[:, 0].min()))))
    local_y0 = max(0, int(math.floor(float(vertices[:, 1].min()))))
    local_x1 = min(crop.shape[1], int(math.ceil(float(vertices[:, 0].max()))) + 1)
    local_y1 = min(crop.shape[0], int(math.ceil(float(vertices[:, 1].max()))) + 1)
    if local_x1 - local_x0 < 8 or local_y1 - local_y0 < 8:
        return None
    return {
        **candidate,
        "bbox": (x0 + local_x0, y0 + local_y0, x0 + local_x1, y0 + local_y1),
        "dino_context_bbox": candidate["bbox"],
        "qr_detector_points": np.rint(vertices).astype(int).tolist(),
    }


def _select_semantic_cluster(
    cluster: list[dict[str, Any]], canvas_size: tuple[int, int], image_rgb: np.ndarray
) -> dict[str, Any]:
    """Choose a stable semantic family when several labels describe one box."""

    best_by_kind: dict[ElementKind, dict[str, Any]] = {}
    for item in cluster:
        kind = _kind_from_label(str(item["label"]))
        current = best_by_kind.get(kind)
        if current is None or float(item["score"]) > float(current["score"]):
            best_by_kind[kind] = item
    highest = max(float(item["score"]) for item in cluster)
    # Generic icon phrases are prone to winning product boxes by a few score
    # points.  A concrete product/QR/badge label that is close to the maximum
    # is more useful and remains subject to the geometry gate afterwards.
    qr_candidate = best_by_kind.get("qr")
    selected: dict[str, Any] | None = None
    if qr_candidate is not None:
        selected = _localize_qr_candidate(image_rgb, qr_candidate)
    family_floors: tuple[tuple[ElementKind, float], ...] = (
        ("badge", 0.62),
        ("product", 0.68),
        ("logo", 0.72),
        ("ribbon", 0.76),
        ("decoration", 0.76),
        ("icon", 0.0),
    )
    for kind, ratio in family_floors:
        if selected is not None:
            break
        candidate = best_by_kind.get(kind)
        if candidate is None or float(candidate["score"]) < highest * ratio:
            continue
        plausible, _ = _semantic_policy(candidate, canvas_size)
        if plausible:
            selected = candidate
            break
    if selected is None:
        selected = max(cluster, key=lambda item: float(item["score"]))
    alternatives = sorted(
        (
            {"label": str(item["label"]), "score": round(float(item["score"]), 6)}
            for item in cluster
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return {**selected, "alternative_labels": alternatives[:12]}


def _run_dino(
    image: Image.Image,
    *,
    device: int,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import pipeline

    snapshot = _snapshot(DINO_REPO, DINO_REVISION, DINO_SHA256)
    # GroundingDINO's multi-scale deformable attention calls grid_sample with
    # float coordinates. On the supported Windows torch/transformers build,
    # forcing the whole detector to FP16 produces a Half/Float type mismatch.
    # FP32 fits the 12 GiB target because DINO is released before SAM2 loads.
    detector = pipeline(
        "zero-shot-object-detection",
        model=str(snapshot),
        device=device,
        dtype=torch.float32,
    )
    started = time.perf_counter()
    try:
        raw = detector(image, candidate_labels=list(SEMANTIC_LABELS), threshold=threshold)
    finally:
        del detector
        _release_cuda()
    width, height = image.size
    parsed: list[dict[str, Any]] = []
    for item in raw:
        value = item.get("box", {})
        bbox = (
            max(0, min(width, int(round(float(value.get("xmin", 0)))))),
            max(0, min(height, int(round(float(value.get("ymin", 0)))))),
            max(0, min(width, int(round(float(value.get("xmax", width)))))),
            max(0, min(height, int(round(float(value.get("ymax", height)))))),
        )
        if bbox[2] - bbox[0] < 4 or bbox[3] - bbox[1] < 4:
            continue
        score = float(item.get("score", 0.0))
        label = str(item.get("label", "object")).strip().lower()
        parsed.append({"bbox": bbox, "score": score, "label": label})
    clusters: list[list[dict[str, Any]]] = []
    for candidate in sorted(parsed, key=lambda item: item["score"], reverse=True):
        cluster = next(
            (
                current
                for current in clusters
                if any(_box_iou(candidate["bbox"], item["bbox"]) >= 0.76 for item in current)
            ),
            None,
        )
        if cluster is None:
            clusters.append([candidate])
        else:
            cluster.append(candidate)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    candidates = [
        _select_semantic_cluster(cluster, (width, height), image_rgb) for cluster in clusters
    ]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates, {
        "model": DINO_REPO,
        "revision": DINO_REVISION,
        "weight_sha256": DINO_SHA256,
        "threshold": threshold,
        "dtype": "float32",
        "raw_detection_count": len(raw),
        "deduplicated_detection_count": len(candidates),
        "seconds": round(time.perf_counter() - started, 3),
    }


def _run_sam_boxes(
    image: Image.Image,
    detections: list[dict[str, Any]],
    *,
    device: str,
    chunk_size: int = 32,
) -> tuple[list[tuple[np.ndarray | None, float]], dict[str, Any]]:
    import torch
    from transformers import Sam2Model, Sam2Processor

    snapshot = _snapshot(SAM_REPO, SAM_REVISION, SAM_SHA256)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Sam2Model.from_pretrained(str(snapshot), dtype=dtype).to(device).eval()
    processor = Sam2Processor.from_pretrained(str(snapshot))
    results: list[tuple[np.ndarray | None, float]] = []
    started = time.perf_counter()
    try:
        for offset in range(0, len(detections), chunk_size):
            chunk = detections[offset : offset + chunk_size]
            boxes = [[list(item["bbox"]) for item in chunk]]
            inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs, multimask_output=True)
            masks = processor.post_process_masks(
                outputs.pred_masks.detach().cpu(), inputs["original_sizes"].detach().cpu()
            )[0]
            scores = outputs.iou_scores.detach().float().cpu()[0]
            masks_array = masks.detach().cpu().numpy()
            scores_array = scores.numpy()
            for index in range(len(chunk)):
                object_masks = masks_array[index]
                object_scores = np.ravel(scores_array[index])
                best = int(np.argmax(object_scores))
                mask = np.asarray(object_masks[best] > 0, dtype=bool)
                score = float(object_scores[best])
                results.append((mask, score))
    finally:
        del model, processor
        _release_cuda()
    return results, {
        "model": SAM_REPO,
        "revision": SAM_REVISION,
        "weight_sha256": SAM_SHA256,
        "prompt": "Grounding DINO bounding boxes",
        "box_count": len(detections),
        "seconds": round(time.perf_counter() - started, 3),
    }


def _fallback_soft_alpha(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    signed = inside - outside
    return np.clip(np.rint((signed + 1.25) / 2.5 * 255.0), 0, 255).astype(np.uint8)


def _expand_box(box: Box, canvas_size: tuple[int, int], fraction: float = 0.08) -> Box:
    width, height = canvas_size
    x0, y0, x1, y1 = box
    margin = max(6, int(round(max(x1 - x0, y1 - y0) * fraction)))
    return max(0, x0 - margin), max(0, y0 - margin), min(width, x1 + margin), min(height, y1 + margin)


def _refine_with_birefnet(
    image_rgb: np.ndarray,
    masks: list[np.ndarray | None],
    boxes: list[Box],
    *,
    device: str,
    process_size: int,
) -> tuple[list[tuple[np.ndarray | None, dict[str, Any]]], dict[str, Any]]:
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    snapshot = _snapshot(BIREF_REPO, BIREF_REVISION, BIREF_SHA256, remote_code=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForImageSegmentation.from_pretrained(
        str(snapshot), trust_remote_code=True, dtype=dtype
    ).to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize((process_size, process_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    refined: list[tuple[np.ndarray | None, dict[str, Any]]] = []
    height, width = image_rgb.shape[:2]
    started = time.perf_counter()
    try:
        for source_mask, box in zip(masks, boxes, strict=True):
            if source_mask is None or not np.any(source_mask):
                refined.append((None, {"accepted": False, "reason": "SAM returned no mask"}))
                continue
            expanded = _expand_box(box, (width, height))
            x0, y0, x1, y1 = expanded
            crop = Image.fromarray(image_rgb[y0:y1, x0:x1], "RGB")
            tensor = transform(crop).unsqueeze(0).to(device=device, dtype=dtype)
            try:
                with torch.inference_mode():
                    prediction = model(tensor)[0][-1].sigmoid().float().cpu().numpy()[0, 0]
                prediction = cv2.resize(
                    prediction,
                    (x1 - x0, y1 - y0),
                    interpolation=cv2.INTER_CUBIC,
                )
            except (RuntimeError, ValueError) as exc:
                refined.append(
                    (
                        _fallback_soft_alpha(source_mask),
                        {"accepted": False, "reason": f"BiRefNet inference fallback: {exc}"},
                    )
                )
                continue
            local_sam = source_mask[y0:y1, x0:x1]
            radius = max(2, int(round(min(x1 - x0, y1 - y0) * 0.025)))
            envelope = cv2.dilate(
                local_sam.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)),
            ).astype(bool)
            predicted_binary = prediction >= 0.35
            intersection = np.count_nonzero(predicted_binary & local_sam)
            union = np.count_nonzero(predicted_binary | local_sam)
            iou = intersection / union if union else 0.0
            area_ratio = np.count_nonzero(predicted_binary) / max(1, np.count_nonzero(local_sam))
            if iou < 0.28 or not 0.30 <= area_ratio <= 3.20:
                alpha_canvas = _fallback_soft_alpha(source_mask)
                refined.append(
                    (
                        alpha_canvas,
                        {
                            "accepted": False,
                            "reason": "BiRefNet disagrees with SAM envelope",
                            "sam_iou": round(iou, 6),
                            "area_ratio": round(area_ratio, 6),
                        },
                    )
                )
                continue
            local_alpha = np.clip(np.rint(prediction * 255.0), 0, 255).astype(np.uint8)
            local_alpha[~envelope] = 0
            sure_core = cv2.erode(
                local_sam.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            ).astype(bool)
            local_alpha[sure_core] = np.maximum(local_alpha[sure_core], 245)
            alpha_canvas = np.zeros((height, width), dtype=np.uint8)
            alpha_canvas[y0:y1, x0:x1] = local_alpha
            refined.append(
                (
                    alpha_canvas,
                    {
                        "accepted": True,
                        "sam_iou": round(iou, 6),
                        "area_ratio": round(area_ratio, 6),
                        "expanded_bbox": list(expanded),
                    },
                )
            )
    finally:
        del model
        _release_cuda()
    return refined, {
        "model": BIREF_REPO,
        "revision": BIREF_REVISION,
        "weight_sha256": BIREF_SHA256,
        "process_size": process_size,
        "candidate_count": len(masks),
        "accepted_count": sum(bool(item[1].get("accepted")) for item in refined),
        "seconds": round(time.perf_counter() - started, 3),
    }


def run_semantic_proposals(
    image_rgb: np.ndarray,
    *,
    device: str = "cuda",
    dino_threshold: float = 0.05,
    biref_process_size: int = 1536,
    progress: Callable[[str], None] = print,
) -> SemanticResult:
    """Detect meaningful assets, box-prompt SAM2, then refine product boundaries."""

    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Semantic input must be uint8 RGB.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA semantic pass requested but unavailable.")
    device_index = 0 if device == "cuda" else -1
    image = Image.fromarray(image_rgb, "RGB")
    progress("  V5 PRO: Grounding DINO đang kiểm kê sản phẩm, logo và biểu tượng...")
    detections, dino_report = _run_dino(image, device=device_index, threshold=dino_threshold)
    policies = [(_semantic_policy(item, image.size), item) for item in detections]
    plausible = [item for (accepted, _), item in policies if accepted]
    deferred = [(item, policy) for (accepted, policy), item in policies if not accepted]
    # A resource cap prevents a corrupt detector response from exhausting RAM,
    # but excess detections and all geometry-rejected detections remain in the
    # ledger as explicit unresolved review work.
    processable = plausible[:256]
    overflow = plausible[256:]
    progress(f"  V5 PRO: SAM 2.1 đang gọt {len(processable)} vùng theo khung...")
    masks_and_scores, sam_report = _run_sam_boxes(image, processable, device=device)
    raw_masks = [item[0] for item in masks_and_scores]
    progress("  V5 PRO: BiRefNet HR-matting đang làm mịn biên sản phẩm...")
    refined, biref_report = _refine_with_birefnet(
        image_rgb,
        raw_masks,
        [item["bbox"] for item in processable],
        device=device,
        process_size=biref_process_size,
    )
    proposals: list[DetectedProposal] = []
    height, width = image_rgb.shape[:2]
    for index, (detection, sam_item, refined_item) in enumerate(
        zip(processable, masks_and_scores, refined, strict=True), 1
    ):
        raw_mask, sam_score = sam_item
        alpha_canvas, refine_report = refined_item
        label = detection["label"]
        kind = _kind_from_label(label)
        _, policy = _semantic_policy(detection, (width, height))
        proposal_id = f"SEMANTIC_{index:04d}"
        if alpha_canvas is None or not np.any(alpha_canvas):
            refinement_reason = str(refine_report.get("reason") or "no usable alpha").strip()
            ambiguity_reasons = [f"semantic refinement unavailable: {refinement_reason}"]
            proposals.append(
                DetectedProposal(
                    ProposalRecord(
                        proposal_id,
                        "grounding_dino_plus_sam2",
                        kind,
                        detection["bbox"],
                        float(detection["score"]),
                        reason="SAM/BiRefNet produced no accepted alpha; manual review required",
                        evidence={
                            "label": label,
                            "alternative_labels": detection.get("alternative_labels", []),
                            "dino_context_bbox": detection.get("dino_context_bbox"),
                            "qr_detector_points": detection.get("qr_detector_points"),
                            "dino_score": detection["score"],
                            "sam_iou_score": sam_score,
                            "refinement": refine_report,
                            "auto_extractable": False,
                            "auto_confirmable": False,
                            "requires_manual_review": True,
                            "semantic_ambiguity_reasons": ambiguity_reasons,
                            "semantic_policy": policy,
                        },
                    )
                )
            )
            continue
        bbox = _bbox_from_alpha(alpha_canvas)
        x0, y0, x1, y1 = bbox
        crop = AlphaCrop(x0, y0, alpha_canvas[y0:y1, x0:x1].copy())
        confidence = max(0.0, min(1.0, 0.45 * detection["score"] + 0.55 * sam_score))
        # A rejected BiRefNet refinement still carries a useful, bounded SAM
        # fallback mask.  Materialise it so the reviewer can inspect/edit it,
        # but never let the mere presence of alpha turn it into an automatic
        # semantic truth.  ``auto_extractable`` means "a candidate layer can
        # be built"; ``auto_confirmable`` is the separate safety decision.
        refinement_accepted = refine_report.get("accepted") is not False
        refinement_reason = str(refine_report.get("reason") or "").strip()
        ambiguity_reasons = (
            []
            if refinement_accepted
            else [
                "Semantic matte refinement was rejected"
                + (f": {refinement_reason}" if refinement_reason else "")
            ]
        )
        proposals.append(
            DetectedProposal(
                ProposalRecord(
                    proposal_id,
                    (
                        "grounding_dino_plus_sam2_birefnet"
                        if refinement_accepted
                        else "grounding_dino_plus_sam2_birefnet_review"
                    ),
                    kind,
                    bbox,
                    confidence,
                    reason=(ambiguity_reasons[0] if ambiguity_reasons else None),
                    evidence={
                        "label": label,
                        "alternative_labels": detection.get("alternative_labels", []),
                        "dino_bbox": list(detection["bbox"]),
                        "dino_context_bbox": detection.get("dino_context_bbox"),
                        "qr_detector_points": detection.get("qr_detector_points"),
                        "dino_score": detection["score"],
                        "sam_iou_score": sam_score,
                        "refinement": refine_report,
                        "auto_extractable": True,
                        "auto_confirmable": refinement_accepted,
                        "requires_manual_review": not refinement_accepted,
                        "semantic_ambiguity_reasons": ambiguity_reasons,
                        "semantic_policy": policy,
                    },
                ),
                mask_hint=crop,
            )
        )
    next_index = len(processable) + 1
    for detection, policy in deferred:
        proposals.append(
            DetectedProposal(
                ProposalRecord(
                    f"SEMANTIC_{next_index:04d}",
                    "grounding_dino_geometry_deferred",
                    _kind_from_label(detection["label"]),
                    detection["bbox"],
                    detection["score"],
                    reason="; ".join(policy["reasons"]),
                    evidence={
                        "label": detection["label"],
                        "alternative_labels": detection.get("alternative_labels", []),
                        "dino_context_bbox": detection.get("dino_context_bbox"),
                        "qr_detector_points": detection.get("qr_detector_points"),
                        "dino_score": detection["score"],
                        "auto_extractable": False,
                        "semantic_policy": policy,
                    },
                )
            )
        )
        next_index += 1
    for detection in overflow:
        proposals.append(
            DetectedProposal(
                ProposalRecord(
                    f"SEMANTIC_{next_index:04d}",
                    "grounding_dino_overflow",
                    _kind_from_label(detection["label"]),
                    detection["bbox"],
                    detection["score"],
                    reason="Detector safety queue above 256 items; retained for manual review",
                    evidence={
                        "label": detection["label"],
                        "alternative_labels": detection.get("alternative_labels", []),
                        "auto_extractable": False,
                    },
                )
            )
        )
        next_index += 1
    return SemanticResult(
        proposals,
        {
            "grounding_dino": dino_report,
            "sam2_box_prompt": sam_report,
            "birefnet_hr_matting": biref_report,
            "proposal_count": len(proposals),
            "masked_proposal_count": sum(item.mask_hint is not None for item in proposals),
            "refinement_review_count": sum(
                isinstance(item.record.evidence.get("refinement"), dict)
                and item.record.evidence["refinement"].get("accepted") is False
                for item in proposals
            ),
            "geometry_deferred_review_count": len(deferred),
            "overflow_review_count": len(overflow),
        },
    )


def _bbox_from_alpha(alpha: np.ndarray) -> Box:
    ys, xs = np.where(alpha > 0)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
