from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from v5lib.segment import detect_text_regions

from .schema import AlphaCrop, Box, ProposalRecord, clip_box


@dataclass(slots=True)
class DetectedProposal:
    record: ProposalRecord
    mask_hint: AlphaCrop | None = None
    polygon: list[tuple[int, int]] = field(default_factory=list)


@dataclass(slots=True)
class InventoryResult:
    proposals: list[DetectedProposal]
    detector_reports: dict[str, Any]

    def records(self) -> list[ProposalRecord]:
        return [item.record for item in self.proposals]


def _box_iou(first: Box, second: Box) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    first_area = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    second_area = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _deduplicate_boxes(items: Iterable[tuple[Box, float]], iou_threshold: float) -> list[tuple[Box, float]]:
    accepted: list[tuple[Box, float]] = []
    for box, score in sorted(items, key=lambda item: item[1], reverse=True):
        if any(_box_iou(box, current) >= iou_threshold for current, _ in accepted):
            continue
        accepted.append((box, score))
    return sorted(accepted, key=lambda item: (item[0][1], item[0][0]))


def detect_text_inventory(
    image_path: Path,
    canvas_size: tuple[int, int],
) -> tuple[list[DetectedProposal], dict[str, Any]]:
    regions, report = detect_text_regions(image_path)
    proposals: list[DetectedProposal] = []
    for index, region in enumerate(regions, 1):
        bbox = clip_box(region.bbox, canvas_size)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        confidence = max(0.0, min(1.0, float(region.confidence) / 100.0))
        proposals.append(
            DetectedProposal(
                ProposalRecord(
                    proposal_id=f"OCR_{index:04d}",
                    source="tesseract_vie_geometry",
                    kind_hint="text",
                    bbox=bbox,
                    confidence=confidence,
                    evidence={"recognized_text": region.text},
                )
            )
        )
    report = dict(report)
    report["proposal_count"] = len(proposals)
    return proposals, report


def detect_qr_inventory(image_rgb: np.ndarray) -> tuple[list[DetectedProposal], dict[str, Any]]:
    height, width = image_rgb.shape[:2]
    detector = cv2.QRCodeDetector()
    proposals: list[DetectedProposal] = []
    decoded_values: list[str] = []
    try:
        found, decoded, points, _ = detector.detectAndDecodeMulti(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        )
    except cv2.error:
        found, decoded, points = False, (), None
    if found and points is not None:
        for index, (value, vertices) in enumerate(zip(decoded, points, strict=False), 1):
            polygon = [
                (max(0, min(width, int(round(x)))), max(0, min(height, int(round(y)))))
                for x, y in np.asarray(vertices).reshape(-1, 2)
            ]
            xs = [item[0] for item in polygon]
            ys = [item[1] for item in polygon]
            bbox = clip_box((min(xs), min(ys), max(xs) + 1, max(ys) + 1), (width, height))
            mask = np.zeros((bbox[3] - bbox[1], bbox[2] - bbox[0]), dtype=np.uint8)
            local = np.asarray([(x - bbox[0], y - bbox[1]) for x, y in polygon], dtype=np.int32)
            cv2.fillPoly(mask, [local], 255)
            proposals.append(
                DetectedProposal(
                    ProposalRecord(
                        proposal_id=f"QR_{index:03d}",
                        source="opencv_qrcode_detector",
                        kind_hint="qr",
                        bbox=bbox,
                        confidence=1.0 if value else 0.8,
                        evidence={"decoded": value or None},
                    ),
                    AlphaCrop(bbox[0], bbox[1], mask),
                    polygon,
                )
            )
            decoded_values.append(value)
    return proposals, {
        "engine": "OpenCV QRCodeDetector.detectAndDecodeMulti",
        "proposal_count": len(proposals),
        "decoded_count": sum(bool(value) for value in decoded_values),
    }


def detect_frame_inventory(image_rgb: np.ndarray) -> tuple[list[DetectedProposal], dict[str, Any]]:
    """Find rectangular design boundaries without pretending they are final masks."""

    height, width = image_rgb.shape[:2]
    canvas_area = height * width
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, L2gradient=True)
    # Closing joins one-pixel gaps in antialiased rounded rectangles while the
    # proposal remains geometry only; ownership is decided later.
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[Box, float]] = []
    for contour in contours:
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter < 1:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = box_width * box_height
        if area < canvas_area * 0.004 or area > canvas_area * 0.92:
            continue
        if box_width < width * 0.18 or box_height < height * 0.025:
            continue
        approximation = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
        if not 4 <= len(approximation) <= 24:
            continue
        contour_area = abs(float(cv2.contourArea(contour)))
        rectangularity = min(1.0, contour_area / max(1.0, area))
        # Thin outlines have low filled rectangularity, while filled cards can
        # be close to one. Both are useful layout proposals.
        score = 0.55 + 0.35 * max(rectangularity, 1.0 - rectangularity)
        candidates.append(((x, y, x + box_width, y + box_height), min(0.95, score)))
    accepted = _deduplicate_boxes(candidates, 0.90)
    proposals = [
        DetectedProposal(
            ProposalRecord(
                proposal_id=f"FRAME_{index:04d}",
                source="opencv_contour_layout",
                kind_hint="frame",
                bbox=box,
                confidence=score,
            )
        )
        for index, (box, score) in enumerate(accepted, 1)
    ]
    return proposals, {
        "engine": "OpenCV Canny+RETR_LIST+approxPolyDP",
        "raw_contour_count": len(contours),
        "proposal_count": len(proposals),
    }


def _cluster_line_segments(
    segments: list[tuple[int, int, int, int]],
    *,
    horizontal: bool,
) -> list[tuple[int, int, int, int]]:
    if not segments:
        return []
    key_index = 1 if horizontal else 0
    ordered = sorted(segments, key=lambda item: (item[key_index], item[0 if horizontal else 1]))
    groups: list[list[tuple[int, int, int, int]]] = []
    for segment in ordered:
        coordinate = (segment[key_index] + segment[key_index + 2]) / 2.0
        if groups:
            last_coordinate = np.median(
                [(item[key_index] + item[key_index + 2]) / 2.0 for item in groups[-1]]
            )
            if abs(coordinate - last_coordinate) <= 3:
                groups[-1].append(segment)
                continue
        groups.append([segment])
    merged: list[tuple[int, int, int, int]] = []
    for group in groups:
        if horizontal:
            y = int(round(float(np.median([(item[1] + item[3]) / 2.0 for item in group]))))
            merged.append((min(item[0] for item in group), y, max(item[2] for item in group), y + 1))
        else:
            x = int(round(float(np.median([(item[0] + item[2]) / 2.0 for item in group]))))
            merged.append((x, min(item[1] for item in group), x + 1, max(item[3] for item in group)))
    return merged


def detect_line_inventory(image_rgb: np.ndarray) -> tuple[list[DetectedProposal], dict[str, Any]]:
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 170, L2gradient=True)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, int(min(width, height) * 0.04)),
        minLineLength=max(20, int(min(width, height) * 0.10)),
        maxLineGap=max(4, int(min(width, height) * 0.012)),
    )
    horizontal: list[tuple[int, int, int, int]] = []
    vertical: list[tuple[int, int, int, int]] = []
    if raw is not None:
        # OpenCV 4 commonly returns N x 1 x 4, while OpenCV 5 may return
        # N x 4.  The semantic result is identical.
        for x0, y0, x1, y1 in np.asarray(raw).reshape(-1, 4):
            dx, dy = abs(int(x1) - int(x0)), abs(int(y1) - int(y0))
            if dx >= max(1, dy * 8):
                horizontal.append((min(x0, x1), min(y0, y1), max(x0, x1) + 1, max(y0, y1) + 1))
            elif dy >= max(1, dx * 8):
                vertical.append((min(x0, x1), min(y0, y1), max(x0, x1) + 1, max(y0, y1) + 1))
    merged = _cluster_line_segments(horizontal, horizontal=True) + _cluster_line_segments(
        vertical, horizontal=False
    )
    proposals: list[DetectedProposal] = []
    for index, box in enumerate(sorted(merged, key=lambda item: (item[1], item[0])), 1):
        bbox = clip_box((box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1), (width, height))
        proposals.append(
            DetectedProposal(
                ProposalRecord(
                    proposal_id=f"LINE_{index:04d}",
                    source="opencv_hough_line",
                    kind_hint="line",
                    bbox=bbox,
                    confidence=0.72,
                )
            )
        )
    return proposals, {
        "engine": "OpenCV HoughLinesP",
        "raw_segment_count": 0 if raw is None else int(len(raw)),
        "proposal_count": len(proposals),
    }


def detect_edge_components(image_rgb: np.ndarray) -> tuple[list[DetectedProposal], dict[str, Any]]:
    """Account for residual visual islands, including tiny ones, in the ledger."""

    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 140, L2gradient=True)
    joined = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    proposals: list[DetectedProposal] = []
    rejected_noise = 0
    for component in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[component])
        bbox = (x, y, x + box_width, y + box_height)
        # Nothing disappears: low-evidence islands are explicit rejected
        # records with a reason, not silently dropped proposals.
        if area < 4:
            status = "rejected"
            reason = "edge island below four connected pixels; recorded as sensor/antialias noise"
            confidence = 0.05
            rejected_noise += 1
        else:
            status = "unresolved"
            reason = None
            confidence = min(0.70, 0.18 + np.log1p(area) / 18.0)
        proposals.append(
            DetectedProposal(
                ProposalRecord(
                    proposal_id=f"EDGE_{component:05d}",
                    source="opencv_residual_edge_component",
                    kind_hint="micro_detail" if area < 36 else "unknown",
                    bbox=bbox,
                    confidence=float(confidence),
                    status=status,  # type: ignore[arg-type]
                    reason=reason,
                    evidence={"dilated_edge_pixels": area},
                )
            )
        )
    return proposals, {
        "engine": "OpenCV Canny+connectedComponentsWithStats",
        "proposal_count": len(proposals),
        "rejected_noise_count": rejected_noise,
    }


def build_inventory(
    image_rgb: np.ndarray,
    image_path: Path,
    *,
    include_text: bool = True,
    include_edge_components: bool = True,
) -> InventoryResult:
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Inventory input must be uint8 RGB.")
    height, width = image_rgb.shape[:2]
    all_proposals: list[DetectedProposal] = []
    reports: dict[str, Any] = {}
    if include_text:
        detected, report = detect_text_inventory(image_path, (width, height))
        all_proposals.extend(detected)
        reports["text"] = report
    for name, detector in (
        ("qr", detect_qr_inventory),
        ("frames", detect_frame_inventory),
        ("lines", detect_line_inventory),
    ):
        detected, report = detector(image_rgb)
        all_proposals.extend(detected)
        reports[name] = report
    if include_edge_components:
        detected, report = detect_edge_components(image_rgb)
        all_proposals.extend(detected)
        reports["edge_components"] = report
    # Prefixes are detector-specific, but verify globally anyway so a future
    # detector cannot corrupt the completeness ledger.
    ids = [item.record.proposal_id for item in all_proposals]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Inventory detectors produced duplicate proposal ids.")
    reports["total_proposals"] = len(all_proposals)
    reports["unresolved_proposals"] = sum(
        item.record.status == "unresolved" for item in all_proposals
    )
    reports["rejected_with_reason"] = sum(
        item.record.status == "rejected" and bool(item.record.reason) for item in all_proposals
    )
    return InventoryResult(all_proposals, reports)
