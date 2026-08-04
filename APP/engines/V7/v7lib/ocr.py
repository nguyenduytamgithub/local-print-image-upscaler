"""Multi-pass Vietnamese OCR with an independent Tesseract vote.

PP-OCRv6 supplies the primary geometry and recognition.  Tesseract is kept as
an independent engine: agreement between multiple augmentations of one neural
network is not treated as independent evidence.  OCR confidence is evidence,
never permission to change the user's wording.
"""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .types import OCRObservation, TextRegion


_SPACE_RE = re.compile(r"\s+")
_HAS_CONTENT_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]")
_CRITICAL_RE = re.compile(
    r"(?:\d|₫|đ(?:\b|/)|%|\b(?:sđt|đt|tel|hotline|sku|mã|giá|km)\b)",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class _RawSpan:
    bbox: tuple[int, int, int, int]
    polygon: list[tuple[int, int]]
    observation: OCRObservation


def clean_ocr_text(value: str) -> str:
    """Keep semantic Unicode distinctions while removing OCR whitespace noise."""

    return unicodedata.normalize("NFC", _SPACE_RE.sub(" ", str(value)).strip())


def is_critical_text(value: str) -> bool:
    return bool(_CRITICAL_RE.search(clean_ocr_text(value)))


def _clip_box(
    box: Iterable[float | int], width: int, height: int, inverse_scale: float
) -> tuple[int, int, int, int]:
    values = list(box)
    if len(values) != 4:
        raise ValueError("OCR box must contain four values.")
    x0, y0, x1, y1 = (int(round(float(v) * inverse_scale)) for v in values)
    x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
    y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
    return x0, y0, x1, y1


def _polygon_from_box(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = box
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    a_area = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    b_area = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = a_area + b_area - intersection
    return intersection / union if union else 0.0


def _same_line_box(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    overlap_y = max(0, min(ay1, by1) - max(ay0, by0))
    min_h = max(1, min(ay1 - ay0, by1 - by0))
    centre_distance = abs((ax0 + ax1) - (bx0 + bx1)) / 2.0
    max_w = max(1, ax1 - ax0, bx1 - bx0)
    width_ratio = min(max(1, ax1 - ax0), max(1, bx1 - bx0)) / max_w
    return overlap_y / min_h >= 0.72 and centre_distance <= max_w * 0.20 and width_ratio >= 0.55


def _median_box(spans: list[_RawSpan]) -> tuple[int, int, int, int]:
    boxes = np.asarray([span.bbox for span in spans], dtype=np.float64)
    return tuple(int(round(value)) for value in np.median(boxes, axis=0))  # type: ignore[return-value]


class OCRPipeline:
    """Run bounded OCR passes and merge them into conservative text regions."""

    def __init__(
        self,
        models_root: Path,
        *,
        tessdata_dir: Path | None = None,
        tesseract_executable: Path | None = None,
    ) -> None:
        self.models_root = Path(models_root)
        self.tessdata_dir = Path(tessdata_dir) if tessdata_dir else None
        self.tesseract_executable = tesseract_executable or self._find_tesseract()
        self.tesseract_vie_available = bool(
            self.tesseract_executable
            and self.tesseract_executable.is_file()
            and self.tessdata_dir
            and (self.tessdata_dir / "vie.traineddata").is_file()
        )
        self._paddle = None

    @staticmethod
    def _find_tesseract() -> Path | None:
        discovered = shutil.which("tesseract")
        candidates = [
            Path(discovered) if discovered else None,
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
        return next((item for item in candidates if item and item.is_file()), None)

    def _build_paddle(self):
        if self._paddle is not None:
            return self._paddle
        model_dir = self.models_root / "paddlex" / "official_models"
        detection = model_dir / "PP-OCRv6_medium_det"
        recognition = model_dir / "PP-OCRv6_medium_rec"
        for required in (detection, recognition):
            if not (required / "inference.json").is_file() or not (
                required / "inference.pdiparams"
            ).is_file():
                raise RuntimeError(
                    f"Missing verified PP-OCRv6 model: {required}. Run setup_v7.ps1."
                )
        cache_root = self.models_root / "paddlex"
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_root))
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        # Paddle 3.3.1 on Windows can fail while converting a PIR attribute in
        # the oneDNN path.  Disabling MKL-DNN is deliberate and covered by the
        # setup smoke test; OCR stays isolated from the CUDA restoration env.
        self._paddle = PaddleOCR(
            text_detection_model_dir=str(detection),
            text_recognition_model_dir=str(recognition),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,
        )
        return self._paddle

    @staticmethod
    def _variants(rgb: np.ndarray, maximum: int) -> list[tuple[str, np.ndarray, float]]:
        variants: list[tuple[str, np.ndarray, float]] = [("original", rgb, 1.0)]
        if maximum <= 1:
            return variants
        height, width = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        clahe_rgb = cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)
        variants.append(("clahe", clahe_rgb, 1.0))
        if maximum >= 3:
            factor = 2.0 if max(width, height) * 2 <= 4_000 else max(1.0, 4_000 / max(width, height))
            if factor > 1.05:
                enlarged = cv2.resize(
                    rgb,
                    None,
                    fx=factor,
                    fy=factor,
                    interpolation=cv2.INTER_LANCZOS4,
                )
                softened = cv2.GaussianBlur(enlarged, (0, 0), 0.85)
                unsharp = cv2.addWeighted(enlarged, 1.65, softened, -0.65, 0)
                variants.append(("lanczos_unsharp", unsharp, factor))
        return variants[:maximum]

    def _run_paddle_variant(
        self,
        rgb: np.ndarray,
        *,
        name: str,
        scale: float,
        source_size: tuple[int, int],
    ) -> list[_RawSpan]:
        engine = self._build_paddle()
        results = list(engine.predict(rgb))
        if not results:
            return []
        payload = results[0].json.get("res", {})
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        boxes = payload.get("rec_boxes") or []
        width, height = source_size
        spans: list[_RawSpan] = []
        for text, score, box in zip(texts, scores, boxes, strict=False):
            clean = clean_ocr_text(text)
            if not clean or not _HAS_CONTENT_RE.search(clean):
                continue
            bbox = _clip_box(box, width, height, 1.0 / scale)
            if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
                continue
            spans.append(
                _RawSpan(
                    bbox=bbox,
                    polygon=_polygon_from_box(bbox),
                    observation=OCRObservation(
                        engine="paddle_ppocrv6_medium",
                        variant=name,
                        text=clean,
                        confidence=float(np.clip(score, 0.0, 1.0)),
                    ),
                )
            )
        return spans

    def _run_tesseract_variant(
        self,
        rgb: np.ndarray,
        *,
        name: str,
        scale: float,
        source_size: tuple[int, int],
    ) -> list[_RawSpan]:
        executable = self.tesseract_executable
        if executable is None or not self.tesseract_vie_available or self.tessdata_dir is None:
            return []
        with tempfile.TemporaryDirectory(prefix="v7_tess_") as temporary:
            path = Path(temporary) / "ocr.png"
            Image.fromarray(rgb, "RGB").save(path, format="PNG", compress_level=1)
            command = [
                str(executable),
                str(path),
                "stdout",
                "--psm",
                "11",
                "--oem",
                "1",
            ]
            command.extend(["--tessdata-dir", str(self.tessdata_dir), "-l", "vie"])
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
                return []
        grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        # Tesseract TSV text is not RFC-CSV escaped. A recognized literal quote
        # must not consume later rows as one multi-line quoted field.
        for row in csv.DictReader(
            io.StringIO(completed.stdout),
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        ):
            text = clean_ocr_text(row.get("text") or "")
            try:
                confidence = float(row.get("conf") or -1.0)
            except ValueError:
                confidence = -1.0
            if text and confidence >= 0:
                key = (
                    row.get("block_num", "0"),
                    row.get("par_num", "0"),
                    row.get("line_num", "0"),
                )
                grouped[key].append(row)
        width, height = source_size
        spans: list[_RawSpan] = []
        for words in grouped.values():
            raw_box = (
                min(int(row["left"]) for row in words),
                min(int(row["top"]) for row in words),
                max(int(row["left"]) + int(row["width"]) for row in words),
                max(int(row["top"]) + int(row["height"]) for row in words),
            )
            bbox = _clip_box(raw_box, width, height, 1.0 / scale)
            text = clean_ocr_text(" ".join(row["text"] for row in words))
            confidence = float(
                np.clip(np.mean([float(row["conf"]) for row in words]) / 100.0, 0.0, 1.0)
            )
            if text and _HAS_CONTENT_RE.search(text):
                spans.append(
                    _RawSpan(
                        bbox=bbox,
                        polygon=_polygon_from_box(bbox),
                        observation=OCRObservation(
                            engine="tesseract_vie",
                            variant=name,
                            text=text,
                            confidence=confidence,
                        ),
                    )
                )
        return spans

    @staticmethod
    def _cluster(spans: list[_RawSpan]) -> list[list[_RawSpan]]:
        clusters: list[list[_RawSpan]] = []
        for span in sorted(spans, key=lambda item: (item.bbox[1], item.bbox[0])):
            ranked: list[tuple[float, int]] = []
            for index, cluster in enumerate(clusters):
                box = _median_box(cluster)
                score = _box_iou(span.bbox, box)
                if score >= 0.28 or _same_line_box(span.bbox, box):
                    ranked.append((score, index))
            if ranked:
                clusters[max(ranked)[1]].append(span)
            else:
                clusters.append([span])
        return clusters

    @staticmethod
    def _region_from_cluster(cluster: list[_RawSpan], index: int) -> TextRegion:
        votes: dict[str, float] = defaultdict(float)
        confidences: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for span in cluster:
            text = clean_ocr_text(span.observation.text)
            weight = 1.0 if span.observation.engine.startswith("paddle") else 0.82
            votes[text] += max(0.05, span.observation.confidence) * weight
            confidences[text][span.observation.engine].append(span.observation.confidence)
        selected = max(
            votes,
            key=lambda text: (votes[text], len(confidences[text]), text),
        )
        per_engine_confidence = {
            engine: max(values)
            for engine, values in confidences[selected].items()
            if values
        }
        strongest_confidence = max(per_engine_confidence.values(), default=0.0)
        paddle_confidence = per_engine_confidence.get("paddle_ppocrv6_medium")
        tesseract_confidence = per_engine_confidence.get("tesseract_vie")
        independent_agreement = paddle_confidence is not None and tesseract_confidence is not None
        independent_confidence_gate = bool(
            independent_agreement
            and paddle_confidence >= 0.84
            and tesseract_confidence >= 0.70
        )
        critical = is_critical_text(selected)
        reasons: list[str] = []
        if critical:
            reasons.append("protected_or_numeric_content_requires_approval")
        if not independent_agreement:
            reasons.append("no_independent_engine_agreement")
        elif not independent_confidence_gate:
            reasons.append("independent_engine_confidence_below_green_gate")
        if not independent_confidence_gate:
            reasons.append("recognition_confidence_below_green_gate")
        status = "green"
        if not selected or not _HAS_CONTENT_RE.search(selected) or strongest_confidence < 0.38:
            status = "red"
            reasons.append("unreadable_text_requires_user_entry")
        elif critical or not independent_confidence_gate:
            status = "yellow"
        bbox = _median_box(cluster)
        return TextRegion(
            region_id=f"text_{index:04d}",
            bbox=bbox,
            polygon=_polygon_from_box(bbox),
            observations=[span.observation for span in cluster],
            selected_text=selected,
            proposal=None,
            status=status,
            critical=critical,
            reasons=reasons,
        )

    def run(self, image_path: Path, *, passes: int = 3) -> tuple[list[TextRegion], dict[str, object]]:
        if not 1 <= passes <= 3:
            raise ValueError("OCR passes must be from 1 through 3.")
        with Image.open(image_path) as opened:
            opened.load()
            image = opened.convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
        source_size = image.size
        variants = self._variants(rgb, passes)
        spans: list[_RawSpan] = []
        paddle_counts: dict[str, int] = {}
        tesseract_counts: dict[str, int] = {}
        for name, variant, factor in variants:
            detected = self._run_paddle_variant(
                variant, name=name, scale=factor, source_size=source_size
            )
            spans.extend(detected)
            paddle_counts[name] = len(detected)
        # Two Tesseract views are enough to keep it independent without making
        # a dense catalogue unnecessarily slow.
        for name, variant, factor in variants[:2]:
            detected = self._run_tesseract_variant(
                variant, name=name, scale=factor, source_size=source_size
            )
            spans.extend(detected)
            tesseract_counts[name] = len(detected)
        regions = [
            self._region_from_cluster(cluster, index)
            for index, cluster in enumerate(self._cluster(spans), 1)
        ]
        regions.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
        for index, region in enumerate(regions, 1):
            region.region_id = f"text_{index:04d}"
        report: dict[str, object] = {
            "primary_engine": "PaddleOCR 3.7 / PP-OCRv6 medium",
            "primary_device": "cpu",
            "primary_mkldnn": False,
            "independent_engine": "Tesseract Vietnamese" if self.tesseract_vie_available else None,
            "tesseract_vie_available": self.tesseract_vie_available,
            "pass_count": len(variants),
            "paddle_region_counts": paddle_counts,
            "tesseract_region_counts": tesseract_counts,
            "merged_region_count": len(regions),
            "green_count": sum(item.status == "green" for item in regions),
            "yellow_count": sum(item.status == "yellow" for item in regions),
            "red_count": sum(item.status == "red" for item in regions),
            "policy": (
                "Only exact agreement from independent engines can enter the green gate; "
                "numeric/protected content always requires approval."
            ),
        }
        return regions, report
