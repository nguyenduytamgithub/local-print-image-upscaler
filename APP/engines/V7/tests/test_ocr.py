from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


V7_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7_DIR))

from v7lib.ocr import OCRPipeline, _RawSpan  # noqa: E402
from v7lib.types import OCRObservation  # noqa: E402


class TesseractTsvRegressionTests(unittest.TestCase):
    def test_literal_unmatched_quote_does_not_consume_following_tsv_row(self) -> None:
        header = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
            "left\ttop\twidth\theight\tconf\ttext\n"
        )
        # Tesseract emits TSV, not RFC-CSV. The unmatched quote is recognized
        # artwork content and must not turn the next physical row into one field.
        stdout = header + (
            '5\t1\t1\t1\t1\t1\t10\t10\t80\t20\t93\tSALE "HOT\n'
            "5\t1\t1\t1\t2\t1\t12\t42\t88\t20\t91\t35.000đ\n"
        )
        completed = SimpleNamespace(stdout=stdout, stderr="", returncode=0)
        image = np.full((80, 140, 3), 245, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "tesseract.exe"
            executable.touch()
            (root / "vie.traineddata").touch()
            pipeline = OCRPipeline(
                Path("unused-model-root"),
                tessdata_dir=root,
                tesseract_executable=executable,
            )
            with patch("v7lib.ocr.subprocess.run", return_value=completed) as run:
                spans = pipeline._run_tesseract_variant(
                    image,
                    name="original",
                    scale=1.0,
                    source_size=(140, 80),
                )

        self.assertEqual(
            [span.observation.text for span in spans],
            ['SALE "HOT', "35.000đ"],
        )
        self.assertEqual([span.bbox for span in spans], [(10, 10, 90, 30), (12, 42, 100, 62)])
        run.assert_called_once()

    def test_missing_vietnamese_tessdata_never_claims_a_tesseract_vie_vote(self) -> None:
        stdout = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
            "left\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t10\t10\t80\t20\t95\tDO GIA DUNG\n"
        )
        completed = SimpleNamespace(stdout=stdout, stderr="", returncode=0)
        image = np.full((60, 120, 3), 245, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "tesseract.exe"
            executable.touch()
            pipeline = OCRPipeline(
                Path("unused-model-root"),
                tessdata_dir=root,
                tesseract_executable=executable,
            )
            with patch("v7lib.ocr.subprocess.run", return_value=completed):
                spans = pipeline._run_tesseract_variant(
                    image,
                    name="original",
                    scale=1.0,
                    source_size=(120, 60),
                )

        self.assertFalse(
            any(span.observation.engine == "tesseract_vie" for span in spans),
            "English fallback must not masquerade as an independent Vietnamese vote.",
        )


class GreenGateSafetyTests(unittest.TestCase):
    def test_repeated_paddle_augmentations_cannot_hide_a_weak_independent_vote(self) -> None:
        bbox = (0, 0, 120, 32)
        polygon = [(0, 0), (120, 0), (120, 32), (0, 32)]
        cluster = [
            _RawSpan(
                bbox,
                polygon,
                OCRObservation("paddle_ppocrv6_medium", variant, "SALE HOT", 0.99),
            )
            for variant in ("original", "clahe", "lanczos_unsharp")
        ]
        cluster.append(
            _RawSpan(
                bbox,
                polygon,
                OCRObservation("tesseract_vie", "original", "SALE HOT", 0.50),
            )
        )

        region = OCRPipeline._region_from_cluster(cluster, 1)

        self.assertEqual(
            region.status,
            "yellow",
            "Every independent engine must clear the green confidence gate itself.",
        )


if __name__ == "__main__":
    unittest.main()
