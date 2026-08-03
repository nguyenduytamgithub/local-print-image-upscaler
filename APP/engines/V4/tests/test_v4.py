from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as element_tree
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[4]
APP_DIR = PROJECT_ROOT / "APP"
V4_DIR = APP_DIR / "engines" / "V4"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(V4_DIR))

from upscale_cli import UserError, parse_command
from upsize_vector_v4 import (
    APP_DIR as V4_APP_DIR,
    PDF_POINTS_PER_MM,
    V4Error,
    aggregate_comparative_crop_metrics,
    build_parser,
    choose_print_geometry,
    comparative_restoration_metrics,
    compare_v3_v4_paths,
    decide_rendered_candidate,
    discover_v3_native,
    find_gmic,
    find_pinned_tool,
    find_resvg,
    find_scribus,
    gaussian_kernel_spec,
    inspect_and_normalize_source,
    local_name,
    normalize_svg,
    sha256_file,
    uniform_restoration_rgb,
    validate_pdf_page_geometry,
    verify_rendered_restoration,
    write_uniform_restoration,
)


class SourceNormalizationTests(unittest.TestCase):
    def test_direct_engine_never_retags_rgba_source_profile_as_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "source.png"
            normalized = directory / "normalized.png"
            image = Image.new("RGBA", (2, 1))
            image.putdata([(255, 0, 0, 255), (0, 0, 0, 0)])
            image.save(source, icc_profile=b"invalid-source-icc")

            report = inspect_and_normalize_source(source, normalized)
            with Image.open(normalized) as result:
                result.load()
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(result.getpixel((1, 0)), (255, 255, 255))
                self.assertTrue(result.info.get("icc_profile"))
            self.assertEqual(
                report["colour_conversion"],
                "invalid_icc_fallback_to_srgb_assumption",
            )
            self.assertTrue(report["alpha_composited_on_white"])


class CommandParsingTests(unittest.TestCase):
    def test_engine_accepts_explicit_v3_comparison_baseline(self) -> None:
        args = build_parser().parse_args(
            [
                "source.png",
                "4",
                "output",
                "--name",
                "poster",
                "--raster-base",
                "candidate.png",
                "--v3-baseline",
                "native.png",
            ]
        )
        self.assertEqual(args.v3_baseline, Path("native.png"))

    def test_print_command_accepts_physical_options(self) -> None:
        mode, token, scale, allow_huge, options = parse_command(
            [
                "print",
                "poster.png",
                "4",
                "--width-mm",
                "3000",
                "--bleed-mm",
                "5",
                "--profile-name",
                "Printer Profile",
            ]
        )
        self.assertEqual(mode, "V4_PRINT")
        self.assertEqual(token, "poster.png")
        self.assertEqual(scale, 4.0)
        self.assertFalse(allow_huge)
        self.assertEqual(options["width_mm"], 3000.0)
        self.assertEqual(options["bleed_mm"], 5.0)
        self.assertEqual(options["profile_name"], "Printer Profile")

    def test_vector_alias_and_decimal_comma(self) -> None:
        mode, _, scale, allow_huge, options = parse_command(
            ["vector", "poster.png", "2,5", "--allow-huge"]
        )
        self.assertEqual(mode, "V4_VECTOR")
        self.assertEqual(scale, 2.5)
        self.assertTrue(allow_huge)
        self.assertIsNone(options["width_mm"])

    def test_v4_options_are_rejected_for_v2_v3(self) -> None:
        with self.assertRaises(UserError):
            parse_command(["high", "poster.png", "4", "--width-mm", "3000"])

    def test_physical_option_bounds_are_enforced(self) -> None:
        with self.assertRaises(UserError):
            parse_command(["print", "poster.png", "4", "--width-mm", "9"])
        with self.assertRaises(UserError):
            parse_command(["print", "poster.png", "4", "--bleed-mm", "101"])


class PrintGeometryTests(unittest.TestCase):
    def test_normal_page_stays_at_one_to_one(self) -> None:
        geometry = choose_print_geometry((4000, 2000), 100.0, 3000.0)
        self.assertEqual(geometry["print_scale"], "1:1")
        self.assertAlmostEqual(geometry["page_width_mm"], 3000.0)
        self.assertAlmostEqual(geometry["page_height_mm"], 1500.0)

    def test_very_wide_page_uses_smallest_safe_denominator(self) -> None:
        geometry = choose_print_geometry((4000, 2000), 100.0, 9000.0)
        self.assertEqual(geometry["print_scale"], "1:2")
        self.assertAlmostEqual(geometry["page_width_mm"], 4500.0)
        self.assertAlmostEqual(geometry["page_height_mm"], 2250.0)

    def test_very_tall_page_uses_height_for_denominator(self) -> None:
        geometry = choose_print_geometry((1000, 3000), 100.0, 3000.0)
        self.assertEqual(geometry["intended_height_mm"], 9000.0)
        self.assertEqual(geometry["print_scale"], "1:2")
        self.assertAlmostEqual(geometry["page_width_mm"], 1500.0)
        self.assertAlmostEqual(geometry["page_height_mm"], 4500.0)

    def test_finished_bleed_is_included_in_scale_decision(self) -> None:
        geometry = choose_print_geometry((1000, 1000), 100.0, 4990.0, bleed_mm=6.0)
        self.assertEqual(geometry["print_scale"], "1:2")
        self.assertAlmostEqual(geometry["page_bleed_mm"], 3.0)
        self.assertAlmostEqual(geometry["page_outer_width_mm"], 2501.0)
        self.assertLessEqual(geometry["page_outer_width_mm"], 5000.0)

    def test_denominator_can_grow_beyond_ten(self) -> None:
        geometry = choose_print_geometry((1000, 1000), 100.0, 100000.0, bleed_mm=100.0)
        self.assertEqual(geometry["print_scale"], "1:21")
        self.assertLessEqual(geometry["page_outer_width_mm"], 5000.0)


class PdfPageGeometryTests(unittest.TestCase):
    @staticmethod
    def rectangle(width_mm: float, height_mm: float, bleed_mm: float = 0.0) -> list[float]:
        return [
            -bleed_mm * PDF_POINTS_PER_MM,
            -bleed_mm * PDF_POINTS_PER_MM,
            (width_mm + bleed_mm) * PDF_POINTS_PER_MM,
            (height_mm + bleed_mm) * PDF_POINTS_PER_MM,
        ]

    def test_expected_trim_and_bleed_are_measured_from_pdf_boxes(self) -> None:
        trim = self.rectangle(300.0, 150.0)
        outer = self.rectangle(300.0, 150.0, bleed_mm=5.0)
        report = validate_pdf_page_geometry(
            outer,
            trim,
            outer,
            expected_page_width_mm=300.0,
            expected_page_height_mm=150.0,
            expected_bleed_mm=5.0,
        )
        self.assertEqual(report["trim_size_mm"], [300.0, 150.0])
        self.assertEqual(report["bleed_size_mm"], [310.0, 160.0])
        self.assertEqual(report["actual_bleed_mm"]["left"], 5.0)
        self.assertTrue(report["geometry_matches_expected"])

    def test_oversized_real_pdf_box_is_rejected(self) -> None:
        oversized = self.rectangle(5001.0, 100.0)
        with self.assertRaisesRegex(V4Error, "exceeds 5000 mm"):
            validate_pdf_page_geometry(oversized, oversized, oversized)

    def test_incorrect_exported_bleed_is_rejected(self) -> None:
        trim = self.rectangle(300.0, 150.0)
        actual_outer = self.rectangle(300.0, 150.0, bleed_mm=4.0)
        with self.assertRaisesRegex(V4Error, "BleedBox"):
            validate_pdf_page_geometry(
                actual_outer,
                trim,
                actual_outer,
                expected_page_width_mm=300.0,
                expected_page_height_mm=150.0,
                expected_bleed_mm=5.0,
            )


class ToolDiscoveryTests(unittest.TestCase):
    def test_bundled_binaries_win_over_path(self) -> None:
        with (
            patch(
                "upsize_vector_v4.shutil.which",
                return_value=r"C:\\system\\tool.exe",
            ) as which,
            patch("upsize_vector_v4.Path.is_file", return_value=True),
        ):
            self.assertEqual(
                find_gmic(),
                (V4_APP_DIR / "shared" / "tools" / "gmic" / "gmic-4.0.2-cli-win64" / "gmic.exe").resolve(),
            )
            self.assertEqual(
                find_resvg(),
                (V4_APP_DIR / "shared" / "tools" / "resvg" / "0.47.0" / "resvg.exe").resolve(),
            )
            self.assertEqual(
                find_scribus(),
                (V4_APP_DIR / "shared" / "tools" / "scribus" / "1.6.6" / "Scribus.exe").resolve(),
            )
            which.assert_not_called()

    def test_path_fallback_must_report_the_pinned_version(self) -> None:
        bundled = Path(r"C:\\bundle\\missing.exe")
        fallback = Path(r"C:\\system\\resvg.exe")
        completed = subprocess.CompletedProcess(
            [str(fallback), "--version"],
            0,
            stdout="resvg 0.47.0",
            stderr="",
        )
        with (
            patch("upsize_vector_v4.Path.is_file", side_effect=[False, True]),
            patch("upsize_vector_v4.subprocess.run", return_value=completed),
        ):
            found = find_pinned_tool(
                bundled,
                [fallback],
                "resvg 0.47.0",
                ["--version"],
                r"(?<![\d.])0\.47\.0(?![\d.])",
            )
        self.assertEqual(found, fallback.resolve())

    def test_wrong_path_version_is_rejected(self) -> None:
        bundled = Path(r"C:\\bundle\\missing.exe")
        fallback = Path(r"C:\\system\\resvg.exe")
        completed = subprocess.CompletedProcess(
            [str(fallback), "--version"],
            0,
            stdout="resvg 0.46.0",
            stderr="",
        )
        with (
            patch("upsize_vector_v4.Path.is_file", side_effect=[False, True]),
            patch("upsize_vector_v4.subprocess.run", return_value=completed),
            self.assertRaisesRegex(V4Error, "Incompatible candidates"),
        ):
            find_pinned_tool(
                bundled,
                [fallback],
                "resvg 0.47.0",
                ["--version"],
                r"(?<![\d.])0\.47\.0(?![\d.])",
            )


class SetupScriptTests(unittest.TestCase):
    def test_scribus_install_directory_is_quoted(self) -> None:
        setup = (V4_DIR / "setup_v4.ps1").read_text(encoding="utf-8")
        self.assertIn('$scribusDirArgument = \'/DIR="{0}"\' -f $scribusDir', setup)
        self.assertIn('$scribusDirArgument)', setup)

    def test_scribus_exporter_scales_imported_svg_to_cover_the_print_page(self) -> None:
        exporter = (V4_DIR / "scribus_export_pdfx4.py").read_text(encoding="utf-8")
        self.assertIn("scale_factor = max(outer_width / initial_width", exporter)
        self.assertIn("scribus.sizeObject(placed_width, placed_height, artwork_name)", exporter)
        self.assertIn('"placement": {', exporter)


class VectorMasterTests(unittest.TestCase):
    def test_normalized_master_contains_paths_and_no_raster(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            raw_svg = directory / "raw.svg"
            master_svg = directory / "master.svg"
            raw_svg.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20">'
                '<path fill="#ff0000" d="M 0 0 L 40 0 L 40 20 L 0 20 Z"/>'
                "</svg>",
                encoding="utf-8",
            )

            report = normalize_svg(
                raw_svg,
                master_svg,
                source_size=(20, 10),
                page_width_mm=200.0,
                page_height_mm=100.0,
                intended_width_mm=200.0,
                print_scale_denominator=1,
                title="Unit test",
            )

            root = element_tree.parse(master_svg).getroot()
            tags = [local_name(node.tag) for node in root.iter()]
            self.assertEqual(report["path_count"], 1)
            self.assertEqual(report["embedded_image_count"], 0)
            self.assertIn("path", tags)
            self.assertNotIn("image", tags)
            self.assertEqual(root.get("data-v4-mode"), "full-vector")
            self.assertEqual(root.get("data-print-scale"), "1:1")

    def test_embedded_raster_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            raw_svg = directory / "raw.svg"
            master_svg = directory / "master.svg"
            raw_svg.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2">'
                '<image width="2" height="2" href="data:image/png;base64,AAAA"/>'
                "</svg>",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(V4Error, "raster"):
                normalize_svg(
                    raw_svg,
                    master_svg,
                    source_size=(1, 1),
                    page_width_mm=10.0,
                    page_height_mm=10.0,
                    intended_width_mm=10.0,
                    print_scale_denominator=1,
                    title="Reject raster",
                )

    def test_hybrid_master_declares_one_embedded_ai_layer(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            raw_svg = directory / "raw.svg"
            raster = directory / "base.png"
            master_svg = directory / "master.svg"
            raw_svg.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20">'
                '<path fill="#ff0000" d="M 0 0 L 40 0 L 40 20 L 0 20 Z"/>'
                "</svg>",
                encoding="utf-8",
            )
            Image.new("RGB", (40, 20), "#ff0000").save(raster)

            report = normalize_svg(
                raw_svg,
                master_svg,
                source_size=(20, 10),
                page_width_mm=200.0,
                page_height_mm=100.0,
                intended_width_mm=200.0,
                print_scale_denominator=1,
                title="Hybrid test",
                raster_base=raster,
                vector_opacity=0.0,
            )

            root = element_tree.parse(master_svg).getroot()
            self.assertEqual(root.get("data-v4-mode"), "hybrid-deep-ai-vector")
            self.assertEqual(report["embedded_image_count"], 1)
            self.assertEqual(report["path_count"], 1)
            self.assertEqual(report["vector_opacity"], 0.0)
            self.assertEqual(report["raster_layer"]["pixel_size"], [40, 20])

    def test_fallback_master_labels_v3_in_root_description_layer_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            raw_svg = directory / "raw.svg"
            raster = directory / "fallback.png"
            master_svg = directory / "master.svg"
            raw_svg.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20">'
                '<path fill="#ff0000" d="M 0 0 L 40 0 L 40 20 L 0 20 Z"/>'
                "</svg>",
                encoding="utf-8",
            )
            Image.new("RGB", (40, 20), "#ff0000").save(raster)
            provenance = {
                "selection": "v3_baseline_fallback",
                "selected_pipeline": "V3_HIGH_NATIVE_BASELINE",
                "v3_source_sha256": "a" * 64,
                "candidate_rejected": True,
            }

            report = normalize_svg(
                raw_svg,
                master_svg,
                source_size=(20, 10),
                page_width_mm=200.0,
                page_height_mm=100.0,
                intended_width_mm=200.0,
                print_scale_denominator=1,
                title="Editable print master",
                raster_base=raster,
                vector_opacity=0.0,
                raster_provenance=provenance,
            )

            root = element_tree.parse(master_svg).getroot()
            description = next(
                node.text for node in root if local_name(node.tag) == "desc"
            )
            metadata_text = next(
                node.text for node in root if local_name(node.tag) == "metadata"
            )
            metadata = json.loads(metadata_text)
            image_node = next(node for node in root.iter() if local_name(node.tag) == "image")
            self.assertEqual(
                root.get("data-v4-mode"),
                "hybrid-v3-baseline-fallback-vector",
            )
            self.assertEqual(root.get("data-raster-selection"), "v3_baseline_fallback")
            self.assertIn("V3 baseline fallback", description)
            self.assertNotIn("two-pass", description)
            self.assertEqual(image_node.get("id"), "V3_BASELINE_FALLBACK_PRINT_LAYER")
            self.assertEqual(metadata["mode"], "hybrid-v3-baseline-fallback-vector")
            self.assertEqual(metadata["raster_provenance"], provenance)
            self.assertEqual(report["mode"], "hybrid-v3-baseline-fallback-vector")
            self.assertEqual(report["raster_layer"]["selection"], "v3_baseline_fallback")

    def test_v3_usm_metadata_declares_deep_diagnostic_not_visible(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            raw_svg = directory / "raw.svg"
            raster = directory / "fusion.png"
            master_svg = directory / "master.svg"
            raw_svg.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 10">'
                '<path fill="#00aa00" d="M 0 0 L 20 0 L 20 10 L 0 10 Z"/>'
                "</svg>",
                encoding="utf-8",
            )
            Image.new("RGB", (40, 20), "#00aa00").save(raster)
            provenance = {
                "selection": "v4_uniform_v3_usm",
                "spatially_uniform": True,
                "v3_weight": 1.0,
                "deep_weight": 0.0,
                "deep_visible": False,
                "raw_deep_role": "diagnostic_ablation_only_not_visible",
                "unsharp_amount": 0.06,
                "sigma_native": 1.5,
            }
            report = normalize_svg(
                raw_svg,
                master_svg,
                source_size=(20, 10),
                page_width_mm=200.0,
                page_height_mm=100.0,
                intended_width_mm=200.0,
                print_scale_denominator=1,
                title="Fusion master",
                raster_base=raster,
                vector_opacity=0.0,
                raster_provenance=provenance,
            )
            root = element_tree.parse(master_svg).getroot()
            image_node = next(node for node in root.iter() if local_name(node.tag) == "image")
            self.assertEqual(
                root.get("data-v4-mode"),
                "hybrid-v4-uniform-v3-usm-vector",
            )
            self.assertEqual(image_node.get("id"), "V4_UNIFORM_V3_USM_PRINT_LAYER")
            description = next(
                node.text for node in root if local_name(node.tag) == "desc"
            )
            self.assertIn("validated V3 AI master", description)
            self.assertIn("not visible", description)
            self.assertEqual(report["raster_layer"]["provenance"]["deep_weight"], 0.0)
            self.assertFalse(report["raster_layer"]["provenance"]["deep_visible"])


class ComparativeQualityTests(unittest.TestCase):
    @staticmethod
    def artwork() -> np.ndarray:
        image = np.full((180, 260, 3), 232, dtype=np.uint8)
        cv2.rectangle(image, (18, 18), (105, 155), (35, 70, 180), -1)
        cv2.circle(image, (175, 85), 55, (220, 45, 35), -1)
        cv2.line(image, (5, 170), (250, 12), (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(
            image,
            "V4",
            (110, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (10, 100, 30),
            3,
            cv2.LINE_AA,
        )
        return image

    def test_identical_candidate_is_safe_but_cannot_claim_improvement(self) -> None:
        baseline = self.artwork()
        report = comparative_restoration_metrics(baseline, baseline.copy())
        self.assertTrue(report["retention_passed"])
        self.assertFalse(report["meaningfully_sharper"])
        self.assertFalse(report["claim_v4_better"])
        self.assertEqual(report["decision"], "fallback_v3")

    def test_mild_multi_metric_sharpening_passes_without_ringing(self) -> None:
        baseline = self.artwork()
        blurred = cv2.GaussianBlur(baseline, (0, 0), 1.0)
        candidate = np.clip(
            baseline.astype(np.float32) * 1.05 - blurred.astype(np.float32) * 0.05,
            0,
            255,
        ).astype(np.uint8)
        report = comparative_restoration_metrics(baseline, candidate)
        self.assertTrue(report["edge_strength_passed"])
        self.assertTrue(report["energy_detail_passed"])
        self.assertTrue(report["retention_passed"])
        self.assertTrue(report["claim_v4_better"])
        self.assertEqual(report["decision"], "use_v4")

    def test_blur_falls_back_to_v3(self) -> None:
        baseline = self.artwork()
        candidate = cv2.GaussianBlur(baseline, (0, 0), 1.2)
        report = comparative_restoration_metrics(baseline, candidate)
        self.assertFalse(report["meaningfully_sharper"])
        self.assertFalse(report["claim_v4_better"])
        self.assertEqual(report["decision"], "fallback_v3")

    def test_noisy_false_sharpness_is_rejected_by_retention_gate(self) -> None:
        baseline = self.artwork()
        noise = np.random.default_rng(7).normal(0.0, 18.0, baseline.shape)
        candidate = np.clip(baseline.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        report = comparative_restoration_metrics(baseline, candidate)
        self.assertTrue(report["energy_detail_passed"])
        self.assertFalse(report["retention_passed"])
        self.assertFalse(report["claim_v4_better"])
        self.assertTrue(report["retention_failures"])

    def test_energy_only_gain_cannot_satisfy_improvement_gate(self) -> None:
        baseline = np.empty((256, 256, 3), dtype=np.uint8)
        baseline[:, :128] = 126
        baseline[:, 128:] = 130
        candidate = baseline.astype(np.int16)
        y, x = np.indices((256, 256))
        checker = np.where((x + y) % 2 == 0, 1, -1)
        flat_region = x < 110
        candidate[flat_region] += np.repeat(checker[:, :, None], 3, axis=2)[
            flat_region
        ]
        candidate = np.clip(candidate, 0, 255).astype(np.uint8)

        report = comparative_restoration_metrics(baseline, candidate)
        self.assertTrue(report["retention_passed"])
        self.assertFalse(report["edge_strength_passed"])
        self.assertTrue(report["energy_detail_passed"])
        self.assertFalse(report["meaningfully_sharper"])
        self.assertFalse(report["claim_v4_better"])

    def test_crop_boundaries_cannot_create_false_sharpness(self) -> None:
        baseline_values = [100, 106, 112, 118]
        candidate_values = [99, 105, 113, 119]
        samples = []
        reports = []
        baseline_mosaic = np.zeros((128, 128, 3), dtype=np.uint8)
        candidate_mosaic = np.zeros_like(baseline_mosaic)
        for index, (baseline_value, candidate_value) in enumerate(
            zip(baseline_values, candidate_values)
        ):
            row, column = divmod(index, 2)
            y, x = row * 64, column * 64
            baseline_crop = np.full((64, 64, 3), baseline_value, dtype=np.uint8)
            candidate_crop = np.full((64, 64, 3), candidate_value, dtype=np.uint8)
            baseline_mosaic[y : y + 64, x : x + 64] = baseline_crop
            candidate_mosaic[y : y + 64, x : x + 64] = candidate_crop
            samples.append(
                {
                    "index": index,
                    "box": [x, y, x + 64, y + 64],
                    "v3": baseline_crop,
                    "deep": candidate_crop,
                }
            )
            reports.append(
                comparative_restoration_metrics(baseline_crop, candidate_crop)
            )

        contaminated = comparative_restoration_metrics(
            baseline_mosaic,
            candidate_mosaic,
        )
        independent = aggregate_comparative_crop_metrics(reports, samples)
        self.assertTrue(contaminated["claim_v4_better"])
        self.assertFalse(independent["claim_v4_better"])
        self.assertEqual(
            independent["crop_pass_fractions"]["retention_and_both_groups"],
            0.0,
        )
        self.assertFalse(independent["aggregation"]["crop_join_seams_in_metrics"])

    def test_spatial_majority_rejects_a_few_sharp_crops(self) -> None:
        baseline = self.artwork()
        blurred = cv2.GaussianBlur(baseline, (0, 0), 1.0)
        sharpened = np.clip(
            baseline.astype(np.float32) * 1.05 - blurred.astype(np.float32) * 0.05,
            0,
            255,
        ).astype(np.uint8)
        samples = []
        reports = []
        for index in range(4):
            candidate = sharpened if index < 2 else baseline.copy()
            samples.append(
                {
                    "index": index,
                    "box": [index * baseline.shape[1], 0, (index + 1) * baseline.shape[1], baseline.shape[0]],
                    "v3": baseline,
                    "deep": candidate,
                }
            )
            reports.append(comparative_restoration_metrics(baseline, candidate))

        aggregate = aggregate_comparative_crop_metrics(reports, samples)
        self.assertEqual(
            aggregate["crop_pass_fractions"]["retention_and_both_groups"],
            0.5,
        )
        self.assertFalse(aggregate["spatially_consistent"])
        self.assertFalse(aggregate["claim_v4_better"])
        self.assertTrue(aggregate["spatial_failures"])

    def test_deep_must_materially_beat_passing_v3_usm_control(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline_path = directory / "v3.png"
            deep_path = directory / "deep.png"
            baseline = self.artwork()
            blurred = cv2.GaussianBlur(baseline, (0, 0), 1.0)
            deep = np.clip(
                baseline.astype(np.float32) * 1.18 - blurred.astype(np.float32) * 0.18,
                0,
                255,
            ).astype(np.uint8)
            Image.fromarray(baseline).save(baseline_path)
            Image.fromarray(deep).save(deep_path)
            report = compare_v3_v4_paths(baseline_path, deep_path)
            self.assertTrue(report["claim_v4_better"])
            self.assertEqual(report["decision"], "use_v4_uniform_v3_usm_control")
            self.assertEqual(report["selected_deep_weight"], 0.0)
            self.assertEqual(report["selected_v3_weight"], 1.0)
            self.assertEqual(len(report["attempted_restorations"]), 5)
            self.assertTrue(report["attempted_restorations"][0]["spatially_uniform"])
            self.assertFalse(report["deep_ablation"]["selected_uses_deep"])
            self.assertIn("no Deep candidate", report["deep_ablation"]["reason"])
            self.assertFalse(report["raw_deep_diagnostic"]["selection_eligible"])

    def test_path_comparison_uses_v3_native_resolution_not_source_x1(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline_path = directory / "v3.png"
            candidate_path = directory / "v4.png"
            artwork = self.artwork()
            Image.fromarray(artwork).resize((520, 360)).save(baseline_path)
            Image.fromarray(artwork).resize((780, 540)).save(candidate_path)
            report = compare_v3_v4_paths(
                baseline_path,
                candidate_path,
            )
            self.assertEqual(report["native_reference_size"], [520, 360])
            self.assertEqual(report["evaluation_size"], [520, 360])
            self.assertEqual(report["strategy"], "full_frame_at_v3_native_resolution")
            self.assertFalse(report["source_scale_downsample_used"])
            self.assertEqual(report["v3_sha256"], sha256_file(baseline_path))
            self.assertEqual(report["raw_deep_sha256"], sha256_file(candidate_path))

    def test_large_native_comparison_uses_stratified_native_pixel_crops(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline_path = directory / "v3.png"
            candidate_path = directory / "deep.png"
            artwork = self.artwork()
            Image.fromarray(artwork).resize((640, 480)).save(baseline_path)
            Image.fromarray(artwork).resize((1280, 960)).save(candidate_path)
            report = compare_v3_v4_paths(
                baseline_path,
                candidate_path,
                full_frame_max_pixels=1,
                tile_size=64,
                grid_size=3,
            )
            self.assertEqual(
                report["strategy"],
                "independent_stratified_crops_at_v3_native_resolution",
            )
            self.assertEqual(report["native_reference_size"], [640, 480])
            self.assertEqual(report["crop_size"], [64, 64])
            self.assertEqual(report["crop_grid"], [3, 3])
            self.assertEqual(report["evaluation_size"], [64, 64])
            self.assertEqual(report["halo_radius_native"], 5)
            self.assertFalse(report["source_scale_downsample_used"])
            self.assertTrue(report["metrics_computed_per_crop"])
            self.assertFalse(report["crop_join_seams_in_metrics"])
            self.assertLess(report["native_area_coverage"], 1.0)

    def test_uniform_restoration_matches_explicit_rgb_formula(self) -> None:
        y, x = np.indices((48, 64))
        baseline = np.stack(
            ((x * 3 + y) % 256, (x + y * 2) % 256, (x * 2 + y * 3) % 256),
            axis=2,
        ).astype(np.uint8)
        deep = np.clip(baseline.astype(np.int16) + 4, 0, 255).astype(np.uint8)
        actual, kernel = uniform_restoration_rgb(
            baseline,
            deep,
            deep_weight=0.075,
            unsharp_amount=0.06,
            sigma_x=1.5,
            sigma_y=1.0,
        )
        expected_kernel = gaussian_kernel_spec(1.5, 1.0)
        baseline_float = baseline.astype(np.float32)
        blurred = cv2.GaussianBlur(
            baseline_float,
            (expected_kernel["kernel_width"], expected_kernel["kernel_height"]),
            sigmaX=1.5,
            sigmaY=1.0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        expected = np.clip(
            np.rint(
                baseline_float * 0.925
                + deep.astype(np.float32) * 0.075
                + 0.06 * (baseline_float - blurred)
            ),
            0,
            255,
        ).astype(np.uint8)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(kernel, expected_kernel)

    def test_gaussian_halo_matches_full_frame_core_exactly(self) -> None:
        rng = np.random.default_rng(42)
        baseline = rng.integers(20, 236, size=(96, 112, 3), dtype=np.uint8)
        deep = np.clip(baseline.astype(np.int16) + 3, 0, 255).astype(np.uint8)
        full, kernel = uniform_restoration_rgb(
            baseline,
            deep,
            deep_weight=0.025,
            unsharp_amount=0.06,
            sigma_x=1.5,
            sigma_y=1.5,
        )
        radius = kernel["radius_x"]
        left, top, right, bottom = 24, 20, 82, 76
        guarded, _ = uniform_restoration_rgb(
            baseline[top - radius : bottom + radius, left - radius : right + radius],
            deep[top - radius : bottom + radius, left - radius : right + radius],
            deep_weight=0.025,
            unsharp_amount=0.06,
            sigma_x=1.5,
            sigma_y=1.5,
        )
        guarded_core = guarded[radius:-radius, radius:-radius]
        np.testing.assert_array_equal(guarded_core, full[top:bottom, left:right])

    def test_uniform_restoration_writer_scales_non_square_sigma_independently(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline_path = directory / "v3.png"
            deep_path = directory / "deep.png"
            restoration_path = directory / "restoration.png"
            Image.fromarray(self.artwork()).resize((80, 60)).save(baseline_path)
            Image.fromarray(self.artwork()).resize((160, 120)).save(deep_path)
            report = write_uniform_restoration(
                baseline_path,
                deep_path,
                restoration_path,
                (203, 123),
                deep_weight=0.0,
                unsharp_amount=0.06,
                sigma_native=1.5,
                stripe_height=32,
            )
            with Image.open(restoration_path) as restoration:
                self.assertEqual(restoration.size, (203, 123))
            self.assertAlmostEqual(report["proof_scale_x"], 203 / 80)
            self.assertAlmostEqual(report["proof_scale_y"], 123 / 60)
            self.assertAlmostEqual(report["gaussian"]["sigma_x"], 1.5 * 203 / 80)
            self.assertAlmostEqual(report["gaussian"]["sigma_y"], 1.5 * 123 / 60)
            self.assertNotEqual(
                report["gaussian"]["kernel_width"],
                report["gaussian"]["kernel_height"],
            )
            self.assertEqual(report["sha256"], sha256_file(restoration_path))
            self.assertFalse(
                restoration_path.with_suffix(restoration_path.suffix + ".pixels.tmp").exists()
            )

    def test_actual_render_is_rechecked_and_failed_render_forces_fallback(self) -> None:
        pre_render = {
            "claim_v4_better": True,
            "decision": "use_v4_uniform_v3_usm_control",
        }
        accepted = decide_rendered_candidate(
            pre_render,
            {"claim_v4_better": True},
        )
        rejected = decide_rendered_candidate(
            pre_render,
            {"claim_v4_better": False},
        )
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["decision"], "use_v4_uniform_v3_usm_control")
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["decision"], "fallback_v3_after_render_verification")
        self.assertEqual(rejected["reason"], "actual_render_verification_failed")

    def test_encoded_render_failure_is_detected_by_authoritative_native_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline_path = directory / "v3.png"
            deep_path = directory / "deep.png"
            rendered_path = directory / "rendered.png"
            artwork = self.artwork()
            Image.fromarray(artwork).save(baseline_path)
            Image.fromarray(artwork).resize((520, 360)).save(deep_path)
            write_uniform_restoration(
                baseline_path,
                deep_path,
                rendered_path,
                (520, 360),
                deep_weight=0.0,
                unsharp_amount=0.5,
                sigma_native=1.5,
                stripe_height=64,
            )
            verification = verify_rendered_restoration(
                baseline_path,
                rendered_path,
            )
            decision = decide_rendered_candidate(
                {
                    "claim_v4_better": True,
                    "decision": "use_v4_uniform_v3_usm_control",
                },
                verification,
            )
            self.assertTrue(verification["actual_render_verified"])
            self.assertEqual(
                verification["rendered_sha256"],
                sha256_file(rendered_path),
            )
            self.assertFalse(verification["claim_v4_better"])
            self.assertFalse(decision["accepted"])
            self.assertEqual(decision["decision"], "fallback_v3_after_render_verification")

    def test_v3_provenance_resolver_verifies_source_and_native_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "source.png"
            native = directory / "native.png"
            candidate = directory / "candidate.png"
            Image.fromarray(self.artwork()).save(source)
            Image.fromarray(self.artwork()).resize((520, 360)).save(native)
            Image.fromarray(self.artwork()).resize((780, 540)).save(candidate)
            sidecar = candidate.with_suffix(candidate.suffix + ".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "pipeline": "V4_DEEP_RECURSIVE_HAT",
                        "source_sha256": sha256_file(source),
                        "v3_native": str(native),
                        "v3_native_sha256": sha256_file(native),
                    }
                ),
                encoding="utf-8",
            )
            report = discover_v3_native(candidate, sha256_file(source))
            self.assertIsNotNone(report)
            self.assertEqual(report["path"], native.resolve())
            self.assertEqual(report["sha256"], sha256_file(native))


if __name__ == "__main__":
    unittest.main()
