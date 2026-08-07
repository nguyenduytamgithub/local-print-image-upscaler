from __future__ import annotations

import hashlib
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image


V7_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7_DIR))

import raster_restore as raster_module  # noqa: E402
import design_repair_v7 as design_module  # noqa: E402
from raster_restore import (  # noqa: E402
    RasterRestoreConfig,
    SRInferenceResult,
    V3SubprocessBackend,
    identity_baseline,
    measure_raster_metrics,
    probe_local_v3_assets,
    restore_raster,
)


def _poster(width: int = 96, height: int = 64) -> np.ndarray:
    image = np.full((height, width, 3), (245, 239, 218), dtype=np.uint8)
    image[10:27, 8 : width - 8] = (190, 24, 24)
    image[35:40, 12 : width - 12] = (18, 96, 45)
    image[45:49, 18 : width - 18] = (226, 146, 16)
    return image


def _spatial_edge_grid(size: int = 128) -> np.ndarray:
    image = np.full((size, size, 3), (104, 112, 120), dtype=np.uint8)
    tile = size // 4
    colours = ((196, 72, 58), (46, 184, 92), (62, 98, 204), (210, 164, 48))
    for row in range(4):
        for column in range(4):
            y0, x0 = row * tile, column * tile
            colour = colours[(row + column) % len(colours)]
            image[y0 + 7 : y0 + tile - 7, x0 + 8 : x0 + tile - 8] = colour
            image[y0 + 12 : y0 + tile - 12, x0 + 4 : x0 + tile - 4] = (
                224,
                210,
                188,
            )
    return image


class _LanczosBackend:
    native_scale = 4
    name = "test-lanczos-x4"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def upscale(self, image: Image.Image, **kwargs: object) -> SRInferenceResult:
        self.calls.append(dict(kwargs))
        output = image.resize(
            (image.width * self.native_scale, image.height * self.native_scale),
            Image.Resampling.LANCZOS,
            reducing_gap=3.0,
        )
        return SRInferenceResult(output, {"fixture": "deterministic-lanczos"})


class _FailingBackend:
    native_scale = 4
    name = "test-failure"

    def upscale(self, image: Image.Image, **kwargs: object) -> SRInferenceResult:
        raise RuntimeError("synthetic CUDA failure")


class _HallucinatingBackend:
    native_scale = 4
    name = "test-checkerboard-hallucination"

    def upscale(self, image: Image.Image, **kwargs: object) -> SRInferenceResult:
        height, width = image.height * 4, image.width * 4
        checker = (np.indices((height, width)).sum(axis=0) % 2 * 255).astype(np.uint8)
        rgb = np.repeat(checker[:, :, None], 3, axis=2)
        return SRInferenceResult(rgb, {"fixture": "invented-checkerboard"})


class DesignReviewBrowserTests(unittest.TestCase):
    def test_initial_gui_prefers_chrome_over_the_registered_browser(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            program_files = Path(raw)
            chrome = program_files / "Google" / "Chrome" / "Application" / "chrome.exe"
            chrome.parent.mkdir(parents=True)
            chrome.write_bytes(b"test executable")
            with (
                mock.patch.dict(
                    design_module.os.environ,
                    {"PROGRAMFILES": str(program_files)},
                    clear=True,
                ),
                mock.patch.object(design_module.shutil, "which", return_value=None),
                mock.patch.object(design_module.subprocess, "Popen") as launch,
                mock.patch.object(design_module.webbrowser, "open") as fallback,
            ):
                opened = design_module.open_review_browser(
                    "http://127.0.0.1:4321/?token=local-test"
                )

            self.assertTrue(opened)
            launch.assert_called_once_with(
                [
                    str(chrome.resolve()),
                    "--new-window",
                    "http://127.0.0.1:4321/?token=local-test",
                ]
            )
            fallback.assert_not_called()


class RasterRestoreBaselineTests(unittest.TestCase):
    def test_x1_identity_is_byte_exact_and_truthful(self) -> None:
        source = _poster()
        config = RasterRestoreConfig(
            scale=1,
            enable_deblur=False,
            enable_denoise=False,
            enable_detail=False,
            enable_sr=False,
        )

        result = restore_raster(source, config)

        self.assertEqual(result.image.mode, "RGB")
        self.assertTrue(np.array_equal(np.asarray(result.image), source))
        self.assertEqual(result.report["status"], "IDENTITY")
        self.assertFalse(result.report["used_neural_prediction"])
        notice = str(result.report["truth_notice"]).casefold()
        self.assertIn("cannot prove", notice)
        self.assertIn("absent from the source", notice)

    def test_x1_can_run_native_x4_sr_then_guarded_back_downsample(self) -> None:
        source = _poster()
        backend = _LanczosBackend()
        config = RasterRestoreConfig(
            scale=1,
            enable_deblur=False,
            enable_denoise=False,
            enable_detail=False,
            enable_sr=True,
            enable_sr_x1=True,
            tile=512,
            overlap=128,
        )

        result = restore_raster(source, config, sr_backend=backend)

        self.assertEqual(result.image.size, (source.shape[1], source.shape[0]))
        self.assertEqual(len(backend.calls), 1)
        self.assertTrue(result.report["super_resolution"]["attempted"])
        self.assertTrue(result.report["super_resolution"]["accepted"])
        self.assertTrue(result.report["used_neural_prediction"])
        self.assertEqual(result.report["status"], "RESTORED_WITH_LOCAL_SR_PREDICTION")

    def test_x1_does_not_call_backend_without_enable_sr_x1(self) -> None:
        source = _poster()
        config = RasterRestoreConfig(
            scale=1,
            enable_deblur=False,
            enable_denoise=False,
            enable_detail=False,
            enable_sr=True,
            enable_sr_x1=False,
        )

        result = restore_raster(source, config, sr_backend=_FailingBackend())

        self.assertEqual(result.report["status"], "IDENTITY")
        self.assertFalse(result.report["super_resolution"]["attempted"])

    def test_identity_baseline_preserves_alpha_and_resizes_once(self) -> None:
        rgb = _poster(40, 28)
        alpha = np.linspace(0, 255, 40, dtype=np.uint8)[None, :].repeat(28, axis=0)
        source = np.dstack((rgb, alpha))

        output = identity_baseline(source, 2)

        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(output.size, (80, 56))
        self.assertEqual(np.asarray(output).shape, (56, 80, 4))

    def test_invalid_tile_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RasterRestoreConfig(tile=256, overlap=160)

    def test_metrics_detect_added_noise(self) -> None:
        clean = _poster(128, 80)
        generator = np.random.default_rng(17)
        noisy = np.clip(
            clean.astype(np.int16) + generator.normal(0, 14, clean.shape),
            0,
            255,
        ).astype(np.uint8)

        clean_metrics = measure_raster_metrics(clean)
        noisy_metrics = measure_raster_metrics(noisy)

        self.assertGreater(noisy_metrics["noise_sigma"], clean_metrics["noise_sigma"])

    def test_sub_byte_sr_texture_on_flat_baseline_has_finite_noise_ratio(self) -> None:
        baseline = np.full((48, 72, 3), 128, dtype=np.uint8)
        candidate = baseline.copy()
        candidate[::8, ::8] = 129

        comparison = raster_module._comparison(baseline, candidate)

        self.assertTrue(np.isfinite(comparison["noise_ratio"]))
        self.assertLess(float(comparison["noise_ratio"]), 5.0)


class RasterRestoreGuardrailTests(unittest.TestCase):
    def test_spatial_sharpness_whole_image_enhancement_passes(self) -> None:
        source = _spatial_edge_grid()
        baseline = cv2.GaussianBlur(source, (0, 0), 1.15)
        candidate = source.copy()
        config = RasterRestoreConfig(scale=1)

        accepted, guard = raster_module._guard_sr(
            source,
            baseline,
            candidate,
            config,
        )

        spatial = guard["spatial_sharpness_coverage"]
        self.assertTrue(accepted, guard)
        self.assertTrue(spatial["applicable"])
        self.assertEqual(spatial["eligible_tile_count"], 16)
        self.assertEqual(spatial["non_softened_coverage"], 1.0)
        self.assertGreater(spatial["sharpened_coverage"], 0.9)
        self.assertEqual(len(spatial["tiles"]), 16)
        self.assertIn("softened_below_ratio", spatial["thresholds"])
        self.assertIn("median_ratio", spatial)
        self.assertIn("p10_ratio", spatial)
        self.assertIn("ignored_by_reason", spatial)
        self.assertIn("noise_sigma", spatial["tiles"][0]["source"])
        self.assertIn("edge_threshold", spatial["tiles"][0]["source"])
        self.assertGreater(spatial["tiles"][0]["mapped_support_pixel_count"], 0)

    def test_spatial_sharpness_one_local_gain_and_distributed_softening_rejects(self) -> None:
        source = _spatial_edge_grid()
        baseline = cv2.GaussianBlur(source, (0, 0), 1.15)
        tile = source.shape[0] // 4
        config = RasterRestoreConfig(scale=1)

        for sigma in (0.6, 0.8, 1.0):
            with self.subTest(sigma=sigma):
                candidate = cv2.GaussianBlur(baseline, (0, 0), sigma)
                candidate[:tile, :tile] = source[:tile, :tile]
                accepted, guard = raster_module._guard_sr(
                    source,
                    baseline,
                    candidate,
                    config,
                )

                spatial = guard["spatial_sharpness_coverage"]
                self.assertFalse(accepted)
                self.assertTrue(spatial["applicable"])
                self.assertGreaterEqual(spatial["sharpened_tile_count"], 1)
                self.assertLess(
                    spatial["median_ratio"],
                    spatial["thresholds"]["minimum_median_ratio"],
                )
                self.assertIn(
                    "median_spatial_ratio_below_threshold",
                    spatial["failure_reasons"],
                )
                self.assertGreater(guard["comparison"]["sharpness_ratio"], 0.70)
                self.assertIn(
                    "sr_spatial_sharpness_coverage_too_patchy",
                    guard["reasons"],
                )

    def test_spatial_sharpness_flat_image_is_not_eligible_or_false_failed(self) -> None:
        source = np.full((128, 128, 3), 128, dtype=np.uint8)
        baseline = source.copy()
        candidate = source.copy()
        config = RasterRestoreConfig(scale=1)

        accepted, guard = raster_module._guard_sr(
            source,
            baseline,
            candidate,
            config,
        )

        spatial = guard["spatial_sharpness_coverage"]
        self.assertTrue(accepted, guard)
        self.assertFalse(spatial["applicable"])
        self.assertEqual(spatial["eligible_tile_count"], 0)
        self.assertEqual(spatial["non_softened_coverage"], 1.0)
        self.assertNotIn("sr_spatial_sharpness_coverage_too_patchy", guard["reasons"])

    def test_spatial_sharpness_noise_only_flat_tiles_are_not_edges(self) -> None:
        generator = np.random.default_rng(904)
        noise = generator.normal(0.0, 3.0, (128, 128, 1))
        source = np.clip(128.0 + noise, 0, 255).astype(np.uint8)
        source = np.repeat(source, 3, axis=2)
        baseline = source.copy()
        candidate = cv2.GaussianBlur(source, (0, 0), 0.8)

        spatial = raster_module._spatial_sharpness_coverage(
            source,
            baseline,
            candidate,
        )

        self.assertTrue(spatial["passed"], spatial)
        self.assertFalse(spatial["applicable"])
        self.assertEqual(spatial["eligible_tile_count"], 0)
        self.assertGreaterEqual(
            spatial["ignored_by_reason"].get("too_few_observed_edge_pixels", 0),
            12,
        )

    def test_spatial_sharpness_too_small_source_tiles_are_ignored(self) -> None:
        source = np.asarray(
            [
                [[20, 20, 20], [230, 230, 230]],
                [[230, 230, 230], [20, 20, 20]],
            ],
            dtype=np.uint8,
        )
        baseline = source.copy()
        config = RasterRestoreConfig(scale=1)

        accepted, guard = raster_module._guard_sr(
            source,
            baseline,
            baseline.copy(),
            config,
        )

        spatial = guard["spatial_sharpness_coverage"]
        self.assertTrue(accepted, guard)
        self.assertFalse(spatial["applicable"])
        self.assertEqual(spatial["eligible_tile_count"], 0)
        self.assertTrue(
            all(
                "source_tile_too_small" in tile["eligibility_reasons"]
                for tile in spatial["tiles"]
            )
        )

    def test_near_white_catalog_background_is_not_misclassified_as_new_clipping(self) -> None:
        baseline = np.full((80, 120, 3), 250, dtype=np.uint8)
        baseline[18:62, 22:98] = (226, 142, 28)
        candidate = baseline.copy()
        candidate[np.all(baseline == 250, axis=2)] = 255
        config = RasterRestoreConfig(scale=1)

        accepted, guard = raster_module._guard_sr(
            baseline,
            baseline,
            candidate,
            config,
        )

        comparison = guard["comparison"]
        self.assertGreater(comparison["clip_fraction_increase"], 0.08)
        self.assertEqual(comparison["newly_clipped_midtone_fraction"], 0.0)
        self.assertNotIn("sr_midtone_information_clipped", guard["reasons"])
        self.assertTrue(accepted, guard)

    def test_new_endpoint_clipping_from_observed_midtone_is_rejected(self) -> None:
        baseline = np.full((80, 120, 3), 128, dtype=np.uint8)
        candidate = baseline.copy()
        candidate[:, :48] = 255
        config = RasterRestoreConfig(scale=1)

        accepted, guard = raster_module._guard_sr(
            baseline,
            baseline,
            candidate,
            config,
        )

        self.assertFalse(accepted)
        self.assertGreater(
            guard["comparison"]["newly_clipped_midtone_fraction"],
            config.sr_max_clip_increase,
        )
        self.assertIn("sr_midtone_information_clipped", guard["reasons"])

    def test_bad_deblur_is_rejected_instead_of_forced(self) -> None:
        source = _poster()
        invented = np.zeros_like(source)
        invented[:, ::2] = 255
        config = RasterRestoreConfig(
            scale=1,
            enable_deblur=True,
            enable_denoise=False,
            enable_detail=False,
            enable_sr=False,
        )

        with mock.patch.object(raster_module, "_deblur_candidate", return_value=invented):
            result = restore_raster(source, config)

        stage = result.report["classical_stages"][0]
        self.assertTrue(stage["applied"])
        self.assertFalse(stage["accepted"])
        self.assertEqual(stage["reason"], "guard_rejected_fallback")
        reasons = stage["guard"]["reasons"]
        self.assertIn("halo_or_overshoot_excursion", reasons)
        self.assertIn("noise_amplified", reasons)
        self.assertTrue(np.array_equal(np.asarray(result.image), source))
        self.assertEqual(result.report["status"], "IDENTITY")

    def test_bounded_deblur_can_improve_a_blurred_edge(self) -> None:
        source = np.zeros((80, 144, 3), dtype=np.uint8)
        source[:, :72] = 36
        source[:, 72:] = 224
        source = cv2.GaussianBlur(source, (0, 0), 2.0)
        config = RasterRestoreConfig(
            scale=1,
            enable_deblur=True,
            enable_denoise=False,
            enable_detail=False,
            enable_sr=False,
        )

        result = restore_raster(source, config)

        stage = result.report["classical_stages"][0]
        self.assertTrue(stage["accepted"], stage)
        comparison = stage["guard"]["comparison"]
        self.assertGreaterEqual(float(comparison["sharpness_ratio"]), 1.0)
        self.assertLessEqual(
            float(comparison["local_excursion_p99"]),
            config.stage_max_excursion_p99,
        )

    def test_sr_failure_falls_back_to_working_lanczos_baseline(self) -> None:
        source = _poster()
        config = RasterRestoreConfig(
            scale=4,
            enable_deblur=False,
            enable_denoise=False,
            enable_detail=False,
            enable_sr=True,
        )
        baseline = identity_baseline(source, 4)

        result = restore_raster(source, config, sr_backend=_FailingBackend())

        self.assertTrue(np.array_equal(np.asarray(result.image), np.asarray(baseline)))
        sr = result.report["super_resolution"]
        self.assertTrue(sr["attempted"])
        self.assertFalse(sr["accepted"])
        self.assertEqual(sr["fallback"], "working_lanczos_baseline")
        self.assertEqual(sr["error_type"], "RuntimeError")

    def test_implausible_sr_detail_is_rejected_by_guardrails(self) -> None:
        source = _poster()
        config = RasterRestoreConfig(
            scale=4,
            enable_deblur=False,
            enable_denoise=False,
            enable_detail=False,
            enable_sr=True,
        )
        baseline = identity_baseline(source, 4)

        result = restore_raster(source, config, sr_backend=_HallucinatingBackend())

        self.assertTrue(np.array_equal(np.asarray(result.image), np.asarray(baseline)))
        guard = result.report["super_resolution"]["guard"]
        self.assertFalse(guard["passed"])
        self.assertIn("sr_noise_implausibly_high", guard["reasons"])
        self.assertIn("sr_halo_or_overshoot_excursion", guard["reasons"])
        self.assertIn("sr_roundtrip_inconsistent_with_source", guard["reasons"])
        self.assertFalse(result.report["used_neural_prediction"])

    def test_valid_backend_gets_12gb_tile_contract_and_is_accepted(self) -> None:
        source = _poster()
        backend = _LanczosBackend()
        config = RasterRestoreConfig(
            scale=4,
            enable_deblur=False,
            enable_denoise=False,
            enable_detail=False,
            enable_sr=True,
            tile=512,
            overlap=128,
            model_key="swin2sr-fidelity",
        )

        first = restore_raster(source, config, sr_backend=backend)
        second = restore_raster(source, config, sr_backend=_LanczosBackend())

        self.assertEqual(first.image.size, (source.shape[1] * 4, source.shape[0] * 4))
        self.assertTrue(first.report["super_resolution"]["accepted"])
        self.assertTrue(first.report["used_neural_prediction"])
        self.assertEqual(backend.calls[0]["tile"], 512)
        self.assertEqual(backend.calls[0]["overlap"], 128)
        self.assertEqual(backend.calls[0]["model_key"], "swin2sr-fidelity")
        self.assertTrue(np.array_equal(np.asarray(first.image), np.asarray(second.image)))
        self.assertEqual(
            first.report["output_pixel_sha256"], second.report["output_pixel_sha256"]
        )


class LocalV3BackendContractTests(unittest.TestCase):
    def test_asset_probe_verifies_local_manifest_without_importing_torch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".venv" / "Scripts").mkdir(parents=True)
            (root / "models").mkdir()
            (root / ".venv" / "Scripts" / "python.exe").write_bytes(b"fixture")
            (root / "upsize_ai_v3.py").write_text("# fixture", encoding="utf-8")
            (root / "upsize_ai_v3_master.py").write_text("# fixture", encoding="utf-8")
            lines: list[str] = []
            for index, filename in enumerate(raster_module.V3_MODEL_FILES.values(), 1):
                payload = f"checkpoint-{index}".encode("ascii")
                (root / "models" / filename).write_bytes(payload)
                lines.append(f"{hashlib.sha256(payload).hexdigest()}  {filename}")
            (root / "MODEL_SHA256SUMS.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

            assets = probe_local_v3_assets(root, verify_hashes=True)

            self.assertEqual(assets.missing, ())
            self.assertTrue(assets.single_available)
            self.assertTrue(assets.master_available)
            self.assertEqual(set(assets.verified_sha256), set(raster_module.V3_MODEL_FILES))

            (root / "upsize_ai_v3_master.py").unlink()
            single_only = probe_local_v3_assets(root, verify_hashes=True)
            self.assertTrue(single_only.single_available)
            self.assertFalse(single_only.master_available)

    def test_v3_command_is_local_tiled_and_explicit(self) -> None:
        assets = probe_local_v3_assets(V7_DIR.parent / "V3", verify_hashes=False)
        backend = V3SubprocessBackend(assets, mode="single", verify_selected_model=False)
        command = backend.build_command(
            Path("source.png"),
            Path("target.png"),
            model_key="swin2sr-fidelity",
            tile=512,
            overlap=128,
            dtype="auto",
        )

        self.assertIn(str(assets.python), command)
        self.assertIn(str(assets.single_engine), command)
        self.assertEqual(command[command.index("--tile") + 1], "512")
        self.assertEqual(command[command.index("--overlap") + 1], "128")
        self.assertEqual(command[command.index("--model") + 1], "swin2sr-fidelity")
        self.assertEqual(command[command.index("--dtype") + 1], "auto")
        self.assertIn("--force", command)


class DesignRepairRasterIntegrationTests(unittest.TestCase):
    def test_cli_defaults_to_auto_print_faithful_restore(self) -> None:
        argv = [
            "design_repair_v7.py",
            "source.png",
            "1",
            "bundle",
            "--models-root",
            "models",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = design_module.parse_args()

        self.assertEqual(args.raster_restore, "auto")

    def test_auto_x1_uses_single_swin_backend_and_keeps_dimensions(self) -> None:
        source_array = _poster()
        source = Image.fromarray(source_array, "RGB")
        backend = _LanczosBackend()

        result, report, warnings = design_module.restore_clean_raster_base(
            source,
            scale=1,
            mode="auto",
            v3_python=None,
            v3_engine=None,
            icc_profile=b"",
            sr_backend=backend,
        )

        self.assertEqual(result.size, source.size)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["model_key"], "swin2sr-fidelity")
        self.assertEqual(report["integration"]["profile"], "PRINT_FAITHFUL")
        self.assertEqual(report["integration"]["backend_mode"], "single")
        self.assertFalse(report["integration"]["gan_or_three_model_fusion"])
        self.assertTrue(report["super_resolution"]["accepted"])
        self.assertEqual(warnings, [])

    def test_backend_factory_is_forced_to_verified_single_model_mode(self) -> None:
        backend = _LanczosBackend()
        v3_engine = Path("C:/fixture/V3/upsize_ai_v3_master.py")
        v3_python = Path("C:/fixture/V3/.venv/Scripts/python.exe")

        with mock.patch.object(
            design_module.V3SubprocessBackend,
            "from_local",
            return_value=backend,
        ) as factory:
            _result, report, _warnings = design_module.restore_clean_raster_base(
                Image.fromarray(_poster(), "RGB"),
                scale=1,
                mode="auto",
                v3_python=v3_python,
                v3_engine=v3_engine,
                icc_profile=b"",
            )

        factory.assert_called_once_with(
            v3_engine.resolve().parent,
            mode="single",
            verify_selected_model=True,
            python=v3_python,
        )
        self.assertEqual(report["integration"]["model_key"], "swin2sr-fidelity")

    def test_off_x1_preserves_legacy_identity_path(self) -> None:
        source_array = _poster()

        result, report, warnings = design_module.restore_clean_raster_base(
            Image.fromarray(source_array, "RGB"),
            scale=1,
            mode="off",
            v3_python=None,
            v3_engine=None,
            icc_profile=b"",
        )

        self.assertTrue(np.array_equal(np.asarray(result), source_array))
        self.assertEqual(report["status"], "DISABLED")
        self.assertEqual(report["legacy_upscale"]["pipeline"], "identity_clean_source_x1")
        self.assertEqual(warnings, [])

    def test_backend_failure_is_nonfatal_and_emits_clear_fallback_warning(self) -> None:
        source = Image.fromarray(_poster(), "RGB")

        result, report, warnings = design_module.restore_clean_raster_base(
            source,
            scale=1,
            mode="auto",
            v3_python=None,
            v3_engine=None,
            icc_profile=b"",
            sr_backend=_FailingBackend(),
        )

        self.assertEqual(result.size, source.size)
        self.assertFalse(report["super_resolution"]["accepted"])
        self.assertTrue(warnings)
        warning = " ".join(warnings).casefold()
        self.assertIn("classical/lanczos fallback", warning)
        self.assertIn("synthetic cuda failure", warning)

    def test_main_keeps_ocr_and_source_geometry_ahead_of_raster_restore(self) -> None:
        source = inspect.getsource(design_module.main)

        self.assertLess(
            source.index("ocr.run(source_asset"),
            source.index("restore_clean_raster_base("),
        )
        self.assertLess(
            source.index("render_regions(\n        clean_source_image"),
            source.index("restore_clean_raster_base("),
        )
        self.assertGreaterEqual(
            source.count('"raster_restore": raster_restore_report'),
            2,
        )


if __name__ == "__main__":
    unittest.main()
