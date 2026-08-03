from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageCms


PROJECT_ROOT = Path(__file__).resolve().parents[4]
APP_DIR = PROJECT_ROOT / "APP"
sys.path.insert(0, str(APP_DIR))

import upscale_cli  # noqa: E402
from upscale_cli import (  # noqa: E402
    HARD_RASTER_MEGAPIXELS,
    V3_ENGINE_FILES,
    UserError,
    canonical_pixel_sha256,
    canonical_source_dpi,
    current_v3_cache_signature,
    find_cached_v3_native,
    find_cached_v4_deep,
    sha256_file,
    stage_input,
    validate_standard_resource_plan,
    validate_v4_resource_plan,
)


class CanonicalInputTests(unittest.TestCase):
    def test_alpha_is_composited_over_white_and_output_is_tagged_srgb(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "alpha.png"
            image = Image.new("RGBA", (2, 1))
            image.putdata([(255, 0, 0, 255), (0, 0, 0, 0)])
            image.save(source, icc_profile=b"not-a-valid-profile")

            staged = stage_input(source, directory)
            with Image.open(staged) as result:
                result.load()
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))
                self.assertEqual(result.getpixel((1, 0)), (255, 255, 255))
                self.assertTrue(result.info.get("icc_profile"))

    def test_exif_orientation_is_applied_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "oriented.jpg"
            image = Image.new("RGB", (3, 2), "red")
            exif = image.getexif()
            exif[274] = 6
            image.save(source, exif=exif)

            staged = stage_input(source, directory)
            with Image.open(staged) as result:
                self.assertEqual(result.size, (2, 3))

    def test_embedded_non_rgb_profile_is_transformed_before_srgb_tagging(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "lab.tif"
            lab_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
            Image.new("LAB", (1, 1), (128, 128, 128)).save(
                source,
                icc_profile=lab_profile,
            )

            staged = stage_input(source, directory)
            with Image.open(staged) as result:
                result.load()
                self.assertEqual(result.mode, "RGB")
                self.assertTrue(result.info.get("icc_profile"))

    def test_valid_source_dpi_is_preserved_in_staged_png(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "dpi.png"
            Image.new("RGB", (30, 15), "blue").save(source, dpi=(300.0, 300.0))

            staged = stage_input(source, directory)
            with Image.open(staged) as result:
                self.assertAlmostEqual(result.info["dpi"][0], 300.0, delta=0.02)
                self.assertAlmostEqual(result.info["dpi"][1], 300.0, delta=0.02)
                natural_width_mm = result.width / result.info["dpi"][0] * 25.4
                self.assertAlmostEqual(natural_width_mm, 2.54, delta=0.001)

    def test_same_pixels_keep_per_job_dpi_while_sharing_pixel_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            first_source = directory / "pixels-96.png"
            second_source = directory / "pixels-300.png"
            pixels = Image.new("RGB", (30, 15), (20, 40, 60))
            pixels.save(first_source, dpi=(96.0, 96.0))
            pixels.save(second_source, dpi=(300.0, 300.0))
            first_job = directory / "job-96"
            second_job = directory / "job-300"
            first_job.mkdir()
            second_job.mkdir()

            first_staged = stage_input(first_source, first_job)
            second_staged = stage_input(second_source, second_job)

            self.assertEqual(
                canonical_pixel_sha256(first_staged),
                canonical_pixel_sha256(second_staged),
            )
            with Image.open(first_staged) as first, Image.open(second_staged) as second:
                self.assertAlmostEqual(first.info["dpi"][0], 96.0, delta=0.02)
                self.assertAlmostEqual(second.info["dpi"][0], 300.0, delta=0.02)
                first_width_mm = first.width / first.info["dpi"][0] * 25.4
                second_width_mm = second.width / second.info["dpi"][0] * 25.4
                self.assertNotAlmostEqual(first_width_mm, second_width_mm, places=2)

    def test_anisotropic_and_malformed_dpi_are_handled_deliberately(self) -> None:
        self.assertEqual(canonical_source_dpi((300.0, 150.0)), (300.0, 300.0))
        self.assertEqual(
            canonical_source_dpi((300.0, 150.0), swap_axes=True),
            (150.0, 150.0),
        )
        self.assertEqual(canonical_source_dpi((300.0, "bad")), (300.0, 300.0))
        self.assertIsNone(canonical_source_dpi((300.0, "bad"), swap_axes=True))
        self.assertIsNone(canonical_source_dpi(("bad", 300.0)))
        self.assertIsNone(canonical_source_dpi((9.0, 300.0)))
        self.assertIsNone(canonical_source_dpi((float("nan"), 300.0)))


class ResourceGuardTests(unittest.TestCase):
    def test_hard_raster_cap_stays_below_pillow_error_threshold(self) -> None:
        pillow_error_threshold_mp = Image.MAX_IMAGE_PIXELS * 2 / 1_000_000
        self.assertLess(HARD_RASTER_MEGAPIXELS, pillow_error_threshold_mp)

    def test_huge_source_is_rejected_before_ai_even_when_output_soft_limit_passes(self) -> None:
        with self.assertRaises(UserError):
            validate_v4_resource_plan(
                (10_000, 10_000),
                (20_000, 20_000),
                full_vector=False,
                allow_huge=False,
            )

    def test_allow_huge_does_not_disable_finite_hard_caps(self) -> None:
        with self.assertRaises(UserError):
            validate_v4_resource_plan(
                (10_000, 10_000),
                (20_000, 20_000),
                full_vector=False,
                allow_huge=True,
            )

    def test_allow_huge_cannot_bypass_standard_output_cap(self) -> None:
        with self.assertRaises(UserError):
            validate_standard_resource_plan(
                (1000, 1000),
                (30_000, 30_000),
                allow_huge=True,
            )


class CacheValidationTests(unittest.TestCase):
    def test_v3_signature_tracks_all_pipeline_python_dependencies(self) -> None:
        names = {path.name for path in V3_ENGINE_FILES}
        self.assertTrue(
            {
                "upsize_ai_v3_master.py",
                "upsize_ai_v3.py",
                "upscale_engine.py",
                "pyramid_fusion.py",
                "blend_outputs.py",
            }.issubset(names)
        )

    def test_changing_pyramid_fusion_invalidates_v3_signature(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            engines = tuple(directory / name for name in ("master.py", "pyramid_fusion.py"))
            models = (directory / "model.pth",)
            for path in (*engines, *models):
                path.write_text(path.name, encoding="utf-8")
            with (
                patch.object(upscale_cli, "V3_ENGINE_FILES", engines),
                patch.object(upscale_cli, "V3_MODEL_FILES", models),
            ):
                current_v3_cache_signature.cache_clear()
                first = current_v3_cache_signature()["signature_sha256"]
                engines[1].write_text("changed", encoding="utf-8")
                current_v3_cache_signature.cache_clear()
                second = current_v3_cache_signature()["signature_sha256"]
                self.assertNotEqual(first, second)
            current_v3_cache_signature.cache_clear()

    def test_canonical_identity_ignores_volatile_container_profile_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            first = directory / "first.png"
            second = directory / "second.png"
            pixels = Image.new("RGB", (3, 2), (20, 40, 60))
            pixels.save(first, icc_profile=b"profile-created-at-time-a")
            pixels.save(second, icc_profile=b"profile-created-at-time-b")
            self.assertNotEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(canonical_pixel_sha256(first), canonical_pixel_sha256(second))

    def test_private_native_without_sidecar_is_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            fake_app = Path(raw_directory)
            native = fake_app / "masters" / "V4_AI_BASE" / "source_NATIVE_x4.png"
            native.parent.mkdir(parents=True)
            Image.new("RGB", (4, 4), "black").save(native)
            with (
                patch.object(upscale_cli, "APP_DIR", fake_app),
                patch.object(
                    upscale_cli,
                    "current_v3_cache_signature",
                    return_value={"signature_sha256": "current"},
                ),
            ):
                self.assertIsNone(find_cached_v3_native("source", "normalized"))

    def test_private_native_cache_is_keyed_by_canonical_pixels_not_container(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            fake_app = Path(raw_directory)
            canonical_identity = "canonical-pixels"
            native = (
                fake_app
                / "masters"
                / "V4_AI_BASE"
                / f"{canonical_identity}_NATIVE_x4.png"
            )
            native.parent.mkdir(parents=True)
            Image.new("RGB", (4, 4), "black").save(native)
            signature = {"signature_sha256": "current"}
            sidecar = {
                "pipeline": "V4_V3_NATIVE_CACHE",
                "source_sha256": "different-container",
                "normalized_stage_sha256": canonical_identity,
                "canonical_pixel_sha256": canonical_identity,
                "v3_cache_signature": signature,
                "native_sha256": sha256_file(native),
            }
            native.with_suffix(native.suffix + ".json").write_text(
                json.dumps(sidecar),
                encoding="utf-8",
            )
            with (
                patch.object(upscale_cli, "APP_DIR", fake_app),
                patch.object(
                    upscale_cli,
                    "current_v3_cache_signature",
                    return_value=signature,
                ),
            ):
                self.assertEqual(
                    find_cached_v3_native("new-container", canonical_identity),
                    native,
                )

    def test_current_signature_can_reuse_legacy_source_named_cache(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            fake_app = Path(raw_directory)
            canonical_identity = "canonical-pixels"
            native = (
                fake_app
                / "masters"
                / "V4_AI_BASE"
                / "old-container_NATIVE_x4.png"
            )
            native.parent.mkdir(parents=True)
            Image.new("RGB", (4, 4), "black").save(native)
            signature = {"signature_sha256": "current"}
            native.with_suffix(native.suffix + ".json").write_text(
                json.dumps(
                    {
                        "pipeline": "V4_V3_NATIVE_CACHE",
                        "source_sha256": "old-container",
                        "normalized_stage_sha256": canonical_identity,
                        "v3_cache_signature": signature,
                        "native_sha256": sha256_file(native),
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(upscale_cli, "APP_DIR", fake_app),
                patch.object(
                    upscale_cli,
                    "current_v3_cache_signature",
                    return_value=signature,
                ),
            ):
                self.assertEqual(
                    find_cached_v3_native("new-container", canonical_identity),
                    native,
                )

    def test_deep_cache_uses_canonical_pixels_with_current_job_provenance_separate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            fake_app = Path(raw_directory)
            canonical_identity = "canonical-pixels"
            native_sha256 = "native-pixels"
            engine = fake_app / "deep_raster_v4.py"
            model = fake_app / "model.pth"
            engine.write_text("engine", encoding="utf-8")
            model.write_bytes(b"model")
            deep = (
                fake_app
                / "masters"
                / "V4_DEEP"
                / f"{canonical_identity}_DEEP_x4.png"
            )
            deep.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "black").save(deep)
            deep.with_suffix(deep.suffix + ".json").write_text(
                json.dumps(
                    {
                        "pipeline": "V4_DEEP_RECURSIVE_HAT",
                        "config_version": upscale_cli.V4_DEEP_CONFIG_VERSION,
                        "source_sha256": "first-container-only-provenance",
                        "canonical_pixel_sha256": canonical_identity,
                        "engine_sha256": sha256_file(engine),
                        "model_sha256": sha256_file(model),
                        "v3_native_sha256": native_sha256,
                        "final_scale": 4.0,
                        "output_sha256": sha256_file(deep),
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(upscale_cli, "APP_DIR", fake_app),
                patch.object(upscale_cli, "V4_DEEP_ENGINE", engine),
                patch.object(upscale_cli, "V4_DEEP_MODEL", model),
            ):
                cached = find_cached_v4_deep(
                    canonical_identity,
                    native_sha256,
                    4.0,
                    legacy_source_sha256="second-container",
                )
            self.assertIsNotNone(cached)
            self.assertEqual(cached[0], deep)


if __name__ == "__main__":
    unittest.main()
