from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


V5_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(V5_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "APP"))

import upscale_cli  # noqa: E402
import layer_engine_v5  # noqa: E402
from v5lib.formats import render_layers  # noqa: E402
from v5lib.geometry import dilate_mask  # noqa: E402
from v5lib.model import LayerSpec  # noqa: E402
from v5lib.restore import restore_background  # noqa: E402


class V5CliTests(unittest.TestCase):
    def test_v5_accepts_source_scale_and_options(self) -> None:
        mode, token, scale, allow_huge, options = upscale_cli.parse_command(
            ["layers", "poster.png", "1", "--max-layers", "18", "--inpaint", "poster"]
        )
        self.assertEqual(mode, "V5_LAYERS")
        self.assertEqual(token, "poster.png")
        self.assertEqual(scale, 1.0)
        self.assertFalse(allow_huge)
        self.assertEqual(options["max_layers"], 18)
        self.assertEqual(options["inpaint"], "poster")

    def test_v5_only_options_are_rejected_by_v3(self) -> None:
        with self.assertRaises(upscale_cli.UserError):
            upscale_cli.parse_command(["high", "poster.png", "4", "--max-layers", "18"])

    def test_v5_hard_canvas_cap_cannot_be_bypassed(self) -> None:
        with self.assertRaises(upscale_cli.UserError):
            upscale_cli.validate_v5_resource_plan(
                (4000, 4000), (24_000, 24_000), allow_huge=True
            )

    def test_v5_resource_plan_accounts_for_foreground_layer_budget(self) -> None:
        small = upscale_cli.validate_v5_resource_plan(
            (256, 256),
            (512, 512),
            allow_huge=False,
            max_layers=4,
        )
        large = upscale_cli.validate_v5_resource_plan(
            (256, 256),
            (512, 512),
            allow_huge=False,
            max_layers=60,
        )
        self.assertGreater(large["estimated_peak_ram_gib"], small["estimated_peak_ram_gib"])
        self.assertGreater(
            large["estimated_working_disk_gib"], small["estimated_working_disk_gib"]
        )

    def test_direct_engine_never_deletes_an_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (8, 8), "white").save(source)
            target = root / "keep-me"
            target.mkdir()
            sentinel = target / "important.txt"
            sentinel.write_text("keep", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "input": source,
                    "output_dir": target,
                    "scale": 1.0,
                    "max_layers": 24,
                    "name": None,
                },
            )()

            with self.assertRaises(SystemExit):
                layer_engine_v5.validate_args(args)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


class RestorationTests(unittest.TestCase):
    def test_poster_restore_is_byte_identical_outside_declared_footprint(self) -> None:
        height, width = 96, 128
        x = np.linspace(0, 1, width, dtype=np.float32)
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.clip(220 + 20 * x, 0, 255).astype(np.uint8)
        image[..., 1] = np.clip(170 + 15 * x, 0, 255).astype(np.uint8)
        image[..., 2] = 30
        mask = np.zeros((height, width), dtype=bool)
        mask[30:66, 42:86] = True
        image[mask] = (15, 40, 230)

        result = restore_background(image, [mask], mode="poster", progress=lambda _m: None)

        self.assertTrue(result.report["outside_footprint_byte_identical"])
        self.assertTrue(np.array_equal(image[~result.removal_footprint], result.background[~result.removal_footprint]))
        self.assertFalse(np.array_equal(image[mask], result.background[mask]))


class CompositionTests(unittest.TestCase):
    def test_parent_child_recomposition_is_exact_and_parent_is_clean_below_child(self) -> None:
        height, width = 80, 120
        background = np.full((height, width, 3), 245, dtype=np.uint8)
        master = background.copy()
        master[10:70, 10:110] = (10, 100, 30)
        master[25:50, 25:95] = (240, 240, 20)
        panel_mask = np.zeros((height, width), dtype=bool)
        panel_mask[10:70, 10:110] = True
        text_mask = np.zeros((height, width), dtype=bool)
        text_mask[25:50, 25:95] = True
        panel = LayerSpec(
            "panel",
            "Bảng xanh",
            "detail_group",
            panel_mask,
            0.99,
            metadata={"parent_id": None, "children": ["text"], "hierarchy_depth": 0},
        )
        text = LayerSpec(
            "text",
            "Chữ vàng",
            "text_raster",
            text_mask,
            0.99,
            metadata={"parent_id": "panel", "children": [], "hierarchy_depth": 1},
        )
        cleaned_panel = master.copy()
        cleaned_panel[text_mask] = (10, 100, 30)

        rendered, composite, report = render_layers(
            master,
            background,
            [text, panel],
            layer_targets={"panel": cleaned_panel},
        )

        self.assertEqual([item.spec.layer_id for item in rendered], ["panel", "text"])
        self.assertEqual(report["recomposition_max_abs_error"], 0)
        self.assertTrue(np.array_equal(np.asarray(composite), master))

    def test_restoration_footprint_recomposes_halo_at_x1_x4_and_x20(self) -> None:
        height, width = 32, 40
        source = np.full((height, width, 3), (238, 229, 206), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)
        mask[12:21, 15:25] = True
        halo = dilate_mask(mask, 4) & ~mask
        source[halo] = (205, 135, 90)
        source[mask] = (15, 70, 215)
        restoration = restore_background(
            source,
            [mask],
            mode="poster",
            progress=lambda _message: None,
        )
        radius = int(restoration.report["removal_radius_source_px"])
        support = dilate_mask(mask, radius)
        spec = LayerSpec(
            "object",
            "Object with halo",
            "object",
            mask,
            0.99,
            metadata={"parent_id": None, "children": [], "hierarchy_depth": 0},
        )

        for scale in (1, 4, 20):
            with self.subTest(scale=scale):
                final_size = (width * scale, height * scale)
                master = np.asarray(
                    Image.fromarray(source, "RGB").resize(final_size, Image.Resampling.LANCZOS),
                    dtype=np.uint8,
                ).copy()
                clean = np.asarray(
                    Image.fromarray(restoration.background, "RGB").resize(
                        final_size, Image.Resampling.LANCZOS
                    ),
                    dtype=np.uint8,
                ).copy()
                footprint_image = Image.fromarray(
                    restoration.removal_footprint.astype(np.uint8) * 255,
                    "L",
                ).resize(final_size, Image.Resampling.NEAREST)
                footprint = np.asarray(footprint_image, dtype=np.uint8) > 0
                clean[~footprint] = master[~footprint]

                _layers, composite, report = render_layers(
                    master,
                    clean,
                    [spec],
                    support_masks={"object": support},
                )

                self.assertLessEqual(report["recomposition_max_abs_error"], 1)
                self.assertLessEqual(
                    int(np.abs(np.asarray(composite, dtype=np.int16) - master.astype(np.int16)).max()),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
