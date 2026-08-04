from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


V7_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7_DIR))

from v7lib.formats import (  # noqa: E402
    SVG_NS,
    EditableText,
    FormatError,
    RasterLayer,
    VectorPrimitive,
    build_bundle_manifest,
    export_editable_svg,
    manifest_asset_record,
    save_png_atomic,
    sha256_file,
    validate_editable_svg,
    write_manifest_atomic,
)


class FormatTests(unittest.TestCase):
    def test_png_is_atomic_reopenable_and_colour_managed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "print.png"
            image = np.full((48, 80, 3), (218, 33, 28), dtype=np.uint8)

            report = save_png_atomic(image, target, dpi=(150.0, 150.0))

            self.assertEqual(report["format"], "PNG")
            self.assertTrue(report["icc_profile_embedded"])
            self.assertEqual(report["sha256"], sha256_file(target))
            self.assertFalse(any(".new-" in item.name for item in root.iterdir()))
            with Image.open(target) as reopened:
                reopened.load()
                self.assertEqual(reopened.format, "PNG")
                self.assertEqual(reopened.size, (80, 48))
                self.assertIn("icc_profile", reopened.info)
                self.assertAlmostEqual(reopened.info["dpi"][0], 150.0, delta=0.1)

    def test_mixed_svg_keeps_real_text_and_declares_raster_truthfully(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "editable.svg"
            background = Image.new("RGB", (240, 120), (248, 244, 222))

            report = export_editable_svg(
                target,
                (240, 120),
                background=background,
                raster_layers=[
                    RasterLayer(
                        Image.new("RGBA", (24, 24), (0, 120, 30, 220)),
                        x=8,
                        y=8,
                        layer_id="logo-raster",
                    )
                ],
                texts=[
                    EditableText(
                        "ĐỒ GIA DỤNG",
                        x=120,
                        y=72,
                        font_family="Arial",
                        font_size_px=32,
                        fill=(196, 24, 24),
                        stroke=(255, 255, 255),
                        stroke_width_px=1.5,
                        text_anchor="middle",
                        region_id="headline",
                    )
                ],
                primitives=[
                    VectorPrimitive(
                        "line",
                        {"x1": 20, "y1": 94, "x2": 220, "y2": 94, "stroke": "#14652a"},
                    )
                ],
            )

            self.assertTrue(report["mixed_raster_vector"])
            self.assertFalse(report["full_vector"])
            self.assertEqual(report["raster_image_count"], 2)
            self.assertEqual(report["editable_text_count"], 1)
            self.assertEqual(report["font_files_embedded"], 0)
            self.assertEqual(report["content_claim"], "mixed_raster_vector_editable")

            tree = ET.parse(target)
            root = tree.getroot()
            self.assertEqual(root.get("data-v7-content-claim"), "mixed_raster_vector_editable")
            text = root.find(f".//{{{SVG_NS}}}text")
            self.assertIsNotNone(text)
            assert text is not None
            self.assertEqual("".join(text.itertext()), "ĐỒ GIA DỤNG")
            self.assertEqual(text.get("data-content-kind"), "editable-unicode-text")
            self.assertEqual(text.get("data-font-embedded"), "false")
            for embedded in root.findall(f".//{{{SVG_NS}}}image"):
                self.assertTrue((embedded.get("href") or "").startswith("data:image/png;base64,"))

    def test_vector_only_svg_is_not_mislabeled_as_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "vector.svg"
            export_editable_svg(
                target,
                (180, 80),
                texts=[EditableText("ĐÚNG DẤU", 90, 50, "Arial", 24, text_anchor="middle")],
                primitives=[VectorPrimitive("rect", {"x": 0, "y": 0, "width": 180, "height": 80})],
            )
            report = validate_editable_svg(target)
            self.assertTrue(report["full_vector"])
            self.assertFalse(report["mixed_raster_vector"])

    def test_svg_preserves_safe_variable_font_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "variable-font.svg"
            settings = "'wdth' 82.5, 'wght' 700"
            export_editable_svg(
                target,
                (1200, 440),
                texts=[
                    EditableText(
                        "T\u1eb6NG G\u1ea0O",
                        x=600,
                        y=340,
                        font_family="Bahnschrift",
                        font_size_px=330,
                        font_weight=700,
                        text_anchor="middle",
                        extra_attributes={
                            "font-variation-settings": settings,
                            "font-stretch": "82.5%",
                        },
                    )
                ],
            )

            tree = ET.parse(target)
            text = tree.getroot().find(f".//{{{SVG_NS}}}text")
            self.assertIsNotNone(text)
            assert text is not None
            self.assertEqual(text.get("font-variation-settings"), settings)
            self.assertEqual(text.get("font-stretch"), "82.5%")
            self.assertEqual(text.get("font-weight"), "700")
            self.assertIsNone(text.get("style"))

            with self.assertRaises(FormatError):
                export_editable_svg(
                    root / "unsafe-variable-font.svg",
                    (100, 40),
                    texts=[
                        EditableText(
                            "X",
                            50,
                            30,
                            "Bahnschrift",
                            20,
                            extra_attributes={
                                "style": "font-variation-settings: 'wdth' 50"
                            },
                        )
                    ],
                )

    def test_manifest_uses_portable_relative_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "assets" / "preview.png"
            save_png_atomic(Image.new("RGB", (12, 8), "white"), asset)

            record = manifest_asset_record(asset, root)
            self.assertEqual(record["path"], "assets/preview.png")
            self.assertEqual(record["sha256"], sha256_file(asset))

            manifest = build_bundle_manifest(
                pipeline="V7_DESIGN_REPAIR",
                app_version="7.0.0",
                source_name="ý-tưởng.png",
                source_sha256="a" * 64,
                source_size=(120, 80),
                scale=4,
                final_size=(480, 320),
                bundle_root=root,
                assets=[asset],
                review_required=True,
                qa={"passed": False, "reason": "human text approval required"},
                limitations=["Raster background remains raster."],
            )
            target = root / "manifest.json"
            report = write_manifest_atomic(manifest, target)

            parsed = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(parsed["assets"][0]["path"], "assets/preview.png")
            self.assertTrue(parsed["review_required"])
            self.assertEqual(report["sha256"], sha256_file(target))
            self.assertFalse(any(".new-" in item.name for item in root.iterdir()))

    def test_unsafe_svg_attributes_and_outside_assets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root.parent / f"outside-{root.name}.txt"
            outside.write_text("not in bundle", encoding="utf-8")
            try:
                with self.assertRaises(FormatError):
                    manifest_asset_record(outside, root)
                with self.assertRaises(FormatError):
                    export_editable_svg(
                        root / "unsafe.svg",
                        (20, 20),
                        texts=[
                            EditableText(
                                "X",
                                1,
                                12,
                                "Arial",
                                10,
                                extra_attributes={"onclick": "alert(1)"},
                            )
                        ],
                    )
                with self.assertRaises(FormatError):
                    export_editable_svg(
                        root / "unsafe-primitive.svg",
                        (20, 20),
                        primitives=[
                            VectorPrimitive(
                                "rect",
                                {"x": 0, "y": 0, "width": 20, "height": 20, "OnClick": "x"},
                            )
                        ],
                    )
            finally:
                outside.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
