from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

import numpy as np
from PIL import Image, ImageFont


V7_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7_DIR))

from v7lib.typography import (  # noqa: E402
    FontRecord,
    TextStyle,
    TypographyError,
    composite_text,
    estimate_text_style,
    fit_font_size,
    font_embedding_policy,
    inspect_font,
    render_text_layer,
    select_font,
    text_for_font,
)


def _portable_test_font() -> FontRecord:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
    ]
    try:
        pillow_font = ImageFont.truetype("DejaVuSans.ttf", 12)
        pillow_path = getattr(pillow_font, "path", None)
        if pillow_path:
            candidates.append(Path(pillow_path))
    except OSError:
        pass
    for candidate in candidates:
        if not candidate.is_file():
            continue
        record = inspect_font(candidate)
        if text_for_font("ĐỒ GIA DỤNG", record)[2] == 1.0:
            return record
    raise unittest.SkipTest("No test font with Vietnamese glyph coverage was found.")


class TypographyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.font = _portable_test_font()

    def test_embedding_flags_are_interpreted_conservatively(self) -> None:
        installable = font_embedding_policy(0)
        self.assertTrue(installable.installable)
        self.assertTrue(installable.embeddable)

        restricted = font_embedding_policy(0x0002 | 0x0100)
        self.assertTrue(restricted.restricted)
        self.assertFalse(restricted.embeddable)
        self.assertTrue(restricted.no_subsetting)

        preview = font_embedding_policy(0x0004 | 0x0200)
        self.assertTrue(preview.preview_print)
        self.assertTrue(preview.embeddable)
        self.assertTrue(preview.bitmap_only)

    def test_font_inspection_and_selection_preserve_vietnamese(self) -> None:
        match = select_font(
            "ĐỒ GIA DỤNG",
            fonts=[self.font],
            preferred_families=[self.font.family],
        )
        self.assertTrue(match.exact_coverage)
        self.assertEqual(match.coverage, 1.0)
        self.assertEqual(match.rendered_text, "ĐỒ GIA DỤNG")
        self.assertTrue(match.font.path.is_file())

    def test_render_is_native_final_scale_and_fits_the_box(self) -> None:
        box = (16, 12, 484, 126)
        style = TextStyle(
            font=self.font,
            fill_rgb=(194, 24, 24),
            stroke_rgb=(255, 250, 230),
            stroke_width_px=3,
            horizontal_align="center",
            vertical_align="middle",
        )
        rendered = render_text_layer(
            "ĐỒ GIA DỤNG",
            box,
            style,
            canvas_size=(500, 140),
        )

        left, top, right, bottom = rendered.visible_bbox
        self.assertGreaterEqual(left, box[0])
        self.assertGreaterEqual(top, box[1])
        self.assertLessEqual(right, box[2])
        self.assertLessEqual(bottom, box[3])
        self.assertGreater(rendered.font_size_px, 1)
        self.assertIn(rendered.layout_engine, {"RAQM/HarfBuzz", "Pillow BASIC fallback"})
        self.assertEqual(rendered.rendered_text, "ĐỒ GIA DỤNG")
        self.assertEqual(rendered.rgba.mode, "RGBA")
        self.assertGreater(np.asarray(rendered.rgba.getchannel("A")).max(), 0)
        self.assertEqual(
            rendered.report["render_policy"],
            "approved Unicode rendered directly at final scale",
        )
        origin_x, origin_y = rendered.svg_text_origin
        self.assertTrue(math.isfinite(origin_x))
        self.assertTrue(math.isfinite(origin_y))
        self.assertEqual(rendered.report["svg_text_origin"], [origin_x, origin_y])
        self.assertGreaterEqual(origin_x, left)
        self.assertLessEqual(origin_x, right)
        self.assertGreaterEqual(origin_y, top)
        self.assertLessEqual(origin_y, bottom)

        base = Image.new("RGB", (500, 140), (247, 241, 215))
        combined = composite_text(base, rendered)
        self.assertEqual(combined.mode, "RGBA")
        self.assertNotEqual(combined.getpixel((250, 70))[:3], base.getpixel((250, 70)))

    def test_style_estimation_finds_fill_and_stroke(self) -> None:
        style = TextStyle(
            font=self.font,
            font_size_px=64,
            fill_rgb=(190, 22, 26),
            stroke_rgb=(252, 246, 222),
            stroke_width_px=5,
        )
        rendered = render_text_layer(
            "ĐỒ",
            (20, 12, 300, 105),
            style,
            canvas_size=(320, 120),
        )
        base = Image.new("RGB", (320, 120), (38, 100, 54))
        composed = composite_text(base, rendered).convert("RGB")
        full_mask = Image.new("L", base.size, 0)
        full_mask.paste(rendered.rgba.getchannel("A"), rendered.position)

        estimate = estimate_text_style(
            np.asarray(composed, dtype=np.uint8),
            np.asarray(full_mask, dtype=np.uint8) > 0,
        )

        fill_error = np.linalg.norm(
            np.asarray(estimate.fill_rgb, dtype=float) - np.asarray(style.fill_rgb, dtype=float)
        )
        self.assertLess(fill_error, 35.0)
        self.assertGreater(estimate.confidence, 0.5)
        self.assertEqual(estimate.report["mask_source"], "provided_mask")
        self.assertIsNotNone(estimate.stroke_rgb)
        assert estimate.stroke_rgb is not None
        stroke_error = np.linalg.norm(
            np.asarray(estimate.stroke_rgb, dtype=float)
            - np.asarray(style.stroke_rgb, dtype=float)
        )
        self.assertLess(stroke_error, 55.0)

    def test_font_size_binary_search_obeys_bounds(self) -> None:
        style = TextStyle(font=self.font, stroke_width_px=1)
        size = fit_font_size(
            "TẶNG GẠO",
            self.font,
            (360, 90),
            style=style,
            maximum_size=120,
        )
        self.assertGreater(size, 1)
        render_text_layer(
            "TẶNG GẠO",
            (0, 0, 360, 90),
            TextStyle(font=self.font, font_size_px=size, stroke_width_px=1),
            canvas_size=(360, 90),
        )

    def test_start_alignment_alias_and_invalid_style_validation(self) -> None:
        rendered = render_text_layer(
            "ĐỒ",
            (10, 5, 190, 75),
            TextStyle(font=self.font, horizontal_align="start", vertical_align="top"),
            canvas_size=(200, 80),
        )
        self.assertEqual(rendered.visible_bbox[0], 10)
        self.assertEqual(rendered.visible_bbox[1], 5)

        with self.assertRaises(TypographyError):
            render_text_layer(
                "ĐỒ",
                (0, 0, 100, 50),
                TextStyle(font=self.font, stroke_width_px=-1),
            )

    def test_missing_glyph_is_not_silently_rendered(self) -> None:
        # U+20000 is an assigned CJK ideograph, not an ignorable/control code,
        # and is absent from the ordinary Latin fonts used by this test.
        missing = chr(0x20000)
        if ord(missing) in self.font.codepoints:
            self.skipTest("The selected test font unexpectedly covers U+20000.")
        with self.assertRaises(TypographyError):
            select_font("ĐỒ " + missing, fonts=[self.font])


if __name__ == "__main__":
    unittest.main()
