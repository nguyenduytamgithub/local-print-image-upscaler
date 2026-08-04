from __future__ import annotations

import ast
import copy
import inspect
import sys
import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[4]
V7_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "APP"))
sys.path.insert(0, str(V7_DIR))

from design_repair_v7 import (  # noqa: E402
    _select_font,
    _tight_mask_bbox,
    main as main_v7,
    render_regions,
    typography_geometry_gate,
    typography_lock_fidelity,
)
from v7lib.formats import SVG_NS, EditableText, export_editable_svg  # noqa: E402
from v7lib.masks import TextMaskResult  # noqa: E402
from v7lib.types import TextRegion  # noqa: E402
from v7lib.typography import (  # noqa: E402
    TextStyle,
    discover_windows_fonts,
    font_variation_axes,
    render_text_layer,
)


class TightMaskGeometryTests(unittest.TestCase):
    def test_loose_ocr_bbox_is_tightened_to_the_old_glyph_mask(self) -> None:
        mask = np.zeros((90, 210), dtype=bool)
        mask[28:61, 72:131] = True
        loose_ocr_bbox = (5, 4, 190, 78)

        self.assertEqual(
            _tight_mask_bbox(mask, loose_ocr_bbox),
            (72, 28, 131, 61),
        )


class SvgOriginContractTests(unittest.TestCase):
    def test_engine_maps_reported_origin_to_start_anchored_svg(self) -> None:
        tree = ast.parse(textwrap.dedent(inspect.getsource(main_v7)))
        editable_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "EditableText"
        ]
        self.assertEqual(len(editable_calls), 1)
        keywords = {
            keyword.arg: keyword.value
            for keyword in editable_calls[0].keywords
            if keyword.arg is not None
        }

        self.assertEqual(ast.unparse(keywords["x"]), "rendered.svg_text_origin[0]")
        self.assertEqual(ast.unparse(keywords["y"]), "rendered.svg_text_origin[1]")
        self.assertIsInstance(keywords["text_anchor"], ast.Constant)
        self.assertEqual(keywords["text_anchor"].value, "start")


class GeometryFontSelectionTests(unittest.TestCase):
    def test_tall_narrow_headline_beats_arial_bold_without_italic_or_missing_glyphs(
        self,
    ) -> None:
        text = "CỬA HÀNG"
        bbox = (0, 0, 250, 90)
        fonts = discover_windows_fonts()
        available = {record.family.casefold() for record in fonts}
        if "arial" not in available or not ({"tahoma", "segoe ui"} & available):
            self.skipTest("Required Windows headline fonts are not installed.")

        match, report = _select_font(text, bbox, fonts)
        candidates = report.get("candidates", [])
        arial_bold = next(
            (
                item
                for item in candidates
                if str(item.get("family", "")).casefold() == "arial"
                and "bold" in str(item.get("subfamily", "")).casefold()
            ),
            None,
        )
        if arial_bold is None or len(candidates) < 2:
            self.skipTest("Arial Bold and an alternate geometry candidate are required.")

        self.assertTrue(match.exact_coverage)
        self.assertEqual(match.coverage, 1.0)
        self.assertEqual(match.rendered_text, text)
        selected_style = f"{match.font.family} {match.font.subfamily}".casefold()
        self.assertFalse(
            {"italic", "oblique", "slanted"} & set(selected_style.replace("-", " ").split())
        )

        selected = report["selected"]
        self.assertGreater(
            float(selected["balanced_fill"]),
            float(arial_bold["balanced_fill"]),
            report,
        )

    def test_bahnschrift_fvar_title_is_applied_deterministically_and_passes_gate(
        self,
    ) -> None:
        text = "T\u1eb6NG G\u1ea0O"
        target_bbox = (111, 32, 1115, 362)
        fonts = tuple(
            record
            for record in discover_windows_fonts()
            if record.family.casefold() in {"arial", "bahnschrift"}
        )
        available = {record.family.casefold() for record in fonts}
        if not {"arial", "bahnschrift"}.issubset(available):
            self.skipTest("Arial and Bahnschrift are required for this Windows regression.")

        first_match, first_report = _select_font(
            text,
            target_bbox,
            fonts,
            stroke_width_px=3,
        )
        second_match, second_report = _select_font(
            text,
            target_bbox,
            fonts,
            stroke_width_px=3,
        )

        self.assertEqual(first_match.font.identifier, second_match.font.identifier)
        self.assertEqual(first_report, second_report)
        self.assertEqual(first_match.font.family.casefold(), "bahnschrift")
        selected = first_report["selected"]
        axes_by_tag = selected["variation_axes"]
        self.assertEqual(float(axes_by_tag["wght"]), 700.0)
        self.assertAlmostEqual(float(axes_by_tag["wdth"]), 82.5, delta=0.001)

        definitions = font_variation_axes(first_match.font)
        variation_axes = tuple(float(axes_by_tag[item.tag]) for item in definitions)
        style = TextStyle(
            font=first_match.font,
            font_size_px=int(selected["font_size_px"]),
            stroke_width_px=3,
            variation_axes=variation_axes,
        )
        first_render = render_text_layer(
            text,
            target_bbox,
            style,
            canvas_size=(1143, 436),
            padding_px=0,
        )
        second_render = render_text_layer(
            text,
            target_bbox,
            style,
            canvas_size=(1143, 436),
            padding_px=0,
        )

        self.assertEqual(first_render.visible_bbox, second_render.visible_bbox)
        self.assertEqual(first_render.report, second_render.report)
        self.assertTrue(
            np.array_equal(np.asarray(first_render.rgba), np.asarray(second_render.rgba))
        )
        self.assertEqual(first_render.report["variation_axes"], axes_by_tag)
        origin_x, origin_y = first_render.svg_text_origin
        self.assertEqual(first_render.report["svg_text_origin"], [origin_x, origin_y])
        self.assertTrue(np.isfinite(origin_x))
        self.assertTrue(np.isfinite(origin_y))
        left, top, right, bottom = first_render.visible_bbox
        self.assertGreaterEqual(origin_x, left)
        self.assertLessEqual(origin_x, right)
        self.assertGreaterEqual(origin_y, top)
        self.assertLessEqual(origin_y, bottom)

        with tempfile.TemporaryDirectory() as temporary:
            svg_path = Path(temporary) / "title-origin.svg"
            export_editable_svg(
                svg_path,
                (1143, 436),
                texts=[
                    EditableText(
                        text,
                        x=origin_x,
                        y=origin_y,
                        font_family=first_render.font.family,
                        font_size_px=first_render.font_size_px,
                        font_weight=700,
                        stroke_width_px=first_render.stroke_width_px,
                        text_anchor="start",
                        region_id="title",
                    )
                ],
            )
            svg_text = ET.parse(svg_path).getroot().find(f".//{{{SVG_NS}}}text")
            self.assertIsNotNone(svg_text)
            assert svg_text is not None
            self.assertEqual(svg_text.get("text-anchor"), "start")
            self.assertAlmostEqual(float(svg_text.get("x", "nan")), origin_x)
            self.assertAlmostEqual(float(svg_text.get("y", "nan")), origin_y)

        gate = typography_geometry_gate(
            [
                {
                    "region_id": "title",
                    "approved_text": text,
                    "source_bbox": list(target_bbox),
                    "visible_bbox": list(first_render.visible_bbox),
                    "font_geometry": first_report,
                }
            ]
        )
        self.assertTrue(gate["passed"], gate)
        self.assertEqual(gate["requires_review_region_ids"], [])
        check = gate["checks"][0]
        self.assertGreater(float(check["width_ratio"]), 0.98)
        self.assertGreater(float(check["height_ratio"]), 0.98)

    def test_geometry_gate_rejects_known_wide_short_arial_regression(self) -> None:
        gate = typography_geometry_gate(
            [
                {
                    "region_id": "title",
                    "approved_text": "T\u1eb6NG G\u1ea0O",
                    "source_bbox": [111, 32, 1115, 362],
                    "visible_bbox": [0, 128, 1170, 360],
                    "font_geometry": {
                        "selected": {"family": "Arial", "subfamily": "Bold"}
                    },
                }
            ]
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(gate["requires_review_region_ids"], ["title"])
        check = gate["checks"][0]
        self.assertEqual(check["status"], "REQUIRES_REVIEW")
        self.assertGreater(float(check["width_ratio"]), 1.12)
        self.assertLess(float(check["height_ratio"]), 0.78)
        self.assertGreater(float(check["center_y_error"]), 0.10)


class TypographyLockTests(unittest.TestCase):
    def test_x4_reuses_source_face_axes_and_size_without_reselecting(self) -> None:
        text = "T\u1eb6NG G\u1ea0O"
        source_size = (1143, 436)
        source_bbox = (111, 32, 1115, 362)
        bahnschrift = next(
            (
                record
                for record in discover_windows_fonts()
                if record.family.casefold() == "bahnschrift"
                and {item.tag for item in font_variation_axes(record)} >= {"wght", "wdth"}
            ),
            None,
        )
        if bahnschrift is None:
            self.skipTest("Bahnschrift with wght/wdth fvar axes is not installed.")

        old_mask = np.zeros((source_size[1], source_size[0]), dtype=bool)
        old_mask[source_bbox[1] : source_bbox[3], source_bbox[0] : source_bbox[2]] = True
        source_rgb = np.full(
            (source_size[1], source_size[0], 3),
            (248, 244, 224),
            dtype=np.uint8,
        )
        source_rgb[old_mask] = (195, 24, 24)
        clean_source = Image.new("RGB", source_size, (248, 244, 224))
        region = TextRegion("title", source_bbox, selected_text=text, status="green")
        masks = {
            "title": TextMaskResult(
                mask=old_mask,
                roi=source_bbox,
                quality=1.0,
                report={"fixture": "tight-title-ink"},
            )
        }
        replacements = {"title": text}
        fonts = (bahnschrift,)

        _source_image, _source_alpha, source_records, source_renders = render_regions(
            clean_source,
            source_rgb,
            [region],
            replacements,
            masks,
            fonts,
            scale=1.0,
        )
        self.assertEqual(len(source_records), 1)
        self.assertTrue(source_records[0]["font"]["variation_axes"])

        scale = 4.0
        with mock.patch.object(
            sys.modules[render_regions.__module__],
            "_select_font",
            side_effect=AssertionError("final render must not select a font again"),
        ) as selector:
            _final_image, _final_alpha, final_records, _final_renders = render_regions(
                Image.new(
                    "RGB",
                    (source_size[0] * 4, source_size[1] * 4),
                    (248, 244, 224),
                ),
                source_rgb,
                [region],
                replacements,
                masks,
                fonts,
                scale=scale,
                typography_locks={"title": source_renders[0]},
            )
        selector.assert_not_called()

        source_record = source_records[0]
        final_record = final_records[0]
        self.assertEqual(final_record["font"]["path"], source_record["font"]["path"])
        self.assertEqual(
            final_record["font"]["face_index"], source_record["font"]["face_index"]
        )
        self.assertEqual(
            final_record["font"]["variation_axes"],
            source_record["font"]["variation_axes"],
        )
        self.assertEqual(
            final_record["font_size_px"],
            int(round(source_record["font_size_px"] * scale)),
        )
        self.assertTrue(final_record["typography_lock"]["locked"])
        self.assertEqual(
            final_record["geometry_target_bbox"],
            [int(round(value * scale)) for value in source_bbox],
        )

        geometry_gate = typography_geometry_gate(final_records)
        self.assertTrue(geometry_gate["passed"], geometry_gate)
        self.assertEqual(
            geometry_gate["checks"][0]["target_ink_bbox"],
            [float(int(round(value * scale))) for value in source_bbox],
        )
        fidelity = typography_lock_fidelity(source_records, final_records, scale=scale)
        self.assertTrue(fidelity["passed"], fidelity)

    def test_lock_fidelity_fails_if_face_axes_or_scaled_size_changes(self) -> None:
        source = {
            "region_id": "title",
            "font_size_px": 279,
            "font": {
                "path": r"C:\Windows\Fonts\bahnschrift.ttf",
                "face_index": 0,
                "variation_axes": {"wght": 700.0, "wdth": 82.5},
            },
        }
        expected_final = {
            "region_id": "title",
            "font_size_px": 1116,
            "font": copy.deepcopy(source["font"]),
        }
        mutations = {
            "face": lambda record: record["font"].__setitem__("face_index", 1),
            "axes": lambda record: record["font"]["variation_axes"].__setitem__(
                "wdth", 85.0
            ),
            "size": lambda record: record.__setitem__("font_size_px", 1115),
        }

        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(expected_final)
                mutate(changed)
                report = typography_lock_fidelity([source], [changed], scale=4.0)
                self.assertFalse(report["passed"])
                self.assertEqual(report["requires_review_region_ids"], ["title"])
                self.assertFalse(report["checks"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
