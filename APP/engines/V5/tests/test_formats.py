from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageCms


V5_ROOT = Path(__file__).resolve().parents[1]
if str(V5_ROOT) not in sys.path:
    sys.path.insert(0, str(V5_ROOT))

from v5lib.formats import (  # noqa: E402
    CONTACT_SHEET_MAX_DIMENSION,
    CONTACT_SHEET_MAX_PIXELS,
    CONTACT_SHEET_PAGE_CAPACITY,
    RenderedLayer,
    asset_records,
    export_contact_sheet,
    export_ora,
    export_png_assets,
    export_psd,
    package_layers_zip,
    render_layers,
    save_color_png,
    sha256_file,
    validate_ora,
)
from v5lib.model import LayerSpec  # noqa: E402
from layer_engine_v5 import assert_portable_bundle_manifest  # noqa: E402


def srgb_icc_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def make_rendered(
    layer_id: str,
    name: str,
    rgba: Image.Image,
    left: int,
    top: int,
    canvas: tuple[int, int],
    *,
    parent_id: str | None = None,
) -> RenderedLayer:
    width, height = canvas
    mask = np.zeros((height, width), dtype=bool)
    right, bottom = left + rgba.width, top + rgba.height
    mask[max(0, top) : min(height, bottom), max(0, left) : min(width, right)] = True
    metadata = {"display_order": 0, "hierarchy_depth": 1 if parent_id else 0}
    if parent_id:
        metadata["parent_id"] = parent_id
    spec = LayerSpec(
        layer_id=layer_id,
        name=name,
        category="object",
        mask=mask,
        score=1.0,
        metadata=metadata,
    )
    return RenderedLayer(
        spec=spec,
        rgba=rgba.convert("RGBA"),
        alpha=rgba.convert("RGBA").getchannel("A"),
        left=left,
        top=top,
        source_bbox=spec.bbox,
    )


def sample_bundle() -> tuple[Image.Image, list[RenderedLayer], Image.Image]:
    canvas = (64, 48)
    background = Image.new("RGB", canvas, (245, 240, 225))
    parent = make_rendered(
        "parent",
        "01 Nhóm sản phẩm",
        Image.new("RGBA", (30, 24), (220, 45, 35, 190)),
        4,
        4,
        canvas,
    )
    child_rgba = Image.new("RGBA", (12, 10), (20, 80, 240, 128))
    child = make_rendered(
        "child",
        "01.1 Chữ tiếng Việt",
        child_rgba,
        10,
        8,
        canvas,
        parent_id="parent",
    )
    top = make_rendered(
        "top",
        "02 Chi tiết trên cùng",
        Image.new("RGBA", (12, 15), (20, 185, 90, 220)),
        40,
        20,
        canvas,
    )
    rendered = [parent, child, top]
    expected = background.convert("RGBA")
    for item in rendered:
        expected.alpha_composite(item.rgba, dest=(item.left, item.top))
    return background, rendered, expected.convert("RGB")


class FormatTests(unittest.TestCase):
    def test_contact_sheet_keeps_single_file_layout_for_small_jobs(self) -> None:
        background, rendered, _expected = sample_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "06_sample_CONTACT_SHEET.png"
            result = export_contact_sheet(output, background, rendered)
            self.assertIsNone(result)
            self.assertTrue(output.is_file())
            self.assertFalse((root / "CONTACT_SHEETS").exists())
            with Image.open(output) as sheet:
                self.assertEqual(sheet.size, (1200, 292))

    def test_large_contact_sheet_is_bounded_paginated_and_deterministic(self) -> None:
        canvas = (64, 48)
        background = Image.new("RGB", canvas, (245, 240, 225))
        rendered = [
            make_rendered(
                f"layer_{index:03d}",
                f"Layer {index:03d}",
                Image.new(
                    "RGBA",
                    (8, 8),
                    ((index * 17) % 256, (index * 29) % 256, (index * 43) % 256, 255),
                ),
                index % 40,
                index % 30,
                canvas,
            )
            for index in range(CONTACT_SHEET_PAGE_CAPACITY * 2 + 6)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "06_large_CONTACT_SHEET.png"
            export_contact_sheet(output, background, rendered)
            pages_dir = root / "CONTACT_SHEETS"
            pages = sorted(pages_dir.glob("page_*.png"))
            self.assertEqual([page.name for page in pages], [
                "page_001.png",
                "page_002.png",
                "page_003.png",
            ])
            for image_path in [output, *pages]:
                with Image.open(image_path) as image:
                    width, height = image.size
                    self.assertLessEqual(max(width, height), CONTACT_SHEET_MAX_DIMENSION)
                    self.assertLessEqual(width * height, CONTACT_SHEET_MAX_PIXELS)

            recorded_paths = {record["path"] for record in asset_records(root)}
            self.assertIn("06_large_CONTACT_SHEET.png", recorded_paths)
            self.assertEqual(
                sorted(path for path in recorded_paths if path.startswith("CONTACT_SHEETS/")),
                [
                    "CONTACT_SHEETS/page_001.png",
                    "CONTACT_SHEETS/page_002.png",
                    "CONTACT_SHEETS/page_003.png",
                ],
            )
            guide = root / "guide.txt"
            guide.write_text("portable guide", encoding="utf-8")
            archive_report = package_layers_zip(
                root / "layers.zip",
                root,
                [guide],
            )
            self.assertEqual(archive_report["members"], ["guide.txt"])
            first_hashes = {
                image_path.relative_to(root).as_posix(): sha256_file(image_path)
                for image_path in [output, *pages]
            }
            stale = pages_dir / "page_999.png"
            stale.write_bytes(b"stale generated page")
            export_contact_sheet(output, background, rendered)
            self.assertFalse(stale.exists())
            second_pages = sorted(pages_dir.glob("page_*.png"))
            second_hashes = {
                image_path.relative_to(root).as_posix(): sha256_file(image_path)
                for image_path in [output, *second_pages]
            }
            self.assertEqual(first_hashes, second_hashes)
            export_contact_sheet(output, background, rendered[:1])
            self.assertFalse(pages_dir.exists())

    def test_color_pngs_keep_srgb_icc_while_alpha_masks_remain_untagged(self) -> None:
        profile = srgb_icc_bytes()
        background, rendered, _expected = sample_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preview = root / "preview.png"
            save_color_png(background, preview, profile)
            with Image.open(preview) as reopened:
                self.assertEqual(reopened.info.get("icc_profile"), profile)

            records = export_png_assets(root, background, rendered, icc_profile=profile)
            with Image.open(root / "LAYERS" / "00_BACKGROUND_SYNTHESIZED.png") as reopened:
                self.assertEqual(reopened.info.get("icc_profile"), profile)
            with Image.open(root / records[0]["rgba"]) as reopened:
                self.assertEqual(reopened.info.get("icc_profile"), profile)
            with Image.open(root / records[0]["mask"]) as reopened:
                self.assertEqual(reopened.mode, "L")
                self.assertNotIn("icc_profile", reopened.info)

    def test_portable_layers_zip_contains_only_explicit_safe_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layers = root / "LAYERS"
            masks = root / "MASKS"
            layers.mkdir()
            masks.mkdir()
            (layers / "one.png").write_bytes(b"layer")
            (masks / "one_MASK.png").write_bytes(b"mask")
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "large.psd").write_bytes(b"must stay outside")
            archive_path = root / "layers.zip"

            report = package_layers_zip(
                archive_path,
                root,
                [root / "manifest.json", layers / "one.png", masks / "one_MASK.png"],
            )

            self.assertEqual(report["member_count"], 3)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    archive.namelist(),
                    ["LAYERS/one.png", "MASKS/one_MASK.png", "manifest.json"],
                )
                self.assertNotIn("large.psd", archive.namelist())

    def test_bundle_manifest_rejects_dead_staging_paths(self) -> None:
        staging = Path(r"C:\work\.poster.new-123")
        assert_portable_bundle_manifest(
            {"formats": {"psd": {"path": "poster_EDITABLE.psd"}}},
            staging,
        )
        with self.assertRaisesRegex(RuntimeError, "staging path"):
            assert_portable_bundle_manifest(
                {"source": r"C:\work\job_v5_dead\input.png"},
                staging,
            )

    def test_render_preview_is_flattened_from_actual_rgba_assets(self) -> None:
        height, width = 30, 40
        background = np.full((height, width, 3), (35, 42, 50), dtype=np.uint8)
        master = background.copy()
        master[4:24, 5:29] = (218, 47, 31)
        master[10:28, 18:37] = (25, 92, 231)
        mask_a = np.zeros((height, width), dtype=bool)
        mask_a[4:24, 5:29] = True
        mask_b = np.zeros((height, width), dtype=bool)
        mask_b[10:28, 18:37] = True
        specs = [
            LayerSpec("a", "A", "object", mask_a, 1.0, metadata={"display_order": 0}),
            LayerSpec("b", "B", "object", mask_b, 1.0, metadata={"display_order": 1}),
        ]

        rendered, preview, report = render_layers(master, background, specs)
        flattened = Image.fromarray(background, "RGB").convert("RGBA")
        for item in rendered:
            flattened.alpha_composite(item.rgba, dest=(item.left, item.top))
        np.testing.assert_array_equal(
            np.asarray(preview), np.asarray(flattened.convert("RGB"))
        )
        actual_error = np.abs(
            np.asarray(preview, dtype=np.int16) - master.astype(np.int16)
        )
        self.assertEqual(report["recomposition_max_abs_error"], int(actual_error.max()))
        self.assertIn("actual cropped 8-bit RGBA", report["preview_basis"])

    def test_render_order_is_container_compatible(self) -> None:
        height, width = 24, 32
        background = np.full((height, width, 3), 30, dtype=np.uint8)
        master = background.copy()
        master[2:18, 2:18] = (210, 40, 30)
        master[6:14, 6:14] = (240, 210, 20)
        master[8:22, 14:30] = (20, 90, 220)

        def mask(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
            value = np.zeros((height, width), dtype=bool)
            value[y0:y1, x0:x1] = True
            return value

        specs = [
            LayerSpec(
                "parent",
                "Parent",
                "object",
                mask(2, 18, 2, 18),
                1.0,
                metadata={"hierarchy_depth": 0, "display_order": 0},
            ),
            LayerSpec(
                "other_root",
                "Other root",
                "object",
                mask(8, 22, 14, 30),
                1.0,
                metadata={"hierarchy_depth": 0, "display_order": 1},
            ),
            LayerSpec(
                "child",
                "Child",
                "object",
                mask(6, 14, 6, 14),
                1.0,
                metadata={
                    "parent_id": "parent",
                    "hierarchy_depth": 1,
                    "display_order": 2,
                },
            ),
        ]
        rendered, preview, _report = render_layers(master, background, specs)
        self.assertEqual(
            [item.spec.layer_id for item in rendered],
            ["parent", "child", "other_root"],
        )
        flattened = Image.fromarray(background, "RGB").convert("RGBA")
        for item in rendered:
            flattened.alpha_composite(item.rgba, dest=(item.left, item.top))
        np.testing.assert_array_equal(np.asarray(preview), np.asarray(flattened.convert("RGB")))

    def test_psd_roundtrip_preserves_unicode_hierarchy_order_alpha(self) -> None:
        background, rendered, expected = sample_bundle()
        profile = srgb_icc_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layers.psd"
            report = export_psd(
                output,
                background,
                rendered,
                expected_composite=expected,
                icc_profile=profile,
            )
            self.assertTrue(report["created"])
            self.assertEqual(report["format"], "PSD")
            self.assertEqual(report["header_version"], 1)
            self.assertEqual(report["user_mask_count"], 0)
            self.assertEqual(report["group_blend_mode"], "pass_through")
            self.assertLessEqual(report["roundtrip_qa"]["max_abs_error"], 1)
            self.assertEqual(report["icc_profile_sha256"], hashlib.sha256(profile).hexdigest())
            from psd_tools import PSDImage
            from psd_tools.constants import Resource

            reopened = PSDImage.open(output)
            self.assertEqual(reopened.image_resources.get_data(Resource.ICC_PROFILE), profile)
            self.assertNotIn(Resource.ICC_UNTAGGED_PROFILE, reopened.image_resources)
            self.assertEqual(
                report["top_level_layers_bottom_to_top"],
                [
                    "00 BACKGROUND - SYNTHESIZED HIDDEN PIXELS",
                    "GROUP - 01 Nhóm sản phẩm",
                    "02 Chi tiết trên cùng",
                ],
            )
            group = report["layer_tree_bottom_to_top"][1]
            self.assertEqual(
                [item["name"] for item in group["children_bottom_to_top"]],
                ["BASE - 01 Nhóm sản phẩm", "01.1 Chữ tiếng Việt"],
            )

    def test_psd_guards_no_fake_psb_and_no_over_30000(self) -> None:
        background, rendered, expected = sample_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / "fake.psb"
            fake_report = export_psd(fake, background, rendered, expected)
            self.assertFalse(fake_report["created"])
            self.assertFalse(fake.exists())
            too_wide = Image.new("RGB", (30_001, 1), "white")
            wide_path = Path(temporary) / "too_wide.psd"
            wide_report = export_psd(wide_path, too_wide, [])
            self.assertFalse(wide_report["created"])
            self.assertFalse(wide_path.exists())

    def test_openraster_roundtrip_structure_and_determinism(self) -> None:
        background, rendered, expected = sample_bundle()
        profile = srgb_icc_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layers.ora"
            first = export_ora(output, background, rendered, expected, icc_profile=profile)
            first_hash = sha256_file(output)
            second = export_ora(output, background, rendered, expected, icc_profile=profile)
            self.assertEqual(first_hash, sha256_file(output))
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertTrue(first["validated"])
            self.assertLessEqual(first["recomposition_qa"]["max_abs_error"], 1)
            self.assertLessEqual(first["expected_composite_qa"]["max_abs_error"], 1)
            expected_profile_hash = hashlib.sha256(profile).hexdigest()
            self.assertTrue(first["png_icc_sha256"])
            self.assertEqual(set(first["png_icc_sha256"].values()), {expected_profile_hash})

            with zipfile.ZipFile(output) as archive:
                infos = archive.infolist()
                self.assertEqual(infos[0].filename, "mimetype")
                self.assertEqual(infos[0].compress_type, zipfile.ZIP_STORED)
                self.assertEqual(archive.read("mimetype"), b"image/openraster")
                tree = ET.fromstring(archive.read("stack.xml"))
            self.assertEqual(tree.attrib["version"], "0.0.6")
            root_stack = tree.find("stack")
            self.assertIsNotNone(root_stack)
            self.assertEqual(root_stack.attrib, {})
            self.assertEqual(
                [child.attrib.get("name") for child in root_stack],
                [
                    "02 Chi tiết trên cùng",
                    "GROUP - 01 Nhóm sản phẩm",
                    "00 BACKGROUND - SYNTHESIZED HIDDEN PIXELS",
                ],
            )
            group = list(root_stack)[1]
            self.assertEqual(group.tag, "stack")
            self.assertEqual(group.attrib.get("isolation"), "auto")

    def test_user_review_group_is_real_in_psd_and_ora_without_reordering(self) -> None:
        background, rendered, expected = sample_bundle()
        groups = [
            {
                "id": "USER_GROUP_1",
                "name": "Cụm sản phẩm",
                "member_ids": ["parent", "top"],
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            psd_report = export_psd(
                root / "grouped.psd",
                background,
                rendered,
                expected,
                user_groups=groups,
            )
            ora_report = export_ora(
                root / "grouped.ora",
                background,
                rendered,
                expected,
                user_groups=groups,
            )
        self.assertLessEqual(psd_report["roundtrip_qa"]["max_abs_error"], 1)
        self.assertLessEqual(ora_report["expected_composite_qa"]["max_abs_error"], 1)
        psd_group = psd_report["layer_tree_bottom_to_top"][1]
        self.assertEqual(psd_group["name"], "USER GROUP - Cụm sản phẩm")
        self.assertEqual(
            [item["name"] for item in psd_group["children_bottom_to_top"]],
            ["GROUP - 01 Nhóm sản phẩm", "02 Chi tiết trên cùng"],
        )
        ora_group = ora_report["layer_tree_bottom_to_top"][1]
        self.assertEqual(ora_group, psd_group)

    def test_openraster_validator_keeps_limit_when_pillow_global_is_none(self) -> None:
        background, rendered, expected = sample_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layers.ora"
            export_ora(output, background, rendered, expected)
            previous = Image.MAX_IMAGE_PIXELS
            try:
                Image.MAX_IMAGE_PIXELS = None
                report = validate_ora(output, expected_composite=expected)
            finally:
                Image.MAX_IMAGE_PIXELS = previous
        self.assertEqual(report["canvas"], list(background.size))

    def test_openraster_validator_rejects_unsafe_member_and_root_attributes(self) -> None:
        background, rendered, expected = sample_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            good = Path(temporary) / "good.ora"
            export_ora(good, background, rendered, expected)

            unsafe = Path(temporary) / "unsafe.ora"
            unsafe.write_bytes(good.read_bytes())
            with zipfile.ZipFile(unsafe, "a") as archive:
                archive.writestr("../escape.png", b"not a png")
            with self.assertRaisesRegex(RuntimeError, "unsafe ZIP member"):
                validate_ora(unsafe)

            bad_root = Path(temporary) / "bad_root.ora"
            with zipfile.ZipFile(good, "r") as source:
                members = [(info, source.read(info.filename)) for info in source.infolist()]
            xml = ET.fromstring(dict((info.filename, data) for info, data in members)["stack.xml"])
            xml.find("stack").set("name", "root must not be named")
            bad_xml = ET.tostring(xml, encoding="utf-8", xml_declaration=True)
            with zipfile.ZipFile(bad_root, "w", allowZip64=True) as target:
                for info, data in members:
                    target.writestr(info, bad_xml if info.filename == "stack.xml" else data)
            with self.assertRaisesRegex(RuntimeError, "root stack"):
                validate_ora(bad_root)


if __name__ == "__main__":
    unittest.main()
