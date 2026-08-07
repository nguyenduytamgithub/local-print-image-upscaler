from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


V7_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7_DIR))

import bundle_layout  # noqa: E402
from bundle_layout import (  # noqa: E402
    BundleCollisionError,
    BundleLayoutError,
    BundlePathError,
    BundleTransactionError,
    arrange_bundle,
    resolve_bundle_manifest,
    resolve_bundle_paths,
    resolve_bundle_result,
    resolve_bundle_review,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_png(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (12, 8), colour).save(path, format="PNG")


def _make_legacy_bundle(
    root: Path,
    *,
    status: str = "PASS",
    editable: bool = True,
) -> dict[str, Path]:
    root.mkdir(parents=True)
    files = {
        "source": root / "poster_SOURCE_NORMALIZED.png",
        "clean": root / "poster_CLEAN_BASE_x4.png",
        "result": root / "poster_REPAIRED_x4.png",
        "overlay": root / "poster_QA_OVERLAY.png",
        "comparison": root / "poster_BEFORE_AFTER.png",
        "review": root / "TEXT_REVIEW.json",
        "qa": root / "QA.json",
        "manifest": root / "manifest.json",
    }
    for index, role in enumerate(("source", "clean", "result", "overlay", "comparison")):
        _write_png(files[role], (20 + index, 40 + index, 60 + index))
    if editable:
        files["editable"] = root / "poster_TEXT_EDITABLE.svg"
        files["editable"].write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><text>ĐỒ GIA DỤNG</text></svg>',
            encoding="utf-8",
        )
    files["review"].write_text(
        json.dumps(
            {
                "schema": "local-print-image-upscaler/v7-text-review/1",
                "source": files["source"].name,
                "source_sha256": "a" * 64,
                "regions": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    files["qa"].write_text(
        json.dumps({"schema": "v7-qa/1", "status": status}, indent=2),
        encoding="utf-8",
    )
    asset_paths = [path for role, path in files.items() if role != "manifest"]
    manifest = {
        "pipeline": "V7_DESIGN_REPAIR",
        "schema_version": 1,
        "status": status,
        "review_required": status != "PASS",
        "path_policy": "bundle-relative assets only",
        "assets": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in asset_paths
        ],
        "qa": {"status": status, "report": files["qa"].name},
    }
    files["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return files


class BundleLayoutTests(unittest.TestCase):
    def test_pass_bundle_has_only_friendly_root_files_and_real_technical_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "poster_V7"
            legacy = _make_legacy_bundle(root, status="PASS")

            before = resolve_bundle_paths(root)
            self.assertEqual(before.layout, "legacy")
            self.assertEqual(before.result, legacy["result"])
            self.assertEqual(resolve_bundle_manifest(root), legacy["manifest"])
            self.assertEqual(resolve_bundle_review(root), legacy["review"])
            self.assertEqual(resolve_bundle_result(root), legacy["result"])

            result = arrange_bundle(root, "PASS")

            self.assertEqual(result.layout, "friendly")
            self.assertEqual(result.result, root / "01_KET_QUA_DA_DAT.png")
            self.assertEqual(result.comparison, root / "02_SO_SANH.png")
            self.assertEqual(result.editable_svg, root / "03_CHINH_SUA.svg")
            self.assertEqual(
                {item.name for item in root.iterdir()},
                {
                    "01_KET_QUA_DA_DAT.png",
                    "02_SO_SANH.png",
                    "03_CHINH_SUA.svg",
                    "_KY_THUAT",
                },
            )
            technical = root / "_KY_THUAT"
            self.assertEqual(
                {item.name for item in technical.iterdir()},
                {
                    "SOURCE.png",
                    "CLEAN.png",
                    "OVERLAY.png",
                    "TEXT_REVIEW.json",
                    "QA.json",
                    "manifest.json",
                },
            )
            review = json.loads((technical / "TEXT_REVIEW.json").read_text(encoding="utf-8"))
            self.assertEqual(review["source"], "SOURCE.png")
            manifest = json.loads((technical / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["qa"]["report"], "_KY_THUAT/QA.json")
            self.assertEqual(manifest["bundle_layout"]["result"], "01_KET_QUA_DA_DAT.png")
            expected_assets = {
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file() and path.name != "manifest.json"
            }
            self.assertEqual({record["path"] for record in manifest["assets"]}, expected_assets)
            for record in manifest["assets"]:
                asset = root.joinpath(*Path(record["path"]).parts)
                self.assertEqual(record["bytes"], asset.stat().st_size)
                self.assertEqual(record["sha256"], _sha256(asset))

    def test_non_pass_statuses_use_preview_name(self) -> None:
        for status in ("REVIEW_REQUIRED", "FAILED_QA"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "poster_V7"
                _make_legacy_bundle(root, status=status)

                result = arrange_bundle(root, status)

                self.assertEqual(result.result, root / "01_XEM_TRUOC_CAN_DUYET.png")
                self.assertTrue(result.result.is_file())
                self.assertFalse((root / "01_KET_QUA_DA_DAT.png").exists())
                manifest = json.loads(result.manifest.read_text(encoding="utf-8"))  # type: ignore[union-attr]
                self.assertEqual(manifest["bundle_layout"]["status"], status)

    def test_missing_svg_does_not_create_a_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "poster_V7"
            _make_legacy_bundle(root, editable=False)

            result = arrange_bundle(root, "PASS")

            self.assertIsNone(result.editable_svg)
            self.assertFalse((root / "03_CHINH_SUA.svg").exists())
            assert result.manifest is not None
            manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
            self.assertNotIn("editable", manifest["bundle_layout"])

    def test_unclassified_real_files_are_preserved_below_technical_other(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "poster_V7"
            _make_legacy_bundle(root)
            note = root / "engine-note.txt"
            note.write_bytes(b"real diagnostic\r\n")

            arrange_bundle(root, "PASS")

            moved = root / "_KY_THUAT" / "_KHAC" / "engine-note.txt"
            self.assertEqual(moved.read_bytes(), b"real diagnostic\r\n")
            self.assertFalse(note.exists())
            manifest = json.loads(
                (root / "_KY_THUAT" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                "_KY_THUAT/_KHAC/engine-note.txt",
                {record["path"] for record in manifest["assets"]},
            )

    def test_second_call_is_idempotent_and_does_not_publish_again(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "poster_V7"
            _make_legacy_bundle(root)
            arrange_bundle(root, "PASS")
            before = _snapshot(root)

            with patch("bundle_layout._publish_directory") as publish:
                result = arrange_bundle(root, "PASS")

            publish.assert_not_called()
            self.assertEqual(_snapshot(root), before)
            self.assertEqual(result.result, root / "01_KET_QUA_DA_DAT.png")

    def test_duplicate_role_is_rejected_before_any_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "poster_V7"
            _make_legacy_bundle(root)
            _write_png(root / "duplicate_REPAIRED_x4.png", (1, 2, 3))
            before = _snapshot(root)

            with self.assertRaises(BundleCollisionError):
                arrange_bundle(root, "PASS")

            self.assertEqual(_snapshot(root), before)

    def test_manifest_path_escape_is_rejected_before_any_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "poster_V7"
            files = _make_legacy_bundle(root)
            outside = parent / "outside.png"
            _write_png(outside, (1, 2, 3))
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            manifest["assets"][0]["path"] = "../outside.png"
            files["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            before = _snapshot(root)

            with self.assertRaises(BundlePathError):
                arrange_bundle(root, "PASS")

            self.assertEqual(_snapshot(root), before)
            self.assertTrue(outside.is_file())

    def test_missing_required_role_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "poster_V7"
            files = _make_legacy_bundle(root)
            files["overlay"].unlink()
            before = _snapshot(root)

            with self.assertRaises(BundleLayoutError):
                arrange_bundle(root, "PASS")

            self.assertEqual(_snapshot(root), before)
            self.assertFalse((root / "01_KET_QUA_DA_DAT.png").exists())

    def test_status_mismatch_cannot_be_mislabeled_as_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "poster_V7"
            _make_legacy_bundle(root, status="FAILED_QA")
            before = _snapshot(root)

            with self.assertRaises(BundleLayoutError):
                arrange_bundle(root, "PASS")

            self.assertEqual(_snapshot(root), before)
            self.assertFalse((root / "01_KET_QUA_DA_DAT.png").exists())

    def test_non_v7_manifest_is_rejected_without_rearranging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "poster_V7"
            files = _make_legacy_bundle(root)
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            manifest["pipeline"] = "SOME_OTHER_ENGINE"
            files["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            before = _snapshot(root)

            with self.assertRaises(BundleLayoutError):
                arrange_bundle(root, "PASS")

            self.assertEqual(_snapshot(root), before)

    def test_failed_directory_swap_rolls_back_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "poster_V7"
            _make_legacy_bundle(root)
            before = _snapshot(root)
            real_replace = os.replace
            calls = 0

            def fail_publication(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated publish failure")
                real_replace(source, destination)

            with patch.object(bundle_layout.os, "replace", side_effect=fail_publication):
                with self.assertRaises(BundleTransactionError):
                    arrange_bundle(root, "PASS")

            self.assertEqual(calls, 3, "old->backup, failed stage->root, backup->root")
            self.assertTrue(root.is_dir())
            self.assertEqual(_snapshot(root), before)
            leftovers = list(root.parent.glob(f".{root.name}.layout-*-*"))
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
