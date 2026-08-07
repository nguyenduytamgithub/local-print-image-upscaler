from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "APP"))

import upscale_cli  # noqa: E402
from upscale_cli import UserError, parse_command  # noqa: E402


class V7LauncherParsingTests(unittest.TestCase):
    def test_repair_and_v7_aliases_accept_exactly_x1(self) -> None:
        for alias in ("repair", "v7"):
            with self.subTest(alias=alias):
                mode, token, scale, allow_huge, options = parse_command(
                    [alias, "poster.png", "1"]
                )
                self.assertEqual(mode, "V7_REPAIR")
                self.assertEqual(token, "poster.png")
                self.assertEqual(scale, 1.0)
                self.assertFalse(allow_huge)
                self.assertEqual(options["review"], "gui")

    def test_repair_and_v7_aliases_reject_x1_point_5(self) -> None:
        for alias in ("repair", "v7"):
            with self.subTest(alias=alias):
                with self.assertRaises(UserError):
                    parse_command([alias, "poster.png", "1.5"])


class V7ReviewCommandTests(unittest.TestCase):
    def _review_bundle(self, root: Path, *, friendly: bool) -> tuple[Path, Path]:
        root.mkdir(parents=True)
        source_original = root.parent / "ảnh gốc.png"
        Image.new("RGB", (8, 6), "white").save(source_original)
        if friendly:
            technical = root / "_KY_THUAT"
            technical.mkdir()
            review = technical / "TEXT_REVIEW.json"
            source = technical / "SOURCE.png"
            manifest = technical / "manifest.json"
        else:
            review = root / "TEXT_REVIEW.json"
            source = root / "poster_SOURCE_NORMALIZED.png"
            manifest = root / "manifest.json"
        Image.new("RGB", (8, 6), "white").save(source)
        review.write_text(json.dumps({"source": source.name}), encoding="utf-8")
        manifest.write_text(
            json.dumps(
                {
                    "pipeline": "V7_DESIGN_REPAIR",
                    "status": "REVIEW_REQUIRED",
                    "original_source": str(source_original),
                    "scale": 4.0,
                }
            ),
            encoding="utf-8",
        )
        return review, source

    def test_review_and_duyet_accept_old_and_new_layout_without_running_engine(self) -> None:
        for command, friendly in (("review", False), ("duyet", True)):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                bundle = root / "bundle"
                review, source = self._review_bundle(bundle, friendly=friendly)
                output = io.StringIO()
                with (
                    patch.object(upscale_cli, "_run_v7_review_ui", return_value={}) as review_ui,
                    patch.object(upscale_cli, "run_v7_job") as render_engine,
                    redirect_stdout(output),
                ):
                    result = upscale_cli.main([command, str(bundle)])

                self.assertEqual(result, 0)
                review_ui.assert_called_once_with(review.resolve(), source.resolve())
                render_engine.assert_not_called()
                rendered = output.getvalue()
                self.assertIn("CHƯA TỰ CHẠY LẠI ENGINE", rendered)
                self.assertIn(str(review.resolve()), rendered)
                self.assertIn("repair", rendered)
                self.assertIn(" 4 ", rendered)

    def test_review_json_inside_bundle_must_be_the_canonical_review(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            bundle = Path(raw) / "bundle"
            self._review_bundle(bundle, friendly=True)
            wrong = bundle / "_KY_THUAT" / "other.json"
            wrong.write_text("{}", encoding="utf-8")

            with self.assertRaises(UserError):
                upscale_cli.resolve_v7_review_target(wrong)

    def test_chrome_is_preferred_over_default_browser(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            chrome = Path(raw) / "chrome.exe"
            chrome.write_bytes(b"exe")
            with (
                patch.object(upscale_cli, "find_chrome_executable", return_value=chrome),
                patch.object(upscale_cli.subprocess, "Popen") as launch,
                patch.object(upscale_cli.webbrowser, "open") as fallback,
            ):
                opened = upscale_cli.open_review_browser("http://127.0.0.1:1234/?token=safe")

            self.assertTrue(opened)
            launch.assert_called_once_with(
                [str(chrome), "--new-window", "http://127.0.0.1:1234/?token=safe"]
            )
            fallback.assert_not_called()

    def test_review_command_requires_exactly_one_target(self) -> None:
        for arguments in (["review"], ["duyet", "one", "two"]):
            with self.subTest(arguments=arguments):
                with self.assertRaises(UserError):
                    upscale_cli.main(arguments)

    def test_cmd_routes_both_review_aliases_to_v7_python(self) -> None:
        command_file = (PROJECT_ROOT / "upscale.cmd").read_text(encoding="utf-8")
        self.assertIn('if /I "%~1"=="review" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\\engines\\V7', command_file)
        self.assertIn('if /I "%~1"=="duyet" set "RESIZE_PYTHON=%RESIZE_ROOT%APP\\engines\\V7', command_file)


if __name__ == "__main__":
    unittest.main()
