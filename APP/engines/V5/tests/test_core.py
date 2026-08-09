from __future__ import annotations

import ast
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image


V5_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(V5_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "APP"))

import upscale_cli  # noqa: E402
import layer_engine_v5  # noqa: E402
from v5lib.formats import (  # noqa: E402
    RenderedLayer,
    _solve_pillow_foreground_u8,
    _spatial_topology_correspondence,
    find_required_alpha_promoted_singletons,
    matte_quality_report,
    project_clean_plate_for_alpha,
    prune_required_alpha_promoted_singletons,
    render_layers,
    rendered_alpha_canvases,
    resize_alpha,
    resize_refined_envelope,
    resize_semantic_support,
)
from v5lib.geometry import dilate_mask  # noqa: E402
from v5lib.model import LayerSpec  # noqa: E402
from v5lib.restore import restore_background  # noqa: E402


class V5CliTests(unittest.TestCase):
    def _write_v5_review_bundle(
        self,
        root: Path,
        *,
        original_source: Path | None = None,
        scale: float = 1.0,
    ) -> Path:
        technical = root / "_KY_THUAT"
        technical.mkdir(parents=True)
        checkpoint = technical / "LAYER_REVIEW.json"
        checkpoint.write_text(
            json.dumps({"schema": "V5_LAYER_REVIEW_V2"}),
            encoding="utf-8",
        )
        manifest: dict[str, object] = {
            "pipeline": "V5_SMART_EDITABLE_LAYERS",
            "engine_generation": "V5_PRO_EXHAUSTIVE_V2",
            "review_resume_signature": upscale_cli.V5_REVIEW_RESUME_SIGNATURE,
            "review_checkpoint": "_KY_THUAT/LAYER_REVIEW.json",
            "scale": scale,
            "detail": "exhaustive",
        }
        if original_source is not None:
            manifest.update(
                {
                    "original_source": str(original_source.resolve()),
                    "original_source_sha256": upscale_cli.sha256_file(original_source),
                }
            )
        (root / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        return checkpoint

    def test_review_auto_routes_v5_output_bundle_by_checkpoint_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "poster_V5_LAYERS_x1"
            checkpoint = self._write_v5_review_bundle(bundle)

            resolved = upscale_cli.resolve_v5_review_target(bundle)

            self.assertIsNotNone(resolved)
            assert resolved is not None
            self.assertEqual(resolved[0], checkpoint.resolve())
            self.assertEqual(resolved[1]["pipeline"], "V5_SMART_EDITABLE_LAYERS")

    def test_review_dispatch_keeps_non_v5_json_on_existing_v7_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary) / "TEXT_REVIEW.json"
            review.write_text(
                json.dumps({"schema": "V7_TEXT_REVIEW_V2"}),
                encoding="utf-8",
            )
            with mock.patch.object(
                upscale_cli, "run_v7_review_command", return_value=23
            ) as run_v7:
                result = upscale_cli.run_review_command(review)

            self.assertEqual(result, 23)
            run_v7.assert_called_once_with(review)

    def test_v5_review_reports_immediate_save_and_required_rerun_truthfully(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "poster.png"
            Image.new("RGB", (8, 8), "white").save(source)
            bundle = root / "bundle"
            checkpoint = self._write_v5_review_bundle(
                bundle, original_source=source, scale=4.0
            )
            output = io.StringIO()
            with mock.patch.object(
                upscale_cli, "_run_v5_review_ui", return_value={}
            ), redirect_stdout(output):
                result = upscale_cli.run_v5_review_command(bundle)

            text = output.getvalue()
            self.assertEqual(result, 0)
            self.assertIn(str(checkpoint.resolve()), text)
            self.assertIn("mỗi thao tác được ghi ngay", text)
            self.assertIn("PSD/ORA HIỆN TẠI CHƯA THAY ĐỔI", text)
            self.assertIn(".\\upscale layers", text)
            self.assertIn("--review gui", text)
            self.assertIn("checkpoint đã lưu sẽ chỉ được áp dụng", text)

    def test_v5_engine_command_receives_existing_review_checkpoint(self) -> None:
        review = Path(r"C:\bundle\_KY_THUAT\LAYER_REVIEW.json")
        command = upscale_cli._build_v5_engine_command(
            staged_input=Path(r"C:\work\source.png"),
            scale=4.0,
            staging_bundle=Path(r"C:\work\new-bundle"),
            master=Path(r"C:\work\master.png"),
            result_stem="poster",
            options={
                "detail": "exhaustive",
                "review": "defer",
                "inpaint": "poster",
                "semantic": True,
            },
            prior_review_file=review,
        )

        review_index = command.index("--review-file")
        self.assertEqual(command[review_index + 1], str(review))
        self.assertEqual(command[command.index("--review-mode") + 1], "defer")

    def test_v5_resume_requires_current_inventory_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "poster.png"
            Image.new("RGB", (8, 8), "white").save(source)
            metadata: dict[str, object] = {
                "pipeline": "V5_SMART_EDITABLE_LAYERS",
                "engine_generation": "V5_PRO_EXHAUSTIVE_V2",
                "review_resume_signature": upscale_cli.V5_REVIEW_RESUME_SIGNATURE,
                "original_source": str(source.resolve()),
                "original_source_sha256": upscale_cli.sha256_file(source),
                "scale": 1.0,
            }
            self.assertTrue(
                upscale_cli._v5_resume_metadata_matches(metadata, source, 1.0)
            )
            metadata.pop("review_resume_signature")
            self.assertFalse(
                upscale_cli._v5_resume_metadata_matches(metadata, source, 1.0)
            )

    def test_v5_hard_qa_fail_is_never_publishable(self) -> None:
        manifest: dict[str, object] = {
            "pipeline": "V5_SMART_EDITABLE_LAYERS",
            "final_size": [100, 80],
            "grouping": {"selected_layer_count": 5},
            "qa_status": "FAIL",
        }
        with self.assertRaisesRegex(RuntimeError, "hard QA failed"):
            upscale_cli._validate_v5_publish_manifest(manifest, (100, 80))
        manifest["qa_status"] = "REVIEW_REQUIRED"
        upscale_cli._validate_v5_publish_manifest(manifest, (100, 80))

    def test_v5_publish_gate_is_wired_only_into_v5_job(self) -> None:
        source_path = PROJECT_ROOT / "APP" / "upscale_cli.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        callers: list[str] = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "_validate_v5_publish_manifest"
                for child in ast.walk(node)
            ):
                callers.append(node.name)
        self.assertEqual(callers, ["run_v5_job"])

    def test_v5_review_signature_is_synced_between_launcher_and_engine(self) -> None:
        engine_path = V5_DIR / "layer_engine_v5_pro.py"
        tree = ast.parse(engine_path.read_text(encoding="utf-8"))
        engine_signature: str | None = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name)
                and target.id == "REVIEW_RESUME_SIGNATURE"
                for target in node.targets
            ):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                engine_signature = node.value.value
        self.assertEqual(engine_signature, upscale_cli.V5_REVIEW_RESUME_SIGNATURE)

    def test_v5_failure_diagnostics_are_small_safe_and_atomically_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            staging = root / "failed-bundle"
            technical = staging / upscale_cli.TECHNICAL_DIR_NAME
            technical.mkdir(parents=True)
            (staging / "manifest.json").write_text(
                json.dumps({"qa_status": "FAIL", "attempt": 1}),
                encoding="utf-8",
            )
            (technical / "QA_REPORT.json").write_text("first-json", encoding="utf-8")
            (technical / "QA_REPORT.html").write_text("first-html", encoding="utf-8")
            (technical / "RECOMPOSITION_DIFF_X8.png").write_bytes(b"diff")
            (technical / "LAYER_REVIEW.json").write_text("review", encoding="utf-8")
            (technical / "SOURCE_FOR_REVIEW.png").write_bytes(b"source")
            (technical / "BACKGROUND_FOR_OCR_QA.png").write_bytes(b"clean")
            layers = staging / "LAYERS"
            layers.mkdir()
            (layers / "must-not-be-copied.png").write_bytes(b"large-layer")

            target = work / "v5_last_failure_bad_poster"
            target.mkdir(parents=True)
            (target / "stale.txt").write_text("stale", encoding="utf-8")

            with mock.patch.object(upscale_cli, "WORK_DIR", work):
                first = upscale_cli._preserve_v5_failure_diagnostics(
                    staging,
                    "bad:poster",
                )
                self.assertEqual(first, target.resolve())
                self.assertEqual(
                    {
                        path.relative_to(first).as_posix()
                        for path in first.rglob("*")
                        if path.is_file()
                    },
                    {
                        "manifest.json",
                        f"{upscale_cli.TECHNICAL_DIR_NAME}/QA_REPORT.json",
                        f"{upscale_cli.TECHNICAL_DIR_NAME}/QA_REPORT.html",
                        f"{upscale_cli.TECHNICAL_DIR_NAME}/RECOMPOSITION_DIFF_X8.png",
                        f"{upscale_cli.TECHNICAL_DIR_NAME}/LAYER_REVIEW.json",
                        f"{upscale_cli.TECHNICAL_DIR_NAME}/SOURCE_FOR_REVIEW.png",
                        f"{upscale_cli.TECHNICAL_DIR_NAME}/BACKGROUND_FOR_OCR_QA.png",
                    },
                )
                self.assertFalse((first / "stale.txt").exists())
                self.assertFalse((first / "LAYERS").exists())

                (staging / "manifest.json").write_text(
                    json.dumps({"qa_status": "FAIL", "attempt": 2}),
                    encoding="utf-8",
                )
                (technical / "QA_REPORT.json").write_text(
                    "second-json",
                    encoding="utf-8",
                )
                second = upscale_cli._preserve_v5_failure_diagnostics(
                    staging,
                    "bad:poster",
                )

            self.assertEqual(second, first)
            self.assertEqual(
                (second / upscale_cli.TECHNICAL_DIR_NAME / "QA_REPORT.json").read_text(
                    encoding="utf-8"
                ),
                "second-json",
            )
            self.assertEqual(
                json.loads((second / "manifest.json").read_text(encoding="utf-8"))[
                    "attempt"
                ],
                2,
            )

    def test_stale_v5_output_staging_cleanup_is_exact_and_preserves_active(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            target = parent / "poster_V5_LAYERS_x1"
            target.mkdir()
            (target / "published.txt").write_text("keep", encoding="utf-8")
            stale = parent / f".{target.name}.new-{'a' * 32}"
            stale.mkdir()
            (stale / "partial.txt").write_text("remove", encoding="utf-8")
            active = parent / f".{target.name}.new-{'b' * 32}"
            active.mkdir()
            (active / "active.txt").write_text("keep", encoding="utf-8")
            near_matches = [
                parent / f".{target.name}.new-{'c' * 31}",
                parent / f".{target.name}.new-{'G' * 32}",
                parent / f".{target.name}.new-{'d' * 32}-extra",
                parent / f".another-target.new-{'e' * 32}",
                parent / ".unrelated-dot-directory",
            ]
            for path in near_matches:
                path.mkdir()

            removed = upscale_cli.cleanup_stale_v5_output_staging(
                target,
                active_staging=active,
            )

            self.assertEqual(removed, [stale.resolve()])
            self.assertFalse(stale.exists())
            self.assertEqual((target / "published.txt").read_text(encoding="utf-8"), "keep")
            self.assertEqual((active / "active.txt").read_text(encoding="utf-8"), "keep")
            self.assertTrue(all(path.is_dir() for path in near_matches))

    def test_stale_v5_output_staging_cleanup_refuses_reparse_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            target = parent / "poster_V5_LAYERS_x1"
            target.mkdir()
            candidate = parent / f".{target.name}.new-{'f' * 32}"
            candidate.mkdir()
            sentinel = candidate / "do-not-delete.txt"
            sentinel.write_text("keep", encoding="utf-8")
            original_check = upscale_cli._is_link_or_reparse_point

            def fake_reparse_check(path: Path) -> bool:
                if path == candidate:
                    return True
                return original_check(path)

            with mock.patch.object(
                upscale_cli,
                "_is_link_or_reparse_point",
                side_effect=fake_reparse_check,
            ):
                with self.assertRaisesRegex(RuntimeError, "link/reparse point"):
                    upscale_cli.cleanup_stale_v5_output_staging(target)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_run_v5_job_never_installs_a_hard_fail_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "poster.png"
            Image.new("RGB", (8, 8), "white").save(source)
            work = root / "work"
            work.mkdir()
            captured: dict[str, Path] = {}

            def fake_build_command(**kwargs: object) -> list[str]:
                staging = kwargs["staging_bundle"]
                assert isinstance(staging, Path)
                captured["staging"] = staging
                return ["fake-v5-engine"]

            def fake_subprocess_run(*_args: object, **_kwargs: object) -> object:
                staging = captured["staging"]
                staging.mkdir(parents=True)
                technical = staging / upscale_cli.TECHNICAL_DIR_NAME
                technical.mkdir()
                (staging / "manifest.json").write_text(
                    json.dumps(
                        {
                            "pipeline": "V5_SMART_EDITABLE_LAYERS",
                            "final_size": [8, 8],
                            "grouping": {"selected_layer_count": 1},
                            "qa_status": "FAIL",
                        }
                    ),
                    encoding="utf-8",
                )
                (technical / "QA_REPORT.json").write_text(
                    json.dumps({"status": "FAIL", "hard_failures": ["unsafe"]}),
                    encoding="utf-8",
                )
                (technical / "QA_REPORT.html").write_text(
                    "<html>unsafe</html>",
                    encoding="utf-8",
                )
                (technical / "RECOMPOSITION_DIFF_X8.png").write_bytes(b"diff")
                (technical / "LAYER_REVIEW.json").write_text(
                    "review",
                    encoding="utf-8",
                )
                (technical / "SOURCE_FOR_REVIEW.png").write_bytes(b"source")
                (technical / "BACKGROUND_FOR_OCR_QA.png").write_bytes(b"clean")
                layers = staging / "LAYERS"
                layers.mkdir()
                (layers / "must-not-survive.png").write_bytes(b"layer")
                return object()

            resource_plan = {
                "output_megapixels": 0.000064,
                "estimated_peak_ram_gib": 0.01,
                "estimated_working_disk_gib": 0.01,
            }
            options: dict[str, object] = {
                "detail": "exhaustive",
                "review": "defer",
                "inpaint": "auto",
                "semantic": True,
                "max_layers": 24,
            }
            last_good = root / "output" / "V5_LAYERS" / "poster_V5_LAYERS_x1"
            last_good.mkdir(parents=True)
            sentinel = last_good / "last-good.txt"
            sentinel.write_text("keep", encoding="utf-8")
            stderr = io.StringIO()
            with (
                mock.patch.object(upscale_cli, "OUTPUT_DIR", root / "output"),
                mock.patch.object(upscale_cli, "WORK_DIR", work),
                mock.patch.object(upscale_cli, "APP_DIR", root / "app"),
                mock.patch.object(upscale_cli, "PYTHON_V5", source),
                mock.patch.object(upscale_cli, "V5_ENGINE", source),
                mock.patch.object(
                    upscale_cli,
                    "inspect_image",
                    return_value=((8, 8), "RGB", None),
                ),
                mock.patch.object(
                    upscale_cli,
                    "validate_v5_resource_plan",
                    return_value=resource_plan,
                ),
                mock.patch.object(upscale_cli, "stage_input", return_value=source),
                mock.patch.object(
                    upscale_cli,
                    "canonical_pixel_sha256",
                    return_value="0" * 64,
                ),
                mock.patch.object(
                    upscale_cli,
                    "prepare_v5_ai_master",
                    return_value=(source, {}),
                ),
                mock.patch.object(
                    upscale_cli,
                    "_build_v5_engine_command",
                    side_effect=fake_build_command,
                ),
                mock.patch.object(
                    upscale_cli.subprocess,
                    "run",
                    side_effect=fake_subprocess_run,
                ),
                mock.patch.object(
                    upscale_cli,
                    "resolve_v5_review_target",
                    return_value=None,
                ),
                redirect_stderr(stderr),
            ):
                with self.assertRaisesRegex(RuntimeError, "hard QA failed"):
                    upscale_cli.run_v5_job(source, 1.0, False, options)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            diagnostic = work / "v5_last_failure_poster"
            self.assertTrue(diagnostic.is_dir())
            self.assertEqual(
                {
                    path.relative_to(diagnostic).as_posix()
                    for path in diagnostic.rglob("*")
                    if path.is_file()
                },
                {
                    "manifest.json",
                    f"{upscale_cli.TECHNICAL_DIR_NAME}/QA_REPORT.json",
                    f"{upscale_cli.TECHNICAL_DIR_NAME}/QA_REPORT.html",
                    f"{upscale_cli.TECHNICAL_DIR_NAME}/RECOMPOSITION_DIFF_X8.png",
                    f"{upscale_cli.TECHNICAL_DIR_NAME}/LAYER_REVIEW.json",
                    f"{upscale_cli.TECHNICAL_DIR_NAME}/SOURCE_FOR_REVIEW.png",
                    f"{upscale_cli.TECHNICAL_DIR_NAME}/BACKGROUND_FOR_OCR_QA.png",
                },
            )
            self.assertFalse((diagnostic / "LAYERS").exists())
            self.assertIn(str(diagnostic), stderr.getvalue())
            self.assertFalse(captured["staging"].exists())

    def test_v5_failure_diagnostics_skip_oversized_optional_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            staging = root / "failed-bundle"
            technical = staging / upscale_cli.TECHNICAL_DIR_NAME
            technical.mkdir(parents=True)
            (staging / "manifest.json").write_text(
                json.dumps({"qa_status": "FAIL"}), encoding="utf-8"
            )
            (technical / "QA_REPORT.json").write_bytes(b"q")
            (technical / "QA_REPORT.html").write_bytes(b"h")
            (technical / "RECOMPOSITION_DIFF_X8.png").write_bytes(b"d" * 30)
            (technical / "LAYER_REVIEW.json").write_bytes(b"r" * 30)
            (technical / "SOURCE_FOR_REVIEW.png").write_bytes(b"s" * 65)
            (technical / "BACKGROUND_FOR_OCR_QA.png").write_bytes(b"b")

            with (
                mock.patch.object(upscale_cli, "WORK_DIR", work),
                mock.patch.object(
                    upscale_cli,
                    "V5_FAILURE_DIAGNOSTIC_MAX_FILE_BYTES",
                    64,
                ),
                mock.patch.object(
                    upscale_cli,
                    "V5_FAILURE_DIAGNOSTIC_MAX_TOTAL_BYTES",
                    80,
                ),
            ):
                diagnostic = upscale_cli._preserve_v5_failure_diagnostics(
                    staging, "bounded"
                )

            self.assertIsNotNone(diagnostic)
            assert diagnostic is not None
            files = {
                path.relative_to(diagnostic).as_posix()
                for path in diagnostic.rglob("*")
                if path.is_file()
            }
            self.assertEqual(
                files,
                {
                    "manifest.json",
                    f"{upscale_cli.TECHNICAL_DIR_NAME}/QA_REPORT.json",
                    f"{upscale_cli.TECHNICAL_DIR_NAME}/QA_REPORT.html",
                    f"{upscale_cli.TECHNICAL_DIR_NAME}/RECOMPOSITION_DIFF_X8.png",
                    f"{upscale_cli.TECHNICAL_DIR_NAME}/BACKGROUND_FOR_OCR_QA.png",
                },
            )
            self.assertLessEqual(
                sum(path.stat().st_size for path in diagnostic.rglob("*") if path.is_file()),
                80,
            )

    def test_cmd_review_runtime_does_not_require_v7_environment(self) -> None:
        command_file = (PROJECT_ROOT / "upscale.cmd").read_text(encoding="utf-8")

        for alias in ("review", "duyet"):
            self.assertIn(
                f'if /I "%~1"=="{alias}" if exist '
                '"%RESIZE_ROOT%APP\\engines\\V3\\.venv\\Scripts\\python.exe"',
                command_file,
            )
            self.assertIn(
                f'if /I "%~1"=="{alias}" if not exist '
                '"%RESIZE_ROOT%APP\\engines\\V3\\.venv\\Scripts\\python.exe" if exist '
                '"%RESIZE_ROOT%APP\\engines\\V5\\.venv\\Scripts\\python.exe"',
                command_file,
            )

    def test_v5_batch_turns_gui_review_into_defer_for_every_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "one.png"
            Image.new("RGB", (8, 8), "white").save(source)
            received: list[dict[str, object]] = []

            def fake_run_job(
                _mode: str,
                _source: Path,
                _scale: float,
                _allow_huge: bool,
                options: dict[str, object],
                **_kwargs: object,
            ) -> Path:
                received.append(dict(options))
                return directory / "result"

            with mock.patch.object(upscale_cli, "PYTHON_V5", source), mock.patch.object(
                upscale_cli, "V5_ENGINE", source
            ), mock.patch.object(upscale_cli, "run_job", side_effect=fake_run_job):
                result = upscale_cli.run_batch(
                    "V5_LAYERS",
                    directory,
                    1.0,
                    False,
                    {"review": "gui"},
                )

            self.assertEqual(result, 0)
            self.assertEqual([item["review"] for item in received], ["defer"])

    def test_topology_failure_summary_names_layer_threshold_and_relation(self) -> None:
        report = {
            "layers": [
                {
                    "layer_id": "text_08",
                    "scaled_topology": {
                        "passed": False,
                        "thresholds": [
                            {
                                "passed": False,
                                "integer_alpha_level": 128,
                                "orphan_actual_components": 12,
                                "new_actual_holes": 3,
                                "missing_expected_pixels": 0,
                            }
                        ],
                    },
                }
            ]
        }

        summary = layer_engine_v5.summarize_topology_failures(report)

        self.assertEqual(
            summary,
            ["text_08@128:orphan_actual_components=12,new_actual_holes=3"],
        )

    def test_promoted_singleton_budget_is_cumulative_across_preflight_passes(self) -> None:
        first = layer_engine_v5.reserve_promoted_singleton_budget(
            [{"layer_id": "text"}, {"layer_id": "text"}],
            {},
            maximum_per_layer=2,
        )
        self.assertEqual(first, {"text": 2})

        with self.assertRaisesRegex(RuntimeError, "safe cumulative cap"):
            layer_engine_v5.reserve_promoted_singleton_budget(
                [{"layer_id": "text"}],
                first,
                maximum_per_layer=2,
            )
        # Reservation is functional: a failed later pass cannot corrupt the
        # accepted count recorded for the prior pass.
        self.assertEqual(first, {"text": 2})

    def test_v5_accepts_source_scale_and_options(self) -> None:
        mode, token, scale, allow_huge, options = upscale_cli.parse_command(
            [
                "layers",
                "poster.png",
                "1",
                "--detail",
                "exhaustive",
                "--review",
                "defer",
                "--inpaint",
                "poster",
            ]
        )
        self.assertEqual(mode, "V5_LAYERS")
        self.assertEqual(token, "poster.png")
        self.assertEqual(scale, 1.0)
        self.assertFalse(allow_huge)
        self.assertEqual(options["detail"], "exhaustive")
        self.assertEqual(options["review"], "defer")
        self.assertEqual(options["inpaint"], "poster")

    def test_v5_rejects_fractional_scale_before_rendering_masks(self) -> None:
        with self.assertRaisesRegex(upscale_cli.UserError, "integer scale"):
            upscale_cli.parse_command(["layers", "poster.png", "1.5"])

    def test_v5_only_options_are_rejected_by_v3(self) -> None:
        with self.assertRaises(upscale_cli.UserError):
            upscale_cli.parse_command(["high", "poster.png", "4", "--max-layers", "18"])

    def test_v5_hard_canvas_cap_cannot_be_bypassed(self) -> None:
        with self.assertRaises(upscale_cli.UserError):
            upscale_cli.validate_v5_resource_plan(
                (4000, 4000), (24_000, 24_000), allow_huge=True
            )

    def test_v5_resource_plan_is_independent_of_legacy_layer_cap(self) -> None:
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
        self.assertEqual(large["estimated_peak_ram_gib"], small["estimated_peak_ram_gib"])
        self.assertEqual(large["estimated_working_disk_gib"], small["estimated_working_disk_gib"])
        self.assertIn("no hard layer cap", str(large["layer_policy"]))

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


class SpatialTopologyGateTests(unittest.TestCase):
    def test_canonical_x1_render_is_reference_and_contract_is_strict(self) -> None:
        shape = (24, 42)
        mask = np.zeros(shape, dtype=bool)
        source_alpha = np.zeros(shape, dtype=np.float32)
        mask[8:17, 5:14] = True
        mask[8:17, 24:33] = True
        mask[11:14, 14:24] = True
        source_alpha[8:17, 5:14] = 1.0
        source_alpha[8:17, 24:33] = 1.0
        source_alpha[11:14, 14:24] = 0.1
        spec = LayerSpec(
            "canonical",
            "Canonical x1",
            "text_raster",
            mask,
            0.99,
            metadata={"grouping_role": "visual_text_row"},
            alpha_matte=source_alpha,
        )
        canonical_alpha = np.zeros(shape, dtype=np.uint8)
        canonical_alpha[mask] = 255
        rgba = np.zeros((*shape, 4), dtype=np.uint8)
        rgba[..., 3] = canonical_alpha
        item = RenderedLayer(
            spec,
            Image.fromarray(rgba, "RGBA"),
            Image.fromarray(canonical_alpha, "L"),
            0,
            0,
            spec.bbox,
        )
        canonical = rendered_alpha_canvases(
            [item], (shape[1], shape[0]), refined_only=True
        )
        canonical_layer = layer_engine_v5.apply_canonical_refined_alpha(
            [spec], canonical
        )[0]

        fallback = matte_quality_report([item], (shape[1], shape[0]))
        report = matte_quality_report(
            [item],
            (shape[1], shape[0]),
            topology_reference=canonical,
        )

        self.assertFalse(fallback["topology_gate_passed"], fallback)
        self.assertTrue(report["topology_gate_passed"], report)
        self.assertEqual(
            report["topology_reference_source"],
            "stable_source_resolution_preflight_render",
        )
        self.assertTrue(np.array_equal(canonical["canonical"], canonical_alpha))
        self.assertAlmostEqual(float(spec.alpha_matte[12, 18]), 0.1, places=6)
        self.assertEqual(float(canonical_layer.alpha_matte[12, 18]), 1.0)
        self.assertTrue(np.array_equal(canonical_layer.mask, spec.mask))
        self.assertEqual(canonical_layer.metadata, spec.metadata)
        self.assertIsNot(canonical_layer.mask, spec.mask)
        x4_size = (shape[1] * 4, shape[0] * 4)
        scaled_canonical = np.rint(
            resize_alpha(
                canonical_layer.mask,
                x4_size,
                alpha_matte=canonical_layer.alpha_matte,
            )
            * 255.0
        ).astype(np.uint8)
        expected_scaled = np.asarray(
            Image.fromarray(canonical_alpha, "L").resize(
                x4_size, Image.Resampling.LANCZOS
            ),
            dtype=np.uint8,
        )
        self.assertTrue(np.array_equal(scaled_canonical, expected_scaled))
        with self.assertRaisesRegex(ValueError, "Missing canonical"):
            matte_quality_report(
                [item],
                (shape[1], shape[0]),
                topology_reference={},
            )
        with self.assertRaisesRegex(ValueError, "two-dimensional uint8"):
            matte_quality_report(
                [item],
                (shape[1], shape[0]),
                topology_reference={"canonical": canonical_alpha.astype(np.float32)},
            )
        with self.assertRaisesRegex(ValueError, "does not match source mask"):
            matte_quality_report(
                [item],
                (shape[1], shape[0]),
                topology_reference={"canonical": canonical_alpha[:-1]},
            )
        escaped = canonical_alpha.copy()
        escaped[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "escapes semantic mask"):
            layer_engine_v5.apply_canonical_refined_alpha(
                [spec], {"canonical": escaped}
            )

    def test_count_preserving_merge_plus_orphan_is_rejected(self) -> None:
        expected = np.zeros((32, 56), dtype=bool)
        expected[11:22, 6:14] = True
        expected[11:22, 24:32] = True
        actual = expected.copy()
        actual[15:18, 14:24] = True  # merge the two expected glyphs
        actual[5:9, 45:49] = True  # compensating orphan keeps total CC=2

        report = _spatial_topology_correspondence(
            expected, actual, scale_x=1.0, scale_y=1.0
        )

        self.assertEqual(report["expected_foreground_component_count_fg8"], 2)
        self.assertEqual(report["actual_foreground_component_count_fg8"], 2)
        self.assertEqual(report["merged_actual_components"], 1)
        self.assertEqual(report["orphan_actual_components"], 1)
        self.assertFalse(report["passed"])

    def test_filled_o_plus_new_c_counter_is_rejected_even_when_hole_count_matches(self) -> None:
        expected = np.zeros((28, 54), dtype=bool)
        expected[5:22, 4:21] = True
        expected[10:17, 9:16] = False  # O counter
        expected[5:22, 31:48] = True
        expected[10:17, 36:44] = False
        expected[12:15, 43:49] = False  # open C/G-style aperture
        actual = expected.copy()
        actual[10:17, 9:16] = True  # fill O
        actual[12:15, 43:48] = True  # close C and create a new counter

        report = _spatial_topology_correspondence(
            expected, actual, scale_x=1.0, scale_y=1.0
        )

        self.assertEqual(report["expected_enclosed_background_count_bg4"], 1)
        self.assertEqual(report["actual_enclosed_background_count_bg4"], 1)
        self.assertEqual(report["missing_expected_holes"], 1)
        self.assertEqual(report["new_actual_holes"], 1)
        self.assertGreater(report["expected_hole_core_intrusion_pixels"], 0)
        self.assertFalse(report["passed"])

    def test_attached_edge_promotion_passes_and_tight_crop_matches_full_canvas(self) -> None:
        expected = np.zeros((44, 62), dtype=bool)
        expected[15:29, 20:38] = True
        actual = expected.copy()
        actual[14, 20:38] = True  # thicker edge, no component/counter change

        full = _spatial_topology_correspondence(
            expected, actual, scale_x=4.0, scale_y=4.0
        )
        tight = _spatial_topology_correspondence(
            expected[14:29, 20:38],
            actual[14:29, 20:38],
            scale_x=4.0,
            scale_y=4.0,
        )

        self.assertTrue(full["passed"], full)
        self.assertEqual(tight, full)

    def test_exact_integer_alpha_thresholds_are_used(self) -> None:
        shape = (12, 24)
        levels = (63, 64, 127, 128, 191, 192)
        mask = np.zeros(shape, dtype=bool)
        source_alpha = np.zeros(shape, dtype=np.float32)
        actual_alpha = np.zeros(shape, dtype=np.uint8)
        for index, level in enumerate(levels):
            x = 2 + index * 3
            mask[6, x] = True
            source_alpha[6, x] = level / 255.0
            actual_alpha[6, x] = level
        spec = LayerSpec(
            "thresholds",
            "Thresholds",
            "text_raster",
            mask,
            0.99,
            metadata={"grouping_role": "standalone_row"},
            alpha_matte=source_alpha,
        )
        rgba = np.zeros((*shape, 4), dtype=np.uint8)
        rgba[..., 3] = actual_alpha
        item = RenderedLayer(
            spec,
            Image.fromarray(rgba, "RGBA"),
            Image.fromarray(actual_alpha, "L"),
            0,
            0,
            spec.bbox,
        )

        report = matte_quality_report([item], (shape[1], shape[0]))

        self.assertTrue(report["topology_gate_passed"], report)
        thresholds = report["layers"][0]["scaled_topology"]["thresholds"]
        self.assertEqual([record["integer_alpha_level"] for record in thresholds], [64, 128, 192])
        self.assertEqual(
            [record["expected_foreground_component_count_fg8"] for record in thresholds],
            [5, 3, 1],
        )

    def test_standalone_low_alpha_corruption_inside_semantic_is_release_blocked(self) -> None:
        shape = (28, 46)
        mask = np.zeros(shape, dtype=bool)
        source_alpha = np.zeros(shape, dtype=np.float32)
        mask[8:22, 8:22] = True
        source_alpha[8:22, 8:22] = 1.0
        mask[24, 40] = True
        source_alpha[24, 40] = 0.1
        spec = LayerSpec(
            "standalone",
            "Standalone row",
            "text_raster",
            mask,
            0.99,
            metadata={"grouping_role": "standalone_row"},
            alpha_matte=source_alpha,
        )
        actual_alpha = np.ceil(source_alpha * 255.0 - 1e-7).astype(np.uint8)
        actual_alpha[24, 40] = 255  # required-alpha style promotion inside mask
        rgba = np.zeros((*shape, 4), dtype=np.uint8)
        rgba[..., 3] = actual_alpha
        item = RenderedLayer(
            spec,
            Image.fromarray(rgba, "RGBA"),
            Image.fromarray(actual_alpha, "L"),
            0,
            0,
            spec.bbox,
        )

        report = matte_quality_report([item], (shape[1], shape[0]))

        self.assertTrue(report["ownership_gate_passed"], report)
        self.assertEqual(report["topology_checked_layer_count"], 1)
        self.assertFalse(report["topology_gate_passed"], report)
        first = report["layers"][0]["scaled_topology"]["thresholds"][0]
        self.assertEqual(first["orphan_actual_components"], 1)


class ExactPillowAlphaTests(unittest.TestCase):
    @staticmethod
    def _expected_projected_lower(
        target: np.ndarray,
        clean: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        target_i = target.astype(np.int32)
        clean_i = clean.astype(np.int32)
        alpha_i = alpha.astype(np.int32)[..., None]
        inverse = 255 - alpha_i
        safe_inverse = np.maximum(inverse, 1)
        lower_numerator = 255 * target_i - 127 - 255 * alpha_i
        lower = -np.floor_divide(-lower_numerator, safe_inverse)
        upper = np.floor_divide(255 * target_i + 127, safe_inverse)
        lower = np.clip(lower, 0, 255)
        upper = np.clip(upper, 0, 255)
        lower = np.where(alpha_i == 255, 0, lower)
        upper = np.where(alpha_i == 255, 255, upper)
        return np.minimum(np.maximum(clean_i, lower), upper).astype(np.uint8)

    def test_projection_exhausts_every_alpha_target_pair_and_strategic_clean_colours(self) -> None:
        alpha_levels = np.arange(256, dtype=np.uint8)
        target_levels = np.arange(256, dtype=np.uint8)
        clean_levels = np.asarray((0, 1, 63, 127, 128, 192, 254, 255), dtype=np.uint8)
        alpha, target, clean = np.meshgrid(
            alpha_levels,
            target_levels,
            clean_levels,
            indexing="ij",
        )
        target_rgb = np.repeat(target.reshape(1, -1, 1), 3, axis=2)
        clean_rgb = np.repeat(clean.reshape(1, -1, 1), 3, axis=2)
        alpha_canvas = alpha.reshape(1, -1)

        projected, report = project_clean_plate_for_alpha(
            target_rgb,
            clean_rgb,
            alpha_canvas,
        )
        expected = self._expected_projected_lower(
            target_rgb,
            clean_rgb,
            alpha_canvas,
        )
        foreground, feasible = _solve_pillow_foreground_u8(
            target_rgb,
            projected,
            alpha_canvas,
        )
        alpha_i = alpha_canvas.astype(np.int32)[..., None]
        composed = np.floor_divide(
            alpha_i * foreground.astype(np.int32)
            + (255 - alpha_i) * projected.astype(np.int32)
            + 127,
            255,
        ).astype(np.uint8)

        self.assertTrue(np.array_equal(projected, expected))
        self.assertTrue(feasible.all())
        self.assertTrue(np.array_equal(composed, target_rgb))
        self.assertEqual(report["serialized_alpha_safety_margin_levels"], 0)
        self.assertEqual(report["maximum_exact_required_alpha_excess_levels"], 0)
        self.assertEqual(report["exact_pillow_recomposition_max_abs_error"], 0)
        self.assertTrue(report["feasibility_gate_passed"])

    def test_projection_random_rgb_is_nearest_and_preserves_endpoint_contracts(self) -> None:
        rng = np.random.default_rng(20260805)
        shape = (257, 389)
        target = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
        clean = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
        alpha = rng.integers(0, 256, size=shape, dtype=np.uint8)
        alpha[0] = 0
        alpha[-1] = 255

        projected, report = project_clean_plate_for_alpha(target, clean, alpha)
        expected = self._expected_projected_lower(target, clean, alpha)
        foreground, feasible = _solve_pillow_foreground_u8(
            target, projected, alpha
        )
        pillow_composite = Image.alpha_composite(
            Image.fromarray(
                np.dstack(
                    (projected, np.full(shape, 255, dtype=np.uint8))
                ),
                "RGBA",
            ),
            Image.fromarray(np.dstack((foreground, alpha)), "RGBA"),
        )

        self.assertTrue(np.array_equal(projected, expected))
        self.assertTrue(np.array_equal(projected[0], target[0]))
        self.assertTrue(np.array_equal(projected[-1], clean[-1]))
        self.assertTrue(feasible.all())
        self.assertTrue(
            np.array_equal(np.asarray(pillow_composite, dtype=np.uint8)[..., :3], target)
        )
        self.assertTrue(report["feasibility_gate_passed"])

        unchanged_target, unchanged_report = project_clean_plate_for_alpha(
            target,
            target,
            alpha,
        )
        self.assertTrue(np.array_equal(unchanged_target, target))
        self.assertEqual(unchanged_report["changed_pixel_count"], 0)

    def test_renderer_does_not_promote_alpha_when_pillow_rounding_is_exact(self) -> None:
        # Continuous source-over needs alpha 1/155 for 100 -> 101, which would
        # serialize to A=2. Pillow's integer rounding can reproduce it at A=1
        # with an integer F from 228 through 255, so the canonical one-level
        # matte must remain untouched.
        master = np.full((1, 1, 3), 101, dtype=np.uint8)
        lower = np.full((1, 1, 3), 100, dtype=np.uint8)
        mask = np.ones((1, 1), dtype=bool)
        spec = LayerSpec(
            "rounding-edge",
            "Rounding edge",
            "text_raster",
            mask,
            0.99,
            metadata={"grouping_role": "visual_text_row"},
            alpha_matte=np.full((1, 1), 1.0 / 255.0, dtype=np.float32),
        )

        rendered, composite, report = render_layers(master, lower, [spec])

        self.assertEqual(len(rendered), 1)
        self.assertEqual(int(np.asarray(rendered[0].alpha, dtype=np.uint8)[0, 0]), 1)
        self.assertEqual(int(np.asarray(rendered[0].rgba, dtype=np.uint8)[0, 0, 0]), 255)
        self.assertTrue(np.array_equal(np.asarray(composite), master))
        self.assertEqual(report["recomposition_max_abs_error"], 0)


class CompositionTests(unittest.TestCase):
    def test_x4_refined_text_envelope_stays_soft_and_topology_is_release_gated(self) -> None:
        source_height, source_width = 36, 76
        mask_u8 = np.zeros((source_height, source_width), dtype=np.uint8)
        # O: one intentional counter.
        cv2.ellipse(mask_u8, (18, 18), (12, 14), 0, 0, 360, 1, -1, cv2.LINE_8)
        cv2.ellipse(mask_u8, (18, 18), (6, 8), 0, 0, 360, 0, -1, cv2.LINE_8)
        # G: an inner negative region with a deliberately open right aperture.
        cv2.ellipse(mask_u8, (55, 18), (13, 14), 0, 0, 360, 1, -1, cv2.LINE_8)
        cv2.ellipse(mask_u8, (55, 18), (7, 8), 0, 0, 360, 0, -1, cv2.LINE_8)
        mask_u8[12:19, 60:76] = 0
        mask_u8[19:22, 55:68] = 1
        mask = mask_u8.astype(bool)
        distance = cv2.distanceTransform(
            mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        source_alpha = np.clip(distance / 2.0, 0.0, 1.0).astype(np.float32)
        source_alpha[~mask] = 0.0
        spec = LayerSpec(
            "og",
            "O G",
            "text_raster",
            mask,
            0.99,
            metadata={"grouping_role": "visual_text_row"},
            alpha_matte=source_alpha,
        )
        final_size = (source_width * 4, source_height * 4)
        semantic = resize_semantic_support(mask, final_size)
        envelope = resize_refined_envelope(mask, final_size)
        base_alpha = resize_alpha(
            mask, final_size, alpha_matte=source_alpha
        ) * envelope

        master = np.full(
            (final_size[1], final_size[0], 3), (8, 42, 8), dtype=np.uint8
        )
        synthesized_lower = np.full_like(master, (190, 20, 180))
        foreground = np.asarray((244, 218, 20), dtype=np.float32)
        faithful_edge = np.clip(
            foreground * base_alpha[..., None]
            + synthesized_lower.astype(np.float32) * (1.0 - base_alpha[..., None])
            + 0.5,
            0,
            255,
        ).astype(np.uint8)
        master[semantic] = faithful_edge[semantic]

        # Even against a deliberately incompatible lower colour, the renderer
        # may not promote positive Lanczos lobes outside nearest ownership.
        wrong_rendered, _wrong_composite, wrong_report = render_layers(
            master, synthesized_lower, [spec]
        )
        self.assertGreater(wrong_report["recomposition_max_abs_error"], 1)
        wrong_canvas_alpha = np.zeros(semantic.shape, dtype=np.uint8)
        wrong_item = wrong_rendered[0]
        wrong_canvas_alpha[
            wrong_item.top : wrong_item.bbox[3],
            wrong_item.left : wrong_item.bbox[2],
        ] = np.asarray(wrong_item.alpha, dtype=np.uint8)
        expected_base_u8 = np.ceil(
            np.maximum(0.0, base_alpha * 255.0 - 1e-7)
        ).astype(np.uint8)
        self.assertTrue(
            np.array_equal(
                wrong_canvas_alpha[~semantic], expected_base_u8[~semantic]
            )
        )
        self.assertFalse((wrong_canvas_alpha[~semantic] >= 128).any())

        # The production cleanup contract leaves master pixels on the lower
        # layer throughout the antialias envelope and synthesizes only nearest
        # semantic ownership. This restores exact composition without opaque
        # lobe promotion.
        correct_lower = master.copy()
        correct_lower[semantic] = synthesized_lower[semantic]
        rendered, composite, report = render_layers(master, correct_lower, [spec])
        self.assertLessEqual(report["recomposition_max_abs_error"], 1)
        self.assertTrue(np.array_equal(np.asarray(composite), master))
        item = rendered[0]
        canvas_alpha = np.zeros(semantic.shape, dtype=np.uint8)
        canvas_alpha[item.top : item.bbox[3], item.left : item.bbox[2]] = np.asarray(
            item.alpha, dtype=np.uint8
        )
        self.assertGreater(
            int(np.logical_and(canvas_alpha > 0, ~semantic).sum()), 0
        )
        self.assertFalse((canvas_alpha[~semantic] >= 128).any())
        high_count = cv2.connectedComponents(
            (canvas_alpha >= 128).astype(np.uint8), 8
        )[0] - 1
        self.assertEqual(high_count, 2)
        self.assertEqual(int(canvas_alpha[18 * 4, 18 * 4]), 0)  # O counter
        self.assertEqual(int(canvas_alpha[15 * 4, 68 * 4]), 0)  # G aperture

        canonical_reference = {
            "og": np.clip(source_alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
        }
        matte_qa = matte_quality_report(
            rendered,
            final_size,
            topology_reference=canonical_reference,
        )
        self.assertTrue(matte_qa["ownership_gate_passed"], matte_qa)
        self.assertTrue(matte_qa["topology_gate_passed"], matte_qa)
        topology = matte_qa["layers"][0]["scaled_topology"]
        middle = next(
            record for record in topology["thresholds"] if record["threshold"] == 0.5
        )
        self.assertEqual(
            middle["actual_foreground_component_count_fg8"], 2
        )
        self.assertEqual(middle["actual_enclosed_background_count_bg4"], 1)

        # A legal-envelope alpha can still be unusable. Simulate the former
        # required_alpha failure by planting separated opaque lobe pixels: the
        # ownership gate alone passes, while the new topology release gate must
        # reject component/microcomponent proliferation.
        corrupted_alpha = expected_base_u8.copy()
        baseline = cv2.dilate(
            (corrupted_alpha >= 64).astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
        ).astype(bool)
        candidate = envelope & ~semantic & ~baseline
        selected = np.zeros(candidate.shape, dtype=bool)
        for y, x in np.argwhere(candidate):
            selected_halo = cv2.dilate(
                selected.astype(np.uint8),
                np.ones((5, 5), dtype=np.uint8),
            ).astype(bool)
            if selected_halo[y, x]:
                continue
            selected[y, x] = True
            if int(selected.sum()) == 20:
                break
        self.assertEqual(int(selected.sum()), 20)
        corrupted_alpha[selected] = 255
        corrupted_rgba = np.zeros((*corrupted_alpha.shape, 4), dtype=np.uint8)
        corrupted_rgba[..., 3] = corrupted_alpha
        corrupted = RenderedLayer(
            spec=spec,
            rgba=Image.fromarray(corrupted_rgba, "RGBA"),
            alpha=Image.fromarray(corrupted_alpha, "L"),
            left=0,
            top=0,
            source_bbox=spec.bbox,
        )
        corrupted_qa = matte_quality_report(
            [corrupted],
            final_size,
            topology_reference=canonical_reference,
        )
        self.assertTrue(corrupted_qa["ownership_gate_passed"], corrupted_qa)
        self.assertFalse(corrupted_qa["topology_gate_passed"], corrupted_qa)
        self.assertGreater(corrupted_qa["topology_failed_threshold_count"], 0)

    def test_promoted_singleton_guard_caps_automatic_removal_per_layer(self) -> None:
        height, width = 60, 100
        lower = np.full((height, width, 3), (20, 70, 20), dtype=np.uint8)
        master = lower.copy()
        mask = np.zeros((height, width), dtype=bool)
        alpha = np.zeros((height, width), dtype=np.float32)
        mask[12:42, 10:35] = True
        alpha[12:42, 10:35] = 1.0
        master[12:42, 10:35] = (245, 218, 18)
        for x in (60, 72, 84):
            mask[52, x] = True
            alpha[52, x] = 0.12
            master[52, x] = (0, 50, 0)
        spec = LayerSpec(
            "text",
            "Text row",
            "text_raster",
            mask,
            0.99,
            metadata={"grouping_role": "ocr_text_line"},
            alpha_matte=alpha,
        )

        rendered, _composite, report = render_layers(master, lower, [spec])
        self.assertEqual(report["recomposition_max_abs_error"], 0)
        records = find_required_alpha_promoted_singletons(
            rendered, (width, height), master
        )

        self.assertEqual(len(records), 2, records)
        self.assertEqual({record["layer_id"] for record in records}, {"text"})

    def test_promoted_text_singleton_moves_to_lower_layer_but_real_marks_survive(self) -> None:
        height, width = 64, 104
        lower = np.full((height, width, 3), (20, 70, 20), dtype=np.uint8)
        master = lower.copy()
        mask = np.zeros((height, width), dtype=bool)
        alpha = np.zeros((height, width), dtype=np.float32)

        mask[20:48, 18:43] = True
        alpha[20:48, 18:43] = 1.0
        master[20:48, 18:43] = (245, 218, 18)
        # A normal-width detached accent must survive.
        mask[8, 22:30] = True
        alpha[8, 22:30] = 1.0
        master[8, 22:30] = (245, 218, 18)
        # Even a true one-pixel mark with a low model alpha is protected by its
        # source colour being unlike the locally supported green background.
        mask[10, 72] = True
        alpha[10, 72] = 0.18
        master[10, 72] = (245, 218, 18)
        # This dark-green, low-matte pixel is locally continuous with the panel
        # but required_alpha must promote it after the lower layer was cleaned.
        false_y, false_x = 54, 88
        mask[false_y, false_x] = True
        alpha[false_y, false_x] = 0.12
        master[false_y, false_x] = (0, 50, 0)

        spec = LayerSpec(
            "text",
            "Text row",
            "text_raster",
            mask,
            0.99,
            metadata={"grouping_role": "visual_text_row", "hierarchy_depth": 1},
            alpha_matte=alpha,
        )
        rendered, composite, report = render_layers(master, lower, [spec])
        self.assertLessEqual(report["recomposition_max_abs_error"], 1)
        self.assertTrue(np.array_equal(np.asarray(composite), master))

        records = find_required_alpha_promoted_singletons(
            rendered,
            (width, height),
            master,
        )
        self.assertEqual(
            [(record["x"], record["y"]) for record in records],
            [(false_x, false_y)],
            records,
        )
        cleaned, prune_report = prune_required_alpha_promoted_singletons(
            [spec], records
        )
        self.assertEqual(prune_report["applied_count"], 1)
        self.assertFalse(cleaned[0].mask[false_y, false_x])
        self.assertEqual(float(cleaned[0].alpha_matte[false_y, false_x]), 0.0)
        self.assertTrue(cleaned[0].mask[8, 22:30].all())
        self.assertTrue(cleaned[0].mask[10, 72])

        # Rebuilding the lower target assigns the rejected support to its
        # parent/background; deleting final alpha alone would fail this check.
        rebuilt_lower = lower.copy()
        rebuilt_lower[~cleaned[0].mask] = master[~cleaned[0].mask]
        rerendered, recomposite, rereport = render_layers(
            master, rebuilt_lower, cleaned
        )
        self.assertLessEqual(rereport["recomposition_max_abs_error"], 1)
        self.assertTrue(np.array_equal(np.asarray(recomposite), master))
        self.assertEqual(
            find_required_alpha_promoted_singletons(
                rerendered,
                (width, height),
                master,
            ),
            [],
        )

        alpha_canvas = np.zeros((height, width), dtype=np.uint8)
        item = rerendered[0]
        alpha_canvas[item.top : item.bbox[3], item.left : item.bbox[2]] = np.asarray(
            item.alpha, dtype=np.uint8
        )
        self.assertEqual(int(alpha_canvas[false_y, false_x]), 0)
        self.assertGreaterEqual(int(alpha_canvas[10, 72]), 128)
        self.assertTrue((alpha_canvas[8, 22:30] >= 128).all())

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
                    matte_policy="faithful",
                )

                self.assertLessEqual(report["recomposition_max_abs_error"], 1)
                self.assertLessEqual(
                    int(np.abs(np.asarray(composite, dtype=np.int16) - master.astype(np.int16)).max()),
                    1,
                )

    def test_clean_matte_never_owns_restoration_or_halo_pixels(self) -> None:
        height, width = 36, 48
        semantic = np.zeros((height, width), dtype=bool)
        semantic[12:25, 17:31] = True
        supplied_support = dilate_mask(semantic, 6)
        master = np.full((height, width, 3), (20, 130, 45), dtype=np.uint8)
        master[semantic] = (245, 190, 15)
        background = master.copy()
        background[semantic] = (20, 130, 45)
        spec = LayerSpec(
            "clean-object",
            "Clean object",
            "object",
            semantic,
            0.99,
            metadata={"parent_id": None, "children": [], "hierarchy_depth": 0},
        )

        rendered, composite, report = render_layers(
            master,
            background,
            [spec],
            support_masks={"clean-object": supplied_support},
        )

        self.assertEqual(report["matte_policy"], "clean")
        self.assertEqual(report["recomposition_max_abs_error"], 0)
        self.assertTrue(np.array_equal(np.asarray(composite), master))
        alpha_canvas = np.zeros((height, width), dtype=np.uint8)
        item = rendered[0]
        alpha_canvas[item.top : item.bbox[3], item.left : item.bbox[2]] = np.asarray(
            item.alpha, dtype=np.uint8
        )
        self.assertFalse((alpha_canvas[~semantic] > 0).any())
        self.assertTrue((alpha_canvas[semantic] > 0).all())
        self.assertEqual(int(alpha_canvas[18, 23]), 255)
        self.assertTrue((alpha_canvas[semantic] < 255).any())
        matte_qa = matte_quality_report(rendered, (width, height))
        self.assertTrue(matte_qa["ownership_gate_passed"])
        self.assertEqual(matte_qa["alpha_outside_semantic_pixels"], 0)

    def test_fractional_text_alpha_scales_smoothly_without_x4_blocks(self) -> None:
        source_shape = (20, 20)
        mask_u8 = np.zeros(source_shape, dtype=np.uint8)
        cv2.ellipse(mask_u8, (10, 10), (7, 8), 0, 0, 360, 1, -1, cv2.LINE_8)
        cv2.ellipse(mask_u8, (10, 10), (3, 4), 0, 0, 360, 0, -1, cv2.LINE_8)
        mask = mask_u8.astype(bool)
        distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        alpha_matte = np.clip(distance / 2.25, 0.0, 1.0).astype(np.float32)
        alpha_matte[~mask] = 0.0
        final_size = (80, 80)
        expected_alpha = resize_alpha(mask, final_size, alpha_matte=alpha_matte)
        expected_alpha *= resize_refined_envelope(mask, final_size)
        background = np.full((80, 80, 3), (18, 105, 35), dtype=np.uint8)
        foreground = np.array((245, 220, 25), dtype=np.float32)
        master = np.clip(
            foreground * expected_alpha[..., None]
            + background.astype(np.float32) * (1.0 - expected_alpha[..., None])
            + 0.5,
            0,
            255,
        ).astype(np.uint8)
        spec = LayerSpec(
            "smooth-o",
            "Smooth O",
            "text_raster",
            mask,
            0.99,
            metadata={"grouping_role": "ocr_text_line"},
            alpha_matte=alpha_matte,
        )

        rendered, _composite, report = render_layers(master, background, [spec])

        self.assertLessEqual(report["recomposition_max_abs_error"], 1)
        item = rendered[0]
        canvas_alpha = np.zeros((80, 80), dtype=np.uint8)
        canvas_alpha[item.top : item.bbox[3], item.left : item.bbox[2]] = np.asarray(
            item.alpha, dtype=np.uint8
        )
        levels = np.unique(canvas_alpha)
        self.assertGreater(len(levels), 32)
        self.assertTrue(np.logical_and(canvas_alpha > 0, canvas_alpha < 128).any())
        self.assertEqual(int(canvas_alpha[40, 40]), 0)
        nonuniform_x4_blocks = 0
        for y in range(0, 80, 4):
            for x in range(0, 80, 4):
                if len(np.unique(canvas_alpha[y : y + 4, x : x + 4])) > 1:
                    nonuniform_x4_blocks += 1
        self.assertGreater(nonuniform_x4_blocks, 0)
        matte_qa = matte_quality_report(rendered, final_size)
        self.assertTrue(matte_qa["ownership_gate_passed"])
        self.assertEqual(matte_qa["alpha_outside_semantic_pixels"], 0)
        self.assertEqual(matte_qa["layers"][0]["matte_source"], "vitmatte_fractional")


if __name__ == "__main__":
    unittest.main()
