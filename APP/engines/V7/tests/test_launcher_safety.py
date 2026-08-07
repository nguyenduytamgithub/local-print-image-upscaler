from __future__ import annotations

import io
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "APP"))

import upscale_cli  # noqa: E402


def _write_owner_manifest(
    bundle: Path,
    source: Path,
    *,
    batch_root: Path | None = None,
) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "pipeline": "V7_DESIGN_REPAIR",
        "original_source": str(source.resolve()),
    }
    if batch_root is not None:
        payload["batch_root"] = str(batch_root.resolve())
    (bundle / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_complete_engine_bundle(
    bundle: Path,
    review_bytes: bytes,
    *,
    status: str = "REVIEW_REQUIRED",
    final_size: tuple[int, int] = (2, 2),
) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    raster_names = (
        "poster_SOURCE_NORMALIZED.png",
        "poster_CLEAN_BASE_x1.png",
        "poster_REPAIRED_x1.png",
        "poster_QA_OVERLAY.png",
        "poster_BEFORE_AFTER.png",
    )
    for index, name in enumerate(raster_names):
        Image.new("RGB", final_size, (240 - index, 240, 240)).save(bundle / name)
    review_document = json.loads(review_bytes.decode("utf-8"))
    review_document.setdefault("source", "poster_SOURCE_NORMALIZED.png")
    (bundle / "TEXT_REVIEW.json").write_text(
        json.dumps(review_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (bundle / "QA.json").write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )
    assets = [path for path in bundle.iterdir() if path.is_file()]
    manifest = {
        "pipeline": "V7_DESIGN_REPAIR",
        "final_size": list(final_size),
        "scale": 1.0,
        "status": status,
        "review_required": status != "PASS",
        "qa": {"status": status, "report": "QA.json"},
        "assets": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in assets
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class V7OutputSelectionTests(unittest.TestCase):
    def test_target_reuses_proven_owner_and_disambiguates_same_stem(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "OUTPUT"
            first = root / "customer-a" / "poster.png"
            second = root / "customer-b" / "poster.png"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            with patch.object(upscale_cli, "OUTPUT_DIR", output):
                legacy = upscale_cli.select_v7_target_directory(
                    Path(), "poster", "4", first, batch_root=None
                )
                _write_owner_manifest(legacy, first)

                self.assertEqual(
                    upscale_cli.select_v7_target_directory(
                        Path(), "poster", "4", first, batch_root=None
                    ),
                    legacy,
                )
                collision_safe = upscale_cli.select_v7_target_directory(
                    Path(), "poster", "4", second, batch_root=None
                )

            self.assertNotEqual(collision_safe, legacy)
            self.assertIn(upscale_cli.canonical_path_identity(second)[:10], collision_safe.name)

    def test_target_refuses_a_collision_safe_path_owned_by_another_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "OUTPUT"
            first = root / "a" / "poster.png"
            second = root / "b" / "poster.png"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            with patch.object(upscale_cli, "OUTPUT_DIR", output):
                legacy = upscale_cli.select_v7_target_directory(
                    Path(), "poster", "4", first, batch_root=None
                )
                _write_owner_manifest(legacy, first)
                collision_safe = upscale_cli.select_v7_target_directory(
                    Path(), "poster", "4", second, batch_root=None
                )
                _write_owner_manifest(collision_safe, first)

                with self.assertRaises(upscale_cli.UserError):
                    upscale_cli.select_v7_target_directory(
                        Path(), "poster", "4", second, batch_root=None
                    )

            self.assertEqual(
                json.loads((legacy / "manifest.json").read_text(encoding="utf-8"))[
                    "original_source"
                ],
                str(first.resolve()),
            )

    def test_batch_group_reuses_matching_legacy_and_separates_equal_leaf_names(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "OUTPUT"
            first_root = root / "customer-a" / "images"
            second_root = root / "customer-b" / "images"
            first_root.mkdir(parents=True)
            second_root.mkdir(parents=True)
            first_source = first_root / "poster.png"
            first_source.write_bytes(b"first")

            legacy = output / "V7_REPAIR" / "images_x4"
            _write_owner_manifest(
                legacy / "poster_V7_REPAIR_x4",
                first_source,
                batch_root=first_root,
            )
            with patch.object(upscale_cli, "OUTPUT_DIR", output):
                self.assertEqual(
                    upscale_cli.select_v7_batch_group(first_root, "4"),
                    Path("images_x4"),
                )
                second_group = upscale_cli.select_v7_batch_group(second_root, "4")

            self.assertNotEqual(second_group, Path("images_x4"))
            self.assertIn(
                upscale_cli.canonical_path_identity(second_root)[:10],
                second_group.name,
            )

    def test_batch_group_refuses_mismatched_existing_hashed_group(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "OUTPUT"
            first_root = root / "first" / "images"
            second_root = root / "second" / "images"
            first_root.mkdir(parents=True)
            second_root.mkdir(parents=True)
            first_source = first_root / "poster.png"
            first_source.write_bytes(b"first")

            legacy = output / "V7_REPAIR" / "images_x4"
            _write_owner_manifest(
                legacy / "legacy",
                first_source,
                batch_root=first_root,
            )
            safe_name = (
                "images_x4_" + upscale_cli.canonical_path_identity(second_root)[:10]
            )
            _write_owner_manifest(
                output / "V7_REPAIR" / safe_name / "wrong-owner",
                first_source,
                batch_root=first_root,
            )

            with patch.object(upscale_cli, "OUTPUT_DIR", output):
                with self.assertRaises(upscale_cli.UserError):
                    upscale_cli.select_v7_batch_group(second_root, "4")


class DirectoryPublishTests(unittest.TestCase):
    def test_second_rename_failure_restores_previous_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            source = parent / ".bundle.new-test"
            target = parent / "bundle"
            source.mkdir()
            target.mkdir()
            (source / "value.txt").write_text("new", encoding="utf-8")
            (target / "value.txt").write_text("old", encoding="utf-8")
            source_resolved = source.resolve()
            target_resolved = target.resolve()
            real_replace = upscale_cli.os.replace

            def fail_publish(from_path: object, to_path: object) -> None:
                if Path(from_path) == source_resolved and Path(to_path) == target_resolved:
                    raise OSError("simulated publish failure")
                real_replace(from_path, to_path)

            with patch.object(upscale_cli.os, "replace", side_effect=fail_publish):
                with self.assertRaises(OSError):
                    upscale_cli.atomic_install_directory(source, target)

            self.assertEqual((target / "value.txt").read_text(encoding="utf-8"), "old")
            self.assertEqual((source / "value.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse(upscale_cli._directory_publish_journal(target).exists())
            self.assertEqual(list(parent.glob(".bundle.old-*")), [])

    def test_recovery_restores_verified_backup_when_target_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            target = parent / "bundle"
            backup = parent / f".bundle.old-{'a' * 32}"
            backup.mkdir()
            (backup / "value.txt").write_text("old", encoding="utf-8")
            journal = upscale_cli._directory_publish_journal(target)
            journal.write_text(
                json.dumps(
                    {
                        "schema": upscale_cli.DIRECTORY_PUBLISH_SCHEMA,
                        "target": str(target.resolve()),
                        "source": str((parent / ".bundle.new-test").resolve()),
                        "backup": str(backup.resolve()),
                        "phase": "old_moved",
                    }
                ),
                encoding="utf-8",
            )

            upscale_cli.recover_interrupted_directory_publish(target)

            self.assertEqual((target / "value.txt").read_text(encoding="utf-8"), "old")
            self.assertFalse(backup.exists())
            self.assertFalse(journal.exists())

    def test_backup_cleanup_failure_keeps_new_bundle_and_recovers_next_run(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            source = parent / ".bundle.new-test"
            target = parent / "bundle"
            source.mkdir()
            target.mkdir()
            (source / "value.txt").write_text("new", encoding="utf-8")
            (target / "value.txt").write_text("old", encoding="utf-8")
            real_rmtree = upscale_cli.shutil.rmtree

            def fail_backup_cleanup(path: object, *args: object, **kwargs: object) -> None:
                if Path(path).name.startswith(".bundle.old-"):
                    raise OSError("simulated locked backup")
                real_rmtree(path, *args, **kwargs)

            stderr = io.StringIO()
            with (
                patch.object(upscale_cli.shutil, "rmtree", side_effect=fail_backup_cleanup),
                redirect_stderr(stderr),
            ):
                upscale_cli.atomic_install_directory(source, target)

            backups = list(parent.glob(".bundle.old-*"))
            journal = upscale_cli._directory_publish_journal(target)
            self.assertEqual((target / "value.txt").read_text(encoding="utf-8"), "new")
            self.assertEqual(len(backups), 1)
            self.assertTrue(journal.exists())
            self.assertIn("backup", stderr.getvalue().lower())

            upscale_cli.recover_interrupted_directory_publish(target)
            self.assertEqual((target / "value.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse(backups[0].exists())
            self.assertFalse(journal.exists())


class V7BatchReviewSnapshotTests(unittest.TestCase):
    def test_batch_rerun_snapshots_and_republishes_existing_review(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "OUTPUT"
            fake_app = root / "APP"
            work = fake_app / "work"
            work.mkdir(parents=True)
            batch_root = root / "source" / "images"
            batch_root.mkdir(parents=True)
            source = batch_root / "poster.png"
            Image.new("RGB", (2, 2), "white").save(source)
            python_v7 = root / "python-v7.exe"
            engine_v7 = root / "design-repair-v7.py"
            python_v7.write_bytes(b"runtime")
            engine_v7.write_text("engine", encoding="utf-8")

            relative_dir = Path("images_x1")
            target = (
                output
                / "V7_REPAIR"
                / relative_dir
                / "poster_V7_REPAIR_x1"
            )
            _write_owner_manifest(target, source, batch_root=batch_root)
            prior_technical = target / "_KY_THUAT"
            prior_technical.mkdir()
            (target / "manifest.json").replace(prior_technical / "manifest.json")
            prior_review = {
                "schema": "local-print-image-upscaler/v7-text-review/1",
                "source_sha256": "a" * 64,
                "regions": [
                    {
                        "region_id": "text_0001",
                        "action": "replace",
                        "approved_text": "NỘI DUNG ĐÃ DUYỆT",
                    }
                ],
            }
            prior_bytes = json.dumps(prior_review, ensure_ascii=False, indent=2).encode(
                "utf-8"
            )
            (prior_technical / "TEXT_REVIEW.json").write_bytes(prior_bytes)
            captured: dict[str, object] = {}

            def fake_engine_run(
                command: list[str],
                *,
                check: bool,
                cwd: Path,
            ) -> subprocess.CompletedProcess[str]:
                self.assertTrue(check)
                review_index = command.index("--review-file")
                snapshot = Path(command[review_index + 1])
                captured["snapshot"] = snapshot
                captured["bytes"] = snapshot.read_bytes()
                staging = Path(command[5])
                _write_complete_engine_bundle(staging, snapshot.read_bytes())
                return subprocess.CompletedProcess(command, 0)

            resource_plan = {
                "source_megapixels": 0.0,
                "output_megapixels": 0.0,
                "estimated_peak_ram_gib": 0.0,
                "estimated_working_disk_gib": 0.0,
            }
            with (
                patch.object(upscale_cli, "OUTPUT_DIR", output),
                patch.object(upscale_cli, "APP_DIR", fake_app),
                patch.object(upscale_cli, "WORK_DIR", work),
                patch.object(upscale_cli, "ROOT_DIR", root),
                patch.object(upscale_cli, "PYTHON_V7", python_v7),
                patch.object(upscale_cli, "V7_ENGINE", engine_v7),
                patch.object(upscale_cli, "V7_TESSDATA", root / "tessdata"),
                patch.object(
                    upscale_cli,
                    "validate_v7_resource_plan",
                    return_value=resource_plan,
                ),
                patch.object(upscale_cli.subprocess, "run", side_effect=fake_engine_run),
            ):
                result = upscale_cli.run_v7_job(
                    source,
                    1.0,
                    False,
                    {"review": "defer", "ocr_passes": 1, "language_model": False},
                    output_subdir=relative_dir,
                    output_stem="poster",
                    batch_root=batch_root,
                )

            self.assertEqual(result, target)
            self.assertEqual(captured["bytes"], prior_bytes)
            snapshot = captured["snapshot"]
            self.assertIsInstance(snapshot, Path)
            self.assertEqual(snapshot.name, "previous_TEXT_REVIEW.json")
            published_review = target / "_KY_THUAT" / "TEXT_REVIEW.json"
            self.assertNotEqual(snapshot, published_review)
            published = json.loads(published_review.read_text(encoding="utf-8"))
            self.assertEqual(published["regions"], prior_review["regions"])
            self.assertEqual(published["source"], "SOURCE.png")
            self.assertTrue((target / "01_XEM_TRUOC_CAN_DUYET.png").is_file())
            self.assertFalse((target / "manifest.json").exists())
            self.assertEqual(
                {entry.name for entry in target.iterdir()},
                {"01_XEM_TRUOC_CAN_DUYET.png", "02_SO_SANH.png", "_KY_THUAT"},
            )
            published_manifest = json.loads(
                (target / "_KY_THUAT" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(published_manifest["original_source"], str(source))
            self.assertEqual(published_manifest["scale"], 1.0)
            self.assertEqual(published_manifest["rerun_argv"][2], str(batch_root))
            self.assertNotIn("--review-file", published_manifest["rerun_argv"])


class V7IntegratedPublishRollbackTests(unittest.TestCase):
    def test_friendly_staging_failure_restores_previous_user_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "OUTPUT"
            fake_app = root / "APP"
            work = fake_app / "work"
            work.mkdir(parents=True)
            source = root / "source" / "poster.png"
            source.parent.mkdir()
            Image.new("RGB", (2, 2), "white").save(source)
            python_v7 = root / "python-v7.exe"
            engine_v7 = root / "design-repair-v7.py"
            python_v7.write_bytes(b"runtime")
            engine_v7.write_text("engine", encoding="utf-8")
            target = output / "V7_REPAIR" / "poster_V7_REPAIR_x1"
            _write_owner_manifest(target, source)
            old_marker = target / "KEEP_OLD.txt"
            old_marker.write_text("old bundle", encoding="utf-8")

            def fake_engine_run(
                command: list[str],
                *,
                check: bool,
                cwd: Path,
            ) -> subprocess.CompletedProcess[str]:
                self.assertTrue(check)
                _write_complete_engine_bundle(
                    Path(command[5]),
                    json.dumps({"regions": []}).encode("utf-8"),
                )
                return subprocess.CompletedProcess(command, 0)

            resource_plan = {
                "source_megapixels": 0.0,
                "output_megapixels": 0.0,
                "estimated_peak_ram_gib": 0.0,
                "estimated_working_disk_gib": 0.0,
            }
            real_replace = upscale_cli.os.replace
            saw_friendly_stage = False

            def fail_final_publish(from_path: object, to_path: object) -> None:
                nonlocal saw_friendly_stage
                source_path = Path(from_path)
                destination = Path(to_path)
                if destination == target.resolve() and source_path.name.startswith(
                    f".{target.name}.new-"
                ):
                    saw_friendly_stage = (source_path / "01_XEM_TRUOC_CAN_DUYET.png").is_file()
                    self.assertFalse((source_path / "manifest.json").exists())
                    raise OSError("simulated final publish failure")
                real_replace(from_path, to_path)

            with (
                patch.object(upscale_cli, "OUTPUT_DIR", output),
                patch.object(upscale_cli, "APP_DIR", fake_app),
                patch.object(upscale_cli, "WORK_DIR", work),
                patch.object(upscale_cli, "ROOT_DIR", root),
                patch.object(upscale_cli, "PYTHON_V7", python_v7),
                patch.object(upscale_cli, "V7_ENGINE", engine_v7),
                patch.object(upscale_cli, "V7_TESSDATA", root / "tessdata"),
                patch.object(
                    upscale_cli,
                    "validate_v7_resource_plan",
                    return_value=resource_plan,
                ),
                patch.object(upscale_cli.subprocess, "run", side_effect=fake_engine_run),
                patch.object(upscale_cli.os, "replace", side_effect=fail_final_publish),
            ):
                with self.assertRaises(OSError):
                    upscale_cli.run_v7_job(
                        source,
                        1.0,
                        False,
                        {"review": "defer", "ocr_passes": 1, "language_model": False},
                    )

            self.assertTrue(saw_friendly_stage)
            self.assertEqual(old_marker.read_text(encoding="utf-8"), "old bundle")
            self.assertTrue((target / "manifest.json").is_file())
            self.assertFalse(upscale_cli._directory_publish_journal(target).exists())
            self.assertEqual(list(target.parent.glob(f".{target.name}.old-*")), [])
            self.assertEqual(list(target.parent.glob(f".{target.name}.new-*")), [])


if __name__ == "__main__":
    unittest.main()
