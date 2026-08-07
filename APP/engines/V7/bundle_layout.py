"""Transactional, user-facing layout for a completed V7 engine bundle.

The V7 engine intentionally writes descriptive technical filenames.  This
module is a separate publication step: it gives the few files a user normally
opens stable Vietnamese names and moves diagnostics below ``_KY_THUAT``.

No placeholder is ever created.  A bundle is first validated and copied into a
sibling staging directory, then the old and new directories are exchanged with
``os.replace``.  If publication fails after the old directory was moved, the
old directory is put back before the error is returned.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping


TECHNICAL_DIR_NAME = "_KY_THUAT"
PASS_RESULT_NAME = "01_KET_QUA_DA_DAT.png"
REVIEW_RESULT_NAME = "01_XEM_TRUOC_CAN_DUYET.png"
COMPARISON_NAME = "02_SO_SANH.png"
EDITABLE_NAME = "03_CHINH_SUA.svg"
VALID_STATUSES = frozenset({"PASS", "REVIEW_REQUIRED", "FAILED_QA"})


class BundleLayoutError(RuntimeError):
    """Base error for an unsafe or incomplete V7 bundle layout."""


class BundleCollisionError(BundleLayoutError):
    """Raised when more than one real file claims the same logical role."""


class BundlePathError(BundleLayoutError):
    """Raised for path traversal, links, or paths outside the bundle."""


class BundleTransactionError(BundleLayoutError):
    """Raised when publication fails (with rollback attempted)."""


@dataclass(frozen=True, slots=True)
class BundlePaths:
    """Resolved logical files in either the legacy or friendly layout."""

    root: Path
    status: str | None
    layout: str
    result: Path | None
    comparison: Path | None
    editable_svg: Path | None
    technical_dir: Path
    source: Path | None
    clean: Path | None
    overlay: Path | None
    review: Path | None
    qa: Path | None
    manifest: Path | None


def _normalise_status(status: str | None, *, required: bool) -> str | None:
    if status is None:
        if required:
            raise BundleLayoutError("V7 bundle status is required.")
        return None
    value = str(status).strip().upper()
    if value not in VALID_STATUSES:
        allowed = ", ".join(sorted(VALID_STATUSES))
        raise BundleLayoutError(f"Unsupported V7 status {status!r}; expected {allowed}.")
    return value


def _root_directory(bundle_dir: Path | str) -> Path:
    root = Path(bundle_dir)
    if root.is_symlink() or _is_junction(root):
        raise BundlePathError(f"Bundle root must not be a link or junction: {root}")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise BundlePathError(f"Bundle directory does not exist: {root}") from exc
    if not resolved.is_dir():
        raise BundlePathError(f"Bundle path is not a directory: {resolved}")
    if resolved.parent == resolved:
        raise BundlePathError("A filesystem root can never be used as a V7 bundle.")
    return resolved


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    if checker is not None and checker():
        return True
    # V7 currently supports Python 3.11 on Windows, before Path.is_junction().
    # Junctions and other reparse points can otherwise make a path resolve
    # outside its apparent bundle even when followlinks=False.
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return False
    return bool(attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))


def _existing_files(paths: Iterable[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        if path.is_symlink() or _is_junction(path):
            raise BundlePathError(f"Bundle file must not be a link or junction: {path}")
        if not path.is_file():
            continue
        key = os.path.normcase(str(path.resolve(strict=True)))
        unique.setdefault(key, path.resolve(strict=True))
    return sorted(unique.values(), key=lambda item: item.as_posix().casefold())


def _one_file(role: str, paths: Iterable[Path]) -> Path | None:
    candidates = _existing_files(paths)
    if len(candidates) > 1:
        listed = ", ".join(path.name for path in candidates)
        raise BundleCollisionError(f"Multiple files claim V7 role {role}: {listed}")
    return candidates[0] if candidates else None


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleLayoutError(f"Cannot read valid {role} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BundleLayoutError(f"{role} JSON root must be an object: {path}")
    return value


def _status_from_manifest(path: Path | None) -> str | None:
    if path is None:
        return None
    document = _read_json_object(path, "manifest")
    raw = document.get("status")
    if raw is None and isinstance(document.get("qa"), dict):
        raw = document["qa"].get("status")
    return _normalise_status(raw, required=False) if raw is not None else None


def resolve_bundle_paths(
    bundle_dir: Path | str,
    status: str | None = None,
) -> BundlePaths:
    """Resolve V7 files without caring whether the bundle is old or friendly.

    Ambiguous roles are rejected rather than selected by filename order.  This
    makes the resolver safe for a future launcher to use directly.
    """

    root = _root_directory(bundle_dir)
    technical = root / TECHNICAL_DIR_NAME
    manifest = _one_file(
        "manifest",
        (root / "manifest.json", technical / "manifest.json"),
    )
    resolved_status = _normalise_status(status, required=False)
    if resolved_status is None:
        resolved_status = _status_from_manifest(manifest)

    result = _one_file(
        "result",
        (
            root / PASS_RESULT_NAME,
            root / REVIEW_RESULT_NAME,
            *root.glob("*_REPAIRED_x*.png"),
        ),
    )
    comparison = _one_file(
        "comparison",
        (root / COMPARISON_NAME, *root.glob("*_BEFORE_AFTER.png")),
    )
    editable = _one_file(
        "editable SVG",
        (root / EDITABLE_NAME, *root.glob("*_TEXT_EDITABLE.svg")),
    )
    source = _one_file(
        "source",
        (technical / "SOURCE.png", *root.glob("*_SOURCE_NORMALIZED.png")),
    )
    clean = _one_file(
        "clean base",
        (technical / "CLEAN.png", *root.glob("*_CLEAN_BASE_x*.png")),
    )
    overlay = _one_file(
        "QA overlay",
        (technical / "OVERLAY.png", *root.glob("*_QA_OVERLAY.png")),
    )
    review = _one_file(
        "text review",
        (root / "TEXT_REVIEW.json", technical / "TEXT_REVIEW.json"),
    )
    qa = _one_file("QA report", (root / "QA.json", technical / "QA.json"))

    found_new = any(
        path is not None and (path.parent == technical or path.name in {
            PASS_RESULT_NAME,
            REVIEW_RESULT_NAME,
            COMPARISON_NAME,
            EDITABLE_NAME,
        })
        for path in (result, comparison, editable, source, clean, overlay, review, qa, manifest)
    )
    found_legacy = any(
        path is not None
        and path.parent == root
        and path.name not in {
            PASS_RESULT_NAME,
            REVIEW_RESULT_NAME,
            COMPARISON_NAME,
            EDITABLE_NAME,
        }
        for path in (result, comparison, editable, source, clean, overlay, review, qa, manifest)
    )
    layout = "mixed" if found_new and found_legacy else "friendly" if found_new else "legacy"
    return BundlePaths(
        root=root,
        status=resolved_status,
        layout=layout,
        result=result,
        comparison=comparison,
        editable_svg=editable,
        technical_dir=technical,
        source=source,
        clean=clean,
        overlay=overlay,
        review=review,
        qa=qa,
        manifest=manifest,
    )


def resolve_bundle_manifest(bundle_dir: Path | str) -> Path | None:
    """Return the legacy or friendly manifest path, if present."""

    return resolve_bundle_paths(bundle_dir).manifest


def resolve_bundle_review(bundle_dir: Path | str) -> Path | None:
    """Return the legacy or friendly text-review path, if present."""

    return resolve_bundle_paths(bundle_dir).review


def resolve_bundle_result(
    bundle_dir: Path | str,
    status: str | None = None,
) -> Path | None:
    """Return the legacy or friendly raster result path, if present."""

    return resolve_bundle_paths(bundle_dir, status=status).result


def _enumerate_bundle_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(directory_names):
            child = parent / name
            if child.is_symlink() or _is_junction(child):
                raise BundlePathError(f"Bundle contains a linked directory: {child}")
        for name in file_names:
            child = parent / name
            if child.is_symlink() or _is_junction(child):
                raise BundlePathError(f"Bundle contains a linked file: {child}")
            try:
                resolved = child.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise BundlePathError(f"Bundle file escapes its root: {child}") from exc
            if not resolved.is_file():
                raise BundlePathError(f"Bundle entry is not a regular file: {child}")
            files.append(resolved)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix().casefold())


def _safe_manifest_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise BundlePathError("Manifest asset path must be a non-empty string.")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value.replace("\\", "/"))
    if windows.is_absolute() or windows.drive or posix.is_absolute():
        raise BundlePathError(f"Manifest asset path must be bundle-relative: {value!r}")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise BundlePathError(f"Manifest asset path is not canonical: {value!r}")
    candidate = root.joinpath(*posix.parts)
    if candidate.is_symlink() or _is_junction(candidate):
        raise BundlePathError(f"Manifest asset must not be a link: {value!r}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise BundlePathError(f"Manifest asset is missing or escapes the bundle: {value!r}") from exc
    if not resolved.is_file():
        raise BundlePathError(f"Manifest asset is not a regular file: {value!r}")
    return resolved


def _manifest_asset_sources(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[list[tuple[dict[str, Any], Path]], set[Path]]:
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list):
        raise BundleLayoutError("V7 manifest must contain an assets list.")
    records: list[tuple[dict[str, Any], Path]] = []
    seen: set[Path] = set()
    for index, raw in enumerate(raw_assets):
        if not isinstance(raw, dict):
            raise BundleLayoutError(f"Manifest asset #{index + 1} must be an object.")
        source = _safe_manifest_path(root, raw.get("path"))
        if source in seen:
            raise BundleCollisionError(f"Manifest lists an asset more than once: {raw.get('path')}")
        seen.add(source)
        records.append((dict(raw), source))
    return records, seen


def _require_complete(paths: BundlePaths) -> None:
    required = {
        "result": paths.result,
        "comparison": paths.comparison,
        "source": paths.source,
        "clean base": paths.clean,
        "QA overlay": paths.overlay,
        "TEXT_REVIEW.json": paths.review,
        "QA.json": paths.qa,
        "manifest.json": paths.manifest,
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise BundleLayoutError("Incomplete V7 bundle; missing: " + ", ".join(missing))
    empty = [name for name, path in required.items() if path is not None and path.stat().st_size == 0]
    if empty:
        raise BundleLayoutError("V7 bundle contains empty required files: " + ", ".join(empty))
    if paths.editable_svg is not None and paths.editable_svg.stat().st_size == 0:
        raise BundleLayoutError("V7 editable SVG exists but is empty.")


def _recorded_status(value: object, location: str) -> str | None:
    if value is None:
        return None
    try:
        return _normalise_status(str(value), required=True)
    except BundleLayoutError as exc:
        raise BundleLayoutError(f"Invalid status recorded in {location}: {value!r}") from exc


def _validate_status_documents(
    status: str,
    manifest: Mapping[str, Any],
    qa_document: Mapping[str, Any],
) -> None:
    recorded: list[tuple[str, str]] = []
    top = _recorded_status(manifest.get("status"), "manifest.status")
    if top is not None:
        recorded.append(("manifest.status", top))
    qa_metadata = manifest.get("qa")
    if isinstance(qa_metadata, dict):
        qa_status = _recorded_status(qa_metadata.get("status"), "manifest.qa.status")
        if qa_status is not None:
            recorded.append(("manifest.qa.status", qa_status))
    report_status = _recorded_status(qa_document.get("status"), "QA.json status")
    if report_status is not None:
        recorded.append(("QA.json status", report_status))
    for location, value in recorded:
        if value != status:
            raise BundleLayoutError(
                f"Requested status {status} conflicts with {location}={value}; bundle was not renamed."
            )
    review_required = manifest.get("review_required")
    if isinstance(review_required, bool) and review_required != (status != "PASS"):
        raise BundleLayoutError(
            "Requested status conflicts with manifest.review_required; bundle was not renamed."
        )


def _validate_bundle_identity(manifest: Mapping[str, Any]) -> None:
    if manifest.get("pipeline") != "V7_DESIGN_REPAIR":
        raise BundleLayoutError(
            "Directory is not identified as a V7_DESIGN_REPAIR engine bundle."
        )


def _desired_result_name(status: str) -> str:
    return PASS_RESULT_NAME if status == "PASS" else REVIEW_RESULT_NAME


def _destination_plan(
    root: Path,
    paths: BundlePaths,
    files: list[Path],
    status: str,
) -> dict[Path, PurePosixPath]:
    assert paths.result is not None
    assert paths.comparison is not None
    assert paths.source is not None
    assert paths.clean is not None
    assert paths.overlay is not None
    assert paths.review is not None
    assert paths.qa is not None
    assert paths.manifest is not None

    plan: dict[Path, PurePosixPath] = {
        paths.result: PurePosixPath(_desired_result_name(status)),
        paths.comparison: PurePosixPath(COMPARISON_NAME),
        paths.source: PurePosixPath(TECHNICAL_DIR_NAME, "SOURCE.png"),
        paths.clean: PurePosixPath(TECHNICAL_DIR_NAME, "CLEAN.png"),
        paths.overlay: PurePosixPath(TECHNICAL_DIR_NAME, "OVERLAY.png"),
        paths.review: PurePosixPath(TECHNICAL_DIR_NAME, "TEXT_REVIEW.json"),
        paths.qa: PurePosixPath(TECHNICAL_DIR_NAME, "QA.json"),
        paths.manifest: PurePosixPath(TECHNICAL_DIR_NAME, "manifest.json"),
    }
    if paths.editable_svg is not None:
        plan[paths.editable_svg] = PurePosixPath(EDITABLE_NAME)

    for source in files:
        if source in plan:
            continue
        relative = PurePosixPath(source.relative_to(root).as_posix())
        if relative.parts and relative.parts[0].casefold() == TECHNICAL_DIR_NAME.casefold():
            destination = relative
        else:
            destination = PurePosixPath(TECHNICAL_DIR_NAME, "_KHAC", *relative.parts)
        plan[source] = destination

    destinations: dict[str, tuple[Path, PurePosixPath]] = {}
    for source, destination in plan.items():
        key = destination.as_posix().casefold()
        if key in destinations and destinations[key][0] != source:
            other = destinations[key][0]
            raise BundleCollisionError(
                f"Files {other} and {source} would both become {destination.as_posix()}."
            )
        destinations[key] = (source, destination)

    # A destination file may not also be the parent directory of another file.
    ordered = sorted(destinations)
    for index, key in enumerate(ordered):
        prefix = key + "/"
        if any(other.startswith(prefix) for other in ordered[index + 1 :]):
            raise BundleCollisionError(f"Destination file/directory collision at {key}.")
    return plan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_or_link(source: Path, destination: Path, *, force_copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not force_copy:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _stage_path(stage: Path, destination: PurePosixPath) -> Path:
    return stage.joinpath(*destination.parts)


def _write_review_for_layout(review_path: Path) -> dict[str, Any]:
    review = _read_json_object(review_path, "TEXT_REVIEW")
    review["source"] = "SOURCE.png"
    payload = json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    review_path.write_text(payload, encoding="utf-8")
    return review


def _write_manifest_for_layout(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    stage: Path,
    plan: Mapping[Path, PurePosixPath],
    original_records: list[tuple[dict[str, Any], Path]],
    status: str,
    has_editable: bool,
) -> dict[str, Any]:
    document = dict(manifest)
    original_by_source = {source: record for record, source in original_records}
    assets: list[dict[str, Any]] = []
    manifest_source = next(
        source
        for source, destination in plan.items()
        if _stage_path(stage, destination) == manifest_path
    )
    for source, destination in sorted(plan.items(), key=lambda item: item[1].as_posix().casefold()):
        if source == manifest_source:
            continue
        staged_asset = _stage_path(stage, destination)
        record = dict(original_by_source.get(source, {}))
        record.update(
            {
                "path": destination.as_posix(),
                "bytes": staged_asset.stat().st_size,
                "sha256": _sha256(staged_asset),
            }
        )
        assets.append(record)
    document["assets"] = assets
    document["path_policy"] = "bundle-relative assets only"
    qa_metadata = document.get("qa")
    if isinstance(qa_metadata, dict):
        qa_metadata = dict(qa_metadata)
    else:
        qa_metadata = {}
    qa_metadata["status"] = status
    qa_metadata["report"] = f"{TECHNICAL_DIR_NAME}/QA.json"
    document["qa"] = qa_metadata
    document["status"] = status
    document["review_required"] = status != "PASS"
    layout_metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "result": _desired_result_name(status),
        "comparison": COMPARISON_NAME,
        "technical_dir": TECHNICAL_DIR_NAME,
    }
    if has_editable:
        layout_metadata["editable"] = EDITABLE_NAME
    document["bundle_layout"] = layout_metadata
    payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    manifest_path.write_text(payload, encoding="utf-8")
    return document


def _is_canonical(
    *,
    root: Path,
    paths: BundlePaths,
    files: list[Path],
    plan: Mapping[Path, PurePosixPath],
    manifest: Mapping[str, Any],
    review: Mapping[str, Any],
    original_records: list[tuple[dict[str, Any], Path]],
    status: str,
) -> bool:
    for source, destination in plan.items():
        if source.relative_to(root).as_posix() != destination.as_posix():
            return False
    allowed_root_files = {_desired_result_name(status), COMPARISON_NAME}
    if paths.editable_svg is not None:
        allowed_root_files.add(EDITABLE_NAME)
    for entry in root.iterdir():
        if entry.is_dir():
            if entry.name != TECHNICAL_DIR_NAME:
                return False
        elif entry.name not in allowed_root_files:
            return False
    if review.get("source") != "SOURCE.png":
        return False
    layout = manifest.get("bundle_layout")
    if not isinstance(layout, dict):
        return False
    expected_layout: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "result": _desired_result_name(status),
        "comparison": COMPARISON_NAME,
        "technical_dir": TECHNICAL_DIR_NAME,
    }
    if paths.editable_svg is not None:
        expected_layout["editable"] = EDITABLE_NAME
    if layout != expected_layout:
        return False
    qa_metadata = manifest.get("qa")
    if not isinstance(qa_metadata, dict) or qa_metadata.get("report") != f"{TECHNICAL_DIR_NAME}/QA.json":
        return False

    manifest_path = paths.manifest
    assert manifest_path is not None
    expected_assets = {
        destination.as_posix(): source
        for source, destination in plan.items()
        if source != manifest_path
    }
    if len(original_records) != len(expected_assets):
        return False
    for record, source in original_records:
        destination = plan.get(source)
        if destination is None or record.get("path") != destination.as_posix():
            return False
        if record.get("bytes") != source.stat().st_size or record.get("sha256") != _sha256(source):
            return False
    return len(files) == len(plan)


def _remove_tree(path: Path) -> None:
    if path.exists():
        def make_writable_and_retry(
            operation: Any,
            failed_path: str,
            _error: Any,
        ) -> None:
            os.chmod(failed_path, stat.S_IWRITE)
            operation(failed_path)

        shutil.rmtree(path, onerror=make_writable_and_retry)


def _publish_directory(root: Path, stage: Path) -> None:
    backup = root.parent / f".{root.name}.layout-old-{uuid.uuid4().hex}"
    moved_old = False
    try:
        os.replace(root, backup)
        moved_old = True
        try:
            os.replace(stage, root)
        except BaseException as publish_error:
            try:
                os.replace(backup, root)
                moved_old = False
            except BaseException as rollback_error:
                raise BundleTransactionError(
                    f"Publishing failed and automatic rollback also failed; original bundle remains at {backup}."
                ) from rollback_error
            raise BundleTransactionError(
                "Publishing the friendly layout failed; the original bundle was restored."
            ) from publish_error
    except BundleTransactionError:
        raise
    except BaseException as exc:
        raise BundleTransactionError("Could not begin atomic bundle publication.") from exc
    finally:
        if stage.exists():
            _remove_tree(stage)
    if moved_old and backup.exists():
        try:
            _remove_tree(backup)
        except OSError as exc:
            raise BundleTransactionError(
                f"Friendly layout was published, but old backup cleanup failed: {backup}"
            ) from exc


def arrange_bundle(bundle_dir: Path | str, status: str) -> BundlePaths:
    """Publish one complete engine bundle in the stable user-facing layout.

    ``status`` controls only the friendly result filename and must agree with
    the statuses already recorded by the engine.  All required files must be
    real and non-empty; the editable SVG remains optional.
    """

    wanted_status = _normalise_status(status, required=True)
    assert wanted_status is not None
    paths = resolve_bundle_paths(bundle_dir, status=wanted_status)
    _require_complete(paths)
    root = paths.root
    files = _enumerate_bundle_files(root)
    assert paths.manifest is not None
    assert paths.review is not None
    assert paths.qa is not None
    manifest = _read_json_object(paths.manifest, "manifest")
    review = _read_json_object(paths.review, "TEXT_REVIEW")
    qa_document = _read_json_object(paths.qa, "QA")
    _validate_bundle_identity(manifest)
    _validate_status_documents(wanted_status, manifest, qa_document)
    original_records, _ = _manifest_asset_sources(root, manifest)
    plan = _destination_plan(root, paths, files, wanted_status)
    if _is_canonical(
        root=root,
        paths=paths,
        files=files,
        plan=plan,
        manifest=manifest,
        review=review,
        original_records=original_records,
        status=wanted_status,
    ):
        return paths

    stage = root.parent / f".{root.name}.layout-new-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=False, exist_ok=False)
        for source, destination in plan.items():
            force_copy = source in {paths.review, paths.manifest}
            _copy_or_link(source, _stage_path(stage, destination), force_copy=force_copy)
        staged_review = _stage_path(stage, plan[paths.review])
        _write_review_for_layout(staged_review)
        staged_manifest = _stage_path(stage, plan[paths.manifest])
        _write_manifest_for_layout(
            staged_manifest,
            manifest,
            stage=stage,
            plan=plan,
            original_records=original_records,
            status=wanted_status,
            has_editable=paths.editable_svg is not None,
        )
        # Re-open the two rewritten control files before any directory rename.
        _read_json_object(staged_review, "staged TEXT_REVIEW")
        _read_json_object(staged_manifest, "staged manifest")
        _publish_directory(root, stage)
    except BundleLayoutError:
        if stage.exists():
            _remove_tree(stage)
        raise
    except BaseException as exc:
        if stage.exists():
            _remove_tree(stage)
        raise BundleTransactionError(
            "Could not stage the friendly layout; the original bundle was not changed."
        ) from exc
    return resolve_bundle_paths(root, status=wanted_status)


# A descriptive alias for callers that think of the operation as publication.
publish_bundle_layout = arrange_bundle


__all__ = [
    "BundleCollisionError",
    "BundleLayoutError",
    "BundlePathError",
    "BundlePaths",
    "BundleTransactionError",
    "COMPARISON_NAME",
    "EDITABLE_NAME",
    "PASS_RESULT_NAME",
    "REVIEW_RESULT_NAME",
    "TECHNICAL_DIR_NAME",
    "VALID_STATUSES",
    "arrange_bundle",
    "publish_bundle_layout",
    "resolve_bundle_manifest",
    "resolve_bundle_paths",
    "resolve_bundle_result",
    "resolve_bundle_review",
]
