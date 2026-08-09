from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from v5lib.formats import (
    RenderedLayer,
    _minimum_pillow_alpha_u8,
    _solve_pillow_foreground_u8,
    export_contact_sheet,
    export_ora,
    export_png_assets,
    export_psd,
    save_color_png,
)
from v5lib.geometry import safe_name
from v5lib.model import LayerSpec

from .ownership import ownership_priority
from .schema import AlphaCrop, DocumentGraph, ElementNode


@dataclass(slots=True)
class RenderedDocument:
    background: Image.Image
    rendered: list[RenderedLayer]
    composite: Image.Image
    report: dict[str, Any]
    user_groups: list[dict[str, object]]


@dataclass(slots=True)
class ExportedDocument:
    rendered_document: RenderedDocument
    asset_records: list[dict[str, object]]
    format_reports: dict[str, dict[str, object]]
    paths: dict[str, Path]


def _resize_u8(channel: np.ndarray, size: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    if channel.shape[::-1] == size:
        return np.array(channel, dtype=np.uint8, copy=True)
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return np.array(Image.fromarray(channel, "L").resize(size, resampling), dtype=np.uint8, copy=True)


def _scaled_box(crop: AlphaCrop, scale: float) -> tuple[int, int, int, int]:
    left = int(round(crop.left * scale))
    top = int(round(crop.top * scale))
    right = int(round((crop.left + crop.width) * scale))
    bottom = int(round((crop.top + crop.height) * scale))
    right = max(left + 1, right)
    bottom = max(top + 1, bottom)
    return left, top, right, bottom


def _resize_straight_rgba(rgba: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    value = np.asarray(rgba, dtype=np.uint8)
    if value.shape[1::-1] == size:
        return value.copy()
    alpha = value[:, :, 3].astype(np.float32) / 255.0
    premultiplied = value[:, :, :3].astype(np.float32) * alpha[:, :, None]
    resized_alpha = _resize_u8(value[:, :, 3], size).astype(np.float32) / 255.0
    resized_premultiplied = np.empty((size[1], size[0], 3), dtype=np.float32)
    for channel in range(3):
        image = Image.fromarray(premultiplied[:, :, channel], "F")
        resized_premultiplied[:, :, channel] = np.asarray(
            image.resize(size, Image.Resampling.LANCZOS), dtype=np.float32
        )
    rgb = np.divide(
        resized_premultiplied,
        np.maximum(resized_alpha[:, :, None], 1.0 / 255.0),
        out=np.zeros_like(resized_premultiplied),
        where=resized_alpha[:, :, None] > 0,
    )
    return np.dstack(
        [
            np.clip(np.rint(rgb), 0, 255).astype(np.uint8),
            np.clip(np.rint(resized_alpha * 255.0), 0, 255).astype(np.uint8),
        ]
    )


def _ordered_nodes(graph: DocumentGraph) -> list[ElementNode]:
    """Return one container-compatible, complete bottom-to-top traversal."""

    active = [node for node in graph.nodes if node.review_status != "rejected"]
    by_id = {node.element_id: node for node in active}
    children: dict[str, list[ElementNode]] = {}
    roots: list[ElementNode] = []
    for node in active:
        if node.parent_id and node.parent_id in by_id:
            children.setdefault(node.parent_id, []).append(node)
        else:
            roots.append(node)
    roots.sort(key=ownership_priority)
    for values in children.values():
        values.sort(key=ownership_priority)
    result: list[ElementNode] = []
    visiting: set[str] = set()

    def emit(node: ElementNode) -> None:
        if node.element_id in visiting:
            raise RuntimeError(f"V5 Pro hierarchy cycle at {node.element_id!r}.")
        visiting.add(node.element_id)
        result.append(node)
        for child in children.get(node.element_id, []):
            emit(child)
        visiting.remove(node.element_id)

    for root in roots:
        emit(root)
    if len(result) != len(active):
        missing = sorted(set(by_id) - {node.element_id for node in result})
        raise RuntimeError(f"V5 Pro hierarchy omitted nodes: {missing[:8]}")
    return result


def _recomposition_attribution(
    graph: DocumentGraph,
    target: np.ndarray,
    composite: np.ndarray,
    rendered: list[RenderedLayer],
    error: np.ndarray,
    *,
    scale: float,
    solver_tested_pixels: int,
    solver_infeasible_records: list[dict[str, object]],
) -> dict[str, object]:
    """Attribute a failed final pixel to its owner and actual last writer."""

    different = np.any(error > 0, axis=2)
    coordinates = np.argwhere(different)
    different_count = int(len(coordinates))
    solver_infeasible_count = sum(
        int(record["infeasible_pixel_count"])
        for record in solver_infeasible_records
    )
    report: dict[str, object] = {
        "different_pixel_count": different_count,
        "diff_bbox": None,
        "sample_coordinates": [],
        "categorical_visible_owner_counts": {},
        "last_writer_counts": {},
        "last_writer_hidden_full_support_counts": {},
        "unattributed_visible_owner_pixels": 0,
        "unattributed_last_writer_pixels": 0,
        "solver_feasibility": {
            "tested_alpha_pixel_events": int(solver_tested_pixels),
            "infeasible_pixel_events": int(solver_infeasible_count),
            "infeasible_layer_count": len(solver_infeasible_records),
            "layers": solver_infeasible_records,
        },
    }
    if not different_count:
        return report

    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    report["diff_bbox"] = [int(x0), int(y0), int(x1), int(y1)]
    source_resolution = target.shape[1::-1] == graph.canvas_size
    by_id = graph.node_map()

    if source_resolution:
        owner_counts: dict[str, int] = {}
        accounted = np.zeros_like(different)
        for node in graph.nodes:
            if node.review_status == "rejected":
                continue
            ys, xs = node.visible_alpha.canvas_slice(graph.canvas_size)
            hits = different[ys, xs] & (node.visible_alpha.alpha > 0)
            count = int(np.count_nonzero(hits))
            if count:
                owner_counts[node.element_id] = count
                accounted[ys, xs] |= hits
        report["categorical_visible_owner_counts"] = dict(
            sorted(owner_counts.items(), key=lambda item: (-item[1], item[0]))
        )
        report["unattributed_visible_owner_pixels"] = int(
            np.count_nonzero(different & ~accounted)
        )
    else:
        report["categorical_visible_owner_counts"] = {
            "unavailable_at_delivery_scale": different_count
        }

    remaining = different.copy()
    writer_counts: dict[str, int] = {}
    hidden_counts: dict[str, int] = {}
    for layer in reversed(rendered):
        left, top, right, bottom = layer.bbox
        alpha = np.asarray(layer.alpha, dtype=np.uint8) > 0
        hits = remaining[top:bottom, left:right] & alpha
        count = int(np.count_nonzero(hits))
        if not count:
            continue
        layer_id = layer.spec.layer_id
        writer_counts[layer_id] = writer_counts.get(layer_id, 0) + count
        if source_resolution:
            node = by_id.get(layer_id)
            if node is not None and node.visible_alpha.bbox == layer.bbox:
                hidden = hits & (node.visible_alpha.alpha == 0)
                hidden_count = int(np.count_nonzero(hidden))
                if hidden_count:
                    hidden_counts[layer_id] = hidden_count
        local_remaining = remaining[top:bottom, left:right]
        local_remaining[hits] = False
        if not np.any(remaining):
            break
    report["last_writer_counts"] = dict(
        sorted(writer_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    report["last_writer_hidden_full_support_counts"] = dict(
        sorted(hidden_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    report["unattributed_last_writer_pixels"] = int(np.count_nonzero(remaining))

    infeasible_samples = {
        str(record["node_id"]): {
            (int(point["x"]), int(point["y"]))
            for point in record.get("sample_coordinates", [])
            if isinstance(point, dict) and "x" in point and "y" in point
        }
        for record in solver_infeasible_records
    }
    sample_count = min(32, different_count)
    sample_indices = np.linspace(
        0,
        different_count - 1,
        num=sample_count,
        dtype=np.int64,
    )
    samples: list[dict[str, object]] = []
    active_nodes = [
        node for node in graph.nodes if node.review_status != "rejected"
    ]
    for raw_index in sample_indices:
        y, x = (int(value) for value in coordinates[int(raw_index)])
        source_x = min(graph.canvas_size[0] - 1, int(x / scale))
        source_y = min(graph.canvas_size[1] - 1, int(y / scale))
        visible_owners: list[str] = []
        for node in active_nodes:
            crop = node.visible_alpha
            if not (
                crop.left <= source_x < crop.bbox[2]
                and crop.top <= source_y < crop.bbox[3]
            ):
                continue
            if crop.alpha[source_y - crop.top, source_x - crop.left] > 0:
                visible_owners.append(node.element_id)
        last_writer: str | None = None
        last_writer_hidden = False
        for layer in reversed(rendered):
            left, top, right, bottom = layer.bbox
            if not (left <= x < right and top <= y < bottom):
                continue
            if np.asarray(layer.alpha, dtype=np.uint8)[y - top, x - left] <= 0:
                continue
            last_writer = layer.spec.layer_id
            node = by_id.get(last_writer)
            if node is not None:
                crop = node.visible_alpha
                if (
                    crop.left <= source_x < crop.bbox[2]
                    and crop.top <= source_y < crop.bbox[3]
                ):
                    last_writer_hidden = bool(
                        crop.alpha[source_y - crop.top, source_x - crop.left] == 0
                    )
            break
        samples.append(
            {
                "x": x,
                "y": y,
                "source_x": source_x,
                "source_y": source_y,
                "target_rgb": [int(value) for value in target[y, x]],
                "composite_rgb": [int(value) for value in composite[y, x]],
                "absolute_error_rgb": [int(value) for value in error[y, x]],
                "categorical_visible_owner_ids": visible_owners,
                "last_writer_id": last_writer,
                "last_writer_uses_hidden_full_support": last_writer_hidden,
                "last_writer_solver_infeasible_sample": bool(
                    last_writer is not None
                    and (x, y) in infeasible_samples.get(last_writer, set())
                ),
            }
        )
    report["sample_coordinates"] = samples
    return report


def _editable_background(
    target_rgb: np.ndarray,
    clean_background_rgb: np.ndarray,
    graph: DocumentGraph,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep target pixels outside ownership and reveal clean pixels below layers."""

    height, width = target_rgb.shape[:2]
    source_width, source_height = graph.canvas_size
    if clean_background_rgb.shape != (source_height, source_width, 3):
        raise ValueError("Clean background does not match the source canvas.")
    clean = np.asarray(
        Image.fromarray(clean_background_rgb, "RGB").resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    union = np.zeros((source_height, source_width), dtype=np.uint8)
    for node in graph.nodes:
        if node.review_status == "rejected":
            continue
        support = node.full_support or node.visible_alpha
        ys, xs = support.canvas_slice(graph.canvas_size)
        np.maximum(union[ys, xs], support.alpha, out=union[ys, xs])
    scaled_union = _resize_u8(union, (width, height), nearest=True) > 0
    # The bottom layer is genuinely clean. Source texture or low-contrast
    # remnants outside semantic ownership are represented by an explicit V5
    # review layer, never baked back into this background for a perfect preview.
    background = clean.copy()
    return background, scaled_union


def render_document(
    graph: DocumentGraph,
    target_rgb: np.ndarray,
    clean_background_rgb: np.ndarray,
    *,
    scale: float,
) -> RenderedDocument:
    """Render cropped, exclusive mattes without allocating one canvas per layer.

    Pixel colours are solved against the actual clean lower composite with the
    same integer source-over equation used by Pillow, PSD and OpenRaster.  The
    solver may increase alpha only inside nearest-neighbour semantic ownership;
    it can never pull a neighbour into the exported crop.
    """

    if not math.isfinite(scale) or scale < 1.0 or scale > 20.0:
        raise ValueError("scale must be finite and from 1 through 20")
    target = np.asarray(target_rgb)
    if target.dtype != np.uint8 or target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("target_rgb must be uint8 RGB")
    expected_size = (
        int(round(graph.canvas_size[0] * scale)),
        int(round(graph.canvas_size[1] * scale)),
    )
    if target.shape[1::-1] != expected_size:
        raise ValueError(f"Target is {target.shape[1::-1]}, expected {expected_size}.")
    graph.validate()
    background_rgb, scaled_union = _editable_background(target, clean_background_rgb, graph, scale)
    current = background_rgb.copy()
    rendered: list[RenderedLayer] = []
    promoted_pixels = 0
    solver_tested_pixels = 0
    solver_infeasible_records: list[dict[str, object]] = []

    for node in _ordered_nodes(graph):
        support_crop = node.full_support or node.visible_alpha
        alpha_source = support_crop.alpha
        if not np.any(alpha_source):
            continue
        left, top, right, bottom = _scaled_box(support_crop, scale)
        local_size = (right - left, bottom - top)
        base_alpha = _resize_u8(alpha_source, local_size)
        binary_source = (alpha_source > 0).astype(np.uint8) * 255
        nearest_support = _resize_u8(binary_source, local_size, nearest=True) > 0
        # Lanczos has small positive ringing lobes.  A one-source-pixel legal
        # envelope retains real antialiasing but clips remote contamination.
        envelope_source = cv2.dilate(binary_source, np.ones((3, 3), np.uint8), iterations=1)
        legal_envelope = _resize_u8(envelope_source, local_size, nearest=True) > 0
        base_alpha[~legal_envelope] = 0
        if not np.any(base_alpha):
            continue
        below = current[top:bottom, left:right]
        desired = target[top:bottom, left:right].copy()
        # A geometry/panel node may contain a synthesized clean reference.
        # Render that reference throughout its full support, not only below
        # already-known children: otherwise the exact foreground solver would
        # bake every faint unrecognised glyph/shadow back into the standalone
        # geometry PNG. The top technical SOURCE REMAINDER owns all observed
        # deviations and restores the exact source composite independently.
        if node.full_support is not None and node.rgba is not None:
            visible_scaled = _resize_u8(node.visible_alpha.alpha, local_size)
            hidden = (base_alpha > 0) & (visible_scaled == 0)
            cleanliness = (
                node.metadata.get("geometry_cleanliness")
                if isinstance(node.metadata, dict)
                else None
            )
            reference_surface = bool(
                isinstance(cleanliness, dict)
                and cleanliness.get("policy") == "reference_surface_delta_e_carve_v1"
                and cleanliness.get("reference_type") in {
                    "surface_rgb",
                    "constant_colour_rgb",
                }
            )
            reference_domain = (base_alpha > 0) if reference_surface else hidden
            if np.any(reference_domain):
                supplied = _resize_straight_rgba(node.rgba, local_size)
                supplied_alpha = supplied[:, :, 3].astype(np.uint32)
                lower_weight = 255 - supplied_alpha
                supplied_after = (
                    supplied[:, :, :3].astype(np.uint32) * supplied_alpha[:, :, None]
                    + below.astype(np.uint32) * lower_weight[:, :, None]
                    + 127
                ) // 255
                desired[reference_domain] = supplied_after.astype(np.uint8)[reference_domain]
        required = _minimum_pillow_alpha_u8(desired, below)
        promotable = np.where(nearest_support, required, 0).astype(np.uint8)
        alpha = np.maximum(base_alpha, promotable)
        promoted_pixels += int(np.count_nonzero(alpha > base_alpha))
        foreground, feasible = _solve_pillow_foreground_u8(desired, below, alpha)
        tested = alpha > 0
        solver_tested_pixels += int(np.count_nonzero(tested))
        infeasible = tested & ~feasible
        infeasible_count = int(np.count_nonzero(infeasible))
        if infeasible_count:
            infeasible_coordinates = np.argwhere(infeasible)
            solver_infeasible_records.append(
                {
                    "node_id": node.element_id,
                    "infeasible_pixel_count": infeasible_count,
                    "bbox": [left, top, right, bottom],
                    "sample_coordinates": [
                        {"x": int(left + x), "y": int(top + y)}
                        for y, x in infeasible_coordinates[:32]
                    ],
                    "sample_complete": infeasible_count <= 32,
                }
            )
        rgba = np.dstack([foreground, alpha])
        rgba[alpha == 0, :3] = 0
        rgba_image = Image.fromarray(rgba, "RGBA")
        lower = Image.fromarray(
            np.dstack([below, np.full(alpha.shape, 255, dtype=np.uint8)]), "RGBA"
        )
        lower.alpha_composite(rgba_image)
        current[top:bottom, left:right] = np.asarray(lower, dtype=np.uint8)[:, :, :3]

        # The legacy container adapters need only an id/name/category and
        # parent metadata.  Keeping this mask crop-sized avoids hundreds of
        # redundant source-canvas arrays for exhaustive posters.
        spec = LayerSpec(
            layer_id=node.element_id,
            name=node.name,
            category=node.kind,
            mask=alpha_source > 0,
            score=float(node.confidence),
            label=node.kind,
            text=node.text,
            metadata={
                "parent_id": node.parent_id,
                "review_status": node.review_status,
                "move_safe": node.move_safe,
                "occluded": node.occluded,
                "synthesized_hidden_pixels": node.synthesized_hidden_pixels,
                **node.metadata,
            },
        )
        rendered.append(
            RenderedLayer(
                spec=spec,
                rgba=rgba_image,
                alpha=Image.fromarray(alpha, "L"),
                left=left,
                top=top,
                source_bbox=node.bbox,
            )
        )

    error = np.abs(current.astype(np.int16) - target.astype(np.int16))
    mse = float(np.mean(error.astype(np.float64) ** 2)) if error.size else 0.0
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(255.0**2 / mse)
    composite = Image.fromarray(current, "RGB")
    attribution = _recomposition_attribution(
        graph,
        target,
        current,
        rendered,
        error,
        scale=scale,
        solver_tested_pixels=solver_tested_pixels,
        solver_infeasible_records=solver_infeasible_records,
    )
    review_metadata = graph.metadata.get("review")
    raw_groups = (
        review_metadata.get("groups", [])
        if isinstance(review_metadata, dict)
        else []
    )
    user_groups = [dict(group) for group in raw_groups if isinstance(group, dict)]
    return RenderedDocument(
        Image.fromarray(background_rgb, "RGB"),
        rendered,
        composite,
        {
            "rendered_layer_count": len(rendered),
            "user_group_count": len(user_groups),
            "target_size": list(expected_size),
            "background_replacement_pixels": int(np.count_nonzero(scaled_union)),
            "required_alpha_promoted_pixels": promoted_pixels,
            "recomposition_max_abs_error": int(error.max()) if error.size else 0,
            "recomposition_mean_abs_error": round(float(error.mean()), 8) if error.size else 0.0,
            "recomposition_psnr_db": "infinite" if math.isinf(psnr) else round(psnr, 5),
            "recomposition_attribution": attribution,
            "policy": (
                "cropped exclusive alpha; Lanczos within a one-source-pixel envelope; "
                "required alpha restricted to nearest semantic ownership"
            ),
        },
        user_groups,
    )


def export_document(
    output_dir: Path,
    document_name: str,
    rendered_document: RenderedDocument,
    *,
    icc_profile: bytes | None,
) -> ExportedDocument:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_name(document_name) or "ARTWORK"
    paths = {
        "psd": output_dir / f"01_{safe}_EDITABLE.psd",
        "ora": output_dir / f"02_{safe}_MASTER.ora",
        "preview": output_dir / f"03_{safe}_PREVIEW.png",
        "background": output_dir / f"04_{safe}_CLEAN_BACKGROUND.png",
        "contact_sheet": output_dir / f"06_{safe}_CONTACT_SHEET.png",
    }
    save_color_png(rendered_document.composite, paths["preview"], icc_profile)
    save_color_png(rendered_document.background, paths["background"], icc_profile)
    records = export_png_assets(
        output_dir,
        rendered_document.background,
        rendered_document.rendered,
        icc_profile=icc_profile,
    )
    psd_report = export_psd(
        paths["psd"],
        rendered_document.background,
        rendered_document.rendered,
        rendered_document.composite,
        icc_profile=icc_profile,
        user_groups=rendered_document.user_groups,
    )
    if not psd_report.get("created"):
        paths["psd"].unlink(missing_ok=True)
    ora_report = export_ora(
        paths["ora"],
        rendered_document.background,
        rendered_document.rendered,
        rendered_document.composite,
        icc_profile=icc_profile,
        user_groups=rendered_document.user_groups,
    )
    export_contact_sheet(
        paths["contact_sheet"],
        rendered_document.background,
        rendered_document.rendered,
        icc_profile=icc_profile,
    )
    return ExportedDocument(
        rendered_document,
        records,
        {"psd": psd_report, "ora": ora_report},
        paths,
    )
