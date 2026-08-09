"""V5 Pro: exhaustive, reviewable reconstruction of flat poster artwork.

The engine intentionally fails closed.  It never equates a perfect preview
with successful decomposition: every detector proposal is accounted for and
the bundle remains REVIEW_REQUIRED until ownership and clean-background gates
are satisfied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageCms

from v5lib.formats import asset_records, package_layers_zip, save_color_png, sha256_file, write_json
from v5lib.segment import detect_text_regions
from v5pro.cleanplate import build_clean_plate
from v5pro.exporter import export_document, render_document
from v5pro.fusion import fuse_inventory_and_layerd
from v5pro.geometry_backend import build_geometry_nodes
from v5pro.hierarchy import AUTO_ORGANIZATION_SOURCE, assign_spatial_hierarchy
from v5pro.inventory import InventoryResult, build_inventory
from v5pro.layerd_backend import run_layerd
from v5pro.ownership import enforce_exclusive_ownership
from v5pro.qa import (
    build_quality_report,
    enforce_geometry_export_topology_preflight,
    write_inventory_files,
    write_qa_artifacts,
)
from v5pro.residual_backend import ResidualResult, reconcile_residual_elements
from v5pro.review_adapter import apply_review_checkpoint, prepare_review_bundle
from v5pro.review_server import run_review_ui, write_review_checkpoint
from v5pro.schema import AlphaCrop, DocumentGraph, ElementNode, ProposalRecord, union_alpha
from v5pro.semantic_backend import run_semantic_proposals
from v5pro.text_export_preflight import enforce_text_export_purity_preflight


Image.MAX_IMAGE_PIXELS = 500_000_000
ENGINE_SCHEMA = "V5_PRODUCTION_LAYERS_V2"
PIPELINE_NAME = "V5_SMART_EDITABLE_LAYERS"
REVIEW_RESUME_SIGNATURE = "V5_PRO_INVENTORY_2026_08_09_K"
MIN_SCALE = 1.0
MAX_SCALE = 20.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5 Pro production poster layer reconstruction")
    parser.add_argument("input", type=Path)
    parser.add_argument("scale", type=float)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--master", type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--detail", choices=("exhaustive", "grouped"), default="exhaustive")
    parser.add_argument("--inpaint", choices=("auto", "poster", "lama"), default="auto")
    parser.add_argument("--review-mode", choices=("gui", "defer", "auto", "strict"), default="gui")
    parser.add_argument("--review-file", type=Path, default=None)
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--layerd-iterations", type=int, default=8)
    # Accepted only so older launchers do not break.  Exhaustive V5 Pro never
    # truncates useful elements to this legacy count.
    parser.add_argument("--max-layers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--app-version", default="dev")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, str]:
    source = args.input.expanduser().resolve()
    target = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Không tìm thấy ảnh đầu vào: {source}")
    if not MIN_SCALE <= float(args.scale) <= MAX_SCALE:
        raise SystemExit("Hệ số n phải từ 1 đến 20.")
    if not float(args.scale).is_integer():
        raise SystemExit(
            "V5 layers requires an integer scale x1..x20 for stable editable masks; "
            "separate at x1, then upscale the edited composite for fractional sizing."
        )
    if not 1 <= int(args.layerd_iterations) <= 32:
        raise SystemExit("--layerd-iterations phải từ 1 đến 32.")
    if target.exists():
        raise SystemExit(f"Thư mục output đã tồn tại; V5 Pro không tự xóa: {target}")
    target.mkdir(parents=True)
    return source, target, str(args.name or source.stem)


def _load_rgb(path: Path) -> tuple[np.ndarray, bytes | None]:
    with Image.open(path) as image:
        image.load()
        profile = image.info.get("icc_profile")
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy(), bytes(profile) if profile else None


def _srgb_profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _sha256_rgb(image_rgb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image_rgb).tobytes()).hexdigest()


def _print_qa_hard_failures(qa_report: dict[str, object]) -> None:
    """Print every hard QA reason before the launcher removes failed staging."""

    if str(qa_report.get("status") or "") != "FAIL":
        return
    raw = qa_report.get("hard_failures")
    failures = raw if isinstance(raw, list) else []
    print("  LỖI QA CỨNG:", flush=True)
    if not failures:
        print("    - QA trả về FAIL nhưng không ghi hard_failures.", flush=True)
        return
    for index, failure in enumerate(failures, start=1):
        concise = " ".join(str(failure).split()) or "(lý do trống)"
        print(f"    {index}. {concise}", flush=True)


def _open_chrome(url: str) -> bool:
    candidates = [
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            subprocess.Popen(
                [str(candidate), "--new-window", url],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
    import webbrowser

    return bool(webbrowser.open(url))


def _protected_alpha(graph: DocumentGraph) -> np.ndarray:
    def is_child_content(node) -> bool:
        if node.review_status == "rejected":
            return False
        if node.kind in {"text", "price", "qr"}:
            return True
        if node.kind not in {"product", "logo", "icon", "badge", "decoration"}:
            return False
        sources = {
            str(record.get("source", ""))
            for record in node.evidence
            if isinstance(record, dict)
        }
        return any(
            source.startswith("grounding_dino")
            or source.startswith("sam2")
            or source.startswith("birefnet")
            for source in sources
        )

    return union_alpha(
        [node.visible_alpha for node in graph.nodes if is_child_content(node)],
        graph.canvas_size,
    )


def _merge_geometry(graph: DocumentGraph, geometry) -> None:
    existing_nodes = set(graph.node_map())
    for node in geometry.nodes:
        if node.element_id in existing_nodes:
            raise RuntimeError(f"Trùng mã geometry node: {node.element_id}")
        graph.add_node(node)
        existing_nodes.add(node.element_id)
    by_id = {proposal.proposal_id: proposal for proposal in graph.proposals}
    decisions = {decision.proposal_id: decision for decision in geometry.decisions}
    geometry_records = {record.proposal_id: record for record in geometry.proposals}
    for proposal_id, decision in decisions.items():
        current = by_id.get(proposal_id)
        replacement = geometry_records.get(proposal_id)
        if current is None or replacement is None:
            continue
        current.status = replacement.status
        current.owner_ids = list(replacement.owner_ids)
        current.reason = replacement.reason
        current.evidence = dict(replacement.evidence)
    graph.metadata["geometry"] = geometry.report
    graph.validate()


def _merge_residual(graph: DocumentGraph, residual: ResidualResult) -> None:
    """Append independently reconciled residuals without hiding ledger conflicts."""

    node_ids = set(graph.node_map())
    proposal_ids = {proposal.proposal_id for proposal in graph.proposals}
    for node in residual.nodes:
        if node.element_id in node_ids:
            raise RuntimeError(f"Trùng mã residual node: {node.element_id}")
        graph.add_node(node)
        node_ids.add(node.element_id)
    for proposal in residual.proposals:
        if proposal.proposal_id in proposal_ids:
            raise RuntimeError(f"Trùng mã residual proposal: {proposal.proposal_id}")
        graph.add_proposal(proposal)
        proposal_ids.add(proposal.proposal_id)
    graph.metadata["residual_reconciliation"] = residual.report
    graph.validate()


def _geometry_source_remainder_reservation(graph: DocumentGraph) -> np.ndarray:
    """Reserve carved clean-geometry holes for the technical source layer.

    The initial graph can still contain low-priority raw LayerD/residual owners
    at pixels that poster geometry deliberately carved.  Seeding the technical
    remainder over those hidden supports lets categorical ownership transfer
    the pixels away from raw guesses, while verified text/semantic/frame owners
    retain their higher-priority claims.  The later visible-complement
    reconciliation then keeps exactly the genuinely unowned subset.
    """

    width, height = graph.canvas_size
    reservation = np.zeros((height, width), dtype=bool)
    for node in graph.nodes:
        if node.review_status == "rejected" or not isinstance(node.metadata, dict):
            continue
        policy = node.metadata.get("geometry_cleanliness")
        if not isinstance(policy, dict):
            continue
        try:
            carved_pixels = int(policy.get("carved_pixel_count", 0))
        except (TypeError, ValueError):
            carved_pixels = 0
        if carved_pixels <= 0:
            continue
        full = (node.full_support or node.visible_alpha).to_canvas(graph.canvas_size)
        visible = node.visible_alpha.to_canvas(graph.canvas_size)
        reservation |= (full > 0) & (visible == 0)
    return reservation


def _add_base_residual_layer(
    graph: DocumentGraph,
    source_rgb: np.ndarray,
    bottom_surface_rgb: np.ndarray,
) -> dict[str, object]:
    """Expose every still-unowned source pixel above the genuinely clean base.

    A flat bitmap has no authoritative hidden design graph. The clean bottom
    surface is useful for editing, but replacing all low-contrast source
    texture with it would silently lose pixels. Conversely, painting those
    pixels back into the background would make the clean plate dishonest. This
    explicitly unresolved lower layer keeps the source composite exact while
    remaining independently hideable in PSD/ORA.

    The mask is the complement of categorical *visible* ownership.  It can
    therefore cover a clean panel's hidden full support at contamination holes
    while remaining mutually exclusive with every extracted visible child.
    The layer is deliberately above extracted nodes: visible source pixels
    reproduce the exact bitmap; hiding it reveals the synthesized clean
    surfaces underneath.
    """

    source = np.asarray(source_rgb)
    surface = np.asarray(bottom_surface_rgb)
    width, height = graph.canvas_size
    if source.dtype != np.uint8 or source.shape != (height, width, 3):
        raise ValueError("Base residual source must match the uint8 graph canvas.")
    if surface.dtype != np.uint8 or surface.shape != source.shape:
        raise ValueError("Bottom surface must match the uint8 source image.")

    active_supports = [
        node.visible_alpha
        for node in graph.nodes
        if node.review_status != "rejected"
    ]
    owned = union_alpha(active_supports, graph.canvas_size) > 0
    geometry_reservation = _geometry_source_remainder_reservation(graph)
    remainder = ~owned | geometry_reservation
    pixel_count = int(np.count_nonzero(remainder))
    component_count = (
        int(cv2.connectedComponents(remainder.astype(np.uint8), 8)[0] - 1)
        if pixel_count
        else 0
    )
    report: dict[str, object] = {
        "policy": (
            "all source pixels outside extracted visible ownership are stored "
            "in one explicit hideable technical layer above extracted nodes"
        ),
        "pixel_count": pixel_count,
        "pixel_fraction": round(pixel_count / max(1, width * height), 8),
        "component_count": component_count,
        "geometry_hidden_reservation_pixels": int(
            np.count_nonzero(geometry_reservation)
        ),
        "geometry_reservation_overlap_with_initial_visible_owners": int(
            np.count_nonzero(geometry_reservation & owned)
        ),
        "node_id": None,
        "proposal_id": None,
    }
    if not pixel_count:
        graph.metadata["base_source_residual"] = report
        return report

    node_id = "BASE_SOURCE_RESIDUAL_0001"
    proposal_id = "BASE_SOURCE_RESIDUAL_PROPOSAL_0001"
    if node_id in graph.node_map() or any(
        proposal.proposal_id == proposal_id for proposal in graph.proposals
    ):
        raise RuntimeError("Base source residual identifiers already exist in the graph.")

    ys, xs = np.where(remainder)
    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
    local_alpha = remainder[top:bottom, left:right].astype(np.uint8) * 255
    rgba = np.dstack(
        [source[top:bottom, left:right].copy(), local_alpha.copy()]
    ).astype(np.uint8, copy=False)
    visible = AlphaCrop(left, top, local_alpha.copy())
    full_support = AlphaCrop(left, top, local_alpha.copy())
    envelope = AlphaCrop(left, top, local_alpha.copy())
    top_z = max(
        (node.z_index for node in graph.nodes if node.review_status != "rejected"),
        default=0,
    ) + 1
    graph.add_node(
        ElementNode(
            element_id=node_id,
            name="REVIEW - SOURCE REMAINDER (hide to reveal clean background)",
            kind="unknown",
            visible_alpha=visible,
            full_support=full_support,
            semantic_envelope=envelope,
            z_index=top_z,
            confidence=0.25,
            review_status="unresolved",
            move_safe=False,
            rgba=rgba,
            evidence=[
                {
                    "source": "exact_unowned_source_remainder",
                    "pixel_count": pixel_count,
                    "bottom_surface_sha256": _sha256_rgb(surface),
                }
            ],
            metadata={
                "role": "exact_source_remainder_above_clean_base",
                "user_action": (
                    "Hide this layer to reveal the reconstructed clean background; "
                    "keep it visible for exact source appearance."
                ),
                "contains_quiet_source_regions": True,
                "production_gate": "review_required",
                "ownership_policy": "complement_of_nontechnical_visible_ownership",
                "stack_policy": "above_all_extracted_nodes",
            },
        )
    )
    graph.add_proposal(
        ProposalRecord(
            proposal_id=proposal_id,
            source="exact_unowned_source_remainder",
            kind_hint="unknown",
            bbox=(left, top, right, bottom),
            confidence=0.25,
            status="assigned",
            owner_ids=[node_id],
            reason=(
                "Explicit source remainder preserves the exact composite without "
                "baking unclassified pixels into the clean background."
            ),
            evidence={
                "pixel_count": pixel_count,
                "component_count": component_count,
            },
        )
    )
    report["node_id"] = node_id
    report["proposal_id"] = proposal_id
    graph.metadata["base_source_residual"] = report
    graph.validate()
    return report


def _reconcile_base_residual_layers(
    graph: DocumentGraph,
    source_rgb: np.ndarray,
    bottom_surface_rgb: np.ndarray,
) -> dict[str, object]:
    """Make technical remainder exactly cover non-semantic visible ownership.

    Review operations may delete a false-positive node, subtract its mask, add
    a missing object, or split the technical remainder. Recomputing this
    complement after ownership guarantees that those edits never create a
    source-pixel hole or re-bake content into clean geometry.  Full supports
    may overlap below this top technical layer by design.
    """

    source = np.asarray(source_rgb)
    surface = np.asarray(bottom_surface_rgb)
    width, height = graph.canvas_size
    if source.dtype != np.uint8 or source.shape != (height, width, 3):
        raise ValueError("Base residual reconciliation source is invalid.")
    if surface.dtype != np.uint8 or surface.shape != source.shape:
        raise ValueError("Base residual reconciliation surface is invalid.")

    def is_base(node: ElementNode) -> bool:
        return (
            node.review_status != "rejected"
            and node.metadata.get("role")
            == "exact_source_remainder_above_clean_base"
        )

    base_nodes = sorted(
        [node for node in graph.nodes if is_base(node)],
        key=lambda node: (node.z_index, node.element_id),
    )
    other_supports = [
        node.visible_alpha
        for node in graph.nodes
        if node.review_status != "rejected" and not is_base(node)
    ]
    target = ~(union_alpha(other_supports, graph.canvas_size) > 0)
    target_pixels = int(np.count_nonzero(target))
    top_z = max(
        (
            node.z_index
            for node in graph.nodes
            if node.review_status != "rejected" and not is_base(node)
        ),
        default=0,
    ) + 1

    if not base_nodes and target_pixels:
        node_id = "BASE_SOURCE_RESIDUAL_0001"
        if node_id in graph.node_map():
            suffix = 2
            while f"BASE_SOURCE_RESIDUAL_{suffix:04d}" in graph.node_map():
                suffix += 1
            node_id = f"BASE_SOURCE_RESIDUAL_{suffix:04d}"
        ys, xs = np.where(target)
        left, top = int(xs.min()), int(ys.min())
        right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
        alpha = target[top:bottom, left:right].astype(np.uint8) * 255
        crop = AlphaCrop(left, top, alpha.copy())
        graph.add_node(
            ElementNode(
                node_id,
                "REVIEW - SOURCE REMAINDER (hide to reveal clean background)",
                "unknown",
                crop,
                top_z,
                semantic_envelope=AlphaCrop(left, top, alpha.copy()),
                full_support=AlphaCrop(left, top, alpha.copy()),
                confidence=0.25,
                review_status="unresolved",
                move_safe=False,
                rgba=np.dstack(
                    [source[top:bottom, left:right].copy(), alpha.copy()]
                ),
                evidence=[
                    {
                        "source": "exact_unowned_source_remainder",
                        "bottom_surface_sha256": _sha256_rgb(surface),
                    }
                ],
                metadata={
                    "role": "exact_source_remainder_above_clean_base",
                    "user_action": (
                        "Hide this layer to reveal the reconstructed clean background; "
                        "keep it visible for exact source appearance."
                    ),
                    "contains_quiet_source_regions": True,
                    "production_gate": "review_required",
                    "rebuilt_after_review": True,
                    "ownership_policy": "complement_of_nontechnical_visible_ownership",
                    "stack_policy": "above_all_extracted_nodes",
                },
            )
        )
        base_nodes = [graph.node_map()[node_id]]

    allocated = np.zeros((height, width), dtype=bool)
    masks: dict[str, np.ndarray] = {}
    for base_index, node in enumerate(base_nodes):
        current = (node.full_support or node.visible_alpha).to_canvas(graph.canvas_size) > 0
        kept = current & target & ~allocated
        masks[node.element_id] = kept
        allocated |= kept
    if base_nodes:
        masks[base_nodes[0].element_id] |= target & ~allocated

    removed_ids: set[str] = set()
    changed_ids: list[str] = []
    for node in base_nodes:
        mask = masks[node.element_id]
        if not np.any(mask):
            removed_ids.add(node.element_id)
            continue
        previous_mask = (node.full_support or node.visible_alpha).to_canvas(
            graph.canvas_size
        ) > 0
        changed = not np.array_equal(mask, previous_mask)
        ys, xs = np.where(mask)
        left, top = int(xs.min()), int(ys.min())
        right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
        alpha = mask[top:bottom, left:right].astype(np.uint8) * 255
        node.visible_alpha = AlphaCrop(left, top, alpha.copy())
        node.full_support = AlphaCrop(left, top, alpha.copy())
        node.semantic_envelope = AlphaCrop(left, top, alpha.copy())
        node.removal_footprint = None
        node.rgba = np.dstack(
            [source[top:bottom, left:right].copy(), alpha.copy()]
        )
        node.kind = "unknown"
        node.move_safe = False
        node.parent_id = None
        node.z_index = top_z + base_index
        node.metadata["ownership_policy"] = (
            "complement_of_nontechnical_visible_ownership"
        )
        node.metadata["stack_policy"] = "above_all_extracted_nodes"
        if changed:
            node.review_status = "unresolved"
            node.metadata["reconciled_after_review"] = True
            changed_ids.append(node.element_id)

    if removed_ids:
        graph.nodes[:] = [node for node in graph.nodes if node.element_id not in removed_ids]
        for node in graph.nodes:
            if node.parent_id in removed_ids:
                node.parent_id = None
        review = graph.metadata.get("review")
        if isinstance(review, dict) and isinstance(review.get("groups"), list):
            cleaned_groups: list[dict[str, object]] = []
            for raw_group in review["groups"]:
                if not isinstance(raw_group, dict):
                    continue
                members = [
                    str(member)
                    for member in raw_group.get("member_ids", [])
                    if str(member) not in removed_ids
                ]
                if len(members) >= 2:
                    raw_group["member_ids"] = members
                    cleaned_groups.append(raw_group)
            review["groups"] = cleaned_groups

    active_base_ids = [
        node.element_id for node in graph.nodes if is_base(node)
    ]
    base_proposals = [
        proposal
        for proposal in graph.proposals
        if proposal.source == "exact_unowned_source_remainder"
    ]
    for proposal in graph.proposals:
        if removed_ids.intersection(proposal.owner_ids):
            proposal.owner_ids = [
                owner for owner in proposal.owner_ids if owner not in removed_ids
            ]
            if proposal.status == "assigned" and not proposal.owner_ids:
                proposal.status = "unresolved"
                proposal.reason = "review_removed_previous_owner"
    if base_proposals:
        primary = base_proposals[0]
        primary.owner_ids = list(active_base_ids)
        if active_base_ids:
            primary.status = "assigned"
            primary.reason = (
                "Technical source remainder reconciled after review/ownership."
            )
        else:
            primary.status = "rejected"
            primary.reason = "No unowned source pixels remain."
        for duplicate in base_proposals[1:]:
            duplicate.status = "rejected"
            duplicate.owner_ids = []
            duplicate.reason = "Duplicate technical remainder proposal consolidated."
    elif active_base_ids:
        node = graph.node_map()[active_base_ids[0]]
        graph.add_proposal(
            ProposalRecord(
                "BASE_SOURCE_RESIDUAL_PROPOSAL_0001",
                "exact_unowned_source_remainder",
                "unknown",
                node.bbox,
                0.25,
                "assigned",
                list(active_base_ids),
                "Technical source remainder rebuilt after review.",
            )
        )

    other_full_union = union_alpha(
        [
            node.full_support or node.visible_alpha
            for node in graph.nodes
            if node.review_status != "rejected" and not is_base(node)
        ],
        graph.canvas_size,
    ) > 0
    base_visible_union = union_alpha(
        [node.visible_alpha for node in graph.nodes if is_base(node)],
        graph.canvas_size,
    ) > 0
    report: dict[str, object] = {
        "policy": "technical remainder is the exact complement of all active non-base visible ownership",
        "target_pixels": target_pixels,
        "base_node_ids": active_base_ids,
        "changed_node_ids": changed_ids,
        "removed_empty_node_ids": sorted(removed_ids),
        "coverage_exact": bool(
            np.array_equal(
                base_visible_union,
                target,
            )
        ),
        "visible_overlap_with_extracted_pixels": int(
            np.count_nonzero(base_visible_union & ~target)
        ),
        "lower_full_support_overlap_pixels": int(
            np.count_nonzero(base_visible_union & other_full_union)
        ),
        "minimum_remainder_z_index": min(
            (node.z_index for node in graph.nodes if is_base(node)),
            default=None,
        ),
        "maximum_extracted_z_index": max(
            (
                node.z_index
                for node in graph.nodes
                if node.review_status != "rejected" and not is_base(node)
            ),
            default=None,
        ),
    }
    graph.metadata["base_source_residual"] = report
    graph.validate()
    return report


def _proposal_masks(inventory: InventoryResult) -> dict[str, object]:
    return {
        item.record.proposal_id: item.mask_hint
        for item in inventory.proposals
        if item.mask_hint is not None
    }


def _apply_compatible_previous_review(
    graph: DocumentGraph,
    checkpoint_path: Path,
    source_rgb: np.ndarray,
) -> tuple[DocumentGraph, dict[str, object]]:
    """Reuse user decisions only when they belong to this exact source/inventory."""

    checkpoint = checkpoint_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"Không tìm thấy checkpoint V5 đã duyệt: {checkpoint}")
    try:
        document = json.loads(checkpoint.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Không đọc được checkpoint V5 đã duyệt: {checkpoint}") from exc
    if not isinstance(document, dict) or document.get("schema") != "V5_LAYER_REVIEW_V2":
        raise RuntimeError("Checkpoint cũ không đúng schema V5_LAYER_REVIEW_V2.")
    if document.get("review_resume_signature") != REVIEW_RESUME_SIGNATURE:
        raise RuntimeError(
            "Checkpoint belongs to an older or incompatible V5 inventory; "
            "start a fresh V5 Pro review instead of applying stale decisions."
        )
    reference = document.get("source")
    if not isinstance(reference, str) or not reference.strip():
        raise RuntimeError("Checkpoint cũ thiếu ảnh nguồn kiểm chứng.")
    relative = PurePosixPath(reference.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Checkpoint cũ chứa đường dẫn ảnh nguồn không an toàn.")
    old_source = (checkpoint.parent / Path(*relative.parts)).resolve()
    if checkpoint.parent.resolve() not in old_source.parents or not old_source.is_file():
        raise RuntimeError("Ảnh nguồn của checkpoint cũ bị thiếu hoặc nằm ngoài bundle.")
    old_rgb, _profile = _load_rgb(old_source)
    if old_rgb.shape != source_rgb.shape or not np.array_equal(old_rgb, source_rgb):
        raise RuntimeError("Checkpoint đã duyệt thuộc ảnh khác; V5 từ chối áp nhầm quyết định.")
    old_proposals = document.get("proposals")
    if not isinstance(old_proposals, list):
        raise RuntimeError("Checkpoint cũ thiếu proposal ledger.")
    previous_engine_ids = {
        str(record.get("id"))
        for record in old_proposals
        if isinstance(record, dict)
        and record.get("id")
        and record.get("source") != "local_review_rectangle"
    }
    unsupported_user_sources = {
        str(record.get("source"))
        for record in old_proposals
        if isinstance(record, dict)
        and record.get("id")
        and str(record.get("id")) not in previous_engine_ids
        and record.get("source") != "local_review_rectangle"
    }
    current_ids = {record.proposal_id for record in graph.proposals}
    if previous_engine_ids != current_ids or unsupported_user_sources:
        raise RuntimeError(
            "Checkpoint cũ được tạo bởi inventory khác; cần duyệt lại để không bỏ sót layer mới."
        )
    restored = apply_review_checkpoint(graph, checkpoint, source_rgb)
    return restored, document


def _preserve_review_metadata(
    checkpoint_path: Path,
    previous_document: dict[str, object],
) -> None:
    fresh = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
    valid_node_ids = {
        str(record.get("id"))
        for record in fresh.get("nodes", [])
        if isinstance(record, dict) and record.get("id")
    }
    groups = previous_document.get("groups")
    if isinstance(groups, list):
        # Previous automatic folders are implementation output, not user
        # authority. Preserve only explicit review-created groups, then append
        # freshly regenerated auto folders that do not touch those members.
        explicit_groups: list[dict[str, object]] = []
        explicit_members: set[str] = set()
        explicit_ids: set[str] = set()
        fresh_explicit_by_id = {
            str(raw_group.get("id") or ""): raw_group
            for raw_group in fresh.get("groups", [])
            if isinstance(raw_group, dict)
            and raw_group.get("source") != AUTO_ORGANIZATION_SOURCE
            and str(raw_group.get("id") or "")
        }
        for raw_group in groups:
            if (
                not isinstance(raw_group, dict)
                or raw_group.get("source") == AUTO_ORGANIZATION_SOURCE
            ):
                continue
            group_id = str(raw_group.get("id") or "")
            # assign_organizational_groups is the final-stack authority. If it
            # dissolved an old cross-parent/non-contiguous user folder, never
            # resurrect that stale record merely because its leaves still
            # exist in the checkpoint.
            fresh_group = fresh_explicit_by_id.get(group_id)
            if fresh_group is None:
                continue
            members = [
                str(member)
                for member in raw_group.get("member_ids", [])
                if str(member) in valid_node_ids
            ]
            if len(members) >= 2:
                fresh_members = [
                    str(member) for member in fresh_group.get("member_ids", [])
                ]
                cleaned = (
                    dict(raw_group)
                    if members == fresh_members
                    else dict(fresh_group)
                )
                cleaned["member_ids"] = fresh_members
                explicit_groups.append(cleaned)
                explicit_members.update(fresh_members)
                explicit_ids.add(str(cleaned.get("id") or ""))
        fresh_auto_groups: list[dict[str, object]] = []
        for raw_group in fresh.get("groups", []):
            if (
                not isinstance(raw_group, dict)
                or raw_group.get("source") != AUTO_ORGANIZATION_SOURCE
                or str(raw_group.get("id") or "") in explicit_ids
            ):
                continue
            members = [str(member) for member in raw_group.get("member_ids", [])]
            if (
                len(members) >= 2
                and set(members).issubset(valid_node_ids)
                and not explicit_members.intersection(members)
            ):
                fresh_auto_groups.append(dict(raw_group))
        fresh["groups"] = [*explicit_groups, *fresh_auto_groups]
    review = previous_document.get("review")
    if isinstance(review, dict):
        pending = any(
            isinstance(record, dict)
            and record.get("review_status", record.get("status")) == "unresolved"
            for record in fresh.get("nodes", [])
        ) or any(
            isinstance(record, dict) and record.get("status") == "unresolved"
            for record in fresh.get("proposals", [])
        )
        preserved_review = dict(review)
        preserved_review["finished"] = bool(review.get("finished")) and not pending
        fresh["review"] = preserved_review
    write_review_checkpoint(checkpoint_path, fresh)


def _write_user_guide(output_dir: Path, name: str, status: str) -> Path:
    path = output_dir / "00_HUONG_DAN_MO_FILE.txt"
    path.write_text(
        "V5 PRO - FILE LAYER CHỈNH SỬA\n"
        "================================\n\n"
        f"Ảnh: {name}\nTrạng thái kiểm định: {status}\n\n"
        "1. Mở file 01_*_EDITABLE.psd bằng Photoshop, Photopea hoặc Canva.\n"
        "2. Nếu phần mềm không đọc PSD tốt, mở 02_*_MASTER.ora bằng Krita/Photopea.\n"
        "3. LAYERS chứa PNG trong suốt; MASKS chứa mask tương ứng.\n"
        "4. Xem ELEMENT_INVENTORY.csv để tìm layer theo tên và loại.\n"
        "5. PSD/ORA dùng folder pass-through để gọn màn hình; mỗi layer nhỏ bên trong vẫn\n"
        "   giữ mask và chỉnh riêng được. Folder không gộp các pixel thành một ảnh.\n"
        "6. Xem nhanh ở 06_*_CONTACT_SHEET.png. Với bộ nhiều layer, mở thêm từng trang\n"
        "   CONTACT_SHEETS/page_*.png; file bìa cố ý được giữ nhỏ để dễ mở.\n"
        "7. Nếu trạng thái là REVIEW_REQUIRED, mở _KY_THUAT/QA_REPORT.html và chạy\n"
        "   lại giao diện duyệt. Không nên giao bán trước khi các vùng cần duyệt đã được xác nhận.\n\n"
        "Trong trang duyệt, mỗi mục chờ có lời giải thích dễ hiểu. Một mask sản phẩm sạch có\n"
        "thể kéo cả cụm nhưng vẫn cần xác nhận nếu chưa đủ bằng chứng đó là một vật thể đơn.\n\n"
        "Lưu ý trung thực: pixel bị che trong ảnh phẳng không tồn tại. Nền phía dưới là phần\n"
        "được tổng hợp có kiểm định, không phải file thiết kế gốc được khôi phục thần kỳ.\n",
        encoding="utf-8-sig",
    )
    return path


def _write_ghost_heatmap(path: Path, heatmap: np.ndarray, icc_profile: bytes | None) -> None:
    value = np.asarray(heatmap, dtype=np.float32)
    if value.size and float(value.max()) > 1.0:
        value = value / 255.0
    value = np.clip(value, 0.0, 1.0)
    red = np.clip(np.rint(value * 255), 0, 255).astype(np.uint8)
    image = np.dstack([red, np.zeros_like(red), 255 - red])
    save_color_png(Image.fromarray(image, "RGB"), path, icc_profile)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_path, output_dir, name = validate_args(args)
    started = time.perf_counter()
    source_rgb, source_icc = _load_rgb(source_path)
    source_height, source_width = source_rgb.shape[:2]
    output_icc = source_icc or _srgb_profile()
    final_size = (int(round(source_width * args.scale)), int(round(source_height * args.scale)))
    if args.master:
        master_path = args.master.expanduser().resolve()
        target_rgb, master_icc = _load_rgb(master_path)
        if target_rgb.shape[1::-1] != final_size:
            raise RuntimeError(f"Ảnh master là {target_rgb.shape[1::-1]}, cần {final_size}.")
        output_icc = master_icc or output_icc
        master_policy = "V3 AI master supplied by unified launcher"
    else:
        target_rgb = np.asarray(
            Image.fromarray(source_rgb, "RGB").resize(final_size, Image.Resampling.LANCZOS),
            dtype=np.uint8,
        ).copy()
        master_policy = "Lanczos fallback because no --master was supplied"

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("  V5 PRO bước 1/10: kiểm kê chữ, QR, khung, đường và mọi đảo cạnh...", flush=True)
    inventory = build_inventory(source_rgb, source_path)
    semantic_report: dict[str, object]
    if args.no_semantic:
        semantic_report = {"disabled": True, "proposal_count": 0}
    else:
        semantic = run_semantic_proposals(source_rgb, device=device, progress=print)
        inventory.proposals.extend(semantic.proposals)
        semantic_report = semantic.report
    ids = [item.record.proposal_id for item in inventory.proposals]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Bộ kiểm kê đa nguồn tạo mã proposal trùng nhau.")
    inventory.detector_reports["semantic"] = semantic_report
    inventory.detector_reports["total_after_semantic"] = len(inventory.proposals)

    print("  V5 PRO bước 2/10: LayerD tạo nguồn matte đồ họa phụ trợ...", flush=True)
    engine_root = Path(__file__).resolve().parent
    layerd = run_layerd(
        source_rgb,
        vendor_root=engine_root / "vendor" / "LayerD",
        lama_model=Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "big-lama.pt",
        device=device,
        max_iterations=args.layerd_iterations,
        process_size=1024,
    )
    print("  V5 PRO bước 3/10: hợp nhất ownership chữ, sản phẩm và chi tiết nhỏ...", flush=True)
    fusion = fuse_inventory_and_layerd(source_rgb, layerd, inventory)
    graph = fusion.graph

    print("  V5 PRO bước 4/10: dựng khung/rule/ribbon bằng hình học bảo toàn lỗ...", flush=True)
    geometry = build_geometry_nodes(
        source_rgb,
        inventory,
        protected_alpha=_protected_alpha(graph),
        z_start=400_000,
    )
    _merge_geometry(graph, geometry)
    print("  V5 PRO bước 5/10: truy tìm mọi chi tiết còn dính trong nền...", flush=True)
    residual = reconcile_residual_elements(
        source_rgb,
        graph,
        # Residual supports are inferred after the primary layers, but their
        # flat surfaces are lower layers. Existing semantic/text/raw owners
        # must remain above any synthesized child holes.
        z_start=-100_000,
        progress=print,
    )
    _merge_residual(graph, residual)
    _add_base_residual_layer(graph, source_rgb, residual.bottom_surface_rgb)
    graph.metadata["review_resume_signature"] = REVIEW_RESUME_SIGNATURE
    print("  V5 PRO bước 6/10: khóa quyền sở hữu độc quyền cho từng pixel...", flush=True)
    ownership_report = enforce_exclusive_ownership(graph, source_rgb)
    _reconcile_base_residual_layers(graph, source_rgb, residual.bottom_surface_rgb)
    geometry_topology_preflight = enforce_geometry_export_topology_preflight(graph)
    text_export_purity_preflight = enforce_text_export_purity_preflight(
        graph, source_rgb
    )
    assign_spatial_hierarchy(graph)

    previous_review: dict[str, object] | None = None
    if args.review_file is not None:
        graph, previous_review = _apply_compatible_previous_review(
            graph, args.review_file, source_rgb
        )
        ownership_report = {
            "before_review": ownership_report,
            "after_review": enforce_exclusive_ownership(graph, source_rgb),
        }
        _reconcile_base_residual_layers(graph, source_rgb, residual.bottom_surface_rgb)
        geometry_topology_preflight = enforce_geometry_export_topology_preflight(graph)
        text_export_purity_preflight = enforce_text_export_purity_preflight(
            graph, source_rgb
        )
        assign_spatial_hierarchy(graph)
        print("  Đã áp lại toàn bộ quyết định duyệt V5 của đúng ảnh này.", flush=True)

    technical_dir = output_dir / "_KY_THUAT"
    checkpoint_path = prepare_review_bundle(
        graph,
        source_rgb,
        technical_dir,
        source_sha256=sha256_file(source_path),
        icc_profile=output_icc,
        proposal_masks=_proposal_masks(inventory),  # type: ignore[arg-type]
    )
    if previous_review is not None:
        _preserve_review_metadata(checkpoint_path, previous_review)
    review_document: dict[str, object] | None = previous_review
    checkpoint_document = json.loads(
        checkpoint_path.read_text(encoding="utf-8-sig")
    )
    previous_finished = bool(
        isinstance(checkpoint_document.get("review"), dict)
        and checkpoint_document["review"].get("finished")
    )
    if args.review_mode == "gui" and not previous_finished:
        print("  V5 PRO bước 7/10: Chrome đang mở giao diện duyệt trực quan...", flush=True)
        run_review_ui(checkpoint_path, open_browser=_open_chrome)
        review_document = json.loads(
            checkpoint_path.read_text(encoding="utf-8-sig")
        )
        graph = apply_review_checkpoint(graph, checkpoint_path, source_rgb)
        ownership_report = {
            "before_review": ownership_report,
            "after_review": enforce_exclusive_ownership(graph, source_rgb),
        }
        _reconcile_base_residual_layers(graph, source_rgb, residual.bottom_surface_rgb)
        geometry_topology_preflight = enforce_geometry_export_topology_preflight(graph)
        text_export_purity_preflight = enforce_text_export_purity_preflight(
            graph, source_rgb
        )
        assign_spatial_hierarchy(graph)
    elif args.review_mode == "gui" and previous_finished:
        print(
            "  V5 PRO bước 7/10: checkpoint cũ đã hoàn tất; không bắt người dùng duyệt lại.",
            flush=True,
        )
    else:
        print(
            "  V5 PRO bước 7/10: đã lưu checkpoint; chế độ này không mở giao diện duyệt.",
            flush=True,
        )

    # Re-evaluate the exact final export supports after every possible review
    # path. The first pass keeps unsafe geometry/text out of the UI's automatic
    # state; this pass prevents a resumed/manual edit from restoring it.
    geometry_topology_preflight = enforce_geometry_export_topology_preflight(graph)
    text_export_purity_preflight = enforce_text_export_purity_preflight(
        graph, source_rgb
    )
    print("  V5 PRO bước 8/10: tổng hợp nền sạch và kiểm tra bóng ma/đường nối...", flush=True)
    # Re-emit the checkpoint from the reconciled graph so the next standalone
    # review sees the exact same nodes/masks as PSD, ORA and the inventory.
    checkpoint_path = prepare_review_bundle(
        graph,
        source_rgb,
        technical_dir,
        source_sha256=sha256_file(source_path),
        icc_profile=output_icc,
        proposal_masks=_proposal_masks(inventory),  # type: ignore[arg-type]
    )
    if review_document is not None:
        _preserve_review_metadata(checkpoint_path, review_document)

    clean = build_clean_plate(
        source_rgb,
        graph,
        mode=args.inpaint,
        progress=print,
        poster_surface_rgb=residual.bottom_surface_rgb,
    )
    technical_dir.mkdir(parents=True, exist_ok=True)
    ghost_path = technical_dir / "CLEAN_PLATE_GHOST_RISK.png"
    _write_ghost_heatmap(ghost_path, clean.ghost_heatmap, output_icc)
    save_color_png(
        Image.fromarray(residual.bottom_surface_rgb, "RGB"),
        technical_dir / "RESIDUAL_BOTTOM_SURFACE.png",
        output_icc,
    )
    save_color_png(
        Image.fromarray(residual.saliency, "L").convert("RGB"),
        technical_dir / "RESIDUAL_SALIENCY.png",
        output_icc,
    )
    save_color_png(
        Image.fromarray(residual.background_connected_mask, "L").convert("RGB"),
        technical_dir / "RESIDUAL_BACKGROUND_CONNECTED.png",
        output_icc,
    )

    # Source-resolution rendering is the editability authority even when the
    # requested delivery is x4/x10.  It catches hidden ghosting without letting
    # a large AI master conceal a bad original mask.
    qa_render = render_document(graph, source_rgb, clean.background_rgb, scale=1.0)
    qa_background_path = technical_dir / "BACKGROUND_FOR_OCR_QA.png"
    save_color_png(qa_render.background, qa_background_path, output_icc)
    residual_regions, residual_ocr_report = detect_text_regions(qa_background_path)

    print("  V5 PRO bước 9/10: xuất PSD, ORA, PNG layer/mask và preview...", flush=True)
    final_render = (
        qa_render
        if args.scale == 1.0 and np.array_equal(target_rgb, source_rgb)
        else render_document(graph, target_rgb, clean.background_rgb, scale=args.scale)
    )
    exported = export_document(output_dir, name, final_render, icc_profile=output_icc)
    inventory_paths = write_inventory_files(graph, output_dir)
    print("  V5 PRO bước 10/10: kiểm định ownership, nền sạch và container...", flush=True)
    qa_report = build_quality_report(
        graph,
        source_rgb,
        np.asarray(qa_render.background, dtype=np.uint8),
        np.asarray(qa_render.composite, dtype=np.uint8),
        residual_background_text_regions=len(residual_regions),
        format_reports=exported.format_reports,
        cleanplate_report=clean.report,
        residual_report=residual.report,
        delivery_render_report=final_render.report,
    )
    qa_report["background_ghost"]["ocr_detector"] = residual_ocr_report
    qa_report["render_source_resolution"] = qa_render.report
    qa_report["render_delivery"] = final_render.report
    qa_paths = write_qa_artifacts(
        output_dir,
        graph,
        source_rgb,
        np.asarray(qa_render.background, dtype=np.uint8),
        np.asarray(qa_render.composite, dtype=np.uint8),
        qa_report,
        icc_profile=output_icc,
    )
    guide_path = _write_user_guide(output_dir, name, str(qa_report["status"]))

    geometry.report["export_topology_preflight"] = geometry_topology_preflight
    portable_manifest = {
        "schema": ENGINE_SCHEMA,
        "pipeline": PIPELINE_NAME,
        "engine_generation": "V5_PRO_EXHAUSTIVE_V2",
        "review_resume_signature": REVIEW_RESUME_SIGNATURE,
        "app_version": args.app_version,
        "source_name": source_path.name,
        "source_sha256": sha256_file(source_path),
        "source_pixel_sha256": _sha256_rgb(source_rgb),
        "source_size": [source_width, source_height],
        "scale": args.scale,
        "final_size": list(final_size),
        "device": device,
        "detail": args.detail,
        "review_mode": args.review_mode,
        "review_checkpoint": "_KY_THUAT/LAYER_REVIEW.json",
        "qa_status": qa_report["status"],
        "grouping": {
            "selected_layer_count": len(graph.nodes),
            "unresolved_node_count": len(graph.unresolved_nodes()),
            "proposal_accounting": graph.proposal_accounting(),
            "hard_layer_cap": None,
        },
        "inventory": inventory.detector_reports,
        "semantic": semantic_report,
        "layerd": layerd.report,
        "fusion": fusion.report,
        "geometry": geometry.report,
        "geometry_export_topology_preflight": geometry_topology_preflight,
        "text_export_purity_preflight": text_export_purity_preflight,
        "residual_reconciliation": residual.report,
        "base_source_residual": graph.metadata.get("base_source_residual"),
        "ownership": ownership_report,
        "cleanplate": clean.report,
        "hierarchy": graph.metadata.get("hierarchy"),
        "master_policy": master_policy,
        "formats": exported.format_reports,
        "render": final_render.report,
        "qa": {
            "status": qa_report["status"],
            "report": "_KY_THUAT/QA_REPORT.json",
            "report_html": "_KY_THUAT/QA_REPORT.html",
        },
        "truth_notice": (
            "Hidden pixels are synthesized. A PASS describes proposal accounting, mask ownership, "
            "background residual checks and container round-trip; it does not claim recovery of the original source file."
        ),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "engine_seconds": round(time.perf_counter() - started, 3),
    }
    portable_path = technical_dir / "PORTABLE_MANIFEST.json"
    write_json(portable_path, portable_manifest)
    zip_path = output_dir / f"05_{name}_LAYERS.zip"
    included = [
        guide_path,
        *inventory_paths.values(),
        *qa_paths.values(),
        portable_path,
        *sorted((output_dir / "LAYERS").glob("*.png")),
        *sorted((output_dir / "MASKS").glob("*.png")),
    ]
    zip_report = package_layers_zip(zip_path, output_dir, included)
    manifest = dict(portable_manifest)
    manifest["formats"] = {**exported.format_reports, "layers_zip": zip_report}
    manifest_path = output_dir / "manifest.json"
    manifest["assets"] = asset_records(output_dir, exclude={manifest_path})
    write_json(manifest_path, manifest)

    print("\nHOÀN TẤT V5 PRO")
    print(f"  Trạng thái : {qa_report['status']}")
    print(f"  PSD        : {exported.paths['psd'] if exported.paths['psd'].is_file() else 'không tạo do giới hạn PSD'}")
    print(f"  ORA        : {exported.paths['ora']}")
    print(f"  Preview    : {exported.paths['preview']}")
    print(f"  QA dễ đọc : {qa_paths['report_html']}")
    _print_qa_hard_failures(qa_report)
    if qa_report["status"] != "PASS":
        print("  Lưu ý      : mở QA_REPORT.html và checkpoint duyệt trước khi giao bán.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
