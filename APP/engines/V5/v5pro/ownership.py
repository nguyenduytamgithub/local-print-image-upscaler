from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .schema import AlphaCrop, Box, DocumentGraph, ElementNode


_IMMUTABLE_KINDS = {"text", "price", "qr"}
_SEMANTIC_KINDS = {
    "product",
    "logo",
    "icon",
    "badge",
    "ribbon",
    "decoration",
}
_GEOMETRY_KINDS = {"frame", "line"}


def _sources(node: ElementNode) -> tuple[str, ...]:
    return tuple(
        str(item.get("source", "")).strip().lower()
        for item in node.evidence
        if isinstance(item, dict)
    )


def _is_raw_layerd(node: ElementNode) -> bool:
    sources = _sources(node)
    return any(source.startswith("layerd") for source in sources) or (
        node.element_id.startswith("ELEMENT_")
        and (
            "source_iteration" in node.metadata
            or "member_proposal_ids" in node.metadata
        )
    )


def _is_unvalidated_residual_geometry(node: ElementNode) -> bool:
    if node.kind not in {"panel", "frame", "line"}:
        return False
    policy = node.metadata.get("geometry_cleanliness")
    if not isinstance(policy, dict) or str(policy.get("status", "")).casefold() == "pass":
        return False
    return any(source.startswith("poster_surface_residual") for source in _sources(node))


def _is_verified_semantic(node: ElementNode) -> bool:
    if node.kind not in _SEMANTIC_KINDS:
        return False
    sources = _sources(node)
    model_evidence = any(
        source.startswith(("grounding_dino", "sam", "birefnet", "semantic"))
        for source in sources
    )
    return bool(
        model_evidence
        or node.element_id.startswith("SEMANTIC_")
        or "semantic_extraction" in node.metadata
        or node.metadata.get("semantic_verified") is True
        or (
            node.review_status == "user_confirmed"
            and not _is_raw_layerd(node)
        )
    )


def _is_verified_geometry(node: ElementNode) -> bool:
    if node.kind not in _GEOMETRY_KINDS or _is_raw_layerd(node):
        return False
    sources = _sources(node)
    return bool(
        "geometry_backend" in node.metadata
        or node.metadata.get("surface_reconciliation") is True
        or any(
            source.startswith(
                (
                    "opencv_",
                    "poster_geometry",
                    "poster_surface_residual",
                    "hough",
                )
            )
            for source in sources
        )
        # Hand-authored/review-confirmed frame and line nodes are geometry too.
        or node.review_status == "user_confirmed"
    )


def _is_user_authored_node(node: ElementNode) -> bool:
    """Recognise a local review node whose pixels the user explicitly kept."""

    return bool(
        node.review_status == "user_confirmed"
        and node.element_id.startswith("USER_")
    )


def _priority(node: ElementNode) -> tuple[int, int, str]:
    """Return categorical ownership priority; z only breaks equal-class ties."""

    if node.kind in _IMMUTABLE_KINDS:
        band = 600
    elif _is_user_authored_node(node):
        # A hand-drawn review rectangle is explicit user evidence and uses
        # observed source RGB, so it may safely trim the technical remainder.
        band = 475
    elif _is_verified_semantic(node):
        band = 500
    elif _is_verified_geometry(node):
        band = 400
    elif node.metadata.get("role") == "exact_source_remainder_above_clean_base":
        # The technical layer is seeded over clean-geometry carve holes so it
        # can absorb raw LayerD, unvalidated residual guesses *and another
        # verified geometry surface crossing the carved hole*.  It remains
        # below semantic/text owners, which keep real recognised children.
        # Hiding this top source remainder then reveals every synthesized
        # geometry surface without baking observed contamination into it.
        band = 450
    elif node.kind == "panel" and not _is_raw_layerd(node):
        band = 300
    elif _is_raw_layerd(node):
        band = 100
    elif node.kind in _GEOMETRY_KINDS:
        # An unverified geometry-looking residual must not beat a verified
        # panel merely because a raw classifier guessed "frame".
        band = 180
    else:
        band = 200
    return band, int(node.z_index), node.element_id


def ownership_priority(node: ElementNode) -> tuple[int, int, str]:
    """Public stack/ownership order; lower bands render below higher bands."""

    return _priority(node)


def _union_box(first: Box, second: Box) -> Box:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _place_alpha(crop: AlphaCrop, bbox: Box) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    result = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    ox = crop.left - x0
    oy = crop.top - y0
    result[oy : oy + crop.height, ox : ox + crop.width] = crop.alpha
    return result


def _place_rgb(
    node: ElementNode,
    source_rgb: np.ndarray,
    support: AlphaCrop,
    bbox: Box,
) -> tuple[np.ndarray, bool]:
    """Place straight RGB on bbox, preferring the node's synthesized pixels."""

    x0, y0, x1, y1 = bbox
    rgb = source_rgb[y0:y1, x0:x1].copy()
    if node.rgba is None:
        return rgb, True
    rgba_bbox = (support.left, support.top, support.left + node.rgba.shape[1], support.top + node.rgba.shape[0])
    rx0, ry0, rx1, ry1 = rgba_bbox
    ix0, iy0 = max(x0, rx0), max(y0, ry0)
    ix1, iy1 = min(x1, rx1), min(y1, ry1)
    if ix0 < ix1 and iy0 < iy1:
        rgb[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = node.rgba[
            iy0 - ry0 : iy1 - ry0, ix0 - rx0 : ix1 - rx0, :3
        ]
    return rgb, False


def _tight_bbox(alpha: AlphaCrop) -> Box | None:
    ys, xs = np.where(alpha.alpha > 0)
    if not len(xs):
        return None
    return (
        alpha.left + int(xs.min()),
        alpha.top + int(ys.min()),
        alpha.left + int(xs.max()) + 1,
        alpha.top + int(ys.max()) + 1,
    )


def _tight_crop(crop: AlphaCrop, bbox: Box) -> AlphaCrop:
    x0, y0, x1, y1 = bbox
    local_x0 = x0 - crop.left
    local_y0 = y0 - crop.top
    local_x1 = x1 - crop.left
    local_y1 = y1 - crop.top
    return AlphaCrop(
        x0,
        y0,
        np.ascontiguousarray(crop.alpha[local_y0:local_y1, local_x0:local_x1]),
    )


def _append_reason(existing: str | None, addition: str) -> str:
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing}; {addition}"


def enforce_exclusive_ownership(
    graph: DocumentGraph,
    source_rgb: np.ndarray,
) -> dict[str, Any]:
    """Enforce one categorical visible owner per source pixel.

    OCR text, prices, QR codes and verified semantic mattes are immutable.  The
    pipeline already builds those owners against a protected-alpha map; an
    overlap between two immutable owners therefore signals an upstream defect
    and fails closed instead of silently eroding either trusted matte.

    Geometry and panel nodes retain ``full_support`` (and its synthesized RGBA)
    below higher-priority children, while their ``visible_alpha`` receives
    categorical holes.  Raw LayerD residuals are destructive duplicates: both
    visible and export support are trimmed, and an entirely absorbed raw node is
    removed with its proposal ledger explicitly reassigned or rejected.
    """

    image = np.asarray(source_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Ownership source must be a uint8 RGB image")
    height, width = image.shape[:2]
    if graph.canvas_size != (width, height):
        raise ValueError("Ownership graph canvas differs from source image")
    graph.validate()

    owner_before, _ = graph.ownership_maps()
    before_overlap = int(np.count_nonzero(owner_before > 1))
    before_max = int(owner_before.max()) if owner_before.size else 0

    claimed = np.zeros((height, width), dtype=bool)
    claimed_owner = np.full((height, width), -1, dtype=np.int32)
    ordered = sorted(graph.nodes, key=_priority, reverse=True)
    raw_node_ids = {
        node.element_id for node in graph.nodes if _is_raw_layerd(node)
    }
    owner_id_by_index: list[str] = []
    removed_ids: set[str] = set()
    absorbed_by: dict[str, Counter[str]] = {}
    modified_ids: list[str] = []
    immutable_ids: list[str] = []
    transferred_pixels = 0
    synthesized_fallback_ids: list[str] = []
    priority_counts: Counter[str] = Counter()

    for node in ordered:
        band = _priority(node)[0]
        priority_counts[str(band)] += 1
        visible = node.visible_alpha
        ys, xs = visible.canvas_slice(graph.canvas_size)
        original = visible.alpha.copy()
        original_binary = original > 0
        collision = original_binary & claimed[ys, xs]
        immutable = node.kind in _IMMUTABLE_KINDS or _is_verified_semantic(node)
        if immutable:
            immutable_ids.append(node.element_id)
            if np.any(collision):
                conflicting_indices = np.unique(claimed_owner[ys, xs][collision])
                conflicts = [
                    owner_id_by_index[int(index)]
                    for index in conflicting_indices
                    if 0 <= int(index) < len(owner_id_by_index)
                ]
                raise RuntimeError(
                    "Immutable ownership conflict: "
                    f"{node.element_id} overlaps {', '.join(conflicts) or 'another trusted owner'}"
                )
            local_visible = original
        else:
            local_visible = original.copy()
            local_visible[collision] = 0
            lost = int(np.count_nonzero(collision))
            if lost:
                modified_ids.append(node.element_id)
                transferred_pixels += lost
                winners = claimed_owner[ys, xs][collision]
                counter: Counter[str] = Counter()
                for index, count in zip(*np.unique(winners, return_counts=True)):
                    if 0 <= int(index) < len(owner_id_by_index):
                        counter[owner_id_by_index[int(index)]] += int(count)
                absorbed_by[node.element_id] = counter

        destructive_unvalidated = _is_unvalidated_residual_geometry(node)
        preserve_hidden = (
            (node.kind in {"panel", "frame", "line"})
            and not _is_raw_layerd(node)
            and not destructive_unvalidated
        )
        is_raw = _is_raw_layerd(node)
        destructive_candidate = is_raw or destructive_unvalidated
        remaining = int(np.count_nonzero(local_visible))
        # Any ordinary non-immutable leaf that has lost every visible pixel is
        # a duplicate, regardless of which backend produced it.  Keeping a
        # zero-alpha residual node makes the exported node count diverge from
        # the graph and hard-fails delivery QA.  Validated geometry is the sole
        # exception: its clean full support remains useful below the winner.
        if not remaining and not preserve_hidden:
            removed_ids.add(node.element_id)
            continue

        if not immutable and (np.any(collision) or destructive_candidate):
            if preserve_hidden:
                support = node.full_support or node.visible_alpha
                bbox = _union_box(support.bbox, node.visible_alpha.bbox)
                full_alpha = _place_alpha(support, bbox)
                visible_alpha = np.zeros_like(full_alpha)
                vx = node.visible_alpha.left - bbox[0]
                vy = node.visible_alpha.top - bbox[1]
                visible_alpha[
                    vy : vy + node.visible_alpha.height,
                    vx : vx + node.visible_alpha.width,
                ] = local_visible
                rgb, used_fallback = _place_rgb(node, image, support, bbox)
                if used_fallback and np.any(full_alpha > 0):
                    # Never bake a higher-priority child into a missing hidden
                    # surface.  A robust flat estimate is safer for poster
                    # panels/rules and is explicitly reported for review.
                    observed = rgb[(visible_alpha > 0) & (full_alpha > 0)]
                    if len(observed):
                        colour = np.median(observed, axis=0).astype(np.uint8)
                        rgb[(full_alpha > 0) & (visible_alpha == 0)] = colour
                    synthesized_fallback_ids.append(node.element_id)
                node.visible_alpha = AlphaCrop(bbox[0], bbox[1], visible_alpha)
                node.full_support = AlphaCrop(bbox[0], bbox[1], full_alpha)
                node.rgba = np.dstack([rgb, full_alpha])
                node.occluded = bool(np.any((full_alpha > 0) & (visible_alpha == 0)))
                node.synthesized_hidden_pixels = node.occluded
                node.metadata["exclusive_ownership"] = {
                    "policy": "visible holes; synthesized full support retained",
                    "transferred_pixel_count": int(np.count_nonzero(collision)),
                }
            else:
                current = AlphaCrop(visible.left, visible.top, local_visible)
                tight = _tight_bbox(current)
                if tight is None:
                    # Only non-raw empty nodes reach this branch. Keep a valid
                    # zero crop so the reviewer can decide whether to reject it.
                    tight = current.bbox
                new_visible = _tight_crop(current, tight)
                rgb = image[tight[1] : tight[3], tight[0] : tight[2]].copy()
                if node.rgba is not None:
                    support = node.full_support or visible
                    placed, _fallback = _place_rgb(node, image, support, tight)
                    rgb = placed
                node.visible_alpha = new_visible
                node.full_support = new_visible
                node.rgba = np.dstack([rgb, new_visible.alpha.copy()])
                node.removal_footprint = new_visible
                node.occluded = False
                node.synthesized_hidden_pixels = False
                node.metadata["exclusive_ownership"] = {
                    "policy": "destructive duplicate trim",
                    "transferred_pixel_count": int(np.count_nonzero(collision)),
                }

        current_visible = node.visible_alpha
        cys, cxs = current_visible.canvas_slice(graph.canvas_size)
        binary = current_visible.alpha > 0
        owner_index = len(owner_id_by_index)
        claimed[cys, cxs] |= binary
        claimed_owner[cys, cxs][binary] = owner_index
        owner_id_by_index.append(node.element_id)

    parent_by_id = {node.element_id: node.parent_id for node in graph.nodes}
    reparented_node_count = 0
    if removed_ids:
        graph.nodes = [node for node in graph.nodes if node.element_id not in removed_ids]
        surviving_after_removal = {node.element_id for node in graph.nodes}

        def surviving_parent(parent_id: str | None) -> str | None:
            current = parent_id
            seen: set[str] = set()
            while current in removed_ids and current not in seen:
                seen.add(current)
                current = parent_by_id.get(current)
            return current if current in surviving_after_removal else None

        for node in graph.nodes:
            if node.parent_id in removed_ids:
                node.parent_id = surviving_parent(node.parent_id)
                reparented_node_count += 1

    reassigned = 0
    rejected = 0
    repaired_existing = 0
    surviving_ids = set(graph.node_map())
    for proposal in graph.proposals:
        original_owners = list(proposal.owner_ids)
        replacement: list[str] = []
        duplicate_notes: list[str] = []
        for owner_id in original_owners:
            if owner_id in surviving_ids:
                replacement.append(owner_id)
                continue
            if owner_id not in removed_ids:
                continue
            winners = [
                item[0]
                for item in absorbed_by.get(owner_id, Counter()).most_common()
                if item[0] in surviving_ids
            ]
            if winners:
                replacement.extend(winners)
                duplicate_notes.append(
                    f"owner {owner_id} absorbed by {', '.join(winners)}"
                )
            else:
                duplicate_notes.append(f"empty owner {owner_id} removed")
        replacement = list(dict.fromkeys(replacement))
        if replacement != original_owners:
            proposal.owner_ids = replacement
            if replacement:
                proposal.status = "assigned"
                proposal.reason = _append_reason(
                    proposal.reason,
                    "Ownership arbitration: " + "; ".join(duplicate_notes),
                )
                proposal.evidence["ownership_arbitration"] = {
                    "previous_owner_ids": original_owners,
                    "final_owner_ids": replacement,
                    "duplicate_resolution": duplicate_notes,
                }
                reassigned += 1
            else:
                proposal.status = "rejected"
                proposal.reason = _append_reason(
                    proposal.reason,
                    "Ownership arbitration rejected an empty/duplicate owner",
                )
                proposal.evidence["ownership_arbitration"] = {
                    "previous_owner_ids": original_owners,
                    "final_owner_ids": [],
                    "duplicate_resolution": duplicate_notes,
                }
                rejected += 1
        elif proposal.status == "assigned" and any(
            owner not in surviving_ids for owner in proposal.owner_ids
        ):
            # Defensive fail-closed repair for a malformed pre-existing ledger.
            proposal.owner_ids = [owner for owner in proposal.owner_ids if owner in surviving_ids]
            if proposal.owner_ids:
                repaired_existing += 1
            else:
                proposal.status = "rejected"
                proposal.reason = _append_reason(
                    proposal.reason,
                    "Ownership arbitration removed dangling proposal owners",
                )
                rejected += 1

    graph.metadata["exclusive_ownership"] = {
        "policy": (
            "text/price/QR > verified semantic > user-authored review nodes > "
            "technical clean-geometry source reservation > verified frame/line > "
            "panel > raw LayerD; "
            "visible ownership is categorical"
        ),
        "removed_raw_node_ids": sorted(removed_ids & raw_node_ids),
        "removed_empty_node_ids": sorted(removed_ids),
    }
    graph.validate()
    owner_after, _ = graph.ownership_maps()
    after_overlap = int(np.count_nonzero(owner_after > 1))
    after_max = int(owner_after.max()) if owner_after.size else 0
    if after_overlap or after_max > 1:
        raise RuntimeError("Ownership arbitration failed to produce exclusive visible mattes")

    return {
        "backend": "categorical priority ownership v2",
        "node_count_before": len(ordered),
        "node_count_after": len(graph.nodes),
        "overlap_pixels_before": before_overlap,
        "overlap_pixels_after": after_overlap,
        "max_owner_count_before": before_max,
        "max_owner_count_after": after_max,
        "modified_node_count": len(set(modified_ids)),
        "modified_node_ids": sorted(set(modified_ids)),
        "removed_raw_node_count": len(removed_ids & raw_node_ids),
        "removed_raw_node_ids": sorted(removed_ids & raw_node_ids),
        "removed_empty_node_count": len(removed_ids),
        "removed_empty_node_ids": sorted(removed_ids),
        "transferred_visible_pixel_count": transferred_pixels,
        "immutable_node_count": len(immutable_ids),
        "proposal_reassigned_count": reassigned,
        "proposal_rejected_count": rejected,
        "reparented_node_count": reparented_node_count,
        "preexisting_dangling_owner_repairs": repaired_existing,
        "synthesized_rgb_fallback_node_ids": sorted(set(synthesized_fallback_ids)),
        "priority_band_node_counts": dict(sorted(priority_counts.items(), reverse=True)),
        "assigned_dangling_owner_count": 0,
    }


__all__ = ["enforce_exclusive_ownership", "ownership_priority"]
