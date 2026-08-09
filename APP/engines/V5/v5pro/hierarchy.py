from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np

from .ownership import ownership_priority
from .schema import Box, DocumentGraph, ElementNode


AUTO_ORGANIZATION_SOURCE = "auto_organization_v1"

_ORGANIZATION_BUCKETS: dict[str, tuple[str, str]] = {
    "raw_layerd": ("B100", "Raw LayerD details / Chi tiet tho"),
    "residual": ("B200", "Residual details / Chi tiet con lai"),
    "layout": ("B300", "Layout surfaces / Mang va bang"),
    "geometry": ("B400", "Validated geometry / Hinh hoc da kiem"),
    "remainder": ("B450", "Technical source remainder / Phan anh goc"),
    "user": ("B475", "User-created details / Chi tiet nguoi dung"),
    "semantic": ("B500", "Semantic objects / Vat the"),
    "text": ("B600", "Text, prices and QR / Chu, gia va QR"),
}


def _area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(first: Box, second: Box) -> int:
    return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _alpha_overlap(first, second) -> int:
    x0 = max(first.left, second.left)
    y0 = max(first.top, second.top)
    x1 = min(first.bbox[2], second.bbox[2])
    y1 = min(first.bbox[3], second.bbox[3])
    if x0 >= x1 or y0 >= y1:
        return 0
    first_view = first.alpha[
        y0 - first.top : y1 - first.top,
        x0 - first.left : x1 - first.left,
    ]
    second_view = second.alpha[
        y0 - second.top : y1 - second.top,
        x0 - second.left : x1 - second.left,
    ]
    return int(np.count_nonzero((first_view > 0) & (second_view > 0)))


def _clean_reference_geometry(node: ElementNode) -> bool:
    policy = node.metadata.get("geometry_cleanliness")
    return bool(
        node.review_status != "rejected"
        and node.kind in {"panel", "frame", "line"}
        and node.full_support is not None
        and isinstance(policy, dict)
        and policy.get("policy") == "reference_surface_delta_e_carve_v1"
        and policy.get("reference_type")
        in {"surface_rgb", "constant_colour_rgb"}
    )


def _would_create_cycle(
    node: ElementNode,
    parent: ElementNode,
    by_id: dict[str, ElementNode],
) -> bool:
    current: ElementNode | None = parent
    seen: set[str] = set()
    while current is not None and current.element_id not in seen:
        if current.element_id == node.element_id:
            return True
        seen.add(current.element_id)
        current = by_id.get(current.parent_id) if current.parent_id else None
    return False


def _container_stack_indices(graph: DocumentGraph) -> dict[str, int]:
    """Return the actual preorder used by editable container layer groups."""

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
    ordered: list[ElementNode] = []

    def emit(node: ElementNode) -> None:
        ordered.append(node)
        for child in children.get(node.element_id, []):
            emit(child)

    for root in roots:
        emit(root)
    return {node.element_id: index for index, node in enumerate(ordered)}


def _organization_bucket(node: ElementNode) -> str:
    """Return a broad, auditable folder class without changing stack order."""

    band = ownership_priority(node)[0]
    if band >= 600:
        return "text"
    if band >= 500:
        return "semantic"
    if band >= 475:
        return "user"
    if band >= 450:
        return "remainder"
    if band >= 400:
        return "geometry"
    if band >= 300:
        return "layout"
    if band <= 100:
        return "raw_layerd"
    return "residual"


def _active_sibling_map(
    graph: DocumentGraph,
) -> tuple[list[ElementNode], dict[str, list[ElementNode]]]:
    active = [node for node in graph.nodes if node.review_status != "rejected"]
    by_id = {node.element_id: node for node in active}
    roots: list[ElementNode] = []
    children: dict[str, list[ElementNode]] = {}
    for node in active:
        if node.parent_id and node.parent_id in by_id:
            children.setdefault(node.parent_id, []).append(node)
        else:
            roots.append(node)
    roots.sort(key=ownership_priority)
    for siblings in children.values():
        siblings.sort(key=ownership_priority)
    return roots, children


def _auto_group_id(
    parent_id: str | None,
    bucket: str,
    run_number: int,
    used_ids: set[str],
) -> str:
    parent_token = (
        "ROOT"
        if parent_id is None
        else hashlib.sha256(parent_id.encode("utf-8")).hexdigest()[:12].upper()
    )
    bucket_token = _ORGANIZATION_BUCKETS[bucket][0]
    base = f"AUTO_ORG_{parent_token}_{bucket_token}_{run_number:03d}"
    identifier = base
    collision = 1
    while identifier in used_ids:
        identifier = f"{base}_{collision}"
        collision += 1
    used_ids.add(identifier)
    return identifier


def assign_organizational_groups(graph: DocumentGraph) -> dict[str, Any]:
    """Create pass-through folders while preserving every atomic leaf and pixel.

    Groups are made only from maximal contiguous sibling runs in the exact
    bottom-to-top ownership order used by the renderer. Existing review-created
    groups are authority: valid records are copied byte-for-byte and their
    members are excluded from automatic folders. An old group made impossible
    by a later hierarchy change is dissolved with an audit reason; its leaves
    stay ungrouped and are never silently claimed by an automatic folder.
    """

    review = graph.metadata.get("review")
    if not isinstance(review, dict):
        review = {}
        graph.metadata["review"] = review
    raw_groups = review.get("groups")
    if not isinstance(raw_groups, list):
        raw_groups = []

    roots, children = _active_sibling_map(graph)
    siblings_by_parent: dict[str | None, list[ElementNode]] = {None: roots, **children}
    active_by_id = {
        node.element_id: node
        for node in graph.nodes
        if node.review_status != "rejected"
    }
    explicit_groups: list[dict[str, object]] = []
    explicit_members: set[str] = set()
    dissolved_explicit_groups: list[dict[str, object]] = []
    for raw_group in raw_groups:
        if (
            not isinstance(raw_group, dict)
            or raw_group.get("source") == AUTO_ORGANIZATION_SOURCE
        ):
            continue
        raw_members = raw_group.get("member_ids")
        members = (
            list(raw_members)
            if isinstance(raw_members, list)
            and all(isinstance(member, str) for member in raw_members)
            else []
        )
        active_members = [member for member in members if member in active_by_id]
        reason: str | None = None
        if len(members) < 2 or len(members) != len(set(members)):
            reason = "invalid_or_duplicate_member_list"
        elif len(active_members) != len(members):
            reason = "member_missing_or_rejected"
        elif explicit_members.intersection(members):
            reason = "member_already_claimed_by_an_explicit_group"
        else:
            parents = {
                (
                    active_by_id[member].parent_id
                    if active_by_id[member].parent_id in active_by_id
                    else None
                )
                for member in members
            }
            if len(parents) != 1:
                reason = "members_no_longer_share_one_parent"
            else:
                parent_id = next(iter(parents))
                siblings = siblings_by_parent.get(parent_id, [])
                positions = sorted(
                    next(
                        index
                        for index, node in enumerate(siblings)
                        if node.element_id == member
                    )
                    for member in members
                )
                if positions != list(range(positions[0], positions[-1] + 1)):
                    reason = "members_no_longer_contiguous_in_final_stack"
        # Even an impossible old folder remains user authority for automatic
        # organization: dissolve only the unsafe container, leave its atomic
        # members ungrouped instead of silently putting them elsewhere.
        explicit_members.update(active_members)
        if reason is None:
            explicit_groups.append(copy.deepcopy(raw_group))
        else:
            dissolved_explicit_groups.append(
                {
                    "id": str(raw_group.get("id") or ""),
                    "name": str(raw_group.get("name") or ""),
                    "member_ids": active_members,
                    "reason": reason,
                }
            )
    used_ids = {
        str(group.get("id"))
        for group in raw_groups
        if isinstance(group, dict)
        if str(group.get("id") or "").strip()
    }

    sibling_sets: list[tuple[str | None, list[ElementNode]]] = [(None, roots)]
    sibling_sets.extend(
        (parent_id, children[parent_id]) for parent_id in sorted(children)
    )
    auto_groups: list[dict[str, object]] = []
    run_counts: dict[tuple[str | None, str], int] = {}

    def flush(
        parent_id: str | None,
        bucket: str | None,
        members: list[ElementNode],
    ) -> None:
        if bucket is None or len(members) < 2:
            return
        key = (parent_id, bucket)
        run_number = run_counts.get(key, 0) + 1
        run_counts[key] = run_number
        identifier = _auto_group_id(
            parent_id, bucket, run_number, used_ids
        )
        auto_groups.append(
            {
                "id": identifier,
                "name": _ORGANIZATION_BUCKETS[bucket][1],
                "member_ids": [node.element_id for node in members],
                "source": AUTO_ORGANIZATION_SOURCE,
                "organization_bucket": bucket,
                "parent_id": parent_id,
                "order_policy": "same_parent_contiguous_bottom_to_top",
            }
        )

    for parent_id, siblings in sibling_sets:
        run_bucket: str | None = None
        run_members: list[ElementNode] = []
        for node in siblings:
            if node.element_id in explicit_members:
                flush(parent_id, run_bucket, run_members)
                run_bucket, run_members = None, []
                continue
            bucket = _organization_bucket(node)
            if bucket != run_bucket:
                flush(parent_id, run_bucket, run_members)
                run_bucket, run_members = bucket, []
            run_members.append(node)
        flush(parent_id, run_bucket, run_members)

    groups = [*explicit_groups, *auto_groups]
    review["groups"] = groups

    root_ids = {node.element_id for node in roots}
    savings = sum(len(group["member_ids"]) - 1 for group in groups)
    root_groups = [
        group
        for group in groups
        if group.get("member_ids")
        and all(str(member) in root_ids for member in group["member_ids"])
    ]
    root_savings = sum(len(group["member_ids"]) - 1 for group in root_groups)
    report: dict[str, Any] = {
        "policy": (
            "pass-through folders over maximal same-parent contiguous runs; "
            "atomic leaves, hierarchy, masks and bottom-to-top order are unchanged"
        ),
        "source": AUTO_ORGANIZATION_SOURCE,
        "explicit_group_count": len(explicit_groups),
        "dissolved_explicit_group_count": len(dissolved_explicit_groups),
        "dissolved_explicit_groups": dissolved_explicit_groups,
        "auto_group_count": len(auto_groups),
        "auto_grouped_member_count": sum(
            len(group["member_ids"]) for group in auto_groups
        ),
        "sibling_entries_before": len(
            [node for node in graph.nodes if node.review_status != "rejected"]
        ),
        "sibling_entries_after": len(
            [node for node in graph.nodes if node.review_status != "rejected"]
        )
        - savings,
        "root_entries_before": len(roots),
        "root_entries_after": len(roots) - root_savings,
        "bucket_group_counts": {
            bucket: sum(
                group.get("organization_bucket") == bucket for group in auto_groups
            )
            for bucket in _ORGANIZATION_BUCKETS
        },
    }
    graph.metadata["organization"] = report
    return report


def assign_spatial_hierarchy(graph: DocumentGraph) -> dict[str, Any]:
    """Place atomic leaves below the smallest credible enclosing layout node.

    A hierarchy changes organization only.  It never unions masks, changes
    ownership or merges two editable leaves.
    """

    # Rebuild inferred containment deterministically. Review groups are stored
    # separately by the UI; carrying an old inferred parent into a new graph
    # can otherwise invert the actual bottom-to-top render order.
    for node in graph.nodes:
        node.parent_id = None

    parents = [
        node
        for node in graph.nodes
        if node.kind in {"panel", "frame"}
        and node.review_status != "rejected"
        and _area(node.bbox) >= 64
    ]
    parents.sort(key=lambda item: (_area(item.bbox), item.z_index, item.element_id))
    assignments: list[dict[str, object]] = []

    # A synthesized card surface is the lower layer and its paired frame is
    # the upper child. Pure bbox containment often chooses the reverse because
    # the panel is slightly inset from the outer stroke.
    by_id = graph.node_map()
    paired_frames: set[str] = set()
    for frame in graph.nodes:
        paired_panel_id = frame.metadata.get("paired_panel_id")
        if frame.kind != "frame" or not isinstance(paired_panel_id, str):
            continue
        panel = by_id.get(paired_panel_id)
        if panel is None or panel.kind != "panel" or panel.z_index >= frame.z_index:
            continue
        frame.parent_id = panel.element_id
        paired_frames.add(frame.element_id)
        assignments.append({"child": frame.element_id, "parent": panel.element_id})

    for node in graph.nodes:
        if node.review_status == "rejected" or node.kind == "background":
            continue
        if node.element_id in paired_frames:
            continue
        candidates: list[ElementNode] = []
        node_area = max(1, _area(node.bbox))
        for parent in parents:
            if parent.element_id == node.element_id or _area(parent.bbox) <= node_area:
                continue
            # Parent layers are rendered first. Never let an enclosing layer
            # with a higher z-index become a parent and overwrite its child.
            if parent.z_index >= node.z_index:
                continue
            coverage = _intersection(node.bbox, parent.bbox) / node_area
            if coverage >= 0.94:
                candidates.append(parent)
        if not candidates:
            continue
        selected = candidates[0]
        # Do not create frame-in-frame chains from nearly identical duplicate
        # proposals; those remain visible in the ledger for review instead.
        if node.kind in {"panel", "frame"} and _area(node.bbox) / _area(selected.bbox) > 0.88:
            continue
        node.parent_id = selected.element_id
        assignments.append({"child": node.element_id, "parent": selected.element_id})

    # Full-support clean geometry is deliberately retained underneath known
    # text/semantic children. A merely spatial parent can place such a child
    # in an earlier root group, after which a later geometry group paints its
    # synthesized full support over the recognised source pixels. Reparent
    # every higher-priority visible owner to the last overlapping clean
    # geometry dependency. This changes stack organization only; no alpha or
    # proposal ownership is mutated.
    clean_geometry = [node for node in graph.nodes if _clean_reference_geometry(node)]
    initial_stack_index = _container_stack_indices(graph)
    stack_repairs: list[dict[str, object]] = []
    for node in graph.nodes:
        if (
            node.review_status == "rejected"
            or ownership_priority(node)[0] <= 400
            or node.metadata.get("role")
            == "exact_source_remainder_above_clean_base"
        ):
            continue
        candidates: list[tuple[int, int, ElementNode]] = []
        for geometry in clean_geometry:
            if geometry.element_id == node.element_id:
                continue
            overlap = _alpha_overlap(
                geometry.full_support or geometry.visible_alpha,
                node.visible_alpha,
            )
            if overlap <= 0 or _would_create_cycle(node, geometry, by_id):
                continue
            candidates.append(
                (initial_stack_index.get(geometry.element_id, -1), overlap, geometry)
            )
        if not candidates:
            continue
        selected_stack_index, overlap, selected = max(
            candidates,
            key=lambda item: (item[0], item[1], item[2].element_id),
        )
        previous_parent = node.parent_id
        if previous_parent == selected.element_id:
            continue
        node.parent_id = selected.element_id
        repair = {
            "child": node.element_id,
            "parent": selected.element_id,
            "previous_parent": previous_parent,
            "overlap_pixels": overlap,
            "selected_geometry_stack_index_before_repair": selected_stack_index,
            "reason": "clean_full_support_must_render_below_visible_owner",
        }
        assignments.append(repair)
        stack_repairs.append(repair)
    graph.metadata["hierarchy"] = {
        "policy": (
            "paired panel below frame, otherwise smallest lower-z enclosing "
            "panel/frame at >=94% bbox coverage; higher-priority visible owners "
            "are then attached above the last overlapping clean full-support "
            "geometry; masks remain atomic"
        ),
        "assignment_count": len(assignments),
        "assignments": assignments,
        "stack_dependency_repair_count": len(stack_repairs),
        "stack_dependency_repairs": stack_repairs,
    }
    graph.validate()
    graph.metadata["hierarchy"]["organization"] = assign_organizational_groups(
        graph
    )
    return graph.metadata["hierarchy"]
