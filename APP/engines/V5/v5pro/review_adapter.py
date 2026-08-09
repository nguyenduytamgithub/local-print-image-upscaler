from __future__ import annotations

import copy
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from PIL import Image

from v5lib.formats import save_color_png, sha256_file

from .review_server import build_review_checkpoint, write_review_checkpoint
from .schema import AlphaCrop, DocumentGraph, ElementNode, ProposalRecord, clip_box


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe review asset path: {value!r}")
    return path


def prepare_review_bundle(
    graph: DocumentGraph,
    source_rgb: np.ndarray,
    technical_dir: Path,
    *,
    source_sha256: str,
    icc_profile: bytes | None,
    proposal_masks: Mapping[str, AlphaCrop] | None = None,
) -> Path:
    technical_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = technical_dir / "REVIEW_MASKS"
    masks_dir.mkdir(parents=True, exist_ok=True)
    source_path = technical_dir / "SOURCE_FOR_REVIEW.png"
    save_color_png(Image.fromarray(source_rgb, "RGB"), source_path, icc_profile)
    mask_paths: dict[str, str] = {}
    for node in graph.nodes:
        mask_path = masks_dir / f"{node.element_id}.png"
        Image.fromarray(node.visible_alpha.alpha, "L").save(mask_path, format="PNG", compress_level=4)
        mask_paths[node.element_id] = f"REVIEW_MASKS/{mask_path.name}"
    checkpoint = build_review_checkpoint(
        graph,
        source=source_path.name,
        # ReviewSession validates the portable asset it actually serves, not
        # the original container bytes (which can change after PNG/ICC
        # normalization even when pixels are identical).
        source_sha256=sha256_file(source_path),
        mask_paths=mask_paths,
    )
    checkpoint["original_source_sha256"] = source_sha256
    by_proposal = {
        str(record.get("id")): record
        for record in checkpoint.get("proposals", [])
        if isinstance(record, dict)
    }
    for proposal_id, crop in (proposal_masks or {}).items():
        record = by_proposal.get(proposal_id)
        if record is None:
            continue
        mask_path = masks_dir / f"PROPOSAL_{proposal_id}.png"
        Image.fromarray(crop.alpha, "L").save(mask_path, format="PNG", compress_level=4)
        record["mask"] = f"REVIEW_MASKS/{mask_path.name}"
    checkpoint_path = technical_dir / "LAYER_REVIEW.json"
    write_review_checkpoint(checkpoint_path, checkpoint)
    return checkpoint_path


def _read_mask(root: Path, record: Mapping[str, object], canvas: tuple[int, int]) -> AlphaCrop:
    bbox_raw = record.get("bbox")
    if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
        raise ValueError(f"Review node {record.get('id')} has invalid bbox.")
    bbox = clip_box(tuple(int(value) for value in bbox_raw), canvas)  # type: ignore[arg-type]
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Review node {record.get('id')} has empty bbox.")
    reference = record.get("mask")
    if reference is None:
        return AlphaCrop(x0, y0, np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8))
    relative = _safe_relative(str(reference))
    path = (root / Path(*relative.parts)).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Review mask is missing or escapes bundle: {reference}")
    with Image.open(path) as opened:
        alpha = np.asarray(opened.convert("L"), dtype=np.uint8).copy()
    if alpha.shape == (canvas[1], canvas[0]):
        alpha = alpha[y0:y1, x0:x1].copy()
    elif alpha.shape != (y1 - y0, x1 - x0):
        raise ValueError(f"Review mask size mismatch for {record.get('id')}.")
    return AlphaCrop(x0, y0, alpha).tight()


def _same_alpha_crop(left: AlphaCrop, right: AlphaCrop) -> bool:
    """Return whether the user left the original visible matte unchanged."""

    return left.bbox == right.bbox and np.array_equal(left.alpha, right.alpha)


def apply_review_checkpoint(
    original: DocumentGraph,
    checkpoint_path: Path,
    source_rgb: np.ndarray,
) -> DocumentGraph:
    document = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
    if document.get("schema") != "V5_LAYER_REVIEW_V2":
        raise ValueError("Review checkpoint schema is not V5_LAYER_REVIEW_V2.")
    canvas_raw = document.get("canvas")
    if canvas_raw != list(original.canvas_size):
        raise ValueError("Review checkpoint canvas differs from the source graph.")
    root = checkpoint_path.parent.resolve()
    old_nodes = original.node_map()
    graph = DocumentGraph(original.canvas_size, metadata=copy.deepcopy(original.metadata))
    records = document.get("nodes")
    if not isinstance(records, list):
        raise ValueError("Review checkpoint has no node list.")
    height, width = source_rgb.shape[:2]
    if (width, height) != original.canvas_size:
        raise ValueError("Source RGB differs from the review canvas.")
    for index, raw in enumerate(records, 1):
        if not isinstance(raw, dict):
            raise ValueError("Review node record is not an object.")
        node_id = str(raw.get("id") or "")
        crop = _read_mask(root, raw, graph.canvas_size)
        previous = old_nodes.get(node_id)
        inherited = previous
        if inherited is None:
            raw_metadata = raw.get("metadata")
            if isinstance(raw_metadata, dict):
                lineage_id = raw_metadata.get("lineage_root_id")
                if not isinstance(lineage_id, str) or not lineage_id:
                    lineage_id = raw_metadata.get("derived_from_node_id")
                candidate = old_nodes.get(str(lineage_id)) if lineage_id else None
                if candidate is not None:
                    candidate_support = candidate.full_support or candidate.visible_alpha
                    candidate_canvas = candidate_support.to_canvas(original.canvas_size) > 0
                    crop_canvas = crop.to_canvas(original.canvas_size) > 0
                    # A derived review mask may inherit provenance only while
                    # it remains a true subset of its original authority. This
                    # preserves SOURCE REMAINDER split children without letting
                    # arbitrary checkpoint metadata exempt new pixels from the
                    # clean-plate removal audit.
                    if not np.any(crop_canvas & ~candidate_canvas):
                        inherited = candidate
        matte_unchanged = previous is not None and _same_alpha_crop(
            previous.visible_alpha, crop
        )
        if matte_unchanged:
            # Renaming, regrouping or confirming a node must not destroy a
            # geometry layer's reconstructed support below an occluding child.
            semantic_envelope = previous.semantic_envelope
            full_support = previous.full_support
            removal_footprint = previous.removal_footprint
            rgba = previous.rgba.copy() if previous.rgba is not None else None
        else:
            x0, y0, x1, y1 = crop.bbox
            semantic_envelope = crop
            full_support = crop
            removal_footprint = crop
            rgba = np.dstack([source_rgb[y0:y1, x0:x1].copy(), crop.alpha])
        node = ElementNode(
            element_id=node_id,
            name=str(raw.get("name") or node_id),
            kind=str(raw.get("kind") or "unknown"),  # type: ignore[arg-type]
            visible_alpha=crop,
            z_index=int(raw.get("z_index", previous.z_index if previous else 3_000_000 + index)),
            parent_id=(
                str(raw["parent_id"])
                if raw.get("parent_id") and str(raw["parent_id"]) != node_id
                else None
            ),
            semantic_envelope=semantic_envelope,
            full_support=full_support,
            removal_footprint=removal_footprint,
            confidence=float(raw.get("confidence", previous.confidence if previous else 0.5)),
            review_status=str(raw.get("review_status", raw.get("status", "unresolved"))),  # type: ignore[arg-type]
            move_safe=bool(raw.get("move_safe", previous.move_safe if previous else False)),
            occluded=bool(raw.get("occluded", previous.occluded if previous else False)),
            synthesized_hidden_pixels=bool(
                raw.get(
                    "synthesized_hidden_pixels",
                    previous.synthesized_hidden_pixels if previous else False,
                )
            ),
            text=(
                str(raw["text"])
                if raw.get("text") is not None
                else (previous.text if previous else None)
            ),
            rgba=rgba,
            evidence=copy.deepcopy(inherited.evidence if inherited else []),
            metadata=copy.deepcopy(inherited.metadata if inherited else {}),
        )
        raw_metadata = raw.get("metadata")
        if (
            isinstance(raw_metadata, dict)
            and raw_metadata.get("promoted_from_technical_remainder") is True
            and str(raw.get("kind") or "unknown") != "unknown"
            and node.metadata.get("role")
            == "exact_source_remainder_above_clean_base"
        ):
            node.metadata.pop("role", None)
            node.metadata["promoted_from_technical_remainder"] = True
        if inherited is not None and previous is None:
            node.metadata["derived_from_node_id"] = str(
                raw.get("metadata", {}).get("derived_from_node_id", inherited.element_id)
            ) if isinstance(raw.get("metadata"), dict) else inherited.element_id
            node.metadata["lineage_root_id"] = inherited.element_id
        graph.add_node(node)
    valid_ids = set(graph.node_map())
    for node in graph.nodes:
        if node.parent_id not in valid_ids:
            node.parent_id = None
    proposal_records = document.get("proposals")
    if not isinstance(proposal_records, list):
        raise ValueError("Review checkpoint has no proposal ledger.")
    for raw in proposal_records:
        if not isinstance(raw, dict):
            raise ValueError("Review proposal record is not an object.")
        owners = [str(value) for value in raw.get("owner_ids", []) if str(value) in valid_ids]
        status = str(raw.get("status", "unresolved"))
        if status == "assigned" and not owners:
            status = "unresolved"
        reason = str(raw.get("reason") or "") or None
        if status == "rejected" and not reason:
            reason = "Rejected by the user in V5 review."
        graph.add_proposal(
            ProposalRecord(
                proposal_id=str(raw.get("id") or ""),
                source=str(raw.get("source") or "review"),
                kind_hint=str(raw.get("kind_hint") or "unknown"),  # type: ignore[arg-type]
                bbox=tuple(int(value) for value in raw.get("bbox", [0, 0, 0, 0])),  # type: ignore[arg-type]
                confidence=float(raw.get("confidence", 0.0)),
                status=status,  # type: ignore[arg-type]
                owner_ids=owners,
                reason=reason,
                evidence=copy.deepcopy(raw.get("evidence", {})),
            )
        )
    graph.metadata["review"] = {
        "checkpoint": checkpoint_path.name,
        "finished": bool(document.get("review", {}).get("finished", False))
        if isinstance(document.get("review"), dict)
        else False,
        "groups": copy.deepcopy(document.get("groups", [])),
    }
    graph.validate()
    return graph
