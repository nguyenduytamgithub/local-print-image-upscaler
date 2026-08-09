from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np


Box = tuple[int, int, int, int]
ElementKind = Literal[
    "background",
    "panel",
    "frame",
    "line",
    "ribbon",
    "text",
    "price",
    "product",
    "logo",
    "qr",
    "icon",
    "badge",
    "decoration",
    "micro_detail",
    "unknown",
]
ReviewStatus = Literal["auto_confirmed", "user_confirmed", "unresolved", "rejected"]
ProposalStatus = Literal["assigned", "unresolved", "rejected"]


def clip_box(box: Box, canvas_size: tuple[int, int]) -> Box:
    width, height = canvas_size
    x0, y0, x1, y1 = (int(value) for value in box)
    x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
    y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
    return x0, y0, x1, y1


@dataclass(slots=True)
class AlphaCrop:
    """A memory-bounded straight-alpha mask positioned on the document canvas."""

    left: int
    top: int
    alpha: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.alpha)
        if value.ndim != 2:
            raise ValueError("AlphaCrop.alpha must be a 2-D array.")
        if value.dtype != np.uint8:
            raise ValueError("AlphaCrop.alpha must use uint8 straight alpha.")
        if value.shape[0] < 1 or value.shape[1] < 1:
            raise ValueError("AlphaCrop may not be empty.")
        if self.left < 0 or self.top < 0:
            raise ValueError("AlphaCrop offset may not be negative.")
        self.alpha = np.ascontiguousarray(value)

    @property
    def width(self) -> int:
        return int(self.alpha.shape[1])

    @property
    def height(self) -> int:
        return int(self.alpha.shape[0])

    @property
    def bbox(self) -> Box:
        return self.left, self.top, self.left + self.width, self.top + self.height

    @property
    def nonzero_pixels(self) -> int:
        return int(np.count_nonzero(self.alpha))

    @property
    def opaque_pixels(self) -> int:
        return int(np.count_nonzero(self.alpha == 255))

    def binary(self, threshold: int = 1) -> np.ndarray:
        if not 1 <= threshold <= 255:
            raise ValueError("threshold must be from 1 through 255")
        return self.alpha >= threshold

    def canvas_slice(self, canvas_size: tuple[int, int]) -> tuple[slice, slice]:
        width, height = canvas_size
        if self.left + self.width > width or self.top + self.height > height:
            raise ValueError(f"Alpha crop {self.bbox} escapes canvas {canvas_size}.")
        return slice(self.top, self.top + self.height), slice(self.left, self.left + self.width)

    def to_canvas(self, canvas_size: tuple[int, int]) -> np.ndarray:
        width, height = canvas_size
        result = np.zeros((height, width), dtype=np.uint8)
        ys, xs = self.canvas_slice(canvas_size)
        result[ys, xs] = self.alpha
        return result

    def tight(self, padding: int = 0) -> "AlphaCrop":
        if padding < 0:
            raise ValueError("padding may not be negative")
        ys, xs = np.where(self.alpha > 0)
        if not len(xs):
            return self
        x0 = max(0, int(xs.min()) - padding)
        y0 = max(0, int(ys.min()) - padding)
        x1 = min(self.width, int(xs.max()) + 1 + padding)
        y1 = min(self.height, int(ys.max()) + 1 + padding)
        return AlphaCrop(self.left + x0, self.top + y0, self.alpha[y0:y1, x0:x1].copy())


@dataclass(slots=True)
class ElementNode:
    """Smallest useful editable element; groups never destroy atomic leaves."""

    element_id: str
    name: str
    kind: ElementKind
    visible_alpha: AlphaCrop
    z_index: int
    parent_id: str | None = None
    semantic_envelope: AlphaCrop | None = None
    full_support: AlphaCrop | None = None
    removal_footprint: AlphaCrop | None = None
    confidence: float = 0.0
    review_status: ReviewStatus = "unresolved"
    move_safe: bool = False
    occluded: bool = False
    synthesized_hidden_pixels: bool = False
    text: str | None = None
    rgba: np.ndarray | None = field(default=None, repr=False)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.element_id or not self.name:
            raise ValueError("Element id and name are required.")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Element confidence must be in [0, 1].")
        if self.rgba is not None:
            rgba = np.asarray(self.rgba)
            if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
                raise ValueError("Element rgba must be a uint8 HxWx4 array.")
            export_support = self.full_support or self.visible_alpha
            if export_support.bbox != self.visible_alpha.bbox:
                raise ValueError("Visible alpha and full support must share one crop bbox.")
            if rgba.shape[:2] != export_support.alpha.shape:
                raise ValueError("Element rgba and export support crop sizes differ.")
            if not np.array_equal(rgba[:, :, 3], export_support.alpha):
                raise ValueError("RGBA alpha must equal full support (or visible alpha when unoccluded).")
            self.rgba = np.ascontiguousarray(rgba)

    @property
    def bbox(self) -> Box:
        return self.visible_alpha.bbox

    def manifest_record(self) -> dict[str, Any]:
        return {
            "id": self.element_id,
            "name": self.name,
            "kind": self.kind,
            "bbox": list(self.bbox),
            "z_index": self.z_index,
            "parent_id": self.parent_id,
            "confidence": round(float(self.confidence), 6),
            "review_status": self.review_status,
            "move_safe": self.move_safe,
            "occluded": self.occluded,
            "synthesized_hidden_pixels": self.synthesized_hidden_pixels,
            "text": self.text,
            "alpha_nonzero_pixels": self.visible_alpha.nonzero_pixels,
            "full_support_nonzero_pixels": (
                self.full_support.nonzero_pixels if self.full_support else self.visible_alpha.nonzero_pixels
            ),
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ProposalRecord:
    proposal_id: str
    source: str
    kind_hint: ElementKind
    bbox: Box
    confidence: float
    status: ProposalStatus = "unresolved"
    owner_ids: list[str] = field(default_factory=list)
    reason: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.source:
            raise ValueError("Proposal id and source are required.")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Proposal confidence must be in [0, 1].")
        if self.status == "assigned" and not self.owner_ids:
            raise ValueError("Assigned proposal must identify at least one owner.")
        if self.status == "rejected" and not self.reason:
            raise ValueError("Rejected proposal must record a reason.")

    def manifest_record(self) -> dict[str, Any]:
        return {
            "id": self.proposal_id,
            "source": self.source,
            "kind_hint": self.kind_hint,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 6),
            "status": self.status,
            "owner_ids": list(self.owner_ids),
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class DocumentGraph:
    canvas_size: tuple[int, int]
    nodes: list[ElementNode] = field(default_factory=list)
    proposals: list[ProposalRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        width, height = self.canvas_size
        if width < 1 or height < 1:
            raise ValueError("Document canvas must be positive.")

    def add_node(self, node: ElementNode) -> None:
        if any(current.element_id == node.element_id for current in self.nodes):
            raise ValueError(f"Duplicate element id: {node.element_id}")
        node.visible_alpha.canvas_slice(self.canvas_size)
        self.nodes.append(node)

    def add_proposal(self, proposal: ProposalRecord) -> None:
        if any(current.proposal_id == proposal.proposal_id for current in self.proposals):
            raise ValueError(f"Duplicate proposal id: {proposal.proposal_id}")
        proposal.bbox = clip_box(proposal.bbox, self.canvas_size)
        self.proposals.append(proposal)

    def node_map(self) -> dict[str, ElementNode]:
        return {node.element_id: node for node in self.nodes}

    def validate(self) -> None:
        by_id = self.node_map()
        if len(by_id) != len(self.nodes):
            raise RuntimeError("Document graph contains duplicate node ids.")
        proposal_ids = {proposal.proposal_id for proposal in self.proposals}
        if len(proposal_ids) != len(self.proposals):
            raise RuntimeError("Proposal ledger contains duplicate ids.")
        for node in self.nodes:
            node.visible_alpha.canvas_slice(self.canvas_size)
            if node.parent_id and node.parent_id not in by_id:
                raise RuntimeError(f"Unknown parent {node.parent_id!r} for {node.element_id!r}.")
            chain: set[str] = set()
            current = node
            while current.parent_id:
                if current.element_id in chain:
                    raise RuntimeError(f"Hierarchy cycle at {current.element_id!r}.")
                chain.add(current.element_id)
                current = by_id[current.parent_id]
        for proposal in self.proposals:
            unknown = sorted(set(proposal.owner_ids) - set(by_id))
            if unknown:
                raise RuntimeError(
                    f"Proposal {proposal.proposal_id!r} references unknown owner(s): {unknown}"
                )
            if proposal.status == "assigned" and not proposal.owner_ids:
                raise RuntimeError(f"Assigned proposal {proposal.proposal_id!r} has no owner.")

    def unresolved_proposals(self) -> list[ProposalRecord]:
        return [proposal for proposal in self.proposals if proposal.status == "unresolved"]

    def unresolved_nodes(self) -> list[ElementNode]:
        return [node for node in self.nodes if node.review_status == "unresolved"]

    def ownership_maps(self, threshold: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return owner count and visible union without allocating per-node canvases."""

        width, height = self.canvas_size
        owner_count = np.zeros((height, width), dtype=np.uint16)
        visible_union = np.zeros((height, width), dtype=bool)
        for node in self.nodes:
            ys, xs = node.visible_alpha.canvas_slice(self.canvas_size)
            binary = node.visible_alpha.alpha >= threshold
            owner_count[ys, xs] += binary.astype(np.uint16)
            visible_union[ys, xs] |= binary
        return owner_count, visible_union

    def proposal_accounting(self) -> dict[str, int]:
        counts = {"total": len(self.proposals), "assigned": 0, "unresolved": 0, "rejected": 0}
        for proposal in self.proposals:
            counts[proposal.status] += 1
        return counts

    def manifest_record(self) -> dict[str, Any]:
        self.validate()
        owner_count, visible_union = self.ownership_maps()
        overlap = owner_count > 1
        return {
            "schema": "V5_DOCUMENT_GRAPH_V2",
            "canvas": list(self.canvas_size),
            "node_count": len(self.nodes),
            "atomic_leaf_count": sum(
                1 for node in self.nodes if not any(item.parent_id == node.element_id for item in self.nodes)
            ),
            "review_unresolved_node_count": len(self.unresolved_nodes()),
            "proposal_accounting": self.proposal_accounting(),
            "visible_union_pixels": int(np.count_nonzero(visible_union)),
            "multiply_owned_pixels": int(np.count_nonzero(overlap)),
            "max_owner_count": int(owner_count.max()) if owner_count.size else 0,
            "nodes": [node.manifest_record() for node in sorted(self.nodes, key=lambda item: item.z_index)],
            "proposals": [proposal.manifest_record() for proposal in self.proposals],
            "metadata": self.metadata,
        }


def union_alpha(crops: Iterable[AlphaCrop], canvas_size: tuple[int, int]) -> np.ndarray:
    width, height = canvas_size
    result = np.zeros((height, width), dtype=np.uint8)
    for crop in crops:
        ys, xs = crop.canvas_slice(canvas_size)
        np.maximum(result[ys, xs], crop.alpha, out=result[ys, xs])
    return result
