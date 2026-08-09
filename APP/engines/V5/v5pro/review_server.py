"""Local, reviewable V5 layer checkpoint editor.

The browser is deliberately only a friendly view over a portable checkpoint.
It never receives a filesystem path and never edits the technical JSON
directly.  Every mutation is validated, persisted with ``os.replace`` and can
be undone while the session remains open.  Generated masks are immutable
assets, which makes undo/redo reliable without copying a whole poster for each
operation.

Checkpoint contract (``V5_LAYER_REVIEW_V2``)
--------------------------------------------

``source`` and optional node ``mask`` values are POSIX-style paths relative to
the checkpoint directory.  A node mask may either be a canvas-sized grayscale
PNG or a tight crop matching its ``bbox``.  Nodes without a mask are treated as
rectangular provisional masks; the first brush/split action materialises a
real PNG.  Node/proposal records intentionally use the same public keys as
``DocumentGraph.manifest_record()`` in :mod:`v5pro.schema`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
import unicodedata
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, MutableMapping, Sequence, get_args
from urllib.parse import parse_qs, quote, urlparse

from PIL import Image, ImageDraw

from .schema import DocumentGraph, ElementKind


REVIEW_SCHEMA = "V5_LAYER_REVIEW_V2"
LOOPBACK_HOST = "127.0.0.1"
TOKEN_HEADER = "X-V5-Review-Token"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_NODES = 10_000
MAX_PROPOSALS = 20_000
MAX_GROUPS = 5_000
MAX_BRUSH_POINTS = 4_096
MAX_TEXT_CHARS = 240

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NODE_STATUSES = {"auto_confirmed", "user_confirmed", "unresolved", "rejected"}
_PROPOSAL_STATUSES = {"assigned", "unresolved", "rejected"}
_ELEMENT_KINDS = frozenset(get_args(ElementKind))
_IMMUTABLE_KINDS = {"text", "price", "qr"}
_SEMANTIC_KINDS = {"product", "logo", "icon", "badge", "ribbon", "decoration"}
_GEOMETRY_KINDS = {"frame", "line"}
_KIND_LABELS = {
    "background": "Nền",
    "panel": "Ô / mảng",
    "frame": "Khung",
    "line": "Đường kẻ",
    "ribbon": "Dải băng",
    "text": "Chữ",
    "price": "Giá",
    "product": "Sản phẩm",
    "logo": "Logo",
    "qr": "Mã QR",
    "icon": "Biểu tượng",
    "badge": "Nhãn / huy hiệu",
    "decoration": "Trang trí",
    "micro_detail": "Chi tiết nhỏ",
    "unknown": "Chưa xác định",
}


class ReviewUIError(RuntimeError):
    """Base exception for the V5 review service."""


class ReviewValidationError(ReviewUIError):
    """The checkpoint or requested edit is structurally invalid."""


class ReviewPathError(ReviewUIError):
    """An asset path is unsafe, missing, or outside the review bundle."""


class ReviewConflictError(ReviewUIError):
    """The checkpoint changed outside this browser session."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record_stack_priority(record: Mapping[str, object]) -> tuple[int, int, str]:
    """Mirror V5 ownership order for checkpoint-only grouping validation."""

    identifier = str(record.get("id") or "")
    kind = str(record.get("kind") or "unknown")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    sources = tuple(
        str(item.get("source", "")).strip().lower()
        for item in evidence
        if isinstance(item, dict)
    )
    raw_layerd = any(source.startswith("layerd") for source in sources) or (
        identifier.startswith("ELEMENT_")
        and ("source_iteration" in metadata or "member_proposal_ids" in metadata)
    )
    user_authored = bool(
        record.get("review_status") == "user_confirmed"
        and identifier.startswith("USER_")
    )
    verified_semantic = bool(
        kind in _SEMANTIC_KINDS
        and (
            any(
                source.startswith(("grounding_dino", "sam", "birefnet", "semantic"))
                for source in sources
            )
            or identifier.startswith("SEMANTIC_")
            or "semantic_extraction" in metadata
            or metadata.get("semantic_verified") is True
            or (record.get("review_status") == "user_confirmed" and not raw_layerd)
        )
    )
    verified_geometry = bool(
        kind in _GEOMETRY_KINDS
        and not raw_layerd
        and (
            "geometry_backend" in metadata
            or metadata.get("surface_reconciliation") is True
            or any(
                source.startswith(
                    ("opencv_", "poster_geometry", "poster_surface_residual", "hough")
                )
                for source in sources
            )
            or record.get("review_status") == "user_confirmed"
        )
    )
    if kind in _IMMUTABLE_KINDS:
        band = 600
    elif user_authored:
        band = 475
    elif verified_semantic:
        band = 500
    elif verified_geometry:
        band = 400
    elif metadata.get("role") == "exact_source_remainder_above_clean_base":
        band = 450
    elif kind == "panel" and not raw_layerd:
        band = 300
    elif raw_layerd:
        band = 100
    elif kind in _GEOMETRY_KINDS:
        band = 180
    else:
        band = 200
    return band, int(record.get("z_index", 0)), identifier


def _friendly_node_review_note(record: Mapping[str, object]) -> str | None:
    """Explain an unresolved node without exposing technical JSON to the user."""

    if str(record.get("review_status", "unresolved")) != "unresolved":
        return None
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if metadata.get("role") == "exact_source_remainder_above_clean_base":
        return (
            "Đây là phần ảnh nguồn chưa được tách thành đối tượng. Giữ nó để "
            "đối chiếu ảnh gốc; ẩn nó để xem nền và hình học sạch phía dưới."
        )
    atomicity = metadata.get("semantic_atomicity")
    if isinstance(atomicity, dict) and atomicity.get("atomic_leaf_confirmed") is not True:
        classification = str(atomicity.get("classification") or "")
        if classification == "compound_subassembly":
            return (
                "Mask này sạch để di chuyển cả cụm, nhưng có nhiều vật thể chạm "
                "nhau. Hãy xác nhận giữ nguyên cụm hoặc tách tiếp bằng công cụ review."
            )
        return (
            "Mask có thể dùng để di chuyển toàn bộ phần đang thấy, nhưng chưa đủ "
            "bằng chứng đây là một vật thể đơn. Hãy kiểm tra trước khi xác nhận."
        )
    refinement = metadata.get("refinement")
    if isinstance(refinement, dict) and refinement.get("accepted") is False:
        return (
            "Biên tách tự động chưa đủ ổn định. Hãy xem lớp phủ mask và chỉnh bằng "
            "cọ hoặc khung tách trước khi xác nhận."
        )
    export_purity = metadata.get("text_export_purity_preflight")
    if isinstance(export_purity, dict) and export_purity.get("status") != "pass":
        reasons = {
            str(item) for item in export_purity.get("reasons", [])
        }
        if "foreground_core_not_retained" in reasons:
            return (
                "Mask xuất đã bỏ sót phần lõi đậm/sáng của chữ gốc hoặc chỉ giữ viền vụn. "
                "Layer này chưa an toàn để di chuyển; hãy kiểm tra mask và tách lại."
            )
        if "isolated_out_of_primary_band_edge_component" in reasons:
            return (
                "Mask chữ xuất còn dính một mảnh màu ở ngoài dải chữ và sát mép khung. "
                "Hãy gỡ mảnh thừa trước khi xác nhận layer sạch."
            )
        return (
            "Mask chữ xuất cuối chưa chứng minh được glyph nguyên vẹn và không dính vật thể. "
            "Hãy kiểm tra ở mức phóng to trước khi xác nhận."
        )
    purity = metadata.get("text_purity")
    if isinstance(purity, dict) and purity.get("status") != "pass":
        return (
            "Mask chữ hoặc giá có thể còn dính nền hay vật bên cạnh. Hãy phóng to "
            "kiểm tra biên rồi mới xác nhận layer sạch."
        )
    cleanliness = metadata.get("geometry_cleanliness")
    if str(record.get("kind")) in {"panel", "frame", "line"} and (
        not isinstance(cleanliness, dict) or cleanliness.get("status") != "pass"
    ):
        return (
            "Mảng hoặc đường hình học này chưa có đủ bằng chứng bề mặt sạch. Hãy "
            "kiểm tra phần nền/chữ còn dính trước khi xác nhận."
        )
    return (
        "Mục này chưa đủ bằng chứng để tự xác nhận. Hãy xem mask, chỉnh nếu cần, "
        "rồi bấm “Xác nhận layer sạch”."
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
    try:
        payload = (
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReviewValidationError("Checkpoint chứa dữ liệu không thể lưu thành JSON.") from exc
    _atomic_write_bytes(path, payload)


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewValidationError(f"Không đọc được checkpoint V5: {path.name}") from exc
    if not isinstance(value, dict):
        raise ReviewValidationError("Checkpoint V5 phải là một JSON object.")
    return value


def _resolve_checkpoint_path(path: Path | str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ReviewPathError(f"Không tìm thấy checkpoint V5: {path}") from exc
    if not resolved.is_file() or resolved.suffix.lower() != ".json":
        raise ReviewPathError("Checkpoint V5 phải là file .json có thật.")
    return resolved


def _relative_asset_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ReviewPathError(f"{label} phải là đường dẫn tương đối trong bundle.")
    raw = value.replace("\\", "/")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ReviewPathError(f"{label} không được thoát khỏi bundle.")
    if ":" in relative.parts[0]:
        raise ReviewPathError(f"{label} không được là đường dẫn ổ đĩa tuyệt đối.")
    return relative


def _resolve_bundle_asset(
    root: Path,
    value: object,
    *,
    label: str,
    must_exist: bool = True,
) -> Path:
    relative = _relative_asset_path(value, label=label)
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise ReviewPathError(f"Không tìm thấy {label}: {relative.as_posix()}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReviewPathError(f"{label} không được thoát khỏi bundle.") from exc
    if must_exist and not resolved.is_file():
        raise ReviewPathError(f"{label} phải là một file.")
    return resolved


def _clean_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ReviewValidationError(f"{label} phải là chuỗi.")
    normalized = value.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ReviewValidationError(
            f"{label} chỉ được chứa chữ không dấu, số và . _ : - (tối đa 128 ký tự)."
        )
    return normalized


def _clean_name(value: object, *, label: str = "Tên") -> str:
    if not isinstance(value, str):
        raise ReviewValidationError(f"{label} phải là chuỗi Unicode.")
    normalized = unicodedata.normalize("NFC", value).strip()
    if not normalized or len(normalized) > MAX_TEXT_CHARS:
        raise ReviewValidationError(f"{label} phải có 1–{MAX_TEXT_CHARS} ký tự.")
    if any(unicodedata.category(char) == "Cc" for char in normalized):
        raise ReviewValidationError(f"{label} chứa ký tự điều khiển không an toàn.")
    return normalized


def _integer_bbox(value: object, *, canvas: tuple[int, int], label: str = "Vùng") -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ReviewValidationError(f"{label} phải có bốn tọa độ.")
    if any(isinstance(item, bool) for item in value):
        raise ReviewValidationError(f"Tọa độ {label.lower()} không hợp lệ.")
    try:
        box = tuple(int(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReviewValidationError(f"Tọa độ {label.lower()} không hợp lệ.") from exc
    x0, y0, x1, y1 = box
    width, height = canvas
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ReviewValidationError(f"{label} {box} nằm ngoài ảnh {width}x{height}.")
    return box  # type: ignore[return-value]


def _records(document: Mapping[str, object], key: str) -> list[dict[str, object]]:
    value = document.get(key, [])
    if not isinstance(value, list):
        raise ReviewValidationError(f"Trường {key} phải là danh sách.")
    if any(not isinstance(item, dict) for item in value):
        raise ReviewValidationError(f"Mỗi mục trong {key} phải là object.")
    return value  # type: ignore[return-value]


def _mask_reference(node: Mapping[str, object]) -> str | None:
    for key in ("mask", "mask_path", "visible_alpha_path"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def validate_review_checkpoint(
    document: Mapping[str, object],
    *,
    bundle_root: Path,
    image_size: tuple[int, int],
    validate_mask_files: bool = True,
) -> None:
    """Validate every authority-bearing field and every portable asset path."""

    if document.get("schema") != REVIEW_SCHEMA:
        raise ReviewValidationError(f"Sai schema; cần {REVIEW_SCHEMA}.")
    canvas_value = document.get("canvas")
    if (
        not isinstance(canvas_value, (list, tuple))
        or len(canvas_value) != 2
        or any(isinstance(item, bool) for item in canvas_value)
    ):
        raise ReviewValidationError("Checkpoint thiếu canvas [rộng, cao].")
    try:
        canvas = int(canvas_value[0]), int(canvas_value[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReviewValidationError("Kích thước canvas không hợp lệ.") from exc
    if canvas != image_size or canvas[0] < 1 or canvas[1] < 1:
        raise ReviewValidationError(
            f"Canvas checkpoint {canvas} không khớp ảnh nguồn {image_size}."
        )

    source = _resolve_bundle_asset(bundle_root, document.get("source"), label="ảnh nguồn")
    source_sha = document.get("source_sha256")
    if source_sha is not None:
        normalized_sha = str(source_sha).strip().lower()
        if not _SHA256_RE.fullmatch(normalized_sha):
            raise ReviewValidationError("source_sha256 không hợp lệ.")
        if not secrets.compare_digest(normalized_sha, _sha256_file(source)):
            raise ReviewConflictError("Ảnh nguồn đã thay đổi sau khi tạo checkpoint.")

    nodes = _records(document, "nodes")
    proposals = _records(document, "proposals")
    groups = _records(document, "groups")
    if len(nodes) > MAX_NODES or len(proposals) > MAX_PROPOSALS or len(groups) > MAX_GROUPS:
        raise ReviewValidationError("Checkpoint vượt giới hạn số mục an toàn.")

    node_ids: set[str] = set()
    for node in nodes:
        identifier = _clean_identifier(node.get("id"), label="Mã layer")
        if identifier in node_ids:
            raise ReviewValidationError(f"Trùng mã layer: {identifier}")
        node_ids.add(identifier)
        _clean_name(node.get("name"), label=f"Tên layer {identifier}")
        if node.get("kind") not in _ELEMENT_KINDS:
            raise ReviewValidationError(f"Loại layer {identifier} không hợp lệ.")
        bbox = _integer_bbox(node.get("bbox"), canvas=canvas, label=f"Layer {identifier}")
        if node.get("review_status", "unresolved") not in _NODE_STATUSES:
            raise ReviewValidationError(f"Trạng thái layer {identifier} không hợp lệ.")
        mask_ref = _mask_reference(node)
        if mask_ref is not None:
            mask_path = _resolve_bundle_asset(bundle_root, mask_ref, label=f"mask {identifier}")
            if mask_path.suffix.lower() != ".png":
                raise ReviewPathError(f"Mask {identifier} phải là PNG.")
            if validate_mask_files:
                try:
                    with Image.open(mask_path) as opened:
                        opened.load()
                        mask_size = opened.size
                except (OSError, ValueError) as exc:
                    raise ReviewPathError(f"Không mở được mask {identifier}.") from exc
                crop_size = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if mask_size not in {canvas, crop_size}:
                    raise ReviewValidationError(
                        f"Mask {identifier} phải bằng canvas hoặc bbox {crop_size}, hiện là {mask_size}."
                    )

    node_map = {str(node["id"]): node for node in nodes}
    for identifier, node in node_map.items():
        parent = node.get("parent_id")
        if parent is not None and (not isinstance(parent, str) or parent not in node_ids):
            raise ReviewValidationError(f"Layer cha của {identifier} không tồn tại.")
        chain: set[str] = set()
        current = node
        while current.get("parent_id") is not None:
            current_id = str(current["id"])
            if current_id in chain:
                raise ReviewValidationError(f"Cây layer bị vòng lặp tại {identifier}.")
            chain.add(current_id)
            current = node_map[str(current["parent_id"])]

    proposal_ids: set[str] = set()
    for proposal in proposals:
        identifier = _clean_identifier(proposal.get("id"), label="Mã đề xuất")
        if identifier in proposal_ids:
            raise ReviewValidationError(f"Trùng mã đề xuất: {identifier}")
        proposal_ids.add(identifier)
        if proposal.get("kind_hint", "unknown") not in _ELEMENT_KINDS:
            raise ReviewValidationError(f"Loại đề xuất {identifier} không hợp lệ.")
        bbox = _integer_bbox(proposal.get("bbox"), canvas=canvas, label=f"Đề xuất {identifier}")
        status = proposal.get("status", "unresolved")
        if status not in _PROPOSAL_STATUSES:
            raise ReviewValidationError(f"Trạng thái đề xuất {identifier} không hợp lệ.")
        owners = proposal.get("owner_ids", [])
        if not isinstance(owners, list) or any(not isinstance(item, str) for item in owners):
            raise ReviewValidationError(f"owner_ids của {identifier} không hợp lệ.")
        if set(owners) - node_ids:
            raise ReviewValidationError(f"Đề xuất {identifier} tham chiếu layer không tồn tại.")
        if status == "assigned" and not owners:
            raise ReviewValidationError(f"Đề xuất {identifier} đã nhận nhưng chưa có layer sở hữu.")
        if status == "rejected" and not str(proposal.get("reason") or "").strip():
            raise ReviewValidationError(f"Đề xuất {identifier} bị từ chối nhưng thiếu lý do.")
        mask_ref = _mask_reference(proposal)
        if mask_ref is not None:
            mask_path = _resolve_bundle_asset(bundle_root, mask_ref, label=f"mask đề xuất {identifier}")
            if mask_path.suffix.lower() != ".png":
                raise ReviewPathError(f"Mask đề xuất {identifier} phải là PNG.")
            if validate_mask_files:
                try:
                    with Image.open(mask_path) as opened:
                        opened.load()
                        mask_size = opened.size
                except (OSError, ValueError) as exc:
                    raise ReviewPathError(f"Không mở được mask đề xuất {identifier}.") from exc
                crop_size = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if mask_size not in {canvas, crop_size}:
                    raise ReviewValidationError(
                        f"Mask đề xuất {identifier} phải bằng canvas hoặc bbox {crop_size}."
                    )

    group_ids: set[str] = set()
    seen_members: set[str] = set()
    for group in groups:
        identifier = _clean_identifier(group.get("id"), label="Mã nhóm")
        if identifier in group_ids:
            raise ReviewValidationError(f"Trùng mã nhóm: {identifier}")
        group_ids.add(identifier)
        _clean_name(group.get("name"), label=f"Tên nhóm {identifier}")
        members = group.get("member_ids")
        if not isinstance(members, list) or len(members) < 2:
            raise ReviewValidationError(f"Nhóm {identifier} phải có ít nhất hai layer.")
        if any(not isinstance(item, str) for item in members) or set(members) - node_ids:
            raise ReviewValidationError(f"Nhóm {identifier} chứa layer không tồn tại.")
        if len(set(members)) != len(members):
            raise ReviewValidationError(f"Nhóm {identifier} chứa layer trùng.")
        overlap = seen_members.intersection(members)
        if overlap:
            raise ReviewValidationError(
                f"Layer chỉ được nằm trong một nhóm thao tác: {sorted(overlap)}"
            )
        seen_members.update(members)
        parents = {node_map[member].get("parent_id") for member in members}
        if len(parents) != 1:
            raise ReviewValidationError(
                f"Nhóm {identifier} chỉ được chứa layer cùng một thư mục cha."
            )
        parent_id = next(iter(parents))
        siblings = sorted(
            [node for node in nodes if node.get("parent_id") == parent_id],
            key=_record_stack_priority,
        )
        positions = sorted(
            next(
                index
                for index, node in enumerate(siblings)
                if str(node.get("id")) == member
            )
            for member in members
        )
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ReviewValidationError(
                f"Nhóm {identifier} phải liền nhau theo thứ tự layer để không đổi hình ảnh."
            )


def build_review_checkpoint(
    graph: DocumentGraph | Mapping[str, object],
    *,
    source: str,
    source_sha256: str | None = None,
    mask_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Adapt a :class:`DocumentGraph` manifest to the portable review schema."""

    record = graph.manifest_record() if isinstance(graph, DocumentGraph) else copy.deepcopy(dict(graph))
    canvas = record.get("canvas")
    nodes = copy.deepcopy(record.get("nodes", []))
    proposals = copy.deepcopy(record.get("proposals", []))
    if not isinstance(nodes, list) or not isinstance(proposals, list):
        raise ReviewValidationError("Graph manifest thiếu nodes/proposals.")
    masks = dict(mask_paths or {})
    for node in nodes:
        if isinstance(node, dict) and str(node.get("id")) in masks:
            node["mask"] = masks[str(node["id"])]
    checkpoint: dict[str, object] = {
        "schema": REVIEW_SCHEMA,
        "source": source,
        "canvas": copy.deepcopy(canvas),
        "nodes": nodes,
        "proposals": proposals,
        "groups": [],
        "review": {"revision": 0, "finished": False},
    }
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        organization = metadata.get("organization")
        if isinstance(organization, dict):
            checkpoint["organization"] = copy.deepcopy(organization)
        review_metadata = metadata.get("review")
        if isinstance(review_metadata, dict):
            groups = review_metadata.get("groups")
            if isinstance(groups, list):
                checkpoint["groups"] = copy.deepcopy(
                    [group for group in groups if isinstance(group, dict)]
                )
        signature = metadata.get("review_resume_signature")
        if isinstance(signature, str) and signature.strip():
            checkpoint["review_resume_signature"] = signature.strip()
    if source_sha256 is not None:
        checkpoint["source_sha256"] = str(source_sha256).strip().lower()
    return checkpoint


def write_review_checkpoint(path: Path | str, document: Mapping[str, object]) -> None:
    """Persist a checkpoint atomically; full asset validation happens on open."""

    _atomic_write_json(Path(path).expanduser().resolve(), document)


@dataclass(slots=True)
class ReviewSession:
    checkpoint_path: Path
    source_path: Path
    document: dict[str, object]
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    finished: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _disk_sha256: str = field(default="", repr=False)
    _canvas: tuple[int, int] = field(default=(0, 0), repr=False)
    _undo: list[dict[str, object]] = field(default_factory=list, repr=False)
    _redo: list[dict[str, object]] = field(default_factory=list, repr=False)
    _source_cache: tuple[bytes, str] | None = field(default=None, repr=False)
    _mask_cache: dict[tuple[str, str | None], bytes] = field(default_factory=dict, repr=False)

    @classmethod
    def open(cls, checkpoint_path: Path | str) -> "ReviewSession":
        resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path)
        root = resolved_checkpoint.parent.resolve()
        document = _load_json(resolved_checkpoint)
        source = _resolve_bundle_asset(root, document.get("source"), label="ảnh nguồn")
        try:
            with Image.open(source) as opened:
                opened.load()
                image_size = opened.size
        except (OSError, ValueError) as exc:
            raise ReviewPathError("Không mở được ảnh nguồn của checkpoint.") from exc
        validate_review_checkpoint(
            document,
            bundle_root=root,
            image_size=image_size,
            validate_mask_files=True,
        )
        session = cls(
            checkpoint_path=resolved_checkpoint,
            source_path=source,
            document=copy.deepcopy(document),
            _disk_sha256=_sha256_file(resolved_checkpoint),
            _canvas=image_size,
        )
        review = document.get("review")
        if isinstance(review, dict) and review.get("finished") is True:
            session.finished.set()
        return session

    @property
    def bundle_root(self) -> Path:
        return self.checkpoint_path.parent.resolve()

    def _nodes(self, document: MutableMapping[str, object] | None = None) -> list[dict[str, object]]:
        return _records(document or self.document, "nodes")

    def _proposals(self, document: MutableMapping[str, object] | None = None) -> list[dict[str, object]]:
        return _records(document or self.document, "proposals")

    def _groups(self, document: MutableMapping[str, object] | None = None) -> list[dict[str, object]]:
        return _records(document or self.document, "groups")

    def _drop_groups_containing(
        self,
        document: MutableMapping[str, object],
        member_ids: set[str],
    ) -> None:
        """Remove organizational containers invalidated by an authority edit."""

        groups = self._groups(document)
        groups[:] = [
            group
            for group in groups
            if not member_ids.intersection(
                str(member) for member in group.get("member_ids", [])
            )
        ]

    def _node(
        self, identifier: object, document: MutableMapping[str, object] | None = None
    ) -> dict[str, object]:
        selected = _clean_identifier(identifier, label="Mã layer")
        for node in self._nodes(document):
            if secrets.compare_digest(str(node.get("id")), selected):
                return node
        raise ReviewValidationError(f"Không tìm thấy layer {selected}.")

    def _proposal(
        self, identifier: object, document: MutableMapping[str, object] | None = None
    ) -> dict[str, object]:
        selected = _clean_identifier(identifier, label="Mã đề xuất")
        for proposal in self._proposals(document):
            if secrets.compare_digest(str(proposal.get("id")), selected):
                return proposal
        raise ReviewValidationError(f"Không tìm thấy đề xuất {selected}.")

    def _check_disk(self) -> None:
        if _sha256_file(self.checkpoint_path) != self._disk_sha256:
            raise ReviewConflictError(
                "Checkpoint đã thay đổi ở nơi khác; hãy mở lại trang để tránh ghi đè."
            )

    def _revision(self, document: Mapping[str, object] | None = None) -> int:
        review = (document or self.document).get("review")
        if isinstance(review, dict):
            try:
                return max(0, int(review.get("revision", 0)))
            except (TypeError, ValueError):
                return 0
        return 0

    def _stamp(self, document: MutableMapping[str, object], *, finished: bool | None = None) -> None:
        review_value = document.get("review")
        review = copy.deepcopy(review_value) if isinstance(review_value, dict) else {}
        review["revision"] = self._revision() + 1
        if finished is not None:
            review["finished"] = finished
        else:
            review.setdefault("finished", False)
        review["last_editor"] = "local-v5-review-ui"
        document["review"] = review

    def _commit(
        self,
        candidate: dict[str, object],
        *,
        prior: dict[str, object],
        generated_assets: Sequence[Path] = (),
        record_undo: bool = True,
    ) -> None:
        try:
            validate_review_checkpoint(
                candidate,
                bundle_root=self.bundle_root,
                image_size=self._canvas,
                # The complete bundle was decoded on open.  Mutations never
                # accept an arbitrary mask path, so existence/path validation
                # is sufficient here and keeps large catalogs responsive.
                validate_mask_files=False,
            )
            _atomic_write_json(self.checkpoint_path, candidate)
        except BaseException:
            for path in generated_assets:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        if record_undo:
            self._undo.append(prior)
            if len(self._undo) > 100:
                del self._undo[:-100]
            self._redo.clear()
        self.document = candidate
        self._disk_sha256 = _sha256_file(self.checkpoint_path)
        self._mask_cache.clear()

    def _mutate(
        self,
        operation: Callable[[dict[str, object], list[Path]], None],
    ) -> dict[str, object]:
        with self._lock:
            self._check_disk()
            prior = copy.deepcopy(self.document)
            candidate = copy.deepcopy(self.document)
            generated_assets: list[Path] = []
            operation(candidate, generated_assets)
            self._stamp(candidate, finished=False)
            self._commit(
                candidate,
                prior=prior,
                generated_assets=generated_assets,
                record_undo=True,
            )
            self.finished.clear()
            return self.public_state()

    def _unique_id(
        self,
        prefix: str,
        document: Mapping[str, object],
        *,
        include_proposals: bool = True,
    ) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", prefix).strip("_.:-") or "USER"
        identifiers = {str(node.get("id")) for node in _records(document, "nodes")}
        if include_proposals:
            identifiers.update(str(item.get("id")) for item in _records(document, "proposals"))
        identifiers.update(str(item.get("id")) for item in _records(document, "groups"))
        for index in range(1, 1_000_000):
            candidate = f"{normalized}_{index:04d}"
            if candidate not in identifiers:
                return candidate
        raise ReviewValidationError("Không tạo được mã mục mới duy nhất.")

    def _node_mask_canvas(self, node: Mapping[str, object]) -> Image.Image:
        bbox = _integer_bbox(node.get("bbox"), canvas=self._canvas, label="Layer")
        mask_ref = _mask_reference(node)
        canvas = Image.new("L", self._canvas, 0)
        if mask_ref is None:
            ImageDraw.Draw(canvas).rectangle(
                (bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=255
            )
            return canvas
        path = _resolve_bundle_asset(self.bundle_root, mask_ref, label="mask layer")
        with Image.open(path) as opened:
            opened.load()
            mask = opened.convert("L")
        if mask.size == self._canvas:
            return mask.copy()
        crop_size = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if mask.size != crop_size:
            raise ReviewValidationError("Kích thước mask không khớp layer.")
        canvas.paste(mask, (bbox[0], bbox[1]))
        return canvas

    def _save_mask_asset(
        self,
        mask_canvas: Image.Image,
        *,
        node_id: str,
        generated_assets: list[Path],
    ) -> tuple[list[int], str, str]:
        mask = mask_canvas.convert("L")
        tight = mask.getbbox()
        if tight is None:
            raise ReviewValidationError("Mask sau thao tác bị rỗng; hãy dùng nút Xóa layer nếu muốn bỏ hẳn.")
        cropped = mask.crop(tight)
        output = BytesIO()
        cropped.save(output, format="PNG", optimize=False, compress_level=6)
        payload = output.getvalue()
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", node_id)[:80] or "layer"
        relative = PurePosixPath("review_assets", "masks", f"{safe_id}_{secrets.token_hex(6)}.png")
        path = _resolve_bundle_asset(
            self.bundle_root, relative.as_posix(), label="mask mới", must_exist=False
        )
        _atomic_write_bytes(path, payload)
        generated_assets.append(path)
        return list(tight), relative.as_posix(), _sha256_bytes(payload)

    def source_asset(self) -> tuple[bytes, str]:
        with self._lock:
            if self._source_cache is not None:
                return self._source_cache
            suffix = self.source_path.suffix.lower()
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".tif": "image/tiff",
                ".tiff": "image/tiff",
            }.get(suffix, "application/octet-stream")
            value = self.source_path.read_bytes(), mime
            self._source_cache = value
            return value

    def mask_overlay_png(self, node_id: object) -> bytes:
        with self._lock:
            node = self._node(node_id)
            mask_ref = _mask_reference(node)
            key = str(node["id"]), mask_ref
            if key in self._mask_cache:
                return self._mask_cache[key]
            alpha = self._node_mask_canvas(node)
            overlay = Image.new("RGBA", self._canvas, (0, 170, 255, 0))
            overlay.putalpha(alpha.point(lambda value: min(118, value * 118 // 255)))
            output = BytesIO()
            overlay.save(output, format="PNG", compress_level=4)
            payload = output.getvalue()
            self._mask_cache[key] = payload
            return payload

    def public_state(self) -> dict[str, object]:
        with self._lock:
            nodes = sorted(
                self._nodes(),
                key=lambda item: (int(item.get("z_index", 0)), str(item.get("id"))),
            )
            proposals = sorted(
                self._proposals(),
                key=lambda item: (item.get("status") != "unresolved", str(item.get("id"))),
            )
            public_nodes: list[dict[str, object]] = []
            public_proposals: list[dict[str, object]] = []
            unresolved: list[dict[str, object]] = []
            number = 1
            for node in nodes:
                identifier = str(node["id"])
                status = str(node.get("review_status", "unresolved"))
                entry: dict[str, object] = {
                    "number": number,
                    "item_type": "node",
                    "id": identifier,
                    "name": str(node["name"]),
                    "kind": str(node["kind"]),
                    "kind_label": _KIND_LABELS[str(node["kind"])],
                    "bbox": list(node["bbox"]),
                    "status": status,
                    "confirmed": status in {"auto_confirmed", "user_confirmed"},
                    "mask_url": f"/asset/mask/{quote(identifier)}.png?token={quote(self.token)}",
                }
                review_note = _friendly_node_review_note(node)
                if review_note is not None:
                    entry["review_note"] = review_note
                public_nodes.append(entry)
                if status == "unresolved":
                    unresolved.append(
                        {"number": number, "item_type": "node", "id": identifier, "label": str(node["name"])}
                    )
                number += 1
            for proposal in proposals:
                identifier = str(proposal["id"])
                status = str(proposal.get("status", "unresolved"))
                label = _KIND_LABELS[str(proposal.get("kind_hint", "unknown"))]
                entry = {
                    "number": number,
                    "item_type": "proposal",
                    "id": identifier,
                    "name": f"Đề xuất {label}",
                    "kind": str(proposal.get("kind_hint", "unknown")),
                    "kind_label": label,
                    "bbox": list(proposal["bbox"]),
                    "status": status,
                    "confidence": round(float(proposal.get("confidence", 0.0)), 4),
                }
                public_proposals.append(entry)
                if status == "unresolved":
                    unresolved.append(
                        {"number": number, "item_type": "proposal", "id": identifier, "label": f"Đề xuất {label}"}
                    )
                number += 1
            groups = [
                {
                    "id": str(group["id"]),
                    "name": str(group["name"]),
                    "member_ids": list(group["member_ids"]),
                }
                for group in self._groups()
            ]
            return {
                "title": "V5 – Kiểm tra và tách layer",
                "canvas": list(self._canvas),
                "source_url": f"/asset/source?token={quote(self.token)}",
                "nodes": public_nodes,
                "proposals": public_proposals,
                "groups": groups,
                "unresolved": unresolved,
                "kinds": [
                    {"value": kind, "label": _KIND_LABELS[kind]}
                    for kind in sorted(_ELEMENT_KINDS, key=lambda value: _KIND_LABELS[value])
                ],
                "summary": {
                    "node_count": len(public_nodes),
                    "proposal_count": len(public_proposals),
                    "unresolved": len(unresolved),
                    "confirmed_nodes": sum(bool(item["confirmed"]) for item in public_nodes),
                },
                "can_undo": bool(self._undo),
                "can_redo": bool(self._redo),
                "revision": self._revision(),
                "finished": self.finished.is_set(),
            }

    def accept_proposal(self, proposal_id: object) -> dict[str, object]:
        def operation(candidate: dict[str, object], generated: list[Path]) -> None:
            proposal = self._proposal(proposal_id, candidate)
            if proposal.get("status", "unresolved") != "unresolved":
                raise ReviewValidationError("Chỉ nhận một đề xuất đang chờ duyệt.")
            node_id = self._unique_id(f"USER_{proposal['id']}", candidate)
            bbox = _integer_bbox(proposal.get("bbox"), canvas=self._canvas, label="Đề xuất")
            proposal_mask = _mask_reference(proposal)
            if proposal_mask is not None:
                mask = self._node_mask_canvas({"bbox": list(bbox), "mask": proposal_mask})
            else:
                mask = Image.new("L", self._canvas, 0)
                ImageDraw.Draw(mask).rectangle(
                    (bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=255
                )
            tight, mask_ref, mask_sha = self._save_mask_asset(
                mask, node_id=node_id, generated_assets=generated
            )
            z_values = [int(node.get("z_index", 0)) for node in self._nodes(candidate)]
            kind = str(proposal.get("kind_hint", "unknown"))
            node: dict[str, object] = {
                "id": node_id,
                "name": f"{_KIND_LABELS[kind]} – người dùng xác nhận",
                "kind": kind,
                "bbox": tight,
                "z_index": max(z_values, default=-1) + 1,
                "parent_id": None,
                "confidence": float(proposal.get("confidence", 0.0)),
                # Accepting a semantic proposal is deliberately separate from
                # confirming that its pixel mask is clean.
                "review_status": "unresolved",
                "move_safe": False,
                "mask": mask_ref,
                "mask_sha256": mask_sha,
                "metadata": {"created_from_proposal": str(proposal["id"])},
            }
            self._nodes(candidate).append(node)
            proposal["status"] = "assigned"
            proposal["owner_ids"] = [node_id]
            proposal["reason"] = None

        return self._mutate(operation)

    def reject_proposal(self, proposal_id: object) -> dict[str, object]:
        def operation(candidate: dict[str, object], _generated: list[Path]) -> None:
            proposal = self._proposal(proposal_id, candidate)
            if proposal.get("status", "unresolved") != "unresolved":
                raise ReviewValidationError("Chỉ từ chối một đề xuất đang chờ duyệt.")
            proposal["status"] = "rejected"
            proposal["owner_ids"] = []
            proposal["reason"] = "user_rejected_in_local_review"

        return self._mutate(operation)

    def accept_node(self, node_id: object) -> dict[str, object]:
        def operation(candidate: dict[str, object], _generated: list[Path]) -> None:
            node = self._node(node_id, candidate)
            old_band = _record_stack_priority(node)[0]
            node["review_status"] = "user_confirmed"
            node["review_source"] = "local-v5-review-ui"
            if _record_stack_priority(node)[0] != old_band:
                self._drop_groups_containing(candidate, {str(node["id"])})

        return self._mutate(operation)

    def edit_node(self, node_id: object, *, name: object, kind: object) -> dict[str, object]:
        clean_name = _clean_name(name)
        if kind not in _ELEMENT_KINDS:
            raise ReviewValidationError("Loại layer không hợp lệ.")

        def operation(candidate: dict[str, object], _generated: list[Path]) -> None:
            node = self._node(node_id, candidate)
            old_band = _record_stack_priority(node)[0]
            node["name"] = clean_name
            node["kind"] = str(kind)
            metadata = copy.deepcopy(node.get("metadata", {}))
            if not isinstance(metadata, dict):
                metadata = {}
            if (
                metadata.get("role") == "exact_source_remainder_above_clean_base"
                and str(kind) != "unknown"
            ):
                metadata.pop("role", None)
                metadata["promoted_from_technical_remainder"] = True
            node["metadata"] = metadata
            node["review_status"] = "user_confirmed"
            node["review_source"] = "local-v5-review-ui"
            if _record_stack_priority(node)[0] != old_band:
                self._drop_groups_containing(candidate, {str(node["id"])})

        return self._mutate(operation)

    def merge_group(self, member_ids: object, *, name: object = "Nhóm layer") -> dict[str, object]:
        if not isinstance(member_ids, list) or len(member_ids) < 2:
            raise ReviewValidationError("Hãy chọn ít nhất hai layer để gộp nhóm.")
        cleaned = [_clean_identifier(item, label="Mã layer") for item in member_ids]
        if len(set(cleaned)) != len(cleaned):
            raise ReviewValidationError("Danh sách gộp nhóm có layer bị trùng.")
        clean_name = _clean_name(name, label="Tên nhóm")

        def operation(candidate: dict[str, object], _generated: list[Path]) -> None:
            node_records = self._nodes(candidate)
            existing = {str(node["id"]) for node in node_records}
            if set(cleaned) - existing:
                raise ReviewValidationError("Có layer được chọn không còn tồn tại.")
            by_id = {str(node["id"]): node for node in node_records}
            parents = {by_id[identifier].get("parent_id") for identifier in cleaned}
            if len(parents) != 1:
                raise ReviewValidationError(
                    "Selected layers must share one parent so grouping cannot change the composite."
                )
            parent_id = next(iter(parents))
            siblings = sorted(
                [node for node in node_records if node.get("parent_id") == parent_id],
                key=_record_stack_priority,
            )
            positions = sorted(
                next(
                    index
                    for index, node in enumerate(siblings)
                    if str(node.get("id")) == identifier
                )
                for identifier in cleaned
            )
            if positions != list(range(positions[0], positions[-1] + 1)):
                raise ReviewValidationError(
                    "Selected layers must be contiguous so grouping cannot change the composite."
                )
            groups = self._groups(candidate)
            retained: list[dict[str, object]] = []
            for group in groups:
                if set(group.get("member_ids", [])).intersection(cleaned):
                    # The selected pixels remain atomic. Drop the old folder
                    # as a whole; retaining two sides of a split run would make
                    # a non-contiguous container and silently change z-order.
                    continue
                remaining = [item for item in group.get("member_ids", []) if item not in cleaned]
                if len(remaining) >= 2:
                    group["member_ids"] = remaining
                    retained.append(group)
            groups[:] = retained
            group_id = self._unique_id("USER_GROUP", candidate, include_proposals=False)
            groups.append({"id": group_id, "name": clean_name, "member_ids": cleaned})

        return self._mutate(operation)

    def add_rectangular_proposal(
        self,
        bbox: object,
        *,
        kind: object = "unknown",
    ) -> dict[str, object]:
        clean_bbox = list(_integer_bbox(bbox, canvas=self._canvas, label="Vùng mới"))
        if kind not in _ELEMENT_KINDS:
            raise ReviewValidationError("Loại vùng mới không hợp lệ.")

        def operation(candidate: dict[str, object], _generated: list[Path]) -> None:
            identifier = self._unique_id("USER_PROPOSAL", candidate)
            self._proposals(candidate).append(
                {
                    "id": identifier,
                    "source": "local_review_rectangle",
                    "kind_hint": str(kind),
                    "bbox": clean_bbox,
                    "confidence": 1.0,
                    "status": "unresolved",
                    "owner_ids": [],
                    "reason": None,
                    "evidence": {"created_by": "local-v5-review-ui"},
                }
            )

        return self._mutate(operation)

    def delete_node(self, node_id: object) -> dict[str, object]:
        selected = _clean_identifier(node_id, label="Mã layer")

        def operation(candidate: dict[str, object], _generated: list[Path]) -> None:
            removed = self._node(selected, candidate)
            replacement_parent = removed.get("parent_id")
            nodes = self._nodes(candidate)
            direct_children = {
                str(node.get("id"))
                for node in nodes
                if node.get("parent_id") == selected
            }
            # Removing a container reparents every direct child. Any folder
            # containing the deleted node or one of those children was valid
            # only in the old sibling list and can become non-contiguous after
            # reparenting. Dissolve it now; the final organizer recreates safe
            # automatic folders on resume without touching atomic leaves.
            self._drop_groups_containing(
                candidate,
                {selected, *direct_children},
            )
            nodes[:] = [node for node in nodes if str(node.get("id")) != selected]
            for node in nodes:
                if node.get("parent_id") == selected:
                    node["parent_id"] = replacement_parent
            for proposal in self._proposals(candidate):
                owners = [item for item in proposal.get("owner_ids", []) if item != selected]
                proposal["owner_ids"] = owners
                if proposal.get("status") == "assigned" and not owners:
                    proposal["status"] = "unresolved"
                    proposal["reason"] = "assigned_owner_deleted_during_review"
        return self._mutate(operation)

    def split_node_by_rectangle(self, node_id: object, rectangle: object) -> dict[str, object]:
        selected = _clean_identifier(node_id, label="Mã layer")
        split_box = _integer_bbox(rectangle, canvas=self._canvas, label="Khung tách")

        def operation(candidate: dict[str, object], generated: list[Path]) -> None:
            original = self._node(selected, candidate)
            if any(node.get("parent_id") == selected for node in self._nodes(candidate)):
                raise ReviewValidationError(
                    "Chỉ tách được layer lá; hãy bỏ nhóm cha hoặc chọn một chi tiết con."
                )
            source_mask = self._node_mask_canvas(original)
            inside = Image.new("L", self._canvas, 0)
            inside.paste(source_mask.crop(split_box), (split_box[0], split_box[1]))
            outside = source_mask.copy()
            ImageDraw.Draw(outside).rectangle(
                (split_box[0], split_box[1], split_box[2] - 1, split_box[3] - 1), fill=0
            )
            if inside.getbbox() is None or outside.getbbox() is None:
                raise ReviewValidationError(
                    "Khung tách phải cắt mask thành hai phần đều có điểm ảnh."
                )
            first_id = self._unique_id(f"{selected}_IN", candidate)
            # Reserve the first id while deriving the second.
            temporary = copy.deepcopy(original)
            temporary["id"] = first_id
            self._nodes(candidate).append(temporary)
            second_id = self._unique_id(f"{selected}_OUT", candidate)
            self._nodes(candidate).pop()

            first_bbox, first_ref, first_sha = self._save_mask_asset(
                inside, node_id=first_id, generated_assets=generated
            )
            second_bbox, second_ref, second_sha = self._save_mask_asset(
                outside, node_id=second_id, generated_assets=generated
            )
            base_z = int(original.get("z_index", 0))
            first = copy.deepcopy(original)
            first_metadata = copy.deepcopy(original.get("metadata", {}))
            if not isinstance(first_metadata, dict):
                first_metadata = {}
            first_metadata["derived_from_node_id"] = selected
            first_metadata.setdefault("lineage_root_id", selected)
            first.update(
                {
                    "id": first_id,
                    "name": f"{original['name']} – phần trong",
                    "bbox": first_bbox,
                    "mask": first_ref,
                    "mask_sha256": first_sha,
                    "z_index": base_z,
                    "review_status": "unresolved",
                    "metadata": first_metadata,
                }
            )
            second = copy.deepcopy(original)
            second_metadata = copy.deepcopy(first_metadata)
            second.update(
                {
                    "id": second_id,
                    "name": f"{original['name']} – phần ngoài",
                    "bbox": second_bbox,
                    "mask": second_ref,
                    "mask_sha256": second_sha,
                    "z_index": base_z + 1,
                    "review_status": "unresolved",
                    "metadata": second_metadata,
                }
            )
            nodes = self._nodes(candidate)
            for node in nodes:
                if str(node.get("id")) != selected and int(node.get("z_index", 0)) > base_z:
                    node["z_index"] = int(node.get("z_index", 0)) + 1
            position = next(index for index, node in enumerate(nodes) if str(node.get("id")) == selected)
            nodes[position : position + 1] = [first, second]
            for proposal in self._proposals(candidate):
                owners = list(proposal.get("owner_ids", []))
                if selected in owners:
                    expanded: list[str] = []
                    for owner in owners:
                        expanded.extend([first_id, second_id] if owner == selected else [owner])
                    proposal["owner_ids"] = expanded
            for group in self._groups(candidate):
                members = list(group.get("member_ids", []))
                if selected in members:
                    expanded_members: list[str] = []
                    for member in members:
                        expanded_members.extend([first_id, second_id] if member == selected else [member])
                    group["member_ids"] = expanded_members

        return self._mutate(operation)

    def brush_node(
        self,
        node_id: object,
        *,
        mode: object,
        points: object,
        radius: object,
    ) -> dict[str, object]:
        selected = _clean_identifier(node_id, label="Mã layer")
        if mode not in {"add", "subtract"}:
            raise ReviewValidationError("Cọ chỉ nhận chế độ thêm hoặc bớt mask.")
        if not isinstance(points, list) or not 1 <= len(points) <= MAX_BRUSH_POINTS:
            raise ReviewValidationError(
                f"Mỗi nét cọ phải có 1–{MAX_BRUSH_POINTS} điểm."
            )
        try:
            clean_radius = int(radius)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReviewValidationError("Bán kính cọ không hợp lệ.") from exc
        if not 1 <= clean_radius <= min(256, max(self._canvas)):
            raise ReviewValidationError("Bán kính cọ nằm ngoài giới hạn an toàn.")
        clean_points: list[tuple[int, int]] = []
        for value in points:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ReviewValidationError("Mỗi điểm cọ phải có tọa độ x, y.")
            try:
                x, y = int(round(float(value[0]))), int(round(float(value[1])))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ReviewValidationError("Tọa độ cọ không hợp lệ.") from exc
            if not (0 <= x < self._canvas[0] and 0 <= y < self._canvas[1]):
                raise ReviewValidationError("Nét cọ nằm ngoài ảnh.")
            clean_points.append((x, y))

        def operation(candidate: dict[str, object], generated: list[Path]) -> None:
            node = self._node(selected, candidate)
            mask = self._node_mask_canvas(node)
            draw = ImageDraw.Draw(mask)
            fill = 255 if mode == "add" else 0
            diameter = clean_radius * 2
            if len(clean_points) > 1:
                draw.line(clean_points, fill=fill, width=diameter, joint="curve")
            for x, y in clean_points:
                draw.ellipse(
                    (x - clean_radius, y - clean_radius, x + clean_radius, y + clean_radius),
                    fill=fill,
                )
            bbox, mask_ref, mask_sha = self._save_mask_asset(
                mask, node_id=selected, generated_assets=generated
            )
            node["bbox"] = bbox
            node["mask"] = mask_ref
            node["mask_sha256"] = mask_sha
            node["review_status"] = "unresolved"
            node["review_source"] = "local-v5-review-ui-brush"

        return self._mutate(operation)

    def undo(self) -> dict[str, object]:
        with self._lock:
            self._check_disk()
            if not self._undo:
                raise ReviewValidationError("Không còn thao tác để hoàn tác.")
            prior = copy.deepcopy(self.document)
            candidate = self._undo.pop()
            self._stamp(candidate, finished=False)
            try:
                self._commit(candidate, prior=prior, record_undo=False)
            except BaseException:
                self._undo.append(candidate)
                raise
            self._redo.append(prior)
            self.finished.clear()
            return self.public_state()

    def redo(self) -> dict[str, object]:
        with self._lock:
            self._check_disk()
            if not self._redo:
                raise ReviewValidationError("Không còn thao tác để làm lại.")
            prior = copy.deepcopy(self.document)
            candidate = self._redo.pop()
            self._stamp(candidate, finished=False)
            try:
                self._commit(candidate, prior=prior, record_undo=False)
            except BaseException:
                self._redo.append(candidate)
                raise
            self._undo.append(prior)
            self.finished.clear()
            return self.public_state()

    def mark_finished(self) -> dict[str, object]:
        """Flush the completed flag before the HTTP response is emitted."""

        with self._lock:
            self._check_disk()
            prior = copy.deepcopy(self.document)
            candidate = copy.deepcopy(self.document)
            self._stamp(candidate, finished=True)
            self._commit(candidate, prior=prior, record_undo=True)
            self.finished.set()
            return self.public_state()


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>V5 – Kiểm tra layer</title>
  <style nonce="__NONCE__">
    :root { color-scheme:light; font-family:"Segoe UI",Arial,sans-serif; color:#172235; background:#eef3f8; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; }
    header { padding:14px 20px; background:#12223a; color:white; display:flex; flex-wrap:wrap; align-items:center; gap:12px; }
    header h1 { font-size:20px; margin:0 auto 0 0; }
    header button { background:#314766; color:white; }
    button, input, select { font:inherit; }
    button { border:0; border-radius:9px; padding:9px 12px; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    #finish { background:#16834f; }
    #summary { color:#cfdaea; font-weight:650; }
    main { display:grid; grid-template-columns:minmax(0,1fr) 350px; gap:16px; padding:16px; max-width:1800px; margin:auto; }
    .card { background:white; border-radius:14px; box-shadow:0 9px 28px #1a2d4720; overflow:hidden; }
    .toolbar { display:flex; flex-wrap:wrap; gap:8px; padding:12px; border-bottom:1px solid #e2e8f0; align-items:center; }
    .toolbar button { background:#e8eff8; color:#20344f; }
    .toolbar button.active { background:#126bd1; color:white; }
    .toolbar label { display:flex; align-items:center; gap:6px; color:#53647a; font-size:13px; }
    .toolbar .technical-toggle { padding:7px 9px; border-radius:9px; background:#fff5d9; color:#5f4b13; font-weight:700; }
    .technical-toggle input { width:18px; height:18px; margin:0; }
    #stageWrap { overflow:auto; background:#202937; padding:14px; min-height:520px; max-height:calc(100vh - 150px); }
    #stage { position:relative; width:max-content; max-width:100%; margin:auto; line-height:0; }
    #source { display:block; max-width:100%; max-height:calc(100vh - 190px); width:auto; height:auto; }
    #overlay { position:absolute; inset:0; width:100%; height:100%; cursor:crosshair; touch-action:none; }
    aside { display:grid; align-content:start; gap:14px; max-height:calc(100vh - 95px); overflow:auto; }
    .section { padding:15px; }
    .section h2 { margin:0 0 11px; font-size:17px; }
    .muted { color:#68798f; font-size:13px; line-height:1.4; }
    #unresolved, #allNodes, #groups { display:grid; gap:6px; max-height:250px; overflow:auto; }
    .item { width:100%; text-align:left; background:#f2f6fb; color:#23364f; border:1px solid #dce5f0; }
    .item.selected { border-color:#0878da; background:#e6f2ff; }
    .item.proposal { border-left:5px solid #e69b16; }
    .item.node { border-left:5px solid #168cca; }
    #detail { display:grid; gap:10px; }
    #detail label { display:grid; gap:5px; color:#56677d; font-size:13px; font-weight:650; }
    #detail input, #detail select, #groupName { border:1px solid #b9c6d8; border-radius:8px; padding:8px; width:100%; }
    .actions { display:flex; flex-wrap:wrap; gap:7px; }
    .primary { background:#126bd1; color:white; }
    .accept { background:#16834f; color:white; }
    .reject, .danger { background:#fbe5e7; color:#922938; }
    #message { min-height:42px; padding:10px 12px; background:#edf5ff; color:#294b70; border-radius:9px; line-height:1.35; }
    .node-check { display:flex; align-items:center; gap:7px; padding:5px 1px; font-size:13px; }
    .node-check input { width:17px; height:17px; }
    @media (max-width:980px) { main { grid-template-columns:1fr; } aside { max-height:none; } #stageWrap { max-height:70vh; } }
  </style>
</head>
<body>
  <header>
    <h1>V5 – Kiểm tra và tách layer</h1>
    <span id="summary">Đang tải…</span>
    <button id="undo" disabled>↶ Hoàn tác</button>
    <button id="redo" disabled>↷ Làm lại</button>
    <button id="finish">Lưu và hoàn tất</button>
  </header>
  <main>
    <section class="card">
      <div class="toolbar">
        <button id="modeSelect" class="active">Chọn mục</button>
        <button id="modeAddRect">Khoanh vùng còn thiếu</button>
        <button id="modeSplit">Tách layer bằng khung</button>
        <button id="modeBrushAdd">Cọ thêm mask</button>
        <button id="modeBrushSub">Cọ bớt mask</button>
        <label class="technical-toggle"><input id="showPending" type="checkbox">Hiện toàn bộ mục chờ duyệt</label>
        <label class="technical-toggle"><input id="showTechnical" type="checkbox">Hiện cả mục kỹ thuật/đã xử lý</label>
        <label>Cỡ cọ <input id="radius" type="range" min="1" max="120" value="18"><span id="radiusValue">18</span> px</label>
      </div>
      <div id="stageWrap"><div id="stage"><img id="source" alt="Ảnh nguồn cần tách layer"><canvas id="overlay"></canvas></div></div>
    </section>
    <aside>
      <section class="card section">
        <h2>Cần người dùng kiểm tra</h2>
        <div id="unresolved"></div>
        <p class="muted" id="emptyUnresolved" hidden>Không còn mục chưa xác nhận.</p>
      </section>
      <section class="card section">
        <h2>Mục đang chọn</h2>
        <div id="detail"><p class="muted">Chọn một mục trong danh sách cần kiểm tra.</p></div>
      </section>
      <section class="card section">
        <h2>Gộp thành nhóm thao tác</h2>
        <input id="groupName" value="Nhóm layer" maxlength="240" aria-label="Tên nhóm">
        <div id="allNodes"></div>
        <p class="muted"><strong>Nhóm đã tạo trong PSD/ORA</strong></p>
        <div id="groups"></div>
        <div class="actions"><button id="merge" class="primary">Gộp các layer đã chọn</button></div>
        <p class="muted">Gộp chỉ tạo nhóm để di chuyển thuận tiện; các layer nhỏ bên trong vẫn được giữ nguyên.</p>
      </section>
      <section class="card section"><div id="message">Mọi thao tác được lưu ngay vào checkpoint.</div></section>
    </aside>
  </main>
  <script nonce="__NONCE__">
    const TOKEN=__TOKEN_JSON__;
    let state=null, selected=null, mode="select", dragStart=null, brushPoints=[], busy=false, maskImage=null;
    const $=id=>document.getElementById(id), canvas=$("overlay"), ctx=canvas.getContext("2d"), source=$("source");
    async function api(path, body) {
      let response;
      try {
        response=await fetch(path,{method:body===undefined?"GET":"POST",headers:body===undefined?{"X-V5-Review-Token":TOKEN}:{"Content-Type":"application/json","X-V5-Review-Token":TOKEN},body:body===undefined?undefined:JSON.stringify(body),cache:"no-store"});
      } catch(error) { throw new Error("Mất kết nối với V5. Các thao tác trước đó đã được lưu ngay; hãy chạy lại lệnh để mở checkpoint."); }
      let payload={}; try { payload=await response.json(); } catch(error) { throw new Error("V5 trả về dữ liệu không hoàn chỉnh."); }
      if(!response.ok) throw new Error(payload.message||"Không thực hiện được thao tác.");
      return payload;
    }
    function allItems(){ return [...state.nodes,...state.proposals]; }
    function selectedItem(){ return selected ? allItems().find(item=>item.item_type===selected.item_type&&item.id===selected.id) : null; }
    function isSelectedItem(item){ return selected?.item_type===item.item_type&&selected.id===item.id; }
    function canvasModeMessage(){
      if($("showTechnical").checked) return "Đang hiện cả proposal kỹ thuật và mục đã xử lý.";
      if($("showPending").checked) return "Đang hiện toàn bộ mục còn chờ duyệt.";
      return "Canvas chỉ hiện mục đang chọn.";
    }
    function canvasItems(){
      if($("showTechnical").checked) return allItems();
      if($("showPending").checked) {
        return allItems().filter(item=>item.status==="unresolved"||isSelectedItem(item));
      }
      // The normal workspace stays visually clean even when an exhaustive
      // inventory contains hundreds of pending regions. Selecting a row or a
      // canvas item shows that one overlay; the two explicit toggles reveal
      // pending work or the complete technical ledger on demand.
      const current=selectedItem();
      return current?[current]:[];
    }
    function setMessage(value,error=false){ $("message").textContent=value; $("message").style.background=error?"#fff0f1":"#edf5ff"; $("message").style.color=error?"#8a2531":"#294b70"; }
    function pick(item){ selected={item_type:item.item_type,id:item.id}; loadMask(); renderLists(); renderDetail(); draw(); }
    function setMode(value){ mode=value; dragStart=null; brushPoints=[]; for(const [id,key] of [["modeSelect","select"],["modeAddRect","add_rect"],["modeSplit","split"],["modeBrushAdd","brush_add"],["modeBrushSub","brush_sub"]]) $(id).classList.toggle("active",key===mode); setMessage(value==="select"?"Chọn mục trong danh sách bên phải; canvas sẽ hiện riêng mục đó.":"Kéo trực tiếp trên ảnh để thực hiện thao tác."); }
    function loadMask(){ maskImage=null; const item=selectedItem(); if(!item||item.item_type!=="node"){draw();return;} const image=new Image(); image.onload=()=>{ if(selectedItem()?.id===item.id){maskImage=image;draw();} }; image.src=item.mask_url; }
    function color(item){ if(item.item_type==="proposal") return item.status==="unresolved"?"#ff9d00":"#a1722c"; return item.status==="unresolved"?"#00a8ff":"#16a36a"; }
    function draw(preview=null){
      if(!state||!canvas.width)return; ctx.clearRect(0,0,canvas.width,canvas.height);
      if(maskImage) ctx.drawImage(maskImage,0,0,canvas.width,canvas.height);
      ctx.font=`bold ${Math.max(12,Math.round(Math.min(canvas.width,canvas.height)/75))}px Segoe UI`;
      ctx.textAlign="center";ctx.textBaseline="middle";
      for(const item of canvasItems()){
        const [x0,y0,x1,y1]=item.bbox, active=selected&&selected.id===item.id&&selected.item_type===item.item_type;
        ctx.strokeStyle=active?"#ff2d55":color(item);ctx.lineWidth=active?4:2;ctx.setLineDash(item.item_type==="proposal"?[8,5]:[]);ctx.strokeRect(x0,y0,x1-x0,y1-y0);ctx.setLineDash([]);
        const r=Math.max(10,Math.min(22,Math.min(canvas.width,canvas.height)/55));const cx=Math.max(r,x0+r),cy=Math.max(r,y0+r);
        ctx.fillStyle=active?"#ff2d55":color(item);ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.fill();ctx.fillStyle="white";ctx.fillText(String(item.number),cx,cy+0.5);
      }
      if(preview){ctx.strokeStyle="#ff2d55";ctx.lineWidth=3;ctx.setLineDash([9,6]);ctx.strokeRect(preview[0],preview[1],preview[2]-preview[0],preview[3]-preview[1]);ctx.setLineDash([]);}
    }
    function renderLists(){
      $("summary").textContent=`${state.summary.node_count} layer · ${state.summary.unresolved} mục cần kiểm tra`;
      $("undo").disabled=!state.can_undo; $("redo").disabled=!state.can_redo;
      const unresolved=$("unresolved"); unresolved.replaceChildren(); $("emptyUnresolved").hidden=state.unresolved.length!==0;
      for(const row of state.unresolved){const button=document.createElement("button");button.className=`item ${row.item_type}`+(selected?.id===row.id&&selected.item_type===row.item_type?" selected":"");button.textContent=`${row.number}. ${row.label}`;button.onclick=()=>pick(allItems().find(item=>item.id===row.id&&item.item_type===row.item_type));unresolved.append(button);}
      const checks=$("allNodes");const chosen=new Set([...checks.querySelectorAll("input:checked")].map(input=>input.value));checks.replaceChildren();
      for(const node of state.nodes){const label=document.createElement("label");label.className="node-check";const input=document.createElement("input");input.type="checkbox";input.value=node.id;input.checked=chosen.has(node.id);const span=document.createElement("span");span.textContent=`${node.number}. ${node.name}`;label.append(input,span);checks.append(label);}
      const groups=$("groups");groups.replaceChildren();for(const group of state.groups){const row=document.createElement("div");row.className="item";row.textContent=`${group.name} · ${group.member_ids.length} layer`;groups.append(row);}
    }
    function renderDetail(){
      const root=$("detail"),item=selectedItem();root.replaceChildren();if(!item){const p=document.createElement("p");p.className="muted";p.textContent="Chọn một mục trong danh sách cần kiểm tra.";root.append(p);return;}
      const title=document.createElement("strong");title.textContent=`${item.number}. ${item.name}`;root.append(title);
      if(item.item_type==="proposal"){
        const p=document.createElement("p");p.className="muted";p.textContent=`Loại dự kiến: ${item.kind_label} · độ tin cậy ${Math.round(item.confidence*100)}%`;const actions=document.createElement("div");actions.className="actions";
        const accept=document.createElement("button");accept.className="accept";accept.textContent="Nhận thành layer";accept.disabled=item.status!=="unresolved";accept.onclick=()=>act("accept_proposal",{proposal_id:item.id});
        const reject=document.createElement("button");reject.className="reject";reject.textContent="Không phải layer";reject.disabled=item.status!=="unresolved";reject.onclick=()=>act("reject_proposal",{proposal_id:item.id});actions.append(accept,reject);root.append(p,actions);return;
      }
      const nameLabel=document.createElement("label");nameLabel.textContent="Tên dễ nhớ";const name=document.createElement("input");name.value=item.name;name.maxLength=240;nameLabel.append(name);
      if(item.review_note){const note=document.createElement("p");note.className="muted";note.textContent=item.review_note;root.append(note);}
      const kindLabel=document.createElement("label");kindLabel.textContent="Loại chi tiết";const kind=document.createElement("select");for(const option of state.kinds){const node=document.createElement("option");node.value=option.value;node.textContent=option.label;node.selected=option.value===item.kind;kind.append(node);}kindLabel.append(kind);
      const actions=document.createElement("div");actions.className="actions";const save=document.createElement("button");save.className="primary";save.textContent="Lưu tên và loại";save.onclick=()=>act("edit_node",{node_id:item.id,name:name.value,kind:kind.value});
      const accept=document.createElement("button");accept.className="accept";accept.textContent="Xác nhận layer sạch";accept.onclick=()=>act("accept_node",{node_id:item.id});
      const remove=document.createElement("button");remove.className="danger";remove.textContent="Xóa layer";remove.onclick=()=>{if(confirm(`Xóa layer “${item.name}”?`))act("delete_node",{node_id:item.id});};actions.append(save,accept,remove);root.append(nameLabel,kindLabel,actions);
    }
    function render(){ if(selected&&!selectedItem())selected=null; renderLists();renderDetail();draw(); }
    async function act(action,extra={}){if(busy)return;busy=true;setMessage("Đang lưu…");try{state=await api("/api/action",{action,...extra});render();setMessage("Đã lưu an toàn vào checkpoint.");}catch(error){setMessage(error.message,true);}finally{busy=false;}}
    function point(event){const rect=canvas.getBoundingClientRect();return [Math.max(0,Math.min(canvas.width-1,(event.clientX-rect.left)*canvas.width/rect.width)),Math.max(0,Math.min(canvas.height-1,(event.clientY-rect.top)*canvas.height/rect.height))];}
    function hitTest(position){
      const [x,y]=position;
      const area=item=>(item.bbox[2]-item.bbox[0])*(item.bbox[3]-item.bbox[1]);
      return canvasItems()
        .filter(item=>x>=item.bbox[0]&&x<item.bbox[2]&&y>=item.bbox[1]&&y<item.bbox[3])
        .sort((a,b)=>Number(!isSelectedItem(a))-Number(!isSelectedItem(b))||Number(a.status!=="unresolved")-Number(b.status!=="unresolved")||Number(a.item_type!=="node")-Number(b.item_type!=="node")||area(a)-area(b))[0];
    }
    canvas.onpointerdown=event=>{if(busy)return;canvas.setPointerCapture(event.pointerId);const p=point(event);if(mode==="select"){const hit=hitTest(p);if(hit)pick(hit);return;}if(mode==="add_rect"||mode==="split"){dragStart=p;draw([p[0],p[1],p[0],p[1]]);return;}const item=selectedItem();if(!item||item.item_type!=="node"){setMessage("Hãy chọn một layer trước khi dùng cọ.",true);return;}brushPoints=[p];};
    canvas.onpointermove=event=>{const p=point(event);if(dragStart){draw([Math.min(dragStart[0],p[0]),Math.min(dragStart[1],p[1]),Math.max(dragStart[0],p[0]),Math.max(dragStart[1],p[1])]);}else if(brushPoints.length){brushPoints.push(p);ctx.strokeStyle=mode==="brush_add"?"#00b7ff":"#ff2d55";ctx.lineWidth=Number($("radius").value)*2;ctx.lineCap="round";const prev=brushPoints[brushPoints.length-2];ctx.beginPath();ctx.moveTo(prev[0],prev[1]);ctx.lineTo(p[0],p[1]);ctx.stroke();}};
    canvas.onpointerup=event=>{const p=point(event);if(dragStart){const box=[Math.round(Math.min(dragStart[0],p[0])),Math.round(Math.min(dragStart[1],p[1])),Math.round(Math.max(dragStart[0],p[0])+1),Math.round(Math.max(dragStart[1],p[1])+1)];dragStart=null;if(box[2]-box[0]<2||box[3]-box[1]<2){draw();return;}if(mode==="add_rect")act("add_rect_proposal",{bbox:box,kind:"unknown"});else{const item=selectedItem();if(!item||item.item_type!=="node")setMessage("Hãy chọn layer cần tách trước.",true);else act("split_node",{node_id:item.id,bbox:box});}}else if(brushPoints.length){brushPoints.push(p);const item=selectedItem(),points=brushPoints;brushPoints=[];act("brush",{node_id:item.id,mode:mode==="brush_add"?"add":"subtract",points,radius:Number($("radius").value)});}};
    source.onload=()=>{canvas.width=source.naturalWidth;canvas.height=source.naturalHeight;draw();};
    $("radius").oninput=()=>$("radiusValue").textContent=$("radius").value;
    $("showPending").onchange=()=>{draw();setMessage(canvasModeMessage());};
    $("showTechnical").onchange=()=>{draw();setMessage(canvasModeMessage());};
    $("modeSelect").onclick=()=>setMode("select");$("modeAddRect").onclick=()=>setMode("add_rect");$("modeSplit").onclick=()=>setMode("split");$("modeBrushAdd").onclick=()=>setMode("brush_add");$("modeBrushSub").onclick=()=>setMode("brush_sub");
    $("undo").onclick=async()=>{if(busy)return;busy=true;try{state=await api("/api/undo",{});render();setMessage("Đã hoàn tác.");}catch(error){setMessage(error.message,true);}finally{busy=false;}};
    $("redo").onclick=async()=>{if(busy)return;busy=true;try{state=await api("/api/redo",{});render();setMessage("Đã làm lại.");}catch(error){setMessage(error.message,true);}finally{busy=false;}};
    $("merge").onclick=()=>{const ids=[...$("allNodes").querySelectorAll("input:checked")].map(input=>input.value);act("merge_group",{member_ids:ids,name:$("groupName").value});};
    $("finish").onclick=async()=>{if(busy)return;if(state.summary.unresolved&&!confirm(`Còn ${state.summary.unresolved} mục chưa kiểm tra. Vẫn lưu và hoàn tất?`))return;busy=true;try{state=await api("/api/finish",{});render();setMessage("Đã ghi xong checkpoint. Bạn có thể đóng trang này.");}catch(error){setMessage(error.message,true);}finally{busy=false;}};
    api("/api/state").then(value=>{state=value;source.src=state.source_url;render();}).catch(error=>setMessage(error.message,true));
  </script>
</body>
</html>"""


def render_review_html(token: str) -> tuple[str, str]:
    nonce = secrets.token_urlsafe(18)
    html = _HTML_TEMPLATE.replace("__NONCE__", nonce).replace(
        "__TOKEN_JSON__", json.dumps(str(token), ensure_ascii=False)
    )
    return html, nonce


class _ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], session: ReviewSession) -> None:
        self.session = session
        super().__init__(address, _ReviewRequestHandler)

    @property
    def origin(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


class _ReviewRequestHandler(BaseHTTPRequestHandler):
    server: _ReviewHTTPServer
    _shutdown_after_response = False

    def finish(self) -> None:
        """Close/flush the response first, then stop the one-shot server."""

        shutdown = self._shutdown_after_response
        try:
            super().finish()
        finally:
            if shutdown:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, _format: str, *args: object) -> None:
        return

    def _secure_headers(
        self, *, content_type: str, length: int, nonce: str | None = None
    ) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        if nonce:
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; "
                f"style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'; "
                "img-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'none'",
            )

    def _host_is_loopback(self) -> bool:
        expected = f"{LOOPBACK_HOST}:{self.server.server_address[1]}"
        return secrets.compare_digest(self.headers.get("Host", ""), expected)

    def _authorized(self, parsed: Any, *, mutation: bool = False) -> bool:
        if not self._host_is_loopback():
            return False
        header = self.headers.get(TOKEN_HEADER, "")
        query = parse_qs(parsed.query).get("token", [""])[0]
        supplied = header if mutation or header else query
        return bool(supplied) and secrets.compare_digest(supplied, self.server.session.token)

    def _json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._secure_headers(content_type="application/json; charset=utf-8", length=len(data))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._json(status, {"ok": False, "code": code, "message": message})

    def _body(self) -> dict[str, object]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ReviewValidationError("Thiếu Content-Length.")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ReviewValidationError("Content-Length không hợp lệ.") from exc
        if not 0 <= length <= MAX_REQUEST_BYTES:
            raise ReviewValidationError("Yêu cầu gửi lên quá lớn.")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReviewValidationError("Body phải là JSON UTF-8 hợp lệ.") from exc
        if not isinstance(value, dict):
            raise ReviewValidationError("Body phải là JSON object.")
        return value

    def _perform_action(self, body: Mapping[str, object]) -> dict[str, object]:
        action = str(body.get("action", ""))
        session = self.server.session
        if action == "accept_proposal":
            return session.accept_proposal(body.get("proposal_id"))
        if action == "reject_proposal":
            return session.reject_proposal(body.get("proposal_id"))
        if action == "accept_node":
            return session.accept_node(body.get("node_id"))
        if action == "edit_node":
            return session.edit_node(
                body.get("node_id"), name=body.get("name"), kind=body.get("kind")
            )
        if action == "merge_group":
            return session.merge_group(body.get("member_ids"), name=body.get("name", "Nhóm layer"))
        if action == "split_node":
            return session.split_node_by_rectangle(body.get("node_id"), body.get("bbox"))
        if action == "add_rect_proposal":
            return session.add_rectangular_proposal(body.get("bbox"), kind=body.get("kind", "unknown"))
        if action == "delete_node":
            return session.delete_node(body.get("node_id"))
        if action == "brush":
            return session.brush_node(
                body.get("node_id"),
                mode=body.get("mode"),
                points=body.get("points"),
                radius=body.get("radius"),
            )
        raise ReviewValidationError("Thao tác V5 không được hỗ trợ.")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not self._authorized(parsed):
            self._error(HTTPStatus.FORBIDDEN, "forbidden", "Liên kết duyệt không hợp lệ.")
            return
        if parsed.path == "/":
            html, nonce = render_review_html(self.server.session.token)
            data = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self._secure_headers(
                content_type="text/html; charset=utf-8", length=len(data), nonce=nonce
            )
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/state":
            self._json(HTTPStatus.OK, self.server.session.public_state())
            return
        if parsed.path == "/asset/source":
            try:
                data, mime = self.server.session.source_asset()
            except OSError:
                self._error(HTTPStatus.NOT_FOUND, "source-not-found", "Không đọc được ảnh nguồn.")
                return
            self.send_response(HTTPStatus.OK)
            self._secure_headers(content_type=mime, length=len(data))
            self.end_headers()
            self.wfile.write(data)
            return
        mask_match = re.fullmatch(r"/asset/mask/([A-Za-z0-9_.:%-]+)\.png", parsed.path)
        if mask_match:
            from urllib.parse import unquote

            try:
                data = self.server.session.mask_overlay_png(unquote(mask_match.group(1)))
            except ReviewUIError as exc:
                self._error(HTTPStatus.NOT_FOUND, "mask-not-found", str(exc))
                return
            self.send_response(HTTPStatus.OK)
            self._secure_headers(content_type="image/png", length=len(data))
            self.end_headers()
            self.wfile.write(data)
            return
        self._error(HTTPStatus.NOT_FOUND, "not-found", "Không tìm thấy trang.")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not self._authorized(parsed, mutation=True):
            self._error(HTTPStatus.FORBIDDEN, "forbidden", "Phiên duyệt không hợp lệ.")
            return
        if self.headers.get_content_type() != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "invalid-content-type", "Chỉ nhận JSON.")
            return
        try:
            body = self._body()
            if parsed.path == "/api/action":
                self._json(HTTPStatus.OK, self._perform_action(body))
                return
            if parsed.path == "/api/undo":
                if body:
                    raise ReviewValidationError("Hoàn tác không nhận dữ liệu bổ sung.")
                self._json(HTTPStatus.OK, self.server.session.undo())
                return
            if parsed.path == "/api/redo":
                if body:
                    raise ReviewValidationError("Làm lại không nhận dữ liệu bổ sung.")
                self._json(HTTPStatus.OK, self.server.session.redo())
                return
            if parsed.path == "/api/finish":
                if body:
                    raise ReviewValidationError("Hoàn tất không nhận dữ liệu bổ sung.")
                state = self.server.session.mark_finished()
                self._json(HTTPStatus.OK, state)
                self._shutdown_after_response = True
                return
            self._error(HTTPStatus.NOT_FOUND, "not-found", "Không tìm thấy API.")
        except ReviewConflictError as exc:
            self._error(HTTPStatus.CONFLICT, "file-conflict", str(exc))
        except (ReviewValidationError, ReviewPathError) as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid-action", str(exc))
        except OSError:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "save-failed", "Không ghi được checkpoint.")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method-not-allowed", "Không hỗ trợ CORS.")


@dataclass(slots=True)
class ReviewServerHandle:
    server: _ReviewHTTPServer
    session: ReviewSession
    url: str

    def serve_forever(self) -> None:
        self.server.serve_forever(poll_interval=0.15)

    def close(self) -> None:
        self.server.server_close()


def create_review_server(
    checkpoint_path: Path | str,
    *,
    host: str = LOOPBACK_HOST,
    port: int = 0,
) -> ReviewServerHandle:
    if host != LOOPBACK_HOST:
        raise ReviewPathError("V5 review chỉ được bind vào 127.0.0.1.")
    try:
        selected_port = int(port)
    except (TypeError, ValueError) as exc:
        raise ReviewValidationError("Port không hợp lệ.") from exc
    if not 0 <= selected_port <= 65_535:
        raise ReviewValidationError("Port phải từ 0 đến 65535.")
    session = ReviewSession.open(checkpoint_path)
    server = _ReviewHTTPServer((host, selected_port), session)
    url = f"{server.origin}/?token={quote(session.token)}"
    return ReviewServerHandle(server=server, session=session, url=url)


def run_review_ui(
    checkpoint_path: Path | str,
    *,
    port: int = 0,
    open_browser: Callable[[str], object] | None = None,
) -> dict[str, object]:
    handle = create_review_server(checkpoint_path, port=port)
    opener = open_browser or webbrowser.open
    try:
        opener(handle.url)
        handle.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        handle.close()
    return copy.deepcopy(handle.session.document)


__all__ = [
    "LOOPBACK_HOST",
    "REVIEW_SCHEMA",
    "ReviewConflictError",
    "ReviewPathError",
    "ReviewServerHandle",
    "ReviewSession",
    "ReviewUIError",
    "ReviewValidationError",
    "build_review_checkpoint",
    "create_review_server",
    "render_review_html",
    "run_review_ui",
    "validate_review_checkpoint",
    "write_review_checkpoint",
]
