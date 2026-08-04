"""Small, JSON-safe value types shared by the V7 text pipeline.

The language stage deliberately has no ``apply`` method.  It can describe a
candidate correction, but only the caller (after explicit human approval) may
choose text for rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, Mapping, Sequence


BBox = tuple[int, int, int, int]
Point = tuple[float, float]
ReviewStatus = Literal["green", "yellow", "red"]


def _confidence(value: float) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be a finite number between 0 and 1")
    return result


def _bbox(value: Sequence[int]) -> BBox:
    if len(value) != 4:
        raise ValueError("bbox must contain x0, y0, x1, y1")
    x0, y0, x1, y1 = (int(item) for item in value)
    if x1 < x0 or y1 < y0:
        raise ValueError("bbox end coordinates must not precede start coordinates")
    return x0, y0, x1, y1


def _polygon(value: Sequence[Sequence[float]]) -> tuple[Point, ...]:
    points: list[Point] = []
    for point in value:
        if len(point) != 2:
            raise ValueError("each polygon point must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not isfinite(x) or not isfinite(y):
            raise ValueError("polygon coordinates must be finite")
        points.append((x, y))
    return tuple(points)


@dataclass(frozen=True, slots=True)
class OCRObservation:
    """One OCR engine's reading of one text region."""

    engine: str
    variant: str
    text: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.engine.strip():
            raise ValueError("OCR observation engine must not be empty")
        if not self.variant.strip():
            raise ValueError("OCR observation variant must not be empty")
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "variant": self.variant,
            "text": self.text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OCRObservation":
        return cls(
            engine=str(value["engine"]),
            variant=str(value.get("variant", "default")),
            text=str(value.get("text", "")),
            confidence=float(value.get("confidence", 0.0)),
        )

@dataclass(frozen=True, slots=True)
class ProtectedSpan:
    """A verbatim source span that a language model must never rewrite."""

    start: int
    end: int
    text: str
    kind: str
    placeholder: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid protected span offsets")
        if self.end - self.start != len(self.text):
            raise ValueError("protected span offsets do not match its text")
        if not self.kind:
            raise ValueError("protected span kind must not be empty")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "kind": self.kind,
        }
        if self.placeholder is not None:
            result["placeholder"] = self.placeholder
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProtectedSpan":
        placeholder = value.get("placeholder")
        return cls(
            start=int(value["start"]),
            end=int(value["end"]),
            text=str(value["text"]),
            kind=str(value["kind"]),
            placeholder=None if placeholder is None else str(placeholder),
        )


@dataclass(frozen=True, slots=True)
class ProtectedText:
    """NFC source text plus a reversible, model-facing masked form."""

    source_text: str
    masked_text: str
    spans: tuple[ProtectedSpan, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "masked_text": self.masked_text,
            "spans": [span.to_dict() for span in self.spans],
        }


@dataclass(frozen=True, slots=True)
class LanguageProposal:
    """A non-binding correction suggestion that always requires review."""

    source_text: str
    normalized_text: str
    proposed_text: str
    confidence: float
    changed: bool
    requires_approval: bool = True
    safe: bool = True
    protected_spans: tuple[ProtectedSpan, ...] = ()
    reasons: tuple[str, ...] = ()
    model: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if not self.requires_approval:
            raise ValueError("V7 language proposals must require explicit approval")
        if self.changed != (self.proposed_text != self.normalized_text):
            raise ValueError("changed must compare proposed_text with normalized_text")

    @property
    def candidate_text(self) -> str:
        """Compatibility alias for callers that call proposals candidates."""

        return self.proposed_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "normalized_text": self.normalized_text,
            "proposed_text": self.proposed_text,
            "confidence": self.confidence,
            "changed": self.changed,
            "requires_approval": True,
            "safe": self.safe,
            "protected_spans": [span.to_dict() for span in self.protected_spans],
            "reasons": list(self.reasons),
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LanguageProposal":
        normalized = str(value.get("normalized_text", value.get("source_text", "")))
        proposed = str(value.get("proposed_text", value.get("candidate_text", normalized)))
        return cls(
            source_text=str(value.get("source_text", normalized)),
            normalized_text=normalized,
            proposed_text=proposed,
            confidence=float(value.get("confidence", 0.0)),
            changed=bool(value.get("changed", proposed != normalized)),
            requires_approval=bool(value.get("requires_approval", True)),
            safe=bool(value.get("safe", True)),
            protected_spans=tuple(
                ProtectedSpan.from_dict(item)
                for item in value.get("protected_spans", ())
            ),
            reasons=tuple(str(item) for item in value.get("reasons", ())),
            model=None if value.get("model") is None else str(value["model"]),
        )


@dataclass(slots=True)
class TextRegion:
    """OCR evidence and review state for a geometrically stable text region."""

    region_id: str
    bbox: BBox
    polygon: tuple[Point, ...] = ()
    observations: list[OCRObservation] = field(default_factory=list)
    selected_text: str | None = None
    proposal: LanguageProposal | None = None
    status: ReviewStatus = "red"
    critical: bool = False
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id must not be empty")
        self.bbox = _bbox(self.bbox)
        self.polygon = _polygon(self.polygon)
        self.observations = [
            item if isinstance(item, OCRObservation) else OCRObservation.from_dict(item)
            for item in self.observations
        ]
        if isinstance(self.proposal, Mapping):
            self.proposal = LanguageProposal.from_dict(self.proposal)
        if self.status not in ("green", "yellow", "red"):
            raise ValueError("status must be green, yellow, or red")
        self.reasons = [str(item) for item in self.reasons]

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bbox": list(self.bbox),
            "polygon": [list(point) for point in self.polygon],
            "observations": [item.to_dict() for item in self.observations],
            "selected_text": self.selected_text,
            "proposal": None if self.proposal is None else self.proposal.to_dict(),
            "status": self.status,
            "critical": self.critical,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TextRegion":
        proposal = value.get("proposal")
        return cls(
            region_id=str(value["region_id"]),
            bbox=_bbox(value["bbox"]),
            polygon=_polygon(value.get("polygon", ())),
            observations=[
                OCRObservation.from_dict(item)
                for item in value.get("observations", ())
            ],
            selected_text=(
                None if value.get("selected_text") is None else str(value["selected_text"])
            ),
            proposal=(
                None
                if proposal is None
                else LanguageProposal.from_dict(proposal)
            ),
            status=str(value.get("status", "red")),  # type: ignore[arg-type]
            critical=bool(value.get("critical", False)),
            reasons=[str(item) for item in value.get("reasons", ())],
        )
