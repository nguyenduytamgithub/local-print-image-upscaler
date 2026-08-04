"""Conservative Vietnamese text normalization and protection helpers.

Only canonical NFC normalization is automatic.  Spelling models are allowed to
produce :class:`LanguageProposal` objects, never authoritative replacement
text.  Prices, numbers, phone numbers, SKUs, addresses and caller-supplied brand
names are masked verbatim before any model call.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable, Literal, Sequence

from .types import LanguageProposal, ProtectedSpan, ProtectedText


class ProtectedTextError(ValueError):
    """Raised when a protected placeholder cannot be restored losslessly."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    start: int
    end: int
    kind: str
    priority: int


_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?84|0)(?:[\s.()\-]*\d){8,10}(?!\w)",
    re.UNICODE,
)

_PRICE_RE = re.compile(
    r"(?<!\w)(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)(?:[.,]\d+)?"
    r"\s*(?:₫|đ|vnđ|vnd)(?!\w)(?:\s*/\s*[^\s,;|]+)?",
    re.IGNORECASE | re.UNICODE,
)

_SKU_CUE_RE = re.compile(
    r"(?<!\w)(?:sku|mã(?:\s+(?:sp|sản\s+phẩm|hàng))?)\s*[:#-]?\s*"
    r"(?=[A-Za-z0-9._/\-]{3,40}(?!\w))(?=[A-Za-z0-9._/\-]*\d)"
    r"(?P<token>[A-Za-z0-9](?:[A-Za-z0-9._/\-]{1,38}[A-Za-z0-9])?)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)

_SKU_SEPARATED_RE = re.compile(
    r"(?<!\w)(?=[A-Z0-9._/\-]{4,40}(?!\w))(?=[A-Z0-9._/\-]*[A-Z])"
    r"[A-Z0-9]{1,16}(?:[._/\-][A-Z0-9]{1,20})+(?!\w)",
    re.UNICODE,
)

_SKU_COMPACT_RE = re.compile(
    r"(?<!\w)(?=[A-Z0-9]{5,20}(?!\w))(?=[A-Z0-9]*[A-Z])"
    r"(?=[A-Z0-9]*\d)[A-Z0-9]{5,20}(?!\w)",
    re.UNICODE,
)

_BRAND_MARK_RE = re.compile(
    r"(?<!\w)[^\W\d_][\w&+.'\- ]{0,48}?\s*[®™](?!\w)",
    re.UNICODE,
)

_ADDRESS_NUMBERED_RE = re.compile(
    r"(?<!\w)(?:số\s*)?\d{1,5}[a-z]?(?:[/.-]\d+[a-z]?)?\s+"
    r"(?=[^;\n|]{0,150}(?:phường|xã|quận|huyện|thị\s+xã|thành\s+phố|"
    r"tp\.?(?=\s|$)))"
    r"[^,;\n|]{2,64}(?:,\s*[^,;\n|]{1,64}){1,3}",
    re.IGNORECASE | re.UNICODE,
)

_ADDRESS_CUE_RE = re.compile(
    r"(?<!\w)(?:số\s*)?\d{1,5}[a-z]?(?:[/.-]\d+[a-z]?)?\s+"
    r"(?:đường|phố|ấp|thôn|khu\s+phố)\s+[^;\n|]{2,120}",
    re.IGNORECASE | re.UNICODE,
)

_NUMBER_RE = re.compile(
    r"(?<!\d)\d+(?:[.,:/\-]\d+)*(?:\s*(?:%|ml|cl|l|g|kg|mg|mm|cm|m|"
    r"điểm|chai|túi|thùng|cuộn|bọc|combo))?",
    re.IGNORECASE | re.UNICODE,
)


def normalize_nfc(text: str) -> str:
    """Return canonical NFC text without compatibility (NFKC) rewriting."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return unicodedata.normalize("NFC", text)


def _literal_candidates(
    text: str,
    values: Iterable[str],
    *,
    kind: str,
    priority: int,
) -> list[_Candidate]:
    result: list[_Candidate] = []
    seen: set[str] = set()
    for raw_value in values:
        value = normalize_nfc(str(raw_value)).strip()
        folded = value.casefold()
        if not value or folded in seen:
            continue
        seen.add(folded)
        pattern = re.compile(
            rf"(?<!\w){re.escape(value)}(?!\w)",
            re.IGNORECASE | re.UNICODE,
        )
        result.extend(
            _Candidate(match.start(), match.end(), kind, priority)
            for match in pattern.finditer(text)
        )
    return result


def _regex_candidates(
    text: str,
    pattern: re.Pattern[str],
    *,
    kind: str,
    priority: int,
) -> list[_Candidate]:
    return [
        _Candidate(match.start(), match.end(), kind, priority)
        for match in pattern.finditer(text)
        if match.end() > match.start()
    ]


def _sku_cue_candidates(text: str) -> list[_Candidate]:
    return [
        _Candidate(match.start("token"), match.end("token"), "sku", 75)
        for match in _SKU_CUE_RE.finditer(text)
        if match.end("token") > match.start("token")
    ]


def _select_non_overlapping(candidates: Iterable[_Candidate]) -> list[_Candidate]:
    # Semantic kinds beat the generic number fallback.  Within a priority, the
    # longest match wins so a complete phone/price/SKU remains one locked span.
    ranked = sorted(
        candidates,
        key=lambda item: (-item.priority, -(item.end - item.start), item.start, item.end),
    )
    selected: list[_Candidate] = []
    for item in ranked:
        if any(item.start < other.end and other.start < item.end for other in selected):
            continue
        selected.append(item)
    return sorted(selected, key=lambda item: (item.start, item.end))


def find_protected_spans(
    text: str,
    *,
    brands: Iterable[str] = (),
    addresses: Iterable[str] = (),
) -> tuple[ProtectedSpan, ...]:
    """Locate source content that must round-trip byte-for-text after NFC.

    Brand and address detection is most reliable when the caller supplies its
    approved glossary.  The regex address detector is intentionally cautious
    and only recognizes conventional, visibly structured address lines.
    """

    source = normalize_nfc(text)
    candidates: list[_Candidate] = []
    candidates.extend(
        _literal_candidates(source, brands, kind="brand", priority=110)
    )
    candidates.extend(
        _literal_candidates(source, addresses, kind="address", priority=110)
    )
    candidates.extend(
        _regex_candidates(source, _BRAND_MARK_RE, kind="brand", priority=100)
    )
    candidates.extend(
        _regex_candidates(source, _PHONE_RE, kind="phone", priority=90)
    )
    candidates.extend(
        _regex_candidates(source, _PRICE_RE, kind="price", priority=80)
    )
    candidates.extend(_sku_cue_candidates(source))
    candidates.extend(
        _regex_candidates(source, _SKU_SEPARATED_RE, kind="sku", priority=70)
    )
    candidates.extend(
        _regex_candidates(source, _SKU_COMPACT_RE, kind="sku", priority=70)
    )
    candidates.extend(
        _regex_candidates(source, _ADDRESS_NUMBERED_RE, kind="address", priority=60)
    )
    candidates.extend(
        _regex_candidates(source, _ADDRESS_CUE_RE, kind="address", priority=60)
    )
    candidates.extend(
        _regex_candidates(source, _NUMBER_RE, kind="number", priority=10)
    )
    return tuple(
        ProtectedSpan(
            start=item.start,
            end=item.end,
            text=source[item.start : item.end],
            kind=item.kind,
        )
        for item in _select_non_overlapping(candidates)
    )


def _validate_spans(source: str, spans: Sequence[ProtectedSpan]) -> None:
    previous_end = 0
    for span in spans:
        if span.start < previous_end:
            raise ProtectedTextError("protected spans overlap or are not sorted")
        if span.end > len(source) or source[span.start : span.end] != span.text:
            raise ProtectedTextError("protected span does not match NFC source text")
        previous_end = span.end


def protect_text(
    text: str,
    spans: Sequence[ProtectedSpan] | None = None,
    *,
    brands: Iterable[str] = (),
    addresses: Iterable[str] = (),
    placeholder_style: Literal["plain", "t5"] = "plain",
) -> ProtectedText:
    """Replace protected spans with unique placeholders for a model call.

    ``t5`` placeholders use the model's existing ``<extra_id_N>`` vocabulary.
    Restoration is strict: missing, duplicate or reordered placeholders cause a
    fail-closed error instead of leaking a model rewrite into the proposal.
    """

    source = normalize_nfc(text)
    located = tuple(spans) if spans is not None else find_protected_spans(
        source, brands=brands, addresses=addresses
    )
    _validate_spans(source, located)
    if placeholder_style == "t5" and len(located) > 96:
        raise ProtectedTextError("the local T5 model exposes only 96 safe sentinels")

    masked_parts: list[str] = []
    protected: list[ProtectedSpan] = []
    cursor = 0
    for index, span in enumerate(located):
        placeholder = (
            f"<extra_id_{index}>"
            if placeholder_style == "t5"
            else f"__V7_PROTECTED_{index:04d}__"
        )
        if placeholder in source:
            raise ProtectedTextError("source text already contains a V7 placeholder")
        masked_parts.append(source[cursor : span.start])
        masked_parts.append(placeholder)
        protected.append(
            ProtectedSpan(
                start=span.start,
                end=span.end,
                text=span.text,
                kind=span.kind,
                placeholder=placeholder,
            )
        )
        cursor = span.end
    masked_parts.append(source[cursor:])
    return ProtectedText(source, "".join(masked_parts), tuple(protected))


def restore_protected_text(
    masked_text: str,
    spans: Sequence[ProtectedSpan],
    *,
    strict: bool = True,
) -> str:
    """Restore a model result and fail closed if any lock token was damaged."""

    result = normalize_nfc(masked_text)
    last_position = -1
    for span in spans:
        placeholder = span.placeholder
        if not placeholder:
            raise ProtectedTextError("protected span has no placeholder")
        count = result.count(placeholder)
        if strict and count != 1:
            raise ProtectedTextError(
                f"placeholder {placeholder!r} occurred {count} times instead of once"
            )
        position = result.find(placeholder)
        if position < 0:
            if strict:
                raise ProtectedTextError(f"placeholder {placeholder!r} is missing")
            continue
        if strict and position < last_position:
            raise ProtectedTextError("the language model reordered protected content")
        last_position = position
        result = result.replace(placeholder, span.text, 1)

    if strict and ("__V7_PROTECTED_" in result or re.search(r"<extra_id_\d+>", result)):
        raise ProtectedTextError("unresolved or invented protection placeholder")
    return normalize_nfc(result)


def protected_content_preserved(
    candidate_text: str,
    spans: Sequence[ProtectedSpan],
    *,
    brands: Iterable[str] = (),
    addresses: Iterable[str] = (),
) -> bool:
    """Return whether critical values are identical, complete, and in order."""

    candidate = normalize_nfc(candidate_text)
    cursor = 0
    for span in spans:
        position = candidate.find(span.text, cursor)
        if position < 0:
            return False
        cursor = position + len(span.text)
    # A generator must not introduce a new price, phone number, identifier or
    # other numeric fact either.  Comparing the full semantic signature catches
    # additions as well as replacements that happen to retain the old value.
    candidate_spans = find_protected_spans(
        candidate,
        brands=brands,
        addresses=addresses,
    )
    source_signature = tuple((span.kind, span.text) for span in spans)
    candidate_signature = tuple((span.kind, span.text) for span in candidate_spans)
    return candidate_signature == source_signature


def _deduplicate_reasons(reasons: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_reason in reasons:
        reason = str(raw_reason).strip()
        if reason and reason not in seen:
            seen.add(reason)
            result.append(reason)
    return tuple(result)


def build_conservative_proposal(
    source_text: str,
    candidate_text: str | None = None,
    *,
    confidence: float = 0.0,
    brands: Iterable[str] = (),
    addresses: Iterable[str] = (),
    protected_spans: Sequence[ProtectedSpan] | None = None,
    reasons: Iterable[str] = (),
    model: str | None = None,
) -> LanguageProposal:
    """Package a model candidate without authorizing it for rendering.

    If a candidate loses or reorders critical source content, it is discarded
    and the NFC source becomes the displayed proposal.  ``requires_approval``
    remains true even for high-confidence or unchanged model output.
    """

    source = str(source_text)
    normalized = normalize_nfc(source)
    brand_values = tuple(brands)
    address_values = tuple(addresses)
    spans = tuple(protected_spans) if protected_spans is not None else find_protected_spans(
        normalized, brands=brand_values, addresses=address_values
    )
    _validate_spans(normalized, spans)
    reason_list = list(reasons)
    if source != normalized:
        reason_list.append("nfc-normalized")
    reason_list.extend(f"protected-{span.kind}" for span in spans)

    candidate = normalized if candidate_text is None else normalize_nfc(candidate_text)
    safe = protected_content_preserved(
        candidate,
        spans,
        brands=brand_values,
        addresses=address_values,
    )
    try:
        bounded_confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        bounded_confidence = 0.0
    if not safe:
        candidate = normalized
        bounded_confidence = 0.0
        reason_list.append("protected-content-mismatch")
    changed = candidate != normalized
    if changed:
        reason_list.append("model-suggested-change")
    else:
        reason_list.append("no-content-change")
    reason_list.append("human-approval-required")

    return LanguageProposal(
        source_text=source,
        normalized_text=normalized,
        proposed_text=candidate,
        confidence=bounded_confidence,
        changed=changed,
        requires_approval=True,
        safe=safe,
        protected_spans=spans,
        reasons=_deduplicate_reasons(reason_list),
        model=model,
    )


__all__ = [
    "ProtectedTextError",
    "build_conservative_proposal",
    "find_protected_spans",
    "normalize_nfc",
    "protect_text",
    "protected_content_preserved",
    "restore_protected_text",
]
