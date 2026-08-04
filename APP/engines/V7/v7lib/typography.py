"""Windows font discovery, style estimation and final-scale text rendering.

V7 never asks an image generator to invent letters.  Approved Unicode text is
rendered with a real local OpenType/TrueType font through Pillow/FreeType and,
when available, RAQM/HarfBuzz.  Font embedding permissions are inspected from
the OS/2 ``fsType`` field so later SVG/PDF exporters do not silently package a
restricted font.
"""

from __future__ import annotations

import math
import os
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from fontTools.ttLib import TTCollection, TTFont, TTLibError
from PIL import Image, ImageDraw, ImageFont, features as pillow_features


class TypographyError(RuntimeError):
    """Raised when V7 cannot render approved text faithfully."""


@dataclass(frozen=True, slots=True)
class FontEmbeddingPolicy:
    fs_type: int
    restricted: bool
    installable: bool
    preview_print: bool
    editable: bool
    embeddable: bool
    no_subsetting: bool
    bitmap_only: bool


@dataclass(frozen=True, slots=True)
class FontRecord:
    path: Path
    face_index: int
    family: str
    subfamily: str
    full_name: str
    postscript_name: str | None
    embedding: FontEmbeddingPolicy
    codepoints: frozenset[int] = field(repr=False)

    @property
    def identifier(self) -> str:
        suffix = f"#{self.face_index}" if self.face_index else ""
        return f"{self.path}{suffix}"


@dataclass(frozen=True, slots=True)
class FontMatch:
    font: FontRecord
    rendered_text: str
    normalization: str
    coverage: float
    exact_coverage: bool
    score: float


@dataclass(frozen=True, slots=True)
class FontVariationAxis:
    tag: str
    minimum: float
    default: float
    maximum: float


@dataclass(frozen=True, slots=True)
class TextStyleEstimate:
    fill_rgb: tuple[int, int, int]
    stroke_rgb: tuple[int, int, int] | None
    stroke_width_source_px: float
    confidence: float
    foreground_mask: np.ndarray = field(repr=False, compare=False)
    report: dict[str, object] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class TextStyle:
    font: FontRecord
    font_size_px: int | None = None
    fill_rgb: tuple[int, int, int] = (0, 0, 0)
    stroke_rgb: tuple[int, int, int] | None = None
    stroke_width_px: int = 0
    horizontal_align: str = "center"
    vertical_align: str = "middle"
    language: str = "vi"
    direction: str = "ltr"
    line_spacing_px: int = 4
    features: tuple[str, ...] = ()
    variation_axes: tuple[float, ...] | None = None


@dataclass(slots=True)
class TextRender:
    rgba: Image.Image
    position: tuple[int, int]
    visible_bbox: tuple[int, int, int, int]
    source_text: str
    rendered_text: str
    normalization: str
    font: FontRecord
    font_size_px: int
    fill_rgb: tuple[int, int, int]
    stroke_rgb: tuple[int, int, int] | None
    stroke_width_px: int
    layout_engine: str
    variation_axes: tuple[float, ...] | None
    svg_text_origin: tuple[float, float]
    report: dict[str, object]


def font_embedding_policy(fs_type: int | None) -> FontEmbeddingPolicy:
    """Interpret OpenType OS/2 embedding flags without broadening permission."""

    value = int(fs_type or 0)
    restricted = bool(value & 0x0002)
    preview_print = bool(value & 0x0004)
    editable = bool(value & 0x0008)
    installable = (value & 0x000F) == 0
    return FontEmbeddingPolicy(
        fs_type=value,
        restricted=restricted,
        installable=installable,
        preview_print=preview_print,
        editable=editable,
        embeddable=not restricted,
        no_subsetting=bool(value & 0x0100),
        bitmap_only=bool(value & 0x0200),
    )


def _name_value(font: TTFont, name_id: int, fallback: str) -> str:
    table = font.get("name")
    if table is None:
        return fallback
    records = [record for record in table.names if int(record.nameID) == name_id]
    records.sort(
        key=lambda record: (
            0 if (record.platformID == 3 and record.langID in {0x0409, 0}) else 1,
            0 if record.platformID == 3 else 1,
        )
    )
    for record in records:
        try:
            value = record.toUnicode().strip("\x00 ")
        except (UnicodeDecodeError, AttributeError):
            continue
        if value:
            return value
    return fallback


def inspect_font(path: Path | str, face_index: int = 0) -> FontRecord:
    """Read names, Unicode cmap and embedding policy from one font face."""

    font_path = Path(path).resolve()
    if not font_path.is_file():
        raise TypographyError(f"Font file not found: {font_path}")
    try:
        font = TTFont(str(font_path), fontNumber=int(face_index), lazy=False)
    except (TTLibError, OSError, IndexError) as exc:
        raise TypographyError(f"Cannot inspect font face {font_path}#{face_index}") from exc
    try:
        family = _name_value(font, 1, font_path.stem)
        subfamily = _name_value(font, 2, "Regular")
        full_name = _name_value(font, 4, f"{family} {subfamily}".strip())
        postscript = _name_value(font, 6, "") or None
        best_cmap = font.getBestCmap() or {}
        os2 = font.get("OS/2")
        fs_type = int(getattr(os2, "fsType", 0)) if os2 is not None else 0
        return FontRecord(
            path=font_path,
            face_index=int(face_index),
            family=family,
            subfamily=subfamily,
            full_name=full_name,
            postscript_name=postscript,
            embedding=font_embedding_policy(fs_type),
            codepoints=frozenset(int(codepoint) for codepoint in best_cmap),
        )
    finally:
        font.close()


def _font_face_count(path: Path) -> int:
    if path.suffix.lower() not in {".ttc", ".otc"}:
        return 1
    try:
        collection = TTCollection(str(path), lazy=True)
    except (TTLibError, OSError):
        return 0
    try:
        return len(collection.fonts)
    finally:
        collection.close()


def _registry_font_paths(font_directories: Sequence[Path]) -> set[Path]:
    try:
        import winreg
    except ImportError:
        return set()

    discovered: set[Path] = set()
    keys = (
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows NT\CurrentVersion\Fonts"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
    )
    for hive, key_name in keys:
        try:
            key = winreg.OpenKey(hive, key_name)
        except OSError:
            continue
        try:
            index = 0
            while True:
                try:
                    _name, raw_value, _kind = winreg.EnumValue(key, index)
                except OSError:
                    break
                index += 1
                if not isinstance(raw_value, str) or not raw_value.strip():
                    continue
                value = Path(os.path.expandvars(raw_value.strip()))
                candidates = [value] if value.is_absolute() else [root / value for root in font_directories]
                discovered.update(candidate for candidate in candidates if candidate.is_file())
        finally:
            winreg.CloseKey(key)
    return discovered


def windows_font_directories() -> tuple[Path, ...]:
    values: list[Path] = []
    windows = os.environ.get("WINDIR")
    local = os.environ.get("LOCALAPPDATA")
    roaming = os.environ.get("APPDATA")
    if windows:
        values.append(Path(windows) / "Fonts")
    if local:
        values.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    if roaming:
        values.append(Path(roaming) / "Microsoft" / "Windows" / "Fonts")
    unique: list[Path] = []
    seen: set[str] = set()
    for path in values:
        key = str(path).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return tuple(unique)


def _discover_font_files(extra_directories: Sequence[Path]) -> list[Path]:
    directories = (*windows_font_directories(), *extra_directories)
    extensions = {".ttf", ".otf", ".ttc", ".otc"}
    files = _registry_font_paths(directories)
    for directory in directories:
        if not directory.is_dir():
            continue
        try:
            files.update(
                item
                for item in directory.iterdir()
                if item.is_file() and item.suffix.lower() in extensions
            )
        except OSError:
            continue
    return sorted(files, key=lambda item: str(item).casefold())


@lru_cache(maxsize=1)
def _default_font_cache() -> tuple[FontRecord, ...]:
    return _discover_fonts(())


def _discover_fonts(extra_directories: Sequence[Path]) -> tuple[FontRecord, ...]:
    records: list[FontRecord] = []
    seen: set[tuple[str, int]] = set()
    for path in _discover_font_files(extra_directories):
        for face_index in range(_font_face_count(path)):
            key = (str(path.resolve()).casefold(), face_index)
            if key in seen:
                continue
            seen.add(key)
            try:
                records.append(inspect_font(path, face_index))
            except TypographyError:
                continue
    records.sort(
        key=lambda record: (
            record.family.casefold(),
            record.subfamily.casefold(),
            str(record.path).casefold(),
            record.face_index,
        )
    )
    return tuple(records)


def discover_windows_fonts(
    extra_directories: Sequence[Path | str] = (),
) -> tuple[FontRecord, ...]:
    """Enumerate system, per-user and explicitly supplied OpenType fonts."""

    if not extra_directories:
        return _default_font_cache()
    extras = tuple(Path(value).resolve() for value in extra_directories)
    return _discover_fonts(extras)


def _meaningful_codepoints(text: str) -> set[int]:
    return {
        ord(character)
        for character in text
        if not character.isspace() and unicodedata.category(character)[0] != "C"
    }


def text_for_font(text: str, font: FontRecord) -> tuple[str, str, float]:
    """Choose NFC when possible, otherwise a canonically equivalent NFD form."""

    source = unicodedata.normalize("NFC", text)
    best_text = source
    best_form = "NFC"
    best_coverage = -1.0
    for form in ("NFC", "NFD"):
        candidate = unicodedata.normalize(form, source)
        points = _meaningful_codepoints(candidate)
        coverage = 1.0 if not points else len(points & font.codepoints) / len(points)
        if coverage > best_coverage:
            best_text, best_form, best_coverage = candidate, form, coverage
    return best_text, best_form, float(best_coverage)


def select_font(
    text: str,
    fonts: Sequence[FontRecord] | None = None,
    *,
    preferred_families: Sequence[str] = (),
    preferred_style: str | None = None,
    require_full_coverage: bool = True,
    require_embeddable: bool = False,
) -> FontMatch:
    """Select a real font deterministically; never accept a missing-glyph box."""

    candidates = tuple(fonts) if fonts is not None else discover_windows_fonts()
    if not candidates:
        raise TypographyError("No usable Windows OpenType/TrueType fonts were found.")
    preferred = [value.casefold().strip() for value in preferred_families if value.strip()]
    style_tokens = set((preferred_style or "").casefold().replace("-", " ").split())
    scored: list[FontMatch] = []
    for font in candidates:
        if require_embeddable and not font.embedding.embeddable:
            continue
        rendered_text, normalization, coverage = text_for_font(text, font)
        family = font.family.casefold()
        subfamily_tokens = set(font.subfamily.casefold().replace("-", " ").split())
        family_bonus = 0.0
        for rank, wanted in enumerate(preferred):
            if family == wanted:
                family_bonus = 180.0 - rank * 5.0
                break
            if wanted in family or family in wanted:
                family_bonus = max(family_bonus, 80.0 - rank * 3.0)
        style_bonus = 12.0 * len(style_tokens & subfamily_tokens)
        slant_tokens = {"italic", "oblique", "slanted"}
        unwanted_slant = bool(subfamily_tokens & slant_tokens) and not bool(
            style_tokens & slant_tokens
        )
        slant_penalty = 30.0 if unwanted_slant else 0.0
        regular_bonus = 1.0 if not style_tokens and "regular" in subfamily_tokens else 0.0
        scored.append(
            FontMatch(
                font=font,
                rendered_text=rendered_text,
                normalization=normalization,
                coverage=coverage,
                exact_coverage=coverage >= 1.0,
                score=(
                    coverage * 1000.0
                    + family_bonus
                    + style_bonus
                    + regular_bonus
                    - slant_penalty
                ),
            )
        )
    if not scored:
        raise TypographyError("No font satisfies the requested embedding policy.")
    selected = max(
        scored,
        key=lambda item: (
            item.score,
            item.font.family.casefold(),
            item.font.subfamily.casefold(),
            str(item.font.path).casefold(),
            -item.font.face_index,
        ),
    )
    if require_full_coverage and not selected.exact_coverage:
        raise TypographyError(
            f"No discovered font covers every glyph in {unicodedata.normalize('NFC', text)!r}."
        )
    return selected


def _validate_rgb_tuple(value: Sequence[int], label: str) -> tuple[int, int, int]:
    if len(value) != 3:
        raise TypographyError(f"{label} must contain exactly three channels.")
    result = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in result):
        raise TypographyError(f"{label} channels must be from 0 through 255.")
    return result  # type: ignore[return-value]


def _pillow_horizontal_align(value: str) -> str:
    folded = value.casefold()
    aliases = {"start": "left", "end": "right"}
    folded = aliases.get(folded, folded)
    if folded not in {"left", "center", "right", "justify"}:
        raise TypographyError(
            "horizontal_align must be left/start, center, right/end or justify."
        )
    return folded


def _validate_text_style(style: TextStyle) -> None:
    _validate_rgb_tuple(style.fill_rgb, "fill_rgb")
    if style.stroke_rgb is not None:
        _validate_rgb_tuple(style.stroke_rgb, "stroke_rgb")
    if style.font_size_px is not None and int(style.font_size_px) < 1:
        raise TypographyError("font_size_px must be positive when supplied.")
    if int(style.stroke_width_px) < 0:
        raise TypographyError("stroke_width_px cannot be negative.")
    if int(style.line_spacing_px) < 0:
        raise TypographyError("line_spacing_px cannot be negative.")
    if style.variation_axes is not None and any(
        not math.isfinite(float(value)) for value in style.variation_axes
    ):
        raise TypographyError("variation_axes must contain only finite values.")
    _pillow_horizontal_align(style.horizontal_align)
    if style.vertical_align.casefold() not in {
        "top",
        "start",
        "middle",
        "center",
        "bottom",
        "end",
    }:
        raise TypographyError(
            "vertical_align must be top/start, middle/center or bottom/end."
        )
    if style.direction.casefold() not in {"ltr", "rtl", "ttb"}:
        raise TypographyError("direction must be ltr, rtl or ttb.")


def _bbox_mask(
    image_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    left, top, right, bottom = (
        max(0, int(bbox[0])),
        max(0, int(bbox[1])),
        min(width, int(bbox[2])),
        min(height, int(bbox[3])),
    )
    result = np.zeros((height, width), dtype=bool)
    if right <= left or bottom <= top:
        return result
    crop = image_rgb[top:bottom, left:right].astype(np.int16)
    border = np.concatenate(
        (crop[0], crop[-1], crop[:, 0], crop[:, -1]), axis=0
    )
    background = np.median(border, axis=0)
    difference = np.linalg.norm(crop - background, axis=2)
    difference_u8 = np.clip(np.rint(difference), 0, 255).astype(np.uint8)
    threshold, foreground = cv2.threshold(
        difference_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    foreground = difference_u8 > max(12.0, float(threshold))
    foreground = cv2.morphologyEx(
        foreground.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
    ).astype(bool)
    result[top:bottom, left:right] = foreground
    return result


def estimate_text_style(
    image_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    bbox: tuple[int, int, int, int] | None = None,
) -> TextStyleEstimate:
    """Estimate fill/stroke colours from a text mask without changing pixels."""

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise TypographyError("image_rgb must be an HxWx3 uint8 array.")
    if mask is None:
        if bbox is None:
            raise TypographyError("mask or bbox is required for style estimation.")
        foreground = _bbox_mask(image, bbox)
        mask_source = "bbox_border_contrast"
    else:
        foreground = np.asarray(mask).astype(bool, copy=False)
        if foreground.shape != image.shape[:2]:
            raise TypographyError("text mask shape does not match the image.")
        mask_source = "provided_mask"
    if not foreground.any():
        raise TypographyError("text style cannot be estimated from an empty mask.")

    distance = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 5)
    pixels = image[foreground]
    distances = distance[foreground]
    quantized = (pixels // 16).astype(np.int16)
    colours, inverse, counts = np.unique(
        quantized, axis=0, return_inverse=True, return_counts=True
    )
    top_indices = np.argsort(counts)[-min(8, len(counts)) :]
    clusters: list[dict[str, object]] = []
    for index in top_indices:
        membership = inverse == index
        actual = pixels[membership]
        mean_distance = float(np.mean(distances[membership]))
        clusters.append(
            {
                "index": int(index),
                "count": int(membership.sum()),
                "mean_distance": mean_distance,
                "rgb": tuple(int(round(value)) for value in np.median(actual, axis=0)),
            }
        )
    fill_cluster = max(
        clusters,
        key=lambda item: (
            float(item["mean_distance"]),
            math.log1p(int(item["count"])),
        ),
    )
    fill_rgb = _validate_rgb_tuple(fill_cluster["rgb"], "fill_rgb")  # type: ignore[arg-type]
    fill_distance = float(fill_cluster["mean_distance"])
    minimum_stroke_count = max(3, int(len(pixels) * 0.025))
    stroke_candidates = []
    for item in clusters:
        if item is fill_cluster or int(item["count"]) < minimum_stroke_count:
            continue
        rgb = np.asarray(item["rgb"], dtype=np.float32)
        separation = float(np.linalg.norm(rgb - np.asarray(fill_rgb, dtype=np.float32)))
        if separation < 38.0 or float(item["mean_distance"]) >= fill_distance * 0.92:
            continue
        stroke_candidates.append((item, separation))

    stroke_rgb: tuple[int, int, int] | None = None
    stroke_width = 0.0
    stroke_separation = 0.0
    if stroke_candidates:
        selected_stroke, stroke_separation = max(
            stroke_candidates,
            key=lambda pair: (int(pair[0]["count"]), pair[1]),
        )
        stroke_rgb = _validate_rgb_tuple(
            selected_stroke["rgb"], "stroke_rgb"  # type: ignore[arg-type]
        )
        membership = inverse == int(selected_stroke["index"])
        stroke_width = float(np.percentile(distances[membership], 90.0))
        stroke_width = float(np.clip(stroke_width, 1.0, 32.0))

    confidence = min(1.0, 0.45 + math.log1p(len(pixels)) / 20.0)
    if stroke_rgb is not None:
        confidence *= min(1.0, 0.65 + stroke_separation / 160.0)
    report = {
        "mask_source": mask_source,
        "foreground_pixels": int(foreground.sum()),
        "palette_cluster_count": len(clusters),
        "fill_mean_distance_px": round(fill_distance, 4),
        "stroke_detected": stroke_rgb is not None,
        "stroke_colour_distance": round(stroke_separation, 4),
        "colour_space": "sRGB uint8; Euclidean estimate",
    }
    return TextStyleEstimate(
        fill_rgb=fill_rgb,
        stroke_rgb=stroke_rgb,
        stroke_width_source_px=stroke_width,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        foreground_mask=foreground.copy(),
        report=report,
    )


def _layout_engine() -> tuple[ImageFont.Layout, str]:
    try:
        if pillow_features.check_feature("raqm"):
            return ImageFont.Layout.RAQM, "RAQM/HarfBuzz"
    except (ValueError, AttributeError):
        pass
    return ImageFont.Layout.BASIC, "Pillow BASIC fallback"


@lru_cache(maxsize=512)
def _variation_axes_for_file(path: str, face_index: int) -> tuple[FontVariationAxis, ...]:
    try:
        value = TTFont(path, fontNumber=int(face_index), lazy=True)
    except (OSError, TTLibError, ValueError):
        return ()
    try:
        if "fvar" not in value:
            return ()
        return tuple(
            FontVariationAxis(
                tag=str(axis.axisTag),
                minimum=float(axis.minValue),
                default=float(axis.defaultValue),
                maximum=float(axis.maxValue),
            )
            for axis in value["fvar"].axes
        )
    finally:
        value.close()


def font_variation_axes(font: FontRecord) -> tuple[FontVariationAxis, ...]:
    """Return ordered OpenType fvar axes used by FreeType/Pillow."""

    return _variation_axes_for_file(str(font.path.resolve()), int(font.face_index))


def load_pillow_font(
    font: FontRecord,
    size_px: int,
    *,
    variation_axes: Sequence[float] | None = None,
) -> tuple[ImageFont.FreeTypeFont, str]:
    if int(size_px) < 1:
        raise TypographyError("font size must be positive.")
    engine, label = _layout_engine()
    try:
        loaded = ImageFont.truetype(
            str(font.path),
            size=int(size_px),
            index=font.face_index,
            layout_engine=engine,
        )
    except (OSError, ValueError) as exc:
        raise TypographyError(f"Cannot load font: {font.identifier}") from exc
    if variation_axes is not None:
        definitions = font_variation_axes(font)
        values = tuple(float(value) for value in variation_axes)
        if not definitions or len(values) != len(definitions):
            raise TypographyError(
                f"Font {font.full_name!r} does not expose the requested variation axes."
            )
        for definition, value in zip(definitions, values, strict=True):
            if not definition.minimum <= value <= definition.maximum:
                raise TypographyError(
                    f"Variation {definition.tag}={value:g} is outside "
                    f"{definition.minimum:g}..{definition.maximum:g}."
                )
        try:
            loaded.set_variation_by_axes(list(values))
        except (AttributeError, OSError, ValueError) as exc:
            raise TypographyError(
                f"FreeType cannot apply variation axes to {font.full_name!r}."
            ) from exc
    return loaded, label


def _text_bbox(
    text: str,
    font: ImageFont.FreeTypeFont,
    style: TextStyle,
) -> tuple[int, int, int, int]:
    canvas = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(canvas)
    kwargs: dict[str, object] = {
        "font": font,
        "spacing": int(style.line_spacing_px),
        "stroke_width": int(style.stroke_width_px),
        "align": _pillow_horizontal_align(style.horizontal_align),
    }
    if getattr(font, "layout_engine", None) == ImageFont.Layout.RAQM:
        kwargs.update(
            {
                "language": style.language,
                "direction": style.direction,
                "features": list(style.features),
            }
        )
    return tuple(int(value) for value in draw.multiline_textbbox((0, 0), text, **kwargs))


def fit_font_size(
    text: str,
    font: FontRecord,
    target_size: tuple[int, int],
    *,
    style: TextStyle | None = None,
    maximum_size: int | None = None,
) -> int:
    """Find the largest native font size that fits a final-resolution box."""

    width, height = (int(target_size[0]), int(target_size[1]))
    if width < 1 or height < 1:
        raise TypographyError("target text box must have positive dimensions.")
    base_style = style or TextStyle(font=font)
    _validate_text_style(base_style)
    normalized, _form, coverage = text_for_font(text, font)
    if coverage < 1.0:
        raise TypographyError("selected font does not cover all text glyphs.")
    low = 1
    high = int(maximum_size or max(8, height * 4))
    best = 0
    while low <= high:
        middle = (low + high) // 2
        loaded, _engine = load_pillow_font(
            font,
            middle,
            variation_axes=base_style.variation_axes,
        )
        bounds = _text_bbox(normalized, loaded, base_style)
        measured = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        if measured[0] <= width and measured[1] <= height:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best < 1:
        raise TypographyError("text cannot fit the target box even at 1 px.")
    return best


def render_text_layer(
    text: str,
    box: tuple[int, int, int, int],
    style: TextStyle,
    *,
    canvas_size: tuple[int, int] | None = None,
    padding_px: int | None = None,
) -> TextRender:
    """Render approved text once, directly at the requested final pixel scale."""

    source_text = unicodedata.normalize("NFC", text)
    if not source_text.strip():
        raise TypographyError("text cannot be empty.")
    _validate_text_style(style)
    left, top, right, bottom = (int(value) for value in box)
    if right <= left or bottom <= top:
        raise TypographyError("text box must have positive dimensions.")
    if canvas_size is not None:
        canvas_width, canvas_height = (int(canvas_size[0]), int(canvas_size[1]))
        if left < 0 or top < 0 or right > canvas_width or bottom > canvas_height:
            raise TypographyError("text box lies outside the final canvas.")

    rendered_text, normalization, coverage = text_for_font(source_text, style.font)
    if coverage < 1.0:
        raise TypographyError(
            f"Font {style.font.full_name!r} does not cover every approved glyph."
        )
    size = int(style.font_size_px or fit_font_size(
        rendered_text,
        style.font,
        (right - left, bottom - top),
        style=style,
    ))
    font, engine_label = load_pillow_font(
        style.font,
        size,
        variation_axes=style.variation_axes,
    )
    bounds = _text_bbox(rendered_text, font, style)
    visible_width = bounds[2] - bounds[0]
    visible_height = bounds[3] - bounds[1]
    if visible_width > right - left or visible_height > bottom - top:
        raise TypographyError(
            f"Text at {size}px is {visible_width}x{visible_height}, larger than its box."
        )

    padding = int(
        padding_px
        if padding_px is not None
        else max(2, style.stroke_width_px + math.ceil(size * 0.04))
    )
    if padding < 0 or padding > 512:
        raise TypographyError("padding_px must be from 0 through 512.")
    layer = Image.new(
        "RGBA", (visible_width + padding * 2, visible_height + padding * 2), (0, 0, 0, 0)
    )
    draw = ImageDraw.Draw(layer)
    draw_kwargs: dict[str, object] = {
        "font": font,
        "fill": (*_validate_rgb_tuple(style.fill_rgb, "fill_rgb"), 255),
        "spacing": int(style.line_spacing_px),
        "align": _pillow_horizontal_align(style.horizontal_align),
        "stroke_width": int(style.stroke_width_px),
        "stroke_fill": (
            (*_validate_rgb_tuple(style.stroke_rgb, "stroke_rgb"), 255)
            if style.stroke_rgb is not None and style.stroke_width_px > 0
            else None
        ),
    }
    if getattr(font, "layout_engine", None) == ImageFont.Layout.RAQM:
        draw_kwargs.update(
            {
                "language": style.language,
                "direction": style.direction,
                "features": list(style.features),
            }
        )
    draw.multiline_text(
        (padding - bounds[0], padding - bounds[1]),
        rendered_text,
        **draw_kwargs,
    )

    horizontal = style.horizontal_align.casefold()
    if horizontal in {"left", "start"}:
        visible_x = left
    elif horizontal in {"right", "end"}:
        visible_x = right - visible_width
    else:
        visible_x = left + ((right - left) - visible_width) // 2
    vertical = style.vertical_align.casefold()
    if vertical in {"top", "start"}:
        visible_y = top
    elif vertical in {"bottom", "end"}:
        visible_y = bottom - visible_height
    else:
        visible_y = top + ((bottom - top) - visible_height) // 2

    alpha = np.asarray(layer.getchannel("A"), dtype=np.uint8)
    if not np.any(alpha):
        raise TypographyError("font renderer produced an empty alpha mask.")
    ascent, _descent = font.getmetrics()
    # Pillow's default horizontal text anchor is left-ascender.  SVG uses an
    # alphabetic baseline, so preserve the shaped draw origin explicitly
    # instead of approximating y as visible_top + font_size.
    svg_text_origin = (
        float(visible_x - bounds[0]),
        float(visible_y - bounds[1] + ascent),
    )
    return TextRender(
        rgba=layer,
        position=(visible_x - padding, visible_y - padding),
        visible_bbox=(
            visible_x,
            visible_y,
            visible_x + visible_width,
            visible_y + visible_height,
        ),
        source_text=source_text,
        rendered_text=rendered_text,
        normalization=normalization,
        font=style.font,
        font_size_px=size,
        fill_rgb=_validate_rgb_tuple(style.fill_rgb, "fill_rgb"),
        stroke_rgb=(
            _validate_rgb_tuple(style.stroke_rgb, "stroke_rgb")
            if style.stroke_rgb is not None
            else None
        ),
        stroke_width_px=int(style.stroke_width_px),
        layout_engine=engine_label,
        variation_axes=(
            tuple(float(value) for value in style.variation_axes)
            if style.variation_axes is not None
            else None
        ),
        svg_text_origin=svg_text_origin,
        report={
            "render_policy": "approved Unicode rendered directly at final scale",
            "font_embedding_performed": False,
            "font_embedding_policy": {
                "fs_type": style.font.embedding.fs_type,
                "restricted": style.font.embedding.restricted,
                "embeddable": style.font.embedding.embeddable,
            },
            "target_box": [left, top, right, bottom],
            "visible_bbox": [
                visible_x,
                visible_y,
                visible_x + visible_width,
                visible_y + visible_height,
            ],
            "glyph_coverage": coverage,
            "svg_text_origin": [svg_text_origin[0], svg_text_origin[1]],
            "variation_axes": (
                {
                    definition.tag: float(value)
                    for definition, value in zip(
                        font_variation_axes(style.font),
                        style.variation_axes,
                        strict=True,
                    )
                }
                if style.variation_axes is not None
                else {}
            ),
        },
    )


def composite_text(base: Image.Image, rendered: TextRender) -> Image.Image:
    """Alpha-composite a rendered crop without resampling its glyphs."""

    result = base.convert("RGBA")
    result.alpha_composite(rendered.rgba, dest=rendered.position)
    return result


__all__ = [
    "FontEmbeddingPolicy",
    "FontMatch",
    "FontRecord",
    "FontVariationAxis",
    "TextRender",
    "TextStyle",
    "TextStyleEstimate",
    "TypographyError",
    "composite_text",
    "discover_windows_fonts",
    "estimate_text_style",
    "fit_font_size",
    "font_embedding_policy",
    "font_variation_axes",
    "inspect_font",
    "load_pillow_font",
    "render_text_layer",
    "select_font",
    "text_for_font",
    "windows_font_directories",
]
