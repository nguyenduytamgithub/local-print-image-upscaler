from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


V5_DIR = Path(__file__).resolve().parents[1]
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))

from v5lib.model import Detection, LayerSpec, MaskCandidate, TextRegion  # noqa: E402
from v5lib.poster_group import add_ocr_text_layers, group_poster_layers  # noqa: E402


def _rectangle(shape: tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _mask_candidate(
    image: np.ndarray,
    mask: np.ndarray,
    candidate_id: int,
    score: float = 0.98,
) -> MaskCandidate:
    ys, xs = np.where(mask)
    bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )
    area = int(mask.sum())
    pixels = image[mask].astype(np.float32)
    return MaskCandidate(
        candidate_id=candidate_id,
        mask=mask.copy(),
        score=score,
        bbox=bbox,
        area=area,
        fill_ratio=area / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
        mean_rgb=tuple(float(value) for value in pixels.mean(axis=0)),
        std_rgb=float(pixels.std(axis=0).mean()),
    )


def _synthetic_poster() -> tuple[np.ndarray, list[np.ndarray], list[float], int]:
    height, width = 160, 240
    shape = (height, width)
    image = np.full((height, width, 3), (250, 210, 20), dtype=np.uint8)
    masks: list[np.ndarray] = []
    scores: list[float] = []

    def add(mask: np.ndarray, colour: tuple[int, int, int], score: float = 0.98) -> None:
        masks.append(mask.copy())
        scores.append(score)
        image[mask] = colour

    top_panel = _rectangle(shape, 30, 8, 210, 34)
    add(top_panel, (5, 80, 20), 0.99)
    # Exact duplicate must be removed deterministically.
    masks.append(top_panel.copy())
    scores.append(0.97)

    left_card = _rectangle(shape, 8, 84, 105, 156)
    right_card = _rectangle(shape, 135, 84, 232, 156)
    add(left_card, (4, 75, 10), 0.99)
    add(right_card, (205, 12, 8), 0.99)

    # Four red cores with a white outline in the pixels but not in the SAM mask.
    headline_masks = [
        _rectangle(shape, 20, 43, 45, 73),
        _rectangle(shape, 50, 43, 75, 73),
        _rectangle(shape, 80, 43, 105, 73),
        _rectangle(shape, 110, 43, 135, 73),
    ]
    for mask in headline_masks:
        outline = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        image[outline] = (250, 250, 245)
        add(mask, (220, 5, 5), 0.985)

    plus = _rectangle(shape, 113, 98, 127, 143)
    plus |= _rectangle(shape, 104, 114, 136, 127)
    add(plus, (10, 90, 25), 0.98)

    # Complete colour-distinct child inside the left owner.
    child_icon = _rectangle(shape, 25, 111, 55, 141)
    add(child_icon, (250, 210, 5), 0.98)

    # Three aligned child fragments look like one row and must stay attached to
    # the right panel instead of becoming three or one promoted object layer.
    for x0 in (150, 164, 178):
        add(_rectangle(shape, x0, 98, x0 + 8, 116), (245, 245, 240), 0.97)

    return image, masks, scores, sum(int(mask.sum()) for mask in headline_masks)


class PosterGroupingTests(unittest.TestCase):
    def test_groups_panels_row_object_and_child_without_fragments(self) -> None:
        image, masks, scores, headline_core_area = _synthetic_poster()
        layers, report = group_poster_layers(image, masks, scores, max_layers=10)

        self.assertEqual(report["duplicate_candidate_count"], 1)
        self.assertEqual(report["owner_panel_count_before_budget"], 3)
        self.assertEqual(report["selected_layer_count"], 6)
        roles = [layer.metadata["grouping_role"] for layer in layers]
        self.assertEqual(roles.count("owner_panel"), 3)
        self.assertEqual(roles.count("standalone_row"), 1)
        self.assertEqual(roles.count("standalone_object"), 1)
        self.assertEqual(roles.count("promoted_child_object"), 1)

        row = next(layer for layer in layers if layer.metadata["grouping_role"] == "standalone_row")
        self.assertGreater(row.area, headline_core_area, "guided halo should retain the white outline")
        self.assertEqual(len(row.source_ids), 4)

        child = next(
            layer for layer in layers if layer.metadata["grouping_role"] == "promoted_child_object"
        )
        self.assertIsNotNone(child.metadata["parent_id"])
        parent = next(layer for layer in layers if layer.layer_id == child.metadata["parent_id"])
        self.assertIn(child.layer_id, parent.metadata["children"])
        self.assertEqual(child.metadata["hierarchy_depth"], 1)
        self.assertTrue(np.all(parent.mask[child.mask]))

        # Geometric naming must not claim fixture-specific words or semantics.
        for layer in layers:
            self.assertRegex(layer.name, r"^(PANEL|ROW GROUP|OBJECT GROUP|CHILD OBJECT|DECORATION GROUP)")
            self.assertIsNone(layer.text)
            self.assertIsNone(layer.label)

    def test_budget_is_hard_and_hierarchy_remains_valid(self) -> None:
        image, masks, scores, _ = _synthetic_poster()
        layers, report = group_poster_layers(image, masks, scores, max_layers=4)
        self.assertEqual(len(layers), 4)
        self.assertTrue(report["dropped_by_budget"])
        ids = {layer.layer_id for layer in layers}
        for layer in layers:
            parent_id = layer.metadata["parent_id"]
            self.assertTrue(parent_id is None or parent_id in ids)
            self.assertTrue(set(layer.metadata["children"]).issubset(ids))

    def test_is_deterministic_and_does_not_mutate_inputs(self) -> None:
        image, masks, scores, _ = _synthetic_poster()
        image_before = image.copy()
        masks_before = [mask.copy() for mask in masks]
        first, first_report = group_poster_layers(image, masks, scores, max_layers=10)
        second, second_report = group_poster_layers(image, masks, scores, max_layers=10)
        self.assertEqual(first_report, second_report)
        self.assertEqual([layer.layer_id for layer in first], [layer.layer_id for layer in second])
        self.assertEqual([layer.name for layer in first], [layer.name for layer in second])
        for a, b in zip(first, second, strict=True):
            self.assertTrue(np.array_equal(a.mask, b.mask))
            self.assertEqual(a.metadata, b.metadata)
        self.assertTrue(np.array_equal(image, image_before))
        for mask, before in zip(masks, masks_before, strict=True):
            self.assertTrue(np.array_equal(mask, before))

    def test_validates_shapes_lengths_and_budget(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        mask = np.zeros((20, 30), dtype=bool)
        with self.assertRaises(ValueError):
            group_poster_layers(image, [mask], [], max_layers=4)
        with self.assertRaises(ValueError):
            group_poster_layers(image, [np.zeros((21, 30), dtype=bool)], [0.9], max_layers=4)
        with self.assertRaises(ValueError):
            group_poster_layers(image, [mask], [0.9], max_layers=0)

    def test_ocr_lines_upgrade_rows_and_attach_to_smallest_panel(self) -> None:
        image, masks, scores, _ = _synthetic_poster()
        base_layers, _ = group_poster_layers(image, masks, scores, max_layers=10)
        base_snapshot = [
            (layer.layer_id, layer.mask.copy(), layer.category, layer.text, dict(layer.metadata))
            for layer in base_layers
        ]
        candidates = [
            _mask_candidate(image, mask, candidate_id=index, score=scores[index])
            for index, mask in enumerate(masks)
        ]
        regions = [
            TextRegion(bbox=(16, 39, 140, 77), text="SALE", confidence=92.0),
            TextRegion(bbox=(146, 94, 190, 121), text="ABC", confidence=88.0),
            TextRegion(bbox=(147, 95, 191, 122), text="ABC", confidence=55.0),
            TextRegion(bbox=(5, 5, 18, 18), text="---", confidence=99.0),
            TextRegion(bbox=(200, 40, 230, 60), text="NO MASK", confidence=90.0),
        ]
        detections = [Detection(bbox=(135, 84, 232, 156), label="package", score=0.8)]

        layers, report = add_ocr_text_layers(
            image,
            base_layers,
            candidates,
            regions,
            max_layers=10,
            detections=detections,
        )

        text_layers = [layer for layer in layers if layer.category == "text_raster"]
        self.assertEqual(len(text_layers), 2)
        self.assertEqual(report["matched_existing_row_count"], 1)
        self.assertEqual(report["created_text_layer_count"], 1)
        self.assertEqual(report["duplicate_text_region_count"], 1)
        self.assertEqual(report["rejected_text_regions"]["non_alphanumeric_text"], 1)
        self.assertEqual(report["rejected_text_regions"]["no_sam_support"], 1)
        self.assertEqual(len(layers), len(base_layers) + 1)

        headline = next(layer for layer in text_layers if layer.text == "SALE")
        self.assertEqual(headline.metadata["grouping_role"], "standalone_row")
        original_headline = next(
            layer
            for layer in base_layers
            if layer.metadata["grouping_role"] == "standalone_row"
        )
        self.assertTrue(np.array_equal(headline.mask, original_headline.mask))
        self.assertEqual(
            headline.metadata["ocr_mask_policy"], "existing_geometry_annotation_only"
        )
        panel_text = next(layer for layer in text_layers if layer.text == "ABC")
        self.assertEqual(panel_text.metadata["grouping_role"], "ocr_text_line")
        self.assertRegex(panel_text.name, r"^TEXT LINE 02 - ")
        self.assertNotIn("ABC", panel_text.name)
        self.assertEqual(panel_text.label, "text_line")
        self.assertTrue(panel_text.metadata["semantic_hints"])

        parent_id = panel_text.metadata["parent_id"]
        self.assertIsNotNone(parent_id)
        parent = next(layer for layer in layers if layer.layer_id == parent_id)
        self.assertEqual(parent.metadata["grouping_role"], "owner_panel")
        self.assertIn(panel_text.layer_id, parent.metadata["children"])
        self.assertTrue(np.all(parent.mask[panel_text.mask]))
        self.assertNotIn("package", parent.name.lower())

        # This helper is pure: callers may safely reuse the original poster hierarchy.
        for layer, snapshot in zip(base_layers, base_snapshot, strict=True):
            layer_id, mask, category, text, metadata = snapshot
            self.assertEqual(layer.layer_id, layer_id)
            self.assertTrue(np.array_equal(layer.mask, mask))
            self.assertEqual(layer.category, category)
            self.assertEqual(layer.text, text)
            self.assertEqual(layer.metadata, metadata)

    def test_ocr_text_has_priority_but_never_exceeds_hard_cap(self) -> None:
        image, masks, scores, _ = _synthetic_poster()
        base_layers, _ = group_poster_layers(image, masks, scores, max_layers=10)
        candidates = [
            _mask_candidate(image, mask, candidate_id=index, score=scores[index])
            for index, mask in enumerate(masks)
        ]
        regions = [
            TextRegion(bbox=(16, 39, 140, 77), text="SALE", confidence=92.0),
            TextRegion(bbox=(146, 94, 190, 121), text="ABC", confidence=88.0),
        ]

        layers, report = add_ocr_text_layers(
            image, base_layers, candidates, regions, max_layers=6
        )

        self.assertEqual(len(layers), 6)
        self.assertEqual(report["selected_layer_count"], 6)
        self.assertEqual(report["selected_text_layer_count"], 2)
        self.assertTrue(report["optional_layers_evicted_for_text"])
        ids = {layer.layer_id for layer in layers}
        for layer in layers:
            parent_id = layer.metadata.get("parent_id")
            self.assertTrue(parent_id is None or parent_id in ids)
            self.assertTrue(set(layer.metadata.get("children", [])).issubset(ids))

    def test_ocr_helper_validates_layer_and_candidate_shapes(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        mask = _rectangle((20, 30), 2, 2, 8, 9)
        candidate = _mask_candidate(image, mask, 1)
        with self.assertRaises(ValueError):
            add_ocr_text_layers(image, [], [candidate], [], max_layers=0)
        bad_candidate = _mask_candidate(image, mask, 2)
        bad_candidate.mask = np.zeros((21, 30), dtype=bool)
        with self.assertRaises(ValueError):
            add_ocr_text_layers(image, [], [bad_candidate], [], max_layers=2)

    def test_ocr_rejects_weak_geometry_for_low_confidence_or_wide_boxes(self) -> None:
        image = np.full((80, 200, 3), (245, 215, 20), dtype=np.uint8)
        glyph = _rectangle((80, 200), 20, 30, 40, 42)
        image[glyph] = (10, 10, 10)
        candidate = _mask_candidate(image, glyph, 1)

        low_layers, low_report = add_ocr_text_layers(
            image,
            [],
            [candidate],
            [TextRegion(bbox=(10, 26, 185, 46), text="bad guess", confidence=12.0)],
            max_layers=4,
        )
        self.assertFalse(low_layers)
        self.assertEqual(
            low_report["rejected_text_regions"]["low_confidence_weak_geometry"], 1
        )

        wide_layers, wide_report = add_ocr_text_layers(
            image,
            [],
            [candidate],
            [TextRegion(bbox=(0, 26, 198, 46), text="wide guess", confidence=95.0)],
            max_layers=4,
        )
        self.assertFalse(wide_layers)
        self.assertEqual(wide_report["rejected_text_regions"]["wide_box_weak_geometry"], 1)

    def test_ocr_subtracts_an_existing_child_icon_from_wide_line(self) -> None:
        shape = (120, 240)
        image = np.full((*shape, 3), (245, 215, 20), dtype=np.uint8)
        owner_mask = _rectangle(shape, 10, 10, 230, 110)
        image[owner_mask] = (5, 75, 15)
        child_mask = _rectangle(shape, 20, 35, 60, 85)
        image[child_mask] = (220, 20, 10)
        glyphs = [
            _rectangle(shape, x0, 42, x0 + 12, 76)
            for x0 in (82, 100, 118, 136, 154)
        ]
        for glyph in glyphs:
            image[glyph] = (245, 245, 240)
        owner = LayerSpec(
            layer_id="panel",
            name="PANEL 01 - CENTER",
            category="detail_group",
            mask=owner_mask,
            score=0.99,
            metadata={
                "grouping_role": "owner_panel",
                "parent_id": None,
                "children": ["child"],
                "hierarchy_depth": 0,
                "display_order": 1,
                "optional": False,
            },
        )
        child = LayerSpec(
            layer_id="child",
            name="CHILD OBJECT 01 - LEFT",
            category="object",
            mask=child_mask,
            score=0.98,
            metadata={
                "grouping_role": "promoted_child_object",
                "parent_id": "panel",
                "children": [],
                "hierarchy_depth": 1,
                "display_order": 2,
                "optional": True,
            },
        )
        candidates = [_mask_candidate(image, child_mask, 1)] + [
            _mask_candidate(image, glyph, index)
            for index, glyph in enumerate(glyphs, 2)
        ]

        layers, report = add_ocr_text_layers(
            image,
            [owner, child],
            candidates,
            [TextRegion(bbox=(18, 32, 175, 88), text="WIDE LINE", confidence=91.0)],
            max_layers=6,
        )

        text = next(layer for layer in layers if layer.category == "text_raster")
        kept_child = next(layer for layer in layers if layer.layer_id == "child")
        self.assertFalse(np.logical_and(text.mask, kept_child.mask).any())
        self.assertGreater(text.metadata["ocr_protected_removed_fraction"], 0.0)
        self.assertEqual(report["created_text_layer_count"], 1)
        self.assertEqual(report["created_visual_text_layer_count"], 0)

    def test_visual_fallback_merges_words_but_rejects_child_and_two_piece_bow(self) -> None:
        shape = (140, 260)
        image = np.full((*shape, 3), (245, 215, 20), dtype=np.uint8)
        owner_mask = _rectangle(shape, 10, 10, 250, 130)
        image[owner_mask] = (5, 75, 15)
        child_mask = _rectangle(shape, 20, 38, 62, 91)
        image[child_mask] = (220, 20, 10)
        first_word = [
            _rectangle(shape, x0, 42, x0 + 11, 75) for x0 in (82, 98, 114)
        ]
        second_word = [
            _rectangle(shape, x0, 42, x0 + 11, 75) for x0 in (158, 174, 190)
        ]
        glyphs = first_word + second_word
        for glyph in glyphs:
            image[glyph] = (245, 245, 240)
        bow = [
            _rectangle(shape, 95, 94, 116, 108),
            _rectangle(shape, 122, 94, 143, 108),
        ]
        for piece in bow:
            image[piece] = (245, 205, 5)
        owner = LayerSpec(
            layer_id="panel",
            name="PANEL 01 - CENTER",
            category="detail_group",
            mask=owner_mask,
            score=0.99,
            metadata={
                "grouping_role": "owner_panel",
                "parent_id": None,
                "children": ["child"],
                "hierarchy_depth": 0,
                "display_order": 1,
                "optional": False,
            },
        )
        child = LayerSpec(
            layer_id="child",
            name="CHILD OBJECT 01 - LEFT",
            category="object",
            mask=child_mask,
            score=0.98,
            metadata={
                "grouping_role": "promoted_child_object",
                "parent_id": "panel",
                "children": [],
                "hierarchy_depth": 1,
                "display_order": 2,
                "optional": True,
            },
        )
        masks = [child_mask, *glyphs, *bow]
        candidates = [
            _mask_candidate(image, mask, index)
            for index, mask in enumerate(masks, 1)
        ]

        layers, report = add_ocr_text_layers(
            image, [owner, child], candidates, [], max_layers=5
        )

        visual = [
            layer
            for layer in layers
            if layer.metadata.get("grouping_role") == "visual_text_row"
        ]
        self.assertEqual(len(visual), 1)
        self.assertEqual(visual[0].metadata["visual_component_count"], 6)
        self.assertEqual(visual[0].metadata["parent_id"], "panel")
        self.assertFalse(np.logical_and(visual[0].mask, child_mask).any())
        self.assertFalse(any(np.logical_and(visual[0].mask, piece).any() for piece in bow))
        self.assertEqual(report["visual_text_proposal_count"], 1)
        self.assertGreaterEqual(report["rejected_visual_text_rows"]["too_few_components"], 1)
        self.assertLessEqual(len(layers), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
