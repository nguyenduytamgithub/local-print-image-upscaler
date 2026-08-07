from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


V5_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V5_DIR))

import layer_engine_v5  # noqa: E402
from v5lib.formats import (  # noqa: E402
    project_clean_plate_for_alpha,
    render_layers,
    rendered_alpha_canvases,
    serialized_base_alpha_u8,
)
from v5lib.model import LayerSpec  # noqa: E402


def serialized_alpha(layer: LayerSpec, size: tuple[int, int]) -> np.ndarray:
    return serialized_base_alpha_u8(layer, size, matte_policy="clean")


class FixedAlphaStackPlannerTests(unittest.TestCase):
    def test_shared_base_alpha_matches_render_for_soft_binary_and_refined_edges(self) -> None:
        source_shape = (13, 17)
        mask = np.zeros(source_shape, dtype=bool)
        mask[2:11, 3:14] = True
        mask[5:8, 7:10] = False
        refined_alpha = np.zeros(source_shape, dtype=np.float32)
        refined_alpha[mask] = 93.0 / 255.0
        refined_alpha[4:9, 5:12] = np.where(
            mask[4:9, 5:12], 211.0 / 255.0, 0.0
        )
        final_size = (61, 47)
        specs = (
            LayerSpec("binary", "Binary", "object", mask, 1.0),
            LayerSpec(
                "refined",
                "Refined",
                "text_raster",
                mask,
                1.0,
                alpha_matte=refined_alpha,
            ),
        )
        rng = np.random.default_rng(8025)
        master = rng.integers(
            0,
            256,
            size=(final_size[1], final_size[0], 3),
            dtype=np.uint8,
        )
        clean_candidate = 255 - master

        for spec in specs:
            with self.subTest(layer=spec.layer_id):
                cached = serialized_base_alpha_u8(
                    spec,
                    final_size,
                    matte_policy="clean",
                )
                feasible_lower, projection = project_clean_plate_for_alpha(
                    master,
                    clean_candidate,
                    cached,
                )
                rendered, composite, composition = render_layers(
                    master,
                    feasible_lower,
                    [spec],
                    matte_policy="clean",
                )
                delivered = rendered_alpha_canvases(rendered, final_size)[spec.layer_id]

                self.assertTrue(projection["feasibility_gate_passed"])
                self.assertEqual(composition["recomposition_max_abs_error"], 0)
                self.assertTrue(np.array_equal(np.asarray(composite), master))
                self.assertTrue(np.array_equal(delivered, cached))
                self.assertTrue(np.logical_and(cached > 0, cached < 255).any())

    def test_reverse_plan_handles_overlapping_parent_child_without_alpha_growth(self) -> None:
        height, width = 9, 11
        parent_mask = np.zeros((height, width), dtype=bool)
        parent_mask[1:8, 1:10] = True
        parent_alpha = np.zeros((height, width), dtype=np.float32)
        parent_alpha[parent_mask] = 67.0 / 255.0
        parent_alpha[2:7, 2:9] = 151.0 / 255.0
        parent_alpha[4, 5] = 0.0

        child_mask = np.zeros((height, width), dtype=bool)
        child_mask[2:7, 3:9] = True
        child_alpha = np.zeros((height, width), dtype=np.float32)
        child_alpha[child_mask] = 103.0 / 255.0
        child_alpha[3:6, 4:8] = 209.0 / 255.0
        child_alpha[4, 6] = 0.0

        parent = LayerSpec(
            "parent",
            "Parent",
            "object",
            parent_mask,
            1.0,
            metadata={
                "display_order": 0,
                "hierarchy_depth": 0,
                "children": ["child"],
            },
            alpha_matte=parent_alpha,
        )
        child = LayerSpec(
            "child",
            "Child",
            "text_raster",
            child_mask,
            1.0,
            metadata={
                "display_order": 1,
                "hierarchy_depth": 1,
                "parent_id": "parent",
            },
            alpha_matte=child_alpha,
        )
        layers = [child, parent]  # Deliberately not already in render order.
        rng = np.random.default_rng(94021)
        master = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        background_candidate = np.full_like(master, 245)
        parent_preference = master.copy()
        parent_preference[child_mask] = np.array([7, 235, 31], dtype=np.uint8)
        alpha_canvases = {
            layer.layer_id: serialized_alpha(layer, (width, height))
            for layer in layers
        }

        background, targets, report = layer_engine_v5.plan_fixed_alpha_stack(
            master,
            background_candidate,
            layers,
            alpha_canvases,
            preferred_layer_targets={"parent": parent_preference},
        )
        rendered, composite, composition = render_layers(
            master,
            background,
            layers,
            layer_targets=targets,
            matte_policy="clean",
        )
        delivered = rendered_alpha_canvases(rendered, (width, height))

        self.assertTrue(report["all_steps_feasible"])
        self.assertEqual(
            [record["layer_id"] for record in report["layers"]],
            ["parent", "child"],
        )
        self.assertEqual(composition["recomposition_max_abs_error"], 0)
        self.assertTrue(np.array_equal(np.asarray(composite), master))
        self.assertTrue(np.array_equal(delivered["parent"], alpha_canvases["parent"]))
        self.assertTrue(np.array_equal(delivered["child"], alpha_canvases["child"]))
        # No layer owns the corner, so backward induction must restore the
        # exact target there instead of leaking the preferred clean plate.
        self.assertTrue(np.array_equal(background[0, 0], master[0, 0]))

    def test_plan_is_deterministic_and_rejects_unknown_target(self) -> None:
        mask = np.ones((2, 3), dtype=bool)
        alpha = np.full((2, 3), 127.0 / 255.0, dtype=np.float32)
        layer = LayerSpec(
            "only",
            "Only",
            "object",
            mask,
            1.0,
            alpha_matte=alpha,
        )
        master = np.array(
            [[[0, 255, 17], [91, 4, 222], [255, 0, 128]]] * 2,
            dtype=np.uint8,
        )
        clean = 255 - master
        cache = {"only": serialized_alpha(layer, (3, 2))}
        first = layer_engine_v5.plan_fixed_alpha_stack(master, clean, [layer], cache)
        second = layer_engine_v5.plan_fixed_alpha_stack(master, clean, [layer], cache)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1]["only"], second[1]["only"]))
        with self.assertRaisesRegex(ValueError, "unknown layer ids"):
            layer_engine_v5.plan_fixed_alpha_stack(
                master,
                clean,
                [layer],
                cache,
                preferred_layer_targets={"missing": master},
            )


if __name__ == "__main__":
    unittest.main()
