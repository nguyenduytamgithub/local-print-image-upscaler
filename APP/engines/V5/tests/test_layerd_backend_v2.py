from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from v5pro.layerd_backend import LayerDRawLayer, run_layerd


class LayerDBackendV2Tests(unittest.TestCase):
    def test_raw_layer_contract(self) -> None:
        layer = LayerDRawLayer(np.zeros((3, 4, 4), dtype=np.uint8), 1, 1)
        self.assertEqual(layer.rgba.shape, (3, 4, 4))
        with self.assertRaises(ValueError):
            LayerDRawLayer(np.zeros((3, 4, 3), dtype=np.uint8), 1, 1)

    def test_argument_validation_precedes_model_load(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                run_layerd(
                    image,
                    vendor_root=root,
                    lama_model=root / "missing.pt",
                    device="cuda",
                    max_iterations=0,
                )
            with self.assertRaises(ValueError):
                run_layerd(
                    image,
                    vendor_root=root,
                    lama_model=root / "missing.pt",
                    device="cuda",
                    process_size=128,
                )

    def test_vendored_birefnet_has_transformers5_tied_weight_compatibility(self) -> None:
        vendor_source = Path(__file__).resolve().parents[1] / "vendor" / "LayerD" / "src"
        sys.path.insert(0, str(vendor_source))
        try:
            from layerd.matting import birefnet
            from transformers import PreTrainedModel

            sentinel = object()
            with mock.patch.object(
                birefnet.AutoModelForImageSegmentation,
                "from_pretrained",
                return_value=sentinel,
            ):
                self.assertIs(birefnet.build_birefnet("pinned-local"), sentinel)
            self.assertEqual(getattr(PreTrainedModel, "all_tied_weights_keys"), {})
        finally:
            sys.path.remove(str(vendor_source))


if __name__ == "__main__":
    unittest.main()
