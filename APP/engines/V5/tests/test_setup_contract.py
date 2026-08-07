from __future__ import annotations

import ast
import importlib.util
import re
import unittest
from pathlib import Path


V5_DIR = Path(__file__).resolve().parents[1]


def pinned_lines(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise AssertionError(f"Unpinned requirement: {line}")
        name, version = line.split("==", 1)
        key = name.lower().replace("_", "-")
        if key in result:
            raise AssertionError(f"Duplicate requirement: {name}")
        result[key] = version
    return result


def literal_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    result[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return result


class SetupContractTests(unittest.TestCase):
    def test_direct_requirements_are_exactly_represented_in_lock(self) -> None:
        direct = pinned_lines(V5_DIR / "requirements.in")
        locked = pinned_lines(V5_DIR / "requirements.lock")
        for name, version in direct.items():
            self.assertEqual(version, locked.get(name), name)

    def test_all_large_artifact_hashes_are_documented(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "v5_model_setup_contract", V5_DIR / "model_setup.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        documented = {}
        for line in (V5_DIR / "MODEL_SHA256SUMS.txt").read_text(
            encoding="utf-8"
        ).splitlines():
            digest, artifact = line.split(maxsplit=1)
            documented[artifact] = digest.lower()

        self.assertEqual(module.LAMA_SHA256, documented["big-lama.pt"])
        for repo_id, revision, digest in module.MODELS:
            key = f"{repo_id}@{revision}/model.safetensors"
            self.assertEqual(digest, documented[key])
        self.assertRegex(
            documented["models/tessdata/vie.traineddata"], r"^[0-9a-f]{64}$"
        )
        self.assertRegex(
            documented[
                "UB-Mannheim/TesseractOCR@5.4.0.20240606/windows-x64-installer.exe"
            ],
            r"^[0-9a-f]{64}$",
        )

    def test_setup_pins_both_torch_flavors_and_ocr_assets(self) -> None:
        source = (V5_DIR / "setup_v5.ps1").read_text(encoding="utf-8")
        self.assertIn("https://download.pytorch.org/whl/cpu", source)
        self.assertIn("https://download.pytorch.org/whl/cu126", source)
        self.assertIn('"torch==2.12.1"', source)
        self.assertIn('"torchvision==0.27.1"', source)
        self.assertIn('"UB-Mannheim.TesseractOCR"', source)
        self.assertRegex(source, r'\$tessdataRevision\s*=\s*"[0-9a-f]{40}"')
        self.assertRegex(source, r'\$vietnameseSha256\s*=\s*"[0-9A-F]{64}"')
        self.assertNotIn('"simple-lama-inpainting==', source)
        self.assertNotIn("from simple_lama_inpainting", source)

    def test_runtime_model_revisions_and_hashes_match_setup_contract(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "v5_model_setup_runtime_contract", V5_DIR / "model_setup.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        models = {repo_id: (revision, digest) for repo_id, revision, digest in module.MODELS}
        segment = literal_constants(V5_DIR / "v5lib" / "segment.py")
        matting = literal_constants(V5_DIR / "v5lib" / "matting.py")
        restore = literal_constants(V5_DIR / "v5lib" / "restore.py")

        self.assertEqual(
            (segment["SAM_REVISION"], segment["SAM_WEIGHT_SHA256"]),
            models[segment["SAM_MODEL"]],
        )
        self.assertEqual(
            (segment["DINO_REVISION"], segment["DINO_WEIGHT_SHA256"]),
            models[segment["DINO_MODEL"]],
        )
        self.assertEqual(
            (matting["VITMATTE_REVISION"], matting["VITMATTE_WEIGHT_SHA256"]),
            models[matting["VITMATTE_MODEL"]],
        )
        self.assertEqual(
            (
                "53222614392e8bd24ed804fbd2f9a43c46ac3850",
                "bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb",
            ),
            models["hustvl/vitmatte-small-composition-1k"],
        )
        self.assertEqual(str(restore["LAMA_MODEL_SHA256"]), module.LAMA_SHA256)

        setup = (V5_DIR / "setup_v5.ps1").read_text(encoding="utf-8")
        tess_revision = re.search(r'\$tessdataRevision\s*=\s*"([0-9a-f]{40})"', setup)
        tess_hash = re.search(r'\$vietnameseSha256\s*=\s*"([0-9A-F]{64})"', setup)
        lama_hash = re.search(r'\$lamaSha256\s*=\s*"([0-9A-F]{64})"', setup)
        self.assertIsNotNone(tess_revision)
        self.assertIsNotNone(tess_hash)
        self.assertIsNotNone(lama_hash)
        self.assertEqual(segment["TESSDATA_BEST_REVISION"], tess_revision.group(1))
        self.assertEqual(
            str(segment["TESSDATA_VIE_SHA256"]).lower(), tess_hash.group(1).lower()
        )
        self.assertEqual(module.LAMA_SHA256.lower(), lama_hash.group(1).lower())


if __name__ == "__main__":
    unittest.main()
