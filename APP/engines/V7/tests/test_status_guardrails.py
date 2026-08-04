from __future__ import annotations

import ast
import unittest
from pathlib import Path


V7_DIR = Path(__file__).resolve().parents[1]
ENGINE_PATH = V7_DIR / "design_repair_v7.py"


def _engine_tree() -> ast.Module:
    return ast.parse(ENGINE_PATH.read_text(encoding="utf-8"), filename=str(ENGINE_PATH))


class NoTextStatusGuardrailTests(unittest.TestCase):
    def test_empty_ocr_result_participates_in_review_required_branch(self) -> None:
        tree = _engine_tree()
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "no_text_detected"
                for target in node.targets
            )
        ]
        self.assertTrue(assignments, "V7 must explicitly derive no_text_detected.")
        self.assertIn("regions", ast.unparse(assignments[0].value))

        guarded_review_branches = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            assigned_review = any(
                isinstance(child, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "status"
                    for target in child.targets
                )
                and isinstance(child.value, ast.Constant)
                and child.value.value == "REVIEW_REQUIRED"
                for child in node.body
            )
            names = {child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)}
            if assigned_review and {"unresolved", "no_text_detected"} <= names:
                guarded_review_branches.append(node)

        self.assertTrue(
            guarded_review_branches,
            "Zero OCR regions must force REVIEW_REQUIRED instead of PASS.",
        )

    def test_qa_document_exposes_no_text_detected_for_auditing(self) -> None:
        tree = _engine_tree()
        matching_fields = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "no_text_detected"
                    and isinstance(value, ast.Name)
                    and value.id == "no_text_detected"
                ):
                    matching_fields.append((key, value))
        self.assertTrue(
            matching_fields,
            "QA output must disclose why a no-text run is not print-ready.",
        )


if __name__ == "__main__":
    unittest.main()
