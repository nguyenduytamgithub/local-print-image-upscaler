from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "APP"))

from upscale_cli import UserError, parse_command  # noqa: E402


class V7LauncherParsingTests(unittest.TestCase):
    def test_repair_and_v7_aliases_accept_exactly_x1(self) -> None:
        for alias in ("repair", "v7"):
            with self.subTest(alias=alias):
                mode, token, scale, allow_huge, options = parse_command(
                    [alias, "poster.png", "1"]
                )
                self.assertEqual(mode, "V7_REPAIR")
                self.assertEqual(token, "poster.png")
                self.assertEqual(scale, 1.0)
                self.assertFalse(allow_huge)
                self.assertEqual(options["review"], "gui")

    def test_repair_and_v7_aliases_reject_x1_point_5(self) -> None:
        for alias in ("repair", "v7"):
            with self.subTest(alias=alias):
                with self.assertRaises(UserError):
                    parse_command([alias, "poster.png", "1.5"])


if __name__ == "__main__":
    unittest.main()
