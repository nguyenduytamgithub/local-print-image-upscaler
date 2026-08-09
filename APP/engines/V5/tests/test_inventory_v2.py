from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from v5pro.inventory import (
    build_inventory,
    detect_edge_components,
    detect_frame_inventory,
    detect_line_inventory,
)


class InventoryV2Tests(unittest.TestCase):
    def test_layout_detectors_find_frame_and_lines(self) -> None:
        image = np.full((240, 320, 3), 245, dtype=np.uint8)
        cv2.rectangle(image, (20, 40), (300, 210), (20, 150, 30), 3)
        cv2.line(image, (30, 110), (290, 110), (20, 150, 30), 2)
        frames, _ = detect_frame_inventory(image)
        lines, _ = detect_line_inventory(image)
        self.assertTrue(any(item.record.kind_hint == "frame" for item in frames))
        self.assertTrue(any(item.record.kind_hint == "line" for item in lines))

    def test_edge_components_are_never_silently_dropped(self) -> None:
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        image[10:20, 10:20] = 0
        image[50:70, 60:90] = 0
        proposals, report = detect_edge_components(image)
        self.assertEqual(report["proposal_count"], len(proposals))
        for item in proposals:
            self.assertIn(item.record.status, {"unresolved", "rejected"})
            if item.record.status == "rejected":
                self.assertTrue(item.record.reason)

    def test_complete_inventory_has_unique_accounted_records(self) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (15, 30), (285, 170), (0, 100, 0), 3)
        result = build_inventory(
            image,
            Path("unused.png"),
            include_text=False,
            include_edge_components=True,
        )
        ids = [item.record.proposal_id for item in result.proposals]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(result.detector_reports["total_proposals"], len(ids))
        self.assertTrue(all(item.record.status in {"unresolved", "rejected"} for item in result.proposals))


if __name__ == "__main__":
    unittest.main()
