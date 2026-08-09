from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from v5pro.review_server import (
    LOOPBACK_HOST,
    REVIEW_SCHEMA,
    ReviewPathError,
    ReviewSession,
    ReviewValidationError,
    create_review_server,
    render_review_html,
    write_review_checkpoint,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReviewBundle:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True)
        self.source = root / "poster.png"
        Image.new("RGB", (120, 80), (245, 242, 230)).save(self.source)

        mask_dir = root / "masks"
        mask_dir.mkdir()
        self.mask = mask_dir / "icon.png"
        alpha = Image.new("L", (30, 30), 0)
        ImageDraw.Draw(alpha).ellipse((2, 2, 27, 27), fill=255)
        alpha.save(self.mask)

        self.document: dict[str, object] = {
            "schema": REVIEW_SCHEMA,
            "source": "poster.png",
            "source_sha256": _sha256(self.source),
            "canvas": [120, 80],
            "nodes": [
                {
                    "id": "NODE_ICON",
                    "name": "Biểu tượng tròn",
                    "kind": "icon",
                    "bbox": [10, 10, 40, 40],
                    "z_index": 1,
                    "review_status": "unresolved",
                    "mask": "masks/icon.png",
                },
                {
                    "id": "NODE_TEXT",
                    "name": "Dòng chữ",
                    "kind": "text",
                    "bbox": [50, 10, 105, 32],
                    "z_index": 2,
                    "review_status": "auto_confirmed",
                },
            ],
            "proposals": [
                {
                    "id": "PROPOSAL_ONE",
                    "source": "edge_residue",
                    "kind_hint": "decoration",
                    "bbox": [20, 48, 48, 72],
                    "confidence": 0.72,
                    "status": "unresolved",
                    "owner_ids": [],
                    "reason": None,
                }
            ],
            "groups": [],
            "review": {"revision": 0, "finished": False},
        }
        self.checkpoint = root / "LAYER_REVIEW.json"
        write_review_checkpoint(self.checkpoint, self.document)


class ReviewSessionPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.bundle = ReviewBundle(Path(self.temporary.name) / "bundle")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_public_state_and_html_hide_technical_json(self) -> None:
        session = ReviewSession.open(self.bundle.checkpoint)
        state = session.public_state()
        self.assertEqual(state["summary"]["node_count"], 2)  # type: ignore[index]
        self.assertEqual(state["summary"]["unresolved"], 2)  # type: ignore[index]
        self.assertEqual([row["number"] for row in state["nodes"]], [1, 2])  # type: ignore[index]
        self.assertEqual(state["proposals"][0]["number"], 3)  # type: ignore[index]
        self.assertTrue(state["source_url"].startswith("/asset/source?token="))  # type: ignore[union-attr]

        html, nonce = render_review_html(session.token)
        self.assertIn("Kiểm tra và tách layer", html)
        self.assertIn("Khoanh vùng còn thiếu", html)
        self.assertIn("Cọ bớt mask", html)
        self.assertIn("Gộp các layer đã chọn", html)
        self.assertIn('id="showPending"', html)
        self.assertIn('id="showTechnical"', html)
        self.assertIn("function canvasItems()", html)
        self.assertIn('item.status==="unresolved"||isSelectedItem(item)', html)
        self.assertIn("return current?[current]:[]", html)
        self.assertIn("for(const item of canvasItems())", html)
        self.assertIn("return canvasItems()", html)
        self.assertNotIn("for(const item of allItems())", html)
        canvas_logic = html[
            html.index("function canvasItems()") : html.index("function setMessage")
        ]
        self.assertLess(
            canvas_logic.index('if($("showTechnical").checked) return allItems()'),
            canvas_logic.index('if($("showPending").checked)'),
        )
        self.assertIn(session.token, html)
        self.assertIn(nonce, html)
        self.assertNotIn(str(self.bundle.checkpoint), html)
        self.assertNotIn("<textarea", html)

    def test_public_state_keeps_resolved_proposals_for_technical_toggle(self) -> None:
        document = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        document["proposals"].extend(
            [
                {
                    "id": "PROPOSAL_ASSIGNED",
                    "source": "semantic",
                    "kind_hint": "product",
                    "bbox": [52, 42, 84, 70],
                    "confidence": 0.91,
                    "status": "assigned",
                    "owner_ids": ["NODE_ICON"],
                    "reason": None,
                },
                {
                    "id": "PROPOSAL_REJECTED",
                    "source": "residual",
                    "kind_hint": "unknown",
                    "bbox": [86, 44, 116, 74],
                    "confidence": 0.24,
                    "status": "rejected",
                    "owner_ids": [],
                    "reason": "not_an_element",
                },
            ]
        )
        write_review_checkpoint(self.bundle.checkpoint, document)

        state = ReviewSession.open(self.bundle.checkpoint).public_state()
        statuses = {item["id"]: item["status"] for item in state["proposals"]}
        self.assertEqual(
            statuses,
            {
                "PROPOSAL_ONE": "unresolved",
                "PROPOSAL_ASSIGNED": "assigned",
                "PROPOSAL_REJECTED": "rejected",
            },
        )
        self.assertEqual(state["summary"]["proposal_count"], 3)
        self.assertEqual(state["summary"]["unresolved"], 2)

    def test_public_state_explains_compound_product_without_exposing_metadata(self) -> None:
        document = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        product = document["nodes"][0]
        product["kind"] = "product"
        product["move_safe"] = True
        product["metadata"] = {
            "semantic_atomicity": {
                "policy": "semantic_product_atomicity_fail_closed_v1",
                "classification": "compound_subassembly",
                "atomic_leaf_confirmed": False,
                "private_detector_payload": "must-not-leak",
            },
            "semantic_atomicity_requires_review": True,
        }
        write_review_checkpoint(self.bundle.checkpoint, document)

        state = ReviewSession.open(self.bundle.checkpoint).public_state()
        node = next(item for item in state["nodes"] if item["id"] == "NODE_ICON")
        self.assertIn("di chuyển cả cụm", node["review_note"])
        self.assertNotIn("metadata", node)
        self.assertNotIn("private_detector_payload", json.dumps(state, ensure_ascii=False))
        html, _nonce = render_review_html("local-test")
        self.assertIn("item.review_note", html)

    def test_public_state_explains_failed_exact_text_export_in_plain_language(self) -> None:
        document = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        text_node = document["nodes"][0]
        text_node["kind"] = "text"
        text_node["review_status"] = "unresolved"
        text_node["move_safe"] = False
        text_node["metadata"] = {
            "text_export_purity_preflight": {
                "policy": "exact_export_text_purity_v1",
                "status": "unsafe",
                "reasons": ["foreground_core_not_retained"],
                "private_pixel_metrics": {"must_not_leak": True},
            }
        }
        write_review_checkpoint(self.bundle.checkpoint, document)

        state = ReviewSession.open(self.bundle.checkpoint).public_state()
        node = next(item for item in state["nodes"] if item["id"] == "NODE_ICON")
        self.assertIn("lõi", node["review_note"])
        self.assertIn("chưa an toàn", node["review_note"])
        self.assertNotIn("metadata", node)
        self.assertNotIn("private_pixel_metrics", json.dumps(state, ensure_ascii=False))

    def test_edit_is_atomic_and_undo_redo_are_real_persisted_states(self) -> None:
        session = ReviewSession.open(self.bundle.checkpoint)
        with patch("v5pro.review_server.os.replace", wraps=os.replace) as replace:
            session.edit_node("NODE_ICON", name="Logo cửa hàng", kind="logo")
        replace.assert_called_once()

        saved = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(saved["nodes"][0]["name"], "Logo cửa hàng")
        self.assertEqual(saved["nodes"][0]["kind"], "logo")
        self.assertEqual(saved["nodes"][0]["review_status"], "user_confirmed")

        undone = session.undo()
        saved = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(saved["nodes"][0]["name"], "Biểu tượng tròn")
        self.assertTrue(undone["can_redo"])

        redone = session.redo()
        saved = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(saved["nodes"][0]["name"], "Logo cửa hàng")
        self.assertTrue(redone["can_undo"])

    def test_accept_proposal_materialises_portable_mask_and_can_be_undone(self) -> None:
        session = ReviewSession.open(self.bundle.checkpoint)
        state = session.accept_proposal("PROPOSAL_ONE")
        self.assertEqual(state["summary"]["node_count"], 3)  # type: ignore[index]

        saved = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        proposal = saved["proposals"][0]
        self.assertEqual(proposal["status"], "assigned")
        new_id = proposal["owner_ids"][0]
        node = next(item for item in saved["nodes"] if item["id"] == new_id)
        self.assertEqual(node["review_status"], "unresolved")
        self.assertFalse(Path(node["mask"]).is_absolute())
        generated = self.bundle.root.joinpath(*Path(node["mask"]).parts)
        self.assertTrue(generated.is_file())
        with Image.open(generated) as mask:
            self.assertEqual(mask.size, (28, 24))

        session.undo()
        saved = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(saved["proposals"][0]["status"], "unresolved")
        self.assertEqual(len(saved["nodes"]), 2)

    def test_accept_proposal_uses_detector_mask_instead_of_opaque_bbox(self) -> None:
        proposal_mask = Image.new("L", (28, 24), 0)
        ImageDraw.Draw(proposal_mask).rectangle((8, 6, 13, 11), fill=255)
        proposal_mask.save(self.bundle.root / "masks" / "proposal.png")
        document = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        document["proposals"][0]["mask"] = "masks/proposal.png"
        write_review_checkpoint(self.bundle.checkpoint, document)

        session = ReviewSession.open(self.bundle.checkpoint)
        session.accept_proposal("PROPOSAL_ONE")
        saved = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        node_id = saved["proposals"][0]["owner_ids"][0]
        node = next(item for item in saved["nodes"] if item["id"] == node_id)
        # Proposal starts at (20, 48); nonzero local mask starts at (8, 6).
        self.assertEqual(node["bbox"], [28, 54, 34, 60])
        self.assertEqual(node["review_status"], "unresolved")
        with Image.open(self.bundle.root / Path(node["mask"])) as mask:
            self.assertEqual(mask.size, (6, 6))

    def test_split_brush_group_delete_and_reject_cover_user_repairs(self) -> None:
        session = ReviewSession.open(self.bundle.checkpoint)
        state = session.split_node_by_rectangle("NODE_ICON", [10, 10, 25, 40])
        split_nodes = [item for item in state["nodes"] if item["id"].startswith("NODE_ICON_")]  # type: ignore[index]
        self.assertEqual(len(split_nodes), 2)
        first_id, second_id = split_nodes[0]["id"], split_nodes[1]["id"]

        state = session.brush_node(
            first_id,
            mode="add",
            points=[[4, 4], [8, 8]],
            radius=3,
        )
        first = next(item for item in state["nodes"] if item["id"] == first_id)  # type: ignore[index]
        self.assertEqual(first["status"], "unresolved")

        state = session.merge_group([first_id, second_id], name="Biểu tượng hoàn chỉnh")
        self.assertEqual(state["groups"][0]["member_ids"], [first_id, second_id])  # type: ignore[index]
        state = session.delete_node(second_id)
        self.assertEqual(state["groups"], [])
        self.assertFalse(any(item["id"] == second_id for item in state["nodes"]))  # type: ignore[index]

        state = session.reject_proposal("PROPOSAL_ONE")
        self.assertEqual(state["proposals"][0]["status"], "rejected")  # type: ignore[index]
        saved = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(saved["proposals"][0]["reason"], "user_rejected_in_local_review")

    def test_delete_parent_dissolves_child_group_before_reparenting(self) -> None:
        document = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        document["nodes"] = [
            {
                "id": "PANEL_PARENT",
                "name": "Parent",
                "kind": "panel",
                "bbox": [0, 0, 100, 70],
                "z_index": 0,
                "review_status": "unresolved",
            },
            {
                "id": "ELEMENT_CHILD_A",
                "name": "Child A",
                "kind": "unknown",
                "bbox": [5, 5, 15, 15],
                "z_index": 1,
                "parent_id": "PANEL_PARENT",
                "review_status": "unresolved",
                "evidence": [{"source": "layerd"}],
                "metadata": {"source_iteration": 1},
            },
            {
                "id": "ELEMENT_CHILD_B",
                "name": "Child B",
                "kind": "unknown",
                "bbox": [25, 5, 35, 15],
                "z_index": 3,
                "parent_id": "PANEL_PARENT",
                "review_status": "unresolved",
                "evidence": [{"source": "layerd"}],
                "metadata": {"source_iteration": 1},
            },
            {
                "id": "ELEMENT_ROOT_MIDDLE",
                "name": "Root middle",
                "kind": "unknown",
                "bbox": [45, 5, 55, 15],
                "z_index": 2,
                "review_status": "unresolved",
                "evidence": [{"source": "layerd"}],
                "metadata": {"source_iteration": 1},
            },
        ]
        document["proposals"] = []
        document["groups"] = [
            {
                "id": "AUTO_ORG_CHILDREN",
                "name": "Child details",
                "member_ids": ["ELEMENT_CHILD_A", "ELEMENT_CHILD_B"],
                "source": "auto_organization_v1",
            }
        ]
        write_review_checkpoint(self.bundle.checkpoint, document)

        state = ReviewSession.open(self.bundle.checkpoint).delete_node("PANEL_PARENT")
        self.assertEqual(state["groups"], [])
        children = {
            item["id"]: item
            for item in json.loads(
                self.bundle.checkpoint.read_text(encoding="utf-8")
            )["nodes"]
        }
        self.assertIsNone(children["ELEMENT_CHILD_A"].get("parent_id"))
        self.assertIsNone(children["ELEMENT_CHILD_B"].get("parent_id"))

    def test_grouping_uses_final_ownership_order_not_raw_z_order(self) -> None:
        document = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        document["nodes"] = [
            {
                "id": "ELEMENT_RAW_1",
                "name": "Raw one",
                "kind": "unknown",
                "bbox": [2, 2, 8, 8],
                "z_index": 1,
                "review_status": "unresolved",
                "evidence": [{"source": "layerd"}],
                "metadata": {"source_iteration": 1},
            },
            {
                "id": "TEXT_MIDDLE_Z",
                "name": "Text",
                "kind": "text",
                "bbox": [20, 2, 30, 8],
                "z_index": 2,
                "review_status": "unresolved",
            },
            {
                "id": "ELEMENT_RAW_2",
                "name": "Raw two",
                "kind": "unknown",
                "bbox": [10, 2, 16, 8],
                "z_index": 3,
                "review_status": "unresolved",
                "evidence": [{"source": "layerd"}],
                "metadata": {"source_iteration": 1},
            },
        ]
        write_review_checkpoint(self.bundle.checkpoint, document)
        session = ReviewSession.open(self.bundle.checkpoint)

        state = session.merge_group(
            ["ELEMENT_RAW_1", "ELEMENT_RAW_2"], name="Raw details"
        )

        self.assertEqual(
            state["groups"][0]["member_ids"],  # type: ignore[index]
            ["ELEMENT_RAW_1", "ELEMENT_RAW_2"],
        )

        session.undo()
        with self.assertRaises(ReviewValidationError):
            session.merge_group(
                ["ELEMENT_RAW_1", "TEXT_MIDDLE_Z"], name="Unsafe stack jump"
            )

    def test_checkpoint_rejects_group_crossing_container_parents(self) -> None:
        document = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        document["nodes"][0]["parent_id"] = None
        document["nodes"][1]["parent_id"] = "NODE_ICON"
        document["groups"] = [
            {
                "id": "BROKEN_GROUP",
                "name": "Cross-parent",
                "member_ids": ["NODE_ICON", "NODE_TEXT"],
            }
        ]
        write_review_checkpoint(self.bundle.checkpoint, document)

        with self.assertRaisesRegex(ReviewValidationError, "thư mục cha"):
            ReviewSession.open(self.bundle.checkpoint)

    def test_add_rectangle_and_atomic_failure_do_not_corrupt_checkpoint(self) -> None:
        session = ReviewSession.open(self.bundle.checkpoint)
        session.add_rectangular_proposal([72, 45, 110, 70], kind="micro_detail")
        saved = json.loads(self.bundle.checkpoint.read_text(encoding="utf-8"))
        created = saved["proposals"][-1]
        self.assertEqual(created["source"], "local_review_rectangle")
        self.assertEqual(created["kind_hint"], "micro_detail")

        before = self.bundle.checkpoint.read_bytes()
        with patch("v5pro.review_server.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                session.accept_node("NODE_ICON")
        self.assertEqual(self.bundle.checkpoint.read_bytes(), before)
        self.assertEqual(list(self.bundle.root.glob(".LAYER_REVIEW.json.*.tmp")), [])

    def test_external_change_is_never_overwritten(self) -> None:
        session = ReviewSession.open(self.bundle.checkpoint)
        with self.bundle.checkpoint.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        changed = self.bundle.checkpoint.read_bytes()
        from v5pro.review_server import ReviewConflictError

        with self.assertRaises(ReviewConflictError):
            session.accept_node("NODE_ICON")
        self.assertEqual(self.bundle.checkpoint.read_bytes(), changed)


class ReviewPathAndHTTPTests(unittest.TestCase):
    def test_source_and_mask_cannot_escape_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.png"
            Image.new("RGB", (120, 80), "white").save(outside)
            bundle = ReviewBundle(root / "bundle")

            document = copy.deepcopy(bundle.document)
            document["source"] = "../outside.png"
            write_review_checkpoint(bundle.checkpoint, document)
            with self.assertRaises(ReviewPathError):
                ReviewSession.open(bundle.checkpoint)

            document = copy.deepcopy(bundle.document)
            document["nodes"][0]["mask"] = "../outside.png"  # type: ignore[index]
            write_review_checkpoint(bundle.checkpoint, document)
            with self.assertRaises(ReviewPathError):
                ReviewSession.open(bundle.checkpoint)

    def test_server_is_loopback_token_protected_and_action_persists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = ReviewBundle(Path(temporary) / "bundle")
            with self.assertRaises(ReviewPathError):
                create_review_server(bundle.checkpoint, host="0.0.0.0")

            handle = create_review_server(bundle.checkpoint)
            thread = threading.Thread(target=handle.serve_forever, daemon=True)
            thread.start()
            try:
                self.assertEqual(handle.server.server_address[0], LOOPBACK_HOST)
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    urllib.request.urlopen(handle.server.origin + "/", timeout=3)
                self.assertEqual(denied.exception.code, 403)

                with urllib.request.urlopen(handle.url, timeout=3) as response:
                    html = response.read().decode("utf-8")
                    self.assertIn("Content-Security-Policy", response.headers)
                    self.assertIn("Kiểm tra và tách layer", html)

                body = json.dumps(
                    {"action": "accept_node", "node_id": "NODE_ICON"}, ensure_ascii=False
                ).encode("utf-8")
                request = urllib.request.Request(
                    handle.server.origin + "/api/action",
                    data=body,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "X-V5-Review-Token": handle.session.token,
                    },
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    state = json.loads(response.read().decode("utf-8"))
                self.assertEqual(state["nodes"][0]["status"], "user_confirmed")
                saved = json.loads(bundle.checkpoint.read_text(encoding="utf-8"))
                self.assertEqual(saved["nodes"][0]["review_status"], "user_confirmed")
            finally:
                handle.server.shutdown()
                thread.join(timeout=3)
                handle.close()

    def test_finish_flushes_complete_response_and_checkpoint_before_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = ReviewBundle(Path(temporary) / "bundle")
            handle = create_review_server(bundle.checkpoint)

            def serve_and_close() -> None:
                try:
                    handle.serve_forever()
                finally:
                    handle.close()

            thread = threading.Thread(target=serve_and_close, daemon=True)
            thread.start()
            request = urllib.request.Request(
                handle.server.origin + "/api/finish",
                data=b"{}",
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "X-V5-Review-Token": handle.session.token,
                },
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                raw = response.read()
                payload = json.loads(raw.decode("utf-8"))
                self.assertEqual(response.status, 200)
                self.assertEqual(int(response.headers["Content-Length"]), len(raw))

            saved = json.loads(bundle.checkpoint.read_text(encoding="utf-8"))
            self.assertTrue(saved["review"]["finished"])
            self.assertTrue(payload["finished"])
            self.assertTrue(handle.session.finished.is_set())
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive(), "server must stop only after the finish body is flushed")

    def test_invalid_api_token_cannot_mutate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = ReviewBundle(Path(temporary) / "bundle")
            original = bundle.checkpoint.read_bytes()
            handle = create_review_server(bundle.checkpoint)
            thread = threading.Thread(target=handle.serve_forever, daemon=True)
            thread.start()
            try:
                request = urllib.request.Request(
                    handle.server.origin + "/api/action",
                    data=b'{"action":"delete_node","node_id":"NODE_ICON"}',
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "X-V5-Review-Token": "wrong-token",
                    },
                )
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    urllib.request.urlopen(request, timeout=3)
                self.assertEqual(denied.exception.code, 403)
                self.assertEqual(bundle.checkpoint.read_bytes(), original)
            finally:
                handle.server.shutdown()
                thread.join(timeout=3)
                handle.close()


if __name__ == "__main__":
    unittest.main()
