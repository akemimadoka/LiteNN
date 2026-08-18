#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("manage_build_artifacts", ROOT / "scripts" / "manage_build_artifacts.py")
assert SPEC is not None and SPEC.loader is not None
manager = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manager
SPEC.loader.exec_module(manager)


class BuildArtifactManagerTest(unittest.TestCase):
    def make_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory(dir=ROOT)
        root = Path(temporary.name) / "build-scratch"
        root.mkdir()
        return temporary, root

    def test_scan_uses_direct_children_as_retention_units(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            (root / "run-a" / "nested").mkdir(parents=True)
            (root / "run-a" / "nested" / "weights.bin").write_bytes(b"1234")
            (root / "trace.json").write_bytes(b"12")
            entries = manager.scan_entries(root, manager.DEFAULT_PROTECTED_NAMES)
            self.assertEqual({entry.name for entry in entries}, {"run-a", "trace.json"})
            by_name = {entry.name: entry for entry in entries}
            self.assertEqual(by_name["run-a"].size, 4)
            self.assertEqual(by_name["run-a"].file_count, 1)

    def test_size_and_count_limits_delete_oldest_unprotected_entries(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            now_ns = 10_000_000_000
            entries = [
                manager.ArtifactEntry(".litenn-cache", root / ".litenn-cache", 8, 1, 1, "directory", True),
                manager.ArtifactEntry("old", root / "old", 7, 1, 2, "directory", False),
                manager.ArtifactEntry("middle", root / "middle", 6, 1, 3, "directory", False),
                manager.ArtifactEntry("new", root / "new", 5, 1, 4, "directory", False),
            ]
            plan = manager.plan_cleanup(entries, now_ns=now_ns, max_total_bytes=14, max_entries=2)
            self.assertEqual([item.entry.name for item in plan], ["old", "middle"])
            self.assertTrue(all(item.entry.name != ".litenn-cache" for item in plan))

    def test_age_policy_uses_newest_file_activity(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            day_ns = 86400 * 1_000_000_000
            entries = [
                manager.ArtifactEntry("old", root / "old", 1, 1, day_ns, "directory", False),
                manager.ArtifactEntry("new", root / "new", 1, 1, 9 * day_ns, "directory", False),
            ]
            plan = manager.plan_cleanup(entries, now_ns=10 * day_ns, older_than_days=5)
            self.assertEqual([item.entry.name for item in plan], ["old"])

    def test_validate_root_refuses_repository_and_cmake_tree(self) -> None:
        with self.assertRaisesRegex(ValueError, "repository root"):
            manager.validate_root(ROOT, ROOT)
        temporary, root = self.make_root()
        with temporary:
            (root / "CMakeCache.txt").write_text("fixture", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CMake build tree"):
                manager.validate_root(root, ROOT)
            self.assertEqual(manager.validate_root(root, ROOT, allow_cmake_tree=True), root.resolve())

    def test_delete_planned_removes_only_selected_direct_child(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            selected = root / "selected"
            selected.mkdir()
            (selected / "artifact.bin").write_bytes(b"payload")
            retained = root / "retained"
            retained.mkdir()
            entry = manager.scan_entries(root, set())[0]
            if entry.name != "selected":
                entry = next(value for value in manager.scan_entries(root, set()) if value.name == "selected")
            manager.delete_planned(root, [manager.CleanupItem(entry, ("test",))])
            self.assertFalse(selected.exists())
            self.assertTrue(retained.exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support is unavailable")
    def test_symlink_is_always_protected(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            target = root / "target"
            target.mkdir()
            link = root / "link"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest("creating symlinks requires additional privileges")
            entry = next(value for value in manager.scan_entries(root, set()) if value.name == "link")
            self.assertEqual(entry.kind, "symlink")
            self.assertTrue(entry.protected)


if __name__ == "__main__":
    unittest.main()
